"""Provide xarray I/O and redistribution across MPI ranks."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from mpi4py.MPI import Intracomm

import xarray as xr

from ..mpi.runtime import MPIRuntime

if TYPE_CHECKING:
    from .core import MPIXarray

from .cartesian import compute_layout
from .chunks import (
    compute_save_chunks,
    get_balanced_bounds,
    get_chunk_bounds,
    get_chunk_info,
    get_effective_chunk_size,
    prune_chunk_info,
)
from .meta import (
    choose_partition_dim,
    delayed_local,
    get_mpi_meta,
    localize_coord,
    log_partition_report,
    log_partition_report_cartesian,
    resolve_sizes,
    set_mpi_meta,
    set_save_chunks,
    should_log_partitions,
    strip_mpi_meta,
)
from .netcdf import append, to_netcdf_parallel, to_netcdf_serial

__all__ = ["append", "dataset_is_empty", "empty_dataset", "to_netcdf"]

_NO_DATA_ATTR = "_climtools_no_data"


def _open_dataset_1d(
    runtime,
    filename_or_obj: Any,
    partition_dim: Hashable | Literal["auto"],
    open_fn: Callable,
    chunks: Any,
    log_partitions: bool,
    **kwargs: Any,
) -> xr.Dataset:
    """Open and partition a Dataset along one dimension."""
    automatic = partition_dim == "auto"

    # Build the metadata plan on rank 0.
    plan: dict[str, Any] | None = None
    error: BaseException | None = None
    if runtime.is_root():
        try:
            with open_fn(filename_or_obj, chunks=None, **kwargs) as metadata:
                if automatic:
                    partition_dim = choose_partition_dim(
                        metadata.sizes,
                        runtime.comm.size,
                        rank=runtime.comm.rank,
                    )
                if partition_dim not in metadata.dims:
                    raise ValueError(
                        f"partition_dim {partition_dim!r} is not in "
                        + f"{list(metadata.dims)!r}."
                    )
                chunk_info = get_chunk_info(metadata, runtime.comm.size)
                global_size = int(metadata.sizes[partition_dim])
                longest_size = max(int(length) for length in metadata.sizes.values())

                if not automatic and global_size < longest_size:
                    longest_dims = [
                        str(dim)
                        for dim, length in metadata.sizes.items()
                        if int(length) == longest_size
                    ]
                    warnings.warn(
                        f"partition_dim {partition_dim!r} has length "
                        + f"{global_size}, but it should be a longest "
                        + "dataset dimension. Longest dimension(s) "
                        + f"{longest_dims!r} have length {longest_size}.",
                        UserWarning,
                        stacklevel=2,
                    )

                # Pack the plan into a dictionary for broadcasting
                plan = {
                    "partition_dim": partition_dim,
                    "chunk_info": chunk_info,
                    "global_size": global_size,
                }
        except BaseException as exc:
            error = exc

    # Synchronize rank-0 planning failures before broadcasting the plan.
    runtime.raise_if_error(error, "open_dataset planning")

    # Broadcast the plan.
    plan = runtime.broadcast(plan, root=0)

    partition_dim = plan["partition_dim"]
    chunk_info = plan["chunk_info"]
    global_size = plan["global_size"]

    # Compute this rank's bounds.
    partition_chunk = chunk_info[str(partition_dim)]
    start, stop = get_chunk_bounds(
        global_size,
        partition_chunk,
        runtime.comm.rank,
        runtime.comm.size,
    )

    # Synchronize before opening the dataset.
    runtime.comm.Barrier()

    # Open this rank's lazy slice.
    data: xr.Dataset = open_fn(filename_or_obj, chunks=chunks, **kwargs)
    data = data.isel({partition_dim: slice(start, stop)})

    set_mpi_meta(
        data,
        dim=partition_dim,
        global_size=global_size,
        start=start,
        stop=stop,
        chunk_info=chunk_info,
    )
    if should_log_partitions(runtime, log_partitions):
        log_partition_report(
            runtime,
            data,
            partition_dim,
            origin="open_dataset",
            global_size=global_size,
            start=start,
            stop=stop,
            automatic=automatic,
        )
    return data


def _open_dataset_cartesian(
    runtime,
    filename_or_obj: Any,
    dims: tuple[Hashable, ...],
    open_fn: Callable,
    chunks: Any,
    log_partitions: bool,
    **kwargs: Any,
) -> xr.Dataset:
    """Open a Dataset lazily on an MPI Cartesian process grid.

    The multi-dimensional counterpart of :func:`_open_dataset_1d`, mirroring
    :meth:`partition`'s own
    multi-dimensional path (:meth:`_partition_pieces_nd`) but for a
    lazy on-disk open rather than an in-memory root-owned object:
    every rank opens ``filename_or_obj`` itself and computes its own
    bounds from :func:`~.cartesian.compute_layout` and
    :func:`~.chunks.get_balanced_bounds` -- the same deterministic,
    rank-invariant computation :meth:`_partition_pieces_nd` performs
    once per rank on root's behalf -- so no root-side slicing or
    point-to-point transfer of the data itself is needed here either;
    only the small metadata plan (each requested dimension's global
    length) needs the same single rank-0-computed, broadcast round
    trip the one-dimensional path above already does.

    Parameters
    ----------
    filename_or_obj : str, path-like, file-like, or list of these
        As in :func:`mpi_open_dataset`.
    dims : tuple of Hashable
        Two or more partition dimensions, already validated non-empty
        by :meth:`_as_partition_dims`.
    open_fn : callable
        ``xarray.open_dataset`` or ``xarray.open_mfdataset``, chosen
        by :func:`mpi_open_dataset` from ``filename_or_obj``'s shape.
    chunks : Any
        As in :func:`mpi_open_dataset`.
    log_partitions : bool
        As in :func:`mpi_open_dataset`.
    **kwargs : Any
        Forwarded to ``open_fn``.

    Returns
    -------
    xarray.Dataset
        Lazy rank-local Dataset carrying Cartesian ``mpi_meta``.

    Raises
    ------
    ValueError
        If any of ``dims`` is not a dimension of the opened dataset.
    """
    comm = runtime.comm

    plan: dict[str, Any] | None = None
    error: BaseException | None = None
    if runtime.is_root():
        try:
            with open_fn(filename_or_obj, chunks=None, **kwargs) as metadata:
                for d in dims:
                    if d not in metadata.dims:
                        raise ValueError(
                            f"partition_dim {d!r} is not in "
                            + f"{list(metadata.dims)!r}."
                        )
                plan = {"extents": tuple(int(metadata.sizes[d]) for d in dims)}
        except BaseException as exc:
            error = exc

    # Synchronize rank-0 planning failures before broadcasting the plan.
    runtime.raise_if_error(error, "open_dataset planning")

    # Broadcast the plan (just the per-dimension global lengths).
    plan = runtime.broadcast(plan, root=0)
    extents = plan["extents"]

    # Every rank derives its own Cartesian coordinates and per-axis
    # bounds from `extents` and `comm.size` alone -- identical on
    # every rank, no further communication needed to agree on it.
    grid_shape = compute_layout(extents, comm.size)
    coords = tuple(int(c) for c in np.unravel_index(comm.rank, grid_shape))
    bounds = {
        d: get_balanced_bounds(extents[axis], coords[axis], grid_shape[axis])
        for axis, d in enumerate(dims)
    }

    # Synchronize before opening the dataset (mirrors the
    # single-dimension path's own barrier here).
    comm.Barrier()

    # Open this rank's lazy slice.
    data: xr.Dataset = open_fn(filename_or_obj, chunks=chunks, **kwargs)
    data = data.isel({d: slice(*bounds[d]) for d in dims})

    chunk_info = {
        str(other_dim): get_effective_chunk_size(int(other_length), None, comm.size)
        for other_dim, other_length in data.sizes.items()
    }
    set_mpi_meta(
        data,
        dim=dims,
        global_size=dict(zip(dims, extents, strict=True)),
        start={d: bounds[d][0] for d in dims},
        stop={d: bounds[d][1] for d in dims},
        chunk_info=chunk_info,
        cart={
            "grid_shape": grid_shape,
            "coords": coords,
            "periods": (False,) * len(dims),
        },
    )
    if should_log_partitions(runtime, log_partitions):
        log_partition_report_cartesian(
            runtime,
            data,
            dims,
            origin="open_dataset",
            global_sizes=dict(zip(dims, extents, strict=True)),
            starts={d: bounds[d][0] for d in dims},
            stops={d: bounds[d][1] for d in dims},
            grid_shape=grid_shape,
            coords=coords,
        )
    return data


# mpi4py point-to-point tag for partition(); arbitrary but fixed so a
# stray message from unrelated code can never be mistaken for a piece
# this call is expecting.
_DISTRIBUTE_TAG = 0x6469_7374  # b"dist" as an int, easy to spot in a trace


def partition(
    runtime,
    value: xr.Dataset | xr.DataArray | None,
    dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    *,
    root: int = 0,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> xr.Dataset | xr.DataArray:
    """partition a root-owned xarray object across MPI ranks.

    The root slices the object along ``dim`` and sends each rank only its local
    piece. Use :meth:`repartition` when the full object already exists on every
    rank.

    Parameters
    ----------
    value : xarray.Dataset, xarray.DataArray, or None
        Complete object on ``root``; non-root ranks must pass None.
    dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Partition dimension(s). ``"auto"`` selects the single largest
        dimension (unchanged one-dimensional default). A single
        dimension, given directly (``"lat"``) or as a length-one
        sequence (``("lat",)``), takes the same one-dimensional path
        either way. A sequence of two or more dimensions (e.g.
        ``("lat", "lon")``) lays ranks out on an MPI Cartesian process
        grid and partitions every listed dimension simultaneously --
        see :mod:`.cartesian`. ``"auto"`` does not extend to choosing
        more than one dimension automatically; request multiple
        dimensions explicitly.
    root : int, optional
        Rank that owns ``value``. Default is 0.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints. Only consulted for the
        one-dimensional path; a multi-dimensional partition always
        uses :func:`~.chunks.get_balanced_bounds` per axis (see
        :func:`~.cartesian.compute_layout`), since native on-disk
        chunk alignment is inherently a single-axis concept here.
    log_partitions : bool, optional
        Log the resulting rank layout. Default is False.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Rank-local slice carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If ownership, metadata, or ``dim`` is invalid.

    Notes
    -----
    Dask-backed inputs remain lazy: the root sends sliced task graphs rather
    than materializing the full array."""
    comm = runtime.comm
    is_root = runtime.is_root(root)
    requested_dims = _as_partition_dims(dim)
    multi_dim = isinstance(requested_dims, tuple) and len(requested_dims) > 1

    # Prepare every slice before communication so a root-side failure is
    # synchronized before any rank can block in send/receive.
    error: BaseException | None = None
    pieces: list[Any] | None = None
    replicated_value: xr.Dataset | xr.DataArray | None = None
    try:
        if is_root:
            if value is None:
                raise ValueError(f"Rank {root} (root) must provide a value, not None.")
            if get_mpi_meta(value) is not None:
                raise ValueError(
                    "Cannot partition an already distributed object. "
                    + "Reduce or gather its distributed dimension first."
                )
            stripped = strip_mpi_meta(value)

            if not stripped.dims:
                # Nothing to partition: send the (necessarily small)
                # whole object to every rank as replicated data,
                # mirroring repartition's handling of the same case.
                replicated_value = stripped
            elif multi_dim:
                pieces = _partition_pieces_nd(
                    stripped,
                    cast("tuple[Hashable, ...]", requested_dims),
                    comm.size,
                )
            else:
                resolved_dim = (
                    requested_dims[0]
                    if isinstance(requested_dims, tuple)
                    else requested_dims
                )
                automatic = resolved_dim == "auto"
                if automatic:
                    resolved_dim = choose_partition_dim(
                        stripped.sizes, comm.size, rank=comm.rank
                    )
                if resolved_dim not in stripped.dims:
                    raise ValueError(
                        f"Distribution dimension {resolved_dim!r} does not exist."
                    )
                pieces = _partition_pieces_1d(
                    stripped, resolved_dim, comm.size, chunk_info
                )
        elif value is not None:
            raise ValueError(
                f"Only rank {root} (root) may provide a value; "
                + f"got one on rank {comm.rank}."
            )
    except BaseException as exc:
        error = exc
    runtime.raise_if_error(error, "partition")

    # Broadcast which transfer path root prepared.
    dimensionless = runtime.broadcast(
        replicated_value is not None if is_root else None, root=root
    )

    # Transfer the validated pieces.
    if dimensionless:
        # Nothing to partition: same small object broadcast to every
        # rank, no per-rank slicing or point-to-point send needed.
        output = runtime.broadcast(replicated_value if is_root else None, root=root)
        return cast("xr.Dataset | xr.DataArray", output)

    if is_root:
        assert pieces is not None
        for rank, piece in enumerate(pieces):
            if rank == root:
                output = piece
            else:
                runtime.send(piece, dest=rank, tag=_DISTRIBUTE_TAG)
    else:
        output = runtime.receive(source=root, tag=_DISTRIBUTE_TAG)

    if should_log_partitions(runtime, log_partitions):
        meta = get_mpi_meta(output)
        if meta is not None and "cart" in meta:
            log_partition_report_cartesian(
                runtime,
                output,
                meta["dims"],
                origin="partition",
                global_sizes=meta["global_sizes"],
                starts=meta["starts"],
                stops=meta["stops"],
                grid_shape=meta["cart"]["grid_shape"],
                coords=meta["cart"]["coords"],
            )
        elif meta is not None:
            log_partition_report(
                runtime,
                output,
                meta["dim"],
                origin="partition",
                global_size=meta["global_size"],
                start=meta["start"],
                stop=meta["stop"],
                automatic=(dim == "auto"),
            )
    return output


def _as_partition_dims(
    dim: Hashable | Sequence[Hashable] | Literal["auto"],
) -> Literal["auto"] | tuple[Hashable, ...]:
    """Normalize ``partition()``'s ``dim`` argument.

    Returns ``"auto"`` unchanged, or a non-empty tuple of dimension names
    for anything else -- a bare dimension name becomes a length-one
    tuple, so a caller passing ``dim="lat"`` and one passing
    ``dim=("lat",)`` are indistinguishable from here on and take the
    identical one-dimensional code path.
    """
    if dim == "auto":
        return "auto"
    if isinstance(dim, (list, tuple)):
        dims = tuple(dim)
        if not dims:
            raise ValueError("partition_dim sequence must not be empty.")
        return dims
    return (dim,)


def _partition_pieces_1d(
    stripped: xr.Dataset | xr.DataArray,
    resolved_dim: Hashable,
    comm_size: int,
    chunk_info: Mapping[str, int] | None,
) -> list[Any]:
    """Slice ``stripped`` into one piece per rank along one dimension.

    Unchanged in every respect from the sole implementation this
    method was extracted from, other than the extraction itself: the
    one-dimensional path a caller passing a single ``dim`` (directly
    or as a length-one sequence) takes is exactly this, with no added
    indirection or cost.
    """
    length = int(stripped.sizes[resolved_dim])
    info = dict(chunk_info or {})
    chunk_size = int(
        info.get(
            str(resolved_dim),
            get_effective_chunk_size(length, None, comm_size),
        )
    )
    chunk_size = get_effective_chunk_size(length, chunk_size, comm_size)
    info[str(resolved_dim)] = chunk_size

    pieces = []
    for rank in range(comm_size):
        start, stop = get_chunk_bounds(length, chunk_size, rank, comm_size)
        piece = stripped.isel({resolved_dim: slice(start, stop)})
        # Break shallow-copy attribute sharing before adding rank metadata.
        piece.attrs = dict(piece.attrs)
        if isinstance(piece, xr.Dataset):
            for variable in piece.variables.values():
                variable.attrs = dict(variable.attrs)
        piece_info = prune_chunk_info(info, piece)
        for other_dim, other_length in piece.sizes.items():
            piece_info.setdefault(
                str(other_dim),
                get_effective_chunk_size(int(other_length), None, comm_size),
            )
        set_mpi_meta(
            piece,
            dim=resolved_dim,
            global_size=length,
            start=start,
            stop=stop,
            chunk_info=piece_info,
        )
        pieces.append(piece)
    return pieces


def _partition_pieces_nd(
    stripped: xr.Dataset | xr.DataArray,
    dims: tuple[Hashable, ...],
    comm_size: int,
) -> list[Any]:
    """Slice ``stripped`` into one piece per rank on a Cartesian grid.

    Every rank's process-grid coordinates and per-axis bounds are
    computed here purely from ``comm_size`` and each dimension's
    global length -- the same deterministic, rank-invariant
    computation :func:`~.cartesian.build_cartesian_topology` performs
    for a rank's own coordinates, applied here to every rank at once
    so root can slice and address every piece without needing a live
    ``Create_cart`` communicator (a collective every rank would have
    to enter together) just to compute bounds. Each receiving rank
    builds and caches its own live Cartesian topology lazily, the
    first time it actually needs one (a collective like a reduction
    or halo exchange spanning more than one partition axis) --
    see :func:`~.cartesian.get_cartesian_topology`.
    """
    for d in dims:
        if d not in stripped.dims:
            raise ValueError(f"Distribution dimension {d!r} does not exist.")
    extents = tuple(int(stripped.sizes[d]) for d in dims)
    grid_shape = compute_layout(extents, comm_size)

    pieces = []
    for rank in range(comm_size):
        coords = tuple(int(c) for c in np.unravel_index(rank, grid_shape))
        bounds = {
            d: get_balanced_bounds(extents[axis], coords[axis], grid_shape[axis])
            for axis, d in enumerate(dims)
        }
        piece = stripped.isel({d: slice(*bounds[d]) for d in dims})
        piece.attrs = dict(piece.attrs)
        if isinstance(piece, xr.Dataset):
            for variable in piece.variables.values():
                variable.attrs = dict(variable.attrs)
        piece_info = {
            str(other_dim): get_effective_chunk_size(int(other_length), None, comm_size)
            for other_dim, other_length in piece.sizes.items()
        }
        set_mpi_meta(
            piece,
            dim=dims,
            global_size=dict(zip(dims, extents, strict=True)),
            start={d: bounds[d][0] for d in dims},
            stop={d: bounds[d][1] for d in dims},
            chunk_info=piece_info,
            cart={
                "grid_shape": grid_shape,
                "coords": coords,
                "periods": (False,) * len(dims),
            },
        )
        pieces.append(piece)
    return pieces


def _normalize_create_dim(
    dim: Hashable | int | Sequence[Hashable], dims: Sequence[Hashable]
) -> tuple[Hashable, ...]:
    """Normalize ``create_dataarray``/``create_dataset``'s ``dim`` to a tuple.

    A bare dimension name or axis ``int`` becomes a length-one tuple --
    the existing single-dimension behavior, unchanged in every respect
    including accepting an axis index, which a multi-dimension request
    cannot (there is no unambiguous per-axis ordering to infer an index
    against once more than one dimension is named). A sequence of two or
    more dimension names requests a Cartesian-topology partition,
    mirroring :func:`partition`'s own ``dim`` argument and using the same
    :func:`~.cartesian.compute_layout` process-grid factorization.
    """
    if isinstance(dim, (list, tuple)):
        if not dim:
            raise ValueError("dim sequence must not be empty.")
        for d in dim:
            if d not in dims:
                raise ValueError(f"dim {d!r} is not in dims {tuple(dims)!r}.")
        if len(set(dim)) != len(dim):
            raise ValueError(f"dim entries must be unique; got {tuple(dim)!r}.")
        return tuple(dim)
    axis_or_name = dims.index(dim) if not isinstance(dim, Integral) else int(dim)
    if not 0 <= axis_or_name < len(dims):
        raise ValueError(f"dim {dim!r} is not in dims {tuple(dims)!r}.")
    return (dims[axis_or_name],)


def create_dataarray(
    runtime,
    fill: Callable[..., Any],
    dims: Sequence[Hashable],
    *,
    shape: Sequence[int] | Mapping[Hashable, int] | None = None,
    dim: Hashable | int | Sequence[Hashable] = 0,
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    name: Hashable | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = False,
) -> xr.DataArray:
    """Create a distributed DataArray from a rank-local fill function.

    Parameters
    ----------
    fill : callable
        Function called as ``fill(start, stop)`` for this rank's bounds
        when ``dim`` names a single dimension. When ``dim`` names two or
        more dimensions, called instead as
        ``fill(start_0, stop_0, start_1, stop_1, ...)`` -- one
        ``(start, stop)`` pair per partitioned dimension, in ``dim``'s
        own order, laid out on an MPI Cartesian process grid exactly like
        :func:`partition`'s own multi-dimensional support.
    dims : sequence of Hashable
        Dimension names.
    shape : sequence of int, mapping, or None, optional
        Global dimension sizes. Missing sizes may be inferred from ``coords``.
    dim : Hashable, int, or sequence of Hashable, optional
        Dimension or axis to partition. Default is 0. A sequence of two
        or more dimension names requests a Cartesian-topology partition;
        an axis ``int`` is only valid for a single dimension.
    dtype : Any, optional
        Data type returned by ``fill``. Default is ``numpy.float64``.
    coords : mapping, optional
        Coordinates passed to :class:`xarray.DataArray`.
    name : Hashable, optional
        DataArray name.
    attrs : mapping, optional
        DataArray attributes.
    log_partitions : bool, optional
        Log the resulting rank layout. Default is False.

    Returns
    -------
    xarray.DataArray
        Lazy rank-local DataArray carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If ``dim`` is invalid or global sizes cannot be resolved."""
    partition_dims = _normalize_create_dim(dim, dims)

    if shape is None or isinstance(shape, Mapping):
        explicit_sizes = dict(shape) if shape else None
    else:
        if len(shape) != len(dims):
            raise ValueError(
                f"shape has {len(shape)} entries but dims has {len(dims)}."
            )
        explicit_sizes = dict(zip(dims, shape, strict=True))
    resolved_sizes = resolve_sizes(dims, explicit_sizes, coords)

    comm = runtime.comm
    extents = tuple(int(resolved_sizes[d]) for d in partition_dims)

    cart: dict[str, Any] | None = None
    if len(partition_dims) > 1:
        grid_shape = compute_layout(extents, comm.size)
        cart_coords = tuple(int(c) for c in np.unravel_index(comm.rank, grid_shape))
        bounds = {
            d: get_balanced_bounds(extents[axis], cart_coords[axis], grid_shape[axis])
            for axis, d in enumerate(partition_dims)
        }
        cart = {
            "grid_shape": grid_shape,
            "coords": cart_coords,
            "periods": (False,) * len(partition_dims),
        }
    else:
        start, stop = get_balanced_bounds(extents[0], comm.rank, comm.size)
        bounds = {partition_dims[0]: (start, stop)}

    local_shape = tuple(
        (bounds[name][1] - bounds[name][0]) if name in bounds
        else int(resolved_sizes[name])
        for name in dims
    )

    fill_args = tuple(v for d in partition_dims for v in bounds[d])
    local_data = delayed_local(fill, fill_args, local_shape, dtype)

    local_coords = dict(coords) if coords else {}
    for d in partition_dims:
        if d in local_coords:
            d_start, d_stop = bounds[d]
            local_coords[d] = localize_coord(
                local_coords[d], int(resolved_sizes[d]), d_start, d_stop
            )

    da = xr.DataArray(
        local_data, dims=tuple(dims), coords=local_coords, name=name, attrs=attrs
    )
    chunk_info = {str(d): bounds[d][1] - bounds[d][0] for d in partition_dims}
    set_mpi_meta(
        da,
        dim=partition_dims if len(partition_dims) > 1 else partition_dims[0],
        global_size={d: int(resolved_sizes[d]) for d in partition_dims},
        start={d: bounds[d][0] for d in partition_dims},
        stop={d: bounds[d][1] for d in partition_dims},
        chunk_info=chunk_info,
        cart=cart,
    )
    if should_log_partitions(runtime, log_partitions):
        if len(partition_dims) > 1:
            log_partition_report_cartesian(
                runtime,
                da,
                partition_dims,
                origin="create_dataarray",
                global_sizes={d: int(resolved_sizes[d]) for d in partition_dims},
                starts={d: bounds[d][0] for d in partition_dims},
                stops={d: bounds[d][1] for d in partition_dims},
                grid_shape=cart["grid_shape"],
                coords=cart["coords"],
            )
        else:
            d0 = partition_dims[0]
            log_partition_report(
                runtime,
                da,
                d0,
                origin="create_dataarray",
                global_size=int(resolved_sizes[d0]),
                start=bounds[d0][0],
                stop=bounds[d0][1],
            )
    return da


def create_dataset(
    runtime,
    data_vars: Mapping[
        Hashable,
        xr.DataArray | tuple[Sequence[Hashable], Callable[[int, int], Any]],
    ],
    sizes: Mapping[Hashable, int] | None = None,
    *,
    dim: Hashable | Sequence[Hashable],
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = True,
) -> xr.Dataset:
    """Create a distributed Dataset from rank-local variables.

    Parameters
    ----------
    data_vars : mapping
        Variables as DataArrays or ``(dims, fill)`` pairs. A fill function
        is called with one ``(start, stop)`` pair per partition dimension
        present in that variable's own ``dims``, in ``dim``'s order (a
        single pair for the common single-dimension case); a variable
        whose ``dims`` contain none of the partition dimensions is
        unpartitioned -- identical on every rank -- and its fill takes no
        arguments.
    sizes : mapping, optional
        Global dimension sizes. Missing sizes may be inferred from ``coords``.
    dim : Hashable or sequence of Hashable
        Dimension(s) to partition. A sequence of two or more dimension
        names lays ranks out on an MPI Cartesian process grid and
        partitions every one simultaneously, exactly like
        :func:`partition`'s own multi-dimensional support.
    dtype : Any or mapping, optional
        Default or per-variable fill dtype. Default is ``numpy.float64``.
    coords : mapping, optional
        Coordinates passed to :class:`xarray.Dataset`.
    attrs : mapping, optional
        Dataset attributes.
    log_partitions : bool, optional
        Log the resulting rank layout. Default is True.

    Returns
    -------
    xarray.Dataset
        Lazy rank-local Dataset carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If sizes cannot be resolved or a partitioned DataArray has the wrong
        local length."""
    if isinstance(dim, (list, tuple)):
        if not dim:
            raise ValueError("dim sequence must not be empty.")
        if len(set(dim)) != len(dim):
            raise ValueError(f"dim entries must be unique; got {tuple(dim)!r}.")
        partition_dims = tuple(dim)
    else:
        partition_dims = (dim,)

    required_dims: set[Hashable] = set(partition_dims)
    for spec in data_vars.values():
        if not isinstance(spec, xr.DataArray):
            var_dims, _ = spec
            required_dims.update(var_dims)
    resolved_sizes = resolve_sizes(required_dims, sizes, coords)

    comm = runtime.comm
    extents = tuple(int(resolved_sizes[d]) for d in partition_dims)

    cart: dict[str, Any] | None = None
    if len(partition_dims) > 1:
        grid_shape = compute_layout(extents, comm.size)
        cart_coords = tuple(int(c) for c in np.unravel_index(comm.rank, grid_shape))
        bounds = {
            d: get_balanced_bounds(extents[axis], cart_coords[axis], grid_shape[axis])
            for axis, d in enumerate(partition_dims)
        }
        cart = {
            "grid_shape": grid_shape,
            "coords": cart_coords,
            "periods": (False,) * len(partition_dims),
        }
    else:
        start, stop = get_balanced_bounds(extents[0], comm.rank, comm.size)
        bounds = {partition_dims[0]: (start, stop)}

    dtype_map = dtype if isinstance(dtype, Mapping) else None

    built_vars: dict[Hashable, Any] = {}
    for var_name, spec in data_vars.items():
        if isinstance(spec, xr.DataArray):
            for d in partition_dims:
                if d in spec.dims:
                    d_start, d_stop = bounds[d]
                    expected_len = d_stop - d_start
                    if int(spec.sizes[d]) != expected_len:
                        raise ValueError(
                            f"data_vars[{var_name!r}] is a DataArray of length "
                            + f"{spec.sizes[d]} along {d!r}, but this rank "
                            + f"owns [{d_start}:{d_stop}) ({expected_len} "
                            + "elements). Pass a DataArray already sized to "
                            + "this rank's own bounds (e.g. from "
                            + "create_dataarray), not the full global array."
                        )
            built_vars[var_name] = spec
            continue

        var_dims, var_fill = spec
        var_dtype = (
            dtype_map.get(var_name, np.float64) if dtype_map is not None else dtype
        )
        local_dims_here = [d for d in partition_dims if d in var_dims]
        local_shape = tuple(
            (bounds[name][1] - bounds[name][0])
            if name in local_dims_here
            else int(resolved_sizes[name])
            for name in var_dims
        )
        if local_dims_here:
            fill_args = tuple(v for d in local_dims_here for v in bounds[d])
            local_data = delayed_local(var_fill, fill_args, local_shape, var_dtype)
        elif callable(var_fill):
            # Not partitioned: identical on every rank, so there is no
            # (start, stop) to give -- fill() takes no arguments and
            # closes over whatever sizes it needs itself.
            local_data = delayed_local(var_fill, (), local_shape, var_dtype)
        else:
            local_data = var_fill
        built_vars[var_name] = (tuple(var_dims), local_data)

    local_coords = dict(coords) if coords else {}
    for d in partition_dims:
        if d in local_coords:
            d_start, d_stop = bounds[d]
            local_coords[d] = localize_coord(
                local_coords[d], int(resolved_sizes[d]), d_start, d_stop
            )

    ds = xr.Dataset(built_vars, coords=local_coords, attrs=attrs)
    chunk_info = {str(d): bounds[d][1] - bounds[d][0] for d in partition_dims}
    set_mpi_meta(
        ds,
        dim=partition_dims if len(partition_dims) > 1 else partition_dims[0],
        global_size={d: int(resolved_sizes[d]) for d in partition_dims},
        start={d: bounds[d][0] for d in partition_dims},
        stop={d: bounds[d][1] for d in partition_dims},
        chunk_info=chunk_info,
        cart=cart,
    )
    if should_log_partitions(runtime, log_partitions):
        if len(partition_dims) > 1:
            log_partition_report_cartesian(
                runtime,
                ds,
                partition_dims,
                origin="create_dataset",
                global_sizes={d: int(resolved_sizes[d]) for d in partition_dims},
                starts={d: bounds[d][0] for d in partition_dims},
                stops={d: bounds[d][1] for d in partition_dims},
                grid_shape=cart["grid_shape"],
                coords=cart["coords"],
            )
        else:
            d0 = partition_dims[0]
            log_partition_report(
                runtime,
                ds,
                d0,
                origin="create_dataset",
                global_size=int(resolved_sizes[d0]),
                start=bounds[d0][0],
                stop=bounds[d0][1],
            )
    return ds


def repartition(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable | Literal["auto"] = "auto",
    *,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Partition a replicated xarray object across MPI ranks.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Complete object present on every rank.
    dim : Hashable or {"auto"}, optional
        New partition dimension. ``"auto"`` selects the largest dimension.
        Default is ``"auto"``.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints.
    log_partitions : bool, optional
        Log the resulting rank layout. Default is False.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Rank-local slice carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If ``value`` is already distributed or ``dim`` is invalid."""
    if get_mpi_meta(value) is not None:
        raise ValueError(
            "Cannot repartition an already distributed object. "
            + "Reduce or gather its distributed dimension first."
        )

    automatic = dim == "auto"
    if automatic:
        if not value.dims:
            return strip_mpi_meta(value)
        dim = choose_partition_dim(
            value.sizes, runtime.comm.size, rank=runtime.comm.rank
        )

    if dim not in value.dims:
        raise ValueError(f"Repartition dimension {dim!r} does not exist.")

    info = dict(chunk_info or {})
    length = int(value.sizes[dim])
    chunk_size = int(
        info.get(
            str(dim),
            get_effective_chunk_size(length, None, runtime.comm.size),
        )
    )
    chunk_size = get_effective_chunk_size(length, chunk_size, runtime.comm.size)
    info[str(dim)] = chunk_size

    start, stop = get_chunk_bounds(
        length, chunk_size, runtime.comm.rank, runtime.comm.size
    )
    output = strip_mpi_meta(value).isel({dim: slice(start, stop)})
    info = prune_chunk_info(info, output)
    for other_dim, other_length in output.sizes.items():
        info.setdefault(
            str(other_dim),
            get_effective_chunk_size(int(other_length), None, runtime.comm.size),
        )

    set_mpi_meta(
        output, dim=dim, global_size=length, start=start, stop=stop, chunk_info=info
    )
    if should_log_partitions(runtime, log_partitions):
        log_partition_report(
            runtime,
            output,
            dim,
            origin="repartition",
            global_size=length,
            start=start,
            stop=stop,
            automatic=automatic,
        )
    return output


def attach_save_chunks(
    runtime, value: xr.Dataset | xr.DataArray
) -> xr.Dataset | xr.DataArray:
    """Attach write-time chunk metadata to a distributed object.

    The save-chunk plan is computed on rank 0 from distribution metadata and
    broadcast to all ranks. No data are materialized.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Distributed rank-local object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        ``value`` with ``mpi_meta["save_chunks"]`` attached. Undistributed input
        is returned unchanged.

    Raises
    ------
    ValueError
        If required partition chunk metadata are missing."""
    meta = get_mpi_meta(value)
    if meta is None:
        return value
    if len(meta["dims"]) > 1:
        raise NotImplementedError(
            "attach_save_chunks() (and therefore compute_save_chunks()) "
            + "only supports a single partition dimension so far "
            + f"(dims={meta['dims']!r} under this multi-dimensional "
            + "partition); NetCDF save-chunk planning for more than one "
            + "partition axis is not yet implemented -- see "
            + "write_distributed(), which does not need it and already "
            + "supports any number of partition dimensions."
        )

    save_chunks: dict[str, tuple[int, ...]] | None = None
    error: BaseException | None = None
    if runtime.is_root():
        try:
            save_chunks = compute_save_chunks(value, meta, runtime.comm.size)
        except BaseException as exc:
            error = exc
    runtime.raise_if_error(error, "attach_save_chunks planning")

    save_chunks = runtime.broadcast(save_chunks, root=0)
    set_save_chunks(value, cast("dict[str, tuple[int, ...]]", save_chunks))
    return value


def mpi_open_dataset(
    filename: Path | str | PathLike,
    mpi_runtime: MPIRuntime | Intracomm,
    *,
    partition_dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    chunks: Any = None,
    log_partitions: bool = True,
    **kwargs: Any,
) -> MPIXarray:
    """Open a Dataset lazily and partition it across MPI ranks.

    Parameters
    ----------
    filename_or_obj : str, path-like, file-like, or list of these
        Input accepted by ``xarray.open_dataset``/``xarray.open_mfdataset``.
        A wildcard string or a list/tuple triggers multi-file loading.
    mpi_runtime : MPIRuntime or mpi4py.MPI.Intracomm
        Runtime whose communicator the result is bound to.
    partition_dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Dimension(s) to partition. "auto" selects the longest dimension
        (single-dimension only). A sequence of two or more dimensions
        (e.g. ``("lat", "lon")``) lays ranks out on an MPI Cartesian
        process grid and distributes every listed dimension at once,
        exactly like ``mpi_partition_data``'s own multi-dimensional
        support -- see ``IO.open_dataset``.
    chunks : int, dict, "auto" or None, optional
        Passed unchanged to xarray.
    log_partitions : bool, optional
        Print one aligned table showing which global interval each rank received.
    **kwargs : Any
        Additional arguments passed unchanged to ``xarray.open_dataset``/
        ``xarray.open_mfdataset`` (e.g. ``engine``, ``decode_times``,
        ``concat_dim``, ``combine``, ``preprocess``, ``parallel``).

    Returns
    -------
    MPIXarray
        Lazy rank-local Dataset with ``.meta`` set.
    """

    if not isinstance(mpi_runtime, MPIRuntime):
        mpi_runtime = MPIRuntime(mpi_runtime)

    xr.set_options(keep_attrs=True)

    use_mfdataset = (isinstance(filename, str) and "*" in filename) or isinstance(
        filename, (list, tuple)
    )
    open_fn: Callable = xr.open_mfdataset if use_mfdataset else xr.open_dataset

    requested_dims = _as_partition_dims(partition_dim)
    if isinstance(requested_dims, tuple) and len(requested_dims) > 1:
        data = _open_dataset_cartesian(
            mpi_runtime,
            filename,
            requested_dims,
            open_fn,
            chunks,
            log_partitions,
            **kwargs,
        )
    else:
        resolved_dim = (
            requested_dims[0] if isinstance(requested_dims, tuple) else requested_dims
        )
        data = _open_dataset_1d(
            mpi_runtime,
            filename,
            resolved_dim,
            open_fn,
            chunks,
            log_partitions,
            **kwargs,
        )

    from .core import MPIXarray

    return MPIXarray(data, mpi_runtime)


def mpi_create_dataarray(
    runtime: MPIRuntime,
    fill: Callable[..., Any],
    dims: Sequence[Hashable],
    *,
    shape: Sequence[int] | Mapping[Hashable, int] | None = None,
    dim: Hashable | int | Sequence[Hashable] = 0,
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    name: Hashable | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = False,
) -> MPIXarray:
    """Create a distributed DataArray from a rank-local fill function.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime whose communicator the result is bound to.
    fill : callable
        Function called as ``fill(start, stop)`` for this rank's bounds
        when ``dim`` names a single dimension, or as
        ``fill(start_0, stop_0, start_1, stop_1, ...)`` -- one pair per
        partitioned dimension, in ``dim``'s order -- when ``dim`` names
        two or more.
    dims : sequence of Hashable
        Dimension names.
    shape : sequence of int, mapping, or None, optional
        Global dimension sizes. Missing sizes may be inferred from ``coords``.
    dim : Hashable, int, or sequence of Hashable, optional
        Dimension or axis to partition. A sequence of two or more
        dimension names lays ranks out on an MPI Cartesian process grid
        and partitions every one simultaneously, exactly like
        ``mpi_partition_data``'s own multi-dimensional support.
    dtype : Any, optional
        Data type returned by ``fill``.
    coords : mapping, optional
        Coordinates passed to ``xarray.DataArray``.
    name : Hashable, optional
        DataArray name.
    attrs : mapping, optional
        DataArray attributes.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    MPIXarray
        Lazy rank-local DataArray with ``.meta`` set.
    """
    from .core import MPIXarray

    data = create_dataarray(
        runtime,
        fill,
        dims,
        shape=shape,
        dim=dim,
        dtype=dtype,
        coords=coords,
        name=name,
        attrs=attrs,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, runtime)


def mpi_create_dataset(
    runtime: MPIRuntime,
    data_vars: Mapping[
        Hashable, xr.DataArray | tuple[Sequence[Hashable], Callable[..., Any]]
    ],
    sizes: Mapping[Hashable, int] | None = None,
    *,
    dim: Hashable | Sequence[Hashable],
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = True,
) -> MPIXarray:
    """Create a distributed Dataset from rank-local variables.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime whose communicator the result is bound to.
    data_vars : mapping
        Variables as DataArrays or ``(dims, fill)`` pairs. A fill function
        receives one ``(start, stop)`` pair per partition dimension present
        in that variable's own ``dims``, in ``dim``'s order; a variable
        naming none of the partition dimensions is unpartitioned and its
        fill takes no arguments.
    sizes : mapping, optional
        Global dimension sizes. Missing sizes may be inferred from ``coords``.
    dim : Hashable or sequence of Hashable
        Dimension(s) to partition. A sequence of two or more dimension
        names lays ranks out on an MPI Cartesian process grid and
        partitions every one simultaneously, exactly like
        ``mpi_partition_data``'s own multi-dimensional support.
    dtype : Any or mapping, optional
        Default or per-variable fill dtype.
    coords : mapping, optional
        Coordinates passed to ``xarray.Dataset``.
    attrs : mapping, optional
        Dataset attributes.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    MPIXarray
        Lazy rank-local Dataset with ``.meta`` set.
    """
    from .core import MPIXarray

    data = create_dataset(
        runtime,
        data_vars,
        sizes,
        dim=dim,
        dtype=dtype,
        coords=coords,
        attrs=attrs,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, runtime)


def mpi_partition_data(
    value: MPIXarray | xr.Dataset | xr.DataArray | None,
    runtime: MPIRuntime,
    dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    *,
    root: int = 0,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> MPIXarray:
    """Partition a root-owned xarray object across MPI ranks.

    Parameters
    ----------
    value : MPIXarray, xarray.Dataset, xarray.DataArray, or None
        Complete object on ``root``; non-root ranks must pass None.
    runtime : MPIRuntime
        Runtime whose communicator the result is bound to.
    dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Partition dimension(s). "auto" selects the largest dimension. A
        sequence of two or more dimensions lays ranks out on an MPI
        Cartesian process grid and partitions every one simultaneously;
        see :meth:`~.io.IO.partition`.
    root : int, optional
        Rank that owns ``value``.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints. Only consulted for a single partition
        dimension; see :meth:`~.io.IO.partition`.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    MPIXarray
        Rank-local slice with ``.meta`` set.
    """
    from .core import MPIXarray, unwrap

    data = partition(
        runtime,
        unwrap(value),
        dim,
        root=root,
        chunk_info=chunk_info,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, runtime)


def empty_dataset() -> xr.Dataset:
    """Return a placeholder Dataset for a non-root MPI rank.

    Returns
    -------
    xarray.Dataset
        Dataset marked as containing no rank-local data.
    """
    return xr.Dataset(attrs={_NO_DATA_ATTR: True})


def dataset_is_empty(data: xr.Dataset | xr.DataArray) -> bool:
    """Return whether an object is a non-root MPI placeholder.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to inspect.

    Returns
    -------
    bool
        True when ``data`` is an MPI placeholder Dataset.
    """
    return isinstance(data, xr.Dataset) and data.attrs.get(_NO_DATA_ATTR) is True


def to_netcdf(
    data: xr.Dataset | xr.DataArray,
    file: str | PathLike[str],
    mpi_runtime: MPIRuntime | Intracomm = None,
    unlimited_dim: str | Iterable[str] | None = None,
    partition_dim: str | None = None,
    *,
    parallel: bool = False,
    batch_size: int = 24,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
    chunks: Mapping[str, Iterable[int]] | None = None,
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
) -> None:
    """Write a Dataset or DataArray to NetCDF.

    Serial output is written incrementally along an unlimited dimension. In
    parallel mode, an object carrying ``mpi_meta`` is already distributed and
    every rank writes its existing local slab directly. Otherwise rank 0 owns
    the complete object and the parallel writer distributes partitioned data.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to write.
    file : str or os.PathLike
        Output path.
    mpi_runtime : MPIRuntime or Intracomm, optional
        MPI runtime or communicator.
    unlimited_dim : str or iterable of str, optional
        Dimension or dimensions made unlimited.
    partition_dim : str, optional
        MPI partition dimension. For an already distributed object this must
        agree with ``mpi_meta["dim"]``.
    parallel : bool, default: False
        Use MPI-parallel NetCDF-4 output.
    batch_size : int, default: 24
        Number of slices written per serial append.
    format : str, default: "NETCDF4"
        NetCDF format for serial output.
    shuffle : bool, default: True
        Apply the HDF5 shuffle filter.
    zlib : bool, default: True
        Apply zlib compression.
    complevel : int, default: 4
        Compression level from 0 through 9.
    show_progress : bool, default: True
        Display serial write progress.
    stdout : Any, optional
        Serial progress output stream.
    chunks : mapping of str to iterable of int, optional
        Explicit NetCDF variable chunk shapes.
    hints : str, optional
        Semicolon-separated MPI-IO hints in ``key=value`` form.
    nofill : bool, default: True
        Disable NetCDF pre-filling during parallel initialization.
    allow_serial : bool, default: False
        Permit the parallel writer with one MPI rank.

    Returns
    -------
    None
    """

    if not isinstance(data, (xr.Dataset, xr.DataArray)):
        raise TypeError("data must be an xarray.Dataset or xarray.DataArray")

    target_path = Path(file)

    if parallel:
        if not mpi_runtime:
            from ..mpi.runtime import mpi

            mpi_runtime = mpi

        mpi_meta = get_mpi_meta(data)
        distributed = mpi_meta is not None

        # Ranks must agree on the write path. If one rank saw valid mpi_meta
        # and another did not, the two paths post different collectives and
        # the writer would block instead of reporting the inconsistency.
        agreed = mpi_runtime.comm.allgather(distributed)
        if any(agreed) and not all(agreed):
            disagreeing = [
                rank for rank, state in enumerate(agreed) if state != agreed[0]
            ]
            raise mpi_runtime.MPIError(
                "MPI ranks disagree about whether the object is distributed; "
                + f"ranks {disagreeing} differ from rank 0. Parallel NetCDF "
                + "output requires the same distribution state on every rank."
            )

        if distributed:
            distributed_dim = str(mpi_meta["dim"])
            if partition_dim is not None and partition_dim != distributed_dim:
                raise ValueError(
                    f"partition_dim {partition_dim!r} does not match "
                    + f"distributed dimension {distributed_dim!r}."
                )
            partition_dim = distributed_dim
        elif mpi_runtime.comm.rank != 0:
            data = empty_dataset()

        to_netcdf_parallel(
            mpi_runtime,
            data,
            target_path,
            partition_dim=partition_dim,
            deflate=complevel if zlib else None,
            shuffle=shuffle,
            chunks=chunks,
            unlimited_dim=unlimited_dim if unlimited_dim is not None else (),
            hints=hints,
            nofill=nofill,
            allow_serial=allow_serial,
        )
        return

    to_netcdf_serial(
        data=data,
        file=target_path,
        unlimited_dim=unlimited_dim,
        batch_size=batch_size,
        format=format,
        shuffle=shuffle,
        zlib=zlib,
        complevel=complevel,
        show_progress=show_progress,
        stdout=stdout,
    )
