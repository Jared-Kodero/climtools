"""Provide xarray I/O and redistribution across MPI ranks."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr

from ..mpi.context import MPIContext
from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from .core import MPIXarray

from .chunks import (
    compute_save_chunks,
    get_chunk_bounds,
    get_chunk_info,
    get_effective_chunk_size,
    prune_chunk_info,
)
from .meta import (
    choose_partition_dim,
    delayed_local,
    localize_coord,
    mpp_get_meta,
    mpp_log_partition_report,
    mpp_should_log_partitions,
    mpp_update_meta,
    resolve_sizes,
    set_save_chunks,
    strip_mpi_meta,
)
from .mpp import mpp_define_domains
from .netcdf import mpp_to_netcdf_parallel, nc_append, to_netcdf_serial

__all__ = ["mpi_dataset_is_empty", "mpi_empty_dataset", "nc_append", "to_netcdf"]

_NO_DATA_ATTR = "_climtools_no_data"


def _open_dataset_1d(
    mpi_context: MPIContext,
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
    if mpi_context.is_root():
        try:
            with open_fn(filename_or_obj, chunks=None, **kwargs) as metadata:
                if automatic:
                    partition_dim = choose_partition_dim(
                        metadata.sizes,
                        mpi_context.comm.size,
                        rank=mpi_context.comm.rank,
                    )
                if partition_dim not in metadata.dims:
                    raise ValueError(
                        f"partition_dim {partition_dim!r} is not in "
                        + f"{list(metadata.dims)!r}."
                    )
                chunk_info = get_chunk_info(metadata, mpi_context.comm.size)
                global_size = int(metadata.sizes[partition_dim])

                # Pack the plan into a dictionary for broadcasting
                plan = {
                    "partition_dim": partition_dim,
                    "chunk_info": chunk_info,
                    "global_size": global_size,
                }
        except BaseException as exc:
            error = exc

    # Synchronize rank-0 planning failures before broadcasting the plan.
    mpi_context.raise_if_error(error, "open_dataset planning")

    # Broadcast the plan.
    plan = mpi_context.broadcast(plan, root=0)

    partition_dim = plan["partition_dim"]
    chunk_info = plan["chunk_info"]
    global_size = plan["global_size"]

    # Compute this rank's bounds.
    partition_chunk = chunk_info[str(partition_dim)]
    start, stop = get_chunk_bounds(
        global_size,
        partition_chunk,
        mpi_context.comm.rank,
        mpi_context.comm.size,
    )

    # Synchronize before opening the dataset.
    mpi_context.comm.Barrier()

    # Open this rank's lazy slice.
    data: xr.Dataset = open_fn(filename_or_obj, chunks=chunks, **kwargs)
    data = data.isel({partition_dim: slice(start, stop)})

    mpp_update_meta(
        data,
        dim=partition_dim,
        global_size=global_size,
        start=start,
        stop=stop,
        chunk_info=chunk_info,
    )
    if mpp_should_log_partitions(mpi_context, log_partitions):
        mpp_log_partition_report(
            mpi_context,
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
    mpi_context: MPIContext,
    filename_or_obj: Any,
    dims: tuple[Hashable, ...],
    open_fn: Callable,
    chunks: Any,
    log_partitions: bool,
    **kwargs: Any,
) -> xr.Dataset:
    """Open a Dataset lazily on an MPI Cartesian process grid."""
    comm = mpi_context.comm

    plan: dict[str, Any] | None = None
    error: BaseException | None = None
    if mpi_context.is_root():
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
    mpi_context.raise_if_error(error, "open_dataset planning")

    # Broadcast the plan (just the per-dimension global lengths).
    plan = mpi_context.broadcast(plan, root=0)
    extents = plan["extents"]

    # Every rank derives its own Cartesian coordinates and per-axis
    # bounds from `extents` and `comm.size` alone -- identical on
    # every rank, no further communication needed to agree on it.
    domain = mpp_define_domains(
        mpi_context, dict(zip(dims, extents, strict=True)), dims
    )
    bounds = {d: (domain.starts[d], domain.stops[d]) for d in dims}

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
    mpp_update_meta(
        data,
        dim=dims,
        global_size=dict(zip(dims, extents, strict=True)),
        start={d: bounds[d][0] for d in dims},
        stop={d: bounds[d][1] for d in dims},
        chunk_info=chunk_info,
        cart=domain.cart,
    )
    if mpp_should_log_partitions(mpi_context, log_partitions):
        mpp_log_partition_report(
            mpi_context,
            data,
            dims,
            origin="open_dataset",
            global_size=dict(zip(dims, extents, strict=True)),
            start={d: bounds[d][0] for d in dims},
            stop={d: bounds[d][1] for d in dims},
            grid_shape=domain.cart["grid_shape"],
            coords=domain.cart["coords"],
        )
    return data


# mpi4py point-to-point tag for mpp_partition(); arbitrary but fixed so a
# stray message from unrelated code can never be mistaken for a piece
# this call is expecting.
_DISTRIBUTE_TAG = 0x6469_7374  # b"dist" as an int, easy to spot in a trace


def mpp_partition(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray | None,
    dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    *,
    root: int = 0,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> xr.Dataset | xr.DataArray:
    """partition a root-owned xarray object across MPI ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset, xarray.DataArray, or None
        Complete object on ``root``; non-root ranks must pass None.
    dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Partition dimension(s).
    root : int, optional
        Rank that owns ``value``.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Rank-local slice carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If ownership, metadata, or ``dim`` is invalid.

    """
    comm = mpi_context.comm
    is_root = mpi_context.is_root(root)
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
            if mpp_get_meta(value) is not None:
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
                    mpi_context,
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
    mpi_context.raise_if_error(error, "partition")

    # Broadcast which transfer path root prepared.
    dimensionless = mpi_context.broadcast(
        replicated_value is not None if is_root else None, root=root
    )

    # Transfer the validated pieces.
    if dimensionless:
        # Nothing to partition: same small object broadcast to every
        # rank, no per-rank slicing or point-to-point send needed.
        output = mpi_context.broadcast(replicated_value if is_root else None, root=root)
        return cast("xr.Dataset | xr.DataArray", output)

    if is_root:
        assert pieces is not None
        output = pieces[root]
        # Post every outgoing piece before waiting on any of them. The
        # previous blocking loop completed one handshake before starting the
        # next, so the scatter cost grew with rank count instead of staying
        # flat -- see MPIContext.send_all.
        mpi_context.send_all(
            {rank: piece for rank, piece in enumerate(pieces) if rank != root},
            tag=_DISTRIBUTE_TAG,
        )
    else:
        output = mpi_context.receive(source=root, tag=_DISTRIBUTE_TAG)

    if mpp_should_log_partitions(mpi_context, log_partitions):
        meta = mpp_get_meta(output)
        if meta is not None and "cart" in meta:
            mpp_log_partition_report(
                mpi_context,
                output,
                meta["dims"],
                origin="partition",
                global_size=meta["global_sizes"],
                start=meta["starts"],
                stop=meta["stops"],
                grid_shape=meta["cart"]["grid_shape"],
                coords=meta["cart"]["coords"],
            )
        elif meta is not None:
            mpp_log_partition_report(
                mpi_context,
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
    """Normalize ``mpp_partition()``'s ``dim`` argument."""
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
    """Slice ``stripped`` into one piece per rank along one dimension."""
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
        mpp_update_meta(
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
    mpi_context: MPIContext,
    stripped: xr.Dataset | xr.DataArray,
    dims: tuple[Hashable, ...],
    comm_size: int,
) -> list[Any]:
    """Slice ``stripped`` into one piece per rank on a Cartesian grid."""
    for d in dims:
        if d not in stripped.dims:
            raise ValueError(f"Distribution dimension {d!r} does not exist.")
    extents = tuple(int(stripped.sizes[d]) for d in dims)
    sizes = dict(zip(dims, extents, strict=True))

    pieces = []
    for rank in range(comm_size):
        domain = mpp_define_domains(mpi_context, sizes, dims, rank=rank)
        bounds = {d: (domain.starts[d], domain.stops[d]) for d in dims}
        piece = stripped.isel({d: slice(*bounds[d]) for d in dims})
        piece.attrs = dict(piece.attrs)
        if isinstance(piece, xr.Dataset):
            for variable in piece.variables.values():
                variable.attrs = dict(variable.attrs)
        piece_info = {
            str(other_dim): get_effective_chunk_size(int(other_length), None, comm_size)
            for other_dim, other_length in piece.sizes.items()
        }
        mpp_update_meta(
            piece,
            dim=dims,
            global_size=sizes,
            start={d: bounds[d][0] for d in dims},
            stop={d: bounds[d][1] for d in dims},
            chunk_info=piece_info,
            cart=domain.cart,
        )
        pieces.append(piece)
    return pieces


def _normalize_create_dim(
    dim: Hashable | int | Sequence[Hashable], dims: Sequence[Hashable]
) -> tuple[Hashable, ...]:
    """Normalize ``create_dataarray``/``create_dataset``'s ``dim`` to a tuple."""
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


def mpp_create_dataarray(
    mpi_context: MPIContext,
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
    min_partition_size: int | Mapping[Hashable, int] | None = None,
) -> xr.DataArray:
    """Create a distributed DataArray from a rank-local fill function.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    fill : callable
        Function called as ``fill(start, stop)`` for this rank's bounds when ``dim`` names a single dimension.
    dims : sequence of Hashable
        Dimension names.
    shape : sequence of int, mapping, or None, optional
        Global dimension sizes.
    dim : Hashable, int, or sequence of Hashable, optional
        Dimension or axis to partition.
    dtype : Any, optional
        Data type returned by ``fill``.
    coords : mapping, optional
        Coordinates passed to :class:`xarray.DataArray`.
    name : Hashable, optional
        DataArray name.
    attrs : mapping, optional
        DataArray attributes.
    log_partitions : bool, optional
        Log the resulting rank layout.
    min_partition_size : int, mapping, or None, optional
        Guaranteed minimum local length for any rank that receives data
        along a partitioned dimension, or a per-dimension mapping of the
        same (missing dimensions get no minimum). When the requested rank
        count would otherwise leave some rank with fewer elements than
        this along a given dimension, that dimension's data is instead
        spread across only as many ranks as keep every active rank at or
        above the minimum -- the remaining, highest-numbered ranks get an
        empty local slice for that dimension, the same outcome already
        used when the global size is smaller than the rank count. Set
        this to the widest halo width, ``rolling_reduce``/``coarsen_reduce``
        window, or ``ffill``/``bfill`` ``limit`` you plan to call on the
        result, so :func:`~.arithmetic.mpp_halo_exchange` never raises its
        "local partition shorter than the requested halo" ``ValueError``
        for that dimension regardless of rank count. See
        :func:`~.chunks.get_balanced_bounds`.

    Returns
    -------
    xarray.DataArray
        Lazy rank-local DataArray carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If ``dim`` is invalid or global sizes cannot be resolved.

    """
    partition_dims = _normalize_create_dim(dim, dims)
    min_chunk_map = (
        dict(min_partition_size)
        if isinstance(min_partition_size, Mapping)
        else dict.fromkeys(partition_dims, min_partition_size)
        if min_partition_size is not None
        else {}
    )

    if shape is None or isinstance(shape, Mapping):
        explicit_sizes = dict(shape) if shape else None
    else:
        if len(shape) != len(dims):
            raise ValueError(
                f"shape has {len(shape)} entries but dims has {len(dims)}."
            )
        explicit_sizes = dict(zip(dims, shape, strict=True))
    resolved_sizes = resolve_sizes(dims, explicit_sizes, coords)

    extents = tuple(int(resolved_sizes[d]) for d in partition_dims)
    sizes = dict(zip(partition_dims, extents, strict=True))

    domain = mpp_define_domains(
        mpi_context, sizes, partition_dims, min_partition_size=min_chunk_map
    )
    bounds = {d: (domain.starts[d], domain.stops[d]) for d in partition_dims}
    cart = domain.cart

    local_shape = tuple(
        (bounds[name][1] - bounds[name][0])
        if name in bounds
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
    mpp_update_meta(
        da,
        dim=partition_dims if len(partition_dims) > 1 else partition_dims[0],
        global_size={d: int(resolved_sizes[d]) for d in partition_dims},
        start={d: bounds[d][0] for d in partition_dims},
        stop={d: bounds[d][1] for d in partition_dims},
        chunk_info=chunk_info,
        cart=cart,
    )
    if mpp_should_log_partitions(mpi_context, log_partitions):
        if len(partition_dims) > 1:
            mpp_log_partition_report(
                mpi_context,
                da,
                partition_dims,
                origin="create_dataarray",
                global_size={d: int(resolved_sizes[d]) for d in partition_dims},
                start={d: bounds[d][0] for d in partition_dims},
                stop={d: bounds[d][1] for d in partition_dims},
                grid_shape=cart["grid_shape"],
                coords=cart["coords"],
            )
        else:
            d0 = partition_dims[0]
            mpp_log_partition_report(
                mpi_context,
                da,
                d0,
                origin="create_dataarray",
                global_size=int(resolved_sizes[d0]),
                start=bounds[d0][0],
                stop=bounds[d0][1],
            )
    return da


def mpp_create_dataset(
    mpi_context: MPIContext,
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
    min_partition_size: int | Mapping[Hashable, int] | None = None,
) -> xr.Dataset:
    """Create a distributed Dataset from rank-local variables.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    data_vars : mapping
        Variables as DataArrays or ``(dims, fill)`` pairs.
    sizes : mapping, optional
        Global dimension sizes.
    dim : Hashable or sequence of Hashable
        Dimension(s) to partition.
    dtype : Any or mapping, optional
        Default or per-variable fill dtype.
    coords : mapping, optional
        Coordinates passed to :class:`xarray.Dataset`.
    attrs : mapping, optional
        Dataset attributes.
    log_partitions : bool, optional
        Log the resulting rank layout.
    min_partition_size : int, mapping, or None, optional
        Guaranteed minimum local length for any rank that receives data
        along a partitioned dimension, or a per-dimension mapping of the
        same. See :func:`create_dataarray`'s parameter of the same name
        and :func:`~.chunks.get_balanced_bounds`.

    Returns
    -------
    xarray.Dataset
        Lazy rank-local Dataset carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If sizes cannot be resolved or a partitioned DataArray has the wrong local length.

    """
    if isinstance(dim, (list, tuple)):
        if not dim:
            raise ValueError("dim sequence must not be empty.")
        if len(set(dim)) != len(dim):
            raise ValueError(f"dim entries must be unique; got {tuple(dim)!r}.")
        partition_dims = tuple(dim)
    else:
        partition_dims = (dim,)
    min_chunk_map = (
        dict(min_partition_size)
        if isinstance(min_partition_size, Mapping)
        else dict.fromkeys(partition_dims, min_partition_size)
        if min_partition_size is not None
        else {}
    )

    required_dims: set[Hashable] = set(partition_dims)
    for spec in data_vars.values():
        if not isinstance(spec, xr.DataArray):
            var_dims, _ = spec
            required_dims.update(var_dims)
    resolved_sizes = resolve_sizes(required_dims, sizes, coords)

    extents = tuple(int(resolved_sizes[d]) for d in partition_dims)
    sizes = dict(zip(partition_dims, extents, strict=True))

    domain = mpp_define_domains(
        mpi_context, sizes, partition_dims, min_partition_size=min_chunk_map
    )
    bounds = {d: (domain.starts[d], domain.stops[d]) for d in partition_dims}
    cart = domain.cart

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
                            f"data_vars[{var_name!r}] is a DataArray of "
                            + f"length {spec.sizes[d]} along {d!r}, but "
                            + f"this rank owns [{d_start}:{d_stop}) "
                            + f"({expected_len} elements)"
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
    mpp_update_meta(
        ds,
        dim=partition_dims if len(partition_dims) > 1 else partition_dims[0],
        global_size={d: int(resolved_sizes[d]) for d in partition_dims},
        start={d: bounds[d][0] for d in partition_dims},
        stop={d: bounds[d][1] for d in partition_dims},
        chunk_info=chunk_info,
        cart=cart,
    )
    if mpp_should_log_partitions(mpi_context, log_partitions):
        if len(partition_dims) > 1:
            mpp_log_partition_report(
                mpi_context,
                ds,
                partition_dims,
                origin="create_dataset",
                global_size={d: int(resolved_sizes[d]) for d in partition_dims},
                start={d: bounds[d][0] for d in partition_dims},
                stop={d: bounds[d][1] for d in partition_dims},
                grid_shape=cart["grid_shape"],
                coords=cart["coords"],
            )
        else:
            d0 = partition_dims[0]
            mpp_log_partition_report(
                mpi_context,
                ds,
                d0,
                origin="create_dataset",
                global_size=int(resolved_sizes[d0]),
                start=bounds[d0][0],
                stop=bounds[d0][1],
            )
    return ds


def mpp_repartition(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable | Literal["auto"] = "auto",
    *,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Partition a replicated xarray object across MPI ranks.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Complete object present on every rank.
    dim : Hashable or {"auto"}, optional
        New partition dimension.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Rank-local slice carrying ``mpi_meta``.

    Raises
    ------
    ValueError
        If ``value`` is already distributed or ``dim`` is invalid.

    """
    if mpp_get_meta(value) is not None:
        raise ValueError(
            "Cannot repartition an already distributed object. "
            + "Reduce or gather its distributed dimension first."
        )

    automatic = dim == "auto"
    if automatic:
        if not value.dims:
            return strip_mpi_meta(value)
        dim = choose_partition_dim(
            value.sizes, mpi_context.comm.size, rank=mpi_context.comm.rank
        )

    if dim not in value.dims:
        raise ValueError(f"Repartition dimension {dim!r} does not exist.")

    info = dict(chunk_info or {})
    length = int(value.sizes[dim])
    chunk_size = int(
        info.get(
            str(dim),
            get_effective_chunk_size(length, None, mpi_context.comm.size),
        )
    )
    chunk_size = get_effective_chunk_size(length, chunk_size, mpi_context.comm.size)
    info[str(dim)] = chunk_size

    start, stop = get_chunk_bounds(
        length, chunk_size, mpi_context.comm.rank, mpi_context.comm.size
    )
    output = strip_mpi_meta(value).isel({dim: slice(start, stop)})
    info = prune_chunk_info(info, output)
    for other_dim, other_length in output.sizes.items():
        info.setdefault(
            str(other_dim),
            get_effective_chunk_size(int(other_length), None, mpi_context.comm.size),
        )

    mpp_update_meta(
        output, dim=dim, global_size=length, start=start, stop=stop, chunk_info=info
    )
    if mpp_should_log_partitions(mpi_context, log_partitions):
        mpp_log_partition_report(
            mpi_context,
            output,
            dim,
            origin="repartition",
            global_size=length,
            start=start,
            stop=stop,
            automatic=automatic,
        )
    return output


def mpp_attach_save_chunks(
    mpi_context: MPIContext, value: xr.Dataset | xr.DataArray
) -> xr.Dataset | xr.DataArray:
    """Attach write-time chunk metadata to a distributed object.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Distributed rank-local object.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        ``value`` with ``mpi_meta["save_chunks"]`` attached.

    Raises
    ------
    ValueError
        If required partition chunk metadata are missing.

    """
    meta = mpp_get_meta(value)
    if meta is None:
        return value

    save_chunks: dict[str, tuple[int, ...]] | None = None
    error: BaseException | None = None
    if mpi_context.is_root():
        try:
            save_chunks = compute_save_chunks(value, meta, mpi_context.comm.size)
        except BaseException as exc:
            error = exc
    mpi_context.raise_if_error(error, "attach_save_chunks planning")

    save_chunks = mpi_context.broadcast(save_chunks, root=0)
    set_save_chunks(value, cast("dict[str, tuple[int, ...]]", save_chunks))
    return value


def mpi_open_dataset(
    filename: Path | str | PathLike,
    mpi_context: MPIContext | MPI.Intracomm,
    *,
    partition_dim: Hashable | Sequence[Hashable] | Literal["auto"] = "auto",
    chunks: Any = None,
    log_partitions: bool = True,
    **kwargs: Any,
) -> MPIXarray:
    """Open a Dataset lazily and partition it across MPI ranks.

    Parameters
    ----------
    filename : str, path-like, file-like, or list of these
        Input accepted by ``xarray.open_dataset``/``xarray.open_mfdataset``.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime whose communicator the result is bound to.
    partition_dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Dimension(s) to partition.
    chunks : int, dict, "auto" or None, optional
        Passed unchanged to xarray.
    log_partitions : bool, optional
        Print one aligned table showing which global interval each rank received.
    **kwargs : Any
        Additional arguments passed unchanged to ``xarray.open_dataset``/ ``xarray.open_mfdataset`` (e.g.

    Returns
    -------
    MPIXarray
        Lazy rank-local Dataset with ``.meta`` set.

    """

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    xr.set_options(keep_attrs=True)

    use_mfdataset = (isinstance(filename, str) and "*" in filename) or isinstance(
        filename, (list, tuple)
    )
    open_fn: Callable = xr.open_mfdataset if use_mfdataset else xr.open_dataset

    requested_dims = _as_partition_dims(partition_dim)
    if isinstance(requested_dims, tuple) and len(requested_dims) > 1:
        data = _open_dataset_cartesian(
            mpi_context,
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
            mpi_context,
            filename,
            resolved_dim,
            open_fn,
            chunks,
            log_partitions,
            **kwargs,
        )

    from .core import MPIXarray

    return MPIXarray(data, mpi_context)


def mpi_create_dataarray(
    mpi_context: MPIContext | MPI.Intracomm,
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
    min_partition_size: int | Mapping[Hashable, int] | None = None,
) -> MPIXarray:
    """Create a distributed DataArray from a rank-local fill function.

    Parameters
    ----------
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime or communicator the result is bound to.
    fill : callable
        Function called as ``fill(start, stop)`` for this rank's bounds when ``dim`` names a single dimension, or as ``fill(start_0, stop_0, start_1, stop_1, ...)`` -- one pair per partitioned dimension, in ``dim``'s order -- when ``dim`` names two or more.
    dims : sequence of Hashable
        Dimension names.
    shape : sequence of int, mapping, or None, optional
        Global dimension sizes.
    dim : Hashable, int, or sequence of Hashable, optional
        Dimension or axis to partition.
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
    min_partition_size : int, mapping, or None, optional
        Guaranteed minimum local length per partitioned dimension, or a
        per-dimension mapping of the same; ranks beyond however many keep
        every active rank at or above the minimum get an empty local
        slice for that dimension instead. See
        :func:`~.chunks.get_balanced_bounds`.

    Returns
    -------
    MPIXarray
        Lazy rank-local DataArray with ``.meta`` set.

    """
    from .core import MPIXarray

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    data = mpp_create_dataarray(
        mpi_context,
        fill,
        dims,
        shape=shape,
        dim=dim,
        dtype=dtype,
        coords=coords,
        name=name,
        attrs=attrs,
        log_partitions=log_partitions,
        min_partition_size=min_partition_size,
    )
    return MPIXarray(data, mpi_context)


def mpi_create_dataset(
    mpi_context: MPIContext | MPI.Intracomm,
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
    min_partition_size: int | Mapping[Hashable, int] | None = None,
) -> MPIXarray:
    """Create a distributed Dataset from rank-local variables.

    Parameters
    ----------
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime or communicator the result is bound to.
    data_vars : mapping
        Variables as DataArrays or ``(dims, fill)`` pairs.
    sizes : mapping, optional
        Global dimension sizes.
    dim : Hashable or sequence of Hashable
        Dimension(s) to partition.
    dtype : Any or mapping, optional
        Default or per-variable fill dtype.
    coords : mapping, optional
        Coordinates passed to ``xarray.Dataset``.
    attrs : mapping, optional
        Dataset attributes.
    log_partitions : bool, optional
        Log the resulting rank layout.
    min_partition_size : int, mapping, or None, optional
        Guaranteed minimum local length per partitioned dimension, or a
        per-dimension mapping of the same. See
        :func:`mpi_create_dataarray`'s parameter of the same name and
        :func:`~.chunks.get_balanced_bounds`.

    Returns
    -------
    MPIXarray
        Lazy rank-local Dataset with ``.meta`` set.

    """
    from .core import MPIXarray

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    data = mpp_create_dataset(
        mpi_context,
        data_vars,
        sizes,
        dim=dim,
        dtype=dtype,
        coords=coords,
        attrs=attrs,
        log_partitions=log_partitions,
        min_partition_size=min_partition_size,
    )
    return MPIXarray(data, mpi_context)


def mpi_partition_data(
    value: MPIXarray | xr.Dataset | xr.DataArray | None,
    mpi_context: MPIContext | MPI.Intracomm,
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
    mpi_context : MPIContext or mpi4py.MPI.Intracomm
        Runtime or communicator the result is bound to.
    dim : Hashable, sequence of Hashable, or {"auto"}, optional
        Partition dimension(s).
    root : int, optional
        Rank that owns ``value``.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    MPIXarray
        Rank-local slice with ``.meta`` set.

    """
    from .core import MPIXarray, unwrap

    if not isinstance(mpi_context, MPIContext):
        mpi_context = MPIContext(mpi_context)

    data = mpp_partition(
        mpi_context,
        unwrap(value),
        dim,
        root=root,
        chunk_info=chunk_info,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, mpi_context)


def mpi_empty_dataset() -> xr.Dataset:
    """Return a placeholder Dataset for a non-root MPI rank.

    Returns
    -------
    xarray.Dataset
        Dataset marked as containing no rank-local data.

    """
    return xr.Dataset(attrs={_NO_DATA_ATTR: True})


def mpi_dataset_is_empty(data: xr.Dataset | xr.DataArray) -> bool:
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
    mpi_context: MPIContext | MPI.Intracomm | None = None,
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

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to write.
    file : str or os.PathLike
        Output path.
    mpi_context : MPIContext or mpi4py.MPI.Intracomm, optional
        MPI context or communicator.
    unlimited_dim : str or iterable of str, optional
        Dimension or dimensions made unlimited.
    partition_dim : str, optional
        MPI partition dimension.
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
        if not mpi_context:
            from ..mpi.context import get_mpi_ctx

            mpi_context = get_mpi_ctx()
        if not isinstance(mpi_context, MPIContext):
            mpi_context = MPIContext(mpi_context)

        mpi_meta = mpp_get_meta(data)
        distributed = mpi_meta is not None

        # Ranks must agree on the write path. If one rank saw valid mpi_meta
        # and another did not, the two paths post different collectives and
        # the writer would block instead of reporting the inconsistency.
        agreed = mpi_context.comm.allgather(distributed)
        if any(agreed) and not all(agreed):
            disagreeing = [
                rank for rank, state in enumerate(agreed) if state != agreed[0]
            ]
            raise mpi_context.MPIError(
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
        elif mpi_context.comm.rank != 0:
            data = mpi_empty_dataset()

        mpp_to_netcdf_parallel(
            mpi_context,
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
