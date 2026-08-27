"""MPI-aware distributed xarray operations."""
# xarray_mpi.py

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from functools import cache
from numbers import Integral
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib as _dtlib

import xarray as xr

from .chunks import (
    compute_save_chunks,
    get_balanced_bounds,
    get_chunk_bounds,
    get_chunk_info,
    get_chunk_overrides,
    get_effective_chunk_size,
    get_native_chunk_sizes,
    get_usable_native_chunk,
    prune_chunk_info,
)
from .meta import (
    _delayed_local,
    _localize_coord,
    _resolve_sizes,
    choose_partition_dim,
    get_mpi_meta,
    indexer_is_scalar,
    log_partition_report,
    set_mpi_meta,
    set_save_chunks,
    strip_mpi_meta,
)
from .operator import ArithmeticMixin

# Re-export communicator-free chunk and metadata helpers for compatibility.
# Their implementations live in xr_chunks.py and xr_meta.py.
__all__ = [
    "XarrayMPI",
    "choose_partition_dim",
    "get_balanced_bounds",
    "get_chunk_bounds",
    "get_chunk_info",
    "get_chunk_overrides",
    "get_effective_chunk_size",
    "get_native_chunk_sizes",
    "get_usable_native_chunk",
    "indexer_is_scalar",
    "prune_chunk_info",
]

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..mpi.runtime import MPIRuntime

_OP_LIST: tuple[tuple[Any, str], ...] = (
    (MPI.SUM, "SUM"),
    (MPI.PROD, "PROD"),
    (MPI.MIN, "MIN"),
    (MPI.MAX, "MAX"),
    (MPI.LAND, "LAND"),
    (MPI.LOR, "LOR"),
)


def _op_name(op: MPI.Op) -> str:
    """Return a rank-stable label for an MPI reduction operation."""
    for candidate, name in _OP_LIST:
        if op == candidate:
            return name
    return "OP"


_MPI_REDUCIBLE_KINDS = "biufc"

# Verify that every rank entered a reduction with the same per-variable plan
# before any buffer collective is posted. The check costs one small object
# allgather per reduction and converts an otherwise silent deadlock into an
# immediate exception. Set to False only for micro-benchmarking.
CHECK_COLLECTIVE_AGREEMENT = True


@cache
def _mpi_representable(dtype_string: str) -> bool:
    """Return whether a NumPy dtype has a usable predefined MPI datatype."""
    dtype = np.dtype(dtype_string)
    try:
        datatype = _dtlib.from_numpy_dtype(dtype)
    except BaseException:
        return False
    try:
        return int(datatype.Get_size()) > 0
    except BaseException:
        return False


@cache
def _partial_dtype(
    dtype_string: str, operation: str, skipna: bool | None
) -> np.dtype[Any]:
    """Return the dtype of a rank-local xarray reduction.

    Parameters
    ----------
    dtype_string : str
        NumPy dtype string.
    operation : {"sum", "prod", "min", "max", "count", "any", "all"}
        Reduction operation.
    skipna : bool or None
        Missing-value behavior passed to xarray.

    Returns
    -------
    numpy.dtype
        Dtype produced by the local reduction."""
    probe = xr.DataArray(np.zeros((1,), dtype=np.dtype(dtype_string)), dims=("_probe",))
    if operation == "count":
        return cast("np.dtype[Any]", probe.count(dim="_probe").dtype)
    if operation in ("any", "all"):
        method = probe.all if operation == "all" else probe.any
        return cast("np.dtype[Any]", method(dim="_probe").dtype)

    method = getattr(probe, operation)
    if operation in ("sum", "prod"):
        result = method(dim="_probe", skipna=skipna, min_count=None)
    else:
        result = method(dim="_probe", skipna=skipna)
    return cast("np.dtype[Any]", result.dtype)


class PlanEntry(NamedTuple):
    """Describe one variable in a rank-independent reduction plan.

    Attributes
    ----------
    name : Hashable
        Variable name.
    dims : tuple of Hashable
        Reduced dimensions present on the variable.
    distributed : bool
        Whether the variable spans the active MPI partition dimension.
    dtype : numpy.dtype
        Variable dtype.
    shape : tuple of tuple of (str, int)
        Global dimensions and lengths that survive the reduction."""

    name: Hashable
    dims: tuple[Hashable, ...]
    distributed: bool
    dtype: np.dtype[Any]
    shape: tuple[tuple[str, int], ...]


class XarrayMPI(ArithmeticMixin):
    """MPI-aware xarray operations bound to an MPI runtime.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime whose communicator is used for distributed operations."""

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime

    def open_dataset(
        self,
        filename_or_obj: Any,
        *,
        partition_dim: Hashable | Literal["auto"] = "auto",
        chunks: Any = None,
        log_partitions: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open a Dataset lazily and distribute one dimension across ranks.

        Parameters
        ----------
        filename_or_obj : str, path-like, file-like, or list of these
            Input accepted by :func:`xarray.open_dataset` or :func:`xarray.open_mfdataset`.
            Strings containing a wildcard ("*") or sequences (e.g., list, tuple) will
            automatically trigger multi-file loading.
        partition_dim : Hashable or {"auto"}, optional
            Dimension to distribute. ``"auto"`` selects the longest dimension,
            which is the choice that leaves the fewest ranks idle. Selection is
            deterministic and identical on every rank. Default is "auto".
        chunks : int, dict, "auto" or None, optional
            Passed unchanged to xarray. ``None`` keeps single-file reads
            backend-lazy without Dask; explicit chunking enables Dask
            according to xarray semantics.
        log_partitions : bool, optional
            Print one aligned table showing which global interval each rank
            received. Default is True.
        engine : str, optional
            Engine to use for reading files. Options include 'netcdf4', 'h5netcdf',
            'scipy', 'cfgrib', 'zarr', etc. Passed via ``**kwargs``.
        concat_dim : str, DataArray, Index or list thereof, optional
            (Multi-file only) Dimension(s) over which to concatenate datasets. Passed
            via ``**kwargs``.
        combine : {"by_coords", "nested"}, optional
            (Multi-file only) Whether to combine datasets by matching coordinates or
            by their nested structure. Passed via ``**kwargs``.
        preprocess : callable, optional
            (Multi-file only) If provided, call this function on each dataset prior to
            concatenation. Passed via ``**kwargs``.
        parallel : bool, optional
            (Multi-file only) If True, the open and preprocess steps will be performed
            in parallel using ``dask.delayed``. Passed via ``**kwargs``.
        decode_cf : bool, optional
            Whether to decode these variables, assuming they were saved according to
            CF conventions (e.g., ``mask_and_scale``, ``decode_times``). Passed via ``**kwargs``.
        **kwargs : Any
            Any additional standard arguments passed unchanged to
            :func:`xarray.open_dataset` or :func:`xarray.open_mfdataset` (e.g.,
            ``decode_times``, ``drop_variables``, ``compat``, ``data_vars``).

        Returns
        -------
        xarray.Dataset
            Lazy rank-local Dataset carrying ``mpi_meta``.
        """

        xr.set_options(keep_attrs=True)

        use_mfdataset = (
            isinstance(filename_or_obj, str) and "*" in filename_or_obj
        ) or isinstance(filename_or_obj, (list, tuple))

        open_dataset: Callable = xr.open_mfdataset if use_mfdataset else xr.open_dataset

        automatic = partition_dim == "auto"

        # Build the metadata plan on rank 0.
        plan: dict[str, Any] | None = None
        error: BaseException | None = None
        if self._runtime.is_root():
            try:
                with open_dataset(filename_or_obj, chunks=None, **kwargs) as metadata:
                    if automatic:
                        partition_dim = choose_partition_dim(
                            metadata.sizes,
                            self._runtime.comm.size,
                            rank=self._runtime.comm.rank,
                        )
                    if partition_dim not in metadata.dims:
                        raise ValueError(
                            f"partition_dim {partition_dim!r} is not in "
                            + f"{list(metadata.dims)!r}."
                        )
                    chunk_info = get_chunk_info(metadata, self._runtime.comm.size)
                    global_size = int(metadata.sizes[partition_dim])
                    longest_size = max(
                        int(length) for length in metadata.sizes.values()
                    )

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
        self._runtime.raise_if_error(error, "mpi.xarray.open_dataset planning")

        # Broadcast the plan.
        plan = self._runtime.broadcast(plan, root=0)

        partition_dim = plan["partition_dim"]
        chunk_info = plan["chunk_info"]
        global_size = plan["global_size"]

        # Compute this rank's bounds.
        partition_chunk = chunk_info[str(partition_dim)]
        start, stop = get_chunk_bounds(
            global_size,
            partition_chunk,
            self._runtime.comm.rank,
            self._runtime.comm.size,
        )

        # Synchronize before opening the dataset.
        self._runtime.comm.Barrier()

        # Open this rank's lazy slice.
        data: xr.Dataset = open_dataset(filename_or_obj, chunks=chunks, **kwargs)
        data = data.isel({partition_dim: slice(start, stop)})

        set_mpi_meta(
            data,
            dim=partition_dim,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info=chunk_info,
        )
        if log_partitions:
            log_partition_report(
                self._runtime,
                data,
                partition_dim,
                origin="mpi.xarray.open_dataset",
                global_size=global_size,
                start=start,
                stop=stop,
                automatic=automatic,
            )
        return data

    # mpi4py point-to-point tag for distribute(); arbitrary but fixed so a
    # stray message from unrelated code can never be mistaken for a piece
    # this call is expecting.
    _DISTRIBUTE_TAG = 0x6469_7374  # b"dist" as an int, easy to spot in a trace

    def distribute(
        self,
        value: xr.Dataset | xr.DataArray | None,
        dim: Hashable | Literal["auto"] = "auto",
        *,
        root: int = 0,
        chunk_info: Mapping[str, int] | None = None,
        log_partitions: bool = False,
    ) -> xr.Dataset | xr.DataArray:
        """Distribute a root-owned xarray object across MPI ranks.

        The root slices the object along ``dim`` and sends each rank only its local
        piece. Use :meth:`redistribute` when the full object already exists on every
        rank.

        Parameters
        ----------
        value : xarray.Dataset, xarray.DataArray, or None
            Complete object on ``root``; non-root ranks must pass None.
        dim : Hashable or {"auto"}, optional
            Partition dimension. ``"auto"`` selects the largest dimension.
            Default is ``"auto"``.
        root : int, optional
            Rank that owns ``value``. Default is 0.
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
            If ownership, metadata, or ``dim`` is invalid.

        Notes
        -----
        Dask-backed inputs remain lazy: the root sends sliced task graphs rather
        than materializing the full array."""
        comm = self._runtime.comm
        is_root = self._runtime.is_root(root)

        # Prepare every slice before communication so a root-side failure is
        # synchronized before any rank can block in send/receive.
        error: BaseException | None = None
        pieces: list[Any] | None = None
        replicated_value: xr.Dataset | xr.DataArray | None = None
        try:
            if is_root:
                if value is None:
                    raise ValueError(
                        f"Rank {root} (root) must provide a value, not None."
                    )
                if get_mpi_meta(value) is not None:
                    raise ValueError(
                        "Cannot distribute an already distributed object. "
                        + "Reduce or gather its distributed dimension first."
                    )
                stripped = strip_mpi_meta(value)

                if not stripped.dims:
                    # Nothing to partition: send the (necessarily small)
                    # whole object to every rank as replicated data,
                    # mirroring redistribute's handling of the same case.
                    replicated_value = stripped
                else:
                    automatic = dim == "auto"
                    resolved_dim = (
                        choose_partition_dim(stripped.sizes, comm.size, rank=comm.rank)
                        if automatic
                        else dim
                    )
                    if resolved_dim not in stripped.dims:
                        raise ValueError(
                            f"Distribution dimension {resolved_dim!r} does not exist."
                        )

                    length = int(stripped.sizes[resolved_dim])
                    info = dict(chunk_info or {})
                    chunk_size = int(
                        info.get(
                            str(resolved_dim),
                            get_effective_chunk_size(length, None, comm.size),
                        )
                    )
                    chunk_size = get_effective_chunk_size(length, chunk_size, comm.size)
                    info[str(resolved_dim)] = chunk_size

                    pieces = []
                    for rank in range(comm.size):
                        start, stop = get_chunk_bounds(
                            length, chunk_size, rank, comm.size
                        )
                        piece = stripped.isel({resolved_dim: slice(start, stop)})
                        # Break shallow-copy attribute sharing before adding
                        # rank metadata.
                        piece.attrs = dict(piece.attrs)
                        if isinstance(piece, xr.Dataset):
                            for variable in piece.variables.values():
                                variable.attrs = dict(variable.attrs)
                        piece_info = prune_chunk_info(info, piece)
                        for other_dim, other_length in piece.sizes.items():
                            piece_info.setdefault(
                                str(other_dim),
                                get_effective_chunk_size(
                                    int(other_length), None, comm.size
                                ),
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
            elif value is not None:
                raise ValueError(
                    f"Only rank {root} (root) may provide a value; "
                    + f"got one on rank {comm.rank}."
                )
        except BaseException as exc:
            error = exc
        self._runtime.raise_if_error(error, "mpi.xarray.distribute")

        # Broadcast which transfer path root prepared.
        dimensionless = self._runtime.broadcast(
            replicated_value is not None if is_root else None, root=root
        )

        # Transfer the validated pieces.
        if dimensionless:
            # Nothing to partition: same small object broadcast to every
            # rank, no per-rank slicing or point-to-point send needed.
            output = self._runtime.broadcast(
                replicated_value if is_root else None, root=root
            )
            return cast("xr.Dataset | xr.DataArray", output)

        if is_root:
            assert pieces is not None
            for rank, piece in enumerate(pieces):
                if rank == root:
                    output = piece
                else:
                    self._runtime.send(piece, dest=rank, tag=self._DISTRIBUTE_TAG)
        else:
            output = self._runtime.receive(source=root, tag=self._DISTRIBUTE_TAG)

        if log_partitions:
            meta = get_mpi_meta(output)
            if meta is not None:
                log_partition_report(
                    self._runtime,
                    output,
                    meta["dim"],
                    origin="mpi.xarray.distribute",
                    global_size=meta["global_size"],
                    start=meta["start"],
                    stop=meta["stop"],
                    automatic=(dim == "auto"),
                )
        return output

    def create_dataarray(
        self,
        fill: Callable[[int, int], Any],
        dims: Sequence[Hashable],
        *,
        shape: Sequence[int] | Mapping[Hashable, int] | None = None,
        dim: Hashable | int = 0,
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
            Function called as ``fill(start, stop)`` for this rank's bounds.
        dims : sequence of Hashable
            Dimension names.
        shape : sequence of int, mapping, or None, optional
            Global dimension sizes. Missing sizes may be inferred from ``coords``.
        dim : Hashable or int, optional
            Dimension or axis to partition. Default is 0.
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
        axis = dims.index(dim) if not isinstance(dim, Integral) else int(dim)
        if not 0 <= axis < len(dims):
            raise ValueError(f"dim {dim!r} is not in dims {tuple(dims)!r}.")
        dim_name = dims[axis]

        if shape is None or isinstance(shape, Mapping):
            explicit_sizes = dict(shape) if shape else None
        else:
            if len(shape) != len(dims):
                raise ValueError(
                    f"shape has {len(shape)} entries but dims has {len(dims)}."
                )
            explicit_sizes = dict(zip(dims, shape, strict=True))
        resolved_sizes = _resolve_sizes(dims, explicit_sizes, coords)

        comm = self._runtime.comm
        global_size = int(resolved_sizes[dim_name])
        start, stop = get_balanced_bounds(global_size, comm.rank, comm.size)
        local_shape = tuple(
            stop - start if name == dim_name else int(resolved_sizes[name])
            for name in dims
        )

        local_data = _delayed_local(fill, (start, stop), local_shape, dtype)

        local_coords = dict(coords) if coords else {}
        if dim_name in local_coords:
            local_coords[dim_name] = _localize_coord(
                local_coords[dim_name], global_size, start, stop
            )

        da = xr.DataArray(
            local_data, dims=tuple(dims), coords=local_coords, name=name, attrs=attrs
        )
        set_mpi_meta(
            da,
            dim=dim_name,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info={str(dim_name): stop - start},
        )
        if log_partitions:
            log_partition_report(
                self._runtime,
                da,
                dim_name,
                origin="mpi.xarray.create_dataarray",
                global_size=global_size,
                start=start,
                stop=stop,
            )
        return da

    def create_dataset(
        self,
        data_vars: Mapping[
            Hashable,
            xr.DataArray | tuple[Sequence[Hashable], Callable[[int, int], Any]],
        ],
        sizes: Mapping[Hashable, int] | None = None,
        *,
        dim: Hashable,
        dtype: Any = np.float64,
        coords: Mapping[Hashable, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        log_partitions: bool = True,
    ) -> xr.Dataset:
        """Create a distributed Dataset from rank-local variables.

        Parameters
        ----------
        data_vars : mapping
            Variables as DataArrays or ``(dims, fill)`` pairs. Partitioned fill
            functions receive ``(start, stop)``; unpartitioned fills take no arguments.
        sizes : mapping, optional
            Global dimension sizes. Missing sizes may be inferred from ``coords``.
        dim : Hashable
            Dimension to partition.
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
        required_dims: set[Hashable] = {dim}
        for spec in data_vars.values():
            if not isinstance(spec, xr.DataArray):
                var_dims, _ = spec
                required_dims.update(var_dims)
        resolved_sizes = _resolve_sizes(required_dims, sizes, coords)

        comm = self._runtime.comm
        global_size = int(resolved_sizes[dim])
        start, stop = get_balanced_bounds(global_size, comm.rank, comm.size)

        dtype_map = dtype if isinstance(dtype, Mapping) else None

        built_vars: dict[Hashable, Any] = {}
        for var_name, spec in data_vars.items():
            if isinstance(spec, xr.DataArray):
                if dim in spec.dims and int(spec.sizes[dim]) != stop - start:
                    raise ValueError(
                        f"data_vars[{var_name!r}] is a DataArray of length "
                        + f"{spec.sizes[dim]} along {dim!r}, but this rank "
                        + f"owns [{start}:{stop}) ({stop - start} elements). "
                        + "Pass a DataArray already sized to this rank's own "
                        + "bounds (e.g. from create_dataarray), not the full "
                        + "global array."
                    )
                built_vars[var_name] = spec
                continue

            var_dims, var_fill = spec
            var_dtype = (
                dtype_map.get(var_name, np.float64) if dtype_map is not None else dtype
            )
            if dim in var_dims:
                local_shape = tuple(
                    stop - start if name == dim else int(resolved_sizes[name])
                    for name in var_dims
                )
                local_data = _delayed_local(
                    var_fill, (start, stop), local_shape, var_dtype
                )
            elif callable(var_fill):
                # Not partitioned: identical on every rank, so there is no
                # (start, stop) to give -- fill() takes no arguments and
                # closes over whatever sizes it needs itself.
                local_shape = tuple(int(resolved_sizes[name]) for name in var_dims)
                local_data = _delayed_local(var_fill, (), local_shape, var_dtype)
            else:
                local_data = var_fill
            built_vars[var_name] = (tuple(var_dims), local_data)

        local_coords = dict(coords) if coords else {}
        if dim in local_coords:
            local_coords[dim] = _localize_coord(
                local_coords[dim], global_size, start, stop
            )

        ds = xr.Dataset(built_vars, coords=local_coords, attrs=attrs)
        set_mpi_meta(
            ds,
            dim=dim,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info={str(dim): stop - start},
        )
        if log_partitions:
            log_partition_report(
                self._runtime,
                ds,
                dim,
                origin="mpi.xarray.create_dataset",
                global_size=global_size,
                start=start,
                stop=stop,
            )
        return ds

    def redistribute(
        self,
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
                "Cannot redistribute an already distributed object. "
                + "Reduce or gather its distributed dimension first."
            )

        automatic = dim == "auto"
        if automatic:
            if not value.dims:
                return strip_mpi_meta(value)
            dim = choose_partition_dim(
                value.sizes, self._runtime.comm.size, rank=self._runtime.comm.rank
            )

        if dim not in value.dims:
            raise ValueError(f"Redistribution dimension {dim!r} does not exist.")

        info = dict(chunk_info or {})
        length = int(value.sizes[dim])
        chunk_size = int(
            info.get(
                str(dim),
                get_effective_chunk_size(length, None, self._runtime.comm.size),
            )
        )
        chunk_size = get_effective_chunk_size(
            length, chunk_size, self._runtime.comm.size
        )
        info[str(dim)] = chunk_size

        start, stop = get_chunk_bounds(
            length, chunk_size, self._runtime.comm.rank, self._runtime.comm.size
        )
        output = strip_mpi_meta(value).isel({dim: slice(start, stop)})
        info = prune_chunk_info(info, output)
        for other_dim, other_length in output.sizes.items():
            info.setdefault(
                str(other_dim),
                get_effective_chunk_size(
                    int(other_length), None, self._runtime.comm.size
                ),
            )

        set_mpi_meta(
            output, dim=dim, global_size=length, start=start, stop=stop, chunk_info=info
        )
        if log_partitions:
            log_partition_report(
                self._runtime,
                output,
                dim,
                origin="mpi.xarray.redistribute",
                global_size=length,
                start=start,
                stop=stop,
                automatic=automatic,
            )
        return output

    def attach_save_chunks(
        self, value: xr.Dataset | xr.DataArray
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

        save_chunks: dict[str, tuple[int, ...]] | None = None
        error: BaseException | None = None
        if self._runtime.is_root():
            try:
                save_chunks = compute_save_chunks(value, meta, self._runtime.comm.size)
            except BaseException as exc:
                error = exc
        self._runtime.raise_if_error(error, "mpi.xarray.attach_save_chunks planning")

        save_chunks = self._runtime.broadcast(save_chunks, root=0)
        set_save_chunks(value, cast("dict[str, tuple[int, ...]]", save_chunks))
        return value

    def isel(
        self,
        value: xr.Dataset | xr.DataArray,
        indexers: Mapping[Any, Any] | None = None,
        **indexers_kwargs: Any,
    ) -> xr.Dataset | xr.DataArray:
        """Index a distributed object with global integer coordinates.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to index.
        indexers : mapping, optional
            Integer indexers using global coordinates on the partition dimension.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Indexed object with updated distribution metadata. A scalar selection on
            the partition dimension is replicated on every rank."""
        supplied = dict(indexers or {})
        supplied.update(indexers_kwargs)
        meta = get_mpi_meta(value)
        if meta is None:
            return value.isel(supplied)

        dim = meta["dim"]
        if dim not in supplied:
            return value.isel(supplied)

        distributed_indexer = supplied.pop(dim)
        if indexer_is_scalar(distributed_indexer):
            return self.isel_scalar(value, dim, int(distributed_indexer), supplied)

        if not isinstance(distributed_indexer, slice):
            raise NotImplementedError(
                "Distributed isel currently supports slices and scalar indices."
            )
        if distributed_indexer.step not in (None, 1):
            raise NotImplementedError(
                "Distributed isel currently requires slice step 1."
            )

        global_size = int(meta["global_size"])
        requested_start, requested_stop, _ = distributed_indexer.indices(global_size)
        local_global_start = max(requested_start, int(meta["start"]))
        local_global_stop = min(requested_stop, int(meta["stop"]))
        local_global_stop = max(local_global_start, local_global_stop)

        local_start = local_global_start - int(meta["start"])
        local_stop = local_global_stop - int(meta["start"])
        local_indexers = dict(supplied)
        local_indexers[dim] = slice(local_start, local_stop)
        output = value.isel(local_indexers)

        counts = self._runtime.comm.allgather(int(output.sizes[dim]))
        new_start = sum(counts[: self._runtime.comm.rank])
        new_stop = new_start + counts[self._runtime.comm.rank]
        new_global_size = sum(counts)
        chunk_info = prune_chunk_info(meta["chunk_info"], output)
        set_mpi_meta(
            output,
            dim=dim,
            global_size=new_global_size,
            start=new_start,
            stop=new_stop,
            chunk_info=chunk_info,
        )
        return output

    def isel_scalar(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable,
        index: int,
        other_indexers: Mapping[Any, Any],
    ) -> xr.Dataset | xr.DataArray:
        """Select one global integer index from the partition dimension.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Distributed object.
        dim : Hashable
            Partition dimension.
        index : int
            Global integer index.
        other_indexers : mapping
            Additional local ``isel`` indexers.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Replicated selected slice.

        Raises
        ------
        IndexError
            If ``index`` is outside the global dimension."""
        meta = get_mpi_meta(value)
        if meta is None:
            return value.isel({dim: index, **other_indexers})

        global_size = int(meta["global_size"])
        normalized = index + global_size if index < 0 else index
        if normalized < 0 or normalized >= global_size:
            raise IndexError(
                f"index {index} is out of bounds for dimension {dim!r} "
                + f"with size {global_size}."
            )

        owner = None
        parts = self._runtime.comm.allgather((int(meta["start"]), int(meta["stop"])))
        for rank, (start, stop) in enumerate(parts):
            if start <= normalized < stop:
                owner = rank
                break
        if owner is None:
            raise RuntimeError("Distributed partitions do not own the requested index.")

        result = None
        if self._runtime.comm.rank == owner:
            local_index = normalized - int(meta["start"])
            result = strip_mpi_meta(value).isel({dim: local_index, **other_indexers})
        return cast(
            "xr.Dataset | xr.DataArray", self._runtime.broadcast(result, root=owner)
        )

    def sel(
        self,
        value: xr.Dataset | xr.DataArray,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: Any = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> xr.Dataset | xr.DataArray:
        """Index a distributed object with global coordinate labels.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to index.
        indexers : mapping, optional
            Label indexers using global semantics on the partition dimension.
        method : str, optional
            Inexact matching method passed to xarray.
        tolerance : Any, optional
            Maximum distance for inexact matches.
        drop : bool, optional
            Drop selected coordinate variables. Default is False.
        **indexers_kwargs : Any
            Additional indexers passed by dimension name.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Indexed object with updated distribution metadata. A scalar selection on
            the partition dimension is replicated on every rank."""
        supplied = dict(indexers or {})
        supplied.update(indexers_kwargs)
        meta = get_mpi_meta(value)
        if meta is None:
            return value.sel(supplied, method=method, tolerance=tolerance, drop=drop)

        dim = meta["dim"]
        if dim not in supplied:
            return value.sel(supplied, method=method, tolerance=tolerance, drop=drop)

        distributed_indexer = supplied.pop(dim)
        if indexer_is_scalar(distributed_indexer):
            return self.sel_scalar(
                value,
                dim,
                distributed_indexer,
                supplied,
                method=method,
                tolerance=tolerance,
                drop=drop,
            )

        if not isinstance(distributed_indexer, slice):
            raise NotImplementedError(
                "Distributed sel currently supports slices and scalar labels."
            )

        local_indexers = dict(supplied)
        local_indexers[dim] = distributed_indexer
        output = value.sel(
            local_indexers, method=method, tolerance=tolerance, drop=drop
        )
        counts = self._runtime.comm.allgather(int(output.sizes[dim]))
        new_start = sum(counts[: self._runtime.comm.rank])
        new_stop = new_start + counts[self._runtime.comm.rank]
        chunk_info = prune_chunk_info(meta["chunk_info"], output)
        set_mpi_meta(
            output,
            dim=dim,
            global_size=sum(counts),
            start=new_start,
            stop=new_stop,
            chunk_info=chunk_info,
        )
        return output

    def sel_scalar(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable,
        label: Any,
        other_indexers: Mapping[Any, Any],
        *,
        method: str | None,
        tolerance: Any,
        drop: bool,
    ) -> xr.Dataset | xr.DataArray:
        """Select one global label from the partition dimension.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Distributed object.
        dim : Hashable
            Partition dimension.
        label : Any
            Global coordinate label.
        other_indexers : mapping
            Additional non-partition ``sel`` indexers.
        method : str or None
            Inexact matching method.
        tolerance : Any
            Maximum distance for inexact matches.
        drop : bool
            Whether to drop selected coordinates.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Replicated selected slice."""
        if method is not None:
            meta = get_mpi_meta(value)
            if meta is None:
                return value.sel(
                    {dim: label, **other_indexers},
                    method=method,
                    tolerance=tolerance,
                    drop=drop,
                )

            if dim in value.coords:
                local_coord = np.asarray(value[dim].values)
            else:
                local_coord = np.arange(int(meta["start"]), int(meta["stop"]))
            coord_parts = self._runtime.comm.allgather(local_coord)
            global_coord = np.concatenate(coord_parts)
            locator = xr.DataArray(
                np.arange(global_coord.size, dtype=np.int64),
                dims=(dim,),
                coords={dim: global_coord},
            )
            selected = locator.sel({dim: label}, method=method, tolerance=tolerance)
            if selected.ndim != 0:
                raise NotImplementedError(
                    "Inexact distributed sel requires a unique one-dimensional index."
                )
            global_index = int(selected.item())

            bounds = self._runtime.comm.allgather(
                (int(meta["start"]), int(meta["stop"]))
            )
            owner = next(
                rank
                for rank, (start, stop) in enumerate(bounds)
                if start <= global_index < stop
            )

            result = None
            error: BaseException | None = None
            if self._runtime.comm.rank == owner:
                try:
                    local_index = global_index - int(meta["start"])
                    result = strip_mpi_meta(value).isel({dim: local_index}, drop=drop)
                    if other_indexers:
                        result = result.sel(
                            other_indexers,
                            method=method,
                            tolerance=tolerance,
                            drop=drop,
                        )
                except BaseException as exc:
                    error = exc
            self._runtime.raise_if_error(error, "distributed scalar selection")
            return cast(
                "xr.Dataset | xr.DataArray",
                self._runtime.broadcast(result, root=owner),
            )

        result = None
        found = False
        try:
            result = strip_mpi_meta(value).sel(
                {dim: label, **other_indexers},
                method=method,
                tolerance=tolerance,
                drop=drop,
            )
            found = True
        except (KeyError, IndexError):
            pass

        found_ranks = self._runtime.comm.allgather(found)
        owners = [rank for rank, state in enumerate(found_ranks) if state]
        if not owners:
            raise KeyError(f"No rank contains label {label!r} on {dim!r}.")
        if len(owners) > 1:
            raise NotImplementedError(
                "Distributed scalar sel requires labels to be owned by one rank."
            )
        owner = owners[0]
        payload = result if self._runtime.comm.rank == owner else None
        return cast(
            "xr.Dataset | xr.DataArray", self._runtime.broadcast(payload, root=owner)
        )

    # -- collective planning -------------------------------------------------

    @staticmethod
    def _normalize_dim(
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
    ) -> tuple[Any, tuple[Hashable, ...]]:
        """Normalize a reduction dimension specification."""
        if not isinstance(value, (xr.DataArray, xr.Dataset)):
            raise TypeError(
                "MPI xarray operations require an xarray DataArray or Dataset."
            )
        if dim is None or dim is ...:
            return dim, tuple(value.dims)
        if isinstance(dim, str):
            return dim, (dim,)
        dims = tuple(dim)
        return dims, dims

    @staticmethod
    def _variable_is_distributed(
        value: xr.DataArray, meta: Mapping[str, Any] | None
    ) -> bool:
        """Return whether a variable spans the active partition dimension."""
        return meta is not None and meta["dim"] in value.dims

    @staticmethod
    def _skipna_enabled(dtype: np.dtype[Any], skipna: bool | None) -> bool:
        """Return the effective dtype-aware ``skipna`` setting."""
        if skipna is not None:
            return skipna
        return dtype.kind in "fc"

    @staticmethod
    def _check_reducible(dtype: np.dtype[Any], operation: str) -> None:
        """Validate that a dtype supports the requested MPI reduction."""
        if operation in ("any", "all"):
            return
        if dtype.kind not in _MPI_REDUCIBLE_KINDS:
            raise TypeError(f"Unsupported MPI xarray dtype: {dtype}.")
        if not _mpi_representable(dtype.str):
            # float16 and long double have a reducible NumPy kind but no
            # predefined MPI datatype. Rejecting them here raises on every
            # rank before any collective, instead of failing inside
            # Allreduce with MPI_ERR_TYPE once buffers are already posted.
            raise TypeError(
                f"Unsupported MPI xarray dtype: {dtype}. "
                + "No predefined MPI datatype represents it."
            )
        if operation in ("min", "max") and dtype.kind == "c":
            name = "minimum" if operation == "min" else "maximum"
            raise TypeError(f"MPI {name} is not defined for complex xarray data.")

    @staticmethod
    def _local_reduction_meta(
        meta: Mapping[str, Any] | None,
        dims: tuple[Hashable, ...],
        *,
        redistribute_on: Hashable | Literal["auto"] | None,
    ) -> Mapping[str, Any] | None:
        """Return metadata when a reduction remains rank-local."""
        if meta is None or meta["dim"] in dims:
            return None
        if redistribute_on not in (None, "auto"):
            raise ValueError(
                "redistribute_on can name a new dimension only after the active "
                + "partition dimension has been reduced away."
            )
        return meta

    @staticmethod
    def _finish_local_reduction(
        result: xr.Dataset | xr.DataArray, *, old_meta: Mapping[str, Any]
    ) -> xr.Dataset | xr.DataArray:
        """Restore metadata after a rank-local reduction."""
        partition_dim = old_meta["dim"]
        if partition_dim not in result.dims:
            return strip_mpi_meta(result)
        set_mpi_meta(
            result,
            dim=partition_dim,
            global_size=int(old_meta["global_size"]),
            start=int(old_meta["start"]),
            stop=int(old_meta["stop"]),
            chunk_info=prune_chunk_info(old_meta["chunk_info"], result),
        )
        return result

    def _agree(self, signature: tuple[Any, ...]) -> None:
        """Verify that all ranks entered the same reduction plan."""
        if not CHECK_COLLECTIVE_AGREEMENT or self._runtime.comm.size == 1:
            return
        digest = hashlib.blake2b(repr(signature).encode(), digest_size=16).digest()
        self._runtime.raise_if_error(
            None,
            "MPI xarray reduction planning",
            signature=("xarray_reduction_plan", digest),
        )

    def _plan(
        self,
        value: xr.Dataset | xr.DataArray,
        dims: tuple[Hashable, ...],
        meta: Mapping[str, Any] | None,
        *,
        operation: str,
    ) -> tuple[PlanEntry, ...]:
        """Build and validate the rank-independent reduction plan."""
        if isinstance(value, xr.DataArray):
            items: tuple[tuple[Hashable, xr.DataArray], ...] = ((value.name, value),)
        else:
            items = tuple((name, value[name]) for name in value.data_vars)

        entries = []
        for name, variable in items:
            variable_dims = tuple(dim for dim in dims if dim in variable.dims)
            if variable_dims:
                self._check_reducible(variable.dtype, operation)
            entries.append(
                PlanEntry(
                    name=name,
                    dims=variable_dims,
                    distributed=self._variable_is_distributed(variable, meta),
                    dtype=variable.dtype,
                    shape=tuple(
                        (str(dim), int(value.sizes[dim]))
                        for dim in variable.dims
                        if dim not in variable_dims
                    ),
                )
            )

        plan = tuple(entries)
        self._agree(
            (
                operation,
                tuple(str(dim) for dim in dims),
                tuple(
                    (
                        str(entry.name),
                        tuple(str(dim) for dim in entry.dims),
                        entry.distributed,
                        str(entry.dtype),
                        entry.shape,
                    )
                    for entry in plan
                ),
            )
        )
        return plan

    @staticmethod
    def _guarded(function: Any) -> tuple[Any, BaseException | None]:
        """Run a local operation and defer any exception for synchronization."""
        try:
            return function(), None
        except BaseException as exc:
            return None, exc

    def _partition_is_empty(self, value: xr.Dataset | xr.DataArray, meta: Any) -> bool:
        """Return whether this rank owns an empty partition."""
        if meta is None:
            return False
        dim = meta["dim"]
        return dim in value.dims and int(value.sizes[dim]) == 0

    # -- collective primitives -----------------------------------------------

    def _comm_reduce(
        self,
        value: xr.DataArray | None,
        op: MPI.Op,
        *,
        expect_dtype: np.dtype[Any] | None = None,
        error: BaseException | None = None,
        phase: str = "MPI xarray reduction buffer preparation",
    ) -> xr.DataArray:
        """Combine a validated DataArray buffer across ranks."""
        send: np.ndarray[Any, Any] | None = None
        if error is None:
            try:
                if value is None:
                    raise AssertionError("MPI xarray reduction buffer is missing.")
                send = np.asarray(value.values)
                if expect_dtype is not None and send.dtype != np.dtype(expect_dtype):
                    send = send.astype(expect_dtype)
                if not send.flags.c_contiguous:
                    send = np.ascontiguousarray(send)
                if send.dtype.kind not in _MPI_REDUCIBLE_KINDS:
                    raise TypeError(f"Unsupported MPI xarray dtype: {send.dtype}.")
                if not _mpi_representable(send.dtype.str):
                    raise TypeError(
                        f"Unsupported MPI xarray dtype: {send.dtype}. "
                        + "No predefined MPI datatype represents it."
                    )
            except BaseException as exc:
                error = exc
                send = None

        signature = (
            None
            if send is None
            else (
                _op_name(op),
                send.dtype.str,
                tuple(int(length) for length in send.shape),
            )
        )
        self._runtime.raise_if_error(error, phase, signature)
        if send is None or value is None:
            raise AssertionError("MPI xarray reduction buffer is missing.")

        recv = self._exchange(send, op)
        return value.copy(data=recv)

    def _exchange(self, send: np.ndarray[Any, Any], op: MPI.Op) -> np.ndarray[Any, Any]:
        """All-reduce a validated contiguous NumPy buffer."""
        recv = np.empty(send.shape, dtype=send.dtype)
        self._runtime.comm.Allreduce(send, recv, op=op)
        return recv

    def _count(self, value: xr.DataArray, dims: tuple[Hashable, ...]) -> xr.DataArray:
        """Count valid values globally across the requested dimensions."""
        count: xr.DataArray | None = None
        error: BaseException | None = None
        try:
            count = value.count(dim=dims, keep_attrs=False)
        except BaseException as exc:
            error = exc
        return self._comm_reduce(
            count,
            MPI.SUM,
            expect_dtype=_partial_dtype(value.dtype.str, "count", None),
            error=error,
            phase="MPI xarray count reduction",
        )

    @staticmethod
    def _dataset_result(
        value: xr.Dataset,
        dims: tuple[Hashable, ...],
        variables: Mapping[Hashable, xr.DataArray],
    ) -> xr.Dataset:
        """Rebuild a Dataset from reduced data variables."""
        reduced = set(dims)
        coords = {
            name: coord
            for name, coord in value.coords.items()
            if not reduced & set(coord.dims)
        }
        return xr.Dataset(dict(variables), coords=coords, attrs=dict(value.attrs))

    @staticmethod
    def _redistribution_candidates(plan: tuple[PlanEntry, ...]) -> frozenset[Hashable]:
        """Return dimensions eligible for post-reduction redistribution."""
        return frozenset(
            dim for entry in plan if entry.distributed for dim, _ in entry.shape
        )

    def _finish(
        self,
        result: xr.Dataset | xr.DataArray,
        *,
        old_meta: Mapping[str, Any] | None,
        redistribute_on: Hashable | Literal["auto"] | None,
        auto_candidates: frozenset[Hashable],
    ) -> xr.Dataset | xr.DataArray:
        """Finalize metadata and optional redistribution after a reduction."""
        result = strip_mpi_meta(result)
        partition_removed = old_meta is not None and old_meta["dim"] not in result.dims

        if redistribute_on is None:
            return result

        target = redistribute_on
        if redistribute_on == "auto":
            if not partition_removed:
                return result
            sizes = {
                dim: length
                for dim, length in result.sizes.items()
                if dim in auto_candidates
            }
            if not any(int(length) > 1 for length in sizes.values()):
                return result
            target = choose_partition_dim(
                sizes, self._runtime.comm.size, rank=self._runtime.comm.rank
            )
        elif redistribute_on not in auto_candidates:
            raise ValueError(
                f"redistribute_on={redistribute_on!r} is not a dimension of any "
                + "variable that required an MPI collective in this reduction; "
                + "an untouched, replicated variable's own dimension cannot be "
                + "used as the new partition dimension."
            )

        chunk_info = (
            prune_chunk_info(old_meta["chunk_info"], result)
            if old_meta is not None
            else {}
        )
        return self.redistribute(result, target, chunk_info=chunk_info)

    # -- per-variable combination --------------------------------------------

    def _combine_sum_or_prod(
        self,
        value: xr.DataArray,
        partial: xr.DataArray,
        dims: tuple[Hashable, ...],
        op: MPI.Op,
        *,
        skipna: bool | None,
        min_count: int | None,
        error: BaseException | None = None,
    ) -> xr.DataArray:
        """Combine rank-local sum or product partials."""
        result = self._comm_reduce(
            partial,
            op,
            expect_dtype=_partial_dtype(
                value.dtype.str, "prod" if _op_name(op) == "PROD" else "sum", skipna
            ),
            error=error,
            phase="MPI xarray sum/prod reduction",
        )
        global_count = None
        if min_count is not None and self._skipna_enabled(value.dtype, skipna):
            global_count = self._count(value, dims)
        if global_count is not None:
            # where() introduces NaN, which requires a floating result. Restore
            # the partial's own dtype so a float32 field stays float32.
            masked = result.where(global_count >= min_count)
            result = (
                masked
                if masked.dtype == result.dtype or result.dtype.kind not in "fc"
                else masked.astype(result.dtype, keep_attrs=True)
            )
        return result

    def _combine_mean(
        self,
        value: xr.DataArray,
        partial_sum: xr.DataArray | None,
        dims: tuple[Hashable, ...],
        *,
        skipna: bool | None = None,
        error: BaseException | None = None,
    ) -> xr.DataArray:
        """Combine rank-local sums and counts into a global mean."""
        global_sum = self._comm_reduce(
            partial_sum,
            MPI.SUM,
            expect_dtype=_partial_dtype(value.dtype.str, "sum", skipna),
            error=error,
            phase="MPI xarray mean reduction",
        )
        global_count = self._count(value, dims)
        # Divide in the dtype numpy.mean would produce for this input. Dividing
        # the float32 sum by the int64 count directly would promote the whole
        # array to float64 and then cast it back, costing two full-width
        # temporaries for a result that is float32 either way.
        target = np.asarray(np.mean(np.zeros(1, dtype=value.dtype))).dtype
        divisor = (
            global_count.astype(target, keep_attrs=False)
            if target.kind in "fc"
            else global_count
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            result = global_sum / divisor
        result = result.where(global_count != 0)
        if result.dtype != target:
            result = result.astype(target, keep_attrs=True)
        return result

    @staticmethod
    def _empty_extreme_partial(
        value: xr.DataArray,
        dims: tuple[Hashable, ...],
        *,
        minimum: bool,
        keep_attrs: bool | None,
    ) -> xr.DataArray:
        """Create the neutral partial for an empty min/max partition."""
        kind = value.dtype.kind
        if kind == "b":
            identity: Any = bool(minimum)
        elif kind in "iu":
            limits = np.iinfo(value.dtype)
            identity = limits.max if minimum else limits.min
        elif kind == "f":
            identity = np.asarray(
                np.inf if minimum else -np.inf, dtype=value.dtype
            ).item()
        else:
            name = "minimum" if minimum else "maximum"
            raise TypeError(f"MPI {name} is not defined for {value.dtype} data.")

        template = value.sum(dim=dims, skipna=False, keep_attrs=keep_attrs)
        return xr.full_like(template, identity, dtype=value.dtype)

    def _local_extreme(
        self,
        variable: xr.DataArray,
        variable_dims: tuple[Hashable, ...],
        *,
        empty: bool,
        minimum: bool,
        skipna: bool | None,
        keep_attrs: bool | None,
    ) -> xr.DataArray:
        """Compute a rank-local min/max partial."""
        if empty:
            return self._empty_extreme_partial(
                variable, variable_dims, minimum=minimum, keep_attrs=keep_attrs
            )
        method = variable.min if minimum else variable.max
        return method(dim=variable_dims, skipna=skipna, keep_attrs=keep_attrs)

    def _combine_extreme(
        self,
        value: xr.DataArray,
        partial: xr.DataArray | None,
        dims: tuple[Hashable, ...],
        *,
        minimum: bool,
        skipna: bool | None,
        error: BaseException | None = None,
    ) -> xr.DataArray:
        """Combine rank-local min/max partials across ranks."""
        # Use the agreed variable dtype, not a rank-local partial dtype. Empty
        # partitions follow a different local path, and dtype-dependent branching
        # could desynchronize collectives. Min/max also require no promotion; using
        # the declared dtype avoids bottleneck's float32-to-float64 scalar promotion.
        operation = "min" if minimum else "max"
        expect_dtype = value.dtype
        kind = value.dtype.kind
        if kind == "b":
            return self._comm_reduce(
                partial,
                MPI.LAND if minimum else MPI.LOR,
                expect_dtype=expect_dtype,
                error=error,
                phase=f"MPI xarray {operation} reduction",
            )

        op = MPI.MIN if minimum else MPI.MAX
        if kind != "f":
            return self._comm_reduce(
                partial,
                op,
                expect_dtype=expect_dtype,
                error=error,
                phase=f"MPI xarray {operation} reduction",
            )

        # Floating reductions carry validity beside the extreme so empty or all-NaN
        # partitions can use an identity without confusing it with real infinity.
        # Encoding the flag in the same buffer avoids a second boolean collective.
        send: np.ndarray[Any, Any] | None = None
        template: xr.DataArray | None = None
        skipna_enabled = self._skipna_enabled(value.dtype, skipna)
        # ANY valid rank suffices under skipna; without it every rank must be
        # NaN-free for the result to be defined.
        flip = -1.0 if ((not minimum) != skipna_enabled) else 1.0

        if error is None:
            try:
                identity = np.asarray(
                    np.inf if minimum else -np.inf, dtype=expect_dtype
                ).item()
                if skipna_enabled:
                    good = value.count(dim=dims, keep_attrs=False) > 0
                else:
                    good = ~value.isnull().any(dim=dims, keep_attrs=False)
                safe_partial = partial.where(good, other=identity)
                if safe_partial.dtype != expect_dtype:
                    safe_partial = safe_partial.astype(expect_dtype, keep_attrs=True)
                template = safe_partial

                values = np.ascontiguousarray(
                    np.asarray(safe_partial.values, dtype=expect_dtype)
                )
                flags = np.where(
                    np.asarray(good.values, dtype=bool),
                    np.asarray(flip, dtype=expect_dtype),
                    np.zeros((), dtype=expect_dtype),
                )
                send = np.empty((2, values.size), dtype=expect_dtype)
                send[0] = np.reshape(values, values.size)
                send[1] = np.reshape(flags, values.size)
            except BaseException as exc:
                error = exc
                send = None
                template = None

        signature = (
            None
            if send is None
            else (
                _op_name(op),
                send.dtype.str,
                tuple(int(length) for length in send.shape),
            )
        )
        self._runtime.raise_if_error(
            error, f"MPI xarray {operation} reduction", signature
        )
        if send is None or template is None:
            raise AssertionError("MPI xarray reduction buffer is missing.")

        recv = self._exchange(send, op)

        shape = tuple(int(length) for length in template.shape)
        combined = np.asarray(recv[0]).reshape(shape)
        valid = (np.asarray(recv[1]).reshape(shape) * flip) > 0
        masked = np.where(valid, combined, np.asarray(np.nan, dtype=expect_dtype))
        return template.copy(data=np.asarray(masked, dtype=expect_dtype).reshape(shape))

    # -- public reductions ---------------------------------------------------

    def sum(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Sum a distributed xarray object over one or more dimensions.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to reduce.
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        min_count : int or None, optional
            Minimum number of valid values required.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            ``"auto"`` selects a surviving dimension; None leaves the result
            replicated. Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Reduced object.

        Notes
        -----
        MPI communication occurs only when ``dim`` includes the active partition
        dimension."""
        return self._sum_prod(
            value,
            dim,
            op=MPI.SUM,
            product=False,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def prod(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Multiply a distributed xarray object over one or more dimensions.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to reduce.
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        min_count : int or None, optional
            Minimum number of valid values required.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            ``"auto"`` selects a surviving dimension; None leaves the result
            replicated. Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Reduced object.

        Notes
        -----
        MPI communication occurs only when ``dim`` includes the active partition
        dimension."""
        return self._sum_prod(
            value,
            dim,
            op=MPI.PROD,
            product=True,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def _sum_prod(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        op: MPI.Op,
        product: bool,
        skipna: bool | None,
        min_count: int | None,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
    ) -> xr.Dataset | xr.DataArray:
        """Implement distributed sum and product reductions."""
        operation = "prod" if product else "sum"
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )
        if local_meta is not None:
            method = value.prod if product else value.sum
            local_result = method(
                dim=local_dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs
            )
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation=operation)

        if isinstance(value, xr.DataArray):
            method = value.prod if product else value.sum
            local, local_error = self._guarded(
                lambda: method(
                    dim=local_dim, skipna=skipna, min_count=None, keep_attrs=keep_attrs
                )
            )
            if not dims:
                if local_error is not None:
                    raise local_error
                return local
            result = self._combine_sum_or_prod(
                value,
                local,
                dims,
                op,
                skipna=skipna,
                min_count=min_count,
                error=local_error,
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            method = variable.prod if product else variable.sum
            local, local_error = self._guarded(
                lambda method=method, entry=entry: method(
                    dim=entry.dims, skipna=skipna, min_count=None, keep_attrs=keep_attrs
                )
            )
            if not entry.distributed:
                if local_error is not None:
                    raise local_error
                variables[entry.name] = local
                continue
            result = self._combine_sum_or_prod(
                variable,
                local,
                entry.dims,
                op,
                skipna=skipna,
                min_count=min_count,
                error=local_error,
            )
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            auto_candidates=self._redistribution_candidates(plan),
        )

    def mean(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Compute the mean of a distributed xarray object.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to reduce.
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Reduced object.

        Notes
        -----
        MPI communication occurs only when ``dim`` includes the active partition
        dimension."""
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )
        if local_meta is not None:
            local_result = value.mean(
                dim=local_dim, skipna=skipna, keep_attrs=keep_attrs
            )
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation="mean")

        if isinstance(value, xr.DataArray):
            if not dims:
                local_mean = value.mean(
                    dim=local_dim, skipna=skipna, keep_attrs=keep_attrs
                )
                return local_mean
            local_sum, local_error = self._guarded(
                lambda: value.sum(
                    dim=local_dim, skipna=skipna, min_count=None, keep_attrs=keep_attrs
                )
            )
            result = self._combine_mean(
                value, local_sum, dims, skipna=skipna, error=local_error
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            if not entry.distributed:
                variables[entry.name] = variable.mean(
                    dim=entry.dims, skipna=skipna, keep_attrs=keep_attrs
                )
                continue
            local_sum, local_error = self._guarded(
                lambda variable=variable, entry=entry: variable.sum(
                    dim=entry.dims, skipna=skipna, min_count=None, keep_attrs=keep_attrs
                )
            )
            result = self._combine_mean(
                variable, local_sum, entry.dims, skipna=skipna, error=local_error
            )
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            auto_candidates=self._redistribution_candidates(plan),
        )

    def min(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Compute the minimum of a distributed xarray object.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to reduce.
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Reduced object."""
        return self._min_max(
            value,
            dim,
            minimum=True,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def max(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Compute the maximum of a distributed xarray object.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to reduce.
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
        skipna : bool or None, optional
            Missing-value behavior, following xarray semantics.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Reduced object."""
        return self._min_max(
            value,
            dim,
            minimum=False,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def _min_max(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        minimum: bool,
        skipna: bool | None,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
    ) -> xr.Dataset | xr.DataArray:
        """Implement distributed minimum and maximum reductions."""
        operation = "min" if minimum else "max"
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )
        if local_meta is not None:
            method = value.min if minimum else value.max
            local_result = method(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation=operation)
        empty_partition = self._partition_is_empty(value, old_meta)

        if isinstance(value, xr.DataArray):
            if not dims:
                method = value.min if minimum else value.max
                return method(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
            local, local_error = self._guarded(
                lambda: self._local_extreme(
                    value,
                    dims,
                    empty=empty_partition,
                    minimum=minimum,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                )
            )
            result = self._combine_extreme(
                value, local, dims, minimum=minimum, skipna=skipna, error=local_error
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            local, local_error = self._guarded(
                lambda variable=variable, entry=entry: self._local_extreme(
                    variable,
                    entry.dims,
                    empty=empty_partition and entry.distributed,
                    minimum=minimum,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                )
            )
            if not entry.distributed:
                if local_error is not None:
                    raise local_error
                variables[entry.name] = local
                continue
            result = self._combine_extreme(
                variable,
                local,
                entry.dims,
                minimum=minimum,
                skipna=skipna,
                error=local_error,
            )
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            auto_candidates=self._redistribution_candidates(plan),
        )

    def any(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Return whether any value is true over the requested dimensions.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to reduce.
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Logical OR over the requested dimensions."""
        return self._logical(
            value,
            dim,
            op=MPI.LOR,
            all_values=False,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def all(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Return whether all values are true over the requested dimensions.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Object to reduce.
        dim : str, iterable of Hashable, ..., or None, optional
            Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
        keep_attrs : bool or None, optional
            Whether to preserve attributes.
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Logical AND over the requested dimensions."""
        return self._logical(
            value,
            dim,
            op=MPI.LAND,
            all_values=True,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def _logical(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        op: MPI.Op,
        all_values: bool,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
    ) -> xr.Dataset | xr.DataArray:
        """Implement distributed logical reductions."""
        operation = "all" if all_values else "any"
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )
        if local_meta is not None:
            method = value.all if all_values else value.any
            local_result = method(dim=local_dim, keep_attrs=keep_attrs)
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation=operation)

        if isinstance(value, xr.DataArray):
            method = value.all if all_values else value.any
            local, local_error = self._guarded(
                lambda: method(dim=local_dim, keep_attrs=keep_attrs)
            )
            if not dims:
                if local_error is not None:
                    raise local_error
                return local
            result = self._comm_reduce(
                local,
                op,
                expect_dtype=_partial_dtype(value.dtype.str, operation, None),
                error=local_error,
                phase=f"MPI xarray {operation} reduction",
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            method = variable.all if all_values else variable.any
            local, local_error = self._guarded(
                lambda method=method, entry=entry: method(
                    dim=entry.dims, keep_attrs=keep_attrs
                )
            )
            if not entry.distributed:
                if local_error is not None:
                    raise local_error
                variables[entry.name] = local
                continue
            result = self._comm_reduce(
                local,
                op,
                expect_dtype=_partial_dtype(variable.dtype.str, operation, None),
                error=local_error,
                phase=f"MPI xarray {operation} reduction",
            )
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            auto_candidates=self._redistribution_candidates(plan),
            redistribute_on=redistribute_on,
        )
