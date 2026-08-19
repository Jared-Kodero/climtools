"""MPI-aware distributed xarray operations."""
# xarray_mpi.py

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Hashable, Iterable, Mapping
from numbers import Integral
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
import xarray as xr
from mpi4py import MPI as _MPI

if TYPE_CHECKING:
    from collections.abc import Callable

    from .lib_mpi import MPIRuntime

MPI_META = "mpi_meta"
_MPI_REDUCIBLE_KINDS = "biufc"

# Verify that every rank entered a reduction with the same per-variable plan
# before any buffer collective is posted. The check costs one small object
# allgather per reduction and converts an otherwise silent deadlock into an
# immediate exception. Set to False only for micro-benchmarking.
CHECK_COLLECTIVE_AGREEMENT = True


class _PlanEntry(NamedTuple):
    """One variable's rank-independent contribution to a reduction.

    Attributes
    ----------
    name : hashable
        Variable name.
    dims : tuple of hashable
        Reduced dimensions present on this variable.
    distributed : bool
        Whether the variable carries the active MPI partition dimension and
        therefore requires a cross-rank collective.
    dtype : numpy.dtype
        Variable dtype, preserved through the reduction without promotion.
    shape : tuple of tuple
        Global ``(dimension, length)`` pairs surviving the reduction.
    """

    name: Hashable
    dims: tuple[Hashable, ...]
    distributed: bool
    dtype: np.dtype[Any]
    shape: tuple[tuple[str, int], ...]


def get_mpi_meta(value: xr.Dataset | xr.DataArray) -> dict[str, Any] | None:
    """Return validated MPI distribution metadata.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object whose distribution metadata is requested.

    Returns
    -------
    dict or None
        Distribution metadata when valid, otherwise None.
    """
    meta = value.attrs.get(MPI_META)
    if not isinstance(meta, dict):
        return None

    required = {"dim", "global_size", "start", "stop", "chunk_info"}
    if not required <= meta.keys():
        return None

    dim = meta["dim"]
    if dim not in value.dims:
        return None

    start = int(meta["start"])
    stop = int(meta["stop"])
    global_size = int(meta["global_size"])
    if start < 0 or stop < start or stop > global_size:
        return None
    if int(value.sizes[dim]) != stop - start:
        return None

    chunk_info = meta["chunk_info"]
    if not isinstance(chunk_info, dict):
        return None

    return cast("dict[str, Any]", meta)


def set_mpi_meta(
    value: xr.Dataset | xr.DataArray,
    *,
    dim: Hashable,
    global_size: int,
    start: int,
    stop: int,
    chunk_info: Mapping[Hashable, int],
) -> None:
    """Attach MPI distribution metadata.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Rank-local xarray object.
    dim : hashable
        Distributed dimension.
    global_size : int
        Global length of ``dim``.
    start, stop : int
        Global half-open interval owned by this rank.
    chunk_info : mapping
        Effective climtools chunk size for every retained dimension.
    """
    meta = {
        "dim": str(dim),
        "global_size": int(global_size),
        "start": int(start),
        "stop": int(stop),
        "chunk_info": {
            str(name): int(size)
            for name, size in chunk_info.items()
            if name in value.dims and int(size) > 0
        },
    }
    value.attrs[MPI_META] = meta

    if isinstance(value, xr.Dataset):
        for variable in value.variables.values():
            variable.attrs.pop(MPI_META, None)
            if dim in variable.dims:
                variable.attrs[MPI_META] = meta.copy()


def strip_mpi_meta(value: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Return a shallow copy without MPI distribution metadata."""
    output = value.copy(deep=False)
    output.attrs.pop(MPI_META, None)
    if isinstance(output, xr.Dataset):
        for variable in output.variables.values():
            variable.attrs.pop(MPI_META, None)
    return output


def _native_chunk_size(data: xr.Dataset, dim: Hashable) -> int | None:
    """Return a representative native on-disk chunk size for a dimension."""
    candidates = [
        variable for variable in data.data_vars.values() if dim in variable.dims
    ]
    if not candidates:
        return None

    variable = max(candidates, key=lambda item: item.nbytes)
    chunksizes = variable.encoding.get("chunksizes")
    if chunksizes is not None:
        size = int(chunksizes[variable.get_axis_num(dim)])
        return size if size > 0 else None

    preferred = variable.encoding.get("preferred_chunks")
    if isinstance(preferred, Mapping) and dim in preferred:
        size = int(preferred[dim])
        return size if size > 0 else None

    return None


def _usable_native_chunk(length: int, native_chunk: int | None) -> bool:
    """Return whether a native chunk provides a useful on-disk partition."""
    if length <= 1 or native_chunk is None or native_chunk <= 1:
        return False
    return math.ceil(length / native_chunk) > 1


def get_effective_chunk_size(
    length: int,
    native_chunk: int | None,
    mpi_size: int,
) -> int:
    """Return the chunk size climtools should retain for one dimension."""
    if length <= 0:
        return 1

    if _usable_native_chunk(length, native_chunk):
        return cast("int", native_chunk)

    return max(1, math.ceil(length / mpi_size))


def get_chunk_info(data: xr.Dataset, mpi_size: int) -> dict[str, int]:
    """Calculate effective chunk sizes for all Dataset dimensions."""
    return {
        str(dim): get_effective_chunk_size(
            int(length),
            _native_chunk_size(data, dim),
            mpi_size,
        )
        for dim, length in data.sizes.items()
    }


def get_chunk_overrides(
    data: xr.Dataset,
    chunk_info: Mapping[str, int],
) -> dict[str, int]:
    """Return only chunk overrides that cannot use useful native chunks."""
    return {
        str(dim): int(chunk_info[str(dim)])
        for dim, length in data.sizes.items()
        if not _usable_native_chunk(
            int(length),
            _native_chunk_size(data, dim),
        )
    }


def _balanced_bounds(length: int, rank: int, size: int) -> tuple[int, int]:
    quotient, remainder = divmod(length, size)
    start = rank * quotient + min(rank, remainder)
    return start, start + quotient + int(rank < remainder)


def get_chunk_bounds(
    length: int,
    chunk_size: int,
    rank: int,
    size: int,
) -> tuple[int, int]:
    """Partition a dimension on effective chunk boundaries."""
    if length <= 0:
        return 0, 0

    chunk_count = math.ceil(length / chunk_size)
    if chunk_count < min(length, size):
        return _balanced_bounds(length, rank, size)

    quotient, remainder = divmod(chunk_count, size)
    first_chunk = rank * quotient + min(rank, remainder)
    local_chunks = quotient + int(rank < remainder)
    start = min(first_chunk * chunk_size, length)
    stop = min((first_chunk + local_chunks) * chunk_size, length)
    return start, stop


def prune_chunk_info(
    chunk_info: Mapping[str, int],
    value: xr.Dataset | xr.DataArray,
) -> dict[str, int]:
    return {
        str(dim): int(chunk_info[str(dim)])
        for dim in value.dims
        if str(dim) in chunk_info
    }


def indexer_is_scalar(indexer: Any) -> bool:
    return not isinstance(indexer, (slice, list, tuple, np.ndarray, xr.DataArray))


class XarrayMPI:
    """MPI-aware distributed xarray operations.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime owning the communicator used by this accessor.
    """

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime

    def open_dataset(
        self,
        filename_or_obj: Any,
        *,
        partition_dim: Hashable,
        chunks: Any = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open a Dataset lazily and distribute one dimension across ranks.

        Parameters
        ----------
        filename_or_obj : Any
            Input accepted by :func:`xarray.open_dataset`.
        partition_dim : hashable
            Dimension to distribute.
        chunks : Any, optional
            Explicit xarray/Dask chunk specification. If omitted, effective
            chunks are derived from usable native chunks, falling back to
            ``ceil(length / nranks)``.
        **kwargs : Any
            Additional arguments passed unchanged to
            :func:`xarray.open_dataset`.

        Returns
        -------
        xarray.Dataset
            Lazy rank-local Dataset carrying ``mpi_meta``.
        """

        use_mfdataset = (
            isinstance(filename_or_obj, str) and "*" in filename_or_obj
        ) or isinstance(filename_or_obj, (list, tuple))

        _open_dataset: Callable = (
            xr.open_mfdataset if use_mfdataset else xr.open_dataset
        )

        with _open_dataset(filename_or_obj, chunks=None, **kwargs) as metadata:
            if partition_dim not in metadata.dims:
                raise ValueError(
                    f"partition_dim {partition_dim!r} is not in "
                    + f"{list(metadata.dims)!r}."
                )
            chunk_info = get_chunk_info(metadata, self._runtime.comm.size)
            open_chunk_overrides = get_chunk_overrides(metadata, chunk_info)
            global_size = int(metadata.sizes[partition_dim])
            longest_size = max(int(length) for length in metadata.sizes.values())
            if self._runtime.comm.rank == 0 and global_size < longest_size:
                longest_dims = [
                    str(dim)
                    for dim, length in metadata.sizes.items()
                    if int(length) == longest_size
                ]
                warnings.warn(
                    f"partition_dim {partition_dim!r} has length {global_size}, "
                    + "but it should be a longest dataset dimension. "
                    + f"Longest dimension(s) {longest_dims!r} have length "
                    + f"{longest_size}.",
                    UserWarning,
                    stacklevel=2,
                )

        partition_chunk = chunk_info[str(partition_dim)]
        start, stop = get_chunk_bounds(
            global_size,
            partition_chunk,
            self._runtime.comm.rank,
            self._runtime.comm.size,
        )

        open_chunks = chunks
        if open_chunks is None:
            open_chunks = open_chunk_overrides

        data = _open_dataset(
            filename_or_obj,
            chunks=open_chunks,
            **kwargs,
        )
        data = data.isel({partition_dim: slice(start, stop)})
        set_mpi_meta(
            data,
            dim=partition_dim,
            global_size=global_size,
            start=start,
            stop=stop,
            chunk_info=chunk_info,
        )
        return data

    def redistribute(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: Hashable | Literal["auto"],
        *,
        chunk_info: Mapping[str, int] | None = None,
    ) -> xr.Dataset | xr.DataArray:
        """Distribute a replicated xarray object across ranks.

        Parameters
        ----------
        value : xarray.Dataset or xarray.DataArray
            Complete object present on every rank.
        dim : hashable or {"auto"}
            New partition dimension. ``"auto"`` chooses the largest remaining
            dimension.
        chunk_info : mapping, optional
            Effective chunk information to preserve from a prior distribution.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Rank-local distributed object.

        Raises
        ------
        ValueError
            If the input is already distributed or the requested dimension
            does not exist.
        """
        if get_mpi_meta(value) is not None:
            raise ValueError(
                "Cannot redistribute an already distributed object. "
                + "Reduce or gather its distributed dimension first."
            )

        if dim == "auto":
            if not value.dims:
                return strip_mpi_meta(value)
            dim = max(value.sizes, key=value.sizes.__getitem__)

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
            length,
            chunk_size,
            self._runtime.comm.size,
        )
        info[str(dim)] = chunk_size

        start, stop = get_chunk_bounds(
            length,
            chunk_size,
            self._runtime.comm.rank,
            self._runtime.comm.size,
        )
        output = strip_mpi_meta(value).isel({dim: slice(start, stop)})
        info = prune_chunk_info(info, output)
        for other_dim, other_length in output.sizes.items():
            info.setdefault(
                str(other_dim),
                get_effective_chunk_size(
                    int(other_length),
                    None,
                    self._runtime.comm.size,
                ),
            )

        set_mpi_meta(
            output,
            dim=dim,
            global_size=length,
            start=start,
            stop=stop,
            chunk_info=info,
        )
        return output

    def isel(
        self,
        value: xr.Dataset | xr.DataArray,
        indexers: Mapping[Any, Any] | None = None,
        **indexers_kwargs: Any,
    ) -> xr.Dataset | xr.DataArray:
        """Index a distributed object using global integer coordinates.

        Slice indexers on the distributed dimension are interpreted against the
        global dimension. Other dimensions use ordinary xarray ``isel``.
        Scalar indexing of the distributed dimension returns a replicated
        result on every rank.
        """
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
            "xr.Dataset | xr.DataArray",
            self._runtime.comm.bcast(result, root=owner),
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
        """Index a distributed object using global coordinate semantics.

        Slice selection on the distributed coordinate is evaluated locally on
        every rank, followed only by an all-gather of local result lengths.
        Scalar selection broadcasts the selected result from its owning rank.
        """
        supplied = dict(indexers or {})
        supplied.update(indexers_kwargs)
        meta = get_mpi_meta(value)
        if meta is None:
            return value.sel(
                supplied,
                method=method,
                tolerance=tolerance,
                drop=drop,
            )

        dim = meta["dim"]
        if dim not in supplied:
            return value.sel(
                supplied,
                method=method,
                tolerance=tolerance,
                drop=drop,
            )

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
            local_indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
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
            selected = locator.sel(
                {dim: label},
                method=method,
                tolerance=tolerance,
            )
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
                    result = strip_mpi_meta(value).isel(
                        {dim: local_index},
                        drop=drop,
                    )
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
                self._runtime.comm.bcast(result, root=owner),
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
            "xr.Dataset | xr.DataArray", self._runtime.comm.bcast(payload, root=owner)
        )

    # -- collective planning -------------------------------------------------

    def _validate_collective(
        self,
        mode: Literal["all", "root"],
        root: int,
    ) -> None:
        if mode not in ("all", "root"):
            raise ValueError("mode must be either 'all' or 'root'.")
        if mode == "root":
            if isinstance(root, bool) or not isinstance(root, Integral) or root < 0:
                raise ValueError("root must be a non-negative integer rank.")
            if root >= self._runtime.comm.size:
                raise ValueError(
                    f"root {root} is outside [0, {self._runtime.comm.size})."
                )

    @staticmethod
    def _normalize_dim(
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
    ) -> tuple[Any, tuple[Hashable, ...]]:
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
    def _variable_dims(
        value: xr.DataArray,
        dims: tuple[Hashable, ...],
    ) -> tuple[Hashable, ...]:
        return tuple(dim for dim in dims if dim in value.dims)

    @staticmethod
    def _variable_is_distributed(
        value: xr.DataArray,
        meta: Mapping[str, Any] | None,
    ) -> bool:
        """Return whether a variable contains the active partition dimension."""
        return meta is not None and meta["dim"] in value.dims

    @staticmethod
    def _skipna_enabled(dtype: np.dtype[Any], skipna: bool | None) -> bool:
        if skipna is not None:
            return skipna
        return dtype.kind in "fc"

    @staticmethod
    def _mean_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
        return np.asarray(np.mean(np.zeros(1, dtype=dtype))).dtype

    @staticmethod
    def _check_reducible(dtype: np.dtype[Any], operation: str) -> None:
        """Reject dtypes with no meaningful MPI reduction for an operation.

        The check uses only the declared dtype, which is identical on every
        rank, so an unsupported variable raises on all ranks before any
        collective is posted rather than on the subset of ranks that happen
        to reach the buffer collective first.
        """
        if operation in ("any", "all"):
            return
        if dtype.kind not in _MPI_REDUCIBLE_KINDS:
            raise TypeError(f"Unsupported MPI xarray dtype: {dtype}.")
        if operation in ("min", "max") and dtype.kind == "c":
            name = "minimum" if operation == "min" else "maximum"
            raise TypeError(f"MPI {name} is not defined for complex xarray data.")

    def _validate_distribution(
        self,
        value: xr.Dataset | xr.DataArray,
        dims: tuple[Hashable, ...],
    ) -> dict[str, Any] | None:
        meta = get_mpi_meta(value)
        if meta is not None and meta["dim"] not in dims:
            raise ValueError(
                "Distributed dimension "
                + f"{meta['dim']!r} must be included in the MPI reduction. "
                + "Use the ordinary xarray reduction for other dimensions."
            )
        return meta

    def _agree(self, signature: tuple[Any, ...]) -> None:
        """Verify that every rank entered the same reduction plan.

        The plan is derived only from metadata that is identical on every
        rank, so a disagreement is a programming error that would otherwise
        block forever inside the following buffer collectives. One small
        object allgather turns that deadlock into an immediate, diagnosable
        exception on every rank.
        """
        if not CHECK_COLLECTIVE_AGREEMENT or self._runtime.comm.size == 1:
            return
        digest = hashlib.blake2b(repr(signature).encode(), digest_size=16).digest()
        digests = self._runtime.comm.allgather(digest)
        if len(set(digests)) == 1:
            return
        disagreeing = [
            rank for rank, value in enumerate(digests) if value != digests[0]
        ]
        raise self._runtime.MPIError(
            "MPI ranks entered different xarray reduction plans. Ranks "
            + f"{disagreeing} disagree with rank 0, which would deadlock the "
            + "following collective reduction."
        )

    def _plan(
        self,
        value: xr.Dataset | xr.DataArray,
        dims: tuple[Hashable, ...],
        meta: Mapping[str, Any] | None,
        *,
        operation: str,
        mode: Literal["all", "root"],
        root: int,
    ) -> tuple[_PlanEntry, ...]:
        """Return the rank-independent per-variable reduction plan.

        Every field is taken from names, dims, dtypes, and global sizes, all
        of which are identical on every rank for a partitioned object. The
        plan therefore fixes the number and shape of the collectives before
        any rank-local data is touched, which is what keeps the collective
        sequence identical on ranks holding an empty partition.
        """
        if isinstance(value, xr.DataArray):
            items: tuple[tuple[Hashable, xr.DataArray], ...] = ((value.name, value),)
        else:
            items = tuple((name, value[name]) for name in value.data_vars)

        entries = []
        for name, variable in items:
            variable_dims = self._variable_dims(variable, dims)
            if variable_dims:
                self._check_reducible(variable.dtype, operation)
            entries.append(
                _PlanEntry(
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
                mode,
                root,
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

    def _partition_is_empty(self, value: xr.Dataset | xr.DataArray, meta: Any) -> bool:
        """Return whether this rank owns no elements of the partition."""
        if meta is None:
            return False
        dim = meta["dim"]
        return dim in value.dims and int(value.sizes[dim]) == 0

    # -- collective primitives -----------------------------------------------

    def _comm_reduce(
        self,
        value: xr.DataArray,
        op: _MPI.Op,
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        send: np.ndarray[Any, Any] | None = None
        error: BaseException | None = None
        try:
            send = np.asarray(value.values)
            if not send.flags.c_contiguous:
                send = np.ascontiguousarray(send)
            if send.dtype.kind not in _MPI_REDUCIBLE_KINDS:
                raise TypeError(f"Unsupported MPI xarray dtype: {send.dtype}.")
        except BaseException as exc:
            error = exc

        # Materializing a lazy rank-local result can fail on only one rank.
        # Synchronize that failure before any rank posts Allreduce/Reduce, or
        # the remaining ranks can block forever in the buffer collective.
        self._runtime.raise_if_error(error, "MPI xarray reduction buffer preparation")
        if send is None:
            raise AssertionError("MPI xarray reduction buffer is missing.")

        # Allocate the receive buffer with the send buffer's own dtype and a
        # C-contiguous layout. np.empty_like would inherit the source memory
        # order, which can differ from the contiguous copy actually sent.
        if mode == "all":
            recv = np.empty(send.shape, dtype=send.dtype)
            self._runtime.comm.Allreduce(send, recv, op=op)
        else:
            recv = (
                np.empty(send.shape, dtype=send.dtype)
                if self._runtime.comm.rank == root
                else None
            )
            self._runtime.comm.Reduce(send, recv, op=op, root=root)
            if recv is None:
                return None

        return value.copy(data=recv)

    def _count(
        self,
        value: xr.DataArray,
        dims: tuple[Hashable, ...],
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        return self._comm_reduce(
            value.count(dim=dims, keep_attrs=False),
            _MPI.SUM,
            mode=mode,
            root=root,
        )

    def _local_result(
        self,
        value: xr.Dataset | xr.DataArray,
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.Dataset | xr.DataArray | None:
        """Return a local result according to collective result placement."""
        if mode == "root" and self._runtime.comm.rank != root:
            return None
        return value

    @staticmethod
    def _reduced_dataset(
        value: xr.Dataset,
        dims: tuple[Hashable, ...],
        variables: Mapping[Hashable, xr.DataArray],
    ) -> xr.Dataset:
        """Assemble a reduced Dataset from per-variable results.

        The Dataset is built from the plan rather than from a whole-Dataset
        local reduction, because different xarray reductions retain different
        variables. Rebuilding from the plan keeps the variable set identical
        on every rank.
        """
        reduced = set(dims)
        coords = {
            name: coord
            for name, coord in value.coords.items()
            if not reduced & set(coord.dims)
        }
        return xr.Dataset(dict(variables), coords=coords, attrs=dict(value.attrs))

    def _dataset_result(
        self,
        value: xr.Dataset,
        dims: tuple[Hashable, ...],
        variables: Mapping[Hashable, xr.DataArray],
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.Dataset | None:
        if mode == "root" and self._runtime.comm.rank != root:
            return None
        return self._reduced_dataset(value, dims, variables)

    def _finish(
        self,
        result: xr.Dataset | xr.DataArray | None,
        *,
        old_meta: Mapping[str, Any] | None,
        redistribute_on: Hashable | Literal["auto"] | None,
        mode: Literal["all", "root"],
    ) -> xr.Dataset | xr.DataArray | None:
        if redistribute_on is not None and mode != "all":
            raise ValueError("redistribute_on requires mode='all'.")
        if result is None:
            return None

        result = strip_mpi_meta(result)
        if redistribute_on is None:
            return result

        chunk_info = (
            prune_chunk_info(old_meta["chunk_info"], result)
            if old_meta is not None
            else {}
        )
        return self.redistribute(
            result,
            redistribute_on,
            chunk_info=chunk_info,
        )

    # -- per-variable combination --------------------------------------------

    def _combine_sum_or_prod(
        self,
        value: xr.DataArray,
        partial: xr.DataArray,
        dims: tuple[Hashable, ...],
        op: _MPI.Op,
        *,
        skipna: bool | None,
        min_count: int | None,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        result = self._comm_reduce(partial, op, mode=mode, root=root)
        global_count = None
        if min_count is not None and self._skipna_enabled(value.dtype, skipna):
            global_count = self._count(value, dims, mode=mode, root=root)
        if result is None:
            return None
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
        partial_sum: xr.DataArray,
        dims: tuple[Hashable, ...],
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        global_sum = self._comm_reduce(partial_sum, _MPI.SUM, mode=mode, root=root)
        global_count = self._count(value, dims, mode=mode, root=root)
        if global_sum is None or global_count is None:
            return None

        # Divide in the dtype numpy.mean would produce for this input. Dividing
        # the float32 sum by the int64 count directly would promote the whole
        # array to float64 and then cast it back, costing two full-width
        # temporaries for a result that is float32 either way.
        target = self._mean_dtype(value.dtype)
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
        """Return the reduction identity for a rank owning no elements."""
        kind = value.dtype.kind
        if kind == "b":
            identity: Any = bool(minimum)
        elif kind in "iu":
            limits = np.iinfo(value.dtype)
            identity = limits.max if minimum else limits.min
        elif kind == "f":
            identity = np.asarray(
                np.inf if minimum else -np.inf,
                dtype=value.dtype,
            ).item()
        else:
            name = "minimum" if minimum else "maximum"
            raise TypeError(f"MPI {name} is not defined for {value.dtype} data.")

        template = value.sum(
            dim=dims,
            skipna=False,
            keep_attrs=keep_attrs,
        )
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
        """Return this rank's local extreme, including for an empty partition."""
        if empty:
            return self._empty_extreme_partial(
                variable,
                variable_dims,
                minimum=minimum,
                keep_attrs=keep_attrs,
            )
        method = variable.min if minimum else variable.max
        return method(dim=variable_dims, skipna=skipna, keep_attrs=keep_attrs)

    def _combine_extreme(
        self,
        value: xr.DataArray,
        partial: xr.DataArray,
        dims: tuple[Hashable, ...],
        *,
        minimum: bool,
        skipna: bool | None,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        kind = partial.dtype.kind
        if kind == "b":
            return self._comm_reduce(
                partial,
                _MPI.LAND if minimum else _MPI.LOR,
                mode=mode,
                root=root,
            )

        op = _MPI.MIN if minimum else _MPI.MAX
        if kind != "f":
            return self._comm_reduce(partial, op, mode=mode, root=root)

        identity = np.asarray(
            np.inf if minimum else -np.inf,
            dtype=partial.dtype,
        ).item()
        if self._skipna_enabled(value.dtype, skipna):
            local_mask = value.count(dim=dims, keep_attrs=False) > 0
            safe_partial = partial.where(local_mask, other=identity)
        else:
            local_mask = value.isnull().any(dim=dims, keep_attrs=False)
            safe_partial = partial.where(~local_mask, other=identity)
        if safe_partial.dtype != partial.dtype:
            safe_partial = safe_partial.astype(partial.dtype, keep_attrs=True)

        result = self._comm_reduce(safe_partial, op, mode=mode, root=root)
        global_mask = self._comm_reduce(
            local_mask,
            _MPI.LOR,
            mode=mode,
            root=root,
        )
        if result is None or global_mask is None:
            return None
        masked = (
            result.where(global_mask)
            if self._skipna_enabled(value.dtype, skipna)
            else result.where(~global_mask)
        )
        if masked.dtype != result.dtype:
            masked = masked.astype(result.dtype, keep_attrs=True)
        return masked

    # -- public reductions ---------------------------------------------------

    def sum(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.Dataset | xr.DataArray | None:
        """Reduce an xarray object by distributed summation."""
        return self._sum_prod(
            value,
            dim,
            op=_MPI.SUM,
            product=False,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            mode=mode,
            root=root,
        )

    def prod(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.Dataset | xr.DataArray | None:
        """Reduce an xarray object by distributed multiplication."""
        return self._sum_prod(
            value,
            dim,
            op=_MPI.PROD,
            product=True,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            mode=mode,
            root=root,
        )

    def _sum_prod(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        op: _MPI.Op,
        product: bool,
        skipna: bool | None,
        min_count: int | None,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.Dataset | xr.DataArray | None:
        operation = "prod" if product else "sum"
        self._validate_collective(mode, root)
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = self._validate_distribution(value, dims)
        plan = self._plan(
            value,
            dims,
            old_meta,
            operation=operation,
            mode=mode,
            root=root,
        )

        if isinstance(value, xr.DataArray):
            method = value.prod if product else value.sum
            local = method(
                dim=local_dim,
                skipna=skipna,
                min_count=None,
                keep_attrs=keep_attrs,
            )
            if not dims:
                return self._local_result(local, mode=mode, root=root)
            result = self._combine_sum_or_prod(
                value,
                local,
                dims,
                op,
                skipna=skipna,
                min_count=min_count,
                mode=mode,
                root=root,
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                mode=mode,
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            method = variable.prod if product else variable.sum
            local = method(
                dim=entry.dims,
                skipna=skipna,
                min_count=None,
                keep_attrs=keep_attrs,
            )
            if not entry.distributed:
                variables[entry.name] = local
                continue
            result = self._combine_sum_or_prod(
                variable,
                local,
                entry.dims,
                op,
                skipna=skipna,
                min_count=min_count,
                mode=mode,
                root=root,
            )
            if result is not None:
                variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables, mode=mode, root=root),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            mode=mode,
        )

    def mean(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.Dataset | xr.DataArray | None:
        """Reduce an xarray object by distributed arithmetic mean."""
        self._validate_collective(mode, root)
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = self._validate_distribution(value, dims)
        plan = self._plan(
            value,
            dims,
            old_meta,
            operation="mean",
            mode=mode,
            root=root,
        )

        if isinstance(value, xr.DataArray):
            if not dims:
                local_mean = value.mean(
                    dim=local_dim,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                )
                return self._local_result(local_mean, mode=mode, root=root)
            local_sum = value.sum(
                dim=local_dim,
                skipna=skipna,
                min_count=None,
                keep_attrs=keep_attrs,
            )
            result = self._combine_mean(
                value,
                local_sum,
                dims,
                mode=mode,
                root=root,
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                mode=mode,
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            if not entry.distributed:
                variables[entry.name] = variable.mean(
                    dim=entry.dims,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                )
                continue
            local_sum = variable.sum(
                dim=entry.dims,
                skipna=skipna,
                min_count=None,
                keep_attrs=keep_attrs,
            )
            result = self._combine_mean(
                variable,
                local_sum,
                entry.dims,
                mode=mode,
                root=root,
            )
            if result is not None:
                variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables, mode=mode, root=root),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            mode=mode,
        )

    def min(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.Dataset | xr.DataArray | None:
        """Reduce an xarray object by distributed minimum."""
        return self._extreme(
            value,
            dim,
            minimum=True,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            mode=mode,
            root=root,
        )

    def max(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.Dataset | xr.DataArray | None:
        """Reduce an xarray object by distributed maximum."""
        return self._extreme(
            value,
            dim,
            minimum=False,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            mode=mode,
            root=root,
        )

    def _extreme(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        minimum: bool,
        skipna: bool | None,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.Dataset | xr.DataArray | None:
        operation = "min" if minimum else "max"
        self._validate_collective(mode, root)
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = self._validate_distribution(value, dims)
        plan = self._plan(
            value,
            dims,
            old_meta,
            operation=operation,
            mode=mode,
            root=root,
        )
        empty_partition = self._partition_is_empty(value, old_meta)

        if isinstance(value, xr.DataArray):
            if not dims:
                method = value.min if minimum else value.max
                return self._local_result(
                    method(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs),
                    mode=mode,
                    root=root,
                )
            local = self._local_extreme(
                value,
                dims,
                empty=empty_partition,
                minimum=minimum,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
            result = self._combine_extreme(
                value,
                local,
                dims,
                minimum=minimum,
                skipna=skipna,
                mode=mode,
                root=root,
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                mode=mode,
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            local = self._local_extreme(
                variable,
                entry.dims,
                empty=empty_partition and entry.distributed,
                minimum=minimum,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
            if not entry.distributed:
                variables[entry.name] = local
                continue
            result = self._combine_extreme(
                variable,
                local,
                entry.dims,
                minimum=minimum,
                skipna=skipna,
                mode=mode,
                root=root,
            )
            if result is not None:
                variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables, mode=mode, root=root),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            mode=mode,
        )

    def any(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.Dataset | xr.DataArray | None:
        """Reduce an xarray object by distributed logical OR."""
        return self._logical(
            value,
            dim,
            op=_MPI.LOR,
            all_values=False,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            mode=mode,
            root=root,
        )

    def all(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.Dataset | xr.DataArray | None:
        """Reduce an xarray object by distributed logical AND."""
        return self._logical(
            value,
            dim,
            op=_MPI.LAND,
            all_values=True,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            mode=mode,
            root=root,
        )

    def _logical(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        op: _MPI.Op,
        all_values: bool,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.Dataset | xr.DataArray | None:
        operation = "all" if all_values else "any"
        self._validate_collective(mode, root)
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = self._validate_distribution(value, dims)
        plan = self._plan(
            value,
            dims,
            old_meta,
            operation=operation,
            mode=mode,
            root=root,
        )

        if isinstance(value, xr.DataArray):
            method = value.all if all_values else value.any
            local = method(dim=local_dim, keep_attrs=keep_attrs)
            if not dims:
                return self._local_result(local, mode=mode, root=root)
            result = self._comm_reduce(local, op, mode=mode, root=root)
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                mode=mode,
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            method = variable.all if all_values else variable.any
            local = method(dim=entry.dims, keep_attrs=keep_attrs)
            if not entry.distributed:
                variables[entry.name] = local
                continue
            result = self._comm_reduce(local, op, mode=mode, root=root)
            if result is not None:
                variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables, mode=mode, root=root),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            mode=mode,
        )
