"""Chunk calculations for saving and partitioning of NECDF4/HDF5 files"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import xarray as xr
from dask import array as dask_array

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

# A single HDF5 chunk's byte size must stay strictly under 2**32 bytes (4
# GiB) -- confirmed directly against this build with a binary search: a
# (16383, 256, 256) float32 chunk (3.9998 GiB) succeeds, (16384, 256, 256)
# (4.0000 GiB) raises "NetCDF: Bad chunk sizes." from the netCDF-C library.
# Half that hard limit is used as the working target below, leaving headroom
# for HDF5/filter (zlib, shuffle) bookkeeping overhead per chunk rather than
# skimming the exact boundary.
MAX_SAVE_CHUNK_BYTES = 2**31


# ---------------------------------------------------------------------------
# distribution_chunks: in-memory MPI partitioning
# ---------------------------------------------------------------------------


def get_native_chunk_sizes(data: xr.Dataset, dim: Hashable) -> int | None:
    """Return the common native on-disk chunk size for a dimension.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset to inspect for native chunk sizes.
    dim : hashable
        Dimension name to evaluate.

    Returns
    -------
    int or None
        Common chunk size for the dimension, or None if unavailable or inconsistent.
    """
    sizes: set[int] = set()
    for variable in data.data_vars.values():
        if dim not in variable.dims:
            continue
        chunksizes = variable.encoding.get("chunksizes")
        if chunksizes is not None:
            size = int(chunksizes[variable.get_axis_num(dim)])
        else:
            preferred = variable.encoding.get("preferred_chunks")
            if not isinstance(preferred, dict) or dim not in preferred:
                return None
            size = int(preferred[dim])
        if size <= 0:
            return None
        sizes.add(size)

    return sizes.pop() if len(sizes) == 1 else None


def get_usable_native_chunk(length: int, native_chunk: int | None) -> bool:
    """Return whether a native chunk provides a useful on-disk partition.

    Parameters
    ----------
    length : int
        Total length of the dimension.
    native_chunk : int or None
        Representative native chunk size.

    Returns
    -------
    bool
        True if the native chunk provides a valid multi-chunk partition.
    """
    if length <= 1 or native_chunk is None or native_chunk <= 1:
        return False
    return math.ceil(length / native_chunk) > 1


def get_effective_chunk_size(
    length: int, native_chunk: int | None, mpi_size: int
) -> int:
    """Return the distribution_chunk length climtools should retain for one dimension.

    Parameters
    ----------
    length : int
        Total dimension length.
    native_chunk : int or None
        Representative native chunk size.
    mpi_size : int
        Number of MPI ranks.

    Returns
    -------
    int
        Effective chunk size for the dimension.
    """
    if length <= 0:
        return 1

    if get_usable_native_chunk(length, native_chunk):
        return cast("int", native_chunk)

    return max(1, math.ceil(length / mpi_size))


def get_chunk_info(data: xr.Dataset, mpi_size: int) -> dict[str, int]:
    """Calculate effective distribution_chunk sizes for all Dataset dimensions.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset to evaluate.
    mpi_size : int
        Number of MPI ranks.

    Returns
    -------
    dict
        Mapping from dimension name to effective chunk size.
    """
    return {
        str(dim): get_effective_chunk_size(
            int(length), get_native_chunk_sizes(data, dim), mpi_size
        )
        for dim, length in data.sizes.items()
    }


def get_chunk_overrides(
    data: xr.Dataset, chunk_info: Mapping[str, int]
) -> dict[str, int]:
    """Return only distribution_chunk overrides that cannot use useful native chunks.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset to evaluate.
    chunk_info : mapping
        Effective chunk sizes for dimensions.

    Returns
    -------
    dict
        Mapping of dimensions requiring overrides to their chunk sizes.
    """
    return {
        str(dim): int(chunk_info[str(dim)])
        for dim, length in data.sizes.items()
        if not get_usable_native_chunk(int(length), get_native_chunk_sizes(data, dim))
    }


def get_balanced_bounds(length: int, rank: int, size: int) -> tuple[int, int]:
    """Split ``length`` into ``size`` contiguous, near-equal ``[start, stop)`` slabs.

    Parameters
    ----------
    length : int
        Total length to split.
    rank : int
        Current MPI rank.
    size : int
        Total number of MPI ranks.

    Returns
    -------
    tuple of int
        Start and stop indices for the given rank.
    """
    quotient, remainder = divmod(length, size)
    start = rank * quotient + min(rank, remainder)
    return start, start + quotient + int(rank < remainder)


def chunk_alignment_holds(length: int, chunk_size: int, size: int) -> bool:
    """Return whether rank bounds for this ``(length, chunk_size, size)`` fall on chunk edges.

    Parameters
    ----------
    length : int
        Global length of the dimension being partitioned.
    chunk_size : int
        Candidate distribution_chunk length.
    size : int
        Number of MPI ranks.

    Returns
    -------
    bool
        True if chunk-aligned bounds apply; False if falling back to balanced bounds.
    """
    if length <= 0:
        return True
    chunk_count = math.ceil(length / chunk_size)
    return chunk_count >= min(length, size)


def get_chunk_bounds(
    length: int, chunk_size: int, rank: int, size: int
) -> tuple[int, int]:
    """Partition a dimension into per-rank distribution_chunk bounds on chunk boundaries.

    Parameters
    ----------
    length : int
        Global length of the dimension.
    chunk_size : int
        Chunk size to align boundaries to.
    rank : int
        Current MPI rank.
    size : int
        Total number of MPI ranks.

    Returns
    -------
    tuple of int
        Start and stop indices for the given rank.
    """
    if length <= 0:
        return 0, 0

    if not chunk_alignment_holds(length, chunk_size, size):
        return get_balanced_bounds(length, rank, size)

    chunk_count = math.ceil(length / chunk_size)
    quotient, remainder = divmod(chunk_count, size)
    first_chunk = rank * quotient + min(rank, remainder)
    local_chunks = quotient + int(rank < remainder)
    start = min(first_chunk * chunk_size, length)
    stop = min((first_chunk + local_chunks) * chunk_size, length)
    return start, stop


def prune_chunk_info(
    chunk_info: Mapping[str, int], value: xr.Dataset | xr.DataArray
) -> dict[str, int]:
    """Restrict a distribution_chunk mapping to dimensions actually present on ``value``.

    Parameters
    ----------
    chunk_info : mapping
        Full mapping of chunk sizes.
    value : xarray.Dataset or xarray.DataArray
        Object whose dimensions are used as a filter.

    Returns
    -------
    dict
        Pruned mapping containing only dimensions present on the object.
    """
    return {
        str(dim): int(chunk_info[str(dim)])
        for dim in value.dims
        if str(dim) in chunk_info
    }


# ---------------------------------------------------------------------------
# save_chunks: on-disk NetCDF/Zarr output shape
# ---------------------------------------------------------------------------


def _other_dims_bytes(
    itemsize: int,
    dims: Iterable[Hashable],
    shape: Iterable[int],
    partition_dim: Hashable | None,
) -> int:
    """Return bytes contributed by one partition-dimension element.

    Parameters
    ----------
    itemsize : int
        Data type item size in bytes.
    dims : iterable of hashable
        Dimension names.
    shape : iterable of int
        Dimension sizes.
    partition_dim : hashable or None
        Partition dimension name to exclude.

    Returns
    -------
    int
        Byte size of a single slice along the non-partition dimensions.
    """
    return itemsize * math.prod(
        length for dim, length in zip(dims, shape, strict=True) if dim != partition_dim
    )


def _cap_partition_chunk_to_hdf5_limit(preferred: int, other_bytes: int) -> int:
    """Shrink a partition-dimension save_chunk length to fit the HDF5 4 GiB chunk limit.

    Parameters
    ----------
    preferred : int
        Partition-dimension chunk length before check.
    other_bytes : int
        Bytes contributed by one partition-dimension element.

    Returns
    -------
    int
        Adjusted chunk length complying with the HDF5 limit.
    """
    if other_bytes <= 0 or preferred * other_bytes <= MAX_SAVE_CHUNK_BYTES:
        return preferred
    return max(1, MAX_SAVE_CHUNK_BYTES // other_bytes)


def get_partition_chunk_size(
    ds: xr.Dataset, partition_dim: str | None, mpi_size: int
) -> int | None:
    """Return the per-rank-aligned HDF5 save_chunk length for ``partition_dim``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset being written.
    partition_dim : str or None
        Dimension that MPI ranks write disjoint slabs of.
    mpi_size : int
        Number of MPI ranks participating in the write.

    Returns
    -------
    int or None
        Optimized save chunk size for the partition dimension.
    """
    if partition_dim is None or partition_dim not in ds.sizes:
        return ds.sizes.get(partition_dim)

    length = int(ds.sizes[partition_dim])
    preferred = max(1, math.ceil(length / mpi_size))

    other_bytes = max(
        (
            _other_dims_bytes(
                variable.dtype.itemsize, variable.dims, variable.shape, partition_dim
            )
            for variable in ds.data_vars.values()
            if partition_dim in variable.dims
        ),
        default=1,
    )
    return _cap_partition_chunk_to_hdf5_limit(preferred, other_bytes)


def get_chunks(
    ds: xr.Dataset,
    chunks: Mapping[str, Iterable[int]] | None,
    partition_dim: str | None = None,
    partition_length: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Return explicit or existing save_chunk shapes for every variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset whose variable save_chunks are inspected.
    chunks : mapping, optional
        Explicit variable save_chunk shapes.
    partition_dim : str, optional
        Dimension that MPI ranks write disjoint slabs of.
    partition_length : int, optional
        Global size of ``partition_dim``.

    Returns
    -------
    dict
        Mapping from variable name to NetCDF save_chunk shape.
    """
    if chunks is not None:
        return {
            name: tuple(int(length) for length in shape)
            for name, shape in chunks.items()
        }

    output: dict[str, tuple[int, ...]] = {}
    for name, da in ds.variables.items():
        if da.ndim == 0 or any(length == 0 for length in da.shape):
            continue
        chunked = da if da.chunks is not None else da.chunk("auto")
        if chunked.chunks is None:
            shape = tuple(int(length) for length in da.shape)
        else:
            shape = tuple(max(chunked.chunksizes[dim]) for dim in da.dims)
        if (
            partition_dim is not None
            and partition_length is not None
            and partition_dim in da.dims
        ):
            axis = da.dims.index(partition_dim)
            shape = shape[:axis] + (int(partition_length),) + shape[axis + 1 :]
        output[name] = shape
    return output


def _largest_divisor_at_most(value: int, ceiling: int) -> int:
    """Return the largest divisor of ``value`` that is at most ``ceiling``.

    Parameters
    ----------
    value : int
        The number being divided.
    ceiling : int
        The largest acceptable result.

    Returns
    -------
    int
        Largest divisor meeting the ceiling constraint.
    """
    if value <= 0 or ceiling <= 0:
        return max(1, ceiling)
    if ceiling >= value:
        return value
    best = 1
    candidate = 1
    while candidate * candidate <= value:
        if value % candidate == 0:
            if candidate <= ceiling:
                best = max(best, candidate)
            paired = value // candidate
            if paired <= ceiling:
                best = max(best, paired)
        candidate += 1
    return best


def compute_save_chunks(
    value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any], mpi_size: int
) -> dict[str, tuple[int, ...]]:
    """Derive save chunks for a distributed object using global metadata.

    Calculates optimal save chunks for writing a distributed dataset or data array
    across MPI ranks. Because each rank holds only a local slice, this function
    reconstructs the global shape using ``meta`` and applies Dask's default
    ``"auto"`` chunking heuristic (~128 MiB target) independently on every rank
    without data communication.

    The partition dimension's chunk size is subsequently adjusted to ensure
    compatibility with HDF5 limits and MPI rank boundaries:

    - **Capped**: Limited by the HDF5 4 GiB chunk-size restriction.
    - **Snapped**: If distribution-chunk alignment holds, snapped down to the
      largest divisor of the distribution chunk size.
    - **Boundary-aligned**: If distribution-chunk alignment does not hold,
      snapped down to the largest divisor common to all balanced MPI rank
      boundaries.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Local slice of a distributed object on the current MPI rank.
    meta : mapping
        Distribution metadata returned by ``get_mpi_meta``. Must include
        ``meta["dim"]`` within ``meta["chunk_info"]``.
    mpi_size : int
        Number of MPI ranks the data is distributed across.

    Returns
    -------
    dict
        Mapping from variable name to save chunk tuple, identical across all ranks.

    Raises
    ------
    ValueError
        If ``meta["chunk_info"]`` lacks the partition dimension.
    """
    dim = str(meta["dim"])
    global_size = int(meta["global_size"])
    chunk_info = meta["chunk_info"]
    if dim not in chunk_info:
        raise ValueError(
            "mpi_meta['chunk_info'] does not include the partition "
            + f"dimension {dim!r}; cannot bound save_chunks against "
            + "distribution_chunks without it."
        )

    distribution_chunk = int(chunk_info[dim])
    aligned = chunk_alignment_holds(global_size, distribution_chunk, mpi_size)

    boundary_gcd = global_size
    if not aligned and mpi_size > 1:
        boundaries = [
            get_balanced_bounds(global_size, rank, mpi_size)[1]
            for rank in range(mpi_size - 1)
        ]
        if boundaries:
            boundary_gcd = math.gcd(*boundaries)

    if isinstance(value, xr.Dataset):
        variables = list(value.variables.items())
    else:
        variables = [
            (str(value.name) if value.name is not None else "__array__", value)
        ]

    output: dict[str, tuple[int, ...]] = {}
    for name, variable in variables:
        if variable.ndim == 0 or any(int(length) == 0 for length in variable.shape):
            continue

        shape = tuple(
            global_size if var_dim == dim else int(length)
            for var_dim, length in zip(variable.dims, variable.shape, strict=True)
        )
        mock = dask_array.zeros(shape, dtype=variable.dtype, chunks="auto")
        other_bytes = _other_dims_bytes(
            variable.dtype.itemsize, variable.dims, shape, dim
        )

        save_chunk: list[int] = []
        for var_dim, length, blocks in zip(
            variable.dims, shape, mock.chunks, strict=True
        ):
            proposed = int(max(blocks)) if blocks else int(length)
            if var_dim != dim:
                save_chunk.append(proposed)
                continue

            capped = _cap_partition_chunk_to_hdf5_limit(proposed, other_bytes)
            divisor_source = distribution_chunk if aligned else boundary_gcd
            save_chunk.append(_largest_divisor_at_most(divisor_source, capped))

        output[str(name)] = tuple(save_chunk)

    return output
