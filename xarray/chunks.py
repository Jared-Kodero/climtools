"""Calculate chunks for MPI partitioning and NetCDF4/HDF5 output."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from dask import array as dask_array

import xarray as xr

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




def get_native_chunk_sizes(data: xr.Dataset, dim: Hashable) -> int | None:
    """Return the common native-aligned boundary interval for a dimension.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset to inspect for native chunk sizes.
    dim : hashable
        Dimension name to evaluate.
    Returns
    -------
    int or None
        Smallest interval whose boundaries align with every available native chunk grid, or None if native chunking is unavailable.
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

    return math.lcm(*sizes) if sizes else None


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


def get_balanced_bounds(
    length: int, rank: int, size: int, min_chunk: int | None = None
) -> tuple[int, int]:
    """Split ``length`` into ``size`` contiguous, near-equal ``[start, stop)`` slabs.

    Parameters
    ----------
    length : int
        Total length to split.
    rank : int
        Current MPI rank.
    size : int
        Total number of MPI ranks.
    min_chunk : int or None, optional
        Guaranteed minimum local length for every rank that receives any
        data. When set, at most ``max(1, length // min_chunk)`` ranks
        (never more than ``size``) are given a non-empty slab; the
        remaining highest-numbered ranks each get an empty ``(length,
        length)`` slab, exactly as already happens here when ``length <
        size`` with no ``min_chunk`` set. This does not itself guarantee
        every downstream halo-based operation will fit -- it only
        guarantees the *partition*, not any later ``before``/``after``
        halo width a caller might request on top of it -- but choosing
        ``min_chunk`` at or above the widest halo/limit/window a
        distributed dimension will ever see (e.g. the largest
        ``rolling_reduce`` window or ``ffill``/``bfill`` limit planned
        for it) rules out ``halo_exchange``'s "local partition shorter
        than the requested halo" ``ValueError`` for that dimension
        entirely, rather than discovering it at call time on some rank
        count. See ``halo_exchange``'s own docstring for that error.
    Returns
    -------
    tuple of int
        Start and stop indices for the given rank.
    """
    if min_chunk is not None and min_chunk > 0 and size > 1 and length > 0:
        active = max(1, min(size, length // min_chunk))
        if active < size:
            if rank >= active:
                return length, length
            return get_balanced_bounds(length, rank, active)

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




def _other_dims_bytes(
    itemsize: int,
    dims: Iterable[Hashable],
    shape: Iterable[int],
    partition_dim: Hashable | None,
) -> int:
    """Return bytes contributed by one partition-dimension element."""
    return itemsize * math.prod(
        length for dim, length in zip(dims, shape, strict=True) if dim != partition_dim
    )


def _cap_partition_chunk_to_hdf5_limit(preferred: int, other_bytes: int) -> int:
    """Shrink a partition-dimension save_chunk length to fit the HDF5 4 GiB chunk limit."""
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


def _validate_explicit_chunk_bytes(
    ds: xr.Dataset, explicit: Mapping[str, tuple[int, ...]]
) -> None:
    """Reject a caller-supplied save_chunk shape that would exceed
    ``MAX_SAVE_CHUNK_BYTES``.

    The auto-inferred path below already keeps every save_chunk under this
    limit via ``_cap_partition_chunk_to_hdf5_limit`` (HDF5/netCDF-C hard-caps
    a single chunk at 4 GiB; the auto path deliberately targets half that,
    2 GiB, for headroom -- see the ``MAX_SAVE_CHUNK_BYTES`` comment above).
    An explicit ``chunks=`` mapping bypassed that entirely: nothing checked
    the byte size of a user-supplied shape before handing it to
    ``createVariable``/``set_collective(True)``, so a chunk shape as
    ordinary as "one chunk spanning every dimension's full length" (a
    completely natural thing to write for a variable the caller does not
    want sub-chunked) silently produced a multi-gigabyte single HDF5 chunk
    at production data sizes. Creation can still succeed under the 4 GiB
    hard limit while the actual parallel collective write to that chunk
    fails with an opaque ``RuntimeError: NetCDF: HDF error`` deep inside
    HDF5 -- exactly the failure this raises a clear, actionable error for
    instead, before any write is attempted.
    """
    offenders: list[str] = []
    for name, shape in explicit.items():
        if name not in ds.variables:
            continue
        itemsize = ds.variables[name].dtype.itemsize
        nbytes = itemsize * math.prod(shape)
        if nbytes > MAX_SAVE_CHUNK_BYTES:
            offenders.append(
                f"{name!r}: chunk shape {shape} * itemsize {itemsize} = "
                f"{nbytes / 2**30:.2f} GiB (limit {MAX_SAVE_CHUNK_BYTES / 2**30:.0f} GiB)"
            )
    if offenders:
        raise ValueError(
            "Explicit chunks= would create an HDF5 chunk larger than the "
            "safe per-chunk byte limit for parallel NetCDF-4 output "
            "(a chunk this large can pass variable creation and still fail "
            "the collective write with an opaque 'NetCDF: HDF error'). "
            "Pass a smaller chunk shape for: " + "; ".join(offenders)
        )


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
        explicit = {
            name: tuple(int(length) for length in shape)
            for name, shape in chunks.items()
        }
        _validate_explicit_chunk_bytes(ds, explicit)
        return explicit

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
    """Return the largest divisor of ``value`` that is at most ``ceiling``."""
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

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Local slice of a distributed object on the current MPI rank.
    meta : mapping
        Distribution metadata returned by ``get_mpi_meta``. May describe
        one or several active partition dimensions (a Cartesian
        partition); each is handled independently, using that axis's
        own division count from ``meta["cart"]["grid_shape"]`` (falling
        back to ``mpi_size`` when there is no Cartesian grid, i.e. the
        single-dimension case, which reproduces the prior behavior
        exactly).
    mpi_size : int
        Number of MPI ranks the data is distributed across.
    Returns
    -------
    dict
        Mapping from variable name to save chunk tuple, identical across all ranks.

    Raises
    ------
    ValueError
        If ``meta["chunk_info"]`` lacks a partition dimension.
    """
    dims = tuple(str(d) for d in meta["dims"])
    global_sizes = {str(d): int(sz) for d, sz in meta["global_sizes"].items()}
    chunk_info = meta["chunk_info"]
    missing = [d for d in dims if d not in chunk_info]
    if missing:
        raise ValueError(
            "mpi_meta['chunk_info'] does not include partition "
            + f"dimension(s) {missing!r}; cannot bound save_chunks "
            + "against distribution_chunks without it."
        )

    cart = meta.get("cart")
    divisor_source: dict[str, int] = {}
    for axis, d in enumerate(dims):
        global_size = global_sizes[d]
        distribution_chunk = int(chunk_info[d])
        divisions = int(cart["grid_shape"][axis]) if cart is not None else mpi_size
        aligned = chunk_alignment_holds(global_size, distribution_chunk, divisions)

        boundary_gcd = global_size
        if not aligned and divisions > 1:
            boundaries = [
                get_balanced_bounds(global_size, i, divisions)[1]
                for i in range(divisions - 1)
            ]
            if boundaries:
                boundary_gcd = math.gcd(*boundaries)

        divisor_source[d] = distribution_chunk if aligned else boundary_gcd

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

        var_dims = tuple(str(d) for d in variable.dims)
        shape = tuple(
            global_sizes[var_dim] if var_dim in dims else int(length)
            for var_dim, length in zip(var_dims, variable.shape, strict=True)
        )
        mock = dask_array.zeros(shape, dtype=variable.dtype, chunks="auto")

        save_chunk: list[int] = []
        for var_dim, length, blocks in zip(var_dims, shape, mock.chunks, strict=True):
            proposed = int(max(blocks)) if blocks else int(length)
            if var_dim not in dims:
                save_chunk.append(proposed)
                continue

            # A safe (if occasionally conservative) upper bound on the
            # bytes every other axis of this chunk could contribute:
            # non-partition axes at their own proposed chunk size, and
            # any *other* partition axis at its full global size (since
            # that axis's own cap, computed in this same loop, isn't
            # available yet to tighten this).
            other_bytes = variable.dtype.itemsize * math.prod(
                (
                    global_sizes[d]
                    if d in dims and d != var_dim
                    else int(blk_length)
                )
                for d, blk_length in zip(var_dims, shape, strict=True)
                if d != var_dim
            )
            capped = _cap_partition_chunk_to_hdf5_limit(proposed, other_bytes)
            save_chunk.append(_largest_divisor_at_most(divisor_source[var_dim], capped))

        output[str(name)] = tuple(save_chunk)

    return output
