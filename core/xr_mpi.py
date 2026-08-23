"""MPI-aware distributed xarray operations."""
# xarray_mpi.py

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from functools import cache
from numbers import Integral
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
import xarray as xr
from mpi4py import MPI as _MPI
from mpi4py.util import dtlib as _dtlib

from .xr_meta import (
    get_mpi_meta,
    log_partition_report,
    set_mpi_meta,
    strip_mpi_meta,
)
from .xr_ops import ArithmeticMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from .lib_mpi import MPIRuntime

_OP_LIST: tuple[tuple[Any, str], ...] = (
    (_MPI.SUM, "SUM"),
    (_MPI.PROD, "PROD"),
    (_MPI.MIN, "MIN"),
    (_MPI.MAX, "MAX"),
    (_MPI.LAND, "LAND"),
    (_MPI.LOR, "LOR"),
)


def _op_name(op: _MPI.Op) -> str:
    """Return a picklable, rank-stable label for a reduction operation.

    mpi4py Op handles are unhashable and their repr embeds an address that
    differs between ranks, so neither can be compared across ranks. The
    label can be.
    """
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
    """Return whether a NumPy dtype has a usable predefined MPI datatype.

    Membership in mpi4py's type dictionary is not sufficient. float16 maps to
    MPI_SHORT_FLOAT, which most implementations do not provide, so the handle
    exists but every use of it fails with MPI_ERR_TYPE. Querying its size is
    the cheapest way to find out whether the running MPI actually supports
    it, and the answer depends only on the dtype, so it is identical on every
    rank and safe to decide locally.
    """
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
    dtype_string: str,
    operation: str,
    skipna: bool | None,
) -> np.dtype[Any]:
    """Return the dtype xarray produces for one rank's partial reduction.

    The reduction dtype is probed on a zero-size array of the requested
    dtype rather than predicted from NumPy promotion rules, so it always
    matches what the real reduction returns for the installed xarray and
    NumPy versions. It depends only on the dtype, the operation and
    ``skipna``, all of which are identical on every rank, so every rank
    derives the same answer. Casting each rank's partial to this dtype
    before the buffer collective is what guarantees that ranks holding an
    empty partition post the same datatype as ranks holding data.

    Parameters
    ----------
    dtype_string : str
        NumPy dtype string of the variable being reduced.
    operation : {"sum", "prod", "min", "max", "count", "any", "all"}
        Reduction whose rank-local partial dtype is requested.
    skipna : bool or None
        Skip-NaN behaviour requested for the reduction.

    Returns
    -------
    numpy.dtype
        Dtype of the rank-local partial for this reduction.
    """
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


def get_native_chunk_sizes(data: xr.Dataset, dim: Hashable) -> int | None:
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


def get_usable_native_chunk(length: int, native_chunk: int | None) -> bool:
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

    if get_usable_native_chunk(length, native_chunk):
        return cast("int", native_chunk)

    return max(1, math.ceil(length / mpi_size))


def get_chunk_info(data: xr.Dataset, mpi_size: int) -> dict[str, int]:
    """Calculate effective chunk sizes for all Dataset dimensions."""
    return {
        str(dim): get_effective_chunk_size(
            int(length),
            get_native_chunk_sizes(data, dim),
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
        if not get_usable_native_chunk(
            int(length),
            get_native_chunk_sizes(data, dim),
        )
    }


def get_balanced_bounds(length: int, rank: int, size: int) -> tuple[int, int]:
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
        return get_balanced_bounds(length, rank, size)

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


def _coord_length(spec: Any) -> int | None:
    """Return a coordinate spec's own length, or None if it has none.

    Accepts the same forms as :func:`_localize_coord`: a bare array-like or
    a ``(dims, array[, attrs])`` tuple. A 0-D (scalar) coordinate has no
    length to offer.
    """
    array = spec[1] if isinstance(spec, tuple) else spec
    array = np.asarray(array)
    return int(array.shape[0]) if array.ndim > 0 else None


def _resolve_sizes(
    required_dims: Iterable[Hashable],
    sizes: Mapping[Hashable, int] | None,
    coords: Mapping[Hashable, Any] | None,
) -> dict[Hashable, int]:
    """Fill in any dimension length missing from ``sizes`` using ``coords``.

    A dimension's length can come from either an explicit entry in
    ``sizes`` (checked first, so it always wins) or the length of a
    matching full-length coordinate in ``coords`` -- reading that length
    never forces any computation, since coordinates passed to
    :meth:`XarrayMPI.create_dataarray`/``create_dataset`` are always plain,
    eager arrays, never lazy ``fill`` functions. Any dimension with
    neither is reported together, in one error, rather than one at a time.
    """
    resolved = dict(sizes) if sizes else {}
    coords = coords or {}
    missing = []
    for dim_name in required_dims:
        if dim_name in resolved:
            continue
        length = _coord_length(coords[dim_name]) if dim_name in coords else None
        if length is None:
            missing.append(dim_name)
        else:
            resolved[dim_name] = length
    if missing:
        raise ValueError(
            f"Cannot determine the length of {sorted(str(d) for d in missing)}: "
            + "not given explicitly and no matching full-length coordinate "
            + "was passed. Pass its length explicitly, or include a "
            + "full-length coordinate array for it."
        )
    return resolved


def _localize_coord(
    spec: Any,
    global_size: int,
    start: int,
    stop: int,
) -> Any:
    """Slice a coordinate spec to ``[start:stop)`` if it is full-length.

    Accepts a coordinate in any of the three forms
    :class:`xarray.DataArray`/:class:`~xarray.Dataset` themselves accept: a
    bare array-like, ``(dims, array)``, or ``(dims, array, attrs)``. A
    coordinate whose own length does not equal ``global_size`` is returned
    unchanged (already rank-local, or a scalar/unrelated-length auxiliary
    coordinate -- not this function's business to guess about).
    """
    if isinstance(spec, tuple):
        coord_dims, coord_array, *rest = spec
    else:
        coord_dims, coord_array, rest = None, spec, []
    coord_array = np.asarray(coord_array)
    if coord_array.shape and coord_array.shape[0] == global_size:
        coord_array = coord_array[start:stop]
    if coord_dims is None:
        return coord_array
    return (coord_dims, coord_array, *rest)


def _delayed_local(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    shape: tuple[int, ...],
    dtype: Any,
) -> Any:
    """Wrap ``fn(*args)`` as one rank's own slice, not yet computed.

    Shared by :meth:`XarrayMPI.create_dataarray` and
    :meth:`XarrayMPI.create_dataset`: every call site passes a *different*
    ``fn``/``args`` (this rank's own ``fill`` and its own bounds, or none
    for a non-partitioned variable), computed independently with no
    communication -- this helper only avoids repeating the
    ``dask.delayed``/``from_delayed`` wiring three times.
    """
    import dask
    import dask.array as dask_array

    return dask_array.from_delayed(dask.delayed(fn)(*args), shape=shape, dtype=dtype)


_SHORT_PARTITION_WARNED: set[tuple[str, int, int]] = set()
"""Distinct (dim, length, mpi_size) triples already warned about this process.

Populated by :func:`choose_partition_dim`. Not meant to be read or mutated
directly; exists at module scope only so the warning survives across many
independent calls within one process without needing to thread state through
every caller.
"""


def choose_partition_dim(
    sizes: Mapping[Hashable, int],
    mpi_size: int,
    *,
    exclude: Iterable[Hashable] = (),
    rank: int | None = None,
) -> Hashable:
    """Select a partition dimension automatically.

    The dimension that keeps the most ranks busy is the longest one, so the
    primary key is length. Ties are broken by dataset declaration order, which
    is identical on every rank, so the choice is rank-invariant without any
    communication. Dimensions of length one are never chosen unless nothing
    else exists, because partitioning them leaves every rank but one empty.

    Parameters
    ----------
    sizes : mapping
        Dimension name to global length.
    mpi_size : int
        Number of ranks the data will be spread over.
    exclude : iterable of hashable, optional
        Dimensions that must not be chosen, for example a dimension the caller
        intends to reduce over.
    rank : int, optional
        Calling rank, used only to gate the short-partition warning below to
        rank 0. The dimension this function returns never depends on it: the
        choice is identical on every rank regardless. None (the default)
        always warns, which is correct both for a caller that has already
        gated its own call to one rank and for a direct, standalone call
        (as in a test) where there is no meaningful "rank" to gate on.

    Returns
    -------
    hashable
        Chosen dimension.

    Raises
    ------
    ValueError
        If no dimension is available.

    Notes
    -----
    A short-partition warning (see below) is emitted at most once per
    distinct ``(dim, length, mpi_size)`` combination for the life of the
    process, and, when ``rank`` is given, only from rank 0. climtools sets
    its own warnings filter to ``"always"`` for ``climtools.*`` modules (see
    ``climtools/__init__.py``) precisely so a climtools warning is never
    silently dropped by Python's default per-callsite dedup -- but that same
    policy means a warning raised identically on every rank of a hot,
    rank-invariant path would otherwise print once per rank in addition to
    repeating on every call, which is pure noise rather than new
    information. Every rank computes the exact same decision here without
    any communication, so only rank 0 needs to report it.
    """
    blocked = set(exclude)
    candidates = [
        (dim, int(length)) for dim, length in sizes.items() if dim not in blocked
    ]
    if not candidates:
        raise ValueError("No dimension is available for automatic partitioning.")

    usable = [item for item in candidates if item[1] > 1] or candidates
    order = {dim: position for position, (dim, _) in enumerate(usable)}
    dim, length = max(usable, key=lambda item: (item[1], -order[item[0]]))

    if length < mpi_size and (rank is None or rank == 0):
        warn_key = (str(dim), length, mpi_size)
        if warn_key not in _SHORT_PARTITION_WARNED:
            _SHORT_PARTITION_WARNED.add(warn_key)
            warnings.warn(
                f"Automatic partition dimension {str(dim)!r} has length "
                + f"{length}, which is shorter than the {mpi_size} available "
                + f"ranks, so {mpi_size - length} rank(s) will hold no data. "
                + "This message will not repeat for the same dimension, "
                + "length, and rank count.",
                UserWarning,
                stacklevel=3,
            )
    return dim


def indexer_is_scalar(indexer: Any) -> bool:
    return not isinstance(indexer, (slice, list, tuple, np.ndarray, xr.DataArray))


class XarrayMPI(ArithmeticMixin):
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
        partition_dim: Hashable | Literal["auto"] = "auto",
        chunks: Any = None,
        log_partitions: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open a Dataset lazily and distribute one dimension across ranks.

        This method dynamically dispatches to either :func:`xarray.open_dataset` or
        :func:`xarray.open_mfdataset` depending on whether ``filename_or_obj`` is a single
        file/object or a glob pattern/list of files.

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
            Explicit xarray/Dask chunk specification. If omitted, effective
            chunks are derived from usable native chunks, falling back to
            ``ceil(length / nranks)``.
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

        # 1. RANK 0 EVALUATES METADATA AND BUILDS THE PLAN
        plan: dict[str, Any] | None = None
        error: BaseException | None = None
        if self._runtime.comm.rank == 0:
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
                    open_chunk_overrides = get_chunk_overrides(metadata, chunk_info)
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
                        "open_chunk_overrides": open_chunk_overrides,
                        "global_size": global_size,
                    }
            except BaseException as exc:
                error = exc

        # Every rank must learn about a rank-0 planning failure through the
        # same collective sequence rank 0 used to detect it. Raising on rank
        # 0 alone leaves ranks 1..N-1 blocked forever in the plan bcast
        # below, since rank 0 never reaches it once it raises.
        self._runtime.raise_if_error(error, "mpi.xarray.open_dataset planning")

        # 2. BROADCAST THE PLAN TO ALL RANKS
        plan = self._runtime.comm.bcast(plan, root=0)

        partition_dim = plan["partition_dim"]
        chunk_info = plan["chunk_info"]
        open_chunk_overrides = plan["open_chunk_overrides"]
        global_size = plan["global_size"]

        # 3. EVERY RANK DETERMINISTICALLY CALCULATES ITS LOCAL BOUNDS
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

        # --- SYNCHRONIZE ALL RANKS BEFORE MASS I/O ---
        self._runtime.comm.Barrier()

        # 4. EVERY RANK OPENS ITS LAZY SLICE OF THE DATA
        data: xr.Dataset = open_dataset(
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
        """Distribute an object that exists on only one rank.

        ``redistribute`` assumes ``value`` is already the same object on
        every rank and slices each rank's own copy locally, with no MPI
        communication for the data itself. ``distribute`` is for the
        different situation this project's parallel NetCDF writer's
        "legacy scatter" path otherwise forces through
        ``DataArray.values`` (an eager ``.compute()`` of the entire array
        on one rank): the source genuinely exists on only one rank, not
        because it was chosen not to replicate it, but because it cannot
        be -- built from rank-local state, read from a resource only that
        rank can reach, or any other reason it cannot simply be
        reconstructed identically everywhere the way ``redistribute``
        requires.

        Each other rank receives, by direct point-to-point message, only
        the slice it owns -- sliced with ``isel`` on ``root`` before
        sending. If ``value`` is dask-backed, that slicing never triggers
        computation: the message carries an uncomputed graph, not data, so
        no rank other than the eventual owner of a slice ever materializes
        it, and ``root`` never holds more than one slice's worth of pickled
        graph in flight at a time. If ``value`` is already a plain
        in-memory (non-dask) object, this still avoids the redundant
        copies ``to_netcdf``'s scatter path makes, but cannot undo the fact
        that the complete array already had to exist in ``root``'s memory
        before this call -- only a dask-backed source avoids that.

        Parameters
        ----------
        value : xarray.Dataset, xarray.DataArray, or None
            The complete object, on ``root`` only. Every other rank must
            pass None.
        dim : hashable or {"auto"}, default "auto"
            Dimension to distribute along. ``"auto"`` chooses ``root``'s
            largest dimension. Ignored (and no partition metadata is
            attached) if ``value`` has no dimensions at all.
        root : int, default 0
            Rank holding ``value``.
        chunk_info : mapping, optional
            Effective chunk size hints, as accepted by ``redistribute``.
        log_partitions : bool, default False
            Print a partition report on ``root`` once every rank has its
            slice.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            This rank's own slice, tagged with ``mpi_meta`` exactly as
            ``redistribute``'s output is, and just as suitable as input to
            ``to_netcdf(..., parallel=True)`` or any ``mpi.xarray``
            reduction. Not yet loaded: call ``.load()`` when ready to
            materialize it, same as ``redistribute``'s output.

        Raises
        ------
        ValueError
            If ``value`` is None on ``root``, is not None on a non-root
            rank, already carries ``mpi_meta``, or names a dimension that
            does not exist.
        """
        comm = self._runtime.comm
        is_root = comm.rank == root

        # Phase 1: validate and fully prepare on root -- including slicing
        # every rank's piece -- without sending anything anywhere yet.
        # Every rank then passes through raise_if_error together. Only
        # after that succeeds on every rank does phase 2 below send or
        # receive a single message, so a failure here can never leave a
        # non-root rank blocked in recv() waiting for a send that root
        # failed before reaching.
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
                        # stripped came from strip_mpi_meta's .copy(deep=False):
                        # every .isel() slice taken from a shallow-copied
                        # object shares the exact same attrs dict as the
                        # copy itself (and therefore with every other slice
                        # taken from it) rather than getting its own, an
                        # xarray behavior specific to slicing a .copy()'d
                        # object rather than an original. Breaking that
                        # sharing explicitly, rather than relying on it not
                        # to happen, is what set_mpi_meta below needs: it
                        # mutates these dicts in place, and every piece
                        # silently ending up with the last rank's metadata
                        # is exactly what happens without this.
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

        # Every rank must agree on which phase-2 branch to take, but only
        # root's local variables reflect which one it prepared -- a plain
        # `pieces is None` check on a non-root rank is always true and
        # would pick the wrong branch there. One small, cheap broadcast
        # settles it for everyone.
        dimensionless = comm.bcast(
            replicated_value is not None if is_root else None, root=root
        )

        # Phase 2: every rank reaches this point only because every rank,
        # root included, passed phase 1 without error, so this is now
        # ordinary data transfer of already-successfully-prepared pieces.
        if dimensionless:
            # Nothing to partition: same small object broadcast to every
            # rank, no per-rank slicing or point-to-point send needed.
            output = comm.bcast(replicated_value if is_root else None, root=root)
            return cast("xr.Dataset | xr.DataArray", output)

        if is_root:
            assert pieces is not None
            for rank, piece in enumerate(pieces):
                if rank == root:
                    output = piece
                else:
                    comm.send(piece, dest=rank, tag=self._DISTRIBUTE_TAG)
        else:
            output = comm.recv(source=root, tag=self._DISTRIBUTE_TAG)

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
        """Build a DataArray whose local slice every rank computes itself.

        No rank ever holds more than its own slice, and nothing crosses a
        rank boundary: ``get_balanced_bounds`` -- the same helper
        :meth:`redistribute` and :meth:`XarrayMPIRuntime.open_dataset` use
        -- is a pure function of the global length along ``dim``, this
        rank's number and the communicator size, so every rank derives
        identical, non-overlapping, gap-free bounds with no communication
        at all. Each rank then calls ``fill(start, stop)`` with only its
        own bounds and wraps the result in ``dask.delayed``, so the call
        does not run until this rank's slice is actually needed (loaded,
        reduced, or written) rather than eagerly here.

        ``fill`` is an ordinary Python function, not an array -- and it
        must be called with this method identically on **every rank**,
        with the same function, not called on rank 0 alone. There is no
        communication step here to ship anything from one rank to another
        (unlike :meth:`distribute`): each rank runs this same call on its
        own, and if only rank 0 calls it, only rank 0 gets a result. Any
        ordinary closure works -- a formula, a per-rank RNG stream keyed on
        ``comm.rank``, one file per rank -- as long as it is picklable
        (plain data in the closure is fine; an open file handle or lock is
        not) and, for correctness, gives the same answer for a given
        ``(start, stop)`` regardless of when it happens to be called.

        Use this to *generate* new distributed data from a description
        every rank can already evaluate (a formula, a per-rank RNG stream,
        one file per rank, ...). It is not for data that inherently exists
        on only one rank already -- there is no way to hand that to every
        rank without moving bytes somewhere, which is exactly what
        :meth:`distribute` is for.

        Parameters
        ----------
        fill : callable
            A plain function (not an array), called identically on every
            rank: ``fill(start, stop) -> array_like`` returning this
            rank's own slice, shaped by ``dims`` except with ``stop -
            start`` along ``dim``. Called once per rank, each with only
            its own bounds -- see the note above on calling this from
            every rank.
        dims : sequence of hashable
            Dimension names, one per axis.
        shape : sequence of int, mapping, or None, optional
            Every dimension's global length, identical on every rank since
            it is metadata, not data. A sequence gives one length per
            entry of ``dims``, in order. A mapping (or None) gives (or
            leaves out) lengths by dimension name; any name missing from
            it -- or every name, if this is None -- is filled in from a
            matching full-length coordinate in ``coords`` instead. A name
            with neither raises :exc:`ValueError` (see below).
        dim : hashable or int, default 0
            Dimension to distribute along, as a name from ``dims`` or an
            integer axis.
        dtype : optional
            Dtype of the array ``fill`` returns. Default is ``float64``.
        coords : mapping, optional
            Forwarded to the :class:`xarray.DataArray` constructor. A
            coordinate matching ``dim`` whose own length equals its
            resolved global length is sliced to this rank's own bounds
            first, so a full-length coordinate array can be passed the
            same way as to the ordinary constructor -- and, per ``shape``
            above, doing so for every dimension makes ``shape`` itself
            unnecessary.
        name, attrs : optional
            Forwarded to the :class:`xarray.DataArray` constructor.
        log_partitions : bool, default False
            Print a partition report on rank 0 once every rank has built
            its slice.

        Returns
        -------
        xarray.DataArray
            This rank's own slice, tagged with ``mpi_meta`` exactly as
            :meth:`distribute`'s output is. Not yet loaded.

        Raises
        ------
        ValueError
            If ``dim`` does not name (or index) an entry of ``dims``, if
            ``shape`` is a sequence whose length does not match ``dims``,
            or if any dimension's length cannot be determined from
            ``shape`` or ``coords``.
        """
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
        """Build a distributed xarray Dataset where each rank computes its local slice.

        Parameters
        ----------
        data_vars : mapping
            Variables to include, formatted as ``{name: (dims, fill)}`` or
            ``{name: dataarray}``. For variables with ``dim``, ``fill(start, stop)``
            returns the local slice. For variables without ``dim``, ``fill`` is a
            zero-argument callable or array-like. Must be called identically on
            every rank with no communication.
        sizes : mapping, optional
            Global lengths of dimensions. If omitted, dimensions are resolved
            from matching full-length coordinates in ``coords``.
        dim : hashable
            Dimension to distribute along.
        dtype : optional, default np.float64
            Data type for ``fill`` outputs. Can be a scalar or a per-variable mapping.
            A mapping may be partial; unspecified fill variables use ``np.float64``.
        coords, attrs : mapping, optional
            Forwarded to the ``xr.Dataset`` constructor. Coordinates matching ``dim``
            are automatically sliced to the rank's bounds.
        log_partitions : bool, default False
            If True, prints a partition report on rank 0.

        Returns
        -------
        xr.Dataset
            This rank's local dataset slice tagged with ``mpi_meta`` (unloaded).

        Raises
        ------
        ValueError
            If any dimension cannot be resolved from ``sizes`` or ``coords``, or if
            a bare DataArray carrying ``dim`` has an incorrect local size.
        """
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
                dtype_map.get(var_name, np.float64)
                if dtype_map is not None
                else dtype
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
        """Return metadata when a reduction can stay entirely rank-local."""
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
        result: xr.Dataset | xr.DataArray,
        *,
        old_meta: Mapping[str, Any],
    ) -> xr.Dataset | xr.DataArray:
        """Restore unchanged partition ownership after a rank-local reduction."""
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
    ) -> tuple[PlanEntry, ...]:
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
    def _guarded(
        function: Any,
    ) -> tuple[Any, BaseException | None]:
        """Run a rank-local computation, deferring any failure.

        A rank-local computation that raises between two collectives removes
        that rank from the collective sequence while the others continue,
        which is a deadlock rather than an error. Deferring the exception
        lets the rank stay in the sequence until the next collective
        synchronizes and re-raises it on every rank.
        """
        try:
            return function(), None
        except BaseException as exc:
            return None, exc

    def _partition_is_empty(self, value: xr.Dataset | xr.DataArray, meta: Any) -> bool:
        """Return whether this rank owns no elements of the partition."""
        if meta is None:
            return False
        dim = meta["dim"]
        return dim in value.dims and int(value.sizes[dim]) == 0

    # -- collective primitives -----------------------------------------------

    def _comm_reduce(
        self,
        value: xr.DataArray | None,
        op: _MPI.Op,
        *,
        expect_dtype: np.dtype[Any] | None = None,
        error: BaseException | None = None,
        phase: str = "MPI xarray reduction buffer preparation",
    ) -> xr.DataArray:
        """All-reduce one rank-local buffer across the communicator."""
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

    def _exchange(
        self,
        send: np.ndarray[Any, Any],
        op: _MPI.Op,
    ) -> np.ndarray[Any, Any]:
        """All-reduce an already validated contiguous send buffer."""
        recv = np.empty(send.shape, dtype=send.dtype)
        self._runtime.comm.Allreduce(send, recv, op=op)
        return recv

    def _count(
        self,
        value: xr.DataArray,
        dims: tuple[Hashable, ...],
    ) -> xr.DataArray:
        count: xr.DataArray | None = None
        error: BaseException | None = None
        try:
            count = value.count(dim=dims, keep_attrs=False)
        except BaseException as exc:
            error = exc
        return self._comm_reduce(
            count,
            _MPI.SUM,
            expect_dtype=_partial_dtype(value.dtype.str, "count", None),
            error=error,
            phase="MPI xarray count reduction",
        )

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
    ) -> xr.Dataset:
        return self._reduced_dataset(value, dims, variables)

    @staticmethod
    def _redistribution_candidates(plan: tuple[PlanEntry, ...]) -> frozenset[Hashable]:
        """Dimensions eligible as a post-reduction partition dimension.

        Restricted to dimensions still present on a variable that itself
        required an MPI collective (``entry.distributed``). A Dataset
        variable that never carried the active partition dimension is
        identical, not partitioned, on every rank; selecting one of its own
        dimensions as the new partition would slice that replicated variable
        differently per rank instead of scattering a real global object, so
        such dimensions are never legitimate ``"auto"`` or explicit targets.
        """
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
        """Finalize a global reduction and choose its next distribution.

        Parameters
        ----------
        result : xarray.Dataset or xarray.DataArray
            Global reduction result, currently replicated on every rank.
        old_meta : mapping, optional
            ``mpi_meta`` of the object that was reduced.
        redistribute_on : hashable, "auto", or None
            Placement requested by the caller.
        auto_candidates : frozenset of hashable
            Dimensions eligible as the new partition dimension -- those
            still present on a variable that itself required an MPI
            collective. See :meth:`_redistribution_candidates`. Also
            enforced for an explicit ``redistribute_on`` so a caller cannot
            accidentally fabricate a fake partition on an untouched,
            replicated sibling variable either.
        """
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
        return self.redistribute(
            result,
            target,
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
        error: BaseException | None = None,
    ) -> xr.DataArray:
        result = self._comm_reduce(
            partial,
            op,
            expect_dtype=_partial_dtype(
                value.dtype.str,
                "prod" if _op_name(op) == "PROD" else "sum",
                skipna,
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
        global_sum = self._comm_reduce(
            partial_sum,
            _MPI.SUM,
            expect_dtype=_partial_dtype(value.dtype.str, "sum", skipna),
            error=error,
            phase="MPI xarray mean reduction",
        )
        global_count = self._count(value, dims)
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
        partial: xr.DataArray | None,
        dims: tuple[Hashable, ...],
        *,
        minimum: bool,
        skipna: bool | None,
        error: BaseException | None = None,
    ) -> xr.DataArray:
        # The number of collectives is decided from the reduced variable's
        # declared dtype, which the plan has already agreed on, never from
        # the rank-local partial. A rank owning an empty partition builds its
        # partial through a different code path from a rank owning data, so
        # branching on the partial's dtype can make those ranks post
        # different numbers of collectives and desynchronize the run.
        #
        # Unlike sum/prod/mean, min/max never need dtype promotion: the
        # extreme of a set of same-dtype values is always one of those
        # values (or, under skipna=False, a NaN already representable in
        # that dtype), so the declared variable dtype is the exact and only
        # correct answer. This is computed directly rather than through
        # _partial_dtype's zero-size probe, because a full (all-dims)
        # reduction of a float32 array through xarray's bottleneck-backed
        # nanmin/nanmax silently promotes the result to float64 when
        # bottleneck is installed, which the probe would otherwise inherit
        # and then force onto every rank's buffer.
        operation = "min" if minimum else "max"
        expect_dtype = value.dtype
        kind = value.dtype.kind
        if kind == "b":
            return self._comm_reduce(
                partial,
                _MPI.LAND if minimum else _MPI.LOR,
                expect_dtype=expect_dtype,
                error=error,
                phase=f"MPI xarray {operation} reduction",
            )

        op = _MPI.MIN if minimum else _MPI.MAX
        if kind != "f":
            return self._comm_reduce(
                partial,
                op,
                expect_dtype=expect_dtype,
                error=error,
                phase=f"MPI xarray {operation} reduction",
            )

        # Floating point needs a validity flag alongside the extreme itself,
        # because a rank with an empty partition, or with an all-NaN slice
        # under skipna, must contribute an identity that is then distinguished
        # from a genuine infinite value in the data. The flag used to travel in
        # a second boolean collective. It now shares the value buffer: the flag
        # is encoded so that the same MIN or MAX operation computes the
        # required ANY or ALL over the ranks. That halves the collectives, and
        # removes a boolean reduction whose MPI datatype handling is the least
        # portable part of this path.
        send: np.ndarray[Any, Any] | None = None
        template: xr.DataArray | None = None
        skipna_enabled = self._skipna_enabled(value.dtype, skipna)
        # ANY valid rank suffices under skipna; without it every rank must be
        # NaN-free for the result to be defined.
        flip = -1.0 if ((not minimum) != skipna_enabled) else 1.0

        if error is None:
            try:
                identity = np.asarray(
                    np.inf if minimum else -np.inf,
                    dtype=expect_dtype,
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
            error,
            f"MPI xarray {operation} reduction",
            signature,
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
        """Sum over one or more dimensions of a distributed xarray object.

        Parameters
        ----------
        value : xarray.DataArray or xarray.Dataset
            Object to reduce. Objects carrying ``mpi_meta`` are interpreted as
            rank-local slabs of one global xarray object.
        dim : str, iterable of hashable, ..., or None, optional
            Dimension or dimensions over which to sum. ``None`` or ``...``
            reduces over all dimensions, matching xarray semantics.
        skipna : bool or None, optional
            If True, skip missing values. If None, use xarray's dtype-dependent
            default.
        min_count : int or None, optional
            Minimum number of valid values required for the result. When the
            active partition dimension is reduced, the count is combined across
            ranks before this threshold is applied.
        keep_attrs : bool or None, optional
            Whether to preserve attributes, with the same meaning as xarray.
        redistribute_on : hashable, "auto", or None, default "auto"
            Placement after the active partition dimension is reduced away.
            ``"auto"`` repartitions on the longest surviving dimension whose
            length exceeds one, an explicit dimension selects that dimension,
            and None leaves the global result replicated on every rank. If the
            active partition dimension survives, ``"auto"`` and None preserve
            it and an explicit replacement dimension is invalid.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Reduced object, distributed when a useful partition dimension
            survives or is selected after the reduction.

        Notes
        -----
        If ``dim`` excludes the active partition dimension, this is exactly a
        native xarray reduction on each rank and performs no MPI collective.
        If ``dim`` includes it, every rank first sums over all requested local
        dimensions in one xarray operation and the resulting partials are
        combined with ``MPI.SUM``.
        """
        return self._sum_prod(
            value,
            dim,
            op=_MPI.SUM,
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
        """Multiply values over one or more dimensions of a distributed object.

        Parameters
        ----------
        value : xarray.DataArray or xarray.Dataset
            Object to reduce. Objects carrying ``mpi_meta`` are interpreted as
            rank-local slabs of one global xarray object.
        dim : str, iterable of hashable, ..., or None, optional
            Dimension or dimensions over which to take the product. ``None``
            or ``...`` reduces over all dimensions, matching xarray semantics.
        skipna : bool or None, optional
            If True, skip missing values. If None, use xarray's dtype-dependent
            default.
        min_count : int or None, optional
            Minimum number of valid values required for the result. When the
            active partition dimension is reduced, the count is combined across
            ranks before this threshold is applied.
        keep_attrs : bool or None, optional
            Whether to preserve attributes, with the same meaning as xarray.
        redistribute_on : hashable, "auto", or None, default "auto"
            Placement after the active partition dimension is reduced away.
            ``"auto"`` repartitions on the longest surviving dimension whose
            length exceeds one, an explicit dimension selects that dimension,
            and None leaves the global result replicated on every rank. If the
            active partition dimension survives, ``"auto"`` and None preserve
            it and an explicit replacement dimension is invalid.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Reduced object, distributed when a useful partition dimension
            survives or is selected after the reduction.

        Notes
        -----
        If ``dim`` excludes the active partition dimension, this is a native
        rank-local xarray product with no MPI collective. If ``dim`` includes
        it, all requested dimensions are collapsed locally first and the rank
        partials are combined with ``MPI.PROD``.
        """
        return self._sum_prod(
            value,
            dim,
            op=_MPI.PROD,
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
        op: _MPI.Op,
        product: bool,
        skipna: bool | None,
        min_count: int | None,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
    ) -> xr.Dataset | xr.DataArray:
        operation = "prod" if product else "sum"
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta,
            dims,
            redistribute_on=redistribute_on,
        )
        if local_meta is not None:
            method = value.prod if product else value.sum
            local_result = method(
                dim=local_dim,
                skipna=skipna,
                min_count=min_count,
                keep_attrs=keep_attrs,
            )
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(
            value,
            dims,
            old_meta,
            operation=operation,
        )

        if isinstance(value, xr.DataArray):
            method = value.prod if product else value.sum
            local, local_error = self._guarded(
                lambda: method(
                    dim=local_dim,
                    skipna=skipna,
                    min_count=None,
                    keep_attrs=keep_attrs,
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
                    dim=entry.dims,
                    skipna=skipna,
                    min_count=None,
                    keep_attrs=keep_attrs,
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
        """Compute the arithmetic mean over one or more dimensions.

        Parameters
        ----------
        value : xarray.DataArray or xarray.Dataset
            Distributed xarray object to reduce.
        dim : str, iterable of hashable, ..., or None, optional
            Dimension or dimensions over which to average. ``None`` or ``...``
            reduces over all dimensions, so a distributed object receives a
            true global mean rather than a rank-local mean.
        skipna : bool or None, optional
            If True, skip missing values. If None, use xarray's dtype-dependent
            default.
        keep_attrs : bool or None, optional
            Whether to preserve attributes, with the same meaning as xarray.
        redistribute_on : hashable, "auto", or None, default "auto"
            Placement after the active partition dimension is reduced away.
            ``"auto"`` repartitions on the longest surviving dimension whose
            length exceeds one, an explicit dimension selects that dimension,
            and None leaves the global result replicated. If the active
            partition dimension survives, ``"auto"`` and None preserve it.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Mean with xarray-compatible coordinates, attributes, and dtype.

        Notes
        -----
        A reduction that excludes the active partition dimension is performed
        entirely by native xarray on each rank. When the partition dimension is
        included, each rank computes the sum over all requested local dimensions
        and the corresponding valid-value count; both are summed across ranks
        before division. Rank means are never averaged directly.
        """
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta,
            dims,
            redistribute_on=redistribute_on,
        )
        if local_meta is not None:
            local_result = value.mean(
                dim=local_dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(
            value,
            dims,
            old_meta,
            operation="mean",
        )

        if isinstance(value, xr.DataArray):
            if not dims:
                local_mean = value.mean(
                    dim=local_dim,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                )
                return local_mean
            local_sum, local_error = self._guarded(
                lambda: value.sum(
                    dim=local_dim,
                    skipna=skipna,
                    min_count=None,
                    keep_attrs=keep_attrs,
                )
            )
            result = self._combine_mean(
                value,
                local_sum,
                dims,
                skipna=skipna,
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
            if not entry.distributed:
                variables[entry.name] = variable.mean(
                    dim=entry.dims,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                )
                continue
            local_sum, local_error = self._guarded(
                lambda variable=variable, entry=entry: variable.sum(
                    dim=entry.dims,
                    skipna=skipna,
                    min_count=None,
                    keep_attrs=keep_attrs,
                )
            )
            result = self._combine_mean(
                variable,
                local_sum,
                entry.dims,
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

    def min(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Compute the minimum over one or more dimensions.

        Parameters
        ----------
        value : xarray.DataArray or xarray.Dataset
            Distributed xarray object to reduce.
        dim : str, iterable of hashable, ..., or None, optional
            Dimension or dimensions over which to compute the minimum. ``None``
            or ``...`` reduces over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values. If None, use xarray's dtype-dependent
            default.
        keep_attrs : bool or None, optional
            Whether to preserve attributes, with the same meaning as xarray.
        redistribute_on : hashable, "auto", or None, default "auto"
            Placement after the active partition dimension is reduced away.
            ``"auto"`` repartitions on the longest surviving dimension whose
            length exceeds one; None leaves the global result replicated. If
            the active partition survives, its existing partition is retained.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Global minimum over the requested dimensions.

        Notes
        -----
        Non-partition dimensions are reduced entirely locally. When the active
        partition dimension participates, each rank first computes its minimum
        over all requested local dimensions and the partial extrema are combined
        globally. Floating-point reductions retain the existing all-NaN and
        empty-partition validity handling; boolean minimum is logical AND.
        """
        return self._extreme(
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
        """Compute the maximum over one or more dimensions.

        Parameters
        ----------
        value : xarray.DataArray or xarray.Dataset
            Distributed xarray object to reduce.
        dim : str, iterable of hashable, ..., or None, optional
            Dimension or dimensions over which to compute the maximum. ``None``
            or ``...`` reduces over all dimensions and therefore gives the true
            global maximum of a distributed DataArray.
        skipna : bool or None, optional
            If True, skip missing values. If None, use xarray's dtype-dependent
            default.
        keep_attrs : bool or None, optional
            Whether to preserve attributes, with the same meaning as xarray.
        redistribute_on : hashable, "auto", or None, default "auto"
            Placement after the active partition dimension is reduced away.
            ``"auto"`` repartitions on the longest surviving dimension whose
            length exceeds one; None leaves the global result replicated. If
            the active partition survives, its existing partition is retained.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Global maximum over the requested dimensions.

        Notes
        -----
        Non-partition dimensions are reduced entirely locally. When the active
        partition dimension participates, each rank first computes its maximum
        over all requested local dimensions and the partial extrema are combined
        globally. Floating-point reductions retain the existing all-NaN and
        empty-partition validity handling; boolean maximum is logical OR.
        """
        return self._extreme(
            value,
            dim,
            minimum=False,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
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
    ) -> xr.Dataset | xr.DataArray:
        operation = "min" if minimum else "max"
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta,
            dims,
            redistribute_on=redistribute_on,
        )
        if local_meta is not None:
            method = value.min if minimum else value.max
            local_result = method(
                dim=local_dim,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(
            value,
            dims,
            old_meta,
            operation=operation,
        )
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
                value,
                local,
                dims,
                minimum=minimum,
                skipna=skipna,
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
        """Return whether any value is true over one or more dimensions.

        Parameters
        ----------
        value : xarray.DataArray or xarray.Dataset
            Distributed xarray object to reduce.
        dim : str, iterable of hashable, ..., or None, optional
            Dimension or dimensions over which to apply logical OR. ``None`` or
            ``...`` reduces over all dimensions.
        keep_attrs : bool or None, optional
            Whether to preserve attributes, with the same meaning as xarray.
        redistribute_on : hashable, "auto", or None, default "auto"
            Placement after the active partition dimension is reduced away.
            ``"auto"`` repartitions on the longest surviving dimension whose
            length exceeds one; None leaves the global result replicated. If
            the active partition survives, its existing partition is retained.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Logical OR over the requested dimensions.

        Notes
        -----
        When the active partition dimension is absent from ``dim``, native
        xarray handles the reduction locally with no MPI communication. When it
        is present, each rank reduces all requested local dimensions first and
        the boolean partials are combined with ``MPI.LOR``.
        """
        return self._logical(
            value,
            dim,
            op=_MPI.LOR,
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
        """Return whether all values are true over one or more dimensions.

        Parameters
        ----------
        value : xarray.DataArray or xarray.Dataset
            Distributed xarray object to reduce.
        dim : str, iterable of hashable, ..., or None, optional
            Dimension or dimensions over which to apply logical AND. ``None`` or
            ``...`` reduces over all dimensions.
        keep_attrs : bool or None, optional
            Whether to preserve attributes, with the same meaning as xarray.
        redistribute_on : hashable, "auto", or None, default "auto"
            Placement after the active partition dimension is reduced away.
            ``"auto"`` repartitions on the longest surviving dimension whose
            length exceeds one; None leaves the global result replicated. If
            the active partition survives, its existing partition is retained.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Logical AND over the requested dimensions.

        Notes
        -----
        When the active partition dimension is absent from ``dim``, native
        xarray handles the reduction locally with no MPI communication. When it
        is present, each rank reduces all requested local dimensions first and
        the boolean partials are combined with ``MPI.LAND``.
        """
        return self._logical(
            value,
            dim,
            op=_MPI.LAND,
            all_values=True,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
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
    ) -> xr.Dataset | xr.DataArray:
        operation = "all" if all_values else "any"
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta,
            dims,
            redistribute_on=redistribute_on,
        )
        if local_meta is not None:
            method = value.all if all_values else value.any
            local_result = method(dim=local_dim, keep_attrs=keep_attrs)
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(
            value,
            dims,
            old_meta,
            operation=operation,
        )

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
