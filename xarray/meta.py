"""Shared MPI metadata helpers for distributed xarray objects."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..core.lib_mpi import MPIRuntime

MPI_META = "mpi_meta"
# The subset of a partition's metadata that decides whether two partitions
# describe the same rank-local ownership. chunk_info and save_chunks are
# deliberately excluded: they record how the split/write was computed for
# the benefit of a later redistribute(..., chunk_info=...) or a NetCDF
# write, not the ownership itself, so two partitions with different (or
# absent) chunk_info/save_chunks but identical dim/global_size/start/stop
# still own the exact same data and are still equal.
_PARTITION_KEYS: tuple[str, ...] = ("dim", "global_size", "start", "stop")


def _partitions_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two partition metadata mappings own the same slice."""
    return all(left.get(key) == right.get(key) for key in _PARTITION_KEYS)


def _validate_mpi_meta(
    value: xr.Dataset | xr.DataArray, meta: Any
) -> dict[str, Any] | None:
    """Return ``meta`` when it describes a valid partition of ``value``."""
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

    if not isinstance(meta["chunk_info"], dict):
        return None

    return cast("dict[str, Any]", meta)


def get_mpi_meta(value: xr.Dataset | xr.DataArray) -> dict[str, Any] | None:
    """Return validated MPI distribution metadata.

    The metadata is looked for on the object itself and, for a Dataset, on its
    variables as a fallback. The fallback exists because
    :meth:`xarray.DataArray.to_dataset` moves the array's attributes onto the
    resulting data variable and leaves ``Dataset.attrs`` empty. Inspecting only
    the top level therefore reported a distributed DataArray as ordinary
    replicated data as soon as any caller converted it, which silently routed
    parallel writes through the rank-0 scatter path and produced a file holding
    only rank 0's slab.

    Variable-level metadata is used only when every variable that carries it
    agrees, and only when it also describes the Dataset as a whole. Disagreeing
    variables mean the object was assembled from separately distributed pieces,
    which no single partition description can represent, so None is returned
    and the caller treats the object as undistributed rather than acting on an
    arbitrary choice.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object whose distribution metadata is requested.

    Returns
    -------
    dict or None
        Distribution metadata when valid, otherwise None.
    """
    meta = _validate_mpi_meta(value, value.attrs.get(MPI_META))
    if meta is not None:
        return meta

    if not isinstance(value, xr.Dataset):
        return None

    candidates: list[dict[str, Any]] = []
    for variable in value.variables.values():
        candidate = variable.attrs.get(MPI_META)
        if isinstance(candidate, dict):
            candidates.append(candidate)

    if not candidates:
        return None

    reference = candidates[0]
    for candidate in candidates[1:]:
        if not _partitions_match(candidate, reference):
            return None

    return _validate_mpi_meta(value, reference)


def _assign_meta(value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any]) -> None:
    """Attach an already-built ``meta`` dict to ``value`` and its variables.

    Shared by :func:`set_mpi_meta` (building ``meta`` from scratch) and
    :func:`set_save_chunks` (adding one key to an existing ``meta``): both
    need the identical propagation rule -- set on the object itself, and
    on every variable that carries ``meta["dim"]``, clearing any stale
    metadata on variables that do not.
    """
    dim = meta["dim"]
    value.attrs[MPI_META] = dict(meta)
    if isinstance(value, xr.Dataset):
        for variable in value.variables.values():
            variable.attrs.pop(MPI_META, None)
            if dim in variable.dims:
                variable.attrs[MPI_META] = dict(meta)


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
    _assign_meta(value, meta)


def set_save_chunks(
    value: xr.Dataset | xr.DataArray, save_chunks: Mapping[str, tuple[int, ...]]
) -> None:
    """Attach save_chunks to ``value``'s existing MPI distribution metadata.

    ``save_chunks`` -- the on-disk NetCDF chunk shape computed by
    :func:`~climtools.core.xr_chunks.compute_save_chunks` -- is stored
    under the ``"save_chunks"`` key alongside the ``dim``/``global_size``/
    ``start``/``stop``/``chunk_info`` keys :func:`set_mpi_meta` already
    sets. It is excluded from ``_PARTITION_KEYS`` deliberately, for the
    same reason ``chunk_info`` is: it records how a write should be
    shaped, not which slice this rank owns, so two partitions with
    different (or absent) ``save_chunks`` but identical
    dim/global_size/start/stop still describe the same ownership and are
    still equal under :func:`_partitions_match`.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Rank-local xarray object that already carries valid MPI
        distribution metadata (see :func:`get_mpi_meta`).
    save_chunks : mapping
        Mapping from variable name to save_chunk shape.

    Raises
    ------
    ValueError
        If ``value`` carries no valid MPI distribution metadata to attach
        ``save_chunks`` to.
    """
    meta = get_mpi_meta(value)
    if meta is None:
        raise ValueError(
            "value carries no MPI distribution metadata; call set_mpi_meta "
            + "(e.g. via mpi.xarray.redistribute/open_dataset) before "
            + "attaching save_chunks."
        )
    updated = dict(meta)
    updated["save_chunks"] = {
        str(name): tuple(int(length) for length in shape)
        for name, shape in save_chunks.items()
    }
    _assign_meta(value, updated)


def strip_mpi_meta(value: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Return a shallow copy without MPI distribution metadata."""
    output = value.copy(deep=False)
    output.attrs.pop(MPI_META, None)
    if isinstance(output, xr.Dataset):
        for variable in output.variables.values():
            variable.attrs.pop(MPI_META, None)
    return output


def _format_label(value: Any, limit: int = 16) -> str:
    """Return a short, fixed-width-friendly label for a coordinate value.

    Datetime labels are the reason the previous report scrolled sideways: a
    numpy datetime64[ns] renders as 29 characters. Seconds and nanoseconds
    carry no information for a partition boundary, so they are dropped.
    """
    if isinstance(value, np.datetime64):
        text = str(np.datetime64(value, "m"))
    elif isinstance(value, (np.floating, float)):
        text = f"{float(value):g}"
    else:
        text = str(value)
    if len(text) > limit:
        text = text[: limit - 1] + "\u2026"
    return text


def _edge_labels(data: xr.Dataset | xr.DataArray, dim: Hashable) -> tuple[str, str]:
    """Return the first and last coordinate labels owned along ``dim``."""
    if dim not in data.coords or int(data.sizes[dim]) == 0:
        return "-", "-"
    values = np.asarray(data.coords[dim].values)
    return _format_label(values[0]), _format_label(values[-1])


def _format_bytes(count: float) -> str:
    """Return a compact binary size label."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if count < 1024.0 or unit == "TiB":
            return f"{count:.0f}{unit}" if unit == "B" else f"{count:.1f}{unit}"
        count /= 1024.0
    return f"{count:.1f}TiB"


def log_partition_report(
    runtime: MPIRuntime,
    data: xr.Dataset | xr.DataArray,
    dim: Hashable,
    *,
    origin: str,
    global_size: int,
    start: int,
    stop: int,
    automatic: bool = False,
    detail: bool = True,
) -> None:
    """Print a structured, compact description of the rank-local partition layout."""
    comm = runtime.comm
    first, last = _edge_labels(data, dim)
    local = (int(comm.rank), int(start), int(stop), first, last, int(data.nbytes))
    rows = comm.gather(local, root=0)
    if comm.rank != 0 or rows is None:
        return

    counts = [row[2] - row[1] for row in rows]
    total = sum(row[5] for row in rows)
    peak_bytes = max(row[5] for row in rows)
    idle = sum(1 for count in counts if count == 0)

    other = " ".join(
        f"{name!s}={int(length)}" for name, length in data.sizes.items() if name != dim
    )
    chunk_text = "  ".join(
        f"{name!s}={max(int(size) for size in chunks)}"
        for name, chunks in (data.chunks or {}).items()
    )

    # Build structured summary values
    dim_str = f"{str(dim)!r}{' (auto)' if automatic else ''}"

    # Use a single value if min and max counts are identical, otherwise show range
    if min(counts) == max(counts):
        split_str = f"{min(counts)}/rank"
    else:
        split_str = f"{min(counts)}-{max(counts)}/rank"

    if idle:
        split_str += f" (IDLE={idle})"

    usage_str = f"{_format_bytes(total)} total (Peak/Rank: {_format_bytes(peak_bytes)})"

    border = "=" * 80
    separator = "-" * 80

    lines = [
        border,
        f" MPI PARTITION REPORT: {origin}",
        border,
        f" 🔹 Dimension   : {dim_str}",
        f" 🔹 Global Size : {global_size} (Ranks: {comm.size})",
        f" 🔹 Split       : {split_str}",
        f" 🔹 Shape       : {other or 'scalar'}",
        f" 🔹 Chunks/Rank : {chunk_text or 'unchunked'}",
        f" 🔹 Memory      : {usage_str}",
    ]

    if detail:
        lines.append(separator)
        # Calculate max widths using the original table logic
        slice_width = max(len("slice"), *(len(f"{row[1]}:{row[2]}") for row in rows))
        first_width = max(len("first"), *(len(str(row[3])) for row in rows))
        last_width = max(len("last"), *(len(str(row[4])) for row in rows))
        count_width = max(len("n"), *(len(str(row[2] - row[1])) for row in rows))

        lines.append(
            f"   {'rank':>4}  {'slice':>{slice_width}}  {'n':>{count_width}}  "
            + f"({'first':>{first_width}}, {'last':>{last_width}})"
        )
        lines.append(separator)

        for row in rows:
            slice_str = f"{row[1]}:{row[2]}"
            count_val = row[2] - row[1]
            first_str = str(row[3])
            last_str = str(row[4])

            lines.append(
                f"   {row[0]:>4}  {slice_str:>{slice_width}}  {count_val:>{count_width}}  "
                + f"({first_str:>{first_width}}, {last_str:>{last_width}})"
            )

    lines.append(border)
    runtime.log("")
    runtime.log("\n".join(lines), flush=True, prefix=False)
    runtime.log("", prefix=False)


def indexer_is_scalar(indexer: Any) -> bool:
    """Return whether an isel/sel indexer selects a single position.

    Parameters
    ----------
    indexer : Any
        Value passed as an index. A :class:`slice`, ``list``, ``tuple``,
        :class:`numpy.ndarray`, or :class:`xarray.DataArray` selects zero or
        more positions; anything else (an int, a label, ...) selects one.

    Returns
    -------
    bool
        True when ``indexer`` selects exactly one position and therefore
        drops its dimension, rather than keeping it with length one.
    """
    return not isinstance(indexer, (slice, list, tuple, np.ndarray, xr.DataArray))


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


def _localize_coord(spec: Any, global_size: int, start: int, stop: int) -> Any:
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
    fn: Callable[..., Any], args: tuple[Any, ...], shape: tuple[int, ...], dtype: Any
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
