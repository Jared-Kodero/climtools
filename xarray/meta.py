"""Manage MPI distribution metadata for xarray objects."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from mpi4py import MPI

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..mpi.runtime import MPIContext

MPI_META = "mpi_meta"
#: Attrs key for the lightweight boolean partition flag set by
#: :func:`~.core.mark_partitioned`. Tracked here (not just in ``MPI_META``)
#: so it can be stripped alongside the full metadata dict by
#: :func:`strip_mpi_meta` and by every NetCDF attribute writer -- a bare
#: Python/NumPy bool is not a valid NetCDF attribute type, so leaving it in
#: ``attrs`` fails schema creation as soon as a freshly partitioned (not
#: reopened) object is passed to ``to_netcdf(..., parallel=True)``.
PARTITIONED_ATTR = "mpi_partitioned"
#: Internal bookkeeping keys that must never reach a NetCDF attribute
#: writer or a "real" attrs comparison. Centralized so every strip/export
#: call site (``strip_mpi_meta``, every schema-building block in
#: ``netcdf.py``) filters the identical set instead of each re-deriving it.
_INTERNAL_ATTRS = frozenset({MPI_META, PARTITIONED_ATTR})
# The subset of a partition's metadata that decides whether two partitions
# describe the same rank-local ownership. chunk_info and save_chunks are
# deliberately excluded: they record how the split/write was computed for
# the benefit of a later repartition(..., chunk_info=...) or a NetCDF
# write, not the ownership itself, so two partitions with different (or
# absent) chunk_info/save_chunks but identical dims/global_sizes/starts/stops
# still own the exact same data and are still equal.
#
# ``meta`` always carries both a plural, canonical description of every
# partition dimension (``dims``/``global_sizes``/``starts``/``stops``) and,
# mirroring ``dims[0]``, the original singular keys (``dim``/``global_size``/
# ``start``/``stop``). The singular keys exist purely so every pre-existing
# single-dimension consumer of ``meta["dim"]`` etc. keeps working unmodified;
# they are correct in full for the (default, most common) one-dimensional
# case, and describe only the first partition axis when more than one
# dimension is partitioned. Any code that must be correct for a
# multi-dimensional partition reads ``dims``/``starts``/``stops``/
# ``global_sizes`` instead. A dict carrying only the legacy singular keys
# (as attached by an older climtools version, e.g. read back from a
# previously written NetCDF file's attrs) is still recognized: see
# :func:`_canonicalize_meta`.


def _canonicalize_meta(meta: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return ``meta`` with both plural and singular partition keys present."""
    if "dims" in meta:
        required = {"dims", "global_sizes", "starts", "stops", "chunk_info"}
        if not required <= meta.keys():
            return None
        dims = tuple(meta["dims"])
        global_sizes, starts, stops = (
            meta["global_sizes"],
            meta["starts"],
            meta["stops"],
        )
        if not dims or not all(
            d in global_sizes and d in starts and d in stops for d in dims
        ):
            return None
        out = dict(meta)
        out["dims"] = dims
        out.setdefault("dim", dims[0])
        out.setdefault("global_size", global_sizes[dims[0]])
        out.setdefault("start", starts[dims[0]])
        out.setdefault("stop", stops[dims[0]])
        return out

    required = {"dim", "global_size", "start", "stop", "chunk_info"}
    if not required <= meta.keys():
        return None
    dim = meta["dim"]
    out = dict(meta)
    out["dims"] = (dim,)
    out["global_sizes"] = {dim: meta["global_size"]}
    out["starts"] = {dim: meta["start"]}
    out["stops"] = {dim: meta["stop"]}
    return out


def _partitions_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two partition metadata mappings own the same slice."""
    left_c = _canonicalize_meta(left)
    right_c = _canonicalize_meta(right)
    if left_c is None or right_c is None:
        return False
    if set(left_c["dims"]) != set(right_c["dims"]):
        return False
    return all(
        left_c["starts"].get(dim) == right_c["starts"].get(dim)
        and left_c["stops"].get(dim) == right_c["stops"].get(dim)
        and left_c["global_sizes"].get(dim) == right_c["global_sizes"].get(dim)
        for dim in left_c["dims"]
    )


def _validate_mpi_meta(
    value: xr.Dataset | xr.DataArray, meta: Any
) -> dict[str, Any] | None:
    """Return ``meta`` when it describes a valid partition of ``value``."""
    if not isinstance(meta, dict):
        return None

    canonical = _canonicalize_meta(meta)
    if canonical is None:
        return None

    dims = canonical["dims"]
    present = [dim for dim in dims if dim in value.dims]
    if not present:
        return None

    for dim in present:
        start = int(canonical["starts"][dim])
        stop = int(canonical["stops"][dim])
        global_size = int(canonical["global_sizes"][dim])
        if start < 0 or stop < start or stop > global_size:
            return None
        if int(value.sizes[dim]) != stop - start:
            return None

    if not isinstance(canonical["chunk_info"], dict):
        return None

    return cast("dict[str, Any]", canonical)


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


def assign_mpi_meta(value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any]) -> None:
    """Attach an already-built ``meta`` dict to ``value`` and its variables.

    Parameters
    ----------
    value : xr.Dataset | xr.DataArray
        Distributed xarray object.
    meta : Mapping[str, Any]
        MPI distribution metadata.
    """
    dims = meta["dims"]
    value.attrs[MPI_META] = dict(meta)
    if isinstance(value, xr.Dataset):
        for variable in value.variables.values():
            variable.attrs.pop(MPI_META, None)
            if any(dim in variable.dims for dim in dims):
                variable.attrs[MPI_META] = dict(meta)


def _as_dims(dim: Hashable | Iterable[Hashable]) -> tuple[str, ...]:
    """Normalize a ``dim`` argument (one dim, or a sequence of dims) to a tuple."""
    dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
    if not dims:
        raise ValueError("At least one partition dimension is required.")
    if len(set(dims)) != len(dims):
        raise ValueError(f"Partition dimensions must be unique; got {dims!r}.")
    return tuple(str(d) for d in dims)


def _as_dim_map(
    dims: tuple[str, ...], value: int | Mapping[Hashable, int], name: str
) -> dict[str, int]:
    """Normalize a per-dimension argument to a ``{dim: value}`` mapping."""
    if isinstance(value, Mapping):
        resolved = {str(k): int(v) for k, v in value.items()}
        missing = [dim for dim in dims if dim not in resolved]
        if missing:
            raise ValueError(f"{name} is missing an entry for {missing!r}.")
        return resolved
    if len(dims) != 1:
        raise ValueError(
            f"{name} must be a mapping of dim -> value when more than one "
            + f"partition dimension is given; got dims={dims!r}."
        )
    return {dims[0]: int(cast("int", value))}


def set_mpi_meta(
    value: xr.Dataset | xr.DataArray,
    *,
    dim: Hashable | Sequence[Hashable],
    global_size: int | Mapping[Hashable, int],
    start: int | Mapping[Hashable, int],
    stop: int | Mapping[Hashable, int],
    chunk_info: Mapping[Hashable, int],
    cart: Mapping[str, Any] | None = None,
) -> None:
    """Attach MPI distribution metadata for one or more partition dimensions.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Rank-local xarray object.
    dim : hashable or sequence of hashable
        Distributed dimension(s).
    global_size : int or mapping
        Global length of ``dim``.
    start, stop : int or mapping
        Global half-open interval owned by this rank, per dimension.
    chunk_info : mapping
        Effective climtools chunk size for every retained dimension.
    cart : mapping, optional
        Cartesian topology descriptor (``grid_shape``, ``coords``, ``periods``), attached only for a multi-dimensional partition.
    """
    dims = _as_dims(dim)
    global_sizes = _as_dim_map(dims, global_size, "global_size")
    starts = _as_dim_map(dims, start, "start")
    stops = _as_dim_map(dims, stop, "stop")

    meta: dict[str, Any] = {
        "dims": dims,
        "global_sizes": {d: global_sizes[d] for d in dims},
        "starts": {d: starts[d] for d in dims},
        "stops": {d: stops[d] for d in dims},
        # Backward-compatible singular aliases; see the module-level note
        # above _canonicalize_meta.
        "dim": dims[0],
        "global_size": global_sizes[dims[0]],
        "start": starts[dims[0]],
        "stop": stops[dims[0]],
        "chunk_info": {
            str(name): int(size)
            for name, size in chunk_info.items()
            if name in value.dims and int(size) > 0
        },
    }
    if cart is not None:
        meta["cart"] = dict(cart)
    assign_mpi_meta(value, meta)


def set_save_chunks(
    value: xr.Dataset | xr.DataArray, save_chunks: Mapping[str, tuple[int, ...]]
) -> None:
    """Attach save_chunks to ``value``'s existing MPI distribution metadata.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Rank-local xarray object that already carries valid MPI distribution metadata (see :func:`get_mpi_meta`).
    save_chunks : mapping
        Mapping from variable name to save_chunk shape.
    Raises
    ------
    ValueError
        If ``value`` carries no valid MPI distribution metadata to attach ``save_chunks`` to.
    """
    meta = get_mpi_meta(value)
    if meta is None:
        raise ValueError("value carries no MPI distribution metadata")
    updated = dict(meta)
    updated["save_chunks"] = {
        str(name): tuple(int(length) for length in shape)
        for name, shape in save_chunks.items()
    }
    assign_mpi_meta(value, updated)


def strip_mpi_meta(value: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Return a shallow copy without MPI distribution metadata.

    Parameters
    ----------
    value : xr.Dataset | xr.DataArray
        Distributed xarray object.
    Returns
    -------
    xr.Dataset | xr.DataArray
        Shallow copy without internal MPI metadata.
    """
    output = value.copy(deep=False)
    for key in _INTERNAL_ATTRS:
        output.attrs.pop(key, None)
    if isinstance(output, xr.Dataset):
        for variable in output.variables.values():
            for key in _INTERNAL_ATTRS:
                variable.attrs.pop(key, None)
    return output


def strip_export_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``attrs`` without internal MPI bookkeeping keys.

    Parameters
    ----------
    attrs : Mapping[str, Any]
        Source attributes, e.g.
    Returns
    -------
    dict[str, Any]
        Copy of ``attrs`` with every key in :data:`_INTERNAL_ATTRS` removed.
    """
    return {key: value for key, value in attrs.items() if key not in _INTERNAL_ATTRS}


def _format_label(value: Any, limit: int = 16) -> str:
    """Return a short, fixed-width-friendly label for a coordinate value."""
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


def format_bytes(count: float) -> str:
    """Return a compact binary size label.

    Parameters
    ----------
    count : float
        Byte count.
    Returns
    -------
    str
        Human-readable binary size label.
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if count < 1024.0 or unit == "TiB":
            return f"{count:.0f}{unit}" if unit == "B" else f"{count:.1f}{unit}"
        count /= 1024.0
    return f"{count:.1f}TiB"


def should_log_partitions(runtime: MPIContext, log_partitions: bool) -> bool:
    """Collectively resolve whether to call :func:`log_partition_report`.

    Parameters
    ----------
    runtime : MPIContext
        Runtime whose communicator backs the collective.
    log_partitions : bool
        This rank's own request.
    Returns
    -------
    bool
        Identical on every rank: whether to call :func:`log_partition_report`.
    """
    return bool(runtime.comm.allreduce(bool(log_partitions), op=MPI.LOR))


def log_partition_report(
    runtime: MPIContext,
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
    """Print a structured, compact description of the rank-local partition layout.

    Parameters
    ----------
    runtime : MPIContext
        MPI runtime used for communication.
    data : xr.Dataset | xr.DataArray
        Input xarray object.
    dim : Hashable
        Dimension to operate on.
    origin : str
        Origin label for diagnostics.
    global_size : int
        Global dimension length.
    start : int
        Global inclusive start index.
    stop : int
        Global exclusive stop index.
    automatic : bool
        Whether partitioning was selected automatically.
    detail : bool
        Whether to print detailed partition diagnostics.
    """
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

    usage_str = f"{format_bytes(total)} total (Peak/Rank: {format_bytes(peak_bytes)})"

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


def log_partition_report_cartesian(
    runtime: MPIContext,
    data: xr.Dataset | xr.DataArray,
    dims: tuple[Hashable, ...],
    *,
    origin: str,
    global_sizes: Mapping[Hashable, int],
    starts: Mapping[Hashable, int],
    stops: Mapping[Hashable, int],
    grid_shape: tuple[int, ...],
    coords: tuple[int, ...],
    automatic: bool = False,
) -> None:
    """Print a compact per-axis summary of a Cartesian-topology partition.

    Parameters
    ----------
    runtime : MPIContext
        MPI runtime used for communication.
    data : xr.Dataset | xr.DataArray
        Input xarray object.
    dims : tuple[Hashable, ...]
        Dimensions to operate on.
    origin : str
        Origin label for diagnostics.
    global_sizes : Mapping[Hashable, int]
        Global sizes by partition dimension.
    starts : Mapping[Hashable, int]
        Global start indices by partition dimension.
    stops : Mapping[Hashable, int]
        Global stop indices by partition dimension.
    grid_shape : tuple[int, ...]
        Cartesian process-grid shape.
    coords : tuple[int, ...]
        Coordinate specifications.
    automatic : bool
        Whether partitioning was selected automatically.
    """
    comm = runtime.comm
    local = (
        int(comm.rank),
        tuple(int(c) for c in coords),
        tuple(int(starts[d]) for d in dims),
        tuple(int(stops[d]) for d in dims),
        int(data.nbytes),
    )
    rows = comm.gather(local, root=0)
    if comm.rank != 0 or rows is None:
        return

    total = sum(row[4] for row in rows)
    peak_bytes = max(row[4] for row in rows)
    dims_str = ", ".join(f"{str(d)!r}{' (auto)' if automatic else ''}" for d in dims)
    grid_str = "x".join(str(n) for n in grid_shape)

    border = "=" * 80
    lines = [
        border,
        f" MPI CARTESIAN PARTITION REPORT: {origin}",
        border,
        f" \U0001f539 Dimensions   : {dims_str}",
        f" \U0001f539 Process grid : {grid_str} ({comm.size} ranks)",
        " \U0001f539 Global sizes : "
        + ", ".join(f"{str(d)!s}={int(global_sizes[d])}" for d in dims),
        f" \U0001f539 Memory       : {format_bytes(total)} total "
        + f"(Peak/Rank: {format_bytes(peak_bytes)})",
        "-" * 80,
    ]
    for rank_id, rank_coords, rank_starts, rank_stops, nbytes in sorted(rows):
        slab = ", ".join(
            f"{str(d)!s}[{s}:{e}]" for d, s, e in zip(dims, rank_starts, rank_stops)
        )
        lines.append(f"   rank {rank_id:>4}  coords={rank_coords}  {slab}")
    lines.append(border)
    runtime.log("")
    runtime.log("\n".join(lines), flush=True, prefix=False)
    runtime.log("", prefix=False)


def indexer_is_scalar(indexer: Any) -> bool:
    """Return whether an isel/sel indexer selects a single position.

    Parameters
    ----------
    indexer : Any
        Value passed as an index.
    Returns
    -------
    bool
        True when ``indexer`` selects exactly one position and therefore drops its dimension, rather than keeping it with length one.
    """
    return not isinstance(indexer, (slice, list, tuple, np.ndarray, xr.DataArray))


def _coord_length(spec: Any) -> int | None:
    """Return a coordinate spec's own length, or None if it has none."""
    array = spec[1] if isinstance(spec, tuple) else spec
    array = np.asarray(array)
    return int(array.shape[0]) if array.ndim > 0 else None


def resolve_sizes(
    required_dims: Iterable[Hashable],
    sizes: Mapping[Hashable, int] | None,
    coords: Mapping[Hashable, Any] | None,
) -> dict[Hashable, int]:
    """Fill in any dimension length missing from ``sizes`` using ``coords``.

    Parameters
    ----------
    required_dims : Iterable[Hashable]
        Dimensions whose sizes must be resolved.
    sizes : Mapping[Hashable, int] | None
        Known dimension sizes.
    coords : Mapping[Hashable, Any] | None
        Coordinate specifications.
    Returns
    -------
    dict[Hashable, int]
        Resolved dimension-size mapping.
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


def localize_coord(spec: Any, global_size: int, start: int, stop: int) -> Any:
    """Slice a coordinate spec to ``[start:stop)`` if it is full-length.

    Parameters
    ----------
    spec : Any
        Coordinate specification.
    global_size : int
        Global dimension length.
    start : int
        Global inclusive start index.
    stop : int
        Global exclusive stop index.
    Returns
    -------
    Any
        Localized coordinate specification.
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


def delayed_local(
    fn: Callable[..., Any], args: tuple[Any, ...], shape: tuple[int, ...], dtype: Any
) -> Any:
    """Wrap ``fn(*args)`` as one rank's own slice, not yet computed.

    Parameters
    ----------
    fn : Callable[..., Any]
        Callable used to construct local data.
    args : tuple[Any, ...]
        Arguments passed to the callable.
    shape : tuple[int, ...]
        Expected local array shape.
    dtype : Any
        NumPy dtype.
    Returns
    -------
    Any
        Delayed Dask array for the local slice.
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

    Parameters
    ----------
    sizes : mapping
        Dimension name to global length.
    mpi_size : int
        Number of ranks the data will be spread over.
    exclude : iterable of hashable, optional
        Dimensions that must not be chosen, for example a dimension the caller intends to reduce over.
    rank : int, optional
        Calling rank, used only to gate the short-partition warning below to rank 0.
    Returns
    -------
    hashable
        Chosen dimension.

    Raises
    ------
    ValueError
        If no dimension is available.
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


def choose_partition_dims(
    sizes: Mapping[Hashable, int],
    ndims: int,
    mpi_size: int,
    *,
    exclude: Iterable[Hashable] = (),
    rank: int | None = None,
) -> tuple[Hashable, ...]:
    """Select ``ndims`` partition dimensions automatically.

    Parameters
    ----------
    sizes : mapping
        Dimension name to global length.
    ndims : int
        Number of partition dimensions to choose.
    mpi_size : int
        Number of ranks the data will be spread over.
    exclude : iterable of hashable, optional
        Dimensions that must not be chosen.
    rank : int, optional
        Forwarded to :func:`choose_partition_dim`.
    Returns
    -------
    tuple of hashable
        Chosen dimensions, longest first (ties broken by declaration order).

    Raises
    ------
    ValueError
        If ``ndims`` is less than 1 or exceeds the number of available dimensions.
    """
    if ndims < 1:
        raise ValueError(f"ndims must be at least 1; got {ndims}.")
    blocked = set(exclude)
    chosen: list[Hashable] = []
    for _ in range(ndims):
        dim = choose_partition_dim(sizes, mpi_size, exclude=blocked, rank=rank)
        chosen.append(dim)
        blocked.add(dim)
    return tuple(chosen)
