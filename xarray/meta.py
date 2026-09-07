"""Manage MPI distribution metadata for xarray objects."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..mpi.context import MPIContext

MPI_META = "mpi_meta"
#: Lightweight partition flag stripped before NetCDF export.
PARTITIONED_ATTR = "mpi_partitioned"
#: Centralize internal attrs that must be removed before export or user-level
#: comparisons.
_INTERNAL_ATTRS = frozenset({MPI_META, PARTITIONED_ATTR})
# Partition identity is defined by dimensions, global sizes, and local ownership bounds.
# Plural keys support multi-D partitions; singular keys mirror the first axis for
# compatibility.


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


def mpp_get_meta(value: xr.Dataset | xr.DataArray) -> dict[str, Any] | None:
    """Return validated MPI distribution metadata."""
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


def mpp_set_meta(value: xr.Dataset | xr.DataArray, meta: Mapping[str, Any]) -> None:
    """Attach an already-built ``meta`` dict to ``value`` and its variables."""
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
            f"{name} must map partition dimensions to values for multi-D partitioning."
        )
    return {dims[0]: int(cast("int", value))}


def mpp_update_meta(
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
    mpp_set_meta(value, meta)


def set_save_chunks(
    value: xr.Dataset | xr.DataArray, save_chunks: Mapping[str, tuple[int, ...]]
) -> None:
    """Attach save_chunks to ``value``'s existing MPI distribution metadata.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Rank-local xarray object that already carries valid MPI distribution metadata (see :func:`mpp_get_meta`).
    save_chunks : mapping
        Mapping from variable name to save_chunk shape.

    Raises
    ------
    ValueError
        If ``value`` carries no valid MPI distribution metadata to attach ``save_chunks`` to.

    """
    meta = mpp_get_meta(value)
    if meta is None:
        raise ValueError("value carries no MPI distribution metadata")
    updated = dict(meta)
    updated["save_chunks"] = {
        str(name): tuple(int(length) for length in shape)
        for name, shape in save_chunks.items()
    }
    mpp_set_meta(value, updated)


def reattach_meta_after_collapse(
    result: xr.Dataset | xr.DataArray, meta: Mapping[str, Any], dim: str
) -> xr.Dataset | xr.DataArray:
    """Carry forward metadata for the dims that survive ``dim`` collapsing away.

    Shared by ops that resolve one global label along ``dim`` and
    replicate the answer to every rank (`mpp_sel_scalar`, `mpp_isel_scalar`,
    `isel`'s singleton-repartition case). ``dim`` itself gets no
    replacement value; other active partition dimensions carry over
    unchanged.

    Returns
    -------
    xr.Dataset | xr.DataArray
        ``result`` with metadata reattached for any surviving partition
        dimension, or unchanged if none survive.

    """
    from .chunks import prune_chunk_info

    remaining_dims = tuple(
        d for d in meta["dims"] if d != dim and d in getattr(result, "dims", ())
    )
    if not remaining_dims:
        return result
    mpp_update_meta(
        result,
        dim=remaining_dims,
        global_size={d: int(meta["global_sizes"][d]) for d in remaining_dims},
        start={d: int(meta["starts"][d]) for d in remaining_dims},
        stop={d: int(meta["stops"][d]) for d in remaining_dims},
        chunk_info=prune_chunk_info(meta["chunk_info"], result),
        # Dropping a Cartesian axis invalidates the cached topology; rebuild it lazily
        # when needed.
        cart=None,
    )
    return result


def strip_mpi_meta(value: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Return a shallow copy without MPI distribution metadata."""
    output = value.copy(deep=False)
    for key in _INTERNAL_ATTRS:
        output.attrs.pop(key, None)
    if isinstance(output, xr.Dataset):
        for variable in output.variables.values():
            for key in _INTERNAL_ATTRS:
                variable.attrs.pop(key, None)
    return output


def strip_export_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``attrs`` without internal MPI bookkeeping keys."""
    return {key: value for key, value in attrs.items() if key not in _INTERNAL_ATTRS}


def mpp_should_log_partitions(mpi_context: MPIContext, log_partitions: bool) -> bool:
    """Resolve partition logging collectively.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context.
    log_partitions : bool
        Local logging request.

    Returns
    -------
    bool
        True on every rank if any rank requested logging.
    """
    return bool(mpi_context.comm.allreduce(bool(log_partitions), op=MPI.LOR))


def mpp_log_partition_report(
    mpi_context: MPIContext,
    data: xr.Dataset | xr.DataArray,
    dim: Hashable | tuple[Hashable, ...],
    *,
    origin: str,
    global_size: int | Mapping[Hashable, int],
    start: int | Mapping[Hashable, int],
    stop: int | Mapping[Hashable, int],
    grid_shape: tuple[int, ...] | None = None,
    coords: tuple[int, ...] | None = None,
    automatic: bool = False,
    detail: bool = True,
) -> None:
    """Print a structured, compact description of a rank-local partition layout (1D or Cartesian)."""
    comm = mpi_context.comm
    is_cartesian = grid_shape is not None or isinstance(dim, tuple | list)

    if is_cartesian:
        dims = dim if isinstance(dim, tuple | list) else (dim,)
        starts_map = start if isinstance(start, Mapping) else {dims[0]: start}
        stops_map = stop if isinstance(stop, Mapping) else {dims[0]: stop}

        local = (
            int(comm.rank),
            tuple(int(c) for c in (coords or ())),
            tuple(int(starts_map[d]) for d in dims),
            tuple(int(stops_map[d]) for d in dims),
        )
    else:
        local = (
            int(comm.rank),
            int(start),
            int(stop),
        )

    rows = comm.gather(local, root=0)
    if comm.rank != 0 or rows is None:
        return

    border = "=" * 80
    separator = "-" * 80
    lines = [border]

    if is_cartesian:
        dims = dim if isinstance(dim, tuple | list) else (dim,)
        global_sizes_map = (
            global_size if isinstance(global_size, Mapping) else {dims[0]: global_size}
        )
        dims_str = ", ".join(
            f"{str(d)!r}{' (auto)' if automatic else ''}" for d in dims
        )
        grid_str = "x".join(str(n) for n in (grid_shape or (comm.size,)))

        lines.extend(
            [
                f" MPI CARTESIAN PARTITION REPORT: {origin}",
                border,
                f" 🔹 Dimensions   : {dims_str}",
                f" 🔹 Process grid : {grid_str} ({comm.size} ranks)",
                " 🔹 Global sizes : "
                + ", ".join(f"{str(d)!s}={int(global_sizes_map[d])}" for d in dims),
            ]
        )

        if detail:
            lines.append(separator)
            slice_widths = [
                max(
                    len(f"{d} slice"),
                    *(len(f"{row[2][i]}:{row[3][i]}") for row in rows),
                )
                for i, d in enumerate(dims)
            ]
            count_widths = [
                max(len(f"{d} n"), *(len(str(row[3][i] - row[2][i])) for row in rows))
                for i, d in enumerate(dims)
            ]
            coord_width = (
                max(len("coords"), *(len(str(row[1])) for row in rows))
                if coords is not None or any(row[1] for row in rows)
                else 0
            )

            header_parts = ["   " + f"{'rank':>4}"]
            if coord_width > 0:
                header_parts.append(f"{'coords':>{coord_width}}")
            for i, d in enumerate(dims):
                header_parts.append(f"{f'{d} slice':>{slice_widths[i]}}")
                header_parts.append(f"{f'{d} n':>{count_widths[i]}}")

            lines.append("  ".join(header_parts))
            lines.append(separator)

            for row in rows:
                rank_id, rank_coords, rank_starts, rank_stops = row
                row_parts = ["   " + f"{rank_id:>4}"]
                if coord_width > 0:
                    row_parts.append(f"{rank_coords!s:>{coord_width}}")
                for i in range(len(dims)):
                    slice_str = f"{rank_starts[i]}:{rank_stops[i]}"
                    count_val = rank_stops[i] - rank_starts[i]
                    row_parts.append(f"{slice_str:>{slice_widths[i]}}")
                    row_parts.append(f"{count_val:>{count_widths[i]}}")
                lines.append("  ".join(row_parts))
    else:
        counts = [row[2] - row[1] for row in rows]
        idle = sum(1 for count in counts if count == 0)

        other = " ".join(
            f"{name!s}={int(length)}"
            for name, length in data.sizes.items()
            if name != dim
        )
        chunk_text = "  ".join(
            f"{name!s}={max(int(size) for size in chunks)}"
            for name, chunks in (data.chunks or {}).items()
        )

        dim_str = f"{str(dim)!r}{' (auto)' if automatic else ''}"
        split_str = (
            f"{min(counts)}/rank"
            if min(counts) == max(counts)
            else f"{min(counts)}-{max(counts)}/rank"
        )
        if idle:
            split_str += f" (IDLE={idle})"

        lines.extend(
            [
                f" MPI PARTITION REPORT: {origin}",
                border,
                f" 🔹 Dimension    : {dim_str}",
                f" 🔹 Global Size  : {global_size} (Ranks: {comm.size})",
                f" 🔹 Split        : {split_str}",
                f" 🔹 Shape        : {other or 'scalar'}",
                f" 🔹 Chunks/Rank  : {chunk_text or 'unchunked'}",
            ]
        )

        if detail:
            slice_width = max(
                len("slice"), *(len(f"{row[1]}:{row[2]}") for row in rows)
            )
            count_width = max(len("n"), *(len(str(row[2] - row[1])) for row in rows))

            lines.extend(
                [
                    separator,
                    f"   {'rank':>4}  {'slice':>{slice_width}}  {'n':>{count_width}}",
                    separator,
                ]
            )

            for row in rows:
                slice_str = f"{row[1]}:{row[2]}"
                count_val = row[2] - row[1]
                lines.append(
                    f"   {row[0]:>4}  {slice_str:>{slice_width}}  {count_val:>{count_width}}"
                )

    lines.append(border)

    mpi_context.log("")
    mpi_context.log("\n".join(lines), flush=True, prefix=False)
    mpi_context.log("", prefix=False)


def indexer_is_scalar(indexer: Any) -> bool:
    """Return whether an isel/sel indexer selects a single position.

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
    """Fill in any dimension length missing from ``sizes`` using ``coords``."""
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
        raise ValueError(f"Missing sizes for dimensions: {sorted(str(d) for d in missing)}.")
    return resolved


def localize_coord(spec: Any, global_size: int, start: int, stop: int) -> Any:
    """Slice a coordinate spec to ``[start:stop)`` if it is full-length."""
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
    """Wrap ``fn(*args)`` as one rank's own slice, not yet computed."""
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

    return dim
