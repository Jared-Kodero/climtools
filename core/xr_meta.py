"""Shared MPI metadata helpers for distributed xarray objects."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from .xr_mpi import MPIRuntime

MPI_META = "mpi_meta"
# The subset of a partition's metadata that decides whether two partitions
# describe the same rank-local ownership. chunk_info is deliberately
# excluded: it records how the split was computed for the benefit of a
# later redistribute(..., chunk_info=...), not the ownership itself, so two
# partitions with different chunk_info but identical dim/global_size/start/
# stop still own the exact same data and are still equal.
_PARTITION_KEYS: tuple[str, ...] = ("dim", "global_size", "start", "stop")


def _partitions_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two partition metadata mappings own the same slice."""
    return all(left.get(key) == right.get(key) for key in _PARTITION_KEYS)


def _validate_mpi_meta(
    value: xr.Dataset | xr.DataArray,
    meta: Any,
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
    """Print a compact description of the rank-local partition layout.

    Every rank contributes its own bounds through a single gather and rank 0
    prints the result. Logging from every rank instead produces interleaved,
    unordered lines that are unreadable at scale.

    The report is deliberately narrow. The full local shape is identical on
    every rank apart from the partitioned dimension, so printing it once in
    the header conveys the same information as repeating it on every row and
    keeps the table inside a standard terminal width.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime owning the communicator.
    data : xarray.Dataset or xarray.DataArray
        Rank-local object after partitioning.
    dim : hashable
        Partitioned dimension.
    origin : str
        Name of the operation that produced the partition.
    global_size : int
        Global length of ``dim``.
    start, stop : int
        Global half-open interval owned by this rank.
    automatic : bool, optional
        Whether ``dim`` was selected automatically.
    detail : bool, optional
        Print the per-rank table in addition to the summary line. When False
        only the two summary lines are printed, which is what long runs that
        open many files usually want.
    """
    comm = runtime.comm
    first, last = _edge_labels(data, dim)
    local = (
        int(comm.rank),
        int(start),
        int(stop),
        first,
        last,
        int(data.nbytes),
    )
    rows = comm.gather(local, root=0)
    if comm.rank != 0 or rows is None:
        return

    counts = [row[2] - row[1] for row in rows]
    total = sum(row[5] for row in rows)
    idle = sum(1 for count in counts if count == 0)
    other = " x ".join(
        f"{name!s}:{int(length)}" for name, length in data.sizes.items() if name != dim
    )

    lines = [
        f"{origin}  dim={str(dim)!r}{' (auto)' if automatic else ''}"
        + f"  size={global_size}  ranks={comm.size}"
        + f"  split={min(counts)}-{max(counts)}/rank"
        + (f"  IDLE={idle}" if idle else ""),
        f"  held: {other or 'scalar'}"
        + f"   {_format_bytes(total)} total"
        + f", {_format_bytes(max(row[5] for row in rows))} peak/rank",
    ]

    if detail:
        # Calculate max widths
        slice_width = max(len("slice"), *(len(f"{row[1]}:{row[2]}") for row in rows))
        first_width = max(len("first"), *(len(str(row[3])) for row in rows))
        last_width = max(len("last"), *(len(str(row[4])) for row in rows))

        # Apply padding to headers
        lines.append(
            f"  {'rank':>4}  {'slice':>{slice_width}}  {'n':>6}  "
            + f"({'first':>{first_width}}, {'last':>{last_width}})"
        )

        # Apply padding to row values
        for row in rows:
            slice_str = f"{row[1]}:{row[2]}"
            first_str = str(row[3])
            last_str = str(row[4])

            lines.append(
                f"  {row[0]:>4}  {slice_str:>{slice_width}}  {row[2] - row[1]:>6}  "
                + f"({first_str:>{first_width}}, {last_str:>{last_width}})"
            )

    runtime.log("\n".join(lines), flush=True)
