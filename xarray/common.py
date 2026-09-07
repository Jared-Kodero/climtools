"""Provide shared reduction constants and dtype helpers."""

from __future__ import annotations

from collections.abc import Hashable
from functools import cache
from typing import Any, NamedTuple, cast

import numpy as np
import xarray as xr
from mpi4py.util import dtlib as _dtlib

from ..mpi.mpi_init import MPI

_OP_LIST: tuple[tuple[Any, str], ...] = (
    (MPI.SUM, "SUM"),
    (MPI.PROD, "PROD"),
    (MPI.MIN, "MIN"),
    (MPI.MAX, "MAX"),
    (MPI.LAND, "LAND"),
    (MPI.LOR, "LOR"),
)

MPI_REDUCIBLE_KINDS = "biufc"

# Check rank agreement before collectives so mismatched plans fail instead of
# deadlocking.
CHECK_COLLECTIVE_AGREEMENT = True


def op_name(op: MPI.Op) -> str:
    """Return a rank-stable label for an MPI reduction operation."""
    for candidate, name in _OP_LIST:
        if op == candidate:
            return name
    return "OP"


@cache
def mpi_representable(dtype_string: str) -> bool:
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
def partial_dtype(
    dtype_string: str, operation: str, skipna: bool | None
) -> np.dtype[Any]:
    """Return the dtype of a rank-local xarray reduction."""
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


def extreme_identity(dtype: np.dtype[Any], *, minimum: bool) -> Any:
    """Return the neutral value for a minimum or maximum reduction."""
    kind = dtype.kind
    if kind == "b":
        return bool(minimum)
    if kind in "iu":
        limits = np.iinfo(dtype)
        return limits.max if minimum else limits.min
    if kind == "f":
        return np.asarray(np.inf if minimum else -np.inf, dtype=dtype).item()
    name = "minimum" if minimum else "maximum"
    raise TypeError(f"MPI {name} is not defined for {dtype} data.")


class PlanEntry(NamedTuple):
    """Describe one variable in a rank-independent reduction plan.

    Attributes
    ----------
    name : Hashable
        Variable name.
    dims : tuple[Hashable, ...]
        Reduced dimensions present on the variable.
    distributed : bool
        Whether the reduction requires MPI communication.
    dtype : numpy.dtype
        Variable dtype.
    shape : tuple[tuple[str, int], ...]
        Global dimensions and lengths surviving the reduction.
    comm_axes : frozenset[str]
        Partition axes included in the collective.
    replica_count : int
        Number of replicated copies included in a SUM collective.
    """

    name: Hashable
    dims: tuple[Hashable, ...]
    distributed: bool
    dtype: np.dtype[Any]
    shape: tuple[tuple[str, int], ...]
    comm_axes: frozenset[str] = frozenset()
    replica_count: int = 1
