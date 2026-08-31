"""Provide shared reduction constants and dtype helpers."""

from __future__ import annotations

from collections.abc import Hashable
from functools import cache
from typing import Any, NamedTuple, cast

import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib as _dtlib

import xarray as xr

_OP_LIST: tuple[tuple[Any, str], ...] = (
    (MPI.SUM, "SUM"),
    (MPI.PROD, "PROD"),
    (MPI.MIN, "MIN"),
    (MPI.MAX, "MAX"),
    (MPI.LAND, "LAND"),
    (MPI.LOR, "LOR"),
)

_MPI_REDUCIBLE_KINDS = "biufc"

# Verify that every rank entered a reduction with the same per-variable plan
# before any buffer collective is posted. The check costs one small object
# allgather per reduction and converts an otherwise silent deadlock into an
# immediate exception. Set to False only for micro-benchmarking.
CHECK_COLLECTIVE_AGREEMENT = True


def _op_name(op: MPI.Op) -> str:
    """Return a rank-stable label for an MPI reduction operation."""
    for candidate, name in _OP_LIST:
        if op == candidate:
            return name
    return "OP"


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


def _extreme_identity(dtype: np.dtype[Any], *, minimum: bool) -> Any:
    """Return the neutral element for a min/max reduction: the dtype's max
    value (for a minimum) or min value (for a maximum), so that combining it
    with any real value leaves the real value unchanged."""
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
    dims : tuple of Hashable
        Reduced dimensions present on the variable.
    distributed : bool
        Whether this variable's reduction requires any MPI communication
        at all -- equivalent to ``bool(comm_axes)``.
    dtype : numpy.dtype
        Variable dtype.
    shape : tuple of tuple of (str, int)
        Global dimensions and lengths that survive the reduction.
    comm_axes : frozenset of str
        Partition dimensions this variable's reduction must communicate
        over: the partition dimensions actually being reduced away
        (``dim`` names on this variable that are also active partition
        dimensions), plus any partition dimension this variable is
        replicated along (present in the object's partition dimensions
        but absent from this variable's own dims). Empty when the
        reduction is entirely rank-local. See
        :meth:`ReductionPlanningMixin._resolve_comm`.
    replica_count : int
        Size of the replicated-axis process subgroup this variable is
        duplicated across (the product of the process-grid extent along
        every dimension in ``comm_axes`` that is *not* actually being
        reduced on this variable, i.e. a dimension the variable is merely
        replicated along). 1 for the common case of no replicated axes
        (including every one-dimensional partition). A collective SUM
        over ``comm_axes`` counts each replica ``replica_count`` times,
        so :meth:`ReductionPlanningMixin._comm_reduce` divides a
        ``MPI.SUM`` result by it to undo the duplication.
    """

    name: Hashable
    dims: tuple[Hashable, ...]
    distributed: bool
    dtype: np.dtype[Any]
    shape: tuple[tuple[str, int], ...]
    comm_axes: frozenset[str] = frozenset()
    replica_count: int = 1
