"""Compatibility hooks for upstream xarray behavior."""

from __future__ import annotations

from typing import Any

import xarray as xr

_MPIXARRAY_TYPES: tuple[type[Any], ...] = globals().get("_MPIXARRAY_TYPES", ())
_DATAARRAY_BINARY_OP = getattr(
    xr.DataArray._binary_op, "__mpixarray_original__", xr.DataArray._binary_op
)
_DATASET_BINARY_OP = getattr(
    xr.Dataset._binary_op, "__mpixarray_original__", xr.Dataset._binary_op
)


def register_mpixarray_type(cls: type[Any]) -> None:
    """Register an ``MPIXarray`` class for reflected binary dispatch."""
    global _MPIXARRAY_TYPES

    if cls not in _MPIXARRAY_TYPES:
        _MPIXARRAY_TYPES += (cls,)


def _dataarray_binary_op(
    self: xr.DataArray, other: Any, f: Any, reflexive: bool = False
) -> Any:
    """Defer mixed binary operations to ``MPIXarray``."""
    if isinstance(other, _MPIXARRAY_TYPES):
        return NotImplemented
    return _DATAARRAY_BINARY_OP(self, other, f, reflexive)


def _dataset_binary_op(
    self: xr.Dataset,
    other: Any,
    f: Any,
    reflexive: bool = False,
    join: Any = None,
) -> Any:
    """Defer mixed binary operations to ``MPIXarray``."""
    if isinstance(other, _MPIXARRAY_TYPES):
        return NotImplemented
    return _DATASET_BINARY_OP(self, other, f, reflexive, join)


def install_binary_op_compatibility() -> None:
    """Install binary-operation dispatch hooks for ``MPIXarray``."""
    _dataarray_binary_op.__mpixarray_original__ = _DATAARRAY_BINARY_OP
    _dataset_binary_op.__mpixarray_original__ = _DATASET_BINARY_OP
    xr.DataArray._binary_op = _dataarray_binary_op
    xr.Dataset._binary_op = _dataset_binary_op
