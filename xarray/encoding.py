"""Encode xarray time coordinates for NetCDF output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cftime
import numpy as np
from xarray.coding.times import encode_cf_datetime, encode_cf_timedelta

if TYPE_CHECKING:
    from typing import Any

    import xarray as xr


def is_cftime(da: xr.DataArray) -> bool:
    """Report whether an object-dtype variable holds cftime datetimes.

    Parameters
    ----------
    da : xarray.DataArray
        Variable to inspect.
    Returns
    -------
    bool
        ``True`` when the first element is a :class:`cftime.datetime`.
    """
    if da.dtype != object:
        return False
    values = np.asarray(da.values).reshape(-1)
    return values.size > 0 and isinstance(values[0], cftime.datetime)


def is_time_like(da: xr.DataArray) -> bool:
    """Report whether a variable carries datetime, cftime or timedelta values.

    Parameters
    ----------
    da : xarray.DataArray
        Variable to inspect.
    Returns
    -------
    bool
        ``True`` when the variable requires CF numeric encoding before it can be written through the ``netCDF4`` interface.
    """
    return (
        np.issubdtype(da.dtype, np.datetime64)
        or np.issubdtype(da.dtype, np.timedelta64)
        or is_cftime(da)
    )


def encode_time(
    da: xr.DataArray,
    units: str | None = None,
    calendar: str | None = None,
) -> xr.DataArray:
    """Encode datetime64, cftime or timedelta64 values to CF numeric values.

    Parameters
    ----------
    da : xarray.DataArray
        Variable to encode.
    units : str or None, optional
        Target CF units, for example ``"seconds since 1970-01-01"``.
    calendar : str or None, optional
        Target CF calendar.
    Returns
    -------
    xarray.DataArray
        Numeric variable carrying ``units`` and, where applicable, ``calendar`` in both ``attrs`` and ``encoding``.
    """
    if np.issubdtype(da.dtype, np.datetime64) and not is_cftime(da):
        target_units = units or "seconds since 1970-01-01 00:00:00"
        target_calendar = calendar or "proleptic_gregorian"
        num, out_units, out_calendar = encode_cf_datetime(
            da,
            units=target_units,
            calendar=target_calendar,
            dtype=np.dtype("int64"),
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units, "calendar": out_calendar})
        encoded.encoding.update({"units": out_units, "calendar": out_calendar})
        return encoded

    if is_cftime(da):
        num, out_units, out_calendar = encode_cf_datetime(
            da,
            units=units or da.encoding.get("units"),
            calendar=calendar or da.encoding.get("calendar"),
            dtype=da.encoding.get("dtype"),
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units, "calendar": out_calendar})
        encoded.encoding.update({"units": out_units, "calendar": out_calendar})
        return encoded

    if np.issubdtype(da.dtype, np.timedelta64):
        num, out_units = encode_cf_timedelta(
            da,
            units=units or da.encoding.get("units"),
            dtype=da.encoding.get("dtype"),
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units})
        encoded.encoding.update({"units": out_units})
        return encoded

    return da


def encode_dataset_time(ds: xr.Dataset) -> xr.Dataset:
    """Encode every time-like variable of a dataset without touching the input.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to encode.
    Returns
    -------
    xarray.Dataset
        Shallow copy in which time-like variables carry CF numeric values.
    """
    out = ds.copy()
    replacements: dict[Any, xr.DataArray] = {}
    for name in list(out.variables):
        variable = out[name]
        if is_time_like(variable):
            replacements[name] = encode_time(variable)
    for name, variable in replacements.items():
        out[name] = variable
    return out


__all__ = [
    "encode_dataset_time",
    "encode_time",
    "is_cftime",
    "is_time_like",
]
