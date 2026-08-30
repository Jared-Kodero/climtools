"""CF encoding helpers shared by the serial and parallel NetCDF writers.

Both writers call :func:`encode_time`/:func:`is_time_like` from this module.
The serial writer encodes each batch as it is appended, because it hands raw
numeric buffers to ``netCDF4``. The parallel writer encodes once, on rank 0,
while building the file schema: only rank 0 ever holds the real Dataset or
DataArray (every other rank supplies :func:`~climtools.core.xgeo.empty_dataset`
in its place, per the ``climtools.mpi`` calling convention), so there is a
single time axis to encode and no cross-rank negotiation of CF units is
needed. The resulting numeric values and ``units``/``calendar`` attrs are
included in the schema broadcast to every rank alongside the rest of the
file layout.
"""

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
        ``True`` when the variable requires CF numeric encoding before it can
        be written through the ``netCDF4`` interface.
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
        Variable to encode. Variables of any other dtype are returned
        unchanged.
    units : str or None, optional
        Target CF units, for example ``"seconds since 1970-01-01"``. When
        ``None``, datetime axes use seconds since the Unix epoch and other
        axes fall back to the variable's own encoding.
    calendar : str or None, optional
        Target CF calendar. When ``None``, datetime64 axes use
        ``"proleptic_gregorian"``.

    Returns
    -------
    xarray.DataArray
        Numeric variable carrying ``units`` and, where applicable,
        ``calendar`` in both ``attrs`` and ``encoding``.

    Notes
    -----
    Passing ``units`` explicitly is required when appending to an existing
    file. Re-deriving units from the values of one batch would place a second
    time origin under the single ``units`` attribute already stored in the
    file, which corrupts the axis without raising.
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

    Notes
    -----
    The copy is what makes this safe to call from a writer. Assigning encoded
    variables back into the caller's dataset would leave the caller holding an
    integer time axis after the write returned.
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
