import operator as op
from typing import Callable

import pandas as pd
import xarray as xr
from xarray.coding.times import encode_cf_datetime

attributes = {
    "t2m": {"units": "degC", "factor": 273.15, "operand": op.sub},
    "t": {"units": "degC", "factor": 273.15, "operand": op.sub},
    "skt": {"units": "degC", "factor": 273.15, "operand": op.sub},
    "d2m": {"units": "degC", "factor": 273.15, "operand": op.sub},
    "sp": {"units": "hPa", "factor": 100.0, "operand": op.truediv},
    "crr": {"units": "mm/hr", "factor": 3600.0, "operand": op.mul},
    "mtpr": {"units": "mm/hr", "factor": 3600.0, "operand": op.mul},
    "z": {"units": "m", "factor": 9.80665, "operand": op.truediv},
    "swvl1": {"units": "%", "factor": 100.0, "operand": op.mul},
    "sst": {"units": "degC", "factor": 273.15, "operand": op.sub},
    "tp": {"units": "mm", "factor": 1000.0, "operand": op.mul},
}


def operand(operation: Callable, data: xr.DataArray, factor: float) -> xr.DataArray:
    return operation(data, factor)


def norm_time(da: xr.DataArray) -> xr.DataArray:

    time = xr.DataArray(
        pd.to_datetime(da.values),
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs,
    )

    _, out_units, out_calendar = encode_cf_datetime(
        time,
        units="seconds since 1970-01-01 00:00:00",
        calendar="proleptic_gregorian",
        dtype="int64",
    )

    time.encoding.update({"units": out_units, "calendar": out_calendar})
    return time


def get_valid_time(ds: xr.Dataset) -> xr.Dataset:
    ds["forecast_initial_time"] = pd.to_datetime(ds["forecast_initial_time"].values)
    ds["forecast_hour"] = pd.to_timedelta(ds["forecast_hour"].values, unit="h")

    valid_times = ds["forecast_initial_time"] + ds["forecast_hour"]

    ds = ds.stack(time=("forecast_initial_time", "forecast_hour"))

    ds = ds.reset_index("time", drop=True)  # Remove MultiIndex
    ds = ds.assign_coords(
        time=valid_times.stack(time=("forecast_initial_time", "forecast_hour")).data
    )

    ds["time"] = norm_time(ds["time"])
    ds = ds.sortby("time")

    return ds


def preprocess_era5(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess an ERA5 dataset to standardize variable names, dimensions, and attributes."""

    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray Dataset.")

    if "pressure_level" in ds.dims:
        ds = ds.rename({"pressure_level": "plev"})

    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    if "forecast_initial_time" in ds.dims and "time" not in ds.dims:
        ds = get_valid_time(ds)
    else:
        ds["time"] = norm_time(ds["time"])

    ds = ds.drop_vars(
        ["number", "expver", "step", "surface", "valid_time"], errors="ignore"
    )

    standard_dims = ("time", "plev", "lat", "lon")
    dims = [dim for dim in standard_dims if dim in ds.dims]

    ds["lon"] = ((ds["lon"] + 180) % 360) - 180

    ds["lat"].attrs = {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    }
    ds["lon"].attrs = {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    }

    ds = ds.sortby(dims)

    for data_var in ds.data_vars:
        if data_var in ds.coords:
            continue  # Skip coordinate variables

        units = ds[data_var].attrs.get("units", None)
        long_name = ds[data_var].attrs.get("long_name", data_var)
        standard_name = ds[data_var].attrs.get("standard_name", data_var)
        attribute = attributes.get(data_var, None)
        ds[data_var].attrs = {}
        if attribute is not None:
            ds[data_var] = operand(
                attribute["operand"], ds[data_var], attribute["factor"]
            )
            units = attribute["units"]

        if data_var in ["crr", "mtpr"]:
            ds[data_var] = ds[data_var].clip(min=0, keep_attrs=True)

        ds[data_var].attrs["long_name"] = long_name
        ds[data_var].attrs["standard_name"] = standard_name
        ds[data_var].attrs["units"] = units

    ds = ds.transpose(*dims)
    return ds
