from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from .statistics import *

# ---- Plot callback ----
from .tools import tmp

warnings.filterwarnings("ignore")
_script_dir = Path(__file__).resolve().parent
_mask_cache = {}  # cache for remapped masks to avoid redundant computations


def append_to_netcdf(
    file: Path,
    da: xr.DataArray,
    name: str = None,
) -> None:
    """Append a DataArray to an existing NetCDF file

    Parameters
    ----------
    file : Path
        Path to a NetCDF4 file opened with read/write access.
    da : xr.DataArray
        DataArray to append. Must have dimensions that already exist in the file.
    name : str, optional
        Name of the variable to create in the NetCDF file. If None, uses da.name.
    """

    with nc.Dataset(file, mode="r+") as ncf:
        varname = name or da.name

        # Overwrite values if the variable was created on a previous run.
        if varname in ncf.variables:
            ncf.variables[varname][:] = da.values

        else:
            missing = [d for d in da.dims if d not in ncf.dimensions]
            if missing:
                raise ValueError(
                    f"Cannot write {varname} to {file}: missing dimensions {missing}"
                )

            ncvar = ncf.createVariable(
                varname=varname,
                datatype=da.dtype,
                dimensions=da.dims,
                zlib=True,
                complevel=4,
            )

            for attr_name, attr_val in da.attrs.items():
                ncvar.setncattr(attr_name, attr_val)

            ncvar[:] = da.values


def mask(
    data: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | Path | None = None,
    keep: str = "land",
    parallel: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Apply a land-sea mask to an xarray object.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input data containing latitude and longitude dimensions.
    mask : xr.DataArray or Path, optional
        Land-sea mask. If a DataArray is provided it is used directly.
        If a Path is provided it is opened as a dataset. If None, the
        default ERA5 0.25° mask bundled with the package is used.
    keep : the data variable to keep in the mask dataset
    parallel : bool, default False
        Enable dask-based parallel masking.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Masked data.
    """

    if mask is None:
        mask = _script_dir / "data" / "mask" / "era5_0.25_mask"

    if isinstance(mask, Path):
        mask = xr.open_dataset(mask)[keep].load()

    if isinstance(mask, xr.Dataset):
        raise TypeError(
            "Mask must be an xarray.DataArray. If a Dataset is provided, specify the variable to use with the 'keep' parameter."
        )

    for coord in ("lat", "lon"):
        if coord not in data.dims:
            raise ValueError(f"Input grid must contain '{coord}' dimension.")
        if coord not in data.dims:
            raise ValueError(f"Output grid must contain '{coord}' dimension.")

    lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
    lat_min, lat_max = float(data.lat.min()), float(data.lat.max())
    lon_step = float(data["lon"].diff("lon").mean().values)
    lat_step = float(data["lat"].diff("lat").mean().values)
    cache_key = (lon_min, lon_max, lat_min, lat_max, lon_step, lat_step)

    if cache_key not in _mask_cache:
        data = data.sortby(["lat", "lon"])
        # slice the mask to the same lat/lon + a small buffer to ensure coverage
        lat_slice = slice(lat_min, lat_max)
        lon_slice = slice(lon_min, lon_max)
        mask_da = mask.sel(lat=lat_slice, lon=lon_slice)
        mask_da = xe_remap(mask_da, data, method="nearest_s2d", parallel=parallel)
        _mask_cache[cache_key] = mask_da

    mask_da = _mask_cache[cache_key]

    new_obj = data.where(mask_da == 1, other=np.nan)
    return new_obj


def mask_data(**kwargs) -> xr.DataArray | xr.Dataset:
    return mask(**kwargs)


def calc_local_solar_time(
    data: xr.Dataset | xr.DataArray, *, lon="lon"
) -> xr.Dataset | xr.DataArray:
    """
    Calculate the local solar time for the dataset based on the longitude coordinate.
    The local solar time is calculated as the UTC time plus the longitude offset.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The input dataset or data array containing a 'time' coordinate and a longitude coordinate.
    lon : str, default 'lon'
        The name of the longitude coordinate in the dataset.
        This coordinate is used to calculate the local solar time offset.


    """

    if lon not in data:
        raise ValueError(f"Dataset must contain '{lon}' coordinate.")

    offset = ((data[lon] + 180) % 360 - 180) * (24 / 360)
    offset = offset.round() * pd.Timedelta(hours=1)

    lst = data["time"] + offset
    lst.attributes["long_name"] = "Local Solar Time"
    lst.attributes["standard_name"] = "local_solar_time"

    return data.assign_coords({"lst": lst})


def _to_lon_180(
    data: xr.Dataset | xr.DataArray, lon: str = "lon"
) -> xr.Dataset | xr.DataArray:
    """
    Standardize the longitude coordinates of an xarray object to be within the range [-180, 180).

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The input dataset or data array containing a longitude coordinate.
    lon : str, default 'lon'
        The name of the longitude coordinate in the dataset.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The dataset or data array with standardized longitude coordinates.
    """
    if lon not in data:
        raise ValueError(f"Dataset must contain '{lon}' coordinate.")

    data = data.copy()
    data[lon] = (data[lon] + 180) % 360 - 180
    data = data.sortby(lon)
    return data


def xe_remap(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray,
    method: Literal[
        "bilinear",
        "conservative",
        "conservative_normed",
        "patch",
        "nearest_s2d",
        "nearest_d2s",
    ] = "bilinear",
    parallel: bool = False,
) -> xr.Dataset:
    """
    Remap source dataset to the grid of the destination dataset using xesmf.
    """
    for coord in ("lat", "lon"):
        if coord not in grid_in.dims:
            raise ValueError(f"Input grid must contain '{coord}' dimension.")
        if coord not in grid_out.dims:
            raise ValueError(f"Output grid must contain '{coord}' dimension.")

    _in = xr.Dataset(
        coords={
            "lat": grid_in["lat"],
            "lon": grid_in["lon"],
        }
    )

    _out = xr.Dataset(
        coords={
            "lat": grid_out["lat"],
            "lon": grid_out["lon"],
        }
    )
    if parallel:
        _out["dummy"] = xr.DataArray(
            np.ones((_out.lat.size, _out.lon.size)),
            dims=("lat", "lon"),
            coords={"lat": _out.lat, "lon": _out.lon},
        )

        _out = _out.chunk({"lat": "auto", "lon": "auto"})

    _id = [
        f"{method}",
        f"{_in.sizes['lat']}x{_in.sizes['lon']}",
        f"{_out.sizes['lat']}x{_out.sizes['lon']}",
    ]

    weight_file = tmp / ("_".join(_id) + ".nc")
    reuse = weight_file.exists()

    regridder = xe.Regridder(
        _in,
        _out,
        method=method,
        parallel=parallel,
        filename=str(weight_file),
        reuse_weights=reuse,
    )

    out = regridder(grid_in)
    return out
