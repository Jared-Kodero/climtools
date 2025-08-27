"""
climtools: Utilities for Climate Data Analysis and Visualization

Author:
    Jared M. Kodero

Overview:
    climtools provides a suite of functions and constants for climate data analysis,
    including trend detection, correlation, time zone handling, spatial operations,
    plotting with CartoPy, file management, and integration with xarray, Dask, and CDO.

Main Features
=============

Data Analysis:
--------------
- calc_trends: Vectorized Mann-Kendall trend estimation (preferred over xr_polyfit).
- xr_polyfit: Polynomial fitting for xarray DataArrays.
- calc_corr: Correlation calculation between datasets.
- calc_signicance: Statistical significance of trends.
- mk_trend_test: Mann-Kendall trend significance test.

Time & Spatial Utilities:
-------------------------
- tz_apply_func: Apply functions across time zones in datasets.
- get_UTC_offset: Compute UTC offset for datetime/location.
- get_local_solar_time: Calculate local solar time from longitude.
- land_sea_mask: Apply land-sea mask to data.
- split_by_15_deg: Split data into 15° longitude bands.
- split_data_by_dims: Split data along specified dimensions.
- split_data_by_timezones: Split data by time zone offsets.
- xr_interp_data: Interpolate xarray datasets.

Plotting (CartoPy):
-------------------
- cartplot: Quick CartoPy map plotting for 2D xarray DataArrays.
- create_map_figure: Create CartoPy map figures.
- get_cbar_axes: Utility for colorbar axes.
- plot_p_values: Plot p-values on maps.

Example:
    >>> import numpy as np, xarray as xr
    >>> da = xr.DataArray(
            np.random.rand(10, 10),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-90, 90, 10), "lon": np.linspace(-180, 180, 10)}
        )
    >>> cartplot(
            data=da,
            plot_type="default",
            projection="PlateCarree",
            global_extent=True,
            figsize=(12, 6),
            cmap="balance",
            cbar_label="Units"
        )

CDO Operations:
---------------
- cdo_interp_data: Interpolate with Climate Data Operators (CDO).
- cdo_mergetime: Merge time dimensions in NetCDF files.
- cdo_pack_nc: Compress NetCDF files.

File & System Utilities:
------------------------
- cp, mv, rm: File copy, move, and remove.
- file_type: Detect file type (NetCDF, GRIB, etc.).
- mkdir: Create directories.
- timeit: Function timing decorator.
- setup_dask, close_dask: Dask parallel computing setup/teardown.

Miscellaneous:
--------------
- log, line_break: Logging and formatting utilities.
- Notebook/Jupyter support: notebook_auto_reload (if available).

Notes:
------
- All plotting functions require 2D xarray.DataArray with latitude and longitude.
- Designed for extensibility and integration with scientific Python workflows.

"""

from .corr import calc_corr
from .handle_nc import (
    cdo_interp_data,
    cdo_mergetime,
    get_local_solar_time,
    get_spatiotemporal_info,
    get_UTC_offset,
    infer_time_frequency,
    land_sea_mask,
    split_by_15_deg,
    split_data_by_dims,
    split_data_by_timezones,
    tz_apply_func,
    xr_interp_data,
)
from .logs import line_break, log
from .plot import cartplot, create_map_figure, get_cbar_axes, plot_p_values, see_data
from .regridder import ESMF_RegridWeightGen, regrid_cam_se
from .tools import (
    CPU_COUNT,
    CWD,
    HOME,
    HOST,
    TMPDIR,
    USER,
    close_dask,
    cp,
    file_type,
    get_func_signature,
    mkdir,
    mv,
    rm,
    setup_dask,
    timeit,
)
from .trends import calc_signicance, calc_trends, mk_trend_test, xr_polyfit

__all__ = [
    "CPU_COUNT",
    "CWD",
    "ESMF_RegridWeightGen",
    "HOME",
    "HOST",
    "TMPDIR",
    "USER",
    "calc_corr",
    "calc_signicance",
    "calc_trends",
    "cdo_interp_data",
    "cdo_mergetime",
    "close_dask",
    "cp",
    "file_type",
    "get_spatiotemporal_info",
    "get_UTC_offset",
    "get_local_solar_time",
    "infer_time_frequency",
    "land_sea_mask",
    "log",
    "mk_trend_test",
    "mkdir",
    "mv",
    "line_break",
    "regrid_cam_se",
    "rm",
    "setup_dask",
    "split_by_15_deg",
    "split_data_by_dims",
    "split_data_by_timezones",
    "timeit",
    "tz_apply_func",
    "xr_interp_data",
    "xr_polyfit",
    "cartplot",
    "create_map_figure",
    "get_cbar_axes",
    "plot_p_values",
    "get_func_signature",
    "see_data",
]
