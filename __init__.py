"""
Module: climtools

Description:
    This module provides utility functions and constants for data analysis,
    processing, and file operations, with support for xarray, Dask, and CDO-based workflows.

Author:
    Jared M. Kodero

Contents:

Data Analysis Functions:
------------------------
- calc_trends: Computes trends using vectorized Mann-Kendall (preferred over xr_polyfit).
- xr_polyfit: Fits a polynomial to xarray data using xarray's polyfit method.
- calc_corr: Calculates correlation between datasets.
- calc_signicance: Computes significance of trends in data.
- mk_trend_test: Creates a trend significance test.

Time and Spatial Utilities:
---------------------------
- tz_apply_func: Applies a function across time zones in a dataset.
- get_UTC_offset: Computes the UTC offset for a given datetime and location.
- get_local_solar_time: Calculates local solar time based on longitude.
- land_sea_mask: Applies a land-sea mask to datasets.
- split_by_15_deg: Splits datasets into 15° longitudinal bands.
- split_data_by_dims: Splits datasets along specified dimensions.
- split_data_by_timezones: Splits datasets based on time zone offsets.
- xr_interp_data: Interpolates xarray datasets.

CDO Operations:
---------------
- cdo_interp_data: Interpolates datasets using Climate Data Operators (CDO).
- cdo_mergetime: Merges time dimensions in netCDF files using CDO.
- cdo_pack_nc: Compresses netCDF files using CDO.

File & System Utilities:
------------------------
- cp: Copies files.
- mv: Moves files.
- rm: Removes files.
- file_type: Determines the type of a file (NetCDF, GRIB, etc.).
- mkdir: Creates a directory if it does not exist.
- timeit: Decorator to measure execution time of a function.
- setup_dask: Configures Dask for parallel computing.
- close_dask: Closes an active Dask client.

Miscellaneous Utilities:
------------------------
- log: Logging utility for standard output and debugging.
- dask_setup: Alias for setup_dask.
- notebook_auto_reload: Enables autoreloading in Jupyter notebooks (if defined elsewhere).

"""

from .corr import calc_corr
from .handle_nc import (
    cdo_interp_data,
    cdo_mergetime,
    cdo_pack_nc,
    get_local_solar_time,
    get_UTC_offset,
    land_sea_mask,
    split_by_15_deg,
    split_data_by_dims,
    split_data_by_timezones,
    tz_apply_func,
    xr_interp_data,
)
from .log import line_break, log
from .my_paths import (
    CWD,
    DATA_DIR,
    DEEPS_SHARE_DIR,
    ERA5_DIR,
    ERA5_RAW_DIR,
    FIG_DIR,
    HOME,
    HOST,
    SCRATCH_DIR,
    SCRIPTS_DIR,
    TMP,
    USER,
    WORK_DIR,
)
from .res import get_spatiotemporal_info, infer_time_frequency
from .tools import (
    CPU_COUNT,
    close_dask,
    cp,
    file_type,
    log,
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
    "DATA_DIR",
    "DEEPS_SHARE_DIR",
    "ERA5_DIR",
    "ERA5_RAW_DIR",
    "FIG_DIR",
    "HOME",
    "HOST",
    "SCRATCH_DIR",
    "SCRIPTS_DIR",
    "TMP",
    "USER",
    "WORK_DIR",
    "calc_corr",
    "calc_signicance",
    "calc_trends",
    "cdo_interp_data",
    "cdo_mergetime",
    "cdo_pack_nc",
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
    "rm",
    "setup_dask",
    "split_by_15_deg",
    "split_data_by_dims",
    "split_data_by_timezones",
    "timeit",
    "tz_apply_func",
    "xr_interp_data",
    "xr_polyfit",
]
