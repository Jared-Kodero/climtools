"""climtools — utilities for climate data analysis and plotting.

This package provides a collection of small helper routines used while
exploring and plotting climate datasets with xarray. The top-level package
exposes plotting helpers, statistical/trend utilities, regridding helpers
that wrap CDO/ESMF calls (when available), and a number of convenience
file/system utilities.

For a short list of the main exported names see :pydata:`__all__` below or
inspect the package interactively with ``help(climtools)``.

The package is intended for interactive analysis and reproducible notebooks.
If you plan to use parts of the package programmatically, import the
individual modules (for example ``from climtools import plot, trends``) or the
specific functions you need.
"""

from .cmap_funcs import gen_cmap_file

gen_cmap_file()
from .cmaps_inventory import CMap, cm, cmaps
from .plot import get_cbar_axes, plot_pvalues
from .plot_theme import IPCCTheme, theme
from .pycdo import cdo
from .regridder import ESMF_RegridWeightGen, regrid_cam_se
from .tools import (
    FileLock,
    LogExc,
    LogMsg,
    MultiProcManager,
    RedirectStreams,
    cp,
    cwd,
    du,
    f_type,
    get_func_signature,
    home,
    host,
    mkdir,
    mv,
    n_cpus,
    rm,
    symlink,
    timeit,
    tmp,
    to_numeric,
    user,
)
from .xgeo import (
    GeoDataArray,
    chunk_by_dims,
    chunk_by_timezones,
    chunk_longitudes,
    close_dask,
    get_local_solar_time,
    get_UTC_offset,
    land_sea_mask,
    open_grib_datatree,
    setup_dask,
    tz_apply_func,
)

GeoDataArray = GeoDataArray
FileLock = FileLock
MultiProcManager = MultiProcManager
RedirectStreams = RedirectStreams


__all__ = [
    "CMap",
    "ESMF_RegridWeightGen",
    "FileLock",
    "GeoDataArray",
    "IPCCTheme",
    "MultiProcManager",
    "RedirectStreams",
    "cdo",
    "chunk_by_dims",
    "chunk_by_timezones",
    "chunk_longitudes",
    "close_dask",
    "cm",
    "cmaps",
    "cp",
    "cwd",
    "du",
    "f_type",
    "get_UTC_offset",
    "get_cbar_axes",
    "get_func_signature",
    "get_local_solar_time",
    "home",
    "host",
    "land_sea_mask",
    "LogMsg",
    "LogExc",
    "mkdir",
    "mv",
    "n_cpus",
    "open_grib_datatree",
    "plot_pvalues",
    "regrid_cam_se",
    "rm",
    "setup_dask",
    "symlink",
    "theme",
    "timeit",
    "tmp",
    "to_numeric",
    "tz_apply_func",
    "user",
]
