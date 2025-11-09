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

from .corr import calc_corr
from .plot import animate, cartplot, get_cbar_axes, make_cyclic, plot_pvalues
from .pycdo import cdo
from .regridder import ESMF_RegridWeightGen, regrid_cam_se
from .tools import (
    FileLock,
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
from .utils import gen_cmap_file, logmsg

gen_cmap_file()

from .cmaps_inventory import ColorMaps, cm, cmaps
from .plot_theme import IPCCTheme, theme
from .trends import calc_signicance, calc_trends, mk_trend_test, polyfit
from .xgeo import (
    GeoDataArray,
    chunk_by_dims,
    chunk_by_timezones,
    chunk_longitudes,
    close_dask,
    get_local_solar_time,
    get_spatiotemporal_info,
    get_UTC_offset,
    infer_time_frequency,
    interp_data,
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
    "ColorMaps",
    "ESMF_RegridWeightGen",
    "FileLock",
    "GeoDataArray",
    "IPCCTheme",
    "MultiProcManager",
    "RedirectStreams",
    "animate",
    "calc_corr",
    "calc_signicance",
    "calc_trends",
    "cartplot",
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
    "get_spatiotemporal_info",
    "home",
    "host",
    "infer_time_frequency",
    "interp_data",
    "land_sea_mask",
    "logmsg",
    "make_cyclic",
    "mk_trend_test",
    "mkdir",
    "mv",
    "n_cpus",
    "open_grib_datatree",
    "plot_pvalues",
    "polyfit",
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
