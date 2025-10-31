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

from .cdo_py import cdo
from .corr import calc_corr
from .logs import log
from .plot import animate, cartplot, get_cbar_axes, make_cyclic, plot_pvalues
from .regridder import ESMF_RegridWeightGen, regrid_cam_se
from .tools import (
    chunk_by_dims,
    chunk_by_timezones,
    chunk_longitudes,
    close_dask,
    cp,
    cwd,
    file_type,
    get_func_signature,
    get_local_solar_time,
    get_spatiotemporal_info,
    get_UTC_offset,
    home,
    host,
    infer_time_frequency,
    interp_data,
    land_sea_mask,
    mkdir,
    mv,
    n_cpus,
    rm,
    setup_dask,
    symlink,
    timeit,
    tmp,
    type_cast,
    tz_apply_func,
    user,
)
from .utils import gen_cmap_file

gen_cmap_file()

from .cmaps_inventory import ColorMaps, cm, cmaps
from .plot_theme import IPCCTheme, theme
from .trends import calc_signicance, calc_trends, mk_trend_test, polyfit
from .xr_plot import GeoDataArray

GeoDataArray = GeoDataArray

__all__ = [
    "animate",
    "cdo",
    "n_cpus",
    "cwd",
    "ESMF_RegridWeightGen",
    "home",
    "host",
    "tmp",
    "user",
    "type_cast",
    "calc_corr",
    "calc_signicance",
    "calc_trends",
    "cartplot",
    "close_dask",
    "cp",
    "file_type",
    "get_UTC_offset",
    "get_cbar_axes",
    "get_func_signature",
    "get_local_solar_time",
    "get_spatiotemporal_info",
    "GeoDataArray",
    "infer_time_frequency",
    "interp_data",
    "land_sea_mask",
    "log",
    "make_cyclic",
    "mk_trend_test",
    "mkdir",
    "mv",
    "plot_pvalues",
    "polyfit",
    "regrid_cam_se",
    "rm",
    "setup_dask",
    "chunk_longitudes",
    "chunk_by_dims",
    "chunk_by_timezones",
    "timeit",
    "tz_apply_func",
    "symlink",
    "cmaps",
    "cm",
    "ColorMaps",
    "theme",
    "IPCCTheme",
]
