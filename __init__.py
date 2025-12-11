from __future__ import annotations

from .cmap_funcs import *
from .cmaps_inventory import cm, cmaps
from .plot import get_cbar_axes, plot_pvalues
from .plot_theme import IPCCTheme, theme
from .pycdo import cdo
from .statistics import calc_significance
from .tools import (
    BoundingBox,
    FileLock,
    RedirectStreams,
    cp,
    cwd,
    du,
    file_kind,
    get_func_signature,
    home,
    host,
    logexc,
    logmsg,
    logobj,
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
    Daskit,
    GeoDataArray,
    GeoDataset,
    apply_func_by_time_zone,
    chunk_by_lon,
    chunk_by_tz,
    lst,
    mask,
    open_grib_datatree,
    utc_offset,
)

# Daskit = Daskit
# BoundingBox = BoundingBox
# GeoDataArray = GeoDataArray
# GeoDataset = GeoDataset
# FileLock = FileLock
# MultiProcManager = MultiProcManager
# RedirectStreams = RedirectStreams

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

__all__ = [
    "BoundingBox",
    "Daskit",
    "FileLock",
    "GeoDataArray",
    "GeoDataset",
    "IPCCTheme",
    "logexc",
    "logmsg",
    "logobj",
    "RedirectStreams",
    "apply_func_by_time_zone",
    "calc_significance",
    "cdo",
    "chunk_by_tz",
    "chunk_by_lon",
    "cm",
    "cmaps",
    "cp",
    "cwd",
    "du",
    "file_kind",
    "get_cbar_axes",
    "get_func_signature",
    "home",
    "host",
    "lst",
    "mask",
    "mkdir",
    "mv",
    "n_cpus",
    "open_grib_datatree",
    "plot_pvalues",
    "rm",
    "symlink",
    "theme",
    "timeit",
    "tmp",
    "to_numeric",
    "user",
    "utc_offset",
]
