from __future__ import annotations

__all__ = [
    "append_to_netcdf",
    "GeoDataArray",
    "GeoDataset",
    "RedirectStreams",
    "SetupDask",
    "calc_significance",
    "cdo",
    "cmaps",
    "cp",
    "cwd",
    "du",
    "file_kind",
    "get_cax",
    "get_fsig",
    "home",
    "host",
    "logexc",
    "mkdir",
    "mv",
    "n_cpus",
    "plot_pvalues",
    "plot_quiver",
    "theming",
    "rm",
    "symlink",
    "timeit",
    "tmp",
    "user",
]

import sys

from . import _cmaps
from . import cmaps as cmaps
from . import pycdo as cdo
from . import theming as theming

# import cmaps as _cmaps --- IGNORE ---
from ._cmaps import *
from .plotting import get_cax, plot_pvalues, plot_quiver
from .statistics import calc_significance
from .tools import (
    RedirectStreams,
    cp,
    cwd,
    du,
    file_kind,
    fix_vscode_widget,
    get_fsig,
    home,
    host,
    logexc,
    mkdir,
    mv,
    n_cpus,
    rm,
    symlink,
    timeit,
    tmp,
    user,
)
# from .update import _self_update
from .xgeo import (
    GeoDataArray,
    GeoDataset,
    SetupDask,
    append_to_netcdf,
)

# _self_update()

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

if "ipykernel" in sys.modules:
    fix_vscode_widget()
    ...
