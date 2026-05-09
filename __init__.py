from __future__ import annotations

__all__ = [
    "BoundingBox",
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
    "plot",
    "rm",
    "symlink",
    "timeit",
    "tmp",
    "user",
]

import sys

from .init_cmap import *
from .plotting import get_cax, plot_pvalues, plot_quiver
from .pycdo import cdo
from .statistics import calc_significance
from .theming import cmaps, plot
from .tools import (
    BoundingBox,
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

# import cmaps as _cmaps --- IGNORE ---
# from .update import _self_update
from .xgeo import (
    GeoDataArray,
    GeoDataset,
    SetupDask,
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
