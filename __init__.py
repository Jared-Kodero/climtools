from __future__ import annotations

__all__ = [
    "cdo",
    "cmaps",
    "xgeo",
]

import sys

import dask

from . import cmaps as cmaps
from . import pycdo as cdo
from . import xgeo

# import cmaps as _cmaps --- IGNORE ---
from .tools import (
    fix_vscode_widget,
)
# from .update import _self_update

# _self_update()

if "ipykernel" in sys.modules:
    fix_vscode_widget()
    dask.diagnostics.ProgressBar = xgeo.DaskProgressBar

    import matplotlib_inline as plt_inline

    plt_inline.backend_inline.set_matplotlib_formats("retina")


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
