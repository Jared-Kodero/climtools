"""climtools: utilities for climate data analysis and plotting.

The package collects the routines used repeatedly when exploring and
publishing gridded climate data with xarray:

- ``plot``      Cartopy map plotting. Entry point :func:`climtools.plotting.geo`.
- ``xgeo``      Geospatial operations: regridding, masking, transects, local solar time.
- ``calc``      Trends, correlations and difference-of-means testing.
- ``cmaps``     Colormap registry spanning local IPCC tables, matplotlib and cmocean.
- ``cdo``       Thin xarray-aware wrapper over the CDO command-line tool.
- ``theme``     Publication styling for matplotlib and seaborn.

Two access patterns are supported and are equivalent::

    from climtools import xgeo as xg
    xg.plot.geo(da, method="contourf")

    import climtools            # registers the accessor
    da.xgeo.plot.geo(method="contourf")

Importing the package registers the ``.xgeo`` accessor on ``xarray.DataArray``
and ``xarray.Dataset``, replaces the dask progress bar with the styled one from
:mod:`climtools.progress`, and, inside a Jupyter kernel, applies the widget CSS
fix and switches inline figures to retina resolution.

Regridding requires ``xesmf``, which is imported on first use. The rest of the
package works without it.
"""

from __future__ import annotations

import dask.diagnostics

from . import calc_stats as calc
from . import cmaps as cmaps
from . import plotting as plot
from . import pycdo as cdo
from . import theming as theme
from . import xgeo as xgeo
from .accessors import *
from .progress import DaskProgressBar, SerialProgressBar
from .tools import *

__all__ = [
    "DaskProgressBar",
    "SerialProgressBar",
    "calc",
    "cdo",
    "cmaps",
    "n_cpus",
    "plot",
    "redirect_streams",
    "theme",
    "xgeo",
]


# from .update import _self_update

# _self_update()

try:
    from ._typing import fix_xarray

    modified = fix_xarray()
    if modified:
        print("Modified", modified)
except Exception:
    ...


# Setup

apply_widget_css()
set_preview_quality()
dask.diagnostics.ProgressBar = DaskProgressBar
