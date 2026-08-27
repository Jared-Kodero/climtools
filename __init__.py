"""climtools: utilities for climate data analysis and plotting.

The package collects routines used repeatedly when exploring and publishing
gridded climate data with xarray:

- ``plot``      Cartopy map plotting. Entry point :func:`climtools.plot.geo`.
- ``xgeo``      Geospatial operations and NetCDF output: regridding, masking,
  transects, and local solar time.
- ``calc``      Trends, correlations, and difference-of-means testing.
- ``cmaps``     Colormap registry spanning local IPCC tables, matplotlib,
  and cmocean.
- ``cdo``       Thin xarray-aware wrapper over the CDO command-line tool.
- ``mpi``       MPI runtime: ``mpi.comm`` for the raw communicator and
  ``mpi.reduce`` for collective reductions.

Two access patterns are supported and are equivalent::

    from climtools import xgeo as xg
    xg.plot.geo(da, method="contourf")

    import climtools
    da.xgeo.plot.geo(method="contourf")

Importing the package registers the ``.xgeo`` accessor on
``xarray.DataArray`` and ``xarray.Dataset``, replaces the dask progress bar
with the styled one from :mod:`climtools.core.progress`, and, inside a
Jupyter kernel, applies the widget CSS fix and switches inline figures to
retina resolution.

Regridding requires ``xesmf``, which is imported on first use. The rest of
the package works without it.
"""

from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import warnings
from importlib import import_module
from typing import TYPE_CHECKING, Any

import dask.diagnostics

from .core.progress import DaskProgressBar
from .core.tools import apply_widget_css, n_cpus
from .xarray import accessors as _xarray_accessors  # noqa: F401

if TYPE_CHECKING:
    from .cdo import pycdo as cdo
    from .core import _operator as operator
    from .core import calc_stats as calc
    from .core import xgeo
    from .core.lib_mpi import mpi
    from .core.progress import SerialProgressBar
    from .core.tools import (
        LockedLogger,
        LockFile,
        RedirectStreams,
        locked_print,
    )
    from .viz import cmaps
    from .viz import plotting as plot

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", module=r"climtools\..*")

__all__ = [
    "DaskProgressBar",
    "LockFile",
    "LockedLogger",
    "RedirectStreams",
    "SerialProgressBar",
    "calc",
    "cdo",
    "cmaps",
    "locked_print",
    "mpi",
    "n_cpus",
    "operator",
    "plot",
    "xgeo",
]


_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "calc": (".core.calc_stats", None),
    "cdo": (".cdo.pycdo", None),
    "cmaps": (".viz.cmaps", None),
    "operator": (".core._operator", None),
    "plot": (".viz.plotting", None),
    "xgeo": (".core.xgeo", None),
    "mpi": (".core.lib_mpi", "mpi"),
    "LockedLogger": (".core.tools", "LockedLogger"),
    "LockFile": (".core.tools", "LockFile"),
    "RedirectStreams": (".core.tools", "RedirectStreams"),
    "SerialProgressBar": (".core.tools", "SerialProgressBar"),
    "locked_print": (".core.tools", "locked_print"),
}


def __getattr__(name: str) -> Any:
    """Import expensive public objects when they are first requested."""
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported objects in interactive discovery."""
    return sorted(set(globals()) | set(__all__))


try:
    from .xarray.patch import fix_xarray

    fix_xarray()
except Exception:
    ...

apply_widget_css()
dask.diagnostics.ProgressBar = DaskProgressBar
