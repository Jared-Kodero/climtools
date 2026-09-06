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
- ``MPIContext``  MPI context: ``MPIContext().comm`` for the raw
  communicator and its reduce methods for collectives. Imported by name
  rather than by star import, so analysis or plotting code never
  initialises MPI by accident.

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
from .core.utils import apply_widget_css, n_cpus
from .xarray.accessors import fix_xarray

if TYPE_CHECKING:
    from .cdo import pycdo as cdo
    from .core import operator, stats, xgeo
    from .core.progress import SerialProgressBar
    from .core.utils import (
        LockedLogger,
        LockFile,
        RedirectStreams,
        exclude_key,
        locked_print,
    )
    from .mpi.context import MPIContext as MPIContext
    from .viz import cmaps, plotting

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", module=r"climtools\..*")

#: Star-import surface. ``MPIContext`` is deliberately absent. ``__all__`` is
#: exactly what ``from climtools import *`` resolves, and resolving that name
#: runs the lazy import of ``.mpi.context``, which imports mpi4py and so calls
#: ``MPI_Init`` -- as a side effect of a star import, in code that may never
#: touch MPI. Inside a Slurm allocation that enrols the process as a PMI
#: client of the job step, after which COMM_WORLD's default
#: ``MPI_ERRORS_ARE_FATAL`` handler ties an ordinary Python error to the fate
#: of the whole step. ``from climtools import MPIContext`` still works, going
#: through ``__getattr__`` exactly as before.
__all__ = [
    "DaskProgressBar",
    "LockFile",
    "LockedLogger",
    "RedirectStreams",
    "SerialProgressBar",
    "cdo",
    "cmaps",
    "exclude_key",
    "locked_print",
    "n_cpus",
    "operator",
    "plotting",
    "stats",
    "xgeo",
]


_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "stats": (".core.stats", None),
    "cdo": (".cdo.pycdo", None),
    "cmaps": (".viz.cmaps", None),
    "operator": (".core.operator", None),
    "plotting": (".viz.plotting", None),
    "xgeo": (".core.xgeo", None),
    "MPIContext": (".mpi.context", "MPIContext"),
    "LockedLogger": (".core.utils", "LockedLogger"),
    "LockFile": (".core.utils", "LockFile"),
    "RedirectStreams": (".core.utils", "RedirectStreams"),
    "SerialProgressBar": (".core.progress", "SerialProgressBar"),
    "locked_print": (".core.utils", "locked_print"),
    "exclude_key": (".core.utils", "exclude_key"),
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
    fix_xarray()
except Exception:
    ...

apply_widget_css()
dask.diagnostics.ProgressBar = DaskProgressBar
