"""xgeo: geospatial analysis, plotting, and distributed xarray for climate data.

The top level exposes the geospatial and plotting API:

- ``plot``      Cartopy map plotting. Entry point :func:`xgeo.plot.geo`.
- ``regrid``, ``mask``, ``fillgaps``, ``sel_transect``, ``to_lon180``,
  ``add_local_solar_time``  Geospatial operations on xarray objects.
- ``to_netcdf``, ``nc_append``, ``to_xnpy``, ``open_xnpy``  Array output.
- ``stats``     Trends, correlations, and difference-of-means testing.
- ``preprocess``  Dataset-specific preprocessing (ERA5, IMERG, CMORPH, GPCP).
- ``cmaps``     Colormap registry spanning local IPCC tables, matplotlib,
  and cmocean.
- ``MPIXarray`` and the ``*_distributed_*`` constructors for MPI-parallel
  xarray.

and the shared utilities: ``LockedLogger``, ``LockFile``, ``RedirectStreams``,
``locked_print``, ``exclude_key``, ``nproc``, ``SerialProgressBar``,
``DaskProgressBar``, ``operator`` and ``MPIContext``. The CDO wrapper is the
module :mod:`xgeo.cdo.pycdo`.

Two access patterns are supported and are equivalent::

    import xgeo as xg
    xg.plot.geo(da, method="contourf")

    import xgeo
    da.xgeo.plot.geo(method="contourf")

Importing the package registers the ``.xgeo`` accessor on
``xarray.DataArray`` and ``xarray.Dataset``, replaces the dask progress bar
with the styled one from :mod:`xgeo.core.progress`, and, inside a Jupyter
kernel, applies the widget CSS fix and switches inline figures to retina
resolution.

Public names are resolved on first attribute access through ``__getattr__``,
so each implementation module loads when first used. Regridding requires
``xesmf``, which is imported on first use. The rest of the package works
without it.
"""

from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import warnings
from importlib import import_module
from typing import TYPE_CHECKING, Any

import dask.diagnostics

from .core.progress import DaskProgressBar
from .core.utils import apply_widget_css
from .xarray.accessors import fix_xarray

if TYPE_CHECKING:
    from .core import operator, preprocess, stats
    from .core.io import (
        SharedMemoryObject,
        open_xnpy_dataframe,
        open_xnpy_dataset,
        open_xnpy_ndarray,
        to_xnpy,
    )
    from .core.progress import SerialProgressBar
    from .core.utils import (
        LockedLogger,
        LockFile,
        RedirectStreams,
        exclude_key,
        locked_print,
        nproc,
    )
    from .mpi.context import MPIContext  # noqa: F401
    from .viz import cmaps
    from .viz import plotting as plot
    from .xarray.core import MPIXarray  # noqa: F401
    from .xarray.io import (
        create_distributed_dataarray,  # noqa: F401
        create_distributed_dataset,  # noqa: F401
        distribute_data,  # noqa: F401
        empty_distributed_dataset,  # noqa: F401
        is_distributed_empty,  # noqa: F401
        nc_append,  # noqa: F401
        open_distributed_dataset,  # noqa: F401
        to_netcdf,  # noqa: F401
    )
    from .xarray.utils import (
        SetupDask,
        add_local_solar_time,
        fillgaps,
        mask,
        regrid,
        sel_transect,
        to_lon180,
    )

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", module=r"xgeo\..*")

#: Star-import surface. The names that import mpi4py on first access are
#: deliberately absent: ``MPIContext``, ``MPIXarray``, ``to_netcdf``,
#: ``nc_append``, ``distribute_data`` and the ``*_distributed_*`` functions.
#: ``__all__`` is exactly what ``from xgeo import *`` resolves, and resolving
#: one of them imports mpi4py and so calls ``MPI_Init`` as a side effect of a
#: star import, in code that may never touch MPI. Inside a Slurm allocation
#: that enrols the process as a PMI client of the job step, after which
#: COMM_WORLD's default ``MPI_ERRORS_ARE_FATAL`` handler ties an ordinary
#: Python error to the fate of the whole step. ``xgeo.<name>`` and
#: ``from xgeo import <name>`` still work for every name.
__all__ = [
    "DaskProgressBar",
    "LockFile",
    "LockedLogger",
    "RedirectStreams",
    "SerialProgressBar",
    "SetupDask",
    "SharedMemoryObject",
    "add_local_solar_time",
    "cmaps",
    "exclude_key",
    "fillgaps",
    "locked_print",
    "mask",
    "nproc",
    "open_xnpy",
    "open_xnpy_dataframe",
    "open_xnpy_dataset",
    "open_xnpy_ndarray",
    "operator",
    "plot",
    "preprocess",
    "regrid",
    "sel_transect",
    "stats",
    "to_lon180",
    "to_xnpy",
]


_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "LockFile": (".core.utils", "LockFile"),
    "LockedLogger": (".core.utils", "LockedLogger"),
    "MPIContext": (".mpi.context", "MPIContext"),
    "MPIXarray": (".xarray.core", "MPIXarray"),
    "nproc": (".core.utils", "nproc"),
    "RedirectStreams": (".core.utils", "RedirectStreams"),
    "SerialProgressBar": (".core.progress", "SerialProgressBar"),
    "SetupDask": (".xarray.utils", "SetupDask"),
    "SharedMemoryObject": (".core.io", "SharedMemoryObject"),
    "add_local_solar_time": (".xarray.utils", "add_local_solar_time"),
    "cmaps": (".viz.cmaps", None),
    "create_distributed_dataarray": (".xarray.io", "create_distributed_dataarray"),
    "create_distributed_dataset": (".xarray.io", "create_distributed_dataset"),
    "distribute_data": (".xarray.io", "distribute_data"),
    "empty_distributed_dataset": (".xarray.io", "empty_distributed_dataset"),
    "exclude_key": (".core.utils", "exclude_key"),
    "fillgaps": (".xarray.utils", "fillgaps"),
    "is_distributed_empty": (".xarray.io", "is_distributed_empty"),
    "locked_print": (".core.utils", "locked_print"),
    "mask": (".xarray.utils", "mask"),
    "nc_append": (".xarray.io", "nc_append"),
    "open_distributed_dataset": (".xarray.io", "open_distributed_dataset"),
    "open_xnpy_ndarray": (".core.io", "open_xnpy_ndarray"),
    "open_xnpy_dataframe": (".core.io", "open_xnpy_dataframe"),
    "open_xnpy_dataset": (".core.io", "open_xnpy_dataset"),
    "operator": (".core.operator", None),
    "plot": (".viz.plotting", None),
    "preprocess": (".core.preprocess", None),
    "regrid": (".xarray.utils", "regrid"),
    "sel_transect": (".xarray.utils", "sel_transect"),
    "stats": (".core.stats", None),
    "to_lon180": (".xarray.utils", "to_lon180"),
    "to_netcdf": (".xarray.io", "to_netcdf"),
    "to_xnpy": (".core.io", "to_xnpy"),
}


def __getattr__(name: str) -> Any:
    """Import a public object when it is first requested."""
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
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


try:
    fix_xarray()
except Exception:
    ...

apply_widget_css()
dask.diagnostics.ProgressBar = DaskProgressBar

# from .update import _self_update

# _self_update()
