from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from ..viz import cmaps
    from ..viz import plotting as plot
    from ..xarray.core import MPIXarray
    from ..xarray.io import (
        SharedMemoryObject,
        XNpyStore,
        create_distributed_dataarray,
        create_distributed_dataset,
        distribute_data,
        empty_distributed_dataset,
        is_distributed_empty,
        nc_append,
        open_distributed_dataset,
        open_xnpy,
        to_netcdf,
        to_xnpy,
    )
    from ..xarray.utils import (
        SetupDask,
        add_local_solar_time,
        fillgaps,
        mask,
        regrid,
        sel_transect,
        to_lon180,
    )
    from . import preprocess, stats
    from .progress import DaskProgressBar, SerialProgressBar
    from .utils import N_CPUS

__all__ = [
    "N_CPUS",
    "DaskProgressBar",
    "MPIXarray",
    "SerialProgressBar",
    "SetupDask",
    "SharedMemoryObject",
    "XNpyStore",
    "add_local_solar_time",
    "cmaps",
    "create_distributed_dataarray",
    "create_distributed_dataset",
    "distribute_data",
    "empty_distributed_dataset",
    "fillgaps",
    "is_distributed_empty",
    "mask",
    "nc_append",
    "open_distributed_dataset",
    "open_xnpy",
    "plot",
    "preprocess",
    "regrid",
    "sel_transect",
    "stats",
    "to_lon180",
    "to_netcdf",
    "to_xnpy",
]


_LAZY_IMPORTS = {
    "add_local_solar_time": ("..xarray.utils", "add_local_solar_time"),
    "cmaps": ("..viz.cmaps", None),
    "create_distributed_dataarray": ("..xarray.io", "create_distributed_dataarray"),
    "create_distributed_dataset": ("..xarray.io", "create_distributed_dataset"),
    "DaskProgressBar": (".progress", "DaskProgressBar"),
    "distribute_data": ("..xarray.io", "distribute_data"),
    "empty_distributed_dataset": ("..xarray.io", "empty_distributed_dataset"),
    "fillgaps": ("..xarray.utils", "fillgaps"),
    "is_distributed_empty": ("..xarray.io", "is_distributed_empty"),
    "mask": ("..xarray.utils", "mask"),
    "MPIXarray": ("..xarray.core", "MPIXarray"),
    "N_CPUS": (".utils", "N_CPUS"),
    "nc_append": ("..xarray.io", "nc_append"),
    "open_distributed_dataset": ("..xarray.io", "open_distributed_dataset"),
    "open_xnpy": ("..xarray.io", "open_xnpy"),
    "plot": ("..viz.plotting", None),
    "preprocess": (".preprocess", None),
    "regrid": ("..xarray.utils", "regrid"),
    "sel_transect": ("..xarray.utils", "sel_transect"),
    "SerialProgressBar": (".progress", "SerialProgressBar"),
    "SetupDask": ("..xarray.utils", "SetupDask"),
    "SharedMemoryObject": ("..xarray.io", "SharedMemoryObject"),
    "stats": (".stats", None),
    "to_lon180": ("..xarray.utils", "to_lon180"),
    "to_netcdf": ("..xarray.io", "to_netcdf"),
    "to_xnpy": ("..xarray.io", "to_xnpy"),
    "XNpyStore": ("..xarray.io", "XNpyStore"),
}


def __getattr__(name: str) -> Any:
    """Import a re-exported implementation when it is first requested."""
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = import_module(module_name, __package__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily re-exported objects in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
