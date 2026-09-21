from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from ..viz import cmaps
    from ..viz import plotting as plot
    from ..xarray.core import MPIXarray
    from ..xarray.io import (
        create_distributed_dataarray,
        create_distributed_dataset,
        is_distributed_empty,
        empty_distributed_dataset,
        open_distributed_dataset,
        distribute_data,
        nc_append,
        to_netcdf,
    )
    from ..xarray.utils import (
        SetupDask,
        XNpyStore,
        add_local_solar_time,
        fill_nan_2d,
        mask,
        regrid,
        sel_transect,
        to_lon180,
    )
    from . import preprocess, stats
    from .progress import DaskProgressBar, SerialProgressBar
    from .shared_mem import SharedMemoryObject
    from .utils import N_CPUS

__all__ = [
    "N_CPUS",
    "DaskProgressBar",
    "MPIXarray",
    "SerialProgressBar",
    "SetupDask",
    "XNpyStore",
    "SharedMemoryObject",
    "add_local_solar_time",
    "cmaps",
    "fill_nan_2d",
    "mask",
    "create_distributed_dataarray",
    "create_distributed_dataset",
    "is_distributed_empty",
    "empty_distributed_dataset",
    "open_distributed_dataset",
    "distribute_data",
    "nc_append",
    "plot",
    "preprocess",
    "regrid",
    "sel_transect",
    "stats",
    "to_lon180",
    "to_netcdf",
]


_LAZY_IMPORTS = {
    "DaskProgressBar": (".progress", "DaskProgressBar"),
    "SerialProgressBar": (".progress", "SerialProgressBar"),
    "SetupDask": ("..xarray.utils", "SetupDask"),
    "XNpyStore": ("..xarray.utils", "XNpyStore"),
    "add_local_solar_time": ("..xarray.utils", "add_local_solar_time"),
    "nc_append": ("..xarray.io", "nc_append"),
    "empty_distributed_dataset": ("..xarray.io", "empty_distributed_dataset"),
    "is_distributed_empty": ("..xarray.io", "is_distributed_empty"),
    "stats": (".stats", None),
    "cmaps": ("..viz.cmaps", None),
    "mask": ("..xarray.utils", "mask"),
    "N_CPUS": (".utils", "N_CPUS"),
    "plot": ("..viz.plotting", None),
    "preprocess": (".preprocess", None),
    "regrid": ("..xarray.utils", "regrid"),
    "sel_transect": ("..xarray.utils", "sel_transect"),
    "to_lon180": ("..xarray.utils", "to_lon180"),
    "fill_nan_2d": ("..xarray.utils", "fill_nan_2d"),
    "to_netcdf": ("..xarray.io", "to_netcdf"),
    "open_distributed_dataset": ("..xarray.io", "open_distributed_dataset"),
    "create_distributed_dataarray": ("..xarray.io", "create_distributed_dataarray"),
    "create_distributed_dataset": ("..xarray.io", "create_distributed_dataset"),
    "distribute_data": ("..xarray.io", "distribute_data"),
    "MPIXarray": ("..xarray.core", "MPIXarray"),
    "SharedMemoryObject": (".shared_mem", "SharedMemoryObject"),
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
