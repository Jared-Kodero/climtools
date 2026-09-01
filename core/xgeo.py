from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from ..viz import cmaps
    from ..viz import plotting as plot
    from ..xarray.io import (
        append,
        dataset_is_empty,
        empty_dataset,
        mpi_create_dataarray,
        mpi_create_dataset,
        mpi_open_dataset,
        mpi_partition_data,
        to_netcdf,
    )
    from ..xarray.utils import (
        SetupDask,
        add_local_solar_time,
        mask,
        remap,
        sel_transect,
        to_lon180,
    )
    from . import preprocess, stats
    from .progress import DaskProgressBar, SerialProgressBar
    from .utils import n_cpus

__all__ = [
    "DaskProgressBar",
    "SerialProgressBar",
    "SetupDask",
    "add_local_solar_time",
    "append",
    "cmaps",
    "dataset_is_empty",
    "empty_dataset",
    "mask",
    "mpi_create_dataarray",
    "mpi_create_dataset",
    "mpi_open_dataset",
    "mpi_partition_data",
    "n_cpus",
    "plot",
    "preprocess",
    "remap",
    "sel_transect",
    "stats",
    "to_lon180",
    "to_netcdf",
]


_LAZY_IMPORTS = {
    "DaskProgressBar": (".progress", "DaskProgressBar"),
    "SerialProgressBar": (".progress", "SerialProgressBar"),
    "SetupDask": (".xr_utils", "SetupDask"),
    "add_local_solar_time": (".xr_utils", "add_local_solar_time"),
    "append": ("..xarray.io", "append"),
    "empty_dataset": ("..xarray.io", "empty_dataset"),
    "dataset_is_empty": ("..xarray.io", "dataset_is_empty"),
    "calc": (".calc_stats", None),
    "cmaps": ("..viz.cmaps", None),
    "mask": (".xr_utils", "mask"),
    "n_cpus": (".tools", "n_cpus"),
    "plot": ("..viz.plotting", None),
    "preprocess": (".preprocess", None),
    "remap": (".xr_utils", "remap"),
    "sel_transect": (".xr_utils", "sel_transect"),
    "to_lon180": (".xr_utils", "to_lon180"),
    "to_netcdf": ("..xarray.io", "to_netcdf"),
    "mpi_open_dataset": ("..xarray.constructors", "mpi_open_dataset"),
    "mpi_create_dataarray": ("..xarray.constructors", "mpi_create_dataarray"),
    "mpi_create_dataset": ("..xarray.constructors", "mpi_create_dataset"),
    "mpi_partition_data": ("..xarray.constructors", "mpi_partition_data"),
    "MPIXarray": ("..xarray.core", "MPIXarray"),
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
