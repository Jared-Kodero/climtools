from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from ..lib_netcdf.netcdf import append_to_netcdf, to_netcdf
    from ..viz import cmaps
    from ..viz import plotting as plot
    from . import calc_stats as calc
    from .preprocess_data import preprocess_era5
    from .progress import DaskProgressBar, SerialProgressBar
    from .tools import n_cpus
    from .xarray_utils import (
        SetupDask,
        add_local_solar_time,
        mask,
        remap,
        sel_transect,
        to_lon180,
    )

__all__ = [
    "DaskProgressBar",
    "SerialProgressBar",
    "SetupDask",
    "add_local_solar_time",
    "append_to_netcdf",
    "calc",
    "cmaps",
    "mask",
    "n_cpus",
    "plot",
    "preprocess_era5",
    "remap",
    "sel_transect",
    "to_lon180",
    "to_netcdf",
]


_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "DaskProgressBar": (".progress", "DaskProgressBar"),
    "SerialProgressBar": (".progress", "SerialProgressBar"),
    "SetupDask": (".xarray_utils", "SetupDask"),
    "add_local_solar_time": (".xarray_utils", "add_local_solar_time"),
    "append_to_netcdf": ("..lib_netcdf.netcdf", "append_to_netcdf"),
    "calc": (".calc_stats", None),
    "cmaps": ("..viz.cmaps", None),
    "mask": (".xarray_utils", "mask"),
    "n_cpus": (".tools", "n_cpus"),
    "plot": ("..viz.plotting", None),
    "preprocess_era5": (".preprocess_data", "preprocess_era5"),
    "remap": (".xarray_utils", "remap"),
    "sel_transect": (".xarray_utils", "sel_transect"),
    "to_lon180": (".xarray_utils", "to_lon180"),
    "to_netcdf": ("..lib_netcdf.netcdf", "to_netcdf"),
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
