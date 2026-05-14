from __future__ import annotations

import logging
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import cartopy.mpl.geoaxes as cgeo
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .plotting import (
    MapPlot,
    animate,
    make_cyclic,
    mapplot,
)
from .statistics import *

# ---- Plot callback ----
from .tools import n_cpus
from .xrext import (
    _to_lon_180,
    append_to_netcdf,
    calc_local_solar_time,
    mask,
    mask_data,
    xe_remap,
)

warnings.filterwarnings("ignore")
_dask_client = None  # global variable to hold the Dask client instance
_dask_cluster = None  # global variable to hold the Dask cluster instance


class SetupDask:
    """
    A class to manage Dask client and cluster setup for parallel computations.
    It provides methods to start and close a Dask client and cluster with specified configurations.
    """

    def __init__(
        self,
        workers: int = 1,
        threads_per_worker: int = n_cpus,
        processes: bool = False,
        filter_warnings: bool = True,
        memory_limit="auto",
    ):
        self.cluster = None
        self.client = None
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self.processes = processes
        self.filter_warnings = filter_warnings
        self.memory_limit = memory_limit

    def close(self):
        """
        Close the active Dask client and cluster if they exist.
        This is useful for cleaning up resources when done with Dask computations.
        """
        if self.client and self.cluster:
            self.client.close()
            self.cluster.close()
            self.client = None
            self.cluster = None

    def start(
        self,
    ):
        """
        Start a Dask client and cluster with the specified configuration.
        If a client and cluster already exist, it will reuse them instead of creating new ones.

        Parameters
        ----------

        workers: int
            Number of worker processes to start in the Dask cluster.
        threads_per_worker: int
            Number of threads to use per worker process.
        processes: bool
            Whether to use separate processes for workers (True) or threads (False).
        filter_warnings: bool
            Whether to filter out Dask-related warnings for cleaner output.
        memory_limit: int or str
            Memory limit for each worker (e.g., '2GB' or 2048). Set to 0 for no limit.

        """

        global _dask_client, _dask_cluster

        if self.client is not None:
            return self.client

        from dask.distributed import Client, LocalCluster

        if self.filter_warnings:
            silence_level = logging.ERROR
        else:
            silence_level = logging.WARN

        if _dask_client and _dask_cluster:
            self.client = _dask_client
            self.cluster = _dask_cluster
        else:
            self.cluster = LocalCluster(
                n_workers=self.workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
                silence_logs=silence_level,
                processes=self.processes,
                dashboard_address=":8787",
            )
            self.client = Client(self.cluster)
            _dask_client = self.client
            _dask_cluster = self.cluster

        return self.client

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class GeoMixin:
    __slots__ = ()

    def _validate_da(self) -> None:
        if isinstance(self, (xr.Dataset, GeoDataset)):
            msg = f"This method requires a DataArray. Select one of {list(self.data_vars)} from the Dataset."
            raise TypeError(msg)

    @wraps(make_cyclic)
    def add_cyclic_point(self, lon: str = "lon") -> GeoDataArray:
        return type(self)(make_cyclic(self, lon))

    @wraps(_to_lon_180)
    def to_lon_180(self, lon: str = "lon") -> GeoDataArray:
        res = _to_lon_180(self, lon=lon)
        return type(self)(res)

    @wraps(calc_local_solar_time)
    def local_solar_time(self, *, lon="lon") -> GeoDataArray:
        res = calc_local_solar_time(data=self, lon=lon)
        return type(self)(res)

    @wraps(xe_remap)
    def remap(
        self,
        grid_out: xr.Dataset | xr.DataArray,
        method: Literal[
            "bilinear",
            "conservative",
            "conservative_normed",
            "patch",
            "nearest_s2d",
            "nearest_d2s",
        ] = "bilinear",
        parallel: bool = False,
    ) -> GeoDataArray:
        _params = locals()
        _ = _params.pop("self")
        remapped = xe_remap(self, **_params)
        return type(self)(remapped)

    @wraps(mask)
    def mask(
        self,
        mask: xr.DataArray | Path | None = None,
        keep: str = "land",
        parallel: bool = False,
    ) -> GeoDataArray:
        _params = locals()
        _ = _params.pop("self")
        masked = mask_data(data=self, **_params)
        return type(self)(masked)

    @wraps(mapplot)
    def mapplot(
        self,
        x: str = None,
        y: str = None,
        ax: cgeo.GeoAxes = None,
        projection: Literal[
            "PlateCarree",
            "Mercator",
            "Robinson",
            "Mollweide",
            "Orthographic",
            "LambertConformal",
            "AlbersEqualArea",
            "Stereographic",
            "NorthPolarStereo",
            "SouthPolarStereo",
        ] = "PlateCarree",
        central_longitude: float = None,
        central_latitude: float = None,
        global_extent: bool = False,
        set_extent: tuple[float, float, float, float] = None,
        figsize: tuple[float, float] = None,
        # Plot appearance
        method: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow"
        ] = "default",
        norm: Any = None,
        cmap: str | LinearSegmentedColormap | ListedColormap = None,
        vmin: float = None,
        vmax: float = None,
        levels: int | list = None,
        extend: str = None,
        cyclic: bool = False,
        robust: bool = False,
        rasterized: bool = False,
        title: str = "",
        orientation: Literal["vertical", "horizontal"] = "vertical",
        add_colorbar: bool = True,
        drawedges: bool = False,
        cbar_label: str = None,
        # Map features
        gridlines: bool = False,
        coastlines: bool = True,
        borders: bool = True,
        states: bool = True,
        ocean: bool = True,
        land: bool = True,
        lakes: bool = False,
        rivers: bool = False,
        p_values: xr.DataArray = None,
        p_value_kwargs: dict = None,
        u_component: xr.DataArray = None,
        v_component: xr.DataArray = None,
        quiver_kwargs: dict = None,
        **kwargs,
    ) -> MapPlot:
        _params = locals()
        _ = _params.pop("self")
        self._validate_da()
        return mapplot(da=self, **_params)

    @wraps(animate)
    def animate(
        self,
        outfile: Path | str = None,
        dim: str = "time",
        *,
        indices: tuple | list | np.ndarray = None,
        quality: Literal["low", "medium", "high"] = "medium",
        fps: int = 10,
        parallel: bool = True,
        # Spatial configuration
        x: str = None,
        y: str = None,
        projection: Literal[
            "PlateCarree",
            "Mercator",
            "Robinson",
            "Mollweide",
            "Orthographic",
            "LambertConformal",
            "AlbersEqualArea",
            "Stereographic",
            "NorthPolarStereo",
            "SouthPolarStereo",
        ] = "PlateCarree",
        global_extent: bool = False,
        set_extent: tuple[float, float, float, float] = None,
        figsize: tuple[float, float] = None,
        central_longitude: float = None,
        central_latitude: float = None,
        # Plot appearance
        method: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow"
        ] = "default",
        cmap: str | LinearSegmentedColormap | ListedColormap = None,
        norm: Any = None,
        vmin: float = None,
        vmax: float = None,
        levels: int | list[int] = None,
        extend: str = None,
        cyclic: bool = False,
        robust: bool = False,
        rasterized: bool = False,
        title: str = "",
        orientation: Literal["vertical", "horizontal"] = "vertical",
        add_colorbar: bool = True,
        drawedges: bool = False,
        cbar_label: str = None,
        # Map features
        gridlines: bool = False,
        coastlines: bool = True,
        borders: bool = True,
        states: bool = True,
        ocean: bool = True,
        land: bool = True,
        lakes: bool = False,
        rivers: bool = False,
        u_component: xr.DataArray = None,
        v_component: xr.DataArray = None,
        quiver_kwargs: dict = None,
        **kwargs,
    ) -> None:
        _params = locals()
        _ = _params.pop("self")
        self._validate_da()
        return animate(da=self, **_params)

    @wraps(calc_trends)
    def trends(
        self,
        along: str = None,
        *,
        scale: float = 1,
        dask_scheduler: Literal["threads", "processes"] = "threads",
    ) -> xr.Dataset:
        _params = locals()
        _ = _params.pop("self")
        self._validate_da()
        return calc_trends(data=self, **_params)

    @wraps(polyfit)
    def polyfit(self, along: str, data_var: str = None, scale: float = 1):
        _params = locals()
        self._validate_da()
        _ = _params.pop("self")
        return polyfit(data=self, **_params)

    @wraps(correlate)
    def correlate(
        self,
        other: xr.DataArray,
        corr_type: Literal["pearson", "spearman", "kendall"] = "pearson",
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        along: str = None,
        dask_scheduler: Literal["threads", "processes"] = "threads",
    ) -> xr.Dataset:
        _params = locals()
        self._validate_da()
        _ = _params.pop("self")
        return correlate(x=self, **_params)

    @wraps(append_to_netcdf)
    def append_to_netcdf(
        self,
        file: Path,
        name: str = None,
    ) -> None:
        _params = locals()
        self._validate_da()
        _ = _params.pop("self")
        append_to_netcdf(da=self, **_params)


class GeoDataArray(GeoMixin, xr.DataArray):
    """
    Extension of xarray.DataArray with Cartopy-based plotting and animation methods.
    """

    __slots__ = ()


class GeoDataset(GeoMixin, xr.Dataset):
    """
    Extension of xarray.Dataset with Cartopy-based plotting and animation methods.
    """

    __slots__ = ()


__all__ = ["GeoDataArray", "GeoDataset", "SetupDask", "append_to_netcdf"]
