from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .plotting import (
    PlotObj,
    animate,
    animate3d,
    make_cyclic,
    mapplot,
    mapplot3d,
    plot_globe,
    plot_pvalues,
    plot_quiver,
)
from .statistics import *

# ---- Plot callback ----
from .tools import n_cpus

warnings.filterwarnings("ignore")
_script_dir = Path(__file__).resolve().parent
_dask_client = None  # global variable to hold the Dask client instance
_dask_cluster = None  # global variable to hold the Dask cluster instance


def mask(
    data: xr.DataArray | xr.Dataset,
    keep: Literal["land", "ocean"],
    parallel: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Apply a land-sea mask to an xarray object using ERA5 0.25 Land-Sea mask

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The xarray object to which the mask will be applied. Must contain 'lat' and 'lon' dimensions.
    keep : {'land', 'ocean'}
        Specify whether to keep land or ocean points. 'land' will mask out ocean points, and 'ocean' will mask out land points.

    """

    file = _script_dir / "data" / "mask" / "era5_0.25_mask"
    mask_da = xr.open_dataset(file)[keep].load()

    for coord in ("lat", "lon"):
        if coord not in data.dims:
            raise ValueError(f"Input grid must contain '{coord}' dimension.")
        if coord not in data.dims:
            raise ValueError(f"Output grid must contain '{coord}' dimension.")

    # slice the mask to the same lat/lon + a small buffer to ensure coverage
    lat_slice = slice(data.lat.min(), data.lat.max())
    lon_slice = slice(data.lon.min(), data.lon.max())
    mask_da = mask_da.sel(lat=lat_slice, lon=lon_slice)

    mask_da = xe_remap(mask_da, data, method="nearest_d2s", parallel=parallel)
    new_obj = data.where(mask_da == 1, other=np.nan)
    return new_obj


def get_local_solar_time(
    data: xr.Dataset | xr.DataArray, *, lon="lon"
) -> xr.Dataset | xr.DataArray:
    """
    Calculate the local solar time for the dataset based on the longitude coordinate.
    The local solar time is calculated as the UTC time plus the longitude offset.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The input dataset or data array containing a 'time' coordinate and a longitude coordinate.
    lon : str, default 'lon'
        The name of the longitude coordinate in the dataset.
        This coordinate is used to calculate the local solar time offset.


    """

    if lon not in data:
        raise ValueError(f"Dataset must contain '{lon}' coordinate.")

    offset = data[lon] * (24 / 360)
    offset = offset.round() * pd.Timedelta(hours=1)

    lst = data["time"] + offset
    lst.attributes["long_name"] = "Local Solar Time"
    lst.attributes["standard_name"] = "local_solar_time"

    return data.assign_coords({"lst": lst})


def xe_remap(
    grid_in: xr.Dataset | xr.DataArray,
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
) -> xr.Dataset:
    """
    Remap source dataset to the grid of the destination dataset using xesmf.
    """
    for coord in ("lat", "lon"):
        if coord not in grid_in.dims:
            raise ValueError(f"Input grid must contain '{coord}' dimension.")
        if coord not in grid_out.dims:
            raise ValueError(f"Output grid must contain '{coord}' dimension.")

    _in = xr.Dataset(
        coords={
            "lat": grid_in["lat"],
            "lon": grid_in["lon"],
        }
    )

    _out = xr.Dataset(
        coords={
            "lat": grid_out["lat"],
            "lon": grid_out["lon"],
        }
    )
    if parallel:
        _out["dummy"] = xr.DataArray(
            np.ones((_out.lat.size, _out.lon.size)),
            dims=("lat", "lon"),
            coords={"lat": _out.lat, "lon": _out.lon},
        )

        _out = _out.chunk({"lat": "auto", "lon": "auto"})

    regridder = xe.Regridder(_in, _out, method=method, parallel=parallel)

    out = regridder(grid_in)
    return out


class SetupDask:
    """
    A class to manage Dask client and cluster setup for parallel computations.
    It provides methods to start and close a Dask client and cluster with specified configurations.
    """

    def __init__(
        self,
        workers: int = n_cpus,
        threads_per_worker: int = 1,
        processes: bool = True,
        filter_warnings: bool = True,
        memory_limit: int = 0,
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

    def add_cyclic_point(self, dim: str = "lon") -> GeoDataArray:
        return type(self)(make_cyclic(self, dim))

    def get_local_solar_time(self, *, lon="lon") -> GeoDataArray:
        """
        Calculate the local solar time for the DataArray based on the longitude coordinate.
        The local solar time is calculated as the UTC time plus the longitude offset.
        """
        res = get_local_solar_time(data=self, lon=lon)
        return type(self)(res)

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
        """
        Remap source dataset to the grid of the destination dataset using xesmf.
        """
        _params = locals()
        _ = _params.pop("self")
        remapped = xe_remap(self, **_params)
        return type(self)(remapped)

    def mask(
        self,
        keep: Literal["land", "ocean"],
        parallel: bool = False,
    ) -> GeoDataArray:
        """
        Apply a land-sea mask to an xarray object using ERA5 0.25 Land-Sea mask.
        """
        _params = locals()
        _ = _params.pop("self")
        masked = mask(data=self, **_params)
        return type(self)(masked)

    def plot_globe(
        self,
        x: str = "lon",
        y: str = "lat",
        cmap: str | LinearSegmentedColormap | ListedColormap = "viridis",
        coarsen_by: int = 10,
        outfile: str | Path = None,
    ):
        """
        Plot this DataArray on a globe using GeoVista.
        """
        _params = locals()
        _ = _params.pop("self")
        self._validate_da()
        return plot_globe(data=self, **_params)

    def mapplot(
        self,
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
        robust: bool = False,
        title: str = None,
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
        **kwargs,
    ) -> PlotObj:
        """
        Plot this DataArray using Cartopy
        """
        _params = locals()
        _ = _params.pop("self")
        _params["data"] = self

        self._validate_da()

        # return mapplot(data=self, **_params)
        return MapPlotter(**_params)

    def mapplot3d(
        self,
        grid_type: Literal["uniform", "structured"] = "uniform",
        sphere: bool = False,
        window_size: tuple[int, int] = None,
        dim_map: tuple = (("level", "z"), ("lat", "y"), ("lon", "x")),
        z_unit: Literal["hpa", "Pa", "km", "generic"] = "hpa",
        z_unit_scale: int | float = 1,
        cmap: str | LinearSegmentedColormap | ListedColormap = "viridis",
        outfile: Path = None,
        format: Literal["html", "png"] = "png",
        vmin: int | float = None,
        vmax: int | float = None,
        log_scale: bool = False,
        zscale: int | float = 1,
        title: str = None,
        cam_elev: int | float = None,
        cam_azim: int | float = None,
        show_scalar_bar: bool = True,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
        n_xlabels: int = None,
        n_ylabels: int = None,
        n_zlabels: int = None,
        opacity: list | Literal["linear", "sigmoid"] = "linear",
        opacity_unit_distance: int | float = None,
        blending: Literal[
            "additive", "maximum", "minimum", "composite", "average"
        ] = "composite",
        padding: int | float = None,
        font_size: int = None,
        animation: bool = False,
    ):
        """
        Render a 3D scalar field from an xarray.DataArray using PyVista.
        """
        _params = locals()
        _ = _params.pop("self")
        self._validate_da()
        return mapplot3d(data=self, **_params)

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
        robust: bool = False,
        title: str = None,
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
        **kwargs,
    ) -> None:
        """
        Animate this DataArray on a Cartopy map using the global `animate()` function.
        """
        _params = locals()
        _ = _params.pop("self")
        self._validate_da()
        return animate(data=self, **_params)

    def animate3d(
        self,
        dim: str = "time",
        grid_type: Literal["uniform", "structured"] = "uniform",
        sphere: bool = False,
        dim_map: tuple = (("level", "z"), ("lat", "y"), ("lon", "x")),
        z_unit: Literal["hpa", "Pa", "km", "generic"] = "hpa",
        z_unit_scale: int | float = 1,
        indices: list = None,
        outfile: Path = None,
        format: Literal["mp4", "html"] = "mp4",
        window_size: tuple[int, int] = None,
        fps: int = 10,
        parallel: bool = True,
        cmap: str | LinearSegmentedColormap | ListedColormap = "viridis",
        vmin: int | float = None,
        vmax: int | float = None,
        log_scale: bool = False,
        zscale: int | float = 1,
        title: str = None,
        cam_elev: int | float = None,
        cam_azim: int | float = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
        show_scalar_bar: bool = True,
        n_xlabels: int = None,
        n_ylabels: int = None,
        n_zlabels: int = None,
        font_size: int = None,
        opacity: list | Literal["linear", "sigmoid"] = "linear",
        opacity_unit_distance: int | float = None,
        blending: Literal[
            "additive", "maximum", "minimum", "composite", "average"
        ] = "composite",
        padding: int | float = None,
        **kwargs,
    ):
        """
        Animate this DataArray in 3D using the global `animate3d()` function.
        """
        _params = locals()
        _ = _params.pop("self")
        self._validate_da()
        return animate3d(data=self, **_params)

    def trends(
        self,
        along: str = None,
        *,
        scale: float = 1,
        dask_scheduler: Literal["threads", "processes"] = "threads",
    ) -> xr.Dataset:
        """
        Calculate the Mann-Kendall trend test for a given dataset.
        """
        _params = locals()
        _ = _params.pop("self")
        self._validate_da()
        return calc_trends(data=self, **_params)

    def polyfit(self, along: str, data_var: str = None, scale: float = 1):
        """
        Calculate the linear trend for the given xarray Dataset or DataArray using xr.polyfit.
        """
        _params = locals()

        self._validate_da()
        _ = _params.pop("self")
        return polyfit(data=self, **_params)

    def correlate(
        self,
        other: xr.DataArray,
        corr_type: Literal["pearson", "spearman", "kendall"] = "pearson",
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        along: str = None,
        dask_scheduler: Literal["threads", "processes"] = "threads",
    ) -> xr.Dataset:
        """
        Compute the correlation between this DataArray/Dataset and another along a specified dimension.
        """
        _params = locals()
        self._validate_da()
        _ = _params.pop("self")
        return correlate(x=self, **_params)


class GeoDataArray(GeoMixin, xr.DataArray):
    """
    Extension of xarray.DataArray with Cartopy-based plotting and animation methods.
    """


class GeoDataset(GeoMixin, xr.Dataset):
    """
    Extension of xarray.Dataset with Cartopy-based plotting and animation methods.
    """


class MapPlotter:
    def __init__(self, **kwargs):
        self.plot_kwargs = kwargs
        self.fig = None
        self.ax = None
        self.artist = None

    def show(self):
        plt.show()

    def save(self, outfile: str | Path, **kwargs):
        plt.savefig(outfile, **kwargs)

    def plot(self):
        p = mapplot(**self.plot_kwargs)
        self.fig = p.fig
        self.ax = p.ax
        self.artist = p.artist
        return self

    def plot_pvalues(
        self,
        data: xr.DataArray,
        ax: plt.Axes = None,
        level: float = 0.05,
        color: str = "grey",
        alpha: float = 0.3,
        marker: str = None,
        edgecolors: str = None,
        step_size: int = 1,
        s: float = 0.25,
    ):

        plot_pvalues(
            ax=self.ax,
            data=data,
            level=level,
            color=color,
            alpha=alpha,
            marker=marker,
            edgecolors=edgecolors,
            step_size=step_size,
            s=s,
        )
        return self

    def plot_quiver(self, u: xr.DataArray, v: xr.DataArray, step: int = 1):
        plot_quiver(u, v, ax=self.ax, step=step)
        return self
