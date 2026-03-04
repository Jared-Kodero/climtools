from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .plotting import PlotObj, animate, make_cyclic, mapplot
from .statistics import *
from .tools import n_cpus

warnings.filterwarnings("ignore")
_script_dir = Path(__file__).resolve().parent
_dask_client = None  # global variable to hold the Dask client instance
_dask_cluster = None  # global variable to hold the Dask cluster instance


class GeoDataArray(xr.DataArray):
    """
    Extension of xarray.DataArray with Cartopy-based plotting and animation methods.
    """

    __slots__ = ()

    def add_cyclic_point(self, dim: str = "lon") -> GeoDataArray:
        return make_cyclic(self, dim)

    def get_local_solar_time(self, *, longitude="lon") -> GeoDataArray:
        """
        Calculate the local solar time for the DataArray based on the longitude coordinate.
        The local solar time is calculated as the UTC time plus the longitude offset.
        """
        lsted = get_local_solar_time(self, longitude=longitude)
        return GeoDataArray(lsted)

    def mask(
        self,
        keep: Literal["land", "ocean"],
        *,
        mask_file: Literal["cartopy", "era5"] = "era5",
    ) -> GeoDataArray:
        """
        Apply a land-sea mask to the DataArray.
        """
        masked = mask(
            self,
            keep=keep,
            mask_file=mask_file,
        )
        return GeoDataArray(masked)

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
        return mapplot(
            self,
            x=x,
            y=y,
            projection=projection,
            global_extent=global_extent,
            set_extent=set_extent,
            figsize=figsize,
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            method=method,
            norm=norm,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            robust=robust,
            title=title,
            extend=extend,
            orientation=orientation,
            add_colorbar=add_colorbar,
            drawedges=drawedges,
            cbar_label=cbar_label,
            gridlines=gridlines,
            coastlines=coastlines,
            borders=borders,
            states=states,
            ocean=ocean,
            land=land,
            lakes=lakes,
            rivers=rivers,
            **kwargs,
        )

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

        return animate(
            self,
            dim=dim,
            indices=indices,
            outfile=outfile,
            quality=quality,
            fps=fps,
            parallel=parallel,
            x=x,
            y=y,
            projection=projection,
            global_extent=global_extent,
            set_extent=set_extent,
            figsize=figsize,
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            method=method,
            norm=norm,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            robust=robust,
            title=title,
            extend=extend,
            orientation=orientation,
            add_colorbar=add_colorbar,
            drawedges=drawedges,
            cbar_label=cbar_label,
            gridlines=gridlines,
            coastlines=coastlines,
            borders=borders,
            states=states,
            ocean=ocean,
            land=land,
            lakes=lakes,
            rivers=rivers,
            **kwargs,
        )

    def trends(
        self,
        along: str = None,
        *,
        scale: float = 1,
        dask_scheduler: Literal["threads", "processes"] = "threads",
    ) -> xr.Dataset:
        """
        Calculate the Mann-Kendall trend test for a given dataset.

        Parameters
        ----------
        along : str
            Dimension along which the trend is evaluated (e.g., "time"). This
            dimension must exist in the data.
        scale : float, default 1
            Factor applied to the Sen slope estimate (e.g., convert from
            units per timestep to units per year).
        dask_scheduler : {"threads", "processes"}, default "threads"
            Scheduler used when executing the trend computations on Dask-backed
            arrays.


        Returns:
            xr.Dataset: DataFrame or Dataset containing the trend test results.
        """
        return calc_trends(
            self,
            along=along,
            scale=scale,
            dask_scheduler=dask_scheduler,
        )

    def polyfit(self, along: str, data_var: str = None, scale: float = 1):
        """
        Calculate the linear trend for the given xarray Dataset or DataArray using xr.polyfit.

        Parameters
        ----------
        - along: str
            Dimension to calculate the trend test along. Also used for sorting the data.
        - scale: float
            The scale to multiply the slope by i.e convert to per hour, per day, etc.

        Returns: xr.Dataset
        """
        return polyfit(
            self,
            along=along,
            data_var=data_var,
            scale=scale,
        )

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

        Parameters
        ----------
        other : xr.DataArray or xr.Dataset
            The other DataArray or Dataset to correlate with.
        along : str, default "time"
            The dimension along which to compute the correlation.
        method : {"pearson", "spearman"}, default "pearson"
            The correlation method to use.

        Returns
        -------
        xr.Dataset
            Dataset containing the correlation coefficients.
        """
        return correlate(
            self,
            other,
            corr_type=corr_type,
            alternative=alternative,
            along=along,
            dask_scheduler=dask_scheduler,
        )


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


mask_data = {}  # cache for loaded mask datasets
grid_data = {}  # cache for loaded grid datasets
masks = {  # mapping of mask_file options to actual file names in the data
    "cartopy": "cartopy_0.1_mask",
    "era5": "era5_0.25_mask",
}


def _grid_signature(obj: xr.DataArray | xr.Dataset) -> tuple:
    """
    Create a deterministic grid signature based on coordinates.
    """
    lat = obj.lat.values
    lon = obj.lon.values

    return (
        float(lat.min()),
        float(lat.max()),
        float(np.diff(lat).mean()),
        float(lon.min()),
        float(lon.max()),
        float(np.diff(lon).mean()),
        float(lat.mean()),
        float(lon.mean()),
        lat.size,
        lon.size,
    )


def mask(
    obj: xr.DataArray | xr.Dataset,
    keep: Literal["land", "ocean"],
    *,
    mask_file: Literal["cartopy", "era5"] = "era5",
) -> xr.DataArray | xr.Dataset:
    """
    Apply a land-sea mask to an xarray object.

    The mask is interpolated to the target grid using nearest-neighbour
    interpolation and cached per grid configuration.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The xarray object to which the mask will be applied. Must contain 'lat' and 'lon' dimensions.
    keep : {'land', 'ocean'}
        Specify whether to keep land or ocean points. 'land' will mask out ocean points, and 'ocean' will mask out land points.
    mask_file : {'cartopy', 'era5'}, default 'era5'
        The source of the land-sea mask. 'cartopy' uses a mask derived from Cartopy's shapefiles, while 'era5' uses a mask derived from ERA5 reanalysis data. The choice may affect the resolution and accuracy of the mask, especially for coastal regions.
    """

    if "lat" not in obj.dims or "lon" not in obj.dims:
        raise ValueError("Object must contain 'lat' and 'lon' dimensions.")

    obj = obj.sortby(["lat", "lon"])

    # Global caches assumed defined outside:
    # mask_data: dict[str, xr.Dataset]
    # grid_data: dict[tuple, xr.DataArray]

    file = _script_dir / "data" / "mask" / masks[mask_file]
    grid_key = (mask_file, _grid_signature(obj))

    # ------------------------------------------------------------------
    # 1. Ensure raw mask file is loaded only once
    # ------------------------------------------------------------------
    if mask_file not in mask_data:
        mask_data[mask_file] = xr.open_dataset(file, engine="netcdf4").load()

    raw_mask = mask_data[mask_file]

    # ------------------------------------------------------------------
    # 2. Ensure interpolated mask exists for this grid
    # ------------------------------------------------------------------
    if grid_key not in grid_data:
        lat_min = float(obj.lat.min())
        lat_max = float(obj.lat.max())
        lon_min = float(obj.lon.min())
        lon_max = float(obj.lon.max())

        mask_interp = (
            raw_mask[keep]
            .sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            .interp(lat=obj.lat, lon=obj.lon, method="nearest")
        )

        grid_data[grid_key] = mask_interp

    mask_interp = grid_data[grid_key]
    return obj.where(mask_interp, other=np.nan)


def get_local_solar_time(
    data: xr.Dataset | xr.DataArray, *, longitude="lon"
) -> xr.Dataset | xr.DataArray:
    """
    Calculate the local solar time for the dataset based on the longitude coordinate.
    The local solar time is calculated as the UTC time plus the longitude offset.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The input dataset or data array containing a 'time' coordinate and a longitude coordinate.
    longitude : str, default 'lon'
        The name of the longitude coordinate in the dataset.
        This coordinate is used to calculate the local solar time offset.


    """

    if longitude not in data:
        raise ValueError(f"Dataset must contain '{longitude}' coordinate.")

    offset = data[longitude] * (24 / 360)
    offset = offset.round() * pd.Timedelta(hours=1)

    lst = data["time"] + offset
    lst.attributes["long_name"] = "Local Solar Time"
    lst.attributes["standard_name"] = "local_solar_time"

    return data.assign_coords({"lst": lst})
