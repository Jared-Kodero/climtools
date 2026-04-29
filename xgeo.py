from __future__ import annotations

import logging
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import cartopy.mpl.geoaxes as cgeo
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .plotting import (
    MapPlot,
    animate,
    make_cyclic,
    mapplot,
)
from .statistics import *

# ---- Plot callback ----
from .tools import n_cpus, tmp

warnings.filterwarnings("ignore")
_script_dir = Path(__file__).resolve().parent
_dask_client = None  # global variable to hold the Dask client instance
_dask_cluster = None  # global variable to hold the Dask cluster instance
_mask_cache = {}  # cache for remapped masks to avoid redundant computations


def mask(
    data: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | Path | None = None,
    keep: str = "land",
    parallel: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Apply a land-sea mask to an xarray object.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input data containing latitude and longitude dimensions.
    mask : xr.DataArray or Path, optional
        Land-sea mask. If a DataArray is provided it is used directly.
        If a Path is provided it is opened as a dataset. If None, the
        default ERA5 0.25° mask bundled with the package is used.
    keep : the data variable to keep in the mask dataset
    parallel : bool, default False
        Enable dask-based parallel masking.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Masked data.
    """

    if mask is None:
        mask = _script_dir / "data" / "mask" / "era5_0.25_mask"

    if isinstance(mask, Path):
        mask = xr.open_dataset(mask)[keep].load()

    if isinstance(mask, xr.Dataset):
        raise TypeError(
            "Mask must be an xarray.DataArray. If a Dataset is provided, specify the variable to use with the 'keep' parameter."
        )

    for coord in ("lat", "lon"):
        if coord not in data.dims:
            raise ValueError(f"Input grid must contain '{coord}' dimension.")
        if coord not in data.dims:
            raise ValueError(f"Output grid must contain '{coord}' dimension.")

    lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
    lat_min, lat_max = float(data.lat.min()), float(data.lat.max())
    lon_step = float(data["lon"].diff("lon").mean().values)
    lat_step = float(data["lat"].diff("lat").mean().values)
    cache_key = (lon_min, lon_max, lat_min, lat_max, lon_step, lat_step)

    if cache_key not in _mask_cache:
        data = data.sortby(["lat", "lon"])
        # slice the mask to the same lat/lon + a small buffer to ensure coverage
        lat_slice = slice(lat_min, lat_max)
        lon_slice = slice(lon_min, lon_max)
        mask_da = mask.sel(lat=lat_slice, lon=lon_slice)
        mask_da = xe_remap(mask_da, data, method="nearest_s2d", parallel=parallel)
        _mask_cache[cache_key] = mask_da

    mask_da = _mask_cache[cache_key]

    new_obj = data.where(mask_da == 1, other=np.nan)
    return new_obj


def mask_data(**kwargs) -> xr.DataArray | xr.Dataset:
    return mask(**kwargs)


def calc_local_solar_time(
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


def _to_lon_180(
    ds: xr.Dataset | xr.DataArray, lon: str = "lon"
) -> xr.Dataset | xr.DataArray:
    """
    Standardize the longitude coordinates of an xarray object to be within the range [-180, 180).

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The input dataset or data array containing a longitude coordinate.
    lon : str, default 'lon'
        The name of the longitude coordinate in the dataset.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The dataset or data array with standardized longitude coordinates.
    """
    if lon not in ds:
        raise ValueError(f"Dataset must contain '{lon}' coordinate.")

    ds = ds.copy()
    ds[lon] = (ds[lon] + 180) % 360 - 180
    ds = ds.sortby(lon)
    return ds


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

    _id = [
        f"{method}",
        f"{_in.sizes['lat']}x{_in.sizes['lon']}",
        f"{_out.sizes['lat']}x{_out.sizes['lon']}",
    ]

    weight_file = tmp / ("_".join(_id) + ".nc")
    reuse = weight_file.exists()

    regridder = xe.Regridder(
        _in,
        _out,
        method=method,
        parallel=parallel,
        filename=str(weight_file),
        reuse_weights=reuse,
    )

    out = regridder(grid_in)
    return out


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


class GeoDataArray(GeoMixin, xr.DataArray):
    """
    Extension of xarray.DataArray with Cartopy-based plotting and animation methods.
    """


class GeoDataset(GeoMixin, xr.Dataset):
    """
    Extension of xarray.Dataset with Cartopy-based plotting and animation methods.
    """
