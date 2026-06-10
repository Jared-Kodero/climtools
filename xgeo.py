"""
xarray utilities for geospatial data, including remapping, masking, and local solar time calculation.
"""

from __future__ import annotations

import logging
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from dask.diagnostics import ProgressBar

from . import calc_stats as calc
from . import plotting as plot
from . import theming as theme
from .tools import n_cpus, tmp

warnings.filterwarnings("ignore")


__all__ = [
    "SetupDask",
    "open_dataset",
    "open_mfdataset",
    "append_to_netcdf",
    "mask_land",
    "add_lst",
    "remap",
    "plot",
    "calc",
    "theme",
]

_script_dir = Path(__file__).resolve().parent
_mask_cache = {}  # cache for remapped masks to avoid redundant computations
_dask_client = None  # global variable to hold the Dask client instance
_dask_cluster = None  # global variable to hold the Dask cluster instance


class DaskProgressBar(ProgressBar):
    """
    Dask progress bar styled like rich.progress.track.

    Requires:
        pip install rich
    """

    def __init__(
        self,
        description: str = "",
        transient: bool = False,
        refresh_per_second: int = 10,
    ) -> None:
        super().__init__()
        self.description = description
        self.transient = transient
        self.refresh_per_second = refresh_per_second

        self._progress = None
        self._task_id = None
        self._total = 0
        self._completed = 0

    def _start(self, dsk: Any) -> None:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._total = len(dsk)
        self._completed = 0

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=self.transient,
            refresh_per_second=self.refresh_per_second,
        )

        self._progress.start()
        self._task_id = self._progress.add_task(
            self.description,
            total=self._total,
        )

    def _posttask(
        self,
        key: Any,
        result: Any,
        dsk: Any,
        state: dict[str, Any],
        worker_id: Any,
    ) -> None:
        self._completed += 1

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self._completed,
            )

    def _finish(
        self,
        dsk: Any,
        state: dict[str, Any],
        errored: bool,
    ) -> None:
        if self._progress is None or self._task_id is None:
            return

        if not errored:
            # Force the bar to full and repaint immediately, so a bursty or
            # late callback feed cannot leave it short of 100 percent.
            self._completed = self._total
            self._progress.update(
                self._task_id,
                completed=self._total,
                refresh=True,
            )
            self._progress.refresh()

        self._progress.stop()


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


def append_to_netcdf(
    file: Path,
    da: xr.DataArray,
    name: str = None,
    mode: Literal["a", "r+"] = "r+",
    format="NETCDF4",
) -> None:
    """Append a DataArray to an existing NetCDF file

    Parameters
    ----------
    file : Path
        Path to a NetCDF4 file opened with read/write access.
    da : xr.DataArray
        DataArray to append. Must have dimensions that already exist in the file.
    name : str, optional
        Name of the variable to create in the NetCDF file. If None, uses da.name.
    """

    with nc.Dataset(file, mode=mode, format=format) as ncf:
        varname = name or da.name

        # Overwrite values if the variable was created on a previous run.
        if varname in ncf.variables:
            ncf.variables[varname][:] = da.values

        else:
            missing = [d for d in da.dims if d not in ncf.dimensions]
            if missing:
                raise ValueError(
                    f"Cannot write {varname} to {file}: missing dimensions {missing}"
                )

            ncvar = ncf.createVariable(
                varname=varname,
                datatype=da.dtype,
                dimensions=da.dims,
                zlib=True,
                complevel=4,
            )

            for attr_name, attr_val in da.attrs.items():
                ncvar.setncattr(attr_name, attr_val)

            ncvar[:] = da.values


def mask_land(
    data: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | Path | None = None,
    keep: str = "land",
    parallel: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Apply a land-sea mask to an xarray object.

    Parameters
    ----------
    data: xr.DataArray or xr.Dataset
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
        mask_da = remap(mask_da, data, method="nearest_s2d", parallel=parallel)
        _mask_cache[cache_key] = mask_da

    mask_da = _mask_cache[cache_key]

    new_obj = data.where(mask_da == 1, other=np.nan)
    return new_obj


def add_lst(data: xr.Dataset | xr.DataArray, *, lon="lon") -> xr.Dataset | xr.DataArray:
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

    offset = ((data[lon] + 180) % 360 - 180) * (24 / 360)
    offset = offset.round() * pd.Timedelta(hours=1)

    lst = data["time"] + offset
    lst.attributes["long_name"] = "Local Solar Time"
    lst.attributes["standard_name"] = "local_solar_time"

    return data.assign_coords({"lst": lst})


def wrap_longitude_to_180(
    data: xr.Dataset | xr.DataArray, lon: str = "lon"
) -> xr.Dataset | xr.DataArray:
    """
    Standardize the longitude coordinates of an xarray object to be within the range [-180, 180).

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The input dataset or data array containing a longitude coordinate.
    lon : str, default 'lon'
        The name of the longitude coordinate in the dataset.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The dataset or data array with standardized longitude coordinates.
    """
    if lon not in data:
        raise ValueError(f"Dataset must contain '{lon}' coordinate.")

    data = data.copy()
    data[lon] = (data[lon] + 180) % 360 - 180
    data = data.sortby(lon)
    return data


def remap(
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

    Parameters
    ----------
    grid_in : xr.Dataset or xr.DataArray
        The input dataset or data array containing 'lat' and 'lon' coordinates.
    grid_out : xr.Dataset or xr.DataArray
        The output dataset or data array containing 'lat' and 'lon' coordinates.
    method : str, default 'bilinear'
        The remapping method to use. Options include:
        - 'bilinear': Bilinear interpolation (default)
        - 'conservative': Conservative remapping
        - 'conservative_normed': Normalized conservative remapping
        - 'patch': Patch remapping
        - 'nearest_s2d': Nearest neighbor remapping from source to destination
        - 'nearest_d2s': Nearest neighbor remapping from destination to source
    parallel : bool, default False
        Whether to enable parallel remapping using Dask.
    """
    for coord in ("lat", "lon"):
        if coord not in grid_in.dims:
            raise ValueError(f"Input grid must contain {coord!r} dimension.")
        if coord not in grid_out.dims:
            raise ValueError(f"Output grid must contain {coord!r} dimension.")

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


def sel_transect(
    data: xr.Dataset | xr.DataArray,
    lat: float = None,
    lon: float = None,
    orientation: float = 0.0,
    width: float = 1.0,
    *,
    snap: bool = True,
    drop: bool = True,
):
    """
    Select a finite-width transect or coordinate band from an xarray object.

    Modes
    -----
    lat and lon given
        Select a great-circle transect passing through the selected point.

    lat only
        Select a latitude band centred on `lat`.

    lon only
        Select a longitude band centred on `lon`.

    Parameters
    ----------
    data
        xarray Dataset or DataArray with 1D 'lat' and 'lon' coordinates.
    lat, lon
        Centre latitude and/or longitude in degrees.
    orientation
        Transect orientation in degrees, measured clockwise from north.
        Only used when both `lat` and `lon` are given.
        0 gives a north-south transect. 90 gives an east-west transect.
    width
        Full transect width in grid cells, not degrees.
    snap
        If True, snap supplied centre coordinate or coordinates to nearest
        grid-cell centre.
    drop
        Passed to xarray.where.

    Returns
    -------
    xarray Dataset or DataArray
        Input object masked to the selected transect or band.
    """
    import numpy as np
    import xarray as xr

    if "lat" not in data.coords or "lon" not in data.coords:
        raise ValueError("Input data must contain 'lat' and 'lon' coordinates.")

    if lat is None and lon is None:
        raise ValueError("At least one of `lat` or `lon` must be provided.")

    if width <= 0:
        raise ValueError("width must be positive.")

    latc = data["lat"]
    lonc = data["lon"]

    rectilinear = (
        latc.ndim == 1 and lonc.ndim == 1 and "lat" in data.dims and "lon" in data.dims
    )

    if not rectilinear:
        raise ValueError(
            "Only rectilinear grids with 1D 'lat' and 'lon' coordinates are supported."
        )

    if latc.sizes["lat"] < 2:
        raise ValueError("Latitude coordinate must contain at least two points.")

    if lonc.sizes["lon"] < 2:
        raise ValueError("Longitude coordinate must contain at least two points.")

    dlat = np.abs(latc.diff("lat")).median("lat")
    dlon = np.abs(((lonc.diff("lon") + 180.0) % 360.0) - 180.0).median("lon")

    if lat is not None and lon is None:
        lat0 = float(latc.sel(lat=lat, method="nearest")) if snap else float(lat)
        half_width_deg = 0.5 * width * dlat
        mask = np.abs(latc - lat0) <= half_width_deg
        return data.where(mask, drop=drop)

    if lon is not None and lat is None:
        if snap:
            dlon_to_centre = np.abs(((lonc - lon + 180.0) % 360.0) - 180.0)
            lon0 = float(lonc.isel(lon=dlon_to_centre.argmin("lon")))
        else:
            lon0 = float(lon)

        half_width_deg = 0.5 * width * dlon
        mask = np.abs(((lonc - lon0 + 180.0) % 360.0) - 180.0) <= half_width_deg
        return data.where(mask, drop=drop)

    if snap:
        lat0 = float(latc.sel(lat=lat, method="nearest"))

        dlon_to_centre = np.abs(((lonc - lon + 180.0) % 360.0) - 180.0)
        lon0 = float(lonc.isel(lon=dlon_to_centre.argmin("lon")))
    else:
        lat0 = float(lat)
        lon0 = float(lon)

    phi0 = np.deg2rad(lat0)
    lam0 = np.deg2rad(lon0)
    theta = np.deg2rad(orientation % 180.0)

    cross_north_weight = abs(np.sin(theta))
    cross_east_weight = abs(np.cos(theta))

    cell_width_deg = xr.apply_ufunc(
        np.sqrt,
        (cross_north_weight * dlat) ** 2
        + (cross_east_weight * dlon * np.cos(phi0)) ** 2,
        dask="allowed",
    )

    half_width_deg = 0.5 * width * cell_width_deg

    ax = np.cos(phi0) * np.cos(lam0)
    ay = np.cos(phi0) * np.sin(lam0)
    az = np.sin(phi0)

    north_x = -np.sin(phi0) * np.cos(lam0)
    north_y = -np.sin(phi0) * np.sin(lam0)
    north_z = np.cos(phi0)

    east_x = -np.sin(lam0)
    east_y = np.cos(lam0)
    east_z = 0.0

    dx = np.cos(theta) * north_x + np.sin(theta) * east_x
    dy = np.cos(theta) * north_y + np.sin(theta) * east_y
    dz = np.cos(theta) * north_z + np.sin(theta) * east_z

    gx = ay * dz - az * dy
    gy = az * dx - ax * dz
    gz = ax * dy - ay * dx

    phi = xr.apply_ufunc(np.deg2rad, latc, dask="allowed")
    lam = xr.apply_ufunc(np.deg2rad, lonc, dask="allowed")

    px = xr.apply_ufunc(np.cos, phi, dask="allowed") * xr.apply_ufunc(
        np.cos, lam, dask="allowed"
    )
    py = xr.apply_ufunc(np.cos, phi, dask="allowed") * xr.apply_ufunc(
        np.sin, lam, dask="allowed"
    )
    pz = xr.apply_ufunc(np.sin, phi, dask="allowed")

    dot_n = gx * px + gy * py + gz * pz
    dot_n = dot_n.clip(min=-1.0, max=1.0)

    cross_track_deg = xr.apply_ufunc(
        np.rad2deg,
        xr.apply_ufunc(np.arcsin, dot_n, dask="allowed"),
        dask="allowed",
    )

    mask = np.abs(cross_track_deg) <= half_width_deg

    return data.where(mask, drop=drop)


@wraps(xr.open_dataset)
def open_dataset(*args, **kwargs):
    return xr.open_dataset(*args, **kwargs)


@wraps(xr.open_mfdataset)
def open_mfdataset(*args, **kwargs):
    return xr.open_mfdataset(*args, **kwargs)
