"""
xarray utilities for geospatial data, including remapping, masking, and local solar time calculation.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from . import calc_stats as calc
from . import cmaps as cmaps
from . import plotting as plot
from . import theming as theme
from .nc4_utils import (
    append_to_netcdf,
    open_dataset,
    parallel_write_netcdf,
    serial_write_netcdf,
    write_netcdf_variable,
)
from .progress import DaskProgressBar, SerialProgressBar
from .tools import n_cpus, tmp
from .xgeo_utils import sel_transect_latlon, sel_transect_xy

# Collapse attributes by default in the repr
xr.set_options(display_expand_attrs=False)
warnings.filterwarnings("ignore")


__all__ = [
    "cmaps",
    "DaskProgressBar",
    "SerialProgressBar",
    "SetupDask",
    "add_lst",
    "append_to_netcdf",
    "calc",
    "mask_land",
    "open_dataset",
    "plot",
    "remap",
    "sel_transect",
    "theme",
    "write_netcdf",
    "write_netcdf_variable",
]

_script_dir = Path(__file__).resolve().parent
_mask_cache = {}  # cache for remapped masks to avoid redundant computations
_dask_client = None  # global variable to hold the Dask client instance
_dask_cluster = None  # global variable to hold the Dask cluster instance


def write_netcdf(
    file: Path,
    data: xr.Dataset,
    unlimited_dim: str = None,
    *,
    batch_size: int = 1,
    parallel: bool = False,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
    n_files: int = None,
) -> None:
    """
    Write an xarray Dataset to a NetCDF file using netCDF4 lib bypassing xarray's built-in overhead

    The dataset is written along an unlimited dimension. The first slice along
    the unlimited dimension initializes the NetCDF file with compression
    settings for all data variables. Remaining slices are appended one at a time
    using ``append_to_netcdf``.

    Parameters
    ----------
    file : Path
        Output NetCDF file path.
    data : xr.Dataset
        Dataset to write to NetCDF.
    unlimited_dim : str, optional
        Dimension to define as unlimited and append along. If None, the first
        dataset dimension is used.
    batch_size : int, optional
        Number of slices along the unlimited dimension to write in each batch.
        Default is 1 (write one slice at a time).
    parallel : bool, optional
        Whether to enable parallel writing, if true multiple files will be written
            with a suffix .{n}.nc where n is the part number starting from 1. Default is False.
    format : str, optional
        NetCDF format passed to xarray and netCDF4. Default is "NETCDF4".
    shuffle : bool, optional
        Whether to apply the HDF5 shuffle filter. Default is True.
    zlib : bool, optional
        Whether to apply zlib compression. Default is True.
    complevel : int, optional
        Compression level for zlib. Must be between 1 and 9. Default is 4.
    show_progress : bool, optional
        Whether to display a progress bar while writing. Default is True.
    stdout : file-like, optional
        Stream to write the progress bar to. If None, uses sys.stdout.
    n_files : int, optional
        Number of files to split the output into when parallel is True.


    Returns
    -------
    None
        Writes the dataset to disk.
    """

    if not isinstance(data, xr.Dataset):
        raise TypeError("data must be an xarray.Dataset, got %s." % type(data))

    if parallel:
        parallel_write_netcdf(
            path=file,
            data=data,
            unlimited_dim=unlimited_dim,
            batch_size=batch_size,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            show_progress=show_progress,
            stdout=stdout,
            n_files=n_files,
        )

    else:
        return serial_write_netcdf(
            file=file,
            data=data,
            unlimited_dim=unlimited_dim,
            batch_size=batch_size,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            show_progress=show_progress,
            stdout=stdout,
        )


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
            raise ValueError(f"Input grid must contain {coord!r} dimension.")
        if coord not in data.dims:
            raise ValueError(f"Output grid must contain {coord!r} dimension.")

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
    Calculate local solar time and add as a coordinate

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
        raise ValueError(f"Dataset must contain {lon!r} coordinate.")

    offset = ((data[lon] + 180) % 360 - 180) * (24 / 360)
    offset = offset.round() * pd.Timedelta(hours=1)

    lst = data["time"] + offset
    lst.attributes["long_name"] = "Local Solar Time"
    lst.attributes["standard_name"] = "local_solar_time"

    return data.assign_coords({"lst": lst})


def to_lon180(
    data: xr.Dataset | xr.DataArray, lon: str = "lon"
) -> xr.Dataset | xr.DataArray:
    """
    Standardize longitude coordinates to [-180, 180).

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
        raise ValueError(f"Dataset must contain {lon!r} coordinate.")

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
    anchor_point: tuple[float | None, float | None],
    geometry: Literal["latlon", "xy"] = "latlon",
    orientation: float = 0.0,
    width: float = 1.0,
    *,
    x_dim: str = "lon",
    y_dim: str = "lat",
    snap: bool = True,
    drop: bool = True,
):
    """
    Select a finite-width transect band or coordinate band from an xarray object.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Input object with 1D horizontal coordinates.
    anchor_point : tuple[float | None, float | None]
        Point through which the transect axis or coordinate band passes,
        in the order ``(lat,lon)`` if geometry is  latlon or ``(x,y)`` if geometry is xy.

        For ``geometry="xy"``, this is usually ``(x, y)``.
        For ``geometry="latlon"``, this is usually ``(lat, lon)``.

        Use ``None`` for one component to select a coordinate band.
    geometry : {"latlon", "xy"}
        Coordinate geometry used for selection.
    orientation : float
        Transect orientation in degrees.

        For ``geometry="xy"``, orientation is measured clockwise from +y.
        For ``geometry="latlon"``, orientation is measured clockwise from north.
    width : float
        Full transect width in grid cells.
    x_dim, y_dim : str
        Horizontal coordinate names. For latitude longitude grids, use
        ``x_dim="lon"`` and ``y_dim="lat"``.
    snap : bool
        If True, snap supplied anchor coordinates to nearest grid-cell centres.
    drop : bool
        Passed to ``xarray.where``.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Input object masked to the selected transect band or coordinate band.
    """

    if len(anchor_point) != 2:
        raise ValueError("center_point must be a two-element tuple.")

    if geometry == "xy":
        return sel_transect_xy(
            data=data,
            x=anchor_point[0],
            y=anchor_point[1],
            orientation=orientation,
            width=width,
            xdim=x_dim,
            ydim=y_dim,
            snap=snap,
            drop=drop,
        )

    if geometry == "latlon":
        return sel_transect_latlon(
            data=data,
            lat=anchor_point[0],
            lon=anchor_point[1],
            orientation=orientation,
            width=width,
            lon_dim=x_dim,
            lat_dim=y_dim,
            snap=snap,
            drop=drop,
        )

    raise ValueError("geometry must be either 'latlon' or 'xy'.")


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
