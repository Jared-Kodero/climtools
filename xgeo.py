"""
Geospatial operations on xarray objects.

This module is the working namespace of the package. It re-exports the
plotting entry points, the statistical routines and the NetCDF writers, and
adds the operations that act directly on gridded data: regridding
(:func:`remap`), land-sea masking (:func:`mask_land`), transect selection
(:func:`sel_transect`), local solar time (:func:`add_lst`) and a Dask cluster
helper (:class:`SetupDask`).

Typical use::

    from climtools import xgeo as xg

    da = xg.remap(da, target_grid, method="conservative")
    da = xg.mask_land(da, keep="land")
    xg.plot.geo(da.mean("time"), method="contourf")

Every function here is also reachable as a method on the ``.xgeo`` accessor,
which is registered on both ``DataArray`` and ``Dataset`` when the package is
imported::

    da.xgeo.remap(target_grid).xgeo.mask_land().xgeo.plot.field()

Regridding requires ``xesmf``. It is imported on first use, so the rest of the
package remains usable in environments where ESMF is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from . import accessors as accessors  # noqa: F401  (registers the .xgeo accessor)
from . import calc_stats as calc
from . import cmaps as cmaps
from . import plotting as plot
from .nc4_utils import (
    append_to_netcdf,
    serial_write_netcdf,
    write_netcdf_variable,
)
from .preprocess_era5 import preprocess_era5
from .progress import DaskProgressBar, SerialProgressBar
from .tools import n_cpus, tmp
from .xgeo_utils import sel_transect_latlon, sel_transect_xy, to_lon180

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# Collapse attributes by default in the repr.
xr.set_options(display_expand_attrs=False)

__all__ = [
    "SetupDask",
    "DaskProgressBar",
    "SerialProgressBar",
    "add_lst",
    "append_to_netcdf",
    "calc",
    "cmaps",
    "mask_land",
    "n_cpus",
    "plot",
    "preprocess_era5",
    "remap",
    "sel_transect",
    "to_lon180",
    "write_netcdf",
    "write_netcdf_variable",
]

_script_dir = Path(__file__).resolve().parent
_default_mask = _script_dir / "data" / "mask" / "era5_0.25_mask"

#: Cache of remapped land-sea masks, keyed by target grid and mask identity.
_mask_cache: dict = {}

#: Module-level Dask handles, shared by every :class:`SetupDask` instance.
_dask_client = None
_dask_cluster = None


def write_netcdf(
    data: xr.Dataset,
    file: Path,
    unlimited_dim: str = None,
    *,
    batch_size: int = 1,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
) -> None:
    """
    Write a Dataset to NetCDF through the netCDF4 library.

    The file is written incrementally along an unlimited dimension, which keeps
    peak memory proportional to one batch rather than to the whole dataset. The
    first slice defines the file, its dimensions, variables, attributes and
    compression settings; the remaining slices are appended with
    :func:`~climtools.nc4_utils.append_to_netcdf`.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset to write.
    file : pathlib.Path
        Output path. An existing file is replaced.
    unlimited_dim : str, optional
        Dimension made unlimited and appended along. Defaults to the first
        dimension of the dataset.
    batch_size : int, default 1
        Number of slices along the unlimited dimension written per append.
    format : str, default "NETCDF4"
        NetCDF format.
    shuffle : bool, default True
        Apply the HDF5 shuffle filter.
    zlib : bool, default True
        Apply zlib compression.
    complevel : int, default 4
        Compression level, between 1 and 9.
    show_progress : bool, default True
        Display a progress bar while writing.
    stdout : file-like, optional
        Stream the progress bar is written to. Defaults to ``sys.stdout``.

    Returns
    -------
    None
    """
    if not isinstance(data, xr.Dataset):
        raise TypeError(f"data must be an xarray.Dataset, got {type(data)}.")

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

    import xesmf as xe

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


def mask_land(
    data: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | xr.Dataset | str | Path | None = None,
    keep: Literal["land", "ocean"] = "land",
    parallel: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Mask an object outside the land or the ocean.

    The mask is regridded onto the grid of ``data`` with nearest-neighbour
    interpolation, which preserves its categorical values, and the result is
    cached so that repeated calls on the same grid do not repeat the
    regridding.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input object, carrying 'lat' and 'lon' dimensions.
    mask : xarray.DataArray, xarray.Dataset, str, pathlib.Path or None, optional
        Land-sea mask, in which 1 marks the retained domain. A DataArray is
        used as given. A Dataset or a path is opened and the ``keep`` variable
        is taken from it. Defaults to the ERA5 0.25 degree mask bundled with
        the package.
    keep : {"land", "ocean"}, default "land"
        Name of the mask variable to select, and hence the domain retained.
        Ignored when ``mask`` is already a DataArray.
    parallel : bool, default False
        Regrid the mask in parallel with Dask.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The input object with cells outside the retained domain set to NaN.
    """

    if mask is None:
        mask = _default_mask

    mask_id = str(mask) if isinstance(mask, (str, Path)) else id(mask)

    if isinstance(mask, (str, Path)):
        mask = xr.open_dataset(mask)

    if isinstance(mask, xr.Dataset):
        if keep not in mask:
            raise KeyError(
                f"Mask variable {keep!r} not found; available: {list(mask.data_vars)}."
            )
        mask = mask[keep].load()

    if not isinstance(mask, xr.DataArray):
        raise TypeError(f"mask must resolve to an xarray.DataArray, got {type(mask)}.")

    # Sort unconditionally: the regridded mask is cached, so the alignment
    # between data and mask must not depend on whether the cache was hit.
    data = data.sortby(["lat", "lon"])

    lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
    lat_min, lat_max = float(data.lat.min()), float(data.lat.max())
    cache_key = (
        mask_id,
        keep,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        data.sizes["lon"],
        data.sizes["lat"],
    )

    if cache_key not in _mask_cache:
        subset = mask.sortby(["lat", "lon"]).sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )
        _mask_cache[cache_key] = remap(
            subset, data, method="nearest_s2d", parallel=parallel
        )

    return data.where(_mask_cache[cache_key] == 1, other=np.nan)


def add_lst(
    data: xr.Dataset | xr.DataArray,
    *,
    lon: str = "lon",
    time: str = "time",
    name: str = "lst",
) -> xr.Dataset | xr.DataArray:
    """
    Add mean local solar time as a coordinate.

    Local solar time is approximated as UTC shifted by the longitude offset,

    .. math::

        \\mathrm{LST} = \\mathrm{UTC} + \\mathrm{round}\\!\\left(
        \\frac{\\lambda}{15^\\circ}\\right)\\,\\mathrm{h}

    where :math:`\\lambda` is longitude in degrees east, wrapped to
    :math:`[-180, 180)`. The offset is rounded to whole hours, so the result is
    a mean solar time on hourly zones rather than apparent solar time: the
    equation of time, which reaches roughly plus or minus 16 minutes over the
    year, is not applied.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object carrying a time coordinate and a longitude coordinate.
    lon : str, default "lon"
        Name of the longitude coordinate, in degrees east.
    time : str, default "time"
        Name of the UTC time coordinate.
    name : str, default "lst"
        Name given to the new coordinate.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input object with the local solar time coordinate attached. The
        coordinate spans both the time and longitude dimensions.
    """
    for coord in (lon, time):
        if coord not in data.coords:
            raise ValueError(f"Input must contain a {coord!r} coordinate.")

    offset = ((data[lon] + 180) % 360 - 180) * (24 / 360)
    offset = offset.round() * pd.Timedelta(hours=1)

    lst = data[time] + offset
    lst.attrs["long_name"] = "Local Solar Time"
    lst.attrs["standard_name"] = "local_solar_time"
    lst.attrs["description"] = (
        "Mean local solar time on whole-hour longitude zones; "
        "the equation of time is not applied."
    )

    return data.assign_coords({name: lst})


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
) -> xr.Dataset | xr.DataArray:
    """
    Select a finite-width transect band, or a single coordinate band.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Input object on a rectilinear grid with one-dimensional horizontal
        coordinates.
    anchor_point : tuple of float or None
        Point the transect axis passes through, ordered ``(lat, lon)`` when
        ``geometry="latlon"`` and ``(x, y)`` when ``geometry="xy"``. Passing
        None for one component selects a band along the other coordinate
        instead of an oriented transect, in which case ``orientation`` is
        ignored.
    geometry : {"latlon", "xy"}, default "latlon"
        Coordinate geometry. ``"latlon"`` measures the cross-track distance on
        the sphere; ``"xy"`` measures it in the plane.
    orientation : float, default 0.0
        Transect orientation in degrees, measured clockwise from north for
        ``"latlon"`` and clockwise from ``+y`` for ``"xy"``.
    width : float, default 1.0
        Full transect width, in grid cells.
    x_dim, y_dim : str, default "lon", "lat"
        Horizontal coordinate names.
    snap : bool, default True
        Snap the anchor point to the nearest grid-cell centre.
    drop : bool, default True
        Drop the masked cells, as in ``xarray.DataArray.where``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input object restricted to the selected band.
    """
    if len(anchor_point) != 2:
        raise ValueError("anchor_point must be a two-element tuple.")

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
    Manage a local Dask cluster and client.

    The cluster and client are shared at module level, so repeated
    instantiation reuses one cluster rather than starting several. The object
    can be used directly or as a context manager::

        with SetupDask(workers=4) as dask_setup:
            result = data.mean("time").compute()

    Parameters
    ----------
    workers : int, default 1
        Number of worker processes in the cluster.
    threads_per_worker : int, optional
        Threads per worker. Defaults to the number of available CPUs.
    processes : bool, default False
        Use separate processes for the workers rather than threads. Processes
        avoid the global interpreter lock but pay a serialization cost.
    filter_warnings : bool, default True
        Silence Dask logging below the error level.
    memory_limit : str or int, default "auto"
        Memory limit per worker, for example ``"8GB"``. Set to 0 for no limit.

    Attributes
    ----------
    client : dask.distributed.Client or None
        The active client, or None before :meth:`start` is called.
    cluster : dask.distributed.LocalCluster or None
        The active cluster, or None before :meth:`start` is called.
    """

    def __init__(
        self,
        workers: int = 1,
        threads_per_worker: int = n_cpus,
        processes: bool = False,
        filter_warnings: bool = True,
        memory_limit: str | int = "auto",
    ):
        self.cluster = None
        self.client = None
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self.processes = processes
        self.filter_warnings = filter_warnings
        self.memory_limit = memory_limit

    def start(self):
        """
        Start the cluster and client, or return the existing ones.

        Returns
        -------
        dask.distributed.Client
            The active client. The dashboard is served on port 8787.
        """
        global _dask_client, _dask_cluster

        if self.client is not None:
            return self.client

        from dask.distributed import Client, LocalCluster

        if _dask_client is not None and _dask_cluster is not None:
            self.client = _dask_client
            self.cluster = _dask_cluster
            return self.client

        self.cluster = LocalCluster(
            n_workers=self.workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
            silence_logs=logging.ERROR if self.filter_warnings else logging.WARNING,
            processes=self.processes,
            dashboard_address=":8787",
        )
        self.client = Client(self.cluster)
        _dask_client = self.client
        _dask_cluster = self.cluster

        return self.client

    def close(self) -> None:
        """Close the active client and cluster and release the module handles."""
        global _dask_client, _dask_cluster

        if self.client is not None:
            self.client.close()
        if self.cluster is not None:
            self.cluster.close()

        if self.client is _dask_client:
            _dask_client = None
        if self.cluster is _dask_cluster:
            _dask_cluster = None

        self.client = None
        self.cluster = None

    def __enter__(self) -> SetupDask:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
