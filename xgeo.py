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
from .nc4_utils import append_to_netcdf, dataarray_to_netcdf, dataset_to_netcdf
from .preprocess_era5 import preprocess_era5
from .progress import DaskProgressBar, SerialProgressBar
from .tools import n_cpus, tmp
from .xgeo_utils import grid_id, sel_transect, to_lon180

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# Collapse attributes by default in the repr.
xr.set_options(display_expand_attrs=False)

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
_script_dir = Path(__file__).resolve().parent


#: Module-level Dask handles, shared by every :class:`SetupDask` instance.
_dask_client = None
_dask_cluster = None


def to_netcdf(
    data: xr.Dataset | xr.DataArray,
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
    if isinstance(data, xr.Dataset):
        return dataset_to_netcdf(
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
    elif isinstance(data, xr.DataArray):
        return dataarray_to_netcdf(
            file=file,
            da=data,
            format=format,
            shuffle=format,
            zlib=zlib,
            complevel=complevel,
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

    in_coords = xr.Dataset(
        coords={
            "lat": grid_in["lat"],
            "lon": grid_in["lon"],
        }
    )

    out_coords = xr.Dataset(
        coords={
            "lat": grid_out["lat"],
            "lon": grid_out["lon"],
        }
    )
    if parallel:
        out_coords["dummy"] = xr.DataArray(
            np.ones((out_coords.lat.size, out_coords.lon.size)),
            dims=("lat", "lon"),
            coords={"lat": out_coords.lat, "lon": out_coords.lon},
        )

        out_coords = out_coords.chunk({"lat": "auto", "lon": "auto"})

    weight_file = tmp / f"{method}_{grid_id(in_coords)}_{grid_id(out_coords)}"
    reuse = weight_file.exists()

    regridder = xe.Regridder(
        in_coords,
        out_coords,
        method=method,
        parallel=parallel,
        filename=str(weight_file),
        reuse_weights=reuse,
    )

    out = regridder(grid_in)

    return out


def mask(
    data: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | xr.Dataset | Path | None = None,
    data_var: str = "land",
    valid_value: float | int = 1,
    parallel: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Mask grid cells that do not match a specified land-sea mask value.

    The mask is remapped to the horizontal grid of ``data`` using
    nearest-neighbour interpolation. This method preserves categorical mask
    values. The remapped mask is cached so that repeated calls using the same
    mask and target-grid specification do not repeat the remapping operation.

    Before masking, ``data`` is sorted by increasing latitude and longitude.
    Consequently, the returned object may have a different coordinate order
    from the input.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Object to mask. It must contain one-dimensional ``lat`` and ``lon``
        coordinates and corresponding dimensions.

    mask : xarray.DataArray, xarray.Dataset, str, pathlib.Path, or None, optional
        Categorical land-sea mask.

        - If a DataArray is supplied, it is used directly.
        - If a Dataset is supplied, the variable named by ``data_var`` is used.
        - If a path is supplied, the Dataset at that path is opened and the
          variable named by ``data_var`` is used.
        - If None, the package's default land-sea mask is used.

        The mask must contain ``lat`` and ``lon`` coordinates. By convention,
        values equal to ``valid_value`` identify cells to retain.

    data_var : str, default "land"
        Name of the mask variable to extract when ``mask`` is a Dataset or a
        path to a Dataset. This argument is ignored when ``mask`` is already
        a DataArray.

    valid_value : float or int, default 1
        Mask value identifying grid cells to retain. Cells whose remapped mask
        value differs from ``valid_value`` are replaced with NaN.

    parallel : bool, default False
        Whether to perform mask remapping in parallel with Dask. This option
        is passed to :func:`remap`.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        A latitude- and longitude-sorted object with cells outside the
        retained mask category replaced by NaN. The return type matches the
        type of ``data``.

    Raises
    ------
    KeyError
        If ``mask`` resolves to a Dataset that does not contain ``data_var``.

    TypeError
        If ``mask`` cannot be resolved to an xarray.DataArray.

    Notes
    -----
    The cache key is based on the mask identity, mask-variable name, target
    coordinate bounds, and target-grid dimensions. The cached object is the
    remapped categorical mask, not the final Boolean mask, so different
    ``valid_value`` values may reuse the same cached remapping.
    """

    if mask is None:
        _default_mask = _script_dir / "data" / "mask" / "era5_0.25_mask"
        print(f"mask is None: Using {_default_mask}")
        mask = _default_mask

    if isinstance(mask, (str, Path)):
        mask = xr.open_dataset(mask)

    if isinstance(mask, xr.Dataset):
        if data_var not in mask:
            raise KeyError(
                f"Mask variable {data_var!r} not found; available: {list(mask.data_vars)}."
            )
        mask = mask[data_var].load()

    if not isinstance(mask, xr.DataArray):
        raise TypeError(f"mask must resolve to an xarray.DataArray, got {type(mask)}.")

    # Sort unconditionally: the regridded mask is cached, so the alignment
    # between data and mask must not depend on whether the cache was hit.
    data = data.sortby(["lat", "lon"])

    lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
    lat_min, lat_max = float(data.lat.min()), float(data.lat.max())

    subset_mask = mask.sortby(["lat", "lon"]).sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )
    remapped_mask = remap(subset_mask, data, method="nearest_s2d", parallel=parallel)

    return data.where(remapped_mask == valid_value, other=np.nan)


def add_local_solar_time(
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
    lst.attrs = {}
    lst.attrs["long_name"] = "Local Solar Time"
    lst.attrs["standard_name"] = "local_solar_time"
    lst.attrs["description"] = "Mean local solar time on whole-hour longitude zones"

    return data.assign_coords({name: lst})


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
