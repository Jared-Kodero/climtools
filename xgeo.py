from __future__ import annotations

import ast
import logging
import tempfile
import uuid
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Literal, Mapping, Tuple, Union

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from cfgrib.dataset import DatasetBuildError
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from .statistics import *
from .tools import n_cpus

warnings.filterwarnings("ignore")


from .plot import animate, cartplot, make_cyclic
from .pycdo import cdo
from .tools import _tmp_files, n_cpus

script_dir = Path(__file__).resolve().parent


def open_grib_datatree(infile: Path) -> xr.DataTree:
    """
    Parse a GRIB file into separate xarray Datasets grouped by filter_by_keys.
    Handles both multi-level and single-level fields.
    """

    tmpdir = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}"
    _tmp_files.append(tmpdir)
    files = cdo.split(infile, operator="splitname", outdir=tmpdir)

    combined_datasets = {}
    rejected_singles = {}
    single_level_datasets = []

    standard_dims = ["time", "latitude", "longitude", "lat", "lon", "x", "y"]

    for f in files:
        datasets = {}
        try:
            ds = xr.open_dataset(f, engine="cfgrib").squeeze()
            dims = list(ds.dims)

            # Determine expected dimensionality
            single_dims = 3 if ("time" in dims and len(ds["time"]) > 1) else 2

            # Detect non-standard dimensions (vertical or ensemble)
            extra_dims = [d for d in dims if d not in standard_dims]

            if len(dims) > single_dims:
                # Multi-level dataset detected
                if len(extra_dims) != 1:
                    raise Exception("Too many non-standard dimensions in dataset")

                level_dim = extra_dims[0]
                rejected_singles.setdefault(level_dim, []).append(ds)
                continue  # handled; skip to next file

            # Otherwise, treat as single-level
            single_level_datasets.append(ds)

        except DatasetBuildError as e:
            # Handle multi-field (multi-level) GRIB subsets
            lines = str(e).split("\n")[1:]
            for line in lines:
                if "=" in line:

                    keys = ast.literal_eval(line.split("=", 1)[1])
                    key = list(keys.keys())[0]
                    value = keys[key]
                    ds = xr.open_dataset(f, engine="cfgrib", filter_by_keys=keys)
                    datasets.setdefault(key, {})[value] = ds.squeeze()

        grouped_datasets = {}
        for key, value in datasets.items():
            for subkey, ds in value.items():
                grouped_datasets.setdefault(subkey, []).append(ds)

        for subkey, rejected in rejected_singles.items():
            if subkey in grouped_datasets:
                grouped_datasets[subkey].extend(rejected)
            else:
                grouped_datasets[subkey] = list(rejected)

        for k, groups in grouped_datasets.items():
            combined = xr.merge(groups) if len(groups) > 1 else groups[0]
            combined_datasets[k] = combined

        single_level = xr.merge(single_level_datasets, compat="override")
        combined_datasets["single_level"] = single_level

        dt = xr.DataTree.from_dict(combined_datasets)

    return dt


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
        lsted = lst(self, longitude=longitude)
        return GeoDataArray(lsted)

    def mask(
        self,
        *,
        keep: Literal["land", "ocean"] = None,
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

    def cartplot(
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
        figsize: tuple[float, float] = None,
        # Plot appearance
        plot_type: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow"
        ] = "default",
        cmap: str | mcolors.Colormap = None,
        vmin: float = None,
        vmax: float = None,
        levels: int | list = None,
        extend: str = None,
        robust: bool = False,
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
        edgecolor: str = "face",
        **kwargs,
    ) -> Tuple[Figure, Axes | GeoAxes, QuadMesh | QuadContourSet | AxesImage | Artist]:
        """
        Plot this DataArray on a Cartopy map using the global `cartplot()` function.
        """
        return cartplot(
            self,
            x=x,
            y=y,
            projection=projection,
            global_extent=global_extent,
            figsize=figsize,
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            plot_type=plot_type,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            robust=robust,
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
            edgecolor=edgecolor,
            **kwargs,
        )

    def animate(
        self,
        dim: str = "time",
        *,
        indices: tuple | list | np.ndarray = None,
        outfile: Path | str = None,
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
        figsize: tuple[float, float] = None,
        central_longitude: float = None,
        central_latitude: float = None,
        # Plot appearance
        plot_type: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow"
        ] = "default",
        cmap: str | mcolors.Colormap = None,
        vmin: float = None,
        vmax: float = None,
        levels: int | list = None,
        extend: str = None,
        robust: bool = False,
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
        edgecolor: str = "face",
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
            figsize=figsize,
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            plot_type=plot_type,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            robust=robust,
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
            edgecolor=edgecolor,
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

    def period_difference(
        self,
        period1: tuple[str | pd.Timestamp],
        period2: tuple[str | pd.Timestamp],
        along: str = "time",
        level: float = 0.05,
    ) -> xr.Dataset:
        """
        Compute the difference between two time periods in a DataArray or Dataset.

        Parameters
        ----------
        period1 : tuple of str
            (start, end) timestamps for the first period.
        period2 : tuple of str
            (start, end) timestamps for the second period.
        along : str, default "time"
            Name of the time dimension.
        level : float, default 0.05
            Significance level for the significance test.

        Returns
        -------
        xr.Dataset
            Dataset containing the mean difference between the two periods.
        """
        return period_difference(
            self,
            period1=period1,
            period2=period2,
            along=along,
            level=level,
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


class GeoDataset(xr.Dataset):
    __slots__ = ()

    def __getitem__(self, key):
        obj = super().__getitem__(key)
        if isinstance(obj, xr.DataArray):
            return GeoDataArray(obj)
        return obj


class Daskit:
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
        - Imports Dask and Dask distributed.
        - Creates a Dask client.
        - Sets up the Dask dashboard.

        Parameters:
        ___________
            workers (int, optional): Number of workers to create. Default is 8.
            threads_per_worker (int, optional): Number of threads per worker. Default is 4.
            processes (bool, optional): Whether to use processes instead of threads. Default is True.
            get_info (bool, optional): Whether to return the Dask dashboard URL. Default is False.
            dynamic_port (bool, optional): Whether to use a dynamic port. Default is False and uses port 8787.
            filter_warnings (bool, optional): Whether to filter warnings. Default is True.

        Example:
        ________
            >>> setup_dask(get_info=True, filter_warnings=False)
        """

        if self.client is not None:
            return self.client

        from dask.distributed import Client, LocalCluster

        if self.filter_warnings:
            silence_level = logging.ERROR
        else:
            silence_level = logging.WARN

        self.cluster = LocalCluster(
            n_workers=self.workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
            silence_logs=silence_level,
            processes=self.processes,
        )
        self.client = Client(self.cluster)

        return self.client

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def mask(
    obj: xr.DataArray | xr.Dataset,
    *,
    keep: Literal["land", "ocean"] = None,
    mask_file: Literal["cartopy", "era5"] = "era5",
) -> xr.DataArray | xr.Dataset:
    """
    Apply a land-sea mask to the dataset. This function uses a netCDF file containing
    land-sea masks to filter out specific features from the dataset.
    """

    if "lat" not in obj.dims or "lon" not in obj.dims:
        raise ValueError(
            "The dataset must have 'lat' and 'lon' dimensions to apply the land-sea mask."
        )

    masks = {
        "cartopy": "cartopy_0.1.mask",
        "era5": "era5_0.25_mask",
    }

    file = script_dir / "data" / "mask" / masks[mask_file]

    mask = xr.open_dataset(file)

    obj = obj.sortby(["lat", "lon"])

    lat_min, lat_max = obj.lat.min().values, obj.lat.max().values
    lon_min, lon_max = obj.lon.min().values, obj.lon.max().values

    mask = mask[keep].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    mask = mask.interp(lat=obj.lat, lon=obj.lon, method="nearest")

    if isinstance(obj, xr.Dataset):
        new_obj = xr.Dataset()
        for data_var in list(obj.data_vars):
            new_obj[data_var] = obj[data_var].where(mask, other=np.nan)

    elif isinstance(obj, xr.DataArray):
        new_obj = obj.where(mask, other=np.nan)

    return new_obj


def lst(
    data: xr.Dataset | xr.DataArray, *, longitude="lon"
) -> xr.Dataset | xr.DataArray:
    """
    Calculate the local solar time for the dataset based on the longitude coordinate.
    The local solar time is calculated as the UTC time plus the longitude offset.
    """

    if longitude not in data:
        raise ValueError(f"Dataset must contain '{longitude}' coordinate.")

    offset = data[longitude] * (24 / 360)
    offset = offset.round() * pd.Timedelta(hours=1)

    lst = data["time"] + offset
    lst.attributes["long_name"] = "Local Solar Time"
    lst.attributes["standard_name"] = "local_solar_time"

    return data.assign_coords({"lst": lst})


def utc_offset(
    obj: int | float | np.ndarray | xr.DataArray | xr.Dataset,
    *,
    name: Literal["lon", "longitude", "x"] = "lon",
) -> int | float | np.ndarray | xr.DataArray | xr.Dataset:
    """
    Computes the hour offset from UTC time based on the longitude coordinate.
    """

    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        data = obj.copy()
        data[name] = ((data[name] + 180) % 360) - 180
        offset = data[name] * (24 / 360)
        offset = offset.round() * pd.Timedelta(hours=1)

    else:
        if isinstance(obj, str):
            obj = float(obj)
        obj = ((obj + 180) % 360) - 180
        offset = obj * (24 / 360)
        offset = np.round(offset) * pd.Timedelta(hours=1)

    return offset


def chunk_by_lon(
    obj: Union[xr.DataArray, xr.Dataset],
    *,
    lon: str = "lon",
    deg: int = 15,
) -> dict[str, Union[xr.DataArray, xr.Dataset]]:
    """
    Partition a dataset into longitude bins of fixed width.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The input data containing a longitude coordinate.
    lon : {"lon", "longitude", "x"}, optional
        Name of the longitude coordinate. Default is "lon".
    deg : int, optional
        Bin width in degrees of longitude. Default is 15.

    Returns
    -------
    dict[str, xarray.DataArray or xarray.Dataset]
        A dictionary mapping each longitude bin (e.g., "-180", "-165", ..., "165")
        to the corresponding subset of the data. The longitude values within each
        bin are restored to their original values and sorted.

    Notes
    -----
    - Longitudes are first normalized to the range [-180, 180).
    - Each partition is labeled by its central longitude value (rounded to the nearest
      multiple of `deg`).
    """

    chunks = {}
    data = obj.copy()
    data[lon] = ((data[lon] + 180) % 360) - 180

    original_lons = data[lon]
    lon_rounded = (original_lons / deg).round() * deg

    idx_df = pd.DataFrame(
        {
            "lon_original": original_lons,
            "lon_rounded": lon_rounded,
        }
    )

    data[lon] = idx_df["lon_rounded"].values

    for lon_val in idx_df["lon_rounded"].unique():
        lon_val_data = data.sel({lon: lon_val})

        lon_val_df = idx_df[idx_df["lon_rounded"] == lon_val]
        lon_val_data[lon] = lon_val_df["lon_original"].values

        chunks[f"{lon_val}"] = lon_val_data.sortby(lon)

    return chunks


def chunk_by_tz(
    obj: xr.DataArray | xr.Dataset,
) -> dict[str, xr.DataArray | xr.Dataset]:
    """
    Chunk the dataset into chunks based on time zones.
    This function splits the dataset into 15-degree longitude chunks and adjusts the time coordinate
    based on the hour offset from UTC time for each chunk.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The dataset to be split into time zone chunks.

    Returns
    -------
    dict[str, xarray.DataArray or xarray.Dataset]
        A dictionary where keys are time zone identifiers (e.g., "UTC+0", "UTC+1", etc.)
        and values are the corresponding xarray objects for each time zone chunk.
    """

    data = obj.copy()
    tz_chunks = {}
    chunks = chunk_by_lon(data)
    for chunk in chunks:
        offset = utc_offset(chunk)
        data = chunks[chunk]
        data["time"] = data["time"] + offset
        timezone = str(np.timedelta64(offset, "h")).split(" ")[0]
        tz_chunks[f"UTC{timezone}"] = data

    return tz_chunks


def _process_chunk(func, kwargs, data, chunk):
    offset = utc_offset(chunk)
    data["time"] = data["time"] + offset
    res = func(data, **kwargs)
    print(f"{float(chunk):7.1f}°E : UTC {np.timedelta64(offset, 'h'):>5} - Done")

    return res


def _tz_apply_parallel(
    func: Callable,
    chunks: dict[str, xr.DataArray | xr.Dataset],
    kwargs: Mapping | None,
) -> xr.DataArray | xr.Dataset:

    args = [(func, kwargs, chunks[chunk], chunk) for chunk in chunks]

    processes = max(1, min(n_cpus, len(args)))
    chunksize = max(1, len(args) // n_cpus)

    if chunksize == 1:
        maxtasksperchild = 2
    else:
        maxtasksperchild = chunksize

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        datasets = pool.starmap(_process_chunk, args, chunksize=chunksize)

    kwargs = {
        "dim": "lon",
        "join": "exact",
        "compat": "override",
        "data_vars": "minimal",
        "coords": "minimal",
    }

    return xr.concat(datasets, **kwargs).sortby("lon")


def _tz_apply_serial(
    func,
    chunks,
    kwargs,
):
    datasets = []
    for chunk in chunks:
        datasets.append(_process_chunk(func, kwargs, chunks[chunk], chunk))

    if len(datasets) > 1:
        result = xr.concat(datasets, dim="lon").sortby("lon")
    else:
        result = datasets[0]

    return result


def apply_func_by_time_zone(
    func: Callable,
    obj: xr.DataArray | xr.Dataset,
    multiprocess: bool = True,
    kwargs: Mapping | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    Process the dataset by time zones using a specified function.

    Parameters
    ----------
    func : Callable,
        The function to be applied to each chunk of the dataset.
    obj : xarray.DataArray or xarray.Dataset
        The dataset to be processed.
    multiprocess : bool, optional
        If True, the function will be applied in parallel using multiple processes.
    **kwargs : Any
        Additional keyword arguments to be passed to the applied function.
    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The processed dataset after applying the function to each time zone chunk.

    """

    if "longitude" in obj.dims:
        obj = obj.rename({"longitude": "lon"})
    elif "latitude" in obj.dims:
        obj = obj.rename({"latitude": "lat"})

    if "lon" not in obj.dims or "lat" not in obj.dims or "time" not in obj.dims:
        raise ValueError(
            "The dataset must have (time, lat, lon) dimensions to apply the function."
        )

    if kwargs is None:
        kwargs = {}

    chunks = chunk_by_lon(obj)

    if multiprocess and len(chunks) > 1:
        return _tz_apply_parallel(func, chunks, kwargs)
    else:
        return _tz_apply_serial(func, chunks, kwargs)
