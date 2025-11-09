from __future__ import annotations

import ast
import atexit
import logging
import tempfile
import uuid
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Literal, Mapping, Union

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
from cfgrib.dataset import DatasetBuildError

warnings.filterwarnings("ignore")


from .plot import animate, cartplot, make_cyclic
from .pycdo import cdo
from .tools import n_cpus, tmp_files
from .trends import calc_trends, polyfit

script_dir = Path(__file__).resolve().parent

current_dask_cluster = None
current_dask_client = None


def land_sea_mask(
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


def get_local_solar_time(data: xr.Dataset):
    """
    Calculate the local solar time for the dataset based on the longitude coordinate.
    The local solar time is calculated as the UTC time plus the longitude offset.
    """

    if "lon" not in data or "lat" not in data:
        raise ValueError("Dataset must contain 'lon' and 'lat' coordinates.")

    offset = data["lon"] * (24 / 360) * (data["lat"] / data["lat"])
    offset = offset.round() * pd.Timedelta(hours=1)

    lst = (data["time"] + offset).transpose("time", "lat", "lon")

    return data.assign_coords({"local_solar_time": lst})


def get_UTC_offset(
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


def chunk_by_dims(
    obj: xr.DataArray | xr.Dataset,
    dim: str,
    N: int,
) -> dict[str, xr.DataArray | xr.Dataset]:
    """
    Chunk an xarray DataArray or Dataset into N parts along the specified dimension.

    Parameters:
        obj (xr.DataArray or xr.Dataset): Input data to chunk.
        dim (str): Dimension along which to chunk.
        N (int): Number of chunks.

    Returns:
        dict[str, xr.DataArray or xr.Dataset]: Dictionary with keys '0', ..., '{N-1}'.
    """
    if dim not in obj.dims:
        raise ValueError(f"Dimension '{dim}' not found in the input data.")

    dim_size = obj.sizes[dim]
    if N < 1 or N > dim_size:
        raise ValueError(
            f"Invalid number of chunks N={N} for dimension size {dim_size}."
        )

    # Compute chunk indices
    indices = np.linspace(0, dim_size, N + 1, dtype=int)

    chunks = {}
    for i in range(N):
        chunk = obj.isel({dim: slice(indices[i], indices[i + 1])})
        chunks[f"{i}"] = chunk

    return chunks


def chunk_longitudes(
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


def chunk_by_timezones(
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
    chunks = chunk_longitudes(data)
    for chunk in chunks:
        offset = get_UTC_offset(chunk)
        data = chunks[chunk]
        data["time"] = data["time"] + offset
        timezone = str(np.timedelta64(offset, "h")).split(" ")[0]
        tz_chunks[f"UTC{timezone}"] = data

    return tz_chunks


def _process_chunk(func, kwargs, data, chunk):
    offset = get_UTC_offset(chunk)
    data["time"] = data["time"] + offset
    res = func(data, **kwargs)
    print(f"{float(chunk):7.1f}°E : UTC {np.timedelta64(offset, 'h'):>5} - Done")

    return res


def _tz_apply_func_parallel(
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


def _tz_apply_func_serial(
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


def tz_apply_func(
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

    if "lon" not in obj.dims or "lat" not in obj.dims or "time" not in obj.dims:
        raise ValueError(
            "The dataset must have (time, lat, lon) dimensions to apply the function."
        )

    if kwargs is None:
        kwargs = {}

    chunks = chunk_longitudes(obj)

    if multiprocess and len(chunks) > 1:
        return _tz_apply_func_parallel(func, chunks, kwargs)
    else:
        return _tz_apply_func_serial(func, chunks, kwargs)


def infer_time_frequency(
    times: Union[pd.Series, np.ndarray, xr.DataArray],
) -> tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Infer the time frequency of a series of timestamps.
    This function analyzes the time intervals in the provided timestamps and returns
    a frequency string along with the ranges of hours, months, and years present in the data.

    Parameters
    ----------
    times : Union[pd.Series, np.ndarray, xr.DataArray]
        A series of timestamps, which can be a pandas Series, a NumPy array, or
        an xarray DataArray containing datetime objects.
    Returns
    -------
    tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]
        A tuple containing:
        - freq: A string representing the inferred frequency (e.g., '1H', '1D', '1M').
        - hour_range: A tuple of integers representing the minimum and maximum hours present in the data.
        - month_range: A tuple of integers representing the minimum and maximum months present in the data.
        - year_range: A tuple of integers representing the minimum and maximum years present in the data.
    """

    time_vals = pd.DataFrame({"time": times})
    time_vals["diff"] = time_vals["time"].diff().dt.total_seconds()
    diffs = time_vals["diff"].value_counts()
    time_vals = time_vals[time_vals["diff"] == diffs.idxmax()]  # 2nd filtering

    mean_step_seconds = time_vals["diff"].mean()
    mean_step_seconds = int(mean_step_seconds)

    # Step 4: Infer frequency string
    freq = None

    if mean_step_seconds < 60:
        freq = f"{mean_step_seconds}S"  # seconds
    elif mean_step_seconds < 3600:
        freq = f"{mean_step_seconds // 60}T"  # minutes
    elif mean_step_seconds < 86400:
        freq = f"{mean_step_seconds // 3600}H"  # hours
    elif mean_step_seconds < 604800:
        freq = f"{mean_step_seconds // 86400}D"  # days
    elif mean_step_seconds < 2419200:
        freq = f"{mean_step_seconds // 604800}W"  # weeks
    elif mean_step_seconds < 29030400:
        freq = f"{mean_step_seconds // 2419200}M"  # months (approx 28 days)
    elif mean_step_seconds < 290304000:
        freq = f"{mean_step_seconds // 29030400}Y"  # years (approx 336 days)
    else:
        freq = f"{mean_step_seconds // 290304000}10Y"  # years (approx 336 days)

        # Time part ranges
    hour_range = (time_vals["time"].dt.hour.min(), time_vals["time"].dt.hour.max())
    month_range = (time_vals["time"].dt.month.min(), time_vals["time"].dt.month.max())
    year_range = (time_vals["time"].dt.year.min(), time_vals["time"].dt.year.max())

    return freq, hour_range, month_range, year_range


def interp_data(
    obj: xr.DataArray | xr.Dataset,
    resolution: float = 0.25,
    *,
    x: str = "lon",
    y: str = "lat",
    method: Literal["linear", "nearest", "cubic"] = "linear",
    bbox: tuple[float, float, float, float] = None,
) -> xr.DataArray | xr.Dataset:
    """
    Interpolate data to a regular grid using xarray.
    This function uses the xarray library to interpolate
    data to a regular grid. The function will create temporary files in the system
    temporary directory and delete them after use.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The data to be interpolated. The data must have latitude and longitude
        coordinates.
    resolution : float, optional
        The resolution of the output grid in degrees. The default is 0.25.
    method : str, optional
        The interpolation method to be used. The default is "linear".
        Other options are "nearest" and "cubic".
    x : str, optional
        The name of the longitude coordinate in the data. The default is "lon".
    y : str, optional
        The name of the latitude coordinate in the data. The default is "lat".


    """

    obj = obj.sortby([y, x])

    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox

        obj = obj.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    else:

        lat_min, lat_max = obj[y].min().values, obj[y].max().values
        lon_min, lon_max = obj[x].min().values, obj[x].max().values

    new_lat = np.arange(lat_min, lat_max, resolution)
    new_lon = np.arange(lon_min, lon_max, resolution)

    interp_data = obj.interp(lat=new_lat, lon=new_lon, method=method)

    return interp_data


# get the total number of grid points
def get_spatiotemporal_info(
    obj: xr.DataArray | xr.Dataset,
) -> dict:
    """
    Get the spatiotemporal information of an xarray object.
    This function extracts the dimensions, resolution, and time frequency of the provided
    xarray object, along with the bounds of each dimension.

    Parameters
    ----------
    obj : Union[xr.Dataset, xr.DataArray]
        An xarray object (either a Dataset or DataArray) containing spatial and temporal data.
    Returns
    -------
    dict
    """

    dims = list(obj.dims)

    resolution = {}

    t_freq, hours_range, months_range, years_range = None, None, None, None

    for k in dims:
        if str(obj[k].dtype) == "datetime64[ns]":
            t_freq, hours_range, months_range, years_range = infer_time_frequency(
                obj[k]
            )
            resolution[k] = t_freq

        else:
            resolution[k] = float(np.round(obj[k].diff(k).mean().values, 2))

    names = ["resolution", "hours_range", "months_range", "years_range"]
    data = [resolution, hours_range, months_range, years_range]

    result = {}
    for k, v in zip(names, data):
        if v is not None:
            result[k] = v

    for k in dims:
        if k != "time":
            result[f"{k}_bounds"] = (
                float(np.round(obj[k].min().values, 2)),
                float(np.round(obj[k].max().values, 2)),
            )

    return result


def close_dask():
    """
    Close the active Dask client and cluster if they exist.
    This is useful for cleaning up resources when done with Dask computations.
    """
    global current_dask_client, current_dask_cluster
    if current_dask_client and current_dask_cluster:
        current_dask_client.close()
        current_dask_cluster.close()
        current_dask_client = None
        current_dask_cluster = None


def setup_dask(
    *,
    workers: int = n_cpus,
    threads_per_worker: int = 1,
    processes=True,
    filter_warnings=True,
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

    global current_dask_client, current_dask_cluster

    if current_dask_client and current_dask_cluster:
        return current_dask_client

    from dask.distributed import Client, LocalCluster

    if filter_warnings:
        silence_level = logging.ERROR
    else:
        silence_level = logging.WARN

    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=threads_per_worker,
        memory_limit=0,
        silence_logs=silence_level,
        processes=processes,
    )
    client = Client(cluster)

    current_dask_client = client
    current_dask_cluster = cluster

    def _cleanup():
        current_dask_client.close()
        current_dask_cluster.close()

    atexit.register(_cleanup)

    return client


def open_grib_datatree(infile: Path) -> xr.DataTree:
    """
    Parse a GRIB file into separate xarray Datasets grouped by filter_by_keys.
    Handles both multi-level and single-level fields.
    """

    tmpdir = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}"
    tmp_files.append(tmpdir)
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
    ):
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
        use_dask: bool = True,
        dask_scheduler: Literal["threads", "processes"] = "threads",
    ) -> xr.Dataset:
        """
        Calculate trends along a specified dimension using the Mann-Kendall trend test.
        """
        return calc_trends(
            self,
            along=along,
            scale=scale,
            use_dask=use_dask,
            dask_scheduler=dask_scheduler,
        )

    def trendfit(self, along: str, data_var: str = None, scale: float = 1):
        """
        Calculate the linear trend for the given xarray Dataset or DataArray using polynomial fitting.
        """
        return polyfit(
            self,
            along=along,
            data_var=data_var,
            scale=scale,
        )


# Alias for convenience


@xr.register_dataarray_accessor("cartplot")
class CartPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, *args, **kwargs):
        return cartplot(self._obj, *args, **kwargs)

    def animate(self, *args, **kwargs):
        return animate(self._obj, *args, **kwargs)
