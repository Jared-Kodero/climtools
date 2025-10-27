import atexit
import functools
import getpass
import inspect
import logging
import os
import shutil
import socket
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Literal, Mapping, Union

import numpy as np
import pandas as pd
import xarray as xr

from .logs import *

host = socket.gethostname()
user = getpass.getuser()
home = Path.home()
tmp = Path(os.environ.get("TMPDIR", "/tmp"))
n_cpu = len(os.sched_getaffinity(0))
SCRIPT_DIR = Path(__file__).resolve().parent
CURRENT_DASK_CLUSTER = None
CURRENT_DASK_CLIENT = None
_TMP_FILES = []


class ConfigMap(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def cwd():
    """
    Get the current working directory.
    """
    return Path.cwd().resolve()


def cleanup():
    rm(_TMP_FILES)


atexit.register(cleanup)


def execute_cmd(cmd: list[str]):

    try:
        res = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )
        return res
    except subprocess.CalledProcessError as e:
        print("ERROR :", e.stderr)
        return None


def type_cast(x, use_numpy: bool = False):
    """
    Cast input x to int or float (optionally using numpy types).
    Returns the original input if casting fails.
    """
    if x is None or str(x).strip() == "":
        return np.nan if use_numpy else None

    _int = np.int64 if use_numpy else int
    _float = np.float64 if use_numpy else float

    try:
        if "." in str(x):
            res = _float(x)
        else:
            res = _int(x)
    except (ValueError, TypeError):
        try:
            res = _float(x)
        except (ValueError, TypeError):
            res = None

    if res is None:
        return x

    return res


def timeit(func):
    """
    Decorator to time a function and print its runtime in appropriate units.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)
        end = time.perf_counter()

        elapsed = end - start
        unit = "seconds"

        if elapsed > 86400:  # > 1 day
            elapsed /= 86400
            unit = "days"
        elif elapsed > 3600:  # > 1 hour
            elapsed /= 3600
            unit = "hours"
        elif elapsed > 60:  # > 1 minute
            elapsed /= 60
            unit = "minutes"

        print(f"[ {func.__name__} ] finished in {elapsed:.2f} {unit}")
        return result

    return wrapper


def mkdir(path: Path | PathLike):
    """
    Create a directory using the mkdir command in unix-like systems.

    """
    path = Path(path).resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def get_func_signature(func):
    """
    Get the signature of a function as a dictionary.
    """
    sig = inspect.signature(func)
    return {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in sig.parameters.items()
    }


def file_type(file_path: Path | PathLike) -> str:
    """
    Get the file type using the `file` command in unix-like systems.
    """

    if isinstance(file_path, Path):
        file_path = str(file_path)

    res = execute_cmd(["file", "-b", file_path])
    return res.stdout.strip()


def symlink(
    src: Path | PathLike,
    dst: Path | PathLike,
):
    """
    Create a symbolic link from src to dst.
    """
    src = Path(src).resolve()
    dst = Path(dst).resolve()

    # Create parent directories for the link
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing link or file if already exists
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    # Create symbolic link
    if src.is_dir():
        dst.symlink_to(src, target_is_directory=True)
    elif src.is_file():
        dst.symlink_to(src)
    else:
        raise FileNotFoundError(f"Source path does not exist: {src}")


def rm(arg: Path | PathLike | list[Path | PathLike]):
    """
    Remove files or directories

    """

    if not isinstance(arg, list):
        arg = [arg]

    for f in arg:
        f = Path(f).resolve()
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f, ignore_errors=True)


def cp(
    src: Path | PathLike,
    dst: Path | PathLike,
):
    """
    Copy files or directories

    """

    src = Path(src).resolve()
    dst = Path(dst).resolve()

    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    elif src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def mv(
    src: Path | PathLike,
    dst: Path | PathLike,
):
    """
    Move files or directories

    """

    src = Path(src).resolve()
    dst = Path(dst).resolve()

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dst)


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

    file = SCRIPT_DIR / "data" / masks[mask_file]

    mask = xr.open_dataset(file)

    obj = obj.sortby(["lat", "lon"])

    lat_min, lat_max = obj.lat.min().values, obj.lat.max().values
    lon_min, lon_max = obj.lon.min().values, obj.lon.max().values

    mask = mask[keep].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    mask = mask.interp(lat=obj.lat, lon=obj.lon, method="nearest")

    if isinstance(obj, xr.Dataset):

        log(f"Removing feature(s): {keep} in {list(obj.data_vars)}", level="WARNING")
        new_obj = xr.Dataset()
        for data_var in list(obj.data_vars):
            new_obj[data_var] = obj[data_var].where(mask, other=np.nan)

    elif isinstance(obj, xr.DataArray):
        log(f"Removing feature(s): {keep} in {obj.name}", level="WARNING")
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

    processes = max(1, min(n_cpu, len(args)))
    chunksize = max(1, len(args) // n_cpu)

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
    global CURRENT_DASK_CLIENT, CURRENT_DASK_CLUSTER
    if CURRENT_DASK_CLIENT and CURRENT_DASK_CLUSTER:
        CURRENT_DASK_CLIENT.close()
        CURRENT_DASK_CLUSTER.close()
        CURRENT_DASK_CLIENT = None
        CURRENT_DASK_CLUSTER = None


def setup_dask(
    *,
    workers: int = n_cpu,
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

    global CURRENT_DASK_CLIENT, CURRENT_DASK_CLUSTER

    if CURRENT_DASK_CLIENT and CURRENT_DASK_CLUSTER:
        return CURRENT_DASK_CLIENT

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

    CURRENT_DASK_CLIENT = client
    CURRENT_DASK_CLUSTER = cluster

    def _cleanup():
        CURRENT_DASK_CLIENT.close()
        CURRENT_DASK_CLUSTER.close()

    atexit.register(_cleanup)

    return client
    return client
    return client
