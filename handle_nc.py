import subprocess
import uuid
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Callable, Literal, Mapping, Union

import numpy as np
import pandas as pd
import xarray as xr
from pac_man import which

from .tools import _TMP_FILES, CPU_COUNT, CWD, SCRIPT_DIR, log, rm

# from joblib import Parallel, delayed


def cdo_mergetime(
    infiles: list[str],
    outfile: str,
    *,
    delete_input: bool = False,
):
    """
    Merge multiple netCDF files along the time dimension using CDO.
    This function uses the Climate Data Operators (CDO) to merge multiple netCDF files.

    Parameters
    ----------
    infiles : list[str]
        List of input netCDF files to be merged.
    outfile : str
    """

    try:
        is_cdo = which("cdo")
        if not is_cdo:
            raise FileNotFoundError(
                "CDO is not installed or not available in the system path."
            )

        infiles = [str(Path(f).resolve()) for f in infiles]
        infiles.sort()

        subprocess.run(
            [
                "cdo",
                "--no_history",
                "-s",
                "-w",
                "-z",
                "zip",
                "-mergetime",
                *infiles,
                outfile,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if Path(outfile).exists() and delete_input:
            rm(infiles)

        return outfile

    except subprocess.CalledProcessError as e:
        print("ERROR:", e.stderr)


def cdo_interp_data(
    infile: Path | PathLike,
    outfile: Path | PathLike,
    *,
    resolution: float = 0.25,
    method: Literal["remapdis", "remapcon", "remapbil"] = "remapdis",
    bbox: tuple[float, float, float, float] = None,
) -> xr.DataArray | xr.Dataset | str | PathLike:
    """
    Interpolate data to a regular grid using CDO. This function uses the Climate Data Operators (CDO) to interpolate
    data to a regular grid. The CDO must be installed and available in the
    system path. The function will create temporary files in the system
    temporary directory and delete them after use.

    Parameters
    ----------
    infile : PathLike
        The input netCDF file to be interpolated.
    outfile : PathLike
        The output netCDF file after interpolation.
    resolution : float, optional
        The resolution of the output grid in degrees. The default is 0.25.
    method : str, optional
        The interpolation method to be used. The default is "remapdis".
        Other options are "remapcon" and "remapbil".
    bbox : tuple, optional
        The bounding box of the data in the form (lon_min, lat_min, lon_max, lat_max).
        If not provided, the bounding box will be determined from the data.

    pack : bool, optional
        If True, the function will pack the output file using CDO. The default is False.
        This will reduce the file size and improve performance when reading the file.

    """

    is_cdo = which("cdo")
    if not is_cdo:
        raise ValueError("CDO is not installed or not available in the system path.")

    CWD_DIR = CWD()

    FUNC_TMP = CWD_DIR / ".tmp"
    FUNC_TMP.mkdir(exist_ok=True)

    grdfile = f"{FUNC_TMP}/{uuid.uuid4()}.grid"

    if bbox:
        lon_min, lat_min, lon_max, lat_max = bbox
    else:
        raise ValueError(
            "Bounding box (bbox) must be provided in the form (lon_min, lat_min, lon_max, lat_max)."
        )

    xsize = int((lon_max - lon_min) / resolution + 1)
    ysize = int((lat_max - lat_min) / resolution + 1)

    grid_description = f"""
    gridtype = lonlat
    xsize = {xsize}
    ysize = {ysize}
    xfirst = {lon_min}
    xinc = {resolution}
    yfirst = {lat_min}
    yinc = {resolution}
    """

    with open(grdfile, "w") as f:
        f.write(grid_description.strip())

    try:

        subprocess.run(
            [
                "cdo",
                "--no_history",
                "-s",
                "-w",
                "-P",
                "4",
                f"{method},{grdfile}",
                infile,
                outfile,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except subprocess.CalledProcessError as e:
        print("ERROR:", e.stderr)
        return None

    _TMP_FILES.extend([outfile, FUNC_TMP])

    return outfile


def xr_interp_data(
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
    bbox : tuple, optional
        The bounding box of the data in the form (lon_min, lat_min, lon_max, lat_max).
        If not provided, the bounding box will be determined from the data.

    """
    try:

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
    except Exception:
        log()


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

    mask = xr.open_dataarray(file)

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


def split_data_by_dims(
    obj: xr.DataArray | xr.Dataset,
    dim: str,
    N: int,
) -> dict[str, xr.DataArray | xr.Dataset]:
    """
    Splits an xarray DataArray or Dataset into N parts along the specified dimension.

    Parameters:
        obj (xr.DataArray or xr.Dataset): Input data to split.
        dim (str): Dimension along which to split.
        N (int): Number of splits.

    Returns:
        dict[str, xr.DataArray or xr.Dataset]: Dictionary with keys 'split_0', ..., 'split_{N-1}'.
    """
    if dim not in obj.dims:
        raise ValueError(f"Dimension '{dim}' not found in the input data.")

    dim_size = obj.sizes[dim]
    if N < 1 or N > dim_size:
        raise ValueError(
            f"Invalid number of splits N={N} for dimension size {dim_size}."
        )

    # Compute split indices
    indices = np.linspace(0, dim_size, N + 1, dtype=int)

    splits = {}
    for i in range(N):
        split = obj.isel({dim: slice(indices[i], indices[i + 1])})
        splits[f"{i}"] = split

    return splits


def split_by_15_deg(
    obj: Union[xr.DataArray, xr.Dataset],
    *,
    name: Literal["lon", "longitude", "x"] = "lon",
) -> dict[str, Union[xr.DataArray, xr.Dataset]]:
    """
    Split the dataset into 15-degree longitude chunks.
    """

    chunks = {}
    data = obj.copy()
    data[name] = ((data[name] + 180) % 360) - 180

    original_lons = data[name]
    lon_rounded = (original_lons / 15).round() * 15

    idx_df = pd.DataFrame(
        {
            "lon_original": original_lons,
            "lon_rounded": lon_rounded,
        }
    )

    data[name] = idx_df["lon_rounded"].values

    for lon_val in idx_df["lon_rounded"].unique():
        lon_val_data = data.sel({name: lon_val})

        lon_val_df = idx_df[idx_df["lon_rounded"] == lon_val]
        lon_val_data[name] = lon_val_df["lon_original"].values

        chunks[f"{lon_val}"] = lon_val_data.sortby(name)

    return chunks


def split_data_by_timezones(
    obj: xr.DataArray | xr.Dataset,
) -> dict[str, xr.DataArray | xr.Dataset]:
    """
    Split the dataset into chunks based on time zones.
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
    chunks = split_by_15_deg(data)
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
    func,
    chunks,
    kwargs,
):

    args = [(func, kwargs, chunks[chunk], chunk) for chunk in chunks]

    processes = max(1, min(CPU_COUNT, len(args)))
    chunksize = max(1, len(args) // CPU_COUNT)

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

    chunks = split_by_15_deg(obj)

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
