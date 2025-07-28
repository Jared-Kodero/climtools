from collections import namedtuple
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
import xarray as xr


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
def get_grid_resolution(
    obj: Union[xr.Dataset, xr.DataArray],
    x: str = "lon",
    y: str = "lat",
    time: str = None,
) -> NamedTuple:
    """
    Get the resolution of a dataset along specified dimensions.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The dataset or data array to analyze.
    x : str
        The name of the x dimension (e.g., 'lon').
    y : str
        The name of the y dimension (e.g., 'lat').
    time : str, optional
        The name of the time dimension (e.g., 'time'). If provided, the function
        will also return the inferred time frequency.
    Returns
    -------
    namedtuple : A named tuple containing:
        - n_cells: Total number of grid cells.
        - resolution: A tuple of (lat_res, lon_res) representing the spatial resolution.
        - freq: The inferred time frequency (if time is provided).
        - hours: A tuple of (min_hour, max_hour) representing the range of hours in the time dimension.
        - months: A tuple of (min_month, max_month) representing the range of months in the time dimension.
        - years: A tuple of (min_year, max_year) representing the range of years in the time dimension.
        - bbox: A tuple of (min_lon, min_lat, max_lon, max_lat) representing the bounding box of the dataset.
    """

    if not isinstance(obj, (xr.Dataset, xr.DataArray)):
        raise TypeError("Input must be an xarray Dataset or DataArray.")

    if x not in obj.dims or y not in obj.dims:
        raise ValueError(f"Dimensions {x}, {y}, not found in the provided dataset.")
    if time and time not in obj.dims:
        raise ValueError(f"Dimension {time} not found in the provided dataset.")

    time_res = None, None, None, None
    n_cells = obj.sizes[y] * obj.sizes[x]
    lon_res = np.abs(np.round(obj[x].diff(x).mean().values, 2))
    lat_res = np.abs(np.round(obj[y].diff(y).mean().values, 2))

    res = (float(np.round(lat_res, 2)), float(np.round(lon_res, 2)))

    bbox = (
        float(np.round(obj[x].min().values, 2)),
        float(np.round(obj[y].min().values, 2)),
        float(np.round(obj[x].max().values, 2)),
        float(np.round(obj[y].max().values, 2)),
    )

    if time:
        time_res = infer_time_frequency(obj[time])

    names = [
        "n_cells",
        "resolution",
    ]
    data = [
        float(n_cells),
        res,
    ]
    if time:
        names.extend(["freq", "hours", "months", "years"])
        data.extend([*time_res])

    names.append("bbox")
    data.append(bbox)

    return namedtuple("info", names)(*data)
