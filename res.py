from typing import Union

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
def get_spatiotemporal_info(
    obj: Union[xr.Dataset, xr.DataArray],
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
