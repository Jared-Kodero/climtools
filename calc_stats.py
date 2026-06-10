"""
This module provides functions for calculating trends and correlations in xarray DataArrays and Datasets.
It includes implementations of the Mann-Kendall trend test, linear regression using polynomial fitting, and correlation tests (Pearson, Spearman, Kendall).
The functions are designed to handle missing data and can be applied along specified dimensions. Dask is supported for parallelized computations on large datasets.
"""

import warnings
from typing import Literal, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr

try:
    import pymannkendall as mk
except ImportError:
    warnings.warn(
        "pymannkendall is not installed. Trend calculation functionalities will not work."
    )


def _mktrend_test(
    array: np.ndarray,
    scale: float = 1,
) -> np.ndarray:
    result_array = None
    nan_array = np.array([np.nan] * 7)

    df = pd.DataFrame({"array": array})
    df = df.dropna()

    try:
        if not df.empty or len(df) < 2:
            result = mk.hamed_rao_modification_test(df["array"])
            mean_val = df["array"].mean()
            std_val = df["array"].std()

            trend = {"increasing": 1, "decreasing": -1}.get(result.trend, 0)
            stats = [
                result.slope * scale,
                result.p,
                trend,
                mean_val,
                std_val,
                result.Tau,
                result.z,
            ]

            result_array = np.array(stats)

            return result_array
        else:
            return nan_array
    except Exception:
        return nan_array


def _polyfit(data: xr.DataArray | xr.Dataset, dim: str, data_var=None, scale=1):
    """
    Calculate the linear trend for the given xarray Dataset or DataArray using xr.polyfit.

    - data: xr.Dataset
    - data_var: The variable to calculate the trend test for.
    - along: dim to calculate the trend test along. also used for sorting the data.
    - scale: The scale to multiply the slope by i.e convert to per hour, per day, etc.

    Returns: xr.Dataset
    """

    data.attrs = {}
    data = data.sortby(dim)
    data[dim] = (np.arange(1, len(data[dim]) + 1)).astype(np.int32)
    n = data.sizes[dim]

    if isinstance(data, xr.Dataset):
        if data_var is None:
            raise ValueError("Argument 'data_var' is required for xr.Dataset input.")
        data = data[data_var]

    res = data.polyfit(dim=dim, deg=1, cov=True)
    slope = res["polyfit_coefficients"].sel(degree=1)
    slope_variance = res["polyfit_covariance"].sel(cov_i=0, cov_j=0)
    stderr = slope_variance**0.5
    t_stat = slope / stderr

    p_values = xr.DataArray(
        2 * (1 - stats.t.cdf(np.abs(t_stat), (n - 2))),
        coords=slope.coords,
        dims=slope.dims,
    )

    mean_val = data.mean(dim=dim)
    std_val = data.std(dim=dim)

    trends = xr.Dataset()
    trends["slope"] = slope * scale
    trends["p_value"] = p_values
    trends["mean_val"] = mean_val
    trends["std_val"] = std_val

    # add attributes
    trends["slope"].attrs = {
        "long_name": "slope",
        "description": f"Slope of the linear trend per {scale} units of {dim}",
    }
    trends["p_value"].attrs = {
        "long_name": "p_value",
        "description": "p-value of the trend significance test",
    }
    trends["mean_val"].attrs = {
        "long_name": "mean_val",
        "description": f"Mean value along {dim}",
    }
    trends["std_val"].attrs = {
        "long_name": "std_val",
        "description": f"Standard deviation along {dim}",
    }

    return trends


def _corr_test(
    array_x: np.ndarray,
    array_y: np.ndarray,
    corr_type: str,
    alternative: str = "two-sided",
) -> np.ndarray:
    nan_list = [np.nan] * 2

    nan_data = np.array(nan_list)

    df = pd.DataFrame({"x": array_x, "y": array_y})
    df = df.dropna()
    if df.empty or len(df) < 2:
        return nan_data

    corr = np.nan
    p_value = np.nan

    if corr_type == "pearson":
        corr, p_value = pearsonr(df["x"], df["y"], alternative=alternative)

    elif corr_type == "spearman":
        corr, p_value = spearmanr(df["x"], df["y"], alternative=alternative)

    elif corr_type == "kendall":
        corr, p_value = kendalltau(df["x"], df["y"], alternative=alternative)

    del array_x, array_y, df

    stats = [corr, p_value]

    array = np.array(stats)

    return array


def corr(
    x: xr.DataArray,
    y: xr.DataArray,
    *,
    corr_type: Literal["pearson", "spearman", "kendall"] = "pearson",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    dim: str = None,
    dask_scheduler: Literal["threads", "processes"] = "threads",
) -> xr.Dataset:
    """
    Compute correlation coefficients between two xarray objects.

    This function evaluates the association between two variables over a
    specified dimension using Pearson, Spearman, or Kendall correlation.
    When x or y are `xr.Dataset`, a target variable must be specified via
    `x_var` or `y_var`. The statistical test is applied independently at
    each grid point across all remaining dimensions.

    Parameters
    ----------
    x : xr.DataArray
        First input.
    y : xr.DataArray
        Second input.
    corr_type : {"pearson", "spearman", "kendall"}, default "pearson"
        Type of correlation coefficient:
        - "pearson": linear correlation
        - "spearman": rank correlation
        - "kendall": Kendall τ rank correlation
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Defines the alternative hypothesis for the p-value calculation.
    dim : str
        Dimension along which the correlation is computed (e.g., "time").
        Required for xarray objects.
    dask_scheduler : {"threads", "processes"}, default "threads"
        Scheduler used when computing correlations lazily with Dask.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - "corr": correlation coefficient
        - "p": p-value associated with the chosen alternative hypothesis
        Additional attributes describe the correlation type and hypothesis.

    """

    out_vars = [
        "corr",
        "p_value",
    ]

    if isinstance(x, xr.Dataset):
        raise ValueError("Argument 'x' must be an xarray.DataArray.")
    if isinstance(y, xr.Dataset):
        raise ValueError("Argument 'y' must be an xarray.DataArray.")

    # check data resolution if they don't match, resample

    dims_x = list(x.dims)
    dims_y = list(y.dims)

    # if the dimensions are not the same, raise an error
    if dims_x != dims_y:
        # get which dimensions are different from each other
        diff_dims = list(set(dims_x) ^ set(dims_y))

        raise ValueError(
            f"Dimensions of x and y do not match. {diff_dims} are not found in both datasets."
        )

    if x.shape != y.shape:
        raise ValueError(
            f"Shape of x with shape {x.shape} and y with shape {y.shape} do not match !"
        )

    # if not x[along].equals(y[along]):
    if not np.array_equal(x[dim].values, y[dim].values, equal_nan=True):
        raise ValueError(f"{dim} dimension in x and y do not match !")

    dask_gufunc_kwargs = (
        {"output_sizes": {"stats": 2}} if x.chunks or y.chunks else None
    )

    if x.chunks:
        x = x.chunk({dim: -1})
    if y.chunks:
        y = y.chunk({dim: -1})

    x = x.sortby(dim).squeeze(drop=True)
    y = y.sortby(dim).squeeze(drop=True)

    result = xr.apply_ufunc(
        _corr_test,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["stats"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        kwargs={
            "data_type": "xr",
            "corr_type": corr_type,
            "alternative": alternative,
        },
    )

    corrs = xr.Dataset()
    for i, name in enumerate(out_vars):
        stat = result.isel(stats=i).to_dataset(name=name)
        corrs[name] = stat[name]

    return corrs.compute(scheduler=dask_scheduler)


def trends(
    data: xr.DataArray,
    dim: str = None,
    *,
    scale: float = 1,
    dask_scheduler: Literal["threads", "processes"] = "threads",
    polyfit: bool = False,
) -> xr.Dataset:
    """
    Calculate the Mann-Kendall trend test for a given dataset.
    Parameters:
        data ( xr.DataArray): Input dataset.
        dim (str): Dimension along which to calculate the trend test (Required for xarray).
        scale (float, optional): Scaling factor for the slope (e.g., convert to per hour, per day). Default is 1.
        dask_scheduler (str, optional): Dask scheduler type. Default is "processes".
        polyfit (bool, optional): Whether to use polynomial fitting. Default is False.
    Returns:
         xr.Dataset: DataFrame or Dataset containing the trend test results.
    """

    if polyfit:
        return _polyfit(data, dim=dim, scale=scale)

    out_vars = [
        "slope",
        "p_value",
        "trend",
        "mean_val",
        "std_val",
        "tau",
        "z_score",
    ]

    if isinstance(data, xr.Dataset):
        raise ValueError("Argument 'data' must be an xarray.DataArray.")

    if not dim:
        raise ValueError("Argument 'dim' is required for xarray input (e.g., 'time').")
    dask_gufunc_kwargs = None
    if data.chunks:
        data = data.chunk({dim: -1})
        dask_gufunc_kwargs = {"output_sizes": {"stats": 7}}

    data = data.squeeze(drop=True)

    result = xr.apply_ufunc(
        _mktrend_test,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[["stats"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        kwargs={"scale": scale},
    )

    trends = xr.Dataset()
    for i, name in enumerate(out_vars):
        stat = result.isel(stats=i).to_dataset(name=name)
        trends[name] = stat[name]

    return trends.compute(scheduler=dask_scheduler)


def pvalues(
    a: Union[xr.DataArray, xr.Dataset],
    b: Union[xr.DataArray, xr.Dataset],
    dim: str = "time",
    *,
    data_var: str = None,
) -> xr.Dataset:
    """
    Calculate the significance of the difference between two datasets.
    Parameters:
        a (xr.DataArray | xr.Dataset): First dataset.
        b (xr.DataArray | xr.Dataset): Second dataset.
        dim (str): Dimension along which to calculate the significance test, e.g., "time" or a time dimension, if 'a' and 'b' represent two periods, check the temporal dimension.
        data_var (str): Variable to calculate the significance for.
        level (float): Significance level for the test, default is 0.05.


    Returns:
        xr.Dataset: Dataset containing the significance test results.
    """

    res = xr.Dataset()

    dims_a = list(a.dims)
    dims_b = list(b.dims)

    # if the dimensions are not the same, raise an error
    if dims_a != dims_b:
        # get which dimensions are different from each other
        diff_dims = list(set(dims_a) ^ set(dims_b))

        raise ValueError(
            f"Dimensions of a and b do not match. {diff_dims} are not found in both datasets."
        )

    if isinstance(a, xr.Dataset) or isinstance(b, xr.Dataset):
        if data_var is None:
            raise ValueError("Argument 'data_var' is required for xr.Dataset input.")

        a = a[data_var] if isinstance(a, xr.Dataset) else a
        b = b[data_var] if isinstance(b, xr.Dataset) else b

    if a.sizes[dim] < 2 or b.sizes[dim] < 2:
        raise ValueError(
            f"At least two samples required along '{dim}' for t-test. Got {a.sizes[dim]} and {b.sizes[dim]}."
        )

    if dim not in a.dims or dim not in b.dims:
        raise ValueError(f"Dimension '{dim}' not found in input datasets.")

    a = a.transpose(dim, ...)
    b = b.transpose(dim, ...)

    t_stat, p_values = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit")

    a = a.mean(dim=dim).squeeze(drop=True)
    b = b.mean(dim=dim).squeeze(drop=True)

    p_values = xr.DataArray(data=p_values, coords=a.coords, dims=b.dims)
    t_stats = xr.DataArray(data=t_stat, coords=a.coords, dims=b.dims)

    res["p_values"] = p_values

    res["p_values"].attrs = {
        "long_name": "p_value",
        "description": "p-value of the significance test",
    }

    res["t_stats"] = t_stats
    res["t_stats"].attrs = {
        "long_name": "t_stat",
        "description": "t-statistic of the significance test",
    }

    return res
