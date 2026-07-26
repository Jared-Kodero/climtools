"""
This module provides functions for calculating trends and correlations in xarray DataArrays and Datasets.
It includes implementations of the Mann-Kendall trend test, linear regression using polynomial fitting, and correlation tests (Pearson, Spearman, Kendall).
The functions are designed to handle missing data and can be applied along specified dimensions. Dask is supported for parallelized computations on large datasets.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

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
    nan_array = np.array([np.nan] * 7)

    df = pd.DataFrame({"array": array})
    df = df.dropna()

    try:
        if len(df) < 2:
            return nan_array

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

        return np.array(stats)
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
    if isinstance(data, xr.Dataset):
        if data_var is None:
            raise ValueError("Argument 'data_var' is required for xr.Dataset input.")
        data = data[data_var]

    data.attrs = {}
    data = data.sortby(dim)
    data = data.assign_coords(
        {dim: (np.arange(1, len(data[dim]) + 1)).astype(np.int32)}
    )
    n = data.sizes[dim]

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
        corr, p_value = stats.pearsonr(df["x"], df["y"], alternative=alternative)

    elif corr_type == "spearman":
        corr, p_value = stats.spearmanr(df["x"], df["y"], alternative=alternative)

    elif corr_type == "kendall":
        corr, p_value = stats.kendalltau(df["x"], df["y"], alternative=alternative)

    del array_x, array_y, df

    array = np.array([corr, p_value])

    return array


def corr(
    x: xr.DataArray,
    y: xr.DataArray,
    *,
    corr_type: Literal["pearson", "spearman", "kendall"] = "pearson",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    dim: str | None = None,
    dask_scheduler: Literal["threads", "processes"] = "threads",
) -> xr.Dataset:
    """
    Compute a pointwise correlation between two DataArrays along a dimension.

    The correlation is evaluated independently at every grid point across the
    remaining dimensions, using the Pearson, Spearman or Kendall coefficient.
    Missing values are dropped pairwise before each test.

    Parameters
    ----------
    x : xr.DataArray
        First input.
    y : xr.DataArray
        Second input. Must match ``x`` in dimensions, shape and the values of
        ``dim``.
    corr_type : {"pearson", "spearman", "kendall"}, default "pearson"
        Correlation coefficient:
        - "pearson": linear correlation
        - "spearman": rank correlation
        - "kendall": Kendall tau rank correlation
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis used for the p-value.
    dim : str
        Dimension the correlation is computed along, for example "time".
    dask_scheduler : {"threads", "processes"}, default "threads"
        Scheduler used when the inputs are chunked.

    Returns
    -------
    xr.Dataset
        Dataset with:
        - "corr": the correlation coefficient
        - "p_value": the p-value of the chosen alternative hypothesis
    """

    out_vars = [
        "corr",
        "p_value",
    ]

    if isinstance(x, xr.Dataset):
        raise TypeError("Argument 'x' must be an xarray.DataArray.")
    if isinstance(y, xr.Dataset):
        raise TypeError("Argument 'y' must be an xarray.DataArray.")

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
    dim: str | None = None,
    *,
    scale: float = 1,
    dask_scheduler: Literal["threads", "processes"] = "threads",
    polyfit: bool = False,
) -> xr.Dataset:
    """
    Compute a pointwise trend along a dimension.

    By default the modified Mann-Kendall test (Hamed and Rao) is applied at
    every grid point, returning the Sen slope together with the trend
    direction, its p-value and summary statistics. With ``polyfit=True`` an
    ordinary least squares fit is used instead, returning the slope and its
    p-value.

    Parameters
    ----------
    data : xr.DataArray
        Input field.
    dim : str
        Dimension the trend is computed along, for example "time". The data is
        sorted along this dimension first.
    scale : float, default 1
        Multiplier applied to the slope, to convert its time unit, for example
        to a per-decade rate.
    dask_scheduler : {"threads", "processes"}, default "threads"
        Scheduler used when the input is chunked.
    polyfit : bool, default False
        Use ordinary least squares instead of the Mann-Kendall test.

    Returns
    -------
    xr.Dataset
        Trend statistics. The Mann-Kendall path returns ``slope``, ``p_value``,
        ``trend``, ``mean_val``, ``std_val``, ``tau`` and ``z_score``. The
        ``polyfit`` path returns ``slope``, ``p_value``, ``mean_val`` and
        ``std_val``.
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
        raise TypeError("Argument 'data' must be an xarray.DataArray.")

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
    a: xr.DataArray | xr.Dataset,
    b: xr.DataArray | xr.Dataset,
    dim: str = "time",
    *,
    data_var: str | None = None,
) -> xr.DataArray:
    """
    Test the difference in mean between two datasets with a Welch t-test.

    The two-sample t-test is applied independently at every grid point across
    the dimensions other than ``dim``, with unequal variances assumed and NaNs
    omitted.

    Parameters
    ----------
    a, b : xr.DataArray or xr.Dataset
        The two samples, for example two periods. They must share dimensions.
    dim : str, default "time"
        Sample dimension the test is applied along.
    data_var : str, optional
        Variable to test. Required when ``a`` or ``b`` is a Dataset.

    Returns
    -------
    xr.DataArray
        Pointwise p-values of the difference in mean.
    """

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

    _, p_value = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit")

    a = a.mean(dim=dim).squeeze(drop=True)
    b = b.mean(dim=dim).squeeze(drop=True)

    p_value = xr.DataArray(data=p_value, coords=a.coords, dims=b.dims)

    p_value.attrs = {
        "long_name": "p_value",
        "description": "p-value of the significance test",
    }

    return p_value
