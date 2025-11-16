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


def correlate(
    x: xr.DataArray,
    y: xr.DataArray,
    *,
    corr_type: Literal["pearson", "spearman", "kendall"] = "pearson",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    along: str = None,
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
    along : str
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
    if not np.array_equal(x[along].values, y[along].values, equal_nan=True):
        raise ValueError(f"{along} dimension in x and y do not match !")

    dask_gufunc_kwargs = (
        {"output_sizes": {"stats": 2}} if x.chunks or y.chunks else None
    )

    if x.chunks:
        x = x.chunk({along: -1})
    if y.chunks:
        y = y.chunk({along: -1})

    x = x.sortby(along).squeeze(drop=True)
    y = y.sortby(along).squeeze(drop=True)

    result = xr.apply_ufunc(
        _corr_test,
        x,
        y,
        input_core_dims=[[along], [along]],
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


def calc_trends(
    data: xr.DataArray,
    along: str = None,
    *,
    scale: float = 1,
    dask_scheduler: Literal["threads", "processes"] = "threads",
) -> xr.Dataset:
    """
    Calculate the Mann-Kendall trend test for a given dataset.
    Parameters:
        data ( xr.DataArray): Input dataset.
        along (str): Dimension along which to calculate the trend test (Required for xarray).
        scale (float, optional): Scaling factor for the slope (e.g., convert to per hour, per day). Default is 1.
        dask_scheduler (str, optional): Dask scheduler type. Default is "processes".

    Returns:
         xr.Dataset: DataFrame or Dataset containing the trend test results.
    """

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

    if not along:
        raise ValueError(
            "Argument 'along' is required for xarray input (e.g., 'time')."
        )
    dask_gufunc_kwargs = None
    if data.chunks:
        data = data.chunk({along: -1})
        dask_gufunc_kwargs = {"output_sizes": {"stats": 7}}

    data = data.squeeze(drop=True)

    result = xr.apply_ufunc(
        _mktrend_test,
        data,
        input_core_dims=[[along]],
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


def period_difference(
    da: xr.DataArray,
    period1: tuple[str, str],
    period2: tuple[str, str],
    *,
    along: str = "time",
    level: float = 0.05,
) -> xr.Dataset:
    """
    Compute the difference between two time periods in a DataArray or Dataset.

    Parameters
    ----------
    da : xr.DataArray
        Input data containing the time dimension.
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
    name = da.name if da.name else "change"
    p1 = da.sel({along: slice(period1[0], period1[1])})
    p2 = da.sel({along: slice(period2[0], period2[1])})

    sig = calc_significance(p1, p2, along=along, data_var=None, level=level)
    p_values = sig["p_values"]

    p1 = p1.mean(dim="time").squeeze(drop=True)
    p2 = p2.mean(dim="time").squeeze(drop=True)

    change = p2 - p1

    ds = xr.Dataset()
    ds["pvalues"] = p_values
    ds[name] = change
    ds[name].attrs = {
        "long_name": f"Mean difference between {period2[0]}-{period2[1]} and {period1[0]}-{period1[1]}",
        "description": f"Difference calculated as mean({period2[0]}-{period2[1]}) - mean({period1[0]}-{period1[1]})",
    }

    return ds


def calc_significance(
    a: Union[xr.DataArray, xr.Dataset],
    b: Union[xr.DataArray, xr.Dataset],
    along: str,
    *,
    data_var: str = None,
    level: float = 0.05,
) -> xr.Dataset:
    """
    Calculate the significance of the difference between two datasets.
    Parameters:
        a (xr.DataArray | xr.Dataset): First dataset.
        b (xr.DataArray | xr.Dataset): Second dataset.
        along (str): Dimension along which to calculate the significance test, e.g., "time" or a time dimension, if 'a' and 'b' represent two periods, check the temporal dimension.
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

    if a.sizes[along] < 2 or b.sizes[along] < 2:
        raise ValueError(
            f"At least two samples required along '{along}' for t-test. Got {a.sizes[along]} and {b.sizes[along]}."
        )

    if along not in a.dims or along not in b.dims:
        raise ValueError(f"Dimension '{along}' not found in input datasets.")

    a = a.transpose(along, ...)
    b = b.transpose(along, ...)

    t_stat, p_values = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit")

    a = a.mean(dim=along).squeeze(drop=True)
    b = b.mean(dim=along).squeeze(drop=True)

    p_values = xr.DataArray(
        data=np.where(p_values < level, 1, np.nan), coords=a.coords, dims=b.dims
    )

    t_stats = xr.DataArray(data=t_stat, coords=a.coords, dims=b.dims)

    res["p_values"] = p_values

    res["p_values"].attrs = {
        "long_name": "p_value",
        "description": f"Indicates if the difference is significant at the {level} level (1 = significant)",
    }

    res["t_stats"] = t_stats
    res["t_stats"].attrs = {
        "long_name": "t_stat",
        "description": "t-statistic of the significance test",
    }

    return res


def polyfit(data: xr.DataArray | xr.Dataset, along: str, data_var=None, scale=1):
    """
    Calculate the linear trend for the given xarray Dataset or DataArray using xr.polyfit.

    - data: xr.Dataset
    - data_var: The variable to calculate the trend test for.
    - along: dim to calculate the trend test along. also used for sorting the data.
    - scale: The scale to multiply the slope by i.e convert to per hour, per day, etc.

    Returns: xr.Dataset
    """

    data.attrs = {}
    data = data.sortby(along)
    data[along] = (np.arange(1, len(data[along]) + 1)).astype(np.int32)
    n = data.sizes[along]

    if isinstance(data, xr.Dataset):
        if data_var is None:
            raise ValueError("Argument 'data_var' is required for xr.Dataset input.")
        data = data[data_var]

    res = data.polyfit(dim=along, deg=1, cov=True)
    slope = res["polyfit_coefficients"].sel(degree=1)
    slope_variance = res["polyfit_covariance"].sel(cov_i=0, cov_j=0)
    stderr = slope_variance**0.5
    t_stat = slope / stderr

    p_values = xr.DataArray(
        2 * (1 - stats.t.cdf(np.abs(t_stat), (n - 2))),
        coords=slope.coords,
        dims=slope.dims,
    )

    mean_val = data.mean(dim=along)
    std_val = data.std(dim=along)

    trends = xr.Dataset()
    trends["slope"] = slope * scale
    trends["p_value"] = p_values
    trends["mean_val"] = mean_val
    trends["std_val"] = std_val

    # add attributes
    trends["slope"].attrs = {
        "long_name": "slope",
        "description": f"Slope of the linear trend per {scale} units of {along}",
    }
    trends["p_value"].attrs = {
        "long_name": "p_value",
        "description": "p-value of the trend significance test",
    }
    trends["mean_val"].attrs = {
        "long_name": "mean_val",
        "description": f"Mean value along {along}",
    }
    trends["std_val"].attrs = {
        "long_name": "std_val",
        "description": f"Standard deviation along {along}",
    }

    return trends
