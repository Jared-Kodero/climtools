from typing import Literal, Union

import dask
import numpy as np
import pandas as pd
import pymannkendall as mk
import xarray as xr
from scipy import stats

from .logs import log


def mk_trend_test(
    array: np.ndarray,
    scale: float = 1,
    data_type: str = None,
    **coords,
) -> np.ndarray:

    nan_list = [np.nan] * 7

    if data_type == "pd":

        nan_data = np.array(list(coords.values()) + nan_list)

    else:
        nan_data = np.array(nan_list)

    df = pd.DataFrame({"array": array})
    df = df.dropna()
    if df.empty or len(df) < 2:
        log("Not enough data to calculate trend ! Returning NaN values.")
        return nan_data

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

    if data_type == "pd":

        array = np.array(list(coords.values()) + stats)

    else:
        array = np.array(stats)

    return array


def _dataset_dispacher(
    data,
    along,
    scale,
    use_dask,
    dask_scheduler,
) -> xr.Dataset:

    trends = []

    for data_var in data.data_vars:
        trends_ds = calc_trends(
            data[data_var],
            along=along,
            scale=scale,
            use_dask=use_dask,
            dask_scheduler=dask_scheduler,
        )
        # rename the all the trends_ds data_vars to include the data_var name
        trends_ds = trends_ds.rename(
            {var: f"{data_var}_{var}" for var in trends_ds.data_vars}
        )
        trends.append(trends_ds)

    return xr.merge(
        trends
    ).compute()  # compute to ensure all data is loaded and processed


def _xr_dispacher(
    data,
    along,
    scale,
    use_dask,
    dask_scheduler,
    out_vars,
):
    if not along:
        raise ValueError(
            "Argument 'along' is required for xarray input (e.g., 'time')."
        )
    if isinstance(data, xr.Dataset):

        if len(list(data.data_vars)) == 1:
            data = data[list(data.data_vars)[0]]
        else:

            return _dataset_dispacher(
                data,
                along,
                scale,
                use_dask,
                dask_scheduler,
            )
    dask_gufunc_kwargs = None
    if data.chunks:
        data = data.chunk({along: -1})
        dask_gufunc_kwargs = {"output_sizes": {"stats": 7}}

    data = data.squeeze(drop=True)

    result = xr.apply_ufunc(
        mk_trend_test,
        data,
        input_core_dims=[[along]],
        output_core_dims=[["stats"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        kwargs={"scale": scale, "data_type": "xr"},
    )

    trends = xr.Dataset()
    for i, name in enumerate(out_vars):
        stat = result.isel(stats=i).to_dataset(name=name)
        trends[name] = stat[name]

    return trends.compute(scheduler=dask_scheduler)


def _pd_dispatcher(
    data,
    along,
    data_var,
    groupby,
    scale,
    use_dask,
    dask_scheduler,
    out_vars,
):
    if not data_var:
        raise ValueError("Argument 'data_var' is required for pd.DataFrame input.")

    if groupby:

        if isinstance(groupby, str):
            groupby = [groupby]
        elif isinstance(groupby, tuple):
            groupby = list(groupby)

        if along in groupby:
            groupby.remove(along)

        names = groupby + out_vars

        dfs = []
        tasks = []
        grouped_data = data.groupby(groupby)
        for key, group in grouped_data:

            if along is not None:
                group = group.sort_values(by=along).reset_index(drop=True)

            if not isinstance(key, tuple):
                key = (key,)
            coords = dict(zip(groupby, key))

            array = group[data_var].values

            if use_dask:

                task = dask.delayed(mk_trend_test)(array, scale, "pd", **coords)
                tasks.append(task)

            else:
                res = mk_trend_test(array, scale, "pd", **coords)
                dfs.append(pd.DataFrame(res.reshape(1, -1), columns=names))

        if use_dask:
            results = dask.compute(*tasks, scheduler=dask_scheduler)
            dfs = [pd.DataFrame(res.reshape(1, -1), columns=names) for res in results]

        trends = pd.concat(dfs).reset_index(drop=True)

        return trends

    elif not groupby and data_var:

        if data_var is not None:
            array = data[data_var].values
            res = mk_trend_test(array, scale, data_type="np")
            trends = pd.DataFrame(res.reshape(1, -1), columns=out_vars)
        return trends


def calc_trends(
    data: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    along: str = None,
    *,
    data_var: str = None,
    groupby: Union[str, list[str]] = None,
    scale: float = 1,
    use_dask: bool = True,
    dask_scheduler: Literal["threads", "processes"] = "threads",
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculate the Mann-Kendall trend test for a given dataset.

    Parameters:
        data ( pd.DataFrame | xr.Dataset): Input dataset.
        along (str): Dimension along which to calculate the trend test (Required for xarray).
        data_var (str): Variable to calculate the trend test for (Required for xarray.Dataset).
        groupby (str | list[str], optional): Dimensions to group data by (Required for DataFrame).
        scale (float, optional): Scaling factor for the slope (e.g., convert to per hour, per day). Default is 1.
        use_dask (bool, optional): Whether to parallelize calculations using Dask. Default is True.
        dask_scheduler (str, optional): Dask scheduler type. Default is "processes".

    Returns:
        pd.DataFrame | xr.Dataset: DataFrame or Dataset containing the trend test results.
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

    if isinstance(data, (xr.Dataset, xr.DataArray)):
        return _xr_dispacher(
            data,
            along,
            scale,
            use_dask,
            dask_scheduler,
            out_vars,
        )
    elif isinstance(data, pd.DataFrame):
        return _pd_dispatcher(
            data,
            along,
            data_var,
            groupby,
            scale,
            use_dask,
            dask_scheduler,
            out_vars,
        )
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected pd.DataFrame, xr.Dataset, or xr.DataArray."
        )


def calc_signicance(
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


def xr_polyfit(data, data_var, along, scale=1):
    """
    Calculate the linear trend for the given xarray Dataset.

    - data: xr.Dataset
    - data_var: The variable to calculate the trend test for.
    - along: dim to calculate the trend test along. also used for sorting the data.
    - scale: The scale to multiply the slope by i.e convert to per hour, per day, etc.

    Returns: xr.Dataset
    """

    data.attrs = {}
    data = data.sortby(along)
    data[along] = (np.arange(1, len(data[along]) + 1)).astype(np.int32)
    n = data.dims[along]  #

    res = data[data_var].polyfit(dim=along, deg=1, cov=True)
    slope = res["polyfit_coefficients"].sel(degree=1)
    slope_variance = res["polyfit_covariance"].sel(cov_i=0, cov_j=0)
    stderr = slope_variance**0.5
    t_stat = slope / stderr

    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), (n - 2)))

    mean_val = data[data_var].mean(dim=along)
    std_val = data[data_var].std(dim=along)

    trends = xr.Dataset()
    trends["slope"] = slope * scale
    trends["p_value"] = p_values
    trends["mean_val"] = mean_val
    trends["std_val"] = std_val

    return trends
