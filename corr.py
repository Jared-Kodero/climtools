from gc import collect
from typing import Literal, Optional, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kendalltau, pearsonr, spearmanr

from .tools import log


def corr_test(
    array_x: np.ndarray,
    array_y: np.ndarray,
    corr_type: str,
    alternative: str = "two-sided",
    data_type: str = None,
    debug: bool = False,
    **coords,
) -> np.ndarray:
    try:

        nan_list = [np.nan] * 2

        if data_type == "pd":

            nan_data = np.array(list(coords.values()) + nan_list)

        else:
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
        collect()

        stats = [
            corr,
            p_value,
        ]

        if data_type == "pd":
            array = np.array(list(coords.values()) + stats)

        else:
            array = np.array(stats)

        return array
    except Exception:
        if debug:
            log()
        return nan_data


def _xr_dispacher(
    x,
    y,
    x_var,
    y_var,
    corr_type,
    alternative,
    along,
    out_vars,
    use_dask,
    dask_scheduler,
    debug,
):
    if not along:
        raise ValueError(
            "Argument 'along' is required for xarray input (e.g., 'time')."
        )
    if isinstance(x, xr.Dataset):
        if x_var is None:
            raise ValueError(
                "Argument 'x_var' must be provided when x is an xarray.Dataset."
            )
        x = x[x_var]

    if isinstance(y, xr.Dataset):
        if y_var is None:
            raise ValueError(
                "Argument 'y_var' must be provided when y is an xarray.Dataset."
            )
        y = y[y_var]

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

    x = x.sortby(along)
    y = y.sortby(along)

    result = xr.apply_ufunc(
        corr_test,
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
            "debug": debug,
        },
    )

    corrs = xr.Dataset()
    for i, name in enumerate(out_vars):
        stat = result.isel(stats=i).to_dataset(name=name)
        corrs[name] = stat[name]

    return corrs.compute(scheduler=dask_scheduler)


def _pd_dispatcher(
    x,
    y,
    x_var,
    y_var,
    corr_type,
    alternative,
    along,
    groupby,
    out_vars,
    use_dask,
    dask_scheduler,
    debug,
):

    if not x_var or not y_var:
        raise ValueError(
            "Arguments 'x_var' and 'y_var' must be provided for pd.DataFrame input."
        )

    if groupby:

        if isinstance(groupby, str):
            groupby = [groupby]
        if isinstance(groupby, tuple):
            groupby = list(groupby)

        data = pd.merge(x, y, on=groupby, how="left")

        data = data.dropna()

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

            array_x = group[x_var].values
            array_y = group[y_var].values

            if use_dask:

                task = dask.delayed(corr_test)(
                    array_x,
                    array_y,
                    corr_type,
                    alternative,
                    "pd",
                    debug,
                    **coords,
                )
                tasks.append(task)

            else:
                res = corr_test(
                    array_x,
                    array_y,
                    corr_type,
                    alternative,
                    "pd",
                    debug,
                    **coords,
                )
                dfs.append(pd.DataFrame(res.reshape(1, -1), columns=names))

        if use_dask:
            results = dask.compute(*tasks, scheduler=dask_scheduler)
            dfs = [pd.DataFrame(res.reshape(1, -1), columns=names) for res in results]

        corrs = pd.concat(dfs).reset_index(drop=True)

        return corrs

    elif groupby is None:

        data = pd.merge(x, y, on=groupby, how="left")
        data = data.dropna()

        if x_var is not None and y_var is not None:
            array_x = data[x_var].values
            array_y = data[y_var].values
            res = corr_test(
                array_x,
                array_y,
                corr_type,
                alternative=alternative,
                debug=debug,
                data_type="np",
            )
            corrs = pd.DataFrame(res.reshape(1, -1), columns=out_vars)

        return corrs


def calc_corr(
    x: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    y: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    *,
    x_var: str = None,
    y_var: str = None,
    corr_type: Literal["pearson", "spearman", "kendall"] = "pearson",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    along: str = None,
    groupby: Optional[Union[str, list[str]]] = None,
    use_dask: bool = True,
    dask_scheduler: Literal["threads", "processes"] = "threads",
    debug: bool = False,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    Calculate the correlation between two datasets.

    Parameters
    ----------

        x (pd.DataFrame | xr.DataArray | xr.Dataset): First dataset (independent variable).
        y (pd.DataFrame | xr.DataArray | xr.Dataset): Second dataset (dependent variable).
        x_var (str): Name of the variable in x.
        y_var (str): Name of the variable in y.
        corr_type (Literal["pearson", "spearman", "kendall"], optional): Type of correlation. Default is "pearson".
        alternative (Literal["two-sided", "less", "greater"], optional): Alternative hypothesis. Default is "two-sided".
        along (str, optional): Dimension for xarray input. Required for xarray.
        groupby (str | list[str], optional): Dimensions to group by for DataFrame.
        use_dask (bool, optional): Use Dask for parallelization. Default is True.
        dask_scheduler (str, optional): Dask scheduler type. Default is "processes".
        debug (bool, optional): Print debug information. Default is False.

    Returns
    --------

        pd.DataFrame or xr.Dataset: DataFrame or Dataset with correlation results.

    """

    out_vars = [
        "corr",
        "p_value",
    ]

    if isinstance(x, (xr.Dataset, xr.DataArray)) and isinstance(
        y, (xr.Dataset, xr.DataArray)
    ):
        return _xr_dispacher(
            x,
            y,
            x_var,
            y_var,
            corr_type,
            alternative,
            along,
            out_vars,
            use_dask,
            dask_scheduler,
            debug,
        )
    elif isinstance(x, (pd.DataFrame)) and isinstance(y, (pd.DataFrame)):
        return _pd_dispatcher(
            x,
            y,
            x_var,
            y_var,
            corr_type,
            alternative,
            along,
            groupby,
            out_vars,
            use_dask,
            dask_scheduler,
            debug,
        )
    else:
        raise TypeError(
            f"Unsupported data type: x: {type(x)}, y: {type(y)}. Expected pd.DataFrame, xr.Dataset, or xr.DataArray."
        )
