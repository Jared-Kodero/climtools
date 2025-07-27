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
    data_type: str = "pd",
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


def calc_corr(
    x: Union[np.ndarray, pd.DataFrame, xr.DataArray, xr.Dataset],
    y: Union[np.ndarray, pd.DataFrame, xr.DataArray, xr.Dataset],
    *,
    var_x: str = None,
    var_y: str = None,
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

        x (np.ndarray | pd.DataFrame | xr.DataArray | xr.Dataset): First dataset (independent variable).
        y (np.ndarray | pd.DataFrame | xr.DataArray | xr.Dataset): Second dataset (dependent variable).
        var_x (str): Name of the variable in x.
        var_y (str): Name of the variable in y.
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

    Examples:
        >>> x = np.random.rand(100, 2)
        >>> y = np.random.rand(100, 2)
        >>> result = calc_corr(x=x, y=y, var_x="var1", var_y="var2", corr_type="pearson")
        >>> result = calc_corr(x=x, y=y, var_x="var1", var_y="var2", corr_type="pearson", lag={"x": 1, "freq": "D"}) # other freqs: "S", "H",  "M", "Y"
    """
    try:

        out_vars = [
            "corr",
            "p_value",
        ]

        if isinstance(x, (xr.Dataset, xr.DataArray)) and isinstance(
            y, (xr.Dataset, xr.DataArray)
        ):

            if along is None:
                raise ValueError("Argument 'along' is required for xarray input.")

            if not isinstance(along, str):
                raise ValueError(f"'along' must be of type str, not {type(along)}.")

            if isinstance(x, xr.Dataset):
                if var_x is None:
                    raise ValueError(
                        "Arguments 'var_x' must be provided for xarray input."
                    )
                x = x[var_x]
            if isinstance(y, xr.Dataset):
                if var_y is None:
                    raise ValueError(
                        "Arguments 'var_y' must be provided for xarray input."
                    )

                y = y[var_y]

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

            return corrs.compute()

        if (
            groupby is not None
            and var_x is not None
            and var_y is not None
            and isinstance(x, pd.DataFrame)
            and isinstance(y, pd.DataFrame)
        ):

            if not (
                isinstance(var_x, str)
                and isinstance(var_y, str)
                and isinstance(along, str)
            ):
                raise ValueError(
                    "Invalid input: all of var_x, var_y, and along must be strings."
                )

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
            for key, group in data.groupby(groupby):

                if along is not None:
                    group = group.sort_values(by=along).reset_index(drop=True)

                if not isinstance(key, tuple):
                    key = (key,)
                coords = dict(zip(groupby, key))

                array_x = group[var_x].values
                array_y = group[var_y].values

                if use_dask:

                    task = dask.delayed(corr_test)(
                        array_x,
                        array_y,
                        corr_type,
                        alternative=alternative,
                        debug=debug,
                        **coords,
                    )
                    tasks.append(task)

                else:
                    res = corr_test(
                        array_x,
                        array_y,
                        corr_type,
                        alternative=alternative,
                        debug=debug,
                        **coords,
                    )
                    dfs.append(pd.DataFrame(res.reshape(1, -1), columns=names))

            if tasks:
                results = dask.compute(*tasks, scheduler=dask_scheduler)
                dfs = [
                    pd.DataFrame(res.reshape(1, -1), columns=names) for res in results
                ]

            corrs = pd.concat(dfs).reset_index(drop=True)

            return corrs

        elif (
            isinstance(x, (pd.DataFrame, np.ndarray))
            and isinstance(y, (pd.DataFrame, np.ndarray))
            or groupby is None
        ):

            data = pd.merge(x, y, on=groupby, how="left")
            data = data.dropna()

            if (
                isinstance(x, pd.DataFrame)
                and isinstance(y, pd.DataFrame)
                and var_x is not None
                and var_y is not None
            ):
                array_x = data[var_x].values
                array_y = data[var_y].values
                res = corr_test(
                    array_x,
                    array_y,
                    corr_type,
                    alternative=alternative,
                    debug=debug,
                    data_type="np",
                )
                corrs = pd.DataFrame(res.reshape(1, -1), columns=out_vars)

            elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                res = corr_test(
                    x,
                    y,
                    corr_type,
                    alternative=alternative,
                    debug=debug,
                    data_type="np",
                )
                corrs = pd.DataFrame(res.reshape(1, -1), columns=out_vars)

            return corrs
    except Exception:
        log()
