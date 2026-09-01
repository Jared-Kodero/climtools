"""Compute distributed variance and standard deviation.

The implementation uses one collective for the mean and one for squared
deviations.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from types import EllipsisType
from typing import Literal

import numpy as np
from mpi4py import MPI

import xarray as xr

from .common import partial_dtype
from .meta import get_mpi_meta
from .planning import (
    comm_reduce,
    count_valid_values,
    dataset_result,
    finish,
    finish_local_reduction,
    guarded,
    local_reduction_meta,
    normalize_dim,
    reduction_plan,
    repartition_candidates,
    resolve_comm,
)
from .reductions import mean_reduce


def _var_or_std(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
    *,
    skipna: bool | None,
    ddof: int,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
    root: bool,
) -> xr.Dataset | xr.DataArray:
    """Shared implementation for :meth:`var` and :meth:`std`."""
    local_dim, dims = normalize_dim(value, dim)
    old_meta = get_mpi_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        method = value.std if root else value.var
        local_result = method(
            dim=local_dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs
        )
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = reduction_plan(runtime, value, dims, old_meta, operation="std" if root else "var")

    def combine(
        variable: xr.DataArray,
        variable_dims: tuple[Hashable, ...],
        mean: xr.DataArray,
        *,
        comm: MPI.Comm | None = None,
        replica_count: int = 1,
    ) -> xr.DataArray:
        """Combine rank-local sums of squared deviation into variance
        (``std`` when ``root``), via one ``Allreduce``."""
        deviation = variable - mean
        partial_sq_sum, error = guarded(
            lambda: (deviation * deviation).sum(
                dim=variable_dims, skipna=skipna, min_count=None, keep_attrs=False
            )
        )
        global_sq_sum = comm_reduce(
            runtime,
            partial_sq_sum,
            MPI.SUM,
            expect_dtype=partial_dtype(variable.dtype.str, "sum", skipna),
            error=error,
            phase="MPI xarray variance reduction",
            comm=comm,
            replica_count=replica_count,
        )
        denominator = (
            count_valid_values(
                runtime, variable, variable_dims, comm=comm, replica_count=replica_count
            )
            - ddof
        )
        target = np.asarray(np.var(np.zeros(1, dtype=variable.dtype))).dtype
        divisor = (
            denominator.astype(target, keep_attrs=False)
            if target.kind in "fc"
            else denominator
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            result = global_sq_sum / divisor
        result = result.where(denominator > 0)
        if result.dtype != target:
            result = result.astype(target, keep_attrs=True)
        if root:
            result = np.sqrt(result)
        if keep_attrs:
            result.attrs.update(variable.attrs)
        return result

    if isinstance(value, xr.DataArray):
        if not dims:
            method = value.std if root else value.var
            return method(
                dim=local_dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs
            )
        mean = mean_reduce(
            runtime,  # type: ignore[attr-defined]
            value,
            dim,
            skipna=skipna,
            keep_attrs=False,
            partition_dim=None,
        )
        result = combine(
            value,
            dims,
            mean,
            comm=resolve_comm(runtime, old_meta, reduce_plan[0].comm_axes),
            replica_count=reduce_plan[0].replica_count,
        )
        return finish(
            runtime,
            result,
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=repartition_candidates(reduce_plan),
        )

    mean_ds = mean_reduce(
        runtime,  # type: ignore[attr-defined]
        value,
        dim,
        skipna=skipna,
        keep_attrs=False,
        partition_dim=None,
    )
    variables: dict[Hashable, xr.DataArray] = {}
    for entry in reduce_plan:
        variable = value[entry.name]
        if not entry.dims:
            variables[entry.name] = variable
            continue
        if not entry.distributed:
            method = variable.std if root else variable.var
            variables[entry.name] = method(
                dim=entry.dims, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs
            )
            continue
        variables[entry.name] = combine(
            variable,
            entry.dims,
            mean_ds[entry.name],
            comm=resolve_comm(runtime, old_meta, entry.comm_axes),
            replica_count=entry.replica_count,
        )
    return finish(
        runtime,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        partition_dim=partition_dim,
        auto_candidates=repartition_candidates(reduce_plan),
    )


def var(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    ddof: int = 0,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the variance of a distributed xarray object.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    ddof : int, optional
        Delta degrees of freedom; the divisor is ``N - ddof``. Default 0.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
        Default is ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object.

    Notes
    -----
    MPI communication occurs only when ``dim`` includes the active
    partition dimension, and costs two ``Allreduce`` calls (mean, then
    sum of squared deviation) rather than one."""
    return _var_or_std(
        runtime,
        value,
        dim,
        skipna=skipna,
        ddof=ddof,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        root=False,
    )


def std(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    ddof: int = 0,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the standard deviation of a distributed xarray object.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    ddof : int, optional
        Delta degrees of freedom; the divisor is ``N - ddof``. Default 0.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
        Default is ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object.

    Notes
    -----
    MPI communication occurs only when ``dim`` includes the active
    partition dimension, and costs two ``Allreduce`` calls (mean, then
    sum of squared deviation) rather than one."""
    return _var_or_std(
        runtime,
        value,
        dim,
        skipna=skipna,
        ddof=ddof,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        root=True,
    )
