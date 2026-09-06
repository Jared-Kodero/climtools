"""Compute distributed variance and standard deviation."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from types import EllipsisType
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from .common import partial_dtype
from .meta import mpp_get_meta
from .planning import (
    dataset_result,
    finish_local_reduction,
    guarded,
    local_reduction_meta,
    mpp_comm_reduce,
    mpp_count_valid_values,
    mpp_finish,
    mpp_finish_scatter,
    mpp_plan_scatter_target,
    mpp_reduction_plan,
    mpp_resolve_comm,
    mpp_scatter_replicated_slice,
    normalize_dim,
    repartition_candidates,
)
from .reductions import mpp_mean_reduce


def _var_or_std(
    mpi_context: MPIContext,
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
    old_meta = mpp_get_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        method = value.std if root else value.var
        local_result = method(
            dim=local_dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs
        )
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = mpp_reduction_plan(
        mpi_context, value, dims, old_meta, operation="std" if root else "var"
    )

    def combine(
        variable: xr.DataArray,
        variable_dims: tuple[Hashable, ...],
        mean: xr.DataArray,
        *,
        comm: MPI.Comm | None = None,
        replica_count: int = 1,
        scatter: tuple[Hashable, list[int]] | None = None,
    ) -> xr.DataArray:
        """Combine local squared deviations into global variance or standard deviation.

        ``scatter``, when given, is forwarded to both the squared-deviation
        sum and the valid-value count (see :func:`~.planning.mpp_scatter_target`),
        so this rank only ever holds its own post-reduction slice.
        """
        deviation = variable - mean
        # `deviation`'s dtype (always floating: subtracting a float
        # `mean` promotes even an integer `variable`) is what the
        # squared-deviation sum below actually produces -- using
        # `variable.dtype` here instead would predict an *integer*
        # expected dtype for an integer-typed variable, silently
        # truncating the genuinely-fractional sum of squares to an
        # integer before the Allreduce ever runs.
        partial_sq_sum, error = guarded(
            lambda: (deviation * deviation).sum(
                dim=variable_dims, skipna=skipna, min_count=None, keep_attrs=False
            )
        )
        global_sq_sum = mpp_comm_reduce(
            mpi_context,
            partial_sq_sum,
            MPI.SUM,
            expect_dtype=partial_dtype(deviation.dtype.str, "sum", skipna),
            error=error,
            phase="MPI xarray variance reduction",
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
        )
        denominator = (
            mpp_count_valid_values(
                mpi_context,
                variable,
                variable_dims,
                comm=comm,
                replica_count=replica_count,
                scatter=scatter,
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
        mean = mpp_mean_reduce(
            mpi_context,  # type: ignore[attr-defined]
            value,
            dim,
            skipna=skipna,
            keep_attrs=False,
            partition_dim=None,
        )
        scattered = mpp_plan_scatter_target(
            mpi_context, old_meta, dims, partition_dim, reduce_plan
        )
        comm = (
            scattered[2]
            if scattered is not None
            else mpp_resolve_comm(mpi_context, old_meta, reduce_plan[0].comm_axes)
        )
        result = combine(
            value,
            dims,
            mean,
            comm=comm,
            replica_count=reduce_plan[0].replica_count,
            scatter=None if scattered is None else scattered[:2],
        )
        if scattered is not None:
            return mpp_finish_scatter(
                result, target=scattered[0], counts=scattered[1], comm=comm
            )
        return mpp_finish(
            mpi_context,
            result,
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=repartition_candidates(reduce_plan),
        )

    mean_ds = mpp_mean_reduce(
        mpi_context,  # type: ignore[attr-defined]
        value,
        dim,
        skipna=skipna,
        keep_attrs=False,
        partition_dim=None,
    )
    variables: dict[Hashable, xr.DataArray] = {}
    scattered = mpp_plan_scatter_target(
        mpi_context, old_meta, dims, partition_dim, reduce_plan
    )
    scatter_start = scatter_stop = None
    if scattered is not None:
        _, scatter_counts, scatter_comm = scattered
        scatter_start = sum(scatter_counts[: scatter_comm.rank])
        scatter_stop = scatter_start + scatter_counts[scatter_comm.rank]
    for entry in reduce_plan:
        variable = value[entry.name]
        if not entry.dims:
            variables[entry.name] = (
                mpp_scatter_replicated_slice(
                    variable, scattered[0], scatter_start, scatter_stop
                )
                if scattered is not None
                else variable
            )
            continue
        if not entry.distributed:
            method = variable.std if root else variable.var
            variables[entry.name] = method(
                dim=entry.dims, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs
            )
            continue
        comm = (
            scattered[2]
            if scattered is not None
            else mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes)
        )
        variables[entry.name] = combine(
            variable,
            entry.dims,
            mean_ds[entry.name],
            comm=comm,
            replica_count=entry.replica_count,
            scatter=None if scattered is None else scattered[:2],
        )
    coord_source = (
        value.isel({scattered[0]: slice(scatter_start, scatter_stop)})
        if scattered is not None and scattered[0] in value.dims
        else value
    )
    dataset = dataset_result(coord_source, dims, variables)
    if scattered is not None:
        return mpp_finish_scatter(
            dataset, target=scattered[0], counts=scattered[1], comm=scattered[2]
        )
    return mpp_finish(
        mpi_context,
        dataset,
        old_meta=old_meta,
        partition_dim=partition_dim,
        auto_candidates=repartition_candidates(reduce_plan),
    )


def mpp_var(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    ddof : int, optional
        Delta degrees of freedom; the divisor is ``N - ddof``.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object -- see :func:`~.planning.finish` for the exact
        replication/no-duplication guarantee this carries.
    """
    return _var_or_std(
        mpi_context,
        value,
        dim,
        skipna=skipna,
        ddof=ddof,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        root=False,
    )


def mpp_std(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    ddof : int, optional
        Delta degrees of freedom; the divisor is ``N - ddof``.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object -- see :func:`~.planning.finish` for the exact
        replication/no-duplication guarantee this carries.
    """
    return _var_or_std(
        mpi_context,
        value,
        dim,
        skipna=skipna,
        ddof=ddof,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        root=True,
    )
