"""Provide distributed numerical and logical reductions."""

from __future__ import annotations

import functools
from collections.abc import Hashable, Iterable
from types import EllipsisType
from typing import Any, Literal

import numpy as np
from mpi4py import MPI

import xarray as xr

from .common import extreme_identity, op_name, partial_dtype
from .meta import get_mpi_meta
from .planning import (
    comm_reduce,
    count_valid_values,
    dataset_result,
    exchange,
    finish,
    finish_local_reduction,
    guarded,
    local_reduction_meta,
    normalize_dim,
    reduction_plan,
    repartition_candidates,
    resolve_comm,
    skipna_enabled,
)


def _combine_sum_or_prod(
    runtime,
    value: xr.DataArray,
    partial: xr.DataArray,
    dims: tuple[Hashable, ...],
    op: MPI.Op,
    *,
    skipna: bool | None,
    min_count: int | None,
    error: BaseException | None = None,
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
) -> xr.DataArray:
    """Combine rank-local sum or product partials."""
    result = comm_reduce(
        runtime,
        partial,
        op,
        expect_dtype=partial_dtype(
            value.dtype.str, "prod" if op_name(op) == "PROD" else "sum", skipna
        ),
        error=error,
        phase="MPI xarray sum/prod reduction",
        comm=comm,
        replica_count=replica_count,
    )
    global_count = None
    if min_count is not None and skipna_enabled(value.dtype, skipna):
        global_count = count_valid_values(
            runtime, value, dims, comm=comm, replica_count=replica_count
        )
    if global_count is not None:
        # where() introduces NaN, which requires a floating result. Restore
        # the partial's own dtype so a float32 field stays float32.
        masked = result.where(global_count >= min_count)
        result = (
            masked
            if masked.dtype == result.dtype or result.dtype.kind not in "fc"
            else masked.astype(result.dtype, keep_attrs=True)
        )
    return result


def _combine_mean(
    runtime,
    value: xr.DataArray,
    partial_sum: xr.DataArray | None,
    dims: tuple[Hashable, ...],
    *,
    skipna: bool | None = None,
    error: BaseException | None = None,
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
) -> xr.DataArray:
    """Combine rank-local sums and counts into a global mean."""
    global_sum = comm_reduce(
        runtime,
        partial_sum,
        MPI.SUM,
        expect_dtype=partial_dtype(value.dtype.str, "sum", skipna),
        error=error,
        phase="MPI xarray mean reduction",
        comm=comm,
        replica_count=replica_count,
    )
    global_count = count_valid_values(
        runtime, value, dims, comm=comm, replica_count=replica_count
    )
    # Divide in the dtype xarray's own .mean() would produce for this
    # input. This is genuinely shape-dependent, not just dtype-dependent:
    # confirmed directly, a float32 array reduced over one dimension
    # while keeping others (an ordinary partial reduction) stays float32
    # in xarray's own .mean(), but the same array reduced over *every*
    # dimension to a scalar promotes to float64 -- an earlier version of
    # this line asked a synthetic size-1 array for its dtype, which
    # reliably reproduces the partial-reduction case (kept-dimension
    # size is what governs it, confirmed across several shapes) but not
    # the full-reduction one, where a genuinely size-1 sample take the
    # *other*, non-promoting branch a real, larger reduction does not
    # -- so a same-dtype full reduction of, e.g., a length-3 real array
    # silently disagreed with xarray by staying in the narrower dtype.
    # Rather than chase further shape-dependent thresholds, this uses
    # the two independently-verified, stable end cases directly:
    # non-floating dtypes always promote to float64 (confirmed for
    # int32); floating dtypes are dtype-preserving for a partial
    # reduction and promote to float64 for a full one *except*
    # complex, which never promotes either way (confirmed both ways
    # for complex64) -- xarray evidently special-cases complex
    # dtype preservation the same way regardless of reduction shape.
    kind = value.dtype.kind
    if kind not in "fc":
        target = np.dtype(np.float64)
    elif kind == "c":
        target = value.dtype
    else:
        is_full_reduction = set(dims) == set(value.dims)
        target = np.dtype(np.float64) if is_full_reduction else value.dtype
    divisor = (
        global_count.astype(target, keep_attrs=False)
        if target.kind in "fc"
        else global_count
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = global_sum / divisor
    result = result.where(global_count != 0)
    if result.dtype != target:
        result = result.astype(target, keep_attrs=True)
    return result


def _local_extreme(
    runtime,
    variable: xr.DataArray,
    variable_dims: tuple[Hashable, ...],
    *,
    empty: bool,
    minimum: bool,
    skipna: bool | None,
    keep_attrs: bool | None,
) -> xr.DataArray:
    """Compute a rank-local min/max partial."""
    if empty:
        identity = extreme_identity(variable.dtype, minimum=minimum)
        template = variable.sum(dim=variable_dims, skipna=False, keep_attrs=keep_attrs)
        return xr.full_like(template, identity, dtype=variable.dtype)
    method = variable.min if minimum else variable.max
    return method(dim=variable_dims, skipna=skipna, keep_attrs=keep_attrs)


def _combine_extreme(
    runtime,
    value: xr.DataArray,
    partial: xr.DataArray | None,
    dims: tuple[Hashable, ...],
    *,
    minimum: bool,
    skipna: bool | None,
    error: BaseException | None = None,
    comm: MPI.Comm | None = None,
) -> xr.DataArray:
    """Combine rank-local min/max partials across ranks.

    No ``replica_count`` parameter: unlike a sum, MIN/MAX/LAND/LOR are
    idempotent under duplication, so a rank redundantly holding the
    same value as another rank in ``comm`` (a variable replicated
    along one axis of a multi-dimensional partition) needs no
    correction here.
    """
    # Use the agreed variable dtype, not a rank-local partial dtype. Empty
    # partitions follow a different local path, and dtype-dependent branching
    # could desynchronize collectives. Min/max also require no promotion; using
    # the declared dtype avoids bottleneck's float32-to-float64 scalar promotion.
    operation = "min" if minimum else "max"
    expect_dtype = value.dtype
    kind = value.dtype.kind
    if kind == "b":
        return comm_reduce(
            runtime,
            partial,
            MPI.LAND if minimum else MPI.LOR,
            expect_dtype=expect_dtype,
            error=error,
            phase=f"MPI xarray {operation} reduction",
            comm=comm,
        )

    op = MPI.MIN if minimum else MPI.MAX
    if kind != "f":
        return comm_reduce(
            runtime,
            partial,
            op,
            expect_dtype=expect_dtype,
            error=error,
            phase=f"MPI xarray {operation} reduction",
            comm=comm,
        )

    # Floating reductions carry validity beside the extreme so empty or all-NaN
    # partitions can use an identity without confusing it with real infinity.
    # Encoding the flag in the same buffer avoids a second boolean collective.
    send: np.ndarray[Any, Any] | None = None
    template: xr.DataArray | None = None
    use_skipna = skipna_enabled(value.dtype, skipna)
    # ANY valid rank suffices under skipna; without it every rank must be
    # NaN-free for the result to be defined.
    flip = -1.0 if ((not minimum) != use_skipna) else 1.0

    if error is None:
        try:
            identity = extreme_identity(expect_dtype, minimum=minimum)
            if use_skipna:
                good = value.count(dim=dims, keep_attrs=False) > 0
            else:
                good = ~value.isnull().any(dim=dims, keep_attrs=False)
            safe_partial = partial.where(good, other=identity)
            if safe_partial.dtype != expect_dtype:
                safe_partial = safe_partial.astype(expect_dtype, keep_attrs=True)
            template = safe_partial

            values = np.ascontiguousarray(
                np.asarray(safe_partial.values, dtype=expect_dtype)
            )
            flags = np.where(
                np.asarray(good.values, dtype=bool),
                np.asarray(flip, dtype=expect_dtype),
                np.zeros((), dtype=expect_dtype),
            )
            send = np.empty((2, values.size), dtype=expect_dtype)
            send[0] = np.reshape(values, values.size)
            send[1] = np.reshape(flags, values.size)
        except BaseException as exc:
            error = exc
            send = None
            template = None

    signature = (
        None
        if send is None
        else (
            op_name(op),
            send.dtype.str,
            tuple(int(length) for length in send.shape),
        )
    )
    runtime.raise_if_error(
        error, f"MPI xarray {operation} reduction", signature, comm=comm
    )
    if send is None or template is None:
        raise AssertionError("MPI xarray reduction buffer is missing.")

    recv = exchange(runtime, send, op, comm=comm)

    shape = tuple(int(length) for length in template.shape)
    combined = np.asarray(recv[0]).reshape(shape)
    valid = (np.asarray(recv[1]).reshape(shape) * flip) > 0
    masked = np.where(valid, combined, np.asarray(np.nan, dtype=expect_dtype))
    return template.copy(data=np.asarray(masked, dtype=expect_dtype).reshape(shape))


# -- public reductions ---------------------------------------------------


def sum_reduce(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    min_count: int | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Sum a distributed xarray object over one or more dimensions.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    min_count : int or None, optional
        Minimum number of valid values required.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
        ``"auto"`` selects a surviving dimension; None leaves the result
        replicated. Default is ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object.

    Notes
    -----
    MPI communication occurs only when ``dim`` includes the active partition
    dimension."""
    return _sum_prod(
        runtime,
        value,
        dim,
        op=MPI.SUM,
        product=False,
        skipna=skipna,
        min_count=min_count,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def prod_reduce(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    min_count: int | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Multiply a distributed xarray object over one or more dimensions.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    min_count : int or None, optional
        Minimum number of valid values required.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
        ``"auto"`` selects a surviving dimension; None leaves the result
        replicated. Default is ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object.

    Notes
    -----
    MPI communication occurs only when ``dim`` includes the active partition
    dimension."""
    return _sum_prod(
        runtime,
        value,
        dim,
        op=MPI.PROD,
        product=True,
        skipna=skipna,
        min_count=min_count,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def _sum_prod(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
    *,
    op: MPI.Op,
    product: bool,
    skipna: bool | None,
    min_count: int | None,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
) -> xr.Dataset | xr.DataArray:
    """Implement distributed sum and product reductions."""
    operation = "prod" if product else "sum"
    local_dim, dims = normalize_dim(value, dim)
    old_meta = get_mpi_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        method = value.prod if product else value.sum
        local_result = method(
            dim=local_dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs
        )
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = reduction_plan(runtime, value, dims, old_meta, operation=operation)

    if isinstance(value, xr.DataArray):
        method = value.prod if product else value.sum
        local, local_error = guarded(
            lambda: method(
                dim=local_dim, skipna=skipna, min_count=None, keep_attrs=keep_attrs
            )
        )
        if not dims:
            if local_error is not None:
                raise local_error
            return local
        result = _combine_sum_or_prod(
            runtime,
            value,
            local,
            dims,
            op,
            skipna=skipna,
            min_count=min_count,
            error=local_error,
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

    variables: dict[Hashable, xr.DataArray] = {}
    for entry in reduce_plan:
        variable = value[entry.name]
        if not entry.dims:
            variables[entry.name] = variable
            continue
        method = variable.prod if product else variable.sum
        local, local_error = guarded(
            lambda method=method, entry=entry: method(
                dim=entry.dims, skipna=skipna, min_count=None, keep_attrs=keep_attrs
            )
        )
        if not entry.distributed:
            if local_error is not None:
                raise local_error
            variables[entry.name] = local
            continue
        result = _combine_sum_or_prod(
            runtime,
            variable,
            local,
            entry.dims,
            op,
            skipna=skipna,
            min_count=min_count,
            error=local_error,
            comm=resolve_comm(runtime, old_meta, entry.comm_axes),
            replica_count=entry.replica_count,
        )
        variables[entry.name] = result
    return finish(
        runtime,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        partition_dim=partition_dim,
        auto_candidates=repartition_candidates(reduce_plan),
    )


def mean_reduce(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the mean of a distributed xarray object.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
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
    MPI communication occurs only when ``dim`` includes the active partition
    dimension."""
    local_dim, dims = normalize_dim(value, dim)
    old_meta = get_mpi_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        local_result = value.mean(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = reduction_plan(runtime, value, dims, old_meta, operation="mean")

    if isinstance(value, xr.DataArray):
        if not dims:
            local_mean = value.mean(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
            return local_mean
        local_sum, local_error = guarded(
            lambda: value.sum(
                dim=local_dim, skipna=skipna, min_count=None, keep_attrs=keep_attrs
            )
        )
        result = _combine_mean(
            runtime,
            value,
            local_sum,
            dims,
            skipna=skipna,
            error=local_error,
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

    variables: dict[Hashable, xr.DataArray] = {}
    for entry in reduce_plan:
        variable = value[entry.name]
        if not entry.dims:
            variables[entry.name] = variable
            continue
        if not entry.distributed:
            variables[entry.name] = variable.mean(
                dim=entry.dims, skipna=skipna, keep_attrs=keep_attrs
            )
            continue
        local_sum, local_error = guarded(
            lambda variable=variable, entry=entry: variable.sum(
                dim=entry.dims, skipna=skipna, min_count=None, keep_attrs=keep_attrs
            )
        )
        result = _combine_mean(
            runtime,
            variable,
            local_sum,
            entry.dims,
            skipna=skipna,
            error=local_error,
            comm=resolve_comm(runtime, old_meta, entry.comm_axes),
            replica_count=entry.replica_count,
        )
        variables[entry.name] = result
    return finish(
        runtime,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        partition_dim=partition_dim,
        auto_candidates=repartition_candidates(reduce_plan),
    )


def min_reduce(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the minimum of a distributed xarray object.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
        Default is ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object."""
    return _min_max(
        runtime,
        value,
        dim,
        minimum=True,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def max_reduce(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Compute the maximum of a distributed xarray object.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
        Default is ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced object."""
    return _min_max(
        runtime,
        value,
        dim,
        minimum=False,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def _min_max(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
    *,
    minimum: bool,
    skipna: bool | None,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
) -> xr.Dataset | xr.DataArray:
    """Implement distributed minimum and maximum reductions."""
    operation = "min" if minimum else "max"
    local_dim, dims = normalize_dim(value, dim)
    old_meta = get_mpi_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        method = value.min if minimum else value.max
        local_result = method(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = reduction_plan(runtime, value, dims, old_meta, operation=operation)

    def locally_empty(variable: xr.DataArray) -> bool:
        """Return whether this rank's local slice of ``variable`` is
        empty along any partition dimension it owns -- generalizes the
        single-dimension ``old_meta["dim"] in value.dims and size==0``
        check to look at every one of ``meta["dims"]`` a given
        variable actually varies over, since under a multi-dimensional
        partition different variables can own different subsets of
        the partition dimensions.
        """
        if old_meta is None:
            return False
        return any(
            dim in variable.dims and int(variable.sizes[dim]) == 0
            for dim in old_meta["dims"]
        )

    if isinstance(value, xr.DataArray):
        if not dims:
            method = value.min if minimum else value.max
            return method(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
        local, local_error = guarded(
            lambda: _local_extreme(
                runtime,
                value,
                dims,
                empty=locally_empty(value),
                minimum=minimum,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
        )
        result = _combine_extreme(
            runtime,
            value,
            local,
            dims,
            minimum=minimum,
            skipna=skipna,
            error=local_error,
            comm=resolve_comm(runtime, old_meta, reduce_plan[0].comm_axes),
        )
        return finish(
            runtime,
            result,
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=repartition_candidates(reduce_plan),
        )

    variables: dict[Hashable, xr.DataArray] = {}
    for entry in reduce_plan:
        variable = value[entry.name]
        if not entry.dims:
            variables[entry.name] = variable
            continue
        local, local_error = guarded(
            lambda variable=variable, entry=entry: _local_extreme(
                runtime,
                variable,
                entry.dims,
                empty=locally_empty(variable) and entry.distributed,
                minimum=minimum,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
        )
        if not entry.distributed:
            if local_error is not None:
                raise local_error
            variables[entry.name] = local
            continue
        result = _combine_extreme(
            runtime,
            variable,
            local,
            entry.dims,
            minimum=minimum,
            skipna=skipna,
            error=local_error,
            comm=resolve_comm(runtime, old_meta, entry.comm_axes),
        )
        variables[entry.name] = result
    return finish(
        runtime,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        partition_dim=partition_dim,
        auto_candidates=repartition_candidates(reduce_plan),
    )


def any_reduce(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Return whether any value is true over the requested dimensions.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
        Default is ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Logical OR over the requested dimensions."""
    return _logical(
        runtime,
        value,
        dim,
        op=MPI.LOR,
        all_values=False,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def all_reduce(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Return whether all values are true over the requested dimensions.

    Parameters
    ----------
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce. ``None`` or ``...`` reduces all dimensions.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
        Default is ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Logical AND over the requested dimensions."""
    return _logical(
        runtime,
        value,
        dim,
        op=MPI.LAND,
        all_values=True,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def _logical(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None,
    *,
    op: MPI.Op,
    all_values: bool,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
) -> xr.Dataset | xr.DataArray:
    """Implement distributed logical reductions."""
    operation = "all" if all_values else "any"
    local_dim, dims = normalize_dim(value, dim)
    old_meta = get_mpi_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        method = value.all if all_values else value.any
        local_result = method(dim=local_dim, keep_attrs=keep_attrs)
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = reduction_plan(runtime, value, dims, old_meta, operation=operation)

    if isinstance(value, xr.DataArray):
        method = value.all if all_values else value.any
        local, local_error = guarded(
            lambda: method(dim=local_dim, keep_attrs=keep_attrs)
        )
        if not dims:
            if local_error is not None:
                raise local_error
            return local
        result = comm_reduce(
            runtime,
            local,
            op,
            expect_dtype=partial_dtype(value.dtype.str, operation, None),
            error=local_error,
            phase=f"MPI xarray {operation} reduction",
            comm=resolve_comm(runtime, old_meta, reduce_plan[0].comm_axes),
        )
        return finish(
            runtime,
            result,
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=repartition_candidates(reduce_plan),
        )

    variables: dict[Hashable, xr.DataArray] = {}
    for entry in reduce_plan:
        variable = value[entry.name]
        if not entry.dims:
            variables[entry.name] = variable
            continue
        method = variable.all if all_values else variable.any
        local, local_error = guarded(
            lambda method=method, entry=entry: method(
                dim=entry.dims, keep_attrs=keep_attrs
            )
        )
        if not entry.distributed:
            if local_error is not None:
                raise local_error
            variables[entry.name] = local
            continue
        result = comm_reduce(
            runtime,
            local,
            op,
            expect_dtype=partial_dtype(variable.dtype.str, operation, None),
            error=local_error,
            phase=f"MPI xarray {operation} reduction",
            comm=resolve_comm(runtime, old_meta, entry.comm_axes),
        )
        variables[entry.name] = result
    return finish(
        runtime,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        auto_candidates=repartition_candidates(reduce_plan),
        partition_dim=partition_dim,
    )


# -- first/last -------------------------------------------------------


def _first_last_local(
    runtime,
    variable: xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None,
    want_first: bool,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rank-local first/last valid value along ``dim``, and its any-valid
    mask (both without ``dim``). A partition of size zero along ``dim``
    reports ``any_valid=False`` everywhere."""
    size = int(variable.sizes[dim])
    if size == 0:
        template = variable.isel({dim: slice(0, 0)}).sum(
            dim=dim, skipna=False, keep_attrs=False
        )
        return template, xr.zeros_like(template, dtype=bool)

    if not skipna_enabled(variable.dtype, skipna):
        index = 0 if want_first else size - 1
        picked = variable.isel({dim: index}, drop=True)
        return picked, xr.ones_like(picked, dtype=bool)

    mask = variable.notnull()
    if want_first:
        index = mask.argmax(dim=dim)
    else:
        index = (size - 1) - mask.isel({dim: slice(None, None, -1)}).argmax(dim=dim)
    return variable.isel({dim: index}, drop=True), mask.any(dim=dim)


def _first_last_pick(
    runtime,
    variable: xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None,
    want_first: bool,
) -> xr.DataArray:
    """Rank-local first/last, used when ``dim`` is not the partition
    dimension. NaN where nothing was valid, for float/complex dtypes."""
    picked, any_valid = _first_last_local(
        runtime, variable, dim, skipna=skipna, want_first=want_first
    )
    return picked.where(any_valid) if variable.dtype.kind in "fc" else picked


def _first_last_combine(
    runtime,
    variable: xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None,
    want_first: bool,
    comm: MPI.Comm | None = None,
) -> xr.DataArray:
    """Combine rank-local first/last candidates into a global result.

    Ranks are ordered along ``dim`` by construction (the lowest-ranked
    member of ``comm`` owns the lowest global indices along ``dim``),
    so "first/last valid" reduces to "lowest/highest rank with any
    valid data", via two ``Allreduce`` calls:

    1. ``MIN``/``MAX`` elects, per element, the owning rank (a rank
       without valid data reports a sentinel that always loses).
    2. Every rank masks its candidate to zero/``False`` except where it
       is the elected owner; a ``SUM`` (``LOR`` for boolean data) then
       combines the masked candidates, recovering the one nonzero
       contribution per element exactly.

    ``comm`` matters here in a way it does not for the other
    reductions in this module: unlike a sum or an extreme, "first/
    last" is order-dependent, and rank order along ``dim`` is only
    meaningful within a communicator that varies *exclusively* along
    ``dim``. The caller therefore always passes (via
    :meth:`ReductionPlanningMixin._resolve_comm`) the sub-communicator
    for ``{dim}`` alone -- deliberately *not* unioned with any
    replicated axis the way an additive reduction's group is. That is
    still correct for a replicated axis: every rank replicated along
    it holds identical local data, so each such rank's own ``{dim}``-
    only sub-communicator independently computes the identical
    answer, with no cross-replica communication needed at all. It
    also does not need :attr:`~.common.PlanEntry.replica_count`
    correction the way an additive reduction does: the elected
    ``owner`` is a single rank per element (``MIN``/``MAX`` over a
    strictly unique rank id cannot tie), so exactly one rank's value
    ever contributes a nonzero term to the following ``SUM``,
    regardless of how many duplicate copies of the same data exist
    elsewhere.

    Elements with no valid data anywhere become NaN for float/complex
    dtypes; other dtypes keep their neutral placeholder, matching how
    :meth:`_combine_extreme` handles the same edge case for min/max.
    """
    candidate, any_valid = _first_last_local(
        runtime, variable, dim, skipna=skipna, want_first=want_first
    )
    active_comm = runtime.comm if comm is None else comm
    rank, size = active_comm.rank, active_comm.size
    sentinel = size if want_first else -1
    owner, error = guarded(lambda: xr.where(any_valid, rank, sentinel).astype(np.int32))
    owner = comm_reduce(
        runtime,
        owner,
        MPI.MIN if want_first else MPI.MAX,
        expect_dtype=np.dtype(np.int32),
        error=error,
        phase="MPI xarray first/last owner election",
        comm=comm,
    )
    is_owner = owner == rank

    kind = variable.dtype.kind
    neutral = False if kind == "b" else np.zeros((), dtype=variable.dtype).item()
    payload, error = guarded(lambda: candidate.where(is_owner, other=neutral))
    combined = comm_reduce(
        runtime,
        payload,
        MPI.LOR if kind == "b" else MPI.SUM,
        expect_dtype=variable.dtype,
        error=error,
        phase="MPI xarray first/last value reduction",
        comm=comm,
    )
    return combined.where(owner != sentinel) if kind in "fc" else combined


def first_reduce(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Select the first valid value along one dimension.

    Unlike the other reductions in this module, ``first``/``last``
    operate on exactly one dimension: they pick a position along it
    rather than collapsing a set of dimensions. ``skipna``/``keep_attrs``
    follow xarray semantics; MPI communication (two ``Allreduce`` calls)
    occurs only when ``dim`` is the active partition dimension."""
    return _first_or_last(
        runtime,
        value,
        dim,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        want_first=True,
    )


def last_reduce(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Select the last valid value along one dimension. See :meth:`first`."""
    return _first_or_last(
        runtime,
        value,
        dim,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        want_first=False,
    )


def _first_or_last(
    runtime,
    value: xr.Dataset | xr.DataArray,
    dim: str,
    *,
    skipna: bool | None,
    keep_attrs: bool | None,
    partition_dim: Hashable | Literal["auto"] | None,
    want_first: bool,
) -> xr.Dataset | xr.DataArray:
    """Shared implementation for :meth:`first` and :meth:`last`."""
    if not isinstance(dim, str):
        raise TypeError("MPI xarray first/last reduce exactly one dimension.")
    dims = (dim,)
    old_meta = get_mpi_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)

    if local_meta is not None:
        if isinstance(value, xr.DataArray):
            result = _first_last_pick(
                runtime, value, dim, skipna=skipna, want_first=want_first
            )
            if keep_attrs:
                result.attrs.update(value.attrs)
        else:
            result = value.map(
                functools.partial(_first_last_pick, runtime),
                dim=dim,
                skipna=skipna,
                want_first=want_first,
                keep_attrs=keep_attrs,
            )
        return finish_local_reduction(result, old_meta=local_meta)

    reduce_plan = reduction_plan(
        runtime, value, dims, old_meta, operation="first" if want_first else "last"
    )

    if isinstance(value, xr.DataArray):
        result = _first_last_combine(
            runtime,
            value,
            dim,
            skipna=skipna,
            want_first=want_first,
            comm=resolve_comm(runtime, old_meta, (dim,)),
        )
        if keep_attrs:
            result.attrs.update(value.attrs)
        return finish(
            runtime,
            result,
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=repartition_candidates(reduce_plan),
        )

    variables: dict[Hashable, xr.DataArray] = {}
    for entry in reduce_plan:
        variable = value[entry.name]
        if not entry.dims:
            variables[entry.name] = variable
            continue
        if entry.distributed:
            result = _first_last_combine(
                runtime,
                variable,
                dim,
                skipna=skipna,
                want_first=want_first,
                comm=resolve_comm(runtime, old_meta, (dim,)),
            )
        else:
            result = _first_last_pick(
                runtime, variable, dim, skipna=skipna, want_first=want_first
            )
        if keep_attrs:
            result.attrs.update(variable.attrs)
        variables[entry.name] = result
    return finish(
        runtime,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        auto_candidates=repartition_candidates(reduce_plan),
        partition_dim=partition_dim,
    )
