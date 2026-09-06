"""Provide distributed numerical and logical reductions."""

from __future__ import annotations

import functools
from collections.abc import Hashable, Iterable
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from mpi4py import MPI

import xarray as xr

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from .common import extreme_identity, op_name, partial_dtype
from .meta import mpp_get_meta
from .mpp import _mpp_reduce, mpp_reduce_scatter
from .planning import (
    mpp_comm_reduce,
    mpp_count_valid_values,
    dataset_result,
    mpp_finish,
    mpp_finish_scatter,
    mpp_plan_scatter_target,
    mpp_scatter_replicated_slice,
    finish_local_reduction,
    guarded,
    local_reduction_meta,
    normalize_dim,
    mpp_reduction_plan,
    repartition_candidates,
    mpp_resolve_comm,
    skipna_enabled,
)


# Leading dimension the packed integer companions of a reproducing product
# travel under, so mantissa and companions can each go through one collective.
_PROD_FIELD_DIM = "_mpp_prod_field"


def _combine_sum_or_prod(
    mpi_context: MPIContext,
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
    scatter: tuple[Hashable, list[int]] | None = None,
) -> xr.DataArray:
    """Combine rank-local sum or product partials.

    ``scatter``, when given, is forwarded to :func:`~.planning.mpp_comm_reduce`
    (and to the ``min_count`` valid-value count below) so each rank keeps
    only its own post-reduction slice instead of materializing the full
    combined result -- see :func:`~.planning.mpp_scatter_target`.
    """
    if op_name(op) == "PROD":
        result = _combine_prod(
            mpi_context,
            value,
            partial,
            dims,
            skipna=skipna,
            error=error,
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
        )
    else:
        result = mpp_comm_reduce(
            mpi_context,
            partial,
            op,
            expect_dtype=partial_dtype(value.dtype.str, "sum", skipna),
            error=error,
            phase="MPI xarray sum/prod reduction",
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
        )
    global_count = None
    if min_count is not None and skipna_enabled(value.dtype, skipna):
        global_count = mpp_count_valid_values(
            mpi_context,
            value,
            dims,
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
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


def _combine_prod(
    mpi_context: MPIContext,
    value: xr.DataArray,
    partial: xr.DataArray | None,
    dims: tuple[Hashable, ...],
    *,
    skipna: bool | None,
    error: BaseException | None,
    comm: MPI.Comm | None,
    replica_count: int,
    scatter: tuple[Hashable, list[int]] | None,
) -> xr.DataArray:
    """Combine rank-local products without letting overflow decide the answer.

    Multiplying rank-local ``np.prod`` partials is not associative once a
    partial leaves the representable range: ``inf * 0`` is NaN, so the result
    depends on which rank happened to hold the zero and therefore on the rank
    count. Instead each rank contributes the over/underflow-free
    ``(mantissa, companions)`` pair of :func:`~.mpp.mpp_prod_decompose`, and
    the two halves reduce under ``PROD`` and ``SUM`` respectively -- both
    associative -- through the same :func:`~.planning.mpp_comm_reduce` every
    other reduction uses, so the scatter and replicated-axis handling is
    inherited unchanged.

    ``partial`` is used only for its coordinates and dtype; its values are
    recomputed here in the safe representation.
    """
    from .mpp import mpp_prod_decompose, mpp_prod_recombine

    mantissa_da: xr.DataArray | None = None
    companion_da: xr.DataArray | None = None
    if error is None and partial is not None:
        try:
            axes = tuple(value.dims.index(d) for d in dims)
            mantissa, companions = mpp_prod_decompose(np.asarray(value.values), axes)
            mantissa_da = partial.copy(data=mantissa.astype(np.float64))
            companion_da = xr.DataArray(
                companions,
                dims=(_PROD_FIELD_DIM, *partial.dims),
                coords={
                    d: partial.coords[d] for d in partial.dims if d in partial.coords
                },
            )
        except BaseException as exc:  # noqa: BLE001 - deferred to raise_if_error
            error = exc

    global_mantissa = mpp_comm_reduce(
        mpi_context,
        mantissa_da,
        MPI.PROD,
        expect_dtype=np.dtype(np.float64),
        error=error,
        phase="MPI xarray prod reduction (mantissa)",
        comm=comm,
        scatter=scatter,
    )
    global_companions = mpp_comm_reduce(
        mpi_context,
        companion_da,
        MPI.SUM,
        expect_dtype=np.dtype(np.int64),
        error=error,
        phase="MPI xarray prod reduction (exponent and tallies)",
        comm=comm,
        scatter=scatter,
    )

    mantissa_values = np.asarray(global_mantissa.values)
    companion_values = np.asarray(global_companions.values)
    if replica_count != 1:
        # Each of the replica_count duplicate copies contributed one factor,
        # so the raw product is the true one raised to that power. Undoing it
        # is a root of the mantissa and an exact division of the exponent and
        # of every tally -- mpp_comm_reduce only knows how to undo replication
        # for SUM, which is why it is done here instead of being delegated.
        # Every companion is an exact multiple of replica_count (the copies
        # are bit-identical), so those divisions are exact; only the mantissa
        # root is inexact, and it is taken on a value bounded in (0, 1].
        mantissa_values = mantissa_values ** (1.0 / replica_count)
        companion_values = companion_values // replica_count

    expect = partial_dtype(value.dtype.str, "prod", skipna)
    combined = mpp_prod_recombine(mantissa_values, companion_values, expect)
    return global_mantissa.copy(data=combined)


def _global_valid_count(
    mpi_context: MPIContext,
    value: xr.DataArray,
    template: xr.DataArray,
    dims: tuple[Hashable, ...],
    *,
    skipna: bool | None,
    comm: MPI.Comm | None,
    replica_count: int,
    scatter: tuple[Hashable, list[int]] | None,
) -> xr.DataArray:
    """Global valid-value count for a mean, communicating only when necessary.

    When missing values cannot occur -- an integer or boolean field, or an
    explicit ``skipna=False`` -- every element counts, so the denominator is
    just the product of the reduced dimensions' *global* extents. That is
    already in the partition metadata, exactly as FMS reads ``gxsize``/
    ``gysize`` off the domain in ``mpp_global_sum`` rather than reducing to
    find them. Taking it from there removes both a second full pass over the
    data and a second collective from the common case, leaving ``mean`` with
    the same single ``Allreduce`` as ``sum``.

    Otherwise the count genuinely depends on where the NaNs are and is
    reduced as before.
    """
    if skipna_enabled(value.dtype, skipna):
        return mpp_count_valid_values(
            mpi_context,
            value,
            dims,
            comm=comm,
            replica_count=replica_count,
            scatter=scatter,
        )

    meta = mpp_get_meta(value)
    global_sizes = dict(meta["global_sizes"]) if meta is not None else {}
    total = 1
    for reduced in dims:
        total *= int(global_sizes.get(reduced, value.sizes[reduced]))
    return xr.full_like(template, total, dtype=np.int64)


def _combine_mean(
    mpi_context: MPIContext,
    value: xr.DataArray,
    partial_sum: xr.DataArray | None,
    dims: tuple[Hashable, ...],
    *,
    skipna: bool | None = None,
    error: BaseException | None = None,
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
    scatter: tuple[Hashable, list[int]] | None = None,
) -> xr.DataArray:
    """Combine rank-local sums and counts into a global mean.

    ``scatter``, when given, is forwarded to both the sum and the
    valid-value count below (see :func:`~.planning.mpp_scatter_target`), so
    the division that follows happens between two already-matching local
    slices instead of two full replicated arrays.
    """
    global_sum = mpp_comm_reduce(
        mpi_context,
        partial_sum,
        MPI.SUM,
        expect_dtype=partial_dtype(value.dtype.str, "sum", skipna),
        error=error,
        phase="MPI xarray mean reduction",
        comm=comm,
        replica_count=replica_count,
        scatter=scatter,
    )
    global_count = _global_valid_count(
        mpi_context,
        value,
        global_sum,
        dims,
        skipna=skipna,
        comm=comm,
        replica_count=replica_count,
        scatter=scatter,
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
    mpi_context: MPIContext,
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
    mpi_context: MPIContext,
    value: xr.DataArray,
    partial: xr.DataArray | None,
    dims: tuple[Hashable, ...],
    *,
    minimum: bool,
    skipna: bool | None,
    error: BaseException | None = None,
    comm: MPI.Comm | None = None,
    scatter: tuple[Hashable, list[int]] | None = None,
) -> xr.DataArray:
    """Combine rank-local min/max partials across ranks.

    ``scatter``, when given (see :func:`~.planning.mpp_scatter_target`),
    is forwarded to :func:`~.planning.mpp_comm_reduce` for the boolean and
    non-float dtype branches below (which route through it directly), and
    handled explicitly for the float branch's own ``Reduce_scatter`` call,
    since that branch packs value and validity into one ``(2, N)`` buffer
    and calls :func:`~.mpp.mpp_reduce_scatter`/`_mpp_reduce` directly
    rather than going through :func:`~.planning.mpp_comm_reduce`. FMS's
    own ``mpp_max``/``mpp_min`` always broadcast the full reduced result
    to every PE (see ``mpp.F90``); this scattering behavior has no FMS
    counterpart at all, same as the sum/mean case (see
    :func:`~.mpp.mpp_reduce_scatter`).
    """
    # Use the agreed variable dtype, not a rank-local partial dtype. Empty
    # partitions follow a different local path, and dtype-dependent branching
    # could desynchronize collectives. Min/max also require no promotion; using
    # the declared dtype avoids bottleneck's float32-to-float64 scalar promotion.
    operation = "min" if minimum else "max"
    expect_dtype = value.dtype
    kind = value.dtype.kind
    if kind == "b":
        return mpp_comm_reduce(
            mpi_context,
            partial,
            MPI.LAND if minimum else MPI.LOR,
            expect_dtype=expect_dtype,
            error=error,
            phase=f"MPI xarray {operation} reduction",
            comm=comm,
            scatter=scatter,
        )

    op = MPI.MIN if minimum else MPI.MAX
    if kind != "f":
        return mpp_comm_reduce(
            mpi_context,
            partial,
            op,
            expect_dtype=expect_dtype,
            error=error,
            phase=f"MPI xarray {operation} reduction",
            comm=comm,
            scatter=scatter,
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
            None
            if scatter is None
            else (str(scatter[0]), tuple(int(c) for c in scatter[1])),
        )
    )
    mpi_context.raise_if_error(
        error, f"MPI xarray {operation} reduction", signature, comm=comm
    )
    if send is None or template is None:
        raise AssertionError("MPI xarray reduction buffer is missing.")

    resolved_comm = comm if comm is not None else mpi_context.comm
    if scatter is not None:
        target, counts = scatter
        axis = 1 + template.get_axis_num(target)
        recv = mpp_reduce_scatter(send, op, resolved_comm, counts, axis=axis)
        start = sum(counts[: resolved_comm.rank])
        stop = start + counts[resolved_comm.rank]
        template = template.isel({target: slice(start, stop)})
    else:
        recv = _mpp_reduce(send, op, resolved_comm)

    shape = tuple(int(length) for length in template.shape)
    combined = np.asarray(recv[0]).reshape(shape)
    valid = (np.asarray(recv[1]).reshape(shape) * flip) > 0
    masked = np.where(valid, combined, np.asarray(np.nan, dtype=expect_dtype))
    return template.copy(data=np.asarray(masked, dtype=expect_dtype).reshape(shape))


def mpp_sum_reduce(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    min_count : int or None, optional
        Minimum number of valid values required.
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
    return _sum_prod(
        mpi_context,
        value,
        dim,
        op=MPI.SUM,
        product=False,
        skipna=skipna,
        min_count=min_count,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def mpp_prod_reduce(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    min_count : int or None, optional
        Minimum number of valid values required.
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
    return _sum_prod(
        mpi_context,
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
    mpi_context: MPIContext,
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
    old_meta = mpp_get_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        method = value.prod if product else value.sum
        local_result = method(
            dim=local_dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs
        )
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = mpp_reduction_plan(
        mpi_context, value, dims, old_meta, operation=operation
    )

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
        scattered = mpp_plan_scatter_target(
            mpi_context, old_meta, dims, partition_dim, reduce_plan
        )
        comm = (
            scattered[2]
            if scattered is not None
            else mpp_resolve_comm(mpi_context, old_meta, reduce_plan[0].comm_axes)
        )
        scatter = None if scattered is None else scattered[:2]
        result = _combine_sum_or_prod(
            mpi_context,
            value,
            local,
            dims,
            op,
            skipna=skipna,
            min_count=min_count,
            error=local_error,
            comm=comm,
            replica_count=reduce_plan[0].replica_count,
            scatter=scatter,
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
        comm = (
            scattered[2]
            if scattered is not None
            else mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes)
        )
        scatter = None if scattered is None else scattered[:2]
        result = _combine_sum_or_prod(
            mpi_context,
            variable,
            local,
            entry.dims,
            op,
            skipna=skipna,
            min_count=min_count,
            error=local_error,
            comm=comm,
            replica_count=entry.replica_count,
            scatter=scatter,
        )
        variables[entry.name] = result
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


def _materialize_local(value: xr.DataArray) -> xr.DataArray:
    """Force a dask-backed local array to a concrete, in-memory one.

    Avoids re-running the upstream task graph once per derived quantity
    (mean's sum and valid-count both read ``value`` independently) --
    already this rank's own slice, so no extra data volume is read.
    """
    return value.load() if getattr(value, "chunks", None) is not None else value


def mpp_mean_reduce(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
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
    local_dim, dims = normalize_dim(value, dim)
    old_meta = mpp_get_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        local_result = value.mean(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = mpp_reduction_plan(
        mpi_context, value, dims, old_meta, operation="mean"
    )

    if isinstance(value, xr.DataArray):
        if not dims:
            local_mean = value.mean(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
            return local_mean
        value = _materialize_local(value)
        local_sum, local_error = guarded(
            lambda: value.sum(
                dim=local_dim, skipna=skipna, min_count=None, keep_attrs=keep_attrs
            )
        )
        scattered = mpp_plan_scatter_target(
            mpi_context, old_meta, dims, partition_dim, reduce_plan
        )
        comm = (
            scattered[2]
            if scattered is not None
            else mpp_resolve_comm(mpi_context, old_meta, reduce_plan[0].comm_axes)
        )
        scatter = None if scattered is None else scattered[:2]
        result = _combine_mean(
            mpi_context,
            value,
            local_sum,
            dims,
            skipna=skipna,
            error=local_error,
            comm=comm,
            replica_count=reduce_plan[0].replica_count,
            scatter=scatter,
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
            variables[entry.name] = variable.mean(
                dim=entry.dims, skipna=skipna, keep_attrs=keep_attrs
            )
            continue
        variable = _materialize_local(variable)
        local_sum, local_error = guarded(
            lambda variable=variable, entry=entry: variable.sum(
                dim=entry.dims, skipna=skipna, min_count=None, keep_attrs=keep_attrs
            )
        )
        comm = (
            scattered[2]
            if scattered is not None
            else mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes)
        )
        scatter = None if scattered is None else scattered[:2]
        result = _combine_mean(
            mpi_context,
            variable,
            local_sum,
            entry.dims,
            skipna=skipna,
            error=local_error,
            comm=comm,
            replica_count=entry.replica_count,
            scatter=scatter,
        )
        variables[entry.name] = result
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


def mpp_min_reduce(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
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
    return _min_max(
        mpi_context,
        value,
        dim,
        minimum=True,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def mpp_max_reduce(
    mpi_context: MPIContext,
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
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
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
    return _min_max(
        mpi_context,
        value,
        dim,
        minimum=False,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def _min_max(
    mpi_context: MPIContext,
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
    old_meta = mpp_get_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        method = value.min if minimum else value.max
        local_result = method(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = mpp_reduction_plan(
        mpi_context, value, dims, old_meta, operation=operation
    )

    def locally_empty(variable: xr.DataArray) -> bool:
        """Return whether the local variable is empty along any owned partition axis."""
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
                mpi_context,
                value,
                dims,
                empty=locally_empty(value),
                minimum=minimum,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
        )
        scattered = mpp_plan_scatter_target(
            mpi_context, old_meta, dims, partition_dim, reduce_plan
        )
        comm = (
            scattered[2]
            if scattered is not None
            else mpp_resolve_comm(mpi_context, old_meta, reduce_plan[0].comm_axes)
        )
        result = _combine_extreme(
            mpi_context,
            value,
            local,
            dims,
            minimum=minimum,
            skipna=skipna,
            error=local_error,
            comm=comm,
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
        local, local_error = guarded(
            lambda variable=variable, entry=entry: _local_extreme(
                mpi_context,
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
        comm = (
            scattered[2]
            if scattered is not None
            else mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes)
        )
        result = _combine_extreme(
            mpi_context,
            variable,
            local,
            entry.dims,
            minimum=minimum,
            skipna=skipna,
            error=local_error,
            comm=comm,
            scatter=None if scattered is None else scattered[:2],
        )
        variables[entry.name] = result
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


def mpp_any_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Return whether any value is true over the requested dimensions.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Logical OR over the requested dimensions -- see
        :func:`~.planning.finish` for the exact replication/
        no-duplication guarantee this carries.
    """
    return _logical(
        mpi_context,
        value,
        dim,
        op=MPI.LOR,
        all_values=False,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def mpp_all_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str | Iterable[Hashable] | EllipsisType | None = None,
    *,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Return whether all values are true over the requested dimensions.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : str, iterable of Hashable, ..., or None, optional
        Dimensions to reduce.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after reducing the active partition dimension.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Logical AND over the requested dimensions -- see
        :func:`~.planning.finish` for the exact replication/
        no-duplication guarantee this carries.
    """
    return _logical(
        mpi_context,
        value,
        dim,
        op=MPI.LAND,
        all_values=True,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )


def _logical(
    mpi_context: MPIContext,
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
    old_meta = mpp_get_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)
    if local_meta is not None:
        method = value.all if all_values else value.any
        local_result = method(dim=local_dim, keep_attrs=keep_attrs)
        return finish_local_reduction(local_result, old_meta=local_meta)

    reduce_plan = mpp_reduction_plan(
        mpi_context, value, dims, old_meta, operation=operation
    )

    if isinstance(value, xr.DataArray):
        method = value.all if all_values else value.any
        local, local_error = guarded(
            lambda: method(dim=local_dim, keep_attrs=keep_attrs)
        )
        if not dims:
            if local_error is not None:
                raise local_error
            return local
        result = mpp_comm_reduce(
            mpi_context,
            local,
            op,
            expect_dtype=partial_dtype(value.dtype.str, operation, None),
            error=local_error,
            phase=f"MPI xarray {operation} reduction",
            comm=mpp_resolve_comm(mpi_context, old_meta, reduce_plan[0].comm_axes),
        )
        return mpp_finish(
            mpi_context,
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
        result = mpp_comm_reduce(
            mpi_context,
            local,
            op,
            expect_dtype=partial_dtype(variable.dtype.str, operation, None),
            error=local_error,
            phase=f"MPI xarray {operation} reduction",
            comm=mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes),
        )
        variables[entry.name] = result
    return mpp_finish(
        mpi_context,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        auto_candidates=repartition_candidates(reduce_plan),
        partition_dim=partition_dim,
    )


def _first_last_local(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None,
    want_first: bool,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rank-local first/last valid value along ``dim``, and its any-valid mask (both without ``dim``)."""
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
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None,
    want_first: bool,
) -> xr.DataArray:
    """Rank-local first/last, used when ``dim`` is not the partition dimension."""
    picked, any_valid = _first_last_local(
        mpi_context, variable, dim, skipna=skipna, want_first=want_first
    )
    return picked.where(any_valid) if variable.dtype.kind in "fc" else picked


def _first_last_combine(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    *,
    skipna: bool | None,
    want_first: bool,
    comm: MPI.Comm | None = None,
) -> xr.DataArray:
    """Combine rank-local first/last candidates into a global result."""
    candidate, any_valid = _first_last_local(
        mpi_context, variable, dim, skipna=skipna, want_first=want_first
    )
    active_comm = mpi_context.comm if comm is None else comm
    rank, size = active_comm.rank, active_comm.size
    sentinel = size if want_first else -1
    owner, error = guarded(lambda: xr.where(any_valid, rank, sentinel).astype(np.int32))
    owner = mpp_comm_reduce(
        mpi_context,
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
    combined = mpp_comm_reduce(
        mpi_context,
        payload,
        MPI.LOR if kind == "b" else MPI.SUM,
        expect_dtype=variable.dtype,
        error=error,
        phase="MPI xarray first/last value reduction",
        comm=comm,
    )
    result = combined.where(owner != sentinel) if kind in "fc" else combined

    # Any coordinate of `variable` that itself varies along `dim` (most
    # commonly `dim`'s own dimension coordinate, e.g. "lat" or a real
    # "time" axis) rides along with the vectorized `.isel(dim=index,
    # drop=True)` inside `_first_last_local`: `candidate` ends up
    # carrying that coordinate's value *at the locally-picked index*,
    # not a scalar. `mpp_comm_reduce` above only Allreduces the requested
    # data array and otherwise copies `payload`'s own (rank-local)
    # coordinates onto the result verbatim -- correct for every other
    # caller in this module, where a surviving coordinate is already
    # identical on every rank, but wrong here: each rank's own local
    # pick differs, so left alone this coordinate silently reports
    # whichever value *this* rank's own local slice happened to pick,
    # not the value at the true, cross-rank-elected first/last
    # position. It needs the identical owner-election combine the data
    # itself just got.
    index_coords = {
        name: coord for name, coord in variable.coords.items() if dim in coord.dims
    }
    if index_coords:
        combined_coords: dict[Hashable, xr.DataArray] = {}
        for name, coord in index_coords.items():
            local_coord = candidate.coords[name]
            coord_kind = local_coord.dtype.kind
            # datetime64/timedelta64 (a real "time" axis, most commonly)
            # have no MPI reduction operator; Allreduce their lossless
            # int64 view instead -- the same reinterpretation
            # mpi.mpp.mpp_update_domains's `_view` uses for halo exchange
            # -- and cast back afterward.
            as_int = coord_kind in "mM"
            reducible = local_coord.astype(np.int64) if as_int else local_coord
            reducible_kind = reducible.dtype.kind
            coord_neutral = (
                False
                if reducible_kind == "b"
                else np.zeros((), dtype=reducible.dtype).item()
            )
            coord_payload, coord_error = guarded(
                lambda reducible=reducible: reducible.where(
                    is_owner, other=coord_neutral
                )
            )
            coord_combined = mpp_comm_reduce(
                mpi_context,
                coord_payload,
                MPI.LOR if reducible_kind == "b" else MPI.SUM,
                expect_dtype=reducible.dtype,
                error=coord_error,
                phase="MPI xarray first/last coordinate reduction",
                comm=comm,
            )
            if reducible_kind in "fc":
                coord_combined = coord_combined.where(owner != sentinel)
            combined_coords[name] = (
                coord_combined.astype(local_coord.dtype) if as_int else coord_combined
            )
        result = result.assign_coords(combined_coords)
    return result


def mpp_first_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Select the first valid value along one dimension.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xr.Dataset | xr.DataArray
        Distributed xarray object.
    dim : str
        Dimension to operate on.
    skipna : bool | None
        Whether to ignore missing values.
    keep_attrs : bool | None
        Whether to preserve xarray attributes.
    partition_dim : Hashable | Literal['auto'] | None
        Partition dimension to use for the result.
    Returns
    -------
    xr.Dataset | xr.DataArray
        First valid value along the requested dimension -- see
        :func:`~.planning.finish` for the exact replication/
        no-duplication guarantee this carries.
    """
    return _first_or_last(
        mpi_context,
        value,
        dim,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        want_first=True,
    )


def mpp_last_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: str,
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Select the last valid value along one dimension.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xr.Dataset | xr.DataArray
        Distributed xarray object.
    dim : str
        Dimension to operate on.
    skipna : bool | None
        Whether to ignore missing values.
    keep_attrs : bool | None
        Whether to preserve xarray attributes.
    partition_dim : Hashable | Literal['auto'] | None
        Partition dimension to use for the result.
    Returns
    -------
    xr.Dataset | xr.DataArray
        Last valid value along the requested dimension -- see
        :func:`~.planning.finish` for the exact replication/
        no-duplication guarantee this carries.
    """
    return _first_or_last(
        mpi_context,
        value,
        dim,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
        want_first=False,
    )


def _first_or_last(
    mpi_context: MPIContext,
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
    old_meta = mpp_get_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)

    if local_meta is not None:
        if isinstance(value, xr.DataArray):
            result = _first_last_pick(
                mpi_context, value, dim, skipna=skipna, want_first=want_first
            )
            if keep_attrs:
                result.attrs.update(value.attrs)
        else:
            result = value.map(
                functools.partial(_first_last_pick, mpi_context),
                dim=dim,
                skipna=skipna,
                want_first=want_first,
                keep_attrs=keep_attrs,
            )
        return finish_local_reduction(result, old_meta=local_meta)

    reduce_plan = mpp_reduction_plan(
        mpi_context, value, dims, old_meta, operation="first" if want_first else "last"
    )

    if isinstance(value, xr.DataArray):
        result = _first_last_combine(
            mpi_context,
            value,
            dim,
            skipna=skipna,
            want_first=want_first,
            comm=mpp_resolve_comm(mpi_context, old_meta, (dim,)),
        )
        if keep_attrs:
            result.attrs.update(value.attrs)
        return mpp_finish(
            mpi_context,
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
                mpi_context,
                variable,
                dim,
                skipna=skipna,
                want_first=want_first,
                comm=mpp_resolve_comm(mpi_context, old_meta, (dim,)),
            )
        else:
            result = _first_last_pick(
                mpi_context, variable, dim, skipna=skipna, want_first=want_first
            )
        if keep_attrs:
            result.attrs.update(variable.attrs)
        variables[entry.name] = result
    return mpp_finish(
        mpi_context,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        auto_candidates=repartition_candidates(reduce_plan),
        partition_dim=partition_dim,
    )
