"""Provide distributed groupby and resample reductions."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from mpi4py import MPI

import xarray as xr

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from .common import extreme_identity, partial_dtype
from .meta import mpp_get_meta, mpp_update_meta, strip_mpi_meta
from .mpp import mpp_reduce_scatter
from .planning import (
    dataset_result,
    finish_local_reduction,
    local_reduction_meta,
    mpp_comm_reduce,
    mpp_finish,
    mpp_reduction_plan,
    mpp_resolve_comm,
)

_GROUP_DIM = "_mpi_group"
_GROUP_OPS = ("sum", "mean", "count", "min", "max")


def _balanced_counts(total: int, size: int) -> list[int]:
    """Near-equal split of ``total`` into ``size`` nonnegative integer counts."""
    base, rem = divmod(total, size)
    return [base + (1 if r < rem else 0) for r in range(size)]


def _group_combine_scatter(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    group: xr.DataArray,
    global_labels: np.ndarray,
    *,
    op: str,
    skipna: bool | None,
    comm: MPI.Comm,
) -> tuple[xr.DataArray, int, int]:
    """Like :func:`_group_combine`, but each rank keeps only its own slice.

    Uses ``mpp_reduce_scatter`` instead of ``Allreduce`` so no rank ever
    materializes the full grouped result -- only worthwhile when the
    result will actually be distributed; a replicated partition
    dimension (no ``replica_count`` param here) stays on the Allreduce
    path in :func:`_group_combine` instead.
    """
    counts = _balanced_counts(len(global_labels), comm.size)
    start = sum(counts[: comm.rank])
    stop = start + counts[comm.rank]
    my_labels = global_labels[start:stop]

    if op == "mean":
        local_sum = _group_reduce_local(
            mpi_context, variable, dim, group, op="sum", skipna=skipna
        )
        local_count = _group_reduce_local(
            mpi_context, variable, dim, group, op="count", skipna=None
        )
        local_sum = local_sum.reindex({_GROUP_DIM: global_labels}, fill_value=0)
        local_count = local_count.reindex({_GROUP_DIM: global_labels}, fill_value=0)
        axis = local_sum.get_axis_num(_GROUP_DIM)
        global_sum = mpp_reduce_scatter(
            np.asarray(local_sum.values), MPI.SUM, comm, counts, axis=axis
        )
        global_count = mpp_reduce_scatter(
            np.asarray(local_count.values), MPI.SUM, comm, counts, axis=axis
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = global_sum / global_count
        dims = local_sum.dims
        result = xr.DataArray(raw, dims=dims).assign_coords({_GROUP_DIM: my_labels})
        if variable.dtype.kind in "fc" and result.dtype != variable.dtype:
            result = result.astype(variable.dtype, keep_attrs=True)
        return result.where(global_count > 0), start, stop

    if op in ("sum", "count"):
        local = _group_reduce_local(
            mpi_context, variable, dim, group, op=op, skipna=skipna
        )
        local = local.reindex({_GROUP_DIM: global_labels}, fill_value=0)
        axis = local.get_axis_num(_GROUP_DIM)
        raw = mpp_reduce_scatter(
            np.asarray(local.values), MPI.SUM, comm, counts, axis=axis
        )
        result = xr.DataArray(raw, dims=local.dims).assign_coords(
            {_GROUP_DIM: my_labels}
        )
        return result, start, stop

    minimum = op == "min"
    local = _group_reduce_local(mpi_context, variable, dim, group, op=op, skipna=skipna)
    identity = extreme_identity(variable.dtype, minimum=minimum)
    local = local.reindex({_GROUP_DIM: global_labels}, fill_value=identity)
    axis = local.get_axis_num(_GROUP_DIM)
    raw = mpp_reduce_scatter(
        np.asarray(local.values),
        MPI.MIN if minimum else MPI.MAX,
        comm,
        counts,
        axis=axis,
    )
    result = xr.DataArray(raw, dims=local.dims).assign_coords({_GROUP_DIM: my_labels})
    return result, start, stop


def _resample_bin_labels(
    timestamps: pd.DatetimeIndex, freq: str, comm: MPI.Comm
) -> pd.DatetimeIndex:
    """Rank-consistent resample bin-start label per element of ``timestamps``."""
    offset = pd.tseries.frequencies.to_offset(freq)

    # Fixed-duration ("Tick"-like) frequencies expose a working `.nanos`
    # property; note that pandas' own `Tick` base class is, confusingly,
    # NOT the right classifier here -- `pandas.tseries.offsets.Day` (so
    # `"D"`/`"7D"`) has a perfectly well-defined, always-24h `.nanos` but
    # is NOT a `Tick` subclass in this pandas version (only sub-day
    # units -- h/min/s/ms/us/ns -- are). Checking `.nanos` directly
    # covers both correctly instead of misclassifying `"D"`-and-coarser
    # Tick-equivalents as calendar-anchored.
    try:
        delta_ns = int(offset.nanos)
        fixed_duration = True
    except (ValueError, AttributeError):
        fixed_duration = False

    if not fixed_duration:
        # Calendar-anchored: absolute regardless of which timestamps
        # this rank happens to hold, so no shared origin is needed.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            edges = pd.Series(0, index=timestamps).resample(freq).count().index
        positions = edges.searchsorted(timestamps, side="right") - 1
        return edges[positions]

    # Tick-based: reproduce xarray/pandas' own default `origin="start_day"`
    # -- midnight of the day containing the *global* first timestamp --
    # via a single MPI.MIN reduction of every rank's local minimum, so
    # every rank bins against the identical anchor without gathering the
    # coordinate itself. A rank with no local timestamps contributes the
    # int64 maximum as a no-op sentinel rather than skewing the MIN.
    #
    # pandas (>=2.0) DatetimeIndex can carry any of several time
    # resolutions ("ns"/"us"/"ms"/"s"; `pd.date_range`'s own default
    # changed to "us" in pandas 3.0), and `.asi8` counts in *that*
    # index's own unit, not always nanoseconds -- normalize to "ns"
    # first so the raw-integer arithmetic below has an unambiguous,
    # fixed-width unit regardless of the caller's input resolution.
    timestamps_ns = timestamps.as_unit("ns")
    local_min_ns = (
        int(timestamps_ns.asi8.min()) if len(timestamps_ns) else np.iinfo(np.int64).max
    )
    global_min_ns = comm.allreduce(local_min_ns, op=MPI.MIN)
    if global_min_ns == np.iinfo(np.int64).max:
        # No rank holds any timestamps at all; nothing to label.
        return timestamps
    anchor = pd.Timestamp(global_min_ns, unit="ns").normalize()

    if delta_ns <= 0:
        raise ValueError(f"non-positive Tick frequency {freq!r}")

    offsets_ns = timestamps_ns.asi8 - anchor.value
    bin_index = offsets_ns // delta_ns  # floor division: works for negatives too
    label_ns = anchor.value + bin_index.astype(np.int64) * delta_ns
    return pd.DatetimeIndex(label_ns.astype("datetime64[ns]")).as_unit(timestamps.unit)


def _group_reduce_local(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    group: xr.DataArray,
    *,
    op: str,
    skipna: bool | None,
) -> xr.DataArray:
    """Apply a grouped reduction locally, preserving variables without ``dim``."""
    if dim not in variable.dims:
        return variable
    grouped = variable.groupby(group)
    if op == "count":
        return grouped.count(dim=dim, keep_attrs=False)
    method = getattr(grouped, op)
    return method(dim=dim, skipna=skipna, keep_attrs=False)


def _group_combine(
    mpi_context: MPIContext,
    variable: xr.DataArray,
    dim: Hashable,
    group: xr.DataArray,
    global_labels: np.ndarray,
    *,
    op: str,
    skipna: bool | None,
    comm: MPI.Comm | None = None,
    replica_count: int = 1,
) -> xr.DataArray:
    """Combine rank-local per-group partials into a global result."""
    if op == "mean":
        local_sum = _group_reduce_local(
            mpi_context, variable, dim, group, op="sum", skipna=skipna
        )
        local_count = _group_reduce_local(
            mpi_context, variable, dim, group, op="count", skipna=None
        )
        local_sum = local_sum.reindex({_GROUP_DIM: global_labels}, fill_value=0)
        local_count = local_count.reindex({_GROUP_DIM: global_labels}, fill_value=0)
        global_sum = mpp_comm_reduce(
            mpi_context,
            local_sum,
            MPI.SUM,
            expect_dtype=partial_dtype(variable.dtype.str, "sum", skipna),
            phase="MPI xarray groupby sum reduction",
            comm=comm,
            replica_count=replica_count,
        )
        global_count = mpp_comm_reduce(
            mpi_context,
            local_count,
            MPI.SUM,
            expect_dtype=partial_dtype(variable.dtype.str, "count", None),
            phase="MPI xarray groupby count reduction",
            comm=comm,
            replica_count=replica_count,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            result = global_sum / global_count
        # A groupby-mean's result is always array-shaped (one value per
        # group -- reducing to a single group still leaves a size-1
        # group axis, never a bare scalar), so it follows the same
        # dtype rule confirmed for an ordinary *partial* (array-result)
        # xarray reduction, not a full/scalar one: a non-floating dtype
        # promotes to float64 (the plain division above already does
        # this correctly on its own), but xarray keeps a floating or
        # complex dtype exactly as it was, which the plain division
        # above does NOT (float32/int64 division always promotes to
        # float64 -- confirmed directly against native
        # DataArray.groupby(...).mean(), which stays float32).
        if variable.dtype.kind in "fc" and result.dtype != variable.dtype:
            result = result.astype(variable.dtype, keep_attrs=True)
        return result.where(global_count > 0)

    if op in ("sum", "count"):
        local = _group_reduce_local(
            mpi_context, variable, dim, group, op=op, skipna=skipna
        )
        local = local.reindex({_GROUP_DIM: global_labels}, fill_value=0)
        return mpp_comm_reduce(
            mpi_context,
            local,
            MPI.SUM,
            expect_dtype=partial_dtype(variable.dtype.str, op, skipna),
            phase=f"MPI xarray groupby {op} reduction",
            comm=comm,
            replica_count=replica_count,
        )

    minimum = op == "min"
    local = _group_reduce_local(mpi_context, variable, dim, group, op=op, skipna=skipna)
    identity = extreme_identity(variable.dtype, minimum=minimum)
    local = local.reindex({_GROUP_DIM: global_labels}, fill_value=identity)
    return mpp_comm_reduce(
        mpi_context,
        local,
        MPI.MIN if minimum else MPI.MAX,
        expect_dtype=variable.dtype,
        phase=f"MPI xarray groupby {op} reduction",
        comm=comm,
    )


def mpp_groupby_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    labels: xr.DataArray | np.ndarray,
    op: Literal["sum", "mean", "count", "min", "max"] = "mean",
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Reduce ``value`` over ``dim``, grouped by ``labels``.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : Hashable
        Dimension being grouped and reduced.
    labels : array-like
        Group key for every position along this rank's local ``dim`` axis (same length as ``value.sizes[dim]``); need not be sorted or unique.
    op : {"sum", "mean", "count", "min", "max"}, optional
        Reduction applied within each group.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after grouping; ``"auto"`` may place the result on the new group dimension.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced over ``dim``, with a new dimension of the same name indexed by the sorted, global set of group labels.
    """
    if op not in _GROUP_OPS:
        raise ValueError(f"Unsupported groupby op: {op!r}. Supported: {_GROUP_OPS}.")
    dims = (dim,)
    group = xr.DataArray(np.asarray(labels), dims=dim, name=_GROUP_DIM)
    old_meta = mpp_get_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)

    if local_meta is not None:
        if isinstance(value, xr.DataArray):
            result = _group_reduce_local(
                mpi_context, value, dim, group, op=op, skipna=skipna
            )
            if keep_attrs:
                result.attrs.update(value.attrs)
        else:
            result = value.map(
                functools.partial(_group_reduce_local, mpi_context),
                dim=dim,
                group=group,
                op=op,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
        return finish_local_reduction(result, old_meta=local_meta)

    plan = mpp_reduction_plan(mpi_context, value, dims, old_meta, operation=op)
    local_labels = np.unique(group.values)
    labels_comm = mpp_resolve_comm(mpi_context, old_meta, (dim,))
    global_labels = np.unique(np.concatenate(labels_comm.allgather(local_labels)))

    if isinstance(value, xr.DataArray):
        combine_comm = mpp_resolve_comm(mpi_context, old_meta, plan[0].comm_axes)
        old_dims_da: tuple[Hashable, ...] = () if old_meta is None else old_meta["dims"]
        partition_removed_da = old_meta is not None and not any(
            d != dim for d in old_dims_da
        )
        # Scatter apart instead of Allreduce-then-slice whenever the
        # result is actually going to end up distributed: skips ever
        # materializing the full grouped/resampled result on every rank
        # (see _group_combine_scatter's docstring). Not applicable when
        # the caller explicitly wants a replicated result
        # (partition_dim=None), there's nothing worth splitting (a
        # single group, or a single rank), plan[0] is itself a replica
        # subgroup (every member of a replica group is *supposed* to
        # end up with the same answer, so scattering apart would defeat
        # that), or -- matching mpp_finish()'s own "at least one, but not
        # every, previous partition dimension survived" branch -- some
        # OTHER active partition dimension besides `dim` is still
        # present (a multi-dim Cartesian partition where only one axis
        # is being grouped over): that case keeps distributing on the
        # surviving dimension, exactly as before this reduction, not on
        # the new group dimension, so there is nothing to scatter onto.
        if (
            partition_removed_da
            and partition_dim is not None
            and len(global_labels) > 1
            and combine_comm.size > 1
            and plan[0].replica_count == 1
        ):
            result, start, stop = _group_combine_scatter(
                mpi_context,
                value,
                dim,
                group,
                global_labels,
                op=op,
                skipna=skipna,
                comm=combine_comm,
            )
            if keep_attrs:
                result.attrs.update(value.attrs)
            from .chunks import get_effective_chunk_size

            chunk_info = {
                str(other_dim): get_effective_chunk_size(
                    int(other_length), None, combine_comm.size
                )
                for other_dim, other_length in result.sizes.items()
            }
            mpp_update_meta(
                result,
                dim=_GROUP_DIM,
                global_size=len(global_labels),
                start=start,
                stop=stop,
                chunk_info=chunk_info,
            )
            return result

        result = _group_combine(
            mpi_context,
            value,
            dim,
            group,
            global_labels,
            op=op,
            skipna=skipna,
            comm=combine_comm,
            replica_count=plan[0].replica_count,
        )
        if keep_attrs:
            result.attrs.update(value.attrs)
        return mpp_finish(
            mpi_context,
            result,
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=frozenset({_GROUP_DIM}),
        )

    # Same Reduce_scatter question as the DataArray branch above, but for
    # a whole Dataset every variable in `variables` must end up the same
    # length along _GROUP_DIM -- so the choice is made once, for every
    # entry together, not per variable. Only usable when no entry needing
    # cross-rank combination is itself a replica subgroup (see the
    # DataArray branch's docstring on why that path stays Allreduce), and
    # every one of them shares the same combine communicator size (a
    # different-sized sub-comm per variable would need a different
    # counts split per variable, defeating a single consistent
    # _GROUP_DIM length across the Dataset).
    old_dims: tuple[Hashable, ...] = () if old_meta is None else old_meta["dims"]
    remaining_dims = tuple(d for d in old_dims if d != dim)
    partition_removed = old_meta is not None and not remaining_dims
    combine_comms = {
        entry.name: mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes)
        for entry in plan
        if entry.dims and entry.distributed
    }
    can_scatter = (
        partition_removed
        and partition_dim is not None
        and len(global_labels) > 1
        and combine_comms
        and len({c.size for c in combine_comms.values()}) == 1
        and all(
            entry.replica_count == 1
            for entry in plan
            if entry.dims and entry.distributed
        )
    )

    variables: dict[Hashable, xr.DataArray] = {}
    if can_scatter:
        scatter_comm = next(iter(combine_comms.values()))
        counts = _balanced_counts(len(global_labels), scatter_comm.size)
        start = sum(counts[: scatter_comm.rank])
        stop = start + counts[scatter_comm.rank]
        my_labels = global_labels[start:stop]
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            if entry.distributed:
                result, _s, _e = _group_combine_scatter(
                    mpi_context,
                    variable,
                    dim,
                    group,
                    global_labels,
                    op=op,
                    skipna=skipna,
                    comm=combine_comms[entry.name],
                )
            else:
                # Already fully replicated pre-reduction, so its local
                # reduce alone already covers every group in
                # `global_labels` (see the module-level note this
                # mirrors in _group_combine's docstring) -- reindexed by
                # *label value*, not position, since nothing guarantees
                # this variable's own local group order matches
                # `global_labels`'s sorted order.
                full = _group_reduce_local(
                    mpi_context, variable, dim, group, op=op, skipna=skipna
                )
                fill = (
                    extreme_identity(variable.dtype, minimum=(op == "min"))
                    if op in ("min", "max")
                    else 0
                )
                result = full.reindex({_GROUP_DIM: my_labels}, fill_value=fill)
            if keep_attrs:
                result.attrs.update(variable.attrs)
            variables[entry.name] = result
        from .chunks import get_effective_chunk_size

        result_ds = dataset_result(value, dims, variables)
        chunk_info = {
            str(other_dim): get_effective_chunk_size(
                int(other_length), None, scatter_comm.size
            )
            for other_dim, other_length in result_ds.sizes.items()
        }
        mpp_update_meta(
            result_ds,
            dim=_GROUP_DIM,
            global_size=len(global_labels),
            start=start,
            stop=stop,
            chunk_info=chunk_info,
        )
        return result_ds

    for entry in plan:
        variable = value[entry.name]
        if not entry.dims:
            variables[entry.name] = variable
            continue
        if entry.distributed:
            result = _group_combine(
                mpi_context,
                variable,
                dim,
                group,
                global_labels,
                op=op,
                skipna=skipna,
                comm=combine_comms.get(entry.name)
                or mpp_resolve_comm(mpi_context, old_meta, entry.comm_axes),
                replica_count=entry.replica_count,
            )
        else:
            result = _group_reduce_local(
                mpi_context, variable, dim, group, op=op, skipna=skipna
            )
        if keep_attrs:
            result.attrs.update(variable.attrs)
        variables[entry.name] = result
    return mpp_finish(
        mpi_context,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        partition_dim=partition_dim,
        auto_candidates=frozenset({_GROUP_DIM}),
    )


def mpp_resample_reduce(
    mpi_context: MPIContext,
    value: xr.Dataset | xr.DataArray,
    dim: Hashable,
    freq: str,
    op: Literal["sum", "mean", "count", "min", "max"] = "mean",
    *,
    skipna: bool | None = None,
    keep_attrs: bool | None = None,
    partition_dim: Hashable | Literal["auto"] | None = "auto",
) -> xr.Dataset | xr.DataArray:
    """Resample a datetime dimension to ``freq``, then reduce.

    Parameters
    ----------
    mpi_context : MPIContext
        MPI context used for communication.
    value : xr.Dataset | xr.DataArray
        Distributed xarray object.
    dim : Hashable
        Dimension to operate on.
    freq : str
        Resampling frequency.
    op : Literal['sum', 'mean', 'count', 'min', 'max']
        Reduction or MPI operation.
    skipna : bool | None
        Whether to ignore missing values.
    keep_attrs : bool | None
        Whether to preserve xarray attributes.
    partition_dim : Hashable | Literal['auto'] | None
        Partition dimension to use for the result.
    Returns
    -------
    xr.Dataset | xr.DataArray
        Resampled distributed reduction.
    """
    timestamps = pd.DatetimeIndex(value[dim].values)
    labels = _resample_bin_labels(timestamps, freq, mpi_context.comm)
    result = mpp_groupby_reduce(
        mpi_context,
        value,
        dim,
        labels,
        op,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )

    # mpp_groupby_reduce() always names its new dimension _GROUP_DIM
    # ("_mpi_group"), an internal convention appropriate for an
    # arbitrary label array. resample() groups by intervals of `dim`
    # itself, so -- mirroring plain xarray's own
    # `Dataset.resample(**{dim: freq}).mean()`, which keeps the
    # dimension named "time" (or whatever `dim` was), not some generic
    # group name -- the result is renamed back to `dim` here.
    if _GROUP_DIM not in getattr(result, "dims", ()):
        return result
    meta = mpp_get_meta(result)
    renamed = strip_mpi_meta(result).rename({_GROUP_DIM: dim})
    if meta is not None and _GROUP_DIM in meta["dims"]:
        # _GROUP_DIM is itself an active partition dimension only when
        # mpp_groupby_reduce() took its cross-rank combine path and
        # mpp_finish() then (auto-)redistributed the reduced result onto
        # the new group dimension -- rename that one entry to `dim`,
        # keeping every other partition dimension (relevant under a
        # multi-dimensional partition) unchanged.
        new_dims = tuple(dim if d == _GROUP_DIM else d for d in meta["dims"])
        remap = {(dim if d == _GROUP_DIM else d): d for d in meta["dims"]}
        mpp_update_meta(
            renamed,
            dim=new_dims,
            global_size={nd: meta["global_sizes"][od] for nd, od in remap.items()},
            start={nd: meta["starts"][od] for nd, od in remap.items()},
            stop={nd: meta["stops"][od] for nd, od in remap.items()},
            chunk_info=meta["chunk_info"],
            cart=meta.get("cart"),
        )
    elif meta is not None:
        # The active partition dimension is something else entirely
        # (the common resample() case: `dim` -- the axis being
        # resampled -- is not the distributed axis at all, so
        # mpp_groupby_reduce() took its local, non-communicating path and
        # returned `meta` describing that other, untouched dimension
        # unchanged). That metadata is still exactly correct for
        # `renamed` (only `_GROUP_DIM` was renamed; every other
        # dimension, including the real partition one, is untouched)
        # and must be reattached as-is -- an earlier version of this
        # function instead force-relabeled it under `dim`, mislabeling
        # that other dimension's own start/stop/global_size as if they
        # belonged to `dim` and corrupting `.meta` for every resample()
        # call where the partition dimension isn't the one resampled.
        mpp_update_meta(
            renamed,
            dim=meta["dims"],
            global_size=dict(meta["global_sizes"]),
            start=dict(meta["starts"]),
            stop=dict(meta["stops"]),
            chunk_info=meta["chunk_info"],
            cart=meta.get("cart"),
        )
    return renamed
