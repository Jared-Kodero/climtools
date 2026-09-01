"""Provide distributed groupby and resample reductions."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Hashable
from typing import Literal

import numpy as np
import pandas as pd
from mpi4py import MPI

import xarray as xr

from .common import extreme_identity, partial_dtype
from .meta import get_mpi_meta, set_mpi_meta, strip_mpi_meta
from .planning import (
    comm_reduce,
    dataset_result,
    finish,
    finish_local_reduction,
    local_reduction_meta,
    reduction_plan,
    resolve_comm,
)

_GROUP_DIM = "_mpi_group"
_GROUP_OPS = ("sum", "mean", "count", "min", "max")


def _resample_bin_labels(
    timestamps: pd.DatetimeIndex, freq: str, comm: MPI.Comm
) -> pd.DatetimeIndex:
    """Rank-consistent resample bin-start label per element of ``timestamps``.

    Dispatches on whether ``freq`` is a fixed-duration (pandas ``Tick``)
    offset -- ``"h"``, ``"6min"``, ``"D"``, ``"7D"``, and similar -- or a
    calendar-anchored one -- ``"MS"``, ``"YS"``, ``"W"``, ``"QS"``, and
    similar. See :meth:`Groupby.resample_reduce` for why these need
    different treatment: a calendar frequency's bins are already
    absolute and rank-independent, but a Tick frequency's are not
    unless every rank anchors to the same shared origin, which pandas'
    own ``origin=`` argument cannot be relied on to provide for
    day-or-coarser Tick frequencies.

    Parameters
    ----------
    timestamps : pandas.DatetimeIndex
        This rank's own local timestamps along the resampled dimension;
        may be empty on some ranks.
    freq : str
        Pandas frequency string, exactly as ``xarray.Dataset.resample``
        accepts.
    comm : mpi4py.MPI.Comm
        Communicator spanning every rank that holds a piece of the
        resampled dimension; used for exactly one scalar ``Allreduce``
        when ``freq`` is Tick-based, and not at all otherwise.

    Returns
    -------
    pandas.DatetimeIndex
        One bin-start label per element of ``timestamps``, identical in
        meaning (though each rank only sees its own labels) to what a
        single-process ``pandas.Series(...).resample(freq).mean()``
        would assign that same timestamp.
    """
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
        raise ValueError(f"resample(): non-positive Tick frequency {freq!r}.")

    offsets_ns = timestamps_ns.asi8 - anchor.value
    bin_index = offsets_ns // delta_ns  # floor division: works for negatives too
    label_ns = anchor.value + bin_index.astype(np.int64) * delta_ns
    return pd.DatetimeIndex(label_ns.astype("datetime64[ns]")).as_unit(timestamps.unit)


def _group_reduce_local(
    runtime,
    variable: xr.DataArray,
    dim: Hashable,
    group: xr.DataArray,
    *,
    op: str,
    skipna: bool | None,
) -> xr.DataArray:
    """Rank-local ``groupby(group).<op>(dim)``; passes a variable through
    unchanged if it doesn't have ``dim`` (relevant for
    :meth:`Dataset.map`, which calls this on every data variable)."""
    if dim not in variable.dims:
        return variable
    grouped = variable.groupby(group)
    if op == "count":
        return grouped.count(dim=dim, keep_attrs=False)
    method = getattr(grouped, op)
    return method(dim=dim, skipna=skipna, keep_attrs=False)


def _group_combine(
    runtime,
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
    """Combine rank-local per-group partials into a global result.

    ``mean`` costs two ``Allreduce`` calls (a sum and a count, divided
    afterward); every other supported op costs one. ``comm``/
    ``replica_count`` follow the same convention as
    :meth:`ReductionPlanningMixin._comm_reduce`: default to the full
    runtime communicator with no correction (the one-dimensional
    path, unchanged), or a Cartesian sub-communicator plus its
    duplication count under a multi-dimensional partition -- see
    :meth:`ReductionPlanningMixin._resolve_comm`. The min/max branch
    needs no ``replica_count`` correction, exactly as in
    :meth:`.reductions.Reduction._combine_extreme`: MIN/MAX are
    idempotent under duplication.
    """
    if op == "mean":
        local_sum = _group_reduce_local(
            runtime, variable, dim, group, op="sum", skipna=skipna
        )
        local_count = _group_reduce_local(
            runtime, variable, dim, group, op="count", skipna=None
        )
        local_sum = local_sum.reindex({_GROUP_DIM: global_labels}, fill_value=0)
        local_count = local_count.reindex({_GROUP_DIM: global_labels}, fill_value=0)
        global_sum = comm_reduce(
            runtime,
            local_sum,
            MPI.SUM,
            expect_dtype=partial_dtype(variable.dtype.str, "sum", skipna),
            phase="MPI xarray groupby sum reduction",
            comm=comm,
            replica_count=replica_count,
        )
        global_count = comm_reduce(
            runtime,
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
        local = _group_reduce_local(runtime, variable, dim, group, op=op, skipna=skipna)
        local = local.reindex({_GROUP_DIM: global_labels}, fill_value=0)
        return comm_reduce(
            runtime,
            local,
            MPI.SUM,
            expect_dtype=partial_dtype(variable.dtype.str, op, skipna),
            phase=f"MPI xarray groupby {op} reduction",
            comm=comm,
            replica_count=replica_count,
        )

    minimum = op == "min"
    local = _group_reduce_local(runtime, variable, dim, group, op=op, skipna=skipna)
    identity = extreme_identity(variable.dtype, minimum=minimum)
    local = local.reindex({_GROUP_DIM: global_labels}, fill_value=identity)
    return comm_reduce(
        runtime,
        local,
        MPI.MIN if minimum else MPI.MAX,
        expect_dtype=variable.dtype,
        phase=f"MPI xarray groupby {op} reduction",
        comm=comm,
    )


def groupby_reduce(
    runtime,
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
    value : xarray.Dataset or xarray.DataArray
        Object to reduce.
    dim : Hashable
        Dimension being grouped and reduced.
    labels : array-like
        Group key for every position along this rank's local ``dim``
        axis (same length as ``value.sizes[dim]``); need not be sorted
        or unique.
    op : {"sum", "mean", "count", "min", "max"}, optional
        Reduction applied within each group. Default ``"mean"``.
    skipna : bool or None, optional
        Missing-value behavior, following xarray semantics.
    keep_attrs : bool or None, optional
        Whether to preserve attributes.
    partition_dim : Hashable or {"auto"} or None, optional
        Partition placement after grouping; ``"auto"`` may place the
        result on the new group dimension. Default is ``"auto"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Reduced over ``dim``, with a new dimension of the same name
        indexed by the sorted, global set of group labels.

    Notes
    -----
    MPI communication occurs only when ``dim`` is the active partition
    dimension: one small ``allgather`` of distinct labels, then one
    ``Allreduce`` per variable (two for ``mean``)."""
    if op not in _GROUP_OPS:
        raise ValueError(f"Unsupported groupby op: {op!r}. Supported: {_GROUP_OPS}.")
    dims = (dim,)
    group = xr.DataArray(np.asarray(labels), dims=dim, name=_GROUP_DIM)
    old_meta = get_mpi_meta(value)
    local_meta = local_reduction_meta(old_meta, dims, partition_dim=partition_dim)

    if local_meta is not None:
        if isinstance(value, xr.DataArray):
            result = _group_reduce_local(
                runtime, value, dim, group, op=op, skipna=skipna
            )
            if keep_attrs:
                result.attrs.update(value.attrs)
        else:
            result = value.map(
                functools.partial(_group_reduce_local, runtime),
                dim=dim,
                group=group,
                op=op,
                skipna=skipna,
                keep_attrs=keep_attrs,
            )
        return finish_local_reduction(result, old_meta=local_meta)

    plan = reduction_plan(runtime, value, dims, old_meta, operation=op)
    local_labels = np.unique(group.values)
    labels_comm = resolve_comm(runtime, old_meta, (dim,))
    global_labels = np.unique(np.concatenate(labels_comm.allgather(local_labels)))

    if isinstance(value, xr.DataArray):
        result = _group_combine(
            runtime,
            value,
            dim,
            group,
            global_labels,
            op=op,
            skipna=skipna,
            comm=resolve_comm(runtime, old_meta, plan[0].comm_axes),
            replica_count=plan[0].replica_count,
        )
        if keep_attrs:
            result.attrs.update(value.attrs)
        return finish(
            runtime,
            result,
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=frozenset({_GROUP_DIM}),
        )

    variables: dict[Hashable, xr.DataArray] = {}
    for entry in plan:
        variable = value[entry.name]
        if not entry.dims:
            variables[entry.name] = variable
            continue
        if entry.distributed:
            result = _group_combine(
                runtime,
                variable,
                dim,
                group,
                global_labels,
                op=op,
                skipna=skipna,
                comm=resolve_comm(runtime, old_meta, entry.comm_axes),
                replica_count=entry.replica_count,
            )
        else:
            result = _group_reduce_local(
                runtime, variable, dim, group, op=op, skipna=skipna
            )
        if keep_attrs:
            result.attrs.update(variable.attrs)
        variables[entry.name] = result
    return finish(
        runtime,
        dataset_result(value, dims, variables),
        old_meta=old_meta,
        partition_dim=partition_dim,
        auto_candidates=frozenset({_GROUP_DIM}),
    )


def resample_reduce(
    runtime,
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

    A thin wrapper over :meth:`groupby_reduce`: group labels are each
    timestamp's resampled bin-start. Every rank must agree on
    identical bin edges without gathering the full coordinate, so
    this splits on frequency kind (see :func:`_resample_bin_labels`):

    - **Fixed-duration (Tick) frequencies** (``"h"``, ``"6min"``,
      ``"D"``, ``"7D"``, ...): bin edges are computed by direct
      integer-nanosecond arithmetic against a single shared anchor
      (an ``MPI.MIN`` reduction of every rank's local minimum
      timestamp -- one ``int64``, negligible cost), reproducing
      ``xarray``'s own default ``origin="start_day"`` resample
      anchor exactly. This is required, not just an optimization:
      pandas' own ``Series.resample(freq, origin=...)`` silently
      ignores ``origin`` for any frequency of a day or coarser (its
      own warning names the frequencies where ``origin`` *does*
      apply: ``h``, ``m``, ``s``, ``ms``, ``us``, ``ns`` -- ``D`` is
      not among them), always falling back to anchoring on
      whatever local slice of timestamps it was called with. Doing
      the previous, simpler thing -- calling ``.resample(freq)`` on
      each rank's own local timestamps -- would then silently
      anchor every rank's bin grid on that rank's own first local
      timestamp instead of the true global one, so ranks would
      disagree about where a bin starts. That is not a rounding
      difference: it fragments what should be one bin into two
      separate (wrong) labels at every rank boundary that does not
      already fall on a bin edge -- the common case, not an edge
      case -- silently corrupting the result with no error raised.
    - **Calendar-anchored frequencies** (``"MS"``, ``"YS"``,
      ``"W"``, ``"QS"``, ...): every rank's local
      ``.resample(freq).count().index`` already agrees, since these
      bin on the absolute calendar (month/year/week/quarter
      boundaries) independent of which subset of timestamps a rank
      happens to hold, so no shared anchor is needed.

    See :meth:`groupby_reduce` for parameters and MPI cost.

    Limitation
    ----------
    Pre-existing, not addressed by this pass: this is built on
    :meth:`groupby_reduce`, which -- like any group-by -- only
    returns a bin for a label that actually occurs in the data. It
    cannot materialize an empty bin. Native ``xarray``'s own
    ``.resample(freq).mean()`` always returns a *complete*, regular
    grid from the data's first to last timestamp, inserting ``NaN``
    for any bin with no observations -- e.g. resampling daily data
    to ``"6h"`` (a target frequency *finer* than the data's own
    spacing) yields three ``NaN`` bins between every pair of real
    days in native ``xarray``, and yields nothing at all for those
    gaps here. This does not affect the ordinary downsampling case
    (target frequency coarser than or equal to the data's own
    spacing, e.g. daily -> weekly/monthly), which is unaffected and
    is what :func:`_resample_bin_labels`'s fix targets; it only
    matters when upsampling to a finer grid than the source data.
    """
    timestamps = pd.DatetimeIndex(value[dim].values)
    labels = _resample_bin_labels(timestamps, freq, runtime.comm)
    result = groupby_reduce(
        runtime,
        value,
        dim,
        labels,
        op,
        skipna=skipna,
        keep_attrs=keep_attrs,
        partition_dim=partition_dim,
    )

    # groupby_reduce() always names its new dimension _GROUP_DIM
    # ("_mpi_group"), an internal convention appropriate for an
    # arbitrary label array. resample() groups by intervals of `dim`
    # itself, so -- mirroring plain xarray's own
    # `Dataset.resample(**{dim: freq}).mean()`, which keeps the
    # dimension named "time" (or whatever `dim` was), not some generic
    # group name -- the result is renamed back to `dim` here.
    if _GROUP_DIM not in getattr(result, "dims", ()):
        return result
    meta = get_mpi_meta(result)
    renamed = strip_mpi_meta(result).rename({_GROUP_DIM: dim})
    if meta is not None:
        set_mpi_meta(
            renamed,
            dim=dim,
            global_size=meta["global_size"],
            start=meta["start"],
            stop=meta["stop"],
            chunk_info=meta["chunk_info"],
        )
    return renamed
