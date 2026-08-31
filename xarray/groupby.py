"""Provide distributed groupby and resample reductions."""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from mpi4py import MPI

import xarray as xr

from .common import _extreme_identity, _partial_dtype
from .meta import get_mpi_meta, set_mpi_meta, strip_mpi_meta
from .planning import ReductionPlanningMixin

if TYPE_CHECKING:
    from ..mpi.runtime import MPIRuntime

_GROUP_DIM = "_mpi_group"
_GROUP_OPS = ("sum", "mean", "count", "min", "max")


class Groupby(ReductionPlanningMixin):
    """Provide distributed groupby and resample reductions.

    The host class must provide ``self._runtime``.
    """

    _runtime: MPIRuntime

    def _group_reduce_local(
        self,
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
        self,
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
            local_sum = self._group_reduce_local(
                variable, dim, group, op="sum", skipna=skipna
            )
            local_count = self._group_reduce_local(
                variable, dim, group, op="count", skipna=None
            )
            local_sum = local_sum.reindex({_GROUP_DIM: global_labels}, fill_value=0)
            local_count = local_count.reindex({_GROUP_DIM: global_labels}, fill_value=0)
            global_sum = self._comm_reduce(
                local_sum,
                MPI.SUM,
                expect_dtype=_partial_dtype(variable.dtype.str, "sum", skipna),
                phase="MPI xarray groupby sum reduction",
                comm=comm,
                replica_count=replica_count,
            )
            global_count = self._comm_reduce(
                local_count,
                MPI.SUM,
                expect_dtype=_partial_dtype(variable.dtype.str, "count", None),
                phase="MPI xarray groupby count reduction",
                comm=comm,
                replica_count=replica_count,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                return (global_sum / global_count).where(global_count > 0)

        if op in ("sum", "count"):
            local = self._group_reduce_local(variable, dim, group, op=op, skipna=skipna)
            local = local.reindex({_GROUP_DIM: global_labels}, fill_value=0)
            return self._comm_reduce(
                local,
                MPI.SUM,
                expect_dtype=_partial_dtype(variable.dtype.str, op, skipna),
                phase=f"MPI xarray groupby {op} reduction",
                comm=comm,
                replica_count=replica_count,
            )

        minimum = op == "min"
        local = self._group_reduce_local(variable, dim, group, op=op, skipna=skipna)
        identity = _extreme_identity(variable.dtype, minimum=minimum)
        local = local.reindex({_GROUP_DIM: global_labels}, fill_value=identity)
        return self._comm_reduce(
            local,
            MPI.MIN if minimum else MPI.MAX,
            expect_dtype=variable.dtype,
            phase=f"MPI xarray groupby {op} reduction",
            comm=comm,
        )

    def groupby_reduce(
        self,
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
            raise ValueError(
                f"Unsupported groupby op: {op!r}. Supported: {_GROUP_OPS}."
            )
        dims = (dim,)
        group = xr.DataArray(np.asarray(labels), dims=dim, name=_GROUP_DIM)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, partition_dim=partition_dim
        )

        if local_meta is not None:
            if isinstance(value, xr.DataArray):
                result = self._group_reduce_local(
                    value, dim, group, op=op, skipna=skipna
                )
                if keep_attrs:
                    result.attrs.update(value.attrs)
            else:
                result = value.map(
                    self._group_reduce_local,
                    dim=dim,
                    group=group,
                    op=op,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                )
            return self._finish_local_reduction(result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation=op)
        local_labels = np.unique(group.values)
        labels_comm = self._resolve_comm(old_meta, (dim,))
        global_labels = np.unique(np.concatenate(labels_comm.allgather(local_labels)))

        if isinstance(value, xr.DataArray):
            result = self._group_combine(
                value,
                dim,
                group,
                global_labels,
                op=op,
                skipna=skipna,
                comm=self._resolve_comm(old_meta, plan[0].comm_axes),
                replica_count=plan[0].replica_count,
            )
            if keep_attrs:
                result.attrs.update(value.attrs)
            return self._finish(
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
                result = self._group_combine(
                    variable,
                    dim,
                    group,
                    global_labels,
                    op=op,
                    skipna=skipna,
                    comm=self._resolve_comm(old_meta, entry.comm_axes),
                    replica_count=entry.replica_count,
                )
            else:
                result = self._group_reduce_local(
                    variable, dim, group, op=op, skipna=skipna
                )
            if keep_attrs:
                result.attrs.update(variable.attrs)
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=frozenset({_GROUP_DIM}),
        )

    def resample_reduce(
        self,
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
        timestamp's resampled bin-start, computed from this rank's local
        ``dim`` coordinate against a fixed, data-independent grid anchored
        at the Unix epoch (``origin="epoch"``) -- not the rank's own local
        min/max -- so every rank agrees on identical bin edges without any
        extra communication, exactly reproducing ``xarray``'s own
        ``.resample(freq)`` bins (e.g. ``"D"``, ``"MS"``, ``"YS"``, and,
        unlike an earlier ``DatetimeIndex.to_period(freq)``-based
        implementation, frequency multiples such as ``"12h"``/``"6min"``/
        ``"3D"`` too -- ``to_period`` silently mis-bins those, since pandas
        ``Period`` arithmetic with a multiple does not snap to a shared
        grid the way ``resample``/``Grouper`` does, instead giving nearly
        every timestamp its own singleton period). See :meth:`groupby_reduce`
        for parameters and MPI cost."""
        timestamps = pd.DatetimeIndex(value[dim].values)
        with warnings.catch_warnings():
            # pandas warns that `origin` has no effect for calendar
            # (non-Tick-like, e.g. "MS"/"YS") frequencies -- expected and
            # harmless here: those frequencies are already anchored to an
            # absolute calendar grid regardless of origin, which is exactly
            # the rank-independent property this needs.
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            edges = (
                pd.Series(0, index=timestamps)
                .resample(freq, origin="epoch")
                .count()
                .index
            )
        positions = edges.searchsorted(timestamps, side="right") - 1
        labels = edges[positions]
        result = self.groupby_reduce(
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
