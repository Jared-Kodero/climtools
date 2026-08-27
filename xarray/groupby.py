"""Distributed ``groupby``/``resample`` reductions.

Supports ``sum``, ``mean``, ``count``, ``min``, ``max`` grouped by an
arbitrary label array, including the case where the group boundaries cross
MPI rank boundaries (e.g. resampling a time dimension that is itself the
active partition dimension).

Algorithm: every rank reduces its own partition locally per group, then all
ranks ``allgather`` their (small) set of distinct group labels and agree on
one global, sorted label axis. Each rank reindexes its local per-group
partial onto that shared axis (missing groups filled with the operation's
identity element), and the reindexed partials -- now identically shaped on
every rank -- are combined with the same ``Allreduce`` primitive
(:meth:`~.engine.ReductionPlanningMixin._comm_reduce`) the rest of this
package uses. This reuses the existing collective machinery and needs no new
communication pattern, but its cost scales with ``n_groups`` times the size
of the non-grouped dimensions, gathered onto every rank -- appropriate for
coarsening reductions with a modest number of groups (daily/monthly/yearly
resampling, categorical groupby), not for high-cardinality grouping."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
from mpi4py import MPI

from .common import _extreme_identity, _partial_dtype
from .engine import ReductionPlanningMixin
from .meta import get_mpi_meta

if TYPE_CHECKING:
    from ..mpi.runtime import MPIRuntime

_GROUP_DIM = "_mpi_group"
_GROUP_OPS = ("sum", "mean", "count", "min", "max")


class Groupby(ReductionPlanningMixin):
    """Distributed ``groupby_reduce``/``resample_reduce``.

    Requires a ``self._runtime`` attribute set by :class:`~.mpi.XarrayMPI`.
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
    ) -> xr.DataArray:
        """Combine rank-local per-group partials into a global result.

        ``mean`` costs two ``Allreduce`` calls (a sum and a count, divided
        afterward); every other supported op costs one."""
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
            )
            global_count = self._comm_reduce(
                local_count,
                MPI.SUM,
                expect_dtype=_partial_dtype(variable.dtype.str, "count", None),
                phase="MPI xarray groupby count reduction",
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
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
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
            old_meta, dims, redistribute_on=redistribute_on
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
        global_labels = np.unique(
            np.concatenate(self._runtime.comm.allgather(local_labels))
        )

        if isinstance(value, xr.DataArray):
            result = self._group_combine(
                value, dim, group, global_labels, op=op, skipna=skipna
            )
            if keep_attrs:
                result.attrs.update(value.attrs)
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
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
                    variable, dim, group, global_labels, op=op, skipna=skipna
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
            redistribute_on=redistribute_on,
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
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Resample a datetime dimension to ``freq``, then reduce.

        A thin wrapper over :meth:`groupby_reduce`: group labels are the
        pandas period start for ``freq`` (e.g. ``"D"``, ``"MS"``, ``"YS"``)
        computed from this rank's local ``dim`` coordinate, so ranks agree on
        bin boundaries without any extra communication. See
        :meth:`groupby_reduce` for parameters and MPI cost."""
        labels = pd.DatetimeIndex(value[dim].values).to_period(freq).to_timestamp()
        return self.groupby_reduce(
            value,
            dim,
            labels,
            op,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )
