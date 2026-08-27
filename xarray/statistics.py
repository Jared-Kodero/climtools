"""Distributed standard deviation and variance.

Two-collective algorithm: first an ``Allreduce`` computes the exact global
mean (via :meth:`~.reductions.ReductionMixin.mean`, replicated on every
rank), then a second ``Allreduce`` sums each rank's local squared deviation
from that mean. This is simpler and easier to audit than a single-pass
parallel (Chan et al. 1979) combiner built on a custom commutative
``MPI.Op``, at the cost of one extra collective per call; for the partition
sizes this package targets, the collective's latency floor is negligible
next to the two local ``.sum()`` passes over the data. Note ``skipna``
selects the mean over valid values only.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from types import EllipsisType
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from mpi4py import MPI

from .common import _partial_dtype
from .engine import ReductionPlanningMixin
from .meta import get_mpi_meta

if TYPE_CHECKING:
    from ..mpi.runtime import MPIRuntime


class Statistics(ReductionPlanningMixin):
    """Distributed ``std``/``var``.

    Requires a ``self._runtime`` attribute set by :class:`~.mpi.XarrayMPI`.
    """

    _runtime: MPIRuntime

    def _var_or_std(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        skipna: bool | None,
        ddof: int,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
        root: bool,
    ) -> xr.Dataset | xr.DataArray:
        """Shared implementation for :meth:`var` and :meth:`std`."""
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )
        if local_meta is not None:
            method = value.std if root else value.var
            local_result = method(
                dim=local_dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs
            )
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation="std" if root else "var")

        def combine(
            variable: xr.DataArray,
            variable_dims: tuple[Hashable, ...],
            mean: xr.DataArray,
        ) -> xr.DataArray:
            """Combine rank-local sums of squared deviation into variance
            (``std`` when ``root``), via one ``Allreduce``."""
            deviation = variable - mean
            partial_sq_sum, error = self._guarded(
                lambda: (deviation * deviation).sum(
                    dim=variable_dims, skipna=skipna, min_count=None, keep_attrs=False
                )
            )
            global_sq_sum = self._comm_reduce(
                partial_sq_sum,
                MPI.SUM,
                expect_dtype=_partial_dtype(variable.dtype.str, "sum", skipna),
                error=error,
                phase="MPI xarray variance reduction",
            )
            denominator = self._count(variable, variable_dims) - ddof
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
            mean = self.mean(  # type: ignore[attr-defined]
                value, dim, skipna=skipna, keep_attrs=False, redistribute_on=None
            )
            result = combine(value, dims, mean)
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        mean_ds = self.mean(  # type: ignore[attr-defined]
            value, dim, skipna=skipna, keep_attrs=False, redistribute_on=None
        )
        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
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
            variables[entry.name] = combine(variable, entry.dims, mean_ds[entry.name])
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            auto_candidates=self._redistribution_candidates(plan),
        )

    def var(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
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
        return self._var_or_std(
            value,
            dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            root=False,
        )

    def std(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        ddof: int = 0,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
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
        return self._var_or_std(
            value,
            dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            root=True,
        )
