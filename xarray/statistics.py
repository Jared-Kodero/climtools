"""Compute distributed variance and standard deviation.

The implementation uses one collective for the mean and one for squared
deviations.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from types import EllipsisType
from typing import TYPE_CHECKING, Literal

import numpy as np
from mpi4py import MPI

import xarray as xr

from .common import _partial_dtype
from .meta import get_mpi_meta
from .planning import ReductionPlanningMixin

if TYPE_CHECKING:
    from ..mpi.runtime import MPIRuntime


class Statistics(ReductionPlanningMixin):
    """Provide distributed variance and standard-deviation reductions.

    The host class must provide ``self._runtime``.
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
        partition_dim: Hashable | Literal["auto"] | None,
        root: bool,
    ) -> xr.Dataset | xr.DataArray:
        """Shared implementation for :meth:`var` and :meth:`std`."""
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, partition_dim=partition_dim
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
            *,
            comm: MPI.Comm | None = None,
            replica_count: int = 1,
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
                comm=comm,
                replica_count=replica_count,
            )
            denominator = (
                self._count(
                    variable, variable_dims, comm=comm, replica_count=replica_count
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
            mean = self.mean(  # type: ignore[attr-defined]
                value, dim, skipna=skipna, keep_attrs=False, partition_dim=None
            )
            result = combine(
                value,
                dims,
                mean,
                comm=self._resolve_comm(old_meta, plan[0].comm_axes),
                replica_count=plan[0].replica_count,
            )
            return self._finish(
                result,
                old_meta=old_meta,
                partition_dim=partition_dim,
                auto_candidates=self._repartition_candidates(plan),
            )

        mean_ds = self.mean(  # type: ignore[attr-defined]
            value, dim, skipna=skipna, keep_attrs=False, partition_dim=None
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
            variables[entry.name] = combine(
                variable,
                entry.dims,
                mean_ds[entry.name],
                comm=self._resolve_comm(old_meta, entry.comm_axes),
                replica_count=entry.replica_count,
            )
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            partition_dim=partition_dim,
            auto_candidates=self._repartition_candidates(plan),
        )

    def var(
        self,
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
        return self._var_or_std(
            value,
            dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
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
        return self._var_or_std(
            value,
            dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            partition_dim=partition_dim,
            root=True,
        )
