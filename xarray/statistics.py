"""Distributed variance and standard deviation.

Both use the two-pass form: the global mean first, then the global sum of
squared deviations about it. Results follow the same replication guarantee
as :mod:`~.reductions`.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from ..mpi.mpi_init import MPI

if TYPE_CHECKING:
    from ..mpi.context import MPIContext

from .common import PlanEntry, partial_dtype
from .planning import (
    guarded,
    mpp_comm_reduce,
    mpp_count_valid_values,
    mpp_global_reduce,
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
    """Shared implementation for :func:`mpp_var` and :func:`mpp_std`."""
    cached: list[Any] = []

    def global_mean() -> xr.Dataset | xr.DataArray:
        """Return the global mean, computed once and reused per variable."""
        if not cached:
            cached.append(
                mpp_mean_reduce(
                    mpi_context,
                    value,
                    dim,
                    skipna=skipna,
                    keep_attrs=False,
                    partition_dim=None,
                )
            )
        return cached[0]

    def serial(obj: Any, dims: Any) -> Any:
        """Reduce without communication."""
        method = obj.std if root else obj.var
        return method(dim=dims, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs)

    def combine(
        variable: xr.DataArray,
        dims: tuple[Hashable, ...],
        entry: PlanEntry,
        comm: MPI.Comm,
        scatter: tuple[Hashable, list[int]] | None,
    ) -> xr.DataArray:
        """Combine local squared deviations into a global variance."""
        mean = global_mean()
        if not isinstance(mean, xr.DataArray):
            mean = mean[entry.name]
        deviation = variable - mean
        # Squared deviations carry ``deviation.dtype`` because integer inputs
        # are promoted before reduction.
        partial, error = guarded(
            lambda: (deviation * deviation).sum(
                dim=dims, skipna=skipna, min_count=None, keep_attrs=False
            )
        )
        total = mpp_comm_reduce(
            mpi_context,
            partial,
            MPI.SUM,
            expect_dtype=partial_dtype(deviation.dtype.str, "sum", skipna),
            error=error,
            phase="MPI xarray variance reduction",
            comm=comm,
            replica_count=entry.replica_count,
            scatter=scatter,
        )
        denominator = (
            mpp_count_valid_values(
                mpi_context,
                variable,
                dims,
                comm=comm,
                replica_count=entry.replica_count,
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
            result = total / divisor
        result = result.where(denominator > 0)
        if result.dtype != target:
            result = result.astype(target, keep_attrs=True)
        if root:
            result = np.sqrt(result)
        if keep_attrs:
            result.attrs.update(variable.attrs)
        return result

    return mpp_global_reduce(
        mpi_context,
        value,
        dim,
        operation="std" if root else "var",
        serial=serial,
        combine=combine,
        partition_dim=partition_dim,
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
    """Compute the variance of a distributed xarray object."""
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
    """Compute the standard deviation of a distributed xarray object."""
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
