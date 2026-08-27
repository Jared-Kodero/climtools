"""Distributed sum, product, mean, min, max, first, last, any, and all.

Concrete numerical/logical reductions built on
:class:`~.engine.ReductionPlanningMixin`. Each public method plans
the reduction once, runs a rank-local partial, combines partials with one
``Allreduce`` (two for ``first``/``last``), and hands the result to
:meth:`~.engine.ReductionPlanningMixin._finish` for metadata
restoration and optional redistribution.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr
from mpi4py import MPI

from .common import _extreme_identity, _op_name, _partial_dtype
from .engine import ReductionPlanningMixin
from .meta import get_mpi_meta

if TYPE_CHECKING:
    from ..mpi.runtime import MPIRuntime


class Reduction(ReductionPlanningMixin):
    """Distributed ``sum``/``prod``/``mean``/``min``/``max``/``first``/
    ``last``/``any``/``all``.

    Requires a ``self._runtime`` attribute set by :class:`~.mpi.XarrayMPI`.
    """

    _runtime: MPIRuntime

    # -- per-variable combination --------------------------------------------

    def _combine_sum_or_prod(
        self,
        value: xr.DataArray,
        partial: xr.DataArray,
        dims: tuple[Hashable, ...],
        op: MPI.Op,
        *,
        skipna: bool | None,
        min_count: int | None,
        error: BaseException | None = None,
    ) -> xr.DataArray:
        """Combine rank-local sum or product partials."""
        result = self._comm_reduce(
            partial,
            op,
            expect_dtype=_partial_dtype(
                value.dtype.str, "prod" if _op_name(op) == "PROD" else "sum", skipna
            ),
            error=error,
            phase="MPI xarray sum/prod reduction",
        )
        global_count = None
        if min_count is not None and self._skipna_enabled(value.dtype, skipna):
            global_count = self._count(value, dims)
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
        self,
        value: xr.DataArray,
        partial_sum: xr.DataArray | None,
        dims: tuple[Hashable, ...],
        *,
        skipna: bool | None = None,
        error: BaseException | None = None,
    ) -> xr.DataArray:
        """Combine rank-local sums and counts into a global mean."""
        global_sum = self._comm_reduce(
            partial_sum,
            MPI.SUM,
            expect_dtype=_partial_dtype(value.dtype.str, "sum", skipna),
            error=error,
            phase="MPI xarray mean reduction",
        )
        global_count = self._count(value, dims)
        # Divide in the dtype numpy.mean would produce for this input. Dividing
        # the float32 sum by the int64 count directly would promote the whole
        # array to float64 and then cast it back, costing two full-width
        # temporaries for a result that is float32 either way.
        target = np.asarray(np.mean(np.zeros(1, dtype=value.dtype))).dtype
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
        self,
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
            identity = _extreme_identity(variable.dtype, minimum=minimum)
            template = variable.sum(
                dim=variable_dims, skipna=False, keep_attrs=keep_attrs
            )
            return xr.full_like(template, identity, dtype=variable.dtype)
        method = variable.min if minimum else variable.max
        return method(dim=variable_dims, skipna=skipna, keep_attrs=keep_attrs)

    def _combine_extreme(
        self,
        value: xr.DataArray,
        partial: xr.DataArray | None,
        dims: tuple[Hashable, ...],
        *,
        minimum: bool,
        skipna: bool | None,
        error: BaseException | None = None,
    ) -> xr.DataArray:
        """Combine rank-local min/max partials across ranks."""
        # Use the agreed variable dtype, not a rank-local partial dtype. Empty
        # partitions follow a different local path, and dtype-dependent branching
        # could desynchronize collectives. Min/max also require no promotion; using
        # the declared dtype avoids bottleneck's float32-to-float64 scalar promotion.
        operation = "min" if minimum else "max"
        expect_dtype = value.dtype
        kind = value.dtype.kind
        if kind == "b":
            return self._comm_reduce(
                partial,
                MPI.LAND if minimum else MPI.LOR,
                expect_dtype=expect_dtype,
                error=error,
                phase=f"MPI xarray {operation} reduction",
            )

        op = MPI.MIN if minimum else MPI.MAX
        if kind != "f":
            return self._comm_reduce(
                partial,
                op,
                expect_dtype=expect_dtype,
                error=error,
                phase=f"MPI xarray {operation} reduction",
            )

        # Floating reductions carry validity beside the extreme so empty or all-NaN
        # partitions can use an identity without confusing it with real infinity.
        # Encoding the flag in the same buffer avoids a second boolean collective.
        send: np.ndarray[Any, Any] | None = None
        template: xr.DataArray | None = None
        skipna_enabled = self._skipna_enabled(value.dtype, skipna)
        # ANY valid rank suffices under skipna; without it every rank must be
        # NaN-free for the result to be defined.
        flip = -1.0 if ((not minimum) != skipna_enabled) else 1.0

        if error is None:
            try:
                identity = _extreme_identity(expect_dtype, minimum=minimum)
                if skipna_enabled:
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
                _op_name(op),
                send.dtype.str,
                tuple(int(length) for length in send.shape),
            )
        )
        self._runtime.raise_if_error(
            error, f"MPI xarray {operation} reduction", signature
        )
        if send is None or template is None:
            raise AssertionError("MPI xarray reduction buffer is missing.")

        recv = self._exchange(send, op)

        shape = tuple(int(length) for length in template.shape)
        combined = np.asarray(recv[0]).reshape(shape)
        valid = (np.asarray(recv[1]).reshape(shape) * flip) > 0
        masked = np.where(valid, combined, np.asarray(np.nan, dtype=expect_dtype))
        return template.copy(data=np.asarray(masked, dtype=expect_dtype).reshape(shape))

    # -- public reductions ---------------------------------------------------

    def sum(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
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
        return self._sum_prod(
            value,
            dim,
            op=MPI.SUM,
            product=False,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def prod(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
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
        return self._sum_prod(
            value,
            dim,
            op=MPI.PROD,
            product=True,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def _sum_prod(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        op: MPI.Op,
        product: bool,
        skipna: bool | None,
        min_count: int | None,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
    ) -> xr.Dataset | xr.DataArray:
        """Implement distributed sum and product reductions."""
        operation = "prod" if product else "sum"
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )
        if local_meta is not None:
            method = value.prod if product else value.sum
            local_result = method(
                dim=local_dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs
            )
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation=operation)

        if isinstance(value, xr.DataArray):
            method = value.prod if product else value.sum
            local, local_error = self._guarded(
                lambda: method(
                    dim=local_dim, skipna=skipna, min_count=None, keep_attrs=keep_attrs
                )
            )
            if not dims:
                if local_error is not None:
                    raise local_error
                return local
            result = self._combine_sum_or_prod(
                value,
                local,
                dims,
                op,
                skipna=skipna,
                min_count=min_count,
                error=local_error,
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            method = variable.prod if product else variable.sum
            local, local_error = self._guarded(
                lambda method=method, entry=entry: method(
                    dim=entry.dims, skipna=skipna, min_count=None, keep_attrs=keep_attrs
                )
            )
            if not entry.distributed:
                if local_error is not None:
                    raise local_error
                variables[entry.name] = local
                continue
            result = self._combine_sum_or_prod(
                variable,
                local,
                entry.dims,
                op,
                skipna=skipna,
                min_count=min_count,
                error=local_error,
            )
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            auto_candidates=self._redistribution_candidates(plan),
        )

    def mean(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
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
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )
        if local_meta is not None:
            local_result = value.mean(
                dim=local_dim, skipna=skipna, keep_attrs=keep_attrs
            )
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation="mean")

        if isinstance(value, xr.DataArray):
            if not dims:
                local_mean = value.mean(
                    dim=local_dim, skipna=skipna, keep_attrs=keep_attrs
                )
                return local_mean
            local_sum, local_error = self._guarded(
                lambda: value.sum(
                    dim=local_dim, skipna=skipna, min_count=None, keep_attrs=keep_attrs
                )
            )
            result = self._combine_mean(
                value, local_sum, dims, skipna=skipna, error=local_error
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            if not entry.distributed:
                variables[entry.name] = variable.mean(
                    dim=entry.dims, skipna=skipna, keep_attrs=keep_attrs
                )
                continue
            local_sum, local_error = self._guarded(
                lambda variable=variable, entry=entry: variable.sum(
                    dim=entry.dims, skipna=skipna, min_count=None, keep_attrs=keep_attrs
                )
            )
            result = self._combine_mean(
                variable, local_sum, entry.dims, skipna=skipna, error=local_error
            )
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            auto_candidates=self._redistribution_candidates(plan),
        )

    def min(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Reduced object."""
        return self._min_max(
            value,
            dim,
            minimum=True,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def max(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Reduced object."""
        return self._min_max(
            value,
            dim,
            minimum=False,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def _min_max(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        minimum: bool,
        skipna: bool | None,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
    ) -> xr.Dataset | xr.DataArray:
        """Implement distributed minimum and maximum reductions."""
        operation = "min" if minimum else "max"
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )
        if local_meta is not None:
            method = value.min if minimum else value.max
            local_result = method(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation=operation)
        empty_partition = (
            old_meta is not None
            and old_meta["dim"] in value.dims
            and (int(value.sizes[old_meta["dim"]]) == 0)
        )

        if isinstance(value, xr.DataArray):
            if not dims:
                method = value.min if minimum else value.max
                return method(dim=local_dim, skipna=skipna, keep_attrs=keep_attrs)
            local, local_error = self._guarded(
                lambda: self._local_extreme(
                    value,
                    dims,
                    empty=empty_partition,
                    minimum=minimum,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                )
            )
            result = self._combine_extreme(
                value, local, dims, minimum=minimum, skipna=skipna, error=local_error
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            local, local_error = self._guarded(
                lambda variable=variable, entry=entry: self._local_extreme(
                    variable,
                    entry.dims,
                    empty=empty_partition and entry.distributed,
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
            result = self._combine_extreme(
                variable,
                local,
                entry.dims,
                minimum=minimum,
                skipna=skipna,
                error=local_error,
            )
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            redistribute_on=redistribute_on,
            auto_candidates=self._redistribution_candidates(plan),
        )

    def any(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Logical OR over the requested dimensions."""
        return self._logical(
            value,
            dim,
            op=MPI.LOR,
            all_values=False,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def all(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
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
        redistribute_on : Hashable or {"auto"} or None, optional
            Partition placement after reducing the active partition dimension.
            Default is ``"auto"``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Logical AND over the requested dimensions."""
        return self._logical(
            value,
            dim,
            op=MPI.LAND,
            all_values=True,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
        )

    def _logical(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
        *,
        op: MPI.Op,
        all_values: bool,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
    ) -> xr.Dataset | xr.DataArray:
        """Implement distributed logical reductions."""
        operation = "all" if all_values else "any"
        local_dim, dims = self._normalize_dim(value, dim)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )
        if local_meta is not None:
            method = value.all if all_values else value.any
            local_result = method(dim=local_dim, keep_attrs=keep_attrs)
            return self._finish_local_reduction(local_result, old_meta=local_meta)

        plan = self._plan(value, dims, old_meta, operation=operation)

        if isinstance(value, xr.DataArray):
            method = value.all if all_values else value.any
            local, local_error = self._guarded(
                lambda: method(dim=local_dim, keep_attrs=keep_attrs)
            )
            if not dims:
                if local_error is not None:
                    raise local_error
                return local
            result = self._comm_reduce(
                local,
                op,
                expect_dtype=_partial_dtype(value.dtype.str, operation, None),
                error=local_error,
                phase=f"MPI xarray {operation} reduction",
            )
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            method = variable.all if all_values else variable.any
            local, local_error = self._guarded(
                lambda method=method, entry=entry: method(
                    dim=entry.dims, keep_attrs=keep_attrs
                )
            )
            if not entry.distributed:
                if local_error is not None:
                    raise local_error
                variables[entry.name] = local
                continue
            result = self._comm_reduce(
                local,
                op,
                expect_dtype=_partial_dtype(variable.dtype.str, operation, None),
                error=local_error,
                phase=f"MPI xarray {operation} reduction",
            )
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            auto_candidates=self._redistribution_candidates(plan),
            redistribute_on=redistribute_on,
        )

    # -- first/last -------------------------------------------------------

    def _first_last_local(
        self,
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

        if not self._skipna_enabled(variable.dtype, skipna):
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
        self,
        variable: xr.DataArray,
        dim: Hashable,
        *,
        skipna: bool | None,
        want_first: bool,
    ) -> xr.DataArray:
        """Rank-local first/last, used when ``dim`` is not the partition
        dimension. NaN where nothing was valid, for float/complex dtypes."""
        picked, any_valid = self._first_last_local(
            variable, dim, skipna=skipna, want_first=want_first
        )
        return picked.where(any_valid) if variable.dtype.kind in "fc" else picked

    def _first_last_combine(
        self,
        variable: xr.DataArray,
        dim: Hashable,
        *,
        skipna: bool | None,
        want_first: bool,
    ) -> xr.DataArray:
        """Combine rank-local first/last candidates into a global result.

        Ranks are ordered along ``dim`` by construction (rank 0 owns the
        lowest global indices), so "first/last valid" reduces to "lowest/
        highest rank with any valid data", via two ``Allreduce`` calls:

        1. ``MIN``/``MAX`` elects, per element, the owning rank (a rank
           without valid data reports a sentinel that always loses).
        2. Every rank masks its candidate to zero/``False`` except where it
           is the elected owner; a ``SUM`` (``LOR`` for boolean data) then
           combines the masked candidates, recovering the one nonzero
           contribution per element exactly.

        Elements with no valid data anywhere become NaN for float/complex
        dtypes; other dtypes keep their neutral placeholder, matching how
        :meth:`_combine_extreme` handles the same edge case for min/max."""
        candidate, any_valid = self._first_last_local(
            variable, dim, skipna=skipna, want_first=want_first
        )
        rank, size = self._runtime.comm.rank, self._runtime.comm.size
        sentinel = size if want_first else -1
        owner, error = self._guarded(
            lambda: xr.where(any_valid, rank, sentinel).astype(np.int32)
        )
        owner = self._comm_reduce(
            owner,
            MPI.MIN if want_first else MPI.MAX,
            expect_dtype=np.dtype(np.int32),
            error=error,
            phase="MPI xarray first/last owner election",
        )
        is_owner = owner == rank

        kind = variable.dtype.kind
        neutral = False if kind == "b" else np.zeros((), dtype=variable.dtype).item()
        payload, error = self._guarded(lambda: candidate.where(is_owner, other=neutral))
        combined = self._comm_reduce(
            payload,
            MPI.LOR if kind == "b" else MPI.SUM,
            expect_dtype=variable.dtype,
            error=error,
            phase="MPI xarray first/last value reduction",
        )
        return combined.where(owner != sentinel) if kind in "fc" else combined

    def first(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Select the first valid value along one dimension.

        Unlike the other reductions in this module, ``first``/``last``
        operate on exactly one dimension: they pick a position along it
        rather than collapsing a set of dimensions. ``skipna``/``keep_attrs``
        follow xarray semantics; MPI communication (two ``Allreduce`` calls)
        occurs only when ``dim`` is the active partition dimension."""
        return self._first_or_last(
            value,
            dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            want_first=True,
        )

    def last(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        redistribute_on: Hashable | Literal["auto"] | None = "auto",
    ) -> xr.Dataset | xr.DataArray:
        """Select the last valid value along one dimension. See :meth:`first`."""
        return self._first_or_last(
            value,
            dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            redistribute_on=redistribute_on,
            want_first=False,
        )

    def _first_or_last(
        self,
        value: xr.Dataset | xr.DataArray,
        dim: str,
        *,
        skipna: bool | None,
        keep_attrs: bool | None,
        redistribute_on: Hashable | Literal["auto"] | None,
        want_first: bool,
    ) -> xr.Dataset | xr.DataArray:
        """Shared implementation for :meth:`first` and :meth:`last`."""
        if not isinstance(dim, str):
            raise TypeError("MPI xarray first/last reduce exactly one dimension.")
        dims = (dim,)
        old_meta = get_mpi_meta(value)
        local_meta = self._local_reduction_meta(
            old_meta, dims, redistribute_on=redistribute_on
        )

        if local_meta is not None:
            if isinstance(value, xr.DataArray):
                result = self._first_last_pick(
                    value, dim, skipna=skipna, want_first=want_first
                )
                if keep_attrs:
                    result.attrs.update(value.attrs)
            else:
                result = value.map(
                    self._first_last_pick,
                    dim=dim,
                    skipna=skipna,
                    want_first=want_first,
                    keep_attrs=keep_attrs,
                )
            return self._finish_local_reduction(result, old_meta=local_meta)

        plan = self._plan(
            value, dims, old_meta, operation="first" if want_first else "last"
        )

        if isinstance(value, xr.DataArray):
            result = self._first_last_combine(
                value, dim, skipna=skipna, want_first=want_first
            )
            if keep_attrs:
                result.attrs.update(value.attrs)
            return self._finish(
                result,
                old_meta=old_meta,
                redistribute_on=redistribute_on,
                auto_candidates=self._redistribution_candidates(plan),
            )

        variables: dict[Hashable, xr.DataArray] = {}
        for entry in plan:
            variable = value[entry.name]
            if not entry.dims:
                variables[entry.name] = variable
                continue
            if entry.distributed:
                result = self._first_last_combine(
                    variable, dim, skipna=skipna, want_first=want_first
                )
            else:
                result = self._first_last_pick(
                    variable, dim, skipna=skipna, want_first=want_first
                )
            if keep_attrs:
                result.attrs.update(variable.attrs)
            variables[entry.name] = result
        return self._finish(
            self._dataset_result(value, dims, variables),
            old_meta=old_meta,
            auto_candidates=self._redistribution_candidates(plan),
            redistribute_on=redistribute_on,
        )
