"""Reduction planning and MPI collective primitives.

Shared machinery used by every reduction mixin (:mod:`.reductions`,
:mod:`.statistics`, :mod:`.groupby`): building a rank-independent
:class:`~.common.PlanEntry` plan per variable, verifying every rank agrees
on that plan before posting a collective, running the low-level ``Allreduce``
exchange, and restoring ``mpi_meta`` once a reduction finishes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Iterable, Mapping
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr
from mpi4py import MPI

from .chunks import prune_chunk_info
from .common import (
    _MPI_REDUCIBLE_KINDS,
    CHECK_COLLECTIVE_AGREEMENT,
    PlanEntry,
    _mpi_representable,
    _op_name,
    _partial_dtype,
)
from .meta import choose_partition_dim, set_mpi_meta, strip_mpi_meta

if TYPE_CHECKING:
    from ..mpi.runtime import MPIRuntime


class ReductionPlanningMixin:
    """Rank-independent reduction planning and MPI collective primitives.

    Requires a ``self._runtime`` attribute set by :class:`~.mpi.XarrayMPI`.
    Concrete reductions (:class:`~.reductions.ReductionMixin`,
    :class:`~.statistics.StatisticsMixin`, ...) build on these methods; this
    class performs no reduction itself.
    """

    _runtime: MPIRuntime

    # -- collective planning -------------------------------------------------

    @staticmethod
    def _normalize_dim(
        value: xr.Dataset | xr.DataArray,
        dim: str | Iterable[Hashable] | EllipsisType | None,
    ) -> tuple[Any, tuple[Hashable, ...]]:
        """Normalize a reduction dimension specification."""
        if not isinstance(value, (xr.DataArray, xr.Dataset)):
            raise TypeError(
                "MPI xarray operations require an xarray DataArray or Dataset."
            )
        if dim is None or dim is ...:
            return dim, tuple(value.dims)
        if isinstance(dim, str):
            return dim, (dim,)
        dims = tuple(dim)
        return dims, dims

    @staticmethod
    def _skipna_enabled(dtype: np.dtype[Any], skipna: bool | None) -> bool:
        """Return the effective dtype-aware ``skipna`` setting."""
        if skipna is not None:
            return skipna
        return dtype.kind in "fc"

    @staticmethod
    def _check_reducible(dtype: np.dtype[Any], operation: str) -> None:
        """Validate that a dtype supports the requested MPI reduction."""
        if operation in ("any", "all"):
            return
        if dtype.kind not in _MPI_REDUCIBLE_KINDS:
            raise TypeError(f"Unsupported MPI xarray dtype: {dtype}.")
        if not _mpi_representable(dtype.str):
            # float16 and long double have a reducible NumPy kind but no
            # predefined MPI datatype. Rejecting them here raises on every
            # rank before any collective, instead of failing inside
            # Allreduce with MPI_ERR_TYPE once buffers are already posted.
            raise TypeError(
                f"Unsupported MPI xarray dtype: {dtype}. "
                + "No predefined MPI datatype represents it."
            )
        if operation in ("min", "max") and dtype.kind == "c":
            name = "minimum" if operation == "min" else "maximum"
            raise TypeError(f"MPI {name} is not defined for complex xarray data.")

    @staticmethod
    def _local_reduction_meta(
        meta: Mapping[str, Any] | None,
        dims: tuple[Hashable, ...],
        *,
        partition_dim: Hashable | Literal["auto"] | None,
    ) -> Mapping[str, Any] | None:
        """Return metadata when a reduction remains rank-local."""
        if meta is None or meta["dim"] in dims:
            return None
        if partition_dim not in (None, "auto"):
            raise ValueError(
                "partition_dim can name a new dimension only after the active "
                + "partition dimension has been reduced away."
            )
        return meta

    @staticmethod
    def _finish_local_reduction(
        result: xr.Dataset | xr.DataArray, *, old_meta: Mapping[str, Any]
    ) -> xr.Dataset | xr.DataArray:
        """Restore metadata after a rank-local reduction."""
        partition_dim = old_meta["dim"]
        if partition_dim not in result.dims:
            return strip_mpi_meta(result)
        set_mpi_meta(
            result,
            dim=partition_dim,
            global_size=int(old_meta["global_size"]),
            start=int(old_meta["start"]),
            stop=int(old_meta["stop"]),
            chunk_info=prune_chunk_info(old_meta["chunk_info"], result),
        )
        return result

    def _agree(self, signature: tuple[Any, ...]) -> None:
        """Verify that all ranks entered the same reduction plan."""
        if not CHECK_COLLECTIVE_AGREEMENT or self._runtime.comm.size == 1:
            return
        digest = hashlib.blake2b(repr(signature).encode(), digest_size=16).digest()
        self._runtime.raise_if_error(
            None,
            "MPI xarray reduction planning",
            signature=("xarray_reduction_plan", digest),
        )

    def _plan(
        self,
        value: xr.Dataset | xr.DataArray,
        dims: tuple[Hashable, ...],
        meta: Mapping[str, Any] | None,
        *,
        operation: str,
    ) -> tuple[PlanEntry, ...]:
        """Build and validate the rank-independent reduction plan."""
        if isinstance(value, xr.DataArray):
            items: tuple[tuple[Hashable, xr.DataArray], ...] = ((value.name, value),)
        else:
            items = tuple((name, value[name]) for name in value.data_vars)

        entries = []
        for name, variable in items:
            variable_dims = tuple(dim for dim in dims if dim in variable.dims)
            if variable_dims:
                self._check_reducible(variable.dtype, operation)
            entries.append(
                PlanEntry(
                    name=name,
                    dims=variable_dims,
                    distributed=meta is not None and meta["dim"] in variable.dims,
                    dtype=variable.dtype,
                    shape=tuple(
                        (str(dim), int(value.sizes[dim]))
                        for dim in variable.dims
                        if dim not in variable_dims
                    ),
                )
            )

        plan = tuple(entries)
        self._agree(
            (
                operation,
                tuple(str(dim) for dim in dims),
                tuple(
                    (
                        str(entry.name),
                        tuple(str(dim) for dim in entry.dims),
                        entry.distributed,
                        str(entry.dtype),
                        entry.shape,
                    )
                    for entry in plan
                ),
            )
        )
        return plan

    @staticmethod
    def _guarded(function: Any) -> tuple[Any, BaseException | None]:
        """Run a local operation and defer any exception for synchronization."""
        try:
            return function(), None
        except BaseException as exc:
            return None, exc

    # -- collective primitives -----------------------------------------------

    def _comm_reduce(
        self,
        value: xr.DataArray | None,
        op: MPI.Op,
        *,
        expect_dtype: np.dtype[Any] | None = None,
        error: BaseException | None = None,
        phase: str = "MPI xarray reduction buffer preparation",
    ) -> xr.DataArray:
        """Combine a validated DataArray buffer across ranks."""
        send: np.ndarray[Any, Any] | None = None
        if error is None:
            try:
                if value is None:
                    raise AssertionError("MPI xarray reduction buffer is missing.")
                send = np.asarray(value.values)
                if expect_dtype is not None and send.dtype != np.dtype(expect_dtype):
                    send = send.astype(expect_dtype)
                if not send.flags.c_contiguous:
                    send = np.ascontiguousarray(send)
                if send.dtype.kind not in _MPI_REDUCIBLE_KINDS:
                    raise TypeError(f"Unsupported MPI xarray dtype: {send.dtype}.")
                if not _mpi_representable(send.dtype.str):
                    raise TypeError(
                        f"Unsupported MPI xarray dtype: {send.dtype}. "
                        + "No predefined MPI datatype represents it."
                    )
            except BaseException as exc:
                error = exc
                send = None

        signature = (
            None
            if send is None
            else (
                _op_name(op),
                send.dtype.str,
                tuple(int(length) for length in send.shape),
            )
        )
        self._runtime.raise_if_error(error, phase, signature)
        if send is None or value is None:
            raise AssertionError("MPI xarray reduction buffer is missing.")

        recv = self._exchange(send, op)
        return value.copy(data=recv)

    def _exchange(self, send: np.ndarray[Any, Any], op: MPI.Op) -> np.ndarray[Any, Any]:
        """All-reduce a validated contiguous NumPy buffer."""
        recv = np.empty(send.shape, dtype=send.dtype)
        self._runtime.comm.Allreduce(send, recv, op=op)
        return recv

    def _count(self, value: xr.DataArray, dims: tuple[Hashable, ...]) -> xr.DataArray:
        """Count valid values globally across the requested dimensions."""
        count: xr.DataArray | None = None
        error: BaseException | None = None
        try:
            count = value.count(dim=dims, keep_attrs=False)
        except BaseException as exc:
            error = exc
        return self._comm_reduce(
            count,
            MPI.SUM,
            expect_dtype=_partial_dtype(value.dtype.str, "count", None),
            error=error,
            phase="MPI xarray count reduction",
        )

    @staticmethod
    def _dataset_result(
        value: xr.Dataset,
        dims: tuple[Hashable, ...],
        variables: Mapping[Hashable, xr.DataArray],
    ) -> xr.Dataset:
        """Rebuild a Dataset from reduced data variables."""
        reduced = set(dims)
        coords = {
            name: coord
            for name, coord in value.coords.items()
            if not reduced & set(coord.dims)
        }
        return xr.Dataset(dict(variables), coords=coords, attrs=dict(value.attrs))

    @staticmethod
    def _repartition_candidates(plan: tuple[PlanEntry, ...]) -> frozenset[Hashable]:
        """Return dimensions eligible for post-reduction repartition."""
        return frozenset(
            dim for entry in plan if entry.distributed for dim, _ in entry.shape
        )

    def _finish(
        self,
        result: xr.Dataset | xr.DataArray,
        *,
        old_meta: Mapping[str, Any] | None,
        partition_dim: Hashable | Literal["auto"] | None,
        auto_candidates: frozenset[Hashable],
    ) -> xr.Dataset | xr.DataArray:
        """Finalize metadata and optional repartition after a reduction."""
        result = strip_mpi_meta(result)
        partition_removed = old_meta is not None and old_meta["dim"] not in result.dims

        if partition_dim is None:
            return result

        target = partition_dim
        if partition_dim == "auto":
            if not partition_removed:
                return result
            sizes = {
                dim: length
                for dim, length in result.sizes.items()
                if dim in auto_candidates
            }
            if not any(int(length) > 1 for length in sizes.values()):
                return result
            target = choose_partition_dim(
                sizes, self._runtime.comm.size, rank=self._runtime.comm.rank
            )
        elif partition_dim not in auto_candidates:
            raise ValueError(
                f"partition_dim={partition_dim!r} is not a dimension of any "
                + "variable that required an MPI collective in this reduction; "
                + "an untouched, replicated variable's own dimension cannot be "
                + "used as the new partition dimension."
            )

        chunk_info = (
            prune_chunk_info(old_meta["chunk_info"], result)
            if old_meta is not None
            else {}
        )
        return self.repartition(result, target, chunk_info=chunk_info)
