"""Small user-facing MPI namespace built on :mod:`mpi4py`."""

from __future__ import annotations

import builtins
import datetime
import functools
import os
from collections.abc import Callable, Hashable, Iterable, Sequence
from numbers import Integral
from types import EllipsisType
from typing import Any, Literal, ParamSpec, TypeVar, cast

import numpy as np
import xarray as xr
from mpi4py import MPI as _MPI
from mpi4py.MPI import Intracomm
from mpi4py.util import dtlib as _dtlib
from numpy.typing import DTypeLike, NDArray

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# NumPy dtype kinds mpi4py.util.dtlib can translate to a meaningful MPI
# datatype: boolean, unsigned/signed integer, float, complex. Other kinds
# (strings, objects, structured dtypes) produce an opaque byte-derived type
# that is not meaningful for the reductions in this module.
_REDUCIBLE_DTYPE_KINDS = "biufc"

_LAUNCH_ENV = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "SLURM_PROCID",
    "MV2_COMM_WORLD_RANK",
    "I_MPI_COMM_WORLD_RANK",
)


class MPIError(Exception):
    """MPI runtime or synchronized distributed-execution error."""


def mpi_alive(comm: _MPI.Comm) -> bool:
    if comm.Get_size() > 1 or builtins.any(key in os.environ for key in _LAUNCH_ENV):
        return True
    try:
        return _MPI.Comm.Get_parent() != _MPI.COMM_NULL
    except (AttributeError, RuntimeError):
        return False


def is_dataarray(value: Any) -> bool:
    """Check if value is an xarray DataArray."""

    return isinstance(value, xr.DataArray)


def is_dataset(value: Any) -> bool:
    """Check if value is an xarray Dataset."""

    return isinstance(value, xr.Dataset)


def mpi_comm_reduce(
    runtime: MPIRuntime,
    value: T,
    op: _MPI.Op,
    *,
    mode: Literal["all", "root"] = "all",
    root: int = 0,
) -> T | None:
    """Execute a reduction collective for the active communicator.

    Parameters
    ----------
    runtime : _MPIRuntime
        MPI runtime that owns the active communicator.
    value : T
        Python object, NumPy array, or xarray DataArray/Dataset to reduce.
        xarray inputs are reduced element-wise per variable and rewrapped
        with the original dims, coords, and attrs.
    op : mpi4py.MPI.Op
        MPI reduction operator.
    mode : {"all", "root"}, optional
        ``"all"`` selects ``Allreduce``/``allreduce`` and returns the result
        on every rank. ``"root"`` selects ``Reduce``/``reduce`` and returns
        the result only on ``root``. Default is ``"all"``.
    root : int, optional
        Destination rank when ``mode="root"``. Ignored when ``mode="all"``.
        Default is 0.

    Returns
    -------
    T or None
        Reduced value according to ``mode``.
    """
    if mode not in ("all", "root"):
        raise ValueError("mode must be either 'all' or 'root'.")

    comm = runtime.comm
    if mode == "root":
        if isinstance(root, bool) or not isinstance(root, Integral) or root < 0:
            raise ValueError("root must be a non-negative integer rank.")
        if root >= comm.size:
            raise ValueError(f"root {root} is outside [0, {comm.size}).")

    if is_dataset(value):
        reduced_vars = {
            name: mpi_comm_reduce(runtime, da, op, mode=mode, root=root)
            for name, da in value.data_vars.items()
        }
        if mode == "root" and comm.rank != root:
            return None
        return cast("T", value.copy(data=reduced_vars))

    if is_dataarray(value):
        reduced = mpi_comm_reduce(
            runtime, np.asarray(value.values), op, mode=mode, root=root
        )
        if reduced is None:
            return None
        return cast("T", value.copy(data=reduced))

    if not isinstance(value, np.ndarray):
        if mode == "all":
            return cast("T", comm.allreduce(value, op=op))
        return cast("T | None", comm.reduce(value, op=op, root=root))

    send = np.asarray(value)
    if not send.flags.c_contiguous:
        send = np.ascontiguousarray(send)
    if send.dtype.kind not in _REDUCIBLE_DTYPE_KINDS:
        raise MPIError(f"Unsupported MPI NumPy dtype: {send.dtype}.")

    # Pass the buffer-provider array directly rather than the explicit
    # [array, MPI.Datatype] form: mpi4py infers the MPI datatype from the
    # NumPy dtype automatically for buffer-like arguments to fixed-size
    # collectives such as Allreduce/Reduce. See the mpi4py tutorial,
    # "Communication of buffer-like objects". Vector collectives such as
    # scatterv still need the datatype spelled out explicitly; that
    # inference does not extend to them.
    if mode == "all":
        recv = np.empty_like(send)
        comm.Allreduce(send, recv, op=op)
        return cast("T", recv)

    recv = np.empty_like(send) if comm.rank == root else None
    comm.Reduce(send, recv, op=op, root=root)
    return cast("T | None", recv)


class ReduceAccessor:
    """Typed reduction operations for the active communicator.

    Reduction methods select either an all-reduce or a reduce-to-root
    operation through ``mode``. NumPy arrays use mpi4py's uppercase
    buffer collectives, while other Python objects use lowercase object
    collectives.

    Parameters
    ----------
    runtime : _MPIRuntime
        MPI runtime that owns the active communicator.
    """

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime

    def sum(
        self,
        value: T,
        *,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> T | None:
        """Reduce values by summation.

        Parameters
        ----------
        value : T
            Python object or NumPy array to reduce.
        mode : {"all", "root"}, optional
            Reduction mode. ``"all"`` performs ``Allreduce``/``allreduce``
            and returns the result on every rank. ``"root"`` performs
            ``Reduce``/``reduce`` and returns the result only on ``root``.
            Default is ``"all"``.
        root : int, optional
            Destination rank when ``mode="root"``. Ignored when
            ``mode="all"``. Default is 0.

        Returns
        -------
        T or None
            Reduced value on every rank for ``mode="all"``. For
            ``mode="root"``, the reduced value is returned on ``root``
            and None is returned on other ranks.
        """
        return mpi_comm_reduce(self._runtime, value, _MPI.SUM, mode=mode, root=root)

    def prod(
        self,
        value: T,
        *,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> T | None:
        """Reduce values by multiplication.

        Parameters
        ----------
        value : T
            Python object or NumPy array to reduce.
        mode : {"all", "root"}, optional
            Collective mode. ``"all"`` selects ``Allreduce``/``allreduce``
            and returns the result on every rank. ``"root"`` selects
            ``Reduce``/``reduce`` and returns the result only on ``root``.
            Default is ``"all"``.
        root : int, optional
            Destination rank when ``mode="root"``. Ignored when
            ``mode="all"``. Default is 0.

        Returns
        -------
        T or None
            Reduced value according to ``mode``.
        """
        return mpi_comm_reduce(self._runtime, value, _MPI.PROD, mode=mode, root=root)

    def min(
        self,
        value: T,
        *,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> T | None:
        """Reduce values by minimum.

        Parameters
        ----------
        value : T
            Python object or NumPy array to reduce.
        mode : {"all", "root"}, optional
            Collective mode. ``"all"`` selects ``Allreduce``/``allreduce``
            and returns the result on every rank. ``"root"`` selects
            ``Reduce``/``reduce`` and returns the result only on ``root``.
            Default is ``"all"``.
        root : int, optional
            Destination rank when ``mode="root"``. Ignored when
            ``mode="all"``. Default is 0.

        Returns
        -------
        T or None
            Reduced value according to ``mode``.
        """
        return mpi_comm_reduce(self._runtime, value, _MPI.MIN, mode=mode, root=root)

    def max(
        self,
        value: T,
        *,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> T | None:
        """Reduce values by maximum.

        Parameters
        ----------
        value : T
            Python object or NumPy array to reduce.
        mode : {"all", "root"}, optional
            Collective mode. ``"all"`` selects ``Allreduce``/``allreduce``
            and returns the result on every rank. ``"root"`` selects
            ``Reduce``/``reduce`` and returns the result only on ``root``.
            Default is ``"all"``.
        root : int, optional
            Destination rank when ``mode="root"``. Ignored when
            ``mode="all"``. Default is 0.

        Returns
        -------
        T or None
            Reduced value according to ``mode``.
        """
        return mpi_comm_reduce(self._runtime, value, _MPI.MAX, mode=mode, root=root)

    def mean(
        self,
        value: T,
        *,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> Any | None:
        """Reduce values by arithmetic mean across ranks.

        Parameters
        ----------
        value : T
            Python object or NumPy array to average.
        mode : {"all", "root"}, optional
            Collective mode. ``"all"`` selects ``Allreduce``/``allreduce``
            and returns the result on every rank. ``"root"`` selects
            ``Reduce``/``reduce`` and returns the result only on ``root``.
            Default is ``"all"``.
        root : int, optional
            Destination rank when ``mode="root"``. Ignored when
            ``mode="all"``. Default is 0.

        Returns
        -------
        Any or None
            Arithmetic mean according to ``mode``.
        """
        result = self.sum(value, mode=mode, root=root)
        if result is None:
            return None
        return result / self._runtime.comm.size

    def any(
        self,
        value: T,
        *,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> bool | NDArray[np.bool_] | None:
        """Reduce truth values by logical OR.

        Parameters
        ----------
        value : T
            Scalar-like value, NumPy array, or xarray DataArray/Dataset
            converted to Boolean values.
        mode : {"all", "root"}, optional
            Collective mode. ``"all"`` selects ``Allreduce``/``allreduce``
            and returns the result on every rank. ``"root"`` selects
            ``Reduce``/``reduce`` and returns the result only on ``root``.
            Default is ``"all"``.
        root : int, optional
            Destination rank when ``mode="root"``. Ignored when
            ``mode="all"``. Default is 0.

        Returns
        -------
        bool, numpy.ndarray, xarray.DataArray, xarray.Dataset, or None
            Logical-OR reduction according to ``mode``.
        """
        boolean_value = (
            value.astype(bool)
            if is_dataset(value) or is_dataarray(value)
            else np.asarray(value, dtype=bool)
        )
        result = mpi_comm_reduce(
            self._runtime, boolean_value, _MPI.LOR, mode=mode, root=root
        )
        if np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result

    def all(
        self,
        value: T,
        *,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> bool | NDArray[np.bool_] | None:
        """Reduce truth values by logical AND.

        Parameters
        ----------
        value : T
            Scalar-like value, NumPy array, or xarray DataArray/Dataset
            converted to Boolean values.
        mode : {"all", "root"}, optional
            Collective mode. ``"all"`` selects ``Allreduce``/``allreduce``
            and returns the result on every rank. ``"root"`` selects
            ``Reduce``/``reduce`` and returns the result only on ``root``.
            Default is ``"all"``.
        root : int, optional
            Destination rank when ``mode="root"``. Ignored when
            ``mode="all"``. Default is 0.

        Returns
        -------
        bool, numpy.ndarray, xarray.DataArray, xarray.Dataset, or None
            Logical-AND reduction according to ``mode``.
        """
        boolean_value = (
            value.astype(bool)
            if is_dataset(value) or is_dataarray(value)
            else np.asarray(value, dtype=bool)
        )
        result = mpi_comm_reduce(
            self._runtime, boolean_value, _MPI.LAND, mode=mode, root=root
        )
        if np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result


class XarrayReduceAccessor:
    """Distributed reductions with xarray-style dimension semantics.

    Each method first performs the named-dimension reduction locally with
    xarray, then combines the resulting partial reductions across the active
    MPI communicator. Dimensions removed by ``dim`` are therefore reduced
    across both the local xarray object and the MPI partitioning.

    The distributed dimension or dimensions must be included in ``dim``.
    Dimensions retained in the result must have matching shapes, coordinate
    values, and ordering on every rank. Variables that do not contain a
    reduced dimension are treated as replicated and remain unchanged. The
    MPI combination is limited to buffer dtypes supported by this module;
    complex minimum/maximum and nonnumeric extrema are not supported.

    Parameters
    ----------
    runtime : MPIRuntime
        MPI runtime that owns the active communicator.
    """

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime

    def _validate_collective(
        self,
        mode: Literal["all", "root"],
        root: int,
    ) -> None:
        if mode not in ("all", "root"):
            raise ValueError("mode must be either 'all' or 'root'.")
        if mode == "root":
            if isinstance(root, bool) or not isinstance(root, Integral) or root < 0:
                raise ValueError("root must be a non-negative integer rank.")
            if root >= self._runtime.comm.size:
                raise ValueError(
                    f"root {root} is outside [0, {self._runtime.comm.size})."
                )

    @staticmethod
    def _normalize_dim(
        value: xr.DataArray | xr.Dataset,
        dim: str,
    ) -> tuple[str, tuple[Hashable, ...]]:
        if not isinstance(value, (xr.DataArray, xr.Dataset)):
            raise TypeError("xreduce requires an xarray DataArray or Dataset.")
        if dim is None or dim is ...:
            return dim, tuple(value.dims)
        if isinstance(dim, str):
            return dim, (dim,)
        dims = tuple(dim)
        return dims, dims

    @staticmethod
    def _variable_dims(
        value: xr.DataArray,
        dims: tuple[Hashable, ...],
    ) -> tuple[Hashable, ...]:
        return tuple(dim for dim in dims if dim in value.dims)

    @staticmethod
    def _skipna_enabled(dtype: np.dtype[Any], skipna: bool | None) -> bool:
        if skipna is not None:
            return skipna
        return dtype.kind in "fc"

    @staticmethod
    def _mean_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
        sample = np.zeros(1, dtype=dtype)
        return np.asarray(np.mean(sample)).dtype

    def _local_result(
        self,
        value: xr.DataArray | xr.Dataset,
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | xr.Dataset | None:
        if mode == "root" and self._runtime.comm.rank != root:
            return None
        return value

    def _dataset_result(
        self,
        local: xr.Dataset,
        updates: dict[str, xr.DataArray],
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.Dataset | None:
        if mode == "root" and self._runtime.comm.rank != root:
            return None
        data = {
            name: updates[name] if name in updates else local[name]
            for name in local.data_vars
        }
        return local.copy(data=data)

    def _count(
        self,
        value: xr.DataArray,
        dims: tuple[Hashable, ...],
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        local_count = value.count(dim=dims, keep_attrs=False)
        return mpi_comm_reduce(
            self._runtime,
            local_count,
            _MPI.SUM,
            mode=mode,
            root=root,
        )

    def _combine_sum_or_prod(
        self,
        value: xr.DataArray,
        partial: xr.DataArray,
        dims: tuple[Hashable, ...],
        op: _MPI.Op,
        *,
        skipna: bool | None,
        min_count: int | None,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        result = mpi_comm_reduce(
            self._runtime,
            partial,
            op,
            mode=mode,
            root=root,
        )
        global_count = None
        if min_count is not None and self._skipna_enabled(value.dtype, skipna):
            global_count = self._count(value, dims, mode=mode, root=root)

        if result is None:
            return None
        if global_count is not None:
            result = result.where(global_count >= min_count)
        return result

    def _combine_mean(
        self,
        value: xr.DataArray,
        partial_sum: xr.DataArray,
        dims: tuple[Hashable, ...],
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        global_sum = mpi_comm_reduce(
            self._runtime,
            partial_sum,
            _MPI.SUM,
            mode=mode,
            root=root,
        )
        global_count = self._count(value, dims, mode=mode, root=root)
        if global_sum is None or global_count is None:
            return None

        with np.errstate(divide="ignore", invalid="ignore"):
            result = global_sum / global_count
        result = result.where(global_count != 0)
        return result.astype(self._mean_dtype(value.dtype), keep_attrs=True)

    def _combine_extreme(
        self,
        value: xr.DataArray,
        partial: xr.DataArray,
        dims: tuple[Hashable, ...],
        *,
        minimum: bool,
        skipna: bool | None,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        kind = partial.dtype.kind
        if kind == "c":
            name = "minimum" if minimum else "maximum"
            raise MPIError(f"MPI {name} is not defined for complex xarray data.")
        if kind not in "biuf":
            raise MPIError(f"Unsupported MPI xarray dtype: {partial.dtype}.")

        if kind == "b":
            op = _MPI.LAND if minimum else _MPI.LOR
            return mpi_comm_reduce(
                self._runtime,
                partial,
                op,
                mode=mode,
                root=root,
            )

        op = _MPI.MIN if minimum else _MPI.MAX
        if kind != "f":
            return mpi_comm_reduce(
                self._runtime,
                partial,
                op,
                mode=mode,
                root=root,
            )

        identity = np.asarray(
            np.inf if minimum else -np.inf,
            dtype=partial.dtype,
        ).item()
        if self._skipna_enabled(value.dtype, skipna):
            local_mask = value.count(dim=dims, keep_attrs=False) > 0
            safe_partial = partial.where(local_mask, other=identity)
            mask_op = _MPI.LOR
        else:
            local_mask = value.isnull().any(dim=dims, keep_attrs=False)
            safe_partial = partial.where(~local_mask, other=identity)
            mask_op = _MPI.LOR

        result = mpi_comm_reduce(
            self._runtime,
            safe_partial,
            op,
            mode=mode,
            root=root,
        )
        global_mask = mpi_comm_reduce(
            self._runtime,
            local_mask,
            mask_op,
            mode=mode,
            root=root,
        )
        if result is None or global_mask is None:
            return None
        if self._skipna_enabled(value.dtype, skipna):
            return result.where(global_mask)
        return result.where(~global_mask)

    def _combine_logical(
        self,
        partial: xr.DataArray,
        op: _MPI.Op,
        *,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | None:
        return mpi_comm_reduce(
            self._runtime,
            partial,
            op,
            mode=mode,
            root=root,
        )

    def sum(
        self,
        value: xr.DataArray | xr.Dataset,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.DataArray | xr.Dataset | None:
        """Reduce an xarray object by distributed summation."""
        self._validate_collective(mode, root)
        local_dim, dims = self._normalize_dim(value, dim)
        local = value.sum(
            dim=local_dim,
            skipna=skipna,
            min_count=None,
            keep_attrs=keep_attrs,
        )

        if isinstance(value, xr.DataArray):
            if not dims:
                return self._local_result(local, mode=mode, root=root)
            return self._combine_sum_or_prod(
                value,
                local,
                dims,
                _MPI.SUM,
                skipna=skipna,
                min_count=min_count,
                mode=mode,
                root=root,
            )

        updates: dict[str, xr.DataArray] = {}
        for name in local.data_vars:
            variable = value[name]
            variable_dims = self._variable_dims(variable, dims)
            if not variable_dims:
                continue
            result = self._combine_sum_or_prod(
                variable,
                local[name],
                variable_dims,
                _MPI.SUM,
                skipna=skipna,
                min_count=min_count,
                mode=mode,
                root=root,
            )
            if result is not None:
                updates[name] = result
        return self._dataset_result(local, updates, mode=mode, root=root)

    def prod(
        self,
        value: xr.DataArray | xr.Dataset,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.DataArray | xr.Dataset | None:
        """Reduce an xarray object by distributed multiplication."""
        self._validate_collective(mode, root)
        local_dim, dims = self._normalize_dim(value, dim)
        local = value.prod(
            dim=local_dim,
            skipna=skipna,
            min_count=None,
            keep_attrs=keep_attrs,
        )

        if isinstance(value, xr.DataArray):
            if not dims:
                return self._local_result(local, mode=mode, root=root)
            return self._combine_sum_or_prod(
                value,
                local,
                dims,
                _MPI.PROD,
                skipna=skipna,
                min_count=min_count,
                mode=mode,
                root=root,
            )

        updates: dict[str, xr.DataArray] = {}
        for name in local.data_vars:
            variable = value[name]
            variable_dims = self._variable_dims(variable, dims)
            if not variable_dims:
                continue
            result = self._combine_sum_or_prod(
                variable,
                local[name],
                variable_dims,
                _MPI.PROD,
                skipna=skipna,
                min_count=min_count,
                mode=mode,
                root=root,
            )
            if result is not None:
                updates[name] = result
        return self._dataset_result(local, updates, mode=mode, root=root)

    def min(
        self,
        value: xr.DataArray | xr.Dataset,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.DataArray | xr.Dataset | None:
        """Reduce an xarray object by distributed minimum."""
        return self._extreme(
            value,
            dim,
            minimum=True,
            skipna=skipna,
            keep_attrs=keep_attrs,
            mode=mode,
            root=root,
        )

    def max(
        self,
        value: xr.DataArray | xr.Dataset,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.DataArray | xr.Dataset | None:
        """Reduce an xarray object by distributed maximum."""
        return self._extreme(
            value,
            dim,
            minimum=False,
            skipna=skipna,
            keep_attrs=keep_attrs,
            mode=mode,
            root=root,
        )

    def _extreme(
        self,
        value: xr.DataArray | xr.Dataset,
        dim: str,
        *,
        minimum: bool,
        skipna: bool | None,
        keep_attrs: bool | None,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | xr.Dataset | None:
        self._validate_collective(mode, root)
        local_dim, dims = self._normalize_dim(value, dim)
        method = value.min if minimum else value.max
        local = method(
            dim=local_dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
        )

        if isinstance(value, xr.DataArray):
            if not dims:
                return self._local_result(local, mode=mode, root=root)
            return self._combine_extreme(
                value,
                local,
                dims,
                minimum=minimum,
                skipna=skipna,
                mode=mode,
                root=root,
            )

        updates: dict[str, xr.DataArray] = {}
        for name in local.data_vars:
            variable = value[name]
            variable_dims = self._variable_dims(variable, dims)
            if not variable_dims:
                continue
            result = self._combine_extreme(
                variable,
                local[name],
                variable_dims,
                minimum=minimum,
                skipna=skipna,
                mode=mode,
                root=root,
            )
            if result is not None:
                updates[name] = result
        return self._dataset_result(local, updates, mode=mode, root=root)

    def mean(
        self,
        value: xr.DataArray | xr.Dataset,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.DataArray | xr.Dataset | None:
        """Reduce an xarray object by distributed arithmetic mean."""
        self._validate_collective(mode, root)
        local_dim, dims = self._normalize_dim(value, dim)
        local_sum = value.sum(
            dim=local_dim,
            skipna=skipna,
            min_count=None,
            keep_attrs=keep_attrs,
        )

        if isinstance(value, xr.DataArray):
            if not dims:
                local_mean = value.mean(
                    dim=local_dim,
                    skipna=skipna,
                    keep_attrs=keep_attrs,
                )
                return self._local_result(local_mean, mode=mode, root=root)
            return self._combine_mean(
                value,
                local_sum,
                dims,
                mode=mode,
                root=root,
            )

        updates: dict[str, xr.DataArray] = {}
        for name in local_sum.data_vars:
            variable = value[name]
            variable_dims = self._variable_dims(variable, dims)
            if not variable_dims:
                continue
            result = self._combine_mean(
                variable,
                local_sum[name],
                variable_dims,
                mode=mode,
                root=root,
            )
            if result is not None:
                updates[name] = result

        return self._dataset_result(local_sum, updates, mode=mode, root=root)

    def any(
        self,
        value: xr.DataArray | xr.Dataset,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.DataArray | xr.Dataset | None:
        """Reduce an xarray object by distributed logical OR."""
        return self._logical(
            value,
            dim,
            op=_MPI.LOR,
            all_values=False,
            keep_attrs=keep_attrs,
            mode=mode,
            root=root,
        )

    def all(
        self,
        value: xr.DataArray | xr.Dataset,
        dim: str | Iterable[Hashable] | EllipsisType | None = None,
        *,
        keep_attrs: bool | None = None,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> xr.DataArray | xr.Dataset | None:
        """Reduce an xarray object by distributed logical AND."""
        return self._logical(
            value,
            dim,
            op=_MPI.LAND,
            all_values=True,
            keep_attrs=keep_attrs,
            mode=mode,
            root=root,
        )

    def _logical(
        self,
        value: xr.DataArray | xr.Dataset,
        dim: str,
        *,
        op: _MPI.Op,
        all_values: bool,
        keep_attrs: bool | None,
        mode: Literal["all", "root"],
        root: int,
    ) -> xr.DataArray | xr.Dataset | None:
        self._validate_collective(mode, root)
        local_dim, dims = self._normalize_dim(value, dim)
        method = value.all if all_values else value.any
        local = method(dim=local_dim, keep_attrs=keep_attrs)

        if isinstance(value, xr.DataArray):
            if not dims:
                return self._local_result(local, mode=mode, root=root)
            return self._combine_logical(local, op, mode=mode, root=root)

        updates: dict[str, xr.DataArray] = {}
        for name in local.data_vars:
            variable_dims = self._variable_dims(value[name], dims)
            if not variable_dims:
                continue
            result = self._combine_logical(
                local[name],
                op,
                mode=mode,
                root=root,
            )
            if result is not None:
                updates[name] = result
        return self._dataset_result(local, updates, mode=mode, root=root)


class MPIRuntime:
    """User-facing MPI runtime namespace.

    The runtime owns one intracommunicator and exposes it directly through
    :attr:`comm`, preserving the native :class:`mpi4py.MPI.Intracomm` type,
    method signatures, IDE completion, and third-party interoperability.
    Direct MPI reductions are grouped under :attr:`reduce`, while distributed
    xarray-style dimension reductions are grouped under :attr:`xreduce`.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm or None, optional
        Intracommunicator used by the runtime. If None, use
        ``MPI.COMM_WORLD``.

    Attributes
    ----------
    MPI : module
        The :mod:`mpi4py.MPI` module for MPI constants and object types.
    MPIError : type[MPIError]
        Exception type used for synchronized MPI failures.
    comm : mpi4py.MPI.Intracomm
        Native intracommunicator used by the runtime.
    reduce : ReduceAccessor
        Direct element-wise reductions across MPI ranks.
    xreduce : XarrayReduceAccessor
        Distributed xarray-style reductions over named dimensions.
    """

    MPI = _MPI
    MPIError = MPIError

    def __init__(self, comm: Intracomm | None = None) -> None:
        self.comm: Intracomm = comm if comm is not None else _MPI.COMM_WORLD
        self._reduce: ReduceAccessor = ReduceAccessor(self)
        self._xreduce: XarrayReduceAccessor = XarrayReduceAccessor(self)

    @property
    def launched(self) -> bool:
        """Return whether this process appears to have been launched by MPI.

        Returns
        -------
        bool
            True when an MPI or Slurm launch environment is detected.
        """
        return mpi_alive(self.comm)

    def is_root(self, root: int = 0) -> bool:
        """Return whether this process has the requested root rank.

        Parameters
        ----------
        root : int, optional
            Root rank to compare against. Default is 0.

        Returns
        -------
        bool
            True when the current rank equals ``root``.
        """
        return self.comm.rank == root

    def log(
        self,
        message: str,
        *args: Any,
        root: int = 0,
        timestamp: bool = False,
        prefix: bool = False,
        logger: Callable[..., None] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Emit a message from a specific MPI rank.

        Parameters
        ----------
        message : str
            Message or format string to be logged.
        *args : Any
            Positional arguments passed to ``logger``. If ``logger`` is None,
            these trigger percent-formatting of ``message`` before printing.
        root : int, optional
            Rank allowed to emit the message. Default is 0.
        timestamp : bool, optional
            If True, prepends a standard ISO-like timestamp to the message.
            Only applies when falling back to the built-in print. Default is False.
        prefix : bool, optional
            If True, prepends an MPI rank indicator (e.g., "[MPI]") to the message.
            This flag only toggles the prefix for the default :func:`print`.
            Custom loggers will always receive the prefix. Default is False.
        logger : callable, optional
            Callable used to emit the message. Default is None, which falls back
            to the built-in :func:`print`.
        **kwargs : Any
            Keyword arguments forwarded to the ``logger`` (or :func:`print`).

        Returns
        -------
        None
        """
        if not self.is_root(root):
            return

        # Generate the MPI string once
        mpi_str = f"[MPI RANK {root}]" if root != 0 else "[MPI]"

        if logger is None:
            # Apply string formatting if args exist
            if args:
                message = message % args

            # Build the prefix dynamically based on boolean flags for print
            msg_prefix = ""
            if timestamp:
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg_prefix += f"{time_str} - "

            if prefix:
                msg_prefix += f"{mpi_str} "
            # Print the final assembled string
            print(f"{msg_prefix}{message}", **kwargs)

        else:
            # MPI prefix goes to the custom logger without checking the flag
            if prefix:
                message = f"{mpi_str} {message}"
            logger(message, *args, **kwargs)

    def scatterv(
        self,
        array: NDArray[Any] | None,
        counts: Sequence[int],
        recv_shape: Sequence[int],
        dtype: DTypeLike,
        *,
        root: int = 0,
    ) -> NDArray[Any]:
        """Scatter unequal contiguous leading-axis slabs from one rank.

        This is a NumPy convenience wrapper around :meth:`MPI.Comm.Scatterv`.
        Use ``mpi.comm.Scatterv`` directly for the complete mpi4py buffer API.

        Parameters
        ----------
        array : numpy.ndarray or None
            Source array on ``root``. Non-root ranks may pass None.
        counts : sequence of int
            Number of leading-axis rows sent to each rank. The sequence must
            contain exactly ``mpi.comm.size`` entries.
        recv_shape : sequence of int
            Shape of the local receive array on this rank.
        dtype : numpy.dtype or type
            NumPy dtype of the send and receive arrays.
        root : int, optional
            Rank that owns ``array``. Default is 0.

        Returns
        -------
        numpy.ndarray
            Contiguous local slab received by this rank.

        Raises
        ------
        ValueError
            If ``counts`` does not contain one entry per rank, or if ``array``
            is None on the root rank.
        MPIError
            If ``dtype`` has no supported MPI datatype mapping.
        """
        counts_array = np.asarray(counts, dtype=np.int64)
        if counts_array.shape != (self.comm.size,):
            raise ValueError(f"counts must contain {self.comm.size} values.")

        shape = tuple(int(length) for length in recv_shape)
        recv = np.empty(shape, dtype=dtype)
        row_size = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        element_counts = counts_array * row_size
        offsets = np.zeros(self.comm.size, dtype=np.int64)
        offsets[1:] = np.cumsum(element_counts[:-1])
        mpi_type = self.datatype(np.dtype(dtype))

        send: Any = None
        if self.is_root(root):
            if array is None:
                raise ValueError("array cannot be None on the scatter root.")
            send = [
                np.ascontiguousarray(array, dtype=dtype),
                element_counts,
                offsets,
                mpi_type,
            ]

        self.comm.Scatterv(send, [recv, mpi_type], root=root)
        return recv

    @property
    def reduce(self) -> ReduceAccessor:
        """Return direct MPI reduction operations.

        Returns
        -------
        ReduceAccessor
            Element-wise cross-rank ``sum``, ``prod``, ``min``, ``max``,
            ``mean``, ``any``, and ``all`` operations.
        """
        return self._reduce

    @property
    def xreduce(self) -> XarrayReduceAccessor:
        """Return distributed xarray-style reduction operations.

        Returns
        -------
        XarrayReduceAccessor
            Named-dimension ``sum``, ``prod``, ``min``, ``max``, ``mean``,
            ``any``, and ``all`` operations.
        """
        return self._xreduce

    def datatype(self, dtype: DTypeLike) -> _MPI.Datatype:
        """Return the MPI datatype corresponding to a NumPy dtype.

        Backed by :func:`mpi4py.util.dtlib.from_numpy_dtype`, the datatype
        conversion mpi4py itself maintains, rather than a hand-kept mapping.

        Raises
        ------
        MPIError
            If ``dtype`` is not boolean, integer, float, or complex; other
            kinds (strings, objects, structured dtypes) are not meaningful
            for the reductions in this module even though ``dtlib`` itself
            would still produce an opaque derived type for them.
        """
        key = np.dtype(dtype)
        if key.kind not in _REDUCIBLE_DTYPE_KINDS:
            raise MPIError(f"Unsupported MPI NumPy dtype: {key}.")
        try:
            return _dtlib.from_numpy_dtype(key)
        except (KeyError, ValueError) as exc:
            raise MPIError(f"Unsupported MPI NumPy dtype: {key}.") from exc

    def raise_if_error(self, error: BaseException | None, phase: str) -> None:
        """Raise a synchronized error on all ranks if any rank failed."""
        failed = self.comm.allgather(error is not None)
        if not builtins.any(failed):
            return

        failed_ranks = [index for index, state in enumerate(failed) if state]
        first = failed_ranks[0]
        if error is not None and len(failed_ranks) == self.comm.size:
            raise error

        detail = None
        if self.comm.rank == first and error is not None:
            detail = (type(error).__name__, str(error))
        detail = self.comm.bcast(detail, root=first)
        if detail is None:
            raise MPIError(f"Rank {first} failed during {phase}.")
        name, message = detail
        raise MPIError(f"Rank {first} failed during {phase} with {name}: {message}")

    def __call__(
        self,
        function: Callable[P, R] | None = None,
        /,
        *,
        all_ranks: bool = False,
        broadcast: bool = False,
        root: int = 0,
    ) -> (
        Callable[P, R]
        | Callable[P, R | None]
        | Callable[[Callable[P, R]], Callable[P, R]]
        | Callable[[Callable[P, R]], Callable[P, R | None]]
    ):
        """Decorate a function for MPI-aware execution.

        By default, the decorated function executes only on the designated
        root rank. It can instead execute on all ranks, or execute on ``root``
        and broadcast its return value to every rank.

        Parameters
        ----------
        function : callable or None, optional
            Function to decorate. Passed positionally when using ``@mpi``.
            None supports decorator use with keyword arguments.
        all_ranks : bool, optional
            If True, execute the function on every rank. Default is False.
        broadcast : bool, optional
            If True, execute on ``root`` and broadcast its return value to
            every rank. Default is False.
        root : int, optional
            Root rank used for root-only execution and broadcasting. Default
            is 0.

        Returns
        -------
        callable
            Decorated function, or a decorator closure when ``function`` is
            None. Root-only execution returns None on non-root ranks; broadcast
            mode returns the root result on every rank.

        Raises
        ------
        TypeError
            If the positional argument is not callable.
        ValueError
            If ``broadcast`` and ``all_ranks`` are both True, or if ``root`` is
            not a non-negative integer rank.
        MPIError
            If distributed execution fails on only a subset of ranks.

        Examples
        --------
        Run a function on the root rank only.

        >>> @mpi
        ... def compute_metrics():
        ...     pass

        Run a function on every rank.

        >>> @mpi(all_ranks=True)
        ... def initialize_worker():
        ...     pass

        Run a function on the root rank and broadcast the result.

        >>> @mpi(broadcast=True)
        ... def load_shared_configuration():
        ...     return {"learning_rate": 0.01}
        """
        if function is None:
            return functools.partial(
                self,
                all_ranks=all_ranks,
                broadcast=broadcast,
                root=root,
            )
        if not callable(function):
            raise TypeError("mpi's positional argument must be callable.")
        if broadcast and all_ranks:
            raise ValueError("broadcast=True is incompatible with all_ranks=True.")
        if isinstance(root, bool) or not isinstance(root, Integral) or root < 0:
            raise ValueError("root must be a non-negative integer rank.")

        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
            if root >= self.comm.size:
                raise ValueError(f"root {root} is outside [0, {self.comm.size}).")

            result: R | None = None
            error: BaseException | None = None
            if all_ranks or self.is_root(root):
                try:
                    result = function(*args, **kwargs)
                except BaseException as exc:
                    error = exc

            self.raise_if_error(error, function.__name__)
            if broadcast:
                return cast("R", self.comm.bcast(result, root=root))
            return result

        wrapper.mpi = True  # type: ignore[attr-defined]
        return wrapper


mpi: MPIRuntime = MPIRuntime()

__all__ = ["mpi"]
