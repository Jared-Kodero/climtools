"""Small user-facing MPI namespace built on :mod:`mpi4py`."""

# lib_mpi.py
from __future__ import annotations

import builtins
import datetime
import faulthandler
import functools
import os
import sys
import threading
import time
from collections.abc import Callable, Generator, Hashable, Sequence
from contextlib import contextmanager
from numbers import Integral
from typing import Any, Literal, ParamSpec, TypeVar, cast

import numpy as np
import xarray as xr
from mpi4py import MPI as _MPI
from mpi4py.MPI import Intracomm
from mpi4py.util import dtlib as _dtlib
from numpy.typing import DTypeLike, NDArray

from .tools import LockFile, tmp
from .xr_mpi import XarrayMPI

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# NumPy dtype kinds mpi4py.util.dtlib can translate to a meaningful MPI
# datatype: boolean, unsigned/signed integer, float, complex. Other kinds
# (strings, objects, structured dtypes) produce an opaque byte-derived type
# that is not meaningful for the reductions in this module.
_REDUCIBLE_DTYPE_KINDS = "biufc"

# Verify that every rank posts the same buffer signature before a reduction.
# The check costs one small all-gather per array reduction, which is
# negligible beside the reduction itself but is not free, so it can be
# disabled for latency-bound production runs by setting
# CLIMTOOLS_CHECK_COLLECTIVES=0. It is on by default because a mismatched
# buffer is undefined behaviour: depending on the algorithm the MPI library
# selects, it either returns silently wrong data or deadlocks, and both are
# far more expensive to diagnose than the all-gather is to post.
CHECK_COLLECTIVE_BUFFERS = os.environ.get("CLIMTOOLS_CHECK_COLLECTIVES", "1") != "0"

_LAUNCH_ENV = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "SLURM_PROCID",
    "MV2_COMM_WORLD_RANK",
    "I_MPI_COMM_WORLD_RANK",
)


_OP_LABELS: tuple[tuple[Any, str], ...] = (
    (_MPI.SUM, "SUM"),
    (_MPI.PROD, "PROD"),
    (_MPI.MIN, "MIN"),
    (_MPI.MAX, "MAX"),
    (_MPI.LAND, "LAND"),
    (_MPI.LOR, "LOR"),
    (_MPI.BAND, "BAND"),
    (_MPI.BOR, "BOR"),
)


class MPIError(Exception):
    """MPI runtime or synchronized distributed-execution error."""


class ReduceAccessor:
    """Typed element-wise reductions across MPI ranks.

    NumPy arrays use mpi4py's uppercase buffer collectives (``Allreduce``/
    ``Reduce``); other Python objects use the lowercase object collectives.
    xarray ``DataArray``/``Dataset`` inputs are reduced per variable and
    rewrapped with their original dims, coords, and attrs.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime that owns the active communicator.
    """

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime

    @staticmethod
    def _label(op: _MPI.Op) -> str:
        """Return a picklable, rank-stable label for a reduction operator."""
        for candidate, name in _OP_LABELS:
            if op == candidate:
                return name
        return "OP"

    @staticmethod
    def _is_dataarray(value: Any) -> bool:
        """Return whether ``value`` is an xarray DataArray."""
        return isinstance(value, xr.DataArray)

    @staticmethod
    def _is_dataset(value: Any) -> bool:
        """Return whether ``value`` is an xarray Dataset."""
        return isinstance(value, xr.Dataset)

    @staticmethod
    def _divide_by_ranks(value: Any, size: int) -> Any:
        """Divide a reduced value by ``size`` without widening its dtype."""
        dtype = getattr(value, "dtype", None)
        if dtype is not None and np.dtype(dtype).kind in "fc":
            return value / np.dtype(dtype).type(size)
        return value / size

    def _reduce(
        self,
        value: Any,
        op: _MPI.Op,
        *,
        mode: Literal["all", "root"] = "all",
        root: int = 0,
    ) -> T | None:
        """Execute a reduction collective for the active communicator.

        Parameters
        ----------
        value : Any
            Python object, NumPy array, or xarray DataArray/Dataset.
        op : mpi4py.MPI.Op
            MPI reduction operator.
        mode : {"all", "root"}, optional
            "all" returns the result on every rank; "root" only on ``root``.
        root : int, optional
            Destination rank for ``mode="root"``. Default 0.

        Returns
        -------
        Any or None
            Reduced value according to ``mode``.

        Raises
        ------
        ValueError
            If ``mode`` or ``root`` is invalid.
        MPIError
            If an unsupported dtype is posted, or ranks disagree on the
            reduction buffer signature.
        """
        if mode not in ("all", "root"):
            raise ValueError("mode must be either 'all' or 'root'.")

        comm = self._runtime.comm
        if mode == "root":
            if isinstance(root, bool) or not isinstance(root, Integral) or root < 0:
                raise ValueError("root must be a non-negative integer rank.")
            if root >= comm.size:
                raise ValueError(f"root {root} is outside [0, {comm.size}).")

        if self._is_dataset(value):
            # Validate every variable and stage its send buffer before
            # posting any collective, so an unsupported variable raises on
            # every rank at the same point instead of leaving some ranks
            # blocked inside a collective the failing rank never reaches.
            arrays: dict[Hashable, np.ndarray[Any, Any]] = {}
            for name, da in value.data_vars.items():
                array = np.asarray(da.values)
                if array.dtype.kind not in _REDUCIBLE_DTYPE_KINDS:
                    raise MPIError(f"Unsupported MPI NumPy dtype: {array.dtype}.")
                if not array.flags.c_contiguous:
                    array = np.ascontiguousarray(array)
                arrays[name] = array

            # A Dataset reduces one variable per Allreduce/Reduce call, but
            # every variable's buffer-agreement signature is checked inside
            # a single allgather here rather than one per variable, halving
            # the collective count for a Dataset with many variables.
            if CHECK_COLLECTIVE_BUFFERS and comm.size > 1 and arrays:
                signature = tuple(
                    (
                        str(name),
                        self._label(op),
                        mode,
                        int(root),
                        array.dtype.str,
                        tuple(int(length) for length in array.shape),
                    )
                    for name, array in arrays.items()
                )
                gathered = comm.allgather(signature)
                if len(set(gathered)) != 1:
                    disagreeing = [
                        index
                        for index, item in enumerate(gathered)
                        if item != gathered[0]
                    ]
                    raise MPIError(
                        "MPI ranks posted different reduction buffers for a "
                        + f"Dataset reduction. Rank 0 has {gathered[0]!r}, ranks "
                        + f"{disagreeing} disagree (rank {disagreeing[0]} has "
                        + f"{gathered[disagreeing[0]]!r})."
                    )

            reduced_vars: dict[Hashable, np.ndarray[Any, Any]] = {}
            for name, array in arrays.items():
                if mode == "all":
                    recv = np.empty(array.shape, dtype=array.dtype)
                    comm.Allreduce(array, recv, op=op)
                else:
                    recv = (
                        np.empty(array.shape, dtype=array.dtype)
                        if comm.rank == root
                        else None
                    )
                    comm.Reduce(array, recv, op=op, root=root)
                reduced_vars[name] = recv

            if mode == "root" and comm.rank != root:
                return None
            return cast("T", value.copy(data=reduced_vars))

        if self._is_dataarray(value):
            reduced = self._reduce(np.asarray(value.values), op, mode=mode, root=root)
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

        # Buffer-like arguments let mpi4py infer the MPI datatype from the
        # NumPy dtype for fixed-size collectives (Allreduce/Reduce); vector
        # collectives such as scatterv still need it spelled out explicitly.
        # Allreduce/Reduce require an identical count and datatype on every
        # rank -- a mismatch hangs some ranks with no error, so the buffer
        # signature is compared first to turn that into a named exception.
        if CHECK_COLLECTIVE_BUFFERS and comm.size > 1:
            signature = (
                self._label(op),
                mode,
                int(root),
                send.dtype.str,
                tuple(int(length) for length in send.shape),
            )
            gathered = comm.allgather(signature)
            if len(set(gathered)) != 1:
                disagreeing = [
                    index for index, item in enumerate(gathered) if item != gathered[0]
                ]
                raise MPIError(
                    "MPI ranks posted different reduction buffers. Rank 0 has "
                    + f"{gathered[0]!r}, ranks {disagreeing} disagree "
                    + f"(rank {disagreeing[0]} has {gathered[disagreeing[0]]!r})."
                )

        if mode == "all":
            recv = np.empty(send.shape, dtype=send.dtype)
            comm.Allreduce(send, recv, op=op)
            return cast("T", recv)

        recv = np.empty(send.shape, dtype=send.dtype) if comm.rank == root else None
        comm.Reduce(send, recv, op=op, root=root)
        return cast("T | None", recv)

    def sum(
        self, value: T, *, mode: Literal["all", "root"] = "all", root: int = 0
    ) -> T | None:
        """Reduce values by summation.

        Parameters
        ----------
        value : T
            Value to reduce.
        mode : {"all", "root"}, optional
            "all" returns the result on every rank; "root" only on ``root``.
        root : int, optional
            Destination rank for ``mode="root"``. Default 0.

        Returns
        -------
        T or None
            Reduced value, or None on non-root ranks when ``mode="root"``.
        """
        return self._reduce(value, _MPI.SUM, mode=mode, root=root)

    def prod(
        self, value: T, *, mode: Literal["all", "root"] = "all", root: int = 0
    ) -> T | None:
        """Reduce values by multiplication.

        Parameters
        ----------
        value : T
            Value to reduce.
        mode : {"all", "root"}, optional
            "all" returns the result on every rank; "root" only on ``root``.
        root : int, optional
            Destination rank for ``mode="root"``. Default 0.

        Returns
        -------
        T or None
            Reduced value, or None on non-root ranks when ``mode="root"``.
        """
        return self._reduce(value, _MPI.PROD, mode=mode, root=root)

    def min(
        self, value: T, *, mode: Literal["all", "root"] = "all", root: int = 0
    ) -> T | None:
        """Reduce values by minimum.

        Parameters
        ----------
        value : T
            Value to reduce.
        mode : {"all", "root"}, optional
            "all" returns the result on every rank; "root" only on ``root``.
        root : int, optional
            Destination rank for ``mode="root"``. Default 0.

        Returns
        -------
        T or None
            Reduced value, or None on non-root ranks when ``mode="root"``.
        """
        return self._reduce(value, _MPI.MIN, mode=mode, root=root)

    def max(
        self, value: T, *, mode: Literal["all", "root"] = "all", root: int = 0
    ) -> T | None:
        """Reduce values by maximum.

        Parameters
        ----------
        value : T
            Value to reduce.
        mode : {"all", "root"}, optional
            "all" returns the result on every rank; "root" only on ``root``.
        root : int, optional
            Destination rank for ``mode="root"``. Default 0.

        Returns
        -------
        T or None
            Reduced value, or None on non-root ranks when ``mode="root"``.
        """
        return self._reduce(value, _MPI.MAX, mode=mode, root=root)

    def mean(
        self, value: T, *, mode: Literal["all", "root"] = "all", root: int = 0
    ) -> Any | None:
        """Reduce values by arithmetic mean across ranks.

        Parameters
        ----------
        value : T
            Value to average.
        mode : {"all", "root"}, optional
            "all" returns the result on every rank; "root" only on ``root``.
        root : int, optional
            Destination rank for ``mode="root"``. Default 0.

        Returns
        -------
        Any or None
            Mean value, or None on non-root ranks when ``mode="root"``.
        """
        result = self.sum(value, mode=mode, root=root)
        if result is None:
            return None
        return self._divide_by_ranks(result, self._runtime.comm.size)

    def any(
        self, value: T, *, mode: Literal["all", "root"] = "all", root: int = 0
    ) -> bool | NDArray[np.bool_] | None:
        """Reduce truth values by logical OR.

        Parameters
        ----------
        value : T
            Scalar, array, or xarray DataArray/Dataset, cast to bool.
        mode : {"all", "root"}, optional
            "all" returns the result on every rank; "root" only on ``root``.
        root : int, optional
            Destination rank for ``mode="root"``. Default 0.

        Returns
        -------
        bool, numpy.ndarray, xarray object, or None
            Logical-OR reduction according to ``mode``.
        """
        boolean_value = (
            value.astype(bool)
            if self._is_dataset(value) or self._is_dataarray(value)
            else np.asarray(value, dtype=bool)
        )
        result = self._reduce(boolean_value, _MPI.LOR, mode=mode, root=root)
        if not self._is_dataset(value) and np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result

    def all(
        self, value: T, *, mode: Literal["all", "root"] = "all", root: int = 0
    ) -> bool | NDArray[np.bool_] | None:
        """Reduce truth values by logical AND.

        Parameters
        ----------
        value : T
            Scalar, array, or xarray DataArray/Dataset, cast to bool.
        mode : {"all", "root"}, optional
            "all" returns the result on every rank; "root" only on ``root``.
        root : int, optional
            Destination rank for ``mode="root"``. Default 0.

        Returns
        -------
        bool, numpy.ndarray, xarray object, or None
            Logical-AND reduction according to ``mode``.
        """
        boolean_value = (
            value.astype(bool)
            if self._is_dataset(value) or self._is_dataarray(value)
            else np.asarray(value, dtype=bool)
        )
        result = self._reduce(boolean_value, _MPI.LAND, mode=mode, root=root)
        if not self._is_dataset(value) and np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result


class MPIRuntime:
    """User-facing MPI runtime namespace.

    Owns one intracommunicator, exposed directly through :attr:`comm`.
    Direct reductions live under :attr:`reduce`; MPI-aware xarray operations
    live under :attr:`xarray`.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm or None, optional
        Communicator used by the runtime. None uses ``MPI.COMM_WORLD``.
    """

    MPI = _MPI
    MPIError = MPIError

    # -------------------------------------------------------------------------
    # Runtime state and accessors
    # -------------------------------------------------------------------------

    def __init__(self, comm: Intracomm | None = None) -> None:
        self.comm: Intracomm = comm if comm is not None else _MPI.COMM_WORLD
        self._reduce: ReduceAccessor = ReduceAccessor(self)
        self._xarray: XarrayMPI = XarrayMPI(self)
        self._mpi_lock = LockFile(tmp / ".mpi.lock")
        self._child: MPIRuntime | None = None
        self.info: tuple[int, ...] = ()
        self.task: int | None = None
        self._install_abort_hook()
        self._warn_if_parallel_netcdf_missing()

    @staticmethod
    def alive(comm: _MPI.Comm) -> bool:
        """Return whether a process appears to have been launched by MPI.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Communicator to inspect.

        Returns
        -------
        bool
            True if ``comm`` has more than one rank, a known launch
            environment variable is set, or this process has an MPI parent.
        """
        if comm.Get_size() > 1 or builtins.any(
            key in os.environ for key in _LAUNCH_ENV
        ):
            return True
        try:
            return _MPI.Comm.Get_parent() != _MPI.COMM_NULL
        except (AttributeError, RuntimeError):
            return False

    @property
    def launched(self) -> bool:
        """bool: Whether this process appears to have been launched by MPI."""
        return self.alive(self.comm)

    def is_root(self, root: int = 0) -> bool:
        """Return whether this process has the requested root rank.

        Parameters
        ----------
        root : int, optional
            Root rank to compare against. Default 0.

        Returns
        -------
        bool
            True when the current rank equals ``root``.
        """
        return self.comm.rank == root

    @property
    def xarray(self) -> XarrayMPI:
        """XarrayMPI: MPI-aware xarray indexing, redistribution, reductions."""
        return self._xarray

    @property
    def reduce(self) -> ReduceAccessor:
        """ReduceAccessor: Element-wise cross-rank reductions."""
        return self._reduce

    @property
    def child(self) -> MPIRuntime:
        """MPIRuntime: Runtime for this rank's child communicator.

        Raises
        ------
        RuntimeError
            If :meth:`decompose` has not been called.
        """
        if self._child is None:
            raise RuntimeError("MPI runtime has not been decomposed.")

        return self._child

    # -------------------------------------------------------------------------
    # Child communicator decomposition
    # -------------------------------------------------------------------------

    def decompose(self, ntasks: int) -> None:
        """Split the communicator into ``ntasks`` equally sized children.

        Every rank belongs to exactly one child, available through
        :attr:`child`; ``child.task`` is the zero-based child index and
        ``child.info`` holds the parent ranks belonging to it. All ranks in
        the current communicator must call this.

        Parameters
        ----------
        ntasks : int
            Number of child communicators. Must evenly divide the current
            communicator size.

        Raises
        ------
        ValueError
            If the communicator size is not divisible by ``ntasks``.
        """
        size = self.comm.size

        if size % ntasks != 0:
            raise ValueError(
                f"MPI size ({size}) must be divisible by ntasks ({ntasks})."
            )

        ranks_per_task = size // ntasks
        task = self.comm.rank // ranks_per_task

        start = task * ranks_per_task
        stop = start + ranks_per_task

        child_comm = self.comm.Split(color=task, key=self.comm.rank)

        self._child = MPIRuntime(child_comm)
        self._child.task = task
        self._child.info = tuple(range(start, stop))

    # -------------------------------------------------------------------------
    # Parent-child communication
    # -------------------------------------------------------------------------

    def bcast_to_children(self, value: T | None, *, root: int = 0) -> T:
        """Broadcast one value from the parent root to every child rank.

        Parameters
        ----------
        value : T or None
            Value on the parent ``root`` rank. Non-root ranks may pass None.
        root : int, optional
            Root rank in the parent communicator. Default 0.

        Returns
        -------
        T
            Broadcast value on every rank in every child communicator.
        """
        result = self.comm.bcast(value if self.is_root(root) else None, root=root)

        return cast("T", result)

    def scatter_to_children(self, values: Sequence[T] | None, *, root: int = 0) -> T:
        """Scatter one value per child, broadcast within each child.

        Parameters
        ----------
        values : sequence of T or None
            One value per child on the parent ``root`` rank. Non-root ranks
            may pass None.
        root : int, optional
            Root rank in the parent communicator. Default 0.

        Returns
        -------
        T
            Value assigned to this rank's child; identical across the child.
        """
        child = self.child
        ranks_per_child = child.comm.size

        send: list[T | None] | None = None

        if self.is_root(root):
            source = cast("Sequence[T]", values)
            send = [None] * self.comm.size

            for task, value in enumerate(source):
                child_root = task * ranks_per_child
                send[child_root] = value

        received = self.comm.scatter(send, root=root)

        result = child.comm.bcast(received if child.is_root() else None, root=0)

        return cast("T", result)

    def gather_from_children(self, value: T, *, root: int = 0) -> list[T] | None:
        """Gather one value per child, ordered by child task index.

        Only rank zero of each child contributes its value to the gather.

        Parameters
        ----------
        value : T
            This child's complete result.
        root : int, optional
            Root rank in the parent communicator. Default 0.

        Returns
        -------
        list of T or None
            Values ordered by child task index, on the parent ``root``
            rank. None on every other rank.
        """
        child = self.child

        payload = (cast("int", child.task), value) if child.is_root() else None

        gathered = self.comm.gather(payload, root=root)

        if not self.is_root(root):
            return None

        results = [item for item in gathered if item is not None]
        results.sort(key=lambda item: item[0])

        return [item[1] for item in results]

    # -------------------------------------------------------------------------
    # NumPy and MPI helpers
    # -------------------------------------------------------------------------

    def datatype(self, dtype: DTypeLike) -> _MPI.Datatype:
        """Return the MPI datatype corresponding to a NumPy dtype.

        Parameters
        ----------
        dtype : numpy.dtype or type
            NumPy dtype to convert.

        Returns
        -------
        mpi4py.MPI.Datatype
            MPI datatype corresponding to ``dtype``.

        Raises
        ------
        MPIError
            If ``dtype`` has no supported MPI mapping.
        """
        key = np.dtype(dtype)
        if key.kind not in _REDUCIBLE_DTYPE_KINDS:
            raise MPIError(f"Unsupported MPI NumPy dtype: {key}.")
        try:
            return _dtlib.from_numpy_dtype(key)
        except (KeyError, ValueError) as exc:
            raise MPIError(f"Unsupported MPI NumPy dtype: {key}.") from exc

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

        NumPy convenience wrapper around :meth:`MPI.Comm.Scatterv`; use
        ``mpi.comm.Scatterv`` directly for the full mpi4py buffer API.

        Parameters
        ----------
        array : numpy.ndarray or None
            Source array on ``root``. Non-root ranks may pass None.
        counts : sequence of int
            Leading-axis rows sent to each rank; one entry per rank.
        recv_shape : sequence of int
            Shape of this rank's local receive array.
        dtype : numpy.dtype or type
            Dtype of the send and receive arrays.
        root : int, optional
            Rank that owns ``array``. Default 0.

        Returns
        -------
        numpy.ndarray
            Contiguous local slab received by this rank.

        Raises
        ------
        ValueError
            If ``counts`` is the wrong length, or ``array`` is None on root.
        MPIError
            If ``dtype`` has no supported MPI mapping.
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
            source_dtype = np.asarray(array).dtype
            if source_dtype != np.dtype(dtype):
                raise MPIError(
                    f"scatterv source dtype {source_dtype} does not match the "
                    + f"requested dtype {np.dtype(dtype)}. Pass the array's own "
                    + "dtype; silent conversion would copy the whole buffer."
                )
            send = [
                np.ascontiguousarray(array, dtype=dtype),
                element_counts,
                offsets,
                mpi_type,
            ]

        self.comm.Scatterv(send, [recv, mpi_type], root=root)
        return recv

    # -------------------------------------------------------------------------
    # Logging and diagnostics
    # -------------------------------------------------------------------------

    def log(
        self,
        message: str,
        *args: Any,
        root: int = 0,
        timestamp: bool = False,
        prefix: bool = True,
        logger: Callable[..., None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Emit a message from a specific MPI rank (default: :func:`print`).

        Parameters
        ----------
        message : str
            Message or format string.
        *args : Any
            Passed to ``logger``, or used for %-formatting when no logger.
        root : int, optional
            Rank allowed to log. -1 logs on every rank. Default 0.
        timestamp : bool, optional
            Prepend a timestamp (print fallback only). Default False.
        prefix : bool, optional
            Prepend an ``[MPI RANK n]`` tag. Default True.
        logger : callable, optional
            Callable used instead of :func:`print`. Default None.
        **kwargs : Any
            Forwarded to ``logger`` (or :func:`print`).
        """
        if root != -1 and not self.is_root(root):
            return

        current_rank = root if root != -1 else self.comm.rank

        # Space-pad the rank using the calculated length
        mpi_str = f"[MPI RANK {current_rank:{len(str(self.comm.size))}d}]"

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

            # Print the final assembled string. Flush by default: rank
            # output under a batch launcher is redirected to a file and is
            # therefore block-buffered, so an un-flushed log leaves the last
            # message before a hang sitting in the buffer and makes the
            # deadlock appear to be somewhere it is not.
            kwargs.setdefault("flush", True)

            with self._mpi_lock:
                print(f"{msg_prefix}{message}", **kwargs)

        else:
            # Respect the prefix flag for custom loggers
            if prefix:
                message = f"{mpi_str} {message}"
            with self._mpi_lock:
                logger(message, *args, **kwargs)

    @contextmanager
    def watchdog(
        self, phase: str = "", timeout: float = 3600.0, *, abort: bool = True
    ) -> Generator:
        """Dump every rank's stack if the enclosed block stalls.

        Arms a daemon thread per rank so a rank blocked inside a collective
        (which never reaches a barrier) still gets its traceback printed,
        naming the line it is stuck on. Every rank dumps independently, so
        the log shows both the ranks that arrived and the ones that did not.

        Parameters
        ----------
        phase : str, optional
            Label reported with the traceback dump.
        timeout : float, optional
            Seconds of no progress before dumping. <= 0 disables the
            watchdog. Default 3600.
        abort : bool, optional
            Call ``MPI_Abort`` after dumping. Default True.

        Yields
        ------
        None
        """
        if timeout <= 0.0:
            yield
            return

        finished = threading.Event()
        rank = self.comm.rank
        label = phase or "unnamed phase"

        def _fire() -> None:
            if finished.wait(timeout):
                return
            sys.stderr.write(
                f"\n[MPI RANK {rank}] WATCHDOG: no progress for {timeout:g} s "
                + f"at {label}. Stack for this rank follows.\n"
            )
            sys.stderr.flush()
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            sys.stderr.flush()
            # Give every rank a chance to print its own stack before the
            # first one tears the job down. The delay must not depend on
            # this rank's own number, or the rank actually stuck (often
            # rank 0) can end up with the shortest delay and abort before
            # higher-ranked ranks have flushed their dumps.
            if abort:
                time.sleep(5.0 + 0.25 * (self.comm.size - 1))
                sys.stderr.write(
                    f"[MPI RANK {rank}] WATCHDOG: aborting MPI_COMM_WORLD.\n"
                )
                sys.stderr.flush()
                self.comm.Abort(1)

        thread = threading.Thread(
            target=_fire, name=f"climtools-mpi-watchdog-{rank}", daemon=True
        )
        thread.start()
        try:
            yield
        finally:
            finished.set()

    # -------------------------------------------------------------------------
    # Collective error handling
    # -------------------------------------------------------------------------

    def raise_if_error(
        self, error: BaseException | None, phase: str, signature: Any = None
    ) -> None:
        """Raise a synchronized error on all ranks if any rank failed.

        Parameters
        ----------
        error : BaseException or None
            This rank's pending error, if any.
        phase : str
            Label reported in the synchronized error message.
        signature : Any, optional
            Description of the collective this rank is about to post (op,
            mode, root, dtype, shape). When given, it is compared across
            ranks in the same all-gather that synchronizes errors, so a
            divergent collective sequence raises immediately instead of
            hanging in the next collective.
        """
        gathered = self.comm.allgather((error is not None, signature))
        failed = [state for state, _ in gathered]

        if signature is not None and not builtins.any(failed):
            signatures = [item for _, item in gathered]
            if builtins.any(item != signatures[0] for item in signatures):
                disagreeing = [
                    index
                    for index, item in enumerate(signatures)
                    if item != signatures[0]
                ]
                raise MPIError(
                    f"MPI ranks posted different collectives during {phase}. "
                    + f"Ranks {disagreeing} disagree with rank 0 "
                    + f"({signatures[0]!r} on rank 0, "
                    + f"{signatures[disagreeing[0]]!r} on rank "
                    + f"{disagreeing[0]})."
                )

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

    def _install_abort_hook(self) -> bool:
        """Install a fallback ``sys.excepthook`` that calls ``MPI_Abort``.

        Prevents remaining ranks from deadlocking when a script starts with
        plain ``python`` instead of ``python -m mpi4py``.

        Returns
        -------
        bool
            True if this call installed the hook.
        """
        # Already running under `python -m mpi4py`, which installs its own hook.
        if getattr(sys.excepthook, "__module__", "") == "mpi4py.run":
            return False
        if getattr(sys.excepthook, "_climtools_mpi_abort", False):
            return False
        if not self.alive(_MPI.COMM_WORLD):
            return False

        previous = sys.excepthook

        def _abort_excepthook(
            exc_type: type[BaseException], exc_value: BaseException, traceback: Any
        ) -> None:
            try:
                previous(exc_type, exc_value, traceback)
                sys.stderr.write(
                    f"[MPI RANK {_MPI.COMM_WORLD.Get_rank()}] unhandled "
                    + f"{exc_type.__name__}; aborting MPI_COMM_WORLD so the "
                    + "remaining ranks do not deadlock in the next collective.\n"
                )
                sys.stderr.flush()
            finally:
                _MPI.COMM_WORLD.Abort(1)

        _abort_excepthook._climtools_mpi_abort = True  # type: ignore[attr-defined]
        sys.excepthook = _abort_excepthook
        return True

    def _warn_if_parallel_netcdf_missing(self) -> None:
        """Print a one-time root-only hint if parallel NetCDF-4 is missing.

        Never raises; a build without the parallel writer still works for
        everything else in climtools.
        """
        comm = self.comm
        if comm.Get_size() <= 1 or comm.Get_rank() != 0:
            return
        try:
            import netCDF4

            if netCDF4.__has_parallel4_support__:
                return
        except Exception:
            return
        sys.stderr.write(
            "[climtools] netCDF4 is not built with parallel NetCDF-4/HDF5 "
            + "support, so xgeo.to_netcdf(..., parallel=True) will raise on "
            + f"this {comm.Get_size()}-rank run. mpi.reduce, mpi.xarray, and "
            + "everything else are unaffected. To build the parallel stack, "
            + "run `env/setup_env.sh` from the climtools repository (see the "
            + "README's Installation section); nothing else needs it.\n"
        )
        sys.stderr.flush()

    # -------------------------------------------------------------------------
    # Decorator interface
    # -------------------------------------------------------------------------

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

        By default the function runs only on ``root``. It can instead run
        on every rank, or run on ``root`` and broadcast its return value.

        Parameters
        ----------
        function : callable or None, optional
            Function to decorate (positional, for bare ``@mpi`` use).
        all_ranks : bool, optional
            Run on every rank. Default False.
        broadcast : bool, optional
            Run on ``root`` and broadcast the result to every rank. Default
            False.
        root : int, optional
            Root rank for root-only execution/broadcasting. Default 0.

        Returns
        -------
        callable
            Decorated function, or a decorator when ``function`` is None.
            Root-only mode returns None on non-root ranks.

        Raises
        ------
        TypeError
            If the positional argument is not callable.
        ValueError
            If ``broadcast`` and ``all_ranks`` are both True, or ``root`` is
            invalid.
        MPIError
            If distributed execution fails on only a subset of ranks.

        Examples
        --------
        >>> @mpi
        ... def compute_metrics(): ...
        >>> @mpi(all_ranks=True)
        ... def initialize_worker(): ...
        >>> @mpi(broadcast=True)
        ... def load_shared_configuration(): ...
        """
        if function is None:
            return functools.partial(
                self, all_ranks=all_ranks, broadcast=broadcast, root=root
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
