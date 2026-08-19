"""Small user-facing MPI namespace built on :mod:`mpi4py`."""

# lib_mpi.py
from __future__ import annotations

import builtins
import datetime
import functools
import os
import sys
import time
from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Any, Literal, ParamSpec, TypeVar, cast

import numpy as np
import xarray as xr
from mpi4py import MPI as _MPI
from mpi4py.MPI import Intracomm
from mpi4py.util import dtlib as _dtlib
from numpy.typing import DTypeLike, NDArray

from .xarray_mpi import XarrayMPI

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


def install_abort_excepthook() -> bool:
    """Abort ``MPI_COMM_WORLD`` on an unhandled exception.

    mpi4py calls ``MPI_Init_thread`` on import and registers ``MPI_Finalize``
    to run at interpreter exit. An unhandled exception on a subset of ranks
    therefore does not terminate the job: the failing ranks block in
    ``MPI_Finalize`` waiting for the others, while the others block in the
    collective the failing ranks never reached. mpi4py's documented remedy is
    to launch with ``python -m mpi4py``, whose finalizer hook calls
    ``MPI_Abort``; see
    https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks

    That remedy depends on the launch command, which climtools does not
    control, so the same hook is installed here as a fallback for scripts
    started with a plain ``python``. Set ``CLIMTOOLS_MPI_ABORT_ON_EXCEPTION=0``
    to disable.

    Returns
    -------
    bool
        True if this call installed the hook.
    """
    if os.environ.get("CLIMTOOLS_MPI_ABORT_ON_EXCEPTION", "1") == "0":
        return False
    # Already running under `python -m mpi4py`, which installs its own hook.
    if getattr(sys.excepthook, "__module__", "") == "mpi4py.run":
        return False
    if getattr(sys.excepthook, "_climtools_mpi_abort", False):
        return False
    if not mpi_alive(_MPI.COMM_WORLD):
        return False

    previous = sys.excepthook

    def _abort_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: Any,
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


def _divide_by_ranks(value: Any, size: int) -> Any:
    """Divide a reduced value by the rank count without widening its dtype.

    A float32 array divided by a Python int stays float32, but the divisor is
    built in the value's own dtype so the result cannot be promoted on any
    NumPy promotion rule. Integer and Boolean inputs keep NumPy's own mean
    semantics, which produce a floating result.

    Parameters
    ----------
    value : Any
        Reduced scalar, NumPy array, or xarray object.
    size : int
        Number of MPI ranks contributing to the reduction.

    Returns
    -------
    Any
        ``value`` divided by ``size``.
    """
    dtype = getattr(value, "dtype", None)
    if dtype is not None and np.dtype(dtype).kind in "fc":
        return value / np.dtype(dtype).type(size)
    return value / size


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
        # Validate every variable before posting any collective. The dtypes are
        # identical on every rank, so an unsupported variable now raises on all
        # ranks at the same point instead of leaving some ranks blocked inside
        # a collective that the failing rank never reaches.
        for da in value.data_vars.values():
            dtype = np.asarray(da.values).dtype
            if dtype.kind not in _REDUCIBLE_DTYPE_KINDS:
                raise MPIError(f"Unsupported MPI NumPy dtype: {dtype}.")
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
    # Allocate the receive buffer with the send buffer's own dtype and a
    # C-contiguous layout. np.empty_like inherits the source memory order,
    # which can differ from the contiguous copy actually being sent.
    if mode == "all":
        recv = np.empty(send.shape, dtype=send.dtype)
        comm.Allreduce(send, recv, op=op)
        return cast("T", recv)

    recv = np.empty(send.shape, dtype=send.dtype) if comm.rank == root else None
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
        return _divide_by_ranks(result, self._runtime.comm.size)

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
        if not is_dataset(value) and np.ndim(value) == 0 and result is not None:
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
        if not is_dataset(value) and np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result


class MPIRuntime:
    """User-facing MPI runtime namespace.

    The runtime owns one intracommunicator and exposes it directly through
    :attr:`comm`, preserving the native :class:`mpi4py.MPI.Intracomm` type,
    method signatures, IDE completion, and third-party interoperability.
    Direct MPI reductions are grouped under :attr:`reduce`, while MPI-aware
    xarray operations are grouped under :attr:`xarray`.

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
    xarray : XarrayMPI
        MPI-aware xarray indexing, redistribution, and named-dimension reductions.
    """

    MPI = _MPI
    MPIError = MPIError

    def __init__(self, comm: Intracomm | None = None) -> None:
        self.comm: Intracomm = comm if comm is not None else _MPI.COMM_WORLD
        self._reduce: ReduceAccessor = ReduceAccessor(self)
        self._xarray: XarrayMPI = XarrayMPI(self)
        install_abort_excepthook()

    @property
    def xarray(self) -> XarrayMPI:
        """Return MPI-aware xarray indexing, redistribution, and reductions."""
        return self._xarray

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
        prefix: bool = True,
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
            Rank allowed to emit the message. Default is 0. If -1, log all.
        timestamp : bool, optional
            If True, prepends a standard ISO-like timestamp to the message.
            Only applies when falling back to the built-in print. Default is False.
        prefix : bool, optional
            If True, prepends an MPI rank indicator to the message
            for both the built-in print and custom loggers. Default is True.
        logger : callable, optional
            Callable used to emit the message. Default is None, which falls back
            to the built-in :func:`print`.
        **kwargs : Any
            Keyword arguments forwarded to the ``logger`` (or :func:`print`).

        Returns
        -------
        None
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
            print(f"{msg_prefix}{message}", **kwargs)

        else:
            # Respect the prefix flag for custom loggers
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

    def sync(self, phase: str = "", timeout: float | None = None) -> None:
        """Barrier that reports and aborts instead of blocking forever.

        A plain ``comm.barrier()`` has no timeout: if one rank never arrives,
        every other rank blocks until the batch scheduler kills the job, with
        no indication of where. This posts a non-blocking barrier and polls it,
        so a straggler produces a diagnosable abort instead of a silent hang.

        Parameters
        ----------
        phase : str, optional
            Label reported if the barrier times out.
        timeout : float, optional
            Seconds to wait before aborting. If None, read from the
            ``CLIMTOOLS_MPI_SYNC_TIMEOUT`` environment variable; a value of
            zero or an unset variable falls back to a plain blocking barrier.

        Returns
        -------
        None
        """
        if self.comm.size == 1:
            return
        if timeout is None:
            try:
                timeout = float(os.environ.get("CLIMTOOLS_MPI_SYNC_TIMEOUT", "") or 0.0)
            except ValueError:
                timeout = 0.0
        if timeout <= 0.0:
            self.comm.barrier()
            return

        request = self.comm.Ibarrier()
        deadline = time.monotonic() + timeout
        while not request.Test():
            if time.monotonic() >= deadline:
                sys.stderr.write(
                    f"[MPI RANK {self.comm.rank}] collective timeout after "
                    + f"{timeout:g} s at {phase or 'unnamed phase'}: at least "
                    + "one rank never reached this barrier. Aborting "
                    + "MPI_COMM_WORLD to avoid an unbounded deadlock.\n"
                )
                sys.stderr.flush()
                self.comm.Abort(1)
            time.sleep(0.05)

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
