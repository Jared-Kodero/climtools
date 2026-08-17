"""Small user-facing MPI namespace built on :mod:`mpi4py`."""

from __future__ import annotations

import builtins
import functools
import os
from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Any, Literal, ParamSpec, TypeVar, cast

import numpy as np
from mpi4py import MPI as _MPI
from mpi4py.MPI import Intracomm
from numpy.typing import DTypeLike, NDArray

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class MPIError(Exception):
    """MPI runtime or synchronized distributed-execution error."""


_MPI_TYPE_BY_DTYPE: dict[np.dtype[Any], _MPI.Datatype] = {
    np.dtype(np.bool_): getattr(_MPI, "C_BOOL", _MPI.BOOL),
    np.dtype(np.float32): _MPI.FLOAT,
    np.dtype(np.float64): _MPI.DOUBLE,
    np.dtype(np.int8): _MPI.INT8_T,
    np.dtype(np.int16): _MPI.INT16_T,
    np.dtype(np.int32): _MPI.INT32_T,
    np.dtype(np.int64): _MPI.INT64_T,
    np.dtype(np.uint8): _MPI.UINT8_T,
    np.dtype(np.uint16): _MPI.UINT16_T,
    np.dtype(np.uint32): _MPI.UINT32_T,
    np.dtype(np.uint64): _MPI.UINT64_T,
}
if hasattr(_MPI, "C_FLOAT_COMPLEX"):
    _MPI_TYPE_BY_DTYPE[np.dtype(np.complex64)] = _MPI.C_FLOAT_COMPLEX
if hasattr(_MPI, "C_DOUBLE_COMPLEX"):
    _MPI_TYPE_BY_DTYPE[np.dtype(np.complex128)] = _MPI.C_DOUBLE_COMPLEX

_LAUNCH_ENV = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "SLURM_PROCID",
    "MV2_COMM_WORLD_RANK",
    "I_MPI_COMM_WORLD_RANK",
)


def _launched(comm: _MPI.Comm) -> bool:
    if comm.Get_size() > 1 or builtins.any(key in os.environ for key in _LAUNCH_ENV):
        return True
    try:
        return _MPI.Comm.Get_parent() != _MPI.COMM_NULL
    except (AttributeError, RuntimeError):
        return False


def _reduce(
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
        Python object or NumPy array to reduce.
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

    if not isinstance(value, np.ndarray):
        if mode == "all":
            return cast("T", comm.allreduce(value, op=op))
        return cast("T | None", comm.reduce(value, op=op, root=root))

    send = np.ascontiguousarray(value)
    mpi_type = runtime.datatype(send.dtype)
    send_buffer = [send, mpi_type]
    if mode == "all":
        recv = np.empty_like(send)
        comm.Allreduce(send_buffer, [recv, mpi_type], op=op)
        return cast("T", recv)

    recv = np.empty_like(send) if comm.rank == root else None
    recv_buffer = [recv, mpi_type] if recv is not None else None
    comm.Reduce(send_buffer, recv_buffer, op=op, root=root)
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
        return _reduce(self._runtime, value, _MPI.SUM, mode=mode, root=root)

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
        return _reduce(self._runtime, value, _MPI.PROD, mode=mode, root=root)

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
        return _reduce(self._runtime, value, _MPI.MIN, mode=mode, root=root)

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
        return _reduce(self._runtime, value, _MPI.MAX, mode=mode, root=root)

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
            Scalar-like value or array converted to Boolean values.
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
        bool or numpy.ndarray or None
            Logical-OR reduction according to ``mode``.
        """
        result = _reduce(
            self._runtime,
            np.asarray(value, dtype=bool),
            _MPI.LOR,
            mode=mode,
            root=root,
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
            Scalar-like value or array converted to Boolean values.
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
        bool or numpy.ndarray or None
            Logical-AND reduction according to ``mode``.
        """
        result = _reduce(
            self._runtime,
            np.asarray(value, dtype=bool),
            _MPI.LAND,
            mode=mode,
            root=root,
        )
        if np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result


class MPIRuntime:
    """User-facing MPI runtime namespace.

    The runtime owns one intracommunicator and exposes it directly through
    :attr:`comm`, preserving the native :class:`mpi4py.MPI.Intracomm` type,
    method signatures, IDE completion, and third-party interoperability.
    High-level reductions are grouped under :attr:`reduce`.

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
    reduce : _ReduceAccessor
        Typed high-level reduction operations.
    """

    MPI = _MPI
    MPIError = MPIError

    def __init__(self, comm: Intracomm | None = None) -> None:
        self.comm: Intracomm = comm if comm is not None else _MPI.COMM_WORLD
        self._reduce: ReduceAccessor = ReduceAccessor(self)

    @property
    def launched(self) -> bool:
        """Return whether this process appears to have been launched by MPI.

        Returns
        -------
        bool
            True when an MPI or Slurm launch environment is detected.
        """
        return _launched(self.comm)

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
        logger: Callable[..., None] = print,
        **kwargs: Any,
    ) -> None:
        """Emit a message from one rank.

        Parameters
        ----------
        message : str
            Message or format string passed to ``logger``.
        *args : Any
            Positional arguments passed to ``logger``. With the default
            ``print`` logger, positional arguments trigger percent-formatting
            of ``message`` before printing.
        root : int, optional
            Rank allowed to emit the message. Default is 0.
        logger : callable, optional
            Callable used to emit the message. Default is :func:`print`.
        **kwargs : Any
            Keyword arguments forwarded to ``logger``.

        Returns
        -------
        None
        """
        if not self.is_root(root):
            return
        if logger is print and args:
            logger(message % args, **kwargs)
        else:
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
        """Return high-level reduction operations.

        Returns
        -------
        _ReduceAccessor
            Typed reduction accessor providing ``sum``, ``prod``, ``min``,
            ``max``, ``mean``, ``any``, and ``all``.
        """
        return self._reduce

    def datatype(self, dtype: DTypeLike) -> _MPI.Datatype:
        """Return the MPI datatype corresponding to a NumPy dtype."""
        key = np.dtype(dtype)
        try:
            return _MPI_TYPE_BY_DTYPE[key]
        except KeyError:
            raise MPIError(f"Unsupported MPI NumPy dtype: {key}.") from None

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
