"""Small user-facing MPI namespace built on :mod:`mpi4py`."""

from __future__ import annotations

import builtins
import functools
import os
import warnings
from numbers import Integral
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import numpy as np
from mpi4py import MPI as _MPI

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    _CommBase = _MPI.Comm
else:
    _CommBase = object

P = ParamSpec("P")
R = TypeVar("R")

mpi: _MPIRuntime = cast("_MPIRuntime", None)


class MPIError(Exception):
    """MPI runtime or synchronized distributed-execution error."""


MPI_TYPE_BY_DTYPE: dict[np.dtype[Any], _MPI.Datatype] = {
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
    MPI_TYPE_BY_DTYPE[np.dtype(np.complex64)] = _MPI.C_FLOAT_COMPLEX
if hasattr(_MPI, "C_DOUBLE_COMPLEX"):
    MPI_TYPE_BY_DTYPE[np.dtype(np.complex128)] = _MPI.C_DOUBLE_COMPLEX

_LAUNCH_ENV = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "SLURM_PROCID",
    "MV2_COMM_WORLD_RANK",
    "I_MPI_COMM_WORLD_RANK",
)
_warned_unlaunched = False


def _launched(comm: _MPI.Comm) -> bool:
    if comm.Get_size() > 1 or builtins.any(key in os.environ for key in _LAUNCH_ENV):
        return True
    try:
        return _MPI.Comm.Get_parent() != _MPI.COMM_NULL
    except (AttributeError, RuntimeError):
        return False


def _warn_unlaunched() -> None:
    global _warned_unlaunched
    if _warned_unlaunched:
        return
    _warned_unlaunched = True
    warnings.warn(
        "MPI accessed outside a detected mpirun/mpiexec/srun launch; "
        + "MPI_COMM_WORLD contains one process.",
        RuntimeWarning,
        stacklevel=3,
    )


class _MPIRuntime(_CommBase):
    """User-facing MPI runtime namespace.

    The object exposes convenience properties and reductions while delegating
    unwrapped communicator operations to ``MPI.COMM_WORLD``. Static type
    checkers see this class as an ``mpi4py.MPI.Comm`` subclass, so delegated
    communicator methods retain their mpi4py signatures and IDE completion.

    Parameters
    ----------
    world : mpi4py.MPI.Comm or None, optional
        Communicator wrapped by the runtime. If None, use ``MPI.COMM_WORLD``.

    Attributes
    ----------
    MPI : module
        The :mod:`mpi4py.MPI` module.
    MPI_TYPE_BY_DTYPE : dict
        Mapping from supported NumPy dtypes to MPI datatypes.
    MPIError : type[MPIError]
        Exception type used for synchronized MPI failures.
    """

    MPI = _MPI
    MPI_TYPE_BY_DTYPE = MPI_TYPE_BY_DTYPE
    MPIError = MPIError

    def __init__(self, world: _MPI.Comm | None = None) -> None:
        self._world = world if world is not None else _MPI.COMM_WORLD

    @property
    def launched(self) -> bool:
        """Return whether this process appears to have been launched by MPI.

        Returns
        -------
        bool
            True when an MPI or Slurm launch environment is detected.
        """
        return _launched(self._world)

    def _check_launcher(self) -> None:
        if not self.launched:
            _warn_unlaunched()

    @property
    def world(self) -> _MPI.Comm:
        """Return the wrapped communicator.

        Returns
        -------
        mpi4py.MPI.Comm
            Wrapped communicator, normally ``MPI.COMM_WORLD``.

        Warns
        -----
        RuntimeWarning
            If no MPI or Slurm launcher is detected.
        """
        self._check_launcher()
        return self._world

    @property
    def rank(self) -> int:
        """Return the rank of this process in the wrapped communicator.

        Returns
        -------
        int
            Zero-based MPI rank.

        Warns
        -----
        RuntimeWarning
            If no MPI or Slurm launcher is detected.
        """
        self._check_launcher()
        return self._world.Get_rank()

    @property
    def size(self) -> int:
        """Return the number of ranks in the wrapped communicator.

        Returns
        -------
        int
            Communicator size.

        Warns
        -----
        RuntimeWarning
            If no MPI or Slurm launcher is detected.
        """
        self._check_launcher()
        return self._world.Get_size()

    def datatype(self, dtype: np.dtype[Any] | type[Any]) -> _MPI.Datatype:
        """Return the MPI datatype corresponding to a NumPy dtype.

        Parameters
        ----------
        dtype : numpy.dtype or type
            NumPy dtype or scalar type to translate.

        Returns
        -------
        mpi4py.MPI.Datatype
            MPI datatype corresponding to ``dtype``.

        Raises
        ------
        MPIError
            If ``dtype`` has no supported MPI datatype mapping.
        """
        key = np.dtype(dtype)
        try:
            return self.MPI_TYPE_BY_DTYPE[key]
        except KeyError:
            raise MPIError(f"Unsupported MPI NumPy dtype: {key}.") from None

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
        return self.rank == root

    def scatterv(
        self,
        array: np.ndarray[Any, Any] | None,
        counts: Sequence[int],
        recv_shape: Sequence[int],
        dtype: np.dtype[Any] | type[Any],
        *,
        root: int = 0,
    ) -> np.ndarray[Any, Any]:
        """Scatter unequal contiguous leading-axis slabs from one rank.

        Parameters
        ----------
        array : numpy.ndarray or None
            Source array on ``root``. Non-root ranks may pass None.
        counts : sequence of int
            Number of leading-axis rows sent to each rank. The sequence must
            contain exactly ``mpi.size`` entries.
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
        if counts_array.shape != (self.size,):
            raise ValueError(f"counts must contain {self.size} values.")

        shape = tuple(int(length) for length in recv_shape)
        recv = np.empty(shape, dtype=dtype)
        row_size = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        element_counts = counts_array * row_size
        offsets = np.zeros(self.size, dtype=np.int64)
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

        self.world.Scatterv(send, [recv, mpi_type], root=root)
        return recv

    def _reduce(self, value: Any, op: _MPI.Op, root: int | None) -> Any:
        if not isinstance(value, np.ndarray):
            if root is None:
                return self.world.allreduce(value, op=op)
            return self.world.reduce(value, op=op, root=root)

        send = np.ascontiguousarray(value)
        mpi_type = self.datatype(send.dtype)
        send_buffer = [send, mpi_type]
        if root is None:
            recv = np.empty_like(send)
            self.world.Allreduce(send_buffer, [recv, mpi_type], op=op)
            return recv

        recv = np.empty_like(send) if self.is_root(root) else None
        recv_buffer = [recv, mpi_type] if recv is not None else None
        self.world.Reduce(send_buffer, recv_buffer, op=op, root=root)
        return recv

    def sum(self, value: Any, *, root: int | None = None) -> Any:
        """Reduce values by summation.

        Parameters
        ----------
        value : Any
            Scalar-like Python object or NumPy array to reduce.
        root : int or None, optional
            Destination rank. If None, perform an all-reduce and return the
            result on every rank. Default is None.

        Returns
        -------
        Any
            Reduced value on every rank when ``root`` is None, on ``root``
            otherwise, and None on non-root ranks for rooted reductions.
        """
        return self._reduce(value, _MPI.SUM, root)

    def prod(self, value: Any, *, root: int | None = None) -> Any:
        """Reduce values by multiplication.

        Parameters
        ----------
        value : Any
            Scalar-like Python object or NumPy array to reduce.
        root : int or None, optional
            Destination rank. If None, perform an all-reduce and return the
            result on every rank. Default is None.

        Returns
        -------
        Any
            Reduced value on every rank when ``root`` is None, on ``root``
            otherwise, and None on non-root ranks for rooted reductions.
        """
        return self._reduce(value, _MPI.PROD, root)

    def min(self, value: Any, *, root: int | None = None) -> Any:
        """Reduce values by minimum.

        Parameters
        ----------
        value : Any
            Scalar-like Python object or NumPy array to reduce.
        root : int or None, optional
            Destination rank. If None, perform an all-reduce and return the
            result on every rank. Default is None.

        Returns
        -------
        Any
            Reduced value on every rank when ``root`` is None, on ``root``
            otherwise, and None on non-root ranks for rooted reductions.
        """
        return self._reduce(value, _MPI.MIN, root)

    def max(self, value: Any, *, root: int | None = None) -> Any:
        """Reduce values by maximum.

        Parameters
        ----------
        value : Any
            Scalar-like Python object or NumPy array to reduce.
        root : int or None, optional
            Destination rank. If None, perform an all-reduce and return the
            result on every rank. Default is None.

        Returns
        -------
        Any
            Reduced value on every rank when ``root`` is None, on ``root``
            otherwise, and None on non-root ranks for rooted reductions.
        """
        return self._reduce(value, _MPI.MAX, root)

    def mean(self, value: Any, *, root: int | None = None) -> Any:
        """Reduce values by arithmetic mean across ranks.

        Parameters
        ----------
        value : Any
            Scalar-like Python object or NumPy array to average.
        root : int or None, optional
            Destination rank. If None, return the mean on every rank. Default
            is None.

        Returns
        -------
        Any
            Arithmetic mean on every rank when ``root`` is None, on ``root``
            otherwise, and None on non-root ranks for rooted reductions.
        """
        result = self.sum(value, root=root)
        if root is not None and not self.is_root(root):
            return None
        return result / self.size

    def any(self, value: Any, *, root: int | None = None) -> Any:
        """Reduce truth values by logical OR.

        Parameters
        ----------
        value : Any
            Scalar-like value or array converted to Boolean values.
        root : int or None, optional
            Destination rank. If None, return the result on every rank.
            Default is None.

        Returns
        -------
        bool or numpy.ndarray or None
            Logical-OR reduction. Scalar inputs return bool where a result is
            present; array inputs return an array.
        """
        result = self._reduce(np.asarray(value, dtype=bool), _MPI.LOR, root)
        if np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result

    def all(self, value: Any, *, root: int | None = None) -> Any:
        """Reduce truth values by logical AND.

        Parameters
        ----------
        value : Any
            Scalar-like value or array converted to Boolean values.
        root : int or None, optional
            Destination rank. If None, return the result on every rank.
            Default is None.

        Returns
        -------
        bool or numpy.ndarray or None
            Logical-AND reduction. Scalar inputs return bool where a result is
            present; array inputs return an array.
        """
        result = self._reduce(np.asarray(value, dtype=bool), _MPI.LAND, root)
        if np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result

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

    def raise_if_error(self, error: BaseException | None, phase: str) -> None:
        """Raise a synchronized error on all ranks if any rank failed.

        Parameters
        ----------
        error : BaseException or None
            Local exception, or None if the local rank completed successfully.
        phase : str
            Name of the distributed execution phase used in error messages.

        Returns
        -------
        None

        Raises
        ------
        BaseException
            The original local exception when every rank reports failure.
        MPIError
            If only a subset of ranks failed. The raised error identifies the
            first failed rank and, when available, its exception type and
            message.
        """
        failed = self.allgather(error is not None)
        if not builtins.any(failed):
            return

        failed_ranks = [index for index, state in enumerate(failed) if state]
        first = failed_ranks[0]
        if error is not None and len(failed_ranks) == self.size:
            raise error

        detail = None
        if self.rank == first and error is not None:
            detail = (type(error).__name__, str(error))
        detail = self.bcast(detail, root=first)
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
    ) -> Any:
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
            if root >= self.size:
                raise ValueError(f"root {root} is outside [0, {self.size}).")

            result: R | None = None
            error: BaseException | None = None
            if all_ranks or self.is_root(root):
                try:
                    result = function(*args, **kwargs)
                except BaseException as exc:
                    error = exc

            self.raise_if_error(error, function.__name__)
            if broadcast:
                return cast("R", self.bcast(result, root=root))
            return result

        wrapper.mpi = True  # type: ignore[attr-defined]
        return wrapper

    def __getattr__(self, name: str) -> Any:
        """Delegate an unwrapped attribute to the wrapped communicator.

        Parameters
        ----------
        name : str
            Communicator attribute name.

        Returns
        -------
        Any
            Attribute resolved from :attr:`world`.

        Notes
        -----
        Static type checkers obtain known communicator members from the
        type-only ``mpi4py.MPI.Comm`` base class. This fallback preserves the
        existing runtime delegation for the full communicator API.
        """
        return getattr(self.world, name)


try:
    mpi = _MPIRuntime()
except Exception:
    ...

__all__ = ["mpi"]
