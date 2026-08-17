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

P = ParamSpec("P")
R = TypeVar("R")

mpi: _MPIRuntime = None


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


class _MPIRuntime:
    """Namespace for MPI state, collectives, reductions, and ``@mpi`` execution."""

    MPI = _MPI
    MPI_TYPE_BY_DTYPE = MPI_TYPE_BY_DTYPE
    MPIError = MPIError

    def __init__(self, world: _MPI.Comm | None = None) -> None:
        self._world = world if world is not None else _MPI.COMM_WORLD

    @property
    def launched(self) -> bool:
        """Whether this process appears to have been started by MPI or Slurm."""
        return _launched(self._world)

    def _check_launcher(self) -> None:
        if not self.launched:
            _warn_unlaunched()

    @property
    def world(self) -> _MPI.Comm:
        """The wrapped communicator, normally ``MPI.COMM_WORLD``."""
        self._check_launcher()
        return self._world

    @property
    def rank(self) -> int:
        """This process's rank in ``mpi.world``."""
        self._check_launcher()
        return self._world.Get_rank()

    @property
    def size(self) -> int:
        """Number of ranks in ``mpi.world``."""
        self._check_launcher()
        return self._world.Get_size()

    def datatype(self, dtype: np.dtype[Any] | type[Any]) -> _MPI.Datatype:
        """Return the MPI datatype corresponding to a NumPy dtype."""
        key = np.dtype(dtype)
        try:
            return self.MPI_TYPE_BY_DTYPE[key]
        except KeyError:
            raise MPIError(f"Unsupported MPI NumPy dtype: {key}.") from None

    def is_root(self, root: int = 0) -> bool:
        """Return whether this process has rank ``root``."""
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
        """Scatter unequal contiguous leading-axis slabs from ``root``."""
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
        return self._reduce(value, _MPI.SUM, root)

    def prod(self, value: Any, *, root: int | None = None) -> Any:
        return self._reduce(value, _MPI.PROD, root)

    def min(self, value: Any, *, root: int | None = None) -> Any:
        return self._reduce(value, _MPI.MIN, root)

    def max(self, value: Any, *, root: int | None = None) -> Any:
        return self._reduce(value, _MPI.MAX, root)

    def mean(self, value: Any, *, root: int | None = None) -> Any:
        result = self.sum(value, root=root)
        if root is not None and not self.is_root(root):
            return None
        return result / self.size

    def any(self, value: Any, *, root: int | None = None) -> Any:
        result = self._reduce(np.asarray(value, dtype=bool), _MPI.LOR, root)
        if np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result

    def all(self, value: Any, *, root: int | None = None) -> Any:
        result = self._reduce(np.asarray(value, dtype=bool), _MPI.LAND, root)
        if np.ndim(value) == 0 and result is not None:
            return bool(np.asarray(result).item())
        return result

    def mpi_log(
        self,
        message: str,
        *args: Any,
        root: int = 0,
        logger: Callable[..., None] = print,
        **kwargs: Any,
    ) -> None:
        """Emit a message from one rank, normally rank 0."""
        if not self.is_root(root):
            return
        if logger is print and args:
            logger(message % args, **kwargs)
        else:
            logger(message, *args, **kwargs)

    def raise_if_error(self, error: BaseException | None, phase: str) -> None:
        """Raise a synchronized error on all ranks if any rank failed."""
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
        """
        Decorate a function for root-only or synchronized all-rank execution.

        By default, the decorated function executes only on the designated
        root rank. Keyword arguments can modify this behavior to execute
        on all ranks or to broadcast the root's return value to all ranks.

        Parameters
        ----------
        function : Callable[P, R] or None, optional
            The function to be decorated. Passed as a positional-only argument.
            Defaults to None to support decorator usage with keyword arguments.
        all_ranks : bool, optional
            If True, the function is executed on every rank. Default is False.
        broadcast : bool, optional
            If True, the function is executed on ``root`` and its return value
            is broadcast to every rank. Default is False.
        root : int, optional
            The integer identifier of the root rank. Default is 0.

        Returns
        -------
        Any
            The decorated function or a decorator closure (if called with arguments).
            When the decorated function is invoked, it returns the original
            function's result, which may be broadcasted to all ranks if
            ``broadcast=True``.

        Examples
        --------
        Run the function on the ``root`` rank only:

        >>> @mpi
        ... def compute_metrics():
        ...     pass

        Run the function on every rank:

        >>> @mpi(all_ranks=True)
        ... def initialize_worker():
        ...     pass

        Run the function on ``root`` and broadcast the result to all ranks:

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
        """Delegate unwrapped communicator operations to ``mpi.world``."""
        return getattr(self.world, name)


try:
    mpi = _MPIRuntime()
except Exception:
    ...

__all__ = ["mpi"]


@mpi
def x(): ...
