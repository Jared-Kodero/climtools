"""MPI setup and decorators for functions executed by an existing MPI job.

This module never starts processes. The caller launches Python with ``srun``
or ``mpirun``. MPI is initialized through :mod:`climtools.lib_mpi.native` on
first use rather than at import, so importing the package on a machine where
the extension has not been built does not fail.

Without the extension, and only when the process launcher reports a single
task, :func:`world` reports ``(0, 1)`` and the decorators execute the wrapped
function locally. One script therefore runs unchanged under ``python`` and
under ``mpirun -n N python``. A launcher reporting more than one task with no
usable extension raises :class:`MPIError`, because reporting ``(0, 1)`` there
would let every rank believe it owns the whole dataset.
"""

from __future__ import annotations

import functools
import os
import pickle
from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from . import native

P = ParamSpec("P")
R = TypeVar("R")

if TYPE_CHECKING:  # resolved at runtime by the module-level __getattr__
    MPI_RANK: int
    MPI_SIZE: int

#: Environment variables set by the common launchers, used to detect a
#: multi-rank job when the native library itself cannot be loaded.
_LAUNCHER_SIZE_VARS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_RANK",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
    "SLURM_NPROCS",
)


class MPIError(native.NativeLibraryError):
    """An error propagated from another rank."""


def available() -> bool:
    """Report whether the compiled MPI extension can be used.

    Returns
    -------
    bool
        ``True`` when the native library loads and MPI initializes.
    """
    if not native.available():
        return False
    try:
        native.init()
    except native.NativeLibraryError:
        return False
    return True


def launcher_size() -> int:
    """Return the world size advertised by the process launcher.

    Returns
    -------
    int
        Number of tasks reported by the launcher environment, or ``1`` when
        the process was started without one.

    Notes
    -----
    This inspects the environment only. It is the sole way to tell a genuine
    one-rank job from a multi-rank job whose native library failed to load,
    which decides whether a serial fallback is safe.
    """
    for name in _LAUNCHER_SIZE_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            size = int(value)
        except ValueError:
            continue
        if name == "PMIX_RANK":
            size += 1
        if size > 1:
            return size
    return 1


def world() -> tuple[int, int]:
    """Return the rank and size of ``MPI_COMM_WORLD``.

    Returns
    -------
    tuple of int
        ``(rank, size)``. Without the native library this is ``(0, 1)``, so
        that a script written for MPI also runs unmodified in serial.

    Raises
    ------
    MPIError
        If the launcher reports more than one task but the native library
        cannot be loaded. Reporting ``(0, 1)`` there would let every rank
        believe it owns the whole dataset and overwrite the same output.
    """
    if native.available():
        return native.init()

    size = launcher_size()
    if size > 1:
        raise MPIError(
            f"the launcher reports {size} tasks but the MPI extension is not "
            + f"built at {native.library_path()}. Run lib_mpi/install.sh "
            + "before launching a multi-rank job."
        )
    return 0, 1


def __getattr__(name: str) -> Any:
    """Resolve ``MPI_RANK`` and ``MPI_SIZE`` on first access."""
    if name in ("MPI_RANK", "MPI_SIZE"):
        rank, size = world()
        return rank if name == "MPI_RANK" else size
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class MPI:
    """Initialize and coordinate calls over ``MPI_COMM_WORLD``.

    Parameters
    ----------
    all_ranks : bool, optional
        Execute the decorated function on every rank. By default, execute
        the function only on ``root``. Every rank must still call the wrapper.
    broadcast : bool, optional
        Broadcast the root function's picklable return value to every rank.
        Requires ``all_ranks=False``.
    root : int, optional
        Root rank used for root-only execution and broadcasting.
    require_ranks : int or None, optional
        Minimum acceptable size of ``MPI_COMM_WORLD``.


    Examples
    --------
    Execute only on the default root rank:

    >>> @MPI()
    ... def write_output() -> None:
    ...     pass

    Execute on every rank:

    >>> @MPI(all_ranks=True)
    ... def compute() -> int:
    ...     return MPI_RANK

    Execute on the root and broadcast the result to every rank:

    >>> @MPI(broadcast=True)
    ... def configuration() -> dict[str, int]:
    ...     return {"size": MPI_SIZE}

    Use a different root rank:

    >>> @MPI(root=1)
    ... def write_from_rank_one() -> None:
    ...     pass

    Use as a context manager to finalize MPI automatically:

    >>> with MPI() as mpi:
    ...     mpi.barrier()


    Notes
    -----
    Every rank must call a decorated function in the same order because the
    wrapper performs collective MPI communication for exception propagation
    and optional result broadcasting. When used as a context manager,
    ``finalize()`` is called on exit, including when the context exits with an
    exception. Exceptions raised inside the context are not suppressed.
    """

    MPI_RANK: int = 0
    MPI_SIZE: int = 1

    def __init__(
        self,
        *,
        all_ranks: bool = False,
        broadcast: bool = False,
        root: int = 0,
        require_ranks: int | None = None,
    ) -> None:
        rank, size = world()
        type(self).MPI_RANK = rank
        type(self).MPI_SIZE = size
        self.native = native.available()

        if not isinstance(all_ranks, bool) or not isinstance(broadcast, bool):
            raise TypeError("all_ranks and broadcast must be bool values.")
        if isinstance(root, bool) or not isinstance(root, Integral):
            raise TypeError("root must be an integer rank.")
        root = int(root)
        if not 0 <= root < size:
            raise ValueError(
                f"Target root rank {root} is outside valid range [0, {size})."
            )
        if broadcast and all_ranks:
            raise ValueError(
                "Invalid configuration: broadcast=True is incompatible with all_ranks=True."
            )
        if require_ranks is not None and (
            isinstance(require_ranks, bool) or not isinstance(require_ranks, Integral)
        ):
            raise TypeError("require_ranks must be an integer or None.")
        if require_ranks is not None and require_ranks < 1:
            raise ValueError(
                f"require_ranks must be a positive integer, got {require_ranks}."
            )
        if require_ranks is not None and size < require_ranks:
            raise MPIError(
                f"MPI_COMM_WORLD size ({size}) is smaller than required minimum ranks ({require_ranks}). "
                + f"Launch job with `srun --ntasks={require_ranks} --mpi=pmix ...` or `mpirun -n {require_ranks} ...`."
            )

        self.all_ranks = bool(all_ranks)
        self.broadcast = bool(broadcast)
        self.root = root
        self.rank = rank
        self.size = size

    @property
    def is_root(self) -> bool:
        """Whether the current process is the configured root rank."""
        return self.rank == self.root

    def __repr__(self) -> str:
        return (
            f"MPI(rank={self.rank}, size={self.size}, root={self.root}, "
            + f"all_ranks={self.all_ranks}, broadcast={self.broadcast})"
        )

    def __enter__(self):
        """Enter the MPI context and return this coordinator."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finalize()

    def barrier(self) -> None:
        """Wait until every rank reaches this call."""
        if not self.native:
            return
        native.check(native.lib.mpi_netcdf_barrier(), "MPI barrier")

    def allgather(self, value: int) -> list[int]:
        """Gather one integer from each rank in rank order.

        Parameters
        ----------
        value : int
            Integer contributed by the current rank.

        Returns
        -------
        list of int
            Values contributed by all ranks in rank order.
        """
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("allgather value must be an integer.")
        if not self.native:
            return [int(value)]
        return native.allgather_i64(int(value), self.size)

    def bcast(self, value: Any = None, *, root: int | None = None) -> Any:
        """Broadcast a picklable object from a root rank.

        Parameters
        ----------
        value : Any, optional
            Object supplied by the root rank. Values supplied by non-root
            ranks are ignored.
        root : int or None, optional
            Broadcast source rank. If ``None``, use the configured ``root``.

        Returns
        -------
        Any
            Object broadcast by the source rank.

        Raises
        ------
        ValueError
            If ``root`` is outside ``MPI_COMM_WORLD``.
        """
        if root is not None and (
            isinstance(root, bool) or not isinstance(root, Integral)
        ):
            raise TypeError("broadcast root must be an integer rank or None.")
        source = self.root if root is None else int(root)
        if not 0 <= source < self.size:
            raise ValueError(
                f"Target broadcast root rank {source} is outside valid range [0, {self.size})."
            )
        if not self.native:
            return value
        return native.bcast_obj(value, source)

    def allgather_obj(self, value: Any) -> list[Any]:
        """Gather one picklable object from every rank in rank order.

        Parameters
        ----------
        value : Any
            Object contributed by the current rank.

        Returns
        -------
        list of Any
            Objects contributed by all ranks, ordered by rank.

        Notes
        -----
        Implemented as one broadcast per rank, so the list is assembled in the
        same order on every rank. Reductions built on it are bit-identical
        everywhere, which the parallel writer requires of any array that is
        replicated rather than partitioned.
        """
        if not self.native:
            return [value]
        return [self.bcast(value, root=source) for source in range(self.size)]

    def allreduce_sum(self, value: Any) -> Any:
        """Sum one picklable object across all ranks in rank order.

        Parameters
        ----------
        value : Any
            Addend contributed by the current rank. Any type supporting ``+``
            is accepted, including numpy arrays and xarray objects.

        Returns
        -------
        Any
            Sum over all ranks, identical on every rank.
        """
        parts = self.allgather_obj(value)
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total

    def abort(self, code: int = 1) -> None:
        """Terminate all ranks in ``MPI_COMM_WORLD``.

        Parameters
        ----------
        code : int, optional
            Error code supplied to the MPI abort operation.
        """
        if not self.native:
            raise SystemExit(code)
        native.abort(code)

    def finalize(self) -> None:
        """Finalize MPI if this package initialized it."""
        if not self.native:
            return
        native.finalize()

    def _raise_distributed(self, error: BaseException | None) -> None:
        failed = self.allgather(1 if error is not None else 0)

        try:
            failed_rank = failed.index(1)
        except ValueError:
            return

        # A rank that failed already holds the original exception, with its
        # own type and traceback. Only ranks that succeeded need the remote
        # description, and only those may be told about another rank.
        if error is not None and all(failed):
            raise error.with_traceback(error.__traceback__)

        detail: tuple[str, str] | None = None

        if self.rank == failed_rank and error is not None:
            detail = (type(error).__name__, str(error))

        detail = cast(
            tuple[str, str],
            self.bcast(detail, root=failed_rank),
        )

        if error is not None:
            raise error.with_traceback(error.__traceback__)

        name, message = detail
        raise MPIError(
            f"Distributed execution failed on rank {failed_rank} with {name}: {message}"
        )

    def _call_all_ranks(
        self,
        function: Callable[P, R],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> R:
        result: R | None = None
        error: BaseException | None = None

        try:
            result = function(*args, **kwargs)
        except BaseException as exc:
            error = exc

        self._raise_distributed(error)
        return cast(R, result)

    def _call_root(
        self,
        function: Callable[P, R],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> R | None:
        result: R | None = None
        error: BaseException | None = None

        if self.is_root:
            try:
                result = function(*args, **kwargs)

                if self.broadcast:
                    pickle.dumps(
                        result,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            except BaseException as exc:
                error = exc

        self._raise_distributed(error)

        if self.broadcast:
            return cast(R, self.bcast(result))

        return result

    def __call__(self, function: Callable[P, R]) -> Callable[P, R | None]:
        """Decorate a function for root-only or all-rank execution.

        Parameters
        ----------
        function : Callable
            Function to execute according to this MPI configuration.

        Returns
        -------
        Callable
            Wrapped function. Every rank must call the wrapper.
        """

        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
            if self.all_ranks:
                return self._call_all_ranks(function, args, kwargs)

            return self._call_root(function, args, kwargs)

        wrapper.mpi = self
        return wrapper


__all__ = [
    "MPI",
    "MPI_RANK",
    "MPI_SIZE",
    "MPIError",
    "available",
    "launcher_size",
    "world",
]
