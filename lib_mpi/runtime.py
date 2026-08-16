"""MPI setup and decorators for functions executed by an existing MPI job.

This module never starts processes. The caller launches Python with ``srun``
or ``mpirun``. MPI is initialized through :mod:`climtools.lib_mpi.native` on
first use rather than at import, so importing the package on a machine where
the extension has not been built does not fail.

Without the extension, and only when the process launcher reports a single
task, ``MPI.world`` reports rank 0 of a one-rank world and decorated calls
execute locally. One script therefore runs unchanged under ``python`` and
under ``mpirun -n N python``. A launcher reporting more than one task with no
usable extension raises :class:`MPIError`, because a serial fallback there
would let every rank believe it owns the whole dataset.
"""

from __future__ import annotations

import builtins
import functools
import operator as _operator
import os
import pickle
import shutil
import sys
from collections.abc import Callable
from numbers import Integral
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
)

import numpy as np

from . import native
from .module_env import check_env_stack

P = ParamSpec("P")
R = TypeVar("R")

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


#: Environment variables that carry the world size, used to detect a
#: multi-rank job when the native library itself cannot be loaded.
_LAUNCHER_SIZE_VARS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
    "SLURM_NPROCS",
)

#: Variables that carry this process's rank. A rank above zero proves the job
#: has more than one task, but says nothing about how many, so these are read
#: only as evidence of a multi-rank launch and never reported as a size.
_LAUNCHER_RANK_VARS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
)


class MPIError(native.NativeLibraryError):
    """MPI runtime or distributed-execution error."""


def relaunch_with_mpi(
    ntasks: int | None = None,
    path: str | Path | None = None,
) -> None:
    """Relaunch the current script under mpirun if not already running under MPI.

    The recorded build-time module stack is loaded into the current environment
    before resolving and launching mpirun.

    Parameters
    ----------
    ntasks : int or None, optional
        Number of MPI processes to launch. If None, use the number of available
        CPUs, capped at 32.
    path : str or Path or None, optional
        Path to the mpirun executable. If None, search for mpirun in PATH after
        loading the recorded module stack.
    """
    env = os.environ

    under_mpi = any(
        name in env
        for name in (
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_RANK",
        )
    )

    under_srun = "SLURM_STEP_ID" in env and "SLURM_PROCID" in env

    if under_mpi or under_srun:
        return

    check_env_stack()

    if path is not None:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise RuntimeError(f"mpirun was not found at {candidate}")
        mpirun = str(candidate)
    else:
        mpirun = shutil.which("mpirun")
        if mpirun is None:
            raise RuntimeError("mpirun was not found in PATH")

    if not sys.argv[0]:
        raise RuntimeError(
            "Automatic MPI restart requires execution from a Python script"
        )

    if ntasks is None:
        try:
            n = min(len(os.sched_getaffinity(0)), 32)
        except AttributeError:
            n = min(os.cpu_count() or 1, 32)
    elif ntasks > 0:
        n = ntasks
    else:
        raise ValueError("ntasks must be greater than zero")

    sys.stdout.flush()
    sys.stderr.flush()

    os.execv(
        mpirun,
        [
            mpirun,
            "-n",
            str(n),
            sys.executable,
            *sys.argv,
        ],
    )


def _available() -> bool:
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


def _launcher_size() -> int:
    """Return the world size advertised by the process launcher.

    Returns
    -------
    int
        Number of tasks reported by the launcher environment. ``1`` when the
        process was started without a launcher, and ``2`` when the launcher
        reveals only that this rank is not rank zero, which proves the job has
        at least two tasks without disclosing how many.

    Notes
    -----
    This inspects the environment only. It is the sole way to tell a genuine
    one-rank job from a multi-rank job whose native library failed to load,
    which decides whether a serial fallback is safe. Only the ``>1`` decision
    is load-bearing, so a lower bound is sufficient.
    """
    for name in _LAUNCHER_SIZE_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            size = int(value)
        except ValueError:
            continue
        if size > 1:
            return size

    for name in _LAUNCHER_RANK_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            rank = int(value)
        except ValueError:
            continue
        if rank > 0:
            return 2
    return 1


def _resolve_world() -> tuple[int, int]:
    """Return the rank and size of ``MPI_COMM_WORLD``.

    Returns
    -------
    tuple of int
        ``(rank, size)``. Without a usable native library this is ``(0, 1)``
        for a serial process, so MPI-aware code also runs under plain Python.

    Raises
    ------
    MPIError
        If the launcher reports more than one task but the native MPI runtime
        cannot be initialized. A serial fallback is unsafe in that case because
        every process would otherwise believe it owns the entire dataset.
    """
    try:
        return native.init()
    except native.NativeLibraryError as exc:
        size = _launcher_size()
        if size > 1:
            raise MPIError(
                f"the launcher reports {size} tasks but the MPI runtime is "
                + f"unavailable: {exc}"
            ) from exc
        return 0, 1


# --------------------------------------------------------------- reductions


def _is_plain_number(value: Any) -> bool:
    """Whether a value is a Python number that numpy would needlessly wrap."""
    return isinstance(value, (int, float, complex)) and not isinstance(value, bool)


def _minimum(left: Any, right: Any) -> Any:
    """Elementwise minimum that leaves plain Python numbers unwrapped."""

    return np.minimum(left, right)


def _maximum(left: Any, right: Any) -> Any:
    """Elementwise maximum that leaves plain Python numbers unwrapped."""
    if _is_plain_number(left) and _is_plain_number(right):
        return max(left, right)

    return np.maximum(left, right)


def _logical_or(left: Any, right: Any) -> Any:
    """Elementwise logical OR, preserving ``bool`` for scalar operands."""
    if isinstance(left, bool) and isinstance(right, bool):
        return left or right

    return np.logical_or(left, right)


def _logical_and(left: Any, right: Any) -> Any:
    """Elementwise logical AND, preserving ``bool`` for scalar operands."""
    if isinstance(left, bool) and isinstance(right, bool):
        return left and right

    return np.logical_and(left, right)


#: Binary operators backing the named reductions. Every one is associative and
#: commutative, which is what allows a partitioned reduction to be formed as
#: partial results per rank and combined afterwards.
_REDUCE_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "sum": _operator.add,
    "prod": _operator.mul,
    "min": _minimum,
    "max": _maximum,
    "any": _logical_or,
    "all": _logical_and,
}


class mpi:
    """Initialize and coordinate calls over ``MPI_COMM_WORLD``.

    The object is both a decorator and the handle carrying the collective
    operations. Nothing about MPI is resolved when it is constructed: the rank
    and size are read on first access, and ``MPI_Init_thread`` runs then. A
    module that decorates its functions at import time therefore does not turn
    a serial process into an MPI one.

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
        Minimum acceptable size of ``MPI_COMM_WORLD``. Checked on first use,
        not at construction.

    Examples
    --------
    Execute only on the default root rank, as a bare decorator:

    >>> @MPI
    ... def write_output() -> None:
    ...     pass

    Execute on every rank:

    >>> @MPI(all_ranks=True)
    ... def compute() -> int:
    ...     return MPI.world.rank()

    Execute on the root and broadcast the result to every rank:

    >>> @MPI(broadcast=True)
    ... def configuration() -> dict[str, int]:
    ...     return {"size": MPI.world.size()}

    Use a different root rank:

    >>> @MPI(root=1)
    ... def write_from_rank_one() -> None:
    ...     pass

    Reduce across ranks through the world accessor:

    >>> total = MPI.world.sum(local_partial)    # doctest: +SKIP

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

    Reductions are evaluated in rank order on every rank, so their result is
    bit-identical everywhere. This is what the parallel writer requires of any
    array it treats as replicated rather than partitioned. Results are not
    reproducible across different rank counts, because partitioning changes
    the order in which partial sums are associated and floating-point
    addition is not associative.
    """

    def __new__(cls, function: Any = None, /, **kwargs: Any) -> Any:
        """Support both ``@MPI`` and ``@MPI(...)`` decoration.

        A bare ``@MPI`` calls this with the decorated function, in which case
        a default coordinator is built and the wrapper is returned. Returning
        a non-instance suppresses the usual ``__init__`` call, so the wrapper
        is handed back untouched.
        """
        if function is None:
            return super().__new__(cls)
        if not callable(function):
            raise TypeError(
                "MPI's only positional argument is the function being "
                + f"decorated; got {type(function).__name__}. Options are "
                + "keyword-only, for example MPI(all_ranks=True)."
            )
        return cls(**kwargs)(function)

    def __init__(
        self,
        function: Any = None,
        /,
        *,
        all_ranks: bool = False,
        broadcast: bool = False,
        root: int = 0,
        require_ranks: int | None = None,
    ) -> None:

        if not isinstance(all_ranks, bool) or not isinstance(broadcast, bool):
            raise TypeError("all_ranks and broadcast must be bool values.")
        if isinstance(root, bool) or not isinstance(root, Integral):
            raise TypeError("root must be an integer rank.")
        if broadcast and all_ranks:
            raise ValueError(
                "Invalid configuration: broadcast=True is incompatible "
                + "with all_ranks=True."
            )
        if require_ranks is not None and (
            isinstance(require_ranks, bool) or not isinstance(require_ranks, Integral)
        ):
            raise TypeError("require_ranks must be an integer or None.")
        if require_ranks is not None and require_ranks < 1:
            raise ValueError(
                f"require_ranks must be a positive integer, got {require_ranks}."
            )
        if int(root) < 0:
            raise ValueError(f"root must be a non-negative rank, got {root}.")

        self.all_ranks = bool(all_ranks)
        self.broadcast = bool(broadcast)
        self.root = int(root)
        self.require_ranks = None if require_ranks is None else int(require_ranks)

        # Resolved on first use, never at construction. `_resolve_world()`
        # initializes MPI, and doing that from a decorator evaluated at import time
        # make every module that merely imports this one an MPI process.
        self._rank: int | None = None
        self._size: int | None = None
        self._native: bool = False

    # -- lazy resolution --------------------------------------------------
    def _resolve(self) -> tuple[int, int]:
        """Resolve the world on first use and validate the configuration."""
        if self._rank is not None and self._size is not None:
            return self._rank, self._size

        rank, size = _resolve_world()

        if not 0 <= self.root < size:
            raise ValueError(
                f"Target root rank {self.root} is outside valid range [0, {size})."
            )
        if self.require_ranks is not None and size < self.require_ranks:
            raise MPIError(
                f"MPI_COMM_WORLD size ({size}) is smaller than required "
                + f"minimum ranks ({self.require_ranks}). Launch job with "
                + f"`srun --ntasks={self.require_ranks} --mpi=pmix ...` or "
                + f"`mpirun -n {self.require_ranks} ...`."
            )

        self._rank = rank
        self._size = size
        self._native = native.available()
        return rank, size

    world: ClassVar[MPIWorldAccessor]

    @property
    def rank(self) -> int:
        """Rank of this process in ``MPI_COMM_WORLD``."""
        return self._resolve()[0]

    @property
    def size(self) -> int:
        """Number of ranks in ``MPI_COMM_WORLD``."""
        return self._resolve()[1]

    @property
    def native(self) -> bool:
        """Whether the compiled extension backs this coordinator."""
        self._resolve()
        return self._native

    @property
    def is_root(self) -> bool:
        """Whether the current process is the configured root rank."""
        return self.rank == self.root

    def __repr__(self) -> str:
        if self._rank is None:
            return (
                f"MPI(unresolved, root={self.root}, "
                + f"all_ranks={self.all_ranks}, broadcast={self.broadcast})"
            )
        return (
            f"MPI(rank={self._rank}, size={self._size}, root={self.root}, "
            + f"all_ranks={self.all_ranks}, broadcast={self.broadcast})"
        )

    def __enter__(self):
        """Enter the MPI context and return this coordinator."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finalize()

    def log(
        self,
        message: str,
        *args: Any,
        logger: Callable[..., None] = print,
        **kwargs: Any,
    ) -> None:
        """Emit an informational message from the root rank only.

        Progress lines come from the root rank alone, so an eight-rank job does
        not produce eight copies of every message.

        Args:
            message (str): The message string or format string to log.
            *args (Any): Variable length arguments for lazy string formatting.
            logger (Callable[..., None], optional): The logging callable to use.
                Defaults to the built-in `print` function.
            **kwargs (Any): Arbitrary keyword arguments passed directly to the `logger`.
                For example, `exc_info=True` for standard loggers, or `end=""` for `print`.
        """
        if self.is_root():
            # Standard loggers handle lazy % formatting with *args, but print() does not.
            # If the fallback print is used alongside args, we format it manually.
            if logger is print and args:
                logger(message % args, **kwargs)
            else:
                logger(message, *args, **kwargs)

    # -- point-to-point and collective primitives -------------------------
    def barrier(self) -> None:
        """Wait until every rank reaches this call."""
        if not self.native or self.size == 1:
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
        if not self.native or self.size == 1:
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
        source = self._check_root(root)
        if not self.native or self.size == 1:
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
        One ``MPI_Allgatherv`` where the native library supports it, and a
        broadcast per rank otherwise. Both assemble the list in the same order
        on every rank, so reductions built on it are bit-identical everywhere,
        which the parallel writer requires of any array that is replicated
        rather than partitioned.
        """
        if not self.native or self.size == 1:
            return [value]
        return native.allgather_obj(value, self.size)

    def gather(self, value: Any, *, root: int | None = None) -> list[Any] | None:
        """Gather one picklable object from every rank onto ``root``.

        Parameters
        ----------
        value : Any
            Object contributed by the current rank.
        root : int or None, optional
            Destination rank. If ``None``, use the configured ``root``.

        Returns
        -------
        list of Any or None
            Objects from all ranks in rank order on ``root``, and ``None``
            on every other rank.

        Notes
        -----
        Implemented on the all-gather, so every rank pays for the full
        payload. Use :meth:`allgather_obj` when all ranks need the result
        anyway, and prefer a reduction when only an aggregate is wanted.
        """
        source = self._check_root(root)
        gathered = self.allgather_obj(value)
        return gathered if self.rank == source else None

    def scatter(self, values: Any = None, *, root: int | None = None) -> Any:
        """Distribute one element of a sequence to each rank.

        Parameters
        ----------
        values : sequence, optional
            Sequence of exactly ``size`` items, supplied by ``root``. Ignored
            on every other rank.
        root : int or None, optional
            Source rank. If ``None``, use the configured ``root``.

        Returns
        -------
        Any
            The item belonging to this rank.

        Raises
        ------
        ValueError
            If ``root`` does not supply exactly ``size`` items.
        """
        source = self._check_root(root)
        if self.rank == source:
            items = list(values or [])
            if len(items) != self.size:
                raise ValueError(
                    f"scatter expects one item per rank: got {len(items)} "
                    + f"items for {self.size} ranks."
                )
        else:
            items = []
        return self.bcast(items, root=source)[self.rank]

    def consensus(self, ok: bool) -> bool:
        """Report whether every rank passed a true value.

        Parameters
        ----------
        ok : bool
            Local verdict.

        Returns
        -------
        bool
            ``True`` only when every rank supplied a true value.

        Notes
        -----
        Use this before entering a collective that some ranks might skip. A
        rank that proceeds alone into a collective call hangs the job, so the
        agreement has to be established first.
        """
        if not self.native or self.size == 1:
            return bool(ok)
        return bool(native.lib.mpi_netcdf_consensus(1 if ok else 0))

    # -- reductions -------------------------------------------------------
    def reduce(self, value: Any, op: str = "sum") -> Any:
        """Combine one value per rank with an associative operator.

        Parameters
        ----------
        value : Any
            Contribution of the current rank. Any type supporting the
            operator is accepted, including numpy arrays and xarray objects.
        op : {"sum", "prod", "min", "max", "any", "all"}, default "sum"
            Reduction operator.

        Returns
        -------
        Any
            Reduction over all ranks, identical on every rank.

        Raises
        ------
        ValueError
            If ``op`` is not a supported operator.
        """
        try:
            operation = _REDUCE_OPS[op]
        except (KeyError, TypeError):
            raise ValueError(
                f"Unsupported reduction operator {op!r}; expected one of "
                + f"{sorted(_REDUCE_OPS)}."
            ) from None

        parts = self.allgather_obj(value)
        total = parts[0]
        for part in parts[1:]:
            total = operation(total, part)
        return total

    def allreduce_sum(self, value: Any) -> Any:
        """Sum one picklable object across all ranks in rank order.

        Parameters
        ----------
        value : Any
            Addend contributed by the current rank.

        Returns
        -------
        Any
            Sum over all ranks, identical on every rank.
        """
        return self.reduce(value, "sum")

    def sum(self, value: Any) -> Any:
        """Sum one value per rank. See :meth:`reduce`."""
        return self.reduce(value, "sum")

    def prod(self, value: Any) -> Any:
        """Multiply one value per rank. See :meth:`reduce`."""
        return self.reduce(value, "prod")

    def min(self, value: Any) -> Any:
        """Elementwise minimum over ranks. See :meth:`reduce`."""
        return self.reduce(value, "min")

    def max(self, value: Any) -> Any:
        """Elementwise maximum over ranks. See :meth:`reduce`."""
        return self.reduce(value, "max")

    def any(self, value: Any) -> Any:
        """Elementwise logical OR over ranks. See :meth:`reduce`."""
        return self.reduce(value, "any")

    def all(self, value: Any) -> Any:
        """Elementwise logical AND over ranks. See :meth:`reduce`."""
        return self.reduce(value, "all")

    def mean(self, value: Any) -> Any:
        """Arithmetic mean over ranks of the contributed values.

        Parameters
        ----------
        value : Any
            Contribution of the current rank.

        Returns
        -------
        Any
            ``sum(values) / size``, identical on every rank.

        Notes
        -----
        Every rank carries equal weight. This is the mean of one value per
        rank, not the mean over a partitioned dimension: a rank holding three
        elements would otherwise count as much as a rank holding three
        thousand. A partitioned mean is a ratio of two sums, so form the local
        numerator and denominator, reduce each with :meth:`sum`, and divide
        afterwards.
        """
        return self.reduce(value, "sum") / self.size

    # -- decomposition ----------------------------------------------------
    def partition(self, total: int, *, rank: int | None = None) -> tuple[int, int]:
        """Return this rank's contiguous half-open block of ``total`` items.

        Parameters
        ----------
        total : int
            Number of items to divide across the world.
        rank : int or None, optional
            Rank whose block is wanted. Defaults to this rank.

        Returns
        -------
        tuple of int
            ``(start, stop)`` bounds of the block, as a half-open interval.

        Notes
        -----
        The split is contiguous and the remainder is spread over the leading
        ranks, so block lengths differ by at most one. Contiguity is what the
        parallel writer requires: it recovers each rank's file offset from an
        all-gather of the local lengths, so a strided or interleaved split
        would scatter a rank's records across the whole file.
        """
        if isinstance(total, bool) or not isinstance(total, Integral):
            raise TypeError("total must be an integer.")
        if total < 0:
            raise ValueError(f"total must be non-negative, got {total}.")

        target = self.rank if rank is None else int(rank)
        if not 0 <= target < self.size:
            raise ValueError(f"rank {target} is outside valid range [0, {self.size}).")

        base, remainder = divmod(int(total), self.size)
        start = base * target + builtins.min(target, remainder)
        stop = start + base + (1 if target < remainder else 0)
        return start, stop

    def split(self, sequence: Any) -> Any:
        """Return this rank's contiguous slice of a sliceable sequence.

        Parameters
        ----------
        sequence : sequence
            Object supporting ``len()`` and slicing, identical on every rank.

        Returns
        -------
        Any
            The local block, sliced from ``sequence``.
        """
        start, stop = self.partition(len(sequence))
        return sequence[start:stop]

    # -- lifecycle --------------------------------------------------------
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
        """Finalize MPI if this coordinator initialized it."""
        if not self._native:
            return
        native.finalize()
        self._rank = None
        self._size = None
        self._native = False

    # -- internals --------------------------------------------------------
    def _check_root(self, root: int | None) -> int:
        if root is not None and (
            isinstance(root, bool) or not isinstance(root, Integral)
        ):
            raise TypeError("root must be an integer rank or None.")
        source = self.root if root is None else int(root)
        if not 0 <= source < self.size:
            raise ValueError(
                f"Target root rank {source} is outside valid range [0, {self.size})."
            )
        return source

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
            "tuple[str, str]",
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
        return cast("R", result)

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
            return cast("R", self.bcast(result))

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
        if not callable(function):
            raise TypeError(
                f"MPI can only decorate a callable, got {type(function).__name__}."
            )

        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
            # Resolving here rather than at decoration keeps import of a
            # decorated module free of MPI_Init.
            self._resolve()

            if self.all_ranks:
                return self._call_all_ranks(function, args, kwargs)

            return self._call_root(function, args, kwargs)

        wrapper.mpi = self  # type: ignore[attr-defined]
        return wrapper


# ------------------------------------------------------- world accessor


class MPIWorldAccessor:
    """
    Lazy namespace for operations on ``MPI_COMM_WORLD``.

    ``MPI.world`` provides process-wide collectives without requiring callers to
    construct an :class:`MPI` coordinator. The accessor itself is created at
    import time, but its shared all-rank coordinator is created only on first
    use. MPI initialization therefore remains lazy.

    Examples
    --------
    Query the active world:

    >>> MPI.world.rank()
    0
    >>> MPI.world.size()
    1

    Use collective operations through the same namespace:

    >>> MPI.world.barrier()
    >>> MPI.world.sum(3)
    3
    """

    __slots__ = ("_coordinator",)

    def __init__(self) -> None:
        self._coordinator: mpi | None = None

    def _comm(self) -> mpi:
        """
        Return the shared all-rank coordinator, creating it lazily.

        Returns
        -------
        MPI
            The active MPI coordinator instance.
        """
        if self._coordinator is None:
            self._coordinator = mpi(all_ranks=True)
        return self._coordinator

    # -- admin & environment ----------------------------------------------
    def available(self) -> bool:
        """
        Return whether the native MPI runtime can be loaded and initialized.

        Returns
        -------
        bool
            True if MPI runtime is available, False otherwise.
        """
        return _available()

    def launcher_size(self) -> int:
        """
        Return the world size advertised by the process launcher.

        Returns
        -------
        int
            World size retrieved from launcher metadata.
        """
        return _launcher_size()

    def rank(self) -> int:
        """
        Return this process's rank in ``MPI_COMM_WORLD``.

        Returns
        -------
        int
            Rank of the current process.
        """
        return self._comm().rank

    def size(self) -> int:
        """
        Return the number of ranks in ``MPI_COMM_WORLD``.

        Returns
        -------
        int
            Total number of processes in the world.
        """
        return self._comm().size

    def is_root(self, root: int = 0) -> bool:
        """
        Return whether this process has rank ``root``.

        Parameters
        ----------
        root : int, default 0
            The rank to check against.

        Returns
        -------
        bool
            True if the process matches the root rank, False otherwise.
        """
        return self._comm().rank == root

    def abort(self, code: int = 1) -> None:
        """
        Abort all ranks in ``MPI_COMM_WORLD`` with a process exit code.

        Parameters
        ----------
        code : int, default 1
            Exit code to return to the process launcher.
        """
        self._comm().abort(code)

    def finalize(self) -> None:
        """
        Finalize MPI when initialized by the shared world coordinator.
        """
        if self._coordinator is None:
            return
        self._coordinator.finalize()
        self._coordinator = None

    # -- synchronization --------------------------------------------------
    def barrier(self) -> None:
        """
        Block until every rank in ``MPI_COMM_WORLD`` reaches the barrier.
        """
        self._comm().barrier()

    def consensus(self, ok: bool) -> bool:
        """
        Return ``True`` only when every rank contributes a true value.

        Parameters
        ----------
        ok : bool
            The local boolean value contributed by this rank.

        Returns
        -------
        bool
            True if all ranks contribute True, False otherwise.
        """
        return self._comm().consensus(ok)

    # -- data movement ----------------------------------------------------
    def bcast(self, value: Any = None, *, root: int = 0) -> Any:
        """
        Broadcast a picklable object from ``root`` to every rank.

        Parameters
        ----------
        value : Any, optional
            Object to broadcast. Required only on the root rank. Default is None.
        root : int, default 0
            Source rank.

        Returns
        -------
        Any
            The broadcasted object.
        """
        return self._comm().bcast(value, root=root)

    def gather(self, value: Any, *, root: int = 0) -> list[Any] | None:
        """
        Gather one picklable object per rank onto ``root`` in rank order.

        Parameters
        ----------
        value : Any
            The local object contributed by this rank.
        root : int, default 0
            Destination rank.

        Returns
        -------
        list or None
            A list of gathered objects on the root rank, None elsewhere.
        """
        return self._comm().gather(value, root=root)

    def allgather(self, value: Any) -> list[Any]:
        """
        Gather one picklable object from every rank in rank order.

        Parameters
        ----------
        value : Any
            The local object contributed by this rank.

        Returns
        -------
        list
            A list of objects contributed by all ranks, identical everywhere.
        """
        return self._comm().allgather_obj(value)

    def scatter(self, values: Any = None, *, root: int = 0) -> Any:
        """
        Scatter one item per rank from a sequence supplied by ``root``.

        Parameters
        ----------
        values : Any, optional
            Sequence to scatter. Required only on the root rank. Default is None.
        root : int, default 0
            Source rank.

        Returns
        -------
        Any
            The scattered item assigned to this rank.
        """
        return self._comm().scatter(values, root=root)

    def concat(
        self, sequence: list[Any], *, root: int | None = None
    ) -> list[Any] | None:
        """
        Join every rank's sequence in rank order.

        Parameters
        ----------
        sequence : list
            The local sequence contributed by this rank.
        root : int or None, optional
            Rank the result is assembled on. If None, assemble on every rank.
            Default is None.

        Returns
        -------
        list or None
            The concatenated global sequence, or None on non-root ranks if
            a root is specified.
        """
        if root is None:
            parts = self.allgather(sequence)
            return [item for sublist in parts for item in sublist]
        else:
            parts = self.gather(sequence, root=root)
            if parts is None:
                return None
            return [item for sublist in parts for item in sublist]

    def partition(self, total: int, *, rank: int | None = None) -> tuple[int, int]:
        """
        Return a rank's contiguous half-open block of ``total`` items.

        Parameters
        ----------
        total : int
            Total number of items to partition across the world.
        rank : int or None, optional
            Rank to compute the partition block for. If None, uses current rank.

        Returns
        -------
        tuple of int
            Start and stop indices `(start, stop)` for the block.
        """
        return self._comm().partition(total, rank=rank)

    def split(self, sequence: Any) -> Any:
        """
        Return this rank's contiguous slice of a sequence.

        Parameters
        ----------
        sequence : Any
            Sequence to be split.

        Returns
        -------
        Any
            Contiguous slice of the sequence for this rank.
        """
        return self._comm().split(sequence)

    # -- reductions -------------------------------------------------------
    def reduce(
        self,
        value: Any,
        op: Literal["sum", "prod", "min", "max", "any", "all"] = "sum",
    ) -> Any:
        """
        Reduce one value contributed by each rank using ``op``.

        Parameters
        ----------
        value : Any
            The local value to contribute.
        op : {"sum", "prod", "min", "max", "any", "all"}, default "sum"
            The reduction operator to use.

        Returns
        -------
        Any
            The reduced value, identical on every rank.
        """

        return self._comm().reduce(value, op)

    def sum(self, value: Any) -> Any:
        """
        Return the sum of one value contributed by each rank.

        Parameters
        ----------
        value : Any
            The local value to sum.

        Returns
        -------
        Any
            The sum across all ranks.
        """
        return self._comm().sum(value)

    def prod(self, value: Any) -> Any:
        """
        Return the product of one value contributed by each rank.

        Parameters
        ----------
        value : Any
            The local value to multiply.

        Returns
        -------
        Any
            The product across all ranks.
        """
        return self._comm().prod(value)

    def min(self, value: Any) -> Any:
        """
        Return the elementwise minimum over values contributed by all ranks.

        Parameters
        ----------
        value : Any
            The local value to evaluate.

        Returns
        -------
        Any
            The minimum across all ranks.
        """
        return self._comm().min(value)

    def max(self, value: Any) -> Any:
        """
        Return the elementwise maximum over values contributed by all ranks.

        Parameters
        ----------
        value : Any
            The local value to evaluate.

        Returns
        -------
        Any
            The maximum across all ranks.
        """
        return self._comm().max(value)

    def any(self, value: Any) -> Any:
        """
        Return the elementwise logical OR over values contributed by all ranks.

        Parameters
        ----------
        value : Any
            The local value to evaluate.

        Returns
        -------
        Any
            The logical OR across all ranks.
        """
        return self._comm().any(value)

    def all(self, value: Any) -> Any:
        """
        Return the elementwise logical AND over values contributed by all ranks.

        Parameters
        ----------
        value : Any
            The local value to evaluate.

        Returns
        -------
        Any
            The logical AND across all ranks.
        """
        return self._comm().all(value)

    def mean(self, value: Any) -> Any:
        """
        Return the equal-rank arithmetic mean of contributed values.

        Parameters
        ----------
        value : Any
            The local value to evaluate.

        Returns
        -------
        Any
            The mean across all ranks.
        """
        return self._comm().mean(value)


mpi.world = MPIWorldAccessor()


__all__ = ["MPIError", "mpi", "relaunch_with_mpi"]
