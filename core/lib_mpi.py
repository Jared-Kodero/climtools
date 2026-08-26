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
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from numbers import Integral
from typing import Any, ParamSpec, TypeVar, cast

from mpi4py import MPI as _MPI
from mpi4py.MPI import Intracomm

from .tools import LockFile, tmp
from .xr_mpi import XarrayMPI

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

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


class ToChildrenRuntime:
    """Parent-to-child-group communication namespace.

    Accessed through :attr:`MPIRuntime.to_children` after
    :meth:`MPIRuntime.decompose`. Operations originate on a parent rank and
    deliver values to the logical child communicators.

    Parameters
    ----------
    runtime : MPIRuntime
        Parent runtime owning the decomposed child communicators.
    """

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime

    def broadcast(self, value: T | None, *, root: int = 0) -> T:
        """Broadcast one parent-root value to every rank in every child.

        Parameters
        ----------
        value : T or None
            Value on the parent ``root`` rank. Non-root ranks may pass None.
        root : int, optional
            Source rank in the parent communicator. Default 0.

        Returns
        -------
        T
            Broadcast value on every rank in every child communicator.

        Raises
        ------
        RuntimeError
            If :meth:`MPIRuntime.decompose` has not been called.
        ValueError
            If ``root`` is not a valid parent rank.
        MPIError
            If ranks disagree on the collective call.
        """
        runtime = self._runtime
        _ = runtime.child
        comm = runtime.comm

        error: BaseException | None = None
        signature: tuple[str, int] | None = None
        try:
            if isinstance(root, bool) or not isinstance(root, Integral):
                raise TypeError("root must be a non-negative integer rank.")
            if root < 0 or root >= comm.size:
                raise ValueError(f"root {root} is outside [0, {comm.size}).")
            signature = ("to_children.broadcast", int(root))
        except BaseException as exc:
            error = exc

        runtime.raise_if_error(
            error,
            "mpi.to_children.broadcast",
            signature=signature,
        )
        return cast(
            "T",
            comm.bcast(value if runtime.is_root(root) else None, root=root),
        )

    def scatter(self, values: Sequence[T] | None, *, root: int = 0) -> T:
        """Scatter one parent-root value per child and replicate within it.

        ``root`` supplies exactly one value per child communicator. The
        parent communicator sends each value to that child's leader, which
        then broadcasts it within the child.

        Parameters
        ----------
        values : sequence of T or None
            One value per child on the parent ``root`` rank. Non-root ranks
            may pass None.
        root : int, optional
            Source rank in the parent communicator. Default 0.

        Returns
        -------
        T
            Value assigned to this rank's child; identical on every rank in
            that child.

        Raises
        ------
        RuntimeError
            If :meth:`MPIRuntime.decompose` has not been called.
        ValueError
            If ``root`` is invalid, ``values`` is None on ``root``, or the
            number of values does not equal the number of children.
        MPIError
            If ranks disagree on the collective call or root-side validation
            fails on only a subset of ranks.
        """
        runtime = self._runtime
        child = runtime.child
        comm = runtime.comm
        ranks_per_child = child.comm.size
        ntasks = comm.size // ranks_per_child

        error: BaseException | None = None
        signature: tuple[str, int, int] | None = None
        send: list[T | None] | None = None

        try:
            if isinstance(root, bool) or not isinstance(root, Integral):
                raise TypeError("root must be a non-negative integer rank.")
            if root < 0 or root >= comm.size:
                raise ValueError(f"root {root} is outside [0, {comm.size}).")

            if runtime.is_root(root):
                if values is None:
                    raise ValueError("values cannot be None on the parent root.")
                if len(values) != ntasks:
                    raise ValueError(
                        f"values must contain exactly {ntasks} items, one per "
                        + f"child; got {len(values)}."
                    )

                send = [None] * comm.size
                for task, value in enumerate(values):
                    send[task * ranks_per_child] = value

            signature = ("to_children.scatter", int(root), ntasks)
        except BaseException as exc:
            error = exc

        runtime.raise_if_error(
            error,
            "mpi.to_children.scatter",
            signature=signature,
        )

        received = comm.scatter(send, root=root)
        return cast(
            "T",
            child.comm.bcast(
                received if child.is_root() else None,
                root=0,
            ),
        )


class FromChildrenRuntime:
    """Child-group-to-parent communication namespace.

    Accessed through :attr:`MPIRuntime.from_children` after
    :meth:`MPIRuntime.decompose`. Operations collect one logical result from
    each child communicator back onto the parent communicator.

    Parameters
    ----------
    runtime : MPIRuntime
        Parent runtime owning the decomposed child communicators.
    """

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime

    def gather(self, value: T, *, root: int = 0) -> list[T] | None:
        """Gather one child result onto one parent rank.

        Only rank zero of each child contributes ``value`` to the parent
        gather. Results on ``root`` are ordered by child task index.

        Parameters
        ----------
        value : T
            This child's result. Only the child leader contributes it.
        root : int, optional
            Destination rank in the parent communicator. Default 0.

        Returns
        -------
        list of T or None
            One value per child, ordered by child task index, on ``root``.
            None on every other parent rank.

        Raises
        ------
        RuntimeError
            If :meth:`MPIRuntime.decompose` has not been called.
        ValueError
            If ``root`` is not a valid parent rank.
        MPIError
            If ranks disagree on the collective call.
        """
        runtime = self._runtime
        child = runtime.child
        comm = runtime.comm

        error: BaseException | None = None
        signature: tuple[str, int] | None = None
        try:
            if isinstance(root, bool) or not isinstance(root, Integral):
                raise TypeError("root must be a non-negative integer rank.")
            if root < 0 or root >= comm.size:
                raise ValueError(f"root {root} is outside [0, {comm.size}).")
            signature = ("from_children.gather", int(root))
        except BaseException as exc:
            error = exc

        runtime.raise_if_error(error, "mpi.from_children.gather", signature=signature)

        payload = (cast("int", child.task), value) if child.is_root() else None
        gathered = comm.gather(payload, root=root)

        if not runtime.is_root(root):
            return None

        results = [item for item in gathered if item is not None]
        results.sort(key=lambda item: item[0])
        return [item[1] for item in results]


class MPIRuntime:
    """User-facing MPI runtime namespace.

    Owns one intracommunicator, exposed directly through :attr:`comm`.
    MPI-aware xarray operations, including distributed reductions, live
    under :attr:`xarray`.

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
        self._xarray: XarrayMPI = XarrayMPI(self)
        self._mpi_lock = LockFile(tmp / ".mpi.lock")
        self._child: MPIRuntime | None = None
        self._to_children = ToChildrenRuntime(self)
        self._from_children = FromChildrenRuntime(self)
        self.info: tuple[int, ...] = ()
        self.task: int | None = None
        self._install_abort_hook()
        self._warn_if_parallel_netcdf_missing()

    @property
    def xarray(self) -> XarrayMPI:
        """XarrayMPI: MPI-aware xarray indexing, redistribution, reductions."""
        return self._xarray

    @property
    def to_children(self) -> ToChildrenRuntime:
        """ToChildrenRuntime: Parent-to-child-group communication namespace."""
        return self._to_children

    @property
    def from_children(self) -> FromChildrenRuntime:
        """FromChildrenRuntime: Child-group-to-parent communication namespace."""
        return self._from_children

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

    @property
    def launched(self) -> bool:
        """bool: Whether this process appears to have been launched by MPI."""
        return self.alive(self.comm)

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

    # -------------------------------------------------------------------------
    # Child communicator decomposition
    # -------------------------------------------------------------------------

    def decompose(self, ntasks: int) -> None:
        """Split the communicator into ``ntasks`` equally sized children.

        Every rank belongs to exactly one child, available through
        :attr:`child`; ``child.task`` is the zero-based child index and
        ``child.info`` holds the parent ranks belonging to it. The child is a
        complete :class:`MPIRuntime`, so ordinary operations such as
        ``child.log()``, ``child.broadcast()``, and ``child.xarray`` operate
        within that child communicator.

        Parent-to-child-group communication is exposed through
        :attr:`to_children`; child-group-to-parent communication is exposed
        through :attr:`from_children`. All ranks in the current communicator
        must call this method with the same ``ntasks`` value.

        Parameters
        ----------
        ntasks : int
            Number of child communicators. Must evenly divide the current
            communicator size.

        Raises
        ------
        TypeError
            If ``ntasks`` is not an integer.
        ValueError
            If ``ntasks`` is less than one, exceeds the communicator size,
            or does not evenly divide it.
        MPIError
            If ranks call this method with different valid ``ntasks`` values,
            or validation fails on only a subset of ranks.
        """
        comm = self.comm
        error: BaseException | None = None
        signature: tuple[str, int] | None = None

        try:
            if isinstance(ntasks, bool) or not isinstance(ntasks, Integral):
                raise TypeError("ntasks must be an integer.")
            if ntasks < 1:
                raise ValueError("ntasks must be at least 1.")
            if ntasks > comm.size:
                raise ValueError(
                    f"ntasks ({ntasks}) cannot exceed MPI size ({comm.size})."
                )
            if comm.size % ntasks != 0:
                raise ValueError(
                    f"MPI size ({comm.size}) must be divisible by ntasks ({ntasks})."
                )
            signature = ("decompose", int(ntasks))
        except BaseException as exc:
            error = exc

        self.raise_if_error(error, "mpi.decompose", signature=signature)

        ranks_per_task = comm.size // ntasks
        task = comm.rank // ranks_per_task
        start = task * ranks_per_task
        stop = start + ranks_per_task

        child_comm = comm.Split(color=task, key=comm.rank)
        child = MPIRuntime(child_comm)
        child.task = task
        child.info = tuple(range(start, stop))
        self._child = child

    # -------------------------------------------------------------------------
    # Point-to-point and collective object communication
    # -------------------------------------------------------------------------

    def send(self, value: T, dest: int, *, tag: int = 0) -> None:
        """Send a Python object to one rank.

        Parameters
        ----------
        value : T
            Object to send.
        dest : int
            Destination rank.
        tag : int, optional
            MPI message tag. Default 0.
        """
        self.comm.send(value, dest=dest, tag=tag)

    def receive(
        self,
        source: int = _MPI.ANY_SOURCE,
        *,
        tag: int = _MPI.ANY_TAG,
    ) -> T:
        """Receive a Python object from one rank.

        Parameters
        ----------
        source : int, optional
            Source rank. Default ``MPI.ANY_SOURCE``.
        tag : int, optional
            MPI message tag. Default ``MPI.ANY_TAG``.

        Returns
        -------
        T
            Received object.
        """
        return cast("T", self.comm.recv(source=source, tag=tag))

    def broadcast(self, value: T | None, *, root: int = 0) -> T:
        """Broadcast one Python object from ``root`` to every rank.

        Parameters
        ----------
        value : T or None
            Object on ``root``. Non-root ranks may pass None.
        root : int, optional
            Broadcasting rank. Default 0.

        Returns
        -------
        T
            Broadcast object on every rank.
        """
        return cast("T", self.comm.bcast(value, root=root))

    def scatter(self, values: Sequence[T] | None, *, root: int = 0) -> T:
        """Scatter one Python object from ``root`` to each rank.

        Parameters
        ----------
        values : sequence of T or None
            One value per rank on ``root``. Non-root ranks may pass None.
        root : int, optional
            Source rank. Default 0.

        Returns
        -------
        T
            Object assigned to this rank.
        """
        return cast("T", self.comm.scatter(values, root=root))

    def gather(self, value: T, *, root: int = 0) -> list[T] | None:
        """Gather one Python object from every rank onto ``root``.

        Parameters
        ----------
        value : T
            This rank's contribution.
        root : int, optional
            Destination rank. Default 0.

        Returns
        -------
        list of T or None
            Rank-ordered values on ``root``; None on all other ranks.
        """
        return cast("list[T] | None", self.comm.gather(value, root=root))

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
            + f"this {comm.Get_size()}-rank run. mpi.xarray and the rest of the "
            + "MPI runtime are unaffected. To build the parallel stack, "
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
