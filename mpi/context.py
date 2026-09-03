"""Small user-facing MPI namespace built on :mod:`mpi4py`."""

# mpi.py
from __future__ import annotations

import builtins
import functools
import os
from collections.abc import Callable, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

from mpi4py import MPI

from ..core.utils import _LAUNCH_ENV, LockFile, tmp
from .diagnostics import MPIDiagnostics, MPIError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class ToChildrenContext:
    """Parent-to-child-group communication namespace.

    Accessed through :attr:`MPIContext.to_children` after
    :meth:`MPIContext.decompose`. Operations originate on a parent rank and
    deliver values to the logical child communicators.

    Parameters
    ----------
    mpi_context : MPIContext
        Parent mpi_context owning the decomposed child communicators.
    """

    def __init__(self, mpi_context: MPIContext) -> None:
        self._runtime = mpi_context

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
            If :meth:`MPIContext.decompose` has not been called.
        ValueError
            If ``root`` is not a valid parent rank.
        MPIError
            If ranks disagree on the collective call.
        """
        mpi_context = self._runtime
        _ = mpi_context.child
        comm = mpi_context.comm

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

        mpi_context.raise_if_error(
            error,
            "mpi.to_children.broadcast",
            signature=signature,
        )
        return cast(
            "T",
            comm.bcast(value if mpi_context.is_root(root) else None, root=root),
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
            If :meth:`MPIContext.decompose` has not been called.
        ValueError
            If ``root`` is invalid, ``values`` is None on ``root``, or the
            number of values does not equal the number of children.
        MPIError
            If ranks disagree on the collective call or root-side validation
            fails on only a subset of ranks.
        """
        mpi_context = self._runtime
        child = mpi_context.child
        comm = mpi_context.comm
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

            if mpi_context.is_root(root):
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

        mpi_context.raise_if_error(
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


class FromChildrenContext:
    """Child-group-to-parent communication namespace.

    Accessed through :attr:`MPIContext.from_children` after
    :meth:`MPIContext.decompose`. Operations collect one logical result from
    each child communicator back onto the parent communicator.

    Parameters
    ----------
    mpi_context : MPIContext
        Parent mpi_context owning the decomposed child communicators.
    """

    def __init__(self, mpi_context: MPIContext) -> None:
        self._runtime = mpi_context

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
            If :meth:`MPIContext.decompose` has not been called.
        ValueError
            If ``root`` is not a valid parent rank.
        MPIError
            If ranks disagree on the collective call.
        """
        mpi_context = self._runtime
        child = mpi_context.child
        comm = mpi_context.comm

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

        mpi_context.raise_if_error(
            error, "mpi.from_children.gather", signature=signature
        )

        payload = (cast("int", child.task), value) if child.is_root() else None
        gathered = comm.gather(payload, root=root)

        if not mpi_context.is_root(root):
            return None

        results = [item for item in gathered if item is not None]
        results.sort(key=lambda item: item[0])
        return [item[1] for item in results]


class MPIContext(MPIDiagnostics):
    """User-facing MPI context  namespace.

    Owns one intracommunicator, exposed directly through :attr:`comm`.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm or None, optional
        Communicator used by the mpi_context. None uses ``MPI.COMM_WORLD``.
    """

    MPIError = MPIError

    # -------------------------------------------------------------------------
    # Runtime state and accessors
    # -------------------------------------------------------------------------

    def __init__(self, comm: MPI.Intracomm | None = None) -> None:
        self.comm: MPI.Intracomm = comm if comm is not None else MPI.COMM_WORLD
        self._mpi_lock = LockFile(tmp / ".mpi.lock")
        self._child: MPIContext | None = None
        self._to_children = ToChildrenContext(self)
        self._from_children = FromChildrenContext(self)
        self.info: tuple[int, ...] = ()
        self.task: int | None = None
        self._install_abort_hook()
        self.missing_pnetcdf()

    @property
    def to_children(self) -> ToChildrenContext:
        """ToChildrenContext: Parent-to-child-group communication namespace."""
        return self._to_children

    @property
    def from_children(self) -> FromChildrenContext:
        """FromChildrenContext: Child-group-to-parent communication namespace."""
        return self._from_children

    @property
    def child(self) -> MPIContext:
        """MPIContext: Runtime for this rank's child communicator.

        Raises
        ------
        RuntimeError
            If :meth:`decompose` has not been called.
        """
        if self._child is None:
            raise RuntimeError("MPI context has not been decomposed.")

        return self._child

    @property
    def launched(self) -> bool:
        """bool: Whether this process appears to have been launched by MPI."""
        return self.alive(self.comm)

    @staticmethod
    def alive(comm: MPI.Comm) -> bool:
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
            return MPI.Comm.Get_parent() != MPI.COMM_NULL
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
        complete :class:`MPIContext`, so ordinary operations such as
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
        child = MPIContext(child_comm)
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
        source: int = MPI.ANY_SOURCE,
        *,
        tag: int = MPI.ANY_TAG,
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

    def scatterv(
        self,
        send: np.ndarray | None,
        counts: Sequence[int] | np.ndarray,
        local_shape: Sequence[int],
        dtype: npt.DTypeLike,
        *,
        root: int = 0,
    ) -> np.ndarray:
        """Scatter an array's leading axis with a variable per-rank count.

        Zero-copy counterpart to :meth:`scatter`: rather than one Python
        object per rank, this splits ``send``'s axis 0 into ``counts[rank]``
        rows per rank via ``MPI_Scatterv`` and returns each rank's
        contiguous local slab directly as a NumPy array, without pickling.
        Use this whenever every rank's row count may differ (e.g. an
        uneven partition), unlike :meth:`scatter`, whose one-Python-object-
        per-rank contract has no notion of a shared leading axis to split.

        Parameters
        ----------
        send : numpy.ndarray or None
            Complete array on ``root``, axis 0 already ordered rank by
            rank (row ``i`` for ``sum(counts[:r]) <= i < sum(counts[:r+1])``
            belongs to rank ``r``). Ignored (may be None) on every other
            rank.
        counts : sequence of int or numpy.ndarray
            Row count along axis 0 assigned to each rank, ``counts[rank]``
            summing to ``send.shape[0]``. Every rank must pass the
            identical ``counts``.
        local_shape : sequence of int
            This rank's output shape; ``local_shape[0]`` must equal
            ``counts[self.comm.rank]`` and the trailing dimensions must
            match ``send.shape[1:]``.
        dtype : numpy dtype-like
            Element dtype of ``send`` and the returned array. Every rank
            must agree on this dtype; :func:`~mpi4py.util.dtlib.from_numpy_dtype`
            maps it to the matching ``MPI.Datatype`` for the transfer, so
            an unsupported (e.g. non-numeric) dtype raises there before
            any communication happens.
        root : int, optional
            Rank holding ``send``. Default 0.

        Returns
        -------
        numpy.ndarray
            This rank's contiguous local slab, shape ``local_shape``.

        Raises
        ------
        ValueError
            If ``root`` is asked to scatter without providing ``send``.
        """
        import numpy as np
        from mpi4py.util import dtlib

        dtype = np.dtype(dtype)
        counts = np.asarray(counts, dtype=np.int64)
        row_elems = (
            int(np.prod(local_shape[1:], dtype=np.int64)) if len(local_shape) > 1 else 1
        )
        elem_counts = counts * row_elems
        displs = np.zeros(len(counts), dtype=np.int64)
        np.cumsum(elem_counts[:-1], out=displs[1:])

        mpi_dtype = dtlib.from_numpy_dtype(dtype)
        local = np.empty(local_shape, dtype=dtype)

        sendbuf = None
        if self.comm.rank == root:
            if send is None:
                raise ValueError("`send` is required on the scattering root rank.")
            contiguous = np.ascontiguousarray(send, dtype=dtype)
            sendbuf = [contiguous, (elem_counts, displs), mpi_dtype]

        self.comm.Scatterv(sendbuf, [local, mpi_dtype], root=root)
        return local

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


mpi: MPIContext = MPIContext()


__all__ = ["mpi"]
