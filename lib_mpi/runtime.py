"""MPI setup and decorators for functions executed by an existing MPI job.

This module never starts processes. The caller must launch Python with
``srun`` or ``mpirun``. Importing the module initializes MPI in the current
process through :mod:`mpi.native` and publishes the immutable world rank and
size.
"""

from __future__ import annotations

import functools
import pickle
from collections.abc import Callable
from numbers import Integral
from typing import Any, ParamSpec, TypeVar, cast

from . import native

P = ParamSpec("P")
R = TypeVar("R")


class MPIError(native.NativeLibraryError):
    """An error propagated from another rank."""


MPI_RANK, MPI_SIZE = native.init()


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

    Notes
    -----
    Every rank must call a decorated function in the same order because the
    wrapper performs collective MPI communication for exception propagation
    and optional result broadcasting.
    """

    MPI_RANK = MPI_RANK
    MPI_SIZE = MPI_SIZE

    def __init__(
        self,
        *,
        all_ranks: bool = False,
        broadcast: bool = False,
        root: int = 0,
        require_ranks: int | None = None,
    ) -> None:
        global MPI_RANK, MPI_SIZE

        rank, size = native.init()
        MPI_RANK, MPI_SIZE = rank, size
        type(self).MPI_RANK = rank
        type(self).MPI_SIZE = size

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

    def barrier(self) -> None:
        """Wait until every rank reaches this call."""
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
        return native.bcast_obj(value, source)

    def abort(self, code: int = 1) -> None:
        """Terminate all ranks in ``MPI_COMM_WORLD``.

        Parameters
        ----------
        code : int, optional
            Error code supplied to the MPI abort operation.
        """
        native.abort(code)

    def finalize(self) -> None:
        """Finalize MPI if this package initialized it."""
        native.finalize()

    def _raise_distributed(self, error: BaseException | None) -> None:
        failed = self.allgather(1 if error is not None else 0)

        try:
            failed_rank = failed.index(1)
        except ValueError:
            return

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


def mpi(
    all_ranks: bool = False,
    broadcast: bool = False,
    root: int = 0,
    require_ranks: int | None = None,
) -> MPI:
    """Create an MPI function decorator.

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

    Returns
    -------
    MPI
        Configured MPI decorator.

    Raises
    ------
    ValueError
        If ``root`` is outside ``MPI_COMM_WORLD``, if ``broadcast=True`` is
        combined with ``all_ranks=True``, or if ``require_ranks`` is less
        than one.
    MPIError
        If ``MPI_COMM_WORLD`` contains fewer than ``require_ranks`` ranks.

    Notes
    -----
    Every rank must call the decorated function in the same order because
    exception propagation and optional broadcasting use collective MPI
    operations.

    Examples
    --------
    Execute only on the default root rank:

    >>> @mpi()
    ... def write_output() -> None:
    ...     pass

    Execute on every rank:

    >>> @mpi(all_ranks=True)
    ... def compute() -> int:
    ...     return MPI_RANK

    Execute on the root and broadcast the result to every rank:

    >>> @mpi(broadcast=True)
    ... def configuration() -> dict[str, int]:
    ...     return {"size": MPI_SIZE}

    Use a different root rank:

    >>> @mpi(root=1)
    ... def write_from_rank_one() -> None:
    ...     pass
    """
    return MPI(
        all_ranks=all_ranks,
        broadcast=broadcast,
        root=root,
        require_ranks=require_ranks,
    )


__all__ = ["MPI", "MPI_RANK", "MPI_SIZE", "MPIError", "mpi"]
