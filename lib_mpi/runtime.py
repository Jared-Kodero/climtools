"""MPI runtime and decorators backed by the bundled native C ABI.

The public runtime surface mirrors process-management and collective operations
that are exposed by ``mpi_netcdf.h``. Higher-level collectives that are not
implemented by the native library are intentionally not exposed here.

MPI initialization remains lazy. Without the native extension, a genuine
single-process invocation may execute locally; a launcher reporting multiple
ranks without a usable native runtime raises :class:`MPIError`.
"""

from __future__ import annotations

import ctypes
import functools
import os
import pickle
from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Any, ClassVar, ParamSpec, TypeVar, cast

from . import native

P = ParamSpec("P")
R = TypeVar("R")

if TYPE_CHECKING:
    from typing import Self


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
#: has more than one task, but says nothing about how many.
_LAUNCHER_RANK_VARS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
)


class MPIError(native.NativeLibraryError):
    """MPI runtime or distributed-execution error."""


def _available() -> bool:
    """Return whether the native MPI runtime can load and initialize."""
    if not native.available():
        return False
    try:
        native.init()
    except native.NativeLibraryError:
        return False
    return True


def _launcher_size() -> int:
    """Return the minimum world size implied by launcher environment data."""
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
    """Return ``(rank, size)`` after native initialization when available."""
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


class mpi:
    """Initialize and coordinate calls over ``MPI_COMM_WORLD``.

    The object is both a decorator and a thin public wrapper around the
    process-management and collective calls exposed by the native C ABI.
    Construction does not initialize MPI; initialization occurs on first use.

    Parameters
    ----------
    all_ranks : bool, optional
        Execute a decorated function on every rank. By default, execute only
        on ``root``. Every rank must still call the wrapper.
    broadcast : bool, optional
        Broadcast the root function's picklable return value to every rank.
        Requires ``all_ranks=False``. This is decorator machinery implemented
        privately on the native byte-broadcast primitives.
    root : int, optional
        Root rank used for root-only decorated execution and broadcasting.
    require_ranks : int or None, optional
        Minimum acceptable size of ``MPI_COMM_WORLD``.
    """

    def __new__(cls, function: Any = None, /, **kwargs: Any) -> Any:
        """Support both ``@MPI`` and ``@MPI(...)`` decoration."""
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
        self._rank: int | None = None
        self._size: int | None = None
        self._native = False

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
                + f"minimum ranks ({self.require_ranks})."
            )

        self._rank = rank
        self._size = size
        self._native = native.available()
        return rank, size

    def _require_native(self) -> ctypes.CDLL:
        """Return the loaded native library or raise when it is unavailable."""
        self._resolve()
        if not self._native:
            raise MPIError("the native MPI runtime is unavailable")
        return native.load()

    @property
    def native(self) -> bool:
        """Return whether this coordinator is backed by the native runtime."""
        self._resolve()
        return self._native

    @property
    def is_root(self) -> bool:
        """Return whether this process is the configured root rank."""
        return self.rank == self.root

    world: ClassVar[MPIWorldAccessor]

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

    def __enter__(self) -> Self:
        """Enter the MPI context and return this coordinator."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:  # noqa: PYI036
        self.finalize()

    # -- runtime helpers ---------------------------------------------------
    def log(
        self,
        message: str,
        *args: Any,
        logger: Callable[..., None] = print,
        **kwargs: Any,
    ) -> None:
        """Emit a message from the configured root rank only."""
        if not self.is_root:
            return
        if logger is print and args:
            logger(message % args, **kwargs)
        else:
            logger(message, *args, **kwargs)

    # -- native C process API ---------------------------------------------
    def init(self) -> tuple[int, int]:
        """Initialize MPI if required and return ``(rank, size)``."""
        return self._resolve()

    @property
    def rank(self) -> int:
        """Return the native ``MPI_COMM_WORLD`` rank."""
        return self._resolve()[0]

    @property
    def size(self) -> int:
        """Return the native ``MPI_COMM_WORLD`` size."""
        return self._resolve()[1]

    def thread_level(self) -> int:
        """Return the thread level reported by ``mpi_netcdf_thread_level``."""
        library = self._require_native()
        return int(library.mpi_netcdf_thread_level())

    def consensus(self, ok: bool) -> bool:
        """Return the native all-rank logical consensus."""
        self._resolve()
        if not self._native or self.size == 1:
            return bool(ok)
        return bool(native.lib.mpi_netcdf_consensus(1 if ok else 0))

    def barrier(self) -> None:
        """Execute the native ``MPI_Barrier`` wrapper."""
        self._resolve()
        if not self._native or self.size == 1:
            return
        native.check(native.lib.mpi_netcdf_barrier(), "MPI barrier")

    def allgather_i64(self, value: int) -> list[int]:
        """All-gather one signed 64-bit integer from every rank."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("allgather_i64 value must be an integer.")
        self._resolve()
        if not self._native or self.size == 1:
            return [int(value)]
        return native.allgather_i64(int(value), self.size)

    def allgatherv_bytes(self, payload: bytes) -> list[bytes]:
        """All-gather one variable-length byte payload from every rank."""
        if not isinstance(payload, bytes):
            raise TypeError("allgatherv_bytes payload must be bytes.")
        self._resolve()
        if not self._native or self.size == 1:
            return [payload]
        return native.allgather_bytes(payload, self.size)

    def bcast_i64(self, value: int = 0, *, root: int | None = None) -> int:
        """Broadcast one signed 64-bit integer from ``root``."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("bcast_i64 value must be an integer.")
        source = self._check_root(root)
        self._resolve()
        if not self._native or self.size == 1:
            return int(value)

        cell = ctypes.c_longlong(int(value) if self.rank == source else 0)
        native.check(
            native.lib.mpi_netcdf_bcast_i64(ctypes.byref(cell), source),
            "MPI integer broadcast",
        )
        return int(cell.value)

    def bcast_bytes(self, payload: bytes = b"", *, root: int | None = None) -> bytes:
        """Broadcast a variable-length byte payload from ``root``."""
        if not isinstance(payload, bytes):
            raise TypeError("bcast_bytes payload must be bytes.")
        source = self._check_root(root)
        self._resolve()
        if not self._native or self.size == 1:
            return payload
        return native.bcast_bytes(payload, source, self.size)

    def abort(self, code: int = 1) -> None:
        """Terminate ``MPI_COMM_WORLD`` through ``mpi_netcdf_abort``."""
        self._resolve()
        if not self._native:
            raise SystemExit(code)
        native.abort(code)

    def finalize(self) -> None:
        """Finalize MPI through ``mpi_netcdf_finalize`` when owned here."""
        if not self._native:
            return
        native.finalize()
        self._rank = None
        self._size = None
        self._native = False

    def strerror(self) -> str:
        """Return the last error reported by the native C library."""
        return native.last_error()

    def version(self) -> str:
        """Return the NetCDF-C version reported by ``mpi_netcdf_version``."""
        value = native.load().mpi_netcdf_version()
        return value.decode("utf-8", errors="replace") if value else ""

    def abi_version(self) -> int:
        """Return the version of the exposed native C ABI."""
        return native.abi_version()

    def has_parallel_filters(self) -> bool:
        """Return whether the native library reports parallel filter support."""
        return bool(native.load().mpi_netcdf_has_parallel_filters())

    # -- private decorator transport --------------------------------------
    def _bcast_obj(self, value: Any, *, root: int | None = None) -> Any:
        """Broadcast a pickled object using the exposed byte-broadcast ABI."""
        source = self._check_root(root)
        self._resolve()
        if not self._native or self.size == 1:
            return value
        return native.bcast_obj(value, source)

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
        failed = self.allgather_i64(1 if error is not None else 0)

        try:
            failed_rank = failed.index(1)
        except ValueError:
            return

        if error is not None and all(failed):
            raise error.with_traceback(error.__traceback__)

        detail: tuple[str, str] | None = None
        if self.rank == failed_rank and error is not None:
            detail = (type(error).__name__, str(error))

        detail = cast(
            "tuple[str, str]",
            self._bcast_obj(detail, root=failed_rank),
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
                    pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            except BaseException as exc:
                error = exc

        self._raise_distributed(error)

        if self.broadcast:
            return cast("R", self._bcast_obj(result))
        return result

    def __call__(self, function: Callable[P, R]) -> Callable[P, R | None]:
        """Decorate a function for root-only or all-rank execution."""
        if not callable(function):
            raise TypeError(
                f"MPI can only decorate a callable, got {type(function).__name__}."
            )

        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
            self._resolve()
            if self.all_ranks:
                return self._call_all_ranks(function, args, kwargs)
            return self._call_root(function, args, kwargs)

        wrapper.mpi = self  # type: ignore[attr-defined]
        return wrapper


class MPIWorldAccessor:
    """Lazy namespace exposing only native C runtime operations."""

    __slots__ = ("_coordinator",)

    def __init__(self) -> None:
        self._coordinator: mpi | None = None

    def _comm(self) -> mpi:
        if self._coordinator is None:
            self._coordinator = mpi(all_ranks=True)
        return self._coordinator

    # -- runtime helpers ---------------------------------------------------
    def available(self) -> bool:
        """Return whether the native MPI runtime can load and initialize."""
        return _available()

    def launcher_size(self) -> int:
        """Return the minimum world size implied by launcher metadata."""
        return _launcher_size()

    def is_root(self, root: int = 0) -> bool:
        """Return whether this process has rank ``root``."""
        if isinstance(root, bool) or not isinstance(root, Integral):
            raise TypeError("root must be an integer rank.")
        return self._comm().rank == int(root)

    def log(
        self,
        message: str,
        *args: Any,
        logger: Callable[..., None] = print,
        **kwargs: Any,
    ) -> None:
        """Emit a message from rank zero only."""
        self._comm().log(message, *args, logger=logger, **kwargs)

    # -- native C process API ---------------------------------------------
    def init(self) -> tuple[int, int]:
        """Initialize MPI if required and return ``(rank, size)``."""
        return self._comm().init()

    def rank(self) -> int:
        """Return this process's rank in ``MPI_COMM_WORLD``."""
        return self._comm().rank

    def size(self) -> int:
        """Return the number of ranks in ``MPI_COMM_WORLD``."""
        return self._comm().size

    def thread_level(self) -> int:
        """Return the thread level supplied by the native MPI runtime."""
        return self._comm().thread_level()

    def consensus(self, ok: bool) -> bool:
        """Return true only when every rank contributes true."""
        return self._comm().consensus(ok)

    def barrier(self) -> None:
        """Block until every rank reaches the native barrier."""
        self._comm().barrier()

    def allgather_i64(self, value: int) -> list[int]:
        """All-gather one signed 64-bit integer from every rank."""
        return self._comm().allgather_i64(value)

    def allgatherv_bytes(self, payload: bytes) -> list[bytes]:
        """All-gather one variable-length byte payload from every rank."""
        return self._comm().allgatherv_bytes(payload)

    def bcast_i64(self, value: int = 0, *, root: int = 0) -> int:
        """Broadcast one signed 64-bit integer from ``root``."""
        return self._comm().bcast_i64(value, root=root)

    def bcast_bytes(self, payload: bytes = b"", *, root: int = 0) -> bytes:
        """Broadcast a variable-length byte payload from ``root``."""
        return self._comm().bcast_bytes(payload, root=root)

    def abort(self, code: int = 1) -> None:
        """Abort ``MPI_COMM_WORLD`` through the native C ABI."""
        self._comm().abort(code)

    def finalize(self) -> None:
        """Finalize MPI when initialized by the shared coordinator."""
        if self._coordinator is None:
            return
        self._coordinator.finalize()
        self._coordinator = None

    def strerror(self) -> str:
        """Return the last error reported by the native C library."""
        return self._comm().strerror()

    def version(self) -> str:
        """Return the NetCDF-C version reported by the native C library."""
        return self._comm().version()

    def abi_version(self) -> int:
        """Return the native C ABI version."""
        return self._comm().abi_version()

    def has_parallel_filters(self) -> bool:
        """Return whether parallel NetCDF filters are supported."""
        return self._comm().has_parallel_filters()


mpi.world = MPIWorldAccessor()


__all__ = ["MPIError", "mpi"]
