"""ctypes binding for the bundled MPI-NetCDF C library."""

from __future__ import annotations

import atexit
import ctypes
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from .module_env import ModuleLoadError, ensure_required_modules

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any


class NativeLibraryError(RuntimeError):
    """Error reported by the native MPI-NetCDF library."""


NC_BYTE = 1
NC_CHAR = 2
NC_SHORT = 3
NC_INT = 4
NC_FLOAT = 5
NC_DOUBLE = 6
NC_UBYTE = 7
NC_USHORT = 8
NC_UINT = 9
NC_INT64 = 10
NC_UINT64 = 11

INDEPENDENT = 0
COLLECTIVE = 1

_PROTOTYPES_APPLIED = False
_LIBRARY: ctypes.CDLL | None = None


def library_path() -> Path:
    """Return the location of the compiled native library.

    Returns
    -------
    pathlib.Path
        Path to the compiled ``libmpi_netcdf.so``.

    Raises
    ------
    FileNotFoundError
        If the native library has not been built.
    """
    path = Path(__file__).parent / "lib" / "libmpi_netcdf.so"

    if path.exists():
        return path

    install_script = Path(__file__).parent / "install.sh"
    raise FileNotFoundError(
        f"MPI-NetCDF native library not found: {path}\n Build the library by running:\n{install_script}"
    )


def _declare(library: ctypes.CDLL) -> None:
    """Attach argument and result types to every exported symbol."""
    c_char_p = ctypes.c_char_p
    c_int = ctypes.c_int
    c_ll = ctypes.c_longlong
    c_size = ctypes.c_size_t
    c_void_p = ctypes.c_void_p

    library.mpi_netcdf_init.argtypes = []
    library.mpi_netcdf_init.restype = c_int
    library.mpi_netcdf_rank.argtypes = []
    library.mpi_netcdf_rank.restype = c_int
    library.mpi_netcdf_size.argtypes = []
    library.mpi_netcdf_size.restype = c_int
    library.mpi_netcdf_thread_level.argtypes = []
    library.mpi_netcdf_thread_level.restype = c_int
    library.mpi_netcdf_consensus.argtypes = [c_int]
    library.mpi_netcdf_consensus.restype = c_int
    library.mpi_netcdf_allgather_i64.argtypes = [c_ll, ctypes.POINTER(c_ll)]
    library.mpi_netcdf_allgather_i64.restype = c_int
    library.mpi_netcdf_bcast_i64.argtypes = [ctypes.POINTER(c_ll), c_int]
    library.mpi_netcdf_bcast_i64.restype = c_int
    library.mpi_netcdf_bcast_bytes.argtypes = [c_void_p, c_ll, c_int]
    library.mpi_netcdf_bcast_bytes.restype = c_int
    library.mpi_netcdf_barrier.argtypes = []
    library.mpi_netcdf_barrier.restype = c_int
    library.mpi_netcdf_abort.argtypes = [c_int]
    library.mpi_netcdf_abort.restype = None
    library.mpi_netcdf_finalize.argtypes = []
    library.mpi_netcdf_finalize.restype = c_int
    library.mpi_netcdf_strerror.argtypes = []
    library.mpi_netcdf_strerror.restype = c_char_p
    library.mpi_netcdf_version.argtypes = []
    library.mpi_netcdf_version.restype = c_char_p
    library.mpi_netcdf_has_parallel_filters.argtypes = []
    library.mpi_netcdf_has_parallel_filters.restype = c_int

    library.mpi_netcdf_create.argtypes = [c_char_p, c_int, c_char_p]
    library.mpi_netcdf_create.restype = c_void_p
    library.mpi_netcdf_def_dim.argtypes = [c_void_p, c_char_p, c_size, c_int]
    library.mpi_netcdf_def_dim.restype = c_int
    library.mpi_netcdf_def_var.argtypes = [
        c_void_p,
        c_char_p,
        c_int,
        c_int,
        ctypes.POINTER(c_char_p),
        c_int,
        c_int,
        ctypes.POINTER(c_size),
    ]
    library.mpi_netcdf_def_var.restype = c_int
    library.mpi_netcdf_put_att_text.argtypes = [
        c_void_p,
        c_char_p,
        c_char_p,
        c_char_p,
    ]
    library.mpi_netcdf_put_att_text.restype = c_int
    library.mpi_netcdf_put_att_num.argtypes = [
        c_void_p,
        c_char_p,
        c_char_p,
        c_int,
        c_size,
        c_void_p,
    ]
    library.mpi_netcdf_put_att_num.restype = c_int
    library.mpi_netcdf_enddef.argtypes = [c_void_p, c_int]
    library.mpi_netcdf_enddef.restype = c_int
    library.mpi_netcdf_set_access.argtypes = [c_void_p, c_char_p, c_int]
    library.mpi_netcdf_set_access.restype = c_int
    library.mpi_netcdf_write.argtypes = [
        c_void_p,
        c_char_p,
        ctypes.POINTER(c_size),
        ctypes.POINTER(c_size),
        c_void_p,
    ]
    library.mpi_netcdf_write.restype = c_int
    library.mpi_netcdf_close.argtypes = [c_void_p]
    library.mpi_netcdf_close.restype = c_int


def load() -> ctypes.CDLL:
    """Load the native library, configuring its prototypes once.

    Returns
    -------
    ctypes.CDLL
        Handle to ``libmpi_netcdf.so``.

    Raises
    ------
    NativeLibraryError
        If the library is absent or cannot be loaded. Build it with
        ``lib_mpi/install.sh``.
    """
    global _LIBRARY, _PROTOTYPES_APPLIED

    if _LIBRARY is not None:
        return _LIBRARY

    try:
        ensure_required_modules()
    except ModuleLoadError as exc:
        raise NativeLibraryError(str(exc)) from exc

    path = library_path()
    try:
        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise NativeLibraryError(
            f"cannot load {path}: {exc}. Build the extension by running "
            + "lib_mpi/install.sh, or set MPI_NETCDF_LIBRARY."
        ) from exc

    if not _PROTOTYPES_APPLIED:
        _declare(library)
        _PROTOTYPES_APPLIED = True

    _LIBRARY = library

    return library


def available() -> bool:
    """Report whether the native library can be loaded in this process.

    Returns
    -------
    bool
        ``True`` when the compiled extension is importable.
    """
    try:
        load()
    except NativeLibraryError:
        return False
    return True


def __getattr__(name: str) -> Any:
    """Expose ``native.lib`` without loading the library at import time."""
    if name == "lib":
        return load()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def b(value: str | bytes) -> bytes:
    """Encode a Python string for the C ABI.

    Parameters
    ----------
    value : str or bytes
        Value to pass through or encode as UTF-8.

    Returns
    -------
    bytes
        Byte representation accepted by the C ABI.
    """
    return value if isinstance(value, bytes) else value.encode("utf-8")


def last_error() -> str:
    """Return the last C-library error for this process.

    Returns
    -------
    str
        Error text reported by the native library.
    """
    value = load().mpi_netcdf_strerror()
    return (
        value.decode("utf-8", errors="replace") if value else "unknown MPI-NetCDF error"
    )


def check(status: int, operation: str) -> None:
    """Raise an exception when a native operation fails.

    Parameters
    ----------
    status : int
        Native status code. Negative values indicate failure.
    operation : str
        Human-readable operation name used in the exception message.

    Raises
    ------
    NativeLibraryError
        If ``status`` is negative.
    """
    if status < 0:
        raise NativeLibraryError(f"{operation} failed: {last_error()}")


def init() -> tuple[int, int]:
    """Initialize MPI if required.

    Returns
    -------
    tuple of int
        Rank and size of ``MPI_COMM_WORLD``.
    """
    library = load()
    check(library.mpi_netcdf_init(), "MPI initialization")
    return int(library.mpi_netcdf_rank()), int(library.mpi_netcdf_size())


def abort(code: int = 1) -> None:
    """Abort ``MPI_COMM_WORLD``.

    Parameters
    ----------
    code : int, optional
        Process exit code supplied to ``MPI_Abort``.
    """
    load().mpi_netcdf_abort(int(code))


def finalize() -> None:
    """Finalize MPI only when this library initialized it."""
    check(load().mpi_netcdf_finalize(), "MPI finalization")


def allgather_i64(value: int, size: int) -> list[int]:
    """All-gather one signed 64-bit integer from every rank.

    Parameters
    ----------
    value : int
        Integer contributed by the current rank.
    size : int
        Number of ranks in ``MPI_COMM_WORLD``.

    Returns
    -------
    list of int
        Values contributed by all ranks in rank order.
    """
    if size < 1:
        raise ValueError(f"size must be positive, got {size}.")
    library = load()
    out = (ctypes.c_longlong * size)()
    check(library.mpi_netcdf_allgather_i64(int(value), out), "MPI all-gather")
    return [int(item) for item in out]


def bcast_obj(value: Any, root: int) -> Any:
    """Broadcast a picklable Python object from a root rank.

    Parameters
    ----------
    value : Any
        Object supplied by the root rank. Values from other ranks are ignored.
    root : int
        Source rank.

    Returns
    -------
    Any
        Object broadcast by the source rank.
    """
    library = load()
    rank = int(library.mpi_netcdf_rank())
    payload = (
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) if rank == root else b""
    )
    n = ctypes.c_longlong(len(payload))
    check(
        library.mpi_netcdf_bcast_i64(ctypes.byref(n), int(root)),
        "MPI broadcast size",
    )

    if n.value <= 0:
        raise NativeLibraryError(
            f"MPI broadcast reported an invalid payload size: {n.value}."
        )
    buffer = ctypes.create_string_buffer(n.value)
    if rank == root and n.value:
        ctypes.memmove(buffer, payload, n.value)
    if n.value:
        check(
            library.mpi_netcdf_bcast_bytes(buffer, n.value, int(root)),
            "MPI broadcast payload",
        )
    return pickle.loads(buffer.raw[: n.value])


def str_array(values: Iterable[str]) -> Any:
    """Create a temporary ``char **`` array for a C call.

    Parameters
    ----------
    values : iterable of str
        Strings to encode as UTF-8.

    Returns
    -------
    ctypes.Array
        Null-terminated C string pointer array.
    """
    encoded = [b(value) for value in values]
    return (ctypes.c_char_p * len(encoded))(*encoded)


def size_array(values: Iterable[int]) -> Any:
    """Create a temporary ``size_t *`` array for a C call.

    Parameters
    ----------
    values : iterable of int
        Integer values to convert to ``size_t``.

    Returns
    -------
    ctypes.Array
        C ``size_t`` array.
    """
    items = [int(value) for value in values]
    if any(value < 0 for value in items):
        raise ValueError("size_t values cannot be negative.")
    return (ctypes.c_size_t * len(items))(*items)


def _finalize_at_exit() -> None:
    if _LIBRARY is None:
        return
    try:
        finalize()
    except NativeLibraryError:
        pass


atexit.register(_finalize_at_exit)
