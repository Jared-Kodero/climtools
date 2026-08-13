"""ctypes binding for the bundled MPI-NetCDF C library."""

from __future__ import annotations

import atexit
import ctypes
import os
import pickle
from collections.abc import Iterable
from pathlib import Path
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

_default_path = Path(__file__).parent / "lib" / "libmpi_netcdf.so"
_path = Path(os.environ.get("MPI_NETCDF_LIBRARY", _default_path)).expanduser()
try:
    lib = ctypes.CDLL(str(_path))
except OSError as exc:
    raise NativeLibraryError(f"cannot load {_path}: {exc}") from exc

_c_char_p = ctypes.c_char_p
_c_int = ctypes.c_int
_c_ll = ctypes.c_longlong
_c_size = ctypes.c_size_t
_c_void_p = ctypes.c_void_p

lib.mpi_netcdf_init.argtypes = []
lib.mpi_netcdf_init.restype = _c_int
lib.mpi_netcdf_rank.argtypes = []
lib.mpi_netcdf_rank.restype = _c_int
lib.mpi_netcdf_size.argtypes = []
lib.mpi_netcdf_size.restype = _c_int
lib.mpi_netcdf_thread_level.argtypes = []
lib.mpi_netcdf_thread_level.restype = _c_int
lib.mpi_netcdf_consensus.argtypes = [_c_int]
lib.mpi_netcdf_consensus.restype = _c_int
lib.mpi_netcdf_allgather_i64.argtypes = [_c_ll, ctypes.POINTER(_c_ll)]
lib.mpi_netcdf_allgather_i64.restype = _c_int
lib.mpi_netcdf_bcast_i64.argtypes = [ctypes.POINTER(_c_ll), _c_int]
lib.mpi_netcdf_bcast_i64.restype = _c_int
lib.mpi_netcdf_bcast_bytes.argtypes = [_c_void_p, _c_ll, _c_int]
lib.mpi_netcdf_bcast_bytes.restype = _c_int
lib.mpi_netcdf_barrier.argtypes = []
lib.mpi_netcdf_barrier.restype = _c_int
lib.mpi_netcdf_abort.argtypes = [_c_int]
lib.mpi_netcdf_abort.restype = None
lib.mpi_netcdf_finalize.argtypes = []
lib.mpi_netcdf_finalize.restype = _c_int
lib.mpi_netcdf_strerror.argtypes = []
lib.mpi_netcdf_strerror.restype = _c_char_p
lib.mpi_netcdf_version.argtypes = []
lib.mpi_netcdf_version.restype = _c_char_p
lib.mpi_netcdf_has_parallel_filters.argtypes = []
lib.mpi_netcdf_has_parallel_filters.restype = _c_int

lib.mpi_netcdf_create.argtypes = [_c_char_p, _c_int, _c_char_p]
lib.mpi_netcdf_create.restype = _c_void_p
lib.mpi_netcdf_def_dim.argtypes = [_c_void_p, _c_char_p, _c_size, _c_int]
lib.mpi_netcdf_def_dim.restype = _c_int
lib.mpi_netcdf_def_var.argtypes = [
    _c_void_p,
    _c_char_p,
    _c_int,
    _c_int,
    ctypes.POINTER(_c_char_p),
    _c_int,
    _c_int,
    ctypes.POINTER(_c_size),
]
lib.mpi_netcdf_def_var.restype = _c_int
lib.mpi_netcdf_put_att_text.argtypes = [_c_void_p, _c_char_p, _c_char_p, _c_char_p]
lib.mpi_netcdf_put_att_text.restype = _c_int
lib.mpi_netcdf_put_att_num.argtypes = [
    _c_void_p,
    _c_char_p,
    _c_char_p,
    _c_int,
    _c_size,
    _c_void_p,
]
lib.mpi_netcdf_put_att_num.restype = _c_int
lib.mpi_netcdf_enddef.argtypes = [_c_void_p, _c_int]
lib.mpi_netcdf_enddef.restype = _c_int
lib.mpi_netcdf_set_access.argtypes = [_c_void_p, _c_char_p, _c_int]
lib.mpi_netcdf_set_access.restype = _c_int
lib.mpi_netcdf_write.argtypes = [
    _c_void_p,
    _c_char_p,
    ctypes.POINTER(_c_size),
    ctypes.POINTER(_c_size),
    _c_void_p,
]
lib.mpi_netcdf_write.restype = _c_int
lib.mpi_netcdf_close.argtypes = [_c_void_p]
lib.mpi_netcdf_close.restype = _c_int


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
    value = lib.mpi_netcdf_strerror()
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
    check(lib.mpi_netcdf_init(), "MPI initialization")
    return int(lib.mpi_netcdf_rank()), int(lib.mpi_netcdf_size())


def abort(code: int = 1) -> None:
    """Abort ``MPI_COMM_WORLD``.

    Parameters
    ----------
    code : int, optional
        Process exit code supplied to ``MPI_Abort``.
    """
    lib.mpi_netcdf_abort(int(code))


def finalize() -> None:
    """Finalize MPI only when this library initialized it."""
    check(lib.mpi_netcdf_finalize(), "MPI finalization")


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
    out = (_c_ll * size)()
    check(lib.mpi_netcdf_allgather_i64(int(value), out), "MPI all-gather")
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
    rank = int(lib.mpi_netcdf_rank())
    payload = (
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) if rank == root else b""
    )
    n = _c_ll(len(payload))
    check(lib.mpi_netcdf_bcast_i64(ctypes.byref(n), int(root)), "MPI broadcast size")

    if n.value <= 0:
        raise NativeLibraryError(
            f"MPI broadcast reported an invalid payload size: {n.value}."
        )
    buffer = ctypes.create_string_buffer(n.value)
    if rank == root and n.value:
        ctypes.memmove(buffer, payload, n.value)
    if n.value:
        check(
            lib.mpi_netcdf_bcast_bytes(buffer, n.value, int(root)),
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
    return (_c_char_p * len(encoded))(*encoded)


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
    return (_c_size * len(items))(*items)


def _finalize_at_exit() -> None:
    try:
        finalize()
    except NativeLibraryError:
        pass


atexit.register(_finalize_at_exit)
