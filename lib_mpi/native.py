"""ctypes binding for the bundled MPI-NetCDF C library."""

from __future__ import annotations

import atexit
import ctypes
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from .module_env import ModuleLoadError, check_env_stack

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
#: Reason the library could not be loaded, cached so that a failure is
#: diagnosed once instead of on every `available()` call. Loading is
#: deterministic within a process: a missing file, an unloadable object or an
#: absent module system will not become loadable later, so retrying only costs
#: a `dlopen` attempt and a subprocess call to Lmod per query.
_LOAD_ERROR: NativeLibraryError | None = None
#: Rank and size of MPI_COMM_WORLD, cached after the first successful
#: `init()`. MPI_Init_thread runs exactly once per process regardless, but
#: caching keeps `world()` free of a C call on every access.
_WORLD: tuple[int, int] | None = None


#: ABI revision this Python layer is written against. Must match
#: ``MPI_NETCDF_ABI_VERSION`` in ``src/mpi_netcdf.h``.
ABI_VERSION = 2


def library_path() -> Path:
    """Return the location of the compiled native library.

    Returns
    -------
    pathlib.Path
        Path to the compiled ``libmpi_netcdf.so``. ``MPI_NETCDF_LIBRARY``
        overrides the bundled location when set.

    Raises
    ------
    FileNotFoundError
        If the native library has not been built.
    """
    override = os.environ.get("MPI_NETCDF_LIBRARY")
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(
            f"MPI_NETCDF_LIBRARY points at {path}, which is not a file."
        )

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
    # Symbols added after the first release. A library built from older
    # sources still loads; the Python layer falls back to the broadcast loop
    # and reports abi_version() == 1.
    if hasattr(library, "mpi_netcdf_abi_version"):
        library.mpi_netcdf_abi_version.argtypes = []
        library.mpi_netcdf_abi_version.restype = c_int
    if hasattr(library, "mpi_netcdf_allgatherv_bytes"):
        library.mpi_netcdf_allgatherv_bytes.argtypes = [
            c_void_p,
            c_ll,
            c_void_p,
            ctypes.POINTER(c_ll),
        ]
        library.mpi_netcdf_allgatherv_bytes.restype = c_int
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
    global _LIBRARY, _LOAD_ERROR, _PROTOTYPES_APPLIED

    if _LIBRARY is not None:
        return _LIBRARY
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR

    try:
        try:
            check_env_stack()
        except ModuleLoadError as exc:
            raise NativeLibraryError(str(exc)) from exc

        try:
            path = library_path()
        except FileNotFoundError as exc:
            raise NativeLibraryError(str(exc)) from exc

        # Import netCDF4 first, if it is installed at all. The wheel on PyPI
        # bundles its own serial HDF5 and NetCDF-C, and the library below is
        # loaded RTLD_GLOBAL, which Open MPI needs so that its dlopened
        # components resolve. Loading the parallel stack globally first makes
        # the wheel's private HDF5 bind to the parallel symbols already in the
        # global namespace, and the process segfaults on the first call into
        # it. Importing the wheel first pins its own symbols and both stacks
        # coexist. The order is what matters, not which stack "wins".
        # try:
        #     import netCDF4
        # except Exception:  # pragma: no cover - absent or broken install
        #     pass

        try:
            library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise NativeLibraryError(
                f"cannot load {path}: {exc}. Build the extension by running "
                + "lib_mpi/install.sh, or set MPI_NETCDF_LIBRARY."
            ) from exc
    except NativeLibraryError as exc:
        _LOAD_ERROR = exc
        raise

    if not _PROTOTYPES_APPLIED:
        _declare(library)
        _PROTOTYPES_APPLIED = True

    _LIBRARY = library

    return library


def abi_version() -> int:
    """Return the ABI revision of the loaded native library.

    Returns
    -------
    int
        Value reported by the compiled library, or ``1`` for a library built
        before the ABI was versioned.
    """
    library = load()
    if not hasattr(library, "mpi_netcdf_abi_version"):
        return 1
    return int(library.mpi_netcdf_abi_version())


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

    Notes
    -----
    ``MPI_Init_thread`` is called at most once per process, by the C layer.
    The rank and size are cached here as well, so that the hot paths that ask
    for them do not cross the ctypes boundary on every call.
    """
    global _WORLD

    if _WORLD is not None:
        return _WORLD

    library = load()
    check(library.mpi_netcdf_init(), "MPI initialization")
    _WORLD = int(library.mpi_netcdf_rank()), int(library.mpi_netcdf_size())
    return _WORLD


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
    global _WORLD

    check(load().mpi_netcdf_finalize(), "MPI finalization")
    # The world no longer exists. Clearing the cache means a later `init()`
    # asks the C layer again and gets its "already finalized" error, rather
    # than handing back a rank and size that are no longer meaningful.
    _WORLD = None


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
    size = int(library.mpi_netcdf_size())
    if size <= 1:
        return value
    rank = int(library.mpi_netcdf_rank())
    payload = (
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) if rank == root else b""
    )
    return pickle.loads(bcast_bytes(payload, int(root), size))


def allgather_bytes(payload: bytes, size: int) -> list[bytes]:
    """All-gather one variable-length byte payload from every rank.

    Parameters
    ----------
    payload : bytes
        Bytes contributed by the current rank.
    size : int
        Number of ranks in ``MPI_COMM_WORLD``.

    Returns
    -------
    list of bytes
        Payloads from all ranks in rank order.

    Notes
    -----
    One ``MPI_Allgatherv`` when the library supports it, and a per-rank
    broadcast loop otherwise. Both produce rank-ordered output, so any
    reduction built on this is associated identically on every rank and is
    therefore bit-identical everywhere.
    """
    if size < 1:
        raise ValueError(f"size must be positive, got {size}.")
    if size == 1:
        return [payload]

    library = load()
    counts = allgather_i64(len(payload), size)
    total = sum(counts)

    if hasattr(library, "mpi_netcdf_allgatherv_bytes") and total <= 0x7FFFFFFF:
        send = ctypes.create_string_buffer(payload, max(len(payload), 1))
        recv = ctypes.create_string_buffer(max(total, 1))
        count_array = (ctypes.c_longlong * size)(*counts)
        status = library.mpi_netcdf_allgatherv_bytes(
            ctypes.cast(send, ctypes.c_void_p) if payload else None,
            len(payload),
            ctypes.cast(recv, ctypes.c_void_p),
            count_array,
        )
        if status == 0:
            raw = recv.raw
            out: list[bytes] = []
            offset = 0
            for count in counts:
                out.append(raw[offset : offset + count])
                offset += count
            return out
        # A refusal here is a capability limit reported by the C layer, not a
        # communication failure: nothing has been sent, so falling through to
        # the broadcast loop is safe and leaves no rank mid-collective.

    return [bcast_bytes(payload, source, size) for source in range(size)]


def bcast_bytes(payload: bytes, root: int, size: int) -> bytes:
    """Broadcast a byte payload from ``root``.

    Parameters
    ----------
    payload : bytes
        Bytes supplied by the root rank. Ignored on other ranks.
    root : int
        Source rank.
    size : int
        Number of ranks in ``MPI_COMM_WORLD``.

    Returns
    -------
    bytes
        Payload held by the root rank.
    """
    library = load()
    if size == 1:
        return payload

    rank = int(library.mpi_netcdf_rank())
    n = ctypes.c_longlong(len(payload) if rank == root else 0)
    check(
        library.mpi_netcdf_bcast_i64(ctypes.byref(n), int(root)), "MPI broadcast size"
    )
    if n.value < 0:
        raise NativeLibraryError(
            f"MPI broadcast reported an invalid payload size: {n.value}."
        )
    if n.value == 0:
        return b""

    buffer = ctypes.create_string_buffer(n.value)
    if rank == root:
        ctypes.memmove(buffer, payload, n.value)
    check(
        library.mpi_netcdf_bcast_bytes(buffer, n.value, int(root)),
        "MPI broadcast payload",
    )
    return buffer.raw[: n.value]


def allgather_obj(value: Any, size: int) -> list[Any]:
    """All-gather one picklable object from every rank in rank order.

    Parameters
    ----------
    value : Any
        Object contributed by the current rank.
    size : int
        Number of ranks in ``MPI_COMM_WORLD``.

    Returns
    -------
    list of Any
        Objects from all ranks, ordered by rank.
    """
    if size == 1:
        return [value]
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return [pickle.loads(item) for item in allgather_bytes(payload, size)]


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
