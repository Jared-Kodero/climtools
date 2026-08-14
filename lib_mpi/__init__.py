"""MPI runtime layer: process coordination and the native C ABI.

This package owns everything that talks to MPI itself. It holds no NetCDF
Python code; the writers live in :mod:`climtools.netcdf`, which calls into the
C ABI exposed here.

Importing this package never initializes MPI and never loads the compiled
extension. Both happen on first use, so ``import climtools`` succeeds on a
machine where ``install.sh`` has not been run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import native, runtime
from .native import NativeLibraryError
from .runtime import MPI, MPIError, available, launcher_size, mpi, world

if TYPE_CHECKING:  # resolved at runtime by the module-level __getattr__
    MPI_RANK: int
    MPI_SIZE: int


def info() -> dict[str, object]:
    """Return the active native-library and MPI configuration.

    Returns
    -------
    dict of str to object
        NetCDF-C version, MPI rank and size, parallel-filter capability and
        the granted MPI thread level. When the extension is not built, the
        dictionary reports ``available=False`` and a one-rank serial world.
    """
    if not native.available():
        return {
            "available": False,
            "library": str(native.library_path()),
            "netcdf": None,
            "rank": 0,
            "size": 1,
            "parallel_filters": False,
            "thread_level": -1,
        }

    rank, size = native.init()
    version = native.lib.mpi_netcdf_version()
    return {
        "available": True,
        "library": str(native.library_path()),
        "netcdf": version.decode("utf-8", errors="replace") if version else None,
        "rank": rank,
        "size": size,
        "parallel_filters": bool(native.lib.mpi_netcdf_has_parallel_filters()),
        "thread_level": int(native.lib.mpi_netcdf_thread_level()),
    }


def has_parallel_filters() -> bool:
    """Report whether compression is usable during a parallel write.

    Returns
    -------
    bool
        ``True`` when NetCDF-C and HDF5 were built with parallel filter
        support. Deflate is unavailable during collective output otherwise.
    """
    if not native.available():
        return False
    return bool(native.lib.mpi_netcdf_has_parallel_filters())


def __getattr__(name: str) -> Any:
    """Resolve ``MPI_RANK`` and ``MPI_SIZE`` without initializing MPI early."""
    if name in ("MPI_RANK", "MPI_SIZE"):
        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MPI",
    "MPI_RANK",
    "MPI_SIZE",
    "MPIError",
    "NativeLibraryError",
    "available",
    "has_parallel_filters",
    "info",
    "launcher_size",
    "mpi",
    "native",
    "runtime",
    "world",
]
