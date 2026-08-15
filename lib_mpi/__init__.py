"""MPI runtime layer and native MPI-NetCDF diagnostics.

The public coordination API is :class:`MPI`. Process-wide operations on
``MPI_COMM_WORLD`` are available through ``MPI.world``. Importing this package
does not initialize MPI or load the compiled extension; both remain lazy until
first use.
"""

from __future__ import annotations

from . import native
from .module_env import load_env_stack
from .native import NativeLibraryError
from .runtime import MPIError, mpi


def env_stack():
    import pprint

    pprint.pprint(load_env_stack())


def info() -> dict[str, object]:
    """Return the active native-library and MPI configuration.

    Returns
    -------
    dict of str to object
        Native-library location, NetCDF-C version, ABI revision, world rank and
        size, parallel-filter capability, and granted MPI thread level. Without
        a usable native library, a serial process reports rank 0 of size 1. A
        multi-rank launcher without a usable runtime raises :class:`MPIError`,
        consistent with ``MPI.world``.
    """
    runtime_available = mpi.world.available()
    rank = mpi.world.rank()
    size = mpi.world.size()

    if not runtime_available:
        try:
            location = str(native.library_path())
        except FileNotFoundError as exc:
            location = str(exc).splitlines()[0]
        return {
            "available": False,
            "library": location,
            "netcdf": None,
            "abi": None,
            "abi_expected": native.ABI_VERSION,
            "rank": rank,
            "size": size,
            "parallel_filters": False,
            "thread_level": -1,
        }

    version = native.lib.mpi_netcdf_version()
    return {
        "available": True,
        "library": str(native.library_path()),
        "netcdf": version.decode("utf-8", errors="replace") if version else None,
        "abi": native.abi_version(),
        "abi_expected": native.ABI_VERSION,
        "rank": rank,
        "size": size,
        "parallel_filters": bool(native.lib.mpi_netcdf_has_parallel_filters()),
        "thread_level": int(native.lib.mpi_netcdf_thread_level()),
    }


def has_parallel_filters() -> bool:
    """Return whether parallel NetCDF output supports compression filters.

    Returns
    -------
    bool
        ``True`` when the loaded NetCDF-C/HDF5 stack supports parallel filters;
        otherwise ``False``.
    """
    if not native.available():
        return False
    return bool(native.lib.mpi_netcdf_has_parallel_filters())


__all__ = [
    "MPIError",
    "NativeLibraryError",
    "env_stack",
    "has_parallel_filters",
    "info",
    "mpi",
]
