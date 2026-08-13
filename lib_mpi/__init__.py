"""MPI-parallel NetCDF-4 output for xarray."""

from __future__ import annotations

from . import native
from .native import NativeLibraryError
from .parallel_netcdf import InconsistentRanksError, to_netcdf
from .runtime import MPI, MPI_RANK, MPI_SIZE, MPIError, mpi


def info() -> dict[str, object]:
    """Return the active native-library and MPI configuration.

    Returns
    -------
    dict of str to object
        NetCDF-C version, MPI rank and size, and parallel-filter capability.
    """
    rank, size = native.init()
    version = native.lib.mpi_netcdf_version()
    return {
        "netcdf": version.decode("utf-8", errors="replace") if version else None,
        "rank": rank,
        "size": size,
        "parallel_filters": bool(native.lib.mpi_netcdf_has_parallel_filters()),
        "thread_level": int(native.lib.mpi_netcdf_thread_level()),
    }


__all__ = [
    "MPI",
    "MPI_RANK",
    "MPI_SIZE",
    "InconsistentRanksError",
    "MPIError",
    "NativeLibraryError",
    "info",
    "mpi",
    "to_netcdf",
]
