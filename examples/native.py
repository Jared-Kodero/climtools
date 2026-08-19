"""Minimal serial and parallel NetCDF write example, verified by read-back.

``main()`` always runs both demos in one process: the MPI-collective parallel
write first (across every launched rank), then the plain serial write (rank 0
only). Run with a single process for the serial-only path, or under an MPI
launcher to also exercise the collective path::

    python native.py
    mpirun -n 4 python native.py

The parallel writer requires netCDF4-python built against a parallel-enabled
HDF5/netCDF-C (``netCDF4.__has_parallel4_support__``). On a single rank, pass
``allow_serial=True`` to exercise the same code path without a real MPI
launch; see :func:`climtools.xgeo.to_netcdf`.
"""

import os
import time
from pathlib import Path

import numpy as np
import xarray as xr
from climtools import mpi, xgeo


def create_precipitation_dataset() -> xr.Dataset:
    """Return the example (time, lat, lon) precipitation dataset.

    Reads ``CLIMTOOLS_EXAMPLE_NETCDF`` if it is set and points at a readable
    file, otherwise builds a synthetic field so the example runs anywhere.
    """
    source = os.environ.get("CLIMTOOLS_EXAMPLE_NETCDF")
    if source:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"CLIMTOOLS_EXAMPLE_NETCDF is not readable: {path}")
        with xr.open_dataset(path) as handle:
            return handle[["pr"]].load()

    nt, nlat, nlon = 100, 91, 180
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars={
            "pr": (
                ("time", "lat", "lon"),
                rng.random((nt, nlat, nlon)).astype(np.float32),
                {"units": "mm day-1", "long_name": "precipitation rate"},
            ),
        },
        coords={
            "time": (
                "time",
                np.arange(nt, dtype=np.float64),
                {"units": "days since 2000-01-01"},
            ),
            "lat": ("lat", np.linspace(-90.0, 90.0, nlat, dtype=np.float32)),
            "lon": ("lon", np.linspace(0.0, 358.0, nlon, dtype=np.float32)),
        },
        attrs={"title": "climtools NetCDF write example: precipitation"},
    )


def run_serial(out_dir: Path) -> None:
    """Write and verify a serial NetCDF file. Rank 0 only.

    Serial NetCDF I/O is not MPI-collective, so every rank would otherwise
    race to create the same file when this runs under ``mpirun -n N``; the
    rank guard below is what makes "rank 0 only" true rather than aspirational.
    """
    if mpi.comm.rank != 0:
        return

    ds = create_precipitation_dataset()
    path = out_dir / "pr_serial.nc"
    path.unlink(missing_ok=True)

    mpi.log(f"writing {path} serially netcdf4 lib")
    started = time.perf_counter()
    ds.xgeo.to_netcdf(path, unlimited_dim="time", show_progress=False)
    mpi.log(f"serial netcdf4 lib write done in {time.perf_counter() - started:.3f} s")

    with xr.open_dataset(path) as check:
        if not np.array_equal(check["pr"].values, ds["pr"].values):
            raise AssertionError(f"{path}: data mismatch after serial write")
    mpi.log(f"{path}: verified")

    path = out_dir / "pr_xarray.nc"
    path.unlink(missing_ok=True)

    mpi.log(f"writing {path} xarray")
    started = time.perf_counter()
    ds.to_netcdf(path)
    mpi.log(f"serial xarray done in {time.perf_counter() - started:.3f} s")


def run_parallel(out_dir: Path) -> None:
    """Write and verify a NetCDF file with MPI-collective parallel I/O.

    Rank 0 builds the full dataset; every other rank binds an
    :func:`climtools.xgeo.empty_dataset` so the ``.xgeo`` accessor is still
    available. Its contents are unused: the writer scatters rank 0's buffers.
    """
    path = out_dir / "pr_parallel.nc"
    if mpi.comm.rank == 0:
        ds = create_precipitation_dataset()
        path.unlink(missing_ok=True)
    else:
        ds = xgeo.empty_dataset()

    mpi.log(f"writing {path} in parallel across {mpi.comm.size} rank(s)")
    started = time.perf_counter()
    ds.xgeo.to_netcdf(
        path,
        unlimited_dim="time",
        partition_dim="time",
        parallel=True,
        allow_serial=(mpi.comm.size == 1),
    )
    mpi.comm.barrier()
    mpi.log(f"parallel write done in {time.perf_counter() - started:.3f} s")

    if mpi.comm.rank == 0:
        # Reuse the buffers that were written rather than rebuilding the
        # dataset: a second read of an external source is not guaranteed to
        # reproduce the same values, which would make the check unsound.
        reference = ds["pr"].values
        with xr.open_dataset(path) as check:
            if not np.array_equal(check["pr"].values, reference):
                raise AssertionError(f"{path}: data mismatch after parallel write")
        mpi.log(f"{path}: verified")


def main() -> None:

    out_dir = Path.home() / "scratch" / "io_mpi_test"

    out_dir.mkdir(parents=True, exist_ok=True)

    run_parallel(out_dir)

    run_serial(out_dir)


if __name__ == "__main__":
    main()
