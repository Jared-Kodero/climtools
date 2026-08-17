"""Minimal serial and parallel NetCDF write example, verified by read-back.

Run serially::

    python native.py

Run with MPI-collective parallel output::

    mpirun -n 4 python native.py --parallel

The parallel writer requires netCDF4-python built against a parallel-enabled
HDF5/netCDF-C (``netCDF4.__has_parallel4_support__``). On a single rank, pass
``allow_serial=True`` to exercise the same code path without a real MPI
launch; see :func:`climtools.xgeo.to_netcdf`.
"""

import time
from pathlib import Path

import numpy as np
import xarray as xr

from climtools import mpi


def create_precipitation_dataset() -> xr.Dataset:
    """Build a synthetic (time, lat, lon) precipitation dataset."""
    nt, nlat, nlon = 100, 91, 180
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars={
            "precipitation": (
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
    """Write and verify a serial NetCDF file. Rank 0 only."""
    ds = create_precipitation_dataset()
    path = out_dir / "pr_serial.nc"

    mpi.log(f"writing {path} serially")
    started = time.perf_counter()
    ds.xgeo.to_netcdf(path, unlimited_dim="time", show_progress=False)
    mpi.log(f"serial write done in {time.perf_counter() - started:.3f} s")

    with xr.open_dataset(path) as check:
        if not np.array_equal(
            check["precipitation"].values, ds["precipitation"].values
        ):
            raise AssertionError(f"{path}: data mismatch after serial write")
    mpi.log(f"{path}: verified")


def run_parallel(out_dir: Path) -> None:
    """Write and verify a NetCDF file with MPI-collective parallel I/O.

    Rank 0 builds the full dataset; every other rank binds an
    :func:`climtools.xgeo.empty_dataset` so the ``.xgeo`` accessor is still
    available. Its contents are unused: the writer scatters rank 0's buffers.
    """
    if mpi.comm.rank == 0:
        ds = create_precipitation_dataset()

    path = out_dir / "pr_parallel.nc"
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
        reference = create_precipitation_dataset()["precipitation"].values
        with xr.open_dataset(path) as check:
            if not np.array_equal(check["precipitation"].values, reference):
                raise AssertionError(f"{path}: data mismatch after parallel write")
        mpi.log(f"{path}: verified")


def main() -> None:

    out_dir = Path.home() / "scratch" / "io_mpi_test"

    out_dir.mkdir(parents=True, exist_ok=True)

    run_parallel(out_dir)

    run_serial(out_dir)


if __name__ == "__main__":
    main()
