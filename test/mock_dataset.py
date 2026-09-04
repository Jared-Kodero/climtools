"""Mock NetCDF dataset(s) for the climtools MPI test suite."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from climtools import MPIContext

import xarray as xr

mpi = MPIContext()

OUTPUT_DIR = (Path.home() / "scratch" / "mpi_test").resolve()
PATH = OUTPUT_DIR / "mock_data.nc"
PATH2D = OUTPUT_DIR / "mock_data2d.nc"


# ---------------------------------------------------------------------------
# Shared, whole-suite geophysical mock dataset
# ---------------------------------------------------------------------------


from pathlib import Path


def build_dataset(
    path: Path,
    n_time: int,
    resolution_deg: float,
    plev_step: float,
) -> None:
    """Write a deterministic (time, plev, lat, lon) geophysical mock file.

    Four variables, matching what the split ``test_mpi_*.py`` modules
    exercise: "pr" and "t2m" vary over (time, lat, lon), "t" is one static
    (plev, lat, lon) profile (a full time-varying pressure-level field is
    unnecessary for any test here and would dominate build time), "slmsk"
    is an integer (lat, lon) land mask. Same formula shapes as climtools'
    own performance suite's mock dataset, at sizes meant for fast unit
    tests rather than profiling.
    """
    if isinstance(n_time, bool) or not isinstance(n_time, int) or n_time < 1:
        raise ValueError("n_time must be a positive integer.")

    n_lat = int(180 / resolution_deg) + 1
    n_lon = int(360 / resolution_deg)

    time_dt = pd.date_range(start="1970-01-01 00:00:00", periods=n_time, freq="h")
    plev = np.arange(1000.0, -1.0, -plev_step, dtype=np.float32)
    lat = np.linspace(-90, 90, n_lat, dtype=np.float32)
    lon = np.linspace(-180, 180, n_lon, endpoint=False, dtype=np.float32)

    # Generate mock precipitation from 0 up to 50 mm/hr max intensity
    max_precipitation_mm_hr = 50.0
    precipitation = (
        np.random.rand(n_time, n_lat, n_lon).astype(np.float32)
        * max_precipitation_mm_hr
    )

    air_temperature = (
        200.0 + np.random.rand(len(plev), n_lat, n_lon).astype(np.float32) * 100.0
    )

    sea_land_mask = np.random.randint(0, 2, size=(n_lat, n_lon), dtype=np.int8)

    ds = xr.Dataset(
        data_vars={
            "pr": (
                ("time", "lat", "lon"),
                precipitation,
                {"units": "mm/hr", "long_name": "precipitation rate"},
            ),
            "t": (
                ("plev", "lat", "lon"),
                air_temperature,
                {"units": "K", "long_name": "air temperature"},
            ),
            "slmsk": (
                ("lat", "lon"),
                sea_land_mask.astype(np.int64),
                {"units": "1", "long_name": "sea-land-ice mask"},
            ),
        },
        coords={
            "time": time_dt,
            "plev": ("plev", plev, {"units": "hPa", "positive": "down"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={"title": "climtools MPI test suite mock dataset"},
    )

    ds = ds.chunk("auto")
    ds.to_netcdf(path)


def create_dataset(
    path: Path | None = None,
    path2d: Path | None = None,
    n_time: int = 12,
    resolution_deg: float = 1,
    plev_step: float = -100,
) -> Path:

    path = path or PATH
    path2d = path2d or PATH2D

    if mpi.comm.rank == 0:
        shutil.rmtree(path.parent, ignore_errors=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        build_dataset(path, n_time, resolution_deg, plev_step)

        os.system(f"cp {path} {path2d}")

    mpi.comm.barrier()
    return path


__all__ = ["create_dataset"]
