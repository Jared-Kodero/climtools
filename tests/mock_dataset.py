"""Mock NetCDF dataset(s) for the climtools MPI test suite."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
from climtools import mpi

import xarray as xr

OUTPUT_DIR = (Path.home() / "jobtmp" / "climtools_mock_dataset_test").resolve()


name = uuid.uuid4().hex if mpi.comm.rank == 0 else None
name = mpi.comm.bcast(name, root=0)
path = OUTPUT_DIR / f"{name}.nc"


# ---------------------------------------------------------------------------
# Shared, whole-suite geophysical mock dataset
# ---------------------------------------------------------------------------


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

    time = np.arange(n_time, dtype=np.float64)
    n_lat = int(180 / resolution_deg) + 1
    lat = np.linspace(-90, 90, n_lat, dtype=np.float32)
    n_lon = int(360 / resolution_deg)
    lon = np.linspace(-180, 180, n_lon, endpoint=False, dtype=np.float32)
    plev = np.arange(1000.0, -1.0, plev_step, dtype=np.float32)

    lat_rad = np.deg2rad(lat)[None, :, None]
    lon_rad = np.deg2rad(lon)[None, None, :]
    time_phase = (time % 24.0)[:, None, None]

    precipitation = (
        1.0e-4
        * (1.25 + np.cos(lat_rad) ** 2)
        * (1.0 + 0.15 * np.sin(lon_rad))
        * (1.0 + 0.01 * time_phase)
    ).astype(np.float32)

    surface_temperature_base = (
        288.0 - 42.0 * np.sin(lat_rad) ** 2 + 2.0 * np.cos(lon_rad)
    ).astype(np.float32)
    surface_temperature = (surface_temperature_base + 0.05 * time_phase).astype(
        np.float32
    )

    pressure_cooling = (7.0 * np.log(1000.0 / plev.astype(np.float64))).astype(
        np.float32
    )[:, None, None]
    air_temperature = surface_temperature_base[0][None, :, :] - pressure_cooling

    lat_index = np.arange(n_lat)[:, None]
    lon_index = np.arange(n_lon)[None, :]
    sea_land_mask = ((lat_index + lon_index) % 3).astype(np.int8)

    ds = xr.Dataset(
        data_vars={
            "pr": (
                ("time", "lat", "lon"),
                precipitation,
                {"units": "kg m-2 s-1", "long_name": "precipitation rate"},
            ),
            "t2m": (
                ("time", "lat", "lon"),
                surface_temperature,
                {"units": "K", "long_name": "2 m air temperature"},
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
            "time": (
                "time",
                time.astype(np.float64),
                {"units": "hours since 1970-01-01 00:00:00"},
            ),
            "plev": ("plev", plev, {"units": "hPa", "positive": "down"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={"title": "climtools MPI test suite mock dataset"},
    )

    ds.to_netcdf(path, format="NETCDF4")


def create_dataset(
    path: Path,
    n_time: int = 48,
    resolution_deg: float = 10.0,
    plev_step: float = -200.0,
) -> Path:

    if mpi.comm.rank == 0:
        if path.parent.exists():
            shutil.rmtree(path.parent)
            path.parent.mkdir(parents=True, exist_ok=True)

        # we need to broadcaste the constanst
        build_dataset(path, n_time, resolution_deg, plev_step)
    mpi.comm.barrier()
