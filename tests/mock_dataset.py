"""Mock NetCDF dataset(s) for the climtools MPI test suite."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Datetime-indexed mock dataset -- for resample()/groupby()-over-time tests.
# Deliberately a real ``datetime64`` coordinate (unlike ``build_dataset``'s
# numeric "hours since" time), an odd, non-bin-aligned start time, and a
# length not evenly divisible by common rank counts, since all three are
# exactly what previously exposed the resample() bin-alignment bug.
# ---------------------------------------------------------------------------


def build_timeseries_dataset(
    path: Path,
    n: int,
    freq: str,
    start: str,
) -> None:
    """Write a deterministic, datetime-indexed (time, station) mock file.

    One float32 variable ("v") varying smoothly over time and station, plus
    one static (per-station) float32 variable ("station_elev") that does not
    carry the "time" dimension at all -- exercises the same "variable
    doesn't carry the partitioned dim" case ``build_dataset``'s "t" does,
    at a size small enough that tests build/open it in well under a second.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")

    time = pd.date_range(start, periods=n, freq=freq)
    n_station = 4
    station = np.arange(n_station)

    t = np.arange(n, dtype=np.float64)[:, None]
    s = station.astype(np.float64)[None, :]
    values = (np.sin(t / 5.0 + s) + 0.01 * t).astype(np.float32)
    station_elev = (100.0 * (station + 1)).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "v": (
                ("time", "station"),
                values,
                {"units": "1", "long_name": "mock time series"},
            ),
            "station_elev": (
                ("station",),
                station_elev,
                {"units": "m", "long_name": "mock station elevation"},
            ),
        },
        coords={"time": time, "station": station},
        attrs={"title": "climtools MPI test suite mock timeseries"},
    )
    ds.to_netcdf(path, format="NETCDF4")


def create_timeseries_dataset(
    path: Path,
    n: int = 40,
    freq: str = "D",
    start: str = "2020-01-01",
) -> Path:
    if mpi.comm.rank == 0:
        # Deliberately no `shutil.rmtree(path.parent)` here (unlike
        # `create_dataset` above): a single test process may build more
        # than one mock file (e.g. a geophysical one and a timeseries
        # one) sharing `OUTPUT_DIR`, and each file's name is already a
        # fresh UUID, so wiping the whole directory on every call would
        # delete a sibling mock file this same test still needs.
        path.parent.mkdir(parents=True, exist_ok=True)
        build_timeseries_dataset(path, n, freq, start)
    mpi.comm.barrier()
    return path


# ---------------------------------------------------------------------------
# Two-independent-dimension mock dataset -- for 2D Cartesian (multi-dim
# partition) tests. Both dimensions are deliberately sized so they don't
# divide evenly by common rank counts.
# ---------------------------------------------------------------------------


def build_grid_dataset(path: Path, n_a: int, n_b: int) -> None:
    """Write a deterministic (a, b) mock file for multi-dim partition tests."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (n_a, n_b)):
        raise ValueError("n_a and n_b must be positive integers.")

    a = np.arange(n_a)
    b = np.arange(n_b)
    values = (
        np.sin(a.astype(np.float64)[:, None] / 3.0)
        + np.cos(b.astype(np.float64)[None, :] / 5.0)
    ).astype(np.float32)

    ds = xr.Dataset(
        data_vars={"v": (("a", "b"), values, {"units": "1"})},
        coords={"a": a, "b": b},
        attrs={"title": "climtools MPI test suite mock grid"},
    )
    ds.to_netcdf(path, format="NETCDF4")


def create_grid_dataset(path: Path, n_a: int = 13, n_b: int = 9) -> Path:
    if mpi.comm.rank == 0:
        # See create_timeseries_dataset's comment: no directory wipe here,
        # so this can coexist with other mock files under OUTPUT_DIR.
        path.parent.mkdir(parents=True, exist_ok=True)
        build_grid_dataset(path, n_a, n_b)
    mpi.comm.barrier()
    return path


# ---------------------------------------------------------------------------
# Multi-dtype 1D mock dataset -- for reindex()/sortby() dtype-preservation
# regression tests. One dimension ("x"), three variables spanning the
# dtypes that matter for the float32->float64 promotion regression
# (float32, float64, int32), plus a deterministically shuffled "key"
# coordinate for exercising sortby() as a genuine permutation.
# ---------------------------------------------------------------------------


def build_multitype_dataset(path: Path, n: int, seed: int = 0) -> None:
    """Write a deterministic 1D mock file spanning float32/float64/int32."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")

    rng = np.random.default_rng(seed)
    x = np.arange(n)
    key = rng.permutation(n).astype(np.float32)
    # Smooth and nonlinear (not just arange) so the same variable also
    # serves shift()/differentiate() tests meaningfully -- a purely
    # linear function can't distinguish edge_order=1 from edge_order=2.
    var32 = (np.sin(x.astype(np.float64) / 4.0) + 0.02 * x).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "var32": ("x", var32, {"units": "1"}),
            "var64": ("x", np.arange(n, dtype=np.float64), {"units": "1"}),
            "varint": ("x", np.arange(n, dtype=np.int32), {"units": "1"}),
        },
        coords={"x": x, "key": ("x", key)},
        attrs={"title": "climtools MPI test suite mock multitype dataset"},
    )
    ds.to_netcdf(path, format="NETCDF4")


def create_multitype_dataset(path: Path, n: int = 17, seed: int = 0) -> Path:
    if mpi.comm.rank == 0:
        # See create_timeseries_dataset's comment: no directory wipe here.
        path.parent.mkdir(parents=True, exist_ok=True)
        build_multitype_dataset(path, n, seed)
    mpi.comm.barrier()
    return path


# ---------------------------------------------------------------------------
# Larger (time, lat) mock dataset -- for the benchmark suite, where sizes
# need to be big enough that MPI startup and Python call overhead don't
# dominate the measurement (see tests/bench_mpi_suite.py). Deliberately
# plain 2D (unlike the geophysical dataset's (time, lat, lon)) so bench
# results reflect one dominant operation's cost, not incidental grid size.
# ---------------------------------------------------------------------------


def build_bench_dataset(path: Path, n_time: int, n_lat: int, seed: int = 0) -> None:
    """Write a deterministic (time, lat) float32 mock file for benchmarking."""
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (n_time, n_lat)
    ):
        raise ValueError("n_time and n_lat must be positive integers.")

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_time, n_lat)).astype(np.float32)

    ds = xr.Dataset(
        data_vars={"v": (("time", "lat"), values, {"units": "1"})},
        coords={"time": np.arange(n_time), "lat": np.arange(n_lat)},
        attrs={"title": "climtools MPI test suite mock benchmark dataset"},
    )
    ds.to_netcdf(path, format="NETCDF4")


def create_bench_dataset(
    path: Path, n_time: int = 400_000, n_lat: int = 20, seed: int = 0
) -> Path:
    if mpi.comm.rank == 0:
        # See create_timeseries_dataset's comment: no directory wipe here.
        path.parent.mkdir(parents=True, exist_ok=True)
        build_bench_dataset(path, n_time, n_lat, seed)
    mpi.comm.barrier()
    return path

