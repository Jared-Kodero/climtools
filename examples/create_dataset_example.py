"""Complete example: mpi.xarray.create_dataset.

Every rank builds only its own slice of each variable directly -- no
distribute(), no scatter, no MPI transfer of the actual data at all. Run
with, e.g.:

    mpirun -n 8 python examples/create_dataset_example.py
"""

from __future__ import annotations

import numpy as np
from climtools import mpi, xgeo

TIME_STEPS = 720
N_LAT, N_LON = 8, 8


# --- one fill function per variable -----------------------------------
#
# fill(start, stop) is called ONCE PER RANK, with that rank's own global
# bounds along "time". start/stop are global indices: rank 3 of 8 gets
# (270, 360), not (0, 90) -- so a formula that depends on absolute time
# must use `start`/`stop` directly, not re-zero them.
def fill_pr(start: int, stop: int) -> np.ndarray:
    """Precipitation: (time, lat, lon), partitioned along "time"."""
    t = np.arange(start, stop, dtype=np.float32)
    lat = np.linspace(-90, 90, N_LAT, dtype=np.float32)
    lon = np.linspace(-180, 180, N_LON, endpoint=False, dtype=np.float32)
    lat_rad = np.deg2rad(lat)[None, :, None]
    lon_rad = np.deg2rad(lon)[None, None, :]
    phase = (t % 24.0)[:, None, None]
    return (
        1.0e-4
        * (1.25 + np.cos(lat_rad) ** 2)
        * (1.0 + 0.15 * np.sin(lon_rad))
        * (1.0 + 0.01 * phase)
    ).astype(np.float32)


def fill_t2m(start: int, stop: int) -> np.ndarray:
    """Surface temperature: (time, lat, lon), partitioned along "time"."""
    t = np.arange(start, stop, dtype=np.float32)
    lat = np.linspace(-90, 90, N_LAT, dtype=np.float32)
    lon = np.linspace(-180, 180, N_LON, endpoint=False, dtype=np.float32)
    lat_rad = np.deg2rad(lat)[None, :, None]
    lon_rad = np.deg2rad(lon)[None, None, :]
    phase = (t % 24.0)[:, None, None]
    return (
        288.0 - 42.0 * np.sin(lat_rad) ** 2 + 2.0 * np.cos(lon_rad) + 0.05 * phase
    ).astype(np.float32)


def fill_slmsk() -> np.ndarray:
    """Sea/land mask: (lat, lon), NOT partitioned -- identical on every
    rank, so this takes no arguments at all.
    """
    lat_idx = np.arange(N_LAT)[:, None]
    lon_idx = np.arange(N_LON)[None, :]
    return ((lat_idx + lon_idx) % 3).astype(np.int8)


def main() -> None:
    # This runs identically, in parallel, on every rank -- there is no
    # "rank 0 builds it, others wait" step anywhere in this function.
    ds = mpi.xarray.create_dataset(
        data_vars={
            "pr": (("time", "lat", "lon"), fill_pr),
            "t2m": (("time", "lat", "lon"), fill_t2m),
            "slmsk": (("lat", "lon"), fill_slmsk),
        },
        sizes={"time": TIME_STEPS, "lat": N_LAT, "lon": N_LON},
        dim="time",
        coords={
            # A full-length coordinate is auto-sliced to this rank's own
            # bounds -- pass it exactly as you would to xr.Dataset itself.
            "time": (
                "time",
                np.arange(TIME_STEPS, dtype=np.float64),
                {"units": "hours since 1970-01-01 00:00:00"},
            ),
            "lat": np.linspace(-90, 90, N_LAT, dtype=np.float32),
            "lon": np.linspace(-180, 180, N_LON, endpoint=False, dtype=np.float32),
        },
        log_partitions=True,  # prints a per-rank partition table on rank 0
    )

    mpi.log(f"rank {mpi.comm.rank}: local ds sizes = {dict(ds.sizes)}")

    # ds is not yet computed -- to_netcdf triggers each rank's fill()
    # calls as it writes that rank's own slab, still with no scatter.
    xgeo.to_netcdf(
        ds,
        "mock_climate.nc",
        unlimited_dim="time",
        partition_dim="time",
        parallel=True,
        allow_serial=(mpi.comm.size == 1),
    )
    mpi.log("done")


if __name__ == "__main__":
    main()
