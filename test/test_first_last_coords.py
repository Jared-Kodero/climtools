"""Regression test: MPIXarray.first()/last() must combine coordinates
that vary along the reduced dimension, not just the data.

Run directly under ``mpirun``, e.g.::

    mpirun -n 4 python -m mpi4py test/test_first_last_coords.py

Background
----------
``reductions._first_last_combine`` picks, per output position, the
value at the first/last non-missing index along the reduced dimension
and cross-rank-elects the true global answer via an owner-election
Allreduce (``MPI.MIN``/``MAX`` to pick the owning rank, then
``MPI.SUM``/``LOR`` to recover its value). That correctly combines the
*data* -- but the reduced dimension's own coordinate (e.g. "lat", or a
real "time" axis) rides along through the same vectorized
``.isel(dim=index, drop=True)` that picks the data, becoming a
non-scalar, per-output-position coordinate that reflects *this rank's
own local pick*. The Allreduce that combines the data does nothing to
that coordinate -- ``comm_reduce`` copies it verbatim from whichever
rank happened to build the operand passed in. Left unfixed, every
rank silently reports its own local candidate's coordinate value
instead of the true, globally-elected one, for every position where
the correct answer came from a different rank.

This test partitions a field across ranks such that the true global
first/last valid value is known to live on a specific rank (not rank
0, so a bug that merely happens to match rank 0's own view would not
be masked), then checks the coordinate MPIXarray reports for it
against plain, single-process xarray.
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr

from climtools import mpi
from climtools.xarray.core import MPIXarray

rank = mpi.comm.rank
size = mpi.comm.size


def gather_full(mx_obj: MPIXarray) -> xr.Dataset:
    meta = mx_obj.meta
    if meta is None:
        return mx_obj.data
    pieces = mpi.comm.allgather(mx_obj.data)
    return xr.concat(pieces, dim=meta["dim"], data_vars="minimal")


def main() -> None:
    nlat, nlon, ntime = 23, 17, 5
    lat = np.linspace(-90, 90, nlat, dtype=np.float32)
    lon = np.linspace(-180, 180, nlon, endpoint=False, dtype=np.float32)
    time = np.arange(ntime, dtype=np.float64)

    rng = np.random.default_rng(42)
    t2m = (280 + 10 * rng.standard_normal((ntime, nlat, nlon))).astype(np.float32)
    # Not at lat index 0: a bug that merely echoes rank 0's own local
    # pick would still get this position wrong.
    t2m[0, 3, 5] = np.nan
    t2m[2, nlat - 1, 0] = np.nan

    ds = xr.Dataset(
        {"t2m": (("time", "lat", "lon"), t2m)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    # Reference "first valid" computed directly, independent of xarray's
    # own (dim-less) reduction API.
    valid = ~np.isnan(t2m)
    first_idx = np.where(valid.any(axis=1), valid.argmax(axis=1), 0)
    last_idx = np.where(
        valid.any(axis=1),
        valid.shape[1] - 1 - valid[:, ::-1, :].argmax(axis=1),
        valid.shape[1] - 1,
    )
    time_idx, lon_idx = np.meshgrid(np.arange(ntime), np.arange(nlon), indexing="ij")
    expected_first_lat = lat[first_idx]
    expected_first_t2m = t2m[time_idx, first_idx, lon_idx]
    expected_last_lat = lat[last_idx]
    expected_last_t2m = t2m[time_idx, last_idx, lon_idx]

    mx = MPIXarray(ds.copy(deep=True), mpi, dim="lat")
    mx_sub = mx.apply(lambda d: d[["t2m"]], mx)

    mx_first = gather_full(mx_sub.first(dim="lat"))
    mx_last = gather_full(mx_sub.last(dim="lat"))

    if rank == 0:
        ok = True

        for label, mx_result, expected_lat, expected_t2m in (
            ("first", mx_first, expected_first_lat, expected_first_t2m),
            ("last", mx_last, expected_last_lat, expected_last_t2m),
        ):
            lat_ok = np.allclose(mx_result["lat"].values, expected_lat)
            t2m_ok = np.allclose(mx_result["t2m"].values, expected_t2m, equal_nan=True)
            status = "PASS" if (lat_ok and t2m_ok) else "FAIL"
            print(f"[{status}] {label}(dim=lat) lat_coord_ok={lat_ok} t2m_ok={t2m_ok}")
            ok = ok and lat_ok and t2m_ok

        print(f"[{'PASS' if ok else 'FAIL'}] overall, under {size} ranks")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
