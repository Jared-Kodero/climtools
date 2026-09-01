"""Regression test: MPIXarray.resample() must not corrupt distribution
metadata when the resampled dimension is not the active partition dimension.

Run directly under ``mpirun``, e.g.::

    mpirun -n 4 python -m mpi4py test/test_resample_meta.py

Background
----------
``resample_reduce`` is a thin wrapper over ``groupby_reduce``: it groups
the target dimension (e.g. "time") into bins and reduces, then renames
``groupby_reduce``'s internal group dimension ("_mpi_group") back to the
original dimension name. The common case -- and the one this test
drives -- is resampling a dimension that is *not* the active MPI
partition dimension at all (e.g. distributed along "lat", resampling
"time"): ``groupby_reduce`` then takes its local, non-communicating
path, and correctly returns metadata describing the untouched "lat"
partition unchanged.

The post-processing rename step used to reattach that metadata under
`dim` ("time") unconditionally -- `set_mpi_meta(renamed, dim=dim,
global_size=meta["global_size"], ...)` -- mislabeling "lat"'s own
global size/start/stop as if they belonged to "time". Every rank ended
up with self-inconsistent metadata (a "time"-named partition whose
bounds were actually "lat"'s), which downstream code discarded as
invalid, silently losing the distribution entirely and leaving every
rank with a *different*, unreconciled local slice reported as if it
were the full, replicated answer -- exactly the state a caller
gathering "the" result (assuming ``.meta is None`` means "identical on
every rank") would silently get wrong data from.

This test checks that resample()'s metadata correctly still names the
untouched partition dimension (not the resampled one) with its true,
unchanged bounds, and that the gathered values match plain,
single-process xarray.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
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
    nlat, nlon, ntime = 23, 17, 8
    lat = np.linspace(-90, 90, nlat, dtype=np.float32)
    lon = np.linspace(-180, 180, nlon, endpoint=False, dtype=np.float32)
    time = pd.date_range("2020-01-01", periods=ntime, freq="6h")
    rng = np.random.default_rng(7)
    t2m = (280 + 10 * rng.standard_normal((ntime, nlat, nlon))).astype(np.float32)

    ds = xr.Dataset(
        {"t2m": (("time", "lat", "lon"), t2m)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    native = ds.resample(time="1D").mean()

    # Distributed along "lat" -- not the dimension being resampled.
    mx = MPIXarray(ds.copy(deep=True), mpi, dim="lat")
    mx_rs = mx.resample("time", "1D").mean()

    meta = mx_rs.meta
    meta_ok = (
        meta is not None
        and meta["dims"] == ("lat",)
        and meta["global_size"] == nlat
        and mx_rs.data.sizes["lat"] == meta["stop"] - meta["start"]
    )
    print(
        f"[rank {rank}] resample meta={meta} "
        f"local lat size={mx_rs.data.sizes.get('lat')} -> meta_ok={meta_ok}"
    )

    gathered = gather_full(mx_rs)
    if rank == 0:
        values_ok = True
        try:
            xr.testing.assert_allclose(native, gathered, rtol=1e-5, atol=1e-6)
        except Exception as exc:
            values_ok = False
            print(f"  value mismatch: {exc}")

        ok = meta_ok and values_ok
        print(f"[{'PASS' if ok else 'FAIL'}] resample metadata/values, under {size} ranks")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
