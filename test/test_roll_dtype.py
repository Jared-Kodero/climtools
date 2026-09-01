"""Regression test: MPIXarray.roll() must preserve integer/bool dtype.

Run directly under ``mpirun``, e.g.::

    mpirun -n 4 python -m mpi4py test/test_roll_dtype.py

Background
----------
``elementwise.roll`` borrows boundary elements from the relevant
neighbor via a *periodic* halo exchange (wrapping at the true global
edge), then reuses xarray's own windowed ``.shift()`` on the padded
local array to perform the actual circular move, trimming the padding
away afterward. ``.shift()`` unconditionally reserves a float NaN fill
value for the boundary it introduces and upcasts any integer or bool
variable to accommodate it -- correct for a genuine ``shift()``, where
that boundary really can be missing, but not for ``roll()``: with
``periodic=True``, the halo exchange already supplied genuine neighbor
data at every position (real data, borrowed via wraparound at the true
edge), so nothing in the final, trimmed result is ever actually
missing. The upcast survived the trim regardless, silently turning an
``int64`` mask (or a boolean flag) into ``float64`` on every
``roll()`` call, even though the values themselves stayed numerically
correct -- exactly the kind of dtype-preservation break this project
is meant to avoid.

This test checks several dtypes against plain, single-process xarray's
own ``.roll()``, which never has this problem (no ``.shift()``
involved), under both an even and an uneven partition.
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
    nlat, nlon = 23, 9  # 23: deliberately uneven across common rank counts
    lat = np.linspace(-90, 90, nlat, dtype=np.float32)
    lon = np.linspace(-180, 180, nlon, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(5)

    ds = xr.Dataset(
        {
            "mask_i64": (("lat", "lon"), np.arange(nlat * nlon, dtype=np.int64).reshape(nlat, nlon) % 5),
            "mask_i32": (("lat", "lon"), (np.arange(nlat * nlon, dtype=np.int32).reshape(nlat, nlon) % 7)),
            "flag_bool": (("lat", "lon"), (rng.standard_normal((nlat, nlon)) > 0)),
            "field_f32": (("lat", "lon"), rng.standard_normal((nlat, nlon)).astype(np.float32)),
        },
        coords={"lat": lat, "lon": lon},
    )
    native = ds.roll(lat=2, roll_coords=False)

    mx = MPIXarray(ds.copy(deep=True), mpi, dim="lat")
    mx_rolled = mx.roll(dim="lat", shift_by=2)
    gathered = gather_full(mx_rolled)

    if rank == 0:
        ok = True
        for name in ds.data_vars:
            expected_dtype = ds[name].dtype
            got_dtype = gathered[name].dtype
            dtype_ok = got_dtype == expected_dtype
            values_ok = np.array_equal(native[name].values, gathered[name].values)
            status = "PASS" if (dtype_ok and values_ok) else "FAIL"
            print(
                f"[{status}] {name}: dtype expected={expected_dtype} got={got_dtype} "
                f"(dtype_ok={dtype_ok}) values_ok={values_ok}"
            )
            ok = ok and dtype_ok and values_ok

        print(f"[{'PASS' if ok else 'FAIL'}] overall, under {size} ranks")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
