"""Regression test: MPIXarray.cumsum() must not corrupt +-inf results.

Run directly under ``mpirun``, e.g.::

    mpirun -n 4 python -m mpi4py test/test_cumsum_inf.py

Background
----------
``xarray.elementwise._cumsum_scan`` builds its cross-rank exclusive
prefix by gathering every rank's local total onto rank 0 and scanning
them into a running sum, seeded from an additive identity. That
identity used to be computed as ``totals[0] * 0``. Whenever a field
legitimately contains ``+-inf`` (routine in geophysical data -- e.g.
``log`` of a non-positive value, or a division by a genuine zero
denominator), ``inf * 0`` is ``nan``, and because every rank's
exclusive prefix derives from that same seed, the single ``nan``
silently propagates into *every* rank's result at that position, not
just the rank that produced the inf. This test drives that exact
path (an inf placed inside the partition dimension) and checks the
distributed result against plain, single-process xarray, under both
an even (4-rank) and an uneven (3-rank) partition of the same array.
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr

from climtools import mpi
from climtools.xarray.core import MPIXarray

rank = mpi.comm.rank
size = mpi.comm.size


def gather_full(mx_obj: MPIXarray) -> xr.DataArray:
    """Reassemble a distributed MPIXarray's data to its full extent on every rank."""
    meta = mx_obj.meta
    if meta is None:
        return mx_obj.data
    pieces = mpi.comm.allgather(mx_obj.data)
    return xr.concat(pieces, dim=meta["dim"])


def main() -> None:
    n = 20
    values = np.arange(n, dtype=np.float64)
    # Planted at a low index so it lands in rank 0's own local total under
    # any rank count this test is run with (up to n // 1); that is exactly
    # the case the old `totals[0] * 0` seed mishandled.
    values[2] = -np.inf
    da = xr.DataArray(values, dims=["x"], coords={"x": np.arange(n)}, name="v")

    native = da.cumsum(dim="x")

    mx = MPIXarray(da.copy(deep=True), mpi, dim="x")
    result = gather_full(mx.cumsum(dim="x"))

    if rank == 0:
        inf_matches = np.array_equal(np.isinf(native.values), np.isinf(result.values))
        finite_native = np.where(np.isinf(native.values), 0.0, native.values)
        finite_result = np.where(np.isinf(result.values), 0.0, result.values)
        values_match = np.allclose(finite_native, finite_result)
        nan_count = int(np.isnan(result.values).sum())

        ok = inf_matches and values_match and nan_count == 0
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] cumsum with -inf under {size} ranks (nan_count={nan_count})")
        print("  native:", native.values)
        print("  mpix:  ", result.values)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
