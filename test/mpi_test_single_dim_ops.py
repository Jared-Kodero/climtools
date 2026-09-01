"""Correctness suite for cumsum/median/sortby/reindex/matmul: single-dim vs
multi-dim partition. These use gather/scatter or gather-to-root, not
halo_exchange, and are documented as single-partition-dimension only.

Run with: mpirun --oversubscribe -n <N> python mpi_test_single_dim_ops.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from climtools import mpi, xgeo
from climtools.xarray.core import MPIXarray

RESULTS: list[tuple[str, str, bool, str]] = []


def record(op, case, ok, msg=""):
    RESULTS.append((op, case, ok, msg))


def local_of(value):
    return value._prepare().load() if isinstance(value, MPIXarray) else value


def check_single_dim(op_name, fn, native_fn):
    """Single-dim: dist result's local slice must match native's."""
    try:
        result = fn()
        local = local_of(result)
        m = result.meta if isinstance(result, MPIXarray) else None
        expected_full = native_fn()
        if m is None:
            # replicated result (e.g. median/cumsum's cross-rank reduce can
            # legitimately leave meta=None); compare in full on every rank
            xr.testing.assert_allclose(local, expected_full, rtol=1e-6)
        else:
            d = m["dims"][0]
            s, e = m["starts"][d], m["stops"][d]
            xr.testing.assert_allclose(local, expected_full.isel({d: slice(s, e)}), rtol=1e-6)
        record(op_name, "1d(time)", True)
    except Exception as e:
        record(op_name, "1d(time)", False, f"{type(e).__name__}: {str(e)[:200]}")


def check_multidim_raises(op_name, fn):
    """Multi-dim: must raise NotImplementedError cleanly, not silently misbehave."""
    try:
        fn()
        record(op_name, "2d raises cleanly", False, "no exception raised -- silently wrong?")
    except NotImplementedError as e:
        record(op_name, "2d raises cleanly", True, str(e)[:120])
    except Exception as e:
        record(op_name, "2d raises cleanly", False, f"wrong exception type {type(e).__name__}: {e}")


def report():
    gathered = mpi.comm.gather(RESULTS, root=0)
    if mpi.comm.rank != 0:
        return
    combined = {}
    msgs = {}
    for rank_results in gathered:
        for op, case, ok, msg in rank_results:
            key = (op, case)
            combined.setdefault(key, []).append(ok)
            if msg:
                msgs[key] = msg
    print(f"\n=== single-dim-only op results ({mpi.comm.size} ranks) ===")
    for (op, case), oks in combined.items():
        status = "PASS" if all(oks) else "FAIL"
        print(f"[{status}] {op:<10} {case:<20} {msgs.get((op, case), '')}")


# ---------------------------------------------------------------------------
from mock_dataset import _path, create_dataset  # noqa: E402

create_dataset(n_time=12, resolution_deg=10, plev_step=-250)
mpi.comm.barrier()
native = xr.open_dataset(_path).load()

dist_1d = xgeo.mpi_open_dataset(_path, mpi, partition_dim="time", log_partitions=False)
dist_2d = xgeo.mpi_open_dataset(_path, mpi, partition_dim=("lat", "lon"), log_partitions=False)

# cumsum
check_single_dim("cumsum", lambda: dist_1d.cumsum("time"), lambda: native.cumsum("time"))
check_multidim_raises("cumsum", lambda: dist_2d.cumsum("lat"))
mpi.comm.barrier()

# median
check_single_dim("median", lambda: dist_1d.median("time"), lambda: native.median("time"))
check_multidim_raises("median", lambda: dist_2d.median("lat"))
mpi.comm.barrier()

# sortby -- sort by a coordinate that varies along the partition dim
check_single_dim(
    "sortby",
    lambda: dist_1d.sortby("time", ascending=False),
    lambda: native.sortby("time", ascending=False),
)
check_multidim_raises("sortby", lambda: dist_2d.sortby("lat", ascending=False))
mpi.comm.barrier()

# reindex -- new labels along the partition dim, subset + reorder
new_time = native.time.values[::-1][:8]
check_single_dim(
    "reindex",
    lambda: dist_1d.reindex(time=new_time),
    lambda: native.reindex(time=new_time),
)
new_lat = native.lat.values[::-1]
check_multidim_raises("reindex", lambda: dist_2d.reindex(lat=new_lat))
mpi.comm.barrier()

# matmul -- 2D DataArrays, partitioned along the shared contraction dim
GX, GY = 12, 5


def fill_left(a, b):
    return np.arange(a, b, dtype=np.float64)[:, None] * np.ones((1, GY))


left_1d = xgeo.mpi_create_dataarray(
    mpi, fill_left, dims=("x", "y"), shape={"x": GX, "y": GY}, dim="x",
    log_partitions=False, name="left",
)
right_native = xr.DataArray(np.arange(GY * 3, dtype=np.float64).reshape(GY, 3), dims=("y", "z"))
native_left = xr.DataArray(
    np.arange(GX, dtype=np.float64)[:, None] * np.ones((1, GY)), dims=("x", "y")
)
check_single_dim(
    "matmul",
    lambda: left_1d.matmul(right_native),
    lambda: native_left.dot(right_native, dim="y"),
)

report()
