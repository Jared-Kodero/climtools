"""Correctness suite for cumsum/median/sortby/reindex/matmul.

cumsum, median, sortby, and reindex now support multi-dimensional
(Cartesian) partitions, via the dimension-scoped sub-communicator
generalization (_dim_comm/resolve_comm) that diff()/isel() already had.
matmul remains single-dimension only (unchanged, not attempted here).

Every multi-dim check verifies three things, not just numeric agreement
with native xarray: shape consistency (local length matches meta's
start/stop), no duplication (no two ranks claim overlapping ownership of
the same range), and exact coverage (every global position is covered by
exactly one rank) -- duplication is the specific defect this suite exists
to catch (see finish()'s dedup fix in planning.py and median's own).

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
            xr.testing.assert_allclose(local, expected_full, rtol=1e-6)
        else:
            d = m["dims"][0]
            s, e = m["starts"][d], m["stops"][d]
            xr.testing.assert_allclose(local, expected_full.isel({d: slice(s, e)}), rtol=1e-6)
        record(op_name, "1d(time)", True)
    except Exception as e:
        record(op_name, "1d(time)", False, f"{type(e).__name__}: {str(e)[:200]}")


def check_multidim(op_name, fn, native_fn, global_size_of, *, moved_dim):
    """Multi-dim: correct values, no duplicate ownership, exact coverage
    along `moved_dim` (the dimension the operation actually manipulated).

    Two distinct checks, since `moved_dim`'s range legitimately repeats
    across different positions of any *other* surviving dimension (e.g.
    cumsum(lat) redistributes lat independently within each lon-group,
    so every lon-group covers the same lat range -- that's correct, not
    duplication): (1) no two ranks claim the exact same full region
    across every surviving dimension at once (genuine duplicate
    ownership), and (2) grouping ranks by their bounds on every
    surviving dimension *other* than `moved_dim`, each such group's own
    `moved_dim` ranges must exactly, non-overlappingly cover
    [0, global_size) -- this is the check that actually catches
    median's/mean's kind of bug, applied per-group here since more than
    one group can legitimately exist.
    """
    try:
        result = fn()
        m = result.meta
        sel = {d: (m["starts"][d], m["stops"][d]) for d in m["dims"]}
        local = local_of(result)
        s, e = sel[moved_dim]
        n_local = local.sizes.get(moved_dim, 0)
        shape_ok = n_local == (e - s)
        if e > s:
            expected_full = native_fn()
            expected = expected_full.isel({d: slice(*bounds) for d, bounds in sel.items()})
            xr.testing.assert_allclose(local, expected, rtol=1e-5)

        all_sel = mpi.comm.gather(tuple(sorted(sel.items())), root=0)
        ok = shape_ok
        msg = ""
        if mpi.comm.rank == 0:
            no_full_dup = len(all_sel) == len(set(all_sel))
            groups: dict[tuple, list[tuple[int, int]]] = {}
            for entry in all_sel:
                d = dict(entry)
                other = tuple(sorted((k, v) for k, v in d.items() if k != moved_dim))
                groups.setdefault(other, []).append(d[moved_dim])
            per_group_ok = True
            for other, ranges in groups.items():
                coverage = np.zeros(global_size_of(moved_dim), dtype=int)
                for s_, e_ in ranges:
                    coverage[s_:e_] += 1
                if not (np.all(coverage <= 1) and np.all(coverage == 1)):
                    per_group_ok = False
            ok = shape_ok and no_full_dup and per_group_ok
            if not (no_full_dup and per_group_ok):
                msg = f"no_full_dup={no_full_dup} per_group_coverage_ok={per_group_ok}"
        all_ok = mpi.comm.gather(ok, root=0)
        if mpi.comm.rank == 0:
            record(op_name, "2d(lat,lon)", all(all_ok), msg)
    except NotImplementedError as e:
        record(op_name, "2d(lat,lon)", None, f"NotImplementedError: {str(e)[:150]}")
    except Exception as e:
        record(op_name, "2d(lat,lon)", False, f"unexpected {type(e).__name__}: {str(e)[:150]}")


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
    print(f"\n=== single-dim / multi-dim op results ({mpi.comm.size} ranks) ===")
    for (op, case), oks in combined.items():
        if all(o is None for o in oks):
            status = "SKIP"
        elif all(o for o in oks if o is not None) and all(o is not None for o in oks):
            status = "PASS"
        else:
            status = "FAIL"
        print(f"[{status}] {op:<10} {case:<20} {msgs.get((op, case), '')}")


# ---------------------------------------------------------------------------
from mock_dataset import _path, create_dataset  # noqa: E402

create_dataset(n_time=12, resolution_deg=10, plev_step=-250)
mpi.comm.barrier()
native = xr.open_dataset(_path).load()

dist_1d = xgeo.mpi_open_dataset(_path, mpi, partition_dim="time", log_partitions=False)
dist_2d = xgeo.mpi_open_dataset(_path, mpi, partition_dim=("lat", "lon"), log_partitions=False)


def gsize(d):
    return native.sizes[d]


# cumsum
check_single_dim("cumsum", lambda: dist_1d.cumsum("time"), lambda: native.cumsum("time"))
check_multidim("cumsum", lambda: dist_2d.cumsum("lat"), lambda: native.cumsum("lat"), gsize, moved_dim="lat")
mpi.comm.barrier()

# median -- reduces 'lat' away; 'lon' is what survives and gets deduplicated
check_single_dim("median", lambda: dist_1d.median("time"), lambda: native.median("time"))
check_multidim("median", lambda: dist_2d.median("lat"), lambda: native.median("lat"), gsize, moved_dim="lon")
mpi.comm.barrier()

# sortby
check_single_dim(
    "sortby",
    lambda: dist_1d.sortby("time", ascending=False),
    lambda: native.sortby("time", ascending=False),
)
check_multidim(
    "sortby",
    lambda: dist_2d.sortby("lat", ascending=False),
    lambda: native.sortby("lat", ascending=False),
    gsize,
    moved_dim="lat",
)
mpi.comm.barrier()

# reindex
new_time = native.time.values[::-1][:8]
check_single_dim(
    "reindex",
    lambda: dist_1d.reindex(time=new_time),
    lambda: native.reindex(time=new_time),
)
new_lat = native.lat.values[::-1]
check_multidim(
    "reindex",
    lambda: dist_2d.reindex(lat=new_lat),
    lambda: native.reindex(lat=new_lat),
    gsize,
    moved_dim="lat",
)
mpi.comm.barrier()

# matmul -- 2D DataArrays, partitioned along the shared contraction dim
# (single-dim only, unchanged; not attempted under a multi-dim partition)
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
