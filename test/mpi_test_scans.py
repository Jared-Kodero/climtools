"""Scan, redistribution, and dispatch correctness: np.log/isel (rank-local
NumPy dispatch and indexing), cumsum, sortby, reindex, interp, matmul --
single-dim and, where supported, multi-dim, with explicit no-duplication
+ exact-coverage verification for the redistributing operations.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from climtools import mpi, xgeo
from climtools.xarray.core import MPIXarray
from mpi_test_common import Fixtures, local_of, record


def run(fx: Fixtures) -> None:
    native, dist, dist2d = fx.native, fx.dist, fx.dist2d
    start, stop = dist.meta["start"], dist.meta["stop"]

    # -- rank-local NumPy dispatch and indexing ------------------------------
    try:
        logged = np.log(dist.data["pr"])
        expected = np.log(native["pr"]).isel(time=slice(start, stop))
        xr.testing.assert_allclose(local_of(logged), expected, rtol=1e-5)
        record("np.log", "1d(time), NumPy dispatch", True)
    except Exception as e:
        record("np.log", "1d(time), NumPy dispatch", False, str(e)[:200])

    try:
        sub = local_of(dist.isel(time=slice(0, 3)))
        record("isel", "1d(time), indexing", True, str(dict(sub.sizes)))
    except Exception as e:
        record("isel", "1d(time), indexing", False, str(e)[:200])
    mpi.comm.barrier()

    # -- scans and redistribution --------------------------------------------
    def check_single_dim(op_name, fn, native_fn, case="1d(time)"):
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
            record(op_name, case, True)
        except Exception as e:
            record(op_name, case, False, f"{type(e).__name__}: {str(e)[:200]}")

    def check_multidim(op_name, fn, native_fn, *, moved_dim, case="2d(lat,lon)"):
        """No two ranks claim the exact same full region across every
        surviving dimension at once, AND -- grouping ranks by their bounds
        on every surviving dimension other than moved_dim -- each such
        group's own moved_dim ranges exactly, non-overlappingly cover
        [0, global_size). moved_dim's range legitimately repeats across
        different positions of any OTHER surviving dimension (e.g.
        cumsum(lat) redistributes lat independently within each
        lon-group), so a naive global check would wrongly flag that as
        duplication.
        """
        try:
            result = fn()
            m = result.meta
            sel = {d: (m["starts"][d], m["stops"][d]) for d in m["dims"]}
            local = local_of(result)
            s, e = sel[moved_dim]
            shape_ok = local.sizes.get(moved_dim, 0) == (e - s)
            if e > s:
                expected_full = native_fn()
                expected = expected_full.isel({d: slice(*b) for d, b in sel.items()})
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
                    coverage = np.zeros(fx.gsize(moved_dim), dtype=int)
                    for s_, e_ in ranges:
                        coverage[s_:e_] += 1
                    if not np.all(coverage == 1):
                        per_group_ok = False
                ok = shape_ok and no_full_dup and per_group_ok
                if not (no_full_dup and per_group_ok):
                    msg = f"no_full_dup={no_full_dup} per_group_coverage_ok={per_group_ok}"
            all_ok = mpi.comm.gather(ok, root=0)
            if mpi.comm.rank == 0:
                record(op_name, case, all(all_ok), msg)
        except NotImplementedError as e:
            record(op_name, case, None, f"NotImplementedError: {str(e)[:150]}")
        except Exception as e:
            record(op_name, case, False, f"unexpected {type(e).__name__}: {str(e)[:150]}")

    check_single_dim("cumsum", lambda: dist.cumsum("time"), lambda: native.cumsum("time"))
    check_multidim("cumsum", lambda: dist2d.cumsum("lat"), lambda: native.cumsum("lat"), moved_dim="lat")
    mpi.comm.barrier()

    check_single_dim(
        "sortby",
        lambda: dist.sortby("time", ascending=False),
        lambda: native.sortby("time", ascending=False),
    )
    check_multidim(
        "sortby",
        lambda: dist2d.sortby("lat", ascending=False),
        lambda: native.sortby("lat", ascending=False),
        moved_dim="lat",
    )
    mpi.comm.barrier()

    new_time = native.time.values[::-1][:8]
    check_single_dim(
        "reindex", lambda: dist.reindex(time=new_time), lambda: native.reindex(time=new_time),
    )
    new_lat = native.lat.values[::-1]
    check_multidim(
        "reindex", lambda: dist2d.reindex(lat=new_lat), lambda: native.reindex(lat=new_lat),
        moved_dim="lat",
    )
    mpi.comm.barrier()

    # interp -- Allgather-based; not halo-bounded, checked under the
    # partition dimension it interpolates along.
    from climtools.xarray.cartesian import dim_comm as _dim_comm_check
    from climtools.xarray.chunks import get_balanced_bounds as _gbb_check

    new_lat_fine = np.linspace(native.lat.values.min(), native.lat.values.max(), 37)
    sub = _dim_comm_check(mpi, dist2d.meta, "lat")
    s, e = _gbb_check(len(new_lat_fine), sub.rank, sub.size)
    check_multidim(
        "interp",
        lambda: dist2d.interp("lat", new_lat_fine[s:e]),
        lambda: native.interp(lat=new_lat_fine),
        moved_dim="lat",
    )
    mpi.comm.barrier()

    # matmul -- 2D DataArrays, partitioned along the shared contraction dim
    # (single-dim only by nature; not attempted under a multi-dim partition).
    GXM, GYM = 12, 5

    def fill_left(a, b):
        return np.arange(a, b, dtype=np.float64)[:, None] * np.ones((1, GYM))

    left_1d = xgeo.mpi_create_dataarray(
        mpi, fill_left, dims=("x", "y"), shape={"x": GXM, "y": GYM}, dim="x",
        log_partitions=False, name="left",
    )
    right_native = xr.DataArray(
        np.arange(GYM * 3, dtype=np.float64).reshape(GYM, 3), dims=("y", "z")
    )
    native_left = xr.DataArray(
        np.arange(GXM, dtype=np.float64)[:, None] * np.ones((1, GYM)), dims=("x", "y")
    )
    check_single_dim(
        "matmul", lambda: left_1d.matmul(right_native),
        lambda: native_left.dot(right_native, dim="y"), case="1d(x)",
    )
