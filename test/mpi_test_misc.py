"""Correctness checks for the remaining public MPIXarray methods not
covered by any other mpi_test_*.py module: prod, any, all, first, last,
align, evaluate, roll, repartition, apply.
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

    def check_reduce_1d(op_name, apply_fn, native_fn):
        try:
            result = apply_fn()
            local = local_of(result)
            m = result.meta if isinstance(result, MPIXarray) else None
            expected_full = native_fn()
            if m is None:
                xr.testing.assert_allclose(local, expected_full, rtol=1e-5)
            else:
                d = m["dims"][0]
                s, e = m["starts"][d], m["stops"][d]
                xr.testing.assert_allclose(local, expected_full.isel({d: slice(s, e)}), rtol=1e-5)
            record(op_name, "1d(time)", True)
        except Exception as e:
            record(op_name, "1d(time)", False, f"{type(e).__name__}: {str(e)[:200]}")

    def check_reduce_2d(op_name, reduce_dim, apply_fn, native_fn):
        try:
            result = apply_fn()
            m = result.meta
            surviving = m["dims"][0]
            s, e = m["starts"][surviving], m["stops"][surviving]
            local = local_of(result)
            shape_ok = local.sizes.get(surviving, 0) == (e - s)
            if e > s:
                expected = native_fn().isel({surviving: slice(s, e)})
                xr.testing.assert_allclose(local, expected, rtol=1e-5)
            bounds = mpi.comm.gather((s, e), root=0)
            ok = shape_ok
            if mpi.comm.rank == 0:
                coverage = np.zeros(fx.gsize(surviving), dtype=int)
                for s_, e_ in bounds:
                    coverage[s_:e_] += 1
                ok = ok and bool(np.all(coverage == 1))
            all_ok = mpi.comm.gather(ok, root=0)
            if mpi.comm.rank == 0:
                record(op_name, f"2d(lat,lon)/{reduce_dim}", all(all_ok))
        except Exception as e:
            record(op_name, f"2d(lat,lon)/{reduce_dim}", False, str(e)[:200])

    # -- prod, any, all: standard reductions, same shape as sum/mean --------
    check_reduce_1d("prod", lambda: dist.prod(dim="lat"), lambda: native.prod(dim="lat"))
    check_reduce_2d("prod", "lat", lambda: dist2d.prod(dim="lat"), lambda: native.prod(dim="lat"))
    mpi.comm.barrier()

    def check_bool_reduce_1d(op_name, method):
        try:
            cond = dist.apply(lambda d: d["pr"] > 0.0002, dist._prepare())
            result = getattr(cond, method)(dim="lat")
            local = local_of(result)
            m = result.meta if isinstance(result, MPIXarray) else None
            native_cond = native["pr"] > 0.0002
            expected_full = getattr(native_cond, method)(dim="lat")
            if m is None:
                xr.testing.assert_allclose(local, expected_full, rtol=1e-5)
            else:
                d = m["dims"][0]
                s, e = m["starts"][d], m["stops"][d]
                xr.testing.assert_allclose(local, expected_full.isel({d: slice(s, e)}), rtol=1e-5)
            record(op_name, "1d(time)", True)
        except Exception as e:
            record(op_name, "1d(time)", False, f"{type(e).__name__}: {str(e)[:200]}")

    check_bool_reduce_1d("any", "any")
    check_bool_reduce_1d("all", "all")
    mpi.comm.barrier()

    # -- first, last: pick a position along one dimension. Compare data
    #    values only (dropping any 'lat' coordinate from both sides):
    #    first()/last() legitimately retain 'lat' as a coordinate
    #    broadcast across the other dims (showing which position was
    #    picked per element), an xarray-inherited quirk of
    #    isel(dim=<vectorized index>, drop=True) not fully dropping the
    #    coordinate the way a scalar-index isel(dim=i, drop=True) does
    #    -- confirmed directly against plain xarray, not a climtools
    #    behavior. -------------------------------------------------------
    def check_first_last(op_name, apply_fn, native_fn, expected_lat_value):
        try:
            result = apply_fn()
            local = local_of(result)
            lat_coord_ok = True
            if "lat" in getattr(local, "coords", {}):
                lat_coord_ok = bool(np.all(local.coords["lat"].values == expected_lat_value))
                local = local.drop_vars("lat")
            m = result.meta if isinstance(result, MPIXarray) else None
            expected_full = native_fn()
            if "lat" in getattr(expected_full, "coords", {}):
                expected_full = expected_full.drop_vars("lat")
            if m is None:
                xr.testing.assert_allclose(local, expected_full, rtol=1e-5)
            else:
                d = m["dims"][0]
                s, e = m["starts"][d], m["stops"][d]
                xr.testing.assert_allclose(local, expected_full.isel({d: slice(s, e)}), rtol=1e-5)
            record(op_name, "1d(time)", lat_coord_ok, "" if lat_coord_ok else "retained lat coordinate has wrong value")
        except Exception as e:
            record(op_name, "1d(time)", False, f"{type(e).__name__}: {str(e)[:200]}")

    check_first_last(
        "first", lambda: dist.first("lat"),
        lambda: native.isel(lat=0, drop=True),
        float(native.lat.values[0]),
    )
    check_first_last(
        "last", lambda: dist.last("lat"),
        lambda: native.isel(lat=-1, drop=True),
        float(native.lat.values[-1]),
    )
    mpi.comm.barrier()

    # -- roll: circular shift, same shape/distribution as input -------------
    try:
        result = local_of(dist.roll("lat", shift_by=2))
        expected = native.isel(time=slice(start, stop)).roll(lat=2, roll_coords=False)
        xr.testing.assert_allclose(result, expected, rtol=1e-6)
        record("roll", "1d(time)/lat", True)
    except Exception as e:
        record("roll", "1d(time)/lat", False, f"{type(e).__name__}: {str(e)[:200]}")
    mpi.comm.barrier()

    # -- evaluate: string-expression evaluation, rank-local ------------------
    try:
        result = local_of(dist.evaluate("a + b * 2", a=dist.data["pr"], b=dist.data["t2m"]))
        expected = (
            native["pr"].isel(time=slice(start, stop))
            + native["t2m"].isel(time=slice(start, stop)) * 2
        )
        xr.testing.assert_allclose(result, expected, rtol=1e-5)
        record("evaluate", "1d(time), rank-local", True)
    except Exception as e:
        record("evaluate", "1d(time), rank-local", False, f"{type(e).__name__}: {str(e)[:200]}")
    mpi.comm.barrier()

    # -- apply: call a rank-local callable, propagating MPI metadata --------
    try:
        result = dist.apply(lambda d: d * 2.0, dist._prepare())
        local = local_of(result)
        expected = native.isel(time=slice(start, stop)) * 2.0
        xr.testing.assert_allclose(local, expected, rtol=1e-6)
        meta_ok = isinstance(result, MPIXarray) and result.meta is not None
        record("apply", "1d(time), rank-local", meta_ok, "" if meta_ok else "meta did not propagate")
    except Exception as e:
        record("apply", "1d(time), rank-local", False, f"{type(e).__name__}: {str(e)[:200]}")
    mpi.comm.barrier()

    # -- repartition: distribute an object every rank already holds fully ---
    def fill_replicated(n):
        idx = np.arange(n, dtype=np.float64)
        return xr.DataArray(np.sin(idx), dims=("z",), name="v")

    GZ = 21  # uneven for several rank counts, exercising the same path as
    #          get_balanced_bounds's own uneven-split checks
    replicated = fill_replicated(GZ)
    try:
        # repartition() is the MPIXarray method for an object every rank
        # ALREADY holds fully -- unlike mpi_partition_data's root-scatter
        # contract, no data movement is actually required here since
        # every rank starts with the same full array; it just
        # relabels/slices locally.
        result2 = MPIXarray(replicated, mpi, auto_partition=False).repartition(dim="z")
        local2 = local_of(result2)
        m2 = result2.meta
        s2, e2 = m2["start"], m2["stop"]
        expected2 = replicated.isel(z=slice(s2, e2))
        xr.testing.assert_allclose(local2, expected2, rtol=1e-10)
        record("repartition", "1d(z), uneven", True)
    except Exception as e:
        record("repartition", "1d(z), uneven", False, f"{type(e).__name__}: {str(e)[:200]}")
    mpi.comm.barrier()

    # -- align: partition two operands identically ---------------------------
    try:
        other_full = fill_replicated(GZ) * 3.0
        left, right = MPIXarray(replicated, mpi, auto_partition=False).align(other_full, dim="z")
        m_left, m_right = left.meta, right.meta
        same_partition = (
            m_left["dims"] == m_right["dims"]
            and m_left["starts"] == m_right["starts"]
            and m_left["stops"] == m_right["stops"]
        )
        s3, e3 = m_left["start"], m_left["stop"]
        expected_left = replicated.isel(z=slice(s3, e3))
        expected_right = (fill_replicated(GZ) * 3.0).isel(z=slice(s3, e3))
        xr.testing.assert_allclose(local_of(left), expected_left, rtol=1e-10)
        xr.testing.assert_allclose(local_of(right), expected_right, rtol=1e-10)
        record("align", "1d(z), matching partitions", same_partition)
    except Exception as e:
        record("align", "1d(z), matching partitions", False, f"{type(e).__name__}: {str(e)[:200]}")
