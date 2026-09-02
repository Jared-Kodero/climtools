"""groupby and resample correctness: previously untested in the permanent
suite. Checked under a single partition dimension (the primary use
case -- 'time' is what both naturally group/resample along) and under a
multi-dimensional partition (no explicit guard exists in groupby.py for
this, so its multi-dim status was genuinely unverified rather than a
known, declared limitation -- worth checking directly instead of
assuming either way).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from climtools import mpi
from mpi_test_common import Fixtures, local_of, record


def run(fx: Fixtures) -> None:
    native, dist, dist2d = fx.native, fx.dist, fx.dist2d

    # -- resample('time', freq).mean(): single-dim, the natural case ------
    try:
        result = dist.resample("time", "3h").mean()
        local = local_of(result)
        m = result.meta if hasattr(result, "meta") else None
        expected_full = native.resample(time="3h").mean()
        if m is not None:
            d = m["dims"][0]
            s, e = m["starts"][d], m["stops"][d]
            expected_full = expected_full.isel({d: slice(s, e)})
        # Compare only variables that genuinely depend on 'time': native's
        # own .resample(...).mean() broadcasts a variable lacking 'time'
        # (e.g. 'slmsk') across the new bins with different, non-constant
        # values per bin (confirmed directly, not assumed) -- an xarray
        # resample quirk, not a meaningful reference for a variable this
        # operation never touched.
        time_vars = [v for v in native.data_vars if "time" in native[v].dims]
        for var in time_vars:
            xr.testing.assert_allclose(local[var], expected_full[var], rtol=1e-5)
        record("resample", "1d(time), mean", True)
    except Exception as e:
        record("resample", "1d(time), mean", False, f"{type(e).__name__}: {str(e)[:200]}")
    mpi.comm.barrier()

    # -- groupby('time', labels).mean(): single-dim ------------------------
    m0 = dist.meta
    start, stop = m0["start"], m0["stop"]
    global_labels = (native.time.values.astype("int64") // 4) % 3  # 3 arbitrary bins
    local_labels = global_labels[start:stop]
    try:
        result = dist.groupby("time", local_labels).mean()
        local = local_of(result)
        native_labeled = native.assign_coords(_grp=("time", global_labels))
        expected_full = native_labeled.groupby("_grp").mean()
        m = result.meta if hasattr(result, "meta") else None
        if m is not None:
            d = m["dims"][0]
            s, e = m["starts"][d], m["stops"][d]
            expected_full = expected_full.isel({d: slice(s, e)})
        time_vars = [v for v in native.data_vars if "time" in native[v].dims]
        group_dim = [dd for dd in local.dims if dd not in expected_full.dims]
        if group_dim:
            local = local.rename({group_dim[0]: "_grp"})
        local = local.sortby("_grp") if "_grp" in getattr(local, "dims", ()) else local
        expected_full = expected_full.sortby("_grp")
        for var in time_vars:
            xr.testing.assert_allclose(
                local[var].transpose(*expected_full[var].dims), expected_full[var], rtol=1e-5
            )
        record("groupby", "1d(time), mean", True)
    except Exception as e:
        record("groupby", "1d(time), mean", False, f"{type(e).__name__}: {str(e)[:200]}")
    mpi.comm.barrier()

    # -- groupby('lat', labels).mean(): multi-dim, status genuinely
    #    unverified prior to this check (no guard exists either way) -------
    m2 = dist2d.meta
    lat_s, lat_e = m2["starts"]["lat"], m2["stops"]["lat"]
    global_lat_labels = np.digitize(native.lat.values, bins=[-90, -30, 30, 90]) - 1
    local_lat_labels = global_lat_labels[lat_s:lat_e]
    try:
        result = dist2d.groupby("lat", local_lat_labels).mean()
        local = local_of(result)
        native_labeled = native.assign_coords(_grp=("lat", global_lat_labels))
        expected_full = native_labeled.groupby("_grp").mean()
        m = result.meta
        sel = {d: slice(m["starts"][d], m["stops"][d]) for d in m["dims"]}
        expected = expected_full.isel(sel)
        group_dim = [dd for dd in local.dims if dd not in expected.dims]
        if group_dim:
            local = local.rename({group_dim[0]: "_grp"})
        local = local.sortby("_grp") if "_grp" in getattr(local, "dims", ()) else local
        expected = expected.sortby("_grp")
        for var in expected.data_vars:
            xr.testing.assert_allclose(
                local[var].transpose(*expected[var].dims), expected[var], rtol=1e-5
            )
        ok = True
        msg = ""
    except NotImplementedError as e:
        ok = None
        msg = f"NotImplementedError (declared): {str(e)[:120]}"
    except Exception as e:
        ok = False
        msg = f"unexpected {type(e).__name__}: {str(e)[:150]}"
    all_ok = mpi.comm.allgather(ok)
    if mpi.comm.rank == 0:
        if all(o is None for o in all_ok):
            record("groupby", "2d(lat,lon), mean", None, msg)
        else:
            record("groupby", "2d(lat,lon), mean", all(o for o in all_ok if o is not None), msg)
