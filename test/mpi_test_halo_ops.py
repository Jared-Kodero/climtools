"""Halo-dependent operation correctness: rolling_reduce, coarsen_reduce,
diff, shift, differentiate, ffill, bfill -- under a single partition
dimension and a two-dimensional (Cartesian) partition.
"""

from __future__ import annotations

from climtools import mpi
from climtools.xarray.core import MPIXarray
from mpi_test_common import Fixtures, local_of, record


def run(fx: Fixtures) -> None:
    native, dist, dist2d = fx.native, fx.dist, fx.dist2d

    def run_op(op_name, fn, native_fn, case):
        try:
            result = fn()
        except NotImplementedError as e:
            record(op_name, case, None, f"NotImplementedError (declared): {str(e)[:120]}")
            return
        except Exception as e:
            record(op_name, case, False, f"unexpected {type(e).__name__}: {str(e)[:150]}")
            return
        try:
            import xarray as xr

            local = local_of(result)
            m = result.meta if isinstance(result, MPIXarray) else None
            if m is None:
                record(op_name, case, False, "result lost its MPI metadata")
                return
            expected_full = native_fn()
            sel = {d: slice(m["starts"][d], m["stops"][d]) for d in m["dims"]}
            expected = expected_full.isel(sel)
            xr.testing.assert_allclose(local, expected, rtol=1e-6)
            record(op_name, case, True)
        except Exception as e:
            record(op_name, case, False, str(e)[:200])

    for case_label, d, dm in [
        ("1d(time)", dist, "time"),
        ("2d(lat,lon)/lat", dist2d, "lat"),
        ("2d(lat,lon)/lon", dist2d, "lon"),
    ]:
        run_op(
            "rolling_reduce", lambda d=d, dm=dm: d.rolling_reduce(dm, window=3, reduce="mean"),
            lambda dm=dm: native.rolling({dm: 3}, center=True).mean(), case_label,
        )
        run_op(
            "coarsen_reduce",
            lambda d=d, dm=dm: d.coarsen_reduce(dm, window=2, reduce="mean", boundary="trim"),
            lambda dm=dm: native.coarsen({dm: 2}, boundary="trim").mean(), case_label,
        )
        run_op(
            "diff", lambda d=d, dm=dm: d.diff(dm, n=1),
            lambda dm=dm: native.diff(dm, n=1), case_label,
        )
        run_op(
            "shift", lambda d=d, dm=dm: d.shift(dm, periods=1),
            lambda dm=dm: native.shift({dm: 1}), case_label,
        )
        run_op(
            "differentiate",
            lambda d=d, dm=dm: (
                d.differentiate(dm) if dm != "time" else d.differentiate(dm, datetime_unit="s")
            ),
            lambda dm=dm: (
                native.differentiate(dm) if dm != "time"
                else native.differentiate(dm, datetime_unit="s")
            ),
            case_label,
        )
        run_op(
            "ffill", lambda d=d, dm=dm: d.ffill(dm, limit=2),
            lambda dm=dm: native.ffill(dm, limit=2), case_label,
        )
        run_op(
            "bfill", lambda d=d, dm=dm: d.bfill(dm, limit=2),
            lambda dm=dm: native.bfill(dm, limit=2), case_label,
        )
        mpi.comm.barrier()
