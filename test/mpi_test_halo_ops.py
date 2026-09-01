"""Halo-dependent operation correctness suite: single-dim vs multi-dim partition.

Run with: mpirun --oversubscribe -n <N> python mpi_test_halo_ops.py

Tests every operation in climtools that internally uses halo_exchange()
(rolling_reduce, coarsen_reduce, diff, shift, differentiate, ffill, bfill)
against plain xarray, once under a single partition dimension and once
under a two-dimensional (Cartesian) partition -- since several of these
are documented as multi-dim-unsupported (raising NotImplementedError) and
the rest have never been exercised under multi-dim at all until now.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from climtools import mpi, xgeo
from climtools.xarray.core import MPIXarray

RESULTS: list[tuple[str, str, bool, str]] = []


def record(op: str, case: str, ok: bool, msg: str = "") -> None:
    RESULTS.append((op, case, ok, msg))


def local_of(value):
    return value._prepare().load() if isinstance(value, MPIXarray) else value


def run_op(op_name, fn, native_fn, case, meta_dim):
    """fn(dist) -> MPIXarray or raises; native_fn(native) -> xr object."""
    try:
        result = fn()
    except NotImplementedError as e:
        record(op_name, case, None, f"NotImplementedError (declared limitation): {str(e)[:120]}")
        return
    except Exception as e:
        record(op_name, case, False, f"unexpected {type(e).__name__}: {str(e)[:150]}")
        return

    try:
        local = local_of(result)
        m = result.meta if isinstance(result, MPIXarray) else None
        if m is None:
            record(op_name, case, False, "result lost its MPI metadata")
            return
        expected_full = native_fn()
        if len(m["dims"]) == 1:
            d = m["dims"][0]
            s, e = m["starts"][d], m["stops"][d]
            expected = expected_full.isel({d: slice(s, e)})
        else:
            sel = {d: slice(m["starts"][d], m["stops"][d]) for d in m["dims"]}
            expected = expected_full.isel(sel)
        xr.testing.assert_allclose(local, expected, rtol=1e-6)
        record(op_name, case, True)
    except Exception as e:
        record(op_name, case, False, str(e)[:200])


def rank_gather_and_report():
    all_ok = mpi.comm.gather(RESULTS, root=0)
    if mpi.comm.rank != 0:
        return
    combined: dict[tuple[str, str], list[bool | None]] = {}
    msgs: dict[tuple[str, str], str] = {}
    for rank_results in all_ok:
        for op, case, ok, msg in rank_results:
            key = (op, case)
            combined.setdefault(key, []).append(ok)
            if msg:
                msgs[key] = msg
    print(f"\n=== halo-op results ({mpi.comm.size} ranks) ===")
    for (op, case), oks in combined.items():
        if all(o is None for o in oks):
            status = "SKIP"
        elif all(o for o in oks if o is not None) and all(o is not None for o in oks):
            status = "PASS"
        else:
            status = "FAIL"
        print(f"[{status}] {op:<16} {case:<10} {msgs.get((op, case), '')}")


# ---------------------------------------------------------------------------
# Fixtures: 1D partition (dim='time') and 2D partition (dims=('lat','lon'))
# on the same underlying dataset, so results are directly comparable.
# ---------------------------------------------------------------------------
from mock_dataset import _path, create_dataset  # noqa: E402

create_dataset(n_time=12, resolution_deg=10, plev_step=-250)
mpi.comm.barrier()
native = xr.open_dataset(_path).load()

dist_1d = xgeo.mpi_open_dataset(_path, mpi, partition_dim="time", log_partitions=False)
dist_2d = xgeo.mpi_open_dataset(_path, mpi, partition_dim=("lat", "lon"), log_partitions=False)

cases = [
    ("1d(time)", dist_1d, "time"),
    ("2d(lat,lon)/lat", dist_2d, "lat"),
    ("2d(lat,lon)/lon", dist_2d, "lon"),
]

for case_label, dist, dim in cases:
    run_op(
        "rolling_reduce", lambda d=dist, dm=dim: d.rolling_reduce(dm, window=3, reduce="mean"),
        lambda dm=dim: native.rolling({dm: 3}, center=True).mean(),
        case_label, dim,
    )
    run_op(
        "coarsen_reduce", lambda d=dist, dm=dim: d.coarsen_reduce(dm, window=2, reduce="mean", boundary="trim"),
        lambda dm=dim: native.coarsen({dm: 2}, boundary="trim").mean(),
        case_label, dim,
    )
    run_op(
        "diff", lambda d=dist, dm=dim: d.diff(dm, n=1),
        lambda dm=dim: native.diff(dm, n=1),
        case_label, dim,
    )
    run_op(
        "shift", lambda d=dist, dm=dim: d.shift(dm, periods=1),
        lambda dm=dim: native.shift({dm: 1}),
        case_label, dim,
    )
    run_op(
        "differentiate",
        lambda d=dist, dm=dim: d.differentiate(dm) if dm != "time" else d.differentiate(dm, datetime_unit="s"),
        lambda dm=dim: (native.differentiate(dm) if dm != "time" else native.differentiate(dm, datetime_unit="s")),
        case_label, dim,
    )
    run_op(
        "ffill", lambda d=dist, dm=dim: d.ffill(dm, limit=2),
        lambda dm=dim: native.ffill(dm, limit=2),
        case_label, dim,
    )
    run_op(
        "bfill", lambda d=dist, dm=dim: d.bfill(dm, limit=2),
        lambda dm=dim: native.bfill(dm, limit=2),
        case_label, dim,
    )
    mpi.comm.barrier()

rank_gather_and_report()
