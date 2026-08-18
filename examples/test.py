#!/usr/bin/env python3
"""climtools_test.py -- correctness + speed suite for climtools.mpi.

For every parallel feature this script:
  1. runs the distributed/parallel version across whatever ranks the
     script was launched with (mpirun -n N ...),
  2. runs an equivalent pure-serial baseline on rank 0 alone, over the
     same total amount of data/work,
  3. checks the two results agree (within floating-point tolerance), and
  4. times both and reports the ratio.

Run:

    # single process, serial fallback -- sanity check only, no real
    # parallelism, speedups will be ~1x or worse
    python climtools_test.py

    # real multi-rank run -- this is what actually demonstrates speedups
    mpirun -n 8 python climtools_test.py

    # scale the workload up for a bigger machine
    mpirun -n 16 python climtools_test.py \
        --n-events 2000000 --xreduce-events 40000 --netcdf-steps 500

Speedups from mpi.reduce / mpi.xreduce / the parallel NetCDF writer only
show up when ranks genuinely run on separate cores. On a single-core
machine, or an oversubscribed launch with more ranks than cores, results
will be flat or even slower, since ranks are then time-sliced rather than
run concurrently -- that is expected, not a bug.

Requires climtools importable (e.g. run from the parent of the cloned
repo, or with climtools installed). Parallel NetCDF-4 output additionally
requires netCDF4 built against a parallel-enabled MPI/HDF5/NetCDF-C stack
(see climtools/env/setup_env.sh); if that support is missing, the NetCDF
write test is skipped automatically when running with more than one rank.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import climtools
from climtools.core.xgeo import empty_dataset, to_netcdf

mpi = climtools.mpi
RANK: int = mpi.comm.rank
SIZE: int = mpi.comm.size


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


@contextmanager
def timed():
    """Time a block the same way on every rank, reporting the slowest rank.

    Uses the real climtools/mpi4py API throughout, not plain Python
    timing: `mpi.MPI.Wtime()` (mpi4py's MPI-aware timer, reached exactly
    the way the README documents -- "anything not covered above ... is
    reached directly as mpi.comm.<method>"/`mpi.MPI`) for the clock, and
    `mpi.reduce.max` (the very reduction this suite is testing) to combine
    each rank's elapsed time into the slowest rank's time -- the wall time
    a caller waiting on the whole collective actually experiences, not any
    single rank's local time, which could understate how long the group as
    a whole took. Barriers before and after make every rank start and stop
    together.
    """
    box = {"seconds": 0.0}
    mpi.comm.barrier()
    start = mpi.MPI.Wtime()
    yield box
    mpi.comm.barrier()
    local_elapsed = mpi.MPI.Wtime() - start
    box["seconds"] = mpi.reduce.max(local_elapsed)


def run_serial_baseline(fn):
    """Run `fn` on rank 0 only and get (result, elapsed) back on every rank.

    This is `@mpi(broadcast=True)` -- "execute on root and broadcast its
    return value to every rank" -- applied directly, rather than
    hand-rolling the same root-only-then-broadcast pattern with raw
    `mpi.comm` calls. Every non-root rank simply waits inside the
    decorator's own synchronization for root's timed result, which is
    exactly the single-process cost a script with no MPI at all would pay.
    """

    @mpi(broadcast=True)
    def _timed_on_root():
        start = mpi.MPI.Wtime()
        result = fn()
        elapsed = mpi.MPI.Wtime() - start
        return result, elapsed

    return _timed_on_root()


@dataclass
class Result:
    name: str
    correct: bool
    serial_s: float
    parallel_s: float
    note: str = ""

    @property
    def speedup(self) -> float:
        if self.serial_s <= 0.0 or self.parallel_s <= 0.0:
            return float("nan")
        return self.serial_s / self.parallel_s


RESULTS: list[Result] = []


def record(
    name: str,
    correct: bool,
    serial_s: float,
    parallel_s: float,
    note: str = "",
) -> None:
    result = Result(name, correct, serial_s, parallel_s, note)
    RESULTS.append(result)
    status = "OK  " if correct else "FAIL"
    speedup_str = "  n/a " if np.isnan(result.speedup) else f"{result.speedup:5.2f}x"
    mpi.log(
        f"[{status}] {name:<46} serial={serial_s:8.4f}s  "
        f"parallel={parallel_s:8.4f}s  speedup={speedup_str}"
        + (f"  ({note})" if note else "")
    )


def safe_run(fn, *args, **kwargs) -> None:
    """Run a test function on every rank, synchronizing any failure.

    Mirrors the pattern climtools.mpi itself uses internally
    (mpi.raise_if_error): if a test raises on some ranks but not others, an
    un-synchronized try/except per rank would leave the failing rank done
    while the others hang forever at the test's next collective call. This
    lets one test's failure stop that test cleanly without deadlocking the
    remaining ranks, and without aborting the rest of the suite.
    """
    error = None
    try:
        fn(*args, **kwargs)
    except BaseException as exc:
        error = exc
    try:
        mpi.raise_if_error(error, fn.__name__)
    except mpi.MPIError as exc:
        mpi.log(f"[ERROR] {fn.__name__} failed: {exc}")


# ---------------------------------------------------------------------------
# mpi.reduce -- element-wise collective reductions
# ---------------------------------------------------------------------------


def test_reduce_sum_scalar(n_total: int) -> None:
    """Scalar mpi.reduce.sum: each rank contributes one partial sum."""
    per_rank = max(1, n_total // SIZE)

    def make_local(rank: int) -> np.ndarray:
        rng = np.random.default_rng(1000 + rank)
        return rng.standard_normal(per_rank).astype(np.float64)

    local = make_local(RANK)

    with timed() as box:
        local_partial = float(local.sum())
        combined = mpi.reduce.sum(local_partial)
    parallel_s = box["seconds"]

    def serial_fn() -> float:
        return sum(float(make_local(r).sum()) for r in range(SIZE))

    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(np.isclose(combined, expected, rtol=1e-8))
    record(
        f"mpi.reduce.sum scalar ({n_total} values total)",
        correct,
        serial_s,
        parallel_s,
    )


def test_reduce_composite(n_events_total: int, ny: int, nx: int) -> None:
    """mpi.reduce.sum on a NumPy array: distributed point-event binning.

    Mirrors the README's parallel-output pattern: each rank owns a disjoint
    share of point events, bins its own share onto a shared (ny, nx) grid,
    and mpi.reduce.sum combines every rank's partial grid into one global
    composite. Compared against binning every event serially on one rank.
    """
    per_rank = max(1, n_events_total // SIZE)

    def make_events(rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(5000 + rank)
        lat_idx = rng.integers(0, ny, size=per_rank)
        lon_idx = rng.integers(0, nx, size=per_rank)
        values = rng.standard_normal(per_rank)
        return lat_idx, lon_idx, values

    def bin_events(
        lat_idx: np.ndarray, lon_idx: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        grid = np.zeros((ny, nx), dtype=np.float64)
        np.add.at(grid, (lat_idx, lon_idx), values)
        return grid

    lat_idx, lon_idx, values = make_events(RANK)

    with timed() as box:
        local_grid = bin_events(lat_idx, lon_idx, values)
        combined = mpi.reduce.sum(local_grid)
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        grid = np.zeros((ny, nx), dtype=np.float64)
        for r in range(SIZE):
            li, lo, v = make_events(r)
            np.add.at(grid, (li, lo), v)
        return grid

    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(np.allclose(combined, expected))
    record(
        f"mpi.reduce.sum composite grid ({ny}x{nx}, {n_events_total} events)",
        correct,
        serial_s,
        parallel_s,
    )


def test_reduce_xarray_object(ny: int, nx: int) -> None:
    """mpi.reduce.sum on an xarray DataArray: dims/coords/attrs preserved."""
    da = xr.DataArray(
        np.full((ny, nx), RANK + 1.0),
        dims=("y", "x"),
        coords={"y": np.arange(ny), "x": np.arange(nx)},
        attrs={"units": "K"},
    )

    with timed() as box:
        combined = mpi.reduce.sum(da)
    parallel_s = box["seconds"]

    def serial_fn() -> float:
        return float(sum(r + 1.0 for r in range(SIZE)))

    expected_scalar, serial_s = run_serial_baseline(serial_fn)
    correct = (
        bool(np.allclose(combined.values, expected_scalar))
        and combined.attrs.get("units") == "K"
    )
    record(
        "mpi.reduce.sum xarray DataArray (dims/attrs kept)",
        correct,
        serial_s,
        parallel_s,
        note="tiny op, correctness-focused",
    )


# ---------------------------------------------------------------------------
# mpi.xreduce -- distributed xarray dimension reductions
# ---------------------------------------------------------------------------


def test_xreduce(n_events_total: int, ny: int, nx: int, op_name: str) -> None:
    """mpi.xreduce.<op> vs plain xarray's <op> on the fully assembled array."""
    per_rank = max(1, n_events_total // SIZE)

    def make_local(rank: int) -> xr.DataArray:
        rng = np.random.default_rng(7000 + rank)
        data = rng.standard_normal((per_rank, ny, nx)).astype(np.float64)
        return xr.DataArray(data, dims=("event", "y", "x"))

    local = make_local(RANK)
    op = getattr(mpi.xreduce, op_name)

    with timed() as box:
        result = op(local, dim="event")
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        parts = [make_local(r).values for r in range(SIZE)]
        full = xr.DataArray(np.concatenate(parts, axis=0), dims=("event", "y", "x"))
        return getattr(full, op_name)(dim="event").values

    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(np.allclose(result.values, expected, rtol=1e-9))
    record(
        f"mpi.xreduce.{op_name} ({n_events_total} events, {ny}x{nx} field)",
        correct,
        serial_s,
        parallel_s,
    )


# ---------------------------------------------------------------------------
# mpi.scatterv -- vector scatter (data movement, not a compute reduction)
# ---------------------------------------------------------------------------


def test_scatterv(n_total: int) -> None:
    counts = [n_total // SIZE + (1 if r < n_total % SIZE else 0) for r in range(SIZE)]
    total = sum(counts)

    with timed() as box:
        source = None
        if RANK == 0:
            source = np.arange(total * 3, dtype=np.float64).reshape(total, 3)
        recv = mpi.scatterv(source, counts, (counts[RANK], 3), np.float64, root=0)
    parallel_s = box["seconds"]

    start = sum(counts[:RANK])
    expected_local = np.arange(total * 3, dtype=np.float64).reshape(total, 3)[
        start : start + counts[RANK]
    ]
    correct = bool(np.allclose(recv, expected_local))
    record(
        f"mpi.scatterv ({total} rows across {SIZE} rank(s))",
        correct,
        0.0,
        parallel_s,
        note="data movement, no serial-compute equivalent",
    )


# ---------------------------------------------------------------------------
# A realistic xarray + mpi.reduce composition: cosine-latitude weighted mean
# ---------------------------------------------------------------------------


def test_weighted_mean(n_lat_total: int, n_lon: int) -> None:
    """Cosine-latitude weighted global mean via distributed partial sums.

    Each rank holds a disjoint latitude band, computes its own weighted
    partial sums with ordinary xarray arithmetic, and mpi.reduce.sum
    combines the numerator and denominator across ranks before dividing --
    demonstrating xarray operations composed with mpi.reduce, not just
    mpi.reduce in isolation.
    """
    per_rank = max(1, n_lat_total // SIZE)

    def make_local(rank: int) -> xr.DataArray:
        lat0 = rank * per_rank
        lat = np.linspace(-90.0, 90.0, n_lat_total)[lat0 : lat0 + per_rank]
        rng = np.random.default_rng(9000 + rank)
        data = rng.standard_normal((per_rank, n_lon)).astype(np.float64)
        return xr.DataArray(
            data, dims=("lat", "lon"), coords={"lat": lat, "lon": np.arange(n_lon)}
        )

    local = make_local(RANK)

    with timed() as box:
        weights = np.cos(np.deg2rad(local["lat"]))
        local_weighted_sum = (local * weights).sum()
        local_weight_sum = (xr.ones_like(local) * weights).sum()
        global_weighted_sum = mpi.reduce.sum(float(local_weighted_sum))
        global_weight_sum = mpi.reduce.sum(float(local_weight_sum))
        weighted_mean = global_weighted_sum / global_weight_sum
    parallel_s = box["seconds"]

    def serial_fn() -> float:
        parts = [make_local(r) for r in range(SIZE)]
        full = xr.concat(parts, dim="lat")
        w = np.cos(np.deg2rad(full["lat"]))
        return float((full * w).sum() / (xr.ones_like(full) * w).sum())

    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(np.isclose(weighted_mean, expected, rtol=1e-8))
    record(
        f"cosine-lat weighted mean ({n_lat_total}x{n_lon}, xarray + mpi.reduce)",
        correct,
        serial_s,
        parallel_s,
    )


# ---------------------------------------------------------------------------
# @mpi decorator -- usage demonstration and correctness checks
# ---------------------------------------------------------------------------


def test_mpi_decorator() -> None:
    mpi.log("\n--- @mpi decorator usage ---")

    # 1) Bare @mpi: runs only on rank 0 (the default root), returns None on
    #    every other rank.
    @mpi
    def only_on_root() -> str:
        return f"computed on rank {RANK}"

    root_result = only_on_root()
    ok_root_only = (RANK == 0 and root_result == "computed on rank 0") or (
        RANK != 0 and root_result is None
    )
    mpi.log(f"  @mpi                    (root-only): rank 0 result = {root_result!r}")

    # 2) @mpi(all_ranks=True): every rank runs the function independently
    #    and keeps its own return value -- no combining happens.
    @mpi(all_ranks=True)
    def on_every_rank() -> int:
        return RANK

    all_ranks_result = on_every_rank()
    ok_all_ranks = all_ranks_result == RANK
    mpi.log("  @mpi(all_ranks=True)    : every rank returns its own rank id")

    # 3) @mpi(broadcast=True): root computes once; every rank (including
    #    root) ends up with the identical, broadcast result.
    @mpi(broadcast=True)
    def expensive_setup() -> dict:
        return {"config_value": 42, "computed_on_rank": RANK}

    cfg = expensive_setup()
    ok_broadcast = cfg["config_value"] == 42 and cfg["computed_on_rank"] == 0
    mpi.log(f"  @mpi(broadcast=True)    : every rank sees rank 0's result: {cfg}")

    # 4) A failure on the executing rank(s) is raised as a synchronized
    #    error on every rank in the communicator, instead of leaving the
    #    other ranks hanging forever at the next collective call the
    #    failed rank never reaches. When only a strict subset of ranks
    #    fail, mpi.raise_if_error wraps it as a catchable
    #    climtools.mpi.MPIError; when every rank in the communicator fails
    #    (as happens here when running on a single rank, since rank 0 is
    #    then the only rank), the original exception type is re-raised
    #    instead -- both are demonstrated below.
    @mpi
    def fails_on_root() -> str:
        if RANK == 0:
            raise ValueError("deliberate failure for this demo")
        return "unreached"

    ok_error_propagation = False
    try:
        fails_on_root()
    except (mpi.MPIError, ValueError) as exc:
        ok_error_propagation = True
        mpi.log(
            f"  @mpi error propagation  : caught {type(exc).__name__} on every rank: {exc}"
        )

    overall = ok_root_only and ok_all_ranks and ok_broadcast and ok_error_propagation
    record(
        "@mpi decorator (root/all_ranks/broadcast/error)",
        overall,
        0.0,
        0.0,
        note="usage demo, not a speed test",
    )


# ---------------------------------------------------------------------------
# NetCDF write: MPI-collective parallel writer vs ordinary serial writer
# ---------------------------------------------------------------------------


def test_netcdf_write(n_time: int, ny: int, nx: int, out_dir: str) -> None:
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        mpi.log(
            "\n--- NetCDF write speed test SKIPPED: netCDF4 lacks parallel4 "
            "support (see climtools/env/setup_env.sh) ---"
        )
        return

    mpi.log("\n--- NetCDF write: parallel collective vs serial ---")

    def make_full_dataset() -> xr.Dataset:
        rng = np.random.default_rng(42)
        times = pd.date_range("2020-01-01", periods=n_time, freq="6h")
        return xr.Dataset(
            {
                "precipitation": xr.DataArray(
                    rng.random((n_time, ny, nx)).astype(np.float32),
                    dims=("time", "y", "x"),
                    coords={"time": times},
                    attrs={"units": "mm/day", "long_name": "precipitation rate"},
                )
            }
        )

    parallel_path = os.path.join(out_dir, "climtools_test_parallel.nc")
    serial_path = os.path.join(out_dir, "climtools_test_serial.nc")

    with timed() as box:
        ds = make_full_dataset() if RANK == 0 else empty_dataset()
        to_netcdf(
            ds,
            parallel_path,
            unlimited_dim="time",
            partition_dim="time",
            parallel=True,
            allow_serial=(SIZE == 1),
        )
    parallel_s = box["seconds"]

    def serial_fn() -> None:
        full = make_full_dataset()
        to_netcdf(full, serial_path, unlimited_dim="time", show_progress=False)

    _, serial_s = run_serial_baseline(serial_fn)

    correct = True
    if RANK == 0:
        with xr.open_dataset(parallel_path) as a, xr.open_dataset(serial_path) as b:
            correct = bool(
                np.allclose(a["precipitation"].values, b["precipitation"].values)
            ) and np.array_equal(a["time"].values, b["time"].values)
    correct = mpi.comm.bcast(correct, root=0)
    record(
        f"NetCDF write ({n_time} steps, {ny}x{nx}, float32)",
        correct,
        serial_s,
        parallel_s,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="climtools MPI test/benchmark suite")
    parser.add_argument(
        "--n-events", type=int, default=2_000_000, help="mpi.reduce tests"
    )
    parser.add_argument("--grid-ny", type=int, default=180)
    parser.add_argument("--grid-nx", type=int, default=360)
    parser.add_argument(
        "--xreduce-events", type=int, default=5_000, help="events for mpi.xreduce tests"
    )
    parser.add_argument("--xreduce-ny", type=int, default=40)
    parser.add_argument("--xreduce-nx", type=int, default=40)
    parser.add_argument("--n-lat", type=int, default=180, help="weighted-mean test")
    parser.add_argument("--n-lon", type=int, default=360)
    parser.add_argument("--netcdf-steps", type=int, default=200)
    parser.add_argument("--netcdf-ny", type=int, default=200)
    parser.add_argument("--netcdf-nx", type=int, default=200)
    parser.add_argument(
        "--out-dir", type=str, default=str(Path.home() / "scratch" / "io_mpi_test")
    )
    parser.add_argument("--skip-netcdf", action="store_true")
    return parser.parse_args()


def print_summary() -> None:
    mpi.log("\n" + "=" * 88)
    mpi.log(f"SUMMARY -- {SIZE} rank(s)")
    mpi.log("=" * 88)
    for result in RESULTS:
        speedup_str = (
            "  n/a " if np.isnan(result.speedup) else f"{result.speedup:5.2f}x"
        )
        status = "OK  " if result.correct else "FAIL"
        mpi.log(
            f"[{status}] {result.name:<52} speedup={speedup_str}  "
            f"serial={result.serial_s:7.4f}s  parallel={result.parallel_s:7.4f}s"
        )
    mpi.log("-" * 88)
    n_fail = sum(1 for r in RESULTS if not r.correct)
    if n_fail:
        mpi.log(f"{n_fail} test(s) FAILED: parallel and serial results disagree.")
    else:
        mpi.log("All tests: parallel and serial results agree.")
    if SIZE == 1:
        mpi.log(
            "\nRan on 1 rank: speedups will be ~1x or worse. mpi.reduce/mpi.xreduce/\n"
            "the parallel NetCDF writer all still pay collective-call overhead even\n"
            "with nothing to parallelize against. Run `mpirun -n N python "
            "climtools_test.py`\nwith N >= 2 real cores to see actual speedups."
        )
    else:
        n_cpus = os.cpu_count() or 1
        if SIZE > n_cpus:
            mpi.log(
                f"\nNote: {SIZE} ranks launched on a machine reporting {n_cpus} CPU(s) "
                "(os.cpu_count()).\nOversubscribed ranks are time-sliced rather than "
                "run concurrently, which caps or\ncan even invert the speedups above; "
                "for a clean comparison, run with N <= cores."
            )


def main() -> None:
    args = parse_args()

    if RANK == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    mpi.comm.barrier()

    mpi.log("=" * 88)
    mpi.log(f"climtools MPI test suite -- {SIZE} rank(s), mpi.launched={mpi.launched}")
    mpi.log(
        "mpi4py initializes MPI on import and finalizes automatically at exit; "
        "see the mpi4py Overview docs (https://mpi4py.readthedocs.io/en/stable/"
        "overview.html) for the underlying collective/error-handling semantics "
        "climtools.mpi builds on."
    )
    mpi.log("=" * 88)

    mpi.log("\n--- mpi.reduce ---")
    safe_run(test_reduce_sum_scalar, args.n_events)
    safe_run(test_reduce_composite, args.n_events, args.grid_ny, args.grid_nx)
    safe_run(test_reduce_xarray_object, args.grid_ny, args.grid_nx)

    mpi.log("\n--- mpi.xreduce (vs plain xarray on the assembled array) ---")
    for op_name in ("sum", "mean", "max", "min"):
        safe_run(
            test_xreduce,
            args.xreduce_events,
            args.xreduce_ny,
            args.xreduce_nx,
            op_name,
        )

    mpi.log("\n--- mpi.scatterv ---")
    safe_run(test_scatterv, args.n_events)

    mpi.log("\n--- xarray operations + mpi.reduce ---")
    safe_run(test_weighted_mean, args.n_lat, args.n_lon)

    safe_run(test_mpi_decorator)

    if not args.skip_netcdf:
        safe_run(
            test_netcdf_write,
            args.netcdf_steps,
            args.netcdf_ny,
            args.netcdf_nx,
            args.out_dir,
        )

    mpi.comm.barrier()
    print_summary()


if __name__ == "__main__":
    main()
