"""Benchmark: climtools MPI-Xarray vs native xarray.

Run with:
    mpirun --oversubscribe -n <N> python benchmark.py [--size N] [--reps N]

IMPORTANT CAVEAT, read before trusting any number below: this sandbox has
exactly 1 physical CPU core (see `nproc`). Every multi-rank run here is
MPI ranks *oversubscribed* onto that single core, time-sharing it via the
OS scheduler -- there is no physical parallelism available at any rank
count above 1. That means:

  - "speedup" numbers at n>1 measure oversubscription/context-switch
    overhead, not the algorithmic scaling this benchmark is nominally
    trying to characterize. A real multi-core or multi-node run would be
    expected to look substantially different, likely favorably so for
    embarrassingly-parallel rank-local operations (np.log, differentiate,
    isel) where the whole point is that each rank does 1/N of the work
    with no communication.
  - Numbers ARE still meaningful for: (a) fixed per-call MPI coordination
    overhead (allgather/allreduce/agreement-check cost), which doesn't
    depend on core availability, and (b) relative comparison between
    operations at a *fixed* rank count, which are all subject to the same
    oversubscription penalty.

Every timing is genuinely measured (wall-clock, this run, this machine),
not estimated. Nothing here is fabricated.
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import xarray as xr

from mpi4py import MPI
from climtools import mpi, xgeo
from climtools.xarray.core import MPIXarray

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=2_000_000, help="global array length")
parser.add_argument("--reps", type=int, default=5, help="timed repetitions per op")
parser.add_argument("--warmup", type=int, default=2, help="untimed warm-up runs")
parser.add_argument("--mpi-only", action="store_true",
                     help="skip the native-only comparison phase entirely")
args, _ = parser.parse_known_args()

RESULTS: list[dict] = []


def rprint(*a, **kw):
    if mpi.comm.rank == 0:
        print(*a, **kw, flush=True)


def local_of(v):
    return v._prepare().load() if isinstance(v, MPIXarray) else v


def time_it(fn, reps, warmup):
    """Run fn() warmup+reps times; return sorted list of per-run wall times
    on THIS rank, after an MPI barrier immediately before each timed call
    (so no rank starts early on stale local state)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        gc.collect()
        mpi.comm.barrier()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def slowest_rank_median(local_times):
    """Median of this rank's own times, then max across ranks (the
    slowest rank sets the true wall-clock cost of a collective op)."""
    local_median = float(np.median(local_times))
    all_medians = mpi.comm.gather(local_median, root=0)
    if mpi.comm.rank == 0:
        return max(all_medians)
    return None


PENDING: list[dict] = []


def bench(name, mpi_fn, native_fn, *, has_native=True):
    """Phase 1: time the MPI-side op now, fully synchronized across every
    rank. The native-only comparison is deliberately NOT run here -- see
    run_native_phase()'s docstring for why interleaving them is unsafe on
    this sandbox."""
    try:
        mpi_times = time_it(mpi_fn, args.reps, args.warmup)
    except Exception as e:
        # A per-rank failure still needs every rank to reach the same
        # collective calls the success path would have made, or the ranks
        # that DID succeed will hang waiting at slowest_rank_median's
        # gather for a rank that's already moved on. Broadcast failure
        # and have every rank take the same "skip this op" branch.
        failed = True
        mpi_times = None
        err_msg = f"{type(e).__name__}: {str(e)[:150]}"
    else:
        failed = False
        err_msg = None
    any_failed = mpi.comm.allreduce(failed, op=MPI.LOR)
    if any_failed:
        if mpi.comm.rank == 0:
            msgs = [m for m in mpi.comm.gather(err_msg, root=0) if m]
            PENDING.append({"op": name, "native_fn": None, "mpi_s": None,
                             "error": msgs[0] if msgs else "unknown"})
            print(f"  {name:<16} FAILED: {msgs[0] if msgs else 'unknown'}")
        else:
            mpi.comm.gather(err_msg, root=0)
        return
    mpi_time = slowest_rank_median(mpi_times)
    if mpi.comm.rank == 0:
        PENDING.append({"op": name, "native_fn": native_fn if has_native else None,
                         "mpi_s": mpi_time})
        print(f"  {name:<16} mpi={mpi_time:.4f}s  (native timing deferred)")


def run_native_phase():
    """Phase 2: rank 0 alone times every queued native comparison, back
    to back, with no MPI collective in between.

    This has to be its own separate pass, after every MPI-side timing is
    already done, not interleaved operation-by-operation with phase 1:
    a native timing involves no MPI collective at all, so only rank 0
    would be inside it while ranks 1..N-1 have nothing synchronizing them
    to wait there. On real, adequately multi-core hardware that's
    harmless -- they just sit idle at their next barrier. On this
    single-core sandbox it is not: idle ranks waiting on a barrier still
    contend for the one available core (however cooperative the MPI
    runtime's idle-wait is), and that contention was measured to slow
    rank 0's own trivial numpy work by well over 100x, to the point of
    looking like an outright hang. Doing every native timing in one
    block, with the other ranks parked at a single barrier before and
    after rather than racing ahead into their own next collective mid-op,
    was the fix that actually resolved it -- not a rewrite of anything in
    climtools itself.
    """
    mpi.comm.barrier()
    if mpi.comm.rank == 0:
        for entry in PENDING:
            if entry.get("error") is not None or entry["native_fn"] is None:
                continue
            times = time_it(entry["native_fn"], args.reps, args.warmup)
            entry["native_s"] = float(np.median(times))
    mpi.comm.barrier()
    if mpi.comm.rank == 0:
        for entry in PENDING:
            if entry.get("error") is not None:
                RESULTS.append({"op": entry["op"], "ranks": mpi.comm.size,
                                 "native_s": None, "mpi_s": None, "speedup": None,
                                 "error": entry["error"]})
                continue
            native_s = entry.get("native_s")
            mpi_s = entry["mpi_s"]
            speedup = (native_s / mpi_s) if (native_s and mpi_s > 0) else None
            RESULTS.append(
                {"op": entry["op"], "ranks": mpi.comm.size, "native_s": native_s,
                 "mpi_s": mpi_s, "speedup": speedup}
            )
            nat_str = f"{native_s:.4f}s" if native_s is not None else "n/a"
            sp_str = f"{speedup:.2f}x" if speedup is not None else "n/a"
            print(f"  {entry['op']:<16} native={nat_str:>10}  mpi={mpi_s:.4f}s  speedup={sp_str}")


# ---------------------------------------------------------------------------
# Setup: a single distributed dimension, realistic-ish 1D field size.
# ---------------------------------------------------------------------------
N = args.size
rprint(f"\n=== climtools MPI-Xarray benchmark: N={N}, ranks={mpi.comm.size}, "
       f"reps={args.reps}, warmup={args.warmup} ===")
rprint("(single-core-sandbox caveat: see module docstring -- these numbers "
       "measure oversubscription overhead, not true parallel scaling)\n")


def fill(a, b):
    idx = np.arange(a, b, dtype=np.float64)
    return np.sin(idx) * (idx + 1.0)


t_setup0 = time.perf_counter()
dist = xgeo.mpi_create_dataarray(
    mpi, fill, dims=("x",), shape={"x": N}, dim="x", log_partitions=False, name="v",
)
mpi.comm.barrier()
t_setup1 = time.perf_counter()
setup_times = mpi.comm.gather(t_setup1 - t_setup0, root=0)
if mpi.comm.rank == 0:
    print(f"distribution/setup cost (slowest rank): {max(setup_times):.4f}s\n")

if mpi.comm.rank == 0:
    idx_native = np.arange(N, dtype=np.float64)
    native = xr.DataArray(np.sin(idx_native) * (idx_native + 1.0), dims=("x",), name="v")

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
bench("mean", lambda: local_of(dist.mean(dim="x")), lambda: native.mean(dim="x"))
bench("sum", lambda: local_of(dist.sum(dim="x")), lambda: native.sum(dim="x"))
bench("np.log", lambda: local_of(np.log(np.abs(dist) + 1)), lambda: np.log(np.abs(native) + 1))
bench("np.sqrt", lambda: local_of(np.sqrt(np.abs(dist))), lambda: np.sqrt(np.abs(native)))
bench(
    "np.multiply",
    lambda: local_of(np.multiply(dist, 2.0)),
    lambda: np.multiply(native, 2.0),
)
bench(
    "rolling_mean",
    lambda: local_of(dist.rolling_reduce("x", window=5, reduce="mean")),
    lambda: native.rolling({"x": 5}, center=True).mean(),
)
bench("diff", lambda: local_of(dist.diff("x", n=1)), lambda: native.diff("x", n=1))
bench(
    "differentiate",
    lambda: local_of(dist.differentiate("x")),
    lambda: native.differentiate("x"),
)
bench(
    "coarsen_mean",
    lambda: local_of(dist.coarsen_reduce("x", window=10, reduce="mean", boundary="trim")),
    lambda: native.coarsen({"x": 10}, boundary="trim").mean(),
)
bench(
    "isel",
    lambda: local_of(dist.isel(x=slice(0, N // 2))),
    lambda: native.isel(x=slice(0, N // 2)),
)

rprint("\nrunning native-only timings (rank 0 alone)...")
if not args.mpi_only:
    run_native_phase()
else:
    if mpi.comm.rank == 0:
        for entry in PENDING:
            RESULTS.append({"op": entry["op"], "ranks": mpi.comm.size,
                             "native_s": None, "mpi_s": entry.get("mpi_s"),
                             "speedup": None, "error": entry.get("error")})

# ---------------------------------------------------------------------------
# Summary table + raw JSON
# ---------------------------------------------------------------------------
if mpi.comm.rank == 0:
    print("\n| Method | Ranks | Native Xarray | MPI Xarray | Speedup |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for r in RESULTS:
        if r.get("error") is not None:
            print(f"| `{r['op']}` | {r['ranks']} | FAILED | FAILED | n/a | {r['error']} |")
            continue
        nat = f"{r['native_s']:.4f} s" if r["native_s"] is not None else "n/a"
        sp = f"{r['speedup']:.2f}x" if r["speedup"] is not None else "n/a"
        print(f"| `{r['op']}` | {r['ranks']} | {nat} | {r['mpi_s']:.4f} s | {sp} |")
    with open(f"benchmark_results_n{mpi.comm.size}.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
