"""Benchmark MPIXarray's public operations against native, single-process xarray.

Usage
-----
Run under MPI with more than one rank -- a single-process run cannot
measure any distributed speedup and is only useful as a smoke test::

    mpirun -n <N> python -m mpi4py test/benchmark_mpi_xarray.py \\
        [--n-time N] [--resolution-deg D] [--plev-step S] [--repeats R]

Or via the SLURM script (``test/test.sh``), which generates its own
dataset with ``tests/mock_dataset.py`` at a size suited to the
allocation.

What this measures
-------------------
For each public operation benchmarked, this script:

1. Times plain, single-process xarray performing the identical
   operation on the complete, in-memory dataset (loaded once on rank
   0, replicated to every rank so every rank's native timing is on
   the same data -- the timing itself only runs on rank 0).
2. Times MPIXarray performing the same operation, distributed across
   every rank in ``MPI_COMM_WORLD``, wall-clock bounded by an
   ``MPI.Barrier`` immediately before and after so every rank agrees
   on the same window (the reported time is the slowest rank's own
   elapsed time -- the wait any real caller actually experiences).
3. Gathers MPIXarray's distributed result back to rank 0 and checks
   it against the native result for value and dtype agreement.
4. Reports ``speedup = native_xarray_time / mpi_xarray_time`` -- a
   number below 1 means the distributed path was *slower* than
   plain xarray for this operation at this size and rank count, which
   is expected for a small dataset (communication overhead dominates)
   and is exactly the information this script exists to surface.

This script deliberately does not commit or assume any output: it is
meant to be run locally or via SLURM at whatever scale the caller
chooses, and prints its summary table to stdout on rank 0 only.

Design notes
------------
- Every timed region runs each operation ``--repeats`` times (default
  3) and reports the minimum, following the usual microbenchmark
  convention of discarding the effect of first-call warmup (Python
  import caches, MPI's own lazy connection setup) and transient
  scheduling noise, while still surfacing genuine best-case cost.
- Distributed operands are always constructed once per operation from
  a fresh ``MPIXarray(...)`` (not reused across repeats), so no
  timing includes another operation's leftover state.
- Correctness is checked with a moderate ``rtol``/``atol`` (see
  ``_CLOSE_KWARGS``) because several operations (rolling means,
  centered differences) legitimately accumulate float32 rounding
  differences between native xarray's own reduction order and this
  package's halo-exchange-based one -- verified case-by-case during
  development against `test/test_*.py`'s tighter, targeted checks.
  This script's tolerance is for *regression detection* at benchmark
  scale, not a substitute for those unit tests.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from climtools import mpi
from climtools.xarray.core import MPIXarray
from mock_dataset import build_dataset

_CLOSE_KWARGS = {"rtol": 1e-3, "atol": 1e-4}


@dataclass
class BenchResult:
    name: str
    native_time: float
    mpix_time: float
    correct: bool
    dtype_match: bool
    error: str | None = None

    @property
    def speedup(self) -> float:
        return self.native_time / self.mpix_time if self.mpix_time > 0 else float("nan")


def _gather_full(mx_obj: MPIXarray):
    meta = mx_obj.meta
    data = mx_obj.data
    if meta is None:
        return data
    pieces = mpi.comm.allgather(data)
    dim = meta["dim"]
    if isinstance(data, xr.Dataset):
        return xr.concat(pieces, dim=dim, data_vars="minimal")
    return xr.concat(pieces, dim=dim)


def _time_repeated(fn, repeats: int) -> tuple[float, object]:
    """Run ``fn()`` ``repeats`` times, returning (min elapsed seconds, last result)."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return best, result


def _time_distributed(fn, repeats: int) -> tuple[float, object]:
    """Like _time_repeated, but bounded by a Barrier and reduced to the slowest rank."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        mpi.comm.Barrier()
        start = time.perf_counter()
        result = fn()
        mpi.comm.Barrier()
        elapsed = time.perf_counter() - start
        elapsed = mpi.comm.allreduce(elapsed, op=__import__("mpi4py").MPI.MAX)
        best = min(best, elapsed)
    return best, result


def _check(native_result, mx_result) -> tuple[bool, bool]:
    """Return (values_match, dtype_match) on rank 0; True/True elsewhere (unchecked)."""
    gathered = _gather_full(mx_result) if isinstance(mx_result, MPIXarray) else mx_result
    if mpi.comm.rank != 0:
        return True, True
    try:
        xr.testing.assert_allclose(native_result, gathered, **_CLOSE_KWARGS)
        values_match = True
    except Exception:
        values_match = False
    try:
        if isinstance(native_result, xr.Dataset):
            dtype_match = all(
                native_result[name].dtype == gathered[name].dtype
                for name in native_result.data_vars
            )
        else:
            dtype_match = native_result.dtype == gathered.dtype
    except Exception:
        dtype_match = False
    return values_match, dtype_match


def run_benchmark(
    name: str,
    native_fn,
    mpix_fn,
    native_ref,
    repeats: int,
) -> BenchResult:
    error = None
    try:
        native_time, native_result = _time_repeated(native_fn, repeats)
    except Exception as exc:  # pragma: no cover - defensive, reported not raised
        return BenchResult(name, float("nan"), float("nan"), False, False, error=f"native: {exc!r}")

    try:
        mpix_time, mpix_result = _time_distributed(mpix_fn, repeats)
    except Exception as exc:  # pragma: no cover
        return BenchResult(name, native_time, float("nan"), False, False, error=f"mpix: {exc!r}")

    try:
        correct, dtype_match = _check(native_ref if native_ref is not None else native_result, mpix_result)
    except Exception as exc:  # pragma: no cover
        return BenchResult(name, native_time, mpix_time, False, False, error=f"check: {exc!r}")

    return BenchResult(name, native_time, mpix_time, correct, dtype_match, error=error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-time", type=int, default=60)
    parser.add_argument("--resolution-deg", type=float, default=0.5)
    parser.add_argument("--plev-step", type=float, default=-25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--partition-dim", type=str, default="lat")
    args = parser.parse_args()

    rank = mpi.comm.rank
    size = mpi.comm.size
    if size < 2 and rank == 0:
        print(
            "WARNING: running with a single rank. This script still runs "
            "correctly, but a distributed speedup cannot be measured -- "
            "use `mpirun -n <N> ...` with N > 1 for meaningful numbers.",
            file=sys.stderr,
        )

    import tempfile

    tmpdir = tempfile.mkdtemp() if rank == 0 else None
    tmpdir = mpi.comm.bcast(tmpdir, root=0)
    path = Path(tmpdir) / "benchmark_dataset.nc"
    if rank == 0:
        build_dataset(path, args.n_time, args.resolution_deg, args.plev_step)
    mpi.comm.Barrier()

    ds_native = xr.open_dataset(path).load()
    if rank == 0:
        nbytes = ds_native.nbytes / 1e6
        print(f"Dataset: n_time={args.n_time} resolution_deg={args.resolution_deg} "
              f"plev_step={args.plev_step} -> {nbytes:.1f} MB, {size} ranks, "
              f"{args.repeats} repeats (reporting minimum)\n")

    def fresh_mx() -> MPIXarray:
        return MPIXarray(ds_native.copy(deep=True), mpi, dim=args.partition_dim)

    results: list[BenchResult] = []

    # sum / mean / min / max
    for reducer in ("sum", "mean", "min", "max", "std"):
        results.append(
            run_benchmark(
                f"{reducer}(dim={args.partition_dim})",
                lambda r=reducer: getattr(ds_native, r)(dim=args.partition_dim, skipna=True),
                lambda r=reducer: getattr(fresh_mx(), r)(dim=args.partition_dim, skipna=True),
                None,
                args.repeats,
            )
        )

    # diff / shift / roll
    results.append(
        run_benchmark(
            f"diff(dim={args.partition_dim}, n=1)",
            lambda: ds_native.diff(dim=args.partition_dim, n=1),
            lambda: fresh_mx().diff(dim=args.partition_dim, n=1),
            None,
            args.repeats,
        )
    )
    results.append(
        run_benchmark(
            f"shift(dim={args.partition_dim}, periods=3)",
            lambda: ds_native.shift({args.partition_dim: 3}),
            lambda: fresh_mx().shift(dim=args.partition_dim, periods=3),
            None,
            args.repeats,
        )
    )
    results.append(
        run_benchmark(
            f"roll(dim={args.partition_dim}, shift_by=5)",
            lambda: ds_native.roll({args.partition_dim: 5}, roll_coords=False),
            lambda: fresh_mx().roll(dim=args.partition_dim, shift_by=5),
            None,
            args.repeats,
        )
    )

    # cumsum
    results.append(
        run_benchmark(
            f"cumsum(dim={args.partition_dim})",
            lambda: ds_native.cumsum(dim=args.partition_dim, skipna=True),
            lambda: fresh_mx().cumsum(dim=args.partition_dim, skipna=True),
            None,
            args.repeats,
        )
    )

    # rolling mean
    results.append(
        run_benchmark(
            f"rolling({args.partition_dim}, window=5).mean()",
            lambda: ds_native.rolling(**{args.partition_dim: 5}, center=True, min_periods=1).mean(),
            lambda: fresh_mx().rolling(dim=args.partition_dim, window=5, center=True, min_periods=1).mean(),
            None,
            args.repeats,
        )
    )

    # numpy ufunc interop (should be zero-communication, purely elementwise).
    # Built once outside the timed region -- unlike the reductions above,
    # whose own repartition cost is negligible next to the reduction
    # itself, a ufunc benchmark should time the ufunc alone, not a fresh
    # partition on every repeat.
    mx_t2m = fresh_mx().apply(lambda d: d["t2m"], fresh_mx())
    results.append(
        run_benchmark(
            "np.sqrt(np.abs(t2m))",
            lambda: np.sqrt(np.abs(ds_native["t2m"])),
            lambda: np.sqrt(np.abs(mx_t2m)),
            None,
            args.repeats,
        )
    )

    # isel on the partition dimension
    results.append(
        run_benchmark(
            f"isel({args.partition_dim}=slice(2,-2))",
            lambda: ds_native.isel({args.partition_dim: slice(2, -2)}),
            lambda: fresh_mx().isel(**{args.partition_dim: slice(2, -2)}),
            None,
            args.repeats,
        )
    )

    if rank == 0:
        header = f"{'operation':<40} {'correct':<8} {'dtype':<7} {'native (s)':<12} {'mpix (s)':<12} {'speedup':<8}"
        print(header)
        print("-" * len(header))
        for r in results:
            if r.error:
                print(f"{r.name:<40} ERROR: {r.error}")
                continue
            print(
                f"{r.name:<40} {str(r.correct):<8} {str(r.dtype_match):<7} "
                f"{r.native_time:<12.4f} {r.mpix_time:<12.4f} {r.speedup:<8.3f}"
            )
        n_incorrect = sum(1 for r in results if not r.error and not r.correct)
        n_dtype_mismatch = sum(1 for r in results if not r.error and not r.dtype_match)
        n_errors = sum(1 for r in results if r.error)
        print()
        if n_incorrect or n_dtype_mismatch or n_errors:
            print(
                f"WARNING: {n_incorrect} correctness mismatch(es), "
                f"{n_dtype_mismatch} dtype mismatch(es), {n_errors} error(s)."
            )
        else:
            print("All operations matched native xarray (values and dtype).")


if __name__ == "__main__":
    main()
