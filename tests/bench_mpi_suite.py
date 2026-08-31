"""Benchmark suite: native xarray vs MPI-Xarray, several operations.

    mpirun -np <N> --oversubscribe python3 tests/bench_mpi_suite.py

Uses ``mock_dataset.create_bench_dataset`` (rank 0 writes one NetCDF
file, every rank barriers, then opens it -- via ``mpi_open_dataset`` for
the distributed object and plain ``xarray.open_dataset`` for the native
reference) rather than an in-memory array broadcast from rank 0.

Follows the task's protocol: warm-up runs, several timed repetitions,
a barrier before each timed region, slowest-rank wall time via
``MPI.MAX``, and a native-vs-MPI comparison per operation. Prints a
markdown table row per operation plus a PASS/FAIL accuracy/dtype check
against the native result.

IMPORTANT: see STATUS.md. This sandbox has exactly one physical CPU
core, so every multi-rank run here oversubscribes that one core --
ranks time-share rather than run in parallel. The wall-clock numbers
below are only a correctness/harness smoke test, not a performance
finding; rerun on real multi-core/multi-node hardware for a number
that means anything.
"""

from __future__ import annotations

import time

import numpy as np
import xarray as xr
from mock_dataset import OUTPUT_DIR, create_bench_dataset
from mpi4py import MPI

from climtools import mpi
from climtools.xarray.core import unwrap
from climtools.xarray.io import mpi_open_dataset

comm = mpi.comm
rank = comm.rank
size = comm.size

N_TIME = 400_000
N_LAT = 20
REPS = 5
WARMUP = 2

path = OUTPUT_DIR / "bench.nc"
create_bench_dataset(path, n_time=N_TIME, n_lat=N_LAT, seed=0)

native_da = xr.open_dataset(path)["v"]
mda = mpi_open_dataset(str(path), mpi, partition_dim="time", log_partitions=False)["v"]

rows: list[str] = []


def time_native(fn):
    times = []
    if rank == 0:
        for i in range(WARMUP + REPS):
            t0 = time.perf_counter()
            result = fn()
            t1 = time.perf_counter()
            if i >= WARMUP:
                times.append(t1 - t0)
    else:
        result = None
    times = comm.bcast(times, root=0)
    result = comm.bcast(result, root=0)
    return float(np.median(times)), result


def time_mpi(fn):
    times = []
    result = None
    for i in range(WARMUP + REPS):
        comm.Barrier()
        t0 = time.perf_counter()
        result = fn()
        comm.Barrier()
        t1 = time.perf_counter()
        slowest = comm.allreduce(t1 - t0, op=MPI.MAX)
        if i >= WARMUP:
            times.append(slowest)
    return float(np.median(times)), result


def gather(mpi_obj):
    local = unwrap(mpi_obj)
    meta = mpi_obj.meta
    if meta is None:
        return local
    dim = meta["dims"] if isinstance(meta["dims"], str) else meta["dims"][0]
    return xr.concat(comm.allgather(local), dim=dim).sortby(dim)


def bench(label: str, native_fn, mpi_fn, native_result_fn=None):
    native_t, native_result = time_native(native_fn)
    mpi_t, mpi_result = time_mpi(mpi_fn)
    gathered = gather(mpi_result)
    ref = native_result_fn() if native_result_fn else native_result
    # float32 reductions over 4e5 elements: MPI combines per-rank partial
    # sums in a different order than NumPy's own (pairwise) summation, so
    # the last few bits legitimately differ -- a relative tolerance is the
    # right comparison here, not bitwise/absolute equality (see the task's
    # own accuracy-tests guidance on reduction order).
    accurate = bool(
        np.allclose(gathered.values, ref.values, equal_nan=True, rtol=1e-3, atol=1e-3)
    )
    dtype_ok = gathered.dtype == ref.dtype
    speedup = native_t / mpi_t if mpi_t > 0 else float("nan")
    if rank == 0:
        rows.append(
            f"| {label} | {size} | {native_t * 1000:.2f} ms | {mpi_t * 1000:.2f} ms "
            f"| {speedup:.2f}x | {'PASS' if accurate else 'FAIL'} "
            f"| {'PASS' if dtype_ok else 'FAIL'} |"
        )


bench("sum", lambda: native_da.sum("time"), lambda: mda.sum("time"))
bench("mean", lambda: native_da.mean("time"), lambda: mda.mean("time"))
bench("std", lambda: native_da.std("time"), lambda: mda.std("time"))
bench("min", lambda: native_da.min("time"), lambda: mda.min("time"))
bench("max", lambda: native_da.max("time"), lambda: mda.max("time"))
bench(
    "isel",
    lambda: native_da.isel(lat=slice(0, 5)),
    lambda: mda.isel(lat=slice(0, 5)),
)
new_labels = np.arange(-1000, N_TIME + 1000)
bench(
    "reindex",
    lambda: native_da.reindex(time=new_labels),
    lambda: mda.reindex(time=new_labels),
    native_result_fn=lambda: native_da.reindex(time=new_labels),
)

if rank == 0:
    header = (
        "| Method | Ranks | Native Xarray | MPI Xarray | Speedup | Accuracy | Dtype |\n"
        "| ------ | ----: | ------------: | ---------: | ------: | -------- | ----- |"
    )
    print(header)
    for row in rows:
        print(row)
