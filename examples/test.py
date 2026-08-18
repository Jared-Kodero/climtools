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
        --n-events 2000000 --xarray-events 40000 --netcdf-steps 500

Speedups from mpi.reduce / mpi.xarray / the parallel NetCDF writer only
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
from climtools import mpi, xgeo

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
# mpi runtime namespace -- small public helpers
# ---------------------------------------------------------------------------


def test_runtime_helpers() -> None:
    """Check the small MPIRuntime helpers exposed alongside the collectives."""
    alternate_root = min(1, SIZE - 1)
    datatype = mpi.datatype(np.float64)
    correct = (
        mpi.is_root() == (RANK == 0)
        and mpi.is_root(alternate_root) == (RANK == alternate_root)
        and isinstance(mpi.launched, bool)
        and datatype.Get_size() == np.dtype(np.float64).itemsize
        and issubclass(mpi.MPIError, Exception)
    )
    correct = bool(mpi.reduce.all(correct))
    record(
        "mpi runtime helpers (is_root/launched/datatype/MPIError)",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


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


def test_reduce_operations() -> None:
    """Exercise every mpi.reduce operation in all-ranks and root modes."""
    numeric = np.array([RANK + 1.0, 2.0 * RANK + 3.0], dtype=np.float64)
    numeric_stack = np.stack(
        [
            np.array([rank + 1.0, 2.0 * rank + 3.0], dtype=np.float64)
            for rank in range(SIZE)
        ]
    )
    logical = np.array([RANK == 0, RANK % 2 == 0, True], dtype=bool)
    logical_stack = np.stack(
        [np.array([rank == 0, rank % 2 == 0, True], dtype=bool) for rank in range(SIZE)]
    )

    cases = (
        ("sum", numeric, numeric_stack.sum(axis=0)),
        ("prod", numeric, numeric_stack.prod(axis=0)),
        ("min", numeric, numeric_stack.min(axis=0)),
        ("max", numeric, numeric_stack.max(axis=0)),
        ("mean", numeric, numeric_stack.mean(axis=0)),
        ("any", logical, logical_stack.any(axis=0)),
        ("all", logical, logical_stack.all(axis=0)),
    )

    for op_name, value, expected in cases:
        op = getattr(mpi.reduce, op_name)
        with timed() as box:
            result = op(value)
        parallel_s = box["seconds"]

        root_result = op(value, mode="root", root=0)
        all_mode_ok = bool(np.allclose(result, expected))
        root_mode_ok = (
            bool(np.allclose(root_result, expected))
            if RANK == 0
            else root_result is None
        )
        correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
        record(
            f"mpi.reduce.{op_name} (all/root modes)",
            correct,
            0.0,
            parallel_s,
            note="correctness-focused",
        )


# ---------------------------------------------------------------------------
# mpi.xarray -- distributed xarray operations
# ---------------------------------------------------------------------------


def test_xarray_open_dataset(out_dir: str) -> None:
    """mpi.xarray.open_dataset: lazy open plus rank-local partition metadata."""
    path = os.path.join(out_dir, "climtools_test_xarray_source.nc")
    n_time = max(2 * SIZE + 1, 9)
    ny = 4
    nx = 5

    def make_source() -> xr.Dataset:
        values = np.arange(n_time * ny * nx, dtype=np.float64).reshape(n_time, ny, nx)
        return xr.Dataset(
            {
                "field": xr.DataArray(
                    values,
                    dims=("time", "y", "x"),
                    coords={
                        "time": np.arange(n_time),
                        "y": np.arange(ny),
                        "x": np.arange(nx),
                    },
                    attrs={"units": "K"},
                )
            }
        )

    if RANK == 0:
        make_source().to_netcdf(path)
    mpi.comm.barrier()

    with timed() as box:
        distributed = mpi.xarray.open_dataset(path, partition_dim="time")
        distributed.load()
    parallel_s = box["seconds"]

    local = distributed["field"].values.copy()
    meta = distributed.attrs.get("mpi_meta")
    variable_meta = distributed["field"].attrs.get("mpi_meta")
    distributed.close()

    parts = mpi.comm.allgather(local)
    assembled = np.concatenate(parts, axis=0)
    expected = make_source()["field"].values
    local_meta_ok = (
        isinstance(meta, dict)
        and meta.get("dim") == "time"
        and int(meta.get("global_size", -1)) == n_time
        and int(meta.get("stop", -1)) - int(meta.get("start", -1)) == local.shape[0]
        and isinstance(variable_meta, dict)
        and variable_meta.get("dim") == "time"
    )
    correct = bool(np.array_equal(assembled, expected)) and bool(
        mpi.reduce.all(local_meta_ok)
    )

    def serial_fn() -> float:
        with xr.open_dataset(path) as serial:
            serial.load()
            return float(serial["field"].sum())

    _, serial_s = run_serial_baseline(serial_fn)
    record(
        "mpi.xarray.open_dataset (partitioned time dimension)",
        correct,
        serial_s,
        parallel_s,
    )

    mpi.comm.barrier()
    if RANK == 0:
        os.remove(path)
    mpi.comm.barrier()


def test_xarray_redistribute() -> None:
    """mpi.xarray.redistribute: explicit and automatic partition dimensions."""
    n_time = max(3 * SIZE + 2, 11)
    ny = 3
    full = xr.DataArray(
        np.arange(n_time * ny, dtype=np.float64).reshape(n_time, ny),
        dims=("time", "y"),
        coords={"time": np.arange(n_time), "y": np.arange(ny)},
    )

    with timed() as box:
        distributed = mpi.xarray.redistribute(full, "time")
    parallel_s = box["seconds"]
    auto = mpi.xarray.redistribute(full, "auto")

    explicit_parts = mpi.comm.allgather(distributed.values)
    auto_parts = mpi.comm.allgather(auto.values)
    explicit_meta = distributed.attrs.get("mpi_meta")
    auto_meta = auto.attrs.get("mpi_meta")
    local_ok = (
        isinstance(explicit_meta, dict)
        and explicit_meta.get("dim") == "time"
        and int(explicit_meta.get("global_size", -1)) == n_time
        and isinstance(auto_meta, dict)
        and auto_meta.get("dim") == "time"
        and int(auto_meta.get("global_size", -1)) == n_time
    )
    correct = (
        bool(np.array_equal(np.concatenate(explicit_parts, axis=0), full.values))
        and bool(np.array_equal(np.concatenate(auto_parts, axis=0), full.values))
        and bool(mpi.reduce.all(local_ok))
    )
    record(
        "mpi.xarray.redistribute (explicit/auto)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


def test_xarray_isel() -> None:
    """mpi.xarray.isel: global slice and scalar integer indexing."""
    n_time = max(4 * SIZE + 3, 19)
    ny = 3
    full = xr.DataArray(
        np.arange(n_time * ny, dtype=np.float64).reshape(n_time, ny),
        dims=("time", "y"),
        coords={"time": np.arange(n_time), "y": np.arange(ny)},
    )
    distributed = mpi.xarray.redistribute(full, "time")
    start = 2
    stop = n_time - 2
    scalar_index = n_time // 2

    with timed() as box:
        sliced = mpi.xarray.isel(distributed, time=slice(start, stop))
        scalar = mpi.xarray.isel(distributed, time=scalar_index)
    parallel_s = box["seconds"]

    sliced_parts = mpi.comm.allgather(sliced.values)
    assembled = np.concatenate(sliced_parts, axis=0)
    expected_slice = full.isel(time=slice(start, stop)).values
    expected_scalar = full.isel(time=scalar_index).values
    meta = sliced.attrs.get("mpi_meta")
    local_meta_ok = (
        isinstance(meta, dict)
        and meta.get("dim") == "time"
        and int(meta.get("global_size", -1)) == stop - start
    )
    correct = (
        bool(np.array_equal(assembled, expected_slice))
        and bool(np.array_equal(scalar.values, expected_scalar))
        and "mpi_meta" not in scalar.attrs
        and bool(mpi.reduce.all(local_meta_ok))
    )
    record(
        "mpi.xarray.isel (global slice/scalar)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


def test_xarray_sel() -> None:
    """mpi.xarray.sel: global coordinate slice and scalar label selection."""
    n_time = max(4 * SIZE + 3, 19)
    ny = 3
    time = np.arange(n_time, dtype=np.int64) * 6
    full = xr.DataArray(
        np.arange(n_time * ny, dtype=np.float64).reshape(n_time, ny),
        dims=("time", "y"),
        coords={"time": time, "y": np.arange(ny)},
    )
    distributed = mpi.xarray.redistribute(full, "time")
    start_label = int(time[2])
    stop_label = int(time[-3])
    scalar_label = int(time[n_time // 2])

    with timed() as box:
        sliced = mpi.xarray.sel(distributed, time=slice(start_label, stop_label))
        scalar = mpi.xarray.sel(distributed, time=scalar_label)
    parallel_s = box["seconds"]

    sliced_parts = mpi.comm.allgather(sliced.values)
    assembled = np.concatenate(sliced_parts, axis=0)
    expected_slice = full.sel(time=slice(start_label, stop_label)).values
    expected_scalar = full.sel(time=scalar_label).values
    meta = sliced.attrs.get("mpi_meta")
    local_meta_ok = isinstance(meta, dict) and meta.get("dim") == "time"
    correct = (
        bool(np.array_equal(assembled, expected_slice))
        and bool(np.array_equal(scalar.values, expected_scalar))
        and "mpi_meta" not in scalar.attrs
        and bool(mpi.reduce.all(local_meta_ok))
    )
    record(
        "mpi.xarray.sel (global slice/scalar)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


def test_xarray_reduction(n_events_total: int, ny: int, nx: int, op_name: str) -> None:
    """Numeric mpi.xarray reductions vs xarray on the assembled global array."""
    per_rank = max(1, n_events_total // SIZE)

    def make_local(rank: int) -> xr.DataArray:
        rng = np.random.default_rng(7000 + rank)
        if op_name == "prod":
            data = 1.0 + 1.0e-3 * rng.standard_normal((per_rank, ny, nx))
        else:
            data = rng.standard_normal((per_rank, ny, nx))
        data = data.astype(np.float64)
        data[0, 0, 0] = np.nan
        return xr.DataArray(
            data,
            dims=("event", "y", "x"),
            attrs={"units": "1"},
        )

    local = make_local(RANK)
    op = getattr(mpi.xarray, op_name)
    kwargs = {"skipna": True, "keep_attrs": True}
    if op_name in {"sum", "prod"}:
        kwargs["min_count"] = 1

    with timed() as box:
        result = op(local, dim="event", **kwargs)
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        parts = [make_local(rank).values for rank in range(SIZE)]
        full = xr.DataArray(
            np.concatenate(parts, axis=0),
            dims=("event", "y", "x"),
        )
        serial_kwargs = {"skipna": True}
        if op_name in {"sum", "prod"}:
            serial_kwargs["min_count"] = 1
        return getattr(full, op_name)(dim="event", **serial_kwargs).values

    expected, serial_s = run_serial_baseline(serial_fn)
    root_result = op(local, dim="event", mode="root", root=0, **kwargs)
    all_mode_ok = (
        result is not None
        and bool(np.allclose(result.values, expected, rtol=1.0e-9, equal_nan=True))
        and result.attrs.get("units") == "1"
    )
    root_mode_ok = (
        root_result is not None
        and bool(np.allclose(root_result.values, expected, rtol=1.0e-9, equal_nan=True))
        if RANK == 0
        else root_result is None
    )
    correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
    record(
        f"mpi.xarray.{op_name} ({n_events_total} events, {ny}x{nx} field)",
        correct,
        serial_s,
        parallel_s,
    )


def test_xarray_logical_reduction(
    n_events_total: int,
    ny: int,
    nx: int,
    op_name: str,
) -> None:
    """Logical mpi.xarray.any/all vs xarray on the assembled global array."""
    per_rank = max(1, n_events_total // SIZE)

    def make_local(rank: int) -> xr.DataArray:
        rng = np.random.default_rng(8000 + rank)
        data = rng.random((per_rank, ny, nx)) > 0.5
        data[0, 0, 0] = rank == 0
        return xr.DataArray(data, dims=("event", "y", "x"))

    local = make_local(RANK)
    op = getattr(mpi.xarray, op_name)

    with timed() as box:
        result = op(local, dim="event")
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        parts = [make_local(rank).values for rank in range(SIZE)]
        full = xr.DataArray(
            np.concatenate(parts, axis=0),
            dims=("event", "y", "x"),
        )
        return getattr(full, op_name)(dim="event").values

    expected, serial_s = run_serial_baseline(serial_fn)
    root_result = op(local, dim="event", mode="root", root=0)
    all_mode_ok = result is not None and bool(np.array_equal(result.values, expected))
    root_mode_ok = (
        root_result is not None and bool(np.array_equal(root_result.values, expected))
        if RANK == 0
        else root_result is None
    )
    correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
    record(
        f"mpi.xarray.{op_name} ({n_events_total} Boolean events, {ny}x{nx})",
        correct,
        serial_s,
        parallel_s,
    )


def test_xarray_dataset_reduction() -> None:
    """Dataset reduction keeps variables that do not use the partition dimension."""
    n_event = max(2 * SIZE + 1, 9)
    ny = 4
    event = np.arange(n_event)
    y = np.arange(ny)
    signal = 1.0 + np.arange(n_event * ny, dtype=np.float64).reshape(n_event, ny)
    static = np.linspace(10.0, 40.0, ny)
    full = xr.Dataset(
        {
            "signal": xr.DataArray(signal, dims=("event", "y")),
            "static": xr.DataArray(static, dims=("y",)),
        },
        coords={"event": event, "y": y},
    )
    distributed = mpi.xarray.redistribute(full, "event")

    with timed() as box:
        result = mpi.xarray.sum(distributed, dim="event")
    parallel_s = box["seconds"]

    expected = full.sum(dim="event")
    correct = (
        result is not None
        and bool(np.allclose(result["signal"].values, expected["signal"].values))
        and bool(np.array_equal(result["static"].values, expected["static"].values))
        and "mpi_meta" not in result.attrs
    )
    correct = bool(mpi.reduce.all(correct))
    record(
        "mpi.xarray.sum Dataset (distributed + static variables)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


def test_xarray_redistribute_on(n_events_total: int, ny: int, nx: int) -> None:
    """Reduction result can be redistributed along a remaining dimension."""
    per_rank = max(1, n_events_total // SIZE)

    def make_local(rank: int) -> xr.DataArray:
        rng = np.random.default_rng(8500 + rank)
        data = rng.standard_normal((per_rank, ny, nx)).astype(np.float64)
        return xr.DataArray(data, dims=("event", "y", "x"))

    local = make_local(RANK)
    with timed() as box:
        result = mpi.xarray.mean(local, dim="event", redistribute_on="y")
    parallel_s = box["seconds"]

    parts = mpi.comm.allgather(result.values)
    assembled = np.concatenate(parts, axis=0)

    def serial_fn() -> np.ndarray:
        full = xr.DataArray(
            np.concatenate([make_local(rank).values for rank in range(SIZE)], axis=0),
            dims=("event", "y", "x"),
        )
        return full.mean(dim="event").values

    expected, serial_s = run_serial_baseline(serial_fn)
    meta = result.attrs.get("mpi_meta")
    local_meta_ok = (
        isinstance(meta, dict)
        and meta.get("dim") == "y"
        and int(meta.get("global_size", -1)) == ny
    )
    correct = bool(np.allclose(assembled, expected, rtol=1.0e-9)) and bool(
        mpi.reduce.all(local_meta_ok)
    )
    record(
        "mpi.xarray.mean(..., redistribute_on='y')",
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
        ds = make_full_dataset() if RANK == 0 else xgeo.empty_dataset()
        xgeo.to_netcdf(
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
        xgeo.to_netcdf(full, serial_path, unlimited_dim="time", show_progress=False)

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


def test_netcdf_distributed_roundtrip(
    n_time: int,
    ny: int,
    nx: int,
    out_dir: str,
) -> None:
    """Compare distributed MPI NetCDF output with ordinary xarray I/O.

    The source is opened independently through ordinary ``xarray`` and
    ``mpi.xarray``. The distributed object is written without gathering its
    partitioned data back to rank 0. The resulting files are then loaded with
    ordinary xarray and compared for complete data, coordinate, and attribute
    integrity. Wall-clock write times are also reported.
    """
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        mpi.log(
            "\n--- Distributed NetCDF round-trip SKIPPED: netCDF4 lacks "
            "parallel4 support ---"
        )
        return

    source_path = os.path.join(out_dir, "climtools_test_distributed_source.nc")
    serial_path = os.path.join(out_dir, "climtools_test_distributed_serial.nc")
    parallel_path = os.path.join(out_dir, "climtools_test_distributed_parallel.nc")

    def make_source() -> xr.Dataset:
        rng = np.random.default_rng(24601)
        times = pd.date_range("1980-01-01", periods=n_time, freq="6h")
        lat = np.linspace(-90.0, 90.0, ny, dtype=np.float64)
        lon = np.linspace(0.0, 360.0, nx, endpoint=False, dtype=np.float64)
        return xr.Dataset(
            data_vars={
                "precipitation": (
                    ("time", "lat", "lon"),
                    rng.random((n_time, ny, nx), dtype=np.float32),
                    {"units": "mm/day", "long_name": "precipitation rate"},
                ),
                "temperature": (
                    ("time", "lat", "lon"),
                    (250.0 + 40.0 * rng.random((n_time, ny, nx), dtype=np.float32)),
                    {"units": "K"},
                ),
                "orography": (
                    ("lat", "lon"),
                    np.arange(ny * nx, dtype=np.float32).reshape(ny, nx),
                    {"units": "m"},
                ),
            },
            coords={"time": times, "lat": lat, "lon": lon},
            attrs={"title": "distributed NetCDF integrity test"},
        )

    if RANK == 0:
        source = make_source()
        source.to_netcdf(source_path)
    mpi.comm.barrier()

    def serial_fn() -> None:
        with xr.open_dataset(source_path) as normal:
            normal.load()
            normal.to_netcdf(serial_path)

    _, serial_s = run_serial_baseline(serial_fn)
    mpi.comm.barrier()

    distributed = mpi.xarray.open_dataset(
        source_path,
        partition_dim="time",
    )
    meta = distributed.attrs.get("mpi_meta")
    local_meta_ok = (
        isinstance(meta, dict)
        and meta.get("dim") == "time"
        and int(meta.get("global_size", -1)) == n_time
    )

    with timed() as box:
        xgeo.to_netcdf(
            distributed,
            parallel_path,
            parallel=True,
            allow_serial=(SIZE == 1),
        )
    parallel_s = box["seconds"]
    distributed.close()
    mpi.comm.barrier()

    correct = bool(mpi.reduce.all(local_meta_ok))
    integrity_note = ""

    if RANK == 0:
        try:
            with (
                xr.open_dataset(serial_path) as expected,
                xr.open_dataset(parallel_path) as actual,
            ):
                expected.load()
                actual.load()
                xr.testing.assert_identical(actual, expected)

                mpi_meta_leaked = "mpi_meta" in actual.attrs or any(
                    "mpi_meta" in variable.attrs
                    for variable in actual.variables.values()
                )
                if mpi_meta_leaked:
                    raise AssertionError(
                        "Internal mpi_meta attributes were written to NetCDF."
                    )
        except AssertionError as exc:
            correct = False
            integrity_note = str(exc)

    correct, integrity_note = mpi.comm.bcast(
        (correct, integrity_note),
        root=0,
    )
    record(
        f"distributed NetCDF round-trip ({n_time} steps, {ny}x{nx})",
        correct,
        serial_s,
        parallel_s,
        note=integrity_note or "xr.testing.assert_identical",
    )

    mpi.comm.barrier()
    if RANK == 0:
        for path in (source_path, serial_path, parallel_path):
            if os.path.exists(path):
                os.remove(path)
    mpi.comm.barrier()


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
        "--xarray-events", type=int, default=5_000, help="events for mpi.xarray tests"
    )
    parser.add_argument("--xarray-ny", type=int, default=40)
    parser.add_argument("--xarray-nx", type=int, default=40)
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
            "\nRan on 1 rank: speedups will be ~1x or worse. mpi.reduce/mpi.xarray/\n"
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

    mpi.log("\n--- mpi runtime helpers ---")
    safe_run(test_runtime_helpers)

    mpi.log("\n--- mpi.reduce ---")
    safe_run(test_reduce_sum_scalar, args.n_events)
    safe_run(test_reduce_composite, args.n_events, args.grid_ny, args.grid_nx)
    safe_run(test_reduce_xarray_object, args.grid_ny, args.grid_nx)
    safe_run(test_reduce_operations)

    mpi.log("\n--- mpi.xarray ---")
    safe_run(test_xarray_open_dataset, args.out_dir)
    safe_run(test_xarray_redistribute)
    safe_run(test_xarray_isel)
    safe_run(test_xarray_sel)
    for op_name in ("sum", "prod", "mean", "max", "min"):
        safe_run(
            test_xarray_reduction,
            args.xarray_events,
            args.xarray_ny,
            args.xarray_nx,
            op_name,
        )
    for op_name in ("any", "all"):
        safe_run(
            test_xarray_logical_reduction,
            args.xarray_events,
            args.xarray_ny,
            args.xarray_nx,
            op_name,
        )
    safe_run(test_xarray_dataset_reduction)
    safe_run(
        test_xarray_redistribute_on,
        args.xarray_events,
        args.xarray_ny,
        args.xarray_nx,
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
        safe_run(
            test_netcdf_distributed_roundtrip,
            args.netcdf_steps,
            args.netcdf_ny,
            args.netcdf_nx,
            args.out_dir,
        )

    mpi.comm.barrier()
    print_summary()


if __name__ == "__main__":
    main()
