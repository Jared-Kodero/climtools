#!/usr/bin/env python3
"""climtools_test.py -- correctness + speed suite for climtools.mpi.

For every performance-oriented parallel feature this script:
  1. runs the distributed/parallel version across whatever ranks the
     script was launched with (mpirun -n N ...),
  2. runs an equivalent pure-serial baseline on rank 0 alone, over the
     same total amount of data/work,
  3. checks the two results agree (within floating-point tolerance), and
  4. times both and reports the ratio.

Correctness-only contract tests additionally exercise validation, edge cases,
metadata, nonzero roots, empty partitions, and NetCDF configuration. Flushed
timestamped progress messages are emitted before each test and major phase so
a stalled batch job identifies the active operation before the final summary.

Run:

    # single process, serial fallback -- sanity check only, no real
    # parallelism, speedups will be ~1x or worse
    python climtools_test.py

    # real multi-rank run -- this is what actually demonstrates speedups
    mpirun -n 8 python climtools_test.py

    # scale the workload up for a bigger machine
    mpirun -n 16 python climtools_test.py \
        --n-events 2000000 --xarray-events 40000

Speedups from mpi.reduce / mpi.xarray / the parallel NetCDF writer only
show up when ranks genuinely run on separate cores. On a single-core
machine, or an oversubscribed launch with more ranks than cores, results
will be flat or even slower, since ranks are then time-sliced rather than
run concurrently -- that is expected, not a bug.

Requires climtools importable (e.g. run from the parent of the cloned
repo, or with climtools installed). Parallel NetCDF-4 output additionally
requires netCDF4 built against a parallel-enabled MPI/HDF5/NetCDF-C stack
(see climtools/env/setup_env.sh); if that support is missing, the NetCDF
write tests are skipped automatically when running with more than one rank.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from climtools import mpi, xgeo

RANK: int = mpi.comm.rank
SIZE: int = mpi.comm.size
DEFAULT_NETCDF_SOURCE = Path(
    "/oscar/data/deeps/private/jl322/jkodero/data/models/gfdl_shield/archive/"
    + "2024081400Z/C96.NESTED.R4x2.R2x1.CNTRL/mem01/case/fv3_hist.nest04.nc"
)


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

    error: BaseException | None = None
    try:
        yield box
    except BaseException as exc:
        error = exc

    # Synchronize Python exceptions before the closing barrier. Without this,
    # one failing rank skips the barrier while successful ranks wait in it,
    # hiding the original exception behind an MPI deadlock.
    mpi.raise_if_error(error, "timed block")
    mpi.comm.barrier()
    local_elapsed = mpi.MPI.Wtime() - start
    box["seconds"] = mpi.reduce.max(local_elapsed)


def run_serial_baseline(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run `fn` on rank 0 only and get (result, elapsed) back on every rank.

    This is `@mpi(broadcast=True)` -- "execute on root and broadcast its
    return value to every rank" -- applied directly, rather than
    hand-rolling the same root-only-then-broadcast pattern with raw
    `mpi.comm` calls. Every non-root rank simply waits inside the
    decorator's own synchronization for root's timed result, which is
    exactly the single-process cost a script with no MPI at all would pay.
    """

    @mpi(broadcast=True)
    def _timed_on_root() -> tuple[Any, float]:
        start = mpi.MPI.Wtime()
        result = fn()
        elapsed = mpi.MPI.Wtime() - start
        return result, elapsed

    return _timed_on_root()


def dtype_rtol(*values: Any, factor: float = 64.0) -> float:
    """Return a relative tolerance derived from the data's own precision.

    Distributed and serial reductions sum the same values in different
    associative orders, so they agree only to the resolution of the dtype
    being reduced. For float32 SHiELD fields this is 64 * 1.19e-7 ~ 7.6e-6;
    for float64 it is 64 * 2.22e-16 ~ 1.4e-14. Integer and Boolean results
    must match exactly and receive a tolerance of zero.

    Parameters
    ----------
    *values : Any
        Values whose common dtype sets the tolerance.
    factor : float, optional
        Multiple of the dtype epsilon allowed. Default is 64.

    Returns
    -------
    float
        Relative tolerance for numpy.isclose/numpy.allclose.
    """
    dtypes = [np.asarray(value).dtype for value in values]
    common = np.result_type(*dtypes) if dtypes else np.dtype(np.float64)
    if common.kind not in "fc":
        return 0.0
    return float(factor) * float(np.finfo(common).eps)


@dataclass
class Result:
    name: str
    correct: bool
    serial_s: float
    parallel_s: float
    note: str = ""
    skipped: bool = False

    @property
    def speedup(self) -> float:
        if self.skipped or self.serial_s <= 0.0 or self.parallel_s <= 0.0:
            return float("nan")
        return self.serial_s / self.parallel_s


RESULTS: list[Result] = []
CURRENT_TEST_NAME = ""
CURRENT_TEST_NUMBER = 0


def progress(message: str) -> None:
    """Emit an immediately flushed rank-0 progress message."""
    label = (
        f"test {CURRENT_TEST_NUMBER:02d} {CURRENT_TEST_NAME}: "
        if CURRENT_TEST_NAME
        else ""
    )
    mpi.log(
        f"[PROGRESS] {label}{message}",
        timestamp=True,
        flush=True,
    )


def _raises(
    expected: type[BaseException] | tuple[type[BaseException], ...],
    function: Any,
    *args: Any,
    contains: str | None = None,
    **kwargs: Any,
) -> bool:
    """Return whether a call raises the expected exception and message."""
    try:
        function(*args, **kwargs)
    except expected as exc:
        return contains is None or contains in str(exc)
    except BaseException:
        return False
    return False


def record(
    name: str,
    correct: bool,
    serial_s: float,
    parallel_s: float,
    note: str = "",
    *,
    skipped: bool = False,
) -> None:
    result = Result(name, correct, serial_s, parallel_s, note, skipped)
    RESULTS.append(result)
    status = "SKIP" if skipped else ("OK  " if correct else "FAIL")
    speedup_str = "  n/a " if np.isnan(result.speedup) else f"{result.speedup:5.2f}x"
    mpi.log(
        f"[{status}] {name:<46} serial={serial_s:8.4f}s  "
        + f"parallel={parallel_s:8.4f}s  speedup={speedup_str}"
        + (f"  ({note})" if note else ""),
        flush=True,
    )


def record_skip(name: str, note: str) -> None:
    """Record a test that could not run because a required capability is absent."""
    record(name, True, 0.0, 0.0, note=note, skipped=True)


def safe_run(fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """Run a test function on every rank, synchronizing and recording failures."""
    global CURRENT_TEST_NAME, CURRENT_TEST_NUMBER

    CURRENT_TEST_NUMBER += 1
    CURRENT_TEST_NAME = fn.__name__
    before = len(RESULTS)

    # Bound the entry barrier. A rank that never arrives (a straggling
    # collective, or a blocked NetCDF read on a locking filesystem) would
    # otherwise leave every other rank blocked with no indication of where.
    mpi.sync(f"entry barrier for {fn.__name__}")
    progress("START")

    error: BaseException | None = None
    try:
        fn(*args, **kwargs)
    except BaseException as exc:
        error = exc

    # raise_if_error is itself collective, so it only synchronizes failures
    # that every rank reaches. A rank that raised part-way through a
    # multi-collective operation arrives here while the others are still
    # blocked inside that operation; bound the wait so the mismatch aborts
    # with a diagnostic rather than deadlocking.
    mpi.sync(f"exit barrier for {fn.__name__}")

    synchronized_error: BaseException | None = None
    try:
        mpi.raise_if_error(error, fn.__name__)
    except BaseException as exc:
        synchronized_error = exc

    if synchronized_error is not None:
        record(
            f"{fn.__name__} (uncaught exception)",
            False,
            0.0,
            0.0,
            note=f"{type(synchronized_error).__name__}: {synchronized_error}",
        )
        progress("FAILED")
    elif len(RESULTS) == before:
        record(
            f"{fn.__name__} (result accounting)",
            False,
            0.0,
            0.0,
            note="test completed without recording a result or skip",
        )
        progress("FAILED: no result recorded")
    else:
        new_results = RESULTS[before:]
        if any(not result.correct and not result.skipped for result in new_results):
            progress("DONE with failed check(s)")
        elif all(result.skipped for result in new_results):
            progress("SKIPPED")
        else:
            progress("DONE")

    CURRENT_TEST_NAME = ""


def _require_source() -> None:
    """Require the configured SHiELD NetCDF file on every MPI rank."""
    visible = DEFAULT_NETCDF_SOURCE.is_file()
    if not bool(mpi.reduce.all(visible)):
        raise FileNotFoundError(
            "SHiELD NetCDF source is not visible on every rank: "
            + f"{DEFAULT_NETCDF_SOURCE}"
        )


def _rank_bounds(size: int, rank: int = RANK) -> tuple[int, int]:
    """Return this rank's contiguous bounds within a global dimension."""
    return size * rank // SIZE, size * (rank + 1) // SIZE


def _load_source_variable(
    variable: str,
    **indexers: int | slice,
) -> xr.DataArray:
    """Load only the requested selection of one variable from the SHiELD file.

    Every rank opens the same file concurrently. HDF5 takes POSIX advisory
    locks by default, which can block indefinitely on a parallel filesystem;
    ``examples/test.sh`` exports ``HDF5_USE_FILE_LOCKING=FALSE`` for this
    reason.
    """
    with xr.open_dataset(DEFAULT_NETCDF_SOURCE) as source:
        data = source[variable]
        valid_indexers = {
            dim: indexer for dim, indexer in indexers.items() if dim in data.dims
        }
        return data.isel(valid_indexers).load()


def _load_source_dataset(
    variables: tuple[str, ...],
    **indexers: int | slice,
) -> xr.Dataset:
    """Load only requested variables and dimension slices from the SHiELD file."""
    with xr.open_dataset(DEFAULT_NETCDF_SOURCE) as source:
        data = source[list(variables)]
        valid_indexers = {
            dim: indexer for dim, indexer in indexers.items() if dim in data.dims
        }
        return data.isel(valid_indexers).load()


# ---------------------------------------------------------------------------
# mpi runtime namespace -- small public helpers
# ---------------------------------------------------------------------------


def test_runtime_helpers() -> None:
    """Check the small MPIRuntime helpers exposed alongside the collectives."""
    progress("checking rank helpers and supported MPI datatype mappings")
    alternate_root = min(1, SIZE - 1)
    supported_dtypes = (
        np.bool_,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    )
    datatype_ok = all(
        mpi.datatype(dtype).Get_size() == np.dtype(dtype).itemsize
        for dtype in supported_dtypes
    )
    correct = (
        mpi.is_root() == (RANK == 0)
        and mpi.is_root(alternate_root) == (RANK == alternate_root)
        and isinstance(mpi.launched, bool)
        and datatype_ok
        and issubclass(mpi.MPIError, Exception)
    )
    correct = bool(mpi.reduce.all(correct))
    progress("recording result")
    record(
        "mpi runtime helpers (is_root/launched/datatype/MPIError)",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


def test_runtime_logging_and_errors() -> None:
    """Check mpi.log behavior and synchronized error propagation."""
    progress("checking mpi.log formatting, rank filtering, and timestamp output")
    captured: list[str] = []

    def capture(message: str, *args: Any, **kwargs: Any) -> None:
        del kwargs
        captured.append(message % args if args else message)

    mpi.log(
        "formatted %s",
        "message",
        root=-1,
        prefix=True,
        logger=capture,
    )
    rank_width = len(str(SIZE))
    logger_ok = captured == [f"[MPI RANK {RANK:{rank_width}d}] formatted message"]

    root_only: list[str] = []

    def capture_root(message: str, *args: Any, **kwargs: Any) -> None:
        del kwargs
        root_only.append(message % args if args else message)

    mpi.log(
        "root only",
        root=0,
        prefix=False,
        logger=capture_root,
    )
    root_filter_ok = root_only == ["root only"] if RANK == 0 else not root_only

    stream = StringIO()
    mpi.log(
        "timestamp probe",
        root=-1,
        timestamp=True,
        prefix=False,
        file=stream,
        flush=True,
    )
    rendered = stream.getvalue().strip()
    timestamp_text, separator, payload = rendered.partition(" - ")
    timestamp_ok = (
        separator == " - "
        and payload == "timestamp probe"
        and len(timestamp_text) == 19
        and timestamp_text[4] == "-"
        and timestamp_text[7] == "-"
        and timestamp_text[10] == " "
        and timestamp_text[13] == ":"
        and timestamp_text[16] == ":"
    )

    progress("checking mpi.raise_if_error no-error/all-rank/subset-rank paths")
    no_error_ok = True
    try:
        mpi.raise_if_error(None, "no-error phase")
    except BaseException:
        no_error_ok = False

    all_rank_error_ok = _raises(
        ValueError,
        mpi.raise_if_error,
        ValueError("all-rank failure"),
        "all-rank phase",
        contains="all-rank failure",
    )

    subset_error_ok = True
    if SIZE > 1:
        try:
            mpi.raise_if_error(
                ValueError("rank-zero failure") if RANK == 0 else None,
                "subset phase",
            )
        except mpi.MPIError as exc:
            subset_error_ok = "Rank 0 failed during subset phase" in str(
                exc
            ) and "rank-zero failure" in str(exc)
        except BaseException:
            subset_error_ok = False
        else:
            subset_error_ok = False

    correct = bool(
        mpi.reduce.all(
            logger_ok
            and root_filter_ok
            and timestamp_ok
            and no_error_ok
            and all_rank_error_ok
            and subset_error_ok
        )
    )
    progress("recording result")
    record(
        "mpi.log/raise_if_error contracts",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


def test_runtime_contracts() -> None:
    """Exercise reduction validation, non-contiguous buffers, and nonzero roots."""
    progress("checking scalar and non-contiguous reductions")
    scalar = mpi.reduce.sum(RANK + 1)
    expected_scalar = SIZE * (SIZE + 1) // 2

    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    noncontiguous = base[:, ::2]
    noncontiguous_result = mpi.reduce.sum(noncontiguous)
    noncontiguous_ok = bool(np.array_equal(noncontiguous_result, noncontiguous * SIZE))

    dataset = xr.Dataset(
        {
            "a": ("x", np.asarray([RANK + 1.0, 2.0 * (RANK + 1.0)])),
            "b": ("x", np.asarray([1.0, -1.0]) * (RANK + 1.0)),
        },
        attrs={"source": "synthetic"},
    )
    dataset_result = mpi.reduce.sum(dataset)
    rank_sum = SIZE * (SIZE + 1) / 2.0
    dataset_ok = (
        isinstance(dataset_result, xr.Dataset)
        and dataset_result.attrs == dataset.attrs
        and bool(
            np.array_equal(
                dataset_result["a"].values,
                np.asarray([rank_sum, 2.0 * rank_sum]),
            )
        )
        and bool(
            np.array_equal(
                dataset_result["b"].values,
                np.asarray([rank_sum, -rank_sum]),
            )
        )
    )
    any_scalar = mpi.reduce.any(RANK == 0)
    all_scalar = mpi.reduce.all(RANK >= 0)
    scalar_logical_ok = (
        isinstance(any_scalar, bool)
        and isinstance(all_scalar, bool)
        and any_scalar
        and all_scalar
    )

    progress("checking reduce-to-nonzero-root behavior")
    root = SIZE - 1
    root_result = mpi.reduce.max(float(RANK), mode="root", root=root)
    root_ok = root_result == float(SIZE - 1) if RANK == root else root_result is None

    progress("checking invalid reduction arguments and unsupported dtypes")
    validation_ok = all(
        (
            _raises(ValueError, mpi.reduce.sum, 1.0, mode="invalid", contains="mode"),
            _raises(
                ValueError,
                mpi.reduce.sum,
                1.0,
                mode="root",
                root=-1,
                contains="root",
            ),
            _raises(
                ValueError,
                mpi.reduce.sum,
                1.0,
                mode="root",
                root=True,
                contains="root",
            ),
            _raises(
                ValueError,
                mpi.reduce.sum,
                1.0,
                mode="root",
                root=SIZE,
                contains="outside",
            ),
            _raises(
                mpi.MPIError,
                mpi.reduce.sum,
                np.asarray(["unsupported"], dtype=object),
                contains="Unsupported MPI NumPy dtype",
            ),
            _raises(
                mpi.MPIError,
                mpi.datatype,
                np.dtype("U1"),
                contains="Unsupported MPI NumPy dtype",
            ),
        )
    )

    correct = bool(
        mpi.reduce.all(
            scalar == expected_scalar
            and noncontiguous_ok
            and dataset_ok
            and scalar_logical_ok
            and root_ok
            and validation_ok
        )
    )
    progress("recording result")
    record(
        "mpi runtime/reduce contracts (validation/noncontiguous/nonzero root)",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


# ---------------------------------------------------------------------------
# mpi.reduce -- element-wise collective reductions
# ---------------------------------------------------------------------------


def test_reduce_sum_scalar(n_total: int) -> None:
    """Scalar mpi.reduce.sum using real SHiELD precipitation values."""
    with xr.open_dataset(DEFAULT_NETCDF_SOURCE) as source:
        n_lat = int(source.sizes["lat"])
        n_lon = int(source.sizes["lon"])

    n_rows = min(n_lat, max(1, (n_total + n_lon - 1) // n_lon))
    start, stop = _rank_bounds(n_rows)
    local = _load_source_variable(
        "pr",
        time=0,
        lat=slice(start, stop),
    )

    progress("entering timed parallel section")
    with timed() as box:
        local_partial = float(local.sum(skipna=True))
        combined = mpi.reduce.sum(local_partial)
    parallel_s = box["seconds"]

    def serial_fn() -> float:
        field = _load_source_variable("pr", time=0, lat=slice(0, n_rows))
        return float(field.sum(skipna=True))

    progress("running serial baseline")
    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(
        np.isclose(
            combined,
            expected,
            rtol=dtype_rtol(local.values),
            equal_nan=True,
        )
    )
    progress("recording result")
    record(
        f"mpi.reduce.sum scalar ({n_rows * n_lon} real pr values)",
        correct,
        serial_s,
        parallel_s,
    )


def test_reduce_composite(n_events_total: int, ny: int, nx: int) -> None:
    """mpi.reduce.sum on real SHiELD precipitation fields from rank-selected times."""
    with xr.open_dataset(DEFAULT_NETCDF_SOURCE) as source:
        n_time = int(source.sizes["time"])
        n_lat = min(int(source.sizes["lat"]), ny)
        n_lon = min(int(source.sizes["lon"]), nx)

    max_points_per_rank = max(1, n_events_total // SIZE)
    if n_lat * n_lon > max_points_per_rank:
        n_lat = max(1, min(n_lat, max_points_per_rank // max(1, n_lon)))

    def load_rank_field(rank: int) -> np.ndarray:
        field = _load_source_variable(
            "pr",
            time=rank % n_time,
            lat=slice(0, n_lat),
            lon=slice(0, n_lon),
        )
        return np.asarray(field.values)

    local = load_rank_field(RANK)

    progress("entering timed parallel section")
    with timed() as box:
        combined = mpi.reduce.sum(local)
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        fields = [load_rank_field(rank) for rank in range(SIZE)]
        return np.sum(np.stack(fields, axis=0), axis=0)

    progress("running serial baseline")
    expected, serial_s = run_serial_baseline(serial_fn)
    correct = (
        bool(np.allclose(combined, expected, rtol=dtype_rtol(local), equal_nan=True))
        and combined.dtype == local.dtype
    )
    progress("recording result")
    record(
        f"mpi.reduce.sum real pr fields ({n_lat}x{n_lon}, {SIZE} rank selections)",
        correct,
        serial_s,
        parallel_s,
    )


def test_reduce_xarray_object(ny: int, nx: int) -> None:
    """mpi.reduce.sum on a real SHiELD xarray DataArray with metadata preserved."""
    with xr.open_dataset(DEFAULT_NETCDF_SOURCE) as source:
        n_time = int(source.sizes["time"])
        n_lat = min(int(source.sizes["lat"]), ny)
        n_lon = min(int(source.sizes["lon"]), nx)

    def load_rank_field(rank: int) -> xr.DataArray:
        return _load_source_variable(
            "t2m",
            time=rank % n_time,
            lat=slice(0, n_lat),
            lon=slice(0, n_lon),
        )

    local = load_rank_field(RANK)

    progress("entering timed parallel section")
    with timed() as box:
        combined = mpi.reduce.sum(local)
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        fields = [load_rank_field(rank).values for rank in range(SIZE)]
        return np.sum(np.stack(fields, axis=0), axis=0)

    progress("running serial baseline")
    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(
        np.allclose(
            combined.values,
            expected,
            rtol=dtype_rtol(local.values),
            equal_nan=True,
        )
    ) and combined.attrs.get("units") == local.attrs.get("units")
    progress("recording result")
    record(
        "mpi.reduce.sum real t2m DataArray (dims/attrs kept)",
        correct,
        serial_s,
        parallel_s,
        note="correctness-focused",
    )


def test_reduce_operations() -> None:
    """Exercise every mpi.reduce operation using real SHiELD values."""
    with xr.open_dataset(DEFAULT_NETCDF_SOURCE) as source:
        n_time = int(source.sizes["time"])
        n_lat = int(source.sizes["lat"])
        n_lon = int(source.sizes["lon"])

    numeric_width = min(2, n_lon)
    logical_width = min(3, n_lon)

    def load_numeric(rank: int) -> np.ndarray:
        data = _load_source_variable(
            "t2m",
            time=rank % n_time,
            lat=rank % n_lat,
            lon=slice(0, numeric_width),
        )
        return np.asarray(data.values)

    def load_logical(rank: int) -> np.ndarray:
        mask = _load_source_variable(
            "slmsk",
            time=rank % n_time,
            lat=rank % n_lat,
            lon=slice(0, logical_width),
        )
        return np.asarray(mask.values == 1)

    numeric = load_numeric(RANK)
    numeric_stack = np.stack([load_numeric(rank) for rank in range(SIZE)], axis=0)
    logical = load_logical(RANK)
    logical_stack = np.stack([load_logical(rank) for rank in range(SIZE)], axis=0)

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
        progress(f"running mpi.reduce.{op_name} all mode")
        with timed() as box:
            result = op(value)
        parallel_s = box["seconds"]

        progress(f"checking mpi.reduce.{op_name} root mode")
        root_result = op(value, mode="root", root=0)
        tolerance = dtype_rtol(expected)
        all_mode_ok = bool(
            np.allclose(result, expected, rtol=tolerance, equal_nan=True)
        )
        root_mode_ok = (
            bool(np.allclose(root_result, expected, rtol=tolerance, equal_nan=True))
            if RANK == 0
            else root_result is None
        )
        correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
        progress("recording result")
        record(
            f"mpi.reduce.{op_name} real SHiELD values (all/root modes)",
            correct,
            0.0,
            parallel_s,
            note="correctness-focused",
        )

    mask = _load_source_variable(
        "slmsk",
        time=RANK % n_time,
        lat=RANK % n_lat,
        lon=slice(0, logical_width),
    )
    logical_dataset = xr.Dataset(
        {
            "land": mask == 1,
            "nonsea": mask != 0,
        }
    )
    dataset_any = mpi.reduce.any(logical_dataset)
    dataset_all = mpi.reduce.all(logical_dataset)

    land_stack = np.stack(
        [
            _load_source_variable(
                "slmsk",
                time=rank % n_time,
                lat=rank % n_lat,
                lon=slice(0, logical_width),
            ).values
            == 1
            for rank in range(SIZE)
        ],
        axis=0,
    )
    nonsea_stack = np.stack(
        [
            _load_source_variable(
                "slmsk",
                time=rank % n_time,
                lat=rank % n_lat,
                lon=slice(0, logical_width),
            ).values
            != 0
            for rank in range(SIZE)
        ],
        axis=0,
    )
    dataset_ok = (
        isinstance(dataset_any, xr.Dataset)
        and isinstance(dataset_all, xr.Dataset)
        and bool(np.array_equal(dataset_any["land"].values, land_stack.any(axis=0)))
        and bool(np.array_equal(dataset_any["nonsea"].values, nonsea_stack.any(axis=0)))
        and bool(np.array_equal(dataset_all["land"].values, land_stack.all(axis=0)))
        and bool(np.array_equal(dataset_all["nonsea"].values, nonsea_stack.all(axis=0)))
    )
    progress("recording result")
    record(
        "mpi.reduce.any/all real slmsk Dataset",
        bool(mpi.reduce.all(dataset_ok)),
        0.0,
        0.0,
        note="correctness-focused",
    )


# ---------------------------------------------------------------------------
# mpi.xarray -- distributed xarray operations
# ---------------------------------------------------------------------------


def test_xarray_open_dataset() -> None:
    """mpi.xarray.open_dataset on the real SHiELD file partitioned by latitude."""
    progress("entering timed parallel section")
    with timed() as box:
        distributed = mpi.xarray.open_dataset(
            str(DEFAULT_NETCDF_SOURCE),
            partition_dim="lat",
        )[["pr"]]
        distributed["pr"].isel(time=0).load()
    parallel_s = box["seconds"]

    meta = distributed.attrs.get("mpi_meta")
    if isinstance(meta, dict):
        partition_dim = meta.get("dim")
        start_idx = int(meta.get("start", -1))
        stop_idx = int(meta.get("stop", -1))
        local_size = (
            int(distributed.sizes.get(partition_dim, 0))
            if isinstance(partition_dim, str)
            else 0
        )
        coordinate = (
            distributed[partition_dim]
            if isinstance(partition_dim, str) and partition_dim in distributed.coords
            else None
        )
        start_value = (
            coordinate.values[0] if coordinate is not None and local_size else None
        )
        end_value = (
            coordinate.values[-1] if coordinate is not None and local_size else None
        )
        end_idx = stop_idx - 1 if local_size else None
    else:
        partition_dim = None
        start_idx = None
        stop_idx = None
        end_idx = None
        start_value = None
        end_value = None

    partition_rows = mpi.comm.allgather(
        (partition_dim, start_idx, end_idx, start_value, end_value)
    )
    if RANK == 0:
        for rank, row in enumerate(partition_rows):
            dim, idx_start, idx_end, value_start, value_end = row
            mpi.log(
                f"[PARTITION RANK {rank}/{SIZE}] partition_dim={dim} "
                + f"idx_start={idx_start} idx_end={idx_end} "
                + f"value_start={value_start} value_end={value_end}",
                prefix=False,
                flush=True,
            )

    local = distributed["pr"].isel(time=0).values.copy()
    variable_meta = distributed["pr"].attrs.get("mpi_meta")
    n_lat = int(meta.get("global_size", -1)) if isinstance(meta, dict) else -1
    local_lat_axis = distributed["pr"].isel(time=0).get_axis_num("lat")
    distributed.close()

    parts = mpi.comm.allgather(local)
    assembled = np.concatenate(parts, axis=local_lat_axis)
    expected = _load_source_variable("pr", time=0).values
    local_meta_ok = (
        isinstance(meta, dict)
        and meta.get("dim") == "lat"
        and n_lat == expected.shape[local_lat_axis]
        and int(meta.get("stop", -1)) - int(meta.get("start", -1))
        == local.shape[local_lat_axis]
        and isinstance(variable_meta, dict)
        and variable_meta.get("dim") == "lat"
    )
    correct = bool(np.array_equal(assembled, expected, equal_nan=True)) and bool(
        mpi.reduce.all(local_meta_ok)
    )

    progress("checking open_dataset partition-dimension validation")
    open_validation_ok = _raises(
        ValueError,
        mpi.xarray.open_dataset,
        str(DEFAULT_NETCDF_SOURCE),
        partition_dim="missing",
        contains="partition_dim",
    )
    correct = correct and bool(mpi.reduce.all(open_validation_ok))

    def serial_fn() -> float:
        field = _load_source_variable("pr", time=0)
        return float(field.sum(skipna=True))

    progress("running serial baseline")
    _, serial_s = run_serial_baseline(serial_fn)
    progress("recording result")
    record(
        "mpi.xarray.open_dataset (real pr, partitioned latitude)",
        correct,
        serial_s,
        parallel_s,
    )


def test_xarray_redistribute(ny: int, nx: int) -> None:
    """mpi.xarray.redistribute using a real SHiELD precipitation field."""
    full = _load_source_variable(
        "pr",
        time=0,
        lat=slice(0, ny),
        lon=slice(0, nx),
    )

    progress("entering timed parallel section")
    with timed() as box:
        distributed = mpi.xarray.redistribute(full, "lat")
    parallel_s = box["seconds"]
    auto = mpi.xarray.redistribute(full, "auto")

    explicit_parts = mpi.comm.allgather(distributed.values)
    explicit_meta = distributed.attrs.get("mpi_meta")
    auto_parts = mpi.comm.allgather(auto.values)
    auto_meta = auto.attrs.get("mpi_meta")

    auto_dim = auto_meta.get("dim") if isinstance(auto_meta, dict) else None
    auto_axis = full.get_axis_num(auto_dim) if isinstance(auto_dim, str) else 0
    local_ok = (
        isinstance(explicit_meta, dict)
        and explicit_meta.get("dim") == "lat"
        and int(explicit_meta.get("global_size", -1)) == full.sizes["lat"]
        and isinstance(auto_meta, dict)
        and isinstance(auto_dim, str)
        and int(auto_meta.get("global_size", -1)) == full.sizes[auto_dim]
    )
    correct = (
        bool(
            np.array_equal(
                np.concatenate(explicit_parts, axis=full.get_axis_num("lat")),
                full.values,
                equal_nan=True,
            )
        )
        and bool(
            np.array_equal(
                np.concatenate(auto_parts, axis=auto_axis),
                full.values,
                equal_nan=True,
            )
        )
        and bool(mpi.reduce.all(local_ok))
    )
    progress("recording result")
    record(
        "mpi.xarray.redistribute real pr (explicit/auto)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


def test_xarray_isel(ny: int, nx: int) -> None:
    """mpi.xarray.isel using global latitude indices on real SHiELD precipitation."""
    full = _load_source_variable(
        "pr",
        time=0,
        lat=slice(0, ny),
        lon=slice(0, nx),
    )
    distributed = mpi.xarray.redistribute(full, "lat")
    n_lat = full.sizes["lat"]
    if n_lat < 3:
        raise ValueError(
            "The selected SHiELD latitude range must contain at least 3 rows."
        )

    start = 1
    stop = n_lat - 1
    scalar_index = n_lat // 2

    progress("entering timed parallel section")
    with timed() as box:
        sliced = mpi.xarray.isel(distributed, lat=slice(start, stop))
        scalar = mpi.xarray.isel(distributed, lat=scalar_index)
    parallel_s = box["seconds"]

    sliced_parts = mpi.comm.allgather(sliced.values)
    assembled = np.concatenate(sliced_parts, axis=sliced.get_axis_num("lat"))
    expected_slice = full.isel(lat=slice(start, stop)).values
    expected_scalar = full.isel(lat=scalar_index).values
    meta = sliced.attrs.get("mpi_meta")
    local_meta_ok = (
        isinstance(meta, dict)
        and meta.get("dim") == "lat"
        and int(meta.get("global_size", -1)) == stop - start
    )
    correct = (
        bool(np.array_equal(assembled, expected_slice, equal_nan=True))
        and bool(np.array_equal(scalar.values, expected_scalar, equal_nan=True))
        and "mpi_meta" not in scalar.attrs
        and bool(mpi.reduce.all(local_meta_ok))
    )
    progress("recording result")
    record(
        "mpi.xarray.isel real pr (global latitude slice/scalar)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


def test_xarray_sel(ny: int, nx: int) -> None:
    """mpi.xarray.sel using real SHiELD latitude coordinate labels."""
    full = _load_source_variable(
        "pr",
        time=0,
        lat=slice(0, ny),
        lon=slice(0, nx),
    )
    distributed = mpi.xarray.redistribute(full, "lat")
    n_lat = full.sizes["lat"]
    if n_lat < 3:
        raise ValueError(
            "The selected SHiELD latitude range must contain at least 3 rows."
        )

    start_label = full["lat"].values[1].item()
    stop_label = full["lat"].values[-2].item()
    scalar_label = full["lat"].values[n_lat // 2].item()

    progress("entering timed parallel section")
    with timed() as box:
        sliced = mpi.xarray.sel(distributed, lat=slice(start_label, stop_label))
        scalar = mpi.xarray.sel(distributed, lat=scalar_label)
        nearest = mpi.xarray.sel(
            distributed,
            lat=scalar_label,
            method="nearest",
        )
    parallel_s = box["seconds"]

    sliced_parts = mpi.comm.allgather(sliced.values)
    assembled = np.concatenate(sliced_parts, axis=sliced.get_axis_num("lat"))
    expected_slice = full.sel(lat=slice(start_label, stop_label)).values
    expected_scalar = full.sel(lat=scalar_label).values
    expected_nearest = full.sel(lat=scalar_label, method="nearest").values
    meta = sliced.attrs.get("mpi_meta")
    local_meta_ok = isinstance(meta, dict) and meta.get("dim") == "lat"
    correct = (
        bool(np.array_equal(assembled, expected_slice, equal_nan=True))
        and bool(np.array_equal(scalar.values, expected_scalar, equal_nan=True))
        and bool(np.array_equal(nearest.values, expected_nearest, equal_nan=True))
        and "mpi_meta" not in scalar.attrs
        and "mpi_meta" not in nearest.attrs
        and bool(mpi.reduce.all(local_meta_ok))
    )
    progress("recording result")
    record(
        "mpi.xarray.sel real pr (global latitude slice/scalar)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


def test_xarray_reduction(n_levels_max: int, ny: int, nx: int, op_name: str) -> None:
    """Numeric mpi.xarray reductions using the real SHiELD temperature profile."""
    progress(f"preparing mpi.xarray.{op_name} source data")
    full = _load_source_variable(
        "t",
        time=0,
        plev=slice(0, n_levels_max),
        lat=slice(0, ny),
        lon=slice(0, nx),
    )
    distributed = mpi.xarray.redistribute(full, "plev")
    op = getattr(mpi.xarray, op_name)
    kwargs = {"skipna": True, "keep_attrs": True}
    if op_name in {"sum", "prod"}:
        kwargs["min_count"] = 1

    progress("entering timed parallel section")
    with timed() as box:
        result = op(distributed, dim="plev", **kwargs)
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        serial_kwargs = {"skipna": True}
        if op_name in {"sum", "prod"}:
            serial_kwargs["min_count"] = 1
        return getattr(full, op_name)(dim="plev", **serial_kwargs).values

    progress("running serial baseline")
    expected, serial_s = run_serial_baseline(serial_fn)
    root_result = op(distributed, dim="plev", mode="root", root=0, **kwargs)
    tolerance = dtype_rtol(expected)
    all_mode_ok = (
        result is not None
        and result.dtype == expected.dtype
        and bool(np.allclose(result.values, expected, rtol=tolerance, equal_nan=True))
        and result.attrs.get("units") == full.attrs.get("units")
    )
    root_mode_ok = (
        root_result is not None
        and bool(
            np.allclose(root_result.values, expected, rtol=tolerance, equal_nan=True)
        )
        if RANK == 0
        else root_result is None
    )
    correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
    progress("recording result")
    record(
        f"mpi.xarray.{op_name} real t over {full.sizes['plev']} pressure levels",
        correct,
        serial_s,
        parallel_s,
    )


def test_xarray_logical_reduction(
    n_lat_max: int,
    nx: int,
    op_name: str,
) -> None:
    """Logical mpi.xarray.any/all using the real SHiELD sea-land-ice mask."""
    progress(f"preparing mpi.xarray.{op_name} source data")
    mask = _load_source_variable(
        "slmsk",
        time=0,
        lat=slice(0, n_lat_max),
        lon=slice(0, nx),
    )
    full = mask == 1
    distributed = mpi.xarray.redistribute(full, "lat")
    op = getattr(mpi.xarray, op_name)

    progress("entering timed parallel section")
    with timed() as box:
        result = op(distributed, dim="lat")
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        return getattr(full, op_name)(dim="lat").values

    progress("running serial baseline")
    expected, serial_s = run_serial_baseline(serial_fn)
    root_result = op(distributed, dim="lat", mode="root", root=0)
    all_mode_ok = result is not None and bool(np.array_equal(result.values, expected))
    root_mode_ok = (
        root_result is not None and bool(np.array_equal(root_result.values, expected))
        if RANK == 0
        else root_result is None
    )
    correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
    progress("recording result")
    record(
        f"mpi.xarray.{op_name} real slmsk land mask ({full.sizes['lat']} latitudes)",
        correct,
        serial_s,
        parallel_s,
    )


def test_xarray_dataset_reduction(ny: int, nx: int) -> None:
    """Dataset reductions using real distributed t2m plus real static plev values."""
    t2m = _load_source_variable(
        "t2m",
        time=0,
        lat=slice(0, ny),
        lon=slice(0, nx),
    )
    plev_values = _load_source_variable("plev").rename("plev_values")
    full = xr.merge([t2m.to_dataset(name="t2m"), plev_values.to_dataset()])
    distributed = mpi.xarray.redistribute(full, "lat")

    progress("entering timed parallel section")
    with timed() as box:
        progress("reducing Dataset sum")
        result = mpi.xarray.sum(distributed, dim="lat")
        progress("reducing Dataset mean")
        mean_result = mpi.xarray.mean(distributed, dim=("lat", "lon"))
    parallel_s = box["seconds"]

    progress("checking Dataset reduction results")
    expected = full.sum(dim="lat")
    expected_mean = full.mean(dim=("lat", "lon"))
    correct = (
        result is not None
        and mean_result is not None
        and result["t2m"].dtype == expected["t2m"].dtype
        and bool(
            np.allclose(
                result["t2m"].values,
                expected["t2m"].values,
                rtol=dtype_rtol(expected["t2m"].values),
                equal_nan=True,
            )
        )
        and bool(
            np.array_equal(
                result["plev_values"].values,
                expected["plev_values"].values,
            )
        )
        and mean_result["t2m"].dtype == expected_mean["t2m"].dtype
        and bool(
            np.allclose(
                mean_result["t2m"].values,
                expected_mean["t2m"].values,
                rtol=dtype_rtol(expected_mean["t2m"].values),
                equal_nan=True,
            )
        )
        and bool(
            np.array_equal(
                mean_result["plev_values"].values,
                expected_mean["plev_values"].values,
            )
        )
        and "mpi_meta" not in result.attrs
        and "mpi_meta" not in mean_result.attrs
    )

    progress("preparing empty-partition extreme reductions")
    profile = _load_source_variable(
        "t",
        time=0,
        plev=slice(0, max(1, SIZE - 1)),
        lat=0,
        lon=0,
    )
    profile_distributed = mpi.xarray.redistribute(profile, "plev")
    progress("reducing empty-partition minimum")
    minimum = mpi.xarray.min(profile_distributed, dim="plev")
    progress("reducing empty-partition maximum")
    maximum = mpi.xarray.max(profile_distributed, dim="plev")
    correct = (
        correct
        and minimum is not None
        and maximum is not None
        and bool(
            np.isclose(
                float(minimum.item()),
                float(profile.min(skipna=True).item()),
                rtol=dtype_rtol(profile.values),
            )
        )
        and bool(
            np.isclose(
                float(maximum.item()),
                float(profile.max(skipna=True).item()),
                rtol=dtype_rtol(profile.values),
            )
        )
    )
    correct = bool(mpi.reduce.all(correct))
    progress("recording result")
    record(
        "mpi.xarray Dataset reductions (real distributed/static variables)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


def test_xarray_contracts() -> None:
    """Exercise distributed xarray edge cases and validation with deterministic data."""
    full = xr.DataArray(
        np.arange(30, dtype=np.float64).reshape(5, 6),
        dims=("lat", "lon"),
        coords={
            "lat": np.asarray([-60.0, -30.0, 0.0, 30.0, 60.0]),
            "lon": np.arange(6, dtype=np.int64),
        },
        attrs={"units": "1"},
        name="field",
    )
    distributed = mpi.xarray.redistribute(full, "lat")
    chunked = mpi.xarray.redistribute(
        full,
        "lat",
        chunk_info={"lat": 2, "lon": 3},
    )
    chunked_meta = chunked.attrs.get("mpi_meta")
    chunk_info_ok = isinstance(chunked_meta, dict) and chunked_meta.get(
        "chunk_info"
    ) == {"lat": 2, "lon": 3}

    progress("checking global scalar indexing and plain-xarray passthrough")
    negative = mpi.xarray.isel(distributed, lat=-1)
    mapped = mpi.xarray.isel(
        distributed,
        {"lat": -2},
        lon=slice(1, 4),
    )
    plain = mpi.xarray.isel(full, lat=1, lon=slice(0, 2))
    local_lon = mpi.xarray.isel(distributed, lon=slice(1, 3))
    nearest = mpi.xarray.sel(distributed, lat=-20.0, method="nearest")
    scalar_auto = mpi.xarray.redistribute(xr.DataArray(7.0), "auto")
    local_lon_meta = local_lon.attrs.get("mpi_meta")
    indexing_ok = (
        bool(np.array_equal(negative.values, full.isel(lat=-1).values))
        and bool(
            np.array_equal(
                mapped.values,
                full.isel(lat=-2, lon=slice(1, 4)).values,
            )
        )
        and bool(
            np.array_equal(
                plain.values,
                full.isel(lat=1, lon=slice(0, 2)).values,
            )
        )
        and bool(
            np.array_equal(
                nearest.values,
                full.sel(lat=-20.0, method="nearest").values,
            )
        )
        and isinstance(local_lon_meta, dict)
        and local_lon_meta.get("dim") == "lat"
        and chunk_info_ok
        and scalar_auto.attrs.get("mpi_meta") is None
    )

    progress("checking reduction edge cases, including empty rank partitions")
    total_sum = mpi.xarray.sum(distributed, dim=None)
    ellipsis_sum = mpi.xarray.sum(distributed, dim=...)
    tuple_mean = mpi.xarray.mean(distributed, dim=("lat", "lon"))
    minimum = mpi.xarray.min(distributed, dim="lat")
    maximum = mpi.xarray.max(distributed, dim="lat")

    nan_full = xr.DataArray(
        np.asarray(
            [
                [np.nan, 1.0, 1.0],
                [np.nan, 2.0, 1.0],
                [3.0, 3.0, 1.0],
                [4.0, 4.0, 1.0],
                [5.0, 5.0, 1.0],
            ]
        ),
        dims=("lat", "lon"),
        name="nan_field",
    )
    nan_distributed = mpi.xarray.redistribute(nan_full, "lat")
    min_count_sum = mpi.xarray.sum(
        nan_distributed,
        dim="lat",
        skipna=True,
        min_count=4,
    )
    min_count_prod = mpi.xarray.prod(
        nan_distributed,
        dim="lat",
        skipna=True,
        min_count=4,
    )
    nan_mean_skip = mpi.xarray.mean(nan_distributed, dim="lat", skipna=True)
    nan_mean_keep = mpi.xarray.mean(nan_distributed, dim="lat", skipna=False)
    nan_min_skip = mpi.xarray.min(nan_distributed, dim="lat", skipna=True)
    nan_min_keep = mpi.xarray.min(nan_distributed, dim="lat", skipna=False)
    nan_max_skip = mpi.xarray.max(nan_distributed, dim="lat", skipna=True)
    nan_max_keep = mpi.xarray.max(nan_distributed, dim="lat", skipna=False)

    empty_length = max(1, SIZE - 1)
    integer_full = xr.DataArray(
        np.arange(empty_length, dtype=np.int64),
        dims=("sample",),
        name="integer_field",
    )
    integer_distributed = mpi.xarray.redistribute(integer_full, "sample")
    integer_min = mpi.xarray.min(integer_distributed, dim="sample")
    integer_max = mpi.xarray.max(integer_distributed, dim="sample")
    boolean_full = (integer_full % 2 == 0).rename("boolean_field")
    boolean_distributed = mpi.xarray.redistribute(boolean_full, "sample")
    boolean_min = mpi.xarray.min(boolean_distributed, dim="sample")
    boolean_max = mpi.xarray.max(boolean_distributed, dim="sample")
    empty_partition_seen = mpi.reduce.any(int(integer_distributed.sizes["sample"]) == 0)
    empty_partition_ok = SIZE == 1 or empty_partition_seen

    reduction_ok = (
        total_sum is not None
        and ellipsis_sum is not None
        and tuple_mean is not None
        and minimum is not None
        and maximum is not None
        and min_count_sum is not None
        and min_count_prod is not None
        and nan_mean_skip is not None
        and nan_mean_keep is not None
        and nan_min_skip is not None
        and nan_min_keep is not None
        and nan_max_skip is not None
        and nan_max_keep is not None
        and integer_min is not None
        and integer_max is not None
        and boolean_min is not None
        and boolean_max is not None
        and bool(np.isclose(float(total_sum.item()), float(full.sum().item())))
        and bool(np.isclose(float(ellipsis_sum.item()), float(full.sum().item())))
        and bool(np.isclose(float(tuple_mean.item()), float(full.mean().item())))
        and bool(np.array_equal(minimum.values, full.min(dim="lat").values))
        and bool(np.array_equal(maximum.values, full.max(dim="lat").values))
        and bool(
            np.allclose(
                min_count_sum.values,
                nan_full.sum(dim="lat", skipna=True, min_count=4).values,
                equal_nan=True,
            )
        )
        and bool(
            np.allclose(
                min_count_prod.values,
                nan_full.prod(dim="lat", skipna=True, min_count=4).values,
                equal_nan=True,
            )
        )
        and bool(
            np.allclose(
                nan_mean_skip.values,
                nan_full.mean(dim="lat", skipna=True).values,
                equal_nan=True,
            )
        )
        and bool(
            np.allclose(
                nan_mean_keep.values,
                nan_full.mean(dim="lat", skipna=False).values,
                equal_nan=True,
            )
        )
        and bool(
            np.allclose(
                nan_min_skip.values,
                nan_full.min(dim="lat", skipna=True).values,
                equal_nan=True,
            )
        )
        and bool(
            np.allclose(
                nan_min_keep.values,
                nan_full.min(dim="lat", skipna=False).values,
                equal_nan=True,
            )
        )
        and bool(
            np.allclose(
                nan_max_skip.values,
                nan_full.max(dim="lat", skipna=True).values,
                equal_nan=True,
            )
        )
        and bool(
            np.allclose(
                nan_max_keep.values,
                nan_full.max(dim="lat", skipna=False).values,
                equal_nan=True,
            )
        )
        and int(integer_min.item()) == int(integer_full.min().item())
        and int(integer_max.item()) == int(integer_full.max().item())
        and bool(boolean_min.item()) == bool(boolean_full.min().item())
        and bool(boolean_max.item()) == bool(boolean_full.max().item())
        and bool(empty_partition_ok)
    )

    progress("checking nonzero-root reduction placement")
    root = SIZE - 1
    root_sum = mpi.xarray.sum(distributed, dim="lat", mode="root", root=root)
    root_ok = (
        root_sum is not None
        and bool(np.array_equal(root_sum.values, full.sum(dim="lat").values))
        if RANK == root
        else root_sum is None
    )

    progress("checking distributed xarray validation failures")
    complex_full = xr.DataArray(
        np.arange(max(1, SIZE - 1), dtype=np.float64) + 1.0j,
        dims=("sample",),
        name="complex_field",
    )
    complex_distributed = mpi.xarray.redistribute(complex_full, "sample")
    validation_ok = all(
        (
            _raises(
                ValueError,
                mpi.xarray.redistribute,
                distributed,
                "lon",
                contains="already distributed",
            ),
            _raises(
                ValueError,
                mpi.xarray.redistribute,
                full,
                "missing",
                contains="does not exist",
            ),
            _raises(
                ValueError,
                mpi.xarray.sum,
                distributed,
                dim="lon",
                contains="Distributed dimension",
            ),
            _raises(
                ValueError,
                mpi.xarray.sum,
                distributed,
                dim="lat",
                mode="invalid",
                contains="mode",
            ),
            _raises(
                ValueError,
                mpi.xarray.sum,
                distributed,
                dim="lat",
                mode="root",
                root=SIZE,
                contains="outside",
            ),
            _raises(
                NotImplementedError,
                mpi.xarray.isel,
                distributed,
                lat=slice(None, None, 2),
                contains="step 1",
            ),
            _raises(
                NotImplementedError,
                mpi.xarray.isel,
                distributed,
                lat=[0, 1],
                contains="slices and scalar",
            ),
            _raises(
                IndexError,
                mpi.xarray.isel,
                distributed,
                lat=full.sizes["lat"],
                contains="out of bounds",
            ),
            _raises(
                NotImplementedError,
                mpi.xarray.sel,
                distributed,
                lat=[-60.0, 0.0],
                contains="slices and scalar",
            ),
            _raises(
                KeyError,
                mpi.xarray.sel,
                distributed,
                lat=999.0,
                contains="No rank contains label",
            ),
            _raises(
                KeyError,
                mpi.xarray.sel,
                distributed,
                lat=-20.0,
                method="nearest",
                tolerance=1.0,
            ),
            _raises(
                TypeError,
                mpi.xarray.sum,
                np.arange(3),
                dim="x",
                contains="require an xarray",
            ),
            _raises(
                TypeError,
                mpi.xarray.min,
                complex_distributed,
                dim="sample",
                contains="minimum",
            ),
            _raises(
                TypeError,
                mpi.xarray.max,
                complex_distributed,
                dim="sample",
                contains="maximum",
            ),
        )
    )

    correct = bool(
        mpi.reduce.all(indexing_ok and reduction_ok and root_ok and validation_ok)
    )
    progress("recording result")
    record(
        "mpi.xarray contracts (edge reductions/indexing/validation/nonzero root)",
        correct,
        0.0,
        0.0,
        note="deterministic edge cases",
    )


def test_xarray_redistribute_on(n_lat_max: int, nx: int) -> None:
    """Redistribute a real reduction result along the remaining longitude dimension."""
    full = _load_source_variable(
        "t2m",
        time=0,
        lat=slice(0, n_lat_max),
        lon=slice(0, nx),
    )
    distributed = mpi.xarray.redistribute(full, "lat")

    progress("entering timed parallel section")
    with timed() as box:
        result = mpi.xarray.mean(distributed, dim="lat", redistribute_on="lon")
    parallel_s = box["seconds"]

    progress("checking redistribute_on='auto'")
    auto_result = mpi.xarray.mean(
        distributed,
        dim="lat",
        redistribute_on="auto",
    )
    parts = mpi.comm.allgather(result.values)
    assembled = np.concatenate(parts, axis=result.get_axis_num("lon"))
    auto_parts = mpi.comm.allgather(auto_result.values)
    auto_assembled = np.concatenate(
        auto_parts,
        axis=auto_result.get_axis_num("lon"),
    )

    def serial_fn() -> np.ndarray:
        return full.mean(dim="lat").values

    progress("running serial baseline")
    expected, serial_s = run_serial_baseline(serial_fn)
    meta = result.attrs.get("mpi_meta")
    auto_meta = auto_result.attrs.get("mpi_meta")
    local_meta_ok = (
        isinstance(meta, dict)
        and meta.get("dim") == "lon"
        and int(meta.get("global_size", -1)) == full.sizes["lon"]
        and isinstance(auto_meta, dict)
        and auto_meta.get("dim") == "lon"
        and int(auto_meta.get("global_size", -1)) == full.sizes["lon"]
    )
    correct = (
        bool(
            np.allclose(
                assembled,
                expected,
                rtol=dtype_rtol(expected),
                equal_nan=True,
            )
        )
        and bool(
            np.allclose(
                auto_assembled,
                expected,
                rtol=dtype_rtol(expected),
                equal_nan=True,
            )
        )
        and bool(mpi.reduce.all(local_meta_ok))
    )
    progress("recording result")
    record(
        "mpi.xarray.mean(real t2m, redistribute_on='lon')",
        correct,
        serial_s,
        parallel_s,
    )


# ---------------------------------------------------------------------------
# mpi.scatterv -- vector scatter (data movement, not a compute reduction)
# ---------------------------------------------------------------------------


def test_scatterv(n_total: int) -> None:
    """Scatter rows from a real SHiELD t2m field."""
    with xr.open_dataset(DEFAULT_NETCDF_SOURCE) as source_ds:
        n_lat = int(source_ds.sizes["lat"])
        n_lon = min(3, int(source_ds.sizes["lon"]))
        dtype = np.dtype(source_ds["t2m"].dtype)

    total = min(n_lat, max(1, n_total // max(1, n_lon)))
    counts = [total // SIZE + (1 if rank < total % SIZE else 0) for rank in range(SIZE)]

    progress("entering timed parallel section")
    with timed() as box:
        source = None
        if RANK == 0:
            source = _load_source_variable(
                "t2m",
                time=0,
                lat=slice(0, total),
                lon=slice(0, n_lon),
            ).values
        recv = mpi.scatterv(source, counts, (counts[RANK], n_lon), dtype, root=0)
    parallel_s = box["seconds"]

    start = sum(counts[:RANK])
    expected_local = _load_source_variable(
        "t2m",
        time=0,
        lat=slice(start, start + counts[RANK]),
        lon=slice(0, n_lon),
    ).values
    correct = bool(np.array_equal(recv, expected_local, equal_nan=True))
    progress("recording result")
    record(
        f"mpi.scatterv real t2m ({total} rows across {SIZE} rank(s))",
        bool(mpi.reduce.all(correct)),
        0.0,
        parallel_s,
        note="data movement, no serial-compute equivalent",
    )


def test_scatterv_contracts() -> None:
    """Exercise scatterv validation, nonzero roots, non-contiguous sends,
    and zero rows.
    """
    progress("checking scatterv validation failures")
    invalid_counts_ok = _raises(
        ValueError,
        mpi.scatterv,
        None,
        [0] * (SIZE + 1),
        (0,),
        np.float64,
        contains="counts",
    )
    unsupported_dtype_ok = _raises(
        mpi.MPIError,
        mpi.scatterv,
        None,
        [0] * SIZE,
        (0,),
        np.dtype(object),
        contains="Unsupported MPI NumPy dtype",
    )
    missing_root_array_ok = True
    if SIZE == 1:
        missing_root_array_ok = _raises(
            ValueError,
            mpi.scatterv,
            None,
            [1],
            (1,),
            np.float64,
            contains="cannot be None",
        )

    progress("checking nonzero-root scatter and zero-length local receives")
    root = SIZE - 1
    total = max(1, SIZE - 1)
    counts = [total // SIZE + (1 if rank < total % SIZE else 0) for rank in range(SIZE)]
    expected_source = np.arange(total * 4, dtype=np.float64).reshape(total, 4)[:, ::2]
    source = expected_source if RANK == root else None
    received = mpi.scatterv(
        source,
        counts,
        (counts[RANK], 2),
        np.float64,
        root=root,
    )
    start = sum(counts[:RANK])
    expected = expected_source[start : start + counts[RANK]]
    scatter_ok = bool(np.array_equal(received, expected))

    correct = bool(
        mpi.reduce.all(
            invalid_counts_ok
            and unsupported_dtype_ok
            and missing_root_array_ok
            and scatter_ok
        )
    )
    progress("recording result")
    record(
        "mpi.scatterv contracts (validation/nonzero root/noncontiguous/zero rows)",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


# ---------------------------------------------------------------------------
# A realistic xarray + mpi.reduce composition: cosine-latitude weighted mean
# ---------------------------------------------------------------------------


def test_weighted_mean(n_lat_total: int, n_lon: int) -> None:
    """Cosine-latitude weighted mean of the real SHiELD 2 m temperature field."""
    with xr.open_dataset(DEFAULT_NETCDF_SOURCE) as source:
        n_lat = min(int(source.sizes["lat"]), n_lat_total)
        n_lon_used = min(int(source.sizes["lon"]), n_lon)

    start, stop = _rank_bounds(n_lat)
    local = _load_source_variable(
        "t2m",
        time=0,
        lat=slice(start, stop),
        lon=slice(0, n_lon_used),
    )

    progress("entering timed parallel section")
    with timed() as box:
        weights = np.cos(np.deg2rad(local["lat"]))
        local_weighted_sum = (local * weights).sum(skipna=True)
        local_weight_sum = (xr.ones_like(local) * weights).where(local.notnull()).sum()
        global_weighted_sum = mpi.reduce.sum(float(local_weighted_sum))
        global_weight_sum = mpi.reduce.sum(float(local_weight_sum))
        weighted_mean = global_weighted_sum / global_weight_sum
    parallel_s = box["seconds"]

    def serial_fn() -> float:
        full = _load_source_variable(
            "t2m",
            time=0,
            lat=slice(0, n_lat),
            lon=slice(0, n_lon_used),
        )
        weights = np.cos(np.deg2rad(full["lat"]))
        numerator = (full * weights).sum(skipna=True)
        denominator = (xr.ones_like(full) * weights).where(full.notnull()).sum()
        return float(numerator / denominator)

    progress("running serial baseline")
    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(
        np.isclose(
            weighted_mean,
            expected,
            rtol=dtype_rtol(local.values),
            equal_nan=True,
        )
    )
    progress("recording result")
    record(
        f"cosine-lat weighted mean real t2m ({n_lat}x{n_lon_used})",
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
            "  @mpi error propagation  : caught "
            + f"{type(exc).__name__} on every rank: {exc}"
        )

    progress("checking nonzero-root decorator and argument validation")

    @mpi(broadcast=True, root=SIZE - 1)
    def nonzero_root_setup() -> int:
        return RANK

    nonzero_root_result = nonzero_root_setup()
    ok_nonzero_root = nonzero_root_result == SIZE - 1
    invalid_root_function = mpi(lambda: None, root=SIZE)
    ok_validation = all(
        (
            _raises(
                ValueError,
                mpi,
                lambda: None,
                all_ranks=True,
                broadcast=True,
                contains="incompatible",
            ),
            _raises(
                TypeError,
                mpi,
                42,
                contains="must be callable",
            ),
            _raises(
                ValueError,
                mpi,
                lambda: None,
                root=-1,
                contains="non-negative",
            ),
            _raises(
                ValueError,
                invalid_root_function,
                contains="outside",
            ),
            getattr(only_on_root, "mpi", False) is True,
            only_on_root.__name__ == "only_on_root",
        )
    )

    overall = bool(
        mpi.reduce.all(
            ok_root_only
            and ok_all_ranks
            and ok_broadcast
            and ok_error_propagation
            and ok_nonzero_root
            and ok_validation
        )
    )
    progress("recording result")
    record(
        "@mpi decorator (root/all_ranks/broadcast/error/validation)",
        overall,
        0.0,
        0.0,
        note="usage demo, not a speed test",
    )


def test_xgeo_helpers(out_dir: str) -> None:
    """Check public xgeo placeholder detection and front-door validation."""
    progress("checking MPI placeholder semantics")
    placeholder = xgeo.empty_dataset()
    normal_empty = xr.Dataset()
    integer_marker = xr.Dataset(attrs={"_climtools_no_data": 1})
    marker_dataarray = xr.DataArray(
        [1.0],
        dims=("x",),
        attrs={"_climtools_no_data": True},
    )
    placeholder_ok = (
        xgeo.dataset_is_empty(placeholder)
        and not placeholder.dims
        and not placeholder.data_vars
        and not xgeo.dataset_is_empty(normal_empty)
        and not xgeo.dataset_is_empty(integer_marker)
        and not xgeo.dataset_is_empty(marker_dataarray)
    )
    exports_ok = all(
        callable(getattr(xgeo, name, None))
        for name in ("append", "dataset_is_empty", "empty_dataset", "to_netcdf")
    )

    progress("checking xgeo.to_netcdf input and distributed-dimension validation")
    invalid_type_ok = _raises(
        TypeError,
        xgeo.to_netcdf,
        np.arange(3),
        os.path.join(out_dir, "should_not_exist_invalid_type.nc"),
        contains="xarray.Dataset or xarray.DataArray",
    )
    full = xr.DataArray(
        np.arange(12, dtype=np.float64).reshape(3, 4),
        dims=("time", "x"),
        name="field",
    )
    distributed = mpi.xarray.redistribute(full, "time")
    mismatch_ok = _raises(
        ValueError,
        xgeo.to_netcdf,
        distributed,
        os.path.join(out_dir, "should_not_exist_mismatch.nc"),
        partition_dim="x",
        parallel=True,
        allow_serial=True,
        contains="does not match distributed dimension",
    )

    correct = bool(
        mpi.reduce.all(
            exports_ok and placeholder_ok and invalid_type_ok and mismatch_ok
        )
    )
    progress("recording result")
    record(
        "xgeo helpers/to_netcdf front-door validation",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


def test_netcdf_validation_and_options(out_dir: str) -> None:
    """Check parallel NetCDF validation plus explicit chunk/filter/unlimited options."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        record_skip(
            "parallel NetCDF validation/options",
            "netCDF4 lacks parallel4 support",
        )
        return

    field_values = np.arange(12, dtype=np.float64).reshape(4, 3)
    data = xr.Dataset(
        {
            "field": (
                ("time", "x"),
                field_values,
                {"units": "1"},
            ),
            "transposed": (
                ("x", "time"),
                field_values.T + 100.0,
                {"units": "1"},
            ),
            "flag": (("time", "x"), field_values % 2 == 0),
            "label": ("x", np.asarray(["a", "b", "c"], dtype=object)),
        },
        coords={"time": np.arange(4), "x": np.arange(3)},
        attrs={"title": "parallel options test"},
    )
    expected_error = (ValueError, mpi.MPIError)

    progress("checking synchronized NetCDF preparation validation")
    invalid_partition_ok = _raises(
        expected_error,
        xgeo.to_netcdf,
        data,
        os.path.join(out_dir, "invalid_partition.nc"),
        partition_dim="missing",
        parallel=True,
        allow_serial=(SIZE == 1),
        contains="partition_dim",
    )
    invalid_compression_ok = _raises(
        expected_error,
        xgeo.to_netcdf,
        data,
        os.path.join(out_dir, "invalid_compression.nc"),
        partition_dim="time",
        parallel=True,
        complevel=10,
        allow_serial=(SIZE == 1),
        contains="0 through 9",
    )
    invalid_unlimited_ok = _raises(
        expected_error,
        xgeo.to_netcdf,
        data,
        os.path.join(out_dir, "invalid_unlimited.nc"),
        partition_dim="time",
        unlimited_dim="missing",
        parallel=True,
        allow_serial=(SIZE == 1),
        contains="Unknown unlimited dimensions",
    )

    unnamed_ok = True
    allow_serial_ok = True
    if SIZE == 1:
        unnamed = xr.DataArray(np.arange(4), dims=("time",))
        unnamed_ok = _raises(
            ValueError,
            xgeo.to_netcdf,
            unnamed,
            os.path.join(out_dir, "unnamed_dataarray.nc"),
            partition_dim="time",
            parallel=True,
            allow_serial=True,
            contains="must have a name",
        )
        allow_serial_ok = _raises(
            mpi.MPIError,
            xgeo.to_netcdf,
            data,
            os.path.join(out_dir, "allow_serial_required.nc"),
            partition_dim="time",
            parallel=True,
            allow_serial=False,
            contains="one process",
        )

    progress("writing explicit parallel chunk/compression/unlimited configuration")
    path = Path(out_dir) / "climtools_test_parallel_options.nc"
    xgeo.to_netcdf(
        data,
        path,
        unlimited_dim=("time",),
        partition_dim="time",
        parallel=True,
        chunks={
            "field": (2, 3),
            "transposed": (3, 2),
            "flag": (2, 3),
        },
        shuffle=False,
        zlib=True,
        complevel=1,
        nofill=False,
        allow_serial=(SIZE == 1),
    )
    mpi.comm.barrier()

    options_ok = True
    integrity_note = ""
    if RANK == 0:
        try:
            with netCDF4.Dataset(path) as nc:
                variable = nc.variables["field"]
                transposed = nc.variables["transposed"]
                flag = nc.variables["flag"]
                labels = nc.variables["label"]
                filters = variable.filters()
                chunking = variable.chunking()
                options_ok = (
                    nc.dimensions["time"].isunlimited()
                    and list(chunking) == [2, 3]
                    and list(transposed.chunking()) == [3, 2]
                    and bool(filters.get("zlib"))
                    and int(filters.get("complevel", -1)) == 1
                    and not bool(filters.get("shuffle"))
                    and np.array_equal(variable[:], data["field"].values)
                    and np.array_equal(
                        transposed[:],
                        data["transposed"].values,
                    )
                    and np.array_equal(
                        flag[:],
                        data["flag"].values.astype(np.int8),
                    )
                    and list(labels[:]) == ["a", "b", "c"]
                    and nc.getncattr("title") == data.attrs["title"]
                    and variable.getncattr("units") == data["field"].attrs["units"]
                )
        except BaseException as exc:
            options_ok = False
            integrity_note = f"{type(exc).__name__}: {exc}"

    options_ok, integrity_note = mpi.comm.bcast(
        (options_ok, integrity_note),
        root=0,
    )

    progress("checking automatic partition selection with compression disabled")
    uncompressed_path = Path(out_dir) / "climtools_test_parallel_uncompressed.nc"
    xgeo.to_netcdf(
        data[["field"]],
        uncompressed_path,
        unlimited_dim=None,
        parallel=True,
        chunks={"field": (2, 3)},
        zlib=False,
        allow_serial=(SIZE == 1),
    )
    mpi.comm.barrier()

    uncompressed_ok = True
    uncompressed_note = ""
    if RANK == 0:
        try:
            with netCDF4.Dataset(uncompressed_path) as nc:
                variable = nc.variables["field"]
                uncompressed_ok = (
                    not nc.dimensions["time"].isunlimited()
                    and not bool(variable.filters().get("zlib"))
                    and np.array_equal(variable[:], data["field"].values)
                )
        except BaseException as exc:
            uncompressed_ok = False
            uncompressed_note = f"{type(exc).__name__}: {exc}"
    uncompressed_ok, uncompressed_note = mpi.comm.bcast(
        (uncompressed_ok, uncompressed_note),
        root=0,
    )
    if uncompressed_note:
        integrity_note = (
            f"{integrity_note}; {uncompressed_note}"
            if integrity_note
            else uncompressed_note
        )

    correct = bool(
        mpi.reduce.all(
            invalid_partition_ok
            and invalid_compression_ok
            and invalid_unlimited_ok
            and unnamed_ok
            and allow_serial_ok
            and options_ok
            and uncompressed_ok
        )
    )
    progress("recording result")
    record(
        "parallel NetCDF validation/options (chunks/filters/unlimited/attrs)",
        correct,
        0.0,
        0.0,
        note=integrity_note or "correctness-focused",
    )

    progress("cleaning synthetic NetCDF outputs")
    mpi.comm.barrier()
    if RANK == 0:
        for output in (path, uncompressed_path):
            if output.exists():
                output.unlink()
    mpi.comm.barrier()


# ---------------------------------------------------------------------------
# NetCDF write: MPI-collective parallel writer vs ordinary serial writer
# ---------------------------------------------------------------------------


def test_netcdf_write(out_dir: str) -> None:
    """Compare parallel and serial writes of selected variables from the SHiELD file."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        record_skip(
            "NetCDF write (selected real SHiELD variables)",
            "netCDF4 lacks parallel4 support",
        )
        return

    mpi.log("\n--- NetCDF write: selected real SHiELD data, parallel vs serial ---")

    progress("loading selected serial SHiELD source on rank 0")
    full: xr.Dataset | None = None
    error: BaseException | None = None
    if RANK == 0:
        try:
            full = _load_source_dataset(
                ("pr", "t", "slmsk"),
                plev=slice(0, 5),
                lat=slice(0, 128),
                lon=slice(0, 128),
            )
        except BaseException as exc:
            error = exc
    mpi.raise_if_error(error, "load selected real NetCDF source")

    parallel_path = os.path.join(out_dir, "climtools_test_parallel.nc")
    serial_path = os.path.join(out_dir, "climtools_test_serial.nc")

    progress("entering timed parallel section")
    with timed() as box:
        ds = full if RANK == 0 else xgeo.empty_dataset()
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
        if full is None:
            raise AssertionError("Rank 0 did not load the NetCDF source.")
        xgeo.to_netcdf(
            full,
            serial_path,
            unlimited_dim="time",
            show_progress=False,
        )

    progress("running serial baseline")
    _, serial_s = run_serial_baseline(serial_fn)

    progress("validating parallel output against serial output")
    correct = True
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
        except AssertionError as exc:
            correct = False
            integrity_note = str(exc)

    correct, integrity_note = mpi.comm.bcast((correct, integrity_note), root=0)
    progress("recording result")
    record(
        "NetCDF write (selected real SHiELD variables)",
        correct,
        serial_s,
        parallel_s,
        note=integrity_note or "xr.testing.assert_identical",
    )

    progress("cleaning NetCDF comparison outputs")
    mpi.comm.barrier()
    if RANK == 0:
        for path in (serial_path, parallel_path):
            if os.path.exists(path):
                os.remove(path)
    mpi.comm.barrier()


def test_netcdf_distributed_roundtrip(out_dir: str) -> None:
    """Compare distributed and serial writes of selected real SHiELD variables."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        record_skip(
            "distributed NetCDF round-trip (selected real SHiELD variables)",
            "netCDF4 lacks parallel4 support",
        )
        return

    serial_path = os.path.join(out_dir, "climtools_test_distributed_serial.nc")
    parallel_path = os.path.join(out_dir, "climtools_test_distributed_parallel.nc")

    progress("loading serial reference dataset on rank 0")
    serial_data: xr.Dataset | None = None
    error: BaseException | None = None
    if RANK == 0:
        try:
            serial_data = _load_source_dataset(
                ("pr", "t", "slmsk"),
                plev=slice(0, 5),
                lat=slice(0, 128),
                lon=slice(0, 128),
            )
        except BaseException as exc:
            error = exc
    mpi.raise_if_error(error, "load selected serial NetCDF source")

    n_time = None if serial_data is None else int(serial_data.sizes["time"])
    n_time = mpi.comm.bcast(n_time, root=0)

    def serial_fn() -> None:
        if serial_data is None:
            raise AssertionError("Rank 0 did not load the NetCDF source.")
        xgeo.to_netcdf(
            serial_data,
            serial_path,
            unlimited_dim="time",
            show_progress=False,
        )

    progress("running serial baseline")
    _, serial_s = run_serial_baseline(serial_fn)
    serial_data = None
    mpi.comm.barrier()

    progress("opening and loading rank-local distributed dataset")
    distributed: xr.Dataset | None = None
    error = None
    try:
        distributed = mpi.xarray.open_dataset(
            str(DEFAULT_NETCDF_SOURCE),
            partition_dim="time",
        )[["pr", "t", "slmsk"]]
        distributed = distributed.isel(
            plev=slice(0, 5),
            lat=slice(0, 128),
            lon=slice(0, 128),
        )
        distributed.load()
    except BaseException as exc:
        error = exc
    mpi.raise_if_error(error, "open selected distributed NetCDF source")
    if distributed is None:
        raise AssertionError("Distributed Dataset was not created.")

    meta = distributed.attrs.get("mpi_meta")
    local_meta_ok = (
        isinstance(meta, dict)
        and meta.get("dim") == "time"
        and int(meta.get("global_size", -1)) == n_time
    )

    progress("entering timed parallel section")
    with timed() as box:
        xgeo.to_netcdf(
            distributed,
            parallel_path,
            unlimited_dim="time",
            parallel=True,
            allow_serial=(SIZE == 1),
        )
    parallel_s = box["seconds"]
    distributed.close()
    mpi.comm.barrier()

    progress("validating distributed output and internal metadata stripping")
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
    progress("recording result")
    record(
        "distributed NetCDF round-trip (selected real SHiELD variables)",
        correct,
        serial_s,
        parallel_s,
        note=integrity_note or "xr.testing.assert_identical",
    )

    progress("cleaning distributed NetCDF comparison outputs")
    mpi.comm.barrier()
    if RANK == 0:
        for path in (serial_path, parallel_path):
            if os.path.exists(path):
                os.remove(path)
    mpi.comm.barrier()


def test_netcdf_distributed_dataarray(out_dir: str) -> None:
    """Round-trip a selected real SHiELD precipitation DataArray in distributed mode."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        record_skip(
            "distributed NetCDF DataArray round-trip (selected real SHiELD pr)",
            "netCDF4 lacks parallel4 support",
        )
        return

    serial_path = os.path.join(out_dir, "climtools_test_pr_serial.nc")
    parallel_path = os.path.join(out_dir, "climtools_test_pr_parallel.nc")

    progress("loading serial precipitation reference on rank 0")
    serial_pr: xr.DataArray | None = None
    error: BaseException | None = None
    if RANK == 0:
        try:
            serial_pr = _load_source_variable(
                "pr",
                lat=slice(0, 256),
                lon=slice(0, 256),
            )
        except BaseException as exc:
            error = exc
    mpi.raise_if_error(error, "load selected serial precipitation DataArray")

    @mpi(broadcast=True)
    def write_serial_pr() -> None:
        if serial_pr is None:
            raise AssertionError("Rank 0 did not load precipitation.")
        xgeo.to_netcdf(
            serial_pr,
            serial_path,
            unlimited_dim="time",
            show_progress=False,
        )

    write_serial_pr()

    progress("opening distributed precipitation DataArray")
    distributed_ds: xr.Dataset | None = None
    error = None
    try:
        distributed_ds = mpi.xarray.open_dataset(
            str(DEFAULT_NETCDF_SOURCE),
            partition_dim="time",
        )[["pr"]]
        distributed_ds = distributed_ds.isel(
            lat=slice(0, 256),
            lon=slice(0, 256),
        )
        distributed_ds["pr"].load()
    except BaseException as exc:
        error = exc
    mpi.raise_if_error(error, "open selected distributed precipitation DataArray")
    if distributed_ds is None:
        raise AssertionError("Distributed Dataset was not created.")

    distributed_pr = distributed_ds["pr"]
    progress("writing distributed precipitation DataArray")
    xgeo.to_netcdf(
        distributed_pr,
        parallel_path,
        unlimited_dim="time",
        parallel=True,
        allow_serial=(SIZE == 1),
    )
    distributed_ds.close()

    progress("validating distributed DataArray output")
    correct = True
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
        except AssertionError as exc:
            correct = False
            integrity_note = str(exc)

    correct, integrity_note = mpi.comm.bcast((correct, integrity_note), root=0)
    progress("recording result")
    record(
        "distributed NetCDF DataArray round-trip (selected real SHiELD pr)",
        correct,
        0.0,
        0.0,
        note=integrity_note or "correctness-focused",
    )

    progress("cleaning DataArray NetCDF comparison outputs")
    mpi.comm.barrier()
    if RANK == 0:
        for path in (serial_path, parallel_path):
            if os.path.exists(path):
                os.remove(path)
    mpi.comm.barrier()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="climtools MPI test/benchmark suite")
    parser.add_argument(
        "--n-events",
        type=int,
        default=2_000_000,
        help="maximum number of real source values used by reduce/scatter tests",
    )
    parser.add_argument("--grid-ny", type=int, default=180)
    parser.add_argument("--grid-nx", type=int, default=360)
    parser.add_argument(
        "--xarray-events",
        type=int,
        default=5_000,
        help=(
            "maximum real source pressure levels/latitude rows used by mpi.xarray tests"
        ),
    )
    parser.add_argument("--xarray-ny", type=int, default=40)
    parser.add_argument("--xarray-nx", type=int, default=40)
    parser.add_argument("--n-lat", type=int, default=180, help="weighted-mean test")
    parser.add_argument("--n-lon", type=int, default=360)
    parser.add_argument(
        "--out-dir", type=str, default=str(Path.home() / "scratch" / "io_mpi_test")
    )
    parser.add_argument("--skip-netcdf", action="store_true")
    return parser.parse_args()


def print_summary() -> int:
    mpi.log("\n" + "=" * 88)
    mpi.log(f"SUMMARY -- {SIZE} rank(s)")
    mpi.log("=" * 88)
    for result in RESULTS:
        speedup_str = (
            "  n/a " if np.isnan(result.speedup) else f"{result.speedup:5.2f}x"
        )
        status = "SKIP" if result.skipped else ("OK  " if result.correct else "FAIL")
        mpi.log(
            f"[{status}] {result.name:<52} speedup={speedup_str}  "
            + f"serial={result.serial_s:7.4f}s  parallel={result.parallel_s:7.4f}s"
            + (f"  ({result.note})" if result.note else "")
        )
    mpi.log("-" * 88)
    n_fail = sum(1 for result in RESULTS if not result.correct and not result.skipped)
    n_skip = sum(1 for result in RESULTS if result.skipped)
    n_pass = len(RESULTS) - n_fail - n_skip
    mpi.log(
        f"Results: {n_pass} passed, {n_fail} failed, {n_skip} skipped, "
        + f"{len(RESULTS)} recorded checks."
    )
    if n_fail:
        mpi.log(f"{n_fail} recorded check(s) FAILED.")
    elif n_skip:
        mpi.log(
            "All executed checks passed; skipped checks lacked required capability."
        )
    else:
        mpi.log("All recorded checks passed.")
    if SIZE == 1:
        mpi.log(
            "\nRan on 1 rank: speedups will be ~1x or worse. "
            + "mpi.reduce/mpi.xarray/\n"
            + "the parallel NetCDF writer all still pay collective-call overhead "
            + "even\nwith nothing to parallelize against. Run `mpirun -n N python "
            + "climtools_test.py`\nwith N >= 2 real cores to see actual speedups."
        )
    else:
        n_cpus = os.cpu_count() or 1
        if SIZE > n_cpus:
            mpi.log(
                f"\nNote: {SIZE} ranks launched on a machine reporting {n_cpus} CPU(s) "
                + "(os.cpu_count()).\nOversubscribed ranks are time-sliced rather than "
                + "run concurrently, which caps or\ncan even invert the speedups "
                + "above; "
                + "for a clean comparison, run with N <= cores."
            )
    return n_fail


def main() -> None:
    args = parse_args()

    mpi.log("[SETUP] validating SHiELD source visibility", timestamp=True, flush=True)
    try:
        _require_source()
    except BaseException as exc:
        mpi.log(
            f"[SETUP FAILED] {type(exc).__name__}: {exc}",
            timestamp=True,
            flush=True,
        )
        raise SystemExit(1) from exc

    mpi.log("[SETUP] preparing output directory", timestamp=True, flush=True)
    setup_error: BaseException | None = None
    if RANK == 0:
        try:
            os.makedirs(args.out_dir, exist_ok=True)
        except BaseException as exc:
            setup_error = exc
    try:
        mpi.raise_if_error(setup_error, "create output directory")
    except BaseException as exc:
        mpi.log(
            f"[SETUP FAILED] {type(exc).__name__}: {exc}",
            timestamp=True,
            flush=True,
        )
        raise SystemExit(1) from exc
    mpi.comm.barrier()

    mpi.log("=" * 88)
    mpi.log(f"climtools MPI test suite -- {SIZE} rank(s), mpi.launched={mpi.launched}")
    mpi.log(
        "mpi4py initializes MPI on import and finalizes automatically at exit; "
        + "see the mpi4py Overview docs (https://mpi4py.readthedocs.io/en/stable/"
        + "overview.html) for the underlying collective/error-handling semantics "
        + "climtools.mpi builds on."
    )
    mpi.log("=" * 88)

    mpi.log("\n--- mpi runtime helpers ---")
    safe_run(test_runtime_helpers)
    safe_run(test_runtime_logging_and_errors)
    safe_run(test_runtime_contracts)

    mpi.log("\n--- mpi.reduce ---")
    safe_run(test_reduce_sum_scalar, args.n_events)
    safe_run(test_reduce_composite, args.n_events, args.grid_ny, args.grid_nx)
    safe_run(test_reduce_xarray_object, args.grid_ny, args.grid_nx)
    safe_run(test_reduce_operations)

    mpi.log("\n--- mpi.xarray ---")
    safe_run(test_xarray_open_dataset)
    safe_run(test_xarray_redistribute, args.xarray_ny, args.xarray_nx)
    safe_run(test_xarray_isel, args.xarray_ny, args.xarray_nx)
    safe_run(test_xarray_sel, args.xarray_ny, args.xarray_nx)
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
            args.xarray_nx,
            op_name,
        )
    safe_run(test_xarray_dataset_reduction, args.xarray_ny, args.xarray_nx)
    safe_run(test_xarray_contracts)
    safe_run(
        test_xarray_redistribute_on,
        args.xarray_events,
        args.xarray_nx,
    )

    mpi.log("\n--- mpi.scatterv ---")
    safe_run(test_scatterv, args.n_events)
    safe_run(test_scatterv_contracts)

    mpi.log("\n--- xarray operations + mpi.reduce ---")
    safe_run(test_weighted_mean, args.n_lat, args.n_lon)

    safe_run(test_mpi_decorator)

    mpi.log("\n--- xgeo NetCDF interface ---")
    safe_run(test_xgeo_helpers, args.out_dir)
    if not args.skip_netcdf:
        safe_run(test_netcdf_validation_and_options, args.out_dir)
        safe_run(test_netcdf_write, args.out_dir)
        safe_run(
            test_netcdf_distributed_roundtrip,
            args.out_dir,
        )
        safe_run(
            test_netcdf_distributed_dataarray,
            args.out_dir,
        )
    else:
        for name in (
            "parallel NetCDF validation/options",
            "NetCDF write (selected real SHiELD variables)",
            "distributed NetCDF round-trip (selected real SHiELD variables)",
            "distributed NetCDF DataArray round-trip (selected real SHiELD pr)",
        ):
            record_skip(name, "--skip-netcdf requested")

    mpi.comm.barrier()
    failures = print_summary()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
