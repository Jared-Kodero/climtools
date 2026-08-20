"""Correctness and performance tests for :mod:`climtools.mpi` and xgeo.

Performance-oriented checks run the distributed implementation on all active MPI
ranks, run an equivalent serial baseline on rank 0, compare the results, and
report the elapsed-time ratio. Contract tests cover validation, metadata,
nonzero roots, empty partitions, error propagation, and NetCDF behavior.

The suite is self-contained. Rank 0 creates a deterministic NetCDF test dataset
containing precipitation, temperature, sea-land mask, and pressure-level fields.
The only configurable input is the number of mock time steps, which defaults to
3600::

    python climtools_test.py
    mpirun -n 8 python climtools_test.py
    mpirun -n 8 python climtools_test.py --time-steps 7200

Parallel NetCDF checks require netCDF4 with parallel HDF5/NetCDF-C support. When
that capability is unavailable on a multi-rank run, those checks are skipped.
"""

from __future__ import annotations

import argparse
import operator
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from climtools import mpi, xgeo

TIME_STEPS = 100
RESOLUTION_DEG = 0.25
PLEV_STEP = -50


LATITUDE_COUNT: int = 0
LONGITUDE_COUNT: int = 0
PLEV_COUNT: int = 0


OUTPUT_DIR = Path.home() / "scratch" / "io_mpi_test"
TEST_DATA_PATH = OUTPUT_DIR / "mock_in.nc"

RANK: int = mpi.comm.rank
SIZE: int = mpi.comm.size


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def timer(
    function: Callable[..., Any] | None = None,
    *,
    synchronize: bool = True,
) -> Callable[..., Any]:
    """Decorate a function to return ``(result, elapsed_seconds)``.

    Synchronized timing brackets an all-rank operation with barriers and reports
    the slowest rank. ``synchronize=False`` measures only the calling rank, which
    is used for serial baselines executed under ``@mpi(broadcast=True)``.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., tuple[Any, float]]:
        @wraps(func)
        def timed(*args: Any, **kwargs: Any) -> tuple[Any, float]:
            if not synchronize:
                start = mpi.MPI.Wtime()
                result = func(*args, **kwargs)
                return result, mpi.MPI.Wtime() - start

            mpi.comm.barrier()
            start = mpi.MPI.Wtime()
            result: Any = None
            error: BaseException | None = None
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                error = exc

            mpi.raise_if_error(error, func.__name__)
            mpi.comm.barrier()
            elapsed = mpi.reduce.max(mpi.MPI.Wtime() - start)
            return result, elapsed

        return timed

    if function is None:
        return decorate
    return decorate(function)


def build_mock_dataset(
    path: Path,
    n_time_steps: int,
) -> None:

    mpi.log("Creating mock dataset", timestamp=True, prefix=True)

    if isinstance(n_time_steps, bool) or not isinstance(n_time_steps, int):
        raise TypeError("n_time_steps must be an integer")
    if n_time_steps < 1:
        raise ValueError("n_time_steps must be at least 1")

    time = np.arange(n_time_steps, dtype=np.float32)

    # FIXED: Use linspace to guarantee bounds (-90 to 90 inclusive)
    n_lat = int(180 / RESOLUTION_DEG) + 1
    lat = np.linspace(-90, 90, n_lat, dtype=np.float32)

    # FIXED: Use linspace with endpoint=False to exclude 180 and prevent duplication
    n_lon = int(360 / RESOLUTION_DEG)
    lon = np.linspace(-180, 180, n_lon, endpoint=False, dtype=np.float32)

    plev = np.arange(1000.0, -1.0, PLEV_STEP, dtype=np.float32)

    time_phase = (time % 24.0)[:, None, None]
    lat_rad = np.deg2rad(lat)[None, :, None]
    lon_rad = np.deg2rad(lon)[None, None, :]

    precipitation = (
        1.0e-4
        * (1.25 + np.cos(lat_rad) ** 2)
        * (1.0 + 0.15 * np.sin(lon_rad))
        * (1.0 + 0.01 * time_phase)
    ).astype(np.float32)

    surface_temperature = (
        288.0 - 42.0 * np.sin(lat_rad) ** 2 + 2.0 * np.cos(lon_rad) + 0.05 * time_phase
    ).astype(np.float32)

    pressure_cooling = (7.0 * np.log(1000.0 / plev.astype(np.float64))).astype(
        np.float32
    )[None, :, None, None]
    air_temperature = surface_temperature[:, None, :, :] - pressure_cooling

    # FIXED: Extracting lengths directly from the arrays to prevent race conditions
    lat_index = np.arange(len(lat))[:, None]
    lon_index = np.arange(len(lon))[None, :]
    sea_land_mask = ((lat_index + lon_index) % 3).astype(np.int8)

    ds = xr.Dataset(
        data_vars={
            "pr": (
                ("time", "lat", "lon"),
                precipitation,
                {"units": "kg m-2 s-1", "long_name": "precipitation rate"},
            ),
            "t2m": (
                ("time", "lat", "lon"),
                surface_temperature,
                {"units": "K", "long_name": "2 m air temperature"},
            ),
            "t": (
                ("time", "plev", "lat", "lon"),
                air_temperature.astype(np.float32),
                {"units": "K", "long_name": "air temperature"},
            ),
            "slmsk": (
                ("lat", "lon"),
                sea_land_mask,
                {"units": "1", "long_name": "sea-land-ice mask"},
            ),
        },
        coords={
            "time": (
                "time",
                time.astype(np.float64),
                {"units": "hours since 1970-01-01 00:00:00"},
            ),
            "plev": ("plev", plev, {"units": "hPa", "positive": "down"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={"title": "climtools MPI deterministic test dataset"},
    )

    path.unlink(missing_ok=True)
    ds.to_netcdf(path, format="NETCDF4")

    mpi.log(f"Mock dataset saved to {path}", timestamp=True, prefix=True)


def load_configuration(path: Path) -> None:
    """Set the shared shape constants from the written file, on every rank.

    These constants size almost every test, so they must be identical on all
    ranks. Assigning them inside ``build_mock_dataset`` cannot achieve that,
    because only rank 0 builds the dataset: every other rank keeps the
    module-level zeros. The tests then compute different slice bounds per
    rank, which silently corrupts reductions whose buffers still happen to
    match, and deadlocks those whose buffers do not.

    Reading the dimensions back from the file makes it a single source of
    truth that every rank evaluates identically, with no broadcast to keep in
    sync and no ordering hazard beyond the barrier that already guards the
    file's existence.
    """
    global LATITUDE_COUNT
    global LONGITUDE_COUNT
    global PLEV_COUNT
    global TIME_STEPS

    with xr.open_dataset(path) as source:
        LATITUDE_COUNT = int(source.sizes["lat"])
        LONGITUDE_COUNT = int(source.sizes["lon"])
        PLEV_COUNT = int(source.sizes["plev"])
        TIME_STEPS = int(source.sizes["time"])

    # A divergence here would misconfigure every downstream test, so it is
    # checked once, explicitly, rather than being left to surface as a
    # mismatched collective thousands of lines later.
    constants = (LATITUDE_COUNT, LONGITUDE_COUNT, PLEV_COUNT, TIME_STEPS)
    gathered = mpi.comm.allgather(constants)
    if len(set(gathered)) != 1:
        disagreeing = [
            rank for rank, item in enumerate(gathered) if item != gathered[0]
        ]
        raise mpi.MPIError(
            "Test configuration constants differ across ranks. "
            + f"Rank 0 has {gathered[0]}, ranks {disagreeing} disagree "
            + f"(rank {disagreeing[0]} has {gathered[disagreeing[0]]})."
        )


def relative_tolerance_for_dtype(*values: Any, factor: float = 64.0) -> float:
    """Return a relative tolerance derived from the data's own precision.

    Distributed and serial reductions sum the same values in different
    associative orders, so they agree only to the resolution of the dtype
    being reduced. For float32 fields this is 64 * 1.19e-7 ~ 7.6e-6;
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
CURRENT_TEST_NUMBER = 0


def call_raises(
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


def record_result(
    name: str,
    correct: bool,
    serial_s: float,
    parallel_s: float,
    note: str = "",
    *,
    skipped: bool = False,
) -> None:
    """Store and log one correctness or performance result."""
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


def run_test(function: Callable[..., None]) -> Callable[..., None]:
    """Decorate a test with synchronized execution and result accounting."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> None:
        global CURRENT_TEST_NUMBER

        CURRENT_TEST_NUMBER += 1
        test_name = function.__name__
        before = len(RESULTS)
        mpi.log(
            f"[TEST {CURRENT_TEST_NUMBER:02d}] {test_name}: START",
            timestamp=True,
            flush=True,
        )

        error: BaseException | None = None
        try:
            with mpi.watchdog(f"inside {function.__name__}"):
                function(*args, **kwargs)
        except BaseException as exc:
            error = exc

        synchronized_error: BaseException | None = None
        try:
            # Ranks that leave the body early (or run ahead because they
            # posted fewer collectives) block in this all-gather. Leaving it
            # unguarded means those ranks never dump a stack, so the log
            # names only the ranks that stayed behind.
            with mpi.watchdog(f"synchronizing {function.__name__}"):
                mpi.raise_if_error(error, function.__name__)
        except BaseException as exc:
            synchronized_error = exc

        if synchronized_error is not None:
            record_result(
                f"{function.__name__} (uncaught exception)",
                False,
                0.0,
                0.0,
                note=f"{type(synchronized_error).__name__}: {synchronized_error}",
            )
            status = "FAILED"
        elif len(RESULTS) == before:
            record_result(
                f"{function.__name__} (result accounting)",
                False,
                0.0,
                0.0,
                note="test completed without recording a result or skip",
            )
            status = "FAILED: no result recorded"
        else:
            new_results = RESULTS[before:]
            if any(not result.correct and not result.skipped for result in new_results):
                status = "DONE with failed check(s)"
            elif all(result.skipped for result in new_results):
                status = "SKIPPED"
            else:
                status = "DONE"

        mpi.log(
            f"[TEST {CURRENT_TEST_NUMBER:02d}] {test_name}: {status}",
            timestamp=True,
            flush=True,
        )

    return wrapped


def partition_bounds(size: int, rank: int = RANK) -> tuple[int, int]:
    """Return this rank's contiguous bounds within a global dimension."""
    return size * rank // SIZE, size * (rank + 1) // SIZE


def load_test_variable(
    variable: str,
    **indexers: int | slice,
) -> xr.DataArray:
    """Load a requested variable selection from the generated NetCDF dataset.

    Every rank opens the same file concurrently. HDF5 takes POSIX advisory
    locks by default, which can block indefinitely on a parallel filesystem,
    so ``HDF5_USE_FILE_LOCKING=FALSE`` must be set in the environment.
    """
    with xr.open_dataset(TEST_DATA_PATH) as source:
        data = source[variable]
        valid_indexers = {
            dim: indexer for dim, indexer in indexers.items() if dim in data.dims
        }
        return data.isel(valid_indexers).load()


def load_test_dataset(
    variables: tuple[str, ...],
    **indexers: int | slice,
) -> xr.Dataset:
    """Load selected variables and slices from the generated NetCDF dataset."""
    with xr.open_dataset(TEST_DATA_PATH) as source:
        data = source[list(variables)]
        valid_indexers = {
            dim: indexer for dim, indexer in indexers.items() if dim in data.dims
        }
        return data.isel(valid_indexers).load()


# ---------------------------------------------------------------------------
# mpi runtime namespace -- small public helpers
# ---------------------------------------------------------------------------


@run_test
def test_mpi_runtime_helpers() -> None:
    """Check the small MPIRuntime helpers exposed alongside the collectives."""
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
    record_result(
        "mpi runtime helpers (is_root/launched/datatype/MPIError)",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


@run_test
def test_mpi_logging_and_error_propagation() -> None:
    """Check mpi.log behavior and synchronized error propagation."""
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

    no_error_ok = True
    try:
        mpi.raise_if_error(None, "no-error phase")
    except BaseException:
        no_error_ok = False

    all_rank_error_ok = call_raises(
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
    record_result(
        "mpi.log/raise_if_error contracts",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


@run_test
def test_mpi_reduction_contracts() -> None:
    """Exercise reduction validation, non-contiguous buffers, and nonzero roots."""
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

    root = SIZE - 1
    root_result = mpi.reduce.max(float(RANK), mode="root", root=root)
    root_ok = root_result == float(SIZE - 1) if RANK == root else root_result is None

    validation_ok = all(
        (
            call_raises(
                ValueError,
                mpi.reduce.sum,
                1.0,
                mode="invalid",
                contains="mode",
            ),
            call_raises(
                ValueError,
                mpi.reduce.sum,
                1.0,
                mode="root",
                root=-1,
                contains="root",
            ),
            call_raises(
                ValueError,
                mpi.reduce.sum,
                1.0,
                mode="root",
                root=True,
                contains="root",
            ),
            call_raises(
                ValueError,
                mpi.reduce.sum,
                1.0,
                mode="root",
                root=SIZE,
                contains="outside",
            ),
            call_raises(
                mpi.MPIError,
                mpi.reduce.sum,
                np.asarray(["unsupported"], dtype=object),
                contains="Unsupported MPI NumPy dtype",
            ),
            call_raises(
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
    record_result(
        "mpi runtime/reduce contracts (validation/noncontiguous/nonzero root)",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


# ---------------------------------------------------------------------------
# mpi.reduce -- element-wise collective reductions
# ---------------------------------------------------------------------------


@run_test
def test_reduce_scalar_sum(n_total: int) -> None:
    """Scalar mpi.reduce.sum using mock precipitation values."""
    with xr.open_dataset(TEST_DATA_PATH) as source:
        n_lat = int(source.sizes["lat"])
        n_lon = int(source.sizes["lon"])

    n_rows = min(n_lat, max(1, (n_total + n_lon - 1) // n_lon))
    start, stop = partition_bounds(n_rows)
    local = load_test_variable(
        "pr",
        time=0,
        lat=slice(start, stop),
    )

    @timer
    def parallel_sum() -> float:
        local_partial = float(local.sum(skipna=True))
        return mpi.reduce.sum(local_partial)

    combined, parallel_s = parallel_sum()

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> float:
        field = load_test_variable("pr", time=0, lat=slice(0, n_rows))
        return float(field.sum(skipna=True))

    expected, serial_s = serial_fn()
    correct = bool(
        np.isclose(
            combined,
            expected,
            rtol=relative_tolerance_for_dtype(local.values),
            equal_nan=True,
        )
    )
    record_result(
        f"mpi.reduce.sum scalar ({n_rows * n_lon} mock pr values)",
        correct,
        serial_s,
        parallel_s,
    )


@run_test
def test_reduce_array_sum(n_events_total: int, ny: int, nx: int) -> None:
    """mpi.reduce.sum on mock precipitation fields from rank-selected times."""
    with xr.open_dataset(TEST_DATA_PATH) as source:
        n_time = int(source.sizes["time"])
        n_lat = min(int(source.sizes["lat"]), ny)
        n_lon = min(int(source.sizes["lon"]), nx)

    max_points_per_rank = max(1, n_events_total // SIZE)
    if n_lat * n_lon > max_points_per_rank:
        n_lat = max(1, min(n_lat, max_points_per_rank // max(1, n_lon)))

    def load_rank_field(rank: int) -> np.ndarray:
        field = load_test_variable(
            "pr",
            time=rank % n_time,
            lat=slice(0, n_lat),
            lon=slice(0, n_lon),
        )
        return np.asarray(field.values)

    local = load_rank_field(RANK)

    @timer
    def parallel_sum() -> Any:
        return mpi.reduce.sum(local)

    combined, parallel_s = parallel_sum()

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> np.ndarray:
        fields = [load_rank_field(rank) for rank in range(SIZE)]
        return np.sum(np.stack(fields, axis=0), axis=0)

    expected, serial_s = serial_fn()
    correct = (
        bool(
            np.allclose(
                combined,
                expected,
                rtol=relative_tolerance_for_dtype(local),
                equal_nan=True,
            )
        )
        and combined.dtype == local.dtype
    )
    record_result(
        f"mpi.reduce.sum mock pr fields ({n_lat}x{n_lon}, {SIZE} rank selections)",
        correct,
        serial_s,
        parallel_s,
    )


@run_test
def test_reduce_dataarray_sum(ny: int, nx: int) -> None:
    """mpi.reduce.sum on a mock xarray DataArray with metadata preserved."""
    with xr.open_dataset(TEST_DATA_PATH) as source:
        n_time = int(source.sizes["time"])
        n_lat = min(int(source.sizes["lat"]), ny)
        n_lon = min(int(source.sizes["lon"]), nx)

    def load_rank_field(rank: int) -> xr.DataArray:
        return load_test_variable(
            "t2m",
            time=rank % n_time,
            lat=slice(0, n_lat),
            lon=slice(0, n_lon),
        )

    local = load_rank_field(RANK)

    @timer
    def parallel_sum() -> Any:
        return mpi.reduce.sum(local)

    combined, parallel_s = parallel_sum()

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> np.ndarray:
        fields = [load_rank_field(rank).values for rank in range(SIZE)]
        return np.sum(np.stack(fields, axis=0), axis=0)

    expected, serial_s = serial_fn()
    correct = bool(
        np.allclose(
            combined.values,
            expected,
            rtol=relative_tolerance_for_dtype(local.values),
            equal_nan=True,
        )
    ) and combined.attrs.get("units") == local.attrs.get("units")
    record_result(
        "mpi.reduce.sum mock t2m DataArray (dims/attrs kept)",
        correct,
        serial_s,
        parallel_s,
        note="correctness-focused",
    )


@run_test
def test_reduce_all_operations() -> None:
    """Exercise every mpi.reduce operation using mock values."""
    with xr.open_dataset(TEST_DATA_PATH) as source:
        n_time = int(source.sizes["time"])
        n_lat = int(source.sizes["lat"])
        n_lon = int(source.sizes["lon"])

    numeric_width = min(2, n_lon)
    logical_width = min(3, n_lon)

    def load_numeric(rank: int) -> np.ndarray:
        data = load_test_variable(
            "t2m",
            time=rank % n_time,
            lat=rank % n_lat,
            lon=slice(0, numeric_width),
        )
        return np.asarray(data.values)

    def load_logical(rank: int) -> np.ndarray:
        mask = load_test_variable(
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

        @timer
        def parallel_reduce(op, value) -> Any:
            return op(value)

        result, parallel_s = parallel_reduce(op, value)

        root_result = op(value, mode="root", root=0)
        tolerance = relative_tolerance_for_dtype(expected)
        all_mode_ok = bool(
            np.allclose(result, expected, rtol=tolerance, equal_nan=True)
        )
        root_mode_ok = (
            bool(np.allclose(root_result, expected, rtol=tolerance, equal_nan=True))
            if RANK == 0
            else root_result is None
        )
        correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
        record_result(
            f"mpi.reduce.{op_name} mock values (all/root modes)",
            correct,
            0.0,
            parallel_s,
            note="correctness-focused",
        )

    mask = load_test_variable(
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
            load_test_variable(
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
            load_test_variable(
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
    record_result(
        "mpi.reduce.any/all mock slmsk Dataset",
        bool(mpi.reduce.all(dataset_ok)),
        0.0,
        0.0,
        note="correctness-focused",
    )


# ---------------------------------------------------------------------------
# mpi.xarray -- distributed xarray operations
# ---------------------------------------------------------------------------


@run_test
def test_distributed_open_dataset() -> None:
    """mpi.xarray.open_dataset on the mock file partitioned by latitude."""

    @timer
    def open_distributed_dataset() -> xr.Dataset:
        distributed = mpi.xarray.open_dataset(
            str(TEST_DATA_PATH),
            partition_dim="lat",
        )[["pr"]]
        distributed["pr"].isel(time=0).load()
        return distributed

    distributed, parallel_s = open_distributed_dataset()

    # The partition layout is now reported by mpi.xarray.open_dataset itself,
    # so the suite no longer prints it here.
    meta = distributed.attrs.get("mpi_meta")

    local = distributed["pr"].isel(time=0).values.copy()
    variable_meta = distributed["pr"].attrs.get("mpi_meta")
    n_lat = int(meta.get("global_size", -1)) if isinstance(meta, dict) else -1
    local_lat_axis = distributed["pr"].isel(time=0).get_axis_num("lat")
    distributed.close()

    parts = mpi.comm.allgather(local)
    assembled = np.concatenate(parts, axis=local_lat_axis)
    expected = load_test_variable("pr", time=0).values
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

    open_validation_ok = call_raises(
        (ValueError, mpi.MPIError),
        mpi.xarray.open_dataset,
        str(TEST_DATA_PATH),
        partition_dim="missing",
        contains="partition_dim",
    )
    correct = correct and bool(mpi.reduce.all(open_validation_ok))

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> float:
        field = load_test_variable("pr", time=0)
        return float(field.sum(skipna=True))

    _, serial_s = serial_fn()
    record_result(
        "mpi.xarray.open_dataset (mock pr, partitioned latitude)",
        correct,
        serial_s,
        parallel_s,
    )


@run_test
def test_distributed_redistribution(ny: int, nx: int) -> None:
    """mpi.xarray.redistribute using a mock precipitation field."""
    full = load_test_variable(
        "pr",
        time=0,
        lat=slice(0, ny),
        lon=slice(0, nx),
    )

    @timer
    def redistribute_latitude() -> xr.DataArray:
        return mpi.xarray.redistribute(full, "lat")

    distributed, parallel_s = redistribute_latitude()
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
    record_result(
        "mpi.xarray.redistribute mock pr (explicit/auto)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


@run_test
def test_distributed_isel(ny: int, nx: int) -> None:
    """mpi.xarray.isel using global latitude indices on mock precipitation."""
    full = load_test_variable(
        "pr",
        time=0,
        lat=slice(0, ny),
        lon=slice(0, nx),
    )
    distributed = mpi.xarray.redistribute(full, "lat")
    n_lat = full.sizes["lat"]
    if n_lat < 3:
        raise ValueError(
            "The selected mock latitude range must contain at least 3 rows."
        )

    start = 1
    stop = n_lat - 1
    scalar_index = n_lat // 2

    @timer
    def select_global_indices() -> tuple[xr.DataArray, xr.DataArray]:
        sliced = mpi.xarray.isel(distributed, lat=slice(start, stop))
        scalar = mpi.xarray.isel(distributed, lat=scalar_index)
        return sliced, scalar

    (sliced, scalar), parallel_s = select_global_indices()

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
    record_result(
        "mpi.xarray.isel mock pr (global latitude slice/scalar)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


@run_test
def test_distributed_sel(ny: int, nx: int) -> None:
    """mpi.xarray.sel using mock latitude coordinate labels."""
    full = load_test_variable(
        "pr",
        time=0,
        lat=slice(0, ny),
        lon=slice(0, nx),
    )
    distributed = mpi.xarray.redistribute(full, "lat")
    n_lat = full.sizes["lat"]
    if n_lat < 3:
        raise ValueError(
            "The selected mock latitude range must contain at least 3 rows."
        )

    start_label = full["lat"].values[1].item()
    stop_label = full["lat"].values[-2].item()
    scalar_label = full["lat"].values[n_lat // 2].item()

    @timer
    def select_global_labels() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        sliced = mpi.xarray.sel(distributed, lat=slice(start_label, stop_label))
        scalar = mpi.xarray.sel(distributed, lat=scalar_label)
        nearest = mpi.xarray.sel(
            distributed,
            lat=scalar_label,
            method="nearest",
        )
        return sliced, scalar, nearest

    (sliced, scalar, nearest), parallel_s = select_global_labels()

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
    record_result(
        "mpi.xarray.sel mock pr (global latitude slice/scalar)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


@run_test
def test_distributed_numeric_reduction(
    n_levels_max: int,
    ny: int,
    nx: int,
    op_name: str,
) -> None:
    """Numeric mpi.xarray reductions using the mock temperature profile."""
    full = load_test_variable(
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

    @timer
    def reduce_pressure_levels() -> Any:
        return op(distributed, dim="plev", **kwargs)

    result, parallel_s = reduce_pressure_levels()

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> np.ndarray:
        serial_kwargs = {"skipna": True}
        if op_name in {"sum", "prod"}:
            serial_kwargs["min_count"] = 1
        return getattr(full, op_name)(dim="plev", **serial_kwargs).values

    expected, serial_s = serial_fn()
    root_result = op(distributed, dim="plev", mode="root", root=0, **kwargs)
    tolerance = relative_tolerance_for_dtype(expected)
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
    record_result(
        f"mpi.xarray.{op_name} mock t over {full.sizes['plev']} pressure levels",
        correct,
        serial_s,
        parallel_s,
    )


@run_test
def test_distributed_logical_reduction(
    n_lat_max: int,
    nx: int,
    op_name: str,
) -> None:
    """Logical mpi.xarray.any/all using the mock sea-land-ice mask."""
    mask = load_test_variable(
        "slmsk",
        time=0,
        lat=slice(0, n_lat_max),
        lon=slice(0, nx),
    )
    full = mask == 1
    distributed = mpi.xarray.redistribute(full, "lat")
    op = getattr(mpi.xarray, op_name)

    @timer
    def reduce_latitudes() -> Any:
        return op(distributed, dim="lat")

    result, parallel_s = reduce_latitudes()

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> np.ndarray:
        return getattr(full, op_name)(dim="lat").values

    expected, serial_s = serial_fn()
    root_result = op(distributed, dim="lat", mode="root", root=0)
    all_mode_ok = result is not None and bool(np.array_equal(result.values, expected))
    root_mode_ok = (
        root_result is not None and bool(np.array_equal(root_result.values, expected))
        if RANK == 0
        else root_result is None
    )
    correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
    record_result(
        f"mpi.xarray.{op_name} mock slmsk land mask ({full.sizes['lat']} latitudes)",
        correct,
        serial_s,
        parallel_s,
    )


@run_test
def test_distributed_dataset_reduction(ny: int, nx: int) -> None:
    """Dataset reductions using mock distributed t2m plus static plev values."""
    t2m = load_test_variable(
        "t2m",
        time=0,
        lat=slice(0, ny),
        lon=slice(0, nx),
    )
    plev_values = load_test_variable("plev").rename("plev_values")
    full = xr.merge([t2m.to_dataset(name="t2m"), plev_values.to_dataset()])
    distributed = mpi.xarray.redistribute(full, "lat")

    @timer
    def reduce_dataset() -> tuple[xr.Dataset, xr.Dataset]:
        result = mpi.xarray.sum(distributed, dim="lat")
        mean_result = mpi.xarray.mean(distributed, dim=("lat", "lon"))
        return result, mean_result

    (result, mean_result), parallel_s = reduce_dataset()

    expected = full.sum(dim="lat")
    # bottleneck, when installed, silently promotes a Dataset-level
    # reduction that collapses every remaining dimension to float64 even
    # though the equivalent DataArray-level reduction stays float32; the
    # dim="lat" sum above never fully collapses t2m and is unaffected, but
    # dim=("lat", "lon") here reduces it to a scalar, so the reference is
    # computed with bottleneck disabled to get the stable, dtype-preserving
    # answer that mpi.xarray.mean (which never routes through bottleneck's
    # Dataset path) is expected to match.
    with xr.set_options(use_bottleneck=False):
        expected_mean = full.mean(dim=("lat", "lon"))
    correct = (
        result is not None
        and mean_result is not None
        and result["t2m"].dtype == expected["t2m"].dtype
        and bool(
            np.allclose(
                result["t2m"].values,
                expected["t2m"].values,
                rtol=relative_tolerance_for_dtype(expected["t2m"].values),
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
                rtol=relative_tolerance_for_dtype(expected_mean["t2m"].values),
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

    profile = load_test_variable(
        "t",
        time=0,
        plev=slice(0, max(1, SIZE - 1)),
        lat=0,
        lon=0,
    )
    profile_distributed = mpi.xarray.redistribute(profile, "plev")
    minimum = mpi.xarray.min(profile_distributed, dim="plev")
    maximum = mpi.xarray.max(profile_distributed, dim="plev")
    correct = (
        correct
        and minimum is not None
        and maximum is not None
        and bool(
            np.isclose(
                float(minimum.item()),
                float(profile.min(skipna=True).item()),
                rtol=relative_tolerance_for_dtype(profile.values),
            )
        )
        and bool(
            np.isclose(
                float(maximum.item()),
                float(profile.max(skipna=True).item()),
                rtol=relative_tolerance_for_dtype(profile.values),
            )
        )
    )
    correct = bool(mpi.reduce.all(correct))
    record_result(
        "mpi.xarray Dataset reductions (mock distributed/static variables)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


@run_test
def test_distributed_xarray_contracts() -> None:
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

    root = SIZE - 1
    root_sum = mpi.xarray.sum(distributed, dim="lat", mode="root", root=root)
    root_ok = (
        root_sum is not None
        and bool(np.array_equal(root_sum.values, full.sum(dim="lat").values))
        if RANK == root
        else root_sum is None
    )

    complex_full = xr.DataArray(
        np.arange(max(1, SIZE - 1), dtype=np.float64) + 1.0j,
        dims=("sample",),
        name="complex_field",
    )
    complex_distributed = mpi.xarray.redistribute(complex_full, "sample")
    validation_ok = all(
        (
            call_raises(
                ValueError,
                mpi.xarray.redistribute,
                distributed,
                "lon",
                contains="already distributed",
            ),
            call_raises(
                ValueError,
                mpi.xarray.redistribute,
                full,
                "missing",
                contains="does not exist",
            ),
            call_raises(
                ValueError,
                mpi.xarray.sum,
                distributed,
                dim="lon",
                contains="Distributed dimension",
            ),
            call_raises(
                ValueError,
                mpi.xarray.sum,
                distributed,
                dim="lat",
                mode="invalid",
                contains="mode",
            ),
            call_raises(
                ValueError,
                mpi.xarray.sum,
                distributed,
                dim="lat",
                mode="root",
                root=SIZE,
                contains="outside",
            ),
            call_raises(
                NotImplementedError,
                mpi.xarray.isel,
                distributed,
                lat=slice(None, None, 2),
                contains="step 1",
            ),
            call_raises(
                NotImplementedError,
                mpi.xarray.isel,
                distributed,
                lat=[0, 1],
                contains="slices and scalar",
            ),
            call_raises(
                IndexError,
                mpi.xarray.isel,
                distributed,
                lat=full.sizes["lat"],
                contains="out of bounds",
            ),
            call_raises(
                NotImplementedError,
                mpi.xarray.sel,
                distributed,
                lat=[-60.0, 0.0],
                contains="slices and scalar",
            ),
            call_raises(
                KeyError,
                mpi.xarray.sel,
                distributed,
                lat=999.0,
                contains="No rank contains label",
            ),
            call_raises(
                KeyError,
                mpi.xarray.sel,
                distributed,
                lat=-20.0,
                method="nearest",
                tolerance=1.0,
            ),
            call_raises(
                TypeError,
                mpi.xarray.sum,
                np.arange(3),
                dim="x",
                contains="require an xarray",
            ),
            call_raises(
                TypeError,
                mpi.xarray.min,
                complex_distributed,
                dim="sample",
                contains="minimum",
            ),
            call_raises(
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
    record_result(
        "mpi.xarray contracts (edge reductions/indexing/validation/nonzero root)",
        correct,
        0.0,
        0.0,
        note="deterministic edge cases",
    )


@run_test
def test_distributed_arithmetic() -> None:
    """mpi.xarray.apply/align/evaluate: locality, compatibility checks, precedence."""
    a_full = xr.DataArray(
        np.arange(30, dtype=np.float32).reshape(5, 6),
        dims=("plev", "lat"),
        name="a",
    )
    b_full = xr.DataArray(
        (np.arange(30, dtype=np.float32).reshape(5, 6) * 0.5 + 1.0),
        dims=("plev", "lat"),
        name="b",
    )
    a = mpi.xarray.redistribute(a_full, "plev")
    b = mpi.xarray.redistribute(b_full, "plev")

    def local_slice(full: xr.DataArray) -> np.ndarray:
        meta = a.attrs["mpi_meta"]
        return full.isel(plev=slice(meta["start"], meta["stop"])).values

    # -- apply: matching distributions run locally and tag the result -----
    added = mpi.xarray.apply(a, "+", b)
    subtracted = mpi.xarray.apply(a, operator.sub, b)
    apply_ok = (
        bool(np.array_equal(added.values, local_slice(a_full) + local_slice(b_full)))
        and bool(
            np.array_equal(subtracted.values, local_slice(a_full) - local_slice(b_full))
        )
        and added.attrs.get("mpi_meta") == a.attrs.get("mpi_meta")
        and subtracted.attrs.get("mpi_meta") == a.attrs.get("mpi_meta")
    )

    # -- apply: distributed against a scalar, and against a replicated ----
    # -- array that does not carry the distributed dimension --------------
    scaled = mpi.xarray.apply(a, "*", 2.0)
    weights = xr.DataArray(np.arange(6, dtype=np.float32) + 1.0, dims=("lat",))
    weighted = mpi.xarray.apply(a, "*", weights)
    broadcast_ok = bool(
        np.array_equal(scaled.values, local_slice(a_full) * 2.0)
    ) and bool(np.array_equal(weighted.values, local_slice(a_full) * weights.values))

    # -- apply: mismatched partitions must raise, not silently combine ----
    b_by_lat = mpi.xarray.redistribute(b_full, "lat")
    mismatch_ok = call_raises(
        ValueError,
        mpi.xarray.apply,
        a,
        "+",
        b_by_lat,
        contains="different partitions",
    )

    # -- align: replicated operand sliced onto a distributed partner ------
    aligned_a, aligned_b_full = mpi.xarray.align(a, b_full)
    align_slice_ok = (
        aligned_a is a
        and bool(np.array_equal(aligned_b_full.values, local_slice(b_full)))
        and aligned_b_full.attrs.get("mpi_meta") == a.attrs.get("mpi_meta")
    )

    # -- align: two replicated operands jointly distributed along dim -----
    aligned_c, aligned_d = mpi.xarray.align(a_full, b_full, dim="plev")
    align_join_ok = (
        aligned_c.attrs.get("mpi_meta") is not None
        and aligned_c.attrs.get("mpi_meta") == aligned_d.attrs.get("mpi_meta")
        and bool(np.array_equal(aligned_c.values, local_slice(a_full)))
        and bool(np.array_equal(aligned_d.values, local_slice(b_full)))
    )

    # -- align: already-identical distributions and neither-distributed ---
    # -- with no dim both return their inputs unchanged --------------------
    same_a, same_b = mpi.xarray.align(a, b)
    noop_a, noop_b = mpi.xarray.align(a_full, b_full)
    align_noop_ok = (
        same_a is a and same_b is b and noop_a is a_full and noop_b is b_full
    )

    # -- align: genuinely different partitions cannot be reconciled -------
    align_mismatch_ok = call_raises(
        ValueError,
        mpi.xarray.align,
        a,
        b_by_lat,
        contains="different",
    )

    align_ok = align_slice_ok and align_join_ok and align_noop_ok and align_mismatch_ok

    # -- evaluate: real operator precedence, not left-to-right -------------
    precedence_ok = bool(
        np.array_equal(
            mpi.xarray.evaluate("a + b * a", a=a, b=b).values,
            local_slice(a_full) + local_slice(b_full) * local_slice(a_full),
        )
    )
    parens_ok = bool(
        np.array_equal(
            mpi.xarray.evaluate("(a + b) * a", a=a, b=b).values,
            (local_slice(a_full) + local_slice(b_full)) * local_slice(a_full),
        )
    )
    unary_ok = bool(
        np.array_equal(
            mpi.xarray.evaluate("-a + b", a=a, b=b).values,
            -local_slice(a_full) + local_slice(b_full),
        )
    )
    literal_ok = bool(
        np.array_equal(
            mpi.xarray.evaluate("a * 2 + 1", a=a).values,
            local_slice(a_full) * 2 + 1,
        )
    )
    comparison_ok = bool(
        np.array_equal(
            mpi.xarray.evaluate("a > b", a=a, b=b).values,
            local_slice(a_full) > local_slice(b_full),
        )
    )
    chained_reduction = mpi.xarray.sum(
        mpi.xarray.evaluate("(a + b) - a", a=a, b=b), dim="plev", skipna=True
    )
    expected_chained = b_full.sum(dim="plev", skipna=True)
    chained_ok = bool(
        np.allclose(
            chained_reduction.values,
            expected_chained.values,
            rtol=relative_tolerance_for_dtype(expected_chained.values),
        )
    )
    unsafe_rejected = all(
        call_raises((ValueError, NameError), mpi.xarray.evaluate, expression, a=a)
        for expression in ("a.attrs", "a[0]", "__import__('os')", "a + z")
    )

    evaluate_ok = (
        precedence_ok
        and parens_ok
        and unary_ok
        and literal_ok
        and comparison_ok
        and chained_ok
        and unsafe_rejected
    )

    correct = bool(
        mpi.reduce.all(
            apply_ok and broadcast_ok and mismatch_ok and align_ok and evaluate_ok
        )
    )
    record_result(
        "mpi.xarray apply/align/evaluate (locality, compatibility, precedence)",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


@run_test
def test_reduction_redistribution(n_lat_max: int, nx: int) -> None:
    """Redistribute a mock-data reduction result along longitude."""
    full = load_test_variable(
        "t2m",
        time=0,
        lat=slice(0, n_lat_max),
        lon=slice(0, nx),
    )
    distributed = mpi.xarray.redistribute(full, "lat")

    @timer
    def reduce_and_redistribute() -> xr.DataArray:
        return mpi.xarray.mean(distributed, dim="lat", redistribute_on="lon")

    result, parallel_s = reduce_and_redistribute()

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

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> np.ndarray:
        return full.mean(dim="lat").values

    expected, serial_s = serial_fn()
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
                rtol=relative_tolerance_for_dtype(expected),
                equal_nan=True,
            )
        )
        and bool(
            np.allclose(
                auto_assembled,
                expected,
                rtol=relative_tolerance_for_dtype(expected),
                equal_nan=True,
            )
        )
        and bool(mpi.reduce.all(local_meta_ok))
    )
    record_result(
        "mpi.xarray.mean(mock t2m, redistribute_on='lon')",
        correct,
        serial_s,
        parallel_s,
    )


# ---------------------------------------------------------------------------
# mpi.scatterv -- vector scatter (data movement, not a compute reduction)
# ---------------------------------------------------------------------------


@run_test
def test_scatterv_rows(n_total: int) -> None:
    """Scatter rows from a mock t2m field."""
    with xr.open_dataset(TEST_DATA_PATH) as source_ds:
        n_lat = int(source_ds.sizes["lat"])
        n_lon = min(3, int(source_ds.sizes["lon"]))
        dtype = np.dtype(source_ds["t2m"].dtype)

    total = min(n_lat, max(1, n_total // max(1, n_lon)))
    counts = [total // SIZE + (1 if rank < total % SIZE else 0) for rank in range(SIZE)]

    @timer
    def scatter_rows() -> np.ndarray:
        source = None
        if RANK == 0:
            source = load_test_variable(
                "t2m",
                time=0,
                lat=slice(0, total),
                lon=slice(0, n_lon),
            ).values
        return mpi.scatterv(source, counts, (counts[RANK], n_lon), dtype, root=0)

    recv, parallel_s = scatter_rows()

    start = sum(counts[:RANK])
    expected_local = load_test_variable(
        "t2m",
        time=0,
        lat=slice(start, start + counts[RANK]),
        lon=slice(0, n_lon),
    ).values
    correct = bool(np.array_equal(recv, expected_local, equal_nan=True))
    record_result(
        f"mpi.scatterv mock t2m ({total} rows across {SIZE} rank(s))",
        bool(mpi.reduce.all(correct)),
        0.0,
        parallel_s,
        note="data movement, no serial-compute equivalent",
    )


@run_test
def test_scatterv_validation_and_edge_cases() -> None:
    """Exercise scatterv validation, nonzero roots, non-contiguous sends,
    and zero rows.
    """
    invalid_counts_ok = call_raises(
        ValueError,
        mpi.scatterv,
        None,
        [0] * (SIZE + 1),
        (0,),
        np.float64,
        contains="counts",
    )
    unsupported_dtype_ok = call_raises(
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
        missing_root_array_ok = call_raises(
            ValueError,
            mpi.scatterv,
            None,
            [1],
            (1,),
            np.float64,
            contains="cannot be None",
        )

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
    record_result(
        "mpi.scatterv contracts (validation/nonzero root/noncontiguous/zero rows)",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


# ---------------------------------------------------------------------------
# A realistic xarray + mpi.reduce composition: cosine-latitude weighted mean
# ---------------------------------------------------------------------------


@run_test
def test_cosine_weighted_mean(n_lat_total: int, n_lon: int) -> None:
    """Cosine-latitude weighted mean of the mock 2 m temperature field."""
    with xr.open_dataset(TEST_DATA_PATH) as source:
        n_lat = min(int(source.sizes["lat"]), n_lat_total)
        n_lon_used = min(int(source.sizes["lon"]), n_lon)

    start, stop = partition_bounds(n_lat)
    local = load_test_variable(
        "t2m",
        time=0,
        lat=slice(start, stop),
        lon=slice(0, n_lon_used),
    )

    @timer
    def parallel_weighted_mean() -> float:
        weights = np.cos(np.deg2rad(local["lat"]))
        local_weighted_sum = (local * weights).sum(skipna=True)
        local_weight_sum = (xr.ones_like(local) * weights).where(local.notnull()).sum()
        global_weighted_sum = mpi.reduce.sum(float(local_weighted_sum))
        global_weight_sum = mpi.reduce.sum(float(local_weight_sum))
        return global_weighted_sum / global_weight_sum

    weighted_mean, parallel_s = parallel_weighted_mean()

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> float:
        full = load_test_variable(
            "t2m",
            time=0,
            lat=slice(0, n_lat),
            lon=slice(0, n_lon_used),
        )
        weights = np.cos(np.deg2rad(full["lat"]))
        numerator = (full * weights).sum(skipna=True)
        denominator = (xr.ones_like(full) * weights).where(full.notnull()).sum()
        return float(numerator / denominator)

    expected, serial_s = serial_fn()
    correct = bool(
        np.isclose(
            weighted_mean,
            expected,
            rtol=relative_tolerance_for_dtype(local.values),
            equal_nan=True,
        )
    )
    record_result(
        f"cosine-lat weighted mean mock t2m ({n_lat}x{n_lon_used})",
        correct,
        serial_s,
        parallel_s,
    )


# ---------------------------------------------------------------------------
# @mpi decorator -- usage demonstration and correctness checks
# ---------------------------------------------------------------------------


@run_test
def test_mpi_decorator_modes() -> None:
    """Exercise root, all-rank, broadcast, error, and validation modes of @mpi."""
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

    @mpi(broadcast=True, root=SIZE - 1)
    def nonzero_root_setup() -> int:
        return RANK

    nonzero_root_result = nonzero_root_setup()
    ok_nonzero_root = nonzero_root_result == SIZE - 1
    invalid_root_function = mpi(lambda: None, root=SIZE)
    ok_validation = all(
        (
            call_raises(
                ValueError,
                mpi,
                lambda: None,
                all_ranks=True,
                broadcast=True,
                contains="incompatible",
            ),
            call_raises(
                TypeError,
                mpi,
                42,
                contains="must be callable",
            ),
            call_raises(
                ValueError,
                mpi,
                lambda: None,
                root=-1,
                contains="non-negative",
            ),
            call_raises(
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
    record_result(
        "@mpi decorator (root/all_ranks/broadcast/error/validation)",
        overall,
        0.0,
        0.0,
        note="usage demo, not a speed test",
    )


@run_test
def test_xgeo_interface_contracts(out_dir: str) -> None:
    """Check public xgeo placeholder detection and front-door validation."""
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

    invalid_type_ok = call_raises(
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
    mismatch_ok = call_raises(
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
    record_result(
        "xgeo helpers/to_netcdf front-door validation",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


@run_test
def test_parallel_netcdf_writer_options(out_dir: str) -> None:
    """Check parallel NetCDF validation plus explicit chunk/filter/unlimited options."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        record_result(
            "parallel NetCDF validation/options",
            True,
            0.0,
            0.0,
            note="netCDF4 lacks parallel4 support",
            skipped=True,
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

    mpi.log(
        "checking synchronized NetCDF preparation validation",
        timestamp=True,
        flush=True,
    )
    invalid_partition_ok = call_raises(
        expected_error,
        xgeo.to_netcdf,
        data,
        os.path.join(out_dir, "invalid_partition.nc"),
        partition_dim="missing",
        parallel=True,
        allow_serial=(SIZE == 1),
        contains="partition_dim",
    )
    invalid_compression_ok = call_raises(
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
    invalid_unlimited_ok = call_raises(
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
        unnamed_ok = call_raises(
            ValueError,
            xgeo.to_netcdf,
            unnamed,
            os.path.join(out_dir, "unnamed_dataarray.nc"),
            partition_dim="time",
            parallel=True,
            allow_serial=True,
            contains="must have a name",
        )
        allow_serial_ok = call_raises(
            mpi.MPIError,
            xgeo.to_netcdf,
            data,
            os.path.join(out_dir, "allow_serial_required.nc"),
            partition_dim="time",
            parallel=True,
            allow_serial=False,
            contains="one process",
        )

    mpi.log(
        "writing explicit parallel chunk/compression/unlimited configuration",
        timestamp=True,
        flush=True,
    )
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

    mpi.log(
        "checking automatic partition selection with compression disabled",
        timestamp=True,
        flush=True,
    )
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
    record_result(
        "parallel NetCDF validation/options (chunks/filters/unlimited/attrs)",
        correct,
        0.0,
        0.0,
        note=integrity_note or "correctness-focused",
    )

    mpi.comm.barrier()
    if RANK == 0:
        for output in (path, uncompressed_path):
            if output.exists():
                output.unlink()
    mpi.comm.barrier()


# ---------------------------------------------------------------------------
# NetCDF write: MPI-collective parallel writer vs ordinary serial writer
# ---------------------------------------------------------------------------


@run_test
def test_parallel_netcdf_write(out_dir: str) -> None:
    """Compare parallel and serial writes of selected variables from the mock file."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        record_result(
            "NetCDF write (selected mock variables)",
            True,
            0.0,
            0.0,
            note="netCDF4 lacks parallel4 support",
            skipped=True,
        )
        return

    mpi.log("\n--- NetCDF write: selected mock data, parallel vs serial ---")

    mpi.log(
        "loading selected serial mock source on rank 0",
        timestamp=True,
        flush=True,
    )
    full: xr.Dataset | None = None
    error: BaseException | None = None
    if RANK == 0:
        try:
            full = load_test_dataset(
                ("pr", "t", "slmsk"),
                plev=slice(0, 5),
                lat=slice(0, 128),
                lon=slice(0, 128),
            )
        except BaseException as exc:
            error = exc
    mpi.raise_if_error(error, "load selected mock NetCDF source")

    parallel_path = os.path.join(out_dir, "climtools_test_parallel.nc")
    serial_path = os.path.join(out_dir, "climtools_test_serial.nc")

    @timer
    def write_parallel_dataset() -> None:
        ds = full if RANK == 0 else xgeo.empty_dataset()
        xgeo.to_netcdf(
            ds,
            parallel_path,
            unlimited_dim="time",
            partition_dim="time",
            parallel=True,
            allow_serial=(SIZE == 1),
        )

    _, parallel_s = write_parallel_dataset()

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> None:
        if full is None:
            raise AssertionError("Rank 0 did not load the NetCDF source.")
        xgeo.to_netcdf(
            full,
            serial_path,
            unlimited_dim="time",
            show_progress=False,
        )

    _, serial_s = serial_fn()

    mpi.log(
        "validating parallel output against serial output",
        timestamp=True,
        flush=True,
    )
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
    record_result(
        "NetCDF write (selected mock variables)",
        correct,
        serial_s,
        parallel_s,
        note=integrity_note or "xr.testing.assert_identical",
    )

    mpi.comm.barrier()
    if RANK == 0:
        for path in (serial_path, parallel_path):
            if os.path.exists(path):
                os.remove(path)
    mpi.comm.barrier()


@run_test
def test_distributed_netcdf_roundtrip(out_dir: str) -> None:
    """Compare distributed and serial writes of selected mock variables."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        record_result(
            "distributed NetCDF round-trip (selected mock variables)",
            True,
            0.0,
            0.0,
            note="netCDF4 lacks parallel4 support",
            skipped=True,
        )
        return

    serial_path = os.path.join(out_dir, "climtools_test_distributed_serial.nc")
    parallel_path = os.path.join(out_dir, "climtools_test_distributed_parallel.nc")

    mpi.log(
        "loading serial reference dataset on rank 0",
        timestamp=True,
        flush=True,
    )
    serial_data: xr.Dataset | None = None
    error: BaseException | None = None
    if RANK == 0:
        try:
            serial_data = load_test_dataset(
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

    @mpi(broadcast=True)
    @timer(synchronize=False)
    def serial_fn() -> None:
        if serial_data is None:
            raise AssertionError("Rank 0 did not load the NetCDF source.")
        xgeo.to_netcdf(
            serial_data,
            serial_path,
            unlimited_dim="time",
            show_progress=False,
        )

    _, serial_s = serial_fn()
    serial_data = None
    mpi.comm.barrier()

    mpi.log(
        "opening and loading rank-local distributed dataset",
        timestamp=True,
        flush=True,
    )
    distributed: xr.Dataset | None = None
    error = None
    try:
        distributed = mpi.xarray.open_dataset(
            str(TEST_DATA_PATH),
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

    @timer
    def write_distributed_dataset() -> None:
        xgeo.to_netcdf(
            distributed,
            parallel_path,
            unlimited_dim="time",
            parallel=True,
            allow_serial=(SIZE == 1),
        )

    _, parallel_s = write_distributed_dataset()
    distributed.close()
    mpi.comm.barrier()

    mpi.log(
        "validating distributed output and internal metadata stripping",
        timestamp=True,
        flush=True,
    )
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
    record_result(
        "distributed NetCDF round-trip (selected mock variables)",
        correct,
        serial_s,
        parallel_s,
        note=integrity_note or "xr.testing.assert_identical",
    )

    mpi.comm.barrier()
    if RANK == 0:
        for path in (serial_path, parallel_path):
            if os.path.exists(path):
                os.remove(path)
    mpi.comm.barrier()


@run_test
def test_distributed_dataarray_roundtrip(out_dir: str) -> None:
    """Round-trip a selected mock precipitation DataArray in distributed mode."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        record_result(
            "distributed NetCDF DataArray round-trip (selected mock pr)",
            True,
            0.0,
            0.0,
            note="netCDF4 lacks parallel4 support",
            skipped=True,
        )
        return

    serial_path = os.path.join(out_dir, "climtools_test_pr_serial.nc")
    parallel_path = os.path.join(out_dir, "climtools_test_pr_parallel.nc")

    mpi.log(
        "loading serial precipitation reference on rank 0",
        timestamp=True,
        flush=True,
    )
    serial_pr: xr.DataArray | None = None
    error: BaseException | None = None
    if RANK == 0:
        try:
            serial_pr = load_test_variable(
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

    mpi.log(
        "opening distributed precipitation DataArray",
        timestamp=True,
        flush=True,
    )
    distributed_ds: xr.Dataset | None = None
    error = None
    try:
        distributed_ds = mpi.xarray.open_dataset(
            str(TEST_DATA_PATH),
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
    mpi.log(
        "writing distributed precipitation DataArray",
        timestamp=True,
        flush=True,
    )
    xgeo.to_netcdf(
        distributed_pr,
        parallel_path,
        unlimited_dim="time",
        parallel=True,
        allow_serial=(SIZE == 1),
    )
    distributed_ds.close()

    mpi.log(
        "validating distributed DataArray output",
        timestamp=True,
        flush=True,
    )
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
    record_result(
        "distributed NetCDF DataArray round-trip (selected mock pr)",
        correct,
        0.0,
        0.0,
        note=integrity_note or "correctness-focused",
    )

    mpi.comm.barrier()
    if RANK == 0:
        for path in (serial_path, parallel_path):
            if os.path.exists(path):
                os.remove(path)
    mpi.comm.barrier()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data placement: rank-0-only sources, distributed sources, distributed open
# ---------------------------------------------------------------------------


def _reference_reduction(field: xr.DataArray, op_name: str, dim: str) -> np.ndarray:
    """Return the serial result an mpi.xarray reduction must reproduce."""
    return getattr(field, op_name)(dim=dim).values


@run_test
def test_rank0_source_distribution(ny: int, nx: int) -> None:
    """Reductions over data that initially exists only on rank 0.

    The realistic ingest pattern is that one rank reads or generates a field
    and the others hold nothing. This checks both routes from that state into
    a distributed object: broadcasting the whole field and then partitioning
    it, and scattering rows directly. Both must agree with the serial answer
    and must leave every rank in the same collective sequence.
    """
    source = load_test_variable("t2m", time=0, lat=slice(0, ny), lon=slice(0, nx))

    @mpi(broadcast=True)
    def read_on_root_only() -> xr.DataArray:
        return source

    replicated = read_on_root_only()
    only_root_had_data = bool(mpi.reduce.all(isinstance(replicated, xr.DataArray)))

    distributed = mpi.xarray.redistribute(replicated, "lat")

    @timer
    def reduce_broadcast_source() -> tuple[Any, Any]:
        return (
            mpi.xarray.sum(distributed, dim="lat"),
            mpi.xarray.mean(distributed, dim="lat"),
        )

    (total, average), parallel_s = reduce_broadcast_source()

    # Scatter route: rank 0 owns the rows, every other rank starts empty.
    rows = np.ascontiguousarray(source.values, dtype=np.float64)
    counts = np.asarray(
        [
            partition_bounds(ny, rank)[1] - partition_bounds(ny, rank)[0]
            for rank in range(SIZE)
        ],
        dtype=np.int64,
    )
    local_rows = int(counts[RANK])
    scattered = mpi.scatterv(
        rows if RANK == 0 else None,
        counts,
        (local_rows, nx),
        rows.dtype,
        root=0,
    )
    scattered_total = mpi.reduce.sum(scattered.sum())

    expected_sum = _reference_reduction(source, "sum", "lat")
    expected_mean = _reference_reduction(source, "mean", "lat")
    tolerance = relative_tolerance_for_dtype(expected_sum)
    correct = (
        only_root_had_data
        and total is not None
        and average is not None
        and bool(np.allclose(total.values, expected_sum, rtol=tolerance))
        and bool(np.allclose(average.values, expected_mean, rtol=tolerance))
        and bool(
            np.isclose(
                float(scattered_total),
                float(source.sum()),
                # The rows are widened to float64 for transport, but the
                # values are float32, so the tolerance must come from the
                # source precision rather than from the transport dtype.
                rtol=relative_tolerance_for_dtype(source.values),
            )
        )
    )
    correct = bool(mpi.reduce.all(correct))
    record_result(
        "rank-0-only source (broadcast and scatterv routes into distributed form)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


@run_test
def test_distributed_open_reductions() -> None:
    """Reductions on a Dataset opened directly in distributed mode.

    Unlike redistribute, open_dataset partitions on effective chunk bounds
    and leaves the data lazy, so the rank-local partial is materialized
    inside the reduction itself. Every operation is exercised in both result
    placements against the serial answer.
    """
    distributed = mpi.xarray.open_dataset(
        str(TEST_DATA_PATH),
        partition_dim="lat",
    )[["pr", "slmsk"]].isel(time=0)
    serial = load_test_dataset(("pr", "slmsk"), time=0)

    @timer
    def reduce_opened_dataset() -> dict[str, Any]:
        return {
            "sum": mpi.xarray.sum(distributed, dim="lat"),
            "mean": mpi.xarray.mean(distributed, dim="lat"),
            "min": mpi.xarray.min(distributed, dim="lat"),
            "max": mpi.xarray.max(distributed, dim="lat"),
        }

    results, parallel_s = reduce_opened_dataset()

    checks = []
    for op_name, result in results.items():
        expected = getattr(serial, op_name)(dim="lat")
        checks.append(result is not None)
        if result is None:
            continue
        for variable in ("pr", "slmsk"):
            checks.append(
                bool(
                    np.allclose(
                        result[variable].values,
                        expected[variable].values,
                        rtol=relative_tolerance_for_dtype(expected[variable].values),
                        equal_nan=True,
                    )
                )
            )

    # Root placement returns the result only on the destination rank.
    root_result = mpi.xarray.sum(distributed, dim="lat", mode="root", root=SIZE - 1)
    checks.append(
        (root_result is not None) if RANK == SIZE - 1 else (root_result is None)
    )

    land = mpi.xarray.any(distributed["slmsk"] == 1, dim="lat")
    checks.append(
        land is not None
        and bool(np.array_equal(land.values, (serial["slmsk"] == 1).any(dim="lat")))
    )
    distributed.close()

    correct = bool(mpi.reduce.all(all(checks)))
    record_result(
        "mpi.xarray reductions on a distributed open_dataset (lazy, chunk bounds)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


@run_test
def test_empty_partition_reductions() -> None:
    """Reductions where some ranks own no elements of the partition.

    A dimension shorter than the communicator leaves trailing ranks empty.
    Those ranks build their partial through a different code path from ranks
    holding data, so this is the configuration in which a reduction can post
    a different number or shape of collectives per rank and deadlock. Lengths
    below, at and above the rank count are all covered, in both placements.
    """
    checks: list[bool] = []
    lengths = sorted({1, max(1, SIZE // 2), max(1, SIZE - 1), SIZE, SIZE + 1})
    for length in lengths:
        profile = load_test_variable(
            "t",
            time=0,
            plev=slice(0, length),
            lat=0,
            lon=0,
        )
        if int(profile.sizes["plev"]) != length:
            continue
        distributed = mpi.xarray.redistribute(profile, "plev")
        empty_here = int(distributed.sizes["plev"]) == 0
        checks.append(bool(mpi.reduce.any(empty_here)) or length >= SIZE)

        for op_name in ("sum", "prod", "mean", "min", "max"):
            result = getattr(mpi.xarray, op_name)(distributed, dim="plev")
            expected = getattr(profile, op_name)(dim="plev")
            checks.append(
                result is not None
                and bool(
                    np.isclose(
                        float(result.item()),
                        float(expected.item()),
                        rtol=relative_tolerance_for_dtype(profile.values),
                    )
                )
            )
            root_result = getattr(mpi.xarray, op_name)(
                distributed,
                dim="plev",
                mode="root",
                root=SIZE - 1,
            )
            checks.append(
                (root_result is not None) if RANK == SIZE - 1 else (root_result is None)
            )

        flags = mpi.xarray.redistribute(profile > float(profile.mean()), "plev")
        for op_name in ("any", "all"):
            result = getattr(mpi.xarray, op_name)(flags, dim="plev")
            expected = getattr(profile > float(profile.mean()), op_name)(dim="plev")
            checks.append(
                result is not None and bool(result.item()) == bool(expected.item())
            )

        # Dataset form: one distributed variable beside a static one.
        bundle = xr.Dataset(
            {
                "profile": profile,
                "static": xr.DataArray(np.arange(3, dtype=np.float32), dims=("other",)),
            }
        )
        bundle_distributed = mpi.xarray.redistribute(bundle, "plev")
        for op_name in ("sum", "mean", "min", "max"):
            result = getattr(mpi.xarray, op_name)(bundle_distributed, dim="plev")
            checks.append(
                result is not None
                and bool(
                    np.allclose(
                        result["static"].values,
                        bundle["static"].values,
                    )
                )
            )

    correct = bool(mpi.reduce.all(all(checks)))
    record_result(
        f"empty-partition reductions (plev lengths {lengths} over {SIZE} rank(s))",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


class _RecordingComm:
    """Communicator proxy that records the collective sequence it posts."""

    COLLECTIVES = frozenset(
        {
            "allgather",
            "allreduce",
            "alltoall",
            "barrier",
            "bcast",
            "gather",
            "reduce",
            "scatter",
            "Allgather",
            "Allgatherv",
            "Allreduce",
            "Alltoall",
            "Alltoallv",
            "Barrier",
            "Bcast",
            "Gather",
            "Gatherv",
            "Reduce",
            "Scatter",
            "Scatterv",
        }
    )

    def __init__(self, comm: Any, log: list[str]) -> None:
        self._comm = comm
        self._log = log

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._comm, name)
        if name not in self.COLLECTIVES or not callable(attribute):
            return attribute

        def recorded(*args: Any, **kwargs: Any) -> Any:
            # Only the send buffer is compared. A root-mode Reduce passes
            # None as the receive buffer on every rank except the
            # destination, which is correct usage, not a divergence.
            leading = args[0] if args else None
            parts = [
                f"{leading.shape}:{leading.dtype}"
                if isinstance(leading, np.ndarray)
                else type(leading).__name__
            ]
            if "root" in kwargs:
                parts.append(f"root={kwargs['root']}")
            self._log.append(f"{name}({','.join(parts)})")
            return attribute(*args, **kwargs)

        return recorded

    @property
    def rank(self) -> int:
        return int(self._comm.rank)

    @property
    def size(self) -> int:
        return int(self._comm.size)


@run_test
def test_collective_sequence_symmetry() -> None:
    """Every rank must post an identical sequence of collectives.

    Ranks that post different collectives can still appear to succeed under
    one MPI implementation and deadlock under another, because whether a
    mismatched buffer collective completes depends on the algorithm the
    library selects. Comparing the recorded sequences makes that class of
    defect fail here, deterministically, instead of in a production run.
    """
    real_comm = mpi.comm
    recorded: list[str] = []
    mpi.comm = _RecordingComm(real_comm, recorded)

    divergent: list[str] = []
    try:
        for length in sorted({1, max(1, SIZE - 1), SIZE, 2 * SIZE + 1}):
            profile = load_test_variable(
                "t",
                time=0,
                plev=slice(0, min(length, PLEV_COUNT)),
                lat=slice(0, 2),
                lon=slice(0, 2),
            )
            distributed = mpi.xarray.redistribute(profile, "plev")
            flags = mpi.xarray.redistribute(profile > float(profile.mean()), "plev")
            scenarios: tuple[tuple[str, Any], ...] = (
                ("sum", lambda d=distributed: mpi.xarray.sum(d, dim="plev")),
                ("prod", lambda d=distributed: mpi.xarray.prod(d, dim="plev")),
                ("mean", lambda d=distributed: mpi.xarray.mean(d, dim="plev")),
                ("min", lambda d=distributed: mpi.xarray.min(d, dim="plev")),
                ("max", lambda d=distributed: mpi.xarray.max(d, dim="plev")),
                (
                    "sum(min_count)",
                    lambda d=distributed: mpi.xarray.sum(d, dim="plev", min_count=1),
                ),
                (
                    "min(root)",
                    lambda d=distributed: mpi.xarray.min(
                        d, dim="plev", mode="root", root=SIZE - 1
                    ),
                ),
                ("any", lambda d=flags: mpi.xarray.any(d, dim="plev")),
                ("all", lambda d=flags: mpi.xarray.all(d, dim="plev")),
            )
            for label, scenario in scenarios:
                recorded.clear()
                scenario()
                sequence = tuple(recorded)
                gathered = real_comm.allgather(sequence)
                if len(set(gathered)) != 1:
                    offenders = [
                        rank
                        for rank, item in enumerate(gathered)
                        if item != gathered[0]
                    ]
                    divergent.append(f"{label} (plev={length}) ranks {offenders}")
    finally:
        mpi.comm = real_comm

    correct = bool(mpi.reduce.all(not divergent))
    record_result(
        "identical collective sequence on every rank (all reductions)",
        correct,
        0.0,
        0.0,
        note="; ".join(divergent) if divergent else "correctness-focused",
    )


@run_test
def test_reduction_dtype_contracts() -> None:
    """Reduction buffers must not depend on rank-local data or unusable dtypes."""
    checks: list[bool] = []

    # A dtype with a reducible NumPy kind but no predefined MPI datatype must
    # be rejected identically on every rank, before any collective is posted.
    half = xr.DataArray(np.zeros(SIZE + 1, dtype=np.float16), dims=("lat",))
    half_distributed = mpi.xarray.redistribute(half, "lat")
    checks.append(
        call_raises(
            TypeError,
            mpi.xarray.sum,
            half_distributed,
            dim="lat",
            contains="Unsupported MPI xarray dtype",
        )
    )

    # Empty and non-empty partitions must agree on the reduced dtype. The
    # reference reduction runs with bottleneck disabled: bottleneck, when
    # installed, promotes a full (all-dims) float32 min/max reduction to
    # float64, which would make this reference itself dtype-unstable
    # across environments rather than testing climtools' own contract.
    for dtype in (np.int8, np.int32, np.float32, np.float64):
        field = xr.DataArray(
            np.arange(max(1, SIZE - 1), dtype=dtype),
            dims=("lat",),
        )
        distributed = mpi.xarray.redistribute(field, "lat")
        for op_name in ("min", "max"):
            result = getattr(mpi.xarray, op_name)(distributed, dim="lat")
            with xr.set_options(use_bottleneck=False):
                expected = getattr(field, op_name)(dim="lat")
            checks.append(result is not None and result.dtype == expected.dtype)
            checks.append(
                result is not None and float(result.item()) == float(expected.item())
            )

    correct = bool(mpi.reduce.all(all(checks)))
    record_result(
        "reduction dtype contracts (MPI-representable dtypes, empty partitions)",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


@run_test
def test_shared_configuration_agreement() -> None:
    """Every rank must agree on the constants that size the other tests.

    The shape constants decide slice bounds, buffer sizes and partition
    lengths throughout the suite. If they are derived on one rank only, the
    remaining ranks silently keep their module defaults: reductions whose
    buffers still match return wrong answers, and reductions whose buffers do
    not match deadlock with rank 0 blocked in Allreduce and every other rank
    waiting in the following all-gather. This checks the constants directly,
    and checks that the values they produce are themselves rank-invariant.
    """
    constants = {
        "LATITUDE_COUNT": LATITUDE_COUNT,
        "LONGITUDE_COUNT": LONGITUDE_COUNT,
        "PLEV_COUNT": PLEV_COUNT,
        "TIME_STEPS": TIME_STEPS,
    }
    gathered = mpi.comm.allgather(tuple(sorted(constants.items())))
    agreed = len(set(gathered)) == 1

    # None of them may still hold the module-level sentinel, which is what a
    # rank that never ran the loader would report.
    populated = all(value > 0 for value in constants.values())

    # The constants must also match the file every rank can see.
    with xr.open_dataset(TEST_DATA_PATH) as source:
        matches_file = (
            LATITUDE_COUNT == int(source.sizes["lat"])
            and LONGITUDE_COUNT == int(source.sizes["lon"])
            and PLEV_COUNT == int(source.sizes["plev"])
            and TIME_STEPS == int(source.sizes["time"])
        )

    # A derived buffer shape, computed the way the reduction tests compute
    # theirs, must come out identical on every rank.
    derived = np.empty(
        (max(1, LATITUDE_COUNT // SIZE), LONGITUDE_COUNT), dtype=np.float32
    )
    shapes = mpi.comm.allgather(derived.shape)
    derived_agreed = len(set(shapes)) == 1

    correct = bool(
        mpi.reduce.all(agreed and populated and matches_file and derived_agreed)
    )
    note = "correctness-focused" if correct else f"rank {RANK} has {constants}"
    record_result(
        "shared configuration constants identical on every rank",
        correct,
        0.0,
        0.0,
        note=note,
    )


@run_test
def test_reduce_buffer_agreement() -> None:
    """A mismatched reduction buffer must raise, not deadlock.

    This is the failure mode that a rank-dependent constant produces. Ranks
    posting the shorter buffer can return from the collective while the rest
    block indefinitely, so the guard has to fire before the collective is
    posted rather than detect the problem afterwards.
    """
    if SIZE == 1:
        record_result(
            "mismatched reduction buffers raise instead of deadlocking",
            True,
            0.0,
            0.0,
            note="skipped: needs more than one rank",
        )
        return

    # Rank 0 posts a longer buffer, exactly as a rank-0-only constant would
    # cause. Without the guard this deadlocks.
    mismatched = np.ones(8 if RANK == 0 else 4, dtype=np.float32)
    raised = call_raises(
        mpi.MPIError,
        mpi.reduce.sum,
        mismatched,
        contains="different reduction buffers",
    )

    # A dtype mismatch is equally undefined and equally caught.
    wrong_dtype = np.ones(4, dtype=np.float64 if RANK == 0 else np.float32)
    raised_dtype = call_raises(
        mpi.MPIError,
        mpi.reduce.sum,
        wrong_dtype,
        contains="different reduction buffers",
    )

    # Matching buffers must still reduce normally afterwards.
    matched = np.full(4, float(RANK), dtype=np.float32)
    total = mpi.reduce.sum(matched)
    expected = float(sum(range(SIZE)))
    recovered = bool(np.allclose(total, expected))

    correct = bool(mpi.reduce.all(raised and raised_dtype and recovered))
    record_result(
        "mismatched reduction buffers raise instead of deadlocking",
        correct,
        0.0,
        0.0,
        note="correctness-focused",
    )


def print_test_summary() -> int:
    """Print the final result table and return the number of failed checks."""
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


def parse_arguments() -> argparse.Namespace:
    """Parse the sizing options for the generated test dataset.

    The defaults reproduce the full-resolution cluster configuration. Smaller
    values let the suite run on a workstation, which matters because the
    collective-symmetry and buffer-agreement checks do not depend on
    resolution: they need many ranks, not much data.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time-steps",
        type=int,
        default=TIME_STEPS,
        help=f"time steps in the generated dataset (default {TIME_STEPS})",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=RESOLUTION_DEG,
        help=f"grid spacing in degrees (default {RESOLUTION_DEG})",
    )
    # Parsing happens identically on every rank from the same argv, so the
    # resulting values are rank-invariant by construction.
    return parser.parse_args()


def main() -> None:
    """Create test data and run the complete suite on the active MPI ranks."""
    global TIME_STEPS
    global RESOLUTION_DEG

    arguments = parse_arguments()
    TIME_STEPS = arguments.time_steps
    RESOLUTION_DEG = arguments.resolution

    if RANK == 0:
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        build_mock_dataset(TEST_DATA_PATH, TIME_STEPS)
    # ADD A BARRIER HERE
    # Wait for Rank 0 to finish writing the file before other ranks proceed.
    # (Adjust the syntax below depending on the specific MPI wrapper you are using)
    mpi.comm.Barrier()

    mpi.log("[SETUP] validating test dataset visibility", timestamp=True, flush=True)
    visible = TEST_DATA_PATH.is_file()

    if not bool(mpi.reduce.all(visible)):
        mpi.log(
            "[SETUP FAILED] test dataset is not visible on every rank: "
            + f"{TEST_DATA_PATH}",
            timestamp=True,
            flush=True,
        )
        raise SystemExit(1)

    mpi.comm.barrier()

    # Every rank derives the shape constants from the file it can now see.
    load_configuration(TEST_DATA_PATH)

    mpi.log(
        "Configuration Constants:\n"
        + f"  TIME_STEPS:      {TIME_STEPS}\n"
        + f"  RESOLUTION_DEG:  {RESOLUTION_DEG}\n"
        + f"  PLEV_STEP:       {PLEV_STEP}\n"
        + f"  LATITUDE_COUNT:  {LATITUDE_COUNT}\n"
        + f"  LONGITUDE_COUNT: {LONGITUDE_COUNT}\n"
        + f"  PLEV_COUNT:      {PLEV_COUNT}\n"
        + f"  TEST_DATA_PATH:  {TEST_DATA_PATH}"
    )

    mpi.log("=" * 88)
    mpi.log(f"climtools MPI test suite -- {SIZE} rank(s), mpi.launched={mpi.launched}")
    mpi.log("=" * 88)

    n_points = LATITUDE_COUNT * LONGITUDE_COUNT

    mpi.log("\n--- mpi runtime helpers ---")
    test_shared_configuration_agreement()
    test_reduce_buffer_agreement()
    test_mpi_runtime_helpers()
    test_mpi_logging_and_error_propagation()
    test_mpi_reduction_contracts()

    mpi.log("\n--- mpi.reduce ---")
    test_reduce_scalar_sum(n_points)
    test_reduce_array_sum(n_points, LATITUDE_COUNT, LONGITUDE_COUNT)
    test_reduce_dataarray_sum(LATITUDE_COUNT, LONGITUDE_COUNT)
    test_reduce_all_operations()

    mpi.log("\n--- mpi.xarray ---")
    test_distributed_open_dataset()
    test_distributed_redistribution(LATITUDE_COUNT, LONGITUDE_COUNT)
    test_distributed_isel(LATITUDE_COUNT, LONGITUDE_COUNT)
    test_distributed_sel(LATITUDE_COUNT, LONGITUDE_COUNT)
    for op_name in ("sum", "prod", "mean", "max", "min"):
        test_distributed_numeric_reduction(
            PLEV_COUNT,
            LATITUDE_COUNT,
            LONGITUDE_COUNT,
            op_name,
        )
    for op_name in ("any", "all"):
        test_distributed_logical_reduction(
            LATITUDE_COUNT,
            LONGITUDE_COUNT,
            op_name,
        )
    test_distributed_dataset_reduction(LATITUDE_COUNT, LONGITUDE_COUNT)
    test_empty_partition_reductions()
    test_reduction_dtype_contracts()
    test_collective_sequence_symmetry()
    test_distributed_xarray_contracts()
    test_distributed_arithmetic()
    test_reduction_redistribution(LATITUDE_COUNT, LONGITUDE_COUNT)

    mpi.log("\n--- data placement ---")
    test_rank0_source_distribution(LATITUDE_COUNT, LONGITUDE_COUNT)
    test_distributed_open_reductions()

    mpi.log("\n--- mpi.scatterv ---")
    test_scatterv_rows(n_points)
    test_scatterv_validation_and_edge_cases()

    mpi.log("\n--- xarray operations + mpi.reduce ---")
    test_cosine_weighted_mean(LATITUDE_COUNT, LONGITUDE_COUNT)
    test_mpi_decorator_modes()

    mpi.log("\n--- xgeo NetCDF interface ---")
    output_dir = str(OUTPUT_DIR)
    test_xgeo_interface_contracts(output_dir)
    test_parallel_netcdf_writer_options(output_dir)
    test_parallel_netcdf_write(output_dir)
    test_distributed_netcdf_roundtrip(output_dir)
    test_distributed_dataarray_roundtrip(output_dir)

    mpi.comm.barrier()
    if print_test_summary():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
