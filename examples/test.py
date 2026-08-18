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
write test is skipped automatically when running with more than one rank.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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


def _require_source() -> None:
    """Require the configured SHiELD NetCDF file on every MPI rank."""
    visible = DEFAULT_NETCDF_SOURCE.is_file()
    if not bool(mpi.reduce.all(visible)):
        raise FileNotFoundError(
            "SHiELD NetCDF source is not visible on every rank: "
            f"{DEFAULT_NETCDF_SOURCE}"
        )


def _rank_bounds(size: int, rank: int = RANK) -> tuple[int, int]:
    """Return this rank's contiguous bounds within a global dimension."""
    return size * rank // SIZE, size * (rank + 1) // SIZE


def _load_source_variable(
    variable: str,
    **indexers: int | slice,
) -> xr.DataArray:
    """Load only the requested selection of one variable from the SHiELD file."""
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

    with timed() as box:
        local_partial = float(local.sum(skipna=True))
        combined = mpi.reduce.sum(local_partial)
    parallel_s = box["seconds"]

    def serial_fn() -> float:
        field = _load_source_variable("pr", time=0, lat=slice(0, n_rows))
        return float(field.sum(skipna=True))

    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(np.isclose(combined, expected, rtol=1.0e-8, equal_nan=True))
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
        return np.asarray(field.values, dtype=np.float64)

    local = load_rank_field(RANK)

    with timed() as box:
        combined = mpi.reduce.sum(local)
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        fields = [load_rank_field(rank) for rank in range(SIZE)]
        return np.sum(np.stack(fields, axis=0), axis=0)

    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(np.allclose(combined, expected, rtol=1.0e-8, equal_nan=True))
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

    with timed() as box:
        combined = mpi.reduce.sum(local)
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        fields = [load_rank_field(rank).values for rank in range(SIZE)]
        return np.sum(np.stack(fields, axis=0), axis=0)

    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(
        np.allclose(combined.values, expected, rtol=1.0e-8, equal_nan=True)
    ) and combined.attrs.get("units") == local.attrs.get("units")
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
        return np.asarray(data.values, dtype=np.float64)

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
        with timed() as box:
            result = op(value)
        parallel_s = box["seconds"]

        root_result = op(value, mode="root", root=0)
        all_mode_ok = bool(np.allclose(result, expected, equal_nan=True))
        root_mode_ok = (
            bool(np.allclose(root_result, expected, equal_nan=True))
            if RANK == 0
            else root_result is None
        )
        correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
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
    with timed() as box:
        distributed = mpi.xarray.open_dataset(
            str(DEFAULT_NETCDF_SOURCE),
            partition_dim="lat",
        )[["pr"]]
        distributed["pr"].isel(time=0).load()
    parallel_s = box["seconds"]

    local = distributed["pr"].isel(time=0).values.copy()
    meta = distributed.attrs.get("mpi_meta")
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

    def serial_fn() -> float:
        field = _load_source_variable("pr", time=0)
        return float(field.sum(skipna=True))

    _, serial_s = run_serial_baseline(serial_fn)
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
    record(
        "mpi.xarray.sel real pr (global latitude slice/scalar)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
    )


def test_xarray_reduction(n_levels_max: int, ny: int, nx: int, op_name: str) -> None:
    """Numeric mpi.xarray reductions using the real SHiELD temperature profile."""
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

    with timed() as box:
        result = op(distributed, dim="plev", **kwargs)
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        serial_kwargs = {"skipna": True}
        if op_name in {"sum", "prod"}:
            serial_kwargs["min_count"] = 1
        return getattr(full, op_name)(dim="plev", **serial_kwargs).values

    expected, serial_s = run_serial_baseline(serial_fn)
    root_result = op(distributed, dim="plev", mode="root", root=0, **kwargs)
    all_mode_ok = (
        result is not None
        and bool(np.allclose(result.values, expected, rtol=1.0e-9, equal_nan=True))
        and result.attrs.get("units") == full.attrs.get("units")
    )
    root_mode_ok = (
        root_result is not None
        and bool(np.allclose(root_result.values, expected, rtol=1.0e-9, equal_nan=True))
        if RANK == 0
        else root_result is None
    )
    correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
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
    mask = _load_source_variable(
        "slmsk",
        time=0,
        lat=slice(0, n_lat_max),
        lon=slice(0, nx),
    )
    full = mask == 1
    distributed = mpi.xarray.redistribute(full, "lat")
    op = getattr(mpi.xarray, op_name)

    with timed() as box:
        result = op(distributed, dim="lat")
    parallel_s = box["seconds"]

    def serial_fn() -> np.ndarray:
        return getattr(full, op_name)(dim="lat").values

    expected, serial_s = run_serial_baseline(serial_fn)
    root_result = op(distributed, dim="lat", mode="root", root=0)
    all_mode_ok = result is not None and bool(np.array_equal(result.values, expected))
    root_mode_ok = (
        root_result is not None and bool(np.array_equal(root_result.values, expected))
        if RANK == 0
        else root_result is None
    )
    correct = bool(mpi.reduce.all(all_mode_ok and root_mode_ok))
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

    with timed() as box:
        result = mpi.xarray.sum(distributed, dim="lat")
        mean_result = mpi.xarray.mean(distributed, dim=("lat", "lon"))
    parallel_s = box["seconds"]

    expected = full.sum(dim="lat")
    expected_mean = full.mean(dim=("lat", "lon"))
    correct = (
        result is not None
        and mean_result is not None
        and bool(
            np.allclose(
                result["t2m"].values,
                expected["t2m"].values,
                equal_nan=True,
            )
        )
        and bool(
            np.array_equal(
                result["plev_values"].values,
                expected["plev_values"].values,
            )
        )
        and bool(
            np.allclose(
                mean_result["t2m"].values,
                expected_mean["t2m"].values,
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

    profile = _load_source_variable(
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
            )
        )
        and bool(
            np.isclose(
                float(maximum.item()),
                float(profile.max(skipna=True).item()),
            )
        )
    )
    correct = bool(mpi.reduce.all(correct))
    record(
        "mpi.xarray Dataset reductions (real distributed/static variables)",
        correct,
        0.0,
        parallel_s,
        note="correctness-focused",
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

    with timed() as box:
        result = mpi.xarray.mean(distributed, dim="lat", redistribute_on="lon")
    parallel_s = box["seconds"]

    parts = mpi.comm.allgather(result.values)
    assembled = np.concatenate(parts, axis=result.get_axis_num("lon"))

    def serial_fn() -> np.ndarray:
        return full.mean(dim="lat").values

    expected, serial_s = run_serial_baseline(serial_fn)
    meta = result.attrs.get("mpi_meta")
    local_meta_ok = (
        isinstance(meta, dict)
        and meta.get("dim") == "lon"
        and int(meta.get("global_size", -1)) == full.sizes["lon"]
    )
    correct = bool(
        np.allclose(assembled, expected, rtol=1.0e-9, equal_nan=True)
    ) and bool(mpi.reduce.all(local_meta_ok))
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
    record(
        f"mpi.scatterv real t2m ({total} rows across {SIZE} rank(s))",
        bool(mpi.reduce.all(correct)),
        0.0,
        parallel_s,
        note="data movement, no serial-compute equivalent",
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

    expected, serial_s = run_serial_baseline(serial_fn)
    correct = bool(np.isclose(weighted_mean, expected, rtol=1.0e-8, equal_nan=True))
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
            f"{type(exc).__name__} on every rank: {exc}"
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


def test_netcdf_write(out_dir: str) -> None:
    """Compare parallel and serial writes of selected variables from the SHiELD file."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        mpi.log(
            "\n--- NetCDF write speed test SKIPPED: netCDF4 lacks parallel4 "
            "support (see climtools/env/setup_env.sh) ---"
        )
        return

    mpi.log("\n--- NetCDF write: selected real SHiELD data, parallel vs serial ---")

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

    _, serial_s = run_serial_baseline(serial_fn)

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
    record(
        "NetCDF write (selected real SHiELD variables)",
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


def test_netcdf_distributed_roundtrip(out_dir: str) -> None:
    """Compare distributed and serial writes of selected real SHiELD variables."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        mpi.log(
            "\n--- Distributed NetCDF round-trip SKIPPED: netCDF4 lacks "
            "parallel4 support ---"
        )
        return

    serial_path = os.path.join(out_dir, "climtools_test_distributed_serial.nc")
    parallel_path = os.path.join(out_dir, "climtools_test_distributed_parallel.nc")

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

    _, serial_s = run_serial_baseline(serial_fn)
    serial_data = None
    mpi.comm.barrier()

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
        "distributed NetCDF round-trip (selected real SHiELD variables)",
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


def test_netcdf_distributed_dataarray(out_dir: str) -> None:
    """Round-trip a selected real SHiELD precipitation DataArray in distributed mode."""
    import netCDF4

    if SIZE > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
        mpi.log(
            "\n--- Distributed DataArray NetCDF test SKIPPED: netCDF4 lacks "
            "parallel4 support ---"
        )
        return

    serial_path = os.path.join(out_dir, "climtools_test_pr_serial.nc")
    parallel_path = os.path.join(out_dir, "climtools_test_pr_parallel.nc")

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
    xgeo.to_netcdf(
        distributed_pr,
        parallel_path,
        unlimited_dim="time",
        parallel=True,
        allow_serial=(SIZE == 1),
    )
    distributed_ds.close()

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
    record(
        "distributed NetCDF DataArray round-trip (selected real SHiELD pr)",
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
    _require_source()

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
    safe_run(
        test_xarray_redistribute_on,
        args.xarray_events,
        args.xarray_nx,
    )

    mpi.log("\n--- mpi.scatterv ---")
    safe_run(test_scatterv, args.n_events)

    mpi.log("\n--- xarray operations + mpi.reduce ---")
    safe_run(test_weighted_mean, args.n_lat, args.n_lon)

    safe_run(test_mpi_decorator)

    if not args.skip_netcdf:
        safe_run(test_netcdf_write, args.out_dir)
        safe_run(
            test_netcdf_distributed_roundtrip,
            args.out_dir,
        )
        safe_run(
            test_netcdf_distributed_dataarray,
            args.out_dir,
        )

    mpi.comm.barrier()
    print_summary()


if __name__ == "__main__":
    main()
