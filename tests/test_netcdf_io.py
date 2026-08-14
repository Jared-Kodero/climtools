"""Test suite for the serial and MPI-parallel NetCDF writers.

The same file runs in both modes and adapts to the world it finds:

    python -m climtools.tests.test_netcdf_io
    mpirun -n 4 python -m climtools.tests.test_netcdf_io
    srun --mpi=pmix --ntasks=4 python -m climtools.tests.test_netcdf_io

Serial cases run on rank zero. Parallel cases are collective and run on every
rank, so each one must be entered by all ranks in the same order. Cases that
require more than one rank are skipped, not failed, in a one-rank world.

The decisive check is :func:`test_parallel_matches_serial`: the file produced
by a collective write must be identical, value for value, to the file the
serial writer produces from the same global dataset.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

if __package__ in (None, ""):  # executed as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from climtools import lib_mpi
from climtools.netcdf import append_to_netcdf, to_netcdf_serial
from climtools.netcdf.parallel import (
    InconsistentRanksError,
    to_netcdf_parallel,
)

RANK, SIZE = lib_mpi.world()
ROOT = RANK == 0

Case = Callable[[Path], None]
_SERIAL: list[Case] = []
_PARALLEL: list[Case] = []


def serial_case(function: Case) -> Case:
    """Register a case that runs on rank zero only."""
    _SERIAL.append(function)
    return function


def parallel_case(function: Case) -> Case:
    """Register a collective case that every rank must enter."""
    _PARALLEL.append(function)
    return function


class Skip(Exception):
    """Raised by a case that cannot run in the current world."""


# ----------------------------------------------------------------- assertions


def check(condition: bool, message: str) -> None:
    """Fail a case when ``condition`` does not hold."""
    if not condition:
        raise AssertionError(message)


def check_close(actual: Any, expected: Any, message: str) -> None:
    """Fail a case when two arrays differ beyond floating-point tolerance."""
    a = np.asarray(actual)
    b = np.asarray(expected)
    check(a.shape == b.shape, f"{message}: shape {a.shape} != {b.shape}")
    if a.dtype.kind in "fc":
        check(
            bool(np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True)),
            f"{message}: values differ",
        )
    else:
        check(bool(np.array_equal(a, b)), f"{message}: values differ")


def raises(
    exception: type[BaseException],
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> BaseException:
    """Run a callable and require it to raise ``exception``.

    Returns
    -------
    BaseException
        The exception raised, so the caller can inspect its message.
    """
    try:
        function(*args, **kwargs)
    except exception as exc:
        return exc
    except BaseException as exc:
        raise AssertionError(
            f"expected {exception.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exception.__name__}, nothing was raised")


# --------------------------------------------------------------- test datasets


def global_dataset(n_time: int = 24, n_lat: int = 8, n_lon: int = 12) -> xr.Dataset:
    """Build a reproducible gridded dataset with realistic metadata.

    Parameters
    ----------
    n_time, n_lat, n_lon : int
        Shape of the field.

    Returns
    -------
    xarray.Dataset
        Dataset carrying a datetime axis, a partitioned field, a field that
        does not span time, a scalar, and global attributes.
    """
    rng = np.random.default_rng(20240513)
    time = np.datetime64("2020-01-01T00", "ns") + np.arange(n_time) * np.timedelta64(
        6, "h"
    ).astype("timedelta64[ns]")
    lat = np.linspace(-87.5, 87.5, n_lat)
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)

    t2m = (
        273.15
        + 20.0 * np.cos(np.deg2rad(lat))[None, :, None]
        + rng.normal(0.0, 0.5, size=(n_time, n_lat, n_lon))
    )
    ds = xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), t2m.astype("float32")),
            "orography": (("lat", "lon"), rng.uniform(0, 3000, (n_lat, n_lon))),
            "reference_height": xr.DataArray(np.float32(2.0)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"title": "climtools writer test", "Conventions": "CF-1.8"},
    )
    ds["t2m"].attrs = {"units": "K", "long_name": "2 metre temperature"}
    ds["orography"].attrs = {"units": "m", "long_name": "surface height"}
    ds["reference_height"].attrs = {"units": "m"}
    ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    return ds


def slab_bounds(total: int, rank: int, size: int) -> tuple[int, int]:
    """Return the half-open slab this rank owns along a partitioned axis.

    Parameters
    ----------
    total : int
        Global length of the partitioned dimension.
    rank, size : int
        Position in and size of ``MPI_COMM_WORLD``.

    Returns
    -------
    tuple of int
        ``(start, stop)``. The remainder is spread over the leading ranks so
        that the slabs stay contiguous and differ by at most one element.
    """
    base, remainder = divmod(total, size)
    start = rank * base + min(rank, remainder)
    stop = start + base + (1 if rank < remainder else 0)
    return start, stop


def local_slab(ds: xr.Dataset, dim: str = "time") -> xr.Dataset:
    """Return the contiguous slab of ``ds`` owned by the current rank."""
    start, stop = slab_bounds(ds.sizes[dim], RANK, SIZE)
    return ds.isel({dim: slice(start, stop)})


# ------------------------------------------------------------- serial coverage


@serial_case
def test_serial_roundtrip(tmp: Path) -> None:
    """A serial write must round-trip values, times and attributes."""
    ds = global_dataset()
    path = tmp / "serial_roundtrip.nc"
    to_netcdf_serial(ds, path, unlimited_dim="time", show_progress=False)

    with xr.open_dataset(path) as out:
        check_close(out["t2m"].values, ds["t2m"].values, "t2m")
        check_close(out["orography"].values, ds["orography"].values, "orography")
        check_close(
            out["time"].values.astype("datetime64[ns]").astype("int64"),
            ds["time"].values.astype("int64"),
            "time axis",
        )
        check(out.attrs.get("title") == ds.attrs["title"], "global attribute lost")
        check(out["t2m"].attrs.get("units") == "K", "variable attribute lost")


@serial_case
def test_serial_accepts_iterable_unlimited_dim(tmp: Path) -> None:
    """An iterable unlimited_dim must be accepted, not raise TypeError.

    Regression: the documented ``str | Iterable[str]`` signature previously
    reached ``dim0 not in data.sizes`` with a list and raised
    ``TypeError: unhashable type: 'list'``.
    """
    ds = global_dataset(n_time=6)
    path = tmp / "serial_iterable_dim.nc"
    to_netcdf_serial(ds, path, unlimited_dim=["time"], show_progress=False)

    with xr.open_dataset(path) as out:
        check(out.sizes["time"] == 6, "unlimited dimension not written in full")

    raises(
        ValueError,
        to_netcdf_serial,
        ds,
        tmp / "serial_bad_dim.nc",
        unlimited_dim=["not_a_dim"],
        show_progress=False,
    )


@serial_case
def test_serial_dataarray_creates_file(tmp: Path) -> None:
    """A named DataArray must be writable to a path that does not exist yet.

    Regression: the DataArray branch required the file to exist already and
    raised ``FileNotFoundError`` for every new output.
    """
    da = global_dataset(n_time=4)["t2m"]
    path = tmp / "serial_dataarray.nc"
    to_netcdf_serial(da, path, unlimited_dim="time", show_progress=False)

    with xr.open_dataset(path) as out:
        check_close(out["t2m"].values, da.values, "DataArray values")

    unnamed = da.rename(None)
    raises(
        ValueError,
        to_netcdf_serial,
        unnamed,
        tmp / "serial_unnamed.nc",
        show_progress=False,
    )


@serial_case
def test_serial_does_not_mutate_input(tmp: Path) -> None:
    """The writer must not encode time inside the caller's dataset.

    Regression: variables were assigned back into the argument, so the
    caller's time coordinate became int64 once the write returned.
    """
    ds = global_dataset(n_time=4)
    before = ds["time"].dtype
    to_netcdf_serial(ds, tmp / "serial_mutation.nc", show_progress=False)
    check(
        ds["time"].dtype == before,
        f"input time dtype changed from {before} to {ds['time'].dtype}",
    )
    check(np.issubdtype(ds["time"].dtype, np.datetime64), "input no longer datetime")


@serial_case
def test_append_preserves_time_units(tmp: Path) -> None:
    """Appending datetimes must respect the units already in the file.

    Regression: an unencoded datetime64 batch was cast straight to the stored
    integer type, writing nanoseconds into a seconds axis. The file then held
    two time origins and nothing raised.
    """
    ds = global_dataset(n_time=8)
    first, second = ds.isel(time=slice(0, 4)), ds.isel(time=slice(4, 8))
    path = tmp / "serial_append.nc"

    to_netcdf_serial(first, path, unlimited_dim="time", show_progress=False)
    append_to_netcdf(path, second, dim="time")

    with xr.open_dataset(path) as out:
        check(out.sizes["time"] == 8, "appended records missing")
        check_close(
            out["time"].values.astype("datetime64[ns]").astype("int64"),
            ds["time"].values.astype("int64"),
            "appended time axis",
        )
        check_close(out["t2m"].values, ds["t2m"].values, "appended field")


# ----------------------------------------------------------- parallel coverage


@parallel_case
def test_parallel_matches_serial(tmp: Path) -> None:
    """A collective write must reproduce the serial file exactly."""
    ds = global_dataset()
    parallel_path = tmp / "parallel_reference.nc"
    serial_path = tmp / "serial_reference.nc"

    to_netcdf_parallel(local_slab(ds), parallel_path, allow_serial=True)

    if ROOT:
        to_netcdf_serial(ds, serial_path, unlimited_dim="time", show_progress=False)
        with xr.open_dataset(parallel_path) as got, xr.open_dataset(serial_path) as ref:
            check(dict(got.sizes) == dict(ref.sizes), "dimension sizes differ")
            for name in ("t2m", "orography", "reference_height"):
                check_close(got[name].values, ref[name].values, f"{name} differs")
            check_close(
                got["time"].values.astype("datetime64[ns]").astype("int64"),
                ref["time"].values.astype("datetime64[ns]").astype("int64"),
                "time axis differs",
            )
            check(got.attrs.get("title") == ref.attrs.get("title"), "attrs differ")
            check(got["t2m"].attrs.get("units") == "K", "variable attrs lost")


@parallel_case
def test_parallel_does_not_mutate_input(tmp: Path) -> None:
    """The parallel bridge must leave the caller's slab untouched.

    Regression: time variables were encoded in place before the collective
    write, so every rank's dataset came back with an integer time axis.
    """
    slab = local_slab(global_dataset(n_time=16))
    before = slab["time"].dtype
    to_netcdf_parallel(slab, tmp / "parallel_mutation.nc", allow_serial=True)
    check(
        slab["time"].dtype == before,
        f"input time dtype changed from {before} to {slab['time'].dtype}",
    )


@parallel_case
def test_parallel_explicit_partition_dim(tmp: Path) -> None:
    """An explicit partition dimension must be honoured."""
    ds = global_dataset(n_time=20)
    path = tmp / "parallel_explicit.nc"
    to_netcdf_parallel(local_slab(ds), path, partition_dim="time", allow_serial=True)

    if ROOT:
        with xr.open_dataset(path) as out:
            check(out.sizes["time"] == ds.sizes["time"], "partition length wrong")
            check_close(out["t2m"].values, ds["t2m"].values, "field differs")


@parallel_case
def test_parallel_partitioned_along_space(tmp: Path) -> None:
    """Partitioning a dimension other than time must work equally well."""
    ds = global_dataset(n_time=6, n_lat=12, n_lon=9)
    path = tmp / "parallel_lat.nc"
    to_netcdf_parallel(
        local_slab(ds, dim="lat"),
        path,
        partition_dim="lat",
        allow_serial=True,
    )

    if ROOT:
        with xr.open_dataset(path) as out:
            check_close(out["t2m"].values, ds["t2m"].values, "field differs")
            check_close(out["lat"].values, ds["lat"].values, "latitude differs")


@parallel_case
def test_parallel_strings_and_scalars(tmp: Path) -> None:
    """Replicated strings, scalars and static fields must survive the write.

    String width is negotiated globally, so ranks holding shorter strings must
    still produce a file readable as one character array.
    """
    ds = local_slab(global_dataset(n_time=12))
    ds = ds.assign(
        source=xr.DataArray(
            np.array(["reanalysis", "model", "obs"], dtype=object),
            dims=("dataset",),
        ),
        ensemble_size=xr.DataArray(np.int32(50)),
    )
    path = tmp / "parallel_strings.nc"
    to_netcdf_parallel(ds, path, allow_serial=True)

    if ROOT:
        with xr.open_dataset(path) as out:
            values = [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in np.asarray(out["source"].values).ravel()
            ]
            check(values == ["reanalysis", "model", "obs"], f"strings wrong: {values}")
            check(int(out["ensemble_size"].values) == 50, "scalar wrong")


@parallel_case
def test_parallel_harmonises_time_units(tmp: Path) -> None:
    """Ranks whose values imply different CF units must still agree.

    Regression: the bridge encoded time before the collective negotiation, so
    a rank holding whole hours chose ``hours`` while another chose
    ``seconds`` and the write failed on a schema mismatch. The negotiation
    must pick the finer unit and encode every rank against it.
    """
    if SIZE < 2:
        raise Skip("needs at least two ranks")

    values = (
        np.array([np.timedelta64(1, "h"), np.timedelta64(2, "h")])
        if RANK % 2 == 0
        else np.array([np.timedelta64(90, "s"), np.timedelta64(30, "s")])
    )
    ds = xr.Dataset(
        {
            "lead": (("time",), values),
            "field": (("time",), np.full(2, float(RANK))),
        },
        coords={"time": np.arange(RANK * 2, RANK * 2 + 2)},
    )
    path = tmp / "parallel_time_units.nc"
    to_netcdf_parallel(ds, path)

    if ROOT:
        with xr.open_dataset(path) as out:
            seconds = (
                out["lead"].values.astype("timedelta64[s]").astype("int64").tolist()
            )
            expected: list[int] = []
            for rank in range(SIZE):
                expected += [3600, 7200] if rank % 2 == 0 else [90, 30]
            check(seconds == expected, f"lead wrong: {seconds} != {expected}")


@parallel_case
def test_parallel_unlimited_dimension(tmp: Path) -> None:
    """An unlimited record dimension must be usable in collective output."""
    ds = global_dataset(n_time=16)
    path = tmp / "parallel_unlimited.nc"
    to_netcdf_parallel(
        local_slab(ds),
        path,
        unlimited_dim="time",
        allow_serial=True,
    )

    if ROOT:
        import netCDF4

        with netCDF4.Dataset(path) as raw:
            check(raw.dimensions["time"].isunlimited(), "time is not unlimited")
        with xr.open_dataset(path) as out:
            check_close(out["t2m"].values, ds["t2m"].values, "field differs")


@parallel_case
def test_parallel_compression_policy(tmp: Path) -> None:
    """Compression must either apply or be declined explicitly, never crash.

    Filter support is a property of the linked library and therefore the same
    on every rank, so the downgrade cannot desynchronise the collective.
    """
    ds = local_slab(global_dataset(n_time=12))
    path = tmp / "parallel_deflate.nc"

    if lib_mpi.has_parallel_filters():
        to_netcdf_parallel(ds, path, deflate=4, allow_serial=True)
        if ROOT:
            import netCDF4

            with netCDF4.Dataset(path) as raw:
                filters = raw.variables["t2m"].filters()
            check(bool(filters and filters.get("zlib")), "deflate was not applied")
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            to_netcdf_parallel(ds, path, deflate=4, allow_serial=True)
        if ROOT:
            check(
                any(issubclass(item.category, RuntimeWarning) for item in caught),
                "no warning was issued when compression was unavailable",
            )
        raises(
            Exception,
            to_netcdf_parallel,
            ds,
            tmp / "parallel_deflate_strict.nc",
            deflate=4,
            allow_serial=True,
            strict_compression=True,
        )

    if ROOT:
        with xr.open_dataset(path) as out:
            check(out.sizes["time"] > 0, "compressed output is empty")


@parallel_case
def test_parallel_reports_symmetric_errors_faithfully(tmp: Path) -> None:
    """A bad argument must keep its own exception type on every rank.

    Regression: a ``TypeError`` raised identically everywhere was reported as
    ``InconsistentRanksError: Rank 0 failed ...``, which names a culprit that
    does not exist and hides the actual fault.
    """
    ds = local_slab(global_dataset(n_time=8))
    error = raises(
        TypeError,
        to_netcdf_parallel,
        ds,
        tmp / "parallel_bad_option.nc",
        deflate="four",
        allow_serial=True,
    )
    check("deflate" in str(error), f"unhelpful message: {error}")


@parallel_case
def test_parallel_detects_rank_disagreement(tmp: Path) -> None:
    """Ranks holding different variables must fail, on every rank."""
    if SIZE < 2:
        raise Skip("needs at least two ranks")

    ds = local_slab(global_dataset(n_time=12))
    if RANK == SIZE - 1:
        ds = ds.assign(extra=ds["t2m"].isel(lat=0, lon=0))

    raises(
        InconsistentRanksError,
        to_netcdf_parallel,
        ds,
        tmp / "parallel_disagree.nc",
    )


@parallel_case
def test_parallel_missing_partition_dim(tmp: Path) -> None:
    """A partition dimension that is not partitioned must be rejected."""
    if SIZE < 2:
        raise Skip("needs at least two ranks")

    ds = local_slab(global_dataset(n_time=12))
    raises(
        InconsistentRanksError,
        to_netcdf_parallel,
        ds,
        tmp / "parallel_wrong_dim.nc",
        partition_dim="lat",
    )


@parallel_case
def test_parallel_requires_allow_serial_on_one_rank(tmp: Path) -> None:
    """A one-rank world must be opted into rather than assumed."""
    if SIZE != 1:
        raise Skip("only meaningful in a one-rank world")

    ds = global_dataset(n_time=4)
    raises(
        lib_mpi.NativeLibraryError,
        to_netcdf_parallel,
        ds,
        tmp / "parallel_single.nc",
    )
    to_netcdf_parallel(ds, tmp / "parallel_single_ok.nc", allow_serial=True)


@parallel_case
def test_mpi_decorators(tmp: Path) -> None:
    """The decorators must place execution where they claim to."""
    from climtools.lib_mpi import mpi

    @mpi(all_ranks=True)
    def everywhere() -> int:
        return RANK

    @mpi()
    def root_only() -> str:
        return f"written by rank {RANK}"

    @mpi(broadcast=True)
    def shared() -> dict[str, int]:
        return {"size": SIZE}

    check(everywhere() == RANK, "all_ranks did not run locally")

    result = root_only()
    if ROOT:
        check(result == "written by rank 0", f"root result wrong: {result}")
    else:
        check(result is None, "a non-root rank executed a root-only function")

    check(shared() == {"size": SIZE}, "broadcast did not reach every rank")

    raises(ValueError, mpi, root=SIZE + 10)
    raises(ValueError, mpi, all_ranks=True, broadcast=True)


@parallel_case
def test_xgeo_api(tmp: Path) -> None:
    """The public xgeo entry point must drive both writers."""
    try:
        from climtools import xgeo as xg
    except ImportError as exc:  # plotting extras absent
        raise Skip(f"xgeo unavailable: {exc}") from exc

    ds = global_dataset(n_time=16)
    parallel_path = tmp / "xgeo_parallel.nc"
    xg.to_netcdf(local_slab(ds), parallel_path, parallel=True, allow_serial=True)

    if ROOT:
        serial_path = tmp / "xgeo_serial.nc"
        xg.to_netcdf(ds, serial_path, unlimited_dim="time", show_progress=False)
        with xr.open_dataset(parallel_path) as got, xr.open_dataset(serial_path) as ref:
            check_close(got["t2m"].values, ref["t2m"].values, "xgeo paths disagree")

    raises(
        ValueError,
        xg.to_netcdf,
        local_slab(ds),
        tmp / "xgeo_bad_format.nc",
        parallel=True,
        format="NETCDF3_CLASSIC",
        allow_serial=True,
    )


# ------------------------------------------------------------------- execution


def shared_directory() -> Path:
    """Return a scratch directory that every rank agrees on.

    Returns
    -------
    pathlib.Path
        Directory created by rank zero and broadcast to the others, so all
        ranks write into one location on the shared file system.
    """
    if SIZE == 1:
        return Path(tempfile.mkdtemp(prefix="climtools_io_"))

    from climtools.lib_mpi import mpi

    @mpi(broadcast=True)
    def make() -> str:
        return tempfile.mkdtemp(prefix="climtools_io_", dir=str(Path.cwd()))

    return Path(str(make()))


def run(cases: list[Case], tmp: Path, label: str) -> tuple[int, int, int]:
    """Run a list of cases and report the outcome.

    Returns
    -------
    tuple of int
        Counts of passed, skipped and failed cases.
    """
    passed = skipped = failed = 0
    for case in cases:
        name = case.__name__
        try:
            case(tmp)
        except Skip as exc:
            skipped += 1
            if ROOT:
                print(f"  SKIP {name}: {exc}")
        except BaseException:  # a failure must not stop the rest of the run
            failed += 1
            if ROOT:
                print(f"  FAIL {name}")
                traceback.print_exc()
        else:
            passed += 1
            if ROOT:
                print(f"  pass {name}")
    if ROOT:
        print(f"{label}: {passed} passed, {skipped} skipped, {failed} failed")
    return passed, skipped, failed


def main() -> int:
    """Run the suite and return a process exit code."""
    if ROOT:
        print(f"climtools NetCDF writer tests: {SIZE} rank(s)")
        print(f"extension: {lib_mpi.info()}")

    tmp = shared_directory()
    failed = 0

    if ROOT:
        print("\nserial writer")
        _, _, serial_failed = run(_SERIAL, tmp, "serial")
        failed += serial_failed

    if lib_mpi.available():
        if ROOT:
            print("\nparallel writer")
        _, _, parallel_failed = run(_PARALLEL, tmp, "parallel")
        failed += parallel_failed
    elif ROOT:
        print("\nparallel writer: skipped, extension not built")

    if ROOT:
        shutil.rmtree(tmp, ignore_errors=True)
        print("\nRESULT:", "FAILURE" if failed else "SUCCESS")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
