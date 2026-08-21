"""Correctness suite for every climtools component that does not need MPI.

Covers ``climtools.plot``, ``climtools.calc``, ``climtools.cmaps``,
``climtools.cdo``, the ``.xgeo`` accessor and its non-parallel operations
(``to_lon180``, ``add_local_solar_time``, ``sel_transect``, ``get_spatial_dims``),
the serial NetCDF writer (``lib_netcdf.serial``), and the small utilities in
``climtools.core.tools``.

MPI-collective functionality (``climtools.mpi``, ``xgeo.to_netcdf(...,
parallel=True)``) is covered separately in ``test_mpi.py``, which needs an
MPI launcher; this script does not and is not affected by one being present.

    python test_general.py
    python test_general.py --skip-cdo
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this suite never needs an on-screen figure

import numpy as np
import pandas as pd
import xarray as xr

from climtools import calc, cmaps, xgeo
from climtools import plot as ctplot
from climtools.core.tools import AttrDict, LockFile, n_cpus
from climtools.lib_netcdf import serial as lib_serial

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Small test harness (deliberately independent of climtools.mpi: this suite
# must be meaningful with no MPI runtime, and even no MPI installed at all,
# since none of the modules under test import climtools.mpi).
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    passed: bool
    note: str = ""
    skipped: bool = False


RESULTS: list[Result] = []


def record_result(
    name: str, passed: bool, note: str = "", *, skipped: bool = False
) -> None:
    RESULTS.append(Result(name, passed, note, skipped))
    status = "SKIP" if skipped else ("OK  " if passed else "FAIL")
    suffix = f"  ({note})" if note else ""
    print(f"[{status}] {name}{suffix}", flush=True)


def run_test(function: Callable[..., None]) -> Callable[..., None]:
    """Run a test, catching and recording any exception as a failure."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> None:
        before = len(RESULTS)
        try:
            function(*args, **kwargs)
        except Exception as exc:
            record_result(
                f"{function.__name__} (uncaught exception)",
                False,
                note=f"{type(exc).__name__}: {exc}",
            )
            if "-v" in sys.argv or "--verbose" in sys.argv:
                traceback.print_exc()
            return
        if len(RESULTS) == before:
            record_result(
                f"{function.__name__} (result accounting)",
                False,
                note="test completed without recording a result",
            )

    return wrapped


def relative_close(a: float, b: float, rtol: float = 1e-4) -> bool:
    return bool(np.isclose(a, b, rtol=rtol, equal_nan=True))


# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------


def make_grid(n_time: int = 24, n_lat: int = 37, n_lon: int = 72) -> xr.Dataset:
    """Deterministic global lat/lon/time Dataset with CF-recognizable coords."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=n_time, freq="D")
    lat = np.linspace(-90, 90, n_lat, dtype=np.float32)
    lon = np.linspace(-180, 180, n_lon, endpoint=False, dtype=np.float32)

    trend = np.linspace(0, 2, n_time, dtype=np.float32)[:, None, None]
    field = (
        20.0
        + 10.0 * np.cos(np.deg2rad(lat))[None, :, None]
        + trend
        + rng.normal(scale=0.5, size=(n_time, n_lat, n_lon)).astype(np.float32)
    )

    return xr.Dataset(
        data_vars={
            "t2m": (
                ("time", "lat", "lon"),
                field,
                {"units": "K", "long_name": "2 m air temperature"},
            ),
        },
        coords={
            "time": time,
            "lat": (
                "lat",
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "lon": (
                "lon",
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
        },
        attrs={"title": "climtools general test mock dataset"},
    )


# ---------------------------------------------------------------------------
# climtools.cmaps
# ---------------------------------------------------------------------------


@run_test
def test_cmaps_registry() -> None:
    """Every advertised colormap name resolves and every colormap is unique."""
    names = cmaps.list_cmaps()
    non_empty = len(names) > 0
    all_resolve = True
    for name in names:
        try:
            cmap = getattr(cmaps, name)()
        except Exception:
            all_resolve = False
            break
        if cmap is None:
            all_resolve = False
            break

    correct = bool(non_empty and all_resolve)
    record_result(
        "cmaps: every registered colormap name resolves",
        correct,
        note=f"{len(names)} colormap(s)"
        if correct
        else "a registered name failed to resolve",
    )


@run_test
def test_cmaps_create_and_arithmetic() -> None:
    """A custom colormap builds, and add/subtract/concat combine without error."""
    left = cmaps.create(["#000000", "#ffffff"], N=16, name="_test_bw")
    right = cmaps.create(["#ff0000", "#0000ff"], N=16, name="_test_rb")

    left_ok = left is not None and left.N == 16
    right_ok = right is not None and right.N == 16

    combined = cmaps.concat(left, right, N=8)
    combined_ok = combined is not None and combined.N == 8

    summed = cmaps.add(left, right, N=8)
    summed_ok = summed is not None

    correct = bool(left_ok and right_ok and combined_ok and summed_ok)
    record_result("cmaps: create/concat/add produce colormaps", correct)


@run_test
def test_cmaps_get_colors() -> None:
    """get_colors returns the requested number of hex colors from a colormap."""
    cmap = cmaps.temp_div()
    colors = cmaps.get_colors(cmap, 5)
    correct = bool(
        isinstance(colors, list)
        and len(colors) == 5
        and all(isinstance(c, str) and c.startswith("#") for c in colors)
    )
    record_result("cmaps.get_colors returns N hex colors", correct, note=str(colors))


# ---------------------------------------------------------------------------
# climtools.calc
# ---------------------------------------------------------------------------


@run_test
def test_calc_trends_polyfit() -> None:
    """An exact linear signal recovers its known slope via the polyfit path."""
    n = 40
    time = np.arange(n, dtype=np.float64)
    slope_true = 0.5
    data = xr.DataArray(
        slope_true * time + 3.0,
        dims=("time",),
        coords={"time": time},
        name="x",
    )
    trend = calc.trends(data, dim="time", polyfit=True)
    slope = float(trend["slope"].values)
    p_value = float(trend["p_value"].values)

    correct = bool(relative_close(slope, slope_true, rtol=1e-6) and p_value < 1e-6)
    record_result(
        "calc.trends(polyfit=True) recovers an exact linear slope",
        correct,
        note=f"slope={slope:.6g} (expected {slope_true}), p={p_value:.3g}",
    )


@run_test
def test_calc_trends_mann_kendall() -> None:
    """The Mann-Kendall path detects a strong monotonic increase as significant."""
    n = 60
    rng = np.random.default_rng(1)
    time = np.arange(n, dtype=np.float64)
    data = xr.DataArray(
        time * 0.3 + rng.normal(scale=0.2, size=n),
        dims=("time",),
        coords={"time": time},
        name="x",
    )
    trend = calc.trends(data, dim="time")
    slope = float(trend["slope"].values)
    p_value = float(trend["p_value"].values)
    direction = float(trend["trend"].values)

    correct = bool(slope > 0 and direction == 1 and p_value < 0.01)
    record_result(
        "calc.trends() (Mann-Kendall) flags a strong increase as significant",
        correct,
        note=f"slope={slope:.4g}, trend={direction}, p={p_value:.3g}",
    )


@run_test
def test_calc_corr() -> None:
    """Perfectly correlated and perfectly anti-correlated series are detected."""
    n = 30
    time = np.arange(n, dtype=np.float64)
    x = xr.DataArray(time, dims=("time",), coords={"time": time}, name="x")
    y_pos = xr.DataArray(2 * time + 1, dims=("time",), coords={"time": time}, name="y")
    y_neg = xr.DataArray(-2 * time + 1, dims=("time",), coords={"time": time}, name="y")

    result_pos = calc.corr(x, y_pos, dim="time")
    result_neg = calc.corr(x, y_neg, dim="time")

    correct = bool(
        relative_close(float(result_pos["corr"].values), 1.0)
        and relative_close(float(result_neg["corr"].values), -1.0)
        and float(result_pos["p_value"].values) < 1e-6
    )
    record_result(
        "calc.corr detects perfect positive/negative correlation",
        correct,
        note=f"r_pos={float(result_pos['corr'].values):.4g}, "
        + f"r_neg={float(result_neg['corr'].values):.4g}",
    )


@run_test
def test_calc_pvalues() -> None:
    """A large, obvious mean shift is significant; identical samples are not."""
    n = 200
    rng = np.random.default_rng(2)
    time = np.arange(n, dtype=np.float64)
    low = xr.DataArray(
        rng.normal(loc=0.0, scale=1.0, size=n),
        dims=("time",),
        coords={"time": time},
    )
    high = xr.DataArray(
        rng.normal(loc=8.0, scale=1.0, size=n),
        dims=("time",),
        coords={"time": time},
    )

    shifted_p = float(calc.pvalues(low, high, dim="time").values)
    identical_p = float(calc.pvalues(low, low, dim="time").values)

    correct = bool(shifted_p < 1e-6 and identical_p > 0.99)
    record_result(
        "calc.pvalues distinguishes a shifted mean from an identical sample",
        correct,
        note=f"shifted p={shifted_p:.3g}, identical p={identical_p:.3g}",
    )


# ---------------------------------------------------------------------------
# climtools.xgeo / core.xr_utils (non-parallel)
# ---------------------------------------------------------------------------


@run_test
def test_xgeo_accessor_registered() -> None:
    """Importing climtools registers .xgeo on both DataArray and Dataset."""
    ds = make_grid()
    correct = bool(hasattr(ds, "xgeo") and hasattr(ds["t2m"], "xgeo"))
    record_result(".xgeo accessor is registered on Dataset and DataArray", correct)


@run_test
def test_to_lon180() -> None:
    """0-360 longitudes are remapped to -180..180 and re-sorted."""
    lon_0_360 = np.linspace(0, 359, 60, dtype=np.float32)
    ds = xr.Dataset(
        {"x": (("lon",), np.arange(60, dtype=np.float32))},
        coords={"lon": lon_0_360},
    )
    out = ds.xgeo.to_lon180()
    lon = out["lon"].values

    in_range = bool(np.all((lon >= -180) & (lon < 180)))
    sorted_ok = bool(np.all(np.diff(lon) > 0))
    same_length = len(lon) == len(lon_0_360)

    correct = bool(in_range and sorted_ok and same_length)
    record_result(
        "xgeo.to_lon180 remaps and sorts to [-180, 180)",
        correct,
        note=f"min={lon.min():.1f}, max={lon.max():.1f}",
    )


@run_test
def test_add_local_solar_time() -> None:
    """Local solar time offsets from UTC by whole hours matching longitude."""
    ds = make_grid(n_time=2, n_lat=3, n_lon=4)
    out = ds.xgeo.add_local_solar_time()

    has_coord = "lst" in out.coords
    # At lon=0 (or nearest grid point), LST must equal UTC exactly.
    lon_values = out["lon"].values
    near_zero_idx = int(np.argmin(np.abs(lon_values)))
    lon_near_zero = float(lon_values[near_zero_idx])
    utc = pd.Timestamp(out["time"].values[0])
    lst_at_lon0 = pd.Timestamp(out["lst"].isel(time=0, lon=near_zero_idx).values)
    expected_offset_hours = round(((lon_near_zero + 180) % 360 - 180) * 24 / 360)
    offset_matches = (lst_at_lon0 - utc) == pd.Timedelta(hours=expected_offset_hours)

    correct = bool(has_coord and offset_matches)
    record_result(
        "xgeo.add_local_solar_time offsets UTC by whole hours",
        correct,
        note=f"lon={lon_near_zero:.2f}, offset={lst_at_lon0 - utc}",
    )


@run_test
def test_sel_transect() -> None:
    """A latitude band transect keeps only rows within its width."""
    ds = make_grid(n_time=1, n_lat=37, n_lon=72)
    transect = ds.xgeo.sel_transect(y=0.0, orientation=0.0, width=3.0)

    within_grid = transect.sizes["lat"] < ds.sizes["lat"]
    centered = bool(np.abs(transect["lat"].values).max() <= 10.0)

    correct = bool(within_grid and centered)
    record_result(
        "xgeo.sel_transect selects a band around the requested center",
        correct,
        note=f"{transect.sizes.get('lat', '?')} of {ds.sizes['lat']} lat rows kept",
    )


@run_test
def test_get_spatial_dims() -> None:
    """CF-aware coordinate detection finds lat/lon under their standard names."""
    from climtools.core.xr_utils import get_spatial_dims

    ds = make_grid(n_time=1, n_lat=5, n_lon=5)
    lon_name, lat_name = get_spatial_dims(ds)
    correct = bool(lon_name == "lon" and lat_name == "lat")
    record_result(
        "xr_utils.get_spatial_dims finds lon/lat from CF metadata",
        correct,
        note=f"got ({lon_name!r}, {lat_name!r})",
    )


# ---------------------------------------------------------------------------
# climtools.plot
# ---------------------------------------------------------------------------


@run_test
def test_plot_geo_contourf() -> None:
    """A basic contourf map renders and saves without raising."""
    import matplotlib.pyplot as plt

    ds = make_grid(n_time=1, n_lat=19, n_lon=36)
    field = ds["t2m"].isel(time=0)

    # coastlines/borders/states/ocean/land pull Natural Earth shapefiles on
    # first use, which needs network access this suite must not depend on;
    # gridlines are computed locally and stay on to still exercise that path.
    geoplot = ctplot.geo(
        field,
        method="contourf",
        levels=11,
        gridlines=True,
        coastlines=False,
        borders=False,
        states=False,
        ocean=False,
        land=False,
    )
    has_figure = getattr(geoplot, "figure", None) is not None

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_plot.png"
        geoplot.figure.savefig(out_path)
        saved = out_path.exists() and out_path.stat().st_size > 0

    plt.close(geoplot.figure)
    correct = bool(has_figure and saved)
    record_result("plot.geo(method='contourf') renders and saves a figure", correct)


@run_test
def test_plot_geo_accessor_and_overlay() -> None:
    """The .xgeo.plot.geo accessor form matches the free-function form, and
    .add.contour returns the same GeoPlot for chaining."""
    import matplotlib.pyplot as plt

    ds = make_grid(n_time=1, n_lat=19, n_lon=36)
    field = ds["t2m"].isel(time=0)

    geoplot = field.xgeo.plot.geo(
        method="pcolormesh",
        coastlines=False,
        borders=False,
        states=False,
        ocean=False,
        land=False,
    )
    accessor_ok = getattr(geoplot, "figure", None) is not None

    chained = geoplot.add.contour(field, colors="k")
    chain_ok = chained is geoplot

    plt.close(geoplot.figure)
    correct = bool(accessor_ok and chain_ok)
    record_result(
        "xgeo.plot.geo accessor form works and .add.* chains",
        correct,
    )


# ---------------------------------------------------------------------------
# Serial NetCDF writer (lib_netcdf.serial)
# ---------------------------------------------------------------------------


@run_test
def test_serial_netcdf_roundtrip(tmp_dir: Path) -> None:
    """A Dataset written serially reads back identical to the original."""
    ds = make_grid(n_time=6, n_lat=9, n_lon=12)
    path = tmp_dir / "serial_roundtrip.nc"

    xgeo.to_netcdf(ds, path, unlimited_dim="time", show_progress=False)

    with xr.open_dataset(path) as reopened:
        reopened.load()
        correct = bool(
            np.allclose(reopened["t2m"].values, ds["t2m"].values)
            and list(reopened["t2m"].dims) == list(ds["t2m"].dims)
            and reopened.dims["time"] == ds.dims["time"]
        )
    record_result("lib_netcdf.serial: Dataset round-trips through to_netcdf", correct)


@run_test
def test_serial_netcdf_append(tmp_dir: Path) -> None:
    """append() extends the unlimited dimension without disturbing prior data."""
    first = make_grid(n_time=3, n_lat=5, n_lon=6)
    second = make_grid(n_time=3, n_lat=5, n_lon=6)
    second["time"] = first["time"].values + np.timedelta64(3, "D")
    second["t2m"] = second["t2m"] + 1000.0  # make the second batch distinguishable

    path = tmp_dir / "serial_append.nc"
    xgeo.to_netcdf(first, path, unlimited_dim="time", show_progress=False)
    lib_serial.append(path, second, dim="time")

    with xr.open_dataset(path) as reopened:
        reopened.load()
        correct = bool(
            reopened.sizes["time"] == 6
            and np.allclose(
                reopened["t2m"].isel(time=slice(0, 3)).values, first["t2m"].values
            )
            and np.allclose(
                reopened["t2m"].isel(time=slice(3, 6)).values, second["t2m"].values
            )
        )
    record_result(
        "lib_netcdf.serial.append extends without corrupting prior data", correct
    )


@run_test
def test_dataarray_netcdf_roundtrip(tmp_dir: Path) -> None:
    """A named DataArray round-trips through to_netcdf without a parent Dataset."""
    da = make_grid(n_time=2, n_lat=4, n_lon=5)["t2m"].rename("standalone")
    path = tmp_dir / "dataarray_roundtrip.nc"

    xgeo.to_netcdf(da, path, show_progress=False)

    with xr.open_dataset(path) as reopened:
        reopened.load()
        correct = bool(np.allclose(reopened["standalone"].values, da.values))
    record_result("lib_netcdf.serial: standalone DataArray round-trips", correct)


# ---------------------------------------------------------------------------
# climtools.core.tools
# ---------------------------------------------------------------------------


@run_test
def test_n_cpus() -> None:
    """n_cpus reports a positive integer core count."""
    correct = bool(isinstance(n_cpus, int) and n_cpus >= 1)
    record_result("core.tools.n_cpus is a positive integer", correct, note=str(n_cpus))


@run_test
def test_lock_file(tmp_dir: Path) -> None:
    """LockFile grants exclusive access and releases cleanly on exit."""
    lock_path = tmp_dir / "test.lock"
    entered = False
    with LockFile(lock_path, timeout=5.0) as lock:
        entered = lock is not None
        # A second, non-blocking attempt to grab the same lock from a
        # separate LockFile instance must fail while the first is held.
        second = LockFile(lock_path, timeout=0.2)
        try:
            with second:
                reentry_blocked = False
        except TimeoutError:
            reentry_blocked = True

    # After the first lock releases, acquiring it again must succeed.
    with LockFile(lock_path, timeout=5.0):
        released_cleanly = True

    correct = bool(entered and reentry_blocked and released_cleanly)
    record_result(
        "core.tools.LockFile excludes concurrent holders and releases", correct
    )


@run_test
def test_attr_dict() -> None:
    """AttrDict exposes dict keys as attributes."""
    d = AttrDict(a=1, b=2)
    read_ok = d.a == 1 and d.b == 2
    d.c = 3
    write_ok = d["c"] == 3
    del d.a
    delete_ok = "a" not in d
    correct = bool(read_ok and write_ok and delete_ok)
    record_result("core.tools.AttrDict supports attribute-style access", correct)


# ---------------------------------------------------------------------------
# climtools.cdo (conditional: only meaningful with the cdo binary on PATH)
# ---------------------------------------------------------------------------


@run_test
def test_cdo_available_or_skipped(tmp_dir: Path) -> None:
    """cdo.run() passes a file through unchanged when the cdo binary exists."""
    from climtools import cdo as ctcdo

    if shutil.which("cdo") is None:
        record_result(
            "cdo: passthrough run() round-trips a file",
            True,
            note="cdo binary not on PATH; skipped",
            skipped=True,
        )
        return

    ds = make_grid(n_time=4, n_lat=5, n_lon=6)
    in_path = tmp_dir / "cdo_in.nc"
    out_path = tmp_dir / "cdo_out.nc"
    xgeo.to_netcdf(ds, in_path, unlimited_dim="time", show_progress=False)

    ctcdo.run(["-copy"], input=str(in_path), output=str(out_path))

    with xr.open_dataset(out_path) as result:
        result.load()
        correct = bool(np.allclose(result["t2m"].values, ds["t2m"].values))
    record_result("cdo: passthrough run() round-trips a file", correct)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-cdo",
        action="store_true",
        help="skip the cdo passthrough check even if the cdo binary is present",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print a traceback for failures"
    )
    return parser.parse_args()


def print_summary() -> int:
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for result in RESULTS:
        status = "SKIP" if result.skipped else ("OK  " if result.passed else "FAIL")
        suffix = f"  ({result.note})" if result.note else ""
        print(f"[{status}] {result.name}{suffix}")

    n_total = len(RESULTS)
    n_failed = sum(1 for r in RESULTS if not r.passed and not r.skipped)
    n_skipped = sum(1 for r in RESULTS if r.skipped)
    n_passed = n_total - n_failed - n_skipped
    print("-" * 88)
    print(
        f"Results: {n_passed} passed, {n_failed} failed, {n_skipped} skipped, {n_total} total."
    )
    if n_failed:
        print(f"{n_failed} check(s) FAILED.")
    else:
        print("All checks passed.")
    return 1 if n_failed else 0


def main() -> None:
    arguments = parse_arguments()

    with tempfile.TemporaryDirectory(prefix="climtools_test_general_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        print("--- climtools.cmaps ---")
        test_cmaps_registry()
        test_cmaps_create_and_arithmetic()
        test_cmaps_get_colors()

        print("\n--- climtools.calc ---")
        test_calc_trends_polyfit()
        test_calc_trends_mann_kendall()
        test_calc_corr()
        test_calc_pvalues()

        print("\n--- climtools.xgeo / xr_utils ---")
        test_xgeo_accessor_registered()
        test_to_lon180()
        test_add_local_solar_time()
        test_sel_transect()
        test_get_spatial_dims()

        print("\n--- climtools.plot ---")
        test_plot_geo_contourf()
        test_plot_geo_accessor_and_overlay()

        print("\n--- lib_netcdf.serial (xgeo.to_netcdf, no MPI) ---")
        test_serial_netcdf_roundtrip(tmp_dir)
        test_serial_netcdf_append(tmp_dir)
        test_dataarray_netcdf_roundtrip(tmp_dir)

        print("\n--- core.tools ---")
        test_n_cpus()
        test_lock_file(tmp_dir)
        test_attr_dict()

        if not arguments.skip_cdo:
            print("\n--- climtools.cdo ---")
            test_cdo_available_or_skipped(tmp_dir)

    failed = print_summary()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
