from __future__ import annotations

import gc
import logging
import shutil
import subprocess
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from tracker import run_pyflextrkr

from climtools.xgeo import DaskProgressBar, remap, write_netcdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger("SPATIAL COMPOSITES")

home = Path("/users/jkodero")
gfdl_shield = home / "research/models/gfdl_shield"
data_store = gfdl_shield / "archive"
final_dir = gfdl_shield / "analysis/spatial_composites"
tmp_dir = home / "jobtmp/data/002/spatial_composites"


n_cpus = 24

SMC_CLIMO = {}


def get_smc_climo(ds: xr.Dataset) -> xr.DataArray:
    months = np.unique(ds["time"].dt.month.values)

    path = gfdl_shield / "src/fix/era5/sm_monthly_1950_2025.nc"
    smc = xr.open_dataset(path, engine="netcdf4")

    smc = smc.sel(time=smc.time.dt.month.isin(months))
    smc = smc.isel(zaxis_1=0, drop=True)
    smc = smc.mean(dim="time", skipna=True).squeeze(drop=True)
    smc = remap(smc, ds, method="bilinear")["smc"]

    if smc.dtype != ds["soilw1"].dtype:
        smc = smc.astype(ds["soilw1"].dtype)

    smc.attrs["long_name"] = "Climatological Soil Moisture"
    smc.attrs["units"] = ds["soilw1"].attrs["units"]
    SMC_CLIMO["soilw1_climo"] = smc

    # Climatological soil moisture gradient magnitude
    valid_land = smc < 0.95
    d_dlat = smc.where(valid_land).differentiate("lat")
    d_dlon = smc.where(valid_land).differentiate("lon")
    soilw1_cgm = (d_dlat**2 + d_dlon**2) ** 0.5
    soilw1_cgm.attrs["long_name"] = "Climatological Soil Moisture Gradient Magnitude"
    soilw1_cgm.attrs["units"] = smc.attrs["units"] + "/deg"
    SMC_CLIMO["soilw1_cgm"] = soilw1_cgm

    logger.info("Loaded and remapped soil moisture climatology")


def add_dry_fields(
    ds: xr.Dataset,
    data_var: str,
    cv_threshold: float,
    nr_threshold: float,
    window_before: int,
) -> xr.Dataset:
    """
    Add four per-cell boolean dryness fields on the native grid.

    cv_dry     : pr at or below cv_threshold at the current step.
    nr_dry     : pr at or below nr_threshold at the current step.
    pre_cv_dry : cv_dry true at every one of the window_before steps strictly
                 before the current step.
    pre_nr_dry : nr_dry true at every one of the window_before steps strictly
                 before the current step.

    pre_nr_dry implies pre_cv_dry since nr_threshold < cv_threshold. The leading
    window_before steps have an incomplete antecedent window and are set False.
    """
    pr = ds[data_var]
    cv_dry = pr <= cv_threshold
    nr_dry = pr <= nr_threshold

    def _pre(dry: xr.DataArray) -> xr.DataArray:
        rolled = (
            dry.astype("float32")
            .rolling(time=window_before, min_periods=window_before)
            .min()
            .shift(time=1)
        )
        return rolled >= 1.0

    ds["cv_dry"] = cv_dry
    ds["nr_dry"] = nr_dry
    ds["pre_cv_dry"] = _pre(cv_dry)
    ds["pre_nr_dry"] = _pre(nr_dry)
    for name in ("cv_dry", "nr_dry", "pre_cv_dry", "pre_nr_dry"):
        ds[name].attrs["units"] = "1"

    # Continuous run lengths along time. Data is hourly, so steps equal hours.
    wet = ~nr_dry
    cw = wet.cumsum("time")
    ds["wet_run"] = cw - cw.where(nr_dry).ffill("time").fillna(
        0
    )  # hours raining continuously up to t
    cd = nr_dry.cumsum("time")
    dry_run = cd - cd.where(wet).ffill("time").fillna(0)
    # antecedent dry hours before the wet run at t
    ds["dry_before"] = dry_run.where(nr_dry).shift(time=1).ffill("time")
    ds["wet_run"].attrs["units"] = "hours"
    ds["dry_before"].attrs["units"] = "hours"
    return ds


def derived_vars(ds: xr.Dataset, smc: dict) -> xr.Dataset:

    # Add climatological soil moisture if not already present.
    ds["soilw1_climo"] = smc["soilw1_climo"]
    ds["soilw1_cgm"] = smc["soilw1_cgm"]

    # Temperature gradient (units: K per degree, since lat/lon are in degrees)
    ds["dT_dlat"] = ds["t"].differentiate("lat")
    ds["dT_dlon"] = ds["t"].differentiate("lon")
    ds["dT_dlat"].attrs["long_name"] = "Meridional Temperature Gradient"
    ds["dT_dlat"].attrs["units"] = "K/deg"
    ds["dT_dlon"].attrs["long_name"] = "Zonal Temperature Gradient"
    ds["dT_dlon"].attrs["units"] = "K/deg"

    # Soil moisture gradient magnitude
    valid_land = ds["soilw1"] < 0.95
    dsoilw1_dlat = ds["soilw1"].where(valid_land).differentiate("lat")
    dsoilw1_dlon = ds["soilw1"].where(valid_land).differentiate("lon")
    ds["soilw1_gm"] = (dsoilw1_dlat**2 + dsoilw1_dlon**2) ** 0.5
    ds["soilw1_gm"].attrs["long_name"] = "Soil Moisture Gradient Magnitude"
    ds["soilw1_gm"].attrs["units"] = ds["soilw1"].attrs["units"] + "/deg"

    return ds


def _object_peaks(
    pr2d: xr.DataArray,
    connectivity: int,
    threshold: float,
    edge_pad: int,
    merge_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (peak_y, peak_x, obj_labels, obj_mean_pr, obj_max_pr) for connected pr >= threshold objects.

    If merge_cells > 0, precipitation objects separated by small dry gaps are
    grouped before labeling. The returned obj_labels are assigned only on the
    original pr >= threshold mask, so dry bridge cells are not treated as
    precipitating object cells.

    obj_mean_pr is the mean precipitation over each object's original
    pr >= threshold cells only.
    """
    pr2d = np.asarray(pr2d.values)
    ny, nx = pr2d.shape

    mask = pr2d >= threshold
    if not mask.any():
        empty_int = np.empty(0, dtype=int)
        empty_float = np.empty(0, dtype=float)
        obj_labels = np.zeros_like(pr2d, dtype=int)
        return empty_int, empty_int, obj_labels, empty_float, empty_float

    structure = ndimage.generate_binary_structure(2, connectivity)

    if merge_cells > 0:
        grouped_mask = ndimage.binary_dilation(
            mask,
            structure=structure,
            iterations=int(merge_cells),
        )
        grouped_labels, _ = ndimage.label(grouped_mask, structure=structure)

        obj_labels = np.where(mask, grouped_labels, 0).astype(grouped_labels.dtype)
        obj_ids = np.unique(obj_labels[mask])
        obj_ids = obj_ids[obj_ids > 0]
    else:
        obj_labels, n_obj = ndimage.label(mask, structure=structure)
        obj_ids = np.arange(1, n_obj + 1)

    peaks = ndimage.maximum_position(pr2d, obj_labels, obj_ids)
    peaks = np.atleast_2d(np.asarray(peaks, dtype=int))

    peak_y, peak_x = peaks[:, 0], peaks[:, 1]

    obj_mean_pr = np.asarray(
        ndimage.mean(pr2d, labels=obj_labels, index=obj_ids),
        dtype=float,
    )
    interior = (
        (peak_y >= edge_pad)
        & (peak_y < ny - edge_pad)
        & (peak_x >= edge_pad)
        & (peak_x < nx - edge_pad)
    )

    peak_y = peak_y[interior]
    peak_x = peak_x[interior]
    obj_mean_pr = obj_mean_pr[interior]

    obj_max_pr = pr2d[peak_y, peak_x].astype(float)

    order = np.argsort(-obj_max_pr, kind="stable")
    peak_y = peak_y[order]
    peak_x = peak_x[order]
    obj_max_pr = obj_max_pr[order]
    obj_mean_pr = obj_mean_pr[order]

    return peak_y, peak_x, obj_labels, obj_mean_pr, obj_max_pr


def _window_regrid(
    frame: xr.Dataset,
    lat0: float,
    lon0: float,
    offsets: np.ndarray,
    radius_km: float,
    method: str,
) -> xr.Dataset:
    """Resample a single-time frame onto an equidistant km grid about (lat0, lon0).

    Target offsets are mapped to lat/lon by the inverse equirectangular relation
    with cos(lat) evaluated per target row, then interpolated. Target points
    outside the model domain return NaN.
    """
    n = offsets.size
    deg = 180.0 / np.pi
    lat1d = lat0 + (offsets / radius_km) * deg  # (n,) over y_off
    lat_grid = np.repeat(lat1d[:, None], n, axis=1)  # (y_off, x_off)
    coslat = np.cos(np.deg2rad(lat_grid))
    lon_grid = lon0 + (offsets[None, :] / (radius_km * coslat)) * deg

    lat_da = xr.DataArray(lat_grid, dims=("y_off", "x_off"))
    lon_da = xr.DataArray(lon_grid, dims=("y_off", "x_off"))

    pad = 0.1
    sub = frame.sel(
        lat=slice(float(lat_grid.min()) - pad, float(lat_grid.max()) + pad),
        lon=slice(float(lon_grid.min()) - pad, float(lon_grid.max()) + pad),
    )
    win = sub.interp(
        lat=lat_da,
        lon=lon_da,
        method=method,
        kwargs={"bounds_error": False, "fill_value": np.nan},
    )
    win = win.assign_coords(y_off=offsets, x_off=offsets)
    win = win.drop_vars(["lat", "lon"], errors="ignore")
    return win


def object_peaks_wrapper(
    input_dir, it, data_var, connectivity, threshold, edge_pad, merge_cells
):
    with xr.open_dataset(input_dir / f"{it}") as ds:
        ds = ds.drop_vars("time", errors="ignore")
        pr2d = ds[data_var].squeeze(drop=True).load()
    return _object_peaks(pr2d, connectivity, threshold, edge_pad, merge_cells)


def build_object_windows_for_time(
    it: int,
    input_path: Path,
    output_path: Path,
    peak_y: np.ndarray,
    peak_x: np.ndarray,
    obj_labels: np.ndarray,
    obj_mean_pr: np.ndarray,
    obj_max_pr: np.ndarray,
    n_obj: int,
    has_current_pr: bool,
    c_it: int,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    offsets: np.ndarray,
    radius_km: float,
    interp_method: str,
    template: xr.Dataset,
    data_var: str = "pr",
) -> tuple[
    int,
    Path,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build all object-centered regridded windows for one time index.

    The dataset is opened inside the worker process. The returned xarray object
    is loaded before return so it no longer depends on an open file handle.
    """

    frame: xr.Dataset = None
    object_var: str = "pr_obj"
    with xr.open_dataset(input_path) as ds:
        frame = ds.squeeze(drop=True).drop_vars("time", errors="ignore").load()

    label_data = frame[data_var].copy(deep=True)

    lat_row = np.full(n_obj, np.nan)
    lon_row = np.full(n_obj, np.nan)
    mean_pr_row = np.full(n_obj, np.nan)
    max_pr_row = np.full(n_obj, np.nan)
    center_max_pr_row = np.full(n_obj, np.nan)
    center_mean_pr_row = np.full(n_obj, np.nan)
    itime_row = np.full(n_obj, -1, dtype=int)
    cv_dry = np.full(n_obj, False, dtype=bool)
    nr_dry = np.full(n_obj, False, dtype=bool)
    pre_cv_dry = np.full(n_obj, False, dtype=bool)
    pre_nr_dry = np.full(n_obj, False, dtype=bool)
    dry_before = np.full(n_obj, np.nan)
    wet_run = np.full(n_obj, np.nan)
    template[object_var] = template[data_var]

    windows: list[xr.Dataset] = []

    for k in range(n_obj):
        if k < peak_y.size:
            lat0 = float(lat_vals[peak_y[k]])
            lon0 = float(lon_vals[peak_x[k]])

            if has_current_pr:
                object_label = int(obj_labels[peak_y[k], peak_x[k]])

                object_mask = xr.DataArray(
                    obj_labels == object_label,
                    dims=("lat", "lon"),
                    coords={"lat": frame["lat"], "lon": frame["lon"]},
                )
                frame[object_var] = label_data.where(object_mask, 0)

                m = object_mask.values
                if m.any():
                    py, px = int(peak_y[k]), int(peak_x[k])
                    r = 1  # half-width in cells, so a (2r+1) square at the centre
                    ys = slice(max(py - r, 0), py + r + 1)
                    xs = slice(max(px - r, 0), px + r + 1)
                    cv_dry[k] = np.median(frame["cv_dry"].values[ys, xs]) >= 0.5
                    nr_dry[k] = np.median(frame["nr_dry"].values[ys, xs]) >= 0.5
                    pre_cv_dry[k] = np.median(frame["pre_cv_dry"].values[ys, xs]) >= 0.5
                    pre_nr_dry[k] = np.median(frame["pre_nr_dry"].values[ys, xs]) >= 0.5
                    dry_before[k] = np.nanmedian(frame["dry_before"].values[ys, xs])
                    wet_run[k] = np.nanmedian(frame["wet_run"].values[ys, xs])

            else:
                frame[object_var] = xr.full_like(label_data, np.nan)

            window = _window_regrid(
                frame,
                lat0,
                lon0,
                offsets,
                radius_km,
                interp_method,
            )

            windows.append(window)

            lat_row[k] = lat0
            lon_row[k] = lon0
            mean_pr_row[k] = float(obj_mean_pr[k]) if has_current_pr else np.nan
            max_pr_row[k] = float(obj_max_pr[k]) if has_current_pr else np.nan
            center_max_pr_row[k] = float(obj_max_pr[k])
            center_mean_pr_row[k] = float(obj_mean_pr[k])
            itime_row[k] = c_it
        else:
            windows.append(template)

    out = xr.concat(windows, dim="object")

    for v in out.data_vars:
        out[v] = out[v].astype(np.float32)

    out.to_netcdf(output_path, format="NETCDF4")
    return (
        it,
        output_path,
        lat_row,
        lon_row,
        mean_pr_row,
        max_pr_row,
        center_mean_pr_row,
        center_max_pr_row,
        itime_row,
        cv_dry,
        nr_dry,
        pre_cv_dry,
        pre_nr_dry,
        dry_before,
        wet_run,
    )


def _save_one_step(input_path: Path, i: int, out_dir: Path, aux_path: Path) -> int:
    with xr.open_dataset(input_path) as ds, xr.open_dataset(aux_path) as aux:
        out = (
            ds.isel(time=i)
            .drop_vars("time", errors="ignore")
            .sortby("lat")
            .sortby("lon")
            .transpose(..., "lat", "lon")
            .load()
        )
        out = derived_vars(out, SMC_CLIMO)
        a = (
            aux.isel(time=i)
            .drop_vars("time", errors="ignore")
            .sortby("lat")
            .sortby("lon")
            .load()
        )
        for name in (
            "cv_dry",
            "nr_dry",
            "pre_cv_dry",
            "pre_nr_dry",
            "dry_before",
            "wet_run",
        ):
            out[name] = a[name]
        out.to_netcdf(out_dir / f"{i}", format="NETCDF4")
    return i


def slice_and_save(path: Path, out_dir: Path, ntime: int, aux_path: Path) -> None:
    tasks = [
        dask.delayed(_save_one_step)(path, i, out_dir, aux_path) for i in range(ntime)
    ]
    with DaskProgressBar(description="Saving time steps"):
        dask.compute(*tasks, scheduler="processes", num_workers=n_cpus)
    logger.info("Sliced and saved %d time steps", ntime)


def load_initial_data(
    input_root,
    in_fname,
    half_extent_km,
    dx_km,
    tmp_in,
    cv_threshold,
    nr_threshold,
    window_before,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    path = input_root / "case" / in_fname
    aux_path = tmp_in.parent / "dry.nc"
    with xr.open_dataset(path) as ds:
        ds = ds.sortby("lat").sortby("lon")
        get_smc_climo(ds)
        ds = add_dry_fields(ds, "pr", cv_threshold, nr_threshold, window_before)
        aux = ds[
            ["cv_dry", "nr_dry", "pre_cv_dry", "pre_nr_dry", "dry_before", "wet_run"]
        ]
        aux.to_netcdf(aux_path, format="NETCDF4")

        half = int(round(half_extent_km / dx_km))
        offsets = (np.arange(-half, half + 1) * dx_km).astype(float)

        ntime = ds.sizes["time"]
        lat_vals = ds["lat"].values
        lon_vals = ds["lon"].values
        time_vals = ds["time"].values

        frame0 = (
            ds.isel(time=0)
            .drop_vars("time", errors="ignore")
            .sortby("lat")
            .sortby("lon")
            .transpose(..., "lat", "lon")
            .load()
        )

    slice_and_save(path, tmp_in, ntime, aux_path)
    return frame0, lat_vals, lon_vals, time_vals, offsets, ntime


def spatial_composites(
    input_root,
    output_root,
    in_fname,
    data_var="pr",
    threshold=1,
    dx_km=3.0,
    half_extent_km=498.0,
    connectivity=2,
    radius_km=6371.0,
    interp_method="linear",
    edge_pad=10,
    merge_cells=2,
    nr_threshold=0.1,
    window_before=24,
) -> None:
    """Build storm-centered spatial composites and an event-metadata CSV.

    Parameters
    ----------
    input_root, output_root : Path
        Input case directory root and output directory root.
    in_fname : str
        File name inside ``input_root / "case"``.
    data_var : str
        Precipitation variable name; must be 2D (lat, lon) after time selection.
    threshold : float
        Minimum pr value to consider a cell as precipitating; the default of 1 mm/hr is a common threshold for convective precipitation.
    dx_km : float
        Target offset spacing in kilometres (the composite resolution).
    half_extent_km : float
        Half-width of the composite window in kilometres; the window spans
        ``[-half_extent_km, +half_extent_km]`` in x and y.
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity in object labeling.
    radius_km : float
        Planetary radius used in the lat/lon to km mapping. 6371 for Earth; set
        the appropriate value for other bodies.
    interp_method : str
        Interpolation method passed to ``Dataset.interp`` ("linear", "nearest",
        "cubic"). See note on precipitation below.

    edge_pad : int
        Minimum number of grid cells between any object peak and the domain edge; objects peaking within this distance of the edge are dropped, so they are
        never used as window centres. This avoids windows with large NaN regions that occur when the centre is near the edge
    merge_cells : int
        Minimum number of grid cells to merge adjacent objects.
    threshold : float
        Minimum precipitation value to consider a cell as precipitating.
    nr_threshold : float
        Threshold for no-rain periods.
    window_before : int
        Number of dry steps required before the current step to consider the storm as having an antecedent dry period.

    Returns
    -------
    xr.Dataset or None
        Composite store with dims (time, ..., objects, y_off, x_off), or None
        when no finite pr > threshold exists anywhere in the file.

    Notes
    -----
    Object ``id`` is the within-time object index and does not track a storm
    across time.

    our precipitation metadata fields are written per (time, object):
    ``obj_max_pr`` is the local peak at the current step (NaN on antecedent steps,
    which have no local precipitation), and ``center_max_pr`` is the peak of the
    storm that defines the centre, taken at that storm's own detection time, so
    it is populated on antecedent steps and lets composites be stratified by the
    eventual storm intensity. ``center_time``, ``center_time_idx``, and
    ``center_from_future_time`` record where the centre came from.

    The equirectangular mapping is a tangent-plane approximation, accurate for
    windows of order 1000 km away from the poles but degrading where cos(lat) is
    small. Longitude wrapping across the dateline or 0/360 boundary is not
    handled. ``linear`` interpolation is monotone and preserves positivity but
    smooths precipitation maxima; for amount or threshold statistics consider
    ``nearest`` for pr or a conservative remap (Jones, 1999).
    """

    # these temp dirs should node local
    tmp_base = output_root / "tmp"
    shutil.rmtree(tmp_base, ignore_errors=True)

    tmp_out = tmp_base / "out"
    tmp_in = tmp_base / "in"

    tmp_out.mkdir(parents=True)
    tmp_in.mkdir(parents=True)

    frame0, lat_vals, lon_vals, time_vals, offsets, ntime = load_initial_data(
        input_root,
        in_fname,
        half_extent_km,
        dx_km,
        tmp_in,
        threshold,
        nr_threshold,
        window_before,
    )
    logger.info("Loaded dataset with %d time steps", ntime)

    peaks_per_time: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    tasks = [
        (tmp_in, it, data_var, connectivity, threshold, edge_pad, merge_cells)
        for it in range(ntime)
    ]
    delayed_tasks = [dask.delayed(object_peaks_wrapper)(*task) for task in tasks]
    with DaskProgressBar(description="Detecting objects"):
        peaks_per_time = dask.compute(
            *delayed_tasks, scheduler="processes", num_workers=n_cpus
        )

    logger.info("Running PyFLEXTRKR linking/statistics from external labels")
    track_lookup = run_pyflextrkr(
        tmp_base=tmp_base,
        tmp_frames=tmp_in,
        peaks_per_time=peaks_per_time,
        data_var=data_var,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        time_vals=time_vals,
        dx_km=3.26,  # km
        overlap_threshold=0.3,
    )
    logger.info(f"Completed PyFLEXTRKR tracking with {len(track_lookup)} events.")

    # Resolve centres: borrow the next precipitating step for empty steps.
    resolved: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = [None] * ntime
    center_time_idx = np.full(ntime, -1, dtype=int)
    next_valid = None
    next_valid_it = -1
    for it in range(ntime - 1, -1, -1):
        if peaks_per_time[it][0].size > 0:
            next_valid = peaks_per_time[it]
            next_valid_it = it
        if peaks_per_time[it][0].size > 0:
            resolved[it] = peaks_per_time[it]
            center_time_idx[it] = it
        elif next_valid is not None:
            resolved[it] = next_valid
            center_time_idx[it] = next_valid_it
        else:
            resolved[it] = (
                np.empty(0, dtype=int),  # peak_y
                np.empty(0, dtype=int),  # ix
                peaks_per_time[it][2],  # obj_labels
                np.empty(0, dtype=float),  # obj_mean_pr
                np.empty(0, dtype=float),  # obj_max_pr
            )
            center_time_idx[it] = -1

    n_obj = max((r[0].size for r in resolved), default=0)
    if n_obj == 0:
        return None

    logger.info(f"Maximum number of objects in any time step: {n_obj}")
    # NaN template for object slots a step does not fill.
    iy0, ix0, _, _, _ = next(r for r in resolved if r[0].size > 0)
    template = (
        _window_regrid(
            frame0,
            float(lat_vals[iy0[0]]),
            float(lon_vals[ix0[0]]),
            offsets,
            radius_km,
            interp_method,
        )
        .where(False)
        .load()
    )

    center_lat = np.full((ntime, n_obj), np.nan)
    center_lon = np.full((ntime, n_obj), np.nan)
    obj_mean_pr = np.full((ntime, n_obj), np.nan)
    obj_max_pr = np.full((ntime, n_obj), np.nan)
    center_mean_pr = np.full((ntime, n_obj), np.nan)
    center_max_pr = np.full((ntime, n_obj), np.nan)
    center_time_idx_2d = np.full((ntime, n_obj), -1, dtype=int)
    nr_dry = np.full((ntime, n_obj), False, dtype=bool)
    cv_dry = np.full((ntime, n_obj), False, dtype=bool)
    pre_nr_dry = np.full((ntime, n_obj), False, dtype=bool)
    pre_cv_dry = np.full((ntime, n_obj), False, dtype=bool)
    dry_steps = np.full((ntime, n_obj), -1, dtype=int)
    dry_hours = np.full((ntime, n_obj), np.nan, dtype=float)
    dry_before = np.full((ntime, n_obj), np.nan)
    wet_run = np.full((ntime, n_obj), np.nan)

    per_time: list[Path | None] = [None] * ntime

    tasks = [
        (
            it,
            tmp_in / str(it),
            tmp_out / str(it),
            resolved[it][0],  # peak_y
            resolved[it][1],  # peak_x
            resolved[it][2],  # obj_labels
            resolved[it][3],  # obj_mean_pr
            resolved[it][4],  # obj_max_pr
            n_obj,
            peaks_per_time[it][0].size > 0,
            center_time_idx[it],
            lat_vals,
            lon_vals,
            offsets,
            radius_km,
            interp_method,
            template,
            data_var,
        )
        for it in range(ntime)
    ]

    logger.info(
        "Building windows for %d time steps with up to %d objects each",
        ntime,
        n_obj,
    )

    delayed_tasks = [
        dask.delayed(build_object_windows_for_time)(*task) for task in tasks
    ]

    with DaskProgressBar(description="Transforming to XY grid"):
        results = dask.compute(
            *delayed_tasks,
            scheduler="processes",
            num_workers=n_cpus,
        )

    for (
        it,
        output_path,
        lat_row,
        lon_row,
        mean_pr_row,
        max_pr_row,
        center_mean_pr_row,
        center_max_pr_row,
        center_time_idx_row,
        it_cv_dry,
        it_nr_dry,
        pre_cv_dry_row,
        pre_nr_dry_row,
        dry_before_row,
        wet_run_row,
    ) in results:
        per_time[it] = output_path
        center_lat[it, :] = lat_row
        center_lon[it, :] = lon_row
        obj_mean_pr[it, :] = mean_pr_row
        obj_max_pr[it, :] = max_pr_row
        center_mean_pr[it, :] = center_mean_pr_row
        center_max_pr[it, :] = center_max_pr_row
        center_time_idx_2d[it, :] = center_time_idx_row
        cv_dry[it, :] = it_cv_dry
        nr_dry[it, :] = it_nr_dry
        pre_cv_dry[it, :] = pre_cv_dry_row
        pre_nr_dry[it, :] = pre_nr_dry_row
        dry_before[it, :] = dry_before_row
        wet_run[it, :] = wet_run_row

    time_idx_2d = np.broadcast_to(
        np.arange(ntime)[:, None],
        center_time_idx_2d.shape,
    )

    valid_center = center_time_idx_2d >= 0

    dry_steps[valid_center] = (
        center_time_idx_2d[valid_center] - time_idx_2d[valid_center]
    )

    if np.issubdtype(time_vals.dtype, np.datetime64):
        dry_hours[valid_center] = (
            time_vals[center_time_idx_2d[valid_center]]
            - time_vals[time_idx_2d[valid_center]]
        ) / np.timedelta64(1, "h")

    not_dry_valid = valid_center & ~pre_cv_dry

    dry_steps[not_dry_valid] = 0
    dry_hours[not_dry_valid] = 0.0

    gc.collect()
    per_time = [x for x in per_time if x is not None]

    logger.info(f"Merging windows for {len(per_time)} time steps")

    result = xr.open_mfdataset(
        per_time,
        combine="nested",
        concat_dim="time",
        parallel=True,
        chunks="auto",
    )

    result = result.assign_coords(time=time_vals, object=np.arange(n_obj))

    # center_time as a data variable only when the time axis is datetime64,
    # which avoids casting issues for integer or cftime time coordinates.

    center_time_vals = np.full(
        (ntime, n_obj), np.datetime64("NaT"), dtype=time_vals.dtype
    )
    valid = center_time_idx_2d >= 0
    center_time_vals[valid] = time_vals[center_time_idx_2d[valid]]

    result = result.transpose("time", ..., "object", "y_off", "x_off")

    # Event metadata CSV.
    logger.info("Building event metadata CSV")
    rows = []
    for it in range(ntime):
        for k in range(n_obj):
            center_time_idx_val = int(center_time_idx_2d[it, k])
            has_center_object = center_time_idx_val >= 0

            rows.append(
                {
                    "time_idx": it,
                    "time": time_vals[it],
                    "lat": round(float(center_lat[it, k]), 3),
                    "lon": round(float(center_lon[it, k]), 3),
                    "object": k,
                    "dry_hrs_before": round(float(dry_before[it, k]), 3),
                    "continuous_rain": round(float(wet_run[it, k]), 3),
                    f"no_conv_rain_prev{window_before}h": bool(pre_cv_dry[it, k]),
                    f"no_rain_prev{window_before}h": bool(pre_nr_dry[it, k]),
                    "mean_pr": round(float(obj_mean_pr[it, k]), 3),
                    "max_pr": round(float(obj_max_pr[it, k]), 3),
                    "dry_steps": int(dry_steps[it, k]),
                    "dry_hours": round(float(dry_hours[it, k]), 3),
                    "future_rain": bool(center_time_idx_val > it),
                    "center_mean_pr": round(float(center_mean_pr[it, k]), 3),
                    "center_max_pr": round(float(center_max_pr[it, k]), 3),
                    "center_time": (
                        time_vals[center_time_idx_val] if has_center_object else pd.NaT
                    ),
                    "center_time_idx": (
                        center_time_idx_val if has_center_object else -1
                    ),
                }
            )
    events = pd.DataFrame(
        rows,
        columns=[
            "time_idx",
            "time",
            "lat",
            "lon",
            "object",
            "dry_hrs_before",
            "continuous_rain",
            f"no_conv_rain_prev{window_before}h",
            f"no_rain_prev{window_before}h",
            "mean_pr",
            "max_pr",
            "dry_steps",
            "dry_hours",
            "future_rain",
            "center_mean_pr",
            "center_max_pr",
            "center_time",
            "center_time_idx",
        ],
    )
    events["lead_steps"] = (events["center_time_idx"] - events["time_idx"]).astype(int)

    events = events.merge(
        track_lookup,
        how="left",
        on=["center_time_idx", "object"],
        validate="many_to_one",
    )

    events = events.sort_values(["time_idx", "time", "object"]).reset_index(drop=True)

    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "event.store.nc"
    csv_path = output_root / "event.store.csv"
    out_path.unlink(missing_ok=True)

    logger.info(f"Writing composite store to {out_path}")
    write_netcdf(
        out_path, result, unlimited_dim="time", format="NETCDF4", parallel=True
    )

    events.to_csv(csv_path, index=False)

    shutil.rmtree(tmp_base)
    logger.info(f"Finished writing file to {out_path}")


def main() -> None:

    experiments = [
        "C96.NESTED.R4x2.R2x1.CNTRL",
        "C96.NESTED.R4x2.R2x1.2SIGMA_DRY",
        "C96.NESTED.R4x2.R2x1.2SIGMA_WET",
        "C96.NESTED.R4x2.R2x1.CLIMO",
        "C96.NESTED.R4x2.R2x1.1SIGMA_DRY",
        "C96.NESTED.R4x2.R2x1.1SIGMA_WET",
    ]
    in_fname = "fv3_hist.nest04.nc"
    init_datetimes = [
        "2016072800Z",
        "2018052500Z",
        "2025092206Z",
        "2017062900Z",
        "2025071206Z",
    ]

    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(final_dir, ignore_errors=True)

    for init_date in init_datetimes:
        for exp in experiments:
            logger.info("Running %s %s", init_date, exp)
            input_root = data_store / init_date / exp
            output_root = tmp_dir / init_date / exp
            output_root.mkdir(parents=True, exist_ok=True)

            # fv3_hist.nest04.nc is the input file, 0.03 deg in resolution, with 50 variables

            spatial_composites(
                input_root,
                output_root,
                in_fname,
                data_var="pr",
                dx_km=3.0,
                half_extent_km=198,
                radius_km=6371.0,
                connectivity=2,
                threshold=5,  # mm/hr
                interp_method="linear",
                merge_cells=1,
                window_before=24,
                nr_threshold=0.1,
            )

            final_path = final_dir / init_date / exp
            final_path.parent.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                ["rsync", "-a", "--delete", f"{output_root}/", f"{final_path}/"],
                check=True,
            )
            logger.info("Finished %s %s", init_date, exp)
            gc.collect()

        shutil.rmtree(output_root)
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
