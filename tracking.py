from __future__ import annotations

import logging
import operator
import os
import sys
import tempfile
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr
import yaml
from scipy.ndimage import map_coordinates

_EARTH_RADIUS_KM = 6371.0
_TRACKING_LOG = "tracking.log"

# PyFLEXTRKR parallelises across input files. Writing one file per time step is
# the documented input layout and is what allows Step 1 (identify features) and
# Step 2 (link pairs) to use more than one worker. Set to False to restore the
# previous single-file behaviour if a reference comparison requires it.
_ONE_FILE_PER_TIME = True

_LOGGER = logging.getLogger(__name__)

comparators: dict[str, Callable[[Any, Any], Any]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


@dataclass(frozen=True)
class TrackedFeatures:
    track_mask: xr.DataArray
    center_lat: xr.DataArray
    center_lon: xr.DataArray
    center_value: xr.DataArray
    peak_lat: xr.DataArray
    peak_lon: xr.DataArray
    peak: xr.DataArray
    track_time: xr.DataArray


def cpu_count() -> int:
    """Number of CPUs this process is actually allowed to run on (Linux)."""
    return max(1, len(os.sched_getaffinity(0)))


def _available_bytes() -> int:
    """Physically available memory in bytes, or 0 if it cannot be determined."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def _scratch_root() -> str | None:
    """Prefer tmpfs so the PyFLEXTRKR round trip never touches physical disk."""
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return str(shm)
    return None


@contextmanager
def tracker_env() -> Generator[Path]:
    """Write PyFLEXTRKR output to tracking.log and forward only errors."""
    log_path = Path.cwd() / _TRACKING_LOG
    root_logger = logging.getLogger()
    pyflex_logger = logging.getLogger("pyflextrkr")

    pyflex_state = (
        pyflex_logger.disabled,
        pyflex_logger.propagate,
        pyflex_logger.level,
        list(pyflex_logger.handlers),
    )
    child_states: dict[str, tuple[bool, bool, int, list[logging.Handler]]] = {}
    for name, logger in logging.root.manager.loggerDict.items():
        if name.startswith("pyflextrkr.") and isinstance(logger, logging.Logger):
            child_states[name] = (
                logger.disabled,
                logger.propagate,
                logger.level,
                list(logger.handlers),
            )

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    console_stream = os.fdopen(
        os.dup(stderr_fd),
        "w",
        encoding="utf-8",
        buffering=1,
    )
    root_stream_states: list[tuple[logging.StreamHandler[Any], Any]] = []
    filtered_handlers: list[logging.Handler] = []
    fallback_handler: logging.Handler | None = None

    def main_log_filter(record: logging.LogRecord) -> bool:
        return (
            not record.name.startswith("pyflextrkr") or record.levelno >= logging.ERROR
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix="pyflextrkr_",
            dir=_scratch_root(),
        ) as temporary_directory:
            work = Path(temporary_directory)

            with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
                formatter = logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s"
                )
                file_handler = logging.StreamHandler(log_stream)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)

                for handler in root_logger.handlers:
                    handler.addFilter(main_log_filter)
                    filtered_handlers.append(handler)
                    if isinstance(
                        handler, logging.StreamHandler
                    ) and handler.stream in (sys.stdout, sys.stderr):
                        root_stream_states.append((handler, handler.stream))
                        handler.setStream(console_stream)

                if not root_logger.handlers:
                    fallback_handler = logging.StreamHandler(console_stream)
                    fallback_handler.setLevel(logging.ERROR)
                    fallback_handler.setFormatter(formatter)
                    fallback_handler.addFilter(main_log_filter)
                    root_logger.addHandler(fallback_handler)

                pyflex_logger.disabled = False
                pyflex_logger.propagate = True
                pyflex_logger.setLevel(logging.DEBUG)
                pyflex_logger.handlers = [file_handler]

                for name in child_states:
                    logger = logging.getLogger(name)
                    logger.disabled = False
                    logger.propagate = True
                    logger.setLevel(logging.NOTSET)
                    logger.handlers = []

                log_stream.write("\n--- PyFLEXTRKR run ---\n")
                log_stream.flush()

                with redirect_stdout(log_stream), redirect_stderr(log_stream):
                    os.dup2(log_stream.fileno(), 1)
                    os.dup2(log_stream.fileno(), 2)
                    try:
                        yield work
                    finally:
                        log_stream.flush()
                        os.dup2(stdout_fd, 1)
                        os.dup2(stderr_fd, 2)
    finally:
        for handler, stream in root_stream_states:
            handler.setStream(stream)
        for handler in filtered_handlers:
            handler.removeFilter(main_log_filter)
        if fallback_handler is not None:
            root_logger.removeHandler(fallback_handler)
            fallback_handler.close()

        (
            pyflex_logger.disabled,
            pyflex_logger.propagate,
            pyflex_logger.level,
            pyflex_logger.handlers,
        ) = pyflex_state
        for name, state in child_states.items():
            logger = logging.getLogger(name)
            logger.disabled, logger.propagate, logger.level, logger.handlers = state

        console_stream.close()
        os.close(stdout_fd)
        os.close(stderr_fd)


def _read_feature_numbers(
    cloudid_files: Sequence[str | Path],
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Read every cloudid feature mask exactly once.

    The masks are needed twice, first to size the PyFLEXTRKR allocations and
    again to map track numbers back onto the native grid. Reading them once and
    caching halves the number of netCDF opens.
    """
    labels: list[np.ndarray] = []
    position: dict[str, int] = {}
    for index, filename in enumerate(cloudid_files):
        with xr.open_dataset(filename, mask_and_scale=False) as feature_ds:
            labels.append(
                np.asarray(
                    feature_ds["feature_number"].squeeze("time", drop=True).values,
                    dtype=np.int64,
                )
            )
        position[Path(filename).name] = index
    return labels, position


def _tracking_capacity(
    label_frames: Sequence[np.ndarray],
    overlap_threshold: float,
) -> tuple[int, int]:
    """Derive PyFLEXTRKR allocation sizes from identified feature masks."""
    max_features = 0
    max_links = 0
    previous: np.ndarray | None = None

    for labels in label_frames:
        positive = labels[labels > 0]
        max_features = max(
            max_features,
            int(positive.max()) if positive.size else 0,
        )

        if previous is not None:
            valid = (previous > 0) & (labels > 0)
            if np.any(valid):
                reference_ids = previous[valid]
                candidate_ids = labels[valid]
                max_reference = int(reference_ids.max())
                max_candidate = int(candidate_ids.max())
                stride = max_candidate + 1
                pair_codes = reference_ids * stride + candidate_ids
                unique_codes, overlap_pixels = np.unique(
                    pair_codes,
                    return_counts=True,
                )
                pair_reference = unique_codes // stride
                pair_candidate = unique_codes % stride
                reference_pixels = np.bincount(
                    previous[previous > 0],
                    minlength=max_reference + 1,
                )
                candidate_pixels = np.bincount(
                    labels[labels > 0],
                    minlength=max_candidate + 1,
                )
                forward = (
                    overlap_pixels.astype(np.float64) / reference_pixels[pair_reference]
                    > overlap_threshold
                )
                backward = (
                    overlap_pixels.astype(np.float64) / candidate_pixels[pair_candidate]
                    > overlap_threshold
                )
                forward_counts = np.bincount(
                    pair_reference[forward],
                    minlength=max_reference + 1,
                )
                backward_counts = np.bincount(
                    pair_candidate[backward],
                    minlength=max_candidate + 1,
                )
                max_links = max(
                    max_links,
                    int(forward_counts.max(initial=0)),
                    int(backward_counts.max(initial=0)),
                )
        previous = labels

    return max(1, max_features + 1), max(1, max_links)


def run_pyflextrkr(
    ds: xr.Dataset,
    data_var: str,
    threshold: float,
    threshold_operator: str,
    overlap_threshold: float,
    fill_value: int,
    parallel: bool,
) -> xr.DataArray:
    """Run PyFLEXTRKR for one time-lat-lon field and return track IDs.

    ``parallel`` selects between serial execution and a Dask ``LocalCluster``
    sized to the CPUs this process is permitted to use. It is written to the
    generated configuration as the integer ``run_parallel`` key, alongside the
    companion ``nprocesses`` key that PyFLEXTRKR requires when
    ``run_parallel = 1``. The Dask distributed mode (``run_parallel = 2``) is
    deliberately not offered because it needs a scheduler JSON file supplied at
    process start, which is not reachable through this in-process API.
    """
    try:
        from pyflextrkr.ft_utilities import load_config, subset_files_timerange
        from pyflextrkr.gettracks import gettracknumbers
        from pyflextrkr.idfeature_driver import idfeature_driver
        from pyflextrkr.tracksingle_driver import tracksingle_driver
    except ImportError as exc:
        raise ImportError(
            "track_feature and get_relative_time require PyFLEXTRKR. "
        ) from exc

    if threshold_operator not in comparators:
        raise ValueError("threshold_operator must be one of '>', '>=', '<', or '<='.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if not np.isfinite(overlap_threshold) or not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be between 0 and 1 inclusive.")
    if fill_value >= 0:
        raise ValueError("fill_value must be negative.")
    if not isinstance(parallel, (bool, np.bool_)):
        raise TypeError("parallel must be a bool.")

    field = ds[data_var]
    if field.dims != ("time", "lat", "lon"):
        raise ValueError(
            f"{data_var!r} must have dimensions ('time', 'lat', 'lon'); got {field.dims!r}."
        )
    for coordinate in ("time", "lat", "lon"):
        if coordinate not in ds.coords:
            raise ValueError(f"Dataset is missing the {coordinate!r} coordinate.")

    times = np.asarray(ds["time"].values).astype("datetime64[ns]")
    if times.size < 2:
        raise ValueError("At least two time steps are required for feature tracking.")
    differences = np.diff(times).astype("timedelta64[ns]").astype(np.int64)
    if np.any(differences <= 0):
        raise ValueError("The 'time' coordinate must increase strictly monotonically.")
    time_resolution_hours = float(np.median(differences) / 3_600_000_000_000.0)

    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    dlat = float(np.median(np.abs(np.diff(lat)))) if lat.size > 1 else 0.0
    dlon = float(np.median(np.abs(np.diff(lon)))) if lon.size > 1 else 0.0
    mean_lat = float(np.nanmean(lat))
    dy_km = _EARTH_RADIUS_KM * np.deg2rad(dlat)
    dx_km = _EARTH_RADIUS_KM * np.cos(np.deg2rad(mean_lat)) * np.deg2rad(dlon)
    pixel_radius_km = float(np.nanmean([abs(dx_km), abs(dy_km)]))
    if not np.isfinite(pixel_radius_km) or pixel_radius_km <= 0.0:
        pixel_radius_km = 1.0

    selected = comparators[threshold_operator](field, threshold)
    tracking_field = xr.where(selected, np.float32(1.0), np.float32(0.0))
    tracking_field = tracking_field.where(np.isfinite(field), np.float32(0.0))
    tracking_input = tracking_field.to_dataset(name=data_var)
    tracking_input = tracking_input.transpose("time", "lat", "lon")

    first = times[0].astype("datetime64[m]")
    last = times[-1].astype("datetime64[m]")
    if times[-1] > last.astype("datetime64[ns]"):
        last += np.timedelta64(1, "m")
    startdate = np.datetime_as_string(first, unit="m").replace("-", "")
    startdate = startdate.replace(":", "").replace("T", ".")
    enddate = np.datetime_as_string(last, unit="m").replace("-", "")
    enddate = enddate.replace(":", "").replace("T", ".")

    def _input_stamp(value: np.datetime64) -> str:
        stamp = np.datetime_as_string(value, unit="s")
        return stamp.replace("-", "").replace(":", "").replace("T", ".")

    with tracker_env() as work:
        input_path = work / "input"
        input_path.mkdir()

        if _ONE_FILE_PER_TIME:
            for index in range(times.size):
                stamp = _input_stamp(times[index])
                tracking_input.isel(time=slice(index, index + 1)).to_netcdf(
                    input_path / f"input_{stamp}.nc"
                )
        else:
            stamp = _input_stamp(times[0])
            tracking_input.to_netcdf(input_path / f"input_{stamp}.nc")

        config_file = work / "config.yml"
        config = {
            # PyFLEXTRKR expects an integer here: 0 serial, 1 Dask LocalCluster,
            # 2 Dask distributed. The public API of this module takes a bool and
            # maps it onto the first two values.
            "run_parallel": int(bool(parallel)),
            "nprocesses": cpu_count() if parallel else 1,
            "input_format": "netcdf",
            "startdate": startdate,
            "enddate": enddate,
            "databasename": "input_",
            "time_format": "yyyymodd.hhmmss",
            "clouddata_path": f"{input_path}/",
            "root_path": str(work),
            "tracking_path_name": "tracking",
            "stats_path_name": "stats",
            "pixel_path_name": "pixel",
            "feature_type": "generic",
            "datatimeresolution": time_resolution_hours,
            "pixel_radius": pixel_radius_km,
            "area_method": "fixed",
            "time_dimname": "time",
            "x_dimname": "lon",
            "y_dimname": "lat",
            "time_coordname": "time",
            "x_coordname": "lon",
            "y_coordname": "lat",
            "field_varname": data_var,
            "label_method": "ndimage.label",
            "field_thresh": [0.5, 1.5],
            "min_size": 0.0,
            "R_earth": _EARTH_RADIUS_KM,
            "timegap": max(
                time_resolution_hours * 1.5,
                time_resolution_hours + 1e-6,
            ),
            "othresh": overlap_threshold,
            "auto_update_maxnclouds": True,
            "feature_varname": "feature_number",
            "nfeature_varname": "nfeatures",
            "featuresize_varname": "npix_feature",
            "fillval": fill_value,
        }
        config_file.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

        pyflex_config = load_config(str(config_file))
        idfeature_driver(pyflex_config)

        cloudid_files, _, _, _ = subset_files_timerange(
            pyflex_config["tracking_outpath"],
            pyflex_config["cloudid_filebase"],
            pyflex_config["start_basetime"],
            pyflex_config["end_basetime"],
        )
        if not cloudid_files:
            raise ValueError("PyFLEXTRKR produced no feature-identification files.")

        label_frames, label_position = _read_feature_numbers(cloudid_files)
        maxnclouds, nmaxlinks = _tracking_capacity(label_frames, overlap_threshold)
        pyflex_config["maxnclouds"] = maxnclouds
        pyflex_config["nmaxlinks"] = nmaxlinks

        tracksingle_driver(pyflex_config)
        tracknumbers_file = gettracknumbers(pyflex_config)

        with xr.open_dataset(
            tracknumbers_file,
            mask_and_scale=False,
        ) as track_numbers_ds:
            track_numbers = np.asarray(
                track_numbers_ds["track_numbers"]
                .squeeze("time", drop=True)
                .load()
                .values,
                dtype=np.int64,
            )
            basetimes = (
                track_numbers_ds["basetimes"].load().values.astype("datetime64[ns]")
            )

        tracking_outpath = Path(pyflex_config["tracking_outpath"])
        cloudid_filebase = pyflex_config["cloudid_filebase"]
        source_index = {
            int(value.astype(np.int64)): index
            for index, value in enumerate(times.astype("datetime64[ns]"))
        }
        mask = np.zeros(
            (ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"]),
            dtype=np.int64,
        )

        for file_index, base_time in enumerate(basetimes):
            base_time_ns = base_time.astype("datetime64[ns]")
            time_index = source_index.get(int(base_time_ns.astype(np.int64)))
            if time_index is None:
                continue

            stamp = np.datetime_as_string(base_time_ns, unit="s")
            stamp = stamp.replace("-", "").replace(":", "").replace("T", "_")
            cloudid_name = f"{cloudid_filebase}{stamp}.nc"

            cached = label_position.get(cloudid_name)
            if cached is not None:
                feature_number = label_frames[cached]
            else:
                with xr.open_dataset(
                    tracking_outpath / cloudid_name,
                    mask_and_scale=False,
                ) as cloudid_ds:
                    feature_number = np.asarray(
                        cloudid_ds["feature_number"].squeeze("time", drop=True).values,
                        dtype=np.int64,
                    )

            file_tracks = np.asarray(track_numbers[file_index], dtype=np.int64).ravel()
            file_tracks = np.where(file_tracks > 0, file_tracks, 0)
            lookup = np.zeros(file_tracks.size + 1, dtype=np.int64)
            lookup[1:] = file_tracks
            valid_feature = (feature_number >= 0) & (feature_number < lookup.size)
            mapped = np.zeros_like(feature_number, dtype=np.int64)
            mapped[valid_feature] = lookup[feature_number[valid_feature]]
            mask[time_index] = mapped

        return xr.DataArray(
            mask,
            dims=("time", "lat", "lon"),
            coords={"time": ds["time"], "lat": ds["lat"], "lon": ds["lon"]},
            name="track_mask",
        )


def _tracking_dimensions(
    field: xr.DataArray,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if field.dims[-2:] != ("lat", "lon"):
        raise ValueError(
            f"Tracking variable must end with ('lat', 'lon'); got {field.dims!r}."
        )
    leading_dims = field.dims[:-2]
    if "time" not in leading_dims:
        raise ValueError("Tracking variable must contain the 'time' dimension.")
    group_dims = tuple(dim for dim in leading_dims if dim != "time")
    return leading_dims, group_dims


def _build_track_mask(
    ds: xr.Dataset,
    data_var: str,
    threshold: float,
    threshold_operator: str,
    reference_data: dict[str, Any] | None,
    overlap_threshold: float,
    fill_value: int,
    parallel: bool,
) -> xr.DataArray:
    field = ds[data_var]
    _, group_dims = _tracking_dimensions(field)

    for coordinate in ("time", "lat", "lon"):
        if coordinate not in ds.coords:
            raise ValueError(f"Dataset is missing the {coordinate!r} coordinate.")
    if ds["lat"].dims != ("lat",) or ds["lon"].dims != ("lon",):
        raise ValueError("Coordinates 'lat' and 'lon' must be one-dimensional.")

    if reference_data is None:
        reference_field = field
    else:
        invalid = set(reference_data) - set(group_dims)
        if invalid:
            raise ValueError(
                f"reference_data may select only non-time leading dimensions; invalid keys: {sorted(invalid)!r}."
            )
        reference_field = field.sel(reference_data, drop=True)

    _, reference_group_dims = _tracking_dimensions(reference_field)
    canonical = reference_field.transpose(*reference_group_dims, "time", "lat", "lon")
    canonical_mask = np.zeros(canonical.shape, dtype=np.int64)
    group_shape = tuple(canonical.sizes[dim] for dim in reference_group_dims)

    for index in np.ndindex(*group_shape):
        indexers = dict(zip(reference_group_dims, index, strict=True))
        field_slice = canonical.isel(indexers, drop=True)
        mask_slice = run_pyflextrkr(
            field_slice.to_dataset(name=data_var),
            data_var,
            threshold,
            threshold_operator,
            overlap_threshold,
            fill_value,
            parallel,
        )
        canonical_mask[index + (slice(None), slice(None), slice(None))] = np.asarray(
            mask_slice.values,
            dtype=np.int64,
        )

    mask = xr.DataArray(
        canonical_mask,
        dims=canonical.dims,
        coords={dim: canonical[dim] for dim in canonical.dims},
        name="track_mask",
    )
    if reference_data is not None:
        mask, _ = xr.broadcast(mask, field)
    return mask.transpose(*field.dims)


def _frame_extrema(
    values_2d: np.ndarray,
    labels_2d: np.ndarray,
    center_on: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Locate the extremum of every labelled feature in a single frame.

    A single sort over the labelled pixels replaces one full-frame scan per
    feature. The flat pixel index is the last sort key, so ties are broken by
    the first occurrence in C order, matching ``numpy.argmax``.
    """
    flat_labels = labels_2d.ravel()
    flat_index = np.flatnonzero(flat_labels > 0)
    if flat_index.size == 0:
        empty_int = np.empty(0, dtype=np.int64)
        return empty_int, empty_int, empty_int, np.empty(0, dtype=np.float64)

    label = flat_labels[flat_index]
    value = values_2d.ravel()[flat_index]

    sign = -1.0 if center_on == "max" else 1.0
    key = np.where(np.isfinite(value), sign * value, np.inf)

    order = np.lexsort((flat_index, key, label))
    label = label[order]
    flat_index = flat_index[order]
    value = value[order]

    head = np.flatnonzero(np.r_[True, label[1:] != label[:-1]])
    label = label[head]
    flat_index = flat_index[head]
    value = value[head]

    keep = np.isfinite(value)
    label = label[keep]
    flat_index = flat_index[keep]
    value = value[keep]

    nx = labels_2d.shape[1]
    return label, flat_index // nx, flat_index % nx, value


def _track_metadata(
    ds: xr.Dataset,
    data_var: str,
    track_mask: xr.DataArray,
    center_on: str,
) -> TrackedFeatures:
    field = ds[data_var]
    leading_dims, group_dims = _tracking_dimensions(field)
    ordered_field = field.transpose(*group_dims, "time", "lat", "lon")
    ordered_mask = track_mask.transpose(*group_dims, "time", "lat", "lon")

    values = np.asarray(ordered_field.values, dtype=np.float64)
    labels = np.asarray(ordered_mask.values, dtype=np.int64)
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)

    # PyFLEXTRKR assigns non-contiguous track identifiers, so allocating a
    # contiguous range up to the largest identifier pads the track dimension
    # with entries that are absent from the mask and stay all-NaN.
    max_track = int(labels.max(initial=0))
    track_ids = np.unique(labels[labels > 0]).astype(np.int64)
    track_position = np.full(max_track + 1, -1, dtype=np.int64)
    track_position[track_ids] = np.arange(track_ids.size, dtype=np.int64)
    group_shape = tuple(ordered_field.sizes[dim] for dim in group_dims)
    ntime = ordered_field.sizes["time"]
    ntrack = track_ids.size

    center_shape = (*group_shape, ntime, ntrack)
    center_lat_values = np.full(center_shape, np.nan, dtype=np.float64)
    center_lon_values = np.full(center_shape, np.nan, dtype=np.float64)
    center_value_values = np.full(center_shape, np.nan, dtype=np.float64)

    for group_index in np.ndindex(*group_shape):
        group_values = values[group_index]
        group_labels = labels[group_index]
        for time_index in range(ntime):
            label, iy, ix, value = _frame_extrema(
                group_values[time_index],
                group_labels[time_index],
                center_on,
            )
            if label.size == 0:
                continue
            track_index = track_position[label]
            output_index = (*group_index, time_index, track_index)
            center_lat_values[output_index] = lat[iy]
            center_lon_values[output_index] = lon[ix]
            center_value_values[output_index] = value

    center_dims = (*group_dims, "time", "track")
    center_coords: dict[str, Any] = {
        **{dim: ordered_field[dim] for dim in group_dims},
        "time": ds["time"],
        "track": track_ids,
    }
    center_lat = xr.DataArray(
        center_lat_values,
        dims=center_dims,
        coords=center_coords,
        name="center_lat",
    ).transpose(*leading_dims, "track")
    center_lon = xr.DataArray(
        center_lon_values,
        dims=center_dims,
        coords=center_coords,
        name="center_lon",
    ).transpose(*leading_dims, "track")
    center_value = xr.DataArray(
        center_value_values,
        dims=center_dims,
        coords=center_coords,
        name="center_value",
    ).transpose(*leading_dims, "track")

    peak_shape = (*group_shape, ntrack)
    time_values = np.asarray(ds["time"].values).astype("datetime64[ns]")

    if ntrack == 0:
        peak_lat_values = np.full(peak_shape, np.nan, dtype=np.float64)
        peak_lon_values = np.full(peak_shape, np.nan, dtype=np.float64)
        peak_values = np.full(peak_shape, np.nan, dtype=np.float64)
        track_time_values = np.full(
            (*peak_shape, 3),
            np.datetime64("NaT"),
            dtype="datetime64[ns]",
        )
    else:
        # Fully vectorised replacement for the per-track lifetime loop. Filling
        # the invalid entries with a signed infinity leaves argmax and argmin
        # selecting the same index the compacted search would have returned.
        finite = np.isfinite(center_value_values)
        any_valid = finite.any(axis=-2)
        first_index = np.argmax(finite, axis=-2)
        last_index = finite.shape[-2] - 1 - np.argmax(finite[..., ::-1, :], axis=-2)

        sentinel = -np.inf if center_on == "max" else np.inf
        filled = np.where(finite, center_value_values, sentinel)
        peak_index = (
            np.argmax(filled, axis=-2)
            if center_on == "max"
            else np.argmin(filled, axis=-2)
        )

        def _gather(source: np.ndarray, index: np.ndarray) -> np.ndarray:
            return np.take_along_axis(source, index[..., None, :], axis=-2)[..., 0, :]

        peak_lat_values = np.where(
            any_valid, _gather(center_lat_values, peak_index), np.nan
        )
        peak_lon_values = np.where(
            any_valid, _gather(center_lon_values, peak_index), np.nan
        )
        peak_values = np.where(
            any_valid, _gather(center_value_values, peak_index), np.nan
        )
        track_time_values = np.stack(
            (
                time_values[first_index],
                time_values[peak_index],
                time_values[last_index],
            ),
            axis=-1,
        )
        track_time_values[~any_valid] = np.datetime64("NaT")

    track_dims = (*group_dims, "track")
    track_coords: dict[str, Any] = {
        **{dim: ordered_field[dim] for dim in group_dims},
        "track": track_ids,
    }
    peak_lat = xr.DataArray(
        peak_lat_values,
        dims=track_dims,
        coords=track_coords,
        name="peak_lat",
    )
    peak_lon = xr.DataArray(
        peak_lon_values,
        dims=track_dims,
        coords=track_coords,
        name="peak_lon",
    )
    peak = xr.DataArray(
        peak_values,
        dims=track_dims,
        coords=track_coords,
        name="peak",
    )
    track_time = xr.DataArray(
        track_time_values,
        dims=(*track_dims, "track_phase"),
        coords={
            **track_coords,
            "track_phase": ["start", "peak", "end"],
        },
        name="track_time",
    )

    units = ds[data_var].attrs.get("units", "")
    center_lat.attrs["units"] = "degrees_north"
    center_lon.attrs["units"] = "degrees_east"
    center_value.attrs["units"] = units
    peak_lat.attrs["units"] = "degrees_north"
    peak_lon.attrs["units"] = "degrees_east"
    peak.attrs["units"] = units
    track_time.attrs.update(
        {
            "long_name": "tracked feature phase time",
            "description": (
                "Timestamp of track start, lifetime center-value extremum, "
                "and track end."
            ),
        }
    )
    track_time["track_phase"].attrs.update(
        {
            "long_name": "tracked feature lifecycle phase",
            "description": "start, peak, and end timestamps stored in track_time",
        }
    )

    return TrackedFeatures(
        track_mask=track_mask,
        center_lat=center_lat,
        center_lon=center_lon,
        center_value=center_value,
        peak_lat=peak_lat,
        peak_lon=peak_lon,
        peak=peak,
        track_time=track_time,
    )


def _track_features(
    ds: xr.Dataset,
    data_var: str,
    threshold: float,
    threshold_operator: str,
    center_on: str,
    reference_data: dict[str, Any] | None,
    overlap_threshold: float,
    fill_value: int,
    parallel: bool,
) -> TrackedFeatures:
    if threshold_operator not in comparators:
        raise ValueError("threshold_operator must be one of '>', '>=', '<', or '<='.")
    if center_on not in ("min", "max"):
        raise ValueError("center_on must be 'min' or 'max'.")

    track_mask = _build_track_mask(
        ds,
        data_var,
        threshold,
        threshold_operator,
        reference_data,
        overlap_threshold,
        fill_value,
        parallel,
    )
    return _track_metadata(ds, data_var, track_mask, center_on)


def _subset_tracks(tracked: TrackedFeatures, keep: np.ndarray) -> TrackedFeatures:
    return TrackedFeatures(
        track_mask=tracked.track_mask,
        center_lat=tracked.center_lat.isel(track=keep),
        center_lon=tracked.center_lon.isel(track=keep),
        center_value=tracked.center_value.isel(track=keep),
        peak_lat=tracked.peak_lat.isel(track=keep),
        peak_lon=tracked.peak_lon.isel(track=keep),
        peak=tracked.peak.isel(track=keep),
        track_time=tracked.track_time.isel(track=keep),
    )


def _center_on_tracks(
    source: xr.Dataset,
    variable_names: Sequence[str],
    data_var: str,
    tracked: TrackedFeatures,
    leading_dims: tuple[str, ...],
    offsets: np.ndarray,
    method: str,
    output_dtype: str,
    workers: int,
) -> dict[str, np.ndarray]:
    """Interpolate variables onto a feature-centered grid, valid samples only.

    A track occupies only a small fraction of the record, so the dense product
    of leading dimensions and tracks is mostly empty. This routine visits each
    leading index once, gathers only the tracks that are actually present at
    that index, and issues a single ``map_coordinates`` call per variable for
    the whole group. The membership footprint is obtained by sampling the label
    field at the nearest source cell rather than by broadcasting a comparison
    over the full source grid.
    """
    lat_source = np.asarray(source["lat"].values, dtype=np.float64)
    lon_source = np.asarray(source["lon"].values, dtype=np.float64)
    ny = lat_source.size
    nx = lon_source.size
    index_y = np.arange(ny, dtype=np.float64)
    index_x = np.arange(nx, dtype=np.float64)

    lead_shape = tuple(tracked.center_lat.sizes[dim] for dim in leading_dims)
    n_lead = int(np.prod(lead_shape)) if lead_shape else 1
    track_ids = np.asarray(tracked.peak["track"].values, dtype=np.int64)
    n_track = track_ids.size
    n_cell = offsets.size

    center_lat = (
        tracked.center_lat.transpose(*leading_dims, "track")
        .values.astype(np.float64)
        .reshape(n_lead, n_track)
    )
    center_lon = (
        tracked.center_lon.transpose(*leading_dims, "track")
        .values.astype(np.float64)
        .reshape(n_lead, n_track)
    )
    valid = np.isfinite(center_lat) & np.isfinite(center_lon)

    feature_name = f"{data_var}_feature"
    output_names = [*variable_names, feature_name]
    itemsize = np.dtype(output_dtype).itemsize
    required = n_lead * n_track * n_cell * n_cell * itemsize * len(output_names)
    available = _available_bytes()
    if available and required > 0.9 * available:
        raise MemoryError(
            "The feature-centered output would require about "
            f"{required / 2**30:.1f} GiB for {len(output_names)} variables of "
            f"shape {(*lead_shape, n_track, n_cell, n_cell)}, against roughly "
            f"{available / 2**30:.1f} GiB available. Increase dx_km, reduce "
            "half_extent_km, restrict 'variables', or subset the tracks."
        )

    outputs = {
        name: np.full(
            (n_lead, n_track, n_cell, n_cell), np.nan, dtype=np.dtype(output_dtype)
        )
        for name in output_names
    }

    source_arrays = {
        name: np.ascontiguousarray(
            source[name].transpose(*leading_dims, "lat", "lon").values
        ).reshape(n_lead, ny, nx)
        for name in variable_names
    }
    label_array = np.ascontiguousarray(
        tracked.track_mask.transpose(*leading_dims, "lat", "lon").values,
        dtype=np.int64,
    ).reshape(n_lead, ny, nx)

    x_km, y_km = np.meshgrid(offsets, offsets)
    lat_offset_deg = np.rad2deg(y_km / _EARTH_RADIUS_KM)
    order = 1 if method == "linear" else 0

    lead_indices = np.flatnonzero(valid.any(axis=1))

    def _process(lead: int) -> None:
        selection = np.flatnonzero(valid[lead])
        target_lat = center_lat[lead, selection][:, None, None] + lat_offset_deg[None]
        cos_lat = np.cos(np.deg2rad(target_lat))
        cos_lat = np.where(np.abs(cos_lat) < 1e-10, 1e-10, cos_lat)
        target_lon = center_lon[lead, selection][:, None, None] + np.rad2deg(
            x_km[None] / (_EARTH_RADIUS_KM * cos_lat)
        )

        inside = (
            (target_lat >= lat_source[0])
            & (target_lat <= lat_source[-1])
            & (target_lon >= lon_source[0])
            & (target_lon <= lon_source[-1])
        )
        shape = target_lat.shape

        fractional_y = np.interp(target_lat.ravel(), lat_source, index_y)
        fractional_x = np.interp(target_lon.ravel(), lon_source, index_x)
        coordinates = np.stack((fractional_y, fractional_x))

        for name in variable_names:
            sampled = map_coordinates(
                source_arrays[name][lead],
                coordinates,
                order=order,
                mode="nearest",
                prefilter=False,
                output=np.float64,
            ).reshape(shape)
            sampled[~inside] = np.nan
            outputs[name][lead, selection] = sampled

        nearest_y = np.clip(np.rint(fractional_y).astype(np.intp), 0, ny - 1)
        nearest_x = np.clip(np.rint(fractional_x).astype(np.intp), 0, nx - 1)
        sampled_labels = label_array[lead][nearest_y, nearest_x].reshape(shape)
        member = (sampled_labels == track_ids[selection][:, None, None]) & inside

        feature = outputs[data_var][lead, selection].copy()
        feature[~member] = np.nan
        outputs[feature_name][lead, selection] = feature

    if workers > 1 and lead_indices.size > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_process, lead_indices.tolist()))
    else:
        for lead in lead_indices.tolist():
            _process(int(lead))

    return {
        name: array.reshape(*lead_shape, n_track, n_cell, n_cell)
        for name, array in outputs.items()
    }


def track_feature(
    ds: xr.Dataset,
    data_var: str,
    threshold: float,
    *,
    threshold_operator: Literal[">", ">=", "<", "<="] = ">=",
    overlap_threshold: float = 0.5,
    parallel: bool = False,
    fill_value: int = -9999,
    center_on: Literal["min", "max"] = "max",
    center_object: bool = True,
    dx_km: float = 3.0,
    variables: str | Sequence[str] | None = None,
    half_extent_km: float | None = None,
    method: str = "linear",
    reference_data: dict[str, Any] | None = None,
    output_dtype: str = "float32",
    workers: int | None = None,
) -> xr.Dataset:
    """Track threshold-defined features and optionally center fields on them.

    Connected features are identified from a threshold mask using PyFLEXTRKR
    and linked through time according to their spatial overlap. The original
    ``time`` coordinate is retained.

    For each tracked object and time step, ``center_on`` determines whether the
    object center is defined by the minimum or maximum value of ``data_var``.
    The same criterion determines the lifetime extremum reported by ``peak``
    and its corresponding time in ``track_time``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the field to track and any variables to retain or
        interpolate.
    data_var : str
        Name of the variable used to identify and track features.
    threshold : float
        Threshold applied to ``data_var`` to define candidate feature cells.
    threshold_operator : {">", ">=", "<", "<="}, default: ">="
        Comparison operator used to construct the threshold mask.
    overlap_threshold : float, default: 0.5
        Minimum overlap required to associate features between consecutive
        time steps.
    parallel : bool, default: False
        If True, PyFLEXTRKR runs on a Dask ``LocalCluster`` with one worker per
        CPU available to this process. The flag is written to the generated
        configuration as the integer ``run_parallel`` key, with the companion
        ``nprocesses`` key. If False, PyFLEXTRKR runs in serial.
    fill_value : int, default: -9999
        Integer fill value used for cells without a tracked feature.
    center_on : {"min", "max"}, default: "max"
        Extremum of ``data_var`` used to define the feature center at each
        time step and the lifetime value reported by ``peak``.
    center_object : bool, default: True
        If False, retain selected variables on the native grid and return
        ``track_mask`` containing the integer track ID at each grid cell and
        time. If True, interpolate selected variables to a feature-centered
        Cartesian grid for each track and time.
    dx_km : float, default: 3.0
        Horizontal grid spacing, in kilometers, of the feature-centered
        Cartesian grid. Used only when ``center_object=True``.
    variables : str or sequence of str, optional
        Variable or variables to retain in the output. When
        ``center_object=True``, these variables are interpolated to the
        feature-centered grid.
    half_extent_km : float, optional
        Half-width, in kilometers, of the feature-centered Cartesian domain.
        Used only when ``center_object=True``.
    method : str, default: "linear"
        Interpolation method used for the feature-centered grid.
    reference_data : dict[str, Any], optional
        Reference data supplied during feature-centered processing.
    output_dtype : str, default: "float32"
        Storage type of the feature-centered arrays. Single precision halves
        the memory footprint of the output. Set to ``"float64"`` to retain the
        precision of the source fields.
    workers : int, optional
        Threads used for the feature-centered interpolation. Defaults to the
        number of CPUs available to this process.

    Returns
    -------
    xarray.Dataset
        Dataset containing the tracked features and associated diagnostics.

        With ``center_object=False``, selected variables remain on their
        native grid and ``track_mask`` gives the integer track ID at each grid
        cell and time.

        With ``center_object=True``, selected variables are interpolated onto
        a local Cartesian grid with dimensions including ``time``, ``track``,
        ``y_km``, and ``x_km``. ``<data_var>_feature`` is additionally masked
        to the membership footprint of the corresponding tracked object.

    Notes
    -----
    PyFLEXTRKR ``DEBUG``, ``INFO``, and ``WARNING`` log records, together with
    captured standard output and standard error, are appended to
    ``tracking.log``. Only PyFLEXTRKR ``ERROR`` and ``CRITICAL`` records
    propagate to the caller's existing logging handlers.
    """
    if not isinstance(parallel, (bool, np.bool_)):
        raise TypeError("parallel must be a bool.")
    if center_object and method not in ("linear", "nearest"):
        raise ValueError("method must be 'linear' or 'nearest'.")
    if center_object and dx_km <= 0.0:
        raise ValueError("dx_km must be greater than zero.")
    if center_object and half_extent_km is not None and half_extent_km <= 0.0:
        raise ValueError("half_extent_km must be greater than zero.")
    if np.dtype(output_dtype).kind != "f":
        raise ValueError("output_dtype must be a floating point type.")

    field = ds[data_var]
    leading_dims, _ = _tracking_dimensions(field)
    tracked = _track_features(
        ds,
        data_var,
        threshold,
        threshold_operator,
        center_on,
        reference_data,
        overlap_threshold,
        fill_value,
        parallel,
    )
    _LOGGER.debug("tracking complete: %d tracks", tracked.peak.sizes.get("track", 0))

    if tracked.peak.sizes.get("track", 0) == 0:
        raise ValueError("No tracked features identified matching the criteria.")

    if variables is None:
        variable_names = [
            name
            for name, variable in ds.data_vars.items()
            if {"lat", "lon"}.issubset(variable.dims)
        ]
    elif isinstance(variables, str):
        variable_names = [variables]
    else:
        variable_names = list(variables)
    variable_names = list(dict.fromkeys((data_var, *variable_names)))
    source = ds[variable_names]

    track_coordinates: dict[str, xr.DataArray] = {
        "center_lat": tracked.center_lat,
        "center_lon": tracked.center_lon,
        "peak_lat": tracked.peak_lat,
        "peak_lon": tracked.peak_lon,
        "peak": tracked.peak,
        "track_phase": tracked.track_time["track_phase"],
        "track_time": tracked.track_time,
    }

    if not center_object:
        output = source.copy().assign_coords(track_coordinates)
        output["track_mask"] = tracked.track_mask
        output.attrs.update(
            {
                "tracking_variable": data_var,
                "threshold": threshold,
                "threshold_operator": threshold_operator,
                "overlap_threshold": overlap_threshold,
                "fill_value": fill_value,
                "center_on": center_on,
                "center_object": False,
                "parallel": int(bool(parallel)),
                "reference_data": (
                    "independent" if reference_data is None else repr(reference_data)
                ),
                "tracking_backend": "PyFLEXTRKR generic feature tracking",
            }
        )
        return output

    source = source.sortby("lat").sortby("lon")
    sorted_mask = tracked.track_mask.sortby("lat").sortby("lon")

    if half_extent_km is None:
        mean_lat_rad = np.deg2rad(float(source["lat"].mean()))
        lon_span_rad = np.deg2rad(abs(float(source["lon"].max() - source["lon"].min())))
        extent_km = _EARTH_RADIUS_KM * np.cos(mean_lat_rad) * lon_span_rad / 4.0
    else:
        extent_km = half_extent_km

    half_cells = int(np.floor(extent_km / dx_km))
    if half_cells < 1:
        raise ValueError("half_extent_km must be greater than or equal to dx_km.")
    offsets = np.arange(-half_cells, half_cells + 1, dtype=np.float64) * dx_km

    centering_input = TrackedFeatures(
        track_mask=sorted_mask,
        center_lat=tracked.center_lat,
        center_lon=tracked.center_lon,
        center_value=tracked.center_value,
        peak_lat=tracked.peak_lat,
        peak_lon=tracked.peak_lon,
        peak=tracked.peak,
        track_time=tracked.track_time,
    )
    arrays = _center_on_tracks(
        source,
        variable_names,
        data_var,
        centering_input,
        leading_dims,
        offsets,
        method,
        output_dtype,
        cpu_count() if workers is None else max(1, int(workers)),
    )

    target_dims = (*leading_dims, "track", "y_km", "x_km")
    target_coords: dict[str, Any] = {
        **{
            dim: field[dim]
            for dim in leading_dims
            if dim in field.coords and field[dim].dims == (dim,)
        },
        "track": tracked.peak["track"],
        "y_km": offsets,
        "x_km": offsets,
    }
    centered = xr.Dataset(
        {name: (target_dims, array) for name, array in arrays.items()},
        coords=target_coords,
    )

    for name in variable_names:
        centered[name].attrs.update(ds[name].attrs)
    centered[f"{data_var}_feature"].attrs.update(ds[data_var].attrs)
    centered[f"{data_var}_feature"].attrs["long_name"] = (
        f"{data_var} masked to the tracked feature footprint"
    )

    centered = centered.assign_coords(track_coordinates)
    centered["x_km"].attrs = {
        "long_name": "eastward distance from tracked feature center",
        "units": "km",
    }
    centered["y_km"].attrs = {
        "long_name": "northward distance from tracked feature center",
        "units": "km",
    }
    centered.attrs.update(
        {
            "tracking_variable": data_var,
            "threshold": threshold,
            "threshold_operator": threshold_operator,
            "overlap_threshold": overlap_threshold,
            "fill_value": fill_value,
            "center_on": center_on,
            "center_object": True,
            "parallel": int(bool(parallel)),
            "horizontal_spacing_km": dx_km,
            "half_extent_km": half_cells * dx_km,
            "interpolation_method": method,
            "reference_data": (
                "independent" if reference_data is None else repr(reference_data)
            ),
            "tracking_backend": "PyFLEXTRKR generic feature tracking",
        }
    )
    return centered


def get_relative_time(
    ds: xr.Dataset,
    data_var: str,
    threshold: float,
    delta_time: int,
    *,
    intensity_edges: tuple[float, ...] | None = None,
    threshold_operator: Literal[">", ">=", "<", "<="] = ">=",
    overlap_threshold: float = 0.5,
    parallel: bool = False,
    fill_value: int = -9999,
    center_on: Literal["min", "max"] = "max",
) -> xr.Dataset | None:
    """Extract track-relative windows and optionally composite by peak intensity.

    Features are identified and tracked from a threshold mask of ``data_var``.
    For each track, a window spanning ``delta_time`` samples before and after
    the track start is extracted and expressed relative to that start time.

    ``relative_time`` gives the integer sample offset from the track start,
    while the original ``time`` coordinate retains the corresponding valid
    timestamp for every track and relative-time sample. When
    ``intensity_edges`` is provided, tracks are grouped into intensity bins
    according to their lifetime peak value of ``data_var``. Otherwise, a
    single composite is calculated across all tracks.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the field from which features are identified and
        tracked.
    data_var : str
        Name of the variable used to identify, track, and characterize
        features.
    threshold : float
        Threshold applied to ``data_var`` to define candidate feature cells.
    delta_time : int
        Number of samples included on either side of each track start. The
        resulting relative-time window spans from ``-delta_time`` to
        ``+delta_time`` samples.
    intensity_edges : tuple of float, optional
        Bin edges used to group tracks according to their lifetime peak
        intensity. The final bin extends from the last edge to infinity.
        If None, no intensity binning is applied and a single composite is
        calculated across all tracks.
    threshold_operator : {">", ">=", "<", "<="}, default: ">="
        Comparison operator used to construct the threshold mask.
    overlap_threshold : float, default: 0.5
        Minimum spatial overlap required to associate features between
        consecutive time steps.
    parallel : bool, default: False
        If True, PyFLEXTRKR runs on a Dask ``LocalCluster`` with one worker per
        CPU available to this process. If False, PyFLEXTRKR runs in serial.
    fill_value : int, default: -9999
        Integer fill value used for cells or track identifiers without valid
        tracked-feature data.
    center_on : {"min", "max"}, default: "max"
        Extremum of ``data_var`` used to characterize each tracked feature.
        The same criterion determines the lifetime peak used for intensity
        binning and the ``peak`` entry of ``track_time``.

    Returns
    -------
    xarray.Dataset or None
        Dataset containing track-relative samples and weighted composites.
        ``relative_time`` contains integer sample offsets from each track
        start, and ``time`` contains the corresponding valid timestamps.

        If ``intensity_edges`` is provided, composites are grouped by
        peak-intensity bin. Otherwise, a single composite is calculated
        across all tracks.

        ``track_time`` has an explicit ``track_phase`` dimension with
        ``"start"``, ``"peak"``, and ``"end"`` entries.

        Returns ``None`` when no qualifying tracks are available for
        extraction or compositing.
    """
    if not isinstance(parallel, (bool, np.bool_)):
        raise TypeError("parallel must be a bool.")
    if delta_time < 1:
        raise ValueError("delta_time must be at least 1.")
    if intensity_edges is not None:
        if len(intensity_edges) == 0:
            raise ValueError("intensity_edges must contain at least one edge.")
        if any(
            right <= left
            for left, right in zip(
                intensity_edges,
                intensity_edges[1:],
                strict=False,
            )
        ):
            raise ValueError("intensity_edges must increase strictly.")

    field = ds[data_var]
    _, group_dims = _tracking_dimensions(field)
    tracked = _track_features(
        ds,
        data_var,
        threshold,
        threshold_operator,
        center_on,
        None,
        overlap_threshold,
        fill_value,
        parallel,
    )
    if tracked.peak.sizes.get("track", 0) == 0:
        return None

    def _window_positions(current: TrackedFeatures) -> tuple[xr.DataArray, np.ndarray]:
        starts = current.track_time.sel(track_phase="start")
        time_index = ds.get_index("time")
        start_values = np.asarray(starts.values).astype("datetime64[ns]")
        positions = time_index.get_indexer(start_values.ravel()).reshape(
            start_values.shape
        )
        return starts, positions

    starts, center_positions = _window_positions(tracked)
    complete = (center_positions >= delta_time) & (
        center_positions < ds.sizes["time"] - delta_time
    )
    if not np.any(complete):
        return None

    # Dropping incomplete tracks up front shrinks the track dimension of every
    # windowed array, which is far cheaper than masking them afterwards. This is
    # only unambiguous when completeness does not vary across other dimensions.
    if not group_dims and not complete.all():
        tracked = _subset_tracks(tracked, np.flatnonzero(complete))
        starts, center_positions = _window_positions(tracked)
        complete = np.ones_like(center_positions, dtype=bool)

    safe_positions = np.where(complete, center_positions, delta_time)
    position_coords = {dim: starts[dim] for dim in starts.dims}
    centers = xr.DataArray(
        safe_positions,
        dims=starts.dims,
        coords=position_coords,
    )
    complete_da = xr.DataArray(
        complete,
        dims=starts.dims,
        coords=position_coords,
    )
    relative_time = xr.DataArray(
        np.arange(-delta_time, delta_time + 1, dtype=np.int32),
        dims="relative_time",
        name="relative_time",
        attrs={
            "long_name": "sample offset from tracked feature start",
            "units": "time steps",
        },
    )
    window_indices = centers + relative_time

    events = ds.isel(time=window_indices).assign_coords(relative_time=relative_time)
    if not bool(complete.all()):
        for name, variable in events.data_vars.items():
            if "track" in variable.dims:
                events[name] = variable.where(complete_da)
        events = events.assign_coords(time=events["time"].where(complete_da))
    events = events.assign_coords(
        peak=tracked.peak,
        peak_lat=tracked.peak_lat,
        peak_lon=tracked.peak_lon,
        track_phase=tracked.track_time["track_phase"],
        track_time=tracked.track_time,
    )

    # Only tracks with a complete window contribute to the numerator, so the
    # denominator must be restricted to the same set. Leaving the weights of
    # excluded tracks in place scales the composite down by the fraction of
    # tracks retained.
    weights = (
        np.cos(np.deg2rad(events["peak_lat"]))
        .where(complete_da)
        .assign_coords(peak=events["peak"])
    )
    contributing = weights.notnull()
    sample_dims = (*group_dims, "track")
    track_vars = [
        name
        for name, variable in events.data_vars.items()
        if "track" in variable.dims and np.issubdtype(variable.dtype, np.number)
    ]
    composite_source = events[track_vars].reset_coords(drop=True)
    composite_source = composite_source.assign_coords(peak=events["peak"])

    if group_dims:
        composite_source = composite_source.stack(sample=sample_dims)
        weights = weights.stack(sample=sample_dims)
        contributing = contributing.stack(sample=sample_dims)
        composite_source = composite_source.assign_coords(
            peak=events["peak"].stack(sample=sample_dims)
        )
        reduce_dim = "sample"
    else:
        reduce_dim = "track"

    if intensity_edges is not None:
        bins = [*intensity_edges, np.inf]
        bin_labels = list(intensity_edges)

        numerator = (
            (composite_source * weights)
            .groupby_bins("peak", bins=bins, labels=bin_labels)
            .sum(dim=reduce_dim)
        )
        denominator = weights.groupby_bins(
            "peak",
            bins=bins,
            labels=bin_labels,
        ).sum(dim=reduce_dim)
        composite = numerator / denominator
        composite["n_tracks"] = (
            contributing.astype(np.int32)
            .assign_coords(peak=composite_source["peak"])
            .groupby_bins("peak", bins=bins, labels=bin_labels)
            .sum(dim=reduce_dim)
            .fillna(0)
            .astype(np.int32)
        )
    else:
        numerator = (composite_source * weights).sum(dim=reduce_dim)
        denominator = weights.sum(dim=reduce_dim)
        composite = numerator / denominator
        composite["n_tracks"] = (
            contributing.astype(np.int32).sum(dim=reduce_dim).astype(np.int32)
        )

    composite = composite.rename(
        {
            name: f"{name}_composite"
            for name in composite.data_vars
            if name != "n_tracks"
        }
    )

    store = xr.merge([events, composite], combine_attrs="no_conflicts")
    store.attrs.update(
        {
            "delta_time": delta_time,
            "relative_time_units": "time steps from track start",
            "intensity_edges": (
                None if intensity_edges is None else list(intensity_edges)
            ),
            "tracking_variable": data_var,
            "threshold": threshold,
            "threshold_operator": threshold_operator,
            "overlap_threshold": overlap_threshold,
            "fill_value": fill_value,
            "center_on": center_on,
            "parallel": int(bool(parallel)),
            "tracking_backend": "PyFLEXTRKR generic feature tracking",
        }
    )
    return store
