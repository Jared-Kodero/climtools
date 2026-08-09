from __future__ import annotations

import logging
import operator
import os
import sys
import tempfile
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr
import yaml

_EARTH_RADIUS_KM = 6371.0
_TRACKING_LOG = "tracking.log"

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
        with tempfile.TemporaryDirectory(prefix="pyflextrkr_") as temporary_directory:
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


def _tracking_capacity(
    cloudid_files: Sequence[str | Path],
    overlap_threshold: float,
) -> tuple[int, int]:
    """Derive PyFLEXTRKR allocation sizes from identified feature masks."""
    max_features = 0
    max_links = 0
    previous: np.ndarray | None = None

    for filename in cloudid_files:
        with xr.open_dataset(filename, mask_and_scale=False) as feature_ds:
            labels = np.asarray(
                feature_ds["feature_number"].squeeze("time", drop=True).values,
                dtype=np.int64,
            )

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
) -> xr.DataArray:
    """Run PyFLEXTRKR for one time-lat-lon field and return track IDs."""
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

    first = times[0].astype("datetime64[m]")
    last = times[-1].astype("datetime64[m]")
    if times[-1] > last.astype("datetime64[ns]"):
        last += np.timedelta64(1, "m")
    startdate = np.datetime_as_string(first, unit="m").replace("-", "")
    startdate = startdate.replace(":", "").replace("T", ".")
    enddate = np.datetime_as_string(last, unit="m").replace("-", "")
    enddate = enddate.replace(":", "").replace("T", ".")

    with tracker_env() as work:
        input_path = work / "input"
        input_path.mkdir()

        first_stamp = np.datetime_as_string(times[0], unit="s")
        first_stamp = first_stamp.replace("-", "").replace(":", "")
        first_stamp = first_stamp.replace("T", ".")
        input_file = input_path / f"input_{first_stamp}.nc"
        tracking_input.transpose("time", "lat", "lon").to_netcdf(input_file)

        config_file = work / "config.yml"
        config = {
            "run_parallel": 0,
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

        maxnclouds, nmaxlinks = _tracking_capacity(
            cloudid_files,
            overlap_threshold,
        )
        pyflex_config["maxnclouds"] = maxnclouds
        pyflex_config["nmaxlinks"] = nmaxlinks

        tracksingle_driver(pyflex_config)
        tracknumbers_file = gettracknumbers(pyflex_config)

        with xr.open_dataset(
            tracknumbers_file,
            mask_and_scale=False,
        ) as track_numbers_ds:
            track_numbers = (
                track_numbers_ds["track_numbers"]
                .squeeze("time", drop=True)
                .load()
                .astype(np.int64)
            )
            basetimes = (
                track_numbers_ds["basetimes"].load().values.astype("datetime64[ns]")
            )

        tracking_outpath = Path(pyflex_config["tracking_outpath"])
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
            cloudid_file = tracking_outpath / f"cloudid_{stamp}.nc"
            with xr.open_dataset(
                cloudid_file,
                mask_and_scale=False,
            ) as cloudid_ds:
                feature_number = np.asarray(
                    cloudid_ds["feature_number"].squeeze("time", drop=True).values,
                    dtype=np.int64,
                )

            file_tracks = np.asarray(
                track_numbers.isel(nfiles=file_index).values,
                dtype=np.int64,
            )
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
                "reference_data may select only non-time leading dimensions; invalid keys: {sorted(invalid)!r}."
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

    max_track = int(labels.max(initial=0))
    track_ids = np.arange(1, max_track + 1, dtype=np.int64)
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
            labels_at_time = group_labels[time_index]
            present = np.unique(labels_at_time)
            present = present[present > 0]
            for track_id in present:
                track_index = int(track_id) - 1
                iy, ix = np.where(labels_at_time == track_id)
                feature_values = group_values[time_index, iy, ix]
                finite = np.isfinite(feature_values)
                if not np.any(finite):
                    continue

                iy = iy[finite]
                ix = ix[finite]
                feature_values = feature_values[finite]
                local_index = int(
                    np.argmax(feature_values)
                    if center_on == "max"
                    else np.argmin(feature_values)
                )
                output_index = (*group_index, time_index, track_index)
                center_lat_values[output_index] = lat[iy[local_index]]
                center_lon_values[output_index] = lon[ix[local_index]]
                center_value_values[output_index] = feature_values[local_index]

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
    peak_lat_values = np.full(peak_shape, np.nan, dtype=np.float64)
    peak_lon_values = np.full(peak_shape, np.nan, dtype=np.float64)
    peak_values = np.full(peak_shape, np.nan, dtype=np.float64)
    track_time_values = np.full(
        (*peak_shape, 3),
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )
    time_values = np.asarray(ds["time"].values).astype("datetime64[ns]")

    for group_index in np.ndindex(*group_shape):
        for track_index in range(ntrack):
            track_values = center_value_values[group_index + (slice(None), track_index)]
            valid = np.flatnonzero(np.isfinite(track_values))
            if valid.size == 0:
                continue

            start_index = int(valid[0])
            end_index = int(valid[-1])
            valid_values = track_values[valid]
            extrema_index = int(
                np.argmax(valid_values)
                if center_on == "max"
                else np.argmin(valid_values)
            )
            peak_time_index = int(valid[extrema_index])
            output_index = (*group_index, track_index)

            peak_lat_values[output_index] = center_lat_values[
                group_index + (peak_time_index, track_index)
            ]
            peak_lon_values[output_index] = center_lon_values[
                group_index + (peak_time_index, track_index)
            ]
            peak_values[output_index] = center_value_values[
                group_index + (peak_time_index, track_index)
            ]
            track_time_values[output_index] = [
                time_values[start_index],
                time_values[peak_time_index],
                time_values[end_index],
            ]

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
    )
    return _track_metadata(ds, data_var, track_mask, center_on)


def track_feature(
    ds: xr.Dataset,
    data_var: str,
    threshold: float,
    *,
    threshold_operator: Literal[">", ">=", "<", "<="] = ">=",
    overlap_threshold: float = 0.5,
    fill_value: int = -9999,
    center_on: Literal["min", "max"] = "max",
    center_object: bool = False,
    dx_km: float = 3.0,
    variables: str | Sequence[str] | None = None,
    half_extent_km: float | None = None,
    method: str = "linear",
    reference_data: dict[str, Any] | None = None,
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
    fill_value : int, default: -9999
        Integer fill value used for cells without a tracked feature.
    center_on : {"min", "max"}, default: "max"
        Extremum of ``data_var`` used to define the feature center at each
        time step and the lifetime value reported by ``peak``.
    center_object : bool, default: False
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
    if center_object and method not in ("linear", "nearest"):
        raise ValueError("method must be 'linear' or 'nearest'.")
    if center_object and dx_km <= 0.0:
        raise ValueError("dx_km must be greater than zero.")
    if center_object and half_extent_km is not None and half_extent_km <= 0.0:
        raise ValueError("half_extent_km must be greater than zero.")

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
    )
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
                "reference_data": (
                    "independent" if reference_data is None else repr(reference_data)
                ),
                "tracking_backend": "PyFLEXTRKR generic feature tracking",
            }
        )
        return output

    source = source.sortby("lat").sortby("lon")
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
    x_km, y_km = np.meshgrid(offsets, offsets)

    valid_track = tracked.center_lat.notnull() & tracked.center_lon.notnull()
    interpolation_lat = tracked.center_lat.fillna(float(source["lat"].values[0]))
    interpolation_lon = tracked.center_lon.fillna(float(source["lon"].values[0]))
    target_lat_values = np.asarray(interpolation_lat.values)[
        ..., None, None
    ] + np.rad2deg(y_km / _EARTH_RADIUS_KM)
    cos_lat = np.cos(np.deg2rad(target_lat_values))
    cos_lat = np.where(np.abs(cos_lat) < 1e-10, 1e-10, cos_lat)
    target_lon_values = np.asarray(interpolation_lon.values)[
        ..., None, None
    ] + np.rad2deg(x_km / (_EARTH_RADIUS_KM * cos_lat))

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
    target_lat = xr.DataArray(
        target_lat_values,
        dims=target_dims,
        coords=target_coords,
    )
    target_lon = xr.DataArray(
        target_lon_values,
        dims=target_dims,
        coords=target_coords,
    )

    centered = source.interp(
        lat=target_lat,
        lon=target_lon,
        method=method,
        assume_sorted=True,
        kwargs={"bounds_error": False, "fill_value": np.nan},
    ).drop_vars(["lat", "lon"], errors="ignore")
    centered = centered.where(valid_track)

    labels = tracked.track_mask.sortby("lat").sortby("lon")
    track_index = xr.DataArray(
        tracked.peak["track"].values,
        dims="track",
        coords={"track": tracked.peak["track"]},
    )
    membership = (labels == track_index).transpose(
        *leading_dims,
        "track",
        "lat",
        "lon",
    )
    membership_interp = (
        membership.astype(np.int8)
        .interp(
            lat=target_lat,
            lon=target_lon,
            method="nearest",
            assume_sorted=True,
            kwargs={"bounds_error": False, "fill_value": 0},
        )
        .drop_vars(["lat", "lon"], errors="ignore")
    )
    centered[f"{data_var}_feature"] = centered[data_var].where(
        valid_track & membership_interp.astype(bool)
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
            "horizontal_spacing_km": dx_km,
            "half_extent_km": half_cells * dx_km,
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
    delta_time : int
        Number of samples included on either side of each track start. The
        resulting relative-time window spans from ``-delta_time`` to
        ``+delta_time`` samples.
    intensity_edges : tuple of float, optional
        Bin edges used to group tracks according to their lifetime peak
        intensity. The final bin extends from the last edge to infinity.
        If None, no intensity binning is applied and a single composite is
        calculated across all tracks.
    threshold : float, default: 0.1
        Threshold applied to ``data_var`` to define candidate feature cells.
    threshold_operator : {">", ">=", "<", "<="}, default: ">="
        Comparison operator used to construct the threshold mask.
    overlap_threshold : float, default: 0.5
        Minimum spatial overlap required to associate features between
        consecutive time steps.
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
    )
    if tracked.peak.sizes.get("track", 0) == 0:
        return None

    starts = tracked.track_time.sel(track_phase="start")
    time_index = ds.get_index("time")
    start_values = np.asarray(starts.values).astype("datetime64[ns]")
    center_positions = time_index.get_indexer(start_values.ravel()).reshape(
        start_values.shape
    )
    complete = (center_positions >= delta_time) & (
        center_positions < ds.sizes["time"] - delta_time
    )
    if not np.any(complete):
        return None

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

    weights = np.cos(np.deg2rad(events["peak_lat"])).assign_coords(peak=events["peak"])
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
            xr.ones_like(weights, dtype=np.int32)
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
            xr.ones_like(weights, dtype=np.int32).sum(dim=reduce_dim).astype(np.int32)
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
            "tracking_backend": "PyFLEXTRKR generic feature tracking",
        }
    )
    return store
