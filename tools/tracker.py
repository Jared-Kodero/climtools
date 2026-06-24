from __future__ import annotations

# get_bounding_box_for_fft func modified  in  movement_speed
import logging
import time as pytime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyflextrkr.ft_utilities import load_config
from pyflextrkr.gettracks import gettracknumbers
from pyflextrkr.mapfeature_driver import mapfeature_driver
from pyflextrkr.movement_speed import movement_speed
from pyflextrkr.tracksingle_driver import tracksingle_driver
from pyflextrkr.trackstats_driver import trackstats_driver

logger = logging.getLogger("pyflextrkr")
logger.setLevel(logging.INFO)
logger.propagate = False

handler = logging.FileHandler("pyflextrkr.log", mode="a", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _timestamp(value: Any) -> pd.Timestamp:
    """Convert a numpy datetime64, pandas timestamp, or cftime-like object to pandas Timestamp."""
    try:
        ts = pd.Timestamp(value)
    except Exception:
        ts = pd.Timestamp(
            year=int(value.year),
            month=int(value.month),
            day=int(value.day),
            hour=int(getattr(value, "hour", 0)),
            minute=int(getattr(value, "minute", 0)),
            second=int(getattr(value, "second", 0)),
        )

    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    return ts


def _epoch_seconds(value: Any) -> int:
    ts = _timestamp(value)
    return int((ts - pd.Timestamp("1970-01-01T00:00:00")).total_seconds())


def _pyflex_time_string(value: Any) -> tuple[str, str, str]:
    """
    Return:
    - start/end format: YYYYMMDD.HHMM
    - date string: YYYYMMDD
    - time string: HHMMSS
    """
    ts = _timestamp(value)
    ymdhm = ts.strftime("%Y%m%d.%H%M")
    ymd = ts.strftime("%Y%m%d")
    hms = ts.strftime("%H%M%S")
    return ymdhm, ymd, hms


def _time_resolution_hours(time_vals: np.ndarray) -> float:
    if len(time_vals) < 2:
        return 1.0

    epochs = np.asarray([_epoch_seconds(t) for t in time_vals], dtype=float)
    dt_hours = np.nanmedian(np.diff(epochs)) / 3600.0

    if not np.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError("Could not infer a positive time resolution from time_vals.")

    return float(dt_hours)


def _trackstats_filebase_from_path(
    trackstats_file: str | Path,
    startdate: str,
    enddate: str,
) -> str:
    name = Path(trackstats_file).name
    suffix = f"{startdate}_{enddate}.nc"

    if not name.endswith(suffix):
        raise ValueError(
            f"Could not infer trackstats filebase from {trackstats_file!s}. Expected filename to end with {suffix!r}."
        )

    return name[: -len(suffix)]


def _write_pyflex_cloudids_from_object_peaks(
    tmp_frames: Path,
    tmp_cloudid: Path,
    peaks_per_time: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    data_var: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    time_vals: np.ndarray,
) -> int:
    """
    Write PyFLEXTRKR Step-1-style cloudid files using this script's _object_peaks output.

    Important mapping:
    - PyFLEXTRKR cloudnumber == CSV object + 1 at that detection time.
    - The mask is remapped into peak-sorted order, matching the object order already used
      by build_object_windows_for_time.
    """
    tmp_cloudid.mkdir(parents=True, exist_ok=True)

    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    max_nfeatures = 0

    for it, (peak_y, peak_x, obj_labels, _, _) in enumerate(peaks_per_time):
        _, datestr, timestr = _pyflex_time_string(time_vals[it])
        base_time = float(_epoch_seconds(time_vals[it]))

        with xr.open_dataset(tmp_frames / f"{it}") as ds:
            frame = ds.drop_vars("time", errors="ignore").load()
            pr2d = frame[data_var].squeeze(drop=True).values.astype(np.float32)

        feature_mask = np.zeros_like(obj_labels, dtype=np.int32)

        if peak_y.size > 0:
            old_label_ids = obj_labels[peak_y, peak_x].astype(np.int32)

            for new_id, old_id in enumerate(old_label_ids, start=1):
                if old_id > 0:
                    feature_mask[obj_labels == old_id] = new_id

        nfeatures = int(feature_mask.max())
        max_nfeatures = max(max_nfeatures, nfeatures)

        if nfeatures > 0:
            npix_feature = np.bincount(feature_mask.ravel(), minlength=nfeatures + 1)[
                1:
            ].astype(np.int32)
            feature_coord = np.arange(1, nfeatures + 1, dtype=np.int32)
        else:
            npix_feature = np.empty(0, dtype=np.int32)
            feature_coord = np.empty(0, dtype=np.int32)

        dsout = xr.Dataset(
            data_vars={
                "base_time": (
                    ("time",),
                    np.asarray([base_time], dtype=np.float64),
                    {
                        "long_name": "Base time in Epoch",
                        "units": "Seconds since 1970-1-1 0:00:00 0:00",
                    },
                ),
                "longitude": (
                    ("lat", "lon"),
                    lon2d.astype(np.float32),
                    {"long_name": "Longitude", "units": "degrees_east"},
                ),
                "latitude": (
                    ("lat", "lon"),
                    lat2d.astype(np.float32),
                    {"long_name": "Latitude", "units": "degrees_north"},
                ),
                data_var: (
                    ("time", "lat", "lon"),
                    pr2d[None, :, :],
                    {"long_name": data_var},
                ),
                "feature_number": (
                    ("time", "lat", "lon"),
                    feature_mask[None, :, :],
                    {
                        "long_name": "Labeled precipitation object number for tracking",
                        "units": "unitless",
                    },
                ),
                "nfeatures": (
                    ("time",),
                    np.asarray([nfeatures], dtype=np.int32),
                    {"long_name": "Number of labeled features", "units": "unitless"},
                ),
                "npix_feature": (
                    ("features",),
                    npix_feature,
                    {
                        "long_name": "Number of pixels for each labeled feature",
                        "units": "unitless",
                    },
                ),
            },
            coords={
                "time": (
                    ("time",),
                    np.asarray([base_time], dtype=np.float64),
                    {
                        "long_name": "Base time in Epoch",
                        "units": "Seconds since 1970-1-1 0:00:00 0:00",
                    },
                ),
                "lat": (("lat",), lat_vals),
                "lon": (("lon",), lon_vals),
                "features": (("features",), feature_coord),
            },
            attrs={
                "Title": f"PyFLEXTRKR cloudid file from _object_peaks, {datestr}.{timestr}",
                "Created_on": pytime.ctime(pytime.time()),
            },
        )

        outfile = tmp_cloudid / f"cloudid_{datestr}_{timestr}.nc"
        outfile.unlink(missing_ok=True)
        dsout.to_netcdf(outfile, mode="w", format="NETCDF4")

    return max_nfeatures


def _write_pyflex_config(
    tmp_base: Path,
    time_vals: np.ndarray,
    dx_km: float,
    max_nfeatures: int,
    ntime: int,
    overlap_threshold: float,
    data_var: str,
) -> Path:
    """
    Write a temporary PyFLEXTRKR config.

    The tracking_path_name is intentionally 'cloudid' so cloudid and track_*.nc
    files live under tmp_base/cloudid.
    """
    startdate, _, _ = _pyflex_time_string(time_vals[0])
    enddate, _, _ = _pyflex_time_string(time_vals[-1])
    dt_hours = _time_resolution_hours(time_vals)

    config = {
        "run_idfeature": False,
        "run_tracksingle": True,
        "run_gettracks": True,
        "run_trackstats": True,
        "run_mergesplit": False,
        "run_mapfeature": True,
        "run_speed": True,
        "startdate": startdate,
        "enddate": enddate,
        "run_parallel": 0,
        "nprocesses": 1,
        "root_path": str(tmp_base),
        "tracking_path_name": "cloudid",
        "stats_path_name": "trackstats",
        "pixel_path_name": "pixeltracking",
        "feature_type": "generic",
        "datatimeresolution": dt_hours,
        "timegap": dt_hours * 3.1,
        "pixel_radius": float(dx_km),
        "area_method": "fixed",
        "othresh": float(overlap_threshold),
        "maxnclouds": int(max(max_nfeatures + 5, 10)),
        "nmaxlinks": 10,
        "duration_range": [2, int(max(ntime + 1, 2))],
        "duration_range_auto_update": True,
        "remove_shorttracks": 1,
        "trackstats_dense_netcdf": 1,
        "match_pixel_dt_thresh": 60.0,
        "feature_varname": "feature_number",
        "nfeature_varname": "nfeatures",
        "featuresize_varname": "npix_feature",
        "tracks_dimname": "tracks",
        "times_dimname": "times",
        "fillval": -9999,
        "finalstats_filebase": "trackstats_final_",
        "speedstats_filebase": "trackstats_speed_",
        "pixeltracking_filebase": "tracks_",
        "lag_for_speed": 1,
        "track_number_for_speed": "tracknumber",
        "track_field_for_speed": data_var,
        "min_size_thresh_for_speed": 1,
        "max_speed_thresh": 50.0,
        "clouddata_path": str(tmp_base / "cloudid") + "/",
        "databasename": "cloudid_",
        "time_format": "yyyymodd_hhmmss",
    }

    config_path = tmp_base / "pyflextrkr_config.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return config_path


def _trackstats_to_event_lookup(
    trackstats_file: str | Path,
    time_vals: np.ndarray,
    dt_hours: float,
    fillval: int = -9999,
) -> pd.DataFrame:
    """
    Convert dense PyFLEXTRKR track statistics to a table keyed by:
    - center_time_idx
    - object

    Because cloudid writing remaps feature_number to object + 1, this lookup joins
    directly onto the current CSV using ['center_time_idx', 'object'].
    """
    epoch_to_it = {_epoch_seconds(t): i for i, t in enumerate(time_vals)}
    rows: list[dict[str, Any]] = []

    def valid_number(x: Any) -> bool:
        return np.isfinite(x) and int(x) != fillval

    with xr.open_dataset(
        trackstats_file, mask_and_scale=False, decode_times=False
    ) as ds:
        base_time = ds["base_time"].values
        cloudnumber = ds["cloudnumber"].values
        track_duration = ds["track_duration"].values.astype(int)

        area = ds["area"].values if "area" in ds else None
        meanlat = ds["meanlat"].values if "meanlat" in ds else None
        meanlon = ds["meanlon"].values if "meanlon" in ds else None
        track_status = ds["track_status"].values if "track_status" in ds else None
        merge_tracknumbers = (
            ds["merge_tracknumbers"].values if "merge_tracknumbers" in ds else None
        )
        split_tracknumbers = (
            ds["split_tracknumbers"].values if "split_tracknumbers" in ds else None
        )

        start_basetime = ds["start_basetime"].values if "start_basetime" in ds else None
        end_basetime = ds["end_basetime"].values if "end_basetime" in ds else None

        movement_vars = {
            "track_movement_distance_km": (
                ds["movement_distance"].values if "movement_distance" in ds else None
            ),
            "track_movement_speed_ms": (
                ds["movement_speed"].values if "movement_speed" in ds else None
            ),
            "track_movement_theta_deg": (
                ds["movement_theta"].values if "movement_theta" in ds else None
            ),
            "track_movement_distance_x_km": (
                ds["movement_distance_x"].values
                if "movement_distance_x" in ds
                else None
            ),
            "track_movement_distance_y_km": (
                ds["movement_distance_y"].values
                if "movement_distance_y" in ds
                else None
            ),
        }

        ntracks, ntimes_track = base_time.shape

        for itrack in range(ntracks):
            track_id = int(itrack + 1)
            duration = int(track_duration[itrack])

            start_time = (
                pd.to_datetime(start_basetime[itrack], unit="s")
                if start_basetime is not None and valid_number(start_basetime[itrack])
                else pd.NaT
            )
            end_time = (
                pd.to_datetime(end_basetime[itrack], unit="s")
                if end_basetime is not None and valid_number(end_basetime[itrack])
                else pd.NaT
            )

            for j in range(ntimes_track):
                bt = base_time[itrack, j]
                cn = cloudnumber[itrack, j]

                if not valid_number(bt) or not valid_number(cn) or int(cn) <= 0:
                    continue

                center_time_idx = epoch_to_it.get(int(round(float(bt))))
                if center_time_idx is None:
                    continue

                row = {
                    "center_time_idx": int(center_time_idx),
                    "object": int(cn) - 1,
                    "track_id": track_id,
                    "track_duration": duration,
                    "track_duration_hours": float(duration * dt_hours),
                    "track_age": int(j),
                    "track_age_hours": float(j * dt_hours),
                    "track_start_time": start_time,
                    "track_end_time": end_time,
                }

                if area is not None:
                    row["track_area_km2"] = (
                        round(float(area[itrack, j]), 3)
                        if valid_number(area[itrack, j])
                        else np.nan
                    )
                if meanlat is not None:
                    row["track_meanlat"] = (
                        round(float(meanlat[itrack, j]), 3)
                        if valid_number(meanlat[itrack, j])
                        else np.nan
                    )
                if meanlon is not None:
                    row["track_meanlon"] = (
                        round(float(meanlon[itrack, j]), 3)
                        if valid_number(meanlon[itrack, j])
                        else np.nan
                    )
                if track_status is not None:
                    row["track_status"] = (
                        int(track_status[itrack, j])
                        if valid_number(track_status[itrack, j])
                        else pd.NA
                    )
                if merge_tracknumbers is not None:
                    row["merge_track_id"] = (
                        int(merge_tracknumbers[itrack, j])
                        if valid_number(merge_tracknumbers[itrack, j])
                        and int(merge_tracknumbers[itrack, j]) > 0
                        else pd.NA
                    )
                if split_tracknumbers is not None:
                    row["split_track_id"] = (
                        int(split_tracknumbers[itrack, j])
                        if valid_number(split_tracknumbers[itrack, j])
                        and int(split_tracknumbers[itrack, j]) > 0
                        else pd.NA
                    )

                for out_name, values in movement_vars.items():
                    if values is not None:
                        row[out_name] = (
                            round(float(values[itrack, j]), 3)
                            if valid_number(values[itrack, j])
                            else np.nan
                        )

                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["center_time_idx", "object", "track_id"])

    return (
        pd.DataFrame(rows)
        .sort_values(["center_time_idx", "object", "track_id"])
        .drop_duplicates(["center_time_idx", "object"], keep="first")
        .reset_index(drop=True)
    )


def track_events(
    tmp_base: Path,
    tmp_frames: Path,
    peaks_per_time: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    data_var: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    time_vals: np.ndarray,
    dx_km: float,
    overlap_threshold: float = 0.3,
) -> pd.DataFrame:
    tmp_cloudid = tmp_base / "cloudid"
    max_nfeatures = _write_pyflex_cloudids_from_object_peaks(
        tmp_frames=tmp_frames,
        tmp_cloudid=tmp_cloudid,
        peaks_per_time=peaks_per_time,
        data_var=data_var,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        time_vals=time_vals,
    )

    config_path = _write_pyflex_config(
        tmp_base=tmp_base,
        time_vals=time_vals,
        dx_km=dx_km,
        max_nfeatures=max_nfeatures,
        ntime=len(time_vals),
        overlap_threshold=overlap_threshold,
        data_var=data_var,
    )

    config = load_config(str(config_path))

    tracksingle_driver(config)
    gettracknumbers(config)
    trackstats_file = trackstats_driver(config)

    trackstats_filebase = _trackstats_filebase_from_path(
        trackstats_file=trackstats_file,
        startdate=config["startdate"],
        enddate=config["enddate"],
    )

    mapfeature_driver(
        config,
        trackstats_filebase=trackstats_filebase,
        outpath_basename=config["pixel_path_name"],
        outfile_basename=config["pixeltracking_filebase"],
    )

    trackstats_file = movement_speed(
        config,
        trackstats_filebase=trackstats_filebase,
        trackstats_outfilebase=config["speedstats_filebase"],
        pixelpath_basename=config["pixel_path_name"],
        pixeltracking_filebase=config["pixeltracking_filebase"],
    )

    return _trackstats_to_event_lookup(
        trackstats_file=trackstats_file,
        time_vals=time_vals,
        dt_hours=_time_resolution_hours(time_vals),
        fillval=int(config["fillval"]),
    )


def run_pyflextrkr(
    tmp_base: Path,
    tmp_frames: Path,
    peaks_per_time: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    data_var: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    time_vals: np.ndarray,
    dx_km: float,
    overlap_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Wrapper called immediately after peaks_per_time has been filled.

    This bypasses PyFLEXTRKR labeling. It uses _object_peaks output as the
    externally supplied feature masks, then runs PyFLEXTRKR linking and
    statistics.
    """

    df = track_events(
        tmp_base=tmp_base,
        tmp_frames=tmp_frames,
        peaks_per_time=peaks_per_time,
        data_var=data_var,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        time_vals=time_vals,
        dx_km=dx_km,
        overlap_threshold=overlap_threshold,
    )

    # add new lines for readability in log
    logger.info("\n" * 5)

    return df
