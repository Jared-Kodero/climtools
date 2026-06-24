from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from climtools import xgeo as xg


def open_event_store(path: Path) -> xr.Dataset:
    path = Path(path)

    ds = xg.open_dataset(path)

    ds.attrs["metadata"] = path.with_suffix(".csv")

    return ds


def select_events(
    ds: xr.Dataset,
    condition: str,
) -> xr.Dataset:

    if not condition:
        return ds

    path = Path(ds.attrs["metadata"])
    meta = pd.read_csv(
        path,
        parse_dates=[
            "time",
            "center_time",
            "track_start_time",
            "track_end_time",
        ],
    )

    meta["time_idx"] = meta["time_idx"].astype(int)
    meta["object"] = meta["object"].astype(int)
    meta["lead_hours"] = (
        meta["center_time"] - meta["time"]
    ).dt.total_seconds() / 3600.0

    selected = (
        meta.query(condition).sort_values(["time_idx", "object"]).reset_index(drop=True)
    )

    if len(selected) == 0:
        print(f"Selected {len(selected)} events out of {len(meta)}")
        return ds[[~list(ds.data_vars)]]

    # Shared dimension gives paired selection:
    # selected row i --> ds.isel(time=time_idx[i], object=object[i])
    time_index = xr.DataArray(
        selected["time_idx"].to_numpy(dtype=np.int64),
        dims="time",
    )
    object_index = xr.DataArray(
        selected["object"].to_numpy(dtype=np.int64),
        dims="time",
    )

    out = ds.isel(
        time=time_index,
        object=object_index,
    )
    return out
