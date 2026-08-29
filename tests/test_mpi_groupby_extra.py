"""Probe: groupby_reduce(op="min"/"max", skipna=True) when a group's local
partial on one rank is entirely NaN, but the same group has valid values on
another rank overall (NaN-poisoned Allreduce check)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from climtools import mpi
from mpi_fixtures import check, finish

import xarray as xr

RANK = mpi.comm.rank
SIZE = mpi.comm.size


def gather_labeled(result: xr.DataArray, dim: str) -> tuple[np.ndarray, np.ndarray]:
    """Gather a (possibly dim-distributed) DataArray's values and group-label
    coordinate back to every rank, sorted by the coordinate."""
    group_dim = result.dims[0]
    local_vals = np.asarray(result.values)
    local_coord = np.asarray(result[group_dim].values)
    all_vals = mpi.comm.allgather(local_vals)
    all_coord = mpi.comm.allgather(local_coord)
    vals = np.concatenate(all_vals)
    coord = np.concatenate(all_coord)
    order = np.argsort(coord)
    return coord[order], vals[order]


if __name__ == "__main__":
    n = 24 * SIZE
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    data = np.arange(n, dtype=float) + 1.0  # avoid zero, easier to eyeball

    full = xr.DataArray(data, dims="t", coords={"t": times}, name="v")
    dist = mpi.xarray.repartition(full, dim="t")

    # Blank out day-0 (2020-01-01) entries that live in rank 0's local shard,
    # while day-0 hours on any other rank (if the balanced split ever spreads
    # one day across two ranks) stay valid. With n=24*SIZE and a balanced
    # split, day 0 lands entirely on rank 0, so this makes an entire group's
    # local partial all-NaN on the rank that owns it -- but the *global*
    # group still has zero valid members only if SIZE==1. To get a case
    # where the group is all-NaN on the owning rank yet has valid members
    # elsewhere, duplicate day-0's timestamps onto rank 1 as well by
    # relabeling the last hour of rank 1's local shard to day 0.
    if RANK == 0:
        local_times = dist["t"].values
        day0 = (local_times >= np.datetime64("2020-01-01")) & (
            local_times < np.datetime64("2020-01-02")
        )
        dist.values[day0] = np.nan

    if SIZE > 1 and RANK == 1:
        # Relabel this rank's first local timestamp into day 0 so the group
        # has valid data on rank 1 even though rank 0's contribution to the
        # same group is entirely NaN.
        new_coord = dist["t"].values.copy()
        new_coord[0] = np.datetime64("2020-01-01T00:30:00")
        dist = dist.assign_coords(t=new_coord)

    result = mpi.xarray.resample_reduce(dist, "t", "D", op="min", skipna=True)
    coord, vals = gather_labeled(result, "t")

    day0_idx = np.where(coord == np.datetime64("2020-01-01"))[0]
    if RANK == 0:
        check("exactly one day-0 group in gathered result", len(day0_idx) == 1)
        if len(day0_idx) == 1:
            v = vals[day0_idx[0]]
            if SIZE > 1:
                check(
                    "day-0 min is NOT NaN when another rank contributes valid data "
                    f"(got {v})",
                    not np.isnan(v),
                )
            else:
                check(
                    "day-0 min is NaN with a single rank (no valid data)", np.isnan(v)
                )

    finish()
