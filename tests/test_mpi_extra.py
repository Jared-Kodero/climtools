"""Supplementary edge-case probes not covered by the shipped suite:
all-NaN partitions, boolean dtype reductions, more ranks than elements,
zero-length dims, first/last with fully-empty local shards."""

from __future__ import annotations

import numpy as np
import pandas as pd
from climtools import mpi
from mpi_fixtures import check, finish

import xarray as xr

RANK = mpi.comm.rank
SIZE = mpi.comm.size


def make_series_allnan(n=12):
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    data = np.full(n, np.nan)
    return xr.DataArray(data, dims="t", coords={"t": times}, name="v")


def make_bool_series(n=12, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    data = rng.integers(0, 2, size=n).astype(bool)
    return xr.DataArray(data, dims="t", coords={"t": times}, name="v")


if __name__ == "__main__":
    # --- all-NaN reductions ---
    full = make_series_allnan(12)
    dist = mpi.xarray.repartition(full, dim="t")

    r = mpi.xarray.min(dist, dim="t")
    check("all-NaN min is NaN", bool(np.isnan(r.values)))

    r = mpi.xarray.max(dist, dim="t")
    check("all-NaN max is NaN", bool(np.isnan(r.values)))

    r = mpi.xarray.first(dist, dim="t")
    check("all-NaN first is NaN", bool(np.isnan(r.values)))

    r = mpi.xarray.last(dist, dim="t")
    check("all-NaN last is NaN", bool(np.isnan(r.values)))

    r = mpi.xarray.mean(dist, dim="t")
    check("all-NaN mean is NaN", bool(np.isnan(r.values)))

    r = mpi.xarray.sum(dist, dim="t")
    ref = full.sum(dim="t", skipna=True)
    check("all-NaN sum matches serial (0.0)", bool(r.values == ref.values))

    # --- boolean any/all ---
    bfull = make_bool_series(16, seed=3)
    bdist = mpi.xarray.repartition(bfull, dim="t")
    r_any = mpi.xarray.any(bdist, dim="t")
    r_all = mpi.xarray.all(bdist, dim="t")
    check("bool any matches serial", bool(r_any.values) == bool(bfull.any().values))
    check("bool all matches serial", bool(r_all.values) == bool(bfull.all().values))

    # --- all-False boolean any ---
    bfalse = xr.DataArray(
        np.zeros(10, dtype=bool),
        dims="t",
        coords={"t": pd.date_range("2020-01-01", periods=10, freq="h")},
        name="v",
    )
    bfalse_dist = mpi.xarray.repartition(bfalse, dim="t")
    r_any = mpi.xarray.any(bfalse_dist, dim="t")
    check("all-False any is False", bool(r_any.values) is False)

    # --- more ranks than elements (some ranks own 0 elements) ---
    tiny = make_series_allnan(3) * 0 + 1.0  # all ones, length 3
    tiny.values[:] = np.arange(1, 4)
    tiny_dist = mpi.xarray.repartition(tiny, dim="t")
    r = mpi.xarray.sum(tiny_dist, dim="t")
    check(
        f"length-3 sum matches serial with {SIZE} ranks (empty ranks handled)",
        float(r.values) == 6.0,
    )
    r = mpi.xarray.first(tiny_dist, dim="t")
    check(
        "length-3 first == 1 (empty local shard on some ranks)", float(r.values) == 1.0
    )
    r = mpi.xarray.last(tiny_dist, dim="t")
    check(
        "length-3 last == 3 (empty local shard on some ranks)", float(r.values) == 3.0
    )

    # --- single NaN at the very first/last global index, skipna first/last ---
    nan_first = make_series_allnan(1)  # placeholder, replaced below
    n = 10
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    data = np.arange(n, dtype=float)
    data[0] = np.nan
    data[-1] = np.nan
    edge = xr.DataArray(data, dims="t", coords={"t": times}, name="v")
    edge_dist = mpi.xarray.repartition(edge, dim="t")
    r = mpi.xarray.first(edge_dist, dim="t", skipna=True)
    check("first skips leading NaN", float(r.values) == 1.0)
    r = mpi.xarray.last(edge_dist, dim="t", skipna=True)
    check("last skips trailing NaN", float(r.values) == float(n - 2))

    finish()
