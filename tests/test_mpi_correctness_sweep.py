"""Broad correctness sweep: MPI-Xarray public ops vs native xarray.

    mpirun -np <N> --oversubscribe python3 tests/test_mpi_correctness_sweep.py

Uses ``mock_dataset.create_dataset`` (rank 0 builds and writes one NetCDF
file, every rank barriers, then every rank opens it -- via
``mpi_open_dataset`` for the distributed object and plain
``xarray.open_dataset`` for the native reference) rather than building an
in-memory array and broadcasting it, matching how climtools is actually
used and how ``tests/test.sh`` exercises it.

Not exhaustive (see STATUS.md for what remains), but covers the
highest-traffic public operations end to end on both DataArray and
Dataset, with an uneven partition (``n_time=17`` across e.g. 3 or 4
ranks), float32, and NaNs injected via ``.where()`` after opening (so the
same coordinate-based mask applies identically to the native and the
distributed object, with no in-memory array broadcast needed).
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mock_dataset import OUTPUT_DIR, create_dataset

from climtools import mpi
from climtools.xarray.core import unwrap
from climtools.xarray.io import mpi_open_dataset

comm = mpi.comm
rank = comm.rank
size = comm.size

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        failures.append(f"{name}: {detail}")


def close(a, b, **kw):
    return bool(np.allclose(np.asarray(a), np.asarray(b), equal_nan=True, **kw))


def gather(mpi_obj):
    local = unwrap(mpi_obj)
    meta = mpi_obj.meta
    if meta is None:
        return local
    dim = meta["dims"] if isinstance(meta["dims"], str) else meta["dims"][0]
    pieces = comm.allgather(local)
    combined = xr.concat(pieces, dim=dim)
    return combined.sortby(dim) if dim in combined.coords else combined


# --------------------------------------------------------------------------
# Build (rank 0) and open (every rank) the shared mock dataset.
# n_time=17 is deliberately uneven against common rank counts; a coarse
# resolution keeps the grid (lat, lon, plev) tiny so tests run fast.
# --------------------------------------------------------------------------
path = OUTPUT_DIR / "correctness_sweep.nc"
create_dataset(path, n_time=17, resolution_deg=60.0, plev_step=-400.0)

mds = mpi_open_dataset(str(path), mpi, partition_dim="time", log_partitions=False)
native_ds = xr.open_dataset(path)  # every rank opens the same small file directly

# Inject NaNs at a couple of global "time" positions via a coordinate-based
# `.where()` mask -- applies identically whether run against the full
# native object or the distributed one (each rank masks its own local
# slice against the same global condition), so no in-memory broadcast of
# a NaN-holed array is needed.
# Inject NaNs at a couple of global "time" positions into the "pr"
# DataArray specifically -- via a coordinate-based `.where()` mask, so it
# applies identically to the native and the distributed object. Applying
# `.where()` to the whole *Dataset* instead would broadcast the
# "time"-indexed condition against every variable, including the static
# "t" (which doesn't carry "time" at all), silently adding a spurious
# "time" dimension to it -- an xarray broadcasting quirk, not a
# climtools one, but worth avoiding here since a later check relies on
# "t" staying exactly as read.
nan_times = native_ds["time"].isel(time=[2, 9]).values
mda = mds["pr"].where(~mds["pr"].data["time"].isin(nan_times))
native_da = native_ds["pr"].where(~native_ds["pr"]["time"].isin(nan_times))

# --------------------------------------------------------------------------
# Reductions: sum, mean, min, max, first, last, any, all, std, var, prod
# --------------------------------------------------------------------------
for skipna in (True, False):
    tag = f"skipna={skipna}"

    r = gather(mda.sum("time", skipna=skipna))
    n = native_da.sum("time", skipna=skipna)
    check(f"sum DataArray {tag} values", close(r.values, n.values))
    check(f"sum DataArray {tag} dtype", r.dtype == n.dtype, f"{r.dtype} vs {n.dtype}")

    r = gather(mds.sum("time", skipna=skipna))
    n = native_ds["pr"].sum("time", skipna=skipna)
    check(f"sum Dataset.pr {tag} values (no injected NaNs at Dataset level)", close(r["pr"].values, n.values))

    r = gather(mda.mean("time", skipna=skipna))
    n = native_da.mean("time", skipna=skipna)
    check(f"mean DataArray {tag} values", close(r.values, n.values))
    check(f"mean DataArray {tag} dtype", r.dtype == n.dtype, f"{r.dtype} vs {n.dtype}")

    r = gather(mda.min("time", skipna=skipna))
    n = native_da.min("time", skipna=skipna)
    check(f"min DataArray {tag} values", close(r.values, n.values))

    r = gather(mda.max("time", skipna=skipna))
    n = native_da.max("time", skipna=skipna)
    check(f"max DataArray {tag} values", close(r.values, n.values))

    r = gather(mda.std("time", skipna=skipna))
    n = native_da.std("time", skipna=skipna)
    check(f"std DataArray {tag} values", close(r.values, n.values, atol=1e-4))

    r = gather(mda.var("time", skipna=skipna))
    n = native_da.var("time", skipna=skipna)
    check(f"var DataArray {tag} values", close(r.values, n.values, atol=1e-4))

# prod isn't wrapped on MPIXarray directly; call the engine the same way
# core.py's other reduction wrappers do, then wrap the raw result back up
# so it can go through the same gather() every other check uses (reducing
# the sole partition dim can leave the result redistributed along a
# remaining dim, e.g. "lon", rather than replicated -- same as sum/mean
# above).
from climtools.xarray.core import finalize  # noqa: E402

prod_result = gather(finalize(mds._ops.prod(unwrap(mds), "time"), mpi))
native_prod = native_ds.prod("time")
check(
    "prod Dataset.pr values",
    close(prod_result["pr"].values, native_prod["pr"].values, rtol=1e-3),
)

r_first = gather(mda.first("time", skipna=True))
n_first = native_da.reduce(
    lambda a, axis: np.take_along_axis(
        a, np.expand_dims(np.nanargmax(~np.isnan(a), axis=axis), axis=axis), axis=axis
    ).squeeze(axis=axis),
    dim="time",
)
check("first DataArray values", close(r_first.values, n_first.values))

r_last = gather(mda.last("time", skipna=True))
flipped = native_da.isel(time=slice(None, None, -1))
n_last = flipped.reduce(
    lambda a, axis: np.take_along_axis(
        a, np.expand_dims(np.nanargmax(~np.isnan(a), axis=axis), axis=axis), axis=axis
    ).squeeze(axis=axis),
    dim="time",
)
check("last DataArray values", close(r_last.values, n_last.values))

# any()/all() need a boolean-carrying, time-varying condition; derive one
# from "pr" identically on both sides rather than requiring the mock
# schema to carry a dedicated boolean variable.
r_any = gather((mda > mda.mean()).any("time"))
n_any = (native_da > native_da.mean()).any("time")
check("any DataArray (pr > mean) matches native", bool((r_any == n_any).all().item()))

r_all = gather((mda > mda.min() - 1).all("time"))
n_all = (native_da > native_da.min() - 1).all("time")
check("all DataArray matches native", bool((r_all == n_all).all().item()))

# --------------------------------------------------------------------------
# isel / sel
# --------------------------------------------------------------------------
r = gather(mda.isel(lat=slice(0, 2)))
n = native_da.isel(lat=slice(0, 2))
check("isel non-partition dim values", close(r.values, n.values))
check("isel non-partition dim dtype", r.dtype == n.dtype)

lat_vals = native_ds["lat"].values[:2].tolist()
r = gather(mds.sel(lat=lat_vals))
n = native_ds.sel(lat=lat_vals)
check("sel non-partition dim values", close(r["pr"].values, n["pr"].values))

# --------------------------------------------------------------------------
# groupby (reduce) -- labels are rank-local, one per this rank's own local
# "time" slice (see xarray/groupby.py::groupby_reduce's docstring).
# --------------------------------------------------------------------------
local_labels = unwrap(mda)["time"].values.astype(np.int64) % 3
gb_mean = gather(mda.groupby("time", local_labels).mean())
native_labels_full = native_ds["time"].values.astype(np.int64) % 3
native_gb_mean = native_da.groupby(
    xr.DataArray(native_labels_full, dims="time", name="grp")
).mean()
check(
    "groupby mean matches native (sorted by group)",
    close(
        gb_mean.sortby(gb_mean.dims[0]).values,
        native_gb_mean.sortby(native_gb_mean.dims[0]).values,
    ),
    f"shapes {gb_mean.shape} vs {native_gb_mean.shape}",
)

# --------------------------------------------------------------------------
# rolling
# --------------------------------------------------------------------------
r = gather(mda.rolling_reduce("time", window=3, reduce="mean", center=True, min_periods=1))
n = native_da.rolling(time=3, center=True, min_periods=1).mean()
check(
    "rolling mean values",
    close(r.values, n.values),
    f"maxdiff={np.nanmax(np.abs(np.asarray(r.values) - np.asarray(n.values)))}",
)

r = gather(mda.rolling(dim="time", window=3, center=True, min_periods=1).sum())
n = native_da.rolling(time=3, center=True, min_periods=1).sum()
check("rolling sum via handle values", close(r.values, n.values))

# --------------------------------------------------------------------------
# matmul -- a small in-memory coefficient matrix (not mock-dataset-scale
# data in its own right, so built directly rather than via a NetCDF mock).
# --------------------------------------------------------------------------
n_lat = native_ds.sizes["lat"]
coeff = None
if rank == 0:
    coeff = xr.DataArray(
        np.random.default_rng(0).normal(size=(n_lat, 4)).astype(np.float32),
        dims=("lat", "k"),
    )
native_coeff = comm.bcast(coeff if rank == 0 else None, root=0)
r = gather(mda.matmul(native_coeff))
n = xr.dot(native_da, native_coeff, dims="lat")
check(
    "matmul values",
    close(r.transpose(*n.dims).values, n.values),
    f"shapes {r.shape} vs {n.shape}",
)

# --------------------------------------------------------------------------
# align -- rank-ownership reconciliation, not xarray's label-based join;
# no direct native counterpart (see the module comment above
# Arithmetic._shuffle_by_position in xarray/arithmetic.py), so this checks
# self-consistency rather than an inapplicable native comparison, per the
# task's own guidance for MPI-specific operations.
# --------------------------------------------------------------------------
mds_by_lat = mpi_open_dataset(str(path), mpi, partition_dim="lat", log_partitions=False)
al_a, al_b = mda.align(mds_by_lat["t2m"])
r_a, r_b = gather(al_a), gather(al_b)
check(
    "align: left values unchanged by ownership reconciliation",
    close(r_a.values, native_da.values),
)
check(
    "align: right values unchanged by ownership reconciliation",
    close(r_b.values, native_ds["t2m"].values),
)
check(
    "align: both operands share the same partition dimension afterward",
    al_a.meta["dims"] == al_b.meta["dims"],
    f"{al_a.meta['dims']} vs {al_b.meta['dims']}",
)

# --------------------------------------------------------------------------
# static (non-partition-dim) variable survives reindex unchanged -- "t"
# carries (plev, lat, lon), not "time", exactly like the dedicated
# test_mpi_reindex_static_var.py case, exercised here against the shared
# mock file's own static variable instead of a bespoke one.
# --------------------------------------------------------------------------
new_time = np.concatenate([native_ds["time"].values[:1] - 2, native_ds["time"].values])
reind = mds.reindex(time=new_time)
local_reind = unwrap(reind)
check(
    "reindex: static (non-time) variable 't' unchanged",
    close(local_reind["t"].values, unwrap(mds)["t"].values),
)

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
comm.Barrier()
all_failures = comm.allgather(failures)
flat = [f"[rank {r}] {msg}" for r, fs in enumerate(all_failures) for msg in fs]

if rank == 0:
    if flat:
        print(f"FAILED ({len(flat)}):")
        for msg in flat:
            print(" -", msg)
    else:
        print(f"PASSED on {size} ranks")

comm.Barrier()
sys.exit(1 if flat else 0)
