"""Test sum/mean/min/var/groupby_reduce under a 2-D Cartesian partition.

The central correctness test: a variable owning both partition axes and
one replicated along only "lat" (present under a (lat, lon) partition but
missing the "lon" dimension) are reduced together in the same Dataset
call. The replicated variable must NOT be double-counted across the
lon-varying ranks that hold identical copies of it -- this is the
correctness hazard a naive full-communicator Allreduce would introduce
under multi-dimensional decomposition; see the PlanEntry.replica_count
mechanism in planning.py.

Also exercises var(ddof=1), which is the case where the sum/count
cancellation that makes plain mean() safe without correction does NOT
apply (ddof is a bare constant, not itself duplicated), so the count
genuinely needs the replica_count correction on its own.

Run: mpirun -n 6 python3 test_reductions_multidim.py
"""

import numpy as np
from climtools.mpi.runtime import MPIRuntime
from climtools.xarray import cartesian
from climtools.xarray.groupby import Groupby
from climtools.xarray.meta import set_mpi_meta
from climtools.xarray.reductions import Reduction
from climtools.xarray.statistics import Statistics
from mpi4py import MPI

import xarray as xr


class Ops(Reduction, Groupby, Statistics):
    def __init__(self, runtime):
        self._runtime = runtime


comm = MPI.COMM_WORLD
runtime = MPIRuntime(comm)
ops = Ops(runtime)

NLAT, NLON = 6, 10
sizes = {"lat": NLAT, "lon": NLON}
dims = ("lat", "lon")
topo = cartesian.get_cartesian_topology(comm, dims, sizes)
lat0, lat1 = topo.bounds["lat"]
lon0, lon1 = topo.bounds["lon"]

full_global = np.arange(NLAT * NLON, dtype=np.float64).reshape(NLAT, NLON)
lat_only_global = np.arange(NLAT, dtype=np.float64) * 100.0  # depends only on lat

local_full = full_global[lat0:lat1, lon0:lon1]
local_lat_only = lat_only_global[lat0:lat1]

ds = xr.Dataset(
    {
        "full": (("lat", "lon"), local_full),
        "lat_only": (("lat",), local_lat_only),
    }
)
set_mpi_meta(
    ds,
    dim=dims,
    global_size=sizes,
    start={"lat": lat0, "lon": lon0},
    stop={"lat": lat1, "lon": lon1},
    chunk_info={},
    cart=topo.as_meta_cart(),
)

errors = []


def check(name, got, expected):
    got = np.asarray(got)
    expected = np.asarray(expected)
    if not np.allclose(got, expected):
        errors.append(f"{name}: got {got!r} expected {expected!r}")


# --- reduce over "lat" only (leaves "lon" partitioned, "full" var genuinely
# reduced, "lat_only" var is replicated along lon and must NOT double count) ---
r = ops.sum(ds, dim="lat", partition_dim=None)
check(
    "sum(dim=lat) full",
    r["full"].values,
    full_global.sum(axis=0)[lon0:lon1],
)
check(
    "sum(dim=lat) lat_only (must equal single true total, not N-lon-groups-times)",
    r["lat_only"].values,
    lat_only_global.sum(),
)

# --- reduce over everything (both partition dims) ---
r_all = ops.sum(ds, partition_dim=None)
check("sum() full total", r_all["full"].values, full_global.sum())
check("sum() lat_only total", r_all["lat_only"].values, lat_only_global.sum())

# --- mean over lat ---
r_mean = ops.mean(ds, dim="lat", partition_dim=None)
check(
    "mean(dim=lat) full",
    r_mean["full"].values,
    full_global.mean(axis=0)[lon0:lon1],
)
check("mean(dim=lat) lat_only", r_mean["lat_only"].values, lat_only_global.mean())

# --- min/max over lat ---
r_min = ops.min(ds, dim="lat", partition_dim=None)
check("min(dim=lat) full", r_min["full"].values, full_global.min(axis=0)[lon0:lon1])
check("min(dim=lat) lat_only", r_min["lat_only"].values, lat_only_global.min())

# --- var/std over lat (ddof=1 exercises the count-correction path) ---
r_var = ops.var(ds, dim="lat", ddof=1, partition_dim=None)
check(
    "var(dim=lat, ddof=1) full",
    r_var["full"].values,
    full_global.var(axis=0, ddof=1)[lon0:lon1],
)
check(
    "var(dim=lat, ddof=1) lat_only",
    r_var["lat_only"].values,
    lat_only_global.var(ddof=1),
)

# --- groupby_reduce over "lat" (grouped into 2 bins by parity) ---
local_lat_labels = np.arange(lat0, lat1) % 2
gb = ops.groupby_reduce(ds, "lat", local_lat_labels, op="sum", partition_dim=None)
global_lat_labels = np.arange(NLAT) % 2
expected_full_g0 = full_global[global_lat_labels == 0].sum(axis=0)[lon0:lon1]
expected_full_g1 = full_global[global_lat_labels == 1].sum(axis=0)[lon0:lon1]
expected_lat_only_g0 = lat_only_global[global_lat_labels == 0].sum()
expected_lat_only_g1 = lat_only_global[global_lat_labels == 1].sum()
order = np.argsort(gb["_mpi_group"].values)
full_sorted = gb["full"].values[order]
lat_only_sorted = gb["lat_only"].values[order]
check("groupby full g0", full_sorted[0], expected_full_g0)
check("groupby full g1", full_sorted[1], expected_full_g1)
check("groupby lat_only g0", lat_only_sorted[0], expected_lat_only_g0)
check("groupby lat_only g1", lat_only_sorted[1], expected_lat_only_g1)

if errors:
    print(f"rank {comm.rank} FAILURES:\n  " + "\n  ".join(errors))
    raise SystemExit(1)
print(
    f"rank {comm.rank} coords={topo.coords} "
    f"bounds=({lat0}:{lat1},{lon0}:{lon1}) ALL CHECKS PASSED"
)
