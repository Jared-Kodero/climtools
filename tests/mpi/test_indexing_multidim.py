"""Test isel()/sel()/isel_scalar() on a second (non-first) partition axis.

This is the regression test for a real bug: prior to this patch, indexing
a dimension other than the first partition dimension silently lost or
corrupted mpi_meta, and any cross-rank "who owns this index" lookup was
scoped to the full communicator instead of the correct Cartesian
sub-communicator.

Run: mpirun -n 6 python3 test_indexing_multidim.py
"""

import numpy as np
from climtools.mpi.runtime import MPIRuntime
from climtools.xarray import cartesian
from climtools.xarray.indexing import Indexing
from climtools.xarray.meta import get_mpi_meta, set_mpi_meta
from mpi4py import MPI

import xarray as xr


class Ops(Indexing):
    def __init__(self, runtime):
        self._runtime = runtime


comm = MPI.COMM_WORLD
runtime = MPIRuntime(comm)
ops = Ops(runtime)

NLAT, NLON = 8, 12
sizes = {"lat": NLAT, "lon": NLON}
dims = ("lat", "lon")
topo = cartesian.get_cartesian_topology(comm, dims, sizes)
lat0, lat1 = topo.bounds["lat"]
lon0, lon1 = topo.bounds["lon"]

full_global = np.arange(NLAT * NLON, dtype=np.float64).reshape(NLAT, NLON)
local_full = full_global[lat0:lat1, lon0:lon1]
da = xr.DataArray(
    local_full,
    dims=("lat", "lon"),
    coords={"lat": np.arange(lat0, lat1), "lon": np.arange(lon0, lon1)},
    name="full",
)
set_mpi_meta(
    da,
    dim=dims,
    global_size=sizes,
    start={"lat": lat0, "lon": lon0},
    stop={"lat": lat1, "lon": lon1},
    chunk_info={},
    cart=topo.as_meta_cart(),
)

errors = []


def check(name, cond):
    if not cond:
        errors.append(name)


# --- slice on the SECOND partition dim (lon) -- this is exactly the bug ---
sliced = ops.isel(da, lon=slice(2, 5))
meta_s = get_mpi_meta(sliced)
check("meta survives lon-slice", meta_s is not None)
if meta_s is not None:
    expected_lon_local_start = max(2, lon0) - lon0
    expected_lon_local_stop = min(5, lon1) - lon0
    expected_lon_local_stop = max(expected_lon_local_start, expected_lon_local_stop)
    expected_local_lon_size = expected_lon_local_stop - expected_lon_local_start
    check("lon local size correct", int(sliced.sizes["lon"]) == expected_local_lon_size)
    check(
        "lat meta unchanged",
        meta_s["starts"]["lat"] == lat0 and meta_s["stops"]["lat"] == lat1,
    )
    check("lon meta reflects new global size (3)", meta_s["global_sizes"]["lon"] == 3)
    # cross-check values against the true global slice for this rank's lat range
    global_expected = full_global[lat0:lat1, 2:5]
    lon_new0, lon_new1 = meta_s["starts"]["lon"], meta_s["stops"]["lon"]
    check(
        "sliced values correct",
        np.allclose(sliced.values, global_expected[:, lon_new0:lon_new1]),
    )

# --- scalar select on the SECOND partition dim (lon); broadcasts within
# the lat-fixed lon-group only ---
scalar_sel = ops.isel_scalar(da, "lon", 5, {})
expected_col = full_global[lat0:lat1, 5]
check(
    "isel_scalar lon selects correct column",
    np.allclose(scalar_sel.values, expected_col),
)
meta_scalar = get_mpi_meta(scalar_sel)
check(
    "isel_scalar leaves lat meta intact",
    meta_scalar is not None and meta_scalar["dims"] == ("lat",),
)
if meta_scalar is not None:
    check(
        "isel_scalar lat bounds preserved",
        meta_scalar["starts"]["lat"] == lat0 and meta_scalar["stops"]["lat"] == lat1,
    )

# --- sel() with coordinate labels on lon ---
label_sliced = ops.sel(da, lon=slice(3, 6))
expected_label = full_global[lat0:lat1][
    :, (np.arange(NLON) >= 3) & (np.arange(NLON) <= 6)
]
lon_labels_local = np.arange(lon0, lon1)
mask = (lon_labels_local >= 3) & (lon_labels_local <= 6)
check(
    "sel slice on lon correct values",
    np.allclose(label_sliced.values, local_full[:, mask]),
)

if errors:
    print(f"rank {comm.rank} FAILURES: {errors}")
    raise SystemExit(1)
print(
    f"rank {comm.rank} coords={topo.coords} bounds=(lat {lat0}:{lat1}, "
    f"lon {lon0}:{lon1}) ALL INDEXING CHECKS PASSED"
)
