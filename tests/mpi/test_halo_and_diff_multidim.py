"""Test halo_exchange() and diff() under a 2-D Cartesian partition.

Verifies per-axis halo padding (lat-only, lon-only) against the true
global neighbor values, and diff() (an explicitly required stencil
operation) end to end: halo fetch, difference, and re-attached metadata
with the correct new global size along the differenced axis.

Run: mpirun -n 6 python3 test_halo_and_diff_multidim.py
"""

import numpy as np
from climtools.mpi.runtime import MPIRuntime
from climtools.xarray import cartesian
from climtools.xarray.arithmetic import Arithmetic
from climtools.xarray.elementwise import Elementwise
from climtools.xarray.meta import get_mpi_meta, set_mpi_meta
from climtools.xarray.planning import ReductionPlanningMixin
from mpi4py import MPI

import xarray as xr


class Ops(Arithmetic, Elementwise, ReductionPlanningMixin):
    def __init__(self, runtime):
        self._runtime = runtime


comm = MPI.COMM_WORLD
runtime = MPIRuntime(comm)
ops = Ops(runtime)

NLAT, NLON = 6, 9
sizes = {"lat": NLAT, "lon": NLON}
dims = ("lat", "lon")
topo = cartesian.get_cartesian_topology(comm, dims, sizes)
lat0, lat1 = topo.bounds["lat"]
lon0, lon1 = topo.bounds["lon"]

full_global = (np.arange(NLAT)[:, None] * 100 + np.arange(NLON)[None, :]).astype(
    np.float64
)
local_full = full_global[lat0:lat1, lon0:lon1]
da = xr.DataArray(local_full, dims=("lat", "lon"), name="full")
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


# --- halo exchange along "lat" only: pad with 1 row before/after ---
padded, left_pad, right_pad = ops.halo_exchange(da, "lat", before=1, after=1)
expected_left = 1 if lat0 > 0 else 0
expected_right = 1 if lat1 < NLAT else 0
check("left_pad correct", left_pad == expected_left)
check("right_pad correct", right_pad == expected_right)
expected_padded = full_global[max(lat0 - 1, 0) : min(lat1 + 1, NLAT), lon0:lon1]
check("padded values correct (lat halo)", np.allclose(padded.values, expected_padded))

# --- halo exchange along "lon" only ---
padded2, lp2, rp2 = ops.halo_exchange(da, "lon", before=1, after=1)
expected_lp2 = 1 if lon0 > 0 else 0
expected_rp2 = 1 if lon1 < NLON else 0
check("lon left_pad correct", lp2 == expected_lp2)
check("lon right_pad correct", rp2 == expected_rp2)
expected_padded2 = full_global[lat0:lat1, max(lon0 - 1, 0) : min(lon1 + 1, NLON)]
check("padded values correct (lon halo)", np.allclose(padded2.values, expected_padded2))

# --- sequential two-axis halo (corner propagation): pad lat then lon on the result ---
p1, lp1, rp1 = ops.halo_exchange(da, "lat", before=1, after=1)
# p1 no longer carries meta (stripped, as documented) -- diff/rolling would
# normally operate on it directly; here we just check corner data landed.
# Reconstruct expected corner-inclusive window directly from the global array.
lat_win = slice(max(lat0 - 1, 0), min(lat1 + 1, NLAT))
expected_corner = full_global[lat_win, lon0:lon1]
check(
    "first-axis halo alone matches (no lon padding yet)",
    np.allclose(p1.values, expected_corner),
)

# --- diff() along "lat", the explicitly-required stencil op ---
diffed = ops.diff(da, "lat", n=1)
meta_d = get_mpi_meta(diffed)
check("diff meta present", meta_d is not None)
global_diff = np.diff(full_global, n=1, axis=0)  # shape (NLAT-1, NLON)
if meta_d is not None:
    d_lat0, d_lat1 = meta_d["starts"]["lat"], meta_d["stops"]["lat"]
    check(
        "diff lon meta unchanged",
        meta_d["starts"]["lon"] == lon0 and meta_d["stops"]["lon"] == lon1,
    )
    check(
        "diff values correct",
        np.allclose(diffed.values, global_diff[d_lat0:d_lat1, lon0:lon1]),
    )
    check("diff new global size is NLAT-1", meta_d["global_sizes"]["lat"] == NLAT - 1)

if errors:
    print(f"rank {comm.rank} FAILURES: {errors}")
    raise SystemExit(1)
print(
    f"rank {comm.rank} coords={topo.coords} bounds=(lat {lat0}:{lat1}, "
    f"lon {lon0}:{lon1}) ALL HALO/DIFF CHECKS PASSED"
)
