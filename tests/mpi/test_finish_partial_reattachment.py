"""Test the ReductionPlanningMixin._finish() partial-survival path.

Reducing away exactly one of two active partition dimensions (here:
sum(dim="lat") under a (lat, lon) partition) leaves "lon" both present
and still genuinely partitioned. This path is provably unreachable in the
one-dimensional implementation (removing "the" partition dimension always
removes every partition dimension there), so it is new, multi-dimension-
only code, verified here to reattach correct metadata for the surviving
axis and correctly drop the now-stale Cartesian "cart" descriptor.

Run: mpirun -n 6 python3 test_finish_partial_reattachment.py
"""

import numpy as np
from climtools.mpi.runtime import MPIRuntime
from climtools.xarray import cartesian
from climtools.xarray.meta import get_mpi_meta, set_mpi_meta
from climtools.xarray.reductions import Reduction
from mpi4py import MPI

import xarray as xr


class Ops(Reduction):
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

# partition_dim="auto" (the default) exercises the new _finish partial-
# reattachment path: "lat" is reduced away, "lon" survives untouched.
r = ops.sum(da, dim="lat")
meta = get_mpi_meta(r)
assert meta is not None, (
    "result should still carry mpi_meta for the surviving 'lon' axis"
)
assert meta["dims"] == ("lon",), meta["dims"]
assert meta["starts"]["lon"] == lon0 and meta["stops"]["lon"] == lon1
assert "cart" not in meta, (
    "cart descriptor should be dropped once only 1 of 2 axes survives"
)
expected = full_global.sum(axis=0)[lon0:lon1]
assert np.allclose(r.values, expected), (r.values, expected)
assert int(r.sizes["lon"]) == lon1 - lon0

print(
    f"rank {comm.rank}: partial-reattachment path OK, meta={meta['dims']} "
    f"bounds={meta['starts']}:{meta['stops']}"
)
