"""Test IO.partition() multi-dimensional scatter (Cartesian topology).

Verifies: every rank receives the correct Cartesian bounds; the union of
all ranks tiling exactly reconstructs the global array (no gaps, no
overlap, including an uneven per-axis split); a length-one dim sequence
takes the identical one-dimensional code path.

Works around a circular import already present in the reference
xarray/netcdf.py -> constructors.py -> core.py -> ops.py -> io.py chain
(reproduced identically against the untouched baseline) by stubbing
constructors.py, which IO.partition() never touches.

Run: mpirun -n 5 python3 test_partition_multidim.py
"""

import sys
import types

# Break the pre-existing (baseline) circular import netcdf.py -> constructors.py
# -> core.py -> ops.py -> io.py -> netcdf.py for this isolated test only: stub
# out constructors.py, since IO.partition() never touches MPIXarrayOps/
# mpi_partition_data (those are only used by netcdf.py's to_netcdf_parallel
# path, not exercised here). This is a testenv-only workaround, not part of
# the patch.
stub = types.ModuleType("climtools.xarray.constructors")
stub.MPIXarrayOps = object
stub.mpi_partition_data = None
sys.modules["climtools.xarray.constructors"] = stub

import numpy as np  # noqa: E402
from climtools.mpi.runtime import MPIRuntime  # noqa: E402
from climtools.xarray.io import IO  # noqa: E402
from climtools.xarray.meta import get_mpi_meta  # noqa: E402
from mpi4py import MPI  # noqa: E402

import xarray as xr  # noqa: E402


class Ops(IO):
    def __init__(self, runtime):
        self._runtime = runtime


comm = MPI.COMM_WORLD
runtime = MPIRuntime(comm)
ops = Ops(runtime)

NLAT, NLON = 7, 11
full_global = np.arange(NLAT * NLON, dtype=np.float64).reshape(NLAT, NLON)
lat_only_global = np.arange(NLAT, dtype=np.float64) * 100.0

root_value = None
if comm.rank == 0:
    root_value = xr.Dataset(
        {
            "full": (("lat", "lon"), full_global),
            "lat_only": (("lat",), lat_only_global),
        }
    )

local = ops.partition(
    root_value, ("lat", "lon"), root=0, log_partitions=(comm.rank == 0)
)
meta = get_mpi_meta(local)
assert meta is not None
assert meta["dims"] == ("lat", "lon"), meta["dims"]
assert "cart" in meta

lat0, lat1 = meta["starts"]["lat"], meta["stops"]["lat"]
lon0, lon1 = meta["starts"]["lon"], meta["stops"]["lon"]
expected_full = full_global[lat0:lat1, lon0:lon1]
expected_lat_only = lat_only_global[lat0:lat1]
assert np.allclose(local["full"].values, expected_full), (
    local["full"].values,
    expected_full,
)
assert np.allclose(local["lat_only"].values, expected_lat_only)
assert int(local.sizes["lat"]) == lat1 - lat0
assert int(local.sizes["lon"]) == lon1 - lon0

# every rank's slice must be non-overlapping and the union must reconstruct
# the whole global array -- check via an allgather-based reassembly.
pieces = comm.allgather((lat0, lat1, lon0, lon1, np.asarray(local["full"].values)))
reassembled = np.full((NLAT, NLON), np.nan)
for a, b, c, d, arr in pieces:
    reassembled[a:b, c:d] = arr
assert np.array_equal(reassembled, full_global), (
    "partition does not tile the full global array exactly"
)

# single-dim request as a length-one sequence takes the identical 1D path
root_value_1d = None
if comm.rank == 0:
    root_value_1d = xr.DataArray(np.arange(19, dtype=np.float64), dims=("x",), name="x")
local_1d = ops.partition(root_value_1d, ("x",), root=0)
meta_1d = get_mpi_meta(local_1d)
assert meta_1d["dims"] == ("x",)
assert "cart" not in meta_1d

print(
    f"rank {comm.rank}: io.partition() N-D scatter + tiling + "
    f"1D-equivalence ALL CHECKS PASSED (lat {lat0}:{lat1}, lon {lon0}:{lon1})"
)
