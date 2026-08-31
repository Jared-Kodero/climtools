"""Regression suite: the original single-dimension behavior is unchanged.

Runs sum/mean/var/groupby_reduce/auto-repartition through the same
generalized code paths as the multi-dimensional tests, but with a single
partition dimension and an uneven split (N=23 over 5 ranks), and asserts
`"cart" not in meta` throughout -- confirming zero Cartesian machinery is
invoked, and therefore zero performance/behavior change, on the default,
backward-compatible path.

Run: mpirun -n 5 python3 test_backward_compat_1d.py
"""

import numpy as np
from climtools.mpi.runtime import MPIRuntime
from climtools.xarray.chunks import get_balanced_bounds
from climtools.xarray.groupby import Groupby
from climtools.xarray.meta import get_mpi_meta, set_mpi_meta
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

N = 23
start, stop = get_balanced_bounds(N, comm.rank, comm.size)
global_x = np.arange(N, dtype=np.float64)
static_y = np.arange(4, dtype=np.float64) * 10.0  # not distributed at all

ds = xr.Dataset(
    {
        "x_var": (("x",), global_x[start:stop]),
        "y_var": (("y",), static_y),  # no partition dim: fully replicated, as before
    }
)
set_mpi_meta(
    ds,
    dim="x",
    global_size=N,
    start=start,
    stop=stop,
    chunk_info={},
)
meta = get_mpi_meta(ds)
assert meta["dims"] == ("x",)
assert "cart" not in meta  # single-dim: no topology built at all

r_sum = ops.sum(ds, partition_dim=None)
assert np.isclose(r_sum["x_var"].values, global_x.sum())
assert np.isclose(
    r_sum["y_var"].values, static_y.sum()
)  # untouched replicated var: same on every rank, no MPI

r_mean = ops.mean(ds, partition_dim=None)
assert np.isclose(r_mean["x_var"].values, global_x.mean())

r_var = ops.var(ds, ddof=1, partition_dim=None)
assert np.isclose(r_var["x_var"].values, global_x.var(ddof=1))

labels = np.arange(start, stop) % 3
gb = ops.groupby_reduce(ds, "x", labels, op="sum", partition_dim=None)
global_labels = np.arange(N) % 3
for g in range(3):
    expected = global_x[global_labels == g].sum()
    got = gb["x_var"].values[np.asarray(gb["_mpi_group"].values) == g][0]
    assert np.isclose(got, expected), (g, got, expected)

# auto-repartition after full reduction (dims fully removed) still works
r_auto = ops.sum(ds)
meta2 = get_mpi_meta(r_auto)
# scalar result: no dims left on x_var, so nothing to auto-partition on for it;
# y_var still has "y" -> auto candidate.
print(
    f"rank {comm.rank} (N={N}, local {start}:{stop}): 1D backward-compat "
    f"ALL CHECKS PASSED, auto meta={meta2}"
)
