"""Correctness sweep for shift() and differentiate() (added by the reviewed
patch), against native xarray, across uneven partitions.

    mpirun -np <N> --oversubscribe python3 tests/test_mpi_shift_differentiate.py

Uses ``mock_dataset.create_multitype_dataset``'s "var32" -- a smooth,
nonlinear float32 signal (not just a linear ramp, so edge_order=1 and
edge_order=2 differentiate results actually differ) -- rather than
building the array in memory and broadcasting it.
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr
from mock_dataset import OUTPUT_DIR, create_multitype_dataset

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
    dim = meta["dims"] if isinstance(meta["dims"], str) else meta["dims"][0]
    pieces = comm.allgather(local)
    return xr.concat(pieces, dim=dim).sortby(dim)


N = 23  # uneven across 2-6 ranks

path = OUTPUT_DIR / "shift_differentiate.nc"
create_multitype_dataset(path, n=N, seed=7)

mds = mpi_open_dataset(str(path), mpi, partition_dim="x", log_partitions=False)
mda = mds["var32"]
native_da = xr.open_dataset(path)["var32"]

# -- shift: several periods, default and explicit fill_value ---------------
for periods in (1, -1, 2, -3):
    r = gather(mda.shift("x", periods))
    n = native_da.shift(x=periods)
    check(
        f"shift periods={periods} values",
        close(r.values, n.values),
        f"maxdiff (ignoring nan) = "
        f"{np.nanmax(np.abs(np.asarray(r.values) - np.asarray(n.values))) if np.isfinite(np.asarray(r.values)).any() else 'n/a'}",
    )
    check(f"shift periods={periods} dtype", r.dtype == n.dtype, f"{r.dtype} vs {n.dtype}")

r = gather(mda.shift("x", 2, fill_value=0.0))
n = native_da.shift(x=2, fill_value=0.0)
check("shift with explicit fill_value values", close(r.values, n.values))
check("shift with explicit fill_value dtype (stays float32)", r.dtype == np.float32 == n.dtype, f"{r.dtype}")

# -- differentiate: edge_order 1 and 2 --------------------------------------
for edge_order in (1, 2):
    r = gather(mda.differentiate("x", edge_order=edge_order))
    n = native_da.differentiate("x", edge_order=edge_order)
    check(
        f"differentiate edge_order={edge_order} values",
        close(r.values, n.values, atol=1e-4),
        f"maxdiff={np.nanmax(np.abs(np.asarray(r.values) - np.asarray(n.values)))}",
    )

comm.Barrier()
all_failures = comm.allgather(failures)
flat = [f"[rank {r}] {msg}" for r, fs in enumerate(all_failures) for msg in fs]

if rank == 0:
    if flat:
        print(f"FAILED ({len(flat)}):")
        for msg in flat:
            print(" -", msg)
    else:
        print(f"PASSED on {size} ranks, N={N}")

comm.Barrier()
sys.exit(1 if flat else 0)
