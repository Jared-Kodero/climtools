"""Edge case: reindex() on a Dataset with a variable that doesn't carry the
reindexed dimension at all (e.g. the mock dataset's "t", a static
(plev, lat, lon) profile under a "time" partition) -- must come through as
this rank's own unchanged local copy, not be overwritten by _fill_chunk.
See its docstring in xarray/arithmetic.py. This is also exercised inline
inside test_mpi_correctness_sweep.py; kept here too as a minimal, focused
standalone reproduction of just this one case.

    mpirun -np <N> --oversubscribe python3 tests/test_mpi_reindex_static_var.py

Uses ``mock_dataset.create_dataset`` -- rank 0 builds and writes the
shared mock NetCDF file, every rank barriers, then every rank opens it.
"""

from __future__ import annotations

import sys

import numpy as np
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


path = OUTPUT_DIR / "reindex_static_var.nc"
create_dataset(path, n_time=13, resolution_deg=90.0, plev_step=-500.0)

mds = mpi_open_dataset(str(path), mpi, partition_dim="time", log_partitions=False)
before = unwrap(mds)["t"].values.copy()

new_time = np.arange(-2, 20)
result = mds.reindex(time=new_time)
after = unwrap(result)["t"].values

check(
    "static (non-time) variable 't' unchanged on every rank",
    bool(np.array_equal(after, before)),
    f"max abs diff: {np.max(np.abs(after - before)) if after.shape == before.shape else 'shape mismatch'}",
)
check(
    "static variable dtype unchanged",
    unwrap(result)["t"].dtype == unwrap(mds)["t"].dtype,
    f"got {unwrap(result)['t'].dtype}",
)

comm.Barrier()
all_failures = comm.allgather(failures)
flat = [f"[rank {r}] {msg}" for r, fs in enumerate(all_failures) for msg in fs]
if rank == 0:
    print(
        (f"FAILED ({len(flat)}):\n" + "\n".join(" - " + m for m in flat))
        if flat
        else f"PASSED on {size} ranks"
    )
comm.Barrier()
sys.exit(1 if flat else 0)
