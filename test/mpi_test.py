"""MPI-vs-native correctness suite: single entry point.

The one script test.sh invokes via srun; run locally with:
    mpirun --oversubscribe -n <N> python mpi_test.py

Builds the shared fixtures once (mpi_test_common.build_fixtures), then
imports and runs each mpi_test_*.py module in turn -- each owns one
coherent area rather than everything living in a single file:

  mpi_test_construction.py  construction: mpi_open_dataset,
                             mpi_create_dataarray, mpi_create_dataset;
                             even/uneven/multi-dim partitioning
  mpi_test_reductions.py    mean/sum/min/max/var/std/median; rank-local,
                             reconstruction, and multi-dim dedup+coverage
  mpi_test_halo_ops.py      rolling_reduce, coarsen_reduce, diff, shift,
                             differentiate, ffill, bfill
  mpi_test_scans.py         NumPy dispatch, isel, cumsum, sortby,
                             reindex, interp, matmul

Every numeric check compares this rank's local slice against the
matching slice of a plain, non-distributed xarray/numpy computation --
never against a "rank 0 holds everything" assumption. Multi-dimensional
checks additionally verify reconstruction correctness (exact,
non-overlapping global coverage) and rank consistency (no two ranks
claim the same range).
"""

from __future__ import annotations

import mpi_test_construction
import mpi_test_halo_ops
import mpi_test_reductions
import mpi_test_scans
from mpi_test_common import build_fixtures, report

fixtures = build_fixtures()

mpi_test_construction.run(fixtures)
mpi_test_reductions.run(fixtures)
mpi_test_halo_ops.run(fixtures)
mpi_test_scans.run(fixtures)

report()
