# climtools

Utilities for climate data analysis and plotting. The centerpiece of this
repository is `climtools.xarray`: a distributed, MPI-aware wrapper around
`xarray` for partitioning a `Dataset`/`DataArray` across MPI ranks and
running reductions, groupby/resample, rolling windows, and arithmetic
correctly on the distributed result. This README documents what is
actually implemented and verified today, after a full correctness/timing
audit (`tests/test_mpi_xarray.py`) -- not an aspirational design.

**A previous version of this README described planned behavior that did
not match the code and has been rewritten from scratch against the tested
implementation.**

## Package layout

| Module | What it is |
| --- | --- |
| `climtools.xarray` | The distributed xarray wrapper -- see below. |
| `climtools.mpi` | Thin `mpi4py` runtime: `mpi.comm` (raw communicator), collective helpers (`broadcast`/`scatter`/`gather`), a `@mpi` decorator for root-only/broadcast/all-ranks execution, and `mpi.decompose(ntasks)` for parent/child sub-communicators. |
| `climtools.netcdf` | Serial and MPI-parallel NetCDF-4 I/O (`climtools.xarray.constructors.to_netcdf` is the public entry point). |
| `climtools.xgeo` | Geospatial operations and NetCDF output: regridding, masking, transects, local solar time. Also registers the `.xgeo` accessor on `xarray.DataArray`/`Dataset`. |
| `climtools.stats` | Trends, correlations, difference-of-means testing. |
| `climtools.cdo` | Thin xarray-aware wrapper over the CDO command-line tool. |
| `climtools.plotting` / `climtools.cmaps` | Cartopy map plotting and a colormap registry (local IPCC tables, matplotlib, cmocean). |

Two access patterns are equivalent:

```python
from climtools import xgeo as xg
xg.plot.geo(da, method="contourf")

import climtools
da.xgeo.plot.geo(method="contourf")
```

## Installation

```bash
pip install climtools
```

This gives you everything except MPI-collective **parallel** NetCDF-4
output (`to_netcdf(..., parallel=True)` with more than one rank), which
needs `netCDF4`/`mpi4py` built against a parallel-enabled MPI/HDF5/NetCDF-C
stack -- `pip` cannot build that itself. For that, clone the repository and
run the setup script:

```bash
git clone https://github.com/Jared-Kodero/climtools.git
cd climtools && env/setup_env.sh
```

Everything else in `climtools.xarray` -- partitioning, reductions,
groupby/resample, rolling, arithmetic, and **serial** NetCDF-4 output --
only needs a plain `mpi4py` + `netCDF4`, no special build.

## `climtools.xarray`: the distributed xarray wrapper

### Core design

A `climtools.xarray.core.MPIXarray` wraps one rank-local
`xarray.Dataset`/`DataArray` (`.data`) plus its distribution metadata
(`.meta`: `dim`, `global_size`, `start`, `stop`, `chunk_info`). Data is
partitioned **contiguously along a single dimension** (`partition_dim` --
typically `time`); every other dimension is fully local to each rank. The
wrapper is deliberately thin: local, non-communicating operations run
directly through `xarray`; the wrapper steps in only for the operations
that need MPI communication.

As of the current refactor, **`climtools.mpi` and `climtools.xarray` are
decoupled**: an `MPIXarray` takes an explicit `MPIRuntime` (or a raw
`mpi4py.MPI.Intracomm`, auto-wrapped) rather than reaching for a
module-level `mpi.xarray` namespace. Construct one via the module-level
factories in `climtools.xarray.constructors`:

```python
from climtools import mpi
from climtools.xarray import constructors as xm

# Every rank opens the file and gets its own contiguous slice of `time`.
ds = xm.mpi_open_dataset("data.nc", mpi, partition_dim="time")

# Or distribute an object rank 0 already owns:
ds = xm.mpi_partition_data(full_dataset_or_none, mpi, "time")  # non-root ranks pass None

# Or build one from scratch, one rank-local fill function per rank:
ds = xm.mpi_create_dataarray(mpi, fill_fn, dims=("time",), shape={"time": 100}, dim="time")
```

`MPIXarray(some_plain_dataset, mpi)` also works directly if
`some_plain_dataset` is *replicated* (the same object on every rank) -- it
auto-partitions via `.repartition()`. If only rank 0 actually has the
data, use `mpi_partition_data`/`distribute` instead, which scatters it.

### Operation routing (verified against the implementation)

| Class | Communication | What's actually implemented |
| --- | --- | --- |
| **Local** | None | `isel`/`sel` (non-scalar, non-partition-dim), `astype`, `assign_attrs`, `assign_coords`, `chunk`, `clip`, `compute`, `drop_vars`, `expand_dims`, `fillna`, `load`, `persist`, `rename`/`rename_dims`/`rename_vars`, `reset_coords`, `round`, `set_coords`, `transpose`, `__getitem__` (variable selection by name), and any other rank-local `.apply(func, ...)` call. |
| **Scalar indexing** | One rank sends, broadcast | `isel`/`sel` with a scalar indexer on the partition dimension -- the dimension is dropped and the result is replicated on every rank. |
| **Global reductions** | `Allreduce` | `sum`, `prod`, `mean`, `min`, `max`, `any`, `all`, `first`, `last`, `var`, `std`. `mean`/`var`/`std` combine per-rank sums *and* valid counts so `skipna=True` is correct for uneven partitions or ranks holding all-NaN data. |
| **Stencil (halo exchange)** | Point-to-point with the one neighbor that matters | `diff` (label-aware: `"upper"` borrows from the left neighbor, `"lower"` from the right), `rolling`/`rolling_reduce` (`mean`/`sum`/`min`/`max`/`std`/`count`, `center=True` or `False`). |
| **Cross-rank scan** | `gather` + `scatter` (exclusive prefix) | `cumsum` along the partition dimension. |
| **Gather-based** | `gather` + broadcast | `median` along the partition dimension (gathers the full dimension onto rank 0 only, not every rank -- not memory-scalable for a very large partition dimension, but correct and exact). |
| **Grouped reductions** | Local partials + `Allgather` (label union) + `Allreduce` | `groupby(dim, labels).{sum,mean,count,min,max}()`, `resample(dim, freq).{sum,mean,count,min,max}()` -- both correctly handle group/bin boundaries that cross MPI rank boundaries. Cost scales with `n_groups`, appropriate for coarsening (daily/monthly resampling, categorical groupby), not high-cardinality grouping. |
| **Contraction** | `Allreduce` | `matmul`/`@` when the partition dimension is one of the contracted dimensions (splits the dot product additively over the distributed axis). |
| **Structural redistribution** | Full data movement | `repartition()` (replicated -> distributed) and `align(other)` (bring two distributed operands onto matching partitions). |

### What is *not* implemented (despite being a natural expectation)

Be aware of these gaps -- they are not silent; each fails loudly or simply
isn't a method -- but they're worth knowing about before you plan around
them:

- **No `shift`, `differentiate`, `sortby`, or `reindex`.** These do not
  exist on `MPIXarray`. `diff` and `rolling`/`rolling_reduce` are the only
  implemented stencil operations, and there is no structural-redistribution
  method besides `align`. Calling `.data.shift(...)` directly on the
  rank-local piece is **not** a safe substitute if `time` is the partition
  dimension -- it will silently compute a locally-wrong shift at every
  rank boundary.
- **`groupby`/`resample` never broadcast a variable that lacks the grouped
  dimension.** Plain `xarray.Dataset.groupby(...).mean()` broadcasts such a
  variable across every group; `MPIXarray.groupby`/`.resample` instead
  leave it completely untouched (see `Groupby._group_reduce_local`). This
  is a deliberate efficiency choice, not a bug, but it means a
  variable-by-variable comparison against plain `xarray` needs to account
  for it.
- **`var`/`std` of a variable lacking the reduction dimension** returns
  0 (`ddof=0`) or NaN (`ddof>0`), exactly mirroring plain `xarray`'s own
  quirk for "reducing" over an axis of implicit length one -- not
  something `climtools` tries to paper over.

## Testing

`tests/test_mpi_xarray.py` is a from-scratch, comprehensive
correctness-and-timing suite covering every `MPIXarray` method and
constructor in the table above, validated against an independently
computed serial (single-process) `xarray` reference at every rank count,
including counts that do **not** evenly divide the test dataset (5, 7
ranks against a 41-step time dimension) to exercise uneven-partition
boundaries.

```bash
mpirun -n 4 python tests/test_mpi_xarray.py
# or across a whole rank matrix (default 1 2 3 4 5 7):
tests/run_mpi_xarray_tests.sh
# single-node dev box without one physical core per rank:
MPI_EXTRA_ARGS="--oversubscribe" tests/run_mpi_xarray_tests.sh
# skip the (slower) timing section:
CLIMTOOLS_TEST_PERF=0 tests/run_mpi_xarray_tests.sh
```

It builds its own mock geophysical NetCDF dataset via
`tests/mock_dataset.py` (deterministic `pr`/`t2m`/`t`/`slmsk` fields at a
configurable resolution/time length) and prints a per-check pass/fail
table with MPI-vs-serial timing at the end. **The timing numbers are only
meaningful on real multi-core/multi-node hardware** -- on a single
oversubscribed development core (the environment this suite was developed
and verified in), the MPI column measures scheduling/communication
overhead, not parallel speedup, and the suite says so explicitly in its
own summary.

### Bugs found and fixed by this audit

Running this suite for the first time surfaced the following, all now
fixed (see the patch / commit history for exact diffs):

1. **`climtools.xarray` was completely unimportable.** `core.py` and
   `constructors.py` had a circular top-level import
   (`core.py: from .constructors import _MPIXarrayOps` vs.
   `constructors.py: from .core import MPIXarray, unwrap`); neither module
   could finish initializing first. Fixed by making the `_MPIXarrayOps`
   import inside `MPIXarray.__init__` lazy.
2. **`cumsum()` corrupted variables lacking the partition dimension.** A
   Dataset mixing a `time`-varying variable with a static one (e.g. a
   fixed vertical profile) got the static variable multiplied by a
   rank-index-dependent factor, because the cross-rank prefix-sum scan
   folded it into the same gather/scatter machinery as the genuinely
   reduced variables. Fixed by scanning only the variables that actually
   carry the partition dimension.
3. **`resample()` silently mis-binned any frequency with a multiple**
   (`"12h"`, `"6min"`, `"3D"`, ...) -- everything except a bare unit like
   `"D"`/`"MS"`. It used `pandas.DatetimeIndex.to_period(freq)`, which for
   multiples does not snap to a shared grid, giving nearly every timestamp
   its own singleton bin instead of the intended coarser bins. Fixed by
   computing bin edges via `Series.resample(freq, origin="epoch")`
   (fixed-origin, so every rank still agrees without communicating),
   verified bit-for-bit against real `pandas`/`xarray` resample bins for
   both fixed-tick and calendar-anchored frequencies.
4. **`resample()`'s output dimension was always named `_mpi_group`**
   instead of the resampled dimension's own name (e.g. `time`), unlike
   plain `xarray.Dataset.resample(time=freq)`, which keeps it as `time`.
   Fixed by renaming the result back after the underlying grouped
   reduction.
5. **Deadlock: `log_partitions=(rank == 0)` hung the entire job.** This is
   the natural way to ask for "only print the partition report from rank
   0", but `log_partition_report()` does an unconditional `comm.gather()`
   -- if only some ranks call it (because they passed different booleans),
   the ranks that do call it block forever. Fixed by resolving the
   decision with one `Allreduce(LOR)` before any rank commits to it, so
   passing a different boolean per rank is now safe.
6. **No `MPIXarray.__getitem__`.** Selecting a Dataset variable by name
   (`mds["pr"]`) raised `TypeError: not subscriptable`, forcing a drop to
   `.data["pr"]` and losing distribution metadata for a completely
   partition-safe operation. Added, restricted to string keys (positional
   indexing still correctly refuses -- it is not always partition-safe).
7. **`tests/mock_dataset.py`: the mock-data output directory was never
   created on a fresh machine.** `path.parent.mkdir(...)` was nested
   inside `if path.parent.exists():`, so on first run (no prior directory)
   it was silently skipped and every test run failed before doing
   anything. One-line fix.

Each fix above is exercised directly by a dedicated case in
`test_mpi_xarray.py` (not just incidentally covered), and the full suite
passes `71/71` at every tested rank count (1, 2, 3, 4, 5, 7).

## License

MIT -- see `LICENSE`.
