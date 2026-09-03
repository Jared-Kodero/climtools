# climtools

Utilities for climate data analysis and plotting: geospatial operations, Cartopy map plotting, statistical analysis, colormaps, and CDO utilities. MPIXarray mpi communication is based on GFDL Shiedl FMS mpp_d...

- Repository: https://github.com/Jared-Kodero/climtools

- Requires Python ≥ 3.12

## Contents

- [Package layout](#package-layout)

- [Installation](#installation)

- [Quick start: plotting](#quick-start-plotting)

- [MPI-Xarray: distributed processing](#mpi-xarray-distributed-processing)

- [Testing](#testing)

- [License](#license)

## Package layout

| Namespace | Purpose | Source |

| --- | --- | --- |

| `climtools.plotting` | Cartopy map plotting. Entry point `plotting.geo` (also reachable as `xgeo.plot.geo`, used below). | [`viz/plotting.py`](viz/plotting.py) |

| `climtools.xgeo` | Geospatial operations and NetCDF output: regridding, masking, transects, local solar time, `to_netcdf`. | [`core/xgeo.py`](core/xgeo.py), [`xarray/io.py`](xarray/io.py) |

| `climtools.stats` | Trends, correlations, difference-of-means testing. | [`core/stats.py`](core/stats.py) |

| `climtools.cmaps` | Colormaps: local IPCC tables, matplotlib, cmocean. | [`viz/cmaps.py`](viz/cmaps.py) |

| `climtools.cdo` | Thin wrapper over the CDO command-line tool. | [`cdo/pycdo.py`](cdo/pycdo.py) |

| `climtools.mpi` | MPI runtime/communicator handle, shared by every distributed operation below. | [`mpi/context.py`](mpi/context.py) |

| `climtools.xgeo.mpi_open_dataset`, `.mpi_create_dataarray`, `.mpi_create_dataset`, `.mpi_partition_data` | MPI-parallel Xarray: partition NetCDF data or in-memory arrays across ranks, then operate on them with (mostly) ordinary Xarray syntax. See [MPI-Xarray](#mpi-xarray-distributed-processing) below. | [`xarray/io.py`](xarray/io.py), [`xarray/core.py`](xarray/core.py) |

## Installation

Install from PyPI:

```bash

pip install climtools

```

or, from a clone, editable for development:

```bash

git clone https://github.com/Jared-Kodero/climtools.git

cd climtools

pip install -e .

```

Regridding (`xgeo.remap`) imports `xesmf` only on first call, so the rest of

the package works without it:

```bash

pip install "climtools[regrid]"

```

`climtools.cdo` and `xgeo.plot.animate` shell out to the `cdo`/`nco` and

`ffmpeg` command-line tools respectively. These are not Python packages and

must be on `PATH` separately -- see

[`env/environment.yml`](env/environment.yml), which installs them from

conda-forge alongside everything else in this step.

## Quick start: plotting

from climtools import xgeo as xg
from climtools import cmaps

plot = xg.plot.geo(t2m, method="contourf", cmap=cmaps.temp_div(), gridlines=True)

plot.geo returns a GeoPlot. Overlays are added through its chainable
add namespace, each call returning the same GeoPlot so calls chain:

(
    xg.plot.geo(
        t2m, method="contourf", cmap=cmaps.temp_div(), levels=21, gridlines=True
    )
    .add.contour(z500, colors="k", clabel=True)
    .add.quiver(u10, v10, subsample=4)
    .add.significance(p_value)
)

add provides default, contour, contourf, pcolormesh, imshow,
scatter, quiver, significance, and colorbar. Every overlay applies to
each populated facet of a faceted map, sliced to match that facet, so an
overlaid field must carry the same facet dimension (col/row) as the base
field. Gridlines, coastlines, borders, states, ocean, land, lakes, and rivers
are map-layout keyword arguments on plot.geo itself, not add methods.

Faceting a three-dimensional array needs col or row:

xg.plot.geo(monthly, col="month", col_wrap=4, method="contourf")

All input coordinates are interpreted in PlateCarree. The display
projection is chosen with projection=, or inferred from the data extent.
See viz/plotting.py for the full GeoPlot API and
viz/cmaps.py for the colormap catalog.


## MPI-Xarray: distributed processing

`climtools.xgeo` also exposes an MPI-parallel layer for working with
Xarray `Dataset`/`DataArray` objects too large for one rank's memory,
built on `mpi4py` and (optionally) parallel-I/O-enabled `netCDF4`/HDF5.
Every rank runs the same script; each ends up owning a distinct,
non-overlapping slice of the data and operates on it with mostly
ordinary Xarray syntax.

```python
from climtools import mpi, xgeo

# every rank opens the same file but only loads its own slice
dist = xgeo.mpi_open_dataset("data.nc", mpi, partition_dim="time")

# ordinary NumPy ufuncs and many Xarray methods dispatch automatically,
# running rank-locally with no communication when they can
logged = np.log(dist["pr"])
rolled = dist.rolling_reduce("time", window=5, reduce="mean")

# a reduction that removes the partition dimension gathers/combines
# across ranks automatically
global_mean = dist.mean(dim="time")
```

Call `._prepare().load()` on any of the above to materialize this
rank's own local piece as a plain, in-memory `xarray` object -- e.g.
for saving a per-rank diagnostic plot or feeding a rank-local
computation.

### Constructing a distributed object

Four entry points, all in `xarray/io.py` and re-exported through
`climtools.xgeo`:

| Function | Use when |
| --- | --- |
| `mpi_open_dataset(path, runtime, partition_dim=...)` | Reading a NetCDF file: every rank opens the same file and loads only its own slice. |
| `mpi_partition_data(value, runtime, dim=...)` | An object already fully materialized somewhere (root-owned, or already replicated on every rank) needs to become distributed. |
| `mpi_create_dataarray(runtime, fill, dims, shape, dim=...)` | Synthesizing a DataArray from a function of index bounds, e.g. a test fixture or an analytic field, without ever materializing the full global array anywhere. |
| `mpi_create_dataset(runtime, data_vars, sizes, dim=...)` | Same, for a Dataset with several variables, each independently choosing whether it varies along the partition dimension(s), a subset of them, or none. |

### Partitioning semantics

`dim` may be:

- A single dimension name (or `"auto"`, which picks the longest
  dimension) -- a 1D partition. Each rank owns a contiguous slice
  `[start, stop)` along that one dimension; every other dimension is
  full-length on every rank.
- A sequence of two or more dimension names -- a Cartesian-topology
  partition, laying ranks out on an MPI process grid (via
  `MPI.Cartcomm`) and partitioning every named dimension
  simultaneously. The grid shape is chosen to roughly match the named
  dimensions' aspect ratio (more divisions along the longer
  dimension), the same heuristic GFDL FMS's `mpp_define_layout2D`
  uses. A 2×3 grid over `(lat, lon)` at 6 ranks, for example, gives
  each rank a distinct, non-overlapping `(lat_slice, lon_slice)` tile;
  the tiles' union covers the full `(lat, lon)` grid exactly once.

Bounds are computed by `get_balanced_bounds`, which divides a
dimension of length `L` across `N` ranks as evenly as possible (a
remainder is spread one extra element per rank, starting from rank 0)
and produces empty (`start == stop`) ranks, never a negative or
out-of-range slice, whenever `L < N`. This holds at any dimension
length, including 0 and 1, and at any rank count.

Reduced dtype and metadata guarantees:

- **dtype**: rank-local computation always uses the same NumPy/Xarray
  promotion rules as plain Xarray; the MPI layer does not introduce
  its own dtype coercion, except where a cross-rank reduction's
  intermediate sum needs a specific dtype to avoid silently truncating
  a fractional value to an integer (`var`/`std`'s sum-of-squared-
  deviations, always floating-point even for an integer-typed input
  variable).
- **metadata (`.meta` / `mpi_meta`)**: a distributed object carries its
  own partition metadata -- which dimension(s) are partitioned, this
  rank's own `start`/`stop` and the dimension's `global_size` for
  each, and (for a Cartesian partition) the process-grid shape and
  this rank's own grid coordinates. An operation that doesn't change
  which dimensions are partitioned carries this metadata through
  unchanged; one that does (a reduction, a length-changing scan, an
  interpolation) recomputes it. An object with `.meta is None` is
  either not distributed at all, or -- after a reduction removes every
  partition dimension -- fully replicated: every rank holds an
  identical copy.

### Halo-aware operations

Some operations need values from a neighboring rank to compute correctly
at a rank boundary -- `rolling_reduce`, `coarsen_reduce`, `diff`,
`shift`, `differentiate`, `roll` (a periodic/wraparound halo exchange,
so a rank at the global edge borrows from the rank at the opposite
edge), and bounded-`limit` `ffill`/`bfill`. These use
`halo_exchange`, a point-to-point exchange with each rank's immediate
neighbor(s) along the operated-on dimension, requesting exactly as many
elements as the operation's own parameters imply are needed (a rolling
window of width `w` needs `w // 2` elements from each side; an n-th
order `diff` needs `n`; `differentiate` needs 1 on each side for its
edge-order-1 stencil). No wider halo is ever fetched than the operation
itself requires, and a halo width of 0 (nothing needed) takes a
genuine no-communication fast path. Every halo-touching operation
above is validated under both single-dimension and multi-dimensional
(Cartesian) partitions.

The expensive part of this -- the neighbor topology itself (which rank
is "to the left" along a given dimension, which is "to the right",
under an arbitrary N-dimensional process grid) -- is computed once per
`(communicator, partitioned-dimensions)` pair and cached on the MPI
communicator itself (an `MPI.Cartcomm` attribute keyval), the same
principle GFDL FMS's `mpp_domains` uses for its precomputed
`overlapSpec`: expensive topology discovery happens once, reused by
every subsequent operation on that same communicator. What is *not*
cached is the halo data itself -- every call re-exchanges the current
values fresh, appropriate for this package's immutable-array model
(each operation produces a new object; there is no long-lived,
in-place-updated field the way FMS's persistent model grid has, so
there is nothing meaningful to invalidate a stale cached halo against).
A halo-padded intermediate array is always trimmed back down and has
its clean metadata reattached (or freshly recomputed, if the operation
changed the dimension's length) before returning to the caller; it
never escapes as a still-padded, unlabeled object.

### Reductions, scans, and redistribution

Cross-rank reductions (`mean`, `sum`, `min`, `max`, `var`, `std`,
`prod`, `median`, `any`, `all`, `first`, `last`,
`groupby(dim, labels).mean()`/
`.sum()`/`.min()`/`.max()`/`.count()` -- verified under both single-
and multi-dimensional partitions -- and
`resample(dim, freq).mean()`/`.sum()`/`.min()`/`.max()`/`.count()` --
verified under a single partition dimension only; its behavior under
a multi-dimensional partition combining a datetime axis with another
partitioned dimension is untested) that
remove a partition dimension either replicate the (now small) result
to every rank -- if that was the only partition dimension -- or, under
a multi-dimensional partition, reattach metadata for whichever
dimension(s) survive. In the latter case, **no two ranks ever end up
claiming ownership of the same range**: exactly one rank per distinct
surviving-dimension range keeps the real, computed data; every other
rank that shared that same range before the reduction (differing only
along the now-removed dimension) is left with a genuinely empty
(`start == stop`) slice, not a duplicate copy -- the same convention
already used for a rank a dimension is simply too short to reach.

`cumsum` and unbounded (`limit=None`) `ffill`/`bfill` are cross-rank
scans: a gather/scatter exclusive-prefix pass (`cumsum`) or a
last-value-carried-forward/backward pass (`ffill`/`bfill`), each
scoped to the sub-communicator that varies along the scanned
dimension alone, so independent groups along any other partitioned
dimension scan independently rather than being folded into one flat
rank ordering.

`sortby` and `reindex` redistribute data across ranks when the sort
key or new index varies along an active partition dimension, via a
personalized point-to-point shuffle, then rebalance with fresh
`get_balanced_bounds` bounds -- correct under a multi-dimensional
partition as long as only one active partition dimension is touched
at once; touching more than one simultaneously raises
`NotImplementedError` rather than guessing.

`interp` has no fixed-width halo dependency (a target point's
bracketing source points could, in principle, live on any rank, not
just an immediate neighbor), so it takes the FMS-sanctioned fallback
of a real `Allgather` reconstruction along the interpolated dimension,
scoped to the relevant sub-communicator under a multi-dimensional
partition -- every rank ends up with the full source along that one
dimension and interpolates locally onto its own slice of the *new*
target coordinate (which the caller supplies pre-split per rank, not
the global target grid).

`to_netcdf(..., parallel=True)` writes a distributed object as a
single NetCDF file via a real collective HDF5 write, one hyperslab per
rank -- correct under any number of partition dimensions. Explicit
per-variable chunk shapes (`chunks={"varname": shape, ...}`) are
always honored; without them, chunk shapes are inferred automatically
for any partition dimensionality, each partition axis independently
aligned to rank boundaries (via that axis's own division count on the
Cartesian process grid) and capped to the HDF5 4 GiB per-chunk limit.

A few operations still support only a single active partition dimension
at a time and raise `NotImplementedError` explicitly, rather than
silently, otherwise: `align()` (reconciling two operands partitioned
on different dimensions), and `reindex`/`sortby` when more than one
active partition dimension is touched simultaneously. `matmul()`'s own
restriction is narrower still -- it only fails when *more than one*
contracted (shared-between-operands) dimension is implicated, not
merely because the partition itself is multi-dimensional. Every other
public operation documented above, including `coarsen_reduce` (whose
only remaining restriction, `side="right"`, is unrelated to partition
dimensionality), is validated under both single- and
multi-dimensional partitions.

`where(cond, other)` is rank-local (no communication) except
`drop=True`, which is rejected outright since it could remove a
different number of positions on different ranks. `align(other, dim)`
partitions two operands identically, redistributing whichever one
needs it; `repartition(dim)` distributes an object every rank already
holds fully, with no data movement beyond each rank keeping its own
slice. `evaluate(expr, **vars)` and `apply(func, *args)` both run a
rank-local computation (a string expression or an arbitrary callable,
respectively) while propagating the caller's own distribution
metadata onto the result -- `apply`'s callable must preserve each
rank's owned length and coordinate labels along the partition
dimension, or it raises rather than silently desyncing ranks.

### Performance and memory considerations

- Every MPI-Xarray call carries fixed per-call overhead (a cross-rank
  call-signature agreement check, metadata bookkeeping) on top of the
  underlying Xarray/NumPy computation, independent of array size or
  rank count. For small arrays, this overhead can exceed the actual
  computation cost -- see [`test/benchmark.py`](test/benchmark.py) and
  run it yourself to see whether that trade-off is worthwhile at your
  own data size and rank count.
- No operation reconstructs the full global array unless the operation
  is inherently global (an `Allgather`-based `interp`, or `median`'s
  gather-to-root) -- rank-local operations, halo exchanges, and
  Allreduce-based reductions all keep memory use proportional to a
  rank's own local slice (plus a small, bounded halo), not the global
  array.

## Testing

`test/mock_dataset.py` generates a small synthetic NetCDF fixture (a
handful of geophysical-looking variables over `time`/`lat`/`lon`/`plev`)
used by every test and benchmark below.

Run the MPI-Xarray correctness suite locally (any rank count; ranks
beyond your CPU count need `--oversubscribe`):

```bash
mpirun --oversubscribe -n 4 python test/mpi_test.py
```

`mpi_test.py` is the single entry point: it builds the shared fixtures
once, then imports and runs each `test/mpi_test_*.py` module in turn
(`mpi_test_construction.py`, `mpi_test_reductions.py`,
`mpi_test_halo_ops.py`, `mpi_test_scans.py`, `mpi_test_groupby.py`,
`mpi_test_misc.py` --
each owns one coherent area rather than everything living in one
file). It prints a `[PASS]`/`[FAIL]`/`[SKIP]` line per check (`SKIP`
marks an operation's own declared `NotImplementedError` under an
unsupported partition shape, not a test failure), a final pass count,
and exits nonzero on any `[FAIL]`. On an HPC cluster, `test/test.sh`
submits the same idea through SLURM/`srun` instead.

Run the benchmark suite (compares MPI-Xarray against native Xarray on a
synthetic 1D field; see [`test/benchmark.py`](test/benchmark.py) for
the full option list and its own notes on interpreting results
honestly on constrained hardware):

```bash
mpirun --oversubscribe -n 4 python test/benchmark.py --size 2000000 --reps 5
```

which prints a markdown summary table and writes the raw per-operation
timings to `benchmark_results_n<ranks>.json`.

## License

MIT -- see [`LICENSE`](LICENSE).