# climtools

Utilities for climate data analysis and plotting.

climtools is a small collection of utilities for exploratory and reproducible
climate-data analysis. It wraps common operations on xarray objects, integrates
with CDO where available, provides Cartopy map plotting, and includes colormap,
statistics, NetCDF-writing and theming helpers.

## Layout

| Namespace         | Purpose                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| `climtools.plot`  | Cartopy map plotting. Entry point `plot.geo`.                             |
| `climtools.xgeo`  | Geospatial operations, NetCDF output: regridding, masking, transects, local solar time. |
| `climtools.calc`  | Trends, correlations and difference-of-means testing.                     |
| `climtools.cmaps` | Colormaps spanning local IPCC tables, matplotlib and cmocean.             |
| `climtools.cdo`   | Thin xarray-aware wrapper over the CDO command-line tool.                 |
| `climtools.mpi`   | MPI runtime: `mpi.comm` for the raw communicator, `mpi.reduce`/`mpi.xarray` for collective reductions, backing parallel NetCDF-4 output. |

## Installation

This is a local developer package: clone the repository and import it from
the parent of the cloned directory, or install it into an environment.

```bash
git clone https://github.com/Jared-Kodero/climtools.git
```

```bash
# pip, into an already-activated environment
pip install ./climtools

# or conda: creates/uses the active environment, builds the parallel
# MPI/NetCDF stack, applies the rest of climtools's dependencies, and
# installs climtools itself, in one step
env/setup_env.sh [env_name]
```

`pyproject.toml` lists every third-party package climtools's own source
imports; nothing else is required. Regridding (`climtools.xgeo.remap`)
imports `xesmf` only on first call, so the rest of the package works without
it; install it with the `regrid` extra (`pip install "./climtools[regrid]"`)
or from `environment.yml`. `climtools.cdo` and
`climtools.xgeo.plot.animate` shell out to the `cdo`/`nco` and `ffmpeg`
command-line tools respectively; these are not Python packages and must be
on `PATH` separately (`environment.yml` installs them from conda-forge).

Parallel NetCDF-4 output
(`climtools.xgeo.to_netcdf(..., parallel=True)`) needs `netCDF4` and
`mpi4py` built against a parallel-enabled MPI/HDF5/NetCDF-C stack.
`env/setup_env.sh` builds that stack automatically, without ever using a
distro package manager (apt/yum builds are not available on HPC login or
compute nodes, which is what this stack is for): it first tries `module
load`-ing a matching MPI and `netcdf-mpi` module pair, and falls back to
compiling HDF5 and netCDF-C from source against the active MPI compiler.
Building `netCDF4`-python itself against certain netcdf-c 4.9.x builds (for
example Ubuntu's packaged netcdf-c 4.9.0) hits three known upstream
packaging issues — a redeclared bzip2/blosc filter shim, a
`nc_rc_get`/`nc_rc_set` version guard that assumes those symbols exist
starting at 4.9.0 when they were actually added later, and two
`nc_complex` functions (`pfnc_inq_varndims`, `pfnc_inq_vardimid`) declared
`inline` with no matching out-of-line definition anywhere in that vendored
library, which can leave an unresolved symbol at link time depending on
whether the compiler inlines every call site. `setup_env.sh` patches all
three defensively (a no-op against `netCDF4`/`nc_complex` versions that do
not have the corresponding issue) before building. Everything else in
climtools works with the ordinary serial `netCDF4` and `mpi4py` wheels from
PyPI or conda-forge.

## Usage

Two access patterns are supported and are equivalent. The `.xgeo` accessor is
registered on `xarray.DataArray` and `xarray.Dataset` when the package is
imported.

```python
from climtools import xgeo as xg
from climtools import cmaps

# Free-function form
plot = xg.plot.geo(t2m, method="contourf", cmap=cmaps.temp_div(), gridlines=True)

# Accessor form
plot = t2m.xgeo.plot.geo(method="contourf", cmap=cmaps.temp_div(), gridlines=True)
```

`plot.geo` returns a `GeoPlot`. Overlays are added through its chainable
`add` namespace, each returning the same `GeoPlot`:

```python
(
    t2m.xgeo.plot.geo(
        method="contourf", cmap=cmaps.temp_div(), levels=21, gridlines=True
    )
    .add.contour(z500, colors="k", clabel=True)  # line contours, labelled
    .add.quiver(u10, v10, subsample=4)  # vector overlay
    .add.significance(p_value)  # stippling where p < 0.05
)
```

`add` provides `default`, `contour`, `contourf`, `pcolormesh`, `imshow`,
`scatter`, `quiver`, `significance` and `colorbar`. Every overlay is applied to
each populated facet of a faceted map, with the overlay field sliced to match
each facet, so an overlaid field must carry the same facet dimension (`col` or
`row`) as the base field. Gridlines and coastlines, borders, states, ocean,
land, lakes and rivers are map-layout options set as keyword arguments on
`plot.geo`, not `add` methods.

Chaining geospatial operations reads in execution order:

```python
(
    ds.t2m.xgeo.remap(target_grid, method="conservative")
    .xgeo.mask(valid_value=1)
    .xgeo.add_local_solar_time()
    .mean("time")
    .xgeo.plot.geo(method="pcolormesh", cmap=cmaps.temp_seq())
)
```

Faceting a three-dimensional array requires `col` or `row`:

```python
xg.plot.geo(monthly, col="month", col_wrap=4, method="contourf")
```

An animation over a dimension is encoded as MP4 (requires `ffmpeg`):

```python
t2m.xgeo.plot.animate("time", method="contourf", vmin=-30, vmax=30, fps=6)
```

## Examples

`examples/` contains three files. The Python scripts run directly (`python
examples/<script>.py`) or under an MPI launcher; see
[Running under MPI](#running-under-mpi) for why the launcher line should
include `-m mpi4py`:

| File        | Demonstrates                                                                 |
| ----------- | ---------------------------------------------------------------------------- |
| `native.py` | Minimal serial-vs-parallel NetCDF write comparison, verified by read-back. Start here for the smallest possible `to_netcdf(..., parallel=True)` example. Builds a synthetic precipitation field by default; set `CLIMTOOLS_EXAMPLE_NETCDF` to a file with a `pr` variable to use real data instead. |
| `test.py`   | The correctness and scaling suite for `mpi.reduce`, `mpi.xarray`, `mpi.scatterv` and the NetCDF writers. Self-contained: rank 0 builds a deterministic mock NetCDF file, so no external input is required. `--time-steps` sets the size of that file. |
| `test1.py`  | Parallel NetCDF write benchmark across three source placements: rank-0-only (scattered), distributed from the start, and read back through a distributed open. Sizes are set with `--time-steps`, `--lat` and `--lon`. Skips the collective write cleanly when netCDF4 lacks parallel4 support. |
| `test.sh`   | Slurm batch script (`sbatch examples/test.sh`) running `test.py` on eight ranks, with the environment settings the suite requires. |

### Data placement covered by the suite

Distributed reductions behave differently depending on where the data starts
and how it is partitioned, so the suite covers each placement explicitly:

- **Replicated**: every rank holds the whole object, `mpi.xarray.redistribute`
  partitions it.
- **Rank-0-only**: one rank reads or generates the field and the others hold
  nothing. Both routes out of that state are checked, broadcasting followed by
  partitioning, and `mpi.scatterv` of leading-axis slabs.
- **Distributed open**: `mpi.xarray.open_dataset` partitions lazily on
  effective chunk bounds, so the rank-local partial is materialized inside the
  reduction.
- **Empty partitions**: whenever the partitioned dimension is shorter than the
  communicator, trailing ranks own no elements. These ranks build their
  partials through a different code path, which makes this the configuration
  most likely to expose an asymmetric collective sequence. Lengths below, at
  and above the rank count are all exercised, in both result placements.

### Shared configuration must be rank-invariant

Any value that sizes a collective (slice bounds, buffer shapes, partition
lengths) has to be derived identically on every rank. Deriving one inside a
rank-0-only branch leaves the other ranks holding their module defaults, and
the two failure modes that follow are both hard to read: reductions whose
buffers still happen to match return silently wrong answers, and reductions
whose buffers do not match deadlock, with rank 0 blocked in `Allreduce` while
every other rank waits in the next all-gather. `test.py` therefore reads its
shape constants back from the generated file on every rank and asserts they
agree, and `mpi.reduce` compares buffer signatures before posting.

Set `CLIMTOOLS_CHECK_COLLECTIVES=0` to skip the buffer comparison in
latency-bound production runs. It is on by default, because a mismatched
buffer is undefined behaviour and costs far more to diagnose than the check
costs to post.

### Collective symmetry

`test_collective_sequence_symmetry` records the sequence of collectives each
rank posts during every reduction and compares those sequences across ranks.
Ranks that post different collectives can appear to succeed under one MPI
implementation and deadlock under another, because whether a mismatched buffer
collective completes depends on the algorithm the library selects. Comparing
the sequences directly makes that class of defect fail deterministically in
the suite rather than in a production run.

The same guarantee is enforced at runtime. Every reduction buffer collective
carries a signature (operation, placement, root, dtype, shape) inside the
all-gather that already synchronizes rank-local errors, so a divergence raises
`MPIError` naming the disagreeing ranks instead of blocking. The check costs no
additional communication.

### Running the suite on a cluster

Two environment settings matter and are set in `test.sh`:

```bash
# Required. HDF5 advisory locks block rather than fail on a parallel
# filesystem, which can stall a subset of ranks inside open_dataset.
export HDF5_USE_FILE_LOCKING=FALSE

# Diagnostic only, off by default. Disables HCOLL/UCC collective offload.
export CLIMTOOLS_NO_COLL_OFFLOAD=1
```

Collective offload is left at the site default. A correct program must not
depend on which collective algorithm the MPI library selects, and the suite
verifies that every rank posts an identical collective sequence, so offload is
not a correctness risk; disabling it costs performance on multi-node
reductions. Use `CLIMTOOLS_NO_COLL_OFFLOAD=1` only when bisecting a suspected
hang. If disabling offload changes the outcome, that is a collective-symmetry
bug to investigate, not a setting to keep.

Launch with `python -m mpi4py`, never a bare `python`, so an exception on a
subset of ranks aborts the job instead of leaving it blocked in
`MPI_Finalize`. Reductions are guarded by `mpi.watchdog`, which dumps every
rank's stack after a period without progress and then aborts; the guard covers
the error-synchronization step as well as the work itself, so a rank that runs
ahead of the others still reports where it is.

## Parallel output

MPI ranks can each hold a partial result and combine them into one field, and
a complete Dataset can be written to one shared NetCDF-4 file with every rank
contributing its slab through a single MPI-collective write.

```python
import climtools

mpi = climtools.mpi
from climtools.core.xgeo import empty_dataset


@mpi(all_ranks=True)
def build_local_composite():
    """Each rank computes a partial composite over its own share of events."""
    local_events = ds.xgeo.mask(...).sel(event=this_ranks_events)
    return local_events.sum("event")


def main():
    local = build_local_composite()

    # Element-wise collective reduction. Works on scalars, NumPy arrays, and
    # xarray DataArrays/Datasets alike; dims, coords, and attrs are kept.
    composite = mpi.reduce.sum(local)  # same result on every rank
    # composite = mpi.reduce.sum(local, mode="root", root=0)  # result on rank 0 only, None elsewhere

    # Rank 0 must hold the complete Dataset or DataArray; every other rank
    # binds empty_dataset() so `.xgeo` still resolves on a real xr.Dataset
    # (an accessor cannot bind to None). The writer creates the file schema
    # from rank 0's object, then scatters it back out and has every rank
    # write its own slab collectively, so serial NetCDF I/O never becomes
    # the bottleneck even though rank 0 does hold the full array in memory
    # right before the write.
    if mpi.comm.rank == 0:
        full = assemble_full_dataset(composite)
    else:
        full = empty_dataset()

    full.xgeo.to_netcdf("composite.nc", partition_dim="event", parallel=True)


main()
```

```bash
mpirun -n 8 python -m mpi4py script.py
srun --ntasks=8 --mpi=pmix python -m mpi4py script.py
```

### Running under MPI

Launch with `python -m mpi4py`, not a bare `python`. mpi4py calls
`MPI_Init_thread` when `mpi4py.MPI` is imported and registers `MPI_Finalize`
to run at interpreter exit, so an unhandled exception on a subset of ranks
does not terminate the job: the failing ranks block in `MPI_Finalize` waiting
for the others, and the others block in the collective the failing ranks never
reached. `python -m mpi4py` installs a finalizer hook that calls `MPI_Abort`
instead, converting that deadlock into a non-zero exit
([mpi4py: Exceptions and deadlocks](https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks)).

Because the launch command is outside climtools's control, importing
`climtools.mpi` under an MPI launcher installs the equivalent `sys.excepthook`
as a fallback. It is a no-op when `python -m mpi4py` already installed its own
hook, and on single-rank runs with no launcher.

Neither hook helps when a rank is blocked rather than failing, since a blocked
rank raises nothing. `mpi.watchdog(phase)` covers that case:

```python
with mpi.watchdog("compositing"):
    result = mpi.xarray.mean(local_events, dim="event")
```

It arms a daemon thread on every rank. mpi4py releases the GIL for the
duration of a blocking MPI call, so the thread keeps running while the main
thread sits in `Allreduce` or in a blocking read, and prints that rank's own
traceback naming the line it is stuck on before calling `MPI_Abort`. Every
rank dumps independently, so the log distinguishes the ranks that arrived from
the ones that did not. The default is 600 s of no progress; pass `timeout=` to
change it, or `timeout=0` to leave the block unguarded.

On Lustre or GPFS, `HDF5_USE_FILE_LOCKING=FALSE` must be set in the
environment. HDF5 takes POSIX advisory locks by default, and many ranks
opening the same NetCDF file concurrently can block there indefinitely, which
presents as a hang with no traceback on any rank.

The same script runs unchanged on a single rank without a launcher, by
passing `allow_serial=True` to `to_netcdf`; `mpi.reduce` degrades to a no-op
reduction over one rank. `mpi4py` is a hard dependency of `climtools.mpi` and
of the NetCDF writer, so it must be installed (see
[Installation](#installation)) even for single-rank runs.

`mpi.reduce` also exposes `prod`, `min`, `max`, `any`, and `all`, all
following the same signature as `sum`. Every reduction accepts `mode="all"`
(the default: every rank gets the result) or `mode="root"` with a `root=`
rank (only that rank gets the result; `None` elsewhere). `mpi.comm` is the
native `mpi4py.MPI.Intracomm`, so anything not covered above — point-to-point
`Send`/`Recv`, `Bcast`, `Scatterv`/`Gatherv`, non-blocking collectives — is
reached directly as `mpi.comm.<method>` with full mpi4py signatures and IDE
completion.

`mpi.xarray` reduces an xarray `DataArray`/`Dataset` along a named
dimension that is itself split across ranks — the counterpart to
`mpi.reduce` for when the split is expressed as a dimension rather than as
independent whole-array partials:

```python
# Each rank holds a different slice of the "event" dimension.
local_mean = mpi.xarray.mean(local_events, dim="event")  # same result on every rank
```

`local_mean` is identical to calling plain `xarray`'s
`assembled.mean(dim="event")` on the fully assembled array — `mpi.xarray`
combines each rank's local xarray reduction with one collective rather than
requiring the full array in one place first. `mpi.xarray` exposes `sum`, `prod`,
`min`, `max`, `mean`, `any`, and `all`, with `skipna`/`min_count` applied
consistently across the whole distributed dimension (not per rank): a value
is only dropped by `min_count` once the count is summed across every rank
that holds a share of `dim`, and `skipna=False` propagates a NaN present on
any rank to the combined result, not just NaNs local to the current rank.

Every one of `mpi.reduce`/`mpi.xarray`/`mpi.scatterv`/`to_netcdf(...,
parallel=True)` and the raw `mpi.comm.<method>` calls above is
MPI-collective: every rank in `mpi.comm` must reach the same call, in the
same order, or the call blocks forever waiting for ranks that never arrive.
This is what makes the `to_netcdf(..., parallel=True)` calling convention
above matter — passing real data only on rank 0 and `empty_dataset()`
elsewhere keeps every rank calling the writer at the same point, even though
only one of them supplies data. Work that inherently runs at different
paces per rank (for example, independent cases assigned one-per-rank) should
use ordinary serial `to_netcdf()` for anything each rank writes on its own,
and only bring ranks back through a shared collective (`mpi.comm.barrier()`,
`mpi.reduce`, `mpi.xarray`, or a collective write) at points where every
rank is guaranteed to have arrived — see `examples/time_composites.py`
(above) for a script structured this way.

Parallel output requires `netCDF4` and `mpi4py` built against a
parallel-enabled MPI/HDF5/NetCDF-C stack; see
[Installation](#installation) and `env/setup_env.sh`.

mpi4py itself initializes MPI (`MPI_Init`/`MPI_Init_thread`) as a side
effect of `import mpi4py.MPI`, not on first collective call, and registers
`MPI_Finalize` to run automatically at process exit, so climtools code never
calls either explicitly. mpi4py also sets the `ERRORS_RETURN` error handler
on `COMM_WORLD` by default (rather than MPI's default
`ERRORS_ARE_FATAL`), so a failing raw `mpi.comm.<method>` call raises a
catchable `mpi4py.MPI.Exception` (a `RuntimeError` subclass) instead of
aborting the process outright; `climtools.mpi.MPIError` is climtools's own
exception type, raised by `mpi.reduce`/`mpi.xarray`/the `@mpi` decorator/the
NetCDF writer for climtools-level validation and synchronized-failure
reporting; see the [mpi4py Overview](https://mpi4py.readthedocs.io/en/stable/overview.html)
for further detail on both.

## Notes

- The map entry point is `plot.geo`, reached either as the free function
  `climtools.plot.geo(da, ...)` or as the accessor `da.xgeo.plot.geo(...)`.
- All input coordinates are interpreted in `PlateCarree`. The display
  projection is chosen with `projection=`, or inferred from the data extent.