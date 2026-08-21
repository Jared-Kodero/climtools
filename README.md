# climtools

Utilities for climate data analysis and plotting: xarray-based geospatial
operations, Cartopy map plotting, an MPI runtime for distributed reductions,
and MPI-collective parallel NetCDF-4 output.

- Repository: https://github.com/Jared-Kodero/climtools
- Requires Python ≥ 3.12

## Contents

- [Package layout](#package-layout)
- [Installation](#installation)
  - [1. Plain install (plotting, `calc`, `cdo`, single-rank `mpi`)](#1-plain-install)
  - [2. MPI-collective parallel NetCDF-4 output](#2-mpi-collective-parallel-netcdf-4-output)
- [Quick start: plotting](#quick-start-plotting)
- [The MPI runtime](#the-mpi-runtime)
  - [`mpi.reduce` vs `mpi.xarray`: which one do I want?](#mpireduce-vs-mpixarray-which-one-do-i-want)
  - [`mpi.reduce`: element-wise reductions](#mpireduce-element-wise-reductions)
  - [`mpi.xarray`: named-dimension distributed reductions](#mpixarray-named-dimension-distributed-reductions)
  - [Native `.mean()`/`.sum()`/etc. on a distributed object are node-local](#native-meansumetc-on-a-distributed-object-are-node-local)
  - [Arithmetic on distributed objects](#arithmetic-on-distributed-objects)
  - [Parallel NetCDF-4 output](#parallel-netcdf-4-output)
  - [Running under MPI](#running-under-mpi)
- [Testing](#testing)
- [License](#license)

## Package layout

| Namespace | Purpose | Source |
| --- | --- | --- |
| `climtools.plot` | Cartopy map plotting. Entry point `plot.geo`. | [`viz/plotting.py`](viz/plotting.py) |
| `climtools.xgeo` | Geospatial operations and NetCDF output: regridding, masking, transects, local solar time, `to_netcdf`. | [`core/xgeo.py`](core/xgeo.py), [`lib_netcdf/`](lib_netcdf/) |
| `climtools.calc` | Trends, correlations, difference-of-means testing. | [`core/calc_stats.py`](core/calc_stats.py) |
| `climtools.cmaps` | Colormaps: local IPCC tables, matplotlib, cmocean. | [`viz/cmaps.py`](viz/cmaps.py) |
| `climtools.cdo` | Thin xarray-aware wrapper over the CDO command-line tool. | [`cdo/pycdo.py`](cdo/pycdo.py) |
| `climtools.mpi` | MPI runtime: `mpi.comm`, `mpi.reduce`, `mpi.xarray`, `mpi.scatterv`, the `@mpi` decorator, `mpi.watchdog`. | [`core/lib_mpi.py`](core/lib_mpi.py), [`core/xr_mpi.py`](core/xr_mpi.py) |

The `.xgeo` accessor is registered on `xarray.DataArray`/`xarray.Dataset` as
soon as `climtools` is imported (`import climtools` is enough; you do not
need `from climtools import xgeo` for the accessor form to work) — see
[`accessors/xarray_accessors.py`](accessors/xarray_accessors.py).

## Installation

### 1. Plain install

This covers `climtools.plot`, `climtools.calc`, `climtools.cdo`,
`climtools.cmaps`, and single-rank use of `climtools.mpi` (`mpi.reduce`
degrades to a no-op over one rank; `to_netcdf(..., allow_serial=True)`
writes without MPI collectives). It does **not** cover MPI-collective
parallel NetCDF-4 output (`to_netcdf(..., parallel=True)` under more than
one rank) — that needs step 2 below.

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
must be on `PATH` separately — see
[`env/environment.yml`](env/environment.yml), which installs them from
conda-forge alongside everything else in this step.

### 2. MPI-collective parallel NetCDF-4 output

`to_netcdf(..., parallel=True)` needs `netCDF4` and `mpi4py` built against a
**parallel-enabled** MPI/HDF5/NetCDF-C stack. The plain wheels from PyPI (and
most conda-forge builds) are serial-only: importing them works fine, but
`to_netcdf(..., parallel=True)` raises `NetCDFWriteError` the first time it
runs under more than one rank. climtools checks for this and prints a
one-time hint pointing back here as soon as `climtools.mpi` is used under a
real multi-rank launch — see `_warn_if_parallel_netcdf_missing` in
[`core/lib_mpi.py`](core/lib_mpi.py).

Run the setup script from a clone of this repository:

```bash
git clone https://github.com/Jared-Kodero/climtools.git
cd climtools
env/setup_env.sh [env_name]
```

Step by step, this script:

1. Uses whatever conda environment or virtualenv is already active. With
   none active, it creates and activates a conda environment (default name
   `climtools`, or `env_name` if you passed one) from
   [`env/environment.boot.yml`](env/environment.boot.yml), installing
   Miniconda first if `conda` itself is missing.
2. Locates a parallel-enabled MPI/HDF5/NetCDF-C stack, in order:
   - **HPC environment modules** — `module load`s a matching MPI +
     `netcdf-mpi` module pair, if the `module` command exists (this is the
     fast path on most clusters, since the site has usually already built
     one).
   - **Source build** — otherwise compiles HDF5 and NetCDF-C from source
     against the active MPI compiler.

   Neither path uses a distro package manager: `apt`/`yum` are unavailable on
   HPC login and compute nodes, which is what this stack is for.
3. Builds `mpi4py` and `netCDF4` from source against that stack (patching
   three known netCDF4-python/netcdf-c packaging mismatches along the way,
   documented in [`env/setup_env.sh`](env/setup_env.sh)), and confirms
   `netCDF4.__has_parallel4_support__` is `True` before continuing.
4. Applies the rest of climtools's dependencies from
   [`env/environment.yml`](env/environment.yml) (conda environments only),
   then re-confirms the parallel build survived the solve — a full re-solve
   can occasionally pull in a replacement `mpi4py`/`netCDF4` as someone
   else's transitive dependency, so this rebuilds once if that happened.
5. Installs `climtools` itself, editable, into whichever environment is
   active.

Run it again any time to re-check or repair the parallel stack — every step
is idempotent.

**Required on Lustre/GPFS**: HDF5 takes POSIX advisory locks by default, and
many ranks opening the same file concurrently — or one rank reopening a file
another rank just closed — can block on those locks indefinitely on a
parallel filesystem, which presents as a hang with no traceback on any rank.
`lib_netcdf/parallel.py` sets `HDF5_USE_FILE_LOCKING=FALSE` as a
process-local default the moment it is imported, so this is handled
automatically for anything going through `xgeo.to_netcdf`. If your own code
opens NetCDF files directly (not through `xgeo.to_netcdf`), set it yourself,
before those opens happen:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

## Quick start: plotting

Two access patterns are supported and are equivalent:

```python
from climtools import xgeo as xg
from climtools import cmaps

# Free-function form
plot = xg.plot.geo(t2m, method="contourf", cmap=cmaps.temp_div(), gridlines=True)

# Accessor form (climtools must have been imported at least once)
plot = t2m.xgeo.plot.geo(method="contourf", cmap=cmaps.temp_div(), gridlines=True)
```

`plot.geo` returns a `GeoPlot`. Overlays are added through its chainable
`add` namespace, each call returning the same `GeoPlot` so calls chain:

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
`scatter`, `quiver`, `significance`, and `colorbar`. Every overlay applies to
each populated facet of a faceted map, sliced to match that facet, so an
overlaid field must carry the same facet dimension (`col`/`row`) as the base
field. Gridlines, coastlines, borders, states, ocean, land, lakes, and
rivers are map-layout keyword arguments on `plot.geo` itself, not `add`
methods.

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

Faceting a three-dimensional array needs `col` or `row`:

```python
xg.plot.geo(monthly, col="month", col_wrap=4, method="contourf")
```

Encode an animation over a dimension as MP4 (requires `ffmpeg` on `PATH`):

```python
t2m.xgeo.plot.animate("time", method="contourf", vmin=-30, vmax=30, fps=6)
```

All input coordinates are interpreted in `PlateCarree`. The display
projection is chosen with `projection=`, or inferred from the data extent.
See [`viz/plotting.py`](viz/plotting.py) for the full `GeoPlot` API and
[`viz/cmaps.py`](viz/cmaps.py) for the colormap catalog.

## The MPI runtime

`climtools.mpi` is a single object (an `MPIRuntime`, defined in
[`core/lib_mpi.py`](core/lib_mpi.py)) exposing:

| Attribute | What it is | Source |
| --- | --- | --- |
| `mpi.comm` | The native `mpi4py.MPI.Intracomm` (`MPI.COMM_WORLD` by default) | [`core/lib_mpi.py`](core/lib_mpi.py) |
| `mpi.reduce` | Element-wise collective reductions over whole objects | [`core/lib_mpi.py`](core/lib_mpi.py) (`ReduceAccessor`) |
| `mpi.xarray` | Distributed reductions/indexing/arithmetic along a named dimension | [`core/xr_mpi.py`](core/xr_mpi.py) (`XarrayMPI`) |
| `mpi.scatterv` | Scatter leading-axis slabs of a NumPy array | [`core/lib_mpi.py`](core/lib_mpi.py) |
| `@mpi` | Decorator: run a function on rank 0 (default), every rank, or broadcast the result | [`core/lib_mpi.py`](core/lib_mpi.py) |
| `mpi.watchdog` | Context manager: dump every rank's stack and abort after a period with no progress | [`core/lib_mpi.py`](core/lib_mpi.py) |
| `mpi.MPIError` | climtools's own exception for synchronized MPI failures | [`core/lib_mpi.py`](core/lib_mpi.py) |

`mpi.comm` is the real `mpi4py.MPI.Intracomm`, so anything not covered by
the accessors above — point-to-point `Send`/`Recv`, `Bcast`,
`Scatterv`/`Gatherv`, non-blocking collectives — is reached directly as
`mpi.comm.<method>` with full mpi4py signatures and IDE completion. See the
[mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/) for the
complete API.

### `mpi.reduce` vs `mpi.xarray`: which one do I want?

Both combine data across ranks with one MPI collective. The difference is
**where the split lives**:

| | `mpi.reduce` | `mpi.xarray` |
| --- | --- | --- |
| Each rank holds | A complete partial result (its own whole array/scalar/Dataset) | A slice of one shared dimension of a larger object |
| Combines by | Adding/comparing whole partials together, element-wise | Reducing along the named, distributed dimension |
| Typical source | Independent, embarrassingly-parallel work — one case per rank, then combine | `mpi.xarray.open_dataset`/`redistribute`, which partition a dimension across ranks |
| Example | Each rank sums a different subset of storm events into the same `(lat, lon)` grid; `mpi.reduce.sum` adds the 8 grids together | Each rank holds a different slice of `time`; `mpi.xarray.mean(dim="time")` reduces across ranks to one answer |

```python
# mpi.reduce: every rank already has a complete (lat, lon) composite over
# its own share of events; combine the 8 complete grids into one.
local_composite = build_local_composite()          # shape (lat, lon) on every rank
composite = mpi.reduce.sum(local_composite)         # same (lat, lon) result on every rank

# mpi.xarray: every rank holds a different slice of the "time" dimension of
# the *same* field; reduce across that shared dimension.
local_slice = mpi.xarray.redistribute(t2m, "time")  # each rank: its own time slice
time_mean = mpi.xarray.mean(local_slice, dim="time")  # same result on every rank
```

If you are not sure which applies: does every rank already have its own
*complete* answer that just needs combining (`mpi.reduce`), or does every
rank have a different *piece* of one shared dimension that needs reducing
(`mpi.xarray`)?

### `mpi.reduce`: element-wise reductions

Works on scalars, NumPy arrays, and xarray DataArrays/Datasets alike; dims,
coords, and attrs are kept.

```python
composite = mpi.reduce.sum(local)                        # every rank gets the result
composite = mpi.reduce.sum(local, mode="root", root=0)    # only rank 0 gets it; None elsewhere
```

`mpi.reduce` also exposes `prod`, `min`, `max`, `any`, and `all`, all with
the same `sum(value, *, mode="all", root=0)` signature. `mode="all"`
(default) gives every rank the result; `mode="root"` gives it only to
`root`.

### `mpi.xarray`: named-dimension distributed reductions

`mpi.xarray` reduces a `DataArray`/`Dataset` along a dimension that is
itself split across ranks:

```python
local_mean = mpi.xarray.mean(local_events, dim="event")  # same result on every rank
```

`local_mean` equals plain xarray's `assembled.mean(dim="event")` on the
fully assembled array — `mpi.xarray` gets there with one collective per
reduction instead of requiring the full array in one place first. It
exposes `sum`, `prod`, `min`, `max`, `mean`, `any`, and `all`, with
`skipna`/`min_count` applied consistently across the whole distributed
dimension (not per rank): `min_count` only drops a value once the count is
summed across every rank holding a share of `dim`, and `skipna=False`
propagates a NaN present on *any* rank to the combined result, not just
NaNs local to the current rank. Every reduction accepts a `DataArray` or a
`Dataset` interchangeably — a `Dataset` is reduced variable by variable,
leaving non-distributed variables and static dimensions untouched.

`mpi.xarray.open_dataset`/`redistribute`/`isel`/`sel` produce the
distributed objects these reductions consume; see
[`core/xr_mpi.py`](core/xr_mpi.py) for their full signatures.

### Native `.mean()`/`.sum()`/etc. on a distributed object are node-local

This is the single most common mistake with a distributed object, so it is
worth stating plainly:

> **Calling a distributed object's own `.mean()`/`.sum()`/`.max()`/etc.
> directly — instead of `mpi.xarray.mean`/`mpi.xarray.sum`/... — does not
> fail and does not raise. It silently returns *this rank's own partial
> reduction* over its own local slice of the distributed dimension, not the
> reduction over the whole (conceptual) array.**

```python
distributed = mpi.xarray.redistribute(t2m, "time")  # each rank: its own time slice

distributed.mean()          # WRONG: this rank's mean over its own slice only
mpi.xarray.mean(distributed, dim="time")  # RIGHT: combined across every rank
```

climtools does not patch or intercept native xarray reductions to guard
against this — a distributed object is an ordinary `xarray.Dataset`/
`DataArray` in every other respect, and leaving `.mean()` alone keeps it
that way. The rule is simply: once an object came from
`mpi.xarray.open_dataset`/`redistribute` (check `"mpi_meta" in obj.attrs` if
you need to test this programmatically), reduce it with `mpi.xarray`, not
its own methods, whenever the reduction touches the distributed dimension.
Reducing a dimension that is *not* the distributed one (for example
`distributed.mean(dim="lon")` when the distributed dimension is `"time"`) is
a legitimate, embarrassingly-parallel per-rank operation and needs no
`mpi.xarray` call at all.

### Arithmetic on distributed objects

`mpi.xarray.apply(left, op, right)` combines two operands element-wise with
no MPI communication, after checking that every rank already holds matching,
aligned local slices: either neither operand is distributed, both are
distributed identically (same dimension, same global size, same per-rank
bounds), or one is distributed and the other is replicated. Anything else
raises `ValueError` rather than silently combining misaligned data.

```python
anomaly = mpi.xarray.apply(t2m_distributed, "-", climatology_distributed)
```

`op` accepts a string token (`"+"`, `"-"`, `"*"`, `"/"`, `"//"`, `"%"`,
`"**"`, `"=="`, `"!="`, `"<"`, `"<="`, `">"`, `">="`, `"&"`, `"|"`, `"^"`) or
a two-argument callable such as `operator.add`.

`mpi.xarray.align(left, right, dim=None)` is the counterpart to
`xarray.align`, but for rank ownership rather than coordinate labels: it
returns `(left, right)` repartitioned so `apply` is guaranteed to accept
them.

```python
climatology, t2m = mpi.xarray.align(climatology_full, t2m_distributed)
anomaly = mpi.xarray.apply(t2m, "-", climatology)
```

`mpi.xarray.evaluate(expression, **variables)` parses `expression` with the
standard-library `ast` module and evaluates it through `apply`, so ordinary
operator precedence applies:

```python
result = mpi.xarray.evaluate("(a + b) * c - d / e", a=ds1, b=ds2, c=ds3, d=ds4, e=ds5)
```

See [`core/xr_mpi.py`](core/xr_mpi.py) for the full `apply`/`align`/
`evaluate` API, including the exact no-data-movement cases `align` resolves
locally.

### Parallel NetCDF-4 output

A complete Dataset can be written to one shared NetCDF-4 file with every
rank contributing its slab through a single MPI-collective write:

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
    composite = mpi.reduce.sum(local)  # same result on every rank

    # Rank 0 must hold the complete Dataset or DataArray; every other rank
    # binds empty_dataset() so `.xgeo` still resolves on a real xr.Dataset.
    # The writer builds the file schema from rank 0's object, scatters it
    # back out, and has every rank write its own slab collectively.
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

Every one of `mpi.reduce`/`mpi.xarray`/`mpi.scatterv`/
`to_netcdf(..., parallel=True)` and raw `mpi.comm.<method>` calls is
MPI-collective: every rank in `mpi.comm` must reach the same call, in the
same order, or the call blocks forever waiting for ranks that never arrive.
This is why the example above passes real data only on rank 0 and
`empty_dataset()` elsewhere — every rank still calls the writer at the same
point, even though only one of them supplies data.

Internally, `to_netcdf(..., parallel=True)` posts an `mpi.comm.Barrier()`
immediately after the collective `nc.close()` — inside
[`lib_netcdf/parallel.py`](lib_netcdf/parallel.py)'s `write_partitioned`/
`write_distributed`, right next to the `close()` it protects — so that by
the time the call returns on any rank, the file is fully written and closed
everywhere, and safe to reopen through a different communicator or a
non-parallel handle (as, for example, the test suite's own read-back
validation does immediately afterward). An earlier version of this also
called `nc.sync()` just before `close()`, on the theory that an explicit
flush could only make the guarantee stronger; in practice, on HDF5 1.14
(this project's actual deployment target — the sandbox this was first
validated in only had HDF5 1.10 available), an explicit collective `sync()`
immediately before the already-collective `close()` triggered a genuine
`NetCDF: HDF error` on close instead. `close()` on a file opened through the
MPI-IO/parallel HDF5 driver already performs its own collective,
synchronizing flush; calling `sync()` first was both redundant and, on the
stricter parallel-consistency checking newer HDF5 versions do, actively
harmful. Removed.

See [`core/xgeo.py`](core/xgeo.py) and [`lib_netcdf/parallel.py`](lib_netcdf/parallel.py)
for the full `to_netcdf` signature (chunking, compression, unlimited
dimensions, MPI-IO hints).

### Running under MPI

Launch with `python -m mpi4py`, not a bare `python`. mpi4py calls
`MPI_Init_thread` when `mpi4py.MPI` is imported and registers
`MPI_Finalize` to run at interpreter exit, so an unhandled exception on a
subset of ranks does not terminate the job by default: the failing ranks
block in `MPI_Finalize` waiting for the others, and the others block in the
collective the failing ranks never reached. `python -m mpi4py` installs a
finalizer hook that calls `MPI_Abort` instead, converting that deadlock into
a clean non-zero exit — see
[mpi4py: Exceptions and deadlocks](https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks).
Because the launch command is outside climtools's control, importing
`climtools.mpi` under an MPI launcher also installs the equivalent
`sys.excepthook` as a fallback (a no-op when `python -m mpi4py` already
installed its own, and on single-rank runs with no launcher).

Neither hook helps when a rank is blocked rather than failing, since a
blocked rank raises nothing. `mpi.watchdog(phase)` covers that case:

```python
with mpi.watchdog("compositing"):
    result = mpi.xarray.mean(local_events, dim="event")
```

It arms a daemon thread on every rank. mpi4py releases the GIL for the
duration of a blocking MPI call, so the thread keeps running while the main
thread sits in `Allreduce` or a blocking read, and prints that rank's own
traceback naming the line it is stuck on before calling `MPI_Abort`. Every
rank dumps independently, so the log distinguishes ranks that arrived from
ranks that did not. Default: 600 s of no progress; pass `timeout=` to
change it, or `timeout=0` to leave the block unguarded. Every rank waits the
same delay before calling `Abort` (not one scaled by its own rank number),
so a rank that is genuinely stuck cannot tear the job down before a
slower-but-fine rank has flushed its own dump.

The same script runs unchanged on a single rank without a launcher, passing
`allow_serial=True` to `to_netcdf`; `mpi.reduce` degrades to a no-op
reduction over one rank. `mpi4py` is a hard dependency of `climtools.mpi`
and of the NetCDF writer, so it must be installed (see
[Installation](#installation)) even for single-rank runs.

## Testing

[`tests/`](tests/) holds the test suite, split by whether a test needs an MPI
launcher:

| File | Demonstrates |
| --- | --- |
| [`test_general.py`](tests/test_general.py) | Every non-MPI component: `plot.geo` (rendering, the `.xgeo` accessor form, `.add.*` overlay chaining), `calc` (`trends` — both Mann-Kendall and `polyfit` — `corr`, `pvalues`), `cmaps` (every registered name resolves, `create`/`concat`/`add`/`get_colors`), `xgeo`/`xr_utils` (`to_lon180`, `add_local_solar_time`, `sel_transect`, `get_spatial_dims`), the serial NetCDF writer (`xgeo.to_netcdf` and `append`, round-tripped through `xr.open_dataset`), `core.tools` (`n_cpus`, `LockFile`, `AttrDict`), and `cdo` (skipped cleanly when the `cdo` binary is not on `PATH`). Plain `python`, one process, no MPI launcher, no network access required. |
| [`test_mpi.py`](tests/test_mpi.py) | Correctness and scaling suite for `mpi.reduce`, `mpi.xarray` (`open_dataset`/`redistribute`/`isel`/`sel`, `apply`/`align`/`evaluate`), `mpi.scatterv`, and the parallel NetCDF writer. Self-contained: rank 0 builds a deterministic mock NetCDF file. `--time-steps`/`--resolution` set its size. |
| [`test.sh`](tests/test.sh) | Slurm batch script (`sbatch tests/test.sh`) running both suites — `test_general.py` directly, then `test_mpi.py` on eight ranks — with the environment settings the MPI suite requires, including `HDF5_USE_FILE_LOCKING=FALSE`. |

```bash
python tests/test_general.py

python -m mpi4py tests/test_mpi.py
mpirun -n 8 python -m mpi4py tests/test_mpi.py --time-steps 7200
```

Every timed check in `test_mpi.py` is tagged in the summary with a
plain-language verdict — `MPI (faster)`, `Xarray (faster)`, or `tie` (within
5% either way) — so the final table answers "was this actually worth
distributing" at a glance. Small, in-memory, single-node checks routinely
come back `Xarray (faster)`: real speedups from `mpi.reduce`/`mpi.xarray`
show up once ranks are on separate cores with enough data per rank to
outweigh collective-call overhead — run with `-n <= your core count` for a
meaningful comparison. `test_general.py` is not a scaling benchmark; every
check there is correctness-focused and independent of core count.

## License

MIT — see [`LICENSE`](LICENSE).
