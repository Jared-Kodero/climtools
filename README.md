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
    - [Rank-0-source vs. already-distributed: which one at scale](#rank-0-source-vs-already-distributed-which-one-at-scale)
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

Both provide rank-aware reductions, but `mpi.xarray` communicates only when
the requested reduction crosses its distributed dimension. The key difference
is **where the split lives**:

| | `mpi.reduce` | `mpi.xarray` |
| --- | --- | --- |
| Each rank holds | A complete partial result (its own whole array/scalar/Dataset) | A slice of one shared dimension of a larger object |
| Combines by | Adding/comparing whole partials together, element-wise | Reducing along the named, distributed dimension |
| Typical source | Independent, embarrassingly-parallel work — one case per rank, then combine | `mpi.xarray.open_dataset`/`redistribute`, which partition a dimension across ranks |
| Example | Each rank sums a different subset of storm events into the same `(lat, lon)` grid; `mpi.reduce.sum` adds the 8 grids together | Each rank holds a different slice of `time`; `mpi.xarray.mean(dim="time")` reduces across ranks, then repartitions the result on the longest surviving dimension |

```python
# mpi.reduce: every rank already has a complete (lat, lon) composite over
# its own share of events; combine the 8 complete grids into one.
local_composite = build_local_composite()  # shape (lat, lon) on every rank
composite = mpi.reduce.sum(local_composite)  # same (lat, lon) result on every rank

# mpi.xarray: every rank holds a different slice of the "time" dimension of
# the *same* field; reduce across that shared dimension.
local_slice = mpi.xarray.redistribute(t2m, "time")  # each rank: its own time slice
time_mean = mpi.xarray.mean(local_slice, dim="time")
# "time" is gone, so climtools repartitions time_mean on the longest
# surviving dimension whose length is greater than one.
```

If you are not sure which applies: does every rank already have its own
*complete* answer that just needs combining (`mpi.reduce`), or does every
rank have a different *piece* of one shared dimension that needs reducing
(`mpi.xarray`)?

### `mpi.reduce`: element-wise reductions

Works on scalars, NumPy arrays, and xarray DataArrays/Datasets alike; dims,
coords, and attrs are kept.

```python
composite = mpi.reduce.sum(local)  # every rank gets the result
composite = mpi.reduce.sum(
    local, mode="root", root=0
)  # only rank 0 gets it; None elsewhere
```

`mpi.reduce` also exposes `prod`, `min`, `max`, `any`, and `all`, all with
the same `sum(value, *, mode="all", root=0)` signature. `mode="all"`
(default) gives every rank the result; `mode="root"` gives it only to
`root`.

### `mpi.xarray`: named-dimension distributed reductions

`mpi.xarray` accepts the same named reduction dimensions as xarray and decides
from the object's `mpi_meta` whether communication is actually required. If the
requested dimensions do **not** include the partition dimension, the complete
reduction domain already exists on each rank: climtools calls native xarray
locally, performs no MPI collective, and preserves the existing partition
metadata. For an object partitioned on `time`, for example:

```python
local = mpi.xarray.redistribute(t2m, "time")
local_zonal_mean = mpi.xarray.mean(local, dim="lon")
# no MPI traffic; every rank still owns the same global time[start:stop] slab
```

If the requested dimensions **do** include the partition dimension, xarray
first collapses all requested dimensions over that rank's local slab in one
operation, and MPI combines only those already-reduced partials. Thus
`dim=("time", "lat")` does not communicate the latitude dimension: each rank
reduces its local `(time, lat)` domain first, then MPI combines the remaining
partial array. `sum`/`prod` use `MPI.SUM`/`MPI.PROD`; `min`/`max` use global
extrema with the existing NaN and empty-partition validity handling; `any` and
`all` use `MPI.LOR` and `MPI.LAND`; and `mean` combines global sums and counts
rather than averaging rank means. `min_count` likewise uses a count summed
across ranks whenever the partition dimension participates.

Once a reduction consumes the partition dimension, the old ownership metadata
is no longer valid. The rank-local partials are combined with `Allreduce`, so
the complete global reduction exists briefly on every rank. By default every
`mpi.xarray` reduction has `redistribute_on="auto"`: climtools immediately
repartitions that global result along its longest surviving dimension whose
length is greater than one. This keeps large post-reduction fields distributed
for subsequent work instead of leaving a complete copy on every rank. A scalar,
or a result whose surviving dimensions are all length one, remains replicated.
Pass `redistribute_on=<dim>` to choose the new partition explicitly, or
`redistribute_on=None` to disable post-reduction redistribution and deliberately
leave the complete global result replicated on every rank. When the original
partition dimension survives the reduction, `"auto"` and None both preserve
that existing partition; naming a different partition dimension is invalid
because no redistribution is needed or implied by a local reduction.

Result-placement options such as `mode="root"` and `root=` belong exclusively
to `mpi.reduce`; they do not exist on the `mpi.xarray` reduction API.

```python
local = mpi.xarray.redistribute(t2m, "time")

time_mean = mpi.xarray.mean(local, dim="time")
# redistribute_on="auto" is the default. "time" is gone, so if lon is the
# longest surviving dimension (>1), time_mean now carries mpi_meta for lon.

replicated_mean = mpi.xarray.mean(local, dim="time", redistribute_on=None)
# The same global (lat, lon) mean is deliberately left complete on every rank.

global_max = mpi.xarray.max(local)
# dim=None means all dimensions: each rank computes its local scalar maximum,
# MPI.MAX combines those scalars, and no dimension remains to redistribute.
```

Every reduction accepts a `DataArray` or a `Dataset` interchangeably. Dataset
planning remains variable-specific: a variable that does not carry the active
partition dimension is reduced locally even when another variable in the same
Dataset requires an MPI combine. `mpi.xarray.open_dataset`/`redistribute`/
`distribute`/`isel`/`sel` produce the distributed objects these reductions
consume; see [`core/xr_mpi.py`](core/xr_mpi.py) for their full signatures.

### Native `.mean()`/`.sum()`/etc. on a distributed object are node-local

This is the single most common mistake with a distributed object, so it is
worth stating plainly:

> **Calling a distributed object's own `.mean()`/`.sum()`/`.max()`/etc.
> directly does not know that `mpi_meta` represents a larger conceptual
> array. If the reduction touches the partition dimension, native xarray
> returns only this rank's partial reduction, not the global result.**

```python
distributed = mpi.xarray.redistribute(t2m, "time")  # each rank: its own time slice

distributed.mean()  # WRONG: this rank's mean over its own slice only
mpi.xarray.mean(distributed, dim="time")  # RIGHT: combined across every rank
```

climtools does not patch native xarray methods. Use `mpi.xarray` for reductions
on distributed objects when you want the distribution semantics carried
forward correctly. If the requested dimensions exclude the partition
dimension, `mpi.xarray` now takes the same embarrassingly-parallel native
xarray path internally, but also preserves the object's validated `mpi_meta`
and avoids even the reduction-plan agreement collective. If the partition
dimension is included — explicitly, in a dimension tuple, or implicitly by
`dim=None`/`...` — `mpi.xarray` performs the required global MPI combine.

### Arithmetic on distributed objects

`mpi.xarray.apply(func, *args, **kwargs)` calls `func(*args, **kwargs)`
rank-locally with no MPI communication, after checking that every
distributed xarray argument shares one partition and every replicated
argument that carries the distributed dimension matches its length: either
no argument is distributed, every distributed argument is distributed
identically (same dimension, same global size, same per-rank bounds), or an
argument is distributed and the rest are replicated. Anything else raises
`ValueError` rather than silently combining misaligned data. This is a
`pandas.DataFrame.apply`-style interface: `func` can be any callable, not
only a binary operator.

```python
anomaly = mpi.xarray.apply(operator.sub, t2m_distributed, climatology_distributed)
```


`mpi.xarray.align(left, right, dim=None)` is the counterpart to
`xarray.align`, but for rank ownership rather than coordinate labels: it
returns `(left, right)` repartitioned so `apply` is guaranteed to accept
them.

```python
climatology, t2m = mpi.xarray.align(climatology_full, t2m_distributed)
anomaly = mpi.xarray.apply(operator.sub, t2m, climatology)
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

#### Rank-0-source vs. already-distributed: which one at scale

`to_netcdf(..., parallel=True)` takes any of three kinds of input, and picks
the path by inspecting the object, not by an argument you set:

| Input | Path | Rank 0's peak memory |
| --- | --- | --- |
| A plain, **eager** (non-dask) `Dataset`/`DataArray` on rank 0, `empty_dataset()` elsewhere | Legacy scatter | The **entire** output — every variable, in full, before it is scattered out |
| A plain, **dask-backed** `Dataset`/`DataArray` on rank 0, `empty_dataset()` elsewhere | Auto-distributed | Only rank 0's own slice — `to_netcdf` calls `mpi.xarray.distribute` internally before writing |
| An object already carrying `mpi_meta` (from `mpi.xarray.open_dataset`/`redistribute`/`distribute`) | Distributed | Only that rank's own slice — no gather, no scatter |

The middle row is automatic: nothing about how you call `to_netcdf` changes.
Building the same rank-0-source Dataset dask-backed instead of eager — for
example `xr.open_mfdataset(paths, chunks=...)` on rank 0 instead of loading
eagerly — is enough to opt into it. `to_netcdf` logs which path it took
(search a run's log for `"rank-0 source"`), so this is visible either way:

```text
xgeo.to_netcdf (rank-0 source, dask-backed): distributing lazily instead of materializing on rank 0.
xgeo.to_netcdf (rank-0 source): rank 0 holds 42.3GiB before scatter, ~5.3GiB/rank after.
```

The eager row's cost is unavoidable at the design level, not a bug: the
array is already fully resident in rank 0's memory by the time `to_netcdf`
sees it (that is what "eager" means), so there is nothing left to make lazy.
It only matters at large `TIME_STEPS` or fine resolution — at small sizes it
is unremarkable — but when a write is slow or failing at scale, this row is
the first thing worth ruling out. Making the *source* dask-backed (the
middle row) is what actually avoids it.

One caveat worth knowing rather than being surprised by: on the
auto-distributed and already-distributed paths, rank 0 specifically may
compute its own share of a dask-backed input twice — once while `to_netcdf`
samples dtype and shape to create the file's structure, once more during
the actual collective write. Every other rank computes its own share
exactly once. This is a pre-existing characteristic of the underlying
writer, not something scale-dependent or memory-related: no rank ever
touches more than its own slice, computed twice or not.

If the eager row's cost is uncomfortable and making the source dask-backed
isn't an option, avoid ever holding the full array anywhere by building it
in already-distributed form instead. This does **not** mean shipping pieces
of a rank-0 array out to other ranks over MPI yourself — it means every rank
builds the *same lazy recipe* independently, and only ever materializes its
own slice of it.

**Already on disk.** `mpi.xarray.open_dataset(path, partition_dim=...)` opens
the file on every rank and each rank reads only its own slice from disk —
see [`mpi.xarray`: named-dimension distributed reductions](#mpixarray-named-dimension-distributed-reductions)
and the worked example in [Parallel NetCDF output](#parallel-netcdf-4-output)
above.

**Computed, not read from a file.** If the 200 GiB comes from a computation
rather than a direct file read, run that computation as a **dask-lazy graph,
built identically on every rank** — building a dask graph is cheap metadata
work regardless of the size it describes, since no data moves until
something calls `.load()`/`.compute()` — and hand the *lazy* result to
`mpi.xarray.redistribute`, not `to_netcdf`, directly:

```python
import xarray as xr
from climtools import mpi

# Every rank builds this identically. Nothing is computed yet: dask.array
# operations (indexing, arithmetic, ufuncs, .map_blocks, xr.apply_ufunc with
# dask="parallelized") all stay lazy until something forces evaluation.
lazy = xr.open_mfdataset(paths, chunks={"time": 100}, parallel=True)
transformed = some_expensive_transform(lazy)  # still lazy: no data read yet

# redistribute() is itself pure metadata/slicing -- it never calls .load()
# or .compute(). Only this rank's own bounds along "time" are kept.
local = mpi.xarray.redistribute(transformed, dim="time")

# .load() is the only point anything is computed, and it computes only
# this rank's own slice -- not the full 200 GiB, on any rank, ever.
local = local.load()

local.xgeo.to_netcdf("output.nc", partition_dim="time", parallel=True)
```

This works because `redistribute` assumes its input is already the *same*
object on every rank (see [`core/xr_mpi.py`](core/xr_mpi.py)) and slices it
locally with no MPI communication for the data itself — which is exactly a
lazy scatter when the input is dask-backed, and needs no special support to
be one.

**Data that can only be produced by one rank.** `redistribute` needs every
rank to be able to rebuild `value` identically — a resource only one rank
has credentials for, or a computation depending on rank-local state, breaks
that assumption. `mpi.xarray.distribute` is for exactly that case: `value`
is real on one rank (`root`, default rank 0) and `None` everywhere else.
`root` slices it — lazily, if it is dask-backed, so slicing never triggers
computation — and sends each other rank only its own slice by direct
point-to-point message; `root` never holds more than one slice's worth of
pickled graph in flight, and every rank materializes only what it was sent:

```python
if mpi.comm.rank == 0:
    full = build_dataset_only_possible_on_rank_0()  # dask-backed
else:
    full = None

local = mpi.xarray.distribute(full, dim="time")  # still lazy
local = local.load()  # only now, and only this rank's own slice
local.xgeo.to_netcdf("output.nc", partition_dim="time", parallel=True)
```

If the only thing `local` is for is this `to_netcdf` call, calling
`distribute` explicitly like this is no longer necessary: `to_netcdf`
detects a dask-backed rank-0-source input and does exactly this internally
(see the table above). Calling it explicitly still matters when the result
is needed for something *besides* the write — an `mpi.xarray`/`mpi.reduce`
computation on the distributed data before writing it, for example — or
when explicit control over the moment slicing happens is worth having.

If `full` is already a plain in-memory (non-dask) object rather than
dask-backed, calling `distribute` on it directly still avoids `to_netcdf`'s
own scatter-path redundant copies, but pays point-to-point pickling instead
of `scatterv`'s zero-copy buffer transfer, and cannot undo the fact that the
complete array already had to exist in `root`'s memory before the call —
only a dask-backed source avoids ever holding more than one rank's share
anywhere. For an eager source headed straight to `to_netcdf`, just calling
`to_netcdf` directly (skipping `distribute` and `.load()` above) is both
simpler and faster: it already takes the scatter path this describes.

**Data with no existing source at all — you are generating it.**
`redistribute` needs an identical recipe every rank can already build;
`distribute` needs a source that exists on one rank. Neither fits synthetic
or procedurally generated data — a mock dataset for testing, a per-rank RNG
stream, one file per rank — where there is no array or file to slice in the
first place. `mpi.xarray.create_dataarray`/`create_dataset` are for this:
every rank computes `get_balanced_bounds(length, rank, size)` independently
(a pure function of the global length, this rank's number, and the
communicator size — no communication at all) and calls your own function
with only its own `(start, stop)`, so nothing is ever built on one rank and
sent to another:

```python
import numpy as np
from climtools import mpi, xgeo


def fill_pr(start: int, stop: int) -> np.ndarray:
    # called once per rank, with THIS rank's own global bounds along "time" --
    # rank 3 of 8 gets (270, 360), not (0, 90)
    t = np.arange(start, stop, dtype=np.float32)
    return some_formula(t)  # shape (stop - start, n_lat, n_lon)


def fill_slmsk() -> np.ndarray:
    # not partitioned (no "time" in its dims) -- identical on every rank,
    # so this takes no arguments
    return build_mask()  # shape (n_lat, n_lon)


ds = mpi.xarray.create_dataset(
    data_vars={
        "pr": (("time", "lat", "lon"), fill_pr),
        "slmsk": (("lat", "lon"), fill_slmsk),
    },
    # sizes= is unnecessary here: every dimension named above ("time",
    # "lat", "lon") has a full-length coordinate below, and reading a
    # coordinate's length never forces any computation, since coordinates
    # are always plain, eager arrays -- never a lazy fill function. Give
    # sizes={"lon": 1440} instead of a "lon" coordinate below to cover a
    # dimension with no coordinate of its own.
    coords={
        "time": np.arange(8760, dtype=np.float64),
        "lat": np.linspace(-90, 90, 721, dtype=np.float32),
        "lon": np.linspace(-180, 180, 1440, endpoint=False, dtype=np.float32),
    },
    dim="time",
)
# ds is dask-backed and not yet computed. Writing it runs each rank's own
# fill() calls as part of the write -- rank 0 never materializes more than
# its own slice, at any point, for any variable.
xgeo.to_netcdf(ds, "output.nc", partition_dim="time", parallel=True)
```

`fill` is wrapped in `dask.delayed`, so it runs only when this rank's slice
is actually needed (loaded, reduced, or written), not eagerly at the
`create_dataset` call. A coordinate matching `dim` can be passed the same
way as to the ordinary `xr.Dataset` constructor — a full-length array — and
is auto-sliced to this rank's own bounds. `create_dataset` also accepts a
bare `xr.DataArray` as a `data_vars` value (one already built by
`create_dataarray`, for instance), exactly as `xr.Dataset`'s own constructor
accepts a bare DataArray alongside `(dims, array)` tuples. See
`create_dataarray`'s and `create_dataset`'s docstrings in

[`core/xr_mpi.py`](core/xr_mpi.py) for the single-DataArray form and the
full set of options.

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

#### On-disk chunking: distribution_chunks vs save_chunks

Two different chunk concepts are involved in a parallel write, computed
independently in [`core/xr_chunks.py`](core/xr_chunks.py):

- **distribution_chunks** — how much of the partition dimension each MPI
  rank holds *in memory*. This is what `mpi.xarray.open_dataset`/
  `redistribute`/`distribute` decide, and it is recorded in
  `mpi_meta["chunk_info"]`. This is a single scalar chunk length per
  dimension throughout — not an irregular, multiple-block-per-dimension
  structure like a raw Dask chunk tuple (`(1000, 1000, 240)`); see the
  design doc linked below for that unimplemented extension.
- **save_chunks** — the actual on-disk NetCDF-4/HDF5 chunk shape written
  for each variable. A save chunk's length comes from dask's own
  `"auto"` byte-size heuristic against the variable's true *global*
  shape and dtype (via a lazy `dask.array.zeros` mock — no data is ever
  materialized to decide this), capped to HDF5's 4 GiB per-chunk limit.

Along the partition dimension specifically, that dask-proposed,
HDF5-capped length is not used as-is: `mpi.xarray.redistribute`'s
`get_chunk_bounds` normally splits the partition dimension into whole
`chunk_info[dim]`-sized chunks dealt round-robin across ranks, so a save
chunk that evenly divides `chunk_info[dim]` is guaranteed to sit inside
one rank's slab. But `get_chunk_bounds` falls back to a plain balanced
split (`get_balanced_bounds`, no chunk structure at all) whenever there
are fewer whole `chunk_info[dim]`-sized chunks than ranks — a case that
can occur for perfectly ordinary inputs (e.g. splitting a 1000-length
time axis across 64 ranks). In that fallback regime a divisor of
`chunk_info[dim]` is not actually guaranteed to divide the real,
irregular rank boundaries, so `compute_save_chunks` checks
`chunk_alignment_holds` first and forces the partition-dimension
save_chunk to a single element whenever the fallback applies, rather
than risking a straddle. `chunk_alignment_holds` uses the exact same
condition `get_chunk_bounds` branches on, so the two can never diverge.

For an already-distributed object (the third row of the table above),
`to_netcdf(..., parallel=True)` computes save_chunks this way
automatically, whenever the caller does not pass an explicit `chunks=`
mapping, and records the result under `mpi_meta["save_chunks"]` before
writing. This runs as one extra MPI-collective planning step
(`mpi.xarray.attach_save_chunks`, itself a small `bcast`) ahead of the
write; nothing about it needs a separate call from user code.

For the eager and dask-backed rank-0-source rows, the on-disk chunk
shape is instead derived directly from the real, complete dataset rank 0
already holds (`get_partition_chunk_size`/`get_chunks`), since there is
no local-slice-vs-global-shape gap to close in that case.

Two caveats worth knowing before relying on the no-straddle guarantee in
production:

- The partition-dimension save_chunk still depends on `mpi_size`
  whenever the run's `distribution_chunk` ends up smaller than dask's
  data-driven proposal (in the aligned regime above) — a save chunk
  physically cannot be larger than what one rank's write can supply
  without straddling a neighbor, so at high enough rank counts some
  rank-count sensitivity is unavoidable given the current architecture.
- If `distribution_chunk` happens to be prime and dask's capped proposal
  is smaller than it, the only value that divides it evenly is 1 — still
  correct, but a partition-dimension save_chunk of length 1 defeats
  HDF5 chunking's purpose and should be benchmarked on real dimension
  sizes, not assumed acceptable, before relying on this path at scale.

See [`core/xr_chunks.py`](core/xr_chunks.py)'s module docstring for the
full derivation, and
[`docs/phase2-rank-distribution-design.md`](docs/phase2-rank-distribution-design.md)
for the (not yet implemented) design that would remove the residual
`mpi_size` dependency by generalizing rank-boundary computation to
irregular, Dask-native chunk tuples.

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
| [`test_mpi.py`](tests/test_mpi.py) | Correctness and scaling suite for `mpi.reduce`, `mpi.xarray` (`open_dataset`/`redistribute`/`distribute`/`isel`/`sel`, `apply`/`align`/`evaluate`), `mpi.scatterv`, and the parallel NetCDF writer — including all three `to_netcdf(parallel=True)` input paths (eager, dask-backed auto-distributed, already-distributed) and a scale sweep (`SCALE_SWEEP_CASES`) across several `(time, lat, lon)` sizes, specifically including the exact size that caused a historical hang, so a regression there is caught directly rather than depending on which `--time-steps` a particular run happens to use. Self-contained: rank 0 builds a deterministic mock NetCDF file. `--time-steps`/`--resolution` set its size. |
| [`test_mpi_xarray_reductions.py`](tests/test_mpi_xarray_reductions.py) | Focused regression checks for reduction placement: communication-free non-partition reductions, mixed dimension tuples, automatic longest-dimension redistribution after the partition dimension is consumed, scalar/length-one results, explicit redistribution, attrs, extrema/logical reductions, and the absence of `mode`/`root` from `mpi.xarray` reductions. |
| [`test.sh`](tests/test.sh) | Slurm batch script (`sbatch tests/test.sh`) running both original suites — `test_general.py` directly, then `test_mpi.py` on eight ranks — with the environment settings the MPI suite requires, including `HDF5_USE_FILE_LOCKING=FALSE`. |

```bash
python tests/test_general.py

python -m mpi4py tests/test_mpi.py
mpirun -n 8 python -m mpi4py tests/test_mpi.py --time-steps 7200
mpirun -n 8 python -m mpi4py tests/test_mpi_xarray_reductions.py
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
