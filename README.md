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
| `climtools.mpi`   | MPI runtime: `mpi.comm` for the raw communicator, `mpi.reduce` for collective reductions, backing parallel NetCDF-4 output. |

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
scripts/setup_env.sh [env_name]
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
`scripts/setup_env.sh` builds that stack automatically, without ever using a
distro package manager (apt/yum builds are not available on HPC login or
compute nodes, which is what this stack is for): it first tries `module
load`-ing a matching MPI and `netcdf-mpi` module pair, and falls back to
compiling HDF5 and netCDF-C from source against the active MPI compiler.
Everything else in climtools works with the ordinary serial `netCDF4` and
`mpi4py` wheels from PyPI or conda-forge.

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
    .add.contour(z500, colors="k", clabel=True)   # line contours, labelled
    .add.quiver(u10, v10, subsample=4)             # vector overlay
    .add.significance(p_value)                     # stippling where p < 0.05
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
    ds.t2m
    .xgeo.remap(target_grid, method="conservative")
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
    composite = mpi.reduce.sum(local)                      # same result on every rank
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
mpirun -n 8 python script.py
srun --ntasks=8 --mpi=pmix python script.py
```

The same script runs unchanged on a single rank without a launcher, by
passing `allow_serial=True` to `to_netcdf`; `mpi.reduce` degrades to a no-op
reduction over one rank. `mpi4py` is a hard dependency of `climtools.mpi` and
of the NetCDF writer, so it must be installed (see
[Installation](#installation)) even for single-rank runs, and MPI
initializes as a side effect of importing it, not deferred until a
collective call.

`mpi.reduce` also exposes `prod`, `min`, `max`, `any`, and `all`, all
following the same signature as `sum`. Every reduction accepts `mode="all"`
(the default: every rank gets the result) or `mode="root"` with a `root=`
rank (only that rank gets the result; `None` elsewhere). `mpi.comm` is the
native `mpi4py.MPI.Intracomm`, so anything not covered above — point-to-point
`Send`/`Recv`, `Bcast`, `Scatterv`/`Gatherv`, non-blocking collectives — is
reached directly as `mpi.comm.<method>` with full mpi4py signatures and IDE
completion.

Parallel output requires `netCDF4` and `mpi4py` built against a
parallel-enabled MPI/HDF5/NetCDF-C stack; see
[Installation](#installation) and `scripts/setup_env.sh`.

## Notes

- The map entry point is `plot.geo`, reached either as the free function
  `climtools.plot.geo(da, ...)` or as the accessor `da.xgeo.plot.geo(...)`.
- All input coordinates are interpreted in `PlateCarree`. The display
  projection is chosen with `projection=`, or inferred from the data extent.
