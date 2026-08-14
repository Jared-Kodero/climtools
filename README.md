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
| `climtools.xgeo`  | Geospatial operations: regridding, masking, transects, local solar time. |
| `climtools.calc`  | Trends, correlations and difference-of-means testing.                     |
| `climtools.cmaps` | Colormaps spanning local IPCC tables, matplotlib and cmocean.             |
| `climtools.cdo`   | Thin xarray-aware wrapper over the CDO command-line tool.                 |
| `climtools.theme` | Publication styling for matplotlib and seaborn.                           |
| `climtools.lib_mpi` | MPI decorators, collective reductions and parallel NetCDF-4 output.     |

## Installation

This is a local developer package. Clone the repository and import it from the
repository root, or install it into your environment.

```bash
git clone https://github.com/Jared-Kodero/climtools.git
```

Regridding requires `xesmf`, which is imported only when a regridding function
is called. The rest of the package works without it.

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

Large workloads can be split across MPI ranks and written to one shared
NetCDF-4 file, with no gather to rank zero and no per-rank files to merge.

```python
from climtools import MPI
from climtools import lib_mpi as mpi

@MPI(all_ranks=True)
def main():
    local = ds.xgeo.mpi.partition("event")     # this rank's contiguous block
    composite = mpi.total(local.sum("event"))  # reduce across ranks
    local.xgeo.mpi.to_netcdf("events.nc", partition_dim="event")

main()
```

```bash
mpirun -n 8 python script.py
srun --ntasks=8 --mpi=pmix python script.py
```

The same script runs unchanged without a launcher, on a single rank. MPI is
never initialized until a collective is actually called, so importing
climtools in a serial session costs nothing.

This requires the native extension, built once with `lib_mpi/install.sh`. See
[`lib_mpi/README.md`](lib_mpi/README.md) for the step-by-step guide to writing
code that uses it, and [`lib_mpi/BUILD.md`](lib_mpi/BUILD.md) for the build.

## Notes

- The map entry point is `plot.geo`, reached either as the free function
  `climtools.plot.geo(da, ...)` or as the accessor `da.xgeo.plot.geo(...)`.
- All input coordinates are interpreted in `PlateCarree`. The display
  projection is chosen with `projection=`, or inferred from the data extent.
