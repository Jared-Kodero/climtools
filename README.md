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
    .xgeo.mask_land(keep="land")
    .xgeo.add_lst()
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

## Notes

- The map entry point is `plot.geo`, reached either as the free function
  `climtools.plot.geo(da, ...)` or as the accessor `da.xgeo.plot.geo(...)`.
- All input coordinates are interpreted in `PlateCarree`. The display
  projection is chosen with `projection=`, or inferred from the data extent.
