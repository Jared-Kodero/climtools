# climtools

Utilities for climate data analysis and plotting: geospatial operations, Cartopy map plotting, statistical analysis, colormaps, and CDO utilities.

- Repository: https://github.com/Jared-Kodero/climtools

- Requires Python ≥ 3.12

## Contents

- [Package layout](#package-layout)

- [Installation](#installation)

- [Quick start: plotting](#quick-start-plotting)

- [Testing](#testing)

- [License](#license)

## Package layout

| Namespace | Purpose | Source |

| --- | --- | --- |

| `climtools.plot` | Cartopy map plotting. Entry point `plot.geo`. | [`viz/plotting.py`](viz/plotting.py) |

| `climtools.xgeo` | Geospatial operations and NetCDF output: regridding, masking, transects, local solar time, `to_netcdf`. | [`core/xgeo.py`](core/xgeo.py), [`lib_netcdf/`](lib_netcdf/) |

| `climtools.calc` | Trends, correlations, difference-of-means testing. | [`core/calc_stats.py`](core/calc_stats.py) |

| `climtools.cmaps` | Colormaps: local IPCC tables, matplotlib, cmocean. | [`viz/cmaps.py`](viz/cmaps.py) |

| `climtools.cdo` | Thin wrapper over the CDO command-line tool. | [`cdo/pycdo.py`](cdo/pycdo.py) |

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


## License

MIT -- see [`LICENSE`](LICENSE).