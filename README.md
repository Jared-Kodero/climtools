# climtools

<p align="center">
  <strong>Climate-data analysis, geospatial processing, visualization, and distributed Xarray workflows.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/climtools/"><img src="https://img.shields.io/pypi/v/climtools?label=PyPI" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-%E2%89%A53.12-blue" alt="Python 3.12+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://github.com/Jared-Kodero/climtools"><img src="https://img.shields.io/badge/GitHub-climtools-181717?logo=github" alt="GitHub repository"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#xarray-accessor">Xarray accessor</a> ·
  <a href="#plotting">Plotting</a> ·
  <a href="#mpi-xarray">MPI-Xarray</a> ·
  <a href="#testing">Testing</a>
</p>

`climtools` is a compact toolkit for climate and geoscience workflows built around [Xarray](https://xarray.dev/). It combines geospatial processing, Cartopy-based visualization, statistical analysis, scientific colormaps, NetCDF utilities, CDO integration, and an MPI-parallel Xarray layer for distributed-memory workloads.

A central design feature is the `.xgeo` Xarray accessor. After importing `climtools`, common operations can be called directly from `xarray.DataArray` and `xarray.Dataset` objects instead of repeatedly passing the object into standalone helper functions.

## Highlights

* **Xarray-native API** through `da.xgeo` and `ds.xgeo`.
* **Geospatial analysis** including regridding, masking, transects, longitude handling, and local solar time.
* **Publication-oriented maps** with Cartopy projections, faceting, overlays, vector fields, significance masks, and animation.
* **Statistics** for trends, correlations, and difference-of-means testing.
* **Scientific colormaps** from local IPCC tables, Matplotlib, and cmocean.
* **NetCDF and CDO workflows** for common climate-data processing tasks.
* **MPI-Xarray** for distributed partitioning, halo-aware operations, reductions, redistribution, and parallel NetCDF output.

## Installation

Install from PyPI:

```bash
pip install climtools
```

Optional regridding support uses `xesmf`:

```bash
pip install "climtools[regrid]"
```

For development:

```bash
git clone https://github.com/Jared-Kodero/climtools.git
cd climtools
pip install -e .
```

Python **3.12 or newer** is required.

For MPI-collective parallel NetCDF-4 output across multiple ranks, `netCDF4` and `mpi4py` must be linked against a parallel-enabled MPI/HDF5/NetCDF-C stack. The repository includes [`env/setup_env.sh`](env/setup_env.sh) and [`env/environment.yml`](env/environment.yml) for this environment.

Some features also require external executables on `PATH`:

* `climtools.cdo`: `cdo` and `nco`
* `da.xgeo.plot.animate(...)`: `ffmpeg`

## Quick start

Importing `climtools` registers the `.xgeo` accessor on Xarray objects.

```python
import xarray as xr

import climtools
from climtools import cmaps


ds = xr.open_dataset("climate.nc")
t2m = ds["t2m"]

plot = t2m.xgeo.plot.geo(
    method="contourf",
    cmap=cmaps.temp_div(),
    levels=21,
    gridlines=True,
)
```

The same plotting operation is also available through the functional API:

```python
from climtools import xgeo as xg

plot = xg.plot.geo(
    t2m,
    method="contourf",
    cmap=cmaps.temp_div(),
    levels=21,
    gridlines=True,
)
```

## Xarray accessor

Most day-to-day geospatial operations can be called directly on Xarray objects. The bound `DataArray` or `Dataset` is supplied automatically.

| Task                 | Xarray-style API                       | Functional API                                |
| -------------------- | -------------------------------------- | --------------------------------------------- |
| Plot a field         | `da.xgeo.plot.geo(...)`                | `xgeo.plot.geo(da, ...)`                      |
| Animate a field      | `da.xgeo.plot.animate(...)`            | `xgeo.plot.animate(da, ...)`                  |
| Regrid               | `ds.xgeo.remap(target)`                | `xgeo.remap(ds, target)`                      |
| Apply a mask         | `ds.xgeo.mask(...)`                    | `xgeo.mask(ds, ...)`                          |
| Select a transect    | `da.xgeo.sel_transect(...)`            | `xgeo.sel_transect(da, ...)`                  |
| Convert longitude    | `ds.xgeo.to_lon180()`                  | `xgeo.to_lon180(ds)`                          |
| Add local solar time | `ds.xgeo.add_local_solar_time()`       | `xgeo.add_local_solar_time(ds)`               |
| Write NetCDF         | `ds.xgeo.to_netcdf(...)`               | `xgeo.to_netcdf(ds, ...)`                     |
| Compute trends       | `da.xgeo.calc.trends(...)`             | `climtools.stats.trends(da, ...)`             |
| Correlate fields     | `da.xgeo.calc.corr(other, dim="time")` | `climtools.stats.corr(da, other, dim="time")` |
| Preprocess ERA5      | `ds.xgeo.preprocess.era5()`            | `xgeo.preprocess.era5(ds)`                    |

Plotting and single-field statistics are defined on `DataArray`, while shared geospatial and NetCDF operations are available on both `DataArray` and `Dataset`. For a variable stored in a dataset, use for example:

```python
ds["pr"].xgeo.plot.geo(method="contourf")
```

## Plotting

`da.xgeo.plot.geo(...)` returns a `GeoPlot`, which can be extended with chainable overlays.

```python
(
    t2m.xgeo.plot.geo(
        method="contourf",
        cmap=cmaps.temp_div(),
        levels=21,
        gridlines=True,
    )
    .add.contour(z500, colors="k", clabel=True)
    .add.quiver(u10, v10, subsample=4)
    .add.significance(p_value)
)
```

Available overlays include `contour`, `contourf`, `pcolormesh`, `imshow`, `scatter`, `quiver`, `significance`, and `colorbar`.

Faceting follows Xarray-style dimensions:

```python
monthly.xgeo.plot.geo(
    col="month",
    col_wrap=4,
    method="contourf",
)
```

The plotting accessor also exposes animation and vector/significance helpers:

```python
monthly.xgeo.plot.animate(dim="time", outfile="animation.mp4")
u10.xgeo.plot.quiver(v10, subsample=4)
p_value.xgeo.plot.significance(level=0.05)
```

Input coordinates are interpreted in Plate Carrée. The display projection can be supplied explicitly or inferred from the data extent. See [`viz/plotting.py`](viz/plotting.py) for the complete plotting API and [`viz/cmaps.py`](viz/cmaps.py) for the colormap catalog.

## Geospatial and statistical tools

The `.xgeo` accessor keeps common transformations close to the data:

```python
# Regrid a Dataset to another horizontal grid.
regridded = ds.xgeo.remap(target_grid, method="bilinear")

# Convert longitudes to [-180, 180).
wrapped = ds.xgeo.to_lon180()

# Add mean local solar time.
with_lst = ds.xgeo.add_local_solar_time()

# Select a geographic transect.
section = ds["t2m"].xgeo.sel_transect(
    x=-75.0,
    y=35.0,
    orientation=45.0,
    width=2.0,
)

# Pointwise trend statistics.
trend = ds["t2m"].xgeo.calc.trends(dim="time")
```

`climtools.stats` provides the corresponding statistical functions directly, including correlations, pointwise trend estimation, and difference-of-means significance testing.

## Colormaps

`climtools.cmaps` collects scientific palettes from local IPCC color tables, Matplotlib, and cmocean.

```python
from climtools import cmaps

cmap = cmaps.temp_div()
```

## CDO utilities

`climtools.cdo` is a thin Python interface to Climate Data Operators for workflows that mix Xarray analysis with command-line CDO/NCO processing. The required executables must be available on `PATH`.

## MPI-Xarray

For workloads that exceed convenient single-process memory, `climtools.xgeo` provides an MPI-parallel Xarray layer. Each rank owns a non-overlapping partition of the global object while keeping an Xarray-like interface.

```python
import numpy as np

from climtools import mpi, xgeo


dist = xgeo.open_distributed_dataset("data.nc", mpi, partition_dim="time")

logged = np.log(dist["pr"])
rolled = dist.rolling_reduce("time", window=5, reduce="mean")
global_mean = dist.mean(dim="time")

dist.to_netcdf("output.nc", parallel=True)
```

### Constructors

| Function                            | Purpose                                                  |
| ----------------------------------- | -------------------------------------------------------- |
| `open_distributed_dataset(...)`     | Open a NetCDF file with rank-local partitioned reads.    |
| `create_distributed_dataarray(...)` | Construct a distributed `DataArray`, filled per rank.    |
| `create_distributed_dataset(...)`   | Construct a distributed multi-variable `Dataset`.        |
| `distribute_data(...)`              | Partition an already materialized Xarray object.         |
| `empty_distributed_dataset()`       | Placeholder for ranks that hold no data.                 |
| `is_distributed_empty(...)`         | Whether a rank's partition is empty.                     |

### Supported operations

Rank-local NumPy ufuncs, halo-aware rolling and finite-difference operations, collective reductions (`sum`, `mean`, `min`, `max`, `var`, `std`, `prod`, `any`, `all`, `first`, `last`, `median`, `quantile`), grouped and resampled reductions, cumulative scans, interpolation, reindexing and sorting, and collective NetCDF output.

Halo-aware operations exchange only the neighbouring data they need; nothing is gathered globally unless the operation genuinely requires every value at once (order statistics such as `median`). Interpolation fetches only the source points bracketing each rank's targets. For constructors that accept `min_partition_size`, set it at least as wide as the widest halo a downstream operation will request.

Reductions are bitwise reproducible across rank counts when requested: sums use extended fixed-point accumulation, so the same data give the same bits on 4 ranks or 400.

Multi-rank NetCDF output requires the parallel NetCDF/HDF5 stack described in [Installation](#installation).

### Communication layer

All MPI traffic lives in `climtools.mpp`, an adaptation of GFDL's [FMS](https://github.com/NOAA-GFDL/FMS) `mpp`. Modules mirror the FMS source files one-to-one, and anything FMS provides keeps its FMS name:

| Module                   | FMS source                                  | Contents                                             |
| ------------------------ | ------------------------------------------- | ---------------------------------------------------- |
| `mpp.mpp`                | `mpp.F90`                                   | `mpp_sum`/`max`/`min`, `mpp_broadcast`, `mpp_gather`, `mpp_chksum` |
| `mpp.mpp_domains`        | `mpp_domains.F90`                           | `Domain` and the domain types                        |
| `mpp.mpp_domains_define` | `mpp_domains_define.inc`                    | `mpp_define_domains`, `mpp_define_layout`, `mpp_compute_extent` |
| `mpp.mpp_domains_util`   | `mpp_domains_util.inc`                      | `mpp_get_compute_domain`, `mpp_get_data_domain`, …   |
| `mpp.mpp_do_update`      | `mpp_do_update.fh`                          | `mpp_update_domains`, including corners and folds    |
| `mpp.mpp_group_update`   | `mpp_group_update.fh`                       | `mpp_create_group_update`, `mpp_do_group_update`     |
| `mpp.mpp_global_field`   | `mpp_global_field.fh`                       | `mpp_global_field`                                   |
| `mpp.mpp_global_reduce`  | `mpp_global_reduce.fh`, `mpp_global_sum.fh` | `mpp_global_sum`/`max`/`min`                         |
| `mpp.mpp_efp`            | `mpp_efp.F90`                               | `mpp_reproducing_sum`                                |
| `mpp.mpp_data`           | `mpp_data.F90`                              | the reusable halo buffer stack                       |
| `mpp.mpp_parameter`      | `mpp_parameter.F90`                         | update flags, fold edges, grid positions             |

Functionality FMS does not have lives in `ext_*` modules, without the `mpp_` prefix, so the prefix always means "this is FMS":

| Module                | Contents                                                              |
| --------------------- | --------------------------------------------------------------------- |
| `mpp.ext_collectives` | `gather_v`, `scatter_v`, `reduce_scatter`, `partition_offsets`        |
| `mpp.ext_domains`     | Cartesian process grids, `dim_comm`, `slice_compute_domain`           |
| `mpp.ext_efp`         | `reproducing_prod` (FMS has a reproducible sum but no product)        |

`climtools.mpp` works on NumPy arrays and `Domain` objects only and imports nothing from `climtools.xarray`; the dependency runs one way. It is an internal layer, and the constructors above are the intended entry points.

## Storage

`XNpyStore` saves NumPy arrays and Xarray objects as uncompressed `.npy` payloads with a versioned JSON manifest, so they can be memory-mapped back without loading. No pickle data are written.

```python
from climtools import xgeo

store = xgeo.XNpyStore("run_001")
store.save(ds)

store.variables()          # read from the manifest, no payload opened
pr = store.load("pr")      # opens only pr and its coordinates, memory-mapped
```

## Package overview

| Namespace            | Purpose                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `climtools.xgeo`     | Geospatial operations, plotting entry points, preprocessing, NetCDF utilities, storage, and MPI-Xarray constructors. |
| `climtools.plotting` | Cartopy-based geographic plotting implementation.                                                           |
| `climtools.stats`    | Trends, correlations, and significance testing.                                                             |
| `climtools.cmaps`    | Scientific colormap catalog.                                                                                |
| `climtools.cdo`      | CDO/NCO command-line wrapper.                                                                               |
| `climtools.mpi`      | Shared MPI context and communicator.                                                                        |
| `climtools.mpp`      | FMS-derived communication layer (internal).                                                                 |

## Links

* [GitHub repository](https://github.com/Jared-Kodero/climtools)
* [PyPI package](https://pypi.org/project/climtools/)
* [Environment specification](env/environment.yml)
* [Plotting source](viz/plotting.py)
* [Xarray accessors](xarray/accessors.py)
* [MPI-Xarray tests](test/mpi_test.py)
* [Communication layer](mpp/)
* [Benchmarks](test/benchmark.py)
* [License](LICENSE)

## License

Distributed under the [MIT License](LICENSE).
