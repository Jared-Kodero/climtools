# climtools

<p align="center">
  <strong>Climate-data analysis, geospatial processing, and visualization built around Xarray.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/climtools/"><img src="https://img.shields.io/pypi/v/climtools?label=PyPI" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-%E2%89%A53.12-blue" alt="Python 3.12+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://github.com/Jared-Kodero/climtools"><img src="https://img.shields.io/badge/GitHub-climtools-181717?logo=github" alt="GitHub repository"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#plotting">Plotting</a> ·
  <a href="#geospatial-tools">Geospatial tools</a> ·
  <a href="#statistics-and-colormaps">Statistics & colormaps</a> ·
  <a href="#mpi-xarray">MPI-Xarray</a> ·
  <a href="#testing">Testing</a>
</p>

`climtools` is a compact toolkit for climate and geoscience workflows. It provides Cartopy-based plotting, geospatial operations, statistical analysis, scientific colormaps, CDO integration, NetCDF utilities, and optional MPI-parallel Xarray workflows for larger datasets.

## Highlights

- **Geospatial analysis**: regridding, masking, transects, local solar time, and NetCDF output.
- **Climate visualization**: Cartopy maps, faceting, overlays, vector fields, significance masks, and map features.
- **Statistics**: trends, correlations, and difference-of-means testing.
- **Scientific colormaps**: local IPCC tables, Matplotlib, and cmocean palettes.
- **CDO utilities**: a thin Python interface to Climate Data Operators.
- **Distributed Xarray**: MPI-based partitioning, halo-aware operations, reductions, redistribution, and parallel NetCDF output.

## Installation

Install from PyPI:

```bash
pip install climtools
```

Install optional regridding support:

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

Some features use external command-line tools:

- `climtools.cdo`: `cdo` / `nco`
- `xgeo.plot.animate`: `ffmpeg`

See [`env/environment.yml`](env/environment.yml) for the conda-forge environment used by the project.

## Plotting

Geographic plotting is available through `climtools.xgeo.plot`.

```python
from climtools import cmaps
from climtools import xgeo as xg

plot = xg.plot.geo(
    t2m,
    method="contourf",
    cmap=cmaps.temp_div(),
    levels=21,
    gridlines=True,
)
```

`GeoPlot` supports chainable overlays, making it straightforward to combine scalar fields, contours, vectors, and significance masks.

```python
(
    xg.plot.geo(
        t2m,
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
xg.plot.geo(monthly, col="month", col_wrap=4, method="contourf")
```

Input coordinates are interpreted in Plate Carrée. The display projection can be supplied explicitly or inferred from the data extent.

See [`viz/plotting.py`](viz/plotting.py) for the plotting implementation and [`viz/cmaps.py`](viz/cmaps.py) for the colormap catalog.

## Geospatial tools

`climtools.xgeo` collects geospatial utilities used in climate-analysis workflows, including:

- regridding
- masking
- transects
- local solar time
- NetCDF output

Regridding imports `xesmf` only when required, so the rest of the package can be used without the optional regridding dependency.

## Statistics and colormaps

`climtools.stats` provides common statistical operations for geophysical data, including trends, correlations, and difference-of-means testing.

`climtools.cmaps` provides scientific colormaps from local IPCC tables, Matplotlib, and cmocean.

## CDO utilities

`climtools.cdo` provides a thin Python wrapper around the CDO command-line tool for workflows that combine Python analysis with Climate Data Operators.

The `cdo` and `nco` executables must be available on `PATH`.

## Package overview

| Namespace | Purpose | Source |
| --- | --- | --- |
| `climtools.xgeo` | Geospatial operations, plotting access, NetCDF output, and MPI-Xarray entry points | [`core/xgeo.py`](core/xgeo.py), [`xarray/io.py`](xarray/io.py) |
| `climtools.plotting` | Cartopy-based geographic plotting | [`viz/plotting.py`](viz/plotting.py) |
| `climtools.stats` | Statistical analysis | [`core/stats.py`](core/stats.py) |
| `climtools.cmaps` | Scientific colormaps | [`viz/cmaps.py`](viz/cmaps.py) |
| `climtools.cdo` | CDO command-line wrapper | [`cdo/pycdo.py`](cdo/pycdo.py) |
| `climtools.mpi` | Shared MPI context and communicator | [`mpi/context.py`](mpi/context.py) |

## MPI-Xarray

For distributed workloads, `climtools.xgeo` exposes an MPI-parallel layer for Xarray `Dataset` and `DataArray` objects. Each rank owns a non-overlapping portion of the global object and can use mostly ordinary Xarray-style operations.

```python
import numpy as np

from climtools import mpi, xgeo

# Each rank reads only its own time slice.
dist = xgeo.mpi_open_dataset("data.nc", mpi, partition_dim="time")

# Rank-local operations.
logged = np.log(dist["pr"])
rolled = dist.rolling_reduce("time", window=5, reduce="mean")

# Cross-rank reduction.
global_mean = dist.mean(dim="time")
```


### Creating distributed objects

| Function | Purpose |
| --- | --- |
| `mpi_open_dataset(...)` | Open a NetCDF file and load only each rank's assigned slice. |
| `mpi_partition_data(...)` | Partition an already materialized Xarray object. |
| `mpi_create_dataarray(...)` | Construct a distributed `DataArray` without creating the full global array. |
| `mpi_create_dataset(...)` | Construct a distributed multi-variable `Dataset`. |

### Partitioning

Partitions can span one dimension or multiple dimensions using an MPI Cartesian topology. Ranks receive balanced, non-overlapping slices whose union covers the global domain exactly once.

Distributed objects retain partition metadata including local bounds, global sizes, partition dimensions, and Cartesian-grid information where applicable.

### Distributed operations

| Category | Examples | Communication |
| --- | --- | --- |
| Rank-local | NumPy ufuncs, `where`, `evaluate`, `apply` | None when local |
| Halo-aware | `rolling_reduce`, `coarsen_reduce`, `diff`, `shift`, `differentiate`, `roll`, bounded `ffill` / `bfill` | Neighbor exchange |
| Reductions | `mean`, `sum`, `min`, `max`, `var`, `std`, `prod`, `median`, `any`, `all`, `first`, `last` | Collective communication |
| Grouped reductions | `groupby(...).mean/sum/min/max/count` | Cross-rank reduction |
| Scans | `cumsum`, unbounded `ffill` / `bfill` | Dimension-scoped propagation |
| Redistribution | `sortby`, `reindex`, `align`, `repartition` | Point-to-point redistribution |
| Interpolation | `interp` | Dimension-scoped `Allgather` |
| Parallel output | `to_netcdf(..., parallel=True)` | Collective HDF5 write |

Halo-aware operations exchange only the neighboring data required by the operation. Neighbor topology is cached per communicator and partition layout, while halo values are exchanged fresh for each operation.

### Performance

MPI-Xarray is designed for workloads where distributed memory or sufficiently large computations justify MPI communication and metadata overhead. Most operations keep memory proportional to each rank's local partition plus bounded halo data rather than reconstructing the full global array.

## Testing

Run the MPI-Xarray correctness suite:

```bash
mpirun --oversubscribe -n 4 python test/mpi_test.py
```

The suite reports `[PASS]`, `[FAIL]`, and `[SKIP]` per check and exits nonzero on failure. On SLURM systems, [`test/test.sh`](test/test.sh) provides the corresponding cluster entry point.

Run the benchmark suite:

```bash
mpirun --oversubscribe -n 4 python test/benchmark.py --size 2000000 --reps 5
```

Benchmark results are printed as Markdown and written to `benchmark_results_n<ranks>.json`.

## Links

- [GitHub repository](https://github.com/Jared-Kodero/climtools)
- [PyPI package](https://pypi.org/project/climtools/)
- [Environment specification](env/environment.yml)
- [Plotting source](viz/plotting.py)
- [Colormaps](viz/cmaps.py)
- [MPI-Xarray tests](test/mpi_test.py)
- [Benchmarks](test/benchmark.py)
- [License](LICENSE)

## License

Distributed under the [MIT License](LICENSE).
