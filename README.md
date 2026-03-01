# climtools

climtools — Utilities for climate data analysis and plotting
-----------------------------------------------------------

climtools is a small collection of convenience utilities to speed up exploratory
and reproducible climate-data analysis. It wraps common operations on xarray
objects, integrates with CDO (where available), provides plotting helpers for
CartoPy, and contains a handful of file/system utilities and plotting themes.

This repository is intended as a local developer package. To use it, clone the
repo and import the package from the repository root or install it into your
environment.

Quick start
-----------

Clone the repository and import the package in your Python session:

```bash
git clone <repo-url>
cd climtools
# from the repo root you can import the package in Python:
python -c "import climtools; print(climtools.__name__)"
```

Example usage
-------------

```python
import numpy as np
import pandas as pd
import xarray as xr
from climtools import mapplot, calc_trends, calc_corr

# simple plotting example
da = xr.DataArray(
    np.random.rand(36, 180, 360),
    dims=("time", "lat", "lon"),
    coords={"time": pd.date_range("2000-01-01", periods=36, freq="M"),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(-180, 180, 360)}
)

mapplot(da.isel(time=0), projection="PlateCarree", figsize=(8,4))

# compute trends
trends = calc_trends(da)
```

Main features and available helpers
-----------------------------------

The package exposes several modules and convenience functions. At the time of
writing the top-level exports include (non-exhaustive):

- Plotting & theming: `mapplot`, `animate`, `plot_pvalues`, `plot_cmaps`, `theme`, `IPCCTheme`
- Trends & stats: `calc_trends`, `calc_signicance`, `mk_trend_test`, `polyfit`, `calc_corr`
- Regridding / CDO: helpers in `cdo_py.py` and `regridder.py` (e.g. `cdo`, `regrid_cam_se`)
- Utilities: `tools.py` functions such as `interp_data`, `land_sea_mask`, `get_local_solar_time`, `tz_apply_func`, `setup_dask`, `close_dask`, `timeit`
- IO & system: `cp`, `mv`, `rm`, `mkdir`, `file_type`, `symlink`, `tmp` (file helpers)
- Colormaps: `gen_cmap_file`, `ColorMaps`, `cm`, `cmaps`
- Data wrappers: `GeoDataArray` (light xarray plotting wrapper)

See the module docstrings in each file for details on individual function
signatures and behaviour. The top-level package exports are kept in
`climtools/__init__.py` and can be inspected programmatically with
`help(climtools)` or `dir(climtools)`.

Contributing
------------

If you want to contribute, please open an issue with a short description and a
reproducible example. Small, focused tests are welcome.

Acknowledgements
----------------

This project is a personal collection of utilities and is not published to PyPI.
