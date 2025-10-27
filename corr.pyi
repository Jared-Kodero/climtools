import numpy as np
import pandas as pd
import xarray as xr
from typing import Literal

def corr_test(array_x: np.ndarray, array_y: np.ndarray, corr_type: str, alternative: str = 'two-sided', data_type: str = None, **coords) -> np.ndarray: ...
def calc_corr(x: pd.DataFrame | xr.DataArray | xr.Dataset, y: pd.DataFrame | xr.DataArray | xr.Dataset, *, x_var: str = None, y_var: str = None, corr_type: Literal['pearson', 'spearman', 'kendall'] = 'pearson', alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided', along: str = None, groupby: str | list[str] | None = None, use_dask: bool = True, dask_scheduler: Literal['threads', 'processes'] = 'threads') -> pd.DataFrame | xr.Dataset: ...
