import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from .plot import animate as animate, cartplot as cartplot
from pathlib import Path
from typing import Literal

PathLike = str | Path

class GeoDataArray(xr.DataArray):
    def cartplot(self, dim: str = 'time', *, indices: tuple | list | np.ndarray = None, outfile: PathLike = None, quality: Literal['low', 'medium', 'high'] = 'medium', fps: int = 10, parallel: bool = True, x: str = None, y: str = None, projection: Literal['PlateCarree', 'Mercator', 'Robinson', 'Mollweide', 'Orthographic', 'LambertConformal', 'AlbersEqualArea', 'Stereographic', 'NorthPolarStereo', 'SouthPolarStereo'] = 'PlateCarree', global_extent: bool = False, figsize: tuple[float, float] = None, central_longitude: float = None, central_latitude: float = None, plot_type: Literal['default', 'pcolormesh', 'contourf', 'contour', 'imshow'] = 'default', cmap: str | mcolors.Colormap = None, vmin: float = None, vmax: float = None, levels: int | list = None, robust: bool = False, transform: bool = None, orientation: Literal['vertical', 'horizontal'] = 'vertical', add_colorbar: bool = True, drawedges: bool = False, cbar_label: str = None, gridlines: bool = False, coastlines: bool = True, borders: bool = True, states: bool = True, ocean: bool = True, land: bool = True, facecolor: str = '#d3d3d3', edgecolor: str = 'face', **kwargs) -> tuple[plt.Figure, plt.Axes, plt.Artist]: ...
    def animate(self, dim: str = 'time', *, indices: tuple | list | np.ndarray = None, outfile: PathLike = None, quality: Literal['low', 'medium', 'high'] = 'medium', fps: int = 10, parallel: bool = True, x: str = None, y: str = None, projection: Literal['PlateCarree', 'Mercator', 'Robinson', 'Mollweide', 'Orthographic', 'LambertConformal', 'AlbersEqualArea', 'Stereographic', 'NorthPolarStereo', 'SouthPolarStereo'] = 'PlateCarree', global_extent: bool = False, figsize: tuple[float, float] = None, central_longitude: float = None, central_latitude: float = None, plot_type: Literal['default', 'pcolormesh', 'contourf', 'contour', 'imshow'] = 'default', cmap: str | mcolors.Colormap = None, vmin: float = None, vmax: float = None, levels: int | list = None, robust: bool = False, transform: bool = None, orientation: Literal['vertical', 'horizontal'] = 'vertical', add_colorbar: bool = True, drawedges: bool = False, cbar_label: str = None, gridlines: bool = False, coastlines: bool = True, borders: bool = True, states: bool = True, ocean: bool = True, land: bool = True, facecolor: str = '#d3d3d3', edgecolor: str = 'face', **kwargs) -> None: ...

class CartPlotAccessor:
    def __init__(self, xarray_obj) -> None: ...
    def plot(self, *args, **kwargs): ...
    def animate(self, *args, **kwargs): ...
