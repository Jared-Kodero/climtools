from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Assuming your cartplot() and animate() functions are imported from the same module
from .plot import animate, cartplot  # adjust import path as needed

PathLike = str | Path


class GeoDataArray(xr.DataArray):
    __slots__ = ()
    """
    Extension of xarray.DataArray with Cartopy-based plotting and animation methods.
    """

    def cartplot(
        self,
        # Animation control will be popped from args
        dim: str = "time",
        *,
        indices: tuple | list | np.ndarray = None,
        outfile: PathLike = None,
        quality: Literal["low", "medium", "high"] = "medium",
        fps: int = 10,
        parallel: bool = True,
        # Spatial configuration
        x: str = None,
        y: str = None,
        projection: Literal[
            "PlateCarree",
            "Mercator",
            "Robinson",
            "Mollweide",
            "Orthographic",
            "LambertConformal",
            "AlbersEqualArea",
            "Stereographic",
            "NorthPolarStereo",
            "SouthPolarStereo",
        ] = "PlateCarree",
        global_extent: bool = False,
        figsize: tuple[float, float] = None,
        central_longitude: float = None,
        central_latitude: float = None,
        # Plot appearance
        plot_type: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow"
        ] = "default",
        cmap: str | mcolors.Colormap = None,
        vmin: float = None,
        vmax: float = None,
        levels: int | list = None,
        robust: bool = False,
        transform: bool = None,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        add_colorbar: bool = True,
        drawedges: bool = False,
        cbar_label: str = None,
        # Map features
        gridlines: bool = False,
        coastlines: bool = True,
        borders: bool = True,
        states: bool = True,
        ocean: bool = True,
        land: bool = True,
        facecolor: str = "#d3d3d3",
        edgecolor: str = "face",
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes, plt.Artist]:
        """
        Plot this DataArray on a Cartopy map using the global `cartplot()` function.
        """
        return cartplot(
            self,
            dim=dim,
            indices=indices,
            outfile=outfile,
            quality=quality,
            fps=fps,
            parallel=parallel,
            x=x,
            y=y,
            projection=projection,
            global_extent=global_extent,
            figsize=figsize,
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            plot_type=plot_type,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            robust=robust,
            transform=transform,
            orientation=orientation,
            add_colorbar=add_colorbar,
            drawedges=drawedges,
            cbar_label=cbar_label,
            gridlines=gridlines,
            coastlines=coastlines,
            borders=borders,
            states=states,
            ocean=ocean,
            land=land,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs,
        )

    def animate(
        self,
        dim: str = "time",
        *,
        indices: tuple | list | np.ndarray = None,
        outfile: PathLike = None,
        quality: Literal["low", "medium", "high"] = "medium",
        fps: int = 10,
        parallel: bool = True,
        # Spatial configuration
        x: str = None,
        y: str = None,
        projection: Literal[
            "PlateCarree",
            "Mercator",
            "Robinson",
            "Mollweide",
            "Orthographic",
            "LambertConformal",
            "AlbersEqualArea",
            "Stereographic",
            "NorthPolarStereo",
            "SouthPolarStereo",
        ] = "PlateCarree",
        global_extent: bool = False,
        figsize: tuple[float, float] = None,
        central_longitude: float = None,
        central_latitude: float = None,
        # Plot appearance
        plot_type: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow"
        ] = "default",
        cmap: str | mcolors.Colormap = None,
        vmin: float = None,
        vmax: float = None,
        levels: int | list = None,
        robust: bool = False,
        transform: bool = None,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        add_colorbar: bool = True,
        drawedges: bool = False,
        cbar_label: str = None,
        # Map features
        gridlines: bool = False,
        coastlines: bool = True,
        borders: bool = True,
        states: bool = True,
        ocean: bool = True,
        land: bool = True,
        facecolor: str = "#d3d3d3",
        edgecolor: str = "face",
        **kwargs,
    ) -> None:
        """
        Animate this DataArray on a Cartopy map using the global `animate()` function.
        """
        return animate(
            self,
            dim=dim,
            indices=indices,
            outfile=outfile,
            quality=quality,
            fps=fps,
            parallel=parallel,
            x=x,
            y=y,
            projection=projection,
            global_extent=global_extent,
            figsize=figsize,
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            plot_type=plot_type,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            robust=robust,
            transform=transform,
            orientation=orientation,
            add_colorbar=add_colorbar,
            drawedges=drawedges,
            cbar_label=cbar_label,
            gridlines=gridlines,
            coastlines=coastlines,
            borders=borders,
            states=states,
            ocean=ocean,
            land=land,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs,
        )


# Alias for convenience


@xr.register_dataarray_accessor("cartplot")
class CartPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, *args, **kwargs):
        return cartplot(self._obj, *args, **kwargs)

    def animate(self, *args, **kwargs):
        return animate(self._obj, *args, **kwargs)
