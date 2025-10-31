from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

from .plot import animate, cartplot, make_cyclic
from .trends import calc_trends, polyfit


class GeoDataArray(xr.DataArray):
    """
    Extension of xarray.DataArray with Cartopy-based plotting and animation methods.
    """

    __slots__ = ()

    def add_cyclic_point(self, dim: str = "lon") -> GeoDataArray:
        return make_cyclic(self, dim)

    def cartplot(
        self,
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
        central_longitude: float = None,
        central_latitude: float = None,
        global_extent: bool = False,
        figsize: tuple[float, float] = None,
        # Plot appearance
        plot_type: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow"
        ] = "default",
        cmap: str | mcolors.Colormap = None,
        vmin: float = None,
        vmax: float = None,
        levels: int | list = None,
        extend: str = None,
        robust: bool = False,
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
        edgecolor: str = "face",
        **kwargs,
    ):
        """
        Plot this DataArray on a Cartopy map using the global `cartplot()` function.
        """
        return cartplot(
            self,
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
            extend=extend,
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
            edgecolor=edgecolor,
            **kwargs,
        )

    def animate(
        self,
        dim: str = "time",
        *,
        indices: tuple | list | np.ndarray = None,
        outfile: Path | str = None,
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
        extend: str = None,
        robust: bool = False,
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
            extend=extend,
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
            edgecolor=edgecolor,
            **kwargs,
        )

    def trends(
        self,
        along: str = None,
        *,
        scale: float = 1,
        use_dask: bool = True,
        dask_scheduler: Literal["threads", "processes"] = "threads",
    ) -> xr.Dataset:
        """
        Calculate trends along a specified dimension using the Mann-Kendall trend test.
        """
        return calc_trends(
            self,
            along=along,
            scale=scale,
            use_dask=use_dask,
            dask_scheduler=dask_scheduler,
        )

    def trendfit(self, along: str, data_var: str = None, scale: float = 1):
        """
        Calculate the linear trend for the given xarray Dataset or DataArray using polynomial fitting.
        """
        return polyfit(
            self,
            along=along,
            data_var=data_var,
            scale=scale,
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
