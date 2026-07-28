"""Stateful Cartopy plotting classes and public plotting entry points.

``GeoPlot`` is the primary container. It receives user arguments, normalizes the
inputs, creates the figure and axes, delegates drawing to the stateless
``plot_*`` functions in :mod:`plot_utils`, and stores every resulting primitive.
``FacetedPlot`` owns facet layout and iteration, ``Adder`` adds reusable layers,
and ``Animate`` renders a sequence of ``GeoPlot`` objects to an MP4 file.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cartopy.crs as ccrs
import cartopy.mpl.geoaxes as cgeo
import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.quiver import Quiver, QuiverKey
from matplotlib.text import Text

from .plot_utils import (
    add_colorbar as _add_colorbar,
)
from .plot_utils import (
    add_contour_labels,
    add_cyclic_point,
    add_grid_boundary,
    add_gridlines,
    add_map_features,
    get_facet_figsize,
    get_projection,
    get_quiver_key_mag,
    norm_input,
    norm_levels,
    plot_contour,
    plot_contourf,
    plot_default,
    plot_imshow,
    plot_pcolormesh,
    plot_quiver,
    plot_scatter,
    plot_significance,
    select_facet,
    to_lon180,
    validate_animation_inputs,
    validate_data,
    validate_vector_components,
)
from .progress import DaskProgressBar, SerialProgressBar
from .tools import n_cpus, tmp

__all__ = [
    "Adder",
    "Animate",
    "FacetedPlot",
    "GeoPlot",
    "animate",
    "create_figure",
    "geo",
]

AxesType = Axes | cgeo.GeoAxes
ScalarPrimitive = (
    Artist | ScalarMappable | QuadMesh | QuadContourSet | AxesImage | PathCollection
)


def colorbar(
    fig: Figure,
    ax: AxesType | np.ndarray,
    mappable: ScalarMappable,
    *,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    subplots: bool = False,
    adjust: bool = True,
    cax: Axes | None = None,
    pad_bottom: bool | None = None,
    drawedges: bool = False,
    extend: Literal["neither", "both", "min", "max"] | None = None,
    label: str | None = None,
    ticks: Sequence[float] | np.ndarray | None = None,
    tick_labels: Sequence[str] | None = None,
) -> Colorbar:
    """Add a colorbar for a scalar plotting primitive.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Parent figure.
    ax : matplotlib.axes.Axes or numpy.ndarray
        Axis or axes associated with ``mappable``.
    mappable : matplotlib.cm.ScalarMappable
        Primitive described by the colorbar.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Colorbar orientation.
    subplots : bool, default False
        Position the colorbar relative to a facet grid.
    adjust : bool, default True
        Apply tight layout before creating the colorbar axis.
    cax : matplotlib.axes.Axes, optional
        Existing colorbar axis.
    pad_bottom : bool, optional
        Force additional space below a horizontal colorbar. When omitted, infer
        the requirement from the target axis labels.
    drawedges : bool, default False
        Draw edges between color intervals.
    extend : {"neither", "both", "min", "max"}, optional
        Out-of-range extension behavior.
    label : str, optional
        Colorbar label.
    ticks : sequence of float, optional
        Explicit tick positions.
    tick_labels : sequence of str, optional
        Explicit tick labels.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        Created colorbar.
    """

    return _add_colorbar(**dict(locals()))


def create_figure(
    *,
    projection: ccrs.Projection,
    figsize: tuple[float, float] | None = None,
    nrows: int = 1,
    ncols: int = 1,
    squeeze: bool = False,
) -> tuple[Figure, np.ndarray]:
    """Create a Cartopy figure and an array of geographic axes.

    Parameters
    ----------
    projection : cartopy.crs.Projection
        Projection assigned to every subplot.
    figsize : tuple of float, optional
        Figure size in inches.
    nrows, ncols : int, default 1
        Subplot-grid dimensions.
    squeeze : bool, default False
        Forwarded to :func:`matplotlib.pyplot.subplots`. ``False`` is preferred
        because it guarantees a two-dimensional axis array.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Created figure.
    axes : numpy.ndarray
        Array containing the created Cartopy axes.
    """
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=squeeze,
        subplot_kw={"projection": projection},
    )
    return figure, np.asarray(axes, dtype=object)


def _plot_scalar(
    data: xr.DataArray,
    fig: Figure,
    ax: AxesType,
    *,
    method: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
    ] = "default",
    x: str,
    y: str,
    cmap: str | Colormap | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int | Sequence[float] | np.ndarray | None = None,
    extend: Literal["neither", "both", "min", "max"] | None = None,
    robust: bool = False,
    rasterized: bool = False,
    zorder: float = 1.0,
    add_labels: bool = False,
    **kwargs: Any,
) -> ScalarPrimitive:
    """Dispatch one scalar layer to an explicit plotting primitive.

    Parameters
    ----------
    data : xarray.DataArray
        Two-dimensional field.
    fig : matplotlib.figure.Figure
        Parent figure.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Destination axis.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}
        Plotting method.
    x, y : str
        Horizontal coordinate names.
    cmap, norm, vmin, vmax : optional
        Scalar-color mapping parameters.
    levels : int or sequence of float, optional
        Contour levels.
    extend : {"neither", "both", "min", "max"}, optional
        Out-of-range contour behavior.
    robust : bool, default False
        Use percentile-based limits where supported.
    rasterized : bool, default False
        Rasterize dense primitives where supported.
    zorder : float, default 1
        Drawing order.
    add_labels : bool, default False
        Let xarray add labels.
    **kwargs
        Additional method-specific keyword arguments.

    Returns
    -------
    matplotlib primitive
        Primitive returned by the selected ``plot_*`` function.
    """
    if method == "default":
        return plot_default(
            data,
            fig,
            ax,
            x=x,
            y=y,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
            rasterized=rasterized,
            zorder=zorder,
            add_labels=add_labels,
            **kwargs,
        )
    if method == "pcolormesh":
        return plot_pcolormesh(
            data,
            fig,
            ax,
            x=x,
            y=y,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
            rasterized=rasterized,
            zorder=zorder,
            add_labels=add_labels,
            **kwargs,
        )
    if method == "contourf":
        return plot_contourf(
            data,
            fig,
            ax,
            x=x,
            y=y,
            levels=levels,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            extend=extend,
            robust=robust,
            rasterized=rasterized,
            zorder=zorder,
            add_labels=add_labels,
            **kwargs,
        )
    if method == "contour":
        return plot_contour(
            data,
            fig,
            ax,
            x=x,
            y=y,
            levels=levels,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            extend=extend,
            rasterized=rasterized,
            zorder=zorder,
            add_labels=add_labels,
            **kwargs,
        )
    if method == "imshow":
        return plot_imshow(
            data,
            fig,
            ax,
            x=x,
            y=y,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
            rasterized=rasterized,
            zorder=zorder,
            add_labels=add_labels,
            **kwargs,
        )
    if method == "scatter":
        return plot_scatter(
            data,
            fig,
            ax,
            x=x,
            y=y,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            zorder=zorder,
            add_labels=add_labels,
            **kwargs,
        )
    raise ValueError(f"unsupported plot method {method!r}")


class FacetedPlot:
    """Create and manage a faceted Cartopy layout.

    Parameters
    ----------
    data : xarray.DataArray
        Normalized scalar field containing one or two facet dimensions.
    x, y : str
        Horizontal coordinate names.
    col, row : str, optional
        Column and row facet dimensions.
    col_wrap : int, optional
        Maximum number of columns when only ``col`` is supplied.
    projection : cartopy.crs.Projection
        Projection assigned to every panel.
    figsize : tuple of float, optional
        Figure size in inches. A domain-aware size is inferred when omitted.
    global_extent : bool, default False
        Use a global map extent.
    set_extent : tuple of float, optional
        Explicit geographic extent.
    gridlines : bool, default False
        Draw labeled gridlines.
    add_grid_bounds:
       If True, draw an outline along the outer perimeter of the plotted grid domain.
    coastlines, borders, states : bool, default True
        Add boundary features.
    ocean, land : bool, default True
        Control background fills.
    lakes, rivers : bool, default False
        Add inland-water features.

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        Facet figure.
    axes : numpy.ndarray
        Two-dimensional axis array.
    axis_selectors : list of tuple
        Populated axes paired with xarray selectors.
    artists : list
        Scalar primitives returned by :meth:`render`.
    contour_labels : list of list of matplotlib.text.Text
        Contour labels created for populated panels.
    """

    def __init__(
        self,
        data: xr.DataArray,
        *,
        x: str,
        y: str,
        col: str | None,
        row: str | None,
        col_wrap: int | None,
        projection: ccrs.Projection,
        figsize: tuple[float, float] | None,
        global_extent: bool = False,
        set_extent: tuple[float, float, float, float] | None = None,
        gridlines: bool = False,
        add_grid_bounds: bool = False,
        coastlines: bool = True,
        borders: bool = True,
        states: bool = True,
        ocean: bool = True,
        land: bool = True,
        lakes: bool = False,
        rivers: bool = False,
    ) -> None:
        self.data = data
        self.x = x
        self.y = y
        self.col = col
        self.row = row
        self.col_wrap = col_wrap
        self.nrows, self.ncols, selectors = self._layout()
        if figsize is None:
            figsize = get_facet_figsize(
                data=data,
                x=x,
                y=y,
                nrows=self.nrows,
                ncols=self.ncols,
            )
        self.figure, self.axes = create_figure(
            projection=projection,
            figsize=figsize,
            nrows=self.nrows,
            ncols=self.ncols,
            squeeze=False,
        )
        self.axis_selectors: list[tuple[AxesType, dict[str, Any]]] = []
        self.artists: list[ScalarPrimitive] = []
        self.contour_labels: list[list[Text]] = []
        self.map_features: list[Artist] = []
        self.gridliners: list[Any] = []
        axes_flat = list(self.axes.flat)
        for index, axis in enumerate(axes_flat):
            if index >= len(selectors):
                axis.set_visible(False)
                continue
            selector = selectors[index]
            self.map_features.extend(
                add_map_features(
                    self.figure,
                    axis,
                    global_extent=global_extent,
                    set_extent=set_extent,
                    coastlines=coastlines,
                    states=states,
                    borders=borders,
                    lakes=lakes,
                    rivers=rivers,
                    ocean=ocean,
                    land=land,
                )
            )
            if gridlines:
                self.gridliners.append(add_gridlines(self.figure, axis))
            if add_grid_bounds:
                add_grid_boundary(
                    axis,
                    data[self.x].values,
                    data[self.y].values,
                    transform=ccrs.PlateCarree(),
                    linewidth=1,
                    zorder=1,
                )
            axis.set_title(self._selector_title(selector))
            self.axis_selectors.append((axis, selector))

    def _layout(self) -> tuple[int, int, list[dict[str, Any]]]:
        """Resolve grid shape and selectors for all populated panels."""
        if self.row is not None and self.col is not None:
            row_values = list(self.data[self.row].values)
            col_values = list(self.data[self.col].values)
            selectors = [
                {self.row: row_value, self.col: col_value}
                for row_value in row_values
                for col_value in col_values
            ]
            return len(row_values), len(col_values), selectors
        if self.row is not None:
            row_values = list(self.data[self.row].values)
            selectors = [{self.row: value} for value in row_values]
            return len(row_values), 1, selectors
        if self.col is not None:
            col_values = list(self.data[self.col].values)
            ncols = self.col_wrap or int(np.ceil(np.sqrt(len(col_values))))
            nrows = int(np.ceil(len(col_values) / ncols))
            selectors = [{self.col: value} for value in col_values]
            return nrows, ncols, selectors
        raise ValueError("FacetedPlot requires col or row")

    @staticmethod
    def _selector_title(selector: Mapping[str, Any]) -> str:
        """Format a compact panel title from a facet selector."""
        values: list[str] = []
        for name, value in selector.items():
            if np.issubdtype(np.asarray(value).dtype, np.datetime64):
                value = pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
            values.append(f"{name} = {value}")
        return ", ".join(values)

    def iter_axes(self) -> Iterator[tuple[AxesType, dict[str, Any]]]:
        """Yield populated axes and their xarray selectors.

        Yields
        ------
        axis : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
            Populated map axis.
        selector : dict
            Coordinate selector for the panel.
        """
        yield from self.axis_selectors

    def bottom_left_axis(self) -> AxesType:
        """Return the lowest populated axis in the leftmost occupied column.

        Returns
        -------
        matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
            Axis used for a shared quiver key.
        """
        positions = [(axis.get_position(), axis) for axis, _ in self.axis_selectors]
        left = min(position.x0 for position, _ in positions)
        candidates = [
            (position.y0, axis)
            for position, axis in positions
            if np.isclose(position.x0, left)
        ]
        return min(candidates, key=lambda item: item[0])[1]

    def render(
        self,
        *,
        method: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
        ] = "default",
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        levels: int | Sequence[float] | np.ndarray | None = None,
        extend: Literal["neither", "both", "min", "max"] | None = None,
        robust: bool = False,
        rasterized: bool = False,
        clabel: bool = False,
        clabel_fmt: str | Mapping[float, str] = "%1.0f",
        clabel_fontsize: float = 8.0,
        clabel_inline: bool = True,
        clabel_colors: str | Sequence[str] | None = None,
        clabel_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ScalarPrimitive]:
        """Render the scalar field on every populated facet.

        Parameters
        ----------
        method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}
            Scalar plot type.
        cmap, norm, vmin, vmax : optional
            Scalar-color mapping parameters.
        levels : int or sequence of float, optional
            Contour levels.
        extend : {"neither", "both", "min", "max"}, optional
            Out-of-range contour behavior.
        robust : bool, default False
            Use percentile-based color limits where supported.
        rasterized : bool, default False
            Rasterize dense primitives where supported.
        clabel : bool, default False
            Label line contours.
        clabel_fmt : str or mapping, default "%1.0f"
            Contour-label format.
        clabel_fontsize : float, default 8
            Contour-label font size.
        clabel_inline : bool, default True
            Draw labels inline.
        clabel_colors : str or sequence of str, optional
            Contour-label colors.
        clabel_kwargs : mapping, optional
            Additional arguments forwarded to :func:`add_contour_labels`.
        **kwargs
            Additional method-specific plotting arguments.

        Returns
        -------
        list
            One scalar primitive per populated panel.
        """
        self.artists.clear()
        self.contour_labels.clear()
        for axis, selector in self.axis_selectors:
            field = select_facet(self.data, selector)
            artist = _plot_scalar(
                field,
                self.figure,
                axis,
                method=method,
                x=self.x,
                y=self.y,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                levels=levels,
                extend=extend,
                robust=robust,
                rasterized=rasterized,
                **kwargs,
            )

            self.artists.append(artist)
            if clabel and method == "contour" and isinstance(artist, QuadContourSet):
                self.contour_labels.append(
                    add_contour_labels(
                        self.figure,
                        axis,
                        artist,
                        fmt=clabel_fmt,
                        fontsize=clabel_fontsize,
                        inline=clabel_inline,
                        colors=clabel_colors,
                        kwargs=clabel_kwargs,
                    )
                )
        return self.artists


class GeoPlot:
    """Container and controller for a scalar Cartopy map.

    ``GeoPlot`` owns normalized input data, figure construction, scalar
    rendering, optional overlays, and all returned Matplotlib primitives.
    Single-axis and faceted plots use the same stateless ``plot_*`` functions.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field to plot.
    x, y : str, optional
        Horizontal coordinate names. They are inferred when omitted.
    col, row : str, optional
        Facet dimensions.
    col_wrap : int, optional
        Maximum number of facet columns when only ``col`` is used.
    figsize : tuple of float, optional
        Figure size in inches.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}
        Base scalar plotting method.
    projection : {"PlateCarree", "Mercator", "Robinson", "Mollweide", "Orthographic", "LambertConformal", "AlbersEqualArea", "Stereographic", "NorthPolarStereo", "SouthPolarStereo"}, optional
        Display projection. A domain-dependent projection is inferred when
        omitted.
    cmap : str or matplotlib.colors.Colormap, optional
        Base colormap.
    norm : matplotlib.colors.Normalize, optional
        Base color normalization.
    vmin, vmax : float, optional
        Base scalar color limits.
    units : str, optional
        Units used in the inferred colorbar label.
    levels : int or sequence of float, optional
        Contour levels.
    extend : {"neither", "both", "min", "max"}, optional
        Colorbar and contour extension behavior.
    robust : bool, default False
        Use percentile-based limits where supported.
    rasterized : bool, default False
        Rasterize dense scalar primitives.
    title : str, default ""
        Axis title or facet-figure title.
    orientation : {"vertical", "horizontal"}, optional
        Base colorbar orientation. Defaults to vertical for a single axis and
        horizontal for facets.
    add_colorbar : bool, default True
        Add a base colorbar for scalar plots other than line contours.
        Line contours use inline contour labels instead.
    drawedges : bool, default False
        Draw colorbar interval edges.
    cbar_label : str, optional
        Explicit base colorbar label.
    global_extent : bool, default False
        Use a global map extent.
    set_extent : tuple of float, optional
        Explicit extent ``(lon_min, lon_max, lat_min, lat_max)``.
    gridlines : bool, default False
        Add labeled gridlines.
    add_grid_bounds:
        If True, draw an outline along the outer perimeter of the plotted grid domain.
    coastlines, borders, states : bool, default True
        Add common boundary features.
    ocean, land : bool, default True
        Control background fills.
    lakes, rivers : bool, default False
        Add inland-water features.
    p_value : xarray.DataArray, optional
        Pointwise p-values added as significance markers.
    pvalue_kwargs : mapping, optional
        Arguments forwarded to :meth:`Adder.significance`.
    u_component, v_component : xarray.DataArray, optional
        Vector components added as a quiver layer.
    quiver_kwargs : mapping, optional
        Arguments forwarded to :meth:`Adder.quiver`.
    colorbar_kwargs : mapping, optional
        Additional base colorbar options.
    clabel : bool, default False
        Label a line-contour base. Line contours are labeled automatically.
    clabel_fmt : str or mapping, default "%1.0f"
        Base contour-label format.
    clabel_fontsize : float, default 8
        Base contour-label font size.
    clabel_inline : bool, default True
        Draw base contour labels inline.
    clabel_colors : str or sequence of str, optional
        Base contour-label colors.
    clabel_kwargs : mapping, optional
        Additional base contour-label arguments.
    cyclic : bool, default False
        Append a cyclic horizontal point before plotting.
    **kwargs
        Additional arguments forwarded to the selected base ``plot_*``
        function.

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        Plot figure.
    axes : matplotlib.axes.Axes or numpy.ndarray
        Single map axis or facet-axis array.
    artist : matplotlib primitive or list
        Base scalar primitive or one primitive per facet.
    colorbar : matplotlib.colorbar.Colorbar or None
        Base or most recently added colorbar.
    quiver : matplotlib.quiver.Quiver, list, or None
        Most recently added vector primitive or primitives.
    quiver_key : matplotlib.quiver.QuiverKey or None
        Shared vector key.
    layers : list of dict
        Registered layers in drawing order.
    add : Adder
        Namespace containing chainable overlay methods.
    """

    def __init__(
        self,
        da: xr.DataArray,
        *,
        x: str | None = None,
        y: str | None = None,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        figsize: tuple[float, float] | None = None,
        method: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
        ] = "default",
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
        ]
        | None = None,
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        units: str | None = None,
        levels: int | Sequence[float] | np.ndarray | None = None,
        extend: Literal["neither", "both", "min", "max"] | None = None,
        robust: bool = False,
        rasterized: bool = False,
        title: str = "",
        orientation: Literal["vertical", "horizontal"] | None = None,
        add_colorbar: bool = True,
        drawedges: bool = False,
        cbar_label: str | None = None,
        global_extent: bool = False,
        set_extent: tuple[float, float, float, float] | None = None,
        gridlines: bool = False,
        add_grid_bounds: bool = False,
        coastlines: bool = True,
        borders: bool = True,
        states: bool = True,
        ocean: bool = True,
        land: bool = True,
        lakes: bool = False,
        rivers: bool = False,
        p_value: xr.DataArray | None = None,
        pvalue_kwargs: Mapping[str, Any] | None = None,
        u_component: xr.DataArray | None = None,
        v_component: xr.DataArray | None = None,
        quiver_kwargs: Mapping[str, Any] | None = None,
        colorbar_kwargs: Mapping[str, Any] | None = None,
        clabel: bool = False,
        clabel_fmt: str | Mapping[float, str] = "%1.0f",
        clabel_fontsize: float = 8.0,
        clabel_inline: bool = True,
        clabel_colors: str | Sequence[str] | None = None,
        clabel_kwargs: Mapping[str, Any] | None = None,
        cyclic: bool = False,
        **kwargs: Any,
    ) -> None:
        self.data, self.x, self.y, self.col, self.row = norm_input(
            da,
            x=x,
            y=y,
            col=col,
            row=row,
            col_wrap=col_wrap,
            cyclic=cyclic,
        )
        self.method = method
        self.cyclic = cyclic
        self.projection_object, self.projection = get_projection(
            projection,
            self.data[self.x],
            self.data[self.y],
        )
        self.layers: list[dict[str, Any]] = []
        self.map_features: list[Artist] = []
        self.gridliners: list[Any] = []
        self.contour_labels: list[Text] | list[list[Text]] = []
        self.colorbar: Colorbar | None = None
        self.quiver: Quiver | list[Quiver] | None = None
        self.quiver_key: QuiverKey | None = None
        self.faceted_plot: FacetedPlot | None = None
        self.grid: xr.Dataset = self.data.coords.to_dataset()[[self.x, self.y]]
        self.add = Adder(self)

        if self.method == "contourf":
            levels = norm_levels(vmin, vmax, levels)

        if self.is_faceted:
            facet = FacetedPlot(
                self.data,
                x=self.x,
                y=self.y,
                col=self.col,
                row=self.row,
                col_wrap=col_wrap,
                projection=self.projection_object,
                figsize=figsize,
                global_extent=global_extent,
                set_extent=set_extent,
                gridlines=gridlines,
                add_grid_bounds=add_grid_bounds,
                coastlines=coastlines,
                borders=borders,
                states=states,
                ocean=ocean,
                land=land,
                lakes=lakes,
                rivers=rivers,
            )
            self.faceted_plot = facet
            self.figure = facet.figure
            self.axes: AxesType | np.ndarray = facet.axes
            self.artist: ScalarPrimitive | list[ScalarPrimitive] = facet.render(
                method=method,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                levels=levels,
                extend=extend,
                robust=robust,
                rasterized=rasterized,
                clabel=clabel or method == "contour",
                clabel_fmt=clabel_fmt,
                clabel_fontsize=clabel_fontsize,
                clabel_inline=clabel_inline,
                clabel_colors=clabel_colors,
                clabel_kwargs=clabel_kwargs,
                **kwargs,
            )
            self.contour_labels = facet.contour_labels
            self.map_features = facet.map_features
            self.gridliners = facet.gridliners
            if title:
                self.figure.suptitle(title)
        else:
            self.figure, axes_array = create_figure(
                projection=self.projection_object,
                figsize=figsize,
                nrows=1,
                ncols=1,
                squeeze=False,
            )
            axis = axes_array[0, 0]
            self.axes = axis
            self.map_features = add_map_features(
                self.figure,
                axis,
                global_extent=global_extent,
                set_extent=set_extent,
                coastlines=coastlines,
                states=states,
                borders=borders,
                lakes=lakes,
                rivers=rivers,
                ocean=ocean,
                land=land,
            )
            if gridlines:
                self.gridliners.append(add_gridlines(self.figure, axis))
            self.artist = _plot_scalar(
                self.data,
                self.figure,
                axis,
                method=method,
                x=self.x,
                y=self.y,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                levels=levels,
                extend=extend,
                robust=robust,
                rasterized=rasterized,
                **kwargs,
            )
            if add_grid_bounds:
                add_grid_boundary(
                    axis,
                    self.grid[self.x].values,
                    self.grid[self.y].values,
                    transform=ccrs.PlateCarree(),
                    linewidth=1,
                    zorder=1,
                )

            if method == "contour" and isinstance(self.artist, QuadContourSet):
                self.contour_labels = add_contour_labels(
                    self.figure,
                    axis,
                    self.artist,
                    fmt=clabel_fmt,
                    fontsize=clabel_fontsize,
                    inline=clabel_inline,
                    colors=clabel_colors,
                    kwargs=clabel_kwargs,
                )
            axis.set_title(title)
        self.layers.append(
            {
                "kind": method,
                "artists": self.base_artists,
                "labels": self.contour_labels,
                "base": True,
            }
        )
        if p_value is not None:
            self.add.significance(p_value, **dict(pvalue_kwargs or {}))
        u_component, v_component = validate_vector_components(u_component, v_component)
        if u_component is not None and v_component is not None:
            self.add.quiver(
                u_component,
                v_component,
                **dict(quiver_kwargs or {}),
            )
        if add_colorbar and method != "contour":
            colorbar_options = dict(colorbar_kwargs or {})
            ticks = colorbar_options.pop("ticks", None)
            tick_labels = colorbar_options.pop("tick_labels", None)
            if colorbar_options:
                unexpected = ", ".join(sorted(colorbar_options))
                raise TypeError(f"unsupported colorbar_kwargs: {unexpected}")

            if cbar_label is None:
                long_name = str(self.data.attrs.get("long_name", "")).title()
                inferred_units = units or self.data.attrs.get("units", self.data.name)
                cbar_label = f"{long_name}\n[{inferred_units}]".strip()
            resolved_orientation = orientation or (
                "horizontal" if self.is_faceted else "vertical"
            )
            self.colorbar = _add_colorbar(
                self.figure,
                self.axes,
                self.mappable,
                orientation=resolved_orientation,
                subplots=self.is_faceted,
                adjust=False,
                pad_bottom=True if self.quiver_key is not None else None,
                drawedges=drawedges,
                extend=extend,
                label=cbar_label,
                ticks=ticks,
                tick_labels=tick_labels,
            )
        self.figure.canvas.draw_idle()

    @property
    def is_faceted(self) -> bool:
        """Whether the plot uses a facet layout."""
        return self.col is not None or self.row is not None

    @property
    def base_artists(self) -> list[ScalarPrimitive]:
        """Return base scalar primitives as a list.

        Returns
        -------
        list
            One element for a single-axis plot or one element per facet.
        """
        return self.artist if isinstance(self.artist, list) else [self.artist]

    @property
    def mappable(self) -> ScalarMappable:
        """Return the base primitive used for color mapping.

        Returns
        -------
        matplotlib.cm.ScalarMappable
            Last base scalar primitive.

        Raises
        ------
        TypeError
            If the base primitive cannot drive a colorbar.
        """
        candidate = self.base_artists[-1]
        if not isinstance(candidate, ScalarMappable):
            raise TypeError(
                f"base artist {type(candidate).__name__} is not a ScalarMappable"
            )
        return candidate

    def iter_axes(self) -> Iterator[tuple[AxesType, dict[str, Any]]]:
        """Yield populated axes and facet selectors.

        Yields
        ------
        axis : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
            Destination axis.
        selector : dict
            Empty for a single-axis plot or a facet selector.
        """
        if self.faceted_plot is None:
            assert not isinstance(self.axes, np.ndarray)
            yield self.axes, {}
            return
        yield from self.faceted_plot.iter_axes()

    def select(self, data: xr.DataArray, selector: Mapping[str, Any]) -> xr.DataArray:
        """Select a field for one plot axis.

        Parameters
        ----------
        data : xarray.DataArray
            Overlay field.
        selector : mapping
            Facet selector.

        Returns
        -------
        xarray.DataArray
            Two-dimensional field for the axis.
        """
        return select_facet(data, selector)

    def register_layer(
        self,
        kind: str,
        artists: Sequence[Any],
        *,
        labels: Sequence[Any] | None = None,
        keys: Sequence[Any] | None = None,
    ) -> None:
        """Register returned primitives as object state.

        Parameters
        ----------
        kind : str
            Layer identifier.
        artists : sequence
            Created primitives.
        labels : sequence, optional
            Associated text labels.
        keys : sequence, optional
            Associated quiver keys.

        Returns
        -------
        None
            The parent object is modified in place.
        """
        layer: dict[str, Any] = {"kind": kind, "artists": list(artists)}
        if labels is not None:
            layer["labels"] = list(labels)
        if keys is not None:
            layer["keys"] = list(keys)
        self.layers.append(layer)
        self._promote_contours()

    def _promote_contours(self) -> None:
        """Keep registered line contours above subsequently added layers."""
        contour_layers = [layer for layer in self.layers if layer["kind"] == "contour"]
        for layer_index, layer in enumerate(contour_layers, start=1):
            zorder = 100.0 + layer_index
            for artist in layer["artists"]:
                if hasattr(artist, "set_zorder"):
                    artist.set_zorder(zorder)
                for collection in getattr(artist, "collections", []):
                    collection.set_zorder(zorder)

    def __repr__(self) -> str:
        """Return a compact representation of stored plot state."""
        axes_count = len(list(self.iter_axes()))
        _method = f"method={self.method!r}, "
        if self.method == "default":
            _method = ""
        return (
            f"GeoPlot({_method!r}projection={self.projection!r}, "
            f"axes={axes_count}, layers={len(self.layers)}, "
            f"colorbar={self.colorbar is not None})"
        )


class Adder:
    """Add reusable plotting primitives to an existing :class:`GeoPlot`.

    Every method delegates drawing to the corresponding function in
    :mod:`plot_utils`, stores the returned primitives on the parent object, and
    returns the parent to support method chaining.

    Parameters
    ----------
    plot : GeoPlot
        Parent plot container.
    """

    __slots__ = ("_plot",)

    def __init__(self, plot: GeoPlot) -> None:
        self._plot = plot

    def _normalized(self, data: xr.DataArray) -> xr.DataArray:

        if data is None:
            return None
        data = validate_data(data)

        if self._plot.x not in data.coords:
            raise ValueError(
                f"x coordinate {self._plot.x!r} is not present in related input"
            )
        if self._plot.cyclic:
            data = add_cyclic_point(data, lon=self._plot.x)
        normalized = to_lon180(data, lon=self._plot.x)
        assert normalized is not None
        return normalized

    def default(
        self,
        data: xr.DataArray,
        *,
        x: str | None = None,
        y: str | None = None,
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = False,
        rasterized: bool = False,
        zorder: float = 2.0,
        add_labels: bool = False,
        **kwargs: Any,
    ) -> GeoPlot:
        """Add an xarray default scalar plot.

        Parameters
        ----------
        data : xarray.DataArray
            Overlay field.
        x, y : str, optional
            Horizontal coordinate names. Parent names are used when omitted.
        cmap, norm, vmin, vmax : optional
            Scalar-color mapping parameters.
        robust : bool, default False
            Use percentile-based limits where supported.
        rasterized : bool, default False
            Rasterize dense output where supported.
        zorder : float, default 2
            Drawing order.
        add_labels : bool, default False
            Let xarray add labels.
        **kwargs
            Additional arguments forwarded to :func:`plot_default`.

        Returns
        -------
        GeoPlot
            Parent plot.
        """
        data = self._normalized(data)
        artists: list[ScalarPrimitive] = []
        for axis, selector in self._plot.iter_axes():
            artists.append(
                plot_default(
                    self._plot.select(data, selector),
                    self._plot.figure,
                    axis,
                    x=x or self._plot.x,
                    y=y or self._plot.y,
                    cmap=cmap,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    robust=robust,
                    rasterized=rasterized,
                    zorder=zorder,
                    add_labels=add_labels,
                    **kwargs,
                )
            )
        self._plot.register_layer("default", artists)
        return self._plot

    def pcolormesh(
        self,
        data: xr.DataArray,
        *,
        x: str | None = None,
        y: str | None = None,
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = False,
        rasterized: bool = False,
        zorder: float = 2.0,
        add_labels: bool = False,
        **kwargs: Any,
    ) -> GeoPlot:
        """Add a pseudocolor mesh.

        Parameters
        ----------
        data : xarray.DataArray
            Overlay field.
        x, y : str, optional
            Horizontal coordinate names.
        cmap, norm, vmin, vmax : optional
            Scalar-color mapping parameters.
        robust : bool, default False
            Use percentile-based limits where supported.
        rasterized : bool, default False
            Rasterize the mesh.
        zorder : float, default 2
            Drawing order.
        add_labels : bool, default False
            Let xarray add labels.
        **kwargs
            Additional arguments forwarded to :func:`plot_pcolormesh`.

        Returns
        -------
        GeoPlot
            Parent plot.
        """
        data = self._normalized(data)
        artists: list[QuadMesh] = []
        for axis, selector in self._plot.iter_axes():
            artists.append(
                plot_pcolormesh(
                    self._plot.select(data, selector),
                    self._plot.figure,
                    axis,
                    x=x or self._plot.x,
                    y=y or self._plot.y,
                    cmap=cmap,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    robust=robust,
                    rasterized=rasterized,
                    zorder=zorder,
                    add_labels=add_labels,
                    **kwargs,
                )
            )
        self._plot.register_layer("pcolormesh", artists)
        return self._plot

    def contourf(
        self,
        data: xr.DataArray,
        *,
        x: str | None = None,
        y: str | None = None,
        levels: int | Sequence[float] | np.ndarray | None = None,
        cmap: str | Colormap | None = None,
        colors: str | Sequence[str] | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        extend: Literal["neither", "both", "min", "max"] | None = None,
        robust: bool = False,
        alpha: float | None = None,
        rasterized: bool = False,
        zorder: float = 2.0,
        add_labels: bool = False,
        **kwargs: Any,
    ) -> GeoPlot:
        """Add a filled-contour layer.

        Parameters
        ----------
        data : xarray.DataArray
            Overlay field.
        x, y : str, optional
            Horizontal coordinate names.
        levels : int or sequence of float, optional
            Contour intervals.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap.
        colors : str or sequence of str, optional
            Explicit contour colors.
        norm : matplotlib.colors.Normalize, optional
            Color normalization.
        vmin, vmax : float, optional
            Scalar color limits.
        extend : {"neither", "both", "min", "max"}, optional
            Out-of-range coloring.
        robust : bool, default False
            Use percentile-based color limits.
        alpha : float, optional
            Layer opacity.
        rasterized : bool, default False
            Rasterize contour collections.
        zorder : float, default 2
            Drawing order.
        add_labels : bool, default False
            Let xarray add labels.
        **kwargs
            Additional arguments forwarded to :func:`plot_contourf`.

        Returns
        -------
        GeoPlot
            Parent plot.
        """
        data = self._normalized(data)
        artists: list[QuadContourSet] = []
        for axis, selector in self._plot.iter_axes():
            artists.append(
                plot_contourf(
                    self._plot.select(data, selector),
                    self._plot.figure,
                    axis,
                    x=x or self._plot.x,
                    y=y or self._plot.y,
                    levels=levels,
                    cmap=cmap,
                    colors=colors,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    extend=extend,
                    robust=robust,
                    alpha=alpha,
                    rasterized=rasterized,
                    zorder=zorder,
                    add_labels=add_labels,
                    **kwargs,
                )
            )
        self._plot.register_layer("contourf", artists)
        return self._plot

    def contour(
        self,
        data: xr.DataArray,
        *,
        x: str | None = None,
        y: str | None = None,
        levels: int | Sequence[float] | np.ndarray | None = None,
        cmap: str | Colormap | None = None,
        colors: str | Sequence[str] | None = None,
        linewidths: float | Sequence[float] | None = None,
        linestyles: str | Sequence[str] | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        extend: Literal["neither", "both", "min", "max"] | None = None,
        alpha: float | None = None,
        rasterized: bool = False,
        zorder: float = 3.0,
        add_labels: bool = False,
        clabel: bool = False,
        clabel_fmt: str | Mapping[float, str] = "%1.0f",
        clabel_fontsize: float = 8.0,
        clabel_inline: bool = True,
        clabel_colors: str | Sequence[str] | None = None,
        clabel_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> GeoPlot:
        """Add a line-contour layer.

        Parameters
        ----------
        data : xarray.DataArray
            Overlay field.
        x, y : str, optional
            Horizontal coordinate names.
        levels : int or sequence of float, optional
            Contour levels.
        cmap, colors, norm, vmin, vmax : optional
            Scalar-color mapping parameters.
        linewidths : float or sequence of float, optional
            Contour line widths.
        linestyles : str or sequence of str, optional
            Contour line styles.
        extend : {"neither", "both", "min", "max"}, optional
            Out-of-range coloring.
        alpha : float, optional
            Layer opacity.
        rasterized : bool, default False
            Rasterize contour collections.
        zorder : float, default 3
            Drawing order.
        add_labels : bool, default False
            Let xarray add labels.
        clabel : bool, default False
            Label each contour set.
        clabel_fmt, clabel_fontsize, clabel_inline, clabel_colors, clabel_kwargs
            Contour-label controls.
        **kwargs
            Additional arguments forwarded to :func:`plot_contour`.

        Returns
        -------
        GeoPlot
            Parent plot.
        """
        data = self._normalized(data)
        artists: list[QuadContourSet] = []
        labels: list[list[Text]] = []
        for axis, selector in self._plot.iter_axes():
            artist = plot_contour(
                self._plot.select(data, selector),
                self._plot.figure,
                axis,
                x=x or self._plot.x,
                y=y or self._plot.y,
                levels=levels,
                cmap=cmap,
                colors=colors,
                linewidths=linewidths,
                linestyles=linestyles,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                extend=extend,
                alpha=alpha,
                rasterized=rasterized,
                zorder=zorder,
                add_labels=add_labels,
                **kwargs,
            )
            artists.append(artist)
            if clabel:
                labels.append(
                    add_contour_labels(
                        self._plot.figure,
                        axis,
                        artist,
                        fmt=clabel_fmt,
                        fontsize=clabel_fontsize,
                        inline=clabel_inline,
                        colors=clabel_colors,
                        kwargs=clabel_kwargs,
                    )
                )
        self._plot.register_layer("contour", artists, labels=labels)
        return self._plot

    def imshow(
        self,
        data: xr.DataArray,
        *,
        x: str | None = None,
        y: str | None = None,
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = False,
        interpolation: str | None = None,
        origin: Literal["upper", "lower"] | None = None,
        rasterized: bool = False,
        zorder: float = 2.0,
        add_labels: bool = False,
        **kwargs: Any,
    ) -> GeoPlot:
        """Add an image layer.

        Parameters
        ----------
        data : xarray.DataArray
            Overlay field.
        x, y : str, optional
            Horizontal coordinate names.
        cmap, norm, vmin, vmax : optional
            Scalar-color mapping parameters.
        robust : bool, default False
            Use percentile-based limits where supported.
        interpolation : str, optional
            Image interpolation method.
        origin : {"upper", "lower"}, optional
            Image origin.
        rasterized : bool, default False
            Rasterize the image.
        zorder : float, default 2
            Drawing order.
        add_labels : bool, default False
            Let xarray add labels.
        **kwargs
            Additional arguments forwarded to :func:`plot_imshow`.

        Returns
        -------
        GeoPlot
            Parent plot.
        """
        data = self._normalized(data)
        artists: list[AxesImage] = []
        for axis, selector in self._plot.iter_axes():
            artists.append(
                plot_imshow(
                    self._plot.select(data, selector),
                    self._plot.figure,
                    axis,
                    x=x or self._plot.x,
                    y=y or self._plot.y,
                    cmap=cmap,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    robust=robust,
                    interpolation=interpolation,
                    origin=origin,
                    rasterized=rasterized,
                    zorder=zorder,
                    add_labels=add_labels,
                    **kwargs,
                )
            )
        self._plot.register_layer("imshow", artists)
        return self._plot

    def scatter(
        self,
        data: xr.DataArray,
        *,
        x: str | None = None,
        y: str | None = None,
        hue: str | xr.DataArray | None = None,
        markersize: str | xr.DataArray | None = None,
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        marker: str | None = None,
        size: float | None = None,
        alpha: float | None = None,
        edgecolors: str | Sequence[str] | None = None,
        linewidths: float | Sequence[float] | None = None,
        zorder: float = 3.0,
        add_labels: bool = False,
        **kwargs: Any,
    ) -> GeoPlot:
        """Add a scatter layer.

        Parameters
        ----------
        data : xarray.DataArray
            Source field containing point coordinates.
        x, y : str, optional
            Point-coordinate names.
        hue, markersize : str or xarray.DataArray, optional
            Variables controlling point color and size.
        cmap, norm, vmin, vmax : optional
            Scalar-color mapping parameters.
        marker : str, optional
            Marker style.
        size : float, optional
            Constant marker area.
        alpha : float, optional
            Marker opacity.
        edgecolors : str or sequence of str, optional
            Marker-edge colors.
        linewidths : float or sequence of float, optional
            Marker-edge widths.
        zorder : float, default 3
            Drawing order.
        add_labels : bool, default False
            Let xarray add labels.
        **kwargs
            Additional arguments forwarded to :func:`plot_scatter`.

        Returns
        -------
        GeoPlot
            Parent plot.
        """
        data = self._normalized(data)
        artists: list[PathCollection] = []
        for axis, selector in self._plot.iter_axes():
            artists.append(
                plot_scatter(
                    self._plot.select(data, selector),
                    self._plot.figure,
                    axis,
                    x=x or self._plot.x,
                    y=y or self._plot.y,
                    hue=hue,
                    markersize=markersize,
                    cmap=cmap,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    marker=marker,
                    size=size,
                    alpha=alpha,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                    zorder=zorder,
                    add_labels=add_labels,
                    **kwargs,
                )
            )
        self._plot.register_layer("scatter", artists)
        return self._plot

    def quiver(
        self,
        u: xr.DataArray,
        v: xr.DataArray,
        *,
        x: str | None = None,
        y: str | None = None,
        subsample: int | tuple[int, int] | list[int] = (1, 1),
        add_key: bool = True,
        key_magnitude: float | None = None,
        key_units: str | None = None,
        key_x: float = 0.1,
        key_y: float = -0.045,
        scale: float | None = None,
        color: str | None = None,
        width: float | None = None,
        zorder: float = 4.0,
        **kwargs: Any,
    ) -> GeoPlot:
        """Add a vector layer.

        Parameters
        ----------
        u, v : xarray.DataArray
            Zonal and meridional vector components.
        x, y : str, optional
            Horizontal coordinate names.
        subsample : int or tuple of int, default (1, 1)
            Spatial stride used to thin vectors.
        add_key : bool, default True
            Add one shared reference key.
        key_magnitude : int or float, optional
            Reference magnitude.
        key_units : str, optional
            Units appended to the key label.
        key_x, key_y : float, default 0.1, -0.045
            Key location in axis coordinates.
        scale : float, optional
            Matplotlib quiver scale.
        color : str, optional
            Arrow color.
        width : float, optional
            Arrow-shaft width.
        zorder : float, default 4
            Drawing order.
        **kwargs
            Additional arguments forwarded to :func:`plot_quiver`.

        Returns
        -------
        GeoPlot
            Parent plot.
        """
        u = self._normalized(u)
        v = self._normalized(v)
        validated_u, validated_v = validate_vector_components(u, v)
        assert validated_u is not None and validated_v is not None
        if add_key and key_magnitude is None:
            key_magnitude = get_quiver_key_mag(validated_u, validated_v)
        key_axis = (
            self._plot.faceted_plot.bottom_left_axis()
            if self._plot.faceted_plot is not None
            else next(self._plot.iter_axes())[0]
        )

        artists: list[Quiver] = []
        keys: list[QuiverKey | None] = []
        for axis, selector in self._plot.iter_axes():
            quiver_artist, quiver_key = plot_quiver(
                self._plot.select(validated_u, selector),
                self._plot.select(validated_v, selector),
                self._plot.figure,
                axis,
                x=x or self._plot.x,
                y=y or self._plot.y,
                subsample=subsample,
                add_key=add_key and axis is key_axis,
                key_magnitude=key_magnitude,
                key_units=key_units,
                key_x=key_x,
                key_y=key_y,
                scale=scale,
                color=color,
                width=width,
                zorder=zorder,
                **kwargs,
            )

            artists.append(quiver_artist)
            keys.append(quiver_key)
        self._plot.quiver = artists if self._plot.is_faceted else artists[0]
        self._plot.quiver_key = next((key for key in keys if key is not None), None)
        self._plot.register_layer("quiver", artists, keys=keys)
        return self._plot

    def significance(
        self,
        pvalues: xr.DataArray,
        *,
        x: str | None = None,
        y: str | None = None,
        level: float = 0.05,
        color: str = "grey",
        alpha: float = 0.3,
        marker: str | None = None,
        edgecolors: str | None = None,
        subsample: int | tuple[int, int] | list[int] = (1, 1),
        size: float = 0.25,
        zorder: float = 3.0,
    ) -> GeoPlot:
        """Add pointwise significance markers.

        Parameters
        ----------
        pvalues : xarray.DataArray
            Pointwise p-values.
        x, y : str, optional
            Horizontal coordinate names.
        level : float, default 0.05
            Significance threshold.
        color : str, default "grey"
            Marker face color.
        alpha : float, default 0.3
            Marker opacity.
        marker : str, optional
            Marker style.
        edgecolors : str, optional
            Marker-edge color.
        subsample : int or tuple of int, default (1, 1)
            Spatial stride used to thin markers.
        size : float, default 0.25
            Marker area.
        zorder : float, default 3
            Drawing order.

        Returns
        -------
        GeoPlot
            Parent plot.
        """
        pvalues = self._normalized(pvalues)
        artists: list[PathCollection] = []
        for axis, selector in self._plot.iter_axes():
            artists.append(
                plot_significance(
                    self._plot.select(pvalues, selector),
                    self._plot.figure,
                    axis,
                    x=x or self._plot.x,
                    y=y or self._plot.y,
                    level=level,
                    color=color,
                    alpha=alpha,
                    marker=marker,
                    edgecolors=edgecolors,
                    subsample=subsample,
                    size=size,
                    zorder=zorder,
                )
            )
        self._plot.register_layer("significance", artists)
        return self._plot

    def colorbar(
        self,
        mappable: ScalarMappable | None = None,
        *,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        drawedges: bool = False,
        extend: Literal["neither", "both", "min", "max"] | None = None,
        label: str | None = None,
        ticks: Sequence[float] | np.ndarray | None = None,
        tick_labels: Sequence[str] | None = None,
    ) -> GeoPlot:
        """Add a colorbar for an existing scalar primitive.

        Parameters
        ----------
        mappable : matplotlib.cm.ScalarMappable, optional
            Primitive described by the colorbar. The base mappable is used when
            omitted.
        orientation : {"vertical", "horizontal"}, default "vertical"
            Colorbar orientation.
        drawedges : bool, default False
            Draw interval edges.
        extend : {"neither", "both", "min", "max"}, optional
            Out-of-range extension behavior.
        label : str, optional
            Colorbar label.
        ticks : sequence of float, optional
            Explicit tick positions.
        tick_labels : sequence of str, optional
            Explicit tick labels.

        Returns
        -------
        GeoPlot
            Parent plot.
        """
        colorbar = _add_colorbar(
            self._plot.figure,
            self._plot.axes,
            mappable or self._plot.mappable,
            orientation=orientation,
            subplots=self._plot.is_faceted,
            pad_bottom=True if self._plot.quiver_key is not None else None,
            drawedges=drawedges,
            extend=extend,
            label=label,
            ticks=ticks,
            tick_labels=tick_labels,
        )
        self._plot.colorbar = colorbar
        self._plot.register_layer("colorbar", [colorbar])
        return self._plot

    def grid_boundary(
        self,
        linewidth: float = 1.5,
        color: str = "black",
        zorder: float = 1,
    ) -> None:
        """Draw the exterior boundary of a two-dimensional longitude-latitude grid.

        Parameters
        ----------
        linewidth : float, default 1.5
            Width of the boundary line, in points.
        color : str, default "black"
            Matplotlib-compatible color specification for the boundary line.
        zorder : float, default 20
            Drawing order of the boundary. Artists with higher values are drawn
            above artists with lower values.

        Returns
        -------
        None
        """
        artists: list[PathCollection] = []
        for axis, selector in self._plot.iter_axes():
            artists.append(
                add_grid_boundary(
                    axis,
                    lon=self._plot.grid[self._plot.x].values,
                    lat=self._plot.grid[self._plot.y].values,
                    transform=ccrs.PlateCarree(),
                    linewidth=linewidth,
                    color=color,
                    zorder=zorder,
                )
            )

            self._plot.register_layer("grid_boundary", artists)
        return self._plot


def _encode_ffmpeg(
    input_pattern: str,
    outfile: Path,
    *,
    fps: int,
    session_tmp_dir: Path,
) -> None:
    """Encode numbered PNG frames as an H.264 MP4 file.

    Parameters
    ----------
    input_pattern : str
        FFmpeg input pattern, for example ``/tmp/frames/%06d.png``.
    outfile : pathlib.Path
        Output MP4 path.
    fps : int
        Frames per second.
    session_tmp_dir : pathlib.Path
        Temporary frame directory removed after encoding.

    Raises
    ------
    RuntimeError
        If FFmpeg returns a nonzero status.
    """
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-vf",
        "scale=1920:1080, pad=iw+mod(iw\\,2):ih+mod(ih\\,2), format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-profile:v",
        "high",
        "-tune",
        "animation",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(outfile),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable was not found") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg encoding failed: {exc.stderr.strip()}") from exc
    finally:
        shutil.rmtree(session_tmp_dir, ignore_errors=True)


def _render_animation_frame(
    frame_number: int,
    frame_value: Any,
    dim: str,
    title: str,
    dpi: int,
    session_tmp_dir: Path,
    data: xr.DataArray,
    u: xr.DataArray | None,
    v: xr.DataArray | None,
    geo_options: Mapping[str, Any],
) -> None:
    """Render one animation frame to a numbered PNG file."""
    if np.issubdtype(np.asarray(frame_value).dtype, np.datetime64):
        frame_value = pd.to_datetime(frame_value).strftime("%Y-%m-%d %H:%M")
    frame_title = f"{dim}: {frame_value}"
    if title:
        frame_title = f"{title}\n{frame_title}"
    options = dict(geo_options)
    options["da"] = data
    options["u_component"] = u
    options["v_component"] = v
    options["title"] = frame_title
    plot = GeoPlot(**options)
    filename = session_tmp_dir / f"{frame_number:06d}.png"
    plot.figure.savefig(filename, dpi=dpi, bbox_inches="tight")
    plot.figure.clear()
    plt.close(plot.figure)


class Animate:
    """Render a sequence of :class:`GeoPlot` objects and encode an MP4.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field containing the animation dimension.
    dim : str, default "time"
        Animation dimension.
    x, y : str, optional
        Horizontal coordinate names.
    col, row : str, optional
        Facet dimensions retained within each frame.
    col_wrap : int, optional
        Maximum number of facet columns.
    figsize : tuple of float, optional
        Figure size for each frame.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}
        Base scalar plot method.
    projection : {"PlateCarree", "Mercator", "Robinson", "Mollweide", "Orthographic", "LambertConformal", "AlbersEqualArea", "Stereographic", "NorthPolarStereo", "SouthPolarStereo"}, optional
        Display projection.
    cmap, norm, vmin, vmax, levels, extend : optional
        Base scalar styling.
    robust, rasterized : bool, default False
        Base scalar rendering options.
    title : str, optional
        Frame-title prefix.
    orientation : {"vertical", "horizontal"}, optional
        Base colorbar orientation.
    add_colorbar : bool, default True
        Add a base colorbar to each frame.
    drawedges : bool, default False
        Draw colorbar interval edges.
    cbar_label : str, optional
        Base colorbar label.
    global_extent, gridlines : bool, default False
        Map-layout options.
    add_grid_bounds:
        If True, draw an outline along the outer perimeter of the plotted grid domain.
    set_extent : tuple of float, optional
        Explicit map extent.
    coastlines, borders, states, ocean, land, lakes, rivers : bool
        Map-feature switches.
    u_component, v_component : xarray.DataArray, optional
        Vector components animated with ``data``.
    colorbar_kwargs, quiver_kwargs : mapping, optional
        Base colorbar and vector options.
    clabel : bool, default False
        Label line contours.
    clabel_fmt, clabel_fontsize, clabel_inline, clabel_colors, clabel_kwargs
        Contour-label options.
    cyclic : bool, default False
        Append a cyclic horizontal point to each frame.
    indices : sequence of int, optional
        Frame indices. Every index is rendered when omitted.
    outfile : str or pathlib.Path, optional
        Output MP4 path. A temporary path is generated when omitted.
    quality : {"low", "medium", "high"}, default "medium"
        Frame-resolution preset.
    fps : int, default 1
        Encoded frames per second.
    parallel : bool, default True
        Render frames using the Dask process scheduler.
    display_inline : bool, default True
        Display the encoded MP4 in an active Jupyter kernel.
    **kwargs
        Additional base plotting arguments forwarded to :class:`GeoPlot`.

    Attributes
    ----------
    outfile : pathlib.Path
        Encoded MP4 path.
    indices : list of int
        Rendered frame indices.
    display_result : object or None
        Result returned by IPython display.
    """

    def __init__(
        self,
        da: xr.DataArray,
        dim: str = "time",
        *,
        x: str | None = None,
        y: str | None = None,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        figsize: tuple[float, float] | None = None,
        method: Literal[
            "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
        ] = "default",
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
        ]
        | None = None,
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        units: str | None = None,
        levels: int | Sequence[float] | np.ndarray | None = None,
        extend: Literal["neither", "both", "min", "max"] | None = None,
        robust: bool = False,
        rasterized: bool = False,
        title: str | None = None,
        orientation: Literal["vertical", "horizontal"] | None = None,
        add_colorbar: bool = True,
        drawedges: bool = False,
        cbar_label: str | None = None,
        global_extent: bool = False,
        set_extent: tuple[float, float, float, float] | None = None,
        gridlines: bool = False,
        add_grid_bounds: bool = False,
        coastlines: bool = True,
        borders: bool = True,
        states: bool = True,
        ocean: bool = True,
        land: bool = True,
        lakes: bool = False,
        rivers: bool = False,
        u_component: xr.DataArray | None = None,
        v_component: xr.DataArray | None = None,
        colorbar_kwargs: Mapping[str, Any] | None = None,
        quiver_kwargs: Mapping[str, Any] | None = None,
        clabel: bool = False,
        clabel_fmt: str | Mapping[float, str] = "%1.0f",
        clabel_fontsize: float = 8.0,
        clabel_inline: bool = True,
        clabel_colors: str | Sequence[str] | None = None,
        clabel_kwargs: Mapping[str, Any] | None = None,
        cyclic: bool = False,
        indices: Sequence[int] | np.ndarray | None = None,
        outfile: str | Path | None = None,
        quality: Literal["low", "medium", "high"] = "medium",
        fps: int = 1,
        parallel: bool = True,
        display_inline: bool = True,
        **kwargs: Any,
    ) -> None:
        self.dim = dim
        self.data, self.u_component, self.v_component = validate_animation_inputs(
            dim,
            da,
            u_component,
            v_component,
        )
        if fps < 1:
            raise ValueError("fps must be greater than or equal to 1")
        self.fps = fps
        self.parallel = parallel
        self.display_inline = display_inline
        if indices is None:
            self.indices = list(range(self.data.sizes[dim]))
        else:
            self.indices = [int(index) for index in indices]
        if not self.indices:
            raise ValueError("indices must contain at least one frame")
        for index in self.indices:
            if index < 0 or index >= self.data.sizes[dim]:
                raise IndexError(
                    f"frame index {index} is outside [0, {self.data.sizes[dim] - 1}]"
                )
        if outfile is None:
            self.outfile = (
                tmp / "animations" / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.mp4"  # noqa: DTZ005
            )

            self.user_outfile = False
        else:
            self.outfile = Path(outfile)
            self.user_outfile = True
        self.outfile.parent.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self.title = title or ""
        self.geo_options: dict[str, Any] = {
            "x": x,
            "y": y,
            "col": col,
            "row": row,
            "col_wrap": col_wrap,
            "figsize": figsize,
            "method": method,
            "projection": projection,
            "cmap": cmap,
            "norm": norm,
            "vmin": vmin,
            "vmax": vmax,
            "units": units,
            "levels": levels,
            "extend": extend,
            "robust": robust,
            "rasterized": rasterized,
            "orientation": orientation,
            "add_colorbar": add_colorbar,
            "drawedges": drawedges,
            "cbar_label": cbar_label,
            "global_extent": global_extent,
            "set_extent": set_extent,
            "gridlines": gridlines,
            "add_grid_bounds": add_grid_bounds,
            "coastlines": coastlines,
            "borders": borders,
            "states": states,
            "ocean": ocean,
            "land": land,
            "lakes": lakes,
            "rivers": rivers,
            "colorbar_kwargs": colorbar_kwargs,
            "quiver_kwargs": quiver_kwargs,
            "clabel": clabel,
            "clabel_fmt": clabel_fmt,
            "clabel_fontsize": clabel_fontsize,
            "clabel_inline": clabel_inline,
            "clabel_colors": clabel_colors,
            "clabel_kwargs": clabel_kwargs,
            "cyclic": cyclic,
        }
        self.geo_options.update(kwargs)
        self.display_result: Any | None = None
        self.run()

    def sel(self, data: xr.DataArray | None, index: int) -> xr.DataArray | None:
        """Select one animation frame from an optional field."""
        return None if data is None else data.isel({self.dim: index})

    def run(self) -> Path:
        """Render all frames and encode the animation.

        Returns
        -------
        pathlib.Path
            Encoded MP4 path.
        """
        session_tmp_dir = Path(tempfile.mkdtemp(prefix="geoplot-frames-"))
        dpi = {"low": 300, "medium": 600, "high": 1200}[self.quality]
        tasks = [
            (
                frame_number,
                self.data[self.dim][index].values,
                self.dim,
                self.title,
                dpi,
                session_tmp_dir,
                self.sel(self.data, index),
                self.sel(self.u_component, index),
                self.sel(self.v_component, index),
                self.geo_options,
            )
            for frame_number, index in enumerate(self.indices)
        ]
        if self.parallel and len(tasks) > 1:
            workers = max(1, min(len(tasks), max(1, n_cpus // 2)))
            delayed = [dask.delayed(_render_animation_frame)(*task) for task in tasks]
            with DaskProgressBar():
                dask.compute(*delayed, scheduler="processes", num_workers=workers)
        else:
            for task in SerialProgressBar(tasks, total=len(tasks)):
                _render_animation_frame(*task)
        _encode_ffmpeg(
            str(session_tmp_dir / "%06d.png"),
            self.outfile,
            fps=self.fps,
            session_tmp_dir=session_tmp_dir,
        )
        if self.user_outfile:
            print(f"Animation saved to: {self.outfile}")
        if self.display_inline and "ipykernel" in sys.modules:
            from IPython.display import Video, display

            self.display_result = display(
                Video(
                    str(self.outfile),
                    embed=True,
                    html_attributes="controls autoplay loop",
                    width=800,
                    height=600,
                )
            )
        return self.outfile

    def get_outfile(self, da: xr.DataArray):
        source = self, da.encoding.get("source")

        if source:
            filename = Path(source).name
        else:
            filename = f"{datetime.now():%Y%m%dT%H%M%S}.mp4"  # noqa: DTZ005

        output_dir = Path.cwd() / "animations"
        output_dir.mkdir(parents=True, exist_ok=True)

        outfile = output_dir / filename
        stem = outfile.stem
        suffix = outfile.suffix
        index = 1

        while outfile.exists():
            outfile = output_dir / f"{stem}_{index}{suffix}"
            index += 1

        self.outfile = outfile

    def __repr__(self) -> str:
        """Return a compact representation of animation state."""
        return (
            f"Animate(dim={self.dim!r}, frames={len(self.indices)}, "
            f"fps={self.fps}, outfile={str(self.outfile)!r})"
        )


def geo(
    da: xr.DataArray,
    *,
    x: str | None = None,
    y: str | None = None,
    col: str | None = None,
    row: str | None = None,
    col_wrap: int | None = None,
    figsize: tuple[float, float] | None = None,
    method: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
    ] = "default",
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
    ]
    | None = None,
    cmap: str | Colormap | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    units: str | None = None,
    levels: int | Sequence[float] | np.ndarray | None = None,
    extend: Literal["neither", "both", "min", "max"] | None = None,
    robust: bool = False,
    rasterized: bool = False,
    title: str = "",
    orientation: Literal["vertical", "horizontal"] | None = None,
    add_colorbar: bool = True,
    drawedges: bool = False,
    cbar_label: str | None = None,
    global_extent: bool = False,
    set_extent: tuple[float, float, float, float] | None = None,
    gridlines: bool = False,
    add_grid_bounds: bool = False,
    coastlines: bool = True,
    borders: bool = True,
    states: bool = True,
    ocean: bool = True,
    land: bool = True,
    lakes: bool = False,
    rivers: bool = False,
    p_value: xr.DataArray | None = None,
    pvalue_kwargs: Mapping[str, Any] | None = None,
    u_component: xr.DataArray | None = None,
    v_component: xr.DataArray | None = None,
    quiver_kwargs: Mapping[str, Any] | None = None,
    colorbar_kwargs: Mapping[str, Any] | None = None,
    clabel: bool = False,
    clabel_fmt: str | Mapping[float, str] = "%1.0f",
    clabel_fontsize: float = 8.0,
    clabel_inline: bool = True,
    clabel_colors: str | Sequence[str] | None = None,
    clabel_kwargs: Mapping[str, Any] | None = None,
    cyclic: bool = False,
    **kwargs: Any,
) -> GeoPlot:
    """Create a fully typed :class:`GeoPlot` container.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field to plot.
    x, y : str, optional
        Horizontal coordinate names.
    col, row : str, optional
        Facet dimensions.
    col_wrap : int, optional
        Maximum number of facet columns.
    figsize : tuple of float, optional
        Figure size in inches.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}
        Base scalar plotting method.
    projection : {"PlateCarree", "Mercator", "Robinson", "Mollweide", "Orthographic", "LambertConformal", "AlbersEqualArea", "Stereographic", "NorthPolarStereo", "SouthPolarStereo"}, optional
        Display projection.
    cmap, norm, vmin, vmax, units, levels, extend : optional
        Base scalar and colorbar configuration.
    robust, rasterized : bool, default False
        Base scalar rendering options.
    title : str, default ""
        Plot title.
    orientation : {"vertical", "horizontal"}, optional
        Base colorbar orientation.
    add_colorbar, drawedges : bool
        Base colorbar controls.
    cbar_label : str, optional
        Explicit base colorbar label.
    global_extent, gridlines : bool, default False
        Map-layout controls.
    add_grid_bounds : bool
        If True, draw an outline along the outer perimeter of the plotted grid domain.
    set_extent : tuple of float, optional
        Explicit geographic extent.
    coastlines, borders, states, ocean, land, lakes, rivers : bool
        Map-feature switches.
    p_value : xarray.DataArray, optional
        Pointwise significance field.
    pvalue_kwargs : mapping, optional
        Significance-layer options.
    u_component, v_component : xarray.DataArray, optional
        Vector components.
    quiver_kwargs : mapping, optional
        Vector-layer options.
    colorbar_kwargs : mapping, optional
        Base colorbar tick options.
    clabel, clabel_fmt, clabel_fontsize, clabel_inline, clabel_colors, clabel_kwargs
        Line-contour label controls.
    cyclic : bool, default False
        Append a cyclic horizontal point.
    **kwargs
        Additional method-specific plotting arguments.

    Returns
    -------
    GeoPlot
        Plot container holding the figure, axes, and all primitives.
    """
    return GeoPlot(
        da,
        x=x,
        y=y,
        col=col,
        row=row,
        col_wrap=col_wrap,
        figsize=figsize,
        method=method,
        projection=projection,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        units=units,
        levels=levels,
        extend=extend,
        robust=robust,
        rasterized=rasterized,
        title=title,
        orientation=orientation,
        add_colorbar=add_colorbar,
        drawedges=drawedges,
        cbar_label=cbar_label,
        global_extent=global_extent,
        set_extent=set_extent,
        gridlines=gridlines,
        add_grid_bounds=add_grid_bounds,
        coastlines=coastlines,
        borders=borders,
        states=states,
        ocean=ocean,
        land=land,
        lakes=lakes,
        rivers=rivers,
        p_value=p_value,
        pvalue_kwargs=pvalue_kwargs,
        u_component=u_component,
        v_component=v_component,
        quiver_kwargs=quiver_kwargs,
        colorbar_kwargs=colorbar_kwargs,
        clabel=clabel,
        clabel_fmt=clabel_fmt,
        clabel_fontsize=clabel_fontsize,
        clabel_inline=clabel_inline,
        clabel_colors=clabel_colors,
        clabel_kwargs=clabel_kwargs,
        cyclic=cyclic,
        **kwargs,
    )


def animate(
    da: xr.DataArray,
    dim: str = "time",
    *,
    x: str | None = None,
    y: str | None = None,
    col: str | None = None,
    row: str | None = None,
    col_wrap: int | None = None,
    figsize: tuple[float, float] | None = None,
    method: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow", "scatter"
    ] = "default",
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
    ]
    | None = None,
    cmap: str | Colormap | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    units: str | None = None,
    levels: int | Sequence[float] | np.ndarray | None = None,
    extend: Literal["neither", "both", "min", "max"] | None = None,
    robust: bool = False,
    rasterized: bool = False,
    title: str | None = None,
    orientation: Literal["vertical", "horizontal"] | None = None,
    add_colorbar: bool = True,
    drawedges: bool = False,
    cbar_label: str | None = None,
    global_extent: bool = False,
    set_extent: tuple[float, float, float, float] | None = None,
    gridlines: bool = False,
    add_grid_bounds: bool = False,
    coastlines: bool = True,
    borders: bool = True,
    states: bool = True,
    ocean: bool = True,
    land: bool = True,
    lakes: bool = False,
    rivers: bool = False,
    u_component: xr.DataArray | None = None,
    v_component: xr.DataArray | None = None,
    colorbar_kwargs: Mapping[str, Any] | None = None,
    quiver_kwargs: Mapping[str, Any] | None = None,
    clabel: bool = False,
    clabel_fmt: str | Mapping[float, str] = "%1.0f",
    clabel_fontsize: float = 8.0,
    clabel_inline: bool = True,
    clabel_colors: str | Sequence[str] | None = None,
    clabel_kwargs: Mapping[str, Any] | None = None,
    cyclic: bool = False,
    indices: Sequence[int] | np.ndarray | None = None,
    outfile: str | Path | None = None,
    quality: Literal["low", "medium", "high"] = "medium",
    fps: int = 1,
    parallel: bool = True,
    display_inline: bool = True,
    **kwargs: Any,
) -> Animate:
    """Create and execute a fully typed :class:`Animate` workflow.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field containing ``dim``.
    dim : str, default "time"
        Animation dimension.
    x, y, col, row, col_wrap, figsize : optional
        Per-frame layout configuration.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}
        Per-frame scalar plot type.
    projection : {"PlateCarree", "Mercator", "Robinson", "Mollweide", "Orthographic", "LambertConformal", "AlbersEqualArea", "Stereographic", "NorthPolarStereo", "SouthPolarStereo"}, optional
        Per-frame display projection.
    cmap, norm, vmin, vmax, units, levels, extend : optional
        Per-frame scalar styling.
    robust, rasterized : bool, default False
        Per-frame scalar rendering options.
    title : str, optional
        Frame-title prefix.
    orientation, add_colorbar, drawedges, cbar_label, colorbar_kwargs : optional
        Per-frame colorbar options.
    global_extent, set_extent, gridlines, coastlines, borders, states, ocean, land, lakes, rivers
        Per-frame map-feature options.
    add_grid_bounds:
        If True, draw an outline along the outer perimeter of the plotted grid domain.
    u_component, v_component, quiver_kwargs : optional
        Per-frame vector layer.
    clabel, clabel_fmt, clabel_fontsize, clabel_inline, clabel_colors, clabel_kwargs
        Per-frame line-contour label options.
    cyclic : bool, default False
        Append a cyclic horizontal point.
    indices : sequence of int, optional
        Frame indices.
    outfile : str or pathlib.Path, optional
        Output MP4 path.
    quality : {"low", "medium", "high"}, default "medium"
        Frame-resolution preset.
    fps : int, default 1
        Frames per second.
    parallel : bool, default True
        Render frames using Dask processes.
    display_inline : bool, default True
        Display the MP4 in Jupyter.
    **kwargs
        Additional per-frame plotting arguments.

    Returns
    -------
    Animate
        Animation container holding output state.
    """
    return Animate(
        da,
        dim,
        x=x,
        y=y,
        col=col,
        row=row,
        col_wrap=col_wrap,
        figsize=figsize,
        method=method,
        projection=projection,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        units=units,
        levels=levels,
        extend=extend,
        robust=robust,
        rasterized=rasterized,
        title=title,
        orientation=orientation,
        add_colorbar=add_colorbar,
        drawedges=drawedges,
        cbar_label=cbar_label,
        global_extent=global_extent,
        set_extent=set_extent,
        gridlines=gridlines,
        add_grid_bounds=add_grid_bounds,
        coastlines=coastlines,
        borders=borders,
        states=states,
        ocean=ocean,
        land=land,
        lakes=lakes,
        rivers=rivers,
        u_component=u_component,
        v_component=v_component,
        colorbar_kwargs=colorbar_kwargs,
        quiver_kwargs=quiver_kwargs,
        clabel=clabel,
        clabel_fmt=clabel_fmt,
        clabel_fontsize=clabel_fontsize,
        clabel_inline=clabel_inline,
        clabel_colors=clabel_colors,
        clabel_kwargs=clabel_kwargs,
        cyclic=cyclic,
        indices=indices,
        outfile=outfile,
        quality=quality,
        fps=fps,
        parallel=parallel,
        display_inline=display_inline,
        **kwargs,
    )
