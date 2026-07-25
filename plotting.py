"""
Cartopy-based plotting utilities for xarray DataArrays, including faceted map
plots with customizable features and styling.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as cgeo
import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point
from cf_xarray import *
from IPython.display import DisplayHandle
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver, QuiverKey

from .progress import DaskProgressBar, SerialProgressBar
from .tools import AttrDict, get_fsig, n_cpus, tmp
from .xgeo_utils import to_lon180

pad_quiver_key: list[bool | None] = [None, None]


def _get_lon_lat(
    da: xr.DataArray | xr.Dataset,
) -> tuple[str, str]:
    """Return the longitude and latitude coordinate names."""
    ds = da if isinstance(da, xr.Dataset) else da.to_dataset(name=da.name or "data")

    if "latitude" not in ds.cf.coordinates or "longitude" not in ds.cf.coordinates:
        ds = ds.cf.guess_coord_axis()

    lon = ds.cf["longitude"]
    lat = ds.cf["latitude"]

    if lon.name is None or lat.name is None:
        raise ValueError(
            "Could not determine longitude and latitude coordinate names, specify x and y"
        )

    return lon.name, lat.name


def _get_quiver_key_mag(u: xr.DataArray, v: xr.DataArray) -> int | float:
    """Return a reference quiver-key magnitude from the 75th percentile speed."""
    mag = (u**2 + v**2) ** 0.5
    key_mag = np.round(mag.quantile(0.75, skipna=True).values)
    key_mag_int = int(key_mag)
    key_magnitude = key_mag_int if key_mag_int != 0 else np.round(key_mag, 3)
    return key_magnitude


def _check_cartopy_axis(ax, kwargs) -> dict:
    """Inject a ``PlateCarree`` data transform when ``ax`` is a GeoAxes."""
    if isinstance(ax, cgeo.GeoAxes):
        kwargs["transform"] = ccrs.PlateCarree()

    return kwargs


def _add_contour_labels(
    ax: plt.Axes | cgeo.GeoAxes,
    artist: Any,
    *,
    fmt: str = "%1.0f",
    fontsize: float = 8,
    inline: bool = True,
    colors: str | None = None,
    kwargs: dict | None = None,
) -> Any:
    """Label the line contours of ``artist`` on ``ax``.

    Single point of definition for the ``Axes.clabel`` call shared by the
    single-axis renderer, the faceted renderer, and the :class:`Adder` contour
    overlay. Each contour set is labelled exactly once, from its own artist.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Axis on which the contour artist is drawn.
    artist : Any
        Contour set returned by a ``contour`` plotting call.
    fmt : str, default "%1.0f"
        Contour-label format.
    fontsize : float, default 8
        Contour-label font size.
    inline : bool, default True
        Draw contour labels inline.
    colors : str, optional
        Contour-label color.
    kwargs : dict, optional
        Additional arguments forwarded to ``Axes.clabel``.

    Returns
    -------
    list of matplotlib.text.Text
        The label artists created by ``Axes.clabel``.
    """
    return ax.clabel(
        artist,
        fmt=fmt,
        fontsize=fontsize,
        inline=inline,
        colors=colors,
        **(kwargs or {}),
    )


def _add_gridlines(ax: cgeo.GeoAxes):
    """Add labeled longitude/latitude gridlines with the shared module style.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axis to which gridlines are added.

    Returns
    -------
    cartopy.mpl.gridliner.Gridliner
        The gridliner instance.
    """
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=1,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    return gl


def quiver(
    u: xr.DataArray,
    v: xr.DataArray,
    x: str = "lon",
    y: str = "lat",
    ax: plt.Axes | cgeo.GeoAxes = None,
    subsample: tuple[int] | list[int] = (1, 1),
    add_key: bool = True,
    subplots: bool = False,
    **kwargs,
):
    """
    Plot quiver arrows on a Cartopy axis.

    Parameters
    ----------
    u : xarray.DataArray
        The u-component of the vector field.
    v : xarray.DataArray
        The v-component of the vector field.
    x : str, optional
        Name of the x dimension in the data. Default is "lon".
    y : str, optional
        Name of the y dimension in the data. Default is "lat".
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes, optional
        The axis to plot on. If None, the current axis is used.
    subsample : int or tuple of int, optional
        The subsample stride for plotting points to reduce overplotting (x, y).
    add_key : bool, optional
        Whether to add a quiver key. Default is True.
    subplots : bool, optional
        Whether the axis belongs to a grid of subplots. Default is False.
    **kwargs
        Additional keyword arguments. Accepted keys are
        ``key_magnitude: int | float`` (reference arrow length) and
        ``key_units: str``; any remaining argument is forwarded to
        ``matplotlib.axes.Axes.quiver`` such as ``scale: float``,
        ``color: str`` and ``width: float``. See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html

    Returns
    -------
    tuple
        ``(ax, quiver, quiver_key)`` where ``ax`` is the axis, ``quiver`` is the
        ``matplotlib.quiver.Quiver`` artist and ``quiver_key`` is the
        ``matplotlib.quiver.QuiverKey`` or None when ``add_key`` is False.
    """

    ax = ax or plt.gca()
    kwargs = _check_cartopy_axis(ax, kwargs)
    key_magnitude = kwargs.pop("key_magnitude", None)
    key_units = kwargs.pop("key_units", None)

    if isinstance(subsample, int):
        subsample = (subsample, subsample)

    if len(subsample) > 2:
        raise ValueError("subsample must be a tuple or list with at most 2 elements")

    if (x not in u.coords or y not in u.coords) or (
        x not in v.coords or y not in v.coords
    ):
        x, y = _get_lon_lat(u)

    sel = {x: slice(None, None, subsample[0]), y: slice(None, None, subsample[1])}

    u = u.isel(sel)
    v = v.isel(sel)

    x_vals = u.coords[x]
    y_vals = u.coords[y]

    # Detect if coordinates are already 2D
    if x_vals.ndim == 2 and y_vals.ndim == 2:
        x2d, y2d = x_vals.values, y_vals.values
    else:
        x2d, y2d = np.meshgrid(x_vals.values, y_vals.values)

    q = ax.quiver(
        x2d,
        y2d,
        u.values,
        v.values,
        angles="xy",
        **kwargs,
    )

    qk = None

    if add_key:
        if key_units is None:
            u_units = u.attrs.get("units", "")
            v_units = v.attrs.get("units", "")
            assert u_units == v_units, "units of u and v components must match"
            key_units = u_units

        if not key_magnitude:
            key_magnitude = _get_quiver_key_mag(u, v)

        label = f"{key_magnitude} {key_units}".strip()
        fig = ax.get_figure()

        plt.subplots_adjust()

        cax = get_cax(
            fig=fig,
            axes=ax,
            orientation="horizontal",
            subplots=subplots,
            adjust=False,
        )
        bbox = cax.get_position()
        cax.remove()

        # Desired quiver-key anchor in axes coordinates.
        key_x_ax = 0.100
        key_y_ax = -0.045

        if pad_quiver_key != [None, None]:
            key_x_ax, key_y_ax = ax.transAxes.inverted().transform(
                fig.transFigure.transform((bbox.x0, bbox.y0))
            )

        # Convert that same point from axes coordinates to figure coordinates.
        key_x_fig, key_y_fig = fig.transFigure.inverted().transform(
            ax.transAxes.transform((key_x_ax, key_y_ax))
        )

        padx = 0
        pady = 0
        if "transform" in kwargs:
            padx = 0.05 * bbox.width
            pady = 0.05 * bbox.height

        cax = fig.add_axes(
            [
                key_x_fig + padx,
                key_y_fig - 0.5 * bbox.height + pady,
                bbox.width - key_x_fig,
                bbox.height,
            ],
            zorder=1,
        )

        qk = ax.quiverkey(
            q,
            X=key_x_ax + padx,
            Y=key_y_ax + pady,
            U=key_magnitude,
            label=label,
            labelpos="E",
            coordinates="axes",
            zorder=4,
            fontproperties={"size": 10},
        )

        qk.set_in_layout(True)

        cax.set_frame_on(False)
        cax.set_xticks([])
        cax.set_yticks([])

    plt.sca(ax)
    return ax, q, qk


def colorbar(
    ax: plt.Axes | cgeo.GeoAxes,
    fig: plt.Figure | None = None,
    mappable: ScalarMappable | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    subplots: bool = False,
    adjust: bool = True,
    cax: plt.Axes | None = None,
    drawedges: bool = False,
    extend: str = None,
    cbar_label: str = None,
    ticks: np.ndarray | list = None,
    tick_labels: list[str] | None = None,
) -> Colorbar:
    """
    Add a colorbar to a Cartopy axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axis to add the colorbar to.
    fig : matplotlib.figure.Figure, optional
        The figure to add the colorbar to. Defaults to the current figure.
    mappable : matplotlib.cm.ScalarMappable, optional
        The mappable to create a colorbar for.
    orientation : {"vertical", "horizontal"}, optional
        The orientation of the colorbar. Default is "vertical".
    subplots : bool, optional
        If True, position the colorbar relative to a grid of subplots.
    adjust : bool, optional
        Whether to call ``plt.tight_layout()`` before adding the colorbar.
        Default is True.
    cax : matplotlib.axes.Axes, optional
        The axis to use for the colorbar. If None, a new axis is created.
    drawedges : bool, optional
        Whether to draw edges on the colorbar. Default is False.
    extend : {"neither", "both", "min", "max"}, optional
        How to handle the colorbar extensions. Default is None.
    cbar_label : str, optional
        The label for the colorbar. Default is None.
    ticks : sequence, optional
        The tick positions for the colorbar. Default is None.
    tick_labels : sequence of str, optional
        The labels for the colorbar ticks, placed on the axis matching
        ``orientation``. Default is None.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar instance.
    """

    if cax is None:
        cax = get_cax(
            fig=fig,
            axes=ax,
            orientation=orientation,
            adjust=adjust,
            subplots=subplots,
        )

    cbar = plt.colorbar(
        mappable,
        cax=cax,
        ax=ax,
        orientation=orientation,
        drawedges=drawedges,
        extend=extend,
    )

    if ticks is not None:
        cbar.set_ticks(ticks)

    if tick_labels is not None:
        if orientation == "horizontal":
            cbar.ax.set_xticklabels(tick_labels)
        else:
            cbar.ax.set_yticklabels(tick_labels)

    if cbar_label is not None:
        cbar.set_label(cbar_label)

    plt.sca(ax)
    return cbar


def make_cyclic(obj: xr.DataArray | xr.Dataset, lon: str = "lon"):
    """
    Add a cyclic point to a DataArray along the specified longitude dimension.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The input DataArray or Dataset to which a cyclic point will be added.
    lon : str, optional
        The name of the longitude dimension. Default is "lon".

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The object with a cyclic point added.
    """

    dataset = False

    if isinstance(obj, xr.Dataset) and len(obj.data_vars) > 1:
        raise ValueError(
            "Input object must be a DataArray or a Dataset with only one data variable."
        )

    if isinstance(obj, xr.Dataset):
        obj = list(obj.data_vars.values())[0]
        dataset = True

    if lon not in obj.dims:
        raise ValueError(f"Longitude dimension '{lon}' not found in data dims.")

    attrs = obj.attrs
    cyclic_data, cyclic_dim = add_cyclic_point(obj.values, coord=obj[lon])
    coords = {dim: obj.coords[dim] for dim in obj.dims}
    coords[lon] = cyclic_dim

    new_obj = xr.DataArray(cyclic_data, dims=obj.dims, coords=coords, attrs=attrs)

    if dataset:
        new_obj = new_obj.to_dataset(name=obj.name)

    return new_obj


def significance(
    data: xr.DataArray,
    x: str = "lon",
    y: str = "lat",
    ax: plt.Axes | cgeo.GeoAxes = None,
    level: float = 0.05,
    color: str = "grey",
    alpha: float = 0.3,
    marker: str = None,
    edgecolors: str = None,
    subsample: tuple[int] | list[int] = (1, 1),
    size: float = 0.25,
):
    """
    Plot pointwise significance markers on a Cartopy axis.

    Parameters
    ----------
    data : xarray.DataArray
        The data array containing p-values.
    x : str, optional
        Name of the x dimension in the data. Default is "lon".
    y : str, optional
        Name of the y dimension in the data. Default is "lat".
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes, optional
        The axis to plot on. If None, the current axis is used.
    level : float, optional
        The significance level. Points with p-values below this level are
        plotted. Default is 0.05.
    color : str, optional
        Color of the points to plot. Default is "grey".
    alpha : float, optional
        Alpha transparency of the points. Default is 0.3.
    marker : str, optional
        Marker style for the points. Default is None (default marker).
    edgecolors : str, optional
        Edge color for the points. Default is None (falls back to ``color``).
    subsample : int or tuple of int, optional
        The subsample stride for plotting points to reduce overplotting (x, y).
    size : float, optional
        Size of the points to plot. Default is 0.25.

    Returns
    -------
    matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axis with the significance markers.
    """

    ax = ax or plt.gca()

    transform = _check_cartopy_axis(ax, {})

    if isinstance(subsample, int):
        subsample = (subsample, subsample)

    if x not in data.coords or y not in data.coords:
        x, y = _get_lon_lat(data)

    if len(subsample) > 2:
        raise ValueError("subsample must be a tuple or list with at most 2 elements")

    data = data.isel(
        {
            x: slice(None, None, subsample[0]),
            y: slice(None, None, subsample[1]),
        }
    )

    if isinstance(data, xr.DataArray):
        data = xr.Dataset({"p_value": data})
    else:
        raise ValueError(f"data must be a xr.DataArray object, got {type(data)}")

    pvalues = data.to_dataframe().reset_index()
    pvalues = pvalues.query("p_value < @level")
    pvalues = pvalues.dropna()

    if edgecolors is None:
        edgecolors = color

    ax.scatter(
        pvalues[x],
        pvalues[y],
        color=color,
        alpha=alpha,
        s=size,
        marker=marker,
        edgecolors=edgecolors,
        **transform,
    )
    return ax


def get_cax(
    *,
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    subplots: bool = False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    adjust: bool = True,
) -> plt.Axes:
    """
    Create a new set of axes for a colorbar by stealing space from ``axes``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to which the colorbar axes are added. Defaults to the current
        figure.
    axes : matplotlib.axes.Axes or numpy.ndarray of Axes, optional
        Axes from which space is taken. Defaults to the current axes.
    subplots : bool, optional
        If True, position the colorbar relative to a grid of subplots.
    orientation : {"vertical", "horizontal"}, optional
        Colorbar orientation. Default "vertical".
    adjust : bool, optional
        Whether to call ``plt.tight_layout()`` first. Default is True.

    Returns
    -------
    matplotlib.axes.Axes
        New axes for the colorbar.
    """

    def _has_visible_xtick_labels(ax: plt.Axes) -> bool:
        return any(
            label.get_visible() and bool(label.get_text().strip())
            for label in ax.get_xticklabels()
        )

    def _has_visible_xlabel(ax: plt.Axes) -> bool:
        label = ax.xaxis.label
        return label.get_visible() and bool(ax.get_xlabel().strip())

    def _has_subplots(fig=None):
        axes = [ax for ax in fig.get_axes() if ax.get_label() != "<colorbar>"]
        return len(axes) > 1

    if fig is None:
        fig = plt.gcf()
    if axes is None:
        axes = plt.gca()

    if subplots is False:
        subplots = _has_subplots(fig)

    if subplots and axes is None:
        raise ValueError("If subplots is True, axes and fig must be provided.")

    if adjust:
        plt.tight_layout()

    def _create_cax(y0, x0, y1, x1, x_len, y_len, ax):
        # Vertical uses y0, y_len, x1. Horizontal uses y0, x0, x_len.
        # Hold the bar thickness and gaps constant in inches: a figure-fraction
        # value times the figure size in inches is a physical length, so a fixed
        # fraction grows on larger figures. Scale the short dimension by
        # ref / size. Leave the long dimension (y_len, x_len) unscaled since it
        # tracks the axes extent.

        xticks = pad_quiver_key[0] or _has_visible_xtick_labels(ax)
        xlabel = pad_quiver_key[1] or _has_visible_xlabel(ax)

        ref_width = 5
        ref_height = 4.8

        fig_w, fig_h = fig.get_size_inches()
        scale_w = ref_width / fig_w
        scale_h = ref_height / fig_h

        if orientation == "vertical":
            bottommost = y0
            height = y_len
            rightmost = x1 + 0.04 * scale_w
            width = 0.03 * scale_w
            cax = fig.add_axes([rightmost, bottommost, width, height])

        elif orientation == "horizontal":
            y_pad = 0.05

            if xticks or xlabel:
                pad_quiver_key[0] = True
                pad_quiver_key[1] = True
                y_pad = 0.1

            rightmost = x0
            width = x_len
            bottommost = y0 - y_pad * scale_h
            height = 0.05 * scale_h
            cax = fig.add_axes([rightmost, bottommost, width, height])

        return cax

    if not subplots:
        pos = axes.get_position()
        fig_x_len = pos.x1 - pos.x0
        fig_y_len = pos.y1 - pos.y0
        cax = _create_cax(pos.y0, pos.x0, pos.y1, pos.x1, fig_x_len, fig_y_len, axes)
        plt.sca(axes)
        return cax

    # subplots branch
    if isinstance(axes, plt.Axes):
        nrows, ncols = 1, 1
    elif axes.ndim == 2:
        nrows, ncols = axes.shape
    elif axes.ndim == 1:
        last_ax = fig.axes[-1]
        nrows = last_ax.get_subplotspec().rowspan.stop
        ncols = last_ax.get_subplotspec().colspan.stop
    else:
        raise ValueError("axes must be a single Axes or a 1D/2D array of Axes.")

    axes = np.reshape(axes, (nrows, ncols))
    right_axes = axes[:, -1]  # all rows, last column
    bottom_axes = axes[-1, :]  # last row, all columns

    top_right_ax = right_axes[0].get_position()
    bot_right_ax = right_axes[-1].get_position()
    left_bot_ax = bottom_axes[0].get_position()
    right_bot_ax = bottom_axes[-1].get_position()

    # Vertical colorbar: full grid height, to the right of the last column.
    fig_y_len = top_right_ax.y1 - bot_right_ax.y0

    # Horizontal colorbar: width of a single axis, centered under the grid.
    single_ax_x_len = right_bot_ax.x1 - right_bot_ax.x0
    grid_left = left_bot_ax.x0
    grid_right = right_bot_ax.x1
    horiz_x0 = 0.5 * (grid_left + grid_right) - 0.5 * single_ax_x_len

    cax = _create_cax(
        bot_right_ax.y0,  # y0 (shared)
        horiz_x0,  # x0 for horizontal centering
        top_right_ax.y1,  # y1
        right_bot_ax.x1,  # x1 for vertical positioning
        single_ax_x_len,  # x_len: one axis width (horizontal)
        fig_y_len,  # y_len: full grid height (vertical)
        axes[-1, -1],
    )

    return cax


def _plot_method(data: xr.DataArray, method: str):
    default = data.plot

    methords = ["pcolormesh", "contourf", "contour", "imshow"]
    funcs = [default] + [getattr(default, m) for m in methords]

    pargs = {}
    for f in funcs:
        pargs.update(get_fsig(f))
    if method == "default":
        func = default
    elif method in methords:
        func = getattr(default, method)
        pargs.update(get_fsig(func))

    return func, pargs


def _add_cartopy_features(
    ax: plt.Axes | cgeo.GeoAxes,
    global_extent: bool = False,
    set_extent: tuple[float, float, float, float] = None,
    coastlines: bool = False,
    states: bool = False,
    borders: bool = False,
    lakes: bool = False,
    rivers: bool = False,
    ocean: bool = False,
    land: bool = False,
):

    if global_extent:
        ax.set_global()

    if set_extent:
        ax.set_extent(set_extent)

    features = [
        (coastlines, cfeature.COASTLINE, {}),
        (states, cfeature.STATES, {"linestyle": "-", "alpha": 0.3, "zorder": 3}),
        (borders, cfeature.BORDERS, {"linestyle": "-", "alpha": 0.3, "zorder": 3}),
        (lakes, cfeature.LAKES, {"zorder": 2}),
        (rivers, cfeature.RIVERS, {"zorder": 2}),
    ]

    for condition, feature, kwargs in features:
        if condition:
            ax.add_feature(feature, **kwargs)

    if ocean and not land:
        ax.add_feature(cfeature.LAND, facecolor="#54585f", zorder=2)
    elif land and not ocean:
        ax.add_feature(cfeature.OCEAN, zorder=2)

    return ax


def _resolve_map_aspect(
    da: xr.DataArray = None,
    extent: tuple = None,
    x: str = None,
    y: str = None,
    dim: str = None,
):
    """Resolve lon/lat plot aspect.

    Priority: extent, then coordinate spans, then grid shape.
    """

    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        lon_min %= 360
        lon_max %= 360

        lon_min, lon_max = sorted((lon_min, lon_max))
        return (lon_max - lon_min) / (lat_max - lat_min)
    if da is not None:
        if x in da.coords and y in da.coords:
            x = da[x].values
            y = da[y].values
            return (x.max() - x.min()) / (y.max() - y.min())
        spatial = [d for d in da.dims if d != dim]
        return da.sizes[spatial[-1]] / da.sizes[spatial[-2]]
    raise ValueError("Provide map_aspect, extent, or da to infer aspect.")


def _bottom_left_axis(fg):
    """Return the lowest populated facet in the leftmost column, for any grid shape.

    Handles single-row, single-column, single-facet, and ragged (col_wrap)
    layouts. ``fg.axs`` and ``fg.name_dicts`` are coerced to 2D of shape
    (nrow, ncol); column 0 is the left edge and increasing row index moves
    downward. Returns None if no facet is populated.
    """
    name_dicts = np.atleast_2d(fg.name_dicts)
    axs = np.atleast_2d(fg.axs)
    left_rows = [r for r in range(name_dicts.shape[0]) if name_dicts[r, 0] is not None]
    if not left_rows:
        return None
    return axs[left_rows[-1], 0]


def _facet_figsize(
    col_wrap: int = 1,
    panel_width: float = 5.0,
    cbar_pad_in: float = 0.8,
    da: xr.DataArray = None,
    x: str = None,
    y: str = None,
    dim: str = None,
):
    extent = (
        float(da[x].min()),
        float(da[x].max()),
        float(da[y].min()),
        float(da[y].max()),
    )

    n_facets = da.sizes[dim] if dim else 1
    aspect = _resolve_map_aspect(
        da=da,
        extent=extent,
        x=x,
        y=y,
        dim=dim,
    )
    nrows = int(np.ceil(n_facets / col_wrap))
    panel_height = panel_width / aspect
    width = col_wrap * panel_width
    height = nrows * panel_height + cbar_pad_in
    return width, height


def _get_projection(
    projection: str,
    longitude: xr.DataArray,
    latitude: xr.DataArray,
) -> tuple[Callable, str]:
    lon_w = float(longitude.min().values)
    lon_e = float(longitude.max().values)
    lat_s = float(latitude.min().values)
    lat_n = float(latitude.max().values)

    lon_c = 0.5 * (lon_w + lon_e)
    lat_c = 0.5 * (lat_s + lat_n)

    true_scale_latitude = None
    standard_parallels = None
    cutoff = None

    if not projection:
        d_lon = lon_e - lon_w
        d_lat = lat_n - lat_s

        # Priority:
        # 1. Near-global domains
        # 2. Very wide domains
        # 3. Polar domains
        # 4. Equatorial or cross-equatorial domains
        # 5. Midlatitude regional domains
        if d_lon >= 300.0 and d_lat >= 120.0:
            projection = "Robinson"

        elif d_lon >= 300.0:
            projection = "PlateCarree"

        elif abs(lat_c) >= 70.0 or lat_n >= 85.0 or lat_s <= -85.0:
            if lat_c >= 0.0:
                projection = "NorthPolarStereo"
                true_scale_latitude = float(np.clip(lat_c, 60.0, 89.0))
            else:
                projection = "SouthPolarStereo"
                true_scale_latitude = float(np.clip(lat_c, -89.0, -60.0))

        elif abs(lat_c) <= 25.0 or lat_s * lat_n < 0.0:
            projection = "PlateCarree"

        else:
            projection = "LambertConformal"

            # One-sixth rule; use the tangent case for a thin domain.
            if d_lat < 1.0:
                standard_parallels = (lat_c,)
            else:
                standard_parallels = (
                    lat_s + d_lat / 6.0,
                    lat_n - d_lat / 6.0,
                )

            cutoff = -30.0 if lat_c >= 0.0 else 30.0

    proj_cls: ccrs.Projection = getattr(ccrs, projection)
    available_args = get_fsig(proj_cls)
    cargs = {}

    if "central_longitude" in available_args:
        cargs["central_longitude"] = lon_c

    if "central_latitude" in available_args:
        cargs["central_latitude"] = lat_c

    if "cutoff" in available_args and cutoff is not None:
        cargs["cutoff"] = cutoff

    if "standard_parallels" in available_args and standard_parallels is not None:
        cargs["standard_parallels"] = standard_parallels

    if "true_scale_latitude" in available_args and true_scale_latitude is not None:
        cargs["true_scale_latitude"] = true_scale_latitude

    return proj_cls(**cargs), projection


def _faceted(
    da: xr.DataArray,
    x: str = None,
    y: str = None,
    col: str = None,
    row: str = None,
    col_wrap: int = None,
    figsize: tuple[float, float] = None,
    method: str = None,
    ccrs_proj: str = None,
    cmap: str | LinearSegmentedColormap | ListedColormap = None,
    norm: Any = None,
    vmin: float = None,
    vmax: float = None,
    units: str = None,
    levels: int | list = None,
    extend: str = None,
    robust: bool = False,
    rasterized: bool = False,
    title: str = "",
    orientation: str = None,
    add_colorbar: bool = True,
    drawedges: bool = False,
    cbar_label: str = None,
    global_extent: bool = False,
    set_extent: tuple[float, float, float, float] = None,
    gridlines: bool = False,
    coastlines: bool = True,
    borders: bool = True,
    states: bool = True,
    ocean: bool = True,
    land: bool = True,
    lakes: bool = False,
    rivers: bool = False,
    p_value: xr.DataArray = None,
    pvalue_kwargs: dict = None,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    colorbar_kwargs: dict = None,
    clabel: bool = False,
    clabel_fmt: str = "%1.0f",
    clabel_fontsize: float = 8,
    clabel_inline: bool = True,
    clabel_colors: str = None,
    clabel_kwargs: dict = None,
    quiver_kwargs: dict = None,
    cyclic: bool = False,
    **kwargs,
) -> dict:

    if col is None and row is None:
        raise ValueError(
            "At least one of 'col' or 'row' must be specified for faceting."
        )

    if not x or not y:
        x, y = _get_lon_lat(da)

    add_quiver = (u_component is not None) and (v_component is not None)
    quiver_kwargs = quiver_kwargs or {}
    add_pvalues = p_value is not None
    pvalue_kwargs = pvalue_kwargs or {}

    if add_quiver and quiver_kwargs.get("key_magnitude") is None:
        quiver_kwargs["key_magnitude"] = _get_quiver_key_mag(u_component, v_component)

    long_name = da.attrs.get("long_name", "").title()
    units = units or da.attrs.get("units", da.name)

    proj_obj, proj_name = _get_projection(ccrs_proj, da[x], da[y])
    transform = ccrs.PlateCarree()
    orientation = orientation or "horizontal"

    dim = col or row
    col_wrap = col_wrap or int(np.ceil(np.sqrt(len(da[dim]))))

    # we want all possible args
    plot, pargs = _plot_method(da, method)
    all_args = dict(locals())
    all_args.update(kwargs)

    for k in ("proj_name", "proj_obj"):
        all_args.pop(k)

    pkwargs = {k: v for k, v in all_args.items() if k in pargs}
    pkwargs.update(
        {
            "zorder": 1,
            "transform": transform,
            "rasterized": rasterized,
            "col": col,
            "row": row,
            "col_wrap": col_wrap,
            "subplot_kws": {"projection": proj_obj},
        }
    )

    pkwargs.pop("kwargs", None)
    pkwargs.pop("figsize", None)

    if not cbar_label:
        cbar_label = f"{long_name}\n[{units}]"

    cbar_kwargs = {
        "shrink": 0.5,
        "orientation": orientation,
        "fraction": 0.05,
        "pad": 0.06,
        "label": cbar_label,
    }

    figsize = _facet_figsize(col_wrap=col_wrap, da=da, x=x, y=y, dim=dim)

    fg = plot(figsize=figsize, cbar_kwargs=cbar_kwargs, **pkwargs)

    # One contour set per populated facet; label each from its own artist.
    facet_mappables = list(fg._mappables)
    mappable = facet_mappables[-1]

    if hasattr(fg, "cbar") and fg.cbar is not None:
        fg.cbar.remove()

    key_ax = _bottom_left_axis(fg)

    q_list, qk_list, cb = [], [], None  # initialize in case not added
    populated = -1

    for ax, name_dict in zip(fg.axs.flat, fg.name_dicts.flat):
        if name_dict is None:
            ax.remove()
            continue

        populated += 1

        if clabel and method == "contour":
            facet_mappable = (
                facet_mappables[populated]
                if populated < len(facet_mappables)
                else mappable
            )
            _add_contour_labels(
                ax,
                facet_mappable,
                fmt=clabel_fmt,
                fontsize=clabel_fontsize,
                inline=clabel_inline,
                colors=clabel_colors,
                kwargs=clabel_kwargs,
            )

        ax = _add_cartopy_features(
            ax,
            global_extent,
            set_extent,
            coastlines,
            states,
            borders,
            lakes,
            rivers,
            ocean,
            land,
        )

        if gridlines:
            _add_gridlines(ax)
            if ax is key_ax:
                pad_quiver_key[0] = True
                pad_quiver_key[1] = True

        if add_pvalues:
            significance(
                data=p_value.sel(name_dict),
                ax=ax,
                **pvalue_kwargs,
            )

        if add_quiver:
            ax, q, qk = quiver(
                u=u_component.sel(name_dict),
                v=v_component.sel(name_dict),
                add_key=(ax is key_ax),
                subplots=True,
                ax=ax,
                **quiver_kwargs,
            )
            q_list.append(q)
            qk_list.append(qk)

    if col_wrap == 1:
        orientation = "horizontal"

    cax = get_cax(fig=fg.fig, axes=fg.axs, orientation=orientation, subplots=True)
    cb = fg.fig.colorbar(
        mappable,
        cax=cax,
        orientation=orientation,
        extend=extend,
        drawedges=drawedges,
    )

    if colorbar_kwargs:
        if colorbar_kwargs.get("ticks") is not None:
            cb.set_ticks(colorbar_kwargs["ticks"])

        if colorbar_kwargs.get("tick_labels") is not None:
            if orientation == "horizontal":
                cb.ax.set_xticklabels(colorbar_kwargs["tick_labels"])
            else:
                cb.ax.set_yticklabels(colorbar_kwargs["tick_labels"])

    cb.set_label(cbar_label)

    plt.tight_layout()

    quiver_key = next((qk for qk in qk_list if qk is not None), None)

    return {
        "figure": fg.fig,
        "axes": fg.axs,
        "artist": mappable,
        "facetgrid": fg,
        "colorbar": cb,
        "quiver": q_list if add_quiver else None,
        "quiver_key": quiver_key,
        "projection": proj_name,
    }


def geoplot(
    da: xr.DataArray,
    *,
    x: str = None,
    y: str = None,
    col: str = None,
    row: str = None,
    col_wrap: int = None,
    figsize: tuple[float, float] = None,
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
    ] = None,
    cmap: str | LinearSegmentedColormap | ListedColormap = None,
    norm: Any = None,
    vmin: float = None,
    vmax: float = None,
    units: str = None,
    levels: int | list = None,
    extend: str = None,
    robust: bool = False,
    rasterized: bool = False,
    title: str = "",
    orientation: Literal["vertical", "horizontal"] = None,
    add_colorbar: bool = True,
    drawedges: bool = False,
    cbar_label: str = None,
    global_extent: bool = False,
    set_extent: tuple[float, float, float, float] = None,
    gridlines: bool = False,
    coastlines: bool = True,
    borders: bool = True,
    states: bool = True,
    ocean: bool = True,
    land: bool = True,
    lakes: bool = False,
    rivers: bool = False,
    p_value: xr.DataArray = None,
    pvalue_kwargs: dict = None,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    quiver_kwargs: dict = None,
    colorbar_kwargs: dict = None,
    clabel: bool = False,
    clabel_fmt: str = "%1.0f",
    clabel_fontsize: float = 8,
    clabel_inline: bool = True,
    clabel_colors: str = None,
    clabel_kwargs: dict = None,
    cyclic: bool = False,
    **kwargs,
) -> dict:
    """
    Render the base scalar field of a map and return its primitive artists.

    This is the renderer behind :func:`map`. It draws a two-dimensional or
    faceted (three-dimensional) xarray DataArray on a Cartopy map and returns a
    dictionary of matplotlib and xarray artists. Callers normally use
    :func:`map`, which wraps the returned artists in a :class:`Geoplot`, rather
    than calling this function directly.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field to plot. After ``squeeze()`` the array must be 2D or 3D
        and must contain longitude-latitude coordinates compatible with a
        ``cartopy.crs.PlateCarree()`` data transform.
    x, y : str, optional
        Coordinate names passed to the selected xarray plotting method when
        supported.
    col, row : str, optional
        Faceting coordinate names passed to the selected xarray plotting method
        when supported.
    col_wrap : int, optional
        Number of columns used when wrapping faceted subplots.
    figsize : tuple of float, optional
        Figure size in inches used when creating a new figure.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}, default "default"
        Xarray plotting method used for the scalar field.
    projection : str, default "PlateCarree"
        Cartopy projection used when creating a new axis.
    cmap : str or matplotlib colormap, optional
        Colormap used for the scalar field.
    norm : matplotlib normalization, optional
        Normalization applied to the field.
    vmin, vmax : float, optional
        Lower and upper scalar color limits.
    units : str, optional
        Units used for colorbar labeling. If omitted, inferred from
        ``da.attrs["units"]`` or ``da.name``.
    levels : int or sequence of float, optional
        Contour levels for contour-based methods.
    extend : {"neither", "both", "min", "max"}, optional
        Colorbar extension behavior.
    robust : bool, default False
        Whether to request percentile-based color scaling when supported.
    rasterized : bool, default False
        Whether dense scalar artists should be rasterized when supported.
    title : str, optional
        Plot title.
    orientation : {"vertical", "horizontal"}, optional
        Colorbar orientation.
    add_colorbar : bool, default True
        Whether to add a colorbar when a new axis is created.
    drawedges : bool, default False
        Whether to draw edges between colorbar intervals.
    cbar_label : str, optional
        Explicit colorbar label. If omitted, a label is inferred from metadata.
    global_extent : bool, default False
        If True, set the map extent to the full globe.
    set_extent : tuple of float, optional
        Geographic extent as ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
    gridlines : bool, default False
        Whether to draw labeled longitude and latitude gridlines.
    coastlines, borders, states : bool, default True
        Switches controlling common Cartopy geographic feature overlays.
    ocean, land : bool, default True
        Switches controlling ocean and land background features.
    lakes, rivers : bool, default False
        Switches controlling optional Cartopy inland water feature overlays.
    p_value : xarray.DataArray, optional
        Pointwise p-value field. Values below the selected significance level
        are plotted as markers.
    pvalue_kwargs : dict, optional
        Keyword arguments forwarded to :func:`significance`. Accepted keys are
        ``level: float`` (significance threshold), ``color: str``,
        ``alpha: float``, ``marker: str``, ``edgecolors: str``,
        ``subsample: int | tuple[int, int]``, ``size: float`` (marker size),
        ``x: str`` and ``y: str`` (coordinate names).
    u_component, v_component : xarray.DataArray, optional
        Zonal and meridional vector components for quiver overlays.
    quiver_kwargs : dict, optional
        Keyword arguments forwarded to :func:`quiver`. Accepted keys are
        ``subsample: int | tuple[int, int]``, ``key_magnitude: int | float``
        (reference arrow length), ``key_units: str``, ``x: str`` and ``y: str``
        (coordinate names), plus any argument accepted by
        ``matplotlib.axes.Axes.quiver`` such as ``scale: float``,
        ``color: str`` and ``width: float``. See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    colorbar_kwargs : dict, optional
        Keyword arguments forwarded to :func:`colorbar`. Accepted keys are
        ``ticks: sequence`` and ``tick_labels: sequence of str``. The
        orientation, edges, extension and label of the base colorbar are set
        through the top-level ``orientation``, ``drawedges``, ``extend`` and
        ``cbar_label`` arguments and must not be repeated here. See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
    clabel : bool, default False
        Label line contours. Ignored for ``method="contourf"``.
    clabel_fmt : str, default "%1.0f"
        Contour-label format.
    clabel_fontsize : float, default 8
        Contour-label font size.
    clabel_inline : bool, default True
        Draw contour labels inline.
    clabel_colors : str, optional
        Contour-label color.
    clabel_kwargs : dict, optional
        Additional arguments forwarded to ``Axes.clabel``.
    cyclic : bool, default False
        If True, append a cyclic longitude point before plotting. The longitude
        dimension is assumed to be named ``"lon"``.
    **kwargs
        Additional keyword arguments forwarded to the selected xarray plotting
        method after signature filtering.

    Returns
    -------
    dict
        Mapping with keys ``figure``, ``axes``, ``artist``, ``facetgrid``,
        ``colorbar``, ``quiver``, ``quiver_key``, ``method``, ``base_artists`` and
        ``contour_labels``.

    Notes
    -----
    Input coordinates are plotted with a ``cartopy.crs.PlateCarree()``
    transform. The display projection is controlled by ``projection``.
    """

    if not x or not y:
        x, y = _get_lon_lat(da)

    if cyclic:
        da = make_cyclic(da, lon=x)

    da = to_lon180(da, lon=x)

    da = da.squeeze()

    if da.ndim == 3:
        # Only ``da`` is mutated above, so every other local is a valid
        # ``_faceted`` parameter. Drop the collected ``**kwargs`` dict and
        # re-spread it so user-supplied plotting keywords are forwarded intact.
        faceted_args = {k: v for k, v in locals().items() if k != "kwargs"}
        return _faceted(**faceted_args, **kwargs)

    if da.ndim > 3:
        raise ValueError(
            f"Expected a 2D or 3D DataArray after squeezing; got dimensions {da.dims!r}."
        )

    add_quiver = (u_component is not None) and (v_component is not None)
    quiver_kwargs = quiver_kwargs or {}
    add_pvalues = p_value is not None
    pvalue_kwargs = pvalue_kwargs or {}

    long_name = da.attrs.get("long_name", "").title()
    units = units or da.attrs.get("units", da.name)

    proj_obj, proj_name = _get_projection(projection, da[x], da[y])
    fig, ax = plt.subplots(subplot_kw={"projection": proj_obj}, figsize=figsize)
    transform = ccrs.PlateCarree()

    ax = _add_cartopy_features(
        ax,
        global_extent,
        set_extent,
        coastlines,
        states,
        borders,
        lakes,
        rivers,
        ocean,
        land,
    )

    # we want all possible args
    plot, pargs = _plot_method(da, method)
    all_args = dict(locals())
    all_args.update(kwargs)

    pkwargs = {k: v for k, v in all_args.items() if k in pargs}
    pkwargs.update(
        {
            "ax": ax,
            "add_colorbar": False,
            "zorder": 1,
            "transform": transform,
            "rasterized": rasterized,
        }
    )

    for k in ("col", "row", "col_wrap", "figsize", "kwargs", "proj_name", "proj_obj"):
        pkwargs.pop(k, None)

    mp = plot(**pkwargs)
    if method in ["contour", "contourf"]:
        if hasattr(mp, "set_edgecolor"):
            mp.set_edgecolor("face")

        if hasattr(mp, "set_rasterized"):
            mp.set_rasterized(rasterized)
        else:
            for c in mp.collections:
                c.set_rasterized(rasterized)

    if clabel and method == "contour":
        _add_contour_labels(
            ax,
            mp,
            fmt=clabel_fmt,
            fontsize=clabel_fontsize,
            inline=clabel_inline,
            colors=clabel_colors,
            kwargs=clabel_kwargs,
        )

    plt.title(title)

    if gridlines:
        _add_gridlines(ax)
        pad_quiver_key[0] = True
        pad_quiver_key[1] = True

    q, qk, cb = None, None, None  # initialize in case not added

    if add_pvalues:
        ax = significance(data=p_value, ax=ax, **pvalue_kwargs)

    if add_quiver:
        ax, q, qk = quiver(
            u=u_component,
            v=v_component,
            ax=ax,
            **quiver_kwargs,
        )

    if add_colorbar:
        if cbar_label is None:
            cbar_label = f"{long_name}\n[{units}]"

        orientation = orientation or "vertical"
        colorbar_kwargs = colorbar_kwargs or {}

        cb = colorbar(
            fig=fig,
            ax=ax,
            mappable=mp,
            orientation=orientation,
            drawedges=drawedges,
            extend=extend,
            cbar_label=cbar_label,
            adjust=False,
            **colorbar_kwargs,
        )
    return {
        "figure": fig,
        "axes": ax,
        "artist": mp,
        "facetgrid": None,
        "colorbar": cb,
        "quiver": q,
        "quiver_key": qk,
        "projection": proj_name,
    }


def _ffmpeg_encode(
    input_pattern, outfile, fps, session_tmp_dir, user_path, error
) -> int:

    try:
        cmd = [
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
            outfile,
        ]

        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        error = 1
        print("ERROR:", e.stderr)
    finally:
        shutil.rmtree(session_tmp_dir, ignore_errors=True)

    return error


def _map_wrapper(
    i: int,
    dim_value: Any,
    dim: str,
    title: str,
    dpi: int,
    session_tmp_dir: Path,
    data_slice: xr.DataArray,
    u_slice: xr.DataArray,
    v_slice: xr.DataArray,
    args: dict,
):

    # if dim value is numpy datetime, the string representation is too long,
    # so we shorten it to the date only
    if np.issubdtype(type(dim_value), np.datetime64):
        dim_value = pd.to_datetime(dim_value).strftime("%Y-%m-%d %H:%M")

    if title:
        title = f"{title}\n{dim}: {dim_value}"
    else:
        title = f"{dim}: {dim_value}"

    local_kwargs = args.copy()
    local_kwargs["da"] = data_slice
    local_kwargs["u_component"] = u_slice
    local_kwargs["v_component"] = v_slice

    fname = session_tmp_dir / f"{i:06d}.png"

    plot = geo(**{k: v for k, v in local_kwargs.items() if k in get_fsig(geoplot)})

    faceted = local_kwargs.get("col") or local_kwargs.get("row")
    if faceted is None:
        plot.axes.set_title(title)
    else:
        plot.figure.suptitle(title)

    plt.savefig(fname, dpi=dpi, bbox_inches="tight")

    plot.figure.clear()
    plt.close(plot.figure)

    return None


def _validate_animation_inputs(
    dim: str,
    data: xr.DataArray,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
):

    if not isinstance(data, xr.DataArray):
        raise ValueError("data must be an xarray.DataArray")
    if not isinstance(u_component, (xr.DataArray, type(None))):
        raise ValueError("u_component must be an xarray.DataArray or None")
    if not isinstance(v_component, (xr.DataArray, type(None))):
        raise ValueError("v_component must be an xarray.DataArray or None")

    if dim not in data.dims:
        raise ValueError(f"{dim} not found in data.dims {data.dims}")

    for name, da in {
        "u_component": u_component,
        "v_component": v_component,
    }.items():
        if da is not None and dim not in da.dims:
            raise ValueError(f"{dim} not found in {name}.dims {da.dims}")
    if (u_component is None) != (v_component is None):
        raise ValueError("u_component and v_component must be provided together")
    if u_component is not None:
        assert u_component.size == v_component.size == data.size, (
            "u_component, v_component, and data must have the same size"
        )
    data = data.sortby(dim)
    if u_component is not None:
        u_component = u_component.sortby(dim)
        v_component = v_component.sortby(dim)
        assert data[dim].equals(u_component[dim])
        assert data[dim].equals(v_component[dim])

    return data, u_component, v_component


def animate(
    da: xr.DataArray,
    dim: str = "time",
    *,
    x: str = None,
    y: str = None,
    col: str = None,
    row: str = None,
    col_wrap: int = None,
    figsize: tuple[float, float] = None,
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
    ] = None,
    cmap: str | LinearSegmentedColormap | ListedColormap = None,
    norm: Any = None,
    vmin: float = None,
    vmax: float = None,
    units: str = None,
    levels: int | list = None,
    extend: str = None,
    robust: bool = False,
    rasterized: bool = False,
    title: str = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    add_colorbar: bool = True,
    drawedges: bool = False,
    cbar_label: str = None,
    global_extent: bool = False,
    set_extent: tuple[float, float, float, float] = None,
    gridlines: bool = False,
    coastlines: bool = True,
    borders: bool = True,
    states: bool = True,
    ocean: bool = True,
    land: bool = True,
    lakes: bool = False,
    rivers: bool = False,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    colorbar_kwargs: dict = None,
    quiver_kwargs: dict = None,
    cyclic: bool = False,
    indices: tuple | list | np.ndarray = None,
    outfile: Path = None,
    quality: Literal["low", "medium", "high"] = "medium",
    fps: int = 1,
    parallel: bool = True,
    **kwargs,
) -> DisplayHandle | None:
    """
    Render a map animation from an xarray DataArray and encode it as MP4.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field to animate. The animation dimension must be present in
        ``da.dims``.
    dim : str, default "time"
        Dimension used for animation frames.
    x, y : str, optional
        Coordinate names passed to the selected xarray plotting method when
        supported.
    col, row : str, optional
        Faceting coordinate names passed to the selected xarray plotting method
        when supported.
    col_wrap : int, optional
        Number of columns used when wrapping faceted subplots.
    figsize : tuple of float, optional
        Figure size in inches for each rendered frame.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow", "scatter"}, default "default"
        Xarray plotting method used for the scalar field.
    projection : str, default "PlateCarree"
        Cartopy projection used for each frame.
    cmap : str or matplotlib colormap, optional
        Colormap used for the scalar field.
    norm : matplotlib normalization, optional
        Normalization applied to the field.
    vmin, vmax : float, optional
        Lower and upper scalar color limits. Fixed limits are recommended for
        temporal comparisons.
    units : str, optional
        Units used for colorbar labeling. If omitted, inferred from
        ``da.attrs["units"]`` or ``da.name``.
    levels : int or sequence of float, optional
        Contour levels for contour-based methods.
    extend : {"neither", "both", "min", "max"}, optional
        Colorbar extension behavior.
    robust : bool, default False
        Whether to request percentile-based color scaling when supported.
    rasterized : bool, default False
        Whether dense scalar artists should be rasterized when supported.
    title : str, optional
        Base title passed to frame plotting routines.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Colorbar orientation for non-faceted frames.
    add_colorbar : bool, default True
        Whether to add a colorbar.
    drawedges : bool, default False
        Whether to draw edges between colorbar intervals.
    cbar_label : str, optional
        Explicit colorbar label. If omitted, a label is inferred from metadata..
    global_extent : bool, default False
        If True, set each map extent to the full globe.
    set_extent : tuple of float, optional
        Geographic extent as ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
    gridlines : bool, default False
        Whether to draw labeled longitude and latitude gridlines.
    coastlines, borders, states : bool, default True
        Switches controlling common Cartopy geographic feature overlays.
    ocean, land : bool, default True
        Switches controlling ocean and land background features.
    lakes, rivers : bool, default False
        Switches controlling optional Cartopy inland water feature overlays.
    u_component, v_component : xarray.DataArray, optional
        Zonal and meridional vector components for quiver overlays. Both must
        contain ``dim`` and align with ``da`` along that dimension.
    colorbar_kwargs : dict, optional
        Keyword arguments forwarded to :func:`colorbar`. Accepted keys are
        ``ticks: sequence`` and ``tick_labels: sequence of str``. The
        orientation, edges, extension and label of the base colorbar are set
        through the top-level ``orientation``, ``drawedges``, ``extend`` and
        ``cbar_label`` arguments and must not be repeated here. See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
    quiver_kwargs : dict, optional
        Keyword arguments forwarded to :func:`quiver`. Accepted keys are
        ``subsample: int | tuple[int, int]``, ``key_magnitude: int | float``
        (reference arrow length), ``key_units: str``, ``x: str`` and ``y: str``
        (coordinate names), plus any argument accepted by
        ``matplotlib.axes.Axes.quiver`` such as ``scale: float``,
        ``color: str`` and ``width: float``. See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    cyclic : bool, default False
        If True, append a cyclic longitude point before plotting each frame.
        The longitude dimension is assumed to be named ``"lon"``.
    indices : tuple of int, list of int, or numpy.ndarray, optional
        Positional indices along ``dim`` to render. If omitted, all positions
        are rendered.
    outfile : str or pathlib.Path, optional
        Output path for the MP4 animation. If omitted, a temporary output path
        is used.
    quality : {"low", "medium", "high"}, default "medium"
        Frame-resolution preset used during PNG rendering.
    fps : int, default 1
        Frames per second passed to ffmpeg.
    parallel : bool, default True
        Whether to render frames with multiprocessing.
    **kwargs
        Additional keyword arguments forwarded to the selected xarray plotting
        method after signature filtering.

    Returns
    -------
    IPython.display.DisplayHandle or None
        Inside a Jupyter kernel, the encoded MP4 is embedded and its display
        handle is returned. Otherwise the MP4 is written to ``outfile`` and a
        ``RuntimeError`` is raised if encoding fails.

    Notes
    -----
    Animation output requires an ``ffmpeg`` executable on the system path.
    Parallel rendering can reduce wall-clock time but increases memory use
    because multiple data slices and figures may be active concurrently.
    """

    args = AttrDict(locals())

    # pop non-plotting controls from args
    outfile = args.pop("outfile")
    fps = args.pop("fps")
    dim = args.pop("dim")
    quality = args.pop("quality")
    parallel = args.pop("parallel")
    indices = args.pop("indices")
    title = args.pop("title")
    da = args.pop("da")
    u_component = args.pop("u_component")
    v_component = args.pop("v_component")

    da, u_component, v_component = _validate_animation_inputs(
        dim, da, u_component, v_component
    )

    session_tmp_dir = Path(tempfile.mkdtemp())
    dpi_map = {"low": 300, "medium": 600, "high": 1200}
    dpi = dpi_map.get(quality, 600)

    def sel(da: xr.DataArray | None, i: int):
        return None if da is None else da.isel({dim: i})

    if indices is None:
        indices = range(da.sizes[dim])
    tasks = [
        (
            i,
            da[dim][i].values,
            dim,
            title,
            dpi,
            session_tmp_dir,
            sel(da, i),
            sel(u_component, i),
            sel(v_component, i),
            args,
        )
        for i in indices
    ]

    if parallel:
        processes = min(len(indices), n_cpus // 2)

        delayed_tasks = [dask.delayed(_map_wrapper)(*task) for task in tasks]

        with DaskProgressBar():
            dask.compute(*delayed_tasks, scheduler="processes", num_workers=processes)

    else:
        tasks = SerialProgressBar(tasks, total=len(tasks))
        for task in tasks:
            _map_wrapper(*task)

    # ---- ffmpeg encode (MP4 only) ----

    user_path = False
    if not outfile:
        outfile = Path(tmp / f"{uuid.uuid4().hex}/{uuid.uuid4().hex}.mp4")
    else:
        outfile = Path(outfile)
        user_path = True

    outfile.parent.mkdir(parents=True, exist_ok=True)
    input_pattern = str(Path(session_tmp_dir) / "%06d.png")

    error = _ffmpeg_encode(input_pattern, outfile, fps, session_tmp_dir, user_path, 0)

    if error == 0 and user_path:
        print(f"Animation saved to : {outfile}")

    # optional inline display (Jupyter)
    if "ipykernel" in sys.modules and error == 0:
        from IPython.display import Video, display

        return display(
            Video(
                outfile,
                embed=True,
                html_attributes="controls autoplay loop",
                width=800,
                height=600,
            )
        )

    raise RuntimeError("Animation encoding failed")


class Geoplot:
    """Composable Cartopy map returned by :func:`map`.

    ``Geoplot`` stores the base figure and artists and exposes chainable
    overlay methods through :attr:`add`. Scalar overlays, scatter points,
    vectors, significance markers, and additional colorbars are applied to
    every populated facet automatically.

    Parameters
    ----------
    primitives : dict
        Renderer output produced by :func:`geoplot`.

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        Figure containing the map.
    axes : matplotlib.axes.Axes, cartopy.mpl.geoaxes.GeoAxes or numpy.ndarray
        Single map axis or faceted array of map axes.
    artist : matplotlib.artist.Artist
        Base plotting artist.
    facetgrid : xarray.plot.FacetGrid or None
        Xarray facet grid for faceted maps.
    colorbar : matplotlib.colorbar.Colorbar or None
        Colorbar attached to the base field or most recently assigned
        colorbar.
    layers : list of dict
        Overlay records in call order.
    add : Adder
        Namespace containing the chainable overlay methods.

    Notes
    -----
    Every registered line-contour set is promoted above all other artists
    after each overlay call. Consequently, a contour base may be followed
    by ``contourf``, ``pcolormesh``, ``imshow``, ``scatter``, ``quiver``, or
    significance layers without hiding the contour lines.
    """

    def __init__(self, primitives: dict):
        """Initialize a composable map from renderer primitives."""
        self.figure: Figure | None = primitives["figure"]
        self.axes: Axes | cgeo.GeoAxes | np.ndarray | None = primitives["axes"]
        self.projection: str | None = primitives["projection"]
        self.artist: Artist | None = primitives["artist"]
        self.facetgrid: xr.plot.facetgrid.FacetGrid | None = primitives["facetgrid"]
        self.colorbar: Colorbar | None = primitives["colorbar"]
        self.quiver: Quiver | list[Quiver] | None = primitives["quiver"]
        self.quiver_key: QuiverKey | list[QuiverKey] | None = primitives["quiver_key"]
        self.layers: list[dict] = []

        self.add = Adder(self)

    def _iter_axes(self):
        """Yield ``(axis, selector)`` pairs for populated map axes."""
        if self.facetgrid is None:
            yield self.axes, {}
            return

        for ax, selector in zip(
            self.facetgrid.axs.flat,
            self.facetgrid.name_dicts.flat,
        ):
            if selector is not None:
                yield ax, selector

    @staticmethod
    def _select(da: xr.DataArray, selector: dict) -> xr.DataArray:
        """Return the facet slice of an overlay field."""
        return da.sel(selector).squeeze() if selector else da.squeeze()

    def __repr__(self) -> str:
        """Return a compact summary of the stored artists and overlays."""
        return _map_repr(self)


class Adder:
    """Add overlay layers to a :class:`Geoplot`."""

    __slots__ = ("_map",)

    def __init__(self, Geoplot: Geoplot):
        self._map = Geoplot

    @staticmethod
    def _drop_facet_kwargs(kwargs: dict) -> None:
        """Remove facet options inherited from the base map."""
        for key in ("col", "row", "col_wrap"):
            kwargs.pop(key, None)

    def contour(
        self,
        da: xr.DataArray,
        *,
        x: str = None,
        y: str = None,
        method: Literal["contour", "contourf"] = "contour",
        levels: int | list = None,
        colors: str | list = None,
        cmap: str | LinearSegmentedColormap | ListedColormap = None,
        linewidths: float | list = None,
        linestyles: str | list = None,
        alpha: float = None,
        vmin: float = None,
        vmax: float = None,
        norm: Any = None,
        extend: str = None,
        zorder: int = 2,
        add_labels: bool = False,
        clabel: bool = False,
        clabel_fmt: str = "%1.0f",
        clabel_fontsize: float = 8,
        clabel_inline: bool = True,
        clabel_colors: str = None,
        clabel_kwargs: dict = None,
        **kwargs,
    ) -> Geoplot:
        """Add a line or filled-contour overlay and return the parent map.

        Parameters
        ----------
        da : xarray.DataArray
            Field to contour. For a faceted base map, the field must contain
            the corresponding facet coordinates.
        x, y : str, optional
            Horizontal coordinate names.
        method : {"contour", "contourf"}, default "contour"
            Draw line contours or filled contours.
        levels : int or sequence of float, optional
            Number of contour levels or explicit contour levels.
        colors : str or sequence of str, optional
            Contour colors. This is mutually exclusive with ``cmap``.
        cmap : str or matplotlib colormap, optional
            Colormap used when ``colors`` is not supplied.
        linewidths : float or sequence of float, optional
            Contour line widths.
        linestyles : str or sequence of str, optional
            Contour line styles.
        alpha : float, optional
            Artist opacity.
        vmin, vmax : float, optional
            Lower and upper color limits.
        norm : matplotlib.colors.Normalize, optional
            Color normalization.
        extend : {"neither", "both", "min", "max"}, optional
            Out-of-range color handling.
        zorder : int, default 2
            Drawing order relative to other artists.
        add_labels : bool, default False
            Allow xarray to add axis labels and a title.
        clabel : bool, default False
            Label line contours. Ignored for ``method="contourf"``.
        clabel_fmt : str, default "%1.0f"
            Contour-label format.
        clabel_fontsize : float, default 8
            Contour-label font size.
        clabel_inline : bool, default True
            Draw contour labels inline.
        clabel_colors : str, optional
            Contour-label color.
        clabel_kwargs : dict, optional
            Additional arguments forwarded to ``Axes.clabel``.
        **kwargs
            Additional arguments forwarded to the selected xarray contour
            plotting method. Facet-layout arguments are ignored because the
            overlay inherits the base map layout.

        Returns
        -------
        Geoplot
            The parent map, to allow chaining.
        """
        if method not in ("contour", "contourf"):
            raise ValueError("method must be 'contour' or 'contourf'.")

        if not x or not y:
            x, y = _get_lon_lat(da)

        self._drop_facet_kwargs(kwargs)

        options = {
            "levels": levels,
            "colors": colors,
            "cmap": cmap,
            "linewidths": linewidths,
            "linestyles": linestyles,
            "alpha": alpha,
            "vmin": vmin,
            "vmax": vmax,
            "norm": norm,
            "extend": extend,
            "zorder": zorder,
            "x": x,
            "y": y,
        }
        options = {key: value for key, value in options.items() if value is not None}
        options["add_labels"] = add_labels
        options.update(kwargs)

        artists = []
        labels = []
        transform = ccrs.PlateCarree()

        for ax, selector in self._map._iter_axes():
            field = self._map._select(da, selector)
            artist = getattr(field.plot, method)(
                ax=ax,
                transform=transform,
                add_colorbar=False,
                **options,
            )
            artists.append(artist)

            if clabel and method == "contour":
                labels.append(
                    _add_contour_labels(
                        ax,
                        artist,
                        fmt=clabel_fmt,
                        fontsize=clabel_fontsize,
                        inline=clabel_inline,
                        colors=clabel_colors,
                        kwargs=clabel_kwargs,
                    )
                )

        self._map.layers.append(
            {
                "kind": method,
                "artists": artists,
                "labels": labels,
            }
        )
        return self._map

    def quiver(
        self,
        u: xr.DataArray,
        v: xr.DataArray,
        *,
        x: str = "lon",
        y: str = "lat",
        subsample: tuple[int, int] | int = (1, 1),
        scale: float = None,
        key_magnitude: int | float = None,
        key_units: str = None,
        color: str = None,
        width: float = None,
        **kwargs,
    ) -> Geoplot:
        """Add a vector overlay and return the parent map.

        Parameters
        ----------
        u, v : xarray.DataArray
            Zonal and meridional vector components.
        x, y : str, default "lon", "lat"
            Horizontal coordinate names.
        subsample : int or tuple of int, default (1, 1)
            Grid stride used to thin arrows.
        scale : float, optional
            Quiver scale. Larger values shorten arrows.
        key_magnitude : int or float, optional
            Reference magnitude for the quiver key. When omitted, it is
            derived from the vector field and shared across facets.
        key_units : str, optional
            Units displayed by the quiver key.
        color : str, optional
            Arrow color.
        width : float, optional
            Arrow-shaft width.
        **kwargs
            Additional arguments forwarded to :func:`quiver`. Facet-layout
            arguments are ignored because the overlay inherits the base map
            layout.

        Returns
        -------
        Geoplot
            The parent map, to allow chaining.
        """

        subplots = self._map.facetgrid is not None
        key_ax = _bottom_left_axis(self._map.facetgrid) if subplots else self._map.axes

        if key_magnitude is None:
            key_magnitude = _get_quiver_key_mag(u, v)

        self._drop_facet_kwargs(kwargs)

        options = {
            "scale": scale,
            "key_units": key_units,
            "color": color,
            "width": width,
        }
        options = {key: value for key, value in options.items() if value is not None}
        options.update(kwargs)

        quivers = []
        keys = []

        for ax, selector in self._map._iter_axes():
            _, quiver_artist, quiver_key = quiver(
                u=self._map._select(u, selector),
                v=self._map._select(v, selector),
                x=x,
                y=y,
                subsample=subsample,
                key_magnitude=key_magnitude,
                add_key=ax is key_ax,
                subplots=subplots,
                ax=ax,
                **options,
            )
            quivers.append(quiver_artist)
            keys.append(quiver_key)

        self._map.quiver = quivers
        self._map.quiver_key = next(
            (key for key in keys if key is not None),
            None,
        )
        self._map.layers.append(
            {
                "kind": "quiver",
                "artists": quivers,
                "keys": keys,
            }
        )
        return self._map

    def significance(
        self,
        pvalues: xr.DataArray,
        *,
        level: float = 0.05,
        color: str = "grey",
        alpha: float = 0.3,
        marker: str = None,
        edgecolors: str = None,
        subsample: tuple[int, int] | int = (1, 1),
        size: float = 0.25,
        x: str = "lon",
        y: str = "lat",
    ) -> Geoplot:
        """Add a pointwise significance overlay and return the parent map.

        Parameters
        ----------
        pvalues : xarray.DataArray
            Pointwise p-values.
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
            Grid stride used to thin markers.
        size : float, default 0.25
            Marker size.
        x, y : str, default "lon", "lat"
            Horizontal coordinate names.

        Returns
        -------
        Geoplot
            The parent map, to allow chaining.
        """
        artists = []

        for ax, selector in self._map._iter_axes():
            artists.append(
                significance(
                    data=self._map._select(pvalues, selector),
                    ax=ax,
                    x=x,
                    y=y,
                    level=level,
                    color=color,
                    alpha=alpha,
                    marker=marker,
                    edgecolors=edgecolors,
                    subsample=subsample,
                    size=size,
                )
            )

        self._map.layers.append(
            {
                "kind": "pvalues",
                "artists": artists,
            }
        )
        return self._map

    def colorbar(
        self,
        mappable: Artist = None,
        *,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        drawedges: bool = False,
        extend: str = None,
        cbar_label: str = None,
        ticks: np.ndarray | list = None,
        tick_labels: list[str] = None,
    ) -> Geoplot:
        """Attach a colorbar and return the parent map.

        Parameters
        ----------
        mappable : matplotlib artist, optional
            Artist described by the colorbar. Defaults to the base map artist.
        orientation : {"vertical", "horizontal"}, default "vertical"
            Colorbar orientation.
        drawedges : bool, default False
            Draw edges between color intervals.
        extend : {"neither", "both", "min", "max"}, optional
            Out-of-range colorbar extension.
        cbar_label : str, optional
            Colorbar label.
        ticks : sequence, optional
            Explicit tick positions.
        tick_labels : sequence of str, optional
            Explicit tick labels.

        Returns
        -------
        Geoplot
            The parent map, to allow chaining.
        """
        if mappable is None:
            mappable = self._map.artist

        colorbar_artist = colorbar(
            fig=self._map.figure,
            ax=self._map.axes,
            mappable=mappable,
            subplots=self._map.facetgrid is not None,
            orientation=orientation,
            drawedges=drawedges,
            extend=extend,
            cbar_label=cbar_label,
            ticks=ticks,
            tick_labels=tick_labels,
        )

        self._map.colorbar = colorbar_artist
        self._map.layers.append(
            {
                "kind": "colorbar",
                "artists": [colorbar_artist],
            }
        )
        return self._map


def _map_repr(obj: Geoplot) -> str:
    parts = []

    if obj.projection is not None:
        parts.append(f"projection={obj.projection}")

    for name in (
        "figure",
        "axes",
        "artist",
        "facetgrid",
        "colorbar",
        "quiver",
        "quiver_key",
    ):
        value = getattr(obj, name)
        if value is None:
            continue

        if isinstance(value, (np.ndarray, list)):
            seq = value.ravel().tolist() if isinstance(value, np.ndarray) else value
            count = value.size if isinstance(value, np.ndarray) else len(value)
            elem = next(
                (type(x).__name__ for x in seq if x is not None),
                "object",
            )
            parts.append(f"{name}={count} {elem}(s)")
        else:
            parts.append(f"{name}={type(value).__name__}")

    if obj.layers:
        parts.append(f"layers={len(obj.layers)}")

    return f"Map({', '.join(parts)})"


def geo(
    da: xr.DataArray,
    *,
    x: str = None,
    y: str = None,
    col: str = None,
    row: str = None,
    col_wrap: int = None,
    figsize: tuple[float, float] = None,
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
    ] = None,
    cmap: str | LinearSegmentedColormap | ListedColormap = None,
    norm: Any = None,
    vmin: float = None,
    vmax: float = None,
    units: str = None,
    levels: int | list = None,
    extend: str = None,
    robust: bool = False,
    rasterized: bool = False,
    title: str = "",
    orientation: Literal["vertical", "horizontal"] = None,
    add_colorbar: bool = True,
    drawedges: bool = False,
    cbar_label: str = None,
    global_extent: bool = False,
    set_extent: tuple[float, float, float, float] = None,
    gridlines: bool = False,
    coastlines: bool = True,
    borders: bool = True,
    states: bool = True,
    ocean: bool = True,
    land: bool = True,
    lakes: bool = False,
    rivers: bool = False,
    p_value: xr.DataArray = None,
    pvalue_kwargs: dict = None,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    quiver_kwargs: dict = None,
    colorbar_kwargs: dict = None,
    clabel: bool = False,
    clabel_fmt: str = "%1.0f",
    clabel_fontsize: float = 8,
    clabel_inline: bool = True,
    clabel_colors: str = None,
    clabel_kwargs: dict = None,
    cyclic: bool = False,
    **kwargs,
) -> Geoplot:
    """
    Draw a scalar field on a Cartopy map and return a composable :class:`Geoplot`.

    This is the public entry point. It renders a two-dimensional or faceted
    (three-dimensional) DataArray as the base layer and returns a
    :class:`Geoplot` whose ``add.contour``, ``add.quiver``, ``add.significance``
    and ``add.colorbar`` methods add overlays. The full parameter list is
    declared explicitly so that editors expose every option. The class is named
    ``Geoplot`` to avoid shadowing the builtin inside this module; this callable
    is exposed as ``climtools.plot.map``.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field to plot. After ``squeeze()`` the array must be 2D or 3D
        and must contain longitude-latitude coordinates compatible with a
        ``cartopy.crs.PlateCarree()`` data transform.
    x, y : str, optional
        Coordinate names passed to the selected xarray plotting method.
    col, row : str, optional
        Faceting coordinate names. Supplying either produces a faceted plot.
    col_wrap : int, optional
        Number of columns used when wrapping faceted subplots.
    figsize : tuple of float, optional
        Figure size in inches used when creating a new figure.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
        Xarray plotting method used for the scalar field.
    projection : str, default "PlateCarree"
        Cartopy projection used when creating the axes.
    cmap : str or matplotlib colormap, optional
        Colormap used for the scalar field.
    norm : matplotlib normalization, optional
        Normalization applied to the field.
    vmin, vmax : float, optional
        Lower and upper scalar color limits.
    units : str, optional
        Units used for colorbar labeling. If omitted, inferred from
        ``da.attrs["units"]`` or ``da.name``.
    levels : int or sequence of float, optional
        Contour levels for contour-based methods.
    extend : {"neither", "both", "min", "max"}, optional
        Colorbar extension behavior.
    robust : bool, default False
        Whether to request percentile-based color scaling.
    rasterized : bool, default False
        Whether dense scalar artists should be rasterized.
    title : str, optional
        Plot title for single-axis plots.
    orientation : {"vertical", "horizontal"}, optional
        Colorbar orientation.
    add_colorbar : bool, default True
        Whether to add a colorbar for the base field.
    drawedges : bool, default False
        Whether to draw edges between colorbar intervals.
    cbar_label : str, optional
        Explicit colorbar label. If omitted, a label is inferred from metadata.
    global_extent : bool, default False
        If True, set the map extent to the full globe.
    set_extent : tuple of float, optional
        Geographic extent as ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
    gridlines : bool, default False
        Whether to draw labeled longitude and latitude gridlines.
    coastlines, borders, states : bool, default True
        Switches controlling common Cartopy geographic feature overlays.
    ocean, land : bool, default True
        Switches controlling ocean and land background features.
    lakes, rivers : bool, default False
        Switches controlling optional Cartopy inland water feature overlays.
    p_value : xarray.DataArray, optional
        Pointwise p-value field. Values below the significance level are marked.
    pvalue_kwargs : dict, optional
        Keyword arguments forwarded to :func:`significance`. Accepted keys are
        ``level: float`` (significance threshold), ``color: str``,
        ``alpha: float``, ``marker: str``, ``edgecolors: str``,
        ``subsample: int | tuple[int, int]``, ``size: float`` (marker size),
        ``x: str`` and ``y: str`` (coordinate names).
    u_component, v_component : xarray.DataArray, optional
        Zonal and meridional vector components for a base quiver overlay.
    quiver_kwargs : dict, optional
        Keyword arguments forwarded to :func:`quiver`. Accepted keys are
        ``subsample: int | tuple[int, int]``, ``key_magnitude: int | float``
        (reference arrow length), ``key_units: str``, ``x: str`` and ``y: str``
        (coordinate names), plus any argument accepted by
        ``matplotlib.axes.Axes.quiver`` such as ``scale: float``,
        ``color: str`` and ``width: float``. See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    colorbar_kwargs : dict, optional
        Keyword arguments forwarded to :func:`colorbar`. Accepted keys are
        ``ticks: sequence`` and ``tick_labels: sequence of str``. The
        orientation, edges, extension and label of the base colorbar are set
        through the top-level ``orientation``, ``drawedges``, ``extend`` and
        ``cbar_label`` arguments and must not be repeated here. See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
    clabel : bool, default False
        Label line contours. Ignored for ``method="contourf"``.
    clabel_fmt : str, default "%1.0f"
        Contour-label format.
    clabel_fontsize : float, default 8
        Contour-label font size.
    clabel_inline : bool, default True
        Draw contour labels inline.
    clabel_colors : str, optional
        Contour-label color.
    clabel_kwargs : dict, optional
        Additional arguments forwarded to ``Axes.clabel``.
    cyclic : bool, default False
        If True, append a cyclic longitude point before plotting. The longitude
        dimension is assumed to be named ``"lon"``.
    **kwargs
        Additional keyword arguments forwarded to the selected xarray plotting
        method after signature filtering.

    Returns
    -------
    Geoplot
        Composable map holding the base artists, with chainable overlay methods.

    Notes
    -----
    Input coordinates are plotted with a ``cartopy.crs.PlateCarree()`` transform.
    The display projection is controlled by ``projection``.
    """

    params = dict(locals())
    extra = params.pop("kwargs")
    return Geoplot(geoplot(**params, **extra))
