import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as cgeo
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point
from dask import compute, delayed
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from metpy.units import units as metpy_units

from .tools import AttrDict, get_fsig, ipykernel, n_cpus, tmp

__all__ = ["get_cax", "pvalues", "quiver", "plt", "ccrs"]


@dataclass(frozen=True, repr=False)
class MapPlot:
    fig: Figure
    ax: Axes | cgeo.GeoAxes | np.ndarray
    artist: Artist

    def __repr__(self) -> str:
        if isinstance(self.ax, np.ndarray):
            ax_repr = f"{self.ax.size} axes"
        else:
            ax_repr = type(self.ax).__name__
        _repr = f"MapPlot(Figure={type(self.fig).__name__}, Axes={ax_repr}, Artist={type(self.artist).__name__})"
        return _repr


def get_cax(
    *,
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    subplots: bool = False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    **kwargs,
) -> plt.Axes:
    """
    Create a new set of axes for a colorbar by stealing space from the current axes.
    This is useful for adding a colorbar to a plot without overlapping the existing axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        The figure to which the colorbar axes will be added. If None, uses the current figure.
    ax : matplotlib.axes.Axes, optional
        The axes from which space will be stolen. If None, uses the current axes.
    subplots : bool, optional
        If True, the function will adjust the colorbar position based on the subplots in the figure.
        This is useful when the figure has multiple subplots and you want to ensure the colorbar does not overlap with them.
        if True, the axes and fig must be provided and will be used to determine the position of the colorbar.

    orientation : str, optional
        The orientation of the colorbar. Can be either "vertical" or "horizontal". Default is "vertical".
    Returns
    -------
    matplotlib.axes.Axes
        The new axes for the colorbar.
    """

    if subplots and axes is None:
        raise ValueError("If subplots is True, axes and fig must be provided.")

    if fig is None:
        fig = plt.gcf()
    if axes is None:
        axes = plt.gca()

    plt.tight_layout()
    fig_width, fig_height = fig.get_size_inches()

    def _create_cax(y0, x0, y1, x1, x_len, y_len, ax):
        if orientation == "vertical":
            bottommost = y0
            height = y_len
            rightmost = x1 + 0.04
            width = 0.03

            cax = fig.add_axes([rightmost, bottommost, width, height])

        elif orientation == "horizontal":
            xticks = True if len(list(ax.get_xticks())) > 0 else False
            xlabel = bool(ax.xaxis.label.get_text().strip())
            pad = 0.10
            h_pad = 0
            if (xlabel and not xticks) or (xticks and not xlabel):
                pad = 0.15
            elif xticks and xlabel:
                pad = 0.20

            if kwargs.get("quiver") and kwargs.get("xaxis_ticks"):
                pad = 0.1
                h_pad = 0.05

            if kwargs.get("quiver") and not kwargs.get("xaxis_ticks"):
                pad = 0.05
                h_pad = 0.05

            rightmost = x0
            width = x_len
            bottommost = y0 - pad
            height = 0.05 + h_pad

            cax = fig.add_axes([rightmost, bottommost, width, height])

        return cax

    if not subplots:
        pos = axes.get_position()
        fig_x_len = pos.x1 - pos.x0
        fig_y_len = pos.y1 - pos.y0
        cax = _create_cax(pos.y0, pos.x0, pos.y1, pos.x1, fig_x_len, fig_y_len, axes)

    elif subplots:
        nrows, ncols = 1, 1

        if isinstance(axes, plt.Axes):
            nrows, ncols = 1, 1

        elif axes.ndim == 2:
            nrows, ncols = axes.shape
        elif axes.ndim == 1:
            # Need to ask figure
            last_ax = fig.axes[-1]
            nrows = last_ax.get_subplotspec().rowspan.stop
            ncols = last_ax.get_subplotspec().colspan.stop

        axes = np.reshape(axes, (nrows, ncols))
        right_axes = axes[:, -1]  # All rows, last column
        bottom_axes = axes[-1, :]  # Last row, all columns

        top_right_ax = right_axes[0].get_position()
        bot_right_ax = right_axes[-1].get_position()
        left_bot_ax = bottom_axes[0].get_position()
        right_bot_ax = bottom_axes[-1].get_position()

        fig_x_len = right_bot_ax.x1 - left_bot_ax.x0
        fig_y_len = top_right_ax.y1 - bot_right_ax.y0

        cax = _create_cax(
            bot_right_ax.y0,
            left_bot_ax.x0,
            top_right_ax.y1,
            right_bot_ax.x1,
            fig_x_len,
            fig_y_len,
            axes[-1, -1],
        )

    return cax


def make_cyclic(obj: xr.DataArray | xr.Dataset, lon: str = "lon"):
    """
    Add a cyclic point to a DataArray along the specified longitude dimension.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The input DataArray or Dataset to which a cyclic point will be added.
    lon : str, optional
        The name of the longitude dimension in the DataArray. Default is 'lon'.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The DataArray or Dataset with a cyclic point added.

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


def _check_cartopy_axis(ax):

    transform = None
    if isinstance(ax, cgeo.GeoAxes):
        transform = ccrs.PlateCarree()

    return transform


def plot_pvalues(
    data: xr.DataArray,
    ax: plt.Axes | cgeo.GeoAxes = None,
    level: float = 0.05,
    color: str = "grey",
    alpha: float = 0.3,
    marker: str = None,
    edgecolors: str = None,
    subsample: int = 1,
    s: float = 0.25,
):
    """
    Plot p-values on a Cartopy axis.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot`
        The Cartopy axis to plot on.
    data : xarray.DataArray
        The data array containing p-values.
    level : float, optional
        The significance level to use for plotting. Points with p-values below this level will be plotted
    color : str, optional
        Color of the points to plot. Default is "grey".
    alpha : float, optional
        Alpha transparency of the points. Default is 0.05.
    subsample : int, optional
        subsample size for plotting points to reduce overplotting. Default is 1 (plot all points).
    marker : str, optional
        Marker style for the points. Default is None (default marker).
    edgecolors : str, optional
        Edge color for the points. Default is None.
    s : float, optional
        Size of the points to plot. Default is 1.
    """

    if ax is None:
        ax = plt.gca()

    transform = _check_cartopy_axis(ax)

    if "lon" not in data.dims or "lat" not in data.dims:
        raise ValueError("DataArray must contain 'lon' and 'lat' dimensions.")

    data = data.isel(lat=slice(None, None, subsample), lon=slice(None, None, subsample))
    p_values = data.to_dataframe(name="p_values").reset_index()
    p_values = p_values.query("p_values < @level")
    p_values = p_values.dropna()

    if edgecolors is None:
        edgecolors = color

    ax.scatter(
        p_values["lon"],
        p_values["lat"],
        transform=transform,
        color=color,
        alpha=alpha,
        s=s,
        marker=marker,
        edgecolors=edgecolors,
    )
    return ax


def get_units(units: str):
    try:
        u = metpy_units(units)
        return f"{u.units:~^P}"
    except Exception:
        return units


def escape_matplotlib_label(text: str) -> str:
    """
    Escape special characters in a string for use in Matplotlib labels.
    """
    escape_map = {
        "\\": r"\\",
        "$": r"\$",
        "%": r"\%",
        "_": r"\_",
        "^": r"\^{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "#": r"\#",
    }

    return "".join(escape_map.get(char, char) for char in text)


def get_label(
    long_name: str,
    units: str,
    cbar_label: str = None,
    format: bool = True,
) -> str:

    if format:
        units = get_units(units)

    units = escape_matplotlib_label(units)

    if not cbar_label:
        label = rf"${long_name}$" + "\n" + rf"[${units}$]"
    else:
        label = rf"${cbar_label}$" + "\n" + rf"[${units}$]"
    return label


def plot_quiver(
    u: xr.DataArray,
    v: xr.DataArray,
    ax: plt.Axes | cgeo.GeoAxes = None,
    subsample: int = 1,
    new_ax: bool = False,
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
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes, optional
        The axis to plot on. If None, the current axis is used.
    subsample : int, optional
        The subsample size for plotting points to reduce overplotting. Default is 1.

    **kwargs
        Additional keyword arguments for the quiver plot.

    Returns
    -------
    matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axis with the quiver plot.

    """

    # pop quiver key from kwargs,
    units = kwargs.pop("units", None)
    if units is None:
        u_units = u.attrs.get("units", "")
        v_units = v.attrs.get("units", "")
        assert u_units == v_units, "units of u and v components must match"
        units = u_units

    if ax is None:
        ax = plt.gca()

    transform = _check_cartopy_axis(ax)

    u = u[::subsample, ::subsample]
    v = v[::subsample, ::subsample]

    lon = u.lon
    lat = u.lat

    # Detect if coordinates are already 2D
    if lon.ndim == 2 and lat.ndim == 2:
        lon2d, lat2d = lon.values, lat.values
    else:
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)

    xaxis_ticks = kwargs.pop("xaxis_ticks")
    U = kwargs.pop("U", None)

    Q = ax.quiver(
        lon2d, lat2d, u.values, v.values, transform=transform, angles="xy", **kwargs
    )

    if not U:
        speed = (u**2 + v**2) ** 0.5
        U = np.round(speed.mean(skipna=True).values)
        U = int(U)

    if new_ax:
        cax = get_cax(
            axes=ax,
            orientation="horizontal",
            quiver=True,
            xaxis_ticks=xaxis_ticks,
        )
        bbox = cax.get_position()

        label = get_label(U, units)

        ax.quiverkey(
            Q,
            X=0.45,
            Y=bbox.y0,
            U=U,
            label=label,
            labelpos="E",
            coordinates="figure",
            fontproperties={"size": 14},
        )

        cax.set_frame_on(False)
        cax.set_xticks([])
        cax.set_yticks([])

    return ax


def plot_cbar(
    fig: plt.Figure,
    ax: plt.Axes | cgeo.GeoAxes,
    artist: Artist,
    orientation: str = "vertical",
    drawedges: bool = False,
    extend: str = None,
    cbar_label: str = "",
):
    """
    Add a colorbar to a Cartopy axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the colorbar to.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axis to add the colorbar to.
    artist : matplotlib.artist.Artist
        The artist to create a colorbar for.
    orientation : str, optional
        The orientation of the colorbar. Default is "vertical".
    drawedges : bool, optional
        Whether to draw edges on the colorbar. Default is False.
    extend : str, optional
        How to handle the colorbar extensions. Default is None.
    cbar_label : str, optional
        The label for the colorbar. Default is "".

    Returns
    -------
    matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        The axis with the colorbar.

    """
    cax = get_cax(fig=fig, axes=ax, orientation=orientation)

    cb = plt.colorbar(
        artist,
        cax=cax,
        ax=ax,
        orientation=orientation,
        drawedges=drawedges,
        extend=extend,
    )

    cb.set_label(cbar_label)
    return ax


def _add_map_features(
    fig: plt.Figure,
    ax: plt.Axes | cgeo.GeoAxes,
    artist: Artist,
    # colorbar
    add_colorbar: bool = False,
    cbar_label: str = None,
    orientation: str = "vertical",
    drawedges: bool = False,
    extend: str = None,
    # p-values
    p_values: xr.DataArray = None,
    p_value_kwargs: dict = None,
    # quiver
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    quiver_kwargs: dict = None,
    # meta
    long_name: str = "",
    units: str = "",
    gridlines: bool = False,
    new_ax: bool = False,
):

    if gridlines:
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

    if add_colorbar and new_ax:
        cbar_label = get_label(long_name, units, cbar_label)

        ax = plot_cbar(
            fig,
            ax,
            artist,
            orientation=orientation,
            drawedges=drawedges,
            extend=extend,
            cbar_label=cbar_label,
        )

    if p_values is not None:
        p_kwargs = p_value_kwargs or {}
        ax = plot_pvalues(data=p_values, ax=ax, **p_kwargs)

    if (u_component is not None) and (v_component is not None):
        q_kwargs = quiver_kwargs or {}
        q_kwargs["xaxis_ticks"] = gridlines

        ax = plot_quiver(
            u=u_component,
            v=v_component,
            ax=ax,
            new_ax=new_ax,
            **q_kwargs,
        )

    return ax


def _data_plot(data: xr.DataArray, pt: str):
    p = data.plot

    pts = ["pcolormesh", "contourf", "contour", "imshow"]
    funcs = [p] + [getattr(p, m) for m in pts]

    pargs = {}
    for f in funcs:
        pargs.update(get_fsig(f))
    if pt == "default":
        func = p
    elif pt in pts:
        func = getattr(p, pt)
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


def faceted_mapplot(
    da: xr.DataArray,
    dim: str,
    figsize: tuple[float, float] = None,
    # Spatial configuration
    x: str = None,
    y: str = None,
    shape: tuple[int, int] = None,
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
    set_extent: tuple[float, float, float, float] = None,
    # Plot appearance
    method: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow"
    ] = "default",
    cmap: str | LinearSegmentedColormap | ListedColormap = None,
    norm: Any = None,
    vmin: float = None,
    vmax: float = None,
    units: str = None,
    levels: int | list = None,
    extend: str = None,
    cyclic: bool = False,
    robust: bool = False,
    rasterized: bool = False,
    title: str = "",
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
    lakes: bool = False,
    rivers: bool = False,
    p_values: xr.DataArray = None,
    p_value_kwargs: dict = None,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    quiver_kwargs: dict = None,
    **kwargs,
) -> MapPlot:
    """
    Plot one Cartopy map panel for each value along a DataArray dimension.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field to plot. After ``squeeze()``, the array must have at most
        three dimensions.
    dim : str
        Dimension used to define map panels.
    figsize : tuple of float, optional
        Figure size in inches.
    x, y : str, optional
        Coordinate names passed to the selected xarray plotting method when
        supported.
    shape : tuple of int, optional
        Facet grid shape as ``(nrows, ncols)``. If omitted, a near-square grid
        is inferred.
    projection : str, default "PlateCarree"
        Cartopy projection used for all panels.
    central_longitude, central_latitude : float, optional
        Projection-center arguments passed to the Cartopy projection
        constructor when supported.
    global_extent : bool, default False
        If True, set each map extent to the full globe.
    set_extent : tuple of float, optional
        Geographic extent as ``(lon_min, lon_max, lat_min, lat_max)`` in
        degrees.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
        Xarray plotting method used for each scalar-field panel.
    cmap : str or matplotlib colormap, optional
        Colormap used for the scalar field.
    norm : Any, optional
        Matplotlib normalization object.
    vmin, vmax : float, optional
        Lower and upper scalar color limits. If omitted, symmetric limits are
        inferred from the full faceted field.
    units : str, optional
        Units used for colorbar labeling. If omitted, inferred from
        ``da.attrs["units"]`` or ``da.name``.
    levels : int or sequence of float, optional
        Contour levels for contour-based methods.
    extend : {"neither", "both", "min", "max"}, optional
        Colorbar extension behavior.
    cyclic : bool, default False
        If True, append a cyclic longitude point before plotting. The longitude
        dimension is assumed to be named ``"lon"``.
    rasterized : bool, default False
        Whether dense scalar artists should be rasterized when supported.
    title : str, optional
        Figure-level title.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Colorbar orientation. The shared faceted colorbar is horizontal.
    add_colorbar : bool, default True
        Whether to add a shared colorbar.
    drawedges : bool, default False
        Whether to draw edges between colorbar intervals.
    cbar_label : str, optional
        Explicit colorbar label. If omitted, a label is inferred from metadata.
    gridlines : bool, default False
        Whether to draw labeled longitude and latitude gridlines.
    coastlines, borders, states, lakes, rivers : bool
        Switches controlling Cartopy geographic feature overlays.
    ocean, land : bool
        Switches controlling ocean and land background features.
    p_values : xarray.DataArray, optional
        Pointwise p-value field used for significance markers.
    p_value_kwargs : dict, optional
        Keyword arguments forwarded to ``plot_pvalues``.
    u_component, v_component : xarray.DataArray, optional
        Zonal and meridional vector components for quiver overlays. These must
        include ``dim``.
    quiver_kwargs : dict, optional
        Keyword arguments forwarded to ``plot_quiver``. For faceted vector
        plots, this must include ``"U"`` and ``"subsample"``.
    **kwargs
        Additional keyword arguments forwarded to the selected xarray plotting
        method after signature filtering.

    Returns
    -------
    MapPlot
        Container with ``fig``, ``ax``, and ``artist`` attributes. Here ``ax``
        is an array of Cartopy GeoAxes.

    Notes
    -----
    One panel is rendered for each coordinate value along ``dim``. Empty axes
    are hidden when the requested grid contains more panels than data slices.
    """

    # if data is 3D, raise error
    da = da.squeeze()
    long_name = da.attrs.get("long_name", "").capitalize()
    units = units or da.attrs.get("units", da.name)
    robust = False  # Force

    if da.ndim > 3:
        raise ValueError("DataArray has more than 3 dimensions.")

    if cyclic:
        da = make_cyclic(da, lon="lon")

    plot_quivers = False
    if u_component is not None and v_component is not None:
        plot_quivers = True

    proj = getattr(ccrs, projection)
    _cargs = get_fsig(proj)
    cargs = {}

    if central_longitude is not None and "central_longitude" in _cargs:
        cargs["central_longitude"] = central_longitude
    if central_latitude is not None and "central_latitude" in _cargs:
        cargs["central_latitude"] = central_latitude

    if shape is None:
        ncols = int(np.ceil(np.sqrt(da.sizes[dim])))
        nrows = int(np.ceil(da.sizes[dim] / ncols))
    else:
        nrows, ncols = shape

    fig, axes = plt.subplots(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        subplot_kw={"projection": proj(**cargs)},
        # constrained_layout=True,
    )

    axes = np.asarray(axes).ravel()

    cbar_mins = []
    cbar_maxs = []
    artist = None

    for i in range(len(da[dim])):
        ax = axes[i]
        i_da = da.isel({dim: i})
        if vmin is None or vmax is None:
            cbar_mins.append(da.isel({dim: i}).min(skipna=True).values)
            cbar_maxs.append(da.isel({dim: i}).max(skipna=True).values)

        ax = _add_cartopy_features(
            **{
                k: v
                for k, v in locals().items()
                if k in get_fsig(_add_cartopy_features)
            }
        )

        transform = ccrs.PlateCarree()

        # xarray methods

        # we want all possible args
        plot, pargs = _data_plot(i_da, method)
        all_args = dict(locals())
        all_args.update(kwargs)

        pkwargs = {k: v for k, v in all_args.items() if k in pargs}
        pkwargs["ax"] = ax
        pkwargs["add_colorbar"] = False
        pkwargs["zorder"] = 1
        pkwargs["transform"] = transform
        pkwargs["rasterized"] = rasterized
        del pkwargs["extend"]
        del pkwargs["kwargs"]
        del pkwargs["figsize"]
        del pkwargs["rasterized"]

        artist = plot(**pkwargs, extend=extend)

        if method in ["contour", "contourf"]:
            artist.set_edgecolor("face")
            for c in artist.collections:
                c.set_rasterized(rasterized)

        ax.set_title(f"{dim}: {da[dim][i].values}")

        if plot_quivers:
            u_component_i = u_component.isel({dim: i})
            v_component_i = v_component.isel({dim: i})

            quiver_kwargs = quiver_kwargs or {}
            if "U" not in quiver_kwargs:
                raise ValueError(
                    "quiver_kwargs must include 'U' and 'subsample' keys when plotting quiver overlays on faceted maps."
                )
        else:
            u_component_i = None
            v_component_i = None

        ax = _add_map_features(
            fig,
            ax,
            artist,
            add_colorbar=False,
            cbar_label=None,
            orientation=orientation,
            drawedges=drawedges,
            extend=extend,
            p_values=p_values,
            p_value_kwargs=p_value_kwargs,
            u_component=u_component_i,
            v_component=v_component_i,
            quiver_kwargs=quiver_kwargs,
            long_name=long_name,
            units=units,
            gridlines=gridlines,
            new_ax=True,
        )

    for ax in axes[len(da[dim]) :]:
        ax.set_visible(False)

    fig_width, fig_height = fig.get_size_inches()

    fig.suptitle(title)

    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.12,
        top=0.94,
        wspace=0.08,
        hspace=0.01,
    )

    if vmin is None or vmax is None:
        vlim = np.nanmax(np.abs([np.nanmin(cbar_mins), np.nanmax(cbar_maxs)]))
        vmin, vmax = -vlim, vlim

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    _sm = plt.cm.ScalarMappable(norm=norm, cmap=artist.cmap)
    _sm.set_array([])

    cb = fig.colorbar(
        _sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.06,
        extend=extend,
    )
    cb.set_label(get_label(long_name, units, cbar_label))

    if plot_quivers:
        from matplotlib.quiver import QuiverKey

        qkey = None

        qkeys = [
            child
            for _ax in np.asarray(axes).ravel()
            for child in _ax.get_children()
            if isinstance(child, QuiverKey)
        ]

        if qkeys:
            qkey = qkeys[0]

            for k in qkeys:
                k.remove()

        cb_bbox = cb.ax.get_position()

        qkcax = fig.add_axes(
            [
                0.0,  # x0: start at left edge of figure
                cb_bbox.y0,  # y0: align with colorbar bottom
                cb_bbox.x0,  # width: from fig x0 to colorbar x0
                cb_bbox.height,  # height: match colorbar height
            ],
            frameon=False,
        )
        qkcax.quiverkey(
            qkey.Q,
            X=0.02 + axes.flat[0].get_position().xmin,
            Y=cb_bbox.y0,
            U=qkey.U,
            label=qkey.label,
            labelpos=qkey.labelpos,
            coordinates="figure",
            fontproperties={"size": 14},
        )

        qkcax.set_frame_on(False)
        qkcax.set_xticks([])
        qkcax.set_yticks([])

    return MapPlot(fig, axes, artist)


def mapplot(
    da: xr.DataArray,
    *,
    # Spatial configuration
    x: str = None,
    y: str = None,
    ax: cgeo.GeoAxes = None,
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
    set_extent: tuple[float, float, float, float] = None,
    figsize: tuple[float, float] = None,
    # Plot appearance
    method: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow"
    ] = "default",
    cmap: str | LinearSegmentedColormap | ListedColormap = None,
    norm: Any = None,
    vmin: float = None,
    vmax: float = None,
    units: str = None,
    levels: int | list = None,
    extend: str = None,
    cyclic: bool = False,
    robust: bool = False,
    rasterized: bool = False,
    title: str = "",
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
    lakes: bool = False,
    rivers: bool = False,
    p_values: xr.DataArray = None,
    p_value_kwargs: dict = None,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    quiver_kwargs: dict = None,
    **kwargs,
) -> MapPlot:
    """
    Plot a two-dimensional xarray DataArray on a Cartopy map.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field to plot. After ``squeeze()``, the array must be
        two-dimensional and must contain longitude-latitude coordinates
        compatible with a ``cartopy.crs.PlateCarree()`` data transform.
    x, y : str, optional
        Coordinate names passed to the selected xarray plotting method when
        supported.
    ax : cartopy.mpl.geoaxes.GeoAxes, optional
        Existing Cartopy axis. If omitted, a new figure and projected axis are
        created.
    projection : str, default "PlateCarree"
        Cartopy projection used when creating a new axis.
    central_longitude, central_latitude : float, optional
        Projection-center arguments passed to the Cartopy projection
        constructor when supported.
    global_extent : bool, default False
        If True, set the map extent to the full globe.
    set_extent : tuple of float, optional
        Geographic extent as ``(lon_min, lon_max, lat_min, lat_max)`` in
        degrees.
    figsize : tuple of float, optional
        Figure size in inches used when creating a new figure.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
        Xarray plotting method used for the scalar field.
    cmap : str or matplotlib colormap, optional
        Colormap used for the scalar field.
    norm : Any, optional
        Matplotlib normalization object.
    vmin, vmax : float, optional
        Lower and upper scalar color limits.
    units : str, optional
        Units used for colorbar labeling. If omitted, inferred from
        ``da.attrs["units"]`` or ``da.name``.
    levels : int or sequence of float, optional
        Contour levels for contour-based methods.
    extend : {"neither", "both", "min", "max"}, optional
        Colorbar extension behavior.
    cyclic : bool, default False
        If True, append a cyclic longitude point before plotting. The longitude
        dimension is assumed to be named ``"lon"``.
    robust : bool, default False
        Whether to request percentile-based color scaling when supported by
        xarray.
    rasterized : bool, default False
        Whether dense scalar artists should be rasterized when supported.
    title : str, optional
        Plot title.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Colorbar orientation.
    add_colorbar : bool, default True
        Whether to add a colorbar when a new axis is created.
    drawedges : bool, default False
        Whether to draw edges between colorbar intervals.
    cbar_label : str, optional
        Explicit colorbar label. If omitted, a label is inferred from metadata.
    gridlines : bool, default False
        Whether to draw labeled longitude and latitude gridlines.
    coastlines, borders, states, lakes, rivers : bool
        Switches controlling Cartopy geographic feature overlays.
    ocean, land : bool
        Switches controlling ocean and land background features.
    p_values : xarray.DataArray, optional
        Pointwise p-value field. Values below the selected significance level
        are plotted as markers.
    p_value_kwargs : dict, optional
        Keyword arguments forwarded to ``plot_pvalues``.
    u_component, v_component : xarray.DataArray, optional
        Zonal and meridional vector components for quiver overlays.
    quiver_kwargs : dict, optional
        Keyword arguments forwarded to ``plot_quiver``.
    **kwargs
        Additional keyword arguments forwarded to the selected xarray plotting
        method after signature filtering.

    Returns
    -------
    MapPlot
        Container with ``fig``, ``ax``, and ``artist`` attributes.

    Notes
    -----
    Input coordinates are plotted with a ``cartopy.crs.PlateCarree()``
    transform. The display projection is controlled by ``projection``.
    """

    da = da.squeeze()

    long_name = da.attrs.get("long_name", "").capitalize()
    units = units or da.attrs.get("units", da.name)

    if da.ndim > 2:
        raise ValueError("DataArray has more than 2 dimensions.")

    if cyclic:
        da = make_cyclic(da, lon="lon")

    if ax:
        new_ax = False
        if not isinstance(ax, (cgeo.GeoAxes)):
            raise ValueError("Provided ax must be a cartopy GeoAxes.")
        fig = ax.get_figure()

    if not ax:
        new_ax = True
        proj = getattr(ccrs, projection)
        _cargs = get_fsig(proj)
        cargs = {}

        if central_longitude is not None and "central_longitude" in _cargs:
            cargs["central_longitude"] = central_longitude
        if central_latitude is not None and "central_latitude" in _cargs:
            cargs["central_latitude"] = central_latitude

        fig, ax = plt.subplots(
            subplot_kw={"projection": proj(**cargs)},
            figsize=figsize,
        )

        ax = _add_cartopy_features(
            **{
                k: v
                for k, v in locals().items()
                if k in get_fsig(_add_cartopy_features)
            }
        )

    transform = ccrs.PlateCarree()

    # we want all possible args
    plot, pargs = _data_plot(da, method)
    all_args = dict(locals())
    all_args.update(kwargs)

    pkwargs = {k: v for k, v in all_args.items() if k in pargs}
    pkwargs["ax"] = ax
    pkwargs["add_colorbar"] = False
    pkwargs["zorder"] = 1
    pkwargs["transform"] = transform
    pkwargs["rasterized"] = rasterized
    del pkwargs["extend"]
    del pkwargs["kwargs"]
    del pkwargs["figsize"]
    del pkwargs["rasterized"]

    artist = plot(**pkwargs, extend=extend)

    if method in ["contour", "contourf"]:
        artist.set_edgecolor("face")
        artist.set_rasterized(rasterized)

    plt.title(title)

    ax = _add_map_features(
        **{k: v for k, v in locals().items() if k in get_fsig(_add_map_features)}
    )

    return MapPlot(fig, ax, artist)


def ffmpeg_encode(input_pattern, outfile, fps, session_tmp_dir, user_path, error):

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
    else:
        return None


def _mapplot_i(
    i: int,
    dim_value: Any,
    dim: str,
    facet_dim: str,
    dpi: int,
    session_tmp_dir: Path,
    data_slice: xr.DataArray,
    u_slice: xr.DataArray,
    v_slice: xr.DataArray,
    args: dict,
):

    # if dim values is numpy datetime,the string representation is too long, so we shorten it to the date only
    if np.issubdtype(type(dim_value), np.datetime64):
        dim_value = pd.to_datetime(dim_value).strftime("%Y-%m-%d %H:%M")

    title = f"{dim}: {dim_value}"

    local_kwargs = args.copy()
    local_kwargs["da"] = data_slice
    local_kwargs["u_component"] = u_slice
    local_kwargs["v_component"] = v_slice

    fname = session_tmp_dir / f"{i:06d}.png"

    if local_kwargs.get("faceted", False):
        local_kwargs["dim"] = facet_dim
        plot = faceted_mapplot(
            **{k: v for k, v in local_kwargs.items() if k in get_fsig(faceted_mapplot)}
        )

        plot.fig.suptitle(title)
    else:
        plot = mapplot(
            **{k: v for k, v in local_kwargs.items() if k in get_fsig(mapplot)}
        )
        plot.ax.set_title(title)

    plt.savefig(fname, dpi=dpi, bbox_inches="tight")

    plot.fig.clear()
    plt.close(plot.fig)

    return None


def _validate_animation_inputs(
    dim: str,
    data: xr.DataArray,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    faceted: bool = False,
    faceted_dim: str = None,
    shape: tuple = None,
):

    if faceted and faceted_dim is None:
        raise ValueError("faceted_dim must be provided when faceted=True")
    if faceted and shape is None:
        raise ValueError("shape must be provided when faceted=True")

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
    # Animation control will be popped from args
    dim: str = "time",
    indices: tuple | list | np.ndarray = None,
    outfile: Path = None,
    quality: Literal["low", "medium", "high"] = "medium",
    fps: int = 1,
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
    set_extent: tuple[float, float, float, float] = None,
    figsize: tuple[float, float] = None,
    faceted: bool = False,
    faceted_dim: str = None,
    shape: tuple = None,
    central_longitude: float = None,
    central_latitude: float = None,
    # Plot appearance
    method: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow"
    ] = "default",
    cmap: str | LinearSegmentedColormap | ListedColormap = None,
    norm: Any = None,
    vmin: float = None,
    vmax: float = None,
    units: str = None,
    levels: int | list = None,
    extend: str = None,
    cyclic: bool = False,
    rasterized: bool = False,
    robust: bool = False,
    title: str = "",
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
    lakes: bool = False,
    rivers: bool = False,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    quiver_kwargs: dict = None,
    **kwargs,
):
    """
    Render a map animation from an xarray DataArray and encode it as MP4.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field to animate. The animation dimension must be present in
        ``da.dims``.
    dim : str, default "time"
        Dimension used for animation frames.
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
    x, y : str, optional
        Coordinate names passed to the selected xarray plotting method when
        supported.
    projection : str, default "PlateCarree"
        Cartopy projection used for each frame.
    global_extent : bool, default False
        If True, set each map extent to the full globe.
    set_extent : tuple of float, optional
        Geographic extent as ``(lon_min, lon_max, lat_min, lat_max)`` in
        degrees.
    figsize : tuple of float, optional
        Figure size in inches for each rendered frame.
    faceted : bool, default False
        If True, render a faceted map for each animation frame.
    faceted_dim : str, optional
        Faceting dimension used within each frame when ``faceted=True``.
    shape : tuple of int, optional
        Facet grid shape as ``(nrows, ncols)`` when ``faceted=True``.
    central_longitude, central_latitude : float, optional
        Projection-center arguments passed to the Cartopy projection
        constructor when supported.
    method : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
        Xarray plotting method used for the scalar field.
    cmap : str or matplotlib colormap, optional
        Colormap used for the scalar field.
    norm : Any, optional
        Matplotlib normalization object.
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
    cyclic : bool, default False
        If True, append a cyclic longitude point before plotting each frame.
        The longitude dimension is assumed to be named ``"lon"``.
    rasterized : bool, default False
        Whether dense scalar artists should be rasterized when supported.
    robust : bool, default False
        Whether to request percentile-based color scaling when supported by
        xarray.
    title : str, optional
        Base title passed to frame plotting routines.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Colorbar orientation for non-faceted frames.
    add_colorbar : bool, default True
        Whether to add a colorbar.
    drawedges : bool, default False
        Whether to draw edges between colorbar intervals.
    cbar_label : str, optional
        Explicit colorbar label. If omitted, a label is inferred from metadata.
    gridlines : bool, default False
        Whether to draw labeled longitude and latitude gridlines.
    coastlines, borders, states, lakes, rivers : bool
        Switches controlling Cartopy geographic feature overlays.
    ocean, land : bool
        Switches controlling ocean and land background features.
    u_component, v_component : xarray.DataArray, optional
        Zonal and meridional vector components for quiver overlays. Both must
        contain ``dim`` and align with ``da`` along that dimension.
    quiver_kwargs : dict, optional
        Keyword arguments forwarded to ``plot_quiver``.
    **kwargs
        Additional keyword arguments forwarded to the selected xarray plotting
        method after signature filtering.

    Returns
    -------
    None
        Temporary PNG frames are rendered, encoded with ffmpeg, and removed.

    Notes
    -----
    Animation output requires an ``ffmpeg`` executable on the system path.
    Parallel rendering can reduce wall-clock time but increases memory use
    because multiple data slices and figures may be active concurrently.
    """

    args = AttrDict(locals())

    # pop the above from args
    outfile = args.pop("outfile")
    fps = args.pop("fps")
    dim = args.pop("dim")
    quality = args.pop("quality")
    parallel = args.pop("parallel")
    indices = args.pop("indices")
    da = args.pop("da")
    u_component = args.pop("u_component")
    v_component = args.pop("v_component")
    facet_dim = args.pop("faceted_dim")

    da, u_component, v_component = _validate_animation_inputs(
        dim, da, u_component, v_component, faceted, facet_dim, shape
    )

    session_tmp_dir = Path(tempfile.mkdtemp())
    dpi_map = {"low": 300, "medium": 600, "high": 1200}
    dpi = dpi_map.get(quality, 600)

    def _isel_dim(da: xr.DataArray | None, i: int):
        return None if da is None else da.isel({dim: i})  # .load()

    if indices is None:
        indices = range(da.sizes[dim])
    tasks = [
        (
            i,
            da[dim][i].values,
            dim,
            facet_dim,
            dpi,
            session_tmp_dir,
            _isel_dim(da, i),
            _isel_dim(u_component, i),
            _isel_dim(v_component, i),
            args,
        )
        for i in indices
    ]

    if parallel:
        processes = min(len(indices), n_cpus // 2)

        delayed_tasks = [delayed(_mapplot_i)(*task) for task in tasks]

        if ipykernel:
            # from .xrext import DaskProgressBar
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                compute(
                    *delayed_tasks,
                    scheduler="processes",
                    num_workers=processes,
                )
        else:
            compute(
                *delayed_tasks,
                scheduler="processes",
                num_workers=processes,
            )

    else:
        from rich.progress import track

        for task in track(
            tasks,
            total=len(tasks),
            transient=False,
        ):
            _mapplot_i(*task)

    # ---- ffmpeg encode (MP4 only) ----

    user_path = False
    if not outfile:
        outfile = Path(tmp / f"{uuid.uuid4().hex}/{uuid.uuid4().hex}.mp4")

    else:
        outfile = Path(outfile)
        user_path = True

    outfile.parent.mkdir(parents=True, exist_ok=True)
    input_pattern = str(Path(session_tmp_dir) / "%06d.png")

    ffmpeg_encode(input_pattern, outfile, fps, session_tmp_dir, user_path, 0)


@wraps(plot_pvalues)
def pvalues(*args, **kwargs):
    return plot_pvalues(*args, **kwargs)


@wraps(plot_quiver)
def quiver(*args, **kwargs):
    return plot_quiver(*args, **kwargs)
