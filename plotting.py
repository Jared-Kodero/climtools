"""
Cartopy-based plotting utilities for xarray DataArrays, including faceted map plots with customizable features and styling.
"""

import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as cgeo
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


@dataclass(frozen=True, repr=False)
class MapPlot:
    Figure: Figure
    Axes: Axes | cgeo.GeoAxes | np.ndarray
    Artist: Artist

    def __repr__(self) -> str:
        if isinstance(self.Axes, np.ndarray):
            ax_repr = f"{self.Axes.size} axes"
        else:
            ax_repr = type(self.Axes).__name__
        _repr = f"MapPlot(Figure={type(self.Figure).__name__}, Axes={ax_repr}, Artist={type(self.Artist).__name__})"
        return _repr


def _get_quiver_key_mag(u: xr.DataArray, v: xr.DataArray) -> int | float:
    mag = (u**2 + v**2) ** 0.5
    key_mag = np.round(mag.quantile(0.75, skipna=True).values)
    key_mag_int = int(key_mag)
    key_magnitude = key_mag_int if key_mag_int != 0 else np.round(key_mag, 2)
    return key_magnitude


def plot_quiver(
    u: xr.DataArray,
    v: xr.DataArray,
    ax: plt.Axes | cgeo.GeoAxes = None,
    subsample: int = 1,
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

    key_magnitude = kwargs.pop("key_magnitude", None)
    if not key_magnitude:
        key_magnitude = _get_quiver_key_mag(u, v)

    Q = ax.quiver(
        lon2d,
        lat2d,
        u.values,
        v.values,
        transform=transform,
        angles="xy",
        **kwargs,
    )

    if add_key:
        label = _get_label(key_magnitude, units)

        fig = ax.get_figure()

        cax = _get_cax(fig=fig, axes=ax, orientation="horizontal", subplots=subplots)
        bbox = cax.get_position()
        cax.remove()

        # Desired quiver-key anchor in axes coordinates.
        key_x_ax = 0.100
        key_y_ax = -0.045

        # Convert that same point from axes coordinates to figure coordinates.
        key_x_fig, key_y_fig = fig.transFigure.inverted().transform(
            ax.transAxes.transform((key_x_ax, key_y_ax))
        )

        cax = fig.add_axes(
            [
                key_x_fig,
                key_y_fig - 0.5 * bbox.height,
                bbox.width - key_x_fig,
                bbox.height,
            ],
            zorder=1,
        )

        qk = ax.quiverkey(
            Q,
            X=key_x_ax,
            Y=key_y_ax,
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

    return ax


def plot_cbar(
    fig: plt.Figure,
    ax: plt.Axes | cgeo.GeoAxes,
    artist: Artist,
    orientation: str = "vertical",
    adjust: bool = True,
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
    cax = _get_cax(fig=fig, axes=ax, orientation=orientation, adjust=adjust)

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


def plot_pvalues(
    data: xr.DataArray,
    ax: plt.Axes | cgeo.GeoAxes = None,
    level: float = 0.05,
    color: str = "grey",
    alpha: float = 0.3,
    marker: str = None,
    edgecolors: str = None,
    subsample: int = 1,
    size: float = 0.25,
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
        s=size,
        marker=marker,
        edgecolors=edgecolors,
    )
    return ax


def _check_cartopy_axis(ax):

    transform = None
    if isinstance(ax, cgeo.GeoAxes):
        transform = ccrs.PlateCarree()

    return transform


def _get_cax(
    *,
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    subplots: bool = False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    adjust: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Create a new set of axes for a colorbar by stealing space from the current axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to which the colorbar axes are added. Defaults to the current figure.
    axes : matplotlib.axes.Axes or numpy.ndarray of Axes, optional
        Axes from which space is taken. Defaults to the current axes.
    subplots : bool, optional
        If True, position the colorbar relative to a grid of subplots. Requires axes.
    orientation : {"vertical", "horizontal"}, optional
        Colorbar orientation. Default "vertical".

    Returns
    -------
    matplotlib.axes.Axes
        New axes for the colorbar.
    """

    if subplots and axes is None:
        raise ValueError("If subplots is True, axes and fig must be provided.")

    if fig is None:
        fig = plt.gcf()
    if axes is None:
        axes = plt.gca()

    if adjust:
        plt.tight_layout()

    def _create_cax(y0, x0, y1, x1, x_len, y_len, ax):
        # Vertical uses y0, y_len, x1. Horizontal uses y0, x0, x_len.
        # Hold the bar thickness and gaps constant in inches: a figure-fraction
        # value times the figure size in inches is a physical length, so a fixed
        # fraction grows on larger figures. Scale the short dimension by
        # ref / size. Leave the long dimension (y_len, x_len) unscaled since it
        # tracks the axes extent.

        ref_width = 5
        ref_height = 4.8

        fig_w, fig_h = fig.get_size_inches()
        scale_w = ref_width / fig_w
        scale_h = ref_height / fig_h

        # if not subplots:
        #     scale_w = 1.0
        #     scale_h = 1.0

        if orientation == "vertical":
            bottommost = y0
            height = y_len
            rightmost = x1 + 0.04 * scale_w
            width = 0.03 * scale_w
            cax = fig.add_axes([rightmost, bottommost, width, height])

        elif orientation == "horizontal":
            xticks = len(list(ax.get_xticks())) > 0
            xlabel = bool(ax.xaxis.label.get_text().strip())
            pad = 0.10
            h_pad = 0
            if (xlabel and not xticks) or (xticks and not xlabel):
                pad = 0.15
            elif xticks and xlabel:
                pad = 0.20

            rightmost = x0
            width = x_len
            bottommost = y0 - pad * scale_h
            height = (0.05 + h_pad) * scale_h
            cax = fig.add_axes([rightmost, bottommost, width, height])

        return cax

    if not subplots:
        pos = axes.get_position()
        fig_x_len = pos.x1 - pos.x0
        fig_y_len = pos.y1 - pos.y0
        cax = _create_cax(pos.y0, pos.x0, pos.y1, pos.x1, fig_x_len, fig_y_len, axes)
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


def _get_units(units: str):
    try:
        u = metpy_units(units)
        return f"{u.units:~^P}"
    except Exception:
        return units


def _escape_chars(text: str) -> str:
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


def _get_label(
    long_name: str,
    units: str,
    cbar_label: str = None,
    format: bool = True,
) -> str:
    name = cbar_label or long_name
    name = _escape_chars(str(name))

    if not units:
        return name

    if format:
        units = _get_units(units)

    if not units or str(units).strip() in {"", "dimensionless", "1"}:
        return name

    units = _escape_chars(units)

    return rf"{name} [${units}$]"


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


def _get_projection(
    projection: str, central_longitude: float = None, central_latitude: float = None
) -> dict:
    proj_cls = getattr(ccrs, projection)
    _cargs = get_fsig(proj_cls)
    cargs = {}

    if central_longitude is not None and "central_longitude" in _cargs:
        cargs["central_longitude"] = central_longitude
    if central_latitude is not None and "central_latitude" in _cargs:
        cargs["central_latitude"] = central_latitude

    return {"projection": proj_cls(**cargs)}


def _resolve_map_aspect(
    da: xr.DataArray = None,
    extent: tuple = None,
    x: str = None,
    y: str = None,
    dim: str = None,
):
    """Resolve lon/lat plot aspect.

    Priority: explicit map_aspect, then extent, then coordinate spans,
    then grid shape.
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
    layouts. fg.axs and fg.name_dicts are coerced to 2D of shape (nrow, ncol);
    column 0 is the left edge and increasing row index moves downward. Returns
    None if no facet is populated.
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


def _faceted(
    da: xr.DataArray,
    x: str = None,
    y: str = None,
    col: str = None,
    row: str = None,
    col_wrap: int = None,
    figsize: tuple[float, float] = None,
    method: str = None,
    projection: str = None,
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
    central_longitude: float = None,
    central_latitude: float = None,
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
    p_values: xr.DataArray = None,
    p_value_kwargs: dict = None,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    quiver_kwargs: dict = None,
    cyclic: bool = False,
    **kwargs,
) -> MapPlot:

    if not x or not y:
        spatial = [d for d in da.dims if d not in (col, row)]
        x, y = spatial

    add_quiver = (u_component is not None) and (v_component is not None)
    quiver_kwargs = quiver_kwargs or {}
    add_pvalues = p_values is not None
    p_value_kwargs = p_value_kwargs or {}

    if add_quiver and quiver_kwargs.get("key_magnitude") is None:
        quiver_kwargs["key_magnitude"] = _get_quiver_key_mag(u_component, v_component)

    long_name = da.attrs.get("long_name", "").title()
    units = units or da.attrs.get("units", da.name)

    projection = _get_projection(projection, central_longitude, central_latitude)
    transform = ccrs.PlateCarree()
    orientation = orientation or "horizontal"

    dim = col or row
    col_wrap = col_wrap or int(np.ceil(np.sqrt(len(da[dim]))))

    # we want all possible args
    plot, pargs = _plot_method(da, method)
    all_args = dict(locals())
    all_args.update(kwargs)

    pkwargs = {k: v for k, v in all_args.items() if k in pargs}
    pkwargs.update(
        {
            "zorder": 1,
            "transform": transform,
            "rasterized": rasterized,
            "col": col,
            "row": row,
            "col_wrap": col_wrap,
            "subplot_kws": projection,
        }
    )

    pkwargs.pop("kwargs")
    pkwargs.pop("figsize")

    cbar_kwargs = {
        "shrink": 0.5,
        "orientation": orientation,
        "fraction": 0.05,
        "pad": 0.06,
        "label": _get_label(f"{long_name}\n", units, cbar_label),
    }

    figsize = _facet_figsize(col_wrap=col_wrap, da=da, x=x, y=y, dim=dim)

    fg = plot(figsize=figsize, cbar_kwargs=cbar_kwargs, **pkwargs)
    mappable = fg._mappables[-1]

    if hasattr(fg, "cbar") and fg.cbar is not None:
        fg.cbar.remove()

    key_ax = _bottom_left_axis(fg)

    for i, (ax, name_dict) in enumerate(zip(fg.axs.flat, fg.name_dicts.flat)):
        if name_dict is None:
            ax.remove()
            continue

        _add_cartopy_features(
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

        if add_pvalues:
            plot_pvalues(
                data=p_values.sel(name_dict),
                ax=ax,
                **p_value_kwargs,
            )

        if add_quiver:
            plot_quiver(
                u=u_component.sel(name_dict),
                v=v_component.sel(name_dict),
                add_key=(ax is key_ax),
                subplots=True,
                ax=ax,
                **quiver_kwargs,
            )

    if col_wrap == 1:
        orientation = "horizontal"

    cax = _get_cax(fig=fg.fig, axes=fg.axs, orientation=orientation, subplots=True)
    cb = fg.fig.colorbar(
        mappable,
        cax=cax,
        orientation=orientation,
        extend=extend,
        drawedges=drawedges,
    )

    cb.set_label(_get_label(f"{long_name}\n", units, cbar_label))

    plt.tight_layout()

    return MapPlot(Figure=fg.fig, Axes=fg.axs, Artist=fg)


def mapplot(
    da: xr.DataArray,
    *,
    x: str = None,
    y: str = None,
    col: str = None,
    row: str = None,
    col_wrap: int = None,
    figsize: tuple[float, float] = None,
    method: Literal[
        "default", "pcolormesh", "contourf", "contour", "imshow"
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
    ] = "PlateCarree",
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
    central_longitude: float = None,
    central_latitude: float = None,
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
    p_values: xr.DataArray = None,
    p_value_kwargs: dict = None,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    quiver_kwargs: dict = None,
    cyclic: bool = False,
    **kwargs,
) -> MapPlot:
    """
    Plot a two-dimensional xarray DataArray on a Cartopy map.

    Parameters
    ----------
    da : xarray.DataArray
        Scalar field to plot. After ``squeeze()``, the array must be
        2D or 3D and must contain longitude-latitude coordinates
        compatible with a ``cartopy.crs.PlateCarree()`` data transform.

    x, y : str, optional
        Coordinate names passed to the selected xarray plotting method when
        supported.

    col, row : str, optional
        Faceting coordinate names passed to the selected xarray plotting method
        when supported.

    col_wrap : int, optional
        Number of columns used when wrapping faceted subplots. Passed to the
        selected xarray plotting method when supported.

    figsize : tuple of float, optional
        Figure size in inches used when creating a new figure.

    method : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
        Xarray plotting method used for the scalar field.

    projection : str, default "PlateCarree"
        Cartopy projection used when creating a new axis.

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

    robust : bool, default False
        Whether to request percentile-based color scaling when supported by
        xarray.

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

    central_longitude, central_latitude : float, optional
        Projection-center arguments passed to the Cartopy projection constructor
        when supported.

    global_extent : bool, default False
        If True, set the map extent to the full globe.

    set_extent : tuple of float, optional
        Geographic extent as a tuple ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.

    gridlines : bool, default False
        Whether to draw labeled longitude and latitude gridlines.

    coastlines, borders, states : bool, default True
        Switches controlling common Cartopy geographic feature overlays.

    ocean, land : bool, default True
        Switches controlling ocean and land background features.

    lakes, rivers : bool, default False
        Switches controlling optional Cartopy inland water feature overlays.

    p_values : xarray.DataArray, optional
        Pointwise p-value field. Values below the selected significance level
        are plotted as markers.

    p_value_kwargs : dict, optional
        Keyword arguments forwarded to ``plot_pvalues``. Options include
        ``level: float``, ``color: str``, ``alpha: float`` , ``marker: str``, ``edgecolors: str`` , ``subsample:int``, ``size: float``

    u_component, v_component : xarray.DataArray, optional
        Zonal and meridional vector components for quiver overlays.

    quiver_kwargs : dict, optional
        Keyword arguments forwarded to ``plot_quiver``. Options inclued ``subsample: int``, ``key_magnitude: int|float``, ``scale:int`` etc see
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html

    cyclic : bool, default False
        If True, append a cyclic longitude point before plotting. The longitude
        dimension is assumed to be named ``"lon"``.

    **kwargs
        Additional keyword arguments forwarded to the selected xarray plotting
        method after signature filtering.

    Returns
    -------
    MapPlot
        Container with ``Figure``, ``Axes``, and ``Artist`` attributes.

    Notes
    -----
    Input coordinates are plotted with a ``cartopy.crs.PlateCarree()``
    transform. The display projection is controlled by ``projection``.
    """

    if cyclic:
        da = make_cyclic(da, lon="lon")

    da = da.squeeze()

    if da.ndim == 3:
        return _faceted(**locals())

    elif da.ndim > 3:
        raise ValueError("DataArray must be 2D or 3D after squeezing.")

    add_quiver = (u_component is not None) and (v_component is not None)
    quiver_kwargs = quiver_kwargs or {}
    add_pvalues = p_values is not None
    p_value_kwargs = p_value_kwargs or {}

    long_name = da.attrs.get("long_name", "").title()
    units = units or da.attrs.get("units", da.name)

    projection = _get_projection(projection, central_longitude, central_latitude)
    fig, ax = plt.subplots(subplot_kw=projection, figsize=figsize)
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

    for k in ("col", "row", "col_wrap", "figsize", "kwargs"):
        pkwargs.pop(k)

    artist = plot(**pkwargs)
    if method in ["contour", "contourf"]:
        if hasattr(artist, "set_edgecolor"):
            artist.set_edgecolor("face")

        if hasattr(artist, "set_rasterized"):
            artist.set_rasterized(rasterized)
        else:
            for c in artist.collections:
                c.set_rasterized(rasterized)

    plt.title(title)

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

    if add_pvalues:
        ax = plot_pvalues(data=p_values, ax=ax, **p_value_kwargs)

    if add_quiver:
        ax = plot_quiver(
            u=u_component,
            v=v_component,
            ax=ax,
            **quiver_kwargs,
        )

    if add_colorbar:
        cbar_label = _get_label(f"{long_name}\n", units, cbar_label)
        orientation = orientation or "vertical"

        ax = plot_cbar(
            fig,
            ax,
            artist,
            orientation=orientation,
            drawedges=drawedges,
            extend=extend,
            cbar_label=cbar_label,
            adjust=False,
        )
    return MapPlot(Figure=fig, Axes=ax, Artist=artist)


def _ffmpeg_encode(input_pattern, outfile, fps, session_tmp_dir, user_path, error):

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


def _mapplot_wrapper(
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

    # if dim values is numpy datetime,the string representation is too long, so we shorten it to the date only
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

    plot = mapplot(**{k: v for k, v in local_kwargs.items() if k in get_fsig(mapplot)})

    faceted = local_kwargs.get("col") or local_kwargs.get("row")
    if faceted is None:
        plot.Axes.set_title(title)
    else:
        plot.Figure.suptitle(title)

    plt.savefig(fname, dpi=dpi, bbox_inches="tight")

    plot.Figure.clear()
    plt.close(plot.Figure)

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
        "default", "pcolormesh", "contourf", "contour", "imshow"
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
    ] = "PlateCarree",
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
    central_longitude: float = None,
    central_latitude: float = None,
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
    quiver_kwargs: dict = None,
    cyclic: bool = False,
    indices: tuple | list | np.ndarray = None,
    outfile: Path = None,
    quality: Literal["low", "medium", "high"] = "medium",
    fps: int = 1,
    parallel: bool = True,
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

    x, y : str, optional
        Coordinate names passed to the selected xarray plotting method when
        supported.

    col, row : str, optional
        Faceting coordinate names passed to the selected xarray plotting method
        when supported.

    col_wrap : int, optional
        Number of columns used when wrapping faceted subplots. Passed to the
        selected xarray plotting method when supported.

    figsize : tuple of float, optional
        Figure size in inches for each rendered frame.

    method : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
        Xarray plotting method used for the scalar field.

    projection : str, default "PlateCarree"
        Cartopy projection used for each frame.

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

    robust : bool, default False
        Whether to request percentile-based color scaling when supported by
        xarray.

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
        Explicit colorbar label. If omitted, a label is inferred from metadata.

    central_longitude, central_latitude : float, optional
        Projection-center arguments passed to the Cartopy projection constructor
        when supported.

    global_extent : bool, default False
        If True, set each map extent to the full globe.

    set_extent : tuple of float, optional
        Geographic extent as ``(lon_min, lon_max, lat_min, lat_max)`` in
        degrees.

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

    quiver_kwargs : dict, optional
        Keyword arguments forwarded to ``plot_quiver``. Options inclued ``subsample: int``, ``key_magnitude: int|float``,``scale:int`` etc see
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

    def _sel(da: xr.DataArray | None, i: int):
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
            _sel(da, i),
            _sel(u_component, i),
            _sel(v_component, i),
            args,
        )
        for i in indices
    ]

    if parallel:
        processes = min(len(indices), n_cpus // 2)

        delayed_tasks = [delayed(_mapplot_wrapper)(*task) for task in tasks]

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
            _mapplot_wrapper(*task)

    # ---- ffmpeg encode (MP4 only) ----

    user_path = False
    if not outfile:
        outfile = Path(tmp / f"{uuid.uuid4().hex}/{uuid.uuid4().hex}.mp4")

    else:
        outfile = Path(outfile)
        user_path = True

    outfile.parent.mkdir(parents=True, exist_ok=True)
    input_pattern = str(Path(session_tmp_dir) / "%06d.png")

    _ffmpeg_encode(input_pattern, outfile, fps, session_tmp_dir, user_path, 0)
