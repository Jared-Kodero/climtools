"""Typed plotting primitives and input utilities for Cartopy map plots.

The functions in this module are intentionally stateless. Plotting functions
receive an existing Matplotlib figure and axis, draw one layer, and return the
resulting Matplotlib primitive. They can therefore be used independently of the
stateful classes defined in :mod:`plotting`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as cgeo
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import Gridliner
from cf_xarray import *
from IPython.display import clear_output
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
from matplotlib.ticker import MaxNLocator

from .tools import (
    get_fsig,
    mpl_backend_changed,
    mpl_default_backend,
    set_preview_quality,
)
from .xgeo_utils import add_cyclic_point, get_spatial_dims, set_edges_to_nan, to_lon180

__all__ = [
    "add_colorbar",
    "add_contour_labels",
    "add_gridlines",
    "add_map_features",
    "get_cax",
    "get_facet_figsize",
    "get_map_aspect",
    "get_projection",
    "get_quiver_key_mag",
    "norm_input",
    "norm_levels",
    "normalize_subsample",
    "plot_contour",
    "plot_contourf",
    "plot_default",
    "plot_imshow",
    "plot_pcolormesh",
    "plot_quiver",
    "plot_scatter",
    "plot_significance",
    "select_facet",
    "validate_animation_inputs",
    "validate_data",
    "validate_facets",
    "validate_vector_components",
]


AxesType = Axes | cgeo.GeoAxes
ScalarArtist = Artist | ScalarMappable | QuadContourSet


def interactive_backend(interactive: bool) -> None:
    """Configure matplotlib for interactive use in Jupyter notebooks."""

    global mpl_backend_changed

    if interactive != mpl_backend_changed:
        plt.close("all")
        clear_output(wait=True)

        if interactive:
            try:
                matplotlib.use("module://ipympl.backend_nbagg")
            except Exception:
                matplotlib.use("nbagg")  # fallback
            mpl_backend_changed = True
        else:
            matplotlib.use(mpl_default_backend)
            set_preview_quality()
            mpl_backend_changed = False


def validate_data(data: xr.DataArray) -> xr.DataArray:
    """Validate a scalar plotting field.

    Parameters
    ----------
    data : xarray.DataArray
        Scalar field to validate.

    Returns
    -------
    xarray.DataArray
        The validated input object.

    Raises
    ------
    TypeError
        If ``data`` is not an :class:`xarray.DataArray`.
    ValueError
        If ``data`` is empty.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"data must be an xarray.DataArray, got {type(data)!r}")
    if data.size == 0:
        raise ValueError("data must contain at least one value")
    return data


def norm_levels(
    vmin: float | None,
    vmax: float | None,
    levels: int | Sequence[float] | np.ndarray | None,
    data: xr.DataArray | None = None,
) -> tuple[float | None, float | None, np.ndarray | None]:
    """Normalize plotting limits and level boundaries.

    Explicit level boundaries are returned unchanged. If either ``vmin`` or
    ``vmax`` is missing and ``data`` is provided, the missing limits are
    inferred from the 2nd and 98th percentiles of the data. When both negative
    and positive values are present, the inferred range is made symmetric
    about zero.

    Integer ``levels`` values generate evenly spaced boundaries between
    ``vmin`` and ``vmax``. If ``levels`` is ``None``, suitable boundaries are
    generated with ``MaxNLocator``.

    Parameters
    ----------
    vmin
        Lower plotting limit.
    vmax
        Upper plotting limit.
    levels
        Number of levels, explicit level boundaries, or ``None``.
    data
        Data used to infer missing plotting limits.

    Returns
    -------
    vmin
        Normalized lower plotting limit.
    vmax
        Normalized upper plotting limit.
    levels
        Explicit increasing level boundaries, or ``None`` if they cannot be
        determined.
    """
    if isinstance(levels, (list, tuple, np.ndarray)):
        return vmin, vmax, np.asarray(levels)

    if vmin is None or vmax is None:
        if data is None:
            return vmin, vmax, None

        qmin, qmax = data.quantile([0.02, 0.98], skipna=True).compute().values

        qmin = float(qmin)
        qmax = float(qmax)

        if not np.isfinite(qmin) or not np.isfinite(qmax):
            return vmin, vmax, None

        has_negative = bool((data < 0).any().compute())
        has_positive = bool((data > 0).any().compute())

        if has_negative and has_positive:
            bound = max(abs(qmin), abs(qmax))
            data_vmin = -bound
            data_vmax = bound
        else:
            data_vmin = qmin
            data_vmax = qmax

        if vmin is None:
            vmin = data_vmin
        if vmax is None:
            vmax = data_vmax

    if isinstance(levels, int):
        return vmin, vmax, np.linspace(vmin, vmax, levels)

    if levels is None:
        return vmin, vmax, MaxNLocator(nbins=10).tick_values(vmin, vmax)

    raise TypeError(f"unsupported levels type: {type(levels).__name__}")


def validate_facets(
    data: xr.DataArray,
    *,
    col: str | None,
    row: str | None,
    col_wrap: int | None,
) -> None:
    """Validate facet dimensions and layout arguments.

    Parameters
    ----------
    data : xarray.DataArray
        Normalized plotting field.
    col, row : str, optional
        Dimensions used for column and row faceting.
    col_wrap : int, optional
        Maximum number of columns for a one-dimensional column facet.

    Raises
    ------
    ValueError
        If a facet dimension is absent, duplicated, or incompatible with
        ``col_wrap``.
    """
    if col is not None and col not in data.dims:
        raise ValueError(f"col={col!r} is not present in data.dims {data.dims!r}")
    if row is not None and row not in data.dims:
        raise ValueError(f"row={row!r} is not present in data.dims {data.dims!r}")
    if col is not None and row is not None and col == row:
        raise ValueError("col and row must refer to different dimensions")
    if row is not None and col_wrap is not None:
        raise ValueError("col_wrap is only valid when row is None")
    if col_wrap is not None and col_wrap < 1:
        raise ValueError("col_wrap must be greater than or equal to 1")


def enable_interactive_features(ax: AxesType) -> str:

    def format_coord(x: float, y: float) -> str:
        lon, lat = ccrs.PlateCarree().transform_point(
            x,
            y,
            src_crs=ax.projection,
        )

        if not np.isfinite(lon) or not np.isfinite(lat):
            return ""

        return f"lat={round(lat, 2)} lon={round(lon, 2)}"

    ax.format_coord = format_coord


def validate_vector_components(
    u: xr.DataArray | None,
    v: xr.DataArray | None,
    *,
    reference: xr.DataArray | None = None,
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    """Validate a pair of vector components.

    Parameters
    ----------
    u, v : xarray.DataArray, optional
        Zonal and meridional vector components. Both must be supplied together.
    reference : xarray.DataArray, optional
        Scalar field whose dimensions and coordinates should be compatible with
        the vector components.

    Returns
    -------
    tuple of xarray.DataArray or None
        The validated ``(u, v)`` pair.

    Raises
    ------
    TypeError
        If either component is not an :class:`xarray.DataArray`.
    ValueError
        If only one component is supplied or if the components are not aligned.
    """
    if (u is None) != (v is None):
        raise ValueError("u and v must be provided together")
    if u is None or v is None:
        return None, None
    if not isinstance(u, xr.DataArray):
        raise TypeError(f"u must be an xarray.DataArray, got {type(u)!r}")
    if not isinstance(v, xr.DataArray):
        raise TypeError(f"v must be an xarray.DataArray, got {type(v)!r}")
    try:
        xr.align(u, v, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("u and v must have identical indexes and coordinates") from exc
    if reference is not None:
        missing = set(reference.dims) - set(u.dims)
        if missing:
            raise ValueError(
                f"vector components are missing reference dimensions {sorted(missing)!r}"
            )
    return u, v


def normalize_subsample(
    subsample: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    """Normalize a scalar or two-element spatial stride.

    Parameters
    ----------
    subsample : int or tuple of int
        Spatial stride. A scalar is applied to both dimensions.

    Returns
    -------
    tuple of int
        The normalized ``(x_stride, y_stride)`` pair.

    Raises
    ------
    ValueError
        If the stride does not contain exactly two positive integers.
    """
    if isinstance(subsample, int):
        result = (subsample, subsample)
    else:
        if len(subsample) != 2:
            raise ValueError("subsample must contain exactly two values")
        result = (int(subsample[0]), int(subsample[1]))
    if result[0] < 1 or result[1] < 1:
        raise ValueError("subsample values must be greater than or equal to 1")
    return result


def norm_input(
    data: xr.DataArray,
    *,
    x: str | None = None,
    y: str | None = None,
    col: str | None = None,
    row: str | None = None,
    col_wrap: int | None = None,
    cyclic: bool = False,
) -> tuple[xr.DataArray, str, str, str | None, str | None]:
    """Normalize and validate the scalar plotting input.

    Parameters
    ----------
    data : xarray.DataArray
        Scalar field to normalize.
    x, y : str, optional
        Horizontal coordinate names. They are inferred when omitted.
    col, row : str, optional
        Facet dimensions.
    col_wrap : int, optional
        Maximum number of columns for one-dimensional faceting.
    cyclic : bool, default False
        Append a cyclic point along the horizontal coordinate.

    Returns
    -------
    data : xarray.DataArray
        Squeezed field normalized to the ``[-180, 180)`` longitude convention.
    x, y : str
        Resolved horizontal coordinate names.
    col, row : str, optional
        Validated facet dimensions.

    Raises
    ------
    ValueError
        If the field cannot be reduced to two spatial dimensions plus at most
        two facet dimensions, or if required coordinates are absent.
    """
    data = validate_data(data)
    if x is None or y is None:
        inferred_x, inferred_y = get_spatial_dims(data)
        x = x or inferred_x
        y = y or inferred_y
    if x not in data.coords:
        raise ValueError(f"x coordinate {x!r} is not present in data.coords")
    if y not in data.coords:
        raise ValueError(f"y coordinate {y!r} is not present in data.coords")
    if cyclic:
        data = add_cyclic_point(data, lon=x)
    data = to_lon180(data, lon=x).squeeze()
    validate_facets(data, col=col, row=row, col_wrap=col_wrap)
    allowed_dimensions = {x, y}
    if col is not None:
        allowed_dimensions.add(col)
    if row is not None:
        allowed_dimensions.add(row)
    unresolved = [dim for dim in data.dims if dim not in allowed_dimensions]
    if unresolved:
        raise ValueError(
            f"data contains unresolved non-spatial dimensions {unresolved!r}; select them or assign them to col or row"
        )
    expected_ndim = 2 + int(col is not None) + int(row is not None)
    if data.ndim != expected_ndim:
        raise ValueError(
            f"expected {expected_ndim} dimensions after normalization, got {data.ndim}: {data.dims!r}"
        )
    return data, x, y, col, row


def select_facet(data: xr.DataArray, selector: Mapping[str, Any]) -> xr.DataArray:
    """Select and squeeze one facet from an input field.

    Parameters
    ----------
    data : xarray.DataArray
        Field containing the facet coordinates.
    selector : mapping of str to object
        Coordinate-value selector for one panel.

    Returns
    -------
    xarray.DataArray
        Selected two-dimensional field.
    """
    return data.sel(dict(selector)).squeeze() if selector else data.squeeze()


def validate_animation_inputs(
    dim: str,
    data: xr.DataArray,
    u: xr.DataArray | None = None,
    v: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None, xr.DataArray | None]:
    """Validate and sort animation inputs.

    Parameters
    ----------
    dim : str
        Animation dimension.
    data : xarray.DataArray
        Scalar field to animate.
    u, v : xarray.DataArray, optional
        Vector components to animate with the scalar field.

    Returns
    -------
    data, u, v : tuple
        Inputs sorted along ``dim``.

    Raises
    ------
    ValueError
        If ``dim`` is absent or the vector components are not aligned.
    """
    data = validate_data(data)
    u, v = validate_vector_components(u, v)
    if dim not in data.dims:
        raise ValueError(f"{dim!r} is not present in data.dims {data.dims!r}")
    for name, component in (("u", u), ("v", v)):
        if component is not None and dim not in component.dims:
            raise ValueError(
                f"{dim!r} is not present in {name}.dims {component.dims!r}"
            )
    data = data.sortby(dim)
    if u is not None and v is not None:
        u = u.sortby(dim)
        v = v.sortby(dim)
        try:
            xr.align(data, u, v, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError(
                "data, u, and v must have identical animation coordinates"
            ) from exc
    return data, u, v


def get_quiver_key_mag(u: xr.DataArray, v: xr.DataArray) -> int | float:
    """Return a reference quiver-key magnitude from the 75th percentile speed."""
    mag = (u**2 + v**2) ** 0.5
    key_mag = np.round(mag.quantile(0.75, skipna=True).values)
    key_mag_int = int(key_mag)
    key_magnitude = key_mag_int if key_mag_int != 0 else np.round(key_mag, 3)
    return key_magnitude


def is_geoaxes(ax: AxesType, kwargs: Mapping[str, Any]) -> dict:
    """Inject a ``PlateCarree`` data transform when ``ax`` is a GeoAxes."""
    if isinstance(ax, cgeo.GeoAxes):
        kwargs["transform"] = ccrs.PlateCarree()

    return kwargs


def is_defined(**kwargs: Any) -> dict[str, Any]:
    return {name: value for name, value in kwargs.items() if value is not None}


def get_function_inputs(function: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    accepted = get_fsig(function)
    return {name: value for name, value in kwargs.items() if name in accepted}


def style_contours(artist: QuadContourSet, rasterized: bool) -> None:
    if hasattr(artist, "set_edgecolor"):
        artist.set_edgecolor("face")
    if hasattr(artist, "set_rasterized"):
        artist.set_rasterized(rasterized)
        return
    for collection in artist.collections:
        collection.set_rasterized(rasterized)


def add_contour_labels(
    fig: Figure,
    ax: AxesType,
    artist: QuadContourSet,
    *,
    fmt: str | Mapping[float, str] = "%1.0f",
    fontsize: float = 8.0,
    inline: bool = True,
    colors: str | Sequence[str] | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> list[Text]:
    """Label a line-contour primitive.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing ``ax``. The argument makes the primitive interface
        consistent with the other plotting functions.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Axis containing the contour set.
    artist : matplotlib.contour.QuadContourSet
        Line-contour primitive.
    fmt : str or mapping, default "%1.0f"
        Contour-label format.
    fontsize : float, default 8
        Label font size in points.
    inline : bool, default True
        Draw labels inline with contour lines.
    colors : str or sequence of str, optional
        Label colors.
    kwargs : mapping, optional
        Additional arguments forwarded to :meth:`Axes.clabel`.

    Returns
    -------
    list of matplotlib.text.Text
        Created text primitives.
    """

    labels = ax.clabel(
        artist,
        fmt=fmt,
        fontsize=fontsize,
        inline=inline,
        colors=colors,
        **dict(kwargs or {}),
    )
    return list(labels)


def add_gridlines(
    fig: Figure,
    ax: cgeo.GeoAxes,
    *,
    draw_labels: bool = True,
    linewidth: float = 0.5,
    color: str = "gray",
    alpha: float = 0.5,
    linestyle: str = "--",
    zorder: float = 1.0,
) -> Gridliner:
    """Add longitude and latitude gridlines to a Cartopy axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : cartopy.mpl.geoaxes.GeoAxes
        Geographic axis to modify.
    draw_labels : bool, default True
        Draw coordinate labels.
    linewidth : float, default 0.5
        Gridline width in points.
    color : str, default "gray"
        Gridline color.
    alpha : float, default 0.5
        Gridline opacity.
    linestyle : str, default "--"
        Gridline style.
    zorder : float, default 1
        Gridline drawing order.

    Returns
    -------
    cartopy.mpl.gridliner.Gridliner
        Created gridliner.
    """

    gridliner = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=draw_labels,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )
    gridliner.top_labels = False
    gridliner.right_labels = False

    return gridliner


def add_map_features(
    fig: Figure,
    ax: AxesType,
    *,
    global_extent: bool = False,
    set_extent: tuple[float, float, float, float] | None = None,
    coastlines: bool = True,
    states: bool = True,
    borders: bool = True,
    lakes: bool = False,
    rivers: bool = False,
    ocean: bool = True,
    land: bool = True,
) -> list[Artist]:
    """Add geographic context to an existing axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Axis to modify.
    global_extent : bool, default False
        Set the map to a global extent.
    set_extent : tuple of float, optional
        Explicit extent ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
    coastlines, states, borders : bool, default True
        Add common boundary features.
    lakes, rivers : bool, default False
        Add inland-water features.
    ocean, land : bool, default True
        Control land and ocean background fills.

    Returns
    -------
    list of matplotlib.artist.Artist
        Feature artists added to the axis.

    Raises
    ------
    TypeError
        If ``ax`` is not a Cartopy geographic axis.
    """

    if not isinstance(ax, cgeo.GeoAxes):
        raise TypeError("map features require a cartopy.mpl.geoaxes.GeoAxes")
    if global_extent:
        ax.set_global()
    if set_extent is not None:
        ax.set_extent(set_extent, crs=ccrs.PlateCarree())
    artists: list[Artist] = []
    feature_specs: tuple[tuple[bool, cfeature.Feature, dict[str, Any]], ...] = (
        (coastlines, cfeature.COASTLINE, {}),
        (states, cfeature.STATES, {"linestyle": "-", "alpha": 0.3, "zorder": 3}),
        (borders, cfeature.BORDERS, {"linestyle": "-", "alpha": 0.3, "zorder": 3}),
        (lakes, cfeature.LAKES, {"zorder": 2}),
        (rivers, cfeature.RIVERS, {"zorder": 2}),
    )
    for enabled, feature, options in feature_specs:
        if enabled:
            artists.append(ax.add_feature(feature, **options))
    if ocean and not land:
        artists.append(ax.add_feature(cfeature.LAND, facecolor="#54585f", zorder=0))
    elif land and not ocean:
        artists.append(ax.add_feature(cfeature.OCEAN, zorder=0))
    return artists


def get_map_aspect(
    *,
    data: xr.DataArray | None = None,
    extent: tuple[float, float, float, float] | None = None,
    x: str | None = None,
    y: str | None = None,
    facet_dims: Sequence[str] = (),
) -> float:
    """Infer the horizontal-to-vertical aspect ratio of a map domain.

    Parameters
    ----------
    data : xarray.DataArray, optional
        Field used to infer coordinate spans or grid shape.
    extent : tuple of float, optional
        Geographic extent ``(lon_min, lon_max, lat_min, lat_max)``.
    x, y : str, optional
        Horizontal coordinate names.
    facet_dims : sequence of str, default ()
        Dimensions excluded when grid shape is used.

    Returns
    -------
    float
        Positive map aspect ratio.
    """
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        lon_span = abs(lon_max - lon_min)
        lat_span = abs(lat_max - lat_min)
        return max(lon_span / max(lat_span, np.finfo(float).eps), 0.1)
    if data is None:
        raise ValueError("data or extent must be provided")
    if x is not None and y is not None and x in data.coords and y in data.coords:
        lon_span = float(data[x].max() - data[x].min())
        lat_span = float(data[y].max() - data[y].min())
        return max(abs(lon_span) / max(abs(lat_span), np.finfo(float).eps), 0.1)
    spatial_dims = [dim for dim in data.dims if dim not in facet_dims]
    if len(spatial_dims) < 2:
        raise ValueError("at least two spatial dimensions are required")
    return max(data.sizes[spatial_dims[-1]] / data.sizes[spatial_dims[-2]], 0.1)


def get_facet_figsize(
    *,
    data: xr.DataArray,
    x: str,
    y: str,
    nrows: int,
    ncols: int,
    panel_width: float = 5.0,
    colorbar_padding: float = 0.8,
) -> tuple[float, float]:
    """Compute a figure size for a faceted map layout.

    Parameters
    ----------
    data : xarray.DataArray
        Plotting field.
    x, y : str
        Horizontal coordinate names.
    nrows, ncols : int
        Facet-grid shape.
    panel_width : float, default 5
        Width of one panel in inches.
    colorbar_padding : float, default 0.8
        Additional vertical space in inches.

    Returns
    -------
    tuple of float
        Figure width and height in inches.
    """
    extent = (
        float(data[x].min()),
        float(data[x].max()),
        float(data[y].min()),
        float(data[y].max()),
    )
    aspect = get_map_aspect(data=data, extent=extent, x=x, y=y)
    panel_height = panel_width / aspect
    return ncols * panel_width, nrows * panel_height + colorbar_padding


def get_projection(
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
    | None,
    longitude: xr.DataArray,
    latitude: xr.DataArray,
) -> tuple[ccrs.Projection, str]:
    """Construct a Cartopy projection for a map domain.

    Parameters
    ----------
    projection : str, optional
        Explicit Cartopy projection name. A domain-dependent projection is
        selected when omitted.
    longitude, latitude : xarray.DataArray
        Horizontal coordinates used to infer the domain center and span.

    Returns
    -------
    projection_object : cartopy.crs.Projection
        Constructed projection instance.
    projection_name : str
        Resolved projection class name.
    """
    if isinstance(projection, ccrs.Projection):
        return projection, type(projection).__name__

    lon_w = float(longitude.min())
    lon_e = float(longitude.max())
    lat_s = float(latitude.min())
    lat_n = float(latitude.max())
    lon_c = 0.5 * (lon_w + lon_e)
    lat_c = 0.5 * (lat_s + lat_n)
    true_scale_latitude: float | None = None
    standard_parallels: tuple[float, ...] | None = None
    cutoff: float | None = None
    if projection is None:
        d_lon = lon_e - lon_w
        d_lat = lat_n - lat_s
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
            if d_lat < 1.0:
                standard_parallels = (lat_c,)
            else:
                standard_parallels = (
                    lat_s + d_lat / 6.0,
                    lat_n - d_lat / 6.0,
                )
            cutoff = -30.0 if lat_c >= 0.0 else 30.0
    projection_class = getattr(ccrs, projection, None)
    known = isinstance(projection_class, type) and issubclass(
        projection_class, ccrs.Projection
    )
    if not known:
        raise ValueError(
            f"Unknown projection {projection!r}. Pass a Cartopy projection name "
            "or a cartopy.crs.Projection instance."
        )
    accepted = get_fsig(projection_class)
    options: dict[str, Any] = {}
    if "central_longitude" in accepted:
        options["central_longitude"] = lon_c
    if "central_latitude" in accepted:
        options["central_latitude"] = lat_c
    if "cutoff" in accepted and cutoff is not None:
        options["cutoff"] = cutoff
    if "standard_parallels" in accepted and standard_parallels is not None:
        options["standard_parallels"] = standard_parallels
    if "true_scale_latitude" in accepted and true_scale_latitude is not None:
        options["true_scale_latitude"] = true_scale_latitude
    return projection_class(**options), projection


def get_cax(
    *,
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    subplots: bool | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    adjust: bool = True,
    pad_bottom: bool | None = None,
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
        If True, position the colorbar relative to a grid of subplots. When
        omitted, infer this from the figure.
    orientation : {"vertical", "horizontal"}, optional
        Colorbar orientation. Default "vertical".
    adjust : bool, optional
        Whether to call ``plt.tight_layout()`` first. Default is True.
    pad_bottom : bool, optional
        Force additional space below a horizontal auxiliary axis. When omitted,
        infer the requirement from visible x-axis labels on the target axis.

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
        axes = [
            ax
            for ax in fig.get_axes()
            if ax.get_label() != "<colorbar>"
            and not ax.get_label().startswith("<quiver-key-")
        ]
        return len(axes) > 1

    if fig is None:
        fig = plt.gcf()
    if axes is None:
        axes = plt.gca()

    if subplots is None:
        subplots = _has_subplots(fig)

    if subplots and axes is None:
        raise ValueError("If subplots is True, axes and fig must be provided.")

    if adjust:
        fig.tight_layout()

    # Cartopy applies the geographic aspect and final active axes position during
    # a draw. Resolve that geometry before deriving an auxiliary colorbar axis.
    fig.canvas.draw()

    def _create_cax(y0, x0, y1, x1, x_len, y_len, ax):
        # Vertical uses y0, y_len, x1. Horizontal uses y0, x0, x_len.
        # Hold the bar thickness and gaps constant in inches: a figure-fraction
        # value times the figure size in inches is a physical length, so a fixed
        # fraction grows on larger figures. Scale the short dimension by
        # ref / size. Leave the long dimension (y_len, x_len) unscaled since it
        # tracks the axes extent.

        needs_bottom_padding = (
            pad_bottom
            if pad_bottom is not None
            else _has_visible_xtick_labels(ax) or _has_visible_xlabel(ax)
        )

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

            if needs_bottom_padding:
                y_pad = 0.1

            rightmost = x0
            width = x_len
            bottommost = y0 - y_pad * scale_h
            height = 0.04 * scale_h
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

    grid_bottom = bot_right_ax.y0
    grid_top = top_right_ax.y1
    grid_left = left_bot_ax.x0
    grid_right = right_bot_ax.x1

    grid_x_len = grid_right - grid_left
    grid_y_len = grid_top - grid_bottom

    # Vertical colorbar: use the full grid height for at most two rows.
    # For larger grids, use 60% of the grid height and center the bar.
    if nrows > 1:
        vertical_y_len = 0.5 * grid_y_len
        vertical_y0 = 0.5 * (grid_bottom + grid_top) - 0.5 * vertical_y_len
    else:
        vertical_y_len = grid_y_len
        vertical_y0 = grid_bottom

    # Horizontal colorbar: use one axis width for at most two columns.
    # For larger grids, use 50% of the grid width and center the bar.
    if ncols > 1:
        horizontal_x_len = 0.5 * grid_x_len
    else:
        horizontal_x_len = right_bot_ax.x1 - right_bot_ax.x0

    horizontal_x0 = 0.5 * (grid_left + grid_right) - 0.5 * horizontal_x_len

    cax = _create_cax(
        vertical_y0,
        horizontal_x0,
        grid_top,
        right_bot_ax.x1,
        horizontal_x_len,
        vertical_y_len,
        axes[-1, -1],
    )

    return cax


def add_grid_boundary(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    transform: ccrs.CRS,
    linewidth: float = 1,
    color: str = "black",
    zorder: float = 1,
) -> None:
    """Draw the exterior boundary of a 2-D lon-lat grid."""

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    if lon.shape != lat.shape or lon.ndim != 2:
        raise ValueError("lon and lat must be matching 1-D or 2-D arrays.")

    boundary_lon = np.concatenate(
        [
            lon[0, :],  # northern/southern grid edge
            lon[1:, -1],  # right edge
            lon[-1, -2::-1],  # opposite horizontal edge
            lon[-2:0:-1, 0],  # left edge
            lon[0, :1],  # close polygon
        ]
    )

    boundary_lat = np.concatenate(
        [
            lat[0, :],
            lat[1:, -1],
            lat[-1, -2::-1],
            lat[-2:0:-1, 0],
            lat[0, :1],
        ]
    )

    ax.plot(
        boundary_lon,
        boundary_lat,
        color=color,
        linewidth=linewidth,
        transform=transform,
        zorder=zorder,
    )


def fmt_anim_title(
    title: str,
    dim: str,
    frame_number: int,
    frame_value: Any,
    total_frames: int,
    frame_id: bool,
) -> dict:
    """Format Animation Title"""
    if np.issubdtype(np.asarray(frame_value).dtype, np.datetime64):
        frame_value = pd.to_datetime(frame_value).strftime("%Y-%m-%d %H:%M")

    # 1. Determine the labels before the colon
    idx_label = "index"
    dim_label = str(dim)
    max_label_width = max(len(idx_label), len(dim_label))
    frame_title = f"{dim_label:<{max_label_width}}: {frame_value}"

    if frame_id:
        max_id = max(total_frames - 1, 0)
        id_width = len(str(max_id))
        index_padded = f"{frame_number:0{id_width}d}"

        frame_title = f"{idx_label:<{max_label_width}}: {index_padded}\n{frame_title}"

    if title:
        frame_title = f"{title}\n{frame_title}"

    return {
        "label": frame_title,
        "loc": "left",
        "fontfamily": "monospace",
    }


def add_colorbar(
    mappable: ScalarMappable,
    ax: AxesType | np.ndarray,
    fig: Figure | None = None,
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
    mappable : matplotlib.cm.ScalarMappable
        Primitive described by the colorbar.
    ax : matplotlib.axes.Axes or numpy.ndarray
        Axis or axes associated with ``mappable``.
    fig : matplotlib.figure.Figure
        Parent figure.
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
    if cax is None:
        cax = get_cax(
            fig=fig,
            axes=ax,
            orientation=orientation,
            subplots=subplots,
            adjust=adjust,
            pad_bottom=pad_bottom,
        )
        cax.set_label("<colorbar>")

    if not fig:
        fig = plt.gcf()
    colorbar = fig.colorbar(
        mappable,
        cax=cax,
        ax=None if isinstance(ax, np.ndarray) else ax,
        orientation=orientation,
        drawedges=drawedges,
        extend=extend,
    )
    if ticks is not None:
        colorbar.set_ticks(ticks)
    if tick_labels is not None:
        if orientation == "horizontal":
            colorbar.ax.set_xticklabels(tick_labels)
        else:
            colorbar.ax.set_yticklabels(tick_labels)
    if label is not None:
        colorbar.set_label(label)
    return colorbar


def plot_default(
    data: xr.DataArray,
    fig: Figure,
    ax: AxesType,
    *,
    x: str,
    y: str,
    cmap: str | Colormap | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = False,
    rasterized: bool = False,
    zorder: float = 1.0,
    add_labels: bool = False,
    **kwargs: Any,
) -> ScalarArtist:
    """Draw an xarray field using its default plotting method.

    Parameters
    ----------
    data : xarray.DataArray
        Two-dimensional scalar field.
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Destination axis.
    x, y : str
        Horizontal coordinate names.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap.
    norm : matplotlib.colors.Normalize, optional
        Color normalization.
    vmin, vmax : float, optional
        Scalar color limits.
    robust : bool, default False
        Use percentile-based limits when supported by xarray.
    rasterized : bool, default False
        Rasterize dense output when supported.
    zorder : float, default 1
        Drawing order.
    add_labels : bool, default False
        Let xarray add axis labels and a title.
    **kwargs
        Additional arguments accepted by ``data.plot``.

    Returns
    -------
    matplotlib primitive
        Primitive returned by xarray.
    """

    options = is_defined(
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
        add_colorbar=False,
    )
    options.update(kwargs)
    options = is_geoaxes(ax, options)

    # DataArray.plot is a dispatcher whose method-specific arguments are accepted
    # through **kwargs. Signature filtering would therefore discard x/y, the
    # Cartopy transform, add_colorbar=False, and all color-scaling options.
    return data.plot(ax=ax, **options)


def plot_pcolormesh(
    data: xr.DataArray,
    fig: Figure,
    ax: AxesType,
    *,
    x: str,
    y: str,
    cmap: str | Colormap | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = False,
    rasterized: bool = False,
    zorder: float = 1.0,
    add_labels: bool = False,
    **kwargs: Any,
) -> QuadMesh:
    """Draw a pseudocolor mesh and return its ``QuadMesh`` primitive.

    Parameters
    ----------
    data : xarray.DataArray
        Two-dimensional scalar field.
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Destination axis.
    x, y : str
        Horizontal coordinate names.
    cmap, norm, vmin, vmax : optional
        Scalar-color mapping parameters.
    robust : bool, default False
        Use percentile-based limits where supported.
    rasterized : bool, default False
        Rasterize the mesh.
    zorder : float, default 1
        Drawing order.
    add_labels : bool, default False
        Let xarray add labels.
    **kwargs
        Additional arguments forwarded to xarray pcolormesh plotting.

    Returns
    -------
    matplotlib.collections.QuadMesh
        Created mesh primitive.
    """

    options = is_defined(
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
        add_colorbar=False,
    )
    options.update(kwargs)
    options = is_geoaxes(ax, options)
    return data.plot.pcolormesh(ax=ax, **options)


def plot_contourf(
    data: xr.DataArray,
    fig: Figure,
    ax: AxesType,
    *,
    x: str,
    y: str,
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
    zorder: float = 1.0,
    add_labels: bool = False,
    **kwargs: Any,
) -> QuadContourSet:
    """Draw filled contours and return a ``QuadContourSet`` primitive.

    Parameters
    ----------
    data : xarray.DataArray
        Two-dimensional scalar field.
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Destination axis.
    x, y : str
        Horizontal coordinate names.
    levels : int or sequence of float, optional
        Contour intervals.
    cmap, colors, norm, vmin, vmax : optional
        Scalar-color mapping parameters.
    extend : {"neither", "both", "min", "max"}, optional
        Out-of-range coloring.
    robust : bool, default False
        Use percentile-based color limits.
    alpha : float, optional
        Layer opacity.
    rasterized : bool, default False
        Rasterize contour collections.
    zorder : float, default 1
        Drawing order.
    add_labels : bool, default False
        Let xarray add labels.
    **kwargs
        Additional arguments forwarded to xarray filled-contour plotting.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        Created filled-contour primitive.
    """

    options = is_defined(
        x=x,
        y=y,
        levels=levels,
        cmap=cmap,
        colors=colors,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        extend=extend,
        robust=robust,
        alpha=alpha,
        zorder=zorder,
        add_labels=add_labels,
        add_colorbar=False,
    )
    options.update(kwargs)
    options = is_geoaxes(ax, options)
    artist = data.plot.contourf(ax=ax, **options)
    style_contours(artist, rasterized)
    return artist


def plot_contour(
    data: xr.DataArray,
    fig: Figure,
    ax: AxesType,
    *,
    x: str,
    y: str,
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
    zorder: float = 2.0,
    add_labels: bool = False,
    **kwargs: Any,
) -> QuadContourSet:
    """Draw line contours and return a ``QuadContourSet`` primitive.

    Parameters
    ----------
    data : xarray.DataArray
        Two-dimensional scalar field.
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Destination axis.
    x, y : str
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
    zorder : float, default 2
        Drawing order.
    add_labels : bool, default False
        Let xarray add labels.
    **kwargs
        Additional arguments forwarded to xarray line-contour plotting.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        Created line-contour primitive.
    """

    options = is_defined(
        x=x,
        y=y,
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
        zorder=zorder,
        add_labels=add_labels,
        add_colorbar=False,
    )
    options.update(kwargs)
    options = is_geoaxes(ax, options)
    artist = data.plot.contour(ax=ax, **options)
    style_contours(artist, rasterized)
    return artist


def plot_imshow(
    data: xr.DataArray,
    fig: Figure,
    ax: AxesType,
    *,
    x: str,
    y: str,
    cmap: str | Colormap | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = False,
    interpolation: str | None = None,
    origin: Literal["upper", "lower"] | None = None,
    rasterized: bool = False,
    zorder: float = 1.0,
    add_labels: bool = False,
    **kwargs: Any,
) -> AxesImage:
    """Draw an image and return its ``AxesImage`` primitive.

    Parameters
    ----------
    data : xarray.DataArray
        Two-dimensional scalar field.
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Destination axis.
    x, y : str
        Horizontal coordinate names.
    cmap, norm, vmin, vmax : optional
        Scalar-color mapping parameters.
    robust : bool, default False
        Use percentile-based color limits.
    interpolation : str, optional
        Image interpolation method.
    origin : {"upper", "lower"}, optional
        Image origin.
    rasterized : bool, default False
        Rasterize the image.
    zorder : float, default 1
        Drawing order.
    add_labels : bool, default False
        Let xarray add labels.
    **kwargs
        Additional arguments forwarded to xarray image plotting.

    Returns
    -------
    matplotlib.image.AxesImage
        Created image primitive.
    """

    options = is_defined(
        x=x,
        y=y,
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
        add_colorbar=False,
    )
    options.update(kwargs)
    options = is_geoaxes(ax, options)
    return data.plot.imshow(ax=ax, **options)


def plot_scatter(
    data: xr.DataArray,
    fig: Figure,
    ax: AxesType,
    *,
    x: str,
    y: str,
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
    zorder: float = 2.0,
    add_labels: bool = False,
    **kwargs: Any,
) -> PathCollection:
    """Draw a scatter layer and return its ``PathCollection`` primitive.

    Parameters
    ----------
    data : xarray.DataArray
        Source data containing ``x`` and ``y`` coordinates.
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Destination axis.
    x, y : str
        Coordinate or variable names used for point locations.
    hue, markersize : str or xarray.DataArray, optional
        Variables controlling point color and marker size.
    cmap, norm, vmin, vmax : optional
        Scalar-color mapping parameters.
    marker : str, optional
        Marker style.
    size : float, optional
        Constant marker area passed as ``s``.
    alpha : float, optional
        Marker opacity.
    edgecolors : str or sequence of str, optional
        Marker-edge colors.
    linewidths : float or sequence of float, optional
        Marker-edge widths.
    zorder : float, default 2
        Drawing order.
    add_labels : bool, default False
        Let xarray add labels.
    **kwargs
        Additional arguments forwarded to xarray scatter plotting.

    Returns
    -------
    matplotlib.collections.PathCollection
        Created scatter primitive.
    """

    options = is_defined(
        x=x,
        y=y,
        hue=hue,
        markersize=markersize,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        marker=marker,
        s=size,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths,
        zorder=zorder,
        add_labels=add_labels,
        add_colorbar=False,
    )
    options.update(kwargs)
    options = is_geoaxes(ax, options)
    return data.plot.scatter(ax=ax, **options)


def plot_quiver(
    u: xr.DataArray,
    v: xr.DataArray,
    fig: Figure,
    ax: AxesType,
    *,
    x: str = "lon",
    y: str = "lat",
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
) -> tuple[Quiver, QuiverKey | None]:
    """Draw vector arrows and optionally add a quiver key.

    The key uses the same layout strategy as the previous implementation: a
    temporary horizontal auxiliary axis provides the reserved-region geometry,
    and a transparent persistent axis keeps that region in the figure layout.
    ``key_x`` and ``key_y`` remain the actual key anchor in axes coordinates.

    Parameters
    ----------
    u, v : xarray.DataArray
        Zonal and meridional vector components.
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Destination axis.
    x, y : str, default "lon", "lat"
        Horizontal coordinate names.
    subsample : int or tuple of int, default (1, 1)
        Spatial stride used to thin vectors.
    add_key : bool, default True
        Add a reference vector key. The key is omitted only when explicitly
        set to ``False``.
    key_magnitude : int or float, optional
        Reference magnitude. The 75th percentile is used when omitted.
    key_units : str, optional
        Units appended to the key label. Matching component ``units``
        attributes are used when omitted.
    key_x, key_y : float, default 0.1, -0.045
        Quiver-key anchor in axis coordinates.
    scale : float, optional
        Matplotlib quiver scale.
    color : str, optional
        Arrow color.
    width : float, optional
        Arrow-shaft width.
    zorder : float, default 4
        Drawing order.
    **kwargs
        Additional arguments forwarded to :meth:`Axes.quiver`.

    Returns
    -------
    quiver : matplotlib.quiver.Quiver
        Vector primitive.
    quiver_key : matplotlib.quiver.QuiverKey or None
        Reference key when requested.
    """
    u, v = validate_vector_components(u, v)
    assert u is not None and v is not None

    x_stride, y_stride = normalize_subsample(subsample)
    if (x not in u.coords or y not in u.coords) or (
        x not in v.coords or y not in v.coords
    ):
        x, y = get_spatial_dims(u)

    selection = {
        x: slice(None, None, x_stride),
        y: slice(None, None, y_stride),
    }
    u_selected = u.isel(selection)
    v_selected = v.isel(selection)

    u_selected = set_edges_to_nan(u_selected, dims=(x, y))
    v_selected = set_edges_to_nan(v_selected, dims=(x, y))

    x_values = u_selected.coords[x]
    y_values = u_selected.coords[y]
    if x_values.ndim == 2 and y_values.ndim == 2:
        x2d = np.asarray(x_values.values)
        y2d = np.asarray(y_values.values)
    else:
        x2d, y2d = np.meshgrid(x_values.values, y_values.values)

    options = is_defined(scale=scale, color=color, width=width, zorder=zorder)
    options.update(kwargs)
    options = is_geoaxes(ax, options)

    quiver = ax.quiver(
        x2d,
        y2d,
        np.asarray(u_selected.values),
        np.asarray(v_selected.values),
        angles="xy",
        **options,
    )

    quiver_key: QuiverKey | None = None
    if add_key:
        if key_magnitude is None:
            key_magnitude = get_quiver_key_mag(u_selected, v_selected)

        if key_units is None:
            u_units = str(u_selected.attrs.get("units", ""))
            v_units = str(v_selected.attrs.get("units", ""))
            if u_units != v_units:
                raise ValueError("u and v units must match when adding a quiver key")
            key_units = u_units

        label = f"{key_magnitude} {key_units or ''}".strip()

        # Preserve the previous layout mechanism. The temporary cax supplies the
        # target size and lower padding; the persistent transparent cax reserves
        # that region after the temporary one is removed.
        temporary_cax = get_cax(
            fig=fig,
            axes=ax,
            orientation="horizontal",
            subplots=False,
            adjust=False,
            pad_bottom=True,
        )
        bbox = temporary_cax.get_position()
        temporary_cax.remove()

        # Resolve the actual key anchor from the public arguments. The regression
        # in the intermediate implementation came from deriving the key anchor
        # from bbox while positioning cax from key_x/key_y.
        key_x_ax = float(key_x)
        key_y_ax = float(key_y)
        key_x_fig, key_y_fig = fig.transFigure.inverted().transform(
            ax.transAxes.transform((key_x_ax, key_y_ax))
        )

        # Match the previous Cartopy offset, but keep figure and axes coordinate
        # systems separate.
        padx_fig = 0.0
        pady_fig = 0.0
        if isinstance(ax, cgeo.GeoAxes):
            padx_fig = 0.05 * bbox.width
            pady_fig = 0.05 * bbox.height

        axis_bbox = ax.get_position()
        padx_ax = padx_fig / max(axis_bbox.width, np.finfo(float).eps)
        pady_ax = pady_fig / max(axis_bbox.height, np.finfo(float).eps)

        cax_left = key_x_fig + padx_fig
        cax_bottom = key_y_fig - 0.5 * bbox.height + pady_fig
        cax_right = bbox.x1
        cax_width = max(cax_right - cax_left, np.finfo(float).eps)

        key_cax = fig.add_axes(
            [cax_left, cax_bottom, cax_width, bbox.height],
            label=f"<quiver-key-{id(quiver)}>",
            zorder=1,
        )
        key_cax.set_frame_on(False)
        key_cax.set_xticks([])
        key_cax.set_yticks([])
        key_cax.set_in_layout(True)

        quiver_key = ax.quiverkey(
            quiver,
            X=key_x_ax + padx_ax,
            Y=key_y_ax + pady_ax,
            U=key_magnitude,
            label=label,
            labelpos="E",
            coordinates="axes",
            zorder=zorder,
            fontproperties={"size": 10},
        )
        quiver_key.set_clip_on(False)
        quiver_key.text.set_clip_on(False)
        quiver_key.set_in_layout(True)

    plt.sca(ax)
    return quiver, quiver_key


def plot_significance(
    data: xr.DataArray,
    fig: Figure,
    ax: AxesType,
    *,
    x: str = "lon",
    y: str = "lat",
    level: float = 0.05,
    color: str = "grey",
    alpha: float = 0.3,
    marker: str | None = None,
    edgecolors: str | None = None,
    subsample: int | tuple[int, int] | list[int] = (1, 1),
    size: float = 0.25,
    zorder: float = 3.0,
) -> PathCollection:
    """Draw markers where a p-value field is below a threshold.

    Parameters
    ----------
    data : xarray.DataArray
        Pointwise p-values.
    fig : matplotlib.figure.Figure
        Figure containing ``ax``.
    ax : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Destination axis.
    x, y : str, default "lon", "lat"
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
        Marker-edge color. Defaults to ``color``.
    subsample : int or tuple of int, default (1, 1)
        Spatial stride used to thin markers.
    size : float, default 0.25
        Marker area.
    zorder : float, default 3
        Drawing order.

    Returns
    -------
    matplotlib.collections.PathCollection
        Created significance-marker primitive.
    """

    data = validate_data(data)
    if x not in data.coords or y not in data.coords:
        x, y = get_spatial_dims(data)
    x_stride, y_stride = normalize_subsample(subsample)
    selected = data.isel(
        {
            x: slice(None, None, x_stride),
            y: slice(None, None, y_stride),
        }
    )
    frame = selected.rename("p_value").to_dataframe().reset_index()
    frame = frame.loc[frame["p_value"] < level].dropna(subset=["p_value"])
    options = is_geoaxes(
        ax,
        {
            "color": color,
            "alpha": alpha,
            "s": size,
            "marker": marker,
            "edgecolors": edgecolors or color,
            "zorder": zorder,
        },
    )
    return ax.scatter(frame[x], frame[y], **options)
