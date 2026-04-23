import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from multiprocessing import Pool
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
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure

from .tools import RicedDict, get_fsig, n_cpus, tmp


@dataclass(frozen=True)
class MapPlot:
    fig: Figure
    ax: Axes | cgeo.GeoAxes
    artist: Artist


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
    u_units = u.attrs.get("units", "")
    v_units = v.attrs.get("units", "")
    assert u_units == v_units, "Units of u and v components must match"
    u_units = u_units.replace("[", "").replace("]", "").strip()

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

    Q = ax.quiver(lon2d, lat2d, u.values, v.values, transform=transform, **kwargs)

    if not U:
        speed = (u**2 + v**2) ** 0.5
        U = np.round(speed.median(skipna=True).values)
        U = int(U)

    if new_ax:
        cax = get_cax(
            axes=ax,
            orientation="horizontal",
            quiver=True,
            xaxis_ticks=xaxis_ticks,
        )
        bbox = cax.get_position()

        if "%" in u_units:
            u_units = u_units.replace("%", "\\%")
        label = rf"{U} [${u_units}$]"

        ax.quiverkey(
            Q,
            X=0.45,
            Y=bbox.y0,
            U=U,
            label=label,
            labelpos="E",
            coordinates="figure",
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
        if not cbar_label:
            long_name = long_name.capitalize()
            units = units.replace("[", "").replace("]", "").strip()
            cbar_label = rf"{long_name} [${units}$]"

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
    levels: int | list = None,
    extend: str = None,
    cyclic: bool = False,
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
    p_values: xr.DataArray = None,
    p_value_kwargs: dict = None,
    u_component: xr.DataArray = None,
    v_component: xr.DataArray = None,
    quiver_kwargs: dict = None,
    **kwargs,
) -> MapPlot:
    """
    Visualize a two-dimensional ``xarray.DataArray`` on a Cartopy map.

    This function provides a high-level interface for rendering geospatial
    fields using Cartopy projections together with the native xarray plotting
    API. It supports multiple plotting backends, configurable projections,
    geographic features, and optional overlays such as statistical
    significance markers and vector fields.

    The input array is internally reduced using ``DataArray.squeeze()``.
    After squeezing, the data must be two-dimensional.

    Parameters
    ----------
    da : xarray.DataArray
        Two-dimensional spatial field to visualize. The array must contain
        identifiable horizontal coordinates, typically longitude and latitude.

    x, y : str, optional
        Names of the horizontal coordinate dimensions. If not provided, the
        first two dimensions of the array are used.

    ax : cartopy.mpl.geoaxes.GeoAxes, optional
        Target axis for plotting. If omitted, a new figure and GeoAxes are
        created using the specified projection.

    projection : {"PlateCarree", "Mercator", "Robinson", "Mollweide",
                "Orthographic", "LambertConformal", "AlbersEqualArea",
                "Stereographic", "NorthPolarStereo", "SouthPolarStereo"}, default "PlateCarree"
        Cartopy map projection used to construct the GeoAxes.

    central_longitude : float, optional
        Central longitude of the projection, when supported.

    central_latitude : float, optional
        Central latitude of the projection, when supported.

    global_extent : bool, default False
        If True, sets the map extent to the full globe.

    set_extent : tuple of float, optional
        Geographic extent specified as ``(lon_min, lon_max, lat_min, lat_max)``
        in degrees.

    figsize : tuple of float, optional
        Figure size in inches. Only applied when a new figure is created.

    Plot configuration
    ------------------
    method : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
        Plotting method delegated to the xarray plotting interface.

    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used to render the data field.

    norm : matplotlib.colors.Normalize, optional
        Normalization applied prior to colormap mapping.

    vmin, vmax : float, optional
        Lower and upper bounds for color scaling.

    levels : int or sequence of float, optional
        Contour levels for contour-based methods.

    extend : {"neither", "both", "min", "max"}, optional
        Colorbar extension indicating out-of-range values.

    cyclic : bool, default False
        If True, a cyclic point is appended along the longitudinal dimension.
        This is required for seamless rendering of global fields spanning
        0-360° or -180-180°.

    robust : bool, default False
        If True, color limits are computed from the 2nd and 98th percentiles,
        reducing sensitivity to extreme values.

    title : str, optional
        Title applied to the axes.

    orientation : {"vertical", "horizontal"}, default "vertical"
        Orientation of the colorbar.

    add_colorbar : bool, default True
        If True, a colorbar is added to the plot.

    drawedges : bool, default False
        If True, draw edges between colorbar segments.

    cbar_label : str, optional
        Label for the colorbar. If not provided, the label is inferred from
        ``data.attrs["long_name"]`` and ``data.attrs["units"]`` when available.

    Map features
    ------------
    gridlines : bool, default False
        Draw latitude and longitude gridlines with labels.

    coastlines : bool, default True
        Add coastline boundaries.

    borders : bool, default True
        Add national borders.

    states : bool, default True
        Add subnational administrative boundaries.

    ocean : bool, default True
        Render ocean background.

    land : bool, default True
        Render land background.

    lakes : bool, default False
        Add lake features.

    rivers : bool, default False
        Add river features.

    Overlays
    --------
    p_values : xarray.DataArray, optional
        Field of p-values associated with ``data``. Locations satisfying
        ``p < threshold`` are marked to indicate statistical significance.

    p_value_kwargs : dict, optional
        Keyword arguments passed to ``plot_pvalues`` (e.g., ``level``,
        ``alpha``, ``subsample``).

    u_component, v_component : xarray.DataArray, optional
        Zonal and meridional vector components for quiver plotting. Both
        must share identical dimensions and units.

    quiver_kwargs : dict, optional
        Keyword arguments passed to ``plot_quiver`` (e.g., ``subsample``,
        ``scale``, ``width``, ``U``).

    **kwargs
        Additional keyword arguments forwarded to the underlying xarray
        plotting method.

    Returns
    -------
    MapPlot
        Dataclass containing:
        - ``fig`` : matplotlib.figure.Figure
        - ``ax`` : cartopy GeoAxes
        - ``artist`` : primary matplotlib artist returned by the plot method

    Notes
    -----
    - The data are assumed to be defined in a Plate Carrée coordinate system
    (longitude and latitude in degrees).
    - For global fields, enabling ``cyclic=True`` prevents visual discontinuities
    at the longitudinal boundary.
    - When plotting vector fields, subsampling via ``subsample`` is recommended to
    reduce visual clutter.
    """
    # if data is 3D, raise error
    da = da.squeeze()
    long_name = da.attrs.get("long_name", "")
    units = da.attrs.get("units", da.name)

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

    # xarray methords

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

    # we want all possible args
    plot, pargs = _data_plot(da, method)
    all_args = dict(locals())
    all_args.update(kwargs)

    pkwargs = {k: v for k, v in all_args.items() if k in pargs}
    pkwargs["ax"] = ax
    pkwargs["add_colorbar"] = False
    pkwargs["zorder"] = 1
    pkwargs["transform"] = transform
    del pkwargs["extend"]
    del pkwargs["kwargs"]
    del pkwargs["figsize"]

    artist = plot(**pkwargs, extend=extend)
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
    i,
    dim_value,
    dim,
    dpi,
    args,
    session_tmp_dir,
    data_slice,
    u_slice,
    v_slice,
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

    plot = mapplot(**local_kwargs)
    plot.ax.set_title(title)

    plt.savefig(fname, dpi=dpi, bbox_inches="tight")

    plot.ax.clear()
    plt.close(plot.fig)

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
    # Animation control will be popped from args
    dim: str = "time",
    *,
    indices: tuple | list | np.ndarray = None,
    outfile: Path = None,
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
    set_extent: tuple[float, float, float, float] = None,
    figsize: tuple[float, float] = None,
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
    levels: int | list = None,
    extend: str = None,
    cyclic: bool = False,
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
    Animate a spatial ``xarray.DataArray`` on a Cartopy map.

    This function generates a sequence of map-based visualizations by
    iterating over a specified dimension, typically time. Each slice is
    rendered using the same plotting interface as ``mapplot``, ensuring
    consistent projection, styling, and overlays across frames.

    Frames may be rendered sequentially or in parallel and optionally
    encoded to a video file.

    Parameters
    ----------
    da : xarray.DataArray
        Array containing a spatial field with at least two dimensions and an
        additional animation dimension. After selecting along ``dim``, each
        slice must be two-dimensional.

    dim : str, default "time"
        Name of the dimension over which frames are generated.

    indices : sequence of int, optional
        Subset of indices along ``dim`` to include in the animation. If not
        provided, all indices along the dimension are used.

    outfile : str or pathlib.Path, optional
        Output file path. If provided, frames are encoded and written to
        disk. If None, a matplotlib animation object is returned for
        interactive display.

    quality : {"low", "medium", "high"}, default "medium"
        Output quality preset controlling figure resolution and encoding
        parameters.

    fps : int, default 10
        Frames per second of the output animation.

    parallel : bool, default True
        If True, frames are rendered concurrently across available CPU
        cores. This improves performance for long sequences but increases
        memory usage.

    Spatial configuration
    ---------------------
    x, y : str, optional
        Names of the horizontal coordinate dimensions. If omitted, the first
        two spatial dimensions are used.

    projection : {"PlateCarree", "Mercator", "Robinson", "Mollweide",
                "Orthographic", "LambertConformal", "AlbersEqualArea",
                "Stereographic", "NorthPolarStereo", "SouthPolarStereo"},
        default "PlateCarree"
        Cartopy map projection used for all frames.

    central_longitude : float, optional
        Central longitude of the projection when supported.

    central_latitude : float, optional
        Central latitude of the projection when supported.

    global_extent : bool, default False
        If True, sets the map extent to the entire globe.

    set_extent : tuple of float, optional
        Geographic extent specified as ``(lon_min, lon_max, lat_min, lat_max)``
        in degrees.

    figsize : tuple of float, optional
        Figure size in inches.

    Plot configuration
    ------------------
    method : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
        Plotting method delegated to the xarray plotting interface.

    cmap : str or matplotlib.colors.Colormap, optional
        Colormap applied to the data.

    norm : matplotlib.colors.Normalize, optional
        Normalization applied prior to color mapping.

    vmin, vmax : float, optional
        Lower and upper bounds for color scaling. These should typically be
        fixed across frames to ensure visual consistency.

    levels : int or sequence of float, optional
        Contour levels for contour-based plotting methods.

    extend : {"neither", "both", "min", "max"}, optional
        Colorbar extension indicating out-of-range values.

    cyclic : bool, default False
        If True, a cyclic point is appended along the longitudinal dimension
        to avoid discontinuities in global maps.

    robust : bool, default False
        If True, color limits are computed from the 2nd and 98th percentiles
        of each frame. This may introduce frame-to-frame variability in the
        color scale.

    title : str, optional
        Base title applied to each frame. Dimension values are typically
        appended during rendering.

    orientation : {"vertical", "horizontal"}, default "vertical"
        Orientation of the colorbar.

    add_colorbar : bool, default True
        If True, a colorbar is added to each frame.

    drawedges : bool, default False
        If True, draw edges between colorbar segments.

    cbar_label : str, optional
        Label for the colorbar. If not provided, the label is inferred from
        ``data.attrs["long_name"]`` and ``data.attrs["units"]`` when available.

    Map features
    ------------
    gridlines : bool, default False
        Draw latitude and longitude gridlines.

    coastlines : bool, default True
        Add coastline boundaries.

    borders : bool, default True
        Add national borders.

    states : bool, default True
        Add subnational administrative boundaries.

    ocean : bool, default True
        Render ocean background.

    land : bool, default True
        Render land background.

    lakes : bool, default False
        Add lake features.

    rivers : bool, default False
        Add river features.

    Overlays
    --------
    u_component, v_component : xarray.DataArray, optional
        Zonal and meridional vector components used for quiver plotting.
        These must share dimensions with ``data`` and include the animation
        dimension ``dim``.

    quiver_kwargs : dict, optional
        Keyword arguments passed to ``plot_quiver`` (e.g., ``subsample``,
        ``scale``, ``width``).

    **kwargs
        Additional keyword arguments forwarded to the underlying xarray
        plotting method.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        Animation object if ``outfile`` is None. If an output file is
        specified, the animation is written to disk and the function returns
        None.

    Notes
    -----
    - All frames share a common projection and map configuration.
    - For physical interpretability, fixed color limits (``vmin``, ``vmax``)
    are recommended across frames.
    - Parallel rendering improves performance for large datasets but may
    increase memory consumption.
    - This function is intended for geophysical fields evolving along a
    single dimension, such as time.
    """

    args = RicedDict(locals())

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

    da, u_component, v_component = _validate_animation_inputs(
        dim, da, u_component, v_component
    )

    session_tmp_dir = Path(tempfile.mkdtemp())
    dpi_map = {"low": 300, "medium": 600, "high": 1200}
    dpi = dpi_map.get(quality, 600)

    def _isel_dim(da: xr.DataArray | None, i: int):
        return None if da is None else da.isel({dim: i}).load()

    if indices is None:
        indices = range(da.sizes[dim])

    tasks = [
        (
            i,
            da[dim].values[i],
            dim,
            dpi,
            args,
            session_tmp_dir,
            _isel_dim(da, i),
            _isel_dim(u_component, i),
            _isel_dim(v_component, i),
        )
        for i in indices
    ]
    if parallel:
        processes = min(len(indices), n_cpus)
        with Pool(processes=processes) as pool:
            pool.starmap(_mapplot_i, tasks)

    else:
        for task in tasks:
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
