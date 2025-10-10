import functools
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Literal, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point

from .tools import CPU_COUNT, get_func_signature


def get_cbar_axes(
    *,
    fig: plt.Figure = None,
    axes: plt.Axes = None,
    subplots: bool = False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    pad: float = 0.04,
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
    pad : float, optional
        The padding between the colorbar axes and the existing axes. Default is 0.04. Try 0.04 and 0.05
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

    def _create_cax(y0, x0, y1, x1, x_len, y_len):
        if orientation == "vertical":

            bottommost = y0
            height = y_len
            rightmost = pad * height + x1
            norm = pad if fig_height < 5 else 0.05
            width = norm * height

            cax = fig.add_axes([rightmost, bottommost, width, height])

        elif orientation == "horizontal":

            rightmost = x0
            width = x_len
            bottommost = y0 - (0.12 * width)
            norm = pad if fig_width < 5 else 0.05
            height = norm * width

            cax = fig.add_axes([rightmost, bottommost, width, height])

        return cax

    if not subplots:
        pos = axes.get_position()
        fig_x_len = pos.x1 - pos.x0
        fig_y_len = pos.y1 - pos.y0
        cax = _create_cax(pos.y0, pos.x0, pos.y1, pos.x1, fig_x_len, fig_y_len)

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
        )

    return cax


def make_lon_cyclic(obj: xr.DataArray, longitude: str = "lon"):
    """
    Add a cyclic point to a DataArray along the specified longitude dimension.

    Parameters
    ----------
    obj : xarray.DataArray
        The input DataArray to which a cyclic point will be added.
    longitude : str, optional
        The name of the longitude dimension in the DataArray. Default is 'lon'.
    """

    if not isinstance(obj, xr.DataArray):
        raise ValueError("Input object must be an xarray.DataArray.")
    if longitude not in obj.dims:
        raise ValueError(f"Longitude dimension '{longitude}' not found in data dims.")

    attrs = obj.attrs
    cyclic_data, cyclic_longitude = add_cyclic_point(obj.values, coord=obj[longitude])
    coords = {dim: obj.coords[dim] for dim in obj.dims}
    coords["lon"] = cyclic_longitude

    return xr.DataArray(cyclic_data, dims=obj.dims, coords=coords, attrs=attrs)


def create_map_figure(
    *,
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
    figsize: tuple[float, float] = None,
    global_extent: bool = False,
    central_longitude: float = 0.0,
    states: bool = True,
    borders: bool = True,
    facecolor: str = "grey",
    edgecolor: str = "face",
    coastlines: bool = True,
    ocean: bool = True,
    land: bool = True,
):
    """
    Create a Cartopy map figure using a specified map projection and extent.

    Parameters
    ----------
    projection : {"PlateCarree", "Mercator", "Robinson", "Mollweide", "Orthographic",
                  "LambertConformal", "AlbersEqualArea", "Stereographic",
                  "NorthPolarStereo", "SouthPolarStereo"}, default "PlateCarree"
        The Cartopy map projection to use. Selects from common Cartopy projections.

    figsize : tuple of float, optional
        Matplotlib figure size in inches as (width, height). If None, uses the default size.

    global_extent : bool, default False
        If True, sets the extent of the map to the full globe.

    central_longitude : float, default 0.0
        Central longitude of the projection. Used in projections where applicable.

    central_latitude : float, default 0.0
        Central latitude of the projection. Relevant for Orthographic and some regional projections.

    coastlines : bool, default True
        If True, adds coastlines to the map.

    ocean : bool, default False
        If True, shades ocean areas with a default image and hides land.

    land : bool, default True
        If True, shades land areas with a default image and hides ocean.

    states : bool, default True
        If True, overlays U.S. state boundaries (visible in North America extent).

    borders : bool, default True
        If True, overlays international country borders.

    facecolor : str, default "grey"
        Fill color for continents (if `only_ocean=False`).

    edgecolor : str, default "face"
        Edge color for coastlines, borders, and other map features.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure.

    ax : matplotlib.axes.Axes
        The Cartopy-aware map axes.
    """

    projections = {
        "PlateCarree": ccrs.PlateCarree,
        "Mercator": ccrs.Mercator,
        "Robinson": ccrs.Robinson,
        "Mollweide": ccrs.Mollweide,
        "Orthographic": ccrs.Orthographic,
        "LambertConformal": ccrs.LambertConformal,
        "AlbersEqualArea": ccrs.AlbersEqualArea,
        "Stereographic": ccrs.Stereographic,
        "NorthPolarStereo": ccrs.NorthPolarStereo,
        "SouthPolarStereo": ccrs.SouthPolarStereo,
    }

    crt_projection = projections.get(projection, ccrs.PlateCarree)
    crt_projection = crt_projection(central_longitude=central_longitude)

    fig, ax = plt.subplots(subplot_kw={"projection": crt_projection}, figsize=figsize)

    if global_extent:
        ax.set_global()

    if coastlines:
        ax.add_feature(cfeature.COASTLINE)

    if ocean and not land:

        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "land",
                "50m",
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=0.5,
            )
        )

    if states:
        ax.add_feature(cfeature.STATES, linestyle="-", alpha=0.3)
    if borders:
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.3)

    return fig, ax


def plot_p_values(
    ax: plt.Axes,
    data: xr.DataArray,
    level: float = 0.05,
    color: str = "grey",
    alpha: float = 1,
    s: float = 1,
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
    s : float, optional
        Size of the points to plot. Default is 1.
    """

    if "lon" not in data.dims or "lat" not in data.dims:
        raise ValueError("DataArray must contain 'lon' and 'lat' dimensions.")

    # p_values = xr.where(data > level, 1, np.nan)

    p_values = data.to_dataframe(name="p_values").reset_index()
    p_values = p_values.query("p_values < @level")

    # replace where p_values < 1with NaN pandas

    p_values = p_values.dropna()

    ax.scatter(
        p_values["lon"],
        p_values["lat"],
        transform=ccrs.PlateCarree(),
        color=color,
        alpha=alpha,
        s=s,
    )

    return ax


def update_animation_frame(i, data, dim, kwargs, dpi, session_tmp_dir):
    da = data.isel({dim: i})
    t = (
        da[dim].values.item()
        if dim in da.dims
        else data[dim].isel({dim: i}).values.item()
    )

    if dim == "time":
        if isinstance(t, int):
            t = np.datetime64(t, "ns")
            t = pd.to_datetime(t)
        t = t.strftime("%Y-%m-%d %H:%M")
    t = f"{dim}: {t}"

    local_kwargs = kwargs.copy()
    local_kwargs["data"] = da

    fname = session_tmp_dir / f"{i:06d}.png"
    fig, ax, _ = cartplot(**local_kwargs)
    ax.set_title(t)
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    ax.clear()
    plt.close(fig)
    return None


def animate_data(args):
    kwargs = args
    data = kwargs["data"]
    dim = kwargs["animate_dim"]
    indices = kwargs["animate_indices"]
    fps = kwargs["animate_fps"]
    quality = kwargs["animate_quality"]
    out_file = kwargs["animate_out_file"]
    parallel = kwargs["parallel"]
    # clean kwargs
    for k in [
        "data",
        "animate_dim",
        "animate_indices",
        "animate_fps",
        "animate_quality",
        "animate_out_file",
        "animate",
        "parallel",
    ]:
        del kwargs[k]

    if dim not in data.dims:
        raise ValueError(f"{dim} not found in data.dims {data.dims}")

    session_tmp_dir = Path(tempfile.mkdtemp())
    dpi_map = {"low": 300, "medium": 600, "high": 1200}
    dpi = dpi_map.get(quality, 600)

    if indices is None:
        indices = range(data.sizes[dim])
        print(f"Animating all {data.sizes[dim]} values along {dim}.")

    if parallel:
        if len(indices) >= CPU_COUNT:
            processes = CPU_COUNT
        else:
            processes = len(indices)
        tasks = [(i, data.copy(), dim, kwargs, dpi, session_tmp_dir) for i in indices]
        with Pool(processes=processes) as pool:
            pool.starmap(update_animation_frame, tasks)

    else:

        _warn_msg = (
            f"Generating {data.sizes[dim]} frames without parallel processing. "
            "Set parallel=True for faster animation."
        )

        if len(indices) > 100:
            print(_warn_msg)

        for i in list(indices):
            update_animation_frame(i, data, dim, kwargs, dpi, session_tmp_dir)

    # ---- ffmpeg encode (MP4 only) ----
    if not out_file:
        out_file = Path("videos/animation.mp4")
    else:
        out_file = Path(f"videos/{out_file}")
        out_file = out_file.with_suffix(".mp4")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    input_pattern = str(Path(session_tmp_dir) / "%06d.png")

    error = 0

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
            out_file,
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

    # optional inline display (Jupyter)
    if "ipykernel" in sys.modules and error == 0:
        from IPython.display import Video, display

        return display(
            Video(
                out_file,
                embed=True,
                html_attributes="controls autoplay loop",
                width=800,
                height=600,
            )
        )
    else:
        return None


def cartplot(
    data: xr.DataArray,
    *,
    x: str = None,
    y: str = None,
    plot_type: Literal[
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
    central_longitude: float = -100,
    global_extent: bool = False,
    figsize: tuple[float, float] = None,
    cmap: Union[str, mcolors.Colormap] = None,
    vmin: float = None,
    vmax: float = None,
    levels: Union[int, list] = None,
    robust: bool = False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    add_colorbar: bool = True,
    drawedges: bool = False,
    cbar_label: str = None,
    gridlines: bool = False,
    coastlines: bool = True,
    borders: bool = True,
    states: bool = True,
    ocean: bool = True,
    land: bool = True,
    facecolor: str = "#d3d3d3",
    edgecolor: str = "face",
    animate: bool = False,
    animate_dim: str = "time",
    animate_indices: Union[tuple, list, np.ndarray] = None,
    animate_out_file: PathLike = None,
    animate_quality: Literal["low", "medium", "high"] = "medium",
    animate_fps: int = 10,
    parallel: bool = False,
    **kwargs,
):
    """
    Plot a 2D xarray.DataArray on a Cartopy map with flexible projection and styling options.

    Parameters
    ----------
    data : xr.DataArray
        2D array with spatial dimensions (e.g., lat/lon or x/y).
    x, y : str, optional
        Names of spatial dimensions. Defaults to first and second dims.
    plot_type : str, default "default"
        Plot style: "pcolormesh", "contourf", "contour", "imshow", or "default".
    projection : str, default "PlateCarree"
        Cartopy map projection.
    central_longitude : float, default 0.0
        Central longitude for projection.
    global_extent : bool, default False
        Show full globe if True.
    figsize : tuple, optional
        Figure size in inches.
    cmap : str or Colormap, optional
        Colormap for data.
    vmin, vmax : float, optional
        Color scale limits.
    levels : int or list, optional
        Contour levels for "contour" or "contourf".
    robust : bool, default False
        Use 2nd-98th percentiles for color scale if vmin/vmax not set.
    orientation : str, default "vertical"
        Colorbar orientation.
    add_colorbar : bool, default True
        Show colorbar.
    drawedges : bool, default False
        Draw edges on colorbar.
    cbar_label : str, optional
        Label for colorbar.
    gridlines : bool, default False
        Show lat/lon gridlines.
    coastlines, borders, states : bool, default True
        Show geographic features.
    ocean, land : bool, default True
        Show physical features.
    facecolor : str, default "#d3d3d3"
        Land fill color.
    edgecolor : str, default "face"
        Border edge color.
    animate : bool, default False
        Enable animation over a dimension.
    animate_dim : str, default "time"
        Dimension to animate.
    animate_indices : list, optional
        Indices to animate.
    animate_out_file : str, optional
        Output file path for animation.
    animate_quality : str, default "medium"
        Animation quality.
    animate_fps : int, default 10
        Frames per second.
    parallel : bool, default False
        Use parallel processing for animation.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The map axes.
    p : matplotlib artist
        The plotted data object.

    Notes
    -----
    Requires `cartopy` and `matplotlib`. Designed for 2D geospatial data.
    """

    allargs = locals()

    # Ensure data is 2D unless animating
    data = data.squeeze(drop=True)
    if data.ndim > 2 and not animate:
        raise ValueError(
            f"Data with shape {data.shape} has {data.ndim} dimensions.\n \
            Please set animate=True and specify the dimension to animate\n \
            or select a 2D slice (e.g., using .isel or .sel)."
        )
    if data.ndim < 2:
        raise ValueError(
            f"Data with shape {data.shape} has less than 2 dimensions.\n \
            cartplot requires a 2D DataArray (e.g., with lat/lon or x/y)."
        )

    if animate:
        return animate_data(allargs)

    map_kwags = get_func_signature(create_map_figure)
    map_kwags = {k: v for k, v in allargs.items() if k in map_kwags}
    plot_kwargs = {k: v for k, v in allargs.items() if k not in map_kwags}

    figure, ax = create_map_figure(**map_kwags)

    plot_funcs = {
        "default": [data.plot, get_func_signature(data.plot)],
        "pcolormesh": [data.plot.pcolormesh, get_func_signature(data.plot.pcolormesh)],
        "contourf": [data.plot.contourf, get_func_signature(data.plot.contourf)],
        "contour": [data.plot.contour, get_func_signature(data.plot.contour)],
        "imshow": [data.plot.imshow, get_func_signature(data.plot.imshow)],
    }

    if plot_type not in plot_funcs:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. Choose from {list(plot_funcs)}."
        )

    plot_kwargs = {}
    for _, sig in plot_funcs.values():
        for k, v in sig.items():
            plot_kwargs.setdefault(k, v)

    # plot sig = combine all signature into 1 dict, removing duplicates, we will pass that to everything

    plot_func = plot_funcs[plot_type][0]

    plot_kwargs = {k: v for k, v in allargs.items() if k in plot_kwargs}
    plot_kwargs["ax"] = ax
    plot_kwargs["add_colorbar"] = False
    plot_kwargs["transform"] = ccrs.PlateCarree()
    del plot_kwargs["kwargs"]

    plot_obj = plot_func(**plot_kwargs)

    if ocean and not land:
        ax.add_feature(cfeature.LAND, facecolor="white", zorder=1)
    elif land and not ocean:
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=1)

    if gridlines:
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )

        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True

    if add_colorbar:
        cax = get_cbar_axes(fig=figure, axes=ax, orientation=orientation)

        cb = plt.colorbar(
            plot_obj,
            cax=cax,
            ax=ax,
            orientation=orientation,
            drawedges=drawedges,
        )

        if cbar_label:
            cb.set_label(cbar_label)

        else:

            cbar_label = []
            if "long_name" in data.attrs:
                cbar_label.append(data.attrs["long_name"])
            if "units" in data.attrs:
                cbar_label.append(data.attrs["units"])
            cb.set_label("\n".join(cbar_label))

    return figure, ax, plot_obj


see_data = cartplot


@xr.register_dataarray_accessor("cartopy")
class CartPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @functools.wraps(cartplot)
    def plot(self, *args, **kwargs):
        return cartplot(self._obj, *args, **kwargs)
