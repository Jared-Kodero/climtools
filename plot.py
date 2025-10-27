import shutil
import subprocess
import sys
import tempfile
import warnings
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Literal, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point

from .tools import ConfigMap, get_func_signature, n_cpu


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


def cartplot(
    data: xr.DataArray,
    *,
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
    Plot a 2D or time-evolving `xarray.DataArray` on a Cartopy map with flexible
    projection, style

    Parameters
    ----------
    Data and Coordinates
    --------------------
    data : xr.DataArray
        Two-dimensional or time-evolving array with spatial dimensions
        (e.g., latitude/longitude or x/y).
    x, y : str, optional
        Names of the horizontal dimensions. Defaults to the first and second dims.

    Projection and Layout
    ---------------------
    projection : str, default "PlateCarree"
        Map projection. Options include "Mercator", "Robinson", "Mollweide",
        "Orthographic", "LambertConformal", etc.
    central_longitude : float, default -100
        Central longitude for the map projection.
    central_latitude : float, default None
        Central latitude for the map projection.
    global_extent : bool, default False
        If True, sets extent to show the entire globe.
    figsize : tuple of float, optional
        Figure size in inches (width, height).

    Plot Style and Color Mapping
    ----------------------------
    plot_type : str, default "default"
        Plot method: "pcolormesh", "contourf", "contour", "imshow", or "default".
    cmap : str or Colormap, optional
        Colormap applied to data.
    vmin, vmax : float, optional
        Color scale limits.
    levels : int or sequence, optional
        Contour levels for "contour" and "contourf".
    robust : bool, default False
        If True, ignores outliers using 2nd–98th percentile range.
    orientation : str, default "vertical"
        Colorbar orientation.
    add_colorbar : bool, default True
        Whether to draw a colorbar.
    drawedges : bool, default False
        Draw edges around colorbar color patches.
    cbar_label : str, optional
        Label for the colorbar.

    Map Features
    ------------
    gridlines : bool, default False
        Show latitude/longitude gridlines.
    coastlines : bool, default True
        Draw coastlines.
    borders : bool, default True
        Draw national borders.
    states : bool, default True
        Draw state or province boundaries (if available).
    ocean, land : bool, default True
        Fill ocean and land with default or specified colors.
    facecolor : str, default "#d3d3d3"
        Land face color.
    edgecolor : str, default "face"
        Edge color for land polygons.



    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance.
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Cartopy GeoAxes object.
    p : matplotlib.artist.Artist
        Resulting plot object.

    Notes
    -----
    This function provides a flexible interface for plotting spatial or spatiotemporal
    `xarray.DataArray` objects on Cartopy projections. It supports both static maps
    and animated visualizations with optional geographic and physical map layers.
    Requires `cartopy` and `matplotlib`.
    """

    # if data is 3D, raise error
    if data.ndim > 2:
        raise ValueError("DataArray has more than 2 dimensions.")

    proj = getattr(ccrs, projection)

    proj_all_args = get_func_signature(proj)
    proj_args = {}

    if central_longitude is not None and "central_longitude" in proj_all_args:
        proj_args["central_longitude"] = central_longitude
    if central_latitude is not None and "central_latitude" in proj_all_args:
        proj_args["central_latitude"] = central_latitude

    fig, ax = plt.subplots(
        subplot_kw={"projection": proj(**proj_args)},
        figsize=figsize,
    )

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
                facecolor="#272829",
                alpha=0.5,
            )
        )

    if states:
        ax.add_feature(cfeature.STATES, linestyle="-", alpha=0.3)
    if borders:
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.3)

    # xarray methords

    def _data_plot(data: xr.DataArray, pt: str):

        p = data.plot

        pts = ["pcolormesh", "contourf", "contour", "imshow"]
        funcs = [p] + [getattr(p, m) for m in pts]

        plot_args = {}
        for f in funcs:
            plot_args.update(get_func_signature(f))
        if pt == "default":
            func = p
        elif pt in pts:

            func = getattr(p, pt)
            plot_args.update(get_func_signature(func))

        return func, plot_args

    # we want all possible args
    plot, plot_args = _data_plot(data, plot_type)

    all_args = dict(locals())
    all_args.update(kwargs)

    plot_kwargs = {k: v for k, v in all_args.items() if k in plot_args}
    plot_kwargs["ax"] = ax
    plot_kwargs["add_colorbar"] = False
    plot_kwargs["transform"] = ccrs.PlateCarree()

    del plot_kwargs["kwargs"]

    plot_obj = plot(**plot_kwargs)

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
        cax = get_cbar_axes(fig=fig, axes=ax, orientation=orientation)

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

    return fig, ax, plot_obj


def animate_i_frame(da, i, t, dim, dpi, args, session_tmp_dir):

    t = f"{dim}: {t}"

    local_kwargs = args.copy()
    local_kwargs["data"] = da

    fname = session_tmp_dir / f"{i:06d}.png"
    fig, ax, _ = cartplot(**local_kwargs)
    ax.set_title(t)
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    ax.clear()
    plt.close(fig)
    return None


def animate(
    data: xr.DataArray,
    # Animation control will be popped from args
    dim: str = "time",
    *,
    indices: Union[tuple, list, np.ndarray] = None,
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
):
    """
    Animate a 2D or time-evolving `xarray.DataArray` on a Cartopy map with flexible
    spatial projection, color mapping, and parallel rendering.

    Parameters
    ----------
    Data
    ----
    data : xr.DataArray
        Two-dimensional or time-dependent array containing the field to plot.
        Must include spatial coordinates and optionally a time-like dimension.

    Animation
    ---------
    dim : str, default "time"
        Name of the dimension to animate (e.g., "time").
    indices : sequence of int, optional
        Specific frame indices to include in the animation along `dim`. If None, uses all indices.
    outfile : str or Path, optional
        Path to save the resulting animation. If None, displays interactively.
    quality : {"low", "medium", "high"}, default "medium"
        Output resolution and compression setting.
    fps : int, default 10
        Animation playback speed in frames per second.
    parallel : bool, default False
        Compute animation frames in parallel across available CPUs.

    Spatial Configuration
    ---------------------
    x, y : str, optional
        Names of the horizontal coordinates. Defaults to the first two dimensions.
    projection : str, default "PlateCarree"
        Cartopy map projection name. Supported options include:
        "Mercator", "Robinson", "Mollweide", "Orthographic", "LambertConformal",
        "AlbersEqualArea", "Stereographic", "NorthPolarStereo", "SouthPolarStereo".
    global_extent : bool, default False
        If True, sets the extent to display the full globe.
    figsize : tuple of float, optional
        Figure size (width, height) in inches.
    central_longitude : float, default None
        Central longitude for the map projection.
    central_latitude : float, default None

    Plot Appearance
    ---------------
    plot_type : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
        Rendering method for data visualization.
    cmap : str or Colormap, optional
        Colormap applied to the data field.
    vmin, vmax : float, optional
        Color scaling limits. If None, inferred from data range.
    levels : int or sequence, optional
        Contour levels used in "contour" or "contourf" plots.
    robust : bool, default False
        Exclude outliers using 2nd-98th percentile for color normalization.
    transform : cartopy.crs.Projection, optional
        Coordinate reference system of the data for plotting.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Orientation of the colorbar.
    add_colorbar : bool, default True
        Whether to display a colorbar.
    drawedges : bool, default False
        Draw grid edges on color patches (for `pcolormesh`).
    cbar_label : str, optional
        Label text for the colorbar.

    Map Features
    ------------
    gridlines : bool, default False
        Display latitude/longitude gridlines.
    coastlines : bool, default True
        Draw coastlines on the map.
    borders : bool, default True
        Draw country borders.
    states : bool, default True
        Draw internal administrative boundaries (e.g., states or provinces).
    ocean, land : bool, default True
        Fill ocean and land regions with the specified colors.
    facecolor : str, default "#d3d3d3"
        Land polygon fill color.
    edgecolor : str, default "face"
        Outline color for land polygons.

    Other Parameters
    ----------------
    **kwargs
        Additional arguments passed to the plotting function.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        Animation object if not saved directly to file.

    Notes
    -----
    - Parallel frame rendering significantly accelerates long sequences.
    - Supports any Cartopy projection with a compatible coordinate transform.
    - Intended for geospatial fields such as temperature, precipitation, or pressure.
    """

    args = ConfigMap(locals())

    # pop the above from args
    outfile = args.pop("outfile")
    fps = args.pop("fps")
    dim = args.pop("dim")
    quality = args.pop("quality")
    parallel = args.pop("parallel")
    indices = args.pop("indices")
    data = args.pop("data")

    if dim not in data.dims:
        raise ValueError(f"{dim} not found in data.dims {data.dims}")

    session_tmp_dir = Path(tempfile.mkdtemp())
    dpi_map = {"low": 300, "medium": 600, "high": 1200}
    dpi = dpi_map.get(quality, 600)

    if indices is None:
        indices = range(data.sizes[dim])
    if parallel:
        if len(indices) >= n_cpu:
            processes = n_cpu
        else:
            processes = len(indices)

        tasks = [
            (
                data.isel({dim: i}).load(),
                i,
                data[dim].values[i],
                dim,
                dpi,
                args,
                session_tmp_dir,
            )
            for i in indices
        ]

        with Pool(processes=processes) as pool:
            pool.starmap(animate_i_frame, tasks)

    else:
        if len(indices) > 100:
            warnings.warn(
                f"Generating {data.sizes[dim]} frames sequentially. \
                Set `parallel=True` to enable parallel processing \
                and improve animation speed.",
                UserWarning,
                stacklevel=2,
            )

            for i in list(indices):
                animate_i_frame(
                    data.isel({dim: i}).load(),
                    i,
                    data[dim].values[i],
                    dim,
                    dpi,
                    args,
                    session_tmp_dir,
                )

    # ---- ffmpeg encode (MP4 only) ----
    if not outfile:
        outfile = Path("videos/animation.mp4")
    else:
        outfile = Path(f"videos/{outfile}")
        outfile = outfile.with_suffix(".mp4")

    outfile.parent.mkdir(parents=True, exist_ok=True)
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

    if error == 0:
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


@xr.register_dataarray_accessor("cartopy")
class CartPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, *args, **kwargs):
        return cartplot(self._obj, *args, **kwargs)


@xr.register_dataset_accessor("animate")
class CartAnimateAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plots(self, *args, **kwargs):
        return animate(self._obj, *args, **kwargs)
