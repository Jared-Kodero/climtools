import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes
import geovista as gv
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib.artist import Artist
from matplotlib.axes import Axes

# ---- Plot callback ----
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure

from .pvhtml import build_pvhtml
from .tools import RicedDict, get_func_signature, n_cpus, tmp

vtk.vtkObject.GlobalWarningDisplayOff()
os.environ["PYVISTA_OFF_SCREEN"] = "true"


@dataclass(frozen=True)
class PlotObj:
    fig: Figure
    ax: Axes
    artist: Artist


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


def make_cyclic(obj: xr.DataArray, dim: str = "lon"):
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
    if dim not in obj.dims:
        raise ValueError(f"Longitude dimension '{dim}' not found in data dims.")

    attrs = obj.attrs
    cyclic_data, cyclic_dim = add_cyclic_point(obj.values, coord=obj[dim])
    coords = {dim: obj.coords[dim] for dim in obj.dims}
    coords[dim] = cyclic_dim

    return xr.DataArray(cyclic_data, dims=obj.dims, coords=coords, attrs=attrs)


def plot_pvalues(
    data: xr.DataArray,
    ax: plt.Axes = None,
    level: float = 0.05,
    color: str = "grey",
    alpha: float = 0.3,
    marker: str = None,
    edgecolors: str = None,
    step_size: int = 1,
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
    step_size : int, optional
        Step size for plotting points to reduce overplotting. Default is 1 (plot all points).
    marker : str, optional
        Marker style for the points. Default is None (default marker).
    edgecolors : str, optional
        Edge color for the points. Default is None.
    s : float, optional
        Size of the points to plot. Default is 1.
    """

    if ax is None:
        ax = plt.gca()

    # check if ax is cartopy axis
    transform = None
    if isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
        transform = ccrs.PlateCarree()

    if "lon" not in data.dims or "lat" not in data.dims:
        raise ValueError("DataArray must contain 'lon' and 'lat' dimensions.")

    data = data.isel(lat=slice(None, None, step_size), lon=slice(None, None, step_size))

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


def plot_quiver(
    u: xr.DataArray,
    v: xr.DataArray,
    ax: plt.Axes = None,
    step: int = 1,
    **kwargs,
):
    """
    Plot quiver arrows on a Cartopy axis.
    """

    if ax is None:
        ax = plt.gca()

    import cartopy.mpl.geoaxes as cgeo

    transform = None
    if isinstance(ax, cgeo.GeoAxes):
        import cartopy.crs as ccrs

        transform = ccrs.PlateCarree()

    u = u[::step, ::step]
    v = v[::step, ::step]
    lon2d, lat2d = np.meshgrid(u.lon, u.lat)
    ax.quiver(lon2d, lat2d, u.values, v.values, transform=transform, **kwargs)
    return ax


def mapplot(
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
    robust: bool = False,
    title: str = None,
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
    **kwargs,
) -> PlotObj:
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
    set_extent : BoundingBox, optional
        Set the extent (x0, x1, y0, y1) of the map in the given coordinate system.
    figsize : tuple of float, optional
        Figure size in inches (width, height).

    Plot Style and Color Mapping
    ----------------------------
    method : str, default "default"
        Plot method: "pcolormesh", "contourf", "contour", "imshow", or "default".
    cmap : str or Colormap, optional
        Colormap applied to data.
    norm : Normalize, optional
        Normalization function for the color scale.
    vmin, vmax : float, optional
        Color scale limits.
    levels : int or sequence, optional
        Contour levels for "contour" and "contourf".
    robust : bool, default False
        If True, ignores outliers using 2nd-98th percentile range.
    extend : str, optional
        If 'both', extends color limits to include both ends of the data range.
        If 'min', extends only the minimum limit.
        If 'max', extends only the maximum limit.
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
        Mask ocean and land areas if any is False.
    lakes : bool, default False
        If True, adds lakes to the map.
    rivers : bool, default False
        If True, adds rivers to the map.
    facecolor : str, default "#d3d3d3"
        Land face color.

    Returns
    -------
    PlotObj
        A dataclass containing the figure, axes, and artist objects.
    Notes
    -----
    This function provides a flexible interface for plotting spatial or spatiotemporal
    `xarray.DataArray` objects on Cartopy projections. It supports both static maps
    and animated visualizations with optional geographic and physical map layers.
    Requires `cartopy` and `matplotlib`.
    """

    # if data is 3D, raise error
    data = data.squeeze()
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

    if set_extent:
        ax.set_extent([*set_extent])

    if coastlines:
        ax.add_feature(cfeature.COASTLINE)

    if states:
        ax.add_feature(cfeature.STATES, linestyle="-", alpha=0.3, zorder=3)

    if borders:
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.3, zorder=3)

    if lakes:
        ax.add_feature(cfeature.LAKES, zorder=2)

    if rivers:
        ax.add_feature(cfeature.RIVERS, zorder=2)

    if ocean and not land:
        ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", zorder=2)

    elif land and not ocean:
        ax.add_feature(cfeature.OCEAN, zorder=2)

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

    transform = ccrs.PlateCarree()

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
    plot, plot_args = _data_plot(data, method)

    all_args = dict(locals())
    all_args.update(kwargs)

    plot_kwargs = {k: v for k, v in all_args.items() if k in plot_args}
    plot_kwargs["ax"] = ax
    plot_kwargs["add_colorbar"] = False
    plot_kwargs["zorder"] = 1
    plot_kwargs["transform"] = transform
    del plot_kwargs["extend"]
    del plot_kwargs["kwargs"]

    artist = plot(**plot_kwargs, extend=extend)
    if title:
        plt.title(title)

    if add_colorbar:
        cax = get_cbar_axes(fig=fig, axes=ax, orientation=orientation)

        cb = plt.colorbar(
            artist,
            cax=cax,
            ax=ax,
            orientation=orientation,
            drawedges=drawedges,
            extend=extend,
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

    return PlotObj(fig, ax, artist)


def animate_i_frame(da, i, t, dim, dpi, args, session_tmp_dir):
    t = f"{dim}: {t}"

    local_kwargs = args.copy()
    local_kwargs["data"] = da

    fname = session_tmp_dir / f"{i:06d}.png"
    plot = mapplot(**local_kwargs)

    plot.ax.set_title(t)
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plot.ax.clear()
    plt.close(plot.fig)
    return None


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


def animate(
    data: xr.DataArray,
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
    robust: bool = False,
    title: str = None,
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
        method : {"default", "pcolormesh", "contourf", "contour", "imshow"}, default "default"
            Rendering method for data visualization.
        norm : Normalize, optional
            Normalization function for the color scale.
        cmap : str or Colormap, optional
            Colormap applied to the data field.
        vmin, vmax : float, optional
            Color scaling limits. If None, inferred from data range.
        levels : int or sequence, optional
            Contour levels used in "contour" or "contourf" plots.
        extend : str, optional
            If 'both', extends color limits to include both ends of the data range.
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
        lakes : bool, default False
            If True, adds lakes to the map.
        rivers : bool, default False
            If True, adds rivers to the map
        facecolor : str, default "#d3d3d3"
            Land face color

    .

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

    args = RicedDict(locals())

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
        if len(indices) >= n_cpus:
            processes = n_cpus
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

    user_path = False
    if not outfile:
        outfile = Path(tmp / f"{uuid.uuid4().hex}/{uuid.uuid4().hex}.mp4")

    else:
        outfile = Path(outfile)
        user_path = True

    outfile.parent.mkdir(parents=True, exist_ok=True)
    input_pattern = str(Path(session_tmp_dir) / "%06d.png")

    ffmpeg_encode(input_pattern, outfile, fps, session_tmp_dir, user_path, 0)


def plot_globe(
    data: xr.DataArray,
    x: str = "lon",
    y: str = "lat",
    cmap: str | LinearSegmentedColormap | ListedColormap = "viridis",
    coarsen_by: int = 10,
    outfile: str | Path = None,
):
    """
    Plot this DataArray on a globe using GeoVista.
    """

    resolution = data[x].diff(x).mean().values

    if float(resolution) < 0.25:
        data = data.coarsen(**{x: coarsen_by, y: coarsen_by}, boundary="trim").mean()

    # Create the mesh from the sample data.
    mesh = gv.Transform.from_1d(data.lon, data.lat, data=data.data)

    # Plot the mesh with coastlines.
    p = gv.GeoPlotter()
    sargs = {"title": f"{data.name} / {data.units}"}
    p.add_mesh(mesh, cmap=cmap, scalar_bar_args=sargs)
    p.add_coastlines(color="white")
    p.camera.zoom(1.2)

    if outfile and outfile.endswith(".html"):
        p.export_html(outfile)
    elif outfile:
        p.screenshot(outfile)
    p.show()


def pressure_to_height_std(p, scale=1):
    """
    Convert pressure (hPa) to height (km) using the
    U.S. Standard Atmosphere lapse-rate formulation.
    """

    T0 = 288.15  # K
    p0 = 1013.25  # hPa
    gamma = 6.5  # K/km
    Rd = 287.05  # J/(kg K)
    g = 9.80665  # m/s^2

    exponent = (Rd * gamma) / (g * 1000.0)

    z_km = (T0 / gamma) * (1 - (p / p0) ** exponent)

    return z_km * scale


def to_sphere(X, Y, Z):

    R_earth = 6371.0  # km

    y_rad = np.deg2rad(Y)
    x_rad = np.deg2rad(X)
    z_lev = Z

    lon3, lat3, lev3 = np.meshgrid(x_rad, y_rad, z_lev, indexing="ij")

    r = 1.0 + lev3 / R_earth

    X = r * np.cos(lat3) * np.cos(lon3)
    Y = r * np.cos(lat3) * np.sin(lon3)
    Z = r * np.sin(lat3)

    return X, Y, Z


def check_z_unit(field: xr.DataArray, z_unit: str, scale: int) -> xr.DataArray:
    units = ("hpa", "Pa", "km", "generic")

    if z_unit not in units:
        raise ValueError(f"Unknown unit provided, expected one of {units}")

    field = field.sortby("z", ascending=False)

    if z_unit in ["generic", "km"]:
        return field
    elif z_unit == "hpa":
        field["z"] = pressure_to_height_std(field["z"], scale)
    elif z_unit == "Pa":
        field["z"] = pressure_to_height_std(field["z"] / 100, scale)

    return field


def mapplot3d(
    data: xr.DataArray,
    grid_type: Literal["uniform", "structured"] = "uniform",
    sphere: bool = False,
    window_size: tuple[int, int] = None,
    dim_map: tuple = (("level", "z"), ("lat", "y"), ("lon", "x")),
    z_unit: Literal["hpa", "Pa", "km", "generic"] = "hpa",
    z_unit_scale: int | float = 1,
    cmap: str | LinearSegmentedColormap | ListedColormap = "viridis",
    outfile: Path = None,
    format: Literal["html", "png"] = "png",
    vmin: int | float = None,
    vmax: int | float = None,
    log_scale: bool = False,
    zscale: int | float = 1,
    title: str = None,
    cam_elev: int | float = None,
    cam_azim: int | float = None,
    show_scalar_bar: bool = True,
    xlabel: str = None,
    ylabel: str = None,
    zlabel: str = None,
    n_xlabels: int = None,
    n_ylabels: int = None,
    n_zlabels: int = None,
    opacity: list | Literal["linear", "sigmoid"] = "linear",
    opacity_unit_distance: int | float = None,
    blending: Literal[
        "additive", "maximum", "minimum", "composite", "average"
    ] = "composite",
    padding: int | float = None,
    font_size: int = None,
    animation: bool = False,
):

    if format not in {"html", "png"}:
        raise ValueError("format must be either 'html' or 'png'")

    ipykernel = "ipykernel" in sys.modules
    plot_args = {}
    if ipykernel and not outfile and not animation:
        jupyter_backend = "server"
        plot_args["notebook"] = True
        plot_args["off_screen"] = False

    else:
        jupyter_backend = "static"

    plot_args["window_size"] = window_size
    plot_args["title"] = title
    plot_args = {k: v for k, v in plot_args.items() if v is not None}

    show_args = {}
    show_args["screenshot"] = outfile
    show_args["window_size"] = window_size
    show_args["title"] = title
    show_args["interactive"] = ipykernel and not animation
    show_args = {k: v for k, v in show_args.items() if v is not None}
    screenshot_args = {"filename": outfile, "window_size": window_size}
    screenshot_args = {k: v for k, v in screenshot_args.items() if v is not None}
    cbar_label = (
        f"{data.attrs.get('long_name', data.name)} [{data.attrs.get('units', '')}]"
    )

    dim_map_dict = dict(dim_map)
    dims = [d for axis in ("z", "y", "x") for d, v in dim_map_dict.items() if v == axis]

    missing = [d for d in dims if d not in data.dims]
    if missing:
        raise ValueError(f"{missing} not found in {data.dims}")

    field = data.squeeze(drop=True)
    field = field.rename(dim_map_dict)

    field = check_z_unit(field, z_unit, z_unit_scale)

    vmin = field.min().values if vmin is None else vmin
    vmax = field.max().values if vmax is None else vmax

    field = field.transpose("x", "y", "z")  # xr data array

    data = field.values
    X = field.x.values
    Y = field.y.values
    Z = field.z.values
    axes_ranges = [
        np.min(X),
        np.max(X),
        np.min(Y),
        np.max(Y),
        np.min(Z),
        np.max(Z),
    ]

    if sphere:
        X, Y, Z = to_sphere(X, Y, Z)
    else:
        X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")

    pv.set_jupyter_backend(jupyter_backend)
    grid = None

    if grid_type == "structured" or sphere:
        grid = pv.StructuredGrid(X, Y, Z)

    elif grid_type == "uniform":
        grid = pv.ImageData()
        grid.dimensions = (field.x.size, field.y.size, field.z.size)
        grid.origin = (np.min(X), np.min(Y), np.min(Z))
        dx = abs(float(field["x"].diff("x").mean()))
        dy = abs(float(field["y"].diff("y").mean()))
        dz = abs(float(field["z"].diff("z").mean()))
        dz = round(float(dz))
        grid.spacing = (dx, dy, dz)

    grid.point_data[data.name] = data.flatten(order="F")
    grid.set_active_scalars(data.name)
    p = pv.Plotter(**plot_args)
    # p = gv.GeoPlotter(**plot_args)

    p.set_scale(zscale=zscale)

    scalar_bar_args = {
        "title": cbar_label,
        "n_labels": 10,
    }

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    p.add_volume(
        grid,
        scalars=data.name,
        cmap=cmap,
        opacity_unit_distance=opacity_unit_distance,
        n_colors=cmap.N,
        opacity=opacity,
        blending=blending,
        clim=(vmin, vmax),
        name=data.name,
        show_scalar_bar=show_scalar_bar,
        shade=True,
        log_scale=log_scale,
        scalar_bar_args=scalar_bar_args,
    )

    show_bounds_args = {
        "padding": padding,
        "axes_ranges": axes_ranges,
        "xtitle": xlabel,
        "ytitle": ylabel,
        "ztitle": zlabel,
        "location": "outer",
        "bold": False,
        # "use_2d": True,  # Force 2D bounds for clearer labels and ticks
        "ticks": "outside",
        "font_size": font_size,
        "n_xlabels": n_xlabels,
        "n_ylabels": n_ylabels,
        "n_zlabels": n_zlabels,
    }
    if not sphere:
        show_bounds_args = {k: v for k, v in show_bounds_args.items() if v is not None}
        p.show_bounds(**show_bounds_args)

    if cam_elev is not None or cam_azim is not None:
        p.camera_position = "yz"

        if cam_elev is not None:
            p.camera.elevation = cam_elev

        if cam_azim is not None:
            p.camera.azimuth = cam_azim
    else:
        p.camera_position = "iso"

    if animation and format == "png":
        p.show(**show_args)
        p.screenshot(**screenshot_args)
        p.close()

    elif animation and format == "html":
        p.render()
        p.export_html(outfile)
        p.close()

    elif outfile is not None and format == "html":
        p.render()
        p.export_html(outfile)
        if ipykernel and not animation:
            return p
    elif outfile is not None and format == "png":
        p.show(**show_args)
        p.screenshot(**screenshot_args)
        if ipykernel and not animation:
            return p
    else:
        p.show(**show_args)


def animate3d_i_frame(
    field: xr.DataArray,
    idx: int,
    session_tmp_dir: Path,
    args: RicedDict,
):
    if args.format == "mp4":
        args.format = "png"

    outfile = session_tmp_dir / f"{idx:06d}.{args.format}"

    mapplot3d(
        field,
        outfile=outfile,
        animation=True,
        **args,
    )


def animate3d(
    data: xr.DataArray,
    dim: str = "time",
    grid_type: Literal["uniform", "structured"] = "uniform",
    sphere: bool = False,
    dim_map: tuple = (("level", "z"), ("lat", "y"), ("lon", "x")),
    z_unit: Literal["hpa", "Pa", "km", "generic"] = "hpa",
    z_unit_scale: int | float = 1,
    indices: list = None,
    outfile: Path = None,
    format: Literal["mp4", "html"] = "mp4",
    window_size: tuple[int, int] = None,
    fps: int = 10,
    parallel: bool = True,
    cmap: str | LinearSegmentedColormap | ListedColormap = "viridis",
    vmin: int | float = None,
    vmax: int | float = None,
    log_scale: bool = False,
    zscale: int | float = 1,
    title: str = None,
    cam_elev: int | float = None,
    cam_azim: int | float = None,
    xlabel: str = None,
    ylabel: str = None,
    zlabel: str = None,
    show_scalar_bar: bool = True,
    n_xlabels: int = None,
    n_ylabels: int = None,
    n_zlabels: int = None,
    font_size: int = None,
    opacity: list | Literal["linear", "sigmoid"] = "linear",
    opacity_unit_distance: int | float = None,
    blending: Literal[
        "additive", "maximum", "minimum", "composite", "average"
    ] = "composite",
    padding: int | float = None,
):
    """
    Animate a 3D scalar field stored in an xarray.DataArray using PyVista.
    """

    args = RicedDict(locals())

    # pop flags that wont be passed to mapplot3d
    args.pop("data")
    args.pop("dim")
    args.pop("indices")
    args.pop("outfile")
    args.pop("parallel")
    args.pop("fps")

    if dim not in data.dims:
        raise ValueError(f"{dim} not found in {data.dims}")

    if indices is None:
        indices = list(range(data.sizes[dim]))

    session_tmp_dir = Path(tempfile.mkdtemp())

    tasks = [(data.isel({dim: i}).load(), i, session_tmp_dir, args) for i in indices]

    if parallel:
        processes = min(n_cpus, len(tasks))

        with Pool(processes=processes) as pool:
            pool.starmap(animate3d_i_frame, tasks)

    else:
        if len(indices) > 100:
            warnings.warn(
                "Sequential rendering of many frames may be slow.",
                UserWarning,
                stacklevel=2,
            )

        for t in tasks:
            animate3d_i_frame(*t)

    if outfile is None and format == "mp4":
        outfile = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.mp4"
    elif outfile is None and format == "html":
        outfile = Path.cwd() / "3d.html"
    else:
        outfile = Path(outfile)

    if format == "html":
        build_pvhtml(session_tmp_dir, outfile)

    else:
        outfile.parent.mkdir(parents=True, exist_ok=True)

        input_pattern = str(session_tmp_dir / "%06d.png")
        ffmpeg_encode(
            input_pattern,
            outfile,
            fps,
            session_tmp_dir,
            True,
            0,
        )

    return outfile
