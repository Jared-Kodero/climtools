from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr
from IPython.display import DisplayHandle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from . import calc_stats as calc
from . import plotting as plotting
from . import xgeo
from .plotting import Geoplot
from .xgeo_utils import to_lon180

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class GeoBase:
    """Operations shared by the DataArray and Dataset accessors."""

    __slots__ = ("_obj",)

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj

    def __repr__(self) -> str:
        kind = type(self._obj).__name__
        dims = ", ".join(f"{name}: {size}" for name, size in self._obj.sizes.items())
        return f"<xgeo accessor on {kind} ({dims})>"

    # -- regridding and masking ------------------------------------------
    def remap(
        self,
        grid_out: xr.Dataset | xr.DataArray,
        method: Literal[
            "bilinear",
            "conservative",
            "conservative_normed",
            "patch",
            "nearest_s2d",
            "nearest_d2s",
        ] = "bilinear",
        parallel: bool = False,
        reuse_weights: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        """Regrid onto the horizontal grid of ``grid_out``.

        See :func:`climtools.xgeo.remap` for the full description.

        Parameters
        ----------
        grid_out : xarray.Dataset or xarray.DataArray
            Object whose 'lat' and 'lon' coordinates define the target grid.
        method : {"bilinear", "conservative", "conservative_normed", "patch", "nearest_s2d", "nearest_d2s"}, default "bilinear"
            ESMF regridding method.
        parallel : bool, default False
            Build the weights in parallel with Dask.
        reuse_weights : bool, default True
            Reuse a cached weight file when one exists.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The object on the target grid.
        """

        return xgeo.remap(
            self._obj,
            grid_out,
            method=method,
            parallel=parallel,
            reuse_weights=reuse_weights,
        )

    def mask_land(
        self,
        mask: xr.DataArray | xr.Dataset | str | Path | None = None,
        keep: Literal["land", "ocean"] = "land",
        parallel: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """Mask outside the land or the ocean.

        See :func:`climtools.xgeo.mask_land` for the full description.

        Parameters
        ----------
        mask : xarray.DataArray, xarray.Dataset, str, pathlib.Path or None, optional
            Land-sea mask. Defaults to the bundled ERA5 0.25 degree mask.
        keep : {"land", "ocean"}, default "land"
            Mask variable to select, and hence the domain retained.
        parallel : bool, default False
            Regrid the mask in parallel with Dask.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            The object with cells outside the retained domain set to NaN.
        """
        return xgeo.mask_land(self._obj, mask=mask, keep=keep, parallel=parallel)

    # -- coordinate helpers ----------------------------------------------
    def add_lst(
        self,
        *,
        lon: str = "lon",
        time: str = "time",
        name: str = "lst",
    ) -> xr.Dataset | xr.DataArray:
        """Add mean local solar time as a coordinate.

        See :func:`climtools.xgeo.add_lst` for the full description, including
        the approximation used.

        Parameters
        ----------
        lon : str, default "lon"
            Name of the longitude coordinate, in degrees east.
        time : str, default "time"
            Name of the UTC time coordinate.
        name : str, default "lst"
            Name given to the new coordinate.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The object with the local solar time coordinate attached.
        """
        return xgeo.add_lst(self._obj, lon=lon, time=time, name=name)

    def to_lon180(self, lon: str = "lon") -> xr.Dataset | xr.DataArray:
        """Wrap the longitude coordinate to the interval [-180, 180).

        Parameters
        ----------
        lon : str, default "lon"
            Name of the longitude coordinate.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The object with wrapped and sorted longitudes.
        """
        return to_lon180(self._obj, lon=lon)

    def cyclic(self, lon: str = "lon") -> xr.Dataset | xr.DataArray:
        """Append a cyclic longitude point, closing the seam at the date line.

        Parameters
        ----------
        lon : str, default "lon"
            Name of the longitude dimension.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The object with one extra longitude point.
        """
        return plotting.make_cyclic(self._obj, lon=lon)

    # -- selection --------------------------------------------------------
    def sel_transect(
        self,
        anchor_point: tuple[float | None, float | None],
        geometry: Literal["latlon", "xy"] = "latlon",
        orientation: float = 0.0,
        width: float = 1.0,
        *,
        x_dim: str = "lon",
        y_dim: str = "lat",
        snap: bool = True,
        drop: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        """Select a finite-width transect band, or a single coordinate band.

        See :func:`climtools.xgeo.sel_transect` for the full description.

        Parameters
        ----------
        anchor_point : tuple of float or None
            Point the transect passes through, ordered ``(lat, lon)`` for
            ``geometry="latlon"`` and ``(x, y)`` for ``geometry="xy"``. Pass
            None for one component to select a coordinate band.
        geometry : {"latlon", "xy"}, default "latlon"
            Coordinate geometry used for the cross-track distance.
        orientation : float, default 0.0
            Transect orientation in degrees, clockwise from north or from +y.
        width : float, default 1.0
            Full transect width, in grid cells.
        x_dim, y_dim : str, default "lon", "lat"
            Horizontal coordinate names.
        snap : bool, default True
            Snap the anchor point to the nearest grid-cell centre.
        drop : bool, default True
            Drop the masked cells.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The object restricted to the selected band.
        """
        return xgeo.sel_transect(
            self._obj,
            anchor_point,
            geometry=geometry,
            orientation=orientation,
            width=width,
            x_dim=x_dim,
            y_dim=y_dim,
            snap=snap,
            drop=drop,
        )


class PlotAccessor:
    """Plotting namespace of the ``.xgeo`` accessor.

    Reached as ``da.xgeo.plot``. The bound array is supplied automatically, so
    the methods take only the drawing options.
    """

    __slots__ = ("_obj",)

    def __init__(self, da: xr.DataArray):
        self._obj = da

    def __repr__(self) -> str:
        return f"<xgeo plotting accessor on DataArray {self._obj.name!r}>"

    def geo(
        self,
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
        kws = locals()
        kws1 = kws.pop("kwargs")
        _ = kws.pop("self")

        return plotting.geo(self._obj, **kws, **kws1)

    def animate(
        self,
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

        kws = locals()
        kws1 = kws.pop("kwargs")
        _ = kws.pop("self")

        return plotting.animate(self._obj, **kws, **kws1)

    def quiver(
        self,
        v: xr.DataArray,
        *,
        x: str = "lon",
        y: str = "lat",
        ax: Any = None,
        subsample: int | tuple[int, int] = (1, 1),
        add_key: bool = True,
        subplots: bool = False,
        key_magnitude: int | float = None,
        key_units: str = None,
        **kwargs,
    ):
        """Draw quiver arrows, using the bound array as the zonal component.

        See :func:`climtools.plotting.quiver` for the full description.

        Parameters
        ----------
        v : xarray.DataArray
            Meridional component. The bound array is the zonal component.
        x, y : str, default "lon", "lat"
            Horizontal coordinate names.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. Defaults to the current axis.
        subsample : int or tuple of int, default (1, 1)
            Grid stride used to thin the arrows.
        add_key : bool, default True
            Draw a reference quiver key.
        subplots : bool, default False
            Treat the axis as part of a grid when positioning the key.
        key_magnitude : int or float, optional
            Reference arrow magnitude.
        key_units : str, optional
            Units shown on the key.
        **kwargs
            Additional arguments forwarded to ``Axes.quiver``.

        Returns
        -------
        tuple
            ``(ax, quiver, quiver_key)``.
        """
        return plotting.quiver(
            self._obj,
            v,
            x=x,
            y=y,
            ax=ax,
            subsample=subsample,
            add_key=add_key,
            subplots=subplots,
            key_magnitude=key_magnitude,
            key_units=key_units,
            **kwargs,
        )

    def significance(
        self,
        *,
        x: str = "lon",
        y: str = "lat",
        ax: Any = None,
        level: float = 0.05,
        color: str = "grey",
        alpha: float = 0.3,
        marker: str = None,
        edgecolors: str = None,
        subsample: int | tuple[int, int] = (1, 1),
        size: float = 0.25,
    ):
        """Mark grid points of the bound p-value field below ``level``.

        See :func:`climtools.plotting.significance` for the full description.

        Parameters
        ----------
        x, y : str, default "lon", "lat"
            Horizontal coordinate names.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. Defaults to the current axis.
        level : float, default 0.05
            Significance threshold.
        color : str, default "grey"
            Marker face color.
        alpha : float, default 0.3
            Marker opacity.
        marker : str, optional
            Marker style.
        edgecolors : str, optional
            Marker edge color.
        subsample : int or tuple of int, default (1, 1)
            Grid stride used to thin the markers.
        size : float, default 0.25
            Marker size.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter artist holding the markers.
        """
        return plotting.significance(
            self._obj,
            x=x,
            y=y,
            ax=ax,
            level=level,
            color=color,
            alpha=alpha,
            marker=marker,
            edgecolors=edgecolors,
            subsample=subsample,
            size=size,
        )


@xr.register_dataarray_accessor("xgeo")
class DataArrayXGeo(GeoBase):
    """``.xgeo`` accessor on a ``DataArray``.

    Adds plotting and the single-field statistics to the shared geospatial
    operations of :class:`_XGeoBase`.
    """

    __slots__ = ()

    @property
    def plot(self) -> PlotAccessor:
        """Plotting namespace, for example ``da.xgeo.plot.geo(...)``."""
        return PlotAccessor(self._obj)

    def trends(
        self,
        dim: str = "time",
        *,
        scale: float = 1,
        dask_scheduler: Literal["threads", "processes"] = "threads",
        polyfit: bool = False,
    ) -> xr.Dataset:
        """Compute a pointwise trend along ``dim``.

        See :func:`climtools.calc_stats.trends` for the full description.

        Parameters
        ----------
        dim : str, default "time"
            Dimension the trend is computed along.
        scale : float, default 1
            Multiplier applied to the slope, to convert its time unit.
        dask_scheduler : {"threads", "processes"}, default "threads"
            Scheduler used to evaluate a chunked input.
        polyfit : bool, default False
            Use ordinary least squares instead of the modified Mann-Kendall
            test.

        Returns
        -------
        xarray.Dataset
            Trend statistics, including the slope and its p-value.
        """
        return calc.trends(
            self._obj,
            dim=dim,
            scale=scale,
            dask_scheduler=dask_scheduler,
            polyfit=polyfit,
        )

    def corr(
        self,
        other: xr.DataArray,
        *,
        dim: str = "time",
        corr_type: Literal["pearson", "spearman", "kendall"] = "pearson",
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        dask_scheduler: Literal["threads", "processes"] = "threads",
    ) -> xr.Dataset:
        """Correlate the bound array with ``other`` along ``dim``.

        See :func:`climtools.calc_stats.corr` for the full description.

        Parameters
        ----------
        other : xarray.DataArray
            Second field, matching the bound array in dimensions and shape.
        dim : str, default "time"
            Dimension the correlation is computed along.
        corr_type : {"pearson", "spearman", "kendall"}, default "pearson"
            Correlation coefficient.
        alternative : {"two-sided", "less", "greater"}, default "two-sided"
            Alternative hypothesis used for the p-value.
        dask_scheduler : {"threads", "processes"}, default "threads"
            Scheduler used to evaluate a chunked input.

        Returns
        -------
        xarray.Dataset
            Dataset holding ``corr`` and ``p_value``.
        """
        return calc.corr(
            self._obj,
            other,
            corr_type=corr_type,
            alternative=alternative,
            dim=dim,
            dask_scheduler=dask_scheduler,
        )

    def significance(self, other: xr.DataArray, dim: str = "time") -> xr.DataArray:
        """Test the difference in mean between the bound array and ``other``.

        See :func:`climtools.calc_stats.significance` for the full description.

        Parameters
        ----------
        other : xarray.DataArray
            Second sample, for example a second period.
        dim : str, default "time"
            Sample dimension.

        Returns
        -------
        xarray.DataArray
            Pointwise p-values of a Welch t-test.
        """
        return calc.significance(self._obj, other, dim=dim)


@xr.register_dataset_accessor("xgeo")
class DatasetXGeo(GeoBase):
    """``.xgeo`` accessor on a ``Dataset``.

    Adds the ERA5 preprocessor and the incremental NetCDF writer to the shared
    geospatial operations of :class:`_XGeoBase`.
    """

    __slots__ = ()

    def preprocess_era5(self) -> xr.Dataset:
        """Standardize ERA5 names, dimensions, units and attributes.

        See :func:`climtools.preprocess_era5.preprocess_era5` for the full
        description.

        Returns
        -------
        xarray.Dataset
            The standardized dataset.
        """
        return xgeo.preprocess_era5(self._obj)

    def write_netcdf(
        self,
        file: str | Path,
        unlimited_dim: str = None,
        *,
        batch_size: int = 1,
        format: str = "NETCDF4",
        shuffle: bool = True,
        zlib: bool = True,
        complevel: int = 4,
        show_progress: bool = True,
        stdout: Any = None,
    ) -> None:
        """Write the Dataset to NetCDF incrementally.

        See :func:`climtools.xgeo.write_netcdf` for the full description.

        Parameters
        ----------
        file : str or pathlib.Path
            Output path. An existing file is replaced.
        unlimited_dim : str, optional
            Dimension made unlimited and appended along.
        batch_size : int, default 1
            Number of slices written per append.
        format : str, default "NETCDF4"
            NetCDF format.
        shuffle : bool, default True
            Apply the HDF5 shuffle filter.
        zlib : bool, default True
            Apply zlib compression.
        complevel : int, default 4
            Compression level, between 1 and 9.
        show_progress : bool, default True
            Display a progress bar while writing.
        stdout : file-like, optional
            Stream the progress bar is written to.

        Returns
        -------
        None
        """
        return xgeo.write_netcdf(
            self._obj,
            Path(file),
            unlimited_dim,
            batch_size=batch_size,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            show_progress=show_progress,
            stdout=stdout,
        )
