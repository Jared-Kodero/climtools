"""Expose geographic, plotting, calculation, and preprocessing xarray accessors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from ..core import stats as calc
from ..core import xgeo
from ..core.utils import exclude_key
from ..viz import plotting

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any, Literal

    import numpy as np
    from IPython.display import DisplayHandle
    from matplotlib.collections import PathCollection
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from mpi4py import MPI

    from ..mpi.context import MPIContext
    from ..viz.plotting import GeoPlot


class GeoBase:
    """Operations shared by the DataArray and Dataset accessors."""

    __slots__ = ("_obj",)

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset) -> None:
        """Initialize the geographic accessor."""
        self._obj = xarray_obj

    def __repr__(self) -> str:
        """Return the geographic accessor representation."""
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
    ) -> xr.Dataset | xr.DataArray:
        """Regrid onto the horizontal grid of ``grid_out``.

        Parameters
        ----------
        grid_out : xarray.Dataset or xarray.DataArray
            Object whose 'lat' and 'lon' coordinates define the target grid.
        method : {"bilinear", "conservative", "conservative_normed", "patch", "nearest_s2d", "nearest_d2s"}, default "bilinear"
            ESMF regridding method.
        parallel : bool, default False
            Build the weights in parallel with Dask.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The object on the target grid.

        """

        kwargs = exclude_key("self", dict(locals()))
        return xgeo.remap(self._obj, **kwargs)

    def mask(
        self,
        mask: xr.DataArray | xr.Dataset | str | Path | None = None,
        data_var: str = "land",
        valid_value: float = 1,
        parallel: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """Mask grid cells that do not match a specified land-sea mask value.

        Parameters
        ----------
        mask : xarray.DataArray, xarray.Dataset, str, pathlib.Path, or None, optional
            Categorical land-sea mask.
        data_var : str, default "land"
            Name of the mask variable to extract when ``mask`` is a Dataset or a path to a Dataset.
        valid_value : float or int, default 1
            Mask value identifying grid cells to retain.
        parallel : bool, default False
            Whether to perform mask remapping in parallel with Dask.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            A latitude- and longitude-sorted object with cells outside the retained mask category replaced by NaN.

        Raises
        ------
        KeyError
            If ``mask`` resolves to a Dataset that does not contain ``data_var``.
        TypeError
            If ``mask`` cannot be resolved to an xarray.DataArray.

        """
        kwargs = exclude_key("self", dict(locals()))
        return xgeo.mask(self._obj, **kwargs)

    # -- coordinate helpers ----------------------------------------------
    def add_local_solar_time(
        self,
        *,
        lon: str = "lon",
        time: str = "time",
        name: str = "lst",
    ) -> xr.Dataset | xr.DataArray:
        """Add mean local solar time as a coordinate.

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
        return xgeo.add_local_solar_time(self._obj, lon=lon, time=time, name=name)

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
        return xgeo.to_lon180(self._obj, lon=lon)

    def add_cyclic_point(self, lon: str = "lon") -> xr.Dataset | xr.DataArray:
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
        from .utils import add_cyclic_point

        return add_cyclic_point(self._obj, lon=lon)

    # -- selection --------------------------------------------------------
    def sel_transect(
        self,
        x: float | None = None,
        y: float | None = None,
        orientation: float = 0.0,
        width: float = 1.0,
        *,
        xdim: str | None = None,
        ydim: str | None = None,
        geometry: Literal["xy", "latlon"] = "latlon",
        snap: bool = True,
        drop: bool = True,
    ) -> xr.Dataset | xr.DataArray:
        """Select cells lying within a transect on a rectilinear xarray grid.

        Parameters
        ----------
        x, y : float | None
            Transect centre.
        orientation : float
            Transect orientation in degrees clockwise from the positive y direction.
        width : float
            Transect width in approximate grid-cell units.
        xdim, ydim : str | None
            Names of the x and y coordinates.
        geometry : Literal['xy', 'latlon']
            ``"xy"`` for planar coordinates or ``"latlon"`` for longitude-latitude coordinates in degrees.
        snap : bool
            Snap the supplied centre coordinates to the nearest grid point.
        drop : bool
            Drop coordinate locations outside the transect.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Selected transect subset.

        """
        kwargs = exclude_key("self", dict(locals()))
        return xgeo.sel_transect(self._obj, **kwargs)

    # -- NetCDF output -----------------------------------------------------
    def append(
        self,
        file: str | Path,
        dim: str = "time",
        mode: Literal["a", "r+"] = "r+",
        format: str = "NETCDF4",
        shuffle: bool | None = None,
        zlib: bool | None = None,
        complevel: int | None = None,
    ) -> None:
        """Append the bound Dataset to an existing file along an unlimited dimension.

        Parameters
        ----------
        file : str or pathlib.Path
            NetCDF4 file with read/write access.
        dim : str, default "time"
            Unlimited dimension to append along.
        mode : {"a", "r+"}, default "r+"
            File access mode passed to netCDF4.Dataset.
        format : str, default "NETCDF4"
            NetCDF format passed to netCDF4.Dataset.
        shuffle : bool, optional
            Whether to apply the shuffle filter to newly created variables.
        zlib : bool, optional
            Whether to apply zlib compression to newly created variables.
        complevel : int, optional
            Compression level, between 1 and 9.

        Returns
        -------
        None

        """
        kwargs = exclude_key("self", dict(locals()))
        return xgeo.nc_append(self._obj, **kwargs)

    def to_netcdf(
        self,
        file: str | Path,
        mpi_context: MPIContext | MPI.Intracomm | None = None,
        unlimited_dim: str | Iterable[str] | None = None,
        partition_dim: str | None = None,
        *,
        parallel: bool = False,
        batch_size: int = 24,
        format: str = "NETCDF4",
        shuffle: bool = True,
        zlib: bool = True,
        complevel: int = 4,
        show_progress: bool = True,
        stdout: Any = None,
        chunks: Mapping[str, Iterable[int]] | None = None,
        hints: str | None = None,
        nofill: bool = True,
        allow_serial: bool = False,
    ) -> None:
        """Write the bound Dataset or DataArray to NetCDF.

        Parameters
        ----------
        file : str or pathlib.Path
            Output path.
        mpi_context : MPIContext or mpi4py.MPI.Intracomm, optional
            MPI context or communicator.
        unlimited_dim : str or iterable of str, optional
            Dimension(s) made unlimited in the NetCDF schema.
        partition_dim : str, optional
            Dimension partitioned across MPI ranks in parallel mode.
        parallel : bool, default False
            Use the MPI-parallel NetCDF-4 writer.
        batch_size : int, default 24
            Number of slices along the unlimited dimension written per serial append.
        format : str, default "NETCDF4"
            NetCDF format.
        shuffle : bool, default True
            Apply the HDF5 shuffle filter.
        zlib : bool, default True
            Apply zlib compression.
        complevel : int, default 4
            Compression level, between 1 and 9.
        show_progress : bool, default True
            Display a progress bar while writing serially.
        stdout : file-like, optional
            Stream the serial progress bar is written to.
        chunks : mapping of str to iterable of int, optional
            Explicit chunk shape passed to the parallel writer.
        hints : str, optional
            Semicolon-separated MPI-IO hints in key=value format.
        nofill : bool, default True
            Disable NetCDF pre-filling during parallel initialization.
        allow_serial : bool, default False
            Permit execution when running with a single MPI rank.

        Returns
        -------
        None

        """

        kwargs = exclude_key("self", dict(locals()))
        return xgeo.to_netcdf(self._obj, **kwargs)


@xr.register_dataarray_accessor("xgeo")
class GeoDataArray(GeoBase):
    """DataArray ``.xgeo`` accessor for geospatial, plotting, and calculation operations."""

    __slots__ = ()

    def geoplot(
        self,
        x: str | None = None,
        y: str | None = None,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        figsize: tuple[float, float] | None = None,
        interactive: bool = False,
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
        cmap: str | LinearSegmentedColormap | ListedColormap = None,
        norm: Any = None,
        vmin: float | None = None,
        vmax: float | None = None,
        units: str | None = None,
        levels: int | list | None = None,
        extend: str | None = None,
        robust: bool = False,
        rasterized: bool = False,
        title: str | dict | None = None,
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
        p_value: xr.DataArray = None,
        pvalue_kwargs: dict | None = None,
        u_component: xr.DataArray = None,
        v_component: xr.DataArray = None,
        quiver_kwargs: dict | None = None,
        colorbar_kwargs: dict | None = None,
        clabel: bool = False,
        clabel_fmt: str = "%1.0f",
        clabel_fontsize: float = 8,
        clabel_inline: bool = True,
        clabel_colors: str | None = None,
        clabel_kwargs: dict | None = None,
        cyclic: bool = False,
        **kwargs: Any,
    ) -> GeoPlot:
        """Draw a scalar field on a Cartopy map and return a composable :class:`GeoPlot`.

        Parameters
        ----------
        x, y : str, optional
            Coordinate names passed to the selected xarray plotting method.
        col, row : str, optional
            Faceting coordinate names.
        col_wrap : int, optional
            Number of columns used when wrapping faceted subplots.
        figsize : tuple of float, optional
            Figure size in inches used when creating a new figure.
        interactive : bool, optional
            If True, configures matplotlib for interactive use in Jupyter notebooks.
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
            Units used for colorbar labeling.
        levels : int or sequence of float, optional
            Contour levels for contour-based methods.
        extend : {"neither", "both", "min", "max"}, optional
            Colorbar extension behavior.
        robust : bool, default False
            Whether to request percentile-based color scaling.
        rasterized : bool, default False
            Whether dense scalar artists should be rasterized.
        title : str | Dict, default None
            Plot title.
        orientation : {"vertical", "horizontal"}, optional
            Colorbar orientation.
        add_colorbar : bool, default True
            Whether to add a colorbar for the base field.
        drawedges : bool, default False
            Whether to draw edges between colorbar intervals.
        cbar_label : str, optional
            Explicit colorbar label.
        global_extent : bool, default False
            If True, set the map extent to the full globe.
        set_extent : tuple of float, optional
            Geographic extent as ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
        gridlines : bool, default False
            Whether to draw labeled longitude and latitude gridlines.
        add_grid_bounds : bool
            If True, draw an outline along the outer perimeter of the plotted grid domain.
        coastlines, borders, states : bool, default True
            Switches controlling common Cartopy geographic feature overlays.
        ocean, land : bool, default True
            Switches controlling ocean and land background features.
        lakes, rivers : bool, default False
            Switches controlling optional Cartopy inland water feature overlays.
        p_value : xarray.DataArray, optional
            Pointwise p-value field.
        pvalue_kwargs : dict, optional
            Keyword arguments forwarded to :func:`significance`.
        u_component, v_component : xarray.DataArray, optional
            Zonal and meridional vector components for a base quiver overlay.
        quiver_kwargs : dict, optional
            Keyword arguments forwarded to :func:`quiver`.
        colorbar_kwargs : dict, optional
            Keyword arguments forwarded to :func:`colorbar`.
        clabel : bool, default False
            Label line contours.
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
            If True, append a cyclic longitude point before plotting.
        **kwargs : Any
            Additional keyword arguments forwarded to the selected xarray plotting method after signature filtering.

        Returns
        -------
        GeoPlot
            Composable map holding the base artists, with chainable overlay methods.

        """

        opts = exclude_key("self", dict(locals()))
        kwargs = opts.pop("kwargs")

        return plotting.geo(self._obj, **opts, **kwargs)

    def animate(
        self,
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
        cmap: str | LinearSegmentedColormap | ListedColormap = None,
        norm: Any = None,
        vmin: float | None = None,
        vmax: float | None = None,
        units: str | None = None,
        levels: int | list | None = None,
        extend: str | None = None,
        robust: bool = False,
        rasterized: bool = False,
        title: str | None = None,
        orientation: Literal["vertical", "horizontal"] = "vertical",
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
        u_component: xr.DataArray = None,
        v_component: xr.DataArray = None,
        colorbar_kwargs: dict | None = None,
        quiver_kwargs: dict | None = None,
        clabel: bool = False,
        clabel_fmt: str = "%1.0f",
        clabel_fontsize: float = 8,
        clabel_inline: bool = True,
        clabel_colors: str | None = None,
        clabel_kwargs: dict | None = None,
        cyclic: bool = False,
        indices: tuple | list | np.ndarray = None,
        outfile: Path | str | None = None,
        quality: Literal["low", "medium", "high"] = "medium",
        fps: int = 1,
        parallel: bool = True,
        frame_id: bool = True,
        **kwargs: Any,
    ) -> DisplayHandle | None:
        """Render a map animation from an xarray DataArray and encode it as MP4.

        Parameters
        ----------
        dim : str, default "time"
            Dimension used for animation frames.
        x, y : str, optional
            Coordinate names passed to the selected xarray plotting method when supported.
        col, row : str, optional
            Faceting coordinate names passed to the selected xarray plotting method when supported.
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
            Lower and upper scalar color limits.
        units : str, optional
            Units used for colorbar labeling.
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
            Explicit colorbar label.
        global_extent : bool, default False
            If True, set each map extent to the full globe.
        set_extent : tuple of float, optional
            Geographic extent as ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.
        gridlines : bool, default False
            Whether to draw labeled longitude and latitude gridlines.
        add_grid_bounds : bool
            If True, draw an outline along the outer perimeter of the plotted grid domain.
        coastlines, borders, states : bool, default True
            Switches controlling common Cartopy geographic feature overlays.
        ocean, land : bool, default True
            Switches controlling ocean and land background features.
        lakes, rivers : bool, default False
            Switches controlling optional Cartopy inland water feature overlays.
        u_component, v_component : xarray.DataArray, optional
            Zonal and meridional vector components for quiver overlays.
        colorbar_kwargs : dict, optional
            Keyword arguments forwarded to :func:`colorbar`.
        quiver_kwargs : dict, optional
            Keyword arguments forwarded to :func:`quiver`.
        clabel : bool, default False
            Label line contours.
        clabel_fmt : str, default "%1.0f"
            Contour-label format.
        clabel_fontsize : float, default 8
            Contour-label font size.
        clabel_inline : bool, default True
            Draw contour labels inline.
        clabel_colors : str, optional
            Contour-label color.
        clabel_kwargs : dict, optional
        cyclic : bool, default False
            If True, append a cyclic longitude point before plotting each frame.
        indices : tuple of int, list of int, or numpy.ndarray, optional
            Positional indices along ``dim`` to render.
        outfile : str or pathlib.Path, optional
            Output path for the MP4 animation.
        quality : {"low", "medium", "high"}, default "medium"
            Frame-resolution preset used during PNG rendering.
        fps : int, default 1
            Frames per second passed to ffmpeg.
        parallel : bool, default True
            Whether to render frames with multiprocessing.
        frame_id : bool default True
            If True add frame id to title
        **kwargs : Any
            Additional keyword arguments forwarded to the selected xarray plotting method after signature filtering.

        Returns
        -------
        IPython.display.DisplayHandle or None
            Inside a Jupyter kernel, the encoded MP4 is embedded and its display handle is returned.

        """

        opts = exclude_key("self", dict(locals()))
        kwargs = opts.pop("kwargs")

        return plotting.animate(self._obj, **opts, **kwargs)

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
        key_magnitude: float | None = None,
        key_units: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        """Draw quiver arrows, using the bound array as the zonal component.

        Parameters
        ----------
        v : xarray.DataArray
            Meridional component.
        x, y : str, default "lon", "lat"
            Horizontal coordinate names.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on.
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
        **kwargs : Any
            Additional arguments forwarded to ``Axes.quiver``.

        Returns
        -------
        tuple
            ``(ax, quiver, quiver_key)``.

        """
        from ..viz import plotting

        opts = exclude_key("self", dict(locals()))
        kwargs = opts.pop("kwargs")

        return plotting.quiver(self._obj, **opts, **kwargs)

    def significance(
        self,
        *,
        x: str = "lon",
        y: str = "lat",
        ax: Any = None,
        level: float = 0.05,
        color: str = "grey",
        alpha: float = 0.3,
        marker: str | None = None,
        edgecolors: str | None = None,
        subsample: int | tuple[int, int] = (1, 1),
        size: float = 0.25,
    ) -> PathCollection:
        """Mark grid points of the bound p-value field below ``level``.

        Parameters
        ----------
        x, y : str, default "lon", "lat"
            Horizontal coordinate names.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on.
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

        kwargs = exclude_key("self", dict(locals()))

        return plotting.plot_significance(self._obj, **kwargs)

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

        kwargs = exclude_key("self", dict(locals()))
        kwargs["y"] = kwargs.pop("other")
        return calc.corr(self._obj, kwargs)

    def pvalues(self, other: xr.DataArray, dim: str = "time") -> xr.DataArray:
        """Test the difference in mean between the bound array and ``other``.

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

        return calc.pvalues(self._obj, other, dim=dim)

    def trends(
        self,
        dim: str = "time",
        *,
        scale: float = 1,
        dask_scheduler: Literal["threads", "processes"] = "threads",
        polyfit: bool = False,
    ) -> xr.Dataset:
        """Compute a pointwise trend along ``dim``.

        Parameters
        ----------
        dim : str, default "time"
            Dimension the trend is computed along.
        scale : float, default 1
            Multiplier applied to the slope, to convert its time unit.
        dask_scheduler : {"threads", "processes"}, default "threads"
            Scheduler used to evaluate a chunked input.
        polyfit : bool, default False
            Use ordinary least squares instead of the modified Mann-Kendall test.

        Returns
        -------
        xarray.Dataset
            Trend statistics, including the slope and its p-value.

        """

        kwargs = exclude_key("self", dict(locals()))
        return calc.trends(self._obj, **kwargs)


class PreprocessAccessor:
    """Dataset-specific preprocessing namespace."""

    __slots__ = ("_obj",)

    def __init__(self, ds: xr.Dataset) -> None:
        """Initialize the preprocessing accessor."""
        self._obj = ds

    def era5(self) -> xr.Dataset:
        """Preprocess an ERA5 dataset to standardize variable names, dimensions, and attributes.

        Returns
        -------
        xr.Dataset
            Preprocessed ERA5 dataset.

        """
        return xgeo.preprocess.era5(self._obj)

    def era5_land(self) -> xr.Dataset:
        """Preprocess an ERA5-Land dataset.

        Returns
        -------
        xr.Dataset
            Preprocessed ERA5-Land dataset.

        """
        return xgeo.preprocess.era5_land(self._obj)

    def imerg(self) -> xr.Dataset:
        """Preprocess a GPM IMERG dataset.

        Returns
        -------
        xr.Dataset
            Preprocessed IMERG dataset.

        """
        return xgeo.preprocess.imerg(self._obj)

    def cmorph(self) -> xr.Dataset:
        """Preprocess a CMORPH dataset.

        Returns
        -------
        xr.Dataset
            Preprocessed CMORPH dataset.

        """
        return xgeo.preprocess.cmorph(self._obj)

    def gpcp(self) -> xr.Dataset:
        """Preprocess a GPCP dataset.

        Returns
        -------
        xr.Dataset
            Preprocessed GPCP dataset.

        """
        return xgeo.preprocess.gpcp(self._obj)


@xr.register_dataset_accessor("xgeo")
class GeoDataset(GeoBase):
    """``.xgeo`` accessor on a ``Dataset``.

    Adds preprocessing and Dataset-specific operations to the shared
    geospatial operations of :class:`GeoBase`.
    """

    __slots__ = ()

    @property
    def preprocess(self) -> PreprocessAccessor:
        """Return the preprocessing namespace.

        Returns
        -------
        PreprocessAccessor
            Preprocessing accessor.

        """
        return PreprocessAccessor(self._obj)


def fix_xarray(*, force: bool = False) -> tuple[Path, ...]:
    """Patch xarray source so IDEs resolve registered accessors for completion.

    Parameters
    ----------
    force : bool
        Whether to rebuild an existing source patch.

    Returns
    -------
    tuple[Path, ...]
        Paths modified by the xarray source patch.

    """

    from importlib.util import find_spec

    # Fast path: do not import xarray or integrations if already patched.
    xarray_spec = find_spec("xarray")
    if xarray_spec is None or xarray_spec.origin is None:
        raise RuntimeError("Cannot locate the xarray package.")

    marker = Path(xarray_spec.origin).resolve().parent / ".xgeo_patch"

    if not force and marker.exists():
        return ()

    # Everything below is only needed when creating/rebuilding the patch.
    import ast
    import importlib
    import inspect
    import json
    import os
    import sys

    import xarray as xr

    if not __package__:
        raise RuntimeError("The accessor module must be imported as part of a package.")

    begin = "XGEO_IDE_TYPING BEGIN"
    end = "XGEO_IDE_TYPING END"

    bridge_path = Path(__file__).resolve().parent / "xgeo_patch.py"
    type_module = f"{__package__}.xgeo_patch"

    bridge = (
        "from __future__ import annotations\n"
        "\n"
        "from .accessors import GeoDataArray as GeoDataArray\n"
        "from .accessors import GeoDataset as GeoDataset\n"
        "\n"
        '__all__ = ["GeoDataArray", "GeoDataset"]\n'
    )

    # Verify generated bridge before writing it.
    compile(bridge, str(bridge_path), "exec")

    bridge_changed = (
        not bridge_path.exists() or bridge_path.read_text(encoding="utf-8") != bridge
    )

    if bridge_changed:
        bridge_path.write_text(bridge, encoding="utf-8")

    # (xarray class, class name, local accessor definitions)
    targets: tuple[tuple[type, str, tuple[tuple[str, str], ...]], ...] = (
        (xr.DataArray, "DataArray", (("xgeo", "GeoDataArray"),)),
        (xr.Dataset, "Dataset", (("xgeo", "GeoDataset"),)),
    )

    optional: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("metpy.xarray", ("metpy",)),
        ("cf_xarray", ("cf",)),
        ("pint_xarray", ("pint",)),
        ("rioxarray", ("rio",)),
    )

    integration_names = tuple(module.partition(".")[0] for module, _ in optional)

    sources: dict[str, Path] = {}

    for cls, class_name, _ in targets:
        source_file = inspect.getsourcefile(cls)

        if source_file is None:
            raise RuntimeError(f"Cannot locate xarray.{class_name} source.")

        sources[class_name] = Path(source_file).resolve()

    # force=True restores pristine source before rebuilding the patch.
    if force:
        for path in set(sources.values()):
            backup = path.with_suffix(path.suffix + ".xgeo.bak")

            if backup.exists():
                path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
                backup.unlink()

    def stat_of(path: str | Path | None) -> list[int] | None:
        """Return an immutable file-stat signature."""
        if path is None:
            return None

        try:
            stat = os.stat(path)
        except OSError:
            return None

        return [stat.st_size, stat.st_mtime_ns]

    def signature() -> dict:
        """Return a stable signature for a source file."""
        integrations: dict[str, dict | None] = {}

        for name in integration_names:
            try:
                spec = find_spec(name)
            except (ImportError, ValueError):
                spec = None

            if spec is None:
                integrations[name] = None
                continue

            integrations[name] = {"origin": spec.origin, "stat": stat_of(spec.origin)}

        return {
            "schema": 4,
            "python": (f"{sys.version_info.major}.{sys.version_info.minor}"),
            "files": {label: stat_of(path) for label, path in sources.items()},
            "integrations": integrations,
        }

    def discover() -> dict[type, list[tuple[str, str, str]]]:
        """Discover accessor registrations in a source file."""
        found: dict[type, list[tuple[str, str, str]]] = {
            cls: [] for cls, _, _ in targets
        }

        for module_name, names in optional:
            top = module_name.partition(".")[0]

            try:
                importlib.import_module(module_name)

            except ModuleNotFoundError as exc:
                missing = exc.name or ""

                if missing in {top, module_name} or module_name.startswith(
                    missing + "."
                ):
                    continue

                raise RuntimeError(
                    f"{module_name!r} is installed but dependency "
                    + f"{missing!r} is missing."
                ) from exc

            for name in names:
                registered = False

                for cls, class_name, _ in targets:
                    accessor = getattr(cls, name, None)

                    if accessor is None:
                        continue

                    registered = True

                    if not inspect.isclass(accessor):
                        raise RuntimeError(
                            f"{class_name}.{name} is not an accessor class."
                        )

                    if accessor.__qualname__ != accessor.__name__:
                        raise RuntimeError(
                            f"{class_name}.{name} is a nested class "
                            + "and cannot be imported."
                        )

                    found[cls].append((name, accessor.__module__, accessor.__name__))

                if not registered:
                    raise RuntimeError(f"{module_name!r} did not register {name!r}.")

        return found

    def strip(source: str) -> str:
        """Remove previously injected source regions."""
        output: list[str] = []
        skipping = False

        for line in source.splitlines(keepends=True):
            if begin in line:
                skipping = True
                continue

            if end in line:
                skipping = False
                continue

            if not skipping:
                output.append(line)

        return "".join(output)

    def region(tag: str, indent: str, body: list[str]) -> str:
        """Wrap generated source with patch markers."""
        return f"{indent}# {begin} {tag}\n" + "".join(body) + f"{indent}# {end} {tag}\n"

    def build(class_name: str, stubs: list[tuple[str, str, str]]) -> tuple[str, str]:
        """Build the accessor bridge source."""
        aliases = {attr: f"_xgeo_{class_name}_{attr}" for attr, _, _ in stubs}

        imports = [
            "from typing import TYPE_CHECKING\n",
            "if TYPE_CHECKING:\n",
        ]

        for attr, module, name in stubs:
            imports.append(f"    from {module} import {name} as {aliases[attr]}\n")

        properties = [
            "    if TYPE_CHECKING:\n",
        ]

        for attr, _, _ in stubs:
            properties.append("        @property\n")
            properties.append(f"        def {attr}(self) -> {aliases[attr]}: ...\n")

        return (
            region(f"imports {class_name}", "", imports),
            region(f"properties {class_name}", "    ", properties),
        )

    discovered = discover()
    changed: list[Path] = []

    if bridge_changed:
        changed.append(bridge_path)

    for cls, class_name, own_accessors in targets:
        path = sources[class_name]

        backup = path.with_suffix(path.suffix + ".xgeo.bak")

        source_path = backup if backup.exists() else path

        raw = source_path.read_text(encoding="utf-8")

        pristine = strip(raw)

        if not backup.exists():
            backup.write_text(pristine, encoding="utf-8")

        stubs: list[tuple[str, str, str]] = [
            *((attr, type_module, type_name) for attr, type_name in own_accessors),
            *discovered[cls],
        ]

        for attr, _, _ in stubs:
            if not attr.isidentifier():
                raise RuntimeError(f"Accessor name {attr!r} is not a valid identifier.")

        import_region, property_region = build(class_name, stubs)

        tree = ast.parse(pristine, filename=str(path))

        node = next(
            (
                item
                for item in tree.body
                if (isinstance(item, ast.ClassDef) and item.name == class_name)
            ),
            None,
        )

        if node is None:
            raise RuntimeError(f"Cannot find xarray.{class_name} class definition.")

        head = node.body[0]

        is_docstring = (
            isinstance(head, ast.Expr)
            and isinstance(head.value, ast.Constant)
            and isinstance(head.value.value, str)
        )

        property_at = head.end_lineno if is_docstring else head.lineno - 1

        import_at = node.lineno - 1

        lines = pristine.splitlines(keepends=True)

        # Insert properties first because adding the module-level import
        # region afterward shifts the entire class downward.
        lines.insert(property_at, property_region)
        lines.insert(import_at, import_region)

        patched = "".join(lines)

        # Never write invalid Python into the installed xarray source.
        compile(patched, str(path), "exec")

        if patched != path.read_text(encoding="utf-8"):
            path.write_text(patched, encoding="utf-8")
            changed.append(path)

    # The marker is created only after every source modification succeeds.
    tmp = marker.with_suffix(marker.suffix + ".tmp")

    tmp.write_text(json.dumps(signature(), sort_keys=True), encoding="utf-8")

    tmp.replace(marker)

    return tuple(changed)
