from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from ..core import xgeo
from ..lib_mpi.mpi_xarray import MPIAccessor

if TYPE_CHECKING:
    from typing import Any, Literal

    import numpy as np
    from IPython.display import DisplayHandle
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    from ..viz.plotting import GeoPlot


class GeoBase:
    """Operations shared by the DataArray and Dataset accessors."""

    __slots__ = ("_obj",)

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj

    def __repr__(self) -> str:
        kind = type(self._obj).__name__
        dims = ", ".join(f"{name}: {size}" for name, size in self._obj.sizes.items())
        return f"<xgeo accessor on {kind} ({dims})>"

    # -- MPI ---------------------------------------------------------------
    @property
    def mpi(self) -> MPIAccessor:
        """Collective namespace, for example ``ds.xgeo.mpi.sum()``.

        Returns
        -------
        MPIAccessor
            The same accessor reached as ``ds.mpi``. Building it costs
            nothing: MPI is initialized only when a collective is called.
        """
        return MPIAccessor(self._obj)

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

        See :func:`climtools.xgeo.remap` for the full description.

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

        return xgeo.remap(
            self._obj,
            grid_out,
            method=method,
            parallel=parallel,
        )

    def mask(
        self,
        mask: xr.DataArray | xr.Dataset | str | Path | None = None,
        data_var: str = "land",
        valid_value: float = 1,
        parallel: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """
        Mask grid cells that do not match a specified land-sea mask value.

        The mask is remapped to the horizontal grid of ``data`` using
        nearest-neighbour interpolation. This method preserves categorical mask
        values. The remapped mask is cached so that repeated calls using the same
        mask and target-grid specification do not repeat the remapping operation.

        Before masking, ``data`` is sorted by increasing latitude and longitude.
        Consequently, the returned object may have a different coordinate order
        from the input.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Object to mask. It must contain one-dimensional ``lat`` and ``lon``
            coordinates and corresponding dimensions.

        mask : xarray.DataArray, xarray.Dataset, str, pathlib.Path, or None, optional
            Categorical land-sea mask.

            - If a DataArray is supplied, it is used directly.
            - If a Dataset is supplied, the variable named by ``data_var`` is used.
            - If a path is supplied, the Dataset at that path is opened and the
            variable named by ``data_var`` is used.
            - If None, the package's default land-sea mask is used.

            The mask must contain ``lat`` and ``lon`` coordinates. By convention,
            values equal to ``valid_value`` identify cells to retain.

        data_var : str, default "land"
            Name of the mask variable to extract when ``mask`` is a Dataset or a
            path to a Dataset. This argument is ignored when ``mask`` is already
            a DataArray.

        valid_value : float or int, default 1
            Mask value identifying grid cells to retain. Cells whose remapped mask
            value differs from ``valid_value`` are replaced with NaN.

        parallel : bool, default False
            Whether to perform mask remapping in parallel with Dask. This option
            is passed to :func:`remap`.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            A latitude- and longitude-sorted object with cells outside the
            retained mask category replaced by NaN. The return type matches the
            type of ``data``.

        Raises
        ------
        KeyError
            If ``mask`` resolves to a Dataset that does not contain ``data_var``.

        TypeError
            If ``mask`` cannot be resolved to an xarray.DataArray.

        """
        return xgeo.mask(
            self._obj,
            mask=mask,
            data_var=data_var,
            valid_value=valid_value,
            parallel=parallel,
        )

    # -- coordinate helpers ----------------------------------------------
    def add_local_solar_time(
        self,
        *,
        lon: str = "lon",
        time: str = "time",
        name: str = "lst",
    ) -> xr.Dataset | xr.DataArray:
        """Add mean local solar time as a coordinate.

        See :func:`climtools.xgeo.add_local_solar_time` for the full
        description, including the approximation used.

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
        from ..core.xarray_utils import add_cyclic_point

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
        """
        Select cells lying within a transect on a rectilinear xarray grid.

        Parameters
        ----------
        data
            Input Dataset or DataArray.
        x, y
            Transect centre. For spherical geometry, x is longitude and y is
            latitude. Either coordinate may be omitted to select an axis-aligned
            band.
        orientation
            Transect orientation in degrees clockwise from the positive y
            direction. For spherical geometry, this is clockwise from north.
        width
            Transect width in approximate grid-cell units.
        xdim, ydim
            Names of the x and y coordinates.
        geometry
            ``"xy"`` for planar coordinates or ``"latlon"`` for
            longitude-latitude coordinates in degrees.
        snap
            Snap the supplied centre coordinates to the nearest grid point.
        drop
            Drop coordinate locations outside the transect.
        """
        return xgeo.sel_transect(
            self._obj,
            x=x,
            y=y,
            orientation=orientation,
            width=width,
            xdim=xdim,
            ydim=ydim,
            geometry=geometry,
            snap=snap,
            drop=drop,
        )

    def append_to_netcdf(
        self,
        file: Path,
        dim: str = "time",
        mode: Literal["a", "r+"] = "r+",
        format: str = "NETCDF4",
        shuffle: bool | None = None,
        zlib: bool | None = None,
        complevel: int | None = None,
    ) -> None:
        """Append a Dataset along an unlimited dimension.

        Variables containing ``dim`` are extended from the current end of the file.
        Variables without ``dim`` are written only if not already present.
        datetime64, timedelta64, and cftime variables are encoded to CF numeric
        values. When the target variable already exists, the new batch is encoded
        against the units and calendar already stored in the file so the numeric
        axis stays consistent across appends.

        Parameters
        ----------
        file : Path
            NetCDF4 file with read/write access. ``dim`` must be the unlimited
            dimension.
        dim : str, optional
            Unlimited dimension to append along. Default "time".
        mode : {"a", "r+"}, optional
            File access mode passed to netCDF4.Dataset.
        format : str, optional
            NetCDF format passed to netCDF4.Dataset.
        shuffle : bool, optional
            Whether to apply the shuffle filter to the variable. If None, the default compression settings are used.
        zlib : bool, optional
            Whether to apply zlib compression to the variable. If None, the default compression settings are used.
        complevel : int, optional
            Compression level to apply if zlib is True. Must be between 1 and 9. If None, the default compression settings are used.
        """

        return xgeo.append_to_netcdf(
            data=self._obj,
            file=Path(file),
            dim=dim,
            mode=mode,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
        )

    def to_netcdf(
        self,
        file: str | Path,
        unlimited_dim: str | None = None,
        *,
        batch_size: int = 1,
        format: str = "NETCDF4",
        shuffle: bool = True,
        zlib: bool = True,
        complevel: int = 4,
        show_progress: bool = True,
        stdout: Any = None,
    ) -> None:
        """Write the Dataset to NetCDF incrementally using NetCDF4 lib

        See :func:`climtools.xgeo.to_netcdf` for the full description.

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
        return xgeo.to_netcdf(
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
        **kwargs,
    ) -> GeoPlot:
        """
        Draw a scalar field on a Cartopy map and return a composable :class:`GeoPlot`.

        This is the public entry point. It renders a two-dimensional or faceted
        (three-dimensional) DataArray as the base layer and returns a
        :class:`GeoPlot` whose ``add.contour``, ``add.quiver``, ``add.significance``
        and ``add.colorbar`` methods add overlays. The full parameter list is
        declared explicitly so that editors expose every option. This callable
        is exposed as ``climtools.plot.geo`` and as ``da.xgeo.plot.geo``.

        Parameters
        ----------
        x, y : str, optional
            Coordinate names passed to the selected xarray plotting method.
        col, row : str, optional
            Faceting coordinate names. Supplying either produces a faceted plot.
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
        title : str | Dict, default None
            Plot title. if dict provide options accepted by plt.title or figure.suptitle
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
        add_grid_bounds:
            If True, draw an outline along the outer perimeter of the plotted grid domain.
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
        GeoPlot
            Composable map holding the base artists, with chainable overlay methods.

        Notes
        -----
        Input coordinates are plotted with a ``cartopy.crs.PlateCarree()`` transform.
        The display projection is controlled by ``projection``.
        """
        kwargs0 = locals()
        kwargs1 = kwargs0.pop("kwargs")
        _ = kwargs0.pop("self")

        from ..viz import plotting

        return plotting.geo(self._obj, **kwargs0, **kwargs1)

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
        **kwargs,
    ) -> DisplayHandle | None:
        """
        Render a map animation from an xarray DataArray and encode it as MP4.

        Parameters
        ----------
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
        add_grid_bounds:
            If True, draw an outline along the outer perimeter of the plotted grid domain.
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
        frame_id: bool default True
            If True add frame id to title
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

        kwargs0 = locals()
        kwargs1 = kwargs0.pop("kwargs")
        _ = kwargs0.pop("self")

        from ..viz import plotting

        return plotting.animate(self._obj, **kwargs0, **kwargs1)

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
        from ..viz import plotting

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
        marker: str | None = None,
        edgecolors: str | None = None,
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
        from ..viz import plotting

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


class CalcAccessor:
    """Calc namespace of the ``.xgeo`` accessor."""

    __slots__ = ("_obj",)

    def __init__(self, da: xr.DataArray):
        self._obj = da

    def __repr__(self) -> str:
        return f"<xgeo calc accessor on DataArray {self._obj.name!r}>"

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
        from ..core import calc_stats as calc

        return calc.corr(
            self._obj,
            other,
            corr_type=corr_type,
            alternative=alternative,
            dim=dim,
            dask_scheduler=dask_scheduler,
        )

    def pvalues(self, other: xr.DataArray, dim: str = "time") -> xr.DataArray:
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
        from ..core import calc_stats as calc

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
        from ..core import calc_stats as calc

        return calc.trends(
            self._obj,
            dim=dim,
            scale=scale,
            dask_scheduler=dask_scheduler,
            polyfit=polyfit,
        )


@xr.register_dataarray_accessor("xgeo")
class GeoDataArray(GeoBase):
    """``.xgeo`` accessor on a ``DataArray``.

    Adds plotting and the single-field statistics to the shared geospatial
    operations of :class:`GeoBase`.
    """

    __slots__ = ()

    @property
    def plot(self) -> PlotAccessor:
        """Plotting namespace, for example ``da.xgeo.plot.geo(...)``."""
        return PlotAccessor(self._obj)

    @property
    def calc(self) -> CalcAccessor:
        """Calc namespace for example ``da.xgeo.calc.trends(...)``."""
        return CalcAccessor(self._obj)


@xr.register_dataset_accessor("xgeo")
class GeoDataset(GeoBase):
    """``.xgeo`` accessor on a ``Dataset``.

    Adds the ERA5 preprocessor and the incremental NetCDF writer to the shared
    geospatial operations of :class:`GeoBase`.
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
