from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cartopy.util
import numpy as np
import pandas as pd
import xarray as xr
from cf_xarray import *
from hvplot.xarray import *
from typing_extensions import Self

from ..lib_netcdf import append_to_netcdf, to_netcdf
from .core import calc_stats as calc
from .core.preprocess_data import preprocess_era5
from .core.progress import DaskProgressBar, SerialProgressBar
from .core.tools import n_cpus
from .core.xgeo_utils import (
    SetupDask,
    add_local_solar_time,
    mask,
    remap,
    sel_transect,
    to_lon180,
)
from .viz import cmaps
from .viz import plotting as plot

__all__ = [
    "DaskProgressBar",
    "SerialProgressBar",
    "SetupDask",
    "add_local_solar_time",
    "append_to_netcdf",
    "calc",
    "cmaps",
    "mask",
    "n_cpus",
    "plot",
    "preprocess_era5",
    "remap",
    "sel_transect",
    "to_lon180",
    "to_netcdf",
]


_script_dir = Path(__file__).resolve().parent

# Collapse attributes by default in the repr.
xr.set_options(display_expand_attrs=False)

#: Module-level Dask handles, shared by every :class:`SetupDask` instance.
_dask_client = None
_dask_cluster = None


def get_spatial_dims(
    da: xr.DataArray | xr.Dataset,
) -> tuple[str, str]:
    """Return the longitude and latitude coordinate names."""
    ds = da if isinstance(da, xr.Dataset) else da.to_dataset(name=da.name or "data")

    if "latitude" not in ds.cf.coordinates or "longitude" not in ds.cf.coordinates:
        ds = ds.cf.guess_coord_axis()

    lon = ds.cf["longitude"]
    lat = ds.cf["latitude"]

    if lon.name is None or lat.name is None:
        raise ValueError(
            "Could not determine longitude and latitude coordinate names, specify x= and y="
        )

    return lon.name, lat.name


def set_edges_to_nan(
    da: xr.DataArray,
    dims: str | Sequence[str],
    width: int = 1,
) -> xr.DataArray:
    """Set edge cells along selected dimensions to NaN."""
    if width < 0:
        raise ValueError("width must be non-negative")

    if width == 0:
        return da

    selected_dims = (dims,) if isinstance(dims, str) else tuple(dims)

    missing_dims = set(selected_dims).difference(da.dims)
    if missing_dims:
        raise ValueError(f"Dimensions not found in DataArray: {sorted(missing_dims)}")

    interior: dict[str, slice] = {}

    for dim in selected_dims:
        size = da.sizes[dim]

        if 2 * width >= size:
            return da.where(False)

        interior[dim] = slice(width, size - width)

    mask = xr.zeros_like(da, dtype=bool)
    mask[interior] = True

    return da.where(mask)


def add_cyclic_point(obj: xr.DataArray | xr.Dataset, lon: str = "lon"):
    """
    Add a cyclic point to a DataArray along the specified longitude dimension.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        The input DataArray or Dataset to which a cyclic point will be added.
    lon : str, optional
        The name of the longitude dimension. Default is "lon".

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The object with a cyclic point added.
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
    cyclic_data, cyclic_dim = cartopy.util.add_cyclic_point(obj.values, coord=obj[lon])
    coords = {dim: obj.coords[dim] for dim in obj.dims}
    coords[lon] = cyclic_dim

    new_obj = xr.DataArray(cyclic_data, dims=obj.dims, coords=coords, attrs=attrs)

    if dataset:
        new_obj = new_obj.to_dataset(name=obj.name)

    return new_obj


def sel_transect(
    data: xr.Dataset | xr.DataArray,
    x: float | None = None,
    y: float | None = None,
    orientation: float = 0.0,
    width: float = 1.0,
    *,
    xdim: str | None = None,
    ydim: str | None = None,
    geometry: Literal["xy", "latlon"] = "latlon",
    auto_infer_xy: Literal["min", "max"] | None = None,
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
    auto_infer_xy
        Extreme used to infer the transect centre when both ``x`` and ``y``
        are omitted. The default, ``None``, disables automatic inference.
        Set explicitly to ``"min"`` or ``"max"`` to infer the centre from
        a two-dimensional field.
    snap
        Snap the supplied centre coordinates to the nearest grid point.
    drop
        Drop coordinate locations outside the transect.
    """

    if xdim not in data.coords or ydim not in data.coords:
        xdim, ydim = get_spatial_dims(data)

    if width <= 0:
        raise ValueError("`width` must be positive.")

    if geometry not in {"xy", "latlon"}:
        raise ValueError("`geometry` must be either 'xy' or 'latlon'.")

    if auto_infer_xy not in {None, "min", "max"}:
        raise ValueError("`auto_infer_xy` must be None, 'min', or 'max'.")

    xc = data[xdim]
    yc = data[ydim]

    if xc.ndim != 1 or yc.ndim != 1 or xc.dims != (xdim,) or yc.dims != (ydim,):
        raise ValueError(
            "Coordinates must define a rectilinear grid with one-dimensional coordinate variables."
        )

    if xc.size < 2 or yc.size < 2:
        raise ValueError("Each coordinate must contain at least two points.")

    if x is None and y is None:
        if auto_infer_xy is None:
            raise ValueError(
                "Both `x` and `y` are missing. Provide at least one coordinate or set `auto_infer_xy` explicitly to 'min' or 'max'."
            )

        if isinstance(data, xr.DataArray):
            inference_data = data
        elif len(data.data_vars) == 1:
            inference_data = next(iter(data.data_vars.values()))
        else:
            raise ValueError(
                "Automatic x/y inference for a Dataset requires exactly one data variable."
            )

        if inference_data.ndim != 2 or set(inference_data.dims) != {xdim, ydim}:
            raise ValueError(
                "Automatic x/y inference requires data with exactly the x and y dimensions."
            )

        point_dim = "__transect_point"
        flattened = inference_data.stack({point_dim: (ydim, xdim)})

        if not bool(flattened.notnull().any().compute().item()):
            raise ValueError(
                "Cannot infer x and y from data containing only missing values."
            )

        if auto_infer_xy == "max":
            point_index = flattened.argmax(point_dim, skipna=True)
        else:
            point_index = flattened.argmin(point_dim, skipna=True)

        selected = flattened.isel({point_dim: int(point_index.compute().item())})
        x = float(selected[xdim].item())
        y = float(selected[ydim].item())

    latlon = geometry == "latlon"

    def longitude_delta(
        values: xr.DataArray,
        centre: float,
    ) -> xr.DataArray:
        """Signed shortest longitude difference in degrees."""
        return (values - centre + 180.0) % 360.0 - 180.0

    dx_values = xc.diff(xdim)
    if latlon:
        dx_values = longitude_delta(dx_values, 0.0)

    dx = np.abs(dx_values).median(xdim)
    dy = np.abs(yc.diff(ydim)).median(ydim)

    # Resolve the x-coordinate of the transect centre.
    if x is None:
        x0 = None
    elif not snap:
        x0 = float(x)
    elif latlon:
        distance = np.abs(longitude_delta(xc, x))
        index = distance.argmin(xdim)
        x0 = float(xc.isel({xdim: index}))
    else:
        x0 = float(xc.sel({xdim: x}, method="nearest"))

    # Resolve the y-coordinate of the transect centre.
    if y is None:
        y0 = None
    elif snap:
        y0 = float(yc.sel({ydim: y}, method="nearest"))
    else:
        y0 = float(y)

    # Axis-aligned y band.
    if x0 is None:
        mask = np.abs(yc - y0) <= 0.5 * width * dy
        return data.where(mask, drop=drop)

    # Axis-aligned x or longitude band.
    if y0 is None:
        offset = longitude_delta(xc, x0) if latlon else xc - x0
        mask = np.abs(offset) <= 0.5 * width * dx
        return data.where(mask, drop=drop)

    theta = np.deg2rad(orientation % 180.0)

    if not latlon:
        # Unit normal to a line oriented clockwise from positive y.
        normal_x = np.cos(theta)
        normal_y = -np.sin(theta)

        cross_track = (xc - x0) * normal_x + (yc - y0) * normal_y

        cell_width = np.hypot(
            normal_x * dx,
            normal_y * dy,
        )

        mask = np.abs(cross_track) <= 0.5 * width * cell_width
        return data.where(mask, drop=drop)

    # Spherical great-circle transect.
    phi0 = np.deg2rad(y0)
    lam0 = np.deg2rad(x0)

    cross_north_weight = abs(np.sin(theta))
    cross_east_weight = abs(np.cos(theta))

    cell_width = np.hypot(
        cross_north_weight * dy,
        cross_east_weight * dx * np.cos(phi0),
    )

    anchor = np.array(
        [
            np.cos(phi0) * np.cos(lam0),
            np.cos(phi0) * np.sin(lam0),
            np.sin(phi0),
        ]
    )

    north = np.array(
        [
            -np.sin(phi0) * np.cos(lam0),
            -np.sin(phi0) * np.sin(lam0),
            np.cos(phi0),
        ]
    )

    east = np.array(
        [
            -np.sin(lam0),
            np.cos(lam0),
            0.0,
        ]
    )

    direction = np.cos(theta) * north + np.sin(theta) * east
    normal = np.cross(anchor, direction)

    phi = np.deg2rad(yc)
    lam = np.deg2rad(xc)

    point_x = np.cos(phi) * np.cos(lam)
    point_y = np.cos(phi) * np.sin(lam)
    point_z = np.sin(phi)

    dot_normal = (normal[0] * point_x + normal[1] * point_y + normal[2] * point_z).clip(
        min=-1.0, max=1.0
    )

    cross_track = np.rad2deg(np.arcsin(dot_normal))
    mask = np.abs(cross_track) <= 0.5 * width * cell_width

    return data.where(mask, drop=drop)


def to_lon180(
    data: xr.Dataset | xr.DataArray, lon: str = "lon"
) -> xr.Dataset | xr.DataArray:
    """
    Standardize longitude coordinates to [-180, 180).

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        The input dataset or data array containing a longitude coordinate.
    lon : str, default 'lon'
        The name of the longitude coordinate in the dataset.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The dataset or data array with standardized longitude coordinates.
    """
    if lon not in data.coords:
        raise ValueError(f"Dataset must contain {lon!r} coordinate.")

    data = data.copy()
    data[lon] = (data[lon] + 180) % 360 - 180
    data = data.sortby(lon)
    return data


def coord_id(coord: xr.DataArray) -> str:
    """Return a compact description of a regular coordinate."""
    dim = coord.dims[0]
    step = float(coord.diff(dim).mean())
    mean = float(coord.mean(dim))

    return f"{coord.size}:{float(coord.min()):.8g}:{float(coord.max()):.8g}:{mean:.8g}:{step:.8g}"


def grid_id(coords: xr.DataArray | xr.Dataset) -> str:
    """Return a deterministic hexadecimal identifier for a lat-lon grid."""
    signature = f"lat-{coord_id(coords['lat'])}_lon-{coord_id(coords['lon'])}"

    return hashlib.blake2b(
        signature.encode("utf-8"),
        digest_size=8,
    ).hexdigest()


def remap(
    grid_in: xr.Dataset | xr.DataArray,
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
    """
    Remap source data to the destination grid using xESMF.
    """

    import xesmf as xe

    for coord in ("lat", "lon"):
        if coord not in grid_in.dims:
            raise ValueError(f"Input grid must contain {coord!r} dimension.")
        if coord not in grid_out.dims:
            raise ValueError(f"Output grid must contain {coord!r} dimension.")

    in_coords = xr.Dataset(
        coords={
            "lat": grid_in["lat"],
            "lon": grid_in["lon"],
        }
    )

    out_coords = xr.Dataset(
        coords={
            "lat": grid_out["lat"],
            "lon": grid_out["lon"],
        }
    )

    if isinstance(grid_in, xr.DataArray):
        chunked = grid_in if grid_in.chunks is not None else None
    else:
        chunked = next(
            (
                var
                for var in grid_in.data_vars.values()
                if var.chunks is not None and "lat" in var.dims and "lon" in var.dims
            ),
            None,
        )

    if chunked is not None:
        chunks = {
            "lat": chunked.chunksizes["lat"][0],
            "lon": chunked.chunksizes["lon"][0],
        }
        output_chunks = chunks
    else:
        chunks = {
            "lat": grid_in.sizes["lat"],
            "lon": grid_in.sizes["lon"],
        }
        output_chunks = None

    if parallel:
        out_coords["dummy"] = xr.DataArray(
            np.ones((out_coords.lat.size, out_coords.lon.size)),
            dims=("lat", "lon"),
            coords={
                "lat": out_coords.lat,
                "lon": out_coords.lon,
            },
        ).chunk(chunks)

    weight_file = tmp / f"{method}_{grid_id(in_coords)}_{grid_id(out_coords)}"
    reuse = weight_file.exists()

    regridder = xe.Regridder(
        in_coords,
        out_coords,
        method=method,
        parallel=parallel,
        filename=str(weight_file),
        reuse_weights=reuse,
    )

    return regridder(
        grid_in,
        output_chunks=output_chunks,
    )


def mask(
    data: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | xr.Dataset | Path | None = None,
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

    Notes
    -----
    The cache key is based on the mask identity, mask-variable name, target
    coordinate bounds, and target-grid dimensions. The cached object is the
    remapped categorical mask, not the final Boolean mask, so different
    ``valid_value`` values may reuse the same cached remapping.
    """

    if mask is None:
        _default_mask = _script_dir / "data" / "mask" / "era5_0.25_mask"
        print(f"mask is None: Using {_default_mask}")
        mask = _default_mask

    if isinstance(mask, (str, Path)):
        mask = xr.open_dataset(mask)

    if isinstance(mask, xr.Dataset):
        if data_var not in mask:
            raise KeyError(
                f"Mask variable {data_var!r} not found; available: {list(mask.data_vars)}."
            )
        mask = mask[data_var].load()

    if not isinstance(mask, xr.DataArray):
        raise TypeError(f"mask must resolve to an xarray.DataArray, got {type(mask)}.")

    # Sort unconditionally: the regridded mask is cached, so the alignment
    # between data and mask must not depend on whether the cache was hit.
    data = data.sortby(["lat", "lon"])

    lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
    lat_min, lat_max = float(data.lat.min()), float(data.lat.max())

    subset_mask = mask.sortby(["lat", "lon"]).sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )
    remapped_mask = remap(subset_mask, data, method="nearest_s2d", parallel=parallel)

    return data.where(remapped_mask == valid_value, other=np.nan)


def add_local_solar_time(
    data: xr.Dataset | xr.DataArray,
    *,
    lon: str = "lon",
    time: str = "time",
    name: str = "lst",
) -> xr.Dataset | xr.DataArray:
    """
    Add mean local solar time as a coordinate.

    Local solar time is approximated as UTC shifted by the longitude offset,

    .. math::

        \\mathrm{LST} = \\mathrm{UTC} + \\mathrm{round}\\!\\left(
        \\frac{\\lambda}{15^\\circ}\\right)\\,\\mathrm{h}

    where :math:`\\lambda` is longitude in degrees east, wrapped to
    :math:`[-180, 180)`. The offset is rounded to whole hours, so the result is
    a mean solar time on hourly zones rather than apparent solar time: the
    equation of time, which reaches roughly plus or minus 16 minutes over the
    year, is not applied.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object carrying a time coordinate and a longitude coordinate.
    lon : str, default "lon"
        Name of the longitude coordinate, in degrees east.
    time : str, default "time"
        Name of the UTC time coordinate.
    name : str, default "lst"
        Name given to the new coordinate.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input object with the local solar time coordinate attached. The
        coordinate spans both the time and longitude dimensions.
    """
    for coord in (lon, time):
        if coord not in data.coords:
            raise ValueError(f"Input must contain a {coord!r} coordinate.")

    offset = ((data[lon] + 180) % 360 - 180) * (24 / 360)
    offset = offset.round() * pd.Timedelta(hours=1)

    lst = data[time] + offset
    lst.attrs = {}
    lst.attrs["long_name"] = "Local Solar Time"
    lst.attrs["standard_name"] = "local_solar_time"
    lst.attrs["description"] = "Mean local solar time on whole-hour longitude zones"

    return data.assign_coords({name: lst})


class SetupDask:
    """
    Manage a local Dask cluster and client.

    The cluster and client are shared at module level, so repeated
    instantiation reuses one cluster rather than starting several. The object
    can be used directly or as a context manager::

        with SetupDask(workers=4) as dask_setup:
            result = data.mean("time").compute()

    Parameters
    ----------
    workers : int, default 1
        Number of worker processes in the cluster.
    threads_per_worker : int, optional
        Threads per worker. Defaults to the number of available CPUs.
    processes : bool, default False
        Use separate processes for the workers rather than threads. Processes
        avoid the global interpreter lock but pay a serialization cost.
    filter_warnings : bool, default True
        Silence Dask logging below the error level.
    memory_limit : str or int, default "auto"
        Memory limit per worker, for example ``"8GB"``. Set to 0 for no limit.

    Attributes
    ----------
    client : dask.distributed.Client or None
        The active client, or None before :meth:`start` is called.
    cluster : dask.distributed.LocalCluster or None
        The active cluster, or None before :meth:`start` is called.
    """

    def __init__(
        self,
        workers: int = 1,
        threads_per_worker: int = n_cpus,
        processes: bool = False,
        filter_warnings: bool = True,
        memory_limit: str | int = "auto",
    ):
        self.cluster = None
        self.client = None
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self.processes = processes
        self.filter_warnings = filter_warnings
        self.memory_limit = memory_limit

    def start(self):
        """
        Start the cluster and client, or return the existing ones.

        Returns
        -------
        dask.distributed.Client
            The active client. The dashboard is served on the port given by
            ``DASK_DASHBOARD_PORT`` (default 8787).
        """
        global _dask_client, _dask_cluster

        if self.client is not None:
            return self.client

        import dask
        from dask.distributed import Client, LocalCluster

        if _dask_client is not None and _dask_cluster is not None:
            self.client = _dask_client
            self.cluster = _dask_cluster
            return self.client

        port = os.environ.get("DASK_DASHBOARD_PORT", "8787")
        link = f"http://localhost:{port}/status"
        os.environ["DASK_DISTRIBUTED__DASHBOARD__LINK"] = link
        dask.config.refresh()

        self.cluster = LocalCluster(
            n_workers=self.workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
            silence_logs=logging.ERROR if self.filter_warnings else logging.WARNING,
            processes=self.processes,
            dashboard_address=f":{port}",
        )
        self.client = Client(self.cluster)
        _dask_client = self.client
        _dask_cluster = self.cluster

        return self.client

    def close(self) -> None:
        """Close the active client and cluster and release the module handles."""
        global _dask_client, _dask_cluster

        if self.client is not None:
            self.client.close()
        if self.cluster is not None:
            self.cluster.close()

        if self.client is _dask_client:
            _dask_client = None
        if self.cluster is _dask_cluster:
            _dask_cluster = None

        self.client = None
        self.cluster = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
