"""Provide geographic xarray utilities and local Dask cluster management."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cartopy.util
import numpy as np
import pandas as pd
import xarray as xr
from cf_xarray import *
from scipy.interpolate import griddata

from ..core.utils import N_CPUS, TMP

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType
    from typing import Literal, Self

    from dask.distributed import Client

xr.set_options(display_expand_attrs=False)

script_dir = Path(__file__).resolve().parent


#: Module-level Dask handles, shared by every :class:`SetupDask` instance.
_dask_client = None
_dask_cluster = None


def get_spatial_dims(da: xr.DataArray | xr.Dataset) -> tuple[str, str]:
    """Return the longitude and latitude coordinate names."""
    ds = da if isinstance(da, xr.Dataset) else da.to_dataset(name=da.name or "data")

    if "latitude" not in ds.cf.coordinates or "longitude" not in ds.cf.coordinates:
        ds = ds.cf.guess_coord_axis()

    lon = ds.cf["longitude"]
    lat = ds.cf["latitude"]

    if lon.name is None or lat.name is None:
        raise ValueError(
            "Could not infer longitude/latitude coordinates; pass x= and y=."
        )

    return lon.name, lat.name


def set_edges_to_nan(
    da: xr.DataArray, dims: str | Sequence[str], width: int = 1
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


def add_cyclic_point(
    obj: xr.DataArray | xr.Dataset, lon: str = "lon"
) -> xr.DataArray | xr.Dataset:
    """Add a cyclic point to a DataArray along the specified longitude dimension."""

    dataset = False

    if isinstance(obj, xr.Dataset) and len(obj.data_vars) > 1:
        raise ValueError("Expected a DataArray or single-variable Dataset.")

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
    """Select cells lying within a transect on a rectilinear xarray grid.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Input Dataset or DataArray.
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
    auto_infer_xy : Literal['min', 'max'] | None
        Extreme used to infer the transect centre when both ``x`` and ``y`` are omitted.
    snap : bool
        Snap the supplied centre coordinates to the nearest grid point.
    drop : bool
        Drop coordinate locations outside the transect.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Selected transect subset.

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
        raise ValueError("Coordinates must be 1-D on a rectilinear grid.")

    if xc.size < 2 or yc.size < 2:
        raise ValueError("Each coordinate must contain at least two points.")

    if x is None and y is None:
        if auto_infer_xy is None:
            raise ValueError("Provide x or y, or set auto_infer_xy to 'min' or 'max'.")

        if isinstance(data, xr.DataArray):
            inference_data = data
        elif len(data.data_vars) == 1:
            inference_data = next(iter(data.data_vars.values()))
        else:
            raise ValueError(
                "Automatic x/y inference requires a single-variable Dataset."
            )

        if inference_data.ndim != 2 or set(inference_data.dims) != {xdim, ydim}:
            raise ValueError(
                "Automatic x/y inference requires exactly the x and y dimensions."
            )

        point_dim = "__transect_point"
        flattened = inference_data.stack({point_dim: (ydim, xdim)})

        if not bool(flattened.notnull().any().compute().item()):
            raise ValueError("Cannot infer x/y from all-missing data.")

        if auto_infer_xy == "max":
            point_index = flattened.argmax(point_dim, skipna=True)
        else:
            point_index = flattened.argmin(point_dim, skipna=True)

        selected = flattened.isel({point_dim: int(point_index.compute().item())})
        x = float(selected[xdim].item())
        y = float(selected[ydim].item())

    latlon = geometry == "latlon"

    def longitude_delta(values: xr.DataArray, centre: float) -> xr.DataArray:
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

        cell_width = np.hypot(normal_x * dx, normal_y * dy)

        mask = np.abs(cross_track) <= 0.5 * width * cell_width
        return data.where(mask, drop=drop)

    # Spherical great-circle transect.
    phi0 = np.deg2rad(y0)
    lam0 = np.deg2rad(x0)

    cross_north_weight = abs(np.sin(theta))
    cross_east_weight = abs(np.cos(theta))

    cell_width = np.hypot(
        cross_north_weight * dy, cross_east_weight * dx * np.cos(phi0)
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
    """Standardize longitude coordinates to [-180, 180)."""
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

    return hashlib.blake2b(signature.encode("utf-8"), digest_size=8).hexdigest()


from typing import Literal

import xarray as xr


def regrid(
    grid_in: xr.Dataset | xr.DataArray,
    grid_out: xr.Dataset | xr.DataArray | None = None,
    grid_out_resolution: float | None = None,
    method: Literal[
        "bilinear",
        "conservative",
        "conservative_normed",
        "patch",
        "nearest_s2d",
        "nearest_d2s",
    ] = "bilinear",
    unmapped_to_nan: bool = True,
    parallel: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Regrid source data to the destination grid using xESMF."""

    import xesmf as xe

    if grid_out is not None and grid_out_resolution is not None:
        raise ValueError(
            "Provide either `grid_out` or `grid_out_resolution`, not both."
        )

    if grid_out is None and grid_out_resolution is None:
        raise ValueError("You must provide either `grid_out` or `grid_out_resolution`.")

    for coord in ("lat", "lon"):
        if coord not in grid_in.dims:
            raise ValueError(f"Input grid must contain {coord!r} dimension.")

    if grid_out is None:
        lat_min, lat_max = float(grid_in["lat"].min()), float(grid_in["lat"].max())
        lon_min, lon_max = float(grid_in["lon"].min()), float(grid_in["lon"].max())

        lat_coords = np.arange(
            lat_min, lat_max + grid_out_resolution, grid_out_resolution
        )
        lon_coords = np.arange(
            lon_min, lon_max + grid_out_resolution, grid_out_resolution
        )

        grid_out = xr.Dataset(
            coords={
                "lat": lat_coords,
                "lon": lon_coords,
            }
        )

    for coord in ("lat", "lon"):
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

    weight_file = TMP / f"{method}_{grid_id(in_coords)}_{grid_id(out_coords)}"
    reuse = weight_file.exists()

    regridder = xe.Regridder(
        in_coords,
        out_coords,
        method=method,
        parallel=parallel,
        filename=str(weight_file),
        reuse_weights=reuse,
        unmapped_to_nan=unmapped_to_nan,
    )

    return regridder(grid_in, output_chunks=output_chunks)


def mask(
    data: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | xr.Dataset | Path | None = None,
    data_var: str = "land",
    valid_value: float = 1,
    parallel: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Mask grid cells that do not match a specified land-sea mask value.

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

    if mask is None:
        _default_mask = script_dir / "data" / "mask" / "era5_0.25_mask"
        print(f"mask is None: Using {_default_mask}")
        mask = _default_mask

    if isinstance(mask, (str, Path)):
        mask = xr.open_dataset(mask)

    if isinstance(mask, xr.Dataset):
        if data_var not in mask:
            raise KeyError(f"Mask variable {data_var!r} not found.")
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
    remapped_mask = regrid(subset_mask, data, method="nearest_s2d", parallel=parallel)

    return data.where(remapped_mask == valid_value, other=np.nan)


def _fillgaps_2d(
    values: np.ndarray,
    *,
    method: Literal["linear", "cubic", "nearest"],
    max_cells: int,
    max_iter: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()

    ny, nx = values.shape

    y = np.arange(ny, dtype=float)
    x = np.arange(nx, dtype=float)
    x2d, y2d = np.meshgrid(x, y)

    for _ in range(max_iter):
        missing = ~np.isfinite(values)

        if not np.any(missing):
            break

        target = np.zeros_like(missing, dtype=bool)

        for i in range(ny):
            row = missing[i]

            padded = np.pad(row.astype(np.int8), 1, constant_values=0)

            diff = np.diff(padded)

            starts = np.where(diff == 1)[0]
            stops = np.where(diff == -1)[0]

            for start, stop in zip(starts, stops, strict=True):
                gap = stop - start

                bounded = (
                    start > 0
                    and stop < nx
                    and np.isfinite(values[i, start - 1])
                    and np.isfinite(values[i, stop])
                )

                if gap <= max_cells and bounded:
                    target[i, start:stop] = True

        for j in range(nx):
            col = missing[:, j]

            padded = np.pad(
                col.astype(np.int8),
                1,
                constant_values=0,
            )

            diff = np.diff(padded)

            starts = np.where(diff == 1)[0]
            stops = np.where(diff == -1)[0]

            for start, stop in zip(starts, stops, strict=True):
                gap = stop - start

                bounded = (
                    start > 0
                    and stop < ny
                    and np.isfinite(values[start - 1, j])
                    and np.isfinite(values[stop, j])
                )

                if gap <= max_cells and bounded:
                    target[start:stop, j] = True

        if not np.any(target):
            break

        valid = np.isfinite(values)

        points = np.column_stack((x2d[valid], y2d[valid]))
        targets = np.column_stack((x2d[target], y2d[target]))

        interpolated = griddata(
            points=points, values=values[valid], xi=targets, method=method
        )

        old_count = np.isfinite(values).sum()

        values[target] = interpolated

        new_count = np.isfinite(values).sum()

        if new_count == old_count:
            break

    return values


def fillgaps(
    da: xr.DataArray,
    method: Literal["linear", "cubic", "nearest"] = "linear",
    max_cells: int = 5,
    max_iter: int = 5,
    nan_mask: xr.DataArray | None = None,
    y: str = "lat",
    x: str = "lon",
) -> xr.DataArray:
    """
    Fill thin NaN gaps along two dimensions.

    The operation is applied independently to every ``(y, x)`` slice.
    All other dimensions are vectorized with :func:`xarray.apply_ufunc`.

    Parameters
    ----------
    da : xarray.DataArray
        Input array containing finite values and NaNs.
    method : {"linear", "cubic", "nearest"}, default="linear"
        Interpolation method passed to :func:`scipy.interpolate.griddata`.
    max_cells : int, default=5
        Maximum contiguous NaN run length, in grid cells, eligible for
        interpolation.
    max_iter : int, default=5
        Maximum number of interpolation passes.
    nan_mask : xarray.DataArray, optional
        Boolean mask defining cells that must be NaN in the returned
        array. The mask is applied only after interpolation.
    y : str, default="lat"
        Name of the first interpolation dimension.
    x : str, default="lon"
        Name of the second interpolation dimension.

    Returns
    -------
    xarray.DataArray
        DataArray with eligible NaN gaps interpolated.

    Notes
    -----
    Interpolation is performed in array-index space rather than physical
    coordinate space.
    """
    if y not in da.dims:
        raise ValueError(f"y dimension {y!r} is not present in da.")

    if x not in da.dims:
        raise ValueError(f"x dimension {x!r} is not present in da.")

    if y == x:
        raise ValueError("x and y must refer to different dimensions.")

    if method not in {"linear", "cubic", "nearest"}:
        raise ValueError("method must be 'linear', 'cubic', or 'nearest'.")

    if max_cells < 1:
        raise ValueError("max_cells must be >= 1.")

    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")

    core_dims = [y, x]

    if nan_mask is not None:
        da, nan_mask = xr.align(da, nan_mask, join="exact")

    result = xr.apply_ufunc(
        _fillgaps_2d,
        da,
        input_core_dims=[core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
        keep_attrs=True,
        dask_gufunc_kwargs={
            "allow_rechunk": True,
        },
        kwargs={
            "method": method,
            "max_cells": max_cells,
            "max_iter": max_iter,
        },
    )

    result = result.transpose(*da.dims)

    if nan_mask is not None:
        result = result.where(~nan_mask.astype(bool))
    return result


def add_local_solar_time(
    data: xr.Dataset | xr.DataArray,
    *,
    lon: str = "lon",
    time: str = "time",
    name: str = "lst",
) -> xr.Dataset | xr.DataArray:
    """Add mean local solar time as a coordinate."""
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
    """Manage a reusable local Dask cluster.

    Parameters
    ----------
    workers : int, default 1
        Number of workers.
    threads_per_worker : int, optional
        Threads per worker.
    processes : bool, default False
        Use worker processes instead of threads.
    filter_warnings : bool, default True
        Suppress Dask logs below error level.
    memory_limit : str or int, default "auto"
        Per-worker memory limit.

    Attributes
    ----------
    client : dask.distributed.Client or None
        Active client.
    cluster : dask.distributed.LocalCluster or None
        Active local cluster.
    """

    def __init__(
        self,
        workers: int = 1,
        threads_per_worker: int = N_CPUS,
        processes: bool = False,
        filter_warnings: bool = True,
        memory_limit: str | int = "auto",
    ) -> None:
        """Initialize local Dask cluster settings."""
        self.cluster = None
        self.client = None
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self.processes = processes
        self.filter_warnings = filter_warnings
        self.memory_limit = memory_limit

    def start(self) -> Client:
        """Start the cluster and client, or return the existing ones.

        Returns
        -------
        dask.distributed.Client
            The active client.

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
        """Enter the local Dask cluster context."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the local Dask cluster context."""
        self.close()


class XNpyStore:
    """Store NumPy and xarray objects in a memory-mappable directory format.

    Numerical payloads are stored as uncompressed ``.npy`` files. Metadata are
    stored in a versioned JSON manifest. No pickle data are written.

    Parameters
    ----------
    path : str or pathlib.Path
        Store path. The ``.xnpy`` suffix is appended if absent.

    Attributes
    ----------
    path : pathlib.Path
        Path to the store directory.
    meta_path : pathlib.Path
        Path to ``metadata.json``.

    Notes
    -----
    ``object``-dtype arrays are rejected because they require object
    serialization and cannot be safely memory-mapped as numerical payloads.

    The ``.npy`` header is authoritative for array dtype, shape, byte order,
    and memory order. Values duplicated in JSON are used for inspection and
    validation.
    """

    SUFFIX = ".xnpy"
    META_FILE = "metadata.json"
    COORD_DIR = "_coords"
    FORMAT = "xnpy"
    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != self.SUFFIX:
            path = Path(f"{path}{self.SUFFIX}")
        self.path = path
        self.meta_path = self.path / self.META_FILE

    def save(self, obj: Any, *, overwrite: bool = False) -> Path:
        """Save an array, DataArray, or Dataset.

        Parameters
        ----------
        obj : numpy.ndarray or xarray.DataArray or xarray.Dataset
            Object to store.
        overwrite : bool, default False
            Remove an existing store before writing when True.

        Returns
        -------
        pathlib.Path
            Path to the completed store.

        Raises
        ------
        FileExistsError
            If the store exists and ``overwrite`` is False.
        TypeError
            If ``obj`` is unsupported or holds an object-dtype payload.
        """
        if self.path.exists():
            if not overwrite:
                raise FileExistsError(self.path)
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)

        if isinstance(obj, xr.Dataset):
            metadata = self._save_dataset(obj)
        elif isinstance(obj, xr.DataArray):
            metadata = self._save_dataarray(obj)
        elif isinstance(obj, np.ndarray):
            metadata = self._save_ndarray(obj)
        else:
            raise TypeError(
                "XNpyStore supports numpy.ndarray, xarray.DataArray, and "
                "xarray.Dataset only."
            )

        # Written last: its presence is what marks the store complete.
        with self.meta_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"format": self.FORMAT, "version": self.VERSION, **metadata},
                file,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        return self.path

    def metadata(self) -> dict[str, Any]:
        """Read the JSON manifest without opening array payloads.

        Returns
        -------
        dict
            Store metadata.

        Raises
        ------
        ValueError
            If the directory is not an ``xnpy`` store, or its format version
            is not understood.
        """
        with self.meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("format") != self.FORMAT:
            raise ValueError(f"Not an {self.FORMAT!r} store: {self.path}.")
        if metadata.get("version") != self.VERSION:
            raise ValueError(
                f"Unsupported {self.FORMAT} version: {metadata.get('version')!r}."
            )
        return metadata

    def variables(self) -> tuple[str, ...]:
        """Return data-variable names without opening array payloads.

        Returns
        -------
        tuple of str
            Stored data-variable names, empty for a bare array store.
        """
        metadata = self.metadata()
        if metadata["kind"] == "dataset":
            return tuple(metadata["variables"])
        if metadata["kind"] == "dataarray":
            return (metadata["variable_name"],)
        return ()

    def load(
        self, variable: str | None = None, *, mmap_mode: str | None = "r"
    ) -> np.ndarray | xr.DataArray | xr.Dataset:
        """Load a stored object or one Dataset variable.

        Metadata are read first, so naming ``variable`` opens only that
        variable's payload and the coordinates associated with it.

        Parameters
        ----------
        variable : str, optional
            Dataset variable to reconstruct. None reconstructs the whole
            stored object.
        mmap_mode : {"r", "r+", "w+", "c"} or None, default "r"
            Memory-map mode passed to :func:`numpy.load`. None reads payloads
            into ordinary in-memory arrays.

        Returns
        -------
        numpy.ndarray or xarray.DataArray or xarray.Dataset
            Reconstructed object.

        Raises
        ------
        KeyError
            If ``variable`` is not present.
        ValueError
            If ``variable`` is invalid for the stored object type, or the
            store kind is unknown.
        """
        metadata = self.metadata()
        kind = metadata["kind"]

        if kind == "dataset":
            if variable is None:
                return self._load_dataset(metadata, mmap_mode)
            return self._load_dataset_variable(metadata, variable, mmap_mode)

        if kind == "dataarray":
            if variable is not None and variable != metadata["variable_name"]:
                raise KeyError(variable)
            return self._load_dataarray(metadata, mmap_mode)

        if kind == "ndarray":
            if variable is not None:
                raise ValueError("variable= is valid only for xarray stores.")
            return self._load_array(metadata["array"], mmap_mode)

        raise ValueError(f"Unknown store kind: {kind!r}")

    def _save_dataset(self, dataset: xr.Dataset) -> dict[str, Any]:
        """Save every data variable and coordinate of a Dataset."""
        variables = {}
        for name, variable in dataset.data_vars.items():
            info = self._save_xarray_variable(variable, self._make_dir(self.path, name))
            # Recorded so one variable can later be reopened with just the
            # coordinates that belong to it.
            info["coords"] = list(dataset[name].coords)
            variables[name] = info

        return {
            "kind": "dataset",
            "attrs": self._json_value(dict(dataset.attrs)),
            "variables": variables,
            "coords": self._save_coords(dataset),
        }

    def _save_dataarray(self, array: xr.DataArray) -> dict[str, Any]:
        """Save a DataArray and its coordinates."""
        name = str(array.name) if array.name is not None else "data"
        variable = self._save_xarray_variable(array, self._make_dir(self.path, name))
        variable["coords"] = list(array.coords)
        return {
            "kind": "dataarray",
            "name": array.name,
            "variable_name": name,
            "variable": variable,
            "coords": self._save_coords(array),
        }

    def _save_coords(self, obj: xr.Dataset | xr.DataArray) -> dict[str, Any]:
        """Save every coordinate under the shared coordinate directory."""
        root = self.path / self.COORD_DIR
        root.mkdir()
        return {
            name: self._save_xarray_variable(coord, self._make_dir(root, name))
            for name, coord in obj.coords.items()
        }

    def _save_ndarray(self, array: np.ndarray) -> dict[str, Any]:
        """Save a bare NumPy array."""
        directory = self.path / "array"
        directory.mkdir()
        return {"kind": "ndarray", "array": self._save_array(directory, array)}

    def _save_xarray_variable(
        self, variable: xr.Variable | xr.DataArray, directory: Path
    ) -> dict[str, Any]:
        """Save one variable's payload with the metadata needed to rebuild it."""
        return {
            "dims": list(variable.dims),
            "attrs": self._json_value(dict(variable.attrs)),
            "array": self._save_array(directory, np.asarray(variable.data)),
        }

    def _save_array(self, directory: Path, array: np.ndarray) -> dict[str, Any]:
        """Write one payload as ``.npy``."""
        if array.dtype.hasobject:
            raise TypeError(
                "object-dtype arrays are not supported because XNpyStore does "
                "not use pickle."
            )
        path = directory / "data.npy"
        np.save(path, array, allow_pickle=False)
        return {
            "file": str(path.relative_to(self.path)),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

    def _load_dataset(
        self, metadata: dict[str, Any], mmap_mode: str | None
    ) -> xr.Dataset:
        """Reconstruct a complete Dataset."""
        return xr.Dataset(
            data_vars={
                name: self._as_variable(info, mmap_mode)
                for name, info in metadata["variables"].items()
            },
            coords=self._load_coords(metadata["coords"], metadata["coords"], mmap_mode),
            attrs=metadata["attrs"],
        )

    def _load_dataset_variable(
        self, metadata: dict[str, Any], name: str, mmap_mode: str | None
    ) -> xr.DataArray:
        """Reconstruct one Dataset variable and the coordinates it uses."""
        variables = metadata["variables"]
        if name not in variables:
            raise KeyError(
                f"{name!r} not found; available variables: {tuple(variables)!r}"
            )
        return self._build_dataarray(
            variables[name], metadata["coords"], name, mmap_mode
        )

    def _load_dataarray(
        self, metadata: dict[str, Any], mmap_mode: str | None
    ) -> xr.DataArray:
        """Reconstruct a stored DataArray."""
        return self._build_dataarray(
            metadata["variable"], metadata["coords"], metadata["name"], mmap_mode
        )

    def _build_dataarray(
        self,
        info: dict[str, Any],
        coord_metadata: dict[str, Any],
        name: Any,
        mmap_mode: str | None,
    ) -> xr.DataArray:
        """Assemble a DataArray from its payload and recorded coordinates."""
        return xr.DataArray(
            self._load_array(info["array"], mmap_mode),
            dims=info["dims"],
            coords=self._load_coords(coord_metadata, info["coords"], mmap_mode),
            name=name,
            attrs=info["attrs"],
        )

    def _load_coords(
        self, metadata: dict[str, Any], names: Any, mmap_mode: str | None
    ) -> dict[str, tuple[Any, ...]]:
        """Load the named coordinates."""
        return {name: self._as_variable(metadata[name], mmap_mode) for name in names}

    def _as_variable(
        self, info: dict[str, Any], mmap_mode: str | None
    ) -> tuple[Any, ...]:
        """Return the ``(dims, data, attrs)`` triple xarray builds from."""
        return (info["dims"], self._load_array(info["array"], mmap_mode), info["attrs"])

    def _load_array(self, info: dict[str, Any], mmap_mode: str | None) -> np.ndarray:
        """Load a payload and check it against its recorded dtype and shape."""
        path = self.path / info["file"]
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)

        expected_dtype = np.dtype(info["dtype"])
        expected_shape = tuple(info["shape"])
        if array.dtype != expected_dtype:
            raise TypeError(f"{path}: dtype {array.dtype} != {expected_dtype}")
        if array.shape != expected_shape:
            raise ValueError(f"{path}: shape {array.shape} != {expected_shape}")
        return array

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        """Convert metadata to JSON-safe types, rejecting what cannot survive.

        NumPy scalars and arrays are unwrapped to Python equivalents. Anything
        JSON cannot round-trip faithfully, including non-finite floats, is
        refused rather than silently written as something else.
        """
        if value is None or isinstance(value, str | bool | int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise TypeError("Non-finite float metadata are not supported.")
            return value
        if isinstance(value, np.generic):
            return cls._json_value(value.item())
        if isinstance(value, np.ndarray):
            return cls._json_value(value.tolist())
        if isinstance(value, list | tuple):
            return [cls._json_value(item) for item in value]
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("JSON metadata dictionaries require string keys.")
            return {key: cls._json_value(item) for key, item in value.items()}
        raise TypeError(f"Unsupported JSON metadata type: {type(value).__name__}.")

    @classmethod
    def _make_dir(cls, parent: Path, name: Any) -> Path:
        """Create one payload directory, rejecting reserved or unsafe names."""
        value = str(name)
        if not value or value in {cls.COORD_DIR, cls.META_FILE}:
            raise ValueError(f"Reserved or empty array name: {value!r}.")
        if "/" in value or "\\" in value:
            raise ValueError(f"Array names cannot contain path separators: {value!r}.")
        directory = parent / value
        directory.mkdir()
        return directory
