from __future__ import annotations

import hashlib
from typing import Literal

import cartopy.util
import numpy as np
import xarray as xr
from cf_xarray import *


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
            "Could not determine longitude and latitude coordinate names, specify x and y"
        )

    return lon.name, lat.name


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
    xdim: str = None,
    ydim: str = None,
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

    if xdim not in data.coords or ydim not in data.coords:
        xdim, ydim = get_spatial_dims(data)

    if x is None and y is None:
        raise ValueError("At least one of `x` or `y` must be provided.")

    if width <= 0:
        raise ValueError("`width` must be positive.")

    if geometry not in {"xy", "latlon"}:
        raise ValueError("`geometry` must be either 'xy' or 'latlon'.")

    xc = data[xdim]
    yc = data[ydim]

    if xc.ndim != 1 or yc.ndim != 1 or xc.dims != (xdim,) or yc.dims != (ydim,):
        raise ValueError(
            "Coordinates must define a rectilinear grid with one-dimensional coordinate variables."
        )

    if xc.size < 2 or yc.size < 2:
        raise ValueError("Each coordinate must contain at least two points.")

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
