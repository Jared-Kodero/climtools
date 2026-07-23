from __future__ import annotations

import numpy as np
import xarray as xr


def sel_transect_xy(
    data: xr.Dataset | xr.DataArray,
    x: float = None,
    y: float = None,
    orientation: float = 0.0,
    width: float = 1.0,
    *,
    xdim: str = "x",
    ydim: str = "y",
    snap: bool = True,
    drop: bool = True,
):

    if x is None and y is None:
        raise ValueError("At least one value in center_point must be provided.")

    for name in (xdim, ydim):
        if name not in data.coords:
            raise ValueError(f"Input data must contain a '{name}' coordinate.")

    if x is None and y is None:
        raise ValueError("At least one of `x` or `y` must be provided.")

    if width <= 0:
        raise ValueError("width must be positive.")

    xc = data[xdim]
    yc = data[ydim]

    rectilinear = (
        xc.ndim == 1 and yc.ndim == 1 and xdim in data.dims and ydim in data.dims
    )
    if not rectilinear:
        raise ValueError(
            "Only rectilinear grids with 1D planar coordinates are supported."
        )

    if xc.sizes[xdim] < 2 or yc.sizes[ydim] < 2:
        raise ValueError("Each planar coordinate must contain at least two points.")

    dx = np.abs(xc.diff(xdim)).median(xdim)
    dy = np.abs(yc.diff(ydim)).median(ydim)

    if y is not None and x is None:
        y0 = float(yc.sel({ydim: y}, method="nearest")) if snap else float(y)
        half_width = 0.5 * width * dy
        mask = np.abs(yc - y0) <= half_width
        return data.where(mask, drop=drop)

    if x is not None and y is None:
        x0 = float(xc.sel({xdim: x}, method="nearest")) if snap else float(x)
        half_width = 0.5 * width * dx
        mask = np.abs(xc - x0) <= half_width
        return data.where(mask, drop=drop)

    if snap:
        x0 = float(xc.sel({xdim: x}, method="nearest"))
        y0 = float(yc.sel({ydim: y}, method="nearest"))
    else:
        x0 = float(x)
        y0 = float(y)

    theta = np.deg2rad(orientation % 180.0)
    nx = np.cos(theta)
    ny = -np.sin(theta)

    cross_track = (xc - x0) * nx + (yc - y0) * ny
    cell_width = np.sqrt((nx * dx) ** 2 + (ny * dy) ** 2)
    half_width = 0.5 * width * cell_width

    mask = np.abs(cross_track) <= half_width
    return data.where(mask, drop=drop)


def sel_transect_latlon(
    data: xr.Dataset | xr.DataArray,
    lat: float = None,
    lon: float = None,
    orientation: float = 0.0,
    width: float = 1.0,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    snap: bool = True,
    drop: bool = True,
):

    if lat_dim not in data.coords or lon_dim not in data.coords:
        raise ValueError(
            "Input data must contain the specified latitude and longitude coordinates."
        )

    if lat is None and lon is None:
        raise ValueError("At least one of `lat` or `lon` must be provided.")

    if width <= 0:
        raise ValueError("width must be positive.")

    latc = data[lat_dim]
    lonc = data[lon_dim]

    rectilinear = (
        latc.ndim == 1
        and lonc.ndim == 1
        and lat_dim in data.dims
        and lon_dim in data.dims
    )

    if not rectilinear:
        raise ValueError(
            "Only rectilinear grids with 1D 'lat' and 'lon' coordinates are supported."
        )

    if latc.sizes["lat"] < 2:
        raise ValueError("Latitude coordinate must contain at least two points.")

    if lonc.sizes["lon"] < 2:
        raise ValueError("Longitude coordinate must contain at least two points.")

    dlat = np.abs(latc.diff("lat")).median("lat")
    dlon = np.abs(((lonc.diff("lon") + 180.0) % 360.0) - 180.0).median("lon")

    if lat is not None and lon is None:
        lat0 = float(latc.sel(lat=lat, method="nearest")) if snap else float(lat)
        half_width_deg = 0.5 * width * dlat
        mask = np.abs(latc - lat0) <= half_width_deg
        return data.where(mask, drop=drop)

    if lon is not None and lat is None:
        if snap:
            dlon_to_centre = np.abs(((lonc - lon + 180.0) % 360.0) - 180.0)
            lon0 = float(lonc.isel(lon=dlon_to_centre.argmin("lon")))
        else:
            lon0 = float(lon)

        half_width_deg = 0.5 * width * dlon
        mask = np.abs(((lonc - lon0 + 180.0) % 360.0) - 180.0) <= half_width_deg
        return data.where(mask, drop=drop)

    if snap:
        lat0 = float(latc.sel(lat=lat, method="nearest"))

        dlon_to_centre = np.abs(((lonc - lon + 180.0) % 360.0) - 180.0)
        lon0 = float(lonc.isel(lon=dlon_to_centre.argmin("lon")))
    else:
        lat0 = float(lat)
        lon0 = float(lon)

    phi0 = np.deg2rad(lat0)
    lam0 = np.deg2rad(lon0)
    theta = np.deg2rad(orientation % 180.0)

    cross_north_weight = abs(np.sin(theta))
    cross_east_weight = abs(np.cos(theta))

    cell_width_deg = xr.apply_ufunc(
        np.sqrt,
        (cross_north_weight * dlat) ** 2
        + (cross_east_weight * dlon * np.cos(phi0)) ** 2,
        dask="allowed",
    )

    half_width_deg = 0.5 * width * cell_width_deg

    ax = np.cos(phi0) * np.cos(lam0)
    ay = np.cos(phi0) * np.sin(lam0)
    az = np.sin(phi0)

    north_x = -np.sin(phi0) * np.cos(lam0)
    north_y = -np.sin(phi0) * np.sin(lam0)
    north_z = np.cos(phi0)

    east_x = -np.sin(lam0)
    east_y = np.cos(lam0)
    east_z = 0.0

    dx = np.cos(theta) * north_x + np.sin(theta) * east_x
    dy = np.cos(theta) * north_y + np.sin(theta) * east_y
    dz = np.cos(theta) * north_z + np.sin(theta) * east_z

    gx = ay * dz - az * dy
    gy = az * dx - ax * dz
    gz = ax * dy - ay * dx

    phi = xr.apply_ufunc(np.deg2rad, latc, dask="allowed")
    lam = xr.apply_ufunc(np.deg2rad, lonc, dask="allowed")

    px = xr.apply_ufunc(np.cos, phi, dask="allowed") * xr.apply_ufunc(
        np.cos, lam, dask="allowed"
    )
    py = xr.apply_ufunc(np.cos, phi, dask="allowed") * xr.apply_ufunc(
        np.sin, lam, dask="allowed"
    )
    pz = xr.apply_ufunc(np.sin, phi, dask="allowed")

    dot_n = gx * px + gy * py + gz * pz
    dot_n = dot_n.clip(min=-1.0, max=1.0)

    cross_track_deg = xr.apply_ufunc(
        np.rad2deg,
        xr.apply_ufunc(np.arcsin, dot_n, dask="allowed"),
        dask="allowed",
    )

    mask = np.abs(cross_track_deg) <= half_width_deg

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
