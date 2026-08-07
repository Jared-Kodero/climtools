from __future__ import annotations

import hashlib
import operator
from collections.abc import Callable, Sequence
from typing import Any, Literal

import cartopy.util
import numpy as np
import numpy.typing as npt
import xarray as xr
from cf_xarray import *
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


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


def label_objects_2d(
    data: npt.NDArray[np.floating[Any]],
    threshold: float,
    operator: Callable[[Any, Any], Any],
    center_on: Literal["min", "max"],
    connectivity: int,
    merge_cells: int,
    max_objects: int,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    np.int64,
]:
    """Label threshold-defined objects and locate their extrema in one 2-D field."""

    values = np.asarray(data, dtype=np.float64)
    structure = ndimage.generate_binary_structure(2, connectivity)
    mask = np.isfinite(values) & operator(values, threshold)

    grouped = mask
    if merge_cells > 0:
        grouped = ndimage.binary_dilation(
            mask,
            structure=structure,
            iterations=merge_cells,
        )

    raw_labels, _ = ndimage.label(grouped, structure=structure)
    raw_labels[~mask] = 0
    raw_ids = np.unique(raw_labels)
    raw_ids = raw_ids[raw_ids != 0]

    peak_y_out = np.full(max_objects, -1, dtype=np.int64)
    peak_x_out = np.full(max_objects, -1, dtype=np.int64)
    peak_value_out = np.full(max_objects, np.nan, dtype=np.float64)
    if raw_ids.size == 0:
        return (
            np.zeros_like(raw_labels, dtype=np.int32),
            peak_y_out,
            peak_x_out,
            peak_value_out,
            np.int64(0),
        )

    position_function = (
        ndimage.maximum_position if center_on == "max" else ndimage.minimum_position
    )
    positions = np.atleast_2d(
        position_function(values, labels=raw_labels, index=raw_ids)
    )
    peak_y, peak_x = np.asarray(positions, dtype=np.int64).T
    peak_value = values[peak_y, peak_x]

    order = np.argsort(peak_value)
    if center_on == "max":
        order = order[::-1]

    raw_ids = raw_ids[order]
    peak_y = peak_y[order]
    peak_x = peak_x[order]
    peak_value = peak_value[order]

    object_count = raw_ids.size
    lookup = np.zeros(int(raw_labels.max()) + 1, dtype=np.int32)
    lookup[raw_ids] = np.arange(1, object_count + 1, dtype=np.int32)

    peak_y_out[:object_count] = peak_y
    peak_x_out[:object_count] = peak_x
    peak_value_out[:object_count] = peak_value

    return (
        lookup[raw_labels],
        peak_y_out,
        peak_x_out,
        peak_value_out,
        np.int64(object_count),
    )


def match_objects_1d(
    reference_peak_y: npt.NDArray[np.int64],
    reference_peak_x: npt.NDArray[np.int64],
    reference_count: np.int64,
    candidate_peak_y: npt.NDArray[np.int64],
    candidate_peak_x: npt.NDArray[np.int64],
    candidate_count: np.int64,
    *,
    lat_values: npt.NDArray[np.float64],
    lon_values: npt.NDArray[np.float64],
    earth_radius_km: float,
    max_objects: int,
    match_radius_km: float | None,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    """Match candidate objects one-to-one to reference objects by center distance."""

    matched_index = np.full(max_objects, -1, dtype=np.int64)
    match_distance = np.full(max_objects, np.nan, dtype=np.float64)
    matched = np.zeros(max_objects, dtype=np.bool_)

    n_reference = int(reference_count)
    n_candidate = int(candidate_count)
    if n_reference == 0 or n_candidate == 0:
        return matched_index, match_distance, matched

    reference_y = reference_peak_y[:n_reference]
    reference_x = reference_peak_x[:n_reference]
    candidate_y = candidate_peak_y[:n_candidate]
    candidate_x = candidate_peak_x[:n_candidate]

    reference_lat = np.deg2rad(lat_values[reference_y])[:, None]
    reference_lon = np.deg2rad(lon_values[reference_x])[:, None]
    candidate_lat = np.deg2rad(lat_values[candidate_y])[None, :]
    candidate_lon = np.deg2rad(lon_values[candidate_x])[None, :]

    delta_lat = candidate_lat - reference_lat
    delta_lon = candidate_lon - reference_lon
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(reference_lat) * np.cos(candidate_lat) * np.sin(delta_lon / 2.0) ** 2
    )
    haversine = np.clip(haversine, 0.0, 1.0)
    distance = 2.0 * earth_radius_km * np.arcsin(np.sqrt(haversine))

    assignment_cost = distance
    if match_radius_km is not None:
        unmatched_cost = np.nextafter(match_radius_km, np.inf)
        invalid_cost = unmatched_cost * (n_reference + 1)
        candidate_cost = np.where(
            distance <= match_radius_km,
            distance,
            invalid_cost,
        )
        dummy_cost = np.full(
            (n_reference, n_reference),
            unmatched_cost,
            dtype=np.float64,
        )
        assignment_cost = np.concatenate((candidate_cost, dummy_cost), axis=1)

    reference_indices, assigned_columns = linear_sum_assignment(assignment_cost)
    accepted = assigned_columns < n_candidate
    accepted_reference = reference_indices[accepted]
    accepted_candidate = assigned_columns[accepted]
    if match_radius_km is not None:
        accepted_distance = distance[accepted_reference, accepted_candidate]
        within_radius = accepted_distance <= match_radius_km
        accepted_reference = accepted_reference[within_radius]
        accepted_candidate = accepted_candidate[within_radius]

    matched_index[accepted_reference] = accepted_candidate
    match_distance[accepted_reference] = distance[
        accepted_reference,
        accepted_candidate,
    ]
    matched[accepted_reference] = True

    return matched_index, match_distance, matched


def _detect_objects(
    object_data: xr.DataArray,
    *,
    threshold: float,
    operator_function: Callable[[Any, Any], Any],
    center_on: Literal["min", "max"],
    connectivity: int,
    merge_cells: int,
    max_objects: int,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Apply two-dimensional object detection over all leading dimensions."""

    labels, peak_y, peak_x, peak_value, object_count = xr.apply_ufunc(
        label_objects_2d,
        object_data,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[
            ["lat", "lon"],
            ["candidate"],
            ["candidate"],
            ["candidate"],
            [],
        ],
        kwargs={
            "threshold": threshold,
            "operator": operator_function,
            "center_on": center_on,
            "connectivity": connectivity,
            "merge_cells": merge_cells,
            "max_objects": max_objects,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32, np.int64, np.int64, np.float64, np.int64],
        dask_gufunc_kwargs={
            "output_sizes": {"candidate": max_objects},
            "allow_rechunk": True,
        },
    )
    labels = labels.transpose(*object_data.dims).assign_coords(
        {
            dim: object_data.coords[dim]
            for dim in object_data.dims
            if dim in object_data.coords
        }
    )
    labels.name = "object_label"
    return labels, peak_y, peak_x, peak_value, object_count


def _peak_coordinates(
    peak_y: xr.DataArray,
    peak_x: xr.DataArray,
    valid: xr.DataArray,
    *,
    lat_values: npt.NDArray[np.float64],
    lon_values: npt.NDArray[np.float64],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert integer peak indices to latitude and longitude coordinates."""

    peak_y_values = np.asarray(peak_y.where(valid, 0).values, dtype=np.int64)
    peak_x_values = np.asarray(peak_x.where(valid, 0).values, dtype=np.int64)
    peak_lat = xr.DataArray(
        lat_values[peak_y_values],
        dims=peak_y.dims,
        coords=peak_y.coords,
    ).where(valid)
    peak_lon = xr.DataArray(
        lon_values[peak_x_values],
        dims=peak_x.dims,
        coords=peak_x.coords,
    ).where(valid)
    return peak_lat, peak_lon


def center_on_objects(
    ds: xr.Dataset,
    object_var: str,
    threshold: float,
    *,
    threshold_operator: Literal[">", ">=", "<", "<="] = ">=",
    center_on: Literal["min", "max"] = "max",
    connectivity: int = 2,
    merge_cells: int = 1,
    dx_km: float = 3.0,
    variables: str | Sequence[str] | None = None,
    half_extent_km: float | None = None,
    method: Literal["linear", "nearest"] = "linear",
    reference_data: dict[str, Any] | None = None,
    match_radius_km: float | None = None,
) -> xr.Dataset:
    """
    Extract local Cartesian grids centered on threshold-defined objects.

    When ``reference_data`` is provided, objects in the selected reference slice define the
    canonical event identities and ``obj`` ordering. Objects are then detected in every
    full-data slice and matched one-to-one to the reference events by great-circle
    distance. Each slice is interpolated about its matched, member-specific center, so
    spatially displaced events retain the reference ``obj`` index.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the object variable and target variables.
        Must contain one-dimensional ``lat`` and ``lon`` coordinates.
    object_var : str
        Name of the variable used to identify objects. Must have dimensions
        ``(*leading_dims, lat, lon)``.
    threshold : float
        Threshold value defining object boundaries.
    threshold_operator : {">", ">=", "<", "<="}, default: ">="
        Logical operator applied to compare the object variable against the threshold.
    center_on : {"max", "min"}, default: "max"
        Extremum used as the center of each object.
    connectivity : int, default: 2
        Pixel connectivity for object labeling. Must be 1 or 2.
    merge_cells : int, default: 1
        Number of binary-dilation grid cells used to merge proximate objects.
    dx_km : float, default: 3.0
        Grid spacing of the output local Cartesian grid in kilometres.
    variables : str, Sequence[str], or None, default: None
        Dataset variables to interpolate. If None, all variables containing both
        ``lat`` and ``lon`` dimensions are selected.
    half_extent_km : float or None, default: None
        Half-width of the output Cartesian domain in kilometres. If None, the extent
        is one quarter of the longitudinal span at the mean latitude.
    method : {"linear", "nearest"}, default: "linear"
        Interpolation algorithm used for mapping source fields to the local grid.
    reference_data : dict[str, Any] or None, default: None
        Scalar coordinate selections defining the reference slice, for example
        ``{"mem": 0}``. Reference objects define canonical event identities. If None,
        objects remain independent in every leading-dimension slice.
    match_radius_km : float or None, default: None
        Maximum permitted distance between a reference object and a matched object.
        Assignments beyond this distance are invalid. If None, no distance cutoff is
        applied.

    Returns
    -------
    xr.Dataset
        Object-centered fields with dimensions ``(*leading_dims, obj, y, x)``. Under
        reference mapping, ``obj`` follows the reference ordering while ``peak_lat``
        and ``peak_lon`` contain matched slice-specific centers.
    """
    earth_radius_km = 6371.0
    comparators: dict[str, Callable[[Any, Any], Any]] = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
    }

    if threshold_operator not in comparators:
        raise ValueError(f"Unsupported threshold operator: {threshold_operator!r}.")
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2 for two-dimensional labeling.")
    if merge_cells < 0:
        raise ValueError("merge_cells must be greater than or equal to zero.")
    if dx_km <= 0.0:
        raise ValueError("dx_km must be greater than zero.")
    if match_radius_km is not None and match_radius_km <= 0.0:
        raise ValueError("match_radius_km must be greater than zero when provided.")

    _dataset_coord = ds.coords

    object_data = ds[object_var]
    dims = object_data.dims
    if dims[-2:] != ("lat", "lon"):
        raise ValueError(
            f"{object_var!r} must end with dimensions ('lat', 'lon'); got {dims!r}."
        )

    leading_dims = dims[:-2]

    for coordinate in ("lat", "lon"):
        if coordinate not in ds.coords:
            raise ValueError(f"Dataset is missing the {coordinate!r} coordinate.")
        if ds[coordinate].dims != (coordinate,):
            raise ValueError(f"Coordinate {coordinate!r} must be one-dimensional.")

    if variables is None:
        variable_names = [
            name
            for name, data in ds.data_vars.items()
            if {"lat", "lon"}.issubset(data.dims)
        ]
    elif isinstance(variables, str):
        variable_names = [variables]
    else:
        variable_names = list(variables)

    variable_names = list(dict.fromkeys((object_var, *variable_names)))
    object_output_var = f"{object_var}_object"
    if object_output_var in variable_names:
        raise ValueError(
            f"Output variable name {object_output_var!r} conflicts with an input variable."
        )

    lat_values = np.asarray(ds["lat"].values, dtype=np.float64)
    lon_values = np.asarray(ds["lon"].values, dtype=np.float64)
    max_objects = object_data.sizes["lat"] * object_data.sizes["lon"]

    labels, candidate_y, candidate_x, candidate_value, candidate_count = (
        _detect_objects(
            object_data,
            threshold=threshold,
            operator_function=comparators[threshold_operator],
            center_on=center_on,
            connectivity=connectivity,
            merge_cells=merge_cells,
            max_objects=max_objects,
        )
    )

    if reference_data is None:
        reference_y = candidate_y
        reference_x = candidate_x
        reference_value = candidate_value
        reference_count = candidate_count
    else:
        reference_data = object_data.sel(reference_data, drop=True)
        _, reference_y, reference_x, reference_value, reference_count = _detect_objects(
            reference_data,
            threshold=threshold,
            operator_function=comparators[threshold_operator],
            center_on=center_on,
            connectivity=connectivity,
            merge_cells=merge_cells,
            max_objects=max_objects,
        )

    max_object_count = int(reference_count.max().compute().item())
    total_reference_count = int(reference_count.sum().compute().item())
    if max_object_count == 0:
        raise ValueError("No valid reference objects identified matching the criteria.")

    object_ids = np.arange(max_object_count, dtype=np.int64)
    object_index = xr.DataArray(
        object_ids,
        dims="object",
        coords={"object": object_ids},
    )

    reference_y = reference_y.isel(candidate=slice(max_object_count)).rename(
        {"candidate": "object"}
    )
    reference_x = reference_x.isel(candidate=slice(max_object_count)).rename(
        {"candidate": "object"}
    )
    reference_value = reference_value.isel(candidate=slice(max_object_count)).rename(
        {"candidate": "object"}
    )
    reference_valid = object_index < reference_count
    reference_valid = reference_valid.transpose(
        *reference_count.dims,
        "object",
    )
    reference_value = reference_value.transpose(*reference_valid.dims).where(
        reference_valid
    )
    reference_peak_lat, reference_peak_lon = _peak_coordinates(
        reference_y,
        reference_x,
        reference_valid,
        lat_values=lat_values,
        lon_values=lon_values,
    )

    if reference_data is None:
        peak_y = reference_y
        peak_x = reference_x
        peak_value = reference_value
        valid_object = reference_valid
        matched_object = reference_valid
        match_distance = xr.zeros_like(reference_value, dtype=np.float64).where(
            reference_valid
        )
        object_labels = xr.where(valid_object, object_index + 1, 0).astype(np.int32)
        object_count = candidate_count
    else:
        matched_index, match_distance, matched_object = xr.apply_ufunc(
            match_objects_1d,
            reference_y.rename({"object": "reference_input"}),
            reference_x.rename({"object": "reference_input"}),
            reference_count,
            candidate_y,
            candidate_x,
            candidate_count,
            input_core_dims=[
                ["reference_input"],
                ["reference_input"],
                [],
                ["candidate"],
                ["candidate"],
                [],
            ],
            output_core_dims=[
                ["reference_candidate"],
                ["reference_candidate"],
                ["reference_candidate"],
            ],
            kwargs={
                "lat_values": lat_values,
                "lon_values": lon_values,
                "earth_radius_km": earth_radius_km,
                "max_objects": max_objects,
                "match_radius_km": match_radius_km,
            },
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int64, np.float64, np.bool_],
            dask_gufunc_kwargs={
                "output_sizes": {"reference_candidate": max_objects},
                "allow_rechunk": True,
            },
        )
        matched_index = matched_index.isel(
            reference_candidate=slice(max_object_count)
        ).rename({"reference_candidate": "object"})
        match_distance = match_distance.isel(
            reference_candidate=slice(max_object_count)
        ).rename({"reference_candidate": "object"})
        matched_object = matched_object.isel(
            reference_candidate=slice(max_object_count)
        ).rename({"reference_candidate": "object"})

        reference_valid_broadcast, matched_object = xr.broadcast(
            reference_valid,
            matched_object,
        )
        reference_valid_broadcast = reference_valid_broadcast.transpose(
            *leading_dims,
            "object",
        )
        matched_object = matched_object.transpose(*leading_dims, "object")
        valid_object = reference_valid_broadcast & matched_object
        match_distance = match_distance.transpose(*leading_dims, "object").where(
            valid_object
        )

        safe_index = matched_index.where(valid_object, 0).astype(np.int64).compute()
        peak_y = candidate_y.isel(candidate=safe_index).where(valid_object)
        peak_x = candidate_x.isel(candidate=safe_index).where(valid_object)
        peak_value = candidate_value.isel(candidate=safe_index).where(valid_object)
        peak_y = peak_y.transpose(*leading_dims, "object")
        peak_x = peak_x.transpose(*leading_dims, "object")
        peak_value = peak_value.transpose(*leading_dims, "object")
        object_labels = xr.where(valid_object, matched_index + 1, 0).astype(np.int32)
        object_labels = object_labels.transpose(*leading_dims, "object")
        object_count = valid_object.sum("object").astype(np.int64)

    peak_lat, peak_lon = _peak_coordinates(
        peak_y,
        peak_x,
        valid_object,
        lat_values=lat_values,
        lon_values=lon_values,
    )

    source = ds[variable_names].sortby("lat").sortby("lon")
    labels = labels.sortby("lat").sortby("lon")

    if half_extent_km is None:
        mean_lat_rad = np.deg2rad(float(source["lat"].mean()))
        lon_span_rad = np.deg2rad(abs(float(source["lon"].max() - source["lon"].min())))
        extent_km = earth_radius_km * np.cos(mean_lat_rad) * lon_span_rad / 4.0
    else:
        extent_km = half_extent_km

    half_cells = int(np.floor(extent_km / dx_km))
    if half_cells < 1:
        raise ValueError("half_extent_km must be greater than or equal to dx_km.")

    offsets = np.arange(-half_cells, half_cells + 1, dtype=np.float64) * dx_km
    x_km, y_km = np.meshgrid(offsets, offsets)

    interpolation_peak_lat = peak_lat.fillna(float(source["lat"].values[0]))
    interpolation_peak_lon = peak_lon.fillna(float(source["lon"].values[0]))
    target_lat = np.asarray(interpolation_peak_lat.values)[
        ..., None, None
    ] + np.rad2deg(y_km / earth_radius_km)

    cos_lat = np.cos(np.deg2rad(target_lat))
    cos_lat = np.where(np.abs(cos_lat) < 1e-10, 1e-10, cos_lat)
    target_lon = np.asarray(interpolation_peak_lon.values)[
        ..., None, None
    ] + np.rad2deg(x_km / (earth_radius_km * cos_lat))

    metadata_coords: dict[str, Any] = {"object": object_ids}
    for dim in leading_dims:
        if dim in object_data.coords and object_data[dim].dims == (dim,):
            metadata_coords[dim] = object_data[dim]

    target_dims = (*leading_dims, "object", "y", "x")
    target_coords = {
        **metadata_coords,
        "y": offsets,
        "x": offsets,
    }
    lat_target = xr.DataArray(
        target_lat,
        dims=target_dims,
        coords=target_coords,
    )
    lon_target = xr.DataArray(
        target_lon,
        dims=target_dims,
        coords=target_coords,
    )

    centered = source.interp(
        lat=lat_target,
        lon=lon_target,
        method=method,
        assume_sorted=True,
        kwargs={"bounds_error": False, "fill_value": np.nan},
    ).drop_vars(["lat", "lon"], errors="ignore")
    centered = centered.where(valid_object)

    labels_interp = labels.interp(
        lat=lat_target,
        lon=lon_target,
        method="nearest",
        assume_sorted=True,
        kwargs={"bounds_error": False, "fill_value": 0},
    ).drop_vars(["lat", "lon"], errors="ignore")

    centered[object_output_var] = centered[object_var].where(
        valid_object & (labels_interp == object_labels)
    )

    coordinates: dict[str, Any] = {
        "object_label": object_labels,
        "peak_lat": peak_lat,
        "peak_lon": peak_lon,
        "peak_value": peak_value,
        "valid_object": valid_object,
        "matched_object": matched_object,
        "match_distance_km": match_distance,
        "object_count": object_count,
        "reference_peak_lat": reference_peak_lat,
        "reference_peak_lon": reference_peak_lon,
        "reference_peak_value": reference_value,
        "reference_valid_object": reference_valid,
        "reference_object_count": reference_count,
    }
    centered = centered.assign_coords(coordinates)

    centered["x"].attrs = {
        "long_name": "eastward distance from matched object center",
        "units": "km",
    }
    centered["y"].attrs = {
        "long_name": "northward distance from matched object center",
        "units": "km",
    }
    centered["object_count"].attrs["long_name"] = (
        "number of valid matched objects in each leading-dimension slice"
    )
    centered["reference_object_count"].attrs["long_name"] = (
        "number of canonical objects in each reference slice"
    )
    centered["valid_object"].attrs["long_name"] = "valid matched object slot"
    centered["matched_object"].attrs["long_name"] = (
        "reference object has a matched object in this slice"
    )
    centered["match_distance_km"].attrs = {
        "long_name": "great-circle distance between reference and matched centers",
        "units": "km",
    }
    centered["peak_lat"].attrs["units"] = "degrees_north"
    centered["peak_lon"].attrs["units"] = "degrees_east"
    centered["reference_peak_lat"].attrs["units"] = "degrees_north"
    centered["reference_peak_lon"].attrs["units"] = "degrees_east"
    centered["peak_value"].attrs["units"] = ds[object_var].attrs.get("units", "")
    centered["reference_peak_value"].attrs["units"] = ds[object_var].attrs.get(
        "units", ""
    )

    matched_total = int(valid_object.sum().compute().item())
    centered.attrs.update(
        {
            "object_variable": object_var,
            "threshold": threshold,
            "threshold_operator": threshold_operator,
            "center_on": center_on,
            "connectivity": connectivity,
            "merge_cells": merge_cells,
            "reference_object_count": total_reference_count,
            "matched_object_count": matched_total,
            "max_objects_per_reference_slice": max_object_count,
            "horizontal_spacing_km": dx_km,
            "half_extent_km": half_cells * dx_km,
            "object_mapping": "independent"
            if reference_data is None
            else repr(reference_data),
            "matching_method": (
                "independent"
                if reference_data is None
                else "one-to-one minimum great-circle distance"
            ),
        }
    )
    if match_radius_km is not None:
        centered.attrs["match_radius_km"] = match_radius_km

    return centered.rename({"object": "obj"})


def get_relative_time(
    ds: xr.Dataset,
    data_var: str,
    delta_time: int,
    intensity_edges: tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100),
    threshold: float = 0.1,
) -> xr.Dataset | None:
    """Extract symmetric event windows and compute intensity composites.

    Each event is centered at ``relative_time == 0``. The original ``time``
    coordinate becomes ``valid_time(event, relative_time)``. All dimensions
    other than ``time`` are retained on the extracted event variables.

    Parameters
    ----------
    ds
        Input dataset containing a ``time`` coordinate.
    data_var
        Variable used to identify dry-to-wet event triggers and peak intensity.
    delta_time
        Number of time steps retained before and after each trigger.
    intensity_edges
        Lower edges of the peak-intensity bins. The final bin is open-ended.
    threshold
        Values greater than this threshold are wet.

    Returns
    -------
    xr.Dataset or None
        Event-centered data and binned composites, or ``None`` if no event is
        found.
    """
    if delta_time < 1:
        raise ValueError("delta_time must be at least 1")

    field = ds[data_var]
    if "time" not in field.dims:
        raise ValueError(f"{data_var!r} must contain the 'time' dimension")

    dry = field <= threshold
    wet = field > threshold

    pre_dry = dry.rolling(time=delta_time, min_periods=delta_time).min().shift(time=1)
    peak_fwd = (
        field.rolling(time=delta_time + 1, min_periods=delta_time + 1)
        .max()
        .shift(time=-delta_time)
    )
    trigger = wet & (pre_dry == 1) & peak_fwd.notnull()

    label_dims = tuple(field.dims)
    labels = trigger.stack(event=label_dims)
    labels = labels.where(labels, drop=True).reset_index("event").compute()
    if labels.sizes.get("event", 0) == 0:
        return None

    time_index = ds.get_index("time")
    center_positions = time_index.get_indexer(labels["time"].values)
    complete = (center_positions >= delta_time) & (
        center_positions < ds.sizes["time"] - delta_time
    )
    labels = labels.isel(event=np.flatnonzero(complete))
    center_positions = center_positions[complete]
    if labels.sizes.get("event", 0) == 0:
        return None

    relative_time = xr.DataArray(
        np.arange(-delta_time, delta_time + 1, dtype=np.int32),
        dims="relative_time",
        name="relative_time",
    )
    window_indices = (
        xr.DataArray(
            center_positions,
            dims="event",
        )
        + relative_time
    )

    events = ds.isel(time=window_indices).rename({"time": "valid_time"})
    events = events.assign_coords(
        event=np.arange(events.sizes["event"], dtype=np.int64),
        relative_time=relative_time,
        trigger_time=("event", labels["time"].values),
    )

    point_indexers = {dim: labels[dim] for dim in field.dims}
    peak = peak_fwd.sel(point_indexers)
    events = events.assign_coords(peak=("event", peak.values))

    for dim in field.dims:
        if dim == "time":
            continue
        coord_name = f"event_{dim}"
        events = events.assign_coords({coord_name: ("event", labels[dim].values)})

    bins = [*intensity_edges, np.inf]
    bin_labels = list(intensity_edges)
    if "lat" in field.dims:
        weights = np.cos(np.deg2rad(events["event_lat"]))
    else:
        weights = xr.ones_like(events["peak"], dtype=np.float64)

    event_vars = [
        name
        for name, variable in events.data_vars.items()
        if "event" in variable.dims and np.issubdtype(variable.dtype, np.number)
    ]
    composite_source = events[event_vars].reset_coords(drop=True)
    composite_source = composite_source.assign_coords(peak=events["peak"])

    numerator = (
        (composite_source * weights)
        .groupby_bins("peak", bins=bins, labels=bin_labels)
        .sum(dim="event")
    )
    denominator = weights.groupby_bins(
        "peak",
        bins=bins,
        labels=bin_labels,
    ).sum(dim="event")

    composite = numerator / denominator
    composite["n_events"] = (
        xr.ones_like(events["peak"], dtype=np.int32)
        .groupby_bins("peak", bins=bins, labels=bin_labels)
        .sum(dim="event")
        .fillna(0)
        .astype(np.int32)
    )
    composite = composite.rename(
        {
            name: f"{name}_composite"
            for name in composite.data_vars
            if name != "n_events"
        }
    )

    store = xr.merge([events, composite], combine_attrs="no_conflicts")
    store.attrs["delta_time"] = delta_time
    store.attrs["intensity_edges"] = list(intensity_edges)
    return store
