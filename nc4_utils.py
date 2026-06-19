from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import cftime
import dask
import netCDF4 as nc
import numpy as np
import xarray as xr
from xarray.coding.times import encode_cf_datetime, encode_cf_timedelta

from .progress import DaskProgressBar, SerialProgressBar
from .tools import n_cpus


def is_cftime(da: xr.DataArray) -> bool:
    """True if an object dtype variable holds cftime datetimes."""
    if da.dtype != object:
        return False
    values = np.asarray(da.values).reshape(-1)
    return values.size > 0 and isinstance(values[0], cftime.datetime)


def validate_time_units(
    units_a: str | None, units_b: str | None, calendar: str | None = None
) -> bool:
    """Compare two CF time units strings by meaning, not by text."""
    if units_a is None or units_b is None or units_a == units_b:
        return True
    try:
        cal = calendar or "standard"
        probes = cftime.num2date([0.0, 100.0], units_b, cal)
        mapped = cftime.date2num(probes, units_a, cal)
        return bool(np.allclose(np.asarray(mapped, dtype=float), [0.0, 100.0]))
    except Exception:
        return units_a == units_b


def encode_time(
    da: xr.DataArray, units: str = None, calendar: str = None, dtype: np.dtype = None
):
    """Encode datetime64/cftime/timedelta64 to numeric CF values."""
    if np.issubdtype(da.dtype, np.datetime64) or is_cftime(da):
        num, out_units, out_calendar = encode_cf_datetime(
            da, units=units, calendar=calendar, dtype=dtype
        )
        encoded = da.copy(data=num)
        encoded.attrs = {**da.attrs, "units": out_units, "calendar": out_calendar}
        return encoded
    if np.issubdtype(da.dtype, np.timedelta64):
        num, out_units = encode_cf_timedelta(da, units=units, dtype=dtype)
        encoded = da.copy(data=num)
        encoded.attrs = {**da.attrs, "units": out_units}
        return encoded

    return da


def cast_dtype(arr: np.ndarray, target: np.dtype):
    if arr.dtype == target:
        return arr
    return arr.astype(target)


def createVariable(
    ncf: nc.Dataset,
    da: xr.DataArray,
    varname: str,
    zlib: bool = None,
    complevel: int = None,
    shuffle: bool = None,
    write_values: bool = False,
) -> nc.Variable:
    missing = [d for d in da.dims if d not in ncf.dimensions]
    if missing:
        raise ValueError(
            f"Cannot create {varname} in {ncf.filepath()}: missing dimensions {missing}"
        )

    kwargs = {}
    if zlib is not None:
        kwargs["zlib"] = zlib
    if complevel is not None:
        kwargs["complevel"] = complevel
    if shuffle is not None:
        kwargs["shuffle"] = shuffle

    ncvar = ncf.createVariable(
        varname=varname,
        datatype=da.dtype,
        dimensions=da.dims,
        **kwargs,
    )
    for attr_name, attr_val in da.attrs.items():
        ncvar.setncattr(attr_name, attr_val)

    if write_values:
        ncvar[:] = da.values

    return ncvar


def validateVariable(
    ncf: nc.Dataset, da: xr.DataArray, dim: str, varname: str, exists: bool, enc: dict
):

    enc_units = enc.get("units")
    enc_dtype = np.dtype(enc["dtype"]) if enc.get("dtype") is not None else None

    time_like = (
        np.issubdtype(da.dtype, np.datetime64)
        or np.issubdtype(da.dtype, np.timedelta64)
        or is_cftime(da)
    )
    if time_like:
        units = calendar = on_disk_dtype = None
        if exists:
            ncv = ncf.variables[varname]
            a = ncv.ncattrs()
            units = ncv.getncattr("units") if "units" in a else None
            calendar = ncv.getncattr("calendar") if "calendar" in a else None
            on_disk_dtype = ncv.dtype

        # time_encoding applies to the append coordinate only, and only
        # to define it on first write. The file is authoritative once the
        # coordinate exists.
        if varname == dim:
            enc_calendar = enc.get("calendar")

            if exists:
                if enc_units is not None and not validate_time_units(
                    units, enc_units, calendar
                ):
                    raise ValueError(
                        f"time_encoding units {enc_units!r} conflict with units {units!r} already stored for {varname!r} "
                    )
                if (
                    enc_calendar is not None
                    and calendar is not None
                    and enc_calendar != calendar
                ):
                    raise ValueError(
                        f"time_encoding calendar {enc_calendar!r} conflicts with calendar {calendar!r} stored for {varname!r} on file"
                    )
            else:
                units = enc_units if units is None else units
                calendar = enc_calendar if calendar is None else calendar
                if on_disk_dtype is None:
                    on_disk_dtype = enc_dtype

        da = encode_time(da, units, calendar, on_disk_dtype)
    return da


def serial_write_netcdf(
    file: Path,
    data: xr.Dataset,
    unlimited_dim: str = None,
    batch_size: int = 1,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
) -> None:

    file = Path(file)
    file.unlink(missing_ok=True)

    enc = {
        v: {"zlib": zlib, "complevel": complevel, "shuffle": shuffle}
        for v in data.data_vars
    }

    dim0 = unlimited_dim if unlimited_dim is not None else next(iter(data.sizes))

    if dim0 not in data.sizes:
        raise ValueError(f"{dim0!r} is not a dimension in data.")

    n_items = data.sizes[dim0]

    if n_items < 1:
        raise ValueError(f"Cannot write an empty dimension: {dim0!r}.")

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    # First write defines the file, dimensions, variables, attrs, and encodings.
    # Keep this as a single record.
    data0 = data.isel({dim0: slice(0, 1)})

    data0.to_netcdf(
        file,
        encoding=enc,
        format=format,
        unlimited_dims=[dim0],
    )

    # Append the remaining records in batches.
    starts = range(1, n_items, batch_size)

    if show_progress:
        data_slices = SerialProgressBar(
            starts,
            description="Writing NetCDF",
            stdout=stdout,
        )
    else:
        data_slices = starts

    for start in data_slices:
        stop = min(start + batch_size, n_items)

        append_to_netcdf(
            file,
            data.isel({dim0: slice(start, stop)}),
            dim=dim0,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
        )


def parallel_write_netcdf(
    file,
    data,
    unlimited_dim: str = None,
    batch_size: int = 1,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
):
    # Parallel write by sharding along one dimension.
    # Users can reconstruct with xr.open_mfdataset until true parallel NetCDF4 writes are supported.

    file = Path(file)

    directory = file.parent / file.stem
    directory.mkdir(parents=True, exist_ok=True)

    dim0 = unlimited_dim if unlimited_dim is not None else next(iter(data.sizes))

    if dim0 not in data.sizes:
        raise ValueError(f"{dim0!r} is not a dimension in data.")

    n_items = data.sizes[dim0]

    if n_items < 1:
        raise ValueError(f"Cannot write an empty dimension: {dim0!r}.")

    target_file_size_gb = 4.0
    target_file_size_bytes = int(target_file_size_gb * 1024**3)

    estimated_uncompressed_bytes = max(int(data.nbytes), 1)

    if zlib:
        compression_level = max(0, min(int(complevel), 9))
        compression_factor = 4.0 + float(compression_level)
    else:
        compression_factor = 1.0

    estimated_output_bytes = math.ceil(
        estimated_uncompressed_bytes / compression_factor
    )

    n_files = max(
        1,
        math.ceil(estimated_output_bytes / target_file_size_bytes),
    )
    n_files = min(n_files, n_items)

    chunk_size = math.ceil(n_items / n_files)

    slices = [
        (start, min(start + chunk_size, n_items))
        for start in range(0, n_items, chunk_size)
    ]

    width = max(2, len(str(len(slices))))

    output_files = [
        directory / f"{file.stem}.{i + 1:0{width}d}.nc" for i in range(len(slices))
    ]

    for output_file in output_files:
        output_file.unlink(missing_ok=True)

    tasks = [
        dask.delayed(serial_write_netcdf)(
            output_file,
            data.isel({dim0: slice(start, end)}),
            dim0,
            batch_size,
            format,
            shuffle,
            zlib,
            complevel,
            False,
            stdout,
        )
        for output_file, (start, end) in zip(output_files, slices)
    ]

    # Worker count is a concurrency decision.
    max_workers = max(1, n_cpus // 2)
    n_workers = min(len(tasks), max_workers)

    if show_progress:
        with DaskProgressBar(description="Writing NetCDF", stdout=stdout):
            dask.compute(*tasks, scheduler="processes", num_workers=n_workers)
    else:
        dask.compute(*tasks, scheduler="processes", num_workers=n_workers)

    return output_files


def append_to_netcdf(
    file: Path,
    data: xr.Dataset,
    dim: str = "time",
    mode: Literal["a", "r+"] = "r+",
    format: str = "NETCDF4",
    shuffle: bool = None,
    zlib: bool = None,
    complevel: int = None,
    encoding: dict = None,
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
    data : xr.Dataset
        Data to append. All variables containing ``dim`` must share the same
        length along ``dim``.
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
    encoding : dict, optional
        Encoding dict for the coordinate specified by ``unlimited_dim``.
    """

    if isinstance(data, xr.DataArray):
        ds = data.to_dataset()
    else:
        ds = data

    if dim not in ds.sizes:
        raise ValueError(f"Append dimension {dim!r} not present in the data")

    n_new = ds.sizes[dim]

    enc = encoding or dict(data[dim].encoding)

    with nc.Dataset(file, mode=mode, format=format) as ncf:
        if dim not in ncf.dimensions:
            raise ValueError(f"Append dimension {dim!r} not found in {file}")

        # Confirm dim is the unlimited axis, and report the actual one on mismatch.
        unlimited = [d for d, o in ncf.dimensions.items() if o.isunlimited()]
        if dim not in unlimited:
            raise ValueError(
                f"Dimension {dim!r} in {file} is not unlimited; unlimited dimension(s): {unlimited or 'none'}"
            )

        offset = ncf.dimensions[dim].size

        for varname, da in {**ds.coords, **ds.data_vars}.items():
            exists = varname in ncf.variables

            da = validateVariable(ncf, da, dim, varname, exists, enc)
            # Static variables: write once on creation, then leave untouched.
            if dim not in da.dims:
                if not exists:
                    _ = createVariable(
                        ncf,
                        da,
                        varname,
                        zlib=zlib,
                        complevel=complevel,
                        shuffle=shuffle,
                        write_values=True,
                    )
                continue

            # First append for this variable creates it with size 0 along dim.
            if not exists:
                _ = createVariable(
                    ncf,
                    da,
                    varname,
                    zlib=zlib,
                    complevel=complevel,
                    shuffle=shuffle,
                    write_values=True,
                )

            ncvar = ncf.variables[varname]
            arr = da.transpose(*ncvar.dimensions).values
            arr = cast_dtype(arr, ncvar.dtype)

            index = tuple(
                slice(offset, offset + n_new) if d == dim else slice(None)
                for d in ncvar.dimensions
            )
            ncvar[index] = arr


def write_netcdf_variable(
    file: Path,
    da: xr.DataArray,
    name: str = None,
    mode: Literal["a", "r+"] = "r+",
    format="NETCDF4",
    shuffle: bool = None,
    zlib: bool = None,
    complevel: int = None,
) -> None:
    """Write / append a DataArray to a NetCDF file

    Parameters
    ----------
    file : Path
        Path to a NetCDF4 file opened with read/write access.
    da : xr.DataArray
        DataArray to write. Must have dimensions that already exist in the file.
    name : str, optional
        Name of the variable to create in the NetCDF file. If None, uses da.name.
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

    if not isinstance(da, xr.DataArray):
        raise ValueError("da must be an xarray.DataArray")

    with nc.Dataset(file, mode=mode, format=format) as ncf:
        varname = name or da.name

        # Overwrite values if the variable was created on a previous run.
        if varname in ncf.variables:
            ncf.variables[varname][:] = da.values

        else:
            ncvar = createVariable(
                ncf,
                da,
                varname,
                zlib=zlib,
                complevel=complevel,
                shuffle=shuffle,
                write_values=False,
            )

            ncvar[:] = da.values
