from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cftime
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.times import encode_cf_datetime, encode_cf_timedelta

from .progress import SerialProgressBar


def is_cftime(da: xr.DataArray) -> bool:
    """True if an object dtype variable holds cftime datetimes."""
    if da.dtype != object:
        return False
    values = np.asarray(da.values).reshape(-1)
    return values.size > 0 and isinstance(values[0], cftime.datetime)


def encode_time(da: xr.DataArray):
    """Encode datetime64/cftime/timedelta64 to numeric CF values."""

    if np.issubdtype(da.dtype, np.datetime64) and not is_cftime(da):
        shape = da.shape

        # Flatten so the pandas conversion runs on a 1-D axis, then reshape.
        # pd.to_datetime does not operate element-wise on >1-D arrays.
        flat = np.asarray(da.values).reshape(-1)

        df_unix_sec = (
            (pd.to_datetime(flat) - pd.Timestamp("1970-01-01"))
            .astype("timedelta64[s]")
            .astype("int64")
        )

        da = xr.DataArray(
            pd.to_datetime(df_unix_sec, unit="s", origin="unix")
            .to_numpy()
            .reshape(shape),
            dims=da.dims,
            coords=da.coords,
            attrs=da.attrs,
        )

        num, out_units, out_calendar = encode_cf_datetime(
            da,
            units="seconds since 1970-01-01 00:00:00",
            calendar="proleptic_gregorian",
            dtype="int64",
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units, "calendar": out_calendar})
        encoded.encoding.update({"units": out_units, "calendar": out_calendar})
        return encoded

    elif is_cftime(da):
        num, out_units, out_calendar = encode_cf_datetime(
            da,
            units=da.encoding["units"],
            calendar=da.encoding["calendar"],
            dtype=da.encoding["dtype"],
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units, "calendar": out_calendar})
        encoded.encoding.update({"units": out_units, "calendar": out_calendar})
        return encoded

    elif np.issubdtype(da.dtype, np.timedelta64):
        num, out_units = encode_cf_timedelta(
            da, units=da.encoding["units"], dtype=da.encoding["dtype"]
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units})
        encoded.encoding.update({"units": out_units})
        return encoded

    return da


def createVariable(
    ncf: nc.Dataset,
    da: xr.DataArray,
    varname: str,
    zlib: bool | None = None,
    complevel: int | None = None,
    shuffle: bool | None = None,
    write_values: bool = False,
) -> nc.Variable:

    # we need to use ecoding here
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


def dataset_to_netcdf(
    file: Path,
    data: xr.Dataset,
    unlimited_dim: str | None = None,
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

    dim0 = unlimited_dim if unlimited_dim is not None else next(iter(data.sizes))

    if dim0 not in data.sizes:
        raise ValueError(f"{dim0!r} is not a dimension in data.")

    n_items = data.sizes[dim0]

    if n_items < 1:
        raise ValueError(f"Cannot write an empty dimension: {dim0!r}.")

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    for v in set(list(data.data_vars) + list(data.coords) + [dim0]):
        data[v] = encode_time(data[v])

    # First write defines the file, dimensions, variables, attrs, and encodings.
    # Keep this as a single record.

    data0 = data.isel({dim0: slice(0, 1)})

    enc = {
        v: {"zlib": zlib, "complevel": complevel, "shuffle": shuffle}
        for v in data0.data_vars
    }

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
            description="Writing NetCDF file",
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


def append_to_netcdf(
    file: Path,
    data: xr.Dataset,
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
    """

    if isinstance(data, xr.DataArray):
        ds = data.to_dataset()
    else:
        ds = data

    if dim not in ds.sizes:
        raise ValueError(f"Append dimension {dim!r} not present in the data")

    n_new = ds.sizes[dim]

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
                    write_values=False,
                )

            ncvar = ncf.variables[varname]
            arr = da.transpose(*ncvar.dimensions).values

            if ncvar.dtype != arr.dtype:
                arr = arr.astype(ncvar.dtype)

            index = tuple(
                slice(offset, offset + n_new) if d == dim else slice(None)
                for d in ncvar.dimensions
            )
            ncvar[index] = arr


def dataarray_to_netcdf(
    file: Path,
    da: xr.DataArray,
    format="NETCDF4",
    shuffle: bool | None = None,
    zlib: bool | None = None,
    complevel: int | None = None,
) -> None:
    """Write / append a DataArray to a NetCDF file

    Parameters
    ----------
    file : Path
        Path to a NetCDF4 file opened with read/write access.
    da : xr.DataArray
        DataArray to write. Must have dimensions that already exist in the file.
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
        raise TypeError("da must be an xarray.DataArray")

    if not Path(file).exists():
        raise FileNotFoundError(f"File {file!r} does not exist!")

    with nc.Dataset(file, mode="r+", format=format) as ncf:
        varname = da.name
        if varname is None:
            raise ValueError("DataArray must have a name.")

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
