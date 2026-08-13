from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
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
            units=da.encoding.get("units"),
            calendar=da.encoding.get("calendar"),
            dtype=da.encoding.get("dtype"),
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units, "calendar": out_calendar})
        encoded.encoding.update({"units": out_units, "calendar": out_calendar})
        return encoded

    elif np.issubdtype(da.dtype, np.timedelta64):
        num, out_units = encode_cf_timedelta(
            da, units=da.encoding.get("units"), dtype=da.encoding.get("dtype")
        )
        encoded = da.copy(data=num)
        encoded.attrs.update({"units": out_units})
        encoded.encoding.update({"units": out_units})
        return encoded

    return da


def to_netcdf_parallel(
    data: xr.Dataset | xr.DataArray,
    path: str | PathLike[str],
    partition_dim: str | None = None,
    deflate: int | None = None,
    shuffle: bool = True,
    chunks: Mapping[str, Iterable[int]] | None = None,
    unlimited_dim: str | Iterable[str] = (),
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
) -> str:
    """Write a distributed xarray dataset to one NetCDF-4 file.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Dataset or DataArray slab owned by the current rank.
    path : str or os.PathLike
        Output path visible to every rank.
    dim : str or None, optional
        Partitioned dimension. The writer infers it when omitted.
    deflate : int or None, optional
        Deflate compression level from 0 to 9.
    shuffle : bool, default True
        Enable the HDF5 shuffle filter.
    chunks : mapping of str to iterable of int, optional
        Explicit chunk shape for selected variables.
    unlimited_dims : str or iterable of str, default ()
        Dimensions to define as unlimited record dimensions.
    hints : str or None, optional
        Semicolon-separated MPI-IO hints in key=value form.
    nofill : bool, default True
        Disable NetCDF pre-filling when True.
    allow_serial : bool, default False
        Permit execution with a one-rank MPI world.

    Returns
    -------
    str
        Output path after the collective write completes.
    """
    if isinstance(data, xr.DataArray):
        if data.name is None:
            raise ValueError("DataArray must have a name for parallel output.")
        data: xr.Dataset = data.to_dataset()

    if not isinstance(data, xr.Dataset):
        raise TypeError("data must be an xarray.Dataset or xarray.DataArray")

    if isinstance(unlimited_dim, str):
        unlimited_dim: tuple[str, ...] = (unlimited_dim,)
    elif unlimited_dim:
        unlimited_dim = tuple(unlimited_dim)
    else:
        unlimited_dim = ()

    from .lib_mpi import to_netcdf as mpi_to_netcdf

    for name in data.variables:
        data[name] = encode_time(data[name])

    return mpi_to_netcdf(
        data,
        path,
        partition_dim=partition_dim,
        deflate=deflate,
        shuffle=shuffle,
        chunks=chunks,
        unlimited_dim=unlimited_dim,
        hints=hints,
        nofill=nofill,
        allow_serial=allow_serial,
    )


def to_netcdf_serial(
    data: xr.Dataset | xr.DataArray,
    file: str | PathLike[str],
    unlimited_dim: str | Iterable[str] | None = None,
    *,
    batch_size: int = 24,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
) -> None:
    """Write a Dataset or DataArray serially to NetCDF.

    Increments are appended along the specified unlimited dimension in
    discrete batches to manage memory overhead during serial output.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Data object to be written.
    file : str or os.PathLike
        Output file path.
    unlimited_dim : str or iterable of str, optional
        Dimension(s) designated as unlimited in the NetCDF file structure.
    batch_size : int, default 1
        Slice count processed per file append along the primary unlimited dimension.
    format : str, default "NETCDF4"
        NetCDF underlying disk format.
    shuffle : bool, default True
        Enable HDF5 byte-shuffle filter.
    zlib : bool, default True
        Enable zlib deflate compression filter.
    complevel : int, default 4
        Zlib deflate compression level (1-9).
    show_progress : bool, default True
        Print incremental progress to output stream.
    stdout : file-like, optional
        Destination stream for progress updates; defaults to sys.stdout.

    Returns
    -------
    None
    """

    if isinstance(data, xr.Dataset):
        dataset_to_netcdf(
            file=file,
            data=data,
            unlimited_dim=unlimited_dim,
            batch_size=batch_size,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            show_progress=show_progress,
            stdout=stdout,
        )
        return

    dataarray_to_netcdf(
        file=file,
        da=data,
        format=format,
        shuffle=shuffle,
        zlib=zlib,
        complevel=complevel,
    )
    return


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

    for name in data.variables:
        data[name] = encode_time(data[name])

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
