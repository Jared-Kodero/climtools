"""Serial NetCDF-4 output for xarray objects.

The dataset writer defines the file from its first record along the unlimited
dimension and then appends the remainder in batches, so peak memory follows the
batch rather than the whole dataset. The parallel counterpart lives beside this
module in :mod:`climtools.lib_netcdf.parallel`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4
import xarray as xr

from ..core.progress import SerialProgressBar
from .encoding import encode_dataset_time, encode_time, is_time_like
from .parallel import quiet_netcdf4_writes

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike
    from typing import Any, Literal


def resolve_unlimited_dim(
    unlimited_dim: str | Iterable[str] | None,
    sizes: Iterable[str],
) -> str | None:
    """Reduce an unlimited-dimension specification to a single name.

    Parameters
    ----------
    unlimited_dim : str, iterable of str, or None
        Dimension name, or an iterable of names of which the first is used.
    sizes : iterable of str
        Dimension names present in the data, used to report a useful error.

    Returns
    -------
    str or None
        The dimension to extend, or ``None`` when nothing was requested.

    Raises
    ------
    TypeError
        If the specification is neither a string nor an iterable of strings.
    ValueError
        If the requested dimension is absent from the data.

    Notes
    -----
    Serial output extends exactly one dimension. An iterable is accepted
    because the public signature advertises it, but only its first entry is
    meaningful here.
    """
    if unlimited_dim is None:
        return None

    if isinstance(unlimited_dim, str):
        name = unlimited_dim
    else:
        try:
            names = [item for item in unlimited_dim]
        except TypeError as exc:
            raise TypeError(
                "unlimited_dim must be a string or an iterable of strings, "
                + f"got {type(unlimited_dim).__name__}."
            ) from exc
        if not names:
            return None
        if any(not isinstance(item, str) for item in names):
            raise TypeError("Every unlimited_dim entry must be a string.")
        name = names[0]

    known = list(sizes)
    if name not in known:
        raise ValueError(f"{name!r} is not a dimension in data; available: {known}.")
    return name


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
    batch_size : int, default 24
        Slice count processed per file append along the primary unlimited
        dimension.
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

    Notes
    -----
    A DataArray is written as a single-variable Dataset when the target file
    does not yet exist. When it does exist, the array is added to it, or
    overwritten in place if a variable of the same name is already present.
    """
    if isinstance(data, xr.DataArray) and not Path(file).exists():
        if data.name is None:
            raise ValueError("DataArray must have a name to create a new file.")
        data = data.to_dataset()

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


def dataset_to_netcdf(
    file: str | PathLike[str],
    data: xr.Dataset,
    unlimited_dim: str | Iterable[str] | None = None,
    batch_size: int = 1,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
) -> None:
    """Write a Dataset, defining the file once and appending in batches.

    Parameters
    ----------
    file : str or os.PathLike
        Output path. An existing file is replaced.
    data : xarray.Dataset
        Data to write. The caller's object is not modified.
    unlimited_dim : str, iterable of str, or None, optional
        Dimension extended while appending. Defaults to the first dimension.
    batch_size : int, default 1
        Slices appended per write along ``unlimited_dim``.
    format : str, default "NETCDF4"
        NetCDF disk format.
    shuffle : bool, default True
        Enable the HDF5 shuffle filter.
    zlib : bool, default True
        Enable zlib compression.
    complevel : int, default 4
        Compression level, 1 to 9.
    show_progress : bool, default True
        Display a progress bar.
    stdout : file-like, optional
        Stream the progress bar is written to.

    Returns
    -------
    None
    """
    file = Path(file)
    file.unlink(missing_ok=True)

    dim0 = resolve_unlimited_dim(unlimited_dim, data.sizes)
    if dim0 is None:
        dim0 = next(iter(data.sizes))

    n_items = data.sizes[dim0]

    if n_items < 1:
        raise ValueError(f"Cannot write an empty dimension: {dim0!r}.")

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    # Encode on a copy. Assigning encoded variables back into the argument
    # would leave the caller holding an integer time axis after this returns.
    data = encode_dataset_time(data)

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
            file=stdout,
        )
    else:
        data_slices = starts

    for start in data_slices:
        stop = min(start + batch_size, n_items)

        append(
            file,
            data.isel({dim0: slice(start, stop)}),
            dim=dim0,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            encoded_dataset=True,
        )


def dataarray_to_netcdf(
    file: str | PathLike[str],
    da: xr.DataArray,
    format="NETCDF4",
    shuffle: bool | None = None,
    zlib: bool | None = None,
    complevel: int | None = None,
) -> None:
    """Write / append a DataArray to a NetCDF file

    Parameters
    ----------
    file : str or os.PathLike
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

    with netCDF4.Dataset(file, mode="r+", format=format) as ncf:
        varname = da.name
        if varname is None:
            raise ValueError("DataArray must have a name.")

        if is_time_like(da):
            stored = ncf.variables.get(varname)
            units = getattr(stored, "units", None) if stored is not None else None
            calendar = getattr(stored, "calendar", None) if stored is not None else None
            da = encode_time(da, units=units, calendar=calendar)

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

            with quiet_netcdf4_writes():
                ncvar[:] = da.values


def append(
    file: str | PathLike[str],
    data: xr.Dataset,
    dim: str = "time",
    mode: Literal["a", "r+"] = "r+",
    format: str = "NETCDF4",
    shuffle: bool | None = None,
    zlib: bool | None = None,
    complevel: int | None = None,
    encoded_dataset: bool = False,
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
    file : str or os.PathLike
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

    with netCDF4.Dataset(file, mode=mode, format=format) as ncf:
        if dim not in ncf.dimensions:
            raise ValueError(f"Append dimension {dim!r} not found in {file}")

        # Confirm dim is the unlimited axis, and report the actual one on mismatch.
        unlimited = [d for d, o in ncf.dimensions.items() if o.isunlimited()]
        if dim not in unlimited:
            raise ValueError(
                f"Dimension {dim!r} in {file} is not unlimited; unlimited dimension(s): {unlimited or 'none'}"
            )

        offset = ncf.dimensions[dim].size

        encoded_arrays: dict[Any, xr.DataArray] = {}
        if encoded_dataset:
            encoded_arrays = {**ds.coords, **ds.data_vars}
        else:
            for varname, da in {**ds.coords, **ds.data_vars}.items():
                if not is_time_like(da):
                    encoded_arrays[varname] = da
                    continue

                stored = ncf.variables.get(varname)
                units = getattr(stored, "units", None)
                calendar = (
                    getattr(stored, "calendar", None) if stored is not None else None
                )
                encoded_arrays[varname] = encode_time(
                    da,
                    units=units if stored is not None else None,
                    calendar=calendar,
                )

        for varname, da in encoded_arrays.items():
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

            if arr.dtype.kind in "mMO":
                raise TypeError(
                    f"Variable {varname!r} reached the NetCDF layer with "
                    + f"unencoded dtype {arr.dtype}."
                )

            if ncvar.dtype != arr.dtype:
                arr = arr.astype(ncvar.dtype)

            index = tuple(
                slice(offset, offset + n_new) if d == dim else slice(None)
                for d in ncvar.dimensions
            )
            with quiet_netcdf4_writes():
                ncvar[index] = arr


def createVariable(
    ncf: netCDF4.Dataset,
    da: xr.DataArray,
    varname: str,
    zlib: bool | None = None,
    complevel: int | None = None,
    shuffle: bool | None = None,
    write_values: bool = False,
) -> netCDF4.Variable:

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
        with quiet_netcdf4_writes():
            ncvar[:] = da.values

    return ncvar


__all__ = [
    "append",
    "createVariable",
    "dataarray_to_netcdf",
    "dataset_to_netcdf",
    "resolve_unlimited_dim",
    "to_netcdf_serial",
]
