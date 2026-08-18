from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import xarray as xr

from .parallel import to_netcdf_parallel
from .serial import append, to_netcdf_serial

__all__ = ["append", "dataset_is_empty", "empty_dataset", "to_netcdf"]

_NO_DATA_ATTR = "_climtools_no_data"


def empty_dataset() -> xr.Dataset:
    """Placeholder passed on non-root ranks in place of real data."""
    return xr.Dataset(attrs={_NO_DATA_ATTR: True})


def dataset_is_empty(data: xr.Dataset | xr.DataArray) -> bool:
    return isinstance(data, xr.Dataset) and data.attrs.get(_NO_DATA_ATTR) is True


def to_netcdf(
    data: xr.Dataset | xr.DataArray,
    file: str | PathLike[str],
    unlimited_dim: str | Iterable[str] | None = None,
    partition_dim: str | None = None,
    *,
    parallel: bool = False,
    batch_size: int = 24,
    format: str = "NETCDF4",
    shuffle: bool = True,
    zlib: bool = True,
    complevel: int = 4,
    show_progress: bool = True,
    stdout: Any = None,
    chunks: Mapping[str, Iterable[int]] | None = None,
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
) -> None:
    """Write a Dataset or DataArray to NetCDF.

    Serial output is written incrementally along an unlimited dimension. With
    ``parallel=True``, every MPI rank contributes its local contiguous slab to
    one file through parallel NetCDF-4.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Data to write. In parallel mode, each rank supplies its local slab.
    file : str or os.PathLike
        Output path. An existing file is replaced.
    unlimited_dim : str or iterable of str, optional
        Dimension(s) made unlimited in the NetCDF schema.
    partition_dim : str, optional
        Dimension partitioned across MPI ranks in parallel mode. If omitted,
        the parallel writer infers the partition axis.
    parallel : bool, default False
        Use the MPI-parallel NetCDF-4 writer.
    batch_size : int, default 24
        Number of slices along the unlimited dimension written per serial
        append. Not used in parallel mode.
    format : str, default "NETCDF4"
        NetCDF format. Parallel output supports only ``"NETCDF4"``.
    shuffle : bool, default True
        Apply the HDF5 shuffle filter.
    zlib : bool, default True
        Apply zlib compression.
    complevel : int, default 4
        Compression level, between 1 and 9.
    show_progress : bool, default True
        Display a progress bar while writing serially.
    stdout : file-like, optional
        Stream the serial progress bar is written to. Defaults to
        ``sys.stdout``.
    chunks : mapping of str to iterable of int, optional
        Explicit chunk shape passed to the parallel writer.
    hints : str, optional
        Semicolon-separated MPI-IO hints in key=value format.
    nofill : bool, default True
        Disable NetCDF pre-filling during parallel initialization.
    allow_serial : bool, default False
        Permit execution when running with a single MPI rank.

    Returns
    -------
    None
    """

    if not isinstance(data, (xr.Dataset, xr.DataArray)):
        raise TypeError("data must be an xarray.Dataset or xarray.DataArray")

    from ..core.lib_mpi import mpi

    target_path = Path(file)

    if parallel:
        if mpi.comm.rank == 0:
            if not isinstance(data, (xr.Dataset, xr.DataArray)):
                raise TypeError(
                    "data must be an xarray.Dataset or xarray.DataArray on rank 0"
                )
        else:
            data = empty_dataset()

    if parallel:
        return to_netcdf_parallel(
            data,
            target_path,
            partition_dim=partition_dim,
            deflate=complevel if zlib else None,
            shuffle=shuffle,
            chunks=chunks,
            unlimited_dim=unlimited_dim if unlimited_dim is not None else (),
            hints=hints,
            nofill=nofill,
            allow_serial=allow_serial,
        )
    else:
        return to_netcdf_serial(
            data=data,
            file=target_path,
            unlimited_dim=unlimited_dim,
            batch_size=batch_size,
            format=format,
            shuffle=shuffle,
            zlib=zlib,
            complevel=complevel,
            show_progress=show_progress,
            stdout=stdout,
        )
