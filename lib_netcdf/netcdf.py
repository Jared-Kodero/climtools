from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import xarray as xr

from ..core.xr_meta import get_mpi_meta
from .parallel import to_netcdf_parallel
from .serial import append, to_netcdf_serial

__all__ = ["append", "dataset_is_empty", "empty_dataset", "to_netcdf"]

_NO_DATA_ATTR = "_climtools_no_data"


def empty_dataset() -> xr.Dataset:
    """Return a placeholder Dataset for a non-root MPI rank.

    Returns
    -------
    xarray.Dataset
        Dataset marked as containing no rank-local data.
    """
    return xr.Dataset(attrs={_NO_DATA_ATTR: True})


def dataset_is_empty(data: xr.Dataset | xr.DataArray) -> bool:
    """Return whether an object is a non-root MPI placeholder.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to inspect.

    Returns
    -------
    bool
        True when ``data`` is an MPI placeholder Dataset.
    """
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

    Serial output is written incrementally along an unlimited dimension. In
    parallel mode, an object carrying ``mpi_meta`` is already distributed and
    every rank writes its existing local slab directly. Otherwise rank 0 owns
    the complete object and the parallel writer distributes partitioned data.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Object to write.
    file : str or os.PathLike
        Output path.
    unlimited_dim : str or iterable of str, optional
        Dimension or dimensions made unlimited.
    partition_dim : str, optional
        MPI partition dimension. For an already distributed object this must
        agree with ``mpi_meta["dim"]``.
    parallel : bool, default False
        Use MPI-parallel NetCDF-4 output.
    batch_size : int, default 24
        Number of slices written per serial append.
    format : str, default "NETCDF4"
        NetCDF format for serial output.
    shuffle : bool, default True
        Apply the HDF5 shuffle filter.
    zlib : bool, default True
        Apply zlib compression.
    complevel : int, default 4
        Compression level from 0 through 9.
    show_progress : bool, default True
        Display serial write progress.
    stdout : Any, optional
        Serial progress output stream.
    chunks : mapping of str to iterable of int, optional
        Explicit NetCDF variable chunk shapes.
    hints : str, optional
        Semicolon-separated MPI-IO hints in ``key=value`` form.
    nofill : bool, default True
        Disable NetCDF pre-filling during parallel initialization.
    allow_serial : bool, default False
        Permit the parallel writer with one MPI rank.

    Returns
    -------
    None
    """
    if not isinstance(data, (xr.Dataset, xr.DataArray)):
        raise TypeError("data must be an xarray.Dataset or xarray.DataArray")

    from ..core.lib_mpi import mpi

    target_path = Path(file)

    if parallel:
        mpi_meta = get_mpi_meta(data)
        distributed = mpi_meta is not None

        # Ranks must agree on the write path. If one rank saw valid mpi_meta
        # and another did not, the two paths post different collectives and
        # the writer would block instead of reporting the inconsistency.
        agreed = mpi.comm.allgather(distributed)
        if any(agreed) and not all(agreed):
            disagreeing = [
                rank for rank, state in enumerate(agreed) if state != agreed[0]
            ]
            raise mpi.MPIError(
                "MPI ranks disagree about whether the object is distributed; "
                + f"ranks {disagreeing} differ from rank 0. Parallel NetCDF "
                + "output requires the same distribution state on every rank."
            )

        if distributed:
            distributed_dim = str(mpi_meta["dim"])
            if partition_dim is not None and partition_dim != distributed_dim:
                raise ValueError(
                    f"partition_dim {partition_dim!r} does not match "
                    + f"distributed dimension {distributed_dim!r}."
                )
            partition_dim = distributed_dim
        elif mpi.comm.rank != 0:
            data = empty_dataset()

        to_netcdf_parallel(
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
        return

    to_netcdf_serial(
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
