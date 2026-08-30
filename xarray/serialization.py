"""Serialize distributed xarray objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xarray as xr

from ..mpi.runtime import MPIRuntime, mpi
from ..netcdf import io as netcdf_io
from .core import MPIXarray, unwrap
from .meta import get_mpi_meta
from .ops import _MPIXarrayOps

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from os import PathLike

    from mpi4py.MPI import Intracomm


def to_netcdf(
    data: MPIXarray | xr.Dataset | xr.DataArray,
    file: str | PathLike[str],
    mpi_runtime: MPIRuntime | Intracomm = mpi,
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
    """Write an xarray object to NetCDF.

    Parameters
    ----------
    data : MPIXarray or xarray.Dataset or xarray.DataArray
        Object to write.
    file : str or os.PathLike
        Output path.
    mpi_runtime : MPIRuntime or mpi4py.MPI.Intracomm, optional
        Runtime or communicator used for parallel output.
    unlimited_dim : str or iterable of str, optional
        Unlimited dimension names.
    partition_dim : str, optional
        Dimension used for MPI partitioning.
    parallel : bool, default False
        Use MPI-parallel NetCDF output.
    batch_size : int, default 24
        Number of slices per serial append.
    format : str, default "NETCDF4"
        NetCDF format used for serial output.
    shuffle : bool, default True
        Enable the HDF5 shuffle filter.
    zlib : bool, default True
        Enable zlib compression.
    complevel : int, default 4
        zlib compression level.
    show_progress : bool, default True
        Show serial write progress.
    stdout : Any, optional
        Stream used for progress output.
    chunks : mapping, optional
        Explicit NetCDF variable chunk shapes.
    hints : str, optional
        Semicolon-separated MPI-IO hints.
    nofill : bool, default True
        Disable NetCDF pre-filling for parallel output.
    allow_serial : bool, default False
        Allow the parallel writer to run with one MPI rank.

    Raises
    ------
    ValueError
        If distributed data are passed to the serial writer.
    """
    runtime = (
        mpi_runtime if isinstance(mpi_runtime, MPIRuntime) else MPIRuntime(mpi_runtime)
    )
    unwrapped = unwrap(data)
    if not parallel and get_mpi_meta(unwrapped) is not None:
        raise ValueError(
            "to_netcdf(): data is distributed (carries mpi_meta) but "
            + "parallel=False (the default). Serial NetCDF output is not "
            + "rank-aware and expects the complete object already assembled "
            + "on the calling rank -- writing a distributed object this way "
            + "would silently write only this rank's own local slice as the "
            + "whole file. Pass parallel=True to write a distributed object "
            + "correctly, or gather/replicate it to a single rank first "
            + "(e.g. an MPIXarray reduction that returns a replicated "
            + "result) if serial output is what you actually want."
        )
    prepared = (
        _MPIXarrayOps(runtime).attach_save_chunks(unwrapped) if parallel else unwrapped
    )
    netcdf_io.to_netcdf(
        prepared,
        file,
        runtime,
        unlimited_dim,
        partition_dim,
        parallel=parallel,
        batch_size=batch_size,
        format=format,
        shuffle=shuffle,
        zlib=zlib,
        complevel=complevel,
        show_progress=show_progress,
        stdout=stdout,
        chunks=chunks,
        hints=hints,
        nofill=nofill,
        allow_serial=allow_serial,
    )
