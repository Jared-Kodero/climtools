"""Construct distributed xarray objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from ..mpi.runtime import MPIRuntime
from .core import MPIXarray, unwrap
from .ops import _MPIXarrayOps
from .serialization import to_netcdf as to_netcdf

# ``to_netcdf`` remains available from this module for compatibility.

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence
    from typing import Literal

    from mpi4py.MPI import Intracomm


def mpi_open_dataset(
    filename_or_obj: Any,
    mpi_runtime: MPIRuntime | Intracomm,
    *,
    partition_dim: Hashable | Literal["auto"] = "auto",
    chunks: Any = None,
    log_partitions: bool = True,
    **kwargs: Any,
) -> MPIXarray:
    """Open a Dataset lazily and distribute one dimension across MPI ranks.

    Parameters
    ----------
    filename_or_obj : str, path-like, file-like, or list of these
        Input accepted by ``xarray.open_dataset``/``xarray.open_mfdataset``.
        A wildcard string or a list/tuple triggers multi-file loading.
    mpi_runtime : MPIRuntime or mpi4py.MPI.Intracomm
        Runtime whose communicator the result is bound to.
    partition_dim : Hashable or {"auto"}, optional
        Dimension to distribute. "auto" selects the longest dimension.
    chunks : int, dict, "auto" or None, optional
        Passed unchanged to xarray.
    log_partitions : bool, optional
        Print one aligned table showing which global interval each rank received.
    **kwargs : Any
        Additional arguments passed unchanged to ``xarray.open_dataset``/
        ``xarray.open_mfdataset`` (e.g. ``engine``, ``decode_times``,
        ``concat_dim``, ``combine``, ``preprocess``, ``parallel``).

    Returns
    -------
    MPIXarray
        Lazy rank-local Dataset with ``.meta`` set.
    """
    if not isinstance(mpi_runtime, MPIRuntime):
        mpi_runtime = MPIRuntime(mpi_runtime)
    data = _MPIXarrayOps(mpi_runtime).open_xr_dataset(
        filename_or_obj,
        partition_dim=partition_dim,
        chunks=chunks,
        log_partitions=log_partitions,
        **kwargs,
    )
    return MPIXarray(data, mpi_runtime)


def mpi_create_dataarray(
    runtime: MPIRuntime,
    fill: Callable[[int, int], Any],
    dims: Sequence[Hashable],
    *,
    shape: Sequence[int] | Mapping[Hashable, int] | None = None,
    dim: Hashable | int = 0,
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    name: Hashable | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = False,
) -> MPIXarray:
    """Create a distributed DataArray from a rank-local fill function.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime whose communicator the result is bound to.
    fill : callable
        Function called as ``fill(start, stop)`` for this rank's bounds.
    dims : sequence of Hashable
        Dimension names.
    shape : sequence of int, mapping, or None, optional
        Global dimension sizes. Missing sizes may be inferred from ``coords``.
    dim : Hashable or int, optional
        Dimension or axis to partition.
    dtype : Any, optional
        Data type returned by ``fill``.
    coords : mapping, optional
        Coordinates passed to ``xarray.DataArray``.
    name : Hashable, optional
        DataArray name.
    attrs : mapping, optional
        DataArray attributes.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    MPIXarray
        Lazy rank-local DataArray with ``.meta`` set.
    """
    data = _MPIXarrayOps(runtime).create_dataarray(
        fill,
        dims,
        shape=shape,
        dim=dim,
        dtype=dtype,
        coords=coords,
        name=name,
        attrs=attrs,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, runtime)


def mpi_create_dataset(
    runtime: MPIRuntime,
    data_vars: Mapping[
        Hashable, xr.DataArray | tuple[Sequence[Hashable], Callable[[int, int], Any]]
    ],
    sizes: Mapping[Hashable, int] | None = None,
    *,
    dim: Hashable,
    dtype: Any = np.float64,
    coords: Mapping[Hashable, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    log_partitions: bool = True,
) -> MPIXarray:
    """Create a distributed Dataset from rank-local variables.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime whose communicator the result is bound to.
    data_vars : mapping
        Variables as DataArrays or ``(dims, fill)`` pairs. Partitioned fill
        functions receive ``(start, stop)``; unpartitioned fills take no arguments.
    sizes : mapping, optional
        Global dimension sizes. Missing sizes may be inferred from ``coords``.
    dim : Hashable
        Dimension to partition.
    dtype : Any or mapping, optional
        Default or per-variable fill dtype.
    coords : mapping, optional
        Coordinates passed to ``xarray.Dataset``.
    attrs : mapping, optional
        Dataset attributes.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    MPIXarray
        Lazy rank-local Dataset with ``.meta`` set.
    """
    data = _MPIXarrayOps(runtime).create_dataset(
        data_vars,
        sizes,
        dim=dim,
        dtype=dtype,
        coords=coords,
        attrs=attrs,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, runtime)


def mpi_partition_data(
    value: MPIXarray | xr.Dataset | xr.DataArray | None,
    runtime: MPIRuntime,
    dim: Hashable | Literal["auto"] = "auto",
    *,
    root: int = 0,
    chunk_info: Mapping[str, int] | None = None,
    log_partitions: bool = False,
) -> MPIXarray:
    """Partition a root-owned xarray object across MPI ranks.

    Parameters
    ----------
    value : MPIXarray, xarray.Dataset, xarray.DataArray, or None
        Complete object on ``root``; non-root ranks must pass None.
    runtime : MPIRuntime
        Runtime whose communicator the result is bound to.
    dim : Hashable or {"auto"}, optional
        Partition dimension. "auto" selects the largest dimension.
    root : int, optional
        Rank that owns ``value``.
    chunk_info : mapping of str to int, optional
        Effective chunk-size hints.
    log_partitions : bool, optional
        Log the resulting rank layout.

    Returns
    -------
    MPIXarray
        Rank-local slice with ``.meta`` set.
    """
    data = _MPIXarrayOps(runtime).partition(
        unwrap(value),
        dim,
        root=root,
        chunk_info=chunk_info,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, runtime)
