"""Module-level constructors for :class:`~.core.MPIXarray`.

No :class:`~.core.MPIXarray` exists yet for these to be a method on: each
builds data from scratch (or from a root-owned/external value) and returns
the first wrapped :class:`~.core.MPIXarray`. All take the runtime explicitly
rather than relying on any ambient/bound runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import xarray as xr

from ..mpi.runtime import MPIRuntime, mpi
from ..netcdf import io as netcdf_io
from .core import MPIXarray, unwrap
from .elementwise import Elementwise
from .groupby import Groupby
from .indexing import Indexing
from .io import IO
from .meta import get_mpi_meta
from .operator import Arithmetic
from .reductions import Reduction
from .statistics import Statistics

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
    from os import PathLike
    from typing import Literal

    from mpi4py.MPI import Intracomm


# Make `raw_dataset + mpixarray` (and -/*//etc.) defer correctly: xarray's
# own Dataset.__add__/DataArray.__add__ only return NotImplemented (which
# is what lets Python fall through to MPIXarray's own __radd__/etc.) for a
# closed list of xarray-internal types. MPIXarray isn't on that list, so
# without this it fails inside xarray's own arithmetic instead of ever
# reaching MPIXarray's code. `mpixarray + raw_dataset` (operand order
# reversed) already works without this -- Python always tries the left
# operand's __add__ first.
_da_binary_op = xr.DataArray._binary_op
_ds_binary_op = xr.Dataset._binary_op


def _da_binary_op_patched(
    self: xr.DataArray, other: Any, f: Any, reflexive: bool = False
) -> Any:
    if isinstance(other, MPIXarray):
        return NotImplemented
    return _da_binary_op(self, other, f, reflexive)


def _ds_binary_op_patched(
    self: xr.Dataset, other: Any, f: Any, reflexive: bool = False, join: Any = None
) -> Any:
    if isinstance(other, MPIXarray):
        return NotImplemented
    return _ds_binary_op(self, other, f, reflexive, join)


xr.DataArray._binary_op = _da_binary_op_patched
xr.Dataset._binary_op = _ds_binary_op_patched


class _MPIXarrayOps(
    IO, Indexing, Reduction, Statistics, Groupby, Arithmetic, Elementwise
):
    """Internal MPI-aware xarray engine bound to an MPI runtime.

    Composes, by concern, the same seven mixins :class:`~.core.MPIXarray`
    delegates to: :class:`~.io.IO`, :class:`~.indexing.Indexing`,
    :class:`~.reductions.Reduction`, :class:`~.statistics.Statistics`,
    :class:`~.groupby.Groupby`, :class:`~.operator.Arithmetic`,
    :class:`~.elementwise.Elementwise`.

    Parameters
    ----------
    runtime : MPIRuntime
        Runtime whose communicator is used for distributed operations.
    """

    def __init__(self, runtime: MPIRuntime) -> None:
        self._runtime = runtime


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
    data = _MPIXarrayOps(runtime).distribute(
        unwrap(value),
        dim,
        root=root,
        chunk_info=chunk_info,
        log_partitions=log_partitions,
    )
    return MPIXarray(data, runtime)


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
    """Write a Dataset or DataArray to NetCDF.

    Serial output (``parallel=False``, the default) is written
    incrementally along an unlimited dimension by whichever rank calls
    this; it is not rank-aware and expects ``data`` to already be the
    complete object -- typically called by rank 0 alone with a
    non-distributed (replicated, ``.meta`` is None) object. In parallel
    mode, an object carrying ``mpi_meta`` is already distributed and every
    rank writes its existing local slab directly; otherwise rank 0 owns
    the complete object and the parallel writer distributes it.

    A distributed ``data`` written with ``parallel=True`` first has
    write-time chunk metadata attached internally (the engine's
    ``attach_save_chunks`` step -- computed on rank 0 from distribution
    metadata and broadcast, no data materialized; a no-op for a
    non-distributed object). This is no longer a separate method to call
    beforehand: it is purely an implementation detail of writing, not
    something callers need to reason about.

    Parameters
    ----------
    data : MPIXarray, xarray.Dataset, or xarray.DataArray
        Object to write. An :class:`MPIXarray` is unwrapped so
        :func:`~.netcdf.io.to_netcdf` sees its ``mpi_meta`` in ``.attrs``
        when writing with ``parallel=True``.
    file : str or os.PathLike
        Output path.
    mpi_runtime : MPIRuntime, optional
        Runtime whose communicator backs a parallel write. Defaults to the
        package-wide :data:`~..mpi.runtime.mpi` instance.
    unlimited_dim : str or iterable of str, optional
        Dimension or dimensions made unlimited.
    partition_dim : str, optional
        MPI partition dimension. For an already distributed object this must
        agree with ``mpi_meta["dim"]``.
    parallel : bool, default False
        Use MPI-parallel NetCDF-4 output. Required if ``data`` is
        distributed -- see ``Raises`` below.
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

    Raises
    ------
    ValueError
        If ``data`` is distributed (has ``mpi_meta``) and ``parallel`` is
        False. The serial writer is not rank-aware: it would otherwise
        write only the calling rank's own local slice as if it were the
        complete file, silently producing an incomplete, wrong result
        rather than raising.
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
