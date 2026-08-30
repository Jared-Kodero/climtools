"""Parallel NetCDF-4 output for MPI-distributed xarray data."""

from __future__ import annotations

import contextlib
import math
import sys
import traceback
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask
import netCDF4
import numpy as np
from mpi4py import MPI

import xarray as xr

from ..core.progress import SerialProgressBar
from ..mpi.diagnostics import MPIError
from .chunks import get_chunk_bounds, get_chunks, get_partition_chunk_size
from .constructors import MPIXarrayOps, mpi_partition_data
from .encoding import encode_dataset_time, encode_time, is_time_like
from .meta import MPI_META, _format_bytes, get_mpi_meta

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike
    from typing import Any, Literal

    from mpi4py.MPI import Intracomm

    from ..mpi.runtime import MPIRuntime


class NetCDFWriteError(MPIError):
    """Parallel NetCDF write failure."""


# A single HDF5 chunk's byte size must stay strictly under 2**32 bytes (4
# GiB) -- confirmed directly against this build with a binary search: a
# (16383, 256, 256) float32 chunk (3.9998 GiB) succeeds, (16384, 256, 256)
# (4.0000 GiB) raises "NetCDF: Bad chunk sizes." from the netCDF-C library.
# Half that hard limit is used as the working target below, leaving headroom
# for HDF5/filter (zlib, shuffle) bookkeeping overhead per chunk rather than
# skimming the exact boundary.
def set_attrs(target: Any, attrs: Mapping[str, Any]) -> None:
    """Set serializable NetCDF attributes.

    Parameters
    ----------
    target : Any
        NetCDF object receiving attributes.
    attrs : Mapping[str, Any]
        Attributes to write.
    """
    for key, value in attrs.items():
        if key not in ("_FillValue", MPI_META) and value is not None:
            target.setncattr(str(key), value)


def _normalise_variable(source: xr.DataArray) -> tuple[np.ndarray, str | np.dtype[Any]]:
    """Normalize an xarray variable for NetCDF output.

    Parameters
    ----------
    source : xarray.DataArray
        Variable to normalize.

    Returns
    -------
    numpy.ndarray
        Contiguous normalized values.
    str or numpy.dtype
        NetCDF-compatible data type.
    """
    variable = encode_time(source) if is_time_like(source) else source
    values = np.asarray(variable.values)

    if values.dtype.kind in ("U", "S", "O"):
        values = np.asarray(
            [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in values.ravel()
            ],
            dtype=object,
        ).reshape(values.shape)
        return values, "str"

    if values.dtype.kind == "b":
        values = values.astype(np.int8)
    if values.dtype.byteorder not in ("=", "|"):
        values = values.astype(values.dtype.newbyteorder("="))
    return np.ascontiguousarray(values), values.dtype


@contextlib.contextmanager
def quiet_netcdf4_writes():
    """Suppress the netCDF4 NumPy shape-assignment warning.

    Yields
    ------
    None
        Context with the warning filtered.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated",
            category=DeprecationWarning,
        )
        yield


def preextend_unlimited(
    nc: netCDF4.Dataset,
    schema: Mapping[str, Any],
    name: str,
    ncvar: netCDF4.Variable,
) -> None:
    """Pre-extend unlimited dimensions before parallel writes.

    Parameters
    ----------
    nc : netCDF4.Dataset
        Open NetCDF dataset.
    schema : Mapping[str, Any]
        Output schema.
    name : str
        Variable name.
    ncvar : netCDF4.Variable
        Variable to pre-extend.
    """
    dims = ncvar.dimensions
    unlimited = set(schema["unlimited_dim"])
    if not any(dim in unlimited for dim in dims):
        return

    index = tuple(
        int(schema["sizes"][dim]) - 1 if dim in unlimited else 0 for dim in dims
    )
    if any(position < 0 for position in index):
        return

    fill = ncvar.getncattr("_FillValue") if "_FillValue" in ncvar.ncattrs() else 0
    with quiet_netcdf4_writes():
        ncvar[index] = fill


def create_file(
    path: str,
    schema: Mapping[str, Any],
    root_data: Mapping[str, Mapping[str, Any]],
) -> None:
    """Create NetCDF metadata and write nonpartitioned data.

    Parameters
    ----------
    path : str
        Output path.
    schema : Mapping[str, Any]
        Output schema.
    root_data : Mapping[str, Mapping[str, Any]]
        Rank-0 variable metadata and data.
    """
    partition_dim = schema["partition_dim"]
    unlimited = set(schema["unlimited_dim"])
    chunks = schema["chunks"]
    prewritten = set(schema.get("prewritten", ()))

    with netCDF4.Dataset(path, mode="w", format="NETCDF4") as nc:
        if schema["nofill"]:
            nc.set_fill_off()

        for dim, length in schema["sizes"].items():
            nc.createDimension(dim, None if dim in unlimited else length)

        for name, variable in root_data.items():
            dims = variable["dims"]
            dtype = variable["dtype"]
            attrs = variable["attrs"]
            kwargs: dict[str, Any] = {}

            if "_FillValue" in attrs:
                kwargs["fill_value"] = attrs["_FillValue"]
            if name in chunks and dims and dtype != "str":
                kwargs["chunksizes"] = chunks[name]
            if (
                schema["deflate"] is not None
                and dims
                and dtype != "str"
                and not variable["coord"]
            ):
                kwargs["zlib"] = True
                kwargs["complevel"] = schema["deflate"]
                kwargs["shuffle"] = schema["shuffle"]

            ncvar = nc.createVariable(name, dtype, dims, **kwargs)
            set_attrs(ncvar, attrs)

            partitioned = (
                partition_dim is not None
                and partition_dim in dims
                and name not in prewritten
            )
            if not partitioned:
                with quiet_netcdf4_writes():
                    ncvar[...] = variable["data"]
            elif dtype != "str":
                preextend_unlimited(nc, schema, name, ncvar)

        set_attrs(nc, schema["attrs"])


def writer_comm(mpi_runtime: MPIRuntime, has_data: bool) -> MPI.Comm:
    """Create the communicator used for collective writes.

    Parameters
    ----------
    mpi_runtime : MPIRuntime
        MPI runtime.
    has_data : bool
        Whether the rank owns output data.

    Returns
    -------
    mpi4py.MPI.Comm
        Writer communicator or ``MPI.COMM_NULL``.
    """
    if mpi_runtime.comm.size == 1:
        return mpi_runtime.comm if has_data else MPI.COMM_NULL
    return mpi_runtime.comm.Split(
        1 if has_data else MPI.UNDEFINED, mpi_runtime.comm.rank
    )


def free_writer_comm(mpi_runtime: MPIRuntime, comm: MPI.Comm) -> None:
    """Free a writer communicator when required.

    Parameters
    ----------
    mpi_runtime : MPIRuntime
        MPI runtime.
    comm : mpi4py.MPI.Comm
        Writer communicator.
    """
    if comm != MPI.COMM_NULL and comm != mpi_runtime.comm:
        comm.Free()


def open_in_parallel(
    path: str,
    schema: Mapping[str, Any],
    comm: MPI.Comm,
) -> netCDF4.Dataset:
    """Open a NetCDF file for MPI I/O.

    Parameters
    ----------
    path : str
        NetCDF path.
    schema : Mapping[str, Any]
        Output schema.
    comm : mpi4py.MPI.Comm
        Writer communicator.

    Returns
    -------
    netCDF4.Dataset
        Open dataset.
    """
    info: MPI.Info | None = None
    if comm.size > 1:
        info = MPI.Info.Create()
        try:
            for item in (schema["hints"] or "").split(";"):
                if not item.strip():
                    continue
                key, separator, value = item.partition("=")
                if not separator or not key.strip():
                    raise ValueError(
                        f"Invalid MPI-IO hint: {item!r}; expected key=value."
                    )
                info.Set(key.strip(), value.strip())
            return netCDF4.Dataset(
                path,
                mode="r+",
                parallel=True,
                comm=comm,
                info=info,
            )
        finally:
            info.Free()
    return netCDF4.Dataset(path, mode="r+")


def write_distributed(
    mpi_runtime: MPIRuntime,
    path: str,
    schema: Mapping[str, Any],
    ds: xr.Dataset,
    meta: Mapping[str, Any],
) -> None:
    """Write rank-local slabs from distributed data.

    Parameters
    ----------
    mpi_runtime : MPIRuntime
        MPI runtime.
    path : str
        NetCDF path.
    schema : Mapping[str, Any]
        Output schema.
    ds : xarray.Dataset
        Rank-local dataset.
    meta : Mapping[str, Any]
        MPI partition metadata.
    """

    partition_dim = str(meta["dim"])
    start = int(meta["start"])
    stop = int(meta["stop"])
    prewritten = set(schema.get("prewritten", ()))

    comm = writer_comm(mpi_runtime, stop > start)
    nc: netCDF4.Dataset | None = None
    try:
        if comm == MPI.COMM_NULL:
            return
        nc = open_in_parallel(path, schema, comm)
        for name, spec in schema["variables"].items():
            dims = tuple(spec["dims"])
            if partition_dim not in dims or name in prewritten:
                continue
            if spec["dtype"] == "str":
                raise NetCDFWriteError(
                    f"Partitioned string variable {name!r} is unsupported because "
                    + "netCDF4 parallel I/O cannot write VLEN data types."
                )

            values, _ = _normalise_variable(ds[name])
            ncvar = nc.variables[name]
            if comm.size > 1:
                ncvar.set_collective(True)
            index = tuple(
                slice(start, stop) if dim == partition_dim else slice(None)
                for dim in dims
            )
            with quiet_netcdf4_writes():
                ncvar[index] = values
    finally:
        if nc is not None:
            nc.close()
        free_writer_comm(mpi_runtime, comm)
        # Every rank reaches this barrier -- including the COMM_NULL/no-data
        # ranks that returned above -- so nothing downstream (a different
        # communicator, a non-parallel reopen for validation, ...) can run
        # ahead of a writer rank that is still inside nc.close().
        mpi_runtime.comm.Barrier()


def write_partitioned(
    mpi_runtime: MPIRuntime,
    path: str,
    schema: Mapping[str, Any],
    source_ds: xr.Dataset | None,
) -> None:
    """Scatter root-owned data and write rank-local slabs.

    Parameters
    ----------
    mpi_runtime : MPIRuntime
        MPI runtime.
    path : str
        NetCDF path.
    schema : Mapping[str, Any]
        Output schema.
    source_ds : xarray.Dataset or None
        Complete dataset on rank 0.
    """
    partition_dim = schema["partition_dim"]
    if partition_dim is None:
        return
    prewritten = set(schema.get("prewritten", ()))

    length = int(schema["sizes"][partition_dim])
    # Every rank's slab boundary must land on a multiple of the same chunk
    # length get_chunks() gave this variable's partition axis (see
    # get_partition_chunk_size), or a rank's write straddles two HDF5
    # chunks -- see get_chunks()'s docstring for what that corrupts. This is
    # the same alignment to_netcdf_parallel's already-distributed path gets
    # for free from mpi_meta's own start/stop; write_partitioned computes it
    # fresh here because rank 0 owns the whole array before this call and no
    # partition boundaries exist yet.
    chunk_size = int(
        schema.get("partition_chunk_size")
        or max(1, math.ceil(length / mpi_runtime.comm.size))
    )
    bounds = [
        get_chunk_bounds(length, chunk_size, rank, mpi_runtime.comm.size)
        for rank in range(mpi_runtime.comm.size)
    ]
    counts = np.array([stop - start for start, stop in bounds], dtype=np.int64)
    start, stop = bounds[mpi_runtime.comm.rank]

    comm = writer_comm(mpi_runtime, stop > start)
    nc: netCDF4.Dataset | None = None
    try:
        if comm != MPI.COMM_NULL:
            nc = open_in_parallel(path, schema, comm)

        for name, spec in schema["variables"].items():
            dims = tuple(spec["dims"])
            if partition_dim not in dims or name in prewritten:
                continue
            if spec["dtype"] == "str":
                raise NetCDFWriteError(
                    f"Partitioned string variable {name!r} is unsupported because "
                    + "netCDF4 parallel I/O cannot write VLEN data types."
                )

            axis = dims.index(partition_dim)
            shape = tuple(int(value) for value in spec["shape"])
            moved_shape = (shape[axis], *shape[:axis], *shape[axis + 1 :])
            local_shape = (int(counts[mpi_runtime.comm.rank]), *moved_shape[1:])
            dtype = np.dtype(spec["dtype"])

            send = None
            if mpi_runtime.comm.rank == 0:
                if source_ds is None:
                    raise AssertionError("Rank 0 source Dataset is missing.")
                variable = source_ds[name]
                variable = encode_time(variable) if is_time_like(variable) else variable
                send = np.ascontiguousarray(
                    np.moveaxis(np.asarray(variable.values), axis, 0),
                    dtype=dtype,
                )

            local = mpi_runtime.scatterv(send, counts, local_shape, dtype)
            # `send = None` (not `del send` + `gc.collect()`) is deliberate
            # and sufficient: CPython reclaims a non-cyclic object the
            # moment its refcount hits zero, and mpi_runtime.scatterv's own Scatterv
            # call is synchronous and keeps no reference to `send` after it
            # returns, so dropping this one binding is enough. An explicit
            # gc.collect() here would only add a full generational GC pass
            # with no correctness or memory benefit, since there is no
            # reference cycle to break. What this does NOT release is
            # source_ds's own underlying array for `name`: `.values` is
            # typically a view into source_ds's own storage, not a copy, so
            # source_ds -- which the caller keeps alive for this whole
            # function's duration -- still holds it regardless. What this
            # loop actually avoids is the *old* design's redundant extra
            # copy of every variable sitting in root_data simultaneously
            # (see to_netcdf_parallel's schema-construction step); it does
            # not, and cannot, undo the baseline cost of an eager source
            # already being fully resident before this function is called.
            send = None
            if nc is None:
                continue

            local = np.moveaxis(local, 0, axis)
            ncvar = nc.variables[name]
            if comm.size > 1:
                ncvar.set_collective(True)
            index = tuple(
                slice(start, stop) if dim == partition_dim else slice(None)
                for dim in dims
            )
            with quiet_netcdf4_writes():
                ncvar[index] = local
    finally:
        if nc is not None:
            nc.close()
        free_writer_comm(mpi_runtime, comm)
        # See the matching comment in write_distributed: guarantees every
        # rank waits for every writer rank's close() before anything
        # downstream reopens the file.
        mpi_runtime.comm.Barrier()


def to_netcdf_parallel(
    mpi_runtime: MPIRuntime | Intracomm,
    data: xr.Dataset | xr.DataArray | None,
    path: str | PathLike[str],
    partition_dim: str | None = None,
    deflate: int | None = None,
    shuffle: bool = True,
    chunks: Mapping[str, Iterable[int]] | None = None,
    unlimited_dim: str | Iterable[str] | None = (),
    hints: str | None = None,
    nofill: bool = True,
    allow_serial: bool = False,
) -> str:
    """Write an xarray object with MPI collective NetCDF I/O.

    Parameters
    ----------
    mpi_runtime : MPIRuntime or mpi4py.MPI.Intracomm
        MPI runtime used for communication.
    data : xarray.Dataset or xarray.DataArray or None
        Distributed local data or complete rank-0 data.
    path : str or os.PathLike
        Output path.
    partition_dim : str, optional
        Partition dimension.
    deflate : int, optional
        Compression level from 0 to 9.
    shuffle : bool, default True
        Apply the HDF5 shuffle filter.
    chunks : Mapping[str, Iterable[int]], optional
        Explicit variable chunk shapes.
    unlimited_dim : str or Iterable[str], optional
        Unlimited dimensions.
    hints : str, optional
        Semicolon-separated MPI-IO hints.
    nofill : bool, default True
        Disable NetCDF pre-filling.
    allow_serial : bool, default False
        Permit one-rank execution.

    Returns
    -------
    str
        Absolute output path.
    """
    if mpi_runtime.comm.size == 1 and not allow_serial:
        raise NetCDFWriteError(
            "MPI_COMM_WORLD contains one process. Launch with mpirun/mpiexec/srun "
            + "or pass allow_serial=True."
        )
    if mpi_runtime.comm.size > 1 and not getattr(
        netCDF4, "__has_parallel4_support__", False
    ):
        raise NetCDFWriteError(
            "netCDF4-python is not linked with parallel NetCDF-4/HDF5 support."
        )

    # Rank-local validation is collected rather than raised. Only rank 0 holds
    # real data on the scatter path, so raising here would abort one rank while
    # every other rank waited in the following collectives.
    local_ds: xr.Dataset | None = None
    error: BaseException | None = None
    try:
        if isinstance(data, xr.DataArray):
            if data.name is None:
                raise ValueError("DataArray must have a name for NetCDF output.")
            # to_dataset() moves the array's attributes onto the variable and
            # leaves Dataset.attrs empty. get_mpi_meta falls back to
            # variable-level metadata for exactly this reason, so the
            # distributed path is still selected here.
            local_ds = data.to_dataset()
        elif isinstance(data, xr.Dataset):
            local_ds = data
        elif data is not None:
            raise TypeError("data must be an xarray Dataset, DataArray, or None.")
    except BaseException as exc:
        error = exc
    mpi_runtime.raise_if_error(error, "parallel NetCDF input validation")

    local_meta = get_mpi_meta(local_ds) if local_ds is not None else None
    distributed = local_meta is not None

    # The distributed and scatter paths post different collectives, so every
    # rank must take the same one. Disagreement is reported instead of hanging.
    agreed = mpi_runtime.comm.allgather(distributed)
    if any(agreed) and not all(agreed):
        disagreeing = [rank for rank, state in enumerate(agreed) if state != agreed[0]]
        raise NetCDFWriteError(
            "MPI ranks disagree about whether the object carries mpi_meta; "
            + f"ranks {disagreeing} differ from rank 0."
        )

    root_data: dict[str, dict[str, Any]] | None = None
    schema: dict[str, Any] | None = None
    output_path: str | None = None
    error = None

    if not distributed:
        # A dask-backed rank-0-source input can be distributed lazily
        # instead of forced through np.asarray(...values) below, which
        # would materialize the complete array on rank 0 regardless of
        # size. An eager (already in-memory) input gains nothing from that
        # laziness -- the array is already fully resident wherever it
        # started -- and pays real overhead switching to point-to-point
        # pickling instead of scatterv's zero-copy buffer transfer, so it
        # is left on the original path entirely unchanged below. Every
        # rank must agree on this decision before either branch runs, for
        # the same collective-mismatch reason as the agreement check above:
        # only rank 0 can inspect the real object, so its answer is
        # broadcast rather than each rank guessing from an empty one.
        is_dask_backed = False
        try:
            if mpi_runtime.comm.rank == 0 and local_ds is not None:
                is_dask_backed = any(
                    dask.is_dask_collection(variable.data)
                    for variable in local_ds.variables.values()
                )
        except BaseException as exc:
            error = exc
        mpi_runtime.raise_if_error(error, "parallel NetCDF dask-backed detection")
        is_dask_backed = mpi_runtime.comm.bcast(is_dask_backed, root=0)

        if is_dask_backed:
            local_ds = mpi_partition_data(
                local_ds if mpi_runtime.comm.rank == 0 else None,
                mpi_runtime,
                dim=partition_dim if partition_dim is not None else "auto",
                root=0,
            )
            local_meta = get_mpi_meta(local_ds)
            distributed = local_meta is not None

    if distributed:
        if local_ds is None or local_meta is None:
            raise AssertionError("Distributed data and metadata are missing.")

        distributed_dim = str(local_meta["dim"])
        if partition_dim is not None and partition_dim != distributed_dim:
            error = ValueError(
                f"partition_dim {partition_dim!r} does not match "
                + f"distributed dimension {distributed_dim!r}."
            )
        mpi_runtime.raise_if_error(error, "parallel NetCDF partition dimension")
        partition_dim = distributed_dim

        # Plan save_chunks collectively before any rank-0-only work below,
        # so the schema-construction branch can align the partition-
        # dimension chunk to distribution_chunks and cap it to the HDF5
        # 4 GiB chunk limit -- rank 0's own local slice (used below to read
        # dtype/dims) does not know the array's true global shape, so a
        # dask.chunk("auto") call against it alone would size every other
        # dimension's chunk from 1/mpi_size of the real data volume.
        # attach_save_chunks fixes that using mpi_meta alone (see its
        # docstring), and is skipped when the caller already gave explicit
        # chunks to honor below, since every rank sees the same `chunks`
        # argument and can decide this identically without communication.
        if chunks is None:
            MPIXarrayOps(mpi_runtime).attach_save_chunks(local_ds)  # this is a bug
            local_meta = get_mpi_meta(local_ds)
            if local_meta is None:
                raise AssertionError(
                    "attach_save_chunks unexpectedly cleared mpi_meta."
                )

        # Coordinates that carry partition_dim (a distributed "time" axis is
        # the common case) are genuinely different per rank, unlike lat/lon/
        # plev, so rank 0 cannot just read them off its own local_ds. Gather
        # every rank's piece, verify the pieces tile the global interval with
        # no gap or overlap, and concatenate in start order. This also means
        # such a coordinate is CF-encoded exactly once (one units/calendar
        # pair for the whole axis) instead of once per rank, and it can then
        # be written serially in create_file alongside lat/lon/plev rather
        # than through the collective phase -- sidestepping netCDF4's
        # default chunk size for that variable entirely, not just working
        # around it with an explicit override.
        start = int(local_meta["start"])
        stop = int(local_meta["stop"])
        global_size = int(local_meta["global_size"])
        prewritten_coords: dict[str, xr.DataArray] = {}
        try:
            coord_names = [
                name
                for name, coord in local_ds.coords.items()
                if partition_dim in coord.dims
            ]
        except BaseException as exc:
            coord_names = []
            error = exc
        mpi_runtime.raise_if_error(
            error,
            "parallel NetCDF coordinate discovery",
            signature=tuple(coord_names),
        )

        for coord_name in coord_names:
            coordinate = local_ds[coord_name]
            axis = coordinate.get_axis_num(partition_dim)
            local_values = np.asarray(coordinate.values)
            pieces = mpi_runtime.comm.gather((start, stop, local_values), root=0)
            if mpi_runtime.comm.rank == 0:
                try:
                    ordered = sorted(pieces, key=lambda item: item[0])
                    cursor = 0
                    for piece_start, piece_stop, values in ordered:
                        if piece_start != cursor:
                            raise NetCDFWriteError(
                                f"Partitioned coordinate {coord_name!r} has a "
                                + f"gap or overlap: expected start {cursor}, "
                                + f"got {piece_start}."
                            )
                        expected = piece_stop - piece_start
                        if values.shape[axis] != expected:
                            raise NetCDFWriteError(
                                f"Partitioned coordinate {coord_name!r} rank piece "
                                + f"[{piece_start}:{piece_stop}) has length "
                                + f"{values.shape[axis]} along {partition_dim!r}; "
                                + f"expected {expected}."
                            )
                        cursor = piece_stop
                    if cursor != global_size:
                        raise NetCDFWriteError(
                            f"Partitioned coordinate {coord_name!r} covers "
                            + f"{cursor} of {global_size} global elements."
                        )
                    assembled = np.concatenate(
                        [values for _, _, values in ordered],
                        axis=axis,
                    )
                    rebuilt = xr.DataArray(
                        assembled,
                        dims=coordinate.dims,
                        name=coordinate.name,
                        attrs=dict(coordinate.attrs),
                    )
                    rebuilt.encoding = dict(coordinate.encoding)
                    prewritten_coords[coord_name] = rebuilt
                except BaseException as exc:
                    error = exc
            mpi_runtime.raise_if_error(
                error, f"parallel NetCDF coordinate gather ({coord_name})"
            )

        # No data gather/scatter. Rank 0 only constructs the schema from its
        # local metadata and mpi_meta's global partition length.
        if mpi_runtime.comm.rank == 0:
            try:
                ds = local_ds
                if deflate is not None and not 0 <= int(deflate) <= 9:
                    raise ValueError(
                        "deflate must be None or an integer from 0 through 9."
                    )
                if unlimited_dim is None:
                    unlimited = ()
                elif isinstance(unlimited_dim, str):
                    unlimited = (unlimited_dim,)
                else:
                    unlimited = tuple(unlimited_dim)

                sizes = dict(ds.sizes)
                sizes[partition_dim] = int(local_meta["global_size"])
                missing = set(unlimited) - set(sizes)
                if missing:
                    raise ValueError(
                        f"Unknown unlimited dimensions: {sorted(missing)}."
                    )

                chunk_map = (
                    get_chunks(ds, chunks, partition_dim, sizes[partition_dim])
                    if chunks is not None
                    else local_meta["save_chunks"]
                )
                root_data = {}
                variables: dict[str, dict[str, Any]] = {}
                for name, source in ds.variables.items():
                    if name in prewritten_coords:
                        source = prewritten_coords[name]
                    variable = encode_time(source) if is_time_like(source) else source
                    values, dtype = _normalise_variable(source)
                    attrs = {
                        key: value
                        for key, value in variable.attrs.items()
                        if key != MPI_META
                    }
                    dims = tuple(variable.dims)
                    shape = list(values.shape)
                    if partition_dim in dims:
                        shape[dims.index(partition_dim)] = sizes[partition_dim]

                    root_data[name] = {
                        "attrs": attrs,
                        "data": values,
                        "dims": dims,
                        "dtype": dtype,
                        "coord": name in ds.coords,
                    }
                    variables[name] = {
                        "coord": name in ds.coords,
                        "dims": dims,
                        "dtype": "str" if dtype == "str" else np.dtype(dtype).str,
                        "shape": tuple(shape),
                    }

                output_path = str(Path(path).expanduser().resolve(strict=False))
                schema = {
                    "attrs": {
                        key: value for key, value in ds.attrs.items() if key != MPI_META
                    },
                    "chunks": chunk_map,
                    "deflate": None if deflate is None else int(deflate),
                    "hints": hints,
                    "nofill": bool(nofill),
                    "partition_dim": partition_dim,
                    "prewritten": tuple(sorted(prewritten_coords)),
                    "shuffle": bool(shuffle),
                    "sizes": sizes,
                    "unlimited_dim": unlimited,
                    "variables": variables,
                }
            except BaseException as exc:
                error = exc
    elif mpi_runtime.comm.rank == 0:
        try:
            if local_ds is None:
                raise TypeError("Rank 0 must provide an xarray Dataset or DataArray.")
            ds = local_ds
            if partition_dim is None:
                partition_dim = next(
                    (da.dims[0] for da in ds.data_vars.values() if da.dims),
                    None,
                )
            elif partition_dim not in ds.sizes:
                raise ValueError(
                    f"partition_dim {partition_dim!r} is not in {list(ds.sizes)}."
                )

            if deflate is not None and not 0 <= int(deflate) <= 9:
                raise ValueError("deflate must be None or an integer from 0 through 9.")
            if unlimited_dim is None:
                unlimited = ()
            elif isinstance(unlimited_dim, str):
                unlimited = (unlimited_dim,)
            else:
                unlimited = tuple(unlimited_dim)
            missing = set(unlimited) - set(ds.sizes)
            if missing:
                raise ValueError(f"Unknown unlimited dimensions: {sorted(missing)}.")

            partition_chunk_size = get_partition_chunk_size(
                ds, partition_dim, mpi_runtime.comm.size
            )
            chunk_map = get_chunks(ds, chunks, partition_dim, partition_chunk_size)
            # Rank 0 already holds the complete global array for every
            # coordinate here (there is nothing to gather, unlike the
            # distributed path), so a coordinate carrying partition_dim can
            # be written directly below rather than through the scatter
            # loop -- avoiding netCDF4's own default chunking for it.
            prewritten_names = frozenset(
                name for name in ds.coords if partition_dim in ds[name].dims
            )
            root_data = {}
            variables = {}
            for name, source in ds.variables.items():
                dims = tuple(source.dims)
                partitioned = partition_dim in dims and name not in prewritten_names

                if partitioned and not is_time_like(source):
                    # The actual array is deferred to write_partitioned,
                    # which extracts, encodes and scatters one variable at a
                    # time. Materializing every variable's full array here,
                    # before any writing starts, would hold all of them
                    # resident on rank 0 simultaneously -- on top of `ds`
                    # itself -- for no benefit, since create_file never reads
                    # `data` for a partitioned entry (it only pre-extends the
                    # unlimited dimension). Only cheap shape/dtype metadata
                    # is needed here, mirroring _normalise_variable's dtype
                    # rules without touching `.values`. A time-like
                    # partitioned variable is deliberately excluded from
                    # this and falls through to the eager branch below: its
                    # post-encoding dtype (from encode_time's cftime/
                    # timedelta branches) is not always knowable without
                    # actually running the encode, and getting it wrong here
                    # would create the netCDF variable with a dtype that
                    # write_partitioned's later scatter wouldn't match.
                    if source.dtype.kind in ("U", "S", "O"):
                        dtype: str | np.dtype[Any] = "str"
                    elif source.dtype.kind == "b":
                        dtype = np.dtype(np.int8)
                    elif source.dtype.byteorder not in ("=", "|"):
                        dtype = source.dtype.newbyteorder("=")
                    else:
                        dtype = source.dtype
                    attrs = {
                        key: value
                        for key, value in source.attrs.items()
                        if key != MPI_META
                    }
                    root_data[name] = {
                        "attrs": attrs,
                        "data": None,
                        "dims": dims,
                        "dtype": dtype,
                        "coord": name in ds.coords,
                    }
                    variables[name] = {
                        "coord": name in ds.coords,
                        "dims": dims,
                        "dtype": "str" if dtype == "str" else np.dtype(dtype).str,
                        "shape": tuple(int(length) for length in source.shape),
                    }
                    continue

                variable = encode_time(source) if is_time_like(source) else source
                values, dtype = _normalise_variable(source)
                attrs = {
                    key: value
                    for key, value in variable.attrs.items()
                    if key != MPI_META
                }
                root_data[name] = {
                    "attrs": attrs,
                    "data": values,
                    "dims": tuple(variable.dims),
                    "dtype": dtype,
                    "coord": name in ds.coords,
                }
                variables[name] = {
                    "coord": name in ds.coords,
                    "dims": tuple(variable.dims),
                    "dtype": "str" if dtype == "str" else np.dtype(dtype).str,
                    "shape": values.shape,
                }

            output_path = str(Path(path).expanduser().resolve(strict=False))
            schema = {
                "attrs": {
                    key: value for key, value in ds.attrs.items() if key != MPI_META
                },
                "chunks": chunk_map,
                "deflate": None if deflate is None else int(deflate),
                "hints": hints,
                "nofill": bool(nofill),
                "partition_chunk_size": partition_chunk_size,
                "partition_dim": partition_dim,
                "prewritten": tuple(sorted(prewritten_names)),
                "shuffle": bool(shuffle),
                "sizes": dict(ds.sizes),
                "unlimited_dim": unlimited,
                "variables": variables,
            }
            total_bytes = sum(variable.nbytes for variable in ds.variables.values())
            mpi_runtime.log(
                "xgeo.to_netcdf (rank-0 source): rank 0 holds "
                + f"{_format_bytes(total_bytes)} before scatter, "
                + f"~{_format_bytes(total_bytes / mpi_runtime.comm.size)}/rank after. "
                + "An already-distributed input (open_dataset/"
                + "repartition) avoids this rank-0 peak entirely -- see the "
                + "README's Parallel NetCDF output section.",
            )
        except BaseException as exc:
            error = exc

    mpi_runtime.raise_if_error(error, "parallel NetCDF preparation")
    output_path, schema = mpi_runtime.comm.bcast((output_path, schema), root=0)
    if output_path is None or schema is None:
        raise AssertionError("Rank 0 did not broadcast the NetCDF schema.")

    error = None
    if mpi_runtime.comm.rank == 0:
        try:
            if root_data is None:
                raise AssertionError("Rank 0 data buffers are missing.")
            create_file(output_path, schema, root_data)
        except BaseException as exc:
            error = exc
    mpi_runtime.raise_if_error(error, "serial NetCDF schema creation")
    mpi_runtime.comm.barrier()

    try:
        if distributed:
            if local_ds is None or local_meta is None:
                raise AssertionError("Distributed rank-local data are missing.")
            write_distributed(mpi_runtime, output_path, schema, local_ds, local_meta)
        else:
            write_partitioned(mpi_runtime, output_path, schema, local_ds)
    except BaseException:
        # Aborting without a diagnostic leaves the job log with nothing but
        # "MPI_ABORT was invoked", so the failure is reported first.
        traceback.print_exc()
        sys.stderr.flush()
        if mpi_runtime.comm.size > 1:
            mpi_runtime.comm.Abort(1)
        raise

    mpi_runtime.comm.barrier()
    return output_path


def resolve_unlimited_dim(
    unlimited_dim: str | Iterable[str] | None, sizes: Iterable[str]
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

    data0.to_netcdf(file, encoding=enc, format=format, unlimited_dims=[dim0])

    # Append the remaining records in batches.
    starts = range(1, n_items, batch_size)

    if show_progress:
        data_slices = SerialProgressBar(
            starts, description="Writing NetCDF file", file=stdout
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
                    da, units=units if stored is not None else None, calendar=calendar
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
        varname=varname, datatype=da.dtype, dimensions=da.dims, **kwargs
    )
    for attr_name, attr_val in da.attrs.items():
        ncvar.setncattr(attr_name, attr_val)

    if write_values:
        with quiet_netcdf4_writes():
            ncvar[:] = da.values

    return ncvar


__all__ = [
    "NetCDFWriteError",
    "append",
    "createVariable",
    "dataarray_to_netcdf",
    "dataset_to_netcdf",
    "resolve_unlimited_dim",
    "to_netcdf_parallel",
    "to_netcdf_serial",
    "writer_comm",
]
