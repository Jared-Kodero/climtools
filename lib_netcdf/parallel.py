"""Parallel NetCDF-4 output for root-owned or MPI-distributed xarray data.

Rank-0-owned objects retain the legacy scatter path. Objects carrying
``mpi_meta`` are already distributed: every rank participates directly in
collective NetCDF writes using its recorded global ``start:stop`` interval.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import traceback
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

# HDF5 (via netCDF4's parallel4 backend) takes out a file lock on open and
# expects a clean release on close. On network/parallel filesystems commonly
# used for HPC scratch space (Lustre, GPFS, and some NFS configurations),
# that lock is not always visible as released to a *different* opener the
# instant a collective MPI-IO close returns on this rank: the underlying
# filesystem's lock/lease state propagates to the metadata server
# asynchronously. A file written collectively with `parallel=True` and then
# immediately reopened without `parallel=True` (as `to_netcdf`'s serial
# readback path does) can therefore block indefinitely waiting for a lock
# that, from HDF5's point of view, is still held. Disabling HDF5's own file
# locking sidesteps the race entirely; climtools already serializes
# concurrent access to a given path through explicit MPI barriers, so HDF5's
# additional locking is redundant defense-in-depth, not the only thing
# preventing concurrent writers. See
# https://docs.hdfgroup.org/hdf5/rfc/RFC_file_locking.pdf and
# https://forum.hdfgroup.org for background on this behavior. A caller that
# has already set the variable (e.g. because their filesystem needs the
# opposite setting) is left alone.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import dask
import netCDF4
import numpy as np
import xarray as xr
from mpi4py import MPI

from ..core.lib_mpi import mpi
from ..core.xr_chunks import get_chunk_bounds, get_chunks, get_partition_chunk_size
from ..core.xr_meta import MPI_META, _format_bytes, get_mpi_meta
from .encoding import encode_time, is_time_like

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike


class NetCDFWriteError(mpi.MPIError):
    """Raised when parallel NetCDF output cannot proceed."""


# A single HDF5 chunk's byte size must stay strictly under 2**32 bytes (4
# GiB) -- confirmed directly against this build with a binary search: a
# (16383, 256, 256) float32 chunk (3.9998 GiB) succeeds, (16384, 256, 256)
# (4.0000 GiB) raises "NetCDF: Bad chunk sizes." from the netCDF-C library.
# Half that hard limit is used as the working target below, leaving headroom
# for HDF5/filter (zlib, shuffle) bookkeeping overhead per chunk rather than
# skimming the exact boundary.
def set_attrs(target: Any, attrs: Mapping[str, Any]) -> None:
    """Set serializable NetCDF attributes, excluding internal MPI metadata."""
    for key, value in attrs.items():
        if key not in ("_FillValue", MPI_META) and value is not None:
            target.setncattr(str(key), value)


def _normalise_variable(source: xr.DataArray) -> tuple[np.ndarray, str | np.dtype[Any]]:
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
def quiet_netcdf4_writes() -> Iterator[None]:
    """Suppress netCDF4-python's NumPy 2.5 shape-assignment DeprecationWarning.

    ``Variable.__setitem__`` compares the caller's ``data.shape`` tuple against
    the list returned by ``netCDF4.utils._out_array_shape``. A tuple never
    equals a list, so every write of a variable with more than one dimension
    takes the reshape branch, which assigns to ``ndarray.shape``. NumPy 2.5
    deprecated that assignment, so correct multidimensional writes emit a
    warning that no caller can avoid by changing the array it passes. The fix
    landed upstream (Unidata/netcdf4-python issue 1468, now using ``reshape``)
    but is not in any released version, so the warning is filtered here rather
    than propagated to every user of climtools.
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
    """Grow an unlimited dimension to its final length before parallel writes.

    A netCDF variable with an unlimited dimension starts with a zero extent
    and is grown implicitly by whichever write goes furthest. Under MPI that
    growth is an ``H5Dset_extent`` call, which HDF5 requires every rank to make
    collectively with identical arguments. Ranks writing different slabs ask
    for different extents, so the file ends up sized by one arbitrary rank and
    the slabs beyond that point are lost. Writing a single element at the last
    global index here sets the final extent serially, so no rank has to extend
    anything during the collective phase. That element belongs to the rank that
    owns the last slab and is overwritten by it.
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
    """Create NetCDF metadata and write variables replicated on all ranks."""
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


def writer_comm(has_data: bool) -> MPI.Comm:
    """Return a communicator containing only the ranks that own output data.

    netCDF4-python skips ``nc_put_vara`` entirely when a rank's selection is
    empty (``if dataput.size == 0: continue`` in ``Variable.__setitem__``), so
    a rank with no data never enters the MPI-IO collective that the remaining
    ranks are blocking in. Independent access is not an alternative, because
    netCDF-C rejects it for variables with an unlimited dimension and HDF5
    rejects it for filtered variables. Excluding empty ranks from the file's
    communicator is therefore the only correct option: the collective is then
    posted by exactly the ranks that reach it.

    Parameters
    ----------
    has_data : bool
        Whether the calling rank owns at least one element to write.

    Returns
    -------
    mpi4py.MPI.Comm
        Communicator to open the file with, or ``MPI.COMM_NULL`` on ranks that
        must not touch the file.
    """
    if mpi.comm.size == 1:
        return mpi.comm if has_data else MPI.COMM_NULL
    return mpi.comm.Split(1 if has_data else MPI.UNDEFINED, mpi.comm.rank)


def free_writer_comm(comm: MPI.Comm) -> None:
    """Free a communicator produced by :func:`writer_comm`."""
    if comm != MPI.COMM_NULL and comm != mpi.comm:
        comm.Free()


def open_in_parallel(
    path: str,
    schema: Mapping[str, Any],
    comm: MPI.Comm,
) -> netCDF4.Dataset:
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
    path: str,
    schema: Mapping[str, Any],
    ds: xr.Dataset,
    meta: Mapping[str, Any],
) -> None:
    """Collectively write rank-local slabs from an already distributed Dataset."""

    partition_dim = str(meta["dim"])
    start = int(meta["start"])
    stop = int(meta["stop"])
    prewritten = set(schema.get("prewritten", ()))

    comm = writer_comm(stop > start)
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
        free_writer_comm(comm)
        # Every rank reaches this barrier -- including the COMM_NULL/no-data
        # ranks that returned above -- so nothing downstream (a different
        # communicator, a non-parallel reopen for validation, ...) can run
        # ahead of a writer rank that is still inside nc.close().
        mpi.comm.Barrier()


def write_partitioned(
    path: str,
    schema: Mapping[str, Any],
    source_ds: xr.Dataset | None,
) -> None:
    """Scatter root-owned partitioned buffers and collectively write them.

    ``mpi.scatterv`` is posted on the full communicator by every rank,
    including those that receive nothing, while the NetCDF writes are posted
    only on the writer sub-communicator. The two sequences are internally
    ordered on their own communicators, so they cannot deadlock against one
    another.

    Rank 0's array for each partitioned variable is pulled from
    ``source_ds`` here, immediately before that variable's ``scatterv``,
    rather than upfront for every variable at once: ``root_data`` only
    carries schema metadata (dims/dtype/shape) for these entries (see the
    schema-construction step in ``to_netcdf_parallel``), not the array
    itself. Once a variable's local slab has been sent, the reference to
    its full array falls out of scope before the next variable is touched,
    so at most one partitioned variable's array -- not all of them -- adds
    to rank 0's resident set at any point in this loop.
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
        schema.get("partition_chunk_size") or max(1, math.ceil(length / mpi.comm.size))
    )
    bounds = [
        get_chunk_bounds(length, chunk_size, rank, mpi.comm.size)
        for rank in range(mpi.comm.size)
    ]
    counts = np.array([stop - start for start, stop in bounds], dtype=np.int64)
    start, stop = bounds[mpi.comm.rank]

    comm = writer_comm(stop > start)
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
            local_shape = (int(counts[mpi.comm.rank]), *moved_shape[1:])
            dtype = np.dtype(spec["dtype"])

            send = None
            if mpi.comm.rank == 0:
                if source_ds is None:
                    raise AssertionError("Rank 0 source Dataset is missing.")
                variable = source_ds[name]
                variable = encode_time(variable) if is_time_like(variable) else variable
                send = np.ascontiguousarray(
                    np.moveaxis(np.asarray(variable.values), axis, 0),
                    dtype=dtype,
                )

            local = mpi.scatterv(send, counts, local_shape, dtype)
            # `send = None` (not `del send` + `gc.collect()`) is deliberate
            # and sufficient: CPython reclaims a non-cyclic object the
            # moment its refcount hits zero, and mpi.scatterv's own Scatterv
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
        free_writer_comm(comm)
        # See the matching comment in write_distributed: guarantees every
        # rank waits for every writer rank's close() before anything
        # downstream reopens the file.
        mpi.comm.Barrier()


def to_netcdf_parallel(
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
    """Write an xarray object to NetCDF using MPI collective data writes.

    Objects carrying ``mpi_meta`` are treated as already distributed. Every
    rank contributes its existing local slab directly and no data gather or
    scatter is performed. For an ordinary object, rank 0 owns the complete
    data; if that data is dask-backed, it is distributed lazily first (see
    ``mpi.xarray.distribute``), so no rank -- including rank 0 -- ever
    materializes more than its own share. An eager (already in-memory,
    non-dask) object instead uses the legacy scatter path unchanged: rank 0
    materializes the array (it already had to, to be eager) and every
    rank's slice is scattered to it directly, which is faster for data that
    gains nothing from staying lazy.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray or None
        Rank-local distributed data, or the complete object on rank 0 for the
        legacy or dask-backed-distribute path.
    path : str or os.PathLike
        Output path.
    partition_dim : str, optional
        Partition dimension. In distributed mode this must match ``mpi_meta``.
    deflate : int, optional
        NetCDF compression level.
    shuffle : bool, default True
        Apply the HDF5 shuffle filter.
    chunks : mapping, optional
        Explicit variable chunk shapes.
    unlimited_dim : str or iterable of str, optional
        Unlimited dimensions.
    hints : str, optional
        Semicolon-separated MPI-IO hints.
    nofill : bool, default True
        Disable NetCDF pre-filling.
    allow_serial : bool, default False
        Permit execution with one MPI rank.

    Returns
    -------
    str
        Absolute output path.
    """
    if mpi.comm.size == 1 and not allow_serial:
        raise NetCDFWriteError(
            "MPI_COMM_WORLD contains one process. Launch with mpirun/mpiexec/srun "
            + "or pass allow_serial=True."
        )
    if mpi.comm.size > 1 and not getattr(netCDF4, "__has_parallel4_support__", False):
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
    mpi.raise_if_error(error, "parallel NetCDF input validation")

    local_meta = get_mpi_meta(local_ds) if local_ds is not None else None
    distributed = local_meta is not None

    # The distributed and scatter paths post different collectives, so every
    # rank must take the same one. Disagreement is reported instead of hanging.
    agreed = mpi.comm.allgather(distributed)
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
            if mpi.comm.rank == 0 and local_ds is not None:
                is_dask_backed = any(
                    dask.is_dask_collection(variable.data)
                    for variable in local_ds.variables.values()
                )
        except BaseException as exc:
            error = exc
        mpi.raise_if_error(error, "parallel NetCDF dask-backed detection")
        is_dask_backed = mpi.comm.bcast(is_dask_backed, root=0)

        if is_dask_backed:
            local_ds = mpi.xarray.distribute(
                local_ds if mpi.comm.rank == 0 else None,
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
        mpi.raise_if_error(error, "parallel NetCDF partition dimension")
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
            mpi.xarray.attach_save_chunks(local_ds)
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
        mpi.raise_if_error(
            error,
            "parallel NetCDF coordinate discovery",
            signature=tuple(coord_names),
        )

        for coord_name in coord_names:
            coordinate = local_ds[coord_name]
            axis = coordinate.get_axis_num(partition_dim)
            local_values = np.asarray(coordinate.values)
            pieces = mpi.comm.gather((start, stop, local_values), root=0)
            if mpi.comm.rank == 0:
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
            mpi.raise_if_error(
                error, f"parallel NetCDF coordinate gather ({coord_name})"
            )

        # No data gather/scatter. Rank 0 only constructs the schema from its
        # local metadata and mpi_meta's global partition length.
        if mpi.comm.rank == 0:
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
    elif mpi.comm.rank == 0:
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
                ds, partition_dim, mpi.comm.size
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
            mpi.log(
                "xgeo.to_netcdf (rank-0 source): rank 0 holds "
                + f"{_format_bytes(total_bytes)} before scatter, "
                + f"~{_format_bytes(total_bytes / mpi.comm.size)}/rank after. "
                + "An already-distributed input (mpi.xarray.open_dataset/"
                + "redistribute) avoids this rank-0 peak entirely -- see the "
                + "README's Parallel NetCDF output section.",
            )
        except BaseException as exc:
            error = exc

    mpi.raise_if_error(error, "parallel NetCDF preparation")
    output_path, schema = mpi.comm.bcast((output_path, schema), root=0)
    if output_path is None or schema is None:
        raise AssertionError("Rank 0 did not broadcast the NetCDF schema.")

    error = None
    if mpi.comm.rank == 0:
        try:
            if root_data is None:
                raise AssertionError("Rank 0 data buffers are missing.")
            create_file(output_path, schema, root_data)
        except BaseException as exc:
            error = exc
    mpi.raise_if_error(error, "serial NetCDF schema creation")
    mpi.comm.barrier()

    try:
        if distributed:
            if local_ds is None or local_meta is None:
                raise AssertionError("Distributed rank-local data are missing.")
            write_distributed(output_path, schema, local_ds, local_meta)
        else:
            write_partitioned(output_path, schema, local_ds)
    except BaseException:
        # Aborting without a diagnostic leaves the job log with nothing but
        # "MPI_ABORT was invoked", so the failure is reported first.
        traceback.print_exc()
        sys.stderr.flush()
        if mpi.comm.size > 1:
            mpi.comm.Abort(1)
        raise

    mpi.comm.barrier()
    return output_path


__all__ = ["NetCDFWriteError", "to_netcdf_parallel", "writer_comm"]
