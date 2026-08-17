"""Parallel NetCDF-4 output using serial schema creation and collective writes.

Rank 0 owns the complete xarray object. It creates the NetCDF schema and all
replicated values serially, then every rank reopens the file in parallel and
collectively writes its balanced slab of each partitioned data variable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import netCDF4
import numpy as np
import xarray as xr
from mpi4py import MPI

from ..core.lib_mpi import mpi
from .encoding import encode_time, is_time_like

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike


class NetCDFWriteError(mpi.MPIError):
    """Raised when parallel NetCDF output cannot proceed."""


def get_chunks(
    ds: xr.Dataset,
    chunks: Mapping[str, Iterable[int]] | None,
) -> dict[str, tuple[int, ...]]:
    """Use existing xarray chunks, or xarray ``chunk('auto')`` when absent."""
    if chunks is not None:
        return {
            name: tuple(int(length) for length in shape)
            for name, shape in chunks.items()
        }

    output: dict[str, tuple[int, ...]] = {}
    for name, da in ds.data_vars.items():
        if da.ndim == 0 or any(length == 0 for length in da.shape):
            continue
        chunked = da if da.chunks is not None else da.chunk("auto")
        output[name] = tuple(max(chunked.chunksizes[dim]) for dim in da.dims)
    return output


def set_attrs(target: Any, attrs: Mapping[str, Any]) -> None:
    for key, value in attrs.items():
        if key != "_FillValue" and value is not None:
            target.setncattr(str(key), value)


def create_file(
    path: str,
    schema: Mapping[str, Any],
    root_data: Mapping[str, Mapping[str, Any]],
) -> None:
    """Create all NetCDF metadata and replicated values serially on rank 0."""
    partition_dim = schema["partition_dim"]
    unlimited = set(schema["unlimited_dim"])
    chunks = schema["chunks"]

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
                and not variable["coord"]
            )
            if not partitioned:
                ncvar[...] = variable["data"]

        set_attrs(nc, schema["attrs"])


def write_partitioned(
    path: str,
    schema: Mapping[str, Any],
    root_data: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Scatter each partitioned root buffer and write its local slab."""
    partition_dim = schema["partition_dim"]
    if partition_dim is None:
        return

    length = int(schema["sizes"][partition_dim])
    counts = np.full(mpi.comm.size, length // mpi.comm.size, dtype=np.int64)
    counts[: length % mpi.comm.size] += 1
    start = int(np.sum(counts[: mpi.comm.rank], dtype=np.int64))
    stop = start + int(counts[mpi.comm.rank])

    info: MPI.Info | None = None
    if mpi.comm.size > 1:
        info = MPI.Info.Create()
        for item in (schema["hints"] or "").split(";"):
            if not item.strip():
                continue
            key, separator, value = item.partition("=")
            if not separator or not key.strip():
                info.Free()
                raise ValueError(f"Invalid MPI-IO hint: {item!r}; expected key=value.")
            info.Set(key.strip(), value.strip())

    try:
        if mpi.comm.size > 1:
            nc = netCDF4.Dataset(
                path,
                mode="r+",
                parallel=True,
                comm=mpi.comm,
                info=info,
            )
        else:
            nc = netCDF4.Dataset(path, mode="r+")

        with nc:
            for name, spec in schema["variables"].items():
                dims = tuple(spec["dims"])
                if partition_dim not in dims or spec["coord"]:
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
                    if root_data is None:
                        raise AssertionError("Rank 0 data buffers are missing.")
                    send = np.ascontiguousarray(
                        np.moveaxis(root_data[name]["data"], axis, 0),
                        dtype=dtype,
                    )

                local = mpi.scatterv(send, counts, local_shape, dtype)
                local = np.moveaxis(local, 0, axis)
                ncvar = nc.variables[name]
                if mpi.comm.size > 1:
                    ncvar.set_collective(True)
                index = tuple(
                    slice(start, stop) if dim == partition_dim else slice(None)
                    for dim in dims
                )
                ncvar[index] = local
    finally:
        if info is not None:
            info.Free()


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
    """Write one xarray object to NetCDF using MPI collective data writes.

    Rank 0 supplies the complete Dataset or DataArray. Other ranks may pass
    ``None``. If ``partition_dim`` is omitted, the first dimension of the first
    dimensional data variable is used. Existing DataArray chunks define the
    NetCDF chunk shape; unchunked arrays use ``da.chunk('auto')`` first. Only
    time-like variables are encoded, using :mod:`.encoding`; other variables
    are written from their original values and attributes.
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

    root_data: dict[str, dict[str, Any]] | None = None
    schema: dict[str, Any] | None = None
    output_path: str | None = None
    error: BaseException | None = None

    if mpi.comm.rank == 0:
        try:
            if isinstance(data, xr.DataArray):
                if data.name is None:
                    raise ValueError("DataArray must have a name for NetCDF output.")
                ds = data.to_dataset()
            elif isinstance(data, xr.Dataset):
                ds = data
            else:
                raise TypeError("Rank 0 must provide an xarray Dataset or DataArray.")

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

            chunk_map = get_chunks(ds, chunks)
            root_data = {}
            variables = {}
            for name, source in ds.variables.items():
                variable = encode_time(source) if is_time_like(source) else source
                values = np.asarray(variable.values)
                attrs = dict(variable.attrs)

                if values.dtype.kind in ("U", "S", "O"):
                    values = np.asarray(
                        [
                            value.decode("utf-8")
                            if isinstance(value, bytes)
                            else str(value)
                            for value in values.ravel()
                        ],
                        dtype=object,
                    ).reshape(values.shape)
                    dtype: str | np.dtype[Any] = "str"
                else:
                    if values.dtype.kind == "b":
                        values = values.astype(np.int8)
                    if values.dtype.byteorder not in ("=", "|"):
                        values = values.astype(values.dtype.newbyteorder("="))
                    values = np.ascontiguousarray(values)
                    dtype = values.dtype

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
            unknown_chunks = set(chunk_map) - set(variables)
            if unknown_chunks:
                raise ValueError(
                    "Chunk specifications reference unknown variables: "
                    + f"{sorted(unknown_chunks)}."
                )

            output_path = str(Path(path).expanduser().resolve(strict=False))
            schema = {
                "attrs": dict(ds.attrs),
                "chunks": chunk_map,
                "deflate": None if deflate is None else int(deflate),
                "hints": hints,
                "nofill": bool(nofill),
                "partition_dim": partition_dim,
                "shuffle": bool(shuffle),
                "sizes": dict(ds.sizes),
                "unlimited_dim": unlimited,
                "variables": variables,
            }
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
        write_partitioned(output_path, schema, root_data)
    except BaseException:
        if mpi.comm.size > 1:
            mpi.comm.Abort(1)
        raise

    mpi.comm.barrier()
    return output_path


__all__ = ["NetCDFWriteError", "to_netcdf_parallel"]
