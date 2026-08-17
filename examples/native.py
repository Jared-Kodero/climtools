"""Parallel NetCDF4 write benchmark with diagnostic, timestamped logging.

Two independently shaped/typed datasets are each written two ways: a
serial xarray write (save_native) and a distributed netCDF4 write
(save_parallel), then read back and checked byte-for-byte.

save_parallel uses a two-phase pattern instead of creating the file inside
a parallel session:

1. Rank 0 creates the file, its dimensions, variables, attributes, and
   coordinate values with a plain SERIAL (non-parallel) netCDF4 session,
   then closes it.
2. Every rank reopens the now-fully-structured file with
   mode="r+", parallel=True and writes only its own slab of the main
   variable, collectively.

This avoids a real HDF5/netCDF-C pitfall: netCDF-C enables
H5Pset_all_coll_metadata_ops on files it creates in parallel mode, which
requires every metadata-touching call - including the metadata lookup
behind what looks like an "independent" first write to a variable - to be
issued collectively by every rank, or HDF5's behavior is undefined
(observed here as a hang immediately after variable creation, since only
rank 0 called the independent coordinate write while the other ranks were
not participating in the resulting collective metadata operation).
Creating the structure in a plain serial session sidesteps this
completely: no metadata operation ever happens inside a parallel session.
https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html
https://github.com/Unidata/netcdf-c/issues/781
https://github.com/Unidata/netcdf4-python/issues/1108
https://unidata.github.io/netcdf4-python/

Every MPI-collective call is bracketed by a barrier and a timestamped,
rank-tagged log line. If the job hangs, the last line printed identifies
which collective call is blocked.
"""

import time
from datetime import datetime, timezone

import numpy as np
import xarray as xr
from mpi4py import MPI
from netCDF4 import Dataset

from climtools import xgeo

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MPI_TYPE_BY_DTYPE: dict[np.dtype, MPI.Datatype] = {
    np.dtype(np.float32): MPI.FLOAT,
    np.dtype(np.float64): MPI.DOUBLE,
}


def log(message: str) -> None:
    """Print a UTC-timestamped, rank-tagged, flushed diagnostic line."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{stamp}Z] [rank {rank}/{size}] {message}", flush=True)


def create_precipitation_dataset() -> xr.Dataset:
    """Called only by rank 0. float32, (100, 721, 1440)."""
    nt, nlat, nlon = 100, 721, 1440
    return xr.Dataset(
        data_vars={
            "precipitation": (
                ("time", "lat", "lon"),
                np.random.random((nt, nlat, nlon)).astype(np.float32),
                {"units": "mm day-1", "long_name": "precipitation rate"},
            ),
        },
        coords={
            "time": (
                "time",
                np.arange(nt, dtype=np.float64),
                {"units": "days since 2000-01-01"},
            ),
            "lat": (
                "lat",
                np.linspace(-90.0, 90.0, nlat, dtype=np.float32),
                {"units": "degrees_north"},
            ),
            "lon": (
                "lon",
                np.linspace(0.0, 359.75, nlon, dtype=np.float32),
                {"units": "degrees_east"},
            ),
        },
        attrs={"title": "Parallel NetCDF example: precipitation"},
    )


def create_temperature_dataset() -> xr.Dataset:
    """Called only by rank 0. float64, (60, 181, 360) - deliberately a
    different shape and dtype than the precipitation dataset."""
    nt, nlat, nlon = 60, 181, 360
    return xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lat", "lon"),
                np.random.uniform(250.0, 310.0, (nt, nlat, nlon)).astype(np.float64),
                {"units": "K", "long_name": "surface air temperature"},
            ),
        },
        coords={
            "time": (
                "time",
                np.arange(nt, dtype=np.float64),
                {"units": "days since 2000-01-01"},
            ),
            "lat": (
                "lat",
                np.linspace(-90.0, 90.0, nlat, dtype=np.float32),
                {"units": "degrees_north"},
            ),
            "lon": (
                "lon",
                np.linspace(0.0, 359.0, nlon, dtype=np.float32),
                {"units": "degrees_east"},
            ),
        },
        attrs={"title": "Parallel NetCDF example: temperature"},
    )


def save_native(ds: xr.Dataset | None, path: str) -> None:
    """Rank 0 writes `ds` serially with xarray, timed, then reopens the
    file to verify the written shape matches what was written. No-op on
    other ranks."""
    if rank != 0:
        return

    var_name = next(iter(ds.data_vars))
    log(f"starting serial xarray to_netcdf write: {path}")
    start = time.time()
    ds.to_netcdf(path, format="NETCDF4")
    elapsed = time.time() - start
    log(f"serial write of {path} done in {elapsed:.4f} s")

    expected_shape = ds[var_name].shape
    with xr.open_dataset(path) as ds_check:
        actual_shape = ds_check[var_name].shape
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{path}: shape {actual_shape} != expected {expected_shape}"
        )
    log(f"{path} verified")


def save_parallel(ds: xr.Dataset | None, path: str) -> None:
    """Distribute the single data variable of `ds` (valid on rank 0 only,
    pass None elsewhere) across ranks along its first dimension, write it
    with netCDF4 parallel collective I/O, then verify by a second
    collective read-back. Aborts all ranks together on any failure."""
    if rank == 0:
        var_name = next(iter(ds.data_vars))
        data_array = ds[var_name]
        dims = data_array.dims
        shape = data_array.shape
        metadata = {
            "var_name": var_name,
            "dims": dims,
            "shape": shape,
            "dtype": data_array.dtype,
            "var_attrs": dict(data_array.attrs),
            "global_attrs": dict(ds.attrs),
            "coords": {name: (ds[name].values, dict(ds[name].attrs)) for name in dims},
        }
    else:
        metadata = None

    log(f"entering metadata Bcast for {path}")
    metadata = comm.bcast(metadata, root=0)
    log("metadata Bcast complete")

    var_name = metadata["var_name"]
    dims = metadata["dims"]
    n0, n1, n2 = metadata["shape"]
    dtype = np.dtype(metadata["dtype"])
    mpi_type = MPI_TYPE_BY_DTYPE[dtype]

    counts_per_rank = np.full(size, n0 // size, dtype=np.int64)
    counts_per_rank[: n0 % size] += 1
    if rank == 0 and size > n0:
        log(
            f"WARNING: {size} ranks but only {n0} along {dims[0]}: "
            f"{size - n0} rank(s) get a zero-length slab in the collective write."
        )

    t_start = int(np.sum(counts_per_rank[:rank]))
    local_n0 = int(counts_per_rank[rank])
    t_end = t_start + local_n0
    log(f"assigned {dims[0]} slab [{t_start}:{t_end}) ({local_n0} of {n0})")

    local_data = np.empty((local_n0, n1, n2), dtype=dtype)

    nxy = n1 * n2
    counts = counts_per_rank * nxy
    displacements = np.zeros(size, dtype=np.int64)
    displacements[1:] = np.cumsum(counts[:-1])

    if rank == 0:
        full_data = np.ascontiguousarray(data_array.values, dtype=dtype)
        send_buffer = [full_data, counts, displacements, mpi_type]
    else:
        send_buffer = None

    log("entering Scatterv")
    comm.Scatterv(send_buffer, local_data, root=0)
    log("Scatterv complete")

    if rank == 0:
        del ds, data_array, full_data

    # Phase 1: rank 0 creates the file, dimensions, variables, attributes,
    # and coordinate values in a plain SERIAL session - no parallel=True,
    # so no metadata operation here is subject to the collective-metadata
    # requirement described in this function's docstring.
    if rank == 0:
        log(f"creating file structure serially: {path}")
        with Dataset(path, mode="w", format="NETCDF4") as nc:
            for dim_name, dim_size in zip(dims, shape):
                nc.createDimension(dim_name, dim_size)
            for dim_name in dims:
                values, attrs = metadata["coords"][dim_name]
                coord_var = nc.createVariable(dim_name, values.dtype, (dim_name,))
                coord_var.setncatts(attrs)
                coord_var[:] = values
            data_var = nc.createVariable(var_name, dtype, dims)
            nc.setncatts(metadata["global_attrs"])
            data_var.setncatts(metadata["var_attrs"])
        log(f"file structure for {path} created")

    log("entering post-create Barrier")
    comm.Barrier()
    log("post-create Barrier complete")

    # Phase 2: every rank reopens the now-fixed file in parallel mode and
    # writes only its own slab, collectively. No metadata is created or
    # modified in this session.
    start_parallel = time.time()

    log(f"calling nc_open_par (Dataset open r+, parallel=True): {path}")
    with Dataset(path, mode="r+", parallel=True, comm=comm, info=MPI.Info()) as nc:
        log("nc_open_par returned")
        data_var = nc.variables[var_name]
        data_var.set_collective(True)
        log("entering collective write")
        data_var[t_start:t_end, :, :] = local_data
        log("collective write complete")

    log("nc_close complete")
    comm.Barrier()
    end_parallel = time.time()
    if rank == 0:
        log(f"parallel write of {path} done in {end_parallel - start_parallel:.4f} s")

    log(f"verifying {path}")
    error_message = ""
    try:
        with Dataset(path, mode="r", parallel=True, comm=comm, info=MPI.Info()) as nc:
            data_var_check = nc.variables[var_name]
            data_var_check.set_collective(True)
            readback = data_var_check[t_start:t_end, :, :]
            if not np.array_equal(readback, local_data):
                raise AssertionError(
                    f"{path}: {var_name} slab [{t_start}:{t_end}) mismatch"
                )
            for dim_name in dims:
                expected_values, _ = metadata["coords"][dim_name]
                if not np.array_equal(nc.variables[dim_name][:], expected_values):
                    raise AssertionError(f"{path}: {dim_name} coordinate mismatch")
        passed = True
    except Exception as exc:
        passed = False
        error_message = str(exc)

    if not passed:
        log(f"VERIFICATION FAILED: {error_message}")

    local_passed = np.array([1 if passed else 0], dtype=np.int32)
    global_passed = np.zeros_like(local_passed)
    comm.Allreduce(local_passed, global_passed, op=MPI.LAND)
    all_passed = bool(global_passed[0])
    if rank == 0:
        log(
            f"{path}: verification passed"
            if all_passed
            else f"{path}: verification FAILED on at least one rank"
        )

    if not all_passed:
        comm.Abort(1)


def main() -> None:
    log(f"MPI environment: rank {rank} of {size}")

    if rank == 0:
        precipitation_ds: xr.Dataset = create_precipitation_dataset()
        temperature_ds: xr.Dataset = create_temperature_dataset()
    else:
        precipitation_ds = xgeo.empty_dataset()
        temperature_ds = None

    # save_native(precipitation_ds, "precipitation_serial.nc")
    # save_native(temperature_ds, "temperature_serial.nc")

    # save_parallel(precipitation_ds, "precipitation_parallel.nc")
    # save_parallel(temperature_ds, "temperature_parallel.nc")

    xgeo.to_netcdf(
        precipitation_ds,
        "/users/jkodero/research/scratch/jobtmp/data/pr1",
        unlimited_dim="time",
        partition_dim="time",
        parallel=True,
    )

    if rank == 0:
        log("all datasets written and verified")


if __name__ == "__main__":
    main()
