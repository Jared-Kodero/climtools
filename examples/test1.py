"""Parallel NetCDF4 write benchmark with diagnostic, timestamped logging."""

import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from mpi4py import MPI
from netCDF4 import Dataset

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MPI_TYPE_BY_DTYPE: dict[np.dtype, MPI.Datatype] = {
    np.dtype(np.float32): MPI.FLOAT,
    np.dtype(np.float64): MPI.DOUBLE,
}


def log(message: str) -> None:
    """Print a cleanly aligned, timestamped, rank-tagged diagnostic line."""
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    rank_str = f"R{rank:02d}/{size:02d}"
    print(f"[{stamp}] [{rank_str}] | {message}", flush=True)


def create_mock_dataset(path) -> xr.Dataset:
    """Called only by rank 0. float32, (3600, 721, 1440)."""
    nt, nlat, nlon = 3600, 721, 1440

    ds = xr.Dataset(
        data_vars={
            "mock": (
                ("time", "lat", "lon"),
                np.random.default_rng().random((nt, nlat, nlon), dtype=np.float32),
                {"units": "1", "long_name": "mock"},
            ),
        },
        coords={
            "time": (
                "time",
                np.arange(nt, dtype=np.float64),
                {"units": "days since 1970-01-01"},
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
        attrs={"title": "Parallel NetCDF mock"},
    )

    ds.to_netcdf(path, format="NETCDF4")
    return ds


def save_native(ds: xr.Dataset | None, path: str) -> None:
    if rank != 0:
        return

    var_name = next(iter(ds.data_vars))
    log(f"Starting Normal xarray to_netcdf write: {path}")
    start = time.perf_counter()
    ds.to_netcdf(path, format="NETCDF4")
    elapsed = time.perf_counter() - start
    log(f"Serial write of {path} done in {elapsed:.4f} s")

    expected_shape = ds[var_name].shape
    with xr.open_dataset(path) as ds_check:
        actual_shape = ds_check[var_name].shape
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{path}: shape {actual_shape} != expected {expected_shape}"
        )
    log(f"{path} verified")


def save_parallel(ds: xr.Dataset | None, path: str) -> None:
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

    if rank == 0:
        log("Entering metadata Bcast")

    metadata = comm.bcast(metadata, root=0)

    var_name = metadata["var_name"]
    dims = metadata["dims"]
    n0, n1, n2 = metadata["shape"]
    dtype = np.dtype(metadata["dtype"])
    mpi_type = MPI_TYPE_BY_DTYPE[dtype]

    counts_per_rank = np.full(size, n0 // size, dtype=np.int32)
    counts_per_rank[: n0 % size] += 1

    if rank == 0 and size > n0:
        log(f"WARNING: {size} ranks but only {n0} slabs. Some get zero.")

    t_start = int(np.sum(counts_per_rank[:rank]))
    local_n0 = int(counts_per_rank[rank])
    t_end = t_start + local_n0

    log(f"Assigned slab [{t_start}:{t_end}) ({local_n0} of {n0})")

    local_data = np.empty((local_n0, n1, n2), dtype=dtype)

    nxy = n1 * n2
    slab_type = mpi_type.Create_contiguous(nxy)
    slab_type.Commit()

    counts = counts_per_rank
    displacements = np.zeros(size, dtype=np.int32)
    displacements[1:] = np.cumsum(counts[:-1])

    if rank == 0:
        full_data = np.ascontiguousarray(data_array.values, dtype=dtype)
        send_buffer = [full_data, counts, displacements, slab_type]
    else:
        send_buffer = None

    recv_buffer = [local_data, local_n0, slab_type]

    if rank == 0:
        log("Entering Scatterv")

    comm.Scatterv(send_buffer, recv_buffer, root=0)

    if rank == 0:
        log("Scatterv complete")

    slab_type.Free()

    # --- UPDATED DIAGNOSTIC LOG ---
    if local_n0 > 0:
        val_start = local_data[0, 0, 0]
        val_stop = local_data[-1, -1, -1]
        log(
            f"Post-Scatterv: Owns idx [{t_start}:{t_end}) | Shape: {local_data.shape} | Mean: {local_data.mean():.4f} | Val Start: {val_start:.4f} | Val Stop: {val_stop:.4f}"
        )
    else:
        log(f"Post-Scatterv: Owns idx [{t_start}:{t_end}) | EMPTY SLAB")
    # ------------------------------

    if rank == 0:
        del ds, data_array, full_data

    if rank == 0:
        log(f"Creating file structure serially: {path}")
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
        log("File structure created")

    comm.Barrier()

    start_parallel = time.perf_counter()

    if rank == 0:
        log(f"Calling nc_open_par (r+, parallel=True): {path}")

    with Dataset(path, mode="r+", parallel=True, comm=comm, info=MPI.Info()) as nc:
        data_var = nc.variables[var_name]
        data_var.set_collective(True)

        if rank == 0:
            log("Entering collective write")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=".*Setting the shape on a NumPy array.*",
            )
            data_var[t_start:t_end, :, :] = local_data

        if rank == 0:
            log("Collective write complete")

    comm.Barrier()
    end_parallel = time.perf_counter()
    if rank == 0:
        log(f"Parallel write of {path} done in {end_parallel - start_parallel:.4f} s")
        log(f"Verifying {path}")

    error_message = ""
    try:
        with Dataset(path, mode="r", parallel=True, comm=comm, info=MPI.Info()) as nc:
            data_var_check = nc.variables[var_name]
            data_var_check.set_collective(True)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message=".*Setting the shape on a NumPy array.*",
                )
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
        if all_passed:
            log(f"{path}: Verification PASSED")
        else:
            log(f"{path}: Verification FAILED on at least one rank")

    if not all_passed:
        comm.Abort(1)


def main() -> None:
    if rank == 0:
        log("Started execution")

    out_dir = Path.home() / "scratch" / "io_mpi_test"

    if rank == 0:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = create_mock_dataset(out_dir / "mock_in.nc")
    else:
        ds = None

    save_native(ds, out_dir / "mock_serial_out.nc")
    save_parallel(ds, out_dir / "mock_parallel_out.nc")

    if rank == 0:
        log("All datasets written and verified")


if __name__ == "__main__":
    main()
