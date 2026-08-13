#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <netcdf.h>
#include <netcdf_meta.h>
#include <netcdf_par.h>

#if !defined(NC_HAS_PARALLEL4) || !NC_HAS_PARALLEL4
#error "The active NetCDF-C headers do not provide parallel NetCDF-4 support."
#endif

static void require_netcdf(int status, const char *operation, int rank)
{
    if (status == NC_NOERR) return;
    fprintf(stderr, "[mpi] rank %d: %s failed: %s\n", rank, operation,
            nc_strerror(status));
    MPI_Abort(MPI_COMM_WORLD, status);
}

static void require_mpi(int status, const char *operation, int rank)
{
    if (status == MPI_SUCCESS) return;
    fprintf(stderr, "[mpi] rank %d: %s failed\n", rank, operation);
    MPI_Abort(MPI_COMM_WORLD, status);
}

int main(int argc, char **argv)
{
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided)
        != MPI_SUCCESS) {
        fputs("[mpi] MPI_Init_thread failed\n", stderr);
        return EXIT_FAILURE;
    }

    int rank = -1;
    int size = 0;
    require_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank", rank);
    require_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size", rank);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fputs("[mpi] MPI thread support is insufficient\n", stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (argc != 2 || size < 2) {
        if (rank == 0) {
            fputs("[mpi] the capability probe requires an output path and at "
                  "least two MPI ranks\n",
                  stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const char *path = argv[1];
    if (rank == 0) remove(path);
    require_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier", rank);

    int ncid = -1;
    require_netcdf(
        nc_create_par(path, NC_NETCDF4 | NC_CLOBBER, MPI_COMM_WORLD,
                      MPI_INFO_NULL, &ncid),
        "nc_create_par", rank);

    int dimid = -1;
    int varid = -1;
    require_netcdf(nc_def_dim(ncid, "rank", (size_t)size, &dimid),
                   "nc_def_dim", rank);
    require_netcdf(nc_def_var(ncid, "value", NC_INT, 1, &dimid, &varid),
                   "nc_def_var", rank);
    require_netcdf(nc_enddef(ncid), "nc_enddef", rank);
    require_netcdf(nc_var_par_access(ncid, varid, NC_COLLECTIVE),
                   "nc_var_par_access", rank);

    const size_t start[1] = {(size_t)rank};
    const size_t count[1] = {1};
    const int value = 1000 + rank;
    require_netcdf(nc_put_vara_int(ncid, varid, start, count, &value),
                   "nc_put_vara_int", rank);
    require_netcdf(nc_close(ncid), "nc_close", rank);
    require_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier", rank);

    int verified = 1;
    if (rank == 0) {
        int serial_ncid = -1;
        int serial_varid = -1;
        int *values = calloc((size_t)size, sizeof *values);
        if (!values) {
            verified = 0;
        } else if (nc_open(path, NC_NOWRITE, &serial_ncid) != NC_NOERR
                   || nc_inq_varid(serial_ncid, "value", &serial_varid)
                          != NC_NOERR
                   || nc_get_var_int(serial_ncid, serial_varid, values)
                          != NC_NOERR) {
            verified = 0;
        }

        for (int index = 0; verified && index < size; index++) {
            if (values[index] != 1000 + index) verified = 0;
        }
        free(values);
        if (serial_ncid >= 0 && nc_close(serial_ncid) != NC_NOERR) verified = 0;
        if (remove(path) != 0) verified = 0;
    }

    require_mpi(MPI_Bcast(&verified, 1, MPI_INT, 0, MPI_COMM_WORLD),
                "MPI_Bcast", rank);
    if (!verified && rank == 0) {
        fputs("[mpi] parallel NetCDF-4 capability probe verification failed\n",
              stderr);
    }
    if (MPI_Finalize() != MPI_SUCCESS) return EXIT_FAILURE;
    return verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
