/* mpi_netcdf.h - minimal C ABI for MPI-parallel NetCDF-4 writing from Python.
 *
 * The whole API is plain C (no MPI types in the signatures) so that it can be
 * driven from ctypes without the caller needing mpi4py or an MPI-aware build
 * of Python. Status functions return 0 on success and a negative value on
 * failure unless their individual comments state otherwise. mpi_netcdf_strerror()
 * describes the last failure on the calling process.
 *
 * All routines marked COLLECTIVE must be called by every rank of
 * MPI_COMM_WORLD, in the same order, with identical metadata arguments.
 */
#ifndef MPI_NETCDF_H
#define MPI_NETCDF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NetCDF external type codes, repeated here so the Python side never has to
 * parse netcdf.h.  Values are fixed by the NetCDF file format. */
#define MPI_NETCDF_BYTE    1
#define MPI_NETCDF_CHAR    2
#define MPI_NETCDF_SHORT   3
#define MPI_NETCDF_INT     4
#define MPI_NETCDF_FLOAT   5
#define MPI_NETCDF_DOUBLE  6
#define MPI_NETCDF_UBYTE   7
#define MPI_NETCDF_USHORT  8
#define MPI_NETCDF_UINT    9
#define MPI_NETCDF_INT64  10
#define MPI_NETCDF_UINT64 11

/* Parallel access mode for a variable. */
#define MPI_NETCDF_INDEPENDENT 0
#define MPI_NETCDF_COLLECTIVE  1

typedef struct mpi_netcdf_file mpi_netcdf_file;

/* --- process management -------------------------------------------------- */

/* Initialise MPI if it is not already initialised.  Safe to call repeatedly.
 * Returns 0 on success and does not fail on a single-rank world. */
int mpi_netcdf_init(void);

int mpi_netcdf_rank(void);          /* rank in MPI_COMM_WORLD, -1 before mpi_netcdf_init  */
int mpi_netcdf_size(void);          /* size of MPI_COMM_WORLD, -1 before mpi_netcdf_init  */
int mpi_netcdf_thread_level(void);  /* provided MPI thread level                   */

/* COLLECTIVE. Logical AND of ok across all ranks; returns 1 if every rank
 * passed a non-zero value.  Used to keep ranks from deadlocking in a
 * collective when one of them has raised a Python exception. */
int mpi_netcdf_consensus(int ok);

/* COLLECTIVE. Barrier. */
int mpi_netcdf_barrier(void);

/* COLLECTIVE. All-gather of one 64-bit integer per rank into out[size]. */
int mpi_netcdf_allgather_i64(long long value, long long *out);

/* COLLECTIVE. Broadcast one 64-bit integer from root. */
int mpi_netcdf_bcast_i64(long long *value, int root);

/* COLLECTIVE. Broadcast n bytes from root.  Used to hand the result of a
 * rank-0-only computation back to the other ranks. */
int mpi_netcdf_bcast_bytes(void *buf, long long n, int root);

/* Terminate the whole job.  Called when one rank cannot continue. */
void mpi_netcdf_abort(int code);

/* Finalise MPI, but only if this library was the one that initialised it. */
int mpi_netcdf_finalize(void);

const char *mpi_netcdf_strerror(void);
const char *mpi_netcdf_version(void);       /* NetCDF-C version string */
int mpi_netcdf_has_parallel_filters(void);  /* 1 if compression is usable in parallel */

/* --- file lifecycle ------------------------------------------------------ */

/* Create a NetCDF-4/HDF5 file.
 *   path      : output path on a parallel file system
 *   nofill    : 1 to disable pre-filling (recommended, large speed-up)
 *   hints     : "key=value;key=value" MPI-IO hints, may be NULL
 * Returns NULL on failure. */
mpi_netcdf_file *mpi_netcdf_create(const char *path, int nofill, const char *hints);

/* COLLECTIVE. Define a dimension.  len is the global length; pass len == 0
 * together with unlimited == 1 for a record dimension. */
int mpi_netcdf_def_dim(mpi_netcdf_file *f, const char *name, size_t len, int unlimited);

/* COLLECTIVE. Define a variable of global shape given by dimnames[ndims].
 * deflate < 0 disables compression.  chunks may be NULL for the NetCDF
 * default chunking. */
int mpi_netcdf_def_var(mpi_netcdf_file *f, const char *name, int xtype, int ndims,
                const char *const *dimnames, int deflate, int shuffle,
                const size_t *chunks);

/* COLLECTIVE. Attribute definition.  var == NULL or "" targets the global
 * attribute set. */
int mpi_netcdf_put_att_text(mpi_netcdf_file *f, const char *var, const char *name,
                     const char *value);
int mpi_netcdf_put_att_num(mpi_netcdf_file *f, const char *var, const char *name,
                    int xtype, size_t n, const void *values);

/* COLLECTIVE. Leave define mode and fix the parallel access mode of every
 * variable defined so far. */
int mpi_netcdf_enddef(mpi_netcdf_file *f, int access_mode);

/* COLLECTIVE. Override the parallel access mode of one variable after
 * mpi_netcdf_enddef().  Independent access is illegal for variables that carry an
 * HDF5 filter (compression), so only use it for uncompressed variables. */
int mpi_netcdf_set_access(mpi_netcdf_file *f, const char *var, int access_mode);

/* COLLECTIVE when access_mode is MPI_NETCDF_COLLECTIVE.  Write the hyperslab
 * start[ndims]/count[ndims] of variable var from buf.  A rank that owns no
 * part of the variable passes a count containing a zero; it still has to call
 * the function so that the collective completes. */
int mpi_netcdf_write(mpi_netcdf_file *f, const char *var, const size_t *start,
              const size_t *count, const void *buf);

/* COLLECTIVE. Flush and close. */
int mpi_netcdf_close(mpi_netcdf_file *f);

#ifdef __cplusplus
}
#endif
#endif /* MPI_NETCDF_H */
