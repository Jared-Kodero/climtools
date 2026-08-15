/* mpi_netcdf.c - MPI-parallel NetCDF-4 writer, C ABI for ctypes.
 *
 * Build: see install.sh.  Requires a NetCDF-C built with "NC-4 Parallel
 * Support: yes" against a parallel HDF5, and the MPI stack that HDF5 was
 * built with.
 */
#define _GNU_SOURCE
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <netcdf.h>
#include <netcdf_par.h>
#include <netcdf_meta.h>

#include "mpi_netcdf.h"

/* ------------------------------------------------------------------ state */

static int g_rank = -1;
static int g_size = -1;
static int g_thread_level = -1;
static int g_we_initialised_mpi = 0;
static char g_err[2048] = "";

typedef struct {
    char name[NC_MAX_NAME + 1];
    int varid;
    int xtype;
    int ndims;
} mpi_netcdf_var;

struct mpi_netcdf_file {
    int ncid;
    int defining;
    mpi_netcdf_var *vars;
    size_t nvars;
    size_t cap;
    MPI_Info info;
};

static void set_err(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

#define NCCHECK(call, what)                                                   \
    do {                                                                      \
        int _e = (call);                                                      \
        if (_e != NC_NOERR) {                                                 \
            set_err("rank %d: %s failed: %s", g_rank, (what), nc_strerror(_e));\
            return -1;                                                        \
        }                                                                     \
    } while (0)

const char *mpi_netcdf_strerror(void) { return g_err; }

const char *mpi_netcdf_version(void) { return nc_inq_libvers(); }

int mpi_netcdf_abi_version(void) { return MPI_NETCDF_ABI_VERSION; }



int mpi_netcdf_has_parallel_filters(void)
{
#if defined(NC_HAS_PARALLEL4) && defined(NC_HAS_PAR_FILTERS)
    return NC_HAS_PARALLEL4 && NC_HAS_PAR_FILTERS;
#else
    return 0;
#endif
}

/* ------------------------------------------------------- process handling */

int mpi_netcdf_init(void)
{
    int initialised = 0, finalised = 0;
    if (MPI_Initialized(&initialised) != MPI_SUCCESS) {
        set_err("MPI_Initialized failed");
        return -1;
    }
    if (MPI_Finalized(&finalised) != MPI_SUCCESS) {
        set_err("MPI_Finalized failed");
        return -1;
    }
    if (finalised) {
        set_err("MPI has already been finalized");
        return -1;
    }
    if (!initialised) {
        /* MPI_THREAD_MULTIPLE so that a threaded Dask scheduler in the same
         * process cannot corrupt MPI state.  The level actually granted is
         * reported by mpi_netcdf_thread_level(). */
        int argc = 0;
        char **argv = NULL;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE,
                            &g_thread_level) != MPI_SUCCESS) {
            set_err("MPI_Init_thread failed");
            return -1;
        }
        g_we_initialised_mpi = 1;
    } else if (MPI_Query_thread(&g_thread_level) != MPI_SUCCESS) {
        set_err("MPI_Query_thread failed");
        return -1;
    }
    if (MPI_Comm_rank(MPI_COMM_WORLD, &g_rank) != MPI_SUCCESS) {
        set_err("MPI_Comm_rank failed");
        return -1;
    }
    if (MPI_Comm_size(MPI_COMM_WORLD, &g_size) != MPI_SUCCESS) {
        set_err("MPI_Comm_size failed");
        return -1;
    }
    g_err[0] = '\0';
    return 0;
}

int mpi_netcdf_rank(void) { return g_rank; }
int mpi_netcdf_size(void) { return g_size; }
int mpi_netcdf_thread_level(void) { return g_thread_level; }

int mpi_netcdf_consensus(int ok)
{
    int all = 0;
    int mine = ok ? 1 : 0;
    if (g_size <= 0 && mpi_netcdf_init() != 0) return 0;
    if (MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD)
        != MPI_SUCCESS) {
        set_err("MPI_Allreduce failed in mpi_netcdf_consensus");
        return 0;
    }
    return all;
}

int mpi_netcdf_barrier(void)
{
    if (g_size <= 0 && mpi_netcdf_init() != 0) return -1;
    if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) {
        set_err("MPI_Barrier failed");
        return -1;
    }
    return 0;
}

int mpi_netcdf_allgather_i64(long long value, long long *out)
{
    if (!out) {
        set_err("mpi_netcdf_allgather_i64 received a null output buffer");
        return -1;
    }
    if (g_size <= 0 && mpi_netcdf_init() != 0) return -1;
    if (MPI_Allgather(&value, 1, MPI_LONG_LONG, out, 1, MPI_LONG_LONG,
                      MPI_COMM_WORLD) != MPI_SUCCESS) {
        set_err("MPI_Allgather failed");
        return -1;
    }
    return 0;
}

int mpi_netcdf_allgatherv_bytes(const void *sendbuf, long long sendcount,
                                void *recvbuf, const long long *counts)
{
    int *icounts = NULL, *idispls = NULL;
    long long total = 0;
    int status = 0;

    if (!counts || sendcount < 0) {
        set_err("mpi_netcdf_allgatherv_bytes received invalid arguments");
        return -1;
    }
    if (g_size <= 0 && mpi_netcdf_init() != 0) return -1;

    for (int i = 0; i < g_size; i++) {
        if (counts[i] < 0) {
            set_err("mpi_netcdf_allgatherv_bytes: rank %d reported a negative "
                    "payload length", i);
            return -1;
        }
        /* Allgatherv counts and displacements are int-typed in MPI-2.  Report
         * the limit rather than truncating; the caller has a slower path. */
        if (counts[i] > INT_MAX || total > (long long)INT_MAX - counts[i]) {
            set_err("mpi_netcdf_allgatherv_bytes: total payload exceeds the "
                    "MPI_Allgatherv INT_MAX limit");
            return -1;
        }
        total += counts[i];
    }
    if (total > 0 && !recvbuf) {
        set_err("mpi_netcdf_allgatherv_bytes received a null receive buffer");
        return -1;
    }
    if (sendcount > 0 && !sendbuf) {
        set_err("mpi_netcdf_allgatherv_bytes received a null send buffer");
        return -1;
    }

    icounts = malloc((size_t)g_size * sizeof *icounts);
    idispls = malloc((size_t)g_size * sizeof *idispls);
    if (!icounts || !idispls) {
        free(icounts);
        free(idispls);
        set_err("out of memory in mpi_netcdf_allgatherv_bytes");
        return -1;
    }
    for (int i = 0, offset = 0; i < g_size; i++) {
        icounts[i] = (int)counts[i];
        idispls[i] = offset;
        offset += icounts[i];
    }

    if (MPI_Allgatherv(sendcount > 0 ? sendbuf : MPI_BOTTOM, (int)sendcount,
                       MPI_BYTE, recvbuf, icounts, idispls, MPI_BYTE,
                       MPI_COMM_WORLD) != MPI_SUCCESS) {
        set_err("MPI_Allgatherv failed");
        status = -1;
    }
    free(icounts);
    free(idispls);
    return status;
}

int mpi_netcdf_bcast_i64(long long *value, int root)
{
    if (!value) {
        set_err("mpi_netcdf_bcast_i64 received a null value");
        return -1;
    }
    if (g_size <= 0 && mpi_netcdf_init() != 0) return -1;
    if (root < 0 || root >= g_size) {
        set_err("broadcast root %d is outside [0, %d)", root, g_size);
        return -1;
    }
    if (MPI_Bcast(value, 1, MPI_LONG_LONG, root, MPI_COMM_WORLD)
        != MPI_SUCCESS) {
        set_err("MPI_Bcast failed");
        return -1;
    }
    return 0;
}

int mpi_netcdf_bcast_bytes(void *buf, long long n, int root)
{
    /* MPI_Bcast counts are int; send in chunks so that a large pickled
     * object does not silently truncate. */
    const long long CHUNK = 1LL << 30;
    char *p = (char *)buf;
    if (g_size <= 0 && mpi_netcdf_init() != 0) return -1;
    if (root < 0 || root >= g_size) {
        set_err("broadcast root %d is outside [0, %d)", root, g_size);
        return -1;
    }
    if (n < 0 || (n > 0 && !buf)) {
        set_err("mpi_netcdf_bcast_bytes received an invalid buffer or size");
        return -1;
    }
    while (n > 0) {
        int part = (int)(n < CHUNK ? n : CHUNK);
        if (MPI_Bcast(p, part, MPI_BYTE, root, MPI_COMM_WORLD) != MPI_SUCCESS) {
            set_err("MPI_Bcast failed");
            return -1;
        }
        p += part;
        n -= part;
    }
    return 0;
}

void mpi_netcdf_abort(int code)
{
    int initialised = 0, finalised = 0;
    MPI_Initialized(&initialised);
    MPI_Finalized(&finalised);
    if (initialised && !finalised) MPI_Abort(MPI_COMM_WORLD, code);
    exit(code);
}

int mpi_netcdf_finalize(void)
{
    int initialised = 0, finalised = 0;
    if (MPI_Initialized(&initialised) != MPI_SUCCESS) {
        set_err("MPI_Initialized failed during finalization");
        return -1;
    }
    if (MPI_Finalized(&finalised) != MPI_SUCCESS) {
        set_err("MPI_Finalized failed during finalization");
        return -1;
    }
    if (initialised && !finalised && g_we_initialised_mpi &&
        MPI_Finalize() != MPI_SUCCESS) {
        set_err("MPI_Finalize failed");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------ MPI-IO hints */

static int build_info(const char *hints, MPI_Info *out)
{
    *out = MPI_INFO_NULL;
    if (MPI_Info_create(out) != MPI_SUCCESS) {
        set_err("MPI_Info_create failed");
        return -1;
    }
    if (!hints || !*hints) return 0;

    char *buf = strdup(hints);
    if (!buf) {
        MPI_Info_free(out);
        set_err("out of memory while parsing MPI-IO hints");
        return -1;
    }
    char *save = NULL;
    for (char *tok = strtok_r(buf, ";", &save); tok;
         tok = strtok_r(NULL, ";", &save)) {
        char *eq = strchr(tok, '=');
        if (!eq) {
            set_err("invalid MPI-IO hint '%s'; expected key=value", tok);
            free(buf);
            MPI_Info_free(out);
            return -1;
        }
        *eq = '\0';
        while (*tok == ' ') tok++;
        char *key_end = tok + strlen(tok);
        char *value = eq + 1;
        while (key_end > tok && key_end[-1] == ' ') *--key_end = '\0';
        while (*value == ' ') value++;
        if (!*tok || !*value) {
            set_err("MPI-IO hints require non-empty keys and values");
            free(buf);
            MPI_Info_free(out);
            return -1;
        }
        if (MPI_Info_set(*out, tok, value) != MPI_SUCCESS) {
            set_err("MPI_Info_set failed for hint '%s'", tok);
            free(buf);
            MPI_Info_free(out);
            return -1;
        }
    }
    free(buf);
    return 0;
}

/* ------------------------------------------------------- file and metadata */

mpi_netcdf_file *mpi_netcdf_create(const char *path, int nofill, const char *hints)
{
    if (g_rank < 0 && mpi_netcdf_init() != 0) return NULL;
    if (!path || !*path) {
        set_err("output path is empty");
        return NULL;
    }

    mpi_netcdf_file *f = calloc(1, sizeof *f);
    if (!f) { set_err("out of memory"); return NULL; }

    if (build_info(hints, &f->info) != 0) {
        free(f);
        return NULL;
    }
    f->cap = 32;
    f->vars = calloc(f->cap, sizeof *f->vars);
    if (!f->vars) {
        if (f->info != MPI_INFO_NULL) MPI_Info_free(&f->info);
        free(f);
        set_err("out of memory");
        return NULL;
    }

    int e = nc_create_par(path, NC_NETCDF4 | NC_CLOBBER, MPI_COMM_WORLD,
                          f->info, &f->ncid);
    if (e != NC_NOERR) {
        set_err("rank %d: nc_create_par('%s') failed: %s", g_rank, path,
                nc_strerror(e));
        if (f->info != MPI_INFO_NULL) MPI_Info_free(&f->info);
        free(f->vars);
        free(f);
        return NULL;
    }
    if (nofill) {
        e = nc_set_fill(f->ncid, NC_NOFILL, NULL);
        if (e != NC_NOERR) {
            set_err("rank %d: nc_set_fill failed: %s", g_rank, nc_strerror(e));
            nc_close(f->ncid);
            if (f->info != MPI_INFO_NULL) MPI_Info_free(&f->info);
            free(f->vars);
            free(f);
            return NULL;
        }
    }
    f->defining = 1;
    return f;
}

static mpi_netcdf_var *find_var(mpi_netcdf_file *f, const char *name)
{
    for (size_t i = 0; i < f->nvars; i++)
        if (strcmp(f->vars[i].name, name) == 0) return &f->vars[i];
    return NULL;
}

static int supported_type(int xtype)
{
    switch (xtype) {
    case NC_BYTE:
    case NC_CHAR:
    case NC_SHORT:
    case NC_INT:
    case NC_FLOAT:
    case NC_DOUBLE:
    case NC_UBYTE:
    case NC_USHORT:
    case NC_UINT:
    case NC_INT64:
    case NC_UINT64:
        return 1;
    default:
        return 0;
    }
}

int mpi_netcdf_def_dim(mpi_netcdf_file *f, const char *name, size_t len, int unlimited)
{
    if (!f || !name || !*name) {
        set_err("mpi_netcdf_def_dim received an invalid file or name");
        return -1;
    }
    if (!f->defining) {
        set_err("mpi_netcdf_def_dim called outside define mode");
        return -1;
    }
    int dimid;
    NCCHECK(nc_def_dim(f->ncid, name, unlimited ? NC_UNLIMITED : len, &dimid),
            "nc_def_dim");
    return 0;
}

int mpi_netcdf_def_var(mpi_netcdf_file *f, const char *name, int xtype, int ndims,
                const char *const *dimnames, int deflate, int shuffle,
                const size_t *chunks)
{
    if (!f || !name || !*name || ndims < 0 || (ndims > 0 && !dimnames)) {
        set_err("mpi_netcdf_def_var received invalid arguments");
        return -1;
    }
    if (!f->defining) {
        set_err("mpi_netcdf_def_var called outside define mode");
        return -1;
    }
    if (!supported_type(xtype)) {
        set_err("variable '%s': unsupported type code %d", name, xtype);
        return -1;
    }
    if (deflate < -1 || deflate > 9) {
        set_err("variable '%s': deflate must be -1 or between 0 and 9", name);
        return -1;
    }
    if (deflate >= 0 && !mpi_netcdf_has_parallel_filters()) {
        set_err("variable '%s': parallel HDF5 filters are unavailable", name);
        return -1;
    }
    int dimids[NC_MAX_VAR_DIMS] = {0};
    if (ndims > NC_MAX_VAR_DIMS) {
        set_err("variable '%s' has %d dimensions, limit is %d", name, ndims,
                NC_MAX_VAR_DIMS);
        return -1;
    }
    for (int i = 0; i < ndims; i++) {
        if (!dimnames[i] || !*dimnames[i]) {
            set_err("variable '%s': dimension %d has no name", name, i);
            return -1;
        }
        if (chunks && chunks[i] == 0) {
            set_err("variable '%s': chunk lengths must be positive", name);
            return -1;
        }
        NCCHECK(nc_inq_dimid(f->ncid, dimnames[i], &dimids[i]), "nc_inq_dimid");
    }

    int varid;
    NCCHECK(nc_def_var(f->ncid, name, xtype, ndims, dimids, &varid),
            "nc_def_var");

    if (ndims > 0 && (chunks || deflate >= 0)) {
        if (chunks)
            NCCHECK(nc_def_var_chunking(f->ncid, varid, NC_CHUNKED, chunks),
                    "nc_def_var_chunking");
        if (deflate >= 0)
            NCCHECK(nc_def_var_deflate(f->ncid, varid, shuffle ? 1 : 0, 1,
                                       deflate),
                    "nc_def_var_deflate");
    }

    if (f->nvars == f->cap) {
        size_t cap = f->cap * 2;
        mpi_netcdf_var *p = realloc(f->vars, cap * sizeof *p);
        if (!p) { set_err("out of memory"); return -1; }
        f->vars = p;
        f->cap = cap;
    }
    mpi_netcdf_var *v = &f->vars[f->nvars++];
    snprintf(v->name, sizeof v->name, "%s", name);
    v->varid = varid;
    v->xtype = xtype;
    v->ndims = ndims;
    return 0;
}

static int resolve_varid(mpi_netcdf_file *f, const char *var, int *varid)
{
    if (!var || !*var) { *varid = NC_GLOBAL; return 0; }
    mpi_netcdf_var *v = find_var(f, var);
    if (!v) { set_err("unknown variable '%s'", var); return -1; }
    *varid = v->varid;
    return 0;
}

int mpi_netcdf_put_att_text(mpi_netcdf_file *f, const char *var, const char *name,
                     const char *value)
{
    if (!f || !name || !*name || !value) {
        set_err("mpi_netcdf_put_att_text received invalid arguments");
        return -1;
    }
    if (!f->defining) {
        set_err("mpi_netcdf_put_att_text called outside define mode");
        return -1;
    }
    int varid;
    if (resolve_varid(f, var, &varid) != 0) return -1;
    NCCHECK(nc_put_att_text(f->ncid, varid, name, strlen(value), value),
            "nc_put_att_text");
    return 0;
}

int mpi_netcdf_put_att_num(mpi_netcdf_file *f, const char *var, const char *name, int xtype,
                    size_t n, const void *values)
{
    if (!f || !name || !*name || (n > 0 && !values)) {
        set_err("mpi_netcdf_put_att_num received invalid arguments");
        return -1;
    }
    if (!f->defining) {
        set_err("mpi_netcdf_put_att_num called outside define mode");
        return -1;
    }
    if (!supported_type(xtype) || xtype == NC_CHAR) {
        set_err("attribute '%s': unsupported numeric type code %d", name, xtype);
        return -1;
    }
    int varid;
    if (resolve_varid(f, var, &varid) != 0) return -1;
    NCCHECK(nc_put_att(f->ncid, varid, name, xtype, n, values), "nc_put_att");
    return 0;
}

int mpi_netcdf_enddef(mpi_netcdf_file *f, int access_mode)
{
    if (!f) { set_err("mpi_netcdf_enddef received a null file"); return -1; }
    if (!f->defining) {
        set_err("mpi_netcdf_enddef called outside define mode");
        return -1;
    }
    if (access_mode != MPI_NETCDF_INDEPENDENT &&
        access_mode != MPI_NETCDF_COLLECTIVE) {
        set_err("invalid parallel access mode %d", access_mode);
        return -1;
    }
    NCCHECK(nc_enddef(f->ncid), "nc_enddef");
    f->defining = 0;
    int mode = (access_mode == MPI_NETCDF_INDEPENDENT) ? NC_INDEPENDENT : NC_COLLECTIVE;
    for (size_t i = 0; i < f->nvars; i++)
        NCCHECK(nc_var_par_access(f->ncid, f->vars[i].varid, mode),
                "nc_var_par_access");
    return 0;
}

int mpi_netcdf_set_access(mpi_netcdf_file *f, const char *var, int access_mode)
{
    if (!f || !var || !*var) {
        set_err("mpi_netcdf_set_access received invalid arguments");
        return -1;
    }
    if (f->defining) {
        set_err("mpi_netcdf_set_access called before mpi_netcdf_enddef");
        return -1;
    }
    if (access_mode != MPI_NETCDF_INDEPENDENT &&
        access_mode != MPI_NETCDF_COLLECTIVE) {
        set_err("invalid parallel access mode %d", access_mode);
        return -1;
    }
    mpi_netcdf_var *v = find_var(f, var);
    if (!v) { set_err("unknown variable '%s'", var); return -1; }
    int mode = (access_mode == MPI_NETCDF_INDEPENDENT) ? NC_INDEPENDENT : NC_COLLECTIVE;
    NCCHECK(nc_var_par_access(f->ncid, v->varid, mode), "nc_var_par_access");
    return 0;
}

/* --------------------------------------------------------------- data path */

int mpi_netcdf_write(mpi_netcdf_file *f, const char *var, const size_t *start,
              const size_t *count, const void *buf)
{
    if (!f || !var || !*var) {
        set_err("mpi_netcdf_write received invalid arguments");
        return -1;
    }
    mpi_netcdf_var *v = find_var(f, var);
    if (!v) { set_err("unknown variable '%s'", var); return -1; }
    if (f->defining) { set_err("mpi_netcdf_write before mpi_netcdf_enddef"); return -1; }
    if (v->ndims > 0 && (!start || !count)) {
        set_err("variable '%s': start and count are required", var);
        return -1;
    }
    int has_data = 1;
    for (int i = 0; i < v->ndims; i++)
        if (count[i] == 0) { has_data = 0; break; }
    if (has_data && !buf) {
        set_err("variable '%s': data buffer is null", var);
        return -1;
    }

    int e;
    switch (v->xtype) {
    case NC_DOUBLE: e = nc_put_vara_double(f->ncid, v->varid, start, count, buf); break;
    case NC_FLOAT:  e = nc_put_vara_float (f->ncid, v->varid, start, count, buf); break;
    case NC_INT64:  e = nc_put_vara_longlong (f->ncid, v->varid, start, count, buf); break;
    case NC_UINT64: e = nc_put_vara_ulonglong(f->ncid, v->varid, start, count, buf); break;
    case NC_INT:    e = nc_put_vara_int   (f->ncid, v->varid, start, count, buf); break;
    case NC_UINT:   e = nc_put_vara_uint  (f->ncid, v->varid, start, count, buf); break;
    case NC_SHORT:  e = nc_put_vara_short (f->ncid, v->varid, start, count, buf); break;
    case NC_USHORT: e = nc_put_vara_ushort(f->ncid, v->varid, start, count, buf); break;
    case NC_BYTE:   e = nc_put_vara_schar (f->ncid, v->varid, start, count, buf); break;
    case NC_UBYTE:  e = nc_put_vara_uchar (f->ncid, v->varid, start, count, buf); break;
    case NC_CHAR:   e = nc_put_vara_text  (f->ncid, v->varid, start, count, buf); break;
    default:
        set_err("variable '%s': unsupported type code %d", var, v->xtype);
        return -1;
    }
    if (e != NC_NOERR) {
        set_err("rank %d: nc_put_vara('%s') failed: %s", g_rank, var,
                nc_strerror(e));
        return -1;
    }
    return 0;
}

int mpi_netcdf_close(mpi_netcdf_file *f)
{
    if (!f) return 0;
    int e = nc_close(f->ncid);
    if (f->info != MPI_INFO_NULL) MPI_Info_free(&f->info);
    free(f->vars);
    free(f);
    if (e != NC_NOERR) {
        set_err("rank %d: nc_close failed: %s", g_rank, nc_strerror(e));
        return -1;
    }
    return 0;
}
