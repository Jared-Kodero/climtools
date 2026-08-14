#!/usr/bin/env bash
#
# Build the native MPI-NetCDF extension used by climtools.netcdf.parallel.
#
# The script works in three environments without being told which one it is in:
#
#   1. An HPC system with Lmod or Environment Modules, where the parallel
#      NetCDF-C stack has to be loaded first.
#   2. A system where mpicc and a parallel NetCDF-C are already on PATH, such
#      as a conda environment or a Debian container. No module system is
#      required.
#   3. Either of the above with the compiler flags supplied directly through
#      MPI_NETCDF_CFLAGS and MPI_NETCDF_LIBS.
#
# Capability is established by compiling and running code, never by parsing
# the output of a tool. Distribution packages exist whose nc-config is absent
# and whose netcdf.pc reports has_parallel="" while the installed header
# defines NC_HAS_PARALLEL4 1, so string matching gives the wrong answer.
#
# Environment variables
#   MPI_NETCDF_MODULE           NetCDF-C module to load instead of searching
#   MPI_NETCDF_MPI_MODULE       MPI module to load instead of searching
#   MPI_NETCDF_PYTHON_MODULE    Python module to load before building
#   MPI_NETCDF_PYTHON           absolute path to the Python interpreter
#   MPI_NETCDF_CFLAGS           compiler flags, overriding discovery
#   MPI_NETCDF_LIBS             linker flags, overriding discovery
#   MPI_NETCDF_LAUNCHER         two-rank launcher, e.g. "mpirun -n 2"
#   MPI_NETCDF_SKIP_PROBE       set to 1 to skip the two-rank runtime probe
#
set -Eeuo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PARENT="$(dirname -- "$PACKAGE_DIR")"
PROJECT_ROOT="$(dirname -- "$PACKAGE_PARENT")"
SOURCE_DIR="$PACKAGE_DIR/src"
BUILD_DIR="$PACKAGE_DIR/build"
LIB_DIR="$PACKAGE_DIR/lib"
LIBRARY="$LIB_DIR/libmpi_netcdf.so"
MANIFEST="$BUILD_DIR/build.yml"

SELECTED_MPI_MODULE=""
SELECTED_NETCDF_MODULE=""
SELECTED_PYTHON_MODULE=""
MODULES_AVAILABLE=0
FLAG_SOURCE=""
PYTHON_EXECUTABLE=""
declare -a CFLAGS_LIST=()
declare -a LIBS_LIST=()

log() { printf '[LIB MPI] %s\n' "$*"; }
warn() { printf '[LIB MPI] WARNING: %s\n' "$*" >&2; }
die() { printf '[LIB MPI] ERROR: %s\n' "$*" >&2; exit 1; }

cleanup() {
    rm -f "$BUILD_DIR"/*.$$ 2>/dev/null || true
}
trap cleanup EXIT

# --------------------------------------------------------------- module system

init_modules() {
    if type module >/dev/null 2>&1; then
        MODULES_AVAILABLE=1
        return
    fi

    local init_file
    for init_file in \
        /etc/profile.d/modules.sh \
        /etc/profile.d/lmod.sh \
        /usr/share/lmod/lmod/init/bash; do
        if [[ -r "$init_file" ]]; then
            # shellcheck source=/dev/null
            source "$init_file" >/dev/null 2>&1 || continue
            if type module >/dev/null 2>&1; then
                MODULES_AVAILABLE=1
                return
            fi
        fi
    done

    # Absence of a module system is not an error. It only means the toolchain
    # must already be on PATH, which is checked next.
    MODULES_AVAILABLE=0
}

module_names() {
    local query=$1
    module -t spider "$query" 2>&1 \
        | grep -Eo "${query}/[^[:space:]]+" \
        | sed 's/[,:;)]$//' \
        | sort -Vu
}

# --------------------------------------------------------------- flag discovery

discover_flags() {
    CFLAGS_LIST=()
    LIBS_LIST=()

    if [[ -n "${MPI_NETCDF_CFLAGS:-}" || -n "${MPI_NETCDF_LIBS:-}" ]]; then
        read -r -a CFLAGS_LIST <<< "${MPI_NETCDF_CFLAGS:-}"
        read -r -a LIBS_LIST <<< "${MPI_NETCDF_LIBS:-}"
        FLAG_SOURCE="environment"
        return 0
    fi

    if command -v nc-config >/dev/null 2>&1; then
        read -r -a CFLAGS_LIST <<< "$(nc-config --cflags 2>/dev/null || true)"
        read -r -a LIBS_LIST <<< "$(nc-config --libs 2>/dev/null || true)"
        if ((${#LIBS_LIST[@]})); then
            FLAG_SOURCE="nc-config"
            return 0
        fi
    fi

    if command -v pkg-config >/dev/null 2>&1; then
        local search
        # Distributions install the parallel build beside the serial one, so
        # the parallel pkgconfig directory is searched first.
        for search in \
            "/usr/lib/$(uname -m)-linux-gnu/netcdf/mpi/pkgconfig" \
            "/usr/lib64/netcdf/mpi/pkgconfig" \
            "/usr/lib/netcdf/mpi/pkgconfig"; do
            [[ -d "$search" ]] || continue
            PKG_CONFIG_PATH="$search${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
        done
        export PKG_CONFIG_PATH
        if pkg-config --exists netcdf 2>/dev/null; then
            read -r -a CFLAGS_LIST <<< "$(pkg-config --cflags netcdf)"
            read -r -a LIBS_LIST <<< "$(pkg-config --libs netcdf)"
            # A pkgconfig includedir that does not actually contain netcdf.h
            # is a known packaging fault; recover the real include directory
            # from the prefix.
            local prefix
            prefix=$(pkg-config --variable=prefix netcdf 2>/dev/null || true)
            if [[ -n "$prefix" && -r "$prefix/include/netcdf_par.h" ]]; then
                CFLAGS_LIST=("-I$prefix/include" "${CFLAGS_LIST[@]}")
            fi
            FLAG_SOURCE="pkg-config"
            return 0
        fi
    fi

    return 1
}

can_compile_headers() {
    local header
    for header in netcdf.h netcdf_par.h netcdf_meta.h mpi.h; do
        printf '#include <%s>\n' "$header" \
            | mpicc "${CFLAGS_LIST[@]}" -E -x c - >/dev/null 2>&1 \
            || return 1
    done
    return 0
}

has_parallel_macro() {
    # The authoritative test: the preprocessor either sees NC_HAS_PARALLEL4
    # set to a true value or it does not.
    local program='#include <netcdf_meta.h>
#if !defined(NC_HAS_PARALLEL4) || !NC_HAS_PARALLEL4
#error "no parallel netcdf4"
#endif
int main(void) { return 0; }'
    printf '%s\n' "$program" \
        | mpicc "${CFLAGS_LIST[@]}" -fsyntax-only -x c - >/dev/null 2>&1
}

parallel_stack_usable() {
    command -v mpicc >/dev/null 2>&1 || return 1
    discover_flags || return 1
    can_compile_headers || return 1
    has_parallel_macro || return 1
    return 0
}

report_parallel_filters() {
    local program='#include <stdio.h>
#include <netcdf_meta.h>
int main(void) {
#if defined(NC_HAS_PARALLEL_FILTERS) && NC_HAS_PARALLEL_FILTERS
    printf("yes\n");
#else
    printf("no\n");
#endif
    return 0;
}'
    printf '%s\n' "$program" > "$BUILD_DIR/filters.$$.c"
    if mpicc "${CFLAGS_LIST[@]}" "$BUILD_DIR/filters.$$.c" \
        -o "$BUILD_DIR/filters.$$" >/dev/null 2>&1; then
        "$BUILD_DIR/filters.$$"
    else
        printf 'unknown\n'
    fi
    rm -f "$BUILD_DIR/filters.$$.c" "$BUILD_DIR/filters.$$"
}

# ----------------------------------------------------------------- module stack

try_module_stack() {
    local mpi_module=${1:-}
    local netcdf_module=$2

    module purge >/dev/null 2>&1 || return 1
    [[ -z "$mpi_module" ]] \
        || module load "$mpi_module" >/dev/null 2>&1 \
        || return 1
    module load "$netcdf_module" >/dev/null 2>&1 || return 1
    parallel_stack_usable
}

load_module_stack() {
    local -a netcdf_candidates=()
    local -a mpi_candidates=()
    local candidate discovered mpi_module netcdf_module sorted_mpi sorted_netcdf

    if [[ -n "${MPI_NETCDF_MODULE:-}" ]]; then
        netcdf_candidates=("$MPI_NETCDF_MODULE")
    else
        for candidate in netcdf-c-mpi netcdf-mpi parallel-netcdf netcdf-c netcdf; do
            discovered=$(module_names "$candidate" || true)
            while IFS= read -r netcdf_module; do
                [[ -n "$netcdf_module" ]] && netcdf_candidates+=("$netcdf_module")
            done <<< "$discovered"
        done
    fi
    ((${#netcdf_candidates[@]})) || return 1

    if [[ -n "${MPI_NETCDF_MPI_MODULE:-}" ]]; then
        mpi_candidates=("$MPI_NETCDF_MPI_MODULE")
    else
        for candidate in hpcx-mpi openmpi mpich intel-mpi impi cray-mpich; do
            discovered=$(module_names "$candidate" || true)
            while IFS= read -r mpi_module; do
                [[ -n "$mpi_module" ]] && mpi_candidates+=("$mpi_module")
            done <<< "$discovered"
        done
    fi

    sorted_netcdf=$(printf '%s\n' "${netcdf_candidates[@]}" | sort -Vr)

    # A NetCDF module that pulls its own MPI dependency is preferred, because
    # HDF5 must be used with the MPI it was built against.
    while IFS= read -r netcdf_module; do
        [[ -n "$netcdf_module" ]] || continue
        if try_module_stack "" "$netcdf_module"; then
            SELECTED_MPI_MODULE=""
            SELECTED_NETCDF_MODULE=$netcdf_module
            return 0
        fi
    done <<< "$sorted_netcdf"

    ((${#mpi_candidates[@]})) || return 1
    sorted_mpi=$(printf '%s\n' "${mpi_candidates[@]}" | sort -Vr)

    while IFS= read -r netcdf_module; do
        [[ -n "$netcdf_module" ]] || continue
        while IFS= read -r mpi_module; do
            [[ -n "$mpi_module" ]] || continue
            if try_module_stack "$mpi_module" "$netcdf_module"; then
                SELECTED_MPI_MODULE=$mpi_module
                SELECTED_NETCDF_MODULE=$netcdf_module
                return 0
            fi
        done <<< "$sorted_mpi"
    done <<< "$sorted_netcdf"

    return 1
}

resolve_toolchain() {
    # The environment as the user left it is tried first, so that a working
    # conda or container toolchain is never discarded in favour of a module
    # search that may load a mismatched MPI.
    if parallel_stack_usable; then
        log "using the toolchain already on PATH"
        return
    fi

    if ((MODULES_AVAILABLE)); then
        log "searching the module system for a parallel NetCDF-C stack"
        if load_module_stack; then
            return
        fi
    fi

    die "no parallel NetCDF-4 toolchain found. Provide one by loading the \
appropriate modules, activating an environment containing mpicc and a \
NetCDF-C built with NC_HAS_PARALLEL4, or setting MPI_NETCDF_CFLAGS and \
MPI_NETCDF_LIBS."
}

# ------------------------------------------------------------------ python side

load_python() {
    if [[ -n "${MPI_NETCDF_PYTHON_MODULE:-}" ]]; then
        ((MODULES_AVAILABLE)) \
            || die "MPI_NETCDF_PYTHON_MODULE requires Lmod or Environment Modules"
        module load "$MPI_NETCDF_PYTHON_MODULE" >/dev/null 2>&1 \
            || die "could not load Python module: $MPI_NETCDF_PYTHON_MODULE"
        SELECTED_PYTHON_MODULE=$MPI_NETCDF_PYTHON_MODULE
    fi

    if [[ -n "${MPI_NETCDF_PYTHON:-}" ]]; then
        [[ "$MPI_NETCDF_PYTHON" == /* ]] \
            || die "MPI_NETCDF_PYTHON must be an absolute path"
        PYTHON_EXECUTABLE=$MPI_NETCDF_PYTHON
    else
        PYTHON_EXECUTABLE=$(command -v python || true)
    fi

    [[ -n "$PYTHON_EXECUTABLE" && -x "$PYTHON_EXECUTABLE" ]] \
        || die "no usable Python interpreter found"

    "$PYTHON_EXECUTABLE" - <<'PY' || die "install numpy and xarray first"
import numpy
import xarray

print(f"[LIB MPI]NumPy {numpy.__version__}, xarray {xarray.__version__}")
PY
}

verify_sources() {
    local source
    for source in \
        "$SOURCE_DIR/mpi_netcdf.c" \
        "$SOURCE_DIR/mpi_netcdf.h" \
        "$SOURCE_DIR/verify_parallel_netcdf.c" \
        "$PACKAGE_DIR/__init__.py" \
        "$PACKAGE_DIR/native.py" \
        "$PACKAGE_DIR/runtime.py" \
        "$PACKAGE_PARENT/netcdf/parallel.py" \
        "$PACKAGE_PARENT/netcdf/serial.py"; do
        [[ -f "$source" ]] || die "missing source file: $source"
    done
}

# ----------------------------------------------------------------------- build

compile_probe() {
    log "compiling the two-rank capability probe"
    mpicc -O2 -Wall -Wextra -Wpedantic -std=c99 \
        "${CFLAGS_LIST[@]}" \
        "$SOURCE_DIR/verify_parallel_netcdf.c" \
        "${LIBS_LIST[@]}" \
        -o "$BUILD_DIR/verify_parallel_netcdf"
}

probe_launcher() {
    if [[ -n "${MPI_NETCDF_LAUNCHER:-}" ]]; then
        printf '%s' "$MPI_NETCDF_LAUNCHER"
    elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
        printf 'srun --mpi=pmix --ntasks=2'
    elif command -v mpirun >/dev/null 2>&1; then
        printf 'mpirun -n 2'
    elif command -v mpiexec >/dev/null 2>&1; then
        printf 'mpiexec -n 2'
    else
        printf ''
    fi
}

run_probe() {
    if [[ "${MPI_NETCDF_SKIP_PROBE:-0}" == "1" ]]; then
        warn "skipping the two-rank runtime probe at your request"
        return
    fi

    local -a launcher=()
    read -r -a launcher <<< "$(probe_launcher)"
    if ((${#launcher[@]} == 0)); then
        warn "no two-rank launcher found; skipping the runtime probe. Set \
MPI_NETCDF_LAUNCHER to run it."
        return
    fi

    log "running the capability probe with: ${launcher[*]}"
    rm -f "$BUILD_DIR/probe.nc"
    if "${launcher[@]}" \
        "$BUILD_DIR/verify_parallel_netcdf" \
        "$BUILD_DIR/probe.nc"; then
        log "two-rank parallel write succeeded"
        return
    fi

    # A build host with one core refuses to place two ranks. That is a
    # scheduling limit, not a NetCDF capability limit, so retry once with
    # oversubscription rather than failing an otherwise sound stack.
    local -a relaxed=()
    if [[ -z "${MPI_NETCDF_LAUNCHER:-}" ]] && [[ "${launcher[0]}" == mpirun || "${launcher[0]}" == mpiexec ]]; then
        relaxed=("${launcher[0]}" --oversubscribe -n 2)
        log "retrying the probe oversubscribed: ${relaxed[*]}"
        rm -f "$BUILD_DIR/probe.nc"
        if "${relaxed[@]}" \
            "$BUILD_DIR/verify_parallel_netcdf" \
            "$BUILD_DIR/probe.nc"; then
            log "two-rank parallel write succeeded (oversubscribed)"
            return
        fi
    fi

    die "the loaded stack failed a two-rank parallel NetCDF-4 write. Set \
MPI_NETCDF_LAUNCHER to a working two-rank launcher, or set \
MPI_NETCDF_SKIP_PROBE=1 to build without the runtime check."
}

compile_library() {
    local temporary="$BUILD_DIR/libmpi_netcdf.so.$$"

    log "building $LIBRARY"
    mpicc -O2 -g -fPIC -Wall -Wextra -Wpedantic -std=c99 \
        "${CFLAGS_LIST[@]}" \
        -I"$SOURCE_DIR" \
        -shared \
        -Wl,-soname,libmpi_netcdf.so \
        -Wl,-z,defs \
        -o "$temporary" \
        "$SOURCE_DIR/mpi_netcdf.c" \
        "${LIBS_LIST[@]}" \
        -lm
    mv -f "$temporary" "$LIBRARY"

    if command -v ldd >/dev/null 2>&1 \
        && ldd "$LIBRARY" | grep -q 'not found'; then
        ldd "$LIBRARY" >&2
        die "$LIBRARY has unresolved shared-library dependencies"
    fi
    if command -v nm >/dev/null 2>&1 \
        && ! nm -D "$LIBRARY" | grep -q ' mpi_netcdf_create$'; then
        die "$LIBRARY does not export the expected C ABI"
    fi
}

verify_python_binding() {
    log "checking the installed Python binding"
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        MPI_NETCDF_LIBRARY="$LIBRARY" \
        "$PYTHON_EXECUTABLE" - "$PACKAGE_PARENT" <<'PY'
import pathlib
import sys

package = pathlib.Path(sys.argv[1])
try:
    module = __import__(f"{package.name}.lib_mpi", fromlist=["lib_mpi"])
except ImportError:
    # Importing the whole package needs its plotting dependencies; fall back
    # to the standalone subpackage so the build check stays self-contained.
    sys.path.insert(0, str(package))
    import lib_mpi as module

configuration = module.info()
if not configuration["available"]:
    raise SystemExit("the native library did not load after installation")
if configuration["size"] != 1:
    raise SystemExit(f"unexpected import-check world size: {configuration['size']}")
print(f"[LIB MPI]Python binding: {configuration}")
PY
}

write_manifest() {
    local temporary="$BUILD_DIR/build.yml.$$"
    local filters
    filters=$(report_parallel_filters)

    {
        printf 'schema_version: 2\n'
        printf 'library: %s\n' "$LIBRARY"
        printf 'flag_source: %s\n' "$FLAG_SOURCE"
        printf 'parallel_filters: %s\n' "$filters"
        printf 'python: %s\n' "$PYTHON_EXECUTABLE"
        printf 'mpicc: %s\n' "$(command -v mpicc)"
        printf 'cflags: %s\n' "${CFLAGS_LIST[*]}"
        printf 'libs: %s\n' "${LIBS_LIST[*]}"
        printf 'modules:\n'
        if [[ -n "$SELECTED_MPI_MODULE" ]]; then
            printf '  - %s\n' "$SELECTED_MPI_MODULE"
        fi
        if [[ -n "$SELECTED_NETCDF_MODULE" ]]; then
            printf '  - %s\n' "$SELECTED_NETCDF_MODULE"
        fi
        if [[ -n "$SELECTED_PYTHON_MODULE" ]]; then
            printf '  - %s\n' "$SELECTED_PYTHON_MODULE"
        fi
        if [[ -z "$SELECTED_MPI_MODULE$SELECTED_NETCDF_MODULE$SELECTED_PYTHON_MODULE" ]]; then
            printf '  []\n'
        fi
        printf 'loaded_modules: %s\n' "${LOADEDMODULES:-}"
    } > "$temporary"

    mv -f "$temporary" "$MANIFEST"
    log "recorded the verified build configuration: $MANIFEST"
}

main() {
    (($# == 0)) || die "install.sh accepts no arguments; configure the build with MPI_NETCDF_* environment variables"

    mkdir -p "$BUILD_DIR" "$LIB_DIR"

    init_modules
    resolve_toolchain
    load_python
    verify_sources

    log "flags from: $FLAG_SOURCE"
    log "mpicc: $(command -v mpicc)"
    log "Python: $PYTHON_EXECUTABLE"
    [[ -n "$SELECTED_NETCDF_MODULE" ]] && log "NetCDF module: $SELECTED_NETCDF_MODULE"
    [[ -n "$SELECTED_MPI_MODULE" ]] && log "MPI module: $SELECTED_MPI_MODULE"
    [[ -n "$SELECTED_PYTHON_MODULE" ]] && log "Python module: $SELECTED_PYTHON_MODULE"
    log "parallel NetCDF-4: yes (NC_HAS_PARALLEL4)"
    log "parallel HDF5 filters: $(report_parallel_filters)"

    compile_probe
    run_probe
    compile_library
    verify_python_binding
    write_manifest

    log "installation complete: $LIBRARY"
}



main "$@"

