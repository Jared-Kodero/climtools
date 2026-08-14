#!/usr/bin/env bash
#
# Build the native MPI-NetCDF extension used by climtools.lib_netcdf.parallel.
#
# Run it with no arguments and it will work out the rest:
#
#     ./install.sh
#
# The script handles three environments without being told which one it is in:
#
#   1. An HPC system with Lmod or Environment Modules, where the parallel
#      NetCDF-C stack has to be loaded first.
#   2. A system where mpicc and a parallel NetCDF-C are already on PATH, such
#      as a conda environment or a Debian container. No module system needed.
#   3. Either of the above with the compiler flags supplied directly through
#      MPI_NETCDF_CFLAGS and MPI_NETCDF_LIBS.
#
# Capability is established by compiling and running code, never by parsing
# the output of a tool. Distribution packages exist whose nc-config is absent
# and whose netcdf.pc reports has_parallel="" while the installed header
# defines NC_HAS_PARALLEL4 1, so string matching gives the wrong answer.
#
# Run `./install.sh --help` for options and environment variables.

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
PARALLEL_FILTERS=""
PROBE_RESULT="not run"
STEP=0
TOTAL_STEPS=7
FORCE=0
declare -a CFLAGS_LIST=()
declare -a LIBS_LIST=()

# Colour only when writing to a terminal, so redirected logs stay readable.
if [[ -t 1 ]]; then
    C_BOLD=$'\e[1m'; C_DIM=$'\e[2m'; C_RED=$'\e[31m'
    C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_OFF=$'\e[0m'
else
    C_BOLD=""; C_DIM=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_OFF=""
fi

step() {
    STEP=$((STEP + 1))
    printf '\n%s[%d/%d] %s%s\n' "$C_BOLD" "$STEP" "$TOTAL_STEPS" "$*" "$C_OFF"
}
log()  { printf '      %s\n' "$*"; }
info() { printf '      %s%s%s\n' "$C_DIM" "$*" "$C_OFF"; }
ok()   { printf '      %s[ok]%s %s\n' "$C_GREEN" "$C_OFF" "$*"; }
warn() { printf '      %s[!]%s %s\n' "$C_YELLOW" "$C_OFF" "$*" >&2; }
die()  {
    printf '\n%sERROR%s %s\n\n' "$C_RED" "$C_OFF" "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Build the native MPI-NetCDF extension for climtools.

Usage:
  ./install.sh [options]

Options:
  -h, --help     Show this message and exit.
  -f, --force    Rebuild even if an up-to-date library is already present.
  -c, --clean    Remove the build directory and the compiled library, then exit.

Environment variables:
  MPI_NETCDF_MODULE           NetCDF-C module to load instead of searching
  MPI_NETCDF_MPI_MODULE       MPI module to load instead of searching
  MPI_NETCDF_PYTHON_MODULE    Python module to load before building
  MPI_NETCDF_PYTHON           absolute path to the Python interpreter
  MPI_NETCDF_CFLAGS           compiler flags, overriding discovery
  MPI_NETCDF_LIBS             linker flags, overriding discovery
  MPI_NETCDF_LAUNCHER         two-rank launcher, e.g. "mpirun -n 2"
  MPI_NETCDF_SKIP_PROBE       set to 1 to skip the two-rank runtime probe

Examples:
  ./install.sh
  MPI_NETCDF_LAUNCHER='srun --mpi=pmix --ntasks=2' ./install.sh
  MPI_NETCDF_MPI_MODULE=hpcx-mpi/2.25.1s MPI_NETCDF_MODULE=netcdf-c-mpi/4.9.3 ./install.sh

For the strongest verification, run inside an allocation that can launch two
MPI ranks, for example:

  salloc --nodes=1 --ntasks=2 --time=00:10:00 --mem-per-cpu=1G
  ./install.sh
  exit
USAGE
}

parse_arguments() {
    while (($#)); do
        case "$1" in
            -h|--help)  usage; exit 0 ;;
            -f|--force) FORCE=1 ;;
            -c|--clean)
                rm -rf "$BUILD_DIR" "$LIB_DIR"
                printf 'Removed %s and %s\n' "$BUILD_DIR" "$LIB_DIR"
                exit 0 ;;
            *)
                usage >&2
                die "unknown argument: $1"
                ;;
        esac
        shift
    done
}

cleanup() {
    rm -f "$BUILD_DIR"/*.$$ "$BUILD_DIR"/*.$$.c 2>/dev/null || true
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
            # is a known packaging fault; recover the real include and library
            # directories from the prefix.
            local prefix
            prefix=$(pkg-config --variable=prefix netcdf 2>/dev/null || true)
            if [[ -n "$prefix" && -r "$prefix/include/netcdf_par.h" ]]; then
                CFLAGS_LIST=("-I$prefix/include" "${CFLAGS_LIST[@]}")
            fi
            if [[ -n "$prefix" && -d "$prefix/lib" ]]; then
                LIBS_LIST=("-L$prefix/lib" "${LIBS_LIST[@]}")
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
        info "trying $netcdf_module"
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
            info "trying $mpi_module + $netcdf_module"
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
        ok "using the toolchain already on PATH"
        return
    fi

    if ((MODULES_AVAILABLE)); then
        log "no usable toolchain on PATH; searching the module system"
        if load_module_stack; then
            ok "loaded ${SELECTED_MPI_MODULE:+$SELECTED_MPI_MODULE + }$SELECTED_NETCDF_MODULE"
            return
        fi
    fi

    die "no parallel NetCDF-4 toolchain found.

Provide one in any of these ways:
  - load the appropriate modules, for example
      module load netcdf-c-mpi
  - activate an environment containing mpicc and a NetCDF-C built with
    NC_HAS_PARALLEL4, for example
      conda install -c conda-forge 'netcdf4=*=mpi_openmpi*' openmpi
  - install the system packages, for example
      apt install libopenmpi-dev libnetcdf-mpi-dev
  - set the flags directly:
      MPI_NETCDF_CFLAGS='-I...' MPI_NETCDF_LIBS='-L... -lnetcdf' ./install.sh"
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
        PYTHON_EXECUTABLE=$(command -v python3 || command -v python || true)
    fi

    [[ -n "$PYTHON_EXECUTABLE" && -x "$PYTHON_EXECUTABLE" ]] \
        || die "no usable Python interpreter found. Activate an environment, or set MPI_NETCDF_PYTHON to an absolute path."

    "$PYTHON_EXECUTABLE" - <<'PY' || die "install the Python dependencies first: python -m pip install numpy xarray netCDF4"
import sys

import numpy
import xarray

print(f"      NumPy {numpy.__version__}, xarray {xarray.__version__}")
print(f"      Python {sys.version.split()[0]} at {sys.executable}")
PY
    ok "Python dependencies satisfied"
}

verify_sources() {
    local source missing=0
    for source in \
        "$SOURCE_DIR/mpi_netcdf.c" \
        "$SOURCE_DIR/mpi_netcdf.h" \
        "$SOURCE_DIR/verify_parallel_netcdf.c" \
        "$PACKAGE_DIR/__init__.py" \
        "$PACKAGE_DIR/native.py" \
        "$PACKAGE_DIR/runtime.py" \
        "$PACKAGE_DIR/module_env.py" \
        "$PACKAGE_PARENT/lib_netcdf/parallel.py" \
        "$PACKAGE_PARENT/lib_netcdf/serial.py"; do
        if [[ ! -f "$source" ]]; then
            warn "missing: $source"
            missing=1
        fi
    done
    ((missing == 0)) || die "the checkout is incomplete; re-clone the repository"
    ok "all sources present"
}

# ----------------------------------------------------------------------- build

compile_probe() {
    mpicc -O2 -Wall -Wextra -Wpedantic -std=c99 \
        "${CFLAGS_LIST[@]}" \
        "$SOURCE_DIR/verify_parallel_netcdf.c" \
        "${LIBS_LIST[@]}" \
        -o "$BUILD_DIR/verify_parallel_netcdf" \
        || die "the capability probe did not compile against the selected stack"
    ok "probe compiled"
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
        PROBE_RESULT="skipped at your request"
        warn "skipping the two-rank runtime probe (MPI_NETCDF_SKIP_PROBE=1)"
        return
    fi

    local -a launcher=()
    read -r -a launcher <<< "$(probe_launcher)"
    if ((${#launcher[@]} == 0)); then
        PROBE_RESULT="skipped, no launcher"
        warn "no two-rank launcher found; skipping the runtime probe. Set MPI_NETCDF_LAUNCHER to run it."
        return
    fi

    # Retries, in order: as asked; then oversubscribed, because a build host
    # with one core refuses to place two ranks and that is a scheduling limit
    # rather than a NetCDF capability limit; then as root, because Open MPI
    # refuses to run as root and build containers routinely are.
    local -a attempts=("${launcher[*]}")
    if [[ -z "${MPI_NETCDF_LAUNCHER:-}" ]] \
        && [[ "${launcher[0]}" == mpirun || "${launcher[0]}" == mpiexec ]]; then
        attempts+=("${launcher[0]} --oversubscribe -n 2")
        if [[ "$(id -u)" == "0" ]]; then
            attempts+=("${launcher[0]} --oversubscribe --allow-run-as-root -n 2")
        fi
    fi

    local attempt
    local -a argv=()
    for attempt in "${attempts[@]}"; do
        read -r -a argv <<< "$attempt"
        log "running the two-rank probe: $attempt"
        rm -f "$BUILD_DIR/probe.nc"
        if "${argv[@]}" "$BUILD_DIR/verify_parallel_netcdf" \
            "$BUILD_DIR/probe.nc" >/dev/null 2>&1; then
            PROBE_RESULT="passed with '$attempt'"
            ok "two-rank parallel write succeeded"
            rm -f "$BUILD_DIR/probe.nc"
            return
        fi
    done

    # Show the failure from the first attempt, so the reason is visible.
    read -r -a argv <<< "${attempts[0]}"
    rm -f "$BUILD_DIR/probe.nc"
    "${argv[@]}" "$BUILD_DIR/verify_parallel_netcdf" "$BUILD_DIR/probe.nc" || true

    die "the loaded stack failed a two-rank parallel NetCDF-4 write.

The output above is from the failing run. Either the MPI and HDF5 in this
environment were not built against each other, or the launcher cannot place
two ranks here. Set MPI_NETCDF_LAUNCHER to a working two-rank launcher, or
MPI_NETCDF_SKIP_PROBE=1 to build without the runtime check."
}

compile_library() {
    local temporary="$BUILD_DIR/libmpi_netcdf.so.$$"

    mpicc -O2 -g -fPIC -Wall -Wextra -Wpedantic -std=c99 \
        "${CFLAGS_LIST[@]}" \
        -I"$SOURCE_DIR" \
        -shared \
        -Wl,-soname,libmpi_netcdf.so \
        -Wl,-z,defs \
        -o "$temporary" \
        "$SOURCE_DIR/mpi_netcdf.c" \
        "${LIBS_LIST[@]}" \
        -lm \
        || die "compiling $LIBRARY failed"
    mv -f "$temporary" "$LIBRARY"
    ok "built $LIBRARY"

    if command -v ldd >/dev/null 2>&1 \
        && ldd "$LIBRARY" | grep -q 'not found'; then
        ldd "$LIBRARY" >&2
        die "$LIBRARY has unresolved shared-library dependencies"
    fi
    local symbol
    for symbol in mpi_netcdf_create mpi_netcdf_allgatherv_bytes mpi_netcdf_abi_version; do
        if command -v nm >/dev/null 2>&1 \
            && ! nm -D "$LIBRARY" | grep -q " $symbol\$"; then
            die "$LIBRARY does not export $symbol; the sources and this script are out of step"
        fi
    done
    ok "dynamic dependencies and exported ABI verified"
}

verify_python_binding() {
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        MPI_NETCDF_LIBRARY="$LIBRARY" \
        "$PYTHON_EXECUTABLE" - "$PACKAGE_PARENT" <<'PY' \
        || die "the compiled library did not load through its Python binding"
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
    raise SystemExit("      the native library did not load after installation")
if configuration["size"] != 1:
    raise SystemExit(f"      unexpected import-check world size: {configuration['size']}")
if configuration["abi"] != configuration["abi_expected"]:
    raise SystemExit(
        f"      ABI mismatch: library reports {configuration['abi']}, "
        f"Python layer expects {configuration['abi_expected']}"
    )
print(f"      NetCDF-C {configuration['netcdf']}")
print(f"      ABI version {configuration['abi']}, thread level {configuration['thread_level']}")
PY
    ok "Python binding loads the compiled library"
}

write_manifest() {
    local temporary="$BUILD_DIR/build.yml.$$"
    local abi
    abi=$(sed -n 's/^#define MPI_NETCDF_ABI_VERSION \([0-9]*\).*/\1/p' \
        "$SOURCE_DIR/mpi_netcdf.h" | head -n 1)

    {
        printf '# Written by lib_mpi/install.sh. Read at runtime by module_env.py\n'
        printf '# to reload the same module stack the library was built against.\n'
        printf 'schema_version: 3\n'
        printf 'built_at: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf 'library: %s\n' "$LIBRARY"
        printf 'abi_version: %s\n' "${abi:-unknown}"
        printf 'flag_source: %s\n' "$FLAG_SOURCE"
        printf 'parallel_filters: %s\n' "$PARALLEL_FILTERS"
        printf 'probe: %s\n' "$PROBE_RESULT"
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
    ok "recorded the verified configuration in $MANIFEST"
}

summary() {
    printf '\n%s%s%s\n' "$C_BOLD" "Installation complete" "$C_OFF"
    printf '  library            %s\n' "$LIBRARY"
    printf '  flags from         %s\n' "$FLAG_SOURCE"
    printf '  mpicc              %s\n' "$(command -v mpicc)"
    printf '  python             %s\n' "$PYTHON_EXECUTABLE"
    [[ -n "$SELECTED_NETCDF_MODULE" ]] && printf '  netcdf module      %s\n' "$SELECTED_NETCDF_MODULE"
    [[ -n "$SELECTED_MPI_MODULE" ]]    && printf '  mpi module         %s\n' "$SELECTED_MPI_MODULE"
    [[ -n "$SELECTED_PYTHON_MODULE" ]] && printf '  python module      %s\n' "$SELECTED_PYTHON_MODULE"
    printf '  parallel filters   %s\n' "$PARALLEL_FILTERS"
    printf '  two-rank probe     %s\n' "$PROBE_RESULT"
    printf '  manifest           %s\n' "$MANIFEST"

    if [[ "$PARALLEL_FILTERS" != "yes" ]]; then
        printf '\n%sNote%s  This stack cannot apply compression during a collective write.\n' \
            "$C_YELLOW" "$C_OFF"
        printf '      Parallel output will be uncompressed; pass strict_compression=True\n'
        printf '      to make that an error instead of a warning.\n'
    fi

    printf '\nRun a job with:\n'
    printf '  mpirun -n 4 python your_script.py\n'
    printf '  srun --ntasks=4 --mpi=pmix python your_script.py\n'
    printf '\nUse the same MPI and NetCDF stack at runtime that was verified here.\n'
    printf 'See %s/README.md for how to write code that uses it.\n\n' "$PACKAGE_DIR"
}

main() {
    parse_arguments "$@"

    printf '%sBuilding the climtools MPI-NetCDF extension%s\n' "$C_BOLD" "$C_OFF"
    printf '%s%s%s\n' "$C_DIM" "$PACKAGE_DIR" "$C_OFF"

    mkdir -p "$BUILD_DIR" "$LIB_DIR"

    if ((FORCE == 0)) && [[ -f "$LIBRARY" && -f "$MANIFEST" ]] \
        && [[ "$LIBRARY" -nt "$SOURCE_DIR/mpi_netcdf.c" ]] \
        && [[ "$LIBRARY" -nt "$SOURCE_DIR/mpi_netcdf.h" ]]; then
        printf '\n%s is already up to date.\n' "$LIBRARY"
        printf 'Run with --force to rebuild, or --clean to remove it.\n\n'
        exit 0
    fi

    step "Locating a parallel NetCDF-4 toolchain"
    init_modules
    if ((MODULES_AVAILABLE)); then
        info "module system detected"
    else
        info "no module system; using PATH"
    fi
    resolve_toolchain
    PARALLEL_FILTERS=$(report_parallel_filters)
    info "flags from $FLAG_SOURCE"
    info "cflags: ${CFLAGS_LIST[*]:-none}"
    info "libs:   ${LIBS_LIST[*]:-none}"
    ok "parallel NetCDF-4 available (NC_HAS_PARALLEL4)"
    if [[ "$PARALLEL_FILTERS" == "yes" ]]; then
        ok "parallel HDF5 filters available"
    else
        warn "parallel HDF5 filters unavailable; collective writes cannot compress"
    fi

    step "Checking the Python environment"
    load_python

    step "Checking the checkout"
    verify_sources

    step "Compiling the capability probe"
    compile_probe

    step "Running the two-rank probe"
    run_probe

    step "Building the shared library"
    compile_library

    step "Verifying the Python binding"
    verify_python_binding
    write_manifest

    summary
}

main "$@"
