#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PARENT="$(dirname -- "$PACKAGE_DIR")"
BUILD_DIR="$PACKAGE_DIR/build"
PACKAGE_LIB_DIR="$PACKAGE_DIR/lib"
PACKAGE_LIBRARY="$PACKAGE_LIB_DIR/libmpi_netcdf.so"
MODULE_MANIFEST="$PACKAGE_DIR/modules.yml"

log() { printf '[mpi] %s\n' "$*"; }
die() { printf '[mpi] ERROR: %s\n' "$*" >&2; exit 1; }

init_modules() {
    if type module >/dev/null 2>&1; then
        return
    fi

    local init_file
    for init_file in \
        /etc/profile.d/modules.sh \
        /etc/profile.d/lmod.sh \
        /usr/share/lmod/lmod/init/bash; do
        if [[ -r "$init_file" ]]; then
            # shellcheck source=/dev/null
            source "$init_file"
            type module >/dev/null 2>&1 && return
        fi
    done
    die "Lmod or Environment Modules is unavailable in this shell"
}

module_names() {
    local query=$1
    module -t spider "$query" 2>&1 \
        | grep -Eo "${query}/[^[:space:]]+" \
        | sed 's/[,:;)]$//' \
        | sort -Vu
}

parallel_config_available() {
    command -v mpicc >/dev/null 2>&1 || return 1
    command -v nc-config >/dev/null 2>&1 || return 1
    [[ "$(nc-config --has-nc4 2>/dev/null || true)" == yes ]] || return 1
    [[ "$(nc-config --has-parallel4 2>/dev/null || true)" == yes ]] || return 1
}

try_native_stack() {
    local mpi_module=${1:-}
    local netcdf_module=$2

    module purge >/dev/null 2>&1 || return 1
    [[ -z "$mpi_module" ]] \
        || module load "$mpi_module" >/dev/null 2>&1 \
        || return 1
    module load "$netcdf_module" >/dev/null 2>&1 || return 1
    parallel_config_available
}

load_native_stack() {
    local -a netcdf_candidates=()
    local -a mpi_candidates=()
    local candidate discovered mpi_module netcdf_module sorted_mpi sorted_netcdf

    if [[ -n "${MPI_PACKAGE_NETCDF_MODULE:-}" ]]; then
        netcdf_candidates=("$MPI_PACKAGE_NETCDF_MODULE")
    else
        for candidate in netcdf-c-mpi netcdf-mpi; do
            discovered=$(module_names "$candidate" || true)
            while IFS= read -r netcdf_module; do
                [[ -n "$netcdf_module" ]] \
                    && netcdf_candidates+=("$netcdf_module")
            done <<< "$discovered"
        done
    fi
    ((${#netcdf_candidates[@]})) \
        || die "no parallel NetCDF-C module found; set MPI_PACKAGE_NETCDF_MODULE"

    if [[ -n "${MPI_PACKAGE_MPI_MODULE:-}" ]]; then
        mpi_candidates=("$MPI_PACKAGE_MPI_MODULE")
    else
        for candidate in hpcx-mpi openmpi; do
            discovered=$(module_names "$candidate" || true)
            while IFS= read -r mpi_module; do
                [[ -n "$mpi_module" ]] && mpi_candidates+=("$mpi_module")
            done <<< "$discovered"
        done
    fi

    sorted_netcdf=$(printf '%s\n' "${netcdf_candidates[@]}" | sort -Vr)
    sorted_mpi=$(printf '%s\n' "${mpi_candidates[@]}" | sort -Vr)

    while IFS= read -r netcdf_module; do
        [[ -n "$netcdf_module" ]] || continue
        if try_native_stack "" "$netcdf_module"; then
            SELECTED_MPI_MODULE=""
            SELECTED_MPI_DISPLAY="dependency loaded by $netcdf_module"
            SELECTED_NETCDF_MODULE=$netcdf_module
            return
        fi
    done <<< "$sorted_netcdf"

    while IFS= read -r netcdf_module; do
        [[ -n "$netcdf_module" ]] || continue
        while IFS= read -r mpi_module; do
            [[ -n "$mpi_module" ]] || continue
            if try_native_stack "$mpi_module" "$netcdf_module"; then
                SELECTED_MPI_MODULE=$mpi_module
                SELECTED_MPI_DISPLAY=$mpi_module
                SELECTED_NETCDF_MODULE=$netcdf_module
                return
            fi
        done <<< "$sorted_mpi"
    done <<< "$sorted_netcdf"

    die "no module combination reports NetCDF-4 and parallel NetCDF-4 support"
}

load_python() {
    SELECTED_PYTHON_MODULE=${MPI_PACKAGE_PYTHON_MODULE:-}
    if [[ -n "$SELECTED_PYTHON_MODULE" ]]; then
        module load "$SELECTED_PYTHON_MODULE" >/dev/null 2>&1 \
            || die "cannot load Python module: $SELECTED_PYTHON_MODULE"
    fi

    if [[ -n "${MPI_PACKAGE_PYTHON_EXECUTABLE:-}" ]]; then
        [[ "$MPI_PACKAGE_PYTHON_EXECUTABLE" == /* ]] \
            || die "MPI_PACKAGE_PYTHON_EXECUTABLE must be an absolute path"
        PYTHON_EXECUTABLE=$MPI_PACKAGE_PYTHON_EXECUTABLE
    else
        PYTHON_EXECUTABLE=$(command -v python || true)
    fi

    [[ -n "$PYTHON_EXECUTABLE" && -x "$PYTHON_EXECUTABLE" ]] \
        || die "no Python executable found; activate its environment or set MPI_PACKAGE_PYTHON_EXECUTABLE"
}

verify_loaded_stack() {
    parallel_config_available \
        || die "the final module stack lacks parallel NetCDF-4 support"

    local -a cflags=()
    local header
    read -r -a cflags <<< "$(nc-config --cflags)"
    for header in netcdf.h netcdf_par.h netcdf_meta.h mpi.h; do
        if ! printf '#include <%s>\n' "$header" \
            | mpicc "${cflags[@]}" -E -x c - >/dev/null 2>&1; then
            die "the active compiler cannot include $header"
        fi
    done

    "$PYTHON_EXECUTABLE" - <<'PY' \
        || die "Python dependencies are missing; install numpy and xarray"
import numpy
import xarray

print(f"[mpi] NumPy: {numpy.__version__}")
print(f"[mpi] xarray: {xarray.__version__}")
PY
}

verify_sources() {
    local source
    for source in \
        __init__.py \
        native.py \
        runtime.py \
        parallel_netcdf.py \
        mpi_netcdf.c \
        mpi_netcdf.h \
        verify_parallel_netcdf.c \
        test_mpi.py; do
        [[ -f "$PACKAGE_DIR/$source" ]] \
            || die "missing source file: $PACKAGE_DIR/$source"
    done
}

compile_probe() {
    local -a cflags=()
    local -a libs=()
    read -r -a cflags <<< "$(nc-config --cflags)"
    read -r -a libs <<< "$(nc-config --libs)"

    log "compiling the two-rank parallel NetCDF-4 capability probe"
    mpicc -O2 -Wall -Wextra -Wpedantic -std=c99 \
        "${cflags[@]}" \
        "$PACKAGE_DIR/verify_parallel_netcdf.c" \
        "${libs[@]}" \
        -o "$BUILD_DIR/verify_parallel_netcdf"
}

run_probe() {
    local -a launcher=()
    if [[ -n "${MPI_PACKAGE_PROBE_LAUNCHER:-}" ]]; then
        read -r -a launcher <<< "$MPI_PACKAGE_PROBE_LAUNCHER"
    elif [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
        launcher=(srun --mpi=pmix --ntasks=2)
    elif command -v mpirun >/dev/null 2>&1; then
        launcher=(mpirun -n 2)
    elif command -v mpiexec >/dev/null 2>&1; then
        launcher=(mpiexec -n 2)
    else
        die "no two-rank launcher found; set MPI_PACKAGE_PROBE_LAUNCHER"
    fi

    log "running capability probe with: ${launcher[*]}"
    "${launcher[@]}" \
        "$BUILD_DIR/verify_parallel_netcdf" \
        "$BUILD_DIR/parallel_netcdf_probe.nc" \
        || die "the loaded stack failed an actual two-rank parallel NetCDF-4 write"
}

compile_library() {
    local -a cflags=()
    local -a libs=()
    local temporary_library="$BUILD_DIR/libmpi_netcdf.so.$$"
    read -r -a cflags <<< "$(nc-config --cflags)"
    read -r -a libs <<< "$(nc-config --libs)"

    log "building $PACKAGE_LIBRARY"
    mpicc -O2 -g -fPIC -Wall -Wextra -Wpedantic -std=c99 \
        "${cflags[@]}" \
        -I"$PACKAGE_DIR" \
        -shared \
        -Wl,-soname,libmpi_netcdf.so \
        -Wl,-z,defs \
        -o "$temporary_library" \
        "$PACKAGE_DIR/mpi_netcdf.c" \
        "${libs[@]}" \
        -lm
    mv -f "$temporary_library" "$PACKAGE_LIBRARY"

    if command -v ldd >/dev/null 2>&1 \
        && ldd "$PACKAGE_LIBRARY" | grep -q 'not found'; then
        ldd "$PACKAGE_LIBRARY" >&2
        die "$PACKAGE_LIBRARY has unresolved shared-library dependencies"
    fi
    if command -v nm >/dev/null 2>&1 \
        && ! nm -D "$PACKAGE_LIBRARY" | grep -q ' mpi_netcdf_create$'; then
        die "$PACKAGE_LIBRARY does not export the expected C ABI"
    fi
}

verify_python_import() {
    log "checking the installed Python binding"
    PYTHONPATH="$PACKAGE_PARENT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_EXECUTABLE" - <<'PY'
import mpi

configuration = mpi.info()
if configuration["size"] != 1:
    raise RuntimeError(f"unexpected import-check world size: {configuration['size']}")
print(f"[mpi] Python binding: {configuration}")
PY
}

yaml_quote() {
    local value=$1
    value=${value//\'/\'\'}
    printf "'%s'" "$value"
}

write_module_manifest() {
    local -a resolved_modules=()
    local loaded_module
    local temporary_manifest="$BUILD_DIR/modules.yml.$$"

    if [[ -n "${LOADEDMODULES:-}" ]]; then
        IFS=: read -r -a resolved_modules <<< "$LOADEDMODULES"
    fi

    {
        printf 'schema_version: 1\n'
        printf 'load_order:\n'
        if [[ -n "$SELECTED_MPI_MODULE" ]]; then
            printf '  - '
            yaml_quote "$SELECTED_MPI_MODULE"
            printf '\n'
        fi
        printf '  - '
        yaml_quote "$SELECTED_NETCDF_MODULE"
        printf '\n'
        if [[ -n "$SELECTED_PYTHON_MODULE" ]]; then
            printf '  - '
            yaml_quote "$SELECTED_PYTHON_MODULE"
            printf '\n'
        fi
        printf 'python:\n'
        if [[ -n "$SELECTED_PYTHON_MODULE" ]]; then
            printf '  module: '
            yaml_quote "$SELECTED_PYTHON_MODULE"
            printf '\n'
        else
            printf '  module: null\n'
        fi
        printf '  executable: '
        yaml_quote "$PYTHON_EXECUTABLE"
        printf '\n'
        if ((${#resolved_modules[@]})); then
            printf 'resolved_modules:\n'
            for loaded_module in "${resolved_modules[@]}"; do
                [[ -n "$loaded_module" ]] || continue
                printf '  - '
                yaml_quote "$loaded_module"
                printf '\n'
            done
        else
            printf 'resolved_modules: []\n'
        fi
    } > "$temporary_manifest"

    mv -f "$temporary_manifest" "$MODULE_MANIFEST"
    log "saved verified runtime modules: $MODULE_MANIFEST"
}

main() {
    init_modules
    load_native_stack
    load_python
    verify_loaded_stack
    verify_sources
    mkdir -p "$BUILD_DIR" "$PACKAGE_LIB_DIR"

    log "MPI module: $SELECTED_MPI_DISPLAY"
    log "NetCDF-C module: $SELECTED_NETCDF_MODULE"
    if [[ -n "$SELECTED_PYTHON_MODULE" ]]; then
        log "Python module: $SELECTED_PYTHON_MODULE"
    fi
    log "Python executable: $PYTHON_EXECUTABLE"
    log "mpicc: $(command -v mpicc)"
    log "nc-config: $(command -v nc-config)"
    log "NetCDF-C: $(nc-config --version)"
    log "NetCDF-4: $(nc-config --has-nc4)"
    log "Parallel NetCDF-4: $(nc-config --has-parallel4)"

    compile_probe
    run_probe
    compile_library
    verify_python_import
    write_module_manifest

    module list 2>&1 || true
    log "installation complete: $PACKAGE_LIBRARY"
    log "reload modules later in the order recorded in: $MODULE_MANIFEST"
    log "run from $PACKAGE_PARENT: srun --mpi=pmix --ntasks=4 $PYTHON_EXECUTABLE -m mpi.test_mpi"
}

main "$@"