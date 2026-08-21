#!/usr/bin/env bash
# Set up an environment for XGEO, including MPI-collective parallel
# NetCDF-4 output (XGEO.xgeo.to_netcdf(..., parallel=True)).
#
# The parallel stack (mpi4py + netCDF4 built against a parallel-enabled
# MPI/HDF5/NetCDF-C) is located in one of two ways, in order, and never
# from a distro package manager: apt/yum builds are not available on HPC
# login or compute nodes, which is what this stack is for.
#   1. HPC environment modules: `module load` a matching MPI + netcdf-mpi
#      module pair, if the `module` command exists.
#   2. Source build: compile HDF5 and netCDF-C against the active MPI
#      compiler.
#
# Usage:
#   env/setup_env.sh [env_name]
#   env/setup_env.sh --help
#
# With no active conda environment or virtualenv, creates and activates a
# conda environment named env_name (default "XGEO") from
# environment.boot.yaml, installing Miniconda first if conda itself is
# missing, then applies environment.yaml for the rest of XGEO's
# dependencies. Inside an already-active environment, uses it directly and
# skips both steps; only the parallel stack and XGEO itself are
# (re)built.

set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

# Versions built by the source-build fallback (step 2b below). Bump these
# together; netCDF-C's parallel support is only tested against the HDF5
# minor series it was released alongside.
readonly HDF5_SOURCE_VERSION="1.14.5"
readonly NETCDF_C_SOURCE_VERSION="4.9.3"

log() { printf '\n--- %s ---\n' "$*" >&2; }
die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
mpi_module_name=""
netcdf_module_name=""
env_name=""
is_conda=0

python_bin() { command -v python3 || command -v python; }

# True if the active Python's netCDF4/mpi4py are both importable and
# netCDF4 reports parallel4 support. Used both right after building the
# stack and again after any later dependency re-solve that might have
# silently replaced one of the two with a plain (serial) build.
parallel_netcdf_confirmed() {
    "$(python_bin)" -c '
import sys
try:
    import mpi4py.MPI  # noqa: F401
    import netCDF4
except ImportError:
    sys.exit(1)
sys.exit(0 if netCDF4.__has_parallel4_support__ else 1)
' 2>/dev/null
}

# ---------------------------------------------------------------------------
# 1. Environment: use whatever is active, or create+activate a conda one
# ---------------------------------------------------------------------------
if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    log "using active conda environment: ${CONDA_DEFAULT_ENV}"
    is_conda=1
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    log "using active virtualenv: $(basename "${VIRTUAL_ENV}")"
else
    env_name="${1:-XGEO}"
    log "no active environment; creating conda environment '${env_name}'"

    if ! command -v conda >/dev/null 2>&1; then
        log "installing Miniconda"
        mkdir -p "${HOME}/miniconda3"
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -o "${HOME}/miniconda3/miniconda.sh"
        bash "${HOME}/miniconda3/miniconda.sh" -b -u -p "${HOME}/miniconda3"
        rm "${HOME}/miniconda3/miniconda.sh"
        conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        conda_sh="$(conda info --base)/etc/profile.d/conda.sh"
    fi
    # shellcheck source=/dev/null
    source "${conda_sh}"

    if conda env list | grep -qE "^${env_name}[[:space:]]"; then
        conda env update -n "${env_name}" -f "${repo_dir}/boot.yaml"
    else
        conda env create -n "${env_name}" -f "${repo_dir}/boot.yaml"
    fi
    conda activate "${env_name}"
    is_conda=1
fi

# ---------------------------------------------------------------------------
# 2. Locate or build a parallel-enabled MPI/HDF5/NetCDF-C stack
# ---------------------------------------------------------------------------
module_names() {
    local query="$1"
    module -t spider "${query}" 2>&1 \
        | grep -Eo "${query}/[^[:space:]]+" \
        | sed 's/[,:;)]$//' \
        | sort -Vru
}

parallel_stack_usable() {
    local -a cflags=()
    command -v mpicc >/dev/null 2>&1 || return 1
    command -v nc-config >/dev/null 2>&1 || return 1
    read -r -a cflags <<< "$(nc-config --cflags 2>/dev/null || true)"
    printf '%s\n' '#include <netcdf_meta.h>
#if !defined(NC_HAS_PARALLEL4) || !NC_HAS_PARALLEL4
#error "no parallel netcdf4"
#endif
int main(void) { return 0; }' \
        | mpicc "${cflags[@]}" -fsyntax-only -x c - >/dev/null 2>&1
}

find_module_stack() {
    command -v module >/dev/null 2>&1 || return 1

    local -a mpi_candidates=() netcdf_candidates=()
    mapfile -t mpi_candidates < <(module_names hpcx-mpi || true)
    mapfile -t netcdf_candidates < <(module_names netcdf-mpi || true)

    ((${#mpi_candidates[@]} > 0)) || return 1
    ((${#netcdf_candidates[@]} > 0)) || return 1

    local netcdf_module mpi_module
    for netcdf_module in "${netcdf_candidates[@]}"; do
        for mpi_module in "${mpi_candidates[@]}"; do
            log "trying module stack: ${mpi_module} + ${netcdf_module}"
            if (
                module purge >/dev/null 2>&1 || true
                module load "${mpi_module}" >/dev/null 2>&1 \
                    && module load "${netcdf_module}" >/dev/null 2>&1 \
                    && parallel_stack_usable
            ); then
                module purge >/dev/null 2>&1 || true
                module load "${mpi_module}"
                module load "${netcdf_module}"
                NETCDF4_DIR="$(nc-config --prefix)"
                export NETCDF4_DIR
                CC="$(command -v mpicc)"
                export CC
                mpi_module_name="${mpi_module}"
                netcdf_module_name="${netcdf_module}"
                log "using module stack: ${mpi_module} + ${netcdf_module}"
                return 0
            fi
        done
    done
    return 1
}

build_source_stack() {
    local prefix="$1"
    local hdf5_version="${HDF5_SOURCE_VERSION}"
    local netcdf_version="${NETCDF_C_SOURCE_VERSION}"

    command -v mpicc >/dev/null 2>&1 \
        || die "No MPI compiler (mpicc) found. Load or install an MPI implementation first."

    log "building HDF5 ${hdf5_version} (parallel) into ${prefix}"
    (
        cd "${work_dir}"
        curl -fsSL \
            "https://github.com/HDFGroup/hdf5/releases/download/hdf5_${hdf5_version}/hdf5-${hdf5_version}.tar.gz" \
            -o hdf5.tar.gz
        tar xzf hdf5.tar.gz
        cd "hdf5-${hdf5_version}"
        CC=mpicc ./configure --prefix="${prefix}" --enable-parallel \
            --enable-shared --disable-static --disable-fortran --disable-cxx \
            --disable-tests --disable-tools
        make -j"$(nproc)"
        make install
    )

    log "building netCDF-C ${netcdf_version} into ${prefix}"
    (
        cd "${work_dir}"
        curl -fsSL \
            "https://github.com/Unidata/netcdf-c/archive/refs/tags/v${netcdf_version}.tar.gz" \
            -o netcdf-c.tar.gz
        tar xzf netcdf-c.tar.gz
        cd "netcdf-c-${netcdf_version}"
        CC=mpicc \
            CPPFLAGS="-I${prefix}/include" \
            LDFLAGS="-L${prefix}/lib" \
            LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            ./configure --prefix="${prefix}" --disable-static --enable-shared \
            --disable-dap --disable-byterange --disable-testsets --disable-utilities
        make -j"$(nproc)"
        make install
    )

    NETCDF4_DIR="${prefix}"
    export NETCDF4_DIR
    CC=mpicc
    export CC
    export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    grep -Eq '^#define[[:space:]]+NC_HAS_PARALLEL4[[:space:]]+1' \
        "${prefix}/include/netcdf_meta.h" \
        || die "Source build completed but NC_HAS_PARALLEL4 is not set."
    log "source build complete, parallel NetCDF-4 confirmed"
}

patch_netcdf4_python_compat() {
    local compat_header="$1/include/netcdf-compat.h"
    local nc_complex_header="$1/external/nc_complex/include/nc_complex/nc_complex.h"

    if [[ -f "${compat_header}" ]]; then
        python3 - "${compat_header}" << 'PYEOF'
import re
import sys

path = sys.argv[1]
text = open(path).read()
original = text
text = re.sub(
    r"static inline int nc_(def|inq)_var_(bzip2|blosc)\([^{]*\{[^}]*\}\n",
    "",
    text,
)
text = text.replace(
    "#if NC_VERSION_GE(4, 9, 0)\n#define HAS_NCRCSET 1",
    "#if NC_VERSION_GE(4, 9, 1)\n#define HAS_NCRCSET 1",
)
if text != original:
    open(path, "w").write(text)
PYEOF
    fi

    if [[ -f "${nc_complex_header}" ]]; then
        python3 - "${nc_complex_header}" << 'PYEOF'
import sys

path = sys.argv[1]
text = open(path).read()
original = text
for name in ("pfnc_inq_varndims", "pfnc_inq_vardimid"):
    text = text.replace(
        f"NC_COMPLEX_EXPORT inline int {name}(",
        f"static inline int {name}(",
    )
if text != original:
    open(path, "w").write(text)
PYEOF
    fi
}

build_parallel_io_stack() {
    local py
    py="$(python_bin)"

    if find_module_stack; then
        :
    else
        log "no HPC module stack found; building HDF5/netCDF-C from source"
        build_source_stack "${CONDA_PREFIX:-${VIRTUAL_ENV:-${work_dir}/local}}"
    fi

    "${py}" -m pip install --no-cache-dir --break-system-packages --ignore-installed \
        --upgrade "setuptools>=77" wheel cython

    MPI4PY_BUILD_MPICC="${CC}" \
        "${py}" -m pip install --no-cache-dir --break-system-packages --no-binary=mpi4py \
        --no-deps --force-reinstall mpi4py

    "${py}" -m pip download --no-cache-dir --no-build-isolation --no-binary=netCDF4 \
        --no-deps netCDF4 -d "${work_dir}/netcdf4-src"
    tar xzf "${work_dir}"/netcdf4-src/netcdf4-*.tar.gz -C "${work_dir}/netcdf4-src"
    local src_dir
    src_dir="$(find "${work_dir}/netcdf4-src" -mindepth 1 -maxdepth 1 -type d -name 'netcdf4-*')"
    patch_netcdf4_python_compat "${src_dir}"
    (
        cd "${src_dir}"
        "${py}" -m pip install --no-cache-dir --break-system-packages --no-build-isolation \
            --no-deps --force-reinstall .
    )

    if parallel_netcdf_confirmed; then
        "${py}" -c 'import netCDF4; print(f"netCDF4 {netCDF4.__version__}: parallel4 support confirmed")'
    else
        die "netCDF4 built without parallel4 support"
    fi
}

build_parallel_io_stack

# ---------------------------------------------------------------------------
# 3. Persist LD_LIBRARY_PATH (and module loads, if used) for future shells
# ---------------------------------------------------------------------------
if [[ "${is_conda}" == "1" && -n "${CONDA_PREFIX:-}" ]]; then
    activate_dir="${CONDA_PREFIX}/etc/conda/activate.d"
    deactivate_dir="${CONDA_PREFIX}/etc/conda/deactivate.d"
    mkdir -p "${activate_dir}" "${deactivate_dir}"

    cat > "${activate_dir}/XGEO-parallel-io.sh" << HOOK
#!/usr/bin/env bash
export _XGEO_PIO_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"

$([[ -n "${mpi_module_name}" ]] && printf 'module load %q\n' "${mpi_module_name}")
$([[ -n "${netcdf_module_name}" ]] && printf 'module load %q\n' "${netcdf_module_name}")
export LD_LIBRARY_PATH=$(printf '%q' "${LD_LIBRARY_PATH:-}")"\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
HOOK
    cat > "${deactivate_dir}/XGEO-parallel-io.sh" << 'HOOK'
#!/usr/bin/env bash
if [[ -n "${_XGEO_PIO_OLD_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${_XGEO_PIO_OLD_LD_LIBRARY_PATH}"
else
    unset LD_LIBRARY_PATH
fi
unset _XGEO_PIO_OLD_LD_LIBRARY_PATH
HOOK
    chmod +x "${activate_dir}/XGEO-parallel-io.sh" "${deactivate_dir}/XGEO-parallel-io.sh"
    log "installed conda activate/deactivate hooks in ${CONDA_PREFIX}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    marker_begin="# BEGIN XGEO-parallel-io"
    marker_end="# END XGEO-parallel-io"
    activate_script="${VIRTUAL_ENV}/bin/activate"
    if [[ -f "${activate_script}" ]]; then
        if grep -qF "${marker_begin}" "${activate_script}"; then
            sed -i "/${marker_begin}/,/${marker_end}/d" "${activate_script}"
        fi
        {
            printf '%s\n' "${marker_begin}"
            # shellcheck disable=SC2016  # deliberately literal: written into
            # the activate script to be expanded when *that* script is later
            # sourced, not expanded here at setup time.
            printf 'export LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' \
                "${LD_LIBRARY_PATH:-}"
            printf '%s\n' "${marker_end}"
        } >> "${activate_script}"
        log "appended LD_LIBRARY_PATH export to ${activate_script}"
    fi
fi

# ---------------------------------------------------------------------------
# 4. The rest of XGEO's dependencies (conda only; environment.yaml is a
#    conda spec), then XGEO itself, editable, in either environment
# ---------------------------------------------------------------------------
if [[ "${is_conda}" == "1" && -n "${env_name}" ]]; then
    log "applying environment.yaml"
    conda env update -n "${env_name}" -f "${repo_dir}/environment.yaml"

    # A full re-solve can occasionally pull in a replacement mpi4py or
    # netCDF4 as a transitive dependency of something else, even though
    # environment.yaml itself never names them. Confirm the parallel build
    # survived, and rebuild it once if not.
    if ! parallel_netcdf_confirmed; then
        log "environment.yaml solve replaced the parallel I/O build; restoring it"
        build_parallel_io_stack
    fi
fi

log "installing XGEO (editable)"
"$(python_bin)" -m pip install --no-cache-dir --break-system-packages --no-deps -e "${repo_dir}"

log "done"