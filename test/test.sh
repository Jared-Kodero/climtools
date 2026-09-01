#!/bin/bash -l
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log
#
# Run climtools's MPI-Xarray test/benchmark suite under SLURM.
#
# Usage:
#   sbatch test/test.sh
#   sbatch test/test.sh --benchmark          # also run the benchmark suite
#   sbatch test/test.sh --benchmark-only      # skip correctness tests
#
# Every test/test_*.py file is a plain script run directly under
# `mpirun -n $SLURM_NTASKS python -m mpi4py <file>` (no pytest collection
# -- each file's own `if __name__ == "__main__":` block calls sys.exit(1)
# on failure, which this script checks). The benchmark suite
# (test/benchmark_mpi_xarray.py) is run separately, after correctness,
# since a failing correctness test makes any timing number meaningless.
#
# This script never commits or assumes prior output: every dataset it
# needs is generated fresh, locally, via test/mock_dataset.py, sized to
# the SLURM allocation.

conda activate mother

module list

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_DIR}"

NTASKS="${SLURM_NTASKS:-${SLURM_NPROCS:-4}}"
RUN_TESTS=1
RUN_BENCHMARK=0

for arg in "$@"; do
    case "${arg}" in
        --benchmark) RUN_BENCHMARK=1 ;;
        --benchmark-only) RUN_BENCHMARK=1; RUN_TESTS=0 ;;
        *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done

log() { printf '\n=== %s ===\n' "$*"; }

FAILED_TESTS=()

if [[ "${RUN_TESTS}" -eq 1 ]]; then
    log "Correctness tests (${NTASKS} ranks)"

    # Every test/test_*.py file is self-contained: it builds its own
    # (small, synthetic or mock_dataset-generated) data and exits 1 on
    # failure. Run each one under the full allocation's rank count so
    # partition-boundary behavior is exercised at real scale, not just
    # the handful of rank counts checked during development.
    while IFS= read -r -d '' test_file; do
        name="$(basename "${test_file}")"
        log "test/${name}"
        if mpirun -n "${NTASKS}" python -m mpi4py "${test_file}" < /dev/null; then
            echo "PASS: ${name}"
        else
            echo "FAIL: ${name}"
            FAILED_TESTS+=("${name}")
        fi
    done < <(find test -maxdepth 1 -name 'test_*.py' -print0 | sort -z)

    log "Correctness summary"
    if [[ "${#FAILED_TESTS[@]}" -eq 0 ]]; then
        echo "All correctness tests passed."
    else
        echo "FAILED (${#FAILED_TESTS[@]}): ${FAILED_TESTS[*]}"
    fi
fi

if [[ "${RUN_BENCHMARK}" -eq 1 ]]; then
    if [[ "${#FAILED_TESTS[@]}" -gt 0 ]]; then
        log "Skipping benchmark: correctness tests failed"
    else
        log "Benchmark (${NTASKS} ranks)"
        # Sized for a real allocation, not the tiny smoke-test defaults
        # used during development -- adjust to fit your own node memory
        # and time budget. This is the ONLY place a benchmark summary is
        # produced; nothing here is precomputed or committed.
        mpirun -n "${NTASKS}" python -m mpi4py test/benchmark_mpi_xarray.py \
            --n-time 240 \
            --resolution-deg 0.25 \
            --plev-step -10 \
            --repeats 5 < /dev/null
    fi
fi

if [[ "${RUN_TESTS}" -eq 1 && "${#FAILED_TESTS[@]}" -gt 0 ]]; then
    exit 1
fi
