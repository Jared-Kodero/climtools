#!/bin/bash
# Run tests/test_mpi_xarray.py across a matrix of MPI rank counts, including
# counts that do not evenly divide the test dataset's time dimension (5, 7),
# to exercise uneven-partition boundaries. Fails (non-zero exit) if any
# rank count fails.
#
# Usage:
#   ./run_mpi_xarray_tests.sh                 # default rank matrix, with timing
#   ./run_mpi_xarray_tests.sh 1 2 4            # custom rank counts
#   CLIMTOOLS_TEST_PERF=0 ./run_mpi_xarray_tests.sh   # skip the timing section
#
# On a single-node development machine without one physical core per rank,
# set MPI_EXTRA_ARGS="--oversubscribe" (OpenMPI) so mpirun doesn't refuse to
# launch more ranks than cores.

set -u
cd "$(dirname "$0")"

RANKS=("$@")
if [ ${#RANKS[@]} -eq 0 ]; then
    RANKS=(1 2 3 4 5 7)
fi

MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
FAILED=()

for n in "${RANKS[@]}"; do
    echo "=== climtools.xarray MPI test suite: -n ${n} ==="
    # shellcheck disable=SC2086
    mpirun -n "${n}" ${MPI_EXTRA_ARGS} python3 test_mpi_xarray.py
    status=$?
    if [ "${status}" -ne 0 ]; then
        FAILED+=("${n}")
    fi
    echo
done

if [ ${#FAILED[@]} -ne 0 ]; then
    echo "FAILED rank counts: ${FAILED[*]}"
    exit 1
fi

echo "All rank counts passed: ${RANKS[*]}"
