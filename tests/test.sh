#!/bin/bash -l
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log

# Runs the full climtools test suite: test_general.py once (no MPI), then
# the MPI-collective suite (test_mpi.py) across every {tasks, resolution,
# time-steps} combination exercised in production runs. Every combination
# runs even if an earlier one fails -- this script never `set -e`s -- so a
# single bad configuration doesn't hide results for the rest of the matrix;
# each case's own exit code still counts toward the final nonzero exit that
# fails the job.

conda activate mother

module list  # determine versions of netcdf and open mpi the stuff is linked against

# {tasks, resolution degrees, time steps}. Resolution and time-steps are
# independent of task count -- the collective-symmetry and buffer-agreement
# checks need many ranks, not much data -- so this matrix is deliberately
# the full cross product rather than one axis at a time: it is the same
# coverage previously exercised ad hoc that first surfaced the HDF
# file-locking bug (see build_mock_dataset's retry/diagnostics and
# core/tools.py's HDF5_USE_FILE_LOCKING fix), and keeping it as a fixed,
# named matrix means that class of regression fails a specific, reproducible
# cell instead of an undocumented one-off run.
TASK_COUNTS=(8 16 32)
RESOLUTIONS=(0.5 0.25 0.1)
TIME_STEPS=(24 168 720 8760 43800)

run_general_suite() {
    echo "=== test_general.py (no MPI) ==="
    python test_general.py
    local status=$?
    if [ "${status}" -ne 0 ]; then
        echo "=== FAILED: test_general.py ==="
    fi
    return "${status}"
}

# Runs one {tasks, resolution, time-steps} cell of test_mpi.py under srun.
# Exit code: forwards srun's exit code (nonzero on any failed check inside
# test_mpi.py, or on a SLURM-level abort).
run_case() {
    local ntasks="$1"
    local resolution="$2"
    local steps="$3"

    echo ""
    echo "=== test_mpi.py: tasks=${ntasks} resolution=${resolution} time-steps=${steps} ==="
    srun --ntasks="${ntasks}" --cpu-bind=cores --kill-on-bad-exit=1 \
        python -m mpi4py test_mpi.py \
        --time-steps "${steps}" \
        --resolution "${resolution}"
    local status=$?

    if [ "${status}" -ne 0 ]; then
        echo "=== FAILED: tasks=${ntasks} resolution=${resolution} time-steps=${steps} ==="
    fi
    return "${status}"
}

n_fail=0
n_total=0

if ! run_general_suite; then
    n_fail=$((n_fail + 1))
fi
n_total=$((n_total + 1))

for ntasks in "${TASK_COUNTS[@]}"; do
    for resolution in "${RESOLUTIONS[@]}"; do
        for steps in "${TIME_STEPS[@]}"; do
            n_total=$((n_total + 1))
            if ! run_case "${ntasks}" "${resolution}" "${steps}"; then
                n_fail=$((n_fail + 1))
            fi
        done
    done
done

echo ""
echo "=== test suite: $((n_total - n_fail))/${n_total} configurations passed ==="

if [ "${n_fail}" -ne 0 ]; then
    exit 1
fi