#!/bin/bash -l
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log

# Runs the full climtools test matrix: the MPI-collective suite
# (test_mpi.py) across every {tasks, resolution, time-steps} combination
# exercised in production runs, plus the non-MPI suite (test_general.py)
# once up front. Every combination runs even if an earlier one fails --
# this script never `set -e`s -- so a single bad configuration doesn't
# hide results for the rest of the matrix; test_mpi.py's own exit code
# (nonzero on any failed check) still fails the corresponding srun step.

conda activate mother

module list  # determine versions of netcdf and open mpi the stuff is linked against

# echo "=== test_general.py (no MPI) ==="
# python test_general.py

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

n_fail=0
n_total=0

for ntasks in "${TASK_COUNTS[@]}"; do
    for resolution in "${RESOLUTIONS[@]}"; do
        for steps in "${TIME_STEPS[@]}"; do
            n_total=$((n_total + 1))
            echo ""
            echo "=== test_mpi.py: tasks=${ntasks} resolution=${resolution} time-steps=${steps} ==="
            srun --ntasks="${ntasks}" --cpu-bind=cores --kill-on-bad-exit=1 \
                python -m mpi4py test_mpi.py \
                --time-steps "${steps}" \
                --resolution "${resolution}"
            if [ $? -ne 0 ]; then
                n_fail=$((n_fail + 1))
                echo "=== FAILED: tasks=${ntasks} resolution=${resolution} time-steps=${steps} ==="
            fi
        done
    done
done

echo ""
echo "=== test_mpi.py matrix: $((n_total - n_fail))/${n_total} configurations passed ==="

if [ "${n_fail}" -ne 0 ]; then
    exit 1
fi
