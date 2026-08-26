#!/bin/bash -l
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log

conda activate mother

module list  # determine versions of netcdf and open mpi the stuff is linked against

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
            
            status=$?

            if [ "${status}" -ne 0 ]; then
                echo "=== FAILED: tasks=${ntasks} resolution=${resolution} time-steps=${steps} ==="
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