#!/bin/bash -l
#SBATCH --job-name=mpi_scaling_test
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --mem=620G
#SBATCH -t 12:00:00
#SBATCH --output=mpi_test.log

conda activate mother

module list  # Determine versions of netcdf and open mpi the stuff is linked against

# Define arrays for scaling parameters
NTASKS_LIST=(8 16 32)
RESOLUTIONS=(0.5 0.25 0.1)
TIME_STEPS=(24 168 720 8760 43800)

echo ""
echo "========================================"
echo "=== Starting Full Scaling Test Suite ==="
echo "Start Time: $(date)"
echo "========================================"

# Loop through number of tasks, resolutions, and time steps
for ntasks in "${NTASKS_LIST[@]}"; do
    for res in "${RESOLUTIONS[@]}"; do
        for steps in "${TIME_STEPS[@]}"; do
            
            echo ""
            echo "----------------------------------------"
            echo " [TEST START] $(date)"
            echo " Tasks (n)  : $ntasks"
            echo " Resolution : $res"
            echo " Time Steps : $steps"
            echo "----------------------------------------"
            
            srun --ntasks="$ntasks" --cpu-bind=cores --kill-on-bad-exit=1 \
                python -m mpi4py test_mpi.py \
                --time-steps "$steps" \
                --resolution "$res"
                
            echo "----------------------------------------"
            echo " [TEST END]   $(date)"
            echo "----------------------------------------"
            
        done
    done
done

echo ""
echo "========================================"
echo "=== All MPI Tests Completed Successfully ==="
echo "End Time: $(date)"
echo "========================================"