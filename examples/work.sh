#!/bin/bash -l
#SBATCH -n 16
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH -t 12:00:00
#SBATCH -o work.log


module load hpcx-mpi/2.25.1s-le4f
module load netcdf-mpi/4.9.3-kuxq

conda activate mother


cd /users/jkodero/research/climtools/lib_mpi

chmod +x install.sh

./install.sh


# cd /users/jkodero/research/climtools/examples


# echo "========================================"
# echo "Starting MPI time composites: 32 processes"
# echo "========================================"
# time mpirun -np 16 --map-by core --bind-to core python time_composites.py
# echo "Finished MPI time composites"
# echo


# echo "========================================"
# echo "Starting MPI time composites: 32 processes"
# echo "========================================"
# time mpirun -np 16 --map-by core --bind-to core python save_composites.py
# echo "Finished MPI time composites"
# echo

# echo "========================================"
# echo "Starting serial time composites"
# echo "========================================"
# time python serial_time_composites.py
# echo "Finished serial time composites"
# echo "========================================"































