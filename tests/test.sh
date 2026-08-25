#!/bin/bash -l
#SBATCH -n 8
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log

conda activate mother

module list  # determine versions of netcdf and open mpi the stuff is linked against


srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 24 

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 168 

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 720

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 8760 
