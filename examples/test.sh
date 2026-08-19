#!/bin/bash -l
#SBATCH -n 8
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH -t 12:00:00
#SBATCH -o test_mpi.log

conda activate mother

module list  # determine versions of netcdf and open mpi the stuff is linked against


# `python -m mpi4py`, not a bare `python`: an unhandled exception on a subset
# of ranks otherwise leaves the job blocked in MPI_Finalize until the
# scheduler kills it. See
# https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks
srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test.py
