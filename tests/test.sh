#!/bin/bash -l
#SBATCH -n 8
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log

conda activate mother

module list  # determine versions of netcdf and open mpi the stuff is linked against

# Correctness suite: no MPI launcher needed.
python test_general.py > test_general.log 2>&1

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi_xarray_reductions.py > test_mpi_xarray_reductions.log 2>&1

# `python -m mpi4py`, not a bare `python`: an unhandled exception on a subset
# of ranks otherwise leaves the job blocked in MPI_Finalize until the
# scheduler kills it. See
# https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks
srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 24 > test.24.log 2>&1

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 168 > test.168.log 2>&1

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 720 > test.720.log 2>&1

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 8760 > test.8760.log 2>&1

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 43800 > test.43800.log 2>&1

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py --time-steps 87600 > test.87600.log 2>&1