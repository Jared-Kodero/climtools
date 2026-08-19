#!/bin/bash -l
#SBATCH -n 8
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH -t 12:00:00
#SBATCH -o test_mpi.log



conda activate mother

# Unhandled exceptions on a subset of ranks otherwise leave the job blocked in
# MPI_Finalize until the scheduler kills it; `python -m mpi4py` replaces that
# deadlock with MPI_Abort. See
# https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks
#
# CLIMTOOLS_MPI_SYNC_TIMEOUT bounds every mpi.sync() barrier so a straggling
# rank aborts with a diagnostic instead of hanging silently.
export CLIMTOOLS_MPI_SYNC_TIMEOUT=900
export PYTHONUNBUFFERED=1

# Every rank opens the same NetCDF source concurrently. HDF5 takes POSIX
# advisory locks by default, which block indefinitely on Lustre/GPFS.
export HDF5_USE_FILE_LOCKING=FALSE

srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test.py
