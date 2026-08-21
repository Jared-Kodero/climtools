#!/bin/bash -l
#SBATCH -n 8
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log

conda activate mother

module list  # determine versions of netcdf and open mpi the stuff is linked against

# Required. HDF5 takes POSIX advisory locks by default; on a parallel
# filesystem (Lustre, GPFS) those locks can block indefinitely instead of
# failing fast, which stalls a rank opening a file another rank/process just
# closed -- exactly what test_parallel_netcdf_write's serial read-back does
# right after the collective parallel write. climtools's own parallel writer
# (lib_netcdf/parallel.py) sets this as a process-local default too, but a
# default set after the interpreter starts cannot help other tools (the
# `module`-loaded HDF5/netCDF command-line utilities, srun's own file
# touches) sharing this job's environment, so it belongs here as well.
export HDF5_USE_FILE_LOCKING=FALSE

# Non-MPI component suite: plot, calc, cmaps, cdo, the .xgeo accessor and
# its non-parallel operations, the serial NetCDF writer, and core.tools.
# Plain `python`, one rank, no launcher -- run first since it is the faster
# and more informative half to fail on.
python test_general.py

# `python -m mpi4py`, not a bare `python`: an unhandled exception on a subset
# of ranks otherwise leaves the job blocked in MPI_Finalize until the
# scheduler kills it. See
# https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks
srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py
