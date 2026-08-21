#!/bin/bash -l
#SBATCH -n 8
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log

conda activate mother

module list  # determine versions of netcdf and open mpi the stuff is linked against


import_times=$(python -X importtime -c "import climtools" 2>&1)
echo "$import_times" > import_times.info

# Non-MPI component suite: plot, calc, cmaps, cdo, the .xgeo accessor and
# its non-parallel operations, the serial NetCDF writer, and core.tools.
# Plain `python`, one rank, no launcher -- run first since it is the faster
# and more informative half to fail on.
#python test_general.py

# `python -m mpi4py`, not a bare `python`: an unhandled exception on a subset
# of ranks otherwise leaves the job blocked in MPI_Finalize until the
# scheduler kills it. See
# https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html#exceptions-and-deadlocks
srun --ntasks=8 --cpu-bind=cores --kill-on-bad-exit=1 python -m mpi4py test_mpi.py
