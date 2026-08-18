#!/bin/bash -l
#SBATCH -n 8
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH -t 12:00:00
#SBATCH -o test_mpi.log


conda activate mother
srun --ntasks=8 --cpu-bind=cores python test.py


