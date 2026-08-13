#!/bin/bash -l

#SBATCH -n 4
#SBATCH -t 1:00:00
#SBATCH -o mpi.log






conda activate mother

chmod +x install.sh

./install.sh



srun --mpi=pmix --ntasks=2 python test_mpi.py































