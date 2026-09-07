#!/bin/bash -l
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log



conda activate mother

module list

export OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_io=romio321

# Correctness suite
echo "=== mpi_test.py ==="

srun -n 8 python mpi_test.py


# Benchmark
echo "=== benchmark.py ==="

srun -n 8 python benchmark.py --size 20000000 --reps 5 --warmup 2


python summarize_benchmarks.py