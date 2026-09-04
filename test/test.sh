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
for n in 4 8 16 "$SLURM_NTASKS"; do
    srun -n "$n" python mpi_test.py
done

# Benchmark
echo "=== benchmark.py ==="
for n in 4 8 16 "$SLURM_NTASKS"; do
    echo "Running benchmarks with n=$n..."
    srun -n "$n" python benchmark.py --size 20000000 --reps 5 --warmup 2
done

python summarize_benchmarks.py