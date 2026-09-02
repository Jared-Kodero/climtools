#!/bin/bash -l
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log

conda activate mother

module list

cd "$(dirname "$0")"

# Correctness suites. Each prints a [PASS]/[FAIL]/[SKIP] line per check
# ([SKIP] marks an operation's own declared NotImplementedError under an
# unsupported partition shape, not a test failure) and a final pass
# count; a nonzero exit code below means at least one genuine [FAIL].
set -e
echo "=== mpi_test.py ==="
srun python mpi_test.py

echo "=== mpi_test_halo_ops.py ==="
srun python mpi_test_halo_ops.py

echo "=== mpi_test_single_dim_ops.py ==="
srun python mpi_test_single_dim_ops.py
set +e

# Benchmark: MPI-Xarray vs native Xarray, at every rank count SLURM gave
# this job plus a couple of smaller counts for scaling comparison. See
# benchmark.py --help for the full option list.
echo "=== benchmark.py ==="
for n in 1 4 8 "$SLURM_NTASKS"; do
    echo "--- ranks=$n ---"
    srun -n "$n" python benchmark.py --size 20000000 --reps 5 --warmup 2
done