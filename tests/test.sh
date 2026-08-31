#!/bin/bash -l
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log

conda activate mother

module list

# ---------------------------------------------------------------------------
# Run the correctness/regression suite at several rank counts, not just the
# job's full allocation. Uneven partitions (an N that doesn't divide evenly
# by the rank count) are exactly what has exposed real bugs in this package
# before -- see STATUS.md -- so low, non-power-of-two counts (1, 3, 5) are
# deliberately included alongside the usual powers of two, in addition to
# the full allocation itself. Every count is capped at $SLURM_NTASKS (falls
# back to the #SBATCH -n above if run outside SLURM) so this never
# oversubscribes the job's own allocation.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_RANKS="${SLURM_NTASKS:-32}"

RANK_COUNTS=()
for candidate in 1 2 3 4 5 8 16 32; do
    if [ "$candidate" -le "$MAX_RANKS" ]; then
        RANK_COUNTS+=("$candidate")
    fi
done
if [ "${RANK_COUNTS[-1]}" -ne "$MAX_RANKS" ]; then
    RANK_COUNTS+=("$MAX_RANKS")
fi

TEST_FILES=("$SCRIPT_DIR"/test_mpi_*.py)

FAILED=0
echo "=========================================================================="
echo " climtools MPI test suite -- ranks: ${RANK_COUNTS[*]}"
echo "=========================================================================="

for ranks in "${RANK_COUNTS[@]}"; do
    for test_file in "${TEST_FILES[@]}"; do
        name="$(basename "$test_file")"
        echo "--- [$ranks ranks] $name ---"
        if srun -n "$ranks" python3 "$test_file"; then
            echo "[$ranks ranks] $name: PASSED"
        else
            echo "[$ranks ranks] $name: FAILED"
            FAILED=1
        fi
    done
done

# ---------------------------------------------------------------------------
# Benchmark suite: native xarray vs MPI-Xarray, at the job's full
# allocation (the rank count actually worth benchmarking at) and, for
# comparison, at a small count -- see bench_mpi_suite.py's own docstring
# for what its Speedup column does and doesn't mean on a single node
# with few physical cores relative to $MAX_RANKS.
# ---------------------------------------------------------------------------

echo "=========================================================================="
echo " climtools MPI benchmark suite"
echo "=========================================================================="

for ranks in 4 "$MAX_RANKS"; do
    if [ "$ranks" -le "$MAX_RANKS" ]; then
        echo "--- [$ranks ranks] bench_mpi_suite.py ---"
        srun -n "$ranks" python3 "$SCRIPT_DIR/bench_mpi_suite.py"
    fi
done

echo "=========================================================================="
if [ "$FAILED" -eq 0 ]; then
    echo " RESULT: all correctness/regression tests passed."
else
    echo " RESULT: one or more correctness/regression tests FAILED (see above)."
fi
echo "=========================================================================="

exit "$FAILED"

