#!/bin/bash
#SBATCH --job-name=q1-rowrow
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=q1_%j.log
#SBATCH --error=q1_%j.err
#SBATCH --partition=debug

# Q1: Row-Row distributed matrix multiplication.
# 4 nodes x 2 tasks = 8 slots, which covers the P = 1, 2, 4, 8 sweep.
# Submit with:  sbatch run_q1.slurm

module load hpcx-2.7.0/hpcx-ompi

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_NNODES"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_NODELIST"
echo "========================================="
echo

echo "Compiling..."
mpicxx -O2 -std=c++17 -o matmul_rowrow matmul_rowrow.cpp || {
    echo "Compilation failed!"
    exit 1
}

echo
echo "--- correctness ---"
bash verify_q1.sh || {
    echo "Correctness checks failed; not benchmarking."
    exit 1
}

echo
echo "--- benchmark ---"
bash benchmark_q1.sh

echo
echo "========================================="
echo "Done.  Timings in results/q1_timings.csv"
echo "========================================="
