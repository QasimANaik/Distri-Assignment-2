#!/bin/bash
#SBATCH --job-name=mpi-graph-benchmark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --partition=debug
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@college.edu

# Load necessary modules
module load hpcx-2.7.0/hpcx-ompi

# Navigate to project directory
cd /home/cs3401.16/A1/problem3/newrun

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_NNODES"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_NODELIST"
echo "========================================="
echo ""

# Compile MPI program
echo "Compiling MPI program..."
mpicxx -O2 -std=c++17 -o mpi_graph mpi_graph.cpp
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Generate test graphs if they don't exist
if [ ! -f graph_1m_sparse.txt ]; then
    echo "Generating test graphs..."
    bash generate_large_graphs.sh
fi

echo ""
echo "Starting benchmark tests..."
echo ""

# Run simplified benchmark suite
bash benchmark_large_simple.sh

echo ""
echo "========================================="
echo "Benchmark completed!"
echo "Results saved in: benchmark_$SLURM_JOB_ID.log"
echo "========================================="
