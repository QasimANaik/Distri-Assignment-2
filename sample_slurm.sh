#!/bin/bash
#SBATCH --job-name=sample-job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=job_%j.log
#SBATCH --partition=debug

echo "Hello from $SLURM_JOB_ID on node $SLURM_NODELIST"

# Load modules (example)
# module load hpcx-2.7.0/hpcx-ompi

# Compile (example)
# mpicxx -O2 -o my_program my_program.cpp

# Run
# srun ./my_program
