#!/bin/bash

#SBATCH -J ABC_job
#SBATCH -p general
#SBATCH -o hybrid_%j.txt
#SBATCH -e hybrid_%j.err

### Compute nodes
#SBATCH --nodes=5

### MPI ranks
#SBATCH --ntasks=20

### MPI ranks per node
#SBATCH --ntasks-per-node=4

### Tasks per MPI rank
#SBATCH --cpus-per-task=4

#SBATCH --time=5-00:00:00

module load python/3.6.11

### the number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python hybrid.py
