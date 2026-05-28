#!/bin/bash
#SBATCH -J spatial_trans
#SBATCH -n 20
#SBATCH -N 5
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-socket=1
#SBATCH -p kshdnormal
#SBATCH --exclude=e07r3n[07,09-12]
#SBATCH -o slurm-out
#SBATCH -e slurm-err

module purge; 
module load compiler/dtk/25.04.2  compiler/devtoolset/7.3.1 compiler/cmake/3.25.0 mpi/hpcx/2.7.4-gcc-7.3.1

# Run the spatial transition simulation. 
# We pass 50 steps first to verify startup and MPI communication.
mpirun -np 20 julia run_pipe_spatial.jl 50000
