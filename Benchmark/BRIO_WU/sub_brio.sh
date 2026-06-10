#!/bin/bash
#SBATCH -J BrioWu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH --ntasks-per-node=1
#SBATCH -p kshdnormal
#SBATCH -o slurm-out
#SBATCH -e slurm-err

module purge; 
module load compiler/dtk/25.04.2  compiler/devtoolset/7.3.1 compiler/cmake/3.25.0 mpi/hpcx/2.7.4-gcc-7.3.1

mpirun -np 1 julia run.jl
