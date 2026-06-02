#!/bin/bash
#SBATCH -J gen_mesh
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -p kshcnormal
#SBATCH -o gen_mesh.out
#SBATCH -e gen_mesh.err

module purge; 
module load compiler/dtk/25.04.2 compiler/devtoolset/7.3.1 compiler/cmake/3.25.0 mpi/hpcx/2.7.4-gcc-7.3.1

cd /public/home/ac6narhq4l/gry/rotating_pipe/inhomo

echo "=== Running Mesh Generator on CPU Compute Node ==="
julia --project=. Utils/gen_butterfly_cebl_len15.jl
echo "=== Mesh Generator Completed ==="
