#!/bin/bash
#SBATCH -J cebl_diffrot_fresh
#SBATCH -n 24
#SBATCH -N 6
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -p kshdnormal
#SBATCH -t 24:00:00
#SBATCH -o tmp_runs/cebl_fresh.out
#SBATCH -e tmp_runs/cebl_fresh.err

module purge;
module load compiler/dtk/25.04.2 compiler/devtoolset/7.3.1 compiler/cmake/3.25.0 mpi/hpcx/2.7.4-gcc-7.3.1
export HDF5_USE_FILE_LOCKING=FALSE

ROOT="$HOME/gry/rotating_pipe/inhomo"
WORK="$ROOT/tmp_runs/cebl_fresh_run"

# Create directories
mkdir -p "$WORK/PLT" "$WORK/CHK"

cd "$WORK"

# Symlink all source files and the mesh directory into the work dir
for f in $ROOT/*.jl; do ln -sfn "$f" .; done
ln -sfn "$ROOT/run" .
ln -sfn "$ROOT/Utils" .
ln -sfn "$ROOT/MESH_LEN15" .

mpirun -np 24 julia run_pipe_cebl_diffrot.jl --mesh_dir=MESH_LEN15 50000
