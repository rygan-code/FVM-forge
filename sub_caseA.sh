#!/bin/bash
#SBATCH -J caseA_diffrot
#SBATCH -n 24
#SBATCH -N 6
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -p kshdnormal
#SBATCH -t 48:00:00
#SBATCH -o tmp_runs/caseA.out
#SBATCH -e tmp_runs/caseA.err

module purge;
module load compiler/dtk/25.04.2 compiler/devtoolset/7.3.1 compiler/cmake/3.25.0 mpi/hpcx/2.7.4-gcc-7.3.1
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=$HOME/gry/rotating_pipe/inhomo/tmp_runs/tmp
mkdir -p "$TMPDIR"

ROOT="$HOME/gry/rotating_pipe/inhomo"
WORK="$ROOT/tmp_runs/caseA_diffrot"

mkdir -p "$WORK/PLT" "$WORK/CHK" "$WORK/AVG"

cd "$WORK"

for f in $ROOT/*.jl; do ln -sfn "$f" .; done
ln -sfn "$ROOT/run" .
ln -sfn "$ROOT/Utils" .
ln -sfn "$ROOT/MESH_LEN15" .

# Background cleanup: every 5 min, keep only last 20 PLT timesteps + last CHK
(
  while true; do
    sleep 300
    cd "$WORK/PLT" 2>/dev/null && {
      keep=$(ls | sed 's/plt-//;s/-b.*//;s/.xmf//' | sort -n | uniq | tail -20 | head -1)
      [ -n "$keep" ] && ls | sed 's/plt-//;s/-b.*//;s/.xmf//' | sort -n | uniq | awk -v k=$keep '$1<k{print}' | while read s; do rm -f plt-$s-b*.h5 plt-$s.xmf; done
    }
    cd "$WORK/CHK" 2>/dev/null && {
      keep_c=$(ls | sed 's/chk-//;s/-b.*//' | sort -n | uniq | tail -1)
      [ -n "$keep_c" ] && ls | sed 's/chk-//;s/-b.*//' | sort -n | uniq | awk -v k=$keep_c '$1<k{print}' | while read s; do rm -f chk-$s-b*.h5; done
    }
  done
) &
CLEANUP_PID=$!

mpirun -np 24 julia run_pipe_cebl_diffrot.jl --mesh_dir=MESH_LEN15 500000

kill $CLEANUP_PID 2>/dev/null

# Final cleanup
cd "$WORK/PLT" 2>/dev/null && {
  keep=$(ls | sed 's/plt-//;s/-b.*//;s/.xmf//' | sort -n | uniq | tail -20 | head -1)
  [ -n "$keep" ] && ls | sed 's/plt-//;s/-b.*//;s/.xmf//' | sort -n | uniq | awk -v k=$keep '$1<k{print}' | while read s; do rm -f plt-$s-b*.h5 plt-$s.xmf; done
}
cd "$WORK/CHK" 2>/dev/null && {
  keep_c=$(ls | sed 's/chk-//;s/-b.*//' | sort -n | uniq | tail -1)
  [ -n "$keep_c" ] && ls | sed 's/chk-//;s/-b.*//' | sort -n | uniq | awk -v k=$keep_c '$1<k{print}' | while read s; do rm -f chk-$s-b*.h5; done
}
