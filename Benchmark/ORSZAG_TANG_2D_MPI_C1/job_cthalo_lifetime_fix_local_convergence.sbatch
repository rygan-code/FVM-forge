#!/usr/bin/env bash
# One-rank counterpart to the warped two-block MPI convergence run.
# Both blocks share one GPU and therefore exercise RK scratch serialization.
#SBATCH --job-name=ot-lifetime-1r
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/data/user/yuchangping/gry/ot2d_ct_derivation_20260807/logs/%x-%A_%a.out
#SBATCH --error=/data/user/yuchangping/gry/ot2d_ct_derivation_20260807/logs/%x-%A_%a.err

set -euo pipefail

readonly RESOLUTIONS=(32 64 128)
readonly TIME_STEPS=(0.0025 0.00125 0.000625)
readonly STEP_LIMITS=(20 40 80)
readonly N=${RESOLUTIONS[${SLURM_ARRAY_TASK_ID}]}
readonly FIXED_DT=${TIME_STEPS[${SLURM_ARRAY_TASK_ID}]}
readonly MAX_STEPS=${STEP_LIMITS[${SLURM_ARRAY_TASK_ID}]}

readonly RUN_ROOT=/data/user/yuchangping/gry/ot2d_ct_derivation_20260807
readonly ROOT=${RUN_ROOT}/source
readonly CASE_ID=OT2D-CTHALO-LIFETIMEFIX-WARP-1GPU-N${N}-T005
readonly CASE_DIR=${RUN_ROOT}/results/${CASE_ID}
readonly MESH_DIR=${ROOT}/Benchmark/ORSZAG_TANG_2D_MPI_C1/mesh/warped_N${N}
readonly JULIA=/data/user/yuchangping/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia
readonly PROJECT=${RUN_ROOT}/runtime
readonly OLD_TEST_DEPOT=/data/user/yuchangping/gry/ot2d_mpi_c1_20260806/.julia_depot
readonly OLD_DEPOT=/data/user/yuchangping/gry/ecs_solver/.julia_depot
readonly TEST_DEPOT=${RUN_ROOT}/.julia_depot
readonly DEPOT=${TEST_DEPOT}:${OLD_TEST_DEPOT}:${OLD_DEPOT}

mkdir -p "${CASE_DIR}" "${TEST_DEPOT}"
export JULIA_DEPOT_PATH="${DEPOT}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OT2D_CASE_ID="${CASE_ID}"
export OT2D_CASE_DIR="${CASE_DIR}"
export OT2D_MESH_DIR="${MESH_DIR}"
export OT2D_FINAL_TIME=0.05
export OT_FINAL_TIME=0.05
export OT2D_FIXED_DT="${FIXED_DT}"
export OT2D_MAX_STEPS="${MAX_STEPS}"
export OT2D_SNAPSHOT_TIMES=0.05
export OT2D_TASK_MODE=taskgraph
export OT2D_INTERFACE_FILTER=false
export OT2D_DEBUG_INTERFACE_FLUX=false
export OT_CFL=0.1
export OT2D_CT_SCHEME=weno7

cd "${ROOT}"
readonly COMPILER_ARTIFACT=${TEST_DEPOT}/artifacts/486b1a24f652206e8b1369615e163ab59ef0152c
mkdir -p "${COMPILER_ARTIFACT}/bin" "${COMPILER_ARTIFACT}/lib" \
  "${COMPILER_ARTIFACT}/nvvm/lib64" "${COMPILER_ARTIFACT}/nvvm/libdevice"
ln -sfn /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudadevrt.a \
  "${COMPILER_ARTIFACT}/lib/libcudadevrt.a"
ln -sfn /usr/local/cuda-12.8/nvvm/libdevice/libdevice.10.bc \
  "${COMPILER_ARTIFACT}/nvvm/libdevice/libdevice.10.bc"
ln -sfn /usr/local/cuda-12.8/nvvm/lib64/libnvvm.so \
  "${COMPILER_ARTIFACT}/nvvm/lib64/libnvvm.so"
ln -sfn /usr/local/cuda-12.8/bin/nvdisasm "${COMPILER_ARTIFACT}/bin/nvdisasm"
ln -sfn /usr/local/cuda-12.8/bin/nvlink "${COMPILER_ARTIFACT}/bin/nvlink"
ln -sfn /usr/local/cuda-12.8/bin/ptxas "${COMPILER_ARTIFACT}/bin/ptxas"

srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 \
  "${JULIA}" --startup-file=no --project="${PROJECT}" \
  "${ROOT}/Benchmark/ORSZAG_TANG_2D_MPI_C1/run_case.jl" "${MAX_STEPS}"
