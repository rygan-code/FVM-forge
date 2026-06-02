# run_pipe_cebl_diffrot.jl — CEBL-driven Non-Uniform Rotating Pipe Flow
# Usage: mpirun -np 15 julia run_pipe_cebl_diffrot.jl [nsteps]
# Default: 50 steps

# ─── Parse CLI ───
const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true
const debug_sync::Bool = true
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

# ─── CEBL Settings ───
const cebl_forcing::Bool = true
const cebl_forcing_type::Symbol = :deschamps
const cebl_Nx::Int64 = 256
const cebl_Lx::Float64 = 2.5

# ─── Non-Uniform Rotation (Differential Rotation) Settings ───
const diffrot_enabled::Bool = true
const Ro_min::Float64 = 0.0
const Ro_max::Float64 = 1.0
const L_phys::Float64 = 13.0
const L_total::Float64 = 15.0
const fringe_rise_fraction::Float64 = 0.5
const fringe_lambda_max::Float64 = 0.0         # No fringe damping
const x_rot_start::Float64 = 2.0
const x_rot_end::Float64 = 13.0

const flow_forcing::Bool = true
const forcing_mode::Int64 = 0                  # 0 = No streamwise forcing in the main domain, 3 = Deschamps constant mass flux forcing
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = true

const weno_z::Bool = true
# Type Precision definition
const FT = Float64

# PipeFlow parameters
const Re_target::FT = FT(17000.0)
const Ma_target::FT = FT(0.8e0)
const Ro_target::FT = FT(0.0)
const Tw::FT = FT(307.0e0)
const Lx::FT = FT(15.0e0)
const hit_forcing_A::FT = FT(0.0)

# GPU backend — change to `using AMDGPU` for AMD DCU or `using CUDA` for NVIDIA or comment both and use CPU
# using CUDA
using AMDGPU

# ─── Physics / Equation System ───
const equation_type = :compressible
include("physics.jl")

include("solver.jl")

# ─── LES ───
const LES_smag::Bool = false
const LES_wale::Bool = false

# ─── Thermal state ───
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.71

# ─── Load mesh info from MESH_LEN15/block_connectivity.h5 ───
const mesh_dir = "MESH_LEN15"
const _conf_data = h5open(joinpath(mesh_dir, "block_connectivity.h5"), "r") do file
    nblocks = read(file["Nblocks"])
    nx_b = read(file["Nx_b"])
    ny_b = read(file["Ny_b"])
    nz_b = read(file["Nz_b"])
    ng = h5read(joinpath(mesh_dir, "mesh_b0.h5"), "NG")
    r0 = FT(0.5)
    (nblocks, Tuple(nx_b), Tuple(ny_b), Tuple(nz_b), Int(ng), r0)
end

const Nblocks::Int64 = _conf_data[1]
const Nx_b::NTuple{Nblocks, Int64} = _conf_data[2]
const Ny_b::NTuple{Nblocks, Int64} = _conf_data[3]
const Nz_b::NTuple{Nblocks, Int64} = _conf_data[4]
const NG::Int64 = _conf_data[5]
const R0::FT = _conf_data[6]

# Calculate global Omega limit from Ro range
const U_bulk_ref::FT = Ma_target * sqrt(γ * Rg * Tw)
const Omega_x_min::FT = Ro_min * U_bulk_ref / R0
const Omega_x_max::FT = Ro_max * U_bulk_ref / R0
const Omega_x::FT = Omega_x_max                  # Target Omega at exit

# ─── GPU Partition ───
const auto_partition_enabled::Bool = true
const gpu_vram_gb::Float64 = 16.0

MPI.Init()

# Select GPU device based on local rank on the node
let
    local_comm = MPI.COMM_WORLD
    local_rank_world = MPI.Comm_rank(local_comm)
    shmcomm = MPI.Comm_split_type(local_comm, MPI.COMM_TYPE_SHARED, local_rank_world)
    local_rank_node = MPI.Comm_rank(shmcomm)
    
    if @isdefined(CUDA)
        try
            CUDA.device!(local_rank_node)
        catch e
            @warn "Failed to set CUDA device: $e"
        end
    elseif @isdefined(AMDGPU)
        try
            devs = AMDGPU.devices()
            if !isempty(devs)
                dev_idx = (local_rank_node % length(devs)) + 1
                AMDGPU.device!(devs[dev_idx])
            end
        catch e
            @warn "Failed to set AMDGPU device: $e"
        end
    end
end

include("auto_partition.jl")

const (Block_Nprocs, Block_to_rank) = if auto_partition_enabled
    N_gpus = MPI.Comm_size(MPI.COMM_WORLD)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(">>> Auto GPU partition: $N_gpus GPUs, $Nblocks blocks")
    end
    auto_partition(Nx_b, Ny_b, Nz_b, N_gpus;
        NG=NG, Ncons=Ncons, Nprim=Nprim,
        verbose=(MPI.Comm_rank(MPI.COMM_WORLD) == 0),
        gpu_vram_gb=gpu_vram_gb)
else
    # Fallback manual partition
    ( [SVector(1,1,1) for b in 1:Nblocks], collect(0:Nblocks-1) )
end
const Iperiodic_main = (false, false, false)
const Iperiodic_cebl = (true, false, false)
const Iperiodic = Iperiodic_main

# ─── Flow control ───
const test_case::String = "PipeFlow"
const mesh::String = joinpath(mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(mesh_dir, "metrics_b0.h5")

const adaptive_dt::Bool = true
const CFL::FT = FT(0.5)
const LTS::Bool = false
const dt::FT = FT(2.0e-7)
const Time::FT = 100.0
const maxStep::Int64 = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : (profiling ? PROFILE_STEPS : Time ÷ dt)

# ─── Implicit Time Advancement (LU-SGS) ───
const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1.0e-3)

const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 1000

const chk_out::Bool = true
const step_chk::Int64 = 1000
const restart::String = "none"
const inflow_restart::String = "none"

const average::Bool = false
const avg_step::Int64 = 10
const avg_total::Int64 = 10000
const avg_density_weighted::Bool = true

const sample::Bool = false
const sample_step::Int64 = 1000
const sample_index::SVector{3, Int64} = [-1, -1, -1]

# ─── Explicit Filtering (Pirozzoli-style 8th-order) ───
const filtering::Bool = true
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 10
const intf_filter_interval::Int64 = 1
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.1e0)

const viscous::Bool = true
const viscous_order::Int64 = 6
const gg_blend::FT = one(FT)

# ─── FVM Config ───
const eigen_reconstruction::Bool = true
const splitMethodID::Int32 = 4  # Roe
const hybrid_ϕ1::FT = FT(0.5)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = FT(0.2e0)
const UP7::SVector{7, FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, FT} = UP7 * Linear_ϕ + CD6 * (one(FT) - Linear_ϕ)
const ΔLinear::SVector{7, FT} = UP7 - CD6

# ─── Checkerboard Diagnostic ───
const checkerboard_diag::Bool = false
const checkerboard_diag_step::Int64 = 1000

# GPU Kernel Config
const maxreg::Int64 = 256
const nthreads::Tuple{Int32, Int32, Int32} = (8, 4, 8)
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 8, 8)

# ═════════════════════════════════════════════════════════
#                  ENTRY POINT
# ═════════════════════════════════════════════════════════

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("=" ^ 70)
    println("  Flame3D CEBL Non-Uniform Rotating Pipe Flow (Re=$(Re_target), Ma=$(Ma_target))")
    println("=" ^ 70)
    print_gpu_backend_info()
    println()
end

warmup_start = time_ns()
time_step(rank, comm, Block_Nprocs)
warmup_time = (time_ns() - warmup_start) / 1e9

if rank == 0
    println()
    println("=" ^ 70)
    println("  RUN COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", warmup_time)
    @printf("  Steps completed:     %d\n",     PROFILE_STEPS)
    @printf("  Time per step:       %.4f s\n", warmup_time / PROFILE_STEPS)
    @printf("  Steps per second:    %.1f\n",   PROFILE_STEPS / warmup_time)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
