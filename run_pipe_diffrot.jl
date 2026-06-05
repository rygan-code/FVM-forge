# run_pipe_diffrot.jl — Differential Rotation PipeFlow with Fringe Region
# Usage: mpirun -np 15 julia run_pipe_diffrot.jl [nsteps]
#
# Physics: Non-uniform rotation body force Ω(x) = ΔΩ × x / L_phys
#   Physical zone [0, L_phys]:  Ro linearly grows from 0.0 to 1.0
#   Fringe zone [L_phys, L]:    Ω smoothly returns to 0, λ(x) absorbs waves
#   Periodic BC is maintained throughout.

# ─── Parse CLI ───
const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200

# ─── Debug/Runtime flags ───
const debug_nan::Bool = false
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = true
const forcing_mode::Int64 = 3
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = true

# ─── Spectral Warmup ───
const spectral_warmup::Bool = false
const spectral_warmup_order::Int64 = 4
const weno_z::Bool = true
const FT = Float64

# ─── PipeFlow parameters ───
const Re_target::FT = FT(17000.0)
const Ma_target::FT = FT(0.3e0)       # Low Mach for initial testing
const Ro_target::FT = FT(0.0)         # Base rotation (will be overridden by non-uniform Ω)
const Tw::FT = FT(307.0e0)
const Lx::FT = FT(10.0e0)             # 20R₀ (extended from 7.5 for fringe zone)
const hit_forcing_A::FT = FT(0.0)

# ═══════════════════════════════════════════════════════════════════════
# SIMULATION PHASE
# ═══════════════════════════════════════════════════════════════════════
#   Phase 1: 在加长网格上跑标准非旋转管流，发展湍流 (~20000 步)
#            Ω = 0 everywhere, 无 fringe 恢复力
#            完成后: 用 Utils/prepare_precursor_mean.jl 提取时均截面
#   Phase 2: 开启差分旋转 + fringe 恢复力
#            从 Phase 1 的 checkpoint 重启
const diffrot_phase::Int = 1   # Phase 2: 差分旋转 + fringe

# ═══════════════════════════════════════════════════════════════════════
# DIFFERENTIAL ROTATION PARAMETERS (Phase 2 生效)
# ═══════════════════════════════════════════════════════════════════════
const Ro_min::FT = FT(0.0)                     # Rotation number at x = 0
const Ro_max::FT = FT(1.0)                     # Rotation number at x = L_phys
const L_phys::FT = FT(7.5)                     # Physical zone length (15R₀)
const L_total::FT = Lx                          # Total domain length (20R₀)

# Phase-dependent flags
const diffrot_enabled::Bool = (diffrot_phase == 2)
const fringe_lambda_max::FT = diffrot_phase == 2 ? FT(50.0) : FT(0.0)
const fringe_rise_fraction::FT = FT(0.5)
const precursor_mean_path::String = "PLT/precursor_mean.h5"

# GPU backend
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

# ─── Load mesh info ───
const mesh_dir::String = "MESH_DIFFROT_COARSE"  # Use the extended mesh

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

# Compute Ω range from Ro range
const U_bulk_ref::FT = Ma_target * sqrt(γ * Rg * Tw)
const Omega_x_min::FT = Ro_min * U_bulk_ref / R0
const Omega_x_max::FT = Ro_max * U_bulk_ref / R0
# Phase 1: Omega_x = 0 → fused Deschamps path (no rotation, fast)
# Phase 2: Omega_x > 0 → rotation + fringe forcing path
const Omega_x::FT = diffrot_phase == 2 ? Omega_x_max : zero(FT)

# ─── GPU Partition ───
const auto_partition_enabled::Bool = true
const gpu_vram_gb::Float64 = 16.0

const Block_Nprocs_manual = [
    SVector(4,2,1),
    SVector(4,1,1),
    SVector(4,1,1),
    SVector(4,1,1),
    SVector(4,1,1)
]

MPI.Init()
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
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(">>> Manual GPU partition mode")
    end
    (Block_Nprocs_manual, collect(0:length(Block_Nprocs_manual)-1))
end
const Iperiodic = (true, false, false)

# ─── Flow control ───
const test_case::String = "PipeFlow"          # Must match init_flow.jl dispatcher
const mesh::String = joinpath(mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(mesh_dir, "metrics_b0.h5")

const adaptive_dt::Bool = true
const CFL::FT = FT(0.3)             # Lower for cold start; raise to 0.5 after turbulence develops
const LTS::Bool = false
const dt::FT = FT(2.0e-4)
const Time::FT = 100.0
const maxStep::Int64 = profiling ? PROFILE_STEPS : Time ÷ dt * 100

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
const restart::String = "200000"
const inflow_restart::String = "none"

const average::Bool = true
const avg_step::Int64 = 10
const avg_total::Int64 = 10000
const avg_density_weighted::Bool = true

const sample::Bool = false
const sample_step::Int64 = 1000
const sample_index::SVector{3, Int64} = [-1, -1, -1]

# ─── Filtering ───
const filtering::Bool = true
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 10   # Every step for cold start; raise to 10 after turbulence develops
const intf_filter_interval::Int64 = 1   # Every step: 2Δx mode damping requires high frequency
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.1e0)

const viscous::Bool = true
const viscous_order::Int64 = 6
const gg_blend::FT = one(FT)

# ─── Checkerboard Diagnostic ───
const checkerboard_diag::Bool = false        # Diagnostic complete (2026-05-09)
const checkerboard_diag_step::Int64 = 200110 # Trigger at this absolute step (must match restarted tt)

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
    if diffrot_phase == 1
        println("  Flame3D — Phase 1: Turbulence Development (Ro=0, no fringe)")
    else
        println("  Flame3D — Phase 2: Differential Rotation (Ro=$(Ro_min)→$(Ro_max))")
    end
    println("=" ^ 70)
    print_gpu_backend_info()
    println()
    println("┌─────────────────────────────────────────────────────────────────")
    println("│ Phase $(diffrot_phase) Configuration")
    println("├─────────────────────────────────────────────────────────────────")
    @printf("│  Phase         = %d (%s)\n", diffrot_phase,
            diffrot_phase == 1 ? "turbulence development" : "differential rotation")
    @printf("│  diffrot       = %s\n", diffrot_enabled ? "ON" : "OFF")
    @printf("│  Ro range      = %.2f → %.2f\n", Ro_min, Ro_max)
    @printf("│  Ω range       = %.2f → %.2f rad/s\n", Omega_x_min, Omega_x_max)
    @printf("│  Omega_x       = %.2f (active)\n", Omega_x)
    @printf("│  λ_max         = %.2f\n", fringe_lambda_max)
    println("├─────────────────────────────────────────────────────────────────")
    println("│ Flow Parameters")
    println("├─────────────────────────────────────────────────────────────────")
    @printf("│  Ma            = %.2f\n", Ma_target)
    @printf("│  Re            = %.0f\n", Re_target)
    @printf("│  Tw            = %.1f K\n", Tw)
    @printf("│  U_bulk        = %.1f m/s\n", U_bulk_ref)
    @printf("│  Precision     = %s\n", string(FT))
    println("├─────────────────────────────────────────────────────────────────")
    println("│ Grid")
    println("├─────────────────────────────────────────────────────────────────")
    for b in 0:Nblocks-1
        @printf("│  Block %d:  %d × %d × %d  (NG=%d)\n", b, Nx_b[b+1], Ny_b[b+1], Nz_b[b+1], NG)
    end
    @printf("│  Mesh dir      = %s\n", mesh_dir)
    @printf("│  Restart       = %s\n", restart)
    println("└─────────────────────────────────────────────────────────────────")
    println()
end

warmup_start = time_ns()
time_step(rank, comm, Block_Nprocs)
warmup_time = (time_ns() - warmup_start) / 1e9

if rank == 0
    println()
    println("=" ^ 70)
    println("  SIMULATION COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", warmup_time)
    @printf("  Steps completed:     %d\n",     PROFILE_STEPS)
    @printf("  Time per step:       %.4f s\n", warmup_time / PROFILE_STEPS)
    @printf("  Steps per second:    %.1f\n",   PROFILE_STEPS / warmup_time)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
