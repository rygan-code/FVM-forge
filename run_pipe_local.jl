# run_pipe_local.jl — PipeFlow multi-block solver entry point for local test
# Usage: julia run_pipe_local.jl [nsteps]

# ─── Parse CLI ───
const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true
const debug_sync::Bool = true
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = true
const forcing_mode::Int64 = 3  # 1=proportional (original), 2=constant mass flux (Deschamps, no mass source)
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0  # 0=pipe (cylindrical), 1=flatplate (blowing/suction strip)
const cache_metrics::Bool = true

# Type Precision definition
const weno_z::Bool = true
const FT = Float64

# PipeFlow parameters
const Re_target::FT = FT(17000.0)
const Ma_target::FT = FT(1.5e0)
const Ro_target::FT = FT(0.0)
const Tw::FT = FT(307.0e0)
const Lx::FT = FT(7.5e0)
const hit_forcing_A::FT = FT(0.0)

# GPU backend — use CUDA locally
using CUDA
# using AMDGPU

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

# ─── Load mesh info from block_connectivity.h5 ───
const mesh_dir = "MESH_SMALL"
const _conf_data = h5open(mesh_dir * "/block_connectivity.h5", "r") do file
    nblocks = read(file["Nblocks"])
    nx_b = read(file["Nx_b"])
    ny_b = read(file["Ny_b"])
    nz_b = read(file["Nz_b"])
    ng = h5read(mesh_dir * "/mesh_b0.h5", "NG")
    r0 = FT(0.5)
    (nblocks, Tuple(nx_b), Tuple(ny_b), Tuple(nz_b), Int(ng), r0)
end

const Nblocks::Int64 = _conf_data[1]
const Nx_b::NTuple{Nblocks, Int64} = _conf_data[2]
const Ny_b::NTuple{Nblocks, Int64} = _conf_data[3]
const Nz_b::NTuple{Nblocks, Int64} = _conf_data[4]
const NG::Int64 = _conf_data[5]
const R0::FT = _conf_data[6]

const Omega_x::FT = Ro_target * (Ma_target * sqrt(γ*Rg*Tw)) / R0

# ─── GPU Partition ───
const auto_partition_enabled::Bool = true
const gpu_vram_gb::Float64 = 8.0  # Local RTX 2060 SUPER has 8 GB

# Initialize MPI
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
    # Fallback manual partition
    ( [SVector(1,1,1) for b in 1:Nblocks], collect(0:Nblocks-1) )
end
const Iperiodic = (true, false, false)

# ─── Flow control ───
const test_case::String = "PipeFlow"
const mesh::String = mesh_dir * "/mesh_b0.h5"
const metrics::String = mesh_dir * "/metrics_b0.h5"

const adaptive_dt::Bool = true          # Master switch: compute dt from CFL
const CFL::FT = FT(0.5)             #   CFL number (used when adaptive_dt=true)
const LTS::Bool = false                 #   Sub-option: per-cell dt (true) or global min (false)
const dt::FT = FT(2.0e-4)             # Fixed dt (used when adaptive_dt=false)
const Time::FT = 100.0
const maxStep::Int64 = PROFILE_STEPS

# ─── Implicit Time Advancement (LU-SGS) ───
const implicit::Bool = false               # Enable LU-SGS implicit time stepping
const implicit_CFL::FT = FT(10.0)       # CFL for implicit scheme (>> 1)
const implicit_lusgs_sweeps::Int64 = 1     # Number of LU-SGS symmetric sweeps per step
const dual_time::Bool = false              # Enable dual-time stepping for 2nd-order temporal accuracy
const dual_time_sub_iters::Int64 = 5       # Max inner iterations for dual-time
const dual_time_tol::FT = FT(1.0e-3)     # Convergence tolerance for dual-time inner loop

const plt_xdmf::Bool = true            # Enable VTK output
const plt_out::Bool = true
const step_plt::Int64 = 3000

const chk_out::Bool = false
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
const filtering::Bool = true           # Enable 8th-order spatial filtering with new boundary offsets
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 10     # Apply every 10 steps
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.1e0)    # Filter strength σ: 0.02=conservative start, increase if needed

# ─── Equation (Ncons/Nprim defined in physics.jl) ───

const viscous::Bool = true
const viscous_order::Int64 = 6
const gg_blend::FT = one(FT)  # GG cross-derivative correction: 0=ds-only, 1=full GG correction

# ─── FVM Config ───
const eigen_reconstruction::Bool = true # Turn off characteristic decomposition for KEP (saves computation, prevents implicit filtering)
const splitMethodID::Int32 = 4  # 1=HLLC, 2=SW, 3=VL, 4=Roe, 5=KEP (Kinetic Energy Preserving)
const hybrid_ϕ1::FT = FT(0.5) # Higher threshold to prevent WENO from firing on turbulent eddies
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = FT(0.2e0)  # Base upwind fraction: 0=pure CD6, 0.05=5% UP7 floor (prevents zero-dissipation instability)
const UP7::SVector{7, FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, FT} = UP7 * Linear_ϕ + CD6 * (one(FT) - Linear_ϕ)
const ΔLinear::SVector{7, FT} = UP7 - CD6  # upwind correction stencil for adaptive blending

# ─── Checkerboard Diagnostic ───
const checkerboard_diag::Bool = true
const checkerboard_diag_step::Int64 = 3000
const limit_star::Bool = true

# GPU Kernel Config
const maxreg::Int64 = 256
const nthreads::Tuple{Int32, Int32, Int32} = (8, 4, 8)
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 8, 8)

# ═════════════════════════════════════════════════════════
#                  PROFILING ENTRY POINT
# ═════════════════════════════════════════════════════════

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("=" ^ 70)
    println("  Flame3D Local Run — PipeFlow Multi-Block")
    println("=" ^ 70)
    print_gpu_backend_info()
    println("  Test case:  ", test_case)
    println("  Blocks:     ", Nblocks)
    println("  Steps:      ", maxStep)
    println("=" ^ 70)
end

warmup_start = time_ns()
time_step(rank, comm, Block_Nprocs)
warmup_time = (time_ns() - warmup_start) / 1e9

if rank == 0
    println()
    println("=" ^ 70)
    println("  LOCAL RUN COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", warmup_time)
    @printf("  Steps completed:     %d\n",     maxStep)
    @printf("  Time per step:       %.4f s\n", warmup_time / maxStep)
    @printf("  Steps per second:    %.1f\n",   maxStep / warmup_time)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
