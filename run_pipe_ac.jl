# run_pipe_ac.jl — Incompressible AC PipeFlow (Re=17000, steady-state precursor)
# Usage: mpirun -np N julia run_pipe_ac.jl [nsteps]
# Default: 200 steps
#
# This uses the mesh_coarse O-type pipe mesh (5 blocks, 512×108×108)
# with the Artificial Compressibility method for incompressible flow.
# Wall BCs are patched from compressible "wall" → "ac_wall" at runtime.

# ─── Parse CLI ───
const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true        
const debug_sync::Bool = true
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

# Pipe forcing: constant pressure gradient to drive bulk flow (AC mode)
# AC forcing modes: 1 = constant pressure gradient (pipe)
const flow_forcing::Bool = true
const forcing_mode::Int64 = 1            # 1 = AC constant pressure gradient
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0  # 0 = pipe (cylindrical)
const cache_metrics::Bool = true

# ─── Spectral Warmup ───
const spectral_warmup::Bool = false
const spectral_warmup_order::Int64 = 4

# Flow parameters (used for forcing + initialization, not for compressible EOS)
const Re_target::FT = FT(17000.0)
const Ma_target::FT = zero(FT)        # Not used for AC
const Ro_target::FT = zero(FT)        # No rotation
const Tw::FT = FT(300.0)             # Dummy (no temperature in AC)
const Lx::FT = FT(7.5e0)               # Pipe length / diameter
const hit_forcing_A::FT = zero(FT)

# GPU backend — AMDGPU for cluster (AMD DCU)
# using CUDA
using AMDGPU

# ─── Physics / AC Equation System ───
const equation_type = :incompressible_AC

# AC parameters:
# Re = U_bulk * D / ν  →  ν = U_bulk * D / Re
# With U_bulk = 1.0, D = 1.0 (diameter):  ν = 1/17000 ≈ 5.88e-5
const β_AC::FT  = FT(15.0e0)           # Pseudo-sound speed (β/u_max ≈ 5.6)
const ρ_ref::FT = one(FT)            # Reference density
const ν_AC::FT  = FT(1.0 / Re_target)   # ν = 1/Re
const U_lid_AC::FT = one(FT)         # Reference velocity (not used as lid for pipe)

# HPDC pressure diffusion (divergence cleaning, analogous to GLM-MHD)
# ∂p/∂t + β²∇·u = ε_p·∇²p  —  damps pseudo-acoustic oscillations
# Complements dual-time stepping: accelerates sub-iteration convergence
#
# FVM metric-weighted Laplacian: ε_p is now the TRUE physical diffusivity [m²/s],
# independent of grid resolution. Recommended range: α ≈ 1-10.
# α too small → slow divergence cleaning; α too large → over-diffuses pressure.
const ac_hpdc::Bool     = true
const α_hpdc::FT   = FT(50.0)        # Moderate cleaning (ε_p/β² = 1.3e-5 steady-state error)
const ε_p_AC::FT   = α_hpdc * ν_AC  # Pressure diffusion coefficient [m²/s]

include("physics.jl")
include("solver.jl")

# ─── LES (off for now — laminar/transitional start) ───
const LES_smag::Bool = false
const LES_wale::Bool = false

# ─── Thermal state (required by infra, unused for AC) ───
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.71

# ─── Load mesh info ───
# Pre-generated AC connectivity: run `julia gen_pipe_ac_connectivity.jl MESH` first
const _mesh_dir = "MESH"
const connectivity_file::String = joinpath(_mesh_dir, "block_connectivity_ac.h5")

const _conf_data = h5open(connectivity_file, "r") do file
    nblocks = read(file["Nblocks"])
    nx_b = read(file["Nx_b"])
    ny_b = read(file["Ny_b"])
    nz_b = read(file["Nz_b"])
    ng = h5read(joinpath(_mesh_dir, "mesh_b0.h5"), "NG")
    r0 = FT(0.5)
    (nblocks, Tuple(nx_b), Tuple(ny_b), Tuple(nz_b), Int(ng), r0)
end

const Nblocks::Int64 = _conf_data[1]
const Nx_b::NTuple{Nblocks, Int64} = _conf_data[2]
const Ny_b::NTuple{Nblocks, Int64} = _conf_data[3]
const Nz_b::NTuple{Nblocks, Int64} = _conf_data[4]
const NG::Int64 = _conf_data[5]
const R0::FT = _conf_data[6]

const Omega_x::FT = zero(FT)  # No rotation

# ─── GPU Partition (auto for cluster) ───
const auto_partition_enabled::Bool = true
const gpu_vram_gb::Float64 = 64.0

MPI.Init()
include("auto_partition.jl")

# Manual fallback (unused when auto_partition_enabled=true)
const Block_Nprocs_manual = [SVector(1,1,1) for _ in 1:Nblocks]

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

# Periodic in ξ (streamwise), non-periodic in η/ζ (cross-section)
const Iperiodic = (true, false, false)

# ─── Flow control ───
const test_case::String = "PipeFlow"
const mesh::String = joinpath(_mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(_mesh_dir, "metrics_b0.h5")

const adaptive_dt::Bool = false         # Use fixed dt for dual-time (physical dt controls accuracy)
const CFL::FT = FT(0.5)             #   CFL number (unused when adaptive_dt=false)
const LTS::Bool = false                 #   Global min dt
const dt::FT = FT(5.0e-4)             # Physical dt (convective CFL ~ dt*U/Δx)
const Time::FT = 1000.0
const maxStep::Int64 = profiling ? PROFILE_STEPS : round(Int64, Float64(Time) / Float64(dt))

# ─── Implicit + Dual-Time Stepping ───
# BDF2 temporal integration with pseudo-time relaxation
# Physical: 2nd-order BDF2 (time-accurate)
# Pseudo:   :lusgs (hyperplane sweep) or :gmres (matrix-free Krylov, fully parallel)
const implicit::Bool = true
const implicit_solver::Symbol = :lusgs   # :lusgs (stable) or :gmres (WIP, unstable)
const implicit_CFL::FT = FT(10.0)    # Pseudo-time CFL
const implicit_lusgs_sweeps::Int64 = 4   # LU-SGS sweeps (4 for β=15 stiffness)
const implicit_w_LU::FT = FT(2.0)    # Diagonal relaxation (used by both as preconditioner)
const implicit_residual_smoothing::Bool = false
const dual_time::Bool = true
const dual_time_start_step::Int64 = 0
const dual_time_sub_iters::Int64 = 5     # Multiple iters needed to damp β=15 pressure overshoot
const dual_time_tol::FT = FT(1.0e-3)   # Tight tol forces 3-5 sub-iters (prevents NaN)
const gmres_m::Int64 = 10               # GMRES Krylov subspace dimension
const gmres_max_restarts::Int64 = 2     # Max GMRES restarts (total matvecs = m × restarts)

const plt_xdmf::Bool = true
const plt_out::Bool = true              # Disable PLT for initial testing
const step_plt::Int64 = 2000

const chk_out::Bool = true
const step_chk::Int64 = 2000
const restart::String = "none"
const inflow_restart::String = "none"

const average::Bool = false
const avg_step::Int64 = 10
const avg_total::Int64 = 1000
const avg_density_weighted::Bool = false

const sample::Bool = false
const sample_step::Int64 = 1000
const sample_index::SVector{3, Int64} = [-1, -1, -1]

# ─── No filtering for AC ───
const filtering::Bool = false
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 10
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = zero(FT)

# ─── Viscous ───
const viscous::Bool = true
const viscous_order::Int64 = 2           # 2nd order for initial testing
const gg_blend::FT = one(FT)         # Full GG correction for curvilinear mesh

# ─── FVM Config ───
const weno_z::Bool = false

const eigen_reconstruction::Bool = false # Turn off characteristic decomposition for KEP (saves computation, prevents implicit filtering)
const character::Bool = false
const splitMethodID::Int32 = 4  # 1=HLLC, 2=SW, 3=VL, 4=Roe, 5=KEP (Kinetic Energy Preserving)
const hybrid_ϕ1::FT = FT(0.5)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = one(FT)         # Full upwind for stability
const UP7::SVector{7, FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, FT} = UP7 * Linear_ϕ + CD6 * (one(FT) - Linear_ϕ)
const ΔLinear::SVector{7, FT} = UP7 - CD6

# GPU Kernel Config
const maxreg::Int64 = 256
const nthreads::Tuple{Int32, Int32, Int32} = (8, 4, 8)
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 8, 8)

# ═════════════════════════════════════════════════════════
#                  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("=" ^ 70)
    println("  Flame3D — AC Incompressible PipeFlow (Re=$(Int(Re_target)))")
    println("=" ^ 70)
    print_gpu_backend_info()
    println("  Equation type: ", equation_type)
    println("  Ncons = ", Ncons, ", Nprim = ", Nprim)
    println("  β_AC = ", β_AC, ", ν_AC = ", ν_AC, ", Re = ", 1.0/ν_AC)
    println("  Blocks: ", Nblocks)
    println("  Grid: $(Nx_b[1])×$(Ny_b[1])×$(Nz_b[1]) per block")
    println("  Mesh dir: ", _mesh_dir)
    println("  Steps: ", maxStep)
    println("=" ^ 70)
    println()
end

warmup_start = time_ns()
time_step(rank, comm, Block_Nprocs)
total_time = (time_ns() - warmup_start) / 1e9

if rank == 0
    println()
    println("=" ^ 70)
    println("  AC PIPEFLOW SIMULATION COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", total_time)
    @printf("  Steps completed:     %d\n",     maxStep)
    @printf("  Time per step:       %.4f s\n", total_time / maxStep)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
