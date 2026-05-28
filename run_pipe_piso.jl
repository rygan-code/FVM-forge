# run_pipe_piso.jl — Incompressible PISO PipeFlow (Re=17000)
# Usage: mpirun -np N julia run_pipe_piso.jl [nsteps]
#
# PISO (Pressure-Implicit with Splitting of Operators) on staggered grid.
# Replaces AC method — enforces div(u) = 0 exactly via pressure Poisson.
# Wall BCs are patched from compressible "wall" → "ac_wall" at runtime.

# ─── Parse CLI ───
const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true        
const debug_sync::Bool = true
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

# Pipe forcing: constant pressure gradient to drive bulk flow
const flow_forcing::Bool = true
const forcing_mode::Int64 = 1            # 1 = constant pressure gradient
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0  # 0 = pipe (cylindrical)
const cache_metrics::Bool = true

# ─── Spectral Warmup ───
const spectral_warmup::Bool = false
const spectral_warmup_order::Int64 = 4

# Type Precision definition
const FT = Float32

# Flow parameters
const Re_target::FT = FT(17000.0)
const Ma_target::FT = zero(FT)        # Not used for PISO
const Ro_target::FT = zero(FT)        # No rotation
const Tw::FT = FT(300.0)             # Dummy (no temperature)
const Lx::FT = FT(7.5e0)               # Pipe length / diameter
const hit_forcing_A::FT = zero(FT)

# GPU backend — AMDGPU for cluster (AMD DCU)
# using CUDA
using AMDGPU

# ─── Physics / PISO Equation System ───
const equation_type = :incompressible_PISO

# PISO parameters:
# Re = U_bulk * D / ν  →  ν = U_bulk * D / Re
# With U_bulk = 1.0, D = 1.0 (diameter):  ν = 1/17000 ≈ 5.88e-5
const ρ_ref::FT = one(FT)            # Reference density
const ν_AC::FT  = FT(1.0 / Re_target)   # ν = 1/Re (kept as ν_AC for compatibility)
const U_lid_AC::FT = one(FT)         # Reference velocity (not used as lid for pipe)

# AC compatibility stubs (required by shared infrastructure)
const β_AC::FT  = zero(FT)            # Not used — no pseudo-sound speed
const ac_hpdc::Bool   = false           # No pressure diffusion
const ε_p_AC::FT = zero(FT)

# PISO solver parameters
const piso_poisson_max_iters::Int64 = 5000    # Safety cap — early termination exits sooner
const piso_poisson_tol::FT = FT(1.0e-4)      # Convergence tolerance for max|residual|
const piso_sor_omega::FT = FT(1.70e0)        # Conservative ω for non-uniform O-grid mesh
const piso_solver_type = :PCG                 # :PCG or :SOR
const piso_pcg_precond = :sor                 # :jacobi, :sor, or :mg (MG standalone solver — MG has coarse-grid issues on AMDGPU)
const piso_pcg_sor_sweeps::Int64 = 5          # SOR sweeps per preconditioner apply (only for :sor)
const piso_mg_levels::Int64 = 3               # Number of coarse MG levels
const piso_mg_pre_smooth::Int64 = 2           # Pre-smoothing SOR sweeps
const piso_mg_post_smooth::Int64 = 2          # Post-smoothing SOR sweeps
const piso_mg_coarse_sweeps::Int64 = 30       # Direct SOR sweeps on coarsest grid
const piso_noc_iters::Int64 = 2               # Non-orthogonality correction iterations
const piso_noc_max_iters::Int64 = 2000         # Max PCG iters for NOC correction solve
const piso_noc_tol::FT = FT(3.0e-4)           # Relaxed tolerance for NOC (defect correction)

include("physics.jl")
include("solver.jl")

# ─── LES (off for now — laminar/transitional start) ───
const LES_smag::Bool = false
const LES_wale::Bool = false

# ─── Thermal state (required by infra, unused for PISO) ───
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.71

# ─── Load mesh info ───
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

# PISO uses physical dt (NOT pseudo-time CFL)
const adaptive_dt::Bool = false         # Fixed dt for initial validation
const CFL::FT = FT(0.5)
const LTS::Bool = false
const dt::FT = FT(1.0e-4)             # Physical dt (smaller than AC for explicit convection)
const Time::FT = 1000.0
const maxStep::Int64 = profiling ? PROFILE_STEPS : round(Int64, Float64(Time) / Float64(dt))

# ─── Implicit (OFF for PISO — uses explicit predictor + pressure projection) ───
const implicit::Bool = false
const implicit_solver::Symbol = :lusgs
const implicit_CFL::FT = one(FT)
const implicit_lusgs_sweeps::Int64 = 1
const implicit_w_LU::FT = one(FT)
const implicit_residual_smoothing::Bool = false
const dual_time::Bool = false
const dual_time_start_step::Int64 = 0
const dual_time_sub_iters::Int64 = 1
const dual_time_tol::FT = FT(1.0e-3)
const gmres_m::Int64 = 10
const gmres_max_restarts::Int64 = 2

const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 2000

const chk_out::Bool = true
const step_chk::Int64 = 2000
const restart::String = "none"
const inflow_restart::String = "none"

const average::Bool = true
const avg_step::Int64 = 10
const avg_total::Int64 = 1000
const avg_density_weighted::Bool = false

const sample::Bool = false
const sample_step::Int64 = 1000
const sample_index::SVector{3, Int64} = [-1, -1, -1]

# ─── No filtering for PISO ───
const filtering::Bool = false
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 10
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = zero(FT)

# ─── Viscous ───
const viscous::Bool = true
const viscous_order::Int64 = 2
const gg_blend::FT = one(FT)

# ─── FVM Config (mostly unused for PISO staggered, but required by infra) ───
const eigen_reconstruction::Bool = false
const character::Bool = false
const splitMethodID::Int32 = 4
const hybrid_ϕ1::FT = FT(0.5)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = one(FT)
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
    println("  Flame3D — PISO Incompressible PipeFlow (Re=$(Int(Re_target)))")
    println("=" ^ 70)
    print_gpu_backend_info()
    println("  Equation type: ", equation_type)
    println("  Ncons = ", Ncons, ", Nprim = ", Nprim)
    println("  ν = ", ν_AC, ", Re = ", 1.0/ν_AC)
    println("  PISO: SOR ω = ", piso_sor_omega, 
            ", poisson_max_iters = ", piso_poisson_max_iters,
            ", tol = ", piso_poisson_tol)
    println("  Blocks: ", Nblocks)
    println("  Grid: $(Nx_b[1])×$(Ny_b[1])×$(Nz_b[1]) per block")
    println("  Mesh dir: ", _mesh_dir)
    println("  dt = ", dt, ", Steps: ", maxStep)
    println("=" ^ 70)
    println()
end

warmup_start = time_ns()
time_step(rank, comm, Block_Nprocs)
total_time = (time_ns() - warmup_start) / 1e9

if rank == 0
    println()
    println("=" ^ 70)
    println("  PISO PIPEFLOW SIMULATION COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", total_time)
    @printf("  Steps completed:     %d\n",     maxStep)
    @printf("  Time per step:       %.4f s\n", total_time / maxStep)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
