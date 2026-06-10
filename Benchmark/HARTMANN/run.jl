# Benchmark/HARTMANN/run.jl — Hartmann channel flow validation
# Usage: cd Benchmark/HARTMANN && mpirun -np 1 julia run.jl

const FT = Float64
const PROFILE_STEPS = 20000

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = true
const forcing_mode::Int64 = 1
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = true

# Physics constants
const equation_type = :MHD
const resistive::Bool = true   # Resistive MHD (magnetic diffusion enabled)
const η_mhd::FT = FT(0.1)      # Magnetic resistivity
const B0::FT = FT(1.0)         # Transverse magnetic field (Ha = B0 * H * sqrt(1/(eta * mu)))
const cr_glm::FT = FT(0.18e0)  # GLM damping ratio

# Viscous parameters (mu = C_s * sqrt(T) -> 0.1 at T=1.0)
const viscous::Bool = true
const C_s::FT = FT(0.1)
const T_s::FT = FT(1.0e-10)    # Constant viscosity model
const Tw::FT = FT(1.0)
const R0::FT = FT(1.0)         # Half-channel height H

# Mach number (incompressible limit)
const Ma_target::FT = FT(0.1e0)
const Ro_target::FT = zero(FT)
const Omega_x::FT = zero(FT)
const Lx::FT = FT(1.0)
const hit_forcing_A::FT = zero(FT)
const weno_z::Bool = true

# GPU backend
using AMDGPU

# Project root for includes (two levels up from Benchmark/HARTMANN/)
const _project_root = joinpath(@__DIR__, "..", "..")
include(joinpath(_project_root, "physics.jl"))
include(joinpath(_project_root, "solver.jl"))

# ─── LES ───
const LES_smag::Bool = false
const LES_wale::Bool = false

# ─── Thermal state ───
const γ::FT = FT(1.4)
const Rg::FT = one(FT)
const Cp::FT = Rg*γ/(γ-1)
const Pr::FT = 0.71

# Target parameters for bulk forcing
const u_bulk_target::FT = Ma_target * sqrt(γ * Rg * Tw)
const μw::FT = C_s * Tw * sqrt(Tw) / (Tw + T_s)
# Re_target set to make the target density exactly 1.0 (bulk_density = Re * mu / (u * 2 * R0) = 1.0)
const Re_target::FT = u_bulk_target * 2 * R0 * one(FT) / μw

# ─── Mesh (relative to Benchmark/HARTMANN/) ───
const mesh_dir = joinpath(@__DIR__, "MESH")
const _mesh_dir = mesh_dir

const _conf_data = h5open(joinpath(_mesh_dir, "block_connectivity.h5"), "r") do file
    nblocks = read(file["Nblocks"])
    nx_b = read(file["Nx_b"])
    ny_b = read(file["Ny_b"])
    nz_b = read(file["Nz_b"])
    ng = h5read(joinpath(_mesh_dir, "mesh_b0.h5"), "NG")
    (nblocks, Tuple(nx_b), Tuple(ny_b), Tuple(nz_b), Int(ng))
end

const Nblocks::Int64 = _conf_data[1]
const Nx_b::NTuple{Nblocks, Int64} = _conf_data[2]
const Ny_b::NTuple{Nblocks, Int64} = _conf_data[3]
const Nz_b::NTuple{Nblocks, Int64} = _conf_data[4]
const NG::Int64 = _conf_data[5]

# ─── GPU Partition ───
const auto_partition_enabled::Bool = true
const gpu_vram_gb::Float64 = 16.0
const Block_Nprocs_manual = [SVector(1,1,1)]

MPI.Init()
include(joinpath(_project_root, "auto_partition.jl"))

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
    (Block_Nprocs_manual, collect(0:length(Block_Nprocs_manual)-1))
end
const Iperiodic = (true, false, true)  # Periodic in x and z, wall boundaries in y

# ─── Flow control ───
const test_case::String = "Hartmann"
const mesh::String = joinpath(_mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(_mesh_dir, "metrics_b0.h5")

const adaptive_dt::Bool = true
const CFL::FT = FT(0.3e0)
const LTS::Bool = false
const dt::FT = FT(1.0e-3)
const Time::FT = FT(1000.0)       # Increased to ensure full steady state convergence
const maxStep::Int64 = 80000

# ─── Implicit Time Advancement ───
const implicit::Bool = true
const implicit_CFL::FT = FT(10.0)
const implicit_CFL_max::FT = FT(25000.0)
const implicit_CFL_ramp_steps::Int64 = 2000
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1.0e-3)

# ─── Output ───
const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 10000

const chk_out::Bool = false
const step_chk::Int64 = 5000
const restart::String = "none"
const inflow_restart::String = "none"

const average::Bool = false
const avg_step::Int64 = 10
const avg_total::Int64 = 1000
const avg_density_weighted::Bool = false

const sample::Bool = false
const sample_step::Int64 = 1000
const sample_index::SVector{3, Int64} = [-1, -1, -1]

# ─── Filtering ───
const filtering::Bool = false
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 10
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.02e0)

# ─── Equation ───
const viscous_order::Int64 = 2
const gg_blend::FT = zero(FT)

# ─── FVM Config ───
const eigen_reconstruction::Bool = false
const character::Bool = false
const splitMethodID::Int32 = 1     # Rusanov
const hybrid_ϕ1::FT = FT(0.01e0)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = one(FT)       # Upwind
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

mkpath(joinpath(@__DIR__, "PLT"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    # Calculate analytic Hartmann number for display
    mu = C_s
    Ha = B0 * R0 * sqrt(1.0 / (η_mhd * mu))
    
    println("=" ^ 70)
    println("  Flame3D — Hartmann Flow Validation")
    println("=" ^ 70)
    print_gpu_backend_info()
    println("  Equation:   MHD (Ncons=$Ncons, Nprim=$Nprim)")
    println("  Hartmann:   Ha = $Ha")
    println("  Resistivity: η = $η_mhd")
    println("  B0:          $B0")
    println("  Viscosity:  mu = $mu")
    println("  u_target:   $u_bulk_target")
    println("  Re_target:  $Re_target")
    println("  Test case:  ", test_case)
    println("  Blocks:     ", Nblocks)
    println("  Max steps:  ", maxStep)
    println("  Final time: ", Time)
    println("  CFL:        ", CFL)
    println("=" ^ 70)
    println()
end

warmup_start = time_ns()
time_step(rank, comm, Block_Nprocs)
warmup_time = (time_ns() - warmup_start) / 1e9

if rank == 0
    println()
    println("=" ^ 70)
    println("  HARTMANN COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", warmup_time)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
