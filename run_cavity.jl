# run_cavity.jl — AC Lid-Driven Cavity (Re=400, steady)
# Usage: mpirun -np 1 julia run_cavity.jl [nsteps]
# Default: 5000 steps

# ─── Parse CLI ───
const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100000

# ─── Debug/Runtime flags ───
const debug_nan::Bool = false
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = false
const forcing_mode::Int64 = 0
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = true

# ─── Spectral Warmup ───
const spectral_warmup::Bool = false
const spectral_warmup_order::Int64 = 4

# Dummy PipeFlow parameters (required by solver infrastructure but unused for AC)
const Re_target::FT = FT(400.0)
const Ma_target::FT = zero(FT)
const Ro_target::FT = zero(FT)
const Tw::FT = FT(300.0)
const Lx::FT = one(FT)
const hit_forcing_A::FT = zero(FT)

# GPU backend — CUDA for local testing
using CUDA

# ─── Physics / AC Equation System ───
const equation_type = :incompressible_AC

# Override AC parameters before including physics.jl:
# Re = U_lid * L / ν → ν = U_lid * L / Re = 1.0 * 1.0 / 400 = 2.5e-3
const β_AC::FT  = FT(5.0e0)       # pseudo-sound speed (increased for stability)
const ρ_ref::FT = one(FT)       # reference density
const ν_AC::FT  = FT(2.5e-3)      # kinematic viscosity for Re=400
const U_lid_AC::FT = one(FT)    # lid velocity

include("physics.jl")

include("solver.jl")

# ─── LES (off for laminar cavity) ───
const LES_smag::Bool = false
const LES_wale::Bool = false

# ─── Thermal state (required by solver, but unused for AC) ───
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.71

# ─── Load mesh info ───
const _mesh_dir = "Benchmark/CAVITY/MESH"
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
const R0::FT = one(FT)  # dummy

const Omega_x::FT = zero(FT)  # no rotation

# ─── Single GPU (np=1) ───
const auto_partition_enabled::Bool = false
const gpu_vram_gb::Float64 = 8.0

MPI.Init()
include("auto_partition.jl")

const Block_Nprocs_manual = [SVector(1,1,1)]
const (Block_Nprocs, Block_to_rank) = (Block_Nprocs_manual, collect(0:0))

# Periodic in z only
const Iperiodic = (false, false, true)

# ─── Flow control ───
const test_case::String = "Cavity"
const mesh::String = "Benchmark/CAVITY/MESH/mesh_b0.h5"
const metrics::String = "Benchmark/CAVITY/MESH/metrics_b0.h5"

const adaptive_dt::Bool = true          # Master switch: compute dt from CFL
const CFL::FT = FT(0.3e0)             #   CFL number
const LTS::Bool = false                 #   Sub-option: per-cell dt (true) or global min (false)
const dt::FT = FT(1.0e-4)             # Fixed dt (used when adaptive_dt=false)
const Time::FT = 1000.0
const maxStep::Int64 = PROFILE_STEPS

# ─── Implicit (off) ───
const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1.0e-3)

const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 50000        # output at 50k and 100k

const chk_out::Bool = false
const step_chk::Int64 = 10000
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
const viscous_order::Int64 = 2      # 2nd order for cavity (simple)
const gg_blend::FT = zero(FT)    # no GG correction for Cartesian mesh

# ─── FVM Config ───
const eigen_reconstruction::Bool = false
const character::Bool = false
const splitMethodID::Int32 = 4
const hybrid_ϕ1::FT = FT(0.5)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = one(FT)   # Full upwind for stability
const UP7::SVector{7, FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, FT} = UP7 * Linear_ϕ + CD6 * (one(FT) - Linear_ϕ)
const ΔLinear::SVector{7, FT} = UP7 - CD6

# GPU Kernel Config
const maxreg::Int64 = 256
const nthreads::Tuple{Int32, Int32, Int32} = (8, 8, 1)   # z=1 for quasi-2D
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 16, 1)

# ═════════════════════════════════════════════════════════
#                  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("=" ^ 70)
    println("  Flame3D — AC Lid-Driven Cavity (Re=400)")
    println("=" ^ 70)
    print_gpu_backend_info()
    println("  Equation type: ", equation_type)
    println("  Ncons = ", Ncons, ", Nprim = ", Nprim)
    println("  β_AC = ", β_AC, ", ν_AC = ", ν_AC, ", Re = ", 1.0/ν_AC)
    println("  Grid: $(Nx_b[1])×$(Ny_b[1])×$(Nz_b[1])")
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
    println("  CAVITY SIMULATION COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", total_time)
    @printf("  Steps completed:     %d\n",     maxStep)
    @printf("  Time per step:       %.4f s\n", total_time / maxStep)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
