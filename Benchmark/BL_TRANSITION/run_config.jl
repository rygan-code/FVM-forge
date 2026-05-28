# run_config.jl �?Flat Plate Boundary Layer Transition (Implicit LU-SGS Benchmark)
# Usage: mpirun -np 1 julia Benchmark/BL_TRANSITION/run_config.jl [nsteps]
#
# This benchmark tests the implicit LU-SGS time advancement on a
# hypersonic flat plate boundary layer transition problem at Ma=6.0.
#
# Grid: 512×192×32, single block, with inlet perturbation for transition
# BCs: supersonic inflow, NSCBC outflow, isothermal wall, farfield, periodic span
# Time: Implicit LU-SGS + BDF2 dual-time stepping (CFL=10)

# ─── Parse CLI ───
const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2000

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true
const debug_sync::Bool = true
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = false
const forcing_mode::Int64 = 0
const wall_perturbation::Bool = true
const wall_perturbation_type::Int32 = 1  # 0=pipe (cylindrical), 1=flatplate (blowing/suction strip)
const cache_metrics::Bool = true

# ─── Freestream Parameters ───
const Re_target::FT = FT(1000.0)  # Re_δ* at inlet (not used directly, but required)
const Ma_target::FT = FT(6.0e0)
const Ro_target::FT = zero(FT)
const Tw::FT = FT(300.0)          # Cold wall temperature [K] (Tw/Tr �?0.19)
const Lx::FT = FT(200.0)          # Domain length in δ* units
const hit_forcing_A::FT = zero(FT)

# GPU backend
using AMDGPU
# using CUDA

include("../../solver.jl")

# ─── LES ───
const LES_smag::Bool = false
const LES_wale::Bool = false

# ─── Thermal state ───
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.72

# ─── Reference Physical Scales (ISA at 25km) ───
const T_inf_ref::FT = FT(221.5e0)
const u_inf_ref::FT = Ma_target * sqrt(γ * Rg * T_inf_ref)
const p_inf_ref::FT = FT(2549.0e0)
const rho_inf_ref::FT = p_inf_ref / (Rg * T_inf_ref)
const mu_inf_ref::FT = C_s * T_inf_ref^FT(1.5e0) / (T_inf_ref + T_s)
const delta_star_in::FT = Re_target * mu_inf_ref / (rho_inf_ref * u_inf_ref)
const Lx_phys::FT = Lx * delta_star_in
const Lz_phys::FT = FT(15.0e0) * delta_star_in

# ─── Load mesh info from block_connectivity.h5 ───
const _conf_data = h5open("MESH/block_connectivity.h5", "r") do file
    nblocks = read(file["Nblocks"])
    nx_b = read(file["Nx_b"])
    ny_b = read(file["Ny_b"])
    nz_b = read(file["Nz_b"])
    ng = h5read("MESH/mesh_b0.h5", "NG")
    (nblocks, Tuple(nx_b), Tuple(ny_b), Tuple(nz_b), Int(ng))
end

const Nblocks::Int64 = _conf_data[1]
const Nx_b::NTuple{Nblocks, Int64} = _conf_data[2]
const Ny_b::NTuple{Nblocks, Int64} = _conf_data[3]
const Nz_b::NTuple{Nblocks, Int64} = _conf_data[4]
const NG::Int64 = _conf_data[5]
const R0::FT = FT(0.5)  # dummy value (not used for flat plate)

const Omega_x::FT = zero(FT)

# ─── GPU Partition ───
const auto_partition_enabled::Bool = true
const gpu_vram_gb::Float64 = 16.0

const Block_Nprocs_manual = [SVector(1,1,1)]

MPI.Init()
include("../../auto_partition.jl")

const (Block_Nprocs, Block_to_rank) = if auto_partition_enabled
    N_gpus = MPI.Comm_size(MPI.COMM_WORLD)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(">>> Auto GPU partition: $N_gpus GPUs, $Nblocks blocks")
    end
    auto_partition(Nx_b, Ny_b, Nz_b, N_gpus;
        NG=NG, Ncons=5, Nprim=6,
        verbose=(MPI.Comm_rank(MPI.COMM_WORLD) == 0),
        gpu_vram_gb=gpu_vram_gb)
else
    (Block_Nprocs_manual, collect(0:length(Block_Nprocs_manual)-1))
end
const Iperiodic = (false, false, true)  # only z-direction periodic

# ─── Flow control ───
const test_case::String = "FlatPlate"
const mesh::String = "MESH/mesh_b0.h5"
const metrics::String = "MESH/metrics_b0.h5"

const adaptive_dt::Bool = true          # Master switch: compute dt from CFL
const CFL::FT = FT(0.5)             #   CFL number
const LTS::Bool = false                 #   Sub-option: per-cell dt (true) or global min (false)
const dt::FT = FT(1.0e-4)             # Fixed dt (used when adaptive_dt=false)
const Time::FT = 50.0
const maxStep::Int64 = PROFILE_STEPS

# ─── Implicit Time Advancement (LU-SGS) �?ENABLED ───
const implicit::Bool = true               # Enable LU-SGS implicit time stepping
const implicit_CFL::FT = one(FT)       # Initial implicit CFL after switching
const implicit_CFL_max::FT = FT(5.0e0)    # Ramp to CFL=5 (conservative for hypersonic)
const implicit_CFL_ramp_steps::Int64 = 15000 # Ramp over 15000 steps after switch
const implicit_start_step::Int64 = 2000    # First 2000 steps: explicit RK3 (viscous-stable dt)
const implicit_lusgs_sweeps::Int64 = 2     # 2 symmetric sweeps for better convergence
const implicit_lusgs_debug_jacobi::Bool = false  # false = normal LU-SGS, true = Jacobi diagnostic
const implicit_w_LU::FT = FT(1.5e0)       # LU-SGS relaxation factor (EC uses 1~2; old value 0.5 was too weak)
# ─── Implicit Residual Smoothing (EC-style) ───
const implicit_residual_smoothing::Bool = true  # Enable residual smoothing before LU-SGS
const implicit_res_smooth_eps::FT = FT(0.15e0) # Smoothing coefficient (0.1~0.2)
const implicit_res_smooth_passes::Int64 = 2     # Number of smoothing passes
const dual_time::Bool = true               # Enable BDF2 for unsteady transition
const dual_time_start_step::Int64 = 5000   # First 5000 steps without BDF2 to establish laminar baseflow
const dual_time_sub_iters::Int64 = 10       # max 5 inner iterations
const dual_time_tol::FT = FT(1.0e-3)     # convergence tolerance

const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 500

const chk_out::Bool = true
const step_chk::Int64 = 1000
const restart::String = "none"
const inflow_restart::String = "none"

const average::Bool = false
const avg_step::Int64 = 10
const avg_total::Int64 = 1000
const avg_density_weighted::Bool = true

const sample::Bool = false
const sample_step::Int64 = 1000
const sample_index::SVector{3, Int64} = [-1, -1, -1]

# ─── Filtering ───
const filtering::Bool = true               # Boundary-aware: skips 4 cells near physical BCs
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 5       # Filter every 5 steps (increased frequency)
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.05e0)      # Stronger filter to suppress oscillations

# ─── Equation ───
const Ncons::Int64 = 5
const Nprim::Int64 = 6

const viscous::Bool = true
const viscous_order::Int64 = 6
const gg_blend::FT = one(FT)

# ─── FVM Config ───
const eigen_reconstruction::Bool = false
const character::Bool = true
const splitMethodID::Int32 = 1  # HLLC
const hybrid_ϕ1::FT = zero(FT)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = one(FT)  # Full upwind for stability
const UP7::SVector{7, FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, FT} = UP7 * Linear_ϕ + CD6 * (one(FT) - Linear_ϕ)
const ΔLinear::SVector{7, FT} = UP7 - CD6

# GPU Kernel Config
const maxreg::Int64 = 256
const nthreads::Tuple{Int32, Int32, Int32} = (8, 4, 8)
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 8, 8)

# ══════════════════════════════════════════════════════════�?
#                  RUN ENTRY POINT
# ══════════════════════════════════════════════════════════�?
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("=" ^ 70)
    println("  Flame3D �?Flat Plate BL Transition (Implicit LU-SGS + BDF2)")
    println("=" ^ 70)
    print_gpu_backend_info()
    println("  Test case:    ", test_case)
    println("  Grid:         $(Nx_b[1])×$(Ny_b[1])×$(Nz_b[1])")
    println("  Implicit:     ", implicit, " (CFL=$implicit_CFL, BDF2=$dual_time)")
    println("  Max steps:    ", maxStep)
    println("=" ^ 70)
    println()
end

t_start = time_ns()
time_step(rank, comm, Block_Nprocs)
t_elapsed = (time_ns() - t_start) / 1e9

if rank == 0
    println()
    println("=" ^ 70)
    println("  BL TRANSITION BENCHMARK COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", t_elapsed)
    @printf("  Steps completed:     %d\n",     maxStep)
    @printf("  Time per step:       %.4f s\n", t_elapsed / maxStep)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()

