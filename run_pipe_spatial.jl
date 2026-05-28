# run_pipe_spatial.jl — Spatial Transition to Non-uniform Rotation PipeFlow
# Usage: mpirun -np 15 julia run_pipe_spatial.jl [nsteps]
#
# Physics: Spatially developing pipe flow DNS
#   [0, x_rot_start]: Laminar inflow + Perturbations -> Transition to Turbulence
#   [x_rot_start, x_rot_end]: Non-uniform rotation test section
#   [x_rot_end, L_total]: Outflow / Sponge zone
#
# IMPORTANT: This requires a long mesh (e.g., Lx = 100 R0) and non-periodic BCs.
# The mesh generator must set:
#   face_bc(x_lo) = 40 (BC_TRANSITION_INFLOW)
#   face_bc(x_hi) = 9  (BC_NSCBC_OUTFLOW)

# ─── Parse CLI ───
const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200

# ─── Debug/Runtime flags ───
const debug_nan::Bool = false
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

# DO NOT use global bulk forcing for spatial simulations!
const flow_forcing::Bool = false
const forcing_mode::Int64 = 0
const wall_perturbation::Bool = false
const cache_metrics::Bool = true

const weno_z::Bool = true
const FT = Float64

# ─── PipeFlow parameters ───
const Re_target::FT = FT(17000.0)
const Ma_target::FT = FT(0.3e0)
const Tw::FT = FT(307.0e0)
const Lx::FT = FT(100.0e0)            # 200R0, MUST MATCH MESH

# ─── Rotation Settings (Spatial Envelope) ───
const U_bulk_ref::FT = Ma_target * sqrt(1.4 * 287 * Tw)
const R0_ref::FT = FT(0.5)
const Ro_target::FT = FT(1.0)         # Target rotation number in the test section
const Omega_x::FT = Ro_target * U_bulk_ref / R0_ref

# These globals are read by volume_force.jl
const x_rot_start::FT = FT(60.0) * R0_ref
const x_rot_end::FT   = FT(90.0) * R0_ref

# GPU backend auto-detection
const BACKEND = get(ENV, "FLAME3D_BACKEND", Sys.iswindows() ? "CUDA" : "AMDGPU")
if BACKEND == "CUDA"
    using CUDA
else
    using AMDGPU
end

# ─── Physics / Equation System ───
const equation_type = :compressible
include("physics.jl")
include("solver.jl")

# ─── Thermal state ───
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.71

# ─── Load mesh info ───
const mesh_dir::String = "MESH_SPATIAL"  # <--- USER MUST CREATE THIS MESH

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

# ─── DSRFG Inflow Settings & Initialization ───
const dsrfg_enabled::Bool = true
const dsrfg_N_modes::Int64 = 200        # Number of Fourier modes
const dsrfg_TI::FT = FT(0.05)           # Target turbulence intensity at centerline (5%)
const dsrfg_Lt::FT = FT(0.2) * R0       # Integral length scale (0.2 * R0)
const dsrfg_C_sra::FT = FT(0.7)         # Strong Reynolds Analogy factor (approx Pr)
const dsrfg_seed::Int64 = 42            # Random seed for reproducibility

const dsrfg_params = dsrfg_enabled ? init_dsrfg_params(
    dsrfg_N_modes, dsrfg_Lt, dsrfg_TI, dsrfg_C_sra,
    Ma_target * sqrt(γ * Rg * Tw), Re_target, R0; seed=dsrfg_seed
) : create_dummy_dsrfg_params(FT)


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
    
    if BACKEND == "CUDA"
        try
            CUDA.device!(local_rank_node)
        catch e
            @warn "Failed to set CUDA device: $e"
        end
    else
        # AMDGPU
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
    error("Manual partition not defined for spatial mesh.")
end

# CRITICAL: Spatial simulation is NOT periodic in x
const Iperiodic = (false, false, false)

# ─── Flow control ───
const test_case::String = "PipeFlow"
const mesh::String = joinpath(mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(mesh_dir, "metrics_b0.h5")

const adaptive_dt::Bool = true
const CFL::FT = FT(0.2)             # Start very low for spatial DNS transition
const LTS::Bool = false
const dt::FT = FT(1.0e-4)
const Time::FT = 100.0
const maxStep::Int64 = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : (profiling ? PROFILE_STEPS : Time ÷ dt * 100)

const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1.0e-3)
const inflow_restart::String = "none"

const LES_smag::Bool = false
const LES_wale::Bool = false
const limit_star::Bool = true
const wall_perturbation_type::Int32 = 0
const hit_forcing_A::FT = FT(0.0)

const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 5000

const chk_out::Bool = true
const step_chk::Int64 = 10000
const restart::String = "none"

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
const filtering_interval::Int64 = 1
const intf_filter_interval::Int64 = 1
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.1e0)

const viscous::Bool = true
const viscous_order::Int64 = 6
const gg_blend::FT = one(FT)

const checkerboard_diag::Bool = false
const checkerboard_diag_step::Int64 = -1

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
    println("  Flame3D — Spatial Transition to Rotating Turbulence")
    println("=" ^ 70)
    print_gpu_backend_info()
    println()
    @printf("  Mesh length Lx = %.1f\n", Lx)
    @printf("  Rotation zone  = [%.1f, %.1f]\n", x_rot_start, x_rot_end)
    @printf("  Omega_x max    = %.2f rad/s\n", Omega_x)
    println("=" ^ 70)
    flush(stdout)
end

time_step(rank, comm, Block_Nprocs)

MPI.Finalize()
