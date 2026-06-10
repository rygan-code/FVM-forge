# debug/run_pipe_artifact_test.jl
# Test runner for PipeFlow artifact detection on 128x44x44 5-block Y-mesh.
# Integrates checkerboard and full-field diagnostics to detect numerical artifacts.
# Usage: julia debug/run_pipe_artifact_test.jl [nsteps]
# Default: 100 steps

const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100

println("="^60)
println("  PipeFlow Artifact Detection Test")
println("  Steps: $(PROFILE_STEPS)")
println("  Mesh : debug/MESH_PIPE_TEST/ (5-block, 128x44x44)")
println("="^60)

# ─── Global star-stencil controls ───
const limit_star::Bool = true
const TSVD_REL_TOL::Float64 = 1e-4
const TIKHONOV_LAMBDA::Float64 = 1e-4

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = true
const forcing_mode::Int64 = 3
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = false

# ─── Spectral Warmup ───
const spectral_warmup::Bool = true
const spectral_warmup_order::Int64 = 4
const weno_z::Bool = true
const FT = Float64

# Thermal and Fluid parameters
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.71

# PipeFlow parameters (low Re/Ma for stability and artifact detection)
const Re_target::FT = FT(1000.0)
const Ma_target::FT = FT(0.3)
const Ro_target::FT = zero(FT)
const Tw::FT = FT(300.0)
const Lx::FT = FT(2.0)
const hit_forcing_A::FT = zero(FT)

# ─── GPU Backend ───
using CUDA

# ─── Physics / Equation System ───
const equation_type = :compressible
const _project_root = @__DIR__() * "/.."
include(joinpath(_project_root, "physics.jl"))
include(joinpath(_project_root, "solver.jl"))

# ─── LES (off) ───
const LES_smag::Bool = false
const LES_wale::Bool = false

# ─── Mesh ───
const mesh_dir = "debug/MESH_PIPE_TEST"
const connectivity_file = joinpath(mesh_dir, "block_connectivity.h5")

const _conf_data = h5open(connectivity_file, "r") do file
    nblocks = read(file["Nblocks"])
    nx_b = read(file["Nx_b"])
    ny_b = read(file["Ny_b"])
    nz_b = read(file["Nz_b"])
    ng = h5read(joinpath(mesh_dir, "mesh_b0.h5"), "NG")
    (nblocks, Tuple(nx_b), Tuple(ny_b), Tuple(nz_b), Int(ng))
end

const Nblocks::Int64 = _conf_data[1]
const Nx_b::NTuple{Nblocks, Int64} = _conf_data[2]
const Ny_b::NTuple{Nblocks, Int64} = _conf_data[3]
const Nz_b::NTuple{Nblocks, Int64} = _conf_data[4]
const NG::Int64 = _conf_data[5]
const R0::FT = FT(0.5)
const Omega_x::FT = zero(FT)
const x_rot_start::FT = zero(FT)
const x_rot_end::FT = Lx

# ─── Single GPU Execution Partition ───
const auto_partition_enabled::Bool = false
const gpu_vram_gb::Float64 = 8.0
MPI.Init()
include(joinpath(_project_root, "auto_partition.jl"))

const Block_Nprocs_manual = [SVector(1,1,1) for _ in 1:Nblocks]
const (Block_Nprocs, Block_to_rank) = (Block_Nprocs_manual, zeros(Int, Nblocks))

const Iperiodic = (true, false, false)

# ─── Flow control ───
const test_case::String = "PipeFlow"
const mesh::String = joinpath(mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(mesh_dir, "metrics_b0.h5")

const adaptive_dt::Bool = true
const CFL::FT = FT(0.3)
const LTS::Bool = false
const dt::FT = FT(1e-4)
const Time::FT = 100.0
const maxStep::Int64 = PROFILE_STEPS

const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1e-3)

const plt_xdmf::Bool = false
const plt_out::Bool = true
const step_plt::Int64 = 100
const chk_out::Bool = false
const step_chk::Int64 = 1000
const restart::String = "none"
const inflow_restart::String = "none"

const average::Bool = false
const avg_step::Int64 = 10
const avg_total::Int64 = 1000
const avg_density_weighted::Bool = false

const sample::Bool = false
const sample_step::Int64 = 100
const sample_index::SVector{3, Int64} = [-1, -1, -1]

# ─── Filtering (enabled for stability) ───
const filtering::Bool = true
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 1
const intf_filter_interval::Int64 = 1   # Every step
const intf_filter_sigma::FT = FT(0.20)  # Interface filter strength
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.01) # Baseline filter strength

# ─── Checkerboard Diagnostic ───
const checkerboard_diag::Bool = true
const checkerboard_diag_step::Int64 = PROFILE_STEPS

# ─── Full-Field Diagnostic ───
const fullfield_diag::Bool = true
const fullfield_diag_step::Int64 = PROFILE_STEPS

const viscous::Bool = true
const viscous_order::Int64 = 6
const gg_blend::FT = one(FT)

const eigen_reconstruction::Bool = true
const splitMethodID::Int32 = 4
const hybrid_ϕ1::FT = FT(0.5)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = FT(0.2)
const UP7::SVector{7, FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, FT} = UP7 * Linear_ϕ + CD6 * (one(FT) - Linear_ϕ)
const ΔLinear::SVector{7, FT} = UP7 - CD6

const maxreg::Int64 = 256
const nthreads::Tuple{Int32, Int32, Int32} = (8, 4, 8)
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 8, 8)

# ═══════════════════════════════════════════════════════════
#                  ENTRY POINT
# ═══════════════════════════════════════════════════════════
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

t0 = time_ns()
time_step(rank, comm, Block_Nprocs)
wall = (time_ns() - t0) / 1e9

if rank == 0
    println(">>> Artifact detection test completed in $(round(wall, digits=3)) s.")
end

MPI.Finalize()
