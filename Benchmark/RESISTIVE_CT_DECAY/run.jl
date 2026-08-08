# Periodic force-free magnetic decay for the complete resistive CT path.

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

const FT = Float64
const debug_nan::Bool = true
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false
const flow_forcing::Bool = false
const forcing_mode::Int64 = 0
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = false

const equation_type = :MHD
const ct_mode::Bool = true
const strict_ct_positivity::Bool = true
const ct_emf_scheme::Int32 = Int32(7)
const splitMethodID::Int32 = Int32(4)
const resistive::Bool = true
const η_mhd::FT = FT(0.05)
const ct_resistive_integrator::Symbol = Symbol(lowercase(get(
    ENV, "RESISTIVE_CT_DECAY_INTEGRATOR", "explicit",
)))
const decay_amplitude_legacy::FT = FT(0.1)
const cr_glm::FT = FT(0.18)

const viscous::Bool = false
const C_s::FT = zero(FT)
const T_s::FT = one(FT)
const Tw::FT = one(FT)
const R0::FT = FT(0.5)
const Re_target::FT = FT(1000)
const Ma_target::FT = FT(0.1)
const Ro_target::FT = zero(FT)
const Omega_x::FT = zero(FT)
const Lx::FT = one(FT)
const hit_forcing_A::FT = zero(FT)
const weno_z::Bool = true

using CUDA

const _project_root = abspath(joinpath(@__DIR__, "..", ".."))
include(joinpath(_project_root,"src","core","equation_config.jl"))
const decay_amplitude::FT = FT(0.1) * SQRT_MU0_SI
include(joinpath(_project_root,"src","time","structured_rk3_solver.jl"))
include(joinpath(@__DIR__, "decay_diagnostics.jl"))

const γ::FT = FT(1.4)
const Rg::FT = one(FT)
const Cp::FT = Rg * γ / (γ - one(FT))
const Pr::FT = FT(0.71)

const mesh_dir = joinpath(@__DIR__, "MESH")
const _conf_data = h5open(joinpath(mesh_dir, "block_connectivity.h5"), "r") do file
    nblocks = Int(read(file["Nblocks"]))
    nx_b = Tuple(Int.(read(file["Nx_b"])))
    ny_b = Tuple(Int.(read(file["Ny_b"])))
    nz_b = Tuple(Int.(read(file["Nz_b"])))
    ng = Int(h5read(joinpath(mesh_dir, "mesh_b0.h5"), "NG"))
    (nblocks, nx_b, ny_b, nz_b, ng)
end
const Nblocks::Int64 = _conf_data[1]
const Nx_b::NTuple{Nblocks,Int64} = _conf_data[2]
const Ny_b::NTuple{Nblocks,Int64} = _conf_data[3]
const Nz_b::NTuple{Nblocks,Int64} = _conf_data[4]
const NG::Int64 = _conf_data[5]

const auto_partition_enabled::Bool = true
const gpu_vram_gb::Float64 = 8.0
const Block_Nprocs_manual = [SVector(1, 1, 1)]
MPI.Init()
include(joinpath(_project_root,"src","parallel","auto_partition.jl"))
const (Block_Nprocs, Block_to_rank) = auto_partition(
    Nx_b, Ny_b, Nz_b, MPI.Comm_size(MPI.COMM_WORLD);
    NG=NG, Ncons=Ncons, Nprim=Nprim,
    verbose=(MPI.Comm_rank(MPI.COMM_WORLD) == 0),
    gpu_vram_gb=gpu_vram_gb,
)
const Iperiodic = (true, true, true)

const test_case::String = "ResistiveCTDecay"
const mesh::String = joinpath(mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(mesh_dir, "metrics_b0.h5")
const adaptive_dt::Bool = false
const CFL::FT = FT(0.3)
const LTS::Bool = false
const dt::FT = parse(FT, get(ENV, "RESISTIVE_CT_DECAY_DT", "0.00025"))
const maxStep::Int64 = parse(Int, get(ENV, "RESISTIVE_CT_DECAY_STEPS", "400"))
const Time::FT = FT(100)

const implicit::Bool = false
const implicit_CFL::FT = FT(10)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1e-3)
const plt_xdmf::Bool = false
const plt_out::Bool = false
const step_plt::Int64 = maxStep
const chk_out::Bool = false
const step_chk::Int64 = maxStep
const restart::String = "none"
const average::Bool = false
const avg_step::Int64 = 10
const avg_total::Int64 = 1000
const avg_density_weighted::Bool = false
const filtering::Bool = false
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 10
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.02)
const viscous_order::Int64 = 2
const gg_blend::FT = zero(FT)
const eigen_reconstruction::Bool = true
const hybrid_ϕ1::FT = FT(0.01)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10)
const Linear_ϕ::FT = one(FT)
const UP7::SVector{7,FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7,FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7,FT} = UP7
const ΔLinear::SVector{7,FT} = UP7 - CD6
const maxreg::Int64 = 256
const nthreads::Tuple{Int32,Int32,Int32} = (Int32(8), Int32(4), Int32(4))
const nthreads2::Tuple{Int32,Int32,Int32} = (Int32(8), Int32(8), Int32(4))

@inline function decay_vector_potential(x, y, z, time)
    wave_number = FT(2) * FT(pi) / Lx
    return SVector{3,FT}(
        zero(FT),
        decay_amplitude / wave_number * sin(wave_number*x),
        decay_amplitude / wave_number * cos(wave_number*x),
    )
end

function in_situ_ct_initial_face_flux_process(
    blocks, world_rank, Block_Nprocs, block_comms,
)
    for block in values(blocks)
        ct_initial_face_flux_from_vector_potential!(
            block, decay_vector_potential,
        )
    end
    return nothing
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
rank == 0 && @printf(
    "RESISTIVE_CT_DECAY_START nx=%d steps=%d dt=%.4e eta=%.4e\n",
    Nx_b[1], maxStep, dt, η_mhd,
)
blocks, active_time, completed_steps = time_step(rank, comm, Block_Nprocs)
length(blocks) == 1 || error("resistive CT decay diagnostic requires one block")
result = verify_resistive_ct_decay(
    first(values(blocks)), active_time; output_dir=@__DIR__,
)
rank == 0 && @printf(
    "RESISTIVE_CT_DECAY_COMPLETE steps=%d time=%.8e\n",
    completed_steps, active_time,
)
MPI.Finalize()
