# End-to-end resistive-MHD constrained-transport Hartmann validation.
# Usage:
#   julia --project=../.. gen_mesh.jl
#   julia --project=../.. run.jl

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

const FT = Float64
const debug_nan::Bool = true
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = true
const forcing_mode::Int64 = 1
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = false

const equation_type = :MHD
const ct_mode::Bool = true
const strict_ct_positivity::Bool = true
const splitMethodID::Int32 = Int32(4)
const resistive::Bool = true
const η_mhd::FT = FT(0.1)
const ct_resistive_integrator::Symbol = Symbol(lowercase(get(
    ENV, "HARTMANN_CT_INTEGRATOR", "explicit",
)))
const B0_reference_normalized::FT = FT(0.3)
const cr_glm::FT = FT(0.18)

const viscous::Bool = true
const C_s::FT = FT(0.1)
const T_s::FT = FT(1.0e-10)
const Tw::FT = one(FT)
const R0::FT = one(FT)
const Ma_target::FT = FT(0.05)
const Ro_target::FT = zero(FT)
const Omega_x::FT = zero(FT)
const Lx::FT = one(FT)
const hit_forcing_A::FT = zero(FT)
const weno_z::Bool = true

using CUDA

const _project_root = abspath(joinpath(@__DIR__, "..", ".."))
include(joinpath(_project_root,"src","core","equation_config.jl"))
const B0::FT = B0_reference_normalized * SQRT_MU0_SI  # Stored magnetic field, Tesla
include(joinpath(_project_root,"src","time","structured_rk3_solver.jl"))
include(joinpath(@__DIR__, "hartmann_diagnostics.jl"))

const γ::FT = FT(1.4)
const Rg::FT = one(FT)
const Cp::FT = Rg * γ / (γ - one(FT))
const Pr::FT = FT(0.71)
const u_bulk_target::FT = Ma_target * sqrt(γ * Rg * Tw)
const μw::FT = C_s * Tw * sqrt(Tw) / (Tw + T_s)
const Re_target::FT = u_bulk_target * 2R0 / μw

const mesh_dir = joinpath(@__DIR__, "MESH")
const _conf_data = h5open(joinpath(mesh_dir, "block_connectivity.h5"), "r") do file
    nblocks = read(file["Nblocks"])
    nx_b = read(file["Nx_b"])
    ny_b = read(file["Ny_b"])
    nz_b = read(file["Nz_b"])
    ng = h5read(joinpath(mesh_dir, "mesh_b0.h5"), "NG")
    (Int(nblocks), Tuple(Int.(nx_b)), Tuple(Int.(ny_b)), Tuple(Int.(nz_b)), Int(ng))
end
const Nblocks::Int64 = _conf_data[1]
const Nx_b::NTuple{Nblocks,Int64} = _conf_data[2]
const Ny_b::NTuple{Nblocks,Int64} = _conf_data[3]
const Nz_b::NTuple{Nblocks,Int64} = _conf_data[4]
const NG::Int64 = _conf_data[5]
NG >= 4 || error("resistive CT requires NG >= 4")

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
const Iperiodic = (true, false, true)

const test_case::String = "Hartmann"
const mesh::String = joinpath(mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(mesh_dir, "metrics_b0.h5")
const adaptive_dt::Bool = false
const CFL::FT = FT(0.3)
const LTS::Bool = false
const dt::FT = parse(FT, get(ENV, "HARTMANN_CT_DT", "0.001"))
const Time::FT = FT(100.0)
const maxStep::Int64 = parse(Int, get(ENV, "HARTMANN_CT_STEPS", "5000"))

const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1.0e-3)

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
const filtering_rth::FT = FT(1.0e-5)
const filtering_s0::FT = FT(0.02)
const viscous_order::Int64 = 2
const gg_blend::FT = zero(FT)

const eigen_reconstruction::Bool = true
const hybrid_ϕ1::FT = FT(0.01)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = one(FT)
const UP7::SVector{7,FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7,FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7,FT} = UP7 * Linear_ϕ + CD6 * (one(FT) - Linear_ϕ)
const ΔLinear::SVector{7,FT} = UP7 - CD6
const maxreg::Int64 = 256
const nthreads::Tuple{Int32,Int32,Int32} = (Int32(8), Int32(4), Int32(4))
const nthreads2::Tuple{Int32,Int32,Int32} = (Int32(8), Int32(8), Int32(4))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
if rank == 0
    @printf("HARTMANN_CT_START nx=%d ny=%d nz=%d steps=%d dt=%.4e Ha=%.8f\n",
        Nx_b[1], Ny_b[1], Nz_b[1], maxStep, dt,
        B0 * R0 / sqrt(μw * η_mhd))
end

blocks, active_time, completed_steps = time_step(rank, comm, Block_Nprocs)
if length(blocks) != 1 || MPI.Comm_size(comm) != 1
    error("Hartmann CT diagnostic currently requires one block on one MPI rank")
end
result = verify_hartmann_ct(first(values(blocks)); output_dir=@__DIR__)
if rank == 0
    @printf("HARTMANN_CT_COMPLETE steps=%d time=%.8e\n", completed_steps, active_time)
end

MPI.Finalize()
