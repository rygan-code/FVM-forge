# Benchmark/MHD_DECAY/run.jl — Decaying 3D MHD turbulence with random IC
# Usage: cd Benchmark/MHD_DECAY && julia run.jl [nsteps]
#
# Decaying MHD turbulence from a Passot-Pouquet spectrum initial condition.
# Both velocity and magnetic fields are generated as divergence-free random
# fields (spectral projection onto the k-transverse plane), so this exercises
# the MHD solver on genuinely turbulent (not analytic) data.
#
# Domain: [0, 2π]³, triply periodic, γ = 5/3 (ideal MHD, inviscid)
# Mesh:   generate first with  julia gen_mesh.jl [Nx]   (default 64³)
#
# Tunable IC parameters (defined below): u_rms_init, B_rms_init, k0_init, seed_init

# ─── Parse CLI ───
const FT = Float64
const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2000

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = false       # Decaying turbulence — no forcing
const forcing_mode::Int64 = 0
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = true
# Parameters required by solver (not relevant here but must exist)
const Re_target::FT = FT(1000.0)
const Ma_target::FT = FT(0.3e0)
const Ro_target::FT = zero(FT)
const Tw::FT = FT(300.0)
const Lx::FT = FT(2.0 * pi)
const hit_forcing_A::FT = zero(FT)
const weno_z::Bool = true

# GPU backend
using CUDA
using FFTW            # Required for the Passot-Pouquet divergence-free IC generator
using LinearAlgebra   # dot/cross/norm used by init_mhd_turbulence_field!

# ─── Physics / Equation System ───
const equation_type = :MHD
const resistive::Bool = false   # Ideal MHD (no magnetic diffusion)
const cr_glm::FT = FT(0.18e0)   # GLM damping ratio (Dedner 2002)

# Project root for includes (two levels up from Benchmark/MHD_DECAY/)
const _project_root = joinpath(@__DIR__, "..", "..")
include(joinpath(_project_root, "physics.jl"))
include(joinpath(_project_root, "solver.jl"))

# ─── LES ───
const LES_smag::Bool = false
const LES_wale::Bool = false

# ─── Thermal state (γ = 5/3) ───
const γ::FT = FT(5.0 / 3.0)
const Rg::FT = one(FT)          # Normalized gas constant
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = zero(FT)        # No Sutherland viscosity (inviscid)
const T_s::FT = one(FT)
const Pr::FT = 0.71

# ─── Mesh (relative to Benchmark/MHD_DECAY/) ───
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
const R0::FT = FT(0.5)

const Omega_x::FT = zero(FT)

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
const Iperiodic = (true, true, true)   # Triply periodic

# ─── Flow control ───
const test_case::String = "MHDdecay"
# ── Initial-condition parameters (read by init_mhd_turbulence_field!) ──
# Passot-Pouquet spectrum E(k) ∝ k^4 exp(-2(k/k0)^2), divergence-free v & B.
const u_rms_init::FT = FT(1.0)    # target RMS velocity
const B_rms_init::FT = FT(0.5)    # target RMS magnetic field (sub-Alfvénic)
const k0_init::FT    = FT(4.0)    # peak wavenumber of the spectrum
const rho0_init::FT  = FT(1.0)    # uniform initial density
const p0_init::FT    = FT(1.0)    # uniform initial pressure
const seed_init::Int = 42         # random seed for reproducibility
const mesh::String = joinpath(_mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(_mesh_dir, "metrics_b0.h5")

const adaptive_dt::Bool = true
const CFL::FT = FT(0.3e0)        # Conservative CFL for MHD turbulence
const LTS::Bool = false
const dt::FT = FT(1.0e-3)
const Time::FT = FT(1.0)         # Final time (several eddy turnovers at u_rms~1)
const maxStep::Int64 = profiling ? PROFILE_STEPS : 2000

# ─── Implicit Time Advancement ───
const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1.0e-3)

# ─── Output (relative to Benchmark/MHD_DECAY/) ───
const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 200

const chk_out::Bool = false
const step_chk::Int64 = 1000
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

# ─── Equation (Ncons/Nprim defined in physics.jl) ───
const viscous::Bool = false      # Inviscid ideal MHD
const viscous_order::Int64 = 2
const gg_blend::FT = zero(FT)

# ─── FVM Config ───
const eigen_reconstruction::Bool = false  # Must be false for MHD (no 9×9 eigensystem)
const character::Bool = false
const splitMethodID::Int32 = 1     # 1=Rusanov (robust for MHD turbulence)
const hybrid_ϕ1::FT = FT(0.01e0)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = one(FT)       # Pure upwind in smooth regions
const UP7::SVector{7, FT} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)
const CD6::SVector{7, FT} = SVector(0, 1/60, -2/15, 37/60, 37/60, -2/15, 1/60)
const Linear::SVector{7, FT} = UP7 * Linear_ϕ + CD6 * (one(FT) - Linear_ϕ)
const ΔLinear::SVector{7, FT} = UP7 - CD6

# GPU Kernel Config
const maxreg::Int64 = 256
const nthreads::Tuple{Int32, Int32, Int32} = (8, 4, 8)
const nthreads2::Tuple{Int32, Int32, Int32} = (16, 8, 8)

# ═══════════════════════════════════════════════════════════════════════
# In-situ diagnostics: energy tracking + ∇·B monitoring
# ═══════════════════════════════════════════════════════════════════════
# Wired through the solver's existing in_situ_post_process hook (solver.jl:3322),
# so NO modification to solver.jl is needed — define the function here and the
# solver calls it every step.
#
# For a decaying (unforced) MHD run, total energy E_kin + E_th + E_mag should be
# approximately conserved (only GLM ψ-damping removes a small amount). The
# divergence of B should stay small (~1e-3) if GLM is working.

const _stats_file = joinpath(@__DIR__, "stats.dat")
const _stats_interval = 20

# Write header once on rank 0
function _init_stats_file()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        open(_stats_file, "w") do io
            println(io, "# Decaying 3D MHD turbulence diagnostics (Passot-Pouquet IC)")
            println(io, "# step   time          dt            E_kin         E_th          E_mag         E_total       divB_L2")
        end
    end
end

# Compute L2 norm of ∇·B over all local interior cells (2nd-order central diff).
# Bx,By,Bz are primitive components 7,8,9. For this uniform Cartesian mesh we use
# the constant cell spacing dx=dy=dz=Lx/Nx read from the coordinate arrays.
function _compute_divB_L2_local(blocks)
    divB_sq_local = zero(FT)
    for (bid, b) in blocks
        (b.Nx < 2 || b.Ny < 2 || b.Nz < 2) && continue   # need ≥2 cells per direction
        NGp = NG + 1
        nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG
        # Cell spacing from coordinate arrays (uniform, take first cell).
        # NOTE: b.x/b.y/b.z are GPU arrays — must Array() to host before scalar indexing.
        x_h = Array(@view b.x[NGp:NGp+1, NGp, NGp])
        y_h = Array(@view b.y[NGp, NGp:NGp+1, NGp])
        z_h = Array(@view b.z[NGp, NGp, NGp:NGp+1])
        dx = Float64(x_h[2] - x_h[1])
        dy = Float64(y_h[2] - y_h[1])
        dz = Float64(z_h[2] - z_h[1])
        if dx == 0 || dy == 0 || dz == 0
            continue
        end
        # Pull a padded B-field slab to host (one ghost on each side of interior)
        Bx_h = Array(@view b.Q[NGp-1:nx_end+1, NGp-1:ny_end+1, NGp-1:nz_end+1, 7])
        By_h = Array(@view b.Q[NGp-1:nx_end+1, NGp-1:ny_end+1, NGp-1:nz_end+1, 8])
        Bz_h = Array(@view b.Q[NGp-1:nx_end+1, NGp-1:ny_end+1, NGp-1:nz_end+1, 9])
        # Host array has shape (b.Nx+3, b.Ny+3, b.Nz+3); interior cell (ii=NGp..nx_end)
        # maps to host index h = ii - (NGp-1) + 1 = ii - NGp + 2, i.e. interior h ∈ 2..b.Nx+1
        for kk in 2:(b.Nz+1), jj in 2:(b.Ny+1), ii in 2:(b.Nx+1)
            dBxdx = (Bx_h[ii+1, jj, kk] - Bx_h[ii-1, jj, kk]) / (2.0 * dx)
            dBydy = (By_h[ii, jj+1, kk] - By_h[ii, jj-1, kk]) / (2.0 * dy)
            dBzdz = (Bz_h[ii, jj, kk+1] - Bz_h[ii, jj, kk-1]) / (2.0 * dz)
            divB = dBxdx + dBydy + dBzdz
            divB_sq_local += divB * divB
        end
    end
    return divB_sq_local
end

function in_situ_post_process(tt, activeTime, current_dt, blocks, world_rank, Block_Nprocs, block_comms)
    if tt % _stats_interval != 0 && tt != 1
        return
    end

    ekin, eth, emag = compute_integral_energies(blocks, MPI.COMM_WORLD)
    divB_sq_local = _compute_divB_L2_local(blocks)
    divB_sq_global = MPI.Allreduce(Float64(divB_sq_local), MPI.SUM, MPI.COMM_WORLD)

    # Cell count for normalization
    ncell_local = 0
    for (bid, b) in blocks
        ncell_local += b.Nx * b.Ny * b.Nz
    end
    ncell_global = MPI.Allreduce(ncell_local, MPI.SUM, MPI.COMM_WORLD)
    divB_L2 = sqrt(divB_sq_global / ncell_global)

    etot = ekin + eth + emag

    if world_rank == 0
        @printf("  [diag] step=%-6d t=%.4f dt=%.3e  E_kin=%.4e E_th=%.4e E_mag=%.4e  E_tot=%.4e  |divB|_L2=%.4e\n",
                tt, activeTime, current_dt, ekin, eth, emag, etot, divB_L2)
        open(_stats_file, "a") do io
            @printf(io, "%-7d %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e\n",
                    tt, activeTime, current_dt, ekin, eth, emag, etot, divB_L2)
        end
    end
end

# ═════════════════════════════════════════════════════════
#                  ENTRY POINT
# ═════════════════════════════════════════════════════════

# Create output directories
mkpath(joinpath(@__DIR__, "PLT"))

_init_stats_file()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("=" ^ 70)
    println("  Flame3D — Decaying MHD Turbulence (Passot-Pouquet IC)")
    println("=" ^ 70)
    print_gpu_backend_info()
    println("  Equation:   MHD (Ncons=$Ncons, Nprim=$Nprim)")
    println("  γ:          $γ")
    println("  Rg:         $Rg")
    println("  GLM cr:     $cr_glm")
    println("  Solver:     Rusanov (splitMethodID=$splitMethodID)")
    println("  Test case:  ", test_case)
    println("  Blocks:     ", Nblocks, "  grid: ", Nx_b, "×", Ny_b, "×", Nz_b)
    println("  Max steps:  ", maxStep)
    println("  Final time: ", Time)
    println("  CFL:        ", CFL)
    println("  Periodic:   ", Iperiodic)
    println("  Stats file: ", _stats_file)
    println("=" ^ 70)
    println()
end

warmup_start = time_ns()
time_step(rank, comm, Block_Nprocs)
warmup_time = (time_ns() - warmup_start) / 1e9

if rank == 0
    println()
    println("=" ^ 70)
    println("  MHD DECAY COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", warmup_time)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
