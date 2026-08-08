# Benchmark/ORSZAG_TANG_3D/run.jl — 3D Orszag-Tang MHD turbulence benchmark
# Usage: cd Benchmark/ORSZAG_TANG_3D && mpirun -np 1 julia run.jl [nsteps]
#
# 3D extension of the classic Orszag-Tang vortex (Orszag & Tang, 1979).
# Tests MHD turbulence + shock interactions in 3D, GLM divergence cleaning
# under sustained turbulence, and WENO-MHD robustness.
#
# Domain: [0, 2π]³, triply periodic, γ = 5/3 (ideal MHD, inviscid)
# Mesh:   generate first with  julia gen_mesh.jl [Nx]   (default 64³)

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

# ─── Physics / Equation System ───
const equation_type = :MHD
const resistive::Bool = false   # Ideal MHD (no magnetic diffusion)
const cr_glm::FT = FT(0.18e0)   # GLM damping ratio (Dedner 2002)
const _ot3d_ct_requested::Bool = lowercase(get(ENV, "OT3D_CT", "false")) in
    ("1", "true", "yes", "on")
const ct_mode::Bool = _ot3d_ct_requested
const strict_ct_positivity::Bool = ct_mode &&
    lowercase(get(ENV, "OT3D_STRICT_CT", "true")) in ("1", "true", "yes", "on")
const splitMethodID::Int32 = ct_mode ? Int32(4) : Int32(1)
# Required before loading the solver stack so static viscous branches resolve.
const viscous::Bool = false
function _ot3d_metric_ct_state(value)
    parts = split(strip(value), ',')
    length(parts) == 10 || error("OT3D_METRIC_CT_STATE must contain 10 comma-separated values")
    return ntuple(index -> parse(FT, strip(parts[index])), 10)
end
# Project root for includes (two levels up from Benchmark/ORSZAG_TANG_3D/)
const _project_root = joinpath(@__DIR__, "..", "..")
if !isdefined(@__MODULE__, :METRIC_CT_STATS_COLUMNS)
    include(joinpath(@__DIR__, "..", "METRIC_CT_WARPED", "metric_ct_diagnostics.jl"))
end
include(joinpath(_project_root,"src","core","equation_config.jl"))
const metric_ct_uniform_state::NTuple{10,FT} = if haskey(
    ENV, "OT3D_METRIC_CT_STATE",
)
    # User-provided metric states are already physical SI values.
    _ot3d_metric_ct_state(ENV["OT3D_METRIC_CT_STATE"])
else
    # Preserve the legacy reference state while storing B in Tesla.
    (one(FT), FT(0.23), FT(-0.17), FT(0.11), one(FT), one(FT),
     FT(0.61) * SQRT_MU0_SI, FT(-0.37) * SQRT_MU0_SI,
     FT(0.29) * SQRT_MU0_SI, zero(FT))
end
include(joinpath(_project_root,"src","time","structured_rk3_solver.jl"))

# ─── LES ───

# ─── Thermal state (γ = 5/3 for Orszag-Tang) ───
const γ::FT = FT(5.0 / 3.0)
const Rg::FT = one(FT)          # Normalized gas constant
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = zero(FT)        # No Sutherland viscosity (inviscid)
const T_s::FT = one(FT)
const Pr::FT = 0.71

# ─── Mesh (relative to Benchmark/ORSZAG_TANG_3D/) ───
const mesh_dir = get(ENV, "OT3D_MESH_DIR", joinpath(@__DIR__, "MESH"))
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
include(joinpath(_project_root,"src","parallel","auto_partition.jl"))

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
function _ot3d_periodic_tuple(value)
    parts = split(lowercase(strip(value)), ',')
    length(parts) == 3 || error("OT3D_PERIODIC must contain three comma-separated booleans")
    return Tuple(part in ("true", "1", "yes", "on") for part in parts)
end
const Iperiodic = haskey(ENV, "OT3D_PERIODIC") ?
    _ot3d_periodic_tuple(ENV["OT3D_PERIODIC"]) : (true, true, true)

# ─── Flow control ───
const test_case::String = get(ENV, "OT3D_TEST_CASE", "OrszagTang3D")
const mesh::String = joinpath(_mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(_mesh_dir, "metrics_b0.h5")

const adaptive_dt::Bool = true
const CFL::FT = haskey(ENV, "OT3D_CFL") ? parse(FT, ENV["OT3D_CFL"]) : FT(0.3e0)
const LTS::Bool = false
const dt::FT = FT(1.0e-3)
const Time::FT = haskey(ENV, "OT3D_FINAL_TIME") ?
    parse(FT, ENV["OT3D_FINAL_TIME"]) : FT(1.0)
const maxStep::Int64 = profiling ? PROFILE_STEPS : 2000

# ─── Implicit Time Advancement ───
const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1.0e-3)

# ─── Output (relative to Benchmark/ORSZAG_TANG_3D/) ───
const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 200

const chk_out::Bool = false
const step_chk::Int64 = 1000
const restart::String = "none"

const average::Bool = false
const avg_step::Int64 = 10
const avg_total::Int64 = 1000
const avg_density_weighted::Bool = false


# ─── Filtering ───
const filtering::Bool = false
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 10
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(0.02e0)

# ─── Equation (Ncons/Nprim defined in physics.jl) ───
const viscous_order::Int64 = 2
const gg_blend::FT = zero(FT)

# ─── FVM Config ───
const eigen_reconstruction::Bool = ct_mode &&
    lowercase(get(ENV, "OT3D_CHARACTERISTIC", "false")) in ("1", "true", "yes", "on")
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

const _stats_file = get(ENV, "OT3D_STATS_FILE", joinpath(@__DIR__, "stats.dat"))
const _stats_interval = parse(Int, get(ENV, "OT3D_STATS_INTERVAL", "20"))
const _metric_ct_case = test_case in ("MetricCTUniform", "MetricCTAlfven")

# Write header once on rank 0
function _init_stats_file()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        open(_stats_file, "w") do io
            if _metric_ct_case
                println(io, metric_ct_stats_header())
            else
                println(io, "# 3D Orszag-Tang MHD turbulence diagnostics")
                println(io, "# step   time          dt            E_kin         E_th          E_mag         E_total       divB_cell_L2  divB_face_L2")
            end
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

# CT-compatible divergence from face-centered B. This is the discrete quantity
# preserved by the curl update; cell-centered central differences are diagnostic only.
function _compute_divB_face_local(blocks)
    divB_sq_local = zero(FT)
    for (bid, b) in blocks
        b.Bx_face === nothing && continue
        NGp = NG + 1
        nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG
        Bxf = Array(@view b.Bx_face[NGp:nx_end+1, NGp:ny_end, NGp:nz_end])
        Byf = Array(@view b.By_face[NGp:nx_end, NGp:ny_end+1, NGp:nz_end])
        Bzf = Array(@view b.Bz_face[NGp:nx_end, NGp:ny_end, NGp:nz_end+1])
        inv_volume = Array(@view b.Vol[NGp:nx_end, NGp:ny_end, NGp:nz_end])
        for kk in 1:b.Nz, jj in 1:b.Ny, ii in 1:b.Nx
            divB = inv_volume[ii,jj,kk] * (
                Bxf[ii+1,jj,kk] - Bxf[ii,jj,kk] +
                Byf[ii,jj+1,kk] - Byf[ii,jj,kk] +
                Bzf[ii,jj,kk+1] - Bzf[ii,jj,kk]
            )
            divB_sq_local += divB * divB
        end
    end
    return divB_sq_local
end

function _metric_ct_raw_minima_local(blocks)
    min_rho = Inf
    min_ei = Inf
    min_p = Inf
    valid = true
    for (_, b) in blocks
        lo = NG + 1
        u = Array(@view b.U[lo:NG+b.Nx, lo:NG+b.Ny, lo:NG+b.Nz, :])
        for k in 1:b.Nz, j in 1:b.Ny, i in 1:b.Nx
            raw = mhd_raw_thermo_components(
                u[i,j,k,1], u[i,j,k,2], u[i,j,k,3], u[i,j,k,4],
                u[i,j,k,5], u[i,j,k,6], u[i,j,k,7], u[i,j,k,8], γ,
            )
            valid &= isfinite(raw[1]) && isfinite(raw[4]) && isfinite(raw[5]) &&
                     raw[1] > 0 && raw[4] > 0 && raw[5] > 0
            min_rho = min(min_rho, raw[1])
            min_ei = min(min_ei, raw[4])
            min_p = min(min_p, raw[5])
        end
    end
    return (rho=min_rho, ei=min_ei, p=min_p, valid=valid)
end

function _metric_ct_conservative_energy_local(blocks)
    energy = 0.0
    for (_, b) in blocks
        lo = NG + 1
        u = Array(@view b.U[lo:NG+b.Nx, lo:NG+b.Ny, lo:NG+b.Nz, 5])
        inverse_volume = Array(@view b.Vol[
            lo:NG+b.Nx, lo:NG+b.Ny, lo:NG+b.Nz,
        ])
        for index in eachindex(u, inverse_volume)
            energy += Float64(u[index]) / Float64(inverse_volume[index])
        end
    end
    return energy
end

function _metric_ct_uniform_error_local(blocks)
    target = collect(metric_ct_uniform_state)
    errors = zeros(FT, length(target))
    core_errors = zeros(FT, length(target))
    locations = fill((0, 0, 0, 0), length(target))
    for (bid, b) in blocks
        lo = NG + 1
        q = Array(@view b.Q[lo:NG+b.Nx, lo:NG+b.Ny, lo:NG+b.Nz, :])
        for n in eachindex(target)
            component_error = abs.(view(q, :, :, :, n) .- target[n])
            component_max, component_index = findmax(component_error)
            if component_max > errors[n]
                errors[n] = component_max
                locations[n] = (bid, Tuple(component_index)...)
            end
            if b.Nx > 4 && b.Ny > 4 && b.Nz > 4
                core_errors[n] = max(
                    core_errors[n],
                    maximum(view(component_error, 3:b.Nx-2, 3:b.Ny-2, 3:b.Nz-2)),
                )
            end
        end
    end
    return errors, core_errors, locations
end

function in_situ_post_process(tt, activeTime, current_dt, blocks, world_rank, Block_Nprocs, block_comms)
    if tt % _stats_interval != 0 && tt != 1
        return
    end

    state_time = activeTime + current_dt
    raw_minima = (rho=NaN, ei=NaN, p=NaN)
    conservative_energy = NaN
    if _metric_ct_case
        local_raw = _metric_ct_raw_minima_local(blocks)
        raw_minima = (
            rho=MPI.Allreduce(Float64(local_raw.rho), MPI.MIN, MPI.COMM_WORLD),
            ei=MPI.Allreduce(Float64(local_raw.ei), MPI.MIN, MPI.COMM_WORLD),
            p=MPI.Allreduce(Float64(local_raw.p), MPI.MIN, MPI.COMM_WORLD),
        )
        raw_valid = MPI.Allreduce(
            local_raw.valid ? Int32(1) : Int32(0), MPI.MIN, MPI.COMM_WORLD,
        ) == Int32(1)
        raw_valid && all(isfinite, raw_minima) &&
            raw_minima.rho > 0 && raw_minima.ei > 0 && raw_minima.p > 0 || error(
                "metric CT raw-state validation failed at step $tt: $raw_minima",
            )
        conservative_energy = MPI.Allreduce(
            _metric_ct_conservative_energy_local(blocks),
            MPI.SUM,
            MPI.COMM_WORLD,
        )
    end

    ekin, eth, emag = compute_integral_energies(blocks, MPI.COMM_WORLD)
    divB_sq_local = _compute_divB_L2_local(blocks)
    divB_sq_global = MPI.Allreduce(Float64(divB_sq_local), MPI.SUM, MPI.COMM_WORLD)
    divB_face_sq_local = _compute_divB_face_local(blocks)
    divB_face_sq_global = MPI.Allreduce(
        Float64(divB_face_sq_local), MPI.SUM, MPI.COMM_WORLD,
    )

    # Cell count for normalization
    ncell_local = 0
    for (bid, b) in blocks
        ncell_local += b.Nx * b.Ny * b.Nz
    end
    ncell_global = MPI.Allreduce(ncell_local, MPI.SUM, MPI.COMM_WORLD)
    divB_L2 = sqrt(divB_sq_global / ncell_global)
    divB_face_L2 = ct_mode ? sqrt(divB_face_sq_global / ncell_global) : FT(NaN)
    uniform_errors = if test_case == "MetricCTUniform"
        local_errors, local_core_errors, uniform_locations =
            _metric_ct_uniform_error_local(blocks)
        global_errors = map(local_errors) do local_error
            MPI.Allreduce(Float64(local_error), MPI.MAX, MPI.COMM_WORLD)
        end
        global_core_errors = map(local_core_errors) do local_error
            MPI.Allreduce(Float64(local_error), MPI.MAX, MPI.COMM_WORLD)
        end
        (global_errors, global_core_errors, uniform_locations)
    else
        (Float64[], Float64[], NTuple{4,Int}[])
    end
    alfven_errors = if test_case == "MetricCTAlfven"
        metric_ct_error_norms(blocks, state_time)
    else
        (l1_transverse=0.0, l2_transverse=0.0, physical_volume=0.0)
    end

    etot = ekin + eth + emag

    if world_rank == 0
        @printf("  [diag] step=%-6d t=%.4f dt=%.3e  E_kin=%.4e E_th=%.4e E_mag=%.4e  E_tot=%.4e  |divB|cell=%.4e |divB|face=%.4e\n",
                tt, state_time, current_dt, ekin, eth, emag, etot, divB_L2, divB_face_L2)
        if test_case == "MetricCTUniform"
            global_errors, global_core_errors, uniform_locations = uniform_errors
            @printf(
                "  [metric-ct] uniform L_inf error=%.6e  components=%s\n",
                maximum(global_errors), string(global_errors),
            )
            @printf("  [metric-ct] core components=%s\n", string(global_core_errors))
            @printf("  [metric-ct] local max locations=%s\n", string(uniform_locations))
        elseif test_case == "MetricCTAlfven"
            @printf(
                "  [metric-ct] Alfven L1=%.6e L2=%.6e\n",
                alfven_errors.l1_transverse, alfven_errors.l2_transverse,
            )
        end
        open(_stats_file, "a") do io
            if _metric_ct_case
                uniform_core_linf = test_case == "MetricCTUniform" ?
                    maximum(uniform_errors[2]) : 0.0
                @printf(
                    io,
                    "%d %d %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e\n",
                    Nx_b[1], tt, state_time, current_dt,
                    alfven_errors.l1_transverse,
                    alfven_errors.l2_transverse,
                    uniform_core_linf,
                    raw_minima.rho,
                    raw_minima.ei,
                    raw_minima.p,
                    divB_face_L2,
                    conservative_energy,
                )
            else
                @printf(io, "%-7d %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e\n",
                        tt, state_time, current_dt, ekin, eth, emag, etot,
                        divB_L2, divB_face_L2)
            end
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
    println("  Flame3D — 3D Orszag-Tang MHD Turbulence Benchmark")
    println("=" ^ 70)
    print_gpu_backend_info()
    println("  Equation:   MHD (Ncons=$Ncons, Nprim=$Nprim)")
    println("  γ:          $γ")
    println("  Rg:         $Rg")
    println("  GLM cr:     $cr_glm")
    println("  Solver:     ", splitMethodID == Int32(4) ? "HLLD" : "Rusanov",
            " (splitMethodID=$splitMethodID)")
    println("  CT mode:    ", ct_mode, "  strict positivity: ", strict_ct_positivity)
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
    println("  3D ORSZAG-TANG COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", warmup_time)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
