# Benchmark/ORSZAG_TANG_2D/run.jl — 2D Orszag-Tang MHD vortex benchmark
# Usage: cd Benchmark/ORSZAG_TANG_2D && julia run.jl [nsteps]
#
# Classic 2D Orszag-Tang vortex (Orszag & Tang, 1979): the standard MHD test
# for turbulence and shock interactions. Initial condition is the analytic
# single-mode vortex: u=-sin(y), v=sin(x), Bx=-sin(y), By=sin(2x).
#
# Domain: [0, 2π]², triply periodic, γ = 5/3 (ideal MHD, inviscid)
# Mesh:   generate first with  julia gen_mesh.jl [Nx]   (default 256², quasi-2D)

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
const ct_mode::Bool = true      # Enable Constrained Transport
const strict_ct_positivity::Bool = true
const splitMethodID::Int32 = 4  # HLLD; required before solver.jl for strict static branches
const _ot_ct_scheme = lowercase(get(ENV, "OT2D_CT_SCHEME", "weno7"))
_ot_ct_scheme in ("sg07", "weno7") || error(
    "OT2D_CT_SCHEME must be sg07 or weno7, got '$_ot_ct_scheme'",
)
const ct_emf_scheme::Int32 = _ot_ct_scheme == "weno7" ? Int32(7) : Int32(2)
const initial_state_mode::Symbol = Symbol(lowercase(get(
    ENV, "OT2D_INITIAL_STATE_MODE", "quadrature6",
)))

# Project root for includes (two levels up from Benchmark/ORSZAG_TANG_2D/)
const _project_root = joinpath(@__DIR__, "..", "..")
# The structured solver specializes several kernels at include time.
# Define this compile-time physics flag before loading that stack.
const viscous::Bool = false
include(joinpath(_project_root,"src","core","equation_config.jl"))
include(joinpath(_project_root,"src","time","structured_rk3_solver.jl"))
include(joinpath(@__DIR__, "ot_diagnostics.jl"))

# ─── LES ───

# ─── Thermal state (γ = 5/3 for Orszag-Tang) ───
const γ::FT = FT(5.0 / 3.0)
const Rg::FT = one(FT)          # Normalized gas constant
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = zero(FT)        # No Sutherland viscosity (inviscid)
const T_s::FT = one(FT)
const Pr::FT = 0.71

# ─── Mesh (relative to Benchmark/ORSZAG_TANG_2D/) ───
const mesh_dir = abspath(get(
    ENV, "OT2D_MESH_DIR", joinpath(@__DIR__, "MESH"),
))
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
const Iperiodic = (true, true, true)   # Triply periodic (quasi-2D slab in z)

# ─── Flow control ───
const test_case::String = "OrszagTang"
const mesh::String = joinpath(_mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(_mesh_dir, "metrics_b0.h5")

const _ot_fixed_dt_raw = strip(get(ENV, "OT2D_FIXED_DT", ""))
const adaptive_dt::Bool = isempty(_ot_fixed_dt_raw)
const CFL::FT = haskey(ENV, "OT_CFL") ? parse(FT, ENV["OT_CFL"]) : FT(0.3e0)
const LTS::Bool = false
const dt::FT = adaptive_dt ? FT(1.0e-3) : parse(FT, _ot_fixed_dt_raw)
dt > zero(FT) || error("OT2D_FIXED_DT must be positive")
const Time::FT = haskey(ENV, "OT_FINAL_TIME") ?
    parse(FT, ENV["OT_FINAL_TIME"]) : FT(4.05)
const maxStep::Int64 = profiling ? PROFILE_STEPS : 100000

# ─── Implicit Time Advancement ───
const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1.0e-3)

# ─── Output (relative to Benchmark/ORSZAG_TANG_2D/) ───
const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 100000    # Disable regular PLT (only target-time snapshots)

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
    lowercase(get(
        ENV, "OT2D_CHARACTERISTIC",
        _ot_ct_scheme == "weno7" ? "true" : "false",
    )) in ("1", "true", "yes", "on")
const hybrid_ϕ1::FT = FT(0.01e0)
const hybrid_ϕ2::FT = one(FT)
const hybrid_ϕ3::FT = FT(10.0)
const Linear_ϕ::FT = FT(0.5)       # 50% upwind + 50% central (KEP) in smooth regions — reduces dissipation
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
# Wired through the solver's initial-state and post-step in-situ hooks.
#
# For a decaying (unforced) MHD run, total energy E_kin + E_th + E_mag should be
# approximately conserved. Cell-centered divergence is an observational diagnostic only.
# Strict CT acceptance uses the face-flux divergence with a 1e-12 threshold.

const _stats_file = abspath(get(
    ENV, "OT2D_STATS_FILE", joinpath(@__DIR__, "stats.dat"),
))
const _stats_interval = strict_ct_positivity ? 1 : 20

# Target snapshot times for the classic 2D Orszag-Tang validation (density field).
# A PLT file is emitted once activeTime crosses each target.
const _snapshot_times = let raw = get(
    ENV, "OT2D_SNAPSHOT_TIMES", "0.5,1.0,2.0,3.0,4.0",
)
    values = strip.(split(raw, ','))
    filter!(!isempty, values)
    sort!(unique!(parse.(FT, values)))
end
all(>=(zero(FT)), _snapshot_times) || error(
    "OT2D_SNAPSHOT_TIMES must contain non-negative times",
)
const _snapshot_pending = trues(length(_snapshot_times))   # false once emitted
const _snapshot_checkpoints = lowercase(get(
    ENV, "OT2D_SNAPSHOT_CHECKPOINTS", "false",
)) in ("1", "true", "yes", "on")

# Emit a PLT snapshot bypassing plotFile_multiblock's tt%step_plt guard.
# Uses a large pseudo-step offset so snapshot files don't collide with regular ones.
function _emit_snapshot(tt, time, blocks, world_rank, Nblocks, block_comms)
    snap_id = tt + 1_000_000   # unique, far above any real step count
    mkpath("./PLT")
    if world_rank == 0
        write_XDMF_multiblock(snap_id, time, Nblocks)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    for bid in sort(collect(keys(blocks)))
        b = blocks[bid]
        if b.id >= Nblocks
            continue
        end
        _write_plt_for_block(snap_id, b, block_comms[bid], Nblocks)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    if _snapshot_checkpoints
        mkpath(structured_checkpoint_dir())
        for bid in sort(collect(keys(blocks)))
            b = blocks[bid]
            b.id >= Nblocks && continue
            _write_chk_for_block(snap_id, b, block_comms[bid])
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

# Write header once on rank 0
function _init_stats_file()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        open(_stats_file, "w") do io
            println(io, "# 2D Orszag-Tang MHD vortex diagnostics")
            println(io, ot_stats_header())
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

# Face-form ∇·B from CT face-B arrays (should be ~machine zero if CT is correct)
function _compute_divB_face_local(blocks)
    divB_sq_local = zero(FT)
    for (bid, b) in blocks
        (b.Nx < 2 || b.Ny < 2 || b.Nz < 2) && continue
        b.Bx_face === nothing && continue
        NGp = NG + 1
        nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG
        # Legacy face-B fields store oriented magnetic flux Phi_B.
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

function _write_ot_diagnostics(
    tt, state_time, current_dt, ekin, eth, emag, e_from_q,
    e_cons_global, E_Q_minus_E_cons, min_rho_raw, min_ei_raw,
    min_p_raw, divB_L2, divBf_L2, world_rank,
)
    world_rank == 0 || return
    @printf("  [diag] step=%-6d t=%.4f dt=%.3e  E_kin=%.4e E_th=%.4e E_mag=%.4e  E_Q=%.4e E_cons=%.4e E_Q-E_cons=%.4e min_rho_raw=%.3e min_ei_raw=%.3e min_p_raw=%.3e |divB|cell=%.4e |divB|face=%.4e\n",
            tt, state_time, current_dt, ekin, eth, emag, e_from_q,
            e_cons_global, E_Q_minus_E_cons, min_rho_raw, min_ei_raw,
            min_p_raw, divB_L2, divBf_L2)
    open(_stats_file, "a") do io
        @printf(io,
            "%d %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
            tt, state_time, current_dt, ekin, eth, emag, e_from_q,
            e_cons_global, E_Q_minus_E_cons, min_rho_raw, min_ei_raw,
            min_p_raw, divB_L2, divBf_L2,
        )
    end
end

function _run_ot_diagnostics(
    tt, state_time, current_dt, blocks, world_rank, Block_Nprocs, block_comms,
)
    # ── Target-time snapshots: emit a PLT file once state_time crosses each ──
    for i in eachindex(_snapshot_times)
        if _snapshot_pending[i] && state_time >= _snapshot_times[i]
            _snapshot_pending[i] = false
            if world_rank == 0
                @printf("  [snapshot] emitting PLT at t=%.4f (step %d, target=%.1f)\n",
                        state_time, tt, _snapshot_times[i])
                flush(stdout)
            end
            _emit_snapshot(tt, state_time, blocks, world_rank, Nblocks, block_comms)
        end
    end

    # ── Regular stats logging (every _stats_interval steps) ──
    if tt % _stats_interval != 0 && tt != 1
        return
    end

    ekin, eth, emag = compute_integral_energies(blocks, MPI.COMM_WORLD)
    # Conservative energy: directly integrate U[5] (total conserved energy)
    e_cons_local = zero(FT)
    min_rho_raw_local = FT(Inf)
    min_ei_raw_local = FT(Inf)
    min_p_raw_local = FT(Inf)
    for (bid, b) in blocks
        NGp = NG + 1
        nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG
        U5_v = @view b.U[NGp:nx_end, NGp:ny_end, NGp:nz_end, 5]
        Vol_v = @view b.Vol[NGp:nx_end, NGp:ny_end, NGp:nz_end]
        e_cons_local += mapreduce((u, v) -> Float64(u) / Float64(v), +, U5_v, Vol_v)
        Uh = Array(@view b.U[NGp:nx_end, NGp:ny_end, NGp:nz_end, 1:5])
        Qh = Array(@view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 7:9])
        raw_minima = ot_raw_mhd_minima(Uh, Qh, FT(γ))
        min_rho_raw_local = min(min_rho_raw_local, raw_minima.rho)
        min_ei_raw_local = min(min_ei_raw_local, raw_minima.ei)
        min_p_raw_local = min(min_p_raw_local, raw_minima.p)
    end
    e_cons_global = MPI.Allreduce(Float64(e_cons_local), MPI.SUM, MPI.COMM_WORLD)
    min_rho_raw = MPI.Allreduce(Float64(min_rho_raw_local), MPI.MIN, MPI.COMM_WORLD)
    min_ei_raw = MPI.Allreduce(Float64(min_ei_raw_local), MPI.MIN, MPI.COMM_WORLD)
    min_p_raw = MPI.Allreduce(Float64(min_p_raw_local), MPI.MIN, MPI.COMM_WORLD)
    e_from_q = ekin + eth + emag
    E_Q_minus_E_cons = e_from_q - e_cons_global

    divB_sq_local = _compute_divB_L2_local(blocks)
    divB_sq_global = MPI.Allreduce(Float64(divB_sq_local), MPI.SUM, MPI.COMM_WORLD)
    divBf_sq_local = _compute_divB_face_local(blocks)
    divBf_sq_global = MPI.Allreduce(Float64(divBf_sq_local), MPI.SUM, MPI.COMM_WORLD)

    # Cell count for normalization
    ncell_local = 0
    for (bid, b) in blocks
        ncell_local += b.Nx * b.Ny * b.Nz
    end
    ncell_global = MPI.Allreduce(ncell_local, MPI.SUM, MPI.COMM_WORLD)
    divB_L2 = sqrt(divB_sq_global / ncell_global)
    divBf_L2 = sqrt(divBf_sq_global / ncell_global)

    _write_ot_diagnostics(
        tt, state_time, current_dt, ekin, eth, emag, e_from_q,
        e_cons_global, E_Q_minus_E_cons, min_rho_raw, min_ei_raw,
        min_p_raw, divB_L2, divBf_L2, world_rank,
    )
end

function in_situ_initial_process(
    activeTime, blocks, world_rank, Block_Nprocs, block_comms,
)
    _run_ot_diagnostics(
        0, activeTime, zero(FT), blocks,
        world_rank, Block_Nprocs, block_comms,
    )
end

function in_situ_post_process(
    tt, activeTime, current_dt, blocks, world_rank, Block_Nprocs, block_comms,
)
    state_time = activeTime + current_dt
    _run_ot_diagnostics(
        tt, state_time, current_dt, blocks,
        world_rank, Block_Nprocs, block_comms,
    )
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
    println("  Flame3D — 2D Orszag-Tang MHD Vortex Benchmark")
    println("=" ^ 70)
    print_gpu_backend_info()
    println("  Equation:   MHD (Ncons=$Ncons, Nprim=$Nprim)")
    println("  γ:          $γ")
    println("  Rg:         $Rg")
    println("  GLM cr:     $cr_glm")
    println("  Solver:     ", splitMethodID == Int32(4) ? "HLLD" : "Rusanov",
            " (splitMethodID=$splitMethodID)")
    println("  CT scheme:  ", _ot_ct_scheme,
            " (EMF=$ct_emf_scheme, characteristic=$ct_characteristic_reconstruction,",
            " cell-B=$ct_cell_b_recovery, primitive=$ct_primitive_recovery)")
    println("  Eigen recon:", eigen_reconstruction)
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
    println("  2D ORSZAG-TANG COMPLETE")
    println("=" ^ 70)
    @printf("  Total wall time:     %.3f s\n", warmup_time)
    println("=" ^ 70)
    flush(stdout)
end

MPI.Finalize()
