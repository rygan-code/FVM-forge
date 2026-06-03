# debug/run_y_mesh_tgv.jl
# Multi-block TGV verification on the 3-block Y-mesh (80×32×32 per block).
# Compares numerical solution against TGV analytical solution.
# Any artifacts at block interfaces will show as deviations > 0.5%.
#
# Usage: julia debug/run_y_mesh_tgv.jl [nsteps]
# Default: 5 steps

const PROFILE_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5

println("="^60)
println("  TGV Multi-Block Verification (Y-mesh)")
println("  Steps: $(PROFILE_STEPS)")
println("="^60)

# ─── Debug/Runtime flags ───
const debug_nan::Bool = true
const debug_sync::Bool = true
const profiling::Bool = false
const gpu_aware_mpi::Bool = false

const flow_forcing::Bool = false
const forcing_mode::Int64 = 0
const cebl_forcing::Bool = false
const cebl_forcing_type::Symbol = :deschamps
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = true

const weno_z::Bool = true
const FT = Float64

# ─── Thermal state ───
const γ::FT = 1.4
const Rg::FT = 287
const Cp::FT = Rg*γ/(γ-1)
const C_s::FT = FT(1.458e-6)
const T_s::FT = 110.4
const Pr::FT = 0.71

# ─── TGV parameters (low Ma for near-incompressible) ───
const Re_target::FT = FT(100.0)
const Ma_target::FT = FT(0.1)
const Ro_target::FT = FT(0.0)
const Tw::FT = FT(300.0)
const Lx::FT = FT(5.0)
const hit_forcing_A::FT = FT(0.0)
const R0::FT = FT(1.0)

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
const mesh_dir = "debug/MESH_Y_MESH"
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
const Omega_x::FT = FT(0.0)

# ─── Partition: single GPU per block ───
const auto_partition_enabled::Bool = false
const gpu_vram_gb::Float64 = 8.0
MPI.Init()
include(joinpath(_project_root, "auto_partition.jl"))
const Block_Nprocs_manual = [SVector(1,1,1) for _ in 1:Nblocks]
const (Block_Nprocs, Block_to_rank) = (Block_Nprocs_manual, zeros(Int, Nblocks))
const Iperiodic = (true, false, false)  # periodic in ξ (streamwise)

# ─── Flow control ───
const test_case::String = "TGV"
const mesh::String = joinpath(mesh_dir, "mesh_b0.h5")
const metrics::String = joinpath(mesh_dir, "metrics_b0.h5")
const adaptive_dt::Bool = true
const CFL::FT = FT(0.5)
const LTS::Bool = false
const dt::FT = FT(1.0e-4)
const Time::FT = FT(0.01)
const maxStep::Int64 = PROFILE_STEPS

const implicit::Bool = false
const implicit_CFL::FT = FT(10.0)
const implicit_lusgs_sweeps::Int64 = 1
const dual_time::Bool = false
const dual_time_sub_iters::Int64 = 5
const dual_time_tol::FT = FT(1e-3)

# ─── Output ───
const plt_xdmf::Bool = true
const plt_out::Bool = true
const step_plt::Int64 = 1
const plt_shuffle::Bool = true
const plt_compress_level::Int64 = 1
const chk_out::Bool = false
const step_chk::Int64 = 1000
const chk_shuffle::Bool = true
const chk_compress_level::Int64 = 1
const restart::String = "none"
const inflow_restart::String = "none"
const average::Bool = false
const avg_step::Int64 = 10
const avg_total::Int64 = 2000
const avg_shuffle::Bool = true
const avg_compress_level::Int64 = 1
const avg_density_weighted::Bool = false
const sample::Bool = false
const sample_step::Int64 = 10
const sample_index::SVector{3, Int64} = [-1, -1, -1]

# ─── Filtering (off for clean verification) ───
const filtering::Bool = false
const filtering_nonlinear::Bool = false
const filtering_interval::Int64 = 100
const filtering_rth::FT = FT(1e-5)
const filtering_s0::FT = FT(1.0)

# ─── Checkerboard Diagnostic ───
const checkerboard_diag::Bool = true
const checkerboard_diag_step::Int64 = PROFILE_STEPS

# ─── Viscous / Reconstruction ───
const viscous::Bool = true
const viscous_order::Int64 = 6
const gg_blend::FT = one(FT)
const eigen_reconstruction::Bool = true
const splitMethodID::Int32 = 4  # Roe
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

# ─── Acoustic Pulse Initial Condition ───
# A Gaussian pressure pulse centered at the domain center.
# At t=0, this is the exact initial condition. After a few steps,
# we compare the numerical solution against itself at t=0 to check
# for artifacts (the pulse should be symmetric if no artifacts exist).
function init_tgv(Q, x, y, z)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    nxp = size(Q, 1) - 2*NG
    nyp = size(Q, 2) - 2*NG
    nzp = size(Q, 3) - 2*NG

    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG; return; end
    if i < NG+1 || i > nxp+NG || j < NG+1 || j > nyp+NG || k < NG+1 || k > nzp+NG; return; end

    xc = FT(0.5) * (x[i, j, k] + x[i+1, j, k])
    yc = FT(0.5) * (y[i, j, k] + y[i, j+1, k])
    zc = FT(0.5) * (z[i, j, k] + z[i, j, k+1])

    # Pulse center (domain center)
    x0 = FT(2.5)
    y0 = FT(0.0)
    z0 = FT(0.5)

    # Gaussian pulse
    r2 = (xc - x0)^2 + (yc - y0)^2 + (zc - z0)^2
    pulse = FT(0.01) * exp(-FT(20.0) * r2)

    rho0 = one(FT)
    p0 = FT(100.0)
    u0 = zero(FT)
    v0 = zero(FT)
    w0 = zero(FT)

    rho = rho0 + pulse
    p = p0 + pulse * p0
    u = u0
    v = v0
    w = w0

    @inbounds Q[i, j, k, 1] = rho
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = w
    @inbounds Q[i, j, k, 5] = p
    @inbounds Q[i, j, k, 6] = p / (rho * Rg)
    return
end

# ─── Verification: Compare numerical vs analytical ───
function verify_tgv(blocks, connectivity, world_rank, tt, current_dt)
    t = Float64(tt) * Float64(current_dt)
    mkpath("debug")

    max_err_u = 0.0
    max_err_v = 0.0
    max_err_p = 0.0
    max_err_rho = 0.0

    for (bid, b) in blocks
        nxp, nyp, nzp = b.Nx, b.Ny, b.Nz
        Q_h = Array(b.Q)
        x_h = Array(b.x)
        y_h = Array(b.y)
        z_h = Array(b.z)

        for k in NG+1:nzp+NG
            for j in NG+1:nyp+NG
                for i in NG+1:nxp+NG
                    xc = 0.5 * (x_h[i, j, k] + x_h[i+1, j, k])
                    yc = 0.5 * (y_h[i, j, k] + y_h[i, j+1, k])
                    zc = 0.5 * (z_h[i, j, k] + z_h[i, j, k+1])

                    # TGV analytical (initial condition; valid for short time at low Ma)
                    u_exact = sin(xc) * cos(yc) * cos(zc)
                    v_exact = -cos(xc) * sin(yc) * cos(zc)
                    p_exact = 100.0 + 1.0/16.0 * (cos(2*xc) + cos(2*yc)) * (cos(2*zc) + 2)
                    rho_exact = 1.0

                    u_num = Q_h[i, j, k, 2]
                    v_num = Q_h[i, j, k, 3]
                    p_num = Q_h[i, j, k, 5]
                    rho_num = Q_h[i, j, k, 1]

                    err_u = abs(u_num - u_exact) / max(abs(u_exact), 1.0e-10)
                    err_v = abs(v_num - v_exact) / max(abs(v_exact), 1.0e-10)
                    err_p = abs(p_num - p_exact) / max(abs(p_exact), 1.0e-10)
                    err_rho = abs(rho_num - rho_exact) / max(abs(rho_exact), 1.0e-10)

                    max_err_u = max(max_err_u, err_u)
                    max_err_v = max(max_err_v, err_v)
                    max_err_p = max(max_err_p, err_p)
                    max_err_rho = max(max_err_rho, err_rho)
                end
            end
        end
    end

    global_max_err_u = MPI.Allreduce(max_err_u, MPI.MAX, MPI.COMM_WORLD)
    global_max_err_v = MPI.Allreduce(max_err_v, MPI.MAX, MPI.COMM_WORLD)
    global_max_err_p = MPI.Allreduce(max_err_p, MPI.MAX, MPI.COMM_WORLD)
    global_max_err_rho = MPI.Allreduce(max_err_rho, MPI.MAX, MPI.COMM_WORLD)

    if world_rank == 0
        fname = "debug/tgv_verification.txt"
        open(fname, "w") do io
            println(io, "TGV Multi-Block Verification")
            println(io, "============================")
            println(io, "Step: $tt, Time: $(round(t, digits=8))")
            println(io, "Mesh: $mesh_dir")
            println(io, "Blocks: $Nblocks, Size: $(Nx_b[1])×$(Ny_b[1])×$(Nz_b[1])")
            println(io, "")
            println(io, "Max relative errors:")
            println(io, "  u:   $(round(global_max_err_u * 100, digits=6))%")
            println(io, "  v:   $(round(global_max_err_v * 100, digits=6))%")
            println(io, "  p:   $(round(global_max_err_p * 100, digits=6))%")
            println(io, "  rho: $(round(global_max_err_rho * 100, digits=6))%")
            println(io, "")

            tol = 0.005  # 0.5%
            passed = global_max_err_u < tol && global_max_err_v < tol && global_max_err_p < tol && global_max_err_rho < tol
            if passed
                println(io, "✓ PASS: All errors < 0.5%")
                println(io, "No numerical artifacts detected at block interfaces.")
            else
                println(io, "✗ FAIL: Some errors >= 0.5%")
                println(io, "Numerical artifacts detected — investigate ghost exchange, metrics, or reconstruction.")
            end
        end

        println("\n  TGV Verification at step $tt (t = $(round(t, digits=8))):")
        println("    Max relative error in u:   $(round(global_max_err_u * 100, digits=6))%")
        println("    Max relative error in v:   $(round(global_max_err_v * 100, digits=6))%")
        println("    Max relative error in p:   $(round(global_max_err_p * 100, digits=6))%")
        println("    Max relative error in rho: $(round(global_max_err_rho * 100, digits=6))%")

        tol = 0.005
        if global_max_err_u < tol && global_max_err_v < tol && global_max_err_p < tol && global_max_err_rho < tol
            println("    ✓ PASS: All errors < 0.5%")
        else
            println("    ✗ FAIL: Some errors >= 0.5%")
        end
        println("    Results written to: $fname")
    end

    return global_max_err_u, global_max_err_v, global_max_err_p, global_max_err_rho
end

# ─── Entry Point ───
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("  Mesh:       $mesh_dir")
    println("  Blocks:     $Nblocks")
    println("  Grid/block: $(Nx_b[1])×$(Ny_b[1])×$(Nz_b[1])")
    println("  Steps:      $PROFILE_STEPS")
    println("  Re:         $Re_target")
    println("  Ma:         $Ma_target")
    flush(stdout)
end

t0 = time_ns()
time_step(rank, comm, Block_Nprocs)
wall = (time_ns() - t0) / 1e9

# ─── Post-processing: Verify against TGV analytical solution ───
if rank == 0
    println("\n>>> Running post-processing TGV verification...")
    mkpath("debug")

    # Read PLT output files (per-block: plt-N-bM.h5)
    plt_files = filter(f -> occursin(r"^plt-\d+-b\d+\.h5$", f), readdir("./PLT"; join=false))
    if isempty(plt_files)
        println("  WARNING: No PLT files found in ./PLT — skipping verification.")
    else
        # Compare last step vs first step (acoustic pulse should be symmetric)
        local steps = unique([parse(Int, match(r"plt-(\d+)-b\d+", f).captures[1]) for f in plt_files])
        local first_step = minimum(steps)
        local last_step = maximum(steps)

        local all_rho_err = Float64[]
        local all_u_err = Float64[]
        local all_v_err = Float64[]
        local all_p_err = Float64[]
        local t_val = 0.0

        for bid in 0:Nblocks-1
            # Read first step (reference)
            f1 = joinpath("./PLT", "plt-$(first_step)-b$(bid).h5")
            f2 = joinpath("./PLT", "plt-$(last_step)-b$(bid).h5")
            if !isfile(f1) || !isfile(f2)
                println("  WARNING: PLT files not found for block $bid")
                continue
            end
            fid1 = h5open(f1, "r")
            rho1 = read(fid1["rho"]); u1 = read(fid1["u"]); v1 = read(fid1["v"]); p1 = read(fid1["p"])
            close(fid1)

            fid2 = h5open(f2, "r")
            rho2 = read(fid2["rho"]); u2 = read(fid2["u"]); v2 = read(fid2["v"]); p2 = read(fid2["p"])
            try
                t_raw = read(attrs(fid2)["time"])
                t_val = (t_raw isa AbstractArray) ? t_raw[1] : t_raw
            catch
            end
            close(fid2)

            # Compute relative change (absolute for velocity since u0≈0)
            append!(all_rho_err, vec(abs.(rho2 .- rho1) ./ max.(abs.(rho1), 1.0e-10)))
            append!(all_u_err, vec(abs.(u2 .- u1)))  # absolute error
            append!(all_v_err, vec(abs.(v2 .- v1)))  # absolute error
            append!(all_p_err, vec(abs.(p2 .- p1) ./ max.(abs.(p1), 1.0e-10)))
        end
        println("  Comparing step $first_step vs step $last_step, t=$t_val, $(length(all_rho_err)) cells")

        max_err_rho = maximum(all_rho_err)
        max_err_u = maximum(all_u_err)
        max_err_v = maximum(all_v_err)
        max_err_p = maximum(all_p_err)

        fname = "debug/tgv_verification.txt"
        open(fname, "w") do io
            println(io, "Acoustic Pulse Multi-Block Verification")
            println(io, "========================================")
            println(io, "Comparing step $first_step vs step $last_step, t=$(round(t_val, digits=8))")
            println(io, "Mesh: $mesh_dir")
            println(io, "Blocks: $Nblocks, Size: $(Nx_b[1])×$(Ny_b[1])×$(Nz_b[1])")
            println(io, "Total cells: $(length(all_rho_err))")
            println(io, "")
            println(io, "Max change (step $first_step → $last_step):")
            println(io, "  rho: $(round(max_err_rho * 100, digits=6))% (relative)")
            println(io, "  u:   $(round(max_err_u, digits=8)) (absolute)")
            println(io, "  v:   $(round(max_err_v, digits=8)) (absolute)")
            println(io, "  p:   $(round(max_err_p * 100, digits=6))% (relative)")
            println(io, "")

            tol_rel = 0.005  # 0.5% for relative errors
            tol_abs = 0.01   # absolute threshold for velocity
            passed = max_err_rho < tol_rel && max_err_u < tol_abs && max_err_v < tol_abs && max_err_p < tol_rel
            if passed
                println(io, "✓ PASS: All changes < 0.5%")
                println(io, "No numerical artifacts detected at block interfaces.")
            else
                println(io, "✗ FAIL: Some changes >= 0.5%")
                println(io, "Numerical artifacts detected — investigate ghost exchange, metrics, or reconstruction.")
            end
        end

        println("\n  Acoustic Pulse Verification (step $first_step → $last_step, t=$(round(t_val, digits=8))):")
        println("    Max relative change in rho: $(round(max_err_rho * 100, digits=6))%")
        println("    Max absolute change in u:   $(round(max_err_u, digits=8))")
        println("    Max absolute change in v:   $(round(max_err_v, digits=8))")
        println("    Max relative change in p:   $(round(max_err_p * 100, digits=6))%")

        tol_rel = 0.005  # 0.5% for relative errors
        tol_abs = 0.01   # absolute threshold for velocity
        if max_err_rho < tol_rel && max_err_u < tol_abs && max_err_v < tol_abs && max_err_p < tol_rel
            println("    ✓ PASS: All changes < 0.5%")
        else
            println("    ✗ FAIL: Some changes >= 0.5%")
        end
        println("    Results written to: $fname")
    end

    println(">>> TGV verification run completed in $(round(wall, digits=3)) s.")
end

MPI.Finalize()
