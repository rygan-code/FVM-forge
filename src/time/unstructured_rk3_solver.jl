# unstruct_time_step.jl — Main time-stepping loop for unstructured FVM
# Part of the second-order unstructured FVM branch
#
# Shared infrastructure (must be included BEFORE this file):
#   gpu_backend.jl, physics.jl, bc_types.jl, Riemann_Solver.jl,
#   unstruct_mesh.jl, unstruct_reconstruct.jl, unstruct_div.jl,
#   unstruct_bc.jl, unstruct_io.jl, unstruct_gradient.jl
#
# Implements RK3 explicit time stepping for Euler equations
# on unstructured grids.

using StaticArrays

if !@isdefined(resistive)
    const resistive = false
end

# ═════════════════════════════════════════════════════════════
# Initialization kernel — Sod shock tube
# ═════════════════════════════════════════════════════════════

function init_sod_unstruct_kernel!(Q, cell_cx, ncell_tot::Int)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if c > ncell_tot
        return
    end
    @inbounds xc = cell_cx[c]
    if xc < FT(0.5)
        ρ = one(FT); u = zero(FT); p = one(FT)
    else
        ρ = FT(0.125); u = zero(FT); p = FT(0.1)
    end
    @inbounds Q[c, 1] = ρ
    @inbounds Q[c, 2] = u
    @inbounds Q[c, 3] = zero(FT)
    @inbounds Q[c, 4] = zero(FT)
    @inbounds Q[c, 5] = p
    @inbounds Q[c, 6] = p / (ρ * Rg)
    return
end

# ═════════════════════════════════════════════════════════════
# Initialization kernel — TGV (Taylor-Green Vortex)
# ═════════════════════════════════════════════════════════════

function init_tgv_unstruct_kernel!(Q, cell_cx, cell_cy, cell_cz, ncell_tot::Int)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if c > ncell_tot
        return
    end
    @inbounds x = cell_cx[c]
    @inbounds y = cell_cy[c]
    @inbounds z = cell_cz[c]

    ρ0 = one(FT)
    V0 = FT(0.1)
    p0 = FT(1.0) / γ  # Ma = 0.1 → p0 = ρ0 c² / γ = 1/γ for c=1

    u =  V0 * sin(x) * cos(y) * cos(z)
    v = -V0 * cos(x) * sin(y) * cos(z)
    w = zero(FT)
    p = p0 + ρ0 * V0^2 / FT(16.0) * (cos(2*x) + cos(2*y)) * (cos(2*z) + FT(2.0))

    @inbounds Q[c, 1] = ρ0
    @inbounds Q[c, 2] = u
    @inbounds Q[c, 3] = v
    @inbounds Q[c, 4] = w
    @inbounds Q[c, 5] = p
    @inbounds Q[c, 6] = p / (ρ0 * Rg)
    return
end

function init_brio_wu_unstruct_kernel!(Q, cell_cx, ncell_tot::Int)
    cell = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    cell > ncell_tot && return
    @inbounds x = cell_cx[cell]
    if x < FT(0.5)
        density = one(FT)
        pressure = one(FT)
        magnetic_y = SQRT_MU0_SI
    else
        density = FT(0.125)
        pressure = FT(0.1)
        magnetic_y = -SQRT_MU0_SI
    end
    @inbounds begin
        Q[cell,1] = density
        Q[cell,2] = zero(FT)
        Q[cell,3] = zero(FT)
        Q[cell,4] = zero(FT)
        Q[cell,5] = pressure
        Q[cell,6] = pressure/(density*Rg)
        Q[cell,7] = FT(0.75) * SQRT_MU0_SI
        Q[cell,8] = magnetic_y
        Q[cell,9] = zero(FT)
        Q[cell,10] = zero(FT)
    end
    return
end

# ═════════════════════════════════════════════════════════════
# Initialize wrapper
# ═════════════════════════════════════════════════════════════

function initialize_unstruct!(block::UnstructBlock, test_case_::String)
    nb = cld(block.ncell_tot, block.nthreads_cell)
    if test_case_ == "BrioWu"
        @static if equation_type == :MHD
            @gpu_launch threads=block.nthreads_cell blocks=nb init_brio_wu_unstruct_kernel!(
                block.Q, block.cell_cx, block.ncell_tot)
        else
            error("BrioWu initialization requires equation_type=:MHD")
        end
    elseif test_case_ == "Sod"
        @gpu_launch threads=block.nthreads_cell blocks=nb init_sod_unstruct_kernel!(
            block.Q, block.cell_cx, block.ncell_tot)
    elseif test_case_ == "TGV"
        @gpu_launch threads=block.nthreads_cell blocks=nb init_tgv_unstruct_kernel!(
            block.Q, block.cell_cx, block.cell_cy, block.cell_cz, block.ncell_tot)
    else
        error("Unknown test case: $test_case_")
    end
    unstruct_prim2c!(block)
    return
end

# ═════════════════════════════════════════════════════════════
# NaN check kernel
# ═════════════════════════════════════════════════════════════

function check_nan_kernel!(flag, Q, ncell::Int)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if c > ncell
        return
    end
    @inbounds for n in 1:Nprim
        if !isfinite(Q[c, n])
            @inbounds flag[1] = Int32(1)
            return
        end
    end
    return
end

function check_nan_unstruct(block::UnstructBlock)
    flag = GPUArray([Int32(0)])
    nb = cld(block.ncell, block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=nb check_nan_kernel!(
        flag, block.Q, block.ncell)
    return Array(flag)[1] != Int32(0)
end

# ═════════════════════════════════════════════════════════════
# RK3 coefficients (same as structured solver)
# ═════════════════════════════════════════════════════════════
const UNSTRUCT_RK_A = (FT(1.0), FT(0.25), FT(2.0/3.0))

# ═════════════════════════════════════════════════════════════
# Main time-stepping function
# ═════════════════════════════════════════════════════════════

"""
    unstruct_time_step(block, test_case, Time, maxStep, CFL, step_plt, out_dir, order, splitMethodID)

Run RK3 explicit time stepping on an UnstructBlock.
- order: 1 (first-order upwind) or 2 (second-order MUSCL)
- splitMethodID: 0=Rusanov, 1=HLLC
"""
function unstruct_time_step(block::UnstructBlock, test_case_::String,
                             Time::FT, maxStep::Int, CFL_::FT,
                             step_plt::Int, out_dir::String,
                             order::Int, splitMethodID_::Int32;
                             comm_cart=nothing, gpu_aware::Bool=false,
                             initialize_state::Bool=true,
                             write_output::Bool=true)
    Time >= zero(FT) || throw(ArgumentError("Time must be non-negative"))
    maxStep >= 0 || throw(ArgumentError("maxStep must be non-negative"))
    CFL_ > zero(FT) || throw(ArgumentError("CFL must be positive"))
    order in (1, 2) || throw(ArgumentError(
        "unstructured reconstruction order must be 1 or 2",
    ))
    splitMethodID_ in (Int32(0), Int32(1)) || throw(ArgumentError(
        "unstructured splitMethodID must be 0 (Rusanov) or 1 (HLLC)",
    ))
    step_plt > 0 || throw(ArgumentError("step_plt must be positive"))
    @static if equation_type == :MHD
        if resistive && !(isfinite(η_mhd) && η_mhd > zero(FT))
            throw(ArgumentError(
                "unstructured resistive MHD requires finite η_mhd > 0"))
        end
        splitMethodID_ == Int32(0) || throw(ArgumentError(
            "unstructured GLM-MHD currently supports Rusanov flux only"))
    end

    # ── Initialize ──
    write_output && mkpath(out_dir)
    if initialize_state
        @info "Initializing unstructured solver: test_case=$test_case_, order=$order"
        initialize_unstruct!(block, test_case_)
    end
    validate_unstruct_boundary_support(block)

    # Initial ghost fill (MPI + physical BC)
    if comm_cart !== nothing
        sync_unstruct_ghost!(block, comm_cart; gpu_aware=gpu_aware)
    else
        fill_unstruct_ghost!(block)
    end
    @static if equation_type == :MHD && ct_mode
        initialize_unstruct_ct!(block)
        if comm_cart !== nothing
            sync_unstruct_ct_face_flux!(block,comm_cart;update_backup=true)
            unstruct_reconstruct_ct_B!(block;update_primitive=true)
        end
    end

    activeTime = zero(FT)
    tt = 0

    @info "Starting unstructured time loop: Time=$Time, maxStep=$maxStep, CFL=$CFL_"
    @info "  ncell=$(block.ncell), nface=$(block.nface), nghost=$(block.nghost)"

    local dt::FT = zero(FT)
    local ch_glm::FT = zero(FT)
    while activeTime < Time && tt < maxStep
        tt += 1

        for KRK in 1:3
            rk_a = UNSTRUCT_RK_A[KRK]

            if KRK == 1
                # Compute dt
                @static if equation_type == :MHD
                    local_ch = unstruct_compute_ch!(block)
                    ch_glm = comm_cart === nothing ? local_ch :
                        MPI.Allreduce(local_ch, MPI.MAX, comm_cart)
                end
                local_dt = unstruct_compute_dt!(block, CFL_, ch_glm)
                dt = comm_cart === nothing ? local_dt :
                    MPI.Allreduce(local_dt, MPI.MIN, comm_cart)
                if !isfinite(dt) || dt <= zero(FT)
                    error("Invalid unstructured dt=$dt at step $tt")
                end
                dt = min(dt, Time - activeTime)

                # Exact GLM source solve, symmetrically split around SSP-RK3.
                @static if equation_type == :MHD && !ct_mode
                    unstruct_apply_glm_damping!(block, FT(0.5)*dt, ch_glm)
                end

                # SSP-RK3 uses the post-source state as its time-level backup.
                copyto!(block.Un, block.U)
                @static if equation_type == :MHD && ct_mode
                    backup_unstruct_ct!(block)
                end
            end

            # Fill ghost cells (MPI exchange + physical BC)
            if comm_cart !== nothing
                sync_unstruct_ghost!(block, comm_cart; gpu_aware=gpu_aware)
            else
                fill_unstruct_ghost!(block)
            end

            # Gradients are required by MUSCL and by the viscous face flux.
            if order == 2 || viscous || resistive
                compute_lsq_gradient!(block)
                if comm_cart !== nothing
                    exchange_unstruct_gradient(block, comm_cart; gpu_aware=gpu_aware)
                end
                fill_unstruct_periodic_gradient!(block)
            end
            if order == 2
                compute_venkat_limiter!(block)
                if comm_cart !== nothing
                    exchange_unstruct_limiter(block,comm_cart;gpu_aware=gpu_aware)
                end
                fill_unstruct_periodic_limiter!(block)
            end

            # Compute face fluxes
            compute_unstruct_flux!(block, order, splitMethodID_, ch_glm)
            compute_unstruct_viscous_flux!(block)

            @static if equation_type == :MHD && ct_mode
                unstruct_ct_stage!(block,dt,rk_a;comm_cart=comm_cart)
            end

            # Divergence + RK + clip + c2Prim
            unstruct_div_rk_clip_prim!(
                block, dt, rk_a; step=tt, rk_stage=KRK,
            )
        end
        @static if equation_type == :MHD && !ct_mode
            unstruct_apply_glm_damping!(block, FT(0.5)*dt, ch_glm)
        end
        activeTime += dt

        # Output & diagnostics
        if tt % step_plt == 0 || tt == 1
            has_nan = check_nan_unstruct(block)
            if has_nan
                error("NaN detected in unstructured state at step $tt")
            end
            # Compute mass/momentum/energy for monitoring
            Q_h = Array(block.Q)
            ρ_avg = sum(Q_h[1:block.ncell, 1]) / block.ncell
            p_avg = sum(Q_h[1:block.ncell, 5]) / block.ncell
            @info "Step $tt: t=$(round(activeTime, digits=6)), ρ_avg=$(round(ρ_avg, digits=6)), p_avg=$(round(p_avg, digits=6))"

            # Write VTK
            if write_output
                write_vtu(block, joinpath(out_dir, "plt-$(tt).vtu"), tt)
            end
        end
    end

    # Final output
    check_nan_unstruct(block) && error(
        "NaN detected in final unstructured state at step $tt",
    )
    if write_output
        write_vtu(block, joinpath(out_dir, "plt-final.vtu"), tt)
    end
    @info "Unstructured solver finished: $tt steps, t=$activeTime"
    return (
        steps=tt, time=activeTime, final_dt=dt,
        completed=activeTime >= Time,
    )
end
