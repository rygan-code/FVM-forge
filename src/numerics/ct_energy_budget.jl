# CT discrete energy-budget diagnostics.
#
# The diagnostic is intentionally read-only. It never changes U, face-B, or
# any flux. It is compiled into the solver include chain but allocates no
# storage and launches no kernels unless OPENCFD_CT_ENERGY_BUDGET=true.

if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end

const CT_ENERGY_BUDGET_STATE_LEN = 4
const CT_ENERGY_BUDGET_FLUX_LEN = 4

const CT_ENERGY_STATE_TOTAL = 1
const CT_ENERGY_STATE_KINETIC = 2
const CT_ENERGY_STATE_MAGNETIC = 3
const CT_ENERGY_STATE_INTERNAL = 4

const CT_ENERGY_FLUX_IDEAL = 1
const CT_ENERGY_FLUX_NONIDEAL_BEFORE = 2
const CT_ENERGY_FLUX_TOTAL = 3
const CT_ENERGY_FLUX_NONIDEAL_AFTER = 4

@inline function _ct_energy_budget_indices()
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    return i, j, k
end

function ct_energy_budget_state_kernel!(acc, U, Q, inverse_volume, nxp, nyp, nzp)
    i, j, k = _ct_energy_budget_indices()
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii, jj, kk = i + Int32(NG), j + Int32(NG), k + Int32(NG)
    @inbounds begin
        rho = U[ii, jj, kk, 1]
        rho_inv = rho > zero(FT) ? inv(rho) : zero(FT)
        u = U[ii, jj, kk, 2] * rho_inv
        v = U[ii, jj, kk, 3] * rho_inv
        w = U[ii, jj, kk, 4] * rho_inv
        bx = Q[ii, jj, kk, QBX]
        by = Q[ii, jj, kk, QBY]
        bz = Q[ii, jj, kk, QBZ]
        volume = inv(inverse_volume[ii, jj, kk])
        kinetic_density = FT(0.5) * rho * (u*u + v*v + w*w)
        magnetic_density = FT(0.5) * INV_MU0_SI * (
            bx*bx + by*by + bz*bz)
        total = U[ii, jj, kk, 5] * volume
        kinetic = kinetic_density * volume
        magnetic = magnetic_density * volume
        internal = (U[ii, jj, kk, 5] - kinetic_density - magnetic_density) * volume
    end

    gpu_atomic_add!(acc, CT_ENERGY_STATE_TOTAL, total)
    gpu_atomic_add!(acc, CT_ENERGY_STATE_KINETIC, kinetic)
    gpu_atomic_add!(acc, CT_ENERGY_STATE_MAGNETIC, magnetic)
    gpu_atomic_add!(acc, CT_ENERGY_STATE_INTERNAL, internal)
    return
end

function ct_energy_budget_total_kernel!(acc, U, inverse_volume, nxp, nyp, nzp)
    i, j, k = _ct_energy_budget_indices()
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii, jj, kk = i + Int32(NG), j + Int32(NG), k + Int32(NG)
    @inbounds total = U[ii, jj, kk, 5] * inv(inverse_volume[ii, jj, kk])
    gpu_atomic_add!(acc, CT_ENERGY_STATE_TOTAL, total)
    return
end

function ct_energy_budget_flux_kernel!(
    acc, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z,
    dt_stage::FT, rk_a::FT, nxp, nyp, nzp, mode::Int32,
)
    i, j, k = _ct_energy_budget_indices()
    if i > nxp || j > nyp || k > nzp
        return
    end

    offset = STRUCTURED_FLUX_TANGENTIAL_HALO
    @inbounds begin
        inviscid =
            Fx[i, j + offset, k + offset, 5] -
            Fx[i + Int32(1), j + offset, k + offset, 5] +
            Fy[i + offset, j, k + offset, 5] -
            Fy[i + offset, j + Int32(1), k + offset, 5] +
            Fz[i + offset, j + offset, k, 5] -
            Fz[i + offset, j + offset, k + Int32(1), 5]
        diffusive =
            Fv_x[i, j + offset, k + offset, 5] -
            Fv_x[i + Int32(1), j + offset, k + offset, 5] +
            Fv_y[i + offset, j, k + offset, 5] -
            Fv_y[i + offset, j + Int32(1), k + offset, 5] +
            Fv_z[i + offset, j + offset, k, 5] -
            Fv_z[i + offset, j + offset, k + Int32(1), 5]
        ideal_delta = rk_a * dt_stage * inviscid
        nonideal_delta = -rk_a * dt_stage * diffusive
    end

    if mode == Int32(1)
        gpu_atomic_add!(acc, CT_ENERGY_FLUX_IDEAL, ideal_delta)
        gpu_atomic_add!(acc, CT_ENERGY_FLUX_NONIDEAL_BEFORE, nonideal_delta)
    else
        gpu_atomic_add!(
            acc, CT_ENERGY_FLUX_TOTAL, ideal_delta + nonideal_delta,
        )
        gpu_atomic_add!(acc, CT_ENERGY_FLUX_NONIDEAL_AFTER, nonideal_delta)
    end
    return
end

function ct_energy_budget_state_sample!(
    acc, U, Q, inverse_volume, nxp, nyp, nzp, threads,
)
    fill!(acc, zero(FT))
    blocks = (
        cld(nxp, threads[1]),
        cld(nyp, threads[2]),
        cld(nzp, threads[3]),
    )
    @gpu_launch threads=threads blocks=blocks ct_energy_budget_state_kernel!(
        acc, U, Q, inverse_volume,
        Int32(nxp), Int32(nyp), Int32(nzp),
    )
    gpu_sync()
    return Float64.(Array(acc))
end

function ct_energy_budget_total_sample!(
    acc, U, inverse_volume, nxp, nyp, nzp, threads,
)
    fill!(acc, zero(FT))
    blocks = (
        cld(nxp, threads[1]),
        cld(nyp, threads[2]),
        cld(nzp, threads[3]),
    )
    @gpu_launch threads=threads blocks=blocks ct_energy_budget_total_kernel!(
        acc, U, inverse_volume,
        Int32(nxp), Int32(nyp), Int32(nzp),
    )
    gpu_sync()
    return Float64.(Array(acc))
end

function ct_energy_budget_reset_flux!(acc)
    fill!(acc, zero(FT))
    return nothing
end

function ct_energy_budget_accumulate_flux!(
    acc, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z,
    dt_stage, rk_a, nxp, nyp, nzp, threads, mode::Int32,
)
    blocks = (
        cld(nxp, threads[1]),
        cld(nyp, threads[2]),
        cld(nzp, threads[3]),
    )
    @gpu_launch threads=threads blocks=blocks ct_energy_budget_flux_kernel!(
        acc, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z,
        FT(dt_stage), FT(rk_a),
        Int32(nxp), Int32(nyp), Int32(nzp), mode,
    )
    return nothing
end

function ct_energy_budget_read_flux!(acc)
    gpu_sync()
    return Float64.(Array(acc))
end

function ct_energy_budget_record_stage!(
    io, world_rank, comm, step, rk_stage, dt_stage, rk_a,
    blocks, scratch, stage_start, prediv, threads,
)
    # [1:4] fluxes, [5:14] base/start/pre/post state totals.
    local_values = zeros(Float64, 14)
    for (bid, b) in blocks
        s = scratch[bid]
        base = ct_energy_budget_total_sample!(
            s.base, b.Un, b.Vol, b.Nx, b.Ny, b.Nz, threads,
        )
        post = ct_energy_budget_state_sample!(
            s.state, b.U, b.Q, b.Vol, b.Nx, b.Ny, b.Nz, threads,
        )
        flux = ct_energy_budget_read_flux!(s.flux)
        start = stage_start[bid]
        before = prediv[bid]

        local_values[1] += flux[CT_ENERGY_FLUX_IDEAL]
        local_values[2] += flux[CT_ENERGY_FLUX_NONIDEAL_BEFORE]
        local_values[3] += flux[CT_ENERGY_FLUX_TOTAL]
        local_values[4] += flux[CT_ENERGY_FLUX_NONIDEAL_AFTER]
        local_values[5] += base[CT_ENERGY_STATE_TOTAL]
        local_values[6] += start[CT_ENERGY_STATE_TOTAL]
        local_values[7] += before[CT_ENERGY_STATE_TOTAL]
        local_values[8] += before[CT_ENERGY_STATE_KINETIC]
        local_values[9] += before[CT_ENERGY_STATE_MAGNETIC]
        local_values[10] += before[CT_ENERGY_STATE_INTERNAL]
        local_values[11] += post[CT_ENERGY_STATE_TOTAL]
        local_values[12] += post[CT_ENERGY_STATE_KINETIC]
        local_values[13] += post[CT_ENERGY_STATE_MAGNETIC]
        local_values[14] += post[CT_ENERGY_STATE_INTERNAL]
    end

    global_values = MPI.Allreduce(local_values, MPI.SUM, comm)
    if world_rank == 0
        ideal_flux_delta = global_values[1]
        total_flux_delta = global_values[3]
        nonideal_flux_delta = total_flux_delta - ideal_flux_delta
        forcing_energy_delta = global_values[7] - global_values[6]
        actual_dU5 = global_values[11] - global_values[7]
        base_energy_delta = global_values[11] - global_values[5]
        rk_carry_delta = rk_a * (global_values[7] - global_values[5])
        energy_flux_closure_residual =
            base_energy_delta - rk_carry_delta - total_flux_delta
        delta_kinetic = global_values[12] - global_values[8]
        delta_magnetic = global_values[13] - global_values[9]
        delta_internal = global_values[14] - global_values[10]
        internal_plus_magnetic_residual =
            actual_dU5 - delta_kinetic - delta_magnetic - delta_internal

        if io !== nothing
            @printf(
                io,
                "CT_ENERGY_BUDGET step=%d rk=%d dt=%.17e ideal_flux_delta=%.17e total_flux_delta=%.17e nonideal_flux_delta=%.17e actual_dU5=%.17e delta_magnetic_energy=%.17e delta_internal_energy=%.17e energy_flux_closure_residual=%.17e internal_plus_magnetic_residual=%.17e forcing_energy_delta=%.17e base_energy_delta=%.17e\n",
                step, rk_stage, dt_stage,
                ideal_flux_delta, total_flux_delta, nonideal_flux_delta,
                actual_dU5, delta_magnetic, delta_internal,
                energy_flux_closure_residual,
                internal_plus_magnetic_residual,
                forcing_energy_delta, base_energy_delta,
            )
            flush(io)
        end
        @printf(
            "CT_ENERGY_BUDGET step=%d rk=%d dt=%.17e ideal_flux_delta=%.17e total_flux_delta=%.17e nonideal_flux_delta=%.17e actual_dU5=%.17e delta_magnetic_energy=%.17e delta_internal_energy=%.17e energy_flux_closure_residual=%.17e internal_plus_magnetic_residual=%.17e\n",
            step, rk_stage, dt_stage,
            ideal_flux_delta, total_flux_delta, nonideal_flux_delta,
            actual_dU5, delta_magnetic, delta_internal,
            energy_flux_closure_residual,
            internal_plus_magnetic_residual,
        )
    end
    return global_values
end
