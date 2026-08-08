if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end

function div_update_kernel!(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    @inbounds Jact::FT = J[i+NG, j+NG, k+NG] * dt

    if viscous || (equation_type == :MHD && resistive)
        for n = 1:Nhydro
            @inbounds U[i+NG, j+NG, k+NG, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n]) -
                (Fv_x[i, j, k, n] - Fv_x[i+1, j, k, n]) - 
                (Fv_y[i, j, k, n] - Fv_y[i, j+1, k, n]) - 
                (Fv_z[i, j, k, n] - Fv_z[i, j, k+1, n])
            ) * Jact
        end
    else
        for n = 1:Nhydro
            @inbounds U[i+NG, j+NG, k+NG, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n])
            ) * Jact
        end
    end
    return
end

function div_LTS(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    @inbounds Jact::FT = J[i+NG, j+NG, k+NG] * dt[i+NG, j+NG, k+NG]

    if viscous || (equation_type == :MHD && resistive)
        for n = 1:Nhydro
            @inbounds U[i+NG, j+NG, k+NG, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n]) -
                (Fv_x[i, j, k, n] - Fv_x[i+1, j, k, n]) - 
                (Fv_y[i, j, k, n] - Fv_y[i, j+1, k, n]) - 
                (Fv_z[i, j, k, n] - Fv_z[i, j, k+1, n])
            ) * Jact
        end
    else
        for n = 1:Nhydro
            @inbounds U[i+NG, j+NG, k+NG, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n])
            ) * Jact
        end
    end
    return
end

# ─── Fused div + RK combination + clipping + c2Prim ───
# Combines 2 separate kernels into 1 pass:
#   1. div:           U += dt/Vol * (∇·F_inv - ∇·F_vis)
#   2. non-CT: recover Q and apply the positivity policy in this kernel
#      CT: defer recovery until the face-B update has reached the same RK stage
# Saves: 1 kernel launch + 1 full read-write of U array through HBM on non-CT paths
function _structured_flux_rk_update!(
    U, Un, Q, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J, rk_a,
    i, j, k, ii, jj, kk,
)
    tangential_offset = STRUCTURED_FLUX_TANGENTIAL_HALO
    @inbounds volume_scaled_dt = J[ii,jj,kk]*dt
    @inbounds for variable in 1:Nhydro
        inviscid =
            Fx[i,j+tangential_offset,k+tangential_offset,variable] -
            Fx[i+1,j+tangential_offset,k+tangential_offset,variable] +
            Fy[i+tangential_offset,j,k+tangential_offset,variable] -
            Fy[i+tangential_offset,j+1,k+tangential_offset,variable] +
            Fz[i+tangential_offset,j+tangential_offset,k,variable] -
            Fz[i+tangential_offset,j+tangential_offset,k+1,variable]
        @static if viscous || (equation_type == :MHD && resistive)
            viscous_flux =
                Fv_x[i,j+tangential_offset,k+tangential_offset,variable] -
                Fv_x[i+1,j+tangential_offset,k+tangential_offset,variable] +
                Fv_y[i+tangential_offset,j,k+tangential_offset,variable] -
                Fv_y[i+tangential_offset,j+1,k+tangential_offset,variable] +
                Fv_z[i+tangential_offset,j+tangential_offset,k,variable] -
                Fv_z[i+tangential_offset,j+tangential_offset,k+1,variable]
        else
            viscous_flux = zero(FT)
        end
        updated = U[ii,jj,kk,variable] +
            (inviscid - viscous_flux)*volume_scaled_dt
        U[ii,jj,kk,variable] = Un[ii,jj,kk,variable] +
            rk_a*(updated - Un[ii,jj,kk,variable])
    end
    @static if equation_type == :MHD && ct_mode && isothermal_mhd
        # CT updates face-B after this hydro update. Repair the hydro state
        # now so the next reconstruction never sees a Q-only density floor.
        ct_repair_isothermal_hydro_state!(U, Q, ii, jj, kk)
    end
    return
end

function _structured_raw_thermodynamics(U, Q, ii, jj, kk)
    @inbounds density = U[ii,jj,kk,1]
    valid_density = isfinite(density) && density > zero(FT)
    inverse_density = valid_density ? inv(density) : zero(FT)
    @inbounds velocity = SVector{3,FT}(
        U[ii,jj,kk,2]*inverse_density,
        U[ii,jj,kk,3]*inverse_density,
        U[ii,jj,kk,4]*inverse_density,
    )
    kinetic = valid_density ? FT(0.5)*density*sum(abs2,velocity) : FT(NaN)
    @static if equation_type == :MHD
        magnetic = ct_cell_magnetic(U, Q, ii, jj, kk)
        magnetic_energy = FT(0.5)*INV_MU0_SI*sum(abs2,magnetic)
    else
        magnetic = SVector{3,FT}(zero(FT),zero(FT),zero(FT))
        magnetic_energy = zero(FT)
    end
    @static if isothermal_mhd
        @inbounds pressure = density * Rg * isothermal_temperature
        temperature = isothermal_temperature
    else
        @inbounds pressure = (γ-one(FT))*(
            U[ii,jj,kk,5] - kinetic - magnetic_energy)
        temperature = pressure/(density*Rg)
    end
    return density, velocity, pressure, temperature, magnetic, magnetic_energy
end

function _store_structured_primitive!(Q, U, ii, jj, kk, state)
    density, velocity, pressure, temperature, magnetic = state
    @inbounds begin
        Q[ii,jj,kk,1] = density
        Q[ii,jj,kk,2] = velocity[1]
        Q[ii,jj,kk,3] = velocity[2]
        Q[ii,jj,kk,4] = velocity[3]
        Q[ii,jj,kk,5] = pressure
        Q[ii,jj,kk,6] = temperature
    end
    @static if equation_type == :MHD
        @inbounds begin
            Q[ii,jj,kk,7] = magnetic[1]
            Q[ii,jj,kk,8] = magnetic[2]
            Q[ii,jj,kk,9] = magnetic[3]
            @static if ct_mode
                Q[ii,jj,kk,QPSI] = zero(FT)
            else
                Q[ii,jj,kk,QPSI] = U[ii,jj,kk,UPSI]
            end
        end
    end
    return
end

function _repair_structured_thermodynamics!(
    U, Q, J, repair_count, conservation_delta, ii, jj, kk,
    density, velocity, pressure, magnetic, magnetic_energy,
)
    @inbounds old_state = SVector{5,FT}(
        U[ii,jj,kk,1], U[ii,jj,kk,2], U[ii,jj,kk,3],
        U[ii,jj,kk,4], U[ii,jj,kk,5])
    density = positivity_floor(density,density_floor)
    pressure = positivity_floor(pressure,pressure_floor)
    @static if isothermal_mhd
        temperature = isothermal_temperature
        thermal_energy = density * Rg * isothermal_temperature
    else
        temperature = pressure/(density*Rg)
        thermal_energy = pressure/(γ-one(FT))
    end
    kinetic = FT(0.5)*density*sum(abs2,velocity)
    new_state = SVector{5,FT}(
        density, density*velocity[1], density*velocity[2], density*velocity[3],
        thermal_energy + kinetic + magnetic_energy,
    )
    @inbounds for variable in 1:5
        U[ii,jj,kk,variable] = new_state[variable]
        gpu_atomic_add!(conservation_delta,variable,
            (new_state[variable]-old_state[variable])/J[ii,jj,kk])
    end
    gpu_atomic_add!(repair_count,1,Int32(1))
    _store_structured_primitive!(Q,U,ii,jj,kk,
        (density,velocity,pressure,temperature,magnetic))
    return
end

@inline function _finalize_structured_thermodynamics!(
    U, Q, J, repair_count, conservation_delta, ii, jj, kk,
)
    density,velocity,pressure,temperature,magnetic,magnetic_energy =
        _structured_raw_thermodynamics(U,Q,ii,jj,kk)
    invalid = positivity_is_invalid(density,pressure)
    @static if strict_ct_positivity && ct_mode && splitMethodID == 4
        invalid = false
    end
    if invalid
        if POSITIVITY_STRICT
            gpu_atomic_add!(repair_count,1,Int32(1))
        else
            _repair_structured_thermodynamics!(
                U,Q,J,repair_count,conservation_delta,ii,jj,kk,density,
                velocity,pressure,magnetic,magnetic_energy)
            return
        end
    end
    _store_structured_primitive!(Q,U,ii,jj,kk,
        (density,velocity,pressure,temperature,magnetic))
    return
end

function finalize_structured_primitive(
    U, Q, J, repair_count, conservation_delta, nxp, nyp, nzp,
)
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    (i > nxp || j > nyp || k > nzp) && return
    ii, jj, kk = i+NG, j+NG, k+NG
    _finalize_structured_thermodynamics!(
        U,Q,J,repair_count,conservation_delta,ii,jj,kk)
    return
end

function div_rk_clip_prim(
    U, Un, Q, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, repair_count,
    conservation_delta, dt, J, rk_a::FT, nxp, nyp, nzp,
)
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    (i > nxp || j > nyp || k > nzp) && return
    ii, jj, kk = i+NG, j+NG, k+NG
    _structured_flux_rk_update!(
        U,Un,Q,Fx,Fy,Fz,Fv_x,Fv_y,Fv_z,dt,J,rk_a,i,j,k,ii,jj,kk)
    @static if !(equation_type == :MHD && ct_mode)
        _finalize_structured_thermodynamics!(
            U,Q,J,repair_count,conservation_delta,ii,jj,kk)
    end
    return
end

# ─── Implicit RHS kernel ───
# Computes RHS = -(divF - divFv) * (Vol/dt) + source
# Output goes to dU_rhs buffer (NOT in-place update to U)
# Used by LU-SGS implicit time stepping
function div_to_rhs(dU_rhs, U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z,
                    dU_forced, dt, J, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    # Vol/dt factor (Vol = 1/J is the cell volume, but here J = 1/Vol stored)
    # In the existing code: Jact = J[i+NG, j+NG, k+NG] * dt
    # For the implicit RHS we want: RHS = flux_divergence * (Vol * dt) + source * dt
    # But actually the LU-SGS equation is: (Vol/dt + ...) * ΔU = -R(U^n)
    # where R(U^n) = -flux_divergence * Vol - source * Vol
    # So RHS = (-R) = flux_divergence * Vol + source * Vol
    # We store: dU_rhs = flux_div * Vol + source_term
    # (Note: dt is NOT included here; the diagonal D = Vol/dt handles the scaling)

    @inbounds vol_inv::FT = J[i+NG, j+NG, k+NG]  # J = 1/Vol

    if viscous || (equation_type == :MHD && resistive)
        for n = 1:Nhydro
            @inbounds flux_div = (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) +
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) +
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n]) -
                (Fv_x[i, j, k, n] - Fv_x[i+1, j, k, n]) -
                (Fv_y[i, j, k, n] - Fv_y[i, j+1, k, n]) -
                (Fv_z[i, j, k, n] - Fv_z[i, j, k+1, n])
            )
            # source term (from volume forces, intensive = per unit volume)
            @inbounds src = flow_forcing || test_case == "HIT" ? dU_forced[i, j, k, n] : zero(FT)
            # LU-SGS equation: (V/dt + σ/2) × ΔU = RHS
            # flux_div is EXTENSIVE (already × face area, units [N] for momentum)
            # src is INTENSIVE (force per unit volume, units [N/m³] for momentum)
            # To match dimensions: RHS = flux_div + src × V = flux_div + src / vol_inv
            # Explicit does: ΔU = flux_div×dt/V + src×dt
            # Implicit at σ=0: ΔU = (flux_div + src/vol_inv) × dt/V
            #                     = flux_div×dt/V + src×dt  ✓
            @inbounds dU_rhs[i, j, k, n] = flux_div + src / (vol_inv + FT(1.0e-30))
        end
    else
        for n = 1:Nhydro
            @inbounds flux_div = (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) +
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) +
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n])
            )
            @inbounds src = flow_forcing || test_case == "HIT" ? dU_forced[i, j, k, n] : zero(FT)
            @inbounds dU_rhs[i, j, k, n] = flux_div + src / (vol_inv + FT(1.0e-30))
        end
    end
    return
end
