# Conservative residual, RK3 update, state conversion, and CFL kernels.

if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end

if !@isdefined(POSITIVITY_POLICY_LOADED)
    include(joinpath(@__DIR__, "..", "core", "positivity_policy.jl"))
end

if !@isdefined(viscous)
    const viscous = false
end

function unstruct_residual_kernel!(
    residual, F_faces, Fv_faces,
    cell_face_offset, cell_face_list, cell_face_sign,
    cell_vol, ncell::Int,
)
    cell = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    cell > ncell && return
    @inbounds begin
        inverse_volume = inv(cell_vol[cell])
        first_face = cell_face_offset[cell]
        last_face = cell_face_offset[cell + 1] - Int32(1)
        for variable in 1:Ncons
            outward_flux = zero(FT)
            for entry in first_face:last_face
                face = cell_face_list[entry]
                outward_flux += cell_face_sign[entry]*(
                    F_faces[face,variable] - Fv_faces[face,variable])
            end
            residual[cell,variable] = -outward_flux*inverse_volume
        end
    end
    return
end

function _unstruct_rk_flux_update!(
    U, Un, F_faces, Fv_faces, cell_face_offset, cell_face_list,
    cell_face_sign, cell_vol, cell, time_step, rk_coefficient,
)
    @inbounds inverse_volume_time = time_step/cell_vol[cell]
    @inbounds first_face = cell_face_offset[cell]
    @inbounds last_face = cell_face_offset[cell + 1] - Int32(1)
    @inbounds for variable in 1:Ncons
        @static if equation_type == :MHD && ct_mode
            variable >= 6 && continue
        end
        outward_flux = zero(FT)
        for entry in first_face:last_face
            face = cell_face_list[entry]
            outward_flux += cell_face_sign[entry]*(
                F_faces[face,variable] - Fv_faces[face,variable])
        end
        updated = U[cell,variable] - outward_flux*inverse_volume_time
        U[cell,variable] = Un[cell,variable] +
            rk_coefficient*(updated - Un[cell,variable])
    end
    return
end

function _unstruct_raw_thermodynamics(U, cell)
    @inbounds density = U[cell,1]
    valid_density = isfinite(density) && density > zero(FT)
    inverse_density = valid_density ? inv(density) : zero(FT)
    @inbounds velocity_x = U[cell,2]*inverse_density
    @inbounds velocity_y = U[cell,3]*inverse_density
    @inbounds velocity_z = U[cell,4]*inverse_density
    kinetic = valid_density ? FT(0.5)*density*(
        velocity_x^2 + velocity_y^2 + velocity_z^2) : FT(NaN)
    @static if equation_type == :MHD
        @inbounds magnetic = FT(0.5)*INV_MU0_SI*(
            U[cell,6]^2 + U[cell,7]^2 + U[cell,8]^2)
    else
        magnetic = zero(FT)
    end
    @inbounds pressure = (γ - one(FT))*(U[cell,5] - kinetic - magnetic)
    temperature = pressure/(density*Rg)
    return density, velocity_x, velocity_y, velocity_z, pressure, temperature, magnetic
end

function _store_unstruct_primitive!(Q, U, cell, state)
    density, velocity_x, velocity_y, velocity_z, pressure, temperature = state
    @inbounds begin
        Q[cell,1] = density
        Q[cell,2] = velocity_x
        Q[cell,3] = velocity_y
        Q[cell,4] = velocity_z
        Q[cell,5] = pressure
        Q[cell,6] = temperature
    end
    @static if equation_type == :MHD
        @inbounds begin
            Q[cell,7] = U[cell,6]
            Q[cell,8] = U[cell,7]
            Q[cell,9] = U[cell,8]
            Q[cell,10] = U[cell,9]
        end
    end
    return
end

function _repair_unstruct_thermodynamics!(
    U, Q, cell_vol, repair_count, conservation_delta, cell, raw_state, magnetic,
)
    density, velocity_x, velocity_y, velocity_z, pressure, _ = raw_state
    old_state = SVector{5,FT}(
        U[cell,1], U[cell,2], U[cell,3], U[cell,4], U[cell,5],
    )
    density = positivity_floor(density, density_floor)
    pressure = positivity_floor(pressure, pressure_floor)
    temperature = pressure/(density*Rg)
    kinetic = FT(0.5)*density*(velocity_x^2 + velocity_y^2 + velocity_z^2)
    new_state = SVector{5,FT}(
        density, density*velocity_x, density*velocity_y, density*velocity_z,
        pressure/(γ - one(FT)) + kinetic + magnetic,
    )
    @inbounds for variable in 1:5
        U[cell,variable] = new_state[variable]
        gpu_atomic_add!(conservation_delta, variable,
            (new_state[variable] - old_state[variable])*cell_vol[cell])
    end
    gpu_atomic_add!(repair_count, 1, Int32(1))
    _store_unstruct_primitive!(Q, U, cell, (
        density, velocity_x, velocity_y, velocity_z, pressure, temperature,
    ))
    return
end

function unstruct_div_rk_clip_prim_kernel!(
    U, Un, Q, F_faces, Fv_faces, cell_face_offset, cell_face_list,
    cell_face_sign, cell_vol, repair_count, conservation_delta,
    time_step::FT, rk_coefficient::FT, ncell::Int,
)
    cell = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    cell > ncell && return
    _unstruct_rk_flux_update!(
        U, Un, F_faces, Fv_faces, cell_face_offset, cell_face_list,
        cell_face_sign, cell_vol, cell, time_step, rk_coefficient,
    )
    raw = _unstruct_raw_thermodynamics(U, cell)
    density, velocity_x, velocity_y, velocity_z, pressure, temperature, magnetic = raw
    if positivity_is_invalid(density, pressure)
        if POSITIVITY_STRICT
            gpu_atomic_add!(repair_count, 1, Int32(1))
        else
            _repair_unstruct_thermodynamics!(
                U, Q, cell_vol, repair_count, conservation_delta, cell,
                (density, velocity_x, velocity_y, velocity_z, pressure, temperature),
                magnetic,
            )
            return
        end
    end
    _store_unstruct_primitive!(Q, U, cell, (
        density, velocity_x, velocity_y, velocity_z, pressure, temperature,
    ))
    return
end

function unstruct_prim2c_kernel!(U, Q, ncell_tot::Int)
    cell = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    cell > ncell_tot && return
    @static if equation_type == :MHD
        @inbounds primitive = SVector{10,FT}(
            Q[cell,1], Q[cell,2], Q[cell,3], Q[cell,4], Q[cell,5],
            Q[cell,6], Q[cell,7], Q[cell,8], Q[cell,9], Q[cell,10],
        )
        conservative = mhd_primitive_to_conservative(primitive, FT(γ))
        @inbounds for variable in 1:Ncons
            U[cell,variable] = conservative[variable]
        end
    else
        @inbounds begin
            density = Q[cell,1]
            velocity_x = Q[cell,2]
            velocity_y = Q[cell,3]
            velocity_z = Q[cell,4]
            pressure = Q[cell,5]
            U[cell,1] = density
            U[cell,2] = density*velocity_x
            U[cell,3] = density*velocity_y
            U[cell,4] = density*velocity_z
            U[cell,5] = pressure/(γ - one(FT)) + FT(0.5)*density*(
                velocity_x^2 + velocity_y^2 + velocity_z^2)
        end
    end
    return
end

function unstruct_c2prim_kernel!(Q, U, ncell_tot::Int)
    cell = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    cell > ncell_tot && return
    @static if equation_type == :MHD
        @inbounds conservative = SVector{9,FT}(
            U[cell,1], U[cell,2], U[cell,3], U[cell,4], U[cell,5],
            U[cell,6], U[cell,7], U[cell,8], U[cell,9],
        )
        primitive = mhd_conservative_to_primitive(conservative, FT(γ), FT(Rg))
        @inbounds for variable in 1:Nprim
            Q[cell,variable] = primitive[variable]
        end
    else
        @inbounds density = max(U[cell,1], FT(1.0e-5))
        inverse_density = inv(density)
        @inbounds velocity_x = U[cell,2]*inverse_density
        @inbounds velocity_y = U[cell,3]*inverse_density
        @inbounds velocity_z = U[cell,4]*inverse_density
        @inbounds internal_energy = max(
            U[cell,5] - FT(0.5)*density*(velocity_x^2 + velocity_y^2 + velocity_z^2),
            FT(1.0e-5)/(γ - one(FT)),
        )
        pressure = (γ - one(FT))*internal_energy
        @inbounds begin
            Q[cell,1] = density
            Q[cell,2] = velocity_x
            Q[cell,3] = velocity_y
            Q[cell,4] = velocity_z
            Q[cell,5] = pressure
            Q[cell,6] = pressure/(density*Rg)
        end
    end
    return
end

function unstruct_compute_dt_kernel!(
    dt_arr, Q, cell_vol, cell_cx, cell_cy, cell_cz,
    face_area, face_nx, face_ny, face_nz, face_L, face_R,
    cell_face_offset, cell_face_list,
    ncell::Int, courant::FT, ch_glm::FT,
)
    cell = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    cell > ncell && return
    @inbounds density = max(Q[cell,1], eps(FT))
    @inbounds pressure = max(Q[cell,5], eps(FT))
    @inbounds velocity_x = Q[cell,2]
    @inbounds velocity_y = Q[cell,3]
    @inbounds velocity_z = Q[cell,4]
    @static if equation_type == :MHD
        @inbounds magnetic_squared = Q[cell,7]^2 + Q[cell,8]^2 + Q[cell,9]^2
        wave_speed = sqrt(max(
            γ*pressure/density + magnetic_squared*INV_MU0_SI/density,
            zero(FT),
        ))
    else
        wave_speed = sqrt(γ*pressure/density)
    end

    convective_radius = zero(FT)
    diffusive_radius = zero(FT)
    effective_diffusivity = zero(FT)
    @static if viscous
        @inbounds viscosity = unstruct_viscosity(Q[cell,6])
        effective_diffusivity = max(effective_diffusivity,viscosity/(density*Pr))
    end
    @static if equation_type == :MHD && resistive
        effective_diffusivity = max(effective_diffusivity,FT(η_mhd))
    end
    @inbounds first_face = cell_face_offset[cell]
    @inbounds last_face = cell_face_offset[cell + 1] - Int32(1)
    @inbounds for entry in first_face:last_face
        face = cell_face_list[entry]
        normal_velocity = abs(
            velocity_x*face_nx[face] + velocity_y*face_ny[face] + velocity_z*face_nz[face])
        @static if equation_type == :MHD
            convective_radius += max(normal_velocity + wave_speed, ch_glm)*face_area[face]
        else
            convective_radius += (normal_velocity + wave_speed)*face_area[face]
        end
        @static if viscous || (equation_type == :MHD && resistive)
            other = face_L[face] == cell ? face_R[face] : face_L[face]
            dx = cell_cx[other] - cell_cx[cell]
            dy = cell_cy[other] - cell_cy[cell]
            dz = cell_cz[other] - cell_cz[cell]
            distance = sqrt(dx*dx + dy*dy + dz*dz)
            diffusive_radius += FT(2)*effective_diffusivity*face_area[face]/max(distance, eps(FT))
        end
    end
    @inbounds dt_arr[cell] = courant*cell_vol[cell]/
        max(convective_radius + diffusive_radius, eps(FT))
    return
end

function unstruct_compute_ch_kernel!(speed, Q, ncell::Int)
    cell = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    cell > ncell && return
    @static if equation_type == :MHD
        @inbounds density = max(Q[cell,1], eps(FT))
        @inbounds pressure = max(Q[cell,5], eps(FT))
        @inbounds magnetic_squared = Q[cell,7]^2 + Q[cell,8]^2 + Q[cell,9]^2
        @inbounds speed[cell] = sqrt(max(
            γ*pressure/density + magnetic_squared*INV_MU0_SI/density,
            zero(FT),
        ))
    else
        @inbounds speed[cell] = zero(FT)
    end
    return
end

function unstruct_glm_damping_kernel!(U, Q, time_step, ch, cr, ncell::Int)
    cell = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    cell > ncell && return
    decay = exp(-time_step*ch/(cr + FT(1.0e-20)))
    @inbounds U[cell,9] *= decay
    @inbounds Q[cell,10] = U[cell,9]
    return
end

function unstruct_residual!(residual, block::UnstructBlock)
    size(residual,1) >= block.ncell || throw(DimensionMismatch(
        "unstructured residual has too few cells"))
    size(residual,2) == Ncons || throw(DimensionMismatch(
        "unstructured residual must have Ncons=$Ncons columns"))
    blocks = cld(block.ncell, block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=blocks unstruct_residual_kernel!(
        residual, block.F_faces, block.Fv_faces,
        block.cell_face_offset, block.cell_face_list, block.cell_face_sign,
        block.cell_vol, block.ncell,
    )
    return residual
end

function unstruct_div_rk_clip_prim!(block::UnstructBlock, time_step::FT,
                                    rk_coefficient::FT;
                                    step::Int=0, rk_stage::Int=0)
    blocks = cld(block.ncell, block.nthreads_cell)
    repair_count = gpu_zeros(Int32, 1)
    conservation_delta = gpu_zeros(FT, 5)
    reset_positivity_diagnostics!(repair_count, conservation_delta)
    @gpu_launch threads=block.nthreads_cell blocks=blocks unstruct_div_rk_clip_prim_kernel!(
        block.U, block.Un, block.Q, block.F_faces, block.Fv_faces,
        block.cell_face_offset, block.cell_face_list, block.cell_face_sign,
        block.cell_vol, repair_count, conservation_delta,
        time_step, rk_coefficient, block.ncell,
    )
    gpu_sync()
    return positivity_report_or_throw!(repair_count, conservation_delta;
        context="unstructured block=$(block.id) step=$step RK=$rk_stage")
end

function unstruct_prim2c!(block::UnstructBlock)
    blocks = cld(block.ncell_tot, block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=blocks unstruct_prim2c_kernel!(
        block.U, block.Q, block.ncell_tot)
    return
end

function unstruct_c2prim!(block::UnstructBlock)
    blocks = cld(block.ncell_tot, block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=blocks unstruct_c2prim_kernel!(
        block.Q, block.U, block.ncell_tot)
    return
end

function unstruct_compute_dt!(block::UnstructBlock, courant::FT, ch_glm::FT=zero(FT))
    blocks = cld(block.ncell, block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=blocks unstruct_compute_dt_kernel!(
        block.dt_arr, block.Q,
        block.cell_vol, block.cell_cx, block.cell_cy, block.cell_cz,
        block.face_area, block.face_nx, block.face_ny, block.face_nz,
        block.face_L, block.face_R,
        block.cell_face_offset, block.cell_face_list,
        block.ncell, courant, ch_glm,
    )
    return minimum(Array(block.dt_arr))
end

function unstruct_compute_ch!(block::UnstructBlock)
    @static if equation_type == :MHD
        blocks = cld(block.ncell, block.nthreads_cell)
        @gpu_launch threads=block.nthreads_cell blocks=blocks unstruct_compute_ch_kernel!(
            block.dt_arr, block.Q, block.ncell)
        return maximum(Array(block.dt_arr))
    else
        return zero(FT)
    end
end

function unstruct_apply_glm_damping!(block::UnstructBlock, time_step::FT, ch::FT)
    @static if equation_type == :MHD
        blocks = cld(block.ncell, block.nthreads_cell)
        @gpu_launch threads=block.nthreads_cell blocks=blocks unstruct_glm_damping_kernel!(
            block.U, block.Q, time_step, ch, cr_glm, block.ncell)
    end
    return
end
