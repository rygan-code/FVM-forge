using StaticArrays

if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end

@inline function mhd_primitive_to_conservative(primitive, gamma)
    density = primitive[1]
    velocity_x = primitive[2]
    velocity_y = primitive[3]
    velocity_z = primitive[4]
    pressure = primitive[5]
    magnetic_x = primitive[7]
    magnetic_y = primitive[8]
    magnetic_z = primitive[9]
    cleaning_potential = primitive[10]
    kinetic_energy = FT(0.5)*density*(
        velocity_x^2 + velocity_y^2 + velocity_z^2)
    magnetic_energy = mhd_magnetic_energy_density(
        magnetic_x, magnetic_y, magnetic_z)
    total_energy = mhd_energy_density_from_primitive(
        density, velocity_x, velocity_y, velocity_z, pressure,
        magnetic_x, magnetic_y, magnetic_z, gamma,
    )
    return SVector{9,FT}(
        density,
        density*velocity_x, density*velocity_y, density*velocity_z,
        total_energy,
        magnetic_x, magnetic_y, magnetic_z,
        cleaning_potential,
    )
end

@inline function mhd_conservative_to_primitive(conservative, gamma, gas_constant)
    density = max(conservative[1], density_floor)
    inverse_density = inv(density)
    velocity_x = conservative[2]*inverse_density
    velocity_y = conservative[3]*inverse_density
    velocity_z = conservative[4]*inverse_density
    magnetic_x = conservative[6]
    magnetic_y = conservative[7]
    magnetic_z = conservative[8]
    cleaning_potential = conservative[9]
    kinetic_energy = FT(0.5)*density*(
        velocity_x^2 + velocity_y^2 + velocity_z^2)
    magnetic_energy = mhd_magnetic_energy_density(
        magnetic_x, magnetic_y, magnetic_z)
    internal_energy = conservative[5] - kinetic_energy - magnetic_energy
    pressure = isothermal_mhd ?
        density * gas_constant * isothermal_temperature :
        max((gamma - one(FT))*internal_energy, pressure_floor)
    temperature = isothermal_mhd ? isothermal_temperature :
        pressure/(density*gas_constant)
    return SVector{10,FT}(
        density, velocity_x, velocity_y, velocity_z, pressure, temperature,
        magnetic_x, magnetic_y, magnetic_z, cleaning_potential,
    )
end

@inline function mhd_fast_speed_from_primitive(primitive, gamma)
    density = max(primitive[1], eps(FT))
    pressure = max(primitive[5], eps(FT))
    magnetic_squared = primitive[7]^2 + primitive[8]^2 + primitive[9]^2
    return sqrt(max(
        mhd_sound_speed_squared(density, pressure, gamma) +
        magnetic_squared*INV_MU0_SI/density,
        zero(FT),
    ))
end

@inline function mhd_glm_flux_normal(conservative, nx, ny, nz, gamma, ch)
    density = max(conservative[1], eps(FT))
    inverse_density = inv(density)
    velocity_x = conservative[2]*inverse_density
    velocity_y = conservative[3]*inverse_density
    velocity_z = conservative[4]*inverse_density
    magnetic_x = conservative[6]
    magnetic_y = conservative[7]
    magnetic_z = conservative[8]
    cleaning_potential = conservative[9]
    kinetic_energy = FT(0.5)*density*(
        velocity_x^2 + velocity_y^2 + velocity_z^2)
    magnetic_squared = magnetic_x^2 + magnetic_y^2 + magnetic_z^2
    pressure = isothermal_mhd ?
        density * Rg * isothermal_temperature :
        max((gamma - one(FT))*(
            conservative[5] - kinetic_energy -
            magnetic_squared*(INV_MU0_SI / 2)), pressure_floor)
    total_pressure = pressure + magnetic_squared*(INV_MU0_SI / 2)
    normal_velocity = velocity_x*nx + velocity_y*ny + velocity_z*nz
    normal_magnetic = magnetic_x*nx + magnetic_y*ny + magnetic_z*nz
    velocity_dot_magnetic = velocity_x*magnetic_x +
        velocity_y*magnetic_y + velocity_z*magnetic_z
    return SVector{9,FT}(
        density*normal_velocity,
        density*velocity_x*normal_velocity + total_pressure*nx -
            normal_magnetic*magnetic_x*INV_MU0_SI,
        density*velocity_y*normal_velocity + total_pressure*ny -
            normal_magnetic*magnetic_y*INV_MU0_SI,
        density*velocity_z*normal_velocity + total_pressure*nz -
            normal_magnetic*magnetic_z*INV_MU0_SI,
        isothermal_mhd ? zero(FT) :
            (conservative[5] + total_pressure)*normal_velocity -
            (normal_magnetic*velocity_dot_magnetic)*INV_MU0_SI,
        magnetic_x*normal_velocity - velocity_x*normal_magnetic + cleaning_potential*nx,
        magnetic_y*normal_velocity - velocity_y*normal_magnetic + cleaning_potential*ny,
        magnetic_z*normal_velocity - velocity_z*normal_magnetic + cleaning_potential*nz,
        ch*ch*normal_magnetic,
    )
end

@inline function mhd_glm_rusanov_flux(left, right, nx, ny, nz, gamma, ch)
    left_primitive = mhd_conservative_to_primitive(left, gamma, Rg)
    right_primitive = mhd_conservative_to_primitive(right, gamma, Rg)
    left_normal_velocity = left_primitive[2]*nx + left_primitive[3]*ny + left_primitive[4]*nz
    right_normal_velocity = right_primitive[2]*nx + right_primitive[3]*ny + right_primitive[4]*nz
    maximum_speed = max(
        abs(left_normal_velocity) + mhd_fast_speed_from_primitive(left_primitive, gamma),
        abs(right_normal_velocity) + mhd_fast_speed_from_primitive(right_primitive, gamma),
        ch,
    )
    left_flux = mhd_glm_flux_normal(left, nx, ny, nz, gamma, ch)
    right_flux = mhd_glm_flux_normal(right, nx, ny, nz, gamma, ch)
    state_jump = isothermal_mhd ?
        SVector{9,FT}(
            right[1] - left[1], right[2] - left[2],
            right[3] - left[3], right[4] - left[4], zero(FT),
            right[6] - left[6], right[7] - left[7],
            right[8] - left[8], right[9] - left[9],
        ) : right - left
    return FT(0.5)*(left_flux + right_flux) - FT(0.5)*maximum_speed*state_jump
end
