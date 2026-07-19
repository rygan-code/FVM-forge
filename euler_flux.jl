using StaticArrays

@inline function euler_primitive_to_conservative(
    state::SVector{5,T}, gamma::T,
) where {T<:AbstractFloat}
    rho, u, v, w, pressure = state
    energy = pressure / (gamma - one(T)) +
        T(0.5) * rho * (u*u + v*v + w*w)
    return SVector{5,T}(rho, rho*u, rho*v, rho*w, energy)
end

@inline function euler_rusanov_flux(
    left::SVector{5,T}, right::SVector{5,T},
    nx::T, ny::T, nz::T, gamma::T,
) where {T<:AbstractFloat}
    rho_left = max(left[1], T(1e-10))
    rho_right = max(right[1], T(1e-10))
    inv_rho_left = inv(rho_left)
    inv_rho_right = inv(rho_right)

    u_left = left[2] * inv_rho_left
    v_left = left[3] * inv_rho_left
    w_left = left[4] * inv_rho_left
    u_right = right[2] * inv_rho_right
    v_right = right[3] * inv_rho_right
    w_right = right[4] * inv_rho_right

    pressure_left = max(
        (gamma-one(T)) * (
            left[5] - T(0.5)*rho_left*(
                u_left*u_left + v_left*v_left + w_left*w_left
            )
        ),
        T(1e-10),
    )
    pressure_right = max(
        (gamma-one(T)) * (
            right[5] - T(0.5)*rho_right*(
                u_right*u_right + v_right*v_right + w_right*w_right
            )
        ),
        T(1e-10),
    )

    normal_velocity_left = u_left*nx + v_left*ny + w_left*nz
    normal_velocity_right = u_right*nx + v_right*ny + w_right*nz
    sound_speed_left = sqrt(gamma * pressure_left * inv_rho_left)
    sound_speed_right = sqrt(gamma * pressure_right * inv_rho_right)
    max_speed = max(
        abs(normal_velocity_left) + sound_speed_left,
        abs(normal_velocity_right) + sound_speed_right,
    )

    flux_left = SVector{5,T}(
        rho_left * normal_velocity_left,
        rho_left*u_left*normal_velocity_left + pressure_left*nx,
        rho_left*v_left*normal_velocity_left + pressure_left*ny,
        rho_left*w_left*normal_velocity_left + pressure_left*nz,
        (left[5] + pressure_left) * normal_velocity_left,
    )
    flux_right = SVector{5,T}(
        rho_right * normal_velocity_right,
        rho_right*u_right*normal_velocity_right + pressure_right*nx,
        rho_right*v_right*normal_velocity_right + pressure_right*ny,
        rho_right*w_right*normal_velocity_right + pressure_right*nz,
        (right[5] + pressure_right) * normal_velocity_right,
    )
    return T(0.5) * (flux_left + flux_right) -
        T(0.5) * max_speed * (right - left)
end
