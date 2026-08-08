# ============================================================================
# Prescribed external magnetic-field initialization
#
# V1 semantics: the field is supplied as a divergence-free vector potential
# during CT initialization, then released to the self-consistent CT-MHD
# evolution. This is an initial background field, not a continuously powered
# coil model.
# ============================================================================

if !isdefined(@__MODULE__, :structured_cell_center_coordinates)
    include(joinpath(@__DIR__, "..", "mesh", "structured_coordinates.jl"))
end

struct AxisymmetricMagneticNozzleField{T<:AbstractFloat}
    B0::T
    x0::T
    length::T
    expansion_ratio::T
end

# Exact vacuum-field model used by Sheth et al. (2025). The solver axis is
# x; y-z form the cylindrical cross-section. Keeping this separate preserves
# the existing finite-solenoid and tanh models.
struct BesselMagneticNozzleField{T<:AbstractFloat}
    B0::T
    x0::T
    length::T
    alpha::T
end

# Finite-length solenoid model used by the Lorzel-Mikellides benchmark.
# The field is normalized to B0 at the coil center, while the geometry is
# retained in SI length units. The circular Biot-Savart integral is evaluated
# in a local meridional plane so rotational covariance is exact on every mesh.
const FINITE_SOLENOID_AZIMUTHAL_SEGMENTS = Int32(128)

struct FiniteSolenoidMagneticField{T<:AbstractFloat}
    B0::T
    x_center::T
    coil_radius::T
    coil_length::T
    turns::Int32
end

function FiniteSolenoidMagneticField(
    B0::T, x_center::T, coil_radius::T, coil_length::T, turns::Integer,
) where {T<:AbstractFloat}
    isfinite(B0) || throw(ArgumentError("B0 must be finite"))
    isfinite(x_center) || throw(ArgumentError("x_center must be finite"))
    isfinite(coil_radius) && coil_radius > zero(T) || throw(ArgumentError(
        "coil radius must be finite and positive",
    ))
    isfinite(coil_length) && coil_length > zero(T) || throw(ArgumentError(
        "coil length must be finite and positive",
    ))
    turns > 0 || throw(ArgumentError("turns must be positive"))
    return FiniteSolenoidMagneticField{T}(
        B0, x_center, coil_radius, coil_length, Int32(turns),
    )
end

function FiniteSolenoidMagneticField(
    B0::Real, x_center::Real, coil_radius::Real, coil_length::Real,
    turns::Integer,
)
    values = promote(
        float(B0), float(x_center), float(coil_radius), float(coil_length),
    )
    return FiniteSolenoidMagneticField(values..., turns)
end

@inline function _finite_solenoid_center_field_per_current(
    coil_radius::T, coil_length::T, turns::Int32,
) where {T}
    value = zero(T)
    half_length = coil_length / 2
    for turn in Int32(0):(turns - Int32(1))
        x_turn = -half_length + (T(turn) + T(0.5)) * coil_length / T(turns)
        value += coil_radius^2 /
            (T(2) * (coil_radius^2 + x_turn^2)^(T(1.5)))
    end
    return value
end

@inline function finite_solenoid_external_field_components(
    B0::T, x_center::T, coil_radius::T, coil_length::T, turns::Int32,
    x, y, z,
) where {T}
    current_scale = B0 / _finite_solenoid_center_field_per_current(
        coil_radius, coil_length, turns,
    )
    nseg = FINITE_SOLENOID_AZIMUTHAL_SEGMENTS
    dphi = T(2) * T(pi) / T(nseg)
    value_type = promote_type(T, typeof(x), typeof(y), typeof(z))
    bx = zero(value_type)
    br = zero(value_type)
    radius = sqrt(y*y + z*z)
    half_length = coil_length / 2
    for turn in Int32(0):(turns - Int32(1))
        x_turn = x_center - half_length +
            (T(turn) + T(0.5)) * coil_length / T(turns)
        for segment in Int32(0):(nseg - Int32(1))
            phi = (T(segment) + T(0.5)) * dphi
            source_y = coil_radius * cos(phi)
            source_z = coil_radius * sin(phi)
            dx = x - x_turn
            dy = radius - source_y
            dz = -source_z
            r2 = dx*dx + dy*dy + dz*dz + T(1.0e-18)
            inv_r3 = inv(r2 * sqrt(r2))
            dl_y = -coil_radius * sin(phi) * dphi
            dl_z = coil_radius * cos(phi) * dphi
            bx += (dl_y*dz - dl_z*dy) * inv_r3
            br += dl_z * dx * inv_r3
        end
    end
    coefficient = current_scale / (T(4) * T(pi))
    radial_scale = radius > zero(radius) ? coefficient*br/radius : zero(value_type)
    return coefficient*bx, radial_scale*y, radial_scale*z
end

@inline function modified_bessel_i0_small(x)
    q = x*x / typeof(x)(4)
    term = one(x)
    value = term
    for m in 1:12
        mm = typeof(x)(m)
        term *= q / (mm*mm)
        value += term
    end
    return value
end

@inline function modified_bessel_i1_small(x)
    q = x*x / typeof(x)(4)
    term = x / typeof(x)(2)
    value = term
    for m in 1:12
        mm = typeof(x)(m)
        term *= q / (mm*(mm + one(x)))
        value += term
    end
    return value
end

@inline function bessel_magnetic_nozzle_field_components(
    B0, x0, length, alpha, x, y, z,
)
    radius = sqrt(y*y + z*z)
    wavenumber = typeof(x)(2) * typeof(x)(pi) / length
    phase = wavenumber * (x - x0)
    argument = wavenumber * radius
    i0 = modified_bessel_i0_small(argument)
    i1 = modified_bessel_i1_small(argument)
    bx = B0 * (one(B0) - alpha*cos(phase)*i0)
    br = -B0 * alpha*sin(phase)*i1
    if radius > zero(radius)
        by = br * y / radius
        bz = br * z / radius
    else
        by = zero(br)
        bz = zero(br)
    end
    return bx, by, bz
end

@inline function bessel_magnetic_nozzle_vector_potential(
    B0, x0, length, alpha, x, y, z,
)
    radius = sqrt(y*y + z*z)
    wavenumber = typeof(x)(2) * typeof(x)(pi) / length
    phase = wavenumber * (x - x0)
    i1 = modified_bessel_i1_small(wavenumber * radius)
    aphi = B0 / typeof(x)(2) * (
        radius - alpha * length / typeof(x)(pi) * cos(phase) * i1
    )
    if radius > zero(radius)
        ay = -aphi * z / radius
        az = aphi * y / radius
    else
        ay = zero(aphi)
        az = zero(aphi)
    end
    return SVector{3,typeof(aphi)}(zero(aphi), ay, az)
end

# Structured coordinates are node coordinates.  Keep the sampling locations
# explicit here so the Q ghost field and the CT face flux use the same
# geometric convention on Cartesian and curvilinear meshes.
@inline function external_field_cell_center_coordinates(x, y, z, i, j, k)
    center = structured_cell_center_coordinates(x, y, z, i, j, k)
    return center[1], center[2], center[3]
end

@inline function external_field_i_face_center(x, y, z, i, j, k)
    return (
        (x[i,j,k] + x[i,j+1,k] + x[i,j,k+1] + x[i,j+1,k+1]) / 4,
        (y[i,j,k] + y[i,j+1,k] + y[i,j,k+1] + y[i,j+1,k+1]) / 4,
        (z[i,j,k] + z[i,j+1,k] + z[i,j,k+1] + z[i,j+1,k+1]) / 4,
    )
end

@inline function external_field_j_face_center(x, y, z, i, j, k)
    return (
        (x[i,j,k] + x[i+1,j,k] + x[i,j,k+1] + x[i+1,j,k+1]) / 4,
        (y[i,j,k] + y[i+1,j,k] + y[i,j,k+1] + y[i+1,j,k+1]) / 4,
        (z[i,j,k] + z[i+1,j,k] + z[i,j,k+1] + z[i+1,j,k+1]) / 4,
    )
end

@inline function external_field_k_face_center(x, y, z, i, j, k)
    return (
        (x[i,j,k] + x[i+1,j,k] + x[i,j+1,k] + x[i+1,j+1,k]) / 4,
        (y[i,j,k] + y[i+1,j,k] + y[i,j+1,k] + y[i+1,j+1,k]) / 4,
        (z[i,j,k] + z[i+1,j,k] + z[i,j+1,k] + z[i+1,j+1,k]) / 4,
    )
end

@inline function external_magnetic_face_flux(
    bx, by, bz, area, normal_x, normal_y, normal_z,
)
    return area * (bx*normal_x + by*normal_y + bz*normal_z)
end

@inline function is_prescribed_external_magnetic_field_bc(bc_type)
    return bc_type == Int32(BC_MHD_EXTERNAL_FIELD) ||
           bc_type == Int32(BC_MHD_RESERVOIR_INFLOW) ||
           bc_type == Int32(BC_MHD_PROFILED_INFLOW) ||
           bc_type == Int32(BC_MHD_FIXED_EXTERNAL_FIELD) ||
           bc_type == Int32(BC_MHD_OUTFLOW_EXTERNAL_FIELD)
end

@inline function external_magnetic_field_components_from_bc(bcp, x, y, z)
    model = bcp[BCP_MN_MODEL]
    if model >= typeof(model)(0.5)
        return bessel_magnetic_nozzle_field_components(
            bcp[BCP_MN_B0], bcp[BCP_MN_XC], bcp[BCP_MN_LB],
            bcp[BCP_MN_ALPHA], x, y, z,
        )
    end
    return finite_solenoid_external_field_components(
        bcp[BCP_MN_B0], bcp[BCP_MN_XC], bcp[BCP_MN_RB],
        bcp[BCP_MN_LB], Int32(round(bcp[BCP_MN_TURNS])), x, y, z,
    )
end

@inline function external_magnetic_vector_potential_from_bc(bcp, x, y, z)
    model = bcp[BCP_MN_MODEL]
    if model >= typeof(model)(0.5)
        return bessel_magnetic_nozzle_vector_potential(
            bcp[BCP_MN_B0], bcp[BCP_MN_XC], bcp[BCP_MN_LB],
            bcp[BCP_MN_ALPHA], x, y, z,
        )
    end
    return finite_solenoid_external_vector_potential(
        bcp[BCP_MN_B0], bcp[BCP_MN_XC], bcp[BCP_MN_RB],
        bcp[BCP_MN_LB], Int32(round(bcp[BCP_MN_TURNS])), x, y, z,
    )
end

@inline function _external_field_edge_integral(bcp, x0, y0, z0, x1, y1, z1)
    a0 = external_magnetic_vector_potential_from_bc(bcp, x0, y0, z0)
    a1 = external_magnetic_vector_potential_from_bc(bcp, x1, y1, z1)
    return (a0[1] + a1[1]) * (x1 - x0) / 2 +
           (a0[2] + a1[2]) * (y1 - y0) / 2 +
           (a0[3] + a1[3]) * (z1 - z0) / 2
end

@inline function external_magnetic_face_flux_from_vector_potential(
    bcp, x, y, z, i, j, k, face_kind,
)
    if face_kind == Int32(1)
        return _external_field_edge_integral(
            bcp, x[i,j,k], y[i,j,k], z[i,j,k],
            x[i,j+1,k], y[i,j+1,k], z[i,j+1,k],
        ) + _external_field_edge_integral(
            bcp, x[i,j+1,k], y[i,j+1,k], z[i,j+1,k],
            x[i,j+1,k+1], y[i,j+1,k+1], z[i,j+1,k+1],
        ) - _external_field_edge_integral(
            bcp, x[i,j,k+1], y[i,j,k+1], z[i,j,k+1],
            x[i,j+1,k+1], y[i,j+1,k+1], z[i,j+1,k+1],
        ) - _external_field_edge_integral(
            bcp, x[i,j,k], y[i,j,k], z[i,j,k],
            x[i,j,k+1], y[i,j,k+1], z[i,j,k+1],
        )
    elseif face_kind == Int32(2)
        return _external_field_edge_integral(
            bcp, x[i,j,k], y[i,j,k], z[i,j,k],
            x[i,j,k+1], y[i,j,k+1], z[i,j,k+1],
        ) + _external_field_edge_integral(
            bcp, x[i,j,k+1], y[i,j,k+1], z[i,j,k+1],
            x[i+1,j,k+1], y[i+1,j,k+1], z[i+1,j,k+1],
        ) - _external_field_edge_integral(
            bcp, x[i+1,j,k], y[i+1,j,k], z[i+1,j,k],
            x[i+1,j,k+1], y[i+1,j,k+1], z[i+1,j,k+1],
        ) - _external_field_edge_integral(
            bcp, x[i,j,k], y[i,j,k], z[i,j,k],
            x[i+1,j,k], y[i+1,j,k], z[i+1,j,k],
        )
    end
    return _external_field_edge_integral(
        bcp, x[i,j,k], y[i,j,k], z[i,j,k],
        x[i+1,j,k], y[i+1,j,k], z[i+1,j,k],
    ) + _external_field_edge_integral(
        bcp, x[i+1,j,k], y[i+1,j,k], z[i+1,j,k],
        x[i+1,j+1,k], y[i+1,j+1,k], z[i+1,j+1,k],
    ) - _external_field_edge_integral(
        bcp, x[i,j+1,k], y[i,j+1,k], z[i,j+1,k],
        x[i+1,j+1,k], y[i+1,j+1,k], z[i+1,j+1,k],
    ) - _external_field_edge_integral(
        bcp, x[i,j,k], y[i,j,k], z[i,j,k],
        x[i,j+1,k], y[i,j+1,k], z[i,j+1,k],
    )
end

@inline function structured_external_field_cache_required(
    boundary_types, rx, ry, rz, nprocs,
)
    return (rx == 0 && is_prescribed_external_magnetic_field_bc(boundary_types[1])) ||
           (rx == nprocs[1]-1 && is_prescribed_external_magnetic_field_bc(boundary_types[2])) ||
           (ry == 0 && is_prescribed_external_magnetic_field_bc(boundary_types[3])) ||
           (ry == nprocs[2]-1 && is_prescribed_external_magnetic_field_bc(boundary_types[4])) ||
           (rz == 0 && is_prescribed_external_magnetic_field_bc(boundary_types[5])) ||
           (rz == nprocs[3]-1 && is_prescribed_external_magnetic_field_bc(boundary_types[6]))
end

@inline function boundary_external_magnetic_field_components(
    ::Nothing, B0, x_center, coil_radius, coil_length, turns,
    x, y, z, i, j, k,
)
    center = external_field_cell_center_coordinates(x, y, z, i, j, k)
    return finite_solenoid_external_field_components(
        B0, x_center, coil_radius, coil_length, turns,
        center[1], center[2], center[3],
    )
end

@inline function boundary_external_magnetic_field_components(
    ::Nothing, bcp, x, y, z, i, j, k,
)
    center = external_field_cell_center_coordinates(x, y, z, i, j, k)
    return external_magnetic_field_components_from_bc(
        bcp, center[1], center[2], center[3],
    )
end


@inline function boundary_external_magnetic_field_components(
    cache, B0, x_center, coil_radius, coil_length, turns,
    x, y, z, i, j, k,
)
    return cache[i,j,k,1], cache[i,j,k,2], cache[i,j,k,3]
end

@inline function boundary_external_magnetic_field_components(
    cache, bcp, x, y, z, i, j, k,
)
    return cache[i,j,k,1], cache[i,j,k,2], cache[i,j,k,3]
end


function precompute_finite_solenoid_external_field_cache_kernel!(
    cache, x, y, z,
    rx, ry, rz, nxp, nyp, nzp, nprocs,
    bc_x_lo, bc_x_hi, bc_y_lo, bc_y_hi, bc_z_lo, bc_z_hi,
    bcp_x_lo, bcp_x_hi, bcp_y_lo, bcp_y_hi, bcp_z_lo, bcp_z_hi,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG ||
       i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    # Match fill_x -> fill_y -> fill_z overwrite order at edges and corners.
    active = false
    bcp = bcp_x_lo
    if rx == Int32(0) && i <= NG &&
       is_prescribed_external_magnetic_field_bc(bc_x_lo)
        active = true
        bcp = bcp_x_lo
    elseif rx == nprocs[1]-Int32(1) && i > nxp+NG &&
           is_prescribed_external_magnetic_field_bc(bc_x_hi)
        active = true
        bcp = bcp_x_hi
    end
    if ry == Int32(0) && j <= NG &&
       is_prescribed_external_magnetic_field_bc(bc_y_lo)
        active = true
        bcp = bcp_y_lo
    elseif ry == nprocs[2]-Int32(1) && j > nyp+NG &&
           is_prescribed_external_magnetic_field_bc(bc_y_hi)
        active = true
        bcp = bcp_y_hi
    end
    if rz == Int32(0) && k <= NG &&
       is_prescribed_external_magnetic_field_bc(bc_z_lo)
        active = true
        bcp = bcp_z_lo
    elseif rz == nprocs[3]-Int32(1) && k > nzp+NG &&
           is_prescribed_external_magnetic_field_bc(bc_z_hi)
        active = true
        bcp = bcp_z_hi
    end

    if active
        center = external_field_cell_center_coordinates(x, y, z, i, j, k)
        Bx, By, Bz = external_magnetic_field_components_from_bc(
            bcp, center[1], center[2], center[3],
        )
        @inbounds begin
            cache[i,j,k,1] = Bx
            cache[i,j,k,2] = By
            cache[i,j,k,3] = Bz
        end
    end
    return
end

@inline function _external_field_face_center(
    x, y, z, i, j, k, face_kind,
)
    if face_kind == Int32(1)
        return external_field_i_face_center(x, y, z, i, j, k)
    elseif face_kind == Int32(2)
        return external_field_j_face_center(x, y, z, i, j, k)
    end
    return external_field_k_face_center(x, y, z, i, j, k)
end

@inline function _external_field_face_metric_extent(
    nxp, nyp, nzp, ng, face_kind,
)
    return (
        nxp + Int32(2)*ng + (face_kind == Int32(1) ? Int32(1) : Int32(0)),
        nyp + Int32(2)*ng + (face_kind == Int32(2) ? Int32(1) : Int32(0)),
        nzp + Int32(2)*ng + (face_kind == Int32(3) ? Int32(1) : Int32(0)),
    )
end

@inline function _external_field_boundary_face_active(
    index, ncell, ng, boundary_axis, side, face_kind, include_normal_face,
)
    normal_component = boundary_axis == face_kind
    lower = side == Int32(0)
    if normal_component
        if !include_normal_face
            # The upper physical normal face is at ncell+ng+1.  In
            # ghost-only mode it must remain untouched, just like the lower
            # physical face at ng+1.
            return lower ? index <= ng : index > ncell + ng + Int32(1)
        end
        return lower ? index <= ng + Int32(1) : index >= ncell + ng + Int32(1)
    end
    return lower ? index <= ng : index > ncell + ng
end

# In background-field splitting, the analytic field is B0 and the evolved
# perturbation is B1 = B - B0.  A physical sheet receives B0 plus a zero
# gradient copy of B1 from the nearest interior face.  The mapping is made in
# face-array coordinates because normal faces have a duplicated endpoint while
# tangential face arrays do not.
@inline function _external_field_background_source_index(
    index, ncell, ng, boundary_axis, side, face_kind,
)
    lower = side == Int32(0)
    if boundary_axis == face_kind
        return lower ? ng + Int32(2) : ng + ncell
    end
    return lower ? ng + Int32(1) : ng + ncell
end

# Fill one Cartesian component of the staggered CT face state from the
# analytic external field.  `Bface` stores oriented face flux, so the kernel
# evaluates the field at the face center and projects it with the local CT
# metric before writing the array.
function ct_fill_external_field_face_b_kernel!(
    Bface, Area, normal_x, normal_y, normal_z, x, y, z,
    boundary_axis, side, face_kind,
    nxp, nyp, nzp, bcp, include_normal_face, background_splitting,
    background_face=nothing,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)
    metric_n1, metric_n2, metric_n3 = _external_field_face_metric_extent(
        nxp, nyp, nzp, ng, face_kind,
    )
    if i < Int32(1) || j < Int32(1) || k < Int32(1) ||
       i > metric_n1 || j > metric_n2 || k > metric_n3
        return
    end

    boundary_index = boundary_axis == Int32(1) ? i :
                     (boundary_axis == Int32(2) ? j : k)
    boundary_cells = boundary_axis == Int32(1) ? nxp :
                     (boundary_axis == Int32(2) ? nyp : nzp)
    _external_field_boundary_face_active(
        boundary_index, boundary_cells, ng,
        boundary_axis, side, face_kind, include_normal_face,
    ) || return

    # Use the same discrete Stokes construction as CT initialization. The
    # temporary point-value projection above is intentionally not used: it
    # differs from vector-potential face flux by O(h^2), which is fatal in
    # low-beta total-energy arithmetic.
    analytic_flux = external_magnetic_face_flux_from_vector_potential(
        bcp, x, y, z, i, j, k, face_kind,
    )
    if background_splitting
        source_i = boundary_axis == Int32(1) ?
            _external_field_background_source_index(
                i, nxp, ng, boundary_axis, side, face_kind) : i
        source_j = boundary_axis == Int32(2) ?
            _external_field_background_source_index(
                j, nyp, ng, boundary_axis, side, face_kind) : j
        source_k = boundary_axis == Int32(3) ?
            _external_field_background_source_index(
                k, nzp, ng, boundary_axis, side, face_kind) : k
        @inbounds begin
            # The evolved array stores b.  Impose the prescribed total field
            # by subtracting the fixed B0 flux at the destination and adding
            # the nearest interior perturbation.
            Bface[i,j,k] = analytic_flux -
                _ct_background_face_flux(background_face, i, j, k) +
                Bface[source_i,source_j,source_k]
        end
    else
        @inbounds Bface[i,j,k] = analytic_flux
    end
    return
end

@inline function finite_solenoid_external_vector_potential(
    B0::T, x_center::T, coil_radius::T, coil_length::T, turns::Int32,
    x, y, z,
) where {T}
    current_scale = B0 / _finite_solenoid_center_field_per_current(
        coil_radius, coil_length, turns,
    )
    nseg = FINITE_SOLENOID_AZIMUTHAL_SEGMENTS
    dphi = T(2) * T(pi) / T(nseg)
    value_type = promote_type(T, typeof(x), typeof(y), typeof(z))
    a_phi = zero(value_type)
    radius = sqrt(y*y + z*z)
    half_length = coil_length / 2
    for turn in Int32(0):(turns - Int32(1))
        x_turn = x_center - half_length +
            (T(turn) + T(0.5)) * coil_length / T(turns)
        for segment in Int32(0):(nseg - Int32(1))
            phi = (T(segment) + T(0.5)) * dphi
            source_y = coil_radius * cos(phi)
            source_z = coil_radius * sin(phi)
            dx = x - x_turn
            dy = radius - source_y
            dz = -source_z
            inv_r = inv(sqrt(dx*dx + dy*dy + dz*dz + T(1.0e-18)))
            dl_z = coil_radius * cos(phi) * dphi
            a_phi += dl_z * inv_r
        end
    end
    coefficient = current_scale / (T(4) * T(pi))
    azimuthal_scale = radius > zero(radius) ?
        coefficient*a_phi/radius : zero(value_type)
    return SVector{3,typeof(azimuthal_scale)}(
        zero(azimuthal_scale), -azimuthal_scale*z, azimuthal_scale*y,
    )
end

@inline function magnetic_nozzle_field(
    field::FiniteSolenoidMagneticField, x, y, z, time=zero(field.B0),
)
    return SVector{3}(
        finite_solenoid_external_field_components(
            field.B0, field.x_center, field.coil_radius, field.coil_length,
            field.turns, x, y, z,
        )..., 
    )
end

@inline function magnetic_nozzle_vector_potential(
    field::FiniteSolenoidMagneticField, x, y, z,
    time=zero(field.B0),
)
    return finite_solenoid_external_vector_potential(
        field.B0, field.x_center, field.coil_radius, field.coil_length,
        field.turns, x, y, z,
    )
end

@inline function (field::FiniteSolenoidMagneticField)(x, y, z, time=zero(field.B0))
    return magnetic_nozzle_vector_potential(field, x, y, z, time)
end

function AxisymmetricMagneticNozzleField(
    B0::T, x0::T, length::T, expansion_ratio::T,
) where {T<:AbstractFloat}
    isfinite(B0) || throw(ArgumentError("B0 must be finite"))
    isfinite(x0) || throw(ArgumentError("x0 must be finite"))
    isfinite(length) && length > zero(T) || throw(ArgumentError(
        "magnetic nozzle length must be finite and positive",
    ))
    isfinite(expansion_ratio) && expansion_ratio >= one(T) || throw(ArgumentError(
        "magnetic nozzle expansion_ratio must be finite and at least one",
    ))
    return AxisymmetricMagneticNozzleField{T}(
        B0, x0, length, expansion_ratio,
    )
end

function AxisymmetricMagneticNozzleField(
    B0::Real, x0::Real, length::Real, expansion_ratio::Real,
)
    values = promote(float(B0), float(x0), float(length), float(expansion_ratio))
    return AxisymmetricMagneticNozzleField(values...)
end

function BesselMagneticNozzleField(
    B0::T, x0::T, length::T, alpha::T,
) where {T<:AbstractFloat}
    isfinite(B0) || throw(ArgumentError("B0 must be finite"))
    isfinite(x0) || throw(ArgumentError("x0 must be finite"))
    isfinite(length) && length > zero(T) || throw(ArgumentError(
        "magnetic nozzle length must be finite and positive",
    ))
    isfinite(alpha) && zero(T) <= alpha <= one(T) || throw(ArgumentError(
        "Bessel nozzle alpha must be in [0, 1]",
    ))
    return BesselMagneticNozzleField{T}(B0, x0, length, alpha)
end

function BesselMagneticNozzleField(
    B0::Real, x0::Real, length::Real, alpha::Real,
)
    values = promote(float(B0), float(x0), float(length), float(alpha))
    return BesselMagneticNozzleField(values...)
end

@inline function magnetic_nozzle_profile(field::AxisymmetricMagneticNozzleField, x)
    ξ = (x - field.x0) / field.length
    downstream_fraction = inv(field.expansion_ratio)
    transition = (one(ξ) + tanh(ξ)) / 2
    profile = one(ξ) - (one(ξ) - downstream_fraction) * transition
    derivative = -(one(ξ) - downstream_fraction) *
        (one(ξ) - tanh(ξ)^2) / (2 * field.length)
    return profile, derivative
end

"""Return the divergence-free Cartesian magnetic field of the nozzle model."""
@inline function magnetic_nozzle_field(
    field::AxisymmetricMagneticNozzleField, x, y, z, time=zero(field.B0),
)
    profile, derivative = magnetic_nozzle_profile(field, x)
    scale = field.B0
    return SVector{3,typeof(scale * profile)}(
        scale * profile,
        -scale * derivative * y / 2,
        -scale * derivative * z / 2,
    )
end

@inline function magnetic_nozzle_field(
    field::BesselMagneticNozzleField, x, y, z, time=zero(field.B0),
)
    return SVector{3}(
        bessel_magnetic_nozzle_field_components(
            field.B0, field.x0, field.length, field.alpha, x, y, z,
        )...,
    )
end

"""
Return an axisymmetric vector potential whose curl is
`magnetic_nozzle_field(field, x, y, z, time)`.

The potential is regular on the nozzle axis and therefore avoids the
coordinate singularity of an explicit azimuthal `A_phi` expression.
"""
@inline function magnetic_nozzle_vector_potential(
    field::AxisymmetricMagneticNozzleField, x, y, z,
    time=zero(field.B0),
)
    profile, _ = magnetic_nozzle_profile(field, x)
    scale = field.B0 * profile / 2
    return SVector{3,typeof(scale * y)}(
        zero(scale),
        -scale * z,
        scale * y,
    )
end

@inline function magnetic_nozzle_vector_potential(
    field::BesselMagneticNozzleField, x, y, z,
    time=zero(field.B0),
)
    return bessel_magnetic_nozzle_vector_potential(
        field.B0, field.x0, field.length, field.alpha, x, y, z,
    )
end

@inline function (field::AxisymmetricMagneticNozzleField)(
    x, y, z, time=zero(field.B0),
)
    return magnetic_nozzle_vector_potential(field, x, y, z, time)
end

@inline function (field::BesselMagneticNozzleField)(
    x, y, z, time=zero(field.B0),
)
    return magnetic_nozzle_vector_potential(field, x, y, z, time)
end

function _run_configured_external_magnetic_field_process!(
    host_module::Module, blocks,
)
    enabled = isdefined(host_module, :external_magnetic_field) &&
        Bool(getfield(host_module, :external_magnetic_field))
    enabled || return false

    equation_type == :MHD && ct_mode || throw(ArgumentError(
        "external_magnetic_field currently requires structured CT-MHD",
    ))
    isdefined(host_module, :external_magnetic_field_model) || throw(ArgumentError(
        "external_magnetic_field=true requires " *
        "external_magnetic_field_model",
    ))
    model = getfield(host_module, :external_magnetic_field_model)
    applicable(model, zero(FT), zero(FT), zero(FT), zero(FT)) ||
        throw(ArgumentError(
            "external_magnetic_field_model must be callable as " *
            "(x, y, z, time)",
        ))

    for block in values(blocks)
        block.Bx_face === nothing && throw(ArgumentError(
            "external magnetic field initialization requires CT face arrays",
        ))
        ct_initial_face_flux_from_vector_potential!(
            block, model; time=zero(FT),
        )
    end
    return true
end
