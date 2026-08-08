# High-order finite-volume initialization for analytic structured cases.
#
# The legacy initializer writes primitive point values at cell centers and
# prim2c converts those values algebraically. This opt-in path integrates the
# conservative state directly over each physical cell while leaving the
# provisional center-point Q available for the first CT face-B bootstrap. The
# normal synchronization/task-graph path owns all later Q derivation.

if !@isdefined(STRUCTURED_INITIAL_QUADRATURE_LOADED)
const STRUCTURED_INITIAL_QUADRATURE_LOADED = true

@inline function _structured_gauss3_node(sample::Int32, ::Type{T}) where {T}
    offset = sqrt(T(15)) / T(10)
    return sample == Int32(1) ? -offset :
           sample == Int32(2) ? zero(T) : offset
end

@inline function _structured_gauss3_weight(sample::Int32, ::Type{T}) where {T}
    return sample == Int32(2) ? T(4) / T(9) : T(5) / T(18)
end

@inline function _structured_trilinear_scalar(
    q000, q100, q010, q110, q001, q101, q011, q111,
    a, b, c,
)
    T = typeof(q000)
    ta = a + T(0.5)
    tb = b + T(0.5)
    tc = c + T(0.5)
    ama = one(T) - ta
    amb = one(T) - tb
    amc = one(T) - tc

    value = ama*amb*amc*q000 + ta*amb*amc*q100 +
            ama*tb*amc*q010 + ta*tb*amc*q110 +
            ama*amb*tc*q001 + ta*amb*tc*q101 +
            ama*tb*tc*q011 + ta*tb*tc*q111
    derivative_a = amb*amc*(q100-q000) + tb*amc*(q110-q010) +
                   amb*tc*(q101-q001) + tb*tc*(q111-q011)
    derivative_b = ama*amc*(q010-q000) + ta*amc*(q110-q100) +
                   ama*tc*(q011-q001) + ta*tc*(q111-q101)
    derivative_c = ama*amb*(q001-q000) + ta*amb*(q101-q100) +
                   ama*tb*(q011-q010) + ta*tb*(q111-q110)
    return value, derivative_a, derivative_b, derivative_c
end

@inline function _structured_trilinear_cell_sample(
    x, y, z, i, j, k, a, b, c,
)
    @inbounds begin
        x000, x100 = x[i,j,k], x[i+1,j,k]
        x010, x110 = x[i,j+1,k], x[i+1,j+1,k]
        x001, x101 = x[i,j,k+1], x[i+1,j,k+1]
        x011, x111 = x[i,j+1,k+1], x[i+1,j+1,k+1]
        y000, y100 = y[i,j,k], y[i+1,j,k]
        y010, y110 = y[i,j+1,k], y[i+1,j+1,k]
        y001, y101 = y[i,j,k+1], y[i+1,j,k+1]
        y011, y111 = y[i,j+1,k+1], y[i+1,j+1,k+1]
        z000, z100 = z[i,j,k], z[i+1,j,k]
        z010, z110 = z[i,j+1,k], z[i+1,j+1,k]
        z001, z101 = z[i,j,k+1], z[i+1,j,k+1]
        z011, z111 = z[i,j+1,k+1], z[i+1,j+1,k+1]
    end
    xq, dxa, dxb, dxc = _structured_trilinear_scalar(
        x000,x100,x010,x110,x001,x101,x011,x111,a,b,c,
    )
    yq, dya, dyb, dyc = _structured_trilinear_scalar(
        y000,y100,y010,y110,y001,y101,y011,y111,a,b,c,
    )
    zq, dza, dzb, dzc = _structured_trilinear_scalar(
        z000,z100,z010,z110,z001,z101,z011,z111,a,b,c,
    )
    # Explicit determinant of the physical Jacobian. This avoids allocating a
    # small matrix in a GPU kernel.
    jacobian = dxa*(dyb*dzc-dyc*dzb) -
               dxb*(dya*dzc-dyc*dza) +
               dxc*(dya*dzb-dyb*dza)
    return xq, yq, zq, jacobian
end

# The repository's run scripts define the gas gamma with a legacy Unicode
# symbol. Resolve it on the host and pass it as a scalar kernel argument so
# this new file remains ASCII and the GPU kernel has no dynamic global lookup.
const _STRUCTURED_INITIAL_GAMMA_NAME = Symbol(Char(0x03b3))

@inline function _structured_initial_gamma()
    scope = @__MODULE__
    return isdefined(scope, _STRUCTURED_INITIAL_GAMMA_NAME) ?
           FT(getfield(scope, _STRUCTURED_INITIAL_GAMMA_NAME)) : FT(5) / FT(3)
end

@inline function _structured_orszag_tang_primitive(x, y, gamma)
    rho = gamma * gamma
    pressure = gamma
    u = -sin(y)
    v = sin(x)
    w = zero(FT)
    bscale = SQRT_MU0_SI
    bx = -bscale * sin(y)
    by = bscale * sin(FT(2) * x)
    bz = zero(FT)
    psi = zero(FT)
    return rho, u, v, w, pressure, bx, by, bz, psi
end

@inline function _structured_tgv_primitive(x, y, z, gamma)
    pulse = FT(0.1) * exp(
        -FT(10) * (
            (x-FT(1.5))^2 + (y-FT(1.5))^2 + (z-FT(1.5))^2
        ),
    )
    rho = one(FT) + pulse
    pressure = FT(10) + pulse
    return rho, FT(0.1), FT(0.1), zero(FT), pressure,
           zero(FT), zero(FT), zero(FT), zero(FT)
end

@inline function _structured_analytic_initial_primitive(
    ::Val{:orszag_tang}, x, y, z, gamma,
)
    return _structured_orszag_tang_primitive(x, y, gamma)
end

@inline function _structured_analytic_initial_primitive(
    ::Val{:tgv}, x, y, z, gamma,
)
    return _structured_tgv_primitive(x, y, z, gamma)
end

function structured_cell_average_analytic_kernel!(
    U, x, y, z, nxp, nyp, nzp, gamma, case_tag,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i < Int32(NG+1) || i > nxp + Int32(NG) ||
       j < Int32(NG+1) || j > nyp + Int32(NG) ||
       k < Int32(NG+1) || k > nzp + Int32(NG)
        return
    end

    total_weight = zero(FT)
    rho_average = zero(FT)
    momentum_x_average = zero(FT)
    momentum_y_average = zero(FT)
    momentum_z_average = zero(FT)
    energy_average = zero(FT)
    bx_average = zero(FT)
    by_average = zero(FT)
    bz_average = zero(FT)
    psi_average = zero(FT)

    for a_sample in Int32(1):Int32(3),
        b_sample in Int32(1):Int32(3),
        c_sample in Int32(1):Int32(3)
        a = _structured_gauss3_node(a_sample, FT)
        b = _structured_gauss3_node(b_sample, FT)
        c = _structured_gauss3_node(c_sample, FT)
        quadrature_weight =
            _structured_gauss3_weight(a_sample, FT) *
            _structured_gauss3_weight(b_sample, FT) *
            _structured_gauss3_weight(c_sample, FT)
        xq, yq, zq, jacobian = _structured_trilinear_cell_sample(
            x, y, z, i, j, k, a, b, c,
        )
        physical_weight = quadrature_weight * abs(jacobian)
        rho, u, v, w, pressure, bx, by, bz, psi =
            _structured_analytic_initial_primitive(
                case_tag, xq, yq, zq, gamma,
            )
        @static if equation_type == :MHD
            energy = mhd_energy_density_from_primitive(
                rho, u, v, w, pressure, bx, by, bz, gamma,
            )
        else
            energy = pressure/(gamma-one(FT)) +
                     FT(0.5)*rho*(u*u+v*v+w*w)
        end
        total_weight += physical_weight
        rho_average += physical_weight * rho
        momentum_x_average += physical_weight * rho * u
        momentum_y_average += physical_weight * rho * v
        momentum_z_average += physical_weight * rho * w
        energy_average += physical_weight * energy
        bx_average += physical_weight * bx
        by_average += physical_weight * by
        bz_average += physical_weight * bz
        psi_average += physical_weight * psi
    end

    inverse_weight = inv(total_weight)
    @inbounds begin
        U[i,j,k,1] = rho_average * inverse_weight
        U[i,j,k,2] = momentum_x_average * inverse_weight
        U[i,j,k,3] = momentum_y_average * inverse_weight
        U[i,j,k,4] = momentum_z_average * inverse_weight
        U[i,j,k,5] = energy_average * inverse_weight
    end
    @static if equation_type == :MHD && !ct_mode
        @inbounds begin
            U[i,j,k,UBX] = bx_average * inverse_weight
            U[i,j,k,UBY] = by_average * inverse_weight
            U[i,j,k,UBZ] = bz_average * inverse_weight
            U[i,j,k,UPSI] = psi_average * inverse_weight
        end
    end
    return
end

@inline function structured_cell_average_initialization_active()
    return initial_state_mode != INITIAL_STATE_LEGACY_POINT
end

function structured_initialize_cell_average!(U, Q, x, y, z, nxp, nyp, nzp)
    structured_cell_average_initialization_active() || return false
    nb = (
        cld(nxp + 2*NG, nthreads[1]),
        cld(nyp + 2*NG, nthreads[2]),
        cld(nzp + 2*NG, nthreads[3]),
    )
    gamma = _structured_initial_gamma()
    if test_case == "OrszagTang"
        equation_type == :MHD || throw(ArgumentError(
            "OrszagTang cell-average initialization requires equation_type=:MHD",
        ))
        @gpu_launch threads=nthreads blocks=nb structured_cell_average_analytic_kernel!(
            U, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp), gamma,
            Val(:orszag_tang),
        )
    elseif test_case == "TGV"
        equation_type == :compressible || throw(ArgumentError(
            "TGV cell-average initialization requires equation_type=:compressible",
        ))
        @gpu_launch threads=nthreads blocks=nb structured_cell_average_analytic_kernel!(
            U, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp), gamma,
            Val(:tgv),
        )
    else
        throw(ArgumentError(
            "initial_state_mode=$initial_state_mode has no conservative " *
            "cell-average initializer for test_case=$test_case",
        ))
    end
    return true
end

function structured_initialize_orszag_tang_face_flux!(b)
    structured_cell_average_initialization_active() || return false
    test_case == "OrszagTang" || return false
    vector_potential(x, y, z, time) = SVector{3,FT}(
        zero(FT), zero(FT),
        SQRT_MU0_SI * (cos(y) + FT(0.5) * cos(FT(2) * x)),
    )
    ct_initial_face_flux_from_vector_potential!(b, vector_potential)
    return true
end

end
