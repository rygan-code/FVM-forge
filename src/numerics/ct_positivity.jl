# Observation-only positivity diagnostics for the strict CT+HLLD path.

const CT_POS_META_LEN = 23
const CT_POS_VALUE_LEN = 17
const CT_POS_SITE_RECONSTRUCTED = Int32(1)
const CT_POS_SITE_FACE_B_INVARIANT = Int32(2)
const CT_POS_SITE_HLLD_INPUT = Int32(3)
const CT_POS_SITE_POST_HYDRO = Int32(4)
const CT_POS_SITE_POST_CT_SYNC = Int32(5)
const CT_POS_SITE_GHOST_REFRESH = Int32(6)
const CT_POS_SITE_PRE_CT_SYNC = Int32(7)
const CT_POS_SITE_FINALIZE_FLOOR = Int32(8)
const CT_POS_SITE_POST_CT_RECONCILE = Int32(9)
const CT_POS_SITE_FOFC_CLOSURE = Int32(10)
const CT_POS_SITE_FOFC_ANCHOR = Int32(11)
const CT_POS_WENO_TO_PLM_COUNT = 8
const CT_POS_PLM_TO_FIRST_COUNT = 9
const CT_POS_HLLD_TO_HLLE_COUNT = 10
const CT_POS_POINT6_TO_AO_COUNT = 11
const CT_POS_POINT6_LIMIT_COUNT = 12
if !@isdefined(CT_POS_FACE_P2A_TO_AO_COUNT)
    const CT_POS_FACE_P2A_TO_AO_COUNT = 13
    const CT_POS_FACE_P2A_TO_MIDPOINT_COUNT = 14
end
const CT_POS_FOFC_CELL_COUNT = 15
const CT_POS_FOFC_FACE_COUNT = 16
const CT_POS_FOFC_NEW_CELL_COUNT = 17
const CT_POS_FOFC_BAD_CELL_COUNT = 18
const CT_POS_POINT6_UNRECOVERABLE_COUNT = 19
const CT_POS_FOFC_LIMIT_COUNT = 20
const CT_POS_FOFC_REDUCED_CELL_COUNT = 21
const CT_POS_FOFC_UNRECOVERABLE_COUNT = 22
const CT_POS_CHARACTERISTIC_LIMIT_COUNT = 23
const CT_POS_FAILURE_META_LEN = 7
const CT_FOFC_DIFFUSIVE_ACTIVE =
    (@isdefined(viscous) ? Bool(viscous) : false) ||
    ((@isdefined(equation_type) ? equation_type : :Compressible) == :MHD &&
     (@isdefined(resistive) ? Bool(resistive) : false))

@inline function _ct_try_claim!(meta)
    @static if @isdefined(gpu_atomic_cas!)
        return gpu_atomic_cas!(
            meta, 1, Int32(0), Int32(1),
        ) == Int32(0)
    elseif @isdefined(USE_CUDA) && USE_CUDA
        return CUDA.atomic_cas!(pointer(meta), Int32(0), Int32(1)) == Int32(0)
    else
        return Core.Intrinsics.atomic_pointerreplace(
            pointer(meta), Int32(0), Int32(1),
            :acquire_release, :acquire,
        ).success
    end
end

@inline function ct_record_fallback!(meta, counter_index::Integer)
    @static if @isdefined(gpu_atomic_add!)
        gpu_atomic_add!(meta, counter_index, Int32(1))
    else
        pointer_value = pointer(meta, counter_index)
        old = @inbounds meta[counter_index]
        while true
            result = Core.Intrinsics.atomic_pointerreplace(
                pointer_value, old, old + Int32(1),
                :acquire_release, :acquire,
            )
            result.success && break
            old = result.old
        end
    end
    return nothing
end

@inline function _ct_store_failure!(
    meta, values,
    site::Int32, direction::Int32, side::Int32,
    i::Int32, j::Int32, k::Int32,
    U, bface, rho, kinetic, magnetic, internal, pressure,
)
    if _ct_try_claim!(meta)
        @inbounds begin
            meta[2] = site
            meta[3] = direction
            meta[4] = side
            meta[5] = i
            meta[6] = j
            meta[7] = k

            values[1] = rho
            values[2] = kinetic
            values[3] = magnetic
            values[4] = internal
            values[5] = pressure
            values[6] = U[6]
            values[7] = U[7]
            values[8] = U[8]
            values[9] = bface
            values[10] = U[5]
        end
    end
    return nothing
end

@inline function ct_record_state_if_invalid!(
    meta, values,
    site::Int32, direction::Int32, side::Int32,
    i::Int32, j::Int32, k::Int32,
    U, bface, gamma,
)
    rho, kinetic, magnetic, internal, pressure = mhd_raw_thermo(U, gamma)
    valid = isfinite(rho) && rho > zero(rho) &&
            isfinite(internal) && internal > zero(internal) &&
            isfinite(pressure) && pressure > zero(pressure)
    if valid
        return false
    end

    _ct_store_failure!(
        meta, values, site, direction, side, i, j, k, U, bface,
        rho, kinetic, magnetic, internal, pressure,
    )
    return true
end

@inline function _ct_record_if_invalid!(
    meta, values,
    site::Int32, direction::Int32, side::Int32,
    i::Int32, j::Int32, k::Int32,
    U, bface,
)
    return ct_record_state_if_invalid!(
        meta, values, site, direction, side, i, j, k, U, bface, γ,
    )
end

@inline function ct_thermodynamic_state_is_positive(
    rho, kinetic, magnetic, internal, pressure, energy,
)
    finite_carrier = isfinite(rho) && isfinite(kinetic) &&
                     isfinite(magnetic) && isfinite(pressure) &&
                     isfinite(energy)
    @static if isothermal_mhd
        return finite_carrier && rho > zero(rho) && pressure > zero(pressure)
    else
        return finite_carrier && rho > zero(rho) &&
               isfinite(internal) && internal > zero(internal) &&
               pressure > zero(pressure)
    end
end

@inline function ct_thermodynamic_state_is_above_floors(
    rho, kinetic, magnetic, internal, pressure, energy,
    minimum_density, minimum_pressure,
)
    finite_carrier = isfinite(rho) && isfinite(kinetic) &&
                     isfinite(magnetic) && isfinite(pressure) &&
                     isfinite(energy)
    @static if isothermal_mhd
        return finite_carrier && rho >= minimum_density &&
               pressure >= minimum_pressure
    else
        return finite_carrier && rho >= minimum_density &&
               isfinite(internal) && internal > zero(internal) &&
               pressure >= minimum_pressure
    end
end

@inline function ct_record_cell_values_if_invalid!(
    meta, values, site::Int32, i::Int32, j::Int32, k::Int32,
    rho, momentum_x, momentum_y, momentum_z, energy,
    magnetic_x, magnetic_y, magnetic_z, psi, gamma,
)
    rho_raw, kinetic, magnetic, internal, pressure =
        mhd_raw_thermo_components(
            rho, momentum_x, momentum_y, momentum_z, energy,
            magnetic_x, magnetic_y, magnetic_z, gamma,
        )
    valid = ct_thermodynamic_state_is_positive(
        rho_raw, kinetic, magnetic, internal, pressure, energy,
    )
    if valid
        return false
    end

    if _ct_try_claim!(meta)
        @inbounds begin
            meta[2] = site
            meta[3] = Int32(0)
            meta[4] = Int32(0)
            meta[5] = i
            meta[6] = j
            meta[7] = k

            values[1] = rho_raw
            values[2] = kinetic
            values[3] = magnetic
            values[4] = internal
            values[5] = pressure
            values[6] = magnetic_x
            values[7] = magnetic_y
            values[8] = magnetic_z
            values[9] = oftype(rho_raw, NaN)
            values[10] = energy
        end
    end
    return true
end

@inline function ct_record_point_primitive_if_invalid!(
    meta, values, site::Int32, i::Int32, j::Int32, k::Int32,
    rho, velocity_x, velocity_y, velocity_z, pressure,
    magnetic_x, magnetic_y, magnetic_z, psi, gamma,
)
    kinetic = rho * (
        velocity_x^2 + velocity_y^2 + velocity_z^2
    ) / 2
    magnetic = (
        magnetic_x^2 + magnetic_y^2 + magnetic_z^2
    ) * INV_MU0_SI / 2
    @static if isothermal_mhd
        internal = pressure
    else
        internal = pressure / (gamma-one(gamma))
    end
    energy = kinetic + magnetic + internal
    finite = isfinite(rho) && isfinite(velocity_x) &&
             isfinite(velocity_y) && isfinite(velocity_z) &&
             isfinite(pressure) && isfinite(magnetic_x) &&
             isfinite(magnetic_y) && isfinite(magnetic_z) &&
             isfinite(psi) && isfinite(energy)
    valid = finite && rho > zero(rho) && pressure > zero(pressure)
    @static if !isothermal_mhd
        valid &= isfinite(internal) && internal > zero(internal)
    end
    valid && return false

    point_state = SVector{9,typeof(rho)}(
        rho,rho*velocity_x,rho*velocity_y,rho*velocity_z,energy,
        magnetic_x,magnetic_y,magnetic_z,psi,
    )
    _ct_store_failure!(
        meta,values,site,Int32(0),Int32(0),i,j,k,
        point_state,oftype(rho,NaN),rho,kinetic,magnetic,internal,pressure,
    )
    return true
end

@inline function ct_record_cell_values_if_below_floors!(
    meta, values, site::Int32, i::Int32, j::Int32, k::Int32,
    rho, momentum_x, momentum_y, momentum_z, energy,
    magnetic_x, magnetic_y, magnetic_z, psi, gamma,
    minimum_density, minimum_pressure, gas_constant,
    center_x, center_y, center_z, boundary_distance,
)
    rho_raw, kinetic, magnetic, internal, pressure =
        mhd_raw_thermo_components(
            rho, momentum_x, momentum_y, momentum_z, energy,
            magnetic_x, magnetic_y, magnetic_z, gamma,
        )
    valid = ct_thermodynamic_state_is_above_floors(
        rho_raw, kinetic, magnetic, internal, pressure, energy,
        minimum_density, minimum_pressure,
    )
    if valid
        return false
    end

    if _ct_try_claim!(meta)
        temperature = pressure / (rho_raw * gas_constant)
        @inbounds begin
            meta[2] = site
            meta[3] = Int32(0)
            meta[4] = Int32(0)
            meta[5] = i
            meta[6] = j
            meta[7] = k

            values[1] = rho_raw
            values[2] = kinetic
            values[3] = magnetic
            values[4] = internal
            values[5] = pressure
            values[6] = magnetic_x
            values[7] = magnetic_y
            values[8] = magnetic_z
            values[9] = oftype(rho_raw, NaN)
            values[10] = energy
            values[11] = minimum_density
            values[12] = minimum_pressure
            values[13] = temperature
            values[14] = center_x
            values[15] = center_y
            values[16] = center_z
            values[17] = boundary_distance
        end
    end
    return true
end

function ct_check_cell_floor_positivity_kernel!(
    meta, values, U, Q, x, y, z,
    gamma::FT, minimum_density::FT, minimum_pressure::FT, gas_constant::FT,
    site::Int32, nxp::Int32, nyp::Int32, nzp::Int32,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ng = Int32(NG)
    ii, jj, kk = i + ng, j + ng, k + ng
    @inbounds begin
        rho = U[ii, jj, kk, 1]
        momentum_x = U[ii, jj, kk, 2]
        momentum_y = U[ii, jj, kk, 3]
        momentum_z = U[ii, jj, kk, 4]
        energy = U[ii, jj, kk, 5]
        magnetic_x = Q[ii, jj, kk, QBX]
        magnetic_y = Q[ii, jj, kk, QBY]
        magnetic_z = Q[ii, jj, kk, QBZ]
        center_x = (
            x[ii,jj,kk] + x[ii+Int32(1),jj,kk] +
            x[ii,jj+Int32(1),kk] + x[ii+Int32(1),jj+Int32(1),kk] +
            x[ii,jj,kk+Int32(1)] + x[ii+Int32(1),jj,kk+Int32(1)] +
            x[ii,jj+Int32(1),kk+Int32(1)] +
            x[ii+Int32(1),jj+Int32(1),kk+Int32(1)]
        ) / FT(8)
        center_y = (
            y[ii,jj,kk] + y[ii+Int32(1),jj,kk] +
            y[ii,jj+Int32(1),kk] + y[ii+Int32(1),jj+Int32(1),kk] +
            y[ii,jj,kk+Int32(1)] + y[ii+Int32(1),jj,kk+Int32(1)] +
            y[ii,jj+Int32(1),kk+Int32(1)] +
            y[ii+Int32(1),jj+Int32(1),kk+Int32(1)]
        ) / FT(8)
        center_z = (
            z[ii,jj,kk] + z[ii+Int32(1),jj,kk] +
            z[ii,jj+Int32(1),kk] + z[ii+Int32(1),jj+Int32(1),kk] +
            z[ii,jj,kk+Int32(1)] + z[ii+Int32(1),jj,kk+Int32(1)] +
            z[ii,jj+Int32(1),kk+Int32(1)] +
            z[ii+Int32(1),jj+Int32(1),kk+Int32(1)]
        ) / FT(8)
    end
    boundary_distance = min(
        i - Int32(1), nxp - i,
        j - Int32(1), nyp - j,
        k - Int32(1), nzp - k,
    )
    ct_record_cell_values_if_below_floors!(
        meta, values, site, ii, jj, kk,
        rho, momentum_x, momentum_y, momentum_z, energy,
        magnetic_x, magnetic_y, magnetic_z, zero(FT), gamma,
        minimum_density, minimum_pressure, gas_constant,
        center_x, center_y, center_z, FT(boundary_distance),
    )
    return
end

function ct_check_cell_positivity_kernel!(
    meta, values, U, Q, gamma::FT, site::Int32, include_ghost::Int32,
    nxp::Int32, nyp::Int32, nzp::Int32,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)

    if include_ghost == Int32(0)
        if i > nxp || j > nyp || k > nzp
            return
        end
        ii = i + ng
        jj = j + ng
        kk = k + ng
    else
        if i > nxp + Int32(2)*ng ||
           j > nyp + Int32(2)*ng ||
           k > nzp + Int32(2)*ng
            return
        end
        ii, jj, kk = i, j, k
    end

    @inbounds begin
        rho = U[ii, jj, kk, 1]
        momentum_x = U[ii, jj, kk, 2]
        momentum_y = U[ii, jj, kk, 3]
        momentum_z = U[ii, jj, kk, 4]
        energy = U[ii, jj, kk, 5]
        magnetic_x = Q[ii, jj, kk, QBX]
        magnetic_y = Q[ii, jj, kk, QBY]
        magnetic_z = Q[ii, jj, kk, QBZ]
        psi = zero(FT)
    end
    ct_record_cell_values_if_invalid!(
        meta, values, site, Int32(ii), Int32(jj), Int32(kk),
        rho, momentum_x, momentum_y, momentum_z, energy,
        magnetic_x, magnetic_y, magnetic_z, psi, gamma,
    )
    return
end

function ct_check_point_primitive_positivity_kernel!(
    meta, values, Q, gamma::FT, site::Int32, include_ghost::Int32,
    nxp::Int32, nyp::Int32, nzp::Int32,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)

    if include_ghost == Int32(0)
        if i > nxp || j > nyp || k > nzp
            return
        end
        ii = i + ng
        jj = j + ng
        kk = k + ng
    else
        if i > nxp + Int32(2)*ng ||
           j > nyp + Int32(2)*ng ||
           k > nzp + Int32(2)*ng
            return
        end
        ii, jj, kk = i, j, k
    end

    @inbounds ct_record_point_primitive_if_invalid!(
        meta,values,site,Int32(ii),Int32(jj),Int32(kk),
        Q[ii,jj,kk,1],Q[ii,jj,kk,2],Q[ii,jj,kk,3],Q[ii,jj,kk,4],
        Q[ii,jj,kk,5],Q[ii,jj,kk,QBX],Q[ii,jj,kk,QBY],Q[ii,jj,kk,QBZ],
        Q[ii,jj,kk,QPSI],gamma,
    )
    return
end

function ct_check_sync_transition_kernel!(
    meta, values, U, Q, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    gamma::FT, recovery_mode::Int32,
    nxp::Int32, nyp::Int32, nzp::Int32,
    B0x_face=nothing, B0y_face=nothing, B0z_face=nothing,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii = i + Int32(NG)
    jj = j + Int32(NG)
    kk = k + Int32(NG)
    @inbounds begin
        rho = U[ii, jj, kk, 1]
        momentum_x = U[ii, jj, kk, 2]
        momentum_y = U[ii, jj, kk, 3]
        momentum_z = U[ii, jj, kk, 4]
        energy = U[ii, jj, kk, 5]
        old_bx = Q[ii, jj, kk, QBX]
        old_by = Q[ii, jj, kk, QBY]
        old_bz = Q[ii, jj, kk, QBZ]
        area_i_lo = SVector(
            Areai[ii,jj,kk]*nxi[ii,jj,kk],
            Areai[ii,jj,kk]*nyi[ii,jj,kk],
            Areai[ii,jj,kk]*nzi[ii,jj,kk],
        )
        area_i_hi = SVector(
            Areai[ii+Int32(1),jj,kk]*nxi[ii+Int32(1),jj,kk],
            Areai[ii+Int32(1),jj,kk]*nyi[ii+Int32(1),jj,kk],
            Areai[ii+Int32(1),jj,kk]*nzi[ii+Int32(1),jj,kk],
        )
        area_j_lo = SVector(
            Areaj[ii,jj,kk]*nxj[ii,jj,kk],
            Areaj[ii,jj,kk]*nyj[ii,jj,kk],
            Areaj[ii,jj,kk]*nzj[ii,jj,kk],
        )
        area_j_hi = SVector(
            Areaj[ii,jj+Int32(1),kk]*nxj[ii,jj+Int32(1),kk],
            Areaj[ii,jj+Int32(1),kk]*nyj[ii,jj+Int32(1),kk],
            Areaj[ii,jj+Int32(1),kk]*nzj[ii,jj+Int32(1),kk],
        )
        area_k_lo = SVector(
            Areak[ii,jj,kk]*nxk[ii,jj,kk],
            Areak[ii,jj,kk]*nyk[ii,jj,kk],
            Areak[ii,jj,kk]*nzk[ii,jj,kk],
        )
        area_k_hi = SVector(
            Areak[ii,jj,kk+Int32(1)]*nxk[ii,jj,kk+Int32(1)],
            Areak[ii,jj,kk+Int32(1)]*nyk[ii,jj,kk+Int32(1)],
            Areak[ii,jj,kk+Int32(1)]*nzk[ii,jj,kk+Int32(1)],
        )
        recovered_b = ct_recover_cell_b(
            area_i_lo, area_i_hi, area_j_lo, area_j_hi,
            area_k_lo, area_k_hi,
            _ct_total_face_flux(Bx_face, B0x_face, ii, jj, kk),
            _ct_total_face_flux(Bx_face, B0x_face, ii+Int32(1), jj, kk),
            _ct_total_face_flux(By_face, B0y_face, ii, jj, kk),
            _ct_total_face_flux(By_face, B0y_face, ii, jj+Int32(1), kk),
            _ct_total_face_flux(Bz_face, B0z_face, ii, jj, kk),
            _ct_total_face_flux(Bz_face, B0z_face, ii, jj, kk+Int32(1)),
        )
        new_bx = recovered_b[1]
        new_by = recovered_b[2]
        new_bz = recovered_b[3]
    end

    _, kinetic, new_magnetic, new_internal, new_pressure =
        mhd_raw_thermo_components(
            rho, momentum_x, momentum_y, momentum_z, energy,
            new_bx, new_by, new_bz, gamma,
        )
    if ct_thermodynamic_state_is_positive(
        rho, kinetic, new_magnetic, new_internal, new_pressure, energy,
    )
        return
    end

    old_pressure = if recovery_mode == CT_CELL_B_POINT6
        @inbounds Q[ii,jj,kk,5]
    else
        mhd_raw_thermo_components(
            rho,momentum_x,momentum_y,momentum_z,energy,
            old_bx,old_by,old_bz,gamma,
        )[5]
    end

    if _ct_try_claim!(meta)
        @inbounds begin
            meta[2] = CT_POS_SITE_POST_CT_RECONCILE
            meta[3] = Int32(0)
            meta[4] = Int32(0)
            meta[5] = ii
            meta[6] = jj
            meta[7] = kk
            values[1] = rho
            values[2] = kinetic
            values[3] = new_magnetic
            values[4] = new_internal
            values[5] = new_pressure
            values[6] = new_bx
            values[7] = new_by
            values[8] = new_bz
            values[9] = old_pressure
            values[10] = energy
            values[11] = old_bx
            values[12] = old_by
            values[13] = old_bz
            values[14] = FT(NaN)
            values[15] = FT(NaN)
            values[16] = FT(NaN)
            values[17] = FT(NaN)
        end
    end
    return
end

@inline function _ct_fofc_candidate_i_face(
    face_b, face_b_base, edge_y, edge_z, face_i, cell_j, cell_k,
    dt, rk_a,
)
    ng = Int32(NG)
    ii, jj, kk = face_i+ng, cell_j+ng, cell_k+ng
    euler = ct_stokes_update_i(
        @inbounds(face_b[ii,jj,kk]),
        @inbounds(edge_y[face_i,cell_j,cell_k]),
        @inbounds(edge_y[face_i,cell_j,cell_k+Int32(1)]),
        @inbounds(edge_z[face_i,cell_j,cell_k]),
        @inbounds(edge_z[face_i,cell_j+Int32(1),cell_k]),
        dt,
    )
    base = @inbounds face_b_base[ii,jj,kk]
    return base + rk_a*(euler-base)
end

@inline function _ct_fofc_candidate_j_face(
    face_b, face_b_base, edge_x, edge_z, cell_i, face_j, cell_k,
    dt, rk_a,
)
    ng = Int32(NG)
    ii, jj, kk = cell_i+ng, face_j+ng, cell_k+ng
    euler = ct_stokes_update_j(
        @inbounds(face_b[ii,jj,kk]),
        @inbounds(edge_x[cell_i,face_j,cell_k]),
        @inbounds(edge_x[cell_i,face_j,cell_k+Int32(1)]),
        @inbounds(edge_z[cell_i,face_j,cell_k]),
        @inbounds(edge_z[cell_i+Int32(1),face_j,cell_k]),
        dt,
    )
    base = @inbounds face_b_base[ii,jj,kk]
    return base + rk_a*(euler-base)
end

@inline function _ct_fofc_candidate_k_face(
    face_b, face_b_base, edge_x, edge_y, cell_i, cell_j, face_k,
    dt, rk_a,
)
    ng = Int32(NG)
    ii, jj, kk = cell_i+ng, cell_j+ng, face_k+ng
    euler = ct_stokes_update_k(
        @inbounds(face_b[ii,jj,kk]),
        @inbounds(edge_x[cell_i,cell_j,face_k]),
        @inbounds(edge_x[cell_i,cell_j+Int32(1),face_k]),
        @inbounds(edge_y[cell_i,cell_j,face_k]),
        @inbounds(edge_y[cell_i+Int32(1),cell_j,face_k]),
        dt,
    )
    base = @inbounds face_b_base[ii,jj,kk]
    return base + rk_a*(euler-base)
end

@inline _ct_fofc_add_background(value, ::Nothing, i, j, k) = value
@inline _ct_fofc_add_background(value, background, i, j, k) =
    value + @inbounds(background[i,j,k])

@inline function _ct_fofc_anchor_face(face_b, face_b_base, i, j, k, rk_a)
    @inbounds begin
        base = face_b_base[i,j,k]
        return base+rk_a*(face_b[i,j,k]-base)
    end
end

@inline function _ct_fofc_topology_scale(flags, i, j, k)
    scale = one(FT)
    for dk in Int32(-1):Int32(1),
        dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
        # A cell shares a face or an edge, but not only a vertex, with the
        # target when at most two offsets are nonzero.
        nonzero_offsets = Int32(di != 0)+Int32(dj != 0)+Int32(dk != 0)
        nonzero_offsets <= Int32(2) || continue
        @inbounds value =
            flags[i+di,j+dj,k+dk,CT_FOFC_COMMITTED_CHANNEL]
        ct_fofc_flag_is_active(value) || continue
        scale = min(scale,ct_fofc_flag_scale(value))
    end
    return scale
end

function ct_mark_fofc_candidate_kernel!(
    flags, fallback_meta, fallback_values,
    U, Un, Q, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, Vol,
    Bx_face, By_face, Bz_face,
    Bx_face_base, By_face_base, Bz_face_base,
    B0x_face, B0y_face, B0z_face,
    Ex_edge, Ey_edge, Ez_edge,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    dt, rk_a, gamma, minimum_density, minimum_pressure,
    nxp, nyp, nzp,
)
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    (i > nxp || j > nyp || k > nzp) && return

    ng = Int32(NG)
    ii, jj, kk = i+ng, j+ng, k+ng
    @inbounds old_flag = flags[ii,jj,kk,CT_FOFC_COMMITTED_CHANNEL]
    @inbounds flags[ii,jj,kk,CT_FOFC_PROPOSAL_CHANNEL] = old_flag
    tangent = Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    volume_scaled_dt = @inbounds Vol[ii,jj,kk]*dt
    hydro = MVector{6,FT}(zero(FT),zero(FT),zero(FT),zero(FT),zero(FT),zero(FT))
    anchor_hydro = MVector{6,FT}(
        zero(FT),zero(FT),zero(FT),zero(FT),zero(FT),zero(FT),
    )
    for variable in 1:5
        inviscid =
            @inbounds(Fx[i,j+tangent,k+tangent,variable]) -
            @inbounds(Fx[i+Int32(1),j+tangent,k+tangent,variable]) +
            @inbounds(Fy[i+tangent,j,k+tangent,variable]) -
            @inbounds(Fy[i+tangent,j+Int32(1),k+tangent,variable]) +
            @inbounds(Fz[i+tangent,j+tangent,k,variable]) -
            @inbounds(Fz[i+tangent,j+tangent,k+Int32(1),variable])
        @static if CT_FOFC_DIFFUSIVE_ACTIVE
            diffusive =
                @inbounds(Fv_x[i,j+tangent,k+tangent,variable]) -
                @inbounds(Fv_x[i+Int32(1),j+tangent,k+tangent,variable]) +
                @inbounds(Fv_y[i+tangent,j,k+tangent,variable]) -
                @inbounds(Fv_y[i+tangent,j+Int32(1),k+tangent,variable]) +
                @inbounds(Fv_z[i+tangent,j+tangent,k,variable]) -
                @inbounds(Fv_z[i+tangent,j+tangent,k+Int32(1),variable])
        else
            diffusive = zero(FT)
        end
        euler = @inbounds(U[ii,jj,kk,variable]) +
            (inviscid-diffusive)*volume_scaled_dt
        hydro[variable] = @inbounds(Un[ii,jj,kk,variable]) +
            rk_a*(euler-@inbounds(Un[ii,jj,kk,variable]))
        anchor_hydro[variable] = @inbounds(Un[ii,jj,kk,variable]) +
            rk_a*(@inbounds(U[ii,jj,kk,variable])-@inbounds(Un[ii,jj,kk,variable]))
    end

    phi_i_lo = _ct_fofc_candidate_i_face(
        Bx_face,Bx_face_base,Ey_edge,Ez_edge,i,j,k,dt,rk_a,
    )
    phi_i_hi = _ct_fofc_candidate_i_face(
        Bx_face,Bx_face_base,Ey_edge,Ez_edge,i+Int32(1),j,k,dt,rk_a,
    )
    phi_j_lo = _ct_fofc_candidate_j_face(
        By_face,By_face_base,Ex_edge,Ez_edge,i,j,k,dt,rk_a,
    )
    phi_j_hi = _ct_fofc_candidate_j_face(
        By_face,By_face_base,Ex_edge,Ez_edge,i,j+Int32(1),k,dt,rk_a,
    )
    phi_k_lo = _ct_fofc_candidate_k_face(
        Bz_face,Bz_face_base,Ex_edge,Ey_edge,i,j,k,dt,rk_a,
    )
    phi_k_hi = _ct_fofc_candidate_k_face(
        Bz_face,Bz_face_base,Ex_edge,Ey_edge,i,j,k+Int32(1),dt,rk_a,
    )
    phi_i_lo = _ct_fofc_add_background(phi_i_lo,B0x_face,ii,jj,kk)
    phi_i_hi = _ct_fofc_add_background(phi_i_hi,B0x_face,ii+Int32(1),jj,kk)
    phi_j_lo = _ct_fofc_add_background(phi_j_lo,B0y_face,ii,jj,kk)
    phi_j_hi = _ct_fofc_add_background(phi_j_hi,B0y_face,ii,jj+Int32(1),kk)
    phi_k_lo = _ct_fofc_add_background(phi_k_lo,B0z_face,ii,jj,kk)
    phi_k_hi = _ct_fofc_add_background(phi_k_hi,B0z_face,ii,jj,kk+Int32(1))

    @inbounds begin
        area_i_lo = SVector{3,FT}(
            Areai[ii,jj,kk]*nxi[ii,jj,kk],
            Areai[ii,jj,kk]*nyi[ii,jj,kk],
            Areai[ii,jj,kk]*nzi[ii,jj,kk],
        )
        area_i_hi = SVector{3,FT}(
            Areai[ii+Int32(1),jj,kk]*nxi[ii+Int32(1),jj,kk],
            Areai[ii+Int32(1),jj,kk]*nyi[ii+Int32(1),jj,kk],
            Areai[ii+Int32(1),jj,kk]*nzi[ii+Int32(1),jj,kk],
        )
        area_j_lo = SVector{3,FT}(
            Areaj[ii,jj,kk]*nxj[ii,jj,kk],
            Areaj[ii,jj,kk]*nyj[ii,jj,kk],
            Areaj[ii,jj,kk]*nzj[ii,jj,kk],
        )
        area_j_hi = SVector{3,FT}(
            Areaj[ii,jj+Int32(1),kk]*nxj[ii,jj+Int32(1),kk],
            Areaj[ii,jj+Int32(1),kk]*nyj[ii,jj+Int32(1),kk],
            Areaj[ii,jj+Int32(1),kk]*nzj[ii,jj+Int32(1),kk],
        )
        area_k_lo = SVector{3,FT}(
            Areak[ii,jj,kk]*nxk[ii,jj,kk],
            Areak[ii,jj,kk]*nyk[ii,jj,kk],
            Areak[ii,jj,kk]*nzk[ii,jj,kk],
        )
        area_k_hi = SVector{3,FT}(
            Areak[ii,jj,kk+Int32(1)]*nxk[ii,jj,kk+Int32(1)],
            Areak[ii,jj,kk+Int32(1)]*nyk[ii,jj,kk+Int32(1)],
            Areak[ii,jj,kk+Int32(1)]*nzk[ii,jj,kk+Int32(1)],
        )
    end
    magnetic = ct_recover_cell_b(
        area_i_lo,area_i_hi,area_j_lo,area_j_hi,area_k_lo,area_k_hi,
        phi_i_lo,phi_i_hi,phi_j_lo,phi_j_hi,phi_k_lo,phi_k_hi,
    )
    anchor_phi_i_lo = _ct_fofc_anchor_face(
        Bx_face,Bx_face_base,ii,jj,kk,rk_a,
    )
    anchor_phi_i_hi = _ct_fofc_anchor_face(
        Bx_face,Bx_face_base,ii+Int32(1),jj,kk,rk_a,
    )
    anchor_phi_j_lo = _ct_fofc_anchor_face(
        By_face,By_face_base,ii,jj,kk,rk_a,
    )
    anchor_phi_j_hi = _ct_fofc_anchor_face(
        By_face,By_face_base,ii,jj+Int32(1),kk,rk_a,
    )
    anchor_phi_k_lo = _ct_fofc_anchor_face(
        Bz_face,Bz_face_base,ii,jj,kk,rk_a,
    )
    anchor_phi_k_hi = _ct_fofc_anchor_face(
        Bz_face,Bz_face_base,ii,jj,kk+Int32(1),rk_a,
    )
    anchor_phi_i_lo = _ct_fofc_add_background(
        anchor_phi_i_lo,B0x_face,ii,jj,kk,
    )
    anchor_phi_i_hi = _ct_fofc_add_background(
        anchor_phi_i_hi,B0x_face,ii+Int32(1),jj,kk,
    )
    anchor_phi_j_lo = _ct_fofc_add_background(
        anchor_phi_j_lo,B0y_face,ii,jj,kk,
    )
    anchor_phi_j_hi = _ct_fofc_add_background(
        anchor_phi_j_hi,B0y_face,ii,jj+Int32(1),kk,
    )
    anchor_phi_k_lo = _ct_fofc_add_background(
        anchor_phi_k_lo,B0z_face,ii,jj,kk,
    )
    anchor_phi_k_hi = _ct_fofc_add_background(
        anchor_phi_k_hi,B0z_face,ii,jj,kk+Int32(1),
    )
    anchor_magnetic = ct_recover_cell_b(
        area_i_lo,area_i_hi,area_j_lo,area_j_hi,area_k_lo,area_k_hi,
        anchor_phi_i_lo,anchor_phi_i_hi,anchor_phi_j_lo,anchor_phi_j_hi,
        anchor_phi_k_lo,anchor_phi_k_hi,
    )
    hydro_state = SVector{6,FT}(hydro)
    anchor_hydro_state = SVector{6,FT}(anchor_hydro)
    admissible = ct_point6_state_is_admissible(
        hydro_state,magnetic,gamma,minimum_density,minimum_pressure,
    )
    if !admissible
        was_flagged = ct_fofc_flag_is_active(old_flag)
        ct_record_fallback!(fallback_meta,CT_POS_FOFC_BAD_CELL_COUNT)
        if !was_flagged
            @inbounds flags[ii,jj,kk,CT_FOFC_PROPOSAL_CHANNEL] = one(FT)
            ct_record_fallback!(fallback_meta,CT_POS_FOFC_CELL_COUNT)
            ct_record_fallback!(fallback_meta,CT_POS_FOFC_NEW_CELL_COUNT)
        else
            _,_,theta,recoverable = ct_point6_convex_limit(
                hydro_state,magnetic,anchor_hydro_state,anchor_magnetic,
                gamma,minimum_density,minimum_pressure,
            )
            if recoverable
                topology_scale = _ct_fofc_topology_scale(flags,ii,jj,kk)
                new_scale = min(
                    ct_fofc_flag_scale(old_flag),
                    FT(ct_fofc_theta_safety)*theta*topology_scale,
                )
                new_scale = clamp(new_scale,zero(FT),one(FT))
                if new_scale < old_flag
                    @inbounds flags[ii,jj,kk,CT_FOFC_PROPOSAL_CHANNEL] =
                        new_scale
                    ct_record_fallback!(fallback_meta,CT_POS_FOFC_LIMIT_COUNT)
                    ct_record_fallback!(
                        fallback_meta,CT_POS_FOFC_REDUCED_CELL_COUNT,
                    )
                else
                    ct_record_fallback!(
                        fallback_meta,CT_POS_FOFC_UNRECOVERABLE_COUNT,
                    )
                end
            else
                ct_record_fallback!(
                    fallback_meta,CT_POS_FOFC_UNRECOVERABLE_COUNT,
                )
                anchor_state = SVector{9,FT}(
                    anchor_hydro[1],anchor_hydro[2],anchor_hydro[3],
                    anchor_hydro[4],anchor_hydro[5],anchor_magnetic[1],
                    anchor_magnetic[2],anchor_magnetic[3],zero(FT),
                )
                anchor_rho,anchor_kinetic,anchor_magnetic_energy,
                    anchor_internal,anchor_pressure =
                    mhd_raw_thermo(anchor_state,gamma)
                _ct_store_failure!(
                    fallback_meta,fallback_values,CT_POS_SITE_FOFC_ANCHOR,
                    Int32(0),Int32(0),ii,jj,kk,anchor_state,FT(NaN),
                    anchor_rho,anchor_kinetic,anchor_magnetic_energy,
                    anchor_internal,anchor_pressure,
                )
            end
        end
        state = SVector{9,FT}(
            hydro[1],hydro[2],hydro[3],hydro[4],hydro[5],
            magnetic[1],magnetic[2],magnetic[3],zero(FT),
        )
        rho,kinetic,magnetic_energy,internal,pressure =
            mhd_raw_thermo(state,gamma)
        _ct_store_failure!(
            fallback_meta,fallback_values,CT_POS_SITE_FOFC_CLOSURE,
            Int32(0),Int32(0),ii,jj,kk,state,FT(NaN),rho,kinetic,
            magnetic_energy,internal,pressure,
        )
    end
    return
end

function ct_commit_fofc_candidate_kernel!(flags, nxp, nyp, nzp)
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    (i > nxp || j > nyp || k > nzp) && return

    ii, jj, kk = i+Int32(NG), j+Int32(NG), k+Int32(NG)
    @inbounds flags[ii,jj,kk,CT_FOFC_COMMITTED_CHANNEL] =
        flags[ii,jj,kk,CT_FOFC_PROPOSAL_CHANNEL]
    return
end

@inline function _ct_record_pressure_change!(
    meta, values,
    direction::Int32, side::Int32,
    i::Int32, j::Int32, k::Int32,
    before, after, bface,
)
    _, _, _, _, p_before = mhd_raw_thermo(before, γ)
    rho, kinetic, magnetic, internal, p_after = mhd_raw_thermo(after, γ)
    tol = FT(64) * eps(FT) * max(one(FT), abs(p_before))
    changed = !(isfinite(p_before) && isfinite(p_after)) ||
              abs(p_after - p_before) > tol
    if !changed
        return false
    end

    _ct_store_failure!(
        meta, values, CT_POS_SITE_FACE_B_INVARIANT,
        direction, side, i, j, k, after, bface,
        rho, kinetic, magnetic, internal, p_after,
    )
    return true
end

function ct_reset_positivity!(meta, values)
    fill!(meta, Int32(0))
    fill!(values, zero(eltype(values)))
    return nothing
end

function ct_reset_positivity_violation!(meta, values)
    fill!(view(meta, 1:CT_POS_FAILURE_META_LEN), Int32(0))
    fill!(values, zero(eltype(values)))
    return nothing
end

function ct_reset_fofc_iteration!(meta, values)
    ct_reset_positivity_violation!(meta,values)
    fill!(view(meta,CT_POS_FOFC_NEW_CELL_COUNT:CT_POS_FOFC_BAD_CELL_COUNT),
          Int32(0))
    fill!(view(meta,CT_POS_FOFC_REDUCED_CELL_COUNT:
                    CT_POS_FOFC_UNRECOVERABLE_COUNT),Int32(0))
    return nothing
end

function ct_fofc_iteration_counts(meta)
    meta_h = Array(meta)
    return (
        new_cells=Int(meta_h[CT_POS_FOFC_NEW_CELL_COUNT]),
        bad_cells=Int(meta_h[CT_POS_FOFC_BAD_CELL_COUNT]),
        reduced_cells=Int(meta_h[CT_POS_FOFC_REDUCED_CELL_COUNT]),
        unrecoverable_cells=Int(meta_h[CT_POS_FOFC_UNRECOVERABLE_COUNT]),
    )
end

function ct_point6_unrecoverable_count(meta)
    meta_h = Array(meta)
    return Int(meta_h[CT_POS_POINT6_UNRECOVERABLE_COUNT])
end

function ct_positivity_snapshot(meta, values)
    return Array(meta), Array(values)
end

function ct_fallback_counts(meta)
    meta_h = Array(meta)
    return (
        weno_to_plm=Int(meta_h[CT_POS_WENO_TO_PLM_COUNT]),
        plm_to_first=Int(meta_h[CT_POS_PLM_TO_FIRST_COUNT]),
        hlld_to_hlle=Int(meta_h[CT_POS_HLLD_TO_HLLE_COUNT]),
        characteristic_limited=Int(
            meta_h[CT_POS_CHARACTERISTIC_LIMIT_COUNT],
        ),
        point6_to_ao=Int(meta_h[CT_POS_POINT6_TO_AO_COUNT]),
        point6_limited=Int(meta_h[CT_POS_POINT6_LIMIT_COUNT]),
        face_p2a_to_ao=Int(meta_h[CT_POS_FACE_P2A_TO_AO_COUNT]),
        face_p2a_to_midpoint=Int(meta_h[CT_POS_FACE_P2A_TO_MIDPOINT_COUNT]),
        fofc_cells=Int(meta_h[CT_POS_FOFC_CELL_COUNT]),
        fofc_faces=Int(meta_h[CT_POS_FOFC_FACE_COUNT]),
        fofc_limited=Int(meta_h[CT_POS_FOFC_LIMIT_COUNT]),
    )
end

function ct_check_positivity_or_abort!(
    meta, values;
    rank::Integer, block::Integer, step::Integer, rk_stage::Integer,
)
    meta_h, values_h = ct_positivity_snapshot(meta, values)
    if meta_h[1] == Int32(0)
        return false
    end

    site_names = (
        "reconstruction before face-B",
        "face-B pressure invariant",
        "HLLD input",
        "post-hydro/pre-CT cell",
        "post-CT-sync cell",
        "ghost primitive refresh",
        "pre-CT-sync B/energy transition",
        "primitive finalize floor",
        "post-CT-state reconciliation",
        "FOFC corrected candidate",
        "FOFC no-transport anchor",
    )
    site = Int(meta_h[2])
    site_name = 1 <= site <= length(site_names) ? site_names[site] : "unknown"
    message =
        "STRICT CT POSITIVITY VIOLATION: rank=$(rank) block=$(block) " *
        "step=$(step) rk_stage=$(rk_stage) site=$(site)($(site_name)) " *
        "direction=$(meta_h[3]) side=$(meta_h[4]) " *
        "index=($(meta_h[5]),$(meta_h[6]),$(meta_h[7]))\n" *
        "  rho=$(values_h[1]) KE=$(values_h[2]) ME=$(values_h[3]) " *
        "raw_ei=$(values_h[4]) raw_p=$(values_h[5])\n" *
        "  B=($(values_h[6]),$(values_h[7]),$(values_h[8])) " *
        "face_B=$(values_h[9]) U5=$(values_h[10])"
    if site == Int(CT_POS_SITE_POST_CT_RECONCILE)
        message *=
            "\n  old_B=($(values_h[11]),$(values_h[12]),$(values_h[13])) " *
            "old_p=$(values_h[9])"
    elseif site == Int(CT_POS_SITE_FINALIZE_FLOOR)
        message *=
            "\n  rho_floor=$(values_h[11]) p_floor=$(values_h[12]) " *
            "T=$(values_h[13])\n" *
            "  center=($(values_h[14]),$(values_h[15]),$(values_h[16])) " *
            "logical_boundary_distance=$(values_h[17])"
    end
    printstyled(message * "\n", color=:red)
    flush(stdout)
    MPI.Abort(MPI.COMM_WORLD, 86)
    return true
end
