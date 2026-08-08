# Pure CT state invariants shared by reconstruction and strict diagnostics.

if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end

# CT background-field split helper.  The evolved staggered field stores the
# perturbation flux Phi_b; a `nothing` background is the zero-cost legacy path.
# Keeping this as two methods lets GPU compilation specialize away the branch
# in ordinary CT, compressible, and GLM runs.
@inline _ct_total_face_flux(Bface, ::Nothing, i, j, k) =
    @inbounds Bface[i, j, k]

@inline _ct_total_face_flux(Bface, B0face, i, j, k) =
    @inbounds Bface[i, j, k] + B0face[i, j, k]

@inline _ct_background_face_flux(::Nothing, i, j, k) = zero(FT)

@inline _ct_background_face_flux(B0face, i, j, k) =
    @inbounds B0face[i, j, k]

@inline ct_face_index(::Val{6}, i::Int32, j::Int32, k::Int32) =
    (i + Int32(1), j, k)
@inline ct_face_index(::Val{7}, i::Int32, j::Int32, k::Int32) =
    (i, j + Int32(1), k)
@inline ct_face_index(::Val{8}, i::Int32, j::Int32, k::Int32) =
    (i, j, k + Int32(1))

@inline function ct_emf_weight_density(rho)
    @static if strict_ct_positivity
        return rho
    else
        return max(rho, oftype(rho, 1e-10))
    end
end

@inline function ct_emf_weight_from_density_sum(
    density_flux, density_sum, spacing, dt,
)
    half = one(density_flux) / 2
    v_over_c = oftype(density_flux, 1024.0) * dt * density_flux /
               (spacing * density_sum)
    return half + max(-half, min(half, v_over_c))
end

@inline ct_emf_weight(density_flux, rho_left, rho_right, spacing, dt) =
    ct_emf_weight_from_density_sum(
        density_flux, rho_left + rho_right, spacing, dt,
    )

@inline ct_cell_centered_emf(
    ::Val{1}, rho, momentum_x, momentum_y, momentum_z,
    magnetic_x, magnetic_y, magnetic_z,
) = (momentum_z * magnetic_y - momentum_y * magnetic_z) / rho

@inline ct_cell_centered_emf(
    ::Val{2}, rho, momentum_x, momentum_y, momentum_z,
    magnetic_x, magnetic_y, magnetic_z,
) = (momentum_x * magnetic_z - momentum_z * magnetic_x) / rho

@inline ct_cell_centered_emf(
    ::Val{3}, rho, momentum_x, momentum_y, momentum_z,
    magnetic_x, magnetic_y, magnetic_z,
) = (momentum_y * magnetic_x - momentum_x * magnetic_y) / rho

@inline function ct_cell_centered_emf_from_state(U, component, i, j, k)
    @inbounds return ct_cell_centered_emf(
        component,
        ct_emf_weight_density(U[i, j, k, 1]),
        U[i, j, k, 2], U[i, j, k, 3], U[i, j, k, 4],
        U[i, j, k, 6], U[i, j, k, 7], U[i, j, k, 8],
    )
end

@inline function ct_cell_centered_emf_from_state(U, Q, component, i, j, k)
    @inbounds begin
        rho = ct_emf_weight_density(U[i, j, k, 1])
        momentum_x = U[i, j, k, 2]
        momentum_y = U[i, j, k, 3]
        momentum_z = U[i, j, k, 4]
        @static if @isdefined(ct_mode) && ct_mode
            magnetic_x = Q[i, j, k, QBX]
            magnetic_y = Q[i, j, k, QBY]
            magnetic_z = Q[i, j, k, QBZ]
        else
            magnetic_x = U[i, j, k, UBX]
            magnetic_y = U[i, j, k, UBY]
            magnetic_z = U[i, j, k, UBZ]
        end
    end
    return ct_cell_centered_emf(
        component, rho, momentum_x, momentum_y, momentum_z,
        magnetic_x, magnetic_y, magnetic_z,
    )
end

@inline function ct_cell_centered_emf_vector_from_state(U, i, j, k)
    @inbounds begin
        rho = ct_emf_weight_density(U[i, j, k, 1])
        momentum_x = U[i, j, k, 2]
        momentum_y = U[i, j, k, 3]
        momentum_z = U[i, j, k, 4]
        magnetic_x = U[i, j, k, 6]
        magnetic_y = U[i, j, k, 7]
        magnetic_z = U[i, j, k, 8]
    end
    return SVector(
        ct_cell_centered_emf(
            Val(1), rho, momentum_x, momentum_y, momentum_z,
            magnetic_x, magnetic_y, magnetic_z,
        ),
        ct_cell_centered_emf(
            Val(2), rho, momentum_x, momentum_y, momentum_z,
            magnetic_x, magnetic_y, magnetic_z,
        ),
        ct_cell_centered_emf(
            Val(3), rho, momentum_x, momentum_y, momentum_z,
            magnetic_x, magnetic_y, magnetic_z,
        ),
    )
end

@inline function ct_cell_centered_emf_vector_from_state(U, Q, i, j, k)
    @inbounds begin
        rho = ct_emf_weight_density(U[i, j, k, 1])
        momentum_x = U[i, j, k, 2]
        momentum_y = U[i, j, k, 3]
        momentum_z = U[i, j, k, 4]
        @static if @isdefined(ct_mode) && ct_mode
            magnetic_x = Q[i, j, k, QBX]
            magnetic_y = Q[i, j, k, QBY]
            magnetic_z = Q[i, j, k, QBZ]
        else
            magnetic_x = U[i, j, k, UBX]
            magnetic_y = U[i, j, k, UBY]
            magnetic_z = U[i, j, k, UBZ]
        end
    end
    return SVector(
        ct_cell_centered_emf(
            Val(1), rho, momentum_x, momentum_y, momentum_z,
            magnetic_x, magnetic_y, magnetic_z,
        ),
        ct_cell_centered_emf(
            Val(2), rho, momentum_x, momentum_y, momentum_z,
            magnetic_x, magnetic_y, magnetic_z,
        ),
        ct_cell_centered_emf(
            Val(3), rho, momentum_x, momentum_y, momentum_z,
            magnetic_x, magnetic_y, magnetic_z,
        ),
    )
end

@inline function ct_cell_centered_edge_emf_from_state(
    U, i, j, k, tangent_x, tangent_y, tangent_z,
)
    emf = ct_cell_centered_emf_vector_from_state(U, i, j, k)
    return ct_project_cell_emf_to_edge(
        emf, tangent_x, tangent_y, tangent_z,
    )
end

@inline function ct_cell_centered_edge_emf_from_state(
    U, Q, i, j, k, tangent_x, tangent_y, tangent_z,
)
    emf = ct_cell_centered_emf_vector_from_state(U, Q, i, j, k)
    return ct_project_cell_emf_to_edge(
        emf, tangent_x, tangent_y, tangent_z,
    )
end

@inline function ct_face_normal_emf_from_state_pair(
    U,
    left_i, left_j, left_k,
    right_i, right_j, right_k,
    normal_x, normal_y, normal_z,
)
    emf_left = ct_cell_centered_emf_vector_from_state(
        U, left_i, left_j, left_k,
    )
    emf_right = ct_cell_centered_emf_vector_from_state(
        U, right_i, right_j, right_k,
    )
    half = one(normal_x) / 2
    return half * (
        (emf_left[1] + emf_right[1]) * normal_x +
        (emf_left[2] + emf_right[2]) * normal_y +
        (emf_left[3] + emf_right[3]) * normal_z
    )
end

@inline function ct_face_normal_emf_from_state_pair(
    U, Q,
    left_i, left_j, left_k,
    right_i, right_j, right_k,
    normal_x, normal_y, normal_z,
)
    emf_left = ct_cell_centered_emf_vector_from_state(
        U, Q, left_i, left_j, left_k,
    )
    emf_right = ct_cell_centered_emf_vector_from_state(
        U, Q, right_i, right_j, right_k,
    )
    half = one(normal_x) / 2
    return half * (
        (emf_left[1] + emf_right[1]) * normal_x +
        (emf_left[2] + emf_right[2]) * normal_y +
        (emf_left[3] + emf_right[3]) * normal_z
    )
end

@inline function ct_edge_geometry(edge_x, edge_y, edge_z)
    edge_length = sqrt(edge_x^2 + edge_y^2 + edge_z^2)
    return (
        edge_x / edge_length,
        edge_y / edge_length,
        edge_z / edge_length,
        edge_length,
    )
end

@inline function ct_project_face_flux_to_edge_emf(
    flux_bx, flux_by, flux_bz,
    normal_x, normal_y, normal_z,
    tangent_x, tangent_y, tangent_z,
    face_area, face_normal_emf,
)
    cross_x = normal_y * tangent_z - normal_z * tangent_y
    cross_y = normal_z * tangent_x - normal_x * tangent_z
    cross_z = normal_x * tangent_y - normal_y * tangent_x
    normal_tangent = normal_x * tangent_x +
                     normal_y * tangent_y +
                     normal_z * tangent_z
    tangential_emf = (
        cross_x * flux_bx + cross_y * flux_by + cross_z * flux_bz
    ) / face_area
    # For a non-planar quadrilateral, its area normal need not be orthogonal to
    # an individual edge. Complete E dot t with the face-normal EMF component.
    return tangential_emf + normal_tangent * face_normal_emf
end

@inline function ct_project_resistive_face_flux_to_edge_emf(
    flux_bx, flux_by, flux_bz,
    normal_x, normal_y, normal_z,
    tangent_x, tangent_y, tangent_z,
    face_area, face_normal_emf,
)
    cross_x = normal_y * tangent_z - normal_z * tangent_y
    cross_y = normal_z * tangent_x - normal_x * tangent_z
    cross_z = normal_x * tangent_y - normal_y * tangent_x
    normal_tangent = normal_x * tangent_x +
                     normal_y * tangent_y +
                     normal_z * tangent_z
    # The diffusive arrays store Fv_B = eta*(J x n)*A = -n x E_eta*A.
    tangential_emf = -(
        cross_x * flux_bx + cross_y * flux_by + cross_z * flux_bz
    ) / face_area
    return tangential_emf + normal_tangent * face_normal_emf
end

@inline function ct_cached_edge_emf(
    emf_x, emf_y, emf_z, i, j, k,
    tangent_x, tangent_y, tangent_z,
)
    @inbounds return emf_x[i,j,k]*tangent_x +
                     emf_y[i,j,k]*tangent_y +
                     emf_z[i,j,k]*tangent_z
end

@inline function ct_face_normal_cached_emf(
    emf_x, emf_y, emf_z,
    left_i, left_j, left_k,
    right_i, right_j, right_k,
    normal_x, normal_y, normal_z,
)
    half = one(normal_x)/2
    @inbounds return half * (
        (emf_x[left_i,left_j,left_k] + emf_x[right_i,right_j,right_k])*normal_x +
        (emf_y[left_i,left_j,left_k] + emf_y[right_i,right_j,right_k])*normal_y +
        (emf_z[left_i,left_j,left_k] + emf_z[right_i,right_j,right_k])*normal_z
    )
end

@inline function ct_green_gauss_curl_b(
    Q,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    inverse_volume, i, j, k,
)
    T = eltype(Q)
    half = T(0.5)
    @inbounds begin
        bx = Q[i,j,k,7]; by = Q[i,j,k,8]; bz = Q[i,j,k,9]

        bx_il = half*(Q[i-1,j,k,7] + bx); bx_ir = half*(bx + Q[i+1,j,k,7])
        bx_jl = half*(Q[i,j-1,k,7] + bx); bx_jr = half*(bx + Q[i,j+1,k,7])
        bx_kl = half*(Q[i,j,k-1,7] + bx); bx_kr = half*(bx + Q[i,j,k+1,7])
        by_il = half*(Q[i-1,j,k,8] + by); by_ir = half*(by + Q[i+1,j,k,8])
        by_jl = half*(Q[i,j-1,k,8] + by); by_jr = half*(by + Q[i,j+1,k,8])
        by_kl = half*(Q[i,j,k-1,8] + by); by_kr = half*(by + Q[i,j,k+1,8])
        bz_il = half*(Q[i-1,j,k,9] + bz); bz_ir = half*(bz + Q[i+1,j,k,9])
        bz_jl = half*(Q[i,j-1,k,9] + bz); bz_jr = half*(bz + Q[i,j+1,k,9])
        bz_kl = half*(Q[i,j,k-1,9] + bz); bz_kr = half*(bz + Q[i,j,k+1,9])

        ai_l = Areai[i,j,k]; ai_r = Areai[i+1,j,k]
        aj_l = Areaj[i,j,k]; aj_r = Areaj[i,j+1,k]
        ak_l = Areak[i,j,k]; ak_r = Areak[i,j,k+1]

        dbx_dy = inverse_volume * (
            bx_ir*ai_r*nyi[i+1,j,k] - bx_il*ai_l*nyi[i,j,k] +
            bx_jr*aj_r*nyj[i,j+1,k] - bx_jl*aj_l*nyj[i,j,k] +
            bx_kr*ak_r*nyk[i,j,k+1] - bx_kl*ak_l*nyk[i,j,k]
        )
        dbx_dz = inverse_volume * (
            bx_ir*ai_r*nzi[i+1,j,k] - bx_il*ai_l*nzi[i,j,k] +
            bx_jr*aj_r*nzj[i,j+1,k] - bx_jl*aj_l*nzj[i,j,k] +
            bx_kr*ak_r*nzk[i,j,k+1] - bx_kl*ak_l*nzk[i,j,k]
        )
        dby_dx = inverse_volume * (
            by_ir*ai_r*nxi[i+1,j,k] - by_il*ai_l*nxi[i,j,k] +
            by_jr*aj_r*nxj[i,j+1,k] - by_jl*aj_l*nxj[i,j,k] +
            by_kr*ak_r*nxk[i,j,k+1] - by_kl*ak_l*nxk[i,j,k]
        )
        dby_dz = inverse_volume * (
            by_ir*ai_r*nzi[i+1,j,k] - by_il*ai_l*nzi[i,j,k] +
            by_jr*aj_r*nzj[i,j+1,k] - by_jl*aj_l*nzj[i,j,k] +
            by_kr*ak_r*nzk[i,j,k+1] - by_kl*ak_l*nzk[i,j,k]
        )
        dbz_dx = inverse_volume * (
            bz_ir*ai_r*nxi[i+1,j,k] - bz_il*ai_l*nxi[i,j,k] +
            bz_jr*aj_r*nxj[i,j+1,k] - bz_jl*aj_l*nxj[i,j,k] +
            bz_kr*ak_r*nxk[i,j,k+1] - bz_kl*ak_l*nxk[i,j,k]
        )
        dbz_dy = inverse_volume * (
            bz_ir*ai_r*nyi[i+1,j,k] - bz_il*ai_l*nyi[i,j,k] +
            bz_jr*aj_r*nyj[i,j+1,k] - bz_jl*aj_l*nyj[i,j,k] +
            bz_kr*ak_r*nyk[i,j,k+1] - bz_kl*ak_l*nyk[i,j,k]
        )
    end
    return SVector{3,T}(
        dbz_dy - dby_dz,
        dbx_dz - dbz_dx,
        dby_dx - dbx_dy,
    )
end

@inline ct_project_cell_emf_to_edge(emf, tangent_x, tangent_y, tangent_z) =
    emf[1] * tangent_x + emf[2] * tangent_y + emf[3] * tangent_z

# Vol stores inverse physical cell volume. The symmetric face-normal width is
# the average of the two adjacent cell volumes divided by the face area.
@inline function ct_face_normal_spacing(inv_volume_left, inv_volume_right, face_area)
    half = one(inv_volume_left) / 2
    return half * (one(inv_volume_left) / inv_volume_left +
                   one(inv_volume_right) / inv_volume_right) / face_area
end

@inline function ct_xface_geometry_indices(face_i, cell_j, cell_k, ng)
    return (face_i + ng, cell_j + ng - one(cell_j), cell_k + ng - one(cell_k))
end

@inline function ct_yface_geometry_indices(cell_i, face_j, cell_k, ng)
    return (cell_i + ng - one(cell_i), face_j + ng, cell_k + ng - one(cell_k))
end

@inline function ct_zface_geometry_indices(cell_i, cell_j, face_k, ng)
    return (cell_i + ng - one(cell_i), cell_j + ng - one(cell_j), face_k + ng)
end

@inline function ct_resistive_edge_poynting_flux(
    line_a, line_b,
    tangent_ax, tangent_ay, tangent_az,
    tangent_bx, tangent_by, tangent_bz,
    area_x, area_y, area_z,
    magnetic_x, magnetic_y, magnetic_z,
)
    gram_aa = tangent_ax*tangent_ax + tangent_ay*tangent_ay +
              tangent_az*tangent_az
    gram_ab = tangent_ax*tangent_bx + tangent_ay*tangent_by +
              tangent_az*tangent_bz
    gram_bb = tangent_bx*tangent_bx + tangent_by*tangent_by +
              tangent_bz*tangent_bz
    determinant = gram_aa*gram_bb - gram_ab*gram_ab
    determinant > eps(FT)*gram_aa*gram_bb || return FT(NaN)

    # w = A x B lies in the face tangent plane. Expressing w in the two
    # logical tangent vectors lets the line-integrated edge EMFs evaluate E.w.
    wx = area_y*magnetic_z - area_z*magnetic_y
    wy = area_z*magnetic_x - area_x*magnetic_z
    wz = area_x*magnetic_y - area_y*magnetic_x
    rhs_a = tangent_ax*wx + tangent_ay*wy + tangent_az*wz
    rhs_b = tangent_bx*wx + tangent_by*wy + tangent_bz*wz
    coefficient_a = (rhs_a*gram_bb - rhs_b*gram_ab)/determinant
    coefficient_b = (rhs_b*gram_aa - rhs_a*gram_ab)/determinant
    # The edge values represent the resistive electric field E_eta.  In SI
    # units the associated Poynting energy flux is (E_eta x B)/mu0.
    return INV_MU0_SI * (coefficient_a*line_a + coefficient_b*line_b)
end

@inline function ct_periodic_face_source_index(index, ncell, ng, is_normal)
    # A normal face range contains both periodic copies (0 and N), so its
    # ghost range starts one index later than a tangential cell range. Both
    # directions still have physical period N, not N+1.
    high_ghost_start = ncell + ng + (is_normal ? Int32(2) : Int32(1))
    if index <= ng
        return index + ncell
    elseif index >= high_ghost_start
        return index - ncell
    end
    return index
end

@inline function ct_zerograd_face_source_index(index, ncell, ng, is_normal)
    physical_low = ng + one(index)
    physical_high = ng + ncell + (is_normal ? one(index) : zero(index))
    if index < physical_low
        return physical_low
    elseif index > physical_high
        return physical_high
    end
    return index
end

@inline function ct_periodic_edge_source_index(index, ncell, periodic, duplicated)
    if periodic && duplicated && index == ncell + one(index)
        return one(index)
    end
    return index
end

@inline function ct_sg07_edge_emf(
    emf_b_cm, emf_b_c, emf_c_bm, emf_c_b,
    cc_cm_bm, cc_cm_b, cc_c_bm, cc_c_b,
    weight_b_cm, weight_b_c, weight_c_bm, weight_c_b,
)
    one_emf = one(emf_b_cm)
    delta_lc = (one_emf - weight_b_cm) * (emf_c_b - cc_cm_b) +
               weight_b_cm * (emf_c_bm - cc_cm_bm)
    delta_rc = (one_emf - weight_b_c) * (emf_c_b - cc_c_b) +
               weight_b_c * (emf_c_bm - cc_c_bm)
    delta_lb = (one_emf - weight_c_bm) * (emf_b_c - cc_c_bm) +
               weight_c_bm * (emf_b_cm - cc_cm_bm)
    delta_rb = (one_emf - weight_c_b) * (emf_b_c - cc_c_b) +
               weight_c_b * (emf_b_cm - cc_cm_b)
    return (delta_lc + delta_rc + delta_lb + delta_rb +
            emf_b_cm + emf_b_c + emf_c_bm + emf_c_b) / 4
end

# Tangential flux arrays carry one ghost value on each side. These mappings
# select the two face EMFs that meet at an edge without periodic wrap.
@inline function ct_ez_face_flux_indices(edge_i, edge_j, cell_k)
    one_i = one(edge_i)
    one_j = one(edge_j)
    one_k = one(cell_k)
    return (
        ((edge_i, edge_j, cell_k + one_k),
         (edge_i, edge_j + one_j, cell_k + one_k)),
        ((edge_i, edge_j, cell_k + one_k),
         (edge_i + one_i, edge_j, cell_k + one_k)),
    )
end

@inline function ct_ey_face_flux_indices(edge_i, cell_j, edge_k)
    one_i = one(edge_i)
    one_j = one(cell_j)
    one_k = one(edge_k)
    return (
        ((edge_i, cell_j + one_j, edge_k),
         (edge_i, cell_j + one_j, edge_k + one_k)),
        ((edge_i, cell_j + one_j, edge_k),
         (edge_i + one_i, cell_j + one_j, edge_k)),
    )
end

@inline function ct_ex_face_flux_indices(cell_i, edge_j, edge_k)
    one_i = one(cell_i)
    one_j = one(edge_j)
    one_k = one(edge_k)
    return (
        ((cell_i + one_i, edge_j, edge_k),
         (cell_i + one_i, edge_j, edge_k + one_k)),
        ((cell_i + one_i, edge_j, edge_k),
         (cell_i + one_i, edge_j + one_j, edge_k)),
    )
end

@inline function mhd_raw_thermo_components(
    rho, momentum_x, momentum_y, momentum_z, energy,
    magnetic_x, magnetic_y, magnetic_z, gamma,
)
    half = one(rho) / 2
    if !(isfinite(rho) && rho > zero(rho))
        bad = oftype(rho, NaN)
        return (rho, bad, bad, bad, bad)
    end

    momentum2 = momentum_x^2 + momentum_y^2 + momentum_z^2
    kinetic = half * momentum2 / rho
    magnetic = half * (magnetic_x^2 + magnetic_y^2 + magnetic_z^2) * INV_MU0_SI
    internal = energy - kinetic - magnetic
    pressure = mhd_thermodynamic_pressure(rho, internal, gamma)
    return (rho, kinetic, magnetic, internal, pressure)
end

@inline function mhd_raw_thermo(U, gamma)
    return mhd_raw_thermo_components(
        U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8], gamma,
    )
end

@inline function ct_impose_face_b_preserve_p(
    U::SVector{N,T}, ::Val{BI}, bface::T,
) where {N,T,BI}
    bold = U[BI]
    energy = U[5] + T(0.5) * INV_MU0_SI * (bface * bface - bold * bold)
    return setindex(setindex(U, energy, 5), bface, BI)
end

@inline function ct_replace_normal_component(
    magnetic_x, magnetic_y, magnetic_z,
    normal_x, normal_y, normal_z, face_bn,
)
    old_bn = magnetic_x * normal_x + magnetic_y * normal_y +
             magnetic_z * normal_z
    delta_bn = face_bn - old_bn
    return (
        magnetic_x + delta_bn * normal_x,
        magnetic_y + delta_bn * normal_y,
        magnetic_z + delta_bn * normal_z,
    )
end

@inline function ct_constrain_face_flux(
    magnetic_x, magnetic_y, magnetic_z,
    area, normal_x, normal_y, normal_z, face_flux,
)
    projected_flux = area * (
        magnetic_x*normal_x + magnetic_y*normal_y + magnetic_z*normal_z
    )
    if face_flux == projected_flux
        return (magnetic_x, magnetic_y, magnetic_z)
    end
    surface_x = area * normal_x
    surface_y = area * normal_y
    surface_z = area * normal_z
    correction = (face_flux - projected_flux) /
                 (surface_x^2 + surface_y^2 + surface_z^2)
    return (
        magnetic_x + correction*surface_x,
        magnetic_y + correction*surface_y,
        magnetic_z + correction*surface_z,
    )
end

@inline function ct_impose_face_bn_preserve_p(
    U::SVector{N,T}, normal_x::T, normal_y::T, normal_z::T, face_bn::T,
) where {N,T}
    magnetic_x, magnetic_y, magnetic_z = ct_replace_normal_component(
        U[6], U[7], U[8], normal_x, normal_y, normal_z, face_bn,
    )
    old_magnetic2 = U[6]^2 + U[7]^2 + U[8]^2
    new_magnetic2 = magnetic_x^2 + magnetic_y^2 + magnetic_z^2
    energy = U[5] + T(0.5) * INV_MU0_SI * (new_magnetic2 - old_magnetic2)
    return SVector{N,T}(
        ntuple(Val(N)) do n
            n == 5 ? energy :
            n == 6 ? magnetic_x :
            n == 7 ? magnetic_y :
            n == 8 ? magnetic_z : U[n]
        end,
    )
end

@inline function ct_average_to_point6(v::SVector{5,T}) where {T}
    return T(3)/T(640)*v[1] - T(29)/T(480)*v[2] +
           T(1067)/T(960)*v[3] - T(29)/T(480)*v[4] +
           T(3)/T(640)*v[5]
end

@inline function ct_point_to_average6(v::SVector{5,T}) where {T}
    return -T(17)/T(5760)*v[1] + T(77)/T(1440)*v[2] +
           T(863)/T(960)*v[3] + T(77)/T(1440)*v[4] -
           T(17)/T(5760)*v[5]
end

@inline function ct_face_to_center6(v::SVector{6,T}) where {T}
    return (T(3)*v[1] - T(25)*v[2] + T(150)*v[3] +
            T(150)*v[4] - T(25)*v[5] + T(3)*v[6])/T(256)
end

@inline function _ct_average_to_point6_weight(::Type{T}, index) where {T}
    return index == 1 || index == 5 ? T(3)/T(640) :
           (index == 2 || index == 4 ? -T(29)/T(480) : T(1067)/T(960))
end

@inline function _ct_face_point_data6(
    face_flux, area, normal_x, normal_y, normal_z,
    i, j, k, ::Val{DIRECTION},
) where {DIRECTION}
    return _ct_face_point_data6(
        face_flux, nothing, area, normal_x, normal_y, normal_z,
        i, j, k, Val(DIRECTION),
    )
end

@inline function _ct_face_point_data6(
    face_flux, background_flux, area, normal_x, normal_y, normal_z,
    i, j, k, ::Val{DIRECTION},
) where {DIRECTION}
    T = eltype(face_flux)
    flux_point = zero(T)
    area_x_point = zero(T)
    area_y_point = zero(T)
    area_z_point = zero(T)
    for first_index in 1:5
        first_offset = first_index - 3
        first_weight = _ct_average_to_point6_weight(T, first_index)
        for second_index in 1:5
            second_offset = second_index - 3
            weight = first_weight *
                     _ct_average_to_point6_weight(T, second_index)
            ii = DIRECTION == 1 ? i : i + first_offset
            jj = DIRECTION == 1 ? j + first_offset :
                 (DIRECTION == 2 ? j : j + second_offset)
            kk = DIRECTION == 3 ? k : k + second_offset
            @inbounds begin
                local_area = area[ii, jj, kk]
                flux_point += weight * _ct_total_face_flux(
                    face_flux, background_flux, ii, jj, kk,
                )
                area_x_point += weight * local_area * normal_x[ii, jj, kk]
                area_y_point += weight * local_area * normal_y[ii, jj, kk]
                area_z_point += weight * local_area * normal_z[ii, jj, kk]
            end
        end
    end
    return SVector{4,T}(
        flux_point, area_x_point, area_y_point, area_z_point,
    )
end

@inline function ct_solve_cell_b(
    area_i::SVector{3,T}, area_j::SVector{3,T}, area_k::SVector{3,T},
    flux_i::T, flux_j::T, flux_k::T,
) where {T}
    cross_jk = cross(area_j, area_k)
    determinant = dot(area_i, cross_jk)
    scale = max(norm(area_i), norm(area_j), norm(area_k))
    if !(isfinite(determinant) && isfinite(scale) && scale > zero(T) &&
         abs(determinant) > T(64)*eps(T)*scale^3)
        bad = T(NaN)
        return SVector{3,T}(bad, bad, bad)
    end
    return (
        flux_i*cross_jk + flux_j*cross(area_k, area_i) +
        flux_k*cross(area_i, area_j)
    ) / determinant
end

@inline function ct_recover_cell_b_point6(
    Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    i, j, k,
)
    return ct_recover_cell_b_point6(
        Bx_face, By_face, Bz_face,
        nothing, nothing, nothing,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        i, j, k,
    )
end

@inline function ct_recover_cell_b_point6(
    Bx_face, By_face, Bz_face,
    B0x_face, B0y_face, B0z_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    i, j, k,
)
    i_faces = ntuple(Val(6)) do face
        _ct_face_point_data6(
            Bx_face, B0x_face, Areai, nxi, nyi, nzi,
            i + face - 3, j, k, Val(1),
        )
    end
    j_faces = ntuple(Val(6)) do face
        _ct_face_point_data6(
            By_face, B0y_face, Areaj, nxj, nyj, nzj,
            i, j + face - 3, k, Val(2),
        )
    end
    k_faces = ntuple(Val(6)) do face
        _ct_face_point_data6(
            Bz_face, B0z_face, Areak, nxk, nyk, nzk,
            i, j, k + face - 3, Val(3),
        )
    end
    T = eltype(Bx_face)
    centered_i = SVector{4,T}(ntuple(Val(4)) do component
        ct_face_to_center6(SVector{6,T}(ntuple(
            face -> i_faces[face][component], Val(6),
        )))
    end)
    centered_j = SVector{4,T}(ntuple(Val(4)) do component
        ct_face_to_center6(SVector{6,T}(ntuple(
            face -> j_faces[face][component], Val(6),
        )))
    end)
    centered_k = SVector{4,T}(ntuple(Val(4)) do component
        ct_face_to_center6(SVector{6,T}(ntuple(
            face -> k_faces[face][component], Val(6),
        )))
    end)
    return ct_solve_cell_b(
        SVector{3,T}(centered_i[2], centered_i[3], centered_i[4]),
        SVector{3,T}(centered_j[2], centered_j[3], centered_j[4]),
        SVector{3,T}(centered_k[2], centered_k[3], centered_k[4]),
        centered_i[1], centered_j[1], centered_k[1],
    )
end

@inline function ct_conservative_average_to_point6(
    U, inverse_volume, i, j, k,
)
    T = eltype(U)
    jacobian_point = zero(T)
    density_point = zero(T)
    momentum_x_point = zero(T)
    momentum_y_point = zero(T)
    momentum_z_point = zero(T)
    energy_point = zero(T)
    psi_point = zero(T)
    @static if @isdefined(Ncell_cons)
        has_psi = Ncell_cons >= 9
    else
        # Standalone state tests may provide the legacy nine-component U.
        has_psi = size(U, 4) >= 9
    end
    for x_index in 1:5
        di = x_index - 3
        wx = _ct_average_to_point6_weight(T, x_index)
        for y_index in 1:5
            dj = y_index - 3
            wxy = wx * _ct_average_to_point6_weight(T, y_index)
            for z_index in 1:5
                dk = z_index - 3
                weight = wxy * _ct_average_to_point6_weight(T, z_index)
                ii, jj, kk = i+di, j+dj, k+dk
                @inbounds jacobian_average = one(T)/inverse_volume[ii, jj, kk]
                weighted_jacobian = weight * jacobian_average
                jacobian_point += weighted_jacobian
                @inbounds begin
                    density_point += weighted_jacobian * U[ii, jj, kk, 1]
                    momentum_x_point += weighted_jacobian * U[ii, jj, kk, 2]
                    momentum_y_point += weighted_jacobian * U[ii, jj, kk, 3]
                    momentum_z_point += weighted_jacobian * U[ii, jj, kk, 4]
                    energy_point += weighted_jacobian * U[ii, jj, kk, 5]
                    if has_psi
                        psi_point += weighted_jacobian * U[ii, jj, kk, 9]
                    end
                end
            end
        end
    end
    inverse_jacobian = one(T)/jacobian_point
    return SVector{6,T}(
        density_point*inverse_jacobian,
        momentum_x_point*inverse_jacobian,
        momentum_y_point*inverse_jacobian,
        momentum_z_point*inverse_jacobian,
        energy_point*inverse_jacobian,
        psi_point*inverse_jacobian,
    )
end

@inline function ct_mhd_point_conservative_to_primitive(
    hydro::SVector{6,T}, magnetic::SVector{3,T}, gamma::T,
) where {T}
    rho = hydro[1]
    if !(isfinite(rho) && rho > zero(T))
        bad = T(NaN)
        return SVector{9,T}(ntuple(_ -> bad, Val(9)))
    end
    inverse_rho = one(T)/rho
    velocity_x = hydro[2]*inverse_rho
    velocity_y = hydro[3]*inverse_rho
    velocity_z = hydro[4]*inverse_rho
    kinetic = T(0.5)*(
        hydro[2]^2 + hydro[3]^2 + hydro[4]^2
    )*inverse_rho
    magnetic_energy = T(0.5)*INV_MU0_SI*dot(magnetic, magnetic)
    pressure = mhd_thermodynamic_pressure(
        rho, hydro[5]-kinetic-magnetic_energy, gamma,
    )
    return SVector{9,T}(
        rho, velocity_x, velocity_y, velocity_z, pressure,
        magnetic[1], magnetic[2], magnetic[3], hydro[6],
    )
end

@inline function ct_recover_cell_b(
    area_i_lo::SVector{3,T}, area_i_hi::SVector{3,T},
    area_j_lo::SVector{3,T}, area_j_hi::SVector{3,T},
    area_k_lo::SVector{3,T}, area_k_hi::SVector{3,T},
    flux_i_lo::T, flux_i_hi::T,
    flux_j_lo::T, flux_j_hi::T,
    flux_k_lo::T, flux_k_hi::T,
) where {T}
    s1, s2 = area_i_lo, area_i_hi
    s3, s4 = area_j_lo, area_j_hi
    s5, s6 = area_k_lo, area_k_hi

    mxx = s1[1]^2 + s2[1]^2 + s3[1]^2 + s4[1]^2 + s5[1]^2 + s6[1]^2
    myy = s1[2]^2 + s2[2]^2 + s3[2]^2 + s4[2]^2 + s5[2]^2 + s6[2]^2
    mzz = s1[3]^2 + s2[3]^2 + s3[3]^2 + s4[3]^2 + s5[3]^2 + s6[3]^2
    mxy = s1[1]*s1[2] + s2[1]*s2[2] + s3[1]*s3[2] +
          s4[1]*s4[2] + s5[1]*s5[2] + s6[1]*s6[2]
    mxz = s1[1]*s1[3] + s2[1]*s2[3] + s3[1]*s3[3] +
          s4[1]*s4[3] + s5[1]*s5[3] + s6[1]*s6[3]
    myz = s1[2]*s1[3] + s2[2]*s2[3] + s3[2]*s3[3] +
          s4[2]*s4[3] + s5[2]*s5[3] + s6[2]*s6[3]

    rx = s1[1]*flux_i_lo + s2[1]*flux_i_hi +
         s3[1]*flux_j_lo + s4[1]*flux_j_hi +
         s5[1]*flux_k_lo + s6[1]*flux_k_hi
    ry = s1[2]*flux_i_lo + s2[2]*flux_i_hi +
         s3[2]*flux_j_lo + s4[2]*flux_j_hi +
         s5[2]*flux_k_lo + s6[2]*flux_k_hi
    rz = s1[3]*flux_i_lo + s2[3]*flux_i_hi +
         s3[3]*flux_j_lo + s4[3]*flux_j_hi +
         s5[3]*flux_k_lo + s6[3]*flux_k_hi

    cxx = myy*mzz - myz*myz
    cxy = mxz*myz - mxy*mzz
    cxz = mxy*myz - mxz*myy
    cyy = mxx*mzz - mxz*mxz
    cyz = mxy*mxz - mxx*myz
    czz = mxx*myy - mxy*mxy
    determinant = mxx*cxx + mxy*cxy + mxz*cxz
    scale = max(abs(mxx), abs(myy), abs(mzz))
    if !(isfinite(determinant) &&
         abs(determinant) > T(64) * eps(T) * scale^3)
        bad = T(NaN)
        return SVector{3,T}(bad, bad, bad)
    end
    inverse_determinant = one(T) / determinant
    return SVector{3,T}(
        (cxx*rx + cxy*ry + cxz*rz) * inverse_determinant,
        (cxy*rx + cyy*ry + cyz*rz) * inverse_determinant,
        (cxz*rx + cyz*ry + czz*rz) * inverse_determinant,
    )
end

@inline ct_stokes_update_i(phi, ej_lo, ej_hi, ek_lo, ek_hi, dt) =
    phi - dt * ((ek_hi - ek_lo) - (ej_hi - ej_lo))

@inline ct_stokes_update_j(phi, ei_lo, ei_hi, ek_lo, ek_hi, dt) =
    phi - dt * ((ei_hi - ei_lo) - (ek_hi - ek_lo))

@inline ct_stokes_update_k(phi, ei_lo, ei_hi, ej_lo, ej_hi, dt) =
    phi - dt * ((ej_hi - ej_lo) - (ei_hi - ei_lo))

function ct_face_fluxes_from_edge_integrals(edge_x, edge_y, edge_z)
    ni = size(edge_y, 1)
    nj = size(edge_x, 2)
    nk = size(edge_x, 3)
    size(edge_x) == (ni - 1, nj, nk) || throw(DimensionMismatch(
        "x-edge integrals must have size (ni-1,nj,nk)",
    ))
    size(edge_y) == (ni, nj - 1, nk) || throw(DimensionMismatch(
        "y-edge integrals must have size (ni,nj-1,nk)",
    ))
    size(edge_z) == (ni, nj, nk - 1) || throw(DimensionMismatch(
        "z-edge integrals must have size (ni,nj,nk-1)",
    ))
    T = promote_type(eltype(edge_x), eltype(edge_y), eltype(edge_z))
    phi_x = Array{T,3}(undef, ni, nj - 1, nk - 1)
    phi_y = Array{T,3}(undef, ni - 1, nj, nk - 1)
    phi_z = Array{T,3}(undef, ni - 1, nj - 1, nk)
    @inbounds for k in 1:nk-1, j in 1:nj-1, i in 1:ni
        phi_x[i,j,k] = edge_y[i,j,k] + edge_z[i,j+1,k] -
                       edge_y[i,j,k+1] - edge_z[i,j,k]
    end
    @inbounds for k in 1:nk-1, j in 1:nj, i in 1:ni-1
        phi_y[i,j,k] = edge_z[i,j,k] + edge_x[i,j,k+1] -
                       edge_z[i+1,j,k] - edge_x[i,j,k]
    end
    @inbounds for k in 1:nk, j in 1:nj-1, i in 1:ni-1
        phi_z[i,j,k] = edge_x[i,j,k] + edge_y[i+1,j,k] -
                       edge_x[i,j+1,k] - edge_y[i,j,k]
    end
    return phi_x, phi_y, phi_z
end

function ct_face_flux_divergence(phi_x, phi_y, phi_z)
    nx, ny, nz = size(phi_z, 1), size(phi_x, 2), size(phi_x, 3)
    size(phi_x) == (nx + 1, ny, nz) || throw(DimensionMismatch(
        "x-face fluxes must have size (nx+1,ny,nz)",
    ))
    size(phi_y) == (nx, ny + 1, nz) || throw(DimensionMismatch(
        "y-face fluxes must have size (nx,ny+1,nz)",
    ))
    size(phi_z) == (nx, ny, nz + 1) || throw(DimensionMismatch(
        "z-face fluxes must have size (nx,ny,nz+1)",
    ))
    T = promote_type(eltype(phi_x), eltype(phi_y), eltype(phi_z))
    divergence = Array{T,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        divergence[i,j,k] =
            phi_x[i+1,j,k] - phi_x[i,j,k] +
            phi_y[i,j+1,k] - phi_y[i,j,k] +
            phi_z[i,j,k+1] - phi_z[i,j,k]
    end
    return divergence
end

function ct_relative_face_divergence(phi_x, phi_y, phi_z)
    divergence = ct_face_flux_divergence(phi_x, phi_y, phi_z)
    nx, ny, nz = size(divergence)
    maximum_relative = zero(eltype(divergence))
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        local_flux =
            abs(phi_x[i+1,j,k]) + abs(phi_x[i,j,k]) +
            abs(phi_y[i,j+1,k]) + abs(phi_y[i,j,k]) +
            abs(phi_z[i,j,k+1]) + abs(phi_z[i,j,k])
        scale = max(local_flux, one(local_flux))
        maximum_relative = max(
            maximum_relative, abs(divergence[i,j,k]) / scale,
        )
    end
    return maximum_relative
end

function _ct_negative_graph_laplacian!(output, field, periodic)
    nx, ny, nz = size(field)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        center = field[i,j,k]
        value = zero(center)
        if i > 1
            value += center - field[i-1,j,k]
        elseif periodic[1]
            value += center - field[nx,j,k]
        end
        if i < nx
            value += center - field[i+1,j,k]
        elseif periodic[1]
            value += center - field[1,j,k]
        end
        if j > 1
            value += center - field[i,j-1,k]
        elseif periodic[2]
            value += center - field[i,ny,k]
        end
        if j < ny
            value += center - field[i,j+1,k]
        elseif periodic[2]
            value += center - field[i,1,k]
        end
        if k > 1
            value += center - field[i,j,k-1]
        elseif periodic[3]
            value += center - field[i,j,nz]
        end
        if k < nz
            value += center - field[i,j,k+1]
        elseif periodic[3]
            value += center - field[i,j,1]
        end
        output[i,j,k] = value
    end
    return output
end

function ct_project_face_flux_divergence!(
    phi_x, phi_y, phi_z;
    periodic=(false, false, false), tolerance=1.0e-12, max_iterations=10000,
)
    nx, ny, nz = size(phi_z, 1), size(phi_x, 2), size(phi_x, 3)
    if periodic[1]
        phi_x[1,:,:] .= (phi_x[1,:,:] .+ phi_x[end,:,:]) ./ 2
        phi_x[end,:,:] .= phi_x[1,:,:]
    end
    if periodic[2]
        phi_y[:,1,:] .= (phi_y[:,1,:] .+ phi_y[:,end,:]) ./ 2
        phi_y[:,end,:] .= phi_y[:,1,:]
    end
    if periodic[3]
        phi_z[:,:,1] .= (phi_z[:,:,1] .+ phi_z[:,:,end]) ./ 2
        phi_z[:,:,end] .= phi_z[:,:,1]
    end

    divergence = ct_face_flux_divergence(phi_x, phi_y, phi_z)
    flux_scale = max(
        maximum(abs, phi_x), maximum(abs, phi_y), maximum(abs, phi_z),
        one(eltype(divergence)),
    )
    compatibility = abs(sum(divergence)) / (length(divergence) * flux_scale)
    compatibility <= tolerance || error(
        "CT face projection cannot preserve boundary-normal flux: " *
        "relative net magnetic flux is $compatibility",
    )

    rhs = -divergence
    rhs .-= sum(rhs) / length(rhs)
    potential = zeros(eltype(rhs), nx, ny, nz)
    residual = copy(rhs)
    direction = copy(residual)
    operator_direction = similar(residual)
    residual_norm2 = sum(abs2, residual)
    solve_tolerance = tolerance / 10
    target_norm2 = max(
        solve_tolerance^2 * max(sum(abs2, rhs), one(residual_norm2)),
        eps(eltype(rhs))^2,
    )
    iterations = 0
    while residual_norm2 > target_norm2 && iterations < max_iterations
        iterations += 1
        _ct_negative_graph_laplacian!(
            operator_direction, direction, periodic,
        )
        denominator = sum(direction .* operator_direction)
        abs(denominator) > floatmin(eltype(rhs)) || break
        alpha = residual_norm2 / denominator
        potential .+= alpha .* direction
        residual .-= alpha .* operator_direction
        residual .-= sum(residual) / length(residual)
        new_norm2 = sum(abs2, residual)
        beta = new_norm2 / residual_norm2
        direction .= residual .+ beta .* direction
        direction .-= sum(direction) / length(direction)
        residual_norm2 = new_norm2
    end
    residual_norm2 <= target_norm2 || error(
        "CT face projection did not converge in $iterations iterations; " *
        "residual=$(sqrt(residual_norm2))",
    )

    @inbounds for k in 1:nz, j in 1:ny, i in 2:nx
        phi_x[i,j,k] -= potential[i,j,k] - potential[i-1,j,k]
    end
    if periodic[1]
        correction = @view(potential[1,:,:]) .- @view(potential[nx,:,:])
        phi_x[1,:,:] .-= correction
        phi_x[end,:,:] .= phi_x[1,:,:]
    end
    @inbounds for k in 1:nz, j in 2:ny, i in 1:nx
        phi_y[i,j,k] -= potential[i,j,k] - potential[i,j-1,k]
    end
    if periodic[2]
        correction = @view(potential[:,1,:]) .- @view(potential[:,ny,:])
        phi_y[:,1,:] .-= correction
        phi_y[:,end,:] .= phi_y[:,1,:]
    end
    @inbounds for k in 2:nz, j in 1:ny, i in 1:nx
        phi_z[i,j,k] -= potential[i,j,k] - potential[i,j,k-1]
    end
    if periodic[3]
        correction = @view(potential[:,:,1]) .- @view(potential[:,:,nz])
        phi_z[:,:,1] .-= correction
        phi_z[:,:,end] .= phi_z[:,:,1]
    end
    return (
        iterations=iterations,
        relative_divergence=ct_relative_face_divergence(phi_x, phi_y, phi_z),
    )
end

# Q stores temperature at slot 6, while the nine reconstructed MHD variables
# use slot 6 for Bx. Skip temperature when reading a primitive stencil.
@inline ct_primitive_reconstruction_slot(n::Integer) = n <= 5 ? n : n + 1

@inline function ct_vanleer_slope(qm, qc, qp)
    delta_left = qc - qm
    delta_right = qp - qc
    product = delta_left * delta_right
    if product <= zero(product)
        return zero(product)
    end
    return 2 * product / (delta_left + delta_right)
end

@inline ct_plm_plus(qm, qc, qp) =
    qc + ct_vanleer_slope(qm, qc, qp) / 2

@inline ct_plm_minus(qm, qc, qp) =
    qc - ct_vanleer_slope(qm, qc, qp) / 2

@inline function ct_primitive_state(Q, i, j, k)
    @inbounds return SVector{9,FT}(
        Q[i,j,k,1], Q[i,j,k,2], Q[i,j,k,3], Q[i,j,k,4],
        Q[i,j,k,5], Q[i,j,k,7], Q[i,j,k,8], Q[i,j,k,9],
        Q[i,j,k,10],
    )
end

@inline function _ct_mhd_nan_basis(::Type{T}) where {T}
    nan = T(NaN)
    return (
        rho=nan, sqrtd=nan, isqrtd=nan, asq=nan, sound=nan,
        cf=nan, cs=nan, beta2=nan, beta3=nan,
        alpha_f=nan, alpha_s=nan, bsign=nan,
    )
end

# Seven-wave primitive-variable eigensystem from Athena (2008), Appendix A.
# The primitive order is (rho, vn, vt1, vt2, p, Bt1, Bt2); Bn is supplied
# separately by CT and is not reconstructed.
@inline function ct_mhd_characteristic_basis(W::SVector{7,T}, bn::T, gamma::T) where {T}
    rho = W[1]
    pressure = W[5]
    if !(isfinite(rho) && rho > zero(T) &&
         isfinite(pressure) && pressure > zero(T) && isfinite(bn))
        return _ct_mhd_nan_basis(T)
    end

    # Athena's primitive eigensystem is written for the normalized field
    # b_hat = B/sqrt(mu0). Keep the public CT state in Tesla and convert only
    # inside the characteristic algebra.
    bt1 = W[6] * INV_SQRT_MU0_SI
    bt2 = W[7] * INV_SQRT_MU0_SI
    bn_hat = bn * INV_SQRT_MU0_SI
    btsq = bt1*bt1 + bt2*bt2
    bnsq = bn_hat*bn_hat
    gamma_pressure = gamma*pressure
    tdif = bnsq + btsq - gamma_pressure
    separation = sqrt(tdif*tdif + T(4)*gamma_pressure*btsq)
    cf_numerator = T(0.5)*(bnsq + btsq + gamma_pressure + separation)
    if !(isfinite(cf_numerator) && cf_numerator > zero(T))
        return _ct_mhd_nan_basis(T)
    end

    invrho = one(T)/rho
    cfsq = cf_numerator*invrho
    cssq = gamma_pressure*bnsq*invrho/cf_numerator
    asq = gamma_pressure*invrho
    if !(isfinite(cfsq) && cfsq >= zero(T) &&
         isfinite(cssq) && cssq >= zero(T) &&
         isfinite(asq) && asq > zero(T))
        return _ct_mhd_nan_basis(T)
    end

    cf = sqrt(cfsq)
    cs = sqrt(cssq)
    sound = sqrt(asq)
    bt = sqrt(btsq)
    if bt > zero(T)
        beta2 = bt1/bt
        beta3 = bt2/bt
    else
        beta2 = one(T)
        beta3 = zero(T)
    end

    if cfsq - cssq <= zero(T)
        alpha_f = one(T)
        alpha_s = zero(T)
    elseif asq - cssq <= zero(T)
        alpha_f = zero(T)
        alpha_s = one(T)
    elseif cfsq - asq <= zero(T)
        alpha_f = one(T)
        alpha_s = zero(T)
    else
        inv_separation = one(T)/(cfsq - cssq)
        alpha_f = sqrt((asq - cssq)*inv_separation)
        alpha_s = sqrt((cfsq - asq)*inv_separation)
    end

    sqrtd = sqrt(rho)
    return (
        rho=rho, sqrtd=sqrtd, isqrtd=one(T)/sqrtd,
        asq=asq, sound=sound, cf=cf, cs=cs,
        beta2=beta2, beta3=beta3,
        alpha_f=alpha_f, alpha_s=alpha_s,
        bsign=bn >= zero(T) ? one(T) : -one(T),
    )
end

@inline function ct_mhd_primitive_to_characteristic(basis, delta::SVector{7,T}) where {T}
    nf = T(0.5)/basis.asq
    qf = nf*basis.cf*basis.alpha_f*basis.bsign
    qs = nf*basis.cs*basis.alpha_s*basis.bsign
    af_prime = T(0.5)*basis.alpha_f/(basis.sound*basis.sqrtd)
    as_prime = T(0.5)*basis.alpha_s/(basis.sound*basis.sqrtd)
    velocity_beta = basis.beta2*delta[3] + basis.beta3*delta[4]
    magnetic_beta = INV_SQRT_MU0_SI * (
        basis.beta2*delta[6] + basis.beta3*delta[7]
    )
    pressure_over_rho = delta[5]/basis.rho

    return SVector{7,T}(
        nf*basis.alpha_f*(pressure_over_rho - basis.cf*delta[2]) +
            qs*velocity_beta + as_prime*magnetic_beta,
        T(0.5)*(
            basis.beta2*(delta[7]*INV_SQRT_MU0_SI*basis.bsign*basis.isqrtd + delta[4]) -
            basis.beta3*(delta[6]*INV_SQRT_MU0_SI*basis.bsign*basis.isqrtd + delta[3])
        ),
        nf*basis.alpha_s*(pressure_over_rho - basis.cs*delta[2]) -
            qf*velocity_beta - af_prime*magnetic_beta,
        delta[1] - delta[5]/basis.asq,
        nf*basis.alpha_s*(pressure_over_rho + basis.cs*delta[2]) +
            qf*velocity_beta - af_prime*magnetic_beta,
        T(0.5)*(
            basis.beta2*(delta[7]*INV_SQRT_MU0_SI*basis.bsign*basis.isqrtd - delta[4]) -
            basis.beta3*(delta[6]*INV_SQRT_MU0_SI*basis.bsign*basis.isqrtd - delta[3])
        ),
        nf*basis.alpha_f*(pressure_over_rho + basis.cf*delta[2]) -
            qs*velocity_beta + as_prime*magnetic_beta,
    )
end

@inline function ct_mhd_characteristic_to_primitive(basis, characteristic::SVector{7,T}) where {T}
    fast_sum = characteristic[1] + characteristic[7]
    slow_sum = characteristic[3] + characteristic[5]
    qf = basis.cf*basis.alpha_f*basis.bsign
    qs = basis.cs*basis.alpha_s*basis.bsign
    af = basis.sound*basis.alpha_f*basis.sqrtd
    as = basis.sound*basis.alpha_s*basis.sqrtd
    velocity_beta = qs*(characteristic[1] - characteristic[7]) +
        qf*(characteristic[5] - characteristic[3])
    magnetic_beta = as*fast_sum - af*slow_sum

    return SVector{7,T}(
        basis.rho*(basis.alpha_f*fast_sum + basis.alpha_s*slow_sum) +
            characteristic[4],
        basis.cf*basis.alpha_f*(characteristic[7] - characteristic[1]) +
            basis.cs*basis.alpha_s*(characteristic[5] - characteristic[3]),
        basis.beta2*velocity_beta +
            basis.beta3*(characteristic[6] - characteristic[2]),
        basis.beta3*velocity_beta +
            basis.beta2*(characteristic[2] - characteristic[6]),
        basis.rho*basis.asq*(
            basis.alpha_f*fast_sum + basis.alpha_s*slow_sum
        ),
        SQRT_MU0_SI * (
            basis.beta2*magnetic_beta - basis.beta3*basis.bsign*basis.sqrtd*(
                characteristic[6] + characteristic[2]
            )
        ),
        SQRT_MU0_SI * (
            basis.beta3*magnetic_beta + basis.beta2*basis.bsign*basis.sqrtd*(
                characteristic[6] + characteristic[2]
            )
        ),
    )
end

@inline function ct_mhd_local_frame(nx::T, ny::T, nz::T) where {T}
    normal_norm = sqrt(nx*nx + ny*ny + nz*nz)
    if !(isfinite(normal_norm) && normal_norm > zero(T))
        nan = T(NaN)
        return SVector{9,T}(ntuple(_ -> nan, Val(9)))
    end
    normal_x = nx/normal_norm
    normal_y = ny/normal_norm
    normal_z = nz/normal_norm
    if abs(normal_x) > abs(normal_z)
        tangent_norm = sqrt(normal_x*normal_x + normal_y*normal_y)
        tangent1_x = -normal_y/tangent_norm
        tangent1_y = normal_x/tangent_norm
        tangent1_z = zero(T)
    else
        tangent_norm = sqrt(normal_y*normal_y + normal_z*normal_z)
        tangent1_x = zero(T)
        tangent1_y = -normal_z/tangent_norm
        tangent1_z = normal_y/tangent_norm
    end
    tangent2_x = normal_y*tangent1_z - normal_z*tangent1_y
    tangent2_y = normal_z*tangent1_x - normal_x*tangent1_z
    tangent2_z = normal_x*tangent1_y - normal_y*tangent1_x
    return SVector{9,T}(
        normal_x, normal_y, normal_z,
        tangent1_x, tangent1_y, tangent1_z,
        tangent2_x, tangent2_y, tangent2_z,
    )
end

@inline function ct_mhd_global_to_local(W::SVector{9,T}, frame::SVector{9,T}) where {T}
    return SVector{7,T}(
        W[1],
        W[2]*frame[1] + W[3]*frame[2] + W[4]*frame[3],
        W[2]*frame[4] + W[3]*frame[5] + W[4]*frame[6],
        W[2]*frame[7] + W[3]*frame[8] + W[4]*frame[9],
        W[5],
        W[6]*frame[4] + W[7]*frame[5] + W[8]*frame[6],
        W[6]*frame[7] + W[7]*frame[8] + W[8]*frame[9],
    )
end

@inline function ct_mhd_local_to_global(
    W::SVector{7,T}, psi::T, bn::T, frame::SVector{9,T},
) where {T}
    velocity_x = W[2]*frame[1] + W[3]*frame[4] + W[4]*frame[7]
    velocity_y = W[2]*frame[2] + W[3]*frame[5] + W[4]*frame[8]
    velocity_z = W[2]*frame[3] + W[3]*frame[6] + W[4]*frame[9]
    magnetic_x = bn*frame[1] + W[6]*frame[4] + W[7]*frame[7]
    magnetic_y = bn*frame[2] + W[6]*frame[5] + W[7]*frame[8]
    magnetic_z = bn*frame[3] + W[6]*frame[6] + W[7]*frame[9]
    return SVector{9,T}(
        W[1], velocity_x, velocity_y, velocity_z, W[5],
        magnetic_x, magnetic_y, magnetic_z, psi,
    )
end

@inline function ct_vanleer_limited_delta(delta_left, delta_right)
    product = delta_left*delta_right
    if product <= zero(product)
        return zero(product)
    end
    return 2*product/(delta_left + delta_right)
end

@inline function ct_mhd_characteristic_plm(
    Wm::SVector{9,T}, Wc::SVector{9,T}, Wp::SVector{9,T},
    nx::T, ny::T, nz::T, bn::T, gamma::T, side::T,
) where {T}
    frame = ct_mhd_local_frame(nx, ny, nz)
    local_m = ct_mhd_global_to_local(Wm, frame)
    local_c = ct_mhd_global_to_local(Wc, frame)
    local_p = ct_mhd_global_to_local(Wp, frame)
    basis = ct_mhd_characteristic_basis(local_c, bn, gamma)
    characteristic_left = ct_mhd_primitive_to_characteristic(
        basis, local_c - local_m,
    )
    characteristic_right = ct_mhd_primitive_to_characteristic(
        basis, local_p - local_c,
    )
    limited = SVector{7,T}(ntuple(Val(7)) do n
        ct_vanleer_limited_delta(
            characteristic_left[n], characteristic_right[n],
        )
    end)
    primitive_slope = ct_mhd_characteristic_to_primitive(basis, limited)
    local_face = local_c + side*T(0.5)*primitive_slope
    psi_slope = ct_vanleer_slope(Wm[9], Wc[9], Wp[9])
    return ct_mhd_local_to_global(
        local_face, Wc[9] + side*T(0.5)*psi_slope, bn, frame,
    )
end

@inline ct_mhd_characteristic_plm_plus(
    Wm::SVector{9,T}, Wc::SVector{9,T}, Wp::SVector{9,T},
    nx::T, ny::T, nz::T, bn::T, gamma::T,
) where {T} = ct_mhd_characteristic_plm(
    Wm, Wc, Wp, nx, ny, nz, bn, gamma, one(T),
)

@inline ct_mhd_characteristic_plm_minus(
    Wm::SVector{9,T}, Wc::SVector{9,T}, Wp::SVector{9,T},
    nx::T, ny::T, nz::T, bn::T, gamma::T,
) where {T} = ct_mhd_characteristic_plm(
    Wm, Wc, Wp, nx, ny, nz, bn, gamma, -one(T),
)

@inline function _ct_mhd_characteristic_weno7(
    stencil::NTuple{7,SVector{9,T}},
    reference_left::SVector{9,T}, reference_right::SVector{9,T},
    nx::T, ny::T, nz::T, bn::T, gamma::T, ::Val{SIDE},
) where {T,SIDE}
    frame = ct_mhd_local_frame(nx, ny, nz)
    local_reference = T(0.5) * (
        ct_mhd_global_to_local(reference_left, frame) +
        ct_mhd_global_to_local(reference_right, frame)
    )
    basis = ct_mhd_characteristic_basis(local_reference, bn, gamma)
    characteristic_stencil = ntuple(Val(7)) do s
        local_state = ct_mhd_global_to_local(stencil[s], frame)
        ct_mhd_primitive_to_characteristic(
            basis, local_state - local_reference,
        )
    end
    characteristic_face = SVector{7,T}(ntuple(Val(7)) do wave
        values = SVector{7,T}(ntuple(Val(7)) do s
            characteristic_stencil[s][wave]
        end)
        SIDE == 1 ? weno7_point_face_left(values, one(T)) :
                    weno7_point_face_right(values, one(T))
    end)
    psi_values = SVector{7,T}(ntuple(s -> stencil[s][9], Val(7)))
    psi_face = SIDE == 1 ? weno7_point_face_left(psi_values, one(T)) :
                           weno7_point_face_right(psi_values, one(T))
    local_face = local_reference +
                 ct_mhd_characteristic_to_primitive(
                     basis, characteristic_face,
                 )
    return ct_mhd_local_to_global(local_face, psi_face, bn, frame)
end

@inline function ct_mhd_characteristic_weno7_left(
    W1::SVector{9,T}, W2::SVector{9,T}, W3::SVector{9,T},
    W4::SVector{9,T}, W5::SVector{9,T}, W6::SVector{9,T},
    W7::SVector{9,T}, nx::T, ny::T, nz::T, bn::T, gamma::T,
) where {T}
    return _ct_mhd_characteristic_weno7(
        (W1, W2, W3, W4, W5, W6, W7), W4, W5,
        nx, ny, nz, bn, gamma, Val(1),
    )
end

@inline function ct_mhd_characteristic_weno7_right(
    W1::SVector{9,T}, W2::SVector{9,T}, W3::SVector{9,T},
    W4::SVector{9,T}, W5::SVector{9,T}, W6::SVector{9,T},
    W7::SVector{9,T}, nx::T, ny::T, nz::T, bn::T, gamma::T,
) where {T}
    return _ct_mhd_characteristic_weno7(
        (W1, W2, W3, W4, W5, W6, W7), W3, W4,
        nx, ny, nz, bn, gamma, Val(2),
    )
end

@inline function ct_mhd_characteristic_weno7_interface_primitives(
    Wim3::SVector{9,T}, Wim2::SVector{9,T}, Wim1::SVector{9,T},
    Wi::SVector{9,T}, Wip1::SVector{9,T}, Wip2::SVector{9,T},
    Wip3::SVector{9,T}, Wip4::SVector{9,T},
    nx::T, ny::T, nz::T, bn::T, gamma::T,
) where {T}
    left = ct_mhd_characteristic_weno7_left(
        Wim3, Wim2, Wim1, Wi, Wip1, Wip2, Wip3,
        nx, ny, nz, bn, gamma,
    )
    right = ct_mhd_characteristic_weno7_right(
        Wim2, Wim1, Wi, Wip1, Wip2, Wip3, Wip4,
        nx, ny, nz, bn, gamma,
    )
    return left, right
end

@inline function ct_mhd_characteristic_weno7_interface_states(
    Wim3::SVector{9,T}, Wim2::SVector{9,T}, Wim1::SVector{9,T},
    Wi::SVector{9,T}, Wip1::SVector{9,T}, Wip2::SVector{9,T},
    Wip3::SVector{9,T}, Wip4::SVector{9,T},
    nx::T, ny::T, nz::T, bn::T, gamma::T,
) where {T}
    left, right = ct_mhd_characteristic_weno7_interface_primitives(
        Wim3, Wim2, Wim1, Wi, Wip1, Wip2, Wip3, Wip4,
        nx, ny, nz, bn, gamma,
    )
    return (
        ct_primitive_to_conservative(left, nx, ny, nz, bn, gamma),
        ct_primitive_to_conservative(right, nx, ny, nz, bn, gamma),
    )
end

@inline function ct_primitive_to_conservative(
    W::SVector{9,T}, ::Val{BI}, bface::T, gamma::T,
) where {T,BI}
    rho = W[1]
    velocity_x = W[2]
    velocity_y = W[3]
    velocity_z = W[4]
    pressure = W[5]
    magnetic_x = BI == 6 ? bface : W[6]
    magnetic_y = BI == 7 ? bface : W[7]
    magnetic_z = BI == 8 ? bface : W[8]
    kinetic = T(0.5) * rho * (
        velocity_x^2 + velocity_y^2 + velocity_z^2
    )
    magnetic = T(0.5) * INV_MU0_SI * (
        magnetic_x^2 + magnetic_y^2 + magnetic_z^2
    )
    energy = isothermal_mhd ? kinetic + magnetic + pressure :
        pressure / (gamma - one(T)) + kinetic + magnetic
    return SVector{9,T}(
        rho,
        rho * velocity_x,
        rho * velocity_y,
        rho * velocity_z,
        energy,
        magnetic_x,
        magnetic_y,
        magnetic_z,
        W[9],
    )
end

@inline function ct_mhd_characteristic_interface_states(
    Wm::SVector{9,T}, Wc::SVector{9,T},
    Wp::SVector{9,T}, Wpp::SVector{9,T},
    normal_x::T, normal_y::T, normal_z::T,
    face_bn::T, gamma::T,
) where {T}
    left_primitive = ct_mhd_characteristic_plm_plus(
        Wm, Wc, Wp, normal_x, normal_y, normal_z, face_bn, gamma,
    )
    right_primitive = ct_mhd_characteristic_plm_minus(
        Wc, Wp, Wpp, normal_x, normal_y, normal_z, face_bn, gamma,
    )
    return (
        ct_primitive_to_conservative(
            left_primitive, normal_x, normal_y, normal_z, face_bn, gamma,
        ),
        ct_primitive_to_conservative(
            right_primitive, normal_x, normal_y, normal_z, face_bn, gamma,
        ),
    )
end

@inline function ct_primitive_to_conservative(
    W::SVector{9,T}, normal_x::T, normal_y::T, normal_z::T,
    face_bn::T, gamma::T,
) where {T}
    rho = W[1]
    velocity_x = W[2]
    velocity_y = W[3]
    velocity_z = W[4]
    pressure = W[5]
    magnetic_x, magnetic_y, magnetic_z = ct_replace_normal_component(
        W[6], W[7], W[8], normal_x, normal_y, normal_z, face_bn,
    )
    kinetic = T(0.5) * rho * (
        velocity_x^2 + velocity_y^2 + velocity_z^2
    )
    magnetic = T(0.5) * INV_MU0_SI * (
        magnetic_x^2 + magnetic_y^2 + magnetic_z^2
    )
    energy = isothermal_mhd ? kinetic + magnetic + pressure :
        pressure / (gamma - one(T)) + kinetic + magnetic
    return SVector{9,T}(
        rho,
        rho * velocity_x,
        rho * velocity_y,
        rho * velocity_z,
        energy,
        magnetic_x,
        magnetic_y,
        magnetic_z,
        W[9],
    )
end
