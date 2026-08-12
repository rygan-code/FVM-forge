if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end
if !@isdefined(STRUCTURED_FACE_QUADRATURE_LOADED)
    include(joinpath(@__DIR__, "structured_face_quadrature.jl"))
end

@inline _structured_state_component(U, Q, i, j, k, n) =
    ct_cell_state_component(U, Q, i, j, k, n)

@inline function _ct_background_cell_sample(background_cell, i, j, k)
    @inbounds return SVector{3,FT}(
        background_cell[i,j,k,1],
        background_cell[i,j,k,2],
        background_cell[i,j,k,3],
    )
end

@inline function _ct_background_split_sample(Q, background_cell, i, j, k)
    background = _ct_background_cell_sample(background_cell, i, j, k)
    @inbounds total = SVector{3,FT}(
        Q[i,j,k,QBX],Q[i,j,k,QBY],Q[i,j,k,QBZ],
    )
    return total-background, background
end

@inline function _ct_weno7_vector_interface(samples, ss)
    T = eltype(first(samples))
    left = SVector{3,T}(ntuple(Val(3)) do component
        stencil = SVector{7,T}(ntuple(
            sample -> samples[sample][component], Val(7),
        ))
        weno7_face_left(stencil,ss)
    end)
    right = SVector{3,T}(ntuple(Val(3)) do component
        stencil = SVector{7,T}(ntuple(
            sample -> samples[sample+1][component], Val(7),
        ))
        weno7_face_right(stencil,ss)
    end)
    return left,right
end

@inline function _ct_weno_ao53_vector_interface(samples)
    T=eltype(first(samples))
    left=SVector{3,T}(ntuple(Val(3)) do component
        values=SVector{5,T}(ntuple(
            sample -> samples[sample+1][component],Val(5),
        ))
        ct_weno_ao53_point_left(values)
    end)
    right=SVector{3,T}(ntuple(Val(3)) do component
        values=SVector{5,T}(ntuple(
            sample -> samples[sample+2][component],Val(5),
        ))
        ct_weno_ao53_point_right(values)
    end)
    return left,right
end

@inline function ct_background_interface_field(
    background_cell, i, j, k, ss,
    normal_x, normal_y, normal_z, background_bn,
    ::Val{DIRECTION},
) where {DIRECTION}
    di = DIRECTION == 1 ? Int32(1) : Int32(0)
    dj = DIRECTION == 2 ? Int32(1) : Int32(0)
    dk = DIRECTION == 3 ? Int32(1) : Int32(0)
    samples = ntuple(Val(8)) do sample
        offset = Int32(sample-4)
        _ct_background_cell_sample(
            background_cell,
            i+offset*di,j+offset*dj,k+offset*dk,
        )
    end
    background_left,background_right =
        _ct_weno7_vector_interface(samples,ss)
    background_face = (background_left+background_right)/FT(2)
    return SVector{3,FT}(ct_replace_normal_component(
        background_face[1],background_face[2],background_face[3],
        normal_x,normal_y,normal_z,background_bn,
    ))
end

@inline function ct_background_split_interface_magnetic(
    Q, background_cell, i, j, k, ss,
    normal_x, normal_y, normal_z, background_bn,
    direction::Val{DIRECTION},
    troubled_level::Int32=CT_TROUBLED_SMOOTH,
) where {DIRECTION}
    di = DIRECTION == 1 ? Int32(1) : Int32(0)
    dj = DIRECTION == 2 ? Int32(1) : Int32(0)
    dk = DIRECTION == 3 ? Int32(1) : Int32(0)
    if troubled_level >= CT_TROUBLED_RECOVERABLE
        perturbation_left,perturbation_right =
            _ct_background_split_perturbation_weno_ao53(
                Q,background_cell,i,j,k,di,dj,dk,
            )
    else
        perturbation_left,perturbation_right =
            _ct_background_split_perturbation_weno7(
                Q,background_cell,i,j,k,di,dj,dk,ss,
            )
    end
    background_face = ct_background_interface_field(
        background_cell,i,j,k,ss,
        normal_x,normal_y,normal_z,background_bn,direction,
    )
    return background_face+perturbation_left,
           background_face+perturbation_right,background_face
end

@inline function _ct_background_split_perturbation_weno_ao53(
    Q,background_cell,i,j,k,di,dj,dk,
)
    samples=ntuple(Val(8)) do sample
        offset=Int32(sample-4)
        perturbation,_=_ct_background_split_sample(
            Q,background_cell,
            i+offset*di,j+offset*dj,k+offset*dk,
        )
        perturbation
    end
    return _ct_weno_ao53_vector_interface(samples)
end

@inline function _ct_background_split_perturbation_first_order(
    Q,background_cell,i,j,k,di,dj,dk,
)
    left,_ = _ct_background_split_sample(Q,background_cell,i,j,k)
    right,_ = _ct_background_split_sample(
        Q,background_cell,i+di,j+dj,k+dk,
    )
    return left,right
end

@inline function _ct_background_split_perturbation_plm(
    Q,background_cell,i,j,k,di,dj,dk,
)
    samples = ntuple(Val(4)) do sample
        offset = Int32(sample-2)
        perturbation,_ = _ct_background_split_sample(
            Q,background_cell,
            i+offset*di,j+offset*dj,k+offset*dk,
        )
        perturbation
    end
    left=SVector{3,FT}(ntuple(Val(3)) do component
        ct_plm_plus(
            samples[1][component],samples[2][component],samples[3][component],
        )
    end)
    right=SVector{3,FT}(ntuple(Val(3)) do component
        ct_plm_minus(
            samples[2][component],samples[3][component],samples[4][component],
        )
    end)
    return left,right
end

@inline function _ct_background_split_perturbation_weno7(
    Q,background_cell,i,j,k,di,dj,dk,ss,
)
    samples = ntuple(Val(8)) do sample
        offset = Int32(sample-4)
        perturbation,_ = _ct_background_split_sample(
            Q,background_cell,
            i+offset*di,j+offset*dj,k+offset*dk,
        )
        perturbation
    end
    return _ct_weno7_vector_interface(samples,ss)
end

@inline function ct_background_split_interface_states(
    left_state, right_state, Q, background_cell,
    i, j, k, ss, normal_x, normal_y, normal_z, background_bn,
    ::Val{DIRECTION},
    troubled_level::Int32=CT_TROUBLED_SMOOTH,
) where {DIRECTION}
    magnetic_left,magnetic_right,background_face =
        ct_background_split_interface_magnetic(
            Q,background_cell,i,j,k,ss,
            normal_x,normal_y,normal_z,background_bn,Val(DIRECTION),
            troubled_level,
        )
    left_state = ct_impose_magnetic_preserve_p(
        left_state,magnetic_left,
    )
    right_state = ct_impose_magnetic_preserve_p(
        right_state,magnetic_right,
    )
    return left_state,right_state,background_face
end

@inline function Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕ, hp1, lin_ϕ, splitMethodID, ch_glm::FT)
    @static if equation_type == :MHD
        @static if isothermal_mhd
            if splitMethodID == Int32(6)
                return MHD_HLLE_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
            end
            return MHD_Rusanov_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        end
        @static if strict_ct_positivity && ct_mode
            if splitMethodID == Int32(4)
                return HLLD_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
            end
        end
        # ── MHD mode: use MHD-specific Riemann solvers ──
        if splitMethodID == Int32(5)
            return MHD_KEP_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        end
        if splitMethodID == Int32(6)
            return MHD_HLLE_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        end
        if splitMethodID == Int32(4)
            f_upw = HLLD_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        else
            f_upw = MHD_Rusanov_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        end
        if ϕ >= hp1
            return f_upw
        end
        f_kep = MHD_KEP_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        a = one(FT) - lin_ϕ
        b = lin_ϕ
        return SVector{9, FT}((
            a*f_kep[1]+b*f_upw[1], a*f_kep[2]+b*f_upw[2], a*f_kep[3]+b*f_upw[3],
            a*f_kep[4]+b*f_upw[4], a*f_kep[5]+b*f_upw[5], a*f_kep[6]+b*f_upw[6],
            a*f_kep[7]+b*f_upw[7], a*f_kep[8]+b*f_upw[8], a*f_kep[9]+b*f_upw[9]))
    else
        # ── Compressible mode ──
        if splitMethodID == Int32(5)
            return KEP_Flux(UL_vec, UR_vec, nx, ny, nz)
        end
        f_upw = if splitMethodID == Int32(1)
            HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)
        elseif splitMethodID == Int32(2)
            SW_Flux(UL_vec, UR_vec, nx, ny, nz)
        elseif splitMethodID == Int32(3)
            VL_Flux(UL_vec, UR_vec, nx, ny, nz)
        elseif splitMethodID == Int32(4)
            Roe_Flux(UL_vec, UR_vec, nx, ny, nz)
        else
            HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)
        end
        if ϕ >= hp1
            return f_upw
        end
        f_kep = KEP_Flux(UL_vec, UR_vec, nx, ny, nz)
        a = one(FT) - lin_ϕ
        b = lin_ϕ
        return SVector{5, FT}((
            a*f_kep[1]+b*f_upw[1], a*f_kep[2]+b*f_upw[2], a*f_kep[3]+b*f_upw[3],
            a*f_kep[4]+b*f_upw[4], a*f_kep[5]+b*f_upw[5]))
    end
end

@inline function _ct_record_characteristic_recovery!(
    pos_meta, recovery_mode::Int32,
)
    pos_meta === nothing && return nothing
    if recovery_mode == Int32(2)
        ct_record_fallback!(pos_meta,CT_POS_CHARACTERISTIC_LIMIT_COUNT)
    elseif recovery_mode == Int32(3)
        ct_record_fallback!(pos_meta, CT_POS_WENO_TO_PLM_COUNT)
    elseif recovery_mode >= Int32(4)
        ct_record_fallback!(pos_meta, CT_POS_PLM_TO_FIRST_COUNT)
    end
    return nothing
end

@inline function _ct_finalize_characteristic_state!(
    states, fi, fj, fk, primitive_state, nx, ny, nz, face_bn,
    direction::Int32, side::Int32, i, j, k, pos_meta, pos_values,
    magnetic=nothing,
)
    state = ct_primitive_to_conservative(
        primitive_state, nx, ny, nz, face_bn, FT(γ),
    )
    if magnetic !== nothing
        state = ct_impose_magnetic_preserve_p(state, magnetic)
    end
    _ct_record_if_invalid!(
        pos_meta, pos_values, CT_POS_SITE_RECONSTRUCTED,
        direction, side, i, j, k, state, face_bn,
    )
    before_face_b = state
    state = ct_impose_face_bn_preserve_p(
        state, nx, ny, nz, face_bn,
    )
    _ct_record_pressure_change!(
        pos_meta, pos_values, direction, side, i, j, k,
        before_face_b, state, face_bn,
    )
    _ct_record_if_invalid!(
        pos_meta, pos_values, CT_POS_SITE_HLLD_INPUT,
        direction, side, i, j, k, state, face_bn,
    )
    @inbounds for n = 1:9
        states[fi, fj, fk, n] = state[n]
    end
    return
end

@inline function _ct_hlld_flux_from_interface_states!(
    left_states, right_states, flux, rho_sum,
    fi, fj, fk, nx, ny, nz, area, ch_glm, pos_meta,
    background=nothing,
)
    left_state = SVector{9,FT}(
        ntuple(n -> @inbounds(left_states[fi, fj, fk, n]), Val(9)),
    )
    right_state = SVector{9,FT}(
        ntuple(n -> @inbounds(right_states[fi, fj, fk, n]), Val(9)),
    )
    flux_value=HLLD_Flux(
        left_state,right_state,nx,ny,nz,ch_glm,pos_meta,
    )
    if background !== nothing
        flux_value = ct_remove_background_maxwell_stress(
            flux_value,background,nx,ny,nz,
        )
    end
    @inbounds rho_sum[fi, fj, fk] = left_state[1] + right_state[1]
    @inbounds for n = 1:9
        flux[fi, fj, fk, n] = flux_value[n] * area
    end
    return
end

@inline function _ct_mask_level(value)
    if !isfinite(value) || value >= typeof(value)(CT_TROUBLED_STRONG)
        return CT_TROUBLED_STRONG
    elseif value >= typeof(value)(CT_TROUBLED_RECOVERABLE)
        return CT_TROUBLED_RECOVERABLE
    end
    return CT_TROUBLED_SMOOTH
end

@inline function ct_face_troubled_level(
    ::Nothing,i,j,k,::Val{DIRECTION},
) where {DIRECTION}
    return CT_TROUBLED_SMOOTH
end

@inline function ct_face_troubled_level(
    sensor,i,j,k,::Val{DIRECTION},
) where {DIRECTION}
    di=DIRECTION == 1 ? Int32(1) : Int32(0)
    dj=DIRECTION == 2 ? Int32(1) : Int32(0)
    dk=DIRECTION == 3 ? Int32(1) : Int32(0)
    @inbounds left=sensor[i,j,k]
    @inbounds right=sensor[i+di,j+dj,k+dk]
    (!isfinite(left) || !isfinite(right)) && return CT_TROUBLED_STRONG
    adjacent_level=_ct_mask_level(max(left,right))
    adjacent_level >= CT_TROUBLED_STRONG && return CT_TROUBLED_STRONG
    support_is_troubled=adjacent_level >= CT_TROUBLED_RECOVERABLE
    # Union of the left and right WENO7 supports at this face. Keeping this
    # dilation normal to the face prevents a shock from affecting unrelated
    # tangential rows. Either troubled category selects characteristic AO;
    # only an actually inadmissible candidate may trigger further recovery.
    for offset in Int32(-3):Int32(4)
        @inbounds sample=sensor[
            i+offset*di,j+offset*dj,k+offset*dk,
        ]
        if !isfinite(sample) ||
           _ct_mask_level(sample) >= CT_TROUBLED_RECOVERABLE
            support_is_troubled=true
        end
    end
    return support_is_troubled ?
        CT_TROUBLED_RECOVERABLE : CT_TROUBLED_SMOOTH
end

@inline function _ct_mhd_characteristic_reconstruct_kernel!(
    Q, states, area_array, nx_array, ny_array, nz_array, face_b,
    background_face, background_cell,
    nxp, nyp, nzp, mode::Int32, pos_meta, pos_values, sensor,
    ::Val{DIRECTION}, ::Val{SIDE},
) where {DIRECTION,SIDE}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    tangential_halo = STRUCTURED_FLUX_TANGENTIAL_HALO
    i_lo = DIRECTION == 1 ? Int32(NG) : Int32(NG)+Int32(1)-tangential_halo
    j_lo = DIRECTION == 2 ? Int32(NG) : Int32(NG)+Int32(1)-tangential_halo
    k_lo = DIRECTION == 3 ? Int32(NG) : Int32(NG)+Int32(1)-tangential_halo
    i_hi = DIRECTION == 1 ? nxp+Int32(NG) : nxp+Int32(NG)+tangential_halo
    j_hi = DIRECTION == 2 ? nyp+Int32(NG) : nyp+Int32(NG)+tangential_halo
    k_hi = DIRECTION == 3 ? nzp+Int32(NG) : nzp+Int32(NG)+tangential_halo
    if i > i_hi || j > j_hi || k > k_hi ||
       i < i_lo || j < j_lo || k < k_lo
        return
    end

    normal_index = DIRECTION == 1 ? i : (DIRECTION == 2 ? j : k)
    normal_extent = DIRECTION == 1 ? nxp : (DIRECTION == 2 ? nyp : nzp)
    if mode == Int32(1) &&
       (normal_index < NG+Int32(4) ||
        normal_index > normal_extent+NG-Int32(4))
        return
    end
    if mode == Int32(2) &&
       normal_index >= NG+Int32(4) &&
       normal_index <= normal_extent+NG-Int32(4)
        return
    end

    di = DIRECTION == 1 ? Int32(1) : Int32(0)
    dj = DIRECTION == 2 ? Int32(1) : Int32(0)
    dk = DIRECTION == 3 ? Int32(1) : Int32(0)
    face_i, face_j, face_k = i+di, j+dj, k+dk
    area,nx,ny,nz,face_bn = structured_ct_face_geometry_bn(
        face_b,background_face,area_array,nx_array,ny_array,nz_array,
        face_i,face_j,face_k,Val(DIRECTION),Q,
    )
    background_bn = zero(FT)
    if background_cell !== nothing
        _,_,_,_,background_bn = structured_ct_face_geometry_bn(
            background_face,nothing,
            area_array,nx_array,ny_array,nz_array,
            face_i,face_j,face_k,Val(DIRECTION),nothing,CT_FACE_BN_POINT6,
        )
    end

    troubled_level=ct_face_troubled_level(
        ct_troubled_consumer_sensor(
            sensor,Val(ct_troubled_face_reconstruction_enabled),
        ),
        i,j,k,Val(DIRECTION),
    )

    if ct_characteristic_reconstruction == CT_CHARACTERISTIC_WENO7
        first_offset = SIDE == 1 ? Int32(-3) : Int32(-2)
        stencil = ntuple(Val(7)) do s
            offset = first_offset + Int32(s-1)
            ct_primitive_state(
                Q, i+offset*di, j+offset*dj, k+offset*dk,
            )
        end
        primitive_state, recovery_mode =
            _ct_characteristic_candidate_admissible(
                stencil,nx,ny,nz,face_bn,FT(γ),Val(SIDE),
                troubled_level >= CT_TROUBLED_RECOVERABLE,
            )
        _ct_record_characteristic_recovery!(pos_meta, recovery_mode)
    elseif SIDE == 1
        Wm = ct_primitive_state(Q, i-di, j-dj, k-dk)
        Wc = ct_primitive_state(Q, i, j, k)
        Wp = ct_primitive_state(Q, i+di, j+dj, k+dk)
        primitive_state = ct_mhd_characteristic_plm_plus(
            Wm, Wc, Wp, nx, ny, nz, face_bn, FT(γ),
        )
        if !_ct_primitive_is_physical(primitive_state)
            ct_record_fallback!(pos_meta, CT_POS_PLM_TO_FIRST_COUNT)
            primitive_state = Wc
        end
    else
        Wc = ct_primitive_state(Q, i, j, k)
        Wp = ct_primitive_state(Q, i+di, j+dj, k+dk)
        Wpp = ct_primitive_state(Q, i+2di, j+2dj, k+2dk)
        primitive_state = ct_mhd_characteristic_plm_minus(
            Wc, Wp, Wpp, nx, ny, nz, face_bn, FT(γ),
        )
        if !_ct_primitive_is_physical(primitive_state)
            ct_record_fallback!(pos_meta, CT_POS_PLM_TO_FIRST_COUNT)
            primitive_state = Wp
        end
    end

    fi = DIRECTION == 1 ? i-Int32(NG)+Int32(1) :
         i-Int32(NG)+tangential_halo
    fj = DIRECTION == 2 ? j-Int32(NG)+Int32(1) :
         j-Int32(NG)+tangential_halo
    fk = DIRECTION == 3 ? k-Int32(NG)+Int32(1) :
         k-Int32(NG)+tangential_halo
    if background_cell === nothing
        _ct_finalize_characteristic_state!(
            states, fi, fj, fk, primitive_state, nx, ny, nz, face_bn,
            Int32(DIRECTION), Int32(SIDE), i, j, k, pos_meta, pos_values,
        )
    else
        @inbounds split_ss = FT(2)/(
            area_array[i+di,j+dj,k+dk]+area_array[i,j,k]
        )
        magnetic_left,magnetic_right,_ =
            ct_background_split_interface_magnetic(
                Q,background_cell,i,j,k,split_ss,
                nx,ny,nz,background_bn,Val(DIRECTION),troubled_level,
            )
        magnetic = SIDE == 1 ? magnetic_left : magnetic_right
        _ct_finalize_characteristic_state!(
            states, fi, fj, fk, primitive_state, nx, ny, nz, face_bn,
            Int32(DIRECTION), Int32(SIDE), i, j, k, pos_meta, pos_values,
            magnetic,
        )
    end
    return
end

function ct_mhd_characteristic_reconstruct_left_i_kernel!(
    Q, states, area, nx, ny, nz, face_b,
    nxp, nyp, nzp, mode::Int32, pos_meta, pos_values,
    background_face=nothing, background_cell=nothing, sensor=nothing,
)
    _ct_mhd_characteristic_reconstruct_kernel!(
        Q, states, area, nx, ny, nz, face_b, background_face, background_cell,
        nxp,nyp,nzp,mode,pos_meta,pos_values,sensor,Val(1),Val(1),
    )
end

function ct_mhd_characteristic_reconstruct_right_i_kernel!(
    Q, states, area, nx, ny, nz, face_b,
    nxp, nyp, nzp, mode::Int32, pos_meta, pos_values,
    background_face=nothing, background_cell=nothing, sensor=nothing,
)
    _ct_mhd_characteristic_reconstruct_kernel!(
        Q, states, area, nx, ny, nz, face_b, background_face, background_cell,
        nxp,nyp,nzp,mode,pos_meta,pos_values,sensor,Val(1),Val(2),
    )
end

function ct_mhd_characteristic_reconstruct_left_j_kernel!(
    Q, states, area, nx, ny, nz, face_b,
    nxp, nyp, nzp, mode::Int32, pos_meta, pos_values,
    background_face=nothing, background_cell=nothing, sensor=nothing,
)
    _ct_mhd_characteristic_reconstruct_kernel!(
        Q, states, area, nx, ny, nz, face_b, background_face, background_cell,
        nxp,nyp,nzp,mode,pos_meta,pos_values,sensor,Val(2),Val(1),
    )
end

function ct_mhd_characteristic_reconstruct_right_j_kernel!(
    Q, states, area, nx, ny, nz, face_b,
    nxp, nyp, nzp, mode::Int32, pos_meta, pos_values,
    background_face=nothing, background_cell=nothing, sensor=nothing,
)
    _ct_mhd_characteristic_reconstruct_kernel!(
        Q, states, area, nx, ny, nz, face_b, background_face, background_cell,
        nxp,nyp,nzp,mode,pos_meta,pos_values,sensor,Val(2),Val(2),
    )
end

function ct_mhd_characteristic_reconstruct_left_k_kernel!(
    Q, states, area, nx, ny, nz, face_b,
    nxp, nyp, nzp, mode::Int32, pos_meta, pos_values,
    background_face=nothing, background_cell=nothing, sensor=nothing,
)
    _ct_mhd_characteristic_reconstruct_kernel!(
        Q, states, area, nx, ny, nz, face_b, background_face, background_cell,
        nxp,nyp,nzp,mode,pos_meta,pos_values,sensor,Val(3),Val(1),
    )
end

function ct_mhd_characteristic_reconstruct_right_k_kernel!(
    Q, states, area, nx, ny, nz, face_b,
    nxp, nyp, nzp, mode::Int32, pos_meta, pos_values,
    background_face=nothing, background_cell=nothing, sensor=nothing,
)
    _ct_mhd_characteristic_reconstruct_kernel!(
        Q, states, area, nx, ny, nz, face_b, background_face, background_cell,
        nxp,nyp,nzp,mode,pos_meta,pos_values,sensor,Val(3),Val(2),
    )
end

@inline function _ct_mhd_hlld_flux_kernel!(
    left_states, right_states, flux, rho_sum,
    area_array, nx_array, ny_array, nz_array,
    nxp, nyp, nzp, ch_glm::FT, mode::Int32, pos_meta,
    background_face, background_cell, sensor,
    ::Val{DIRECTION},
) where {DIRECTION}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    tangential_halo = STRUCTURED_FLUX_TANGENTIAL_HALO
    i_lo = DIRECTION == 1 ? Int32(NG) : Int32(NG)+Int32(1)-tangential_halo
    j_lo = DIRECTION == 2 ? Int32(NG) : Int32(NG)+Int32(1)-tangential_halo
    k_lo = DIRECTION == 3 ? Int32(NG) : Int32(NG)+Int32(1)-tangential_halo
    i_hi = DIRECTION == 1 ? nxp+Int32(NG) : nxp+Int32(NG)+tangential_halo
    j_hi = DIRECTION == 2 ? nyp+Int32(NG) : nyp+Int32(NG)+tangential_halo
    k_hi = DIRECTION == 3 ? nzp+Int32(NG) : nzp+Int32(NG)+tangential_halo
    if i > i_hi || j > j_hi || k > k_hi ||
       i < i_lo || j < j_lo || k < k_lo
        return
    end

    normal_index = DIRECTION == 1 ? i : (DIRECTION == 2 ? j : k)
    normal_extent = DIRECTION == 1 ? nxp : (DIRECTION == 2 ? nyp : nzp)
    if mode == Int32(1) &&
       (normal_index < NG+Int32(4) ||
        normal_index > normal_extent+NG-Int32(4))
        return
    end
    if mode == Int32(2) &&
       normal_index >= NG+Int32(4) &&
       normal_index <= normal_extent+NG-Int32(4)
        return
    end

    di = DIRECTION == 1 ? Int32(1) : Int32(0)
    dj = DIRECTION == 2 ? Int32(1) : Int32(0)
    dk = DIRECTION == 3 ? Int32(1) : Int32(0)
    face_i, face_j, face_k = i+di, j+dj, k+dk
    area,nx,ny,nz = structured_ct_face_geometry(
        area_array,nx_array,ny_array,nz_array,
        face_i,face_j,face_k,Val(DIRECTION),
    )
    background = nothing
    if background_cell !== nothing
        _,_,_,_,background_bn = structured_ct_face_geometry_bn(
            background_face,nothing,
            area_array,nx_array,ny_array,nz_array,
            face_i,face_j,face_k,Val(DIRECTION),nothing,CT_FACE_BN_POINT6,
        )
        @inbounds split_ss = FT(2)/(
            area_array[i+di,j+dj,k+dk]+area_array[i,j,k]
        )
        background = ct_background_interface_field(
            background_cell,i,j,k,split_ss,
            nx,ny,nz,background_bn,Val(DIRECTION),
        )
    end
    fi = DIRECTION == 1 ? i-Int32(NG)+Int32(1) :
         i-Int32(NG)+tangential_halo
    fj = DIRECTION == 2 ? j-Int32(NG)+Int32(1) :
         j-Int32(NG)+tangential_halo
    fk = DIRECTION == 3 ? k-Int32(NG)+Int32(1) :
         k-Int32(NG)+tangential_halo
    _ct_hlld_flux_from_interface_states!(
        left_states, right_states, flux, rho_sum,
        fi, fj, fk, nx, ny, nz, area, ch_glm, pos_meta,
        background,
    )
    return
end

function ct_mhd_hlld_flux_i_kernel!(
    left_states, right_states, flux, rho_sum,
    area, nx, ny, nz, nxp, nyp, nzp, ch_glm::FT, mode::Int32,
    pos_meta,background_face=nothing,background_cell=nothing,sensor=nothing,
)
    _ct_mhd_hlld_flux_kernel!(
        left_states, right_states, flux, rho_sum,
        area, nx, ny, nz, nxp, nyp, nzp, ch_glm, mode, pos_meta,
        background_face,background_cell,sensor,Val(1),
    )
end

function ct_mhd_hlld_flux_j_kernel!(
    left_states, right_states, flux, rho_sum,
    area, nx, ny, nz, nxp, nyp, nzp, ch_glm::FT, mode::Int32,
    pos_meta,background_face=nothing,background_cell=nothing,sensor=nothing,
)
    _ct_mhd_hlld_flux_kernel!(
        left_states, right_states, flux, rho_sum,
        area, nx, ny, nz, nxp, nyp, nzp, ch_glm, mode, pos_meta,
        background_face,background_cell,sensor,Val(2),
    )
end

function ct_mhd_hlld_flux_k_kernel!(
    left_states, right_states, flux, rho_sum,
    area, nx, ny, nz, nxp, nyp, nzp, ch_glm::FT, mode::Int32,
    pos_meta,background_face=nothing,background_cell=nothing,sensor=nothing,
)
    _ct_mhd_hlld_flux_kernel!(
        left_states, right_states, flux, rho_sum,
        area, nx, ny, nz, nxp, nyp, nzp, ch_glm, mode, pos_meta,
        background_face,background_cell,sensor,Val(3),
    )
end

@inline function _ct_fofc_face_cell_indices(
    fi, fj, fk, ::Val{DIRECTION},
) where {DIRECTION}
    ng = Int32(NG)
    tangent = Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    if DIRECTION == 1
        left = (fi+ng-Int32(1),fj+ng-tangent,fk+ng-tangent)
        right = (left[1]+Int32(1),left[2],left[3])
        geometry = (fi+ng,left[2],left[3])
    elseif DIRECTION == 2
        left = (fi+ng-tangent,fj+ng-Int32(1),fk+ng-tangent)
        right = (left[1],left[2]+Int32(1),left[3])
        geometry = (left[1],fj+ng,left[3])
    else
        left = (fi+ng-tangent,fj+ng-tangent,fk+ng-Int32(1))
        right = (left[1],left[2],left[3]+Int32(1))
        geometry = (left[1],left[2],fk+ng)
    end
    return left,right,geometry
end

@inline function _ct_fofc_face_activity(flags, left, right)
    @inbounds begin
        left_value = flags[left...,1]
        right_value = flags[right...,1]
    end
    left_active = ct_fofc_flag_is_active(left_value)
    right_active = ct_fofc_flag_is_active(right_value)
    active = left_active || right_active
    scale = min(
        left_active ? ct_fofc_flag_scale(left_value) : one(FT),
        right_active ? ct_fofc_flag_scale(right_value) : one(FT),
    )
    return active,scale
end

@inline function _ct_fofc_cell_magnetic(
    Bx_face,By_face,Bz_face,
    B0x_face,B0y_face,B0z_face,
    Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,
    cell,
)
    return _ct_recover_cell_b_from_face_fluxes(
        Bx_face,By_face,Bz_face,
        Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,
        CT_CELL_B_LSQ2,cell...,B0x_face,B0y_face,B0z_face,
    )
end

@inline function _ct_fofc_cell_state(U,magnetic,cell)
    @inbounds return SVector{9,FT}(
        U[cell...,1],U[cell...,2],U[cell...,3],U[cell...,4],U[cell...,5],
        magnetic[1],magnetic[2],magnetic[3],zero(FT),
    )
end

@inline function _ct_fofc_background_interface(
    ::Nothing,B0y_face,B0z_face,
    Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,
    area,nx,ny,nz,left,right,geometry,
)
    return nothing,nothing,nothing
end

@inline function _ct_fofc_background_interface(
    B0x_face,B0y_face,B0z_face,
    Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,
    area,nx,ny,nz,left,right,geometry,
)
    left_background = _ct_fofc_cell_magnetic(
        B0x_face,B0y_face,B0z_face,nothing,nothing,nothing,
        Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,left,
    )
    right_background = _ct_fofc_cell_magnetic(
        B0x_face,B0y_face,B0z_face,nothing,nothing,nothing,
        Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,right,
    )
    midpoint = FT(0.5)*(left_background+right_background)
    gi,gj,gk = geometry
    @inbounds begin
        face_area = area[gi,gj,gk]
        normal_x,normal_y,normal_z = nx[gi,gj,gk],ny[gi,gj,gk],nz[gi,gj,gk]
        background_face = if geometry[1] != left[1]
            B0x_face
        elseif geometry[2] != left[2]
            B0y_face
        else
            B0z_face
        end
        background_bn = background_face[gi,gj,gk]/face_area
    end
    background = SVector{3,FT}(ct_replace_normal_component(
        midpoint[1],midpoint[2],midpoint[3],
        normal_x,normal_y,normal_z,background_bn,
    ))
    return background,left_background,right_background
end

@inline function _ct_fofc_replace_face_flux!(
    flux,rho_sum,flags,U,
    Bx_face,By_face,Bz_face,B0x_face,B0y_face,B0z_face,
    Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,
    fi, fj, fk, ch_glm, fallback_meta, record_face::Bool,
    direction::Val{DIRECTION},
) where {DIRECTION}
    left,right,geometry = _ct_fofc_face_cell_indices(
        fi,fj,fk,direction,
    )
    active,_ = _ct_fofc_face_activity(flags,left,right)
    active || return false

    face_b = DIRECTION == 1 ? Bx_face : (DIRECTION == 2 ? By_face : Bz_face)
    background_face = DIRECTION == 1 ? B0x_face :
        (DIRECTION == 2 ? B0y_face : B0z_face)
    area_array = DIRECTION == 1 ? Areai : (DIRECTION == 2 ? Areaj : Areak)
    nx_array = DIRECTION == 1 ? nxi : (DIRECTION == 2 ? nxj : nxk)
    ny_array = DIRECTION == 1 ? nyi : (DIRECTION == 2 ? nyj : nyk)
    nz_array = DIRECTION == 1 ? nzi : (DIRECTION == 2 ? nzj : nzk)
    gi,gj,gk = geometry
    @inbounds begin
        area = area_array[gi,gj,gk]
        nx,ny,nz = nx_array[gi,gj,gk],ny_array[gi,gj,gk],nz_array[gi,gj,gk]
        face_bn = (
            face_b[gi,gj,gk] +
            _structured_optional_face_value(
                background_face,gi,gj,gk,eltype(area_array),
            )
        )/area
    end
    left_magnetic = _ct_fofc_cell_magnetic(
        Bx_face,By_face,Bz_face,B0x_face,B0y_face,B0z_face,
        Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,left,
    )
    right_magnetic = _ct_fofc_cell_magnetic(
        Bx_face,By_face,Bz_face,B0x_face,B0y_face,B0z_face,
        Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,right,
    )
    left_state = _ct_fofc_cell_state(U,left_magnetic,left)
    right_state = _ct_fofc_cell_state(U,right_magnetic,right)
    background,left_background,right_background =
        _ct_fofc_background_interface(
            B0x_face,B0y_face,B0z_face,
            Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,
            area_array,nx_array,ny_array,nz_array,left,right,geometry,
        )
    if background !== nothing
        left_interface_magnetic = background+(left_magnetic-left_background)
        right_interface_magnetic = background+(right_magnetic-right_background)
        left_state = ct_impose_magnetic_preserve_p(
            left_state,left_interface_magnetic,
        )
        right_state = ct_impose_magnetic_preserve_p(
            right_state,right_interface_magnetic,
        )
    end
    left_state = ct_impose_face_bn_preserve_p(
        left_state,nx,ny,nz,face_bn,
    )
    right_state = ct_impose_face_bn_preserve_p(
        right_state,nx,ny,nz,face_bn,
    )
    low_order_flux = MHD_HLLE_Flux(
        left_state,right_state,nx,ny,nz,ch_glm,
    )
    if background !== nothing
        low_order_flux = ct_remove_background_maxwell_stress(
            low_order_flux,background,nx,ny,nz,
        )
    end
    @inbounds begin
        rho_sum[fi,fj,fk] = left_state[1]+right_state[1]
        for variable in 1:9
            flux[fi,fj,fk,variable] = low_order_flux[variable]*area
        end
    end
    record_face &&
        ct_record_fallback!(fallback_meta,CT_POS_FOFC_FACE_COUNT)
    return true
end

function ct_fofc_replace_face_flux_kernel!(
    flux,rho_sum,flags,U,
    Bx_face,By_face,Bz_face,B0x_face,B0y_face,B0z_face,
    Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,
    nxp,nyp,nzp,ch_glm,fallback_meta,record_face::Bool,
    ::Val{DIRECTION},
) where {DIRECTION}
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    tangent = Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    ni = DIRECTION == 1 ? nxp+Int32(1) : nxp+Int32(2)*tangent
    nj = DIRECTION == 2 ? nyp+Int32(1) : nyp+Int32(2)*tangent
    nk = DIRECTION == 3 ? nzp+Int32(1) : nzp+Int32(2)*tangent
    (i > ni || j > nj || k > nk) && return
    _ct_fofc_replace_face_flux!(
        flux,rho_sum,flags,U,
        Bx_face,By_face,Bz_face,B0x_face,B0y_face,B0z_face,
        Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,
        i,j,k,ch_glm,fallback_meta,record_face,Val(DIRECTION),
    )
    return
end


@inline function _ct_fofc_scale_face_flux!(
    flux,diffusive_flux,flags,fi,fj,fk,direction::Val{DIRECTION},
) where {DIRECTION}
    left,right,_ = _ct_fofc_face_cell_indices(fi,fj,fk,direction)
    active,scale = _ct_fofc_face_activity(flags,left,right)
    active || return false
    @inbounds for variable in 1:9
        flux[fi,fj,fk,variable] *= scale
        diffusive_flux[fi,fj,fk,variable] *= scale
    end
    return true
end

function ct_fofc_scale_face_flux_kernel!(
    flux,diffusive_flux,flags,nxp,nyp,nzp,::Val{DIRECTION},
) where {DIRECTION}
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    tangent = Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    ni = DIRECTION == 1 ? nxp+Int32(1) : nxp+Int32(2)*tangent
    nj = DIRECTION == 2 ? nyp+Int32(1) : nyp+Int32(2)*tangent
    nk = DIRECTION == 3 ? nzp+Int32(1) : nzp+Int32(2)*tangent
    (i > ni || j > nj || k > nk) && return
    _ct_fofc_scale_face_flux!(
        flux,diffusive_flux,flags,i,j,k,Val(DIRECTION),
    )
    return
end

@inline function _ct_mhd_characteristic_weno7_cache_kernel!(
    Q, cache, area_array, nx_array, ny_array, nz_array,
    face_b, background_face, background_cell, Vol,
    nxp,nyp,nzp,ch_glm::FT,dt_stage::FT,mode::Int32,pos_meta,sensor,
    ::Val{DIRECTION},
) where {DIRECTION}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    normal_index = DIRECTION == 1 ? i : (DIRECTION == 2 ? j : k)
    normal_extent = DIRECTION == 1 ? nxp : (DIRECTION == 2 ? nyp : nzp)
    in_i = DIRECTION == 1 ?
        (i >= NG && i <= nxp+NG) :
        (i >= Int32(1) && i <= nxp+Int32(2)*NG)
    in_j = DIRECTION == 2 ?
        (j >= NG && j <= nyp+NG) :
        (j >= Int32(1) && j <= nyp+Int32(2)*NG)
    in_k = DIRECTION == 3 ?
        (k >= NG && k <= nzp+NG) :
        (k >= Int32(1) && k <= nzp+Int32(2)*NG)
    if !(in_i && in_j && in_k)
        return
    end
    if mode == Int32(1) &&
       (normal_index < NG+Int32(4) ||
        normal_index > normal_extent+NG-Int32(4))
        return
    end
    if mode == Int32(2) &&
       normal_index >= NG+Int32(4) &&
       normal_index <= normal_extent+NG-Int32(4)
        return
    end

    di = DIRECTION == 1 ? Int32(1) : Int32(0)
    dj = DIRECTION == 2 ? Int32(1) : Int32(0)
    dk = DIRECTION == 3 ? Int32(1) : Int32(0)
    face_i, face_j, face_k = i+di, j+dj, k+dk
    area,nx,ny,nz,face_bn = structured_ct_face_geometry_bn(
        face_b,background_face,area_array,nx_array,ny_array,nz_array,
        face_i,face_j,face_k,Val(DIRECTION),Q,
    )
    background_bn = zero(FT)
    if background_cell !== nothing
        _,_,_,_,background_bn = structured_ct_face_geometry_bn(
            background_face,nothing,
            area_array,nx_array,ny_array,nz_array,
            face_i,face_j,face_k,Val(DIRECTION),nothing,CT_FACE_BN_POINT6,
        )
    end

    troubled_level=ct_face_troubled_level(
        ct_troubled_consumer_sensor(
            sensor,Val(ct_troubled_face_reconstruction_enabled),
        ),
        i,j,k,Val(DIRECTION),
    )
    if ct_characteristic_reconstruction == CT_CHARACTERISTIC_WENO7
        stencil = ntuple(Val(8)) do s
            offset = Int32(s-4)
            ct_primitive_state(
                Q, i+offset*di, j+offset*dj, k+offset*dk,
            )
        end
        left_primitive, left_recovery =
            _ct_characteristic_candidate_admissible(
                ntuple(index -> stencil[index], Val(7)),
                nx,ny,nz,face_bn,FT(γ),Val(1),
                troubled_level >= CT_TROUBLED_RECOVERABLE,
            )
        right_primitive, right_recovery =
            _ct_characteristic_candidate_admissible(
                ntuple(index -> stencil[index+1], Val(7)),
                nx,ny,nz,face_bn,FT(γ),Val(2),
                troubled_level >= CT_TROUBLED_RECOVERABLE,
            )
        _ct_record_characteristic_recovery!(pos_meta, left_recovery)
        _ct_record_characteristic_recovery!(pos_meta, right_recovery)
    else
        Wm = ct_primitive_state(Q, i-di, j-dj, k-dk)
        Wc = ct_primitive_state(Q, i, j, k)
        Wp = ct_primitive_state(Q, i+di, j+dj, k+dk)
        Wpp = ct_primitive_state(
            Q, i+Int32(2)*di, j+Int32(2)*dj, k+Int32(2)*dk,
        )
        left_primitive = ct_mhd_characteristic_plm_plus(
            Wm, Wc, Wp, nx, ny, nz, face_bn, FT(γ),
        )
        if !_ct_primitive_is_physical(left_primitive)
            left_primitive = Wc
            if pos_meta !== nothing
                ct_record_fallback!(
                    pos_meta, CT_POS_PLM_TO_FIRST_COUNT,
                )
            end
        end
        right_primitive = ct_mhd_characteristic_plm_minus(
            Wc, Wp, Wpp, nx, ny, nz, face_bn, FT(γ),
        )
        if !_ct_primitive_is_physical(right_primitive)
            right_primitive = Wp
            if pos_meta !== nothing
                ct_record_fallback!(
                    pos_meta, CT_POS_PLM_TO_FIRST_COUNT,
                )
            end
        end
    end
    left_state = ct_primitive_to_conservative(
        left_primitive, nx, ny, nz, face_bn, FT(γ),
    )
    right_state = ct_primitive_to_conservative(
        right_primitive, nx, ny, nz, face_bn, FT(γ),
    )
    background = nothing
    if background_cell !== nothing
        @inbounds split_ss = FT(2)/(
            area_array[i+di,j+dj,k+dk]+area_array[i,j,k]
        )
        left_state,right_state,background =
            ct_background_split_interface_states(
                left_state,right_state,Q,background_cell,
                i,j,k,split_ss,nx,ny,nz,background_bn,Val(DIRECTION),
                troubled_level,
            )
    end
    left_state = ct_impose_face_bn_preserve_p(
        left_state, nx, ny, nz, face_bn,
    )
    right_state = ct_impose_face_bn_preserve_p(
        right_state, nx, ny, nz, face_bn,
    )
    flux_temp=HLLD_Flux(
        left_state,right_state,nx,ny,nz,ch_glm,pos_meta,
    )
    if background !== nothing
        flux_temp = ct_remove_background_maxwell_stress(
            flux_temp,background,nx,ny,nz,
        )
    end
    flux_b = SVector{3,FT}(
        flux_temp[UBX], flux_temp[UBY], flux_temp[UBZ],
    )
    etan = ct_face_tangential_electric(
        flux_b, SVector{3,FT}(nx, ny, nz),
    )
    @inbounds spacing = ct_face_normal_spacing(
        Vol[i, j, k], Vol[i+di, j+dj, k+dk], area,
    )
    weight = ct_emf_weight_from_density_sum(
        flux_temp[1], left_state[1] + right_state[1], spacing, dt_stage,
    )

    ci = DIRECTION == 1 ? i-NG+1 : i
    cj = DIRECTION == 2 ? j-NG+1 : j
    ck = DIRECTION == 3 ? k-NG+1 : k
    @inbounds begin
        cache[ci, cj, ck, 1] = etan[1]
        cache[ci, cj, ck, 2] = etan[2]
        cache[ci, cj, ck, 3] = etan[3]
        cache[ci, cj, ck, 4] = weight
    end
    return
end

function ct_mhd_characteristic_weno7_cache_i_kernel!(
    Q, cache, area, nx, ny, nz, face_b, Vol,
    nxp, nyp, nzp, ch_glm::FT, dt_stage::FT, mode::Int32,
    background_face=nothing,background_cell=nothing,pos_meta=nothing,
    sensor=nothing,
)
    _ct_mhd_characteristic_weno7_cache_kernel!(
        Q, cache, area, nx, ny, nz,
        face_b, background_face, background_cell, Vol,
        nxp,nyp,nzp,ch_glm,dt_stage,mode,pos_meta,sensor,Val(1),
    )
end


function ct_mhd_characteristic_weno7_cache_j_kernel!(
    Q, cache, area, nx, ny, nz, face_b, Vol,
    nxp, nyp, nzp, ch_glm::FT, dt_stage::FT, mode::Int32,
    background_face=nothing,background_cell=nothing,pos_meta=nothing,
    sensor=nothing,
)
    _ct_mhd_characteristic_weno7_cache_kernel!(
        Q, cache, area, nx, ny, nz,
        face_b, background_face, background_cell, Vol,
        nxp,nyp,nzp,ch_glm,dt_stage,mode,pos_meta,sensor,Val(2),
    )
end


function ct_mhd_characteristic_weno7_cache_k_kernel!(
    Q, cache, area, nx, ny, nz, face_b, Vol,
    nxp, nyp, nzp, ch_glm::FT, dt_stage::FT, mode::Int32,
    background_face=nothing,background_cell=nothing,pos_meta=nothing,
    sensor=nothing,
)
    _ct_mhd_characteristic_weno7_cache_kernel!(
        Q, cache, area, nx, ny, nz,
        face_b, background_face, background_cell, Vol,
        nxp,nyp,nzp,ch_glm,dt_stage,mode,pos_meta,sensor,Val(3),
    )
end

@inline function weno_zq_blend(v0::FT, v1::FT, v2::FT, Is1::FT, Is2::FT, Is3::FT, Is4::FT, ss::FT) where {FT}
    β1 = Is2
    β2 = Is3
    β0 = max(Is1, max(Is2, max(Is3, Is4)))
    
    τ = abs(Is1 - Is4) * ss
    ϵ = FT(1.0e-10)
    
    γ0, γ1, γ2 = FT(0.90), FT(0.05), FT(0.05)
    
    α0 = γ0 * (one(FT) + τ / (ϵ + β0 * ss))^2
    α1 = γ1 * (one(FT) + τ / (ϵ + β1 * ss))^2
    α2 = γ2 * (one(FT) + τ / (ϵ + β2 * ss))^2
    
    invsum = one(FT) / (α0 + α1 + α2)
    ω0 = α0 * invsum
    ω1 = α1 * invsum
    ω2 = α2 * invsum
    
    v0_star = (v0 - γ1*v1 - γ2*v2) / γ0
    
    w0 = ω0 / γ0
    w1 = ω1 - ω0 * (γ1 / γ0)
    w2 = ω2 - ω0 * (γ2 / γ0)
    
    return w0 * v0_star + w1 * v1 + w2 * v2
end

function Eigen_reconstruct_i(Q, U, ϕ, S, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp,
                             stencil_arr, Δstencil_arr, lin_phi_arr,
                             stencil_R_arr, Δstencil_R_arr,
                             ch_glm::FT=one(FT), mode::Int32=Int32(0))
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. Boundary check. Point-face quadrature stores two tangential layers.
    tangential_halo = STRUCTURED_FLUX_TANGENTIAL_HALO
    if i > nxp+NG || j > nyp+NG+tangential_halo ||
       k > nzp+NG+tangential_halo || i < NG ||
       j < NG+Int32(1)-tangential_halo ||
       k < NG+Int32(1)-tangential_halo
        return
    end
    # Interior/boundary mode filter (stencil width=4 in i)
    if mode == Int32(1) && (i < NG+Int32(4) || i > nxp+NG-Int32(4)); return; end
    if mode == Int32(2) && (i >= NG+Int32(4) && i <= nxp+NG-Int32(4)); return; end
    # 2. Geometry
    Area,nx,ny,nz = structured_face_geometry(
        Areai,nxi,nyi,nzi,i+Int32(1),j,k,Val(1),
    )

    # 3. 激波传感器
    @inbounds ϕx = max(ϕ[i-2, j, k], ϕ[i-1, j, k], ϕ[i, j, k], ϕ[i+1, j, k], ϕ[i+2, j, k], ϕ[i+3, j, k])

    # ...
    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    # ==============================
    # 分支 A: 光滑???(极速模???
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[i]
    if ϕx < hybrid_ϕ1
        α_adapt = min(ϕx / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        # Left-state weights
        @inbounds L1 = stencil_arr[i,1] + α_adapt * Δstencil_arr[i,1]; @inbounds L2 = stencil_arr[i,2] + α_adapt * Δstencil_arr[i,2]
        @inbounds L3 = stencil_arr[i,3] + α_adapt * Δstencil_arr[i,3]; @inbounds L4 = stencil_arr[i,4] + α_adapt * Δstencil_arr[i,4]
        @inbounds L5 = stencil_arr[i,5] + α_adapt * Δstencil_arr[i,5]; @inbounds L6 = stencil_arr[i,6] + α_adapt * Δstencil_arr[i,6]
        @inbounds L7 = stencil_arr[i,7] + α_adapt * Δstencil_arr[i,7]
        for n = 1:Ncons
            @inbounds v1 = _structured_state_component(U,Q,i-3,j,k,n); v2 = _structured_state_component(U,Q,i-2,j,k,n); v3 = _structured_state_component(U,Q,i-1,j,k,n)
            @inbounds v4 = _structured_state_component(U,Q,i,j,k,n); v5 = _structured_state_component(U,Q,i+1,j,k,n); v6 = _structured_state_component(U,Q,i+2,j,k,n); v7 = _structured_state_component(U,Q,i+3,j,k,n)
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        # Right-state weights (temporal register reuse)
        @inbounds L1 = stencil_R_arr[i,1] + α_adapt * Δstencil_R_arr[i,1]; @inbounds L2 = stencil_R_arr[i,2] + α_adapt * Δstencil_R_arr[i,2]
        @inbounds L3 = stencil_R_arr[i,3] + α_adapt * Δstencil_R_arr[i,3]; @inbounds L4 = stencil_R_arr[i,4] + α_adapt * Δstencil_R_arr[i,4]
        @inbounds L5 = stencil_R_arr[i,5] + α_adapt * Δstencil_R_arr[i,5]; @inbounds L6 = stencil_R_arr[i,6] + α_adapt * Δstencil_R_arr[i,6]
        @inbounds L7 = stencil_R_arr[i,7] + α_adapt * Δstencil_R_arr[i,7]
        for n = 1:Ncons
            @inbounds r1 = _structured_state_component(U,Q,i+4,j,k,n); r2 = _structured_state_component(U,Q,i+3,j,k,n); r3 = _structured_state_component(U,Q,i+2,j,k,n)
            @inbounds r4 = _structured_state_component(U,Q,i+1,j,k,n); r5 = _structured_state_component(U,Q,i,j,k,n); r6 = _structured_state_component(U,Q,i-1,j,k,n); r7 = _structured_state_component(U,Q,i-2,j,k,n)
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end


    # ==============================
    # 分支 B: 间断???(特征分解模式)
    # ==============================
    else 
        # 0. Safety Check for uninitialized/vacuum states
        @inbounds ρL = Q[i, j, k, 1]; ρR = Q[i+1, j, k, 1]
        if ρL < FT(1.0e-10) || ρR < FT(1.0e-10)
            @inbounds for n = 1:Ncons
                Fx[i-NG+1,j-NG+tangential_halo,
                   k-NG+tangential_halo,n] = zero(FT)
            end
            return
        end

        @inbounds uL = Q[i, j, k, 2]; uR = Q[i+1, j, k, 2]
        @inbounds vL = Q[i, j, k, 3]; vR = Q[i+1, j, k, 3]
        @inbounds wL = Q[i, j, k, 4]; wR = Q[i+1, j, k, 4]
        @inbounds pL = Q[i, j, k, 5]; pR = Q[i+1, j, k, 5]

        sqρL = sqrt(ρL); sqρR = sqrt(ρR)
        inv_sq = one(FT) / (sqρL + sqρR)

        u = (sqρL * uL + sqρR * uR) * inv_sq
        v = (sqρL * vL + sqρR * vR) * inv_sq
        w = (sqρL * wL + sqρR * wR) * inv_sq
        
        HL = γ/(γ-one(FT))*pL/ρL + FT(0.5)*(uL^2 + vL^2 + wL^2)
        HR = γ/(γ-one(FT))*pR/ρR + FT(0.5)*(uR^2 + vR^2 + wR^2)
        H = (sqρL * HL + sqρR * HR) * inv_sq
        
        v2 = FT(0.5)*(u^2 + v^2 + w^2)
        c = sqrt(max(FT(1.0e-10), (γ-one(FT))*(H - v2)))
        
        # Tangent vectors: Use more robust selection to avoid zero tangent for any normal orientation
        # If normal is roughly [0, 0, 1], then abs(nz) > abs(ny) and abs(nz) > abs(nx)
        if abs(nx) < FT(0.6e0) && abs(ny) < FT(0.6e0) # Normal is mostly Z-aligned
            den = sqrt(nx*nx + nz*nz + FT(1.0e-12)); lx = -nz / den; ly = zero(FT); lz = nx / den
        elseif abs(nx) < FT(0.6e0) && abs(nz) < FT(0.6e0) # Normal is mostly Y-aligned
            den = sqrt(nx*nx + ny*ny + FT(1.0e-12)); lx = -ny / den; ly = nx / den; lz = zero(FT)
        else # Normal is mostly X-aligned or diagonal
            den = sqrt(ny*ny + nz*nz + FT(1.0e-12)); lx = zero(FT); ly = -nz / den; lz = ny / den
        end
        mx = ny * lz - nz * ly; my = nz * lx - nx * lz; mz = nx * ly - ny * lx

        invc = one(FT)/c; invc2 = invc*invc
        K = γ - one(FT)
        Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
        Kv2 = K*v2*invc2; Kc2 = K*invc2
        un = u*nx + v*ny + w*nz; ul = u*lx + v*ly + w*lz; um = u*mx + v*my + w*mz
        un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
        half = FT(0.5); mhalf = -FT(0.5)

        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        @inbounds ss = FT(2.0)/(S[i+1, j, k] + S[i, j, k] + FT(1.0e-20))

        for n = 1:Ncons
            # 2a. 左特征向???L
            ln1=zero(FT); ln2=zero(FT); ln3=zero(FT); ln4=zero(FT); ln5=zero(FT)
            if n == 1; ln1 = half*(Kv2 + un_invc); ln2 = mhalf*(Ku + nx_invc); ln3 = mhalf*(Kv + ny_invc); ln4 = mhalf*(Kw + nz_invc); ln5 = half*Kc2
            elseif n == 2; ln1 = one(FT) - Kv2; ln2 = Ku; ln3 = Kv; ln4 = Kw; ln5 = -Kc2
            elseif n == 3; ln1 = half*(Kv2 - un_invc); ln2 = mhalf*(Ku - nx_invc); ln3 = mhalf*(Kv - ny_invc); ln4 = mhalf*(Kw - nz_invc); ln5 = half*Kc2
            elseif n == 4; ln1 = -ul; ln2 = lx; ln3 = ly; ln4 = lz; ln5 = zero(FT)
            else; ln1 = -um; ln2 = mx; ln3 = my; ln4 = mz; ln5 = zero(FT); end

            # 2b-L. Project U -> V (L side)
            @inbounds V1 = ln1*U[i-3,j,k,1] + ln2*U[i-3,j,k,2] + ln3*U[i-3,j,k,3] + ln4*U[i-3,j,k,4] + ln5*U[i-3,j,k,5]
            @inbounds V2 = ln1*U[i-2,j,k,1] + ln2*U[i-2,j,k,2] + ln3*U[i-2,j,k,3] + ln4*U[i-2,j,k,4] + ln5*U[i-2,j,k,5]
            @inbounds V3 = ln1*U[i-1,j,k,1] + ln2*U[i-1,j,k,2] + ln3*U[i-1,j,k,3] + ln4*U[i-1,j,k,4] + ln5*U[i-1,j,k,5]
            @inbounds V4 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V5 = ln1*U[i+1,j,k,1] + ln2*U[i+1,j,k,2] + ln3*U[i+1,j,k,3] + ln4*U[i+1,j,k,4] + ln5*U[i+1,j,k,5]
            @inbounds V6 = ln1*U[i+2,j,k,1] + ln2*U[i+2,j,k,2] + ln3*U[i+2,j,k,3] + ln4*U[i+2,j,k,4] + ln5*U[i+2,j,k,5]
            @inbounds V7 = ln1*U[i+3,j,k,1] + ln2*U[i+3,j,k,2] + ln3*U[i+3,j,k,3] + ln4*U[i+3,j,k,4] + ln5*U[i+3,j,k,5]

            valL = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valL=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valL=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valL=V4+FT(0.5)*minmod(V4-V3,V5-V4); end

            # 2b-R. Project U -> V (R side, reuse V1-V7)
            @inbounds V1 = ln1*U[i+4,j,k,1] + ln2*U[i+4,j,k,2] + ln3*U[i+4,j,k,3] + ln4*U[i+4,j,k,4] + ln5*U[i+4,j,k,5]
            @inbounds V2 = ln1*U[i+3,j,k,1] + ln2*U[i+3,j,k,2] + ln3*U[i+3,j,k,3] + ln4*U[i+3,j,k,4] + ln5*U[i+3,j,k,5]
            @inbounds V3 = ln1*U[i+2,j,k,1] + ln2*U[i+2,j,k,2] + ln3*U[i+2,j,k,3] + ln4*U[i+2,j,k,4] + ln5*U[i+2,j,k,5]
            @inbounds V4 = ln1*U[i+1,j,k,1] + ln2*U[i+1,j,k,2] + ln3*U[i+1,j,k,3] + ln4*U[i+1,j,k,4] + ln5*U[i+1,j,k,5]
            @inbounds V5 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V6 = ln1*U[i-1,j,k,1] + ln2*U[i-1,j,k,2] + ln3*U[i-1,j,k,3] + ln4*U[i-1,j,k,4] + ln5*U[i-1,j,k,5]
            @inbounds V7 = ln1*U[i-2,j,k,1] + ln2*U[i-2,j,k,2] + ln3*U[i-2,j,k,3] + ln4*U[i-2,j,k,4] + ln5*U[i-2,j,k,5]

            valR = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valR=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valR=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valR=V4-FT(0.5)*minmod(V4-V3,V4-V5); end

            rn1=zero(FT); rn2=zero(FT); rn3=zero(FT); rn4=zero(FT); rn5=zero(FT)
            if n == 1; rn1=one(FT); rn2=u-nx*c; rn3=v-ny*c; rn4=w-nz*c; rn5=H-un*c
            elseif n == 2; rn1=one(FT); rn2=u; rn3=v; rn4=w; rn5=v2
            elseif n == 3; rn1=one(FT); rn2=u+nx*c; rn3=v+ny*c; rn4=w+nz*c; rn5=H+un*c
            elseif n == 4; rn1=zero(FT); rn2=lx; rn3=ly; rn4=lz; rn5=ul
            else; rn1=zero(FT); rn2=mx; rn3=my; rn4=mz; rn5=um; end
            
            rn = SVector{5, FT}(rn1, rn2, rn3, rn4, rn5)
            for m = 1:Ncons
                UL_final[m] += rn[m] * valL; UR_final[m] += rn[m] * valR
            end
        end
    end

    # Failsafe: if reconstruction produces a non-physical state, fall back to 1st order.
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    @static if equation_type == :MHD
        _B2L = UL_final[6]^2 + UL_final[7]^2 + UL_final[8]^2
        _B2R = UR_final[6]^2 + UR_final[7]^2 + UR_final[8]^2
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT)) -
            FT(0.5) * _B2L * INV_MU0_SI
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT)) -
            FT(0.5) * _B2R * INV_MU0_SI
    else
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    end
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = _structured_state_component(U,Q,i,j,k,n); end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = _structured_state_component(U,Q,i+1,j,k,n); end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})


    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)

    @inbounds for n = 1:Ncons
        Fx[i-NG+1,j-NG+tangential_halo,
           k-NG+tangential_halo,n] = flux_temp[n]*Area
    end
    return
end

function Eigen_reconstruct_j(Q, U, ϕ, S, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp,
                             stencil_arr, Δstencil_arr, lin_phi_arr,
                             stencil_R_arr, Δstencil_R_arr,
                             ch_glm::FT, mode::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. 边界检查(J接口)
    tangential_halo = STRUCTURED_FLUX_TANGENTIAL_HALO
    if i > nxp+NG+tangential_halo || j > nyp+NG ||
       k > nzp+NG+tangential_halo ||
       i < NG+Int32(1)-tangential_halo || j < NG ||
       k < NG+Int32(1)-tangential_halo
        return
    end
    # Interior/boundary mode filter (stencil width=4 in j)
    if mode == Int32(1) && (j < NG+Int32(4) || j > nyp+NG-Int32(4)); return; end
    if mode == Int32(2) && (j >= NG+Int32(4) && j <= nyp+NG-Int32(4)); return; end

    # 2. Geometry
    Area,nx,ny,nz = structured_face_geometry(
        Areaj,nxj,nyj,nzj,i,j+Int32(1),k,Val(2),
    )

    # 3. 激波传感器 (J 方向)
    @inbounds ϕx = max(ϕ[i, j-2, k], ϕ[i, j-1, k], ϕ[i, j, k], ϕ[i, j+1, k], ϕ[i, j+2, k], ϕ[i, j+3, k])

    # ...
    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # 分支 A: 光滑区(极速模式)
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    if ϕx < hybrid_ϕ1
        # ═══ Original 1D 7-point reconstruction ═══
        α_adapt = min(ϕx / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        @inbounds L1 = stencil_arr[j,k,1] + α_adapt * Δstencil_arr[j,k,1]; @inbounds L2 = stencil_arr[j,k,2] + α_adapt * Δstencil_arr[j,k,2]
        @inbounds L3 = stencil_arr[j,k,3] + α_adapt * Δstencil_arr[j,k,3]; @inbounds L4 = stencil_arr[j,k,4] + α_adapt * Δstencil_arr[j,k,4]
        @inbounds L5 = stencil_arr[j,k,5] + α_adapt * Δstencil_arr[j,k,5]; @inbounds L6 = stencil_arr[j,k,6] + α_adapt * Δstencil_arr[j,k,6]
        @inbounds L7 = stencil_arr[j,k,7] + α_adapt * Δstencil_arr[j,k,7]
        for n = 1:Ncons
            @inbounds v1 = _structured_state_component(U,Q,i,j-3,k,n); v2 = _structured_state_component(U,Q,i,j-2,k,n); v3 = _structured_state_component(U,Q,i,j-1,k,n)
            @inbounds v4 = _structured_state_component(U,Q,i,j,k,n); v5 = _structured_state_component(U,Q,i,j+1,k,n); v6 = _structured_state_component(U,Q,i,j+2,k,n); v7 = _structured_state_component(U,Q,i,j+3,k,n)
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        @inbounds L1 = stencil_R_arr[j,k,1] + α_adapt * Δstencil_R_arr[j,k,1]; @inbounds L2 = stencil_R_arr[j,k,2] + α_adapt * Δstencil_R_arr[j,k,2]
        @inbounds L3 = stencil_R_arr[j,k,3] + α_adapt * Δstencil_R_arr[j,k,3]; @inbounds L4 = stencil_R_arr[j,k,4] + α_adapt * Δstencil_R_arr[j,k,4]
        @inbounds L5 = stencil_R_arr[j,k,5] + α_adapt * Δstencil_R_arr[j,k,5]; @inbounds L6 = stencil_R_arr[j,k,6] + α_adapt * Δstencil_R_arr[j,k,6]
        @inbounds L7 = stencil_R_arr[j,k,7] + α_adapt * Δstencil_R_arr[j,k,7]
        for n = 1:Ncons
            @inbounds r1 = _structured_state_component(U,Q,i,j+4,k,n); r2 = _structured_state_component(U,Q,i,j+3,k,n); r3 = _structured_state_component(U,Q,i,j+2,k,n)
            @inbounds r4 = _structured_state_component(U,Q,i,j+1,k,n); r5 = _structured_state_component(U,Q,i,j,k,n); r6 = _structured_state_component(U,Q,i,j-1,k,n); r7 = _structured_state_component(U,Q,i,j-2,k,n)
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end

        # Failsafe check for Branch A
        @inbounds ρL_fs = UL_final[1]; @inbounds ρR_fs = UR_final[1]
        @inbounds eiL_fs = UL_final[5] - FT(0.5)*(UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2)/ρL_fs
        @inbounds eiR_fs = UR_final[5] - FT(0.5)*(UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2)/ρR_fs
        if ρL_fs < FT(1.0e-10) || eiL_fs < FT(1.0e-10) || ρR_fs < FT(1.0e-10) || eiR_fs < FT(1.0e-10) || !isfinite(ρL_fs) || !isfinite(ρR_fs) || !isfinite(eiL_fs) || !isfinite(eiR_fs)
            for n = 1:Ncons
                @inbounds UL_final[n] = _structured_state_component(U,Q,i,j,k,n)
                @inbounds UR_final[n] = _structured_state_component(U,Q,i,j+1,k,n)
            end
        end


    # ==============================
    # 分支 B: 间断区(特征分解模式)
    # ==============================
    else 
        # 0. Safety Check for uninitialized/vacuum states
        @inbounds ρL = Q[i, j, k, 1]; ρR = Q[i, j+1, k, 1]
        if ρL < FT(1.0e-10) || ρR < FT(1.0e-10)
            @inbounds for n = 1:Ncons
                Fy[i-NG+tangential_halo,j-NG+1,
                   k-NG+tangential_halo,n] = zero(FT)
            end
            return
        end

        # Roe Averages
        @inbounds uL = Q[i, j, k, 2]; uR = Q[i, j+1, k, 2]
        @inbounds vL = Q[i, j, k, 3]; vR = Q[i, j+1, k, 3]
        @inbounds wL = Q[i, j, k, 4]; wR = Q[i, j+1, k, 4]
        @inbounds pL = Q[i, j, k, 5]; pR = Q[i, j+1, k, 5]

        sqρL = sqrt(ρL); sqρR = sqrt(ρR)
        inv_sq = one(FT) / (sqρL + sqρR)

        u = (sqρL * uL + sqρR * uR) * inv_sq
        v = (sqρL * vL + sqρR * vR) * inv_sq
        w = (sqρL * wL + sqρR * wR) * inv_sq
        
        HL = γ/(γ-one(FT))*pL/ρL + FT(0.5)*(uL^2 + vL^2 + wL^2)
        HR = γ/(γ-one(FT))*pR/ρR + FT(0.5)*(uR^2 + vR^2 + wR^2)
        H = (sqρL * HL + sqρR * HR) * inv_sq
        
        v2 = FT(0.5)*(u^2 + v^2 + w^2)
        c = sqrt(max(FT(1.0e-10), (γ-one(FT))*(H - v2)))
        
        if abs(nx) < FT(0.6e0) && abs(ny) < FT(0.6e0)
            den = sqrt(nx*nx + nz*nz + FT(1.0e-12)); lx = -nz / den; ly = zero(FT); lz = nx / den
        elseif abs(nx) < FT(0.6e0) && abs(nz) < FT(0.6e0)
            den = sqrt(nx*nx + ny*ny + FT(1.0e-12)); lx = -ny / den; ly = nx / den; lz = zero(FT)
        else
            den = sqrt(ny*ny + nz*nz + FT(1.0e-12)); lx = zero(FT); ly = -nz / den; lz = ny / den
        end
        mx = ny * lz - nz * ly; my = nz * lx - nx * lz; mz = nx * ly - ny * lx

        invc = one(FT)/c; invc2 = invc*invc
        K = γ - one(FT)
        Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
        Kv2 = K*v2*invc2; Kc2 = K*invc2
        un = u*nx + v*ny + w*nz; ul = u*lx + v*ly + w*lz; um = u*mx + v*my + w*mz
        un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
        half = FT(0.5); mhalf = -FT(0.5)

        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        @inbounds ss = FT(2.0)/(S[i, j+1, k] + S[i, j, k] + FT(1.0e-20))

        for n = 1:Ncons
            ln1=zero(FT); ln2=zero(FT); ln3=zero(FT); ln4=zero(FT); ln5=zero(FT)
            if n == 1; ln1 = half*(Kv2 + un_invc); ln2 = mhalf*(Ku + nx_invc); ln3 = mhalf*(Kv + ny_invc); ln4 = mhalf*(Kw + nz_invc); ln5 = half*Kc2
            elseif n == 2; ln1 = one(FT) - Kv2; ln2 = Ku; ln3 = Kv; ln4 = Kw; ln5 = -Kc2
            elseif n == 3; ln1 = half*(Kv2 - un_invc); ln2 = mhalf*(Ku - nx_invc); ln3 = mhalf*(Kv - ny_invc); ln4 = mhalf*(Kw - nz_invc); ln5 = half*Kc2
            elseif n == 4; ln1 = -ul; ln2 = lx; ln3 = ly; ln4 = lz; ln5 = zero(FT)
            else; ln1 = -um; ln2 = mx; ln3 = my; ln4 = mz; ln5 = zero(FT); end

            # 2b-L. Project U -> V (L side)
            @inbounds V1 = ln1*U[i,j-3,k,1] + ln2*U[i,j-3,k,2] + ln3*U[i,j-3,k,3] + ln4*U[i,j-3,k,4] + ln5*U[i,j-3,k,5]
            @inbounds V2 = ln1*U[i,j-2,k,1] + ln2*U[i,j-2,k,2] + ln3*U[i,j-2,k,3] + ln4*U[i,j-2,k,4] + ln5*U[i,j-2,k,5]
            @inbounds V3 = ln1*U[i,j-1,k,1] + ln2*U[i,j-1,k,2] + ln3*U[i,j-1,k,3] + ln4*U[i,j-1,k,4] + ln5*U[i,j-1,k,5]
            @inbounds V4 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V5 = ln1*U[i,j+1,k,1] + ln2*U[i,j+1,k,2] + ln3*U[i,j+1,k,3] + ln4*U[i,j+1,k,4] + ln5*U[i,j+1,k,5]
            @inbounds V6 = ln1*U[i,j+2,k,1] + ln2*U[i,j+2,k,2] + ln3*U[i,j+2,k,3] + ln4*U[i,j+2,k,4] + ln5*U[i,j+2,k,5]
            @inbounds V7 = ln1*U[i,j+3,k,1] + ln2*U[i,j+3,k,2] + ln3*U[i,j+3,k,3] + ln4*U[i,j+3,k,4] + ln5*U[i,j+3,k,5]

            valL = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valL=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valL=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valL=V4+FT(0.5)*minmod(V4-V3,V5-V4); end

            # 2b-R. Project U -> V (R side, reuse V1-V7)
            @inbounds V1 = ln1*U[i,j+4,k,1] + ln2*U[i,j+4,k,2] + ln3*U[i,j+4,k,3] + ln4*U[i,j+4,k,4] + ln5*U[i,j+4,k,5]
            @inbounds V2 = ln1*U[i,j+3,k,1] + ln2*U[i,j+3,k,2] + ln3*U[i,j+3,k,3] + ln4*U[i,j+3,k,4] + ln5*U[i,j+3,k,5]
            @inbounds V3 = ln1*U[i,j+2,k,1] + ln2*U[i,j+2,k,2] + ln3*U[i,j+2,k,3] + ln4*U[i,j+2,k,4] + ln5*U[i,j+2,k,5]
            @inbounds V4 = ln1*U[i,j+1,k,1] + ln2*U[i,j+1,k,2] + ln3*U[i,j+1,k,3] + ln4*U[i,j+1,k,4] + ln5*U[i,j+1,k,5]
            @inbounds V5 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V6 = ln1*U[i,j-1,k,1] + ln2*U[i,j-1,k,2] + ln3*U[i,j-1,k,3] + ln4*U[i,j-1,k,4] + ln5*U[i,j-1,k,5]
            @inbounds V7 = ln1*U[i,j-2,k,1] + ln2*U[i,j-2,k,2] + ln3*U[i,j-2,k,3] + ln4*U[i,j-2,k,4] + ln5*U[i,j-2,k,5]

            valR = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valR=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valR=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valR=V4-FT(0.5)*minmod(V4-V3,V4-V5); end

            rn1=zero(FT); rn2=zero(FT); rn3=zero(FT); rn4=zero(FT); rn5=zero(FT)
            if n == 1; rn1=one(FT); rn2=u-nx*c; rn3=v-ny*c; rn4=w-nz*c; rn5=H-un*c
            elseif n == 2; rn1=one(FT); rn2=u; rn3=v; rn4=w; rn5=v2
            elseif n == 3; rn1=one(FT); rn2=u+nx*c; rn3=v+ny*c; rn4=w+nz*c; rn5=H+un*c
            elseif n == 4; rn1=zero(FT); rn2=lx; rn3=ly; rn4=lz; rn5=ul
            else; rn1=zero(FT); rn2=mx; rn3=my; rn4=mz; rn5=um; end
            
            rn = SVector{5, FT}(rn1, rn2, rn3, rn4, rn5)
            for m = 1:Ncons
                UL_final[m] += rn[m] * valL; UR_final[m] += rn[m] * valR
            end
        end
    end

    # Failsafe: if reconstruction produces a non-physical state, fall back to 1st order.
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    @static if equation_type == :MHD
        _B2L = UL_final[6]^2 + UL_final[7]^2 + UL_final[8]^2
        _B2R = UR_final[6]^2 + UR_final[7]^2 + UR_final[8]^2
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT)) -
            FT(0.5) * _B2L * INV_MU0_SI
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT)) -
            FT(0.5) * _B2R * INV_MU0_SI
    else
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    end
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = _structured_state_component(U,Q,i,j,k,n); end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = _structured_state_component(U,Q,i,j+1,k,n); end
    end

    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})


    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)

    @inbounds for n = 1:Ncons
        Fy[i-NG+tangential_halo,j-NG+1,
           k-NG+tangential_halo,n] = flux_temp[n]*Area
    end
    return
end

function Eigen_reconstruct_k(Q, U, ϕ, S, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp,
                             stencil_arr, Δstencil_arr, lin_phi_arr,
                             stencil_R_arr, Δstencil_R_arr,
                             ch_glm::FT, mode::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. 边界检查(K接口)
    tangential_halo = STRUCTURED_FLUX_TANGENTIAL_HALO
    if i > nxp+NG+tangential_halo ||
       j > nyp+NG+tangential_halo || k > nzp+NG ||
       i < NG+Int32(1)-tangential_halo ||
       j < NG+Int32(1)-tangential_halo || k < NG
        return
    end


    # Interior/boundary mode filter (stencil width=4 in k)
    if mode == Int32(1) && (k < NG+Int32(4) || k > nzp+NG-Int32(4)); return; end
    if mode == Int32(2) && (k >= NG+Int32(4) && k <= nzp+NG-Int32(4)); return; end

    # 2. Geometry
    Area,nx,ny,nz = structured_face_geometry(
        Areak,nxk,nyk,nzk,i,j,k+Int32(1),Val(3),
    )

    # 3. 激波传感器 (K 方向)
    @inbounds ϕx = max(ϕ[i, j, k-2], ϕ[i, j, k-1], ϕ[i, j, k], ϕ[i, j, k+1], ϕ[i, j, k+2], ϕ[i, j, k+3])

    # ...
    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # 分支 A: 光滑区(极速模式)
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    if ϕx < hybrid_ϕ1
        # ═══ Original 1D 7-point reconstruction ═══
        α_adapt = min(ϕx / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        @inbounds L1 = stencil_arr[j,k,1] + α_adapt * Δstencil_arr[j,k,1]; @inbounds L2 = stencil_arr[j,k,2] + α_adapt * Δstencil_arr[j,k,2]
        @inbounds L3 = stencil_arr[j,k,3] + α_adapt * Δstencil_arr[j,k,3]; @inbounds L4 = stencil_arr[j,k,4] + α_adapt * Δstencil_arr[j,k,4]
        @inbounds L5 = stencil_arr[j,k,5] + α_adapt * Δstencil_arr[j,k,5]; @inbounds L6 = stencil_arr[j,k,6] + α_adapt * Δstencil_arr[j,k,6]
        @inbounds L7 = stencil_arr[j,k,7] + α_adapt * Δstencil_arr[j,k,7]
        for n = 1:Ncons
            @inbounds v1 = _structured_state_component(U,Q,i,j,k-3,n); v2 = _structured_state_component(U,Q,i,j,k-2,n); v3 = _structured_state_component(U,Q,i,j,k-1,n)
            @inbounds v4 = _structured_state_component(U,Q,i,j,k,n); v5 = _structured_state_component(U,Q,i,j,k+1,n); v6 = _structured_state_component(U,Q,i,j,k+2,n); v7 = _structured_state_component(U,Q,i,j,k+3,n)
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        @inbounds L1 = stencil_R_arr[j,k,1] + α_adapt * Δstencil_R_arr[j,k,1]; @inbounds L2 = stencil_R_arr[j,k,2] + α_adapt * Δstencil_R_arr[j,k,2]
        @inbounds L3 = stencil_R_arr[j,k,3] + α_adapt * Δstencil_R_arr[j,k,3]; @inbounds L4 = stencil_R_arr[j,k,4] + α_adapt * Δstencil_R_arr[j,k,4]
        @inbounds L5 = stencil_R_arr[j,k,5] + α_adapt * Δstencil_R_arr[j,k,5]; @inbounds L6 = stencil_R_arr[j,k,6] + α_adapt * Δstencil_R_arr[j,k,6]
        @inbounds L7 = stencil_R_arr[j,k,7] + α_adapt * Δstencil_R_arr[j,k,7]
        for n = 1:Ncons
            @inbounds r1 = _structured_state_component(U,Q,i,j,k+4,n); r2 = _structured_state_component(U,Q,i,j,k+3,n); r3 = _structured_state_component(U,Q,i,j,k+2,n)
            @inbounds r4 = _structured_state_component(U,Q,i,j,k+1,n); r5 = _structured_state_component(U,Q,i,j,k,n); r6 = _structured_state_component(U,Q,i,j,k-1,n); r7 = _structured_state_component(U,Q,i,j,k-2,n)
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end

        # Failsafe check for Branch A
        @inbounds ρL_fs = UL_final[1]; @inbounds ρR_fs = UR_final[1]
        @inbounds eiL_fs = UL_final[5] - FT(0.5)*(UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2)/ρL_fs
        @inbounds eiR_fs = UR_final[5] - FT(0.5)*(UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2)/ρR_fs
        if ρL_fs < FT(1.0e-10) || eiL_fs < FT(1.0e-10) || ρR_fs < FT(1.0e-10) || eiR_fs < FT(1.0e-10) || !isfinite(ρL_fs) || !isfinite(ρR_fs) || !isfinite(eiL_fs) || !isfinite(eiR_fs)
            for n = 1:Ncons
                @inbounds UL_final[n] = _structured_state_component(U,Q,i,j,k,n)
                @inbounds UR_final[n] = _structured_state_component(U,Q,i,j,k+1,n)
            end
        end

    # ==============================
    # 分支 B: 间断区(特征分解模式)
    # ==============================
    else 
        # 0. Safety Check for uninitialized/vacuum states
        @inbounds ρL = Q[i, j, k, 1]; ρR = Q[i, j, k+1, 1]
        if ρL < FT(1.0e-10) || ρR < FT(1.0e-10)
            @inbounds for n = 1:Ncons
                Fz[i-NG+tangential_halo,j-NG+tangential_halo,
                   k-NG+1,n] = zero(FT)
            end
            return
        end

        # Roe Averages
        @inbounds uL = Q[i, j, k, 2]; uR = Q[i, j, k+1, 2]
        @inbounds vL = Q[i, j, k, 3]; vR = Q[i, j, k+1, 3]
        @inbounds wL = Q[i, j, k, 4]; wR = Q[i, j, k+1, 4]
        @inbounds pL = Q[i, j, k, 5]; pR = Q[i, j, k+1, 5]

        sqρL = sqrt(ρL); sqρR = sqrt(ρR)
        inv_sq = one(FT) / (sqρL + sqρR)

        u = (sqρL * uL + sqρR * uR) * inv_sq
        v = (sqρL * vL + sqρR * vR) * inv_sq
        w = (sqρL * wL + sqρR * wR) * inv_sq
        
        HL = γ/(γ-one(FT))*pL/ρL + FT(0.5)*(uL^2 + vL^2 + wL^2)
        HR = γ/(γ-one(FT))*pR/ρR + FT(0.5)*(uR^2 + vR^2 + wR^2)
        H = (sqρL * HL + sqρR * HR) * inv_sq
        
        v2 = FT(0.5)*(u^2 + v^2 + w^2)
        c = sqrt(max(FT(1.0e-10), (γ-one(FT))*(H - v2)))
        
        if abs(nx) < FT(0.6e0) && abs(ny) < FT(0.6e0)
            den = sqrt(nx*nx + nz*nz + FT(1.0e-12)); lx = -nz / den; ly = zero(FT); lz = nx / den
        elseif abs(nx) < FT(0.6e0) && abs(nz) < FT(0.6e0)
            den = sqrt(nx*nx + ny*ny + FT(1.0e-12)); lx = -ny / den; ly = nx / den; lz = zero(FT)
        else
            den = sqrt(ny*ny + nz*nz + FT(1.0e-12)); lx = zero(FT); ly = -nz / den; lz = ny / den
        end
        mx = ny * lz - nz * ly; my = nz * lx - nx * lz; mz = nx * ly - ny * lx

        invc = one(FT)/c; invc2 = invc*invc
        K = γ - one(FT)
        Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
        Kv2 = K*v2*invc2; Kc2 = K*invc2
        un = u*nx + v*ny + w*nz; ul = u*lx + v*ly + w*lz; um = u*mx + v*my + w*mz
        un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
        half = FT(0.5); mhalf = -FT(0.5)

        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        @inbounds ss = FT(2.0)/(S[i, j, k+1] + S[i, j, k] + FT(1.0e-20))

        for n = 1:Ncons
            ln1=zero(FT); ln2=zero(FT); ln3=zero(FT); ln4=zero(FT); ln5=zero(FT)
            if n == 1; ln1 = half*(Kv2 + un_invc); ln2 = mhalf*(Ku + nx_invc); ln3 = mhalf*(Kv + ny_invc); ln4 = mhalf*(Kw + nz_invc); ln5 = half*Kc2
            elseif n == 2; ln1 = one(FT) - Kv2; ln2 = Ku; ln3 = Kv; ln4 = Kw; ln5 = -Kc2
            elseif n == 3; ln1 = half*(Kv2 - un_invc); ln2 = mhalf*(Ku - nx_invc); ln3 = mhalf*(Kv - ny_invc); ln4 = mhalf*(Kw - nz_invc); ln5 = half*Kc2
            elseif n == 4; ln1 = -ul; ln2 = lx; ln3 = ly; ln4 = lz; ln5 = zero(FT)
            else; ln1 = -um; ln2 = mx; ln3 = my; ln4 = mz; ln5 = zero(FT); end

            # 2b-L. Project U -> V (L side)
            @inbounds V1 = ln1*U[i,j,k-3,1] + ln2*U[i,j,k-3,2] + ln3*U[i,j,k-3,3] + ln4*U[i,j,k-3,4] + ln5*U[i,j,k-3,5]
            @inbounds V2 = ln1*U[i,j,k-2,1] + ln2*U[i,j,k-2,2] + ln3*U[i,j,k-2,3] + ln4*U[i,j,k-2,4] + ln5*U[i,j,k-2,5]
            @inbounds V3 = ln1*U[i,j,k-1,1] + ln2*U[i,j,k-1,2] + ln3*U[i,j,k-1,3] + ln4*U[i,j,k-1,4] + ln5*U[i,j,k-1,5]
            @inbounds V4 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V5 = ln1*U[i,j,k+1,1] + ln2*U[i,j,k+1,2] + ln3*U[i,j,k+1,3] + ln4*U[i,j,k+1,4] + ln5*U[i,j,k+1,5]
            @inbounds V6 = ln1*U[i,j,k+2,1] + ln2*U[i,j,k+2,2] + ln3*U[i,j,k+2,3] + ln4*U[i,j,k+2,4] + ln5*U[i,j,k+2,5]
            @inbounds V7 = ln1*U[i,j,k+3,1] + ln2*U[i,j,k+3,2] + ln3*U[i,j,k+3,3] + ln4*U[i,j,k+3,4] + ln5*U[i,j,k+3,5]

            valL = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valL=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valL=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valL=V4+FT(0.5)*minmod(V4-V3,V5-V4); end

            # 2b-R. Project U -> V (R side, reuse V1-V7)
            @inbounds V1 = ln1*U[i,j,k+4,1] + ln2*U[i,j,k+4,2] + ln3*U[i,j,k+4,3] + ln4*U[i,j,k+4,4] + ln5*U[i,j,k+4,5]
            @inbounds V2 = ln1*U[i,j,k+3,1] + ln2*U[i,j,k+3,2] + ln3*U[i,j,k+3,3] + ln4*U[i,j,k+3,4] + ln5*U[i,j,k+3,5]
            @inbounds V3 = ln1*U[i,j,k+2,1] + ln2*U[i,j,k+2,2] + ln3*U[i,j,k+2,3] + ln4*U[i,j,k+2,4] + ln5*U[i,j,k+2,5]
            @inbounds V4 = ln1*U[i,j,k+1,1] + ln2*U[i,j,k+1,2] + ln3*U[i,j,k+1,3] + ln4*U[i,j,k+1,4] + ln5*U[i,j,k+1,5]
            @inbounds V5 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V6 = ln1*U[i,j,k-1,1] + ln2*U[i,j,k-1,2] + ln3*U[i,j,k-1,3] + ln4*U[i,j,k-1,4] + ln5*U[i,j,k-1,5]
            @inbounds V7 = ln1*U[i,j,k-2,1] + ln2*U[i,j,k-2,2] + ln3*U[i,j,k-2,3] + ln4*U[i,j,k-2,4] + ln5*U[i,j,k-2,5]

            valR = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valR=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valR=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valR=V4-FT(0.5)*minmod(V4-V3,V4-V5); end

            rn1=zero(FT); rn2=zero(FT); rn3=zero(FT); rn4=zero(FT); rn5=zero(FT)
            if n == 1; rn1=one(FT); rn2=u-nx*c; rn3=v-ny*c; rn4=w-nz*c; rn5=H-un*c
            elseif n == 2; rn1=one(FT); rn2=u; rn3=v; rn4=w; rn5=v2
            elseif n == 3; rn1=one(FT); rn2=u+nx*c; rn3=v+ny*c; rn4=w+nz*c; rn5=H+un*c
            elseif n == 4; rn1=zero(FT); rn2=lx; rn3=ly; rn4=lz; rn5=ul
            else; rn1=zero(FT); rn2=mx; rn3=my; rn4=mz; rn5=um; end
            
            rn = SVector{5, FT}(rn1, rn2, rn3, rn4, rn5)
            for m = 1:Ncons
                UL_final[m] += rn[m] * valL; UR_final[m] += rn[m] * valR
            end
        end
    end

    # Failsafe: if reconstruction produces a non-physical state, fall back to 1st order.
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    @static if equation_type == :MHD
        _B2L = UL_final[6]^2 + UL_final[7]^2 + UL_final[8]^2
        _B2R = UR_final[6]^2 + UR_final[7]^2 + UR_final[8]^2
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT)) -
            FT(0.5) * _B2L * INV_MU0_SI
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT)) -
            FT(0.5) * _B2R * INV_MU0_SI
    else
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    end
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = _structured_state_component(U,Q,i,j,k,n); end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = _structured_state_component(U,Q,i,j,k+1,n); end
    end

    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})


    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)

    @inbounds for n = 1:Ncons
        Fz[i-NG+tangential_halo,j-NG+tangential_halo,
           k-NG+1,n] = flux_temp[n]*Area
    end
end

function Conser_reconstruct_i(Q, U, ϕ, S, Fx, rho_sum_x, Areai, nxi, nyi, nzi, nxp, nyp, nzp,
                              stencil_arr, Δstencil_arr, lin_phi_arr,
                              stencil_R_arr, Δstencil_R_arr,
                              ch_glm::FT, mode::Int32, Bx_face_CT,
                              cache_i, Vol, dt_stage::FT,
                              rk_stage::Int32, pos_meta, pos_values,
                              B0x_face_CT=nothing, B0_cell_CT=nothing)
    @static if strict_ct_positivity && ct_mode
        _use_primitive_reconstruction = splitMethodID == Int32(4)
    else
        _use_primitive_reconstruction = false
    end
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. Boundary check. The WENO7 edge cache spans the full Q halo; only the
    # finite-volume flux is stored in the configured tangential halo.
    tangential_halo = STRUCTURED_FLUX_TANGENTIAL_HALO
    @static if ct_mode && @isdefined(ct_emf_scheme) &&
               ct_emf_scheme == CT_EMF_WENO7_SG07
            if i > nxp+NG || j > nyp+Int32(2)*NG ||
               k > nzp+Int32(2)*NG || i < NG || j < Int32(1) || k < Int32(1)
                return
            end
    else
        if i > nxp+NG || j > nyp+NG+tangential_halo ||
           k > nzp+NG+tangential_halo || i < NG ||
           j < NG+Int32(1)-tangential_halo ||
           k < NG+Int32(1)-tangential_halo
            return
        end
    end
    # Interior/boundary mode filter (stencil width=4 in i)
    if mode == Int32(1) && (i < NG+Int32(4) || i > nxp+NG-Int32(4)); return; end
    if mode == Int32(2) && (i >= NG+Int32(4) && i <= nxp+NG-Int32(4)); return; end
    # 2. Geometry
    @static if ct_mode
        Area,nx,ny,nz,_Bn_geometry = structured_ct_face_geometry_bn(
            Bx_face_CT,B0x_face_CT,Areai,nxi,nyi,nzi,
            i+Int32(1),j,k,Val(1),Q,
        )
        _B0n_geometry = zero(FT)
        if B0_cell_CT !== nothing
            _,_,_,_,_B0n_geometry = structured_ct_face_geometry_bn(
                B0x_face_CT,nothing,Areai,nxi,nyi,nzi,
                i+Int32(1),j,k,Val(1),nothing,CT_FACE_BN_POINT6,
            )
        end
    else
        Area,nx,ny,nz = structured_face_geometry(
            Areai,nxi,nyi,nzi,i+Int32(1),j,k,Val(1),
        )
    end
    # @inbounds Area *= INV_SCALE_FACTOR
    # @cuprintf("nx=%e, ny=%e, nz=%e, Area=%e\n", nx, ny, nz, Area)

    # 3. 激波传感器
    @inbounds ϕx = max(ϕ[i-2, j, k], ϕ[i-1, j, k], ϕ[i, j, k], ϕ[i+1, j, k], ϕ[i+2, j, k], ϕ[i+3, j, k])

    # ...
    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # 分支 A: 光滑???(极速模???
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[i]
    if _use_primitive_reconstruction
        if eigen_reconstruction
            _bi_plm, _bj_plm, _bk_plm = ct_face_index(Val(6), i, j, k)
            _bn_plm = _Bn_geometry
            _wm = ct_primitive_state(Q, i-1, j, k)
            _wc = ct_primitive_state(Q, i,   j, k)
            _wp = ct_primitive_state(Q, i+1, j, k)
            _wl = ct_mhd_characteristic_plm_plus(
                _wm, _wc, _wp, nx, ny, nz, _bn_plm, FT(γ),
            )
            for n = 1:Ncons
                UL_final[n] = _wl[n]
            end
            _wpp = ct_primitive_state(Q, i+2, j, k)
            _wr = ct_mhd_characteristic_plm_minus(
                _wc, _wp, _wpp, nx, ny, nz, _bn_plm, FT(γ),
            )
            for n = 1:Ncons
                UR_final[n] = _wr[n]
            end
        else
            for n = 1:Ncons
                qn = ct_primitive_reconstruction_slot(n)
                @inbounds qm = Q[i-1,j,k,qn]
                @inbounds qc = Q[i  ,j,k,qn]
                @inbounds qp = Q[i+1,j,k,qn]
                @inbounds qpp = Q[i+2,j,k,qn]
                UL_final[n] = ct_plm_plus(qm, qc, qp)
                UR_final[n] = ct_plm_minus(qc, qp, qpp)
            end
        end
    elseif ϕx < hybrid_ϕ1
        α_adapt = min(ϕx / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        # Left-state weights
        @inbounds L1 = stencil_arr[i,1] + α_adapt * Δstencil_arr[i,1]; @inbounds L2 = stencil_arr[i,2] + α_adapt * Δstencil_arr[i,2]
        @inbounds L3 = stencil_arr[i,3] + α_adapt * Δstencil_arr[i,3]; @inbounds L4 = stencil_arr[i,4] + α_adapt * Δstencil_arr[i,4]
        @inbounds L5 = stencil_arr[i,5] + α_adapt * Δstencil_arr[i,5]; @inbounds L6 = stencil_arr[i,6] + α_adapt * Δstencil_arr[i,6]
        @inbounds L7 = stencil_arr[i,7] + α_adapt * Δstencil_arr[i,7]
        for n = 1:Ncons
            @inbounds v1 = _structured_state_component(U,Q,i-3,j,k,n); v2 = _structured_state_component(U,Q,i-2,j,k,n); v3 = _structured_state_component(U,Q,i-1,j,k,n)
            @inbounds v4 = _structured_state_component(U,Q,i,j,k,n); v5 = _structured_state_component(U,Q,i+1,j,k,n); v6 = _structured_state_component(U,Q,i+2,j,k,n); v7 = _structured_state_component(U,Q,i+3,j,k,n)
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        # Right-state weights (temporal register reuse)
        @inbounds L1 = stencil_R_arr[i,1] + α_adapt * Δstencil_R_arr[i,1]; @inbounds L2 = stencil_R_arr[i,2] + α_adapt * Δstencil_R_arr[i,2]
        @inbounds L3 = stencil_R_arr[i,3] + α_adapt * Δstencil_R_arr[i,3]; @inbounds L4 = stencil_R_arr[i,4] + α_adapt * Δstencil_R_arr[i,4]
        @inbounds L5 = stencil_R_arr[i,5] + α_adapt * Δstencil_R_arr[i,5]; @inbounds L6 = stencil_R_arr[i,6] + α_adapt * Δstencil_R_arr[i,6]
        @inbounds L7 = stencil_R_arr[i,7] + α_adapt * Δstencil_R_arr[i,7]
        for n = 1:Ncons
            @inbounds r1 = _structured_state_component(U,Q,i+4,j,k,n); r2 = _structured_state_component(U,Q,i+3,j,k,n); r3 = _structured_state_component(U,Q,i+2,j,k,n)
            @inbounds r4 = _structured_state_component(U,Q,i+1,j,k,n); r5 = _structured_state_component(U,Q,i,j,k,n); r6 = _structured_state_component(U,Q,i-1,j,k,n); r7 = _structured_state_component(U,Q,i-2,j,k,n)
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end


    # ==============================
    # 分支 B: 间断???(特征分解模式)
    # ==============================
    else 
        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        @inbounds ss = FT(2.0)/(S[i+1, j, k] + S[i, j, k])

        for n = 1:Ncons

            # 2b. 投影 U -> V (Component-wise Reconstruction)
            @inbounds V1L = _structured_state_component(U,Q,i-3,j,k,n)
            @inbounds V2L = _structured_state_component(U,Q,i-2,j,k,n)
            @inbounds V3L = _structured_state_component(U,Q,i-1,j,k,n)
            @inbounds V4L = _structured_state_component(U,Q,i,j,k,n)
            @inbounds V5L = _structured_state_component(U,Q,i+1,j,k,n)
            @inbounds V6L = _structured_state_component(U,Q,i+2,j,k,n)
            @inbounds V7L = _structured_state_component(U,Q,i+3,j,k,n)

            @inbounds V1R = _structured_state_component(U,Q,i+4,j,k,n)
            @inbounds V2R = _structured_state_component(U,Q,i+3,j,k,n)
            @inbounds V3R = _structured_state_component(U,Q,i+2,j,k,n)
            @inbounds V4R = _structured_state_component(U,Q,i+1,j,k,n)
            @inbounds V5R = _structured_state_component(U,Q,i,j,k,n)
            @inbounds V6R = _structured_state_component(U,Q,i-1,j,k,n)
            @inbounds V7R = _structured_state_component(U,Q,i-2,j,k,n)

            valL = zero(FT); valR = zero(FT)

            # 2c. WENO Reconstruction
            if ϕx < hybrid_ϕ2 # WENO7
                samplesL = SVector{7,FT}(V1L, V2L, V3L, V4L, V5L, V6L, V7L)
                samplesR = SVector{7,FT}(V7R, V6R, V5R, V4R, V3R, V2R, V1R)
                valL = weno7_face_left(samplesL, ss)
                valR = weno7_face_right(samplesR, ss)

                # ... (valL ???valR 计算完毕) ...
            
            # ===  WENO5 分支 ===
            elseif ϕx < hybrid_ϕ3 # WENO5
                # ...
                # V2L (i-2), V3L (i-1), V4L (i), V5L (i+1), V6L (i+2)
                # 对应标准 WENO5 ???v1...v5
                
                # Left Side
                # Beta 1: (13/12)(v1-2v2+v3)^2 + (1/4)(v1-4v2+3v3)^2
                t1 = V2L - FT(2.0)*V3L + V4L; t2 = V2L - FT(4.0)*V3L + FT(3.0)*V4L
                s1L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                
                # Beta 2: (13/12)(v2-2v3+v4)^2 + (1/4)(v2-v4)^2
                t1 = V3L - FT(2.0)*V4L + V5L; t2 = V3L - V5L
                s2L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                
                # Beta 3: (13/12)(v3-2v4+v5)^2 + (1/4)(3v3-4v4+v5)^2
                t1 = V4L - FT(2.0)*V5L + V6L; t2 = FT(3.0)*V4L - FT(4.0)*V5L + V6L
                s3L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                # Weights (d0=1/10, d1=6/10, d2=3/10 -> relative 1, 6, 3)
                t_d1L = WENOϵ2 + s1L*ss; t_d2L = WENOϵ2 + s2L*ss; t_d3L = WENOϵ2 + s3L*ss
                @static if weno_z
                    τ5L = abs(s1L - s3L) * ss
                    α1L = one(FT)*(one(FT)+τ5L/(t_d1L+WENOϵ2))/(t_d1L*t_d1L)
                    α2L = FT(6.0e0)*(one(FT)+τ5L/(t_d2L+WENOϵ2))/(t_d2L*t_d2L)
                    α3L = FT(3.0)*(one(FT)+τ5L/(t_d3L+WENOϵ2))/(t_d3L*t_d3L)
                else
                    α1L = one(FT)/(t_d1L * t_d1L)
                    α2L = FT(6.0e0)/(t_d2L * t_d2L)
                    α3L = FT(3.0)/(t_d3L * t_d3L)
                end
                invsumL = one(FT)/(α1L+α2L+α3L)

                # Candidates
                v1 = FT(2.0)*V2L - FT(7.0e0)*V3L + FT(11.0)*V4L
                v2 = -one(FT)*V3L + FT(5.0e0)*V4L + FT(2.0)*V5L
                v3 = FT(2.0)*V4L + FT(5.0e0)*V5L - one(FT)*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2 # tmp2 is 1/6

                # Right Side (Symmetric)
                # Use V2R...V6R
                t1 = V2R - FT(2.0)*V3R + V4R; t2 = V2R - FT(4.0)*V3R + FT(3.0)*V4R
                s1R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3R - FT(2.0)*V4R + V5R; t2 = V3R - V5R
                s2R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4R - FT(2.0)*V5R + V6R; t2 = FT(3.0)*V4R - FT(4.0)*V5R + V6R
                s3R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; t_d2R = WENOϵ2 + s2R*ss; t_d3R = WENOϵ2 + s3R*ss
                @static if weno_z
                    τ5R = abs(s1R - s3R) * ss
                    α1R = one(FT)*(one(FT)+τ5R/(t_d1R+WENOϵ2))/(t_d1R*t_d1R)
                    α2R = FT(6.0e0)*(one(FT)+τ5R/(t_d2R+WENOϵ2))/(t_d2R*t_d2R)
                    α3R = FT(3.0)*(one(FT)+τ5R/(t_d3R+WENOϵ2))/(t_d3R*t_d3R)
                else
                    α1R = one(FT)/(t_d1R * t_d1R)
                    α2R = FT(6.0e0)/(t_d2R * t_d2R)
                    α3R = FT(3.0)/(t_d3R * t_d3R)
                end
                invsumR = one(FT)/(α1R+α2R+α3R)

                v1 = FT(2.0)*V2R - FT(7.0e0)*V3R + FT(11.0)*V4R
                v2 = -one(FT)*V3R + FT(5.0e0)*V4R + FT(2.0)*V5R
                v3 = FT(2.0)*V4R + FT(5.0e0)*V5R - one(FT)*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + FT(0.5)*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - FT(0.5)*minmod(V4R - V3R, V4R - V5R)
            end
            
            UL_final[n] = valL; UR_final[n] = valR
        end
    end

    @static if ct_mode
        _bi, _bj, _bk = ct_face_index(Val(6), i, j, k)
    end
    @static if strict_ct_positivity && ct_mode
        if splitMethodID == Int32(4)
            _Bn_recon = _Bn_geometry
            _WL_reconstructed = SVector{Ncons,FT}(
                ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons,FT},
            )
            _WR_reconstructed = SVector{Ncons,FT}(
                ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons,FT},
            )
            _UL_reconstructed = ct_primitive_to_conservative(
                _WL_reconstructed, nx, ny, nz, _Bn_recon, FT(γ),
            )
            _UR_reconstructed = ct_primitive_to_conservative(
                _WR_reconstructed, nx, ny, nz, _Bn_recon, FT(γ),
            )
            for n = 1:Ncons
                @inbounds UL_final[n] = _UL_reconstructed[n]
                @inbounds UR_final[n] = _UR_reconstructed[n]
            end
            _ct_record_if_invalid!(
                pos_meta, pos_values, CT_POS_SITE_RECONSTRUCTED,
                Int32(1), Int32(1), i, j, k, _UL_reconstructed, _Bn_recon,
            )
            _ct_record_if_invalid!(
                pos_meta, pos_values, CT_POS_SITE_RECONSTRUCTED,
                Int32(1), Int32(2), i, j, k, _UR_reconstructed, _Bn_recon,
            )
        end
    end

    # Non-strict paths retain the legacy first-order reconstruction fallback.
    @static if !(strict_ct_positivity && ct_mode)
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    @static if equation_type == :MHD
        _B2L = UL_final[6]^2 + UL_final[7]^2 + UL_final[8]^2
        _B2R = UR_final[6]^2 + UR_final[7]^2 + UR_final[8]^2
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT)) -
            FT(0.5) * _B2L * INV_MU0_SI
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT)) -
            FT(0.5) * _B2R * INV_MU0_SI
    else
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    end
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = _structured_state_component(U,Q,i,j,k,n); end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = _structured_state_component(U,Q,i+1,j,k,n); end
    end
    end

    # 4. 组装并计算通量 (use Tuple construction to avoid StaticArray dynamic dispatch on GPU)
    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})

    _background_face = SVector{3,FT}(zero(FT),zero(FT),zero(FT))
    @static if ct_mode
        if B0_cell_CT !== nothing
            @inbounds _split_ss = FT(2)/(S[i+Int32(1),j,k]+S[i,j,k])
            UL_vec,UR_vec,_background_face =
                ct_background_split_interface_states(
                    UL_vec,UR_vec,Q,B0_cell_CT,i,j,k,_split_ss,
                    nx,ny,nz,_B0n_geometry,Val(1),
                )
        end
    end

    # CT: replace normal B (Bx=UBX) with face-centered Bn for both L and R states
    # Reconstruction thread (i,j,k) solves the interface between cells i and
    # i+1, so the matching staggered x-face is at normal index i+1.
    @static if ct_mode
        _Bn_face = _Bn_geometry
        _UL_before_face_b = UL_vec
        _UR_before_face_b = UR_vec
        UL_vec = ct_impose_face_bn_preserve_p(UL_vec, nx, ny, nz, _Bn_face)
        UR_vec = ct_impose_face_bn_preserve_p(UR_vec, nx, ny, nz, _Bn_face)
        @static if strict_ct_positivity
            if splitMethodID == Int32(4)
                _ct_record_pressure_change!(
                    pos_meta, pos_values, Int32(1), Int32(1), i, j, k,
                    _UL_before_face_b, UL_vec, _Bn_face,
                )
                _ct_record_pressure_change!(
                    pos_meta, pos_values, Int32(1), Int32(2), i, j, k,
                    _UR_before_face_b, UR_vec, _Bn_face,
                )
                _ct_record_if_invalid!(
                    pos_meta, pos_values, CT_POS_SITE_HLLD_INPUT,
                    Int32(1), Int32(1), i, j, k, UL_vec, _Bn_face,
                )
                _ct_record_if_invalid!(
                    pos_meta, pos_values, CT_POS_SITE_HLLD_INPUT,
                    Int32(1), Int32(2), i, j, k, UR_vec, _Bn_face,
                )
            end
        end
    end

    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)
    @static if ct_mode
        if B0_cell_CT !== nothing
            flux_temp = ct_remove_background_maxwell_stress(
                flux_temp,_background_face,nx,ny,nz,
            )
        end
    end

    @static if ct_mode
        @static if @isdefined(ct_emf_scheme) &&
                   ct_emf_scheme == CT_EMF_WENO7_SG07
            flux_b = SVector{3,FT}(
                flux_temp[UBX], flux_temp[UBY], flux_temp[UBZ],
            )
            normal = SVector{3,FT}(nx, ny, nz)
            etan = ct_face_tangential_electric(flux_b, normal)
            @inbounds spacing = ct_face_normal_spacing(
                Vol[i, j, k], Vol[i+Int32(1), j, k], Area,
            )
            weight = ct_emf_weight_from_density_sum(
                flux_temp[1], UL_vec[1] + UR_vec[1], spacing, dt_stage,
            )
            @inbounds begin
                cache_i[i-NG+1, j, k, 1] = etan[1]
                cache_i[i-NG+1, j, k, 2] = etan[2]
                cache_i[i-NG+1, j, k, 3] = etan[3]
                cache_i[i-NG+1, j, k, 4] = weight
            end
        end
    end
    if j >= NG+Int32(1)-tangential_halo &&
       j <= nyp+NG+tangential_halo &&
       k >= NG+Int32(1)-tangential_halo &&
       k <= nzp+NG+tangential_halo
        fi=i-NG+1; fj=j-NG+tangential_halo; fk=k-NG+tangential_halo
        @static if ct_mode
            @inbounds rho_sum_x[fi,fj,fk] = UL_vec[1] + UR_vec[1]
        end
        @inbounds for n = 1:Ncons
            Fx[fi,fj,fk,n] = flux_temp[n]*Area
        end
    end
    return
end

function Conser_reconstruct_j(Q, U, ϕ, S, Fy, rho_sum_y, Areaj, nxj, nyj, nzj, nxp, nyp, nzp,
                              stencil_arr, Δstencil_arr, lin_phi_arr,
                              stencil_R_arr, Δstencil_R_arr,
                              ch_glm::FT, mode::Int32, By_face_CT,
                              cache_j, Vol, dt_stage::FT,
                              rk_stage::Int32, pos_meta, pos_values,
                              B0y_face_CT=nothing, B0_cell_CT=nothing)
    @static if strict_ct_positivity && ct_mode
        _use_primitive_reconstruction = splitMethodID == Int32(4)
    else
        _use_primitive_reconstruction = false
    end
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. Boundary check; WENO7 also fills the full tangential edge cache.
    tangential_halo = STRUCTURED_FLUX_TANGENTIAL_HALO
    @static if ct_mode && @isdefined(ct_emf_scheme) &&
               ct_emf_scheme == CT_EMF_WENO7_SG07
            if i > nxp+Int32(2)*NG || j > nyp+NG ||
               k > nzp+Int32(2)*NG || i < Int32(1) || j < NG || k < Int32(1)
                return
            end
    else
        if i > nxp+NG+tangential_halo || j > nyp+NG ||
           k > nzp+NG+tangential_halo ||
           i < NG+Int32(1)-tangential_halo || j < NG ||
           k < NG+Int32(1)-tangential_halo
            return
        end
    end
    # Interior/boundary mode filter (stencil width=4 in j)
    if mode == Int32(1) && (j < NG+Int32(4) || j > nyp+NG-Int32(4)); return; end
    if mode == Int32(2) && (j >= NG+Int32(4) && j <= nyp+NG-Int32(4)); return; end

    # 2. Geometry
    @static if ct_mode
        Area,nx,ny,nz,_Bn_geometry = structured_ct_face_geometry_bn(
            By_face_CT,B0y_face_CT,Areaj,nxj,nyj,nzj,
            i,j+Int32(1),k,Val(2),Q,
        )
        _B0n_geometry = zero(FT)
        if B0_cell_CT !== nothing
            _,_,_,_,_B0n_geometry = structured_ct_face_geometry_bn(
                B0y_face_CT,nothing,Areaj,nxj,nyj,nzj,
                i,j+Int32(1),k,Val(2),nothing,CT_FACE_BN_POINT6,
            )
        end
    else
        Area,nx,ny,nz = structured_face_geometry(
            Areaj,nxj,nyj,nzj,i,j+Int32(1),k,Val(2),
        )
    end

    # 3. 激波传感器
    @inbounds ϕy = max(ϕ[i, j-2, k], ϕ[i, j-1, k], ϕ[i, j, k], ϕ[i, j+1, k], ϕ[i, j+2, k], ϕ[i, j+3, k])

    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # Branch A: Smooth region
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    if _use_primitive_reconstruction
        if eigen_reconstruction
            _bi_plm, _bj_plm, _bk_plm = ct_face_index(Val(7), i, j, k)
            _bn_plm = _Bn_geometry
            _wm = ct_primitive_state(Q, i, j-1, k)
            _wc = ct_primitive_state(Q, i, j,   k)
            _wp = ct_primitive_state(Q, i, j+1, k)
            _wl = ct_mhd_characteristic_plm_plus(
                _wm, _wc, _wp, nx, ny, nz, _bn_plm, FT(γ),
            )
            for n = 1:Ncons
                UL_final[n] = _wl[n]
            end
            _wpp = ct_primitive_state(Q, i, j+2, k)
            _wr = ct_mhd_characteristic_plm_minus(
                _wc, _wp, _wpp, nx, ny, nz, _bn_plm, FT(γ),
            )
            for n = 1:Ncons
                UR_final[n] = _wr[n]
            end
        else
            for n = 1:Ncons
                qn = ct_primitive_reconstruction_slot(n)
                @inbounds qm = Q[i,j-1,k,qn]
                @inbounds qc = Q[i,j  ,k,qn]
                @inbounds qp = Q[i,j+1,k,qn]
                @inbounds qpp = Q[i,j+2,k,qn]
                UL_final[n] = ct_plm_plus(qm, qc, qp)
                UR_final[n] = ct_plm_minus(qc, qp, qpp)
            end
        end
    elseif ϕy < hybrid_ϕ1
        # ═══ Original 1D 7-point reconstruction ═══
        α_adapt = min(ϕy / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        @inbounds L1 = stencil_arr[j,k,1] + α_adapt * Δstencil_arr[j,k,1]; @inbounds L2 = stencil_arr[j,k,2] + α_adapt * Δstencil_arr[j,k,2]
        @inbounds L3 = stencil_arr[j,k,3] + α_adapt * Δstencil_arr[j,k,3]; @inbounds L4 = stencil_arr[j,k,4] + α_adapt * Δstencil_arr[j,k,4]
        @inbounds L5 = stencil_arr[j,k,5] + α_adapt * Δstencil_arr[j,k,5]; @inbounds L6 = stencil_arr[j,k,6] + α_adapt * Δstencil_arr[j,k,6]
        @inbounds L7 = stencil_arr[j,k,7] + α_adapt * Δstencil_arr[j,k,7]
        for n = 1:Ncons
            @inbounds v1 = _structured_state_component(U,Q,i,j-3,k,n); v2 = _structured_state_component(U,Q,i,j-2,k,n); v3 = _structured_state_component(U,Q,i,j-1,k,n)
            @inbounds v4 = _structured_state_component(U,Q,i,j,k,n); v5 = _structured_state_component(U,Q,i,j+1,k,n); v6 = _structured_state_component(U,Q,i,j+2,k,n); v7 = _structured_state_component(U,Q,i,j+3,k,n)
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        @inbounds L1 = stencil_R_arr[j,k,1] + α_adapt * Δstencil_R_arr[j,k,1]; @inbounds L2 = stencil_R_arr[j,k,2] + α_adapt * Δstencil_R_arr[j,k,2]
        @inbounds L3 = stencil_R_arr[j,k,3] + α_adapt * Δstencil_R_arr[j,k,3]; @inbounds L4 = stencil_R_arr[j,k,4] + α_adapt * Δstencil_R_arr[j,k,4]
        @inbounds L5 = stencil_R_arr[j,k,5] + α_adapt * Δstencil_R_arr[j,k,5]; @inbounds L6 = stencil_R_arr[j,k,6] + α_adapt * Δstencil_R_arr[j,k,6]
        @inbounds L7 = stencil_R_arr[j,k,7] + α_adapt * Δstencil_R_arr[j,k,7]
        for n = 1:Ncons
            @inbounds r1 = _structured_state_component(U,Q,i,j+4,k,n); r2 = _structured_state_component(U,Q,i,j+3,k,n); r3 = _structured_state_component(U,Q,i,j+2,k,n)
            @inbounds r4 = _structured_state_component(U,Q,i,j+1,k,n); r5 = _structured_state_component(U,Q,i,j,k,n); r6 = _structured_state_component(U,Q,i,j-1,k,n); r7 = _structured_state_component(U,Q,i,j-2,k,n)
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end

    # ==============================
    # Branch B: Discontinuous region
    # ==============================
    else 
        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        # 平滑因子 ss
        @inbounds ss = FT(2.0)/(S[i, j+1, k] + S[i, j, k])

        for n = 1:Ncons
            @inbounds V1L = _structured_state_component(U,Q,i,j-3,k,n)
            @inbounds V2L = _structured_state_component(U,Q,i,j-2,k,n)
            @inbounds V3L = _structured_state_component(U,Q,i,j-1,k,n)
            @inbounds V4L = _structured_state_component(U,Q,i,j,k,n)
            @inbounds V5L = _structured_state_component(U,Q,i,j+1,k,n)
            @inbounds V6L = _structured_state_component(U,Q,i,j+2,k,n)
            @inbounds V7L = _structured_state_component(U,Q,i,j+3,k,n)

            @inbounds V1R = _structured_state_component(U,Q,i,j+4,k,n)
            @inbounds V2R = _structured_state_component(U,Q,i,j+3,k,n)
            @inbounds V3R = _structured_state_component(U,Q,i,j+2,k,n)
            @inbounds V4R = _structured_state_component(U,Q,i,j+1,k,n)
            @inbounds V5R = _structured_state_component(U,Q,i,j,k,n)
            @inbounds V6R = _structured_state_component(U,Q,i,j-1,k,n)
            @inbounds V7R = _structured_state_component(U,Q,i,j-2,k,n)

            valL = zero(FT); valR = zero(FT)

            if ϕy < hybrid_ϕ2 # WENO7
                samplesL = SVector{7,FT}(V1L, V2L, V3L, V4L, V5L, V6L, V7L)
                samplesR = SVector{7,FT}(V7R, V6R, V5R, V4R, V3R, V2R, V1R)
                valL = weno7_face_left(samplesL, ss)
                valR = weno7_face_right(samplesR, ss)

            elseif ϕy < hybrid_ϕ3 # WENO5
                t1 = V2L - FT(2.0)*V3L + V4L; t2 = V2L - FT(4.0)*V3L + FT(3.0)*V4L
                s1L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3L - FT(2.0)*V4L + V5L; t2 = V3L - V5L
                s2L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4L - FT(2.0)*V5L + V6L; t2 = FT(3.0)*V4L - FT(4.0)*V5L + V6L
                s3L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1L = WENOϵ2 + s1L*ss; t_d2L = WENOϵ2 + s2L*ss; t_d3L = WENOϵ2 + s3L*ss
                @static if weno_z
                    τ5L = abs(s1L - s3L) * ss
                    α1L = one(FT)*(one(FT)+τ5L/(t_d1L+WENOϵ2))/(t_d1L*t_d1L)
                    α2L = FT(6.0e0)*(one(FT)+τ5L/(t_d2L+WENOϵ2))/(t_d2L*t_d2L)
                    α3L = FT(3.0)*(one(FT)+τ5L/(t_d3L+WENOϵ2))/(t_d3L*t_d3L)
                else
                    α1L = one(FT)/(t_d1L * t_d1L)
                    α2L = FT(6.0e0)/(t_d2L * t_d2L)
                    α3L = FT(3.0)/(t_d3L * t_d3L)
                end
                invsumL = one(FT)/(α1L+α2L+α3L)

                v1 = FT(2.0)*V2L - FT(7.0e0)*V3L + FT(11.0)*V4L
                v2 = -one(FT)*V3L + FT(5.0e0)*V4L + FT(2.0)*V5L
                v3 = FT(2.0)*V4L + FT(5.0e0)*V5L - one(FT)*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2

                t1 = V2R - FT(2.0)*V3R + V4R; t2 = V2R - FT(4.0)*V3R + FT(3.0)*V4R
                s1R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3R - FT(2.0)*V4R + V5R; t2 = V3R - V5R
                s2R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4R - FT(2.0)*V5R + V6R; t2 = FT(3.0)*V4R - FT(4.0)*V5R + V6R
                s3R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; t_d2R = WENOϵ2 + s2R*ss; t_d3R = WENOϵ2 + s3R*ss
                @static if weno_z
                    τ5R = abs(s1R - s3R) * ss
                    α1R = one(FT)*(one(FT)+τ5R/(t_d1R+WENOϵ2))/(t_d1R*t_d1R)
                    α2R = FT(6.0e0)*(one(FT)+τ5R/(t_d2R+WENOϵ2))/(t_d2R*t_d2R)
                    α3R = FT(3.0)*(one(FT)+τ5R/(t_d3R+WENOϵ2))/(t_d3R*t_d3R)
                else
                    α1R = one(FT)/(t_d1R * t_d1R)
                    α2R = FT(6.0e0)/(t_d2R * t_d2R)
                    α3R = FT(3.0)/(t_d3R * t_d3R)
                end
                invsumR = one(FT)/(α1R+α2R+α3R)

                v1 = FT(2.0)*V2R - FT(7.0e0)*V3R + FT(11.0)*V4R
                v2 = -one(FT)*V3R + FT(5.0e0)*V4R + FT(2.0)*V5R
                v3 = FT(2.0)*V4R + FT(5.0e0)*V5R - one(FT)*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + FT(0.5)*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - FT(0.5)*minmod(V4R - V3R, V4R - V5R)
            end
            
            UL_final[n] = valL; UR_final[n] = valR
        end
    end

    @static if ct_mode
        _bi, _bj, _bk = ct_face_index(Val(7), i, j, k)
    end
    @static if strict_ct_positivity && ct_mode
        if splitMethodID == Int32(4)
            _Bn_recon = _Bn_geometry
            _WL_reconstructed = SVector{Ncons,FT}(
                ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons,FT},
            )
            _WR_reconstructed = SVector{Ncons,FT}(
                ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons,FT},
            )
            _UL_reconstructed = ct_primitive_to_conservative(
                _WL_reconstructed, nx, ny, nz, _Bn_recon, FT(γ),
            )
            _UR_reconstructed = ct_primitive_to_conservative(
                _WR_reconstructed, nx, ny, nz, _Bn_recon, FT(γ),
            )
            for n = 1:Ncons
                @inbounds UL_final[n] = _UL_reconstructed[n]
                @inbounds UR_final[n] = _UR_reconstructed[n]
            end
            _ct_record_if_invalid!(
                pos_meta, pos_values, CT_POS_SITE_RECONSTRUCTED,
                Int32(2), Int32(1), i, j, k, _UL_reconstructed, _Bn_recon,
            )
            _ct_record_if_invalid!(
                pos_meta, pos_values, CT_POS_SITE_RECONSTRUCTED,
                Int32(2), Int32(2), i, j, k, _UR_reconstructed, _Bn_recon,
            )
        end
    end

    # Non-strict paths retain the legacy first-order reconstruction fallback.
    @static if !(strict_ct_positivity && ct_mode)
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    @static if equation_type == :MHD
        _B2L = UL_final[6]^2 + UL_final[7]^2 + UL_final[8]^2
        _B2R = UR_final[6]^2 + UR_final[7]^2 + UR_final[8]^2
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT)) -
            FT(0.5) * _B2L * INV_MU0_SI
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT)) -
            FT(0.5) * _B2R * INV_MU0_SI
    else
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    end
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = _structured_state_component(U,Q,i,j,k,n); end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = _structured_state_component(U,Q,i,j+1,k,n); end
    end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})

    _background_face = SVector{3,FT}(zero(FT),zero(FT),zero(FT))
    @static if ct_mode
        if B0_cell_CT !== nothing
            @inbounds _split_ss = FT(2)/(S[i,j+Int32(1),k]+S[i,j,k])
            UL_vec,UR_vec,_background_face =
                ct_background_split_interface_states(
                    UL_vec,UR_vec,Q,B0_cell_CT,i,j,k,_split_ss,
                    nx,ny,nz,_B0n_geometry,Val(2),
                )
        end
    end

    # CT: replace normal B (By=UBY) with face-centered Bn
    @static if ct_mode
        _Bn_face = _Bn_geometry
        _UL_before_face_b = UL_vec
        _UR_before_face_b = UR_vec
        UL_vec = ct_impose_face_bn_preserve_p(UL_vec, nx, ny, nz, _Bn_face)
        UR_vec = ct_impose_face_bn_preserve_p(UR_vec, nx, ny, nz, _Bn_face)
        @static if strict_ct_positivity
            if splitMethodID == Int32(4)
                _ct_record_pressure_change!(
                    pos_meta, pos_values, Int32(2), Int32(1), i, j, k,
                    _UL_before_face_b, UL_vec, _Bn_face,
                )
                _ct_record_pressure_change!(
                    pos_meta, pos_values, Int32(2), Int32(2), i, j, k,
                    _UR_before_face_b, UR_vec, _Bn_face,
                )
                _ct_record_if_invalid!(
                    pos_meta, pos_values, CT_POS_SITE_HLLD_INPUT,
                    Int32(2), Int32(1), i, j, k, UL_vec, _Bn_face,
                )
                _ct_record_if_invalid!(
                    pos_meta, pos_values, CT_POS_SITE_HLLD_INPUT,
                    Int32(2), Int32(2), i, j, k, UR_vec, _Bn_face,
                )
            end
        end
    end

    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕy, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)
    @static if ct_mode
        if B0_cell_CT !== nothing
            flux_temp = ct_remove_background_maxwell_stress(
                flux_temp,_background_face,nx,ny,nz,
            )
        end
    end

    @static if ct_mode
        @static if @isdefined(ct_emf_scheme) &&
                   ct_emf_scheme == CT_EMF_WENO7_SG07
            flux_b = SVector{3,FT}(
                flux_temp[UBX], flux_temp[UBY], flux_temp[UBZ],
            )
            normal = SVector{3,FT}(nx, ny, nz)
            etan = ct_face_tangential_electric(flux_b, normal)
            @inbounds spacing = ct_face_normal_spacing(
                Vol[i, j, k], Vol[i, j+Int32(1), k], Area,
            )
            weight = ct_emf_weight_from_density_sum(
                flux_temp[1], UL_vec[1] + UR_vec[1], spacing, dt_stage,
            )
            @inbounds begin
                cache_j[i, j-NG+1, k, 1] = etan[1]
                cache_j[i, j-NG+1, k, 2] = etan[2]
                cache_j[i, j-NG+1, k, 3] = etan[3]
                cache_j[i, j-NG+1, k, 4] = weight
            end
        end
    end
    if i >= NG+Int32(1)-tangential_halo &&
       i <= nxp+NG+tangential_halo &&
       k >= NG+Int32(1)-tangential_halo &&
       k <= nzp+NG+tangential_halo
        fi=i-NG+tangential_halo; fj=j-NG+1; fk=k-NG+tangential_halo
        @static if ct_mode
            @inbounds rho_sum_y[fi,fj,fk] = UL_vec[1] + UR_vec[1]
        end
        @inbounds for n = 1:Ncons
            Fy[fi,fj,fk,n] = flux_temp[n]*Area
        end
    end
    return 
end

function Conser_reconstruct_k(Q, U, ϕ, S, Fz, rho_sum_z, Areak, nxk, nyk, nzk, nxp, nyp, nzp,
                              stencil_arr, Δstencil_arr, lin_phi_arr,
                              stencil_R_arr, Δstencil_R_arr,
                              ch_glm::FT, mode::Int32, Bz_face_CT,
                              cache_k, Vol, dt_stage::FT,
                              rk_stage::Int32, pos_meta, pos_values,
                              B0z_face_CT=nothing, B0_cell_CT=nothing)
    @static if strict_ct_positivity && ct_mode
        _use_primitive_reconstruction = splitMethodID == Int32(4)
    else
        _use_primitive_reconstruction = false
    end
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. Boundary check; WENO7 also fills the full tangential edge cache.
    tangential_halo = STRUCTURED_FLUX_TANGENTIAL_HALO
    @static if ct_mode && @isdefined(ct_emf_scheme) &&
               ct_emf_scheme == CT_EMF_WENO7_SG07
            if i > nxp+Int32(2)*NG || j > nyp+Int32(2)*NG ||
               k > nzp+NG || i < Int32(1) || j < Int32(1) || k < NG
                return
            end
    else
        if i > nxp+NG+tangential_halo ||
           j > nyp+NG+tangential_halo || k > nzp+NG ||
           i < NG+Int32(1)-tangential_halo ||
           j < NG+Int32(1)-tangential_halo || k < NG
            return
        end
    end


    # Interior/boundary mode filter (stencil width=4 in k)
    if mode == Int32(1) && (k < NG+Int32(4) || k > nzp+NG-Int32(4)); return; end
    if mode == Int32(2) && (k >= NG+Int32(4) && k <= nzp+NG-Int32(4)); return; end

    # 2. Geometry
    @static if ct_mode
        Area,nx,ny,nz,_Bn_geometry = structured_ct_face_geometry_bn(
            Bz_face_CT,B0z_face_CT,Areak,nxk,nyk,nzk,
            i,j,k+Int32(1),Val(3),Q,
        )
        _B0n_geometry = zero(FT)
        if B0_cell_CT !== nothing
            _,_,_,_,_B0n_geometry = structured_ct_face_geometry_bn(
                B0z_face_CT,nothing,Areak,nxk,nyk,nzk,
                i,j,k+Int32(1),Val(3),nothing,CT_FACE_BN_POINT6,
            )
        end
    else
        Area,nx,ny,nz = structured_face_geometry(
            Areak,nxk,nyk,nzk,i,j,k+Int32(1),Val(3),
        )
    end

    # 3. 激波传感器
    @inbounds ϕz = max(ϕ[i, j, k-2], ϕ[i, j, k-1], ϕ[i, j, k], ϕ[i, j, k+1], ϕ[i, j, k+2], ϕ[i, j, k+3])

    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # Branch A: Smooth region
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    if _use_primitive_reconstruction
        if eigen_reconstruction
            _bi_plm, _bj_plm, _bk_plm = ct_face_index(Val(8), i, j, k)
            _bn_plm = _Bn_geometry
            _wm = ct_primitive_state(Q, i, j, k-1)
            _wc = ct_primitive_state(Q, i, j, k)
            _wp = ct_primitive_state(Q, i, j, k+1)
            _wl = ct_mhd_characteristic_plm_plus(
                _wm, _wc, _wp, nx, ny, nz, _bn_plm, FT(γ),
            )
            for n = 1:Ncons
                UL_final[n] = _wl[n]
            end
            _wpp = ct_primitive_state(Q, i, j, k+2)
            _wr = ct_mhd_characteristic_plm_minus(
                _wc, _wp, _wpp, nx, ny, nz, _bn_plm, FT(γ),
            )
            for n = 1:Ncons
                UR_final[n] = _wr[n]
            end
        else
            for n = 1:Ncons
                qn = ct_primitive_reconstruction_slot(n)
                @inbounds qm = Q[i,j,k-1,qn]
                @inbounds qc = Q[i,j,k  ,qn]
                @inbounds qp = Q[i,j,k+1,qn]
                @inbounds qpp = Q[i,j,k+2,qn]
                UL_final[n] = ct_plm_plus(qm, qc, qp)
                UR_final[n] = ct_plm_minus(qc, qp, qpp)
            end
        end
    elseif ϕz < hybrid_ϕ1
        α_adapt = min(ϕz / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        @inbounds L1 = stencil_arr[j,k,1] + α_adapt * Δstencil_arr[j,k,1]; @inbounds L2 = stencil_arr[j,k,2] + α_adapt * Δstencil_arr[j,k,2]
        @inbounds L3 = stencil_arr[j,k,3] + α_adapt * Δstencil_arr[j,k,3]; @inbounds L4 = stencil_arr[j,k,4] + α_adapt * Δstencil_arr[j,k,4]
        @inbounds L5 = stencil_arr[j,k,5] + α_adapt * Δstencil_arr[j,k,5]; @inbounds L6 = stencil_arr[j,k,6] + α_adapt * Δstencil_arr[j,k,6]
        @inbounds L7 = stencil_arr[j,k,7] + α_adapt * Δstencil_arr[j,k,7]
        for n = 1:Ncons
            @inbounds v1 = _structured_state_component(U,Q,i,j,k-3,n); v2 = _structured_state_component(U,Q,i,j,k-2,n); v3 = _structured_state_component(U,Q,i,j,k-1,n)
            @inbounds v4 = _structured_state_component(U,Q,i,j,k,n); v5 = _structured_state_component(U,Q,i,j,k+1,n); v6 = _structured_state_component(U,Q,i,j,k+2,n); v7 = _structured_state_component(U,Q,i,j,k+3,n)
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        @inbounds L1 = stencil_R_arr[j,k,1] + α_adapt * Δstencil_R_arr[j,k,1]; @inbounds L2 = stencil_R_arr[j,k,2] + α_adapt * Δstencil_R_arr[j,k,2]
        @inbounds L3 = stencil_R_arr[j,k,3] + α_adapt * Δstencil_R_arr[j,k,3]; @inbounds L4 = stencil_R_arr[j,k,4] + α_adapt * Δstencil_R_arr[j,k,4]
        @inbounds L5 = stencil_R_arr[j,k,5] + α_adapt * Δstencil_R_arr[j,k,5]; @inbounds L6 = stencil_R_arr[j,k,6] + α_adapt * Δstencil_R_arr[j,k,6]
        @inbounds L7 = stencil_R_arr[j,k,7] + α_adapt * Δstencil_R_arr[j,k,7]
        for n = 1:Ncons
            @inbounds r1 = _structured_state_component(U,Q,i,j,k+4,n); r2 = _structured_state_component(U,Q,i,j,k+3,n); r3 = _structured_state_component(U,Q,i,j,k+2,n)
            @inbounds r4 = _structured_state_component(U,Q,i,j,k+1,n); r5 = _structured_state_component(U,Q,i,j,k,n); r6 = _structured_state_component(U,Q,i,j,k-1,n); r7 = _structured_state_component(U,Q,i,j,k-2,n)
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end

    # ==============================
    # Branch B: Discontinuous region
    # ==============================
    else 
        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        # 平滑因子 ss
        @inbounds ss = FT(2.0)/(S[i, j, k+1] + S[i, j, k])

        for n = 1:Ncons
            @inbounds V1L = _structured_state_component(U,Q,i,j,k-3,n)
            @inbounds V2L = _structured_state_component(U,Q,i,j,k-2,n)
            @inbounds V3L = _structured_state_component(U,Q,i,j,k-1,n)
            @inbounds V4L = _structured_state_component(U,Q,i,j,k,n)
            @inbounds V5L = _structured_state_component(U,Q,i,j,k+1,n)
            @inbounds V6L = _structured_state_component(U,Q,i,j,k+2,n)
            @inbounds V7L = _structured_state_component(U,Q,i,j,k+3,n)

            @inbounds V1R = _structured_state_component(U,Q,i,j,k+4,n)
            @inbounds V2R = _structured_state_component(U,Q,i,j,k+3,n)
            @inbounds V3R = _structured_state_component(U,Q,i,j,k+2,n)
            @inbounds V4R = _structured_state_component(U,Q,i,j,k+1,n)
            @inbounds V5R = _structured_state_component(U,Q,i,j,k,n)
            @inbounds V6R = _structured_state_component(U,Q,i,j,k-1,n)
            @inbounds V7R = _structured_state_component(U,Q,i,j,k-2,n)

            valL = zero(FT); valR = zero(FT)

            if ϕz < hybrid_ϕ2 # WENO7
                samplesL = SVector{7,FT}(V1L, V2L, V3L, V4L, V5L, V6L, V7L)
                samplesR = SVector{7,FT}(V7R, V6R, V5R, V4R, V3R, V2R, V1R)
                valL = weno7_face_left(samplesL, ss)
                valR = weno7_face_right(samplesR, ss)

            elseif ϕz < hybrid_ϕ3 # WENO5
                t1 = V2L - FT(2.0)*V3L + V4L; t2 = V2L - FT(4.0)*V3L + FT(3.0)*V4L
                s1L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3L - FT(2.0)*V4L + V5L; t2 = V3L - V5L
                s2L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4L - FT(2.0)*V5L + V6L; t2 = FT(3.0)*V4L - FT(4.0)*V5L + V6L
                s3L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1L = WENOϵ2 + s1L*ss; t_d2L = WENOϵ2 + s2L*ss; t_d3L = WENOϵ2 + s3L*ss
                @static if weno_z
                    τ5L = abs(s1L - s3L) * ss
                    α1L = one(FT)*(one(FT)+τ5L/(t_d1L+WENOϵ2))/(t_d1L*t_d1L)
                    α2L = FT(6.0e0)*(one(FT)+τ5L/(t_d2L+WENOϵ2))/(t_d2L*t_d2L)
                    α3L = FT(3.0)*(one(FT)+τ5L/(t_d3L+WENOϵ2))/(t_d3L*t_d3L)
                else
                    α1L = one(FT)/(t_d1L * t_d1L)
                    α2L = FT(6.0e0)/(t_d2L * t_d2L)
                    α3L = FT(3.0)/(t_d3L * t_d3L)
                end
                invsumL = one(FT)/(α1L+α2L+α3L)

                v1 = FT(2.0)*V2L - FT(7.0e0)*V3L + FT(11.0)*V4L
                v2 = -one(FT)*V3L + FT(5.0e0)*V4L + FT(2.0)*V5L
                v3 = FT(2.0)*V4L + FT(5.0e0)*V5L - one(FT)*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2

                t1 = V2R - FT(2.0)*V3R + V4R; t2 = V2R - FT(4.0)*V3R + FT(3.0)*V4R
                s1R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3R - FT(2.0)*V4R + V5R; t2 = V3R - V5R
                s2R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4R - FT(2.0)*V5R + V6R; t2 = FT(3.0)*V4R - FT(4.0)*V5R + V6R
                s3R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; t_d2R = WENOϵ2 + s2R*ss; t_d3R = WENOϵ2 + s3R*ss
                @static if weno_z
                    τ5R = abs(s1R - s3R) * ss
                    α1R = one(FT)*(one(FT)+τ5R/(t_d1R+WENOϵ2))/(t_d1R*t_d1R)
                    α2R = FT(6.0e0)*(one(FT)+τ5R/(t_d2R+WENOϵ2))/(t_d2R*t_d2R)
                    α3R = FT(3.0)*(one(FT)+τ5R/(t_d3R+WENOϵ2))/(t_d3R*t_d3R)
                else
                    α1R = one(FT)/(t_d1R * t_d1R)
                    α2R = FT(6.0e0)/(t_d2R * t_d2R)
                    α3R = FT(3.0)/(t_d3R * t_d3R)
                end
                invsumR = one(FT)/(α1R+α2R+α3R)

                v1 = FT(2.0)*V2R - FT(7.0e0)*V3R + FT(11.0)*V4R
                v2 = -one(FT)*V3R + FT(5.0e0)*V4R + FT(2.0)*V5R
                v3 = FT(2.0)*V4R + FT(5.0e0)*V5R - one(FT)*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + FT(0.5)*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - FT(0.5)*minmod(V4R - V3R, V4R - V5R)
            end
            
            UL_final[n] = valL; UR_final[n] = valR
        end
    end

    @static if ct_mode
        _bi, _bj, _bk = ct_face_index(Val(8), i, j, k)
    end
    @static if strict_ct_positivity && ct_mode
        if splitMethodID == Int32(4)
            _Bn_recon = _Bn_geometry
            _WL_reconstructed = SVector{Ncons,FT}(
                ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons,FT},
            )
            _WR_reconstructed = SVector{Ncons,FT}(
                ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons,FT},
            )
            _UL_reconstructed = ct_primitive_to_conservative(
                _WL_reconstructed, nx, ny, nz, _Bn_recon, FT(γ),
            )
            _UR_reconstructed = ct_primitive_to_conservative(
                _WR_reconstructed, nx, ny, nz, _Bn_recon, FT(γ),
            )
            for n = 1:Ncons
                @inbounds UL_final[n] = _UL_reconstructed[n]
                @inbounds UR_final[n] = _UR_reconstructed[n]
            end
            _ct_record_if_invalid!(
                pos_meta, pos_values, CT_POS_SITE_RECONSTRUCTED,
                Int32(3), Int32(1), i, j, k, _UL_reconstructed, _Bn_recon,
            )
            _ct_record_if_invalid!(
                pos_meta, pos_values, CT_POS_SITE_RECONSTRUCTED,
                Int32(3), Int32(2), i, j, k, _UR_reconstructed, _Bn_recon,
            )
        end
    end

    # Non-strict paths retain the legacy first-order reconstruction fallback.
    @static if !(strict_ct_positivity && ct_mode)
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    @static if equation_type == :MHD
        _B2L = UL_final[6]^2 + UL_final[7]^2 + UL_final[8]^2
        _B2R = UR_final[6]^2 + UR_final[7]^2 + UR_final[8]^2
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT)) -
            FT(0.5) * _B2L * INV_MU0_SI
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT)) -
            FT(0.5) * _B2R * INV_MU0_SI
    else
        _eiL = UL_final[5] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
        _eiR = UR_final[5] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    end
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = _structured_state_component(U,Q,i,j,k,n); end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = _structured_state_component(U,Q,i,j,k+1,n); end
    end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})

    _background_face = SVector{3,FT}(zero(FT),zero(FT),zero(FT))
    @static if ct_mode
        if B0_cell_CT !== nothing
            @inbounds _split_ss = FT(2)/(S[i,j,k+Int32(1)]+S[i,j,k])
            UL_vec,UR_vec,_background_face =
                ct_background_split_interface_states(
                    UL_vec,UR_vec,Q,B0_cell_CT,i,j,k,_split_ss,
                    nx,ny,nz,_B0n_geometry,Val(3),
                )
        end
    end

    # CT: replace normal B (Bz=UBZ) with face-centered Bn
    @static if ct_mode
        _Bn_face = _Bn_geometry
        _UL_before_face_b = UL_vec
        _UR_before_face_b = UR_vec
        UL_vec = ct_impose_face_bn_preserve_p(UL_vec, nx, ny, nz, _Bn_face)
        UR_vec = ct_impose_face_bn_preserve_p(UR_vec, nx, ny, nz, _Bn_face)
        @static if strict_ct_positivity
            if splitMethodID == Int32(4)
                _ct_record_pressure_change!(
                    pos_meta, pos_values, Int32(3), Int32(1), i, j, k,
                    _UL_before_face_b, UL_vec, _Bn_face,
                )
                _ct_record_pressure_change!(
                    pos_meta, pos_values, Int32(3), Int32(2), i, j, k,
                    _UR_before_face_b, UR_vec, _Bn_face,
                )
                _ct_record_if_invalid!(
                    pos_meta, pos_values, CT_POS_SITE_HLLD_INPUT,
                    Int32(3), Int32(1), i, j, k, UL_vec, _Bn_face,
                )
                _ct_record_if_invalid!(
                    pos_meta, pos_values, CT_POS_SITE_HLLD_INPUT,
                    Int32(3), Int32(2), i, j, k, UR_vec, _Bn_face,
                )
            end
        end
    end

    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕz, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)
    @static if ct_mode
        if B0_cell_CT !== nothing
            flux_temp = ct_remove_background_maxwell_stress(
                flux_temp,_background_face,nx,ny,nz,
            )
        end
    end

    @static if ct_mode
        @static if @isdefined(ct_emf_scheme) &&
                   ct_emf_scheme == CT_EMF_WENO7_SG07
            flux_b = SVector{3,FT}(
                flux_temp[UBX], flux_temp[UBY], flux_temp[UBZ],
            )
            normal = SVector{3,FT}(nx, ny, nz)
            etan = ct_face_tangential_electric(flux_b, normal)
            @inbounds spacing = ct_face_normal_spacing(
                Vol[i, j, k], Vol[i, j, k+Int32(1)], Area,
            )
            weight = ct_emf_weight_from_density_sum(
                flux_temp[1], UL_vec[1] + UR_vec[1], spacing, dt_stage,
            )
            @inbounds begin
                cache_k[i, j, k-NG+1, 1] = etan[1]
                cache_k[i, j, k-NG+1, 2] = etan[2]
                cache_k[i, j, k-NG+1, 3] = etan[3]
                cache_k[i, j, k-NG+1, 4] = weight
            end
        end
    end
    if i >= NG+Int32(1)-tangential_halo &&
       i <= nxp+NG+tangential_halo &&
       j >= NG+Int32(1)-tangential_halo &&
       j <= nyp+NG+tangential_halo
        fi=i-NG+tangential_halo; fj=j-NG+tangential_halo; fk=k-NG+1
        @static if ct_mode
            @inbounds rho_sum_z[fi,fj,fk] = UL_vec[1] + UR_vec[1]
        end
        @inbounds for n = 1:Ncons
            Fz[fi,fj,fk,n] = flux_temp[n]*Area
        end
    end
    return
end

# =============================================================================
# Topology Singularity Smoothness Protection
# Folded from topology_singularity.jl.
# =============================================================================
# topology_singularity.jl
#
# General (mesh-agnostic) detection of multi-block edge singularities and
# adaptive WENO5 upwind-parameter (ϕ) boost to suppress topology-induced
# reconstruction artifacts.
#
# Design spec: docs/superpowers/specs/2026-06-20-topology-singularity-protection.md
#
# Core idea: a block edge (the line where two of the block's six faces meet)
# is a "singularity edge" iff BOTH faces are interblock AND they connect to
# DIFFERENT neighbor blocks. This captures any ≥3-block junction along a
# line, regardless of whether the interface geometry is smooth. The existing
# `apply_geometric_smoothness_protection!` in solver.jl catches geometrically
# non-smooth interfaces (face-normal angle jump, area-ratio stretch) but
# misses topologically singular but geometrically smooth interfaces — exactly
# the butterfly O-grid case where 5 blocks meet at continuous-metric edges.
#
# This module provides:
#   - EdgeSingularityInfo: per-block, per-edge singularity table
#   - detect_edge_singularities(face_bc, neighbors, n_blocks): build the table
#   - is_near_singularity_edge(bid, i, j, k, ...): per-cell query
#   - apply_topology_smoothness_protection!(...): boost ϕ near singularity edges
#
# Face convention (1-indexed throughout, matching solver.jl):
#   face 1 = ξ- (i low),   face 2 = ξ+ (i high)
#   face 3 = η- (j low),   face 4 = η+ (j high)
#   face 5 = ζ- (k low),   face 6 = ζ+ (k high)
#
# BC convention (matching block_connectivity.h5 `face_bc`):
#   0  = interblock
#   2  = periodic (self-loop)
#   11 = symmetry
#   42 = wall (diffrot_wall, etc.)
#   other positive = other non-interblock

# ─────────────────────────────────────────────────────────────────────────
# Configuration constants (Q1-Q4 all = "A" per user approval 2026-06-20)
# ─────────────────────────────────────────────────────────────────────────

# Number of cell layers from a singularity edge within which ϕ is boosted.
# Q2 = A: same as existing `_CROSSTYPE_N_PROTECT = 4`.
const _TOPO_N_PROTECT::Int = 4

# ϕ boost profile for layers 1..n_protect (layer 1 = closest to the edge).
# Q1 = A: same as existing `_CROSSTYPE_PHI = [1.0, 1.0, 0.8, 0.6]`.
#   ϕ = 1.0  → pure first-order Godunov upwind (full WENO5 → 1st-order degrade)
#   ϕ = 0.6  → 60% upwind + 40% central (mild degrade)
const _TOPO_PHI_PROFILE::Vector{Float64} = [1.0, 1.0, 0.8, 0.6]

# Q4 = A: corners (3-face, 4+-block junctions) use the same profile as edges.
# No extra layer, no stronger profile. The corner-detection logic below still
# runs and is exposed in EdgeSingularityInfo for future tuning, but by default
# `apply_topology_smoothness_protection!` treats corners and edges identically.

# ─────────────────────────────────────────────────────────────────────────
# Block-edge topology
# ─────────────────────────────────────────────────────────────────────────
#
# Each block has 12 edges. We index them 1..12 in a fixed order so that the
# singularity table is a (n_blocks, 12) bit array. The 12 edges are grouped
# by the axis they run along:
#
#   Edges  1..4  run along ξ (i direction), at the 4 (η, ζ) corner pairs:
#       1: (η-, ζ-)    2: (η-, ζ+)    3: (η+, ζ-)    4: (η+, ζ+)
#   Edges  5..8  run along η (j direction), at the 4 (ξ, ζ) corner pairs:
#       5: (ξ-, ζ-)    6: (ξ-, ζ+)    7: (ξ+, ζ-)    8: (ξ+, ζ+)
#   Edges  9..12 run along ζ (k direction), at the 4 (ξ, η) corner pairs:
#       9: (ξ-, η-)   10: (ξ-, η+)   11: (ξ+, η-)   12: (ξ+, η+)
#
# For each edge we record the two faces that meet at it (face_a, face_b).
# An edge is a singularity iff:
#   (a) both face_a and face_b are interblock (face_bc == 0), AND
#   (b) neighbors[block, face_a] != neighbors[block, face_b].
# Periodic faces (face_bc == 2) are treated as self-loops: they count as
# interblock with neighbor == block itself, so an edge between a periodic
# face and an interblock face has neighbor_a == block != neighbor_b → NOT a
# singularity (only 2 physical blocks involved: self + 1 other). This matches
# the spec's "periodic counts as self-neighbor" rule (§8.1).

# Static table: edge index → (face_a, face_b) pair
const _EDGE_FACE_PAIRS::Tuple{Tuple{Int,Int},Tuple{Int,Int},Tuple{Int,Int},Tuple{Int,Int},
                              Tuple{Int,Int},Tuple{Int,Int},Tuple{Int,Int},Tuple{Int,Int},
                              Tuple{Int,Int},Tuple{Int,Int},Tuple{Int,Int},Tuple{Int,Int}} = (
    # Edges along ξ (i): (η, ζ) corner pairs
    (3, 5), (3, 6), (4, 5), (4, 6),
    # Edges along η (j): (ξ, ζ) corner pairs
    (1, 5), (1, 6), (2, 5), (2, 6),
    # Edges along ζ (k): (ξ, η) corner pairs
    (1, 3), (1, 4), (2, 3), (2, 4),
)

# For each edge, the axis it runs along: 1=ξ(i), 2=η(j), 3=ζ(k).
# Used by is_near_singularity_edge to know which index varies freely along
# the edge and which two indices are pinned near a corner.
const _EDGE_AXIS::NTuple{12,Int} = (
    1, 1, 1, 1,   # edges 1-4 run along ξ (i)
    2, 2, 2, 2,   # edges 5-8 run along η (j)
    3, 3, 3, 3,   # edges 9-12 run along ζ (k)
)

# For each edge, the two "pinned" face indices and their sign (low/high).
# face_sign = -1 for low face (face 1,3,5), +1 for high face (face 2,4,6).
# This lets is_near_singularity_edge compute layer distance in index space.
# Layout: (axis_idx_a, is_high_a, axis_idx_b, is_high_b) where axis_idx is
# 1=i, 2=j, 3=k corresponding to face 1/2, 3/4, 5/6.
const _EDGE_PINS::NTuple{12, NTuple{4, Int}} = (
    # edges 1-4 along ξ: pin (η, ζ)
    (2, 0, 3, 0), (2, 0, 3, 1), (2, 1, 3, 0), (2, 1, 3, 1),
    # edges 5-8 along η: pin (ξ, ζ)
    (1, 0, 3, 0), (1, 0, 3, 1), (1, 1, 3, 0), (1, 1, 3, 1),
    # edges 9-12 along ζ: pin (ξ, η)
    (1, 0, 2, 0), (1, 0, 2, 1), (1, 1, 2, 0), (1, 1, 2, 1),
)

# ─────────────────────────────────────────────────────────────────────────
# EdgeSingularityInfo
# ─────────────────────────────────────────────────────────────────────────

"""
    EdgeSingularityInfo

Per-block, per-edge singularity table produced by `detect_edge_singularities`.

Fields:
- `is_singularity_edge::BitArray{2}`: shape `(n_blocks, 12)`, `true` if the
  edge is a multi-block junction singularity (≥3 blocks meeting along it).
- `is_singularity_corner::BitArray{3}`: shape `(n_blocks, 8, 1)` — reserved
  for future corner-specific boosting (Q4=A defaults to no extra treatment,
  but the detection runs for diagnostics). Corner index 1..8 maps to the
  (ξ±, η±, ζ±) octants in the order (---, --+, -+-, -++, +--, +-+, ++-, +++).
- `n_blocks::Int`: number of blocks.
"""
struct EdgeSingularityInfo
    is_singularity_edge::BitArray{2}
    is_singularity_corner::BitArray{3}
    n_blocks::Int
end

"""
    detect_edge_singularities(face_bc, neighbors, n_blocks) -> EdgeSingularityInfo

Build the per-block edge singularity table from the mesh connectivity data.

Arguments:
- `face_bc::AbstractArray{Int,2}`: shape `(n_blocks, 6)`, BC type per block
  per face. `0` = interblock, `2` = periodic (self-loop), other = wall/sym.
- `neighbors::AbstractArray{Int,2}`: shape `(n_blocks, 6)`, neighbor block
  id (1-indexed) for each face. For periodic faces, `neighbors[b,f] = b`
  (self-loop). For non-interblock faces, value is ignored.
- `n_blocks::Int`: number of blocks.

Returns an `EdgeSingularityInfo` with the singularity flags populated.
"""
function detect_edge_singularities(face_bc::AbstractArray{Int,2},
                                    neighbors::AbstractArray{Int,2},
                                    n_blocks::Int)::EdgeSingularityInfo
    is_edge = falses(n_blocks, 12)
    is_corner = falses(n_blocks, 8, 1)

    for b in 1:n_blocks
        for e in 1:12
            fa, fb = _EDGE_FACE_PAIRS[e]
            bc_a = face_bc[b, fa]
            bc_b = face_bc[b, fb]
            # Only interblock (0) faces count toward a multi-block junction.
            # Periodic (2) is a self-loop — it does not introduce a distinct
            # neighbor block, so an edge between a periodic face and an
            # interblock face is NOT a singularity (only 2 physical blocks:
            # self + 1 other). Walls/sym have no neighbor at all.
            if bc_a != 0
                continue
            end
            if bc_b != 0
                continue
            end
            n_a = neighbors[b, fa]
            n_b = neighbors[b, fb]
            # Singularity iff the two neighbors differ. This catches any
            # ≥3-block junction (self + 2 distinct others).
            if n_a != n_b
                is_edge[b, e] = true
            end
        end

        # Corner detection: 8 corners, each is the meeting of 3 mutually
        # adjacent faces. A corner is a "triple singularity" iff all 3 of
        # its edge pairs are singularity edges. Corner → edge mapping:
        #   corner 1 (ξ-,η-,ζ-): edges 1(η-ζ), 5(ξ-ζ), 9(ξ-η)  -- all 3 share this corner
        #   corner 2 (ξ-,η-,ζ+): edges 2, 6, 10
        #   corner 3 (ξ-,η+,ζ-): edges 3, 7, 11
        #   corner 4 (ξ-,η+,ζ+): edges 4, 8, 12
        #   corner 5 (ξ+,η-,ζ-): edges 1, 5, 9   -- ξ+ also touches these? No.
        # Actually each edge spans the FULL ξ range (for ξ-edges), so edge 1
        # touches BOTH ξ- and ξ+ corners. The corner→edge mapping is:
        #   corner (ξs, ηs, ζs) where s∈{-,+}:
        #     ξ-edge  = the (η, ζ) edge: index based on (ηs, ζs)
        #     η-edge  = the (ξ, ζ) edge: index 4 + based on (ξs, ζs)
        #     ζ-edge  = the (ξ, η) edge: index 8 + based on (ξs, ηs)
        # We compute corner singularity as: all three of its edges are singular.
        for c in 1:8
            # Decode corner c into (ξs, ηs, ζs) with s ∈ {0=low, 1=high}
            zs = (c - 1) ÷ 4       # 0 or 1  (ξ sign: 0=-, 1=+)
            ys = ((c - 1) % 4) ÷ 2 # 0 or 1  (η sign)
            xs = ((c - 1) % 4) % 2 # 0 or 1  (ζ sign)
            # ξ-edge index: 1..4, based on (ηs, ζs) = (ys, xs)
            #   (η-,ζ-)=1, (η-,ζ+)=2, (η+,ζ-)=3, (η+,ζ+)=4
            e_xi = 1 + ys*2 + xs
            # η-edge index: 5..8, based on (ξs, ζs) = (zs, xs)
            e_eta = 5 + zs*2 + xs
            # ζ-edge index: 9..12, based on (ξs, ηs) = (zs, ys)
            e_zeta = 9 + zs*2 + ys
            if is_edge[b, e_xi] && is_edge[b, e_eta] && is_edge[b, e_zeta]
                is_corner[b, c, 1] = true
            end
        end
    end

    return EdgeSingularityInfo(is_edge, is_corner, n_blocks)
end

# ─────────────────────────────────────────────────────────────────────────
# Per-cell query
# ─────────────────────────────────────────────────────────────────────────

"""
    is_near_singularity_edge(bid, i, j, k, NG, nxp, nyp, nzp, sing, n_protect) -> Bool

Return `true` if cell `(i, j, k)` of block `bid` is within `n_protect` index
layers of any singularity edge of that block.

`bid` is 1-indexed (matching the `face_bc`/`neighbors` convention). `i, j, k`
are 1-indexed cell indices in the full padded array (ghosts at 1..NG and
Nx+NG+1..Nx+2*NG). `nxp, nyp, nzp` are the INTERIOR extents (without ghosts).

Layer distance is computed in index space: for an edge along ξ pinned at
(j_pin, k_pin), the distance of cell (i, j, k) is `max(|j - j_pin|, |k - k_pin|)`
(in interior-index space, i.e. relative to NG+1). The cell is "near" if this
distance is ≤ n_protect - 1 (layer 1 = on the edge, distance 0).
"""
function is_near_singularity_edge(bid::Int, i::Int, j::Int, k::Int,
                                   NG::Int, nxp::Int, nyp::Int, nzp::Int,
                                   sing::EdgeSingularityInfo,
                                   n_protect::Int)::Bool
    # Convert to interior-relative indices (1 = first interior cell)
    ii = i - NG
    jj = j - NG
    kk = k - NG
    # Quick reject: must be in or near the padded interior
    if ii < 1 - n_protect || ii > nxp + n_protect
        return false
    end
    if jj < 1 - n_protect || jj > nyp + n_protect
        return false
    end
    if kk < 1 - n_protect || kk > nzp + n_protect
        return false
    end

    for e in 1:12
        if !sing.is_singularity_edge[bid, e]
            continue
        end
        # Decode the two pinned axes for this edge
        axis_a, high_a, axis_b, high_b = _EDGE_PINS[e]
        # Compute the pinned interior index for each axis.
        # For a "low" face (high=0), the edge is at interior index 1.
        # For a "high" face (high=1), the edge is at interior index N (extent).
        # axis: 1=i, 2=j, 3=k
        pin_a = if axis_a == 1
            high_a == 0 ? 1 : nxp
        elseif axis_a == 2
            high_a == 0 ? 1 : nyp
        else
            high_a == 0 ? 1 : nzp
        end
        pin_b = if axis_b == 1
            high_b == 0 ? 1 : nxp
        elseif axis_b == 2
            high_b == 0 ? 1 : nyp
        else
            high_b == 0 ? 1 : nzp
        end
        # Distance from the cell to the edge in the pinned plane.
        # The free axis (the one not in {axis_a, axis_b}) doesn't matter —
        # the edge runs along it.
        coord_a = axis_a == 1 ? ii : (axis_a == 2 ? jj : kk)
        coord_b = axis_b == 1 ? ii : (axis_b == 2 ? jj : kk)
        dist = max(abs(coord_a - pin_a), abs(coord_b - pin_b))
        if dist <= n_protect - 1
            return true
        end
    end
    return false
end

# ─────────────────────────────────────────────────────────────────────────
# ϕ boost (host-side, mirrors apply_geometric_smoothness_protection!)
# ─────────────────────────────────────────────────────────────────────────

"""
    apply_topology_smoothness_protection!(
        Lin_j, ΔLin_j, phi_j,
        Lin_k, ΔLin_k, phi_k,
        bid, nxp, nyp, nzp, NG, sing;
        n_protect = _TOPO_N_PROTECT,
        phi_profile = _TOPO_PHI_PROFILE,
        verbose = false
    )

Boost the WENO5 upwind parameter ϕ for cells near any singularity edge of
block `bid`. This is the topological complement to
`apply_geometric_smoothness_protection!`: it catches multi-block junction
singularities that the geometric criterion (face-normal angle, area stretch)
misses.

The boost profile and layer count default to Q1=A and Q2=A (matching the
existing geometric protection): `[1.0, 1.0, 0.8, 0.6]` over 4 layers.

This function modifies `phi_j` and `phi_k` in place. If a cell is already
boosted by the geometric protection (ϕ already high), the topological
boost only raises it further if its profile value is larger (Q3=A: take max).
"""
function apply_topology_smoothness_protection!(
    Lin_j, ΔLin_j, phi_j,
    Lin_k, ΔLin_k, phi_k,
    bid::Int, nxp::Int, nyp::Int, nzp::Int, NG::Int,
    sing::EdgeSingularityInfo;
    n_protect::Int = _TOPO_N_PROTECT,
    phi_profile::Vector{Float64} = _TOPO_PHI_PROFILE,
    verbose::Bool = false
)
    FTl = eltype(phi_j)
    n_eff = min(n_protect, length(phi_profile))
    n_boosted = 0

    # NOTE on array layout (matches apply_geometric_smoothness_protection!):
    #   Lin_j, ΔLin_j : (Nj, Nk, 7)  — stencil weights for η-direction recon,
    #                                    constant along ξ, indexed [j, k, s]
    #   phi_j          : (Nj, Nk)     — η upwind param, indexed [j, k]
    #   Lin_k, ΔLin_k : (Nj, Nk, 7)  — stencil weights for ζ-direction recon,
    #                                    constant along ξ, indexed [j, k, s]
    #   phi_k          : (Nj, Nk)     — ζ upwind param, indexed [j, k]
    # The i (ξ) direction does not appear in these arrays. We iterate only
    # over (j, k) and boost both phi_j and phi_k at [j, k].
    @inbounds for k in (NG+1):(nzp+NG)
        for j in (NG+1):(nyp+NG)
            # Find the minimum layer distance to any singularity edge.
            # Layer 1 = on the edge (dist 0), layer L = dist L-1.
            min_layer = n_eff + 1   # sentinel: "not near"
            for e in 1:12
                if !sing.is_singularity_edge[bid, e]
                    continue
                end
                axis_a, high_a, axis_b, high_b = _EDGE_PINS[e]
                pin_a = if axis_a == 1
                    high_a == 0 ? 1 : nxp
                elseif axis_a == 2
                    high_a == 0 ? 1 : nyp
                else
                    high_a == 0 ? 1 : nzp
                end
                pin_b = if axis_b == 1
                    high_b == 0 ? 1 : nxp
                elseif axis_b == 2
                    high_b == 0 ? 1 : nyp
                else
                    high_b == 0 ? 1 : nzp
                end
                jj = j - NG
                kk = k - NG
                # For edges along ξ (axis_a, axis_b ∈ {2,3}): both pins are
                # in (η, ζ) plane, distance = max(|jj-pin_a|, |kk-pin_b|).
                # For edges along η (axis ∈ {1,3}): one pin is ξ, which is
                # constant along η — every i is on the edge, so the ξ-pin
                # distance is always 0. Same for edges along ζ.
                # → coord for axis=1 (ξ) is irrelevant; use 0 distance.
                coord_a = axis_a == 1 ? 0 : (axis_a == 2 ? jj : kk)
                coord_b = axis_b == 1 ? 0 : (axis_b == 2 ? jj : kk)
                pin_a_eff = axis_a == 1 ? 0 : pin_a
                pin_b_eff = axis_b == 1 ? 0 : pin_b
                dist = max(abs(coord_a - pin_a_eff), abs(coord_b - pin_b_eff))
                layer = dist + 1
                if layer < min_layer
                    min_layer = layer
                end
            end

            if min_layer > n_eff
                continue
            end

            boost_val = FTl(phi_profile[min_layer])
            # Boost phi_j and phi_k at [j, k]. Q3=A: only boost if our value
            # is higher (take max with existing geometric protection).
            if boost_val > phi_j[j, k]
                Δϕ = boost_val - phi_j[j, k]
                for s in 1:7
                    Lin_j[j, k, s] += FTl(Δϕ) * ΔLin_j[j, k, s]
                end
                phi_j[j, k] = boost_val
                n_boosted += 1
            end
            if boost_val > phi_k[j, k]
                Δϕ = boost_val - phi_k[j, k]
                for s in 1:7
                    Lin_k[j, k, s] += FTl(Δϕ) * ΔLin_k[j, k, s]
                end
                phi_k[j, k] = boost_val
                n_boosted += 1
            end
        end
    end

    if verbose && n_boosted > 0
        println("      > Block $(bid-1) topological singularity protection: boosted $n_boosted ϕ entries (n_protect=$n_eff).")
    end
    return n_boosted
end
