using StaticArrays: SVector

if !@isdefined(STRUCTURED_FACE_QUADRATURE_LOADED)
const STRUCTURED_FACE_QUADRATURE_LOADED = true
if !@isdefined(STRUCTURED_FACE_MIDPOINT)
    const STRUCTURED_FACE_MIDPOINT = Int32(2)
end
if !@isdefined(STRUCTURED_FACE_POINT6)
    const STRUCTURED_FACE_POINT6 = Int32(6)
end
if !@isdefined(structured_face_quadrature)
    const structured_face_quadrature = STRUCTURED_FACE_MIDPOINT
end
if !@isdefined(CT_FACE_GEOMETRY_MIDPOINT)
    const CT_FACE_GEOMETRY_MIDPOINT = Int32(2)
end
if !@isdefined(CT_FACE_GEOMETRY_POINT6)
    const CT_FACE_GEOMETRY_POINT6 = Int32(6)
end
if !@isdefined(ct_face_geometry)
    const ct_face_geometry =
        structured_face_quadrature == STRUCTURED_FACE_POINT6 ?
        CT_FACE_GEOMETRY_POINT6 : CT_FACE_GEOMETRY_MIDPOINT
end
if !@isdefined(CT_FACE_BN_AVERAGE)
    const CT_FACE_BN_AVERAGE = Int32(2)
end
if !@isdefined(CT_FACE_BN_POINT6)
    const CT_FACE_BN_POINT6 = Int32(6)
end
if !@isdefined(CT_FACE_BN_ADAPTIVE)
    const CT_FACE_BN_ADAPTIVE = Int32(7)
end
if !@isdefined(ct_face_bn_recovery)
    const ct_face_bn_recovery =
        ct_face_geometry == CT_FACE_GEOMETRY_POINT6 ?
        CT_FACE_BN_ADAPTIVE : CT_FACE_BN_AVERAGE
end
if !@isdefined(STRUCTURED_FLUX_TANGENTIAL_HALO)
    const STRUCTURED_FLUX_TANGENTIAL_HALO =
        structured_face_quadrature == STRUCTURED_FACE_POINT6 ? Int32(2) :
        ((@isdefined(ct_mode) && ct_mode) ? Int32(1) : Int32(0))
end

@inline function structured_point_to_average6(v::SVector{5,T}) where {T}
    return -T(17)/T(5760)*v[1] + T(77)/T(1440)*v[2] +
           T(863)/T(960)*v[3] + T(77)/T(1440)*v[4] -
           T(17)/T(5760)*v[5]
end

const STRUCTURED_P2A_FIXED = Int32(0)
const STRUCTURED_P2A_AO = Int32(1)
const STRUCTURED_P2A_MIDPOINT = Int32(2)

@inline function _structured_face_p2a_adaptive_enabled()
    @static if @isdefined(ct_face_p2a_adaptive)
        return ct_face_p2a_adaptive
    else
        return false
    end
end

@inline function _structured_face_p2a_threshold(::Type{T}) where {T}
    @static if @isdefined(ct_face_p2a_smoothness_threshold)
        return T(ct_face_p2a_smoothness_threshold)
    else
        return T(5.0e-1)
    end
end

@inline function _structured_face_p2a_high_weight(::Type{T}) where {T}
    @static if @isdefined(ct_face_p2a_ao_high_weight)
        return T(ct_face_p2a_ao_high_weight)
    else
        return T(0.85)
    end
end

@inline function _structured_face_p2a_gate_fraction(::Type{T}) where {T}
    @static if @isdefined(ct_face_p2a_ao_gate_fraction)
        return T(ct_face_p2a_ao_gate_fraction)
    else
        return T(0.8)
    end
end

@inline function _structured_p2a_beta_high(v::SVector{5,T}) where {T}
    a1 = -(T(34)*v[2]-T(5)*v[1]-T(34)*v[4]+T(5)*v[5])/T(48)
    a2 = -(T(22)*v[3]-T(12)*v[2]+v[1]-T(12)*v[4]+v[5])/T(16)
    a3 = (T(2)*v[2]-v[1]-T(2)*v[4]+v[5])/T(12)
    a4 = (T(6)*v[3]-T(4)*v[2]+v[1]-T(4)*v[4]+v[5])/T(24)
    return max(
        zero(T),
        a1*a1 + T(0.5)*a1*a3 + T(13)/T(3)*a2*a2 +
        T(21)/T(5)*a2*a4 + T(3129)/T(80)*a3*a3 +
        T(87617)/T(140)*a4*a4,
    )
end

@inline function _structured_p2a_beta_low(v::SVector{5,T}) where {T}
    left = T(13)/T(12)*(v[1]-T(2)*v[2]+v[3])^2 +
           T(0.25)*(v[1]-T(4)*v[2]+T(3)*v[3])^2
    center = T(13)/T(12)*(v[2]-T(2)*v[3]+v[4])^2 +
             T(0.25)*(v[2]-v[4])^2
    right = T(13)/T(12)*(v[3]-T(2)*v[4]+v[5])^2 +
            T(0.25)*(T(3)*v[3]-T(4)*v[4]+v[5])^2
    return SVector{3,T}(left,center,right)
end

@inline function _structured_apply_p2a_coefficients(
    coefficients::SVector{5,T}, values::SVector{5,T},
) where {T}
    return coefficients[1]*values[1] + coefficients[2]*values[2] +
           coefficients[3]*values[3] + coefficients[4]*values[4] +
           coefficients[5]*values[5]
end

@inline function structured_point_to_average_ao(
    values::SVector{5,T}, threshold::T, gamma_high::T,
) where {T}
    finite_stencil = true
    scale = zero(T)
    for index in 1:5
        finite_stencil &= isfinite(values[index])
        scale = max(scale,abs(values[index]))
    end
    if !finite_stencil
        return values[3],STRUCTURED_P2A_MIDPOINT,one(T)
    elseif iszero(scale)
        return structured_point_to_average6(values),STRUCTURED_P2A_FIXED,zero(T)
    end

    normalized = values/scale
    beta_low = _structured_p2a_beta_low(normalized)
    beta_high = _structured_p2a_beta_high(normalized)
    tau = (
        abs(beta_high-beta_low[1]) + abs(beta_high-beta_low[2]) +
        abs(beta_high-beta_low[3])
    )/T(3)
    beta_scale = max(
        beta_high,(beta_low[1]+beta_low[2]+beta_low[3])/T(3),
    )
    epsilon_beta = T(64)*eps(T)
    sensor = tau/(beta_scale+epsilon_beta)
    if !isfinite(sensor)
        return values[3],STRUCTURED_P2A_MIDPOINT,one(T)
    elseif sensor <= threshold
        return structured_point_to_average6(values),STRUCTURED_P2A_FIXED,sensor
    end

    gamma_low = (one(T)-gamma_high)/T(3)
    alpha_high = gamma_high*(
        one(T)+(tau/(beta_high+epsilon_beta))^2
    )
    alpha_left = gamma_low*(
        one(T)+(tau/(beta_low[1]+epsilon_beta))^2
    )
    alpha_center = gamma_low*(
        one(T)+(tau/(beta_low[2]+epsilon_beta))^2
    )
    alpha_right = gamma_low*(
        one(T)+(tau/(beta_low[3]+epsilon_beta))^2
    )
    alpha_sum = alpha_high+alpha_left+alpha_center+alpha_right
    if !(isfinite(alpha_sum) && alpha_sum > zero(T))
        return values[3],STRUCTURED_P2A_MIDPOINT,sensor
    end

    fixed = SVector{5,T}(
        -T(17)/T(5760),T(77)/T(1440),T(863)/T(960),
        T(77)/T(1440),-T(17)/T(5760),
    )
    left = SVector{5,T}(
        T(1)/T(24),-T(1)/T(12),T(25)/T(24),zero(T),zero(T),
    )
    center = SVector{5,T}(
        zero(T),T(1)/T(24),T(11)/T(12),T(1)/T(24),zero(T),
    )
    right = SVector{5,T}(
        zero(T),zero(T),T(25)/T(24),-T(1)/T(12),T(1)/T(24),
    )
    corrected_high = (
        fixed-gamma_low*(left+center+right)
    )/gamma_high
    coefficients = (
        alpha_high*corrected_high + alpha_left*left +
        alpha_center*center + alpha_right*right
    )/alpha_sum
    coefficient_sum = sum(coefficients)
    if !(isfinite(coefficient_sum) && abs(coefficient_sum) > eps(T))
        return values[3],STRUCTURED_P2A_MIDPOINT,sensor
    end
    candidate = _structured_apply_p2a_coefficients(
        coefficients/coefficient_sum,values,
    )
    return isfinite(candidate) ?
        (candidate,STRUCTURED_P2A_AO,sensor) :
        (values[3],STRUCTURED_P2A_MIDPOINT,sensor)
end

const STRUCTURED_A2P_FIXED = Int32(0)
const STRUCTURED_A2P_AO = Int32(1)
const STRUCTURED_A2P_LIMITED = Int32(2)
const STRUCTURED_A2P_AVERAGE = Int32(3)

@inline function _structured_face_bn_threshold(::Type{T}) where {T}
    @static if @isdefined(ct_face_bn_smoothness_threshold)
        return T(ct_face_bn_smoothness_threshold)
    else
        return T(5.0e-1)
    end
end

@inline function _structured_face_bn_high_weight(::Type{T}) where {T}
    @static if @isdefined(ct_face_bn_ao_high_weight)
        return T(ct_face_bn_ao_high_weight)
    else
        return T(0.85)
    end
end

@inline function _structured_a2p_fixed_coefficients(::Type{T}) where {T}
    return SVector{5,T}(
        T(3)/T(640),-T(29)/T(480),T(1067)/T(960),
        -T(29)/T(480),T(3)/T(640),
    )
end

@inline function structured_average_to_point_ao_coefficients(
    values::SVector{5,T}, threshold::T, gamma_high::T,
) where {T}
    finite_stencil = true
    scale = zero(T)
    for index in 1:5
        finite_stencil &= isfinite(values[index])
        scale = max(scale,abs(values[index]))
    end
    center = SVector{5,T}(zero(T),zero(T),one(T),zero(T),zero(T))
    fixed = _structured_a2p_fixed_coefficients(T)
    if !finite_stencil
        return center,STRUCTURED_A2P_AVERAGE,one(T)
    elseif iszero(scale)
        return fixed,STRUCTURED_A2P_FIXED,zero(T)
    end

    normalized = values/scale
    beta_low = _structured_p2a_beta_low(normalized)
    beta_high = _structured_p2a_beta_high(normalized)
    tau = (
        abs(beta_high-beta_low[1]) + abs(beta_high-beta_low[2]) +
        abs(beta_high-beta_low[3])
    )/T(3)
    beta_scale = max(
        beta_high,(beta_low[1]+beta_low[2]+beta_low[3])/T(3),
    )
    epsilon_beta = T(64)*eps(T)
    sensor = tau/(beta_scale+epsilon_beta)
    if !isfinite(sensor)
        return center,STRUCTURED_A2P_AVERAGE,one(T)
    elseif sensor <= threshold
        return fixed,STRUCTURED_A2P_FIXED,sensor
    end

    gamma_low = (one(T)-gamma_high)/T(3)
    alpha_high = gamma_high*(
        one(T)+(tau/(beta_high+epsilon_beta))^2
    )
    alpha_left = gamma_low*(
        one(T)+(tau/(beta_low[1]+epsilon_beta))^2
    )
    alpha_center = gamma_low*(
        one(T)+(tau/(beta_low[2]+epsilon_beta))^2
    )
    alpha_right = gamma_low*(
        one(T)+(tau/(beta_low[3]+epsilon_beta))^2
    )
    alpha_sum = alpha_high+alpha_left+alpha_center+alpha_right
    if !(isfinite(alpha_sum) && alpha_sum > zero(T))
        return center,STRUCTURED_A2P_AVERAGE,sensor
    end

    left = SVector{5,T}(
        -T(1)/T(24),T(1)/T(12),T(23)/T(24),zero(T),zero(T),
    )
    center_stencil = SVector{5,T}(
        zero(T),-T(1)/T(24),T(13)/T(12),-T(1)/T(24),zero(T),
    )
    right = SVector{5,T}(
        zero(T),zero(T),T(23)/T(24),T(1)/T(12),-T(1)/T(24),
    )
    corrected_high = (
        fixed-gamma_low*(left+center_stencil+right)
    )/gamma_high
    coefficients = (
        alpha_high*corrected_high + alpha_left*left +
        alpha_center*center_stencil + alpha_right*right
    )/alpha_sum
    coefficient_sum = sum(coefficients)
    if !(isfinite(coefficient_sum) && abs(coefficient_sum) > eps(T))
        return center,STRUCTURED_A2P_AVERAGE,sensor
    end
    return coefficients/coefficient_sum,STRUCTURED_A2P_AO,sensor
end

@inline function _structured_convex_bound(
    center::T, candidate::T, lower::T, upper::T,
) where {T}
    if !(isfinite(center) && isfinite(candidate) &&
         isfinite(lower) && isfinite(upper) && lower <= center <= upper)
        return center,true
    end
    tolerance = T(64)*eps(T)*max(one(T),abs(lower),abs(upper))
    if lower-tolerance <= candidate <= upper+tolerance
        return clamp(candidate,lower,upper),false
    end
    delta = candidate-center
    if iszero(delta)
        return center,false
    end
    theta = delta > zero(T) ? (upper-center)/delta : (lower-center)/delta
    theta = clamp(theta,zero(T),one(T))
    return center+theta*delta,true
end

@inline function structured_average_to_point_ao(
    values::SVector{5,T}, threshold::T, gamma_high::T,
) where {T}
    coefficients,mode,sensor = structured_average_to_point_ao_coefficients(
        values,threshold,gamma_high,
    )
    candidate = _structured_apply_p2a_coefficients(coefficients,values)
    mode == STRUCTURED_A2P_FIXED && return candidate,mode,sensor
    bounded,limited = _structured_convex_bound(
        values[3],candidate,minimum(values),maximum(values),
    )
    return bounded,
           (limited ? STRUCTURED_A2P_LIMITED : mode),sensor
end

@inline function _structured_a2p6_center_weight(::Type{T}, sample) where {T}
    return sample == 1 || sample == 5 ? T(3)/T(640) :
           (sample == 2 || sample == 4 ? -T(29)/T(480) :
            (sample == 3 ? T(1067)/T(960) : zero(T)))
end

@inline function _structured_a2p6_low_weight(
    ::Type{T}, target::Int32, sample::Int32,
) where {T}
    if target == Int32(1)
        return sample == 1 ? T(1627)/T(1920) :
               sample == 2 ? T(497)/T(960) :
               sample == 3 ? -T(683)/T(960) :
               sample == 4 ? T(41)/T(80) :
               sample == 5 ? -T(127)/T(640) : T(31)/T(960)
    end
    return sample == 1 ? -T(31)/T(960) :
           sample == 2 ? T(1999)/T(1920) :
           sample == 3 ? T(1)/T(30) :
           sample == 4 ? -T(21)/T(320) :
           sample == 5 ? T(9)/T(320) : -T(3)/T(640)
end

@inline function _structured_a2p6_sample(
    ::Type{T}, target, extent, sample,
) where {T}
    target_i = Int32(target)
    extent_i = Int32(extent)
    sample_i = Int32(sample)
    if extent_i < Int32(6)
        return target_i, sample_i == Int32(1) ? one(T) : zero(T)
    elseif target_i <= Int32(2)
        return sample_i, _structured_a2p6_low_weight(T, target_i, sample_i)
    elseif target_i >= extent_i-Int32(1)
        mirrored_target = extent_i-target_i+Int32(1)
        mirrored_sample = Int32(7)-sample_i
        return extent_i-Int32(6)+sample_i,
               _structured_a2p6_low_weight(T, mirrored_target, mirrored_sample)
    end
    if sample_i == Int32(6)
        return target_i, zero(T)
    end
    return target_i+sample_i-Int32(3),
           _structured_a2p6_center_weight(T, sample_i)
end

@inline function _structured_optional_face_value(
    field, i, j, k, ::Type{T},
) where {T}
    field === nothing && return zero(T)
    @inbounds return field[i,j,k]
end

@inline function structured_face_point_data6(
    face_field, background_field,
    area, normal_x, normal_y, normal_z,
    i, j, k, ::Val{DIRECTION},
) where {DIRECTION}
    T = eltype(area)
    first_axis = DIRECTION == 1 ? Int32(2) : Int32(1)
    second_axis = DIRECTION == 3 ? Int32(2) : Int32(3)
    first_target = first_axis == 1 ? i : j
    second_target = second_axis == 2 ? j : k
    first_extent = size(area, first_axis)
    second_extent = size(area, second_axis)
    field_point = zero(T)
    area_x_point = zero(T)
    area_y_point = zero(T)
    area_z_point = zero(T)
    for first_sample in Int32(1):Int32(6)
        first_index, first_weight = _structured_a2p6_sample(
            T, first_target, first_extent, first_sample,
        )
        for second_sample in Int32(1):Int32(6)
            second_index, second_weight = _structured_a2p6_sample(
                T, second_target, second_extent, second_sample,
            )
            weight = first_weight*second_weight
            ii = first_axis == 1 ? first_index : i
            jj = first_axis == 2 ? first_index :
                 (second_axis == 2 ? second_index : j)
            kk = second_axis == 3 ? second_index : k
            @inbounds begin
                local_area = area[ii,jj,kk]
                field_point += weight*(
                    _structured_optional_face_value(face_field,ii,jj,kk,T) +
                    _structured_optional_face_value(background_field,ii,jj,kk,T)
                )
                area_x_point += weight*local_area*normal_x[ii,jj,kk]
                area_y_point += weight*local_area*normal_y[ii,jj,kk]
                area_z_point += weight*local_area*normal_z[ii,jj,kk]
            end
        end
    end
    return SVector{4,T}(
        field_point, area_x_point, area_y_point, area_z_point,
    )
end

@inline function _structured_total_face_value(
    face_field, background_field, i, j, k, ::Type{T},
) where {T}
    return _structured_optional_face_value(face_field,i,j,k,T) +
           _structured_optional_face_value(background_field,i,j,k,T)
end

@inline function _structured_joint_state_sensor(
    values::NTuple{5,SVector{2,T}},
) where {T}
    density_scale = zero(T)
    pressure_scale = zero(T)
    finite_stencil = true
    for index in 1:5
        density_scale = max(density_scale,abs(values[index][1]))
        pressure_scale = max(pressure_scale,abs(values[index][2]))
        finite_stencil &= all(isfinite,values[index])
    end
    finite_stencil || return one(T)
    beta_low = SVector{3,T}(zero(T),zero(T),zero(T))
    beta_high = zero(T)
    for component in 1:2
        scale = component == 1 ? density_scale : pressure_scale
        iszero(scale) && continue
        normalized = SVector{5,T}(ntuple(
            index -> values[index][component]/scale,Val(5),
        ))
        beta_low += _structured_p2a_beta_low(normalized)
        beta_high += _structured_p2a_beta_high(normalized)
    end
    tau = (
        abs(beta_high-beta_low[1]) + abs(beta_high-beta_low[2]) +
        abs(beta_high-beta_low[3])
    )/T(3)
    beta_scale = max(
        beta_high,(beta_low[1]+beta_low[2]+beta_low[3])/T(3),
    )
    sensor = tau/(beta_scale+T(64)*eps(T))
    return isfinite(sensor) ? sensor : one(T)
end

@inline function _structured_face_state_line_sensor(
    state_field, i, j, k, ::Val{DIRECTION}, axis::Int32,
) where {DIRECTION}
    state_field === nothing && return 0.0
    T = eltype(state_field)
    di = DIRECTION == 1 ? Int32(1) : Int32(0)
    dj = DIRECTION == 2 ? Int32(1) : Int32(0)
    dk = DIRECTION == 3 ? Int32(1) : Int32(0)
    low_i,low_j,low_k = i-di,j-dj,k-dk
    values = ntuple(Val(5)) do sample
        offset = Int32(sample-3)
        oi = axis == Int32(1) ? offset : Int32(0)
        oj = axis == Int32(2) ? offset : Int32(0)
        ok = axis == Int32(3) ? offset : Int32(0)
        li = clamp(low_i+oi,Int32(1),Int32(size(state_field,1)))
        lj = clamp(low_j+oj,Int32(1),Int32(size(state_field,2)))
        lk = clamp(low_k+ok,Int32(1),Int32(size(state_field,3)))
        hi = clamp(low_i+di+oi,Int32(1),Int32(size(state_field,1)))
        hj = clamp(low_j+dj+oj,Int32(1),Int32(size(state_field,2)))
        hk = clamp(low_k+dk+ok,Int32(1),Int32(size(state_field,3)))
        @inbounds SVector{2,T}(
            (state_field[li,lj,lk,1]+state_field[hi,hj,hk,1])/T(2),
            (state_field[li,lj,lk,5]+state_field[hi,hj,hk,5])/T(2),
        )
    end
    return _structured_joint_state_sensor(values)
end

@inline function _structured_face_state_support_sensor(
    state_field, i, j, k, direction, first_axis, second_axis, ::Type{T},
) where {T}
    state_field === nothing && return zero(T)
    return max(
        _structured_face_state_line_sensor(
            state_field,i,j,k,direction,first_axis,
        ),
        _structured_face_state_line_sensor(
            state_field,i,j,k,direction,second_axis,
        ),
    )
end

@inline function structured_face_point_bn_ao(
    face_field, background_field, area,
    i, j, k, ::Val{DIRECTION}, fixed_flux_point::T, point_area::T,
    state_field=nothing,
) where {DIRECTION,T}
    first_axis = DIRECTION == 1 ? Int32(2) : Int32(1)
    second_axis = DIRECTION == 3 ? Int32(2) : Int32(3)
    first_target = first_axis == 1 ? i : j
    second_target = second_axis == 2 ? j : k
    first_extent = size(area,first_axis)
    second_extent = size(area,second_axis)
    @inbounds center_area = area[i,j,k]
    center_flux = _structured_total_face_value(
        face_field,background_field,i,j,k,T,
    )
    center_bn = center_flux/center_area
    central_stencil =
        first_extent >= Int32(6) && second_extent >= Int32(6) &&
        first_target > Int32(2) && first_target < first_extent-Int32(1) &&
        second_target > Int32(2) && second_target < second_extent-Int32(1)
    if !central_stencil
        candidate = fixed_flux_point/point_area
        return isfinite(candidate) ?
            (candidate,STRUCTURED_A2P_FIXED,zero(T)) :
            (center_bn,STRUCTURED_A2P_AVERAGE,one(T))
    end

    first_values = SVector{5,T}(ntuple(Val(5)) do sample
        offset = Int32(sample-3)
        ii = first_axis == Int32(1) ? i+offset : i
        jj = first_axis == Int32(2) ? j+offset : j
        _structured_total_face_value(
            face_field,background_field,ii,jj,k,T,
        )
    end)
    second_values = SVector{5,T}(ntuple(Val(5)) do sample
        offset = Int32(sample-3)
        jj = second_axis == Int32(2) ? j+offset : j
        kk = second_axis == Int32(3) ? k+offset : k
        _structured_total_face_value(
            face_field,background_field,i,jj,kk,T,
        )
    end)
    threshold = _structured_face_bn_threshold(T)
    high_weight = _structured_face_bn_high_weight(T)
    first_coefficients,first_mode,first_sensor =
        structured_average_to_point_ao_coefficients(
            first_values,threshold,high_weight,
        )
    second_coefficients,second_mode,second_sensor =
        structured_average_to_point_ao_coefficients(
            second_values,threshold,high_weight,
        )
    state_sensor = _structured_face_state_support_sensor(
        state_field,i,j,k,Val(DIRECTION),first_axis,second_axis,T,
    )
    sensor = max(first_sensor,second_sensor,state_sensor)
    if state_sensor > threshold
        return center_bn,STRUCTURED_A2P_AVERAGE,sensor
    end
    if first_mode == STRUCTURED_A2P_FIXED &&
       second_mode == STRUCTURED_A2P_FIXED
        return fixed_flux_point/point_area,STRUCTURED_A2P_FIXED,sensor
    elseif first_mode == STRUCTURED_A2P_AVERAGE ||
           second_mode == STRUCTURED_A2P_AVERAGE
        return center_bn,STRUCTURED_A2P_AVERAGE,sensor
    end

    flux_point = zero(T)
    lower_bn = center_bn
    upper_bn = center_bn
    finite_support = isfinite(center_bn)
    for first_sample in Int32(1):Int32(5)
        first_offset = first_sample-Int32(3)
        for second_sample in Int32(1):Int32(5)
            second_offset = second_sample-Int32(3)
            ii = DIRECTION == 1 ? i : i+first_offset
            jj = DIRECTION == 1 ? j+first_offset :
                 (DIRECTION == 2 ? j : j+second_offset)
            kk = DIRECTION == 3 ? k : k+second_offset
            @inbounds local_area = area[ii,jj,kk]
            local_flux = _structured_total_face_value(
                face_field,background_field,ii,jj,kk,T,
            )
            local_bn = local_flux/local_area
            finite_support &= isfinite(local_bn)
            lower_bn = min(lower_bn,local_bn)
            upper_bn = max(upper_bn,local_bn)
            flux_point += first_coefficients[first_sample]*
                          second_coefficients[second_sample]*local_flux
        end
    end
    if !(finite_support && isfinite(flux_point) &&
         isfinite(point_area) && point_area > zero(T))
        return center_bn,STRUCTURED_A2P_AVERAGE,sensor
    end
    candidate_bn = flux_point/point_area
    bounded_bn,limited = _structured_convex_bound(
        center_bn,candidate_bn,lower_bn,upper_bn,
    )
    return bounded_bn,
           (limited ? STRUCTURED_A2P_LIMITED : STRUCTURED_A2P_AO),sensor
end

@inline function structured_face_point_geometry6(
    area, normal_x, normal_y, normal_z, i, j, k, direction,
)
    data = structured_face_point_data6(
        nothing, nothing, area, normal_x, normal_y, normal_z,
        i, j, k, direction,
    )
    T = eltype(area)
    area_vector = SVector{3,T}(data[2],data[3],data[4])
    point_area = sqrt(sum(abs2,area_vector))
    inverse_area = inv(point_area)
    return point_area, area_vector[1]*inverse_area,
           area_vector[2]*inverse_area, area_vector[3]*inverse_area
end

@inline function structured_face_geometry(
    area, normal_x, normal_y, normal_z, i, j, k, direction,
)
    if structured_face_quadrature == STRUCTURED_FACE_POINT6
        return structured_face_point_geometry6(
            area,normal_x,normal_y,normal_z,i,j,k,direction,
        )
    else
        @inbounds return area[i,j,k],normal_x[i,j,k],normal_y[i,j,k],normal_z[i,j,k]
    end
end

@inline function structured_ct_face_geometry(
    area, normal_x, normal_y, normal_z, i, j, k, direction,
)
    if ct_face_geometry == CT_FACE_GEOMETRY_POINT6
        return structured_face_point_geometry6(
            area,normal_x,normal_y,normal_z,i,j,k,direction,
        )
    end
    @inbounds return area[i,j,k],normal_x[i,j,k],normal_y[i,j,k],normal_z[i,j,k]
end

@inline function structured_ct_face_geometry_bn(
    face_field, background_field,
    area, normal_x, normal_y, normal_z,
    i, j, k, direction, state_field=nothing,
    recovery_mode=ct_face_bn_recovery,
)
    if ct_face_geometry == CT_FACE_GEOMETRY_POINT6
        if recovery_mode != CT_FACE_BN_AVERAGE
            data = structured_face_point_data6(
                face_field,background_field,area,normal_x,normal_y,normal_z,
                i,j,k,direction,
            )
            T = eltype(area)
            area_vector = SVector{3,T}(data[2],data[3],data[4])
            point_area = sqrt(sum(abs2,area_vector))
            inverse_area = inv(point_area)
            if recovery_mode == CT_FACE_BN_ADAPTIVE
                face_bn,_,_ = structured_face_point_bn_ao(
                    face_field,background_field,area,
                    i,j,k,direction,data[1],point_area,state_field,
                )
                return point_area,area_vector[1]*inverse_area,
                       area_vector[2]*inverse_area,area_vector[3]*inverse_area,
                       face_bn
            end
            return point_area,area_vector[1]*inverse_area,
                   area_vector[2]*inverse_area,area_vector[3]*inverse_area,
                   data[1]*inverse_area
        end
        point_area,point_nx,point_ny,point_nz =
            structured_face_point_geometry6(
                area,normal_x,normal_y,normal_z,i,j,k,direction,
            )
        @inbounds face_bn = (
            _structured_optional_face_value(face_field,i,j,k,eltype(area)) +
            _structured_optional_face_value(background_field,i,j,k,eltype(area))
        )/area[i,j,k]
        return point_area,point_nx,point_ny,point_nz,face_bn
    else
        @inbounds begin
            local_area=area[i,j,k]
            face_bn=(face_field[i,j,k]+
                _structured_optional_face_value(background_field,i,j,k,eltype(area)))/local_area
            return local_area,normal_x[i,j,k],normal_y[i,j,k],normal_z[i,j,k],face_bn
        end
    end
end

@inline function structured_ct_split_face_geometry(
    face_field, background_field,
    area, normal_x, normal_y, normal_z,
    i, j, k, direction,
)
    if background_field === nothing
        local_area, nx, ny, nz, face_bn = structured_ct_face_geometry_bn(
            face_field, nothing, area, normal_x, normal_y, normal_z,
            i, j, k, direction,
        )
        return local_area, nx, ny, nz, face_bn, face_bn, zero(face_bn)
    end
    local_area, nx, ny, nz, perturbation_bn =
        structured_ct_face_geometry_bn(
            face_field, nothing, area, normal_x, normal_y, normal_z,
            i, j, k, direction,
        )
    _, _, _, _, background_bn = structured_ct_face_geometry_bn(
        background_field, nothing, area, normal_x, normal_y, normal_z,
        i, j, k, direction,
    )
    return local_area, nx, ny, nz,
           perturbation_bn + background_bn, perturbation_bn, background_bn
end

@inline function _structured_p2a_axis_value(
    source, i, j, k, component, axis::Int32, extent::Int32,
)
    if extent <= Int32(1)
        @inbounds return source[i,j,k,component]
    end
    T = eltype(source)
    values = SVector{5,T}(ntuple(Val(5)) do sample
        offset = Int32(sample-3)
        ii = axis == Int32(1) ? i+offset : i
        jj = axis == Int32(2) ? j+offset : j
        kk = axis == Int32(3) ? k+offset : k
        @inbounds source[ii,jj,kk,component]
    end)
    return structured_point_to_average6(values)
end

@inline function _structured_p2a_axis_adaptive_value(
    source, i, j, k, component, axis::Int32, extent::Int32,
    allow_adaptive::Bool=true, force_adaptive::Bool=false,
)
    if extent <= Int32(1)
        @inbounds return source[i,j,k,component],STRUCTURED_P2A_FIXED
    end
    T = eltype(source)
    values = SVector{5,T}(ntuple(Val(5)) do sample
        offset = Int32(sample-3)
        ii = axis == Int32(1) ? i+offset : i
        jj = axis == Int32(2) ? j+offset : j
        kk = axis == Int32(3) ? k+offset : k
        @inbounds source[ii,jj,kk,component]
    end)
    if !allow_adaptive || !_structured_face_p2a_adaptive_enabled()
        return structured_point_to_average6(values),STRUCTURED_P2A_FIXED
    end
    value,mode,_ = structured_point_to_average_ao(
        values,
        force_adaptive ? zero(T) : _structured_face_p2a_threshold(T),
        _structured_face_p2a_high_weight(T),
    )
    return value,mode
end

@inline function _structured_record_face_p2a_recovery!(
    meta, used_ao::Bool, used_midpoint::Bool,
)
    meta === nothing && return nothing
    @static if @isdefined(CT_POS_FACE_P2A_TO_AO_COUNT)
        used_ao && ct_record_fallback!(meta,CT_POS_FACE_P2A_TO_AO_COUNT)
        used_midpoint && ct_record_fallback!(
            meta,CT_POS_FACE_P2A_TO_MIDPOINT_COUNT,
        )
    end
    return nothing
end

function structured_face_p2a_first_kernel!(
    scratch, point_flux, nxp, nyp, nzp, ::Val{DIRECTION}, meta=nothing,
    sensor=nothing, shock_threshold=zero(eltype(point_flux)),
) where {DIRECTION}
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k=(blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    halo=Int32(2)
    if DIRECTION == 1
        (i>nxp+Int32(1) || j>nyp || k>nzp+Int32(2)*halo) && return
        fi,fj,fk=i,j+halo,k
        axis,extent=Int32(2),Int32(nyp)
        sensor_i,sensor_j,sensor_k=i,j,k-halo
    elseif DIRECTION == 2
        (i>nxp || j>nyp+Int32(1) || k>nzp+Int32(2)*halo) && return
        fi,fj,fk=i+halo,j,k
        axis,extent=Int32(1),Int32(nxp)
        sensor_i,sensor_j,sensor_k=i,j,k-halo
    else
        (i>nxp || j>nyp+Int32(2)*halo || k>nzp+Int32(1)) && return
        fi,fj,fk=i+halo,j,k
        axis,extent=Int32(1),Int32(nxp)
        sensor_i,sensor_j,sensor_k=i,j-halo,k
    end
    support_sensor = sensor === nothing ? zero(eltype(point_flux)) :
        _structured_face_support_max_sensor(
            sensor,sensor_i,sensor_j,sensor_k,Val(DIRECTION),
        )
    finite_support_sensor=isfinite(support_sensor)
    force_adaptive=sensor !== nothing && finite_support_sensor &&
        support_sensor >= shock_threshold
    allow_adaptive=sensor === nothing || (finite_support_sensor &&
        support_sensor >= shock_threshold*
        _structured_face_p2a_gate_fraction(eltype(point_flux)))
    used_ao=false
    used_midpoint=!finite_support_sensor
    @inbounds for component in 1:Ncons
        if !finite_support_sensor
            value=point_flux[fi,fj,fk,component]
            mode=STRUCTURED_P2A_MIDPOINT
        else
            value,mode=_structured_p2a_axis_adaptive_value(
                point_flux,fi,fj,fk,component,axis,extent,
                allow_adaptive,force_adaptive,
            )
        end
        used_ao |= mode == STRUCTURED_P2A_AO
        used_midpoint |= mode == STRUCTURED_P2A_MIDPOINT
        scratch[fi,fj,fk,component]=value
    end
    _structured_record_face_p2a_recovery!(meta,used_ao,used_midpoint)
    return
end

@inline function _structured_face_support_max_sensor(
    sensor, i, j, k, ::Val{DIRECTION},
) where {DIRECTION}
    ng=Int32(NG)
    maximum_sensor=zero(eltype(sensor))
    if DIRECTION == 1
        li,lj,lk=i+ng-Int32(1),j+ng,k+ng
        ri,rj,rk=li+Int32(1),lj,lk
        for first_offset in Int32(-2):Int32(2)
            for second_offset in Int32(-2):Int32(2)
                @inbounds begin
                    left=sensor[li,lj+first_offset,lk+second_offset]
                    right=sensor[ri,rj+first_offset,rk+second_offset]
                end
                (!isfinite(left) || !isfinite(right)) &&
                    return oftype(maximum_sensor,Inf)
                maximum_sensor=max(maximum_sensor,max(left,right))
            end
        end
    elseif DIRECTION == 2
        li,lj,lk=i+ng,j+ng-Int32(1),k+ng
        ri,rj,rk=li,lj+Int32(1),lk
        for first_offset in Int32(-2):Int32(2)
            for second_offset in Int32(-2):Int32(2)
                @inbounds begin
                    left=sensor[li+first_offset,lj,lk+second_offset]
                    right=sensor[ri+first_offset,rj,rk+second_offset]
                end
                (!isfinite(left) || !isfinite(right)) &&
                    return oftype(maximum_sensor,Inf)
                maximum_sensor=max(maximum_sensor,max(left,right))
            end
        end
    else
        li,lj,lk=i+ng,j+ng,k+ng-Int32(1)
        ri,rj,rk=li,lj,lk+Int32(1)
        for first_offset in Int32(-2):Int32(2)
            for second_offset in Int32(-2):Int32(2)
                @inbounds begin
                    left=sensor[li+first_offset,lj+second_offset,lk]
                    right=sensor[ri+first_offset,rj+second_offset,rk]
                end
                (!isfinite(left) || !isfinite(right)) &&
                    return oftype(maximum_sensor,Inf)
                maximum_sensor=max(maximum_sensor,max(left,right))
            end
        end
    end
    return maximum_sensor
end

@inline function _structured_face_support_is_shocked(
    sensor, i, j, k, direction, threshold,
)
    return _structured_face_support_max_sensor(
        sensor,i,j,k,direction,
    ) >= threshold
end

function structured_face_p2a_second_kernel!(
    face_average, first_average, point_flux, sensor,
    nxp, nyp, nzp, shock_threshold,
    ::Val{DIRECTION}, meta=nothing,
) where {DIRECTION}
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k=(blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    if i>nxp+(DIRECTION==1 ? Int32(1) : Int32(0)) ||
       j>nyp+(DIRECTION==2 ? Int32(1) : Int32(0)) ||
       k>nzp+(DIRECTION==3 ? Int32(1) : Int32(0))
        return
    end
    halo=Int32(2)
    fi=i+(DIRECTION==1 ? Int32(0) : halo)
    fj=j+(DIRECTION==2 ? Int32(0) : halo)
    fk=k+(DIRECTION==3 ? Int32(0) : halo)
    second_axis=DIRECTION==3 ? Int32(2) : Int32(3)
    second_extent=DIRECTION==3 ? Int32(nyp) : Int32(nzp)
    support_sensor=_structured_face_support_max_sensor(
        sensor,i,j,k,Val(DIRECTION),
    )
    finite_support_sensor=isfinite(support_sensor)
    force_adaptive=finite_support_sensor && support_sensor >= shock_threshold
    allow_adaptive=finite_support_sensor && support_sensor >=
        shock_threshold*_structured_face_p2a_gate_fraction(eltype(point_flux))
    used_ao=false
    used_midpoint=!finite_support_sensor
    @inbounds for component in 1:Ncons
        if !finite_support_sensor
            value=point_flux[fi,fj,fk,component]
            mode=STRUCTURED_P2A_MIDPOINT
        else
            value,mode=_structured_p2a_axis_adaptive_value(
                first_average,fi,fj,fk,component,second_axis,second_extent,
                allow_adaptive,force_adaptive,
            )
            if mode == STRUCTURED_P2A_MIDPOINT
                value=point_flux[fi,fj,fk,component]
            end
        end
        used_ao |= mode == STRUCTURED_P2A_AO
        used_midpoint |= mode == STRUCTURED_P2A_MIDPOINT
        face_average[fi,fj,fk,component]=value
    end
    _structured_record_face_p2a_recovery!(meta,used_ao,used_midpoint)
    return
end

end
