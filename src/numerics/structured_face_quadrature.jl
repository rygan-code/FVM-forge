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

@inline function structured_ct_face_geometry_bn(
    face_field, background_field,
    area, normal_x, normal_y, normal_z,
    i, j, k, direction,
)
    if structured_face_quadrature == STRUCTURED_FACE_POINT6
        data = structured_face_point_data6(
            face_field,background_field,area,normal_x,normal_y,normal_z,
            i,j,k,direction,
        )
        T = eltype(area)
        area_vector = SVector{3,T}(data[2],data[3],data[4])
        point_area = sqrt(sum(abs2,area_vector))
        inverse_area = inv(point_area)
        return point_area,area_vector[1]*inverse_area,
               area_vector[2]*inverse_area,area_vector[3]*inverse_area,
               data[1]*inverse_area
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

function structured_face_p2a_first_kernel!(
    scratch, point_flux, nxp, nyp, nzp, ::Val{DIRECTION},
) where {DIRECTION}
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k=(blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    halo=Int32(2)
    if DIRECTION == 1
        (i>nxp+Int32(1) || j>nyp || k>nzp+Int32(2)*halo) && return
        fi,fj,fk=i,j+halo,k
        axis,extent=Int32(2),Int32(nyp)
    elseif DIRECTION == 2
        (i>nxp || j>nyp+Int32(1) || k>nzp+Int32(2)*halo) && return
        fi,fj,fk=i+halo,j,k
        axis,extent=Int32(1),Int32(nxp)
    else
        (i>nxp || j>nyp+Int32(2)*halo || k>nzp+Int32(1)) && return
        fi,fj,fk=i+halo,j,k
        axis,extent=Int32(1),Int32(nxp)
    end
    @inbounds for component in 1:Ncons
        scratch[fi,fj,fk,component]=_structured_p2a_axis_value(
            point_flux,fi,fj,fk,component,axis,extent,
        )
    end
    return
end

@inline function _structured_face_support_is_shocked(
    sensor, i, j, k, ::Val{DIRECTION}, threshold,
) where {DIRECTION}
    ng=Int32(NG)
    if DIRECTION == 1
        li,lj,lk=i+ng-Int32(1),j+ng,k+ng
        ri,rj,rk=li+Int32(1),lj,lk
        for first_offset in Int32(-2):Int32(2)
            for second_offset in Int32(-2):Int32(2)
                @inbounds begin
                    left=sensor[li,lj+first_offset,lk+second_offset]
                    right=sensor[ri,rj+first_offset,rk+second_offset]
                end
                (!isfinite(left) || !isfinite(right) ||
                 max(left,right) >= threshold) && return true
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
                (!isfinite(left) || !isfinite(right) ||
                 max(left,right) >= threshold) && return true
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
                (!isfinite(left) || !isfinite(right) ||
                 max(left,right) >= threshold) && return true
            end
        end
    end
    return false
end

function structured_face_p2a_second_kernel!(
    face_average, first_average, point_flux, sensor,
    nxp, nyp, nzp, shock_threshold,
    ::Val{DIRECTION},
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
    use_midpoint=_structured_face_support_is_shocked(
        sensor,i,j,k,Val(DIRECTION),shock_threshold,
    )
    @inbounds for component in 1:Ncons
        face_average[fi,fj,fk,component]=use_midpoint ?
            point_flux[fi,fj,fk,component] :
            _structured_p2a_axis_value(
                first_average,fi,fj,fk,component,second_axis,second_extent,
            )
    end
    return
end

end
