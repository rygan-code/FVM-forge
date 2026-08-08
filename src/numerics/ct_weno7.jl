if !isdefined(@__MODULE__, :StructuredFaceFrame)
    include(joinpath(@__DIR__, "..", "core", "structured_interface_transform.jl"))
end

@inline function ct_face_tangential_electric(
    magnetic_flux::SVector{3,T}, normal::SVector{3,T},
) where {T}
    return cross(magnetic_flux, normal)
end

# First non-finite WENO7 sample metadata schema:
# [claimed, site, direction, i, j, k, channel, schema_version].
const CT_WENO7_FAIL_META_LEN = 8
const CT_WENO7_FAIL_SCHEMA_VERSION = Int32(1)
const CT_WENO7_FAIL_CLAIMED = 1
const CT_WENO7_FAIL_SITE = 2
const CT_WENO7_FAIL_DIRECTION = 3
const CT_WENO7_FAIL_I = 4
const CT_WENO7_FAIL_J = 5
const CT_WENO7_FAIL_K = 6
const CT_WENO7_FAIL_CHANNEL = 7
const CT_WENO7_FAIL_VERSION = 8

@inline function _ct_weno7_try_claim!(fail_meta)
    @static if @isdefined(gpu_atomic_cas!)
        return gpu_atomic_cas!(
            fail_meta, CT_WENO7_FAIL_CLAIMED, Int32(0), Int32(1),
        ) == Int32(0)
    elseif @isdefined(USE_CUDA) && USE_CUDA
        return CUDA.atomic_cas!(
            pointer(fail_meta), Int32(0), Int32(1),
        ) == Int32(0)
    else
        return Core.Intrinsics.atomic_pointerreplace(
            pointer(fail_meta), Int32(0), Int32(1),
            :acquire_release, :acquire,
        ).success
    end
end

@inline function ct_weno7_record_nonfinite_if_needed!(
    fail_meta, fail_value, value,
    site::Int32, direction::Int32,
    i::Int32, j::Int32, k::Int32, channel::Int32,
)
    if !isfinite(value) && _ct_weno7_try_claim!(fail_meta)
        @inbounds begin
            fail_meta[CT_WENO7_FAIL_SITE] = site
            fail_meta[CT_WENO7_FAIL_DIRECTION] = direction
            fail_meta[CT_WENO7_FAIL_I] = i
            fail_meta[CT_WENO7_FAIL_J] = j
            fail_meta[CT_WENO7_FAIL_K] = k
            fail_meta[CT_WENO7_FAIL_CHANNEL] = channel
            fail_meta[CT_WENO7_FAIL_VERSION] = CT_WENO7_FAIL_SCHEMA_VERSION
            fail_value[1] = value
        end
    end
    return nothing
end

function ct_weno7_validate_finite_kernel!(
    fail_meta, fail_value, array, site::Int32, direction::Int32,
    n1::Int32, n2::Int32, n3::Int32, nchannels::Int32,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > n1 || j > n2 || k > n3
        return
    end
    @inbounds for channel in Int32(1):nchannels
        ct_weno7_record_nonfinite_if_needed!(
            fail_meta, fail_value, array[i, j, k, channel], site, direction,
            Int32(i), Int32(j), Int32(k), Int32(channel),
        )
    end
    return
end

function ct_weno7_reset_failure!(fail_meta, fail_value)
    fill!(fail_meta, Int32(0))
    fill!(fail_value, zero(eltype(fail_value)))
    return nothing
end

function ct_weno7_check_failure!(
    fail_meta, fail_value; rank::Int, block::Int, rk_stage::Int,
)::Nothing
    meta = Array(fail_meta)
    meta[CT_WENO7_FAIL_CLAIMED] == Int32(0) && return nothing
    length(meta) == CT_WENO7_FAIL_META_LEN || error(
        "Invalid WENO7 failure metadata length=$(length(meta)); " *
        "expected=$CT_WENO7_FAIL_META_LEN",
    )
    raw_value = Array(fail_value)[1]
    error(
        "Non-finite WENO7 CT sample: rank=$rank block=$block " *
        "rk_stage=$rk_stage site=$(meta[CT_WENO7_FAIL_SITE]) " *
        "direction=$(meta[CT_WENO7_FAIL_DIRECTION]) " *
        "i=$(meta[CT_WENO7_FAIL_I]) j=$(meta[CT_WENO7_FAIL_J]) " *
        "k=$(meta[CT_WENO7_FAIL_K]) " *
        "channel=$(meta[CT_WENO7_FAIL_CHANNEL]) " *
        "version=$(meta[CT_WENO7_FAIL_VERSION]) value=$raw_value",
    )
end

@inline function ct_weno7_upwind_pair(
    left::T, right::T, weight::T,
) where {T}
    bounded = min(one(T), max(zero(T), weight))
    return (one(T)-bounded)*left + bounded*right
end

@inline function ct_cmd6_edge_vector(pm2, pm1, p0, p1, p2, p3)
    T = eltype(p0)
    return T(75)/T(64)*(p1-p0) - T(25)/T(384)*(p2-pm1) +
           T(3)/T(640)*(p3-pm2)
end

@inline function ct_weno7_midpoint_integrand(
    b_left::SVector{7,T}, b_right::SVector{7,T},
    c_left::SVector{7,T}, c_right::SVector{7,T},
    weight_b::SVector{7,T}, weight_c::SVector{7,T},
    ss_b::T, ss_c::T,
) where {T}
    eb_l = weno7_point_face_left(b_left, ss_b)
    eb_r = weno7_point_face_right(b_right, ss_b)
    ec_l = weno7_point_face_left(c_left, ss_c)
    ec_r = weno7_point_face_right(c_right, ss_c)
    wb = (weno7_point_face_left(weight_b, ss_b) +
          weno7_point_face_right(weight_b, ss_b))/T(2)
    wc = (weno7_point_face_left(weight_c, ss_c) +
          weno7_point_face_right(weight_c, ss_c))/T(2)
    edge_b = ct_weno7_upwind_pair(eb_l, eb_r, wb)
    edge_c = ct_weno7_upwind_pair(ec_l, ec_r, wc)
    return (edge_b+edge_c)/T(2)
end

@inline function ct_weno7_manufactured_electric(
    point::SVector{3,T},
) where {T}
    k1 = SVector{3,T}(one(T), T(2), -one(T))
    k2 = SVector{3,T}(-T(2), one(T), one(T))
    k3 = SVector{3,T}(one(T), -one(T), T(2))
    return SVector{3,T}(
        sin(dot(k1, point)),
        cos(dot(k2, point)),
        sin(dot(k3, point)),
    )
end

@inline function ct_weno7_symmetric_sinc(delta::T) where {T}
    if abs(delta) <= sqrt(eps(T))
        return one(T) - delta*delta/T(24)
    end
    return T(2)*sin(delta/T(2))/delta
end

@inline function ct_weno7_exact_segment_integral(
    p0::SVector{3,T}, p1::SVector{3,T},
) where {T}
    k1 = SVector{3,T}(one(T), T(2), -one(T))
    k2 = SVector{3,T}(-T(2), one(T), one(T))
    k3 = SVector{3,T}(one(T), -one(T), T(2))
    d = p1-p0
    midpoint = (p0+p1)/T(2)
    average_e = SVector{3,T}(
        sin(dot(k1, midpoint))*ct_weno7_symmetric_sinc(dot(k1, d)),
        cos(dot(k2, midpoint))*ct_weno7_symmetric_sinc(dot(k2, d)),
        sin(dot(k3, midpoint))*ct_weno7_symmetric_sinc(dot(k3, d)),
    )
    return dot(d, average_e)
end

@inline function _ct_weno7_edge_point(
    x, y, z, direction::Int32,
    si::Int32, sj::Int32, sk::Int32, offset::Int32,
)
    ii = direction == Int32(1) ? si+offset : si
    jj = direction == Int32(2) ? sj+offset : sj
    kk = direction == Int32(3) ? sk+offset : sk
    T = eltype(x)
    @inbounds return SVector{3,T}(x[ii,jj,kk], y[ii,jj,kk], z[ii,jj,kk])
end

@inline function _ct_weno7_runtime_edge_vector(
    x, y, z, direction::Int32,
    si::Int32, sj::Int32, sk::Int32,
    nxp::Int32, nyp::Int32, nzp::Int32,
)
    tangent_extent = direction == Int32(1) ? nxp+Int32(2NG) :
                     (direction == Int32(2) ? nyp+Int32(2NG) :
                                              nzp+Int32(2NG))
    tangent_index = direction == Int32(1) ? si :
                    (direction == Int32(2) ? sj : sk)
    # The outer two physical-boundary halo segments cannot support centered
    # CMD6. They are outside the formal core, so extend its nearest value.
    centered_index = min(max(tangent_index, Int32(3)), tangent_extent-Int32(2))
    ci = direction == Int32(1) ? centered_index : si
    cj = direction == Int32(2) ? centered_index : sj
    ck = direction == Int32(3) ? centered_index : sk
    return ct_cmd6_edge_vector(
        _ct_weno7_edge_point(x, y, z, direction, ci, cj, ck, Int32(-2)),
        _ct_weno7_edge_point(x, y, z, direction, ci, cj, ck, Int32(-1)),
        _ct_weno7_edge_point(x, y, z, direction, ci, cj, ck, Int32(0)),
        _ct_weno7_edge_point(x, y, z, direction, ci, cj, ck, Int32(1)),
        _ct_weno7_edge_point(x, y, z, direction, ci, cj, ck, Int32(2)),
        _ct_weno7_edge_point(x, y, z, direction, ci, cj, ck, Int32(3)),
    )
end

@inline function _ct_weno7_cache_component(
    cache, ::Val{1}, si::Int32, sj::Int32, sk::Int32, component::Int,
)
    ci = si-Int32(NG)
    @inbounds return cache[ci,sj,sk,component]
end

@inline function _ct_weno7_cache_component(
    cache, ::Val{2}, si::Int32, sj::Int32, sk::Int32, component::Int,
)
    cj = sj-Int32(NG)
    @inbounds return cache[si,cj,sk,component]
end

@inline function _ct_weno7_cache_component(
    cache, ::Val{3}, si::Int32, sj::Int32, sk::Int32, component::Int,
)
    ck = sk-Int32(NG)
    @inbounds return cache[si,sj,ck,component]
end

@inline _ct_weno7_cache_weight(cache, ::Val{1}, si::Int32, sj::Int32, sk::Int32) =
    (@inbounds cache[si-Int32(NG),sj,sk,4])
@inline _ct_weno7_cache_weight(cache, ::Val{2}, si::Int32, sj::Int32, sk::Int32) =
    (@inbounds cache[si,sj-Int32(NG),sk,4])
@inline _ct_weno7_cache_weight(cache, ::Val{3}, si::Int32, sj::Int32, sk::Int32) =
    (@inbounds cache[si,sj,sk-Int32(NG),4])

@inline function _ct_weno7_shift_index(
    si::Int32, sj::Int32, sk::Int32, direction::Int, offset::Int32,
)
    return (
        direction == 1 ? si+offset : si,
        direction == 2 ? sj+offset : sj,
        direction == 3 ? sk+offset : sk,
    )
end

@inline function _ct_weno7_midpoint_direction(
    cache_b, cache_c, ::Val{B}, ::Val{C},
    si::Int32, sj::Int32, sk::Int32, edge_vector,
) where {B,C}
    T = eltype(edge_vector)
    weight_b = SVector{7,T}(ntuple(Val(7)) do sample
        bli, blj, blk = _ct_weno7_shift_index(si, sj, sk, C, Int32(sample-5))
        bri, brj, brk = _ct_weno7_shift_index(si, sj, sk, C, Int32(sample-4))
        (_ct_weno7_cache_weight(cache_b, Val(B), bli, blj, blk) +
         _ct_weno7_cache_weight(cache_b, Val(B), bri, brj, brk))/T(2)
    end)
    weight_c = SVector{7,T}(ntuple(Val(7)) do sample
        cli, clj, clk = _ct_weno7_shift_index(si, sj, sk, B, Int32(sample-5))
        cri, crj, crk = _ct_weno7_shift_index(si, sj, sk, B, Int32(sample-4))
        (_ct_weno7_cache_weight(cache_c, Val(C), cli, clj, clk) +
         _ct_weno7_cache_weight(cache_c, Val(C), cri, crj, crk))/T(2)
    end)
    edge_electric = SVector{3,T}(ntuple(Val(3)) do component
        b_left = SVector{7,T}(ntuple(Val(7)) do sample
            bi, bj, bk = _ct_weno7_shift_index(
                si, sj, sk, C, Int32(sample-5),
            )
            _ct_weno7_cache_component(
                cache_b, Val(B), bi, bj, bk, component,
            )
        end)
        b_right = SVector{7,T}(ntuple(Val(7)) do sample
            bi, bj, bk = _ct_weno7_shift_index(
                si, sj, sk, C, Int32(sample-4),
            )
            _ct_weno7_cache_component(
                cache_b, Val(B), bi, bj, bk, component,
            )
        end)
        c_left = SVector{7,T}(ntuple(Val(7)) do sample
            ci, cj, ck = _ct_weno7_shift_index(
                si, sj, sk, B, Int32(sample-5),
            )
            _ct_weno7_cache_component(
                cache_c, Val(C), ci, cj, ck, component,
            )
        end)
        c_right = SVector{7,T}(ntuple(Val(7)) do sample
            ci, cj, ck = _ct_weno7_shift_index(
                si, sj, sk, B, Int32(sample-4),
            )
            _ct_weno7_cache_component(
                cache_c, Val(C), ci, cj, ck, component,
            )
        end)
        ct_weno7_midpoint_integrand(
            b_left, b_right, c_left, c_right,
            weight_b, weight_c, one(T), one(T),
        )
    end)
    return dot(edge_electric, edge_vector)
end

function ct_weno7_edge_midpoint_kernel!(
    scratch, direction, cache_i, cache_j, cache_k, x, y, z,
    fail_meta, fail_value, rk_stage, nxp, nyp, nzp,
)
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    ng = Int32(NG)
    if direction == Int32(1)
        if i > nxp+Int32(2)*ng || j > nyp+Int32(1) || k > nzp+Int32(1); return; end
        si, sj, sk = Int32(i), Int32(j)+ng, Int32(k)+ng
        edge_vector = _ct_weno7_runtime_edge_vector(x, y, z, direction, si, sj, sk, nxp, nyp, nzp)
        value = _ct_weno7_midpoint_direction(cache_j, cache_k, Val(2), Val(3), si, sj, sk, edge_vector)
    elseif direction == Int32(2)
        if i > nxp+Int32(1) || j > nyp+Int32(2)*ng || k > nzp+Int32(1); return; end
        si, sj, sk = Int32(i)+ng, Int32(j), Int32(k)+ng
        edge_vector = _ct_weno7_runtime_edge_vector(x, y, z, direction, si, sj, sk, nxp, nyp, nzp)
        value = _ct_weno7_midpoint_direction(cache_i, cache_k, Val(1), Val(3), si, sj, sk, edge_vector)
    else
        if i > nxp+Int32(1) || j > nyp+Int32(1) || k > nzp+Int32(2)*ng; return; end
        si, sj, sk = Int32(i)+ng, Int32(j)+ng, Int32(k)
        edge_vector = _ct_weno7_runtime_edge_vector(x, y, z, direction, si, sj, sk, nxp, nyp, nzp)
        value = _ct_weno7_midpoint_direction(cache_i, cache_j, Val(1), Val(2), si, sj, sk, edge_vector)
    end
    @inbounds scratch[si,sj,sk] = value
    ct_weno7_record_nonfinite_if_needed!(
        fail_meta, fail_value, value, Int32(2), Int32(direction),
        Int32(si), Int32(sj), Int32(sk), Int32(1),
    )
    return
end

function ct_weno7_periodic_point_halo_kernel!(
    scratch, direction, periodic_x, periodic_y, periodic_z, nxp, nyp, nzp,
)
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    ng = Int32(NG)
    if direction == Int32(1) && periodic_x
        if i > ng || j > nyp+Int32(1) || k > nzp+Int32(1); return; end
        sj, sk = Int32(j)+ng, Int32(k)+ng
        @inbounds begin
            scratch[ng+Int32(1)-i,sj,sk] = scratch[ng+nxp+Int32(1)-i,sj,sk]
            scratch[ng+nxp+i,sj,sk] = scratch[ng+i,sj,sk]
        end
    elseif direction == Int32(2) && periodic_y
        if i > nxp+Int32(1) || j > ng || k > nzp+Int32(1); return; end
        si, sk = Int32(i)+ng, Int32(k)+ng
        @inbounds begin
            scratch[si,ng+Int32(1)-j,sk] = scratch[si,ng+nyp+Int32(1)-j,sk]
            scratch[si,ng+nyp+j,sk] = scratch[si,ng+j,sk]
        end
    elseif direction == Int32(3) && periodic_z
        if i > nxp+Int32(1) || j > nyp+Int32(1) || k > ng; return; end
        si, sj = Int32(i)+ng, Int32(j)+ng
        @inbounds begin
            scratch[si,sj,ng+Int32(1)-k] = scratch[si,sj,ng+nzp+Int32(1)-k]
            scratch[si,sj,ng+nzp+k] = scratch[si,sj,ng+k]
        end
    end
    return
end

function ct_weno7_edge_line_kernel!(
    Eedge, scratch, direction, fail_meta, fail_value,
    rk_stage, nxp, nyp, nzp,
)
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    ng = Int32(NG)
    if direction == Int32(1)
        if i > nxp || j > nyp+Int32(1) || k > nzp+Int32(1); return; end
    elseif direction == Int32(2)
        if i > nxp+Int32(1) || j > nyp || k > nzp+Int32(1); return; end
    else
        if i > nxp+Int32(1) || j > nyp+Int32(1) || k > nzp; return; end
    end
    si, sj, sk = Int32(i)+ng, Int32(j)+ng, Int32(k)+ng
    T = eltype(scratch)
    values = SVector{7,T}(ntuple(Val(7)) do sample
        offset = Int32(sample-4)
        ii = direction == Int32(1) ? si+offset : si
        jj = direction == Int32(2) ? sj+offset : sj
        kk = direction == Int32(3) ? sk+offset : sk
        @inbounds scratch[ii,jj,kk]
    end)
    value = weno7_gauss4_integral(values)
    @inbounds Eedge[i,j,k] = value
    if !isfinite(value)
        gauss_values = SVector{4,T}(
            weno7_gauss_value(values, Val(1)),
            weno7_gauss_value(values, Val(2)),
            weno7_gauss_value(values, Val(3)),
            weno7_gauss_value(values, Val(4)),
        )
        for quadrature_node in Int32(1):Int32(4)
            ct_weno7_record_nonfinite_if_needed!(
                fail_meta, fail_value, gauss_values[quadrature_node],
                Int32(3), Int32(direction),
                Int32(i), Int32(j), Int32(k), quadrature_node,
            )
        end
        ct_weno7_record_nonfinite_if_needed!(
            fail_meta, fail_value, value, Int32(4), Int32(direction),
            Int32(i), Int32(j), Int32(k), Int32(1),
        )
    end
    return
end

function _ct_weno7_interface_fields(interface)
    if interface isa Pair
        key = first(interface)
        connection = last(interface)
        if !(key isa Tuple && length(key) == 2) ||
           !hasproperty(connection, :src_b) ||
           !hasproperty(connection, :src_f) ||
           !hasproperty(connection, :reverse_tan)
            error("Malformed WENO7 CT interface entry: $interface")
        end
        return (
            Int(key[1]), Int(key[2]),
            Int(getproperty(connection, :src_b)),
            Int(getproperty(connection, :src_f)),
            Bool(getproperty(connection, :reverse_tan)),
        )
    elseif interface isa Tuple && (length(interface) == 5 || length(interface) == 6)
        return (
            Int(interface[1]), Int(interface[2]), Int(interface[3]),
            Int(interface[4]), Bool(interface[5]),
        )
    end
    error("Malformed WENO7 CT interface entry: $interface")
end

@inline function _ct_weno7_interface_transform(interface)
    if interface isa Pair
        connection = last(interface)
        return hasproperty(connection, :transform) ?
            getproperty(connection, :transform) : nothing
    elseif interface isa Tuple && length(interface) == 6
        return interface[6]
    end
    return nothing
end

function _ct_weno7_validate_interface(interface)
    block, local_face, neighbor_block, neighbor_face, reverse_tan =
        _ct_weno7_interface_fields(interface)
    # Solver Connectivity records carry the signed face transform and are
    # valid for every normal direction and all eight in-plane orientations.
    # Keep the five-field tuple contract strict for legacy/static callers so
    # malformed metadata is still diagnosed at startup.
    transform = _ct_weno7_interface_transform(interface)
    general_transform = transform !== nothing
    if general_transform
        1 <= local_face <= 6 || error("Unsupported WENO7 CT local face=$local_face")
        1 <= neighbor_face <= 6 || error("Unsupported WENO7 CT neighbor face=$neighbor_face")
        structured_validate_face_transform(transform)
        return nothing
    end
    same_type = (local_face == 4 && neighbor_face == 3) ||
                (local_face == 3 && neighbor_face == 4)
    if !same_type || reverse_tan
        error(
            "Unsupported WENO7 CT interface: block=$block " *
            "local_face=$local_face neighbor_block=$neighbor_block " *
            "neighbor_face=$neighbor_face reverse_tan=$reverse_tan; " *
            "only same-type face 4 <-> face 3 interfaces with " *
            "reverse_tan=false are supported",
        )
    end
    return nothing
end

function _ct_weno7_validate_physical_bc(bc, context::AbstractString)
    bc_id = Int(bc)
    if bc_id != Int(BC_PERIODIC) && bc_id != Int(BC_ZERO_GRADIENT)
        error(
            "Unsupported WENO7 CT physical boundary: $context" *
            "bc_id=$bc_id; only BC_PERIODIC=$(Int(BC_PERIODIC)) and " *
            "BC_ZERO_GRADIENT=$(Int(BC_ZERO_GRADIENT)) are supported",
        )
    end
    return nothing
end

"""
    ct_validate_weno7_configuration(ng, interfaces, physical_bcs)::Nothing

Validate the startup-static topology supported by the first WENO7 CT delivery.
`interfaces` may be the solver connectivity dictionary or five-field tuples;
`physical_bcs` may be the solver face-BC dictionary or a collection of BC ids.
"""
function ct_validate_weno7_configuration(
    ng::Integer, interfaces, physical_bcs,
)::Nothing
    ng == 4 || error("WENO7 CT requires NG=4, got NG=$ng")

    for interface in interfaces
        _ct_weno7_validate_interface(interface)
    end

    if physical_bcs isa AbstractDict
        for (key, bc) in physical_bcs
            if !(key isa Tuple && length(key) == 2)
                error("Malformed WENO7 CT physical boundary entry: $key => $bc")
            end
            block, local_face = Int(key[1]), Int(key[2])
            if Int(bc) == 0
                if !haskey(interfaces, key)
                    error(
                        "Missing WENO7 CT connectivity: block=$block " *
                        "local_face=$local_face has interblock bc_id=0",
                    )
                end
                continue
            end
            _ct_weno7_validate_physical_bc(
                bc, "block=$block face=$local_face ",
            )
        end
    else
        for bc in physical_bcs
            _ct_weno7_validate_physical_bc(bc, "")
        end
    end
    return nothing
end
