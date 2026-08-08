# Face-local indexing shared by structured inter-block halo exchange and tests.

if !isdefined(@__MODULE__, :StructuredFaceFrame)
    include(joinpath(@__DIR__, "..", "core", "structured_interface_transform.jl"))
end

@inline function _structured_face_axes(fid::Integer)
    frame = structured_face_frame(fid)
    return (Int(frame.normal_axis), Int(frame.u_axis), Int(frame.v_axis))
end

@inline function _structured_face_array_index(
    fid::Integer, normal_index::Integer, u_index::Integer, v_index::Integer,
)
    return structured_face_array_index(fid, normal_index, u_index, v_index)
end

@inline function _structured_face_normal_cells(fid, nx, ny, nz)
    return structured_face_normal_cells(fid, (nx, ny, nz))
end

@inline _structured_buffer_bytes(element_count, ::Type{T}) where {T} =
    element_count * sizeof(T)

# Non-uniform partition helper: returns the global cell range owned by rank r.
@inline function _nonuniform_extent(r, nprocs, nglobal)
    base = nglobal ÷ nprocs
    remainder = nglobal % nprocs
    lo = min(r, remainder) * (base + 1) + max(0, r - remainder) * base + 1
    hi = lo + (r < remainder ? base : base - 1)
    return lo, hi
end

function _get_face_uv_extents(fid, rx, ry, rz, px, py, pz, nx_g, ny_g, nz_g)
    extents = (
        _nonuniform_extent(rx, px, nx_g),
        _nonuniform_extent(ry, py, ny_g),
        _nonuniform_extent(rz, pz, nz_g),
    )
    dimensions = (nx_g, ny_g, nz_g)
    _, u_axis, v_axis = _structured_face_axes(fid)
    u_lo, u_hi = extents[u_axis]
    v_lo, v_hi = extents[v_axis]
    return u_lo, u_hi, v_lo, v_hi, dimensions[v_axis]
end

@inline function _validate_face_transform(src_fid, dst_fid, flip_normal)
    src_normal, _, _ = _structured_face_axes(src_fid)
    dst_normal, _, _ = _structured_face_axes(dst_fid)
    expected_flip = src_normal != dst_normal
    flip_normal == expected_flip || throw(ArgumentError(
        "inconsistent face transform $src_fid->$dst_fid: " *
        "flip_normal=$flip_normal, expected $expected_flip",
    ))
    return expected_flip
end

@inline function _ghost_face_pack_source_index(
    fid, nx, ny, nz, ng, u, normal_layer, v,
)
    normal_cells = _structured_face_normal_cells(fid, nx, ny, nz)
    # `normal_layer=1` is the cell adjacent to the face.  Low faces advance
    # inward from NG+1; high faces advance inward from the last real cell.
    normal_index = isodd(fid) ?
        ng + normal_layer : normal_cells + ng + one(normal_cells) - normal_layer
    return _structured_face_array_index(fid, normal_index, u, v)
end

@inline function _ghost_face_unpack_target_index(
    fid, nx, ny, nz, ng, u, normal_layer, v,
)
    normal_cells = _structured_face_normal_cells(fid, nx, ny, nz)
    normal_index = isodd(fid) ? normal_layer : normal_cells + ng + normal_layer
    return _structured_face_array_index(fid, normal_index, u, v)
end

@inline function _ghost_face_buffer_index(
    src_fid, dst_fid, reverse_tan, ng, u, normal_layer, v, v_len,
)
    return _ghost_face_buffer_index_transform(
        src_fid, dst_fid, reverse_tan ? 4 : 0,
        ng, u, normal_layer, v, u, v_len,
    )
end

@inline function _ghost_face_buffer_index_transform(
    src_fid, dst_fid, transform_code::Integer, ng, u, normal_layer, v,
    u_len, v_len,
)
    # The packet is ordered from the source face inward.  Only the
    # destination side determines whether its ghost layers are indexed
    # outside-in (low face) or inside-out (high face).
    source_layer = isodd(dst_fid) ?
        ng + 1 - normal_layer : normal_layer
    code = transform_code isa Bool ? (transform_code ? 4 : 0) : Int(transform_code)
    swap = (code & 1) != 0
    reverse_u = (code & 2) != 0
    reverse_v = (code & 4) != 0
    source_u = swap ? (reverse_v ? v_len + 1 - v : v) :
        (reverse_u ? u_len + 1 - u : u)
    source_v = swap ? (reverse_u ? u_len + 1 - u : u) :
        (reverse_v ? v_len + 1 - v : v)
    return (source_u, source_layer, source_v)
end

# GPU kernels need a fully concrete integer tuple for ROCm's array indexing
# lowering.  Keep this variant separate from the permissive host/test helper
# above, which intentionally accepts ordinary `Int` values.
@inline function _ghost_face_buffer_index_transform_i32(
    src_fid::Int32, dst_fid::Int32, transform_code::Int32,
    ng::Int32, u::Int32, normal_layer::Int32, v::Int32,
    u_len::Int32, v_len::Int32,
)
    source_layer::Int32 = isodd(dst_fid) ?
        ng + Int32(1) - normal_layer : normal_layer
    swap = (transform_code & Int32(1)) != Int32(0)
    reverse_u = (transform_code & Int32(2)) != Int32(0)
    reverse_v = (transform_code & Int32(4)) != Int32(0)
    source_u::Int32 = swap ?
        (reverse_v ? v_len + Int32(1) - v : v) :
        (reverse_u ? u_len + Int32(1) - u : u)
    source_v::Int32 = swap ?
        (reverse_u ? u_len + Int32(1) - u : u) :
        (reverse_v ? v_len + Int32(1) - v : v)
    return (source_u, source_layer, source_v)
end
