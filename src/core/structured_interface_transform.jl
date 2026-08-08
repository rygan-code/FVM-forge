# Structured block-interface orientation and index transforms.
#
# A face has an oriented local frame (n, u, v): n is the face normal and u/v
# are the two independent in-plane axes.  The old connectivity format only
# represented a v reversal.  The types below keep the full signed 3-D axis
# permutation so all downstream exchange paths can share one convention.

struct StructuredFaceFrame
    normal_axis::Int8
    side_sign::Int8
    u_axis::Int8
    v_axis::Int8
end

struct StructuredFaceTransform
    source_face::Int8
    destination_face::Int8
    # For each destination global axis, the signed source axis supplying it.
    # Values are in {-3,-2,-1,1,2,3}; a negative value reverses that axis.
    source_for_destination::NTuple{3,Int8}
end

@inline function _structured_face_frame_unchecked(fid::Integer)
    normal_axis = Int8((fid + 1) ÷ 2)
    side_sign = Int8(isodd(fid) ? -1 : 1)
    if normal_axis == 1
        return StructuredFaceFrame(normal_axis, side_sign, Int8(2), Int8(3))
    elseif normal_axis == 2
        return StructuredFaceFrame(normal_axis, side_sign, Int8(1), Int8(3))
    else
        return StructuredFaceFrame(normal_axis, side_sign, Int8(1), Int8(2))
    end
end

function structured_face_frame(fid::Integer)
    1 <= fid <= 6 || throw(ArgumentError("face id must be in 1:6, got $fid"))
    return _structured_face_frame_unchecked(fid)
end

@inline structured_face_normal_axis(fid::Integer) = _structured_face_frame_unchecked(fid).normal_axis
@inline structured_face_side_sign(fid::Integer) = _structured_face_frame_unchecked(fid).side_sign
@inline structured_face_tangent_axes(fid::Integer) = begin
    frame = _structured_face_frame_unchecked(fid)
    (frame.u_axis, frame.v_axis)
end

@inline function _structured_signed_axis(value::Integer)
    abs(value) in 1:3 || throw(ArgumentError("signed axis must be ±1, ±2, or ±3"))
    return Int8(value)
end

function _structured_validate_axis_map(axis_map::NTuple{3,<:Integer})
    signed = ntuple(index -> _structured_signed_axis(axis_map[index]), 3)
    sort!(Int[abs(value) for value in signed]) == [1, 2, 3] ||
        throw(ArgumentError("axis map must be a signed permutation of 1:3"))
    return signed
end

"""
    structured_face_transform(source_face, destination_face;
                               swap_tangents=false,
                               reverse_u=false,
                               reverse_v=false,
                               reverse_normal=false)

Build a signed global-axis transform from the local oriented frames of two
faces.  The local normal maps to the destination normal; the two tangents may
be exchanged and independently reversed.  `reverse_normal` is kept explicit
because coordinate-layer order and oriented CT flux signs are separate
consumers of the transform.
"""
function structured_face_transform(
    source_face::Integer,
    destination_face::Integer;
    swap_tangents::Bool=false,
    reverse_u::Bool=false,
    reverse_v::Bool=false,
    reverse_normal::Bool=false,
)
    source = structured_face_frame(source_face)
    destination = structured_face_frame(destination_face)
    source_for_destination = zeros(Int8, 3)

    normal_value = reverse_normal ? -source.normal_axis : source.normal_axis
    source_for_destination[destination.normal_axis] = normal_value

    source_u = swap_tangents ? source.v_axis : source.u_axis
    source_v = swap_tangents ? source.u_axis : source.v_axis
    source_for_destination[destination.u_axis] =
        (reverse_u ? -source_u : source_u)
    source_for_destination[destination.v_axis] =
        (reverse_v ? -source_v : source_v)

    return StructuredFaceTransform(
        Int8(source_face), Int8(destination_face),
        _structured_validate_axis_map(Tuple(source_for_destination)),
    )
end

"""Legacy connectivity mapping: old `reverse_tan` means local v reversal."""
@inline function structured_legacy_face_transform(source_face, destination_face, reverse_tan)
    return structured_face_transform(
        source_face, destination_face;
        reverse_v=Bool(reverse_tan),
    )
end

@inline function structured_face_transform_tangents(transform::StructuredFaceTransform)
    source = _structured_face_frame_unchecked(transform.source_face)
    destination = _structured_face_frame_unchecked(transform.destination_face)
    return (
        transform.source_for_destination[destination.u_axis],
        transform.source_for_destination[destination.v_axis],
    )
end

@inline function structured_face_transform_normal(transform::StructuredFaceTransform)
    destination = _structured_face_frame_unchecked(transform.destination_face)
    return transform.source_for_destination[destination.normal_axis]
end

function structured_validate_face_transform(transform::StructuredFaceTransform)
    source = structured_face_frame(transform.source_face)
    destination = structured_face_frame(transform.destination_face)
    axis_map = _structured_validate_axis_map(transform.source_for_destination)
    abs(axis_map[destination.normal_axis]) == source.normal_axis ||
        throw(ArgumentError("face transform does not map source normal to destination normal"))
    destination_tangents = sort(Int[
        abs(axis_map[destination.u_axis]),
        abs(axis_map[destination.v_axis]),
    ])
    source_tangents = sort(Int[source.u_axis, source.v_axis])
    destination_tangents == source_tangents ||
        throw(ArgumentError("face transform does not map the two tangent axes"))
    return true
end

function structured_inverse_face_transform(transform::StructuredFaceTransform)
    inverse_map = zeros(Int8, 3)
    for destination_axis in 1:3
        source_value = transform.source_for_destination[destination_axis]
        source_axis = abs(source_value)
        inverse_map[source_axis] =
            source_value < 0 ? Int8(-destination_axis) : Int8(destination_axis)
    end
    return StructuredFaceTransform(
        transform.destination_face,
        transform.source_face,
        _structured_validate_axis_map(Tuple(inverse_map)),
    )
end

@inline function structured_face_array_index(
    fid::Integer, normal_index::Integer, u_index::Integer, v_index::Integer,
)
    frame = _structured_face_frame_unchecked(fid)
    if frame.normal_axis == 1
        return (normal_index, u_index, v_index)
    elseif frame.normal_axis == 2
        return (u_index, normal_index, v_index)
    else
        return (u_index, v_index, normal_index)
    end
end

@inline function structured_face_normal_cells(fid::Integer, dimensions)
    return dimensions[_structured_face_frame_unchecked(fid).normal_axis]
end

"""Map a 1-based source index through a signed global-axis transform."""
function structured_transform_index(
    transform::StructuredFaceTransform,
    source_index::NTuple{3,<:Integer},
    source_dimensions::NTuple{3,<:Integer},
)
    structured_validate_face_transform(transform)
    destination_index = zeros(Int, 3)
    for destination_axis in 1:3
        source_value = transform.source_for_destination[destination_axis]
        source_axis = abs(source_value)
        value = Int(source_index[source_axis])
        if source_value < 0
            value = Int(source_dimensions[source_axis]) + 1 - value
        end
        destination_index[destination_axis] = value
    end
    return (destination_index[1], destination_index[2], destination_index[3])
end

"""Return the destination face's local tangent source axes, including signs."""
@inline function structured_face_tangent_source_axes(transform::StructuredFaceTransform)
    destination = _structured_face_frame_unchecked(transform.destination_face)
    return (
        transform.source_for_destination[destination.u_axis],
        transform.source_for_destination[destination.v_axis],
    )
end

"""Return the face-local `(u,v)` total cell counts for a block."""
@inline function structured_face_uv_totals(fid::Integer, dimensions::NTuple{3,<:Integer})
    frame = _structured_face_frame_unchecked(fid)
    return dimensions[frame.u_axis], dimensions[frame.v_axis]
end

"""Map a source sub-domain's global axis ranges into destination face axes."""
function structured_map_face_extent(
    transform::StructuredFaceTransform,
    source_ranges::Tuple{<:Tuple,<:Tuple,<:Tuple},
    source_dimensions::NTuple{3,<:Integer},
)
    structured_validate_face_transform(transform)
    destination = structured_face_frame(transform.destination_face)
    mapped = zeros(Int, 3, 2)
    for destination_axis in (Int(destination.u_axis), Int(destination.v_axis))
        source_value = transform.source_for_destination[destination_axis]
        source_axis = abs(source_value)
        source_lo, source_hi = source_ranges[source_axis]
        if source_value < 0
            mapped[destination_axis, 1] = source_dimensions[source_axis] + 1 - source_hi
            mapped[destination_axis, 2] = source_dimensions[source_axis] + 1 - source_lo
        else
            mapped[destination_axis, 1] = source_lo
            mapped[destination_axis, 2] = source_hi
        end
    end
    return (
        (mapped[destination.u_axis, 1], mapped[destination.u_axis, 2]),
        (mapped[destination.v_axis, 1], mapped[destination.v_axis, 2]),
    )
end

"""Compact the eight in-plane orientations into a GPU-friendly integer."""
@inline function structured_face_transform_code(transform::StructuredFaceTransform)
    destination = _structured_face_frame_unchecked(transform.destination_face)
    source = _structured_face_frame_unchecked(transform.source_face)
    map_u = transform.source_for_destination[destination.u_axis]
    map_v = transform.source_for_destination[destination.v_axis]
    swap = abs(map_u) == source.v_axis
    return Int8((swap ? 1 : 0) | (map_u < 0 ? 2 : 0) | (map_v < 0 ? 4 : 0))
end

"""Build HDF5-ready signed-axis rows from legacy connectivity flags."""
function structured_connectivity_axis_map(connectivity_rows, reverse_tan)
    size(connectivity_rows, 2) == 4 || throw(ArgumentError(
        "connectivity rows must have four columns",
    ))
    length(reverse_tan) == size(connectivity_rows, 1) || throw(ArgumentError(
        "reverse_tan length must match connectivity rows",
    ))
    axis_map = zeros(Int64, size(connectivity_rows, 1), 3)
    for row in axes(connectivity_rows, 1)
        transform = structured_legacy_face_transform(
            Int(connectivity_rows[row, 2]), Int(connectivity_rows[row, 4]),
            Bool(reverse_tan[row]),
        )
        axis_map[row, :] .= Int.(transform.source_for_destination)
    end
    return axis_map
end
