# CT-only packed storage for stencils that extend beyond the block's padded
# arrays. The six slabs are disjoint: x slabs include their edges/corners,
# y slabs exclude x-outside points, and z slabs exclude x/y-outside points.

struct CTPackedShellLayout
    base_dims::NTuple{3,Int32}
    low_reach::NTuple{3,Int32}
    high_reach::NTuple{3,Int32}
    slab_starts::NTuple{6,Int32}
    shell_count::Int32
end

function CTPackedShellLayout(
    base_dims::NTuple{3,<:Integer},
    low_reach::NTuple{3,<:Integer},
    high_reach::NTuple{3,<:Integer},
)
    all(>(0), base_dims) || throw(ArgumentError(
        "CT shell base dimensions must be positive, got $base_dims",
    ))
    all(>=(0), low_reach) || throw(ArgumentError(
        "CT shell low reach must be nonnegative, got $low_reach",
    ))
    all(>=(0), high_reach) || throw(ArgumentError(
        "CT shell high reach must be nonnegative, got $high_reach",
    ))

    nx, ny, nz = Int64.(base_dims)
    lx, ly, lz = Int64.(low_reach)
    hx, hy, hz = Int64.(high_reach)
    ex, ey, ez = nx + lx + hx, ny + ly + hy, nz + lz + hz
    counts = (
        lx * ey * ez,
        hx * ey * ez,
        nx * ly * ez,
        nx * hy * ez,
        nx * ny * lz,
        nx * ny * hz,
    )
    starts = (
        Int64(1),
        counts[1] + 1,
        counts[1] + counts[2] + 1,
        counts[1] + counts[2] + counts[3] + 1,
        counts[1] + counts[2] + counts[3] + counts[4] + 1,
        counts[1] + counts[2] + counts[3] + counts[4] + counts[5] + 1,
    )
    shell_count = sum(counts)
    expected = ex * ey * ez - nx * ny * nz
    shell_count == expected || error("internal CT shell count mismatch")
    shell_count <= typemax(Int32) || throw(ArgumentError(
        "CT shell has $shell_count entries, exceeding the Int32 GPU index limit",
    ))
    all(value -> value <= typemax(Int32), (ex, ey, ez)) || throw(
        ArgumentError("CT shell extended dimensions exceed the Int32 GPU index limit"),
    )

    return CTPackedShellLayout(
        Int32.(base_dims), Int32.(low_reach), Int32.(high_reach),
        Int32.(starts), Int32(shell_count),
    )
end

@inline ct_shell_count(layout::CTPackedShellLayout) = layout.shell_count

@inline function ct_shell_extended_dims(layout::CTPackedShellLayout)
    return ntuple(Val(3)) do axis
        layout.base_dims[axis] + layout.low_reach[axis] +
        layout.high_reach[axis]
    end
end

@inline function ct_shell_inside_base(layout::CTPackedShellLayout, i, j, k)
    nx, ny, nz = layout.base_dims
    return 1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz
end

@inline function ct_shell_inside_extended(layout::CTPackedShellLayout, i, j, k)
    nx, ny, nz = layout.base_dims
    lx, ly, lz = layout.low_reach
    hx, hy, hz = layout.high_reach
    return 1-lx <= i <= nx+hx && 1-ly <= j <= ny+hy &&
           1-lz <= k <= nz+hz
end

@inline function _ct_shell_linear3(i, j, k, ni, nj)
    return i + ni * ((j - Int32(1)) + nj * (k - Int32(1)))
end

"""
Return the one-based packed offset for an outer-shell point. Points inside
the base box return zero; points outside the extended box return `-1`.
"""
@inline function ct_shell_offset(layout::CTPackedShellLayout, i, j, k)
    ii, jj, kk = Int32(i), Int32(j), Int32(k)
    ct_shell_inside_extended(layout, ii, jj, kk) || return Int32(-1)
    ct_shell_inside_base(layout, ii, jj, kk) && return Int32(0)

    nx, ny, nz = layout.base_dims
    lx, ly, lz = layout.low_reach
    hx, hy, hz = layout.high_reach
    _, ey, ez = ct_shell_extended_dims(layout)
    starts = layout.slab_starts

    if ii < 1
        return starts[1] - 1 + _ct_shell_linear3(
            ii + lx, jj + ly, kk + lz, lx, ey,
        )
    elseif ii > nx
        return starts[2] - 1 + _ct_shell_linear3(
            ii - nx, jj + ly, kk + lz, hx, ey,
        )
    elseif jj < 1
        return starts[3] - 1 + _ct_shell_linear3(
            ii, jj + ly, kk + lz, nx, ly,
        )
    elseif jj > ny
        return starts[4] - 1 + _ct_shell_linear3(
            ii, jj - ny, kk + lz, nx, hy,
        )
    elseif kk < 1
        return starts[5] - 1 + _ct_shell_linear3(
            ii, jj, kk + lz, nx, ny,
        )
    end
    return starts[6] - 1 + _ct_shell_linear3(
        ii, jj, kk - nz, nx, ny,
    )
end

@inline function _ct_shell_unlinear3(linear, ni, nj)
    zero_based = linear - Int32(1)
    plane = ni * nj
    k = fld(zero_based, plane) + Int32(1)
    in_plane = zero_based - (k - Int32(1)) * plane
    j = fld(in_plane, ni) + Int32(1)
    i = in_plane - (j - Int32(1)) * ni + Int32(1)
    return i, j, k
end

"""Return the base-coordinate `(i,j,k)` represented by a packed offset."""
@inline function ct_shell_indices(layout::CTPackedShellLayout, offset)
    packed = Int32(offset)
    1 <= packed <= layout.shell_count || return (
        Int32(0), Int32(0), Int32(0),
    )

    nx, ny, nz = layout.base_dims
    lx, ly, lz = layout.low_reach
    hx, hy, hz = layout.high_reach
    _, ey, _ = ct_shell_extended_dims(layout)
    starts = layout.slab_starts

    if packed < starts[2]
        i, j, k = _ct_shell_unlinear3(packed-starts[1]+1, lx, ey)
        return i-lx, j-ly, k-lz
    elseif packed < starts[3]
        i, j, k = _ct_shell_unlinear3(packed-starts[2]+1, hx, ey)
        return i+nx, j-ly, k-lz
    elseif packed < starts[4]
        i, j, k = _ct_shell_unlinear3(packed-starts[3]+1, nx, ly)
        return i, j-ly, k-lz
    elseif packed < starts[5]
        i, j, k = _ct_shell_unlinear3(packed-starts[4]+1, nx, hy)
        return i, j+ny, k-lz
    elseif packed < starts[6]
        i, j, k = _ct_shell_unlinear3(packed-starts[5]+1, nx, ny)
        return i, j, k-lz
    end
    i, j, k = _ct_shell_unlinear3(packed-starts[6]+1, nx, ny)
    return i, j, k+nz
end

@inline function ct_shell_get(base, shell, layout::CTPackedShellLayout, i, j, k)
    offset = ct_shell_offset(layout, i, j, k)
    @inbounds return offset == 0 ? base[i,j,k] : shell[offset]
end

@inline function ct_shell_get_component(
    base, shell, layout::CTPackedShellLayout, i, j, k, component,
)
    offset = ct_shell_offset(layout, i, j, k)
    @inbounds return offset == 0 ? base[i,j,k,component] :
                                  shell[offset,component]
end

# -----------------------------------------------------------------------------
# Logical source addressing
# -----------------------------------------------------------------------------

struct CTDerivationCellSource
    bid::Int32
    index::NTuple{3,Int32}
end

struct CTDerivationOwnedCellSource
    rank::Int32
    bid::Int32
    local_index::NTuple{3,Int32}
end

struct CTDerivationFaceSource
    bid::Int32
    axis::Int8
    index::NTuple{3,Int32}
    orientation::Int8
end

struct CTDerivationOwnedFaceSource
    rank::Int32
    bid::Int32
    axis::Int8
    local_index::NTuple{3,Int32}
    orientation::Int8
end

@inline _ct_derivation_face_id(axis, side) =
    Int32(2 * Int(axis) - (side < 0 ? 1 : 0))

@inline function _ct_derivation_block_dims(block_dims, bid::Integer)
    dims = block_dims isa AbstractDict ? block_dims[Int(bid)] :
           block_dims[Int(bid)+1]
    length(dims) == 3 || throw(DimensionMismatch(
        "CT derivation block $bid must have exactly three dimensions",
    ))
    result = Int.(Tuple(dims))
    all(>(0), result) || throw(ArgumentError(
        "CT derivation block $bid dimensions must be positive",
    ))
    return result
end

function _ct_derivation_connection_transform(local_fid, conn)
    neighbor_fid = Int(getproperty(conn, :src_f))
    if hasproperty(conn, :transform) && getproperty(conn, :transform) !== nothing
        candidate = getproperty(conn, :transform)
        return candidate.source_face == neighbor_fid &&
               candidate.destination_face == local_fid ?
            candidate : structured_inverse_face_transform(candidate)
    end
    return structured_legacy_face_transform(
        neighbor_fid, local_fid, Bool(getproperty(conn, :reverse_tan)),
    )
end

"""
Resolve a block-global cell index through periodic and conformal inter-block
topology. Physical-boundary indices return `nothing`; topological junctions
select a deterministic canonical real cell.
"""
function ct_resolve_cell_source(
    bid::Integer,
    index::NTuple{3,<:Integer},
    block_dims,
    face_bc,
    connectivity;
    max_steps::Integer=12,
)
    State = Tuple{Int,NTuple{3,Int},Int}
    initial_dims = _ct_derivation_block_dims(block_dims, bid)
    queue = State[(Int(bid), Int.(index), 0)]
    visited = Set{Tuple{Int,NTuple{3,Int}}}()
    candidates = Tuple{Int,NTuple{3,Int}}[]

    while !isempty(queue)
        current_bid, current_index, depth = popfirst!(queue)
        state_key = (current_bid, current_index)
        state_key in visited && continue
        push!(visited, state_key)
        depth <= max_steps || continue
        current_dims = _ct_derivation_block_dims(block_dims, current_bid)

        if all(1 <= current_index[axis] <= current_dims[axis] for axis in 1:3)
            push!(candidates, (current_bid, current_index))
            continue
        end

        crossed_topology = false
        for axis in 1:3
            value = current_index[axis]
            side = value < 1 ? -1 : value > current_dims[axis] ? 1 : 0
            side == 0 && continue
            fid = Int(_ct_derivation_face_id(axis, side))
            bc = Int(get(face_bc, (current_bid, fid), -1))

            if bc == 2 # BC_PERIODIC
                wrapped = collect(current_index)
                wrapped[axis] += side < 0 ? current_dims[axis] :
                                                   -current_dims[axis]
                push!(queue, (current_bid, Tuple(wrapped), depth + 1))
                crossed_topology = true
                continue
            end

            bc == 0 || continue # BC_INTERBLOCK
            haskey(connectivity, (current_bid, fid)) || throw(ArgumentError(
                "missing connectivity at CT derivation face ($current_bid,$fid)",
            ))
            conn = connectivity[(current_bid, fid)]
            neighbor_bid = Int(getproperty(conn, :src_b))
            neighbor_fid = Int(getproperty(conn, :src_f))
            neighbor_dims = _ct_derivation_block_dims(block_dims, neighbor_bid)
            transform = _ct_derivation_connection_transform(fid, conn)
            destination_frame = structured_face_frame(fid)
            source_index = zeros(Int, 3)
            ghost_layer = side < 0 ? 1 - value : value - current_dims[axis]
            source_normal_axis = Int(structured_face_frame(neighbor_fid).normal_axis)

            for destination_axis in 1:3
                source_value = transform.source_for_destination[destination_axis]
                source_axis = abs(Int(source_value))
                if destination_axis == Int(destination_frame.normal_axis)
                    source_index[source_axis] = isodd(neighbor_fid) ?
                        ghost_layer : neighbor_dims[source_axis] - ghost_layer + 1
                else
                    tangential_value = current_index[destination_axis]
                    source_index[source_axis] = source_value < 0 ?
                        neighbor_dims[source_axis] - tangential_value + 1 :
                        tangential_value
                end
            end
            source_index[source_normal_axis] > 0 || throw(DimensionMismatch(
                "invalid CT derivation normal mapping at ($current_bid,$fid)",
            ))
            push!(queue, (neighbor_bid, Tuple(source_index), depth + 1))
            crossed_topology = true
        end

        # No topological continuation means at least one crossed face is a
        # physical boundary. Its primitive ghost state is supplied later by
        # the physical-BC task rather than this derivation halo.
        crossed_topology || continue
    end

    isempty(candidates) && return nothing
    sort!(candidates, by=candidate -> (candidate[1], candidate[2]...))
    source_bid, source_index = first(candidates)
    return CTDerivationCellSource(Int32(source_bid), Int32.(source_index))
end

"""Map a one-based global cell index to `(rank_coordinate, local_index)`."""
@inline function ct_partition_owner(index::Integer, ncell::Integer, nprocs::Integer)
    1 <= index <= ncell || throw(BoundsError(1:ncell, index))
    nprocs > 0 || throw(ArgumentError("partition count must be positive"))
    base = fld(Int(ncell), Int(nprocs))
    remainder = mod(Int(ncell), Int(nprocs))
    base > 0 || throw(ArgumentError(
        "cannot split $ncell cells over $nprocs ranks",
    ))
    long_extent = base + 1
    long_cells = remainder * long_extent
    if index <= long_cells
        coordinate = fld(Int(index)-1, long_extent)
        local_index = mod(Int(index)-1, long_extent) + 1
    else
        shifted = Int(index) - long_cells - 1
        coordinate = remainder + fld(shifted, base)
        local_index = mod(shifted, base) + 1
    end
    return Int32(coordinate), Int32(local_index)
end

function ct_owned_cell_source(
    source::CTDerivationCellSource,
    block_dims,
    block_nprocs,
    rank_offsets;
    ng::Integer,
)
    bid = Int(source.bid)
    dims = _ct_derivation_block_dims(block_dims, bid)
    layout = Tuple(Int.(block_nprocs[bid+1]))
    owners = ntuple(Val(3)) do axis
        ct_partition_owner(source.index[axis], dims[axis], layout[axis])
    end
    rx, ry, rz = ntuple(axis -> Int(owners[axis][1]), Val(3))
    rank = Int(rank_offsets[bid+1]) + rx * (layout[2] * layout[3]) +
           ry * layout[3] + rz
    local_index = ntuple(
        axis -> Int32(Int(owners[axis][2]) + Int(ng)), Val(3),
    )
    return CTDerivationOwnedCellSource(
        Int32(rank), Int32(bid), local_index,
    )
end

@inline function ct_signed_axis_map_determinant(axis_map)
    a, b, c = Int(axis_map[1]), Int(axis_map[2]), Int(axis_map[3])
    permutation_sign = if (abs(a), abs(b), abs(c)) in
        ((1,2,3), (2,3,1), (3,1,2))
        1
    else
        -1
    end
    component_sign = sign(a) * sign(b) * sign(c)
    return Int8(permutation_sign * component_sign)
end

@inline function ct_face_orientation_transform(axis_map, face_axis::Integer)
    mapped_axis = Int(axis_map[Int(face_axis)])
    return Int8(ct_signed_axis_map_determinant(axis_map) * sign(mapped_axis))
end

"""
Resolve a logical oriented face-flux sample. `axis` identifies the staggered
face family. The returned orientation maps source flux into the destination
family's positive parametric area-vector convention.
"""
function ct_resolve_face_source(
    bid::Integer,
    axis::Integer,
    index::NTuple{3,<:Integer},
    block_dims,
    face_bc,
    connectivity;
    max_steps::Integer=12,
)
    1 <= axis <= 3 || throw(ArgumentError("face axis must be in 1:3"))
    State = Tuple{Int,Int,NTuple{3,Int},Int8,Int}
    queue = State[(Int(bid), Int(axis), Int.(index), Int8(1), 0)]
    visited = Set{Tuple{Int,Int,NTuple{3,Int},Int8}}()
    candidates = Tuple{Int,Int,NTuple{3,Int},Int8}[]

    while !isempty(queue)
        current_bid, current_axis, current_index, orientation, depth =
            popfirst!(queue)
        state_key = (current_bid, current_axis, current_index, orientation)
        state_key in visited && continue
        push!(visited, state_key)
        depth <= max_steps || continue
        current_dims = _ct_derivation_block_dims(block_dims, current_bid)
        extents = ntuple(
            local_axis -> current_dims[local_axis] +
                          (local_axis == current_axis ? 1 : 0), Val(3),
        )

        if all(1 <= current_index[local_axis] <= extents[local_axis]
               for local_axis in 1:3)
            push!(candidates, (
                current_bid, current_axis, current_index, orientation,
            ))
            continue
        end

        crossed_topology = false
        for crossed_axis in 1:3
            value = current_index[crossed_axis]
            side = value < 1 ? -1 : value > extents[crossed_axis] ? 1 : 0
            side == 0 && continue
            fid = Int(_ct_derivation_face_id(crossed_axis, side))
            bc = Int(get(face_bc, (current_bid, fid), -1))

            if bc == 2 # BC_PERIODIC
                wrapped = collect(current_index)
                wrapped[crossed_axis] += side < 0 ?
                    current_dims[crossed_axis] : -current_dims[crossed_axis]
                push!(queue, (
                    current_bid, current_axis, Tuple(wrapped), orientation,
                    depth + 1,
                ))
                crossed_topology = true
                continue
            end

            bc == 0 || continue # BC_INTERBLOCK
            haskey(connectivity, (current_bid, fid)) || throw(ArgumentError(
                "missing connectivity at CT face-source boundary ($current_bid,$fid)",
            ))
            conn = connectivity[(current_bid, fid)]
            neighbor_bid = Int(getproperty(conn, :src_b))
            neighbor_fid = Int(getproperty(conn, :src_f))
            neighbor_dims = _ct_derivation_block_dims(block_dims, neighbor_bid)
            transform = _ct_derivation_connection_transform(fid, conn)
            axis_map = transform.source_for_destination
            destination_frame = structured_face_frame(fid)
            source_index = zeros(Int, 3)
            mapped_family_value = Int(axis_map[current_axis])
            source_axis = abs(mapped_family_value)
            next_orientation = Int8(
                orientation * ct_face_orientation_transform(axis_map, current_axis),
            )
            ghost_layer = side < 0 ? 1 - value :
                          value - extents[crossed_axis]

            for destination_axis in 1:3
                source_value = Int(axis_map[destination_axis])
                mapped_axis = abs(source_value)
                source_extent = neighbor_dims[mapped_axis] +
                                (mapped_axis == source_axis ? 1 : 0)
                if destination_axis == Int(destination_frame.normal_axis)
                    node_like_normal = current_axis == crossed_axis
                    source_index[mapped_axis] = if isodd(neighbor_fid)
                        node_like_normal ? 1 + ghost_layer : ghost_layer
                    else
                        node_like_normal ? source_extent - ghost_layer :
                                           source_extent - ghost_layer + 1
                    end
                else
                    tangential_value = current_index[destination_axis]
                    source_index[mapped_axis] = source_value < 0 ?
                        source_extent - tangential_value + 1 : tangential_value
                end
            end
            push!(queue, (
                neighbor_bid, source_axis, Tuple(source_index),
                next_orientation, depth + 1,
            ))
            crossed_topology = true
        end
        crossed_topology || continue
    end

    isempty(candidates) && return nothing
    sort!(candidates, by=candidate -> (
        candidate[1], candidate[2], candidate[3]..., candidate[4],
    ))
    source_bid, source_axis, source_index, orientation = first(candidates)
    return CTDerivationFaceSource(
        Int32(source_bid), Int8(source_axis), Int32.(source_index), orientation,
    )
end

function ct_owned_face_source(
    source::CTDerivationFaceSource,
    block_dims,
    block_nprocs,
    rank_offsets;
    ng::Integer,
)
    bid = Int(source.bid)
    face_axis = Int(source.axis)
    dims = _ct_derivation_block_dims(block_dims, bid)
    layout = Tuple(Int.(block_nprocs[bid+1]))
    owner_coordinates = zeros(Int, 3)
    local_indices = zeros(Int32, 3)

    for axis in 1:3
        logical_index = Int(source.index[axis])
        if axis == face_axis
            owner_cell = min(logical_index, dims[axis])
            owner_coordinate, local_cell = ct_partition_owner(
                owner_cell, dims[axis], layout[axis],
            )
            owner_coordinates[axis] = Int(owner_coordinate)
            local_face = Int(local_cell) +
                         (logical_index == dims[axis] + 1 ? 1 : 0)
            local_indices[axis] = Int32(local_face + Int(ng))
        else
            owner_coordinate, local_cell = ct_partition_owner(
                logical_index, dims[axis], layout[axis],
            )
            owner_coordinates[axis] = Int(owner_coordinate)
            local_indices[axis] = Int32(Int(local_cell) + Int(ng))
        end
    end

    rx, ry, rz = owner_coordinates
    rank = Int(rank_offsets[bid+1]) + rx * (layout[2] * layout[3]) +
           ry * layout[3] + rz
    return CTDerivationOwnedFaceSource(
        Int32(rank), Int32(bid), Int8(face_axis), Tuple(local_indices),
        source.orientation,
    )
end
