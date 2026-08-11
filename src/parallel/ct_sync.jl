# CT synchronization across conformal multi-block interfaces.
#
# A face update is topologically divergence-free only when every incident
# block uses the same oriented line EMF on a shared physical edge. Therefore
# edge values are synchronized before ct_update_face_b_from_emf_kernel! runs.

if !isdefined(@__MODULE__, :StructuredFaceFrame)
    include(joinpath(@__DIR__, "..", "core", "structured_interface_transform.jl"))
end

struct CTInterfaceExchange
    bid::Int
    fid::Int
    nb_bid::Int
    nb_fid::Int
    nb_rank::Int
    reverse_tan::Bool
    u_s::Int
    u_e::Int
    v_s::Int
    v_e::Int
    canonical_u_s::Int
    canonical_u_e::Int
    canonical_v_s::Int
    canonical_v_e::Int
    tag::Int
    transform::Union{Nothing,StructuredFaceTransform}
end

CTInterfaceExchange(
    bid, fid, nb_bid, nb_fid, nb_rank, reverse_tan,
    u_s, u_e, v_s, v_e, canonical_u_s, canonical_u_e,
    canonical_v_s, canonical_v_e, tag,
) = CTInterfaceExchange(
    bid, fid, nb_bid, nb_fid, nb_rank, reverse_tan,
    u_s, u_e, v_s, v_e, canonical_u_s, canonical_u_e,
    canonical_v_s, canonical_v_e, tag, nothing,
)

mutable struct CTSyncPlan{T}
    exchanges::Vector{CTInterfaceExchange}
    send_buffers::Vector{Vector{T}}
    recv_buffers::Vector{Vector{T}}
    halo_send_buffers::Vector{Vector{T}}
    halo_recv_buffers::Vector{Vector{T}}
    local_peer::Vector{Int}
    propagation_passes::Int
end

const CT_EDGE_MERGE_AVERAGE = :average
const CT_EDGE_MERGE_CANONICAL = :canonical

@inline function _ct_validate_edge_merge_mode(edge_merge::Symbol)
    edge_merge in (CT_EDGE_MERGE_AVERAGE, CT_EDGE_MERGE_CANONICAL) ||
        throw(ArgumentError(
            "unsupported CT edge merge mode=$edge_merge; expected " *
            ":average or :canonical",
        ))
    return edge_merge
end

struct CTBlockEdge
    bid::Int
    jside::Int
    kside::Int
end

struct CTJunctionFaceMember
    edge::CTBlockEdge
    face::Int8
    normal_orientation::Int8
end

struct CTJunctionPhysicalFace
    first::CTJunctionFaceMember
    second::CTJunctionFaceMember
    negative_edge::CTBlockEdge
    positive_edge::CTBlockEdge
end

struct CTJunctionTopology
    edges::Vector{CTBlockEdge}
    orientations::Dict{CTBlockEdge,Int8}
    faces::Vector{CTJunctionPhysicalFace}
end

struct CTJunctionJob
    local_edge::CTBlockEdge
    remote_edge::CTBlockEdge
    remote_rank::Int
    global_i_s::Int
    global_i_e::Int
    tag::Int
    local_orientation::Int8
    remote_orientation::Int8
    edge_length::Int
    group_index::Int
end


CTJunctionJob(
    local_edge, remote_edge, remote_rank,
    global_i_s, global_i_e, tag,
    local_orientation, remote_orientation, edge_length,
) = CTJunctionJob(
    local_edge, remote_edge, remote_rank,
    global_i_s, global_i_e, tag,
    local_orientation, remote_orientation, edge_length, 0,
)

struct CTJunctionLocalJob
    group_index::Int
    edge::CTBlockEdge
    canonical_i_s::Int
    canonical_i_e::Int
    edge_length::Int
    orientation::Int8
    face_indices::NTuple{2,Int}
    local_faces::NTuple{2,Int8}
    normal_orientations::NTuple{2,Int8}
end

const CT_JUNCTION_PAYLOAD_CELL_EMF = 1
const CT_JUNCTION_PAYLOAD_FACE1_EMF = 2
const CT_JUNCTION_PAYLOAD_FACE1_WEIGHT = 3
const CT_JUNCTION_PAYLOAD_FACE2_EMF = 4
const CT_JUNCTION_PAYLOAD_FACE2_WEIGHT = 5
const CT_JUNCTION_PAYLOAD_EDGE_LENGTH = 6
const CT_JUNCTION_PAYLOAD_RESISTIVE_CELL_EMF = 7
const CT_JUNCTION_PAYLOAD_RESISTIVE_FACE1_EMF = 8
const CT_JUNCTION_PAYLOAD_RESISTIVE_FACE2_EMF = 9
const CT_JUNCTION_PAYLOAD_SECTOR_ANGLE = 10
const CT_JUNCTION_PAYLOAD_FOFC_SCALE = 11
const CT_JUNCTION_PAYLOAD_WIDTH = 11

mutable struct CTJunctionPlan{T}
    send_jobs::Vector{CTJunctionJob}
    recv_jobs::Vector{CTJunctionJob}
    send_buffers::Vector{Vector{T}}
    recv_buffers::Vector{Vector{T}}
    topologies::Vector{CTJunctionTopology}
    local_jobs::Vector{CTJunctionLocalJob}
    payload_device_buffers::Vector{Any}
    payload_host_buffers::Vector{Matrix{T}}
    payload_send_buffers::Vector{Matrix{T}}
    payload_recv_buffers::Vector{Matrix{T}}
    junction_masks::Dict{Int,UInt16}
    pack_epoch::Union{Nothing,Tuple{Int,Int}}
    packed_jobs::BitVector
    resistive_packed_jobs::BitVector
    solved_epoch::Union{Nothing,Tuple{Int,Int}}
    generalized::Bool
end

function CTJunctionPlan(
    send_jobs::Vector{CTJunctionJob}, recv_jobs::Vector{CTJunctionJob},
    send_buffers::Vector{Vector{T}}, recv_buffers::Vector{Vector{T}},
) where {T}
    return CTJunctionPlan{T}(
        send_jobs, recv_jobs, send_buffers, recv_buffers,
        CTJunctionTopology[], CTJunctionLocalJob[], Any[], Matrix{T}[],
        Matrix{T}[], Matrix{T}[], Dict{Int,UInt16}(), nothing, falses(0),
        falses(0), nothing, false,
    )
end

@inline ct_interface_face_supported(fid::Integer) = 1 <= fid <= 6
@inline ct_face_side_sign(fid::Integer) = isodd(fid) ? -1 : 1

const CT_INTERFACE_TAG_BASE = 12000
const CT_JUNCTION_TAG_BASE = 22000
const CT_MPI_TAG_MAX = 32767

@inline function _ct_checked_tag(base, code, limit, description)
    tag = base + code
    if tag > limit
        error("Too many $description messages for the reserved MPI tag range: tag=$tag limit=$limit")
    end
    return tag
end

@inline function ct_interface_face_flux_sign(fid::Integer, nb_fid::Integer)
    return -(ct_face_side_sign(fid) * ct_face_side_sign(nb_fid))
end

@inline function ct_interface_face_flux_sign(
    fid::Integer, nb_fid::Integer, transform::StructuredFaceTransform,
)
    normal_sign = structured_face_transform_normal(transform) < 0 ? -1 : 1
    return ct_interface_face_flux_sign(fid, nb_fid) * normal_sign
end

@inline ct_interface_v_edge_sign(reverse_tan::Bool) = reverse_tan ? -1 : 1

@inline function ct_interface_buffer_length(u_len::Integer, v_len::Integer)
    # face flux: u*v; u-edge: u*(v+1); v-edge: (u+1)*v
    return u_len*v_len + u_len*(v_len + 1) + (u_len + 1)*v_len
end

"""
    ct_interface_line_residual(local, peer, u_len, v_len,
                               fid, nb_fid, reverse_tan)

Compare independently computed interface edge line integrals after orienting
the peer sheet into the local convention. This observer never mutates either
input buffer and deliberately ignores the face-flux portion of the sheet.
"""
function ct_interface_line_residual(
    local_buffer, peer_buffer, u_len::Integer, v_len::Integer,
    fid::Integer, nb_fid::Integer, reverse_tan::Bool;
    transform=nothing,
)
    u_len > 0 && v_len > 0 || throw(
        ArgumentError("interface dimensions must be positive"),
    )
    expected = ct_interface_buffer_length(u_len, v_len)
    length(local_buffer) == expected || throw(DimensionMismatch(
        "local interface buffer has $(length(local_buffer)) values; expected $expected",
    ))
    length(peer_buffer) == expected || throw(DimensionMismatch(
        "peer interface buffer has $(length(peer_buffer)) values; expected $expected",
    ))
    if !all(isfinite, local_buffer) || !all(isfinite, peer_buffer)
        return (
            absolute=Inf, relative=Inf, nonfinite_flag=Int32(1),
        )
    end

    nface = u_len*v_len
    nuedge = u_len*(v_len + 1)
    local_u = reshape(
        @view(local_buffer[nface+1:nface+nuedge]), u_len, v_len + 1,
    )
    local_v = reshape(
        @view(local_buffer[nface+nuedge+1:expected]), u_len + 1, v_len,
    )
    source_u_len, source_v_len = if transform !== nothing &&
        (structured_face_transform_code(transform) & 1) != 0
        (v_len, u_len)
    else
        (u_len, v_len)
    end
    source_nface = source_u_len * source_v_len
    source_nuedge = source_u_len * (source_v_len + 1)
    peer_u = reshape(
        @view(peer_buffer[source_nface+1:source_nface+source_nuedge]),
        source_u_len, source_v_len + 1,
    )
    peer_v = reshape(
        @view(peer_buffer[source_nface+source_nuedge+1:expected]),
        source_u_len + 1, source_v_len,
    )
    if transform !== nothing
        peer_face = reshape(
            @view(peer_buffer[1:source_nface]), source_u_len, source_v_len,
        )
        _, peer_u, peer_v = _ct_reorient_interface_sheet(
            peer_face, peer_u, peer_v, transform, u_len, v_len,
        )
    elseif reverse_tan
        peer_u = @view peer_u[:, end:-1:1]
        peer_v = @view peer_v[:, end:-1:1]
    end
    oriented_peer_v = ct_interface_v_edge_sign(reverse_tan) .* peer_v
    u_absolute = maximum(abs, local_u .- peer_u)
    v_absolute = maximum(abs, local_v .- oriented_peer_v)
    absolute = max(u_absolute, v_absolute)
    scale = max(
        maximum(abs, local_u), maximum(abs, peer_u),
        maximum(abs, local_v), maximum(abs, oriented_peer_v),
    )
    relative = iszero(scale) ? absolute : absolute/scale
    return (
        absolute=Float64(absolute), relative=Float64(relative),
        nonfinite_flag=Int32(0),
    )
end

@inline _ct_canonical_overlap(ex::CTInterfaceExchange) = (
    ex.canonical_u_s, ex.canonical_u_e,
    ex.canonical_v_s, ex.canonical_v_e,
)

function _ct_find_local_peer(exchanges, index, world_rank)
    ex = exchanges[index]
    matches = Int[]
    for candidate in eachindex(exchanges)
        other = exchanges[candidate]
        if other.bid == ex.nb_bid && other.fid == ex.nb_fid &&
           other.nb_bid == ex.bid && other.nb_fid == ex.fid &&
           other.nb_rank == world_rank &&
           (_ct_canonical_overlap(other) == _ct_canonical_overlap(ex) ||
            (ex.transform !== nothing && other.transform !== nothing &&
             (ex.u_e-ex.u_s+1)*(ex.v_e-ex.v_s+1) ==
             (other.u_e-other.u_s+1)*(other.v_e-other.v_s+1)))
            push!(matches, candidate)
        end
    end
    if length(matches) != 1
        error(
            "Expected exactly one local CT interface peer for " *
            "block=$(ex.bid) face=$(ex.fid) overlap=$(_ct_canonical_overlap(ex)); " *
            "found $(length(matches))",
        )
    end
    return only(matches)
end

@inline function ct_block_edge_from_face(bid, fid, endpoint)
    if fid == 1 || fid == 2 || fid == 3 || fid == 4
        return CTBlockEdge(bid, fid, endpoint == 1 ? 5 : 6)
    elseif fid == 5 || fid == 6
        return CTBlockEdge(bid, endpoint == 1 ? 3 : 4, fid)
    end
    error("Unsupported CT junction face $fid")
end

@inline function _ct_block_edge_from_tangent(bid, fid, tangent_axis, endpoint)
    frame = structured_face_frame(fid)
    tangent_axis == frame.u_axis || tangent_axis == frame.v_axis ||
        error("axis $tangent_axis is not tangent to face $fid")
    fixed_axis = tangent_axis
    fixed_face = 2 * fixed_axis - (endpoint == 1 ? 1 : 0)
    return (fid + 1) ÷ 2 < fixed_axis ?
        CTBlockEdge(bid, fid, fixed_face) :
        CTBlockEdge(bid, fixed_face, fid)
end

@inline ct_block_edge_key(edge::CTBlockEdge) = (
    edge.bid, edge.jside, edge.kside,
)

@inline function _ct_connection_transform(fid, conn)
    return hasproperty(conn, :transform) && conn.transform !== nothing ?
        conn.transform :
        structured_legacy_face_transform(fid, conn.src_f, conn.reverse_tan)
end

@inline function _ct_mapped_face_edge(
    bid, fid, conn, local_fixed_axis, endpoint,
)
    transform = _ct_connection_transform(fid, conn)
    peer_frame = structured_face_frame(conn.src_f)
    destination_fixed_axis = if abs(
        transform.source_for_destination[peer_frame.u_axis],
    ) == local_fixed_axis
        peer_frame.u_axis
    elseif abs(transform.source_for_destination[peer_frame.v_axis]) ==
           local_fixed_axis
        peer_frame.v_axis
    else
        error(
            "CT interface transform does not map tangent axis " *
            "$local_fixed_axis on block=$bid face=$fid",
        )
    end
    mapped_fixed_axis = transform.source_for_destination[
        destination_fixed_axis
    ]
    neighbor_endpoint = mapped_fixed_axis < 0 ? 3 - endpoint : endpoint
    local_edge = _ct_block_edge_from_tangent(
        bid, fid, local_fixed_axis, endpoint,
    )
    neighbor_edge = _ct_block_edge_from_tangent(
        conn.src_b, conn.src_f, destination_fixed_axis, neighbor_endpoint,
    )
    local_edge_axis = _ct_edge_axis(local_edge)
    neighbor_edge_axis = _ct_edge_axis(neighbor_edge)
    mapped_edge_axis = transform.source_for_destination[neighbor_edge_axis]
    abs(mapped_edge_axis) == local_edge_axis || error(
        "CT junction transform does not map edge axes: " *
        "$local_edge -> $neighbor_edge",
    )
    edge_orientation = Int8(mapped_edge_axis < 0 ? -1 : 1)
    normal_orientation = Int8(
        ct_interface_face_flux_sign(fid, conn.src_f, transform),
    )
    return local_edge, neighbor_edge, edge_orientation, normal_orientation
end

function _ct_edge_adjacency_and_face_relations(connectivity)
    adjacency = Dict{CTBlockEdge,Vector{Tuple{CTBlockEdge,Int8}}}()
    relations = NamedTuple[]
    for ((bid, fid), conn) in connectivity
        if !ct_interface_face_supported(fid) ||
           !ct_interface_face_supported(conn.src_f)
            continue
        end
        local_frame = structured_face_frame(fid)
        for local_fixed_axis in (local_frame.u_axis, local_frame.v_axis)
            for endpoint in 1:2
                local_edge, neighbor_edge, edge_orientation,
                    normal_orientation = _ct_mapped_face_edge(
                        bid, fid, conn, local_fixed_axis, endpoint,
                    )
                push!(
                    get!(adjacency, local_edge,
                         Tuple{CTBlockEdge,Int8}[]),
                    (neighbor_edge, edge_orientation),
                )
                push!(
                    get!(adjacency, neighbor_edge,
                         Tuple{CTBlockEdge,Int8}[]),
                    (local_edge, edge_orientation),
                )
                push!(relations, (
                    local_edge=local_edge,
                    local_face=Int8(fid),
                    neighbor_edge=neighbor_edge,
                    neighbor_face=Int8(conn.src_f),
                    edge_orientation=edge_orientation,
                    normal_orientation=normal_orientation,
                ))
            end
        end
    end
    for neighbors in values(adjacency)
        unique!(neighbors)
    end
    return adjacency, relations
end

function ct_edge_equivalence_groups_with_orientation(connectivity)
    adjacency, _ = _ct_edge_adjacency_and_face_relations(connectivity)

    result = NamedTuple[]
    visited = Set{CTBlockEdge}()
    seeds = sort!(collect(keys(adjacency)), by=ct_block_edge_key)
    for seed in seeds
        seed in visited && continue
        orientations = Dict{CTBlockEdge,Int8}(seed => Int8(1))
        pending = CTBlockEdge[seed]
        edges = CTBlockEdge[]
        while !isempty(pending)
            edge = pop!(pending)
            edge in visited && continue
            push!(visited, edge)
            push!(edges, edge)
            for (neighbor, relation) in adjacency[edge]
                expected = Int8(orientations[edge] * relation)
                if haskey(orientations, neighbor)
                    orientations[neighbor] == expected || error(
                        "inconsistent CT junction orientation cycle at $neighbor",
                    )
                else
                    orientations[neighbor] = expected
                end
                neighbor in visited || push!(pending, neighbor)
            end
        end
        unique!(edges)
        if length(edges) > 1
            sort!(edges, by=ct_block_edge_key)
            owner_sign = orientations[first(edges)]
            if owner_sign != Int8(1)
                for edge in edges
                    orientations[edge] = Int8(orientations[edge] * owner_sign)
                end
            end
            push!(result, (edges=edges, orientations=orientations))
        end
    end
    sort!(result, by=group -> ct_block_edge_key(first(group.edges)))
    return result
end

ct_junction_edge_groups_with_orientation(connectivity) = filter(
    group -> length(group.edges) >= 3,
    ct_edge_equivalence_groups_with_orientation(connectivity),
)

ct_edge_equivalence_groups(connectivity) = [
    group.edges
    for group in ct_edge_equivalence_groups_with_orientation(connectivity)
]

@inline _ct_interface_endpoint_key(edge, face) = (edge.bid, Int(face))

@inline function _ct_physical_face_key(relation)
    local_endpoint = _ct_interface_endpoint_key(
        relation.local_edge, relation.local_face,
    )
    neighbor_endpoint = _ct_interface_endpoint_key(
        relation.neighbor_edge, relation.neighbor_face,
    )
    return minmax(local_endpoint, neighbor_endpoint)
end

@inline function _ct_member_interior_side(member::CTJunctionFaceMember)
    local_side = isodd(member.face) ? Int8(1) : Int8(-1)
    return Int8(local_side * member.normal_orientation)
end

function ct_generalized_junction_topologies(connectivity)
    groups = ct_junction_edge_groups_with_orientation(connectivity)
    _, relations = _ct_edge_adjacency_and_face_relations(connectivity)
    topologies = CTJunctionTopology[]
    for group in groups
        length(group.edges) >= 3 || continue
        edge_set = Set(group.edges)
        face_relations = Dict{Any,Any}()
        for relation in relations
            relation.local_edge in edge_set || continue
            relation.neighbor_edge in edge_set || continue
            key = _ct_physical_face_key(relation)
            if haskey(face_relations, key)
                previous = face_relations[key]
                previous.edge_orientation == relation.edge_orientation ||
                    error("inconsistent CT junction edge orientation for $key")
                previous.normal_orientation == relation.normal_orientation ||
                    error("inconsistent CT junction face orientation for $key")
            else
                face_relations[key] = relation
            end
        end

        faces = CTJunctionPhysicalFace[]
        sector_degree = Dict(edge => 0 for edge in group.edges)
        for key in sort!(collect(keys(face_relations)))
            relation = face_relations[key]
            local_endpoint = _ct_interface_endpoint_key(
                relation.local_edge, relation.local_face,
            )
            local_is_first = local_endpoint == first(key)
            first_member = if local_is_first
                CTJunctionFaceMember(
                    relation.local_edge, relation.local_face, Int8(1),
                )
            else
                CTJunctionFaceMember(
                    relation.neighbor_edge, relation.neighbor_face, Int8(1),
                )
            end
            second_member = if local_is_first
                CTJunctionFaceMember(
                    relation.neighbor_edge, relation.neighbor_face,
                    relation.normal_orientation,
                )
            else
                CTJunctionFaceMember(
                    relation.local_edge, relation.local_face,
                    relation.normal_orientation,
                )
            end
            first_side = _ct_member_interior_side(first_member)
            second_side = _ct_member_interior_side(second_member)
            first_side == -second_side || error(
                "CT junction face does not separate opposite sectors: " *
                "$(first_member.edge) face=$(first_member.face), " *
                "$(second_member.edge) face=$(second_member.face)",
            )
            negative_edge = first_side < 0 ?
                first_member.edge : second_member.edge
            positive_edge = first_side > 0 ?
                first_member.edge : second_member.edge
            push!(faces, CTJunctionPhysicalFace(
                first_member, second_member, negative_edge, positive_edge,
            ))
            sector_degree[first_member.edge] += 1
            sector_degree[second_member.edge] += 1
        end
        length(faces) == length(group.edges) || error(
            "CT junction must form one closed face/sector cycle: " *
            "members=$(length(group.edges)) faces=$(length(faces))",
        )
        all(==(2), values(sector_degree)) || error(
            "every CT junction sector must be incident to exactly two faces",
        )
        push!(topologies, CTJunctionTopology(
            group.edges, group.orientations, faces,
        ))
    end
    return topologies
end

ct_junction_edge_groups(connectivity) = [
    group.edges for group in ct_junction_edge_groups_with_orientation(connectivity)
]

@inline function _ct_oriented_edge_extent(
    first_index, last_index, edge_length, orientation,
)
    return orientation < 0 ?
        (edge_length - last_index + 1, edge_length - first_index + 1) :
        (first_index, last_index)
end

@inline function _ct_nonuniform_extent(rank, rank_count, global_count)
    base = global_count ÷ rank_count
    remainder = global_count % rank_count
    first_index = min(rank, remainder) * (base + 1) +
        max(0, rank - remainder) * base + 1
    last_index = first_index + (rank < remainder ? base : base - 1)
    return first_index, last_index
end

@inline function _ct_edge_rank_coordinates(edge, layout)
    faces = (edge.jside, edge.kside)
    rx = 1 <= faces[1] <= 2 ? (isodd(faces[1]) ? 0 : layout[1] - 1) :
         1 <= faces[2] <= 2 ? (isodd(faces[2]) ? 0 : layout[1] - 1) : -1
    ry = 3 <= faces[1] <= 4 ? (isodd(faces[1]) ? 0 : layout[2] - 1) :
         3 <= faces[2] <= 4 ? (isodd(faces[2]) ? 0 : layout[2] - 1) : -1
    rz = 5 <= faces[1] <= 6 ? (isodd(faces[1]) ? 0 : layout[3] - 1) :
         5 <= faces[2] <= 6 ? (isodd(faces[2]) ? 0 : layout[3] - 1) : -1
    return rx, ry, rz
end

@inline function _ct_edge_rank_global(bid, rx, ry, rz, layout, rank_offsets)
    return rank_offsets[bid + 1] + rx*(layout[2]*layout[3]) +
        ry*layout[3] + rz
end

@inline function _ct_edge_axis(edge::CTBlockEdge)
    n1 = (edge.jside + 1) ÷ 2
    n2 = (edge.kside + 1) ÷ 2
    n1 != n2 || error("CT edge faces must have different normal axes: $edge")
    return only(setdiff((1, 2, 3), (n1, n2)))
end

@inline function _ct_junction_mask_bit(edge::CTBlockEdge)
    axis = _ct_edge_axis(edge)
    transverse_axes = Tuple(filter(!=(axis), (1, 2, 3)))
    faces = (edge.jside, edge.kside)
    first_face = ((faces[1] + 1) ÷ 2) == transverse_axes[1] ?
        faces[1] : faces[2]
    second_face = faces[1] == first_face ? faces[2] : faces[1]
    first_side = isodd(first_face) ? 0 : 1
    second_side = isodd(second_face) ? 0 : 1
    bit_index = 4 * (axis - 1) + 2 * first_side + second_side
    return UInt16(1) << bit_index
end

@inline ct_junction_edge_mask(plan::CTJunctionPlan, bid::Integer) =
    get(plan.junction_masks, Int(bid), UInt16(0))

function _ct_junction_local_job(
    group_index, topology, edge, block, Block_Nprocs, Nx_b, Ny_b, Nz_b,
)
    layout = Tuple(Int.(Block_Nprocs[edge.bid + 1]))
    edge_axis = _ct_edge_axis(edge)
    global_dims = (
        Nx_b[edge.bid + 1], Ny_b[edge.bid + 1], Nz_b[edge.bid + 1],
    )
    rank_axis = edge_axis == 1 ? block.rx :
                (edge_axis == 2 ? block.ry : block.rz)
    local_i_s, local_i_e = _ct_nonuniform_extent(
        rank_axis, layout[edge_axis], global_dims[edge_axis],
    )
    orientation = topology.orientations[edge]
    canonical_i_s, canonical_i_e = _ct_oriented_edge_extent(
        local_i_s, local_i_e, global_dims[edge_axis], orientation,
    )
    incidence = Tuple{Int,Int8,Int8}[]
    for (face_index, physical_face) in enumerate(topology.faces)
        for member in (physical_face.first, physical_face.second)
            member.edge == edge || continue
            push!(incidence, (
                face_index, member.face, member.normal_orientation,
            ))
        end
    end
    length(incidence) == 2 || error(
        "CT junction sector $edge must have exactly two face incidences",
    )
    sort!(incidence, by=first)
    return CTJunctionLocalJob(
        group_index, edge, canonical_i_s, canonical_i_e,
        global_dims[edge_axis], orientation,
        (incidence[1][1], incidence[2][1]),
        (incidence[1][2], incidence[2][2]),
        (incidence[1][3], incidence[2][3]),
    )
end

function build_ct_junction_plan(
    blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b,
    Ny_b=nothing, Nz_b=nothing,
)
    topologies = ct_generalized_junction_topologies(connectivity)
    groups = [
        (edges=topology.edges, orientations=topology.orientations)
        for topology in topologies
    ]
    send_jobs = CTJunctionJob[]
    recv_jobs = CTJunctionJob[]
    tag_codes = Dict{Tuple{Int,CTBlockEdge},Int}()
    next_tag_code = 0
    for (group_index, group) in enumerate(groups)
        edges = group.edges
        for remote_edge in edges[2:end]
            tag_codes[(group_index, remote_edge)] = next_tag_code
            next_tag_code += 1
        end
    end

    for (group_index, group) in enumerate(groups)
        edges = group.edges
        orientations = group.orientations
        owner = first(edges)
        owner_layout = Tuple(Int.(Block_Nprocs[owner.bid + 1]))
        owner_rx, owner_ry, owner_rz = _ct_edge_rank_coordinates(owner, owner_layout)
        for local_edge in edges
            if !haskey(blocks, local_edge.bid)
                continue
            end
            b = blocks[local_edge.bid]
            local_layout = Tuple(Int.(Block_Nprocs[local_edge.bid + 1]))
            local_rx, local_ry, local_rz = _ct_edge_rank_coordinates(
                local_edge, local_layout,
            )
            if (local_rx >= 0 && b.rx != local_rx) ||
               (local_ry >= 0 && b.ry != local_ry) ||
               (local_rz >= 0 && b.rz != local_rz)
                continue
            end
            local_axis = _ct_edge_axis(local_edge)
            local_global_dims = Ny_b === nothing || Nz_b === nothing ?
                (Nx_b[local_edge.bid + 1], Nx_b[local_edge.bid + 1], Nx_b[local_edge.bid + 1]) :
                (Nx_b[local_edge.bid + 1], Ny_b[local_edge.bid + 1], Nz_b[local_edge.bid + 1])
            local_rank_axis = (local_axis == 1 ? b.rx : local_axis == 2 ? b.ry : b.rz)
            local_layout_axis = local_layout[local_axis]
            local_x_s, local_x_e = _ct_nonuniform_extent(
                local_rank_axis, local_layout_axis, local_global_dims[local_axis],
            )

            if local_edge == owner
                for remote_edge in edges[2:end]
                    remote_layout = Tuple(Int.(Block_Nprocs[remote_edge.bid + 1]))
                    remote_rx_fixed, remote_ry, remote_rz = _ct_edge_rank_coordinates(
                        remote_edge, remote_layout,
                    )
                    remote_axis = _ct_edge_axis(remote_edge)
                    remote_global_dims = Ny_b === nothing || Nz_b === nothing ?
                        (Nx_b[remote_edge.bid + 1], Nx_b[remote_edge.bid + 1], Nx_b[remote_edge.bid + 1]) :
                        (Nx_b[remote_edge.bid + 1], Ny_b[remote_edge.bid + 1], Nz_b[remote_edge.bid + 1])
                    edge_length = local_global_dims[local_axis]
                    remote_global_dims[remote_axis] == edge_length || error(
                        "CT junction edges must have equal lengths: " *
                        "$owner has $edge_length, $remote_edge has " *
                        "$(remote_global_dims[remote_axis])",
                    )
                    remote_axis_rank_fixed = remote_axis == 1 ? remote_rx_fixed :
                        remote_axis == 2 ? remote_ry : remote_rz
                    remote_axis_values = remote_axis_rank_fixed >= 0 ?
                        (remote_axis_rank_fixed:remote_axis_rank_fixed) :
                        (0:remote_layout[remote_axis]-1)
                    for remote_axis_rank in remote_axis_values
                        remote_rx = remote_axis == 1 ? remote_axis_rank : remote_rx_fixed
                        remote_ry_use = remote_axis == 2 ? remote_axis_rank : remote_ry
                        remote_rz_use = remote_axis == 3 ? remote_axis_rank : remote_rz
                        remote_local_s, remote_local_e = _ct_nonuniform_extent(
                            remote_axis_rank, remote_layout[remote_axis],
                            remote_global_dims[remote_axis],
                        )
                        remote_x_s, remote_x_e = _ct_oriented_edge_extent(
                            remote_local_s, remote_local_e, edge_length,
                            orientations[remote_edge],
                        )
                        overlap_s = max(local_x_s, remote_x_s)
                        overlap_e = min(local_x_e, remote_x_e)
                        if overlap_s > overlap_e
                            continue
                        end
                        remote_rank = _ct_edge_rank_global(
                            remote_edge.bid, remote_rx, remote_ry_use, remote_rz_use,
                            remote_layout, rank_offsets,
                        )
                        tag = _ct_checked_tag(
                            CT_JUNCTION_TAG_BASE,
                            tag_codes[(group_index, remote_edge)],
                            CT_MPI_TAG_MAX,
                            "CT junction",
                        )
                        push!(send_jobs, CTJunctionJob(
                            local_edge, remote_edge, remote_rank,
                            overlap_s, overlap_e, tag,
                            orientations[local_edge],
                            orientations[remote_edge], edge_length,
                            group_index,
                        ))
                    end
                end
            else
                owner_rx_fixed, owner_ry, owner_rz = _ct_edge_rank_coordinates(
                    owner, owner_layout,
                )
                owner_axis = _ct_edge_axis(owner)
                owner_global_dims = Ny_b === nothing || Nz_b === nothing ?
                    (Nx_b[owner.bid + 1], Nx_b[owner.bid + 1], Nx_b[owner.bid + 1]) :
                    (Nx_b[owner.bid + 1], Ny_b[owner.bid + 1], Nz_b[owner.bid + 1])
                edge_length = owner_global_dims[owner_axis]
                local_global_dims[local_axis] == edge_length || error(
                    "CT junction edges must have equal lengths: " *
                    "$owner has $edge_length, $local_edge has " *
                    "$(local_global_dims[local_axis])",
                )
                local_x_s, local_x_e = _ct_oriented_edge_extent(
                    local_x_s, local_x_e, edge_length,
                    orientations[local_edge],
                )
                owner_axis_rank_fixed = owner_axis == 1 ? owner_rx_fixed :
                    owner_axis == 2 ? owner_ry : owner_rz
                owner_axis_values = owner_axis_rank_fixed >= 0 ?
                    (owner_axis_rank_fixed:owner_axis_rank_fixed) :
                    (0:owner_layout[owner_axis]-1)
                for owner_axis_rank in owner_axis_values
                    owner_rx = owner_axis == 1 ? owner_axis_rank : owner_rx_fixed
                    owner_ry_use = owner_axis == 2 ? owner_axis_rank : owner_ry
                    owner_rz_use = owner_axis == 3 ? owner_axis_rank : owner_rz
                    owner_x_s, owner_x_e = _ct_nonuniform_extent(
                        owner_axis_rank, owner_layout[owner_axis],
                        owner_global_dims[owner_axis],
                    )
                    overlap_s = max(local_x_s, owner_x_s)
                    overlap_e = min(local_x_e, owner_x_e)
                    if overlap_s > overlap_e
                        continue
                    end
                    owner_rank = _ct_edge_rank_global(
                        owner.bid, owner_rx, owner_ry_use, owner_rz_use,
                        owner_layout, rank_offsets,
                    )
                    tag = _ct_checked_tag(
                        CT_JUNCTION_TAG_BASE,
                        tag_codes[(group_index, local_edge)],
                        CT_MPI_TAG_MAX,
                        "CT junction",
                    )
                    push!(recv_jobs, CTJunctionJob(
                        local_edge, owner, owner_rank,
                        overlap_s, overlap_e, tag,
                        orientations[local_edge],
                        orientations[owner], edge_length,
                        group_index,
                    ))
                end
            end
        end
    end

    sort!(send_jobs, by=job -> (
        job.remote_rank, job.tag, job.global_i_s,
    ))
    sort!(recv_jobs, by=job -> (
        job.remote_rank, job.tag, job.global_i_s,
    ))
    send_buffers = [Vector{FT}(undef, job.global_i_e-job.global_i_s+1)
                    for job in send_jobs]
    recv_buffers = [Vector{FT}(undef, job.global_i_e-job.global_i_s+1)
                    for job in recv_jobs]
    local_jobs = CTJunctionLocalJob[]
    junction_masks = Dict{Int,UInt16}()
    for (group_index, topology) in enumerate(topologies)
        for edge in topology.edges
            haskey(blocks, edge.bid) || continue
            block = blocks[edge.bid]
            layout = Tuple(Int.(Block_Nprocs[edge.bid + 1]))
            edge_rx, edge_ry, edge_rz = _ct_edge_rank_coordinates(edge, layout)
            if (edge_rx >= 0 && block.rx != edge_rx) ||
               (edge_ry >= 0 && block.ry != edge_ry) ||
               (edge_rz >= 0 && block.rz != edge_rz)
                continue
            end
            push!(local_jobs, _ct_junction_local_job(
                group_index, topology, edge, block,
                Block_Nprocs, Nx_b, Ny_b, Nz_b,
            ))
            junction_masks[edge.bid] = get(
                junction_masks, edge.bid, UInt16(0),
            ) | _ct_junction_mask_bit(edge)
        end
    end
    sort!(local_jobs, by=job -> (
        job.group_index, ct_block_edge_key(job.edge), job.canonical_i_s,
    ))
    payload_device_buffers = Any[
        gpu_zeros(
            FT, CT_JUNCTION_PAYLOAD_WIDTH,
            job.canonical_i_e-job.canonical_i_s+1,
        ) for job in local_jobs
    ]
    payload_host_buffers = [
        zeros(
            FT, CT_JUNCTION_PAYLOAD_WIDTH,
            job.canonical_i_e-job.canonical_i_s+1,
        ) for job in local_jobs
    ]
    payload_send_buffers = [
        zeros(
            FT, CT_JUNCTION_PAYLOAD_WIDTH,
            job.global_i_e-job.global_i_s+1,
        ) for job in send_jobs
    ]
    payload_recv_buffers = [
        zeros(
            FT, CT_JUNCTION_PAYLOAD_WIDTH,
            job.global_i_e-job.global_i_s+1,
        ) for job in recv_jobs
    ]
    return CTJunctionPlan{FT}(
        send_jobs, recv_jobs, send_buffers, recv_buffers,
        topologies, local_jobs, payload_device_buffers,
        payload_host_buffers, payload_send_buffers, payload_recv_buffers,
        junction_masks, nothing, falses(length(local_jobs)),
        falses(length(local_jobs)), nothing, true,
    )
end

@inline function _scmm_junction_edge_array_index(
    edge, block, free_axis::Int, edge_cell::Int, ng::Int;
    endpoint::Bool=false,
)
    index = zeros(Int, 3)
    index[free_axis] = ng + edge_cell + (endpoint ? 1 : 0)
    for face in (edge.jside, edge.kside)
        axis = (face + 1) ÷ 2
        ncell = axis == 1 ? block.Nx : axis == 2 ? block.Ny : block.Nz
        index[axis] = isodd(face) ? ng + 1 : ng + ncell + 1
    end
    return Tuple(index)
end

@inline function _scmm_junction_local_cell(
    block, axis::Int, canonical_cell::Int,
    edge_length::Int, orientation::Int8,
)
    global_cell = orientation > 0 ? canonical_cell :
        edge_length - canonical_cell + 1
    offset = axis == 1 ? block.ox : axis == 2 ? block.oy : block.oz
    return global_cell - offset
end

function _pack_scmm_junction_metric_payload(
    metric_auxiliary, blocks, edge, orientation::Int8,
    canonical_start::Int, canonical_end::Int, edge_length::Int, ng::Int,
)
    block = blocks[edge.bid]
    auxiliary = metric_auxiliary[edge.bid]
    coordinates, workspace = auxiliary[1], auxiliary[2]
    axis = _ct_edge_axis(edge)
    count = canonical_end - canonical_start + 1
    payload = zeros(eltype(coordinates.x), 9, count)
    coordinate_arrays = (coordinates.x, coordinates.y, coordinates.z)
    node_dims = size(coordinates.x)
    for (column, canonical_cell) in enumerate(canonical_start:canonical_end)
        local_cell = _scmm_junction_local_cell(
            block, axis, canonical_cell, edge_length, orientation,
        )
        1 <= local_cell <= (axis == 1 ? block.Nx : axis == 2 ? block.Ny : block.Nz) ||
            throw(BoundsError("junction cell $canonical_cell is not local to block $(edge.bid)"))
        edge_index = _scmm_junction_edge_array_index(
            edge, block, axis, local_cell, ng,
        )
        start_index = _scmm_junction_edge_array_index(
            edge, block, axis, local_cell, ng;
            endpoint=orientation < 0,
        )
        for component in 1:3
            payload[component,column] = orientation *
                workspace[axis][component][edge_index...]
            payload[3+component,column] = orientation *
                _structured_scmm_axis_derivative(
                    coordinate_arrays[component], axis,
                    edge_index, node_dims,
                )
            payload[6+component,column] =
                coordinate_arrays[component][start_index...]
        end
    end
    return payload
end

function _unpack_scmm_junction_metric_payload!(
    metric_auxiliary, blocks, edge, orientation::Int8,
    canonical_start::Int, canonical_end::Int, edge_length::Int,
    payload, ng::Int,
)
    block = blocks[edge.bid]
    auxiliary = metric_auxiliary[edge.bid]
    coordinates, workspace = auxiliary[1], auxiliary[2]
    axis = _ct_edge_axis(edge)
    coordinate_arrays = (coordinates.x, coordinates.y, coordinates.z)
    T = eltype(coordinates.x)
    for (column, canonical_cell) in enumerate(canonical_start:canonical_end)
        local_cell = _scmm_junction_local_cell(
            block, axis, canonical_cell, edge_length, orientation,
        )
        edge_index = _scmm_junction_edge_array_index(
            edge, block, axis, local_cell, ng,
        )
        start_index = _scmm_junction_edge_array_index(
            edge, block, axis, local_cell, ng;
            endpoint=orientation < 0,
        )
        local_start = ntuple(
            component -> coordinate_arrays[component][start_index...], 3,
        )
        shift = ntuple(
            component -> local_start[component] - payload[6+component,column], 3,
        )
        canonical = _translated_scmm_edge(
            ntuple(component -> payload[component,column], 3),
            ntuple(component -> payload[3+component,column], 3),
            shift, T,
        )
        for component in 1:3
            workspace[axis][component][edge_index...] =
                orientation * canonical[component]
        end
    end
    return nothing
end

function sync_scmm_junction_metric_edges!(
    metric_auxiliary, blocks, plan::CTJunctionPlan, ng::Int,
)
    isempty(plan.topologies) && return nothing
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    send_payloads = Vector{Any}(undef, length(plan.send_jobs))
    recv_payloads = Vector{Any}(undef, length(plan.recv_jobs))
    for (index, job) in enumerate(plan.send_jobs)
        send_payloads[index] = _pack_scmm_junction_metric_payload(
            metric_auxiliary, blocks, job.local_edge, job.local_orientation,
            job.global_i_s, job.global_i_e, job.edge_length, ng,
        )
    end
    for (index, job) in enumerate(plan.recv_jobs)
        count = job.global_i_e - job.global_i_s + 1
        coordinates = metric_auxiliary[job.local_edge.bid][1]
        recv_payloads[index] = zeros(eltype(coordinates.x), 9, count)
    end

    requests = MPI.Request[]
    for (index, job) in enumerate(plan.recv_jobs)
        job.remote_rank == world_rank && continue
        push!(requests, MPI.Irecv!(
            recv_payloads[index], MPI.COMM_WORLD;
            source=job.remote_rank, tag=job.tag,
        ))
    end
    for (index, job) in enumerate(plan.send_jobs)
        job.remote_rank == world_rank && continue
        push!(requests, MPI.Isend(
            send_payloads[index], MPI.COMM_WORLD;
            dest=job.remote_rank, tag=job.tag,
        ))
    end
    isempty(requests) || MPI.Waitall(requests)

    for (recv_index, recv_job) in enumerate(plan.recv_jobs)
        if recv_job.remote_rank == world_rank
            send_index = findfirst(eachindex(plan.send_jobs)) do candidate
                send_job = plan.send_jobs[candidate]
                send_job.local_edge == recv_job.remote_edge &&
                    send_job.remote_edge == recv_job.local_edge &&
                    send_job.global_i_s == recv_job.global_i_s &&
                    send_job.global_i_e == recv_job.global_i_e &&
                    send_job.tag == recv_job.tag
            end
            send_index === nothing && error(
                "missing local canonical SCMM junction owner for $(recv_job.local_edge)",
            )
            copyto!(recv_payloads[recv_index], send_payloads[send_index])
        end
        _unpack_scmm_junction_metric_payload!(
            metric_auxiliary, blocks,
            recv_job.local_edge, recv_job.local_orientation,
            recv_job.global_i_s, recv_job.global_i_e,
            recv_job.edge_length, recv_payloads[recv_index], ng,
        )
    end
    MPI.Barrier(MPI.COMM_WORLD)
    return nothing
end

function build_ct_sync_plan(
    blocks, face_bc, connectivity, Block_Nprocs, rank_offsets,
    Nx_b, Ny_b, Nz_b,
)
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    exchanges = CTInterfaceExchange[]
    interface_keys = Tuple{Int,Int}[]
    for ((bid, fid), conn) in connectivity
        if ct_interface_face_supported(fid) &&
           ct_interface_face_supported(conn.src_f)
            endpoint = bid*6 + fid - 1
            nb_endpoint = conn.src_b*6 + conn.src_f - 1
            push!(interface_keys, minmax(endpoint, nb_endpoint))
        end
    end
    sort!(unique!(interface_keys))
    interface_tag_codes = Dict(
        key => index - 1 for (index, key) in enumerate(interface_keys)
    )

    for bid in sort(collect(keys(blocks)))
        b = blocks[bid]
        px, py, pz = Tuple(Int.(Block_Nprocs[bid + 1]))
        for fid in 1:6
            if get(face_bc, (bid, fid), -1) != 0
                continue
            end
            if !ct_interface_face_supported(fid)
                error("invalid CT inter-block face: block=$bid face=$fid")
            end
            if !_ms_subdomain_touches_face(fid, b.rx, b.ry, b.rz, px, py, pz)
                continue
            end

            conn = connectivity[(bid, fid)]
            nb_bid = conn.src_b
            nb_fid = conn.src_f
            if !ct_interface_face_supported(nb_fid)
                error("invalid CT neighbor inter-block face: block=$nb_bid face=$nb_fid")
            end

            u_d_s, u_d_e, v_d_s, v_d_e, v_total_d = _ms_face_uv_extent(
                fid, b.rx, b.ry, b.rz, px, py, pz,
                Nx_b[bid + 1], Ny_b[bid + 1], Nz_b[bid + 1],
            )
            npx, npy, npz = Tuple(Int.(Block_Nprocs[nb_bid + 1]))
            face_dims = (Nx_b[bid + 1], Ny_b[bid + 1], Nz_b[bid + 1])
            nb_face_dims = (Nx_b[nb_bid + 1], Ny_b[nb_bid + 1], Nz_b[nb_bid + 1])
            u_total_d = fid <= 2 ? face_dims[2] : fid <= 4 ? face_dims[1] : face_dims[1]
            u_total_nb = nb_fid <= 2 ? nb_face_dims[2] : nb_face_dims[1]
            v_total_nb = nb_fid <= 4 ? nb_face_dims[3] : nb_face_dims[2]
            n_total_d = fid <= 2 ? face_dims[1] : fid <= 4 ? face_dims[2] : face_dims[3]
            n_total_nb = nb_fid <= 2 ? nb_face_dims[1] : nb_fid <= 4 ? nb_face_dims[2] : nb_face_dims[3]
            if hasproperty(conn, :transform) && conn.transform !== nothing
                receive_transform = structured_inverse_face_transform(conn.transform)
                destination_frame = structured_face_frame(fid)
                mapped_u_axis = abs(receive_transform.source_for_destination[destination_frame.u_axis])
                mapped_v_axis = abs(receive_transform.source_for_destination[destination_frame.v_axis])
                u_total_nb = nb_face_dims[mapped_u_axis]
                v_total_nb = nb_face_dims[mapped_v_axis]
            end
            if u_total_d != u_total_nb || v_total_d != v_total_nb ||
               n_total_d != n_total_nb
                error(
                    "CT requires conforming inter-block faces; " *
                    "block=$bid face=$fid has ($u_total_d,$v_total_d) cells, " *
                    "neighbor block=$nb_bid face=$nb_fid has " *
                    "($u_total_nb,$v_total_nb)",
                )
            end
            for rank_s in 0:(npx*npy*npz - 1)
                sx = rank_s ÷ (npy*npz)
                sy = (rank_s ÷ npz) % npy
                sz = rank_s % npz
                if !_ms_subdomain_touches_face(nb_fid, sx, sy, sz, npx, npy, npz)
                    continue
                end
                u_s_s, u_s_e, v_s_s, v_s_e, v_total_s = _ms_face_uv_extent(
                    nb_fid, sx, sy, sz, npx, npy, npz,
                    Nx_b[nb_bid + 1], Ny_b[nb_bid + 1], Nz_b[nb_bid + 1],
                )
                mapped_u_s, mapped_u_e, mapped_v_s, mapped_v_e = if hasproperty(conn, :transform) && conn.transform !== nothing
                    source_ranges = (
                        _ms_nonuniform_extent(sx, npx, Nx_b[nb_bid + 1]),
                        _ms_nonuniform_extent(sy, npy, Ny_b[nb_bid + 1]),
                        _ms_nonuniform_extent(sz, npz, Nz_b[nb_bid + 1]),
                    )
                    mapped = structured_map_face_extent(
                        structured_inverse_face_transform(conn.transform),
                        source_ranges,
                        (Nx_b[nb_bid + 1], Ny_b[nb_bid + 1], Nz_b[nb_bid + 1]),
                    )
                    (mapped[1][1], mapped[1][2], mapped[2][1], mapped[2][2])
                elseif conn.reverse_tan
                    (u_s_s, u_s_e, v_total_s - v_s_e + 1, v_total_s - v_s_s + 1)
                else
                    (u_s_s, u_s_e, v_s_s, v_s_e)
                end
                u_int_s = max(u_d_s, mapped_u_s)
                u_int_e = min(u_d_e, mapped_u_e)
                v_int_s = max(v_d_s, mapped_v_s)
                v_int_e = min(v_d_e, mapped_v_e)
                if u_int_s > u_int_e || v_int_s > v_int_e
                    continue
                end

                local_u_s = u_int_s - u_d_s + 1
                local_u_e = u_int_e - u_d_s + 1
                local_v_s = v_int_s - v_d_s + 1
                local_v_e = v_int_e - v_d_s + 1
                nb_rank = rank_offsets[nb_bid + 1] + rank_s
                endpoint = bid*6 + fid - 1
                nb_endpoint = nb_bid*6 + nb_fid - 1
                interface_key = minmax(endpoint, nb_endpoint)
                tag = _ct_checked_tag(
                    CT_INTERFACE_TAG_BASE,
                    interface_tag_codes[interface_key],
                    CT_JUNCTION_TAG_BASE - 1,
                    "CT interface",
                )
                local_is_canonical = endpoint == first(interface_key)
                canonical_u_s, canonical_u_e, canonical_v_s, canonical_v_e = if local_is_canonical
                    (u_int_s, u_int_e, v_int_s, v_int_e)
                elseif hasproperty(conn, :transform) && conn.transform !== nothing
                    local_frame = structured_face_frame(fid)
                    local_ranges = ntuple(axis ->
                        axis == local_frame.u_axis ? (u_int_s, u_int_e) :
                        axis == local_frame.v_axis ? (v_int_s, v_int_e) :
                        (1, face_dims[axis]), 3)
                    mapped = structured_map_face_extent(
                        conn.transform, local_ranges, face_dims,
                    )
                    (mapped[1][1], mapped[1][2], mapped[2][1], mapped[2][2])
                elseif conn.reverse_tan
                    (u_int_s, u_int_e,
                     v_total_d - v_int_e + 1, v_total_d - v_int_s + 1)
                else
                    (u_int_s, u_int_e, v_int_s, v_int_e)
                end
                push!(exchanges, CTInterfaceExchange(
                    bid, fid, nb_bid, nb_fid, nb_rank, conn.reverse_tan,
                    local_u_s, local_u_e, local_v_s, local_v_e,
                    canonical_u_s, canonical_u_e, canonical_v_s, canonical_v_e, tag,
                    hasproperty(conn, :transform) ? conn.transform : nothing,
                ))
            end
        end
    end

    sort!(exchanges, by=ex -> (
        ex.nb_rank, ex.bid, ex.fid, ex.nb_bid, ex.nb_fid,
        ex.u_s, ex.v_s,
    ))
    send_buffers = Vector{Vector{FT}}(undef, length(exchanges))
    recv_buffers = Vector{Vector{FT}}(undef, length(exchanges))
    halo_send_buffers = Vector{Vector{FT}}(undef, length(exchanges))
    halo_recv_buffers = Vector{Vector{FT}}(undef, length(exchanges))
    for (index, ex) in enumerate(exchanges)
        u_len = ex.u_e - ex.u_s + 1
        v_len = ex.v_e - ex.v_s + 1
        nvalue = ct_interface_buffer_length(u_len, v_len)
        send_buffers[index] = Vector{FT}(undef, nvalue)
        recv_buffers[index] = Vector{FT}(undef, nvalue)
        halo_nvalue = ct_interface_halo_buffer_length(u_len, v_len)
        halo_send_buffers[index] = Vector{FT}(undef, halo_nvalue)
        halo_recv_buffers[index] = Vector{FT}(undef, halo_nvalue)
    end

    remote_message_keys = Set{Tuple{Int,Int}}()
    for ex in exchanges
        if ex.nb_rank == world_rank
            continue
        end
        message_key = (ex.nb_rank, ex.tag)
        if message_key in remote_message_keys
            error(
                "Ambiguous CT MPI messages to rank=$(ex.nb_rank) tag=$(ex.tag); " *
                "each rank pair must share at most one overlap per interface",
            )
        end
        push!(remote_message_keys, message_key)
    end

    local_peer = fill(0, length(exchanges))
    for (index, ex) in enumerate(exchanges)
        if ex.nb_rank != world_rank
            continue
        end
        local_peer[index] = _ct_find_local_peer(exchanges, index, world_rank)
    end

    return CTSyncPlan(
        exchanges, send_buffers, recv_buffers,
        halo_send_buffers, halo_recv_buffers, local_peer,
        1,
    )
end

@inline function _ct_interface_views(b, fid, u_s, u_e, v_s, v_e)
    if fid == 1 || fid == 2
        face_i = fid == 1 ? NG + 1 : b.Nx + NG + 1
        edge_i = fid == 1 ? 1 : b.Nx + 1
        face = @view b.Bx_face[face_i, u_s+NG:u_e+NG, v_s+NG:v_e+NG]
        u_edge = @view b.Ey_edge[edge_i, u_s:u_e, v_s:v_e+1]
        v_edge = @view b.Ez_edge[edge_i, u_s:u_e+1, v_s:v_e]
    elseif fid == 3 || fid == 4
        face_j = fid == 3 ? NG + 1 : b.Ny + NG + 1
        edge_j = fid == 3 ? 1 : b.Ny + 1
        face = @view b.By_face[u_s+NG:u_e+NG, face_j, v_s+NG:v_e+NG]
        u_edge = @view b.Ex_edge[u_s:u_e, edge_j, v_s:v_e+1]
        v_edge = @view b.Ez_edge[u_s:u_e+1, edge_j, v_s:v_e]
    else
        face_k = fid == 5 ? NG + 1 : b.Nz + NG + 1
        edge_k = fid == 5 ? 1 : b.Nz + 1
        face = @view b.Bz_face[u_s+NG:u_e+NG, v_s+NG:v_e+NG, face_k]
        u_edge = @view b.Ex_edge[u_s:u_e, v_s:v_e+1, edge_k]
        v_edge = @view b.Ey_edge[u_s:u_e+1, v_s:v_e, edge_k]
    end
    return face, u_edge, v_edge
end

@inline _ct_reverse_index(index, extent, reversed) =
    reversed ? extent + 1 - index : index

function _ct_reorient_interface_sheet(
    face_src, u_src, v_src, transform::StructuredFaceTransform,
    u_len::Integer, v_len::Integer,
)
    destination = structured_face_frame(transform.destination_face)
    source = structured_face_frame(transform.source_face)
    map_u = transform.source_for_destination[destination.u_axis]
    map_v = transform.source_for_destination[destination.v_axis]
    source_u_len = size(face_src, 1)
    source_v_len = size(face_src, 2)
    face = similar(face_src, u_len, v_len)
    u_edge = similar(u_src, u_len, v_len + 1)
    v_edge = similar(v_src, u_len + 1, v_len)
    for j in 1:v_len, i in 1:u_len
        source_coordinates = zeros(Int, 3)
        map_u_extent = abs(map_u) == source.u_axis ? source_u_len : source_v_len
        map_v_extent = abs(map_v) == source.u_axis ? source_u_len : source_v_len
        source_coordinates[abs(map_u)] = _ct_reverse_index(
            i, map_u_extent, map_u < 0,
        )
        source_coordinates[abs(map_v)] = _ct_reverse_index(
            j, map_v_extent, map_v < 0,
        )
        face[i, j] = face_src[
            source_coordinates[source.u_axis], source_coordinates[source.v_axis],
        ]
    end
    for j in 1:(v_len + 1), i in 1:u_len
        source_u_edge = abs(map_u) == source.u_axis
        source_segment_extent = source_u_edge ? size(u_src, 1) : size(v_src, 2)
        source_node_extent = source_u_edge ? size(u_src, 2) : size(v_src, 1)
        source_segment = _ct_reverse_index(
            i, source_segment_extent, map_u < 0,
        )
        source_node = _ct_reverse_index(
            j, source_node_extent, map_v < 0,
        )
        if abs(map_u) == source.u_axis
            u_edge[i, j] = FT(map_u < 0 ? -1 : 1) * u_src[source_segment, source_node]
        else
            u_edge[i, j] = FT(map_u < 0 ? -1 : 1) * v_src[source_node, source_segment]
        end
    end
    for j in 1:v_len, i in 1:(u_len + 1)
        source_v_edge = abs(map_v) == source.u_axis
        source_segment_extent = source_v_edge ? size(u_src, 1) : size(v_src, 2)
        source_node_extent = source_v_edge ? size(u_src, 2) : size(v_src, 1)
        source_segment = _ct_reverse_index(
            j, source_segment_extent, map_v < 0,
        )
        source_node = _ct_reverse_index(
            i, source_node_extent, map_u < 0,
        )
        if abs(map_v) == source.u_axis
            v_edge[i, j] = FT(map_v < 0 ? -1 : 1) * u_src[source_segment, source_node]
        else
            v_edge[i, j] = FT(map_v < 0 ? -1 : 1) * v_src[source_node, source_segment]
        end
    end
    return face, u_edge, v_edge
end

function ct_pack_interface_sheet!(buffer, b, ex::CTInterfaceExchange)
    face, u_edge, v_edge = _ct_interface_views(
        b, ex.fid, ex.u_s, ex.u_e, ex.v_s, ex.v_e,
    )
    offset = 0
    for values in (face, u_edge, v_edge)
        host_values = Array(values)
        nvalue = length(host_values)
        copyto!(buffer, offset + 1, vec(host_values), 1, nvalue)
        offset += nvalue
    end
    return buffer
end

function _ct_copy_host_sheet_to_device!(destination, source)
    device_source = GPUArray(Array(source))
    copyto!(destination, device_source)
    return nothing
end

@inline function _ct_interface_buffer_views(buffer, u_len, v_len)
    nface = u_len*v_len
    nuedge = u_len*(v_len + 1)
    nvedge = (u_len + 1)*v_len
    length(buffer) == nface + nuedge + nvedge || throw(DimensionMismatch(
        "CT interface buffer has length $(length(buffer)); " *
        "expected $(nface + nuedge + nvedge)",
    ))
    face = reshape(@view(buffer[1:nface]), u_len, v_len)
    u_edge = reshape(
        @view(buffer[nface+1:nface+nuedge]), u_len, v_len + 1,
    )
    v_edge = reshape(
        @view(buffer[nface+nuedge+1:nface+nuedge+nvedge]), u_len + 1, v_len,
    )
    return face, u_edge, v_edge
end

function ct_unpack_interface_sheet!(
    b, ex::CTInterfaceExchange, buffer;
    local_buffer=nothing, sync_face_flux::Bool, sync_edges::Bool,
    edge_merge::Symbol=CT_EDGE_MERGE_AVERAGE,
)
    sync_edges && _ct_validate_edge_merge_mode(edge_merge)
    u_len = ex.u_e - ex.u_s + 1
    v_len = ex.v_e - ex.v_s + 1
    source_u_len, source_v_len = if ex.transform !== nothing &&
        (structured_face_transform_code(structured_inverse_face_transform(ex.transform)) & 1) != 0
        (v_len, u_len)
    else
        (u_len, v_len)
    end
    face_src, u_edge_src, v_edge_src = _ct_interface_buffer_views(
        buffer, source_u_len, source_v_len,
    )

    if ex.transform !== nothing
        # Connectivity stores local->neighbor axes; unpack needs the inverse
        # neighbor->local orientation.
        face_values, u_edge_values, v_edge_values = _ct_reorient_interface_sheet(
            face_src, u_edge_src, v_edge_src,
            structured_inverse_face_transform(ex.transform), u_len, v_len,
        )
    elseif ex.reverse_tan
        face_values = Array(face_src[:, end:-1:1])
        u_edge_values = Array(u_edge_src[:, end:-1:1])
        v_edge_values = Array(v_edge_src[:, end:-1:1])
    else
        face_values = Array(face_src)
        u_edge_values = Array(u_edge_src)
        v_edge_values = Array(v_edge_src)
    end
    face_values .*= FT(ex.transform === nothing ?
        ct_interface_face_flux_sign(ex.fid, ex.nb_fid) :
        ct_interface_face_flux_sign(ex.fid, ex.nb_fid, ex.transform))
    if ex.transform === nothing
        v_edge_values .*= FT(ct_interface_v_edge_sign(ex.reverse_tan))
    end

    face, u_edge, v_edge = _ct_interface_views(
        b, ex.fid, ex.u_s, ex.u_e, ex.v_s, ex.v_e,
    )
    local_is_canonical = (ex.bid, ex.fid) < (ex.nb_bid, ex.nb_fid)
    if sync_face_flux && !local_is_canonical
        _ct_copy_host_sheet_to_device!(face, face_values)
    end
    if sync_edges
        if edge_merge == CT_EDGE_MERGE_CANONICAL
            if !local_is_canonical
                _ct_copy_host_sheet_to_device!(u_edge, u_edge_values)
                _ct_copy_host_sheet_to_device!(v_edge, v_edge_values)
            end
        else
            if local_buffer === nothing
                local_u_values = Array(u_edge)
                local_v_values = Array(v_edge)
            else
                _, local_u_values, local_v_values = _ct_interface_buffer_views(
                    local_buffer, u_len, v_len,
                )
            end
            u_edge_values .+= local_u_values
            u_edge_values .*= FT(0.5)
            v_edge_values .+= local_v_values
            v_edge_values .*= FT(0.5)
            _ct_copy_host_sheet_to_device!(u_edge, u_edge_values)
            _ct_copy_host_sheet_to_device!(v_edge, v_edge_values)
        end
    end
    return nothing
end

function ct_sync_interface_sheets!(
    blocks, plan::CTSyncPlan;
    sync_face_flux::Bool=false,
    sync_edges::Bool=true,
    edge_merge::Symbol=CT_EDGE_MERGE_AVERAGE,
)
    sync_edges && _ct_validate_edge_merge_mode(edge_merge)
    if isempty(plan.exchanges)
        return nothing
    end
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    passes = sync_edges ? plan.propagation_passes : 1

    for _ in 1:passes
        for (index, ex) in enumerate(plan.exchanges)
            ct_pack_interface_sheet!(plan.send_buffers[index], blocks[ex.bid], ex)
        end

        requests = MPI.Request[]
        for (index, ex) in enumerate(plan.exchanges)
            if ex.nb_rank != world_rank
                push!(requests, MPI.Irecv!(
                    plan.recv_buffers[index], MPI.COMM_WORLD;
                    source=ex.nb_rank, tag=ex.tag,
                ))
            end
        end
        for (index, ex) in enumerate(plan.exchanges)
            if ex.nb_rank != world_rank
                push!(requests, MPI.Isend(
                    plan.send_buffers[index], MPI.COMM_WORLD;
                    dest=ex.nb_rank, tag=ex.tag,
                ))
            end
        end
        if !isempty(requests)
            MPI.Waitall(requests)
        end

        for (index, ex) in enumerate(plan.exchanges)
            source = ex.nb_rank == world_rank ?
                plan.send_buffers[plan.local_peer[index]] : plan.recv_buffers[index]
            ct_unpack_interface_sheet!(
                blocks[ex.bid], ex, source;
                local_buffer=plan.send_buffers[index],
                sync_face_flux=sync_face_flux, sync_edges=sync_edges,
                edge_merge=edge_merge,
            )
        end
        gpu_sync()
    end
    return nothing
end

@inline function _ct_block_edge_view(b, edge::CTBlockEdge, global_i_s, global_i_e)
    axis = _ct_edge_axis(edge)
    offsets = (b.ox, b.oy, b.oz)
    dims = (b.Nx, b.Ny, b.Nz)
    local_s = global_i_s - offsets[axis]
    local_e = global_i_e - offsets[axis]
    indices = ntuple(d -> begin
        d == axis ? (local_s:local_e) :
        (d == (edge.jside + 1) ÷ 2 ?
            (isodd(edge.jside) ? 1 : dims[d] + 1) :
         d == (edge.kside + 1) ÷ 2 ?
            (isodd(edge.kside) ? 1 : dims[d] + 1) : 1)
    end, 3)
    if axis == 1
        return @view b.Ex_edge[indices[1], indices[2], indices[3]]
    elseif axis == 2
        return @view b.Ey_edge[indices[1], indices[2], indices[3]]
    else
        return @view b.Ez_edge[indices[1], indices[2], indices[3]]
    end
end

@inline function _ct_junction_local_view(b, job::CTJunctionJob)
    local_s, local_e = _ct_oriented_edge_extent(
        job.global_i_s, job.global_i_e,
        job.edge_length, job.local_orientation,
    )
    return _ct_block_edge_view(b, job.local_edge, local_s, local_e)
end

@inline function _ct_normalize_junction_epoch(epoch)
    length(epoch) == 2 || throw(ArgumentError(
        "CT junction epoch must be (step, rk_stage)",
    ))
    return (Int(epoch[1]), Int(epoch[2]))
end

function _ct_prepare_junction_pack_epoch!(plan::CTJunctionPlan, epoch)
    normalized = _ct_normalize_junction_epoch(epoch)
    if plan.pack_epoch != normalized
        if plan.pack_epoch !== nothing && any(plan.packed_jobs) &&
           plan.solved_epoch != plan.pack_epoch
            error(
                "cannot start CT junction epoch=$normalized before " *
                "epoch=$(plan.pack_epoch) has been solved",
            )
        end
        fill!(plan.packed_jobs, false)
        fill!(plan.resistive_packed_jobs, false)
        plan.pack_epoch = normalized
        plan.solved_epoch = nothing
    end
    return normalized
end

function ct_pack_generalized_junction_payloads!(
    plan::CTJunctionPlan, b, epoch,
    Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z, dt,
)
    plan.generalized || return nothing
    job_indices = findall(job -> job.edge.bid == b.id, plan.local_jobs)
    isempty(job_indices) && return nothing
    normalized = _ct_prepare_junction_pack_epoch!(plan, epoch)
    any(plan.packed_jobs[index] for index in job_indices) && error(
        "duplicate CT junction payload pack for block=$(b.id) " *
        "epoch=$normalized",
    )

    # Fx/Fy/Fz are rank-shared scratch. Complete their producers before this
    # sparse read, then complete the read before the next block may overwrite
    # the same arrays.
    gpu_sync()
    for index in job_indices
        job = plan.local_jobs[index]
        ct_launch_generalized_junction_payload!(
            plan.payload_device_buffers[index],
            b, Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z, dt,
            _ct_edge_axis(job.edge),
            job.local_faces[1], job.local_faces[2],
            job.orientation,
            job.normal_orientations[1], job.normal_orientations[2],
        )
    end
    gpu_sync()
    for index in job_indices
        plan.packed_jobs[index] = true
    end
    return nothing
end

function ct_pack_generalized_junction_resistive_payloads!(
    plan::CTJunctionPlan, b, epoch,
    Fv_x, Fv_y, Fv_z, emf_x, emf_y, emf_z,
)
    plan.generalized || return nothing
    job_indices = findall(job -> job.edge.bid == b.id, plan.local_jobs)
    isempty(job_indices) && return nothing
    normalized = _ct_normalize_junction_epoch(epoch)
    plan.pack_epoch == normalized || error(
        "resistive CT junction payload precedes ideal payload for " *
        "block=$(b.id) epoch=$normalized",
    )
    all(plan.packed_jobs[index] for index in job_indices) || error(
        "resistive CT junction payload precedes ideal payload for " *
        "block=$(b.id) epoch=$normalized",
    )
    any(plan.resistive_packed_jobs[index] for index in job_indices) && error(
        "duplicate resistive CT junction payload pack for block=$(b.id) " *
        "epoch=$normalized",
    )
    gpu_sync()
    for index in job_indices
        job = plan.local_jobs[index]
        ct_launch_generalized_junction_resistive_payload!(
            plan.payload_device_buffers[index],
            b, Fv_x, Fv_y, Fv_z, emf_x, emf_y, emf_z,
            _ct_edge_axis(job.edge),
            job.local_faces[1], job.local_faces[2],
            job.orientation,
        )
    end
    gpu_sync()
    for index in job_indices
        plan.resistive_packed_jobs[index] = true
    end
    return nothing
end

function ct_assert_generalized_junction_payload_ready!(
    plan::CTJunctionPlan, epoch; require_resistive::Bool=false,
)
    plan.generalized || return nothing
    normalized = _ct_normalize_junction_epoch(epoch)
    if isempty(plan.local_jobs)
        return normalized
    end
    plan.pack_epoch == normalized || error(
        "CT junction solve epoch=$normalized has no matching payload epoch; " *
        "found $(plan.pack_epoch)",
    )
    all(plan.packed_jobs) || error(
        "CT junction solve epoch=$normalized started before every local " *
        "rk_edge_emf payload was packed",
    )
    if require_resistive && !all(plan.resistive_packed_jobs)
        error(
            "CT junction solve epoch=$normalized started before every local " *
            "resistive payload was packed",
        )
    end
    return normalized
end

function ct_assert_generalized_junction_solution_ready!(
    plan::CTJunctionPlan, epoch,
)
    plan.generalized || return nothing
    normalized = _ct_normalize_junction_epoch(epoch)
    plan.solved_epoch == normalized || error(
        "face-B update epoch=$normalized started before generalized CT " *
        "junction solve; solved_epoch=$(plan.solved_epoch)",
    )
    return nothing
end

@inline function _ct_payload_tuple(buffer, column)
    return ntuple(
        channel -> buffer[channel, column],
        Val(CT_JUNCTION_PAYLOAD_WIDTH),
    )
end

function _ct_topology_sector_faces(topology::CTJunctionTopology, edge)
    indices = Int[]
    for (face_index, face) in enumerate(topology.faces)
        if face.first.edge == edge || face.second.edge == edge
            push!(indices, face_index)
        end
    end
    length(indices) == 2 || error(
        "CT junction sector $edge does not have two physical faces",
    )
    sort!(indices)
    return (indices[1], indices[2])
end

@inline function _ct_payload_face_sample(payload, sector_faces, face_index)
    if sector_faces[1] == face_index
        return (
            payload[CT_JUNCTION_PAYLOAD_FACE1_EMF],
            payload[CT_JUNCTION_PAYLOAD_FACE1_WEIGHT],
            payload[CT_JUNCTION_PAYLOAD_RESISTIVE_FACE1_EMF],
        )
    elseif sector_faces[2] == face_index
        return (
            payload[CT_JUNCTION_PAYLOAD_FACE2_EMF],
            payload[CT_JUNCTION_PAYLOAD_FACE2_WEIGHT],
            payload[CT_JUNCTION_PAYLOAD_RESISTIVE_FACE2_EMF],
        )
    end
    error("face=$face_index is not incident to sector faces=$sector_faces")
end

function _ct_generalized_junction_line_emf(
    topology::CTJunctionTopology, sector_payloads,
)
    nsectors = length(topology.edges)
    edge_to_sector = Dict(
        edge => index for (index, edge) in enumerate(topology.edges)
    )
    sector_faces = [
        _ct_topology_sector_faces(topology, edge) for edge in topology.edges
    ]
    cell_emf = [
        sector_payloads[index][CT_JUNCTION_PAYLOAD_CELL_EMF]
        for index in 1:nsectors
    ]
    resistive_cell_emf = [
        sector_payloads[index][CT_JUNCTION_PAYLOAD_RESISTIVE_CELL_EMF]
        for index in 1:nsectors
    ]
    sector_angle = [
        sector_payloads[index][CT_JUNCTION_PAYLOAD_SECTOR_ANGLE]
        for index in 1:nsectors
    ]
    fofc_scale = minimum(
        sector_payloads[index][CT_JUNCTION_PAYLOAD_FOFC_SCALE]
        for index in 1:nsectors
    )
    face_emf = zeros(eltype(cell_emf), nsectors)
    face_weight = zeros(eltype(cell_emf), nsectors)
    resistive_face_emf = zeros(eltype(cell_emf), nsectors)
    negative_sector = zeros(Int, nsectors)
    positive_sector = zeros(Int, nsectors)
    for (face_index, face) in enumerate(topology.faces)
        first_sector = edge_to_sector[face.first.edge]
        second_sector = edge_to_sector[face.second.edge]
        first_sample = _ct_payload_face_sample(
            sector_payloads[first_sector], sector_faces[first_sector],
            face_index,
        )
        second_sample = _ct_payload_face_sample(
            sector_payloads[second_sector], sector_faces[second_sector],
            face_index,
        )
        face_emf[face_index] = (first_sample[1] + second_sample[1]) / 2
        face_weight[face_index] = (first_sample[2] + second_sample[2]) / 2
        resistive_face_emf[face_index] =
            (first_sample[3] + second_sample[3]) / 2
        negative_sector[face_index] = edge_to_sector[face.negative_edge]
        positive_sector[face_index] = edge_to_sector[face.positive_edge]
    end
    ideal = ct_generalized_junction_emf(
        cell_emf, face_emf, face_weight,
        negative_sector, positive_sector, sector_faces, sector_angle,
    )
    resistive = ct_generalized_junction_emf(
        resistive_cell_emf, resistive_face_emf,
        fill(eltype(cell_emf)(0.5), nsectors),
        negative_sector, positive_sector, sector_faces, sector_angle,
    )
    angle_sum = sum(sector_angle)
    edge_length = sum(
        sector_angle[index] *
        sector_payloads[index][CT_JUNCTION_PAYLOAD_EDGE_LENGTH]
        for index in 1:nsectors
    ) / angle_sum
    return edge_length*fofc_scale*(ideal+resistive)
end

function ct_solve_generalized_junctions!(
    blocks, plan::CTJunctionPlan{T}, epoch; require_resistive::Bool=false,
) where {T}
    plan.generalized || return ct_sync_junction_edges!(blocks, plan)
    normalized = ct_assert_generalized_junction_payload_ready!(
        plan, epoch; require_resistive=require_resistive,
    )
    if isempty(plan.topologies)
        plan.solved_epoch = normalized
        return nothing
    end

    gpu_sync()
    for index in eachindex(plan.local_jobs)
        copyto!(
            plan.payload_host_buffers[index],
            plan.payload_device_buffers[index],
        )
    end
    Payload = NTuple{CT_JUNCTION_PAYLOAD_WIDTH,T}
    sector_payload = Dict{Tuple{Int,CTBlockEdge,Int},Payload}()
    for (job_index, job) in enumerate(plan.local_jobs)
        buffer = plan.payload_host_buffers[job_index]
        for (offset, global_index) in enumerate(
            job.canonical_i_s:job.canonical_i_e
        )
            sector_payload[(job.group_index, job.edge, global_index)] =
                _ct_payload_tuple(buffer, offset)
        end
    end

    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_member_buffers = Dict{
        Tuple{CTBlockEdge,CTBlockEdge,Int,Int,Int},Int
    }()
    for (index, job) in enumerate(plan.recv_jobs)
        buffer = plan.payload_recv_buffers[index]
        for (offset, global_index) in enumerate(
            job.global_i_s:job.global_i_e
        )
            payload = sector_payload[
                (job.group_index, job.local_edge, global_index)
            ]
            for channel in 1:CT_JUNCTION_PAYLOAD_WIDTH
                buffer[channel, offset] = payload[channel]
            end
        end
        local_member_buffers[(
            job.remote_edge, job.local_edge,
            job.global_i_s, job.global_i_e, job.tag,
        )] = index
    end

    requests = MPI.Request[]
    for (index, job) in enumerate(plan.send_jobs)
        if job.remote_rank != world_rank
            push!(requests, MPI.Irecv!(
                plan.payload_send_buffers[index], MPI.COMM_WORLD;
                source=job.remote_rank, tag=job.tag,
            ))
        else
            member_index = get(local_member_buffers, (
                job.local_edge, job.remote_edge,
                job.global_i_s, job.global_i_e, job.tag,
            ), 0)
            member_index > 0 || error(
                "missing local generalized CT junction payload for $job",
            )
            copyto!(
                plan.payload_send_buffers[index],
                plan.payload_recv_buffers[member_index],
            )
        end
    end
    for (index, job) in enumerate(plan.recv_jobs)
        if job.remote_rank != world_rank
            push!(requests, MPI.Isend(
                plan.payload_recv_buffers[index], MPI.COMM_WORLD;
                dest=job.remote_rank, tag=job.tag,
            ))
        end
    end
    isempty(requests) || MPI.Waitall(requests)
    for (index, job) in enumerate(plan.send_jobs)
        buffer = plan.payload_send_buffers[index]
        for (offset, global_index) in enumerate(
            job.global_i_s:job.global_i_e
        )
            sector_payload[
                (job.group_index, job.remote_edge, global_index)
            ] = _ct_payload_tuple(buffer, offset)
        end
    end

    solutions = Dict{Tuple{Int,Int},T}()
    for job in plan.send_jobs
        topology = plan.topologies[job.group_index]
        for global_index in job.global_i_s:job.global_i_e
            key = (job.group_index, global_index)
            haskey(solutions, key) && continue
            payloads = Payload[
                sector_payload[(job.group_index, edge, global_index)]
                for edge in topology.edges
            ]
            solutions[key] = _ct_generalized_junction_line_emf(
                topology, payloads,
            )
        end
    end
    for (index, job) in enumerate(plan.send_jobs)
        for (offset, global_index) in enumerate(
            job.global_i_s:job.global_i_e
        )
            plan.send_buffers[index][offset] =
                solutions[(job.group_index, global_index)]
        end
        _ct_unpack_junction_local_canonical!(
            blocks[job.local_edge.bid], job, plan.send_buffers[index],
        )
    end

    empty!(requests)
    for (index, job) in enumerate(plan.recv_jobs)
        if job.remote_rank != world_rank
            push!(requests, MPI.Irecv!(
                plan.recv_buffers[index], MPI.COMM_WORLD;
                source=job.remote_rank, tag=job.tag,
            ))
        end
    end
    for (index, job) in enumerate(plan.send_jobs)
        if job.remote_rank != world_rank
            push!(requests, MPI.Isend(
                plan.send_buffers[index], MPI.COMM_WORLD;
                dest=job.remote_rank, tag=job.tag,
            ))
        else
            member_index = local_member_buffers[(
                job.local_edge, job.remote_edge,
                job.global_i_s, job.global_i_e, job.tag,
            )]
            copyto!(plan.recv_buffers[member_index], plan.send_buffers[index])
        end
    end
    isempty(requests) || MPI.Waitall(requests)
    for (index, job) in enumerate(plan.recv_jobs)
        _ct_unpack_junction_local_canonical!(
            blocks[job.local_edge.bid], job, plan.recv_buffers[index],
        )
    end
    gpu_sync()
    plan.solved_epoch = normalized
    return nothing
end

function _ct_pack_junction_local_canonical!(buffer, b, job::CTJunctionJob)
    values = vec(Array(_ct_junction_local_view(b, job)))
    if job.local_orientation < 0
        reverse!(values)
        values .*= -one(FT)
    end
    copyto!(buffer, values)
    return buffer
end

function _ct_unpack_junction_local_canonical!(b, job::CTJunctionJob, buffer)
    if job.local_orientation < 0
        values = -reverse(Array(buffer))
    else
        values = buffer
    end
    _ct_copy_host_sheet_to_device!(_ct_junction_local_view(b, job), values)
    return nothing
end

function ct_sync_junction_edges!(
    blocks, plan::CTJunctionPlan;
    edge_merge::Symbol=CT_EDGE_MERGE_AVERAGE,
)
    _ct_validate_edge_merge_mode(edge_merge)
    if isempty(plan.recv_jobs) && isempty(plan.send_jobs)
        return nothing
    end
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_member_buffers = Dict{
        Tuple{CTBlockEdge,CTBlockEdge,Int,Int,Int},Int
    }()
    for (index, job) in enumerate(plan.recv_jobs)
        _ct_pack_junction_local_canonical!(
            plan.recv_buffers[index], blocks[job.local_edge.bid], job,
        )
        local_member_buffers[(
            job.remote_edge, job.local_edge,
            job.global_i_s, job.global_i_e, job.tag,
        )] = index
    end

    # Gather every non-owner contribution in the owner's canonical orientation.
    requests = MPI.Request[]
    for (index, job) in enumerate(plan.send_jobs)
        if job.remote_rank != world_rank
            push!(requests, MPI.Irecv!(
                plan.send_buffers[index], MPI.COMM_WORLD;
                source=job.remote_rank, tag=job.tag,
            ))
        else
            member_index = get(local_member_buffers, (
                job.local_edge, job.remote_edge,
                job.global_i_s, job.global_i_e, job.tag,
            ), 0)
            member_index > 0 || error(
                "missing local CT junction member buffer for $job",
            )
            copyto!(plan.send_buffers[index], plan.recv_buffers[member_index])
        end
    end
    for (index, job) in enumerate(plan.recv_jobs)
        if job.remote_rank != world_rank
            push!(requests, MPI.Isend(
                plan.recv_buffers[index], MPI.COMM_WORLD;
                dest=job.remote_rank, tag=job.tag,
            ))
        end
    end
    isempty(requests) || MPI.Waitall(requests)

    if edge_merge == CT_EDGE_MERGE_CANONICAL
        for (index, job) in enumerate(plan.send_jobs)
            owner_values = vec(Array(_ct_junction_local_view(
                blocks[job.local_edge.bid], job,
            )))
            if job.local_orientation < 0
                reverse!(owner_values)
                owner_values .*= -one(FT)
            end
            copyto!(plan.send_buffers[index], owner_values)
            _ct_unpack_junction_local_canonical!(
                blocks[job.local_edge.bid], job, plan.send_buffers[index],
            )
        end
    else
        sums = Dict{Tuple{CTBlockEdge,Int},FT}()
        counts = Dict{Tuple{CTBlockEdge,Int},Int}()
        for job in plan.send_jobs
            owner_values = vec(Array(_ct_junction_local_view(
                blocks[job.local_edge.bid], job,
            )))
            for (offset, global_index) in enumerate(job.global_i_s:job.global_i_e)
                key = (job.local_edge, global_index)
                if !haskey(sums, key)
                    sums[key] = owner_values[offset]
                    counts[key] = 1
                end
            end
        end
        for (index, job) in enumerate(plan.send_jobs)
            for (offset, global_index) in enumerate(job.global_i_s:job.global_i_e)
                key = (job.local_edge, global_index)
                sums[key] += plan.send_buffers[index][offset]
                counts[key] += 1
            end
        end
        for (index, job) in enumerate(plan.send_jobs)
            for (offset, global_index) in enumerate(job.global_i_s:job.global_i_e)
                key = (job.local_edge, global_index)
                plan.send_buffers[index][offset] = sums[key] / FT(counts[key])
            end
            _ct_unpack_junction_local_canonical!(
                blocks[job.local_edge.bid], job, plan.send_buffers[index],
            )
        end
    end

    # Broadcast the sum/count average back to every member, reusing both sets
    # of persistent plan buffers after the gather has completed.
    empty!(requests)
    for (index, job) in enumerate(plan.recv_jobs)
        if job.remote_rank != world_rank
            push!(requests, MPI.Irecv!(
                plan.recv_buffers[index], MPI.COMM_WORLD;
                source=job.remote_rank, tag=job.tag,
            ))
        end
    end
    for (index, job) in enumerate(plan.send_jobs)
        if job.remote_rank != world_rank
            push!(requests, MPI.Isend(
                plan.send_buffers[index], MPI.COMM_WORLD;
                dest=job.remote_rank, tag=job.tag,
            ))
        else
            member_index = local_member_buffers[(
                job.local_edge, job.remote_edge,
                job.global_i_s, job.global_i_e, job.tag,
            )]
            copyto!(plan.recv_buffers[member_index], plan.send_buffers[index])
        end
    end
    isempty(requests) || MPI.Waitall(requests)

    for (index, job) in enumerate(plan.recv_jobs)
        _ct_unpack_junction_local_canonical!(
            blocks[job.local_edge.bid], job, plan.recv_buffers[index],
        )
    end
    gpu_sync()
    return nothing
end

@inline function _ct_rank_sheet_views(b, direction, side)
    if direction == 1
        face_i = side < 0 ? NG + 1 : b.Nx + NG + 1
        edge_i = side < 0 ? 1 : b.Nx + 1
        face = @view b.Bx_face[face_i, NG+1:NG+b.Ny, NG+1:NG+b.Nz]
        u_edge = @view b.Ey_edge[edge_i, 1:b.Ny, 1:b.Nz+1]
        v_edge = @view b.Ez_edge[edge_i, 1:b.Ny+1, 1:b.Nz]
    elseif direction == 2
        face_j = side < 0 ? NG + 1 : b.Ny + NG + 1
        edge_j = side < 0 ? 1 : b.Ny + 1
        face = @view b.By_face[NG+1:NG+b.Nx, face_j, NG+1:NG+b.Nz]
        u_edge = @view b.Ex_edge[1:b.Nx, edge_j, 1:b.Nz+1]
        v_edge = @view b.Ez_edge[1:b.Nx+1, edge_j, 1:b.Nz]
    else
        face_k = side < 0 ? NG + 1 : b.Nz + NG + 1
        edge_k = side < 0 ? 1 : b.Nz + 1
        face = @view b.Bz_face[NG+1:NG+b.Nx, NG+1:NG+b.Ny, face_k]
        u_edge = @view b.Ex_edge[1:b.Nx, 1:b.Ny+1, edge_k]
        v_edge = @view b.Ey_edge[1:b.Nx+1, 1:b.Ny, edge_k]
    end
    return face, u_edge, v_edge
end

function ct_pack_rank_sheet(b, direction, side)
    face, u_edge, v_edge = _ct_rank_sheet_views(b, direction, side)
    buffer = Vector{FT}(undef, length(face) + length(u_edge) + length(v_edge))
    offset = 0
    for values in (face, u_edge, v_edge)
        host_values = Array(values)
        copyto!(buffer, offset + 1, vec(host_values), 1, length(host_values))
        offset += length(host_values)
    end
    return buffer
end

function ct_unpack_rank_sheet!(
    b, direction, side, buffer;
    local_buffer=nothing, sync_face_flux::Bool, sync_edges::Bool,
    edge_merge::Symbol=CT_EDGE_MERGE_AVERAGE,
    remote_is_canonical::Bool=false,
)
    sync_edges && _ct_validate_edge_merge_mode(edge_merge)
    face, u_edge, v_edge = _ct_rank_sheet_views(b, direction, side)
    nface = length(face)
    nuedge = length(u_edge)
    nvedge = length(v_edge)
    if sync_face_flux
        values = reshape(@view(buffer[1:nface]), size(face))
        _ct_copy_host_sheet_to_device!(face, values)
    end
    if sync_edges
        u_values = Array(reshape(
            @view(buffer[nface+1:nface+nuedge]), size(u_edge),
        ))
        v_values = Array(reshape(
            @view(buffer[nface+nuedge+1:nface+nuedge+nvedge]), size(v_edge),
        ))
        if edge_merge == CT_EDGE_MERGE_CANONICAL
            if remote_is_canonical
                _ct_copy_host_sheet_to_device!(u_edge, u_values)
                _ct_copy_host_sheet_to_device!(v_edge, v_values)
            end
        else
            if local_buffer === nothing
                local_u_values = Array(u_edge)
                local_v_values = Array(v_edge)
            else
                local_u_values = reshape(
                    @view(local_buffer[nface+1:nface+nuedge]), size(u_edge),
                )
                local_v_values = reshape(
                    @view(local_buffer[nface+nuedge+1:nface+nuedge+nvedge]),
                    size(v_edge),
                )
            end
            u_values .+= local_u_values
            u_values .*= FT(0.5)
            v_values .+= local_v_values
            v_values .*= FT(0.5)
            _ct_copy_host_sheet_to_device!(u_edge, u_values)
            _ct_copy_host_sheet_to_device!(v_edge, v_values)
        end
    end
    return nothing
end

function ct_sync_rank_sheets!(
    blocks, block_comms, Block_Nprocs;
    sync_face_flux::Bool=false,
    sync_edges::Bool=true,
    edge_merge::Symbol=CT_EDGE_MERGE_AVERAGE,
)
    sync_edges && _ct_validate_edge_merge_mode(edge_merge)
    passes = sync_edges ? 3 : 1
    for _ in 1:passes
        for bid in sort(collect(keys(blocks)))
            b = blocks[bid]
            comm = block_comms[bid]
            local_rank = MPI.Comm_rank(comm)
            layout = Block_Nprocs[bid + 1]
            for direction in 1:3
                if layout[direction] <= 1
                    continue
                end
                src, dst = MPI.Cart_shift(comm, direction - 1, 1)
                if src == MPI.PROC_NULL && dst == MPI.PROC_NULL
                    continue
                end
                low_buffer = sync_edges ?
                    ct_pack_rank_sheet(b, direction, -1) : nothing
                high_buffer = ct_pack_rank_sheet(b, direction, 1)
                recv_buffer = similar(high_buffer)
                MPI.Sendrecv!(
                    high_buffer, recv_buffer, comm; dest=dst, source=src,
                )
                if src != MPI.PROC_NULL
                    ct_unpack_rank_sheet!(
                        b, direction, -1, recv_buffer;
                        local_buffer=low_buffer,
                        sync_face_flux=sync_face_flux, sync_edges=sync_edges,
                        edge_merge=edge_merge,
                        remote_is_canonical=src < local_rank,
                    )
                end
                if sync_edges
                    MPI.Sendrecv!(
                        low_buffer, recv_buffer, comm; dest=src, source=dst,
                    )
                    if dst != MPI.PROC_NULL
                        ct_unpack_rank_sheet!(
                            b, direction, 1, recv_buffer;
                            local_buffer=high_buffer,
                            sync_face_flux=false, sync_edges=true,
                            edge_merge=edge_merge,
                            remote_is_canonical=dst < local_rank,
                        )
                    end
                end
            end
        end
        gpu_sync()
    end
    return nothing
end

@inline function ct_interface_halo_buffer_length(u_len::Integer, v_len::Integer)
    nuface = (u_len + 1)*NG*v_len
    nvface = u_len*NG*(v_len + 1)
    # The first six component families are tangential face-B ghost slabs.
    # Normal face-B has one additional staggered scalar family: its normal
    # face range contains the shared interface sheet plus one extra active
    # face, so it cannot use either cell-like family above.
    nnormal = u_len*NG*v_len
    return 3 * (nuface + nvface) + nnormal
end

@inline function _ct_interface_source_layers(fid, ncell)
    return isodd(fid) ? (NG+1:2NG) : (ncell+1:ncell+NG)
end

@inline function _ct_interface_source_normal_face_layers(fid, ncell)
    # Normal face fields contain the canonical interface sheet at the first
    # active index.  Ghost layers therefore start one face later than the
    # cell-like/tangential source slabs.
    return isodd(fid) ? (NG+2:2NG+1) : (ncell+1:ncell+NG)
end

@inline function _ct_interface_destination_normal_face_layers(fid, ncell)
    return isodd(fid) ? (1:NG) : (ncell+NG+2:ncell+2NG+1)
end

function _ct_pack_interface_normal_face!(b, ex, normal_layers)
    if ex.fid == 1 || ex.fid == 2
        raw = Array(@view b.Bx_face[
            normal_layers, ex.u_s+NG:ex.u_e+NG, ex.v_s+NG:ex.v_e+NG,
        ])
        return permutedims(raw, (2, 1, 3))
    elseif ex.fid == 3 || ex.fid == 4
        return Array(@view b.By_face[
            ex.u_s+NG:ex.u_e+NG, normal_layers, ex.v_s+NG:ex.v_e+NG,
        ])
    else
        raw = Array(@view b.Bz_face[
            ex.u_s+NG:ex.u_e+NG, ex.v_s+NG:ex.v_e+NG, normal_layers,
        ])
        return permutedims(raw, (1, 3, 2))
    end
end

function _ct_constrain_interface_face_components(
    components, face_flux, area, normal_x, normal_y, normal_z,
)
    corrected = ct_constrain_face_flux.(
        components[1], components[2], components[3],
        area, normal_x, normal_y, normal_z, face_flux,
    )
    return ntuple(3) do component
        getindex.(corrected, component)
    end
end

function ct_pack_interface_halo!(buffer, b, ex::CTInterfaceExchange)
    normal_layers = ex.fid <= 2 ?
        _ct_interface_source_layers(ex.fid, b.Nx) :
        ex.fid <= 4 ? _ct_interface_source_layers(ex.fid, b.Ny) :
        _ct_interface_source_layers(ex.fid, b.Nz)
    if ex.fid == 1 || ex.fid == 2
        i_face = normal_layers
        j_face = ex.u_s+NG:ex.u_e+NG+1
        k_face = ex.v_s+NG:ex.v_e+NG+1
        u_components = ntuple(3) do component
            variable = QBX + component - 1
            raw = FT(0.5) .* (
                Array(@view b.Q[i_face, first(j_face)-1:last(j_face)-1,
                                first(k_face):last(k_face)-1, variable]) .+
                Array(@view b.Q[i_face, first(j_face):last(j_face),
                                first(k_face):last(k_face)-1, variable])
            )
            permutedims(raw, (2, 1, 3))
        end
        v_components = ntuple(3) do component
            variable = QBX + component - 1
            raw = FT(0.5) .* (
                Array(@view b.Q[i_face, first(j_face):last(j_face)-1,
                                first(k_face)-1:last(k_face)-1, variable]) .+
                Array(@view b.Q[i_face, first(j_face):last(j_face)-1,
                                first(k_face):last(k_face), variable])
            )
            permutedims(raw, (2, 1, 3))
        end
        u_components = _ct_constrain_interface_face_components(
            u_components,
            permutedims(Array(@view b.By_face[
                i_face, j_face, first(k_face):last(k_face)-1,
            ]), (2, 1, 3)),
            permutedims(Array(@view b.Areaj[
                i_face, j_face, first(k_face):last(k_face)-1,
            ]), (2, 1, 3)),
            permutedims(Array(@view b.nxj[
                i_face, j_face, first(k_face):last(k_face)-1,
            ]), (2, 1, 3)),
            permutedims(Array(@view b.nyj[
                i_face, j_face, first(k_face):last(k_face)-1,
            ]), (2, 1, 3)),
            permutedims(Array(@view b.nzj[
                i_face, j_face, first(k_face):last(k_face)-1,
            ]), (2, 1, 3)),
        )
        v_components = _ct_constrain_interface_face_components(
            v_components,
            permutedims(Array(@view b.Bz_face[
                i_face, first(j_face):last(j_face)-1, k_face,
            ]), (2, 1, 3)),
            permutedims(Array(@view b.Areak[
                i_face, first(j_face):last(j_face)-1, k_face,
            ]), (2, 1, 3)),
            permutedims(Array(@view b.nxk[
                i_face, first(j_face):last(j_face)-1, k_face,
            ]), (2, 1, 3)),
            permutedims(Array(@view b.nyk[
                i_face, first(j_face):last(j_face)-1, k_face,
            ]), (2, 1, 3)),
            permutedims(Array(@view b.nzk[
                i_face, first(j_face):last(j_face)-1, k_face,
            ]), (2, 1, 3)),
        )
    elseif ex.fid == 3 || ex.fid == 4
        i_face = ex.u_s+NG:ex.u_e+NG+1
        k_face = ex.v_s+NG:ex.v_e+NG+1
        u_components = ntuple(3) do component
            variable = QBX + component - 1
            FT(0.5) .* (
                Array(@view b.Q[first(i_face)-1:last(i_face)-1,
                                normal_layers, first(k_face):last(k_face)-1,
                                variable]) .+
                Array(@view b.Q[first(i_face):last(i_face),
                                normal_layers, first(k_face):last(k_face)-1,
                                variable])
            )
        end
        v_components = ntuple(3) do component
            variable = QBX + component - 1
            FT(0.5) .* (
                Array(@view b.Q[first(i_face):last(i_face)-1,
                                normal_layers, first(k_face)-1:last(k_face)-1,
                                variable]) .+
                Array(@view b.Q[first(i_face):last(i_face)-1,
                                normal_layers, first(k_face):last(k_face),
                                variable])
            )
        end
        u_components = _ct_constrain_interface_face_components(
            u_components,
            Array(@view b.Bx_face[
                i_face, normal_layers, first(k_face):last(k_face)-1,
            ]),
            Array(@view b.Areai[
                i_face, normal_layers, first(k_face):last(k_face)-1,
            ]),
            Array(@view b.nxi[
                i_face, normal_layers, first(k_face):last(k_face)-1,
            ]),
            Array(@view b.nyi[
                i_face, normal_layers, first(k_face):last(k_face)-1,
            ]),
            Array(@view b.nzi[
                i_face, normal_layers, first(k_face):last(k_face)-1,
            ]),
        )
        v_components = _ct_constrain_interface_face_components(
            v_components,
            Array(@view b.Bz_face[
                first(i_face):last(i_face)-1, normal_layers, k_face,
            ]),
            Array(@view b.Areak[
                first(i_face):last(i_face)-1, normal_layers, k_face,
            ]),
            Array(@view b.nxk[
                first(i_face):last(i_face)-1, normal_layers, k_face,
            ]),
            Array(@view b.nyk[
                first(i_face):last(i_face)-1, normal_layers, k_face,
            ]),
            Array(@view b.nzk[
                first(i_face):last(i_face)-1, normal_layers, k_face,
            ]),
        )
    else
        i_face = ex.u_s+NG:ex.u_e+NG+1
        j_face = ex.v_s+NG:ex.v_e+NG+1
        u_components = ntuple(3) do component
            variable = QBX + component - 1
            raw = FT(0.5) .* (
                Array(@view b.Q[first(i_face)-1:last(i_face)-1,
                                first(j_face):last(j_face)-1, normal_layers,
                                variable]) .+
                Array(@view b.Q[first(i_face):last(i_face),
                                first(j_face):last(j_face)-1, normal_layers,
                                variable])
            )
            permutedims(raw, (1, 3, 2))
        end
        v_components = ntuple(3) do component
            variable = QBX + component - 1
            raw = FT(0.5) .* (
                Array(@view b.Q[first(i_face):last(i_face)-1,
                                first(j_face)-1:last(j_face)-1, normal_layers,
                                variable]) .+
                Array(@view b.Q[first(i_face):last(i_face)-1,
                                first(j_face):last(j_face), normal_layers,
                                variable])
            )
            permutedims(raw, (1, 3, 2))
        end
        u_components = _ct_constrain_interface_face_components(
            u_components,
            permutedims(Array(@view b.Bx_face[
                i_face, first(j_face):last(j_face)-1, normal_layers,
            ]), (1, 3, 2)),
            permutedims(Array(@view b.Areai[
                i_face, first(j_face):last(j_face)-1, normal_layers,
            ]), (1, 3, 2)),
            permutedims(Array(@view b.nxi[
                i_face, first(j_face):last(j_face)-1, normal_layers,
            ]), (1, 3, 2)),
            permutedims(Array(@view b.nyi[
                i_face, first(j_face):last(j_face)-1, normal_layers,
            ]), (1, 3, 2)),
            permutedims(Array(@view b.nzi[
                i_face, first(j_face):last(j_face)-1, normal_layers,
            ]), (1, 3, 2)),
        )
        v_components = _ct_constrain_interface_face_components(
            v_components,
            permutedims(Array(@view b.By_face[
                first(i_face):last(i_face)-1, j_face, normal_layers,
            ]), (1, 3, 2)),
            permutedims(Array(@view b.Areaj[
                first(i_face):last(i_face)-1, j_face, normal_layers,
            ]), (1, 3, 2)),
            permutedims(Array(@view b.nxj[
                first(i_face):last(i_face)-1, j_face, normal_layers,
            ]), (1, 3, 2)),
            permutedims(Array(@view b.nyj[
                first(i_face):last(i_face)-1, j_face, normal_layers,
            ]), (1, 3, 2)),
            permutedims(Array(@view b.nzj[
                first(i_face):last(i_face)-1, j_face, normal_layers,
            ]), (1, 3, 2)),
        )
    end

    offset = 0
    for values in (u_components..., v_components...)
        copyto!(buffer, offset + 1, vec(values), 1, length(values))
        offset += length(values)
    end
    normal_ncell = ex.fid <= 2 ? b.Nx : ex.fid <= 4 ? b.Ny : b.Nz
    normal_layers = _ct_interface_source_normal_face_layers(ex.fid, normal_ncell)
    normal_values = _ct_pack_interface_normal_face!(b, ex, normal_layers)
    copyto!(buffer, offset + 1, vec(normal_values), 1, length(normal_values))
    return buffer
end

@inline function _ct_interface_destination_layers(fid, ncell)
    return isodd(fid) ? (1:NG) : (ncell+NG+1:ncell+2NG)
end

function ct_unpack_interface_halo!(b, ex::CTInterfaceExchange, buffer)
    u_len = ex.u_e - ex.u_s + 1
    v_len = ex.v_e - ex.v_s + 1
    source_u_len, source_v_len = if ex.transform !== nothing &&
        (structured_face_transform_code(structured_inverse_face_transform(ex.transform)) & 1) != 0
        (v_len, u_len)
    else
        (u_len, v_len)
    end
    nuface = (source_u_len + 1)*NG*source_v_len
    nvface = source_u_len*NG*(source_v_len + 1)
    u_sources = ntuple(3) do component
        offset = (component - 1) * nuface
        reshape(
            @view(buffer[offset+1:offset+nuface]), source_u_len + 1, NG, source_v_len,
        )
    end
    v_base = 3nuface
    v_sources = ntuple(3) do component
        offset = v_base + (component - 1) * nvface
        reshape(
            @view(buffer[offset+1:offset+nvface]), source_u_len, NG, source_v_len + 1,
        )
    end
    normal_base = 3*nuface + 3*nvface
    normal_nface = source_u_len*NG*source_v_len
    normal_source = reshape(
        @view(buffer[normal_base+1:normal_base+normal_nface]),
        source_u_len, NG, source_v_len,
    )
    same_side = isodd(ex.fid) == isodd(ex.nb_fid)
    normal_order = same_side ? (NG:-1:1) : (1:NG)
    if ex.reverse_tan
        u_components = ntuple(component ->
            Array(u_sources[component][:, normal_order, end:-1:1]), 3,
        )
        v_components = ntuple(component ->
            Array(v_sources[component][:, normal_order, end:-1:1]), 3,
        )
    else
        u_components = ntuple(component ->
            Array(u_sources[component][:, normal_order, :]), 3,
        )
        v_components = ntuple(component ->
            Array(v_sources[component][:, normal_order, :]), 3,
        )
    end
    normal_values = ex.reverse_tan ?
        Array(normal_source[:, normal_order, end:-1:1]) :
        Array(normal_source[:, normal_order, :])
    if ex.transform !== nothing
        receive_transform = structured_inverse_face_transform(ex.transform)
        destination = structured_face_frame(receive_transform.destination_face)
        source = structured_face_frame(receive_transform.source_face)
        map_u = receive_transform.source_for_destination[destination.u_axis]
        map_v = receive_transform.source_for_destination[destination.v_axis]
        reoriented_u = ntuple(component -> begin
            src_u = u_components[component]
            src_v = v_components[component]
            values = Array{eltype(src_u)}(undef, u_len + 1, NG, v_len)
            su_nodes = abs(map_u) == source.u_axis ? size(src_u, 1) : size(src_v, 1)
            sv_nodes = abs(map_u) == source.u_axis ? size(src_u, 3) : size(src_v, 3)
            for layer in 1:NG, j in 1:v_len, i in 1:(u_len + 1)
                if abs(map_u) == source.u_axis
                    si = _ct_reverse_index(i, su_nodes, map_u < 0)
                    sj = _ct_reverse_index(j, sv_nodes, map_v < 0)
                    values[i, layer, j] = src_u[si, layer, sj]
                else
                    si = _ct_reverse_index(j, size(src_v, 1), map_v < 0)
                    sj = _ct_reverse_index(i, size(src_v, 3), map_u < 0)
                    values[i, layer, j] = src_v[si, layer, sj]
                end
            end
            values
        end, 3)
        reoriented_v = ntuple(component -> begin
            src_u = u_components[component]
            src_v = v_components[component]
            values = Array{eltype(src_v)}(undef, u_len, NG, v_len + 1)
            for layer in 1:NG, j in 1:(v_len + 1), i in 1:u_len
                if abs(map_v) == source.u_axis
                    si = _ct_reverse_index(j, size(src_u, 1), map_v < 0)
                    sj = _ct_reverse_index(i, size(src_u, 3), map_u < 0)
                    values[i, layer, j] = src_u[si, layer, sj]
                else
                    si = _ct_reverse_index(i, size(src_v, 1), map_u < 0)
                    sj = _ct_reverse_index(j, size(src_v, 3), map_v < 0)
                    values[i, layer, j] = src_v[si, layer, sj]
                end
            end
            values
        end, 3)
        u_components, v_components = reoriented_u, reoriented_v
        receive_transform = structured_inverse_face_transform(ex.transform)
        destination = structured_face_frame(receive_transform.destination_face)
        source = structured_face_frame(receive_transform.source_face)
        map_u = receive_transform.source_for_destination[destination.u_axis]
        map_v = receive_transform.source_for_destination[destination.v_axis]
        reoriented_normal = Array{eltype(normal_values)}(undef, u_len, NG, v_len)
        for layer in 1:NG, j in 1:v_len, i in 1:u_len
            if abs(map_u) == source.u_axis
                si = _ct_reverse_index(i, size(normal_values, 1), map_u < 0)
                sj = _ct_reverse_index(j, size(normal_values, 3), map_v < 0)
            else
                si = _ct_reverse_index(j, size(normal_values, 1), map_v < 0)
                sj = _ct_reverse_index(i, size(normal_values, 3), map_u < 0)
            end
            reoriented_normal[i, layer, j] = normal_values[si, layer, sj]
        end
        normal_values = reoriented_normal
    end
    normal_values .*= FT(ex.transform === nothing ?
        ct_interface_face_flux_sign(ex.fid, ex.nb_fid) :
        ct_interface_face_flux_sign(ex.fid, ex.nb_fid, ex.transform))

    destination_layers = ex.fid <= 2 ?
        _ct_interface_destination_layers(ex.fid, b.Nx) :
        ex.fid <= 4 ? _ct_interface_destination_layers(ex.fid, b.Ny) :
        _ct_interface_destination_layers(ex.fid, b.Nz)
    if ex.fid == 1 || ex.fid == 2
        u_components = ntuple(component ->
            permutedims(u_components[component], (2, 1, 3)), 3,
        )
        v_components = ntuple(component ->
            permutedims(v_components[component], (2, 1, 3)), 3,
        )
        u_destination = @view b.By_face[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
        ]
        v_destination = @view b.Bz_face[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
        ]
        u_area = Array(@view b.Areaj[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
        ])
        u_normal_x = Array(@view b.nxj[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
        ])
        u_normal_y = Array(@view b.nyj[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
        ])
        u_normal_z = Array(@view b.nzj[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
        ])
        v_area = Array(@view b.Areak[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
        ])
        v_normal_x = Array(@view b.nxk[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
        ])
        v_normal_y = Array(@view b.nyk[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
        ])
        v_normal_z = Array(@view b.nzk[
            destination_layers,
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
        ])
        u_values = u_area .* (
            u_components[1] .* u_normal_x .+
            u_components[2] .* u_normal_y .+
            u_components[3] .* u_normal_z
        )
        v_values = v_area .* (
            v_components[1] .* v_normal_x .+
            v_components[2] .* v_normal_y .+
            v_components[3] .* v_normal_z
        )
        _ct_copy_host_sheet_to_device!(u_destination, u_values)
        _ct_copy_host_sheet_to_device!(v_destination, v_values)
    elseif ex.fid == 3 || ex.fid == 4
        u_destination = @view b.Bx_face[
            ex.u_s+NG:ex.u_e+NG+1,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG,
        ]
        v_destination = @view b.Bz_face[
            ex.u_s+NG:ex.u_e+NG,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG+1,
        ]
        u_area = Array(@view b.Areai[
            ex.u_s+NG:ex.u_e+NG+1,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG,
        ])
        u_normal_x = Array(@view b.nxi[
            ex.u_s+NG:ex.u_e+NG+1,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG,
        ])
        u_normal_y = Array(@view b.nyi[
            ex.u_s+NG:ex.u_e+NG+1,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG,
        ])
        u_normal_z = Array(@view b.nzi[
            ex.u_s+NG:ex.u_e+NG+1,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG,
        ])
        v_area = Array(@view b.Areak[
            ex.u_s+NG:ex.u_e+NG,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG+1,
        ])
        v_normal_x = Array(@view b.nxk[
            ex.u_s+NG:ex.u_e+NG,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG+1,
        ])
        v_normal_y = Array(@view b.nyk[
            ex.u_s+NG:ex.u_e+NG,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG+1,
        ])
        v_normal_z = Array(@view b.nzk[
            ex.u_s+NG:ex.u_e+NG,
            destination_layers,
            ex.v_s+NG:ex.v_e+NG+1,
        ])
        u_values = u_area .* (
            u_components[1] .* u_normal_x .+
            u_components[2] .* u_normal_y .+
            u_components[3] .* u_normal_z
        )
        v_values = v_area .* (
            v_components[1] .* v_normal_x .+
            v_components[2] .* v_normal_y .+
            v_components[3] .* v_normal_z
        )
        _ct_copy_host_sheet_to_device!(u_destination, u_values)
        _ct_copy_host_sheet_to_device!(v_destination, v_values)
    else
        u_destination = @view b.Bx_face[
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
            destination_layers,
        ]
        v_destination = @view b.By_face[
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
            destination_layers,
        ]
        u_cartesian = ntuple(component ->
            permutedims(u_components[component], (1, 3, 2)), 3,
        )
        v_cartesian = ntuple(component ->
            permutedims(v_components[component], (1, 3, 2)), 3,
        )
        u_area = Array(@view b.Areai[
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
            destination_layers,
        ])
        u_normal_x = Array(@view b.nxi[
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
            destination_layers,
        ])
        u_normal_y = Array(@view b.nyi[
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
            destination_layers,
        ])
        u_normal_z = Array(@view b.nzi[
            ex.u_s+NG:ex.u_e+NG+1,
            ex.v_s+NG:ex.v_e+NG,
            destination_layers,
        ])
        v_area = Array(@view b.Areaj[
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
            destination_layers,
        ])
        v_normal_x = Array(@view b.nxj[
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
            destination_layers,
        ])
        v_normal_y = Array(@view b.nyj[
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
            destination_layers,
        ])
        v_normal_z = Array(@view b.nzj[
            ex.u_s+NG:ex.u_e+NG,
            ex.v_s+NG:ex.v_e+NG+1,
            destination_layers,
        ])
        u_values = u_area .* (
            u_cartesian[1] .* u_normal_x .+
            u_cartesian[2] .* u_normal_y .+
            u_cartesian[3] .* u_normal_z
        )
        v_values = v_area .* (
            v_cartesian[1] .* v_normal_x .+
            v_cartesian[2] .* v_normal_y .+
            v_cartesian[3] .* v_normal_z
        )
        _ct_copy_host_sheet_to_device!(
            u_destination, u_values,
        )
        _ct_copy_host_sheet_to_device!(
            v_destination, v_values,
        )
    end
    normal_ncell = ex.fid <= 2 ? b.Nx : ex.fid <= 4 ? b.Ny : b.Nz
    normal_destination_layers = _ct_interface_destination_normal_face_layers(
        ex.fid, normal_ncell,
    )
    if ex.fid == 1 || ex.fid == 2
        normal_destination = @view b.Bx_face[
            normal_destination_layers,
            ex.u_s+NG:ex.u_e+NG, ex.v_s+NG:ex.v_e+NG,
        ]
        normal_values = permutedims(normal_values, (2, 1, 3))
    elseif ex.fid == 3 || ex.fid == 4
        normal_destination = @view b.By_face[
            ex.u_s+NG:ex.u_e+NG, normal_destination_layers,
            ex.v_s+NG:ex.v_e+NG,
        ]
    else
        normal_destination = @view b.Bz_face[
            ex.u_s+NG:ex.u_e+NG, ex.v_s+NG:ex.v_e+NG,
            normal_destination_layers,
        ]
        normal_values = permutedims(normal_values, (1, 3, 2))
    end
    _ct_copy_host_sheet_to_device!(normal_destination, normal_values)
    return nothing
end

function ct_sync_interface_face_halos!(blocks, plan::CTSyncPlan)
    if isempty(plan.exchanges)
        return nothing
    end
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    for (index, ex) in enumerate(plan.exchanges)
        ct_pack_interface_halo!(plan.halo_send_buffers[index], blocks[ex.bid], ex)
    end

    requests = MPI.Request[]
    for (index, ex) in enumerate(plan.exchanges)
        if ex.nb_rank != world_rank
            push!(requests, MPI.Irecv!(
                plan.halo_recv_buffers[index], MPI.COMM_WORLD;
                source=ex.nb_rank, tag=ex.tag,
            ))
        end
    end
    for (index, ex) in enumerate(plan.exchanges)
        if ex.nb_rank != world_rank
            push!(requests, MPI.Isend(
                plan.halo_send_buffers[index], MPI.COMM_WORLD;
                dest=ex.nb_rank, tag=ex.tag,
            ))
        end
    end
    if !isempty(requests)
        MPI.Waitall(requests)
    end

    for (index, ex) in enumerate(plan.exchanges)
        source = ex.nb_rank == world_rank ?
            plan.halo_send_buffers[plan.local_peer[index]] :
            plan.halo_recv_buffers[index]
        ct_unpack_interface_halo!(blocks[ex.bid], ex, source)
    end
    gpu_sync()
    return nothing
end

function _ct_sync_rank_face_halo_field!(
    array, direction, ncell, comm; normal_face::Bool=false,
)
    active_high = ncell + NG + (normal_face ? 1 : 0)
    active_high_send = active_high - NG + 1
    high_ghost_start = active_high + 1
    high_ghost_end = high_ghost_start + NG - 1
    src, dst = MPI.Cart_shift(comm, direction - 1, 1)
    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        send_high = Array(selectdim(
            array, direction, active_high_send:active_high,
        ))
        recv_low = similar(send_high)
        MPI.Sendrecv!(send_high, recv_low, comm; dest=dst, source=src)
        if src != MPI.PROC_NULL
            _ct_copy_host_sheet_to_device!(
                selectdim(array, direction, 1:NG), recv_low,
            )
        end
    end

    src, dst = MPI.Cart_shift(comm, direction - 1, -1)
    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        send_low = Array(selectdim(array, direction, NG+1:2NG))
        recv_high = similar(send_low)
        MPI.Sendrecv!(send_low, recv_high, comm; dest=dst, source=src)
        if src != MPI.PROC_NULL
            _ct_copy_host_sheet_to_device!(
                selectdim(array, direction, high_ghost_start:high_ghost_end),
                recv_high,
            )
        end
    end
    return nothing
end

function ct_sync_rank_face_halos!(blocks, block_comms, Block_Nprocs)
    for bid in sort(collect(keys(blocks)))
        b = blocks[bid]
        comm = block_comms[bid]
        layout = Block_Nprocs[bid + 1]
        cell_counts = (b.Nx, b.Ny, b.Nz)
        face_fields = (b.Bx_face, b.By_face, b.Bz_face)
        for direction in 1:3
            if layout[direction] <= 1
                continue
            end
            for component in 1:3
                _ct_sync_rank_face_halo_field!(
                    face_fields[component], direction,
                    cell_counts[direction], comm;
                    normal_face=component == direction,
                )
            end
        end
    end
    gpu_sync()
    return nothing
end

function _ct_sync_host_metric_face!(array, direction, ncell, comm)
    src, dst = MPI.Cart_shift(comm, direction - 1, 1)
    send_high = Array(selectdim(array, direction, NG + ncell + 1))
    recv_low = similar(send_high)
    MPI.Sendrecv!(send_high, recv_low, comm; dest=dst, source=src)
    if src != MPI.PROC_NULL
        copyto!(selectdim(array, direction, NG + 1), recv_low)
    end
    return nothing
end

function _ct_sync_host_metric_face_vector!(metric_group, direction, ncell, comm)
    area, normal_x, normal_y, normal_z = metric_group
    src, dst = MPI.Cart_shift(comm, direction - 1, 1)
    low_index = NG + 1
    high_index = NG + ncell + 1

    low_area = selectdim(area, direction, low_index)
    low_nx = selectdim(normal_x, direction, low_index)
    low_ny = selectdim(normal_y, direction, low_index)
    low_nz = selectdim(normal_z, direction, low_index)
    high_area = selectdim(area, direction, high_index)
    high_nx = selectdim(normal_x, direction, high_index)
    high_ny = selectdim(normal_y, direction, high_index)
    high_nz = selectdim(normal_z, direction, high_index)

    send_high = zeros(eltype(area), size(high_area)..., 3)
    send_low = zeros(eltype(area), size(low_area)..., 3)
    recv_from_low = similar(send_high)
    recv_from_high = similar(send_low)
    for I in CartesianIndices(high_area)
        send_high[I, 1] = high_area[I] * high_nx[I]
        send_high[I, 2] = high_area[I] * high_ny[I]
        send_high[I, 3] = high_area[I] * high_nz[I]
    end
    for I in CartesianIndices(low_area)
        send_low[I, 1] = low_area[I] * low_nx[I]
        send_low[I, 2] = low_area[I] * low_ny[I]
        send_low[I, 3] = low_area[I] * low_nz[I]
    end

    # Exchange both sheets before changing either one.  The low rank receives
    # the high rank's low sheet in the second exchange; the high rank receives
    # the low rank's high sheet in the first exchange.
    MPI.Sendrecv!(send_high, recv_from_low, comm; dest=dst, source=src)
    MPI.Sendrecv!(send_low, recv_from_high, comm; dest=src, source=dst)

    function average_face!(area_face, nx_face, ny_face, nz_face, received)
        for I in CartesianIndices(area_face)
            sx = (area_face[I] * nx_face[I] + received[I, 1]) / 2
            sy = (area_face[I] * ny_face[I] + received[I, 2]) / 2
            sz = (area_face[I] * nz_face[I] + received[I, 3]) / 2
            magnitude = sqrt(sx^2 + sy^2 + sz^2)
            magnitude > eps(eltype(area_face)) || throw(DomainError(
                magnitude, "degenerate rank-interface metric",
            ))
            area_face[I] = magnitude
            nx_face[I] = sx / magnitude
            ny_face[I] = sy / magnitude
            nz_face[I] = sz / magnitude
        end
    end

    src != MPI.PROC_NULL && average_face!(
        low_area, low_nx, low_ny, low_nz, recv_from_low,
    )
    dst != MPI.PROC_NULL && average_face!(
        high_area, high_nx, high_ny, high_nz, recv_from_high,
    )
    return nothing
end

function _ct_sync_host_metric_halo!(
    array, direction, ncell, staggered, comm,
)
    src, dst = MPI.Cart_shift(comm, direction - 1, 1)
    send_high = Array(selectdim(array, direction, ncell+1:ncell+NG))
    recv_low = similar(send_high)
    MPI.Sendrecv!(send_high, recv_low, comm; dest=dst, source=src)
    if src != MPI.PROC_NULL
        copyto!(selectdim(array, direction, 1:NG), recv_low)
    end

    src, dst = MPI.Cart_shift(comm, direction - 1, -1)
    low_source = staggered ? (NG+2:2NG+1) : (NG+1:2NG)
    high_destination = staggered ?
        (ncell+NG+2:ncell+2NG+1) :
        (ncell+NG+1:ncell+2NG)
    send_low = Array(selectdim(array, direction, low_source))
    recv_high = similar(send_low)
    MPI.Sendrecv!(send_low, recv_high, comm; dest=dst, source=src)
    if src != MPI.PROC_NULL
        copyto!(selectdim(array, direction, high_destination), recv_high)
    end
    return nothing
end

function ct_sync_rank_metrics!(
    temp_metrics_h, blocks, block_comms, Block_Nprocs,
)
    for bid in sort(collect(keys(temp_metrics_h)))
        b = blocks[bid]
        layout = Block_Nprocs[bid + 1]
        if all(layout .== 1)
            continue
        end
        comm = block_comms[bid]
        values = temp_metrics_h[bid]
        metric_groups = (
            (values[2], values[3], values[4], values[5]),
            (values[6], values[7], values[8], values[9]),
            (values[10], values[11], values[12], values[13]),
        )
        cell_counts = (b.Nx, b.Ny, b.Nz)

        for direction in 1:3
            if layout[direction] <= 1
                continue
            end
            _ct_sync_host_metric_face_vector!(
                metric_groups[direction], direction,
                cell_counts[direction], comm,
            )
            for component in 1:3, array in metric_groups[component]
                _ct_sync_host_metric_halo!(
                    array, direction, cell_counts[direction],
                    component == direction, comm,
                )
            end
            _ct_sync_host_metric_halo!(
                values[14], direction, cell_counts[direction], false, comm,
            )
        end
    end
    return nothing
end

function ct_sync_rank_metric_halos_only!(
    temp_metrics_h, blocks, block_comms, Block_Nprocs,
)
    for bid in sort(collect(keys(temp_metrics_h)))
        block = blocks[bid]
        layout = Block_Nprocs[bid + 1]
        all(layout .== 1) && continue
        communicator = block_comms[bid]
        values = temp_metrics_h[bid]
        metric_groups = (
            (values[2], values[3], values[4], values[5]),
            (values[6], values[7], values[8], values[9]),
            (values[10], values[11], values[12], values[13]),
        )
        cell_counts = (block.Nx, block.Ny, block.Nz)
        for direction in 1:3
            layout[direction] <= 1 && continue
            for component in 1:3, array in metric_groups[component]
                _ct_sync_host_metric_halo!(
                    array, direction, cell_counts[direction],
                    component == direction, communicator,
                )
            end
            _ct_sync_host_metric_halo!(
                values[14], direction, cell_counts[direction], false,
                communicator,
            )
        end
    end
    return nothing
end

function ct_metric_closure_stats(temp_metrics_h, blocks)
    sum_squared = 0.0
    max_norm = 0.0
    ncell_total = 0
    for (bid, values) in temp_metrics_h
        b = blocks[bid]
        Areai, nxi, nyi, nzi = values[2:5]
        Areaj, nxj, nyj, nzj = values[6:9]
        Areak, nxk, nyk, nzk = values[10:13]
        Vol = values[14]
        for k in 1:b.Nz, j in 1:b.Ny, i in 1:b.Nx
            ii = i + NG
            jj = j + NG
            kk = k + NG
            cx = Vol[ii,jj,kk] * (
                Areai[ii+1,jj,kk]*nxi[ii+1,jj,kk] - Areai[ii,jj,kk]*nxi[ii,jj,kk] +
                Areaj[ii,jj+1,kk]*nxj[ii,jj+1,kk] - Areaj[ii,jj,kk]*nxj[ii,jj,kk] +
                Areak[ii,jj,kk+1]*nxk[ii,jj,kk+1] - Areak[ii,jj,kk]*nxk[ii,jj,kk]
            )
            cy = Vol[ii,jj,kk] * (
                Areai[ii+1,jj,kk]*nyi[ii+1,jj,kk] - Areai[ii,jj,kk]*nyi[ii,jj,kk] +
                Areaj[ii,jj+1,kk]*nyj[ii,jj+1,kk] - Areaj[ii,jj,kk]*nyj[ii,jj,kk] +
                Areak[ii,jj,kk+1]*nyk[ii,jj,kk+1] - Areak[ii,jj,kk]*nyk[ii,jj,kk]
            )
            cz = Vol[ii,jj,kk] * (
                Areai[ii+1,jj,kk]*nzi[ii+1,jj,kk] - Areai[ii,jj,kk]*nzi[ii,jj,kk] +
                Areaj[ii,jj+1,kk]*nzj[ii,jj+1,kk] - Areaj[ii,jj,kk]*nzj[ii,jj,kk] +
                Areak[ii,jj,kk+1]*nzk[ii,jj,kk+1] - Areak[ii,jj,kk]*nzk[ii,jj,kk]
            )
            norm_squared = cx*cx + cy*cy + cz*cz
            sum_squared += norm_squared
            max_norm = max(max_norm, sqrt(norm_squared))
            ncell_total += 1
        end
    end
    return sqrt(sum_squared / max(1, ncell_total)), max_norm
end

function ct_metric_closure_relative_stats(temp_metrics_h, blocks)
    sum_squared = 0.0
    max_relative = 0.0
    ncell_total = 0
    for (bid, values) in temp_metrics_h
        b = blocks[bid]
        Areai, nxi, nyi, nzi = values[2:5]
        Areaj, nxj, nyj, nzj = values[6:9]
        Areak, nxk, nyk, nzk = values[10:13]
        Vol = values[14]
        for k in 1:b.Nz, j in 1:b.Ny, i in 1:b.Nx
            ii = i + NG
            jj = j + NG
            kk = k + NG
            sx = (
                Areai[ii+1,jj,kk]*nxi[ii+1,jj,kk] - Areai[ii,jj,kk]*nxi[ii,jj,kk] +
                Areaj[ii,jj+1,kk]*nxj[ii,jj+1,kk] - Areaj[ii,jj,kk]*nxj[ii,jj,kk] +
                Areak[ii,jj,kk+1]*nxk[ii,jj,kk+1] - Areak[ii,jj,kk]*nxk[ii,jj,kk]
            )
            sy = (
                Areai[ii+1,jj,kk]*nyi[ii+1,jj,kk] - Areai[ii,jj,kk]*nyi[ii,jj,kk] +
                Areaj[ii,jj+1,kk]*nyj[ii,jj+1,kk] - Areaj[ii,jj,kk]*nyj[ii,jj,kk] +
                Areak[ii,jj,kk+1]*nyk[ii,jj,kk+1] - Areak[ii,jj,kk]*nyk[ii,jj,kk]
            )
            sz = (
                Areai[ii+1,jj,kk]*nzi[ii+1,jj,kk] - Areai[ii,jj,kk]*nzi[ii,jj,kk] +
                Areaj[ii,jj+1,kk]*nzj[ii,jj+1,kk] - Areaj[ii,jj,kk]*nzj[ii,jj,kk] +
                Areak[ii,jj,kk+1]*nzk[ii,jj,kk+1] - Areak[ii,jj,kk]*nzk[ii,jj,kk]
            )
            residual = sqrt(sx^2 + sy^2 + sz^2) * Vol[ii,jj,kk]
            scale = Vol[ii,jj,kk] * (
                Areai[ii+1,jj,kk] + Areai[ii,jj,kk] +
                Areaj[ii,jj+1,kk] + Areaj[ii,jj,kk] +
                Areak[ii,jj,kk+1] + Areak[ii,jj,kk]
            )
            relative = residual / max(scale, eps(Float64))
            sum_squared += relative^2
            max_relative = max(max_relative, relative)
            ncell_total += 1
        end
    end
    return sqrt(sum_squared / max(1, ncell_total)), max_relative
end
