import Metis
import MPI

const UNSTRUCT_EDGE_INTERNAL = UInt8(0)
const UNSTRUCT_EDGE_PERIODIC = UInt8(1)

"""Cell-dual graph and the weights used by the mesh partitioner."""
struct UnstructDualGraph
    ncell::Int
    xadj::Vector{Int32}
    adjncy::Vector{Int32}
    adjwgt::Vector{Int32}
    edge_left::Vector{Int}
    edge_right::Vector{Int}
    edge_kind::Vector{UInt8}
    edge_weight::Vector{Int}
    compute_weight::Vector{Float64}
    memory_weight::Vector{Float64}
end

"""Global partition quality metrics. MPI ghosts exclude physical BC ghosts."""
struct UnstructPartitionQuality
    compute_load::Vector{Float64}
    memory_load::Vector{Float64}
    owned_cells::Vector{Int}
    mpi_ghost_cells::Vector{Int}
    neighbor_counts::Vector{Int}
    edge_cut::Int
    weighted_edge_cut::Int64
    compute_imbalance::Float64
    memory_imbalance::Float64
    max_ghost_ratio::Float64
end

@inline function _partition_periodic_source_map(block::UnstructBlock)
    ghosts = Array(block.periodic_ghost_cells)
    sources = Array(block.periodic_source_cells)
    length(ghosts) == length(sources) || error(
        "periodic ghost/source list length mismatch",
    )
    return Dict{Int,Int}(ghost => source for (ghost, source) in zip(ghosts, sources))
end

function _partition_periodic_face_map(block::UnstructBlock)
    primary = Array(block.periodic_face_primary)
    partner = Array(block.periodic_face_partner)
    length(primary) == length(partner) || error(
        "periodic primary/partner face list length mismatch",
    )
    face_partner = Dict{Int,Int}()
    face_sign = Dict{Int,Int8}()
    for (primary_face, partner_face) in zip(primary, partner)
        face_partner[primary_face] = partner_face
        face_partner[partner_face] = primary_face
        face_sign[primary_face] = Int8(1)
        face_sign[partner_face] = Int8(-1)
    end
    return face_partner, face_sign
end

function _default_unstruct_compute_weights(
    face_count, neighbor_count, face_node_count;
    order::Int, viscous::Bool, resistive::Bool, mhd::Bool, ct::Bool,
)
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2"))
    weights = Vector{Float64}(undef, length(face_count))
    for cell in eachindex(weights)
        value = 64.0 + 14.0*face_count[cell]
        order == 2 && (value += 20.0*neighbor_count[cell])
        viscous && (value += 18.0*face_count[cell] + 12.0*neighbor_count[cell])
        resistive && (value += 12.0*face_count[cell] + 8.0*neighbor_count[cell])
        mhd && (value += 22.0*face_count[cell])
        ct && (value += 8.0*face_node_count[cell])
        weights[cell] = value
    end
    return weights
end

function _default_unstruct_memory_weights(
    face_count, neighbor_count, face_node_count;
    nprim::Int, ncons::Int, float_bytes::Int,
)
    nprim > 0 || throw(ArgumentError("nprim must be positive"))
    ncons > 0 || throw(ArgumentError("ncons must be positive"))
    float_bytes in (4, 8) || throw(ArgumentError("float_bytes must be 4 or 8"))
    weights = Vector{Float64}(undef, length(face_count))
    persistent_cell_values = 2*nprim + 2*ncons + 3*nprim + 1
    for cell in eachindex(weights)
        cell_bytes = float_bytes*persistent_cell_values
        face_bytes = face_count[cell]*(float_bytes*(2*ncons + 7) + 3*sizeof(Int))
        neighbor_bytes = neighbor_count[cell]*(float_bytes*4 + sizeof(Int))
        topology_bytes = face_node_count[cell]*sizeof(Int)
        weights[cell] = cell_bytes + face_bytes + neighbor_bytes + topology_bytes
    end
    return weights
end

function _validate_partition_weights(name::AbstractString, weights, ncell::Int)
    length(weights) == ncell || throw(ArgumentError(
        "$name must contain one entry per cell; expected $ncell, got $(length(weights))",
    ))
    all(weight -> isfinite(weight) && weight > 0, weights) || throw(ArgumentError(
        "$name entries must be finite and positive",
    ))
    return Float64.(weights)
end

"""
    build_unstruct_dual_graph(block; kwargs...)

Build the cell-dual graph. Internal faces and explicit periodic pairs become
graph edges. The default weights account for face count, LSQ work, enabled
physics, and an estimated per-cell memory footprint.
"""
function build_unstruct_dual_graph(
    block::UnstructBlock;
    order::Int=1,
    viscous_enabled::Bool=false,
    resistive_enabled::Bool=false,
    mhd_enabled::Bool=false,
    ct_enabled::Bool=false,
    nprim::Int=(isdefined(@__MODULE__, :Nprim) ? Int(getfield(@__MODULE__, :Nprim)) : 6),
    ncons::Int=(isdefined(@__MODULE__, :Ncons) ? Int(getfield(@__MODULE__, :Ncons)) : 5),
    float_bytes::Int=(isdefined(@__MODULE__, :FT) ? sizeof(getfield(@__MODULE__, :FT)) : 8),
    compute_weights=nothing,
    memory_weights=nothing,
    periodic_edge_multiplier::Int=16,
)
    block.ncell > 0 || throw(ArgumentError("cannot partition an empty mesh"))
    periodic_edge_multiplier >= 1 || throw(ArgumentError(
        "periodic_edge_multiplier must be at least one",
    ))
    face_left = Array(block.face_L)
    face_right = Array(block.face_R)
    face_node_offset = Array(block.face_node_offset)
    periodic_source = _partition_periodic_source_map(block)

    face_count = zeros(Int, block.ncell)
    face_node_count = zeros(Int, block.ncell)
    raw_edges = Tuple{Int,Int,Int,UInt8}[]
    for face in 1:block.nface
        left = face_left[face]
        right = face_right[face]
        node_count = face_node_offset[face + 1] - face_node_offset[face]
        if left <= block.ncell
            face_count[left] += 1
            face_node_count[left] += node_count
        end
        if right <= block.ncell
            face_count[right] += 1
            face_node_count[right] += node_count
            left == right || push!(raw_edges, (
                min(left, right), max(left, right), 1, UNSTRUCT_EDGE_INTERNAL,
            ))
        elseif haskey(periodic_source, right)
            source = periodic_source[right]
            left == source || push!(raw_edges, (
                min(left, source), max(left, source),
                periodic_edge_multiplier, UNSTRUCT_EDGE_PERIODIC,
            ))
        end
    end

    sort!(raw_edges; by=edge -> (edge[1], edge[2], edge[4]))
    edge_left = Int[]
    edge_right = Int[]
    edge_weight = Int[]
    edge_kind = UInt8[]
    for edge in raw_edges
        if !isempty(edge_left) && edge_left[end] == edge[1] && edge_right[end] == edge[2]
            edge_weight[end] += edge[3]
            edge_kind[end] = max(edge_kind[end], edge[4])
        else
            push!(edge_left, edge[1])
            push!(edge_right, edge[2])
            push!(edge_weight, edge[3])
            push!(edge_kind, edge[4])
        end
    end

    adjacency = [Tuple{Int,Int}[] for _ in 1:block.ncell]
    for edge_index in eachindex(edge_left)
        left = edge_left[edge_index]
        right = edge_right[edge_index]
        weight = edge_weight[edge_index]
        push!(adjacency[left], (right, weight))
        push!(adjacency[right], (left, weight))
    end
    neighbor_count = length.(adjacency)
    xadj = Vector{Int32}(undef, block.ncell + 1)
    xadj[1] = Int32(1)
    adjncy = Int32[]
    adjwgt = Int32[]
    for cell in 1:block.ncell
        sort!(adjacency[cell]; by=first)
        for (neighbor, weight) in adjacency[cell]
            push!(adjncy, Int32(neighbor))
            push!(adjwgt, Int32(weight))
        end
        xadj[cell + 1] = Int32(length(adjncy) + 1)
    end

    default_compute = _default_unstruct_compute_weights(
        face_count, neighbor_count, face_node_count;
        order=order, viscous=viscous_enabled, resistive=resistive_enabled,
        mhd=mhd_enabled, ct=ct_enabled,
    )
    default_memory = _default_unstruct_memory_weights(
        face_count, neighbor_count, face_node_count;
        nprim=nprim, ncons=ncons, float_bytes=float_bytes,
    )
    final_compute = compute_weights === nothing ? default_compute :
        _validate_partition_weights("compute_weights", compute_weights, block.ncell)
    final_memory = memory_weights === nothing ? default_memory :
        _validate_partition_weights("memory_weights", memory_weights, block.ncell)
    return UnstructDualGraph(
        block.ncell, xadj, adjncy, adjwgt,
        edge_left, edge_right, edge_kind, edge_weight, final_compute, final_memory,
    )
end

function _scaled_metis_weights(weights::Vector{Float64})
    total = sum(weights)
    isfinite(total) && total > 0 || throw(ArgumentError("invalid METIS weights"))
    target_total = min(Float64(typemax(Int32) ÷ 8), 1.0e9)
    scale = target_total/total
    return Int32[max(1, round(Int, weight*scale)) for weight in weights]
end

function _unstruct_graph_is_connected(graph::UnstructDualGraph)
    graph.ncell <= 1 && return true
    visited = falses(graph.ncell)
    queue = Vector{Int}(undef, graph.ncell)
    queue[1] = 1
    visited[1] = true
    first_pending = 1
    last_pending = 1
    while first_pending <= last_pending
        cell = queue[first_pending]
        first_pending += 1
        for position in graph.xadj[cell]:(graph.xadj[cell + 1] - 1)
            neighbor = Int(graph.adjncy[position])
            if !visited[neighbor]
                last_pending += 1
                queue[last_pending] = neighbor
                visited[neighbor] = true
            end
        end
    end
    return all(visited)
end

function _normalized_partition_targets(targets, nparts::Int, name::AbstractString)
    values = targets === nothing ? fill(1/nparts, nparts) : Float64.(targets)
    length(values) == nparts || throw(ArgumentError(
        "$name must contain $nparts entries",
    ))
    all(value -> isfinite(value) && value > 0, values) || throw(ArgumentError(
        "$name entries must be finite and positive",
    ))
    return values ./ sum(values)
end

"""Partition a dual graph with two METIS constraints: compute and memory."""
function partition_unstruct_graph(
    graph::UnstructDualGraph, nparts::Int;
    compute_imbalance::Float64=1.05,
    memory_imbalance::Float64=1.05,
    compute_targets=nothing,
    memory_targets=nothing,
    seed::Int=17,
)
    1 <= nparts <= graph.ncell || throw(ArgumentError(
        "nparts must be between 1 and $(graph.ncell), got $nparts",
    ))
    compute_imbalance >= 1 || throw(ArgumentError("compute_imbalance must be >= 1"))
    memory_imbalance >= 1 || throw(ArgumentError("memory_imbalance must be >= 1"))
    nparts == 1 && return ones(Int, graph.ncell)

    library = Metis.LibMetis
    index_type = library.idx_t
    real_type = library.real_t
    nconstraints = index_type(2)
    vertex_count = index_type(graph.ncell)
    part_count = index_type(nparts)
    xadj = index_type.(graph.xadj)
    adjncy = index_type.(graph.adjncy)
    adjwgt = index_type.(graph.adjwgt)
    compute_weight = _scaled_metis_weights(graph.compute_weight)
    memory_weight = _scaled_metis_weights(graph.memory_weight)
    vertex_weights = Vector{index_type}(undef, 2*graph.ncell)
    for cell in 1:graph.ncell
        vertex_weights[2*cell - 1] = compute_weight[cell]
        vertex_weights[2*cell] = memory_weight[cell]
    end

    compute_target = _normalized_partition_targets(
        compute_targets, nparts, "compute_targets",
    )
    memory_target = _normalized_partition_targets(
        memory_targets, nparts, "memory_targets",
    )
    target_weights = Vector{real_type}(undef, 2*nparts)
    for partition in 1:nparts
        target_weights[2*partition - 1] = real_type(compute_target[partition])
        target_weights[2*partition] = real_type(memory_target[partition])
    end
    imbalance = real_type[compute_imbalance, memory_imbalance]
    options = Vector{index_type}(undef, Int(library.METIS_NOPTIONS))
    return_code = library.METIS_SetDefaultOptions(options)
    return_code == library.METIS_OK || error(
        "METIS_SetDefaultOptions failed with code $return_code",
    )
    options[Int(library.METIS_OPTION_NUMBERING) + 1] = index_type(1)
    options[Int(library.METIS_OPTION_SEED) + 1] = index_type(seed)
    options[Int(library.METIS_OPTION_CONTIG) + 1] =
        index_type(_unstruct_graph_is_connected(graph) ? 1 : 0)
    edge_cut = index_type[0]
    assignment = Vector{index_type}(undef, graph.ncell)
    return_code = library.METIS_PartGraphKway(
        Ref(vertex_count), Ref(nconstraints), xadj, adjncy,
        vertex_weights, C_NULL, adjwgt, Ref(part_count),
        target_weights, imbalance, options, edge_cut, assignment,
    )
    return_code == library.METIS_OK || error(
        "METIS_PartGraphKway failed with code $return_code",
    )
    minimum(assignment) == 0 && (assignment .+= index_type(1))
    all(partition -> 1 <= partition <= nparts, assignment) || error(
        "METIS returned partition IDs outside 1:$nparts",
    )
    return Int.(assignment)
end

function evaluate_unstruct_partition(
    graph::UnstructDualGraph, assignment::AbstractVector{<:Integer}, nparts::Int,
)
    length(assignment) == graph.ncell || throw(ArgumentError(
        "assignment length does not match graph cell count",
    ))
    all(partition -> 1 <= partition <= nparts, assignment) || throw(ArgumentError(
        "assignment entries must be between 1 and $nparts",
    ))
    compute_load = zeros(Float64, nparts)
    memory_load = zeros(Float64, nparts)
    owned_cells = zeros(Int, nparts)
    ghost_sets = [Set{Int}() for _ in 1:nparts]
    neighbor_sets = [Set{Int}() for _ in 1:nparts]
    for cell in 1:graph.ncell
        partition = assignment[cell]
        compute_load[partition] += graph.compute_weight[cell]
        memory_load[partition] += graph.memory_weight[cell]
        owned_cells[partition] += 1
    end
    edge_cut = 0
    weighted_edge_cut = Int64(0)
    for edge_index in eachindex(graph.edge_left)
        left = graph.edge_left[edge_index]
        right = graph.edge_right[edge_index]
        left_partition = assignment[left]
        right_partition = assignment[right]
        left_partition == right_partition && continue
        edge_cut += 1
        weighted_edge_cut += graph.edge_weight[edge_index]
        push!(ghost_sets[left_partition], right)
        push!(ghost_sets[right_partition], left)
        push!(neighbor_sets[left_partition], right_partition)
        push!(neighbor_sets[right_partition], left_partition)
    end
    mpi_ghost_cells = length.(ghost_sets)
    neighbor_counts = length.(neighbor_sets)
    compute_mean = sum(compute_load)/nparts
    memory_mean = sum(memory_load)/nparts
    compute_imbalance = maximum(compute_load)/compute_mean
    memory_imbalance = maximum(memory_load)/memory_mean
    ghost_ratios = mpi_ghost_cells ./ max.(owned_cells, 1)
    return UnstructPartitionQuality(
        compute_load, memory_load, owned_cells, mpi_ghost_cells,
        neighbor_counts, edge_cut, weighted_edge_cut,
        compute_imbalance, memory_imbalance, maximum(ghost_ratios),
    )
end

function print_unstruct_partition_quality(
    quality::UnstructPartitionQuality; io::IO=stdout,
)
    println(io, "UNSTRUCT_PARTITION_QUALITY " *
        "compute_imbalance=$(round(quality.compute_imbalance, digits=6)) " *
        "memory_imbalance=$(round(quality.memory_imbalance, digits=6)) " *
        "edge_cut=$(quality.edge_cut) weighted_edge_cut=$(quality.weighted_edge_cut) " *
        "max_ghost_ratio=$(round(quality.max_ghost_ratio, digits=6))")
    for partition in eachindex(quality.owned_cells)
        ratio = quality.mpi_ghost_cells[partition]/max(quality.owned_cells[partition], 1)
        println(io, "UNSTRUCT_PARTITION rank=$(partition - 1) " *
            "owned=$(quality.owned_cells[partition]) " *
            "ghost=$(quality.mpi_ghost_cells[partition]) " *
            "ghost_ratio=$(round(ratio, digits=6)) " *
            "neighbors=$(quality.neighbor_counts[partition])")
    end
    return quality
end

function _serial_global_ids(block::UnstructBlock)
    metadata = block.partition
    if metadata === nothing
        return (
            cell=Int64.(1:block.ncell),
            face=Int64.(1:block.nface),
            node=Int64.(1:block.nnode),
        )
    end
    return (
        cell=metadata.cell_global_ids[1:block.ncell],
        face=metadata.face_global_ids,
        node=metadata.node_global_ids,
    )
end

function _partition_face_nodes(offsets, nodes, face::Int, reverse_order::Bool)
    result = collect(nodes[offsets[face]:(offsets[face + 1] - 1)])
    reverse_order && reverse!(result)
    return result
end

function _partition_source_rows(source, source_indices)
    ndims(source) == 1 && return source[source_indices]
    ndims(source) == 2 && return source[source_indices, :]
    ndims(source) == 3 && return source[source_indices, :, :]
    error("unsupported partitioned array rank $(ndims(source))")
end

"""
    localize_unstruct_partition(block, graph, assignment, rank; ct_enabled=false)

Create one rank-local `UnstructBlock`. Owned cells are first, followed by
unique MPI source ghosts and then one physical ghost per retained boundary
face. A remote cell shared by several interface faces is stored only once.
"""
function localize_unstruct_partition(
    block::UnstructBlock,
    graph::UnstructDualGraph,
    assignment::AbstractVector{<:Integer},
    rank::Int;
    ct_enabled::Bool=false,
)
    block.ct === nothing || throw(ArgumentError(
        "partitioning must happen before CT topology initialization",
    ))
    nparts = maximum(assignment)
    0 <= rank < nparts || throw(ArgumentError("invalid rank $rank for $nparts partitions"))
    target = rank + 1
    global_ids = _serial_global_ids(block)
    length(unique(global_ids.cell)) == block.ncell || error("cell global IDs are not unique")
    owned_source = findall(==(target), assignment)
    isempty(owned_source) && error("partition $target owns no cells")
    sort!(owned_source; by=cell -> global_ids.cell[cell])
    owned_local = Dict(
        source => local_index for (local_index, source) in enumerate(owned_source)
    )

    remote_by_neighbor = Dict{Int,Set{Int}}()
    for source in owned_source
        for position in graph.xadj[source]:(graph.xadj[source + 1] - 1)
            remote = Int(graph.adjncy[position])
            remote_partition = assignment[remote]
            remote_partition == target && continue
            push!(get!(remote_by_neighbor, remote_partition - 1, Set{Int}()), remote)
        end
    end
    neighbors = sort!(collect(keys(remote_by_neighbor)))
    mpi_source = Int[]
    mpi_offsets_by_neighbor = Dict{Int,UnitRange{Int}}()
    for neighbor in neighbors
        first_index = length(mpi_source) + 1
        sources = sort!(collect(remote_by_neighbor[neighbor]); by=cell -> global_ids.cell[cell])
        append!(mpi_source, sources)
        mpi_offsets_by_neighbor[neighbor] = first_index:length(mpi_source)
    end
    mpi_local = Dict(
        source => length(owned_source) + index for (index, source) in enumerate(mpi_source)
    )

    face_left_source = Array(block.face_L)
    face_right_source = Array(block.face_R)
    face_bc_source = Array(block.face_bc_id)
    face_bc_params_source = Array(block.face_bc_params)
    face_node_offset_source = Array(block.face_node_offset)
    face_node_list_source = Array(block.face_node_list)
    face_area_source = Array(block.face_area)
    face_nx_source = Array(block.face_nx)
    face_ny_source = Array(block.face_ny)
    face_nz_source = Array(block.face_nz)
    face_cx_source = Array(block.face_cx)
    face_cy_source = Array(block.face_cy)
    face_cz_source = Array(block.face_cz)
    periodic_source = _partition_periodic_source_map(block)
    periodic_partner, periodic_sign = _partition_periodic_face_map(block)

    retained_faces = Int[]
    flipped_faces = BitVector()
    physical_source = Int[]
    physical_local_by_face = Dict{Int,Int}()
    for face in 1:block.nface
        left = face_left_source[face]
        right = face_right_source[face]
        if right <= block.ncell
            left_owned = haskey(owned_local, left)
            right_owned = haskey(owned_local, right)
            (left_owned || right_owned) || continue
            push!(retained_faces, face)
            push!(flipped_faces, !left_owned)
        elseif haskey(owned_local, left)
            push!(retained_faces, face)
            push!(flipped_faces, false)
            push!(physical_source, right)
            physical_local_by_face[face] =
                length(owned_source) + length(mpi_source) + length(physical_source)
        end
    end
    sort_permutation = sortperm(retained_faces; by=face -> global_ids.face[face])
    retained_faces = retained_faces[sort_permutation]
    flipped_faces = flipped_faces[sort_permutation]

    ncell = length(owned_source)
    mpi_ghost_count = length(mpi_source)
    nghost = mpi_ghost_count + length(physical_source)
    ncell_tot = ncell + nghost
    source_cell_rows = vcat(owned_source, mpi_source, physical_source)
    source_q = Array(block.Q)
    source_u = Array(block.U)
    source_un = Array(block.Un)
    source_grad = Array(block.grad)
    source_limiter = Array(block.reconstruction_limiter)
    source_dt = Array(block.dt_arr)
    source_cell_vol = Array(block.cell_vol)
    source_cell_cx = Array(block.cell_cx)
    source_cell_cy = Array(block.cell_cy)
    source_cell_cz = Array(block.cell_cz)

    face_left = Int[]
    face_right = Int[]
    face_area = FT[]
    face_nx = FT[]
    face_ny = FT[]
    face_nz = FT[]
    face_cx = FT[]
    face_cy = FT[]
    face_cz = FT[]
    face_bc_id = Int[]
    face_bc_params = Matrix{FT}(
        undef, length(retained_faces), size(face_bc_params_source, 2),
    )
    retained_node_source = Vector{Vector{Int}}()
    face_local_by_source = Dict{Int,Int}()
    interface_by_neighbor = Dict{Int,Vector{Tuple{Int64,Int,Int8}}}()
    for (local_face, source_face) in enumerate(retained_faces)
        face_local_by_source[source_face] = local_face
        source_left = face_left_source[source_face]
        source_right = face_right_source[source_face]
        flipped = flipped_faces[local_face]
        if source_right <= block.ncell
            local_owned_source = flipped ? source_right : source_left
            remote_source = flipped ? source_left : source_right
            push!(face_left, owned_local[local_owned_source])
            if haskey(owned_local, remote_source)
                push!(face_right, owned_local[remote_source])
            else
                push!(face_right, mpi_local[remote_source])
                neighbor = assignment[remote_source] - 1
                sign = flipped ? Int8(-1) : Int8(1)
                push!(get!(interface_by_neighbor, neighbor,
                    Tuple{Int64,Int,Int8}[]),
                    (global_ids.face[source_face], local_face, sign))
            end
        else
            push!(face_left, owned_local[source_left])
            push!(face_right, physical_local_by_face[source_face])
        end
        orientation = flipped ? -one(FT) : one(FT)
        push!(face_area, face_area_source[source_face])
        push!(face_nx, orientation*face_nx_source[source_face])
        push!(face_ny, orientation*face_ny_source[source_face])
        push!(face_nz, orientation*face_nz_source[source_face])
        push!(face_cx, face_cx_source[source_face])
        push!(face_cy, face_cy_source[source_face])
        push!(face_cz, face_cz_source[source_face])
        push!(face_bc_id, face_bc_source[source_face])
        face_bc_params[local_face, :] .= face_bc_params_source[source_face, :]
        push!(retained_node_source,
            _partition_face_nodes(
                face_node_offset_source, face_node_list_source,
                source_face, flipped,
            ))
    end

    node_source = sort!(unique(reduce(vcat, retained_node_source; init=Int[]));
        by=node -> global_ids.node[node])
    node_local = Dict(
        source => local_index for (local_index, source) in enumerate(node_source)
    )
    face_node_offset = Vector{Int}(undef, length(retained_faces) + 1)
    face_node_offset[1] = 1
    face_node_list = Int[]
    for local_face in eachindex(retained_faces)
        append!(face_node_list, node_local[node] for node in retained_node_source[local_face])
        face_node_offset[local_face + 1] = length(face_node_list) + 1
    end

    periodic_ghost_cells = Int[]
    periodic_source_cells = Int[]
    for source_face in retained_faces
        source_right = face_right_source[source_face]
        haskey(periodic_source, source_right) || continue
        source_cell = periodic_source[source_right]
        local_periodic_source = get(owned_local, source_cell, get(mpi_local, source_cell, 0))
        local_periodic_source > 0 || error(
            "periodic source cell $source_cell is absent from rank $rank",
        )
        push!(periodic_ghost_cells, physical_local_by_face[source_face])
        push!(periodic_source_cells, local_periodic_source)
    end
    periodic_face_primary = Int[]
    periodic_face_partner = Int[]
    for (source_face, paired_face) in periodic_partner
        periodic_sign[source_face] == Int8(1) || continue
        local_primary = get(face_local_by_source, source_face, 0)
        local_partner = get(face_local_by_source, paired_face, 0)
        if local_primary > 0 && local_partner > 0
            push!(periodic_face_primary, local_primary)
            push!(periodic_face_partner, local_partner)
        elseif ct_enabled && (local_primary > 0 || local_partner > 0)
            error(
                "CT partitioning requires periodic face pair $source_face/$paired_face " *
                "to remain on one rank; increase periodic_edge_multiplier",
            )
        end
    end

    cell_face_offset, cell_face_list, cell_face_sign = build_cell_face_csr(
        face_left, face_right, ncell_tot,
    )
    cell_vol = _partition_source_rows(source_cell_vol, source_cell_rows)
    cell_cx = _partition_source_rows(source_cell_cx, source_cell_rows)
    cell_cy = _partition_source_rows(source_cell_cy, source_cell_rows)
    cell_cz = _partition_source_rows(source_cell_cz, source_cell_rows)
    cn_offset, cn_list, cn_dx, cn_dy, cn_dz, cn_w = build_cell_neighbor_csr(
        face_left, face_right, cell_cx, cell_cy, cell_cz, ncell_tot,
    )

    mpi_send_offsets = Int[1]
    mpi_recv_offsets = Int[1]
    mpi_send_list = Int[]
    mpi_recv_list = Int[]
    mpi_send_global_ids = Int64[]
    mpi_recv_global_ids = Int64[]
    mpi_face_offsets = Int[1]
    mpi_face_list = Int[]
    mpi_face_keys = Int64[]
    mpi_face_signs = Int8[]
    for neighbor in neighbors
        remote_sources = mpi_source[mpi_offsets_by_neighbor[neighbor]]
        local_send_sources = Set{Int}()
        for remote_source in remote_sources
            for position in graph.xadj[remote_source]:(graph.xadj[remote_source + 1] - 1)
                candidate = Int(graph.adjncy[position])
                assignment[candidate] == target && push!(local_send_sources, candidate)
            end
        end
        sorted_send = sort!(collect(local_send_sources); by=cell -> global_ids.cell[cell])
        append!(mpi_send_list, owned_local[source] for source in sorted_send)
        append!(mpi_recv_list, mpi_local[source] for source in remote_sources)
        append!(mpi_send_global_ids, global_ids.cell[source] for source in sorted_send)
        append!(mpi_recv_global_ids, global_ids.cell[source] for source in remote_sources)
        push!(mpi_send_offsets, length(mpi_send_list) + 1)
        push!(mpi_recv_offsets, length(mpi_recv_list) + 1)
        interface = sort!(get(interface_by_neighbor, neighbor,
            Tuple{Int64,Int,Int8}[]); by=first)
        append!(mpi_face_keys, first(entry) for entry in interface)
        append!(mpi_face_list, entry[2] for entry in interface)
        append!(mpi_face_signs, entry[3] for entry in interface)
        push!(mpi_face_offsets, length(mpi_face_list) + 1)
    end

    cell_global_ids = vcat(
        global_ids.cell[owned_source], global_ids.cell[mpi_source],
        zeros(Int64, length(physical_source)),
    )
    metadata = UnstructPartitionMetadata(
        rank, nparts, cell_global_ids, global_ids.face[retained_faces],
        global_ids.node[node_source], mpi_ghost_count,
        mpi_send_global_ids, mpi_recv_global_ids,
        mpi_face_offsets, mpi_face_list, mpi_face_keys, mpi_face_signs,
    )
    buffer_width = 3*Nprim
    send_buffer_length = length(mpi_send_list)*buffer_width
    recv_buffer_length = length(mpi_recv_list)*buffer_width
    nthreads_cell = 256
    nthreads_face = 256
    nface = length(retained_faces)
    nnode = length(node_source)
    return UnstructBlock(
        block.id, ncell, nface, nnode, nghost, ncell_tot,
        GPUArray(_partition_source_rows(source_q, source_cell_rows)),
        GPUArray(_partition_source_rows(source_u, source_cell_rows)),
        GPUArray(_partition_source_rows(source_un, source_cell_rows)),
        GPUArray(source_dt[owned_source]),
        GPUArray(_partition_source_rows(source_grad, source_cell_rows)),
        GPUArray(_partition_source_rows(source_limiter, source_cell_rows)),
        gpu_zeros(FT, nface, Ncons), gpu_zeros(FT, nface, Ncons),
        GPUArray(cell_vol), GPUArray(cell_cx), GPUArray(cell_cy), GPUArray(cell_cz),
        GPUArray(face_area), GPUArray(face_nx), GPUArray(face_ny), GPUArray(face_nz),
        GPUArray(face_cx), GPUArray(face_cy), GPUArray(face_cz),
        GPUArray(face_left), GPUArray(face_right),
        GPUArray(cell_face_offset), GPUArray(cell_face_list), GPUArray(cell_face_sign),
        GPUArray(cn_offset), GPUArray(cn_list),
        GPUArray(cn_dx), GPUArray(cn_dy), GPUArray(cn_dz), GPUArray(cn_w),
        GPUArray(face_bc_id), GPUArray(face_bc_params),
        GPUArray(periodic_ghost_cells), GPUArray(periodic_source_cells),
        GPUArray(periodic_face_primary), GPUArray(periodic_face_partner),
        GPUArray(Array(block.node_x)[node_source]),
        GPUArray(Array(block.node_y)[node_source]),
        GPUArray(Array(block.node_z)[node_source]),
        GPUArray(face_node_offset), GPUArray(face_node_list), length(face_node_list),
        nothing, nothing, metadata,
        neighbors, mpi_send_offsets, mpi_recv_offsets,
        mpi_send_list, mpi_recv_list,
        zeros(FT, send_buffer_length), zeros(FT, recv_buffer_length),
        gpu_zeros(FT, send_buffer_length), gpu_zeros(FT, recv_buffer_length),
        cld(ncell, nthreads_cell), cld(nface, nthreads_face),
        nthreads_cell, nthreads_face,
    )
end

function validate_unstruct_halo_plan!(block::UnstructBlock, comm)
    metadata = block.partition
    metadata === nothing && return block
    isempty(block.mpi_neighbors) && return block
    received_global_ids = Vector{Vector{Int64}}(undef, length(block.mpi_neighbors))
    requests = MPI.Request[]
    for neighbor_index in eachindex(block.mpi_neighbors)
        neighbor = block.mpi_neighbors[neighbor_index]
        send_range = range(
            block.mpi_send_offsets[neighbor_index];
            stop=block.mpi_send_offsets[neighbor_index + 1] - 1,
        )
        recv_range = range(
            block.mpi_recv_offsets[neighbor_index];
            stop=block.mpi_recv_offsets[neighbor_index + 1] - 1,
        )
        received_global_ids[neighbor_index] = Vector{Int64}(undef, length(recv_range))
        push!(requests, MPI.Irecv!(received_global_ids[neighbor_index], comm;
            source=neighbor, tag=9891))
        push!(requests, MPI.Isend(metadata.mpi_send_global_ids[send_range], comm;
            dest=neighbor, tag=9891))
    end
    MPI.Waitall(requests)
    for neighbor_index in eachindex(block.mpi_neighbors)
        recv_range = range(
            block.mpi_recv_offsets[neighbor_index];
            stop=block.mpi_recv_offsets[neighbor_index + 1] - 1,
        )
        expected = metadata.mpi_recv_global_ids[recv_range]
        received_global_ids[neighbor_index] == expected || error(
            "MPI halo global-ID mismatch with rank $(block.mpi_neighbors[neighbor_index]): " *
            "expected=$expected received=$(received_global_ids[neighbor_index])",
        )
    end
    return block
end

"""Partition the replicated global mesh and return this rank's local block."""
function partition_unstruct_mesh(
    block::UnstructBlock, comm;
    order::Int=1,
    viscous_enabled::Bool=false,
    resistive_enabled::Bool=false,
    mhd_enabled::Bool=false,
    ct_enabled::Bool=false,
    compute_imbalance::Float64=1.05,
    memory_imbalance::Float64=1.05,
    max_ghost_ratio::Float64=0.5,
    periodic_edge_multiplier::Int=16,
    seed::Int=17,
)
    rank = MPI.Comm_rank(comm)
    nparts = MPI.Comm_size(comm)
    graph = build_unstruct_dual_graph(
        block; order=order, viscous_enabled=viscous_enabled,
        resistive_enabled=resistive_enabled, mhd_enabled=mhd_enabled,
        ct_enabled=ct_enabled,
        periodic_edge_multiplier=periodic_edge_multiplier,
    )
    assignment = zeros(Int, graph.ncell)
    if rank == 0
        assignment .= partition_unstruct_graph(
            graph, nparts; compute_imbalance=compute_imbalance,
            memory_imbalance=memory_imbalance, seed=seed,
        )
    end
    MPI.Bcast!(assignment, 0, comm)
    quality = evaluate_unstruct_partition(graph, assignment, nparts)
    rank == 0 && print_unstruct_partition_quality(quality)
    quality.max_ghost_ratio <= max_ghost_ratio || error(
        "partition max ghost ratio $(quality.max_ghost_ratio) exceeds " *
        "configured limit $max_ghost_ratio",
    )
    local_block = localize_unstruct_partition(
        block, graph, assignment, rank; ct_enabled=ct_enabled,
    )
    validate_unstruct_halo_plan!(local_block, comm)
    return local_block, assignment, quality
end
