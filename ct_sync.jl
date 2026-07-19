# CT synchronization across conformal multi-block interfaces.
#
# A face update is topologically divergence-free only when every incident
# block uses the same oriented line EMF on a shared physical edge. Therefore
# edge values are synchronized before ct_update_face_b_from_emf_kernel! runs.

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
end

mutable struct CTSyncPlan{T}
    exchanges::Vector{CTInterfaceExchange}
    send_buffers::Vector{Vector{T}}
    recv_buffers::Vector{Vector{T}}
    halo_send_buffers::Vector{Vector{T}}
    halo_recv_buffers::Vector{Vector{T}}
    local_peer::Vector{Int}
    propagation_passes::Int
end

struct CTBlockEdge
    bid::Int
    jside::Int
    kside::Int
end

struct CTJunctionJob
    local_edge::CTBlockEdge
    remote_edge::CTBlockEdge
    remote_rank::Int
    global_i_s::Int
    global_i_e::Int
    tag::Int
end

mutable struct CTJunctionPlan{T}
    send_jobs::Vector{CTJunctionJob}
    recv_jobs::Vector{CTJunctionJob}
    send_buffers::Vector{Vector{T}}
    recv_buffers::Vector{Vector{T}}
end

@inline ct_interface_face_supported(fid::Integer) = 3 <= fid <= 6
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
    fid::Integer, nb_fid::Integer, reverse_tan::Bool,
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
    peer_u = reshape(
        @view(peer_buffer[nface+1:nface+nuedge]), u_len, v_len + 1,
    )
    peer_v = reshape(
        @view(peer_buffer[nface+nuedge+1:expected]), u_len + 1, v_len,
    )
    if reverse_tan
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
           _ct_canonical_overlap(other) == _ct_canonical_overlap(ex)
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
    if fid == 3 || fid == 4
        return CTBlockEdge(bid, fid, endpoint == 1 ? 5 : 6)
    elseif fid == 5 || fid == 6
        return CTBlockEdge(bid, endpoint == 1 ? 3 : 4, fid)
    end
    error("Unsupported CT junction face $fid")
end

@inline ct_block_edge_key(edge::CTBlockEdge) = (
    edge.bid, edge.jside, edge.kside,
)

function _ct_union_find_root!(parent, edge)
    root = get(parent, edge, edge)
    if root != edge
        root = _ct_union_find_root!(parent, root)
        parent[edge] = root
    else
        parent[edge] = edge
    end
    return root
end

function _ct_union_edges!(parent, first_edge, second_edge)
    first_root = _ct_union_find_root!(parent, first_edge)
    second_root = _ct_union_find_root!(parent, second_edge)
    if first_root == second_root
        return
    end
    if ct_block_edge_key(first_root) < ct_block_edge_key(second_root)
        parent[second_root] = first_root
    else
        parent[first_root] = second_root
    end
end

function ct_junction_edge_groups(connectivity)
    parent = Dict{CTBlockEdge,CTBlockEdge}()
    for ((bid, fid), conn) in connectivity
        if !ct_interface_face_supported(fid) ||
           !ct_interface_face_supported(conn.src_f)
            continue
        end
        for endpoint in 1:2
            nb_endpoint = conn.reverse_tan ? 3 - endpoint : endpoint
            local_edge = ct_block_edge_from_face(bid, fid, endpoint)
            neighbor_edge = ct_block_edge_from_face(
                conn.src_b, conn.src_f, nb_endpoint,
            )
            _ct_union_edges!(parent, local_edge, neighbor_edge)
        end
    end

    groups = Dict{CTBlockEdge,Vector{CTBlockEdge}}()
    for edge in keys(parent)
        root = _ct_union_find_root!(parent, edge)
        push!(get!(groups, root, CTBlockEdge[]), edge)
    end
    result = Vector{Vector{CTBlockEdge}}()
    for edges in values(groups)
        unique_edges = unique(edges)
        if length(unique_edges) > 1
            sort!(unique_edges, by=ct_block_edge_key)
            push!(result, unique_edges)
        end
    end
    sort!(result, by=edges -> ct_block_edge_key(first(edges)))
    return result
end

@inline function _ct_edge_rank_coordinates(edge, layout)
    ry = edge.jside == 3 ? 0 : layout[2] - 1
    rz = edge.kside == 5 ? 0 : layout[3] - 1
    return ry, rz
end

@inline function _ct_edge_rank_global(bid, rx, ry, rz, layout, rank_offsets)
    return rank_offsets[bid + 1] + rx*(layout[2]*layout[3]) +
        ry*layout[3] + rz
end

function build_ct_junction_plan(
    blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b,
)
    groups = ct_junction_edge_groups(connectivity)
    send_jobs = CTJunctionJob[]
    recv_jobs = CTJunctionJob[]
    tag_codes = Dict{Tuple{Int,CTBlockEdge},Int}()
    next_tag_code = 0
    for (group_index, edges) in enumerate(groups)
        for remote_edge in edges[2:end]
            tag_codes[(group_index, remote_edge)] = next_tag_code
            next_tag_code += 1
        end
    end

    for (group_index, edges) in enumerate(groups)
        owner = first(edges)
        owner_layout = Tuple(Int.(Block_Nprocs[owner.bid + 1]))
        owner_ry, owner_rz = _ct_edge_rank_coordinates(owner, owner_layout)
        for local_edge in edges
            if !haskey(blocks, local_edge.bid)
                continue
            end
            b = blocks[local_edge.bid]
            local_layout = Tuple(Int.(Block_Nprocs[local_edge.bid + 1]))
            local_ry, local_rz = _ct_edge_rank_coordinates(
                local_edge, local_layout,
            )
            if b.ry != local_ry || b.rz != local_rz
                continue
            end
            local_x_s, local_x_e = _nonuniform_extent(
                b.rx, local_layout[1], Nx_b[local_edge.bid + 1],
            )

            if local_edge == owner
                for remote_edge in edges[2:end]
                    remote_layout = Tuple(Int.(Block_Nprocs[remote_edge.bid + 1]))
                    remote_ry, remote_rz = _ct_edge_rank_coordinates(
                        remote_edge, remote_layout,
                    )
                    for remote_rx in 0:(remote_layout[1] - 1)
                        remote_x_s, remote_x_e = _nonuniform_extent(
                            remote_rx, remote_layout[1],
                            Nx_b[remote_edge.bid + 1],
                        )
                        overlap_s = max(local_x_s, remote_x_s)
                        overlap_e = min(local_x_e, remote_x_e)
                        if overlap_s > overlap_e
                            continue
                        end
                        remote_rank = _ct_edge_rank_global(
                            remote_edge.bid, remote_rx, remote_ry, remote_rz,
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
                        ))
                    end
                end
            else
                for owner_rx in 0:(owner_layout[1] - 1)
                    owner_x_s, owner_x_e = _nonuniform_extent(
                        owner_rx, owner_layout[1], Nx_b[owner.bid + 1],
                    )
                    overlap_s = max(local_x_s, owner_x_s)
                    overlap_e = min(local_x_e, owner_x_e)
                    if overlap_s > overlap_e
                        continue
                    end
                    owner_rank = _ct_edge_rank_global(
                        owner.bid, owner_rx, owner_ry, owner_rz,
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
    return CTJunctionPlan(send_jobs, recv_jobs, send_buffers, recv_buffers)
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
                error("CT multi-block synchronization currently requires inter-block faces 3:6; got block=$bid face=$fid")
            end
            if !_ms_subdomain_touches_face(fid, b.rx, b.ry, b.rz, px, py, pz)
                continue
            end

            conn = connectivity[(bid, fid)]
            nb_bid = conn.src_b
            nb_fid = conn.src_f
            if !ct_interface_face_supported(nb_fid)
                error("CT multi-block synchronization currently requires inter-block faces 3:6; got neighbor block=$nb_bid face=$nb_fid")
            end

            u_d_s, u_d_e, v_d_s, v_d_e, v_total_d = _ms_face_uv_extent(
                fid, b.rx, b.ry, b.rz, px, py, pz,
                Nx_b[bid + 1], Ny_b[bid + 1], Nz_b[bid + 1],
            )
            npx, npy, npz = Tuple(Int.(Block_Nprocs[nb_bid + 1]))
            v_total_nb = nb_fid <= 4 ? Nz_b[nb_bid + 1] : Ny_b[nb_bid + 1]
            if Nx_b[bid + 1] != Nx_b[nb_bid + 1] || v_total_d != v_total_nb
                error(
                    "CT requires conforming inter-block faces; " *
                    "block=$bid face=$fid has ($(Nx_b[bid + 1]),$v_total_d) cells, " *
                    "neighbor block=$nb_bid face=$nb_fid has " *
                    "($(Nx_b[nb_bid + 1]),$v_total_nb)",
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
                mapped_v_s = conn.reverse_tan ? v_total_s - v_s_e + 1 : v_s_s
                mapped_v_e = conn.reverse_tan ? v_total_s - v_s_s + 1 : v_s_e
                u_int_s = max(u_d_s, u_s_s)
                u_int_e = min(u_d_e, u_s_e)
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
                canonical_v_s, canonical_v_e = if local_is_canonical ||
                                                   !conn.reverse_tan
                    (v_int_s, v_int_e)
                else
                    (v_total_d - v_int_e + 1, v_total_d - v_int_s + 1)
                end
                push!(exchanges, CTInterfaceExchange(
                    bid, fid, nb_bid, nb_fid, nb_rank, conn.reverse_tan,
                    local_u_s, local_u_e, local_v_s, local_v_e,
                    u_int_s, u_int_e, canonical_v_s, canonical_v_e, tag,
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
    if fid == 3 || fid == 4
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

function ct_unpack_interface_sheet!(
    b, ex::CTInterfaceExchange, buffer;
    sync_face_flux::Bool, sync_edges::Bool,
)
    # The lexicographically smaller block/face endpoint owns the canonical
    # oriented values. Masters receive data but deliberately keep their state.
    if (ex.bid, ex.fid) < (ex.nb_bid, ex.nb_fid)
        return nothing
    end

    u_len = ex.u_e - ex.u_s + 1
    v_len = ex.v_e - ex.v_s + 1
    nface = u_len*v_len
    nuedge = u_len*(v_len + 1)
    nvedge = (u_len + 1)*v_len
    face_src = reshape(@view(buffer[1:nface]), u_len, v_len)
    u_edge_src = reshape(@view(buffer[nface+1:nface+nuedge]), u_len, v_len + 1)
    v_edge_src = reshape(
        @view(buffer[nface+nuedge+1:nface+nuedge+nvedge]), u_len + 1, v_len,
    )

    if ex.reverse_tan
        face_values = Array(face_src[:, end:-1:1])
        u_edge_values = Array(u_edge_src[:, end:-1:1])
        v_edge_values = Array(v_edge_src[:, end:-1:1])
    else
        face_values = Array(face_src)
        u_edge_values = Array(u_edge_src)
        v_edge_values = Array(v_edge_src)
    end
    face_values .*= FT(ct_interface_face_flux_sign(ex.fid, ex.nb_fid))
    v_edge_values .*= FT(ct_interface_v_edge_sign(ex.reverse_tan))

    face, u_edge, v_edge = _ct_interface_views(
        b, ex.fid, ex.u_s, ex.u_e, ex.v_s, ex.v_e,
    )
    if sync_face_flux
        _ct_copy_host_sheet_to_device!(face, face_values)
    end
    if sync_edges
        _ct_copy_host_sheet_to_device!(u_edge, u_edge_values)
        _ct_copy_host_sheet_to_device!(v_edge, v_edge_values)
    end
    return nothing
end

function ct_sync_interface_sheets!(
    blocks, plan::CTSyncPlan;
    sync_face_flux::Bool=false,
    sync_edges::Bool=true,
)
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
                sync_face_flux=sync_face_flux, sync_edges=sync_edges,
            )
        end
        gpu_sync()
    end
    return nothing
end

@inline function _ct_block_edge_view(b, edge::CTBlockEdge, global_i_s, global_i_e)
    local_i_s = global_i_s - b.ox
    local_i_e = global_i_e - b.ox
    j = edge.jside == 3 ? 1 : b.Ny + 1
    k = edge.kside == 5 ? 1 : b.Nz + 1
    return @view b.Ex_edge[local_i_s:local_i_e, j, k]
end

function ct_sync_junction_edges!(blocks, plan::CTJunctionPlan)
    if isempty(plan.recv_jobs) && isempty(plan.send_jobs)
        return nothing
    end
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    requests = MPI.Request[]

    for (index, job) in enumerate(plan.recv_jobs)
        if job.remote_rank != world_rank
            push!(requests, MPI.Irecv!(
                plan.recv_buffers[index], MPI.COMM_WORLD;
                source=job.remote_rank, tag=job.tag,
            ))
        end
    end
    for (index, job) in enumerate(plan.send_jobs)
        source = _ct_block_edge_view(
            blocks[job.local_edge.bid], job.local_edge,
            job.global_i_s, job.global_i_e,
        )
        copyto!(plan.send_buffers[index], vec(Array(source)))
        if job.remote_rank != world_rank
            push!(requests, MPI.Isend(
                plan.send_buffers[index], MPI.COMM_WORLD;
                dest=job.remote_rank, tag=job.tag,
            ))
        end
    end
    if !isempty(requests)
        MPI.Waitall(requests)
    end

    for (index, job) in enumerate(plan.recv_jobs)
        destination = _ct_block_edge_view(
            blocks[job.local_edge.bid], job.local_edge,
            job.global_i_s, job.global_i_e,
        )
        if job.remote_rank == world_rank
            source = _ct_block_edge_view(
                blocks[job.remote_edge.bid], job.remote_edge,
                job.global_i_s, job.global_i_e,
            )
            values = Array(source)
        else
            values = plan.recv_buffers[index]
        end
        _ct_copy_host_sheet_to_device!(destination, values)
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
    sync_face_flux::Bool, sync_edges::Bool,
)
    face, u_edge, v_edge = _ct_rank_sheet_views(b, direction, side)
    nface = length(face)
    nuedge = length(u_edge)
    nvedge = length(v_edge)
    if sync_face_flux
        values = reshape(@view(buffer[1:nface]), size(face))
        _ct_copy_host_sheet_to_device!(face, values)
    end
    if sync_edges
        u_values = reshape(
            @view(buffer[nface+1:nface+nuedge]), size(u_edge),
        )
        v_values = reshape(
            @view(buffer[nface+nuedge+1:nface+nuedge+nvedge]), size(v_edge),
        )
        _ct_copy_host_sheet_to_device!(u_edge, u_values)
        _ct_copy_host_sheet_to_device!(v_edge, v_values)
    end
    return nothing
end

function ct_sync_rank_sheets!(
    blocks, block_comms, Block_Nprocs;
    sync_face_flux::Bool=false,
    sync_edges::Bool=true,
)
    passes = sync_edges ? 3 : 1
    for _ in 1:passes
        for bid in sort(collect(keys(blocks)))
            b = blocks[bid]
            comm = block_comms[bid]
            layout = Block_Nprocs[bid + 1]
            for direction in 1:3
                if layout[direction] <= 1
                    continue
                end
                src, dst = MPI.Cart_shift(comm, direction - 1, 1)
                if src == MPI.PROC_NULL && dst == MPI.PROC_NULL
                    continue
                end
                send_buffer = ct_pack_rank_sheet(b, direction, 1)
                recv_buffer = similar(send_buffer)
                MPI.Sendrecv!(
                    send_buffer, recv_buffer, comm; dest=dst, source=src,
                )
                if src != MPI.PROC_NULL
                    ct_unpack_rank_sheet!(
                        b, direction, -1, recv_buffer;
                        sync_face_flux=sync_face_flux, sync_edges=sync_edges,
                    )
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
    return 3 * (nuface + nvface)
end

@inline function _ct_interface_source_layers(fid, ncell)
    return isodd(fid) ? (NG+1:2NG) : (ncell+1:ncell+NG)
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
    normal_layers = ex.fid <= 4 ?
        _ct_interface_source_layers(ex.fid, b.Ny) :
        _ct_interface_source_layers(ex.fid, b.Nz)
    if ex.fid == 3 || ex.fid == 4
        i_face = ex.u_s+NG:ex.u_e+NG+1
        k_face = ex.v_s+NG:ex.v_e+NG+1
        u_components = ntuple(3) do component
            variable = UBX + component - 1
            FT(0.5) .* (
                Array(@view b.U[first(i_face)-1:last(i_face)-1,
                                normal_layers, first(k_face):last(k_face)-1,
                                variable]) .+
                Array(@view b.U[first(i_face):last(i_face),
                                normal_layers, first(k_face):last(k_face)-1,
                                variable])
            )
        end
        v_components = ntuple(3) do component
            variable = UBX + component - 1
            FT(0.5) .* (
                Array(@view b.U[first(i_face):last(i_face)-1,
                                normal_layers, first(k_face)-1:last(k_face)-1,
                                variable]) .+
                Array(@view b.U[first(i_face):last(i_face)-1,
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
            variable = UBX + component - 1
            raw = FT(0.5) .* (
                Array(@view b.U[first(i_face)-1:last(i_face)-1,
                                first(j_face):last(j_face)-1, normal_layers,
                                variable]) .+
                Array(@view b.U[first(i_face):last(i_face),
                                first(j_face):last(j_face)-1, normal_layers,
                                variable])
            )
            permutedims(raw, (1, 3, 2))
        end
        v_components = ntuple(3) do component
            variable = UBX + component - 1
            raw = FT(0.5) .* (
                Array(@view b.U[first(i_face):last(i_face)-1,
                                first(j_face)-1:last(j_face)-1, normal_layers,
                                variable]) .+
                Array(@view b.U[first(i_face):last(i_face)-1,
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
    return buffer
end

@inline function _ct_interface_destination_layers(fid, ncell)
    return isodd(fid) ? (1:NG) : (ncell+NG+1:ncell+2NG)
end

function ct_unpack_interface_halo!(b, ex::CTInterfaceExchange, buffer)
    u_len = ex.u_e - ex.u_s + 1
    v_len = ex.v_e - ex.v_s + 1
    nuface = (u_len + 1)*NG*v_len
    nvface = u_len*NG*(v_len + 1)
    u_sources = ntuple(3) do component
        offset = (component - 1) * nuface
        reshape(
            @view(buffer[offset+1:offset+nuface]), u_len + 1, NG, v_len,
        )
    end
    v_base = 3nuface
    v_sources = ntuple(3) do component
        offset = v_base + (component - 1) * nvface
        reshape(
            @view(buffer[offset+1:offset+nvface]), u_len, NG, v_len + 1,
        )
    end
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

    destination_layers = ex.fid <= 4 ?
        _ct_interface_destination_layers(ex.fid, b.Ny) :
        _ct_interface_destination_layers(ex.fid, b.Nz)
    if ex.fid == 3 || ex.fid == 4
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

function _ct_sync_rank_face_halo_field!(array, direction, ncell, comm)
    src, dst = MPI.Cart_shift(comm, direction - 1, 1)
    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        send_high = Array(selectdim(array, direction, ncell+1:ncell+NG))
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
                selectdim(array, direction, ncell+NG+1:ncell+2NG),
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
                # The normal face itself is synchronized as a duplicated sheet;
                # only tangential face families require cell-like ghost slabs.
                if component == direction
                    continue
                end
                _ct_sync_rank_face_halo_field!(
                    face_fields[component], direction,
                    cell_counts[direction], comm,
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
            for array in metric_groups[direction]
                _ct_sync_host_metric_face!(
                    array, direction, cell_counts[direction], comm,
                )
            end
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
