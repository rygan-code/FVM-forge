using Printf

const METRIC_CT_INTERFACE_DIAGNOSTIC_TAG_BASE = 30000
const METRIC_CT_INTERFACE_STATS_COLUMNS = (
    :step, :time,
    :face_flux_abs, :face_flux_rel,
    :line_emf_abs, :line_emf_rel,
    :nonfinite_rank_flag,
)
const _metric_ct_interface_plan_cache = Ref{Any}(nothing)
const _metric_ct_interface_stats_initialized = Ref(false)

metric_ct_interface_stats_header() =
    "# " * join(string.(METRIC_CT_INTERFACE_STATS_COLUMNS), " ")

@inline function _metric_ct_scaled_residual(local_values, peer_values)
    absolute = maximum(abs, local_values .- peer_values)
    scale = max(maximum(abs, local_values), maximum(abs, peer_values))
    relative = iszero(scale) ? absolute : absolute / scale
    return Float64(absolute), Float64(relative)
end

function metric_ct_interface_residual(
    local_buffer, peer_buffer, u_len::Integer, v_len::Integer,
    fid::Integer, nb_fid::Integer, reverse_tan::Bool;
    transform=nothing,
)
    u_len > 0 && v_len > 0 ||
        throw(ArgumentError("interface dimensions must be positive"))
    expected_length = ct_interface_buffer_length(u_len, v_len)
    length(local_buffer) == expected_length || throw(DimensionMismatch(
        "local interface buffer has $(length(local_buffer)) values; expected $expected_length",
    ))
    length(peer_buffer) == expected_length || throw(DimensionMismatch(
        "peer interface buffer has $(length(peer_buffer)) values; expected $expected_length",
    ))

    if !all(isfinite, local_buffer) || !all(isfinite, peer_buffer)
        return (
            face_flux_abs=0.0,
            face_flux_rel=0.0,
            line_emf_abs=0.0,
            line_emf_rel=0.0,
            nonfinite_rank_flag=Int32(1),
        )
    end

    nface = u_len * v_len
    nuedge = u_len * (v_len + 1)
    nvedge = (u_len + 1) * v_len
    local_face = reshape(@view(local_buffer[1:nface]), u_len, v_len)
    local_u_edge = reshape(
        @view(local_buffer[nface+1:nface+nuedge]), u_len, v_len + 1,
    )
    local_v_edge = reshape(
        @view(local_buffer[nface+nuedge+1:nface+nuedge+nvedge]),
        u_len + 1, v_len,
    )
    source_u_len, source_v_len = if transform !== nothing &&
        (structured_face_transform_code(transform) & 1) != 0
        (v_len, u_len)
    else
        (u_len, v_len)
    end
    source_nface = source_u_len * source_v_len
    source_nuedge = source_u_len * (source_v_len + 1)
    source_nvedge = (source_u_len + 1) * source_v_len
    peer_face = reshape(@view(peer_buffer[1:source_nface]), source_u_len, source_v_len)
    peer_u_edge = reshape(
        @view(peer_buffer[source_nface+1:source_nface+source_nuedge]),
        source_u_len, source_v_len + 1,
    )
    peer_v_edge = reshape(
        @view(peer_buffer[source_nface+source_nuedge+1:source_nface+source_nuedge+source_nvedge]),
        source_u_len + 1, source_v_len,
    )

    if transform !== nothing
        peer_face, peer_u_edge, peer_v_edge = _ct_reorient_interface_sheet(
            peer_face, peer_u_edge, peer_v_edge,
            transform, u_len, v_len,
        )
    elseif reverse_tan
        peer_face = peer_face[:, end:-1:1]
        peer_u_edge = peer_u_edge[:, end:-1:1]
        peer_v_edge = peer_v_edge[:, end:-1:1]
    end
    oriented_face = (transform === nothing ?
        ct_interface_face_flux_sign(fid, nb_fid) :
        ct_interface_face_flux_sign(fid, nb_fid, transform)) .* peer_face
    # The canonical u tangent is the global xi direction and is never reversed.
    # The v tangent follows reverse_tan and uses the solver's synchronization sign.
    oriented_u_edge = peer_u_edge
    oriented_v_edge = transform === nothing ?
        ct_interface_v_edge_sign(reverse_tan) .* peer_v_edge : peer_v_edge

    face_flux_abs, face_flux_rel = _metric_ct_scaled_residual(
        local_face, oriented_face,
    )
    u_abs, u_rel = _metric_ct_scaled_residual(local_u_edge, oriented_u_edge)
    v_abs, v_rel = _metric_ct_scaled_residual(local_v_edge, oriented_v_edge)
    return (
        face_flux_abs=face_flux_abs,
        face_flux_rel=face_flux_rel,
        line_emf_abs=max(u_abs, v_abs),
        line_emf_rel=max(u_rel, v_rel),
        nonfinite_rank_flag=Int32(0),
    )
end

function metric_ct_pack_interface_sheet(block, exchange)
    face, u_edge, v_edge = _ct_interface_views(
        block, exchange.fid,
        exchange.u_s, exchange.u_e, exchange.v_s, exchange.v_e,
    )
    return vcat(vec(Array(face)), vec(Array(u_edge)), vec(Array(v_edge)))
end

function _metric_ct_interface_plan(blocks, Block_Nprocs)
    cached = _metric_ct_interface_plan_cache[]
    cached === nothing || return cached

    connectivity_path = joinpath(mesh_dir, "block_connectivity.h5")
    _, connectivity, face_bc, _, nx_b, ny_b, nz_b =
        load_multiblock_connectivity(
            connectivity_path; world_rank=MPI.Comm_rank(MPI.COMM_WORLD),
        )
    rank_offsets = zeros(Int, length(Block_Nprocs) + 1)
    for index in eachindex(Block_Nprocs)
        rank_offsets[index + 1] =
            rank_offsets[index] + prod(Block_Nprocs[index])
    end
    rank_offsets[end] == MPI.Comm_size(MPI.COMM_WORLD) || error(
        "metric CT interface diagnostics require one standard partition per rank",
    )
    cached = build_ct_sync_plan(
        blocks, face_bc, connectivity, Block_Nprocs, rank_offsets,
        nx_b, ny_b, nz_b,
    )
    _metric_ct_interface_plan_cache[] = cached
    return cached
end

function metric_ct_interface_diagnostics(blocks, Block_Nprocs)
    plan = _metric_ct_interface_plan(blocks, Block_Nprocs)
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    local_face_abs = 0.0
    local_face_rel = 0.0
    local_line_abs = 0.0
    local_line_rel = 0.0
    local_nonfinite = Int32(0)
    local_exchange_count = Int32(0)

    for exchange in plan.exchanges
        exchange.nb_rank != world_rank || error(
            "metric CT two-rank diagnostics require a remote peer for every interface",
        )
        send_buffer = metric_ct_pack_interface_sheet(
            blocks[exchange.bid], exchange,
        )
        recv_buffer = similar(send_buffer)
        tag_code = exchange.tag - CT_INTERFACE_TAG_BASE
        tag = _ct_checked_tag(
            METRIC_CT_INTERFACE_DIAGNOSTIC_TAG_BASE,
            tag_code,
            CT_MPI_TAG_MAX,
            "metric CT interface diagnostic",
        )
        MPI.Sendrecv!(
            send_buffer, recv_buffer, MPI.COMM_WORLD;
            dest=exchange.nb_rank, sendtag=tag,
            source=exchange.nb_rank, recvtag=tag,
        )

        u_len = exchange.u_e - exchange.u_s + 1
        v_len = exchange.v_e - exchange.v_s + 1
        residual = metric_ct_interface_residual(
            send_buffer, recv_buffer, u_len, v_len,
            exchange.fid, exchange.nb_fid, exchange.reverse_tan;
            transform=(exchange.transform === nothing ? nothing :
                       structured_inverse_face_transform(exchange.transform)),
        )
        local_face_abs = max(local_face_abs, residual.face_flux_abs)
        local_face_rel = max(local_face_rel, residual.face_flux_rel)
        local_line_abs = max(local_line_abs, residual.line_emf_abs)
        local_line_rel = max(local_line_rel, residual.line_emf_rel)
        local_nonfinite = max(
            local_nonfinite, residual.nonfinite_rank_flag,
        )
        local_exchange_count += Int32(1)
    end

    global_exchange_count = MPI.Allreduce(
        local_exchange_count, MPI.SUM, MPI.COMM_WORLD,
    )
    global_exchange_count > 0 || error(
        "metric CT interface diagnostics found no shared interface sheets",
    )
    return (
        face_flux_abs=MPI.Allreduce(
            local_face_abs, MPI.MAX, MPI.COMM_WORLD,
        ),
        face_flux_rel=MPI.Allreduce(
            local_face_rel, MPI.MAX, MPI.COMM_WORLD,
        ),
        line_emf_abs=MPI.Allreduce(
            local_line_abs, MPI.MAX, MPI.COMM_WORLD,
        ),
        line_emf_rel=MPI.Allreduce(
            local_line_rel, MPI.MAX, MPI.COMM_WORLD,
        ),
        nonfinite_rank_flag=MPI.Allreduce(
            local_nonfinite, MPI.MAX, MPI.COMM_WORLD,
        ),
    )
end

function metric_ct_interface_post_process(
    tt, active_time, current_dt, blocks, world_rank, Block_Nprocs,
)
    if tt % _stats_interval != 0 && tt != 1
        return nothing
    end
    residual = metric_ct_interface_diagnostics(blocks, Block_Nprocs)
    state_time = active_time + current_dt
    if world_rank == 0
        path = get(
            ENV, "METRIC_CT_INTERFACE_STATS_FILE",
            joinpath(@__DIR__, "interface_stats.dat"),
        )
        mode = _metric_ct_interface_stats_initialized[] ? "a" : "w"
        open(path, mode) do io
            if !_metric_ct_interface_stats_initialized[]
                println(io, metric_ct_interface_stats_header())
            end
            @printf(
                io, "%d %.17e %.17e %.17e %.17e %.17e %d\n",
                tt, state_time,
                residual.face_flux_abs, residual.face_flux_rel,
                residual.line_emf_abs, residual.line_emf_rel,
                residual.nonfinite_rank_flag,
            )
        end
        _metric_ct_interface_stats_initialized[] = true
    end
    return residual
end
