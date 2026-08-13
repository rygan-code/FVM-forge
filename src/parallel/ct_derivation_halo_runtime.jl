if !isdefined(@__MODULE__, :CTPackedShellLayout)
    include(joinpath(@__DIR__, "ct_derivation_halo.jl"))
end

mutable struct CTDerivationHalo
    cell_layout::CTPackedShellLayout
    face_layout::CTPackedShellLayout
    conservative_shell
    inverse_volume_shell
    face_b_shells::NTuple{3,Any}
    background_face_shells::NTuple{3,Any}
    face_area_vector_shells::NTuple{3,Any}
end

struct CTLocalCellGatherJob
    destination_bid::Int
    source_bid::Int
    offsets
    source_indices
    count::Int32
end

struct CTLocalFaceGatherJob
    destination_bid::Int
    destination_axis::Int8
    source_bid::Int
    source_axis::Int8
    offsets
    source_indices
    orientations
    count::Int32
end

mutable struct CTDerivationHaloPlan
    halos::Dict{Int,CTDerivationHalo}
    local_cell_jobs::Vector{CTLocalCellGatherJob}
    local_face_jobs::Vector{CTLocalFaceGatherJob}
    remote_cell_send_jobs::Vector{Any}
    remote_cell_receive_jobs::Vector{Any}
    remote_face_send_jobs::Vector{Any}
    remote_face_receive_jobs::Vector{Any}
    communicator
    background_enabled::Bool
    persistent_bytes::Int64
end

struct CTCellDestinationRequest
    source::CTDerivationOwnedCellSource
    destination_bid::Int32
    destination_offset::Int32
end

struct CTFaceDestinationRequest
    source::CTDerivationOwnedFaceSource
    destination_bid::Int32
    destination_axis::Int8
    destination_offset::Int32
end

struct CTRemoteCellSendSubjob
    source_bid::Int
    positions
    source_indices
    count::Int32
end

mutable struct CTRemoteCellSendJob
    peer_rank::Int
    subjobs::Vector{CTRemoteCellSendSubjob}
    dynamic_device
    dynamic_host
    static_device
    static_host
end

struct CTRemoteCellReceiveSubjob
    destination_bid::Int
    positions
    destination_offsets
    count::Int32
end

mutable struct CTRemoteCellReceiveJob
    peer_rank::Int
    subjobs::Vector{CTRemoteCellReceiveSubjob}
    dynamic_device
    dynamic_host
    static_device
    static_host
end

struct CTRemoteFaceSendSubjob
    source_bid::Int
    source_axis::Int8
    positions
    source_indices
    count::Int32
end

mutable struct CTRemoteFaceSendJob
    peer_rank::Int
    subjobs::Vector{CTRemoteFaceSendSubjob}
    dynamic_device
    dynamic_host
    static_device
    static_host
end

struct CTRemoteFaceReceiveSubjob
    destination_bid::Int
    destination_axis::Int8
    positions
    destination_offsets
    orientations
    count::Int32
end

mutable struct CTRemoteFaceReceiveJob
    peer_rank::Int
    subjobs::Vector{CTRemoteFaceReceiveSubjob}
    dynamic_device
    dynamic_host
    static_device
    static_host
end

const CT_DERIVATION_REQUEST_CELL_COUNT_TAG = 28001
const CT_DERIVATION_REQUEST_CELL_DATA_TAG = 28002
const CT_DERIVATION_REQUEST_FACE_COUNT_TAG = 28003
const CT_DERIVATION_REQUEST_FACE_DATA_TAG = 28004
const CT_DERIVATION_CELL_DYNAMIC_TAG = 28101
const CT_DERIVATION_CELL_STATIC_TAG = 28102
const CT_DERIVATION_FACE_DYNAMIC_TAG = 28103
const CT_DERIVATION_FACE_STATIC_TAG = 28104

@inline function _ct_derivation_face_fields(block)
    return (block.Bx_face, block.By_face, block.Bz_face)
end

@inline function _ct_derivation_background_face_fields(block)
    return hasproperty(block, :B0x_face) ?
        (block.B0x_face, block.B0y_face, block.B0z_face) :
        (nothing, nothing, nothing)
end

@inline function _ct_derivation_face_metric_fields(block, axis::Integer)
    if axis == 1
        return block.Areai, block.nxi, block.nyi, block.nzi
    elseif axis == 2
        return block.Areaj, block.nxj, block.nyj, block.nzj
    end
    return block.Areak, block.nxk, block.nyk, block.nzk
end

function _ct_derivation_device_indices(indices::Vector{NTuple{3,Int32}})
    host = Matrix{Int32}(undef, length(indices), 3)
    for row in eachindex(indices)
        host[row,1], host[row,2], host[row,3] = indices[row]
    end
    return GPUArray(host)
end

function _ct_derivation_add_cell_job!(groups, destination_bid, source)
    key = (Int(destination_bid), Int(source.bid))
    offsets, indices = get!(groups, key) do
        (Int32[], NTuple{3,Int32}[])
    end
    return offsets, indices
end

function _ct_derivation_add_face_job!(groups, destination_bid, destination_axis, source)
    key = (
        Int(destination_bid), Int(destination_axis), Int(source.bid),
        Int(source.axis),
    )
    offsets, indices, orientations = get!(groups, key) do
        (Int32[], NTuple{3,Int32}[], Int8[])
    end
    return offsets, indices, orientations
end

function _ct_derivation_exchange_request_rows(
    outgoing_rows::Dict{Int,Vector{NTuple{N,Int32}}},
    communicator,
    count_tag::Integer,
    data_tag::Integer,
) where {N}
    world_rank = MPI.Comm_rank(communicator)
    world_size = MPI.Comm_size(communicator)
    incoming_rows = Dict{Int,Vector{NTuple{N,Int32}}}()
    for peer in 0:world_size-1
        peer == world_rank && continue
        send_rows = get(outgoing_rows, peer, NTuple{N,Int32}[])
        send_count = Int32[length(send_rows)]
        receive_count = zeros(Int32, 1)
        MPI.Sendrecv!(
            send_count, receive_count, communicator;
            dest=peer, source=peer, sendtag=count_tag, recvtag=count_tag,
        )
        receive_rows = Vector{NTuple{N,Int32}}(undef, receive_count[1])
        send_flat = Vector{Int32}(undef, N*length(send_rows))
        for row in eachindex(send_rows), component in 1:N
            send_flat[(row-1)*N+component] = send_rows[row][component]
        end
        receive_flat = Vector{Int32}(undef, N*Int(receive_count[1]))
        MPI.Sendrecv!(
            send_flat, receive_flat, communicator;
            dest=peer, source=peer, sendtag=data_tag, recvtag=data_tag,
        )
        for row in eachindex(receive_rows)
            receive_rows[row] = ntuple(
                component -> receive_flat[(row-1)*N+component], Val(N),
            )
        end
        isempty(receive_rows) || (incoming_rows[peer] = receive_rows)
    end
    return incoming_rows
end

function _ct_derivation_build_remote_cell_jobs(
    destination_requests,
    communicator,
    components::Integer,
)
    outgoing_rows = Dict{Int,Vector{NTuple{4,Int32}}}()
    payload_positions = Dict{Int,Dict{NTuple{4,Int32},Int32}}()
    for (peer, requests) in destination_requests
        sort!(requests, by=request -> (
            request.source.bid, request.source.local_index...,
            request.destination_bid, request.destination_offset,
        ))
        rows = NTuple{4,Int32}[]
        positions = Dict{NTuple{4,Int32},Int32}()
        for request in requests
            row = (
                request.source.bid, request.source.local_index[1],
                request.source.local_index[2], request.source.local_index[3],
            )
            if !haskey(positions, row)
                push!(rows, row)
                positions[row] = Int32(length(rows))
            end
        end
        outgoing_rows[peer] = rows
        payload_positions[peer] = positions
    end
    incoming_rows = _ct_derivation_exchange_request_rows(
        outgoing_rows, communicator, CT_DERIVATION_REQUEST_CELL_COUNT_TAG,
        CT_DERIVATION_REQUEST_CELL_DATA_TAG,
    )

    send_jobs = CTRemoteCellSendJob[]
    for (peer, rows) in sort(collect(incoming_rows); by=first)
        groups = Dict{Int,Tuple{Vector{Int32},Vector{NTuple{3,Int32}}}}()
        for (position, row) in enumerate(rows)
            positions, indices = get!(groups, Int(row[1])) do
                (Int32[], NTuple{3,Int32}[])
            end
            push!(positions, Int32(position))
            push!(indices, (row[2],row[3],row[4]))
        end
        subjobs = CTRemoteCellSendSubjob[]
        for (source_bid, (positions, indices)) in sort(collect(groups); by=first)
            push!(subjobs, CTRemoteCellSendSubjob(
                source_bid, GPUArray(positions),
                _ct_derivation_device_indices(indices), Int32(length(positions)),
            ))
        end
        dynamic_host = zeros(FT, length(rows), components)
        static_host = zeros(FT, length(rows))
        push!(send_jobs, CTRemoteCellSendJob(
            peer, subjobs, GPUArray(dynamic_host), dynamic_host,
            GPUArray(static_host), static_host,
        ))
    end

    receive_jobs = CTRemoteCellReceiveJob[]
    for (peer, requests) in sort(collect(destination_requests); by=first)
        groups = Dict{Int,Tuple{Vector{Int32},Vector{Int32}}}()
        positions_by_source = payload_positions[peer]
        for request in requests
            positions, offsets = get!(groups, Int(request.destination_bid)) do
                (Int32[], Int32[])
            end
            source_row = (
                request.source.bid, request.source.local_index[1],
                request.source.local_index[2], request.source.local_index[3],
            )
            push!(positions, positions_by_source[source_row])
            push!(offsets, request.destination_offset)
        end
        subjobs = CTRemoteCellReceiveSubjob[]
        for (destination_bid, (positions, offsets)) in
            sort(collect(groups); by=first)
            push!(subjobs, CTRemoteCellReceiveSubjob(
                destination_bid, GPUArray(positions), GPUArray(offsets),
                Int32(length(positions)),
            ))
        end
        payload_count = length(positions_by_source)
        dynamic_host = zeros(FT, payload_count, components)
        static_host = zeros(FT, payload_count)
        push!(receive_jobs, CTRemoteCellReceiveJob(
            peer, subjobs, GPUArray(dynamic_host), dynamic_host,
            GPUArray(static_host), static_host,
        ))
    end
    return send_jobs, receive_jobs
end

function _ct_derivation_build_remote_face_jobs(
    destination_requests,
    communicator,
    dynamic_components::Integer,
)
    outgoing_rows = Dict{Int,Vector{NTuple{5,Int32}}}()
    payload_positions = Dict{Int,Dict{NTuple{5,Int32},Int32}}()
    for (peer, requests) in destination_requests
        sort!(requests, by=request -> (
            request.source.bid, request.source.axis,
            request.source.local_index..., request.destination_bid,
            request.destination_axis, request.destination_offset,
        ))
        rows = NTuple{5,Int32}[]
        positions = Dict{NTuple{5,Int32},Int32}()
        for request in requests
            row = (
                request.source.bid, Int32(request.source.axis),
                request.source.local_index[1], request.source.local_index[2],
                request.source.local_index[3],
            )
            if !haskey(positions, row)
                push!(rows, row)
                positions[row] = Int32(length(rows))
            end
        end
        outgoing_rows[peer] = rows
        payload_positions[peer] = positions
    end
    incoming_rows = _ct_derivation_exchange_request_rows(
        outgoing_rows, communicator, CT_DERIVATION_REQUEST_FACE_COUNT_TAG,
        CT_DERIVATION_REQUEST_FACE_DATA_TAG,
    )

    send_jobs = CTRemoteFaceSendJob[]
    for (peer, rows) in sort(collect(incoming_rows); by=first)
        groups = Dict{Tuple{Int,Int},Tuple{
            Vector{Int32},Vector{NTuple{3,Int32}},
        }}()
        for (position, row) in enumerate(rows)
            positions, indices = get!(groups, (Int(row[1]),Int(row[2]))) do
                (Int32[], NTuple{3,Int32}[])
            end
            push!(positions, Int32(position))
            push!(indices, (row[3],row[4],row[5]))
        end
        subjobs = CTRemoteFaceSendSubjob[]
        for ((source_bid, source_axis), (positions, indices)) in
            sort(collect(groups); by=first)
            push!(subjobs, CTRemoteFaceSendSubjob(
                source_bid, Int8(source_axis), GPUArray(positions),
                _ct_derivation_device_indices(indices), Int32(length(positions)),
            ))
        end
        dynamic_host = zeros(FT, length(rows), dynamic_components)
        static_host = zeros(FT, length(rows), 3)
        push!(send_jobs, CTRemoteFaceSendJob(
            peer, subjobs, GPUArray(dynamic_host), dynamic_host,
            GPUArray(static_host), static_host,
        ))
    end

    receive_jobs = CTRemoteFaceReceiveJob[]
    for (peer, requests) in sort(collect(destination_requests); by=first)
        groups = Dict{Tuple{Int,Int},Tuple{
            Vector{Int32},Vector{Int32},Vector{Int8},
        }}()
        positions_by_source = payload_positions[peer]
        for request in requests
            positions, offsets, orientations = get!(groups, (
                Int(request.destination_bid), Int(request.destination_axis),
            )) do
                (Int32[], Int32[], Int8[])
            end
            source_row = (
                request.source.bid, Int32(request.source.axis),
                request.source.local_index[1], request.source.local_index[2],
                request.source.local_index[3],
            )
            push!(positions, positions_by_source[source_row])
            push!(offsets, request.destination_offset)
            push!(orientations, request.source.orientation)
        end
        subjobs = CTRemoteFaceReceiveSubjob[]
        for ((destination_bid, destination_axis),
             (positions, offsets, orientations)) in
            sort(collect(groups); by=first)
            push!(subjobs, CTRemoteFaceReceiveSubjob(
                destination_bid, Int8(destination_axis), GPUArray(positions),
                GPUArray(offsets), GPUArray(orientations),
                Int32(length(positions)),
            ))
        end
        payload_count = length(positions_by_source)
        dynamic_host = zeros(FT, payload_count, dynamic_components)
        static_host = zeros(FT, payload_count, 3)
        push!(receive_jobs, CTRemoteFaceReceiveJob(
            peer, subjobs, GPUArray(dynamic_host), dynamic_host,
            GPUArray(static_host), static_host,
        ))
    end
    return send_jobs, receive_jobs
end

function build_ct_derivation_halo_plan(
    blocks,
    block_dims,
    face_bc,
    connectivity,
    block_nprocs,
    rank_offsets;
    world_rank::Integer,
    ng::Integer,
    communicator=nothing,
    conservative_components::Integer=5,
    include_cell_data::Bool=true,
    cell_reach::NTuple{3,<:Integer}=(2,2,2),
    face_reach::NTuple{3,<:Integer}=(2,2,2),
)
    if include_cell_data
        conservative_components >= 5 || throw(ArgumentError(
            "CT POINT6 requires at least five conservative components",
        ))
    else
        conservative_components >= 0 || throw(ArgumentError(
            "conservative component count cannot be negative",
        ))
    end
    halos = Dict{Int,CTDerivationHalo}()
    cell_groups = Dict{Tuple{Int,Int},Tuple{
        Vector{Int32},Vector{NTuple{3,Int32}},
    }}()
    face_groups = Dict{NTuple{4,Int},Tuple{
        Vector{Int32},Vector{NTuple{3,Int32}},Vector{Int8},
    }}()
    remote_cell_requests = Dict{Int,Vector{CTCellDestinationRequest}}()
    remote_face_requests = Dict{Int,Vector{CTFaceDestinationRequest}}()
    persistent_bytes = Int64(0)

    for bid in sort(collect(keys(blocks)))
        block = blocks[bid]
        active_dims = (Int(block.Nx),Int(block.Ny),Int(block.Nz))
        cell_dims = Tuple(Int.(size(block.U)[1:3]))
        face_dims = Tuple(Int.(size(block.Bx_face)))
        cell_dims == active_dims .+ 2Int(ng) || throw(DimensionMismatch(
            "CT cell array does not match active dimensions plus ghost width",
        ))
        face_dims == cell_dims .+ 1 || throw(DimensionMismatch(
            "CT face array does not match the padded cell dimensions",
        ))
        all(size(field) == face_dims for field in _ct_derivation_face_fields(block)) ||
            throw(DimensionMismatch("CT face arrays must share padded dimensions"))
        cell_low_reach = ntuple(
            axis -> Int(ng) + Int(cell_reach[axis]), Val(3),
        )
        cell_high_reach = cell_low_reach
        face_low_reach = ntuple(
            axis -> Int(ng) + Int(face_reach[axis]), Val(3),
        )
        # A six-face center recovery reaches one face farther on the high side.
        face_high_reach = ntuple(
            axis -> face_low_reach[axis] + 1, Val(3),
        )
        cell_layout = CTPackedShellLayout(
            active_dims,cell_low_reach,cell_high_reach,
        )
        face_layout = CTPackedShellLayout(
            active_dims,face_low_reach,face_high_reach,
        )
        cell_count = Int(ct_shell_count(cell_layout))
        face_count = Int(ct_shell_count(face_layout))
        conservative_shell = include_cell_data ? gpu_zeros(
            FT, cell_count, conservative_components,
        ) : gpu_zeros(FT, 0, 0)
        inverse_volume_shell = include_cell_data ?
            gpu_zeros(FT, cell_count) : gpu_zeros(FT, 0)
        face_b_shells = ntuple(_ -> gpu_zeros(FT, face_count), Val(3))
        background_fields = _ct_derivation_background_face_fields(block)
        background_shells = ntuple(Val(3)) do axis
            background_fields[axis] === nothing ? nothing :
                gpu_zeros(FT, face_count)
        end
        area_vector_shells = ntuple(
            _ -> gpu_zeros(FT, face_count, 3), Val(3),
        )
        halos[bid] = CTDerivationHalo(
            cell_layout, face_layout, conservative_shell,
            inverse_volume_shell, face_b_shells, background_shells,
            area_vector_shells,
        )
        persistent_bytes += Int64(
            (length(conservative_shell) + length(inverse_volume_shell) +
             sum(length, face_b_shells) +
             sum(shell === nothing ? 0 : length(shell)
                 for shell in background_shells) +
             sum(length, area_vector_shells)) * sizeof(FT),
        )

        if include_cell_data
            for offset in Int32(1):ct_shell_count(cell_layout)
                i, j, k = ct_shell_indices(cell_layout, offset)
                global_index = (
                    Int(block.ox) + Int(i),
                    Int(block.oy) + Int(j),
                    Int(block.oz) + Int(k),
                )
                source = ct_resolve_cell_source(
                    bid, global_index, block_dims, face_bc, connectivity,
                )
                owned = if source === nothing
                    CTDerivationOwnedCellSource(
                        Int32(world_rank), Int32(bid),
                        (
                            Int32(clamp(i,1,active_dims[1]) + Int(ng)),
                            Int32(clamp(j,1,active_dims[2]) + Int(ng)),
                            Int32(clamp(k,1,active_dims[3]) + Int(ng)),
                        ),
                    )
                else
                    ct_owned_cell_source(
                        source, block_dims, block_nprocs, rank_offsets; ng=ng,
                    )
                end
                if Int(owned.rank) == Int(world_rank)
                    offsets, indices = _ct_derivation_add_cell_job!(
                        cell_groups, bid, owned,
                    )
                    push!(offsets, offset)
                    push!(indices, owned.local_index)
                else
                    requests = get!(
                        remote_cell_requests, Int(owned.rank),
                        CTCellDestinationRequest[],
                    )
                    push!(requests, CTCellDestinationRequest(
                        owned, Int32(bid), offset,
                    ))
                end
            end
        end

        for destination_axis in 1:3
            for offset in Int32(1):ct_shell_count(face_layout)
                i, j, k = ct_shell_indices(face_layout, offset)
                global_index = (
                    Int(block.ox) + Int(i),
                    Int(block.oy) + Int(j),
                    Int(block.oz) + Int(k),
                )
                source = ct_resolve_face_source(
                    bid, destination_axis, global_index, block_dims,
                    face_bc, connectivity,
                )
                owned = if source === nothing
                    physical_face_extents = ntuple(
                        axis -> active_dims[axis] +
                                (axis == destination_axis ? 1 : 0), Val(3),
                    )
                    CTDerivationOwnedFaceSource(
                        Int32(world_rank), Int32(bid), Int8(destination_axis),
                        (
                            Int32(clamp(i,1,physical_face_extents[1]) + Int(ng)),
                            Int32(clamp(j,1,physical_face_extents[2]) + Int(ng)),
                            Int32(clamp(k,1,physical_face_extents[3]) + Int(ng)),
                        ),
                        Int8(1),
                    )
                else
                    ct_owned_face_source(
                        source, block_dims, block_nprocs, rank_offsets; ng=ng,
                    )
                end
                if Int(owned.rank) == Int(world_rank)
                    offsets, indices, orientations =
                        _ct_derivation_add_face_job!(
                            face_groups, bid, destination_axis, owned,
                        )
                    push!(offsets, offset)
                    push!(indices, owned.local_index)
                    push!(orientations, owned.orientation)
                else
                    requests = get!(
                        remote_face_requests, Int(owned.rank),
                        CTFaceDestinationRequest[],
                    )
                    push!(requests, CTFaceDestinationRequest(
                        owned, Int32(bid), Int8(destination_axis), offset,
                    ))
                end
            end
        end
    end

    cell_jobs = CTLocalCellGatherJob[]
    for ((destination_bid, source_bid), (offsets, indices)) in
        sort(collect(cell_groups); by=first)
        push!(cell_jobs, CTLocalCellGatherJob(
            destination_bid, source_bid, GPUArray(offsets),
            _ct_derivation_device_indices(indices), Int32(length(offsets)),
        ))
        persistent_bytes += Int64(
            length(offsets) * (sizeof(Int32) + 3sizeof(Int32)),
        )
    end
    face_jobs = CTLocalFaceGatherJob[]
    for ((destination_bid, destination_axis, source_bid, source_axis),
         (offsets, indices, orientations)) in
        sort(collect(face_groups); by=first)
        push!(face_jobs, CTLocalFaceGatherJob(
            destination_bid, Int8(destination_axis), source_bid,
            Int8(source_axis), GPUArray(offsets),
            _ct_derivation_device_indices(indices), GPUArray(orientations),
            Int32(length(offsets)),
        ))
        persistent_bytes += Int64(
            length(offsets) *
            (sizeof(Int32) + 3sizeof(Int32) + sizeof(Int8)),
        )
    end

    background_enabled = any(
        halo -> any(shell -> shell !== nothing,
                    halo.background_face_shells),
        values(halos),
    )
    needs_remote = !isempty(remote_cell_requests) ||
                   !isempty(remote_face_requests)
    needs_remote && communicator === nothing && throw(ArgumentError(
        "CT derivation halo has remote sources but no MPI communicator",
    ))

    if communicator === nothing
        remote_cell_send_jobs = Any[]
        remote_cell_receive_jobs = Any[]
        remote_face_send_jobs = Any[]
        remote_face_receive_jobs = Any[]
    else
        MPI.Comm_rank(communicator) == Int(world_rank) || throw(ArgumentError(
            "world_rank does not match the CT derivation communicator rank",
        ))
        background_min = MPI.Allreduce(
            Int32(background_enabled), MPI.MIN, communicator,
        )
        background_max = MPI.Allreduce(
            Int32(background_enabled), MPI.MAX, communicator,
        )
        background_min == background_max || throw(ArgumentError(
            "CT background-field shell allocation differs across MPI ranks",
        ))
        background_enabled = background_max != 0
        remote_cell_send_jobs, remote_cell_receive_jobs =
            _ct_derivation_build_remote_cell_jobs(
                remote_cell_requests, communicator,
                conservative_components,
            )
        remote_face_send_jobs, remote_face_receive_jobs =
            _ct_derivation_build_remote_face_jobs(
                remote_face_requests, communicator,
                background_enabled ? 2 : 1,
            )
    end

    for jobs in (
        remote_cell_send_jobs, remote_cell_receive_jobs,
        remote_face_send_jobs, remote_face_receive_jobs,
    )
        for job in jobs
            persistent_bytes += Int64(
                (length(job.dynamic_device) + length(job.dynamic_host) +
                 length(job.static_device) + length(job.static_host)) *
                sizeof(FT),
            )
            for subjob in job.subjobs
                persistent_bytes += Int64(
                    length(subjob.positions) * sizeof(Int32) +
                    (hasproperty(subjob, :source_indices) ?
                     length(subjob.source_indices) * sizeof(Int32) : 0) +
                    (hasproperty(subjob, :destination_offsets) ?
                     length(subjob.destination_offsets) * sizeof(Int32) : 0) +
                    (hasproperty(subjob, :orientations) ?
                     length(subjob.orientations) * sizeof(Int8) : 0),
                )
            end
        end
    end

    return CTDerivationHaloPlan(
        halos, cell_jobs, face_jobs,
        Any[remote_cell_send_jobs...], Any[remote_cell_receive_jobs...],
        Any[remote_face_send_jobs...], Any[remote_face_receive_jobs...],
        communicator, background_enabled, persistent_bytes,
    )
end

build_local_ct_derivation_halo_plan(args...; kwargs...) =
    build_ct_derivation_halo_plan(args...; communicator=nothing, kwargs...)

function ct_gather_cell_components_kernel!(
    destination, source, offsets, indices, count, components,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds begin
        offset = offsets[entry]
        i, j, k = indices[entry,1], indices[entry,2], indices[entry,3]
        for component in Int32(1):components
            destination[offset,component] = source[i,j,k,component]
        end
    end
    return
end

function ct_gather_cell_scalar_kernel!(
    destination, source, offsets, indices, count,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds begin
        offset = offsets[entry]
        destination[offset] = source[
            indices[entry,1], indices[entry,2], indices[entry,3],
        ]
    end
    return
end

function ct_gather_face_scalar_kernel!(
    destination, source, offsets, indices, orientations, count,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds begin
        offset = offsets[entry]
        destination[offset] = orientations[entry] * source[
            indices[entry,1], indices[entry,2], indices[entry,3],
        ]
    end
    return
end

function ct_gather_face_area_vector_kernel!(
    destination, area, normal_x, normal_y, normal_z,
    offsets, indices, orientations, count,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds begin
        offset = offsets[entry]
        i, j, k = indices[entry,1], indices[entry,2], indices[entry,3]
        signed_area = orientations[entry] * area[i,j,k]
        destination[offset,1] = signed_area * normal_x[i,j,k]
        destination[offset,2] = signed_area * normal_y[i,j,k]
        destination[offset,3] = signed_area * normal_z[i,j,k]
    end
    return
end

function ct_pack_face_scalar_kernel!(
    destination, destination_component, source, positions, indices, count,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds destination[positions[entry],destination_component] = source[
        indices[entry,1], indices[entry,2], indices[entry,3],
    ]
    return
end

function ct_pack_face_area_vector_kernel!(
    destination, area, normal_x, normal_y, normal_z,
    positions, indices, count,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds begin
        position = positions[entry]
        i, j, k = indices[entry,1], indices[entry,2], indices[entry,3]
        local_area = area[i,j,k]
        destination[position,1] = local_area * normal_x[i,j,k]
        destination[position,2] = local_area * normal_y[i,j,k]
        destination[position,3] = local_area * normal_z[i,j,k]
    end
    return
end

function ct_scatter_cell_components_kernel!(
    destination, source, positions, offsets, count, components,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds begin
        position = positions[entry]
        offset = offsets[entry]
        for component in Int32(1):components
            destination[offset,component] = source[position,component]
        end
    end
    return
end

function ct_scatter_cell_scalar_kernel!(
    destination, source, positions, offsets, count,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds destination[offsets[entry]] = source[positions[entry]]
    return
end

function ct_scatter_face_scalar_kernel!(
    destination, source, source_component,
    positions, offsets, orientations, count,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds destination[offsets[entry]] = orientations[entry] *
        source[positions[entry],source_component]
    return
end

function ct_scatter_face_area_vector_kernel!(
    destination, source, positions, offsets, orientations, count,
)
    entry = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    entry <= count || return
    @inbounds begin
        position = positions[entry]
        offset = offsets[entry]
        orientation = orientations[entry]
        destination[offset,1] = orientation * source[position,1]
        destination[offset,2] = orientation * source[position,2]
        destination[offset,3] = orientation * source[position,3]
    end
    return
end

function refresh_local_ct_derivation_dynamic!(plan, blocks)
    threads = 256
    for job in plan.local_cell_jobs
        destination = plan.halos[job.destination_bid].conservative_shell
        source = blocks[job.source_bid].U
        @gpu_launch threads=threads blocks=cld(job.count, threads) ct_gather_cell_components_kernel!(
            destination, source, job.offsets, job.source_indices, job.count,
            Int32(size(destination, 2)),
        )
    end
    for job in plan.local_face_jobs
        destination_halo = plan.halos[job.destination_bid]
        source_block = blocks[job.source_bid]
        destination = destination_halo.face_b_shells[job.destination_axis]
        source = _ct_derivation_face_fields(source_block)[job.source_axis]
        @gpu_launch threads=threads blocks=cld(job.count, threads) ct_gather_face_scalar_kernel!(
            destination, source, job.offsets, job.source_indices,
            job.orientations, job.count,
        )
        destination_background =
            destination_halo.background_face_shells[job.destination_axis]
        source_background =
            _ct_derivation_background_face_fields(source_block)[job.source_axis]
        if destination_background !== nothing && source_background !== nothing
            @gpu_launch threads=threads blocks=cld(job.count, threads) ct_gather_face_scalar_kernel!(
                destination_background, source_background, job.offsets,
                job.source_indices, job.orientations, job.count,
            )
        end
    end
    return nothing
end

function refresh_local_ct_derivation_static!(plan, blocks)
    threads = 256
    for job in plan.local_cell_jobs
        destination = plan.halos[job.destination_bid].inverse_volume_shell
        source = blocks[job.source_bid].Vol
        @gpu_launch threads=threads blocks=cld(job.count, threads) ct_gather_cell_scalar_kernel!(
            destination, source, job.offsets, job.source_indices, job.count,
        )
    end
    for job in plan.local_face_jobs
        destination = plan.halos[job.destination_bid].face_area_vector_shells[
            job.destination_axis
        ]
        source = _ct_derivation_face_metric_fields(
            blocks[job.source_bid], job.source_axis,
        )
        @gpu_launch threads=threads blocks=cld(job.count, threads) ct_gather_face_area_vector_kernel!(
            destination, source..., job.offsets, job.source_indices,
            job.orientations, job.count,
        )
    end
    return nothing
end

function _ct_derivation_exchange_dynamic!(plan, blocks)
    isempty(plan.remote_cell_send_jobs) &&
    isempty(plan.remote_cell_receive_jobs) &&
    isempty(plan.remote_face_send_jobs) &&
    isempty(plan.remote_face_receive_jobs) && return nothing
    communicator = plan.communicator
    communicator === nothing && error("missing CT derivation MPI communicator")
    threads = 256

    for job in plan.remote_cell_send_jobs
        for subjob in job.subjobs
            @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_gather_cell_components_kernel!(
                job.dynamic_device, blocks[subjob.source_bid].U,
                subjob.positions, subjob.source_indices, subjob.count,
                Int32(size(job.dynamic_device, 2)),
            )
        end
    end
    for job in plan.remote_face_send_jobs
        for subjob in job.subjobs
            source_block = blocks[subjob.source_bid]
            source_face = _ct_derivation_face_fields(source_block)[
                subjob.source_axis
            ]
            @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_pack_face_scalar_kernel!(
                job.dynamic_device, Int32(1), source_face,
                subjob.positions, subjob.source_indices, subjob.count,
            )
            if plan.background_enabled
                source_background = _ct_derivation_background_face_fields(
                    source_block,
                )[subjob.source_axis]
                source_background === nothing && error(
                    "missing CT background face field on block $(subjob.source_bid)",
                )
                @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_pack_face_scalar_kernel!(
                    job.dynamic_device, Int32(2), source_background,
                    subjob.positions, subjob.source_indices, subjob.count,
                )
            end
        end
    end
    gpu_sync()
    for job in plan.remote_cell_send_jobs
        copyto!(job.dynamic_host, job.dynamic_device)
    end
    for job in plan.remote_face_send_jobs
        copyto!(job.dynamic_host, job.dynamic_device)
    end

    requests = MPI.Request[]
    for job in plan.remote_cell_receive_jobs
        push!(requests, MPI.Irecv!(
            job.dynamic_host, communicator;
            source=job.peer_rank, tag=CT_DERIVATION_CELL_DYNAMIC_TAG,
        ))
    end
    for job in plan.remote_face_receive_jobs
        push!(requests, MPI.Irecv!(
            job.dynamic_host, communicator;
            source=job.peer_rank, tag=CT_DERIVATION_FACE_DYNAMIC_TAG,
        ))
    end
    for job in plan.remote_cell_send_jobs
        push!(requests, MPI.Isend(
            job.dynamic_host, communicator;
            dest=job.peer_rank, tag=CT_DERIVATION_CELL_DYNAMIC_TAG,
        ))
    end
    for job in plan.remote_face_send_jobs
        push!(requests, MPI.Isend(
            job.dynamic_host, communicator;
            dest=job.peer_rank, tag=CT_DERIVATION_FACE_DYNAMIC_TAG,
        ))
    end
    isempty(requests) || MPI.Waitall(requests)

    for job in plan.remote_cell_receive_jobs
        copyto!(job.dynamic_device, job.dynamic_host)
        for subjob in job.subjobs
            destination = plan.halos[
                subjob.destination_bid
            ].conservative_shell
            @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_scatter_cell_components_kernel!(
                destination, job.dynamic_device, subjob.positions,
                subjob.destination_offsets, subjob.count,
                Int32(size(destination, 2)),
            )
        end
    end
    for job in plan.remote_face_receive_jobs
        copyto!(job.dynamic_device, job.dynamic_host)
        for subjob in job.subjobs
            halo = plan.halos[subjob.destination_bid]
            destination = halo.face_b_shells[subjob.destination_axis]
            @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_scatter_face_scalar_kernel!(
                destination, job.dynamic_device, Int32(1), subjob.positions,
                subjob.destination_offsets, subjob.orientations, subjob.count,
            )
            if plan.background_enabled
                destination_background = halo.background_face_shells[
                    subjob.destination_axis
                ]
                destination_background === nothing && error(
                    "missing destination CT background shell",
                )
                @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_scatter_face_scalar_kernel!(
                    destination_background, job.dynamic_device, Int32(2),
                    subjob.positions, subjob.destination_offsets,
                    subjob.orientations, subjob.count,
                )
            end
        end
    end
    return nothing
end

function _ct_derivation_exchange_static!(plan, blocks)
    isempty(plan.remote_cell_send_jobs) &&
    isempty(plan.remote_cell_receive_jobs) &&
    isempty(plan.remote_face_send_jobs) &&
    isempty(plan.remote_face_receive_jobs) && return nothing
    communicator = plan.communicator
    communicator === nothing && error("missing CT derivation MPI communicator")
    threads = 256

    for job in plan.remote_cell_send_jobs
        for subjob in job.subjobs
            @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_gather_cell_scalar_kernel!(
                job.static_device, blocks[subjob.source_bid].Vol,
                subjob.positions, subjob.source_indices, subjob.count,
            )
        end
    end
    for job in plan.remote_face_send_jobs
        for subjob in job.subjobs
            source = _ct_derivation_face_metric_fields(
                blocks[subjob.source_bid], subjob.source_axis,
            )
            @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_pack_face_area_vector_kernel!(
                job.static_device, source..., subjob.positions,
                subjob.source_indices, subjob.count,
            )
        end
    end
    gpu_sync()
    for job in plan.remote_cell_send_jobs
        copyto!(job.static_host, job.static_device)
    end
    for job in plan.remote_face_send_jobs
        copyto!(job.static_host, job.static_device)
    end

    requests = MPI.Request[]
    for job in plan.remote_cell_receive_jobs
        push!(requests, MPI.Irecv!(
            job.static_host, communicator;
            source=job.peer_rank, tag=CT_DERIVATION_CELL_STATIC_TAG,
        ))
    end
    for job in plan.remote_face_receive_jobs
        push!(requests, MPI.Irecv!(
            job.static_host, communicator;
            source=job.peer_rank, tag=CT_DERIVATION_FACE_STATIC_TAG,
        ))
    end
    for job in plan.remote_cell_send_jobs
        push!(requests, MPI.Isend(
            job.static_host, communicator;
            dest=job.peer_rank, tag=CT_DERIVATION_CELL_STATIC_TAG,
        ))
    end
    for job in plan.remote_face_send_jobs
        push!(requests, MPI.Isend(
            job.static_host, communicator;
            dest=job.peer_rank, tag=CT_DERIVATION_FACE_STATIC_TAG,
        ))
    end
    isempty(requests) || MPI.Waitall(requests)

    for job in plan.remote_cell_receive_jobs
        copyto!(job.static_device, job.static_host)
        for subjob in job.subjobs
            destination = plan.halos[
                subjob.destination_bid
            ].inverse_volume_shell
            @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_scatter_cell_scalar_kernel!(
                destination, job.static_device, subjob.positions,
                subjob.destination_offsets, subjob.count,
            )
        end
    end
    for job in plan.remote_face_receive_jobs
        copyto!(job.static_device, job.static_host)
        for subjob in job.subjobs
            destination = plan.halos[
                subjob.destination_bid
            ].face_area_vector_shells[subjob.destination_axis]
            @gpu_launch threads=threads blocks=cld(subjob.count, threads) ct_scatter_face_area_vector_kernel!(
                destination, job.static_device, subjob.positions,
                subjob.destination_offsets, subjob.orientations, subjob.count,
            )
        end
    end
    return nothing
end

function refresh_ct_derivation_dynamic!(plan, blocks)
    refresh_local_ct_derivation_dynamic!(plan, blocks)
    _ct_derivation_exchange_dynamic!(plan, blocks)
    return nothing
end

function refresh_ct_derivation_static!(plan, blocks)
    refresh_local_ct_derivation_static!(plan, blocks)
    _ct_derivation_exchange_static!(plan, blocks)
    gpu_sync()
    return nothing
end

@inline function _ct_derivation_crosses_physical_face(
    i, j, k, extents, physical_faces,
)
    return (i < 1 && physical_faces[1]) ||
           (i > extents[1] && physical_faces[2]) ||
           (j < 1 && physical_faces[3]) ||
           (j > extents[2] && physical_faces[4]) ||
           (k < 1 && physical_faces[5]) ||
           (k > extents[3] && physical_faces[6])
end

function ct_materialize_inverse_volume_ghost_kernel!(
    inverse_volume, inverse_volume_shell, layout,
    nxp, nyp, nzp, physical_faces,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp + 2NG || j > nyp + 2NG || k > nzp + 2NG
        return
    end

    logical_i, logical_j, logical_k = i - NG, j - NG, k - NG
    if 1 <= logical_i <= nxp && 1 <= logical_j <= nyp &&
       1 <= logical_k <= nzp
        return
    end
    extents = (nxp, nyp, nzp)
    _ct_derivation_crosses_physical_face(
        logical_i, logical_j, logical_k, extents, physical_faces,
    ) && return

    offset = ct_shell_offset(layout, logical_i, logical_j, logical_k)
    if offset > 0
        @inbounds inverse_volume[i,j,k] = inverse_volume_shell[offset]
    end
    return
end

function ct_materialize_face_flux_ghost_kernel!(
    face_flux, face_shell, layout,
    face_axis, nxp, nyp, nzp, physical_faces,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > size(face_flux,1) || j > size(face_flux,2) ||
       k > size(face_flux,3)
        return
    end

    logical_i, logical_j, logical_k = i - NG, j - NG, k - NG
    extents = (
        nxp + (face_axis == Int32(1) ? Int32(1) : Int32(0)),
        nyp + (face_axis == Int32(2) ? Int32(1) : Int32(0)),
        nzp + (face_axis == Int32(3) ? Int32(1) : Int32(0)),
    )
    if 1 <= logical_i <= extents[1] &&
       1 <= logical_j <= extents[2] &&
       1 <= logical_k <= extents[3]
        return
    end
    _ct_derivation_crosses_physical_face(
        logical_i, logical_j, logical_k, extents, physical_faces,
    ) && return

    offset = ct_shell_offset(layout, logical_i, logical_j, logical_k)
    offset > 0 || return
    @inbounds face_flux[i,j,k] = face_shell[offset]
    return
end

function ct_materialize_face_metric_ghost_kernel!(
    area, normal_x, normal_y, normal_z, area_vector_shell, layout,
    face_axis, nxp, nyp, nzp, physical_faces,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > size(area,1) || j > size(area,2) || k > size(area,3)
        return
    end

    logical_i, logical_j, logical_k = i - NG, j - NG, k - NG
    extents = (
        nxp + (face_axis == Int32(1) ? Int32(1) : Int32(0)),
        nyp + (face_axis == Int32(2) ? Int32(1) : Int32(0)),
        nzp + (face_axis == Int32(3) ? Int32(1) : Int32(0)),
    )
    if 1 <= logical_i <= extents[1] &&
       1 <= logical_j <= extents[2] &&
       1 <= logical_k <= extents[3]
        return
    end
    _ct_derivation_crosses_physical_face(
        logical_i, logical_j, logical_k, extents, physical_faces,
    ) && return

    offset = ct_shell_offset(layout, logical_i, logical_j, logical_k)
    offset > 0 || return
    @inbounds begin
        sx = area_vector_shell[offset,1]
        sy = area_vector_shell[offset,2]
        sz = area_vector_shell[offset,3]
        magnitude = sqrt(sx*sx + sy*sy + sz*sz)
        if isfinite(magnitude) && magnitude > zero(magnitude)
            area[i,j,k] = magnitude
            normal_x[i,j,k] = sx / magnitude
            normal_y[i,j,k] = sy / magnitude
            normal_z[i,j,k] = sz / magnitude
        end
    end
    return
end

"""
Materialize the packed authoritative static geometry into the in-allocation
topological ghost cells used by active-cell POINT6 recovery. Physical-boundary
ghost geometry remains owned by the boundary/mesh path.
"""
function materialize_ct_derivation_static_ghosts!(
    plan, blocks, physical_faces_by_block,
)
    threads = (Int32(8), Int32(8), Int32(4))
    for (bid, block) in blocks
        halo = plan.halos[bid]
        physical_faces = get(
            physical_faces_by_block, bid, ntuple(_ -> false, Val(6)),
        )
        if !isempty(halo.inverse_volume_shell)
            cell_blocks = (
                cld(Int(block.Nx) + 2NG, Int(threads[1])),
                cld(Int(block.Ny) + 2NG, Int(threads[2])),
                cld(Int(block.Nz) + 2NG, Int(threads[3])),
            )
            @gpu_launch threads=threads blocks=cell_blocks ct_materialize_inverse_volume_ghost_kernel!(
                block.Vol, halo.inverse_volume_shell, halo.cell_layout,
                Int32(block.Nx), Int32(block.Ny), Int32(block.Nz), physical_faces,
            )
        end

        metric_fields = (
            (block.Areai, block.nxi, block.nyi, block.nzi),
            (block.Areaj, block.nxj, block.nyj, block.nzj),
            (block.Areak, block.nxk, block.nyk, block.nzk),
        )
        for axis in 1:3
            area = metric_fields[axis][1]
            face_blocks = (
                cld(size(area,1), Int(threads[1])),
                cld(size(area,2), Int(threads[2])),
                cld(size(area,3), Int(threads[3])),
            )
            @gpu_launch threads=threads blocks=face_blocks ct_materialize_face_metric_ghost_kernel!(
                metric_fields[axis]..., halo.face_area_vector_shells[axis],
                halo.face_layout, Int32(axis), Int32(block.Nx),
                Int32(block.Ny), Int32(block.Nz), physical_faces,
            )
        end
    end
    gpu_sync()
    return nothing
end

"""
Materialize authoritative topological face-flux shells into the existing CT
staggered ghost allocation. This completes edge/corner halos that ordinary
interface-sheet exchange cannot represent. Physical-boundary entries remain
owned by their boundary kernels.
"""
function materialize_ct_derivation_dynamic_ghosts!(
    plan, blocks, physical_faces_by_block,
)
    threads = (Int32(8), Int32(8), Int32(4))
    for (bid, block) in blocks
        halo = plan.halos[bid]
        physical_faces = get(
            physical_faces_by_block, bid, ntuple(_ -> false, Val(6)),
        )
        face_fields = _ct_derivation_face_fields(block)
        background_fields = _ct_derivation_background_face_fields(block)
        for axis in 1:3
            face_flux = face_fields[axis]
            face_blocks = (
                cld(size(face_flux,1), Int(threads[1])),
                cld(size(face_flux,2), Int(threads[2])),
                cld(size(face_flux,3), Int(threads[3])),
            )
            @gpu_launch threads=threads blocks=face_blocks ct_materialize_face_flux_ghost_kernel!(
                face_flux, halo.face_b_shells[axis], halo.face_layout,
                Int32(axis), Int32(block.Nx), Int32(block.Ny),
                Int32(block.Nz), physical_faces,
            )
            background = background_fields[axis]
            background_shell = halo.background_face_shells[axis]
            if background !== nothing && background_shell !== nothing
                @gpu_launch threads=threads blocks=face_blocks ct_materialize_face_flux_ghost_kernel!(
                    background, background_shell, halo.face_layout,
                    Int32(axis), Int32(block.Nx), Int32(block.Ny),
                    Int32(block.Nz), physical_faces,
                )
            end
        end
    end
    gpu_sync()
    return nothing
end

"""
Release static shell payloads after direct CT has materialized its face-metric
ghosts. POINT6 retains these arrays because every point recovery reads them.
"""
function release_direct_ct_derivation_static_storage!(plan)
    any(!isempty(halo.conservative_shell) for halo in values(plan.halos)) &&
        throw(ArgumentError(
            "cannot release CT derivation static storage with cell shells active",
        ))
    released = Int64(0)
    for halo in values(plan.halos)
        for values in halo.face_area_vector_shells
            released += Int64(length(values) * sizeof(FT))
        end
        halo.face_area_vector_shells = ntuple(
            _ -> gpu_zeros(FT, 0, 3), Val(3),
        )
    end
    for jobs in (
        plan.remote_cell_send_jobs, plan.remote_cell_receive_jobs,
        plan.remote_face_send_jobs, plan.remote_face_receive_jobs,
    )
        for job in jobs
            released += Int64(
                (length(job.static_device) + length(job.static_host)) *
                sizeof(FT),
            )
            job.static_device = gpu_zeros(FT, 0, 3)
            job.static_host = zeros(FT, 0, 3)
        end
    end
    plan.persistent_bytes = max(Int64(0), plan.persistent_bytes - released)
    return released
end

"""
Release the packed fixed-background payload after B0 has been materialized in
the ordinary face arrays. Subsequent stages exchange only the evolved b flux.
"""
function release_ct_derivation_background_storage!(plan)
    plan.background_enabled || return Int64(0)
    released = Int64(0)
    for halo in values(plan.halos)
        for values in halo.background_face_shells
            values === nothing && continue
            released += Int64(length(values) * sizeof(FT))
        end
        halo.background_face_shells = (nothing, nothing, nothing)
    end
    for jobs in (plan.remote_face_send_jobs, plan.remote_face_receive_jobs)
        for job in jobs
            rows = size(job.dynamic_host, 1)
            old_length = length(job.dynamic_device) + length(job.dynamic_host)
            job.dynamic_host = zeros(FT, rows, 1)
            job.dynamic_device = GPUArray(job.dynamic_host)
            new_length = length(job.dynamic_device) + length(job.dynamic_host)
            released += Int64((old_length - new_length) * sizeof(FT))
        end
    end
    plan.background_enabled = false
    plan.persistent_bytes = max(Int64(0), plan.persistent_bytes - released)
    return released
end
