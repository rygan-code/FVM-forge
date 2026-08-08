# unstruct_mpi.jl — MPI ghost cell exchange for unstructured FVM
# Part of the second-order unstructured FVM branch
#
# Shared infrastructure (must be included BEFORE this file):
#   gpu_backend.jl, physics.jl, bc_types.jl, unstruct_mesh.jl
#
# For unstructured meshes, ghost exchange is based on cell index lists
# (CSR-like) rather than structured i/j/k ranges.
#
# Design:
#   - build_partition_connectivity: called at setup, builds send/recv cell lists
#   - exchange_unstruct_ghost: called every RK substep, packs Q → sends → unpacks
#
# Supports both GPU-aware MPI and CPU-staged (D2H → MPI → H2D) modes,
# mirroring the structured mpi.jl pattern.

using MPI

const UNSTRUCT_CT_COUNT_TAG = 9910
const UNSTRUCT_CT_DESCRIPTOR_TAG = 9911
const UNSTRUCT_CT_FACE_TAG = 9912
const UNSTRUCT_CT_EDGE_OWNER_TAG = 9913
const UNSTRUCT_CT_EDGE_VALUE_TAG = 9914

@inline function _canonical_face_sign(nx, ny, nz)
    tolerance = FT(64)*eps(FT)
    abs(nx) > tolerance && return nx > zero(FT) ? Int8(1) : Int8(-1)
    abs(ny) > tolerance && return ny > zero(FT) ? Int8(1) : Int8(-1)
    abs(nz) > tolerance && return nz > zero(FT) ? Int8(1) : Int8(-1)
    error("cannot orient a degenerate MPI-CT interface face")
end

function _canonical_edge_descriptor(geometry,edge)
    node_start=geometry.edge_node_start[edge]
    node_end=geometry.edge_node_end[edge]
    start_point=(geometry.node_x[node_start],geometry.node_y[node_start],geometry.node_z[node_start])
    end_point=(geometry.node_x[node_end],geometry.node_y[node_end],geometry.node_z[node_end])
    if isless(end_point, start_point)
        return (end_point..., start_point...), Int8(-1)
    end
    return (start_point..., end_point...), Int8(1)
end

function _ct_host_geometry(block)
    ct = block.ct
    return (
        face_right=Array(block.face_R),face_area=Array(block.face_area),
        face_center=collect(zip(Array(block.face_cx),Array(block.face_cy),Array(block.face_cz))),
        face_normal=collect(zip(Array(block.face_nx),Array(block.face_ny),Array(block.face_nz))),
        face_edge_offset=Array(ct.face_edge_offset),face_edge_list=Array(ct.face_edge_list),
        edge_node_start=Array(ct.edge_node_start),edge_node_end=Array(ct.edge_node_end),
        node_x=Array(block.node_x),node_y=Array(block.node_y),node_z=Array(block.node_z),
    )
end

function _mpi_ct_interface_faces(block,neighbor_index,geometry)
    face_by_ghost = Dict{Int,Int}()
    for face in 1:block.nface
        right = geometry.face_right[face]
        right > block.ncell || continue
        haskey(face_by_ghost, right) && error(
            "MPI-CT requires one interface face per receive ghost cell",
        )
        face_by_ghost[right] = face
    end

    first_recv = block.mpi_recv_offsets[neighbor_index]
    last_recv = block.mpi_recv_offsets[neighbor_index + 1] - 1
    faces = Int[]
    for position in first_recv:last_recv
        ghost = block.mpi_recv_list[position]
        haskey(face_by_ghost, ghost) || error(
            "MPI-CT receive ghost $ghost has no boundary face",
        )
        push!(faces, face_by_ghost[ghost])
    end
    face_keys = [(
        geometry.face_center[face]...,geometry.face_area[face],
        abs.(geometry.face_normal[face])...,
    ) for face in faces]
    permutation = sortperm(face_keys)
    faces = faces[permutation]
    face_signs = [
        _canonical_face_sign(geometry.face_normal[face]...) for face in faces
    ]
    descriptors=reduce(vcat,
        (collect(face_keys[index]) for index in permutation);init=FT[])
    return faces,face_signs,descriptors
end

function _partition_ct_interface_faces(block,neighbor_index,geometry)
    metadata=block.partition
    first_face=metadata.mpi_face_offsets[neighbor_index]
    last_face=metadata.mpi_face_offsets[neighbor_index+1]-1
    first_face<=last_face || error(
        "MPI-CT neighbor $(block.mpi_neighbors[neighbor_index]) has no shared faces")
    faces=metadata.mpi_face_list[first_face:last_face]
    face_signs=metadata.mpi_face_signs[first_face:last_face]
    keys=metadata.mpi_face_keys[first_face:last_face]
    permutation=sortperm(keys)
    faces=faces[permutation]
    face_signs=face_signs[permutation]
    face_keys=[(
        geometry.face_center[face]...,geometry.face_area[face],
        abs.(geometry.face_normal[face])...,
    ) for face in faces]
    descriptors=reduce(vcat,(collect(key) for key in face_keys);init=FT[])
    return faces,face_signs,descriptors
end

function _mpi_ct_interface_edges(faces,geometry)
    edges = Int[]
    for face in faces
        append!(edges, geometry.face_edge_list[
            geometry.face_edge_offset[face]:(geometry.face_edge_offset[face+1]-1)
        ])
    end
    unique!(edges)
    edge_descriptors=Vector{NTuple{6,FT}}(undef,length(edges))
    edge_signs=Vector{Int8}(undef,length(edges))
    for index in eachindex(edges)
        edge_descriptors[index],edge_signs[index]=
            _canonical_edge_descriptor(geometry,edges[index])
    end
    permutation=sortperm(edge_descriptors)
    descriptors=reduce(vcat,
        (collect(edge_descriptors[index]) for index in permutation);init=FT[])
    return edges[permutation],edge_signs[permutation],descriptors
end

function _ct_interface_segment(block,neighbor_index,geometry)
    metadata=block.partition
    if metadata!==nothing &&
       length(metadata.mpi_face_offsets)==length(block.mpi_neighbors)+1
        faces,face_signs,face_descriptors=
            _partition_ct_interface_faces(block,neighbor_index,geometry)
    else
        faces,face_signs,face_descriptors=
            _mpi_ct_interface_faces(block,neighbor_index,geometry)
    end
    edges,edge_signs,edge_descriptors=_mpi_ct_interface_edges(faces,geometry)
    segment=UnstructCTMPISegment(
        block.mpi_neighbors[neighbor_index],faces,face_signs,edges,edge_signs)
    descriptors=vcat(face_descriptors,edge_descriptors)
    return segment, descriptors
end

function _exchange_ct_plan_metadata!(segments, descriptors, comm_cart)
    count_receives = [zeros(Int, 2) for _ in segments]
    count_sends = [[length(segment.faces), length(segment.edges)] for segment in segments]
    requests = MPI.Request[]
    for index in eachindex(segments)
        neighbor = segments[index].neighbor
        push!(requests, MPI.Irecv!(count_receives[index], comm_cart;
            source=neighbor, tag=UNSTRUCT_CT_COUNT_TAG))
        push!(requests, MPI.Isend(count_sends[index], comm_cart;
            dest=neighbor, tag=UNSTRUCT_CT_COUNT_TAG))
    end
    isempty(requests) || MPI.Waitall(requests)
    for index in eachindex(segments)
        count_receives[index] == count_sends[index] || error(
            "MPI-CT topology count mismatch with rank $(segments[index].neighbor): " *
            "local=$(count_sends[index]), remote=$(count_receives[index])",
        )
    end

    descriptor_receives = [similar(values) for values in descriptors]
    empty!(requests)
    for index in eachindex(segments)
        neighbor = segments[index].neighbor
        push!(requests, MPI.Irecv!(descriptor_receives[index], comm_cart;
            source=neighbor, tag=UNSTRUCT_CT_DESCRIPTOR_TAG))
        push!(requests, MPI.Isend(descriptors[index], comm_cart;
            dest=neighbor, tag=UNSTRUCT_CT_DESCRIPTOR_TAG))
    end
    isempty(requests) || MPI.Waitall(requests)
    for index in eachindex(segments)
        local_values = descriptors[index]
        remote_values = descriptor_receives[index]
        scale = max(maximum(abs, local_values; init=zero(FT)), one(FT))
        tolerance = FT(128)*sqrt(eps(FT))*scale
        all(isapprox.(local_values, remote_values; atol=tolerance, rtol=tolerance)) ||
            error("MPI-CT geometry mismatch with rank $(segments[index].neighbor)")
    end
    return
end

function setup_unstruct_ct_mpi_plan!(block::UnstructBlock, comm_cart)
    block.ct === nothing && error("initialize CT topology before MPI-CT setup")
    block.ct_mpi_plan === nothing || return block.ct_mpi_plan
    isempty(block.mpi_neighbors) && return nothing
    allunique(block.mpi_neighbors) || error(
        "MPI-CT currently requires one exchange segment per neighbor rank",
    )

    segments = UnstructCTMPISegment[]
    descriptors = Vector{FT}[]
    geometry=_ct_host_geometry(block)
    for neighbor_index in eachindex(block.mpi_neighbors)
        segment, descriptor = _ct_interface_segment(block,neighbor_index,geometry)
        push!(segments, segment)
        push!(descriptors, descriptor)
    end
    _exchange_ct_plan_metadata!(segments, descriptors, comm_cart)
    passes = max(MPI.Comm_size(comm_cart) - 1, 1)
    block.ct_mpi_plan = UnstructCTMPIPlan(segments, passes)
    return block.ct_mpi_plan
end

function sync_unstruct_ct_face_flux!(block::UnstructBlock, comm_cart;
                                     update_backup::Bool=false)
    plan = setup_unstruct_ct_mpi_plan!(block, comm_cart)
    plan === nothing && return
    rank = MPI.Comm_rank(comm_cart)
    phi = Array(block.ct.phi_faces)
    sends = [
        FT[sign*phi[face] for (face, sign) in zip(segment.faces, segment.face_signs)]
        for segment in plan.segments
    ]
    receives = [similar(values) for values in sends]
    requests = MPI.Request[]
    for index in eachindex(plan.segments)
        neighbor = plan.segments[index].neighbor
        push!(requests, MPI.Irecv!(receives[index], comm_cart;
            source=neighbor, tag=UNSTRUCT_CT_FACE_TAG))
        push!(requests, MPI.Isend(sends[index], comm_cart;
            dest=neighbor, tag=UNSTRUCT_CT_FACE_TAG))
    end
    isempty(requests) || MPI.Waitall(requests)
    for index in eachindex(plan.segments)
        segment = plan.segments[index]
        canonical = rank < segment.neighbor ? sends[index] : receives[index]
        for (position, face) in enumerate(segment.faces)
            phi[face] = segment.face_signs[position]*canonical[position]
        end
    end
    copyto!(block.ct.phi_faces, phi)
    update_backup && copyto!(block.ct.phi_backup, phi)
    return
end

function _exchange_ct_edge_owner_pass!(plan,owners,canonical,comm_cart)
    owner_sends=[[owners[edge] for edge in segment.edges] for segment in plan.segments]
    value_sends=[[canonical[edge] for edge in segment.edges] for segment in plan.segments]
    owner_receives=[similar(values) for values in owner_sends]
    value_receives=[similar(values) for values in value_sends]
    requests=MPI.Request[]
    for index in eachindex(plan.segments)
        neighbor=plan.segments[index].neighbor
        push!(requests,MPI.Irecv!(owner_receives[index],comm_cart;
            source=neighbor,tag=UNSTRUCT_CT_EDGE_OWNER_TAG))
        push!(requests,MPI.Isend(owner_sends[index],comm_cart;
            dest=neighbor,tag=UNSTRUCT_CT_EDGE_OWNER_TAG))
        push!(requests,MPI.Irecv!(value_receives[index],comm_cart;
            source=neighbor,tag=UNSTRUCT_CT_EDGE_VALUE_TAG))
        push!(requests,MPI.Isend(value_sends[index],comm_cart;
            dest=neighbor,tag=UNSTRUCT_CT_EDGE_VALUE_TAG))
    end
    isempty(requests) || MPI.Waitall(requests)
    for index in eachindex(plan.segments)
        for (position,edge) in enumerate(plan.segments[index].edges)
            remote_owner=owner_receives[index][position]
            if remote_owner<owners[edge]
                owners[edge]=remote_owner
                canonical[edge]=value_receives[index][position]
            end
        end
    end
    return
end

function _store_canonical_edge_emf!(edge_emf,plan,canonical)
    for segment in plan.segments
        for (edge,sign) in zip(segment.edges,segment.edge_signs)
            edge_emf[edge]=sign*canonical[edge]
        end
    end
    return
end

function sync_unstruct_ct_edge_emf!(block::UnstructBlock,comm_cart)
    plan=setup_unstruct_ct_mpi_plan!(block,comm_cart)
    plan===nothing && return
    rank=MPI.Comm_rank(comm_cart)
    edge_emf=Array(block.ct.edge_emf)
    owners=Dict(edge=>rank for segment in plan.segments for edge in segment.edges)
    canonical=Dict(edge=>sign*edge_emf[edge] for segment in plan.segments
        for (edge,sign) in zip(segment.edges,segment.edge_signs))
    for _ in 1:plan.propagation_passes
        _exchange_ct_edge_owner_pass!(plan,owners,canonical,comm_cart)
    end
    _store_canonical_edge_emf!(edge_emf,plan,canonical)
    copyto!(block.ct.edge_emf, edge_emf)
    return
end

# ═════════════════════════════════════════════════════════════
# Partition connectivity builder
# ═════════════════════════════════════════════════════════════
# For single-block single-rank: no-op (empty lists)
# For multi-rank: needs to identify cells on partition boundaries
# and their corresponding ghost cells on the neighbor rank.
#
# This is called during mesh loading. For gen_cartesian_unstruct
# with a single rank, the lists are empty.

"""
    setup_mpi_buffers!(block, comm_cart)

Allocate MPI send/recv buffers based on mpi_send_list / mpi_recv_list.
Called once after mesh loading. If lists are empty (single-rank), this is a no-op.
"""
function setup_mpi_buffers!(block::UnstructBlock, gpu_aware::Bool)
    ns = length(block.mpi_send_list)
    nr = length(block.mpi_recv_list)
    if ns == 0 && nr == 0
        return  # single-rank, no exchange needed
    end

    # Allocate for the largest payload: all three primitive gradients.
    buffer_width = Nprim * 3
    block.sbuf_h = zeros(FT, ns * buffer_width)
    block.rbuf_h = zeros(FT, nr * buffer_width)
    if gpu_aware
        block.sbuf_d = gpu_zeros(FT, ns * buffer_width)
        block.rbuf_d = gpu_zeros(FT, nr * buffer_width)
    else
        block.sbuf_d = gpu_zeros(FT, ns * buffer_width)  # still need for pack kernel
        block.rbuf_d = gpu_zeros(FT, nr * buffer_width)
    end
    return
end

function unstruct_pack_gradient_kernel!(sbuf, grad, send_list, nsend::Int)
    idx = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    idx > nsend && return
    @inbounds cell = send_list[idx]
    @inbounds for variable in 1:Nprim, direction in 1:3
        offset = (idx - Int32(1))*(Nprim*3) + (variable - 1)*3 + direction
        sbuf[offset] = grad[cell,variable,direction]
    end
    return
end

function unstruct_unpack_gradient_kernel!(grad, rbuf, recv_list, nrecv::Int)
    idx = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    idx > nrecv && return
    @inbounds cell = recv_list[idx]
    @inbounds for variable in 1:Nprim, direction in 1:3
        offset = (idx - Int32(1))*(Nprim*3) + (variable - 1)*3 + direction
        grad[cell,variable,direction] = rbuf[offset]
    end
    return
end

function exchange_unstruct_gradient(block::UnstructBlock, comm_cart;
                                    gpu_aware::Bool=false)
    ns = length(block.mpi_send_list)
    nr = length(block.mpi_recv_list)
    (ns == 0 && nr == 0) && return

    width = Nprim*3
    nthreads = 256
    ns > 0 && @gpu_launch threads=nthreads blocks=cld(ns,nthreads) unstruct_pack_gradient_kernel!(
        block.sbuf_d, block.grad, block.mpi_send_list, ns,
    )
    gpu_sync()
    if !gpu_aware && ns > 0
        copyto!(block.sbuf_h, block.sbuf_d)
    end

    send_buffer = gpu_aware ? block.sbuf_d : block.sbuf_h
    recv_buffer = gpu_aware ? block.rbuf_d : block.rbuf_h
    requests = MPI.Request[]
    for neighbor_index in eachindex(block.mpi_neighbors)
        neighbor = block.mpi_neighbors[neighbor_index]
        send_first = (block.mpi_send_offsets[neighbor_index] - 1)*width + 1
        send_last = (block.mpi_send_offsets[neighbor_index + 1] - 1)*width
        recv_first = (block.mpi_recv_offsets[neighbor_index] - 1)*width + 1
        recv_last = (block.mpi_recv_offsets[neighbor_index + 1] - 1)*width
        push!(requests, MPI.Irecv!(view(recv_buffer,recv_first:recv_last), comm_cart;
            source=neighbor, tag=9902))
        push!(requests, MPI.Isend(view(send_buffer,send_first:send_last), comm_cart;
            dest=neighbor, tag=9902))
    end
    isempty(requests) || MPI.Waitall(requests)

    if nr > 0
        if !gpu_aware
            copyto!(block.rbuf_d, block.rbuf_h)
        end
        @gpu_launch threads=nthreads blocks=cld(nr,nthreads) unstruct_unpack_gradient_kernel!(
            block.grad, block.rbuf_d, block.mpi_recv_list, nr,
        )
    end
    return
end

function unstruct_pack_limiter_kernel!(sbuf,limiter,send_list,nsend::Int)
    index=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    index>nsend && return
    @inbounds cell=send_list[index]
    @inbounds for variable in 1:Nprim
        sbuf[(index-Int32(1))*Nprim+variable]=limiter[cell,variable]
    end
    return
end

function unstruct_unpack_limiter_kernel!(limiter,rbuf,recv_list,nrecv::Int)
    index=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    index>nrecv && return
    @inbounds cell=recv_list[index]
    @inbounds for variable in 1:Nprim
        limiter[cell,variable]=rbuf[(index-Int32(1))*Nprim+variable]
    end
    return
end

function exchange_unstruct_limiter(block::UnstructBlock,comm_cart;
                                   gpu_aware::Bool=false)
    nsend=length(block.mpi_send_list)
    nrecv=length(block.mpi_recv_list)
    (nsend==0 && nrecv==0) && return
    threads=256
    if nsend>0
        @gpu_launch threads=threads blocks=cld(nsend,threads) unstruct_pack_limiter_kernel!(
            block.sbuf_d,block.reconstruction_limiter,block.mpi_send_list,nsend)
    end
    gpu_sync()
    if !gpu_aware && nsend>0
        copyto!(block.sbuf_h,block.sbuf_d)
    end
    send_buffer=gpu_aware ? block.sbuf_d : block.sbuf_h
    recv_buffer=gpu_aware ? block.rbuf_d : block.rbuf_h
    requests=MPI.Request[]
    for neighbor_index in eachindex(block.mpi_neighbors)
        neighbor=block.mpi_neighbors[neighbor_index]
        send_first=(block.mpi_send_offsets[neighbor_index]-1)*Nprim+1
        send_last=(block.mpi_send_offsets[neighbor_index+1]-1)*Nprim
        recv_first=(block.mpi_recv_offsets[neighbor_index]-1)*Nprim+1
        recv_last=(block.mpi_recv_offsets[neighbor_index+1]-1)*Nprim
        push!(requests,MPI.Irecv!(view(recv_buffer,recv_first:recv_last),comm_cart;
            source=neighbor,tag=9903))
        push!(requests,MPI.Isend(view(send_buffer,send_first:send_last),comm_cart;
            dest=neighbor,tag=9903))
    end
    isempty(requests) || MPI.Waitall(requests)
    if nrecv>0
        if !gpu_aware
            copyto!(block.rbuf_d,block.rbuf_h)
        end
        @gpu_launch threads=threads blocks=cld(nrecv,threads) unstruct_unpack_limiter_kernel!(
            block.reconstruction_limiter,block.rbuf_d,block.mpi_recv_list,nrecv)
    end
    return
end

# ═════════════════════════════════════════════════════════════
# Pack kernel: extract Q[send_cells] → sbuf_d
# ═════════════════════════════════════════════════════════════

function unstruct_pack_kernel!(sbuf, Q, send_list, nsend::Int)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx > nsend
        return
    end
    c = send_list[idx]
    @inbounds for v in 1:Nprim
        sbuf[(idx - Int32(1)) * Nprim + v] = Q[c, v]
    end
    return
end

# ═════════════════════════════════════════════════════════════
# Unpack kernel: rbuf_d → Q[recv_cells]
# ═════════════════════════════════════════════════════════════

function unstruct_unpack_kernel!(Q, rbuf, recv_list, nrecv::Int)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx > nrecv
        return
    end
    c = recv_list[idx]
    @inbounds for v in 1:Nprim
        Q[c, v] = rbuf[(idx - Int32(1)) * Nprim + v]
    end
    return
end

# ═════════════════════════════════════════════════════════════
# Ghost exchange (main entry point)
# ═════════════════════════════════════════════════════════════
# Mirrors mpi.jl's exchange_ghost pattern:
#   1. Pack Q[send_cells] → sbuf_d (GPU kernel)
#   2. gpu_sync
#   3. D2H: sbuf_d → sbuf_h (unless GPU-aware)
#   4. MPI Isend/Irecv
#   5. MPI Waitall
#   6. H2D: rbuf_h → rbuf_d (unless GPU-aware)
#   7. Unpack rbuf_d → Q[recv_cells] (GPU kernel)
#
# After this, physical BC ghosts are filled by fill_unstruct_ghost! (separate call).

function exchange_unstruct_ghost(block::UnstructBlock, comm_cart;
                                  gpu_aware::Bool=false, compute_fn=nothing)
    ns = length(block.mpi_send_list)
    nr = length(block.mpi_recv_list)
    if ns == 0 && nr == 0
        return  # single-rank, no exchange
    end

    nneighbor = length(block.mpi_neighbors)
    if nneighbor == 0
        return
    end
    length(block.mpi_send_offsets) == nneighbor + 1 || error(
        "invalid unstructured MPI send offsets",
    )
    length(block.mpi_recv_offsets) == nneighbor + 1 || error(
        "invalid unstructured MPI receive offsets",
    )

    nthreads = 256
    nbs = cld(ns, nthreads)
    nbr = cld(nr, nthreads)

    # Step 1: Pack (GPU kernel)
    if ns > 0
        @gpu_launch threads=nthreads blocks=nbs unstruct_pack_kernel!(
            block.sbuf_d, block.Q, block.mpi_send_list, ns)
    end

    # Step 2: Sync
    gpu_sync()

    # Step 3: D2H (skip if GPU-aware)
    if !gpu_aware && ns > 0
        copyto!(block.sbuf_h, block.sbuf_d)
    end

    # Step 4: Post one matched send/receive segment per neighbor.
    reqs = MPI.Request[]
    send_buffer = gpu_aware ? block.sbuf_d : block.sbuf_h
    recv_buffer = gpu_aware ? block.rbuf_d : block.rbuf_h
    for neighbor_index in eachindex(block.mpi_neighbors)
        neighbor = block.mpi_neighbors[neighbor_index]
        send_first = (block.mpi_send_offsets[neighbor_index] - 1) * Nprim + 1
        send_last = (block.mpi_send_offsets[neighbor_index + 1] - 1) * Nprim
        recv_first = (block.mpi_recv_offsets[neighbor_index] - 1) * Nprim + 1
        recv_last = (block.mpi_recv_offsets[neighbor_index + 1] - 1) * Nprim
        send_last >= send_first || error(
            "empty unstructured MPI send segment for rank $neighbor",
        )
        recv_last >= recv_first || error(
            "empty unstructured MPI receive segment for rank $neighbor",
        )
        push!(reqs, MPI.Irecv!(
            view(recv_buffer, recv_first:recv_last), comm_cart;
            source=neighbor, tag=9901,
        ))
        push!(reqs, MPI.Isend(
            view(send_buffer, send_first:send_last), comm_cart;
            dest=neighbor, tag=9901,
        ))
    end

    # Step 5: Optional compute overlap during MPI wait
    if compute_fn !== nothing
        compute_fn(1)
    end

    # Step 6: Wait
    if length(reqs) > 0
        MPI.Waitall(reqs)
    end

    # Step 7: H2D + Unpack (skip H2D if GPU-aware)
    if nr > 0
        if !gpu_aware
            copyto!(block.rbuf_d, block.rbuf_h)
        end
        @gpu_launch threads=nthreads blocks=nbr unstruct_unpack_kernel!(
            block.Q, block.rbuf_d, block.mpi_recv_list, nr)
    end

    return
end

# ═════════════════════════════════════════════════════════════
# Build partition connectivity for Cartesian mesh partitioned along x
# ═════════════════════════════════════════════════════════════
# For gen_cartesian_unstruct, if we partition nx cells into nprocs along x:
#   - Each rank owns cells [ox+1 .. ox+nx_local]
#   - Send cells = last NG cells of this rank → become ghost on neighbor
#   - Recv cells = first NG ghost cells from neighbor
#
# This is a simplified 1D partition. For general unstructured meshes,
# a proper partitioner (METIS-like) would be needed.

function gen_cartesian_unstruct_partitioned(
    rank::Int, nprocs::Int, nx_global::Int, ny::Int, nz::Int,
    x0::FT, x1::FT, y0::FT, y1::FT, z0::FT, z1::FT,
    bc_xlo::Int, bc_xhi::Int, bc_ylo::Int, bc_yhi::Int,
    bc_zlo::Int, bc_zhi::Int,
)
    0 <= rank < nprocs || throw(ArgumentError("invalid MPI rank $rank"))
    nx_global >= nprocs || throw(ArgumentError(
        "nx_global=$nx_global must be at least nprocs=$nprocs",
    ))
    nx_base, nx_remainder = divrem(nx_global, nprocs)
    nx_local = nx_base + (rank < nx_remainder ? 1 : 0)
    cell_offset = rank*nx_base + min(rank, nx_remainder)
    dx = (x1 - x0) / nx_global
    local_x0 = x0 + cell_offset*dx
    local_x1 = local_x0 + nx_local*dx

    local_bc_xlo = rank == 0 ? bc_xlo : Int(BC_INTERBLOCK)
    local_bc_xhi = rank == nprocs-1 ? bc_xhi : Int(BC_INTERBLOCK)
    block = gen_cartesian_unstruct(
        nx_local, ny, nz,
        local_x0, local_x1, y0, y1, z0, z1,
        local_bc_xlo, local_bc_xhi,
        bc_ylo, bc_yhi, bc_zlo, bc_zhi,
    )
    return block, nx_local, cell_offset
end

"""
    build_cartesian_partition(block, comm_cart, rank, nprocs, nx_global, ny, nz, NG_layer)

Build MPI send/recv lists for a 1D partition of a Cartesian mesh.
Assumes the mesh was generated for this rank's local portion. The current
unstructured mesh stores one ghost cell per boundary face, so `NG_layer=1`
is required.
"""
function build_cartesian_partition!(block::UnstructBlock, comm_cart,
                                     rank::Int, nprocs::Int,
                                     nx_local::Int, ny::Int, nz::Int,
                                     NG_layer::Int)
    NG_layer == 1 || throw(ArgumentError(
        "Cartesian unstructured partitions currently require NG_layer=1",
    ))
    low_neighbor, high_neighbor = MPI.Cart_shift(comm_cart, 0, 1)

    neighbors = Int[]
    send_list = Int[]
    recv_list = Int[]
    send_offsets = Int[1]
    recv_offsets = Int[1]

    # gen_cartesian_unstruct orders boundary ghosts as x-low, x-high,
    # y-low, y-high, z-low, z-high.
    xlow_ghost_start = block.ncell + 1
    xhigh_ghost_start = xlow_ghost_start + ny*nz

    if low_neighbor != MPI.PROC_NULL
        push!(neighbors, low_neighbor)
        for k in 1:nz, j in 1:ny, i in 1:NG_layer
            c = i + (j-1)*nx_local + (k-1)*nx_local*ny
            push!(send_list, c)
        end
        for i in 1:(NG_layer * ny * nz)
            push!(recv_list, xlow_ghost_start + i - 1)
        end
        push!(send_offsets, length(send_list) + 1)
        push!(recv_offsets, length(recv_list) + 1)
    end

    if high_neighbor != MPI.PROC_NULL
        push!(neighbors, high_neighbor)
        for k in 1:nz, j in 1:ny,
            i in (nx_local-NG_layer+1):nx_local
            c = i + (j-1)*nx_local + (k-1)*nx_local*ny
            push!(send_list, c)
        end
        for i in 1:(NG_layer * ny * nz)
            push!(recv_list, xhigh_ghost_start + i - 1)
        end
        push!(send_offsets, length(send_list) + 1)
        push!(recv_offsets, length(recv_list) + 1)
    end

    block.mpi_neighbors = neighbors
    block.mpi_send_offsets = send_offsets
    block.mpi_recv_offsets = recv_offsets
    block.mpi_send_list = send_list
    block.mpi_recv_list = recv_list
    setup_mpi_buffers!(block, false)  # CPU-staged by default
    return
end

# ═════════════════════════════════════════════════════════════
# Unified ghost sync: MPI exchange + physical BC fill
# ═════════════════════════════════════════════════════════════

"""
    sync_unstruct_ghost!(block, comm_cart; gpu_aware=false, compute_fn=nothing)

Complete ghost cell synchronization:
  1. Exchange MPI partition ghost cells
  2. Fill physical boundary ghost cells (wall, inflow, etc.)
"""
function sync_unstruct_ghost!(block::UnstructBlock, comm_cart;
                              gpu_aware::Bool=false, compute_fn=nothing)
    # Step 1: MPI exchange (inter-rank ghosts)
    exchange_unstruct_ghost(block, comm_cart; gpu_aware=gpu_aware, compute_fn=compute_fn)
    # Step 2: Physical BC fill (boundary ghosts)
    fill_unstruct_ghost!(block)
    return
end
