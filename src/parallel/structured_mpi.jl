# MPI ghost cell exchange - supports both GPU-aware and CPU-staged modes
include(joinpath(@__DIR__, "structured_face_exchange.jl"))

# MPI Cartesian periods are defined for the whole block communicator, while a
# block may have an inter-block face on only one side of a coordinate axis.
# Suppress only that outer wrap without changing the communicator, so genuine
# physical periodic faces continue to use MPI exchange.
function _structured_masked_cart_shift(
    comm_cart, direction, displacement, periodic_faces, rank_coords, rank_dims,
)
    src, dst = MPI.Cart_shift(comm_cart, direction, displacement)
    axis = direction + 1
    low_face = 2 * axis - 1
    high_face = 2 * axis
    coord = Int(rank_coords[axis])
    dim = Int(rank_dims[axis])

    if displacement > 0
        if coord == 0 && !periodic_faces[low_face]
            src = MPI.PROC_NULL
        end
        if coord == dim - 1 && !periodic_faces[high_face]
            dst = MPI.PROC_NULL
        end
    else
        if coord == 0 && !periodic_faces[low_face]
            dst = MPI.PROC_NULL
        end
        if coord == dim - 1 && !periodic_faces[high_face]
            src = MPI.PROC_NULL
        end
    end
    return src, dst
end

# Phase B optimization: x+/x- use non-blocking Isend/Irecv with separate buffers
# so both directions execute concurrently, saving 1 gpu_sync + MPI overlap.
# compute_fn(slot::Int) is called with slot=1 during MPI Waitall for comm-compute overlap.
function exchange_ghost(Q, NV, comm_cart, nxp, nyp, nzp,
    sbuf_hx, sbuf_dx, rbuf_hx, rbuf_dx,
    sbuf_hy, sbuf_dy, rbuf_hy, rbuf_dy,
    sbuf_hz, sbuf_dz, rbuf_hz, rbuf_dz;
    sbuf_hx2=nothing, sbuf_dx2=nothing, rbuf_hx2=nothing, rbuf_dx2=nothing,
    compute_fn=nothing,
    periodic_faces=(true, true, true, true, true, true),
    rank_coords=(0, 0, 0), rank_dims=(1, 1, 1))
    
    nthreadsx = (NG, 8, 8)
    nthreadsy = (8, NG, 8)
    nthreadsz = (8, 8, NG)

    nblocksx = (1, cld((nyp + 2 * NG), 8), cld((nzp + 2 * NG), 8))
    nblocksy = (cld((nxp + 2 * NG), 8), 1, cld((nzp + 2 * NG), 8))
    nblocksz = (cld((nxp + 2 * NG), 8), cld((nyp + 2 * NG), 8), 1)

    # ═══ x+/x- pipelined non-blocking exchange ═══
    src_xp, dst_xp = _structured_masked_cart_shift(
        comm_cart, 0, 1, periodic_faces, rank_coords, rank_dims,
    )
    src_xm, dst_xm = _structured_masked_cart_shift(
        comm_cart, 0, -1, periodic_faces, rank_coords, rank_dims,
    )
    has_xp = (src_xp != MPI.PROC_NULL || dst_xp != MPI.PROC_NULL)
    has_xm = (src_xm != MPI.PROC_NULL || dst_xm != MPI.PROC_NULL)

    # Use pipelined path only when second x-buffers are available
    use_pipeline = (sbuf_dx2 !== nothing) && (has_xp || has_xm) && !gpu_aware_mpi

    if use_pipeline
        # Step 1: Launch both pack kernels (no sync between them)
        if has_xp && dst_xp != MPI.PROC_NULL
            @gpu_launch threads=nthreadsx blocks=nblocksx pack_R(sbuf_dx, Q, NV, nxp, nyp, nzp)
        end
        if has_xm && dst_xm != MPI.PROC_NULL
            @gpu_launch threads=nthreadsx blocks=nblocksx pack_L(sbuf_dx2, Q, NV, nxp, nyp, nzp)
        end

        # Step 2: Single gpu_sync for both packs
        gpu_sync()

        # Step 3: D2H both directions
        if has_xp && dst_xp != MPI.PROC_NULL; copyto!(sbuf_hx, sbuf_dx); end
        if has_xm && dst_xm != MPI.PROC_NULL; copyto!(sbuf_hx2, sbuf_dx2); end

        # Step 4: Non-blocking MPI sends/receives (both directions simultaneously)
        reqs = MPI.Request[]
        if has_xp
            push!(reqs, MPI.Isend(sbuf_hx, comm_cart; dest=dst_xp, tag=9901))
            push!(reqs, MPI.Irecv!(rbuf_hx, comm_cart; source=src_xp, tag=9901))
        end
        if has_xm
            push!(reqs, MPI.Isend(sbuf_hx2, comm_cart; dest=dst_xm, tag=9902))
            push!(reqs, MPI.Irecv!(rbuf_hx2, comm_cart; source=src_xm, tag=9902))
        end

        # Step 5: Launch interior compute during MPI wait
        if compute_fn !== nothing; compute_fn(1); end

        # Step 6: Wait for all MPI to complete
        MPI.Waitall(reqs)

        # Step 7: H2D + unpack both directions
        if has_xp && src_xp != MPI.PROC_NULL
            copyto!(rbuf_dx, rbuf_hx)
            @gpu_launch threads=nthreadsx blocks=nblocksx unpack_L(rbuf_dx, Q, NV, nxp, nyp, nzp)
        end
        if has_xm && src_xm != MPI.PROC_NULL
            copyto!(rbuf_dx2, rbuf_hx2)
            @gpu_launch threads=nthreadsx blocks=nblocksx unpack_R(rbuf_dx2, Q, NV, nxp, nyp, nzp)
        end
    else
        # Fallback: original serial path (GPU-aware MPI or no second buffer)
        # x+
        if has_xp
            if dst_xp != MPI.PROC_NULL
                @gpu_launch threads=nthreadsx blocks=nblocksx pack_R(sbuf_dx, Q, NV, nxp, nyp, nzp)
            end
            if gpu_aware_mpi
                gpu_sync()
                MPI.Sendrecv!(sbuf_dx, rbuf_dx, comm_cart; dest=dst_xp, source=src_xp)
            else
                if dst_xp != MPI.PROC_NULL; copyto!(sbuf_hx, sbuf_dx); end
                if compute_fn !== nothing; compute_fn(1); end
                MPI.Sendrecv!(sbuf_hx, rbuf_hx, comm_cart; dest=dst_xp, source=src_xp)
                if src_xp != MPI.PROC_NULL; copyto!(rbuf_dx, rbuf_hx); end
            end
            if src_xp != MPI.PROC_NULL
                @gpu_launch threads=nthreadsx blocks=nblocksx unpack_L(rbuf_dx, Q, NV, nxp, nyp, nzp)
            end
        end

        # x-
        if has_xm
            if dst_xm != MPI.PROC_NULL
                @gpu_launch threads=nthreadsx blocks=nblocksx pack_L(sbuf_dx, Q, NV, nxp, nyp, nzp)
            end
            if gpu_aware_mpi
                gpu_sync()
                MPI.Sendrecv!(sbuf_dx, rbuf_dx, comm_cart; dest=dst_xm, source=src_xm)
            else
                if dst_xm != MPI.PROC_NULL; copyto!(sbuf_hx, sbuf_dx); end
                if compute_fn !== nothing; compute_fn(2); end
                MPI.Sendrecv!(sbuf_hx, rbuf_hx, comm_cart; dest=dst_xm, source=src_xm)
                if src_xm != MPI.PROC_NULL; copyto!(rbuf_dx, rbuf_hx); end
            end
            if src_xm != MPI.PROC_NULL
                @gpu_launch threads=nthreadsx blocks=nblocksx unpack_R(rbuf_dx, Q, NV, nxp, nyp, nzp)
            end
        end
    end

    # y+ (unchanged — typically PROC_NULL with (N,1,1) partition)
    src, dst = _structured_masked_cart_shift(
        comm_cart, 1, 1, periodic_faces, rank_coords, rank_dims,
    )
    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @gpu_launch threads=nthreadsy blocks=nblocksy pack_U(sbuf_dy, Q, NV, nxp, nyp, nzp)
        end
        if gpu_aware_mpi
            gpu_sync()
            MPI.Sendrecv!(sbuf_dy, rbuf_dy, comm_cart; dest=dst, source=src)
        else
            if dst != MPI.PROC_NULL; copyto!(sbuf_hy, sbuf_dy); end
            MPI.Sendrecv!(sbuf_hy, rbuf_hy, comm_cart; dest=dst, source=src)
            if src != MPI.PROC_NULL; copyto!(rbuf_dy, rbuf_hy); end
        end
        if src != MPI.PROC_NULL
            @gpu_launch threads=nthreadsy blocks=nblocksy unpack_D(rbuf_dy, Q, NV, nxp, nyp, nzp)
        end
    end

    # y-
    src, dst = _structured_masked_cart_shift(
        comm_cart, 1, -1, periodic_faces, rank_coords, rank_dims,
    )
    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @gpu_launch threads=nthreadsy blocks=nblocksy pack_D(sbuf_dy, Q, NV, nxp, nyp, nzp)
        end
        if gpu_aware_mpi
            gpu_sync()
            MPI.Sendrecv!(sbuf_dy, rbuf_dy, comm_cart; dest=dst, source=src)
        else
            if dst != MPI.PROC_NULL; copyto!(sbuf_hy, sbuf_dy); end
            MPI.Sendrecv!(sbuf_hy, rbuf_hy, comm_cart; dest=dst, source=src)
            if src != MPI.PROC_NULL; copyto!(rbuf_dy, rbuf_hy); end
        end
        if src != MPI.PROC_NULL
            @gpu_launch threads=nthreadsy blocks=nblocksy unpack_U(rbuf_dy, Q, NV, nxp, nyp, nzp)
        end
    end

    # z+
    src, dst = _structured_masked_cart_shift(
        comm_cart, 2, 1, periodic_faces, rank_coords, rank_dims,
    )
    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @gpu_launch threads=nthreadsz blocks=nblocksz pack_F(sbuf_dz, Q, NV, nxp, nyp, nzp)
        end
        if gpu_aware_mpi
            gpu_sync()
            MPI.Sendrecv!(sbuf_dz, rbuf_dz, comm_cart; dest=dst, source=src)
        else
            if dst != MPI.PROC_NULL; copyto!(sbuf_hz, sbuf_dz); end
            MPI.Sendrecv!(sbuf_hz, rbuf_hz, comm_cart; dest=dst, source=src)
            if src != MPI.PROC_NULL; copyto!(rbuf_dz, rbuf_hz); end
        end
        if src != MPI.PROC_NULL
            @gpu_launch threads=nthreadsz blocks=nblocksz unpack_B(rbuf_dz, Q, NV, nxp, nyp, nzp)
        end
    end

    # z-
    src, dst = _structured_masked_cart_shift(
        comm_cart, 2, -1, periodic_faces, rank_coords, rank_dims,
    )
    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @gpu_launch threads=nthreadsz blocks=nblocksz pack_B(sbuf_dz, Q, NV, nxp, nyp, nzp)
        end
        if gpu_aware_mpi
            gpu_sync()
            MPI.Sendrecv!(sbuf_dz, rbuf_dz, comm_cart; dest=dst, source=src)
        else
            if dst != MPI.PROC_NULL; copyto!(sbuf_hz, sbuf_dz); end
            MPI.Sendrecv!(sbuf_hz, rbuf_hz, comm_cart; dest=dst, source=src)
            if src != MPI.PROC_NULL; copyto!(rbuf_dz, rbuf_hz); end
        end
        if src != MPI.PROC_NULL
            @gpu_launch threads=nthreadsz blocks=nblocksz unpack_F(rbuf_dz, Q, NV, nxp, nyp, nzp)
        end
    end
end



function pack_R(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > NG || j > nyp + 2 * NG || k > nzp + 2 * NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[nxp+i, j, k, n]
    end
    return
end

function pack_U(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp + 2 * NG || j > NG || k > nzp + 2 * NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[i, nyp+j, k, n]
    end
    return
end

function pack_F(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp + 2 * NG || j > nyp + 2 * NG || k > NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[i, j, nzp+k, n]
    end
    return
end

function pack_L(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > NG || j > nyp + 2 * NG || k > nzp + 2 * NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[NG+i, j, k, n]
    end
    return
end

function pack_D(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp + 2 * NG || j > NG || k > nzp + 2 * NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[i, NG+j, k, n]
    end
    return
end

function pack_B(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp + 2 * NG || j > nyp + 2 * NG || k > NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[i, j, NG+k, n]
    end
    return
end

function unpack_L(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > NG || j > nyp + 2 * NG || k > nzp + 2 * NG
        return
    end

    for n = 1:NV
        @inbounds Q[i, j, k, n] = buf[i, j, k, n]
    end
    return
end

function unpack_D(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp + 2 * NG || j > NG || k > nzp + 2 * NG
        return
    end

    for n = 1:NV
        @inbounds Q[i, j, k, n] = buf[i, j, k, n]
    end
    return
end

function unpack_B(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp + 2 * NG || j > nyp + 2 * NG || k > NG
        return
    end

    for n = 1:NV
        @inbounds Q[i, j, k, n] = buf[i, j, k, n]
    end
    return
end

function unpack_R(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > NG || j > nyp + 2 * NG || k > nzp + 2 * NG
        return
    end

    for n = 1:NV
        @inbounds Q[i+nxp+NG, j, k, n] = buf[i, j, k, n]
    end
    return
end

function unpack_U(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp + 2 * NG || j > NG || k > nzp + 2 * NG
        return
    end

    for n = 1:NV
        @inbounds Q[i, j+nyp+NG, k, n] = buf[i, j, k, n]
    end
    return
end

function unpack_F(buf, Q, NV, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp + 2 * NG || j > nyp + 2 * NG || k > NG
        return
    end

    for n = 1:NV
        @inbounds Q[i, j, k+nzp+NG, n] = buf[i, j, k, n]
    end
    return
end

# =============================================================================
# Direct-Copy Inter-Block Ghost Exchange (Conformal Grids)
#
# Direct MPI copy for conformal grids: ghost cell = exact copy of neighbor's interior boundary.
#
# full_range=false: pack only real range in perpendicular direction → fills face ghost
# full_range=true:  pack full range (incl. ghost) in perpendicular dir → fills j-k edge ghost
# =============================================================================

# ─── GPU pack kernel: boundary slab → contiguous buffer ───
function ghost_face_pack!(dst, Q, fid, nyp, nzp, NV,
                          u_start, u_len, v_start, v_len)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)
    nxp = Int32(size(Q, 1)) - Int32(2) * ng
    if i > u_len || j > ng || k > v_len; return; end
    source = _ghost_face_pack_source_index(
        fid, nxp, nyp, nzp, ng,
        i + u_start - Int32(1), j, k + v_start - Int32(1),
    )
    for n = Int32(1):NV
        @inbounds dst[i, j, k, n] = Q[source[1], source[2], source[3], n]
    end
    return
end

# ─── GPU unpack kernel: contiguous buffer → ghost cells ───
function ghost_face_unpack!(Q, src, dst_fid, src_fid, nyp, nzp, NV,
                            is_cross, transform_code, u_start, u_len, v_start, v_len)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)
    nxp = Int32(size(Q, 1)) - Int32(2) * ng

    if i > u_len || j > ng || k > v_len; return; end
    target = _ghost_face_unpack_target_index(
        dst_fid, nxp, nyp, nzp, ng,
        i + u_start - Int32(1), j, k + v_start - Int32(1),
    )
    # Keep the transform arithmetic in the kernel body.  AMDGPU's compiler
    # does not reliably inline helper methods that return index tuples.
    # The packed buffer is always ordered from the source face inward.  The
    # destination ghost layer order depends only on the destination side:
    # low-face ghosts are stored outside-in, while high-face ghosts are
    # stored inside-out.  Do not infer this from the source face parity;
    # reverse exchange (high -> low) must reverse the normal layer.
    source_layer::Int32 = isodd(dst_fid) ?
        ng + Int32(1) - j : j
    code::Int32 = Int32(transform_code)
    swap = (code & Int32(1)) != Int32(0)
    reverse_u = (code & Int32(2)) != Int32(0)
    reverse_v = (code & Int32(4)) != Int32(0)
    source_u::Int32 = swap ?
        (reverse_v ? v_len + Int32(1) - k : k) :
        (reverse_u ? u_len + Int32(1) - i : i)
    source_v::Int32 = swap ?
        (reverse_u ? u_len + Int32(1) - i : i) :
        (reverse_v ? v_len + Int32(1) - k : k)
    for n = Int32(1):NV
        @inbounds Q[target[1], target[2], target[3], n] =
            src[source_u, source_layer, source_v, n]
    end
    return
end

# ─── Pre-allocated buffer pool ───
struct GhostBufferPool
    sorted_keys::Vector{Tuple{Int,Int}}
    cpu_send::Vector{Vector{FT}}       # one flat CPU buf per exchange slot
    cpu_recv::Vector{Vector{FT}}
    gpu_pack::GPUArray{FT, 1}          # single GPU staging for pack (legacy fallback)
    gpu_unpack::GPUArray{FT, 1}        # single GPU staging for unpack (legacy fallback)
    # Phase C: per-slot GPU buffers for batch pack/unpack
    gpu_pack_slots::Vector{GPUArray{FT, 1}}   # per-slot GPU pack buffers
    gpu_unpack_slots::Vector{GPUArray{FT, 1}} # per-slot GPU unpack buffers
    max_buf_elems::Int
    n_slots::Int
    # Phase D: consolidated D2H/H2D buffers (Method B)
    gpu_pack_combined::GPUArray{FT, 1}     # all slots packed contiguously on GPU
    cpu_send_combined::Vector{FT}          # single CPU buffer for batch D2H
    gpu_unpack_combined::GPUArray{FT, 1}   # all slots contiguously on GPU for H2D
    cpu_recv_combined::Vector{FT}          # single CPU buffer for batch H2D
end

function init_ghost_buffer_pool(blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b, NV_max)
    comm = MPI.COMM_WORLD
    world_rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)

    sorted_keys = sort(collect(keys(connectivity)))

    # Dry run with full_range=true to count exchanges and find max buffer size
    n_exchanges = 0
    max_buf_elems = 0
    for conn_key in sorted_keys
        dst_b_id, dst_fid = conn_key
        if !haskey(blocks, dst_b_id); continue; end
        conn = connectivity[conn_key]
        src_b_id = conn.src_b
        dst_b = blocks[dst_b_id]
        nxp = dst_b.Nx; nyp = dst_b.Ny; nzp = dst_b.Nz
        my_local_rank = world_rank - rank_offsets[dst_b_id + 1]
        px_d, py_d, pz_d = Block_Nprocs[dst_b_id + 1]
        rx_d = my_local_rank ÷ (py_d * pz_d)
        ry_d = (my_local_rank ÷ pz_d) % py_d
        rz_d = my_local_rank % pz_d
        if dst_fid == 1 && rx_d != 0; continue; end
        if dst_fid == 2 && rx_d != px_d - 1; continue; end
        if dst_fid == 3 && ry_d != 0; continue; end
        if dst_fid == 4 && ry_d != py_d - 1; continue; end
        if dst_fid == 5 && rz_d != 0; continue; end
        if dst_fid == 6 && rz_d != pz_d - 1; continue; end
        u_d_s, u_d_e, v_d_s, v_d_e, V_tot_d = _get_face_uv_extents(
            dst_fid, rx_d, ry_d, rz_d, px_d, py_d, pz_d, Nx_b[dst_b_id+1], Ny_b[dst_b_id+1], Nz_b[dst_b_id+1])
        px_s, py_s, pz_s = Block_Nprocs[src_b_id + 1]
        for rank_s in 0:(px_s * py_s * pz_s - 1)
            sx = rank_s ÷ (py_s * pz_s); sy = (rank_s ÷ pz_s) % py_s; sz = rank_s % pz_s
            src_fid = conn.src_f
            if src_fid == 1 && sx != 0; continue; end
            if src_fid == 2 && sx != px_s - 1; continue; end
            if src_fid == 3 && sy != 0; continue; end
            if src_fid == 4 && sy != py_s - 1; continue; end
            if src_fid == 5 && sz != 0; continue; end
            if src_fid == 6 && sz != pz_s - 1; continue; end
            u_s_s, u_s_e, v_s_s, v_s_e, V_tot_s = _get_face_uv_extents(
                src_fid, sx, sy, sz, px_s, py_s, pz_s, Nx_b[src_b_id+1], Ny_b[src_b_id+1], Nz_b[src_b_id+1])
            mapped_u_s, mapped_u_e, mapped_v_s, mapped_v_e = if hasproperty(conn, :transform) && conn.transform !== nothing
                source_ranges = (
                    _nonuniform_extent(sx, px_s, Nx_b[src_b_id+1]),
                    _nonuniform_extent(sy, py_s, Ny_b[src_b_id+1]),
                    _nonuniform_extent(sz, pz_s, Nz_b[src_b_id+1]),
                )
                mapped = structured_map_face_extent(
                    structured_inverse_face_transform(conn.transform), source_ranges,
                    (Nx_b[src_b_id+1], Ny_b[src_b_id+1], Nz_b[src_b_id+1]),
                )
                (mapped[1][1], mapped[1][2], mapped[2][1], mapped[2][2])
            elseif conn.reverse_tan
                (u_s_s, u_s_e, V_tot_s - v_s_e + 1, V_tot_s - v_s_s + 1)
            else
                (u_s_s, u_s_e, v_s_s, v_s_e)
            end
            u_int_s = max(u_d_s, mapped_u_s); u_int_e = min(u_d_e, mapped_u_e)
            v_int_s = max(v_d_s, mapped_v_s); v_int_e = min(v_d_e, mapped_v_e)
            if u_int_s <= u_int_e && v_int_s <= v_int_e
                n_exchanges += 1
                lu = u_int_s - u_d_s + 1; hu = u_int_e - u_d_s + 1
                lv = v_int_s - v_d_s + 1; hv = v_int_e - v_d_s + 1
                u_len = (hu - lu + 1) + 2NG
                v_s = (v_int_s == 1) ? lv : lv + NG
                v_e = (v_int_e == V_tot_d) ? hv + 2NG : hv + NG
                v_len = v_e - v_s + 1
                buf_elems = u_len * v_len * NG * NV_max
                max_buf_elems = max(max_buf_elems, buf_elems)
            end
        end
    end

    n_slots = max(1, n_exchanges)
    max_buf_elems = max(1, max_buf_elems)
    cpu_send = [Vector{FT}(undef, max_buf_elems) for _ in 1:n_slots]
    cpu_recv = [Vector{FT}(undef, max_buf_elems) for _ in 1:n_slots]
    gpu_pack   = gpu_zeros(FT, max_buf_elems)
    gpu_unpack = gpu_zeros(FT, max_buf_elems)
    # Phase C: per-slot GPU buffers for batch pack/unpack
    gpu_pack_slots   = [gpu_zeros(FT, max_buf_elems) for _ in 1:n_slots]
    gpu_unpack_slots = [gpu_zeros(FT, max_buf_elems) for _ in 1:n_slots]
    # Phase D: consolidated buffers for single-copy D2H/H2D (Method B)
    combined_total = n_slots * max_buf_elems
    gpu_pack_combined   = gpu_zeros(FT, combined_total)
    cpu_send_combined   = Vector{FT}(undef, combined_total)
    gpu_unpack_combined = gpu_zeros(FT, combined_total)
    cpu_recv_combined   = Vector{FT}(undef, combined_total)

    if world_rank == 0
        max_buffer_kb = _structured_buffer_bytes(max_buf_elems, FT) / 1024
        combined_mb = 2 * _structured_buffer_bytes(combined_total, FT) / 1024^2
        @printf "  > GhostBufferPool: %d exchanges/call, max buf %.1f KB, GPU staging 2×%.1f KB + %d×%.1f KB (batch), combined %.2f MB\n" n_exchanges max_buffer_kb max_buffer_kb 2*n_slots max_buffer_kb combined_mb
    end
    return GhostBufferPool(sorted_keys, cpu_send, cpu_recv, gpu_pack, gpu_unpack,
                           gpu_pack_slots, gpu_unpack_slots, max_buf_elems, n_slots,
                           gpu_pack_combined, cpu_send_combined,
                           gpu_unpack_combined, cpu_recv_combined)
end

# ─── Phase D: Optimized ghost exchange ───
# Method A: Local exchanges skip CPU staging (GPU-direct pack→unpack)
# Method C: Delta mode — full_range only packs border sub-regions (v-ghost strips + u-ghost columns)
#           reducing data volume by ~87% while keeping same MPI message count.
function copy_ghost_face!(blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b, 
                          target_field_name::Symbol, NV::Int, pool::GhostBufferPool;
                          full_range::Bool=false, delta_mode::Bool=false,
                          target_arrays=nothing)
    comm = MPI.COMM_WORLD
    world_rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)

    reqs_send = MPI.Request[]
    reqs_recv = MPI.Request[]
    local_unpack_jobs = []   # Method A: GPU-direct local exchanges
    remote_pack_jobs = []    # (slot, n_elems) for D2H
    remote_unpack_jobs = []  # remote exchanges needing MPI
    
    # ── Precompute mapping: (packed_b, packed_f, target_rank_global) -> slot ──
    slot_map = Dict{Tuple{Int,Int,Int}, Int}()
    _temp_slot = 0
    for conn_key in pool.sorted_keys
        _d_b, _d_f = conn_key
        if !haskey(blocks, _d_b); continue; end
        _conn = connectivity[conn_key]
        _s_b = _conn.src_b
        _px_d, _py_d, _pz_d = Block_Nprocs[_d_b + 1]
        _my_rank_local = world_rank - rank_offsets[_d_b + 1]
        _rx_d = _my_rank_local ÷ (_py_d * _pz_d)
        _ry_d = (_my_rank_local ÷ _pz_d) % _py_d
        _rz_d = _my_rank_local % _pz_d
        if _d_f == 1 && _rx_d != 0; continue; end
        if _d_f == 2 && _rx_d != _px_d - 1; continue; end
        if _d_f == 3 && _ry_d != 0; continue; end
        if _d_f == 4 && _ry_d != _py_d - 1; continue; end
        if _d_f == 5 && _rz_d != 0; continue; end
        if _d_f == 6 && _rz_d != _pz_d - 1; continue; end
        
        _u_d_s, _u_d_e, _v_d_s, _v_d_e, _V_d = _get_face_uv_extents(
            _d_f, _rx_d, _ry_d, _rz_d, _px_d, _py_d, _pz_d, Nx_b[_d_b+1], Ny_b[_d_b+1], Nz_b[_d_b+1])
        _px_s, _py_s, _pz_s = Block_Nprocs[_s_b + 1]
        
        for _r_s in 0:(_px_s * _py_s * _pz_s - 1)
            _sx = _r_s ÷ (_py_s * _pz_s); _sy = (_r_s ÷ _pz_s) % _py_s; _sz = _r_s % _pz_s
            _s_f = _conn.src_f
            if _s_f == 1 && _sx != 0; continue; end
            if _s_f == 2 && _sx != _px_s - 1; continue; end
            if _s_f == 3 && _sy != 0; continue; end
            if _s_f == 4 && _sy != _py_s - 1; continue; end
            if _s_f == 5 && _sz != 0; continue; end
            if _s_f == 6 && _sz != _pz_s - 1; continue; end
            
            _u_s_s, _u_s_e, _v_s_s, _v_s_e, _V_s = _get_face_uv_extents(
                _s_f, _sx, _sy, _sz, _px_s, _py_s, _pz_s, Nx_b[_s_b+1], Ny_b[_s_b+1], Nz_b[_s_b+1])
            _map_u_s, _map_u_e, _map_v_s, _map_v_e = if hasproperty(_conn, :transform) && _conn.transform !== nothing
                _source_ranges = (
                    _nonuniform_extent(_sx, _px_s, Nx_b[_s_b+1]),
                    _nonuniform_extent(_sy, _py_s, Ny_b[_s_b+1]),
                    _nonuniform_extent(_sz, _pz_s, Nz_b[_s_b+1]),
                )
                _mapped = structured_map_face_extent(
                    structured_inverse_face_transform(_conn.transform), _source_ranges,
                    (Nx_b[_s_b+1], Ny_b[_s_b+1], Nz_b[_s_b+1]),
                )
                (_mapped[1][1], _mapped[1][2], _mapped[2][1], _mapped[2][2])
            elseif _conn.reverse_tan
                (_u_s_s, _u_s_e, _V_s - _v_s_e + 1, _V_s - _v_s_s + 1)
            else
                (_u_s_s, _u_s_e, _v_s_s, _v_s_e)
            end
            
            _u_i_s = max(_u_d_s, _map_u_s); _u_i_e = min(_u_d_e, _map_u_e)
            _v_i_s = max(_v_d_s, _map_v_s); _v_i_e = min(_v_d_e, _map_v_e)
            if _u_i_s <= _u_i_e && _v_i_s <= _v_i_e
                _temp_slot += 1
                _src_rank_g = rank_offsets[_s_b + 1] + _r_s
                slot_map[(_d_b, _d_f, _src_rank_g)] = _temp_slot
            end
        end
    end

    slot = 0

    for conn_key in pool.sorted_keys
        dst_b_id, dst_fid = conn_key
        if !haskey(blocks, dst_b_id); continue; end
        conn = connectivity[conn_key]
        src_b_id, src_fid, reverse_tan = conn.src_b, conn.src_f, conn.reverse_tan
        transform_code = hasproperty(conn, :transform) && conn.transform !== nothing ?
            Int(structured_face_transform_code(structured_inverse_face_transform(conn.transform))) :
            (reverse_tan ? 4 : 0)
        dst_b = blocks[dst_b_id]
        dst_array = target_arrays === nothing ?
            getfield(dst_b, target_field_name) : target_arrays[dst_b_id]
        nxp = dst_b.Nx; nyp = dst_b.Ny; nzp = dst_b.Nz

        my_local_rank = world_rank - rank_offsets[dst_b_id + 1]
        px_d, py_d, pz_d = Block_Nprocs[dst_b_id + 1]
        rx_d = my_local_rank ÷ (py_d * pz_d)
        ry_d = (my_local_rank ÷ pz_d) % py_d
        rz_d = my_local_rank % pz_d
        if dst_fid == 1 && rx_d != 0; continue; end
        if dst_fid == 2 && rx_d != px_d - 1; continue; end
        if dst_fid == 3 && ry_d != 0; continue; end
        if dst_fid == 4 && ry_d != py_d - 1; continue; end
        if dst_fid == 5 && rz_d != 0; continue; end
        if dst_fid == 6 && rz_d != pz_d - 1; continue; end

        u_d_s, u_d_e, v_d_s, v_d_e, V_tot_d = _get_face_uv_extents(
            dst_fid, rx_d, ry_d, rz_d, px_d, py_d, pz_d, Nx_b[dst_b_id+1], Ny_b[dst_b_id+1], Nz_b[dst_b_id+1])
        px_s, py_s, pz_s = Block_Nprocs[src_b_id + 1]
        is_cross_type = _validate_face_transform(
            src_fid, dst_fid, conn.flip_normal,
        )

        for rank_s in 0:(px_s * py_s * pz_s - 1)
            sx = rank_s ÷ (py_s * pz_s)
            sy = (rank_s ÷ pz_s) % py_s
            sz = rank_s % pz_s
            if src_fid == 1 && sx != 0; continue; end
            if src_fid == 2 && sx != px_s - 1; continue; end
            if src_fid == 3 && sy != 0; continue; end
            if src_fid == 4 && sy != py_s - 1; continue; end
            if src_fid == 5 && sz != 0; continue; end
            if src_fid == 6 && sz != pz_s - 1; continue; end
            u_s_s, u_s_e, v_s_s, v_s_e, V_tot_s = _get_face_uv_extents(
                src_fid, sx, sy, sz, px_s, py_s, pz_s, Nx_b[src_b_id+1], Ny_b[src_b_id+1], Nz_b[src_b_id+1])
            mapped_u_s, mapped_u_e, mapped_v_s, mapped_v_e = if hasproperty(conn, :transform) && conn.transform !== nothing
                source_ranges = (
                    _nonuniform_extent(sx, px_s, Nx_b[src_b_id+1]),
                    _nonuniform_extent(sy, py_s, Ny_b[src_b_id+1]),
                    _nonuniform_extent(sz, pz_s, Nz_b[src_b_id+1]),
                )
                mapped = structured_map_face_extent(
                    structured_inverse_face_transform(conn.transform), source_ranges,
                    (Nx_b[src_b_id+1], Ny_b[src_b_id+1], Nz_b[src_b_id+1]),
                )
                (mapped[1][1], mapped[1][2], mapped[2][1], mapped[2][2])
            elseif reverse_tan
                (u_s_s, u_s_e, V_tot_s - v_s_e + 1, V_tot_s - v_s_s + 1)
            else
                (u_s_s, u_s_e, v_s_s, v_s_e)
            end
            u_int_s = max(u_d_s, mapped_u_s); u_int_e = min(u_d_e, mapped_u_e)
            v_int_s = max(v_d_s, mapped_v_s); v_int_e = min(v_d_e, mapped_v_e)

            if u_int_s <= u_int_e && v_int_s <= v_int_e
                slot += 1
                src_rank_global = rank_offsets[src_b_id + 1] + rank_s
                tag_send = dst_b_id * 100 + dst_fid
                tag_recv = src_b_id * 100 + src_fid
                local_u_dst_s = u_int_s - u_d_s + 1
                local_u_dst_e = u_int_e - u_d_s + 1
                local_v_dst_s = v_int_s - v_d_s + 1
                local_v_dst_e = v_int_e - v_d_s + 1
                if full_range
                    # The full-range pass runs after ξ MPI exchange and must carry
                    # the local ξ ghost columns across inter-block faces. This is
                    # required at block-interface/rank-cut edges when a block is
                    # split along ξ (e.g. 24-GPU production runs).
                    u_pack_start = local_u_dst_s
                    u_pack_end   = local_u_dst_e + 2NG
                else
                    u_pack_start = local_u_dst_s + NG
                    u_pack_end   = local_u_dst_e + NG
                end
                u_pack_len   = u_pack_end - u_pack_start + 1

                # Check if src block is local (on the same rank).  In standard
                # rank-split mode `Block_to_rank[src]` is only the first rank of
                # the block, not an ownership predicate for every subrank.  Only
                # use that fallback for unsplit/multi-block-per-rank blocks.
                src_is_local = (haskey(blocks, src_b_id) && world_rank == src_rank_global)
                if !src_is_local && @isdefined(Block_to_rank) && prod(Block_Nprocs[src_b_id + 1]) == 1
                    src_is_local = (Block_to_rank[src_b_id + 1] == world_rank && haskey(blocks, src_b_id))
                end

                # ── Method C: Delta mode — pack only border sub-regions ──
                # Delta sub-regions are valid only when source and destination use
                # the same face axes and tangential direction.
                use_delta = delta_mode && full_range && !is_cross_type &&
                            transform_code == 0
                if use_delta
                    has_v_bottom = (v_int_s == 1)
                    has_v_top    = (v_int_e == V_tot_d)
                    v_real_start = local_v_dst_s + NG
                    v_real_len   = local_v_dst_e - local_v_dst_s + 1
                    ng = NG

                    # Build list of sub-regions: (u_s, u_l, v_s, v_l)
                    sub_regions = Tuple{Int,Int,Int,Int}[]
                    # Sub-region 1: v-bottom ghost strip (full u, NG v-rows at bottom)
                    if has_v_bottom
                        push!(sub_regions, (u_pack_start, u_pack_len, local_v_dst_s, ng))
                    end
                    # Sub-region 2: v-top ghost strip (full u, NG v-rows at top)
                    if has_v_top
                        push!(sub_regions, (u_pack_start, u_pack_len, local_v_dst_e + ng + 1, ng))
                    end
                    # Sub-region 3: u-ghost left column (NG u-cols, real v range)
                    push!(sub_regions, (u_pack_start, ng, v_real_start, v_real_len))
                    # Sub-region 4: u-ghost right column (NG u-cols, real v range)
                    push!(sub_regions, (u_pack_end - ng + 1, ng, v_real_start, v_real_len))

                    # Pack all sub-regions sequentially into compact buffer
                    compact_offset = 0
                    sub_shapes = Tuple{Int,Int,Int,Int}[]
                    for (us, ul, vs, vl) in sub_regions
                        shape = (ul, ng, vl, NV)
                        n = prod(shape)
                        gpu_buf = reshape(@view(pool.gpu_pack_slots[slot][compact_offset+1:compact_offset+n]), shape)
                        th = (min(ul, 8), min(shape[2], 8), min(shape[3], 8))
                        nb_k = (cld(ul, th[1]), cld(shape[2], th[2]), cld(shape[3], th[3]))
                        @gpu_launch threads=th blocks=nb_k ghost_face_pack!(gpu_buf, dst_array,
                            Int32(dst_fid), Int32(nyp), Int32(nzp), Int32(NV),
                            Int32(us), Int32(ul), Int32(vs), Int32(vl))
                        push!(sub_shapes, shape)
                        compact_offset += n
                    end
                    n_elems = compact_offset

                    recv_sub_shapes = Tuple{Int,Int,Int,Int}[]
                    for (us, ul, vs, vl) in sub_regions
                        push!(recv_sub_shapes, (ul, NG, vl, NV))
                    end

                    if src_is_local
                        src_slot = slot_map[(src_b_id, src_fid, world_rank)]
                        push!(local_unpack_jobs, (dst_array, src_slot, nothing, dst_fid, src_fid, nyp, nzp, NV,
                                                 is_cross_type, transform_code, 0, 0, 0, 0,
                                                 sub_regions, recv_sub_shapes, n_elems))
                    else
                        n_recv = sum(prod.(recv_sub_shapes))
                        push!(remote_pack_jobs, (slot, n_elems))
                        push!(remote_unpack_jobs, (dst_array, slot, nothing, dst_fid, src_fid, nyp, nzp, NV,
                                                  is_cross_type, transform_code, 0, 0, 0, 0,
                                                  src_rank_global, tag_send, tag_recv, n_elems, n_recv,
                                                  sub_regions, recv_sub_shapes))
                    end
                else
                    # ── Standard mode: pack full slab ──
                    swap_orientation = (transform_code & 1) != 0
                    extend_v_ghosts = full_range &&
                        (swap_orientation || v_int_s == 1 || v_int_e == V_tot_d)
                    v_pack_start = extend_v_ghosts ? local_v_dst_s : local_v_dst_s + NG
                    v_pack_end   = extend_v_ghosts ? local_v_dst_e + 2NG : local_v_dst_e + NG
                    v_pack_len   = v_pack_end - v_pack_start + 1

                    pack_shape = (u_pack_len, NG, v_pack_len, NV)
                    n_elems = prod(pack_shape)

                    gpu_buf = reshape(@view(pool.gpu_pack_slots[slot][1:n_elems]), pack_shape)
                    th = (min(u_pack_len, 8), min(pack_shape[2], 8), min(pack_shape[3], 8))
                    nb_k = (cld(u_pack_len, th[1]), cld(pack_shape[2], th[2]), cld(pack_shape[3], th[3]))
                    @gpu_launch threads=th blocks=nb_k ghost_face_pack!(gpu_buf, dst_array,
                        Int32(dst_fid), Int32(nyp), Int32(nzp), Int32(NV),
                        Int32(u_pack_start), Int32(u_pack_len), Int32(v_pack_start), Int32(v_pack_len))

                    recv_shape = (u_pack_len, NG, v_pack_len, NV)

                    if src_is_local
                        src_slot = slot_map[(src_b_id, src_fid, world_rank)]
                        push!(local_unpack_jobs, (dst_array, src_slot, recv_shape, dst_fid, src_fid, nyp, nzp, NV,
                                                 is_cross_type, transform_code, u_pack_start, u_pack_len, v_pack_start, v_pack_len))
                    else
                        n_recv = prod(recv_shape)
                        push!(remote_pack_jobs, (slot, n_elems))
                        push!(remote_unpack_jobs, (dst_array, slot, recv_shape, dst_fid, src_fid, nyp, nzp, NV,
                                                  is_cross_type, transform_code, u_pack_start, u_pack_len, v_pack_start, v_pack_len,
                                                  src_rank_global, tag_send, tag_recv, n_elems, n_recv))
                    end
                end
            end
        end
    end

    # ── Single gpu_sync for ALL pack kernels ──
    if length(remote_pack_jobs) > 0 || length(local_unpack_jobs) > 0
        gpu_sync()
    end

    # ── Method A: Launch local unpack kernels directly from gpu_pack_slots ──
    for job in local_unpack_jobs
        dst_arr = job[1]; uslot = job[2]; buf_shape = job[3]
        d_fid = job[4]; s_fid = job[5]; ny = job[6]; nz = job[7]; nv = job[8]
        is_cross = job[9]; rev_tan = job[10]

        if buf_shape === nothing
            # Delta mode: unpack sub-regions
            sub_regions = job[15]; sub_shapes = job[16]; n_total = job[17]
            offset = 0
            for (idx, (us, ul, vs, vl)) in enumerate(sub_regions)
                shape = sub_shapes[idx]
                n = prod(shape)
                source_shape = (Int(rev_tan) & 1) != 0 ?
                    (shape[3], shape[2], shape[1], shape[4]) : shape
                gpu_buf = reshape(@view(pool.gpu_pack_slots[uslot][offset+1:offset+n]), source_shape)
                th = (min(ul, 8), min(Int(NG), 8), min(vl, 8))
                nb_k = (cld(ul, th[1]), cld(Int(NG), th[2]), cld(vl, th[3]))
                @gpu_launch threads=th blocks=nb_k ghost_face_unpack!(dst_arr, gpu_buf,
                    Int32(d_fid), Int32(s_fid), Int32(ny), Int32(nz), Int32(nv),
                    is_cross, rev_tan, Int32(us), Int32(ul), Int32(vs), Int32(vl))
                offset += n
            end
        else
            # Standard mode: single unpack
            u_s = job[11]; u_l = job[12]; v_s = job[13]; v_l = job[14]
            n = prod(buf_shape)
            source_shape = (Int(rev_tan) & 1) != 0 ?
                (buf_shape[3], buf_shape[2], buf_shape[1], buf_shape[4]) : buf_shape
            gpu_buf = reshape(@view(pool.gpu_pack_slots[uslot][1:n]), source_shape)
            th = (min(u_l, 8), min(Int(NG), 8), min(v_l, 8))
            nb_k = (cld(u_l, th[1]), cld(Int(NG), th[2]), cld(v_l, th[3]))
            @gpu_launch threads=th blocks=nb_k ghost_face_unpack!(dst_arr, gpu_buf,
                Int32(d_fid), Int32(s_fid), Int32(ny), Int32(nz), Int32(nv),
                is_cross, rev_tan, Int32(u_s), Int32(u_l), Int32(v_s), Int32(v_l))
        end
    end

    # ── Per-slot D2H for remote slots ──
    if length(remote_pack_jobs) > 0
        for (s, ne) in remote_pack_jobs
            copyto!(pool.cpu_send[s], 1, pool.gpu_pack_slots[s], 1, ne)
        end
    end

    # ── Issue MPI sends/receives ──
    for job in remote_unpack_jobs
        uslot = job[2]
        src_rank_g = job[15]; t_send = job[16]; t_recv = job[17]
        ne = job[18]; nr = job[19]
        cpu_send_flat = @view pool.cpu_send[uslot][1:ne]
        cpu_recv_flat = @view pool.cpu_recv[uslot][1:nr]
        push!(reqs_send, MPI.Isend(cpu_send_flat, comm; dest=src_rank_g, tag=t_send))
        push!(reqs_recv, MPI.Irecv!(cpu_recv_flat, comm; source=src_rank_g, tag=t_recv))
    end

    if length(reqs_send) > 0 || length(local_unpack_jobs) > 0
        if length(reqs_send) > 0
            MPI.Waitall(reqs_send)
            MPI.Waitall(reqs_recv)
        end

        # ── Per-slot H2D + unpack for remote slots ──
        if length(remote_unpack_jobs) > 0
            for job in remote_unpack_jobs
                uslot = job[2]
                nr = job[19]
                copyto!(pool.gpu_unpack_slots[uslot], 1, pool.cpu_recv[uslot], 1, nr)
            end

            for job in remote_unpack_jobs
                dst_arr = job[1]; uslot = job[2]; buf_shape = job[3]
                d_fid = job[4]; s_fid = job[5]; ny = job[6]; nz = job[7]; nv = job[8]
                is_cross = job[9]; rev_tan = job[10]

                if buf_shape === nothing
                    # Delta mode: unpack sub-regions from compact buffer
                    sub_regions = job[20]; recv_sub_shapes = job[21]
                    offset = 0
                    for (idx, (us, ul, vs, vl)) in enumerate(sub_regions)
                        shape = recv_sub_shapes[idx]
                        n = prod(shape)
                        source_shape = (Int(rev_tan) & 1) != 0 ?
                            (shape[3], shape[2], shape[1], shape[4]) : shape
                        gpu_buf = reshape(@view(pool.gpu_unpack_slots[uslot][offset+1:offset+n]), source_shape)
                        th = (min(ul, 8), min(Int(NG), 8), min(vl, 8))
                        nb_k = (cld(ul, th[1]), cld(Int(NG), th[2]), cld(vl, th[3]))
                        @gpu_launch threads=th blocks=nb_k ghost_face_unpack!(dst_arr, gpu_buf,
                            Int32(d_fid), Int32(s_fid), Int32(ny), Int32(nz), Int32(nv),
                            is_cross, rev_tan, Int32(us), Int32(ul), Int32(vs), Int32(vl))
                        offset += n
                    end
                else
                    # Standard mode: single unpack
                    u_s = job[11]; u_l = job[12]; v_s = job[13]; v_l = job[14]
                    n = prod(buf_shape)
                    source_shape = (Int(rev_tan) & 1) != 0 ?
                        (buf_shape[3], buf_shape[2], buf_shape[1], buf_shape[4]) : buf_shape
                    gpu_buf = reshape(@view(pool.gpu_unpack_slots[uslot][1:n]), source_shape)
                    th = (min(u_l, 8), min(Int(NG), 8), min(v_l, 8))
                    nb_k = (cld(u_l, th[1]), cld(Int(NG), th[2]), cld(v_l, th[3]))
                    @gpu_launch threads=th blocks=nb_k ghost_face_unpack!(dst_arr, gpu_buf,
                        Int32(d_fid), Int32(s_fid), Int32(ny), Int32(nz), Int32(nv),
                        is_cross, rev_tan, Int32(u_s), Int32(u_l), Int32(v_s), Int32(v_l))
                end
            end
        end
    end
end


# =============================================================================
# BEGIN FOLDED METRICS SYNC
# Multi-rank interface metric synchronization
# Folded from metrics_sync.jl.
# =============================================================================
using MPI

# =============================================================================
# pack_face_metrics!
#
# Pack one interblock face's (Area, nx, ny, nz) into a contiguous buffer of
# shape (u_len, v_len, 4) in *local sub-domain* indices [1..u_len, 1..v_len].
#
# Optional sub-range (u_s,u_e,v_s,v_e) restricts packing to a slice of the
# locally-owned face; when omitted the full local face is packed. This is the
# multi-rank-per-block generalization: each rank only owns a sub-domain, so it
# only packs the (u,v) overlap region it shares with a particular neighbor
# sub-rank (computed via _get_face_uv_extents, matching copy_ghost_face!).
#
# The local full-face dimensions (Nx,Ny,Nz) are the *sub-domain* cell counts
# (nxp,nyp,nzp), and the array is padded by NG ghost cells on every side.
# =============================================================================
function pack_face_metrics!(fid, Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk, Nx, Ny, Nz, NG;
                            u_s::Int=1, u_e::Int=-1, v_s::Int=1, v_e::Int=-1)
    # Default sub-range = full local face
    if fid == 1 || fid == 2
        U = Ny; V = Nz
    elseif fid == 3 || fid == 4
        U = Nx; V = Nz
    else  # fid == 5 || fid == 6
        U = Nx; V = Ny
    end
    u_s = u_s < 1 ? 1 : u_s
    u_e = u_e < 1 ? U : u_e
    v_s = v_s < 1 ? 1 : v_s
    v_e = v_e < 1 ? V : v_e
    u_len = u_e - u_s + 1
    v_len = v_e - v_s + 1
    buf = zeros(Float64, u_len, v_len, 4)

    if fid == 1 || fid == 2
        i_idx = (fid == 1) ? NG + 1 : Nx + NG + 1
        for kk in 1:v_len, jj in 1:u_len
            j = u_s + jj - 1
            k = v_s + kk - 1
            buf[jj, kk, 1] = Areai[i_idx, j+NG, k+NG]
            buf[jj, kk, 2] = nxi[i_idx, j+NG, k+NG]
            buf[jj, kk, 3] = nyi[i_idx, j+NG, k+NG]
            buf[jj, kk, 4] = nzi[i_idx, j+NG, k+NG]
        end
    elseif fid == 3 || fid == 4
        j_idx = (fid == 3) ? NG + 1 : Ny + NG + 1
        for kk in 1:v_len, ii in 1:u_len
            i = u_s + ii - 1
            k = v_s + kk - 1
            buf[ii, kk, 1] = Areaj[i+NG, j_idx, k+NG]
            buf[ii, kk, 2] = nxj[i+NG, j_idx, k+NG]
            buf[ii, kk, 3] = nyj[i+NG, j_idx, k+NG]
            buf[ii, kk, 4] = nzj[i+NG, j_idx, k+NG]
        end
    else  # fid == 5 || fid == 6
        k_idx = (fid == 5) ? NG + 1 : Nz + NG + 1
        for jj in 1:v_len, ii in 1:u_len
            i = u_s + ii - 1
            j = v_s + jj - 1
            buf[ii, jj, 1] = Areak[i+NG, j+NG, k_idx]
            buf[ii, jj, 2] = nxk[i+NG, j+NG, k_idx]
            buf[ii, jj, 3] = nyk[i+NG, j+NG, k_idx]
            buf[ii, jj, 4] = nzk[i+NG, j+NG, k_idx]
        end
    end
    return buf
end

# =============================================================================
# unpack_and_average_metrics!
#
# Unpack a received face slice into the local metrics array, applying the
# master-block averaging rule: the block with the lower ID wins, so both sides
# of an interblock face end up with identical (Area, n) — this is what makes
# the numerical flux exactly conservative across block interfaces.
#
# (u_s,u_e,v_s,v_e) is the *local sub-domain* index range to write (the
# overlap region this rank shares with the neighbor sub-rank that sent the
# buffer). buf_recv is indexed [1..u_len, 1..v_len].
#
# reverse_tan: when true, the neighbor's tangential (v) coordinate runs
# opposite to ours. Both sides pack their OWN local slice in ascending v, so
# the received buffer's v ordering is reversed relative to our local v. We
# flip the v index inside unpack (mirroring the original 1-rank-per-block
# version's `v_nb = reverse_tan ? (v_len+1-v) : v`). u is never flipped.
# =============================================================================
function unpack_and_average_metrics!(fid, nb_fid, buf_recv, reverse_tan, bid, nb_bid,
                                     Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk,
                                     Nx, Ny, Nz, NG;
                                     transform=nothing,
                                     u_s::Int=1, u_e::Int=-1, v_s::Int=1, v_e::Int=-1)
    if fid == 1 || fid == 2
        U = Ny; V = Nz
    elseif fid == 3 || fid == 4
        U = Nx; V = Nz
    else
        U = Nx; V = Ny
    end
    u_s = u_s < 1 ? 1 : u_s
    u_e = u_e < 1 ? U : u_e
    v_s = v_s < 1 ? 1 : v_s
    v_e = v_e < 1 ? V : v_e

    u_len = u_e - u_s + 1
    v_len = v_e - v_s + 1

    # Both sides use their pre-synchronization face vectors.  The received
    # vector is reoriented into the local face convention before averaging.
    sign_my = (fid % 2 == 1) ? -1.0 : 1.0
    sign_nb = (nb_fid % 2 == 1) ? -1.0 : 1.0
    flip_normal = - (sign_my * sign_nb)

    for jj in 1:u_len
        u_loc = u_s + jj - 1
        for kk in 1:v_len
            v_loc = v_s + kk - 1
            # Reorient the neighbor's `(u,v)` sheet through the full signed
            # face transform.  The legacy path retains its v-only behavior.
            source_u, source_v = if transform === nothing
                (jj, reverse_tan ? (v_len + 1 - kk) : kk)
            else
                destination = structured_face_frame(fid)
                source = structured_face_frame(nb_fid)
                map_u = transform.source_for_destination[destination.u_axis]
                map_v = transform.source_for_destination[destination.v_axis]
                source_coordinates = zeros(Int, 3)
                source_dimensions = zeros(Int, 3)
                source_dimensions[source.u_axis] = size(buf_recv, 1)
                source_dimensions[source.v_axis] = size(buf_recv, 2)
                source_coordinates[abs(map_u)] = map_u < 0 ?
                    source_dimensions[abs(map_u)] + 1 - jj : jj
                source_coordinates[abs(map_v)] = map_v < 0 ?
                    source_dimensions[abs(map_v)] + 1 - kk : kk
                (source_coordinates[source.u_axis], source_coordinates[source.v_axis])
            end
            A_nb    = buf_recv[source_u, source_v, 1]
            nx_nb   = buf_recv[source_u, source_v, 2]
            ny_nb   = buf_recv[source_u, source_v, 3]
            nz_nb   = buf_recv[source_u, source_v, 4]

            if fid == 1 || fid == 2
                i_idx = (fid == 1) ? NG + 1 : Nx + NG + 1
                A_local = Areai[i_idx, u_loc+NG, v_loc+NG]
                sx_local = A_local * nxi[i_idx, u_loc+NG, v_loc+NG]
                sy_local = A_local * nyi[i_idx, u_loc+NG, v_loc+NG]
                sz_local = A_local * nzi[i_idx, u_loc+NG, v_loc+NG]
            elseif fid == 3 || fid == 4
                j_idx = (fid == 3) ? NG + 1 : Ny + NG + 1
                A_local = Areaj[u_loc+NG, j_idx, v_loc+NG]
                sx_local = A_local * nxj[u_loc+NG, j_idx, v_loc+NG]
                sy_local = A_local * nyj[u_loc+NG, j_idx, v_loc+NG]
                sz_local = A_local * nzj[u_loc+NG, j_idx, v_loc+NG]
            else
                k_idx = (fid == 5) ? NG + 1 : Nz + NG + 1
                A_local = Areak[u_loc+NG, v_loc+NG, k_idx]
                sx_local = A_local * nxk[u_loc+NG, v_loc+NG, k_idx]
                sy_local = A_local * nyk[u_loc+NG, v_loc+NG, k_idx]
                sz_local = A_local * nzk[u_loc+NG, v_loc+NG, k_idx]
            end

            sx_avg = (sx_local + flip_normal * A_nb * nx_nb) / 2
            sy_avg = (sy_local + flip_normal * A_nb * ny_nb) / 2
            sz_avg = (sz_local + flip_normal * A_nb * nz_nb) / 2
            A_avg = sqrt(sx_avg^2 + sy_avg^2 + sz_avg^2)
            A_avg > eps(Float64) || throw(DomainError(
                A_avg,
                "degenerate averaged interface metric at " *
                "block=$bid face=$fid u=$u_loc v=$v_loc " *
                "local=($sx_local,$sy_local,$sz_local) " *
                "neighbor=($(A_nb*nx_nb),$(A_nb*ny_nb),$(A_nb*nz_nb))",
            ))
            nx_avg = sx_avg / A_avg
            ny_avg = sy_avg / A_avg
            nz_avg = sz_avg / A_avg

            if fid == 1 || fid == 2
                i_idx = (fid == 1) ? NG + 1 : Nx + NG + 1
                Areai[i_idx, u_loc+NG, v_loc+NG] = A_avg
                nxi[i_idx, u_loc+NG, v_loc+NG] = nx_avg
                nyi[i_idx, u_loc+NG, v_loc+NG] = ny_avg
                nzi[i_idx, u_loc+NG, v_loc+NG] = nz_avg
            elseif fid == 3 || fid == 4
                j_idx = (fid == 3) ? NG + 1 : Ny + NG + 1
                Areaj[u_loc+NG, j_idx, v_loc+NG] = A_avg
                nxj[u_loc+NG, j_idx, v_loc+NG] = nx_avg
                nyj[u_loc+NG, j_idx, v_loc+NG] = ny_avg
                nzj[u_loc+NG, j_idx, v_loc+NG] = nz_avg
            else
                k_idx = (fid == 5) ? NG + 1 : Nz + NG + 1
                Areak[u_loc+NG, v_loc+NG, k_idx] = A_avg
                nxk[u_loc+NG, v_loc+NG, k_idx] = nx_avg
                nyk[u_loc+NG, v_loc+NG, k_idx] = ny_avg
                nzk[u_loc+NG, v_loc+NG, k_idx] = nz_avg
            end
        end
    end
end

# =============================================================================
# sync_all_interface_metrics!  (multi-rank-per-block aware)
#
# Synchronize interblock face metrics (Area + normal) so that both sides of
# every block-block interface carry identical values (master-block averaging).
# This is required for a conservative numerical flux across interfaces.
#
# Rank matching follows the SAME sub-domain face-overlap logic as
# copy_ghost_face! (mpi.jl): for each interblock face of a block this rank
# owns, compute this rank's (rx,ry,rz) within the block, skip faces the
# sub-domain does not touch, then for each neighbor sub-rank that touches the
# neighbor face compute the (u,v) overlap and exchange only that slice. This
# makes the function correct under standard mode (N_ranks >= N_blocks, blocks
# split into 3D sub-domains) as well as multi-block mode (1 rank per block).
# =============================================================================
const _tmp_Block_Nprocs = Ref{Any}(nothing)

const _STRUCTURED_METRIC_MPI_TAG_LIMIT = 32767

function _structured_metric_interface_tags(connectivity, base_tag::Integer)
    0 <= base_tag <= _STRUCTURED_METRIC_MPI_TAG_LIMIT || throw(ArgumentError(
        "metric interface MPI tag base must be in " *
        "0:$(_STRUCTURED_METRIC_MPI_TAG_LIMIT), got $base_tag",
    ))
    interface_keys = Set{Tuple{Int,Int}}()
    for ((bid, fid), connection) in connectivity
        neighbor_bid = Int(connection.src_b)
        neighbor_fid = Int(connection.src_f)
        bid >= 0 && neighbor_bid >= 0 || throw(ArgumentError(
            "metric connectivity block ids must be nonnegative",
        ))
        1 <= fid <= 6 && 1 <= neighbor_fid <= 6 || throw(ArgumentError(
            "metric connectivity face ids must be in 1:6",
        ))
        endpoint = 6Int(bid) + Int(fid) - 1
        neighbor_endpoint = 6neighbor_bid + neighbor_fid - 1
        push!(interface_keys, (
            min(endpoint, neighbor_endpoint),
            max(endpoint, neighbor_endpoint),
        ))
    end
    ordered_keys = sort!(collect(interface_keys))
    highest_tag = base_tag + length(ordered_keys) - 1
    highest_tag <= _STRUCTURED_METRIC_MPI_TAG_LIMIT || throw(ArgumentError(
        "metric interface count $(length(ordered_keys)) exceeds the guaranteed " *
        "MPI tag range at base $base_tag",
    ))
    return Dict(
        key => Int(base_tag) + index - 1
        for (index, key) in enumerate(ordered_keys)
    )
end

function _structured_metric_gate_tolerance(raw_value, variable_name)
    value = tryparse(Float64, strip(string(raw_value)))
    value === nothing && throw(ArgumentError(
        "$variable_name must be a finite nonnegative number, got '$raw_value'",
    ))
    isfinite(value) && value >= 0 || throw(ArgumentError(
        "$variable_name must be a finite nonnegative number, got '$raw_value'",
    ))
    return value
end

function sync_all_interface_metrics!(local_blocks, temp_metrics_h,
                                     face_bc, connectivity, rank_offsets,
                                     Block_Nprocs, Nx_b, Ny_b, Nz_b, NG)
    _tmp_Block_Nprocs[] = Block_Nprocs
    sync_all_interface_metrics!(local_blocks, temp_metrics_h, nothing, 0, 0,
                                face_bc, connectivity, rank_offsets, Nx_b, Ny_b, Nz_b, NG)
end

function sync_all_interface_metrics!(local_blocks, temp_metrics_h, sync_dict, _a, _b,
                                     face_bc, connectivity,
                                     rank_offsets, Nx_b, Ny_b, Nz_b, NG)
    # ─────────────────────────────────────────────────────────────────
    # MPI interface-metric synchronization (REWRITTEN for multi-rank-per-block
    # safety, 2026-06-23).
    #
    # PROBLEM with previous version (deadlocked at 15 ranks):
    #   tag_send/tag_recv = bid*100 + fid only.  Under sub-domain decomposition
    #   along ξ, multiple ranks share the same (bid, fid) η/ζ face — so three
    #   ranks may issue Isend with the SAME tag to ONE neighbor rank, but that
    #   neighbor only issues one Irecv per tag.  Two of the three sends never
    #   match an Irecv → MPI_Waitall hangs forever.
    #
    # FIX: each logical block-face pair receives a stable compact tag from the
    # global connectivity. MPI source matching distinguishes sub-rank peers,
    # so one tag per interface is sufficient and grows linearly with topology.
    #
    # Plus: replace Isend/Irecv/Waitall with MPI.Sendrecv! when both ends
    # post in the same loop iteration — guaranteed deadlock-free.
    #
    # Signature is 12-param to match HEAD/99cb362 solver.jl call site:
    #   sync_all_interface_metrics!(collect(keys(temp_metrics_h)), sync_dict,
    #       0, 0, 0, face_bc, connectivity, _rank_offsets_setup,
    #       nxp, nyp, nzp, NG)
    # The 3 zeros (positions 3-5) are legacy placeholders, ignored here.
    # ─────────────────────────────────────────────────────────────────
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    world_size = MPI.Comm_size(MPI.COMM_WORLD)

    if world_rank == 0
        println("Rank 0: Synchronizing interface metrics for $(length(local_blocks)) blocks...")
        flush(stdout)
    end

    if world_rank == 0
        println("Rank 0: Synchronizing metrics for $(length(local_blocks)) main blocks (rewritten algorithm)...")
        flush(stdout)
    end

    # Step 1: build a GLOBAL list of all (sender_rank, receiver_rank,
    # sender_bid, sender_fid, recv_bid, recv_fid, ...) directed exchange edges
    # that EVERY rank can agree on. We enumerate ALL blocks (0..Nblocks-1),
    # not just local ones, so every rank produces the same edge list.
    #
    # An "edge" is one directed message: rank A's face (bA, fA) sub-domain →
    # rank B's face (bB, fB) sub-domain, where (bA,fA) and (bB,fB) are connected
    # by `connectivity`, and the two sub-domains overlap in face coordinates.
    #
    # The global list is built by every rank deterministically (same iteration
    # order, same connectivity), so each rank knows exactly which messages to
    # send and which to receive.

    # Connectivity only describes inter-block faces, so it is legitimately
    # empty for a single block.  The mesh-size arrays are the authoritative
    # block metadata and must define the layout even when there are no edges.
    Nblocks_total = length(Nx_b)
    if length(Ny_b) != Nblocks_total || length(Nz_b) != Nblocks_total
        error(
            "Inconsistent block metadata in metric synchronization: " *
            "Nx_b=$(length(Nx_b)), Ny_b=$(length(Ny_b)), Nz_b=$(length(Nz_b))",
        )
    end
    for ((b, f), conn) in connectivity
        if !(0 <= b < Nblocks_total && 1 <= f <= 6 &&
             0 <= conn.src_b < Nblocks_total && 1 <= conn.src_f <= 6)
            error(
                "Connectivity endpoint outside block metadata in metric synchronization: " *
                "($b,$f) -> ($(conn.src_b),$(conn.src_f)), Nblocks=$Nblocks_total",
            )
        end
    end

    # Compute (px, py, pz) for each block.
    block_layout = Dict{Int, Tuple{Int,Int,Int}}()
    for bid in 0:(Nblocks_total-1)
        if _tmp_Block_Nprocs[] !== nothing && bid + 1 <= length(_tmp_Block_Nprocs[])
            val = _tmp_Block_Nprocs[][bid + 1]
            block_layout[bid] = (Int(val[1]), Int(val[2]), Int(val[3]))
        elseif @isdefined(Block_Nprocs) && bid + 1 <= length(Block_Nprocs)
            val = Block_Nprocs[bid + 1]
            block_layout[bid] = (Int(val[1]), Int(val[2]), Int(val[3]))
        elseif isdefined(Main, :Block_Nprocs) && bid + 1 <= length(Main.Block_Nprocs)
            val = Main.Block_Nprocs[bid + 1]
            block_layout[bid] = (Int(val[1]), Int(val[2]), Int(val[3]))
        elseif bid + 2 <= length(rank_offsets)
            npx = rank_offsets[bid + 2] - rank_offsets[bid + 1]
            block_layout[bid] = (npx, 1, 1)
        else
            block_layout[bid] = (1, 1, 1)
        end
    end

    # All faces I (world_rank) am supposed to communicate over.  For each face
    # I own, find every neighbor sub-rank that overlaps with me, and store
    # both the (u,v) overlap on my side and the neighbor's rank id.
    # We then iterate this list, calling Sendrecv! for each entry.

    my_exchanges = Vector{NamedTuple}()

    for bid in local_blocks
        if !haskey(temp_metrics_h, bid); continue; end

        px, py, pz = block_layout[bid]
        my_local_rank = world_rank - rank_offsets[bid + 1]
        if my_local_rank < 0 || my_local_rank >= px * py * pz; continue; end
        rx = my_local_rank ÷ (py * pz)
        ry = (my_local_rank ÷ pz) % py
        rz = my_local_rank % pz

        ngx, ngy, ngz = Nx_b[bid+1], Ny_b[bid+1], Nz_b[bid+1]
        nxp = ngx ÷ px + (rx < (ngx % px) ? 1 : 0)
        nyp = ngy ÷ py + (ry < (ngy % py) ? 1 : 0)
        nzp = ngz ÷ pz + (rz < (ngz % pz) ? 1 : 0)

        for fid in 1:6
            if !haskey(face_bc, (bid, fid)) || face_bc[(bid, fid)] != 0; continue; end
            if !_ms_subdomain_touches_face(fid, rx, ry, rz, px, py, pz); continue; end
            conn = connectivity[(bid, fid)]
            nb_bid = conn.src_b
            nb_fid = conn.src_f
            reverse_tan = conn.reverse_tan

            u_d_s, u_d_e, v_d_s, v_d_e, V_tot_d = _ms_face_uv_extent(
                fid, rx, ry, rz, px, py, pz, Nx_b[bid+1], Ny_b[bid+1], Nz_b[bid+1])

            npx, npy, npz = block_layout[nb_bid]
            npranks = npx * npy * npz
            for rank_s in 0:(npranks - 1)
                sx = rank_s ÷ (npy * npz)
                sy = (rank_s ÷ npz) % npy
                sz = rank_s % npz
                if !_ms_subdomain_touches_face(nb_fid, sx, sy, sz, npx, npy, npz); continue; end

                u_s_s, u_s_e, v_s_s, v_s_e, V_tot_s = _ms_face_uv_extent(
                    nb_fid, sx, sy, sz, npx, npy, npz, Nx_b[nb_bid+1], Ny_b[nb_bid+1], Nz_b[nb_bid+1])

                mapped_u_s, mapped_u_e, mapped_v_s, mapped_v_e = if hasproperty(conn, :transform) && conn.transform !== nothing
                    source_ranges = (
                        _ms_nonuniform_extent(sx, npx, Nx_b[nb_bid+1]),
                        _ms_nonuniform_extent(sy, npy, Ny_b[nb_bid+1]),
                        _ms_nonuniform_extent(sz, npz, Nz_b[nb_bid+1]),
                    )
                    mapped = structured_map_face_extent(
                        structured_inverse_face_transform(conn.transform), source_ranges,
                        (Nx_b[nb_bid+1], Ny_b[nb_bid+1], Nz_b[nb_bid+1]),
                    )
                    (mapped[1][1], mapped[1][2], mapped[2][1], mapped[2][2])
                elseif reverse_tan
                    (u_s_s, u_s_e, V_tot_s - v_s_e + 1, V_tot_s - v_s_s + 1)
                else
                    (u_s_s, u_s_e, v_s_s, v_s_e)
                end

                u_int_s = max(u_d_s, mapped_u_s); u_int_e = min(u_d_e, mapped_u_e)
                v_int_s = max(v_d_s, mapped_v_s); v_int_e = min(v_d_e, mapped_v_e)
                if u_int_s > u_int_e || v_int_s > v_int_e; continue; end

                local_u_s = u_int_s - u_d_s + 1
                local_u_e = u_int_e - u_d_s + 1
                local_v_s = v_int_s - v_d_s + 1
                local_v_e = v_int_e - v_d_s + 1

                nb_rank_global = rank_offsets[nb_bid + 1] + rank_s

                push!(my_exchanges, (
                    bid=bid, fid=fid, conn=conn,
                    u_s=local_u_s, u_e=local_u_e, v_s=local_v_s, v_e=local_v_e,
                    nxp=nxp, nyp=nyp, nzp=nzp,
                    nb_rank=nb_rank_global, nb_bid=nb_bid, nb_fid=nb_fid,
                ))
            end
        end
    end

    # Sort my exchanges by (nb_rank, bid, fid) for deterministic ordering.
    sort!(my_exchanges, by = e -> (e.nb_rank, e.bid, e.fid, e.nb_bid, e.nb_fid))

    # ─── Non-blocking 2-phase pattern ───────────────────────────────
    # PHASE 1: Post ALL Irecv first (no blocking, just registers the receive).
    # PHASE 2: Post ALL Isend after.
    # PHASE 3: Waitall.
    # PHASE 4: Unpack.
    #
    # Why this works: every rank pre-posts ALL its receives before sending
    # anything, so by the time any Isend reaches the network, the matching
    # Irecv is already waiting. No deadlock possible from ordering.
    #
    # Tag uniqueness: both ends map the same canonical endpoint pair to the
    # same compact connectivity index. Distinct logical interfaces have
    # distinct tags; MPI source matching separates their sub-rank overlaps.
    #
    # Pre-allocate buffers for each exchange.
    send_bufs = Vector{Array{Float64,3}}(undef, length(my_exchanges))
    recv_bufs = Vector{Array{Float64,3}}(undef, length(my_exchanges))
    tags = Vector{Int}(undef, length(my_exchanges))
    local_peer = fill(0, length(my_exchanges))
    interface_tags = _structured_metric_interface_tags(connectivity, 7000)
    for (i, ex) in enumerate(my_exchanges)
        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk = temp_metrics_h[ex.bid]
        send_bufs[i] = pack_face_metrics!(ex.fid,
            Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk,
            ex.nxp, ex.nyp, ex.nzp, NG;
            u_s=ex.u_s, u_e=ex.u_e, v_s=ex.v_s, v_e=ex.v_e)
        recv_bufs[i] = similar(send_bufs[i])
        endpoint = ex.bid*6 + ex.fid - 1
        nb_endpoint = ex.nb_bid*6 + ex.nb_fid - 1
        tags[i] = interface_tags[(
            min(endpoint, nb_endpoint), max(endpoint, nb_endpoint),
        )]
    end
    for (i, ex) in enumerate(my_exchanges)
        if ex.nb_rank != world_rank
            continue
        end
        transform_swaps_tangents =
            hasproperty(ex.conn, :transform) &&
            ex.conn.transform !== nothing &&
            (structured_face_transform_code(
                structured_inverse_face_transform(ex.conn.transform),
            ) & 1) != 0
        expected_peer_size = transform_swaps_tangents ?
            (size(recv_bufs[i], 2), size(recv_bufs[i], 1),
             size(recv_bufs[i], 3)) :
            size(recv_bufs[i])
        peer = findfirst(eachindex(my_exchanges)) do candidate
            other = my_exchanges[candidate]
            other.bid == ex.nb_bid && other.fid == ex.nb_fid &&
                other.nb_bid == ex.bid && other.nb_fid == ex.fid &&
                other.nb_rank == world_rank &&
                size(send_bufs[candidate]) == expected_peer_size
        end
        if peer === nothing
            error("Missing local metric interface peer for block=$(ex.bid) face=$(ex.fid)")
        end
        local_peer[i] = peer
    end

    # PHASE 1: Post all Irecv first.
    reqs = MPI.Request[]
    for (i, ex) in enumerate(my_exchanges)
        if ex.nb_rank != world_rank
            push!(reqs, MPI.Irecv!(recv_bufs[i], MPI.COMM_WORLD;
                                   source=ex.nb_rank, tag=tags[i]))
        end
    end
    # PHASE 2: Post all Isend after.
    for (i, ex) in enumerate(my_exchanges)
        if ex.nb_rank != world_rank
            push!(reqs, MPI.Isend(send_bufs[i], MPI.COMM_WORLD;
                                  dest=ex.nb_rank, tag=tags[i]))
        end
    end
    # PHASE 3: Wait for all.
    if !isempty(reqs)
        MPI.Waitall(reqs)
    end

    # PHASE 4: Unpack into local metrics arrays.
    for (i, ex) in enumerate(my_exchanges)
        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk = temp_metrics_h[ex.bid]
        source = ex.nb_rank == world_rank ? send_bufs[local_peer[i]] : recv_bufs[i]
        if hasproperty(ex.conn, :transform) && ex.conn.transform !== nothing &&
           (structured_face_transform_code(structured_inverse_face_transform(ex.conn.transform)) & 1) != 0 &&
           ex.nb_rank != world_rank
            source = reshape(vec(source), size(source, 2), size(source, 1), size(source, 3))
        end
        unpack_and_average_metrics!(ex.fid, ex.nb_fid, source, ex.conn.reverse_tan,
            ex.bid, ex.nb_bid, Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk,
            ex.nxp, ex.nyp, ex.nzp, NG;
            transform=(hasproperty(ex.conn, :transform) && ex.conn.transform !== nothing ?
                       structured_inverse_face_transform(ex.conn.transform) : nothing),
            u_s=ex.u_s, u_e=ex.u_e, v_s=ex.v_s, v_e=ex.v_e)
    end

    MPI.Barrier(MPI.COMM_WORLD)
    if world_rank == 0
        println("Rank 0: Metric synchronization complete ($(length(my_exchanges)) exchanges).")
        flush(stdout)
    end
end

# ─── Local helpers (mirror mpi.jl's _nonuniform_extent / _get_face_uv_extents
#     / face-touch checks, kept private here to avoid cross-module coupling). ───

# 1-indexed global cell range [lo,hi] for rank r within a block of nprocs over
# nglobal cells (non-uniform: first `rem` ranks get one extra cell). Identical
# to mpi.jl _nonuniform_extent.
@inline function _ms_nonuniform_extent(r, nprocs, nglobal)
    base = nglobal ÷ nprocs
    rem  = nglobal % nprocs
    lo   = min(r, rem) * (base + 1) + max(0, r - rem) * base + 1
    hi   = lo + (r < rem ? base : base - 1)
    return lo, hi
end

# True if sub-domain (rx,ry,rz) touches face fid. Mirrors the skip-checks in
# copy_ghost_face! (mpi.jl:622-627). Faces 1/2 are ξ± (x), 3/4 are η± (y),
# 5/6 are ζ± (z).
@inline function _ms_subdomain_touches_face(fid, rx, ry, rz, px, py, pz)
    if fid == 1; return rx == 0; end
    if fid == 2; return rx == px - 1; end
    if fid == 3; return ry == 0; end
    if fid == 4; return ry == py - 1; end
    if fid == 5; return rz == 0; end
    if fid == 6; return rz == pz - 1; end
    return false
end

# Global (u,v) face extent for sub-domain (rx,ry,rz). u is the first in-plane
# axis, v the second. Returns (u_lo,u_hi,v_lo,v_hi,V_tot) where V_tot is the
# full v-length of the face (used for reverse_tan mapping). Mirrors
# _get_face_uv_extents in mpi.jl:390 but also returns the u-total where needed.
@inline function _ms_face_uv_extent(fid, rx, ry, rz, px, py, pz, Nx_g, Ny_g, Nz_g)
    x_lo, x_hi = _ms_nonuniform_extent(rx, px, Nx_g)
    y_lo, y_hi = _ms_nonuniform_extent(ry, py, Ny_g)
    z_lo, z_hi = _ms_nonuniform_extent(rz, pz, Nz_g)
    if fid == 3 || fid == 4   # η face: u=x, v=z
        return x_lo, x_hi, z_lo, z_hi, Nz_g
    elseif fid == 5 || fid == 6  # ζ face: u=x, v=y
        return x_lo, x_hi, y_lo, y_hi, Ny_g
    else  # ξ face (1/2): u=y, v=z — interblock ξ connections are not used by
          # the butterfly topology, but handle them for completeness.
        return y_lo, y_hi, z_lo, z_hi, Nz_g
    end
end

function _structured_metric_block_layout(
    Nblocks_total, Block_Nprocs, rank_offsets,
)
    layout = Dict{Int,Tuple{Int,Int,Int}}()
    for bid in 0:Nblocks_total-1
        if Block_Nprocs !== nothing && bid + 1 <= length(Block_Nprocs)
            value = Block_Nprocs[bid + 1]
            layout[bid] = (Int(value[1]), Int(value[2]), Int(value[3]))
        elseif _tmp_Block_Nprocs[] !== nothing &&
               bid + 1 <= length(_tmp_Block_Nprocs[])
            value = _tmp_Block_Nprocs[][bid + 1]
            layout[bid] = (Int(value[1]), Int(value[2]), Int(value[3]))
        elseif bid + 2 <= length(rank_offsets)
            layout[bid] = (rank_offsets[bid + 2] - rank_offsets[bid + 1], 1, 1)
        else
            layout[bid] = (1, 1, 1)
        end
    end
    return layout
end

function _structured_metric_edge_payload_source_shape(ex)
    destination = structured_face_frame(ex.fid)
    source = structured_face_frame(ex.nb_fid)
    transform = _structured_metric_connection_transform(ex.fid, ex.conn)
    u_length = ex.u_e - ex.u_s + 1
    v_length = ex.v_e - ex.v_s + 1
    mapped_u = abs(transform.source_for_destination[destination.u_axis])
    return mapped_u == source.u_axis ?
        (u_length, v_length) : (v_length, u_length)
end

function sync_all_interface_metric_edges!(
    local_blocks, metric_auxiliary, face_bc, connectivity,
    rank_offsets, Block_Nprocs, Nx_b, Ny_b, Nz_b, NG,
)
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    Nblocks_total = length(Nx_b)
    length(Ny_b) == Nblocks_total && length(Nz_b) == Nblocks_total ||
        throw(DimensionMismatch("inconsistent block dimensions in SCMM edge sync"))
    block_layout = _structured_metric_block_layout(
        Nblocks_total, Block_Nprocs, rank_offsets,
    )
    exchanges = NamedTuple[]

    for bid in sort(collect(local_blocks))
        haskey(metric_auxiliary, bid) || continue
        px, py, pz = block_layout[bid]
        local_rank = world_rank - rank_offsets[bid + 1]
        0 <= local_rank < px * py * pz || continue
        rx = local_rank ÷ (py * pz)
        ry = (local_rank ÷ pz) % py
        rz = local_rank % pz
        nx_global, ny_global, nz_global =
            Nx_b[bid + 1], Ny_b[bid + 1], Nz_b[bid + 1]
        nx_local = nx_global ÷ px + (rx < nx_global % px ? 1 : 0)
        ny_local = ny_global ÷ py + (ry < ny_global % py ? 1 : 0)
        nz_local = nz_global ÷ pz + (rz < nz_global % pz ? 1 : 0)

        for fid in 1:6
            get(face_bc, (bid, fid), 1) == 0 || continue
            haskey(connectivity, (bid, fid)) || throw(ArgumentError(
                "missing connectivity for SCMM edge endpoint ($bid,$fid)",
            ))
            _ms_subdomain_touches_face(fid, rx, ry, rz, px, py, pz) || continue
            conn = connectivity[(bid, fid)]
            nb_bid, nb_fid = conn.src_b, conn.src_f
            u_d_s, u_d_e, v_d_s, v_d_e, _ = _ms_face_uv_extent(
                fid, rx, ry, rz, px, py, pz,
                nx_global, ny_global, nz_global,
            )
            npx, npy, npz = block_layout[nb_bid]
            for neighbor_local_rank in 0:npx*npy*npz-1
                sx = neighbor_local_rank ÷ (npy * npz)
                sy = (neighbor_local_rank ÷ npz) % npy
                sz = neighbor_local_rank % npz
                _ms_subdomain_touches_face(
                    nb_fid, sx, sy, sz, npx, npy, npz,
                ) || continue
                u_s_s, u_s_e, v_s_s, v_s_e, v_total_source =
                    _ms_face_uv_extent(
                        nb_fid, sx, sy, sz, npx, npy, npz,
                        Nx_b[nb_bid + 1], Ny_b[nb_bid + 1], Nz_b[nb_bid + 1],
                    )
                mapped_u_s, mapped_u_e, mapped_v_s, mapped_v_e =
                    if hasproperty(conn, :transform) && conn.transform !== nothing
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
                        (u_s_s, u_s_e,
                         v_total_source - v_s_e + 1,
                         v_total_source - v_s_s + 1)
                    else
                        (u_s_s, u_s_e, v_s_s, v_s_e)
                    end
                u_start, u_end = max(u_d_s, mapped_u_s), min(u_d_e, mapped_u_e)
                v_start, v_end = max(v_d_s, mapped_v_s), min(v_d_e, mapped_v_e)
                u_start <= u_end && v_start <= v_end || continue
                push!(exchanges, (
                    bid=bid, fid=fid, conn=conn,
                    nb_bid=nb_bid, nb_fid=nb_fid,
                    nb_rank=rank_offsets[nb_bid + 1] + neighbor_local_rank,
                    u_s=u_start-u_d_s+1, u_e=u_end-u_d_s+1,
                    v_s=v_start-v_d_s+1, v_e=v_end-v_d_s+1,
                    nx=nx_local, ny=ny_local, nz=nz_local,
                ))
            end
        end
    end
    sort!(exchanges, by=ex ->
        (ex.nb_rank, ex.bid, ex.fid, ex.nb_bid, ex.nb_fid,
         ex.u_s, ex.v_s))

    send_buffers = Vector{Any}(undef, length(exchanges))
    recv_buffers = Vector{Any}(undef, length(exchanges))
    tags = Vector{Int}(undef, length(exchanges))
    local_peer = zeros(Int, length(exchanges))
    interface_tags = _structured_metric_interface_tags(connectivity, 9000)
    for (index, ex) in enumerate(exchanges)
        auxiliary = metric_auxiliary[ex.bid]
        coordinates, workspace = auxiliary[1], auxiliary[2]
        payload = pack_scmm_face_edge_payload(
            workspace, coordinates, ex.fid, ex.nx, ex.ny, ex.nz, NG;
            u_s=ex.u_s, u_e=ex.u_e, v_s=ex.v_s, v_e=ex.v_e,
        )
        send_buffers[index] = scmm_face_edge_payload_vector(payload)
        recv_buffers[index] = similar(send_buffers[index])
        endpoint = ex.bid * 6 + ex.fid - 1
        neighbor_endpoint = ex.nb_bid * 6 + ex.nb_fid - 1
        tags[index] = interface_tags[(
            min(endpoint, neighbor_endpoint),
            max(endpoint, neighbor_endpoint),
        )]
    end
    for (index, ex) in enumerate(exchanges)
        ex.nb_rank == world_rank || continue
        peer = findfirst(eachindex(exchanges)) do candidate
            other = exchanges[candidate]
            other.bid == ex.nb_bid && other.fid == ex.nb_fid &&
                other.nb_bid == ex.bid && other.nb_fid == ex.fid &&
                other.nb_rank == world_rank &&
                length(send_buffers[candidate]) == length(recv_buffers[index])
        end
        peer === nothing && error(
            "missing local SCMM edge peer for block=$(ex.bid) face=$(ex.fid)",
        )
        local_peer[index] = peer
    end

    requests = MPI.Request[]
    for (index, ex) in enumerate(exchanges)
        ex.nb_rank == world_rank && continue
        push!(requests, MPI.Irecv!(
            recv_buffers[index], MPI.COMM_WORLD;
            source=ex.nb_rank, tag=tags[index],
        ))
    end
    for (index, ex) in enumerate(exchanges)
        ex.nb_rank == world_rank && continue
        push!(requests, MPI.Isend(
            send_buffers[index], MPI.COMM_WORLD;
            dest=ex.nb_rank, tag=tags[index],
        ))
    end
    isempty(requests) || MPI.Waitall(requests)

    for (index, ex) in enumerate(exchanges)
        source_key = (ex.nb_bid, ex.nb_fid)
        destination_key = (ex.bid, ex.fid)
        isless(source_key, destination_key) || continue
        values = ex.nb_rank == world_rank ?
            send_buffers[local_peer[index]] : recv_buffers[index]
        source_u, source_v = _structured_metric_edge_payload_source_shape(ex)
        payload = scmm_face_edge_payload_from_vector(
            values, source_u, source_v,
        )
        coordinates, workspace = metric_auxiliary[ex.bid][1:2]
        unpack_canonical_scmm_face_edges!(
            workspace, coordinates, payload, ex.fid, ex.conn,
            ex.nx, ex.ny, ex.nz, NG;
            u_s=ex.u_s, u_e=ex.u_e, v_s=ex.v_s, v_e=ex.v_e,
        )
    end
    MPI.Barrier(MPI.COMM_WORLD)
    world_rank == 0 && println(
        "Rank 0: Canonical SCMM interface-edge synchronization complete.",
    )
    return nothing
end

@inline function _structured_metric_face_arrays(values)
    offset = if length(values) == 14
        1
    elseif length(values) == 12
        0
    else
        throw(DimensionMismatch(
            "expected 12 metric arrays, optionally preceded by cache path and " *
            "followed by volume; got tuple length $(length(values))",
        ))
    end
    return ntuple(index -> values[offset + index], 12)
end

function _structured_metric_buffer_residual(
    local_buffer, source_buffer, fid::Int, neighbor_fid::Int;
    reverse_tan::Bool=false, transform=nothing,
)
    local_u, local_v = size(local_buffer, 1), size(local_buffer, 2)
    destination = structured_face_frame(fid)
    source = structured_face_frame(neighbor_fid)
    flip_normal = -structured_face_side_sign(fid) *
        structured_face_side_sign(neighbor_fid)
    max_absolute = 0.0
    max_relative = 0.0

    for local_v_index in 1:local_v, local_u_index in 1:local_u
        source_u, source_v = if transform === nothing
            (
                local_u_index,
                reverse_tan ? local_v + 1 - local_v_index : local_v_index,
            )
        else
            map_u = transform.source_for_destination[destination.u_axis]
            map_v = transform.source_for_destination[destination.v_axis]
            source_coordinates = zeros(Int, 3)
            source_dimensions = zeros(Int, 3)
            source_dimensions[source.u_axis] = size(source_buffer, 1)
            source_dimensions[source.v_axis] = size(source_buffer, 2)
            source_coordinates[abs(map_u)] = map_u < 0 ?
                source_dimensions[abs(map_u)] + 1 - local_u_index :
                local_u_index
            source_coordinates[abs(map_v)] = map_v < 0 ?
                source_dimensions[abs(map_v)] + 1 - local_v_index :
                local_v_index
            (
                source_coordinates[source.u_axis],
                source_coordinates[source.v_axis],
            )
        end
        1 <= source_u <= size(source_buffer, 1) &&
            1 <= source_v <= size(source_buffer, 2) ||
            throw(BoundsError(
                source_buffer,
                (source_u, source_v, 1),
            ))

        local_area = local_buffer[local_u_index, local_v_index, 1]
        source_area = source_buffer[source_u, source_v, 1]
        local_vector = ntuple(
            component -> local_area *
                local_buffer[local_u_index, local_v_index, component + 1],
            3,
        )
        source_vector = ntuple(
            component -> flip_normal * source_area *
                source_buffer[source_u, source_v, component + 1],
            3,
        )
        absolute = sqrt(sum(
            (local_vector[component] - source_vector[component])^2
            for component in 1:3
        ))
        local_norm = sqrt(sum(value^2 for value in local_vector))
        source_norm = sqrt(sum(value^2 for value in source_vector))
        scale = max(local_norm, source_norm, eps(Float64))
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, absolute / scale)
    end
    return max_absolute, max_relative
end

function structured_interblock_metric_residuals(
    temp_metrics_h, blocks, face_bc, connectivity,
    rank_offsets, Block_Nprocs, Nx_b, Ny_b, Nz_b, NG,
)
    world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    Nblocks_total = length(Nx_b)
    length(Ny_b) == Nblocks_total && length(Nz_b) == Nblocks_total ||
        throw(DimensionMismatch(
            "inconsistent block dimensions in shared-face metric gate",
        ))
    block_layout = _structured_metric_block_layout(
        Nblocks_total, Block_Nprocs, rank_offsets,
    )
    exchanges = NamedTuple[]

    for bid in sort(collect(keys(temp_metrics_h)))
        haskey(blocks, bid) || continue
        px, py, pz = block_layout[bid]
        local_rank = world_rank - rank_offsets[bid + 1]
        0 <= local_rank < px * py * pz || continue
        rx = local_rank ÷ (py * pz)
        ry = (local_rank ÷ pz) % py
        rz = local_rank % pz
        nx_global, ny_global, nz_global =
            Nx_b[bid + 1], Ny_b[bid + 1], Nz_b[bid + 1]
        nx_local = nx_global ÷ px + (rx < nx_global % px ? 1 : 0)
        ny_local = ny_global ÷ py + (ry < ny_global % py ? 1 : 0)
        nz_local = nz_global ÷ pz + (rz < nz_global % pz ? 1 : 0)

        for fid in 1:6
            get(face_bc, (bid, fid), nothing) == BC_INTERBLOCK ||
                continue
            _ms_subdomain_touches_face(fid, rx, ry, rz, px, py, pz) ||
                continue
            haskey(connectivity, (bid, fid)) || throw(ArgumentError(
                "missing connectivity for shared-face metric endpoint ($bid,$fid)",
            ))
            conn = connectivity[(bid, fid)]
            neighbor_bid, neighbor_fid = conn.src_b, conn.src_f
            u_d_s, u_d_e, v_d_s, v_d_e, _ = _ms_face_uv_extent(
                fid, rx, ry, rz, px, py, pz,
                nx_global, ny_global, nz_global,
            )
            npx, npy, npz = block_layout[neighbor_bid]
            for neighbor_local_rank in 0:npx*npy*npz-1
                sx = neighbor_local_rank ÷ (npy * npz)
                sy = (neighbor_local_rank ÷ npz) % npy
                sz = neighbor_local_rank % npz
                _ms_subdomain_touches_face(
                    neighbor_fid, sx, sy, sz, npx, npy, npz,
                ) || continue
                u_s_s, u_s_e, v_s_s, v_s_e, v_total_source =
                    _ms_face_uv_extent(
                        neighbor_fid, sx, sy, sz, npx, npy, npz,
                        Nx_b[neighbor_bid + 1], Ny_b[neighbor_bid + 1],
                        Nz_b[neighbor_bid + 1],
                    )
                mapped_u_s, mapped_u_e, mapped_v_s, mapped_v_e =
                    if hasproperty(conn, :transform) &&
                       conn.transform !== nothing
                        source_ranges = (
                            _ms_nonuniform_extent(
                                sx, npx, Nx_b[neighbor_bid + 1],
                            ),
                            _ms_nonuniform_extent(
                                sy, npy, Ny_b[neighbor_bid + 1],
                            ),
                            _ms_nonuniform_extent(
                                sz, npz, Nz_b[neighbor_bid + 1],
                            ),
                        )
                        mapped = structured_map_face_extent(
                            structured_inverse_face_transform(conn.transform),
                            source_ranges,
                            (
                                Nx_b[neighbor_bid + 1],
                                Ny_b[neighbor_bid + 1],
                                Nz_b[neighbor_bid + 1],
                            ),
                        )
                        (
                            mapped[1][1], mapped[1][2],
                            mapped[2][1], mapped[2][2],
                        )
                    elseif conn.reverse_tan
                        (
                            u_s_s, u_s_e,
                            v_total_source - v_s_e + 1,
                            v_total_source - v_s_s + 1,
                        )
                    else
                        (u_s_s, u_s_e, v_s_s, v_s_e)
                    end
                u_start, u_end = max(u_d_s, mapped_u_s), min(u_d_e, mapped_u_e)
                v_start, v_end = max(v_d_s, mapped_v_s), min(v_d_e, mapped_v_e)
                u_start <= u_end && v_start <= v_end || continue
                push!(exchanges, (
                    bid=bid, fid=fid, conn=conn,
                    neighbor_bid=neighbor_bid, neighbor_fid=neighbor_fid,
                    neighbor_rank=rank_offsets[neighbor_bid + 1] +
                        neighbor_local_rank,
                    u_s=u_start-u_d_s+1, u_e=u_end-u_d_s+1,
                    v_s=v_start-v_d_s+1, v_e=v_end-v_d_s+1,
                    nx=nx_local, ny=ny_local, nz=nz_local,
                ))
            end
        end
    end
    sort!(exchanges, by=exchange -> (
        exchange.neighbor_rank, exchange.bid, exchange.fid,
        exchange.neighbor_bid, exchange.neighbor_fid,
        exchange.u_s, exchange.v_s,
    ))

    send_buffers = Vector{Any}(undef, length(exchanges))
    recv_buffers = Vector{Any}(undef, length(exchanges))
    tags = Vector{Int}(undef, length(exchanges))
    local_peer = zeros(Int, length(exchanges))
    interface_tags = _structured_metric_interface_tags(connectivity, 10000)
    for (index, exchange) in enumerate(exchanges)
        arrays = _structured_metric_face_arrays(temp_metrics_h[exchange.bid])
        send_buffers[index] = pack_face_metrics!(
            exchange.fid, arrays...,
            exchange.nx, exchange.ny, exchange.nz, NG;
            u_s=exchange.u_s, u_e=exchange.u_e,
            v_s=exchange.v_s, v_e=exchange.v_e,
        )
        recv_buffers[index] = similar(send_buffers[index])
        endpoint = exchange.bid * 6 + exchange.fid - 1
        neighbor_endpoint =
            exchange.neighbor_bid * 6 + exchange.neighbor_fid - 1
        tags[index] = interface_tags[(
            min(endpoint, neighbor_endpoint),
            max(endpoint, neighbor_endpoint),
        )]
    end
    for (index, exchange) in enumerate(exchanges)
        exchange.neighbor_rank == world_rank || continue
        peer = findfirst(eachindex(exchanges)) do candidate
            other = exchanges[candidate]
            other.bid == exchange.neighbor_bid &&
                other.fid == exchange.neighbor_fid &&
                other.neighbor_bid == exchange.bid &&
                other.neighbor_fid == exchange.fid &&
                other.neighbor_rank == world_rank &&
                length(send_buffers[candidate]) == length(recv_buffers[index])
        end
        peer === nothing && error(
            "missing local shared-face metric peer for " *
            "block=$(exchange.bid) face=$(exchange.fid)",
        )
        local_peer[index] = peer
    end

    requests = MPI.Request[]
    for (index, exchange) in enumerate(exchanges)
        exchange.neighbor_rank == world_rank && continue
        push!(requests, MPI.Irecv!(
            recv_buffers[index], MPI.COMM_WORLD;
            source=exchange.neighbor_rank, tag=tags[index],
        ))
    end
    for (index, exchange) in enumerate(exchanges)
        exchange.neighbor_rank == world_rank && continue
        push!(requests, MPI.Isend(
            send_buffers[index], MPI.COMM_WORLD;
            dest=exchange.neighbor_rank, tag=tags[index],
        ))
    end
    isempty(requests) || MPI.Waitall(requests)

    max_absolute = 0.0
    max_relative = 0.0
    for (index, exchange) in enumerate(exchanges)
        source = exchange.neighbor_rank == world_rank ?
            send_buffers[local_peer[index]] : recv_buffers[index]
        transform = if hasproperty(exchange.conn, :transform) &&
                       exchange.conn.transform !== nothing
            structured_inverse_face_transform(exchange.conn.transform)
        else
            nothing
        end
        if transform !== nothing &&
           (structured_face_transform_code(transform) & 1) != 0 &&
           exchange.neighbor_rank != world_rank
            source = reshape(
                vec(source), size(source, 2), size(source, 1), size(source, 3),
            )
        end
        absolute, relative = _structured_metric_buffer_residual(
            send_buffers[index], source,
            exchange.fid, exchange.neighbor_fid;
            reverse_tan=exchange.conn.reverse_tan,
            transform=transform,
        )
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, relative)
    end
    return max_absolute, max_relative
end

function structured_rank_metric_residuals(
    temp_metrics_h, blocks, block_comms, Block_Nprocs, NG,
)
    max_absolute = 0.0
    max_relative = 0.0
    for bid in sort(collect(keys(temp_metrics_h)))
        block = blocks[bid]
        layout = Block_Nprocs[bid + 1]
        all(layout .== 1) && continue
        communicator = block_comms[bid]
        arrays = _structured_metric_face_arrays(temp_metrics_h[bid])
        for direction in 1:3
            layout[direction] <= 1 && continue
            low_face = 2direction - 1
            high_face = 2direction
            low_buffer = pack_face_metrics!(
                low_face, arrays..., block.Nx, block.Ny, block.Nz, NG,
            )
            high_buffer = pack_face_metrics!(
                high_face, arrays..., block.Nx, block.Ny, block.Nz, NG,
            )
            source_rank, destination_rank =
                MPI.Cart_shift(communicator, direction - 1, 1)
            from_low = similar(high_buffer)
            from_high = similar(low_buffer)
            MPI.Sendrecv!(
                high_buffer, from_low, communicator;
                dest=destination_rank, source=source_rank,
            )
            MPI.Sendrecv!(
                low_buffer, from_high, communicator;
                dest=source_rank, source=destination_rank,
            )
            if source_rank != MPI.PROC_NULL
                absolute, relative = _structured_metric_buffer_residual(
                    low_buffer, from_low, low_face, high_face,
                )
                max_absolute = max(max_absolute, absolute)
                max_relative = max(max_relative, relative)
            end
            if destination_rank != MPI.PROC_NULL
                absolute, relative = _structured_metric_buffer_residual(
                    high_buffer, from_high, high_face, low_face,
                )
                max_absolute = max(max_absolute, absolute)
                max_relative = max(max_relative, relative)
            end
        end
    end
    return max_absolute, max_relative
end

function structured_shared_face_metric_residuals(
    temp_metrics_h, blocks, block_comms, face_bc, connectivity,
    rank_offsets, Block_Nprocs, Nx_b, Ny_b, Nz_b, NG,
)
    interblock_absolute, interblock_relative =
        structured_interblock_metric_residuals(
            temp_metrics_h, blocks, face_bc, connectivity,
            rank_offsets, Block_Nprocs, Nx_b, Ny_b, Nz_b, NG,
        )
    rank_absolute, rank_relative = structured_rank_metric_residuals(
        temp_metrics_h, blocks, block_comms, Block_Nprocs, NG,
    )
    local_absolute = max(interblock_absolute, rank_absolute)
    local_relative = max(interblock_relative, rank_relative)
    return (
        MPI.Allreduce(local_absolute, MPI.MAX, MPI.COMM_WORLD),
        MPI.Allreduce(local_relative, MPI.MAX, MPI.COMM_WORLD),
    )
end

# =============================================================================
# END FOLDED METRICS SYNC
# =============================================================================
