# MPI ghost cell exchange - supports both GPU-aware and CPU-staged modes
# Phase B optimization: x+/x- use non-blocking Isend/Irecv with separate buffers
# so both directions execute concurrently, saving 1 gpu_sync + MPI overlap.
# compute_fn(slot::Int) is called with slot=1 during MPI Waitall for comm-compute overlap.
function exchange_ghost(Q, NV, comm_cart, nxp, nyp, nzp,
    sbuf_hx, sbuf_dx, rbuf_hx, rbuf_dx,
    sbuf_hy, sbuf_dy, rbuf_hy, rbuf_dy,
    sbuf_hz, sbuf_dz, rbuf_hz, rbuf_dz;
    sbuf_hx2=nothing, sbuf_dx2=nothing, rbuf_hx2=nothing, rbuf_dx2=nothing,
    compute_fn=nothing)
    
    nthreadsx = (NG, 8, 8)
    nthreadsy = (8, NG, 8)
    nthreadsz = (8, 8, NG)

    nblocksx = (1, cld((nyp + 2 * NG), 8), cld((nzp + 2 * NG), 8))
    nblocksy = (cld((nxp + 2 * NG), 8), 1, cld((nzp + 2 * NG), 8))
    nblocksz = (cld((nxp + 2 * NG), 8), cld((nyp + 2 * NG), 8), 1)

    # ═══ x+/x- pipelined non-blocking exchange ═══
    src_xp, dst_xp = MPI.Cart_shift(comm_cart, 0, 1)
    src_xm, dst_xm = MPI.Cart_shift(comm_cart, 0, -1)
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
    src, dst = MPI.Cart_shift(comm_cart, 1, 1)
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
    src, dst = MPI.Cart_shift(comm_cart, 1, -1)
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
    src, dst = MPI.Cart_shift(comm_cart, 2, 1)
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
    src, dst = MPI.Cart_shift(comm_cart, 2, -1)
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
# Replaces interp_ghost! (weight-based interpolation) with direct MPI copy.
# For conformal grids: ghost cell = exact copy of neighbor's interior boundary.
#
# full_range=false: pack only real range in perpendicular direction → fills face ghost
# full_range=true:  pack full range (incl. ghost) in perpendicular dir → fills j-k edge ghost
# =============================================================================

# Non-uniform partition helper: returns (lo, hi) global cell range for rank `r`
@inline function _nonuniform_extent(r, nprocs, nglobal)
    base = nglobal ÷ nprocs
    rem  = nglobal % nprocs
    lo   = min(r, rem) * (base + 1) + max(0, r - rem) * base + 1
    hi   = lo + (r < rem ? base : base - 1)
    return lo, hi
end

function _get_face_uv_extents(fid, rx, ry, rz, px, py, pz, Nx_g, Ny_g, Nz_g)
    x_lo, x_hi = _nonuniform_extent(rx, px, Nx_g)
    z_lo, z_hi = _nonuniform_extent(rz, pz, Nz_g)
    y_lo, y_hi = _nonuniform_extent(ry, py, Ny_g)
    if fid == 3 || fid == 4
        return x_lo, x_hi, z_lo, z_hi, Nz_g
    elseif fid == 5 || fid == 6
        return x_lo, x_hi, y_lo, y_hi, Ny_g
    else
        return 0, 0, 0, 0, 0
    end
end

@inline function _is_cross_type(src_fid, dst_fid)
    src_is_j = (src_fid == 3 || src_fid == 4)
    dst_is_j = (dst_fid == 3 || dst_fid == 4)
    return src_is_j != dst_is_j
end

# ─── GPU pack kernel: boundary slab → contiguous buffer ───
function ghost_face_pack!(dst, Q, fid, nyp, nzp, NV, u_start, u_len, v_start, v_len)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)
    if fid == Int32(3) || fid == Int32(4)
        if i > u_len || j > ng || k > v_len; return; end
        j_src = fid == Int32(3) ? (ng + j) : (nyp + j)
        for n = Int32(1):NV
            @inbounds dst[i, j, k, n] = Q[i + u_start - Int32(1), j_src, k + v_start - Int32(1), n]
        end
    else
        if i > u_len || j > v_len || k > ng; return; end
        k_src = fid == Int32(5) ? (ng + k) : (nzp + k)
        for n = Int32(1):NV
            @inbounds dst[i, j, k, n] = Q[i + u_start - Int32(1), j + v_start - Int32(1), k_src, n]
        end
    end
    return
end

# ─── GPU unpack kernel: contiguous buffer → ghost cells ───
function ghost_face_unpack!(Q, src, dst_fid, src_fid, nyp, nzp, NV,
                            is_cross, reverse_tan, u_start, u_len, v_start, v_len)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)

    _is_low(f) = (f == Int32(3) || f == Int32(5))
    same_side = (_is_low(src_fid) == _is_low(dst_fid))

    if !is_cross
        if dst_fid == Int32(3) || dst_fid == Int32(4)
            if i > u_len || j > ng || k > v_len; return; end
            gs = same_side ? (ng + Int32(1) - j) : j
            j_dst = dst_fid == Int32(3) ? j : (nyp + ng + j)
            for n = Int32(1):NV
                @inbounds Q[i + u_start - Int32(1), j_dst, k + v_start - Int32(1), n] = src[i, gs, k, n]
            end
        else
            if i > u_len || j > v_len || k > ng; return; end
            gs = same_side ? (ng + Int32(1) - k) : k
            k_dst = dst_fid == Int32(5) ? k : (nzp + ng + k)
            for n = Int32(1):NV
                @inbounds Q[i + u_start - Int32(1), j + v_start - Int32(1), k_dst, n] = src[i, j, gs, n]
            end
        end
    else
        # Cross-type: src η↔ζ. Permute dims 2,3 of source on-the-fly.
        if dst_fid == Int32(5) || dst_fid == Int32(6)
            # src is η-type (u, NG, v, NV) → read as (u, v, NG, NV)
            if i > u_len || j > v_len || k > ng; return; end
            gs = same_side ? (ng + Int32(1) - k) : k
            jv = reverse_tan ? (v_len + Int32(1) - j) : j
            k_dst = dst_fid == Int32(5) ? k : (nzp + ng + k)
            for n = Int32(1):NV
                @inbounds Q[i + u_start - Int32(1), j + v_start - Int32(1), k_dst, n] = src[i, gs, jv, n]
            end
        else
            # src is ζ-type (u, v, NG, NV) → read as (u, NG, v, NV)
            if i > u_len || j > ng || k > v_len; return; end
            gs = same_side ? (ng + Int32(1) - j) : j
            kv = reverse_tan ? (v_len + Int32(1) - k) : k
            j_dst = dst_fid == Int32(3) ? j : (nyp + ng + j)
            for n = Int32(1):NV
                @inbounds Q[i + u_start - Int32(1), j_dst, k + v_start - Int32(1), n] = src[i, kv, gs, n]
            end
        end
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
            mapped_u_s = u_s_s; mapped_u_e = u_s_e
            if conn.reverse_tan
                mapped_v_s = V_tot_s - v_s_e + 1; mapped_v_e = V_tot_s - v_s_s + 1
            else
                mapped_v_s = v_s_s; mapped_v_e = v_s_e
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
        combined_mb = 2 * combined_total * 4 / 1024^2
        @printf "  > GhostBufferPool: %d exchanges/call, max buf %.1f KB, GPU staging 2×%.1f KB + %d×%.1f KB (batch), combined %.2f MB\n" n_exchanges max_buf_elems*4/1024 max_buf_elems*4/1024 2*n_slots max_buf_elems*4/1024 combined_mb
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
                          full_range::Bool=false, delta_mode::Bool=false)
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
            _map_v_s = _conn.reverse_tan ? _V_s - _v_s_e + 1 : _v_s_s
            _map_v_e = _conn.reverse_tan ? _V_s - _v_s_s + 1 : _v_s_e
            
            _u_i_s = max(_u_d_s, _u_s_s); _u_i_e = min(_u_d_e, _u_s_e)
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
        dst_b = blocks[dst_b_id]
        dst_array = getfield(dst_b, target_field_name)
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
        is_cross_type = _is_cross_type(src_fid, dst_fid)

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
            mapped_u_s = u_s_s; mapped_u_e = u_s_e
            if reverse_tan
                mapped_v_s = V_tot_s - v_s_e + 1; mapped_v_e = V_tot_s - v_s_s + 1
            else
                mapped_v_s = v_s_s; mapped_v_e = v_s_e
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
                u_pack_start = local_u_dst_s
                u_pack_end   = local_u_dst_e + 2NG
                u_pack_len   = u_pack_end - u_pack_start + 1

                # Check if src block is local (on the same rank)
                src_is_local = (haskey(blocks, src_b_id) && world_rank == src_rank_global)
                if !src_is_local && @isdefined(Block_to_rank)
                    src_is_local = (Block_to_rank[src_b_id + 1] == world_rank)
                end

                # ── Method C: Delta mode — pack only border sub-regions ──
                # Cross-type interfaces (η↔ζ) must NOT use delta mode: sub_regions are
                # built in destination coordinate space, but source packs in its own
                # (permuted) coordinate space → data lands in wrong ghost positions.
                use_delta = delta_mode && full_range && !is_cross_type
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
                        shape = (dst_fid == 3 || dst_fid == 4) ? (ul, ng, vl, NV) : (ul, vl, ng, NV)
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
                        push!(recv_sub_shapes, (src_fid == 3 || src_fid == 4) ? (ul, NG, vl, NV) : (ul, vl, NG, NV))
                    end

                    if src_is_local
                        src_slot = slot_map[(src_b_id, src_fid, world_rank)]
                        push!(local_unpack_jobs, (dst_array, src_slot, nothing, dst_fid, src_fid, nyp, nzp, NV,
                                                 is_cross_type, reverse_tan, 0, 0, 0, 0,
                                                 sub_regions, recv_sub_shapes, n_elems))
                    else
                        n_recv = sum(prod.(recv_sub_shapes))
                        push!(remote_pack_jobs, (slot, n_elems))
                        push!(remote_unpack_jobs, (dst_array, slot, nothing, dst_fid, src_fid, nyp, nzp, NV,
                                                  is_cross_type, reverse_tan, 0, 0, 0, 0,
                                                  src_rank_global, tag_send, tag_recv, n_elems, n_recv,
                                                  sub_regions, recv_sub_shapes))
                    end
                else
                    # ── Standard mode: pack full slab ──
                    v_pack_start = (full_range && v_int_s == 1)       ? local_v_dst_s : local_v_dst_s + NG
                    v_pack_end   = (full_range && v_int_e == V_tot_d) ? local_v_dst_e + 2NG : local_v_dst_e + NG
                    v_pack_len   = v_pack_end - v_pack_start + 1

                    pack_shape = (dst_fid == 3 || dst_fid == 4) ? (u_pack_len, NG, v_pack_len, NV) : (u_pack_len, v_pack_len, NG, NV)
                    n_elems = prod(pack_shape)

                    gpu_buf = reshape(@view(pool.gpu_pack_slots[slot][1:n_elems]), pack_shape)
                    th = (min(u_pack_len, 8), min(pack_shape[2], 8), min(pack_shape[3], 8))
                    nb_k = (cld(u_pack_len, th[1]), cld(pack_shape[2], th[2]), cld(pack_shape[3], th[3]))
                    @gpu_launch threads=th blocks=nb_k ghost_face_pack!(gpu_buf, dst_array,
                        Int32(dst_fid), Int32(nyp), Int32(nzp), Int32(NV),
                        Int32(u_pack_start), Int32(u_pack_len), Int32(v_pack_start), Int32(v_pack_len))

                    recv_shape = (src_fid == 3 || src_fid == 4) ? (u_pack_len, NG, v_pack_len, NV) : (u_pack_len, v_pack_len, NG, NV)

                    if src_is_local
                        src_slot = slot_map[(src_b_id, src_fid, world_rank)]
                        push!(local_unpack_jobs, (dst_array, src_slot, recv_shape, dst_fid, src_fid, nyp, nzp, NV,
                                                 is_cross_type, reverse_tan, u_pack_start, u_pack_len, v_pack_start, v_pack_len))
                    else
                        n_recv = prod(recv_shape)
                        push!(remote_pack_jobs, (slot, n_elems))
                        push!(remote_unpack_jobs, (dst_array, slot, recv_shape, dst_fid, src_fid, nyp, nzp, NV,
                                                  is_cross_type, reverse_tan, u_pack_start, u_pack_len, v_pack_start, v_pack_len,
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
                gpu_buf = reshape(@view(pool.gpu_pack_slots[uslot][offset+1:offset+n]), shape)
                if d_fid == 3 || d_fid == 4
                    th = (min(ul, 8), min(Int(NG), 8), min(vl, 8))
                    nb_k = (cld(ul, th[1]), cld(Int(NG), th[2]), cld(vl, th[3]))
                else
                    th = (min(ul, 8), min(vl, 8), min(Int(NG), 8))
                    nb_k = (cld(ul, th[1]), cld(vl, th[2]), cld(Int(NG), th[3]))
                end
                @gpu_launch threads=th blocks=nb_k ghost_face_unpack!(dst_arr, gpu_buf,
                    Int32(d_fid), Int32(s_fid), Int32(ny), Int32(nz), Int32(nv),
                    is_cross, rev_tan, Int32(us), Int32(ul), Int32(vs), Int32(vl))
                offset += n
            end
        else
            # Standard mode: single unpack
            u_s = job[11]; u_l = job[12]; v_s = job[13]; v_l = job[14]
            n = prod(buf_shape)
            gpu_buf = reshape(@view(pool.gpu_pack_slots[uslot][1:n]), buf_shape)
            if d_fid == 3 || d_fid == 4
                th = (min(u_l, 8), min(Int(NG), 8), min(v_l, 8))
                nb_k = (cld(u_l, th[1]), cld(Int(NG), th[2]), cld(v_l, th[3]))
            else
                th = (min(u_l, 8), min(v_l, 8), min(Int(NG), 8))
                nb_k = (cld(u_l, th[1]), cld(v_l, th[2]), cld(Int(NG), th[3]))
            end
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
                        gpu_buf = reshape(@view(pool.gpu_unpack_slots[uslot][offset+1:offset+n]), shape)
                        if d_fid == 3 || d_fid == 4
                            th = (min(ul, 8), min(Int(NG), 8), min(vl, 8))
                            nb_k = (cld(ul, th[1]), cld(Int(NG), th[2]), cld(vl, th[3]))
                        else
                            th = (min(ul, 8), min(vl, 8), min(Int(NG), 8))
                            nb_k = (cld(ul, th[1]), cld(vl, th[2]), cld(Int(NG), th[3]))
                        end
                        @gpu_launch threads=th blocks=nb_k ghost_face_unpack!(dst_arr, gpu_buf,
                            Int32(d_fid), Int32(s_fid), Int32(ny), Int32(nz), Int32(nv),
                            is_cross, rev_tan, Int32(us), Int32(ul), Int32(vs), Int32(vl))
                        offset += n
                    end
                else
                    # Standard mode: single unpack
                    u_s = job[11]; u_l = job[12]; v_s = job[13]; v_l = job[14]
                    n = prod(buf_shape)
                    gpu_buf = reshape(@view(pool.gpu_unpack_slots[uslot][1:n]), buf_shape)
                    if d_fid == 3 || d_fid == 4
                        th = (min(u_l, 8), min(Int(NG), 8), min(v_l, 8))
                        nb_k = (cld(u_l, th[1]), cld(Int(NG), th[2]), cld(v_l, th[3]))
                    else
                        th = (min(u_l, 8), min(v_l, 8), min(Int(NG), 8))
                        nb_k = (cld(u_l, th[1]), cld(v_l, th[2]), cld(Int(NG), th[3]))
                    end
                    @gpu_launch threads=th blocks=nb_k ghost_face_unpack!(dst_arr, gpu_buf,
                        Int32(d_fid), Int32(s_fid), Int32(ny), Int32(nz), Int32(nv),
                        is_cross, rev_tan, Int32(u_s), Int32(u_l), Int32(v_s), Int32(v_l))
                end
            end
        end
    end
end

