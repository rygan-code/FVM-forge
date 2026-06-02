# =============================================================================
# Interface Flux Synchronization for Multiblock Boundaries
# =============================================================================
# Ensures conservation at interblock faces by exchanging reconstructed
# face values (UL from left block, UR from right block) and computing
# a unique Riemann flux that both blocks use.
# =============================================================================

# ─────────────────────────────────────────────────────────────────
# GPU Kernel: Unpack and orient interface face values
# ─────────────────────────────────────────────────────────────────
function interface_face_unpack_kernel!(dst, src, dst_fid::Int32, nxp::Int32, nyp::Int32, nzp::Int32, NV::Int32, reverse_tan::Bool)
    u = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    v = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    
    if dst_fid == Int32(1) || dst_fid == Int32(2)
        u_len = nyp
        v_len = nzp
    elseif dst_fid == Int32(3) || dst_fid == Int32(4)
        u_len = nxp
        v_len = nzp
    else
        u_len = nxp
        v_len = nyp
    end
    
    if u > u_len || v > v_len; return; end
    
    v_src = reverse_tan ? (v_len - v + Int32(1)) : v
    
    @inbounds for n in 1:NV
        dst[u, v, n] = src[u, v_src, n]
    end
    return
end

function apply_interface_face_unpack!(dst, src, dst_fid::Int, nxp::Int, nyp::Int, nzp::Int, reverse_tan::Bool)
    dst_fid_32 = Int32(dst_fid)
    nxp_32 = Int32(nxp)
    nyp_32 = Int32(nyp)
    nzp_32 = Int32(nzp)
    NV_32 = Int32(Ncons)
    
    if dst_fid == 1 || dst_fid == 2
        u_len = nyp
        v_len = nzp
    elseif dst_fid == 3 || dst_fid == 4
        u_len = nxp
        v_len = nzp
    else
        u_len = nxp
        v_len = nyp
    end
    
    threads = (16, 16)
    blocks_k = (cld(u_len, threads[1]), cld(v_len, threads[2]))
    
    @gpu_launch threads=threads blocks=blocks_k interface_face_unpack_kernel!(
        dst, src, dst_fid_32, nxp_32, nyp_32, nzp_32, NV_32, reverse_tan)
end

# ─────────────────────────────────────────────────────────────────
# GPU Kernel: Overwrite flux at an interface face with unified Riemann solve
# ─────────────────────────────────────────────────────────────────

function interface_riemann_overwrite_i!(Fx, UL_face, UR_face, Areai, nxi, nyi, nzi,
                                        ϕ, nxp, nyp, nzp, i_face::Int32, ch_glm::FT)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    k = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if j > nyp || k > nzp; return; end

    # Face geometry (i_face is in full-array index, face array is +1 offset from cell)
    @inbounds nx = nxi[i_face+1, j+NG, k+NG]
    @inbounds ny = nyi[i_face+1, j+NG, k+NG]
    @inbounds nz = nzi[i_face+1, j+NG, k+NG]
    @inbounds Area = Areai[i_face+1, j+NG, k+NG]

    # Shock sensor at interface face
    @inbounds ϕx = max(ϕ[i_face-1, j+NG, k+NG], ϕ[i_face, j+NG, k+NG],
                       ϕ[i_face+1, j+NG, k+NG], ϕ[i_face+2, j+NG, k+NG])

    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_face[j, k, n]), Val(Ncons)))
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_face[j, k, n]), Val(Ncons)))

    # Force lin_ϕ = 1.0 (pure upwind) at interface to guarantee bit-identical flux on both blocks
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, one(FT), splitMethodID, ch_glm)

    # Overwrite Fx at the interface face
    # Fx indexing: Fx[i-NG+1, j-NG, k-NG, n] where i=i_face
    fx_i = i_face - NG + Int32(1)
    @inbounds for n = 1:Ncons
        Fx[fx_i, j, k, n] = flux_temp[n] * Area
    end
    return
end

function interface_riemann_overwrite_j!(Fy, UL_face, UR_face, Areaj, nxj, nyj, nzj,
                                        ϕ, nxp, nyp, nzp, j_face::Int32, ch_glm::FT)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    k = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i > nxp || k > nzp; return; end

    # Face geometry
    @inbounds nx = nxj[i+NG, j_face+1, k+NG]
    @inbounds ny = nyj[i+NG, j_face+1, k+NG]
    @inbounds nz = nzj[i+NG, j_face+1, k+NG]
    @inbounds Area = Areaj[i+NG, j_face+1, k+NG]

    # Shock sensor at interface face
    @inbounds ϕx = max(ϕ[i+NG, j_face-1, k+NG], ϕ[i+NG, j_face, k+NG],
                       ϕ[i+NG, j_face+1, k+NG], ϕ[i+NG, j_face+2, k+NG])

    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_face[i, k, n]), Val(Ncons)))
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_face[i, k, n]), Val(Ncons)))

    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, one(FT), splitMethodID, ch_glm)

    # Overwrite Fy: Fy[i-NG, j-NG+1, k-NG, n] where i_full=i+NG, j_full=j_face
    fy_j = j_face - NG + Int32(1)
    @inbounds for n = 1:Ncons
        Fy[i, fy_j, k, n] = flux_temp[n] * Area
    end
    return
end

function interface_riemann_overwrite_k!(Fz, UL_face, UR_face, Areak, nxk, nyk, nzk,
                                        ϕ, nxp, nyp, nzp, k_face::Int32, ch_glm::FT)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i > nxp || j > nyp; return; end

    # Face geometry
    @inbounds nx = nxk[i+NG, j+NG, k_face+1]
    @inbounds ny = nyk[i+NG, j+NG, k_face+1]
    @inbounds nz = nzk[i+NG, j+NG, k_face+1]
    @inbounds Area = Areak[i+NG, j+NG, k_face+1]

    # Shock sensor at interface face
    @inbounds ϕx = max(ϕ[i+NG, j+NG, k_face-1], ϕ[i+NG, j+NG, k_face],
                       ϕ[i+NG, j+NG, k_face+1], ϕ[i+NG, j+NG, k_face+2])

    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_face[i, j, n]), Val(Ncons)))
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_face[i, j, n]), Val(Ncons)))

    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, one(FT), splitMethodID, ch_glm)

    fz_k = k_face - NG + Int32(1)
    @inbounds for n = 1:Ncons
        Fz[i, j, fz_k, n] = flux_temp[n] * Area
    end
    return
end

# ─────────────────────────────────────────────────────────────────
# MPI Exchange of Interface Face Values
# ─────────────────────────────────────────────────────────────────

function exchange_interface_face_values!(blocks, connectivity, rank_offsets, Block_Nprocs, Nx_b, Ny_b, Nz_b)
    comm = MPI.COMM_WORLD
    world_rank = MPI.Comm_rank(comm)
    reqs = MPI.Request[]

    # For each interblock connection: send my_face_val, receive peer_face_val
    for ((dst_bid, dst_fid), conn) in connectivity
        if !haskey(blocks, dst_bid); continue; end
        src_bid = conn.src_b
        b = blocks[dst_bid]

        # Determine which buffer pair to use based on face ID
        my_gpu, peer_gpu, my_h, peer_h = _get_intf_buffers(b, dst_fid)
        if my_gpu === nothing; continue; end  # Not an interblock face

        n_elems = length(my_gpu)
        
        # D2H: copy my face values to host
        copyto!(my_h, my_gpu)
        
        # Determine the matching partner process in the neighbor block (src_bid)
        # Using the same coordinate matching logic as in copy_ghost_face!
        my_local_rank = world_rank - rank_offsets[dst_bid + 1]
        px_d, py_d, pz_d = Block_Nprocs[dst_bid + 1]
        rx_d = my_local_rank ÷ (py_d * pz_d)
        ry_d = (my_local_rank ÷ pz_d) % py_d
        rz_d = my_local_rank % pz_d

        u_d_s, u_d_e, v_d_s, v_d_e, V_tot_d = _get_face_uv_extents(
            dst_fid, rx_d, ry_d, rz_d, px_d, py_d, pz_d, Nx_b[dst_bid+1], Ny_b[dst_bid+1], Nz_b[dst_bid+1])
        px_s, py_s, pz_s = Block_Nprocs[src_bid + 1]
        
        src_rank = -1
        for rank_s in 0:(px_s * py_s * pz_s - 1)
            sx = rank_s ÷ (py_s * pz_s)
            sy = (rank_s ÷ pz_s) % py_s
            sz = rank_s % pz_s
            if conn.src_f == 1 && sx != 0; continue; end
            if conn.src_f == 2 && sx != px_s - 1; continue; end
            if conn.src_f == 3 && sy != 0; continue; end
            if conn.src_f == 4 && sy != py_s - 1; continue; end
            if conn.src_f == 5 && sz != 0; continue; end
            if conn.src_f == 6 && sz != pz_s - 1; continue; end
            
            u_s_s, u_s_e, v_s_s, v_s_e, V_tot_s = _get_face_uv_extents(
                conn.src_f, sx, sy, sz, px_s, py_s, pz_s, Nx_b[src_bid+1], Ny_b[src_bid+1], Nz_b[src_bid+1])
            mapped_u_s = u_s_s; mapped_u_e = u_s_e
            if conn.reverse_tan
                mapped_v_s = V_tot_s - v_s_e + 1; mapped_v_e = V_tot_s - v_s_s + 1
            else
                mapped_v_s = v_s_s; mapped_v_e = v_s_e
            end
            u_int_s = max(u_d_s, mapped_u_s); u_int_e = min(u_d_e, mapped_u_e)
            v_int_s = max(v_d_s, mapped_v_s); v_int_e = min(v_d_e, mapped_v_e)
            
            if u_int_s <= u_int_e && v_int_s <= v_int_e
                src_rank = rank_offsets[src_bid + 1] + rank_s
                break
            end
        end

        if src_rank == -1
            error("Could not find matching partner process for block $(dst_bid) face $(dst_fid) on rank $(world_rank)!")
        end

        # Tag convention: sender uses (sender_bid * 6 + sender_fid) as tag
        my_tag = dst_bid * 6 + dst_fid
        peer_tag = src_bid * 6 + conn.src_f

        if src_rank != world_rank
            # Remote exchange via MPI
            push!(reqs, MPI.Isend(my_h, comm; dest=src_rank, tag=my_tag))
            push!(reqs, MPI.Irecv!(peer_h, comm; source=src_rank, tag=peer_tag))
        else
            # Local exchange: source block is on the same rank
            if haskey(blocks, src_bid)
                src_b = blocks[src_bid]
                src_my_gpu, _, _, _ = _get_intf_buffers(src_b, conn.src_f)
                if src_my_gpu !== nothing
                    apply_interface_face_unpack!(peer_gpu, src_my_gpu, dst_fid, b.Nx, b.Ny, b.Nz, conn.reverse_tan)
                end
            end
        end
    end

    # Wait for all MPI operations
    if !isempty(reqs)
        MPI.Waitall(reqs)
    end

    # H2D: copy received peer values to GPU (only for remote exchanges)
    for ((dst_bid, dst_fid), conn) in connectivity
        if !haskey(blocks, dst_bid); continue; end
        src_bid = conn.src_b
        
        # Determine src_rank using same logic
        my_local_rank = world_rank - rank_offsets[dst_bid + 1]
        px_d, py_d, pz_d = Block_Nprocs[dst_bid + 1]
        rx_d = my_local_rank ÷ (py_d * pz_d)
        ry_d = (my_local_rank ÷ pz_d) % py_d
        rz_d = my_local_rank % pz_d

        u_d_s, u_d_e, v_d_s, v_d_e, V_tot_d = _get_face_uv_extents(
            dst_fid, rx_d, ry_d, rz_d, px_d, py_d, pz_d, Nx_b[dst_bid+1], Ny_b[dst_bid+1], Nz_b[dst_bid+1])
        px_s, py_s, pz_s = Block_Nprocs[src_bid + 1]
        
        src_rank = -1
        for rank_s in 0:(px_s * py_s * pz_s - 1)
            sx = rank_s ÷ (py_s * pz_s)
            sy = (rank_s ÷ pz_s) % py_s
            sz = rank_s % pz_s
            if conn.src_f == 1 && sx != 0; continue; end
            if conn.src_f == 2 && sx != px_s - 1; continue; end
            if conn.src_f == 3 && sy != 0; continue; end
            if conn.src_f == 4 && sy != py_s - 1; continue; end
            if conn.src_f == 5 && sz != 0; continue; end
            if conn.src_f == 6 && sz != pz_s - 1; continue; end
            
            u_s_s, u_s_e, v_s_s, v_s_e, V_tot_s = _get_face_uv_extents(
                conn.src_f, sx, sy, sz, px_s, py_s, pz_s, Nx_b[src_bid+1], Ny_b[src_bid+1], Nz_b[src_bid+1])
            mapped_u_s = u_s_s; mapped_u_e = u_s_e
            if conn.reverse_tan
                mapped_v_s = V_tot_s - v_s_e + 1; mapped_v_e = V_tot_s - v_s_s + 1
            else
                mapped_v_s = v_s_s; mapped_v_e = v_s_e
            end
            u_int_s = max(u_d_s, mapped_u_s); u_int_e = min(u_d_e, mapped_u_e)
            v_int_s = max(v_d_s, mapped_v_s); v_int_e = min(v_d_e, mapped_v_e)
            
            if u_int_s <= u_int_e && v_int_s <= v_int_e
                src_rank = rank_offsets[src_bid + 1] + rank_s
                break
            end
        end

        if src_rank == world_rank; continue; end  # Already handled via GPU copy

        b = blocks[dst_bid]
        _, peer_gpu, _, peer_h = _get_intf_buffers(b, dst_fid)
        if peer_gpu === nothing; continue; end
        temp_gpu = similar(peer_gpu)
        copyto!(temp_gpu, peer_h)
        apply_interface_face_unpack!(peer_gpu, temp_gpu, dst_fid, b.Nx, b.Ny, b.Nz, conn.reverse_tan)
    end
end

# Helper: get the appropriate buffer pair for a given face ID
function _get_intf_buffers(b, fid::Int)
    if fid == 1
        return b.my_face_val_ilo, b.peer_face_val_ilo, b.my_face_h_ilo, b.peer_face_h_ilo
    elseif fid == 2
        return b.my_face_val_ihi, b.peer_face_val_ihi, b.my_face_h_ihi, b.peer_face_h_ihi
    elseif fid == 3
        return b.my_face_val_jlo, b.peer_face_val_jlo, b.my_face_h_jlo, b.peer_face_h_jlo
    elseif fid == 4
        return b.my_face_val_jhi, b.peer_face_val_jhi, b.my_face_h_jhi, b.peer_face_h_jhi
    elseif fid == 5
        return b.my_face_val_klo, b.peer_face_val_klo, b.my_face_h_klo, b.peer_face_h_klo
    elseif fid == 6
        return b.my_face_val_khi, b.peer_face_val_khi, b.my_face_h_khi, b.peer_face_h_khi
    else
        return nothing, nothing, nothing, nothing
    end
end

# ─────────────────────────────────────────────────────────────────
# Top-level Dispatch: sync_interface_flux!
# ─────────────────────────────────────────────────────────────────

function sync_interface_flux!(blocks, connectivity, rank_offsets, Block_Nprocs,
                              shared_Fx, shared_Fy, shared_Fz, ch_glm::FT, Nx_b, Ny_b, Nz_b)
    # 1. Exchange interface face values via MPI
    exchange_interface_face_values!(blocks, connectivity, rank_offsets, Block_Nprocs, Nx_b, Ny_b, Nz_b)

    # 2. For each block, overwrite flux at interface faces with unified Riemann solve
    intf_threads = (16, 16)
    for (bid, b) in blocks
        nxp, nyp, nzp = b.Nx, b.Ny, b.Nz

        # ── ξ-lo (face 1): this block is the RIGHT block → peer sends UL, I have UR ──
        if b.is_interblock[1] && b.peer_face_val_ilo !== nothing
            nb_f = (cld(nyp, intf_threads[1]), cld(nzp, intf_threads[2]))
            i_face = Int32(NG)  # Interface face index in full array
            @gpu_launch threads=intf_threads blocks=nb_f interface_riemann_overwrite_i!(
                shared_Fx, b.peer_face_val_ilo, b.my_face_val_ilo,
                b.Areai, b.nxi, b.nyi, b.nzi, b.ϕ, nxp, nyp, nzp, i_face, ch_glm)
        end

        # ── ξ-hi (face 2): this block is the LEFT block → I have UL, peer sends UR ──
        if b.is_interblock[2] && b.peer_face_val_ihi !== nothing
            nb_f = (cld(nyp, intf_threads[1]), cld(nzp, intf_threads[2]))
            i_face = Int32(nxp + NG)
            @gpu_launch threads=intf_threads blocks=nb_f interface_riemann_overwrite_i!(
                shared_Fx, b.my_face_val_ihi, b.peer_face_val_ihi,
                b.Areai, b.nxi, b.nyi, b.nzi, b.ϕ, nxp, nyp, nzp, i_face, ch_glm)
        end

        # ── η-lo (face 3): this block is the RIGHT block ──
        if b.is_interblock[3] && b.peer_face_val_jlo !== nothing
            nb_f = (cld(nxp, intf_threads[1]), cld(nzp, intf_threads[2]))
            j_face = Int32(NG)
            @gpu_launch threads=intf_threads blocks=nb_f interface_riemann_overwrite_j!(
                shared_Fy, b.peer_face_val_jlo, b.my_face_val_jlo,
                b.Areaj, b.nxj, b.nyj, b.nzj, b.ϕ, nxp, nyp, nzp, j_face, ch_glm)
        end

        # ── η-hi (face 4): this block is the LEFT block ──
        if b.is_interblock[4] && b.peer_face_val_jhi !== nothing
            nb_f = (cld(nxp, intf_threads[1]), cld(nzp, intf_threads[2]))
            j_face = Int32(nyp + NG)
            @gpu_launch threads=intf_threads blocks=nb_f interface_riemann_overwrite_j!(
                shared_Fy, b.my_face_val_jhi, b.peer_face_val_jhi,
                b.Areaj, b.nxj, b.nyj, b.nzj, b.ϕ, nxp, nyp, nzp, j_face, ch_glm)
        end

        # ── ζ-lo (face 5): this block is the RIGHT block ──
        if b.is_interblock[5] && b.peer_face_val_klo !== nothing
            nb_f = (cld(nxp, intf_threads[1]), cld(nyp, intf_threads[2]))
            k_face = Int32(NG)
            @gpu_launch threads=intf_threads blocks=nb_f interface_riemann_overwrite_k!(
                shared_Fz, b.peer_face_val_klo, b.my_face_val_klo,
                b.Areak, b.nxk, b.nyk, b.nzk, b.ϕ, nxp, nyp, nzp, k_face, ch_glm)
        end

        # ── ζ-hi (face 6): this block is the LEFT block ──
        if b.is_interblock[6] && b.peer_face_val_khi !== nothing
            nb_f = (cld(nxp, intf_threads[1]), cld(nyp, intf_threads[2]))
            k_face = Int32(nzp + NG)
            @gpu_launch threads=intf_threads blocks=nb_f interface_riemann_overwrite_k!(
                shared_Fz, b.my_face_val_khi, b.peer_face_val_khi,
                b.Areak, b.nxk, b.nyk, b.nzk, b.ϕ, nxp, nyp, nzp, k_face, ch_glm)
        end
    end
end
