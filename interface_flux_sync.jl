# =============================================================================
# Interface Flux Synchronization for Multiblock Boundaries
# =============================================================================
# Ensures conservation at interblock faces by exchanging reconstructed
# face values (UL from left block, UR from right block) and computing
# a unique Riemann flux that both blocks use.
# =============================================================================

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

function exchange_interface_face_values!(blocks, connectivity, rank_offsets, Block_Nprocs)
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
        
        # Determine source rank
        src_rank = rank_offsets[src_bid + 1]

        # MPI tag: encode (dst_bid, dst_fid) to avoid tag collision
        tag_send = dst_bid * 100 + dst_fid + 50  # sending my values
        tag_recv = src_bid * 100 + conn.src_f + 50  # receiving peer values (from peer's perspective, they send with their tag)
        
        # Actually, for a pair (A.face4 ↔ B.face3):
        # Block A (dst_bid=A, dst_fid=4) sends to Block B (src_bid=B)
        # Block B (dst_bid=B, dst_fid=3) sends to Block A (src_bid=A)
        # We need symmetric tags. Use: tag = min_bid*1000 + max_bid*10 + face_pair_id
        # Simpler: each block sends its own face value to its connectivity partner
        # and receives the partner's face value.
        
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
                    copyto!(peer_gpu, src_my_gpu)  # Direct GPU-to-GPU copy
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
        src_rank = rank_offsets[conn.src_b + 1]
        if src_rank == world_rank; continue; end  # Already handled via GPU copy

        b = blocks[dst_bid]
        _, peer_gpu, _, peer_h = _get_intf_buffers(b, dst_fid)
        if peer_gpu === nothing; continue; end
        copyto!(peer_gpu, peer_h)
    end
end

# Helper: get the appropriate buffer pair for a given face ID
function _get_intf_buffers(b::Block, fid::Int)
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
                              shared_Fx, shared_Fy, shared_Fz, ch_glm::FT)
    # 1. Exchange interface face values via MPI
    exchange_interface_face_values!(blocks, connectivity, rank_offsets, Block_Nprocs)

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
