# piso_multigrid.jl — Geometric Multigrid GPU kernels for PISO Poisson solver
#
# Provides restriction, prolongation, and coarse-grid metric computation.
# Smoothing reuses existing piso_poisson_rb_sor! from piso.jl.
#
# Coarsening: 2:1 in all directions (nc = nf ÷ 2)
# Restriction: sum-based (preserves integral form)
# Prolongation: injection (constant interpolation)
#
# NOTE: NG is defined globally in the run config.

# ═══════════════════════════════════════════════════════════════
# 1. Restriction: fine → coarse (sum of 8 fine cells)
#    Both arrays are real-cell arrays (no ghost padding).
#    Coarse cell (ic,jc,kc) ← sum of fine cells
#      (2ic-1:2ic, 2jc-1:2jc, 2kc-1:2kc)
# ═══════════════════════════════════════════════════════════════

function mg_restrict!(r_c, r_f, nxc, nyc, nzc)
    ic = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    jc = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    kc = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if ic > nxc || jc > nyc || kc > nzc; return; end

    # Fine-grid indices (1-based real-cell)
    i1 = Int32(2) * ic - Int32(1)
    i2 = Int32(2) * ic
    j1 = Int32(2) * jc - Int32(1)
    j2 = Int32(2) * jc
    k1 = Int32(2) * kc - Int32(1)
    k2 = Int32(2) * kc

    # Sum of 8 fine cells (preserves integral)
    @inbounds val = r_f[i1, j1, k1] + r_f[i2, j1, k1] +
                    r_f[i1, j2, k1] + r_f[i2, j2, k1] +
                    r_f[i1, j1, k2] + r_f[i2, j1, k2] +
                    r_f[i1, j2, k2] + r_f[i2, j2, k2]

    @inbounds r_c[ic, jc, kc] = val
    return
end

# ═══════════════════════════════════════════════════════════════
# 2. Prolongation: coarse → fine (injection, additive)
#    z_f is ghost-padded, z_c is ghost-padded.
#    Fine cell (if,jf,kf) += z_c[ic,jc,kc] for the parent coarse cell.
#    Kernel iterates over COARSE cells, writes to 8 fine cells.
# ═══════════════════════════════════════════════════════════════

function mg_prolongate_add!(z_f, z_c, nxc, nyc, nzc)
    ic = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    jc = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    kc = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if ic > nxc || jc > nyc || kc > nzc; return; end

    # Ghost-padded indices
    iic = ic + NG
    jjc = jc + NG
    kkc = kc + NG

    @inbounds correction = z_c[iic, jjc, kkc]

    # Fine-grid ghost-padded indices
    i1 = Int32(2) * ic - Int32(1) + NG
    i2 = Int32(2) * ic + NG
    j1 = Int32(2) * jc - Int32(1) + NG
    j2 = Int32(2) * jc + NG
    k1 = Int32(2) * kc - Int32(1) + NG
    k2 = Int32(2) * kc + NG

    # Injection: add same value to all 8 fine cells
    @inbounds z_f[i1, j1, k1] += correction
    @inbounds z_f[i2, j1, k1] += correction
    @inbounds z_f[i1, j2, k1] += correction
    @inbounds z_f[i2, j2, k1] += correction
    @inbounds z_f[i1, j1, k2] += correction
    @inbounds z_f[i2, j1, k2] += correction
    @inbounds z_f[i1, j2, k2] += correction
    @inbounds z_f[i2, j2, k2] += correction
    return
end

# ═══════════════════════════════════════════════════════════════
# 3. Coarse-grid metric computation
#    Compute coarse Vol (ghost-padded) from fine Vol.
#    Vol stores 1/Volume (vol_inv).
#    Coarse Volume = sum of 8 fine Volumes = Σ (1/vol_inv_fine)
#    Coarse vol_inv = 1 / coarse_Volume
# ═══════════════════════════════════════════════════════════════

function mg_compute_coarse_vol!(Vol_c, Vol_f, nxc, nyc, nzc)
    ic = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    jc = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    kc = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if ic > nxc || jc > nyc || kc > nzc; return; end

    # Fine-grid ghost-padded indices
    i1 = Int32(2) * ic - Int32(1) + NG
    i2 = Int32(2) * ic + NG
    j1 = Int32(2) * jc - Int32(1) + NG
    j2 = Int32(2) * jc + NG
    k1 = Int32(2) * kc - Int32(1) + NG
    k2 = Int32(2) * kc + NG

    # Sum fine volumes: V_f = 1/Vol_f (Vol stores vol_inv)
    @inbounds V_sum = one(FT) / (Vol_f[i1, j1, k1] + FT(1e-30)) +
                      one(FT) / (Vol_f[i2, j1, k1] + FT(1e-30)) +
                      one(FT) / (Vol_f[i1, j2, k1] + FT(1e-30)) +
                      one(FT) / (Vol_f[i2, j2, k1] + FT(1e-30)) +
                      one(FT) / (Vol_f[i1, j1, k2] + FT(1e-30)) +
                      one(FT) / (Vol_f[i2, j1, k2] + FT(1e-30)) +
                      one(FT) / (Vol_f[i1, j2, k2] + FT(1e-30)) +
                      one(FT) / (Vol_f[i2, j2, k2] + FT(1e-30))

    iic = ic + NG; jjc = jc + NG; kkc = kc + NG
    @inbounds Vol_c[iic, jjc, kkc] = one(FT) / (V_sum + FT(1e-30))
    return
end

# ═══════════════════════════════════════════════════════════════
# 4. Coarse-grid face areas
#    Coarse i-face between (ic-1,jc,kc) and (ic,jc,kc):
#    The fine i-face at this boundary spans 2×2 fine faces.
#    Coarse Area² ≈ sum of 4 fine Area² (for D_f = 2A²/(V_L+V_R))
#    → Areai_c = sqrt(Σ Areai_f²)
#
#    The i-face of coarse cell ic is at fine position 2*ic (0-based)
#    = fine ghost index 2*ic + NG (since fine face at ii+1 for right face)
#    Actually: the RIGHT face of fine cell 2*ic is Areai_f[2*ic+1+NG, ...]
#    which = LEFT face of fine cell 2*ic+1 = LEFT face of coarse cell ic+1
#    So coarse face between ic and ic+1 corresponds to fine face at 2*ic+NG+1
#    spanning j in (2jc-1:2jc)+NG, k in (2kc-1:2kc)+NG
# ═══════════════════════════════════════════════════════════════

function mg_compute_coarse_area_i!(Areai_c, Areai_f, nxc, nyc, nzc)
    # We need nxc+1 i-faces for nxc coarse cells
    ic = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    jc = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    kc = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if ic > nxc + Int32(1) || jc > nyc || kc > nzc; return; end

    # Fine face: at fine ghost index (2*(ic-1)+NG+1) = 2*ic - 1 + NG
    ii_f = Int32(2) * ic - Int32(1) + NG
    j1 = Int32(2) * jc - Int32(1) + NG
    j2 = Int32(2) * jc + NG
    k1 = Int32(2) * kc - Int32(1) + NG
    k2 = Int32(2) * kc + NG

    # Sum of squares of 4 fine face areas
    @inbounds a2 = Areai_f[ii_f, j1, k1]^2 + Areai_f[ii_f, j2, k1]^2 +
                   Areai_f[ii_f, j1, k2]^2 + Areai_f[ii_f, j2, k2]^2

    iic = ic + NG  # Note: this maps to face position in coarse ghost array
    jjc = jc + NG; kkc = kc + NG
    @inbounds Areai_c[iic, jjc, kkc] = sqrt(a2)
    return
end

function mg_compute_coarse_area_j!(Areaj_c, Areaj_f, nxc, nyc, nzc)
    ic = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    jc = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    kc = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if ic > nxc || jc > nyc + Int32(1) || kc > nzc; return; end

    i1 = Int32(2) * ic - Int32(1) + NG
    i2 = Int32(2) * ic + NG
    jj_f = Int32(2) * jc - Int32(1) + NG
    k1 = Int32(2) * kc - Int32(1) + NG
    k2 = Int32(2) * kc + NG

    @inbounds a2 = Areaj_f[i1, jj_f, k1]^2 + Areaj_f[i2, jj_f, k1]^2 +
                   Areaj_f[i1, jj_f, k2]^2 + Areaj_f[i2, jj_f, k2]^2

    iic = ic + NG; jjc = jc + NG; kkc = kc + NG
    @inbounds Areaj_c[iic, jjc, kkc] = sqrt(a2)
    return
end

function mg_compute_coarse_area_k!(Areak_c, Areak_f, nxc, nyc, nzc)
    ic = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    jc = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    kc = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if ic > nxc || jc > nyc || kc > nzc + Int32(1); return; end

    i1 = Int32(2) * ic - Int32(1) + NG
    i2 = Int32(2) * ic + NG
    j1 = Int32(2) * jc - Int32(1) + NG
    j2 = Int32(2) * jc + NG
    kk_f = Int32(2) * kc - Int32(1) + NG

    @inbounds a2 = Areak_f[i1, j1, kk_f]^2 + Areak_f[i2, j1, kk_f]^2 +
                   Areak_f[i1, j2, kk_f]^2 + Areak_f[i2, j2, kk_f]^2

    iic = ic + NG; jjc = jc + NG; kkc = kc + NG
    @inbounds Areak_c[iic, jjc, kkc] = sqrt(a2)
    return
end

# ═══════════════════════════════════════════════════════════════
# 5. Compute residual on a given level: res = r - A*z
#    where A*z = -L(z) = a_P*z_P - nb_sum  (positive semi-definite)
#    and r is the RHS in the same sign convention.
#    res = r - (a_P*z_P - nb_sum) = r - a_P*z_P + nb_sum
#
#    Note: this is the same as pcg_compute_residual! but operates
#    on arbitrary (level-specific) Vol/Area.
# ═══════════════════════════════════════════════════════════════

function mg_compute_residual!(res, r, z, Vol, Areai, Areaj, Areak, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end

    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds vol_inv_P = Vol[ii, jj, kk]
    V_P = one(FT) / (vol_inv_P + FT(1e-30))

    @inbounds A_iL = Areai[ii, jj, kk];   @inbounds A_iR = Areai[ii+1, jj, kk]
    V_iL = one(FT) / (@inbounds Vol[ii-1, jj, kk] + FT(1e-30))
    V_iR = one(FT) / (@inbounds Vol[ii+1, jj, kk] + FT(1e-30))
    D_iL = FT(2) * A_iL * A_iL / (V_iL + V_P)
    D_iR = FT(2) * A_iR * A_iR / (V_P + V_iR)

    @inbounds A_jB = Areaj[ii, jj, kk];   @inbounds A_jT = Areaj[ii, jj+1, kk]
    V_jB = one(FT) / (@inbounds Vol[ii, jj-1, kk] + FT(1e-30))
    V_jT = one(FT) / (@inbounds Vol[ii, jj+1, kk] + FT(1e-30))
    D_jB = FT(2) * A_jB * A_jB / (V_jB + V_P)
    D_jT = FT(2) * A_jT * A_jT / (V_P + V_jT)

    @inbounds A_kD = Areak[ii, jj, kk];   @inbounds A_kU = Areak[ii, jj, kk+1]
    V_kD = one(FT) / (@inbounds Vol[ii, jj, kk-1] + FT(1e-30))
    V_kU = one(FT) / (@inbounds Vol[ii, jj, kk+1] + FT(1e-30))
    D_kD = FT(2) * A_kD * A_kD / (V_kD + V_P)
    D_kU = FT(2) * A_kU * A_kU / (V_P + V_kU)

    a_P = D_iL + D_iR + D_jB + D_jT + D_kD + D_kU
    @inbounds nb_sum = D_iL * z[ii-1, jj, kk] + D_iR * z[ii+1, jj, kk] +
                       D_jB * z[ii, jj-1, kk] + D_jT * z[ii, jj+1, kk] +
                       D_kD * z[ii, jj, kk-1] + D_kU * z[ii, jj, kk+1]

    # A*z = a_P*z_P - nb_sum; residual = r - A*z
    @inbounds Az = a_P * z[ii, jj, kk] - nb_sum
    @inbounds res[i, j, k] = r[i, j, k] - Az
    return
end

# Copy real-cell data to ghost-padded array (for initial z)
function mg_copy_real_to_padded!(dst, src, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds dst[ii, jj, kk] = src[i, j, k]
    return
end

# ═══════════════════════════════════════════════════════════════
# 7. Neumann (zero-gradient) ghost fill for ghost-padded arrays
#    Copies boundary real cells to ghost cells: dp'/dn = 0
#    For block-local MG, this approximates the physical Neumann BC.
# ═══════════════════════════════════════════════════════════════

function mg_fill_neumann_ghost!(z, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    ng = NG

    # i-direction ghost fill
    if j >= 1 && j <= nyp + 2*ng && k >= 1 && k <= nzp + 2*ng
        if i >= 1 && i <= ng
            # Left ghost: copy from first real cell
            for g = 1:ng
                @inbounds z[g, j, k] = z[ng + 1, j, k]
            end
        end
        if i == ng + 1  # Only execute once per (j,k)
            # Right ghost: copy from last real cell
            for g = 1:ng
                @inbounds z[nxp + ng + g, j, k] = z[nxp + ng, j, k]
            end
        end
    end

    # j-direction ghost fill
    if i >= 1 && i <= nxp + 2*ng && k >= 1 && k <= nzp + 2*ng
        if j == 1
            for g = 1:ng
                @inbounds z[i, g, k] = z[i, ng + 1, k]
            end
        end
        if j == 1
            for g = 1:ng
                @inbounds z[i, nyp + ng + g, k] = z[i, nyp + ng, k]
            end
        end
    end

    # k-direction ghost fill
    if i >= 1 && i <= nxp + 2*ng && j >= 1 && j <= nyp + 2*ng
        if k == 1
            for g = 1:ng
                @inbounds z[i, j, g] = z[i, j, ng + 1]
            end
            for g = 1:ng
                @inbounds z[i, j, nzp + ng + g] = z[i, j, nzp + ng]
            end
        end
    end
    return
end

# piso_mg_driver.jl — Geometric Multigrid V-cycle driver for PISO Poisson
#
# Standalone MG solver: uses V-cycle iterations directly (not as CG preconditioner).
# Fine level uses full ghost exchange.
# Coarse levels use MPI ghost exchange (reuses fine-grid buffers).
#
# Uses kernels from piso_multigrid.jl and piso.jl (SOR smoother).

# ── Data structure for one MG level ──
struct MGLevel
    nxp::Int32
    nyp::Int32
    nzp::Int32
    z::Any       # ghost-padded: solution/correction
    r::Any       # real-cell: RHS
    res::Any     # real-cell: residual = r - A*z
    sor_rhs::Any # real-cell: SOR-format RHS (-r * vol_inv)
    Vol::Any     # ghost-padded metrics
    Areai::Any
    Areaj::Any
    Areak::Any
end

# ── Initialize MG levels from fine-grid block data ──
function mg_init_levels(b, n_levels, nthreads)
    levels = MGLevel[]

    nxf = b.Nx; nyf = b.Ny; nzf = b.Nz

    for lev = 1:n_levels
        nxc = Int32(nxf ÷ 2)
        nyc = Int32(nyf ÷ 2)
        nzc = Int32(nzf ÷ 2)

        if nxc < 4 || nyc < 4 || nzc < 4
            break
        end

        z     = gpu_zeros(FT, nxc + 2*NG, nyc + 2*NG, nzc + 2*NG)
        r     = gpu_zeros(FT, nxc, nyc, nzc)
        res   = gpu_zeros(FT, nxc, nyc, nzc)
        sor_rhs = gpu_zeros(FT, nxc, nyc, nzc)
        Vol_c   = gpu_zeros(FT, nxc + 2*NG, nyc + 2*NG, nzc + 2*NG)
        Areai_c = gpu_zeros(FT, nxc + 1 + 2*NG, nyc + 2*NG, nzc + 2*NG)
        Areaj_c = gpu_zeros(FT, nxc + 2*NG, nyc + 1 + 2*NG, nzc + 2*NG)
        Areak_c = gpu_zeros(FT, nxc + 2*NG, nyc + 2*NG, nzc + 1 + 2*NG)

        Vol_f   = lev == 1 ? b.Vol : levels[lev-1].Vol
        Areai_f = lev == 1 ? b.Areai : levels[lev-1].Areai
        Areaj_f = lev == 1 ? b.Areaj : levels[lev-1].Areaj
        Areak_f = lev == 1 ? b.Areak : levels[lev-1].Areak

        nb = (cld(Int(nxc), nthreads[1]), cld(Int(nyc), nthreads[2]), cld(Int(nzc), nthreads[3]))
        @gpu_launch threads=nthreads blocks=nb mg_compute_coarse_vol!(Vol_c, Vol_f, nxc, nyc, nzc)

        nb_i = (cld(Int(nxc)+1, nthreads[1]), cld(Int(nyc), nthreads[2]), cld(Int(nzc), nthreads[3]))
        @gpu_launch threads=nthreads blocks=nb_i mg_compute_coarse_area_i!(Areai_c, Areai_f, nxc, nyc, nzc)

        nb_j = (cld(Int(nxc), nthreads[1]), cld(Int(nyc)+1, nthreads[2]), cld(Int(nzc), nthreads[3]))
        @gpu_launch threads=nthreads blocks=nb_j mg_compute_coarse_area_j!(Areaj_c, Areaj_f, nxc, nyc, nzc)

        nb_k = (cld(Int(nxc), nthreads[1]), cld(Int(nyc), nthreads[2]), cld(Int(nzc)+1, nthreads[3]))
        @gpu_launch threads=nthreads blocks=nb_k mg_compute_coarse_area_k!(Areak_c, Areak_f, nxc, nyc, nzc)

        push!(levels, MGLevel(nxc, nyc, nzc, z, r, res, sor_rhs, Vol_c, Areai_c, Areaj_c, Areak_c))

        nxf = Int(nxc); nyf = Int(nyc); nzf = Int(nzc)
    end

    # Neumann ghost fill on coarse metrics
    gpu_sync()
    for lev in levels
        _mg_neumann_ghost_fill!(lev.Vol, Int(lev.nxp), Int(lev.nyp), Int(lev.nzp), nthreads)
    end
    gpu_sync()

    return levels
end

# ── Helper: apply Neumann ghost fill ──
function _mg_neumann_ghost_fill!(z, nxp, nyp, nzp, nthreads)
    Nx_t = nxp + 2*NG; Ny_t = nyp + 2*NG; Nz_t = nzp + 2*NG
    nb_tot = (cld(Nx_t, nthreads[1]), cld(Ny_t, nthreads[2]), cld(Nz_t, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_tot mg_fill_neumann_ghost!(z, Int32(nxp), Int32(nyp), Int32(nzp))
end

# ── Coarse-grid ghost exchange ──
# Reuses fine-grid MPI buffers (oversized) with coarse dimensions.
# Also applies Neumann ghost fill for physical boundaries.
function _mg_coarse_ghost_exchange!(lev, b, block_comms_bid, nthreads)
    nxc = Int(lev.nxp); nyc = Int(lev.nyp); nzc = Int(lev.nzp)

    # Neumann ghost fill (handles physical boundaries)
    _mg_neumann_ghost_fill!(lev.z, nxc, nyc, nzc, nthreads)

    # MPI ghost exchange (handles inter-rank boundaries)
    z_4d = reshape(lev.z, size(lev.z, 1), size(lev.z, 2), size(lev.z, 3), 1)
    exchange_ghost(z_4d, 1, block_comms_bid, nxc, nyc, nzc,
        b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
        b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
        b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
        sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
        rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2)
end

# ═══════════════════════════════════════════════════════════════════════
# Standalone MG Poisson solver (replaces PCG entirely when precond=:mg)
#
# Uses V-cycle iterations directly:
#   1. Compute residual: r = b - A*x
#   2. V-cycle correction: e ≈ A^{-1} * r
#   3. x += e
#   4. Repeat until convergence
#
# No symmetry requirement — works with any smoother and ghost exchange.
# ═══════════════════════════════════════════════════════════════════════

function mg_standalone_solve!(shared_p_prime, shared_dpdt, pcg_r, pcg_q, pcg_ztmp,
                              mg_levels_dict, blocks, nthreads,
                              ghost_exchange_fn!, block_comms,
                              max_iters, tol, tt, world_rank)
    _omega = isdefined(@__MODULE__, :piso_sor_omega) ? piso_sor_omega : FT(1.7)
    _n_pre = isdefined(@__MODULE__, :piso_mg_pre_smooth) ? piso_mg_pre_smooth : 2
    _n_post = isdefined(@__MODULE__, :piso_mg_post_smooth) ? piso_mg_post_smooth : 2
    _n_coarse = isdefined(@__MODULE__, :piso_mg_coarse_sweeps) ? piso_mg_coarse_sweeps : 20

    # ── Build RHS: SV = dpdt = Σ dUf/dA (already in shared_dpdt) ──
    # SOR-RHS = -dpdt * vol_inv (stored in pcg_q)
    for (bid, b) in blocks
        nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
        nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
        dpdt_view = @view shared_dpdt[1:nxp, 1:nyp, 1:nzp]
        rhs_view = @view pcg_q[1:nxp, 1:nyp, 1:nzp]
        @gpu_launch threads=nthreads blocks=nb_f pcg_prepare_sor_rhs!(
            rhs_view, dpdt_view, b.Vol, nxp, nyp, nzp)
    end



    # ── V-cycle iterations ──
    for iter = 1:max_iters
        # V-cycle: smooth + coarse correction on shared_p_prime
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
            levels = mg_levels_dict[bid]
            n_levels = length(levels)

            pp_view = @view shared_p_prime[1:nxp+2*NG, 1:nyp+2*NG, 1:nzp+2*NG]
            rhs_view = @view pcg_q[1:nxp, 1:nyp, 1:nzp]

            # ── Fine-level SOR smoothing ──
            for color = Int32(0):Int32(1)
                @gpu_launch threads=nthreads blocks=nb_f piso_poisson_rb_sor!(
                    pp_view, rhs_view, b.Vol, b.Areai, b.Areaj, b.Areak,
                    Int32(nxp), Int32(nyp), Int32(nzp), _omega, color)
            end
        end
        ghost_exchange_fn!(shared_p_prime)



        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
            levels = mg_levels_dict[bid]
            n_levels = length(levels)

            pp_view = @view shared_p_prime[1:nxp+2*NG, 1:nyp+2*NG, 1:nzp+2*NG]
            rhs_view = @view pcg_q[1:nxp, 1:nyp, 1:nzp]
            dpdt_view = @view shared_dpdt[1:nxp, 1:nyp, 1:nzp]

            if n_levels > 0
                # ── Compute fine residual ──
                # Use pcg_compute_residual! (proven on AMDGPU, mg_compute_residual! returns zero)
                r_view = @view pcg_r[1:nxp, 1:nyp, 1:nzp]
                @gpu_launch threads=nthreads blocks=nb_f pcg_compute_residual!(
                    r_view, pp_view, dpdt_view,
                    b.Vol, b.Areai, b.Areaj, b.Areak,
                    Int32(nxp), Int32(nyp), Int32(nzp))

                # ── Restrict → coarse ──
                lev = levels[1]
                nxc = lev.nxp; nyc = lev.nyp; nzc = lev.nzp
                nb_c = (cld(Int(nxc), nthreads[1]), cld(Int(nyc), nthreads[2]), cld(Int(nzc), nthreads[3]))
                r_c = @view lev.r[1:nxc, 1:nyc, 1:nzc]
                @gpu_launch threads=nthreads blocks=nb_c mg_restrict!(r_c, r_view, nxc, nyc, nzc)

                # ── Coarse solve ──
                _mg_solve_level!(levels, 1, blocks, bid, block_comms, nthreads,
                                 _omega, _n_pre, _n_post, _n_coarse)

                # ── Prolongate correction ──
                @gpu_launch threads=nthreads blocks=nb_c mg_prolongate_add!(pp_view, lev.z, nxc, nyc, nzc)
            end
        end

        # ── Post-smooth ──
        ghost_exchange_fn!(shared_p_prime)
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
            rhs_view = @view pcg_q[1:nxp, 1:nyp, 1:nzp]
            pp_view = @view shared_p_prime[1:nxp+2*NG, 1:nyp+2*NG, 1:nzp+2*NG]
            for color = Int32(0):Int32(1)
                @gpu_launch threads=nthreads blocks=nb_f piso_poisson_rb_sor!(
                    pp_view, rhs_view, b.Vol, b.Areai, b.Areaj, b.Areak,
                    Int32(nxp), Int32(nyp), Int32(nzp), _omega, color)
            end
        end
        ghost_exchange_fn!(shared_p_prime)

        # ── Convergence check (every 10 iterations) ──
        if iter % 10 == 0
            local_max = Float64(0.0)
            for (bid, b) in blocks
                nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
                nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
                pp_view = @view shared_p_prime[1:nxp+2*NG, 1:nyp+2*NG, 1:nzp+2*NG]
                dpdt_view = @view shared_dpdt[1:nxp, 1:nyp, 1:nzp]
                r_view = @view pcg_r[1:nxp, 1:nyp, 1:nzp]
                # Use pcg_compute_residual! (proven on AMDGPU) instead of mg_compute_residual!
                # PCG residual: r = L(p') - dpdt*V_P = (nb-a_P*p_P) - dpdt*V_P
                @gpu_launch threads=nthreads blocks=nb_f pcg_compute_residual!(
                    r_view, pp_view, dpdt_view,
                    b.Vol, b.Areai, b.Areaj, b.Areak,
                    Int32(nxp), Int32(nyp), Int32(nzp))
                gpu_sync()
                local_max = max(local_max, Float64(maximum(abs, @view pcg_r[1:nxp, 1:nyp, 1:nzp])))
            end
            global_max = MPI.Allreduce(local_max, MPI.MAX, MPI.COMM_WORLD)

            if world_rank == 0
                @printf "        [MG iter=%d] max_res=%.3e\n" iter global_max
            end
            if global_max < Float64(tol)
                if world_rank == 0
                    @printf "        [MG CONVERGED at iter=%d] max_res=%.3e < tol=%.1e\n" iter global_max tol
                end
                break
            end
        end
    end
end

# ── Recursive coarse-level solve ──
function _mg_solve_level!(levels, lev_idx, blocks, bid, block_comms, nthreads,
                          omega, n_pre, n_post, n_coarse)
    lev = levels[lev_idx]
    b = blocks[bid]
    nxp = lev.nxp; nyp = lev.nyp; nzp = lev.nzp
    nb = (cld(Int(nxp), nthreads[1]), cld(Int(nyp), nthreads[2]), cld(Int(nzp), nthreads[3]))

    # Zero z
    Nx_t = Int(nxp) + 2*NG; Ny_t = Int(nyp) + 2*NG; Nz_t = Int(nzp) + 2*NG
    nb_tot = (cld(Nx_t, nthreads[1]), cld(Ny_t, nthreads[2]), cld(Nz_t, nthreads[3]))
    z_view = @view lev.z[1:Nx_t, 1:Ny_t, 1:Nz_t]
    @gpu_launch threads=nthreads blocks=nb_tot piso_zero_field!(z_view, Int32(Nx_t), Int32(Ny_t), Int32(Nz_t))

    # Convert r to SOR-RHS
    r_view = @view lev.r[1:nxp, 1:nyp, 1:nzp]
    rhs_view = @view lev.sor_rhs[1:nxp, 1:nyp, 1:nzp]
    @gpu_launch threads=nthreads blocks=nb pcg_prepare_sor_rhs!(
        rhs_view, r_view, lev.Vol, nxp, nyp, nzp)

    is_coarsest = (lev_idx == length(levels))

    if is_coarsest
        for sweep = 1:n_coarse
            for color = Int32(0):Int32(1)
                @gpu_launch threads=nthreads blocks=nb piso_poisson_rb_sor!(
                    lev.z, rhs_view, lev.Vol, lev.Areai, lev.Areaj, lev.Areak,
                    nxp, nyp, nzp, omega, color)
            end
            _mg_coarse_ghost_exchange!(lev, b, block_comms[bid], nthreads)
        end
    else
        # Pre-smooth
        for sweep = 1:n_pre
            for color = Int32(0):Int32(1)
                @gpu_launch threads=nthreads blocks=nb piso_poisson_rb_sor!(
                    lev.z, rhs_view, lev.Vol, lev.Areai, lev.Areaj, lev.Areak,
                    nxp, nyp, nzp, omega, color)
            end
            _mg_coarse_ghost_exchange!(lev, b, block_comms[bid], nthreads)
        end

        # Compute residual on coarse level: r_c - A_c*z_c
        # NOTE: mg_compute_residual! used here because the RHS (r_view) is a restricted
        # residual, not a dpdt field — pcg_compute_residual! would apply wrong V_P scaling.
        res_view = @view lev.res[1:nxp, 1:nyp, 1:nzp]
        @gpu_launch threads=nthreads blocks=nb mg_compute_residual!(
            res_view, r_view, lev.z,
            lev.Vol, lev.Areai, lev.Areaj, lev.Areak,
            nxp, nyp, nzp)

        # Restrict
        next_lev = levels[lev_idx + 1]
        nxc = next_lev.nxp; nyc = next_lev.nyp; nzc = next_lev.nzp
        nb_c = (cld(Int(nxc), nthreads[1]), cld(Int(nyc), nthreads[2]), cld(Int(nzc), nthreads[3]))
        r_c = @view next_lev.r[1:nxc, 1:nyc, 1:nzc]
        @gpu_launch threads=nthreads blocks=nb_c mg_restrict!(r_c, res_view, nxc, nyc, nzc)

        # Recurse
        _mg_solve_level!(levels, lev_idx + 1, blocks, bid, block_comms, nthreads,
                         omega, n_pre, n_post, n_coarse)

        # Prolongate
        @gpu_launch threads=nthreads blocks=nb_c mg_prolongate_add!(lev.z, next_lev.z, nxc, nyc, nzc)

        # Post-smooth
        _mg_coarse_ghost_exchange!(lev, b, block_comms[bid], nthreads)
        for sweep = 1:n_post
            for color = Int32(0):Int32(1)
                @gpu_launch threads=nthreads blocks=nb piso_poisson_rb_sor!(
                    lev.z, rhs_view, lev.Vol, lev.Areai, lev.Areaj, lev.Areak,
                    nxp, nyp, nzp, omega, color)
            end
            _mg_coarse_ghost_exchange!(lev, b, block_comms[bid], nthreads)
        end
    end
end
