# piso_nonortho.jl — Non-orthogonality correction kernels for PISO Poisson solver
#
# On non-orthogonal meshes (e.g., O-grid pipe), the face normal S_f is not aligned
# with the cell-center vector d_f. The standard orthogonal Laplacian misses the
# cross-diffusion flux, causing div_post >> 0 even when SOR/PCG converges.
#
# Decomposition (Jasak 1996):
#   S_f = Δ_f + k_f
# where Δ_f is along d (implicit, handled by existing SOR D_f)
# and k_f is the cross-diffusion vector (explicit correction).
#
# The correction flux per cell:
#   E_P = Σ_faces_outward [k_f · (∇p')_f]
# is added to the Poisson RHS via deferred (defect) correction.

# ═══════════════════════════════════════════════════════════════
# 1. Precompute k_f = S_f - D_f × d for each face
#    S_f = A_f × n̂,  D_f = 2A²/(V_L+V_R),  d = x_R - x_L
#    k_f is geometry-only → computed once at initialization.
# ═══════════════════════════════════════════════════════════════

# --- I-faces: between cell (ii_L=i+NG-1) and cell (ii_R=i+NG) ---
function piso_precompute_kf_i!(ki_x, ki_y, ki_z,
    x, y, z_coord, nxi, nyi, nzi, Areai, Vol,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp + Int32(1) || j > nyp || k > nzp; return; end
    # Skip boundary faces: ghost cell metrics may be wrong (extrapolated coordinates)
    if i < Int32(2) || i > nxp; return; end

    ii_L = i + Int32(NG) - Int32(1)   # left cell ghost index
    ii_R = i + Int32(NG)              # right cell ghost index
    jj = j + Int32(NG);  kk = k + Int32(NG)
    fi = ii_R   # face stored at Areai[ii_R, jj, kk]

    # Face area vector S_f = A_f × n̂
    @inbounds A_f = Areai[fi, jj, kk]
    @inbounds Sx = nxi[fi, jj, kk] * A_f
    @inbounds Sy = nyi[fi, jj, kk] * A_f
    @inbounds Sz = nzi[fi, jj, kk] * A_f

    # Cell-center vector d = center(R) - center(L)
    @inbounds dx = x[ii_R, jj, kk] - x[ii_L, jj, kk]
    @inbounds dy = y[ii_R, jj, kk] - y[ii_L, jj, kk]
    @inbounds dz = z_coord[ii_R, jj, kk] - z_coord[ii_L, jj, kk]

    # D_f = 2A²/(V_L + V_R) — same as SOR kernel
    @inbounds V_L = one(FT) / (Vol[ii_L, jj, kk] + FT(1.0e-30))
    @inbounds V_R = one(FT) / (Vol[ii_R, jj, kk] + FT(1.0e-30))
    D_f = FT(2.0) * A_f * A_f / (V_L + V_R + FT(1.0e-30))

    # k_f = S_f - D_f × d
    @inbounds ki_x[i, j, k] = Sx - D_f * dx
    @inbounds ki_y[i, j, k] = Sy - D_f * dy
    @inbounds ki_z[i, j, k] = Sz - D_f * dz
    return
end

# --- J-faces: between cell (jj_B=j+NG-1) and cell (jj_T=j+NG) ---
function piso_precompute_kf_j!(kj_x, kj_y, kj_z,
    x, y, z_coord, nxj, nyj, nzj, Areaj, Vol,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp + Int32(1) || k > nzp; return; end
    # Skip boundary faces: ghost cell metrics may be wrong (extrapolated coordinates)
    if j < Int32(2) || j > nyp; return; end

    ii = i + Int32(NG)
    jj_B = j + Int32(NG) - Int32(1)
    jj_T = j + Int32(NG)
    kk = k + Int32(NG)
    fj = jj_T   # face stored at Areaj[ii, jj_T, kk]

    @inbounds A_f = Areaj[ii, fj, kk]
    @inbounds Sx = nxj[ii, fj, kk] * A_f
    @inbounds Sy = nyj[ii, fj, kk] * A_f
    @inbounds Sz = nzj[ii, fj, kk] * A_f

    @inbounds dx = x[ii, jj_T, kk] - x[ii, jj_B, kk]
    @inbounds dy = y[ii, jj_T, kk] - y[ii, jj_B, kk]
    @inbounds dz = z_coord[ii, jj_T, kk] - z_coord[ii, jj_B, kk]

    @inbounds V_B = one(FT) / (Vol[ii, jj_B, kk] + FT(1.0e-30))
    @inbounds V_T = one(FT) / (Vol[ii, jj_T, kk] + FT(1.0e-30))
    D_f = FT(2.0) * A_f * A_f / (V_B + V_T + FT(1.0e-30))

    @inbounds kj_x[i, j, k] = Sx - D_f * dx
    @inbounds kj_y[i, j, k] = Sy - D_f * dy
    @inbounds kj_z[i, j, k] = Sz - D_f * dz
    return
end

# --- K-faces: between cell (kk_D=k+NG-1) and cell (kk_U=k+NG) ---
function piso_precompute_kf_k!(kk_x, kk_y, kk_z,
    x, y, z_coord, nxk, nyk, nzk, Areak, Vol,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp + Int32(1); return; end
    # Skip boundary faces: ghost cell metrics may be wrong (extrapolated coordinates)
    if k < Int32(2) || k > nzp; return; end

    ii = i + Int32(NG); jj = j + Int32(NG)
    kk_D = k + Int32(NG) - Int32(1)
    kk_U = k + Int32(NG)
    fk = kk_U   # face stored at Areak[ii, jj, kk_U]

    @inbounds A_f = Areak[ii, jj, fk]
    @inbounds Sx = nxk[ii, jj, fk] * A_f
    @inbounds Sy = nyk[ii, jj, fk] * A_f
    @inbounds Sz = nzk[ii, jj, fk] * A_f

    @inbounds dx = x[ii, jj, kk_U] - x[ii, jj, kk_D]
    @inbounds dy = y[ii, jj, kk_U] - y[ii, jj, kk_D]
    @inbounds dz = z_coord[ii, jj, kk_U] - z_coord[ii, jj, kk_D]

    @inbounds V_D = one(FT) / (Vol[ii, jj, kk_D] + FT(1.0e-30))
    @inbounds V_U = one(FT) / (Vol[ii, jj, kk_U] + FT(1.0e-30))
    D_f = FT(2.0) * A_f * A_f / (V_D + V_U + FT(1.0e-30))

    @inbounds kk_x[i, j, k] = Sx - D_f * dx
    @inbounds kk_y[i, j, k] = Sy - D_f * dy
    @inbounds kk_z[i, j, k] = Sz - D_f * dz
    return
end

# ═══════════════════════════════════════════════════════════════
# 2. Compute Green-Gauss gradient of p' at cell centers
#    ∇p'_P = (1/V) × Σ_faces [p'_face × S_f]
#    Output: ghost-padded grad_px/py/pz arrays.
#    Computed at real cells + 1 ghost layer (ii=NG..nxp+NG+1).
# ═══════════════════════════════════════════════════════════════

function piso_compute_grad_p!(grad_px, grad_py, grad_pz, p_prime,
    Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk, Vol,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    # Extended range: compute at real cells + 1 ghost layer on each side
    if i > nxp + Int32(2) || j > nyp + Int32(2) || k > nzp + Int32(2); return; end

    # Ghost index: i=1 → ii=NG, i=nxp+2 → ii=nxp+NG+1
    ii = i + Int32(NG) - Int32(1)
    jj = j + Int32(NG) - Int32(1)
    kk = k + Int32(NG) - Int32(1)

    # Bounds check: need neighbors at ii±1, jj±1, kk±1
    if ii < Int32(2) || ii > nxp + Int32(2) * Int32(NG) - Int32(1); return; end
    if jj < Int32(2) || jj > nyp + Int32(2) * Int32(NG) - Int32(1); return; end
    if kk < Int32(2) || kk > nzp + Int32(2) * Int32(NG) - Int32(1); return; end

    @inbounds pP = p_prime[ii, jj, kk]

    # I-direction faces
    @inbounds pL = p_prime[ii-1, jj, kk]; p_iL = FT(0.5) * (pL + pP)
    @inbounds pR = p_prime[ii+1, jj, kk]; p_iR = FT(0.5) * (pP + pR)
    @inbounds A_iL = Areai[ii, jj, kk];   @inbounds A_iR = Areai[ii+1, jj, kk]

    @inbounds dpdx = p_iR * nxi[ii+1, jj, kk] * A_iR - p_iL * nxi[ii, jj, kk] * A_iL
    @inbounds dpdy = p_iR * nyi[ii+1, jj, kk] * A_iR - p_iL * nyi[ii, jj, kk] * A_iL
    @inbounds dpdz = p_iR * nzi[ii+1, jj, kk] * A_iR - p_iL * nzi[ii, jj, kk] * A_iL

    # J-direction faces
    @inbounds pB = p_prime[ii, jj-1, kk]; p_jB = FT(0.5) * (pB + pP)
    @inbounds pT = p_prime[ii, jj+1, kk]; p_jT = FT(0.5) * (pP + pT)
    @inbounds A_jB = Areaj[ii, jj, kk];   @inbounds A_jT = Areaj[ii, jj+1, kk]

    @inbounds dpdx += p_jT * nxj[ii, jj+1, kk] * A_jT - p_jB * nxj[ii, jj, kk] * A_jB
    @inbounds dpdy += p_jT * nyj[ii, jj+1, kk] * A_jT - p_jB * nyj[ii, jj, kk] * A_jB
    @inbounds dpdz += p_jT * nzj[ii, jj+1, kk] * A_jT - p_jB * nzj[ii, jj, kk] * A_jB

    # K-direction faces
    @inbounds pD = p_prime[ii, jj, kk-1]; p_kD = FT(0.5) * (pD + pP)
    @inbounds pU = p_prime[ii, jj, kk+1]; p_kU = FT(0.5) * (pP + pU)
    @inbounds A_kD = Areak[ii, jj, kk];   @inbounds A_kU = Areak[ii, jj, kk+1]

    @inbounds dpdx += p_kU * nxk[ii, jj, kk+1] * A_kU - p_kD * nxk[ii, jj, kk] * A_kD
    @inbounds dpdy += p_kU * nyk[ii, jj, kk+1] * A_kU - p_kD * nyk[ii, jj, kk] * A_kD
    @inbounds dpdz += p_kU * nzk[ii, jj, kk+1] * A_kU - p_kD * nzk[ii, jj, kk] * A_kD

    # Green-Gauss: ∇p' = (1/V) × Σ [p'_f × S_f]
    @inbounds vi = Vol[ii, jj, kk]
    @inbounds grad_px[ii, jj, kk] = dpdx * vi
    @inbounds grad_py[ii, jj, kk] = dpdy * vi
    @inbounds grad_pz[ii, jj, kk] = dpdz * vi
    return
end

# ═══════════════════════════════════════════════════════════════
# 3. Compute non-orthogonality correction source:
#    E_P = Σ_faces_outward [k_f · (∇p')_f]
#
#    For each face, (∇p')_f = 0.5 × ((∇p')_P + (∇p')_N)
#    Sign convention:
#      Right face: +k_iR · (∇p')_R  (outward)
#      Left face:  -k_iL · (∇p')_L  (outward = -inward)
# ═══════════════════════════════════════════════════════════════

function piso_compute_noc_source!(noc_E, grad_px, grad_py, grad_pz,
    ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, kkf_x, kkf_y, kkf_z,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end

    ii = i + Int32(NG); jj = j + Int32(NG); kk = k + Int32(NG)

    # Cell-center gradient at P
    @inbounds gx_P = grad_px[ii, jj, kk]
    @inbounds gy_P = grad_py[ii, jj, kk]
    @inbounds gz_P = grad_pz[ii, jj, kk]

    # ── I-direction ──
    # Right i-face (face index i+1 in real-face 1-based): outward flux
    @inbounds gx_fR = FT(0.5) * (gx_P + grad_px[ii+1, jj, kk])
    @inbounds gy_fR = FT(0.5) * (gy_P + grad_py[ii+1, jj, kk])
    @inbounds gz_fR = FT(0.5) * (gz_P + grad_pz[ii+1, jj, kk])
    @inbounds flux_iR = ki_x[i+1, j, k] * gx_fR + ki_y[i+1, j, k] * gy_fR + ki_z[i+1, j, k] * gz_fR

    # Left i-face (face index i): inward flux → negate for outward
    @inbounds gx_fL = FT(0.5) * (grad_px[ii-1, jj, kk] + gx_P)
    @inbounds gy_fL = FT(0.5) * (grad_py[ii-1, jj, kk] + gy_P)
    @inbounds gz_fL = FT(0.5) * (grad_pz[ii-1, jj, kk] + gz_P)
    @inbounds flux_iL = ki_x[i, j, k] * gx_fL + ki_y[i, j, k] * gy_fL + ki_z[i, j, k] * gz_fL

    # ── J-direction ──
    # Top j-face (face index j+1): outward
    @inbounds gx_fT = FT(0.5) * (gx_P + grad_px[ii, jj+1, kk])
    @inbounds gy_fT = FT(0.5) * (gy_P + grad_py[ii, jj+1, kk])
    @inbounds gz_fT = FT(0.5) * (gz_P + grad_pz[ii, jj+1, kk])
    @inbounds flux_jT = kj_x[i, j+1, k] * gx_fT + kj_y[i, j+1, k] * gy_fT + kj_z[i, j+1, k] * gz_fT

    # Bottom j-face (face index j): inward → negate
    @inbounds gx_fB = FT(0.5) * (grad_px[ii, jj-1, kk] + gx_P)
    @inbounds gy_fB = FT(0.5) * (grad_py[ii, jj-1, kk] + gy_P)
    @inbounds gz_fB = FT(0.5) * (gz_P + grad_pz[ii, jj-1, kk])
    @inbounds flux_jB = kj_x[i, j, k] * gx_fB + kj_y[i, j, k] * gy_fB + kj_z[i, j, k] * gz_fB

    # ── K-direction ──
    # Up k-face (face index k+1): outward
    @inbounds gx_fU = FT(0.5) * (gx_P + grad_px[ii, jj, kk+1])
    @inbounds gy_fU = FT(0.5) * (gy_P + grad_py[ii, jj, kk+1])
    @inbounds gz_fU = FT(0.5) * (gz_P + grad_pz[ii, jj, kk+1])
    @inbounds flux_kU = kkf_x[i, j, k+1] * gx_fU + kkf_y[i, j, k+1] * gy_fU + kkf_z[i, j, k+1] * gz_fU

    # Down k-face (face index k): inward → negate
    @inbounds gx_fD = FT(0.5) * (grad_px[ii, jj, kk-1] + gx_P)
    @inbounds gy_fD = FT(0.5) * (grad_py[ii, jj, kk-1] + gy_P)
    @inbounds gz_fD = FT(0.5) * (gz_P + grad_pz[ii, jj, kk-1])
    @inbounds flux_kD = kkf_x[i, j, k] * gx_fD + kkf_y[i, j, k] * gy_fD + kkf_z[i, j, k] * gz_fD

    # E_P = Σ outward k·grad = (right - left) per direction
    @inbounds noc_E[i, j, k] = (flux_iR - flux_iL) + (flux_jT - flux_jB) + (flux_kU - flux_kD)
    return
end

# ═══════════════════════════════════════════════════════════════
# 4. Correct RHS for defect correction:
#    dpdt_corrected[i,j,k] = -noc_E[i,j,k] × vol_inv
#    This is stored DIRECTLY into dpdt for the correction PCG solve.
#    (The correction solves: A*δp = -E, so RHS = -E × vol_inv)
# ═══════════════════════════════════════════════════════════════

function piso_noc_prepare_correction_rhs!(dpdt, noc_E, Vol, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end

    ii = i + Int32(NG); jj = j + Int32(NG); kk = k + Int32(NG)
    @inbounds dpdt[i, j, k] = -noc_E[i, j, k] * Vol[ii, jj, kk]
    return
end

# ═══════════════════════════════════════════════════════════════
# 5. Add correction to p': p_prime += delta_p
#    Both arrays are ghost-padded.
# ═══════════════════════════════════════════════════════════════

function piso_add_correction!(p_prime, delta_p, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end

    ii = i + Int32(NG); jj = j + Int32(NG); kk = k + Int32(NG)
    @inbounds p_prime[ii, jj, kk] += delta_p[ii, jj, kk]
    return
end

# ═══════════════════════════════════════════════════════════════
# 6. Non-orthogonal velocity correction:
#    ΔUf -= (dt/ρ) × k_f · (∇p')_f / A_f
#
#    This adds the cross-diffusion gradient contribution that the
#    standard orthogonal correction misses. Together with the
#    standard correction, div(u_corrected) → 0.
# ═══════════════════════════════════════════════════════════════

function piso_nonortho_correct_velocity_i!(Uf_i, grad_px, grad_py, grad_pz,
    ki_x, ki_y, ki_z, Areai, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp + Int32(1) || j > nyp || k > nzp; return; end

    ii_L = i + Int32(NG) - Int32(1)
    ii_R = i + Int32(NG)
    jj = j + Int32(NG); kk = k + Int32(NG)
    fi = ii_R

    # Face-interpolated gradient
    @inbounds gx_f = FT(0.5) * (grad_px[ii_L, jj, kk] + grad_px[ii_R, jj, kk])
    @inbounds gy_f = FT(0.5) * (grad_py[ii_L, jj, kk] + grad_py[ii_R, jj, kk])
    @inbounds gz_f = FT(0.5) * (grad_pz[ii_L, jj, kk] + grad_pz[ii_R, jj, kk])

    # k_f · (∇p')_f
    @inbounds kf_dot_grad = ki_x[i, j, k] * gx_f + ki_y[i, j, k] * gy_f + ki_z[i, j, k] * gz_f

    # ΔUf = -(dt/ρ) × k_f · (∇p')_f / A_f
    inv_rho = one(FT) / ρ_ref
    @inbounds A_f = Areai[fi, jj, kk]
    @inbounds Uf_i[i, j, k] -= dt * inv_rho * kf_dot_grad / (A_f + FT(1.0e-30))
    return
end

function piso_nonortho_correct_velocity_j!(Uf_j, grad_px, grad_py, grad_pz,
    kj_x, kj_y, kj_z, Areaj, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp + Int32(1) || k > nzp; return; end

    ii = i + Int32(NG)
    jj_B = j + Int32(NG) - Int32(1)
    jj_T = j + Int32(NG)
    kk = k + Int32(NG)
    fj = jj_T

    @inbounds gx_f = FT(0.5) * (grad_px[ii, jj_B, kk] + grad_px[ii, jj_T, kk])
    @inbounds gy_f = FT(0.5) * (grad_py[ii, jj_B, kk] + grad_py[ii, jj_T, kk])
    @inbounds gz_f = FT(0.5) * (grad_pz[ii, jj_B, kk] + grad_pz[ii, jj_T, kk])

    @inbounds kf_dot_grad = kj_x[i, j, k] * gx_f + kj_y[i, j, k] * gy_f + kj_z[i, j, k] * gz_f

    inv_rho = one(FT) / ρ_ref
    @inbounds A_f = Areaj[ii, fj, kk]
    @inbounds Uf_j[i, j, k] -= dt * inv_rho * kf_dot_grad / (A_f + FT(1.0e-30))
    return
end

function piso_nonortho_correct_velocity_k!(Uf_k, grad_px, grad_py, grad_pz,
    kkf_x, kkf_y, kkf_z, Areak, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp + Int32(1); return; end

    ii = i + Int32(NG); jj = j + Int32(NG)
    kk_D = k + Int32(NG) - Int32(1)
    kk_U = k + Int32(NG)
    fk = kk_U

    @inbounds gx_f = FT(0.5) * (grad_px[ii, jj, kk_D] + grad_px[ii, jj, kk_U])
    @inbounds gy_f = FT(0.5) * (grad_py[ii, jj, kk_D] + grad_py[ii, jj, kk_U])
    @inbounds gz_f = FT(0.5) * (grad_pz[ii, jj, kk_D] + grad_pz[ii, jj, kk_U])

    @inbounds kf_dot_grad = kkf_x[i, j, k] * gx_f + kkf_y[i, j, k] * gy_f + kkf_z[i, j, k] * gz_f

    inv_rho = one(FT) / ρ_ref
    @inbounds A_f = Areak[ii, jj, fk]
    @inbounds Uf_k[i, j, k] -= dt * inv_rho * kf_dot_grad / (A_f + FT(1.0e-30))
    return
end
