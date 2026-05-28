# piso.jl — PISO (Pressure-Implicit with Splitting of Operators) kernels
#
# Staggered-grid fractional-step method for incompressible Navier-Stokes.
# Replaces the Artificial Compressibility (AC) method.
#
# Algorithm per time step:
#   1. Momentum predictor: u* = u^n + Δt × RHS(u^n, p^n)
#   2. First pressure correction:
#      - Compute div(u*)
#      - Solve ∇²p' = (ρ/Δt) × div(u*)  via Red-Black SOR
#      - Correct: u** = u* - (Δt/ρ) × ∇p'
#      - Update:  p** = p^n + p'
#   3. Second pressure correction (PISO):
#      - Recompute momentum with u**: u*** = u^n + Δt × RHS(u**, p**)
#      - Compute div(u***)
#      - Solve ∇²p'' = (ρ/Δt) × div(u***)
#      - Correct: u^{n+1} = u*** - (Δt/ρ) × ∇p''
#      - Update:  p^{n+1} = p** + p''
#
# Data layout (same as ac_staggered.jl):
#   Pressure p        → Q[i,j,k,1]  (cell center, ghost-padded)
#   Cell-center vel   → Q[i,j,k,2:4] (diagnostic, averaged from face velocities)
#   I-face velocity   → Uf_i[i,j,k]  (i=1..nxp+1, j=1..nyp, k=1..nzp)
#   J-face velocity   → Uf_j[i,j,k]  (i=1..nxp, j=1..nyp+1, k=1..nzp)
#   K-face velocity   → Uf_k[i,j,k]  (i=1..nxp, j=1..nyp, k=1..nzp+1)
#
# NOTE: NG is defined globally in the run config — do NOT redefine here.

# ═══════════════════════════════════════════════════════════════
# 1. Divergence of face velocity field
#    div(u) = (1/V) × Σ_faces(Uf × A)
#
#    Uses SIMPLE Uf×A form (one Cartesian component per face).
#    This is EXACTLY consistent with the SOR D_f coefficients
#    and the single-component velocity correction.
#    For non-orthogonal grids, this ignores cross-direction
#    contributions but guarantees div=0 after correction.
# ═══════════════════════════════════════════════════════════════

function piso_compute_divergence!(div_u, Q, Uf_i, Uf_j, Uf_k,
    Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk, Vol,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii = i + NG; jj = j + NG; kk = k + NG

    # Simple Uf×A form: exactly consistent with PISO velocity correction
    # I-faces
    @inbounds flux_iL = Uf_i[i, j, k]   * Areai[ii, jj, kk]
    @inbounds flux_iR = Uf_i[i+1, j, k] * Areai[ii+1, jj, kk]

    # J-faces
    @inbounds flux_jB = Uf_j[i, j, k]   * Areaj[ii, jj, kk]
    @inbounds flux_jT = Uf_j[i, j+1, k] * Areaj[ii, jj+1, kk]

    # K-faces
    @inbounds flux_kD = Uf_k[i, j, k]   * Areak[ii, jj, kk]
    @inbounds flux_kU = Uf_k[i, j, k+1] * Areak[ii, jj, kk+1]

    # Divergence = (1/V) × Σ(outward fluxes)
    @inbounds vol_inv = Vol[ii, jj, kk]
    @inbounds div_u[i, j, k] = (flux_iR - flux_iL + flux_jT - flux_jB + flux_kU - flux_kD) * vol_inv
    return
end

# ═══════════════════════════════════════════════════════════════
# 2. Red-Black SOR for pressure Poisson equation
#    ∇²p' = S   where S = (ρ/Δt) × div(u*)
#
#    FVM discretization:
#      Σ_faces [D_f × (p'_N - p'_P)] = S_P × V_P
#    where D_f = 2 × A_f² / (V_L + V_R) is the face diffusion coefficient
#
#    SOR update (only for cells with (i+j+k)%2 == color):
#      p'_P = (1-ω) × p'_old + ω × [Σ D_f × p'_N - S_P × V_P] / (Σ D_f)
# ═══════════════════════════════════════════════════════════════

function piso_poisson_rb_sor!(p_prime, rhs, Vol, Areai, Areaj, Areak,
    nxp, nyp, nzp, omega, color)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    # Red-Black: only update cells with (i+j+k) % 2 == color
    if mod(i + j + k, Int32(2)) != color
        return
    end

    ii = i + NG; jj = j + NG; kk = k + NG

    # Cell volume (V = 1/vol_inv)
    @inbounds vol_inv_P = Vol[ii, jj, kk]
    V_P = one(FT) / (vol_inv_P + FT(1.0e-30))

    # ── Compute face diffusion coefficients D_f = 2A²/(V_L+V_R) ──
    # I-direction (left face: between ig-1 and ig, right face: between ig and ig+1)
    @inbounds A_iL = Areai[ii, jj, kk]
    @inbounds A_iR = Areai[ii+1, jj, kk]
    @inbounds V_iL_inv = Vol[ii-1, jj, kk]
    @inbounds V_iR_inv = Vol[ii+1, jj, kk]
    V_iL = one(FT) / (V_iL_inv + FT(1.0e-30))
    V_iR = one(FT) / (V_iR_inv + FT(1.0e-30))
    D_iL = FT(2.0) * A_iL * A_iL / (V_iL + V_P)
    D_iR = FT(2.0) * A_iR * A_iR / (V_P + V_iR)

    # J-direction
    @inbounds A_jB = Areaj[ii, jj, kk]
    @inbounds A_jT = Areaj[ii, jj+1, kk]
    @inbounds V_jB_inv = Vol[ii, jj-1, kk]
    @inbounds V_jT_inv = Vol[ii, jj+1, kk]
    V_jB = one(FT) / (V_jB_inv + FT(1.0e-30))
    V_jT = one(FT) / (V_jT_inv + FT(1.0e-30))
    D_jB = FT(2.0) * A_jB * A_jB / (V_jB + V_P)
    D_jT = FT(2.0) * A_jT * A_jT / (V_P + V_jT)

    # K-direction
    @inbounds A_kD = Areak[ii, jj, kk]
    @inbounds A_kU = Areak[ii, jj, kk+1]
    @inbounds V_kD_inv = Vol[ii, jj, kk-1]
    @inbounds V_kU_inv = Vol[ii, jj, kk+1]
    V_kD = one(FT) / (V_kD_inv + FT(1.0e-30))
    V_kU = one(FT) / (V_kU_inv + FT(1.0e-30))
    D_kD = FT(2.0) * A_kD * A_kD / (V_kD + V_P)
    D_kU = FT(2.0) * A_kU * A_kU / (V_P + V_kU)

    # Diagonal coefficient
    a_P = D_iL + D_iR + D_jB + D_jT + D_kD + D_kU

    # Neighbor contributions
    @inbounds nb_sum = D_iL * p_prime[ii-1, jj, kk] +
                       D_iR * p_prime[ii+1, jj, kk] +
                       D_jB * p_prime[ii, jj-1, kk] +
                       D_jT * p_prime[ii, jj+1, kk] +
                       D_kD * p_prime[ii, jj, kk-1] +
                       D_kU * p_prime[ii, jj, kk+1]

    # Volume-integrated source: S_P × V_P
    @inbounds S_V = rhs[i, j, k] * V_P

    # SOR update
    p_new = (nb_sum - S_V) / (a_P + FT(1.0e-30))
    @inbounds p_prime[ii, jj, kk] = (one(FT) - omega) * p_prime[ii, jj, kk] + omega * p_new
    return
end

# ═══════════════════════════════════════════════════════════════
# 2b. Poisson residual: r = Σ D_f × (p'_N - p'_P) - RHS × V
#     If SOR converged, max|r| ≈ 0.
# ═══════════════════════════════════════════════════════════════

function piso_poisson_residual!(residual, p_prime, rhs, Vol, Areai, Areaj, Areak,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end

    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds vol_inv_P = Vol[ii, jj, kk]
    V_P = one(FT) / (vol_inv_P + FT(1.0e-30))

    @inbounds A_iL = Areai[ii, jj, kk];   @inbounds A_iR = Areai[ii+1, jj, kk]
    V_iL = one(FT) / (@inbounds Vol[ii-1, jj, kk] + FT(1.0e-30))
    V_iR = one(FT) / (@inbounds Vol[ii+1, jj, kk] + FT(1.0e-30))
    D_iL = FT(2.0) * A_iL * A_iL / (V_iL + V_P)
    D_iR = FT(2.0) * A_iR * A_iR / (V_P + V_iR)

    @inbounds A_jB = Areaj[ii, jj, kk];   @inbounds A_jT = Areaj[ii, jj+1, kk]
    V_jB = one(FT) / (@inbounds Vol[ii, jj-1, kk] + FT(1.0e-30))
    V_jT = one(FT) / (@inbounds Vol[ii, jj+1, kk] + FT(1.0e-30))
    D_jB = FT(2.0) * A_jB * A_jB / (V_jB + V_P)
    D_jT = FT(2.0) * A_jT * A_jT / (V_P + V_jT)

    @inbounds A_kD = Areak[ii, jj, kk];   @inbounds A_kU = Areak[ii, jj, kk+1]
    V_kD = one(FT) / (@inbounds Vol[ii, jj, kk-1] + FT(1.0e-30))
    V_kU = one(FT) / (@inbounds Vol[ii, jj, kk+1] + FT(1.0e-30))
    D_kD = FT(2.0) * A_kD * A_kD / (V_kD + V_P)
    D_kU = FT(2.0) * A_kU * A_kU / (V_P + V_kU)

    a_P = D_iL + D_iR + D_jB + D_jT + D_kD + D_kU
    @inbounds nb_sum = D_iL * p_prime[ii-1, jj, kk] + D_iR * p_prime[ii+1, jj, kk] +
                       D_jB * p_prime[ii, jj-1, kk] + D_jT * p_prime[ii, jj+1, kk] +
                       D_kD * p_prime[ii, jj, kk-1] + D_kU * p_prime[ii, jj, kk+1]
    @inbounds S_V = rhs[i, j, k] * V_P

    # residual = Laplacian(p') - rhs = (nb_sum - a_P*p'_P) - S_V
    #          should be 0 if converged
    @inbounds residual[i, j, k] = (nb_sum - a_P * p_prime[ii, jj, kk]) - S_V
    return
end

# ═══════════════════════════════════════════════════════════════
# 3. Velocity correction
#    Two-part correction for consistency with full dot-product divergence:
#    (a) Face velocity:    Uf -= (dt/ρ) × dp'/ds  (stored component)
#    (b) Cell-center Q:    Q[2:4] -= (dt/ρ) × ∇p' (Green-Gauss gradient)
#
#    Part (a) corrects the STORED Cartesian component at each face.
#    Part (b) corrects the cell-center Q so that the INTERPOLATED
#    cross-direction components at faces are also updated.
#    Together, the full dot-product divergence (u·n)×A is correctly
#    reduced to zero by the Poisson pressure correction.
#
#    ds_f ≈ V_avg / A_f  (face-normal distance between cell centers)
# ═══════════════════════════════════════════════════════════════

# --- 3a. Face velocity corrections (same as before) ---

function piso_correct_velocity_i!(Uf_i, p_prime, Vol, Areai, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+1 || j > nyp || k > nzp
        return
    end

    ii = i + NG - 1  # Left cell global index
    jj = j + NG
    kk = k + NG

    # Pressure correction gradient at i-face
    @inbounds p_L = p_prime[ii, jj, kk]
    @inbounds p_R = p_prime[ii+1, jj, kk]

    # Face-normal distance: ds = V_avg / A
    @inbounds vol_inv_L = Vol[ii, jj, kk]
    @inbounds vol_inv_R = Vol[ii+1, jj, kk]
    V_L = one(FT) / (vol_inv_L + FT(1.0e-30))
    V_R = one(FT) / (vol_inv_R + FT(1.0e-30))
    @inbounds A_f = Areai[ii+1, jj, kk]
    ds = FT(0.5) * (V_L + V_R) / (A_f + FT(1.0e-14))

    # Correction: u -= (dt/ρ) × dp'/ds
    inv_rho = one(FT) / ρ_ref
    @inbounds Uf_i[i, j, k] -= dt * inv_rho * (p_R - p_L) / ds
    return
end

function piso_correct_velocity_j!(Uf_j, p_prime, Vol, Areaj, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp+1 || k > nzp
        return
    end

    ii = i + NG
    jj = j + NG - 1  # Bottom cell global index
    kk = k + NG

    @inbounds p_B = p_prime[ii, jj, kk]
    @inbounds p_T = p_prime[ii, jj+1, kk]

    @inbounds vol_inv_B = Vol[ii, jj, kk]
    @inbounds vol_inv_T = Vol[ii, jj+1, kk]
    V_B = one(FT) / (vol_inv_B + FT(1.0e-30))
    V_T = one(FT) / (vol_inv_T + FT(1.0e-30))
    @inbounds A_f = Areaj[ii, jj+1, kk]
    ds = FT(0.5) * (V_B + V_T) / (A_f + FT(1.0e-14))

    inv_rho = one(FT) / ρ_ref
    @inbounds Uf_j[i, j, k] -= dt * inv_rho * (p_T - p_B) / ds
    return
end

function piso_correct_velocity_k!(Uf_k, p_prime, Vol, Areak, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp+1
        return
    end

    ii = i + NG
    jj = j + NG
    kk = k + NG - 1  # Down cell global index

    @inbounds p_D = p_prime[ii, jj, kk]
    @inbounds p_U = p_prime[ii, jj, kk+1]

    @inbounds vol_inv_D = Vol[ii, jj, kk]
    @inbounds vol_inv_U = Vol[ii, jj, kk+1]
    V_D = one(FT) / (vol_inv_D + FT(1.0e-30))
    V_U = one(FT) / (vol_inv_U + FT(1.0e-30))
    @inbounds A_f = Areak[ii, jj, kk+1]
    ds = FT(0.5) * (V_D + V_U) / (A_f + FT(1.0e-14))

    inv_rho = one(FT) / ρ_ref
    @inbounds Uf_k[i, j, k] -= dt * inv_rho * (p_U - p_D) / ds
    return
end

# --- 3b. Cell-center velocity correction via Green-Gauss gradient ---
# Corrects Q[2:4] (cell-center u,v,w) so that cross-direction velocity
# components interpolated from Q at faces are properly updated.
# ∂p'/∂x_i ≈ (1/V) × Σ_faces [ p'_face × n_i × A ]
# where p'_face = 0.5*(p'_here + p'_neighbor)

function piso_correct_cell_center!(Q, p_prime,
    Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk, Vol,
    dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii = i + NG; jj = j + NG; kk = k + NG

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

    # Green-Gauss: ∇p' = (1/V) × Σ [p'_f × n × A]
    @inbounds vol_inv = Vol[ii, jj, kk]
    dpdx *= vol_inv
    dpdy *= vol_inv
    dpdz *= vol_inv

    # Apply correction: u -= (dt/ρ) × ∇p'
    inv_rho = one(FT) / ρ_ref
    @inbounds Q[ii, jj, kk, 2] -= dt * inv_rho * dpdx
    @inbounds Q[ii, jj, kk, 3] -= dt * inv_rho * dpdy
    @inbounds Q[ii, jj, kk, 4] -= dt * inv_rho * dpdz
    return
end

# ═══════════════════════════════════════════════════════════════
# 4. Pressure correction: p = p_old + p'
#    Applied at cell centers (ghost-padded Q array)
# ═══════════════════════════════════════════════════════════════

function piso_correct_pressure!(Q, p_prime, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds Q[ii, jj, kk, 1] += p_prime[ii, jj, kk]
    return
end

# NOTE: max|div_u| is computed on CPU via:
#   div_h = Array(div_u)
#   local_max = maximum(abs, div_h)
#   global_max = MPI.Allreduce(local_max, MPI.MAX, MPI.COMM_WORLD)
# This avoids GPU atomics and matches the existing codebase pattern.

# ═══════════════════════════════════════════════════════════════
# 6. Zero out p' field (ghost-padded)
# ═══════════════════════════════════════════════════════════════

function piso_zero_field!(p_prime, nx_tot, ny_tot, nz_tot)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nx_tot || j > ny_tot || k > nz_tot
        return
    end
    @inbounds p_prime[i, j, k] = zero(FT)
    return
end

# ═══════════════════════════════════════════════════════════════
# 6b. Subtract mean from p' (remove Neumann null-space constant)
#     Prevents unbounded pressure drift in Q[1] += p' accumulation.
#     Called HOST-SIDE after SOR convergence.
# ═══════════════════════════════════════════════════════════════

function piso_subtract_mean!(p_prime, mean_val, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds p_prime[ii, jj, kk] -= mean_val
    return
end

# ═══════════════════════════════════════════════════════════════
# 7. Pressure correction Neumann BC (dp'/dn = 0)
#    Reuse the existing ac_pressure_neumann_i/j/k! kernels from
#    ac_staggered.jl — they operate on Q[:,:,:,1] but we apply
#    them to p_prime with the same interface.
# ═══════════════════════════════════════════════════════════════

function piso_pressure_neumann_i!(p_prime, nxp, nyp, nzp, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > 1 || j > nyp+2*NG || k > nzp+2*NG
        return
    end
    if side == Int32(0)
        for g = Int32(1):Int32(NG)
            @inbounds p_prime[NG+1-g, j, k] = p_prime[NG+1, j, k]
        end
    else
        for g = Int32(1):Int32(NG)
            @inbounds p_prime[nxp+NG+g, j, k] = p_prime[nxp+NG, j, k]
        end
    end
    return
end

function piso_pressure_neumann_j!(p_prime, nxp, nyp, nzp, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > 1 || k > nzp+2*NG
        return
    end
    if side == Int32(0)
        for g = Int32(1):Int32(NG)
            @inbounds p_prime[i, NG+1-g, k] = p_prime[i, NG+1, k]
        end
    else
        for g = Int32(1):Int32(NG)
            @inbounds p_prime[i, nyp+NG+g, k] = p_prime[i, nyp+NG, k]
        end
    end
    return
end

function piso_pressure_neumann_k!(p_prime, nxp, nyp, nzp, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > 1
        return
    end
    if side == Int32(0)
        for g = Int32(1):Int32(NG)
            @inbounds p_prime[i, j, NG+1-g] = p_prime[i, j, NG+1]
        end
    else
        for g = Int32(1):Int32(NG)
            @inbounds p_prime[i, j, nzp+NG+g] = p_prime[i, j, nzp+NG]
        end
    end
    return
end

# ═══════════════════════════════════════════════════════════════
# 8. Host-side helper: apply p' Neumann BCs at ALL faces
#    For inter-block boundaries: Neumann (dp'/dn=0) is the correct
#    approximation when inter-block p' ghost exchange is not available.
#    For periodic faces: Neumann is applied first, then exchange_ghost
#    overwrites with correct periodic values.
#    For walls: Neumann is the physical BC for pressure correction.
# ═══════════════════════════════════════════════════════════════

function piso_apply_pprime_bc!(b, bid, face_bc, p_prime)
    nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
    nthreads_bc = (16, 8, 8)
    Ny_tot = nyp + 2*NG; Nz_tot = nzp + 2*NG; Nx_tot = nxp + 2*NG

    # I-faces (ξ direction) — always apply Neumann; periodic ξ exchange_ghost overwrites later
    nb_i = (1, cld(Ny_tot, nthreads_bc[2]), cld(Nz_tot, nthreads_bc[3]))
    @gpu_launch threads=nthreads_bc blocks=nb_i piso_pressure_neumann_i!(p_prime, nxp, nyp, nzp, Int32(0))
    @gpu_launch threads=nthreads_bc blocks=nb_i piso_pressure_neumann_i!(p_prime, nxp, nyp, nzp, Int32(1))

    # J-faces (η direction) — ONLY apply Neumann at physical walls, NOT interblock
    # face_bc[(bid, fid)] stores Int: 0=interblock, 1=wall, 2=periodic, etc.
    bc_jm = get(face_bc, (bid, 3), Int(BC_INTERBLOCK))
    bc_jp = get(face_bc, (bid, 4), Int(BC_INTERBLOCK))
    nb_j = (cld(Nx_tot, nthreads_bc[1]), 1, cld(Nz_tot, nthreads_bc[3]))
    if bc_jm != Int(BC_INTERBLOCK)
        @gpu_launch threads=nthreads_bc blocks=nb_j piso_pressure_neumann_j!(p_prime, nxp, nyp, nzp, Int32(0))
    end
    if bc_jp != Int(BC_INTERBLOCK)
        @gpu_launch threads=nthreads_bc blocks=nb_j piso_pressure_neumann_j!(p_prime, nxp, nyp, nzp, Int32(1))
    end

    # K-faces (ζ direction) — ONLY apply Neumann at physical walls, NOT interblock
    bc_km = get(face_bc, (bid, 5), Int(BC_INTERBLOCK))
    bc_kp = get(face_bc, (bid, 6), Int(BC_INTERBLOCK))
    nb_k = (cld(Nx_tot, nthreads_bc[1]), cld(Ny_tot, nthreads_bc[2]), 1)
    if bc_km != Int(BC_INTERBLOCK)
        @gpu_launch threads=nthreads_bc blocks=nb_k piso_pressure_neumann_k!(p_prime, nxp, nyp, nzp, Int32(0))
    end
    if bc_kp != Int(BC_INTERBLOCK)
        @gpu_launch threads=nthreads_bc blocks=nb_k piso_pressure_neumann_k!(p_prime, nxp, nyp, nzp, Int32(1))
    end
end

# ═══════════════════════════════════════════════════════════════
# 9. Host-side helper: exchange p' ghost cells via MPI
#    Wraps the p_prime 3D array into a 4D view for exchange_ghost
# ═══════════════════════════════════════════════════════════════

function piso_exchange_pprime_ghost!(p_prime, b, block_comms, bid)
    p4d = reshape(p_prime, size(p_prime, 1), size(p_prime, 2), size(p_prime, 3), 1)
    exchange_ghost(p4d, 1, block_comms[bid], b.Nx, b.Ny, b.Nz,
        b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
        b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
        b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
        sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
        rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2)
end

# ═══════════════════════════════════════════════════════════════
# 10. Momentum predictor: explicit forward Euler
#     u* = u^n + Δt × RHS(u^n, p^n)
#     Reuses existing ac_momentum_rhs_i/j/k_staggered! for RHS.
#     This is a host-side wrapper that does: compute RHS → update Uf
# ═══════════════════════════════════════════════════════════════

function piso_euler_update_uf!(Uf, dUf, Uf_n, dt, n1, n2, n3)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > n1 || j > n2 || k > n3
        return
    end
    # u* = u^n + dt × RHS
    @inbounds Uf[i, j, k] = Uf_n[i, j, k] + dt * dUf[i, j, k]
    return
end

# ═══════════════════════════════════════════════════════════════
# 11. Compute Poisson RHS: rhs = (ρ/Δt) × div(u*)
# ═══════════════════════════════════════════════════════════════

function piso_scale_rhs!(rhs, div_u, rho_over_dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    @inbounds rhs[i, j, k] = rho_over_dt * div_u[i, j, k]
    return
end
