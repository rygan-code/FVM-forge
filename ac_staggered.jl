# ac_staggered.jl — Staggered-grid (MAC-like) Artificial Compressibility solver
#
# Data layout:
#   Pressure p        → Q[i,j,k,1]  (cell center, ghost-padded)
#   Cell-center vel   → Q[i,j,k,2:4] (diagnostic only, averaged from face velocities)
#   I-face velocity   → Uf_i[i,j,k]  (i=1..nxp+1, j=1..nyp, k=1..nzp, real faces only)
#   J-face velocity   → Uf_j[i,j,k]  (i=1..nxp, j=1..nyp+1, k=1..nzp)
#   K-face velocity   → Uf_k[i,j,k]  (i=1..nxp, j=1..nyp, k=1..nzp+1)
#
# Equations:
#   Pressure:  dp/dt + β²/V * Σ_faces(Uf * A) = 0
#   Momentum:  dUf/dt + F_conv_skew + (p_R-p_L)/(ρ*Δs) = F_visc + F_source
#
# Convective term (skew-symmetric, full 3D):
#   F_conv_skew = 0.5 * (F_conservative + F_advective)
#   F_conservative = ∇·(u⊗u)  — momentum-conserving
#   F_advective    = u·∇u     — kinetic-energy-conserving
#
# NOTE: NG is defined globally in the run config — do NOT redefine here.

# ═══════════════════════════════════════════════════════════════
# Face velocity ↔ Cell-center velocity interpolation
# ═══════════════════════════════════════════════════════════════

function ac_face_to_cell_vel!(Q, Uf_i, Uf_j, Uf_k, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds Q[ii, jj, kk, 2] = FT(0.5) * (Uf_i[i, j, k] + Uf_i[i+1, j, k])
    @inbounds Q[ii, jj, kk, 3] = FT(0.5) * (Uf_j[i, j, k] + Uf_j[i, j+1, k])
    @inbounds Q[ii, jj, kk, 4] = FT(0.5) * (Uf_k[i, j, k] + Uf_k[i, j, k+1])
    return
end

function ac_cell_to_face_vel!(Uf_i, Uf_j, Uf_k, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i <= nxp+1 && j <= nyp && k <= nzp
        ii = i + NG - 1; jj = j + NG; kk = k + NG
        @inbounds Uf_i[i, j, k] = FT(0.5) * (Q[ii, jj, kk, 2] + Q[ii+1, jj, kk, 2])
    end
    if i <= nxp && j <= nyp+1 && k <= nzp
        ii2 = i + NG; jj2 = j + NG - 1; kk2 = k + NG
        @inbounds Uf_j[i, j, k] = FT(0.5) * (Q[ii2, jj2, kk2, 3] + Q[ii2, jj2+1, kk2, 3])
    end
    if i <= nxp && j <= nyp && k <= nzp+1
        ii3 = i + NG; jj3 = j + NG; kk3 = k + NG - 1
        @inbounds Uf_k[i, j, k] = FT(0.5) * (Q[ii3, jj3, kk3, 4] + Q[ii3, jj3, kk3+1, 4])
    end
    return
end

# ═══════════════════════════════════════════════════════════════
# Pressure equation: dp/dt = -β²/V * Σ_faces(u·n * A)
# Full dot product u·n = u*nx + v*ny + w*nz at each face
# (correct for curvilinear grids where normals have mixed components)
# Missing velocity components are interpolated from cell-center Q.
# ═══════════════════════════════════════════════════════════════

function ac_pressure_rhs_staggered!(dpdt, Q, Uf_i, Uf_j, Uf_k,
    Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk, Vol,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii = i + NG; jj = j + NG; kk = k + NG

    @inbounds vol_inv = Vol[ii, jj, kk]

    # ── I-faces: Uf_i is the stored u-velocity; interpolate v,w from Q ──
    @inbounds uf_L = Uf_i[i, j, k]
    @inbounds v_iL = FT(0.5) * (Q[ii-1, jj, kk, 3] + Q[ii, jj, kk, 3])
    @inbounds w_iL = FT(0.5) * (Q[ii-1, jj, kk, 4] + Q[ii, jj, kk, 4])
    @inbounds uf_R = Uf_i[i+1, j, k]
    @inbounds v_iR = FT(0.5) * (Q[ii, jj, kk, 3] + Q[ii+1, jj, kk, 3])
    @inbounds w_iR = FT(0.5) * (Q[ii, jj, kk, 4] + Q[ii+1, jj, kk, 4])

    @inbounds aL = Areai[ii, jj, kk];   nxL = nxi[ii, jj, kk];   nyL = nyi[ii, jj, kk];   nzL = nzi[ii, jj, kk]
    @inbounds aR = Areai[ii+1, jj, kk]; nxR = nxi[ii+1, jj, kk]; nyR = nyi[ii+1, jj, kk]; nzR = nzi[ii+1, jj, kk]
    flux_iL = (uf_L*nxL + v_iL*nyL + w_iL*nzL) * aL
    flux_iR = (uf_R*nxR + v_iR*nyR + w_iR*nzR) * aR

    # ── J-faces: Uf_j is the stored v-velocity; interpolate u,w from Q ──
    @inbounds vf_B = Uf_j[i, j, k]
    @inbounds u_jB = FT(0.5) * (Q[ii, jj-1, kk, 2] + Q[ii, jj, kk, 2])
    @inbounds w_jB = FT(0.5) * (Q[ii, jj-1, kk, 4] + Q[ii, jj, kk, 4])
    @inbounds vf_T = Uf_j[i, j+1, k]
    @inbounds u_jT = FT(0.5) * (Q[ii, jj, kk, 2] + Q[ii, jj+1, kk, 2])
    @inbounds w_jT = FT(0.5) * (Q[ii, jj, kk, 4] + Q[ii, jj+1, kk, 4])

    @inbounds aB = Areaj[ii, jj, kk];   nxB = nxj[ii, jj, kk];   nyB = nyj[ii, jj, kk];   nzB = nzj[ii, jj, kk]
    @inbounds aT = Areaj[ii, jj+1, kk]; nxT = nxj[ii, jj+1, kk]; nyT = nyj[ii, jj+1, kk]; nzT = nzj[ii, jj+1, kk]
    flux_jB = (u_jB*nxB + vf_B*nyB + w_jB*nzB) * aB
    flux_jT = (u_jT*nxT + vf_T*nyT + w_jT*nzT) * aT

    # ── K-faces: Uf_k is the stored w-velocity; interpolate u,v from Q ──
    @inbounds wf_D = Uf_k[i, j, k]
    @inbounds u_kD = FT(0.5) * (Q[ii, jj, kk-1, 2] + Q[ii, jj, kk, 2])
    @inbounds v_kD = FT(0.5) * (Q[ii, jj, kk-1, 3] + Q[ii, jj, kk, 3])
    @inbounds wf_U = Uf_k[i, j, k+1]
    @inbounds u_kU = FT(0.5) * (Q[ii, jj, kk, 2] + Q[ii, jj, kk+1, 2])
    @inbounds v_kU = FT(0.5) * (Q[ii, jj, kk, 3] + Q[ii, jj, kk+1, 3])

    @inbounds aD = Areak[ii, jj, kk];   nxD = nxk[ii, jj, kk];   nyD = nyk[ii, jj, kk];   nzD = nzk[ii, jj, kk]
    @inbounds aU = Areak[ii, jj, kk+1]; nxU = nxk[ii, jj, kk+1]; nyU = nyk[ii, jj, kk+1]; nzU = nzk[ii, jj, kk+1]
    flux_kD = (u_kD*nxD + v_kD*nyD + wf_D*nzD) * aD
    flux_kU = (u_kU*nxU + v_kU*nyU + wf_U*nzU) * aU

    # Divergence: Σ(outward fluxes)
    div_u = (flux_iR - flux_iL) + (flux_jT - flux_jB) + (flux_kU - flux_kD)

    β2 = β_AC * β_AC
    @inbounds dpdt[i, j, k] = -β2 * div_u * vol_inv
    return
end

# ═══════════════════════════════════════════════════════════════
# Momentum equation: I-face (u-component)
# Full 3D: dUf_i/dt + 0.5*(∇·(uu) + u·∇u) + ∂p/∂x/ρ = ν∇²u + f
#   includes cross-direction convection (v·∂u/∂y, w·∂u/∂z)
#   and full Laplacian viscosity (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
# ═══════════════════════════════════════════════════════════════

function ac_momentum_rhs_i_staggered!(dUf_i, Q, Uf_i, Uf_j, Uf_k,
    Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk, Vol,
    nxp, nyp, nzp, f1_val, dt_val)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+1 || j > nyp || k > nzp
        return
    end

    ii = i + NG - 1
    jj = j + NG
    kk = k + NG

    # ── Pressure gradient ──
    @inbounds p_L = Q[ii, jj, kk, 1]
    @inbounds p_R = Q[ii+1, jj, kk, 1]

    @inbounds vol_inv_L = Vol[ii, jj, kk]
    @inbounds vol_inv_R = Vol[ii+1, jj, kk]
    vol_L = one(FT) / (vol_inv_L + FT(1.0e-30))
    vol_R = one(FT) / (vol_inv_R + FT(1.0e-30))
    vol_half = FT(0.5) * (vol_L + vol_R)

    @inbounds area_i = Areai[ii+1, jj, kk]

    ds_i = vol_half / (area_i + FT(1.0e-14))

    inv_ρ = one(FT) / ρ_ref
    dp_dx = (p_R - p_L) * inv_ρ / ds_i

    # ── Cell-center velocities at I-face (average of left/right cells) ──
    @inbounds u_L = Q[ii, jj, kk, 2]; @inbounds u_R = Q[ii+1, jj, kk, 2]
    @inbounds v_L = Q[ii, jj, kk, 3]; @inbounds v_R = Q[ii+1, jj, kk, 3]
    @inbounds w_L = Q[ii, jj, kk, 4]; @inbounds w_R = Q[ii+1, jj, kk, 4]
    u_avg = FT(0.5) * (u_L + u_R)
    v_avg = FT(0.5) * (v_L + v_R)
    w_avg = FT(0.5) * (w_L + w_R)

    # ── Cross-direction grid spacing (for j and k derivatives) ──
    @inbounds area_j_avg = FT(0.25) * (Areaj[ii,jj,kk] + Areaj[ii+1,jj,kk] +
                                       Areaj[ii,jj+1,kk] + Areaj[ii+1,jj+1,kk])
    @inbounds area_k_avg = FT(0.25) * (Areak[ii,jj,kk] + Areak[ii+1,jj,kk] +
                                       Areak[ii,jj,kk+1] + Areak[ii+1,jj,kk+1])
    ds_j = vol_half / (area_j_avg + FT(1.0e-14))
    ds_k = vol_half / (area_k_avg + FT(1.0e-14))

    # ── u interpolated to I-face at j±1, k±1 neighbors ──
    @inbounds u_jp = FT(0.5) * (Q[ii,jj+1,kk,2] + Q[ii+1,jj+1,kk,2])
    @inbounds u_jm = FT(0.5) * (Q[ii,jj-1,kk,2] + Q[ii+1,jj-1,kk,2])
    @inbounds u_kp = FT(0.5) * (Q[ii,jj,kk+1,2] + Q[ii+1,jj,kk+1,2])
    @inbounds u_km = FT(0.5) * (Q[ii,jj,kk-1,2] + Q[ii+1,jj,kk-1,2])

    # ══════════════════════════════════════════════
    # Skew-symmetric convective term (full 3D)
    # ══════════════════════════════════════════════

    # ── I-direction (self): ∂(uu)/∂x ──
    F_cons_i = zero(FT)
    F_adv_i = zero(FT)
    if i > 1 && i <= nxp
        @inbounds uf_Lm = Uf_i[i-1, j, k]
        @inbounds uf_Rp = Uf_i[i+1, j, k]
        @inbounds aL2 = Areai[ii, jj, kk]
        @inbounds aR2 = Areai[ii+2, jj, kk]
        @inbounds u_cell_LL = Q[ii-1, jj, kk, 2]
        @inbounds u_cell_RR = Q[ii+2, jj, kk, 2]
        u_face_L = FT(0.5) * (u_cell_LL + u_L)
        u_face_R = FT(0.5) * (u_R + u_cell_RR)
        F_cons_i = (uf_Rp * u_face_R * aR2 - uf_Lm * u_face_L * aL2) / (vol_half + FT(1.0e-30))
    end
    if i > 1 && i < nxp+1
        @inbounds u_L2 = Q[ii-1, jj, kk, 2]; @inbounds u_R2 = Q[ii+2, jj, kk, 2]
        F_adv_i = u_avg * (u_R2 - u_L2) / (FT(2.0) * ds_i)
    end

    # ── J-direction (cross): ∂(vu)/∂y ──
    # Conservative: flux of u-momentum through j-faces of dual cell
    @inbounds v_jp = FT(0.5) * (Q[ii,jj+1,kk,3] + Q[ii+1,jj+1,kk,3])
    @inbounds v_jm = FT(0.5) * (Q[ii,jj-1,kk,3] + Q[ii+1,jj-1,kk,3])
    vu_top = FT(0.5) * (v_avg + v_jp) * FT(0.5) * (u_avg + u_jp)
    vu_bot = FT(0.5) * (v_avg + v_jm) * FT(0.5) * (u_avg + u_jm)
    @inbounds A_j_top = FT(0.5) * (Areaj[ii,jj+1,kk] + Areaj[ii+1,jj+1,kk])
    @inbounds A_j_bot = FT(0.5) * (Areaj[ii,jj,kk] + Areaj[ii+1,jj,kk])
    F_cons_j = (vu_top * A_j_top - vu_bot * A_j_bot) / (vol_half + FT(1.0e-30))
    # Advective: v·∂u/∂y
    F_adv_j = v_avg * (u_jp - u_jm) / (FT(2.0) * ds_j)

    # ── K-direction (cross): ∂(wu)/∂z ──
    @inbounds w_kp = FT(0.5) * (Q[ii,jj,kk+1,4] + Q[ii+1,jj,kk+1,4])
    @inbounds w_km = FT(0.5) * (Q[ii,jj,kk-1,4] + Q[ii+1,jj,kk-1,4])
    wu_top = FT(0.5) * (w_avg + w_kp) * FT(0.5) * (u_avg + u_kp)
    wu_bot = FT(0.5) * (w_avg + w_km) * FT(0.5) * (u_avg + u_km)
    @inbounds A_k_top = FT(0.5) * (Areak[ii,jj,kk+1] + Areak[ii+1,jj,kk+1])
    @inbounds A_k_bot = FT(0.5) * (Areak[ii,jj,kk] + Areak[ii+1,jj,kk])
    F_cons_k = (wu_top * A_k_top - wu_bot * A_k_bot) / (vol_half + FT(1.0e-30))
    # Advective: w·∂u/∂z
    F_adv_k = w_avg * (u_kp - u_km) / (FT(2.0) * ds_k)

    F_conv = FT(0.5) * ((F_cons_i + F_cons_j + F_cons_k) +
                       (F_adv_i  + F_adv_j  + F_adv_k))

    # ══════════════════════════════════════════════
    # Viscous term: full Laplacian ν(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    # ══════════════════════════════════════════════
    mu = ρ_ref * ν_AC
    d2u_dx2 = zero(FT)
    if i > 1 && i < nxp+1
        @inbounds u_L2 = Q[ii-1, jj, kk, 2]; @inbounds u_R2 = Q[ii+2, jj, kk, 2]
        d2u_dx2 = (u_R2 - FT(2.0) * u_avg + u_L2) / (ds_i * ds_i)
    end
    d2u_dy2 = (u_jp - FT(2.0) * u_avg + u_jm) / (ds_j * ds_j)
    d2u_dz2 = (u_kp - FT(2.0) * u_avg + u_km) / (ds_k * ds_k)
    F_visc = mu * (d2u_dx2 + d2u_dy2 + d2u_dz2)

    @inbounds dUf_i[i, j, k] = -F_conv - dp_dx + F_visc + f1_val
    return
end

# ═══════════════════════════════════════════════════════════════
# Momentum equation: J-face (v-component)
# Full 3D: dUf_j/dt + 0.5*(∇·(vu) + u·∇v) + ∂p/∂y/ρ = ν∇²v
# ═══════════════════════════════════════════════════════════════

function ac_momentum_rhs_j_staggered!(dUf_j, Q, Uf_i, Uf_j, Uf_k,
    Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk, Vol,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp+1 || k > nzp
        return
    end

    ii = i + NG
    jj = j + NG - 1
    kk = k + NG

    # ── Pressure gradient ──
    @inbounds p_B = Q[ii, jj, kk, 1]
    @inbounds p_T = Q[ii, jj+1, kk, 1]

    @inbounds vol_inv_B = Vol[ii, jj, kk]
    @inbounds vol_inv_T = Vol[ii, jj+1, kk]
    vol_B = one(FT) / (vol_inv_B + FT(1.0e-30))
    vol_T = one(FT) / (vol_inv_T + FT(1.0e-30))
    vol_half = FT(0.5) * (vol_B + vol_T)

    @inbounds area_j = Areaj[ii, jj+1, kk]

    ds_j = vol_half / (area_j + FT(1.0e-14))

    inv_ρ = one(FT) / ρ_ref
    dp_dy = (p_T - p_B) * inv_ρ / ds_j

    # ── Cell-center velocities at J-face (average of bottom/top cells) ──
    @inbounds u_B = Q[ii, jj, kk, 2]; @inbounds u_T = Q[ii, jj+1, kk, 2]
    @inbounds v_B = Q[ii, jj, kk, 3]; @inbounds v_T = Q[ii, jj+1, kk, 3]
    @inbounds w_B = Q[ii, jj, kk, 4]; @inbounds w_T = Q[ii, jj+1, kk, 4]
    u_avg = FT(0.5) * (u_B + u_T)
    v_avg = FT(0.5) * (v_B + v_T)
    w_avg = FT(0.5) * (w_B + w_T)

    # ── Cross-direction grid spacing (for i and k derivatives) ──
    @inbounds area_i_avg = FT(0.25) * (Areai[ii,jj,kk] + Areai[ii,jj+1,kk] +
                                       Areai[ii+1,jj,kk] + Areai[ii+1,jj+1,kk])
    @inbounds area_k_avg = FT(0.25) * (Areak[ii,jj,kk] + Areak[ii,jj+1,kk] +
                                       Areak[ii,jj,kk+1] + Areak[ii,jj+1,kk+1])
    ds_i = vol_half / (area_i_avg + FT(1.0e-14))
    ds_k = vol_half / (area_k_avg + FT(1.0e-14))

    # ── v interpolated to J-face at i±1, k±1 neighbors ──
    @inbounds v_ip = FT(0.5) * (Q[ii+1,jj,kk,3] + Q[ii+1,jj+1,kk,3])
    @inbounds v_im = FT(0.5) * (Q[ii-1,jj,kk,3] + Q[ii-1,jj+1,kk,3])
    @inbounds v_kp = FT(0.5) * (Q[ii,jj,kk+1,3] + Q[ii,jj+1,kk+1,3])
    @inbounds v_km = FT(0.5) * (Q[ii,jj,kk-1,3] + Q[ii,jj+1,kk-1,3])

    # ══════════════════════════════════════════════
    # Skew-symmetric convective term (full 3D)
    # ══════════════════════════════════════════════

    # ── J-direction (self): ∂(vv)/∂y ──
    F_cons_j = zero(FT)
    F_adv_j = zero(FT)
    if j > 1 && j <= nyp
        @inbounds vf_Bm = Uf_j[i, j-1, k]
        @inbounds vf_Tp = Uf_j[i, j+1, k]
        @inbounds aB2 = Areaj[ii, jj, kk]
        @inbounds aT2 = Areaj[ii, jj+2, kk]
        @inbounds v_cell_BB = Q[ii, jj-1, kk, 3]
        @inbounds v_cell_TT = Q[ii, jj+2, kk, 3]
        v_face_B = FT(0.5) * (v_cell_BB + v_B)
        v_face_T = FT(0.5) * (v_T + v_cell_TT)
        F_cons_j = (vf_Tp * v_face_T * aT2 - vf_Bm * v_face_B * aB2) / (vol_half + FT(1.0e-30))
    end
    if j > 1 && j < nyp+1
        @inbounds v_B2 = Q[ii, jj-1, kk, 3]; @inbounds v_T2 = Q[ii, jj+2, kk, 3]
        F_adv_j = v_avg * (v_T2 - v_B2) / (FT(2.0) * ds_j)
    end

    # ── I-direction (cross): ∂(uv)/∂x ──
    @inbounds u_ip = FT(0.5) * (Q[ii+1,jj,kk,2] + Q[ii+1,jj+1,kk,2])
    @inbounds u_im = FT(0.5) * (Q[ii-1,jj,kk,2] + Q[ii-1,jj+1,kk,2])
    uv_right = FT(0.5) * (u_avg + u_ip) * FT(0.5) * (v_avg + v_ip)
    uv_left  = FT(0.5) * (u_avg + u_im) * FT(0.5) * (v_avg + v_im)
    @inbounds A_i_right = FT(0.5) * (Areai[ii+1,jj,kk] + Areai[ii+1,jj+1,kk])
    @inbounds A_i_left  = FT(0.5) * (Areai[ii,jj,kk] + Areai[ii,jj+1,kk])
    F_cons_i = (uv_right * A_i_right - uv_left * A_i_left) / (vol_half + FT(1.0e-30))
    F_adv_i = u_avg * (v_ip - v_im) / (FT(2.0) * ds_i)

    # ── K-direction (cross): ∂(wv)/∂z ──
    @inbounds w_kp2 = FT(0.5) * (Q[ii,jj,kk+1,4] + Q[ii,jj+1,kk+1,4])
    @inbounds w_km2 = FT(0.5) * (Q[ii,jj,kk-1,4] + Q[ii,jj+1,kk-1,4])
    wv_top = FT(0.5) * (w_avg + w_kp2) * FT(0.5) * (v_avg + v_kp)
    wv_bot = FT(0.5) * (w_avg + w_km2) * FT(0.5) * (v_avg + v_km)
    @inbounds A_k_top = FT(0.5) * (Areak[ii,jj,kk+1] + Areak[ii,jj+1,kk+1])
    @inbounds A_k_bot = FT(0.5) * (Areak[ii,jj,kk] + Areak[ii,jj+1,kk])
    F_cons_k = (wv_top * A_k_top - wv_bot * A_k_bot) / (vol_half + FT(1.0e-30))
    F_adv_k = w_avg * (v_kp - v_km) / (FT(2.0) * ds_k)

    F_conv = FT(0.5) * ((F_cons_j + F_cons_i + F_cons_k) +
                       (F_adv_j  + F_adv_i  + F_adv_k))

    # ══════════════════════════════════════════════
    # Viscous term: full Laplacian ν(∂²v/∂x² + ∂²v/∂y² + ∂²v/∂z²)
    # ══════════════════════════════════════════════
    mu = ρ_ref * ν_AC
    d2v_dy2 = zero(FT)
    if j > 1 && j < nyp+1
        @inbounds v_B2 = Q[ii, jj-1, kk, 3]; @inbounds v_T2 = Q[ii, jj+2, kk, 3]
        d2v_dy2 = (v_T2 - FT(2.0) * v_avg + v_B2) / (ds_j * ds_j)
    end
    d2v_dx2 = (v_ip - FT(2.0) * v_avg + v_im) / (ds_i * ds_i)
    d2v_dz2 = (v_kp - FT(2.0) * v_avg + v_km) / (ds_k * ds_k)
    F_visc = mu * (d2v_dx2 + d2v_dy2 + d2v_dz2)

    @inbounds dUf_j[i, j, k] = -F_conv - dp_dy + F_visc
    return
end

# ═══════════════════════════════════════════════════════════════
# Momentum equation: K-face (w-component)
# Full 3D: dUf_k/dt + 0.5*(∇·(wu) + u·∇w) + ∂p/∂z/ρ = ν∇²w
# ═══════════════════════════════════════════════════════════════

function ac_momentum_rhs_k_staggered!(dUf_k, Q, Uf_i, Uf_j, Uf_k,
    Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk, Vol,
    nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp+1
        return
    end

    ii = i + NG
    jj = j + NG
    kk = k + NG - 1

    # ── Pressure gradient ──
    @inbounds p_D = Q[ii, jj, kk, 1]
    @inbounds p_U = Q[ii, jj, kk+1, 1]

    @inbounds vol_inv_D = Vol[ii, jj, kk]
    @inbounds vol_inv_U = Vol[ii, jj, kk+1]
    vol_D = one(FT) / (vol_inv_D + FT(1.0e-30))
    vol_U = one(FT) / (vol_inv_U + FT(1.0e-30))
    vol_half = FT(0.5) * (vol_D + vol_U)

    @inbounds area_k = Areak[ii, jj, kk+1]

    ds_k = vol_half / (area_k + FT(1.0e-14))

    inv_ρ = one(FT) / ρ_ref
    dp_dz = (p_U - p_D) * inv_ρ / ds_k

    # ── Cell-center velocities at K-face (average of down/up cells) ──
    @inbounds u_D = Q[ii, jj, kk, 2]; @inbounds u_U = Q[ii, jj, kk+1, 2]
    @inbounds v_D = Q[ii, jj, kk, 3]; @inbounds v_U = Q[ii, jj, kk+1, 3]
    @inbounds w_D = Q[ii, jj, kk, 4]; @inbounds w_U = Q[ii, jj, kk+1, 4]
    u_avg = FT(0.5) * (u_D + u_U)
    v_avg = FT(0.5) * (v_D + v_U)
    w_avg = FT(0.5) * (w_D + w_U)

    # ── Cross-direction grid spacing (for i and j derivatives) ──
    @inbounds area_i_avg = FT(0.25) * (Areai[ii,jj,kk] + Areai[ii,jj,kk+1] +
                                       Areai[ii+1,jj,kk] + Areai[ii+1,jj,kk+1])
    @inbounds area_j_avg = FT(0.25) * (Areaj[ii,jj,kk] + Areaj[ii,jj,kk+1] +
                                       Areaj[ii,jj+1,kk] + Areaj[ii,jj+1,kk+1])
    ds_i = vol_half / (area_i_avg + FT(1.0e-14))
    ds_j = vol_half / (area_j_avg + FT(1.0e-14))

    # ── w interpolated to K-face at i±1, j±1 neighbors ──
    @inbounds w_ip = FT(0.5) * (Q[ii+1,jj,kk,4] + Q[ii+1,jj,kk+1,4])
    @inbounds w_im = FT(0.5) * (Q[ii-1,jj,kk,4] + Q[ii-1,jj,kk+1,4])
    @inbounds w_jp = FT(0.5) * (Q[ii,jj+1,kk,4] + Q[ii,jj+1,kk+1,4])
    @inbounds w_jm = FT(0.5) * (Q[ii,jj-1,kk,4] + Q[ii,jj-1,kk+1,4])

    # ══════════════════════════════════════════════
    # Skew-symmetric convective term (full 3D)
    # ══════════════════════════════════════════════

    # ── K-direction (self): ∂(ww)/∂z ──
    F_cons_k = zero(FT)
    F_adv_k = zero(FT)
    if k > 1 && k <= nzp
        @inbounds wf_Dm = Uf_k[i, j, k-1]
        @inbounds wf_Up = Uf_k[i, j, k+1]
        @inbounds aD2 = Areak[ii, jj, kk]
        @inbounds aU2 = Areak[ii, jj, kk+2]
        @inbounds w_cell_DD = Q[ii, jj, kk-1, 4]
        @inbounds w_cell_UU = Q[ii, jj, kk+2, 4]
        w_face_D = FT(0.5) * (w_cell_DD + w_D)
        w_face_U = FT(0.5) * (w_U + w_cell_UU)
        F_cons_k = (wf_Up * w_face_U * aU2 - wf_Dm * w_face_D * aD2) / (vol_half + FT(1.0e-30))
    end
    if k > 1 && k < nzp+1
        @inbounds w_D2 = Q[ii, jj, kk-1, 4]; @inbounds w_U2 = Q[ii, jj, kk+2, 4]
        F_adv_k = w_avg * (w_U2 - w_D2) / (FT(2.0) * ds_k)
    end

    # ── I-direction (cross): ∂(uw)/∂x ──
    @inbounds u_ip = FT(0.5) * (Q[ii+1,jj,kk,2] + Q[ii+1,jj,kk+1,2])
    @inbounds u_im = FT(0.5) * (Q[ii-1,jj,kk,2] + Q[ii-1,jj,kk+1,2])
    uw_right = FT(0.5) * (u_avg + u_ip) * FT(0.5) * (w_avg + w_ip)
    uw_left  = FT(0.5) * (u_avg + u_im) * FT(0.5) * (w_avg + w_im)
    @inbounds A_i_right = FT(0.5) * (Areai[ii+1,jj,kk] + Areai[ii+1,jj,kk+1])
    @inbounds A_i_left  = FT(0.5) * (Areai[ii,jj,kk] + Areai[ii,jj,kk+1])
    F_cons_i = (uw_right * A_i_right - uw_left * A_i_left) / (vol_half + FT(1.0e-30))
    F_adv_i = u_avg * (w_ip - w_im) / (FT(2.0) * ds_i)

    # ── J-direction (cross): ∂(vw)/∂y ──
    @inbounds v_jp2 = FT(0.5) * (Q[ii,jj+1,kk,3] + Q[ii,jj+1,kk+1,3])
    @inbounds v_jm2 = FT(0.5) * (Q[ii,jj-1,kk,3] + Q[ii,jj-1,kk+1,3])
    vw_top = FT(0.5) * (v_avg + v_jp2) * FT(0.5) * (w_avg + w_jp)
    vw_bot = FT(0.5) * (v_avg + v_jm2) * FT(0.5) * (w_avg + w_jm)
    @inbounds A_j_top = FT(0.5) * (Areaj[ii,jj+1,kk] + Areaj[ii,jj+1,kk+1])
    @inbounds A_j_bot = FT(0.5) * (Areaj[ii,jj,kk] + Areaj[ii,jj,kk+1])
    F_cons_j = (vw_top * A_j_top - vw_bot * A_j_bot) / (vol_half + FT(1.0e-30))
    F_adv_j = v_avg * (w_jp - w_jm) / (FT(2.0) * ds_j)

    F_conv = FT(0.5) * ((F_cons_k + F_cons_i + F_cons_j) +
                       (F_adv_k  + F_adv_i  + F_adv_j))

    # ══════════════════════════════════════════════
    # Viscous term: full Laplacian ν(∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)
    # ══════════════════════════════════════════════
    mu = ρ_ref * ν_AC
    d2w_dz2 = zero(FT)
    if k > 1 && k < nzp+1
        @inbounds w_D2 = Q[ii, jj, kk-1, 4]; @inbounds w_U2 = Q[ii, jj, kk+2, 4]
        d2w_dz2 = (w_U2 - FT(2.0) * w_avg + w_D2) / (ds_k * ds_k)
    end
    d2w_dx2 = (w_ip - FT(2.0) * w_avg + w_im) / (ds_i * ds_i)
    d2w_dy2 = (w_jp - FT(2.0) * w_avg + w_jm) / (ds_j * ds_j)
    F_visc = mu * (d2w_dx2 + d2w_dy2 + d2w_dz2)

    @inbounds dUf_k[i, j, k] = -F_conv - dp_dz + F_visc
    return
end

# ═══════════════════════════════════════════════════════════════
# RK3 time integration for staggered grid
# ═══════════════════════════════════════════════════════════════

function ac_rk3_staggered_p!(Q, Q_n, dpdt, dt, rk_a, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii = i + NG; jj = j + NG; kk = k + NG
    # SSP-RK3: p = (1-α)*p^n + α*(p_current + dt*R)
    @inbounds p_n = Q_n[ii, jj, kk, 1]
    @inbounds p_cur = Q[ii, jj, kk, 1]
    @inbounds Q[ii, jj, kk, 1] = p_n + rk_a * (p_cur - p_n + dt * dpdt[i, j, k])
    return
end

function ac_rk3_staggered_uf!(Uf, Uf_n, dUf, dt, rk_a, n1, n2, n3)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > n1 || j > n2 || k > n3
        return
    end
    @inbounds Uf[i, j, k] = Uf_n[i, j, k] + rk_a * (Uf[i, j, k] - Uf_n[i, j, k] + dt * dUf[i, j, k])
    return
end

# ═══════════════════════════════════════════════════════════════
# Dual-Time Stepping (DTS) RK3 kernels
# Physical: BDF2 (2nd-order time-accurate)
# Pseudo-time: SSP-RK3 (naturally handles hyperbolic AC coupling)
# ═══════════════════════════════════════════════════════════════

# DTS-RK3 pressure update: SSP-RK3 + BDF2 source
# Q_pseudo0: pseudo-time start (for RK3 reset)
# Q_n_phys, Q_nm1_phys: physical time levels n, n-1 (for BDF2 source)
function ac_dts_rk3_p!(Q, Q_pseudo0, dpdt, dt_pseudo, rk_a,
                        Q_n_phys, Q_nm1_phys, dt_phys, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds p_0   = Q_pseudo0[ii, jj, kk, 1]   # pseudo-time start
    @inbounds p_cur = Q[ii, jj, kk, 1]            # current iterate
    @inbounds p_n   = Q_n_phys[ii, jj, kk, 1]     # physical p^n
    @inbounds p_nm1 = Q_nm1_phys[ii, jj, kk, 1]   # physical p^{n-1}

    # BDF2 source: (3p - 4p^n + p^{n-1}) / (2Δt)
    bdf2_src = (FT(1.5e0) * p_cur - FT(2.0) * p_n + FT(0.5) * p_nm1) / dt_phys

    # Modified RHS = spatial_RHS - BDF2_source
    @inbounds rhs = dpdt[i, j, k] - bdf2_src

    # SSP-RK3: p = p⁰ + α*(p_cur - p⁰ + Δτ*rhs)
    @inbounds Q[ii, jj, kk, 1] = p_0 + rk_a * (p_cur - p_0 + dt_pseudo * rhs)
    return
end

# DTS-RK3 face velocity update: SSP-RK3 + BDF2 source
function ac_dts_rk3_uf!(Uf, Uf_pseudo0, dUf, dt_pseudo, rk_a,
                          Uf_n_phys, Uf_nm1_phys, dt_phys, n1, n2, n3)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > n1 || j > n2 || k > n3
        return
    end
    @inbounds uf_0   = Uf_pseudo0[i, j, k]
    @inbounds uf_cur = Uf[i, j, k]
    @inbounds uf_n   = Uf_n_phys[i, j, k]
    @inbounds uf_nm1 = Uf_nm1_phys[i, j, k]

    bdf2_src = (FT(1.5e0) * uf_cur - FT(2.0) * uf_n + FT(0.5) * uf_nm1) / dt_phys
    @inbounds rhs = dUf[i, j, k] - bdf2_src

    @inbounds Uf[i, j, k] = uf_0 + rk_a * (uf_cur - uf_0 + dt_pseudo * rhs)
    return
end

# ═══════════════════════════════════════════════════════════════
# BDF2 implicit time integration for staggered grid
# Point-implicit Jacobi iteration (fully GPU-parallel)
# ═══════════════════════════════════════════════════════════════

# BDF2 pressure update with coupling-aware diagonal
# D_p = 3/(2dt) + w_LU × β² × Σ(A²)/V² × (2dt/3)
# The coupling term captures: p → ∇p/Δs → ΔUf → β²∇·Uf/V → Δ(dpdt)
# For O-grid center cells (small V, large A/V), D_p_coupling >> D_temp,
# naturally preventing unbounded pressure corrections.
function ac_bdf2_update_p!(Q, dpdt, Un, U_nm1, dt, nxp, nyp, nzp,
                            Vol, Areai, Areaj, Areak, w_LU_p)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii = i + NG; jj = j + NG; kk = k + NG

    @inbounds p_cur = Q[ii, jj, kk, 1]
    @inbounds pn = Un[ii, jj, kk, 1]
    @inbounds pnm1 = U_nm1[ii, jj, kk, 1]

    # BDF2 temporal source
    bdf2_src = (FT(3.0) * p_cur - FT(4.0) * pn + pnm1) / (FT(2.0) * dt)

    # Residual: spatial RHS - temporal derivative
    @inbounds res = dpdt[i, j, k] - bdf2_src

    # Temporal diagonal
    D_temp = FT(1.5e0) / dt

    # Coupling diagonal: estimated from p→Uf→div(Uf)→dpdt chain
    # dpdt = -β²/V × Σ(Uf×A), Uf responds to ∇p with ΔUf = Δp×A/(V×D_uf)
    # Effective: ∂(dpdt)/∂p ≈ β²/V × Σ(A²/V) / D_uf ≈ β² × Σ(A²) × vol_inv² / D_temp
    @inbounds vol_inv = Vol[ii, jj, kk]
    @inbounds Ai_L = Areai[ii, jj, kk];   @inbounds Ai_R = Areai[ii+1, jj, kk]
    @inbounds Aj_B = Areaj[ii, jj, kk];   @inbounds Aj_T = Areaj[ii, jj+1, kk]
    @inbounds Ak_D = Areak[ii, jj, kk];   @inbounds Ak_U = Areak[ii, jj, kk+1]

    sum_A2 = Ai_L*Ai_L + Ai_R*Ai_R + Aj_B*Aj_B + Aj_T*Aj_T + Ak_D*Ak_D + Ak_U*Ak_U
    β2 = β_AC * β_AC
    D_coupling = β2 * sum_A2 * vol_inv * vol_inv / D_temp

    D_p = D_temp + w_LU_p * D_coupling

    @inbounds Q[ii, jj, kk, 1] += res / D_p
    return
end

# BDF2 face velocity update (generic for i/j/k faces)
# D_uf = 1.5/dt + w_LU * D_spatial
# where D_spatial = |u_conv|/ds + 2ν/ds² estimated from grid metrics
function ac_bdf2_update_uf!(Uf, dUf, Uf_n_arr, Uf_nm1_arr, Q, Vol,
                             Areai, Areaj, Areak, dt, w_LU,
                             n1, n2, n3, face_dir)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > n1 || j > n2 || k > n3
        return
    end

    @inbounds uf_cur = Uf[i, j, k]
    @inbounds uf_n = Uf_n_arr[i, j, k]
    @inbounds uf_nm1 = Uf_nm1_arr[i, j, k]

    # BDF2 temporal source
    bdf2_src = (FT(3.0) * uf_cur - FT(4.0) * uf_n + uf_nm1) / (FT(2.0) * dt)

    # Residual
    @inbounds res = dUf[i, j, k] - bdf2_src

    # Estimate spatial spectral radius from grid metrics at face location
    # Map face index to cell indices for metric access
    if face_dir == Int32(1)      # I-face
        ii = i + NG - Int32(1); jj = j + NG; kk = k + NG
    elseif face_dir == Int32(2)  # J-face
        ii = i + NG; jj = j + NG - Int32(1); kk = k + NG
    else                         # K-face
        ii = i + NG; jj = j + NG; kk = k + NG - Int32(1)
    end

    # Cell volumes on either side of face
    @inbounds vol_inv_L = Vol[ii, jj, kk]
    @inbounds vol_inv_R = Vol[ii + (face_dir==Int32(1) ? Int32(1) : Int32(0)),
                               jj + (face_dir==Int32(2) ? Int32(1) : Int32(0)),
                               kk + (face_dir==Int32(3) ? Int32(1) : Int32(0))]
    vol_L = one(FT) / (vol_inv_L + FT(1.0e-30))
    vol_R = one(FT) / (vol_inv_R + FT(1.0e-30))
    vol_half = FT(0.5) * (vol_L + vol_R)

    # Average face areas for grid spacing estimation
    @inbounds Ai = FT(0.5) * (Areai[ii, jj, kk] + Areai[ii+1, jj, kk])
    @inbounds Aj = FT(0.5) * (Areaj[ii, jj, kk] + Areaj[ii, jj+1, kk])
    @inbounds Ak = FT(0.5) * (Areak[ii, jj, kk] + Areak[ii, jj, kk+1])

    # Grid spacings (ds = V / A)
    ds_i = vol_half / (Ai + FT(1.0e-14))
    ds_j = vol_half / (Aj + FT(1.0e-14))
    ds_k = vol_half / (Ak + FT(1.0e-14))

    # Cell-center velocities at face (interpolated from Q)
    @inbounds u_L = Q[ii, jj, kk, 2]
    @inbounds u_R = Q[ii + (face_dir==Int32(1) ? Int32(1) : Int32(0)),
                       jj + (face_dir==Int32(2) ? Int32(1) : Int32(0)),
                       kk + (face_dir==Int32(3) ? Int32(1) : Int32(0)), 2]
    @inbounds v_L = Q[ii, jj, kk, 3]
    @inbounds v_R = Q[ii + (face_dir==Int32(1) ? Int32(1) : Int32(0)),
                       jj + (face_dir==Int32(2) ? Int32(1) : Int32(0)),
                       kk + (face_dir==Int32(3) ? Int32(1) : Int32(0)), 3]
    @inbounds w_L = Q[ii, jj, kk, 4]
    @inbounds w_R = Q[ii + (face_dir==Int32(1) ? Int32(1) : Int32(0)),
                       jj + (face_dir==Int32(2) ? Int32(1) : Int32(0)),
                       kk + (face_dir==Int32(3) ? Int32(1) : Int32(0)), 4]
    u_avg = FT(0.5) * (u_L + u_R)
    v_avg = FT(0.5) * (v_L + v_R)
    w_avg = FT(0.5) * (w_L + w_R)

    # Convective spectral radius: Σ |u_dir| / ds_dir
    D_conv = abs(u_avg) / ds_i + abs(v_avg) / ds_j + abs(w_avg) / ds_k

    # Acoustic spectral radius: β is the AC pseudo-sound speed
    # Without this, the Jacobi amplification factor |1-iβ/(Δx·D_total)| > 1
    D_acoustic = β_AC * (one(FT)/ds_i + one(FT)/ds_j + one(FT)/ds_k)

    # Viscous spectral radius: ν × Σ 2/ds²
    D_visc = ν_AC * FT(2.0) * (one(FT)/(ds_i*ds_i) + one(FT)/(ds_j*ds_j) + one(FT)/(ds_k*ds_k))

    D_spatial = D_conv + D_acoustic + D_visc

    # Point-implicit diagonal
    D_total = FT(1.5e0) / dt + w_LU * D_spatial

    # Update
    @inbounds Uf[i, j, k] += res / D_total
    return
end

# ═══════════════════════════════════════════════════════════════
# Face velocity boundary conditions
# ═══════════════════════════════════════════════════════════════

function ac_wall_face_vel_i!(Uf_i, nxp, nyp, nzp, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if j > nyp || k > nzp
        return
    end
    if side == Int32(0)
        if i != 1; return; end
    else
        if i != nxp+1; return; end
    end
    @inbounds Uf_i[i, j, k] = zero(FT)
    return
end

function ac_lid_face_vel_i!(Uf_i, nxp, nyp, nzp, u_lid, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if j > nyp || k > nzp
        return
    end
    if side == Int32(0)
        if i != 1; return; end
    else
        if i != nxp+1; return; end
    end
    @inbounds Uf_i[i, j, k] = u_lid
    return
end

function ac_wall_face_vel_j!(Uf_j, nxp, nyp, nzp, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || k > nzp
        return
    end
    if side == Int32(0)
        if j != 1; return; end
    else
        if j != nyp+1; return; end
    end
    @inbounds Uf_j[i, j, k] = zero(FT)
    return
end

function ac_lid_face_vel_j!(Uf_j, nxp, nyp, nzp, v_lid, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || k > nzp
        return
    end
    if side == Int32(0)
        if j != 1; return; end
    else
        if j != nyp+1; return; end
    end
    @inbounds Uf_j[i, j, k] = v_lid
    return
end

function ac_wall_face_vel_k!(Uf_k, nxp, nyp, nzp, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp
        return
    end
    if side == Int32(0)
        if k != 1; return; end
    else
        if k != nzp+1; return; end
    end
    @inbounds Uf_k[i, j, k] = zero(FT)
    return
end

# ═══════════════════════════════════════════════════════════════
# Pressure ghost cell Neumann BC (dp/dn = 0)
# Fills ALL NG ghost layers (not just 1).
# Thread indices j,k cover 1..Ny_tot, 1..Nz_tot (full array extent).
# ═══════════════════════════════════════════════════════════════

function ac_pressure_neumann_i!(Q, nxp, nyp, nzp, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > 1 || j > nyp+2*NG || k > nzp+2*NG
        return
    end
    if side == Int32(0)
        # Fill ghost layers 1..NG (left side)
        for g = Int32(1):Int32(NG)
            @inbounds Q[NG+1-g, j, k, 1] = Q[NG+1, j, k, 1]
        end
    else
        # Fill ghost layers nxp+NG+1..nxp+2*NG (right side)
        for g = Int32(1):Int32(NG)
            @inbounds Q[nxp+NG+g, j, k, 1] = Q[nxp+NG, j, k, 1]
        end
    end
    return
end

function ac_pressure_neumann_j!(Q, nxp, nyp, nzp, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > 1 || k > nzp+2*NG
        return
    end
    if side == Int32(0)
        for g = Int32(1):Int32(NG)
            @inbounds Q[i, NG+1-g, k, 1] = Q[i, NG+1, k, 1]
        end
    else
        for g = Int32(1):Int32(NG)
            @inbounds Q[i, nyp+NG+g, k, 1] = Q[i, nyp+NG, k, 1]
        end
    end
    return
end

function ac_pressure_neumann_k!(Q, nxp, nyp, nzp, side)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > 1
        return
    end
    if side == Int32(0)
        for g = Int32(1):Int32(NG)
            @inbounds Q[i, j, NG+1-g, 1] = Q[i, j, NG+1, 1]
        end
    else
        for g = Int32(1):Int32(NG)
            @inbounds Q[i, j, nzp+NG+g, 1] = Q[i, j, nzp+NG, 1]
        end
    end
    return
end

# ═══════════════════════════════════════════════════════════════
# Velocity clipping and NaN guard
# ═══════════════════════════════════════════════════════════════

function ac_clip_face_vel!(Uf, Uf_n, n1, n2, n3, u_max)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > n1 || j > n2 || k > n3
        return
    end
    @inbounds v = Uf[i, j, k]
    if isnan(v) || isinf(v)
        @inbounds Uf[i, j, k] = Uf_n[i, j, k]
    elseif abs(v) > u_max
        @inbounds Uf[i, j, k] = sign(v) * u_max
    end
    return
end

function ac_clip_pressure!(Q, nxp, nyp, nzp, p_max)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds p = Q[ii, jj, kk, 1]
    if isnan(p) || isinf(p)
        @inbounds Q[ii, jj, kk, 1] = zero(FT)
    elseif abs(p) > p_max
        @inbounds Q[ii, jj, kk, 1] = sign(p) * p_max
    end
    return
end

# ═══════════════════════════════════════════════════════════════
# Host-side helper: apply face velocity BCs based on face_bc table
# ═══════════════════════════════════════════════════════════════

function ac_apply_face_vel_bc!(b, bid, face_bc)
    nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
    nthreads_bc = (16, 8, 8)

    for (fid, bc) in face_bc
        if fid[1] != bid; continue; end

        if bc == BC_AC_WALL || bc == BC_ISOTHERMAL_WALL || bc == BC_ADIABATIC_WALL
            if fid[2] == 1
                nb_bc = (1, cld(nyp, nthreads_bc[2]), cld(nzp, nthreads_bc[3]))
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_wall_face_vel_i!(b.Uf_i, nxp, nyp, nzp, Int32(0))
            elseif fid[2] == 2
                nb_bc = (1, cld(nyp, nthreads_bc[2]), cld(nzp, nthreads_bc[3]))
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_wall_face_vel_i!(b.Uf_i, nxp, nyp, nzp, Int32(1))
            elseif fid[2] == 3
                nb_bc = (cld(nxp, nthreads_bc[1]), 1, cld(nzp, nthreads_bc[3]))
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_wall_face_vel_j!(b.Uf_j, nxp, nyp, nzp, Int32(0))
            elseif fid[2] == 4
                nb_bc = (cld(nxp, nthreads_bc[1]), 1, cld(nzp, nthreads_bc[3]))
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_wall_face_vel_j!(b.Uf_j, nxp, nyp, nzp, Int32(1))
            elseif fid[2] == 5
                nb_bc = (cld(nxp, nthreads_bc[1]), cld(nyp, nthreads_bc[2]), 1)
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_wall_face_vel_k!(b.Uf_k, nxp, nyp, nzp, Int32(0))
            elseif fid[2] == 6
                nb_bc = (cld(nxp, nthreads_bc[1]), cld(nyp, nthreads_bc[2]), 1)
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_wall_face_vel_k!(b.Uf_k, nxp, nyp, nzp, Int32(1))
            end

        elseif bc == BC_AC_LID
            u_lid = FT(U_lid_AC)
            if fid[2] == 1
                nb_bc = (1, cld(nyp, nthreads_bc[2]), cld(nzp, nthreads_bc[3]))
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_lid_face_vel_i!(b.Uf_i, nxp, nyp, nzp, u_lid, Int32(0))
            elseif fid[2] == 2
                nb_bc = (1, cld(nyp, nthreads_bc[2]), cld(nzp, nthreads_bc[3]))
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_lid_face_vel_i!(b.Uf_i, nxp, nyp, nzp, u_lid, Int32(1))
            elseif fid[2] == 3
                nb_bc = (cld(nxp, nthreads_bc[1]), 1, cld(nzp, nthreads_bc[3]))
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_lid_face_vel_j!(b.Uf_j, nxp, nyp, nzp, u_lid, Int32(0))
            elseif fid[2] == 4
                nb_bc = (cld(nxp, nthreads_bc[1]), 1, cld(nzp, nthreads_bc[3]))
                @gpu_launch threads=nthreads_bc blocks=nb_bc ac_lid_face_vel_j!(b.Uf_j, nxp, nyp, nzp, u_lid, Int32(1))
            end
        end
    end
end

function ac_apply_pressure_bc!(b, bid, face_bc)
    nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
    nthreads_bc = (16, 8, 8)
    Ny_tot = nyp + 2*NG; Nz_tot = nzp + 2*NG; Nx_tot = nxp + 2*NG

    for (fid, bc) in face_bc
        if fid[1] != bid; continue; end
        is_wall = (bc == BC_AC_WALL || bc == BC_ISOTHERMAL_WALL || bc == BC_ADIABATIC_WALL || bc == BC_AC_LID)
        if !is_wall; continue; end

        if fid[2] == 1
            nb_bc = (1, cld(Ny_tot, nthreads_bc[2]), cld(Nz_tot, nthreads_bc[3]))
            @gpu_launch threads=nthreads_bc blocks=nb_bc ac_pressure_neumann_i!(b.Q, nxp, nyp, nzp, Int32(0))
        elseif fid[2] == 2
            nb_bc = (1, cld(Ny_tot, nthreads_bc[2]), cld(Nz_tot, nthreads_bc[3]))
            @gpu_launch threads=nthreads_bc blocks=nb_bc ac_pressure_neumann_i!(b.Q, nxp, nyp, nzp, Int32(1))
        elseif fid[2] == 3
            nb_bc = (cld(Nx_tot, nthreads_bc[1]), 1, cld(Nz_tot, nthreads_bc[3]))
            @gpu_launch threads=nthreads_bc blocks=nb_bc ac_pressure_neumann_j!(b.Q, nxp, nyp, nzp, Int32(0))
        elseif fid[2] == 4
            nb_bc = (cld(Nx_tot, nthreads_bc[1]), 1, cld(Nz_tot, nthreads_bc[3]))
            @gpu_launch threads=nthreads_bc blocks=nb_bc ac_pressure_neumann_j!(b.Q, nxp, nyp, nzp, Int32(1))
        elseif fid[2] == 5
            nb_bc = (cld(Nx_tot, nthreads_bc[1]), cld(Ny_tot, nthreads_bc[2]), 1)
            @gpu_launch threads=nthreads_bc blocks=nb_bc ac_pressure_neumann_k!(b.Q, nxp, nyp, nzp, Int32(0))
        elseif fid[2] == 6
            nb_bc = (cld(Nx_tot, nthreads_bc[1]), cld(Ny_tot, nthreads_bc[2]), 1)
            @gpu_launch threads=nthreads_bc blocks=nb_bc ac_pressure_neumann_k!(b.Q, nxp, nyp, nzp, Int32(1))
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════
#  AC Collocated Skew-Symmetric Convection Kernels
#  
#  Replaces WENO+Rusanov for incompressible AC on collocated grids.
#  Uses 2nd-order central skew-symmetric form (Morinishi et al. 1998):
#
#    F_conv = 0.5 × [∇·(u⊗u) + u·∇u]   (energy-conserving)
#
#  AC pressure equation:  dp/dt = -β²∇·u
#  AC momentum equation:  du/dt = -∇·(u⊗u) - ∇p/ρ + ν∇²u
#
#  Face flux convention:
#    Fx[i,j,k,n] = face flux at i-½ face × Area
#    div kernel computes: (Fx[i] - Fx[i+1]) = flux_left - flux_right
#
#  The skew-symmetric form is implemented as a face flux:
#    F_skew = 0.5*(F_conservative + F_advective_as_flux)
#  where:
#    F_cons = q_n × var_face                    (conservative form)
#    F_adv  = 0.5 × u_face × (var_R - var_L)   (advective correction as flux diff)
#  
#  But since div expects face fluxes (not volume terms), we express the
#  advective form contribution as a modified face flux:
#    F_skew[i+½] = q_avg × var_avg × Area   (symmetric central flux)
#  This is mathematically equivalent to the skew-symmetric discretization
#  on uniform grids and extends naturally to curvilinear grids.
# ═══════════════════════════════════════════════════════════════════════

# ─── ξ-direction face flux ───
# Computes flux at i+½ face for cells [NG+1, ..., Nx+NG+1]
# Q = [p, u, v, w] for AC
function ac_skew_sym_flux_i!(Fx, Q, Areai, nxi, nyi, nzi, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    # Face i ranges from 1 to nxp+1 in local coords (i.e., NG+1 to Nx+NG+1 in global)
    if i > nxp + Int32(1) || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    # Global indices: face i sits between cells ig-1 and ig
    ig = i + NG      # right cell
    jg = j + NG
    kg = k + NG

    @inbounds begin
        # Left and right cell primitive variables
        pL = Q[ig-1, jg, kg, 1]; pR = Q[ig, jg, kg, 1]
        uL = Q[ig-1, jg, kg, 2]; uR = Q[ig, jg, kg, 2]
        vL = Q[ig-1, jg, kg, 3]; vR = Q[ig, jg, kg, 3]
        wL = Q[ig-1, jg, kg, 4]; wR = Q[ig, jg, kg, 4]

        # Face area and normal
        area = Areai[ig, jg, kg]
        fnx = nxi[ig, jg, kg]
        fny = nyi[ig, jg, kg]
        fnz = nzi[ig, jg, kg]

        # Arithmetic averages at face
        p_avg = FT(0.5) * (pL + pR)
        u_avg = FT(0.5) * (uL + uR)
        v_avg = FT(0.5) * (vL + vR)
        w_avg = FT(0.5) * (wL + wR)

        # Contravariant velocity at face (normal velocity × area)
        q_n = u_avg * fnx + v_avg * fny + w_avg * fnz  # [m/s] (unit normal)

        # ─── Pressure equation: dp/dt = -β²∇·u ───
        # Face flux for continuity: F_p = β² × q_n × Area
        # (positive q_n means mass flux leaving left cell)
        β2 = β_AC * β_AC
        Fx[i, j, k, 1] = β2 * q_n * area

        # ─── Momentum equations: skew-symmetric convection + pressure gradient ───
        # Skew-symmetric face flux = q_avg × var_avg × Area  (Morinishi 1998)
        # This is the central KEP form which preserves kinetic energy
        # Plus pressure gradient: (p/ρ) × n × Area
        inv_ρ = one(FT) / ρ_ref
        q_flux = q_n * area  # volume flux through face [m³/s]

        Fx[i, j, k, 2] = q_flux * u_avg + p_avg * inv_ρ * fnx * area
        Fx[i, j, k, 3] = q_flux * v_avg + p_avg * inv_ρ * fny * area
        Fx[i, j, k, 4] = q_flux * w_avg + p_avg * inv_ρ * fnz * area
    end
    return
end

# ─── η-direction face flux ───
function ac_skew_sym_flux_j!(Fy, Q, Areaj, nxj, nyj, nzj, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp || j > nyp + Int32(1) || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    ig = i + NG
    jg = j + NG
    kg = k + NG

    @inbounds begin
        pL = Q[ig, jg-1, kg, 1]; pR = Q[ig, jg, kg, 1]
        uL = Q[ig, jg-1, kg, 2]; uR = Q[ig, jg, kg, 2]
        vL = Q[ig, jg-1, kg, 3]; vR = Q[ig, jg, kg, 3]
        wL = Q[ig, jg-1, kg, 4]; wR = Q[ig, jg, kg, 4]

        area = Areaj[ig, jg, kg]
        fnx = nxj[ig, jg, kg]
        fny = nyj[ig, jg, kg]
        fnz = nzj[ig, jg, kg]

        p_avg = FT(0.5) * (pL + pR)
        u_avg = FT(0.5) * (uL + uR)
        v_avg = FT(0.5) * (vL + vR)
        w_avg = FT(0.5) * (wL + wR)

        q_n = u_avg * fnx + v_avg * fny + w_avg * fnz
        β2 = β_AC * β_AC
        Fy[i, j, k, 1] = β2 * q_n * area

        inv_ρ = one(FT) / ρ_ref
        q_flux = q_n * area

        Fy[i, j, k, 2] = q_flux * u_avg + p_avg * inv_ρ * fnx * area
        Fy[i, j, k, 3] = q_flux * v_avg + p_avg * inv_ρ * fny * area
        Fy[i, j, k, 4] = q_flux * w_avg + p_avg * inv_ρ * fnz * area
    end
    return
end

# ─── ζ-direction face flux ───
function ac_skew_sym_flux_k!(Fz, Q, Areak, nxk, nyk, nzk, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp || j > nyp || k > nzp + Int32(1) || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    ig = i + NG
    jg = j + NG
    kg = k + NG

    @inbounds begin
        pL = Q[ig, jg, kg-1, 1]; pR = Q[ig, jg, kg, 1]
        uL = Q[ig, jg, kg-1, 2]; uR = Q[ig, jg, kg, 2]
        vL = Q[ig, jg, kg-1, 3]; vR = Q[ig, jg, kg, 3]
        wL = Q[ig, jg, kg-1, 4]; wR = Q[ig, jg, kg, 4]

        area = Areak[ig, jg, kg]
        fnx = nxk[ig, jg, kg]
        fny = nyk[ig, jg, kg]
        fnz = nzk[ig, jg, kg]

        p_avg = FT(0.5) * (pL + pR)
        u_avg = FT(0.5) * (uL + uR)
        v_avg = FT(0.5) * (vL + vR)
        w_avg = FT(0.5) * (wL + wR)

        q_n = u_avg * fnx + v_avg * fny + w_avg * fnz
        β2 = β_AC * β_AC
        Fz[i, j, k, 1] = β2 * q_n * area

        inv_ρ = one(FT) / ρ_ref
        q_flux = q_n * area

        Fz[i, j, k, 2] = q_flux * u_avg + p_avg * inv_ρ * fnx * area
        Fz[i, j, k, 3] = q_flux * v_avg + p_avg * inv_ρ * fny * area
        Fz[i, j, k, 4] = q_flux * w_avg + p_avg * inv_ρ * fnz * area
    end
    return
end
