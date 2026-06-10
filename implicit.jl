# =============================================================================
# LU-SGS Implicit Time Advancement for Flame3D
# Reference: Yoon & Jameson (1988), "Lower-Upper Symmetric-Gauss-Seidel
#            method for the Euler and Navier-Stokes equations"
#
# Strategy: Hyperplane-parallel LU-SGS with spectral radius approximation
#   - No explicit Jacobian assembly (memory-efficient for GPU)
#   - Diagonal planes m = i+j+k allow full GPU parallelism within each plane
#   - Spectral radius approximation for off-diagonal implicit operator
# =============================================================================

# ─── Spectral radius at cell faces ───
# σ_face = (|V_n| + c) * Area  at each face center
function compute_spectral_radius_i!(σ_i, Q, Areai, nxi, nyi, nzi, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    # Face indices: i ∈ [1, nxp+1], j ∈ [1, nyp], k ∈ [1, nzp]
    if i > nxp + Int32(1) || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    # Map to global indices (with ghost)
    ig = i + NG - Int32(1)  # face i sits between cell ig and ig+1
    jg = j + NG
    kg = k + NG

    @inbounds begin
        # Average primitive from left and right cells
        uL = Q[ig, jg, kg, 2]; uR = Q[ig+1, jg, kg, 2]
        vL = Q[ig, jg, kg, 3]; vR = Q[ig+1, jg, kg, 3]
        wL = Q[ig, jg, kg, 4]; wR = Q[ig+1, jg, kg, 4]

        u_f = FT(0.5) * (uL + uR)
        v_f = FT(0.5) * (vL + vR)
        w_f = FT(0.5) * (wL + wR)

        # Face normal and area
        fnx = nxi[ig+1, jg, kg]
        fny = nyi[ig+1, jg, kg]
        fnz = nzi[ig+1, jg, kg]
        area = Areai[ig+1, jg, kg]

        # Contravariant velocity
        V_n = u_f * fnx + v_f * fny + w_f * fnz

        # Sound speed: √(γRgT) for compressible
        TL = Q[ig, jg, kg, 6]; TR = Q[ig+1, jg, kg, 6]
        T_f = FT(0.5) * (TL + TR)
        c = sqrt(γ * Rg * max(T_f, FT(1.0e-10)))

        σ_i[i, j, k] = (abs(V_n) + c) * area
    end
    return
end

function compute_spectral_radius_j!(σ_j, Q, Areaj, nxj, nyj, nzj, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp || j > nyp + Int32(1) || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    ig = i + NG
    jg = j + NG - Int32(1)
    kg = k + NG

    @inbounds begin
        u_f = FT(0.5) * (Q[ig, jg, kg, 2] + Q[ig, jg+1, kg, 2])
        v_f = FT(0.5) * (Q[ig, jg, kg, 3] + Q[ig, jg+1, kg, 3])
        w_f = FT(0.5) * (Q[ig, jg, kg, 4] + Q[ig, jg+1, kg, 4])

        fnx = nxj[ig, jg+1, kg]
        fny = nyj[ig, jg+1, kg]
        fnz = nzj[ig, jg+1, kg]
        area = Areaj[ig, jg+1, kg]

        V_n = u_f * fnx + v_f * fny + w_f * fnz
        T_f = FT(0.5) * (Q[ig, jg, kg, 6] + Q[ig, jg+1, kg, 6])
        c = sqrt(γ * Rg * max(T_f, FT(1.0e-10)))

        σ_j[i, j, k] = (abs(V_n) + c) * area
    end
    return
end

function compute_spectral_radius_k!(σ_k, Q, Areak, nxk, nyk, nzk, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp || j > nyp || k > nzp + Int32(1) || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    ig = i + NG
    jg = j + NG
    kg = k + NG - Int32(1)

    @inbounds begin
        u_f = FT(0.5) * (Q[ig, jg, kg, 2] + Q[ig, jg, kg+1, 2])
        v_f = FT(0.5) * (Q[ig, jg, kg, 3] + Q[ig, jg, kg+1, 3])
        w_f = FT(0.5) * (Q[ig, jg, kg, 4] + Q[ig, jg, kg+1, 4])

        fnx = nxk[ig, jg, kg+1]
        fny = nyk[ig, jg, kg+1]
        fnz = nzk[ig, jg, kg+1]
        area = Areak[ig, jg, kg+1]

        V_n = u_f * fnx + v_f * fny + w_f * fnz
        T_f = FT(0.5) * (Q[ig, jg, kg, 6] + Q[ig, jg, kg+1, 6])
        c = sqrt(γ * Rg * max(T_f, FT(1.0e-10)))

        σ_k[i, j, k] = (abs(V_n) + c) * area
    end
    return
end


# ─── Diagonal block D and its inverse ───
# D[i,j,k] = Vol/dt + 0.5 * Σ(σ at all 6 faces)
# Scalar approximation: D is a scalar multiplying I (identity), not a full 5×5 block
function compute_lusgs_diagonal!(D_inv, Q, Vol, σ_i, σ_j, σ_k,
                                  Areai, Areaj, Areak, dt, w_LU, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    ig = i + NG
    jg = j + NG
    kg = k + NG

    @inbounds begin
        vol_inv = Vol[ig, jg, kg]  # Vol stores 1/Volume (Jacobian)
        cell_vol = one(FT) / (vol_inv + FT(1.0e-30))  # Actual cell volume [m³]

        # Convective spectral radii sum [m³/s]
        σ_sum = σ_i[i, j, k] + σ_i[i+1, j, k] +
                σ_j[i, j, k] + σ_j[i, j+1, k] +
                σ_k[i, j, k] + σ_k[i, j, k+1]

        # Viscous spectral radius (Blazek Eq. 6.22):
        #   σ_visc = (μ/ρ) × A²/V for each face direction
        # This stabilizes the LU-SGS for viscous-dominated wall cells
        if viscous
            # Average face areas for each direction
            Ai = FT(0.5) * (Areai[ig, jg, kg] + Areai[ig+1, jg, kg])
            Aj = FT(0.5) * (Areaj[ig, jg, kg] + Areaj[ig, jg+1, kg])
            Ak = FT(0.5) * (Areak[ig, jg, kg] + Areak[ig, jg, kg+1])

            ρ_c = max(Q[ig, jg, kg, 1], FT(1.0e-10))
            T_c = max(Q[ig, jg, kg, 6], FT(1.0e-10))
            μ_c = get_viscosity(T_c)
            nu_eff = max(FT(4.0)/FT(3.0) * μ_c, γ * μ_c / Pr) / ρ_c

            σ_visc = nu_eff * (Ai*Ai + Aj*Aj + Ak*Ak) / (cell_vol + FT(1.0e-30))
            σ_sum += FT(2.0) * σ_visc  # factor 2 for both sides of the cell
        end

        @static if equation_type == :MHD
            if resistive
                Ai_res = FT(0.5) * (Areai[ig, jg, kg] + Areai[ig+1, jg, kg])
                Aj_res = FT(0.5) * (Areaj[ig, jg, kg] + Areaj[ig, jg+1, kg])
                Ak_res = FT(0.5) * (Areak[ig, jg, kg] + Areak[ig, jg, kg+1])
                σ_res = η_mhd * (Ai_res*Ai_res + Aj_res*Aj_res + Ak_res*Ak_res) / (cell_vol + FT(1.0e-30))
                σ_sum += FT(2.0) * σ_res
            end
        end

        # w_LU: LU-SGS relaxation factor (EC uses 1~2, default 1.5)
        # Larger w_LU → stronger diagonal dominance → more stable but slower convergence
        # Standard 0.5 is insufficient for WENO's imaginary eigenvalues
        D = cell_vol / dt + w_LU * σ_sum  # [m³/s] — dimensionally consistent
        D_inv[ig, jg, kg] = one(FT) / (D + FT(1.0e-30))
    end
    return
end


# ─── LU-SGS Forward Sweep (single hyperplane) ───
# For plane m = i+j+k, update ΔU using already-swept lower neighbors
# Lower neighbors: (i-1,j,k), (i,j-1,k), (i,j,k-1) have smaller i+j+k
function lusgs_forward_sweep_plane!(ΔU, dU_rhs, D_inv, σ_i, σ_j, σ_k,
                                    nxp, nyp, nzp, plane_m)
    # Thread index maps to position within the diagonal plane
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # Enumerate all (i,j,k) with i+j+k == plane_m, 1 <= i <= nxp, etc.
    # Use a 2D parameterization: fix k, then j = plane_m - i - k
    # For efficiency, we linearize the plane cells and assign by thread ID

    # Count valid cells on this plane and map tid -> (i,j,k)
    count = Int32(0)
    local ci::Int32, cj::Int32, ck::Int32
    ci = Int32(0); cj = Int32(0); ck = Int32(0)

    # k ranges: max(1, plane_m - nxp - nyp) to min(nzp, plane_m - 2)
    k_lo = max(Int32(1), plane_m - nxp - nyp)
    k_hi = min(nzp, plane_m - Int32(2))

    for kk = k_lo:k_hi
        # j ranges: max(1, plane_m - nxp - kk) to min(nyp, plane_m - 1 - kk)
        j_lo = max(Int32(1), plane_m - nxp - kk)
        j_hi = min(nyp, plane_m - Int32(1) - kk)
        for jj = j_lo:j_hi
            ii = plane_m - jj - kk
            if ii >= Int32(1) && ii <= nxp
                count += Int32(1)
                if count == tid
                    ci = ii; cj = jj; ck = kk
                end
            end
        end
    end

    # Early exit if this thread has no work
    if ci == Int32(0)
        return
    end

    i = ci; j = cj; k = ck
    ig = i + NG; jg = j + NG; kg = k + NG

    @inbounds begin
        # Start with explicit RHS
        for n = Int32(1):Int32(Ncons)
            ΔU[ig, jg, kg, n] = dU_rhs[i, j, k, n]
        end

        # Add contributions from lower neighbors (already swept)
        # Neighbor (i-1, j, k): uses σ_i[i, j, k] (the face between cell i-1 and i)
        if i > Int32(1)
            σ_f = σ_i[i, j, k]
            for n = Int32(1):Int32(Ncons)
                ΔU[ig, jg, kg, n] += FT(0.5) * σ_f * ΔU[ig-1, jg, kg, n]
            end
        end

        # Neighbor (i, j-1, k): uses σ_j[i, j, k]
        if j > Int32(1)
            σ_f = σ_j[i, j, k]
            for n = Int32(1):Int32(Ncons)
                ΔU[ig, jg, kg, n] += FT(0.5) * σ_f * ΔU[ig, jg-1, kg, n]
            end
        end

        # Neighbor (i, j, k-1): uses σ_k[i, j, k]
        if k > Int32(1)
            σ_f = σ_k[i, j, k]
            for n = Int32(1):Int32(Ncons)
                ΔU[ig, jg, kg, n] += FT(0.5) * σ_f * ΔU[ig, jg, kg-1, n]
            end
        end

        # Multiply by D_inv
        d_inv = D_inv[ig, jg, kg]
        for n = Int32(1):Int32(Ncons)
            ΔU[ig, jg, kg, n] *= d_inv
        end
    end
    return
end


# ─── LU-SGS Backward Sweep (single hyperplane) ───
# For plane m = i+j+k, correct ΔU using upper neighbors (already swept in backward direction)
# Upper neighbors: (i+1,j,k), (i,j+1,k), (i,j,k+1) have larger i+j+k
function lusgs_backward_sweep_plane!(ΔU, D_inv, σ_i, σ_j, σ_k,
                                     nxp, nyp, nzp, plane_m)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    count = Int32(0)
    local ci::Int32, cj::Int32, ck::Int32
    ci = Int32(0); cj = Int32(0); ck = Int32(0)

    k_lo = max(Int32(1), plane_m - nxp - nyp)
    k_hi = min(nzp, plane_m - Int32(2))

    for kk = k_lo:k_hi
        j_lo = max(Int32(1), plane_m - nxp - kk)
        j_hi = min(nyp, plane_m - Int32(1) - kk)
        for jj = j_lo:j_hi
            ii = plane_m - jj - kk
            if ii >= Int32(1) && ii <= nxp
                count += Int32(1)
                if count == tid
                    ci = ii; cj = jj; ck = kk
                end
            end
        end
    end

    if ci == Int32(0)
        return
    end

    i = ci; j = cj; k = ck
    ig = i + NG; jg = j + NG; kg = k + NG

    @inbounds begin
        # Correction from upper neighbors
        # We need to subtract upper contributions and re-apply D_inv
        # ΔU_new = ΔU_old - D_inv * 0.5 * Σ_upper(σ * ΔU_upper_new)
        d_inv = D_inv[ig, jg, kg]

        for n = Int32(1):Int32(Ncons)
            corr = zero(FT)

            # Neighbor (i+1, j, k): uses σ_i[i+1, j, k]
            if i < nxp
                corr += σ_i[i+1, j, k] * ΔU[ig+1, jg, kg, n]
            end

            # Neighbor (i, j+1, k): uses σ_j[i, j+1, k]
            if j < nyp
                corr += σ_j[i, j+1, k] * ΔU[ig, jg+1, kg, n]
            end

            # Neighbor (i, j, k+1): uses σ_k[i, j, k+1]
            if k < nzp
                corr += σ_k[i, j, k+1] * ΔU[ig, jg, kg+1, n]
            end

            ΔU[ig, jg, kg, n] -= FT(0.5) * d_inv * corr
        end
    end
    return
end


# ─── Jacobi (diagonal-only) update: ΔU = D_inv × RHS (diagnostic) ───
function jacobi_update!(ΔU, dU_rhs, D_inv, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    @inbounds d_inv = D_inv[ig, jg, kg]
    for n = Int32(1):Int32(Ncons)
        @inbounds ΔU[ig, jg, kg, n] = d_inv * dU_rhs[i, j, k, n]
    end
    return
end

# ─── Implicit Residual Smoothing (EC-style, explicit 3D Jacobi) ───
# Smooths dU_rhs to damp high-frequency modes before LU-SGS sweep.
# Reads from dU_temp (copy of original dU_rhs), writes smoothed result to dU_rhs.
# Uses 6-point stencil: R_smooth = (1-ε×N_neighbors)×R_center + ε×Σ(R_neighbor)
# where ε is the smoothing coefficient (typically 0.1~0.2)
function residual_smooth_kernel!(dU_rhs, dU_temp, eps_s, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    for n = Int32(1):Int32(Ncons)
        @inbounds center = dU_temp[i, j, k, n]
        neighbor_sum = zero(FT)
        n_neighbors = Int32(0)

        # i-direction neighbors
        if i > Int32(1)
            @inbounds neighbor_sum += dU_temp[i-Int32(1), j, k, n]
            n_neighbors += Int32(1)
        end
        if i < nxp
            @inbounds neighbor_sum += dU_temp[i+Int32(1), j, k, n]
            n_neighbors += Int32(1)
        end
        # j-direction neighbors
        if j > Int32(1)
            @inbounds neighbor_sum += dU_temp[i, j-Int32(1), k, n]
            n_neighbors += Int32(1)
        end
        if j < nyp
            @inbounds neighbor_sum += dU_temp[i, j+Int32(1), k, n]
            n_neighbors += Int32(1)
        end
        # k-direction neighbors
        if k > Int32(1)
            @inbounds neighbor_sum += dU_temp[i, j, k-Int32(1), n]
            n_neighbors += Int32(1)
        end
        if k < nzp
            @inbounds neighbor_sum += dU_temp[i, j, k+Int32(1), n]
            n_neighbors += Int32(1)
        end

        @inbounds dU_rhs[i, j, k, n] = (one(FT) - eps_s * FT(n_neighbors)) * center +
                                        eps_s * neighbor_sum
    end
    return
end

# ─── Implicit update: U^{n+1} = U^n + ΔU, then clip + c2Prim ───
function implicit_update!(U, Q, ΔU, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp + NG || j > nyp + NG || k > nzp + NG || i < NG + Int32(1) || j < NG + Int32(1) || k < NG + Int32(1)
        return
    end

    @inbounds begin
        # Update conserved variables
        for n = Int32(1):Int32(Ncons)
            U[i, j, k, n] += ΔU[i, j, k, n]
        end

        @static if equation_type == :MHD
            ρ = max(U[i, j, k, 1], eps(FT))
            ρinv = one(FT) / ρ
            u = U[i, j, k, 2] * ρinv
            v = U[i, j, k, 3] * ρinv
            w = U[i, j, k, 4] * ρinv
            Bx = U[i, j, k, 6]
            By = U[i, j, k, 7]
            Bz = U[i, j, k, 8]
            B2 = Bx*Bx + By*By + Bz*Bz
            ei = max(U[i, j, k, 5] - FT(0.5) * ρ * (u * u + v * v + w * w) - FT(0.5) * B2, eps(FT))
            p = (γ - one(FT)) * ei

            # Positivity clipping
            ρ_min = FT(1.0e-5)
            p_min = FT(1.0e-5)
            ρ = max(ρ, ρ_min)
            p = max(p, p_min)
            T = p / (ρ * Rg)

            Q[i, j, k, 1] = ρ
            Q[i, j, k, 2] = u
            Q[i, j, k, 3] = v
            Q[i, j, k, 4] = w
            Q[i, j, k, 5] = p
            Q[i, j, k, 6] = T
            Q[i, j, k, 7] = Bx
            Q[i, j, k, 8] = By
            Q[i, j, k, 9] = Bz
            Q[i, j, k, 10] = U[i, j, k, 9]

            # Write back clamped U
            U[i, j, k, 1] = ρ
            U[i, j, k, 2] = ρ * u
            U[i, j, k, 3] = ρ * v
            U[i, j, k, 4] = ρ * w
            U[i, j, k, 5] = p / (γ - one(FT)) + FT(0.5) * ρ * (u * u + v * v + w * w) + FT(0.5) * B2
        else
            # In-place c2Prim + positivity clipping (same as linComb_clip_prim)
            ρ = max(U[i, j, k, 1], eps(FT))
            ρinv = one(FT) / ρ
            u = U[i, j, k, 2] * ρinv
            v = U[i, j, k, 3] * ρinv
            w = U[i, j, k, 4] * ρinv
            ei = max(U[i, j, k, 5] - FT(0.5) * ρ * (u * u + v * v + w * w), eps(FT))
            p = (γ - one(FT)) * ei

            # Positivity clipping
            ρ_min = FT(1.0e-5)
            p_min = FT(1.0e-5)
            ρ = max(ρ, ρ_min)
            p = max(p, p_min)
            T = p / (ρ * Rg)

            Q[i, j, k, 1] = ρ
            Q[i, j, k, 2] = u
            Q[i, j, k, 3] = v
            Q[i, j, k, 4] = w
            Q[i, j, k, 5] = p
            Q[i, j, k, 6] = T

            # Write back clamped U
            if ρ != U[i, j, k, 1] || p != (γ - one(FT)) * ei
                U[i, j, k, 1] = ρ
                U[i, j, k, 2] = ρ * u
                U[i, j, k, 3] = ρ * v
                U[i, j, k, 4] = ρ * w
                U[i, j, k, 5] = p / (γ - one(FT)) + FT(0.5) * ρ * (u * u + v * v + w * w)
            end
        end
    end
    return
end


# ─── Count cells on a hyperplane ───
# Returns the number of real cells (i,j,k) satisfying i+j+k == m
@inline function _plane_cell_count(nxp, nyp, nzp, m)
    count = 0
    k_lo = max(1, m - nxp - nyp)
    k_hi = min(nzp, m - 2)
    for kk = k_lo:k_hi
        j_lo = max(1, m - nxp - kk)
        j_hi = min(nyp, m - 1 - kk)
        n_j = max(0, j_hi - j_lo + 1)
        count += n_j
    end
    return count
end


# ═══════════════════════════════════════════════════════════════════════
#  Implicit step main function
#  Called from the time loop when `implicit == true`
# ═══════════════════════════════════════════════════════════════════════
function implicit_step!(block::Block, dt_val::FT,
                        shared_Fx, shared_Fy, shared_Fz,
                        shared_Fvx, shared_Fvy, shared_Fvz,
                        shared_dU_forced, world_rank, tt,
                        threads_recon_i, threads_recon_j, threads_recon_k,
                        threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                        forcex, flowx, cmf_f1_val, ac_f1_val,
                        hit_u_mean, hit_v_mean, hit_w_mean, activeTime)
    nxp = block.Nx; nyp = block.Ny; nzp = block.Nz
    Nx_tot = nxp + 2 * NG; Ny_tot = nyp + 2 * NG; Nz_tot = nzp + 2 * NG

    nb_light = (Int32(cld(Nx_tot, threads_light[1])),
                Int32(cld(Ny_tot, threads_light[2])),
                Int32(cld(Nz_tot, threads_light[3])))

    # ── Step 1: Compute spectral radii on all faces ──
    nb_si = (cld(nxp + 1, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    nb_sj = (cld(nxp, nthreads[1]), cld(nyp + 1, nthreads[2]), cld(nzp, nthreads[3]))
    nb_sk = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp + 1, nthreads[3]))

    @gpu_launch threads=nthreads blocks=nb_si compute_spectral_radius_i!(
        block.σ_i, block.Q, block.Areai, block.nxi, block.nyi, block.nzi, nxp, nyp, nzp)
    @gpu_launch threads=nthreads blocks=nb_sj compute_spectral_radius_j!(
        block.σ_j, block.Q, block.Areaj, block.nxj, block.nyj, block.nzj, nxp, nyp, nzp)
    @gpu_launch threads=nthreads blocks=nb_sk compute_spectral_radius_k!(
        block.σ_k, block.Q, block.Areak, block.nxk, block.nyk, block.nzk, nxp, nyp, nzp)

    # ── Step 2: Compute diagonal D and its inverse ──
    nb_real = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    w_LU = isdefined(Main, :implicit_w_LU) ? Main.implicit_w_LU : FT(1.5e0)
    @gpu_launch threads=nthreads blocks=nb_real compute_lusgs_diagonal!(
        block.D_inv, block.Q, block.Vol, block.σ_i, block.σ_j, block.σ_k,
        block.Areai, block.Areaj, block.Areak, dt_val, w_LU, nxp, nyp, nzp)

    # ── Step 3: Compute explicit RHS ──
    # 3a. Reconstruction + Riemann solver + viscous flux (reuse blockAdvance)
    blockAdvance(block, dt_val, block.ϕ, shared_Fx, shared_Fy, shared_Fz,
                 shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt,
                 threads_recon_i, threads_recon_j, threads_recon_k,
                 threads_visc_i, threads_visc_j, threads_visc_k)

    # 3b. Volume forces (reuse existing logic)
    if flow_forcing
        x_rot_start_val = isdefined(Main, :x_rot_start) ? FT(Main.x_rot_start) : (isdefined(@__MODULE__, :x_rot_start) ? FT(x_rot_start) : zero(FT))
        x_rot_end_val   = isdefined(Main, :x_rot_end)   ? FT(Main.x_rot_end)   : (isdefined(@__MODULE__, :x_rot_end)   ? FT(x_rot_end)   : zero(FT))
        omega_x_val     = isdefined(Main, :Omega_x)     ? FT(Main.Omega_x)     : (isdefined(@__MODULE__, :Omega_x)     ? FT(Omega_x)     : zero(FT))
        @gpu_launch threads=threads_light blocks=nb_light Volume_force_kernel!(
            shared_dU_forced, block.Q, block.x, block.y, block.z,
            nxp, nyp, nzp, block.Ωx, block.Ωy, block.Ωz, x_rot_start_val, x_rot_end_val, omega_x_val)
        if forcing_mode == 1
            Apply_bulk_force!(shared_dU_forced, block.Q, forcex, flowx, dt_val, nxp, nyp, nzp)
        elseif forcing_mode == 2
            Apply_const_massflux_force!(shared_dU_forced, block.Q, cmf_f1_val, nxp, nyp, nzp)
        end
        Apply_trip_force!(shared_dU_forced, block.Q, block.x, block.y, block.z,
                          nxp, nyp, nzp, activeTime)
    end
    if test_case == "HIT"
        Apply_HIT_forcing!(shared_dU_forced, block.Q, hit_forcing_A,
                           hit_u_mean, hit_v_mean, hit_w_mean, nxp, nyp, nzp)
    end

    # 3c. Compute RHS = -divF + source, store in dU_rhs (not in-place)
    @gpu_launch threads=nthreads blocks=nb_real div_to_rhs(
        block.dU_rhs, block.U, shared_Fx, shared_Fy, shared_Fz,
        shared_Fvx, shared_Fvy, shared_Fvz,
        shared_dU_forced, dt_val, block.Vol, nxp, nyp, nzp)



    # ── Step 3d: Implicit Residual Smoothing (EC-style) ──
    # Smooth dU_rhs to damp high-frequency modes before LU-SGS
    # Uses ΔU as temp buffer (will be zeroed in Step 4)
    implicit_res_smooth = isdefined(@__MODULE__, :implicit_residual_smoothing) ? implicit_residual_smoothing : false
    if implicit_res_smooth
        res_smooth_eps = isdefined(@__MODULE__, :implicit_res_smooth_eps) ? implicit_res_smooth_eps : FT(0.15e0)
        res_smooth_passes = isdefined(@__MODULE__, :implicit_res_smooth_passes) ? implicit_res_smooth_passes : 2
        # dU_rhs is [1:nxp, 1:nyp, 1:nzp, 1:Ncons] — local indexing
        # We need a temp buffer with the same layout. Use a view of ΔU in local coords.
        # ΔU is [1:Nx+2NG, ...] in global coords. Extract local subarray.
        dU_temp = @view block.ΔU[1+NG:nxp+NG, 1+NG:nyp+NG, 1+NG:nzp+NG, :]
        for pass = 1:res_smooth_passes
            copyto!(dU_temp, block.dU_rhs)  # copy RHS → temp
            @gpu_launch threads=nthreads blocks=nb_real residual_smooth_kernel!(
                block.dU_rhs, dU_temp, res_smooth_eps, nxp, nyp, nzp)
        end
    end

    # ── Step 4: Initialize ΔU to zero ──
    fill!(block.ΔU, zero(FT))

    # ── Step 5: LU-SGS sweeps ──
    # DIAGNOSTIC: use Jacobi (diagonal-only) to isolate sweep bugs
    lusgs_debug_jacobi = isdefined(@__MODULE__, :implicit_lusgs_debug_jacobi) ? implicit_lusgs_debug_jacobi : false

    if lusgs_debug_jacobi
        # ΔU = D_inv × RHS — equivalent to damped forward Euler
        @gpu_launch threads=nthreads blocks=nb_real jacobi_update!(
            block.ΔU, block.dU_rhs, block.D_inv, nxp, nyp, nzp)
    else
        for sweep = 1:implicit_lusgs_sweeps
            # Forward sweep: planes m = 3, 4, ..., nxp+nyp+nzp
            for m = Int32(3):Int32(nxp + nyp + nzp)
                n_cells = _plane_cell_count(nxp, nyp, nzp, m)
                if n_cells > 0
                    nb_plane = cld(n_cells, 256)
                    @gpu_launch threads=256 blocks=nb_plane lusgs_forward_sweep_plane!(
                        block.ΔU, block.dU_rhs, block.D_inv,
                        block.σ_i, block.σ_j, block.σ_k,
                        Int32(nxp), Int32(nyp), Int32(nzp), m)
                    gpu_sync()  # Must sync between planes (data dependency)
                end
            end

            # Backward sweep: planes m = nxp+nyp+nzp, ..., 3
            for m = Int32(nxp + nyp + nzp):-Int32(1):Int32(3)
                n_cells = _plane_cell_count(nxp, nyp, nzp, m)
                if n_cells > 0
                    nb_plane = cld(n_cells, 256)
                    @gpu_launch threads=256 blocks=nb_plane lusgs_backward_sweep_plane!(
                        block.ΔU, block.D_inv,
                        block.σ_i, block.σ_j, block.σ_k,
                        Int32(nxp), Int32(nyp), Int32(nzp), m)
                    gpu_sync()
                end
            end
        end
    end

    # ── Step 6: Update U^{n+1} = U^n + ΔU, derive Q, clip ──
    @gpu_launch threads=threads_light blocks=nb_light implicit_update!(
        block.U, block.Q, block.ΔU, nxp, nyp, nzp)
end


# ═══════════════════════════════════════════════════════════════════════
#  BDF2 Dual-Time Stepping Support
#  For 2nd-order temporal accuracy: BDF2 + LU-SGS pseudo-time iteration
#
#  Physical time discretization (BDF2):
#    (3/(2Δt)) · Vol · U^{n+1} - (2/Δt) · Vol · U^n + (1/(2Δt)) · Vol · U^{n-1} = -R(U^{n+1})
#
#  Rearranged for pseudo-time iteration (m = inner iteration index):
#    [(3/(2Δt))·Vol·I + ∂R/∂U] · δU = -R(U^{n+1,m}) - (3/(2Δt))·Vol·U^{n+1,m}
#                                       + (2/Δt)·Vol·U^n - (1/(2Δt))·Vol·U^{n-1}
#
#  where δU = U^{n+1,m+1} - U^{n+1,m}
# ═══════════════════════════════════════════════════════════════════════

# ─── BDF2 source: adds temporal derivative terms to dU_rhs ───
# Adds: (2/dt)·Vol·U^n - (1/(2dt))·Vol·U^{n-1} - (3/(2dt))·Vol·U^{n+1,m}
# to the existing spatial RHS (which already contains flux_div + source)
function add_bdf2_source!(dU_rhs, U_curr, U_n, U_nm1, Vol, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    ig = i + NG; jg = j + NG; kg = k + NG

    @inbounds begin
        vol_inv = Vol[ig, jg, kg]  # Vol stores 1/Volume (Jacobian)
        cell_vol = one(FT) / (vol_inv + FT(1.0e-30))  # Actual cell volume [m³]
        dt_inv = one(FT) / dt

        for n = Int32(1):Int32(Ncons)
            # BDF2 temporal terms (using actual cell volume):
            # + (2/dt)·V·U^n - (1/(2dt))·V·U^{n-1} - (3/(2dt))·V·U^{n+1,m}
            bdf2_src = cell_vol * dt_inv * (
                FT(2.0) * U_n[ig, jg, kg, n] -
                FT(0.5) * U_nm1[ig, jg, kg, n] -
                FT(1.5e0) * U_curr[ig, jg, kg, n]
            )
            dU_rhs[i, j, k, n] += bdf2_src
        end
    end
    return
end


# ─── Diagonal for BDF2: D = (3/(2dt))·Vol + 0.5·Σσ ───
# The diagonal is modified to include the BDF2 time derivative coefficient
function compute_lusgs_diagonal_bdf2!(D_inv, Q, Vol, σ_i, σ_j, σ_k,
                                       Areai, Areaj, Areak, dt, w_LU, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end

    ig = i + NG
    jg = j + NG
    kg = k + NG

    @inbounds begin
        vol_inv = Vol[ig, jg, kg]  # Vol stores 1/Volume (Jacobian)
        cell_vol = one(FT) / (vol_inv + FT(1.0e-30))  # Actual cell volume [m³]
        σ_sum = σ_i[i, j, k] + σ_i[i+1, j, k] +
                σ_j[i, j, k] + σ_j[i, j+1, k] +
                σ_k[i, j, k] + σ_k[i, j, k+1]

        # Viscous spectral radius (same as 1st-order diagonal)
        if viscous
            Ai = FT(0.5) * (Areai[ig, jg, kg] + Areai[ig+1, jg, kg])
            Aj = FT(0.5) * (Areaj[ig, jg, kg] + Areaj[ig, jg+1, kg])
            Ak = FT(0.5) * (Areak[ig, jg, kg] + Areak[ig, jg, kg+1])

            ρ_c = max(Q[ig, jg, kg, 1], FT(1.0e-10))
            T_c = max(Q[ig, jg, kg, 6], FT(1.0e-10))
            μ_c = get_viscosity(T_c)
            nu_eff = max(FT(4.0)/FT(3.0) * μ_c, γ * μ_c / Pr) / ρ_c
            σ_visc = nu_eff * (Ai*Ai + Aj*Aj + Ak*Ak) / (cell_vol + FT(1.0e-30))
            σ_sum += FT(2.0) * σ_visc
        end

        @static if equation_type == :MHD
            if resistive
                Ai_res = FT(0.5) * (Areai[ig, jg, kg] + Areai[ig+1, jg, kg])
                Aj_res = FT(0.5) * (Areaj[ig, jg, kg] + Areaj[ig, jg+1, kg])
                Ak_res = FT(0.5) * (Areak[ig, jg, kg] + Areak[ig, jg, kg+1])
                σ_res = η_mhd * (Ai_res*Ai_res + Aj_res*Aj_res + Ak_res*Ak_res) / (cell_vol + FT(1.0e-30))
                σ_sum += FT(2.0) * σ_res
            end
        end



        # BDF2 diagonal: D = 1.5V/dt + w_LU·Σσ (consistent with add_bdf2_source!)
        w_LU = isdefined(@__MODULE__, :implicit_w_LU) ? implicit_w_LU : FT(1.5e0)
        D = FT(1.5e0) * cell_vol / dt + w_LU * σ_sum  # [m³/s]
        D_inv[ig, jg, kg] = one(FT) / (D + FT(1.0e-30))
    end
    return
end


# ─── L2 residual norm (reduction on CPU after GPU copy) ───
function _residual_l2(dU_rhs, nxp, nyp, nzp)
    # dU_rhs is a standalone ROCArray (not SubArray) — mapreduce is safe on AMDGPU
    return Float64(mapreduce(abs, max, dU_rhs))
end


# ═══════════════════════════════════════════════════════════════════════
#  BDF2 Dual-Time Stepping Functions
# ═══════════════════════════════════════════════════════════════════════

function bdf2_prepare_step!(block::Block, dt_val::FT)
    nxp = block.Nx; nyp = block.Ny; nzp = block.Nz
    nb_real = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))

    # Compute spectral radii 
    nb_si = (cld(nxp + 1, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    nb_sj = (cld(nxp, nthreads[1]), cld(nyp + 1, nthreads[2]), cld(nzp, nthreads[3]))
    nb_sk = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp + 1, nthreads[3]))

    @gpu_launch threads=nthreads blocks=nb_si compute_spectral_radius_i!(
        block.σ_i, block.Q, block.Areai, block.nxi, block.nyi, block.nzi, nxp, nyp, nzp)
    @gpu_launch threads=nthreads blocks=nb_sj compute_spectral_radius_j!(
        block.σ_j, block.Q, block.Areaj, block.nxj, block.nyj, block.nzj, nxp, nyp, nzp)
    @gpu_launch threads=nthreads blocks=nb_sk compute_spectral_radius_k!(
        block.σ_k, block.Q, block.Areak, block.nxk, block.nyk, block.nzk, nxp, nyp, nzp)

    # Compute BDF2 diagonal
    w_LU = isdefined(Main, :implicit_w_LU) ? Main.implicit_w_LU : FT(1.5e0)
    @gpu_launch threads=nthreads blocks=nb_real compute_lusgs_diagonal_bdf2!(
        block.D_inv, block.Q, block.Vol, block.σ_i, block.σ_j, block.σ_k,
        block.Areai, block.Areaj, block.Areak, dt_val, w_LU, nxp, nyp, nzp)
end

function bdf2_inner_iteration!(block::Block, dt_val::FT,
                         shared_Fx, shared_Fy, shared_Fz,
                         shared_Fvx, shared_Fvy, shared_Fvz,
                         shared_dU_forced, world_rank, tt,
                         threads_recon_i, threads_recon_j, threads_recon_k,
                         threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                         forcex, flowx, cmf_f1_val, ac_f1_val,
                         hit_u_mean, hit_v_mean, hit_w_mean, activeTime;
                         sync_ghost_fn=nothing)  # ghost exchange callback for GMRES matvec
    nxp = block.Nx; nyp = block.Ny; nzp = block.Nz
    Nx_tot = nxp + 2 * NG; Ny_tot = nyp + 2 * NG; Nz_tot = nzp + 2 * NG

    nb_light = (Int32(cld(Nx_tot, threads_light[1])),
                Int32(cld(Ny_tot, threads_light[2])),
                Int32(cld(Nz_tot, threads_light[3])))
    nb_real = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))

    # Compute spatial RHS at current state U^{n+1,m}
    blockAdvance(block, dt_val, block.ϕ, shared_Fx, shared_Fy, shared_Fz,
                 shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt,
                 threads_recon_i, threads_recon_j, threads_recon_k,
                 threads_visc_i, threads_visc_j, threads_visc_k)

    # Volume forces
    if flow_forcing
        x_rot_start_val = isdefined(Main, :x_rot_start) ? FT(Main.x_rot_start) : (isdefined(@__MODULE__, :x_rot_start) ? FT(x_rot_start) : zero(FT))
        x_rot_end_val   = isdefined(Main, :x_rot_end)   ? FT(Main.x_rot_end)   : (isdefined(@__MODULE__, :x_rot_end)   ? FT(x_rot_end)   : zero(FT))
        omega_x_val     = isdefined(Main, :Omega_x)     ? FT(Main.Omega_x)     : (isdefined(@__MODULE__, :Omega_x)     ? FT(Omega_x)     : zero(FT))
        @gpu_launch threads=threads_light blocks=nb_light Volume_force_kernel!(
            shared_dU_forced, block.Q, block.x, block.y, block.z,
            nxp, nyp, nzp, block.Ωx, block.Ωy, block.Ωz, x_rot_start_val, x_rot_end_val, omega_x_val)
        if forcing_mode == 1
            Apply_bulk_force!(shared_dU_forced, block.Q, forcex, flowx, dt_val, nxp, nyp, nzp)
        elseif forcing_mode == 2
            Apply_const_massflux_force!(shared_dU_forced, block.Q, cmf_f1_val, nxp, nyp, nzp)
        end
        Apply_trip_force!(shared_dU_forced, block.Q, block.x, block.y, block.z,
                          nxp, nyp, nzp, activeTime)
    end
    if test_case == "HIT"
        Apply_HIT_forcing!(shared_dU_forced, block.Q, hit_forcing_A,
                           hit_u_mean, hit_v_mean, hit_w_mean, nxp, nyp, nzp)
    end

    # Compute spatial RHS → dU_rhs
    @gpu_launch threads=nthreads blocks=nb_real div_to_rhs(
        block.dU_rhs, block.U, shared_Fx, shared_Fy, shared_Fz,
        shared_Fvx, shared_Fvy, shared_Fvz,
        shared_dU_forced, dt_val, block.Vol, nxp, nyp, nzp)



    # Save spatial RHS (before BDF2 source) for GMRES matvec if using GMRES
    _use_gmres_solver = isdefined(@__MODULE__, :implicit_solver) ? (implicit_solver == :gmres) : false
    if _use_gmres_solver && block.R_base !== nothing
        copyto!(block.R_base, block.dU_rhs)
    end

    # Add BDF2 temporal source terms to dU_rhs
    @gpu_launch threads=nthreads blocks=nb_real add_bdf2_source!(
        block.dU_rhs, block.U, block.Un, block.U_nm1, block.Vol, dt_val, nxp, nyp, nzp)

    # ── Solve linear system ──
    fill!(block.ΔU, zero(FT))

    if _use_gmres_solver && block.V_krylov !== nothing
        # GMRES(m) with Block-Jacobi preconditioning
        _gm = isdefined(@__MODULE__, :gmres_m) ? gmres_m : 10
        _gr = isdefined(@__MODULE__, :gmres_max_restarts) ? gmres_max_restarts : 2
        gmres_solve!(block, dt_val, FT(1.5e0),  # alpha_bdf = 1.5 for BDF2
                     _gm, _gr, Float64(dual_time_tol),
                     block.V_krylov, block.R_base,
                     shared_Fx, shared_Fy, shared_Fz,
                     shared_Fvx, shared_Fvy, shared_Fvz,
                     shared_dU_forced, world_rank, tt,
                     threads_recon_i, threads_recon_j, threads_recon_k,
                     threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                     forcex, flowx, cmf_f1_val, ac_f1_val,
                     hit_u_mean, hit_v_mean, hit_w_mean, activeTime,
                     sync_ghost_fn !== nothing ? sync_ghost_fn : () -> nothing)
    else
        # LU-SGS sweep (existing path)
        lusgs_debug_jacobi = isdefined(@__MODULE__, :implicit_lusgs_debug_jacobi) ? implicit_lusgs_debug_jacobi : false
        if lusgs_debug_jacobi
            @gpu_launch threads=nthreads blocks=nb_real jacobi_update!(
                block.ΔU, block.dU_rhs, block.D_inv, nxp, nyp, nzp)
        else
            for sweep = 1:implicit_lusgs_sweeps
                for m = Int32(3):Int32(nxp + nyp + nzp)
                    n_cells = _plane_cell_count(nxp, nyp, nzp, m)
                    if n_cells > 0
                        nb_plane = cld(n_cells, 256)
                        @gpu_launch threads=256 blocks=nb_plane lusgs_forward_sweep_plane!(
                            block.ΔU, block.dU_rhs, block.D_inv,
                            block.σ_i, block.σ_j, block.σ_k,
                            Int32(nxp), Int32(nyp), Int32(nzp), m)
                        gpu_sync()
                    end
                end

                for m = Int32(nxp + nyp + nzp):-Int32(1):Int32(3)
                    n_cells = _plane_cell_count(nxp, nyp, nzp, m)
                    if n_cells > 0
                        nb_plane = cld(n_cells, 256)
                        @gpu_launch threads=256 blocks=nb_plane lusgs_backward_sweep_plane!(
                            block.ΔU, block.D_inv,
                            block.σ_i, block.σ_j, block.σ_k,
                            Int32(nxp), Int32(nyp), Int32(nzp), m)
                        gpu_sync()
                    end
                end
            end
        end
    end

    # Update: U^{n+1,m+1} = U^{n+1,m} + δU
    @gpu_launch threads=threads_light blocks=nb_light implicit_update!(
        block.U, block.Q, block.ΔU, nxp, nyp, nzp)

    return _residual_l2(block.dU_rhs, nxp, nyp, nzp)
end

