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

if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end

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
            ei = max(
                U[i, j, k, 5] - FT(0.5) * ρ * (u * u + v * v + w * w) -
                FT(0.5) * B2 * INV_MU0_SI, eps(FT),
            )
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
            U[i, j, k, 5] = p / (γ - one(FT)) +
                FT(0.5) * ρ * (u * u + v * v + w * w) +
                FT(0.5) * B2 * INV_MU0_SI
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
                        shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                        shared_Fvx, shared_Fvy, shared_Fvz,
                        shared_dU_forced, shared_ct_pos_meta, shared_ct_pos_values,
                        world_rank, tt,
                        threads_recon_i, threads_recon_j, threads_recon_k,
                        threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                        forcex, flowx, cmf_f1_val,
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
    # 3a. Reconstruction + Riemann solver + viscous flux
    compute_structured_face_fluxes!(block, dt_val, block.ϕ, shared_Fx, shared_Fy, shared_Fz,
                 shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                 shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt,
                 threads_recon_i, threads_recon_j, threads_recon_k,
                 threads_visc_i, threads_visc_j, threads_visc_k,
                 Int32(1), shared_ct_pos_meta, shared_ct_pos_values)

    # 3b. Volume forces (reuse existing logic)
    if flow_forcing
        omega_x_val     = isdefined(Main, :Omega_x)     ? FT(Main.Omega_x)     : (isdefined(@__MODULE__, :Omega_x)     ? FT(Omega_x)     : zero(FT))
        @gpu_launch threads=threads_light blocks=nb_light Volume_force_kernel!(
            shared_dU_forced, block.Q, block.y, block.z,
            nxp, nyp, nzp, omega_x_val)
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
                         shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                         shared_Fvx, shared_Fvy, shared_Fvz,
                         shared_dU_forced, shared_ct_pos_meta, shared_ct_pos_values,
                         world_rank, tt,
                         threads_recon_i, threads_recon_j, threads_recon_k,
                         threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                         forcex, flowx, cmf_f1_val,
                         hit_u_mean, hit_v_mean, hit_w_mean, activeTime;
                         sync_ghost_fn=nothing)  # ghost exchange callback for GMRES matvec
    nxp = block.Nx; nyp = block.Ny; nzp = block.Nz
    Nx_tot = nxp + 2 * NG; Ny_tot = nyp + 2 * NG; Nz_tot = nzp + 2 * NG

    nb_light = (Int32(cld(Nx_tot, threads_light[1])),
                Int32(cld(Ny_tot, threads_light[2])),
                Int32(cld(Nz_tot, threads_light[3])))
    nb_real = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))

    # Compute spatial RHS at current state U^{n+1,m}
    compute_structured_face_fluxes!(block, dt_val, block.ϕ, shared_Fx, shared_Fy, shared_Fz,
                 shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                 shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt,
                 threads_recon_i, threads_recon_j, threads_recon_k,
                 threads_visc_i, threads_visc_j, threads_visc_k,
                 Int32(1), shared_ct_pos_meta, shared_ct_pos_values)

    # Volume forces
    if flow_forcing
        omega_x_val     = isdefined(Main, :Omega_x)     ? FT(Main.Omega_x)     : (isdefined(@__MODULE__, :Omega_x)     ? FT(Omega_x)     : zero(FT))
        @gpu_launch threads=threads_light blocks=nb_light Volume_force_kernel!(
            shared_dU_forced, block.Q, block.y, block.z,
            nxp, nyp, nzp, omega_x_val)
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
                     shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                     shared_Fvx, shared_Fvy, shared_Fvz,
                     shared_dU_forced, shared_ct_pos_meta, shared_ct_pos_values,
                     world_rank, tt,
                     threads_recon_i, threads_recon_j, threads_recon_k,
                     threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                     forcex, flowx, cmf_f1_val,
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


# =============================================================================
# Matrix-Free GMRES(m) with Block-Jacobi Preconditioning
# Folded from gmres.jl.
# =============================================================================
# ═══════════════════════════════════════════════════════════════════════
#  Matrix-Free GMRES(m) with Block-Jacobi Preconditioning
#  for Implicit Time Stepping in Flame3D
#
#  Replaces LU-SGS hyperplane sweeps with a fully-parallel Krylov solver.
#
#  Solves: A·x = b  where
#    A = (αV/(dt))·I + ∂R/∂U   (α = 1 for BE, 1.5 for BDF2)
#    b = dU_rhs                  (spatial RHS + temporal source)
#    x = ΔU                      (solution increment)
#
#  Matrix-vector product via finite difference:
#    A·v ≈ [R(U+εv) - R(U)] / ε + (αV/dt)·v
#
#  Left-preconditioned: solve P⁻¹Ax = P⁻¹b with P = diag(A) = D
# ═══════════════════════════════════════════════════════════════════════

# ─── GPU kernels for vector operations ───

# w = P⁻¹·v (apply block-Jacobi preconditioner = D_inv × v)
# Both w and v are (Nx_tot, Ny_tot, Nz_tot, Ncons) — global indexing with ghost cells
function gmres_precond!(w, v, D_inv, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    @inbounds d_inv = D_inv[ig, jg, kg]
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = d_inv * v[ig, jg, kg, n]
    end
    return
end

# w = P⁻¹·rhs where rhs is (nxp, nyp, nzp, Ncons) — LOCAL indexing (no ghost cells)
# w is (Nx_tot, Ny_tot, Nz_tot, Ncons) — GLOBAL indexing with ghost cells
function gmres_precond_rhs!(w, rhs, D_inv, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    @inbounds d_inv = D_inv[ig, jg, kg]
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = d_inv * rhs[i, j, k, n]  # local read, global write
    end
    return
end

# w = U_base + ε·v  (perturb state for finite-difference Jacobian)
function gmres_perturb!(w, U_base, v, eps_fd, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = U_base[ig, jg, kg, n] + eps_fd * v[ig, jg, kg, n]
    end
    return
end

# Av = (αV/dt)·v − (R_perturbed - R_base) / ε
# The operator A = αV/dt − ∂R_spatial/∂U  (positive definite for dissipative systems)
# FD gives ∂R/∂U·v ≈ (R(U+εv) - R(U))/ε, so we NEGATE it.
# Then apply preconditioner: result = P⁻¹·Av
function gmres_compute_Av!(Av, R_perturbed, R_base, v, D_inv, Vol, dt, alpha_bdf, eps_inv, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    @inbounds begin
        vol_inv = Vol[ig, jg, kg]
        cell_vol = one(FT) / (vol_inv + FT(1.0e-30))
        d_inv = D_inv[ig, jg, kg]
        temporal = alpha_bdf * cell_vol / dt
        for n = Int32(1):Int32(Ncons)
            # Jacobian-vector product: ∂R_sp/∂U · v ≈ (R(U+εv) - R(U)) / ε
            jac_v = (R_perturbed[i, j, k, n] - R_base[i, j, k, n]) * eps_inv
            # Full operator: A·v = (αV/dt)·v − ∂R_sp/∂U·v  (note: MINUS sign)
            av_n = temporal * v[ig, jg, kg, n] - jac_v
            # Left preconditioning: P⁻¹·A·v
            Av[ig, jg, kg, n] = d_inv * av_n
        end
    end
    return
end

# w = a*x + y (AXPY: GPU parallel)
function gmres_axpy!(w, a, x, y, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = a * x[ig, jg, kg, n] + y[ig, jg, kg, n]
    end
    return
end

# w = x - a*y (for Gram-Schmidt orthogonalization)
function gmres_sub_scaled!(w, x, a, y, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = x[ig, jg, kg, n] - a * y[ig, jg, kg, n]
    end
    return
end

# w = x / a (scale vector)
function gmres_scale!(w, x, a_inv, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = x[ig, jg, kg, n] * a_inv
    end
    return
end

# ─── CPU-side dot product (zero-allocation via GPU BLAS) ───
# Computes ⟨a, b⟩ over real cells [NG+1:Nx+NG, NG+1:Ny+NG, NG+1:Nz+NG, 1:Ncons]
function _gmres_dot(a::GPUArray{FT, 4}, b::GPUArray{FT, 4}, nxp, nyp, nzp)
    # Extract real-cell views and compute dot product
    a_real = @view a[NG+1:NG+nxp, NG+1:NG+nyp, NG+1:NG+nzp, :]
    b_real = @view b[NG+1:NG+nxp, NG+1:NG+nyp, NG+1:NG+nzp, :]
    # sum(a.*b) fuses via GPUArrays broadcast+mapreduce — single kernel
    # NOTE: LinearAlgebra.dot falls back to scalar iteration on N-D SubArrays
    return Float64(sum(a_real .* b_real))
end

# ─── CPU-side norm ───
function _gmres_norm(v::GPUArray{FT, 4}, nxp, nyp, nzp)
    return sqrt(_gmres_dot(v, v, nxp, nyp, nzp))
end


# ═══════════════════════════════════════════════════════════════════════
#  Main GMRES solve function
#
#  Solves P⁻¹·A·x = P⁻¹·b using GMRES(m) with restarts
#
#  Arguments:
#    block      - Block struct (contains U, Q, dU_rhs, ΔU, D_inv, etc.)
#    dt_val     - Time step
#    alpha_bdf  - BDF coefficient (1.0 for BE, 1.5 for BDF2)
#    m          - Krylov subspace dimension
#    max_restarts - Number of restarts
#    tol        - Relative residual tolerance
#    V_krylov   - Pre-allocated Krylov basis vectors [Nx_tot, Ny_tot, Nz_tot, Ncons, m+1]
#    shared_*   - Shared flux buffers
#    ...        - Other arguments passed to the face-flux computation
#
#  Returns: converged residual norm
# ═══════════════════════════════════════════════════════════════════════
function gmres_solve!(block::Block, dt_val::FT, alpha_bdf::FT,
                      m::Int, max_restarts::Int, tol::Float64,
                      V_krylov::GPUArray{FT, 5},
                      R_base::GPUArray{FT, 4},
                      shared_Fx, shared_Fy, shared_Fz,
                      shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                      shared_Fvx, shared_Fvy, shared_Fvz,
                      shared_dU_forced, shared_ct_pos_meta, shared_ct_pos_values,
                      world_rank, tt,
                      threads_recon_i, threads_recon_j, threads_recon_k,
                      threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                      forcex, flowx, cmf_f1_val,
                      hit_u_mean, hit_v_mean, hit_w_mean, activeTime,
                      sync_ghost_fn)  # callback: sync_ghost_fn() exchanges ghost cells for all blocks

    nxp = block.Nx; nyp = block.Ny; nzp = block.Nz
    Nx_tot = nxp + 2 * NG; Ny_tot = nyp + 2 * NG; Nz_tot = nzp + 2 * NG
    nb_real = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    nb_light = (Int32(cld(Nx_tot, threads_light[1])),
                Int32(cld(Ny_tot, threads_light[2])),
                Int32(cld(Nz_tot, threads_light[3])))

    # ── Compute P⁻¹·b (right-hand side with preconditioning) ──
    # b = dU_rhs is already computed, apply P⁻¹
    # Store P⁻¹·b in ΔU temporarily (will be overwritten by solution)
    @gpu_launch threads=nthreads blocks=nb_real gmres_precond_rhs!(
        block.ΔU, block.dU_rhs, block.D_inv, nxp, nyp, nzp)

    # ── Save the base spatial RHS (R_base = dU_rhs, before BDF2 source) ──
    # Note: R_base was computed by the caller and stored in the passed buffer

    # Finite-difference step size for Jacobian approximation
    # ε = sqrt(machine_eps) × (1 + ||U||) / ||v||
    # We'll compute per-iteration. For now, get ||U||.
    U_norm = _gmres_norm(block.U, nxp, nyp, nzp)

    # ── GMRES(m) with restarts ──
    final_res = Inf
    for restart = 1:max_restarts + 1
        # r₀ = P⁻¹·b - P⁻¹·A·x₀
        # Since x₀ = 0 (we start with ΔU = 0), r₀ = P⁻¹·b (already in block.ΔU)
        if restart == 1
            # First restart: r₀ = P⁻¹·b = block.ΔU
            # Copy r₀ into V[1]
            beta = _gmres_norm(block.ΔU, nxp, nyp, nzp)
            if beta < 1e-30
                fill!(block.ΔU, zero(FT))
                return 0.0
            end
            # V[1] = r₀ / β
            V1 = @view V_krylov[:, :, :, :, 1]
            inv_beta = FT(1.0 / beta)
            @gpu_launch threads=nthreads blocks=nb_real gmres_scale!(
                V1, block.ΔU, inv_beta, nxp, nyp, nzp)
        else
            # For restarts > 1: r = P⁻¹·b - P⁻¹·A·x_current
            # x_current is in block.ΔU, need to compute A·x and subtract
            # This is complex — for now, we don't restart (single GMRES(m) pass)
            break
        end

        # Hessenberg matrix H[m+1, m] and Givens rotation vectors (CPU)
        H = zeros(Float64, m + 1, m)
        cs = zeros(Float64, m)  # cosines
        sn = zeros(Float64, m)  # sines
        e1 = zeros(Float64, m + 1)
        e1[1] = beta

        converged = false
        actual_m = m  # May be less if converged early

        for j = 1:m
            # ── Arnoldi step: compute w = P⁻¹·A·V[j] ──
            Vj = @view V_krylov[:, :, :, :, j]
            Vj1 = @view V_krylov[:, :, :, :, j + 1]  # will store w after orthogonalization

            # 1. Perturb state: U_perturb = U + ε·Vj
            v_norm = _gmres_norm(Vj, nxp, nyp, nzp)
            eps_fd = FT(sqrt(eps(Float64)) * max(1.0, U_norm) / max(v_norm, 1e-30))

            # Save current U and Q
            copyto!(Vj1, block.U)  # Use Vj1 as temp to save U

            # Set U = U + ε·Vj, then Q = prim(U)
            @gpu_launch threads=nthreads blocks=nb_real gmres_perturb!(
                block.U, Vj1, Vj, eps_fd, nxp, nyp, nzp)


            # NOTE: Ghost cells are NOT re-synced after perturbation.
            # sync_blocks! is a global MPI collective and cannot be called from
            # inside a per-block GMRES loop (would cause MPI deadlock).
            # The perturbation ε ≈ 1e-7, so ghost error is O(ε²) ≈ O(1e-14).
            # Ghost cells are properly synced at the start of each BDF2 sub-iter.

            # 2. Compute R(U + ε·Vj) → store in a temp buffer
            # We reuse block.dU_rhs as the output for the perturbed RHS
            compute_structured_face_fluxes!(block, dt_val, block.ϕ, shared_Fx, shared_Fy, shared_Fz,
                         shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                         shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt,
                         threads_recon_i, threads_recon_j, threads_recon_k,
                         threads_visc_i, threads_visc_j, threads_visc_k,
                         Int32(1), shared_ct_pos_meta, shared_ct_pos_values)

            # R_perturbed = div_to_rhs output
            # Reuse block.dU_rhs as temp buffer (original RHS saved in R_base)
            @gpu_launch threads=nthreads blocks=nb_real div_to_rhs(
                block.dU_rhs, block.U, shared_Fx, shared_Fy, shared_Fz,
                shared_Fvx, shared_Fvy, shared_Fvz,
                shared_dU_forced, dt_val, block.Vol, nxp, nyp, nzp)



            # 3. Restore U and Q, then re-sync ghost cells
            copyto!(block.U, Vj1)  # Restore from temp

            # (Ghost sync intentionally omitted — see note above)

            # 4. Compute Av = P⁻¹ × [(R_perturbed - R_base) / ε + (αV/dt)·Vj]
            eps_inv = FT(1.0 / Float64(eps_fd))
            @gpu_launch threads=nthreads blocks=nb_real gmres_compute_Av!(
                Vj1, block.dU_rhs, R_base, Vj, block.D_inv, block.Vol,
                dt_val, alpha_bdf, eps_inv, nxp, nyp, nzp)

            # ── Modified Gram-Schmidt orthogonalization ──
            for i_gs = 1:j
                Vi = @view V_krylov[:, :, :, :, i_gs]
                h_ij = _gmres_dot(Vj1, Vi, nxp, nyp, nzp)
                H[i_gs, j] = h_ij
                # Vj1 = Vj1 - h_ij * Vi
                @gpu_launch threads=nthreads blocks=nb_real gmres_sub_scaled!(
                    Vj1, Vj1, FT(h_ij), Vi, nxp, nyp, nzp)
            end

            # Normalize
            h_jp1_j = _gmres_norm(Vj1, nxp, nyp, nzp)
            H[j + 1, j] = h_jp1_j
            if h_jp1_j > 1e-30
                inv_h = FT(1.0 / h_jp1_j)
                @gpu_launch threads=nthreads blocks=nb_real gmres_scale!(
                    Vj1, Vj1, inv_h, nxp, nyp, nzp)
            end

            # ── Apply previous Givens rotations to column j of H ──
            for i_rot = 1:j-1
                tmp = cs[i_rot] * H[i_rot, j] + sn[i_rot] * H[i_rot + 1, j]
                H[i_rot + 1, j] = -sn[i_rot] * H[i_rot, j] + cs[i_rot] * H[i_rot + 1, j]
                H[i_rot, j] = tmp
            end

            # ── Compute new Givens rotation ──
            rr = sqrt(H[j, j]^2 + H[j + 1, j]^2)
            if rr > 1e-30
                cs[j] = H[j, j] / rr
                sn[j] = H[j + 1, j] / rr
            else
                cs[j] = 1.0
                sn[j] = 0.0
            end
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            # ── Update residual estimate ──
            e1[j + 1] = -sn[j] * e1[j]
            e1[j] = cs[j] * e1[j]

            res_norm = abs(e1[j + 1])
            rel_res = res_norm / max(beta, 1e-30)

            if rel_res < tol
                actual_m = j
                converged = true
                break
            end
        end

        # ── Back-substitution: solve H·y = e1 ──
        y = zeros(Float64, actual_m)
        for i_bs = actual_m:-1:1
            y[i_bs] = e1[i_bs]
            for j_bs = i_bs + 1:actual_m
                y[i_bs] -= H[i_bs, j_bs] * y[j_bs]
            end
            y[i_bs] /= H[i_bs, i_bs]
        end

        # ── Construct solution: x = Σ yᵢ × Vᵢ ──
        fill!(block.ΔU, zero(FT))
        for i_sol = 1:actual_m
            Vi = @view V_krylov[:, :, :, :, i_sol]
            # ΔU += y[i] * V[i]
            @gpu_launch threads=nthreads blocks=nb_real gmres_axpy!(
                block.ΔU, FT(y[i_sol]), Vi, block.ΔU, nxp, nyp, nzp)
        end

        final_res = converged ? abs(e1[actual_m + 1]) : abs(e1[m + 1])

        if converged
            break
        end
    end

    return final_res
end
