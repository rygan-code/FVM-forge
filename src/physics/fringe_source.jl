# =============================================================================
#  Fringe Region Forcing Module
#
#  Implements the fringe forcing technique (Nordström et al. 1999) for
#  spatially developing flows in periodic domains.
#
#  Forcing term: F = λ(x) × (U_target - U)
#
#  λ(x) is zero in the physical zone and ramps up smoothly in the fringe zone,
#  absorbing upstream-propagating waves and recycling outflow → inflow.
# =============================================================================

export compute_fringe_lambda_kernel!, fringe_forcing_kernel!

# ─── Smooth step function S(η) for fringe ramp ───
# S(η) = 0 for η ≤ 0
# S(η) = 1 for η ≥ 1
# Infinitely differentiable transition (C∞)
@inline function _fringe_smooth_step(eta::T) where T
    if eta <= zero(T)
        return zero(T)
    elseif eta >= one(T)
        return one(T)
    else
        return one(T) / (one(T) + exp(one(T) / (eta - one(T)) + one(T) / eta))
    end
end

# ─── Pre-compute fringe strength λ(x) into a 3D array ───
# Called once during initialization. λ only depends on x coordinate.
#
#   Physical zone:  x ∈ [0, L_phys]    → λ = 0
#   Rise zone:      x ∈ [L_phys, L_phys + Δ_rise] → λ ramps 0→λ_max
#   Plateau zone:   x ∈ [L_phys + Δ_rise, L_total] → λ = λ_max
#
function compute_fringe_lambda_kernel!(lambda, x, nxp, nyp, nzp,
        L_phys, L_total, lambda_max, rise_fraction)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp + 2*NG || j > nyp + 2*NG || k > nzp + 2*NG
        return
    end

    @inbounds x_local = x[i, j, k]

    L_fringe = L_total - L_phys
    delta_rise = rise_fraction * L_fringe  # width of smooth ramp

    if x_local <= L_phys
        # Physical zone: no forcing
        @inbounds lambda[i, j, k] = zero(FT)
    else
        # Fringe zone: smooth ramp
        eta = (x_local - L_phys) / delta_rise
        @inbounds lambda[i, j, k] = lambda_max * _fringe_smooth_step(FT(eta))
    end

    return
end

# ─── Fringe forcing kernel: apply λ(x)(U_target - U) ───
# Called every RK sub-step, adds to dU_forced.
# U_target is the precursor mean profile (only depends on y, z).
function fringe_forcing_kernel!(dU_forced, U, U_target, lambda, nxp, nyp, nzp, dt_sub)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii, jj, kk = i + NG, j + NG, k + NG

    @inbounds lam = lambda[ii, jj, kk]

    # Skip if λ ≈ 0 (physical zone → no forcing)
    if lam <= FT(1e-10)
        return
    end

    # Add fringe forcing: λ × (U_target - U) × dt
    for m in 1:Ncell_cons
        @inbounds dU_forced[i, j, k, m] += lam * (U_target[ii, jj, kk, m] - U[ii, jj, kk, m]) * dt_sub
    end

    return
end
