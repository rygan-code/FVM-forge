# =============================================================================
#  Outlet Sponge / Buffer Layer Module
#
#  Implements a Mani-style outlet damping zone (mirror of fringe.jl) that
#  absorbs outgoing waves and suppresses mean-pressure drift / backflow
#  feedback near the outlet of each main (non-precursor) block.
#
#  Forcing term: F = σ(x) × (U_target - U)
#
#  σ(x) is zero in the physical zone and ramps up smoothly in the outlet
#  sponge zone (the LAST rise_fraction of the axial extent), reaching σ_max.
#
#  U_target is the running time-averaged / outlet-cross-section-mean profile,
#  so this damps PERTURBATIONS (not a hard Dirichlet target) — no fixed back
#  pressure is imposed.
# =============================================================================

export compute_sponge_sigma_kernel!, sponge_forcing_kernel!, update_sponge_U_target_kernel!, sponge_step_kernel!

# ─── Smooth step function S(η) (C∞ ramp), mirror of fringe.jl ───
@inline function _sponge_smooth_step(eta::T) where T
    if eta <= zero(T)
        return zero(T)
    elseif eta >= one(T)
        return one(T)
    else
        return one(T) / (one(T) + exp(one(T) / (eta - one(T)) + one(T) / eta))
    end
end

# ─── Pre-compute sponge strength σ(x) into a 3D array ───
# Called once during initialization. σ only depends on the axial coordinate x.
#
#   Physical zone:     x ∈ [0, x_sponge_start]      → σ = 0
#   Rise zone:         x ∈ [x_sponge_start, x_total]  → σ ramps 0→σ_max
#   (x_sponge_start = x_total - rise_fraction*(x_total - x_phys_start))
#
#   x_phys_start : start of the physical domain (usually 0)
#   x_total      : outlet end of the domain
#   rise_fraction: fraction of (x_total - x_phys_start) used for the ramp
#   sigma_max    : plateau damping strength (1/time)
#
function compute_sponge_sigma_kernel!(sigma, x, nxp, nyp, nzp,
        x_phys_start, x_total, sigma_max, rise_fraction)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp + 2*NG || j > nyp + 2*NG || k > nzp + 2*NG
        return
    end

    @inbounds x_local = x[i, j, k]

    L_phys = x_total - x_phys_start
    delta_rise = rise_fraction * L_phys  # width of smooth ramp (outlet tail)
    x_sponge_start = x_total - delta_rise

    if x_local <= x_sponge_start
        # Physical zone: no sponge
        @inbounds sigma[i, j, k] = zero(FT)
    else
        # Outlet sponge ramp
        eta = (x_local - x_sponge_start) / delta_rise
        @inbounds sigma[i, j, k] = sigma_max * _sponge_smooth_step(FT(eta))
    end

    return
end

# ─── Sponge forcing kernel: add σ(x)(U_target - U) dt ───
# Called every RK sub-step, adds to dU_forced. Mirrors fringe_forcing_kernel!.
# U_target is a per-cell running mean / outlet profile (only depends on y,z
# tiled along x, or a 3D running average depending on initialization).
function sponge_forcing_kernel!(dU_forced, U, U_target, sigma, nxp, nyp, nzp, dt_sub)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii, jj, kk = i + NG, j + NG, k + NG

    @inbounds s = sigma[ii, jj, kk]

    # Skip if σ ≈ 0 (physical zone → no forcing)
    if s <= FT(1e-10)
        return
    end

    # Add sponge forcing: σ × (U_target - U) × dt
    for m in 1:Ncons
        @inbounds dU_forced[i, j, k, m] += s * (U_target[ii, jj, kk, m] - U[ii, jj, kk, m]) * dt_sub
    end

    return
end

# ─── Running average update: U_target ← (1-α) U_target + α U ───
# Called periodically (every N_avg steps) to slowly adapt the sponge target
# to the current mean state. α small (e.g. 0.01) → slow relaxation, robust.
# Operates ONLY where σ > 0 (sponge zone) so the physical domain is untouched.
function update_sponge_U_target_kernel!(U_target, U, sigma, alpha, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp + 2*NG || j > nyp + 2*NG || k > nzp + 2*NG
        return
    end

    @inbounds s = sigma[i, j, k]
    if s <= FT(1e-10)
        return  # only update inside the sponge zone
    end

    for m in 1:Ncons
        @inbounds U_target[i, j, k, m] = (FT(1) - alpha) * U_target[i, j, k, m] + alpha * U[i, j, k, m]
    end

    return
end

# ─── Operator-split sponge step: U ← U + Δt · σ(x) · (U_target - U) ───
# Called ONCE per RK substep, right after the forcing/update commit, mirroring
# the GLM ψ-damping operator-split pattern in solver.jl. This avoids touching
# the shared_dU_forced accumulation pipeline (which branches into fused/non-fused
# paths) and applies the sponge uniformly to every block regardless of path.
# Uses an exponential-decay stable form: U_new = Ut + (U - Ut)·exp(-σ·Δt).
function sponge_step_kernel!(U, U_target, sigma, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp + 2*NG || j > nyp + 2*NG || k > nzp + 2*NG
        return
    end

    @inbounds s = sigma[i, j, k]
    if s <= FT(1e-10)
        return  # physical zone untouched
    end

    dec = exp(-s * dt)
    for m in 1:Ncons
        @inbounds U[i, j, k, m] = U_target[i, j, k, m] + (U[i, j, k, m] - U_target[i, j, k, m]) * dec
    end

    return
end