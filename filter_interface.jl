export interface_filter_kernel!

# =============================================================================
#  8th-order explicit interface filter with ADAPTIVE σ
#
#  Rationale (revised 2026-06):
#    The earlier rule σ ∝ (1 - lin_phi) GATED the filter OFF wherever the
#    reconstruction was already upwind-biased. That sounds reasonable, but at
#    the butterfly 3-block singularity apply_geometric_smoothness_protection
#    boosts the first 4 layers' phi to [1.0, 1.0, 0.8, 0.6] -> the filter
#    became sigma=0 there, even though those are exactly the cells most prone
#    to 2D odd-even (checkerboard) modes.
#
#    Upwind-bias dampens long-wave error but does NOT dampen 2D high-frequency
#    odd-even modes coming from cross-type metric mismatch / multi-block
#    junctions. The 8th-order filter is the only mechanism that targets that
#    spectrum, so we MUST keep it active there.
#
#    New rule: σ_local = σ_max × (σ_floor + (1 - σ_floor) × phi_local)
#       phi=0 (pure central, smooth interior)  -> σ = σ_floor × σ_max
#       phi=1 (heavily upwind, junction)        -> σ = σ_max
#    σ_floor = 0.5 keeps a baseline filter even in nicely-smooth regions and
#    pushes the strongest filter exactly into the singularity layers.
#
#  Stencil: U_new = U[i] - (σ_local/256) × Δ⁸U[i]
#  Δ⁸U = [1, -8, 28, -56, 70, -56, 28, -8, 1] (binomial coefficients)
#  Requires NG >= 4.  Width: 9 points (-4 to +4).
# =============================================================================

# Returns a multiplier in [σ_floor, 1.0]:
#   - σ_floor at phi=0  (pure central, low default damping is fine)
#   - 1.0      at phi=1  (junction / cross-type, MUST damp odd-even)
@inline function _intf_filter_weight(phi_local::T) where T
    σ_floor = T(0.5)    # never let the filter fully shut off
    return σ_floor + (one(T) - σ_floor) * clamp(phi_local, zero(T), one(T))
end

function interface_filter_kernel!(U, Nx::Int, Ny::Int, Nz::Int, fid::Int, num_layers::Int,
                                   sigma_max, lin_phi)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    k_or_j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if fid == 1
        # ξ- boundary, filter along i.  Thread grid: (j, k)
        j_idx = i + NG
        k_idx = k_or_j + NG
        if j_idx <= Ny + NG && k_idx <= Nz + NG
            for layer in 1:min(num_layers, 4)
                ic = NG + layer
                s256 = sigma_max / FT(256.0)  # fixed σ for ξ direction
                for m in 1:Ncons
                    @inbounds begin
                        d8 = U[ic-4,j_idx,k_idx,m] - FT(8.0)*U[ic-3,j_idx,k_idx,m] +
                             FT(28.0)*U[ic-2,j_idx,k_idx,m] - FT(56.0)*U[ic-1,j_idx,k_idx,m] +
                             FT(70.0)*U[ic,j_idx,k_idx,m] -
                             FT(56.0)*U[ic+1,j_idx,k_idx,m] + FT(28.0)*U[ic+2,j_idx,k_idx,m] -
                             FT(8.0)*U[ic+3,j_idx,k_idx,m] + U[ic+4,j_idx,k_idx,m]
                        U[ic,j_idx,k_idx,m] -= s256 * d8
                    end
                end
            end
        end

    elseif fid == 2
        # ξ+ boundary.  Thread grid: (j, k)
        j_idx = i + NG
        k_idx = k_or_j + NG
        if j_idx <= Ny + NG && k_idx <= Nz + NG
            ei = Nx + NG
            for layer in 1:min(num_layers, 4)
                ic = ei - layer + 1
                s256 = sigma_max / FT(256.0)  # fixed σ for ξ direction
                for m in 1:Ncons
                    @inbounds begin
                        d8 = U[ic-4,j_idx,k_idx,m] - FT(8.0)*U[ic-3,j_idx,k_idx,m] +
                             FT(28.0)*U[ic-2,j_idx,k_idx,m] - FT(56.0)*U[ic-1,j_idx,k_idx,m] +
                             FT(70.0)*U[ic,j_idx,k_idx,m] -
                             FT(56.0)*U[ic+1,j_idx,k_idx,m] + FT(28.0)*U[ic+2,j_idx,k_idx,m] -
                             FT(8.0)*U[ic+3,j_idx,k_idx,m] + U[ic+4,j_idx,k_idx,m]
                        U[ic,j_idx,k_idx,m] -= s256 * d8
                    end
                end
            end
        end

    elseif fid == 3
        # η- boundary, filter along j.  Thread grid: (i, k)
        i_idx = i + NG
        k_idx = k_or_j + NG
        if i_idx <= Nx + NG && k_idx <= Nz + NG
            for layer in 1:min(num_layers, 4)
                jc = NG + layer
                # Adaptive σ: high lin_phi → low σ (already has upwind dissipation)
                @inbounds phi_local = lin_phi[jc, k_idx]
                σ_local = sigma_max * _intf_filter_weight(phi_local)
                s256 = σ_local / FT(256.0)
                for m in 1:Ncons
                    @inbounds begin
                        d8 = U[i_idx,jc-4,k_idx,m] - FT(8.0)*U[i_idx,jc-3,k_idx,m] +
                             FT(28.0)*U[i_idx,jc-2,k_idx,m] - FT(56.0)*U[i_idx,jc-1,k_idx,m] +
                             FT(70.0)*U[i_idx,jc,k_idx,m] -
                             FT(56.0)*U[i_idx,jc+1,k_idx,m] + FT(28.0)*U[i_idx,jc+2,k_idx,m] -
                             FT(8.0)*U[i_idx,jc+3,k_idx,m] + U[i_idx,jc+4,k_idx,m]
                        U[i_idx,jc,k_idx,m] -= s256 * d8
                    end
                end
            end
        end

    elseif fid == 4
        # η+ boundary.  Thread grid: (i, k)
        i_idx = i + NG
        k_idx = k_or_j + NG
        if i_idx <= Nx + NG && k_idx <= Nz + NG
            ej = Ny + NG
            for layer in 1:min(num_layers, 4)
                jc = ej - layer + 1
                @inbounds phi_local = lin_phi[jc, k_idx]
                σ_local = sigma_max * _intf_filter_weight(phi_local)
                s256 = σ_local / FT(256.0)
                for m in 1:Ncons
                    @inbounds begin
                        d8 = U[i_idx,jc-4,k_idx,m] - FT(8.0)*U[i_idx,jc-3,k_idx,m] +
                             FT(28.0)*U[i_idx,jc-2,k_idx,m] - FT(56.0)*U[i_idx,jc-1,k_idx,m] +
                             FT(70.0)*U[i_idx,jc,k_idx,m] -
                             FT(56.0)*U[i_idx,jc+1,k_idx,m] + FT(28.0)*U[i_idx,jc+2,k_idx,m] -
                             FT(8.0)*U[i_idx,jc+3,k_idx,m] + U[i_idx,jc+4,k_idx,m]
                        U[i_idx,jc,k_idx,m] -= s256 * d8
                    end
                end
            end
        end

    elseif fid == 5
        # ζ- boundary, filter along k.  Thread grid: (i, j)
        i_idx = i + NG
        j_idx = k_or_j + NG
        if i_idx <= Nx + NG && j_idx <= Ny + NG
            for layer in 1:min(num_layers, 4)
                kc = NG + layer
                @inbounds phi_local = lin_phi[j_idx, kc]
                σ_local = sigma_max * _intf_filter_weight(phi_local)
                s256 = σ_local / FT(256.0)
                for m in 1:Ncons
                    @inbounds begin
                        d8 = U[i_idx,j_idx,kc-4,m] - FT(8.0)*U[i_idx,j_idx,kc-3,m] +
                             FT(28.0)*U[i_idx,j_idx,kc-2,m] - FT(56.0)*U[i_idx,j_idx,kc-1,m] +
                             FT(70.0)*U[i_idx,j_idx,kc,m] -
                             FT(56.0)*U[i_idx,j_idx,kc+1,m] + FT(28.0)*U[i_idx,j_idx,kc+2,m] -
                             FT(8.0)*U[i_idx,j_idx,kc+3,m] + U[i_idx,j_idx,kc+4,m]
                        U[i_idx,j_idx,kc,m] -= s256 * d8
                    end
                end
            end
        end

    elseif fid == 6
        # ζ+ boundary.   Thread grid: (i, j)
        i_idx = i + NG
        j_idx = k_or_j + NG
        if i_idx <= Nx + NG && j_idx <= Ny + NG
            ek = Nz + NG
            for layer in 1:min(num_layers, 4)
                kc = ek - layer + 1
                @inbounds phi_local = lin_phi[j_idx, kc]
                σ_local = sigma_max * _intf_filter_weight(phi_local)
                s256 = σ_local / FT(256.0)
                for m in 1:Ncons
                    @inbounds begin
                        d8 = U[i_idx,j_idx,kc-4,m] - FT(8.0)*U[i_idx,j_idx,kc-3,m] +
                             FT(28.0)*U[i_idx,j_idx,kc-2,m] - FT(56.0)*U[i_idx,j_idx,kc-1,m] +
                             FT(70.0)*U[i_idx,j_idx,kc,m] -
                             FT(56.0)*U[i_idx,j_idx,kc+1,m] + FT(28.0)*U[i_idx,j_idx,kc+2,m] -
                             FT(8.0)*U[i_idx,j_idx,kc+3,m] + U[i_idx,j_idx,kc+4,m]
                        U[i_idx,j_idx,kc,m] -= s256 * d8
                    end
                end
            end
        end
    end

    return nothing
end
