export interface_filter_kernel!

# =============================================================================
#  8th-order explicit interface filter with ADAPTIVE σ
#
#  σ_local = σ_max × (1 - lin_phi)
#
#  Where lin_phi ∈ [0,1] is the local upwind fraction from spectral_warmup:
#    lin_phi ≈ 0  → pure central (low dissipation) → needs MORE filter
#    lin_phi ≈ 0.5 → interblock taper               → needs MODERATE filter
#    lin_phi ≈ 1  → pure upwind (high dissipation)  → needs NO filter
#
#  Stencil: U_new = U[i] - (σ_local/256) × Δ⁸U[i]
#  Δ⁸U = [1, -8, 28, -56, 70, -56, 28, -8, 1] (binomial coefficients)
#
#  Requires NG >= 4.  Width: 9 points (-4 to +4).
# =============================================================================

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
                s256 = (sigma_max * (one(FT) - lin_phi[j_idx, k_idx])) / FT(256.0)
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
                s256 = (sigma_max * (one(FT) - lin_phi[j_idx, k_idx])) / FT(256.0)
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
                s256 = (sigma_max * (one(FT) - lin_phi[jc, k_idx])) / FT(256.0)
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
                s256 = (sigma_max * (one(FT) - lin_phi[jc, k_idx])) / FT(256.0)
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
                s256 = (sigma_max * (one(FT) - lin_phi[j_idx, kc])) / FT(256.0)
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
                s256 = (sigma_max * (one(FT) - lin_phi[j_idx, kc])) / FT(256.0)
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
