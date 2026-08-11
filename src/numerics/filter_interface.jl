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
                @inbounds phi_local = lin_phi[ic]
                sigma_local = sigma_max * _intf_filter_weight(phi_local)
                s256 = sigma_local / FT(256.0)
                s256 = sigma_max / FT(256.0)  # fixed σ for ξ direction
        for m in 1:Ncell_cons
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
                @inbounds phi_local = lin_phi[ic]
                sigma_local = sigma_max * _intf_filter_weight(phi_local)
                s256 = sigma_local / FT(256.0)
                s256 = sigma_max / FT(256.0)  # fixed σ for ξ direction
        for m in 1:Ncell_cons
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
        for m in 1:Ncell_cons
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
        for m in 1:Ncell_cons
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
        for m in 1:Ncell_cons
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
        for m in 1:Ncell_cons
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

# CT keeps face-B authoritative. The interface filter therefore limits only
# the conservative correction while holding the LSQ2 cell magnetic field
# fixed. LSQ2 is the same low-order anchor used by FOFC and ghost derivation,
# so a post-FOFC filter pass cannot drive E_tot below K+M unnoticed.
@inline function _ct_interface_filter_theta(
    old_hydro::SVector{6,T}, candidate_hydro::SVector{6,T},
    magnetic::SVector{3,T}, gamma::T,
    minimum_density::T, minimum_pressure::T,
) where {T}
    ct_point6_state_is_admissible(
        candidate_hydro, magnetic, gamma, minimum_density, minimum_pressure,
    ) && return one(T)

    _, _, theta, recoverable = ct_point6_convex_limit(
        candidate_hydro, magnetic, old_hydro, magnetic,
        gamma, minimum_density, minimum_pressure,
    )
    recoverable || return zero(T)
    return clamp(T(0.99) * theta, zero(T), one(T))
end

@inline function _ct_interface_filter_d8(U, i, j, k, m, ::Val{1})
    @inbounds return U[i-4,j,k,m] - FT(8.0)*U[i-3,j,k,m] +
        FT(28.0)*U[i-2,j,k,m] - FT(56.0)*U[i-1,j,k,m] +
        FT(70.0)*U[i,j,k,m] - FT(56.0)*U[i+1,j,k,m] +
        FT(28.0)*U[i+2,j,k,m] - FT(8.0)*U[i+3,j,k,m] +
        U[i+4,j,k,m]
end

@inline function _ct_interface_filter_d8(U, i, j, k, m, ::Val{2})
    @inbounds return U[i,j-4,k,m] - FT(8.0)*U[i,j-3,k,m] +
        FT(28.0)*U[i,j-2,k,m] - FT(56.0)*U[i,j-1,k,m] +
        FT(70.0)*U[i,j,k,m] - FT(56.0)*U[i,j+1,k,m] +
        FT(28.0)*U[i,j+2,k,m] - FT(8.0)*U[i,j+3,k,m] +
        U[i,j+4,k,m]
end


@inline function _ct_interface_filter_d8(U, i, j, k, m, ::Val{3})
    @inbounds return U[i,j,k-4,m] - FT(8.0)*U[i,j,k-3,m] +
        FT(28.0)*U[i,j,k-2,m] - FT(56.0)*U[i,j,k-1,m] +
        FT(70.0)*U[i,j,k,m] - FT(56.0)*U[i,j,k+1,m] +
        FT(28.0)*U[i,j,k+2,m] - FT(8.0)*U[i,j,k+3,m] +
        U[i,j,k+4,m]
end

@inline function _ct_apply_interface_filter_cell!(
    U, magnetic, i, j, k, s256, gamma, minimum_density, minimum_pressure,
    direction,
)
    @inbounds old_hydro = SVector{6,FT}(
        U[i,j,k,1], U[i,j,k,2], U[i,j,k,3],
        U[i,j,k,4], U[i,j,k,5], zero(FT),
    )
    candidate_hydro = SVector{6,FT}(
        ntuple(Val(5)) do variable
            old_hydro[variable] - s256 * _ct_interface_filter_d8(
                U, i, j, k, variable, direction,
            )
        end...,
        zero(FT),
    )
    theta = _ct_interface_filter_theta(
        old_hydro, candidate_hydro, magnetic,
        gamma, minimum_density, minimum_pressure,
    )
    theta == zero(theta) && return nothing
    if theta == one(theta)
        @inbounds for variable in 1:5
            U[i,j,k,variable] = candidate_hydro[variable]
        end
    else
        @inbounds for variable in 1:5
            U[i,j,k,variable] = old_hydro[variable] +
                theta * (candidate_hydro[variable] - old_hydro[variable])
        end
    end
    @inbounds for variable in 6:Ncell_cons
        old_value = U[i,j,k,variable]
        candidate_value = old_value - s256 * _ct_interface_filter_d8(
            U, i, j, k, variable, direction,
        )
        U[i,j,k,variable] = theta == one(theta) ? candidate_value :
            old_value + theta * (candidate_value - old_value)
    end
    return nothing
end

@inline function _ct_interface_filter_magnetic(
    Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    i, j, k, B0x_face, B0y_face, B0z_face,
)
    return _ct_recover_cell_b_from_face_fluxes(
        Bx_face, By_face, Bz_face,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        CT_CELL_B_LSQ2, i, j, k,
        B0x_face, B0y_face, B0z_face,
    )
end

function ct_interface_filter_kernel!(
    U, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    Nx::Int, Ny::Int, Nz::Int, fid::Int, num_layers::Int,
    sigma_max, lin_phi, gamma, minimum_density, minimum_pressure,
    B0x_face=nothing, B0y_face=nothing, B0z_face=nothing,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    k_or_j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if fid == 1 || fid == 2
        j_idx = i + NG
        k_idx = k_or_j + NG
        if j_idx <= Ny + NG && k_idx <= Nz + NG
            endpoint = fid == 1 ? NG + 1 : Nx + NG
            step = fid == 1 ? 1 : -1
            for layer in 1:min(num_layers, 4)
                ic = endpoint + (layer - 1) * step
                s256 = sigma_max / FT(256.0)
                magnetic = _ct_interface_filter_magnetic(
                    Bx_face,By_face,Bz_face,
                    Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,
                    Areak,nxk,nyk,nzk,ic,j_idx,k_idx,
                    B0x_face,B0y_face,B0z_face,
                )
                _ct_apply_interface_filter_cell!(
                    U, magnetic, ic, j_idx, k_idx, s256,
                    gamma, minimum_density, minimum_pressure, Val(1),
                )
            end
        end
    elseif fid == 3 || fid == 4
        i_idx = i + NG
        k_idx = k_or_j + NG
        if i_idx <= Nx + NG && k_idx <= Nz + NG
            endpoint = fid == 3 ? NG + 1 : Ny + NG
            step = fid == 3 ? 1 : -1
            for layer in 1:min(num_layers, 4)
                jc = endpoint + (layer - 1) * step
                @inbounds phi_local = lin_phi[jc,k_idx]
                sigma_local = sigma_max * _intf_filter_weight(phi_local)
                magnetic = _ct_interface_filter_magnetic(
                    Bx_face,By_face,Bz_face,
                    Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,
                    Areak,nxk,nyk,nzk,i_idx,jc,k_idx,
                    B0x_face,B0y_face,B0z_face,
                )
                _ct_apply_interface_filter_cell!(
                    U, magnetic, i_idx, jc, k_idx,
                    sigma_local / FT(256.0),
                    gamma, minimum_density, minimum_pressure, Val(2),
                )
            end
        end
    elseif fid == 5 || fid == 6
        i_idx = i + NG
        j_idx = k_or_j + NG
        if i_idx <= Nx + NG && j_idx <= Ny + NG
            endpoint = fid == 5 ? NG + 1 : Nz + NG
            step = fid == 5 ? 1 : -1
            for layer in 1:min(num_layers, 4)
                kc = endpoint + (layer - 1) * step
                @inbounds phi_local = lin_phi[j_idx,kc]
                sigma_local = sigma_max * _intf_filter_weight(phi_local)
                magnetic = _ct_interface_filter_magnetic(
                    Bx_face,By_face,Bz_face,
                    Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,
                    Areak,nxk,nyk,nzk,i_idx,j_idx,kc,
                    B0x_face,B0y_face,B0z_face,
                )
                _ct_apply_interface_filter_cell!(
                    U, magnetic, i_idx, j_idx, kc,
                    sigma_local / FT(256.0),
                    gamma, minimum_density, minimum_pressure, Val(3),
                )
            end
        end
    end
    return nothing
end
