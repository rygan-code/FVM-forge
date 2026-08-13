# Range: 1+NG -> N+NG
if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "mhd_units.jl"))
end

if !@isdefined(POSITIVITY_POLICY_LOADED)
    include(joinpath(@__DIR__, "positivity_policy.jl"))
end

# CT cell state accessor. Persistent CT U stores only hydro variables;
# reconstruction and diagnostics still expose the full 9-component MHD state.
@inline function ct_cell_state_component(U, Q, i, j, k, n)
    @static if equation_type == :MHD && ct_mode
        if n <= Nhydro
            @inbounds return U[i, j, k, n]
        elseif n == UBX
            @inbounds return Q[i, j, k, QBX]
        elseif n == UBY
            @inbounds return Q[i, j, k, QBY]
        elseif n == UBZ
            @inbounds return Q[i, j, k, QBZ]
        else
            return zero(FT)
        end
    else
        @inbounds return U[i, j, k, n]
    end
end

@inline function ct_cell_magnetic(U, Q, i, j, k)
    return SVector{3,FT}(
        ct_cell_state_component(U, Q, i, j, k, UBX),
        ct_cell_state_component(U, Q, i, j, k, UBY),
        ct_cell_state_component(U, Q, i, j, k, UBZ),
    )
end

# CT + isothermal MHD has no thermodynamic energy equation. U[5] is only a
# finite compatibility carrier, while rho and momentum are still evolved by
# the hydro flux. During a CT stage the authoritative face-B update is
# intentionally delayed, so repair the hydro part here using the current
# cell-centered Q[B]. This branch is compile-time excluded from compressible
# and GLM paths.
@inline function ct_repair_isothermal_hydro_state!(U, Q, i, j, k)
    @static if equation_type == :MHD && ct_mode && isothermal_mhd
        @inbounds begin
            rho_raw = U[i, j, k, 1]
            mx_raw = U[i, j, k, 2]
            my_raw = U[i, j, k, 3]
            mz_raw = U[i, j, k, 4]
            energy_raw = U[i, j, k, 5]
        end

        thermal_coefficient = Rg * isothermal_temperature
        required_rho = density_floor
        if isfinite(thermal_coefficient) && thermal_coefficient > zero(FT) &&
           isfinite(pressure_floor)
            required_rho = max(
                required_rho,
                pressure_floor / thermal_coefficient,
            )
        end

        rho_valid = isfinite(rho_raw) && rho_raw > zero(FT)
        momentum_valid = isfinite(mx_raw) && isfinite(my_raw) &&
                         isfinite(mz_raw)
        velocity_valid = false
        u = zero(FT)
        v = zero(FT)
        w = zero(FT)
        if rho_valid && momentum_valid
            inv_rho = inv(rho_raw)
            u = mx_raw * inv_rho
            v = my_raw * inv_rho
            w = mz_raw * inv_rho
            velocity_valid = isfinite(u) && isfinite(v) && isfinite(w)
        end
        if !velocity_valid
            u = zero(FT)
            v = zero(FT)
            w = zero(FT)
        end

        needs_repair = !(rho_valid && rho_raw >= required_rho) ||
                       !momentum_valid || !velocity_valid ||
                       !isfinite(energy_raw)
        needs_repair || return nothing

        rho = max(rho_raw, required_rho)
        pressure = rho * thermal_coefficient

        @inbounds begin
            bx = Q[i, j, k, QBX]
            by = Q[i, j, k, QBY]
            bz = Q[i, j, k, QBZ]
        end
        magnetic_energy = if isfinite(bx) && isfinite(by) && isfinite(bz)
            FT(0.5) * INV_MU0_SI * (bx*bx + by*by + bz*bz)
        else
            zero(FT)
        end
        kinetic = FT(0.5) * rho * (u*u + v*v + w*w)
        carrier = pressure + kinetic + magnetic_energy

        @inbounds begin
            U[i, j, k, 1] = rho
            U[i, j, k, 2] = rho * u
            U[i, j, k, 3] = rho * v
            U[i, j, k, 4] = rho * w
            U[i, j, k, 5] = isfinite(carrier) ? carrier : pressure

            Q[i, j, k, 1] = rho
            Q[i, j, k, 2] = u
            Q[i, j, k, 3] = v
            Q[i, j, k, 4] = w
            Q[i, j, k, 5] = pressure
            Q[i, j, k, 6] = isothermal_temperature
            Q[i, j, k, QPSI] = zero(FT)
        end
    end
    return nothing
end

function c2Prim(U, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    

    # MHD mode: U = (ρ, ρu, ρv, ρw, ρE, Bx, By, Bz, ψ) → Q = (ρ, u, v, w, p, T, Bx, By, Bz, ψ)
    if equation_type == :MHD
        @static if ct_mode && isothermal_mhd
            ct_repair_isothermal_hydro_state!(U, Q, i, j, k)
        end
        @inbounds ρ = ifelse(
            ct_mode && isothermal_mhd,
            max(U[i, j, k, 1], density_floor),
            max(U[i, j, k, 1], eps(FT)),
        )
        ρinv = one(FT) / ρ
        @inbounds u = U[i, j, k, 2] * ρinv
        @inbounds v = U[i, j, k, 3] * ρinv
        @inbounds w = U[i, j, k, 4] * ρinv
        @static if ct_mode
            @inbounds Bx = Q[i, j, k, QBX]
            @inbounds By = Q[i, j, k, QBY]
            @inbounds Bz = Q[i, j, k, QBZ]
        else
            @inbounds Bx = U[i, j, k, UBX]
            @inbounds By = U[i, j, k, UBY]
            @inbounds Bz = U[i, j, k, UBZ]
        end
        B2 = Bx*Bx + By*By + Bz*Bz
        @inbounds ei = max(
            U[i, j, k, 5] - FT(0.5)*ρ*(u*u + v*v + w*w) -
            FT(0.5)*B2*INV_MU0_SI, eps(FT),
        )
        p = isothermal_mhd ? ρ * Rg * isothermal_temperature :
            (γ - one(FT)) * ei
        T = isothermal_mhd ? isothermal_temperature : p / (ρ * Rg)
        @inbounds Q[i,j,k,1] = ρ;  Q[i,j,k,2] = u;  Q[i,j,k,3] = v;  Q[i,j,k,4] = w
        @inbounds Q[i,j,k,5] = p;  Q[i,j,k,6] = T
        @inbounds Q[i,j,k,7] = Bx; Q[i,j,k,8] = By; Q[i,j,k,9] = Bz
        @static if ct_mode
            @inbounds Q[i,j,k,QPSI] = zero(FT)
        else
            @inbounds Q[i,j,k,QPSI] = U[i, j, k, UPSI]
        end
        return
    end

    # correction
    @inbounds ρ = max(U[i, j, k, 1], eps(FT))
    @inbounds ρinv = one(FT)/ρ 

    @inbounds u = U[i, j, k, 2]*ρinv # U
    @inbounds v = U[i, j, k, 3]*ρinv # V
    @inbounds w = U[i, j, k, 4]*ρinv # W
    @inbounds ei = max((U[i, j, k, 5] - FT(0.5)*ρ*(u^2 + v^2 + w^2)), eps(FT))

    p::FT = (γ-one(FT)) * ei
    
    # Positivity clipping
    ρ_min = FT(1.0e-5)
    p_min = FT(1.0e-5)
    ρ = max(ρ, ρ_min)
    p = max(p, p_min)
    
    T::FT = p/(ρ*Rg)
    T = max(T, eps(FT))
    p = ρ * Rg * T  # recompute p from clamped T for consistency

    @inbounds Q[i, j, k, 1] = ρ
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = w
    @inbounds Q[i, j, k, 5] = p
    @inbounds Q[i, j, k, 6] = T
    return
end

# Covers full padded domain
function c2Prim_global(U, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG || i < 1 || j < 1 || k < 1
        return
    end

    

    # MHD mode
    if equation_type == :MHD
        @static if ct_mode && isothermal_mhd
            ct_repair_isothermal_hydro_state!(U, Q, i, j, k)
        end
        @inbounds ρ = ifelse(
            ct_mode && isothermal_mhd,
            max(U[i, j, k, 1], density_floor),
            max(U[i, j, k, 1], eps(FT)),
        )
        ρinv = one(FT) / ρ
        @inbounds u = U[i, j, k, 2] * ρinv
        @inbounds v = U[i, j, k, 3] * ρinv
        @inbounds w = U[i, j, k, 4] * ρinv
        @static if ct_mode
            @inbounds Bx = Q[i, j, k, QBX]
            @inbounds By = Q[i, j, k, QBY]
            @inbounds Bz = Q[i, j, k, QBZ]
        else
            @inbounds Bx = U[i, j, k, UBX]
            @inbounds By = U[i, j, k, UBY]
            @inbounds Bz = U[i, j, k, UBZ]
        end

        B2 = Bx*Bx + By*By + Bz*Bz
        @inbounds ei = max(
            U[i, j, k, 5] - FT(0.5)*ρ*(u*u + v*v + w*w) -
            FT(0.5)*B2*INV_MU0_SI, eps(FT),
        )
        p = isothermal_mhd ? ρ * Rg * isothermal_temperature :
            (γ - one(FT)) * ei
        T = isothermal_mhd ? isothermal_temperature : p / (ρ * Rg)
        @inbounds Q[i,j,k,1] = ρ;  Q[i,j,k,2] = u;  Q[i,j,k,3] = v;  Q[i,j,k,4] = w
        @inbounds Q[i,j,k,5] = p;  Q[i,j,k,6] = T
        @inbounds Q[i,j,k,7] = Bx; Q[i,j,k,8] = By; Q[i,j,k,9] = Bz
        @static if ct_mode
            @inbounds Q[i,j,k,QPSI] = zero(FT)
        else
            @inbounds Q[i,j,k,QPSI] = U[i, j, k, UPSI]
        end
        return
    end

    @inbounds ρ = max(U[i, j, k, 1], eps(FT))
    @inbounds ρinv = one(FT)/ρ 

    @inbounds u = U[i, j, k, 2]*ρinv 
    @inbounds v = U[i, j, k, 3]*ρinv 
    @inbounds w = U[i, j, k, 4]*ρinv 
    @inbounds ei = max((U[i, j, k, 5] - FT(0.5)*ρ*(u^2 + v^2 + w^2)), eps(FT))

    p::FT = (γ-one(FT)) * ei
    
    # Positivity clipping
    ρ_min = FT(1.0e-5)
    p_min = FT(1.0e-5)
    ρ = max(ρ, ρ_min)
    p = max(p, p_min)
    
    T::FT = p/(ρ*Rg)
    T = max(T, eps(FT))
    p = ρ * Rg * T  # recompute p from clamped T for consistency

    @inbounds Q[i, j, k, 1] = ρ
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = w
    @inbounds Q[i, j, k, 5] = p
    @inbounds Q[i, j, k, 6] = T
    return
end

# Range: 1+NG -> N+NG
function prim2c(U, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    

    # MHD mode: Q = (ρ, u, v, w, p, T, Bx, By, Bz, ψ) → U = (ρ, ρu, ρv, ρw, ρE, Bx, By, Bz, ψ)
    if equation_type == :MHD
        @inbounds ρ = Q[i, j, k, 1]
        @inbounds u = Q[i, j, k, 2]; @inbounds v = Q[i, j, k, 3]; @inbounds w = Q[i, j, k, 4]
        @inbounds p = Q[i, j, k, 5]
        @inbounds Bx = Q[i, j, k, 7]; @inbounds By = Q[i, j, k, 8]; @inbounds Bz = Q[i, j, k, 9]
        B2 = Bx*Bx + By*By + Bz*Bz
        @inbounds U[i,j,k,1] = ρ
        @inbounds U[i,j,k,2] = ρ * u
        @inbounds U[i,j,k,3] = ρ * v
        @inbounds U[i,j,k,4] = ρ * w
        @inbounds U[i,j,k,5] = mhd_energy_density_from_primitive(
            ρ, u, v, w, p, Bx, By, Bz,
        )
        @static if !ct_mode
            @inbounds U[i,j,k,UBX] = Bx
        end
        @static if !ct_mode
            @inbounds U[i,j,k,UBY] = By
        end
        @static if !ct_mode
            @inbounds U[i,j,k,UBZ] = Bz
        end
        @static if !ct_mode
            @inbounds U[i,j,k,UPSI] = Q[i, j, k, QPSI]
        end
        return
    end

    @inbounds ρ = Q[i, j, k, 1]
    @inbounds u = Q[i, j, k, 2]
    @inbounds v = Q[i, j, k, 3]
    @inbounds w = Q[i, j, k, 4]
    @inbounds U[i, j, k, 1] = ρ
    @inbounds U[i, j, k, 2] = u * ρ
    @inbounds U[i, j, k, 3] = v * ρ
    @inbounds U[i, j, k, 4] = w * ρ
    @inbounds U[i, j, k, 5] = isothermal_mhd ?
        mhd_energy_density_from_primitive(
            ρ, u, v, w, Q[i, j, k, 5], zero(FT), zero(FT), zero(FT),
        ) :
        Q[i, j, k, 5]/(γ-1) + FT(0.5) * ρ * (u^2 + v^2 + w^2)
    return
end

# Range: 1+NG -> N+NG
function linComb(U, Un, NV, a::FT, b::FT, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    for n = 1:NV
        @inbounds U[i, j, k, n] = U[i, j, k, n] * a + Un[i, j, k, n] * b
    end
    return
end

# Range: 1+NG -> N+NG
# Uses contravariant velocities (spectral radius) for correct CFL on curvilinear grids.
# λ_ξ = (|U_contra| + c) * Area / Vol  for each direction
function compute_dt(dt, Q, J, S1, S2, S3,
                    nxi, nyi, nzi,   # ξ-face normals
                    nxj, nyj, nzj,   # η-face normals
                    nxk, nyk, nzk,   # ζ-face normals
                    nxp, nyp, nzp, ch_glm)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    @inbounds u = Q[i, j, k, 2]
    @inbounds v = Q[i, j, k, 3]
    @inbounds w = Q[i, j, k, 4]

    @inbounds Vol = one(FT) / J[i, j, k]

    # Wave speed: compressible uses sound speed, MHD uses fast magnetosonic
    if equation_type == :MHD
        @inbounds ρ_val = Q[i, j, k, 1]
        @inbounds T_val = Q[i, j, k, 6]
        c2 = isothermal_mhd ? Rg * T_val : γ * Rg * T_val  # sound speed squared
        @inbounds Bx_v = Q[i, j, k, 7]; @inbounds By_v = Q[i, j, k, 8]; @inbounds Bz_v = Q[i, j, k, 9]
        va2 = (Bx_v*Bx_v + By_v*By_v + Bz_v*Bz_v) * INV_MU0_SI /
            (ρ_val + FT(1.0e-30))  # Alfvén speed²
        # Fast magnetosonic speed (isotropic estimate for CFL)
        c = sqrt(c2 + va2)
    else
        @inbounds T = Q[i, j, k, 6]
        c = sqrt(γ*Rg*T)
    end

    # ξ-direction: average face normals from i and i+1 faces
    @inbounds nx_i = FT(0.5) * (nxi[i, j, k] + nxi[i+1, j, k])
    @inbounds ny_i = FT(0.5) * (nyi[i, j, k] + nyi[i+1, j, k])
    @inbounds nz_i = FT(0.5) * (nzi[i, j, k] + nzi[i+1, j, k])
    Ucon_i = abs(u*nx_i + v*ny_i + w*nz_i)
    @inbounds Ai = FT(0.5) * (S1[i, j, k] + S1[i+1, j, k])  # average face area (fallback: use S1[i,j,k])

    # η-direction
    @inbounds nx_j = FT(0.5) * (nxj[i, j, k] + nxj[i, j+1, k])
    @inbounds ny_j = FT(0.5) * (nyj[i, j, k] + nyj[i, j+1, k])
    @inbounds nz_j = FT(0.5) * (nzj[i, j, k] + nzj[i, j+1, k])
    Ucon_j = abs(u*nx_j + v*ny_j + w*nz_j)
    @inbounds Aj = FT(0.5) * (S2[i, j, k] + S2[i, j+1, k])

    # ζ-direction
    @inbounds nx_k = FT(0.5) * (nxk[i, j, k] + nxk[i, j, k+1])
    @inbounds ny_k = FT(0.5) * (nyk[i, j, k] + nyk[i, j, k+1])
    @inbounds nz_k = FT(0.5) * (nzk[i, j, k] + nzk[i, j, k+1])
    Ucon_k = abs(u*nx_k + v*ny_k + w*nz_k)
    @inbounds Ak = FT(0.5) * (S3[i, j, k] + S3[i, j, k+1])

    # Spectral radii: λ = (|U_contra| + c) * Area
    @static if equation_type == :MHD && !ct_mode
        λ_ξ = max(Ucon_i + c, ch_glm) * Ai
        λ_η = max(Ucon_j + c, ch_glm) * Aj
        λ_ζ = max(Ucon_k + c, ch_glm) * Ak
    else
        λ_ξ = (Ucon_i + c) * Ai
        λ_η = (Ucon_j + c) * Aj
        λ_ζ = (Ucon_k + c) * Ak
    end

    # dt = CFL * Vol / (λ_ξ + λ_η + λ_ζ)   — sum formulation (more conservative, standard)
    dt_conv = Vol / (λ_ξ + λ_η + λ_ζ + FT(1.0e-30))

    # Viscous/Resistive stability limit
    if viscous || (equation_type == :MHD && resistive)
        dx = Vol / (Ai + FT(1.0e-30))
        dy = Vol / (Aj + FT(1.0e-30))
        dz = Vol / (Ak + FT(1.0e-30))
        inv_dx2 = one(FT)/(dx*dx)
        inv_dy2 = one(FT)/(dy*dy)
        inv_dz2 = one(FT)/(dz*dz)
        inv_d2_sum = inv_dx2 + inv_dy2 + inv_dz2

        dt_diff = FT(1.0e10)

        if viscous
            @inbounds rho = Q[i, j, k, 1]
            T_local = (equation_type == :MHD) ? T_val : T
            mu = get_viscosity(T_local)
            nu_momentum = mu / (rho + FT(1.0e-30))
            alpha_thermal = nu_momentum / (Pr + FT(1.0e-30))
            nu_eff = max(nu_momentum, alpha_thermal)
            dt_diff_hydro = FT(0.5) / (nu_eff * inv_d2_sum + FT(1.0e-30))
            dt_diff = min(dt_diff, dt_diff_hydro)
        end

        @static if equation_type == :MHD
            if resistive && ct_resistive_main_explicit
                dt_diff_res = FT(0.5) / (η_mhd * inv_d2_sum + FT(1.0e-30))
                dt_diff = min(dt_diff, dt_diff_res)
            end
        end

        @inbounds dt[i, j, k] = min(dt_conv, dt_diff) * CFL
    else
        @inbounds dt[i, j, k] = dt_conv * CFL
    end
    return
end

function pre_x(Q, sc, rth, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    local p_idx::Int32 = Int32(5)
    @inbounds p1 = Q[i-2, j, k, p_idx]
    @inbounds p2 = Q[i-1, j, k, p_idx]
    @inbounds p3 = Q[i,   j, k, p_idx]
    @inbounds p4 = Q[i+1, j, k, p_idx]
    @inbounds p5 = Q[i+2, j, k, p_idx]

    Δp0 = FT(0.25) * (-p4+2p3-p2)
    Δp1 = FT(0.25) * (-p5+2p4-p3)
    Δp2 = FT(0.25) * (-p3+2p2-p1)
    ri = FT(0.5) * ((Δp0-Δp1)^2+(Δp0-Δp2)^2)/p3^2+FT(1e-16)
    @inbounds sc[i, j, k] = FT(0.5)*(FT(1.e0)-rth/ri+abs(FT(1.e0)-rth/ri))
    return
end

function pre_y(Q, sc, rth, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    local p_idx::Int32 = Int32(5)
    @inbounds p1 = Q[i, j-2, k, p_idx]
    @inbounds p2 = Q[i, j-1, k, p_idx]
    @inbounds p3 = Q[i, j,   k, p_idx]
    @inbounds p4 = Q[i, j+1, k, p_idx]
    @inbounds p5 = Q[i, j+2, k, p_idx]

    Δp0 = FT(0.25) * (-p4+2p3-p2)
    Δp1 = FT(0.25) * (-p5+2p4-p3)
    Δp2 = FT(0.25) * (-p3+2p2-p1)
    ri = FT(0.5) * ((Δp0-Δp1)^2+(Δp0-Δp2)^2)/p3^2+FT(1e-16)
    @inbounds sc[i, j, k] = FT(0.5)*(FT(1.e0)-rth/ri+abs(FT(1.e0)-rth/ri))
    return
end

function pre_z(Q, sc, rth, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    local p_idx::Int32 = Int32(5)
    @inbounds p1 = Q[i, j, k-2, p_idx]
    @inbounds p2 = Q[i, j, k-1, p_idx]
    @inbounds p3 = Q[i, j, k,   p_idx]
    @inbounds p4 = Q[i, j, k+1, p_idx]
    @inbounds p5 = Q[i, j, k+2, p_idx]

    Δp0 = FT(0.25) * (-p4+2p3-p2)
    Δp1 = FT(0.25) * (-p5+2p4-p3)
    Δp2 = FT(0.25) * (-p3+2p2-p1)
    ri = FT(0.5) * ((Δp0-Δp1)^2+(Δp0-Δp2)^2)/p3^2+FT(1e-16)
    @inbounds sc[i, j, k] = FT(0.5)*(FT(1.e0)-rth/ri+abs(FT(1.e0)-rth/ri))
    return
end

function filter_x(U, Un, sc, s0, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    c1::FT = -FT(0.210383e0)
    c2::FT = FT(0.039617e0)

    @inbounds sc1 = FT(0.5)*(sc[i, j, k]+sc[i+1, j, k])
    @inbounds sc2 = FT(0.5)*(sc[i, j, k]+sc[i-1, j, k])

    for n = 1:Ncell_cons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (sc1 * (c1 * (Un[i+1, j, k, n] - Un[i, j, k, n]) +
                                                      c2 * (Un[i+2, j, k, n] - Un[i-1, j, k, n])) -
                                               sc2 * (c1 * (Un[i, j, k, n] - Un[i-1, j, k, n]) +
                                                      c2 * (Un[i+1, j, k, n] - Un[i-2, j, k, n])))
    end
    return
end

function filter_y(U, Un, sc, s0, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    c1::FT = -FT(0.210383e0)
    c2::FT = FT(0.039617e0)

    @inbounds sc1 = FT(0.5)*(sc[i, j, k]+sc[i, j+1, k])
    @inbounds sc2 = FT(0.5)*(sc[i, j, k]+sc[i, j-1, k])

    for n = 1:Ncell_cons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (sc1 * (c1 * (Un[i, j+1, k, n] - Un[i, j, k, n]) +
                                                      c2 * (Un[i, j+2, k, n] - Un[i, j-1, k, n])) -
                                               sc2 * (c1 * (Un[i, j, k, n] - Un[i, j-1, k, n]) +
                                                      c2 * (Un[i, j+1, k, n] - Un[i, j-2, k, n])))
    end
    return
end

function filter_z(U, Un, sc, s0, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    c1::FT = -FT(0.210383e0)
    c2::FT = FT(0.039617e0)

    @inbounds sc1 = FT(0.5)*(sc[i, j, k]+sc[i, j, k+1])
    @inbounds sc2 = FT(0.5)*(sc[i, j, k]+sc[i, j, k-1])

    for n = 1:Ncell_cons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (sc1 * (c1 * (Un[i, j, k+1, n] - Un[i, j, k, n]) +
                                                      c2 * (Un[i, j, k+2, n] - Un[i, j, k-1, n])) -
                                               sc2 * (c1 * (Un[i, j, k, n] - Un[i, j, k-1, n]) +
                                                      c2 * (Un[i, j, k+1, n] - Un[i, j, k-2, n])))
    end
    return
end

# 8th-order Pirozzoli-style spatial filter kernels
# ilo/ihi (jlo/jhi, klo/khi): filter range bounds — shrunk near physical
# boundaries to avoid reading from non-physical ghost cells.
# At MPI/periodic boundaries the full stencil is safe.

function linearFilter_x(U, Un, s0, nxp, nyp, nzp, ilo, ihi)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > ihi || j > nyp+NG || k > nzp+NG || i < ilo || j < NG+1 || k < NG+1
        return
    end

    d0::FT = FT(0.243527493120e0)
    d1::FT =-FT(0.204788880640e0)
    d2::FT = FT(0.120007591680e0)
    d3::FT =-FT(0.045211119360e0)
    d4::FT = FT(0.008228661760e0)

    for n = 1:Ncell_cons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (d0 * Un[i, j, k, n] +
                                                         d1 * (Un[i-1, j, k, n] + Un[i+1, j, k, n]) +
                                                         d2 * (Un[i-2, j, k, n] + Un[i+2, j, k, n]) +
                                                         d3 * (Un[i-3, j, k, n] + Un[i+3, j, k, n]) +
                                                         d4 * (Un[i-4, j, k, n] + Un[i+4, j, k, n]))
    end
    return
end

function linearFilter_y(U, Un, s0, nxp, nyp, nzp, jlo, jhi)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > jhi || k > nzp+NG || i < NG+1 || j < jlo || k < NG+1
        return
    end

    d0::FT = FT(0.243527493120e0)
    d1::FT =-FT(0.204788880640e0)
    d2::FT = FT(0.120007591680e0)
    d3::FT =-FT(0.045211119360e0)
    d4::FT = FT(0.008228661760e0)

    for n = 1:Ncell_cons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (d0 * Un[i, j, k, n] +
                                                         d1 * (Un[i, j-1, k, n] + Un[i, j+1, k, n]) +
                                                         d2 * (Un[i, j-2, k, n] + Un[i, j+2, k, n]) +
                                                         d3 * (Un[i, j-3, k, n] + Un[i, j+3, k, n]) +
                                                         d4 * (Un[i, j-4, k, n] + Un[i, j+4, k, n]))
    end
    return
end

function linearFilter_z(U, Un, s0, nxp, nyp, nzp, klo, khi)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > khi || i < NG+1 || j < NG+1 || k < klo
        return
    end

    d0::FT = FT(0.243527493120e0)
    d1::FT =-FT(0.204788880640e0)
    d2::FT = FT(0.120007591680e0)
    d3::FT =-FT(0.045211119360e0)
    d4::FT = FT(0.008228661760e0)

    for n = 1:Ncell_cons
        @inbounds U[i, j, k, n] = Un[i, j, k, n] - s0 * (d0 * Un[i, j, k, n] +
                                                         d1 * (Un[i, j, k-1, n] + Un[i, j, k+1, n]) +
                                                         d2 * (Un[i, j, k-2, n] + Un[i, j, k+2, n]) +
                                                         d3 * (Un[i, j, k-3, n] + Un[i, j, k+3, n]) +
                                                         d4 * (Un[i, j, k-4, n] + Un[i, j, k+4, n]))
    end
    return
end

function positivity_clipping(Q, U, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    

    ρ_min = FT(1.0e-5)
    p_min = FT(1.0e-5)

    # MHD mode: clip ρ and p, leave B and ψ unconstrained
    if equation_type == :MHD
        @inbounds begin
            ρ = Q[i, j, k, 1]
            p = Q[i, j, k, 5]
            if ρ < ρ_min || p < p_min
                ρ = max(ρ, ρ_min)
                p = max(p, p_min)
                Q[i, j, k, 1] = ρ
                Q[i, j, k, 5] = p
                Q[i, j, k, 6] = p / (ρ * Rg)
                u = Q[i, j, k, 2]; v = Q[i, j, k, 3]; w = Q[i, j, k, 4]
                Bx = Q[i, j, k, 7]; By = Q[i, j, k, 8]; Bz = Q[i, j, k, 9]
                B2 = Bx*Bx + By*By + Bz*Bz
                U[i,j,k,1] = ρ
                U[i,j,k,2] = ρ * u; U[i,j,k,3] = ρ * v; U[i,j,k,4] = ρ * w
                U[i,j,k,5] = mhd_energy_density_from_primitive(
                    ρ, u, v, w, p, Bx, By, Bz,
                )
            end
        end
        return
    end

    @inbounds begin
        ρ = Q[i, j, k, 1]
        p = Q[i, j, k, 5]
        T_val = Q[i, j, k, 6]
        
        # Positivity check
        need_fix = (ρ < ρ_min) || (p < p_min) || (T_val < eps(FT))
        if need_fix
            ρ = max(ρ, ρ_min)
            T_val = max(T_val, eps(FT))
            p = max(ρ * Rg * T_val, p_min)
            
            Q[i, j, k, 1] = ρ
            Q[i, j, k, 5] = p
            Q[i, j, k, 6] = T_val
            
            u = Q[i, j, k, 2]
            v = Q[i, j, k, 3]
            w = Q[i, j, k, 4]
            
            U[i, j, k, 1] = ρ
            U[i, j, k, 2] = ρ * u
            U[i, j, k, 3] = ρ * v
            U[i, j, k, 4] = ρ * w
            U[i, j, k, 5] = p / (γ-one(FT)) + FT(0.5) * ρ * (u*u + v*v + w*w)
        end
    end
    return
end

# ── Fused kernel: linComb + c2Prim + positivity_clipping ──
# CT defers primitive recovery until face B reaches the same RK stage.
# Other equation paths retain the fused recovery used by RK substeps 2 and 3.
function linComb_clip_prim(
    U, Un, Q, J, repair_count, conservation_delta,
    NV, a::FT, b::FT, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    # Step 1: RK linear combination  U = Un + a*(U - Un)
    # Equivalent to U = a*U + (1-a)*Un, but guaranteed exact when U==Un
    # (avoids systematic drift from FT(2/3)+FT(1/3) ≠ 1.0)
    for n = 1:NV
        @inbounds U[i, j, k, n] = Un[i, j, k, n] + a * (U[i, j, k, n] - Un[i, j, k, n])
    end

    @static if !(equation_type == :MHD && ct_mode)
        _finalize_structured_thermodynamics!(
            U,Q,J,repair_count,conservation_delta,i,j,k)
    end
    return
end

# ── Ghost-only c2Prim: only operates on ghost cells, skipping interior ──
# Interior cells already have correct Q from positivity_clipping / linComb_clip_prim.
# Ghost cells have U updated by MPI exchange but Q not yet refreshed.
function c2Prim_ghost(U, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG || i < 1 || j < 1 || k < 1
        return
    end

    # Skip deep interior cells (leave an 4-cell outer shell to catch INTERFACE SMOOTHING updates to U)
    if i > NG+4 && i <= nxp+NG-4 && j > NG+4 && j <= nyp+NG-4 && k > NG+4 && k <= nzp+NG-4
        return
    end

    

    # MHD mode
    if equation_type == :MHD
        @static if ct_mode && isothermal_mhd
            ct_repair_isothermal_hydro_state!(U, Q, i, j, k)
        end
        @static if strict_ct_positivity && ct_mode && splitMethodID == 4
            @inbounds state = SVector{8,FT}(
                (
                    U[i, j, k, 1], U[i, j, k, 2], U[i, j, k, 3],
                    U[i, j, k, 4], U[i, j, k, 5],
                    Q[i, j, k, QBX], Q[i, j, k, QBY], Q[i, j, k, QBZ],
                ),
            )
            ρ, kinetic, magnetic, ei, p = mhd_raw_thermo(state, γ)
            ρinv = one(FT) / ρ
            u = state[2] * ρinv
            v = state[3] * ρinv
            w = state[4] * ρinv
            T = isothermal_mhd ? isothermal_temperature : p / (ρ * Rg)
            @inbounds Q[i,j,k,1] = ρ;  Q[i,j,k,2] = u;  Q[i,j,k,3] = v;  Q[i,j,k,4] = w
            @inbounds Q[i,j,k,5] = p;  Q[i,j,k,6] = T
            @inbounds Q[i,j,k,7] = state[6]; Q[i,j,k,8] = state[7]; Q[i,j,k,9] = state[8]
            @inbounds Q[i,j,k,QPSI] = zero(FT)
            return
        end

        @inbounds ρ = ifelse(
            ct_mode && isothermal_mhd,
            max(U[i, j, k, 1], density_floor),
            max(U[i, j, k, 1], eps(FT)),
        )
        ρinv = one(FT) / ρ
        @inbounds u = U[i, j, k, 2] * ρinv
        @inbounds v = U[i, j, k, 3] * ρinv
        @inbounds w = U[i, j, k, 4] * ρinv
        @static if ct_mode
            @inbounds Bx = Q[i,j,k,QBX]; @inbounds By = Q[i,j,k,QBY]; @inbounds Bz = Q[i,j,k,QBZ]
        else
            @inbounds Bx = U[i,j,k,UBX]; @inbounds By = U[i,j,k,UBY]; @inbounds Bz = U[i,j,k,UBZ]
        end

        B2 = Bx*Bx + By*By + Bz*Bz
        @inbounds ei = max(
            U[i, j, k, 5] - FT(0.5)*ρ*(u*u + v*v + w*w) -
            FT(0.5)*B2*INV_MU0_SI, eps(FT),
        )
        p = isothermal_mhd ? ρ * Rg * isothermal_temperature :
            (γ - one(FT)) * ei
        T = isothermal_mhd ? isothermal_temperature : p / (ρ * Rg)
        @inbounds Q[i,j,k,1] = ρ;  Q[i,j,k,2] = u;  Q[i,j,k,3] = v;  Q[i,j,k,4] = w
        @inbounds Q[i,j,k,5] = p;  Q[i,j,k,6] = T
        @inbounds Q[i,j,k,7] = Bx; Q[i,j,k,8] = By; Q[i,j,k,9] = Bz
        @static if ct_mode
            @inbounds Q[i,j,k,QPSI] = zero(FT)
        else
            @inbounds Q[i,j,k,QPSI] = U[i, j, k, UPSI]
        end
        return
    end

    @inbounds ρ = max(U[i, j, k, 1], eps(FT))
    ρinv = one(FT) / ρ
    @inbounds u = U[i, j, k, 2] * ρinv
    @inbounds v = U[i, j, k, 3] * ρinv
    @inbounds w = U[i, j, k, 4] * ρinv
    @inbounds ei = max(U[i, j, k, 5] - FT(0.5)*ρ*(u*u + v*v + w*w), eps(FT))
    p = (γ - one(FT)) * ei

    ρ = max(ρ, FT(1.0e-5))
    p = max(p, FT(1.0e-5))
    T = p / (ρ * Rg)
    T = max(T, eps(FT))
    p = ρ * Rg * T

    @inbounds Q[i, j, k, 1] = ρ
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = w
    @inbounds Q[i, j, k, 5] = p
    @inbounds Q[i, j, k, 6] = T
    # NOTE: Do NOT write back to U — ghost U must remain exactly as received
    # from MPI exchange to avoid systematic floating-point energy drift.
    return
end

@inline _source_timestep(dt::Number, i, j, k) = dt
@inline _source_timestep(dt, i, j, k) = @inbounds dt[i, j, k]

function add_source_kernel!(U, dU_forced, dt, Vol, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    
    # Range is 1:nxp in local indices
    # Pad to Global-in-rank (NG+1)
    ii, jj, kk = i+NG, j+NG, k+NG
    
    # Apply source term: U += S * dt
    fact = _source_timestep(dt, ii, jj, kk)
    for n = 1:Ncell_cons
        @inbounds U[ii, jj, kk, n] += dU_forced[i, j, k, n] * fact
    end
    return
end

function accumulate_avg_kernel!(Q_avg, Q, U, count::Int32, n_prim::Int, nxp::Int, nyp::Int, nzp::Int, NG_val::Int, do_favre::Bool)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+2*NG_val || j > nyp+2*NG_val || k > nzp+2*NG_val || i < 1 || j < 1 || k < 1
        return
    end

    inv_c = 1.0f0 / Float32(count)
    weight_old = Float32(count - 1) * inv_c

    for n = 1:n_prim
        @inbounds val = Q[i, j, k, n]
        
        if do_favre
            if n == 2 || n == 3 || n == 4
                @inbounds val = U[i, j, k, n]
            end
        end

        @inbounds Q_avg[i, j, k, n] = Q_avg[i, j, k, n] * weight_old + val * inv_c
    end
    return
end
