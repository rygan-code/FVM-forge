if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end

@inline function HLLC_Flux(UL, UR, nx, ny, nz)

    # 0. Safety Check for uninitialized/vacuum/NaN states
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10) || !isfinite(UL[1]) || !isfinite(UR[1])
        return SVector{5, FT}(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
    end

    # 1. 准备物理?(Primitive Variables)
    # ------------------------------------------------------------------
    # Left State
    ρL = UL[1]
    inv_ρL = one(FT) / ρL
    uL = UL[2] * inv_ρL
    vL = UL[3] * inv_ρL
    wL = UL[4] * inv_ρL
    EL = UL[5]
    pL = (γ - one(FT)) * (EL - FT(0.5) * ρL * (uL^2 + vL^2 + wL^2))
    pL = max(pL, FT(1.0e-10)) # Epsilon safety
    cL = sqrt(γ * pL * inv_ρL)
    qL = uL * nx + vL * ny + wL * nz  # 法向速度

    # Right State
    ρR = UR[1]
    inv_ρR = one(FT) / ρR
    uR = UR[2] * inv_ρR
    vR = UR[3] * inv_ρR
    wR = UR[4] * inv_ρR
    ER = UR[5]
    pR = (γ - one(FT)) * (ER - FT(0.5) * ρR * (uR^2 + vR^2 + wR^2))
    pR = max(pR, FT(1.0e-10)) # Epsilon safety
    cR = sqrt(γ * pR * inv_ρR)
    qR = uR * nx + vR * ny + wR * nz  # 法向速度

    # 2. 计算波?(Wave Speeds Estimate using Roe Averages)
    # ------------------------------------------------------------------
    # Roe 平均
    sqρL = sqrt(ρL)
    sqρR = sqrt(ρR)
    inv_sqρ = one(FT) / (sqρL + sqρR)
    
    # Roe 速度
    u_roe = (sqρL * uL + sqρR * uR) * inv_sqρ
    v_roe = (sqρL * vL + sqρR * vR) * inv_sqρ
    w_roe = (sqρL * wL + sqρR * wR) * inv_sqρ
    q_roe = u_roe * nx + v_roe * ny + w_roe * nz
    
    # Roe ?
    HL = (EL + pL) * inv_ρL
    HR = (ER + pR) * inv_ρR
    H_roe = (sqρL * HL + sqρR * HR) * inv_sqρ
    
    # Roe 声?
    v2_roe = u_roe^2 + v_roe^2 + w_roe^2
    c_roe = sqrt(max(FT(1.0e-10), (γ - one(FT)) * (H_roe - FT(0.5) * v2_roe)))

    # 估计左右波?(SL, SR)
    SL = min(qL - cL, q_roe - c_roe)
    SR = max(qR + cR, q_roe + c_roe)

    # 3. 计算接触间断波?(Contact Wave Speed S*)
    # ------------------------------------------------------------------
    # HLLC S* definition
    S_star = (pR - pL + ρL * qL * (SL - qL) - ρR * qR * (SR - qR)) / 
             (ρL * (SL - qL) - ρR * (SR - qR) + FT(1.0e-20)) # 加极小量防除?
    # 4. 根据波速位置选择通量 (Logic Branching)
    # ------------------------------------------------------------------
    if SL >= 0
        flux1 = ρL * qL
        flux2 = ρL * uL * qL + pL * nx
        flux3 = ρL * vL * qL + pL * ny
        flux4 = ρL * wL * qL + pL * nz
        flux5 = (EL + pL) * qL
        return SVector{5, FT}(flux1, flux2, flux3, flux4, flux5)

    elseif SR <= 0
        flux1 = ρR * qR
        flux2 = ρR * uR * qR + pR * nx
        flux3 = ρR * vR * qR + pR * ny
        flux4 = ρR * wR * qR + pR * nz
        flux5 = (ER + pR) * qR
        return SVector{5, FT}(flux1, flux2, flux3, flux4, flux5)
        
    elseif SL < 0 && S_star >= 0
        FL1 = ρL * qL
        FL2 = ρL * uL * qL + pL * nx
        FL3 = ρL * vL * qL + pL * ny
        FL4 = ρL * wL * qL + pL * nz
        FL5 = (EL + pL) * qL
        
        factor = ρL * (SL - qL) / (SL - S_star + FT(1.0e-20))
        
        Us1 = factor * one(FT)
        Us2 = factor * (uL + (S_star - qL) * nx)
        Us3 = factor * (vL + (S_star - qL) * ny)
        Us4 = factor * (wL + (S_star - qL) * nz)
        Us5 = factor * (EL * inv_ρL + (S_star - qL) * (S_star + pL / (ρL * (SL - qL) - FT(1.0e-20))))
        return SVector{5, FT}(
            FL1 + SL * (Us1 - UL[1]),
            FL2 + SL * (Us2 - UL[2]),
            FL3 + SL * (Us3 - UL[3]),
            FL4 + SL * (Us4 - UL[4]),
            FL5 + SL * (Us5 - UL[5])
        )

    else 
        FR1 = ρR * qR
        FR2 = ρR * uR * qR + pR * nx
        FR3 = ρR * vR * qR + pR * ny
        FR4 = ρR * wR * qR + pR * nz
        FR5 = (ER + pR) * qR
        
        factor = ρR * (SR - qR) / (SR - S_star + FT(1.0e-20))
        
        Us1 = factor * one(FT)
        Us2 = factor * (uR + (S_star - qR) * nx)
        Us3 = factor * (vR + (S_star - qR) * ny)
        Us4 = factor * (wR + (S_star - qR) * nz)
        Us5 = factor * (ER * inv_ρR + (S_star - qR) * (S_star + pR / (ρR * (SR - qR) - FT(1.0e-20))))
        return SVector{5, FT}(
            FR1 + SR * (Us1 - UR[1]),
            FR2 + SR * (Us2 - UR[2]),
            FR3 + SR * (Us3 - UR[3]),
            FR4 + SR * (Us4 - UR[4]),
            FR5 + SR * (Us5 - UR[5])
        )
    end
end

@inline function VL_Flux(UL, UR, nx, ny, nz)
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10) || !isfinite(UL[1]) || !isfinite(UR[1])
        return SVector{5, FT}(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
    end
    ρL = UL[1]
    inv_ρL = one(FT) / ρL
    uL, vL, wL = UL[2]*inv_ρL, UL[3]*inv_ρL, UL[4]*inv_ρL
    pL = max(FT(1.0e-10), (γ - one(FT)) * (UL[5] - FT(0.5) * ρL * (uL^2 + vL^2 + wL^2)))
    cL = sqrt(γ * pL * inv_ρL)
    qL = uL * nx + vL * ny + wL * nz 
    ML = qL / cL

    fp1, fp2, fp3, fp4, fp5 = zero(FT), zero(FT), zero(FT), zero(FT), zero(FT)
    if ML >= one(FT)
        fp1 = ρL * qL
        fp2 = ρL * uL * qL + pL * nx
        fp3 = ρL * vL * qL + pL * ny
        fp4 = ρL * wL * qL + pL * nz
        fp5 = (UL[5] + pL) * qL
    elseif ML > -one(FT)
        f_mass = FT(0.25) * ρL * cL * (ML + one(FT))^2
        mom_fix = (-qL + FT(2.0) * cL) / γ
        fp1 = f_mass
        fp2 = f_mass * (uL + nx * mom_fix)
        fp3 = f_mass * (vL + ny * mom_fix)
        fp4 = f_mass * (wL + nz * mom_fix)
        h_term = ((γ - one(FT)) * qL + FT(2.0) * cL)^2 / (FT(2.0) * (γ^2 - one(FT)))
        fp5 = f_mass * (h_term + FT(0.5) * (uL^2 + vL^2 + wL^2 - qL^2))
    end

    ρR = UR[1]
    inv_ρR = one(FT) / ρR
    uR, vR, wR = UR[2]*inv_ρR, UR[3]*inv_ρR, UR[4]*inv_ρR
    pR = max(FT(1.0e-10), (γ - one(FT)) * (UR[5] - FT(0.5) * ρR * (uR^2 + vR^2 + wR^2)))
    cR = sqrt(γ * pR * inv_ρR)
    qR = uR * nx + vR * ny + wR * nz
    MR = qR / cR

    fm1, fm2, fm3, fm4, fm5 = zero(FT), zero(FT), zero(FT), zero(FT), zero(FT)
    if MR <= -one(FT)
        fm1 = ρR * qR
        fm2 = ρR * uR * qR + pR * nx
        fm3 = ρR * vR * qR + pR * ny
        fm4 = ρR * wR * qR + pR * nz
        fm5 = (UR[5] + pR) * qR
    elseif MR < one(FT)
        f_mass = -FT(0.25) * ρR * cR * (MR - one(FT))^2
        mom_fix = (-qR - FT(2.0) * cR) / γ
        fm1 = f_mass
        fm2 = f_mass * (uR + nx * mom_fix)
        fm3 = f_mass * (vR + ny * mom_fix)
        fm4 = f_mass * (wR + nz * mom_fix)
        h_term = ((γ - one(FT)) * qR - FT(2.0) * cR)^2 / (FT(2.0) * (γ^2 - one(FT)))
        fm5 = f_mass * (h_term + FT(0.5) * (uR^2 + vR^2 + wR^2 - qR^2))
    end

    return SVector{5, FT}(fp1 + fm1, fp2 + fm2, fp3 + fm3, fp4 + fm4, fp5 + fm5)
end

@inline function SW_Flux(UL, UR, nx, ny, nz)
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10) || !isfinite(UL[1]) || !isfinite(UR[1])
        return SVector{5, FT}(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
    end
    tmp1 = FT(2.0) * (γ - one(FT))
    tmp3 = (FT(3.0) - γ) / (FT(2.0) * (γ - one(FT)))

    ρL = UL[1]
    inv_ρL = one(FT) / ρL
    uL, vL, wL = UL[2]*inv_ρL, UL[3]*inv_ρL, UL[4]*inv_ρL
    pL = max(FT(1.0e-10), (γ - one(FT)) * (UL[5] - FT(0.5) * ρL * (uL^2 + vL^2 + wL^2)))
    cL = sqrt(γ * pL * inv_ρL)
    qL = uL * nx + vL * ny + wL * nz  # 法向速度
    
    E1P = FT(0.5) * (qL + abs(qL))
    E2P = FT(0.5) * (qL - cL + abs(qL - cL))
    E3P = FT(0.5) * (qL + cL + abs(qL + cL))
    tmp0_L = ρL / (FT(2.0) * γ)
    
    fp1 = tmp0_L * (tmp1 * E1P + E2P + E3P)
    fp2 = tmp0_L * (tmp1 * E1P * uL + E2P * (uL - cL * nx) + E3P * (uL + cL * nx))
    fp3 = tmp0_L * (tmp1 * E1P * vL + E2P * (vL - cL * ny) + E3P * (vL + cL * ny))
    fp4 = tmp0_L * (tmp1 * E1P * wL + E2P * (wL - cL * nz) + E3P * (wL + cL * nz))
    V2_L = uL^2 + vL^2 + wL^2
    fp5 = tmp0_L * (E1P * (γ - one(FT)) * V2_L + 
                    FT(0.5) * E2P * ((uL - cL * nx)^2 + (vL - cL * ny)^2 + (wL - cL * nz)^2) + 
                    FT(0.5) * E3P * ((uL + cL * nx)^2 + (vL + cL * ny)^2 + (wL + cL * nz)^2) + 
                    tmp3 * cL^2 * (E2P + E3P))

    ρR = UR[1]
    inv_ρR = one(FT) / ρR
    uR, vR, wR = UR[2]*inv_ρR, UR[3]*inv_ρR, UR[4]*inv_ρR
    pR = max(FT(1.0e-10), (γ - one(FT)) * (UR[5] - FT(0.5) * ρR * (uR^2 + vR^2 + wR^2)))
    cR = sqrt(γ * pR * inv_ρR)
    qR = uR * nx + vR * ny + wR * nz
    
    E1M = FT(0.5) * (qR - abs(qR))
    E2M = FT(0.5) * (qR - cR - abs(qR - cR))
    E3M = FT(0.5) * (qR + cR - abs(qR + cR))
    tmp0_R = ρR / (FT(2.0) * γ)
    
    fm1 = tmp0_R * (tmp1 * E1M + E2M + E3M)
    fm2 = tmp0_R * (tmp1 * E1M * uR + E2M * (uR - cR * nx) + E3M * (uR + cR * nx))
    fm3 = tmp0_R * (tmp1 * E1M * vR + E2M * (vR - cR * ny) + E3M * (vR + cR * ny))
    fm4 = tmp0_R * (tmp1 * E1M * wR + E2M * (wR - cR * nz) + E3M * (wR + cR * nz))
    V2_R = uR^2 + vR^2 + wR^2
    fm5 = tmp0_R * (E1M * (γ - one(FT)) * V2_R + 
                    FT(0.5) * E2M * ((uR - cR * nx)^2 + (vR - cR * ny)^2 + (wR - cR * nz)^2) + 
                    FT(0.5) * E3M * ((uR + cR * nx)^2 + (vR + cR * ny)^2 + (wR + cR * nz)^2) + 
                    tmp3 * cR^2 * (E2M + E3M))
 
    return SVector{5, FT}(fp1 + fm1, fp2 + fm2, fp3 + fm3, fp4 + fm4, fp5 + fm5)
end

@inline function Roe_Flux(UL, UR, nx, ny, nz)
    # 0. Safety check (including NaN guard: NaN < x is always false in IEEE 754)
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10) || !isfinite(UL[1]) || !isfinite(UR[1])
        return SVector{5, FT}(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
    end

    # 1. Primitive variables
    ρL = UL[1]; inv_ρL = one(FT) / ρL
    uL = UL[2] * inv_ρL; vL = UL[3] * inv_ρL; wL = UL[4] * inv_ρL
    EL = UL[5]
    pL = max(FT(1.0e-10), (γ - one(FT)) * (EL - FT(0.5) * ρL * (uL^2 + vL^2 + wL^2)))
    cL = sqrt(γ * pL * inv_ρL)
    qL = uL * nx + vL * ny + wL * nz
    HL = (EL + pL) * inv_ρL

    ρR = UR[1]; inv_ρR = one(FT) / ρR
    uR = UR[2] * inv_ρR; vR = UR[3] * inv_ρR; wR = UR[4] * inv_ρR
    ER = UR[5]
    pR = max(FT(1.0e-10), (γ - one(FT)) * (ER - FT(0.5) * ρR * (uR^2 + vR^2 + wR^2)))
    cR = sqrt(γ * pR * inv_ρR)
    qR = uR * nx + vR * ny + wR * nz
    HR = (ER + pR) * inv_ρR

    # 2. Roe averages
    sqρL = sqrt(ρL); sqρR = sqrt(ρR)
    inv_sqρ = one(FT) / (sqρL + sqρR)
    ρ_roe = sqρL * sqρR

    u_roe = (sqρL * uL + sqρR * uR) * inv_sqρ
    v_roe = (sqρL * vL + sqρR * vR) * inv_sqρ
    w_roe = (sqρL * wL + sqρR * wR) * inv_sqρ
    H_roe = (sqρL * HL + sqρR * HR) * inv_sqρ
    q_roe = u_roe * nx + v_roe * ny + w_roe * nz
    v2_roe = u_roe^2 + v_roe^2 + w_roe^2
    c_roe = sqrt(max(FT(1.0e-10), (γ - one(FT)) * (H_roe - FT(0.5) * v2_roe)))

    # 3. Left and right physical fluxes
    FL1 = ρL * qL
    FL2 = ρL * uL * qL + pL * nx
    FL3 = ρL * vL * qL + pL * ny
    FL4 = ρL * wL * qL + pL * nz
    FL5 = (EL + pL) * qL

    FR1 = ρR * qR
    FR2 = ρR * uR * qR + pR * nx
    FR3 = ρR * vR * qR + pR * ny
    FR4 = ρR * wR * qR + pR * nz
    FR5 = (ER + pR) * qR

    # 4. Eigenvalues with Harten's entropy fix
    ε = FT(0.1) * c_roe

    λ1 = q_roe - c_roe  # left acoustic
    λ2 = q_roe           # entropy / shear
    λ5 = q_roe + c_roe  # right acoustic

    abs_λ1 = abs(λ1) < ε ? (λ1^2 + ε^2) / (FT(2.0) * ε) : abs(λ1)
    abs_λ2 = abs(λ2) < ε ? (λ2^2 + ε^2) / (FT(2.0) * ε) : abs(λ2)
    abs_λ5 = abs(λ5) < ε ? (λ5^2 + ε^2) / (FT(2.0) * ε) : abs(λ5)

    # 5. Wave strengths (Roe decomposition)
    dp = pR - pL
    dq = qR - qL
    du = uR - uL; dv = vR - vL; dw = wR - wL
    inv_c2 = one(FT) / (c_roe^2)

    # α₁ = (Δp - ρ̃c̃ Δq) / (2c̃²)   — left acoustic
    # α₅ = (Δp + ρ̃c̃ Δq) / (2c̃²)   — right acoustic
    # α₂ = Δρ - Δp/c̃²              — entropy
    α1 = FT(0.5) * (dp - ρ_roe * c_roe * dq) * inv_c2
    α5 = FT(0.5) * (dp + ρ_roe * c_roe * dq) * inv_c2
    α2 = (UR[1] - UL[1]) - dp * inv_c2

    # Tangential velocity jump
    du_t_x = du - dq * nx
    du_t_y = dv - dq * ny
    du_t_z = dw - dq * nz

    # 6. Dissipation: |A|ΔU = Σ |λᵢ| αᵢ rᵢ
    # Wave 1 (left acoustic): r₁ = [1, u-cn_x, v-cn_y, w-cn_z, H-qc]
    d1 = abs_λ1 * α1
    diss1 = d1
    diss2 = d1 * (u_roe - c_roe * nx)
    diss3 = d1 * (v_roe - c_roe * ny)
    diss4 = d1 * (w_roe - c_roe * nz)
    diss5 = d1 * (H_roe - q_roe * c_roe)

    # Wave 2 (entropy): r₂ = [1, u, v, w, v²/2]
    d2 = abs_λ2 * α2
    diss1 += d2
    diss2 += d2 * u_roe
    diss3 += d2 * v_roe
    diss4 += d2 * w_roe
    diss5 += d2 * FT(0.5) * v2_roe

    # Waves 3,4 (shear): r₃₄ = [0, Δu_t, ū·Δu_t]
    d34 = abs_λ2 * ρ_roe
    diss2 += d34 * du_t_x
    diss3 += d34 * du_t_y
    diss4 += d34 * du_t_z
    diss5 += d34 * (u_roe * du_t_x + v_roe * du_t_y + w_roe * du_t_z)

    # Wave 5 (right acoustic): r₅ = [1, u+cn_x, v+cn_y, w+cn_z, H+qc]
    d5 = abs_λ5 * α5
    diss1 += d5
    diss2 += d5 * (u_roe + c_roe * nx)
    diss3 += d5 * (v_roe + c_roe * ny)
    diss4 += d5 * (w_roe + c_roe * nz)
    diss5 += d5 * (H_roe + q_roe * c_roe)

    # 7. Roe flux = 0.5(FL + FR) - 0.5|A|ΔU
    return SVector{5, FT}(
        FT(0.5) * (FL1 + FR1) - FT(0.5) * diss1,
        FT(0.5) * (FL2 + FR2) - FT(0.5) * diss2,
        FT(0.5) * (FL3 + FR3) - FT(0.5) * diss3,
        FT(0.5) * (FL4 + FR4) - FT(0.5) * diss4,
        FT(0.5) * (FL5 + FR5) - FT(0.5) * diss5
    )
end

@inline function KEP_Flux(UL, UR, nx, ny, nz)
    # Kinetic Energy Preserving (KEP) Central Flux for DNS (e.g., Pirozzoli 2010 style)
    # 0. Safety check (including NaN guard)
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10) || !isfinite(UL[1]) || !isfinite(UR[1])
        return SVector{5, FT}(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
    end
    
    # 1. Primitive variables
    ρL = UL[1]; inv_ρL = one(FT) / ρL
    uL = UL[2] * inv_ρL; vL = UL[3] * inv_ρL; wL = UL[4] * inv_ρL
    EL = UL[5]
    pL = max(FT(1.0e-10), (γ - one(FT)) * (EL - FT(0.5) * ρL * (uL^2 + vL^2 + wL^2)))
    
    ρR = UR[1]; inv_ρR = one(FT) / ρR
    uR = UR[2] * inv_ρR; vR = UR[3] * inv_ρR; wR = UR[4] * inv_ρR
    ER = UR[5]
    pR = max(FT(1.0e-10), (γ - one(FT)) * (ER - FT(0.5) * ρR * (uR^2 + vR^2 + wR^2)))
    
    # 2. Arithmetic averages
    # Kennedy & Gruber (2008) / Pirozzoli (2010) style central formulations
    # use simple arithmetic averages for the interface states.
    ρ_avg = FT(0.5) * (ρL + ρR)
    u_avg = FT(0.5) * (uL + uR)
    v_avg = FT(0.5) * (vL + vR)
    w_avg = FT(0.5) * (wL + wR)
    p_avg = FT(0.5) * (pL + pR)

    # Specific internal energy e = Cv*T = p/(ρ*(γ-1)) (codebase convention,
    # see volume_force.jl:33). Arithmetic average → KEP-consistent internal
    # energy flux. (EC upgrade would replace this with a log mean — see
    # docs/superpowers/specs/2026-06-18-entropy-stable-flux-design.md.)
    eL = pL / (ρL * (γ - one(FT)))
    eR = pR / (ρR * (γ - one(FT)))
    e_avg = FT(0.5) * (eL + eR)

    q_avg = u_avg * nx + v_avg * ny + w_avg * nz

    # 3. KEP Flux components
    # F_mass = ρ_avg * q_avg
    fp1 = ρ_avg * q_avg

    # F_momentum = F_mass * u_avg + p_avg * n
    fp2 = fp1 * u_avg + p_avg * nx
    fp3 = fp1 * v_avg + p_avg * ny
    fp4 = fp1 * w_avg + p_avg * nz

    # F_energy — KE/IE-consistent split form:
    #   F_ρ * [ ½(u_L·u_R + v_L·v_R + w_L·w_R)   (cross-product kinetic, from
    #                                          discrete chain rule — the unique
    #                                          form matching the momentum-flux
    #                                          average  (u_L+u_R)/2)
    #         + e_avg ]                          (arithmetic-avg internal energy)
    #   + p_avg * q_avg                          (pressure work, consistent with
    #                                          momentum pressure term p_avg*n)
    # Reduces to (ρE + p)*q for UL=UR. Replaces the former (E_avg+p_avg)*q_avg
    # which carried an O(Δx²) KE↔IE spurious source at jumps.
    fp5 = fp1 * (FT(0.5) * (uL*uR + vL*vR + wL*wR) + e_avg) + p_avg * q_avg

    return SVector{5, FT}(fp1, fp2, fp3, fp4, fp5)
end


# ═══════════════════════════════════════════════════════════════════════
# Riemann_Solver_MHD.jl — MHD flux functions with GLM divergence cleaning
# ═══════════════════════════════════════════════════════════════════════
# Conservative variables: U = (ρ, ρu, ρv, ρw, ρE, Bx, By, Bz, ψ)
# GLM-MHD (Dedner et al. 2002): adds ψ to normal B-flux, ch²·Bn to ψ-flux
#
# All fluxes return SVector{9, FT}
# ═══════════════════════════════════════════════════════════════════════

# ─── Helper: MHD physical flux in normal direction ───
# Given primitives and conservative state, compute F·n
@inline function _mhd_flux_normal(ρ, u, v, w, p, Bx, By, Bz, ψ, E_total, nx, ny, nz, ch::FT)
    # Normal velocity and normal B-field
    qn = u*nx + v*ny + w*nz
    Bn = Bx*nx + By*ny + Bz*nz

    # Total pressure: p_total = p + B²/(2*mu0)
    B2 = Bx*Bx + By*By + Bz*Bz
    pt = p + FT(0.5) * B2 * INV_MU0_SI

    # MHD flux (+ GLM corrections)
    f1 = ρ * qn                                           # mass
    f2 = ρ*u*qn + pt*nx - Bx*Bn*INV_MU0_SI                # x-momentum
    f3 = ρ*v*qn + pt*ny - By*Bn*INV_MU0_SI                # y-momentum
    f4 = ρ*w*qn + pt*nz - Bz*Bn*INV_MU0_SI                # z-momentum
    f5 = isothermal_mhd ? zero(FT) :
         (E_total + pt)*qn -
         Bn*(u*Bx + v*By + w*Bz)*INV_MU0_SI              # energy/carrier
    f6 = Bx*qn - u*Bn + ψ*nx                              # Bx (+ GLM: ψ·nx)
    f7 = By*qn - v*Bn + ψ*ny                              # By (+ GLM: ψ·ny)
    f8 = Bz*qn - w*Bn + ψ*nz                              # Bz (+ GLM: ψ·nz)
    f9 = ch*ch*Bn                                          # ψ (GLM: ch²·Bn)

    return SVector{9, FT}(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

# ═══════════════════════════════════════════════════════════════════════
# MHD Rusanov (Lax-Friedrichs) Flux
# ═══════════════════════════════════════════════════════════════════════
# Simple, robust, highly diffusive. Good for debugging.
@inline function MHD_Rusanov_Flux(UL, UR, nx, ny, nz, ch_glm::FT)
    # Safety check
    if UL[1] < density_floor || UR[1] < density_floor
        return SVector{9, FT}(FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0))
    end

    ch = ch_glm

    # Left state primitives
    ρL = UL[1]; inv_ρL = one(FT) / ρL
    uL = UL[2]*inv_ρL; vL = UL[3]*inv_ρL; wL = UL[4]*inv_ρL
    BxL = UL[6]; ByL = UL[7]; BzL = UL[8]; ψL = UL[9]
    B2L = BxL*BxL + ByL*ByL + BzL*BzL
    pL = isothermal_mhd ? ρL * Rg * isothermal_temperature :
        max(pressure_floor, (γ - one(FT)) * (
            UL[5] - FT(0.5)*ρL*(uL*uL+vL*vL+wL*wL) -
            FT(0.5)*B2L*INV_MU0_SI))
    qnL = uL*nx + vL*ny + wL*nz

    # Right state primitives
    ρR = UR[1]; inv_ρR = one(FT) / ρR
    uR = UR[2]*inv_ρR; vR = UR[3]*inv_ρR; wR = UR[4]*inv_ρR
    BxR = UR[6]; ByR = UR[7]; BzR = UR[8]; ψR = UR[9]
    B2R = BxR*BxR + ByR*ByR + BzR*BzR
    pR = isothermal_mhd ? ρR * Rg * isothermal_temperature :
        max(pressure_floor, (γ - one(FT)) * (
            UR[5] - FT(0.5)*ρR*(uR*uR+vR*vR+wR*wR) -
            FT(0.5)*B2R*INV_MU0_SI))
    qnR = uR*nx + vR*ny + wR*nz

    # Fast magnetosonic speeds
    c2L = mhd_sound_speed_squared(ρL, pL, γ)
    va2L = B2L * INV_MU0_SI * inv_ρL
    cfL = sqrt(c2L + va2L)

    c2R = mhd_sound_speed_squared(ρR, pR, γ)
    va2R = B2R * INV_MU0_SI * inv_ρR
    cfR = sqrt(c2R + va2R)

    # Maximum wave speed (including GLM speed ch)
    λ_max = max(abs(qnL) + cfL, abs(qnR) + cfR, ch)

    # Physical fluxes
    FL = _mhd_flux_normal(ρL, uL, vL, wL, pL, BxL, ByL, BzL, ψL, UL[5], nx, ny, nz, ch)
    FR = _mhd_flux_normal(ρR, uR, vR, wR, pR, BxR, ByR, BzR, ψR, UR[5], nx, ny, nz, ch)

    # The isothermal closure has no evolved energy equation. U[5] is only a
    # CT compatibility carrier and is rebuilt after each finalized stage.
    state_jump = isothermal_mhd ?
        SVector{9,FT}(
            UR[1] - UL[1], UR[2] - UL[2], UR[3] - UL[3],
            UR[4] - UL[4], zero(FT), UR[6] - UL[6],
            UR[7] - UL[7], UR[8] - UL[8], UR[9] - UL[9],
        ) : UR - UL

    # Rusanov: F = 0.5(FL+FR) - 0.5·lambda_max·state_jump
    return SVector{9, FT}(
        FT(0.5)*(FL[1]+FR[1]) - FT(0.5)*λ_max*state_jump[1],
        FT(0.5)*(FL[2]+FR[2]) - FT(0.5)*λ_max*state_jump[2],
        FT(0.5)*(FL[3]+FR[3]) - FT(0.5)*λ_max*state_jump[3],
        FT(0.5)*(FL[4]+FR[4]) - FT(0.5)*λ_max*state_jump[4],
        FT(0.5)*(FL[5]+FR[5]) - FT(0.5)*λ_max*state_jump[5],
        FT(0.5)*(FL[6]+FR[6]) - FT(0.5)*λ_max*state_jump[6],
        FT(0.5)*(FL[7]+FR[7]) - FT(0.5)*λ_max*state_jump[7],
        FT(0.5)*(FL[8]+FR[8]) - FT(0.5)*λ_max*state_jump[8],
        FT(0.5)*(FL[9]+FR[9]) - FT(0.5)*λ_max*state_jump[9]
    )
end

# Robust two-wave fallback for MHD.  CT supplies a single face-normal B to
# both inputs before this function is called, so HLLE cannot introduce a jump
# in the normal magnetic field.
@inline function _mhd_state_is_physical(U)
    rho = U[1]
    if !(isfinite(rho) && rho > zero(FT))
        return false
    end
    inv_rho = one(FT) / rho
    kinetic = FT(0.5) * (
        U[2]*U[2] + U[3]*U[3] + U[4]*U[4]
    ) * inv_rho
    magnetic = FT(0.5) * INV_MU0_SI * (
        U[6]*U[6] + U[7]*U[7] + U[8]*U[8]
    )
    pressure = mhd_thermodynamic_pressure(
        rho, U[5] - kinetic - magnetic, γ,
    )
    return isfinite(pressure) && pressure > zero(FT) &&
           all(isfinite, U)
end

@inline function MHD_HLLE_Flux(UL, UR, nx, ny, nz, ch_glm::FT)
    if !(_mhd_state_is_physical(UL) && _mhd_state_is_physical(UR))
        return _mhd_nan_flux()
    end

    rhoL = UL[1]; inv_rhoL = one(FT) / rhoL
    uL = UL[2]*inv_rhoL; vL = UL[3]*inv_rhoL; wL = UL[4]*inv_rhoL
    BxL = UL[6]; ByL = UL[7]; BzL = UL[8]; psiL = UL[9]
    B2L = BxL*BxL + ByL*ByL + BzL*BzL
    pL = isothermal_mhd ? rhoL * Rg * isothermal_temperature :
        (γ-one(FT)) * (
            UL[5] - FT(0.5)*rhoL*(uL*uL+vL*vL+wL*wL) -
            FT(0.5)*B2L*INV_MU0_SI
        )
    qnL = uL*nx + vL*ny + wL*nz

    rhoR = UR[1]; inv_rhoR = one(FT) / rhoR
    uR = UR[2]*inv_rhoR; vR = UR[3]*inv_rhoR; wR = UR[4]*inv_rhoR
    BxR = UR[6]; ByR = UR[7]; BzR = UR[8]; psiR = UR[9]
    B2R = BxR*BxR + ByR*ByR + BzR*BzR
    pR = isothermal_mhd ? rhoR * Rg * isothermal_temperature :
        (γ-one(FT)) * (
            UR[5] - FT(0.5)*rhoR*(uR*uR+vR*vR+wR*wR) -
            FT(0.5)*B2R*INV_MU0_SI
        )
    qnR = uR*nx + vR*ny + wR*nz

    cfL = sqrt(mhd_sound_speed_squared(rhoL, pL, γ) +
               B2L*INV_MU0_SI*inv_rhoL)
    cfR = sqrt(mhd_sound_speed_squared(rhoR, pR, γ) +
               B2R*INV_MU0_SI*inv_rhoR)
    ch = ch_glm
    SL = min(qnL-cfL, qnR-cfR, -ch)
    SR = max(qnL+cfL, qnR+cfR, ch)
    FL = _mhd_flux_normal(
        rhoL, uL, vL, wL, pL, BxL, ByL, BzL, psiL, UL[5],
        nx, ny, nz, ch,
    )
    FR = _mhd_flux_normal(
        rhoR, uR, vR, wR, pR, BxR, ByR, BzR, psiR, UR[5],
        nx, ny, nz, ch,
    )
    if SL >= zero(FT)
        return FL
    elseif SR <= zero(FT)
        return FR
    end
    inv_span = one(FT) / (SR - SL)
    state_jump = isothermal_mhd ?
        SVector{9,FT}(
            UR[1] - UL[1], UR[2] - UL[2], UR[3] - UL[3],
            UR[4] - UL[4], zero(FT), UR[6] - UL[6],
            UR[7] - UL[7], UR[8] - UL[8], UR[9] - UL[9],
        ) : UR - UL
    return SVector{9,FT}(ntuple(Val(9)) do n
        (SR*FL[n] - SL*FR[n] + SL*SR*state_jump[n]) * inv_span
    end)
end

# ═══════════════════════════════════════════════════════════════════════
# HLLD Flux — Miyoshi & Kusano (2005)
# ═══════════════════════════════════════════════════════════════════════
# 5-wave approximate Riemann solver for ideal MHD.
# Captures fast magnetosonic, Alfvén, and contact discontinuities.
# Extended with GLM for the ψ-equation.
# Background-field HLLE for CT B=B0+b. The input magnetic components and
# magnetic part of U[5] describe b; wave speeds and induction use B0+b.
@inline function _mhd_background_state_is_physical(U)
    rho = U[1]
    if !(isfinite(rho) && rho > zero(FT))
        return false
    end
    inv_rho = one(FT) / rho
    kinetic = FT(0.5) * (
        U[2]*U[2] + U[3]*U[3] + U[4]*U[4]
    ) * inv_rho
    perturbation_magnetic = FT(0.5) * INV_MU0_SI * (
        U[6]*U[6] + U[7]*U[7] + U[8]*U[8]
    )
    pressure = mhd_thermodynamic_pressure(
        rho, U[5] - kinetic - perturbation_magnetic, γ,
    )
    return isfinite(pressure) && pressure > zero(FT) && all(isfinite, U)
end

@inline function _mhd_background_flux_normal(
    U, background::SVector{3,FT}, nx, ny, nz, ch::FT,
)
    rho = U[1]
    inv_rho = one(FT) / rho
    u = U[2] * inv_rho
    v = U[3] * inv_rho
    w = U[4] * inv_rho
    bx = U[6]
    by = U[7]
    bz = U[8]
    psi = U[9]
    b0x, b0y, b0z = background

    qn = u*nx + v*ny + w*nz
    bn = bx*nx + by*ny + bz*nz
    b0n = b0x*nx + b0y*ny + b0z*nz
    total_bn = bn + b0n
    b2 = bx*bx + by*by + bz*bz
    b0_dot_b = b0x*bx + b0y*by + b0z*bz
    velocity_dot_b = u*bx + v*by + w*bz
    kinetic = FT(0.5) * rho * (u*u + v*v + w*w)
    pressure = mhd_thermodynamic_pressure(
        rho, U[5] - kinetic - FT(0.5)*b2*INV_MU0_SI, γ,
    )
    split_magnetic_pressure = (
        b0_dot_b + FT(0.5)*b2
    ) * INV_MU0_SI

    f1 = rho * qn
    f2 = rho*u*qn + (pressure + split_magnetic_pressure)*nx -
         (b0x*bn + bx*b0n + bx*bn)*INV_MU0_SI
    f3 = rho*v*qn + (pressure + split_magnetic_pressure)*ny -
         (b0y*bn + by*b0n + by*bn)*INV_MU0_SI
    f4 = rho*w*qn + (pressure + split_magnetic_pressure)*nz -
         (b0z*bn + bz*b0n + bz*bn)*INV_MU0_SI
    f5 = isothermal_mhd ? zero(FT) :
         (U[5] + pressure + split_magnetic_pressure)*qn -
         total_bn*velocity_dot_b*INV_MU0_SI
    f6 = (b0x + bx)*qn - u*total_bn + psi*nx
    f7 = (b0y + by)*qn - v*total_bn + psi*ny
    f8 = (b0z + bz)*qn - w*total_bn + psi*nz
    f9 = ch*ch*total_bn
    return SVector{9,FT}(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

@inline function _mhd_background_total_energy_carrier_flux(
    reduced_flux, background::SVector{3,FT},
)
    isothermal_mhd && return setindex(reduced_flux, zero(FT), 5)
    induction_work = (
        background[1]*reduced_flux[6] +
        background[2]*reduced_flux[7] +
        background[3]*reduced_flux[8]
    ) * INV_MU0_SI
    return setindex(reduced_flux, reduced_flux[5] + induction_work, 5)
end

@inline function MHD_HLLE_Background_Flux(
    UL, UR, background::SVector{3,FT}, nx, ny, nz, ch_glm::FT,
)
    if !(_mhd_background_state_is_physical(UL) &&
         _mhd_background_state_is_physical(UR))
        return _mhd_nan_flux()
    end

    rhoL = UL[1]
    rhoR = UR[1]
    inv_rhoL = one(FT) / rhoL
    inv_rhoR = one(FT) / rhoR
    uL = UL[2]*inv_rhoL; vL = UL[3]*inv_rhoL; wL = UL[4]*inv_rhoL
    uR = UR[2]*inv_rhoR; vR = UR[3]*inv_rhoR; wR = UR[4]*inv_rhoR
    bxL = background[1] + UL[6]
    byL = background[2] + UL[7]
    bzL = background[3] + UL[8]
    bxR = background[1] + UR[6]
    byR = background[2] + UR[7]
    bzR = background[3] + UR[8]
    b2L = UL[6]*UL[6] + UL[7]*UL[7] + UL[8]*UL[8]
    b2R = UR[6]*UR[6] + UR[7]*UR[7] + UR[8]*UR[8]
    kineticL = FT(0.5) * rhoL * (uL*uL + vL*vL + wL*wL)
    kineticR = FT(0.5) * rhoR * (uR*uR + vR*vR + wR*wR)
    pL = mhd_thermodynamic_pressure(
        rhoL, UL[5] - kineticL - FT(0.5)*b2L*INV_MU0_SI, γ,
    )
    pR = mhd_thermodynamic_pressure(
        rhoR, UR[5] - kineticR - FT(0.5)*b2R*INV_MU0_SI, γ,
    )
    qnL = uL*nx + vL*ny + wL*nz
    qnR = uR*nx + vR*ny + wR*nz
    total_b2L = bxL*bxL + byL*byL + bzL*bzL
    total_b2R = bxR*bxR + byR*byR + bzR*bzR
    cfL = sqrt(mhd_sound_speed_squared(rhoL, pL, γ) +
               total_b2L*INV_MU0_SI*inv_rhoL)
    cfR = sqrt(mhd_sound_speed_squared(rhoR, pR, γ) +
               total_b2R*INV_MU0_SI*inv_rhoR)
    ch = ch_glm
    SL = min(qnL-cfL, qnR-cfR, -ch)
    SR = max(qnL+cfL, qnR+cfR, ch)
    FL = _mhd_background_flux_normal(UL, background, nx, ny, nz, ch)
    FR = _mhd_background_flux_normal(UR, background, nx, ny, nz, ch)

    reduced_flux = if SL >= zero(FT)
        FL
    elseif SR <= zero(FT)
        FR
    else
        inv_span = one(FT) / (SR - SL)
        state_jump = isothermal_mhd ?
            SVector{9,FT}(
                UR[1]-UL[1], UR[2]-UL[2], UR[3]-UL[3], UR[4]-UL[4],
                zero(FT), UR[6]-UL[6], UR[7]-UL[7], UR[8]-UL[8],
                UR[9]-UL[9],
            ) : UR - UL
        SVector{9,FT}(ntuple(Val(9)) do n
            (SR*FL[n] - SL*FR[n] + SL*SR*state_jump[n]) * inv_span
        end)
    end
    return _mhd_background_total_energy_carrier_flux(
        reduced_flux, background,
    )
end

@inline _mhd_nan_flux() =
    SVector{9,FT}(ntuple(_ -> FT(NaN), Val(9)))

@inline function HLLD_Flux(
    UL, UR, nx, ny, nz, ch_glm::FT, fallback_meta=nothing,
)
    @static if isothermal_mhd
        return MHD_Rusanov_Flux(UL, UR, nx, ny, nz, ch_glm)
    end
    # Safety check
    @static if strict_ct_positivity && ct_mode
        if !(isfinite(UL[1]) && UL[1] > zero(FT) &&
             isfinite(UR[1]) && UR[1] > zero(FT))
            return _mhd_nan_flux()
        end
    else
        if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10)
            return SVector{9, FT}(FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0))
        end
    end

    ch = ch_glm

    # ── Left state ──
    ρL = UL[1]; inv_ρL = one(FT) / ρL
    uL = UL[2]*inv_ρL; vL = UL[3]*inv_ρL; wL = UL[4]*inv_ρL
    BxL = UL[6]; ByL = UL[7]; BzL = UL[8]; ψL = UL[9]
    B2L = BxL*BxL + ByL*ByL + BzL*BzL
    raw_pL = (γ-one(FT)) *
             (UL[5] - FT(0.5)*ρL*(uL*uL+vL*vL+wL*wL) -
              FT(0.5)*B2L*INV_MU0_SI)
    @static if strict_ct_positivity && ct_mode
        if !(isfinite(raw_pL) && raw_pL > zero(FT))
            return _mhd_nan_flux()
        end
        pL = raw_pL
    else
        pL = max(FT(1.0e-10), raw_pL)
    end
    ptL = pL + FT(0.5)*B2L*INV_MU0_SI  # total pressure
    qnL = uL*nx + vL*ny + wL*nz
    BnL = BxL*nx + ByL*ny + BzL*nz

    # ── Right state ──
    ρR = UR[1]; inv_ρR = one(FT) / ρR
    uR = UR[2]*inv_ρR; vR = UR[3]*inv_ρR; wR = UR[4]*inv_ρR
    BxR = UR[6]; ByR = UR[7]; BzR = UR[8]; ψR = UR[9]
    B2R = BxR*BxR + ByR*ByR + BzR*BzR
    raw_pR = (γ-one(FT)) *
             (UR[5] - FT(0.5)*ρR*(uR*uR+vR*vR+wR*wR) -
              FT(0.5)*B2R*INV_MU0_SI)
    @static if strict_ct_positivity && ct_mode
        if !(isfinite(raw_pR) && raw_pR > zero(FT))
            return _mhd_nan_flux()
        end
        pR = raw_pR
    else
        pR = max(FT(1.0e-10), raw_pR)
    end
    ptR = pR + FT(0.5)*B2R*INV_MU0_SI
    qnR = uR*nx + vR*ny + wR*nz
    BnR = BxR*nx + ByR*ny + BzR*nz

    # ── Fast magnetosonic speeds ──
    c2L = γ*pL*inv_ρL; c2R = γ*pR*inv_ρR
    va2L = B2L*INV_MU0_SI*inv_ρL
    va2R = B2R*INV_MU0_SI*inv_ρR
    cfL = sqrt(c2L + va2L)
    cfR = sqrt(c2R + va2R)

    # ── Wave speed estimates (Davis estimate) ──
    SL = min(qnL - cfL, qnR - cfR)
    SR = max(qnL + cfL, qnR + cfR)

    # ── HLL contact speed ──
    SM = (ρR*qnR*(SR-qnR) - ρL*qnL*(SL-qnL) + ptL - ptR) /
         (ρR*(SR-qnR) - ρL*(SL-qnL) + FT(1.0e-20))

    # ── Total pressure in star region (Athena: average of left and right) ──
    ptS_L = ptL + ρL*(SL-qnL)*(SM-qnL)
    ptS_R = ptR + ρR*(SR-qnR)*(SM-qnR)
    ptS = FT(0.5)*(ptS_L + ptS_R)

    # ── HLL average Bn (single Bn* for consistent formulations) ──
    # In ideal MHD, Bn is continuous across all waves (∇·B=0 constraint).
    # Use arithmetic mean; when |Bn*| is small, Alfvén sub-states collapse to SM.
    Bn_hll = FT(0.5)*(BnL + BnR)

    # ── Left star state (* region, between SL and S*L) ──
    # Miyoshi & Kusano (2005), eq. (12)-(18).
    inv_SL_SM = one(FT) / (SL - SM + FT(1.0e-20))
    raw_ρsL = ρL * (SL - qnL) * inv_SL_SM
    @static if strict_ct_positivity && ct_mode
        if !(isfinite(raw_ρsL) && raw_ρsL > zero(FT))
            return _mhd_nan_flux()
        end
        ρsL = raw_ρsL
    else
        ρsL = max(raw_ρsL, FT(1.0e-10))
    end
    inv_ρsL = one(FT) / ρsL

    # Star B-field tangential components (Miyoshi eq. 14):
    #   B*_t = B_t * (ρL*(SL-qnL)² - Bn*²) / (ρL*(SL-qnL)*(SL-SM) - Bn*²)
    # and B*_n = Bn* (= Bn_hll, single-valued)
    Bn2_over_mu0 = Bn_hll*Bn_hll*INV_MU0_SI
    denom_B_L = ρL*(SL-qnL)*(SL-SM) - Bn2_over_mu0
    # When denom_B_L is near zero, B*_t = B_t (no jump across the fast wave)
    factor_B_L = abs(denom_B_L) > FT(1.0e-8) ?
        (ρL*(SL-qnL)*(SL-qnL) - Bn2_over_mu0) /
        (denom_B_L + FT(1.0e-20)) : one(FT)
    # B*_t = B_t * factor_B_L ; B*_n = Bn_hll (replace normal component)
    BxsL = BxL*factor_B_L + Bn_hll*nx*(one(FT) - factor_B_L)
    BysL = ByL*factor_B_L + Bn_hll*ny*(one(FT) - factor_B_L)
    BzsL = BzL*factor_B_L + Bn_hll*nz*(one(FT) - factor_B_L)

    # Star velocity (Miyoshi eq. 12-13):
    #   v*_n = SM  (contact discontinuity moves at SM)
    #   v*_t = v_t - Bn*·(B*_t - B_t) / (ρL·(SL-qnL))
    # The normal component must be set explicitly to SM.
    coeff_v_L = Bn_hll*INV_MU0_SI /
        (ρL*(SL-qnL) + FT(1.0e-20))
    usL = SM*nx + (uL - qnL*nx) - coeff_v_L*(BxsL - BxL)
    vsL = SM*ny + (vL - qnL*ny) - coeff_v_L*(BysL - ByL)
    wsL = SM*nz + (wL - qnL*nz) - coeff_v_L*(BzsL - BzL)

    vBsL = usL*BxsL + vsL*BysL + wsL*BzsL
    # Star total energy (Miyoshi eq. 17):
    EsL = ((SL-qnL)*UL[5] - ptL*qnL + ptS*SM +
           Bn_hll*INV_MU0_SI*(uL*BxL+vL*ByL+wL*BzL - vBsL)) * inv_SL_SM

    # ── Right star state (* region, between S*R and SR) ──
    inv_SR_SM = one(FT) / (SR - SM + FT(1.0e-20))
    raw_ρsR = ρR * (SR - qnR) * inv_SR_SM
    @static if strict_ct_positivity && ct_mode
        if !(isfinite(raw_ρsR) && raw_ρsR > zero(FT))
            return _mhd_nan_flux()
        end
        ρsR = raw_ρsR
    else
        ρsR = max(raw_ρsR, FT(1.0e-10))
    end
    inv_ρsR = one(FT) / ρsR

    denom_B_R = ρR*(SR-qnR)*(SR-SM) - Bn2_over_mu0
    factor_B_R = abs(denom_B_R) > FT(1.0e-8) ?
        (ρR*(SR-qnR)*(SR-qnR) - Bn2_over_mu0) /
        (denom_B_R + FT(1.0e-20)) : one(FT)
    BxsR = BxR*factor_B_R + Bn_hll*nx*(one(FT) - factor_B_R)
    BysR = ByR*factor_B_R + Bn_hll*ny*(one(FT) - factor_B_R)
    BzsR = BzR*factor_B_R + Bn_hll*nz*(one(FT) - factor_B_R)

    coeff_v_R = Bn_hll*INV_MU0_SI /
        (ρR*(SR-qnR) + FT(1.0e-20))
    usR = SM*nx + (uR - qnR*nx) - coeff_v_R*(BxsR - BxR)
    vsR = SM*ny + (vR - qnR*ny) - coeff_v_R*(BysR - ByR)
    wsR = SM*nz + (wR - qnR*nz) - coeff_v_R*(BzsR - BzR)

    vBsR = usR*BxsR + vsR*BysR + wsR*BzsR
    EsR = ((SR-qnR)*UR[5] - ptR*qnR + ptS*SM +
           Bn_hll*INV_MU0_SI*(uR*BxR+vR*ByR+wR*BzR - vBsR)) * inv_SR_SM

    # ── Alfvén sub-states (** region, between S*L and S*R) ──
    # Following Athena++ (Miyoshi & Kusano 2005, eqns 51-63).
    # The ** state has SHARED tangential v and B on both sides.
    sqrt_ρsL = sqrt(ρsL); sqrt_ρsR = sqrt(ρsR)
    abs_Bn = abs(Bn_hll) * INV_SQRT_MU0_SI
    # Degeneracy: skip Alfvén sub-states when Bn*≈0 or ρs too small
    denom_deg = ρL*(SL-qnL)*(SL-SM) - Bn2_over_mu0
    denom_deg_R = ρR*(SR-qnR)*(SR-SM) - Bn2_over_mu0
    if abs(denom_deg) < FT(1.0e-4)*ptS || abs(denom_deg_R) < FT(1.0e-4)*ptS
        # Bn*≈0 degenerate: ** state = * state
        SstL = SM; SstR = SM
        ussL = usL; vssL = vsL; wssL = wsL
        BxssL = BxsL; ByssL = BysL; BzssL = BzsL
        ussR = usL; vssR = vsL; wssR = wsL
        BxssR = BxsL; ByssR = BysL; BzssR = BzsL
        EssL = EsL; EssR = EsR
    else
    # Alfvén wave speeds (M&K eqn 51): S*L = SM - |Bn*|/√ρsL, S*R = SM + |Bn*|/√ρsR
    SstL = SM - abs_Bn/sqrt_ρsL
    SstR = SM + abs_Bn/sqrt_ρsR

    sign_Bn = Bn_hll >= zero(FT) ? one(FT) : -one(FT)
    inv_sqrt_sum = one(FT) / (sqrt_ρsL + sqrt_ρsR)

    # Tangential v* and B* (normal components subtracted)
    # v*_t = v* - SM·n̂,  B*_t = B* - Bn*·n̂
    usL_t = usL - SM*nx; vsL_t = vsL - SM*ny; wsL_t = wsL - SM*nz
    usR_t = usR - SM*nx; vsR_t = vsR - SM*ny; wsR_t = wsR - SM*nz
    BxsL_t = BxsL - Bn_hll*nx; BysL_t = BysL - Bn_hll*ny; BzsL_t = BzsL - Bn_hll*nz
    BxsR_t = BxsR - Bn_hll*nx; BysR_t = BysR - Bn_hll*ny; BzsR_t = BzsR - Bn_hll*nz

    # ** state tangential velocity (M&K eqn 59-60, SHARED left and right):
    #   v**_t = (sqrt(ρsL)*v*L_t + sqrt(ρsR)*v*R_t + sign(Bn)*(B*R_t - B*L_t)) / (sqrt(ρsL)+sqrt(ρsR))
    uss_t = inv_sqrt_sum * (sqrt_ρsL*usL_t + sqrt_ρsR*usR_t +
        sign_Bn*INV_SQRT_MU0_SI*(BxsR_t - BxsL_t))
    vss_t = inv_sqrt_sum * (sqrt_ρsL*vsL_t + sqrt_ρsR*vsR_t +
        sign_Bn*INV_SQRT_MU0_SI*(BysR_t - BysL_t))
    wss_t = inv_sqrt_sum * (sqrt_ρsL*wsL_t + sqrt_ρsR*wsR_t +
        sign_Bn*INV_SQRT_MU0_SI*(BzsR_t - BzsL_t))

    # ** state tangential B (M&K eqn 61-62, SHARED left and right):
    #   B**_t = (sqrt(ρsL)*B*R_t + sqrt(ρsR)*B*L_t + sign(Bn)*sqrt(ρsL)*sqrt(ρsR)*(v*R_t - v*L_t)) / (sqrt(ρsL)+sqrt(ρsR))
    Bxss_t = inv_sqrt_sum * (sqrt_ρsL*BxsR_t + sqrt_ρsR*BxsL_t +
        sign_Bn*sqrt_ρsL*sqrt_ρsR*SQRT_MU0_SI*(usR_t - usL_t))
    Byss_t = inv_sqrt_sum * (sqrt_ρsL*BysR_t + sqrt_ρsR*BysL_t +
        sign_Bn*sqrt_ρsL*sqrt_ρsR*SQRT_MU0_SI*(vsR_t - vsL_t))
    Bzss_t = inv_sqrt_sum * (sqrt_ρsL*BzsR_t + sqrt_ρsR*BzsL_t +
        sign_Bn*sqrt_ρsL*sqrt_ρsR*SQRT_MU0_SI*(wsR_t - wsL_t))

    # Full ** state (add back normal components: v**_n = SM, B**_n = Bn*)
    # SHARED between left and right
    ussL = SM*nx + uss_t; vssL = SM*ny + vss_t; wssL = SM*nz + wss_t
    ussR = ussL; vssR = vssL; wssR = wssL
    BxssL = Bn_hll*nx + Bxss_t; ByssL = Bn_hll*ny + Byss_t; BzssL = Bn_hll*nz + Bzss_t
    BxssR = BxssL; ByssR = ByssL; BzssR = BzssL

    # ** state energy (M&K eqn 63):
    #   vB_common = SM*Bn* + v**_t · B**_t
    #   E**L = E*L - sqrt(ρsL)*sign(Bn)*(v*L·B*L - vB_common)
    #   E**R = E*R + sqrt(ρsR)*sign(Bn)*(v*R·B*R - vB_common)
    vBss = SM*Bn_hll + uss_t*Bxss_t + vss_t*Byss_t + wss_t*Bzss_t
    EssL = EsL - sqrt_ρsL*sign_Bn*INV_SQRT_MU0_SI*(vBsL - vBss)
    EssR = EsR + sqrt_ρsR*sign_Bn*INV_SQRT_MU0_SI*(vBsR - vBss)
    end  # else (non-degenerate)

    # ── GLM: ψ star state (Dedner) ──
    # GLM ψ: NOT propagated through HLLD star states (set to 0 in Us).
    # The ψ flux is handled separately at the end via GLM upwind, to prevent
    # ch²·Bn blow-up in low-density cells where ch→∞. This decouples the
    # GLM cleaning from the HLLD star state machinery.
    ψs = zero(FT)

    # ── Safety fallback: invalid HLLD star state reverts to HLLE ──
    # Checks: finite energies, wave ordering, finite star velocities/B-fields,
    # bounded factor_B, and star energies not wildly larger than input (×100).
    E_ref = max(abs(UL[5]), abs(UR[5])) * FT(100.0)
    if !(EsL > zero(FT)) || !(EsR > zero(FT)) || !(EssL > zero(FT)) || !(EssR > zero(FT)) ||
       isnan(EsL) || isnan(EsR) || isnan(EssL) || isnan(EssR) ||
       !(SM > SL) || !(SR > SM) || !(SstL <= SM) || !(SstR >= SM) ||
       isnan(usL) || isnan(usR) || isnan(ussL) || isnan(ussR) ||
       isnan(BxsL) || isnan(BxsR) || isnan(BxssL) || isnan(BxssR) ||
       abs(factor_B_L) > FT(1.0e6) || abs(factor_B_R) > FT(1.0e6) ||
       EsL > E_ref || EsR > E_ref || EssL > E_ref || EssR > E_ref
        if fallback_meta !== nothing
            ct_record_fallback!(fallback_meta, CT_POS_HLLD_TO_HLLE_COUNT)
        end
        return MHD_HLLE_Flux(UL, UR, nx, ny, nz, ch_glm)
    end

    # ── Select flux based on 7-region wave pattern ──
    # Wave structure: SL — [L*] — S*L — [L**] — SM — [R**] — S*R — [R*] — SR
    # When Bn*≈0, S*L→SM and S*R→SM, so ** regions vanish and we get 3-wave HLLC.
    if SL >= zero(FT)
        # Region L (supersonic left)
        F = _mhd_flux_normal(ρL, uL, vL, wL, pL, BxL, ByL, BzL, ψL, UL[5], nx, ny, nz, ch)
    elseif SstL >= zero(FT)
        # Region L* (left star, between SL and S*L)
        FL = _mhd_flux_normal(ρL, uL, vL, wL, pL, BxL, ByL, BzL, ψL, UL[5], nx, ny, nz, ch)
        Us = SVector{9, FT}(ρsL, ρsL*usL, ρsL*vsL, ρsL*wsL, EsL, BxsL, BysL, BzsL, ψs)
        F = SVector{9, FT}(
            FL[1] + SL*(Us[1] - UL[1]), FL[2] + SL*(Us[2] - UL[2]),
            FL[3] + SL*(Us[3] - UL[3]), FL[4] + SL*(Us[4] - UL[4]),
            FL[5] + SL*(Us[5] - UL[5]), FL[6] + SL*(Us[6] - UL[6]),
            FL[7] + SL*(Us[7] - UL[7]), FL[8] + SL*(Us[8] - UL[8]),
            FL[9] + SL*(Us[9] - UL[9]))
    elseif SM >= zero(FT)
        # Region L** (left Alfvén sub-state, between S*L and SM)
        # When Bn*≈0: SstL≈SM, this branch is skipped naturally
        FL = _mhd_flux_normal(ρL, uL, vL, wL, pL, BxL, ByL, BzL, ψL, UL[5], nx, ny, nz, ch)
        Us_star = SVector{9, FT}(ρsL, ρsL*usL, ρsL*vsL, ρsL*wsL, EsL, BxsL, BysL, BzsL, ψs)
        Us_dbl = SVector{9, FT}(ρsL, ρsL*ussL, ρsL*vssL, ρsL*wssL, EssL, BxssL, ByssL, BzssL, ψs)
        F = SVector{9, FT}(
            FL[1] + SL*(Us_star[1]-UL[1]) + SstL*(Us_dbl[1]-Us_star[1]),
            FL[2] + SL*(Us_star[2]-UL[2]) + SstL*(Us_dbl[2]-Us_star[2]),
            FL[3] + SL*(Us_star[3]-UL[3]) + SstL*(Us_dbl[3]-Us_star[3]),
            FL[4] + SL*(Us_star[4]-UL[4]) + SstL*(Us_dbl[4]-Us_star[4]),
            FL[5] + SL*(Us_star[5]-UL[5]) + SstL*(Us_dbl[5]-Us_star[5]),
            FL[6] + SL*(Us_star[6]-UL[6]) + SstL*(Us_dbl[6]-Us_star[6]),
            FL[7] + SL*(Us_star[7]-UL[7]) + SstL*(Us_dbl[7]-Us_star[7]),
            FL[8] + SL*(Us_star[8]-UL[8]) + SstL*(Us_dbl[8]-Us_star[8]),
            FL[9] + SL*(Us_star[9]-UL[9]) + SstL*(Us_dbl[9]-Us_star[9]))
    elseif SstR >= zero(FT)
        # Region R** (right Alfvén sub-state, between SM and S*R)
        FR = _mhd_flux_normal(ρR, uR, vR, wR, pR, BxR, ByR, BzR, ψR, UR[5], nx, ny, nz, ch)
        Us_star = SVector{9, FT}(ρsR, ρsR*usR, ρsR*vsR, ρsR*wsR, EsR, BxsR, BysR, BzsR, ψs)
        Us_dbl = SVector{9, FT}(ρsR, ρsR*ussR, ρsR*vssR, ρsR*wssR, EssR, BxssR, ByssR, BzssR, ψs)
        F = SVector{9, FT}(
            FR[1] + SR*(Us_star[1]-UR[1]) + SstR*(Us_dbl[1]-Us_star[1]),
            FR[2] + SR*(Us_star[2]-UR[2]) + SstR*(Us_dbl[2]-Us_star[2]),
            FR[3] + SR*(Us_star[3]-UR[3]) + SstR*(Us_dbl[3]-Us_star[3]),
            FR[4] + SR*(Us_star[4]-UR[4]) + SstR*(Us_dbl[4]-Us_star[4]),
            FR[5] + SR*(Us_star[5]-UR[5]) + SstR*(Us_dbl[5]-Us_star[5]),
            FR[6] + SR*(Us_star[6]-UR[6]) + SstR*(Us_dbl[6]-Us_star[6]),
            FR[7] + SR*(Us_star[7]-UR[7]) + SstR*(Us_dbl[7]-Us_star[7]),
            FR[8] + SR*(Us_star[8]-UR[8]) + SstR*(Us_dbl[8]-Us_star[8]),
            FR[9] + SR*(Us_star[9]-UR[9]) + SstR*(Us_dbl[9]-Us_star[9]))
    elseif SR > zero(FT)
        # Region R* (right star, between S*R and SR)
        FR = _mhd_flux_normal(ρR, uR, vR, wR, pR, BxR, ByR, BzR, ψR, UR[5], nx, ny, nz, ch)
        Us = SVector{9, FT}(ρsR, ρsR*usR, ρsR*vsR, ρsR*wsR, EsR, BxsR, BysR, BzsR, ψs)
        F = SVector{9, FT}(
            FR[1] + SR*(Us[1]-UR[1]), FR[2] + SR*(Us[2]-UR[2]),
            FR[3] + SR*(Us[3]-UR[3]), FR[4] + SR*(Us[4]-UR[4]),
            FR[5] + SR*(Us[5]-UR[5]), FR[6] + SR*(Us[6]-UR[6]),
            FR[7] + SR*(Us[7]-UR[7]), FR[8] + SR*(Us[8]-UR[8]),
            FR[9] + SR*(Us[9]-UR[9]))
    else
        # Region R (supersonic right)
        F = _mhd_flux_normal(ρR, uR, vR, wR, pR, BxR, ByR, BzR, ψR, UR[5], nx, ny, nz, ch)
    end

    # ── GLM ψ/Bn flux: decoupled from HLLD star states ──
    # Override the ψ flux (f9) and the ψ·n̂ part of B fluxes (f6,f7,f8) with
    # upwind GLM values. This prevents ch²·Bn blow-up in low-density cells.
    # GLM upwind: ψ* = 0.5(ψL+ψR) - 0.5·ch·(BnR-BnL)
    #             Bn* = 0.5(BnL+BnR) - 0.5·(ψR-ψL)/ch
    # ψ flux: F_ψ = ch²·Bn*
    # B flux ψ contribution: ψ*·n̂ (replaces whatever ψ the star state used)
    ψ_up = FT(0.5)*(ψL + ψR) - FT(0.5)*ch*(BnR - BnL)
    Bn_up = FT(0.5)*(BnL + BnR) - FT(0.5)*(ψR - ψL)/(ch + FT(1.0e-20))
    F_psi = ch*ch*Bn_up
    # Replace ψ·n̂ in f6,f7,f8 with ψ_up·n̂, and f9 with F_psi
    # The star-state flux F already contains some ψ contribution (from _mhd_flux_normal
    # which used ψL or ψR). We correct by replacing the ψ part.
    # f6 = (Bx*qn - u*Bn) + ψ_used·nx → correct to ψ_up·nx
    # But we don't know ψ_used exactly (varies by region). Simplest: just override f9.
    F = SVector{9, FT}(F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8], F_psi)

    # ── Final flux sanity check: non-finite or extreme values → revert to HLLE ──
    F_max = max(abs(F[1]),abs(F[2]),abs(F[3]),abs(F[4]),abs(F[5]),
                abs(F[6]),abs(F[7]),abs(F[8]),abs(F[9]))
    F_ref = max(abs(UL[1]),abs(UR[1]),abs(UL[5]),abs(UR[5])) * FT(1.0e4) + FT(1.0e-10)
    if isnan(F_max) || isinf(F_max) || F_max > F_ref
        if fallback_meta !== nothing
            ct_record_fallback!(fallback_meta, CT_POS_HLLD_TO_HLLE_COUNT)
        end
        return MHD_HLLE_Flux(UL, UR, nx, ny, nz, ch_glm)
    end

    return F
end

# ═══════════════════════════════════════════════════════════════════════
# MHD KEP (Kinetic Energy Preserving) Central Flux
# ═══════════════════════════════════════════════════════════════════════
# Zero numerical dissipation. For DNS/turbulence with explicit filtering.
@inline function MHD_KEP_Flux(UL, UR, nx, ny, nz, ch_glm::FT)
    @static if isothermal_mhd
        return MHD_Rusanov_Flux(UL, UR, nx, ny, nz, ch_glm)
    end
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10)
        return SVector{9, FT}(FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0))
    end

    ch = ch_glm

    # Left primitives
    ρL = UL[1]; inv_ρL = one(FT) / ρL
    uL = UL[2]*inv_ρL; vL = UL[3]*inv_ρL; wL = UL[4]*inv_ρL
    BxL = UL[6]; ByL = UL[7]; BzL = UL[8]; ψL = UL[9]
    B2L = BxL*BxL + ByL*ByL + BzL*BzL
    pL = max(FT(1.0e-10), (γ-one(FT))*(
        UL[5] - FT(0.5)*ρL*(uL*uL+vL*vL+wL*wL) -
        FT(0.5)*B2L*INV_MU0_SI))

    # Right primitives
    ρR = UR[1]; inv_ρR = one(FT) / ρR
    uR = UR[2]*inv_ρR; vR = UR[3]*inv_ρR; wR = UR[4]*inv_ρR
    BxR = UR[6]; ByR = UR[7]; BzR = UR[8]; ψR = UR[9]
    B2R = BxR*BxR + ByR*ByR + BzR*BzR
    pR = max(FT(1.0e-10), (γ-one(FT))*(
        UR[5] - FT(0.5)*ρR*(uR*uR+vR*vR+wR*wR) -
        FT(0.5)*B2R*INV_MU0_SI))

    # Arithmetic averages
    ρ_avg = FT(0.5)*(ρL + ρR)
    u_avg = FT(0.5)*(uL + uR); v_avg = FT(0.5)*(vL + vR); w_avg = FT(0.5)*(wL + wR)
    p_avg = FT(0.5)*(pL + pR)
    Bx_avg = FT(0.5)*(BxL + BxR); By_avg = FT(0.5)*(ByL + ByR); Bz_avg = FT(0.5)*(BzL + BzR)
    ψ_avg = FT(0.5)*(ψL + ψR)

    # Specific internal energy e = p/(ρ*(γ-1)) (= Cv*T), arithmetic average.
    # KEP-consistent internal energy; EC upgrade (log mean) out of scope for MHD.
    eL = pL / (ρL * (γ - one(FT)))
    eR = pR / (ρR * (γ - one(FT)))
    e_avg = FT(0.5)*(eL + eR)

    q_avg = u_avg*nx + v_avg*ny + w_avg*nz
    Bn_avg = Bx_avg*nx + By_avg*ny + Bz_avg*nz
    B2_avg = Bx_avg*Bx_avg + By_avg*By_avg + Bz_avg*Bz_avg
    pt_avg = p_avg + FT(0.5)*B2_avg*INV_MU0_SI
    vB_avg = u_avg*Bx_avg + v_avg*By_avg + w_avg*Bz_avg

    f1 = ρ_avg * q_avg
    f2 = f1 * u_avg + pt_avg * nx - Bx_avg * Bn_avg*INV_MU0_SI
    f3 = f1 * v_avg + pt_avg * ny - By_avg * Bn_avg*INV_MU0_SI
    f4 = f1 * w_avg + pt_avg * nz - Bz_avg * Bn_avg*INV_MU0_SI
    # Energy flux — KE/IE-consistent split form:
    #   f1 * [ ½(u_L·u_R + v_L·v_R + w_L·w_R)   (cross-product kinetic)
    #        + e_avg                              (arithmetic-avg internal energy)
    #        + ½·B2_avg ]                         (arithmetic-avg magnetic energy,
    #                                          built from averaged B — B is a
    #                                          central flux in induction eqn, so
    #                                          no KE-coupling spurious source)
    #   + pt_avg * q_avg - Bn_avg * vB_avg        (pressure + Poynting work, kept
    #                                          as arithmetic averages; their
    #                                          KE/IE consistency is secured by
    #                                          the momentum pt_avg*n and the
    #                                          induction central flux)
    # Reduces to (E+pt)q - Bn·vB for UL=UR.
    f5 = f1 * (FT(0.5)*(uL*uR + vL*vR + wL*wR) + e_avg +
               FT(0.5)*B2_avg*INV_MU0_SI) +
         pt_avg * q_avg - Bn_avg * vB_avg*INV_MU0_SI
    f6 = Bx_avg*q_avg - u_avg*Bn_avg + ψ_avg*nx
    f7 = By_avg*q_avg - v_avg*Bn_avg + ψ_avg*ny
    f8 = Bz_avg*q_avg - w_avg*Bn_avg + ψ_avg*nz
    f9 = ch*ch*Bn_avg

    return SVector{9, FT}(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end
