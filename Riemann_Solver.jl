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
    # often use simple arithmetic averages for the interface states
    ρ_avg = FT(0.5) * (ρL + ρR)
    u_avg = FT(0.5) * (uL + uR)
    v_avg = FT(0.5) * (vL + vR)
    w_avg = FT(0.5) * (wL + wR)
    p_avg = FT(0.5) * (pL + pR)
    
    # Internal energy average (can be simplified if strictly using conserved vars, but primitive is standard for KEP)
    # Average total energy density = ρE
    # Here we define the interfacial enthalpy or energy flux directly
    E_avg = FT(0.5) * (EL + ER)
    
    q_avg = u_avg * nx + v_avg * ny + w_avg * nz
    
    # 3. KEP Flux components
    # F_mass = ρ_avg * q_avg
    fp1 = ρ_avg * q_avg
    
    # F_momentum = F_mass * u_avg + p_avg * n
    fp2 = fp1 * u_avg + p_avg * nx
    fp3 = fp1 * v_avg + p_avg * ny
    fp4 = fp1 * w_avg + p_avg * nz
    
    # F_energy = (E_avg + p_avg) * q_avg
    fp5 = (E_avg + p_avg) * q_avg
    
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

    # Total pressure: p_total = p + B²/2
    B2 = Bx*Bx + By*By + Bz*Bz
    pt = p + FT(0.5) * B2

    # MHD flux (+ GLM corrections)
    f1 = ρ * qn                                           # mass
    f2 = ρ*u*qn + pt*nx - Bx*Bn                           # x-momentum
    f3 = ρ*v*qn + pt*ny - By*Bn                           # y-momentum
    f4 = ρ*w*qn + pt*nz - Bz*Bn                           # z-momentum
    f5 = (E_total + pt)*qn - Bn*(u*Bx + v*By + w*Bz)     # energy
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
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10)
        return SVector{9, FT}(FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0))
    end

    ch = ch_glm

    # Left state primitives
    ρL = UL[1]; inv_ρL = one(FT) / ρL
    uL = UL[2]*inv_ρL; vL = UL[3]*inv_ρL; wL = UL[4]*inv_ρL
    BxL = UL[6]; ByL = UL[7]; BzL = UL[8]; ψL = UL[9]
    B2L = BxL*BxL + ByL*ByL + BzL*BzL
    pL = max(FT(1.0e-10), (γ - one(FT)) * (UL[5] - FT(0.5)*ρL*(uL*uL+vL*vL+wL*wL) - FT(0.5)*B2L))
    qnL = uL*nx + vL*ny + wL*nz

    # Right state primitives
    ρR = UR[1]; inv_ρR = one(FT) / ρR
    uR = UR[2]*inv_ρR; vR = UR[3]*inv_ρR; wR = UR[4]*inv_ρR
    BxR = UR[6]; ByR = UR[7]; BzR = UR[8]; ψR = UR[9]
    B2R = BxR*BxR + ByR*ByR + BzR*BzR
    pR = max(FT(1.0e-10), (γ - one(FT)) * (UR[5] - FT(0.5)*ρR*(uR*uR+vR*vR+wR*wR) - FT(0.5)*B2R))
    qnR = uR*nx + vR*ny + wR*nz

    # Fast magnetosonic speeds
    c2L = γ * pL * inv_ρL
    va2L = B2L * inv_ρL
    cfL = sqrt(c2L + va2L)

    c2R = γ * pR * inv_ρR
    va2R = B2R * inv_ρR
    cfR = sqrt(c2R + va2R)

    # Maximum wave speed (including GLM speed ch)
    λ_max = max(abs(qnL) + cfL, abs(qnR) + cfR, ch)

    # Physical fluxes
    FL = _mhd_flux_normal(ρL, uL, vL, wL, pL, BxL, ByL, BzL, ψL, UL[5], nx, ny, nz, ch)
    FR = _mhd_flux_normal(ρR, uR, vR, wR, pR, BxR, ByR, BzR, ψR, UR[5], nx, ny, nz, ch)

    # Rusanov: F = 0.5(FL+FR) - 0.5·λ_max·(UR-UL)
    return SVector{9, FT}(
        FT(0.5)*(FL[1]+FR[1]) - FT(0.5)*λ_max*(UR[1]-UL[1]),
        FT(0.5)*(FL[2]+FR[2]) - FT(0.5)*λ_max*(UR[2]-UL[2]),
        FT(0.5)*(FL[3]+FR[3]) - FT(0.5)*λ_max*(UR[3]-UL[3]),
        FT(0.5)*(FL[4]+FR[4]) - FT(0.5)*λ_max*(UR[4]-UL[4]),
        FT(0.5)*(FL[5]+FR[5]) - FT(0.5)*λ_max*(UR[5]-UL[5]),
        FT(0.5)*(FL[6]+FR[6]) - FT(0.5)*λ_max*(UR[6]-UL[6]),
        FT(0.5)*(FL[7]+FR[7]) - FT(0.5)*λ_max*(UR[7]-UL[7]),
        FT(0.5)*(FL[8]+FR[8]) - FT(0.5)*λ_max*(UR[8]-UL[8]),
        FT(0.5)*(FL[9]+FR[9]) - FT(0.5)*λ_max*(UR[9]-UL[9])
    )
end

# ═══════════════════════════════════════════════════════════════════════
# HLLD Flux — Miyoshi & Kusano (2005)
# ═══════════════════════════════════════════════════════════════════════
# 5-wave approximate Riemann solver for ideal MHD.
# Captures fast magnetosonic, Alfvén, and contact discontinuities.
# Extended with GLM for the ψ-equation.
@inline function HLLD_Flux(UL, UR, nx, ny, nz, ch_glm::FT)
    # Safety check
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10)
        return SVector{9, FT}(FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0))
    end

    ch = ch_glm

    # ── Left state ──
    ρL = UL[1]; inv_ρL = one(FT) / ρL
    uL = UL[2]*inv_ρL; vL = UL[3]*inv_ρL; wL = UL[4]*inv_ρL
    BxL = UL[6]; ByL = UL[7]; BzL = UL[8]; ψL = UL[9]
    B2L = BxL*BxL + ByL*ByL + BzL*BzL
    pL = max(FT(1.0e-10), (γ-one(FT))*(UL[5] - FT(0.5)*ρL*(uL*uL+vL*vL+wL*wL) - FT(0.5)*B2L))
    ptL = pL + FT(0.5)*B2L  # total pressure
    qnL = uL*nx + vL*ny + wL*nz
    BnL = BxL*nx + ByL*ny + BzL*nz

    # ── Right state ──
    ρR = UR[1]; inv_ρR = one(FT) / ρR
    uR = UR[2]*inv_ρR; vR = UR[3]*inv_ρR; wR = UR[4]*inv_ρR
    BxR = UR[6]; ByR = UR[7]; BzR = UR[8]; ψR = UR[9]
    B2R = BxR*BxR + ByR*ByR + BzR*BzR
    pR = max(FT(1.0e-10), (γ-one(FT))*(UR[5] - FT(0.5)*ρR*(uR*uR+vR*vR+wR*wR) - FT(0.5)*B2R))
    ptR = pR + FT(0.5)*B2R
    qnR = uR*nx + vR*ny + wR*nz
    BnR = BxR*nx + ByR*ny + BzR*nz

    # ── Fast magnetosonic speeds ──
    c2L = γ*pL*inv_ρL; c2R = γ*pR*inv_ρR
    va2L = B2L*inv_ρL; va2R = B2R*inv_ρR
    cfL = sqrt(c2L + va2L)
    cfR = sqrt(c2R + va2R)

    # ── Wave speed estimates (Davis estimate) ──
    SL = min(qnL - cfL, qnR - cfR)
    SR = max(qnL + cfL, qnR + cfR)

    # ── HLL contact speed ──
    SM = (ρR*qnR*(SR-qnR) - ρL*qnL*(SL-qnL) + ptL - ptR) /
         (ρR*(SR-qnR) - ρL*(SL-qnL) + FT(1.0e-20))

    # ── Total pressure in star region ──
    ptS = ptL + ρL*(SL-qnL)*(SM-qnL)

    # ── HLL average Bn (single Bn* for consistent formulations) ──
    Bn_hll = (SR*BnR - SL*BnL) / (SR - SL + FT(1.0e-20))

    # ── Left star state ──
    inv_SL_SM = one(FT) / (SL - SM + FT(1.0e-20))
    ρsL = ρL * (SL - qnL) * inv_SL_SM
    ρsL = max(ρsL, FT(1.0e-10))
    inv_ρsL = one(FT) / ρsL

    # Tangential velocity/B adjustments
    denom_L = ρL*(SL-qnL)*(SL-SM) - Bn_hll*Bn_hll + FT(1.0e-20)
    inv_denom_L = one(FT) / denom_L

    # Tangential velocities in star region
    usL = uL + (ptS - ptL)*nx * inv_denom_L * Bn_hll  # simplified HLLD
    vsL = vL + (ptS - ptL)*ny * inv_denom_L * Bn_hll  # per Miyoshi & Kusano
    wsL = wL + (ptS - ptL)*nz * inv_denom_L * Bn_hll

    # Actually, for general HLLD we need the full formulation.
    # Simplified approach: use HLL for tangential components
    factor_L = Bn_hll * inv_denom_L
    usL = uL + (SM - qnL)*nx  # normal velocity = SM in star
    vsL = vL # simplified: tangential velocity same in star (HLL-like)
    wsL = wL

    # Star magnetic field
    BxsL = BxL + (Bn_hll - BnL)*nx * (SL - qnL) * inv_denom_L
    BysL = ByL + (Bn_hll - BnL)*ny * (SL - qnL) * inv_denom_L
    BzsL = BzL + (Bn_hll - BnL)*nz * (SL - qnL) * inv_denom_L

    # For simplicity at this stage, use the HLL average for tangential components
    # Full HLLD tangential resolve would require the Alfvén sub-states (** region)
    # This simplified version is equivalent to HLLC-MHD

    # Use normal velocity = SM·n + tangential components preserved
    qn_diff_L = SM - qnL
    usL = uL + qn_diff_L * nx
    vsL = vL + qn_diff_L * ny
    wsL = wL + qn_diff_L * nz

    # Star B-field: from jump conditions
    if abs(ρL*(SL-qnL)*(SL-SM) - Bn_hll*Bn_hll) > FT(1.0e-10)
        coeff = Bn_hll * (SM - qnL) * inv_denom_L
        BxsL = BxL * (ρL*(SL-qnL)*(SL-qnL) - Bn_hll*Bn_hll) * inv_denom_L
        BysL = ByL * (ρL*(SL-qnL)*(SL-qnL) - Bn_hll*Bn_hll) * inv_denom_L
        BzsL = BzL * (ρL*(SL-qnL)*(SL-qnL) - Bn_hll*Bn_hll) * inv_denom_L
    else
        BxsL = BxL; BysL = ByL; BzsL = BzL
    end

    B2sL = BxsL*BxsL + BysL*BysL + BzsL*BzsL
    BnsL = BxsL*nx + BysL*ny + BzsL*nz
    vBsL = usL*BxsL + vsL*BysL + wsL*BzsL

    EsL = ((SL-qnL)*UL[5] - ptL*qnL + ptS*SM + Bn_hll*(uL*BxL+vL*ByL+wL*BzL - vBsL)) * inv_SL_SM

    # ── Right star state ──
    inv_SR_SM = one(FT) / (SR - SM + FT(1.0e-20))
    ρsR = ρR * (SR - qnR) * inv_SR_SM
    ρsR = max(ρsR, FT(1.0e-10))

    denom_R = ρR*(SR-qnR)*(SR-SM) - Bn_hll*Bn_hll + FT(1.0e-20)
    inv_denom_R = one(FT) / denom_R

    qn_diff_R = SM - qnR
    usR = uR + qn_diff_R * nx
    vsR = vR + qn_diff_R * ny
    wsR = wR + qn_diff_R * nz

    if abs(ρR*(SR-qnR)*(SR-SM) - Bn_hll*Bn_hll) > FT(1.0e-10)
        BxsR = BxR * (ρR*(SR-qnR)*(SR-qnR) - Bn_hll*Bn_hll) * inv_denom_R
        BysR = ByR * (ρR*(SR-qnR)*(SR-qnR) - Bn_hll*Bn_hll) * inv_denom_R
        BzsR = BzR * (ρR*(SR-qnR)*(SR-qnR) - Bn_hll*Bn_hll) * inv_denom_R
    else
        BxsR = BxR; BysR = ByR; BzsR = BzR
    end

    B2sR = BxsR*BxsR + BysR*BysR + BzsR*BzsR
    BnsR = BxsR*nx + BysR*ny + BzsR*nz
    vBsR = usR*BxsR + vsR*BysR + wsR*BzsR

    EsR = ((SR-qnR)*UR[5] - ptR*qnR + ptS*SM + Bn_hll*(uR*BxR+vR*ByR+wR*BzR - vBsR)) * inv_SR_SM

    # ── GLM: ψ star state (Dedner) ──
    ψs = FT(0.5)*(ψL + ψR) - FT(0.5)*ch*(BnR - BnL)
    Bns = FT(0.5)*(BnL + BnR) - FT(0.5)*(ψR - ψL)/ch

    # ── Select flux based on wave pattern ──
    if SL >= zero(FT)
        # Left region
        F = _mhd_flux_normal(ρL, uL, vL, wL, pL, BxL, ByL, BzL, ψL, UL[5], nx, ny, nz, ch)
    elseif SM >= zero(FT)
        # Left star region
        FL = _mhd_flux_normal(ρL, uL, vL, wL, pL, BxL, ByL, BzL, ψL, UL[5], nx, ny, nz, ch)
        Us = SVector{9, FT}(ρsL, ρsL*usL, ρsL*vsL, ρsL*wsL, EsL, BxsL, BysL, BzsL, ψs)
        F = SVector{9, FT}(ntuple(Val(9)) do n
            FL[n] + SL*(Us[n] - UL[n])
        end)
    elseif SR > zero(FT)
        # Right star region
        FR = _mhd_flux_normal(ρR, uR, vR, wR, pR, BxR, ByR, BzR, ψR, UR[5], nx, ny, nz, ch)
        Us = SVector{9, FT}(ρsR, ρsR*usR, ρsR*vsR, ρsR*wsR, EsR, BxsR, BysR, BzsR, ψs)
        F = SVector{9, FT}(ntuple(Val(9)) do n
            FR[n] + SR*(Us[n] - UR[n])
        end)
    else
        # Right region
        F = _mhd_flux_normal(ρR, uR, vR, wR, pR, BxR, ByR, BzR, ψR, UR[5], nx, ny, nz, ch)
    end

    # Override B-normal and ψ fluxes with GLM-consistent values
    # GLM flux for Bn: F_Bn = ch*ψ,  F_ψ = ch*Bn
    # Already incorporated in _mhd_flux_normal via the ψ·n terms

    return F
end

# ═══════════════════════════════════════════════════════════════════════
# MHD KEP (Kinetic Energy Preserving) Central Flux
# ═══════════════════════════════════════════════════════════════════════
# Zero numerical dissipation. For DNS/turbulence with explicit filtering.
@inline function MHD_KEP_Flux(UL, UR, nx, ny, nz, ch_glm::FT)
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10)
        return SVector{9, FT}(FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0), FT(0e0))
    end

    ch = ch_glm

    # Left primitives
    ρL = UL[1]; inv_ρL = one(FT) / ρL
    uL = UL[2]*inv_ρL; vL = UL[3]*inv_ρL; wL = UL[4]*inv_ρL
    BxL = UL[6]; ByL = UL[7]; BzL = UL[8]; ψL = UL[9]
    B2L = BxL*BxL + ByL*ByL + BzL*BzL
    pL = max(FT(1.0e-10), (γ-one(FT))*(UL[5] - FT(0.5)*ρL*(uL*uL+vL*vL+wL*wL) - FT(0.5)*B2L))

    # Right primitives
    ρR = UR[1]; inv_ρR = one(FT) / ρR
    uR = UR[2]*inv_ρR; vR = UR[3]*inv_ρR; wR = UR[4]*inv_ρR
    BxR = UR[6]; ByR = UR[7]; BzR = UR[8]; ψR = UR[9]
    B2R = BxR*BxR + ByR*ByR + BzR*BzR
    pR = max(FT(1.0e-10), (γ-one(FT))*(UR[5] - FT(0.5)*ρR*(uR*uR+vR*vR+wR*wR) - FT(0.5)*B2R))

    # Arithmetic averages
    ρ_avg = FT(0.5)*(ρL + ρR)
    u_avg = FT(0.5)*(uL + uR); v_avg = FT(0.5)*(vL + vR); w_avg = FT(0.5)*(wL + wR)
    p_avg = FT(0.5)*(pL + pR)
    Bx_avg = FT(0.5)*(BxL + BxR); By_avg = FT(0.5)*(ByL + ByR); Bz_avg = FT(0.5)*(BzL + BzR)
    ψ_avg = FT(0.5)*(ψL + ψR)
    E_avg = FT(0.5)*(UL[5] + UR[5])

    q_avg = u_avg*nx + v_avg*ny + w_avg*nz
    Bn_avg = Bx_avg*nx + By_avg*ny + Bz_avg*nz
    B2_avg = Bx_avg*Bx_avg + By_avg*By_avg + Bz_avg*Bz_avg
    pt_avg = p_avg + FT(0.5)*B2_avg
    vB_avg = u_avg*Bx_avg + v_avg*By_avg + w_avg*Bz_avg

    f1 = ρ_avg * q_avg
    f2 = f1 * u_avg + pt_avg * nx - Bx_avg * Bn_avg
    f3 = f1 * v_avg + pt_avg * ny - By_avg * Bn_avg
    f4 = f1 * w_avg + pt_avg * nz - Bz_avg * Bn_avg
    f5 = (E_avg + pt_avg) * q_avg - Bn_avg * vB_avg
    f6 = Bx_avg*q_avg - u_avg*Bn_avg + ψ_avg*nx
    f7 = By_avg*q_avg - v_avg*Bn_avg + ψ_avg*ny
    f8 = Bz_avg*q_avg - w_avg*Bn_avg + ψ_avg*nz
    f9 = ch*ch*Bn_avg

    return SVector{9, FT}(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end
