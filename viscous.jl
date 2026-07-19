# Viscous flux — Jacobian metric transformation + optional GG non-orthogonal correction
# Strategy: high-order FD in computational space (ξ,η,ζ), mapped to physical space via J^{-T}
# gg_blend=0 → pure Jacobian (default), gg_blend>0 → blend in GG cross-derivative correction
# GG uses a SINGLE merged function to minimize GPU register pressure

@inline function get_viscosity(T::FT)
    return C_s * T * sqrt(T) / (T + T_s)
end

@inline function gradFace(fLLL, fLL, fL, fR, fRR, fRRR, ds)
    if viscous_order == 2
        return (fR - fL) / ds
    elseif viscous_order == 4
        return (FT(1.125e0) * (fR - fL) - FT(0.0416666667e0) * (fRR - fLL)) / ds
    else
        return (FT(1.171875e0) * (fR - fL) - FT(0.0651041667e0) * (fRR - fLL) + FT(0.0046875e0) * (fRRR - fLLL)) / ds
    end
end

@inline function gradCell(fLLL, fLL, fL, fR, fRR, fRRR, ds)
    if viscous_order == 2
        return (fR - fL) / (FT(2.0) * ds)
    elseif viscous_order == 4
        return (FT(0.6666666667e0) * (fR - fL) - FT(0.0833333333e0) * (fRR - fLL)) / ds
    else
        return (FT(0.75) * (fR - fL) - FT(0.15e0) * (fRR - fLL) + FT(0.0166666667e0) * (fRRR - fLLL)) / ds
    end
end


# ─── Merged GG: compute ALL 12 gradient components at ONE cell center ──
# Returns (dudx,dudy,dudz, dvdx,dvdy,dvdz, dwdx,dwdy,dwdz, dTdx,dTdy,dTdz)
# This avoids 4 separate gg_scalar calls → much less register pressure
@inline function gg_cell_all(i, j, k, Q,
        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj, Areak, nxk, nyk, nzk, Vol)
    @inbounds begin
        v = Vol[i,j,k]
        # Cache cell-center values
        uc = Q[i,j,k,2]; vc = Q[i,j,k,3]; wc = Q[i,j,k,4]; Tc = Q[i,j,k,6]

        # ── I-faces ──
        aR = Areai[i+1,j,k]; aL = Areai[i,j,k]
        nxR = nxi[i+1,j,k]*aR; nyR = nyi[i+1,j,k]*aR; nzR = nzi[i+1,j,k]*aR
        nxL = nxi[i,j,k]*aL;   nyL = nyi[i,j,k]*aL;   nzL = nzi[i,j,k]*aL
        # Face-averaged scalars
        uR = FT(0.5)*(uc+Q[i+1,j,k,2]); uL = FT(0.5)*(Q[i-1,j,k,2]+uc)
        vR = FT(0.5)*(vc+Q[i+1,j,k,3]); vL = FT(0.5)*(Q[i-1,j,k,3]+vc)
        wR = FT(0.5)*(wc+Q[i+1,j,k,4]); wL = FT(0.5)*(Q[i-1,j,k,4]+wc)
        TR = FT(0.5)*(Tc+Q[i+1,j,k,6]); TL = FT(0.5)*(Q[i-1,j,k,6]+Tc)
        # Accumulate (φ_R*n*A - φ_L*n*A) for each gradient component
        dudx=uR*nxR-uL*nxL; dudy=uR*nyR-uL*nyL; dudz=uR*nzR-uL*nzL
        dvdx=vR*nxR-vL*nxL; dvdy=vR*nyR-vL*nyL; dvdz=vR*nzR-vL*nzL
        dwdx=wR*nxR-wL*nxL; dwdy=wR*nyR-wL*nyL; dwdz=wR*nzR-wL*nzL
        dTdx=TR*nxR-TL*nxL; dTdy=TR*nyR-TL*nyL; dTdz=TR*nzR-TL*nzL

        # ── J-faces ──
        aR = Areaj[i,j+1,k]; aL = Areaj[i,j,k]
        nxR = nxj[i,j+1,k]*aR; nyR = nyj[i,j+1,k]*aR; nzR = nzj[i,j+1,k]*aR
        nxL = nxj[i,j,k]*aL;   nyL = nyj[i,j,k]*aL;   nzL = nzj[i,j,k]*aL
        uR = FT(0.5)*(uc+Q[i,j+1,k,2]); uL = FT(0.5)*(Q[i,j-1,k,2]+uc)
        vR = FT(0.5)*(vc+Q[i,j+1,k,3]); vL = FT(0.5)*(Q[i,j-1,k,3]+vc)
        wR = FT(0.5)*(wc+Q[i,j+1,k,4]); wL = FT(0.5)*(Q[i,j-1,k,4]+wc)
        TR = FT(0.5)*(Tc+Q[i,j+1,k,6]); TL = FT(0.5)*(Q[i,j-1,k,6]+Tc)
        dudx+=uR*nxR-uL*nxL; dudy+=uR*nyR-uL*nyL; dudz+=uR*nzR-uL*nzL
        dvdx+=vR*nxR-vL*nxL; dvdy+=vR*nyR-vL*nyL; dvdz+=vR*nzR-vL*nzL
        dwdx+=wR*nxR-wL*nxL; dwdy+=wR*nyR-wL*nyL; dwdz+=wR*nzR-wL*nzL
        dTdx+=TR*nxR-TL*nxL; dTdy+=TR*nyR-TL*nyL; dTdz+=TR*nzR-TL*nzL

        # ── K-faces ──
        aR = Areak[i,j,k+1]; aL = Areak[i,j,k]
        nxR = nxk[i,j,k+1]*aR; nyR = nyk[i,j,k+1]*aR; nzR = nzk[i,j,k+1]*aR
        nxL = nxk[i,j,k]*aL;   nyL = nyk[i,j,k]*aL;   nzL = nzk[i,j,k]*aL
        uR = FT(0.5)*(uc+Q[i,j,k+1,2]); uL = FT(0.5)*(Q[i,j,k-1,2]+uc)
        vR = FT(0.5)*(vc+Q[i,j,k+1,3]); vL = FT(0.5)*(Q[i,j,k-1,3]+vc)
        wR = FT(0.5)*(wc+Q[i,j,k+1,4]); wL = FT(0.5)*(Q[i,j,k-1,4]+wc)
        TR = FT(0.5)*(Tc+Q[i,j,k+1,6]); TL = FT(0.5)*(Q[i,j,k-1,6]+Tc)
        dudx+=uR*nxR-uL*nxL; dudy+=uR*nyR-uL*nyL; dudz+=uR*nzR-uL*nzL
        dvdx+=vR*nxR-vL*nxL; dvdy+=vR*nyR-vL*nyL; dvdz+=vR*nzR-vL*nzL
        dwdx+=wR*nxR-wL*nxL; dwdy+=wR*nyR-wL*nyL; dwdz+=wR*nzR-wL*nzL
        dTdx+=TR*nxR-TL*nxL; dTdy+=TR*nyR-TL*nyL; dTdz+=TR*nzR-TL*nzL
    end
    # Multiply by 1/Volume
    return dudx*v,dudy*v,dudz*v, dvdx*v,dvdy*v,dvdz*v, dwdx*v,dwdy*v,dwdz*v, dTdx*v,dTdy*v,dTdz*v
end

# ═══════════════════════════════════════════════════════════════════════
# Viscous flux kernels — ds-based + optional GG correction via gg_blend
# ═══════════════════════════════════════════════════════════════════════
@inline function gradCell2(fL, fR, ds)
    return (fR - fL) / (FT(2e0) * ds)
end

function viscous_flux_i(Q, Fv_x,
        Areai, Areaj, Areak,
        nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk,
        Vol, nxp, nyp, nzp, is_inter, resistive_only)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    @static if ct_mode
        j_lo = NG; j_hi = nyp+NG+1
        k_lo = NG; k_hi = nzp+NG+1
    else
        j_lo = NG+1; j_hi = nyp+NG
        k_lo = NG+1; k_hi = nzp+NG
    end
    if i > nxp+NG || j > j_hi || k > k_hi || i < NG || j < j_lo || k < k_lo
        return
    end
    near_inter_j = (is_inter[3] && j <= NG + 2) || (is_inter[4] && j >= nyp + NG - 1)
    near_inter_k = (is_inter[5] && k <= NG + 2) || (is_inter[6] && k >= nzp + NG - 1)
    iL = i; iR = i + 1
    @inbounds begin
        u_f = FT(0.5)*(Q[iL,j,k,2]+Q[iR,j,k,2]); v_f = FT(0.5)*(Q[iL,j,k,3]+Q[iR,j,k,3])
        w_f = FT(0.5)*(Q[iL,j,k,4]+Q[iR,j,k,4]); T_f = FT(0.5)*(Q[iL,j,k,6]+Q[iR,j,k,6])
        @static if equation_type == :MHD
            Bx_f = FT(0.5)*(Q[iL,j,k,7]+Q[iR,j,k,7])
            By_f = FT(0.5)*(Q[iL,j,k,8]+Q[iR,j,k,8])
            Bz_f = FT(0.5)*(Q[iL,j,k,9]+Q[iR,j,k,9])
        end
        area = Areai[i+1,j,k]; fnx = nxi[i+1,j,k]; fny = nyi[i+1,j,k]; fnz = nzi[i+1,j,k]
        Vf_inv = FT(2.0) / (FT(1.0)/Vol[iL,j,k] + FT(1.0)/Vol[iR,j,k])
        xix = nxi[i+1,j,k] * Areai[i+1,j,k] * Vf_inv
        xiy = nyi[i+1,j,k] * Areai[i+1,j,k] * Vf_inv
        xiz = nzi[i+1,j,k] * Areai[i+1,j,k] * Vf_inv
        etax = FT(0.25) * (nxj[iL,j,k]*Areaj[iL,j,k] + nxj[iL,j+1,k]*Areaj[iL,j+1,k] + nxj[iR,j,k]*Areaj[iR,j,k] + nxj[iR,j+1,k]*Areaj[iR,j+1,k]) * Vf_inv
        etay = FT(0.25) * (nyj[iL,j,k]*Areaj[iL,j,k] + nyj[iL,j+1,k]*Areaj[iL,j+1,k] + nyj[iR,j,k]*Areaj[iR,j,k] + nyj[iR,j+1,k]*Areaj[iR,j+1,k]) * Vf_inv
        etaz = FT(0.25) * (nzj[iL,j,k]*Areaj[iL,j,k] + nzj[iL,j+1,k]*Areaj[iL,j+1,k] + nzj[iR,j,k]*Areaj[iR,j,k] + nzj[iR,j+1,k]*Areaj[iR,j+1,k]) * Vf_inv
        zetax = FT(0.25) * (nxk[iL,j,k]*Areak[iL,j,k] + nxk[iL,j,k+1]*Areak[iL,j,k+1] + nxk[iR,j,k]*Areak[iR,j,k] + nxk[iR,j,k+1]*Areak[iR,j,k+1]) * Vf_inv
        zetay = FT(0.25) * (nyk[iL,j,k]*Areak[iL,j,k] + nyk[iL,j,k+1]*Areak[iL,j,k+1] + nyk[iR,j,k]*Areak[iR,j,k] + nyk[iR,j,k+1]*Areak[iR,j,k+1]) * Vf_inv
        zetaz = FT(0.25) * (nzk[iL,j,k]*Areak[iL,j,k] + nzk[iL,j,k+1]*Areak[iL,j,k+1] + nzk[iR,j,k]*Areak[iR,j,k] + nzk[iR,j,k+1]*Areak[iR,j,k+1]) * Vf_inv
    end
    resistive_flux_active = resistive &&
        (ct_resistive_main_explicit || resistive_only)
    mu = viscous && !resistive_only ? get_viscosity(T_f) : zero(FT); kappa = mu * Cp / Pr
    @inbounds begin
        dudeta = zero(FT); dvdeta = zero(FT); dwdeta = zero(FT); dTdeta = zero(FT)
        dudzeta = zero(FT); dvdzeta = zero(FT); dwdzeta = zero(FT); dTdzeta = zero(FT)
        @static if equation_type == :MHD
            dBxdeta = zero(FT); dBydeta = zero(FT); dBzdeta = zero(FT)
            dBxdzeta = zero(FT); dBydzeta = zero(FT); dBzdzeta = zero(FT)
        end
        dudxi = gradFace(Q[i-2,j,k,2],Q[i-1,j,k,2],Q[i,j,k,2],Q[i+1,j,k,2],Q[i+2,j,k,2],Q[i+3,j,k,2], FT(1.0))
        dvdxi = gradFace(Q[i-2,j,k,3],Q[i-1,j,k,3],Q[i,j,k,3],Q[i+1,j,k,3],Q[i+2,j,k,3],Q[i+3,j,k,3], FT(1.0))
        dwdxi = gradFace(Q[i-2,j,k,4],Q[i-1,j,k,4],Q[i,j,k,4],Q[i+1,j,k,4],Q[i+2,j,k,4],Q[i+3,j,k,4], FT(1.0))
        dTdxi = gradFace(Q[i-2,j,k,6],Q[i-1,j,k,6],Q[i,j,k,6],Q[i+1,j,k,6],Q[i+2,j,k,6],Q[i+3,j,k,6], FT(1.0))
        
        if near_inter_j
            dudetaL=gradCell2(Q[iL,j-1,k,2],Q[iL,j+1,k,2],FT(1.0));dudetaR=gradCell2(Q[iR,j-1,k,2],Q[iR,j+1,k,2],FT(1.0));dudeta=FT(0.5)*(dudetaL+dudetaR)
            dvdetaL=gradCell2(Q[iL,j-1,k,3],Q[iL,j+1,k,3],FT(1.0));dvdetaR=gradCell2(Q[iR,j-1,k,3],Q[iR,j+1,k,3],FT(1.0));dvdeta=FT(0.5)*(dvdetaL+dvdetaR)
            dwdetaL=gradCell2(Q[iL,j-1,k,4],Q[iL,j+1,k,4],FT(1.0));dwdetaR=gradCell2(Q[iR,j-1,k,4],Q[iR,j+1,k,4],FT(1.0));dwdeta=FT(0.5)*(dwdetaL+dwdetaR)
            dTdetaL=gradCell2(Q[iL,j-1,k,6],Q[iL,j+1,k,6],FT(1.0));dTdetaR=gradCell2(Q[iR,j-1,k,6],Q[iR,j+1,k,6],FT(1.0));dTdeta=FT(0.5)*(dTdetaL+dTdetaR)
        else
            dudetaL=gradCell(Q[iL,j-3,k,2],Q[iL,j-2,k,2],Q[iL,j-1,k,2],Q[iL,j+1,k,2],Q[iL,j+2,k,2],Q[iL,j+3,k,2],FT(1.0));dudetaR=gradCell(Q[iR,j-3,k,2],Q[iR,j-2,k,2],Q[iR,j-1,k,2],Q[iR,j+1,k,2],Q[iR,j+2,k,2],Q[iR,j+3,k,2],FT(1.0));dudeta=FT(0.5)*(dudetaL+dudetaR)
            dvdetaL=gradCell(Q[iL,j-3,k,3],Q[iL,j-2,k,3],Q[iL,j-1,k,3],Q[iL,j+1,k,3],Q[iL,j+2,k,3],Q[iL,j+3,k,3],FT(1.0));dvdetaR=gradCell(Q[iR,j-3,k,3],Q[iR,j-2,k,3],Q[iR,j-1,k,3],Q[iR,j+1,k,3],Q[iR,j+2,k,3],Q[iR,j+3,k,3],FT(1.0));dvdeta=FT(0.5)*(dvdetaL+dvdetaR)
            dwdetaL=gradCell(Q[iL,j-3,k,4],Q[iL,j-2,k,4],Q[iL,j-1,k,4],Q[iL,j+1,k,4],Q[iL,j+2,k,4],Q[iL,j+3,k,4],FT(1.0));dwdetaR=gradCell(Q[iR,j-3,k,4],Q[iR,j-2,k,4],Q[iR,j-1,k,4],Q[iR,j+1,k,4],Q[iR,j+2,k,4],Q[iR,j+3,k,4],FT(1.0));dwdeta=FT(0.5)*(dwdetaL+dwdetaR)
            dTdetaL=gradCell(Q[iL,j-3,k,6],Q[iL,j-2,k,6],Q[iL,j-1,k,6],Q[iL,j+1,k,6],Q[iL,j+2,k,6],Q[iL,j+3,k,6],FT(1.0));dTdetaR=gradCell(Q[iR,j-3,k,6],Q[iR,j-2,k,6],Q[iR,j-1,k,6],Q[iR,j+1,k,6],Q[iR,j+2,k,6],Q[iR,j+3,k,6],FT(1.0));dTdeta=FT(0.5)*(dTdetaL+dTdetaR)
        end
        if near_inter_k
            dudzetaL=gradCell2(Q[iL,j,k-1,2],Q[iL,j,k+1,2],FT(1.0));dudzetaR=gradCell2(Q[iR,j,k-1,2],Q[iR,j,k+1,2],FT(1.0));dudzeta=FT(0.5)*(dudzetaL+dudzetaR)
            dvdzetaL=gradCell2(Q[iL,j,k-1,3],Q[iL,j,k+1,3],FT(1.0));dvdzetaR=gradCell2(Q[iR,j,k-1,3],Q[iR,j,k+1,3],FT(1.0));dvdzeta=FT(0.5)*(dvdzetaL+dvdzetaR)
            dwdzetaL=gradCell2(Q[iL,j,k-1,4],Q[iL,j,k+1,4],FT(1.0));dwdzetaR=gradCell2(Q[iR,j,k-1,4],Q[iR,j,k+1,4],FT(1.0));dwdzeta=FT(0.5)*(dwdzetaL+dwdzetaR)
            dTdzetaL=gradCell2(Q[iL,j,k-1,6],Q[iL,j,k+1,6],FT(1.0));dTdzetaR=gradCell2(Q[iR,j,k-1,6],Q[iR,j,k+1,6],FT(1.0));dTdzeta=FT(0.5)*(dTdzetaL+dTdzetaR)
        else
            dudzetaL=gradCell(Q[iL,j,k-3,2],Q[iL,j,k-2,2],Q[iL,j,k-1,2],Q[iL,j,k+1,2],Q[iL,j,k+2,2],Q[iL,j,k+3,2],FT(1.0));dudzetaR=gradCell(Q[iR,j,k-3,2],Q[iR,j,k-2,2],Q[iR,j,k-1,2],Q[iR,j,k+1,2],Q[iR,j,k+2,2],Q[iR,j,k+3,2],FT(1.0));dudzeta=FT(0.5)*(dudzetaL+dudzetaR)
            dvdzetaL=gradCell(Q[iL,j,k-3,3],Q[iL,j,k-2,3],Q[iL,j,k-1,3],Q[iL,j,k+1,3],Q[iL,j,k+2,3],Q[iL,j,k+3,3],FT(1.0));dvdzetaR=gradCell(Q[iR,j,k-3,3],Q[iR,j,k-2,3],Q[iR,j,k-1,3],Q[iR,j,k+1,3],Q[iR,j,k+2,3],Q[iR,j,k+3,3],FT(1.0));dvdzeta=FT(0.5)*(dvdzetaL+dvdzetaR)
            dwdzetaL=gradCell(Q[iL,j,k-3,4],Q[iL,j,k-2,4],Q[iL,j,k-1,4],Q[iL,j,k+1,4],Q[iL,j,k+2,4],Q[iL,j,k+3,4],FT(1.0));dwdzetaR=gradCell(Q[iR,j,k-3,4],Q[iR,j,k-2,4],Q[iR,j,k-1,4],Q[iR,j,k+1,4],Q[iR,j,k+2,4],Q[iR,j,k+3,4],FT(1.0));dwdzeta=FT(0.5)*(dwdzetaL+dwdzetaR)
            dTdzetaL=gradCell(Q[iL,j,k-3,6],Q[iL,j,k-2,6],Q[iL,j,k-1,6],Q[iL,j,k+1,6],Q[iL,j,k+2,6],Q[iL,j,k+3,6],FT(1.0));dTdzetaR=gradCell(Q[iR,j,k-3,6],Q[iR,j,k-2,6],Q[iR,j,k-1,6],Q[iR,j,k+1,6],Q[iR,j,k+2,6],Q[iR,j,k+3,6],FT(1.0));dTdzeta=FT(0.5)*(dTdzetaL+dTdzetaR)
        end
        @static if equation_type == :MHD
            if resistive_flux_active
                dBxdxi = gradFace(Q[i-2,j,k,7],Q[i-1,j,k,7],Q[i,j,k,7],Q[i+1,j,k,7],Q[i+2,j,k,7],Q[i+3,j,k,7], FT(1.0))
                dBydxi = gradFace(Q[i-2,j,k,8],Q[i-1,j,k,8],Q[i,j,k,8],Q[i+1,j,k,8],Q[i+2,j,k,8],Q[i+3,j,k,8], FT(1.0))
                dBzdxi = gradFace(Q[i-2,j,k,9],Q[i-1,j,k,9],Q[i,j,k,9],Q[i+1,j,k,9],Q[i+2,j,k,9],Q[i+3,j,k,9], FT(1.0))

                if near_inter_j
                    dBxdetaL=gradCell2(Q[iL,j-1,k,7],Q[iL,j+1,k,7],FT(1.0));dBxdetaR=gradCell2(Q[iR,j-1,k,7],Q[iR,j+1,k,7],FT(1.0));dBxdeta=FT(0.5)*(dBxdetaL+dBxdetaR)
                    dBydetaL=gradCell2(Q[iL,j-1,k,8],Q[iL,j+1,k,8],FT(1.0));dBydetaR=gradCell2(Q[iR,j-1,k,8],Q[iR,j+1,k,8],FT(1.0));dBydeta=FT(0.5)*(dBydetaL+dBydetaR)
                    dBzdetaL=gradCell2(Q[iL,j-1,k,9],Q[iL,j+1,k,9],FT(1.0));dBzdetaR=gradCell2(Q[iR,j-1,k,9],Q[iR,j+1,k,9],FT(1.0));dBzdeta=FT(0.5)*(dBzdetaL+dBzdetaR)
                else
                    dBxdetaL=gradCell(Q[iL,j-3,k,7],Q[iL,j-2,k,7],Q[iL,j-1,k,7],Q[iL,j+1,k,7],Q[iL,j+2,k,7],Q[iL,j+3,k,7],FT(1.0));dBxdetaR=gradCell(Q[iR,j-3,k,7],Q[iR,j-2,k,7],Q[iR,j-1,k,7],Q[iR,j+1,k,7],Q[iR,j+2,k,7],Q[iR,j+3,k,7],FT(1.0));dBxdeta=FT(0.5)*(dBxdetaL+dBxdetaR)
                    dBydetaL=gradCell(Q[iL,j-3,k,8],Q[iL,j-2,k,8],Q[iL,j-1,k,8],Q[iL,j+1,k,8],Q[iL,j+2,k,8],Q[iL,j+3,k,8],FT(1.0));dBydetaR=gradCell(Q[iR,j-3,k,8],Q[iR,j-2,k,8],Q[iR,j-1,k,8],Q[iR,j+1,k,8],Q[iR,j+2,k,8],Q[iR,j+3,k,8],FT(1.0));dBydeta=FT(0.5)*(dBydetaL+dBydetaR)
                    dBzdetaL=gradCell(Q[iL,j-3,k,9],Q[iL,j-2,k,9],Q[iL,j-1,k,9],Q[iL,j+1,k,9],Q[iL,j+2,k,9],Q[iL,j+3,k,9],FT(1.0));dBzdetaR=gradCell(Q[iR,j-3,k,9],Q[iR,j-2,k,9],Q[iR,j-1,k,9],Q[iR,j+1,k,9],Q[iR,j+2,k,9],Q[iR,j+3,k,9],FT(1.0));dBzdeta=FT(0.5)*(dBzdetaL+dBzdetaR)
                end
                if near_inter_k
                    dBxdzetaL=gradCell2(Q[iL,j,k-1,7],Q[iL,j,k+1,7],FT(1.0));dBxdzetaR=gradCell2(Q[iR,j,k-1,7],Q[iR,j,k+1,7],FT(1.0));dBxdzeta=FT(0.5)*(dBxdzetaL+dBxdzetaR)
                    dBydzetaL=gradCell2(Q[iL,j,k-1,8],Q[iL,j,k+1,8],FT(1.0));dBydzetaR=gradCell2(Q[iR,j,k-1,8],Q[iR,j,k+1,8],FT(1.0));dBydzeta=FT(0.5)*(dBydzetaL+dBydzetaR)
                    dBzdzetaL=gradCell2(Q[iL,j,k-1,9],Q[iL,j,k+1,9],FT(1.0));dBzdzetaR=gradCell2(Q[iR,j,k-1,9],Q[iR,j,k+1,9],FT(1.0));dBzdzeta=FT(0.5)*(dBzdzetaL+dBzdzetaR)
                else
                    dBxdzetaL=gradCell(Q[iL,j,k-3,7],Q[iL,j,k-2,7],Q[iL,j,k-1,7],Q[iL,j,k+1,7],Q[iL,j,k+2,7],Q[iL,j,k+3,7],FT(1.0));dBxdzetaR=gradCell(Q[iR,j,k-3,7],Q[iR,j,k-2,7],Q[iR,j,k-1,7],Q[iR,j,k+1,7],Q[iR,j,k+2,7],Q[iR,j,k+3,7],FT(1.0));dBxdzeta=FT(0.5)*(dBxdzetaL+dBxdzetaR)
                    dBydzetaL=gradCell(Q[iL,j,k-3,8],Q[iL,j,k-2,8],Q[iL,j,k-1,8],Q[iL,j,k+1,8],Q[iL,j,k+2,8],Q[iL,j,k+3,8],FT(1.0));dBydzetaR=gradCell(Q[iR,j,k-3,8],Q[iR,j,k-2,8],Q[iR,j,k-1,8],Q[iR,j,k+1,8],Q[iR,j,k+2,8],Q[iR,j,k+3,8],FT(1.0));dBydzeta=FT(0.5)*(dBydzetaL+dBydzetaR)
                    dBzdzetaL=gradCell(Q[iL,j,k-3,9],Q[iL,j,k-2,9],Q[iL,j,k-1,9],Q[iL,j,k+1,9],Q[iL,j,k+2,9],Q[iL,j,k+3,9],FT(1.0));dBzdzetaR=gradCell(Q[iR,j,k-3,9],Q[iR,j,k-2,9],Q[iR,j,k-1,9],Q[iR,j,k+1,9],Q[iR,j,k+2,9],Q[iR,j,k+3,9],FT(1.0));dBzdzeta=FT(0.5)*(dBzdzetaL+dBzdzetaR)
                end
            end
        end
    end
    dudx = xix*dudxi + etax*dudeta + zetax*dudzeta; dudy = xiy*dudxi + etay*dudeta + zetay*dudzeta; dudz = xiz*dudxi + etaz*dudeta + zetaz*dudzeta
    dvdx = xix*dvdxi + etax*dvdeta + zetax*dvdzeta; dvdy = xiy*dvdxi + etay*dvdeta + zetay*dvdzeta; dvdz = xiz*dvdxi + etaz*dvdeta + zetaz*dvdzeta
    dwdx = xix*dwdxi + etax*dwdeta + zetax*dwdzeta; dwdy = xiy*dwdxi + etay*dwdeta + zetay*dwdzeta; dwdz = xiz*dwdxi + etaz*dwdeta + zetaz*dwdzeta
    dTdx = xix*dTdxi + etax*dTdeta + zetax*dTdzeta; dTdy = xiy*dTdxi + etay*dTdeta + zetay*dTdzeta; dTdz = xiz*dTdxi + etaz*dTdeta + zetaz*dTdzeta
    # GG correction: blend in physical-space gradients for cross-derivatives
    tangential_halo = j < NG+1 || j > nyp+NG || k < NG+1 || k > nzp+NG
    local_gg_blend = (near_inter_j || near_inter_k || tangential_halo) ? zero(FT) : gg_blend
    if local_gg_blend > zero(FT) && iR <= nxp+NG
        # GG at real cell iR (skip if iR is a ghost cell)
        gg = gg_cell_all(iR,j,k, Q, Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,Vol)
        α = local_gg_blend; β = one(FT) - α
        # Only correct cross-derivatives (dudy,dudz,dvdx,dvdz,dwdx,dwdy,dTdy,dTdz)
        # Keep normal derivatives (dudx,dvdx_not,dwdx_not,dTdx) from ds-based (high-order)
        dudy = β*dudy + α*gg[2];  dudz = β*dudz + α*gg[3]
        dvdx = β*dvdx + α*gg[4];  dvdy = β*dvdy + α*gg[5];  dvdz = β*dvdz + α*gg[6]
        dwdx = β*dwdx + α*gg[7];  dwdy = β*dwdy + α*gg[8];  dwdz = β*dwdz + α*gg[9]
        dTdy = β*dTdy + α*gg[11]; dTdz = β*dTdz + α*gg[12]
    end
    divu = dudx+dvdy+dwdz
    tau_xx=mu*(FT(2e0)*dudx-FT(2e0)/FT(3e0)*divu); tau_yy=mu*(FT(2e0)*dvdy-FT(2e0)/FT(3e0)*divu); tau_zz=mu*(FT(2e0)*dwdz-FT(2e0)/FT(3e0)*divu)
    tau_xy=mu*(dudy+dvdx); tau_xz=mu*(dudz+dwdx); tau_yz=mu*(dvdz+dwdy)
    fv_rhou=tau_xx*fnx+tau_xy*fny+tau_xz*fnz; fv_rhov=tau_xy*fnx+tau_yy*fny+tau_yz*fnz; fv_rhow=tau_xz*fnx+tau_yz*fny+tau_zz*fnz
    qx=-kappa*dTdx; qy=-kappa*dTdy; qz=-kappa*dTdz
    @static if equation_type == :MHD
        if resistive_flux_active
            dBxdx = xix*dBxdxi + etax*dBxdeta + zetax*dBxdzeta; dBxdy = xiy*dBxdxi + etay*dBxdeta + zetay*dBxdzeta; dBxdz = xiz*dBxdxi + etaz*dBxdeta + zetaz*dBxdzeta
            dBydx = xix*dBydxi + etax*dBydeta + zetax*dBydzeta; dBydy = xiy*dBydxi + etay*dBydeta + zetay*dBydzeta; dBydz = xiz*dBydxi + etaz*dBydeta + zetaz*dBydzeta
            dBzdx = xix*dBzdxi + etax*dBzdeta + zetax*dBzdzeta; dBzdy = xiy*dBzdxi + etay*dBzdeta + zetay*dBzdzeta; dBzdz = xiz*dBzdxi + etaz*dBzdeta + zetaz*dBzdzeta

            Jx = dBzdy - dBydz
            Jy = dBxdz - dBzdx
            Jz = dBydx - dBxdy

            fres_Bx = η_mhd * (Jy * fnz - Jz * fny)
            fres_By = η_mhd * (Jz * fnx - Jx * fnz)
            fres_Bz = η_mhd * (Jx * fny - Jy * fnx)
            fres_E  = fres_Bx * Bx_f + fres_By * By_f + fres_Bz * Bz_f
        end
    end
    fv_E=(fv_rhou*u_f+fv_rhov*v_f+fv_rhow*w_f)-(qx*fnx+qy*fny+qz*fnz)
    @static if equation_type == :MHD
        if resistive_flux_active
            fv_E += fres_E
        end
    end
    @static if ct_mode
        fi = i-NG+1; fj = j-NG+1; fk = k-NG+1
    else
        fi = i-NG+1; fj = j-NG; fk = k-NG
    end
    @inbounds begin
        Fv_x[fi,fj,fk,1]=FT(0e0); Fv_x[fi,fj,fk,2]=fv_rhou*area
        Fv_x[fi,fj,fk,3]=fv_rhov*area; Fv_x[fi,fj,fk,4]=fv_rhow*area; Fv_x[fi,fj,fk,5]=fv_E*area
        @static if equation_type == :MHD
            Fv_x[fi,fj,fk,6] = resistive_flux_active ? fres_Bx * area : zero(FT)
            Fv_x[fi,fj,fk,7] = resistive_flux_active ? fres_By * area : zero(FT)
            Fv_x[fi,fj,fk,8] = resistive_flux_active ? fres_Bz * area : zero(FT)
            Fv_x[fi,fj,fk,9] = zero(FT)
        end
    end
    return
end


function viscous_flux_j(Q, Fv_y,
        Areai, Areaj, Areak,
        nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk,
        Vol, nxp, nyp, nzp, is_inter, resistive_only)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    @static if ct_mode
        i_lo = NG; i_hi = nxp+NG+1
        k_lo = NG; k_hi = nzp+NG+1
    else
        i_lo = NG+1; i_hi = nxp+NG
        k_lo = NG+1; k_hi = nzp+NG
    end
    if i > i_hi || j > nyp+NG || k > k_hi || i < i_lo || j < NG || k < k_lo
        return
    end
    near_inter_i = (is_inter[1] && i <= NG + 2) || (is_inter[2] && i >= nxp + NG - 1)
    near_inter_k = (is_inter[5] && k <= NG + 2) || (is_inter[6] && k >= nzp + NG - 1)
    jL = j; jR = j + 1
    @inbounds begin
        u_f = FT(0.5)*(Q[i,jL,k,2]+Q[i,jR,k,2]); v_f = FT(0.5)*(Q[i,jL,k,3]+Q[i,jR,k,3])
        w_f = FT(0.5)*(Q[i,jL,k,4]+Q[i,jR,k,4]); T_f = FT(0.5)*(Q[i,jL,k,6]+Q[i,jR,k,6])
        @static if equation_type == :MHD
            Bx_f = FT(0.5)*(Q[i,jL,k,7]+Q[i,jR,k,7])
            By_f = FT(0.5)*(Q[i,jL,k,8]+Q[i,jR,k,8])
            Bz_f = FT(0.5)*(Q[i,jL,k,9]+Q[i,jR,k,9])
        end
        area = Areaj[i,j+1,k]; fnx = nxj[i,j+1,k]; fny = nyj[i,j+1,k]; fnz = nzj[i,j+1,k]
        Vf_inv = FT(2.0) / (FT(1.0)/Vol[i,jL,k] + FT(1.0)/Vol[i,jR,k])
        xix = FT(0.25) * (nxi[i,jL,k]*Areai[i,jL,k] + nxi[i+1,jL,k]*Areai[i+1,jL,k] + nxi[i,jR,k]*Areai[i,jR,k] + nxi[i+1,jR,k]*Areai[i+1,jR,k]) * Vf_inv
        xiy = FT(0.25) * (nyi[i,jL,k]*Areai[i,jL,k] + nyi[i+1,jL,k]*Areai[i+1,jL,k] + nyi[i,jR,k]*Areai[i,jR,k] + nyi[i+1,jR,k]*Areai[i+1,jR,k]) * Vf_inv
        xiz = FT(0.25) * (nzi[i,jL,k]*Areai[i,jL,k] + nzi[i+1,jL,k]*Areai[i+1,jL,k] + nzi[i,jR,k]*Areai[i,jR,k] + nzi[i+1,jR,k]*Areai[i+1,jR,k]) * Vf_inv
        etax = nxj[i,j+1,k] * Areaj[i,j+1,k] * Vf_inv
        etay = nyj[i,j+1,k] * Areaj[i,j+1,k] * Vf_inv
        etaz = nzj[i,j+1,k] * Areaj[i,j+1,k] * Vf_inv
        zetax = FT(0.25) * (nxk[i,jL,k]*Areak[i,jL,k] + nxk[i,jL,k+1]*Areak[i,jL,k+1] + nxk[i,jR,k]*Areak[i,jR,k] + nxk[i,jR,k+1]*Areak[i,jR,k+1]) * Vf_inv
        zetay = FT(0.25) * (nyk[i,jL,k]*Areak[i,jL,k] + nyk[i,jL,k+1]*Areak[i,jL,k+1] + nyk[i,jR,k]*Areak[i,jR,k] + nyk[i,jR,k+1]*Areak[i,jR,k+1]) * Vf_inv
        zetaz = FT(0.25) * (nzk[i,jL,k]*Areak[i,jL,k] + nzk[i,jL,k+1]*Areak[i,jL,k+1] + nzk[i,jR,k]*Areak[i,jR,k] + nzk[i,jR,k+1]*Areak[i,jR,k+1]) * Vf_inv
    end
    resistive_flux_active = resistive &&
        (ct_resistive_main_explicit || resistive_only)
    mu = viscous && !resistive_only ? get_viscosity(T_f) : zero(FT); kappa = mu * Cp / Pr
    @inbounds begin
        dudxi = zero(FT); dvdxi = zero(FT); dwdxi = zero(FT); dTdxi = zero(FT)
        dudzeta = zero(FT); dvdzeta = zero(FT); dwdzeta = zero(FT); dTdzeta = zero(FT)
        @static if equation_type == :MHD
            dBxdxi = zero(FT); dBydxi = zero(FT); dBzdxi = zero(FT)
            dBxdzeta = zero(FT); dBydzeta = zero(FT); dBzdzeta = zero(FT)
        end
        dudeta = gradFace(Q[i,j-2,k,2],Q[i,j-1,k,2],Q[i,j,k,2],Q[i,j+1,k,2],Q[i,j+2,k,2],Q[i,j+3,k,2], FT(1.0))
        dvdeta = gradFace(Q[i,j-2,k,3],Q[i,j-1,k,3],Q[i,j,k,3],Q[i,j+1,k,3],Q[i,j+2,k,3],Q[i,j+3,k,3], FT(1.0))
        dwdeta = gradFace(Q[i,j-2,k,4],Q[i,j-1,k,4],Q[i,j,k,4],Q[i,j+1,k,4],Q[i,j+2,k,4],Q[i,j+3,k,4], FT(1.0))
        dTdeta = gradFace(Q[i,j-2,k,6],Q[i,j-1,k,6],Q[i,j,k,6],Q[i,j+1,k,6],Q[i,j+2,k,6],Q[i,j+3,k,6], FT(1.0))
        
        if near_inter_i
            dudxiL=gradCell2(Q[i-1,jL,k,2],Q[i+1,jL,k,2],FT(1.0));dudxiR=gradCell2(Q[i-1,jR,k,2],Q[i+1,jR,k,2],FT(1.0));dudxi=FT(0.5)*(dudxiL+dudxiR)
            dvdxiL=gradCell2(Q[i-1,jL,k,3],Q[i+1,jL,k,3],FT(1.0));dvdxiR=gradCell2(Q[i-1,jR,k,3],Q[i+1,jR,k,3],FT(1.0));dvdxi=FT(0.5)*(dvdxiL+dvdxiR)
            dwdxiL=gradCell2(Q[i-1,jL,k,4],Q[i+1,jL,k,4],FT(1.0));dwdxiR=gradCell2(Q[i-1,jR,k,4],Q[i+1,jR,k,4],FT(1.0));dwdxi=FT(0.5)*(dwdxiL+dwdxiR)
            dTdxiL=gradCell2(Q[i-1,jL,k,6],Q[i+1,jL,k,6],FT(1.0));dTdxiR=gradCell2(Q[i-1,jR,k,6],Q[i+1,jR,k,6],FT(1.0));dTdxi=FT(0.5)*(dTdxiL+dTdxiR)
        else
            dudxiL=gradCell(Q[i-3,jL,k,2],Q[i-2,jL,k,2],Q[i-1,jL,k,2],Q[i+1,jL,k,2],Q[i+2,jL,k,2],Q[i+3,jL,k,2],FT(1.0));dudxiR=gradCell(Q[i-3,jR,k,2],Q[i-2,jR,k,2],Q[i-1,jR,k,2],Q[i+1,jR,k,2],Q[i+2,jR,k,2],Q[i+3,jR,k,2],FT(1.0));dudxi=FT(0.5)*(dudxiL+dudxiR)
            dvdxiL=gradCell(Q[i-3,jL,k,3],Q[i-2,jL,k,3],Q[i-1,jL,k,3],Q[i+1,jL,k,3],Q[i+2,jL,k,3],Q[i+3,jL,k,3],FT(1.0));dvdxiR=gradCell(Q[i-3,jR,k,3],Q[i-2,jR,k,3],Q[i-1,jR,k,3],Q[i+1,jR,k,3],Q[i+2,jR,k,3],Q[i+3,jR,k,3],FT(1.0));dvdxi=FT(0.5)*(dvdxiL+dvdxiR)
            dwdxiL=gradCell(Q[i-3,jL,k,4],Q[i-2,jL,k,4],Q[i-1,jL,k,4],Q[i+1,jL,k,4],Q[i+2,jL,k,4],Q[i+3,jL,k,4],FT(1.0));dwdxiR=gradCell(Q[i-3,jR,k,4],Q[i-2,jR,k,4],Q[i-1,jR,k,4],Q[i+1,jR,k,4],Q[i+2,jR,k,4],Q[i+3,jR,k,4],FT(1.0));dwdxi=FT(0.5)*(dwdxiL+dwdxiR)
            dTdxiL=gradCell(Q[i-3,jL,k,6],Q[i-2,jL,k,6],Q[i-1,jL,k,6],Q[i+1,jL,k,6],Q[i+2,jL,k,6],Q[i+3,jL,k,6],FT(1.0));dTdxiR=gradCell(Q[i-3,jR,k,6],Q[i-2,jR,k,6],Q[i-1,jR,k,6],Q[i+1,jR,k,6],Q[i+2,jR,k,6],Q[i+3,jR,k,6],FT(1.0));dTdxi=FT(0.5)*(dTdxiL+dTdxiR)
        end
        if near_inter_k
            dudzetaL=gradCell2(Q[i,jL,k-1,2],Q[i,jL,k+1,2],FT(1.0));dudzetaR=gradCell2(Q[i,jR,k-1,2],Q[i,jR,k+1,2],FT(1.0));dudzeta=FT(0.5)*(dudzetaL+dudzetaR)
            dvdzetaL=gradCell2(Q[i,jL,k-1,3],Q[i,jL,k+1,3],FT(1.0));dvdzetaR=gradCell2(Q[i,jR,k-1,3],Q[i,jR,k+1,3],FT(1.0));dvdzeta=FT(0.5)*(dvdzetaL+dvdzetaR)
            dwdzetaL=gradCell2(Q[i,jL,k-1,4],Q[i,jL,k+1,4],FT(1.0));dwdzetaR=gradCell2(Q[i,jR,k-1,4],Q[i,jR,k+1,4],FT(1.0));dwdzeta=FT(0.5)*(dwdzetaL+dwdzetaR)
            dTdzetaL=gradCell2(Q[i,jL,k-1,6],Q[i,jL,k+1,6],FT(1.0));dTdzetaR=gradCell2(Q[i,jR,k-1,6],Q[i,jR,k+1,6],FT(1.0));dTdzeta=FT(0.5)*(dTdzetaL+dTdzetaR)
        else
            dudzetaL=gradCell(Q[i,jL,k-3,2],Q[i,jL,k-2,2],Q[i,jL,k-1,2],Q[i,jL,k+1,2],Q[i,jL,k+2,2],Q[i,jL,k+3,2],FT(1.0));dudzetaR=gradCell(Q[i,jR,k-3,2],Q[i,jR,k-2,2],Q[i,jR,k-1,2],Q[i,jR,k+1,2],Q[i,jR,k+2,2],Q[i,jR,k+3,2],FT(1.0));dudzeta=FT(0.5)*(dudzetaL+dudzetaR)
            dvdzetaL=gradCell(Q[i,jL,k-3,3],Q[i,jL,k-2,3],Q[i,jL,k-1,3],Q[i,jL,k+1,3],Q[i,jL,k+2,3],Q[i,jL,k+3,3],FT(1.0));dvdzetaR=gradCell(Q[i,jR,k-3,3],Q[i,jR,k-2,3],Q[i,jR,k-1,3],Q[i,jR,k+1,3],Q[i,jR,k+2,3],Q[i,jR,k+3,3],FT(1.0));dvdzeta=FT(0.5)*(dvdzetaL+dvdzetaR)
            dwdzetaL=gradCell(Q[i,jL,k-3,4],Q[i,jL,k-2,4],Q[i,jL,k-1,4],Q[i,jL,k+1,4],Q[i,jL,k+2,4],Q[i,jL,k+3,4],FT(1.0));dwdzetaR=gradCell(Q[i,jR,k-3,4],Q[i,jR,k-2,4],Q[i,jR,k-1,4],Q[i,jR,k+1,4],Q[i,jR,k+2,4],Q[i,jR,k+3,4],FT(1.0));dwdzeta=FT(0.5)*(dwdzetaL+dwdzetaR)
            dTdzetaL=gradCell(Q[i,jL,k-3,6],Q[i,jL,k-2,6],Q[i,jL,k-1,6],Q[i,jL,k+1,6],Q[i,jL,k+2,6],Q[i,jL,k+3,6],FT(1.0));dTdzetaR=gradCell(Q[i,jR,k-3,6],Q[i,jR,k-2,6],Q[i,jR,k-1,6],Q[i,jR,k+1,6],Q[i,jR,k+2,6],Q[i,jR,k+3,6],FT(1.0));dTdzeta=FT(0.5)*(dTdzetaL+dTdzetaR)
        end
        @static if equation_type == :MHD
            if resistive_flux_active
                dBxdeta = gradFace(Q[i,j-2,k,7],Q[i,j-1,k,7],Q[i,j,k,7],Q[i,j+1,k,7],Q[i,j+2,k,7],Q[i,j+3,k,7], FT(1.0))
                dBydeta = gradFace(Q[i,j-2,k,8],Q[i,j-1,k,8],Q[i,j,k,8],Q[i,j+1,k,8],Q[i,j+2,k,8],Q[i,j+3,k,8], FT(1.0))
                dBzdeta = gradFace(Q[i,j-2,k,9],Q[i,j-1,k,9],Q[i,j,k,9],Q[i,j+1,k,9],Q[i,j+2,k,9],Q[i,j+3,k,9], FT(1.0))

                if near_inter_i
                    dBxdxiL=gradCell2(Q[i-1,jL,k,7],Q[i+1,jL,k,7],FT(1.0));dBxdxiR=gradCell2(Q[i-1,jR,k,7],Q[i+1,jR,k,7],FT(1.0));dBxdxi=FT(0.5)*(dBxdxiL+dBxdxiR)
                    dBydxiL=gradCell2(Q[i-1,jL,k,8],Q[i+1,jL,k,8],FT(1.0));dBydxiR=gradCell2(Q[i-1,jR,k,8],Q[i+1,jR,k,8],FT(1.0));dBydxi=FT(0.5)*(dBydxiL+dBydxiR)
                    dBzdxiL=gradCell2(Q[i-1,jL,k,9],Q[i+1,jL,k,9],FT(1.0));dBzdxiR=gradCell2(Q[i-1,jR,k,9],Q[i+1,jR,k,9],FT(1.0));dBzdxi=FT(0.5)*(dBzdxiL+dBzdxiR)
                else
                    dBxdxiL=gradCell(Q[i-3,jL,k,7],Q[i-2,jL,k,7],Q[i-1,jL,k,7],Q[i+1,jL,k,7],Q[i+2,jL,k,7],Q[i+3,jL,k,7],FT(1.0));dBxdxiR=gradCell(Q[i-3,jR,k,7],Q[i-2,jR,k,7],Q[i-1,jR,k,7],Q[i+1,jR,k,7],Q[i+2,jR,k,7],Q[i+3,jR,k,7],FT(1.0));dBxdxi=FT(0.5)*(dBxdxiL+dBxdxiR)
                    dBydxiL=gradCell(Q[i-3,jL,k,8],Q[i-2,jL,k,8],Q[i-1,jL,k,8],Q[i+1,jL,k,8],Q[i+2,jL,k,8],Q[i+3,jL,k,8],FT(1.0));dBydxiR=gradCell(Q[i-3,jR,k,8],Q[i-2,jR,k,8],Q[i-1,jR,k,8],Q[i+1,jR,k,8],Q[i+2,jR,k,8],Q[i+3,jR,k,8],FT(1.0));dBydxi=FT(0.5)*(dBydxiL+dBydxiR)
                    dBzdxiL=gradCell(Q[i-3,jL,k,9],Q[i-2,jL,k,9],Q[i-1,jL,k,9],Q[i+1,jL,k,9],Q[i+2,jL,k,9],Q[i+3,jL,k,9],FT(1.0));dBzdxiR=gradCell(Q[i-3,jR,k,9],Q[i-2,jR,k,9],Q[i-1,jR,k,9],Q[i+1,jR,k,9],Q[i+2,jR,k,9],Q[i+3,jR,k,9],FT(1.0));dBzdxi=FT(0.5)*(dBzdxiL+dBzdxiR)
                end
                if near_inter_k
                    dBxdzetaL=gradCell2(Q[i,jL,k-1,7],Q[i,jL,k+1,7],FT(1.0));dBxdzetaR=gradCell2(Q[i,jR,k-1,7],Q[i,jR,k+1,7],FT(1.0));dBxdzeta=FT(0.5)*(dBxdzetaL+dBxdzetaR)
                    dBydzetaL=gradCell2(Q[i,jL,k-1,8],Q[i,jL,k+1,8],FT(1.0));dBydzetaR=gradCell2(Q[i,jR,k-1,8],Q[i,jR,k+1,8],FT(1.0));dBydzeta=FT(0.5)*(dBydzetaL+dBydzetaR)
                    dBzdzetaL=gradCell2(Q[i,jL,k-1,9],Q[i,jL,k+1,9],FT(1.0));dBzdzetaR=gradCell2(Q[i,jR,k-1,9],Q[i,jR,k+1,9],FT(1.0));dBzdzeta=FT(0.5)*(dBzdzetaL+dBzdzetaR)
                else
                    dBxdzetaL=gradCell(Q[i,jL,k-3,7],Q[i,jL,k-2,7],Q[i,jL,k-1,7],Q[i,jL,k+1,7],Q[i,jL,k+2,7],Q[i,jL,k+3,7],FT(1.0));dBxdzetaR=gradCell(Q[i,jR,k-3,7],Q[i,jR,k-2,7],Q[i,jR,k-1,7],Q[i,jR,k+1,7],Q[i,jR,k+2,7],Q[i,jR,k+3,7],FT(1.0));dBxdzeta=FT(0.5)*(dBxdzetaL+dBxdzetaR)
                    dBydzetaL=gradCell(Q[i,jL,k-3,8],Q[i,jL,k-2,8],Q[i,jL,k-1,8],Q[i,jL,k+1,8],Q[i,jL,k+2,8],Q[i,jL,k+3,8],FT(1.0));dBydzetaR=gradCell(Q[i,jR,k-3,8],Q[i,jR,k-2,8],Q[i,jR,k-1,8],Q[i,jR,k+1,8],Q[i,jR,k+2,8],Q[i,jR,k+3,8],FT(1.0));dBydzeta=FT(0.5)*(dBydzetaL+dBydzetaR)
                    dBzdzetaL=gradCell(Q[i,jL,k-3,9],Q[i,jL,k-2,9],Q[i,jL,k-1,9],Q[i,jL,k+1,9],Q[i,jL,k+2,9],Q[i,jL,k+3,9],FT(1.0));dBzdzetaR=gradCell(Q[i,jR,k-3,9],Q[i,jR,k-2,9],Q[i,jR,k-1,9],Q[i,jR,k+1,9],Q[i,jR,k+2,9],Q[i,jR,k+3,9],FT(1.0));dBzdzeta=FT(0.5)*(dBzdzetaL+dBzdzetaR)
                end
            end
        end
    end
    dudx = xix*dudxi + etax*dudeta + zetax*dudzeta; dudy = xiy*dudxi + etay*dudeta + zetay*dudzeta; dudz = xiz*dudxi + etaz*dudeta + zetaz*dudzeta
    dvdx = xix*dvdxi + etax*dvdeta + zetax*dvdzeta; dvdy = xiy*dvdxi + etay*dvdeta + zetay*dvdzeta; dvdz = xiz*dvdxi + etaz*dvdeta + zetaz*dvdzeta
    dwdx = xix*dwdxi + etax*dwdeta + zetax*dwdzeta; dwdy = xiy*dwdxi + etay*dwdeta + zetay*dwdzeta; dwdz = xiz*dwdxi + etaz*dwdeta + zetaz*dwdzeta
    dTdx = xix*dTdxi + etax*dTdeta + zetax*dTdzeta; dTdy = xiy*dTdxi + etay*dTdeta + zetay*dTdzeta; dTdz = xiz*dTdxi + etaz*dTdeta + zetaz*dTdzeta
    # GG correction for cross-derivatives (including wall faces)
    tangential_halo = i < NG+1 || i > nxp+NG || k < NG+1 || k > nzp+NG
    local_gg_blend = (near_inter_i || near_inter_k || tangential_halo) ? zero(FT) : gg_blend
    if local_gg_blend > zero(FT) && jR <= nyp+NG
        # GG at real cell jR (skip if jR is a ghost cell)
        gg = gg_cell_all(i,jR,k, Q, Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,Vol)
        α = local_gg_blend; β = one(FT) - α
        dudx = β*dudx + α*gg[1];  dudz = β*dudz + α*gg[3]
        dvdx = β*dvdx + α*gg[4];  dvdz = β*dvdz + α*gg[6]
        dwdx = β*dwdx + α*gg[7];  dwdz = β*dwdz + α*gg[9]
        dTdx = β*dTdx + α*gg[10]; dTdz = β*dTdz + α*gg[12]
    end
    divu = dudx+dvdy+dwdz
    tau_xx=mu*(FT(2e0)*dudx-FT(2e0)/FT(3e0)*divu); tau_yy=mu*(FT(2e0)*dvdy-FT(2e0)/FT(3e0)*divu); tau_zz=mu*(FT(2e0)*dwdz-FT(2e0)/FT(3e0)*divu)
    tau_xy=mu*(dudy+dvdx); tau_xz=mu*(dudz+dwdx); tau_yz=mu*(dvdz+dwdy)
    fv_rhou=tau_xx*fnx+tau_xy*fny+tau_xz*fnz; fv_rhov=tau_xy*fnx+tau_yy*fny+tau_yz*fnz; fv_rhow=tau_xz*fnx+tau_yz*fny+tau_zz*fnz
    qx=-kappa*dTdx; qy=-kappa*dTdy; qz=-kappa*dTdz
    @static if equation_type == :MHD
        if resistive_flux_active
            dBxdx = xix*dBxdxi + etax*dBxdeta + zetax*dBxdzeta; dBxdy = xiy*dBxdxi + etay*dBxdeta + zetay*dBxdzeta; dBxdz = xiz*dBxdxi + etaz*dBxdeta + zetaz*dBxdzeta
            dBydx = xix*dBydxi + etax*dBydeta + zetax*dBydzeta; dBydy = xiy*dBydxi + etay*dBydeta + zetay*dBydzeta; dBydz = xiz*dBydxi + etaz*dBydeta + zetaz*dBydzeta
            dBzdx = xix*dBzdxi + etax*dBzdeta + zetax*dBzdzeta; dBzdy = xiy*dBzdxi + etay*dBzdeta + zetay*dBzdzeta; dBzdz = xiz*dBzdxi + etaz*dBzdeta + zetaz*dBzdzeta

            Jx = dBzdy - dBydz
            Jy = dBxdz - dBzdx
            Jz = dBydx - dBxdy

            fres_Bx = η_mhd * (Jy * fnz - Jz * fny)
            fres_By = η_mhd * (Jz * fnx - Jx * fnz)
            fres_Bz = η_mhd * (Jx * fny - Jy * fnx)
            fres_E  = fres_Bx * Bx_f + fres_By * By_f + fres_Bz * Bz_f
        end
    end
    fv_E=(fv_rhou*u_f+fv_rhov*v_f+fv_rhow*w_f)-(qx*fnx+qy*fny+qz*fnz)
    @static if equation_type == :MHD
        if resistive_flux_active
            fv_E += fres_E
        end
    end
    @static if ct_mode
        fi = i-NG+1; fj = j-NG+1; fk = k-NG+1
    else
        fi = i-NG; fj = j-NG+1; fk = k-NG
    end
    @inbounds begin
        Fv_y[fi,fj,fk,1]=FT(0e0); Fv_y[fi,fj,fk,2]=fv_rhou*area
        Fv_y[fi,fj,fk,3]=fv_rhov*area; Fv_y[fi,fj,fk,4]=fv_rhow*area; Fv_y[fi,fj,fk,5]=fv_E*area
        @static if equation_type == :MHD
            Fv_y[fi,fj,fk,6] = resistive_flux_active ? fres_Bx * area : zero(FT)
            Fv_y[fi,fj,fk,7] = resistive_flux_active ? fres_By * area : zero(FT)
            Fv_y[fi,fj,fk,8] = resistive_flux_active ? fres_Bz * area : zero(FT)
            Fv_y[fi,fj,fk,9] = zero(FT)
        end
    end
    return
end


function viscous_flux_k(Q, Fv_z,
        Areai, Areaj, Areak,
        nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk,
        Vol, nxp, nyp, nzp, is_inter, resistive_only)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    @static if ct_mode
        i_lo = NG; i_hi = nxp+NG+1
        j_lo = NG; j_hi = nyp+NG+1
    else
        i_lo = NG+1; i_hi = nxp+NG
        j_lo = NG+1; j_hi = nyp+NG
    end
    if i > i_hi || j > j_hi || k > nzp+NG || i < i_lo || j < j_lo || k < NG
        return
    end
    near_inter_i = (is_inter[1] && i <= NG + 2) || (is_inter[2] && i >= nxp + NG - 1)
    near_inter_j = (is_inter[3] && j <= NG + 2) || (is_inter[4] && j >= nyp + NG - 1)
    kL = k; kR = k + 1
    @inbounds begin
        u_f = FT(0.5)*(Q[i,j,kL,2]+Q[i,j,kR,2]); v_f = FT(0.5)*(Q[i,j,kL,3]+Q[i,j,kR,3])
        w_f = FT(0.5)*(Q[i,j,kL,4]+Q[i,j,kR,4]); T_f = FT(0.5)*(Q[i,j,kL,6]+Q[i,j,kR,6])
        @static if equation_type == :MHD
            Bx_f = FT(0.5)*(Q[i,j,kL,7]+Q[i,j,kR,7])
            By_f = FT(0.5)*(Q[i,j,kL,8]+Q[i,j,kR,8])
            Bz_f = FT(0.5)*(Q[i,j,kL,9]+Q[i,j,kR,9])
        end
        area = Areak[i,j,k+1]; fnx = nxk[i,j,k+1]; fny = nyk[i,j,k+1]; fnz = nzk[i,j,k+1]
        Vf_inv = FT(2.0) / (FT(1.0)/Vol[i,j,kL] + FT(1.0)/Vol[i,j,kR])
        xix = FT(0.25) * (nxi[i,j,kL]*Areai[i,j,kL] + nxi[i+1,j,kL]*Areai[i+1,j,kL] + nxi[i,j,kR]*Areai[i,j,kR] + nxi[i+1,j,kR]*Areai[i+1,j,kR]) * Vf_inv
        xiy = FT(0.25) * (nyi[i,j,kL]*Areai[i,j,kL] + nyi[i+1,j,kL]*Areai[i+1,j,kL] + nyi[i,j,kR]*Areai[i,j,kR] + nyi[i+1,j,kR]*Areai[i+1,j,kR]) * Vf_inv
        xiz = FT(0.25) * (nzi[i,j,kL]*Areai[i,j,kL] + nzi[i+1,j,kL]*Areai[i+1,j,kL] + nzi[i,j,kR]*Areai[i,j,kR] + nzi[i+1,j,kR]*Areai[i+1,j,kR]) * Vf_inv
        etax = FT(0.25) * (nxj[i,j,kL]*Areaj[i,j,kL] + nxj[i,j+1,kL]*Areaj[i,j+1,kL] + nxj[i,j,kR]*Areaj[i,j,kR] + nxj[i,j+1,kR]*Areaj[i,j+1,kR]) * Vf_inv
        etay = FT(0.25) * (nyj[i,j,kL]*Areaj[i,j,kL] + nyj[i,j+1,kL]*Areaj[i,j+1,kL] + nyj[i,j,kR]*Areaj[i,j,kR] + nyj[i,j+1,kR]*Areaj[i,j+1,kR]) * Vf_inv
        etaz = FT(0.25) * (nzj[i,j,kL]*Areaj[i,j,kL] + nzj[i,j+1,kL]*Areaj[i,j+1,kL] + nzj[i,j,kR]*Areaj[i,j,kR] + nzj[i,j+1,kR]*Areaj[i,j+1,kR]) * Vf_inv
        zetax = nxk[i,j,k+1] * Areak[i,j,k+1] * Vf_inv
        zetay = nyk[i,j,k+1] * Areak[i,j,k+1] * Vf_inv
        zetaz = nzk[i,j,k+1] * Areak[i,j,k+1] * Vf_inv
    end
    resistive_flux_active = resistive &&
        (ct_resistive_main_explicit || resistive_only)
    mu = viscous && !resistive_only ? get_viscosity(T_f) : zero(FT); kappa = mu * Cp / Pr
    @inbounds begin
        dudxi = zero(FT); dvdxi = zero(FT); dwdxi = zero(FT); dTdxi = zero(FT)
        dudeta = zero(FT); dvdeta = zero(FT); dwdeta = zero(FT); dTdeta = zero(FT)
        @static if equation_type == :MHD
            dBxdxi = zero(FT); dBydxi = zero(FT); dBzdxi = zero(FT)
            dBxdeta = zero(FT); dBydeta = zero(FT); dBzdeta = zero(FT)
        end
        dudzeta = gradFace(Q[i,j,k-2,2],Q[i,j,k-1,2],Q[i,j,k,2],Q[i,j,k+1,2],Q[i,j,k+2,2],Q[i,j,k+3,2], FT(1.0))
        dvdzeta = gradFace(Q[i,j,k-2,3],Q[i,j,k-1,3],Q[i,j,k,3],Q[i,j,k+1,3],Q[i,j,k+2,3],Q[i,j,k+3,3], FT(1.0))
        dwdzeta = gradFace(Q[i,j,k-2,4],Q[i,j,k-1,4],Q[i,j,k,4],Q[i,j,k+1,4],Q[i,j,k+2,4],Q[i,j,k+3,4], FT(1.0))
        dTdzeta = gradFace(Q[i,j,k-2,6],Q[i,j,k-1,6],Q[i,j,k,6],Q[i,j,k+1,6],Q[i,j,k+2,6],Q[i,j,k+3,6], FT(1.0))
        
        if near_inter_i
            dudxiL=gradCell2(Q[i-1,j,kL,2],Q[i+1,j,kL,2],FT(1.0));dudxiR=gradCell2(Q[i-1,j,kR,2],Q[i+1,j,kR,2],FT(1.0));dudxi=FT(0.5)*(dudxiL+dudxiR)
            dvdxiL=gradCell2(Q[i-1,j,kL,3],Q[i+1,j,kL,3],FT(1.0));dvdxiR=gradCell2(Q[i-1,j,kR,3],Q[i+1,j,kR,3],FT(1.0));dvdxi=FT(0.5)*(dvdxiL+dvdxiR)
            dwdxiL=gradCell2(Q[i-1,j,kL,4],Q[i+1,j,kL,4],FT(1.0));dwdxiR=gradCell2(Q[i-1,j,kR,4],Q[i+1,j,kR,4],FT(1.0));dwdxi=FT(0.5)*(dwdxiL+dwdxiR)
            dTdxiL=gradCell2(Q[i-1,j,kL,6],Q[i+1,j,kL,6],FT(1.0));dTdxiR=gradCell2(Q[i-1,j,kR,6],Q[i+1,j,kR,6],FT(1.0));dTdxi=FT(0.5)*(dTdxiL+dTdxiR)
        else
            dudxiL=gradCell(Q[i-3,j,kL,2],Q[i-2,j,kL,2],Q[i-1,j,kL,2],Q[i+1,j,kL,2],Q[i+2,j,kL,2],Q[i+3,j,kL,2],FT(1.0));dudxiR=gradCell(Q[i-3,j,kR,2],Q[i-2,j,kR,2],Q[i-1,j,kR,2],Q[i+1,j,kR,2],Q[i+2,j,kR,2],Q[i+3,j,kR,2],FT(1.0));dudxi=FT(0.5)*(dudxiL+dudxiR)
            dvdxiL=gradCell(Q[i-3,j,kL,3],Q[i-2,j,kL,3],Q[i-1,j,kL,3],Q[i+1,j,kL,3],Q[i+2,j,kL,3],Q[i+3,j,kL,3],FT(1.0));dvdxiR=gradCell(Q[i-3,j,kR,3],Q[i-2,j,kR,3],Q[i-1,j,kR,3],Q[i+1,j,kR,3],Q[i+2,j,kR,3],Q[i+3,j,kR,3],FT(1.0));dvdxi=FT(0.5)*(dvdxiL+dvdxiR)
            dwdxiL=gradCell(Q[i-3,j,kL,4],Q[i-2,j,kL,4],Q[i-1,j,kL,4],Q[i+1,j,kL,4],Q[i+2,j,kL,4],Q[i+3,j,kL,4],FT(1.0));dwdxiR=gradCell(Q[i-3,j,kR,4],Q[i-2,j,kR,4],Q[i-1,j,kR,4],Q[i+1,j,kR,4],Q[i+2,j,kR,4],Q[i+3,j,kR,4],FT(1.0));dwdxi=FT(0.5)*(dwdxiL+dwdxiR)
            dTdxiL=gradCell(Q[i-3,j,kL,6],Q[i-2,j,kL,6],Q[i-1,j,kL,6],Q[i+1,j,kL,6],Q[i+2,j,kL,6],Q[i+3,j,kL,6],FT(1.0));dTdxiR=gradCell(Q[i-3,j,kR,6],Q[i-2,j,kR,6],Q[i-1,j,kR,6],Q[i+1,j,kR,6],Q[i+2,j,kR,6],Q[i+3,j,kR,6],FT(1.0));dTdxi=FT(0.5)*(dTdxiL+dTdxiR)
        end
        if near_inter_j
            dudetaL=gradCell2(Q[i,j-1,kL,2],Q[i,j+1,kL,2],FT(1.0));dudetaR=gradCell2(Q[i,j-1,kR,2],Q[i,j+1,kR,2],FT(1.0));dudeta=FT(0.5)*(dudetaL+dudetaR)
            dvdetaL=gradCell2(Q[i,j-1,kL,3],Q[i,j+1,kL,3],FT(1.0));dvdetaR=gradCell2(Q[i,j-1,kR,3],Q[i,j+1,kR,3],FT(1.0));dvdeta=FT(0.5)*(dvdetaL+dvdetaR)
            dwdetaL=gradCell2(Q[i,j-1,kL,4],Q[i,j+1,kL,4],FT(1.0));dwdetaR=gradCell2(Q[i,j-1,kR,4],Q[i,j+1,kR,4],FT(1.0));dwdeta=FT(0.5)*(dwdetaL+dwdetaR)
            dTdetaL=gradCell2(Q[i,j-1,kL,6],Q[i,j+1,kL,6],FT(1.0));dTdetaR=gradCell2(Q[i,j-1,kR,6],Q[i,j+1,kR,6],FT(1.0));dTdeta=FT(0.5)*(dTdetaL+dTdetaR)
        else
            dudetaL=gradCell(Q[i,j-3,kL,2],Q[i,j-2,kL,2],Q[i,j-1,kL,2],Q[i,j+1,kL,2],Q[i,j+2,kL,2],Q[i,j+3,kL,2],FT(1.0));dudetaR=gradCell(Q[i,j-3,kR,2],Q[i,j-2,kR,2],Q[i,j-1,kR,2],Q[i,j+1,kR,2],Q[i,j+2,kR,2],Q[i,j+3,kR,2],FT(1.0));dudeta=FT(0.5)*(dudetaL+dudetaR)
            dvdetaL=gradCell(Q[i,j-3,kL,3],Q[i,j-2,kL,3],Q[i,j-1,kL,3],Q[i,j+1,kL,3],Q[i,j+2,kL,3],Q[i,j+3,kL,3],FT(1.0));dvdetaR=gradCell(Q[i,j-3,kR,3],Q[i,j-2,kR,3],Q[i,j-1,kR,3],Q[i,j+1,kR,3],Q[i,j+2,kR,3],Q[i,j+3,kR,3],FT(1.0));dvdeta=FT(0.5)*(dvdetaL+dvdetaR)
            dwdetaL=gradCell(Q[i,j-3,kL,4],Q[i,j-2,kL,4],Q[i,j-1,kL,4],Q[i,j+1,kL,4],Q[i,j+2,kL,4],Q[i,j+3,kL,4],FT(1.0));dwdetaR=gradCell(Q[i,j-3,kR,4],Q[i,j-2,kR,4],Q[i,j-1,kR,4],Q[i,j+1,kR,4],Q[i,j+2,kR,4],Q[i,j+3,kR,4],FT(1.0));dwdeta=FT(0.5)*(dwdetaL+dwdetaR)
            dTdetaL=gradCell(Q[i,j-3,kL,6],Q[i,j-2,kL,6],Q[i,j-1,kL,6],Q[i,j+1,kL,6],Q[i,j+2,kL,6],Q[i,j+3,kL,6],FT(1.0));dTdetaR=gradCell(Q[i,j-3,kR,6],Q[i,j-2,kR,6],Q[i,j-1,kR,6],Q[i,j+1,kR,6],Q[i,j+2,kR,6],Q[i,j+3,kR,6],FT(1.0));dTdeta=FT(0.5)*(dTdetaL+dTdetaR)
        end
        @static if equation_type == :MHD
            if resistive_flux_active
                dBxdzeta = gradFace(Q[i,j,k-2,7],Q[i,j,k-1,7],Q[i,j,k,7],Q[i,j,k+1,7],Q[i,j,k+2,7],Q[i,j,k+3,7], FT(1.0))
                dBydzeta = gradFace(Q[i,j,k-2,8],Q[i,j,k-1,8],Q[i,j,k,8],Q[i,j,k+1,8],Q[i,j,k+2,8],Q[i,j,k+3,8], FT(1.0))
                dBzdzeta = gradFace(Q[i,j,k-2,9],Q[i,j,k-1,9],Q[i,j,k,9],Q[i,j,k+1,9],Q[i,j,k+2,9],Q[i,j,k+3,9], FT(1.0))

                if near_inter_i
                    dBxdxiL=gradCell2(Q[i-1,j,kL,7],Q[i+1,j,kL,7],FT(1.0));dBxdxiR=gradCell2(Q[i-1,j,kR,7],Q[i+1,j,kR,7],FT(1.0));dBxdxi=FT(0.5)*(dBxdxiL+dBxdxiR)
                    dBydxiL=gradCell2(Q[i-1,j,kL,8],Q[i+1,j,kL,8],FT(1.0));dBydxiR=gradCell2(Q[i-1,j,kR,8],Q[i+1,j,kR,8],FT(1.0));dBydxi=FT(0.5)*(dBydxiL+dBydxiR)
                    dBzdxiL=gradCell2(Q[i-1,j,kL,9],Q[i+1,j,kL,9],FT(1.0));dBzdxiR=gradCell2(Q[i-1,j,kR,9],Q[i+1,j,kR,9],FT(1.0));dBzdxi=FT(0.5)*(dBzdxiL+dBzdxiR)
                else
                    dBxdxiL=gradCell(Q[i-3,j,kL,7],Q[i-2,j,kL,7],Q[i-1,j,kL,7],Q[i+1,j,kL,7],Q[i+2,j,kL,7],Q[i+3,j,kL,7],FT(1.0));dBxdxiR=gradCell(Q[i-3,j,kR,7],Q[i-2,j,kR,7],Q[i-1,j,kR,7],Q[i+1,j,kR,7],Q[i+2,j,kR,7],Q[i+3,j,kR,7],FT(1.0));dBxdxi=FT(0.5)*(dBxdxiL+dBxdxiR)
                    dBydxiL=gradCell(Q[i-3,j,kL,8],Q[i-2,j,kL,8],Q[i-1,j,kL,8],Q[i+1,j,kL,8],Q[i+2,j,kL,8],Q[i+3,j,kL,8],FT(1.0));dBydxiR=gradCell(Q[i-3,j,kR,8],Q[i-2,j,kR,8],Q[i-1,j,kR,8],Q[i+1,j,kR,8],Q[i+2,j,kR,8],Q[i+3,j,kR,8],FT(1.0));dBydxi=FT(0.5)*(dBydxiL+dBydxiR)
                    dBzdxiL=gradCell(Q[i-3,j,kL,9],Q[i-2,j,kL,9],Q[i-1,j,kL,9],Q[i+1,j,kL,9],Q[i+2,j,kL,9],Q[i+3,j,kL,9],FT(1.0));dBzdxiR=gradCell(Q[i-3,j,kR,9],Q[i-2,j,kR,9],Q[i-1,j,kR,9],Q[i+1,j,kR,9],Q[i+2,j,kR,9],Q[i+3,j,kR,9],FT(1.0));dBzdxi=FT(0.5)*(dBzdxiL+dBzdxiR)
                end
                if near_inter_j
                    dBxdetaL=gradCell2(Q[i,j-1,kL,7],Q[i,j+1,kL,7],FT(1.0));dBxdetaR=gradCell2(Q[i,j-1,kR,7],Q[i,j+1,kR,7],FT(1.0));dBxdeta=FT(0.5)*(dBxdetaL+dBxdetaR)
                    dBydetaL=gradCell2(Q[i,j-1,kL,8],Q[i,j+1,kL,8],FT(1.0));dBydetaR=gradCell2(Q[i,j-1,kR,8],Q[i,j+1,kR,8],FT(1.0));dBydeta=FT(0.5)*(dBydetaL+dBydetaR)
                    dBzdetaL=gradCell2(Q[i,j-1,kL,9],Q[i,j+1,kL,9],FT(1.0));dBzdetaR=gradCell2(Q[i,j-1,kR,9],Q[i,j+1,kR,9],FT(1.0));dBzdeta=FT(0.5)*(dBzdetaL+dBzdetaR)
                else
                    dBxdetaL=gradCell(Q[i,j-3,kL,7],Q[i,j-2,kL,7],Q[i,j-1,kL,7],Q[i,j+1,kL,7],Q[i,j+2,kL,7],Q[i,j+3,kL,7],FT(1.0));dBxdetaR=gradCell(Q[i,j-3,kR,7],Q[i,j-2,kR,7],Q[i,j-1,kR,7],Q[i,j+1,kR,7],Q[i,j+2,kR,7],Q[i,j+3,kR,7],FT(1.0));dBxdeta=FT(0.5)*(dBxdetaL+dBxdetaR)
                    dBydetaL=gradCell(Q[i,j-3,kL,8],Q[i,j-2,kL,8],Q[i,j-1,kL,8],Q[i,j+1,kL,8],Q[i,j+2,kL,8],Q[i,j+3,kL,8],FT(1.0));dBydetaR=gradCell(Q[i,j-3,kR,8],Q[i,j-2,kR,8],Q[i,j-1,kR,8],Q[i,j+1,kR,8],Q[i,j+2,kR,8],Q[i,j+3,kR,8],FT(1.0));dBydeta=FT(0.5)*(dBydetaL+dBydetaR)
                    dBzdetaL=gradCell(Q[i,j-3,kL,9],Q[i,j-2,kL,9],Q[i,j-1,kL,9],Q[i,j+1,kL,9],Q[i,j+2,kL,9],Q[i,j+3,kL,9],FT(1.0));dBzdetaR=gradCell(Q[i,j-3,kR,9],Q[i,j-2,kR,9],Q[i,j-1,kR,9],Q[i,j+1,kR,9],Q[i,j+2,kR,9],Q[i,j+3,kR,9],FT(1.0));dBzdeta=FT(0.5)*(dBzdetaL+dBzdetaR)
                end
            end
        end
    end
    dudx = xix*dudxi + etax*dudeta + zetax*dudzeta; dudy = xiy*dudxi + etay*dudeta + zetay*dudzeta; dudz = xiz*dudxi + etaz*dudeta + zetaz*dudzeta
    dvdx = xix*dvdxi + etax*dvdeta + zetax*dvdzeta; dvdy = xiy*dvdxi + etay*dvdeta + zetay*dvdzeta; dvdz = xiz*dvdxi + etaz*dvdeta + zetaz*dvdzeta
    dwdx = xix*dwdxi + etax*dwdeta + zetax*dwdzeta; dwdy = xiy*dwdxi + etay*dwdeta + zetay*dwdzeta; dwdz = xiz*dwdxi + etaz*dwdeta + zetaz*dwdzeta
    dTdx = xix*dTdxi + etax*dTdeta + zetax*dTdzeta; dTdy = xiy*dTdxi + etay*dTdeta + zetay*dTdzeta; dTdz = xiz*dTdxi + etaz*dTdeta + zetaz*dTdzeta
    # GG correction for cross-derivatives (including wall faces)
    tangential_halo = i < NG+1 || i > nxp+NG || j < NG+1 || j > nyp+NG
    local_gg_blend = (near_inter_i || near_inter_j || tangential_halo) ? zero(FT) : gg_blend
    if local_gg_blend > zero(FT) && kR <= nzp+NG
        # GG at real cell kR (skip if kR is a ghost cell)
        gg = gg_cell_all(i,j,kR, Q, Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,Vol)
        α = local_gg_blend; β = one(FT) - α
        dudx = β*dudx + α*gg[1];  dudy = β*dudy + α*gg[2]
        dvdx = β*dvdx + α*gg[4];  dvdy = β*dvdy + α*gg[5]
        dwdx = β*dwdx + α*gg[7];  dwdy = β*dwdy + α*gg[8]
        dTdx = β*dTdx + α*gg[10]; dTdy = β*dTdy + α*gg[11]
    end
    divu = dudx+dvdy+dwdz
    tau_xx=mu*(FT(2e0)*dudx-FT(2e0)/FT(3e0)*divu); tau_yy=mu*(FT(2e0)*dvdy-FT(2e0)/FT(3e0)*divu); tau_zz=mu*(FT(2e0)*dwdz-FT(2e0)/FT(3e0)*divu)
    tau_xy=mu*(dudy+dvdx); tau_xz=mu*(dudz+dwdx); tau_yz=mu*(dvdz+dwdy)
    fv_rhou=tau_xx*fnx+tau_xy*fny+tau_xz*fnz; fv_rhov=tau_xy*fnx+tau_yy*fny+tau_yz*fnz; fv_rhow=tau_xz*fnx+tau_yz*fny+tau_zz*fnz
    qx=-kappa*dTdx; qy=-kappa*dTdy; qz=-kappa*dTdz
    @static if equation_type == :MHD
        if resistive_flux_active
            dBxdx = xix*dBxdxi + etax*dBxdeta + zetax*dBxdzeta; dBxdy = xiy*dBxdxi + etay*dBxdeta + zetay*dBxdzeta; dBxdz = xiz*dBxdxi + etaz*dBxdeta + zetaz*dBxdzeta
            dBydx = xix*dBydxi + etax*dBydeta + zetax*dBydzeta; dBydy = xiy*dBydxi + etay*dBydeta + zetay*dBydzeta; dBydz = xiz*dBydxi + etaz*dBydeta + zetaz*dBydzeta
            dBzdx = xix*dBzdxi + etax*dBzdeta + zetax*dBzdzeta; dBzdy = xiy*dBzdxi + etay*dBzdeta + zetay*dBzdzeta; dBzdz = xiz*dBzdxi + etaz*dBzdeta + zetaz*dBzdzeta

            Jx = dBzdy - dBydz
            Jy = dBxdz - dBzdx
            Jz = dBydx - dBxdy

            fres_Bx = η_mhd * (Jy * fnz - Jz * fny)
            fres_By = η_mhd * (Jz * fnx - Jx * fnz)
            fres_Bz = η_mhd * (Jx * fny - Jy * fnx)
            fres_E  = fres_Bx * Bx_f + fres_By * By_f + fres_Bz * Bz_f
        end
    end
    fv_E=(fv_rhou*u_f+fv_rhov*v_f+fv_rhow*w_f)-(qx*fnx+qy*fny+qz*fnz)
    @static if equation_type == :MHD
        if resistive_flux_active
            fv_E += fres_E
        end
    end
    @static if ct_mode
        fi = i-NG+1; fj = j-NG+1; fk = k-NG+1
    else
        fi = i-NG; fj = j-NG; fk = k-NG+1
    end
    @inbounds begin
        Fv_z[fi,fj,fk,1]=FT(0e0); Fv_z[fi,fj,fk,2]=fv_rhou*area
        Fv_z[fi,fj,fk,3]=fv_rhov*area; Fv_z[fi,fj,fk,4]=fv_rhow*area; Fv_z[fi,fj,fk,5]=fv_E*area
        @static if equation_type == :MHD
            Fv_z[fi,fj,fk,6] = resistive_flux_active ? fres_Bx * area : zero(FT)
            Fv_z[fi,fj,fk,7] = resistive_flux_active ? fres_By * area : zero(FT)
            Fv_z[fi,fj,fk,8] = resistive_flux_active ? fres_Bz * area : zero(FT)
            Fv_z[fi,fj,fk,9] = zero(FT)
        end
    end
    return
end

