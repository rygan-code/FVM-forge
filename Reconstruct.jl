
@inline function Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕ, hp1, lin_ϕ, splitMethodID, ch_glm::FT)
    @static if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
        # ── AC mode: always use AC_Rusanov ──
        return AC_Rusanov_Flux(UL_vec, UR_vec, nx, ny, nz)
    elseif equation_type == :MHD
        # ── MHD mode: use MHD-specific Riemann solvers ──
        if splitMethodID == Int32(5)
            return MHD_KEP_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        end
        if splitMethodID == Int32(4)
            f_upw = HLLD_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        else
            f_upw = MHD_Rusanov_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        end
        if ϕ >= hp1
            return f_upw
        end
        f_kep = MHD_KEP_Flux(UL_vec, UR_vec, nx, ny, nz, ch_glm)
        a = one(FT) - lin_ϕ
        b = lin_ϕ
        return SVector{9, FT}((
            a*f_kep[1]+b*f_upw[1], a*f_kep[2]+b*f_upw[2], a*f_kep[3]+b*f_upw[3],
            a*f_kep[4]+b*f_upw[4], a*f_kep[5]+b*f_upw[5], a*f_kep[6]+b*f_upw[6],
            a*f_kep[7]+b*f_upw[7], a*f_kep[8]+b*f_upw[8], a*f_kep[9]+b*f_upw[9]))
    else
        # ── Compressible mode ──
        if splitMethodID == Int32(5)
            return KEP_Flux(UL_vec, UR_vec, nx, ny, nz)
        end
        f_upw = if splitMethodID == Int32(1)
            HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)
        elseif splitMethodID == Int32(2)
            SW_Flux(UL_vec, UR_vec, nx, ny, nz)
        elseif splitMethodID == Int32(3)
            VL_Flux(UL_vec, UR_vec, nx, ny, nz)
        elseif splitMethodID == Int32(4)
            Roe_Flux(UL_vec, UR_vec, nx, ny, nz)
        else
            HLLC_Flux(UL_vec, UR_vec, nx, ny, nz)
        end
        if ϕ >= hp1
            return f_upw
        end
        f_kep = KEP_Flux(UL_vec, UR_vec, nx, ny, nz)
        return SVector{Ncons, FT}(ntuple(Val(Ncons)) do n
            (one(FT) - lin_ϕ) * f_kep[n] + lin_ϕ * f_upw[n]
        end)
    end
end

@inline function weno_zq_blend(v0::FT, v1::FT, v2::FT, Is1::FT, Is2::FT, Is3::FT, Is4::FT, ss::FT) where {FT}
    β1 = Is2
    β2 = Is3
    β0 = max(Is1, max(Is2, max(Is3, Is4)))
    
    τ = abs(Is1 - Is4) * ss
    ϵ = FT(1.0e-10)
    
    γ0, γ1, γ2 = FT(0.90), FT(0.05), FT(0.05)
    
    α0 = γ0 * (one(FT) + τ / (ϵ + β0 * ss))^2
    α1 = γ1 * (one(FT) + τ / (ϵ + β1 * ss))^2
    α2 = γ2 * (one(FT) + τ / (ϵ + β2 * ss))^2
    
    invsum = one(FT) / (α0 + α1 + α2)
    ω0 = α0 * invsum
    ω1 = α1 * invsum
    ω2 = α2 * invsum
    
    v0_star = (v0 - γ1*v1 - γ2*v2) / γ0
    
    w0 = ω0 / γ0
    w1 = ω1 - ω0 * (γ1 / γ0)
    w2 = ω2 - ω0 * (γ2 / γ0)
    
    return w0 * v0_star + w1 * v1 + w2 * v2
end

function Eigen_reconstruct_i(Q, U, ϕ, S, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp,
                             stencil_arr, Δstencil_arr, lin_phi_arr,
                             stencil_R_arr, Δstencil_R_arr,
                             ch_glm::FT=one(FT), mode::Int32=Int32(0),
                             intf_ilo::Int32=Int32(-1), intf_ihi::Int32=Int32(-1),
                             UL_save_ihi=nothing, UR_save_ilo=nothing)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. 边界检???
    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG || j < NG+1 || k < NG+1
        return
    end
    # Interior/boundary mode filter (stencil width=4 in i)
    if mode == Int32(1) && (i < NG+Int32(4) || i > nxp+NG-Int32(4)); return; end
    if mode == Int32(2) && (i >= NG+Int32(4) && i <= nxp+NG-Int32(4)); return; end
    # 2. Geometry
    @inbounds nx=nxi[i+1,j,k]; ny=nyi[i+1,j,k]; nz=nzi[i+1,j,k]; Area=Areai[i+1,j,k]

    # 3. 激波传感器
    @inbounds ϕx = max(ϕ[i-2, j, k], ϕ[i-1, j, k], ϕ[i, j, k], ϕ[i+1, j, k], ϕ[i+2, j, k], ϕ[i+3, j, k])

    # AC: always use linear reconstruction (branch A). Branch B performs
    # 5-variable Roe decomposition that reads Q[...,5:6] — out of bounds for AC.
    if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
        ϕx = zero(FT)
    end

    # ...
    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # 分支 A: 光滑???(极速模???
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[i]
    if ϕx < hybrid_ϕ1
        α_adapt = min(ϕx / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        # Left-state weights
        @inbounds L1 = stencil_arr[i,1] + α_adapt * Δstencil_arr[i,1]; @inbounds L2 = stencil_arr[i,2] + α_adapt * Δstencil_arr[i,2]
        @inbounds L3 = stencil_arr[i,3] + α_adapt * Δstencil_arr[i,3]; @inbounds L4 = stencil_arr[i,4] + α_adapt * Δstencil_arr[i,4]
        @inbounds L5 = stencil_arr[i,5] + α_adapt * Δstencil_arr[i,5]; @inbounds L6 = stencil_arr[i,6] + α_adapt * Δstencil_arr[i,6]
        @inbounds L7 = stencil_arr[i,7] + α_adapt * Δstencil_arr[i,7]
        for n = 1:Ncons
            @inbounds v1 = U[i-3,j,k,n]; v2 = U[i-2,j,k,n]; v3 = U[i-1,j,k,n]
            @inbounds v4 = U[i  ,j,k,n]; v5 = U[i+1,j,k,n]; v6 = U[i+2,j,k,n]; v7 = U[i+3,j,k,n]
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        # Right-state weights (temporal register reuse)
        @inbounds L1 = stencil_R_arr[i,1] + α_adapt * Δstencil_R_arr[i,1]; @inbounds L2 = stencil_R_arr[i,2] + α_adapt * Δstencil_R_arr[i,2]
        @inbounds L3 = stencil_R_arr[i,3] + α_adapt * Δstencil_R_arr[i,3]; @inbounds L4 = stencil_R_arr[i,4] + α_adapt * Δstencil_R_arr[i,4]
        @inbounds L5 = stencil_R_arr[i,5] + α_adapt * Δstencil_R_arr[i,5]; @inbounds L6 = stencil_R_arr[i,6] + α_adapt * Δstencil_R_arr[i,6]
        @inbounds L7 = stencil_R_arr[i,7] + α_adapt * Δstencil_R_arr[i,7]
        for n = 1:Ncons
            @inbounds r1 = U[i+4,j,k,n]; r2 = U[i+3,j,k,n]; r3 = U[i+2,j,k,n]
            @inbounds r4 = U[i+1,j,k,n]; r5 = U[i  ,j,k,n]; r6 = U[i-1,j,k,n]; r7 = U[i-2,j,k,n]
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end


    # ==============================
    # 分支 B: 间断???(特征分解模式)
    # ==============================
    else 
        # 0. Safety Check for uninitialized/vacuum states
        @inbounds ρL = Q[i, j, k, 1]; ρR = Q[i+1, j, k, 1]
        if ρL < FT(1.0e-10) || ρR < FT(1.0e-10)
            @inbounds for n = 1:Ncons
                Fx[i-NG+1, j-NG, k-NG, n] = zero(FT)
            end
            return
        end

        @inbounds uL = Q[i, j, k, 2]; uR = Q[i+1, j, k, 2]
        @inbounds vL = Q[i, j, k, 3]; vR = Q[i+1, j, k, 3]
        @inbounds wL = Q[i, j, k, 4]; wR = Q[i+1, j, k, 4]
        @inbounds pL = Q[i, j, k, 5]; pR = Q[i+1, j, k, 5]

        sqρL = sqrt(ρL); sqρR = sqrt(ρR)
        inv_sq = one(FT) / (sqρL + sqρR)

        u = (sqρL * uL + sqρR * uR) * inv_sq
        v = (sqρL * vL + sqρR * vR) * inv_sq
        w = (sqρL * wL + sqρR * wR) * inv_sq
        
        HL = γ/(γ-one(FT))*pL/ρL + FT(0.5)*(uL^2 + vL^2 + wL^2)
        HR = γ/(γ-one(FT))*pR/ρR + FT(0.5)*(uR^2 + vR^2 + wR^2)
        H = (sqρL * HL + sqρR * HR) * inv_sq
        
        v2 = FT(0.5)*(u^2 + v^2 + w^2)
        c = sqrt(max(FT(1.0e-10), (γ-one(FT))*(H - v2)))
        
        # Tangent vectors: Use more robust selection to avoid zero tangent for any normal orientation
        # If normal is roughly [0, 0, 1], then abs(nz) > abs(ny) and abs(nz) > abs(nx)
        if abs(nx) < FT(0.6e0) && abs(ny) < FT(0.6e0) # Normal is mostly Z-aligned
            den = sqrt(nx*nx + nz*nz + FT(1.0e-12)); lx = -nz / den; ly = zero(FT); lz = nx / den
        elseif abs(nx) < FT(0.6e0) && abs(nz) < FT(0.6e0) # Normal is mostly Y-aligned
            den = sqrt(nx*nx + ny*ny + FT(1.0e-12)); lx = -ny / den; ly = nx / den; lz = zero(FT)
        else # Normal is mostly X-aligned or diagonal
            den = sqrt(ny*ny + nz*nz + FT(1.0e-12)); lx = zero(FT); ly = -nz / den; lz = ny / den
        end
        mx = ny * lz - nz * ly; my = nz * lx - nx * lz; mz = nx * ly - ny * lx

        invc = one(FT)/c; invc2 = invc*invc
        K = γ - one(FT)
        Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
        Kv2 = K*v2*invc2; Kc2 = K*invc2
        un = u*nx + v*ny + w*nz; ul = u*lx + v*ly + w*lz; um = u*mx + v*my + w*mz
        un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
        half = FT(0.5); mhalf = -FT(0.5)

        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        @inbounds ss = FT(2.0)/(S[i+1, j, k] + S[i, j, k] + FT(1.0e-20))

        for n = 1:Ncons
            # 2a. 左特征向???L
            ln1=zero(FT); ln2=zero(FT); ln3=zero(FT); ln4=zero(FT); ln5=zero(FT)
            if n == 1; ln1 = half*(Kv2 + un_invc); ln2 = mhalf*(Ku + nx_invc); ln3 = mhalf*(Kv + ny_invc); ln4 = mhalf*(Kw + nz_invc); ln5 = half*Kc2
            elseif n == 2; ln1 = one(FT) - Kv2; ln2 = Ku; ln3 = Kv; ln4 = Kw; ln5 = -Kc2
            elseif n == 3; ln1 = half*(Kv2 - un_invc); ln2 = mhalf*(Ku - nx_invc); ln3 = mhalf*(Kv - ny_invc); ln4 = mhalf*(Kw - nz_invc); ln5 = half*Kc2
            elseif n == 4; ln1 = -ul; ln2 = lx; ln3 = ly; ln4 = lz; ln5 = zero(FT)
            else; ln1 = -um; ln2 = mx; ln3 = my; ln4 = mz; ln5 = zero(FT); end

            # 2b-L. Project U -> V (L side)
            @inbounds V1 = ln1*U[i-3,j,k,1] + ln2*U[i-3,j,k,2] + ln3*U[i-3,j,k,3] + ln4*U[i-3,j,k,4] + ln5*U[i-3,j,k,5]
            @inbounds V2 = ln1*U[i-2,j,k,1] + ln2*U[i-2,j,k,2] + ln3*U[i-2,j,k,3] + ln4*U[i-2,j,k,4] + ln5*U[i-2,j,k,5]
            @inbounds V3 = ln1*U[i-1,j,k,1] + ln2*U[i-1,j,k,2] + ln3*U[i-1,j,k,3] + ln4*U[i-1,j,k,4] + ln5*U[i-1,j,k,5]
            @inbounds V4 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V5 = ln1*U[i+1,j,k,1] + ln2*U[i+1,j,k,2] + ln3*U[i+1,j,k,3] + ln4*U[i+1,j,k,4] + ln5*U[i+1,j,k,5]
            @inbounds V6 = ln1*U[i+2,j,k,1] + ln2*U[i+2,j,k,2] + ln3*U[i+2,j,k,3] + ln4*U[i+2,j,k,4] + ln5*U[i+2,j,k,5]
            @inbounds V7 = ln1*U[i+3,j,k,1] + ln2*U[i+3,j,k,2] + ln3*U[i+3,j,k,3] + ln4*U[i+3,j,k,4] + ln5*U[i+3,j,k,5]

            valL = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valL=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valL=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valL=V4+FT(0.5)*minmod(V4-V3,V5-V4); end

            # 2b-R. Project U -> V (R side, reuse V1-V7)
            @inbounds V1 = ln1*U[i+4,j,k,1] + ln2*U[i+4,j,k,2] + ln3*U[i+4,j,k,3] + ln4*U[i+4,j,k,4] + ln5*U[i+4,j,k,5]
            @inbounds V2 = ln1*U[i+3,j,k,1] + ln2*U[i+3,j,k,2] + ln3*U[i+3,j,k,3] + ln4*U[i+3,j,k,4] + ln5*U[i+3,j,k,5]
            @inbounds V3 = ln1*U[i+2,j,k,1] + ln2*U[i+2,j,k,2] + ln3*U[i+2,j,k,3] + ln4*U[i+2,j,k,4] + ln5*U[i+2,j,k,5]
            @inbounds V4 = ln1*U[i+1,j,k,1] + ln2*U[i+1,j,k,2] + ln3*U[i+1,j,k,3] + ln4*U[i+1,j,k,4] + ln5*U[i+1,j,k,5]
            @inbounds V5 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V6 = ln1*U[i-1,j,k,1] + ln2*U[i-1,j,k,2] + ln3*U[i-1,j,k,3] + ln4*U[i-1,j,k,4] + ln5*U[i-1,j,k,5]
            @inbounds V7 = ln1*U[i-2,j,k,1] + ln2*U[i-2,j,k,2] + ln3*U[i-2,j,k,3] + ln4*U[i-2,j,k,4] + ln5*U[i-2,j,k,5]

            valR = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valR=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valR=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valR=V4-FT(0.5)*minmod(V4-V3,V4-V5); end

            rn1=zero(FT); rn2=zero(FT); rn3=zero(FT); rn4=zero(FT); rn5=zero(FT)
            if n == 1; rn1=one(FT); rn2=u-nx*c; rn3=v-ny*c; rn4=w-nz*c; rn5=H-un*c
            elseif n == 2; rn1=one(FT); rn2=u; rn3=v; rn4=w; rn5=v2
            elseif n == 3; rn1=one(FT); rn2=u+nx*c; rn3=v+ny*c; rn4=w+nz*c; rn5=H+un*c
            elseif n == 4; rn1=zero(FT); rn2=lx; rn3=ly; rn4=lz; rn5=ul
            else; rn1=zero(FT); rn2=mx; rn3=my; rn4=mz; rn5=um; end
            
            rn = SVector{5, FT}(rn1, rn2, rn3, rn4, rn5)
            for m = 1:Ncons
                UL_final[m] += rn[m] * valL; UR_final[m] += rn[m] * valR
            end
        end
    end

    # Failsafe: if reconstruction (Branch A or B) produces non-physical state, fall back to 1st order
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    _eiL = UL_final[Ncons] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
    _eiR = UR_final[Ncons] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = U[i,j,k,n]; end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = U[i+1,j,k,n]; end
    end

    # ── Save interface face values for flux sync ──
    if intf_ihi > Int32(0) && i == intf_ihi && UL_save_ihi !== nothing
        @inbounds for n = 1:Ncons; UL_save_ihi[j-NG, k-NG, n] = UL_final[n]; end
    end
    if intf_ilo > Int32(0) && i == intf_ilo && UR_save_ilo !== nothing
        @inbounds for n = 1:Ncons; UR_save_ilo[j-NG, k-NG, n] = UR_final[n]; end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    
    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)

    @inbounds for n = 1:Ncons
        Fx[i-NG+1, j-NG, k-NG, n] = flux_temp[n] * Area
    end
    return
end

function Eigen_reconstruct_j(Q, U, ϕ, S, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp,
                             stencil_arr, Δstencil_arr, lin_phi_arr,
                             stencil_R_arr, Δstencil_R_arr,
                             ch_glm::FT, mode::Int32,
                             intf_jlo::Int32=Int32(-1), intf_jhi::Int32=Int32(-1),
                             UL_save_jhi=nothing, UR_save_jlo=nothing)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. 边界检查(J接口)
    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG || k < NG+1
        return
    end
    # Interior/boundary mode filter (stencil width=4 in j)
    if mode == Int32(1) && (j < NG+Int32(4) || j > nyp+NG-Int32(4)); return; end
    if mode == Int32(2) && (j >= NG+Int32(4) && j <= nyp+NG-Int32(4)); return; end

    # 2. Geometry
    @inbounds nx=nxj[i,j+1,k]; ny=nyj[i,j+1,k]; nz=nzj[i,j+1,k]; Area=Areaj[i,j+1,k]

    # 3. 激波传感器 (J 方向)
    @inbounds ϕx = max(ϕ[i, j-2, k], ϕ[i, j-1, k], ϕ[i, j, k], ϕ[i, j+1, k], ϕ[i, j+2, k], ϕ[i, j+3, k])

    # AC: always use linear reconstruction (branch A)
    if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
        ϕx = zero(FT)
    end

    # ...
    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # 分支 A: 光滑区(极速模式)
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    if ϕx < hybrid_ϕ1
        # ═══ Original 1D 7-point reconstruction ═══
        α_adapt = min(ϕx / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        @inbounds L1 = stencil_arr[j,k,1] + α_adapt * Δstencil_arr[j,k,1]; @inbounds L2 = stencil_arr[j,k,2] + α_adapt * Δstencil_arr[j,k,2]
        @inbounds L3 = stencil_arr[j,k,3] + α_adapt * Δstencil_arr[j,k,3]; @inbounds L4 = stencil_arr[j,k,4] + α_adapt * Δstencil_arr[j,k,4]
        @inbounds L5 = stencil_arr[j,k,5] + α_adapt * Δstencil_arr[j,k,5]; @inbounds L6 = stencil_arr[j,k,6] + α_adapt * Δstencil_arr[j,k,6]
        @inbounds L7 = stencil_arr[j,k,7] + α_adapt * Δstencil_arr[j,k,7]
        for n = 1:Ncons
            @inbounds v1 = U[i,j-3,k,n]; v2 = U[i,j-2,k,n]; v3 = U[i,j-1,k,n]
            @inbounds v4 = U[i,j  ,k,n]; v5 = U[i,j+1,k,n]; v6 = U[i,j+2,k,n]; v7 = U[i,j+3,k,n]
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        @inbounds L1 = stencil_R_arr[j,k,1] + α_adapt * Δstencil_R_arr[j,k,1]; @inbounds L2 = stencil_R_arr[j,k,2] + α_adapt * Δstencil_R_arr[j,k,2]
        @inbounds L3 = stencil_R_arr[j,k,3] + α_adapt * Δstencil_R_arr[j,k,3]; @inbounds L4 = stencil_R_arr[j,k,4] + α_adapt * Δstencil_R_arr[j,k,4]
        @inbounds L5 = stencil_R_arr[j,k,5] + α_adapt * Δstencil_R_arr[j,k,5]; @inbounds L6 = stencil_R_arr[j,k,6] + α_adapt * Δstencil_R_arr[j,k,6]
        @inbounds L7 = stencil_R_arr[j,k,7] + α_adapt * Δstencil_R_arr[j,k,7]
        for n = 1:Ncons
            @inbounds r1 = U[i,j+4,k,n]; r2 = U[i,j+3,k,n]; r3 = U[i,j+2,k,n]
            @inbounds r4 = U[i,j+1,k,n]; r5 = U[i,j  ,k,n]; r6 = U[i,j-1,k,n]; r7 = U[i,j-2,k,n]
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end

        # Failsafe check for Branch A
        @inbounds ρL_fs = UL_final[1]; @inbounds ρR_fs = UR_final[1]
        @inbounds eiL_fs = UL_final[5] - FT(0.5)*(UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2)/ρL_fs
        @inbounds eiR_fs = UR_final[5] - FT(0.5)*(UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2)/ρR_fs
        if ρL_fs < FT(1.0e-10) || eiL_fs < FT(1.0e-10) || ρR_fs < FT(1.0e-10) || eiR_fs < FT(1.0e-10) || !isfinite(ρL_fs) || !isfinite(ρR_fs) || !isfinite(eiL_fs) || !isfinite(eiR_fs)
            for n = 1:Ncons
                @inbounds UL_final[n] = U[i, j, k, n]
                @inbounds UR_final[n] = U[i, j+1, k, n]
            end
        end


    # ==============================
    # 分支 B: 间断区(特征分解模式)
    # ==============================
    else 
        # 0. Safety Check for uninitialized/vacuum states
        @inbounds ρL = Q[i, j, k, 1]; ρR = Q[i, j+1, k, 1]
        if ρL < FT(1.0e-10) || ρR < FT(1.0e-10)
            @inbounds for n = 1:Ncons
                Fy[i-NG, j-NG+1, k-NG, n] = zero(FT)
            end
            return
        end

        # Roe Averages
        @inbounds uL = Q[i, j, k, 2]; uR = Q[i, j+1, k, 2]
        @inbounds vL = Q[i, j, k, 3]; vR = Q[i, j+1, k, 3]
        @inbounds wL = Q[i, j, k, 4]; wR = Q[i, j+1, k, 4]
        @inbounds pL = Q[i, j, k, 5]; pR = Q[i, j+1, k, 5]

        sqρL = sqrt(ρL); sqρR = sqrt(ρR)
        inv_sq = one(FT) / (sqρL + sqρR)

        u = (sqρL * uL + sqρR * uR) * inv_sq
        v = (sqρL * vL + sqρR * vR) * inv_sq
        w = (sqρL * wL + sqρR * wR) * inv_sq
        
        HL = γ/(γ-one(FT))*pL/ρL + FT(0.5)*(uL^2 + vL^2 + wL^2)
        HR = γ/(γ-one(FT))*pR/ρR + FT(0.5)*(uR^2 + vR^2 + wR^2)
        H = (sqρL * HL + sqρR * HR) * inv_sq
        
        v2 = FT(0.5)*(u^2 + v^2 + w^2)
        c = sqrt(max(FT(1.0e-10), (γ-one(FT))*(H - v2)))
        
        if abs(nx) < FT(0.6e0) && abs(ny) < FT(0.6e0)
            den = sqrt(nx*nx + nz*nz + FT(1.0e-12)); lx = -nz / den; ly = zero(FT); lz = nx / den
        elseif abs(nx) < FT(0.6e0) && abs(nz) < FT(0.6e0)
            den = sqrt(nx*nx + ny*ny + FT(1.0e-12)); lx = -ny / den; ly = nx / den; lz = zero(FT)
        else
            den = sqrt(ny*ny + nz*nz + FT(1.0e-12)); lx = zero(FT); ly = -nz / den; lz = ny / den
        end
        mx = ny * lz - nz * ly; my = nz * lx - nx * lz; mz = nx * ly - ny * lx

        invc = one(FT)/c; invc2 = invc*invc
        K = γ - one(FT)
        Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
        Kv2 = K*v2*invc2; Kc2 = K*invc2
        un = u*nx + v*ny + w*nz; ul = u*lx + v*ly + w*lz; um = u*mx + v*my + w*mz
        un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
        half = FT(0.5); mhalf = -FT(0.5)

        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        @inbounds ss = FT(2.0)/(S[i, j+1, k] + S[i, j, k] + FT(1.0e-20))

        for n = 1:Ncons
            ln1=zero(FT); ln2=zero(FT); ln3=zero(FT); ln4=zero(FT); ln5=zero(FT)
            if n == 1; ln1 = half*(Kv2 + un_invc); ln2 = mhalf*(Ku + nx_invc); ln3 = mhalf*(Kv + ny_invc); ln4 = mhalf*(Kw + nz_invc); ln5 = half*Kc2
            elseif n == 2; ln1 = one(FT) - Kv2; ln2 = Ku; ln3 = Kv; ln4 = Kw; ln5 = -Kc2
            elseif n == 3; ln1 = half*(Kv2 - un_invc); ln2 = mhalf*(Ku - nx_invc); ln3 = mhalf*(Kv - ny_invc); ln4 = mhalf*(Kw - nz_invc); ln5 = half*Kc2
            elseif n == 4; ln1 = -ul; ln2 = lx; ln3 = ly; ln4 = lz; ln5 = zero(FT)
            else; ln1 = -um; ln2 = mx; ln3 = my; ln4 = mz; ln5 = zero(FT); end

            # 2b-L. Project U -> V (L side)
            @inbounds V1 = ln1*U[i,j-3,k,1] + ln2*U[i,j-3,k,2] + ln3*U[i,j-3,k,3] + ln4*U[i,j-3,k,4] + ln5*U[i,j-3,k,5]
            @inbounds V2 = ln1*U[i,j-2,k,1] + ln2*U[i,j-2,k,2] + ln3*U[i,j-2,k,3] + ln4*U[i,j-2,k,4] + ln5*U[i,j-2,k,5]
            @inbounds V3 = ln1*U[i,j-1,k,1] + ln2*U[i,j-1,k,2] + ln3*U[i,j-1,k,3] + ln4*U[i,j-1,k,4] + ln5*U[i,j-1,k,5]
            @inbounds V4 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V5 = ln1*U[i,j+1,k,1] + ln2*U[i,j+1,k,2] + ln3*U[i,j+1,k,3] + ln4*U[i,j+1,k,4] + ln5*U[i,j+1,k,5]
            @inbounds V6 = ln1*U[i,j+2,k,1] + ln2*U[i,j+2,k,2] + ln3*U[i,j+2,k,3] + ln4*U[i,j+2,k,4] + ln5*U[i,j+2,k,5]
            @inbounds V7 = ln1*U[i,j+3,k,1] + ln2*U[i,j+3,k,2] + ln3*U[i,j+3,k,3] + ln4*U[i,j+3,k,4] + ln5*U[i,j+3,k,5]

            valL = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valL=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valL=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valL=V4+FT(0.5)*minmod(V4-V3,V5-V4); end

            # 2b-R. Project U -> V (R side, reuse V1-V7)
            @inbounds V1 = ln1*U[i,j+4,k,1] + ln2*U[i,j+4,k,2] + ln3*U[i,j+4,k,3] + ln4*U[i,j+4,k,4] + ln5*U[i,j+4,k,5]
            @inbounds V2 = ln1*U[i,j+3,k,1] + ln2*U[i,j+3,k,2] + ln3*U[i,j+3,k,3] + ln4*U[i,j+3,k,4] + ln5*U[i,j+3,k,5]
            @inbounds V3 = ln1*U[i,j+2,k,1] + ln2*U[i,j+2,k,2] + ln3*U[i,j+2,k,3] + ln4*U[i,j+2,k,4] + ln5*U[i,j+2,k,5]
            @inbounds V4 = ln1*U[i,j+1,k,1] + ln2*U[i,j+1,k,2] + ln3*U[i,j+1,k,3] + ln4*U[i,j+1,k,4] + ln5*U[i,j+1,k,5]
            @inbounds V5 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V6 = ln1*U[i,j-1,k,1] + ln2*U[i,j-1,k,2] + ln3*U[i,j-1,k,3] + ln4*U[i,j-1,k,4] + ln5*U[i,j-1,k,5]
            @inbounds V7 = ln1*U[i,j-2,k,1] + ln2*U[i,j-2,k,2] + ln3*U[i,j-2,k,3] + ln4*U[i,j-2,k,4] + ln5*U[i,j-2,k,5]

            valR = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5+V5*(FT(547.0e0)*V5))
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valR=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valR=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valR=V4-FT(0.5)*minmod(V4-V3,V4-V5); end

            rn1=zero(FT); rn2=zero(FT); rn3=zero(FT); rn4=zero(FT); rn5=zero(FT)
            if n == 1; rn1=one(FT); rn2=u-nx*c; rn3=v-ny*c; rn4=w-nz*c; rn5=H-un*c
            elseif n == 2; rn1=one(FT); rn2=u; rn3=v; rn4=w; rn5=v2
            elseif n == 3; rn1=one(FT); rn2=u+nx*c; rn3=v+ny*c; rn4=w+nz*c; rn5=H+un*c
            elseif n == 4; rn1=zero(FT); rn2=lx; rn3=ly; rn4=lz; rn5=ul
            else; rn1=zero(FT); rn2=mx; rn3=my; rn4=mz; rn5=um; end
            
            rn = SVector{5, FT}(rn1, rn2, rn3, rn4, rn5)
            for m = 1:Ncons
                UL_final[m] += rn[m] * valL; UR_final[m] += rn[m] * valR
            end
        end
    end

    # Failsafe: if reconstruction (Branch A or B) produces non-physical state, fall back to 1st order
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    _eiL = UL_final[Ncons] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
    _eiR = UR_final[Ncons] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = U[i,j,k,n]; end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = U[i,j+1,k,n]; end
    end

    # ── Save interface face values for flux sync ──
    if intf_jhi > Int32(0) && j == intf_jhi && UL_save_jhi !== nothing
        @inbounds for n = 1:Ncons; UL_save_jhi[i-NG, k-NG, n] = UL_final[n]; end
    end
    if intf_jlo > Int32(0) && j == intf_jlo && UR_save_jlo !== nothing
        @inbounds for n = 1:Ncons; UR_save_jlo[i-NG, k-NG, n] = UR_final[n]; end
    end

    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    
    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)

    @inbounds for n = 1:Ncons
        Fy[i-NG, j-NG+1, k-NG, n] = flux_temp[n] * Area
    end
    return
end

function Eigen_reconstruct_k(Q, U, ϕ, S, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp,
                             stencil_arr, Δstencil_arr, lin_phi_arr,
                             stencil_R_arr, Δstencil_R_arr,
                             ch_glm::FT, mode::Int32,
                             intf_klo::Int32=Int32(-1), intf_khi::Int32=Int32(-1),
                             UL_save_khi=nothing, UR_save_klo=nothing)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. 边界检查(K接口)
    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG
        return
    end
    # Interior/boundary mode filter (stencil width=4 in k)
    if mode == Int32(1) && (k < NG+Int32(4) || k > nzp+NG-Int32(4)); return; end
    if mode == Int32(2) && (k >= NG+Int32(4) && k <= nzp+NG-Int32(4)); return; end

    # 2. Geometry
    @inbounds nx=nxk[i,j,k+1]; ny=nyk[i,j,k+1]; nz=nzk[i,j,k+1]; Area=Areak[i,j,k+1]

    # 3. 激波传感器 (K 方向)
    @inbounds ϕx = max(ϕ[i, j, k-2], ϕ[i, j, k-1], ϕ[i, j, k], ϕ[i, j, k+1], ϕ[i, j, k+2], ϕ[i, j, k+3])

    # AC: always use linear reconstruction (branch A)
    if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
        ϕx = zero(FT)
    end

    # ...
    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # 分支 A: 光滑区(极速模式)
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    if ϕx < hybrid_ϕ1
        # ═══ Original 1D 7-point reconstruction ═══
        α_adapt = min(ϕx / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        @inbounds L1 = stencil_arr[j,k,1] + α_adapt * Δstencil_arr[j,k,1]; @inbounds L2 = stencil_arr[j,k,2] + α_adapt * Δstencil_arr[j,k,2]
        @inbounds L3 = stencil_arr[j,k,3] + α_adapt * Δstencil_arr[j,k,3]; @inbounds L4 = stencil_arr[j,k,4] + α_adapt * Δstencil_arr[j,k,4]
        @inbounds L5 = stencil_arr[j,k,5] + α_adapt * Δstencil_arr[j,k,5]; @inbounds L6 = stencil_arr[j,k,6] + α_adapt * Δstencil_arr[j,k,6]
        @inbounds L7 = stencil_arr[j,k,7] + α_adapt * Δstencil_arr[j,k,7]
        for n = 1:Ncons
            @inbounds v1 = U[i,j,k-3,n]; v2 = U[i,j,k-2,n]; v3 = U[i,j,k-1,n]
            @inbounds v4 = U[i,j,k  ,n]; v5 = U[i,j,k+1,n]; v6 = U[i,j,k+2,n]; v7 = U[i,j,k+3,n]
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        @inbounds L1 = stencil_R_arr[j,k,1] + α_adapt * Δstencil_R_arr[j,k,1]; @inbounds L2 = stencil_R_arr[j,k,2] + α_adapt * Δstencil_R_arr[j,k,2]
        @inbounds L3 = stencil_R_arr[j,k,3] + α_adapt * Δstencil_R_arr[j,k,3]; @inbounds L4 = stencil_R_arr[j,k,4] + α_adapt * Δstencil_R_arr[j,k,4]
        @inbounds L5 = stencil_R_arr[j,k,5] + α_adapt * Δstencil_R_arr[j,k,5]; @inbounds L6 = stencil_R_arr[j,k,6] + α_adapt * Δstencil_R_arr[j,k,6]
        @inbounds L7 = stencil_R_arr[j,k,7] + α_adapt * Δstencil_R_arr[j,k,7]
        for n = 1:Ncons
            @inbounds r1 = U[i,j,k+4,n]; r2 = U[i,j,k+3,n]; r3 = U[i,j,k+2,n]
            @inbounds r4 = U[i,j,k+1,n]; r5 = U[i,j,k  ,n]; r6 = U[i,j,k-1,n]; r7 = U[i,j,k-2,n]
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end

        # Failsafe check for Branch A
        @inbounds ρL_fs = UL_final[1]; @inbounds ρR_fs = UR_final[1]
        @inbounds eiL_fs = UL_final[5] - FT(0.5)*(UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2)/ρL_fs
        @inbounds eiR_fs = UR_final[5] - FT(0.5)*(UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2)/ρR_fs
        if ρL_fs < FT(1.0e-10) || eiL_fs < FT(1.0e-10) || ρR_fs < FT(1.0e-10) || eiR_fs < FT(1.0e-10) || !isfinite(ρL_fs) || !isfinite(ρR_fs) || !isfinite(eiL_fs) || !isfinite(eiR_fs)
            for n = 1:Ncons
                @inbounds UL_final[n] = U[i, j, k, n]
                @inbounds UR_final[n] = U[i, j, k+1, n]
            end
        end

    # ==============================
    # 分支 B: 间断区(特征分解模式)
    # ==============================
    else 
        # 0. Safety Check for uninitialized/vacuum states
        @inbounds ρL = Q[i, j, k, 1]; ρR = Q[i, j, k+1, 1]
        if ρL < FT(1.0e-10) || ρR < FT(1.0e-10)
            @inbounds for n = 1:Ncons
                Fz[i-NG, j-NG, k-NG+1, n] = zero(FT)
            end
            return
        end

        # Roe Averages
        @inbounds uL = Q[i, j, k, 2]; uR = Q[i, j, k+1, 2]
        @inbounds vL = Q[i, j, k, 3]; vR = Q[i, j, k+1, 3]
        @inbounds wL = Q[i, j, k, 4]; wR = Q[i, j, k+1, 4]
        @inbounds pL = Q[i, j, k, 5]; pR = Q[i, j, k+1, 5]

        sqρL = sqrt(ρL); sqρR = sqrt(ρR)
        inv_sq = one(FT) / (sqρL + sqρR)

        u = (sqρL * uL + sqρR * uR) * inv_sq
        v = (sqρL * vL + sqρR * vR) * inv_sq
        w = (sqρL * wL + sqρR * wR) * inv_sq
        
        HL = γ/(γ-one(FT))*pL/ρL + FT(0.5)*(uL^2 + vL^2 + wL^2)
        HR = γ/(γ-one(FT))*pR/ρR + FT(0.5)*(uR^2 + vR^2 + wR^2)
        H = (sqρL * HL + sqρR * HR) * inv_sq
        
        v2 = FT(0.5)*(u^2 + v^2 + w^2)
        c = sqrt(max(FT(1.0e-10), (γ-one(FT))*(H - v2)))
        
        if abs(nx) < FT(0.6e0) && abs(ny) < FT(0.6e0)
            den = sqrt(nx*nx + nz*nz + FT(1.0e-12)); lx = -nz / den; ly = zero(FT); lz = nx / den
        elseif abs(nx) < FT(0.6e0) && abs(nz) < FT(0.6e0)
            den = sqrt(nx*nx + ny*ny + FT(1.0e-12)); lx = -ny / den; ly = nx / den; lz = zero(FT)
        else
            den = sqrt(ny*ny + nz*nz + FT(1.0e-12)); lx = zero(FT); ly = -nz / den; lz = ny / den
        end
        mx = ny * lz - nz * ly; my = nz * lx - nx * lz; mz = nx * ly - ny * lx

        invc = one(FT)/c; invc2 = invc*invc
        K = γ - one(FT)
        Ku = K*u*invc2; Kv = K*v*invc2; Kw = K*w*invc2
        Kv2 = K*v2*invc2; Kc2 = K*invc2
        un = u*nx + v*ny + w*nz; ul = u*lx + v*ly + w*lz; um = u*mx + v*my + w*mz
        un_invc = un*invc; nx_invc = nx*invc; ny_invc = ny*invc; nz_invc = nz*invc
        half = FT(0.5); mhalf = -FT(0.5)

        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        @inbounds ss = FT(2.0)/(S[i, j, k+1] + S[i, j, k] + FT(1.0e-20))

        for n = 1:Ncons
            ln1=zero(FT); ln2=zero(FT); ln3=zero(FT); ln4=zero(FT); ln5=zero(FT)
            if n == 1; ln1 = half*(Kv2 + un_invc); ln2 = mhalf*(Ku + nx_invc); ln3 = mhalf*(Kv + ny_invc); ln4 = mhalf*(Kw + nz_invc); ln5 = half*Kc2
            elseif n == 2; ln1 = one(FT) - Kv2; ln2 = Ku; ln3 = Kv; ln4 = Kw; ln5 = -Kc2
            elseif n == 3; ln1 = half*(Kv2 - un_invc); ln2 = mhalf*(Ku - nx_invc); ln3 = mhalf*(Kv - ny_invc); ln4 = mhalf*(Kw - nz_invc); ln5 = half*Kc2
            elseif n == 4; ln1 = -ul; ln2 = lx; ln3 = ly; ln4 = lz; ln5 = zero(FT)
            else; ln1 = -um; ln2 = mx; ln3 = my; ln4 = mz; ln5 = zero(FT); end

            # 2b-L. Project U -> V (L side)
            @inbounds V1 = ln1*U[i,j,k-3,1] + ln2*U[i,j,k-3,2] + ln3*U[i,j,k-3,3] + ln4*U[i,j,k-3,4] + ln5*U[i,j,k-3,5]
            @inbounds V2 = ln1*U[i,j,k-2,1] + ln2*U[i,j,k-2,2] + ln3*U[i,j,k-2,3] + ln4*U[i,j,k-2,4] + ln5*U[i,j,k-2,5]
            @inbounds V3 = ln1*U[i,j,k-1,1] + ln2*U[i,j,k-1,2] + ln3*U[i,j,k-1,3] + ln4*U[i,j,k-1,4] + ln5*U[i,j,k-1,5]
            @inbounds V4 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V5 = ln1*U[i,j,k+1,1] + ln2*U[i,j,k+1,2] + ln3*U[i,j,k+1,3] + ln4*U[i,j,k+1,4] + ln5*U[i,j,k+1,5]
            @inbounds V6 = ln1*U[i,j,k+2,1] + ln2*U[i,j,k+2,2] + ln3*U[i,j,k+2,3] + ln4*U[i,j,k+2,4] + ln5*U[i,j,k+2,5]
            @inbounds V7 = ln1*U[i,j,k+3,1] + ln2*U[i,j,k+3,2] + ln3*U[i,j,k+3,3] + ln4*U[i,j,k+3,4] + ln5*U[i,j,k+3,5]

            valL = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valL=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valL=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valL=V4+FT(0.5)*minmod(V4-V3,V5-V4); end

            # 2b-R. Project U -> V (R side, reuse V1-V7)
            @inbounds V1 = ln1*U[i,j,k+4,1] + ln2*U[i,j,k+4,2] + ln3*U[i,j,k+4,3] + ln4*U[i,j,k+4,4] + ln5*U[i,j,k+4,5]
            @inbounds V2 = ln1*U[i,j,k+3,1] + ln2*U[i,j,k+3,2] + ln3*U[i,j,k+3,3] + ln4*U[i,j,k+3,4] + ln5*U[i,j,k+3,5]
            @inbounds V3 = ln1*U[i,j,k+2,1] + ln2*U[i,j,k+2,2] + ln3*U[i,j,k+2,3] + ln4*U[i,j,k+2,4] + ln5*U[i,j,k+2,5]
            @inbounds V4 = ln1*U[i,j,k+1,1] + ln2*U[i,j,k+1,2] + ln3*U[i,j,k+1,3] + ln4*U[i,j,k+1,4] + ln5*U[i,j,k+1,5]
            @inbounds V5 = ln1*U[i,j,k,1] + ln2*U[i,j,k,2] + ln3*U[i,j,k,3] + ln4*U[i,j,k,4] + ln5*U[i,j,k,5]
            @inbounds V6 = ln1*U[i,j,k-1,1] + ln2*U[i,j,k-1,2] + ln3*U[i,j,k-1,3] + ln4*U[i,j,k-1,4] + ln5*U[i,j,k-1,5]
            @inbounds V7 = ln1*U[i,j,k-2,1] + ln2*U[i,j,k-2,2] + ln3*U[i,j,k-2,3] + ln4*U[i,j,k-2,4] + ln5*U[i,j,k-2,5]

            valR = zero(FT)
            if ϕx < hybrid_ϕ2
                q1=-FT(3.0)*V1+FT(13.0)*V2-FT(23.0)*V3+FT(25.0e0)*V4; q2=V2-FT(5.0e0)*V3+FT(13.0)*V4+FT(3.0)*V5
                q3=-V3+FT(7.0e0)*V4+FT(7.0e0)*V5-V6; q4=FT(3.0)*V4+FT(13.0)*V5-FT(5.0e0)*V6+V7
                Is1=V1*(FT(547.0e0)*V1-FT(3882.0)*V2+FT(4642.0)*V3-FT(1854.0)*V4)+V2*(FT(7043.0)*V2-FT(17246.0e0)*V3+FT(7042.0)*V4)+V3*(FT(11003.0)*V3-FT(9402.0)*V4)+V4*(FT(2107.0e0)*V4)
                Is2=V2*(FT(267.0e0)*V2-FT(1642.0)*V3+FT(1602.0)*V4-FT(494.0)*V5)+V3*(FT(2843.0)*V3-FT(5966.0e0)*V4+FT(1922.0)*V5)+V4*(FT(3443.0)*V4-FT(2522.0)*V5)+V5*(FT(547.0e0)*V5)
                Is3=V3*(FT(547.0e0)*V3-FT(2522.0)*V4+FT(1922.0)*V5-FT(494.0)*V6)+V4*(FT(3443.0)*V4-FT(5966.0e0)*V5+FT(1602.0)*V6)+V5*(FT(2843.0)*V5-FT(1642.0)*V6)+V6*(FT(267.0e0)*V6)
                Is4=V4*(FT(2107.0e0)*V4-FT(9402.0)*V5+FT(7042.0)*V6-FT(1854.0)*V7)+V5*(FT(11003.0)*V5-FT(17246.0e0)*V6+FT(4642.0)*V7)+V6*(FT(7043.0)*V6-FT(3882.0)*V7)+V7*(FT(547.0e0)*V7)
                td1=WENOϵ1+Is1*ss; td2=WENOϵ1+Is2*ss; td3=WENOϵ1+Is3*ss; td4=WENOϵ1+Is4*ss
                @static if weno_z
                    τ7 = abs(Is1 - Is4) * ss
                    α1=one(FT)*(one(FT)+τ7/(td1+WENOϵ1))/(td1*td1); α2=FT(12.0)*(one(FT)+τ7/(td2+WENOϵ1))/(td2*td2)
                    α3=FT(18.0e0)*(one(FT)+τ7/(td3+WENOϵ1))/(td3*td3); α4=FT(4.0)*(one(FT)+τ7/(td4+WENOϵ1))/(td4*td4)
                else
                    α1=one(FT)/(td1*td1); α2=FT(12.0)/(td2*td2)
                    α3=FT(18.0e0)/(td3*td3); α4=FT(4.0)/(td4*td4)
                end
                invsum=one(FT)/(α1+α2+α3+α4); valR=invsum*(α1*q1+α2*q2+α3*q3+α4*q4)*tmp1
            elseif ϕx < hybrid_ϕ3
                t1=V2-FT(2.0)*V3+V4; t2=V2-FT(4.0)*V3+FT(3.0)*V4; s1=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V3-FT(2.0)*V4+V5; t2=V3-V5; s2=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                t1=V4-FT(2.0)*V5+V6; t2=FT(3.0)*V4-FT(4.0)*V5+V6; s3=FT(13.0)*t1*t1+FT(3.0)*t2*t2
                td1=WENOϵ2+s1*ss; td2=WENOϵ2+s2*ss; td3=WENOϵ2+s3*ss
                @static if weno_z
                    τ5 = abs(s1 - s3) * ss
                    α1=one(FT)*(one(FT)+τ5/(td1+WENOϵ2))/(td1*td1); α2=FT(6.0e0)*(one(FT)+τ5/(td2+WENOϵ2))/(td2*td2); α3=FT(3.0)*(one(FT)+τ5/(td3+WENOϵ2))/(td3*td3)
                else
                    α1=one(FT)/(td1*td1); α2=FT(6.0e0)/(td2*td2); α3=FT(3.0)/(td3*td3)
                end
                invsum=one(FT)/(α1+α2+α3)
                valR=invsum*(α1*(FT(2.0)*V2-FT(7.0e0)*V3+FT(11.0)*V4)+α2*(-V3+FT(5.0e0)*V4+FT(2.0)*V5)+α3*(FT(2.0)*V4+FT(5.0e0)*V5-V6))*tmp2
            else; valR=V4-FT(0.5)*minmod(V4-V3,V4-V5); end

            rn1=zero(FT); rn2=zero(FT); rn3=zero(FT); rn4=zero(FT); rn5=zero(FT)
            if n == 1; rn1=one(FT); rn2=u-nx*c; rn3=v-ny*c; rn4=w-nz*c; rn5=H-un*c
            elseif n == 2; rn1=one(FT); rn2=u; rn3=v; rn4=w; rn5=v2
            elseif n == 3; rn1=one(FT); rn2=u+nx*c; rn3=v+ny*c; rn4=w+nz*c; rn5=H+un*c
            elseif n == 4; rn1=zero(FT); rn2=lx; rn3=ly; rn4=lz; rn5=ul
            else; rn1=zero(FT); rn2=mx; rn3=my; rn4=mz; rn5=um; end
            
            rn = SVector{5, FT}(rn1, rn2, rn3, rn4, rn5)
            for m = 1:Ncons
                UL_final[m] += rn[m] * valL; UR_final[m] += rn[m] * valR
            end
        end
    end

    # Failsafe: if reconstruction (Branch A or B) produces non-physical state, fall back to 1st order
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    _eiL = UL_final[Ncons] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
    _eiR = UR_final[Ncons] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = U[i,j,k,n]; end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = U[i,j,k+1,n]; end
    end

    # ── Save interface face values for flux sync ──
    if intf_khi > Int32(0) && k == intf_khi && UL_save_khi !== nothing
        @inbounds for n = 1:Ncons; UL_save_khi[i-NG, j-NG, n] = UL_final[n]; end
    end
    if intf_klo > Int32(0) && k == intf_klo && UR_save_klo !== nothing
        @inbounds for n = 1:Ncons; UR_save_klo[i-NG, j-NG, n] = UR_final[n]; end
    end

    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    
    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)

    @inbounds for n = 1:Ncons
        Fz[i-NG, j-NG, k-NG+1, n] = flux_temp[n] * Area
    end
end

function Conser_reconstruct_i(Q, U, ϕ, S, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp,
                              stencil_arr, Δstencil_arr, lin_phi_arr,
                              stencil_R_arr, Δstencil_R_arr,
                              ch_glm::FT=one(FT), mode::Int32=Int32(0),
                              intf_ilo::Int32=Int32(-1), intf_ihi::Int32=Int32(-1),
                              UL_save_ihi=nothing, UR_save_ilo=nothing)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. 边界检???
    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG || j < NG+1 || k < NG+1
        return
    end
    # Interior/boundary mode filter (stencil width=4 in i)
    if mode == Int32(1) && (i < NG+Int32(4) || i > nxp+NG-Int32(4)); return; end
    if mode == Int32(2) && (i >= NG+Int32(4) && i <= nxp+NG-Int32(4)); return; end
    # 2. Geometry
    @inbounds nx=nxi[i+1,j,k]; ny=nyi[i+1,j,k]; nz=nzi[i+1,j,k]; Area=Areai[i+1,j,k]
    # @inbounds Area *= INV_SCALE_FACTOR
    # @cuprintf("nx=%e, ny=%e, nz=%e, Area=%e\n", nx, ny, nz, Area)

    # 3. 激波传感器
    @inbounds ϕx = max(ϕ[i-2, j, k], ϕ[i-1, j, k], ϕ[i, j, k], ϕ[i+1, j, k], ϕ[i+2, j, k], ϕ[i+3, j, k])

    # AC: always use linear reconstruction (branch A)
    if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
        ϕx = zero(FT)
    end

    # ...
    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # 分支 A: 光滑???(极速模???
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[i]
    if ϕx < hybrid_ϕ1
        α_adapt = min(ϕx / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        # Left-state weights
        @inbounds L1 = stencil_arr[i,1] + α_adapt * Δstencil_arr[i,1]; @inbounds L2 = stencil_arr[i,2] + α_adapt * Δstencil_arr[i,2]
        @inbounds L3 = stencil_arr[i,3] + α_adapt * Δstencil_arr[i,3]; @inbounds L4 = stencil_arr[i,4] + α_adapt * Δstencil_arr[i,4]
        @inbounds L5 = stencil_arr[i,5] + α_adapt * Δstencil_arr[i,5]; @inbounds L6 = stencil_arr[i,6] + α_adapt * Δstencil_arr[i,6]
        @inbounds L7 = stencil_arr[i,7] + α_adapt * Δstencil_arr[i,7]
        for n = 1:Ncons
            @inbounds v1 = U[i-3,j,k,n]; v2 = U[i-2,j,k,n]; v3 = U[i-1,j,k,n]
            @inbounds v4 = U[i  ,j,k,n]; v5 = U[i+1,j,k,n]; v6 = U[i+2,j,k,n]; v7 = U[i+3,j,k,n]
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        # Right-state weights (temporal register reuse)
        @inbounds L1 = stencil_R_arr[i,1] + α_adapt * Δstencil_R_arr[i,1]; @inbounds L2 = stencil_R_arr[i,2] + α_adapt * Δstencil_R_arr[i,2]
        @inbounds L3 = stencil_R_arr[i,3] + α_adapt * Δstencil_R_arr[i,3]; @inbounds L4 = stencil_R_arr[i,4] + α_adapt * Δstencil_R_arr[i,4]
        @inbounds L5 = stencil_R_arr[i,5] + α_adapt * Δstencil_R_arr[i,5]; @inbounds L6 = stencil_R_arr[i,6] + α_adapt * Δstencil_R_arr[i,6]
        @inbounds L7 = stencil_R_arr[i,7] + α_adapt * Δstencil_R_arr[i,7]
        for n = 1:Ncons
            @inbounds r1 = U[i+4,j,k,n]; r2 = U[i+3,j,k,n]; r3 = U[i+2,j,k,n]
            @inbounds r4 = U[i+1,j,k,n]; r5 = U[i  ,j,k,n]; r6 = U[i-1,j,k,n]; r7 = U[i-2,j,k,n]
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end


    # ==============================
    # 分支 B: 间断???(特征分解模式)
    # ==============================
    else 
        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        @inbounds ss = FT(2.0)/(S[i+1, j, k] + S[i, j, k])

        for n = 1:Ncons

            # 2b. 投影 U -> V (Component-wise Reconstruction)
            @inbounds V1L = U[i-3,j,k,n]
            @inbounds V2L = U[i-2,j,k,n]
            @inbounds V3L = U[i-1,j,k,n]
            @inbounds V4L = U[i  ,j,k,n]
            @inbounds V5L = U[i+1,j,k,n]
            @inbounds V6L = U[i+2,j,k,n]
            @inbounds V7L = U[i+3,j,k,n]

            @inbounds V1R = U[i+4,j,k,n]
            @inbounds V2R = U[i+3,j,k,n]
            @inbounds V3R = U[i+2,j,k,n]
            @inbounds V4R = U[i+1,j,k,n]
            @inbounds V5R = U[i  ,j,k,n]
            @inbounds V6R = U[i-1,j,k,n]
            @inbounds V7R = U[i-2,j,k,n]

            valL = zero(FT); valR = zero(FT)

            # 2c. WENO Reconstruction
            if ϕx < hybrid_ϕ2 # WENO7
                q1L = -FT(3.0)*V1L + FT(13.0)*V2L - FT(23.0)*V3L + FT(25.0e0)*V4L; q1R = -FT(3.0)*V1R + FT(13.0)*V2R - FT(23.0)*V3R + FT(25.0e0)*V4R
                q2L =  one(FT)*V2L -  FT(5.0e0)*V3L + FT(13.0)*V4L +  FT(3.0)*V5L; q2R =  one(FT)*V2R -  FT(5.0e0)*V3R + FT(13.0)*V4R +  FT(3.0)*V5R
                q3L = -one(FT)*V3L +  FT(7.0e0)*V4L +  FT(7.0e0)*V5L -  one(FT)*V6L; q3R = -one(FT)*V3R +  FT(7.0e0)*V4R +  FT(7.0e0)*V5R -  one(FT)*V6R
                q4L =  FT(3.0)*V4L + FT(13.0)*V5L -  FT(5.0e0)*V6L +  one(FT)*V7L; q4R =  FT(3.0)*V4R + FT(13.0)*V5R -  FT(5.0e0)*V6R +  one(FT)*V7R

                Is1L = V1L*( FT(547.0e0)*V1L - FT(3882.0)*V2L + FT(4642.0)*V3L - FT(1854.0)*V4L) + V2L*( FT(7043.0)*V2L -FT(17246.0e0)*V3L + FT(7042.0)*V4L) + V3L*(FT(11003.0)*V3L - FT(9402.0)*V4L) + V4L*( FT(2107.0e0)*V4L)
                Is2L = V2L*( FT(267.0e0)*V2L - FT(1642.0)*V3L + FT(1602.0)*V4L -  FT(494.0)*V5L) + V3L*( FT(2843.0)*V3L - FT(5966.0e0)*V4L + FT(1922.0)*V5L) + V4L*( FT(3443.0)*V4L - FT(2522.0)*V5L) + V5L*(  FT(547.0e0)*V5L)
                Is3L = V3L*( FT(547.0e0)*V3L - FT(2522.0)*V4L + FT(1922.0)*V5L -  FT(494.0)*V6L) + V4L*( FT(3443.0)*V4L - FT(5966.0e0)*V5L + FT(1602.0)*V6L) + V5L*( FT(2843.0)*V5L - FT(1642.0)*V6L) + V6L*(  FT(267.0e0)*V6L)
                Is4L = V4L*( FT(2107.0e0)*V4L - FT(9402.0)*V5L + FT(7042.0)*V6L - FT(1854.0)*V7L) + V5L*(FT(11003.0)*V5L -FT(17246.0e0)*V6L + FT(4642.0)*V7L) + V6L*( FT(7043.0)*V6L - FT(3882.0)*V7L) + V7L*(  FT(547.0e0)*V7L)

                Is1R = V1R*( FT(547.0e0)*V1R - FT(3882.0)*V2R + FT(4642.0)*V3R - FT(1854.0)*V4R) + V2R*( FT(7043.0)*V2R -FT(17246.0e0)*V3R + FT(7042.0)*V4R) + V3R*(FT(11003.0)*V3R - FT(9402.0)*V4R) + V4R*( FT(2107.0e0)*V4R)
                Is2R = V2R*( FT(267.0e0)*V2R - FT(1642.0)*V3R + FT(1602.0)*V4R -  FT(494.0)*V5R) + V3R*( FT(2843.0)*V3R - FT(5966.0e0)*V4R + FT(1922.0)*V5R) + V4R*( FT(3443.0)*V4R - FT(2522.0)*V5R) + V5R*(  FT(547.0e0)*V5R)
                Is3R = V3R*( FT(547.0e0)*V3R - FT(2522.0)*V4R + FT(1922.0)*V5R -  FT(494.0)*V6R) + V4R*( FT(3443.0)*V4R - FT(5966.0e0)*V5R + FT(1602.0)*V6R) + V5R*( FT(2843.0)*V5R - FT(1642.0)*V6R) + V6R*(  FT(267.0e0)*V6R)
                Is4R = V4R*( FT(2107.0e0)*V4R - FT(9402.0)*V5R + FT(7042.0)*V6R - FT(1854.0)*V7R) + V5R*(FT(11003.0)*V5R -FT(17246.0e0)*V6R + FT(4642.0)*V7R) + V6R*( FT(7043.0)*V6R - FT(3882.0)*V7R) + V7R*(  FT(547.0e0)*V7R)

                t_d1L = WENOϵ1 + Is1L*ss; t_d1R = WENOϵ1 + Is1R*ss
                t_d2L = WENOϵ1 + Is2L*ss; t_d2R = WENOϵ1 + Is2R*ss
                t_d3L = WENOϵ1 + Is3L*ss; t_d3R = WENOϵ1 + Is3R*ss
                t_d4L = WENOϵ1 + Is4L*ss; t_d4R = WENOϵ1 + Is4R*ss
                @static if weno_z
                    τ7L = abs(Is1L - Is4L) * ss; τ7R = abs(Is1R - Is4R) * ss
                    α1L = one(FT)*(one(FT)+τ7L/(t_d1L+WENOϵ1))/(t_d1L*t_d1L); α1R = one(FT)*(one(FT)+τ7R/(t_d1R+WENOϵ1))/(t_d1R*t_d1R)
                    α2L = FT(12.0)*(one(FT)+τ7L/(t_d2L+WENOϵ1))/(t_d2L*t_d2L); α2R = FT(12.0)*(one(FT)+τ7R/(t_d2R+WENOϵ1))/(t_d2R*t_d2R)
                    α3L = FT(18.0e0)*(one(FT)+τ7L/(t_d3L+WENOϵ1))/(t_d3L*t_d3L); α3R = FT(18.0e0)*(one(FT)+τ7R/(t_d3R+WENOϵ1))/(t_d3R*t_d3R)
                    α4L = FT(4.0)*(one(FT)+τ7L/(t_d4L+WENOϵ1))/(t_d4L*t_d4L); α4R = FT(4.0)*(one(FT)+τ7R/(t_d4R+WENOϵ1))/(t_d4R*t_d4R)
                else
                    α1L = one(FT)/(t_d1L * t_d1L); α1R = one(FT)/(t_d1R * t_d1R)
                    α2L = FT(12.0)/(t_d2L * t_d2L); α2R = FT(12.0)/(t_d2R * t_d2R)
                    α3L = FT(18.0e0)/(t_d3L * t_d3L); α3R = FT(18.0e0)/(t_d3R * t_d3R)
                    α4L = FT(4.0)/(t_d4L * t_d4L); α4R = FT(4.0)/(t_d4R * t_d4R)
                end

                invsumL = one(FT)/(α1L+α2L+α3L+α4L); invsumR = one(FT)/(α1R+α2R+α3R+α4R)
                valL = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                valR = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1

                # ... (valL ???valR 计算完毕) ...
            
            # ===  WENO5 分支 ===
            elseif ϕx < hybrid_ϕ3 # WENO5
                # ...
                # V2L (i-2), V3L (i-1), V4L (i), V5L (i+1), V6L (i+2)
                # 对应标准 WENO5 ???v1...v5
                
                # Left Side
                # Beta 1: (13/12)(v1-2v2+v3)^2 + (1/4)(v1-4v2+3v3)^2
                t1 = V2L - FT(2.0)*V3L + V4L; t2 = V2L - FT(4.0)*V3L + FT(3.0)*V4L
                s1L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                
                # Beta 2: (13/12)(v2-2v3+v4)^2 + (1/4)(v2-v4)^2
                t1 = V3L - FT(2.0)*V4L + V5L; t2 = V3L - V5L
                s2L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                
                # Beta 3: (13/12)(v3-2v4+v5)^2 + (1/4)(3v3-4v4+v5)^2
                t1 = V4L - FT(2.0)*V5L + V6L; t2 = FT(3.0)*V4L - FT(4.0)*V5L + V6L
                s3L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                # Weights (d0=1/10, d1=6/10, d2=3/10 -> relative 1, 6, 3)
                t_d1L = WENOϵ2 + s1L*ss; t_d2L = WENOϵ2 + s2L*ss; t_d3L = WENOϵ2 + s3L*ss
                @static if weno_z
                    τ5L = abs(s1L - s3L) * ss
                    α1L = one(FT)*(one(FT)+τ5L/(t_d1L+WENOϵ2))/(t_d1L*t_d1L)
                    α2L = FT(6.0e0)*(one(FT)+τ5L/(t_d2L+WENOϵ2))/(t_d2L*t_d2L)
                    α3L = FT(3.0)*(one(FT)+τ5L/(t_d3L+WENOϵ2))/(t_d3L*t_d3L)
                else
                    α1L = one(FT)/(t_d1L * t_d1L)
                    α2L = FT(6.0e0)/(t_d2L * t_d2L)
                    α3L = FT(3.0)/(t_d3L * t_d3L)
                end
                invsumL = one(FT)/(α1L+α2L+α3L)

                # Candidates
                v1 = FT(2.0)*V2L - FT(7.0e0)*V3L + FT(11.0)*V4L
                v2 = -one(FT)*V3L + FT(5.0e0)*V4L + FT(2.0)*V5L
                v3 = FT(2.0)*V4L + FT(5.0e0)*V5L - one(FT)*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2 # tmp2 is 1/6

                # Right Side (Symmetric)
                # Use V2R...V6R
                t1 = V2R - FT(2.0)*V3R + V4R; t2 = V2R - FT(4.0)*V3R + FT(3.0)*V4R
                s1R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3R - FT(2.0)*V4R + V5R; t2 = V3R - V5R
                s2R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4R - FT(2.0)*V5R + V6R; t2 = FT(3.0)*V4R - FT(4.0)*V5R + V6R
                s3R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; t_d2R = WENOϵ2 + s2R*ss; t_d3R = WENOϵ2 + s3R*ss
                @static if weno_z
                    τ5R = abs(s1R - s3R) * ss
                    α1R = one(FT)*(one(FT)+τ5R/(t_d1R+WENOϵ2))/(t_d1R*t_d1R)
                    α2R = FT(6.0e0)*(one(FT)+τ5R/(t_d2R+WENOϵ2))/(t_d2R*t_d2R)
                    α3R = FT(3.0)*(one(FT)+τ5R/(t_d3R+WENOϵ2))/(t_d3R*t_d3R)
                else
                    α1R = one(FT)/(t_d1R * t_d1R)
                    α2R = FT(6.0e0)/(t_d2R * t_d2R)
                    α3R = FT(3.0)/(t_d3R * t_d3R)
                end
                invsumR = one(FT)/(α1R+α2R+α3R)

                v1 = FT(2.0)*V2R - FT(7.0e0)*V3R + FT(11.0)*V4R
                v2 = -one(FT)*V3R + FT(5.0e0)*V4R + FT(2.0)*V5R
                v3 = FT(2.0)*V4R + FT(5.0e0)*V5R - one(FT)*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + FT(0.5)*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - FT(0.5)*minmod(V4R - V3R, V4R - V5R)
            end
            
            UL_final[n] = valL; UR_final[n] = valR
        end
    end

    # Failsafe: if reconstruction (Branch A or B) produces non-physical state, fall back to 1st order
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    _eiL = UL_final[Ncons] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
    _eiR = UR_final[Ncons] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = U[i,j,k,n]; end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = U[i+1,j,k,n]; end
    end

    # ── Save interface face values for flux sync ──
    if intf_ihi > Int32(0) && i == intf_ihi && UL_save_ihi !== nothing
        @inbounds for n = 1:Ncons; UL_save_ihi[j-NG, k-NG, n] = UL_final[n]; end
    end
    if intf_ilo > Int32(0) && i == intf_ilo && UR_save_ilo !== nothing
        @inbounds for n = 1:Ncons; UR_save_ilo[j-NG, k-NG, n] = UR_final[n]; end
    end

    # 4. 组装并计算通量 (use Tuple construction to avoid StaticArray dynamic dispatch on GPU)
    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    
    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕx, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)

    @inbounds for n = 1:Ncons
        Fx[i-NG+1, j-NG, k-NG, n] = flux_temp[n] * Area
    end
    return 
end

function Conser_reconstruct_j(Q, U, ϕ, S, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp,
                              stencil_arr, Δstencil_arr, lin_phi_arr,
                              stencil_R_arr, Δstencil_R_arr,
                              ch_glm::FT, mode::Int32,
                              intf_jlo::Int32=Int32(-1), intf_jhi::Int32=Int32(-1),
                              UL_save_jhi=nothing, UR_save_jlo=nothing)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. 边界检查
    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG || k < NG+1
        return
    end
    # Interior/boundary mode filter (stencil width=4 in j)
    if mode == Int32(1) && (j < NG+Int32(4) || j > nyp+NG-Int32(4)); return; end
    if mode == Int32(2) && (j >= NG+Int32(4) && j <= nyp+NG-Int32(4)); return; end

    # 2. Geometry
    @inbounds nx=nxj[i,j+1,k]; ny=nyj[i,j+1,k]; nz=nzj[i,j+1,k]; Area=Areaj[i,j+1,k]

    # 3. 激波传感器
    @inbounds ϕy = max(ϕ[i, j-2, k], ϕ[i, j-1, k], ϕ[i, j, k], ϕ[i, j+1, k], ϕ[i, j+2, k], ϕ[i, j+3, k])

    # AC: always use linear reconstruction (branch A)
    if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
        ϕy = zero(FT)
    end

    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # Branch A: Smooth region
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    if ϕy < hybrid_ϕ1
        # ═══ Original 1D 7-point reconstruction ═══
        α_adapt = min(ϕy / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        @inbounds L1 = stencil_arr[j,k,1] + α_adapt * Δstencil_arr[j,k,1]; @inbounds L2 = stencil_arr[j,k,2] + α_adapt * Δstencil_arr[j,k,2]
        @inbounds L3 = stencil_arr[j,k,3] + α_adapt * Δstencil_arr[j,k,3]; @inbounds L4 = stencil_arr[j,k,4] + α_adapt * Δstencil_arr[j,k,4]
        @inbounds L5 = stencil_arr[j,k,5] + α_adapt * Δstencil_arr[j,k,5]; @inbounds L6 = stencil_arr[j,k,6] + α_adapt * Δstencil_arr[j,k,6]
        @inbounds L7 = stencil_arr[j,k,7] + α_adapt * Δstencil_arr[j,k,7]
        for n = 1:Ncons
            @inbounds v1 = U[i,j-3,k,n]; v2 = U[i,j-2,k,n]; v3 = U[i,j-1,k,n]
            @inbounds v4 = U[i,j  ,k,n]; v5 = U[i,j+1,k,n]; v6 = U[i,j+2,k,n]; v7 = U[i,j+3,k,n]
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        @inbounds L1 = stencil_R_arr[j,k,1] + α_adapt * Δstencil_R_arr[j,k,1]; @inbounds L2 = stencil_R_arr[j,k,2] + α_adapt * Δstencil_R_arr[j,k,2]
        @inbounds L3 = stencil_R_arr[j,k,3] + α_adapt * Δstencil_R_arr[j,k,3]; @inbounds L4 = stencil_R_arr[j,k,4] + α_adapt * Δstencil_R_arr[j,k,4]
        @inbounds L5 = stencil_R_arr[j,k,5] + α_adapt * Δstencil_R_arr[j,k,5]; @inbounds L6 = stencil_R_arr[j,k,6] + α_adapt * Δstencil_R_arr[j,k,6]
        @inbounds L7 = stencil_R_arr[j,k,7] + α_adapt * Δstencil_R_arr[j,k,7]
        for n = 1:Ncons
            @inbounds r1 = U[i,j+4,k,n]; r2 = U[i,j+3,k,n]; r3 = U[i,j+2,k,n]
            @inbounds r4 = U[i,j+1,k,n]; r5 = U[i,j  ,k,n]; r6 = U[i,j-1,k,n]; r7 = U[i,j-2,k,n]
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end

    # ==============================
    # Branch B: Discontinuous region
    # ==============================
    else 
        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        # 平滑因子 ss
        @inbounds ss = FT(2.0)/(S[i, j+1, k] + S[i, j, k])

        for n = 1:Ncons
            @inbounds V1L = U[i,j-3,k,n]
            @inbounds V2L = U[i,j-2,k,n]
            @inbounds V3L = U[i,j-1,k,n]
            @inbounds V4L = U[i,j  ,k,n]
            @inbounds V5L = U[i,j+1,k,n]
            @inbounds V6L = U[i,j+2,k,n]
            @inbounds V7L = U[i,j+3,k,n]

            @inbounds V1R = U[i,j+4,k,n]
            @inbounds V2R = U[i,j+3,k,n]
            @inbounds V3R = U[i,j+2,k,n]
            @inbounds V4R = U[i,j+1,k,n]
            @inbounds V5R = U[i,j  ,k,n]
            @inbounds V6R = U[i,j-1,k,n]
            @inbounds V7R = U[i,j-2,k,n]

            valL = zero(FT); valR = zero(FT)

            if ϕy < hybrid_ϕ2 # WENO7
                q1L = -FT(3.0)*V1L + FT(13.0)*V2L - FT(23.0)*V3L + FT(25.0e0)*V4L; q1R = -FT(3.0)*V1R + FT(13.0)*V2R - FT(23.0)*V3R + FT(25.0e0)*V4R
                q2L =  one(FT)*V2L -  FT(5.0e0)*V3L + FT(13.0)*V4L +  FT(3.0)*V5L; q2R =  one(FT)*V2R -  FT(5.0e0)*V3R + FT(13.0)*V4R +  FT(3.0)*V5R
                q3L = -one(FT)*V3L +  FT(7.0e0)*V4L +  FT(7.0e0)*V5L -  one(FT)*V6L; q3R = -one(FT)*V3R +  FT(7.0e0)*V4R +  FT(7.0e0)*V5R -  one(FT)*V6R
                q4L =  FT(3.0)*V4L + FT(13.0)*V5L -  FT(5.0e0)*V6L +  one(FT)*V7L; q4R =  FT(3.0)*V4R + FT(13.0)*V5R -  FT(5.0e0)*V6R +  one(FT)*V7R

                Is1L = V1L*( FT(547.0e0)*V1L - FT(3882.0)*V2L + FT(4642.0)*V3L - FT(1854.0)*V4L) + V2L*( FT(7043.0)*V2L -FT(17246.0e0)*V3L + FT(7042.0)*V4L) + V3L*(FT(11003.0)*V3L - FT(9402.0)*V4L) + V4L*( FT(2107.0e0)*V4L)
                Is2L = V2L*( FT(267.0e0)*V2L - FT(1642.0)*V3L + FT(1602.0)*V4L -  FT(494.0)*V5L) + V3L*( FT(2843.0)*V3L - FT(5966.0e0)*V4L + FT(1922.0)*V5L) + V4L*( FT(3443.0)*V4L - FT(2522.0)*V5L) + V5L*(  FT(547.0e0)*V5L)
                Is3L = V3L*( FT(547.0e0)*V3L - FT(2522.0)*V4L + FT(1922.0)*V5L -  FT(494.0)*V6L) + V4L*( FT(3443.0)*V4L - FT(5966.0e0)*V5L + FT(1602.0)*V6L) + V5L*( FT(2843.0)*V5L - FT(1642.0)*V6L) + V6L*(  FT(267.0e0)*V6L)
                Is4L = V4L*( FT(2107.0e0)*V4L - FT(9402.0)*V5L + FT(7042.0)*V6L - FT(1854.0)*V7L) + V5L*(FT(11003.0)*V5L -FT(17246.0e0)*V6L + FT(4642.0)*V7L) + V6L*( FT(7043.0)*V6L - FT(3882.0)*V7L) + V7L*(  FT(547.0e0)*V7L)

                Is1R = V1R*( FT(547.0e0)*V1R - FT(3882.0)*V2R + FT(4642.0)*V3R - FT(1854.0)*V4R) + V2R*( FT(7043.0)*V2R -FT(17246.0e0)*V3R + FT(7042.0)*V4R) + V3R*(FT(11003.0)*V3R - FT(9402.0)*V4R) + V4R*( FT(2107.0e0)*V4R)
                Is2R = V2R*( FT(267.0e0)*V2R - FT(1642.0)*V3R + FT(1602.0)*V4R -  FT(494.0)*V5R) + V3R*( FT(2843.0)*V3R - FT(5966.0e0)*V4R + FT(1922.0)*V5R) + V4R*( FT(3443.0)*V4R - FT(2522.0)*V5R) + V5R*(  FT(547.0e0)*V5R)
                Is3R = V3R*( FT(547.0e0)*V3R - FT(2522.0)*V4R + FT(1922.0)*V5R -  FT(494.0)*V6R) + V4R*( FT(3443.0)*V4R - FT(5966.0e0)*V5R + FT(1602.0)*V6R) + V5R*( FT(2843.0)*V5R - FT(1642.0)*V6R) + V6R*(  FT(267.0e0)*V6R)
                Is4R = V4R*( FT(2107.0e0)*V4R - FT(9402.0)*V5R + FT(7042.0)*V6R - FT(1854.0)*V7R) + V5R*(FT(11003.0)*V5R -FT(17246.0e0)*V6R + FT(4642.0)*V7R) + V6R*( FT(7043.0)*V6R - FT(3882.0)*V7R) + V7R*(  FT(547.0e0)*V7R)

                t_d1L = WENOϵ1 + Is1L*ss; t_d1R = WENOϵ1 + Is1R*ss
                t_d2L = WENOϵ1 + Is2L*ss; t_d2R = WENOϵ1 + Is2R*ss
                t_d3L = WENOϵ1 + Is3L*ss; t_d3R = WENOϵ1 + Is3R*ss
                t_d4L = WENOϵ1 + Is4L*ss; t_d4R = WENOϵ1 + Is4R*ss
                @static if weno_z
                    τ7L = abs(Is1L - Is4L) * ss; τ7R = abs(Is1R - Is4R) * ss
                    α1L = one(FT)*(one(FT)+τ7L/(t_d1L+WENOϵ1))/(t_d1L*t_d1L); α1R = one(FT)*(one(FT)+τ7R/(t_d1R+WENOϵ1))/(t_d1R*t_d1R)
                    α2L = FT(12.0)*(one(FT)+τ7L/(t_d2L+WENOϵ1))/(t_d2L*t_d2L); α2R = FT(12.0)*(one(FT)+τ7R/(t_d2R+WENOϵ1))/(t_d2R*t_d2R)
                    α3L = FT(18.0e0)*(one(FT)+τ7L/(t_d3L+WENOϵ1))/(t_d3L*t_d3L); α3R = FT(18.0e0)*(one(FT)+τ7R/(t_d3R+WENOϵ1))/(t_d3R*t_d3R)
                    α4L = FT(4.0)*(one(FT)+τ7L/(t_d4L+WENOϵ1))/(t_d4L*t_d4L); α4R = FT(4.0)*(one(FT)+τ7R/(t_d4R+WENOϵ1))/(t_d4R*t_d4R)
                else
                    α1L = one(FT)/(t_d1L * t_d1L); α1R = one(FT)/(t_d1R * t_d1R)
                    α2L = FT(12.0)/(t_d2L * t_d2L); α2R = FT(12.0)/(t_d2R * t_d2R)
                    α3L = FT(18.0e0)/(t_d3L * t_d3L); α3R = FT(18.0e0)/(t_d3R * t_d3R)
                    α4L = FT(4.0)/(t_d4L * t_d4L); α4R = FT(4.0)/(t_d4R * t_d4R)
                end

                invsumL = one(FT)/(α1L+α2L+α3L+α4L); invsumR = one(FT)/(α1R+α2R+α3R+α4R)
                valL = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                valR = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1

            elseif ϕy < hybrid_ϕ3 # WENO5
                t1 = V2L - FT(2.0)*V3L + V4L; t2 = V2L - FT(4.0)*V3L + FT(3.0)*V4L
                s1L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3L - FT(2.0)*V4L + V5L; t2 = V3L - V5L
                s2L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4L - FT(2.0)*V5L + V6L; t2 = FT(3.0)*V4L - FT(4.0)*V5L + V6L
                s3L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1L = WENOϵ2 + s1L*ss; t_d2L = WENOϵ2 + s2L*ss; t_d3L = WENOϵ2 + s3L*ss
                @static if weno_z
                    τ5L = abs(s1L - s3L) * ss
                    α1L = one(FT)*(one(FT)+τ5L/(t_d1L+WENOϵ2))/(t_d1L*t_d1L)
                    α2L = FT(6.0e0)*(one(FT)+τ5L/(t_d2L+WENOϵ2))/(t_d2L*t_d2L)
                    α3L = FT(3.0)*(one(FT)+τ5L/(t_d3L+WENOϵ2))/(t_d3L*t_d3L)
                else
                    α1L = one(FT)/(t_d1L * t_d1L)
                    α2L = FT(6.0e0)/(t_d2L * t_d2L)
                    α3L = FT(3.0)/(t_d3L * t_d3L)
                end
                invsumL = one(FT)/(α1L+α2L+α3L)

                v1 = FT(2.0)*V2L - FT(7.0e0)*V3L + FT(11.0)*V4L
                v2 = -one(FT)*V3L + FT(5.0e0)*V4L + FT(2.0)*V5L
                v3 = FT(2.0)*V4L + FT(5.0e0)*V5L - one(FT)*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2

                t1 = V2R - FT(2.0)*V3R + V4R; t2 = V2R - FT(4.0)*V3R + FT(3.0)*V4R
                s1R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3R - FT(2.0)*V4R + V5R; t2 = V3R - V5R
                s2R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4R - FT(2.0)*V5R + V6R; t2 = FT(3.0)*V4R - FT(4.0)*V5R + V6R
                s3R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; t_d2R = WENOϵ2 + s2R*ss; t_d3R = WENOϵ2 + s3R*ss
                @static if weno_z
                    τ5R = abs(s1R - s3R) * ss
                    α1R = one(FT)*(one(FT)+τ5R/(t_d1R+WENOϵ2))/(t_d1R*t_d1R)
                    α2R = FT(6.0e0)*(one(FT)+τ5R/(t_d2R+WENOϵ2))/(t_d2R*t_d2R)
                    α3R = FT(3.0)*(one(FT)+τ5R/(t_d3R+WENOϵ2))/(t_d3R*t_d3R)
                else
                    α1R = one(FT)/(t_d1R * t_d1R)
                    α2R = FT(6.0e0)/(t_d2R * t_d2R)
                    α3R = FT(3.0)/(t_d3R * t_d3R)
                end
                invsumR = one(FT)/(α1R+α2R+α3R)

                v1 = FT(2.0)*V2R - FT(7.0e0)*V3R + FT(11.0)*V4R
                v2 = -one(FT)*V3R + FT(5.0e0)*V4R + FT(2.0)*V5R
                v3 = FT(2.0)*V4R + FT(5.0e0)*V5R - one(FT)*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + FT(0.5)*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - FT(0.5)*minmod(V4R - V3R, V4R - V5R)
            end
            
            UL_final[n] = valL; UR_final[n] = valR
        end
    end

    # Failsafe: if reconstruction (Branch A or B) produces non-physical state, fall back to 1st order
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    _eiL = UL_final[Ncons] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
    _eiR = UR_final[Ncons] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = U[i,j,k,n]; end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = U[i,j+1,k,n]; end
    end

    # ── Save interface face values for flux sync ──
    if intf_jhi > Int32(0) && j == intf_jhi && UL_save_jhi !== nothing
        @inbounds for n = 1:Ncons; UL_save_jhi[i-NG, k-NG, n] = UL_final[n]; end
    end
    if intf_jlo > Int32(0) && j == intf_jlo && UR_save_jlo !== nothing
        @inbounds for n = 1:Ncons; UR_save_jlo[i-NG, k-NG, n] = UR_final[n]; end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    
    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕy, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)

    @inbounds begin
        # 注意：Fy 的写入位???j 偏移 1
    @inbounds for n = 1:Ncons
        Fy[i-NG, j-NG+1, k-NG, n] = flux_temp[n] * Area
    end
    end
    return 
end

function Conser_reconstruct_k(Q, U, ϕ, S, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp,
                              stencil_arr, Δstencil_arr, lin_phi_arr,
                              stencil_R_arr, Δstencil_R_arr,
                              ch_glm::FT, mode::Int32,
                              intf_klo::Int32=Int32(-1), intf_khi::Int32=Int32(-1),
                              UL_save_khi=nothing, UR_save_klo=nothing)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    # 1. 边界检查
    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG
        return
    end
    # Interior/boundary mode filter (stencil width=4 in k)
    if mode == Int32(1) && (k < NG+Int32(4) || k > nzp+NG-Int32(4)); return; end
    if mode == Int32(2) && (k >= NG+Int32(4) && k <= nzp+NG-Int32(4)); return; end

    # 2. Geometry
    @inbounds nx=nxk[i,j,k+1]; ny=nyk[i,j,k+1]; nz=nzk[i,j,k+1]; Area=Areak[i,j,k+1]

    # 3. 激波传感器
    @inbounds ϕz = max(ϕ[i, j, k-2], ϕ[i, j, k-1], ϕ[i, j, k], ϕ[i, j+1, k], ϕ[i, j+2, k], ϕ[i, j+3, k])

    # AC: always use linear reconstruction (branch A)
    if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
        ϕz = zero(FT)
    end

    UL_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))
    UR_final = MVector{Ncons, FT}(ntuple(_ -> zero(FT), Val(Ncons)))

    # ==============================
    # Branch A: Smooth region
    # ==============================
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    if ϕz < hybrid_ϕ1
        α_adapt = min(ϕz / hybrid_ϕ1, one(FT)) * (one(FT) - local_lin_ϕ)
        @inbounds L1 = stencil_arr[j,k,1] + α_adapt * Δstencil_arr[j,k,1]; @inbounds L2 = stencil_arr[j,k,2] + α_adapt * Δstencil_arr[j,k,2]
        @inbounds L3 = stencil_arr[j,k,3] + α_adapt * Δstencil_arr[j,k,3]; @inbounds L4 = stencil_arr[j,k,4] + α_adapt * Δstencil_arr[j,k,4]
        @inbounds L5 = stencil_arr[j,k,5] + α_adapt * Δstencil_arr[j,k,5]; @inbounds L6 = stencil_arr[j,k,6] + α_adapt * Δstencil_arr[j,k,6]
        @inbounds L7 = stencil_arr[j,k,7] + α_adapt * Δstencil_arr[j,k,7]
        for n = 1:Ncons
            @inbounds v1 = U[i,j,k-3,n]; v2 = U[i,j,k-2,n]; v3 = U[i,j,k-1,n]
            @inbounds v4 = U[i,j,k  ,n]; v5 = U[i,j,k+1,n]; v6 = U[i,j,k+2,n]; v7 = U[i,j,k+3,n]
            UL_final[n] = L1*v1 + L2*v2 + L3*v3 + L4*v4 + L5*v5 + L6*v6 + L7*v7
        end
        @inbounds L1 = stencil_R_arr[j,k,1] + α_adapt * Δstencil_R_arr[j,k,1]; @inbounds L2 = stencil_R_arr[j,k,2] + α_adapt * Δstencil_R_arr[j,k,2]
        @inbounds L3 = stencil_R_arr[j,k,3] + α_adapt * Δstencil_R_arr[j,k,3]; @inbounds L4 = stencil_R_arr[j,k,4] + α_adapt * Δstencil_R_arr[j,k,4]
        @inbounds L5 = stencil_R_arr[j,k,5] + α_adapt * Δstencil_R_arr[j,k,5]; @inbounds L6 = stencil_R_arr[j,k,6] + α_adapt * Δstencil_R_arr[j,k,6]
        @inbounds L7 = stencil_R_arr[j,k,7] + α_adapt * Δstencil_R_arr[j,k,7]
        for n = 1:Ncons
            @inbounds r1 = U[i,j,k+4,n]; r2 = U[i,j,k+3,n]; r3 = U[i,j,k+2,n]
            @inbounds r4 = U[i,j,k+1,n]; r5 = U[i,j,k  ,n]; r6 = U[i,j,k-1,n]; r7 = U[i,j,k-2,n]
            UR_final[n] = L1*r1 + L2*r2 + L3*r3 + L4*r4 + L5*r5 + L6*r6 + L7*r7
        end

    # ==============================
    # Branch B: Discontinuous region
    # ==============================
    else 
        WENOϵ1 = FT(1.0e-10); WENOϵ2 = FT(1.0e-8)
        tmp1 = one(FT)/FT(12.0); tmp2 = one(FT)/FT(6.0e0)
        # 平滑因子 ss
        @inbounds ss = FT(2.0)/(S[i, j, k+1] + S[i, j, k])

        for n = 1:Ncons
            @inbounds V1L = U[i,j,k-3,n]
            @inbounds V2L = U[i,j,k-2,n]
            @inbounds V3L = U[i,j,k-1,n]
            @inbounds V4L = U[i,j,k  ,n]
            @inbounds V5L = U[i,j,k+1,n]
            @inbounds V6L = U[i,j,k+2,n]
            @inbounds V7L = U[i,j,k+3,n]

            @inbounds V1R = U[i,j,k+4,n]
            @inbounds V2R = U[i,j,k+3,n]
            @inbounds V3R = U[i,j,k+2,n]
            @inbounds V4R = U[i,j,k+1,n]
            @inbounds V5R = U[i,j,k  ,n]
            @inbounds V6R = U[i,j,k-1,n]
            @inbounds V7R = U[i,j,k-2,n]

            valL = zero(FT); valR = zero(FT)

            if ϕz < hybrid_ϕ2 # WENO7
                q1L = -FT(3.0)*V1L + FT(13.0)*V2L - FT(23.0)*V3L + FT(25.0e0)*V4L; q1R = -FT(3.0)*V1R + FT(13.0)*V2R - FT(23.0)*V3R + FT(25.0e0)*V4R
                q2L =  one(FT)*V2L -  FT(5.0e0)*V3L + FT(13.0)*V4L +  FT(3.0)*V5L; q2R =  one(FT)*V2R -  FT(5.0e0)*V3R + FT(13.0)*V4R +  FT(3.0)*V5R
                q3L = -one(FT)*V3L +  FT(7.0e0)*V4L +  FT(7.0e0)*V5L -  one(FT)*V6L; q3R = -one(FT)*V3R +  FT(7.0e0)*V4R +  FT(7.0e0)*V5R -  one(FT)*V6R
                q4L =  FT(3.0)*V4L + FT(13.0)*V5L -  FT(5.0e0)*V6L +  one(FT)*V7L; q4R =  FT(3.0)*V4R + FT(13.0)*V5R -  FT(5.0e0)*V6R +  one(FT)*V7R

                Is1L = V1L*( FT(547.0e0)*V1L - FT(3882.0)*V2L + FT(4642.0)*V3L - FT(1854.0)*V4L) + V2L*( FT(7043.0)*V2L -FT(17246.0e0)*V3L + FT(7042.0)*V4L) + V3L*(FT(11003.0)*V3L - FT(9402.0)*V4L) + V4L*( FT(2107.0e0)*V4L)
                Is2L = V2L*( FT(267.0e0)*V2L - FT(1642.0)*V3L + FT(1602.0)*V4L -  FT(494.0)*V5L) + V3L*( FT(2843.0)*V3L - FT(5966.0e0)*V4L + FT(1922.0)*V5L) + V4L*( FT(3443.0)*V4L - FT(2522.0)*V5L) + V5L*(  FT(547.0e0)*V5L)
                Is3L = V3L*( FT(547.0e0)*V3L - FT(2522.0)*V4L + FT(1922.0)*V5L -  FT(494.0)*V6L) + V4L*( FT(3443.0)*V4L - FT(5966.0e0)*V5L + FT(1602.0)*V6L) + V5L*( FT(2843.0)*V5L - FT(1642.0)*V6L) + V6L*(  FT(267.0e0)*V6L)
                Is4L = V4L*( FT(2107.0e0)*V4L - FT(9402.0)*V5L + FT(7042.0)*V6L - FT(1854.0)*V7L) + V5L*(FT(11003.0)*V5L -FT(17246.0e0)*V6L + FT(4642.0)*V7L) + V6L*( FT(7043.0)*V6L - FT(3882.0)*V7L) + V7L*(  FT(547.0e0)*V7L)

                Is1R = V1R*( FT(547.0e0)*V1R - FT(3882.0)*V2R + FT(4642.0)*V3R - FT(1854.0)*V4R) + V2R*( FT(7043.0)*V2R -FT(17246.0e0)*V3R + FT(7042.0)*V4R) + V3R*(FT(11003.0)*V3R - FT(9402.0)*V4R) + V4R*( FT(2107.0e0)*V4R)
                Is2R = V2R*( FT(267.0e0)*V2R - FT(1642.0)*V3R + FT(1602.0)*V4R -  FT(494.0)*V5R) + V3R*( FT(2843.0)*V3R - FT(5966.0e0)*V4R + FT(1922.0)*V5R) + V4R*( FT(3443.0)*V4R - FT(2522.0)*V5R) + V5R*(  FT(547.0e0)*V5R)
                Is3R = V3R*( FT(547.0e0)*V3R - FT(2522.0)*V4R + FT(1922.0)*V5R -  FT(494.0)*V6R) + V4R*( FT(3443.0)*V4R - FT(5966.0e0)*V5R + FT(1602.0)*V6R) + V5R*( FT(2843.0)*V5R - FT(1642.0)*V6R) + V6R*(  FT(267.0e0)*V6R)
                Is4R = V4R*( FT(2107.0e0)*V4R - FT(9402.0)*V5R + FT(7042.0)*V6R - FT(1854.0)*V7R) + V5R*(FT(11003.0)*V5R -FT(17246.0e0)*V6R + FT(4642.0)*V7R) + V6R*( FT(7043.0)*V6R - FT(3882.0)*V7R) + V7R*(  FT(547.0e0)*V7R)

                t_d1L = WENOϵ1 + Is1L*ss; t_d1R = WENOϵ1 + Is1R*ss
                t_d2L = WENOϵ1 + Is2L*ss; t_d2R = WENOϵ1 + Is2R*ss
                t_d3L = WENOϵ1 + Is3L*ss; t_d3R = WENOϵ1 + Is3R*ss
                t_d4L = WENOϵ1 + Is4L*ss; t_d4R = WENOϵ1 + Is4R*ss
                @static if weno_z
                    τ7L = abs(Is1L - Is4L) * ss; τ7R = abs(Is1R - Is4R) * ss
                    α1L = one(FT)*(one(FT)+τ7L/(t_d1L+WENOϵ1))/(t_d1L*t_d1L); α1R = one(FT)*(one(FT)+τ7R/(t_d1R+WENOϵ1))/(t_d1R*t_d1R)
                    α2L = FT(12.0)*(one(FT)+τ7L/(t_d2L+WENOϵ1))/(t_d2L*t_d2L); α2R = FT(12.0)*(one(FT)+τ7R/(t_d2R+WENOϵ1))/(t_d2R*t_d2R)
                    α3L = FT(18.0e0)*(one(FT)+τ7L/(t_d3L+WENOϵ1))/(t_d3L*t_d3L); α3R = FT(18.0e0)*(one(FT)+τ7R/(t_d3R+WENOϵ1))/(t_d3R*t_d3R)
                    α4L = FT(4.0)*(one(FT)+τ7L/(t_d4L+WENOϵ1))/(t_d4L*t_d4L); α4R = FT(4.0)*(one(FT)+τ7R/(t_d4R+WENOϵ1))/(t_d4R*t_d4R)
                else
                    α1L = one(FT)/(t_d1L * t_d1L); α1R = one(FT)/(t_d1R * t_d1R)
                    α2L = FT(12.0)/(t_d2L * t_d2L); α2R = FT(12.0)/(t_d2R * t_d2R)
                    α3L = FT(18.0e0)/(t_d3L * t_d3L); α3R = FT(18.0e0)/(t_d3R * t_d3R)
                    α4L = FT(4.0)/(t_d4L * t_d4L); α4R = FT(4.0)/(t_d4R * t_d4R)
                end

                invsumL = one(FT)/(α1L+α2L+α3L+α4L); invsumR = one(FT)/(α1R+α2R+α3R+α4R)
                valL = invsumL*(α1L*q1L+α2L*q2L+α3L*q3L+α4L*q4L) * tmp1
                valR = invsumR*(α1R*q1R+α2R*q2R+α3R*q3R+α4R*q4R) * tmp1

            elseif ϕz < hybrid_ϕ3 # WENO5
                t1 = V2L - FT(2.0)*V3L + V4L; t2 = V2L - FT(4.0)*V3L + FT(3.0)*V4L
                s1L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3L - FT(2.0)*V4L + V5L; t2 = V3L - V5L
                s2L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4L - FT(2.0)*V5L + V6L; t2 = FT(3.0)*V4L - FT(4.0)*V5L + V6L
                s3L = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1L = WENOϵ2 + s1L*ss; t_d2L = WENOϵ2 + s2L*ss; t_d3L = WENOϵ2 + s3L*ss
                @static if weno_z
                    τ5L = abs(s1L - s3L) * ss
                    α1L = one(FT)*(one(FT)+τ5L/(t_d1L+WENOϵ2))/(t_d1L*t_d1L)
                    α2L = FT(6.0e0)*(one(FT)+τ5L/(t_d2L+WENOϵ2))/(t_d2L*t_d2L)
                    α3L = FT(3.0)*(one(FT)+τ5L/(t_d3L+WENOϵ2))/(t_d3L*t_d3L)
                else
                    α1L = one(FT)/(t_d1L * t_d1L)
                    α2L = FT(6.0e0)/(t_d2L * t_d2L)
                    α3L = FT(3.0)/(t_d3L * t_d3L)
                end
                invsumL = one(FT)/(α1L+α2L+α3L)

                v1 = FT(2.0)*V2L - FT(7.0e0)*V3L + FT(11.0)*V4L
                v2 = -one(FT)*V3L + FT(5.0e0)*V4L + FT(2.0)*V5L
                v3 = FT(2.0)*V4L + FT(5.0e0)*V5L - one(FT)*V6L
                
                valL = invsumL * (α1L*v1 + α2L*v2 + α3L*v3) * tmp2

                t1 = V2R - FT(2.0)*V3R + V4R; t2 = V2R - FT(4.0)*V3R + FT(3.0)*V4R
                s1R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V3R - FT(2.0)*V4R + V5R; t2 = V3R - V5R
                s2R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2
                t1 = V4R - FT(2.0)*V5R + V6R; t2 = FT(3.0)*V4R - FT(4.0)*V5R + V6R
                s3R = FT(13.0) * t1*t1 + FT(3.0) * t2*t2

                t_d1R = WENOϵ2 + s1R*ss; t_d2R = WENOϵ2 + s2R*ss; t_d3R = WENOϵ2 + s3R*ss
                @static if weno_z
                    τ5R = abs(s1R - s3R) * ss
                    α1R = one(FT)*(one(FT)+τ5R/(t_d1R+WENOϵ2))/(t_d1R*t_d1R)
                    α2R = FT(6.0e0)*(one(FT)+τ5R/(t_d2R+WENOϵ2))/(t_d2R*t_d2R)
                    α3R = FT(3.0)*(one(FT)+τ5R/(t_d3R+WENOϵ2))/(t_d3R*t_d3R)
                else
                    α1R = one(FT)/(t_d1R * t_d1R)
                    α2R = FT(6.0e0)/(t_d2R * t_d2R)
                    α3R = FT(3.0)/(t_d3R * t_d3R)
                end
                invsumR = one(FT)/(α1R+α2R+α3R)

                v1 = FT(2.0)*V2R - FT(7.0e0)*V3R + FT(11.0)*V4R
                v2 = -one(FT)*V3R + FT(5.0e0)*V4R + FT(2.0)*V5R
                v3 = FT(2.0)*V4R + FT(5.0e0)*V5R - one(FT)*V6R
                
                valR = invsumR * (α1R*v1 + α2R*v2 + α3R*v3) * tmp2

            else # Minmod
                valL = V4L + FT(0.5)*minmod(V4L - V3L, V5L - V4L)
                valR = V4R - FT(0.5)*minmod(V4R - V3R, V4R - V5R)
            end
            
            UL_final[n] = valL; UR_final[n] = valR
        end
    end

    # Failsafe: if reconstruction (Branch A or B) produces non-physical state, fall back to 1st order
    @inbounds _ρL = UL_final[1]; _ρR = UR_final[1]
    _ρuL2 = UL_final[2]^2 + UL_final[3]^2 + UL_final[4]^2
    _ρuR2 = UR_final[2]^2 + UR_final[3]^2 + UR_final[4]^2
    _eiL = UL_final[Ncons] - FT(0.5) * _ρuL2 / max(_ρL, eps(FT))
    _eiR = UR_final[Ncons] - FT(0.5) * _ρuR2 / max(_ρR, eps(FT))
    if !(_ρL >= eps(FT)) || !(_eiL >= eps(FT)) || !isfinite(_ρL) || !isfinite(_eiL)
        for n = 1:Ncons; @inbounds UL_final[n] = U[i,j,k,n]; end
    end
    if !(_ρR >= eps(FT)) || !(_eiR >= eps(FT)) || !isfinite(_ρR) || !isfinite(_eiR)
        for n = 1:Ncons; @inbounds UR_final[n] = U[i,j,k+1,n]; end
    end

    # ── Save interface face values for flux sync ──
    if intf_khi > Int32(0) && k == intf_khi && UL_save_khi !== nothing
        @inbounds for n = 1:Ncons; UL_save_khi[i-NG, j-NG, n] = UL_final[n]; end
    end
    if intf_klo > Int32(0) && k == intf_klo && UR_save_klo !== nothing
        @inbounds for n = 1:Ncons; UR_save_klo[i-NG, j-NG, n] = UR_final[n]; end
    end

    # 4. 组装并计算通量
    UL_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UL_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    UR_vec = SVector{Ncons, FT}(ntuple(n -> @inbounds(UR_final[n]), Val(Ncons))::NTuple{Ncons, FT})
    
    # Hybrid flux: Continuous blending of KEP with an upwind Riemann flux
    @inbounds local_lin_ϕ = lin_phi_arr[j,k]
    flux_temp = Blend_Flux(UL_vec, UR_vec, nx, ny, nz, ϕz, hybrid_ϕ1, local_lin_ϕ, splitMethodID, ch_glm)

    @inbounds begin
        # 注意：Fz 的写入位???k 偏移 1
    @inbounds for n = 1:Ncons
        Fz[i-NG, j-NG, k-NG+1, n] = flux_temp[n] * Area
    end
    end
    return
end
