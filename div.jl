function div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    @inbounds Jact::FT = J[i+NG, j+NG, k+NG] * dt

    if viscous
        for n = 1:Ncons
            @inbounds U[i+NG, j+NG, k+NG, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n]) -
                (Fv_x[i, j, k, n] - Fv_x[i+1, j, k, n]) - 
                (Fv_y[i, j, k, n] - Fv_y[i, j+1, k, n]) - 
                (Fv_z[i, j, k, n] - Fv_z[i, j, k+1, n])
            ) * Jact
        end
    else
        for n = 1:Ncons
            @inbounds U[i+NG, j+NG, k+NG, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n])
            ) * Jact
        end
    end
    return
end

function div_LTS(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    @inbounds Jact::FT = J[i+NG, j+NG, k+NG] * dt[i+NG, j+NG, k+NG]

    if viscous
        for n = 1:Ncons
            @inbounds U[i+NG, j+NG, k+NG, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n]) -
                (Fv_x[i, j, k, n] - Fv_x[i+1, j, k, n]) - 
                (Fv_y[i, j, k, n] - Fv_y[i, j+1, k, n]) - 
                (Fv_z[i, j, k, n] - Fv_z[i, j, k+1, n])
            ) * Jact
        end
    else
        for n = 1:Ncons
            @inbounds U[i+NG, j+NG, k+NG, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n])
            ) * Jact
        end
    end
    return
end

# ─── Fused div + RK combination + clipping + c2Prim ───
# Combines 2 separate kernels into 1 pass:
#   1. div:           U += dt/Vol * (∇·F_inv - ∇·F_vis)
#   2. linComb_clip:  U = Un + rk_a*(U - Un),  clip ρ/p,  Q = prim(U)
# Saves: 1 kernel launch + 1 full read-write of U array through HBM
# VGPR estimate: ~32 (div intermediates released before c2Prim starts)
function div_rk_clip_prim(U, Un, Q, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z,
                          dt, J, rk_a::FT, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii, jj, kk = i+NG, j+NG, k+NG

    # Step 1: Divergence → update U in place
    @inbounds Jact::FT = J[ii, jj, kk] * dt

    if viscous
        for n = 1:Ncons
            @inbounds U[ii, jj, kk, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n]) -
                (Fv_x[i, j, k, n] - Fv_x[i+1, j, k, n]) - 
                (Fv_y[i, j, k, n] - Fv_y[i, j+1, k, n]) - 
                (Fv_z[i, j, k, n] - Fv_z[i, j, k+1, n])
            ) * Jact
        end
    else
        for n = 1:Ncons
            @inbounds U[ii, jj, kk, n] += (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) + 
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) + 
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n])
            ) * Jact
        end
    end

    # Step 2: RK linear combination  U = Un + rk_a*(U - Un)
    for n = 1:Ncons
        @inbounds U[ii, jj, kk, n] = Un[ii, jj, kk, n] + rk_a * (U[ii, jj, kk, n] - Un[ii, jj, kk, n])
    end

    # Step 3: Clipping + Conservative → Primitive
    # AC mode: Q == U (identity, no clipping)
    if equation_type == :incompressible_AC
        for n = 1:Ncons
            @inbounds Q[ii, jj, kk, n] = U[ii, jj, kk, n]
        end
        return
    end

    # MHD mode: c2Prim with B²/2 in energy
    if equation_type == :MHD
        @inbounds ρ = max(U[ii, jj, kk, 1], eps(FT))
        ρinv = one(FT) / ρ
        @inbounds u_v = U[ii, jj, kk, 2] * ρinv
        @inbounds v_v = U[ii, jj, kk, 3] * ρinv
        @inbounds w_v = U[ii, jj, kk, 4] * ρinv
        @inbounds Bx = U[ii,jj,kk,6]; @inbounds By = U[ii,jj,kk,7]; @inbounds Bz = U[ii,jj,kk,8]
        B2 = Bx*Bx + By*By + Bz*Bz
        @inbounds ei = max(U[ii,jj,kk,5] - FT(0.5)*ρ*(u_v*u_v+v_v*v_v+w_v*w_v) - FT(0.5)*B2, eps(FT))
        p_v = (γ - one(FT)) * ei
        ρ = max(ρ, FT(1.0e-5)); p_v = max(p_v, FT(1.0e-5))
        T_v = p_v / (ρ * Rg)
        @inbounds Q[ii,jj,kk,1]=ρ;  Q[ii,jj,kk,2]=u_v; Q[ii,jj,kk,3]=v_v; Q[ii,jj,kk,4]=w_v
        @inbounds Q[ii,jj,kk,5]=p_v; Q[ii,jj,kk,6]=T_v
        @inbounds Q[ii,jj,kk,7]=Bx; Q[ii,jj,kk,8]=By; Q[ii,jj,kk,9]=Bz
        @inbounds Q[ii,jj,kk,10]=U[ii,jj,kk,9]
        @inbounds U[ii,jj,kk,1]=ρ
        @inbounds U[ii,jj,kk,2]=ρ*u_v; U[ii,jj,kk,3]=ρ*v_v; U[ii,jj,kk,4]=ρ*w_v
        @inbounds U[ii,jj,kk,5]=p_v/(γ-one(FT))+FT(0.5)*ρ*(u_v*u_v+v_v*v_v+w_v*w_v)+FT(0.5)*B2
        return
    end

    @inbounds ρ = max(U[ii, jj, kk, 1], eps(FT))
    ρinv = one(FT) / ρ
    @inbounds u_v = U[ii, jj, kk, 2] * ρinv
    @inbounds v_v = U[ii, jj, kk, 3] * ρinv
    @inbounds w_v = U[ii, jj, kk, 4] * ρinv
    @inbounds ei = max(U[ii, jj, kk, 5] - FT(0.5)*ρ*(u_v*u_v + v_v*v_v + w_v*w_v), eps(FT))
    p_v = (γ - one(FT)) * ei

    # Step 3: Positivity + temperature clipping (inline)
    ρ_min = FT(1.0e-5)
    p_min = FT(1.0e-5)
    T_clamp_lo = FT(50.0)
    T_clamp_hi = FT(2.0) * Tw
    ρ = max(ρ, ρ_min)
    p_v = max(p_v, p_min)

    T_v = p_v / (ρ * Rg)
    T_v = clamp(T_v, T_clamp_lo, T_clamp_hi)
    p_v = ρ * Rg * T_v  # recompute p from clamped T for consistency

    @inbounds Q[ii, jj, kk, 1] = ρ
    @inbounds Q[ii, jj, kk, 2] = u_v
    @inbounds Q[ii, jj, kk, 3] = v_v
    @inbounds Q[ii, jj, kk, 4] = w_v
    @inbounds Q[ii, jj, kk, 5] = p_v
    @inbounds Q[ii, jj, kk, 6] = T_v

    # Always write back consistent U (T clamp may have changed p)
    @inbounds U[ii, jj, kk, 1] = ρ
    @inbounds U[ii, jj, kk, 2] = ρ * u_v
    @inbounds U[ii, jj, kk, 3] = ρ * v_v
    @inbounds U[ii, jj, kk, 4] = ρ * w_v
    @inbounds U[ii, jj, kk, 5] = p_v / (γ-one(FT)) + FT(0.5) * ρ * (u_v*u_v + v_v*v_v + w_v*w_v)
    return
end

# ─── Implicit RHS kernel ───
# Computes RHS = -(divF - divFv) * (Vol/dt) + source
# Output goes to dU_rhs buffer (NOT in-place update to U)
# Used by LU-SGS implicit time stepping
function div_to_rhs(dU_rhs, U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z,
                    dU_forced, dt, J, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    # Vol/dt factor (Vol = 1/J is the cell volume, but here J = 1/Vol stored)
    # In the existing code: Jact = J[i+NG, j+NG, k+NG] * dt
    # For the implicit RHS we want: RHS = flux_divergence * (Vol * dt) + source * dt
    # But actually the LU-SGS equation is: (Vol/dt + ...) * ΔU = -R(U^n)
    # where R(U^n) = -flux_divergence * Vol - source * Vol
    # So RHS = (-R) = flux_divergence * Vol + source * Vol
    # We store: dU_rhs = flux_div * Vol + source_term
    # (Note: dt is NOT included here; the diagonal D = Vol/dt handles the scaling)

    @inbounds vol_inv::FT = J[i+NG, j+NG, k+NG]  # J = 1/Vol

    if viscous
        for n = 1:Ncons
            @inbounds flux_div = (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) +
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) +
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n]) -
                (Fv_x[i, j, k, n] - Fv_x[i+1, j, k, n]) -
                (Fv_y[i, j, k, n] - Fv_y[i, j+1, k, n]) -
                (Fv_z[i, j, k, n] - Fv_z[i, j, k+1, n])
            )
            # source term (from volume forces, intensive = per unit volume)
            @inbounds src = flow_forcing || test_case == "HIT" ? dU_forced[i, j, k, n] : zero(FT)
            # LU-SGS equation: (V/dt + σ/2) × ΔU = RHS
            # flux_div is EXTENSIVE (already × face area, units [N] for momentum)
            # src is INTENSIVE (force per unit volume, units [N/m³] for momentum)
            # To match dimensions: RHS = flux_div + src × V = flux_div + src / vol_inv
            # Explicit does: ΔU = flux_div×dt/V + src×dt
            # Implicit at σ=0: ΔU = (flux_div + src/vol_inv) × dt/V
            #                     = flux_div×dt/V + src×dt  ✓
            @inbounds dU_rhs[i, j, k, n] = flux_div + src / (vol_inv + FT(1.0e-30))
        end
    else
        for n = 1:Ncons
            @inbounds flux_div = (
                (Fx[i, j, k, n]   - Fx[i+1, j, k, n]) +
                (Fy[i, j, k, n]   - Fy[i, j+1, k, n]) +
                (Fz[i, j, k, n]   - Fz[i, j, k+1, n])
            )
            @inbounds src = flow_forcing || test_case == "HIT" ? dU_forced[i, j, k, n] : zero(FT)
            @inbounds dU_rhs[i, j, k, n] = flux_div + src / (vol_inv + FT(1.0e-30))
        end
    end
    return
end
