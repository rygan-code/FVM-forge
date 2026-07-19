function div_update_kernel!(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    @inbounds Jact::FT = J[i+NG, j+NG, k+NG] * dt

    if viscous || (equation_type == :MHD && resistive)
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

    if viscous || (equation_type == :MHD && resistive)
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

    # Step 1: Divergence update U (Nhydro=5 in CT mode, Ncons=9 in GLM mode)
    # CT mode: Fx/Fy/Fz have 1 ghost layer offset in tangential directions
    @static if ct_mode
        _oj = Int32(1); _ok = Int32(1)
    else
        _oj = Int32(0); _ok = Int32(0)
    end
    @inbounds Jact::FT = J[ii, jj, kk] * dt
    if viscous || (equation_type == :MHD && resistive)
        for n = 1:Nhydro
            @inbounds U[ii, jj, kk, n] += (
                (Fx[i, j+_oj, k+_ok, n]   - Fx[i+1, j+_oj, k+_ok, n]) +
                (Fy[i+_oj, j, k+_ok, n]   - Fy[i+_oj, j+1, k+_ok, n]) +
                (Fz[i+_oj, j+_oj, k, n]   - Fz[i+_oj, j+_oj, k+1, n]) -
                (Fv_x[i, j+_oj, k+_ok, n] - Fv_x[i+1, j+_oj, k+_ok, n]) -
                (Fv_y[i+_oj, j, k+_ok, n] - Fv_y[i+_oj, j+1, k+_ok, n]) -
                (Fv_z[i+_oj, j+_oj, k, n] - Fv_z[i+_oj, j+_oj, k+1, n])
            ) * Jact
        end
    else
        for n = 1:Nhydro
            @inbounds U[ii, jj, kk, n] += (
                (Fx[i, j+_oj, k+_ok, n]   - Fx[i+1, j+_oj, k+_ok, n]) +
                (Fy[i+_oj, j, k+_ok, n]   - Fy[i+_oj, j+1, k+_ok, n]) +
                (Fz[i+_oj, j+_oj, k, n]   - Fz[i+_oj, j+_oj, k+1, n])
            ) * Jact
        end
    end

    # Step 2: RK linear combination (only hydro vars in CT mode)
    for n = 1:Nhydro
        @inbounds U[ii, jj, kk, n] = Un[ii, jj, kk, n] + rk_a * (U[ii, jj, kk, n] - Un[ii, jj, kk, n])
    end

    # MHD mode: c2Prim with B²/2 in energy
    if equation_type == :MHD
        @static if strict_ct_positivity && ct_mode && splitMethodID == 4
            @inbounds state = SVector{9,FT}(
                (
                    U[ii, jj, kk, 1], U[ii, jj, kk, 2], U[ii, jj, kk, 3],
                    U[ii, jj, kk, 4], U[ii, jj, kk, 5], U[ii, jj, kk, 6],
                    U[ii, jj, kk, 7], U[ii, jj, kk, 8], U[ii, jj, kk, 9],
                ),
            )
            ρ, kinetic, magnetic, ei, p_v = mhd_raw_thermo(state, γ)
            ρinv = one(FT) / ρ
            u_v = state[2] * ρinv
            v_v = state[3] * ρinv
            w_v = state[4] * ρinv
            T_v = p_v / (ρ * Rg)
            @inbounds Q[ii,jj,kk,1]=ρ;  Q[ii,jj,kk,2]=u_v; Q[ii,jj,kk,3]=v_v; Q[ii,jj,kk,4]=w_v
            @inbounds Q[ii,jj,kk,5]=p_v; Q[ii,jj,kk,6]=T_v
            @inbounds Q[ii,jj,kk,7]=state[6]; Q[ii,jj,kk,8]=state[7]; Q[ii,jj,kk,9]=state[8]
            @inbounds Q[ii,jj,kk,10]=state[9]
            return
        end

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
        # In CT mode: B is updated separately by CT (not by div), so U[6:8] are stale.
        # Write back ρ and momentum (with clipping), but NOT U[5] -- the energy U[5]
        # contains the correct flux update (including Poynting flux). Writing U[5]
        # with the stale B would corrupt the energy. The B correction and pressure
        # recompute happen in ct_update_q_b! after CT sync.
        # In GLM mode: write back all U[1:5] with clipped values (B already updated by div).
        @static if ct_mode
            @inbounds U[ii,jj,kk,1]=ρ
            @inbounds U[ii,jj,kk,2]=ρ*u_v; U[ii,jj,kk,3]=ρ*v_v; U[ii,jj,kk,4]=ρ*w_v
        else
            @inbounds U[ii,jj,kk,1]=ρ
            @inbounds U[ii,jj,kk,2]=ρ*u_v; U[ii,jj,kk,3]=ρ*v_v; U[ii,jj,kk,4]=ρ*w_v
            @inbounds U[ii,jj,kk,5]=p_v/(γ-one(FT))+FT(0.5)*ρ*(u_v*u_v+v_v*v_v+w_v*w_v)+FT(0.5)*B2
        end
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

    if viscous || (equation_type == :MHD && resistive)
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
