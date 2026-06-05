# =============================================================================
# Multi-block boundary conditions — Data-driven BC dispatch (FVM)
# BC types defined in bc_types.jl, stored per-face in block_connectivity.h5
# =============================================================================

# ─── Wall perturbation dispatcher ───
# wall_perturbation_type: 0 = pipe (cylindrical multi-mode), 1 = flat plate (blowing/suction strip)
function _wall_perturbation(i, j, k, rx, x, y, z, tt, nxp)
    if !wall_perturbation
        return zero(FT), zero(FT), zero(FT)
    end
    if wall_perturbation_type == Int32(1)
        return _wall_perturbation_flatplate(i, j, k, x, y, z, tt)
    end
    return _wall_perturbation_pipe(i, j, k, x, y, z, tt)
end

# =============================================================================
# Flat Plate Wall-Normal Blowing/Suction Strip Perturbation
# 
# Targets Mack 2nd mode instability for hypersonic BL transition.
# Only injects wall-normal velocity (v) at the isothermal wall.
#
# v_wall(x, z, t) = A · f_x(x) · f_z(z) · f_t(t) · ramp(t)
#
# - f_x: Gaussian streamwise envelope (narrow strip near inlet)
# - f_z: multi-mode spanwise structure for 3D breakdown
# - f_t: multi-frequency temporal oscillation covering 2nd mode range
# - ramp: smooth onset/shutoff to avoid impulsive transient
#
# Ref: Zhong & Wang (2012) JFM, Franko & Lele (2013) JFM
# =============================================================================
function _wall_perturbation_flatplate(i, j, k, x, y, z, tt)
    tt_f = FT(tt)

    # ── Reference scales ──
    # Perturbation amplitude: 0.1% of u_inf (small enough for linear receptivity)
    amp = FT(0.001e0) * u_inf_ref

    # ── Streamwise envelope: narrow Gaussian strip ──
    @inbounds x_loc = x[i, j, k]
    x_center = FT(0.075e0) * Lx_phys             # strip center at 7.5% of actual physical domain
    σ_x = FT(0.02e0) * Lx_phys                   # strip half-width ~ 2% of actual physical domain
    f_x = exp(-FT(0.5) * ((x_loc - x_center) / σ_x)^2)

    # Early exit if far from strip (GPU optimization)
    if f_x < FT(1.0e-6)
        return zero(FT), zero(FT), zero(FT)
    end

    # ── Spanwise structure: multi-mode for oblique breakdown ──
    # β_fund corresponds to spanwise wavelength = Lz_phys (one full wave exactly across physical span).
    # This correctly ensures STRICT periodicity at the domain boundaries.
    @inbounds z_loc = z[i, j, k]
    β_fund = FT(2.0) * FT(π) / Lz_phys

    f_z = (  FT(0.40e0) * sin(β_fund * z_loc)                            # m=1 fundamental
           + FT(0.30e0) * sin(FT(2.0) * β_fund * z_loc + FT(1.2e0))           # m=2 subharmonic
           + FT(0.20e0) * sin(FT(3.0) * β_fund * z_loc + FT(2.7e0))           # m=3
           + FT(0.10e0) * sin(FT(4.0) * β_fund * z_loc + FT(4.1e0)))          # m=4

    # ── Temporal oscillation: broadband multi-frequency ──
    # For Ma=6, 2nd mode frequency: f ~ u_e / (2δ)
    # Estimate: BL thickness at strip ~ 5·x_strip / √(Re_x_strip)
    μ_w = C_s * Tw * sqrt(Tw) / (Tw + T_s)
    ρ_ref = Re_target * μ_w / (u_inf_ref * FT(2.0) * R0 + FT(1.0e-30))
    x_phys = x_center
    ν_ref = μ_w / (ρ_ref + FT(1.0e-30))
    δ_strip = FT(5.0e0) * sqrt(ν_ref * x_phys / (u_inf_ref + FT(1.0e-30)) + FT(1.0e-30))
    
    # 2nd mode base frequency
    ω_2nd = FT(π) * u_inf_ref / (δ_strip + FT(1.0e-30))

    # Broadband: 5 frequencies
    f_t = (  FT(0.30e0) * sin(FT(0.5) * ω_2nd * tt_f)
           + FT(0.25) * sin(one(FT) * ω_2nd * tt_f + FT(0.9e0))
           + FT(0.20e0) * sin(FT(1.5e0) * ω_2nd * tt_f + FT(1.8e0))
           + FT(0.15e0) * sin(FT(2.0) * ω_2nd * tt_f + FT(3.1e0))
           + FT(0.10e0) * sin(FT(2.5e0) * ω_2nd * tt_f + FT(4.7e0)))

    # ── Temporal ramp: smooth cosine onset over 1 flow-through time ──
    t_conv = Lx_phys / u_inf_ref
    if tt_f < t_conv
        ramp = FT(0.5) * (one(FT) - cos(FT(π) * tt_f / t_conv))
    elseif tt_f < FT(6.0e0) * t_conv
        ramp = one(FT)
    elseif tt_f < FT(8.0e0) * t_conv
        ramp = FT(0.5) * (one(FT) + cos(FT(π) * (tt_f - FT(6.0e0)*t_conv) / (FT(2.0)*t_conv)))
    else
        ramp = zero(FT)                                                         # off
    end

    # Only wall-normal (v) perturbation — pure blowing/suction
    v_turb = amp * f_x * f_z * f_t * ramp

    return zero(FT), v_turb, zero(FT)
end

# =============================================================================
# Pipe Flow Cylindrical Multi-Mode Wall Perturbation (original)
# =============================================================================
function _wall_perturbation_pipe(i, j, k, x, y, z, tt)
    tt_f = FT(tt)
    c_wall_val = sqrt(γ * Rg * Tw)
    u_bulk_val = Ma_target * c_wall_val
    amp = FT(0.005e0) * u_bulk_val

    θ = atan(z[i, j, k], y[i, j, k])
    x_loc = x[i, j, k]

    α_1 = FT(2.0) * FT(π) / Lx
    α_2 = FT(2.0) * α_1
    α_3 = FT(3.0) * α_1
    α_4 = FT(4.0) * α_1

    ϕ_g = FT(1.6180339887e0)
    c_phase = FT(0.6e0) * u_bulk_val

    ω_1 = α_1 * c_phase
    ω_2 = α_2 * c_phase * FT(0.9e0)
    ω_3 = α_3 * c_phase * FT(1.1e0)
    ω_4 = α_4 * c_phase

    V_wave = (
        FT(0.15e0) * sin(α_1 * x_loc + FT(2.0) * θ - ω_1 * tt_f) +
        FT(0.15e0) * sin(α_1 * x_loc + FT(3.0) * θ - ω_1 * tt_f + ϕ_g) +
        FT(0.12e0) * sin(α_2 * x_loc + FT(4.0) * θ - ω_2 * tt_f + FT(2.0)*ϕ_g) +
        FT(0.12e0) * sin(α_2 * x_loc - FT(3.0) * θ - ω_2 * tt_f + FT(3.0)*ϕ_g) +
        FT(0.10e0) * sin(α_3 * x_loc + FT(6.0e0) * θ - ω_3 * tt_f + FT(4.0)*ϕ_g) +
        FT(0.10e0) * sin(α_3 * x_loc - FT(2.0) * θ - ω_3 * tt_f + FT(5.0e0)*ϕ_g) +
        FT(0.08e0) * sin(α_4 * x_loc + FT(3.0) * θ - ω_4 * tt_f + FT(6.0e0)*ϕ_g) +
        FT(0.08e0) * sin(α_4 * x_loc - FT(4.0) * θ - ω_4 * tt_f + FT(7.0e0)*ϕ_g) +
        FT(0.05e0) * sin(α_1 * x_loc + FT(6.0e0) * θ - ω_1 * tt_f * FT(0.7e0) + FT(8.0e0)*ϕ_g) +
        FT(0.05e0) * sin(α_2 * x_loc + FT(2.0) * θ - ω_2 * tt_f * FT(1.3e0) + FT(9.0e0)*ϕ_g)
    )

    V_wave_θ = (
        FT(0.15e0) * FT(2.0) * cos(α_1 * x_loc + FT(2.0) * θ - ω_1 * tt_f) +
        FT(0.15e0) * FT(3.0) * cos(α_1 * x_loc + FT(3.0) * θ - ω_1 * tt_f + ϕ_g) +
        FT(0.12e0) * FT(4.0) * cos(α_2 * x_loc + FT(4.0) * θ - ω_2 * tt_f + FT(2.0)*ϕ_g) +
        FT(0.12e0) *(-FT(3.0))* cos(α_2 * x_loc - FT(3.0) * θ - ω_2 * tt_f + FT(3.0)*ϕ_g) +
        FT(0.10e0) * FT(6.0e0) * cos(α_3 * x_loc + FT(6.0e0) * θ - ω_3 * tt_f + FT(4.0)*ϕ_g) +
        FT(0.10e0) *(-FT(2.0))* cos(α_3 * x_loc - FT(2.0) * θ - ω_3 * tt_f + FT(5.0e0)*ϕ_g) +
        FT(0.08e0) * FT(3.0) * cos(α_4 * x_loc + FT(3.0) * θ - ω_4 * tt_f + FT(6.0e0)*ϕ_g) +
        FT(0.08e0) *(-FT(4.0))* cos(α_4 * x_loc - FT(4.0) * θ - ω_4 * tt_f + FT(7.0e0)*ϕ_g) +
        FT(0.05e0) * FT(6.0e0) * cos(α_1 * x_loc + FT(6.0e0) * θ - ω_1 * tt_f * FT(0.7e0) + FT(8.0e0)*ϕ_g) +
        FT(0.05e0) * FT(2.0) * cos(α_2 * x_loc + FT(2.0) * θ - ω_2 * tt_f * FT(1.3e0) + FT(9.0e0)*ϕ_g)
    )

    t_conv = Lx / u_bulk_val
    t_factor = tt_f < FT(2.0)*t_conv ? one(FT) : (tt_f <= FT(4.0)*t_conv ? one(FT) - (tt_f - FT(2.0)*t_conv) / (FT(2.0)*t_conv) : zero(FT))

    if t_factor > zero(FT)
        r_inv = one(FT) / sqrt(y[i, j, k]^2 + z[i, j, k]^2 + FT(1.0e-30))
        cos_θ = y[i, j, k] * r_inv
        sin_θ = z[i, j, k] * r_inv
        V_r  = amp * V_wave * t_factor
        V_θ  = amp * V_wave_θ * t_factor * FT(0.15e0)
        u_t = zero(FT)
        v_t = V_r * cos_θ - V_θ * sin_θ
        w_t = V_r * sin_θ + V_θ * cos_θ
        return u_t, v_t, w_t
    end
    return zero(FT), zero(FT), zero(FT)
end

# =============================================================================
# Unified BC helper — supports all 3 directions
# dir: 1=ξ, 2=η, 3=ζ   side: 0=lo, 1=hi
# =============================================================================
function _apply_bc!(Q, U, i, j, k, bc_type, dir, side,
                    n_x, n_y, n_z, x, y, z,
                    nxp, nyp, nzp, bcp, tt, dsrfg_params)

    if bc_type == Int32(BC_INTERBLOCK) || bc_type == Int32(BC_PERIODIC)
        return
    end

    # ── Helper indices: boundary cell and mirror interior cell ──
    # idx_bnd: boundary cell index (first interior cell)
    # idx_int: mirror interior cell for ghost reflection
    if dir == Int32(1) # ξ
        if side == Int32(0)
            idx_bnd = NG + 1
            idx_int = 2*NG + 1 - i
        else
            idx_bnd = nxp + NG
            idx_int = 2*(nxp+NG) + 1 - i
        end
    elseif dir == Int32(2) # η
        if side == Int32(0)
            idx_bnd = NG + 1
            idx_int = 2*NG + 1 - j
        else
            idx_bnd = nyp + NG
            idx_int = 2*(nyp+NG) + 1 - j
        end
    else # ζ
        if side == Int32(0)
            idx_bnd = NG + 1
            idx_int = 2*NG + 1 - k
        else
            idx_bnd = nzp + NG
            idx_int = 2*(nzp+NG) + 1 - k
        end
    end

    # idx_wall: face node index for normals (1-based, node-indexed)
    if dir == Int32(1)
        idx_wall = side == Int32(0) ? (1 + NG) : (nxp + NG + 1)
    elseif dir == Int32(2)
        idx_wall = side == Int32(0) ? (1 + NG) : (nyp + NG + 1)
    else
        idx_wall = side == Int32(0) ? (1 + NG) : (nzp + NG + 1)
    end

    # ── Macros: read Q/U from boundary cell or mirror cell by direction ──
    # _Q_bnd(n): Q at boundary cell
    # _Q_int(n): Q at mirror interior cell
    # _U_bnd(n): U at boundary cell
    # _U_int(n): U at mirror interior cell

    # ── Isothermal Wall ──
    if bc_type == Int32(BC_ISOTHERMAL_WALL)
        Tw_val = bcp[1]  # BCP_TW
        if Tw_val <= zero(FT); Tw_val = Tw; end
        if dir == Int32(1)
            u_turb, v_turb, w_turb = _wall_perturbation(idx_wall, j, k, Int32(0), x, y, z, tt, nxp)
            @inbounds begin
                u = -Q[idx_int, j, k, 2] + FT(2.0)*u_turb
                v = -Q[idx_int, j, k, 3] + FT(2.0)*v_turb
                w = -Q[idx_int, j, k, 4] + FT(2.0)*w_turb
                T = FT(2.0)*Tw_val - Q[idx_int, j, k, 6]
                T = max(T, FT(0.1)*Tw_val)  # clamp to avoid negative T when T_int >> 2*Tw
                p = Q[idx_int, j, k, 5]
                ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=p/(γ-one(FT))+FT(0.5)*ρ*(u^2+v^2+w^2)
            end
        elseif dir == Int32(2)
            u_turb, v_turb, w_turb = _wall_perturbation(i, idx_wall, k, Int32(0), x, y, z, tt, nxp)
            @inbounds begin
                u = -Q[i, idx_int, k, 2] + FT(2.0)*u_turb
                v = -Q[i, idx_int, k, 3] + FT(2.0)*v_turb
                w = -Q[i, idx_int, k, 4] + FT(2.0)*w_turb
                T = FT(2.0)*Tw_val - Q[i, idx_int, k, 6]
                T = max(T, FT(0.1)*Tw_val)
                p = Q[i, idx_int, k, 5]
                ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=p/(γ-one(FT))+FT(0.5)*ρ*(u^2+v^2+w^2)
            end
        else # ζ
            u_turb, v_turb, w_turb = _wall_perturbation(i, j, idx_wall, Int32(0), x, y, z, tt, nxp)
            @inbounds begin
                u = -Q[i, j, idx_int, 2] + FT(2.0)*u_turb
                v = -Q[i, j, idx_int, 3] + FT(2.0)*v_turb
                w = -Q[i, j, idx_int, 4] + FT(2.0)*w_turb
                T = FT(2.0)*Tw_val - Q[i, j, idx_int, 6]
                T = max(T, FT(0.1)*Tw_val)
                p = Q[i, j, idx_int, 5]
                ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=p/(γ-one(FT))+FT(0.5)*ρ*(u^2+v^2+w^2)
            end
        end

    # ── Adiabatic Wall ──
    elseif bc_type == Int32(BC_ADIABATIC_WALL)
        if dir == Int32(1)
            @inbounds begin
                u = -Q[idx_int, j, k, 2]; v = -Q[idx_int, j, k, 3]; w = -Q[idx_int, j, k, 4]
                T = Q[idx_int, j, k, 6]; p = Q[idx_int, j, k, 5]; ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=p/(γ-one(FT))+FT(0.5)*ρ*(u^2+v^2+w^2)
            end
        elseif dir == Int32(2)
            @inbounds begin
                u = -Q[i, idx_int, k, 2]; v = -Q[i, idx_int, k, 3]; w = -Q[i, idx_int, k, 4]
                T = Q[i, idx_int, k, 6]; p = Q[i, idx_int, k, 5]; ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=p/(γ-one(FT))+FT(0.5)*ρ*(u^2+v^2+w^2)
            end
        else
            @inbounds begin
                u = -Q[i, j, idx_int, 2]; v = -Q[i, j, idx_int, 3]; w = -Q[i, j, idx_int, 4]
                T = Q[i, j, idx_int, 6]; p = Q[i, j, idx_int, 5]; ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=p/(γ-one(FT))+FT(0.5)*ρ*(u^2+v^2+w^2)
            end
        end

    # ── Slip Wall / Symmetry — reflect normal velocity using face normals ──
    elseif bc_type == Int32(BC_SLIP_WALL) || bc_type == Int32(BC_SYMMETRY)
        if dir == Int32(1)
            @inbounds begin
                nx = n_x[idx_wall, j, k]; ny = n_y[idx_wall, j, k]; nz = n_z[idx_wall, j, k]
                ρ = Q[idx_int, j, k, 1]; u = Q[idx_int, j, k, 2]; v = Q[idx_int, j, k, 3]
                w = Q[idx_int, j, k, 4]; p = Q[idx_int, j, k, 5]; T = Q[idx_int, j, k, 6]
                vn = u*nx + v*ny + w*nz
                u_g = u - FT(2.0)*vn*nx; v_g = v - FT(2.0)*vn*ny; w_g = w - FT(2.0)*vn*nz
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u_g; Q[i,j,k,3]=v_g; Q[i,j,k,4]=w_g; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u_g; U[i,j,k,3]=ρ*v_g; U[i,j,k,4]=ρ*w_g
                U[i,j,k,5]=p/(γ-one(FT))+FT(0.5)*ρ*(u_g^2+v_g^2+w_g^2)
            end
        elseif dir == Int32(2)
            @inbounds begin
                nx = n_x[i, idx_wall, k]; ny = n_y[i, idx_wall, k]; nz = n_z[i, idx_wall, k]
                ρ = Q[i, idx_int, k, 1]; u = Q[i, idx_int, k, 2]; v = Q[i, idx_int, k, 3]
                w = Q[i, idx_int, k, 4]; p = Q[i, idx_int, k, 5]; T = Q[i, idx_int, k, 6]
                vn = u*nx + v*ny + w*nz
                u_g = u - FT(2.0)*vn*nx; v_g = v - FT(2.0)*vn*ny; w_g = w - FT(2.0)*vn*nz
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u_g; Q[i,j,k,3]=v_g; Q[i,j,k,4]=w_g; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u_g; U[i,j,k,3]=ρ*v_g; U[i,j,k,4]=ρ*w_g
                U[i,j,k,5]=p/(γ-one(FT))+FT(0.5)*ρ*(u_g^2+v_g^2+w_g^2)
            end
        else
            @inbounds begin
                nx = n_x[i, j, idx_wall]; ny = n_y[i, j, idx_wall]; nz = n_z[i, j, idx_wall]
                ρ = Q[i, j, idx_int, 1]; u = Q[i, j, idx_int, 2]; v = Q[i, j, idx_int, 3]
                w = Q[i, j, idx_int, 4]; p = Q[i, j, idx_int, 5]; T = Q[i, j, idx_int, 6]
                vn = u*nx + v*ny + w*nz
                u_g = u - FT(2.0)*vn*nx; v_g = v - FT(2.0)*vn*ny; w_g = w - FT(2.0)*vn*nz
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u_g; Q[i,j,k,3]=v_g; Q[i,j,k,4]=w_g; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u_g; U[i,j,k,3]=ρ*v_g; U[i,j,k,4]=ρ*w_g
                U[i,j,k,5]=p/(γ-one(FT))+FT(0.5)*ρ*(u_g^2+v_g^2+w_g^2)
            end
        end

    # ── Supersonic Inflow — all variables fixed ──
    elseif bc_type == Int32(BC_SUPERSONIC_INFLOW)
        @inbounds begin
            ρ1 = bcp[3]; u1 = bcp[4]; v1 = bcp[5]; w1 = bcp[6]; p1 = bcp[7]
            T1 = p1/(ρ1*Rg)
            Q[i,j,k,1]=ρ1; Q[i,j,k,2]=u1; Q[i,j,k,3]=v1; Q[i,j,k,4]=w1; Q[i,j,k,5]=p1; Q[i,j,k,6]=T1
            U[i,j,k,1]=ρ1; U[i,j,k,2]=ρ1*u1; U[i,j,k,3]=ρ1*v1; U[i,j,k,4]=ρ1*w1
            U[i,j,k,5]=p1/(γ-one(FT))+FT(0.5)*ρ1*(u1^2+v1^2+w1^2)
        end

    # ── Transition Inflow — Spatially and temporally varying inflow ──
    elseif bc_type == Int32(BC_TRANSITION_INFLOW)
        @inbounds begin
            # Extrapolate pressure from interior (subsonic characteristic)
            if dir == Int32(1)
                p_ext = Q[idx_bnd,j,k,5]
            elseif dir == Int32(2)
                p_ext = Q[i,idx_bnd,k,5]
            else
                p_ext = Q[i,j,idx_bnd,5]
            end
            
            # Base Flow (1/7 power law for turbulent profile)
            yi = y[i,j,k]; zi = z[i,j,k]; xi = x[i,j,k]
            r2 = yi*yi + zi*zi
            r2_norm = r2 / (R0*R0)
            
            c_wall = sqrt(γ * Rg * Tw)
            u_bulk = Ma_target * c_wall
            
            # 1/7 power law base profile: u_cl = 60/49 * u_bulk
            u_cl = (FT(60.0)/FT(49.0)) * u_bulk
            u_base = u_cl * (max(one(FT) - sqrt(r2_norm), zero(FT))^(one(FT)/FT(7.0)))
            
            # Base temperature profile with viscous heating
            Ma2 = Ma_target * Ma_target
            β = FT(0.5) * Pr * (γ - one(FT)) * Ma2
            T_base = Tw * (one(FT) + β * max(one(FT) - r2_norm*r2_norm, zero(FT)))
            
            # Fluctuations
            if dsrfg_params.enabled
                u_pr, v_pr, w_pr = dsrfg_fluctuation(xi, yi, zi, FT(tt), dsrfg_params)
                
                # Strong Reynolds Analogy (SRA) for temperature fluctuation:
                # T' = - C_sra * ((γ-1)/(γ*Rg)) * u_base * u_pr
                T_pr = - dsrfg_params.C_sra * ((γ - one(FT)) / (γ * Rg)) * u_base * u_pr
            else
                # Fallback to the original deterministic multi-mode perturbations
                envelope = max(one(FT) - r2_norm, zero(FT))
                tt_f = FT(tt)
                amp = FT(0.05e0) * u_bulk  # 5% turbulence intensity at inlet
                θ = atan(zi, yi)
                ω_0 = FT(2.0) * FT(π) * u_bulk / R0  # characteristic frequency
                
                u_pr = amp * envelope * (sin(FT(2.0)*θ - FT(0.5)*ω_0*tt_f) + cos(FT(3.0)*θ - FT(1.2)*ω_0*tt_f))
                v_pr = amp * envelope * (sin(FT(4.0)*θ - FT(0.8)*ω_0*tt_f) + cos(FT(1.0)*θ - FT(1.5)*ω_0*tt_f))
                w_pr = amp * envelope * (cos(FT(2.0)*θ - FT(0.9)*ω_0*tt_f) + sin(FT(5.0)*θ - FT(1.1)*ω_0*tt_f))
                
                T_pr = zero(FT)
            end
            
            u1 = u_base + u_pr
            v1 = v_pr
            w1 = w_pr
            
            T1 = max(T_base + T_pr, FT(50.0))  # clamp to prevent negative/unphysical temperature
            
            ρ1 = p_ext / (Rg * T1)
            
            Q[i,j,k,1]=ρ1; Q[i,j,k,2]=u1; Q[i,j,k,3]=v1; Q[i,j,k,4]=w1; Q[i,j,k,5]=p_ext; Q[i,j,k,6]=T1
            U[i,j,k,1]=ρ1; U[i,j,k,2]=ρ1*u1; U[i,j,k,3]=ρ1*v1; U[i,j,k,4]=ρ1*w1
            U[i,j,k,5]=p_ext/(γ-one(FT))+FT(0.5)*ρ1*(u1^2+v1^2+w1^2)
        end

    # ── Supersonic Outflow — zero gradient (copy from boundary cell) ──
    elseif bc_type == Int32(BC_SUPERSONIC_OUTFLOW)
        if dir == Int32(1)
            for n = 1:Nprim; @inbounds Q[i,j,k,n] = Q[idx_bnd,j,k,n]; end
            for n = 1:Ncons; @inbounds U[i,j,k,n] = U[idx_bnd,j,k,n]; end
        elseif dir == Int32(2)
            for n = 1:Nprim; @inbounds Q[i,j,k,n] = Q[i,idx_bnd,k,n]; end
            for n = 1:Ncons; @inbounds U[i,j,k,n] = U[i,idx_bnd,k,n]; end
        else
            for n = 1:Nprim; @inbounds Q[i,j,k,n] = Q[i,j,idx_bnd,n]; end
            for n = 1:Ncons; @inbounds U[i,j,k,n] = U[i,j,idx_bnd,n]; end
        end

    # ── Zero Gradient (Neumann) — copy from boundary cell ──
    elseif bc_type == Int32(BC_ZERO_GRADIENT)
        if dir == Int32(1)
            for n = 1:Nprim; @inbounds Q[i,j,k,n] = Q[idx_bnd,j,k,n]; end
            for n = 1:Ncons; @inbounds U[i,j,k,n] = U[idx_bnd,j,k,n]; end
        elseif dir == Int32(2)
            for n = 1:Nprim; @inbounds Q[i,j,k,n] = Q[i,idx_bnd,k,n]; end
            for n = 1:Ncons; @inbounds U[i,j,k,n] = U[i,idx_bnd,k,n]; end
        else
            for n = 1:Nprim; @inbounds Q[i,j,k,n] = Q[i,j,idx_bnd,n]; end
            for n = 1:Ncons; @inbounds U[i,j,k,n] = U[i,j,idx_bnd,n]; end
        end

    # ── Subsonic Inflow — total pressure + total temperature + direction ──
    elseif bc_type == Int32(BC_SUBSONIC_INFLOW)
        @inbounds begin
            P0 = bcp[10]; T0 = bcp[11]
            dx = bcp[12]; dy = bcp[13]; dz = bcp[14]
            if dir == Int32(1)
                p_ext = Q[idx_bnd,j,k,5]
            elseif dir == Int32(2)
                p_ext = Q[i,idx_bnd,k,5]
            else
                p_ext = Q[i,j,idx_bnd,5]
            end
            T_local = T0 * (p_ext/P0)^((γ-one(FT))/γ)
            ρ_local = p_ext / (Rg * T_local)
            c_local = sqrt(γ * Rg * T_local)
            Ma_local = sqrt(max(FT(2.0)/(γ-one(FT))*((T0/T_local)-one(FT)), zero(FT)))
            V_mag = Ma_local * c_local
            Q[i,j,k,1]=ρ_local; Q[i,j,k,2]=V_mag*dx; Q[i,j,k,3]=V_mag*dy; Q[i,j,k,4]=V_mag*dz
            Q[i,j,k,5]=p_ext; Q[i,j,k,6]=T_local
            U[i,j,k,1]=ρ_local; U[i,j,k,2]=ρ_local*V_mag*dx; U[i,j,k,3]=ρ_local*V_mag*dy; U[i,j,k,4]=ρ_local*V_mag*dz
            U[i,j,k,5]=p_ext/(γ-one(FT))+FT(0.5)*ρ_local*V_mag^2
        end

    # ── Subsonic Outflow — back pressure specified ──
    elseif bc_type == Int32(BC_SUBSONIC_OUTFLOW)
        @inbounds begin
            p_back = bcp[2]  # BCP_P_TARGET
            if dir == Int32(1)
                ρ = Q[idx_bnd,j,k,1]; u = Q[idx_bnd,j,k,2]; v = Q[idx_bnd,j,k,3]; w = Q[idx_bnd,j,k,4]
            elseif dir == Int32(2)
                ρ = Q[i,idx_bnd,k,1]; u = Q[i,idx_bnd,k,2]; v = Q[i,idx_bnd,k,3]; w = Q[i,idx_bnd,k,4]
            else
                ρ = Q[i,j,idx_bnd,1]; u = Q[i,j,idx_bnd,2]; v = Q[i,j,idx_bnd,3]; w = Q[i,j,idx_bnd,4]
            end
            T = p_back/(ρ*Rg)
            Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p_back; Q[i,j,k,6]=T
            U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
            U[i,j,k,5]=p_back/(γ-one(FT))+FT(0.5)*ρ*(u^2+v^2+w^2)
        end

    # ── NSCBC Outflow — Poinsot & Lele 1992, full LODI formulation ──
    # Outgoing waves: computed from interior gradients (one-sided 2nd order)
    # Incoming wave L1: pressure relaxation toward target
    # Supersonic (Ma_n>1): all waves exit → zero gradient
    elseif bc_type == Int32(BC_NSCBC_OUTFLOW)
        @inbounds begin
            # Read boundary cell primitives
            if dir == Int32(1)
                ρ_b=Q[idx_bnd,j,k,1]; u_b=Q[idx_bnd,j,k,2]; v_b=Q[idx_bnd,j,k,3]; w_b=Q[idx_bnd,j,k,4]; p_b=Q[idx_bnd,j,k,5]
                u_n = u_b  # normal velocity for ξ
            elseif dir == Int32(2)
                ρ_b=Q[i,idx_bnd,k,1]; u_b=Q[i,idx_bnd,k,2]; v_b=Q[i,idx_bnd,k,3]; w_b=Q[i,idx_bnd,k,4]; p_b=Q[i,idx_bnd,k,5]
                u_n = v_b  # normal velocity for η
            else
                ρ_b=Q[i,j,idx_bnd,1]; u_b=Q[i,j,idx_bnd,2]; v_b=Q[i,j,idx_bnd,3]; w_b=Q[i,j,idx_bnd,4]; p_b=Q[i,j,idx_bnd,5]
                u_n = w_b  # normal velocity for ζ
            end
            c = sqrt(γ * p_b / (ρ_b + FT(1.0e-30)))
            Ma_n = abs(u_n) / (c + FT(1.0e-30))

            if Ma_n > one(FT)
                # Supersonic: all characteristics exit → zero gradient
                if dir == Int32(1)
                    for n = 1:Nprim; Q[i,j,k,n] = Q[idx_bnd,j,k,n]; end
                    for n = 1:Ncons; U[i,j,k,n] = U[idx_bnd,j,k,n]; end
                elseif dir == Int32(2)
                    for n = 1:Nprim; Q[i,j,k,n] = Q[i,idx_bnd,k,n]; end
                    for n = 1:Ncons; U[i,j,k,n] = U[i,idx_bnd,k,n]; end
                else
                    for n = 1:Nprim; Q[i,j,k,n] = Q[i,j,idx_bnd,n]; end
                    for n = 1:Ncons; U[i,j,k,n] = U[i,j,idx_bnd,n]; end
                end
            else
                # ── Subsonic NSCBC: full LODI wave decomposition ──
                p_target = bcp[2]; sigma = bcp[8]; L_ref = bcp[9]
                if sigma <= zero(FT); sigma = FT(0.25); end
                if L_ref <= zero(FT); L_ref = one(FT); end

                # One-sided derivatives (2nd order) at boundary cell
                # hi side: df = 1.5*f_b - 2*f_{b-1} + 0.5*f_{b-2}
                # lo side: df = -(1.5*f_b - 2*f_{b+1} + 0.5*f_{b+2})
                if dir == Int32(1)
                    if side == Int32(1)  # ξ+
                        i1 = idx_bnd-1; i2 = idx_bnd-2
                        dρ = FT(1.5e0)*ρ_b - FT(2.0)*Q[i1,j,k,1] + FT(0.5)*Q[i2,j,k,1]
                        du = FT(1.5e0)*u_b - FT(2.0)*Q[i1,j,k,2] + FT(0.5)*Q[i2,j,k,2]
                        dv = FT(1.5e0)*v_b - FT(2.0)*Q[i1,j,k,3] + FT(0.5)*Q[i2,j,k,3]
                        dw = FT(1.5e0)*w_b - FT(2.0)*Q[i1,j,k,4] + FT(0.5)*Q[i2,j,k,4]
                        dp = FT(1.5e0)*p_b - FT(2.0)*Q[i1,j,k,5] + FT(0.5)*Q[i2,j,k,5]
                    else  # ξ-
                        i1 = idx_bnd+1; i2 = idx_bnd+2
                        dρ = -(FT(1.5e0)*ρ_b - FT(2.0)*Q[i1,j,k,1] + FT(0.5)*Q[i2,j,k,1])
                        du = -(FT(1.5e0)*u_b - FT(2.0)*Q[i1,j,k,2] + FT(0.5)*Q[i2,j,k,2])
                        dv = -(FT(1.5e0)*v_b - FT(2.0)*Q[i1,j,k,3] + FT(0.5)*Q[i2,j,k,3])
                        dw = -(FT(1.5e0)*w_b - FT(2.0)*Q[i1,j,k,4] + FT(0.5)*Q[i2,j,k,4])
                        dp = -(FT(1.5e0)*p_b - FT(2.0)*Q[i1,j,k,5] + FT(0.5)*Q[i2,j,k,5])
                    end
                    du_n = du   # ∂u/∂ξ (normal velocity derivative)
                    L3 = u_n * dv; L4 = u_n * dw  # shear waves (tangential)
                elseif dir == Int32(2)
                    if side == Int32(1)  # η+
                        j1 = idx_bnd-1; j2 = idx_bnd-2
                        dρ = FT(1.5e0)*ρ_b - FT(2.0)*Q[i,j1,k,1] + FT(0.5)*Q[i,j2,k,1]
                        du = FT(1.5e0)*u_b - FT(2.0)*Q[i,j1,k,2] + FT(0.5)*Q[i,j2,k,2]
                        dv = FT(1.5e0)*v_b - FT(2.0)*Q[i,j1,k,3] + FT(0.5)*Q[i,j2,k,3]
                        dw = FT(1.5e0)*w_b - FT(2.0)*Q[i,j1,k,4] + FT(0.5)*Q[i,j2,k,4]
                        dp = FT(1.5e0)*p_b - FT(2.0)*Q[i,j1,k,5] + FT(0.5)*Q[i,j2,k,5]
                    else  # η-
                        j1 = idx_bnd+1; j2 = idx_bnd+2
                        dρ = -(FT(1.5e0)*ρ_b - FT(2.0)*Q[i,j1,k,1] + FT(0.5)*Q[i,j2,k,1])
                        du = -(FT(1.5e0)*u_b - FT(2.0)*Q[i,j1,k,2] + FT(0.5)*Q[i,j2,k,2])
                        dv = -(FT(1.5e0)*v_b - FT(2.0)*Q[i,j1,k,3] + FT(0.5)*Q[i,j2,k,3])
                        dw = -(FT(1.5e0)*w_b - FT(2.0)*Q[i,j1,k,4] + FT(0.5)*Q[i,j2,k,4])
                        dp = -(FT(1.5e0)*p_b - FT(2.0)*Q[i,j1,k,5] + FT(0.5)*Q[i,j2,k,5])
                    end
                    du_n = dv   # ∂v/∂η (normal velocity derivative)
                    L3 = u_n * du; L4 = u_n * dw  # shear waves (tangential)
                else  # ζ
                    if side == Int32(1)  # ζ+
                        k1 = idx_bnd-1; k2 = idx_bnd-2
                        dρ = FT(1.5e0)*ρ_b - FT(2.0)*Q[i,j,k1,1] + FT(0.5)*Q[i,j,k2,1]
                        du = FT(1.5e0)*u_b - FT(2.0)*Q[i,j,k1,2] + FT(0.5)*Q[i,j,k2,2]
                        dv = FT(1.5e0)*v_b - FT(2.0)*Q[i,j,k1,3] + FT(0.5)*Q[i,j,k2,3]
                        dw = FT(1.5e0)*w_b - FT(2.0)*Q[i,j,k1,4] + FT(0.5)*Q[i,j,k2,4]
                        dp = FT(1.5e0)*p_b - FT(2.0)*Q[i,j,k1,5] + FT(0.5)*Q[i,j,k2,5]
                    else  # ζ-
                        k1 = idx_bnd+1; k2 = idx_bnd+2
                        dρ = -(FT(1.5e0)*ρ_b - FT(2.0)*Q[i,j,k1,1] + FT(0.5)*Q[i,j,k2,1])
                        du = -(FT(1.5e0)*u_b - FT(2.0)*Q[i,j,k1,2] + FT(0.5)*Q[i,j,k2,2])
                        dv = -(FT(1.5e0)*v_b - FT(2.0)*Q[i,j,k1,3] + FT(0.5)*Q[i,j,k2,3])
                        dw = -(FT(1.5e0)*w_b - FT(2.0)*Q[i,j,k1,4] + FT(0.5)*Q[i,j,k2,4])
                        dp = -(FT(1.5e0)*p_b - FT(2.0)*Q[i,j,k1,5] + FT(0.5)*Q[i,j,k2,5])
                    end
                    du_n = dw   # ∂w/∂ζ (normal velocity derivative)
                    L3 = u_n * du; L4 = u_n * dv  # shear waves (tangential)
                end

                # Wave amplitudes
                L5 = (u_n + c) * (dp + ρ_b * c * du_n)  # outgoing acoustic
                L2 = u_n * (c^2 * dρ - dp)               # entropy
                # L1: incoming acoustic — pressure relaxation (Poinsot & Lele)
                L1 = sigma * c * (one(FT) - Ma_n^2) / L_ref * (p_b - p_target)

                # LODI update relations
                d1 = (L2 + FT(0.5)*(L5+L1)) / (c^2 + FT(1.0e-30))
                d2 = FT(0.5)*(L5-L1) / (ρ_b*c + FT(1.0e-30))
                d3 = L3
                d4 = L4
                d5 = FT(0.5)*(L5+L1)

                # Set ghost = boundary value modified by LODI
                ρ_g = ρ_b - d1; p_g = p_b - d5
                ρ_g = max(ρ_g, FT(1.0e-10)); p_g = max(p_g, FT(1.0e-10))
                if dir == Int32(1)  # ξ: normal=u, tangential=v,w
                    u_g = u_b - d2; v_g = v_b - d3; w_g = w_b - d4
                elseif dir == Int32(2)  # η: normal=v, tangential=u,w
                    v_g = v_b - d2; u_g = u_b - d3; w_g = w_b - d4
                else  # ζ: normal=w, tangential=u,v
                    w_g = w_b - d2; u_g = u_b - d3; v_g = v_b - d4
                end
                T_g = p_g / (ρ_g * Rg + FT(1.0e-30))
                Q[i,j,k,1]=ρ_g; Q[i,j,k,2]=u_g; Q[i,j,k,3]=v_g; Q[i,j,k,4]=w_g; Q[i,j,k,5]=p_g; Q[i,j,k,6]=T_g
                U[i,j,k,1]=ρ_g; U[i,j,k,2]=ρ_g*u_g; U[i,j,k,3]=ρ_g*v_g; U[i,j,k,4]=ρ_g*w_g
                U[i,j,k,5]=p_g/(γ-one(FT))+FT(0.5)*ρ_g*(u_g^2+v_g^2+w_g^2)
            end
        end

    # ── Farfield — Riemann invariant based ──
    elseif bc_type == Int32(BC_FARFIELD)
        @inbounds begin
            ρ_inf = bcp[3]; u_inf = bcp[4]; v_inf = bcp[5]; w_inf = bcp[6]; p_inf = bcp[7]
            if dir == Int32(1)
                ρ_i = Q[idx_bnd,j,k,1]; u_i = Q[idx_bnd,j,k,2]; v_i = Q[idx_bnd,j,k,3]; w_i = Q[idx_bnd,j,k,4]; p_i = Q[idx_bnd,j,k,5]
                u_n_i = u_i; u_n_inf = u_inf
            elseif dir == Int32(2)
                ρ_i = Q[i,idx_bnd,k,1]; u_i = Q[i,idx_bnd,k,2]; v_i = Q[i,idx_bnd,k,3]; w_i = Q[i,idx_bnd,k,4]; p_i = Q[i,idx_bnd,k,5]
                u_n_i = v_i; u_n_inf = v_inf
            else
                ρ_i = Q[i,j,idx_bnd,1]; u_i = Q[i,j,idx_bnd,2]; v_i = Q[i,j,idx_bnd,3]; w_i = Q[i,j,idx_bnd,4]; p_i = Q[i,j,idx_bnd,5]
                u_n_i = w_i; u_n_inf = w_inf
            end
            outflow_sign = side == Int32(1) ? one(FT) : -one(FT)
            if u_n_i * outflow_sign >= zero(FT)  # outflow
                ρg=ρ_i; ug=u_i; vg=v_i; wg=w_i; pg=p_i
            else  # inflow
                ρg=ρ_inf; ug=u_inf; vg=v_inf; wg=w_inf; pg=p_inf
            end
            Tg = pg/(ρg*Rg)
            Q[i,j,k,1]=ρg; Q[i,j,k,2]=ug; Q[i,j,k,3]=vg; Q[i,j,k,4]=wg; Q[i,j,k,5]=pg; Q[i,j,k,6]=Tg
            U[i,j,k,1]=ρg; U[i,j,k,2]=ρg*ug; U[i,j,k,3]=ρg*vg; U[i,j,k,4]=ρg*wg
            U[i,j,k,5]=pg/(γ-one(FT))+FT(0.5)*ρg*(ug^2+vg^2+wg^2)
        end

    # ── AC No-Slip Wall — velocity reflection, pressure Neumann ──
    # For AC: Q = [p, u, v, w], U = Q (identity)
    # Ghost: p_ghost = p_mirror (Neumann), vel_ghost = -vel_mirror (no-slip)
    elseif bc_type == Int32(BC_AC_WALL)
        if dir == Int32(1)
            @inbounds begin
                Q[i,j,k,1] = Q[idx_int,j,k,1]   # pressure: Neumann
                Q[i,j,k,2] = -Q[idx_int,j,k,2]   # u: no-slip
                Q[i,j,k,3] = -Q[idx_int,j,k,3]   # v: no-slip
                Q[i,j,k,4] = -Q[idx_int,j,k,4]   # w: no-slip
                for n = 1:Ncons; U[i,j,k,n] = Q[i,j,k,n]; end
            end
        elseif dir == Int32(2)
            @inbounds begin
                Q[i,j,k,1] = Q[i,idx_int,k,1]
                Q[i,j,k,2] = -Q[i,idx_int,k,2]
                Q[i,j,k,3] = -Q[i,idx_int,k,3]
                Q[i,j,k,4] = -Q[i,idx_int,k,4]
                for n = 1:Ncons; U[i,j,k,n] = Q[i,j,k,n]; end
            end
        else
            @inbounds begin
                Q[i,j,k,1] = Q[i,j,idx_int,1]
                Q[i,j,k,2] = -Q[i,j,idx_int,2]
                Q[i,j,k,3] = -Q[i,j,idx_int,3]
                Q[i,j,k,4] = -Q[i,j,idx_int,4]
                for n = 1:Ncons; U[i,j,k,n] = Q[i,j,k,n]; end
            end
        end

    # ── AC Lid Wall — prescribed velocity, pressure Neumann ──
    # Ghost: p_ghost = p_mirror, vel_ghost = 2*vel_lid - vel_mirror
    elseif bc_type == Int32(BC_AC_LID)
        u_lid = bcp[BCP_AC_U_LID]; v_lid = bcp[BCP_AC_V_LID]; w_lid = bcp[BCP_AC_W_LID]
        if dir == Int32(1)
            @inbounds begin
                Q[i,j,k,1] = Q[idx_int,j,k,1]
                Q[i,j,k,2] = FT(2.0)*u_lid - Q[idx_int,j,k,2]
                Q[i,j,k,3] = FT(2.0)*v_lid - Q[idx_int,j,k,3]
                Q[i,j,k,4] = FT(2.0)*w_lid - Q[idx_int,j,k,4]
                for n = 1:Ncons; U[i,j,k,n] = Q[i,j,k,n]; end
            end
        elseif dir == Int32(2)
            @inbounds begin
                Q[i,j,k,1] = Q[i,idx_int,k,1]
                Q[i,j,k,2] = FT(2.0)*u_lid - Q[i,idx_int,k,2]
                Q[i,j,k,3] = FT(2.0)*v_lid - Q[i,idx_int,k,3]
                Q[i,j,k,4] = FT(2.0)*w_lid - Q[i,idx_int,k,4]
                for n = 1:Ncons; U[i,j,k,n] = Q[i,j,k,n]; end
            end
        else
            @inbounds begin
                Q[i,j,k,1] = Q[i,j,idx_int,1]
                Q[i,j,k,2] = FT(2.0)*u_lid - Q[i,j,idx_int,2]
                Q[i,j,k,3] = FT(2.0)*v_lid - Q[i,j,idx_int,3]
                Q[i,j,k,4] = FT(2.0)*w_lid - Q[i,j,idx_int,4]
                for n = 1:Ncons; U[i,j,k,n] = Q[i,j,k,n]; end
            end
        end

    end

    # ── MHD extension: set B-field and ψ for ghost cells ──
    # Applied AFTER the hydrodynamic BC above has set ρ, u, v, w, p, T
    if equation_type == :MHD
        if bc_type == Int32(BC_ISOTHERMAL_WALL) || bc_type == Int32(BC_ADIABATIC_WALL) || bc_type == Int32(BC_MHD_WALL) || bc_type == Int32(BC_SLIP_WALL) || bc_type == Int32(BC_SYMMETRY)
            # Perfectly conducting wall: copy B from mirror, ψ anti-symmetric
            if dir == Int32(1)
                @inbounds begin
                    Q[i,j,k,7]=Q[idx_int,j,k,7]; Q[i,j,k,8]=Q[idx_int,j,k,8]; Q[i,j,k,9]=Q[idx_int,j,k,9]
                    Q[i,j,k,10]=-Q[idx_int,j,k,10]
                end
            elseif dir == Int32(2)
                @inbounds begin
                    Q[i,j,k,7]=Q[i,idx_int,k,7]; Q[i,j,k,8]=Q[i,idx_int,k,8]; Q[i,j,k,9]=Q[i,idx_int,k,9]
                    Q[i,j,k,10]=-Q[i,idx_int,k,10]
                end
            else
                @inbounds begin
                    Q[i,j,k,7]=Q[i,j,idx_int,7]; Q[i,j,k,8]=Q[i,j,idx_int,8]; Q[i,j,k,9]=Q[i,j,idx_int,9]
                    Q[i,j,k,10]=-Q[i,j,idx_int,10]
                end
            end
            # Update conservative B and ψ, recompute energy with B²/2
            @inbounds begin
                Bx=Q[i,j,k,7]; By=Q[i,j,k,8]; Bz=Q[i,j,k,9]; ψv=Q[i,j,k,10]
                U[i,j,k,6]=Bx; U[i,j,k,7]=By; U[i,j,k,8]=Bz; U[i,j,k,9]=ψv
                B2=Bx*Bx+By*By+Bz*Bz
                U[i,j,k,5]=Q[i,j,k,5]/(γ-one(FT))+FT(0.5)*Q[i,j,k,1]*(Q[i,j,k,2]^2+Q[i,j,k,3]^2+Q[i,j,k,4]^2)+FT(0.5)*B2
            end
        elseif bc_type == Int32(BC_SUPERSONIC_INFLOW) || bc_type == Int32(BC_MHD_INFLOW)
            # Fixed B-field from BC parameters
            @inbounds begin
                Bx_inf=bcp[BCP_BX_INF]; By_inf=bcp[BCP_BY_INF]; Bz_inf=bcp[BCP_BZ_INF]
                Q[i,j,k,7]=Bx_inf; Q[i,j,k,8]=By_inf; Q[i,j,k,9]=Bz_inf; Q[i,j,k,10]=zero(FT)
                U[i,j,k,6]=Bx_inf; U[i,j,k,7]=By_inf; U[i,j,k,8]=Bz_inf; U[i,j,k,9]=zero(FT)
                B2=Bx_inf^2+By_inf^2+Bz_inf^2
                U[i,j,k,5]=Q[i,j,k,5]/(γ-one(FT))+FT(0.5)*Q[i,j,k,1]*(Q[i,j,k,2]^2+Q[i,j,k,3]^2+Q[i,j,k,4]^2)+FT(0.5)*B2
            end
        end
        # zero_gradient / supersonic_outflow / slip_wall / symmetry:
        # Already handled by generic `for n=1:Nprim` / `for n=1:Ncons` loops
    end

    return
end

# =============================================================================
# ξ-direction BCs — GPU kernel
# =============================================================================
function fill_x(Q, U, rx, n_x, n_y, n_z, x, y, z,
                bc_x_lo, bc_x_hi, bcp_x_lo, bcp_x_hi,
                nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG || i < 1 || j < 1 || k < 1; return; end

    # ξ- ghost cells
    if rx == 0 && i <= NG
        _apply_bc!(Q, U, i, j, k, bc_x_lo, Int32(1), Int32(0),
                   n_x, n_y, n_z, x, y, z, nxp, nyp, nzp, bcp_x_lo, tt, dsrfg_params)
    # ξ+ ghost cells
    elseif rx == Nprocs_block[1]-1 && i > nxp+NG
        _apply_bc!(Q, U, i, j, k, bc_x_hi, Int32(1), Int32(1),
                   n_x, n_y, n_z, x, y, z, nxp, nyp, nzp, bcp_x_hi, tt, dsrfg_params)
    end
    return
end

# =============================================================================
# η-direction BCs — GPU kernel
# =============================================================================
function fill_y(Q, U, rx, ry, rz, n_x_j, n_y_j, n_z_j, x, y, z,
                bc_y_lo, bc_y_hi, bcp_y_lo, bcp_y_hi,
                nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG || i < 1 || j < 1 || k < 1; return; end

    # η- ghost
    if ry == 0 && j <= NG
        _apply_bc!(Q, U, i, j, k, bc_y_lo, Int32(2), Int32(0),
                   n_x_j, n_y_j, n_z_j, x, y, z, nxp, nyp, nzp, bcp_y_lo, tt, dsrfg_params)
    # η+ ghost
    elseif ry == Nprocs_block[2]-1 && j > nyp+NG
        _apply_bc!(Q, U, i, j, k, bc_y_hi, Int32(2), Int32(1),
                   n_x_j, n_y_j, n_z_j, x, y, z, nxp, nyp, nzp, bcp_y_hi, tt, dsrfg_params)
    end
    return
end

# =============================================================================
# ζ-direction BCs — GPU kernel
# =============================================================================
function fill_z(Q, U, rx, ry, rz, n_x_k, n_y_k, n_z_k, x, y, z,
                bc_z_lo, bc_z_hi, bcp_z_lo, bcp_z_hi,
                nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG || i < 1 || j < 1 || k < 1; return; end

    # ζ- ghost
    if rz == 0 && k <= NG
        _apply_bc!(Q, U, i, j, k, bc_z_lo, Int32(3), Int32(0),
                   n_x_k, n_y_k, n_z_k, x, y, z, nxp, nyp, nzp, bcp_z_lo, tt, dsrfg_params)
    # ζ+ ghost
    elseif rz == Nprocs_block[3]-1 && k > nzp+NG
        _apply_bc!(Q, U, i, j, k, bc_z_hi, Int32(3), Int32(1),
                   n_x_k, n_y_k, n_z_k, x, y, z, nxp, nyp, nzp, bcp_z_hi, tt, dsrfg_params)
    end
    return
end

# =============================================================================
# fillGhost — top-level BC application
# =============================================================================
function fillGhost(Q, U, rx, ry, rz,
                   n_x_i, n_y_i, n_z_i,
                   n_x_j, n_y_j, n_z_j,
                   n_x_k, n_y_k, n_z_k,
                   x, y, z, block_id, nxp, nyp, nzp, Nprocs_block, tt, face_bc, bc_params,
                   dsrfg_params)
    nb = (cld(nxp+2*NG, nthreads[1]), cld(nyp+2*NG, nthreads[2]), cld(nzp+2*NG, nthreads[3]))

    # Look up BC type for each face
    bc_x_lo = haskey(face_bc, (block_id, 1)) ? Int32(face_bc[(block_id, 1)]) : Int32(0)
    bc_x_hi = haskey(face_bc, (block_id, 2)) ? Int32(face_bc[(block_id, 2)]) : Int32(0)
    bc_y_lo = haskey(face_bc, (block_id, 3)) ? Int32(face_bc[(block_id, 3)]) : Int32(0)
    bc_y_hi = haskey(face_bc, (block_id, 4)) ? Int32(face_bc[(block_id, 4)]) : Int32(0)
    bc_z_lo = haskey(face_bc, (block_id, 5)) ? Int32(face_bc[(block_id, 5)]) : Int32(0)
    bc_z_hi = haskey(face_bc, (block_id, 6)) ? Int32(face_bc[(block_id, 6)]) : Int32(0)

    # Look up BC params for each face (NTuple of N_BC_PARAMS Float32s)
    _get_bcp(fid) = haskey(bc_params, (block_id, fid)) ? bc_params[(block_id, fid)] : ntuple(i->zero(FT), Val(N_BC_PARAMS))
    bcp_x_lo = _get_bcp(1); bcp_x_hi = _get_bcp(2)
    bcp_y_lo = _get_bcp(3); bcp_y_hi = _get_bcp(4)
    bcp_z_lo = _get_bcp(5); bcp_z_hi = _get_bcp(6)

    @gpu_launch threads=nthreads blocks=nb fill_x(Q, U, rx, n_x_i, n_y_i, n_z_i,
        x, y, z, bc_x_lo, bc_x_hi, bcp_x_lo, bcp_x_hi, nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params)
    @gpu_launch threads=nthreads blocks=nb fill_y(Q, U, rx, ry, rz, n_x_j, n_y_j, n_z_j,
        x, y, z, bc_y_lo, bc_y_hi, bcp_y_lo, bcp_y_hi, nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params)
    @gpu_launch threads=nthreads blocks=nb fill_z(Q, U, rx, ry, rz, n_x_k, n_y_k, n_z_k,
        x, y, z, bc_z_lo, bc_z_hi, bcp_z_lo, bcp_z_hi, nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params)
end



include("init_flow.jl")
