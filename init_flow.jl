function init_tgv(Q, x, y, z)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG; return; end
    if i < NG+1 || i > Nxp+NG || j < NG+1 || j > Nyp+NG || k < NG+1 || k > Nzp+NG; return; end

    yc = FT(0.5) * (y[i, j, k] + y[i, j+1, k])
    zc = FT(0.5) * (z[i, j, k] + z[i, j, k+1])
    
    xc = FT(0.5) * (x[i, j, k] + x[i+1, j, k])
    V0 = one(FT); p0 = FT(10.0); rho0 = one(FT)
    
    u = FT(0.1); v = FT(0.1); w = zero(FT)
    r2 = (xc - FT(1.5))^2 + (yc - FT(1.5))^2 + (zc - FT(1.5))^2
    pulse = FT(0.1) * exp(-FT(10.0) * r2)
    p = p0 + pulse
    rho = rho0 + pulse
    @inbounds Q[i, j, k, 1] = rho; Q[i, j, k, 2] = u; Q[i, j, k, 3] = v; Q[i, j, k, 4] = w; Q[i, j, k, 5] = p; Q[i, j, k, 6] = p/(rho*Rg)
    return
end

function init_hit(Q, x, y, z)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG; return; end
    if i < NG+1 || i > Nxp+NG || j < NG+1 || j > Nyp+NG || k < NG+1 || k > Nzp+NG; return; end

    # Cell-center coordinates
    xc = FT(0.5) * (x[i, j, k] + x[i+1, j, k])
    yc = FT(0.5) * (y[i, j, k] + y[i, j+1, k])
    zc = FT(0.5) * (z[i, j, k] + z[i, j, k+1])
    
    # Multi-mode initial velocity field (broadband TGV for faster turbulence transition)
    V0 = one(FT); p0 = FT(100.0); rho0 = one(FT)
    
    u = V0 * ( sin(xc)*cos(yc)*cos(zc) + FT(0.5)*sin(FT(2.0)*xc)*cos(FT(2.0)*zc) + FT(0.25)*sin(FT(3.0)*yc)*cos(FT(3.0)*xc))
    v = V0 * (-cos(xc)*sin(yc)*cos(zc) + FT(0.5)*sin(FT(2.0)*yc)*cos(FT(2.0)*xc) + FT(0.25)*sin(FT(3.0)*zc)*cos(FT(3.0)*yc))
    w = V0 * ( FT(0.5)*cos(FT(2.0)*yc)*sin(FT(2.0)*zc) + FT(0.25)*cos(FT(3.0)*xc)*sin(FT(3.0)*zc))
    
    p = p0 + rho0 / FT(16.0e0) * (cos(FT(2.0)*xc) + cos(FT(2.0)*yc)) * (cos(FT(2.0)*zc) + FT(2.0))
    rho = rho0
    
    @inbounds Q[i, j, k, 1] = rho
    @inbounds Q[i, j, k, 2] = u
    @inbounds Q[i, j, k, 3] = v
    @inbounds Q[i, j, k, 4] = w
    @inbounds Q[i, j, k, 5] = p
    @inbounds Q[i, j, k, 6] = p / (rho * Rg)
    return
end

function init_oblique_shock(Q, x, y, z)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG; return; end
    if i < NG+1 || i > Nxp+NG || j < NG+1 || j > Nyp+NG || k < NG+1 || k > Nzp+NG; return; end

    u1 = FT(2.36643e0)
    @inbounds Q[i, j, k, 1] = one(FT); Q[i, j, k, 2] = u1; Q[i, j, k, 3] = zero(FT); Q[i, j, k, 4] = zero(FT); Q[i, j, k, 5] = one(FT); Q[i, j, k, 6] = one(FT)/(one(FT)*Rg)
    return
end

function init_uniform_dimensional(Q, x, y, z)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG; return; end
    if i < NG+1 || i > Nxp+NG || j < NG+1 || j > Nyp+NG || k < NG+1 || k > Nzp+NG; return; end

    rho0 = FT(1.2)
    u0 = FT(100.0)
    T0 = FT(300.0)
    p0 = rho0 * Rg * T0

    @inbounds begin
        Q[i, j, k, 1] = rho0
        Q[i, j, k, 2] = u0
        Q[i, j, k, 3] = zero(FT)
        Q[i, j, k, 4] = zero(FT)
        Q[i, j, k, 5] = p0
        Q[i, j, k, 6] = T0
    end
    return
end

function init_sod(Q, x, y, z)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG; return; end
    if i < NG+1 || i > Nxp+NG || j < NG+1 || j > Nyp+NG || k < NG+1 || k > Nzp+NG; return; end

    xc = x[i, j, k]
    if xc < FT(0.5)
        rho = one(FT); u = zero(FT); p = one(FT)
    else
        rho = FT(0.125e0); u = zero(FT); p = FT(0.1)
    end
    @inbounds Q[i, j, k, 1] = rho; Q[i, j, k, 2] = u; Q[i, j, k, 3] = zero(FT); Q[i, j, k, 4] = zero(FT); Q[i, j, k, 5] = p; Q[i, j, k, 6] = p/(rho*Rg)
    return
end

function init_pipe_flow(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32,
                        eddy_pos_x, eddy_pos_y, eddy_pos_z,
                        eddy_sign1, eddy_sign2, eddy_sign3,
                        n_sem::Int32, l_sem::FT, amp_sem::FT)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+Int32(2*NG) || j > nyp+Int32(2*NG) || k > nzp+Int32(2*NG); return; end
    if i < Int32(NG+1) || i > nxp+Int32(NG) || j < Int32(NG+1) || j > nyp+Int32(NG) || k < Int32(NG+1) || k > nzp+Int32(NG); return; end

    # Physics-based targets
    c_wall = sqrt(γ*Rg*Tw)
    u_bulk = Ma_target * c_wall
    
    # Hagen-Poiseuille parabolic profile
    @inbounds xi = x[i,j,k]
    @inbounds yi = y[i,j,k]
    @inbounds zi = z[i,j,k]
    r2 = yi*yi + zi*zi
    r2_norm = r2 / (R0*R0)
    u_hp = FT(2.0) * u_bulk * max(one(FT) - r2_norm, zero(FT))
    
    # Temperature with frictional heating profile
    Ma2 = Ma_target * Ma_target
    β = FT(0.5) * Pr * (γ - one(FT)) * Ma2
    T_local = Tw * (one(FT) + β * max(one(FT) - r2_norm*r2_norm, zero(FT)))
    
    # Reference metrics (u_bulk matches Re_target)
    μw = C_s * Tw * sqrt(Tw) / (Tw + T_s)
    ρ_ref = Re_target * μw / (u_bulk * FT(2.0) * R0)
    
    # Analytical correction for volume-averaged density due to viscous heating:
    # ∫ 1/(1 + β(1-u²)) du = (1 / 2A√β) * ln((A+√β)/(A-√β)) where A = √(1+β)
    if β > 1e-6
        A = sqrt(one(FT) + β)
        sqrt_β = sqrt(β)
        integral_factor = (one(FT) / (FT(2.0) * A * sqrt_β)) * log((A + sqrt_β) / (A - sqrt_β))
        p_ref = ρ_ref * Rg * Tw / integral_factor
    else
        p_ref = ρ_ref * Rg * Tw
    end
    
    ρ = p_ref / (Rg * T_local)
    p = p_ref

    # ── SEM (Synthetic Eddy Method) perturbation ──────────────────────
    # Spatially correlated fluctuations via superposition of tent-function eddies.
    # Each eddy contributes within a cube of side 2*l_sem centered at its position.
    # Ref: Jarrin et al. (2006), adapted from Incompact3d's sem_init_channel.
    
    # Volume of the SEM domain (pipe length × diameter × diameter)
    vol_sem = Lx * (FT(2.0) * R0) * (FT(2.0) * R0)
    
    upr = zero(FT)
    vpr = zero(FT)
    wpr = zero(FT)
    
    for jj = Int32(1):n_sem
        @inbounds dx_e = abs(xi - eddy_pos_x[jj])
        @inbounds dy_e = abs(yi - eddy_pos_y[jj])
        @inbounds dz_e = abs(zi - eddy_pos_z[jj])
        
        # Periodic in x: check also wrapped distance
        dx_e = min(dx_e, Lx - dx_e)
        
        if dx_e < l_sem && dy_e < l_sem && dz_e < l_sem
            # Tent function: f = (1-|dx|/l)(1-|dy|/l)(1-|dz|/l) / (sqrt(2l/3))^3
            ftent = (one(FT) - dx_e/l_sem) * (one(FT) - dy_e/l_sem) * (one(FT) - dz_e/l_sem)
            ftent = ftent / (sqrt(FT(2.0)/FT(3.0) * l_sem))^3
            
            @inbounds upr += eddy_sign1[jj] * ftent
            @inbounds vpr += eddy_sign2[jj] * ftent
            @inbounds wpr += eddy_sign3[jj] * ftent
        end
    end
    
    # Scale by sqrt(V_domain / N_sem)
    scale = sqrt(vol_sem / FT(n_sem))
    upr *= scale
    vpr *= scale
    wpr *= scale
    
    # Radial envelope: (1 - r²/R0²) — perturbation vanishes at wall
    # Also scale by local mean velocity for realistic turbulence intensity
    envelope = max(one(FT) - r2_norm, zero(FT))
    turb_scale = amp_sem * sqrt(FT(2.0)/FT(3.0) * envelope)
    
    u_final = u_hp  + upr * turb_scale
    v_final = vpr * turb_scale
    w_final = wpr * turb_scale
    
    @inbounds begin
        Q[i, j, k, 1] = ρ
        Q[i, j, k, 2] = u_final
        Q[i, j, k, 3] = v_final
        Q[i, j, k, 4] = w_final
        Q[i, j, k, 5] = p
        Q[i, j, k, 6] = T_local
    end
    return
end

function init_flatplate(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32,
                       T_inf_val::FT, p_inf_val::FT)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+Int32(2*NG) || j > nyp+Int32(2*NG) || k > nzp+Int32(2*NG); return; end
    if i < Int32(NG+1) || i > nxp+Int32(NG) || j < Int32(NG+1) || j > nyp+Int32(NG) || k < Int32(NG+1) || k > nzp+Int32(NG); return; end

    # Freestream conditions
    c_inf = sqrt(γ * Rg * T_inf_val)
    u_inf = Ma_target * c_inf
    ρ_inf = p_inf_val / (Rg * T_inf_val)

    # Cell-center y-coordinate (wall distance)
    @inbounds yc = FT(0.5) * (y[i, j, k] + y[i, j+1, k])
    @inbounds xc = FT(0.5) * (x[i, j, k] + x[i+1, j, k])

    # ── Compressible boundary layer profile ──
    # Local BL thickness from flat plate correlation with compressibility correction
    μ_inf_val = C_s * T_inf_val * sqrt(T_inf_val) / (T_inf_val + T_s)
    ν_inf = μ_inf_val / ρ_inf
    δ_star = FT(1000.0) * μ_inf_val / (ρ_inf * u_inf)  # inlet δ*

    # BL thickness grows with sqrt(x), with virtual origin offset
    x_inlet = FT(10.0) * δ_star
    x_local = max(xc + x_inlet, FT(1.0e-10))
    Re_x_local = u_inf * x_local / ν_inf
    # Compressibility correction: δ ~ sqrt(T_ref/T_inf) for high Ma
    T_ref = FT(0.5) * (T_inf_val + Tw) + FT(0.22e0) * FT(0.5) * (γ - one(FT)) * Ma_target^2 * T_inf_val
    comp_factor = sqrt(T_ref / T_inf_val)
    δ_bl = FT(5.0e0) * x_local / (sqrt(Re_x_local) + FT(1.0e-10)) * comp_factor
    δ_bl = max(δ_bl, FT(2.0) * δ_star)

    # Velocity: smooth tanh profile
    η = yc / δ_bl
    u_profile = u_inf * tanh(FT(2.0) * η)

    # Temperature: Crocco-Busemann relation
    u_ratio = u_profile / u_inf
    r_factor = Pr^(one(FT)/FT(3.0))
    T_aw = T_inf_val * (one(FT) + r_factor * FT(0.5) * (γ - one(FT)) * Ma_target^2)
    T_profile = Tw + (T_inf_val - Tw) * u_ratio +
                (T_aw - T_inf_val) * u_ratio * (one(FT) - u_ratio)
    T_profile = max(T_profile, FT(100.0))

    # Density from EOS (constant pressure across BL)
    ρ_profile = p_inf_val / (Rg * T_profile)

    @inbounds begin
        Q[i, j, k, 1] = ρ_profile
        Q[i, j, k, 2] = u_profile
        Q[i, j, k, 3] = zero(FT)
        Q[i, j, k, 4] = zero(FT)
        Q[i, j, k, 5] = p_inf_val
        Q[i, j, k, 6] = T_profile
    end
    return
end

function initialize(Q, x, y, z, rankx, ranky, Nprocs, nxp, nyp, nzp)
    nb = (cld(nxp+2*NG, nthreads[1]), cld(nyp+2*NG, nthreads[2]), cld(nzp+2*NG, nthreads[3]))
    if test_case == "TGV"
        @gpu_launch threads=nthreads blocks=nb init_tgv(Q, x, y, z)
    elseif test_case == "HIT"
        @gpu_launch threads=nthreads blocks=nb init_hit(Q, x, y, z)
    elseif test_case == "ObliqueShock"
        @gpu_launch threads=nthreads blocks=nb init_oblique_shock(Q, x, y, z)
    elseif test_case == "Sod"
        @gpu_launch threads=nthreads blocks=nb init_sod(Q, x, y, z)
    elseif test_case == "FlatPlate"
        # Read freestream T∞ and p∞ from bc_params (supersonic inflow face, block 0)
        fp_bc_params = h5read("MESH/block_connectivity.h5", "bc_params")
        fp_T_inf = FT(fp_bc_params[1, 1, BCP_P_INF] / (Rg * fp_bc_params[1, 1, BCP_RHO_INF]))
        fp_p_inf = FT(fp_bc_params[1, 1, BCP_P_INF])
        @gpu_launch threads=nthreads blocks=nb init_flatplate(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp),
                                                               fp_T_inf, fp_p_inf)
    elseif test_case == "PipeFlow"
        # ── SEM: generate synthetic eddies on CPU, transfer to GPU ──
        N_sem = 1000
        l_sem = FT(0.15 * R0)   # eddy size ~ 15% of pipe radius
        amp_sem = FT(0.10 * Ma_target * sqrt(γ * Rg * Tw))  # 10% of u_bulk

        # Random eddy positions within the pipe domain
        pos_x_h = FT.(rand(N_sem) .* Lx)                     # x ∈ [0, Lx]
        pos_y_h = FT.((rand(N_sem) .- FT(0.5)) .* FT(2.0) .* R0) # y ∈ [-R0, R0]
        pos_z_h = FT.((rand(N_sem) .- FT(0.5)) .* FT(2.0) .* R0) # z ∈ [-R0, R0]
        
        # Random eddy signs: ±1 for each velocity component
        sign1_h = FT.((rand(N_sem) .> 0.5) .* 2.0 .- 1.0)
        sign2_h = FT.((rand(N_sem) .> 0.5) .* 2.0 .- 1.0)
        sign3_h = FT.((rand(N_sem) .> 0.5) .* 2.0 .- 1.0)
        
        # Transfer to GPU using project's GPUArray abstraction
        pos_x_d = GPUArray(pos_x_h)
        pos_y_d = GPUArray(pos_y_h)
        pos_z_d = GPUArray(pos_z_h)
        sign1_d = GPUArray(sign1_h)
        sign2_d = GPUArray(sign2_h)
        sign3_d = GPUArray(sign3_h)
        
        @gpu_launch threads=nthreads blocks=nb init_pipe_flow(
            Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp),
            pos_x_d, pos_y_d, pos_z_d,
            sign1_d, sign2_d, sign3_d,
            Int32(N_sem), l_sem, amp_sem
        )
    elseif test_case == "BrioWu"
        @gpu_launch threads=nthreads blocks=nb init_brio_wu(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp))
    elseif test_case == "OrszagTang"
        @gpu_launch threads=nthreads blocks=nb init_orszag_tang(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp))
    elseif test_case == "TG5_debug"
        @gpu_launch threads=nthreads blocks=nb init_tg5_debug(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp))
    elseif test_case == "UniformDimensional"
        @gpu_launch threads=nthreads blocks=nb init_uniform_dimensional(Q, x, y, z)
    end
end

function init_tg5_debug(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+Int32(2*NG) || j > nyp+Int32(2*NG) || k > nzp+Int32(2*NG); return; end
    if i < NG+1 || i > nxp+NG || j < NG+1 || j > nyp+NG || k < NG+1 || k > nzp+NG; return; end

    yc = FT(0.5) * (y[i, j, k] + y[i, j+1, k])
    zc = FT(0.5) * (z[i, j, k] + z[i, j, k+1])
    xc = FT(0.5) * (x[i, j, k] + x[i+1, j, k])
    
    # 3D Gaussian Acoustic Pulse — centered at (2.5, 0, 0) with streamwise drift
    p0 = FT(10.0); rho0 = one(FT)
    u = FT(0.5); v = zero(FT); w = zero(FT)
    r2 = (xc - FT(2.5))^2 + yc^2 + zc^2
    pulse = FT(0.1) * exp(-FT(30.0) * r2)
    p = p0 + pulse
    rho = rho0 + pulse
    
    @inbounds Q[i, j, k, 1] = rho; Q[i, j, k, 2] = u; Q[i, j, k, 3] = v; Q[i, j, k, 4] = w; Q[i, j, k, 5] = p; Q[i, j, k, 6] = p/(rho*Rg)
    return
end


# ═══════════════════════════════════════════════════════════════════════
# Brio-Wu MHD Shock Tube (Brio & Wu, 1988)
# ═══════════════════════════════════════════════════════════════════════
# Standard 1D MHD Riemann problem for validation.
# Domain: x ∈ [0, 1], discontinuity at x = 0.5
# Left:  ρ=1.0,   p=1.0,   u=v=w=0, Bx=0.75, By=1.0,  Bz=0, ψ=0
# Right: ρ=0.125, p=0.1,   u=v=w=0, Bx=0.75, By=-1.0, Bz=0, ψ=0
# γ = 2.0 (must be set in run script!)
# Q = [ρ, u, v, w, p, T, Bx, By, Bz, ψ]
function init_brio_wu(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+Int32(2*NG) || j > nyp+Int32(2*NG) || k > nzp+Int32(2*NG); return; end
    if i < Int32(NG+1) || i > nxp+Int32(NG) || j < Int32(NG+1) || j > nyp+Int32(NG) || k < Int32(NG+1) || k > nzp+Int32(NG); return; end

    @inbounds xc = x[i, j, k]

    if xc < FT(0.5)
        # Left state
        rho = one(FT); u = zero(FT); v = zero(FT); w = zero(FT); p = one(FT)
        Bx = FT(0.75); By = one(FT); Bz = zero(FT)
    else
        # Right state
        rho = FT(0.125e0); u = zero(FT); v = zero(FT); w = zero(FT); p = FT(0.1)
        Bx = FT(0.75); By = -one(FT); Bz = zero(FT)
    end

    T = p / (rho * Rg)

    @inbounds begin
        Q[i,j,k,1] = rho; Q[i,j,k,2] = u; Q[i,j,k,3] = v; Q[i,j,k,4] = w
        Q[i,j,k,5] = p;   Q[i,j,k,6] = T
        Q[i,j,k,7] = Bx;  Q[i,j,k,8] = By; Q[i,j,k,9] = Bz
        Q[i,j,k,10] = zero(FT)  # ψ
    end
    return
end

# ═══════════════════════════════════════════════════════════════════════
# Orszag-Tang Vortex (Orszag & Tang, 1979)
# ═══════════════════════════════════════════════════════════════════════
# 2D MHD vortex problem for testing MHD turbulence and shock interactions.
# Domain: [0, 2π] × [0, 2π], periodic BCs
# ρ = γ² (= 25/9 for γ=5/3), p = γ (= 5/3)
# u = -sin(y), v = sin(x), w = 0
# Bx = -sin(y), By = sin(2x), Bz = 0, ψ = 0
# γ = 5/3
function init_orszag_tang(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+Int32(2*NG) || j > nyp+Int32(2*NG) || k > nzp+Int32(2*NG); return; end
    if i < Int32(NG+1) || i > nxp+Int32(NG) || j < Int32(NG+1) || j > nyp+Int32(NG) || k < Int32(NG+1) || k > nzp+Int32(NG); return; end

    @inbounds xc = x[i, j, k]
    @inbounds yc = y[i, j, k]

    rho = γ * γ
    p = γ
    u = -sin(yc)
    v = sin(xc)
    w = zero(FT)
    Bx = -sin(yc)
    By = sin(FT(2.0) * xc)
    Bz = zero(FT)
    T = p / (rho * Rg)

    @inbounds begin
        Q[i,j,k,1] = rho; Q[i,j,k,2] = u; Q[i,j,k,3] = v; Q[i,j,k,4] = w
        Q[i,j,k,5] = p;   Q[i,j,k,6] = T
        Q[i,j,k,7] = Bx;  Q[i,j,k,8] = By; Q[i,j,k,9] = Bz
        Q[i,j,k,10] = zero(FT)  # ψ
    end
    return
end
