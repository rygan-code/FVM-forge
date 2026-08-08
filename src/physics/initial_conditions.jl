if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end
if !isdefined(@__MODULE__, :structured_cell_center_coordinates)
    include(joinpath(@__DIR__, "..", "mesh", "structured_coordinates.jl"))
end

if !@isdefined(magnetic_nozzle_rho0)
    const magnetic_nozzle_rho0 = FT(5.0e-5)
end
if !@isdefined(magnetic_nozzle_T0)
    const magnetic_nozzle_T0 = FT(1.160451812e6)
end


function init_magnetic_nozzle(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG; return; end
    if i < NG+1 || i > nxp+NG || j < NG+1 || j > nyp+NG || k < NG+1 || k > nzp+NG; return; end

    rho = magnetic_nozzle_rho0
    T = magnetic_nozzle_T0
    p = rho * Rg * T
    @inbounds begin
        Q[i,j,k,1]=rho; Q[i,j,k,2]=zero(FT); Q[i,j,k,3]=zero(FT)
        Q[i,j,k,4]=zero(FT); Q[i,j,k,5]=p; Q[i,j,k,6]=T
        if equation_type == :MHD
            Q[i,j,k,7]=zero(FT); Q[i,j,k,8]=zero(FT); Q[i,j,k,9]=zero(FT); Q[i,j,k,10]=zero(FT)
        end
    end
    return
end

function init_sheth_magnetic_nozzle(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG; return; end
    if i < NG+1 || i > nxp+NG || j < NG+1 || j > nyp+NG || k < NG+1 || k > nzp+NG; return; end

    # Sheth et al. start from a stationary, uniform plasma; the inlet
    # sech^2 density/velocity profile is imposed only by the x-lo BC.
    rho = magnetic_nozzle_rho0
    T = magnetic_nozzle_T0
    p = rho * Rg * T
    @inbounds begin
        Q[i,j,k,1]=rho; Q[i,j,k,2]=zero(FT); Q[i,j,k,3]=zero(FT)
        Q[i,j,k,4]=zero(FT); Q[i,j,k,5]=p; Q[i,j,k,6]=T
        if equation_type == :MHD
            Q[i,j,k,7]=zero(FT); Q[i,j,k,8]=zero(FT); Q[i,j,k,9]=zero(FT); Q[i,j,k,10]=zero(FT)
        end
    end
    return
end

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
                        n_sem::Int32, l_sem::FT, amp_sem::FT,
                        is_visc::Bool, L_domain::FT)
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
    
    if is_visc
        u_hp = FT(2.0) * u_bulk * max(one(FT) - r2_norm, zero(FT))
        
        # Temperature with frictional heating profile
        Ma2 = Ma_target * Ma_target
        β = FT(0.5) * Pr * (γ - one(FT)) * Ma2
        T_local = Tw * (one(FT) + β * max(one(FT) - r2_norm*r2_norm, zero(FT)))
        
        # Reference metrics (u_bulk matches Re_target)
        μw = C_s * Tw * sqrt(Tw) / (Tw + T_s)
        ρ_ref = Re_target * μw / (u_bulk * FT(2.0) * R0)
        
        # Analytical correction for volume-averaged density due to viscous heating:
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
        vol_sem = L_domain * (FT(2.0) * R0) * (FT(2.0) * R0)
        upr = zero(FT)
        vpr = zero(FT)
        wpr = zero(FT)
        
        for jj = Int32(1):n_sem
            @inbounds dx_e = abs(xi - eddy_pos_x[jj])
            @inbounds dy_e = abs(yi - eddy_pos_y[jj])
            @inbounds dz_e = abs(zi - eddy_pos_z[jj])
            
            dx_e = min(dx_e, L_domain - dx_e)
            
            if dx_e < l_sem && dy_e < l_sem && dz_e < l_sem
                ftent = (one(FT) - dx_e/l_sem) * (one(FT) - dy_e/l_sem) * (one(FT) - dz_e/l_sem)
                ftent = ftent / (sqrt(FT(2.0)/FT(3.0) * l_sem))^3
                
                @inbounds upr += eddy_sign1[jj] * ftent
                @inbounds vpr += eddy_sign2[jj] * ftent
                @inbounds wpr += eddy_sign3[jj] * ftent
            end
        end
        
        scale = sqrt(vol_sem / FT(n_sem))
        upr *= scale
        vpr *= scale
        wpr *= scale
        
        envelope = max(one(FT) - r2_norm, zero(FT))
        turb_scale = amp_sem * sqrt(FT(2.0)/FT(3.0) * envelope)
        
        u_final = u_hp  + upr * turb_scale
        v_final = vpr * turb_scale
        w_final = wpr * turb_scale
    else
        u_final = u_bulk
        v_final = zero(FT)
        w_final = zero(FT)
        T_local = Tw
        
        # Consistent initial density/pressure for inviscid cases
        μw = C_s * Tw * sqrt(Tw) / (Tw + T_s)
        ρ_ref = Re_target * μw / (u_bulk * FT(2.0) * R0)
        p_ref = ρ_ref * Rg * Tw
        
        ρ = ρ_ref
        p = p_ref
    end
    
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

# ═══════════════════════════════════════════════════════════════════════
# Decaying MHD turbulence — Passot-Pouquet spectrum, divergence-free IC
# ═══════════════════════════════════════════════════════════════════════
# Generates a random solenoidal (divergence-free) velocity and magnetic field
# on the CPU using a Passot-Pouquet energy spectrum
#   E(k) ∝ k^4 · exp(-2 (k/k0)^2)
# peaking at k0, then projects each Fourier mode onto the plane transverse to
# its wavevector k so that ∇·v = 0 and ∇·B = 0 exactly (in spectral space).
# The real-space fields are written into the GPU Q array (including periodic
# ghost-cell fill).
#
# Parameters (read from Main if defined, else defaults):
#   u_rms  — target RMS velocity (default 1.0)
#   B_rms  — target RMS magnetic field in Tesla (default corresponds to
#            the legacy mu0=1 value 0.5 at p0=1 Pa)
#   k0     — peak wavenumber of the spectrum (default 4.0)
#   rho0   — uniform density (default 1.0)
#   p0     — uniform pressure (default 1.0)
#   seed   — random seed for reproducibility (default 42)
#
# This is a HOST function (not a GPU kernel): it generates the full field on
# the CPU, then copies it into the GPU Q array. Requires `using FFTW` in the
# calling run script.
function init_mhd_turbulence_field!(Q, x, y, z, nxp, nyp, nzp)
    FFTW = Main.FFTW

    # ── Read parameters (with defaults) ──
    u_rms = isdefined(Main, :u_rms_init) ? Main.u_rms_init : FT(1.0)
    B_rms = if isdefined(Main, :B_rms_physical)
        Main.B_rms_physical
    elseif isdefined(Main, :B_rms_init)
        Main.B_rms_init
    else
        FT(0.5) * SQRT_MU0_SI
    end
    k0    = isdefined(Main, :k0_init)    ? Main.k0_init    : FT(4.0)
    rho0  = isdefined(Main, :rho0_init)  ? Main.rho0_init  : FT(1.0)
    p0    = isdefined(Main, :p0_init)    ? Main.p0_init    : FT(1.0)
    seed  = isdefined(Main, :seed_init)  ? Main.seed_init  : Int(42)

    # Pull coordinate arrays to host (uniform Cartesian expected)
    x_h = Array(x)
    y_h = Array(y)
    z_h = Array(z)

    # Interior cell centers: index NG+1 .. nxp+NG map to real domain.
    # For a uniform periodic box, cell spacing is constant.
    Lx = x_h[nxp + NG, NG+1, NG+1] - x_h[NG+1, NG+1, NG+1]   # length spanned by interior cells (≈ domain)
    Ly = y_h[NG+1, nyp + NG, NG+1] - y_h[NG+1, NG+1, NG+1]
    Lz = z_h[NG+1, NG+1, nzp + NG] - z_h[NG+1, NG+1, NG+1]
    # dx measured cell-to-cell (node-based coords → cell center spacing = node spacing)
    dx = x_h[NG+2, NG+1, NG+1] - x_h[NG+1, NG+1, NG+1]
    dy = y_h[NG+1, NG+2, NG+1] - y_h[NG+1, NG+1, NG+1]
    dz = z_h[NG+1, NG+1, NG+2] - z_h[NG+1, NG+1, NG+1]

    Random = Main.Random
    Random.seed!(seed)

    # ── Generate one divergence-free vector field of given rms and spectrum ──
    # Returns a real Array of size (nxp, nyp, nzp, 3).
    function _gen_solenoidal_field(nx, ny, nz, rms_target)
        # Fourier-space coefficients with Passot-Pouquet amplitude
        # rfft of a real (nx,ny,nz) array has shape (nx÷2+1, ny, nz) complex
        vh = zeros(ComplexF64, nx÷2+1, ny, nz, 3)
        for kz in 0:nz-1, ky in 0:ny-1, kx in 0:(nx÷2)
            # Physical wavenumber (wrap negative frequencies for ky, kz)
            kxi = kx
            kyi = ky <= ny÷2 ? ky : ky - ny
            kzi = kz <= nz÷2 ? kz : kz - nz
            kx_p = 2π * kxi / Lx
            ky_p = 2π * kyi / Ly
            kz_p = 2π * kzi / Lz
            kmag = sqrt(kx_p^2 + ky_p^2 + kz_p^2)
            if kmag < 1e-12
                continue   # leave DC mode zero (zero mean)
            end
            # Passot-Pouquet: E(k) ∝ k^4 exp(-2(k/k0)^2); amplitude per mode ∝ sqrt(E(k))
            amp = kmag^2 * exp(-(kmag/k0)^2)
            # Random complex coefficient (Hermitian symmetry handled by rfft automatically
            # for the kx dimension; ky/kz full range is fine since we build all modes)
            phase = 2π * rand()
            coeff = amp * (cos(phase) + im * sin(phase)) / sqrt(kmag)
            # Two random transverse unit vectors e1, e2 perpendicular to k
            # Build via Gram-Schmidt against k
            kvec = [kx_p, ky_p, kz_p]
            # Pick a vector not parallel to k
            ref = abs(kvec[1]) < 0.7*norm(kvec) ? [1.0,0.0,0.0] : [0.0,1.0,0.0]
            e1 = ref - (dot(ref, kvec)/dot(kvec,kvec)) * kvec
            e1 = e1 / norm(e1)
            e2 = cross(kvec, e1); e2 = e2 / norm(e2)
            # Split energy equally between the two polarizations
            c1 = coeff / sqrt(2.0)
            c2 = coeff / sqrt(2.0) * exp(2π*im*rand())   # independent phase for e2
            vh[kx+1, ky+1, kz+1, 1] = c1*e1[1] + c2*e2[1]
            vh[kx+1, ky+1, kz+1, 2] = c1*e1[2] + c2*e2[2]
            vh[kx+1, ky+1, kz+1, 3] = c1*e1[3] + c2*e2[3]
        end
        # Inverse rfft → real field (each component)
        field = zeros(Float64, nx, ny, nz, 3)
        for d in 1:3
            field[:,:,:,d] = FFTW.irfft(view(vh,:,:,:,d), nx)
        end
        # Scale to target rms
        # rms of a vector field = sqrt(<|v|^2>) = sqrt(mean(vx^2+vy^2+vz^2))
        v2 = sum(@. field[:,:,:,1]^2 + field[:,:,:,2]^2 + field[:,:,:,3]^2)
        rms_current = sqrt(v2 / (nx*ny*nz))
        if rms_current > 1e-12
            field .*= rms_target / rms_current
        end
        return field
    end

    # ── Generate v and B fields (interior only, size nxp×nyp×nzp) ──
    v_field = _gen_solenoidal_field(nxp, nyp, nzp, u_rms)
    B_field = _gen_solenoidal_field(nxp, nyp, nzp, B_rms)

    # ── Assemble full Q array on host including ghosts (periodic fill) ──
    Q_h = zeros(FT, nxp + 2*NG, nyp + 2*NG, nzp + 2*NG, Nprim)
    T0 = p0 / (rho0 * Rg)
    NGp = NG + 1
    for kk in 1:nzp, jj in 1:nyp, ii in 1:nxp
        Q_h[ii+NG, jj+NG, kk+NG, 1] = rho0
        Q_h[ii+NG, jj+NG, kk+NG, 2] = v_field[ii, jj, kk, 1]
        Q_h[ii+NG, jj+NG, kk+NG, 3] = v_field[ii, jj, kk, 2]
        Q_h[ii+NG, jj+NG, kk+NG, 4] = v_field[ii, jj, kk, 3]
        Q_h[ii+NG, jj+NG, kk+NG, 5] = p0
        Q_h[ii+NG, jj+NG, kk+NG, 6] = T0
        Q_h[ii+NG, jj+NG, kk+NG, 7] = B_field[ii, jj, kk, 1]
        Q_h[ii+NG, jj+NG, kk+NG, 8] = B_field[ii, jj, kk, 2]
        Q_h[ii+NG, jj+NG, kk+NG, 9] = B_field[ii, jj, kk, 3]
        Q_h[ii+NG, jj+NG, kk+NG, 10] = zero(FT)   # ψ
    end
    # Periodic ghost fill: left ghost [1:NG] ← interior right end [nxp+1-NG:nxp];
    # right ghost [nxp+NG+1:end] ← interior left start [1:NG]. Same for y, z.
    # Interior lives at indices [NG+1 : nxp+NG] in each direction.
    int_range_x = (NG+1):(nxp+NG)
    int_range_y = (NG+1):(nyp+NG)
    int_range_z = (NG+1):(nzp+NG)
    # x-direction ghosts
    Q_h[1:NG,           int_range_y, int_range_z, :] = Q_h[(nxp+1):(nxp+NG), int_range_y, int_range_z, :]
    Q_h[nxp+NG+1:end,   int_range_y, int_range_z, :] = Q_h[(NG+1):(NG+NG),   int_range_y, int_range_z, :]
    # y-direction ghosts (x already filled, use full x range now)
    Q_h[:, 1:NG,           int_range_z, :] = Q_h[:, (nyp+1):(nyp+NG), int_range_z, :]
    Q_h[:, nyp+NG+1:end,   int_range_z, :] = Q_h[:, (NG+1):(NG+NG),   int_range_z, :]
    # z-direction ghosts
    Q_h[:, :, 1:NG,           :] = Q_h[:, :, (nzp+1):(nzp+NG), :]
    Q_h[:, :, nzp+NG+1:end,   :] = Q_h[:, :, (NG+1):(NG+NG),   :]

    # ── Copy host field into the GPU Q array ──
    copyto!(Q, Q_h)
    return
end

@inline function _run_initial_ct_face_flux_process!(
    host_module::Module, blocks, world_rank, Block_Nprocs, block_comms,
)
    if isdefined(host_module, :in_situ_ct_initial_face_flux_process)
        getfield(host_module, :in_situ_ct_initial_face_flux_process)(
            blocks, world_rank, Block_Nprocs, block_comms,
        )
        return true
    end
    return false
end

function initialize(Q, x, y, z, rankx, ranky, Nprocs, nxp, nyp, nzp, bid::Int=0)
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
    elseif test_case == "MagneticNozzle"
        @gpu_launch threads=nthreads blocks=nb init_magnetic_nozzle(
            Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp),
        )
    elseif test_case == "MagneticNozzleSheth"
        @gpu_launch threads=nthreads blocks=nb init_sheth_magnetic_nozzle(
            Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp),
        )
    elseif test_case == "PipeFlow"
        # ── SEM: generate synthetic eddies on CPU, transfer to GPU ──
        N_sem = 1000
        l_sem = FT(0.15 * R0)   # eddy size ~ 15% of pipe radius
        amp_sem = FT(0.10 * Ma_target * sqrt(γ * Rg * Tw))  # 10% of u_bulk

        is_precursor = (bid >= 5)
        precursor_length = isdefined(Main, :cebl_Lx) ? FT(Main.cebl_Lx) : Lx
        L_domain = is_precursor ? precursor_length : Lx
        is_visc = isdefined(Main, :viscous) ? Bool(Main.viscous) : true

        # Random eddy positions within the pipe domain
        pos_x_h = is_precursor ? FT.(rand(N_sem) .* L_domain .- L_domain) : FT.(rand(N_sem) .* L_domain)
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
            Int32(N_sem), l_sem, amp_sem, is_visc, FT(L_domain)
        )
    elseif test_case == "BrioWu"
        @gpu_launch threads=nthreads blocks=nb init_brio_wu(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp))
    elseif test_case == "OrszagTang"
        @gpu_launch threads=nthreads blocks=nb init_orszag_tang(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp))
    elseif test_case == "OrszagTang3D"
        @gpu_launch threads=nthreads blocks=nb init_orszag_tang_3d(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp))
    elseif test_case == "MetricCTAlfven"
        @gpu_launch threads=nthreads blocks=nb init_metric_ct_alfven(
            Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp),
        )
    elseif test_case == "MetricCTUniform"
        @gpu_launch threads=nthreads blocks=nb init_metric_ct_uniform(
            Q, Int32(nxp), Int32(nyp), Int32(nzp),
        )
    elseif test_case == "TG5_debug"
        @gpu_launch threads=nthreads blocks=nb init_tg5_debug(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp))
    elseif test_case == "UniformDimensional"
        @gpu_launch threads=nthreads blocks=nb init_uniform_dimensional(Q, x, y, z)
    elseif test_case == "MagneticDecay"
        Lx_val = isdefined(Main, :Lx) ? Main.Lx : FT(7.5)
        @gpu_launch threads=nthreads blocks=nb init_magnetic_decay(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp), Lx_val)
    elseif test_case == "ResistiveCTDecay"
        Lx_val = isdefined(Main, :Lx) ? Main.Lx : one(FT)
        amplitude = isdefined(Main, :decay_amplitude) ?
            Main.decay_amplitude : FT(0.1) * SQRT_MU0_SI
        @gpu_launch threads=nthreads blocks=nb init_resistive_ct_decay(
            Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp),
            FT(Lx_val), FT(amplitude),
        )
    elseif test_case == "MHDdecay"
        # Decaying MHD turbulence with Passot-Pouquet divergence-free IC (CPU-generated)
        init_mhd_turbulence_field!(Q, x, y, z, nxp, nyp, nzp)
    elseif test_case == "MHDHIT"
        # Forced MHD turbulence: same Passot-Pouquet IC, but driven by HIT linear forcing
        init_mhd_turbulence_field!(Q, x, y, z, nxp, nyp, nzp)
    elseif test_case == "Hartmann"
        @gpu_launch threads=nthreads blocks=nb init_hartmann(Q, x, y, z, Int32(nxp), Int32(nyp), Int32(nzp))
    end
end

const _METRIC_CT_UNIFORM_STATE = isdefined(Main, :metric_ct_uniform_state) ?
    Main.metric_ct_uniform_state :
    (one(FT), FT(0.23), FT(-0.17), FT(0.11), one(FT), one(FT),
     FT(0.61) * SQRT_MU0_SI, FT(-0.37) * SQRT_MU0_SI,
     FT(0.29) * SQRT_MU0_SI, zero(FT))

function init_metric_ct_uniform(Q, nxp::Int32, nyp::Int32, nzp::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i < Int32(NG+1) || i > nxp+Int32(NG) ||
       j < Int32(NG+1) || j > nyp+Int32(NG) ||
       k < Int32(NG+1) || k > nzp+Int32(NG)
        return
    end
    state = _METRIC_CT_UNIFORM_STATE
    rho = state[1]
    pressure = state[5]
    @inbounds begin
        Q[i,j,k,1] = rho
        Q[i,j,k,2] = state[2]
        Q[i,j,k,3] = state[3]
        Q[i,j,k,4] = state[4]
        Q[i,j,k,5] = pressure
        Q[i,j,k,6] = pressure / (rho * Rg)
        Q[i,j,k,7] = state[7]
        Q[i,j,k,8] = state[8]
        Q[i,j,k,9] = state[9]
        Q[i,j,k,10] = state[10]
    end
    return
end

function init_metric_ct_alfven(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i < Int32(NG+1) || i > nxp+Int32(NG) ||
       j < Int32(NG+1) || j > nyp+Int32(NG) ||
       k < Int32(NG+1) || k > nzp+Int32(NG)
        return
    end
    center = metric_ct_cell_center(x, y, z, i, j, k)
    state = metric_ct_alfven_primitive(
        center[1], zero(FT), FT(Rg), metric_ct_alfven_default_amplitude(FT),
    )
    @inbounds for n in 1:10
        Q[i,j,k,n] = state[n]
    end
    return
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
    
    # Gaussian Acoustic Pulse Initialization
    V0 = one(FT); p0 = FT(10.0); rho0 = one(FT)
    u = FT(0.1); v = FT(0.1); w = zero(FT)
    r2 = (xc - FT(1.5))^2 + (yc - FT(1.5))^2 + (zc - FT(1.5))^2
    pulse = FT(0.1) * exp(-FT(10.0) * r2)
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
# The listed field amplitudes are legacy values; the stored fields are Tesla.
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
        Bx = FT(0.75) * SQRT_MU0_SI
        By = SQRT_MU0_SI
        Bz = zero(FT)
    else
        # Right state
        rho = FT(0.125e0); u = zero(FT); v = zero(FT); w = zero(FT); p = FT(0.1)
        Bx = FT(0.75) * SQRT_MU0_SI
        By = -SQRT_MU0_SI
        Bz = zero(FT)
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

    center = structured_cell_center_coordinates(x, y, z, i, j, k)
    xc = center[1]
    yc = center[2]

    rho = γ * γ
    p = γ
    u = -sin(yc)
    v = sin(xc)
    w = zero(FT)
    B_scale = SQRT_MU0_SI
    Bx = -B_scale * sin(yc)
    By = B_scale * sin(FT(2.0) * xc)
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

# ═══════════════════════════════════════════════════════════════════════
# 3D Orszag-Tang Vortex — 3D extension of the classic 2D MHD vortex problem.
# ═══════════════════════════════════════════════════════════════════════
# The 2D Orszag-Tang setup (u=-sin(y), v=sin(x), Bx=-sin(y), By=sin(2x)) is
# invariant along z, so a 3D run would stay 2D unless symmetry is broken by
# numerical noise (slow and ambiguous). Here we add z-dependent modes so the
# problem is genuinely 3D from t=0, producing MHD turbulence + shock
# interactions in all three directions.
#
# Domain: [0, 2π]³, triply periodic, γ = 5/3
# ρ = γ² (= 25/9), p = γ (= 5/3)
# u = -sin(y) - 0.5·sin(2z)
# v =  sin(x)
# w =  sin(2x) + 0.5·sin(z)
# Bx = -sin(y) - 0.5·sin(2z)
# By =  sin(2x)
# Bz =  sin(y) + 0.5·sin(2x)
# ψ = 0
function init_orszag_tang_3d(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+Int32(2*NG) || j > nyp+Int32(2*NG) || k > nzp+Int32(2*NG); return; end
    if i < Int32(NG+1) || i > nxp+Int32(NG) || j < Int32(NG+1) || j > nyp+Int32(NG) || k < Int32(NG+1) || k > nzp+Int32(NG); return; end

    @inbounds xc = x[i, j, k]
    @inbounds yc = y[i, j, k]
    @inbounds zc = z[i, j, k]

    rho = γ * γ
    p = γ
    u = -sin(yc) - FT(0.5) * sin(FT(2.0) * zc)
    v =  sin(xc)
    w =  sin(FT(2.0) * xc) + FT(0.5) * sin(zc)
    B_scale = SQRT_MU0_SI
    Bx = B_scale * (-sin(yc) - FT(0.5) * sin(FT(2.0) * zc))
    By = B_scale * sin(FT(2.0) * xc)
    # Each component is independent of its own coordinate, so div(B)=0
    # analytically and under the Cartesian face-difference CT operator.
    Bz = B_scale * (sin(yc) + FT(0.5) * sin(FT(2.0) * xc))
    T = p / (rho * Rg)

    @inbounds begin
        Q[i,j,k,1] = rho; Q[i,j,k,2] = u; Q[i,j,k,3] = v; Q[i,j,k,4] = w
        Q[i,j,k,5] = p;   Q[i,j,k,6] = T
        Q[i,j,k,7] = Bx;  Q[i,j,k,8] = By; Q[i,j,k,9] = Bz
        Q[i,j,k,10] = zero(FT)  # ψ
    end
    return
end

function init_magnetic_decay(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32, Lx_val::FT)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+Int32(2*NG) || j > nyp+Int32(2*NG) || k > nzp+Int32(2*NG); return; end
    if i < Int32(NG+1) || i > nxp+Int32(NG) || j < Int32(NG+1) || j > nyp+Int32(NG) || k < Int32(NG+1) || k > nzp+Int32(NG); return; end

    @inbounds xc = x[i, j, k]

    rho = one(FT)
    u = zero(FT)
    v = zero(FT)
    w = zero(FT)
    p = one(FT)
    T = p / (rho * Rg)

    Bx = zero(FT)
    By = SQRT_MU0_SI * FT(0.1) *
         sin(FT(2.0) * FT(pi) * xc / Lx_val)
    Bz = zero(FT)
    psi = zero(FT)

    @inbounds begin
        Q[i, j, k, 1] = rho
        Q[i, j, k, 2] = u
        Q[i, j, k, 3] = v
        Q[i, j, k, 4] = w
        Q[i, j, k, 5] = p
        Q[i, j, k, 6] = T
        Q[i, j, k, 7] = Bx
        Q[i, j, k, 8] = By
        Q[i, j, k, 9] = Bz
        Q[i, j, k, 10] = psi
    end
    return
end

function init_resistive_ct_decay(
    Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32,
    Lx_val::FT, amplitude::FT,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i < Int32(NG+1) || i > nxp+Int32(NG) ||
       j < Int32(NG+1) || j > nyp+Int32(NG) ||
       k < Int32(NG+1) || k > nzp+Int32(NG)
        return
    end

    @inbounds xc = FT(0.5) * (x[i,j,k] + x[i+1,j,k])
    wave_number = FT(2) * FT(pi) / Lx_val
    phase = wave_number * xc
    rho = one(FT)
    pressure = one(FT)
    @inbounds begin
        Q[i,j,k,1] = rho
        Q[i,j,k,2] = zero(FT)
        Q[i,j,k,3] = zero(FT)
        Q[i,j,k,4] = zero(FT)
        Q[i,j,k,5] = pressure
        Q[i,j,k,6] = pressure / (rho * Rg)
        Q[i,j,k,7] = zero(FT)
        Q[i,j,k,8] = amplitude * sin(phase)
        Q[i,j,k,9] = amplitude * cos(phase)
        Q[i,j,k,10] = zero(FT)
    end
    return
end

function init_hartmann(Q, x, y, z, nxp::Int32, nyp::Int32, nzp::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+Int32(2*NG) || j > nyp+Int32(2*NG) || k > nzp+Int32(2*NG); return; end
    if i < Int32(NG+1) || i > nxp+Int32(NG) || j < Int32(NG+1) || j > nyp+Int32(NG) || k < Int32(NG+1) || k > nzp+Int32(NG); return; end

    # B0 is a physical Tesla input.  The fallback preserves the old
    # Hartmann reference field with p0=1 Pa and rho0=1 kg/m^3.
    B0_val = isdefined(Main, :B0) ? Main.B0 : SQRT_MU0_SI

    rho = one(FT)
    u = FT(0.1)
    v = zero(FT)
    w = zero(FT)
    p = one(FT)
    T = p / (rho * Rg)

    Bx = zero(FT)
    By = B0_val
    Bz = zero(FT)
    psi = zero(FT)

    @inbounds begin
        Q[i, j, k, 1] = rho
        Q[i, j, k, 2] = u
        Q[i, j, k, 3] = v
        Q[i, j, k, 4] = w
        Q[i, j, k, 5] = p
        Q[i, j, k, 6] = T
        Q[i, j, k, 7] = Bx
        Q[i, j, k, 8] = By
        Q[i, j, k, 9] = Bz
        Q[i, j, k, 10] = psi
    end
    return
end
