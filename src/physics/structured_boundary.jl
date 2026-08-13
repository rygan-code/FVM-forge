# =============================================================================
# DSRFG Synthetic Turbulence Inflow Support
# Folded from dsrfg_inflow.jl.
# =============================================================================
using Random
using Adapt
if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end
if !isdefined(@__MODULE__, :structured_cell_center_coordinates)
    include(joinpath(@__DIR__, "..", "mesh", "structured_coordinates.jl"))
end
if !@isdefined(ct_mode)
    const ct_mode = false
end
include(joinpath(@__DIR__,"mhd_boundary_state.jl"))

# In CT mode the boundary kernel owns only the gas primitive state. Magnetic
# ghost values are recovered later from the synchronized staggered face fluxes.
const Nboundary_prim = equation_type == :MHD && ct_mode ? 6 : Nprim

# Boundary construction first sets the fluid state and the MHD stage below
# later inserts the prescribed/reflected magnetic field.  This helper keeps
# the intermediate carrier finite for gamma=1; the old p/(gamma-1) form was
# evaluated even when the isothermal branch was going to overwrite it.
@inline function boundary_hydro_energy_density(rho, u, v, w, pressure)
    return mhd_energy_density_from_primitive(
        rho, u, v, w, pressure, zero(FT), zero(FT), zero(FT), γ,
    )
end

struct DSRFGParams{V, FT}
    enabled::Bool
    N::Int32              # Number of Fourier modes
    Lt::FT                # Integral length scale
    TI::FT                # Target turbulence intensity at centerline
    C_sra::FT             # Strong Reynolds Analogy scaling factor
    u_bulk::FT            # Bulk velocity
    R0::FT                # Pipe radius
    # GPU Vectors of size N
    kx::V
    ky::V
    kz::V
    px::V
    py::V
    pz::V
    qx::V
    qy::V
    qz::V
    omega::V
end

# GPU adaptation so that CuArray/ROCArray are converted to CuDeviceArray/ROCDeviceArray inside the struct
Adapt.adapt_structure(to, params::DSRFGParams) = DSRFGParams(
    params.enabled,
    params.N,
    params.Lt,
    params.TI,
    params.C_sra,
    params.u_bulk,
    params.R0,
    adapt(to, params.kx),
    adapt(to, params.ky),
    adapt(to, params.kz),
    adapt(to, params.px),
    adapt(to, params.py),
    adapt(to, params.pz),
    adapt(to, params.qx),
    adapt(to, params.qy),
    adapt(to, params.qz),
    adapt(to, params.omega)
)


"""
    create_dummy_dsrfg_params(FT)

Create a dummy DSRFGParams struct with size 0 vectors of the correct GPU Vector type.
This ensures type stability in the GPU kernel even when DSRFG is disabled.
"""
function create_dummy_dsrfg_params(FT::Type)
    V = GPUVector{FT}
    kx = V(zeros(FT, 0))
    ky = V(zeros(FT, 0))
    kz = V(zeros(FT, 0))
    px = V(zeros(FT, 0))
    py = V(zeros(FT, 0))
    pz = V(zeros(FT, 0))
    qx = V(zeros(FT, 0))
    qy = V(zeros(FT, 0))
    qz = V(zeros(FT, 0))
    omega = V(zeros(FT, 0))
    
    return DSRFGParams{V, FT}(
        false, Int32(0), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT),
        kx, ky, kz, px, py, pz, qx, qy, qz, omega
    )
end

"""
    init_dsrfg_params(N, Lt, TI, C_sra, u_bulk, Re, R0; seed=42)

Initialize DSRFG parameters on the CPU using the Modified von Kármán spectrum,
apply the divergence-free orthogonal projection, and upload to the GPU.
"""
function init_dsrfg_params(N::Int, Lt::FT, TI::FT, C_sra::FT, u_bulk::FT, Re::FT, R0::FT; seed::Int=42) where {FT}
    # Wavenumber range limits (based on integral scale Lt)
    k_e = FT(9.0 * π) / (FT(55.0) * Lt)
    k_min = FT(0.5) * k_e
    k_max = FT(200.0) * k_e
    
    dk = (k_max - k_min) / FT(N)
    u_rms = TI * u_bulk
    
    # Pre-allocate CPU arrays
    h_kx = zeros(FT, N)
    h_ky = zeros(FT, N)
    h_kz = zeros(FT, N)
    h_px = zeros(FT, N)
    h_py = zeros(FT, N)
    h_pz = zeros(FT, N)
    h_qx = zeros(FT, N)
    h_qy = zeros(FT, N)
    h_qz = zeros(FT, N)
    h_omega = zeros(FT, N)
    
    # Set seed for reproducibility
    rng = Random.MersenneTwister(seed)
    
    for n in 1:N
        k_n = k_min + (FT(n) - FT(0.5)) * dk
        
        # 1. Wavenumber unit vector direction
        theta = acos(FT(2.0) * rand(rng, FT) - FT(1.0))
        phi = FT(2.0 * π) * rand(rng, FT)
        
        kn_x = k_n * sin(theta) * cos(phi)
        kn_y = k_n * sin(theta) * sin(phi)
        kn_z = k_n * cos(theta)
        
        h_kx[n] = kn_x
        h_ky[n] = kn_y
        h_kz[n] = kn_z
        
        # 2. Amplitude based on Modified von Kármán spectrum
        ratio = k_n / k_e
        # Integral of ratio^4 / (1 + ratio^2)^(17/6) from 0 to Inf is ~0.82688
        E_kn = FT(1.5) * (u_rms^2 / k_e) * (ratio^4 / (1.0 + ratio^2)^(17/6)) / FT(0.82688)
        sigma_n = sqrt(E_kn * dk)
        
        # 3. Divergence-free projection (p_n and q_n orthogonal to k_n)
        d_x = rand(rng, FT) - FT(0.5)
        d_y = rand(rng, FT) - FT(0.5)
        d_z = rand(rng, FT) - FT(0.5)
        
        # Normalize wavenumber direction
        kx_u = kn_x / k_n
        ky_u = kn_y / k_n
        kz_u = kn_z / k_n
        
        # Cross product: p_vec = d x k_u
        px_val = d_y * kz_u - d_z * ky_u
        py_val = d_z * kx_u - d_x * kz_u
        pz_val = d_x * ky_u - d_y * kx_u
        
        p_mag = sqrt(px_val^2 + py_val^2 + pz_val^2)
        if p_mag < FT(1e-12)
            px_val, py_val, pz_val = -ky_u, kx_u, zero(FT)
            p_mag = sqrt(px_val^2 + py_val^2)
        end
        px_val = (px_val / p_mag) * sigma_n
        py_val = (py_val / p_mag) * sigma_n
        pz_val = (pz_val / p_mag) * sigma_n
        
        h_px[n] = px_val
        h_py[n] = py_val
        h_pz[n] = pz_val
        
        # Cross product: q_vec = k_u x p_vec
        qx_val = ky_u * pz_val - kz_u * py_val
        qy_val = kz_u * px_val - kx_u * pz_val
        qz_val = kx_u * py_val - ky_u * px_val
        
        h_qx[n] = qx_val
        h_qy[n] = qy_val
        h_qz[n] = qz_val
        
        # 4. Temporal frequency
        U_conv = FT(0.8) * u_bulk
        omega_conv = k_n * U_conv
        h_omega[n] = omega_conv + randn(rng, FT) * k_n * u_rms
    end
    
    # 5. Upload to GPU
    V = GPUVector{FT}
    kx = V(h_kx)
    ky = V(h_ky)
    kz = V(h_kz)
    px = V(h_px)
    py = V(h_py)
    pz = V(h_pz)
    qx = V(h_qx)
    qy = V(h_qy)
    qz = V(h_qz)
    omega = V(h_omega)
    
    return DSRFGParams{V, FT}(
        true, Int32(N), Lt, TI, C_sra, u_bulk, R0,
        kx, ky, kz, px, py, pz, qx, qy, qz, omega
    )
end

"""
    dsrfg_fluctuation(xi, yi, zi, t, params)

GPU device function to calculate DSRFG velocity fluctuations at coordinate (xi, yi, zi) and time t.
Returns (u_pr, v_pr, w_pr) scaled by the radial envelope.
"""
@inline function dsrfg_fluctuation(xi::FT, yi::FT, zi::FT, t::FT, params::DSRFGParams{V, FT}) where {FT, V}
    u_pr = zero(FT)
    v_pr = zero(FT)
    w_pr = zero(FT)
    
    if !params.enabled
        return u_pr, v_pr, w_pr
    end
    
    N = params.N
    @inbounds for n in 1:N
        # arg = k_x * x + k_y * y + k_z * z + ω * t
        arg = params.kx[n] * xi + params.ky[n] * yi + params.kz[n] * zi + params.omega[n] * t
        cos_val = cos(arg)
        sin_val = sin(arg)
        
        u_pr += params.px[n] * cos_val + params.qx[n] * sin_val
        v_pr += params.py[n] * cos_val + params.qy[n] * sin_val
        w_pr += params.pz[n] * cos_val + params.qz[n] * sin_val
    end
    
    factor = sqrt(FT(2.0) / FT(N))
    u_pr *= factor
    v_pr *= factor
    w_pr *= factor
    
    # Smooth radial envelope to force fluctuations to zero at the wall (r=R0)
    # peaks at r/R0 = 0.5 with value 1.125 to mimic real wall turbulence intensities
    r2 = yi*yi + zi*zi
    r2_norm = r2 / (params.R0 * params.R0)
    env = max(one(FT) - r2_norm, zero(FT)) * (one(FT) + FT(2.0) * r2_norm)
    
    u_pr *= env
    v_pr *= env
    w_pr *= env
    
    return u_pr, v_pr, w_pr
end


# =============================================================================
# Multi-block boundary conditions — Data-driven BC dispatch (FVM)
# BC types defined in bc_types.jl, stored per-face in block_connectivity.h5
# =============================================================================

# =============================================================================
# Multi-block boundary conditions — Data-driven BC dispatch (FVM)
# BC types defined in bc_types.jl, stored per-face in block_connectivity.h5
# =============================================================================

# ─── Differential-rotation wall-BC parameters (module-load-time capture) ───
#
# These constants are evaluated WHEN this file is first included (transitively
# from `solver.jl`). The hosting run script MUST therefore define
#   Main.diffrot_enabled, Main.x_rot_start, Main.x_rot_end,
#   Main.Omega_x_min,     Main.Omega_x_max
# BEFORE `include("solver.jl")`. If any of them is missing at load time it is
# silently substituted with 0 / false, which makes the wall BC apply Ω = 0 with
# no other warning. See `run_pipe_cebl_diffrot.jl` for the canonical ordering.
const diffrot_enabled_bc::Bool = isdefined(Main, :diffrot_enabled) ? Main.diffrot_enabled : false
const x_rot_start_bc::Float64 = isdefined(Main, :x_rot_start) ? Main.x_rot_start : 0.0
const x_rot_end_bc::Float64 = isdefined(Main, :x_rot_end) ? Main.x_rot_end : 0.0
const Omega_x_min_bc::Float64 = isdefined(Main, :Omega_x_min) ? Main.Omega_x_min : 0.0
const Omega_x_max_bc::Float64 = isdefined(Main, :Omega_x_max) ? Main.Omega_x_max : 0.0

@inline function get_wall_rotation(x_loc::T) where T
    if !diffrot_enabled_bc
        return zero(T)
    end
    if x_loc <= T(x_rot_start_bc)
        return T(Omega_x_min_bc)
    elseif x_loc >= T(x_rot_end_bc)
        return T(Omega_x_max_bc)
    else
        frac = (x_loc - T(x_rot_start_bc)) / T(x_rot_end_bc - x_rot_start_bc)
        return T(Omega_x_min_bc) + frac * T(Omega_x_max_bc - Omega_x_min_bc)
    end
end

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
                    n_x, n_y, n_z, Area, Vol, dt_bc, x, y, z,
                    nxp, nyp, nzp, bcp, tt, dsrfg_params,
                    external_field_cache)

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

    # ── Magnetic-nozzle external field boundary ──
    if bc_type == Int32(BC_MHD_EXTERNAL_FIELD)
        # The fluid sees a no-slip outer boundary. The MHD extension below
        # replaces the ghost magnetic field with the prescribed coil field.
        if dir == Int32(1)
            @inbounds begin
                rho=Q[idx_int,j,k,1]; u=-Q[idx_int,j,k,2]; v=-Q[idx_int,j,k,3]; w=-Q[idx_int,j,k,4]
                p=Q[idx_int,j,k,5]; T=Q[idx_int,j,k,6]
                Q[i,j,k,1]=rho; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=rho; U[i,j,k,2]=rho*u; U[i,j,k,3]=rho*v; U[i,j,k,4]=rho*w
                U[i,j,k,5]=boundary_hydro_energy_density(rho,u,v,w,p)
            end
        elseif dir == Int32(2)
            @inbounds begin
                rho=Q[i,idx_int,k,1]; u=-Q[i,idx_int,k,2]; v=-Q[i,idx_int,k,3]; w=-Q[i,idx_int,k,4]
                p=Q[i,idx_int,k,5]; T=Q[i,idx_int,k,6]
                Q[i,j,k,1]=rho; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=rho; U[i,j,k,2]=rho*u; U[i,j,k,3]=rho*v; U[i,j,k,4]=rho*w
                U[i,j,k,5]=boundary_hydro_energy_density(rho,u,v,w,p)
            end
        else
            @inbounds begin
                rho=Q[i,j,idx_int,1]; u=-Q[i,j,idx_int,2]; v=-Q[i,j,idx_int,3]; w=-Q[i,j,idx_int,4]
                p=Q[i,j,idx_int,5]; T=Q[i,j,idx_int,6]
                Q[i,j,k,1]=rho; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=rho; U[i,j,k,2]=rho*u; U[i,j,k,3]=rho*v; U[i,j,k,4]=rho*w
                U[i,j,k,5]=boundary_hydro_energy_density(rho,u,v,w,p)
            end
        end

    # ── Magnetic-nozzle reservoir inlet ──
    elseif bc_type == Int32(BC_MHD_PROFILED_INFLOW)
        @inbounds begin
            rho0 = bcp[BCP_MN_RHO0]
            T = bcp[BCP_MN_T0]
            a = max(bcp[BCP_MN_RB], FT(1.0e-30))
            kappa = bcp[BCP_MN_KAPPA]
            center = structured_cell_center_coordinates(x, y, z, i, j, k)
            radius = sqrt(center[2]^2 + center[3]^2)
            # Sheth et al. define sech^2[kappa (r/a)^2], not sech^2 of
            # kappa^2.  The distinction changes the injected radial profile
            # and therefore the mass-loading and inlet Mach number.
            profile_arg = kappa * (radius/a)^2
            profile = inv(cosh(profile_arg)^2)
            rho = rho0 * (one(FT) + FT(20.0)*profile)
            p = rho * Rg * T
            c_s = sqrt(isothermal_mhd ? Rg * T : γ * Rg * T)
            u = bcp[BCP_MN_V0] * c_s * profile
            v = zero(FT); w = zero(FT)
            Q[i,j,k,1]=rho; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
            U[i,j,k,1]=rho; U[i,j,k,2]=rho*u; U[i,j,k,3]=zero(FT); U[i,j,k,4]=zero(FT)
            U[i,j,k,5]=mhd_energy_density_from_primitive(
                rho, u, v, w, p, zero(FT), zero(FT), zero(FT), γ,
            )
        end

    elseif bc_type == Int32(BC_MHD_RESERVOIR_INFLOW)
        @inbounds begin
            rho=bcp[BCP_MN_RHO0]; T=bcp[BCP_MN_T0]; p=rho*Rg*T
            if dir == Int32(1)
                u=Q[idx_bnd,j,k,2]; v=Q[idx_bnd,j,k,3]; w=Q[idx_bnd,j,k,4]
            elseif dir == Int32(2)
                u=Q[i,idx_bnd,k,2]; v=Q[i,idx_bnd,k,3]; w=Q[i,idx_bnd,k,4]
            else
                u=Q[i,j,idx_bnd,2]; v=Q[i,j,idx_bnd,3]; w=Q[i,j,idx_bnd,4]
            end
            Q[i,j,k,1]=rho; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
            U[i,j,k,1]=rho; U[i,j,k,2]=rho*u; U[i,j,k,3]=rho*v; U[i,j,k,4]=rho*w
            U[i,j,k,5]=boundary_hydro_energy_density(rho,u,v,w,p)
        end

    # ── Isothermal Wall / Differential Rotation Wall ──
    elseif bc_type == Int32(BC_MHD_FIXED_EXTERNAL_FIELD)
        @inbounds begin
            rho=bcp[BCP_MN_RHO0]; T=bcp[BCP_MN_T0]; p=rho*Rg*T
            u=zero(FT); v=zero(FT); w=zero(FT)
            Q[i,j,k,1]=rho; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
            U[i,j,k,1]=rho; U[i,j,k,2]=zero(FT); U[i,j,k,3]=zero(FT); U[i,j,k,4]=zero(FT)
            U[i,j,k,5]=mhd_energy_density_from_primitive(
                rho, u, v, w, p, zero(FT), zero(FT), zero(FT), γ,
            )
        end

    elseif bc_type == Int32(BC_MHD_OUTFLOW_EXTERNAL_FIELD)
        if dir == Int32(1)
            for n = 1:Nboundary_prim; @inbounds Q[i,j,k,n] = Q[idx_bnd,j,k,n]; end
            for n = 1:Ncell_cons; @inbounds U[i,j,k,n] = U[idx_bnd,j,k,n]; end
        elseif dir == Int32(2)
            for n = 1:Nboundary_prim; @inbounds Q[i,j,k,n] = Q[i,idx_bnd,k,n]; end
            for n = 1:Ncell_cons; @inbounds U[i,j,k,n] = U[i,idx_bnd,k,n]; end
        else
            for n = 1:Nboundary_prim; @inbounds Q[i,j,k,n] = Q[i,j,idx_bnd,n]; end
            for n = 1:Ncell_cons; @inbounds U[i,j,k,n] = U[i,j,idx_bnd,n]; end
        end

    elseif bc_type == Int32(BC_ISOTHERMAL_WALL) || bc_type == Int32(BC_MHD_INSULATING_WALL) || bc_type == Int32(BC_DIFFROT_WALL)
        Tw_val = bcp[1]  # BCP_TW
        if Tw_val <= zero(FT); Tw_val = Tw; end
        if dir == Int32(1)
            x_face = FT(0.5) * (x[i, j, k] + x[idx_int, j, k])
            y_face = FT(0.5) * (y[i, j, k] + y[idx_int, j, k])
            z_face = FT(0.5) * (z[i, j, k] + z[idx_int, j, k])
            Omega_w = bc_type == Int32(BC_DIFFROT_WALL) ? get_wall_rotation(x_face) : zero(FT)
            u_rot = zero(FT); v_rot = -Omega_w * z_face; w_rot = Omega_w * y_face
            
            u_turb, v_turb, w_turb = _wall_perturbation(idx_wall, j, k, Int32(0), x, y, z, tt, nxp)
            @inbounds begin
                u = -Q[idx_int, j, k, 2] + FT(2.0)*(u_turb + u_rot)
                v = -Q[idx_int, j, k, 3] + FT(2.0)*(v_turb + v_rot)
                w = -Q[idx_int, j, k, 4] + FT(2.0)*(w_turb + w_rot)
                T = FT(2.0)*Tw_val - Q[idx_int, j, k, 6]
                T = max(T, FT(0.1)*Tw_val)  # clamp to avoid negative T when T_int >> 2*Tw
                p = Q[idx_int, j, k, 5]
                ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=boundary_hydro_energy_density(ρ,u,v,w,p)
            end
        elseif dir == Int32(2)
            x_face = FT(0.5) * (x[i, j, k] + x[i, idx_int, k])
            y_face = FT(0.5) * (y[i, j, k] + y[i, idx_int, k])
            z_face = FT(0.5) * (z[i, j, k] + z[i, idx_int, k])
            Omega_w = bc_type == Int32(BC_DIFFROT_WALL) ? get_wall_rotation(x_face) : zero(FT)
            u_rot = zero(FT); v_rot = -Omega_w * z_face; w_rot = Omega_w * y_face
            
            u_turb, v_turb, w_turb = _wall_perturbation(i, idx_wall, k, Int32(0), x, y, z, tt, nxp)
            @inbounds begin
                u = -Q[i, idx_int, k, 2] + FT(2.0)*(u_turb + u_rot)
                v = -Q[i, idx_int, k, 3] + FT(2.0)*(v_turb + v_rot)
                w = -Q[i, idx_int, k, 4] + FT(2.0)*(w_turb + w_rot)
                T = FT(2.0)*Tw_val - Q[i, idx_int, k, 6]
                T = max(T, FT(0.1)*Tw_val)
                p = Q[i, idx_int, k, 5]
                ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=boundary_hydro_energy_density(ρ,u,v,w,p)
            end
        else # ζ
            x_face = FT(0.5) * (x[i, j, k] + x[i, j, idx_int])
            y_face = FT(0.5) * (y[i, j, k] + y[i, j, idx_int])
            z_face = FT(0.5) * (z[i, j, k] + z[i, j, idx_int])
            Omega_w = bc_type == Int32(BC_DIFFROT_WALL) ? get_wall_rotation(x_face) : zero(FT)
            u_rot = zero(FT); v_rot = -Omega_w * z_face; w_rot = Omega_w * y_face
            
            u_turb, v_turb, w_turb = _wall_perturbation(i, j, idx_wall, Int32(0), x, y, z, tt, nxp)
            @inbounds begin
                u = -Q[i, j, idx_int, 2] + FT(2.0)*(u_turb + u_rot)
                v = -Q[i, j, idx_int, 3] + FT(2.0)*(v_turb + v_rot)
                w = -Q[i, j, idx_int, 4] + FT(2.0)*(w_turb + w_rot)
                T = FT(2.0)*Tw_val - Q[i, j, idx_int, 6]
                T = max(T, FT(0.1)*Tw_val)
                p = Q[i, j, idx_int, 5]
                ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=boundary_hydro_energy_density(ρ,u,v,w,p)
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
                U[i,j,k,5]=boundary_hydro_energy_density(ρ,u,v,w,p)
            end
        elseif dir == Int32(2)
            @inbounds begin
                u = -Q[i, idx_int, k, 2]; v = -Q[i, idx_int, k, 3]; w = -Q[i, idx_int, k, 4]
                T = Q[i, idx_int, k, 6]; p = Q[i, idx_int, k, 5]; ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=boundary_hydro_energy_density(ρ,u,v,w,p)
            end
        else
            @inbounds begin
                u = -Q[i, j, idx_int, 2]; v = -Q[i, j, idx_int, 3]; w = -Q[i, j, idx_int, 4]
                T = Q[i, j, idx_int, 6]; p = Q[i, j, idx_int, 5]; ρ = p/(Rg*T)
                Q[i,j,k,1]=ρ; Q[i,j,k,2]=u; Q[i,j,k,3]=v; Q[i,j,k,4]=w; Q[i,j,k,5]=p; Q[i,j,k,6]=T
                U[i,j,k,1]=ρ; U[i,j,k,2]=ρ*u; U[i,j,k,3]=ρ*v; U[i,j,k,4]=ρ*w
                U[i,j,k,5]=boundary_hydro_energy_density(ρ,u,v,w,p)
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
                U[i,j,k,5]=boundary_hydro_energy_density(ρ,u_g,v_g,w_g,p)
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
                U[i,j,k,5]=boundary_hydro_energy_density(ρ,u_g,v_g,w_g,p)
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
                U[i,j,k,5]=boundary_hydro_energy_density(ρ,u_g,v_g,w_g,p)
            end
        end

    # ── Supersonic Inflow — all variables fixed ──
    elseif bc_type == Int32(BC_SUPERSONIC_INFLOW)
        @inbounds begin
            p1 = bcp[BCP_P_INF]
            ρ1 = bcp[BCP_RHO_INF]
            u1 = bcp[BCP_U_INF]
            v1 = bcp[BCP_V_INF]
            w1 = bcp[BCP_W_INF]
            
            Q[i,j,k,1]=ρ1; Q[i,j,k,2]=u1; Q[i,j,k,3]=v1; Q[i,j,k,4]=w1; Q[i,j,k,5]=p1; Q[i,j,k,6]=p1/(Rg*ρ1)
            U[i,j,k,1]=ρ1; U[i,j,k,2]=ρ1*u1; U[i,j,k,3]=ρ1*v1; U[i,j,k,4]=ρ1*w1
            U[i,j,k,5]=boundary_hydro_energy_density(ρ1,u1,v1,w1,p1)
        end

    # ── CEBL Dynamic Inflow ──
    elseif bc_type == Int32(BC_CEBL_INFLOW)
        # Directly return since ghost cells are pre-populated by GPU direct memory copy
        return

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
            U[i,j,k,5]=boundary_hydro_energy_density(ρ1,u1,v1,w1,p_ext)
        end

    # ── Wave Inflow — Deterministic wave injection ──
    elseif bc_type == Int32(BC_WAVE_INFLOW)
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
            
            # Read wave parameters from boundary parameters
            amp     = bcp[1] # BCP_WAVE_AMP
            omega   = bcp[2] # BCP_WAVE_OMEGA
            m_mode  = bcp[3] # BCP_WAVE_M
            
            is_visc = (isdefined(Main, :viscous) ? Main.viscous : true)
            if is_visc
                # 1/7 power law base profile
                u_cl = (FT(60.0)/FT(49.0)) * u_bulk
                u_base = u_cl * (max(one(FT) - sqrt(r2_norm), zero(FT))^(one(FT)/FT(7.0)))
                
                # Base temperature profile with viscous heating
                Ma2 = Ma_target * Ma_target
                β = FT(0.5) * Pr * (γ - one(FT)) * Ma2
                T_base = Tw * (one(FT) + β * max(one(FT) - r2_norm*r2_norm, zero(FT)))
            else
                u_base = u_bulk
                T_base = Tw
            end
            
            # Deterministic wave structure
            envelope = max(one(FT) - r2_norm, zero(FT))
            θ = atan(zi, yi)
            tt_f = FT(tt)
            
            # Phase: m * theta - omega * t
            phase = m_mode * θ - omega * tt_f
            
            u_pr = amp * envelope * cos(phase)
            v_pr = amp * envelope * cos(phase + θ)
            w_pr = amp * envelope * sin(phase + θ)
            
            u1 = u_base + u_pr
            v1 = v_pr
            w1 = w_pr
            
            T1 = max(T_base, FT(50.0))
            ρ1 = p_ext / (Rg * T1)
            
            Q[i,j,k,1]=ρ1; Q[i,j,k,2]=u1; Q[i,j,k,3]=v1; Q[i,j,k,4]=w1; Q[i,j,k,5]=p_ext; Q[i,j,k,6]=T1
            U[i,j,k,1]=ρ1; U[i,j,k,2]=ρ1*u1; U[i,j,k,3]=ρ1*v1; U[i,j,k,4]=ρ1*w1
            U[i,j,k,5]=boundary_hydro_energy_density(ρ1,u1,v1,w1,p_ext)
        end

    # ── Supersonic Outflow — zero gradient (copy from boundary cell) ──
    elseif bc_type == Int32(BC_SUPERSONIC_OUTFLOW)
        if dir == Int32(1)
            for n = 1:Nboundary_prim; @inbounds Q[i,j,k,n] = Q[idx_bnd,j,k,n]; end
            for n = 1:Ncell_cons; @inbounds U[i,j,k,n] = U[idx_bnd,j,k,n]; end
        elseif dir == Int32(2)
            for n = 1:Nboundary_prim; @inbounds Q[i,j,k,n] = Q[i,idx_bnd,k,n]; end
            for n = 1:Ncell_cons; @inbounds U[i,j,k,n] = U[i,idx_bnd,k,n]; end
        else
            for n = 1:Nboundary_prim; @inbounds Q[i,j,k,n] = Q[i,j,idx_bnd,n]; end
            for n = 1:Ncell_cons; @inbounds U[i,j,k,n] = U[i,j,idx_bnd,n]; end
        end

    # ── Zero Gradient (Neumann) — copy from boundary cell ──
    elseif bc_type == Int32(BC_ZERO_GRADIENT)
        if dir == Int32(1)
            for n = 1:Nboundary_prim; @inbounds Q[i,j,k,n] = Q[idx_bnd,j,k,n]; end
            for n = 1:Ncell_cons; @inbounds U[i,j,k,n] = U[idx_bnd,j,k,n]; end
        elseif dir == Int32(2)
            for n = 1:Nboundary_prim; @inbounds Q[i,j,k,n] = Q[i,idx_bnd,k,n]; end
            for n = 1:Ncell_cons; @inbounds U[i,j,k,n] = U[i,idx_bnd,k,n]; end
        else
            for n = 1:Nboundary_prim; @inbounds Q[i,j,k,n] = Q[i,j,idx_bnd,n]; end
            for n = 1:Ncell_cons; @inbounds U[i,j,k,n] = U[i,j,idx_bnd,n]; end
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
            U[i,j,k,5]=boundary_hydro_energy_density(
                ρ_local, V_mag*dx, V_mag*dy, V_mag*dz, p_ext,
            )
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
            U[i,j,k,5]=boundary_hydro_energy_density(ρ,u,v,w,p_back)
        end

    # ── NSCBC Outflow — metric-aligned, dt-scaled LODI with local weak anchor ──
    # Outgoing waves: computed from physical-normal gradients (one-sided 2nd order)
    # Incoming wave L1: weak relaxation toward block outlet area-averaged p (p_anchor,
    #   bcp[BCP_OUTLET_PAVG]=[15]) — NOT a fixed inlet/back-pressure. When p_anchor<=0
    #   (uninitialized) we fall back to pure non-reflecting (L1=0).
    # Supersonic outflow (Ma_n>=1): all characteristics exit → zero gradient.
    # Backflow (u_n<=0): soft inflow — strengthened L1 anchor toward p_anchor, tangential
    #   relaxed to boundary value (NOT a reflective zero-gradient wall).
    elseif bc_type == Int32(BC_NSCBC_OUTFLOW)
        @inbounds begin
            # Read boundary cell primitives and face geometry for this direction.
            if dir == Int32(1)
                ρ_b=Q[idx_bnd,j,k,1]; u_b=Q[idx_bnd,j,k,2]; v_b=Q[idx_bnd,j,k,3]; w_b=Q[idx_bnd,j,k,4]; p_b=Q[idx_bnd,j,k,5]
                nx0=n_x[idx_wall,j,k]; ny0=n_y[idx_wall,j,k]; nz0=n_z[idx_wall,j,k]
                A_b=Area[idx_wall,j,k]; J_b=Vol[idx_bnd,j,k]
            elseif dir == Int32(2)
                ρ_b=Q[i,idx_bnd,k,1]; u_b=Q[i,idx_bnd,k,2]; v_b=Q[i,idx_bnd,k,3]; w_b=Q[i,idx_bnd,k,4]; p_b=Q[i,idx_bnd,k,5]
                nx0=n_x[i,idx_wall,k]; ny0=n_y[i,idx_wall,k]; nz0=n_z[i,idx_wall,k]
                A_b=Area[i,idx_wall,k]; J_b=Vol[i,idx_bnd,k]
            else
                ρ_b=Q[i,j,idx_bnd,1]; u_b=Q[i,j,idx_bnd,2]; v_b=Q[i,j,idx_bnd,3]; w_b=Q[i,j,idx_bnd,4]; p_b=Q[i,j,idx_bnd,5]
                nx0=n_x[i,j,idx_wall]; ny0=n_y[i,j,idx_wall]; nz0=n_z[i,j,idx_wall]
                A_b=Area[i,j,idx_wall]; J_b=Vol[i,j,idx_bnd]
            end

            # Outward normal: metric normals point along +computational direction.
            s_out = side == Int32(1) ? one(FT) : -one(FT)
            nx = s_out*nx0; ny = s_out*ny0; nz = s_out*nz0
            nmag = sqrt(nx*nx + ny*ny + nz*nz + FT(1.0e-30))
            nx /= nmag; ny /= nmag; nz /= nmag

            c = sqrt(γ * max(p_b, FT(1.0e-30)) / max(ρ_b, FT(1.0e-30)))
            u_n = u_b*nx + v_b*ny + w_b*nz
            Ma_n = abs(u_n) / (c + FT(1.0e-30))

            # Supersonic outflow: all characteristics exit → zero gradient.
            if (Ma_n >= one(FT)) && (u_n > zero(FT))
                if dir == Int32(1)
                    for n = 1:Nboundary_prim; Q[i,j,k,n] = Q[idx_bnd,j,k,n]; end
                    for n = 1:Ncell_cons; U[i,j,k,n] = U[idx_bnd,j,k,n]; end
                elseif dir == Int32(2)
                    for n = 1:Nboundary_prim; Q[i,j,k,n] = Q[i,idx_bnd,k,n]; end
                    for n = 1:Ncell_cons; U[i,j,k,n] = U[i,idx_bnd,k,n]; end
                else
                    for n = 1:Nboundary_prim; Q[i,j,k,n] = Q[i,j,idx_bnd,n]; end
                    for n = 1:Ncell_cons; U[i,j,k,n] = U[i,j,idx_bnd,n]; end
                end
            else
                # ── Subsonic / backflow: metric-aligned LODI with dt scaling ──
                # Build orthonormal tangential basis (l, m) from outward normal.
                ax = abs(nx) < FT(0.9) ? one(FT) : zero(FT)
                ay = abs(nx) < FT(0.9) ? zero(FT) : one(FT)
                az = zero(FT)
                lx = ay*nz - az*ny; ly = az*nx - ax*nz; lz = ax*ny - ay*nx
                lmag = sqrt(lx*lx + ly*ly + lz*lz + FT(1.0e-30))
                lx /= lmag; ly /= lmag; lz /= lmag
                mx = ny*lz - nz*ly; my = nz*lx - nx*lz; mz = nx*ly - ny*lx

                u_l = u_b*lx + v_b*ly + w_b*lz
                u_m = u_b*mx + v_b*my + w_b*mz

                # One-sided 2nd-order differences in index space at the boundary cell.
                if dir == Int32(1)
                    if side == Int32(1); i1 = idx_bnd-1; i2 = idx_bnd-2; s_der = one(FT)
                    else;                 i1 = idx_bnd+1; i2 = idx_bnd+2; s_der = -one(FT); end
                    dρ = s_der*(FT(1.5)*ρ_b - FT(2.0)*Q[i1,j,k,1] + FT(0.5)*Q[i2,j,k,1])
                    du = s_der*(FT(1.5)*u_b - FT(2.0)*Q[i1,j,k,2] + FT(0.5)*Q[i2,j,k,2])
                    dv = s_der*(FT(1.5)*v_b - FT(2.0)*Q[i1,j,k,3] + FT(0.5)*Q[i2,j,k,3])
                    dw = s_der*(FT(1.5)*w_b - FT(2.0)*Q[i1,j,k,4] + FT(0.5)*Q[i2,j,k,4])
                    dp = s_der*(FT(1.5)*p_b - FT(2.0)*Q[i1,j,k,5] + FT(0.5)*Q[i2,j,k,5])
                elseif dir == Int32(2)
                    if side == Int32(1); j1 = idx_bnd-1; j2 = idx_bnd-2; s_der = one(FT)
                    else;                 j1 = idx_bnd+1; j2 = idx_bnd+2; s_der = -one(FT); end
                    dρ = s_der*(FT(1.5)*ρ_b - FT(2.0)*Q[i,j1,k,1] + FT(0.5)*Q[i,j2,k,1])
                    du = s_der*(FT(1.5)*u_b - FT(2.0)*Q[i,j1,k,2] + FT(0.5)*Q[i,j2,k,2])
                    dv = s_der*(FT(1.5)*v_b - FT(2.0)*Q[i,j1,k,3] + FT(0.5)*Q[i,j2,k,3])
                    dw = s_der*(FT(1.5)*w_b - FT(2.0)*Q[i,j1,k,4] + FT(0.5)*Q[i,j2,k,4])
                    dp = s_der*(FT(1.5)*p_b - FT(2.0)*Q[i,j1,k,5] + FT(0.5)*Q[i,j2,k,5])
                else
                    if side == Int32(1); k1 = idx_bnd-1; k2 = idx_bnd-2; s_der = one(FT)
                    else;                 k1 = idx_bnd+1; k2 = idx_bnd+2; s_der = -one(FT); end
                    dρ = s_der*(FT(1.5)*ρ_b - FT(2.0)*Q[i,j,k1,1] + FT(0.5)*Q[i,j,k2,1])
                    du = s_der*(FT(1.5)*u_b - FT(2.0)*Q[i,j,k1,2] + FT(0.5)*Q[i,j,k2,2])
                    dv = s_der*(FT(1.5)*v_b - FT(2.0)*Q[i,j,k1,3] + FT(0.5)*Q[i,j,k2,3])
                    dw = s_der*(FT(1.5)*w_b - FT(2.0)*Q[i,j,k1,4] + FT(0.5)*Q[i,j,k2,4])
                    dp = s_der*(FT(1.5)*p_b - FT(2.0)*Q[i,j,k1,5] + FT(0.5)*Q[i,j,k2,5])
                end

                # Physical normal derivatives. Vol stores J = 1/cell_volume.
                Δn = one(FT) / (max(J_b, FT(1.0e-30)) * max(A_b, FT(1.0e-30)))
                inv_Δn = one(FT) / max(Δn, FT(1.0e-30))
                dρdn = dρ * inv_Δn
                dudn = du * inv_Δn; dvdn = dv * inv_Δn; dwdn = dw * inv_Δn
                dpdn = dp * inv_Δn
                du_ndn = dudn*nx + dvdn*ny + dwdn*nz
                du_ldn = dudn*lx + dvdn*ly + dwdn*lz
                du_mdn = dudn*mx + dvdn*my + dwdn*mz

                sigma = bcp[8]; L_ref = bcp[9]
                if sigma <= zero(FT); sigma = FT(0.3); end
                if L_ref <= zero(FT); L_ref = inv_Δn; end  # sensible default = 1/Δn (cell-length scale, avoids dimensionless L_ref=1)

                L5 = (u_n + c) * (dpdn + ρ_b * c * du_ndn)   # outgoing acoustic
                L2 = u_n * (c*c * dρdn - dpdn)               # entropy
                L3 = u_n * du_ldn                            # shear (l)
                L4 = u_n * du_mdn                            # shear (m)

                # L1: incoming acoustic. Weak anchor toward block outlet area-averaged
                # p_anchor (bcp[BCP_OUTLET_PAVG]=[15]). When p_anchor<=0 (uninitialized),
                # pure non-reflecting L1=0. Backflow (u_n<=0) strengthens the anchor.
                p_anchor = bcp[BCP_OUTLET_PAVG]
                if p_anchor > zero(FT)
                    backflow = u_n <= zero(FT)
                    if backflow
                        # HEURISTIC soft inflow (NOT derived from LODI):
                        # Poinsot-Lele §III.D gives a strictly LODI-consistent backflow treatment
                        # via the L5 incoming acoustic wave (u_n+c<0 reverses roles). Here we use a
                        # practical engineering approximation instead — strengthen the L1 pressure
                        # anchor (no (1-Ma²) damping) and damp the tangential wave amplitudes —
                        # which empirically pushes the flow back toward outflow without letting
                        # disturbance reflect as a hard wall zero-gradient would.
                        L1 = FT(1.0) * c * (p_b - p_anchor)
                        # damp tangential perturbations to suppress reflective feedback
                        L3 = -abs(u_n) * du_ldn
                        L4 = -abs(u_n) * du_mdn
                    else
                        L1 = sigma * c * (one(FT) - Ma_n*Ma_n) / L_ref * (p_b - p_anchor)
                    end
                else
                    L1 = zero(FT)
                end

                # LODI primitive rates (time derivatives)
                ρ_rate = -(L2 + FT(0.5)*(L5 + L1)) / (c*c + FT(1.0e-30))
                un_rate = -FT(0.5)*(L5 - L1) / (ρ_b*c + FT(1.0e-30))
                ul_rate = -L3
                um_rate = -L4
                p_rate = -FT(0.5)*(L5 + L1)

                # Limit single ghost update for robustness.
                rel_lim = FT(0.20); vel_lim = FT(0.25) * c
                dρ_step = max(-rel_lim*ρ_b, min(rel_lim*ρ_b, dt_bc*ρ_rate))
                dp_step = max(-rel_lim*p_b, min(rel_lim*p_b, dt_bc*p_rate))
                dun_step = max(-vel_lim, min(vel_lim, dt_bc*un_rate))
                dul_step = max(-vel_lim, min(vel_lim, dt_bc*ul_rate))
                dum_step = max(-vel_lim, min(vel_lim, dt_bc*um_rate))

                ρ_g = max(ρ_b + dρ_step, max(FT(1.0e-10), FT(0.05)*ρ_b))
                p_g = max(p_b + dp_step, max(FT(1.0e-10), FT(0.05)*p_b))
                un_g = u_n + dun_step
                ul_g = u_l + dul_step
                um_g = u_m + dum_step

                u_g = un_g*nx + ul_g*lx + um_g*mx
                v_g = un_g*ny + ul_g*ly + um_g*my
                w_g = un_g*nz + ul_g*lz + um_g*mz
                T_g = p_g / (ρ_g * Rg + FT(1.0e-30))

                Q[i,j,k,1]=ρ_g; Q[i,j,k,2]=u_g; Q[i,j,k,3]=v_g; Q[i,j,k,4]=w_g; Q[i,j,k,5]=p_g; Q[i,j,k,6]=T_g
                U[i,j,k,1]=ρ_g; U[i,j,k,2]=ρ_g*u_g; U[i,j,k,3]=ρ_g*v_g; U[i,j,k,4]=ρ_g*w_g
                U[i,j,k,5]=boundary_hydro_energy_density(ρ_g,u_g,v_g,w_g,p_g)
            end
        end

    # ── Riemann invariant outflow (self-consistent with internal Riemann solver) ──
    # Ghost state is reconstructed from extrapated Riemann invariants R±, entropy S,
    # and tangential ul,um. Because the ghost is built FROM invariants the internal
    # WENO/Riemann solver naturally re-derives the same wave structure on the boundary
    # face — no double characteristic decomposition mismatch (the slow-drift root cause
    # of the ghost-cell NSCBC + internal Riemann solver combination).
    #
    # First version (this branch): pure extrapolation, R⁻ also extrapolated (fully
    # non-reflecting, NO back-pressure anchor). The optional α slot BCP_RIEMANN_ALPHA
    # may later relax R⁻ toward an outlet-area-averaged target — but defaults to 0.
    elseif bc_type == Int32(BC_RIEMANN_OUTFLOW)
        @inbounds begin
            # Read boundary cell primitives and outward face normal.
            if dir == Int32(1)
                ρ_b=Q[idx_bnd,j,k,1]; u_b=Q[idx_bnd,j,k,2]; v_b=Q[idx_bnd,j,k,3]; w_b=Q[idx_bnd,j,k,4]; p_b=Q[idx_bnd,j,k,5]
                nx0=n_x[idx_wall,j,k]; ny0=n_y[idx_wall,j,k]; nz0=n_z[idx_wall,j,k]
            elseif dir == Int32(2)
                ρ_b=Q[i,idx_bnd,k,1]; u_b=Q[i,idx_bnd,k,2]; v_b=Q[i,idx_bnd,k,3]; w_b=Q[i,idx_bnd,k,4]; p_b=Q[i,idx_bnd,k,5]
                nx0=n_x[i,idx_wall,k]; ny0=n_y[i,idx_wall,k]; nz0=n_z[i,idx_wall,k]
            else
                ρ_b=Q[i,j,idx_bnd,1]; u_b=Q[i,j,idx_bnd,2]; v_b=Q[i,j,idx_bnd,3]; w_b=Q[i,j,idx_bnd,4]; p_b=Q[i,j,idx_bnd,5]
                nx0=n_x[i,j,idx_wall]; ny0=n_y[i,j,idx_wall]; nz0=n_z[i,j,idx_wall]
            end

            # Outward normal (mirror convention of NSCBC branch).
            s_out = side == Int32(1) ? one(FT) : -one(FT)
            nx = s_out*nx0; ny = s_out*ny0; nz = s_out*nz0
            nmag = sqrt(nx*nx + ny*ny + nz*nz + FT(1.0e-30))
            nx /= nmag; ny /= nmag; nz /= nmag

            # Orthonormal tangential basis (l, m) from outward normal.
            ax = abs(nx) < FT(0.9) ? one(FT) : zero(FT)
            ay = abs(nx) < FT(0.9) ? zero(FT) : one(FT)
            az = zero(FT)
            lx = ay*nz - az*ny; ly = az*nx - ax*nz; lz = ax*ny - ay*nx
            lmag = sqrt(lx*lx + ly*ly + lz*lz + FT(1.0e-30))
            lx /= lmag; ly /= lmag; lz /= lmag
            mx = ny*lz - nz*ly; my = nz*lx - nx*lz; mz = nx*ly - ny*lx

            # Boundary-cell invariants (zeroth-order extrapolation = ghost = boundary).
            ρ_b = max(ρ_b, FT(1.0e-30)); p_b = max(p_b, FT(1.0e-30))
            c_b = sqrt(γ * p_b / ρ_b)
            u_n = u_b*nx + v_b*ny + w_b*nz
            u_l = u_b*lx + v_b*ly + w_b*lz
            u_m = u_b*mx + v_b*my + w_b*mz
            Kgm = γ - one(FT)
            inv_Kgm = one(FT) / max(Kgm, FT(1.0e-30))
            S_b = p_b / (ρ_b^γ)            # entropy S = p/ρ^γ
            Rp_b = u_n + FT(2.0)*c_b*inv_Kgm    # outgoing acoustic invariant
            Rm_b = u_n - FT(2.0)*c_b*inv_Kgm    # incoming acoustic invariant

            # Ghost invariants. First version: pure extrapolation (no anchor).
            Rp_g = Rp_b
            S_g = S_b
            ul_g = u_l
            um_g = u_m

            # Optional soft R⁻ anchor (disabled by default; α=0 passes through).
            alpha = bcp[BCP_RIEMANN_ALPHA]
            if alpha > zero(FT)
                p_anchor = bcp[BCP_OUTLET_PAVG]
                if p_anchor > zero(FT)
                    # Target inflow invariant from outlet mean state.
                    # ρ_target = (p_anchor / S_g)^(1/γ)  -- use extrapolated entropy S_g
                    # c_target = sqrt(γ p_target/ρ_target); Rm_target = u_n - 2 c_target/(γ-1)
                    Sg = max(S_g, FT(1.0e-30))
                    ρ_target = exp((log(max(p_anchor, FT(1.0e-30))) - log(Sg)) / γ)
                    ρ_target = max(ρ_target, FT(1.0e-10))
                    c_target = sqrt(γ * max(p_anchor, FT(1.0e-30)) / max(ρ_target, FT(1.0e-30)))
                    Rm_target = u_n - FT(2.0)*c_target*inv_Kgm
                    Rm_g = (one(FT) - alpha) * Rm_b + alpha * Rm_target
                else
                    Rm_g = Rm_b
                end
            else
                Rm_g = Rm_b
            end

            # Reconstruct primitive ghost state from invariants.
            u_n_g = FT(0.5) * (Rp_g + Rm_g)
            c_g = FT(0.25) * (Rp_g - Rm_g) * Kgm
            c_g = max(c_g, FT(1.0e-30))
            # From S = p/ρ^γ and p = ρ c^2 / γ ⇒ ρ^γ = p/s / (γS) ⇒ ρ=(c²/(γS))^(1/(γ-1))
            c2_over_γS = (c_g * c_g) / (γ * max(S_g, FT(1.0e-30)))
            ρ_g = exp(log(max(c2_over_γS, FT(1.0e-30))) * inv_Kgm)
            ρ_g = max(ρ_g, FT(1.0e-10))
            p_g = ρ_g * c_g * c_g / γ
            p_g = max(p_g, FT(1.0e-10))

            # Tangential stays extrapolated. Reconstruct Cartesian velocity.
            u_g = u_n_g*nx + ul_g*lx + um_g*mx
            v_g = u_n_g*ny + ul_g*ly + um_g*my
            w_g = u_n_g*nz + ul_g*lz + um_g*mz
            T_g = p_g / (ρ_g * Rg + FT(1.0e-30))

            Q[i,j,k,1]=ρ_g; Q[i,j,k,2]=u_g; Q[i,j,k,3]=v_g; Q[i,j,k,4]=w_g; Q[i,j,k,5]=p_g; Q[i,j,k,6]=T_g
            U[i,j,k,1]=ρ_g; U[i,j,k,2]=ρ_g*u_g; U[i,j,k,3]=ρ_g*v_g; U[i,j,k,4]=ρ_g*w_g
            U[i,j,k,5]=boundary_hydro_energy_density(ρ_g,u_g,v_g,w_g,p_g)
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
            U[i,j,k,5]=boundary_hydro_energy_density(ρg,ug,vg,wg,pg)
        end

    end

    # ── MHD extension: set B-field and ψ for ghost cells ──
    # Applied AFTER the hydrodynamic BC above has set ρ, u, v, w, p, T
    if equation_type == :MHD
        if ct_mode
            # Staggered face flux is the sole magnetic owner in CT. Leave
            # Q[B] untouched here; the CT state transaction restores it only
            # after all physical, rank, and inter-block face halos are final.
            @inbounds Q[i,j,k,QPSI] = zero(FT)
            return
        end
        if bc_type == Int32(BC_ISOTHERMAL_WALL) || bc_type == Int32(BC_ADIABATIC_WALL) || bc_type == Int32(BC_MHD_WALL) || bc_type == Int32(BC_SLIP_WALL) || bc_type == Int32(BC_SYMMETRY)
            # Perfectly conducting wall: Bn anti-symmetric, Bt symmetric.
            if dir == Int32(1)
                @inbounds begin
                    bx,by,bz=mhd_reflect_wall_field(
                        Q[idx_int,j,k,7],Q[idx_int,j,k,8],Q[idx_int,j,k,9],
                        n_x[idx_wall,j,k],n_y[idx_wall,j,k],n_z[idx_wall,j,k],false)
                    Q[i,j,k,7]=bx; Q[i,j,k,8]=by; Q[i,j,k,9]=bz
                    @static if ct_mode
                        Q[i,j,k,10]=zero(FT)
                    else
                        Q[i,j,k,10]=-Q[idx_int,j,k,10]
                    end
                end
            elseif dir == Int32(2)
                @inbounds begin
                    bx,by,bz=mhd_reflect_wall_field(
                        Q[i,idx_int,k,7],Q[i,idx_int,k,8],Q[i,idx_int,k,9],
                        n_x[i,idx_wall,k],n_y[i,idx_wall,k],n_z[i,idx_wall,k],false)
                    Q[i,j,k,7]=bx; Q[i,j,k,8]=by; Q[i,j,k,9]=bz
                    @static if ct_mode
                        Q[i,j,k,10]=zero(FT)
                    else
                        Q[i,j,k,10]=-Q[i,idx_int,k,10]
                    end
                end
            else
                @inbounds begin
                    bx,by,bz=mhd_reflect_wall_field(
                        Q[i,j,idx_int,7],Q[i,j,idx_int,8],Q[i,j,idx_int,9],
                        n_x[i,j,idx_wall],n_y[i,j,idx_wall],n_z[i,j,idx_wall],false)
                    Q[i,j,k,7]=bx; Q[i,j,k,8]=by; Q[i,j,k,9]=bz
                    @static if ct_mode
                        Q[i,j,k,10]=zero(FT)
                    else
                        Q[i,j,k,10]=-Q[i,j,idx_int,10]
                    end
                end
            end
            # Update conservative B and ψ, recompute energy with B²/(2*mu0)
            @inbounds begin
                Bx=Q[i,j,k,7]; By=Q[i,j,k,8]; Bz=Q[i,j,k,9]; ψv=Q[i,j,k,10]
                @static if !ct_mode
                    U[i,j,k,6]=Bx; U[i,j,k,7]=By; U[i,j,k,8]=Bz; U[i,j,k,9]=ψv
                end
                B2=Bx*Bx+By*By+Bz*Bz
                U[i,j,k,5]=mhd_energy_density_from_primitive(
                    Q[i,j,k,1], Q[i,j,k,2], Q[i,j,k,3], Q[i,j,k,4],
                    Q[i,j,k,5], Bx, By, Bz, γ,
                )
            end
        elseif bc_type == Int32(BC_MHD_INSULATING_WALL)
            # Local insulating model: Bt anti-symmetric, Bn symmetric.
            if dir == Int32(1)
                @inbounds begin
                    bx,by,bz=mhd_reflect_wall_field(
                        Q[idx_int,j,k,7],Q[idx_int,j,k,8],Q[idx_int,j,k,9],
                        n_x[idx_wall,j,k],n_y[idx_wall,j,k],n_z[idx_wall,j,k],true)
                    Q[i,j,k,7]=bx; Q[i,j,k,8]=by; Q[i,j,k,9]=bz
                    @static if ct_mode
                        Q[i,j,k,10]=zero(FT)
                    else
                        Q[i,j,k,10]=-Q[idx_int,j,k,10]
                    end
                end
            elseif dir == Int32(2)
                @inbounds begin
                    bx,by,bz=mhd_reflect_wall_field(
                        Q[i,idx_int,k,7],Q[i,idx_int,k,8],Q[i,idx_int,k,9],
                        n_x[i,idx_wall,k],n_y[i,idx_wall,k],n_z[i,idx_wall,k],true)
                    Q[i,j,k,7]=bx; Q[i,j,k,8]=by; Q[i,j,k,9]=bz
                    @static if ct_mode
                        Q[i,j,k,10]=zero(FT)
                    else
                        Q[i,j,k,10]=-Q[i,idx_int,k,10]
                    end
                end
            else
                @inbounds begin
                    bx,by,bz=mhd_reflect_wall_field(
                        Q[i,j,idx_int,7],Q[i,j,idx_int,8],Q[i,j,idx_int,9],
                        n_x[i,j,idx_wall],n_y[i,j,idx_wall],n_z[i,j,idx_wall],true)
                    Q[i,j,k,7]=bx; Q[i,j,k,8]=by; Q[i,j,k,9]=bz
                    @static if ct_mode
                        Q[i,j,k,10]=zero(FT)
                    else
                        Q[i,j,k,10]=-Q[i,j,idx_int,10]
                    end
                end
            end
            # Update conservative B and ψ, recompute energy with B²/(2*mu0)
            @inbounds begin
                Bx=Q[i,j,k,7]; By=Q[i,j,k,8]; Bz=Q[i,j,k,9]; ψv=Q[i,j,k,10]
                @static if !ct_mode
                    U[i,j,k,6]=Bx; U[i,j,k,7]=By; U[i,j,k,8]=Bz; U[i,j,k,9]=ψv
                end
                B2=Bx*Bx+By*By+Bz*Bz
                U[i,j,k,5]=mhd_energy_density_from_primitive(
                    Q[i,j,k,1], Q[i,j,k,2], Q[i,j,k,3], Q[i,j,k,4],
                    Q[i,j,k,5], Bx, By, Bz, γ,
                )
                if isothermal_mhd
                    U[i,j,k,5]=mhd_energy_density_from_primitive(
                        Q[i,j,k,1], Q[i,j,k,2], Q[i,j,k,3], Q[i,j,k,4],
                        Q[i,j,k,5], Bx, By, Bz, γ,
                    )
                end
            end
        elseif bc_type == Int32(BC_SUPERSONIC_INFLOW) || bc_type == Int32(BC_MHD_INFLOW)
            # Fixed B-field from BC parameters
            @inbounds begin
                Bx_inf=bcp[BCP_BX_INF]; By_inf=bcp[BCP_BY_INF]; Bz_inf=bcp[BCP_BZ_INF]
                Q[i,j,k,7]=Bx_inf; Q[i,j,k,8]=By_inf; Q[i,j,k,9]=Bz_inf; Q[i,j,k,10]=zero(FT)
                @static if !ct_mode
                    U[i,j,k,6]=Bx_inf; U[i,j,k,7]=By_inf; U[i,j,k,8]=Bz_inf; U[i,j,k,9]=zero(FT)
                end
                B2=Bx_inf^2+By_inf^2+Bz_inf^2
                U[i,j,k,5]=mhd_energy_density_from_primitive(
                    Q[i,j,k,1], Q[i,j,k,2], Q[i,j,k,3], Q[i,j,k,4],
                    Q[i,j,k,5], Bx_inf, By_inf, Bz_inf, γ,
                )
            end
        elseif bc_type == Int32(BC_MHD_EXTERNAL_FIELD) ||
               bc_type == Int32(BC_MHD_RESERVOIR_INFLOW) ||
               bc_type == Int32(BC_MHD_PROFILED_INFLOW) ||
               bc_type == Int32(BC_MHD_FIXED_EXTERNAL_FIELD) ||
               bc_type == Int32(BC_MHD_OUTFLOW_EXTERNAL_FIELD)
            @inbounds begin
                Bx, By, Bz = boundary_external_magnetic_field_components(
                    external_field_cache,
                    bcp,
                    x, y, z, i, j, k,
                )
                Q[i,j,k,7]=Bx; Q[i,j,k,8]=By; Q[i,j,k,9]=Bz; Q[i,j,k,10]=zero(FT)
                @static if !ct_mode
                    U[i,j,k,6]=Bx; U[i,j,k,7]=By; U[i,j,k,8]=Bz; U[i,j,k,9]=zero(FT)
                end
                B2=Bx^2+By^2+Bz^2
                U[i,j,k,5]=mhd_energy_density_from_primitive(
                    Q[i,j,k,1], Q[i,j,k,2], Q[i,j,k,3], Q[i,j,k,4],
                    Q[i,j,k,5], Bx, By, Bz, γ,
                )
            end
        end
        if isothermal_mhd && (
            bc_type == Int32(BC_MHD_EXTERNAL_FIELD) ||
            bc_type == Int32(BC_MHD_PROFILED_INFLOW) ||
            bc_type == Int32(BC_MHD_FIXED_EXTERNAL_FIELD) ||
            bc_type == Int32(BC_MHD_OUTFLOW_EXTERNAL_FIELD)
        )
            Bx_now=Q[i,j,k,7]; By_now=Q[i,j,k,8]; Bz_now=Q[i,j,k,9]
            U[i,j,k,5]=mhd_energy_density_from_primitive(
                Q[i,j,k,1], Q[i,j,k,2], Q[i,j,k,3], Q[i,j,k,4],
                Q[i,j,k,5], Bx_now, By_now, Bz_now,
            )
        end
        # zero_gradient / supersonic_outflow / slip_wall / symmetry:
        # Already handled by the generic primitive/conservative copy loops.
    end

    return
end

# =============================================================================
# ξ-direction BCs — GPU kernel
# =============================================================================
function fill_x(Q, U, rx, n_x, n_y, n_z, Area, Vol, dt_bc, x, y, z,
                bc_x_lo, bc_x_hi, bcp_x_lo, bcp_x_hi,
                nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params,
                external_field_cache)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG || i < 1 || j < 1 || k < 1; return; end

    # ξ- ghost cells
    if rx == 0 && i <= NG
        _apply_bc!(Q, U, i, j, k, bc_x_lo, Int32(1), Int32(0),
                   n_x, n_y, n_z, Area, Vol, dt_bc, x, y, z, nxp, nyp, nzp, bcp_x_lo, tt, dsrfg_params,
                   external_field_cache)
    # ξ+ ghost cells
    elseif rx == Nprocs_block[1]-1 && i > nxp+NG
        _apply_bc!(Q, U, i, j, k, bc_x_hi, Int32(1), Int32(1),
                   n_x, n_y, n_z, Area, Vol, dt_bc, x, y, z, nxp, nyp, nzp, bcp_x_hi, tt, dsrfg_params,
                   external_field_cache)
    end
    return
end

# =============================================================================
# η-direction BCs — GPU kernel
# =============================================================================
function fill_y(Q, U, rx, ry, rz, n_x_j, n_y_j, n_z_j, Area, Vol, dt_bc, x, y, z,
                bc_y_lo, bc_y_hi, bcp_y_lo, bcp_y_hi,
                nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params,
                external_field_cache)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG || i < 1 || j < 1 || k < 1; return; end

    # η- ghost
    if ry == 0 && j <= NG
        _apply_bc!(Q, U, i, j, k, bc_y_lo, Int32(2), Int32(0),
                   n_x_j, n_y_j, n_z_j, Area, Vol, dt_bc, x, y, z, nxp, nyp, nzp, bcp_y_lo, tt, dsrfg_params,
                   external_field_cache)
    # η+ ghost
    elseif ry == Nprocs_block[2]-1 && j > nyp+NG
        _apply_bc!(Q, U, i, j, k, bc_y_hi, Int32(2), Int32(1),
                   n_x_j, n_y_j, n_z_j, Area, Vol, dt_bc, x, y, z, nxp, nyp, nzp, bcp_y_hi, tt, dsrfg_params,
                   external_field_cache)
    end
    return
end

# =============================================================================
# ζ-direction BCs — GPU kernel
# =============================================================================
function fill_z(Q, U, rx, ry, rz, n_x_k, n_y_k, n_z_k, Area, Vol, dt_bc, x, y, z,
                bc_z_lo, bc_z_hi, bcp_z_lo, bcp_z_hi,
                nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params,
                external_field_cache)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG || i < 1 || j < 1 || k < 1; return; end

    # ζ- ghost
    if rz == 0 && k <= NG
        _apply_bc!(Q, U, i, j, k, bc_z_lo, Int32(3), Int32(0),
                   n_x_k, n_y_k, n_z_k, Area, Vol, dt_bc, x, y, z, nxp, nyp, nzp, bcp_z_lo, tt, dsrfg_params,
                   external_field_cache)
    # ζ+ ghost
    elseif rz == Nprocs_block[3]-1 && k > nzp+NG
        _apply_bc!(Q, U, i, j, k, bc_z_hi, Int32(3), Int32(1),
                   n_x_k, n_y_k, n_z_k, Area, Vol, dt_bc, x, y, z, nxp, nyp, nzp, bcp_z_hi, tt, dsrfg_params,
                   external_field_cache)
    end
    return
end

@inline function _ct_is_physical_boundary_type(boundary_type)
    return boundary_type != Int32(BC_INTERBLOCK) &&
           boundary_type != Int32(BC_PERIODIC)
end

# Physical boundary kernels construct gas states before staggered face-B halos
# are final. Close U[5] only after Q[B] has been recovered from that face field.
# Interface and MPI ghosts retain the conservative value received from peers.
function finalize_ct_physical_ghost_energy_kernel!(
    U, Q, rx, ry, rz,
    bc_x_lo, bc_x_hi, bc_y_lo, bc_y_hi, bc_z_lo, bc_z_hi,
    nxp, nyp, nzp, Nprocs_block,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp + 2*NG || j > nyp + 2*NG || k > nzp + 2*NG
        return
    end

    physical_x = (
        rx == 0 && i <= NG && _ct_is_physical_boundary_type(bc_x_lo)
    ) || (
        rx == Nprocs_block[1]-1 && i > nxp+NG &&
        _ct_is_physical_boundary_type(bc_x_hi)
    )
    physical_y = (
        ry == 0 && j <= NG && _ct_is_physical_boundary_type(bc_y_lo)
    ) || (
        ry == Nprocs_block[2]-1 && j > nyp+NG &&
        _ct_is_physical_boundary_type(bc_y_hi)
    )
    physical_z = (
        rz == 0 && k <= NG && _ct_is_physical_boundary_type(bc_z_lo)
    ) || (
        rz == Nprocs_block[3]-1 && k > nzp+NG &&
        _ct_is_physical_boundary_type(bc_z_hi)
    )
    !(physical_x || physical_y || physical_z) && return

    @inbounds U[i,j,k,5] = mhd_energy_density_from_primitive(
        Q[i,j,k,1], Q[i,j,k,2], Q[i,j,k,3], Q[i,j,k,4],
        Q[i,j,k,5], Q[i,j,k,QBX], Q[i,j,k,QBY], Q[i,j,k,QBZ],
    )
    return
end

function finalize_structured_ct_physical_ghost_energy!(
    U, Q, rx, ry, rz, block_id, nxp, nyp, nzp, Nprocs_block,
    face_bc, bc_params,
)
    boundary_types, _ = _structured_face_boundary_data(
        face_bc, bc_params, block_id,
    )
    nb = (
        cld(nxp+2*NG, nthreads[1]), cld(nyp+2*NG, nthreads[2]),
        cld(nzp+2*NG, nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=nb finalize_ct_physical_ghost_energy_kernel!(
        U, Q, rx, ry, rz, boundary_types...,
        Int32(nxp), Int32(nyp), Int32(nzp), Nprocs_block,
    )
    return nothing
end

# =============================================================================
# Structured ghost-cell launch orchestration
# =============================================================================
function _structured_face_boundary_data(face_bc,bc_params,block_id)
    boundary_types=ntuple(6) do face_id
        Int32(get(face_bc,(block_id,face_id),BC_INTERBLOCK))
    end
    empty_parameters=ntuple(_ -> zero(FT),Val(N_BC_PARAMS))
    boundary_parameters=ntuple(6) do face_id
        get(bc_params,(block_id,face_id),empty_parameters)
    end
    return boundary_types,boundary_parameters
end

function fill_structured_ghost_cells!(Q, U, rx, ry, rz,
                   n_x_i, n_y_i, n_z_i,
                   n_x_j, n_y_j, n_z_j,
                   n_x_k, n_y_k, n_z_k,
                   Area_i, Area_j, Area_k, Vol, dt_bc,
                   x, y, z, block_id, nxp, nyp, nzp, Nprocs_block, tt, face_bc, bc_params,
                   dsrfg_params, external_field_cache=nothing)
    nb = (cld(nxp+2*NG, nthreads[1]), cld(nyp+2*NG, nthreads[2]), cld(nzp+2*NG, nthreads[3]))

    boundary_types,boundary_parameters=
        _structured_face_boundary_data(face_bc,bc_params,block_id)
    bc_x_lo,bc_x_hi,bc_y_lo,bc_y_hi,bc_z_lo,bc_z_hi=boundary_types
    bcp_x_lo,bcp_x_hi,bcp_y_lo,bcp_y_hi,bcp_z_lo,bcp_z_hi=boundary_parameters

    @gpu_launch threads=nthreads blocks=nb fill_x(Q, U, rx, n_x_i, n_y_i, n_z_i,
        Area_i, Vol, dt_bc, x, y, z, bc_x_lo, bc_x_hi, bcp_x_lo, bcp_x_hi, nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params,
        external_field_cache)
    @gpu_launch threads=nthreads blocks=nb fill_y(Q, U, rx, ry, rz, n_x_j, n_y_j, n_z_j,
        Area_j, Vol, dt_bc, x, y, z, bc_y_lo, bc_y_hi, bcp_y_lo, bcp_y_hi, nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params,
        external_field_cache)
    @gpu_launch threads=nthreads blocks=nb fill_z(Q, U, rx, ry, rz, n_x_k, n_y_k, n_z_k,
        Area_k, Vol, dt_bc, x, y, z, bc_z_lo, bc_z_hi, bcp_z_lo, bcp_z_hi, nxp, nyp, nzp, Nprocs_block, tt, dsrfg_params,
        external_field_cache)
    return nothing
end



include(joinpath(@__DIR__,"initial_conditions.jl"))
