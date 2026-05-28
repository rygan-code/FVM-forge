using Random
using Adapt

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
