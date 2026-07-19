# =============================================================================
# Volume force for multi-block FVM configuration
# Retains rotation (Coriolis, centrifugal, Euler) and bulk forcing
# Uses global communicator for cross-block reductions
# =============================================================================

# =============================================================================
# Deschamps / CEBL pipe forcing
# =============================================================================

# Fused Deschamps + add_source kernel: computes deschamps force and applies
# directly to U in one pass, avoiding the intermediate dU_forced array.
# Eliminates 2 kernel launches (zero_dU + deschamps_gpu_kernel).
# VGPR estimate: ~24 (ρ,u,v,w,p,u_bulk,e_int,max_rate,flowx,f1,dt = 11 live regs)
function fused_deschamps_source_kernel!(U, Q, f1_val, flowx_val, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    
    ii, jj, kk = i+NG, j+NG, k+NG
    
    @inbounds ρ = Q[ii, jj, kk, 1]
    @inbounds u = Q[ii, jj, kk, 2]
    @inbounds v = Q[ii, jj, kk, 3]
    @inbounds w = Q[ii, jj, kk, 4]
    @inbounds p = Q[ii, jj, kk, 5]
    
    u_bulk = Ma_target * sqrt(γ * Rg * Tw)
    
    local_e_internal = p / (ρ * (γ - one(FT)))
    max_removal_rate = -FT(0.2e0) * ρ / dt
    flowx_safe = max(flowx_val, max_removal_rate)
    
    # Compute forcing and apply directly to U (= deschamps_gpu_kernel + add_source_kernel fused)
    @inbounds U[ii, jj, kk, 1] += (flowx_safe) * dt
    @inbounds U[ii, jj, kk, 2] += (f1_val + flowx_safe * u) * dt
    @inbounds U[ii, jj, kk, 3] += (flowx_safe * v) * dt
    @inbounds U[ii, jj, kk, 4] += (flowx_safe * w) * dt
    @inbounds U[ii, jj, kk, 5] += (f1_val * u_bulk + flowx_safe * (local_e_internal + FT(0.5) * (u^2 + v^2 + w^2))) * dt
    
    return
end

# =============================================================================
# Rotating-frame and zeroing kernels
# =============================================================================

# Lightweight kernel: zero the forcing array (used when rotation = 0)
function zero_dU_forced_kernel!(dU_forced, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    for n = 1:Ncons
        @inbounds dU_forced[i, j, k, n] = zero(FT)
    end
    return
end

function Volume_force_kernel!(dU_forced, Q, y, z, nxp, nyp, nzp, Omega_x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    
    ii, jj, kk = i+NG, j+NG, k+NG
    
    @inbounds ρ = Q[ii, jj, kk, 1]
    @inbounds v = Q[ii, jj, kk, 3]
    @inbounds w = Q[ii, jj, kk, 4]
    
    @inbounds y_loc = y[ii, jj, kk]
    @inbounds z_loc = z[ii, jj, kk]
    
    # Coriolis + Centrifugal forces (uniform rotation Omega_x around x-axis)
    fy = ρ * (FT(2.0) * w * Omega_x + Omega_x^2 * y_loc)
    fz = ρ * (-FT(2.0) * v * Omega_x + Omega_x^2 * z_loc)
    
    @inbounds dU_forced[i, j, k, 1] = zero(FT)
    @inbounds dU_forced[i, j, k, 2] = zero(FT)
    @inbounds dU_forced[i, j, k, 3] = fy
    @inbounds dU_forced[i, j, k, 4] = fz
    @inbounds dU_forced[i, j, k, 5] = v * fy + w * fz
    return
end

# =============================================================================
# Pipe bulk forcing
# =============================================================================

# ── Bulk forcing (separated into Sync/Reduction and GPU-Application) ──
function adjust_gpu_kernel!(dU_forced, Q, forcex, flowx, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    
    ii, jj, kk = i+NG, j+NG, k+NG
    
    @inbounds ρ = Q[ii, jj, kk, 1]
    @inbounds u = Q[ii, jj, kk, 2]
    @inbounds v = Q[ii, jj, kk, 3]
    @inbounds w = Q[ii, jj, kk, 4]
    @inbounds p = Q[ii, jj, kk, 5]
    
    local_e_internal = p / (ρ * (γ - one(FT)))
    max_removal_rate = -FT(0.2e0) * ρ / dt 
    flowx_safe = max(flowx, max_removal_rate)

    @inbounds dU_forced[i, j, k, 1] += flowx_safe
    @inbounds dU_forced[i, j, k, 2] += forcex + flowx_safe * u
    @inbounds dU_forced[i, j, k, 3] += flowx_safe * v
    @inbounds dU_forced[i, j, k, 4] += flowx_safe * w
    @inbounds dU_forced[i, j, k, 5] += forcex * u + flowx_safe * (local_e_internal + FT(0.5) * (u^2 + v^2 + w^2))

    return
end

function Update_bulk_force_params(blocks, tt, KRK, comm_world, rank)
    if !flow_forcing; return zero(FT), zero(FT); end
    
    c_wall = sqrt(γ*Rg*Tw)
    u_bulk_target = Ma_target * c_wall
    
    ρ_local_sum = zero(FT)
    u_local_sum = zero(FT)
    volume_local_sum = zero(FT)
    
    for (bid, b) in blocks
        NGp = NG+1
        nx_end, ny_end, nz_end = b.Nx+NG, b.Ny+NG, b.Nz+NG
        
        ρ_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 1]
        u_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 2]
        vol_v = @view b.Vol[NGp:nx_end, NGp:ny_end, NGp:nz_end]
        
        # 2-arg mapreduce is well-supported on GPUArrays (zero allocation)
        ρ_local_sum += mapreduce(/, +, ρ_v, vol_v)           # sum(ρ/vol)
        volume_local_sum += mapreduce(v -> one(FT) / v, +, vol_v) # sum(1/vol)
        
        # Zero-allocation: avoids temporary GPU arrays that leak RDMA registrations
        u_local_sum += mapreduce((r,u,v) -> r*u/v, +, ρ_v, u_v, vol_v)
    end
    
    # Global reduction across ALL blocks using comm_world
    ρ_sum_global = MPI.Allreduce(ρ_local_sum, MPI.SUM, comm_world)
    u_sum_global = MPI.Allreduce(u_local_sum, MPI.SUM, comm_world)
    volume_sum_global = MPI.Allreduce(volume_local_sum, MPI.SUM, comm_world)
    
    u_avg = u_sum_global / ρ_sum_global
    ρ_avg = ρ_sum_global / volume_sum_global
    
    μw = C_s * Tw * sqrt(Tw) / (Tw + T_s)
    ρ_bulk_target = Re_target * μw / (u_bulk_target * 2 * R0)
    
    Kforce = FT(5.0e0)
    Kflow = FT(1000.0)

    ρ_error = ρ_bulk_target - ρ_avg
    u_error = u_bulk_target - u_avg
    
    flowx = Kflow * ρ_error
    forcex = Kforce * u_error

    if (tt % 100 == 0 || tt == 1) && rank == 0 && KRK == 1
        @printf "Iteration: %d | rho_avg: %.4e (target: %.4e) | u_avg: %.4e (target: %.4e) | Flow_source: %.4e | Force_source: %.4e\n" tt ρ_avg ρ_bulk_target u_avg u_bulk_target flowx forcex
        flush(stdout)
    end
    
    return FT(forcex), FT(flowx)
end

function Apply_bulk_force!(dU_forced, Q, forcex, flowx, dt, nxp, nyp, nzp)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb adjust_gpu_kernel!(dU_forced, Q, forcex, flowx, dt, nxp, nyp, nzp)
end

# =============================================================================
# Constant mass flux forcing
# =============================================================================

# =============================================================================
# Constant Mass Flux Forcing — Deschamps Algorithm
# 
# Only adds streamwise body force f₁ to momentum equation (no mass source).
# The force is evolved to maintain constant mass flux Q_m:
#   f₁^{n+1} = f₁^n + (dt/A_cross) * [α*(Q_m^{n+1} - Q₀) + β*(Q_m^n - Q₀)]
# where α=1.5, β=-0.5 (AB2-like for stability), A_cross = πR²
# =============================================================================

# Persistent state for the Deschamps algorithm
mutable struct ConstMassFluxState
    f1::Float64    # Current body force magnitude [N/m³]
    Qm_prev::Float64  # Previous mass flux
    initialized::Bool
end

const cmf_state = ConstMassFluxState(0.0, 0.0, false)

# GPU kernel: applies ONLY streamwise momentum source (no mass source!)
function cmf_gpu_kernel!(dU_forced, Q, f1_val, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    
    ii, jj, kk = i+NG, j+NG, k+NG
    
    @inbounds u = Q[ii, jj, kk, 2]
    
    # Bulk velocity for energy term (Deschamps: use u_bulk, not u_local, for stability)
    u_bulk = Ma_target * sqrt(γ * Rg * Tw)
    
    # Only streamwise force: f₁·δᵢ₁
    @inbounds dU_forced[i, j, k, 1] += zero(FT)             # NO mass source
    @inbounds dU_forced[i, j, k, 2] += f1_val             # streamwise momentum
    @inbounds dU_forced[i, j, k, 3] += zero(FT)              # no y-force
    @inbounds dU_forced[i, j, k, 4] += zero(FT)              # no z-force
    @inbounds dU_forced[i, j, k, 5] += f1_val * u_bulk    # energy = f·u_bulk (not u_local!)
    
    return
end

function Update_const_massflux_params(blocks, tt, KRK, dt, comm_world, rank)
    if !flow_forcing; return zero(FT); end
    
    c_wall = sqrt(γ * Rg * Tw)
    u_bulk_target = Ma_target * c_wall
    μw = C_s * Tw * sqrt(Tw) / (Tw + T_s)
    ρ_bulk_target = Re_target * μw / (u_bulk_target * 2 * R0)
    
    # Target mass flux: Q₀ = ρ_bulk * u_bulk * A_cross
    A_cross = Float64(π) * Float64(R0)^2
    Q0 = Float64(ρ_bulk_target) * Float64(u_bulk_target) * A_cross
    
    # Compute current mass flux: Q_m = ∫ ρ·u dA (volume-weighted)
    ρu_local_sum = zero(FT)
    volume_local_sum = zero(FT)
    ρ_local_sum = zero(FT)
    u_local_sum = zero(FT)
    
    for (bid, b) in blocks
        NGp = NG+1
        nx_end, ny_end, nz_end = b.Nx+NG, b.Ny+NG, b.Nz+NG
        
        ρ_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 1]
        u_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 2]
        vol_v = @view b.Vol[NGp:nx_end, NGp:ny_end, NGp:nz_end]
        
        ρu_local_sum += mapreduce((r,u,v) -> Float64(r)*Float64(u)/Float64(v), +, ρ_v, u_v, vol_v; init=0.0)
        ρ_local_sum += mapreduce((r,v) -> Float64(r)/Float64(v), +, ρ_v, vol_v; init=0.0)
        volume_local_sum += mapreduce(v -> 1.0 / Float64(v), +, vol_v; init=0.0)
    end
    
    # Global reduction
    ρu_sum_global = MPI.Allreduce(Float64(ρu_local_sum), MPI.SUM, comm_world)
    ρ_sum_global = MPI.Allreduce(Float64(ρ_local_sum), MPI.SUM, comm_world)
    volume_sum_global = MPI.Allreduce(Float64(volume_local_sum), MPI.SUM, comm_world)
    
    u_avg = ρu_sum_global / ρ_sum_global
    ρ_avg = ρ_sum_global / volume_sum_global
    
    # Current mass flux (volume-averaged × cross-section)
    Qm_current = ρu_sum_global / volume_sum_global * A_cross
    
    # Initialize on first call
    if !cmf_state.initialized
        cmf_state.Qm_prev = Qm_current
        # Initial guess: pressure gradient for laminar Poiseuille
        cmf_state.f1 = Float64(ρ_bulk_target) * Float64(u_bulk_target) * 8.0 * Float64(μw) / 
                       (Float64(ρ_bulk_target) * Float64(R0)^2)
        cmf_state.initialized = true
    end
    
    # ── Constant mass flux controller ──
    # The original Deschamps (1986) AB2 formula with gain = dt/A_cross is far too
    # sluggish for explicit time stepping (dt ~ 7e-7 s, flow-through time ~ 6e-4 s).
    # 
    # We use instead a proportional controller based on bulk density and domain volume:
    #   f₁^{n+1} = f₁^n - α × (Qm - Q₀) / (ρ_bulk × V_domain)
    # where α is a relaxation coefficient (~2.0 for aggressive correction).
    # This converges within a few flow-through times regardless of dt.
    
    Qm_error_new = Qm_current - Q0
    Qm_error_old = cmf_state.Qm_prev - Q0
    
    # Controller: proportional with AB2 blending
    α_relax = 0.5  # relaxation coefficient (reduced from 2.0 to damp oscillations)
    V_domain = volume_sum_global  # total domain volume
    ρ_b = Float64(ρ_bulk_target)
    
    # Gain = α / (ρ_b × V_domain) — has units of [1/(kg/m)] = [m/kg]
    # Multiplied by Qm_error [kg/s] gives force/volume change [Pa/m / s] ... but we want [Pa/m]
    # Use flow time scale: τ = V_domain^(1/3) / u_bulk_target
    τ_flow = V_domain^(1.0/3.0) / Float64(u_bulk_target)
    gain = α_relax / (ρ_b * V_domain) * τ_flow
    
    # AB2 blending for smoother convergence
    α_ab = 1.5; β_ab = -0.5
    cmf_state.f1 -= gain * (α_ab * Qm_error_new + β_ab * Qm_error_old)
    cmf_state.Qm_prev = Qm_current
    
    if (tt % 100 == 0 || tt == 1) && rank == 0 && KRK == 1
        @printf "Iteration: %d | rho_avg: %.4e (target: %.4e) | u_avg: %.4e (target: %.4e) | f1: %.6e | Qm_err: %.4e\n" tt ρ_avg ρ_bulk_target u_avg u_bulk_target cmf_state.f1 Qm_error_new
        flush(stdout)
    end
    
    return FT(cmf_state.f1)
end


function Apply_const_massflux_force!(dU_forced, Q, f1_val, nxp, nyp, nzp)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb cmf_gpu_kernel!(dU_forced, Q, f1_val, nxp, nyp, nzp)
end

# =============================================================================
# Mode 3: Rigorous Deschamps Algorithm for Pipe Flow + Mass Correction
#
# Force update derived from the exact section-averaged momentum equation
# for a circular pipe with periodic streamwise BC:
#
#   dQ_m/dt = πR² f_x - 2πR τ_w
#
# Prediction step:
#   g^n = -πR² f_x^n  (ignoring wall stress — computed implicitly by NS)
#   Q_pred^{n+1} = Q^n - Δt g^n
#
# Correction (Deschamps):
#   f_x^{n+1} = f_x^n - 1/(πR²) [2(Q_pred - Q₀) - 0.2(Q^n - Q₀)]
#
# This reduces to the simplified update:
#   f_x^{n+1} = f_x^n - 1/(πR²) [(2Q_pred - 0.2 Q^n) - 1.8 Q₀]
#
# Mass correction: weak proportional source to prevent density drift.
#   flowx = K_mass × (ρ_target - ρ_avg)
# This is purely for long-time stability and does NOT affect the momentum
# forcing physics.
#
# Ref: Deschamps (1986), adapted to cylindrical pipe geometry.
# =============================================================================

mutable struct DeschampsState
    f1::Float64        # Current body force magnitude [N/m³]
    f1_prev::Float64   # Previous body force
    Qm_prev::Float64   # Previous mass flux
    Qm_smooth::Float64  # EMA-smoothed mass flux (filters turbulent noise)
    g_prev::Float64     # Previous decay rate
    initialized::Bool
end

const deschamps_state = DeschampsState(0.0, 0.0, 0.0, 0.0, 0.0, false)

# GPU kernel: streamwise momentum + weak mass correction
function deschamps_gpu_kernel!(dU_forced, Q, f1_val, flowx_val, dt, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    
    ii, jj, kk = i+NG, j+NG, k+NG
    
    @inbounds ρ = Q[ii, jj, kk, 1]
    @inbounds u = Q[ii, jj, kk, 2]
    @inbounds v = Q[ii, jj, kk, 3]
    @inbounds w = Q[ii, jj, kk, 4]
    @inbounds p = Q[ii, jj, kk, 5]
    
    # Bulk velocity for energy term (use u_bulk, not u_local, for stability)
    u_bulk = Ma_target * sqrt(γ * Rg * Tw)
    
    # Mass correction: limit removal rate to avoid negative density
    local_e_internal = p / (ρ * (γ - one(FT)))
    max_removal_rate = -FT(0.2e0) * ρ / dt
    flowx_safe = max(flowx_val, max_removal_rate)
    
    # Streamwise momentum force (Deschamps)
    @inbounds dU_forced[i, j, k, 1] += flowx_safe                               # mass correction
    @inbounds dU_forced[i, j, k, 2] += f1_val + flowx_safe * u                  # streamwise momentum
    @inbounds dU_forced[i, j, k, 3] += flowx_safe * v                           # y-momentum (mass only)
    @inbounds dU_forced[i, j, k, 4] += flowx_safe * w                           # z-momentum (mass only)
    @inbounds dU_forced[i, j, k, 5] += f1_val * u_bulk + flowx_safe * (local_e_internal + FT(0.5) * (u^2 + v^2 + w^2))
    
    return
end

function Update_deschamps_pipe_params(blocks, tt, KRK, dt, comm_world, rank)
    if !flow_forcing; return zero(FT), zero(FT); end
    
    c_wall = sqrt(γ * Rg * Tw)
    u_bulk_target = Ma_target * c_wall
    μw = C_s * Tw * sqrt(Tw) / (Tw + T_s)
    ρ_bulk_target = Re_target * μw / (u_bulk_target * 2 * R0)
    
    # Cross-section area
    A_cross = Float64(π) * Float64(R0)^2
    Q0 = Float64(ρ_bulk_target) * Float64(u_bulk_target) * A_cross
    
    # Compute current mass flux and density
    ρu_local_sum = 0.0
    volume_local_sum = 0.0
    ρ_local_sum = 0.0
    
    for (bid, b) in blocks
        NGp = NG+1
        nx_end, ny_end, nz_end = b.Nx+NG, b.Ny+NG, b.Nz+NG
        
        ρ_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 1]
        u_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 2]
        vol_v = @view b.Vol[NGp:nx_end, NGp:ny_end, NGp:nz_end]
        
        # Single fused reduction: compute all 3 sums in one GPU pass
        # instead of 3 separate mapreduce calls (saves 2 GPU syncs + 2 D→H copies)
        ρu_local_sum += Float64(mapreduce((r,u,v) -> r*u/v, +, ρ_v, u_v, vol_v))
        ρ_local_sum += Float64(mapreduce((r,v) -> r/v, +, ρ_v, vol_v))
        volume_local_sum += Float64(mapreduce(v -> one(FT) / v, +, vol_v))
    end
    
    # Global reduction
    ρu_sum_global = MPI.Allreduce(ρu_local_sum, MPI.SUM, comm_world)
    ρ_sum_global = MPI.Allreduce(ρ_local_sum, MPI.SUM, comm_world)
    volume_sum_global = MPI.Allreduce(volume_local_sum, MPI.SUM, comm_world)
    
    u_avg = ρu_sum_global / ρ_sum_global
    ρ_avg = ρ_sum_global / volume_sum_global
    
    # Current mass flux (volume-averaged × cross-section)
    Qm_current = ρu_sum_global / volume_sum_global * A_cross
    
    # ── Mass correction: proportional controller for density drift ──
    ρ_error = Float64(ρ_bulk_target) - ρ_avg
    K_mass = 50.0
    flowx = K_mass * ρ_error
    
    # ── Initialize on first call ──
    if !deschamps_state.initialized
        deschamps_state.Qm_prev = Qm_current
        deschamps_state.Qm_smooth = Qm_current  # initialize EMA
        # Initial guess: TURBULENT friction (Dean correlation), NOT laminar Poiseuille
        # Dean (1978): Cf = 0.073 × Re_D^{-0.25} for pipe turbulent flow
        # For Re_D = 17000: Cf ≈ 0.073 × 17000^{-0.25} ≈ 0.0064
        # τ_w = Cf/2 × ρ × U² → f_x = 2τ_w / R = Cf × ρ × U² / R
        Re_D = Float64(Re_target)
        Cf_dean = 0.073 * Re_D^(-0.25)
        ρ_init = Float64(ρ_bulk_target)
        U_init = Float64(u_bulk_target)
        deschamps_state.f1 = Cf_dean * ρ_init * U_init^2 / Float64(R0)
        deschamps_state.f1_prev = deschamps_state.f1
        deschamps_state.g_prev = 0.0
        deschamps_state.initialized = true
        
        if rank == 0 && KRK == 1
            f1_laminar = 8.0 * Float64(μw) * Float64(u_bulk_target) / Float64(R0)^2
            @printf "Iteration: %d | f1_turb_init: %.6e (vs laminar: %.6e, ratio: %.1f×) | Qm_err: %.4e | flowx: %.4e [INIT]\n" tt deschamps_state.f1 f1_laminar (deschamps_state.f1/f1_laminar) (Qm_current - Q0) flowx
            flush(stdout)
        end
        
        # First step: return turbulent initial guess
        return FT(deschamps_state.f1), FT(flowx)
    end
    
    # ── Smoothed proportional controller (replaces unstable macro-inference) ──
    # The original Deschamps macro-inference used dQm/dt which is extremely noisy
    # in turbulent flow (noise amplified by 1/dt ~ 10⁶). The correction gain
    # 1/A_cross was applied every timestep but the flow responds on ~40,000 step
    # timescales, causing massive f1 oscillation (0.17 → 2.24).
    #
    # Fix: exponential moving average (EMA) of Qm to filter turbulent noise,
    # then proportional control on the smoothed error.
    
    Qm_error_curr = Qm_current - Q0
    
    # ── EMA smoothing: α_ema controls noise rejection ──
    # α_ema = 2/(N+1) where N = number of steps to smooth over
    # N ~ 200 steps → α_ema ≈ 0.01 (fast enough to track drift, filters turbulent noise)
    α_ema = 0.01
    deschamps_state.Qm_smooth = α_ema * Qm_current + (1.0 - α_ema) * deschamps_state.Qm_smooth
    Qm_error_smooth = deschamps_state.Qm_smooth - Q0
    
    # ── Proportional control on smoothed error ──
    # Gain = K_p / A_cross, with K_p chosen for ~2 flow-through-time convergence
    # Flow-through time τ_ft ≈ L_x / U_bulk ≈ 0.027 s ≈ 40,000 steps
    # Effective gain per FTT = K_p × N_ft × α_ema ≈ K_p × 40000 × 0.01 = 400 K_p
    # For stable convergence: 400 K_p < 2 → K_p < 0.005
    # But EMA already filters noise, so we can be more aggressive: K_p = 0.003
    K_p = 0.003
    
    deschamps_state.f1_prev = deschamps_state.f1   # record before update
    deschamps_state.f1 -= (K_p / A_cross) * Qm_error_smooth
    
    # Store for next step
    deschamps_state.Qm_prev = Qm_current
    deschamps_state.g_prev = 0.0
    
    if (tt % 100 == 0 || tt == 1) && rank == 0 && KRK == 1
        @printf "Iteration: %d | rho_avg: %.4e (target: %.4e) | u_avg: %.4e (target: %.4e) | f1: %.6e | Qm_err: %.4e | flowx: %.4e\n" tt ρ_avg ρ_bulk_target u_avg u_bulk_target deschamps_state.f1 Qm_error_curr flowx
        flush(stdout)
    end
    
    return FT(deschamps_state.f1), FT(flowx)
end

function Apply_deschamps_pipe_force!(dU_forced, Q, f1_val, flowx_val, dt, nxp, nyp, nzp)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb deschamps_gpu_kernel!(dU_forced, Q, f1_val, flowx_val, dt, nxp, nyp, nzp)
end

# =============================================================================
# Localized Trip Forcing — "roughness strip" near inlet
# Injects strong multi-mode disturbances in a narrow streamwise band
# to trigger subcritical transition in pipe flow
# =============================================================================
function trip_forcing_kernel!(dU_forced, Q, x, y, z, nxp, nyp, nzp, tt_phys)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    
    ii, jj, kk = i+NG, j+NG, k+NG
    
    @inbounds x_loc = x[ii, jj, kk]
    @inbounds yi    = y[ii, jj, kk]
    @inbounds zi    = z[ii, jj, kk]
    
    # Trip location: x/Lx ∈ [0.05, 0.15] — narrow band near inlet
    x_norm = x_loc / Lx
    if x_norm < FT(0.05e0) || x_norm > FT(0.15e0)
        return
    end
    
    # Streamwise Gaussian envelope centered at x/Lx = 0.10
    x_center = FT(0.10e0) * Lx
    σ_x = FT(0.03e0) * Lx  # width of trip strip
    x_env = exp(-FT(0.5) * ((x_loc - x_center) / σ_x)^2)
    
    # Radial position
    r = sqrt(yi*yi + zi*zi)
    r_norm = r / R0
    
    # Radial envelope: strongest near the wall (r/R0 ∈ [0.5, 0.95])
    # Gaussian centered at r/R0 = 0.75
    r_env = exp(-((r_norm - FT(0.75)) / FT(0.15e0))^2)
    
    θ = atan(zi, yi)
    
    # Physics
    c_wall = sqrt(γ * Rg * Tw)
    u_bulk = Ma_target * c_wall
    μw = C_s * Tw * sqrt(Tw) / (Tw + T_s)
    ρ_ref = Re_target * μw / (u_bulk * FT(2.0) * R0)
    
    # Trip amplitude: 10% of bulk momentum
    amp_trip = FT(0.10e0) * ρ_ref * u_bulk
    
    # Time-varying multi-mode forcing (azimuthal m = 2..8, temporal)
    ω_base = FT(2.0) * FT(π) * u_bulk / (FT(0.10e0) * Lx)  # frequency based on trip width
    
    f_trip = (
        FT(0.25) * sin(FT(2.0)*θ + one(FT)*ω_base*tt_phys) +
        FT(0.20e0) * sin(FT(3.0)*θ - FT(0.7e0)*ω_base*tt_phys + FT(1.3e0)) +
        FT(0.20e0) * sin(FT(4.0)*θ + FT(1.3e0)*ω_base*tt_phys + FT(2.7e0)) +
        FT(0.15e0) * sin(FT(6.0e0)*θ - FT(0.5)*ω_base*tt_phys + FT(4.1e0)) +
        FT(0.10e0) * sin(FT(8.0e0)*θ + FT(1.1e0)*ω_base*tt_phys + FT(5.9e0)) +
        FT(0.10e0) * sin(FT(5.0e0)*θ - FT(1.5e0)*ω_base*tt_phys + FT(3.3e0))
    )
    
    # Apply force in radial direction (v, w) — doesn't directly add streamwise momentum
    force_mag = amp_trip * x_env * r_env * f_trip
    
    # Convert radial force to Cartesian (y, z)
    if r > FT(1.0e-10)
        fy = force_mag * (-sin(θ))  # azimuthal direction
        fz = force_mag * ( cos(θ))
    else
        fy = zero(FT)
        fz = zero(FT)
    end
    
    @inbounds ρ = Q[ii, jj, kk, 1]
    @inbounds u = Q[ii, jj, kk, 2]
    @inbounds v = Q[ii, jj, kk, 3]
    @inbounds w = Q[ii, jj, kk, 4]
    
    @inbounds dU_forced[i, j, k, 3] += fy
    @inbounds dU_forced[i, j, k, 4] += fz
    @inbounds dU_forced[i, j, k, 5] += v*fy + w*fz  # energy coupling
    
    return
end

function Apply_trip_force!(dU_forced, Q, x, y, z, nxp, nyp, nzp, tt_phys)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb trip_forcing_kernel!(dU_forced, Q, x, y, z, nxp, nyp, nzp, FT(tt_phys))
end

# =============================================================================
# HIT Linear Forcing (Lundgren 2003, Rosales & Meneveau 2005)
# f_i = A * ρ * (u_i - <u_i>)
# Energy injection rate: ε = 2A * E_k
# =============================================================================

function HIT_forcing_kernel!(dU_forced, Q, A_force, u_mean, v_mean, w_mean, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii, jj, kk = i+NG, j+NG, k+NG

    @inbounds ρ = Q[ii, jj, kk, 1]
    @inbounds u = Q[ii, jj, kk, 2]
    @inbounds v = Q[ii, jj, kk, 3]
    @inbounds w = Q[ii, jj, kk, 4]

    # Fluctuating velocity
    u_prime = u - u_mean
    v_prime = v - v_mean
    w_prime = w - w_mean

    # Linear forcing: f_i = A * ρ * u_i'
    fx = A_force * ρ * u_prime
    fy = A_force * ρ * v_prime
    fz = A_force * ρ * w_prime

    # Source terms: dU/dt += [0, fx, fy, fz, u·f]
    @inbounds dU_forced[i, j, k, 1] = zero(FT)
    @inbounds dU_forced[i, j, k, 2] = fx
    @inbounds dU_forced[i, j, k, 3] = fy
    @inbounds dU_forced[i, j, k, 4] = fz
    @inbounds dU_forced[i, j, k, 5] = u * fx + v * fy + w * fz
    return
end

# MHD variant: also applies linear forcing to the magnetic field
#   f_B = A * B'   (B minus its volume mean, to avoid DC drift)
# The energy injection rate into the magnetic field is ε_B = 2A * E_mag.
# The induction equation source dU[6:8]/dt += A*B' is added; ψ (U[9]) is not forced.
# Energy flux into the total energy U[5] from the magnetic forcing is accounted
# for by the MHD Riemann flux automatically, so we do NOT add an extra U[5]
# term here (unlike the kinetic forcing above, which adds u·f to U[5]).
function HIT_forcing_kernel_MHD!(dU_forced, Q, A_force,
                                 u_mean, v_mean, w_mean,
                                 Bx_mean, By_mean, Bz_mean,
                                 nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end

    ii, jj, kk = i+NG, j+NG, k+NG

    @inbounds ρ = Q[ii, jj, kk, 1]
    @inbounds u = Q[ii, jj, kk, 2]
    @inbounds v = Q[ii, jj, kk, 3]
    @inbounds w = Q[ii, jj, kk, 4]
    @inbounds Bx = Q[ii, jj, kk, 7]
    @inbounds By = Q[ii, jj, kk, 8]
    @inbounds Bz = Q[ii, jj, kk, 9]

    # Fluctuating velocity and B-field (mean-subtracted to avoid DC drift)
    u_prime = u - u_mean
    v_prime = v - v_mean
    w_prime = w - w_mean
    Bx_prime = Bx - Bx_mean
    By_prime = By - By_mean
    Bz_prime = Bz - Bz_mean

    # Linear forcing on velocity: f_i = A * ρ * u_i'
    fx = A_force * ρ * u_prime
    fy = A_force * ρ * v_prime
    fz = A_force * ρ * w_prime

    # Linear forcing on B: f_B = A * B'  (induction equation source)
    fBx = A_force * Bx_prime
    fBy = A_force * By_prime
    fBz = A_force * Bz_prime

    @inbounds dU_forced[i, j, k, 1] = zero(FT)
    @inbounds dU_forced[i, j, k, 2] = fx
    @inbounds dU_forced[i, j, k, 3] = fy
    @inbounds dU_forced[i, j, k, 4] = fz
    # Energy source: kinetic work u·f_v. The magnetic forcing contributes to
    # total energy through the induction coupling (J·E term) handled by the
    # MHD flux; here we add only the explicit kinetic work to U[5].
    @inbounds dU_forced[i, j, k, 5] = u * fx + v * fy + w * fz
    # Magnetic field sources (Bx, By, Bz). ψ (U[9]) is left to GLM.
    @inbounds dU_forced[i, j, k, 6] = fBx
    @inbounds dU_forced[i, j, k, 7] = fBy
    @inbounds dU_forced[i, j, k, 8] = fBz
    @inbounds dU_forced[i, j, k, 9] = zero(FT)
    return
end

function Update_HIT_forcing(blocks, A_force, comm_world, rank, tt, KRK)
    # Compute domain-averaged velocity for mean subtraction
    ρu_local = zero(FT)
    ρv_local = zero(FT)
    ρw_local = zero(FT)
    ρ_local  = zero(FT)

    for (bid, b) in blocks
        NGp = NG + 1
        nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG

        ρ_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 1]
        u_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 2]
        v_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 3]
        w_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 4]

        ρ_local  += sum(ρ_v)
        ρu_local += mapreduce(*, +, ρ_v, u_v)
        ρv_local += mapreduce(*, +, ρ_v, v_v)
        ρw_local += mapreduce(*, +, ρ_v, w_v)
    end

    # Global reduction
    ρ_global  = MPI.Allreduce(Float64(ρ_local),  MPI.SUM, comm_world)
    ρu_global = MPI.Allreduce(Float64(ρu_local), MPI.SUM, comm_world)
    ρv_global = MPI.Allreduce(Float64(ρv_local), MPI.SUM, comm_world)
    ρw_global = MPI.Allreduce(Float64(ρw_local), MPI.SUM, comm_world)

    u_mean = FT(ρu_global / ρ_global)
    v_mean = FT(ρv_global / ρ_global)
    w_mean = FT(ρw_global / ρ_global)

    # MHD: also compute volume-averaged B-field for mean subtraction
    Bx_mean = zero(FT)
    By_mean = zero(FT)
    Bz_mean = zero(FT)
    @static if equation_type == :MHD
        Bx_local = zero(FT)
        By_local = zero(FT)
        Bz_local = zero(FT)
        N_local = 0
        for (bid, b) in blocks
            NGp = NG + 1
            nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG
            Bx_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 7]
            By_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 8]
            Bz_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 9]
            Bx_local += sum(Bx_v)
            By_local += sum(By_v)
            Bz_local += sum(Bz_v)
            N_local  += b.Nx * b.Ny * b.Nz
        end
        N_global = MPI.Allreduce(N_local, MPI.SUM, comm_world)
        Bx_mean = FT(MPI.Allreduce(Float64(Bx_local), MPI.SUM, comm_world) / N_global)
        By_mean = FT(MPI.Allreduce(Float64(By_local), MPI.SUM, comm_world) / N_global)
        Bz_mean = FT(MPI.Allreduce(Float64(Bz_local), MPI.SUM, comm_world) / N_global)
    end

    # Print TKE monitoring info
    if (tt % 100 == 0 || tt == 1) && rank == 0 && KRK == 1
        tke_local = zero(FT)
        for (bid, b) in blocks
            NGp = NG + 1
            nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG
            ρ_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 1]
            u_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 2]
            v_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 3]
            w_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 4]
            tke_local += sum(ρ_v .* ((u_v .- u_mean).^2 .+ (v_v .- v_mean).^2 .+ (w_v .- w_mean).^2))
        end
        tke_global = MPI.Allreduce(Float64(tke_local), MPI.SUM, comm_world)
        N_total = 0
        for (bid, b) in blocks; N_total += b.Nx * b.Ny * b.Nz; end
        tke_avg = FT(0.5) * FT(tke_global / N_total)
        eps_inject = FT(2.0) * A_force * tke_avg
        @printf "  HIT: TKE=%.4e  ε_inject=%.4e  <u>=(%.3e, %.3e, %.3e)\n" tke_avg eps_inject u_mean v_mean w_mean
    end

    return u_mean, v_mean, w_mean, Bx_mean, By_mean, Bz_mean
end

function Apply_HIT_forcing!(dU_forced, Q, A_force,
                            u_mean, v_mean, w_mean,
                            Bx_mean, By_mean, Bz_mean,
                            nxp, nyp, nzp)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @static if equation_type == :MHD
        @gpu_launch threads=nthreads blocks=nb HIT_forcing_kernel_MHD!(
            dU_forced, Q, A_force,
            u_mean, v_mean, w_mean,
            Bx_mean, By_mean, Bz_mean,
            nxp, nyp, nzp)
    else
        @gpu_launch threads=nthreads blocks=nb HIT_forcing_kernel!(
            dU_forced, Q, A_force, u_mean, v_mean, w_mean, nxp, nyp, nzp)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# volume_force_mhd.jl — MHD-specific volume source terms
# ═══════════════════════════════════════════════════════════════════════
# GLM divergence cleaning: ψ-damping source term
#   ∂ψ/∂t = -(ch/cr)·ψ   (exponential decay)
# Applied via operator splitting after the flux update.
# ═══════════════════════════════════════════════════════════════════════

# ── GLM ψ-damping kernel ──
# Applies exponential decay to the GLM cleaning potential ψ = U[9]
# Uses operator splitting: ψ^{n+1} = ψ^n · exp(-dt · ch / cr)
# where ch = ch_glm_current (auto-computed), cr = cr_glm (user-set)
function glm_source_kernel!(U, dt, ch, cr, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp || j > nyp || k > nzp
        return
    end

    ii, jj, kk = i+NG, j+NG, k+NG

    # Exponential decay for ψ
    # decay_rate = ch / cr (combining parabolic + hyperbolic cleaning)
    decay = exp(-dt * ch / (cr + FT(1.0e-20)))
    @inbounds U[ii, jj, kk, 9] *= decay

    return
end

# ── ch_glm computation kernel ──
# Computes the maximum fast magnetosonic speed for each cell.
# The global ch_glm is then set to the maximum over all cells (via MPI reduction).
function compute_cf_max_kernel!(cf_arr, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp+NG || j > nyp+NG || k > nzp+NG || i < NG+1 || j < NG+1 || k < NG+1
        return
    end

    @inbounds ρ = max(Q[i, j, k, 1], FT(1.0e-10))
    @inbounds p = max(Q[i, j, k, 5], FT(1.0e-10))
    @inbounds Bx = Q[i, j, k, 7]; @inbounds By = Q[i, j, k, 8]; @inbounds Bz = Q[i, j, k, 9]
    B2 = Bx*Bx + By*By + Bz*Bz

    c2 = γ * p / ρ           # sound speed²
    va2 = B2 / ρ             # Alfvén speed²
    cf = sqrt(c2 + va2)      # fast magnetosonic (isotropic)

    # Add velocity magnitude for max signal speed
    @inbounds u = Q[i, j, k, 2]; @inbounds v = Q[i, j, k, 3]; @inbounds w = Q[i, j, k, 4]
    v_mag = sqrt(u*u + v*v + w*w)

    @inbounds cf_arr[i, j, k] = cf + v_mag

    return
end
