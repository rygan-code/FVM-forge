# =============================================================================
# Volume force for multi-block FVM configuration
# Retains rotation (Coriolis, centrifugal, Euler) and bulk forcing
# Uses global communicator for cross-block reductions
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

function Assign_rotation_var(Ωx, Ωy, Ωz, x, y, z, nxp, nyp, nzp, x_rot_start, x_rot_end)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp+2*NG || j > nyp+2*NG || k > nzp+2*NG
        return
    end

    @inbounds x_loc = x[i, j, k]
    
    if x_loc <= x_rot_start
        spatial_factor = zero(FT)
    elseif x_loc < x_rot_end
        # 逐渐线性增大：(x - x_start) / (x_end - x_start)
        spatial_factor = (x_loc - x_rot_start) / (x_rot_end - x_rot_start)
    else
        # 到达末端后保持最大值 (或者你可以按需让它衰减)
        spatial_factor = one(FT)
    end

    @inbounds Ωx[i, j, k] = Omega_x * spatial_factor
    @inbounds Ωy[i, j, k] = zero(FT)
    @inbounds Ωz[i, j, k] = zero(FT)
    return
end

function Volume_force_kernel!(dU_forced, Q, x, y, z, nxp, nyp, nzp, Ωx, Ωy, Ωz, ramp)
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
    
    # Scale local rotation vectors by ramp factor
    @inbounds Ωx_loc = Ωx[ii, jj, kk] * ramp
    @inbounds Ωy_loc = Ωy[ii, jj, kk] * ramp
    @inbounds Ωz_loc = Ωz[ii, jj, kk] * ramp

    # Coriolis force
    fx = FT(2.0)*ρ*(v*Ωz_loc-w*Ωy_loc)
    fy = FT(2.0)*ρ*(w*Ωx_loc-u*Ωz_loc)
    fz = FT(2.0)*ρ*(u*Ωy_loc-v*Ωx_loc)

    # Centrifugal force
    fx += ρ*((Ωy_loc^2 + Ωz_loc^2)*x[ii, jj, kk]-Ωx_loc*(Ωy_loc*y[ii, jj, kk]+Ωz_loc*z[ii, jj, kk]))
    fy += ρ*((Ωz_loc^2 + Ωx_loc^2)*y[ii, jj, kk]-Ωy_loc*(Ωz_loc*z[ii, jj, kk]+Ωx_loc*x[ii, jj, kk]))
    fz += ρ*((Ωx_loc^2 + Ωy_loc^2)*z[ii, jj, kk]-Ωz_loc*(Ωx_loc*x[ii, jj, kk]+Ωy_loc*y[ii, jj, kk]))

    # Euler force (requires Omega gradients)
    # Note: For constant Omega, this is zero. 
    # Providing the code structure for future non-uniform rotation.
    
    @inbounds dU_forced[i, j, k, 1] = zero(FT)
    @inbounds dU_forced[i, j, k, 2] = fx
    @inbounds dU_forced[i, j, k, 3] = fy
    @inbounds dU_forced[i, j, k, 4] = fz
    @inbounds dU_forced[i, j, k, 5] = u*fx + v*fy + w*fz
    return
end

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
        
        # Avoid multi-array mapreduce on views (which causes compilation or value mismatch bugs in older AMDGPU versions)
        temp_ρ = ρ_v ./ vol_v
        temp_vol = one(FT) ./ vol_v
        temp_ρu = ρ_v .* u_v ./ vol_v
        ρ_local_sum += sum(temp_ρ)
        volume_local_sum += sum(temp_vol)
        u_local_sum += sum(temp_ρu)
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
        
        temp_ρu = Float64.(ρ_v) .* Float64.(u_v) ./ Float64.(vol_v)
        temp_ρ = Float64.(ρ_v) ./ Float64.(vol_v)
        temp_vol = 1.0 ./ Float64.(vol_v)
        ρu_local_sum += sum(temp_ρu)
        ρ_local_sum += sum(temp_ρ)
        volume_local_sum += sum(temp_vol)
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
        
        # Avoid multi-array mapreduce on views (which causes compilation or value mismatch bugs in older AMDGPU versions)
        temp_ρu = ρ_v .* u_v ./ vol_v
        temp_ρ = ρ_v ./ vol_v
        temp_vol = one(FT) ./ vol_v
        ρu_local_sum += Float64(sum(temp_ρu))
        ρ_local_sum += Float64(sum(temp_ρ))
        volume_local_sum += Float64(sum(temp_vol))
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
    
    if (tt % 100 == 0 || tt <= 20) && rank == 0 && KRK == 1
        @printf "Main-Deschamps: Iteration %d | rho_avg: %.4e (target: %.4e) | u_avg: %.4e (target: %.4e) | f1: %.6e | Qm_err: %.4e | flowx: %.4e\n" tt ρ_avg ρ_bulk_target u_avg u_bulk_target deschamps_state.f1 Qm_error_curr flowx
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
        temp_ρu = ρ_v .* u_v
        temp_ρv = ρ_v .* v_v
        temp_ρw = ρ_v .* w_v
        ρu_local += sum(temp_ρu)
        ρv_local += sum(temp_ρv)
        ρw_local += sum(temp_ρw)
    end
    
    # Global reduction
    ρ_global  = MPI.Allreduce(Float64(ρ_local),  MPI.SUM, comm_world)
    ρu_global = MPI.Allreduce(Float64(ρu_local), MPI.SUM, comm_world)
    ρv_global = MPI.Allreduce(Float64(ρv_local), MPI.SUM, comm_world)
    ρw_global = MPI.Allreduce(Float64(ρw_local), MPI.SUM, comm_world)
    
    u_mean = FT(ρu_global / ρ_global)
    v_mean = FT(ρv_global / ρ_global)
    w_mean = FT(ρw_global / ρ_global)
    
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
    
    return u_mean, v_mean, w_mean
end

function Apply_HIT_forcing!(dU_forced, Q, A_force, u_mean, v_mean, w_mean, nxp, nyp, nzp)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb HIT_forcing_kernel!(dU_forced, Q, A_force, u_mean, v_mean, w_mean, nxp, nyp, nzp)
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

# ═══════════════════════════════════════════════════════════════════════
# CEBL-DNS volume source terms
# ═══════════════════════════════════════════════════════════════════════

using Adapt

mutable struct CEBLForcingState{V, QType}
    enabled::Bool
    profile_size::Int32
    y::V
    M::V
    F::V
    Q::V
    Q_cebl::QType
end

Adapt.adapt_structure(to, state::CEBLForcingState) = CEBLForcingState(
    state.enabled,
    state.profile_size,
    adapt(to, state.y),
    adapt(to, state.M),
    adapt(to, state.F),
    adapt(to, state.Q),
    adapt(to, state.Q_cebl)
)

function create_dummy_cebl_state(FT::Type)
    V = GPUVector{FT}
    y_d = V(zeros(FT, 0))
    M_d = V(zeros(FT, 0))
    F_d = V(zeros(FT, 0))
    Q_d = V(zeros(FT, 0))
    return CEBLForcingState{V, Any}(
        false, Int32(0), y_d, M_d, F_d, Q_d, nothing
    )
end

function init_cebl_state(source_file::String)
    h_y = h5read(source_file, "y_coord")
    h_M = h5read(source_file, "M")
    h_F = h5read(source_file, "F")
    h_Q = h5read(source_file, "Q")
    
    h_y = FT.(h_y)
    h_M = FT.(h_M)
    h_F = FT.(h_F)
    h_Q = FT.(h_Q)
    
    V = GPUVector{FT}
    y_d = V(h_y)
    M_d = V(h_M)
    F_d = V(h_F)
    Q_d = V(h_Q)
    
    return CEBLForcingState{V, Any}(
        true, Int32(length(h_y)), y_d, M_d, F_d, Q_d, nothing
    )
end

# GPU-compatible 1D interpolation helper
@inline function interpolate_1d(val::FT, coords, profile, N::Int32) where {FT}
    if N <= 0
        return zero(FT)
    end
    if val <= coords[1]
        return profile[1]
    elseif val >= coords[N]
        return profile[N]
    end
    
    # Binary search
    low = Int32(1)
    high = N
    while high - low > Int32(1)
        mid = (low + high) >> 1
        if val >= coords[mid]
            low = mid
        else
            high = mid
        end
    end
    
    # Linear interpolation
    c_low = coords[low]
    c_high = coords[high]
    t = (val - c_low) / (c_high - c_low + eps(FT))
    return profile[low] + t * (profile[high] - profile[low])
end

# CEBL source terms kernel
function cebl_force_kernel!(dU_forced, Q, y_coord, z_coord, nxp, nyp, nzp,
                            cebl_y, cebl_M, cebl_F, cebl_Q, cebl_N, is_pipe::Bool)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    
    ii, jj, kk = i+NG, j+NG, k+NG
    
    # Calculate wall distance
    if is_pipe
        r = sqrt(y_coord[ii, jj, kk]^2 + z_coord[ii, jj, kk]^2)
        d_wall = R0 - r
    else
        d_wall = y_coord[ii, jj, kk]
    end
    
    # Interpolate source terms
    M_val = interpolate_1d(d_wall, cebl_y, cebl_M, cebl_N)
    F_val = interpolate_1d(d_wall, cebl_y, cebl_F, cebl_N)
    Q_val = interpolate_1d(d_wall, cebl_y, cebl_Q, cebl_N)
    
    # Add to dU_forced (negative source terms as per formulation)
    @inbounds dU_forced[i, j, k, 1] -= M_val
    @inbounds dU_forced[i, j, k, 2] -= F_val
    @inbounds dU_forced[i, j, k, 5] -= Q_val
    
    return
end

function Apply_CEBL_force!(dU_forced, Q, y_coord, z_coord, state::CEBLForcingState, nxp, nyp, nzp)
    if !state.enabled
        return
    end
    is_pipe = (test_case == "PipeFlow")
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb cebl_force_kernel!(
        dU_forced, Q, y_coord, z_coord, nxp, nyp, nzp,
        state.y, state.M, state.F, state.Q, state.profile_size, is_pipe
    )
end

# ═══════════════════════════════════════════════════════════════════════
# Local periodic wrapping in x-direction for CEBL auxiliary block
# ═══════════════════════════════════════════════════════════════════════

function local_periodic_x_kernel!(V, nxp, nyp, nzp, NG, Nvar)
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if j > nyp + 2*NG || k > nzp + 2*NG
        return
    end
    for i in 1:NG
        for n in 1:Nvar
            @inbounds V[i, j, k, n] = V[nxp + i, j, k, n]
            @inbounds V[nxp + NG + i, j, k, n] = V[NG + i, j, k, n]
        end
    end
    return
end

function Apply_local_periodic_x!(V, nxp, nyp, nzp, NG, Nvar)
    nthreads_2d = (16, 16)
    nb_2d = (1, cld(nyp + 2*NG, nthreads_2d[1]), cld(nzp + 2*NG, nthreads_2d[2]))
    @gpu_launch threads=(1, nthreads_2d[1], nthreads_2d[2]) blocks=nb_2d local_periodic_x_kernel!(
        V, nxp, nyp, nzp, NG, Nvar
    )
end

# ═══════════════════════════════════════════════════════════════════════
# Local Deschamps Constant Mass Flux Forcing for CEBL Auxiliary Block
# ═══════════════════════════════════════════════════════════════════════

const cebl_deschamps_state = DeschampsState(0.0, 0.0, 0.0, 0.0, 0.0, false)

function Update_cebl_deschamps_params(cebl_blocks, tt, KRK, dt, comm_world, rank)
    c_wall = sqrt(γ * Rg * Tw)
    u_bulk_target = Ma_target * c_wall
    μw = C_s * Tw * sqrt(Tw) / (Tw + T_s)
    ρ_bulk_target = Re_target * μw / (u_bulk_target * 2 * R0)
    
    # Cross-section area
    A_cross = Float64(π) * Float64(R0)^2
    Q0 = Float64(ρ_bulk_target) * Float64(u_bulk_target) * A_cross
    
    ρu_local_sum = 0.0
    ρ_local_sum = 0.0
    volume_local_sum = 0.0
    
    for (cebl_bid, b) in cebl_blocks
        NGp = NG+1
        nx_end, ny_end, nz_end = b.Nx+NG, b.Ny+NG, b.Nz+NG
        
        ρ_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 1]
        u_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 2]
        vol_v = @view b.Vol[NGp:nx_end, NGp:ny_end, NGp:nz_end]
        
        # Avoid multi-array mapreduce on views (which causes compilation or value mismatch bugs in older AMDGPU versions)
        temp_ρu = ρ_v .* u_v ./ vol_v
        temp_ρ = ρ_v ./ vol_v
        temp_vol = one(FT) ./ vol_v
        
        ρu_local_sum += Float64(sum(temp_ρu))
        ρ_local_sum += Float64(sum(temp_ρ))
        volume_local_sum += Float64(sum(temp_vol))
    end
    
    # Global reduction across ALL ranks in COMM_WORLD
    ρu_sum_global = MPI.Allreduce(ρu_local_sum, MPI.SUM, comm_world)
    ρ_sum_global = MPI.Allreduce(ρ_local_sum, MPI.SUM, comm_world)
    volume_sum_global = MPI.Allreduce(volume_local_sum, MPI.SUM, comm_world)
    
    u_avg = ρu_sum_global / ρ_sum_global
    ρ_avg = ρ_sum_global / volume_sum_global
    Qm_current = ρu_sum_global / volume_sum_global * A_cross
    
    ρ_error = Float64(ρ_bulk_target) - ρ_avg
    K_mass = 50.0
    flowx = K_mass * ρ_error
    
    if !cebl_deschamps_state.initialized
        cebl_deschamps_state.Qm_prev = Qm_current
        cebl_deschamps_state.Qm_smooth = Qm_current
        
        Re_D = Float64(Re_target)
        Cf_dean = 0.073 * Re_D^(-0.25)
        ρ_init = Float64(ρ_bulk_target)
        U_init = Float64(u_bulk_target)
        cebl_deschamps_state.f1 = Cf_dean * ρ_init * U_init^2 / Float64(R0)
        cebl_deschamps_state.f1_prev = cebl_deschamps_state.f1
        cebl_deschamps_state.g_prev = 0.0
        cebl_deschamps_state.initialized = true
        
        if rank == 0 && KRK == 1
            f1_laminar = 8.0 * Float64(μw) * Float64(u_bulk_target) / Float64(R0)^2
            @printf "CEBL-Deschamps: Iteration %d initialized | f1: %.6e (vs laminar: %.6e) | Qm_err: %.4e | flowx: %.4e\n" tt cebl_deschamps_state.f1 f1_laminar (Qm_current - Q0) flowx
            flush(stdout)
        end
        return FT(cebl_deschamps_state.f1), FT(flowx)
    end
    
    # EMA smoothing of Qm
    α_ema = 0.01
    cebl_deschamps_state.Qm_smooth = α_ema * Qm_current + (1.0 - α_ema) * cebl_deschamps_state.Qm_smooth
    Qm_error_smooth = cebl_deschamps_state.Qm_smooth - Q0
    Qm_error_curr = Qm_current - Q0
    
    # PI parameters (consistent with main blocks, K_p = 0.003)
    K_p = 0.003
    
    cebl_deschamps_state.f1_prev = cebl_deschamps_state.f1
    cebl_deschamps_state.f1 -= (K_p / A_cross) * Qm_error_smooth
    cebl_deschamps_state.Qm_prev = Qm_current
    cebl_deschamps_state.g_prev = 0.0
    
    if rank == 0 && KRK == 1 && (tt % 100 == 0 || tt <= 20)
        @printf "CEBL-Deschamps: Iteration %d | rho_avg: %.4e | u_avg: %.4e | f1: %.6e | Qm_err: %.4e | flowx: %.4e\n" tt ρ_avg u_avg cebl_deschamps_state.f1 Qm_error_curr flowx
        flush(stdout)
    end
    
    return FT(cebl_deschamps_state.f1), FT(flowx)
end


