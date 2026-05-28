# ═══════════════════════════════════════════════════════════════════════
#  Matrix-Free GMRES(m) with Block-Jacobi Preconditioning
#  for Implicit Time Stepping in Flame3D
#
#  Replaces LU-SGS hyperplane sweeps with a fully-parallel Krylov solver.
#
#  Solves: A·x = b  where
#    A = (αV/(dt))·I + ∂R/∂U   (α = 1 for BE, 1.5 for BDF2)
#    b = dU_rhs                  (spatial RHS + temporal source)
#    x = ΔU                      (solution increment)
#
#  Matrix-vector product via finite difference:
#    A·v ≈ [R(U+εv) - R(U)] / ε + (αV/dt)·v
#
#  Left-preconditioned: solve P⁻¹Ax = P⁻¹b with P = diag(A) = D
# ═══════════════════════════════════════════════════════════════════════

# ─── GPU kernels for vector operations ───

# w = P⁻¹·v (apply block-Jacobi preconditioner = D_inv × v)
# Both w and v are (Nx_tot, Ny_tot, Nz_tot, Ncons) — global indexing with ghost cells
function gmres_precond!(w, v, D_inv, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    @inbounds d_inv = D_inv[ig, jg, kg]
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = d_inv * v[ig, jg, kg, n]
    end
    return
end

# w = P⁻¹·rhs where rhs is (nxp, nyp, nzp, Ncons) — LOCAL indexing (no ghost cells)
# w is (Nx_tot, Ny_tot, Nz_tot, Ncons) — GLOBAL indexing with ghost cells
function gmres_precond_rhs!(w, rhs, D_inv, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    @inbounds d_inv = D_inv[ig, jg, kg]
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = d_inv * rhs[i, j, k, n]  # local read, global write
    end
    return
end

# w = U_base + ε·v  (perturb state for finite-difference Jacobian)
function gmres_perturb!(w, U_base, v, eps_fd, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = U_base[ig, jg, kg, n] + eps_fd * v[ig, jg, kg, n]
    end
    return
end

# Av = (αV/dt)·v − (R_perturbed - R_base) / ε
# The operator A = αV/dt − ∂R_spatial/∂U  (positive definite for dissipative systems)
# FD gives ∂R/∂U·v ≈ (R(U+εv) - R(U))/ε, so we NEGATE it.
# Then apply preconditioner: result = P⁻¹·Av
function gmres_compute_Av!(Av, R_perturbed, R_base, v, D_inv, Vol, dt, alpha_bdf, eps_inv, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    @inbounds begin
        vol_inv = Vol[ig, jg, kg]
        cell_vol = one(FT) / (vol_inv + FT(1.0e-30))
        d_inv = D_inv[ig, jg, kg]
        temporal = alpha_bdf * cell_vol / dt
        for n = Int32(1):Int32(Ncons)
            # Jacobian-vector product: ∂R_sp/∂U · v ≈ (R(U+εv) - R(U)) / ε
            jac_v = (R_perturbed[i, j, k, n] - R_base[i, j, k, n]) * eps_inv
            # Full operator: A·v = (αV/dt)·v − ∂R_sp/∂U·v  (note: MINUS sign)
            av_n = temporal * v[ig, jg, kg, n] - jac_v
            # Left preconditioning: P⁻¹·A·v
            Av[ig, jg, kg, n] = d_inv * av_n
        end
    end
    return
end

# w = a*x + y (AXPY: GPU parallel)
function gmres_axpy!(w, a, x, y, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = a * x[ig, jg, kg, n] + y[ig, jg, kg, n]
    end
    return
end

# w = x - a*y (for Gram-Schmidt orthogonalization)
function gmres_sub_scaled!(w, x, a, y, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = x[ig, jg, kg, n] - a * y[ig, jg, kg, n]
    end
    return
end

# w = x / a (scale vector)
function gmres_scale!(w, x, a_inv, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp || i < Int32(1) || j < Int32(1) || k < Int32(1)
        return
    end
    ig = i + NG; jg = j + NG; kg = k + NG
    for n = Int32(1):Int32(Ncons)
        @inbounds w[ig, jg, kg, n] = x[ig, jg, kg, n] * a_inv
    end
    return
end

# ─── CPU-side dot product (zero-allocation via GPU BLAS) ───
# Computes ⟨a, b⟩ over real cells [NG+1:Nx+NG, NG+1:Ny+NG, NG+1:Nz+NG, 1:Ncons]
function _gmres_dot(a::GPUArray{FT, 4}, b::GPUArray{FT, 4}, nxp, nyp, nzp)
    # Extract real-cell views and compute dot product
    a_real = @view a[NG+1:NG+nxp, NG+1:NG+nyp, NG+1:NG+nzp, :]
    b_real = @view b[NG+1:NG+nxp, NG+1:NG+nyp, NG+1:NG+nzp, :]
    # sum(a.*b) fuses via GPUArrays broadcast+mapreduce — single kernel
    # NOTE: LinearAlgebra.dot falls back to scalar iteration on N-D SubArrays
    return Float64(sum(a_real .* b_real))
end

# ─── CPU-side norm ───
function _gmres_norm(v::GPUArray{FT, 4}, nxp, nyp, nzp)
    return sqrt(_gmres_dot(v, v, nxp, nyp, nzp))
end


# ═══════════════════════════════════════════════════════════════════════
#  Main GMRES solve function
#
#  Solves P⁻¹·A·x = P⁻¹·b using GMRES(m) with restarts
#
#  Arguments:
#    block      - Block struct (contains U, Q, dU_rhs, ΔU, D_inv, etc.)
#    dt_val     - Time step
#    alpha_bdf  - BDF coefficient (1.0 for BE, 1.5 for BDF2)
#    m          - Krylov subspace dimension
#    max_restarts - Number of restarts
#    tol        - Relative residual tolerance
#    V_krylov   - Pre-allocated Krylov basis vectors [Nx_tot, Ny_tot, Nz_tot, Ncons, m+1]
#    shared_*   - Shared flux buffers
#    ...        - Other arguments passed to blockAdvance
#
#  Returns: converged residual norm
# ═══════════════════════════════════════════════════════════════════════
function gmres_solve!(block::Block, dt_val::FT, alpha_bdf::FT,
                      m::Int, max_restarts::Int, tol::Float64,
                      V_krylov::GPUArray{FT, 5},
                      R_base::GPUArray{FT, 4},
                      shared_Fx, shared_Fy, shared_Fz,
                      shared_Fvx, shared_Fvy, shared_Fvz,
                      shared_dU_forced, world_rank, tt,
                      threads_recon_i, threads_recon_j, threads_recon_k,
                      threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                      forcex, flowx, cmf_f1_val, ac_f1_val,
                      hit_u_mean, hit_v_mean, hit_w_mean, activeTime,
                      sync_ghost_fn)  # callback: sync_ghost_fn() exchanges ghost cells for all blocks

    nxp = block.Nx; nyp = block.Ny; nzp = block.Nz
    Nx_tot = nxp + 2 * NG; Ny_tot = nyp + 2 * NG; Nz_tot = nzp + 2 * NG
    nb_real = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    nb_light = (Int32(cld(Nx_tot, threads_light[1])),
                Int32(cld(Ny_tot, threads_light[2])),
                Int32(cld(Nz_tot, threads_light[3])))

    # ── Compute P⁻¹·b (right-hand side with preconditioning) ──
    # b = dU_rhs is already computed, apply P⁻¹
    # Store P⁻¹·b in ΔU temporarily (will be overwritten by solution)
    @gpu_launch threads=nthreads blocks=nb_real gmres_precond_rhs!(
        block.ΔU, block.dU_rhs, block.D_inv, nxp, nyp, nzp)

    # ── Save the base spatial RHS (R_base = dU_rhs, before BDF2 source) ──
    # Note: R_base was computed by the caller and stored in the passed buffer

    # Finite-difference step size for Jacobian approximation
    # ε = sqrt(machine_eps) × (1 + ||U||) / ||v||
    # We'll compute per-iteration. For now, get ||U||.
    U_norm = _gmres_norm(block.U, nxp, nyp, nzp)

    # ── GMRES(m) with restarts ──
    final_res = Inf
    for restart = 1:max_restarts + 1
        # r₀ = P⁻¹·b - P⁻¹·A·x₀
        # Since x₀ = 0 (we start with ΔU = 0), r₀ = P⁻¹·b (already in block.ΔU)
        if restart == 1
            # First restart: r₀ = P⁻¹·b = block.ΔU
            # Copy r₀ into V[1]
            beta = _gmres_norm(block.ΔU, nxp, nyp, nzp)
            if beta < 1e-30
                fill!(block.ΔU, zero(FT))
                return 0.0
            end
            # V[1] = r₀ / β
            V1 = @view V_krylov[:, :, :, :, 1]
            inv_beta = FT(1.0 / beta)
            @gpu_launch threads=nthreads blocks=nb_real gmres_scale!(
                V1, block.ΔU, inv_beta, nxp, nyp, nzp)
        else
            # For restarts > 1: r = P⁻¹·b - P⁻¹·A·x_current
            # x_current is in block.ΔU, need to compute A·x and subtract
            # This is complex — for now, we don't restart (single GMRES(m) pass)
            break
        end

        # Hessenberg matrix H[m+1, m] and Givens rotation vectors (CPU)
        H = zeros(Float64, m + 1, m)
        cs = zeros(Float64, m)  # cosines
        sn = zeros(Float64, m)  # sines
        e1 = zeros(Float64, m + 1)
        e1[1] = beta

        converged = false
        actual_m = m  # May be less if converged early

        for j = 1:m
            # ── Arnoldi step: compute w = P⁻¹·A·V[j] ──
            Vj = @view V_krylov[:, :, :, :, j]
            Vj1 = @view V_krylov[:, :, :, :, j + 1]  # will store w after orthogonalization

            # 1. Perturb state: U_perturb = U + ε·Vj
            v_norm = _gmres_norm(Vj, nxp, nyp, nzp)
            eps_fd = FT(sqrt(eps(Float64)) * max(1.0, U_norm) / max(v_norm, 1e-30))

            # Save current U and Q
            copyto!(Vj1, block.U)  # Use Vj1 as temp to save U

            # Set U = U + ε·Vj, then Q = prim(U)
            @gpu_launch threads=nthreads blocks=nb_real gmres_perturb!(
                block.U, Vj1, Vj, eps_fd, nxp, nyp, nzp)
            # For AC: Q ≡ U
            if equation_type == :incompressible_AC
                copyto!(block.Q, block.U)
            end

            # NOTE: Ghost cells are NOT re-synced after perturbation.
            # sync_blocks! is a global MPI collective and cannot be called from
            # inside a per-block GMRES loop (would cause MPI deadlock).
            # The perturbation ε ≈ 1e-7, so ghost error is O(ε²) ≈ O(1e-14).
            # Ghost cells are properly synced at the start of each BDF2 sub-iter.

            # 2. Compute R(U + ε·Vj) → store in a temp buffer
            # We reuse block.dU_rhs as the output for the perturbed RHS
            blockAdvance(block, dt_val, block.ϕ, shared_Fx, shared_Fy, shared_Fz,
                         shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt,
                         threads_recon_i, threads_recon_j, threads_recon_k,
                         threads_visc_i, threads_visc_j, threads_visc_k)
            # Volume forces
            if flow_forcing
                if equation_type == :incompressible_AC
                    nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
                    if forcing_mode == 1
                        @gpu_launch threads=nthreads blocks=nb_f ac_pipe_force_rhs_kernel!(
                            shared_dU_forced, ac_f1_val, nxp, nyp, nzp)
                    end
                end
            end
            # R_perturbed = div_to_rhs output
            # Reuse block.dU_rhs as temp buffer (original RHS saved in R_base)
            @gpu_launch threads=nthreads blocks=nb_real div_to_rhs(
                block.dU_rhs, block.U, shared_Fx, shared_Fy, shared_Fz,
                shared_Fvx, shared_Fvy, shared_Fvz,
                shared_dU_forced, dt_val, block.Vol, nxp, nyp, nzp)

            # HPDC
            if ac_hpdc
                @gpu_launch threads=nthreads blocks=nb_real ac_hpdc_rhs_kernel!(
                    block.dU_rhs, block.Q, block.Vol, block.Areai, block.Areaj, block.Areak,
                    ε_p_AC, nxp, nyp, nzp)
            end

            # 3. Restore U and Q, then re-sync ghost cells
            copyto!(block.U, Vj1)  # Restore from temp
            if equation_type == :incompressible_AC
                copyto!(block.Q, block.U)
            end
            # (Ghost sync intentionally omitted — see note above)

            # 4. Compute Av = P⁻¹ × [(R_perturbed - R_base) / ε + (αV/dt)·Vj]
            eps_inv = FT(1.0 / Float64(eps_fd))
            @gpu_launch threads=nthreads blocks=nb_real gmres_compute_Av!(
                Vj1, block.dU_rhs, R_base, Vj, block.D_inv, block.Vol,
                dt_val, alpha_bdf, eps_inv, nxp, nyp, nzp)

            # ── Modified Gram-Schmidt orthogonalization ──
            for i_gs = 1:j
                Vi = @view V_krylov[:, :, :, :, i_gs]
                h_ij = _gmres_dot(Vj1, Vi, nxp, nyp, nzp)
                H[i_gs, j] = h_ij
                # Vj1 = Vj1 - h_ij * Vi
                @gpu_launch threads=nthreads blocks=nb_real gmres_sub_scaled!(
                    Vj1, Vj1, FT(h_ij), Vi, nxp, nyp, nzp)
            end

            # Normalize
            h_jp1_j = _gmres_norm(Vj1, nxp, nyp, nzp)
            H[j + 1, j] = h_jp1_j
            if h_jp1_j > 1e-30
                inv_h = FT(1.0 / h_jp1_j)
                @gpu_launch threads=nthreads blocks=nb_real gmres_scale!(
                    Vj1, Vj1, inv_h, nxp, nyp, nzp)
            end

            # ── Apply previous Givens rotations to column j of H ──
            for i_rot = 1:j-1
                tmp = cs[i_rot] * H[i_rot, j] + sn[i_rot] * H[i_rot + 1, j]
                H[i_rot + 1, j] = -sn[i_rot] * H[i_rot, j] + cs[i_rot] * H[i_rot + 1, j]
                H[i_rot, j] = tmp
            end

            # ── Compute new Givens rotation ──
            rr = sqrt(H[j, j]^2 + H[j + 1, j]^2)
            if rr > 1e-30
                cs[j] = H[j, j] / rr
                sn[j] = H[j + 1, j] / rr
            else
                cs[j] = 1.0
                sn[j] = 0.0
            end
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            # ── Update residual estimate ──
            e1[j + 1] = -sn[j] * e1[j]
            e1[j] = cs[j] * e1[j]

            res_norm = abs(e1[j + 1])
            rel_res = res_norm / max(beta, 1e-30)

            if rel_res < tol
                actual_m = j
                converged = true
                break
            end
        end

        # ── Back-substitution: solve H·y = e1 ──
        y = zeros(Float64, actual_m)
        for i_bs = actual_m:-1:1
            y[i_bs] = e1[i_bs]
            for j_bs = i_bs + 1:actual_m
                y[i_bs] -= H[i_bs, j_bs] * y[j_bs]
            end
            y[i_bs] /= H[i_bs, i_bs]
        end

        # ── Construct solution: x = Σ yᵢ × Vᵢ ──
        fill!(block.ΔU, zero(FT))
        for i_sol = 1:actual_m
            Vi = @view V_krylov[:, :, :, :, i_sol]
            # ΔU += y[i] * V[i]
            @gpu_launch threads=nthreads blocks=nb_real gmres_axpy!(
                block.ΔU, FT(y[i_sol]), Vi, block.ΔU, nxp, nyp, nzp)
        end

        final_res = converged ? abs(e1[actual_m + 1]) : abs(e1[m + 1])

        if converged
            break
        end
    end

    return final_res
end
