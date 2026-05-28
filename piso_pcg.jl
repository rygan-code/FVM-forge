# piso_pcg.jl — Preconditioned Conjugate Gradient solver for PISO pressure Poisson
#
# Solves: L(p') = rhs  where L is the FVM Laplacian (Σ D_f (p'_N - p'_P))
# Preconditioner: Jacobi (P = diag(a_P)^{-1})
#
# GPU kernels for:
#   1. Laplacian (mat-vec): r = L(p) - rhs
#   2. Jacobi preconditioner: z = r / a_P
#   3. Vector operations: axpy, dot, norm
#
# Host-side driver: pcg_solve! orchestrates the CG iteration.
#
# NOTE: NG is defined globally in the run config.

# ═══════════════════════════════════════════════════════════════
# 1. Laplacian kernel: out[i,j,k] = Σ D_f × (p'_N - p'_P)
#    Same D_f as piso_poisson_rb_sor! for perfect consistency.
# ═══════════════════════════════════════════════════════════════

function pcg_laplacian!(Lp, p_prime, Vol, Areai, Areaj, Areak, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end

    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds vol_inv_P = Vol[ii, jj, kk]
    V_P = one(FT) / (vol_inv_P + FT(1.0e-30))

    # I-direction
    @inbounds A_iL = Areai[ii, jj, kk];   @inbounds A_iR = Areai[ii+1, jj, kk]
    V_iL = one(FT) / (@inbounds Vol[ii-1, jj, kk] + FT(1.0e-30))
    V_iR = one(FT) / (@inbounds Vol[ii+1, jj, kk] + FT(1.0e-30))
    D_iL = FT(2.0) * A_iL * A_iL / (V_iL + V_P)
    D_iR = FT(2.0) * A_iR * A_iR / (V_P + V_iR)

    # J-direction
    @inbounds A_jB = Areaj[ii, jj, kk];   @inbounds A_jT = Areaj[ii, jj+1, kk]
    V_jB = one(FT) / (@inbounds Vol[ii, jj-1, kk] + FT(1.0e-30))
    V_jT = one(FT) / (@inbounds Vol[ii, jj+1, kk] + FT(1.0e-30))
    D_jB = FT(2.0) * A_jB * A_jB / (V_jB + V_P)
    D_jT = FT(2.0) * A_jT * A_jT / (V_P + V_jT)

    # K-direction
    @inbounds A_kD = Areak[ii, jj, kk];   @inbounds A_kU = Areak[ii, jj, kk+1]
    V_kD = one(FT) / (@inbounds Vol[ii, jj, kk-1] + FT(1.0e-30))
    V_kU = one(FT) / (@inbounds Vol[ii, jj, kk+1] + FT(1.0e-30))
    D_kD = FT(2.0) * A_kD * A_kD / (V_kD + V_P)
    D_kU = FT(2.0) * A_kU * A_kU / (V_P + V_kU)

    @inbounds nb_sum = D_iL * p_prime[ii-1, jj, kk] + D_iR * p_prime[ii+1, jj, kk] +
                       D_jB * p_prime[ii, jj-1, kk] + D_jT * p_prime[ii, jj+1, kk] +
                       D_kD * p_prime[ii, jj, kk-1] + D_kU * p_prime[ii, jj, kk+1]

    a_P = D_iL + D_iR + D_jB + D_jT + D_kD + D_kU

    # Laplacian(p') = nb_sum - a_P * p'_P
    @inbounds Lp[i, j, k] = nb_sum - a_P * p_prime[ii, jj, kk]
    return
end

# ═══════════════════════════════════════════════════════════════
# 2. Compute residual: r = rhs*V_P - L(p')   [note: rhs*V_P = S_V]
#    Or equivalently: r = S_V - L(p')
# ═══════════════════════════════════════════════════════════════

function pcg_compute_residual!(r, p_prime, rhs, Vol, Areai, Areaj, Areak, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end

    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds vol_inv_P = Vol[ii, jj, kk]
    V_P = one(FT) / (vol_inv_P + FT(1.0e-30))

    @inbounds A_iL = Areai[ii, jj, kk];   @inbounds A_iR = Areai[ii+1, jj, kk]
    V_iL = one(FT) / (@inbounds Vol[ii-1, jj, kk] + FT(1.0e-30))
    V_iR = one(FT) / (@inbounds Vol[ii+1, jj, kk] + FT(1.0e-30))
    D_iL = FT(2.0) * A_iL * A_iL / (V_iL + V_P)
    D_iR = FT(2.0) * A_iR * A_iR / (V_P + V_iR)

    @inbounds A_jB = Areaj[ii, jj, kk];   @inbounds A_jT = Areaj[ii, jj+1, kk]
    V_jB = one(FT) / (@inbounds Vol[ii, jj-1, kk] + FT(1.0e-30))
    V_jT = one(FT) / (@inbounds Vol[ii, jj+1, kk] + FT(1.0e-30))
    D_jB = FT(2.0) * A_jB * A_jB / (V_jB + V_P)
    D_jT = FT(2.0) * A_jT * A_jT / (V_P + V_jT)

    @inbounds A_kD = Areak[ii, jj, kk];   @inbounds A_kU = Areak[ii, jj, kk+1]
    V_kD = one(FT) / (@inbounds Vol[ii, jj, kk-1] + FT(1.0e-30))
    V_kU = one(FT) / (@inbounds Vol[ii, jj, kk+1] + FT(1.0e-30))
    D_kD = FT(2.0) * A_kD * A_kD / (V_kD + V_P)
    D_kU = FT(2.0) * A_kU * A_kU / (V_P + V_kU)

    a_P = D_iL + D_iR + D_jB + D_jT + D_kD + D_kU
    @inbounds nb_sum = D_iL * p_prime[ii-1, jj, kk] + D_iR * p_prime[ii+1, jj, kk] +
                       D_jB * p_prime[ii, jj-1, kk] + D_jT * p_prime[ii, jj+1, kk] +
                       D_kD * p_prime[ii, jj, kk-1] + D_kU * p_prime[ii, jj, kk+1]

    Lp = nb_sum - a_P * p_prime[ii, jj, kk]
    @inbounds S_V = rhs[i, j, k] * V_P

    # CG residual: r = b - A*x where A = -L (PSD), b = -S_V
    # r = -S_V - (-L(p')) = -S_V + (nb_sum - a_P*p_P) = Lp - S_V
    @inbounds r[i, j, k] = Lp - S_V
    return
end

# ═══════════════════════════════════════════════════════════════
# 3. Jacobi preconditioner: z = r / a_P
# ═══════════════════════════════════════════════════════════════

function pcg_jacobi_precond!(z, r, Vol, Areai, Areaj, Areak, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end

    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds vol_inv_P = Vol[ii, jj, kk]
    V_P = one(FT) / (vol_inv_P + FT(1.0e-30))

    @inbounds A_iL = Areai[ii, jj, kk];   @inbounds A_iR = Areai[ii+1, jj, kk]
    V_iL = one(FT) / (@inbounds Vol[ii-1, jj, kk] + FT(1.0e-30))
    V_iR = one(FT) / (@inbounds Vol[ii+1, jj, kk] + FT(1.0e-30))
    D_iL = FT(2.0) * A_iL * A_iL / (V_iL + V_P)
    D_iR = FT(2.0) * A_iR * A_iR / (V_P + V_iR)

    @inbounds A_jB = Areaj[ii, jj, kk];   @inbounds A_jT = Areaj[ii, jj+1, kk]
    V_jB = one(FT) / (@inbounds Vol[ii, jj-1, kk] + FT(1.0e-30))
    V_jT = one(FT) / (@inbounds Vol[ii, jj+1, kk] + FT(1.0e-30))
    D_jB = FT(2.0) * A_jB * A_jB / (V_jB + V_P)
    D_jT = FT(2.0) * A_jT * A_jT / (V_P + V_jT)

    @inbounds A_kD = Areak[ii, jj, kk];   @inbounds A_kU = Areak[ii, jj, kk+1]
    V_kD = one(FT) / (@inbounds Vol[ii, jj, kk-1] + FT(1.0e-30))
    V_kU = one(FT) / (@inbounds Vol[ii, jj, kk+1] + FT(1.0e-30))
    D_kD = FT(2.0) * A_kD * A_kD / (V_kD + V_P)
    D_kU = FT(2.0) * A_kU * A_kU / (V_P + V_kU)

    a_P = D_iL + D_iR + D_jB + D_jT + D_kD + D_kU

    @inbounds z[i, j, k] = r[i, j, k] / (a_P + FT(1.0e-30))
    return
end
# ═══════════════════════════════════════════════════════════════
# 3b. SOR-preconditioner helpers
#     Convert CG residual r to SOR-compatible RHS format.
#     We want to solve A*z = r where A = -L.
#     SOR kernel solves L(z) = rhs*V_P, so -L(z) = -rhs*V_P.
#     Need: -rhs*V_P = r → rhs = -r / V_P = -r * vol_inv.
# ═══════════════════════════════════════════════════════════════

function pcg_prepare_sor_rhs!(sor_rhs, r, Vol, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds sor_rhs[i, j, k] = -r[i, j, k] * Vol[ii, jj, kk]
    return
end

# Extract real cells from ghost-padded array
function pcg_extract_from_padded!(z, z_padded, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds z[i, j, k] = z_padded[ii, jj, kk]
    return
end

# ═══════════════════════════════════════════════════════════════
# 4. Vector operations (all operate on real-cell arrays nxp×nyp×nzp)
# ═══════════════════════════════════════════════════════════════

# p = p + alpha * d  (update solution in ghost-padded array)
function pcg_update_solution!(p_prime, d, alpha, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds p_prime[ii, jj, kk] += alpha * d[ii, jj, kk]
    return
end

# r = r - alpha * q  (update residual, real-cell array)
function pcg_update_residual!(r, q, alpha, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    @inbounds r[i, j, k] -= alpha * q[i, j, k]
    return
end

# d = z + beta * d  (update search direction in ghost-padded array)
function pcg_update_direction!(d, z, beta, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds d[ii, jj, kk] = z[i, j, k] + beta * d[ii, jj, kk]
    return
end

# Copy z (real) to d (ghost-padded) for initial direction
function pcg_copy_to_padded!(d, z, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds d[ii, jj, kk] = z[i, j, k]
    return
end

# Compute L(d) → q  (Laplacian of search direction, output to real-cell array)
function pcg_laplacian_to_real!(q, d, Vol, Areai, Areaj, Areak, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end

    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds vol_inv_P = Vol[ii, jj, kk]
    V_P = one(FT) / (vol_inv_P + FT(1.0e-30))

    @inbounds A_iL = Areai[ii, jj, kk];   @inbounds A_iR = Areai[ii+1, jj, kk]
    V_iL = one(FT) / (@inbounds Vol[ii-1, jj, kk] + FT(1.0e-30))
    V_iR = one(FT) / (@inbounds Vol[ii+1, jj, kk] + FT(1.0e-30))
    D_iL = FT(2.0) * A_iL * A_iL / (V_iL + V_P)
    D_iR = FT(2.0) * A_iR * A_iR / (V_P + V_iR)

    @inbounds A_jB = Areaj[ii, jj, kk];   @inbounds A_jT = Areaj[ii, jj+1, kk]
    V_jB = one(FT) / (@inbounds Vol[ii, jj-1, kk] + FT(1.0e-30))
    V_jT = one(FT) / (@inbounds Vol[ii, jj+1, kk] + FT(1.0e-30))
    D_jB = FT(2.0) * A_jB * A_jB / (V_jB + V_P)
    D_jT = FT(2.0) * A_jT * A_jT / (V_P + V_jT)

    @inbounds A_kD = Areak[ii, jj, kk];   @inbounds A_kU = Areak[ii, jj, kk+1]
    V_kD = one(FT) / (@inbounds Vol[ii, jj, kk-1] + FT(1.0e-30))
    V_kU = one(FT) / (@inbounds Vol[ii, jj, kk+1] + FT(1.0e-30))
    D_kD = FT(2.0) * A_kD * A_kD / (V_kD + V_P)
    D_kU = FT(2.0) * A_kU * A_kU / (V_P + V_kU)

    a_P = D_iL + D_iR + D_jB + D_jT + D_kD + D_kU
    @inbounds nb_sum = D_iL * d[ii-1, jj, kk] + D_iR * d[ii+1, jj, kk] +
                       D_jB * d[ii, jj-1, kk] + D_jT * d[ii, jj+1, kk] +
                       D_kD * d[ii, jj, kk-1] + D_kU * d[ii, jj, kk+1]

    # L(d) = nb_sum - a_P * d_P  (note: NEGATIVE definite for our convention)
    # We want A*d where A*p = b means L(p) = S_V.
    # Since we solve: find p such that -L(p) = -S_V (positive definite)
    # → A = -L, Ax = b means -L(p) = -S_V
    # But our residual is r = S_V - L(p), so:
    # q = -L(d) = -(nb_sum - a_P * d_P) = a_P * d_P - nb_sum
    @inbounds q[i, j, k] = a_P * d[ii, jj, kk] - nb_sum
    return
end

# ═══════════════════════════════════════════════════════════════
# 5. Zero-allocation helper kernels for PCG driver
#    (avoid GPU temporary arrays from broadcast .* and .-=)
# ═══════════════════════════════════════════════════════════════

# tmp[i,j,k] = a[i,j,k] * b[i,j,k]  (element-wise product → preallocated buffer)
function pcg_multiply_fields!(tmp, a, b, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    # Write 0 outside bounds so that sum() over full array safely ignores padding
    if i > nxp || j > nyp || k > nzp
        if i <= size(tmp, 1) && j <= size(tmp, 2) && k <= size(tmp, 3)
            @inbounds tmp[i, j, k] = zero(eltype(tmp))
        end
        return
    end
    @inbounds tmp[i, j, k] = a[i, j, k] * b[i, j, k]
    return
end

# tmp[i,j,k] = a[i+NG,j+NG,k+NG] * b[i,j,k]  (ghost-padded × real-cell)
function pcg_multiply_padded_real!(tmp, a_padded, b, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        if i <= size(tmp, 1) && j <= size(tmp, 2) && k <= size(tmp, 3)
            @inbounds tmp[i, j, k] = zero(eltype(tmp))
        end
        return
    end
    ii = i + NG; jj = j + NG; kk = k + NG
    @inbounds tmp[i, j, k] = a_padded[ii, jj, kk] * b[i, j, k]
    return
end

# field[i,j,k] -= val  (subtract scalar in-place, no GPU temporary)
function pcg_subtract_scalar!(field, val, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    @inbounds field[i, j, k] -= val
    return
end

# tmp[i,j,k] = abs(a[i,j,k]) (copy absolute values, out-of-bounds = 0)
function pcg_copy_abs!(tmp, a, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        if i <= size(tmp, 1) && j <= size(tmp, 2) && k <= size(tmp, 3)
            @inbounds tmp[i, j, k] = zero(eltype(tmp))
        end
        return
    end
    @inbounds tmp[i, j, k] = abs(a[i, j, k])
    return
end

# tmp[i,j,k] = a[i,j,k] (copy elements, out-of-bounds = 0)
function pcg_copy_valid!(tmp, a, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        if i <= size(tmp, 1) && j <= size(tmp, 2) && k <= size(tmp, 3)
            @inbounds tmp[i, j, k] = zero(eltype(tmp))
        end
        return
    end
    @inbounds tmp[i, j, k] = a[i, j, k]
    return
end

# piso_pcg_driver.jl -- Host-side PCG Poisson solve driver for PISO
#
# Solves: A*x = b where A = -L (positive semi-definite), x = p'
#   With pure Neumann BCs, A has a rank-1 null space (constant vector).
#   We use deflated PCG: project out the null-space component.
#
# Preconditioner options (set via piso_pcg_precond):
#   :jacobi  — z = r / a_P  (simple, but slow convergence)
#   :sor     — N sweeps of Red-Black SOR to approximately solve A*z = r
#              (much stronger, captures low-frequency modes)
#   :mg      — Geometric Multigrid V-cycle (strongest, O(1) condition number)
#
# TRUE ZERO-ALLOCATION design:
#   - All GPU temporary arrays are pre-allocated by caller
#   - All reductions (sum/max) are done on CPU via pre-allocated staging buffer
#     (eliminates GPU mapreduce workspace allocations from AMDGPU/GPUArrays)
#   - All MPI collectives use in-place MPI.Allreduce! with pre-allocated buffers
#     (eliminates return-value allocations from MPI.Allreduce)
#   - reshape() for ghost exchange uses a pre-allocated 4D buffer
#     (eliminates ReshapedArray wrapper allocations)
#
# Uses kernels from piso_pcg.jl, piso.jl, piso_multigrid.jl

function piso_pcg_solve!(shared_p_prime, shared_dpdt, pcg_r, pcg_z, pcg_d, pcg_q,
                         pcg_ztmp,   # ghost-padded workspace for SOR/MG preconditioner
                         mg_levels_dict,  # Dict{bid => Vector{MGLevel}} or nothing
                         blocks, nthreads, face_bc, block_comms,
                         max_iters, tol, tt, world_rank,
                         pcg_reduce_cpu,   # CPU staging buffer for GPU→CPU reductions (same shape as pcg_q)
                         pcg_mpi_sbuf,     # pre-allocated MPI send buffer (Vector{Float64}, length≥1)
                         pcg_mpi_rbuf,     # pre-allocated MPI recv buffer (Vector{Float64}, length≥1)
                         pcg_ghost_4d)     # pre-allocated 4D GPU buffer for ghost exchange reshape

    # ── Config ──
    _precond_type = isdefined(@__MODULE__, :piso_pcg_precond) ? piso_pcg_precond : :sor
    _sor_sweeps = isdefined(@__MODULE__, :piso_pcg_sor_sweeps) ? piso_pcg_sor_sweeps : 5
    _sor_omega = isdefined(@__MODULE__, :piso_sor_omega) ? piso_sor_omega : FT(1.7)

    # ── Helper: apply BC + MPI ghost exchange (ZERO-ALLOCATION) ──
    # Uses pre-allocated pcg_ghost_4d to avoid reshape() wrapper allocation.
    function _ghost_exchange!(field)
        for (bid, b) in blocks
            piso_apply_pprime_bc!(b, bid, face_bc, field)
            # Copy 3D field → pre-allocated 4D buffer (no reshape allocation)
            copyto!(pcg_ghost_4d, field)
            exchange_ghost(pcg_ghost_4d, 1, block_comms[bid], b.Nx, b.Ny, b.Nz,
                b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
                b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
                b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
                sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
                rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2)
            # Copy back: 4D buffer → 3D field
            copyto!(field, pcg_ghost_4d)
        end
    end

    # ── Helper: dot product (TRUE ZERO-ALLOCATION) ──
    # GPU kernel writes to tmp_buf → gpu_sync → D2H copy to CPU → CPU sum → MPI in-place
    function _dot_product(a_field, b_field, tmp_buf)
        local_sum = Float64(0.0)
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(size(tmp_buf,1), nthreads[1]), cld(size(tmp_buf,2), nthreads[2]), cld(size(tmp_buf,3), nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_f pcg_multiply_fields!(
                tmp_buf, a_field, b_field, nxp, nyp, nzp)
            gpu_sync()
            copyto!(pcg_reduce_cpu, tmp_buf)
            local_sum += Float64(sum(pcg_reduce_cpu))
        end
        pcg_mpi_sbuf[1] = local_sum
        MPI.Allreduce!(pcg_mpi_sbuf, pcg_mpi_rbuf, MPI.SUM, MPI.COMM_WORLD)
        return pcg_mpi_rbuf[1]
    end

    # ── Helper: dot product for ghost-padded × real-cell (TRUE ZERO-ALLOCATION) ──
    function _dot_product_padded(a_padded, b_field, tmp_buf)
        local_sum = Float64(0.0)
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(size(tmp_buf,1), nthreads[1]), cld(size(tmp_buf,2), nthreads[2]), cld(size(tmp_buf,3), nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_f pcg_multiply_padded_real!(
                tmp_buf, a_padded, b_field, nxp, nyp, nzp)
            gpu_sync()
            copyto!(pcg_reduce_cpu, tmp_buf)
            local_sum += Float64(sum(pcg_reduce_cpu))
        end
        pcg_mpi_sbuf[1] = local_sum
        MPI.Allreduce!(pcg_mpi_sbuf, pcg_mpi_rbuf, MPI.SUM, MPI.COMM_WORLD)
        return pcg_mpi_rbuf[1]
    end

    # ── Helper: max|field| (TRUE ZERO-ALLOCATION) ──
    function _max_abs(field, tmp_buf)
        local_max = Float64(0.0)
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(size(tmp_buf,1), nthreads[1]), cld(size(tmp_buf,2), nthreads[2]), cld(size(tmp_buf,3), nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_f pcg_copy_abs!(
                tmp_buf, field, nxp, nyp, nzp)
            gpu_sync()
            copyto!(pcg_reduce_cpu, tmp_buf)
            local_max = max(local_max, Float64(maximum(pcg_reduce_cpu)))
        end
        pcg_mpi_sbuf[1] = local_max
        MPI.Allreduce!(pcg_mpi_sbuf, pcg_mpi_rbuf, MPI.MAX, MPI.COMM_WORLD)
        return FT(pcg_mpi_rbuf[1])
    end

    # ── Helper: null-space deflation (TRUE ZERO-ALLOCATION) ──
    total_cells = Int64(0)
    for (bid, b) in blocks
        total_cells += b.Nx * b.Ny * b.Nz
    end
    pcg_mpi_sbuf[1] = Float64(total_cells)
    MPI.Allreduce!(pcg_mpi_sbuf, pcg_mpi_rbuf, MPI.SUM, MPI.COMM_WORLD)
    total_cells = Int64(pcg_mpi_rbuf[1])

    function _deflate!(field, tmp_buf)
        local_sum = Float64(0.0)
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(size(tmp_buf,1), nthreads[1]), cld(size(tmp_buf,2), nthreads[2]), cld(size(tmp_buf,3), nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_f pcg_copy_valid!(
                tmp_buf, field, nxp, nyp, nzp)
            gpu_sync()
            copyto!(pcg_reduce_cpu, tmp_buf)
            local_sum += Float64(sum(pcg_reduce_cpu))
        end
        pcg_mpi_sbuf[1] = local_sum
        MPI.Allreduce!(pcg_mpi_sbuf, pcg_mpi_rbuf, MPI.SUM, MPI.COMM_WORLD)
        mean_val = FT(pcg_mpi_rbuf[1] / total_cells)
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_f pcg_subtract_scalar!(
                field, mean_val, nxp, nyp, nzp)
        end
    end

    # ── Preconditioner: z = M^{-1} r ──
    function _apply_precond!()
        if _precond_type == :mg && mg_levels_dict !== nothing
            mg_vcycle_precond!(pcg_z, pcg_r, pcg_q, pcg_ztmp,
                              mg_levels_dict, blocks, nthreads,
                              _ghost_exchange!, block_comms)
        elseif _precond_type == :sor
            for (bid, b) in blocks
                nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
                nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
                @gpu_launch threads=nthreads blocks=nb_f pcg_prepare_sor_rhs!(
                    pcg_q, pcg_r, b.Vol, nxp, nyp, nzp)
            end

            for (bid, b) in blocks
                Nx_t = b.Nx + 2*NG; Ny_t = b.Ny + 2*NG; Nz_t = b.Nz + 2*NG
                nb_tot = (cld(Nx_t, nthreads[1]), cld(Ny_t, nthreads[2]), cld(Nz_t, nthreads[3]))
                @gpu_launch threads=nthreads blocks=nb_tot piso_zero_field!(pcg_ztmp, Int32(Nx_t), Int32(Ny_t), Int32(Nz_t))
            end

            for sweep = 1:_sor_sweeps
                for color = Int32(0):Int32(1)
                    for (bid, b) in blocks
                        nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
                        nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
                        @gpu_launch threads=nthreads blocks=nb_f piso_poisson_rb_sor!(
                            pcg_ztmp, pcg_q, b.Vol, b.Areai, b.Areaj, b.Areak,
                            nxp, nyp, nzp, _sor_omega, color)
                    end
                end
                for color = Int32(1):-Int32(1):Int32(0)
                    for (bid, b) in blocks
                        nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
                        nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
                        @gpu_launch threads=nthreads blocks=nb_f piso_poisson_rb_sor!(
                            pcg_ztmp, pcg_q, b.Vol, b.Areai, b.Areaj, b.Areak,
                            nxp, nyp, nzp, _sor_omega, color)
                    end
                end
                if sweep == _sor_sweeps
                    _ghost_exchange!(pcg_ztmp)
                end
            end

            for (bid, b) in blocks
                nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
                nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
                @gpu_launch threads=nthreads blocks=nb_f pcg_extract_from_padded!(pcg_z, pcg_ztmp, nxp, nyp, nzp)
            end
        else
            for (bid, b) in blocks
                nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
                nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
                @gpu_launch threads=nthreads blocks=nb_f pcg_jacobi_precond!(
                    pcg_z, pcg_r, b.Vol, b.Areai, b.Areaj, b.Areak, nxp, nyp, nzp)
            end
        end
        _deflate!(pcg_z, pcg_q)
    end

    # ═══════════════════════════════════════════════════════════
    # PCG Main Loop
    # ═══════════════════════════════════════════════════════════

    # Step 0: p' ghost exchange
    _ghost_exchange!(shared_p_prime)

    # Step 1: r = b - A*x = Lp - S_V
    for (bid, b) in blocks
        nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
        nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
        @gpu_launch threads=nthreads blocks=nb_f pcg_compute_residual!(
            pcg_r, shared_p_prime, shared_dpdt, b.Vol, b.Areai, b.Areaj, b.Areak,
            nxp, nyp, nzp)
    end
    _deflate!(pcg_r, pcg_q)

    # Step 2: z = M^{-1} r
    _apply_precond!()

    # Step 3: d = z
    for (bid, b) in blocks
        nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
        nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
        @gpu_launch threads=nthreads blocks=nb_f pcg_copy_to_padded!(pcg_d, pcg_z, nxp, nyp, nzp)
    end
    _ghost_exchange!(pcg_d)

    # Step 4: rz = r . z
    rz = _dot_product(pcg_r, pcg_z, pcg_q)

    _print_freq = 20
    final_res = FT(0.0)

    for pcg_iter = 1:max_iters
        # q = A * d = -L(d)
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_f pcg_laplacian_to_real!(
                pcg_q, pcg_d, b.Vol, b.Areai, b.Areaj, b.Areak, nxp, nyp, nzp)
        end
        _deflate!(pcg_q, pcg_z)

        # alpha = rz / (d . q)
        dq = _dot_product_padded(pcg_d, pcg_q, pcg_z)
        alpha = FT(rz / (dq + 1.0e-300))

        # x += alpha * d
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_f pcg_update_solution!(shared_p_prime, pcg_d, alpha, nxp, nyp, nzp)
        end

        # r -= alpha * q
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_f pcg_update_residual!(pcg_r, pcg_q, alpha, nxp, nyp, nzp)
        end

        # Convergence check
        if pcg_iter % _print_freq == 0 || pcg_iter == max_iters
            _diag_res = _max_abs(pcg_r, pcg_q)
            final_res = _diag_res

            if world_rank == 0 && (tt <= 20 || tt % 100 == 0)
                @printf("        [PCG iter=%d] max_res=%.3e\n", pcg_iter, _diag_res)
                flush(stdout)
            end
            if _diag_res < tol
                if world_rank == 0 && (tt <= 20 || tt % 100 == 0)
                    @printf("        [PCG CONVERGED at iter=%d] max_res=%.3e < tol=%.1e\n",
                        pcg_iter, _diag_res, tol)
                    flush(stdout)
                end
                break
            end
        end

        # z = M^{-1} r (preconditioner)
        _apply_precond!()

        # beta = rz_new / rz_old
        rz_new = _dot_product(pcg_r, pcg_z, pcg_q)
        beta = FT(rz_new / (rz + 1.0e-300))
        rz = rz_new

        # d = z + beta * d
        for (bid, b) in blocks
            nxp = b.Nx; nyp = b.Ny; nzp = b.Nz
            nb_f = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_f pcg_update_direction!(pcg_d, pcg_z, beta, nxp, nyp, nzp)
        end
        _ghost_exchange!(pcg_d)
    end

    # Final ghost exchange for p_prime
    _ghost_exchange!(shared_p_prime)
    return final_res
end
