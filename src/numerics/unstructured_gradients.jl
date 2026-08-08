# unstruct_gradient.jl — LSQ gradient and Venkatakrishnan limiter for unstructured FVM
# Part of the second-order unstructured FVM branch
#
# Shared infrastructure (must be included BEFORE this file):
#   gpu_backend.jl, physics.jl, unstruct_mesh.jl
#
# Gradients store physical derivatives. The limiter stores a direct [0, 1]
# multiplier used in center-to-face reconstruction.

# ═════════════════════════════════════════════════════════════
# LSQ gradient kernel (Phase 4 — full implementation)
# ═════════════════════════════════════════════════════════════
# For each cell c, solve the 3×3 least-squares system:
#   A · g = b
# where:
#   A = Σ_i w_i (Δr_i ⊗ Δr_i)   (3×3 symmetric matrix)
#   b = Σ_i w_i Δφ_i Δr_i       (3-vector, per variable)
#   g = ∇φ_c                     (gradient, 3-vector)
#
# The weights w_i = 1/|Δr_i|² are precomputed in cn_w.

function lsq_gradient_kernel!(grad, Q,
                               cn_offset, cn_list, cn_dx, cn_dy, cn_dz, cn_w,
                               ncell::Int)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if c > ncell
        return
    end

    @inbounds n_start = cn_offset[c]
    @inbounds n_end = cn_offset[c + 1] - Int32(1)
    n_neigh = n_end - n_start + Int32(1)

    if n_neigh < Int32(3)
        # Not enough neighbors for 3D LSQ — set gradient to zero
        @inbounds for v in 1:Nprim
            grad[c, v, 1] = zero(FT)
            grad[c, v, 2] = zero(FT)
            grad[c, v, 3] = zero(FT)
        end
        return
    end

    # Build 3×3 LSQ matrix A = Σ w_i (Δr_i ⊗ Δr_i)
    A11 = zero(FT); A12 = zero(FT); A13 = zero(FT)
    A22 = zero(FT); A23 = zero(FT)
    A33 = zero(FT)

    @inbounds for ni in n_start:n_end
        w = cn_w[ni]
        dx = cn_dx[ni]; dy = cn_dy[ni]; dz = cn_dz[ni]
        A11 += w * dx * dx
        A12 += w * dx * dy
        A13 += w * dx * dz
        A22 += w * dy * dy
        A23 += w * dy * dz
        A33 += w * dz * dz
    end

    # Invert 3×3 symmetric matrix (A is symmetric positive definite)
    # Using cofactor / determinant method
    det = A11 * (A22 * A33 - A23 * A23) -
          A12 * (A12 * A33 - A23 * A13) +
          A13 * (A12 * A23 - A22 * A13)

    if abs(det) < eps(FT) * FT(1.0e10)
        # Singular — set gradient to zero
        @inbounds for v in 1:Nprim
            grad[c, v, 1] = zero(FT)
            grad[c, v, 2] = zero(FT)
            grad[c, v, 3] = zero(FT)
        end
        return
    end

    inv_det = one(FT) / det
    # Cofactor matrix (symmetric)
    B11 =  (A22 * A33 - A23 * A23) * inv_det
    B12 = -(A12 * A33 - A23 * A13) * inv_det
    B13 =  (A12 * A23 - A22 * A13) * inv_det
    B22 =  (A11 * A33 - A13 * A13) * inv_det
    B23 = -(A11 * A23 - A12 * A13) * inv_det
    B33 =  (A11 * A22 - A12 * A12) * inv_det

    # For each primitive variable, compute gradient
    @inbounds for v in 1:Nprim
        φ_c = Q[c, v]

        # Build RHS: b = Σ w_i (φ_ni - φ_c) Δr_i
        b1 = zero(FT); b2 = zero(FT); b3 = zero(FT)
        for ni in n_start:n_end
            w = cn_w[ni]
            dx = cn_dx[ni]; dy = cn_dy[ni]; dz = cn_dz[ni]
            nb = cn_list[ni]
            dphi = Q[nb, v] - φ_c
            b1 += w * dphi * dx
            b2 += w * dphi * dy
            b3 += w * dphi * dz
        end

        # g = A⁻¹ · b
        grad[c, v, 1] = B11 * b1 + B12 * b2 + B13 * b3
        grad[c, v, 2] = B12 * b1 + B22 * b2 + B23 * b3
        grad[c, v, 3] = B13 * b1 + B23 * b2 + B33 * b3
    end
    return
end

# ═════════════════════════════════════════════════════════════
# Venkatakrishnan limiter kernel (Phase 4)
# ═════════════════════════════════════════════════════════════
# For an unlimited face increment Δ and the allowed signed increment Δm,
# the Venkatakrishnan factor is
#   (Δm² + 2ΔmΔ + ε²) / (Δm² + ΔmΔ + 2Δ² + ε²).

@inline function venkatakrishnan_factor(allowed, increment, epsilon_squared)
    numerator = allowed*allowed + FT(2)*allowed*increment + epsilon_squared
    denominator = allowed*allowed + allowed*increment +
        FT(2)*increment*increment + epsilon_squared
    return clamp(numerator/max(denominator,eps(FT)),zero(FT),one(FT))
end

function venkat_limiter_kernel!(limiter_field, grad, Q,
                                 cn_offset, cn_list, cn_dx, cn_dy, cn_dz,
                                 ncell::Int, venk_k2::FT)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if c > ncell
        return
    end

    @inbounds n_start = cn_offset[c]
    @inbounds n_end = cn_offset[c + 1] - Int32(1)
    local_length_squared = typemax(FT)
    for ni in n_start:n_end
        distance_squared = cn_dx[ni]^2 + cn_dy[ni]^2 + cn_dz[ni]^2
        if distance_squared > eps(FT)
            local_length_squared = min(local_length_squared,distance_squared)
        end
    end
    if !isfinite(local_length_squared)
        local_length_squared = one(FT)
    end
    epsilon_squared = venk_k2*local_length_squared*sqrt(local_length_squared)

    @inbounds for v in 1:Nprim
        value_cell = Q[c, v]
        dmax = zero(FT)
        dmin = zero(FT)
        for ni in n_start:n_end
            neighbor = cn_list[ni]
            difference = Q[neighbor, v] - value_cell
            dmax = max(dmax,difference)
            dmin = min(dmin,difference)
        end
        gx = grad[c, v, 1]
        gy = grad[c, v, 2]
        gz = grad[c, v, 3]
        limiter_value = one(FT)
        tolerance = sqrt(eps(FT))*max(abs(value_cell),one(FT))
        for ni in n_start:n_end
            # Neighbor midpoints are the reconstruction points for an
            # orthogonal dual face and remain a bounded proxy on skew meshes.
            increment = FT(0.5)*(
                gx*cn_dx[ni] + gy*cn_dy[ni] + gz*cn_dz[ni])
            if increment > tolerance
                limiter_value = min(limiter_value,venkatakrishnan_factor(
                    dmax,increment,epsilon_squared))
            elseif increment < -tolerance
                limiter_value = min(limiter_value,venkatakrishnan_factor(
                    dmin,increment,epsilon_squared))
            end
        end
        @inbounds limiter_field[c,v] = limiter_value
    end
    return
end

# ═════════════════════════════════════════════════════════════
# Wrappers
# ═════════════════════════════════════════════════════════════

const VENK_K2_DEFAULT = FT(1.0)  # Venkatakrishnan κ² parameter

function compute_lsq_gradient!(block::UnstructBlock)
    nb = cld(block.ncell, block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=nb lsq_gradient_kernel!(
        block.grad, block.Q,
        block.cn_offset, block.cn_list, block.cn_dx, block.cn_dy, block.cn_dz, block.cn_w,
        block.ncell)
    return
end

function compute_venkat_limiter!(block::UnstructBlock; venk_k2::FT=VENK_K2_DEFAULT)
    venk_k2 >= zero(FT) || throw(ArgumentError("venk_k2 must be non-negative"))
    nb = cld(block.ncell, block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=nb venkat_limiter_kernel!(
        block.reconstruction_limiter, block.grad, block.Q,
        block.cn_offset, block.cn_list, block.cn_dx, block.cn_dy, block.cn_dz,
        block.ncell, venk_k2)
    return
end
