# =============================================================================
# Butterfly (O-H) Mesh Generator for Pipe Flow (FVM version)
# 5-block topology: 1 center square + 4 surrounding annular sectors
# Outputs REAL nodes only (no ghost cells). Ghost cells are generated at runtime.
# =============================================================================
using HDF5
using WriteVTK
using LinearAlgebra

# ─── Global parameters ───
const NG::Int = 4
const Lx::Float64 = 7.5        # pipe length = 15R₀ (DNS standard)
const R0::Float64 = 0.5
const r_ratio::Float64 = 0.4
const r_inner::Float64 = r_ratio * R0
const vis::Bool = true
const compress_level::Int = 3
const FT = Float32  # Weight precision (matches solver runtime type)

# ─── Grid sizes ───
# DNS targets: Δx⁺<10, Δy⁺<0.5, rΔθ⁺<5
# u_τ ≈ 22.5 m/s, ν ≈ 0.01653 m²/s → ν/u_τ = 7.35e-4
# Δx = Lx/Nx < 10×7.35e-4 = 7.35e-3 → Nx > Lx/7.35e-3
# Δy_wall ≈ max_dr_outer < 0.5×7.35e-4 = 3.67e-4
# rΔθ = R₀(π/2)/Nz < 5×7.35e-4 = 3.67e-3 → Nz > R₀(π/2)/3.67e-3 ≈ 214
# Dimensions are now passed parametrically.

# =============================================================================
# Inner boundary mapping
# =============================================================================
function square_to_inner(s::Float64, t::Float64, r::Float64)
    alpha = 0.18
    ys = s * (1.0 + alpha * (1.0 - t^2))
    zt = t * (1.0 + alpha * (1.0 - s^2))
    return r * ys, r * zt
end

"""
    compute_tanh_param(ds_end, L_radial, N)

Solve for tanh stretching parameter `b` such that the end cell
has physical size `ds_end` across `N` cells spanning `L_radial`.
Uses Newton iteration on: 1 - tanh(b*(1-1/N))/tanh(b) = ds_end/L_radial
"""
function compute_tanh_param(ds_end::Float64, L_radial::Float64, N::Int)
    target = ds_end / L_radial   # normalized end-cell fraction
    ratio  = 1.0 - target         # = tanh(b*(1-1/N)) / tanh(b)
    s1     = 1.0 - 1.0 / N

    # Edge case: if target ≈ 1/N (uniform), return small b
    if abs(target - 1.0/N) < 1e-10
        return 0.01
    end

    b = 2.0  # initial guess
    for iter in 1:300
        tb  = tanh(b)
        tbs = tanh(b * s1)
        val = tbs / tb
        dval = (s1 / cosh(b * s1)^2 * tb - tbs / cosh(b)^2) / tb^2
        res  = val - ratio
        if abs(res) < 1e-14
            break
        end
        b -= res / dval
        b = clamp(b, 0.01, 100.0)
    end
    return b
end

"""
    two_sided_tanh(s, b_left, b_right)

Two-sided tanh stretching: clusters at both s=0 (b_left) and s=1 (b_right).
Uses a linear blend of two one-sided tanh functions.
f(0)=0, f(1)=1.  f'(0) controlled by b_left, f'(1) controlled by b_right.
"""
function two_sided_tanh(s::Float64, b_left::Float64, b_right::Float64)
    f_left  = 1.0 - tanh(b_left  * (1.0 - s)) / tanh(b_left)   # clusters at s=0
    f_right = tanh(b_right * s) / tanh(b_right)                   # clusters at s=1
    return (1.0 - s) * f_left + s * f_right
end

"""
    compute_corrected_b_right(b_left, ds_target, L_radial, N)

Iteratively find b_right so that the ACTUAL last cell of two_sided_tanh(s, b_left, b_right)
matches ds_target. This compensates for the blend distortion.
"""
function compute_corrected_b_right(b_left::Float64, ds_target::Float64, L_radial::Float64, N::Int)
    target_frac = ds_target / L_radial  # normalized target last-cell fraction
    # Initial guess from one-sided solution
    b_r = compute_tanh_param(ds_target, L_radial, N)
    
    for iter in 1:50
        # Compute actual last cell of blended function
        s_prev = Float64(N - 1) / N
        f_end  = 1.0  # f(1) = 1 always
        f_prev = two_sided_tanh(s_prev, b_left, b_r)
        actual_frac = f_end - f_prev
        
        # Adjust b_r: smaller b_r → larger last cell, larger b_r → smaller
        err = actual_frac - target_frac
        if abs(err) < 1e-12
            break
        end
        # Numerical derivative: d(actual_frac)/d(b_r)
        db = 1e-6
        f_prev_p = two_sided_tanh(s_prev, b_left, b_r + db)
        d_frac = ((1.0 - f_prev_p) - actual_frac) / db
        b_r -= err / d_frac
        b_r = clamp(b_r, 0.01, 100.0)
    end
    return b_r
end

# =============================================================================
# Grid Generation Functions — Real nodes only (Nx+1, Ny+1, Nz+1)
# =============================================================================

function generate_block0(Nx_b, Ny_b0, Nz_b0, Lx=Lx)
    Nx, Ny, Nz = Nx_b, Ny_b0, Nz_b0
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)

    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s_norm = 2.0 * Float64(j - 1) / Ny - 1.0
        t_norm = 2.0 * Float64(k - 1) / Nz - 1.0
        yc, zc = square_to_inner(s_norm, t_norm, r_inner)
        x[i, j, k] = xa
        y[i, j, k] = yc
        z[i, j, k] = zc
    end
    return x, y, z, Nx, Ny, Nz
end

function generate_block1(Nx_b, Ny_b0, Nz_b0, N_rad, dr_outer, Lx=Lx)
    Nx, Ny, Nz = Nx_b, N_rad, Nz_b0
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)
    L_rad = R0 - r_inner
    # Match actual center block boundary cell (includes alpha=0.18 deformation)
    ds_inner = 2.0 * r_inner * (1.0 + 0.18) / Ny_b0
    b_wall  = compute_tanh_param(dr_outer, L_rad, Ny)  # wall at s=0
    b_inner = compute_corrected_b_right(b_wall, ds_inner, L_rad, Ny)  # inner at s=1 (corrected)
    
    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s = Float64(j - 1) / Ny
        w_in = two_sided_tanh(s, b_wall, b_inner)  # wall s=0, inner s=1
        t_norm = 2.0 * Float64(k - 1) / Nz - 1.0
        y_in, z_in = square_to_inner(-1.0, t_norm, r_inner)
        param = Float64(k - 1) / Nz
        θ = 5*pi/4 + param * (pi/2)
        y_out = R0 * sin(θ); z_out = R0 * cos(θ)
        x[i, j, k] = xa
        y[i, j, k] = w_in * y_in + (1.0 - w_in) * y_out
        z[i, j, k] = w_in * z_in + (1.0 - w_in) * z_out
    end
    return x, y, z, Nx, Ny, Nz
end

function generate_block2(Nx_b, Ny_b0, Nz_b0, N_rad, dr_outer, Lx=Lx)
    Nx, Ny, Nz = Nx_b, N_rad, Nz_b0
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)
    L_rad = R0 - r_inner
    ds_inner = 2.0 * r_inner * (1.0 + 0.18) / Ny_b0
    b_wall  = compute_tanh_param(dr_outer, L_rad, Ny)  # wall at s=1
    # For block2: inner at s=0, wall at s=1 → two_sided_tanh(s, b_inner, b_wall)
    # The first cell (inner) is the left end → correct as right via symmetry
    b_inner = compute_corrected_b_right(b_wall, ds_inner, L_rad, Ny)
    
    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s = Float64(j - 1) / Ny
        w_in = 1.0 - two_sided_tanh(s, b_inner, b_wall)  # inner s=0, wall s=1
        t_norm = 2.0 * Float64(k - 1) / Nz - 1.0
        y_in, z_in = square_to_inner(1.0, t_norm, r_inner)
        param = Float64(k - 1) / Nz
        θ = 3*pi/4 - param * (pi/2)
        y_out = R0 * sin(θ); z_out = R0 * cos(θ)
        x[i, j, k] = xa
        y[i, j, k] = w_in * y_in + (1.0 - w_in) * y_out
        z[i, j, k] = w_in * z_in + (1.0 - w_in) * z_out
    end
    return x, y, z, Nx, Ny, Nz
end

function generate_block3(Nx_b, Ny_b0, Nz_b0, N_rad, dr_outer, Lx=Lx)
    Nx, Ny, Nz = Nx_b, Ny_b0, N_rad
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)
    L_rad = R0 - r_inner
    ds_inner = 2.0 * r_inner * (1.0 + 0.18) / Nz_b0
    b_wall  = compute_tanh_param(dr_outer, L_rad, Nz)  # wall at s=0
    b_inner = compute_corrected_b_right(b_wall, ds_inner, L_rad, Nz)  # inner at s=1 (corrected)
    
    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s = Float64(k - 1) / Nz
        w_in = two_sided_tanh(s, b_wall, b_inner)  # wall s=0, inner s=1
        s_norm = 2.0 * Float64(j - 1) / Ny - 1.0
        y_in, z_in = square_to_inner(s_norm, -1.0, r_inner)
        param = Float64(j - 1) / Ny
        θ = 5*pi/4 - param * (pi/2)
        y_out = R0 * sin(θ); z_out = R0 * cos(θ)
        x[i, j, k] = xa
        y[i, j, k] = w_in * y_in + (1.0 - w_in) * y_out
        z[i, j, k] = w_in * z_in + (1.0 - w_in) * z_out
    end
    return x, y, z, Nx, Ny, Nz
end

function generate_block4(Nx_b, Ny_b0, Nz_b0, N_rad, dr_outer, Lx=Lx)
    Nx, Ny, Nz = Nx_b, Ny_b0, N_rad
    x = zeros(Float64, Nx+1, Ny+1, Nz+1)
    y = zeros(Float64, Nx+1, Ny+1, Nz+1)
    z = zeros(Float64, Nx+1, Ny+1, Nz+1)
    L_rad = R0 - r_inner
    ds_inner = 2.0 * r_inner * (1.0 + 0.18) / Nz_b0
    b_wall  = compute_tanh_param(dr_outer, L_rad, Nz)  # wall at s=1
    b_inner = compute_corrected_b_right(b_wall, ds_inner, L_rad, Nz)
    
    for k ∈ 1:Nz+1, j ∈ 1:Ny+1, i ∈ 1:Nx+1
        xa = Lx * Float64(i - 1) / Nx
        s = Float64(k - 1) / Nz
        w_in = 1.0 - two_sided_tanh(s, b_inner, b_wall)  # inner s=0, wall s=1
        s_norm = 2.0 * Float64(j - 1) / Ny - 1.0
        y_in, z_in = square_to_inner(s_norm, 1.0, r_inner)
        param = Float64(j - 1) / Ny
        θ = 7*pi/4 + param * (pi/2)
        y_out = R0 * sin(θ); z_out = R0 * cos(θ)
        x[i, j, k] = xa
        y[i, j, k] = w_in * y_in + (1.0 - w_in) * y_out
        z[i, j, k] = w_in * z_in + (1.0 - w_in) * z_out
    end
    return x, y, z, Nx, Ny, Nz
end

# =============================================================================
# Multi-block Connectivity and Weights
# Uses real-only coordinates (no ghost padding)
# =============================================================================
function get_search_bounds(src_f, Ny_src, Nz_src, L_pack)
    # No NG offset — coordinates are real-only (1-based)
    if src_f == 3        # η- boundary: first L_pack rows in j
        return 1, L_pack, 1, Nz_src+1
    elseif src_f == 4    # η+ boundary: last L_pack rows in j
        return Ny_src+1-L_pack+1, Ny_src+1, 1, Nz_src+1
    elseif src_f == 5    # ζ- boundary: first L_pack rows in k
        return 1, Ny_src+1, 1, L_pack
    elseif src_f == 6    # ζ+ boundary: last L_pack rows in k
        return 1, Ny_src+1, Nz_src+1-L_pack+1, Nz_src+1
    end
end

function point_in_quad(p, q1, q2, q3, q4)
    cross2d(A, B, P) = (B[1] - A[1]) * (P[2] - A[2]) - (B[2] - A[2]) * (P[1] - A[1])
    c1 = cross2d(q1, q2, p); c2 = cross2d(q2, q3, p)
    c3 = cross2d(q3, q4, p); c4 = cross2d(q4, q1, p)
    return ((c1 >= -1e-10 && c2 >= -1e-10 && c3 >= -1e-10 && c4 >= -1e-10) ||
            (c1 <= 1e-10 && c2 <= 1e-10 && c3 <= 1e-10 && c4 <= 1e-10))
end

function get_bilinear_weights(p, q1, q2, q3, q4)
    s, t = 0.5, 0.5
    for iter=1:30
        N1 = (1-s)*(1-t); N2 = s*(1-t); N3 = s*t; N4 = (1-s)*t
        pt = N1.*q1 .+ N2.*q2 .+ N3.*q3 .+ N4.*q4
        dN1ds = -(1-t); dN2ds = (1-t); dN3ds = t; dN4ds = -t
        dN1dt = -(1-s); dN2dt = -s; dN3dt = s; dN4dt = (1-s)
        dpds = dN1ds.*q1 .+ dN2ds.*q2 .+ dN3ds.*q3 .+ dN4ds.*q4
        dpdt = dN1dt.*q1 .+ dN2dt.*q2 .+ dN3dt.*q3 .+ dN4dt.*q4
        J_mat = [dpds[1] dpdt[1]; dpds[2] dpdt[2]]
        res = p .- pt
        if sum(abs.(res)) < 1e-12; break; end
        try
            delta = J_mat \ res
            s += delta[1]; t += delta[2]
        catch; break; end
    end
    s = clamp(s, 0.0, 1.0); t = clamp(t, 0.0, 1.0)
    return FT((1-s)*(1-t)), FT(s*(1-t)), FT(s*t), FT((1-s)*t)
end

function build_interpolation_weights(blocks_data, out_dir)
    connections = Dict(
        (0, 3) => (1, 4), (1, 4) => (0, 3),
        (0, 4) => (2, 3), (2, 3) => (0, 4),
        (0, 5) => (3, 6), (3, 6) => (0, 5),
        (0, 6) => (4, 5), (4, 5) => (0, 6),
        (1, 5) => (3, 3), (3, 3) => (1, 5),
        (1, 6) => (4, 3), (4, 3) => (1, 6),
        (2, 5) => (3, 4), (3, 4) => (2, 5),
        (2, 6) => (4, 4), (4, 4) => (2, 6)
    )
    L_pack = 10 

    h5open(joinpath(out_dir, "interp_weights.h5"), "w") do h5f
        for my_b in 0:4
            x_my, y_my, z_my, Nx_my, Ny_my, Nz_my = blocks_data[my_b + 1]
            # Real-only dimensions: (Nx+1, Ny+1, Nz+1)
            Ny_nodes_my = Ny_my + 1; Nz_nodes_my = Nz_my + 1
            for my_f in 3:6
                if !haskey(connections, (my_b, my_f)); continue; end
                src_b, src_f = connections[(my_b, my_f)]
                x_src, y_src, z_src, Nx_s, Ny_s, Nz_s = blocks_data[src_b + 1]
                Ny_nodes_src = Ny_s + 1; Nz_nodes_src = Nz_s + 1
                Nx_nodes_src = Nx_s + 1
                
                j_start_src, j_end_src, k_start_src, k_end_src = get_search_bounds(src_f, Ny_s, Nz_s, L_pack)

                # Ghost region in the destination block (will be at runtime with NG padding)
                # Here we just need to find where each ghost point maps in the source
                # ghost j/k ranges in real-only coord space:
                # face 3 (η-): j = 1..NG ghost cells → map to source, k = full
                # face 4 (η+): j = (Ny+1)..(Ny+NG) ghost → map to source, k = full
                # face 5 (ζ-): k = 1..NG ghost cells → map to source, j = full 
                # face 6 (ζ+): k = (Nz+1)..(Nz+NG) ghost → map to source, j = full
                # At runtime these will be ghost cells. But for weight computation, we need
                # to know which source cell each ghost cell maps to.
                # The ghost coords at runtime will be copies of source real coords,
                # so we compute weights for NG layers of the source boundary.
                
                # Number of ghost destination points
                if my_f == 3
                    j_ghost = 1:NG; k_ghost = 1:Nz_nodes_my
                elseif my_f == 4
                    j_ghost = 1:NG; k_ghost = 1:Nz_nodes_my  # NG layers past boundary
                elseif my_f == 5
                    j_ghost = 1:Ny_nodes_my; k_ghost = 1:NG
                elseif my_f == 6
                    j_ghost = 1:Ny_nodes_my; k_ghost = 1:NG
                end
                
                num_pts = length(j_ghost) * length(k_ghost)
                C1, C2, C3, C4 = zeros(Int32, num_pts), zeros(Int32, num_pts), zeros(Int32, num_pts), zeros(Int32, num_pts)
                W1, W2, W3, W4 = zeros(FT, num_pts), zeros(FT, num_pts), zeros(FT, num_pts), zeros(FT, num_pts)
                dest_J, dest_K = zeros(Int32, num_pts), zeros(Int32, num_pts)
                
                idx_out = 1
                mid_my = (size(x_my,1))÷2
                mid_src = (size(x_src,1))÷2
                
                # For ghost points, we need to find the corresponding real point in the source.
                # Ghost layer g (1..NG) maps to source boundary layer g.
                for k in k_ghost, j in j_ghost
                    # Determine the physical location of this ghost point
                    # by looking at the boundary of the destination block
                    if my_f == 3        # η-: ghost extends below j=1
                        j_real = 1; g = NG + 1 - j  # ghost layer 1..NG
                        # Mirror: ghost point at layer g below boundary
                        y_p = 2*y_my[mid_my, 1, k] - y_my[mid_my, 1+g, k]
                        z_p = 2*z_my[mid_my, 1, k] - z_my[mid_my, 1+g, k]
                    elseif my_f == 4    # η+: ghost extends above j=Ny+1  
                        g = j  # ghost layer 1..NG
                        y_p = 2*y_my[mid_my, Ny_nodes_my, k] - y_my[mid_my, Ny_nodes_my-g, k]
                        z_p = 2*z_my[mid_my, Ny_nodes_my, k] - z_my[mid_my, Ny_nodes_my-g, k]
                    elseif my_f == 5    # ζ-: ghost extends below k=1
                        g = NG + 1 - k  
                        y_p = 2*y_my[mid_my, j, 1] - y_my[mid_my, j, 1+g]
                        z_p = 2*z_my[mid_my, j, 1] - z_my[mid_my, j, 1+g]
                    elseif my_f == 6    # ζ+: ghost extends above k=Nz+1
                        g = k
                        y_p = 2*y_my[mid_my, j, Nz_nodes_my] - y_my[mid_my, j, Nz_nodes_my-g]
                        z_p = 2*z_my[mid_my, j, Nz_nodes_my] - z_my[mid_my, j, Nz_nodes_my-g]
                    end
                    
                    p = [y_p, z_p]
                    best_jb, best_kb, min_dist_sq = j_start_src, k_start_src, 1e18
                    for kb = k_start_src:k_end_src, jb = j_start_src:j_end_src
                        dy, dz = y_src[mid_src, jb, kb] - y_p, z_src[mid_src, jb, kb] - z_p
                        dist_sq = dy*dy + dz*dz
                        if dist_sq < min_dist_sq
                            min_dist_sq = dist_sq; best_jb, best_kb = jb, kb
                        end
                    end
                    if min_dist_sq < 1e-7
                        W1[idx_out] = one(FT); final_jb, final_kb = best_jb, best_kb
                    else
                        found_quad = false
                        for kb = k_start_src:k_end_src-1, jb = j_start_src:j_end_src-1
                            q1 = [y_src[mid_src, jb, kb], z_src[mid_src, jb, kb]]
                            q2 = [y_src[mid_src, jb+1, kb], z_src[mid_src, jb+1, kb]]
                            q3 = [y_src[mid_src, jb+1, kb+1], z_src[mid_src, jb+1, kb+1]]
                            q4 = [y_src[mid_src, jb, kb+1], z_src[mid_src, jb, kb+1]]
                            if point_in_quad(p, q1, q2, q3, q4)
                                W1[idx_out], W2[idx_out], W3[idx_out], W4[idx_out] = get_bilinear_weights(p, q1, q2, q3, q4)
                                final_jb, final_kb = jb, kb; found_quad = true; break
                            end
                        end
                        if !found_quad; W1[idx_out] = one(FT); final_jb, final_kb = best_jb, best_kb; end
                    end
                    # Store offsets in real-only layout (no NG padding)
                    # At solver runtime, NG offset will be added when decoding
                    base1 = (final_kb - 1) * Ny_nodes_src * Nx_nodes_src + (final_jb - 1) * Nx_nodes_src
                    C1[idx_out], C2[idx_out] = base1, base1 + Nx_nodes_src
                    C3[idx_out] = base1 + Nx_nodes_src + Ny_nodes_src * Nx_nodes_src
                    C4[idx_out] = base1 + Ny_nodes_src * Nx_nodes_src
                    
                    # dest_J, dest_K: destination indices in the padded (NG-offset) layout at runtime
                    if my_f == 3
                        dest_J[idx_out] = j  # 1..NG → ghost region at runtime
                        dest_K[idx_out] = k + NG  # real k with NG offset
                    elseif my_f == 4
                        dest_J[idx_out] = Ny_my + NG + j  # past real region
                        dest_K[idx_out] = k + NG
                    elseif my_f == 5
                        dest_J[idx_out] = j + NG
                        dest_K[idx_out] = k  # 1..NG → ghost region
                    elseif my_f == 6
                        dest_J[idx_out] = j + NG
                        dest_K[idx_out] = Nz_my + NG + k
                    end
                    idx_out += 1
                end
                grp = create_group(h5f, "b$(my_b)_f$(my_f)")
                grp["num_pts"] = num_pts; grp["C1"] = C1; grp["C2"] = C2; grp["C3"] = C3; grp["C4"] = C4
                grp["W1"] = W1; grp["W2"] = W2; grp["W3"] = W3; grp["W4"] = W4
                grp["dest_J"] = dest_J; grp["dest_K"] = dest_K
                # Stride_n in real-only layout
                grp["Stride_n"] = Int64(Nx_nodes_src * Ny_nodes_src * (Nz_s + 1))
            end
        end
    end
end

# =============================================================================
# VTK Export (real nodes only)
# =============================================================================
function export_mesh_vtk(blocks_data, out_dir)
    for bid in 0:4
        x, y, z, Nx, Ny, Nz = blocks_data[bid + 1]
        vtk_grid(joinpath(out_dir, "mesh_debug_b$bid"), x, y, z) do vtk
        end
    end
    
    # Manually create VTM file
    vtm_path = joinpath(out_dir, "mesh_debug_all.vtm")
    open(vtm_path, "w") do f
        write(f, "<?xml version=\"1.0\"?>\n")
        write(f, "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n")
        write(f, "  <vtkMultiBlockDataSet>\n")
        for bid in 0:4
            write(f, "    <DataSet index=\"$bid\" name=\"Block_$bid\" file=\"mesh_debug_b$bid.vts\"/>\n")
        end
        write(f, "  </vtkMultiBlockDataSet>\n")
        write(f, "</VTKFile>\n")
    end
end


function build_mesh(out_dir, Nx_b, Ny_b0, Nz_b0, N_rad)
    build_mesh(out_dir, Nx_b, Ny_b0, Nz_b0, N_rad, Lx)
end

function build_mesh(out_dir, Nx_b, Ny_b0, Nz_b0, N_rad, Lx::Float64)
    println("Generating FVM Butterfly Mesh into ", out_dir, " (Lx=$(Lx)) ...")
    mkpath(out_dir)
    blocks_data = []

    max_dr_outer = 0.00035 * (1024.0 / Nx_b)

    push!(blocks_data, generate_block0(Nx_b, Ny_b0, Nz_b0, Lx))
    push!(blocks_data, generate_block1(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer, Lx))
    push!(blocks_data, generate_block2(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer, Lx))
    push!(blocks_data, generate_block3(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer, Lx))
    push!(blocks_data, generate_block4(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer, Lx))

    build_interpolation_weights(blocks_data, out_dir)

    for bid in 0:4
        x, y, z, Nx, Ny, Nz = blocks_data[bid + 1]
        h5open(joinpath(out_dir, "mesh_b$bid.h5"), "w") do f
            f["NG"] = NG
            f["Nx"] = Int64(Nx)
            f["Ny"] = Int64(Ny)
            f["Nz"] = Int64(Nz)
            f["coords"] = Float32.(cat(reshape(x, (1, size(x)...)), reshape(y, (1, size(y)...)), reshape(z, (1, size(z)...)), dims=1))
            f["x"] = Float32.(x)
            f["y"] = Float32.(y)
            f["z"] = Float32.(z)
        end
    end

    # ─── Connectivity with auto-computed reverse_tan ───
    connectivity_rows = [
        0 3 1 4; 1 4 0 3; 0 4 2 3; 2 3 0 4;
        0 5 3 6; 3 6 0 5; 0 6 4 5; 4 5 0 6;
        1 5 3 3; 3 3 1 5; 1 6 4 3; 4 3 1 6;
        2 5 3 4; 3 4 2 5; 2 6 4 4; 4 4 2 6
    ]

    function get_tangential_vector(blocks_data, bid, fid)
        x, y, z, Nx, Ny, Nz = blocks_data[bid + 1]
        mid_i = size(x, 1) ÷ 2
        if fid == 3
            j0 = 1; mid_k = (Nz + 2) ÷ 2
            return [y[mid_i, j0, mid_k+1] - y[mid_i, j0, mid_k],
                    z[mid_i, j0, mid_k+1] - z[mid_i, j0, mid_k]]
        elseif fid == 4
            j0 = Ny + 1; mid_k = (Nz + 2) ÷ 2
            return [y[mid_i, j0, mid_k+1] - y[mid_i, j0, mid_k],
                    z[mid_i, j0, mid_k+1] - z[mid_i, j0, mid_k]]
        elseif fid == 5
            k0 = 1; mid_j = (Ny + 2) ÷ 2
            return [y[mid_i, mid_j+1, k0] - y[mid_i, mid_j, k0],
                    z[mid_i, mid_j+1, k0] - z[mid_i, mid_j, k0]]
        elseif fid == 6
            k0 = Nz + 1; mid_j = (Ny + 2) ÷ 2
            return [y[mid_i, mid_j+1, k0] - y[mid_i, mid_j, k0],
                    z[mid_i, mid_j+1, k0] - z[mid_i, mid_j, k0]]
        end
    end

    reverse_tan_arr = zeros(Int64, size(connectivity_rows, 1))
    flip_normal_arr = zeros(Int64, size(connectivity_rows, 1))
    for i in 1:size(connectivity_rows, 1)
        b1, f1, b2, f2 = connectivity_rows[i, :]
        t1 = get_tangential_vector(blocks_data, b1, f1)
        t2 = get_tangential_vector(blocks_data, b2, f2)
        reverse_tan_arr[i] = dot(t1, t2) < 0 ? 1 : 0

        f1_is_eta = (f1 == 3 || f1 == 4)
        f1_is_zeta = (f1 == 5 || f1 == 6)
        f2_is_eta = (f2 == 3 || f2 == 4)
        f2_is_zeta = (f2 == 5 || f2 == 6)
        if (f1_is_eta && f2_is_zeta) || (f1_is_zeta && f2_is_eta)
            flip_normal_arr[i] = 1
        end
    end

    face_bc = zeros(Int64, 5, 6)
    for bid in 0:4
        face_bc[bid+1, 1] = 2
        face_bc[bid+1, 2] = 2
    end
    for i in 1:size(connectivity_rows, 1)
        b1, f1 = connectivity_rows[i, 1], connectivity_rows[i, 2]
        face_bc[b1+1, f1] = 0
    end
    face_bc[2, 3] = 1
    face_bc[3, 4] = 1
    face_bc[4, 5] = 1
    face_bc[5, 6] = 1

    bc_params = zeros(FT, 5, 6, 17)
    bc_params[2, 3, 1] = FT(307.0e0)
    bc_params[3, 4, 1] = FT(307.0e0)
    bc_params[4, 5, 1] = FT(307.0e0)
    bc_params[5, 6, 1] = FT(307.0e0)

    h5open(joinpath(out_dir, "block_connectivity.h5"), "w") do f
        f["Nblocks"] = 5
        f["Nx_b"] = fill(Int64(Nx_b), 5)
        f["Ny_b"] = Int64[Ny_b0, N_rad, N_rad, Ny_b0, Ny_b0]
        f["Nz_b"] = Int64[Nz_b0, Nz_b0, Nz_b0, N_rad, N_rad]
        f["connectivity"] = connectivity_rows
        f["reverse_tan"] = reverse_tan_arr
        f["flip_normal"] = flip_normal_arr
        f["face_bc"] = face_bc
        f["bc_params"] = bc_params
    end
    if vis; export_mesh_vtk(blocks_data, out_dir); end
    println("Done '", out_dir, "'!")
end

function main()
    build_mesh("MESH_COARSE", 512, 108, 108, 108)  # N_rad: 96 → 108
    build_mesh("MESH_FINE", 1024, 216, 216, 216)   # N_rad: 192 → 216
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
