# =============================================================================
# Geometry Jump Correction for Multiblock Interface Kinks (3rd-Order)
# =============================================================================

"""
    compute_geometric_jump_coefficients(x_full, y_full, z_full, bid, nxp, nyp, nzp, NG, face_bc, connectivity)

Compute geometric displacement projection coefficients D_ξ, D_η, D_ζ at cell centers.
- x_full, y_full, z_full: 3D coordinate arrays of size (nxp+2NG+1, nyp+2NG+1, nzp+2NG+1) on CPU.
- Returns D_i_h, D_j_h, D_k_h: 3D Float64 arrays containing coefficients.
"""
function compute_geometric_jump_coefficients(x_full, y_full, z_full, bid::Int,
                                             nxp::Int, nyp::Int, nzp::Int, NG::Int, face_bc, connectivity)
    Nx_tot = nxp + 2*NG
    Ny_tot = nyp + 2*NG
    Nz_tot = nzp + 2*NG
    
    # 1. Clone coordinate arrays for smooth extrapolation
    x_ext = copy(x_full)
    y_ext = copy(y_full)
    z_ext = copy(z_full)
    
    # 2. Apply quadratic coordinate extrapolation at BC_INTERBLOCK faces
    # ── η- (face 3) ──
    if get(face_bc, (bid, 3), -1) == BC_INTERBLOCK
        for g in 1:NG
            j_g = NG + 1 - g
            j_ref = NG + 1 + g
            j_ref2 = NG + 1 + 2*g
            for k in 1:nzp+2*NG+1, i in 1:nxp+2*NG+1
                x_ext[i, j_g, k] = 3.0 * x_full[i, NG+1, k] - 3.0 * x_full[i, j_ref, k] + x_full[i, j_ref2, k]
                y_ext[i, j_g, k] = 3.0 * y_full[i, NG+1, k] - 3.0 * y_full[i, j_ref, k] + y_full[i, j_ref2, k]
                z_ext[i, j_g, k] = 3.0 * z_full[i, NG+1, k] - 3.0 * z_full[i, j_ref, k] + z_full[i, j_ref2, k]
            end
        end
    end
    
    # ── η+ (face 4) ──
    if get(face_bc, (bid, 4), -1) == BC_INTERBLOCK
        for g in 1:NG
            j_g = nyp + NG + 1 + g
            j_ref = nyp + NG + 1 - g
            j_ref2 = nyp + NG + 1 - 2*g
            for k in 1:nzp+2*NG+1, i in 1:nxp+2*NG+1
                x_ext[i, j_g, k] = 3.0 * x_full[i, nyp+NG+1, k] - 3.0 * x_full[i, j_ref, k] + x_full[i, j_ref2, k]
                y_ext[i, j_g, k] = 3.0 * y_full[i, nyp+NG+1, k] - 3.0 * y_full[i, j_ref, k] + y_full[i, j_ref2, k]
                z_ext[i, j_g, k] = 3.0 * z_full[i, nyp+NG+1, k] - 3.0 * z_full[i, j_ref, k] + z_full[i, j_ref2, k]
            end
        end
    end
    
    # ── ζ- (face 5) ──
    if get(face_bc, (bid, 5), -1) == BC_INTERBLOCK
        for g in 1:NG
            k_g = NG + 1 - g
            k_ref = NG + 1 + g
            k_ref2 = NG + 1 + 2*g
            for j in 1:nyp+2*NG+1, i in 1:nxp+2*NG+1
                x_ext[i, j, k_g] = 3.0 * x_full[i, j, NG+1] - 3.0 * x_full[i, j, k_ref] + x_full[i, j, k_ref2]
                y_ext[i, j, k_g] = 3.0 * y_full[i, j, NG+1] - 3.0 * y_full[i, j, k_ref] + y_full[i, j, k_ref2]
                z_ext[i, j, k_g] = 3.0 * z_full[i, j, NG+1] - 3.0 * z_full[i, j, k_ref] + z_full[i, j, k_ref2]
            end
        end
    end
    
    # ── ζ+ (face 6) ──
    if get(face_bc, (bid, 6), -1) == BC_INTERBLOCK
        for g in 1:NG
            k_g = nzp + NG + 1 + g
            k_ref = nzp + NG + 1 - g
            k_ref2 = nzp + NG + 1 - 2*g
            for j in 1:nyp+2*NG+1, i in 1:nxp+2*NG+1
                x_ext[i, j, k_g] = 3.0 * x_full[i, j, nzp+NG+1] - 3.0 * x_full[i, j, k_ref] + x_full[i, j, k_ref2]
                y_ext[i, j, k_g] = 3.0 * y_full[i, j, nzp+NG+1] - 3.0 * y_full[i, j, k_ref] + y_full[i, j, k_ref2]
                z_ext[i, j, k_g] = 3.0 * z_full[i, j, nzp+NG+1] - 3.0 * z_full[i, j, k_ref] + z_full[i, j, k_ref2]
            end
        end
    end
    
    # 3. Allocate coefficients
    D_i_h = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    D_j_h = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    D_k_h = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    
    # 4. Compute cell-centered coordinates
    xc_B = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    yc_B = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    zc_B = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
        xc_B[i, j, k] = 0.5 * (x_full[i, j, k] + x_full[i+1, j, k])
        yc_B[i, j, k] = 0.5 * (y_full[i, j, k] + y_full[i, j+1, k])
        zc_B[i, j, k] = 0.5 * (z_full[i, j, k] + z_full[i, j, k+1])
    end
    
    xc_ext = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    yc_ext = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    zc_ext = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
        xc_ext[i, j, k] = 0.5 * (x_ext[i, j, k] + x_ext[i+1, j, k])
        yc_ext[i, j, k] = 0.5 * (y_ext[i, j, k] + y_ext[i, j+1, k])
        zc_ext[i, j, k] = 0.5 * (z_ext[i, j, k] + z_ext[i, j, k+1])
    end
    
    # 5. Find interblock faces
    interblock_faces = Int[]
    for f in 3:6
        if get(face_bc, (bid, f), -1) == BC_INTERBLOCK
            push!(interblock_faces, f)
        end
    end
    
    if isempty(interblock_faces)
        return D_i_h, D_j_h, D_k_h
    end
    
    for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
        is_ghost = false
        g_min = NG
        normal_dir = 0 # 2=j (eta), 3=k (zeta)
        
        # 1D-Reconstruction Band Filtering:
        # Only apply geometric jump corrections if the cell is a ghost cell in the NORMAL direction, 
        # AND is a regular interior/real cell in the other two TANGENTIAL directions.
        if (3 in interblock_faces) && (j <= NG) && (NG < i <= nxp + NG) && (NG < k <= nzp + NG)
            is_ghost = true
            g_min = min(g_min, NG + 1 - j)
            normal_dir = 2
        elseif (4 in interblock_faces) && (j > nyp + NG) && (NG < i <= nxp + NG) && (NG < k <= nzp + NG)
            is_ghost = true
            g_min = min(g_min, j - (nyp + NG))
            normal_dir = 2
        elseif (5 in interblock_faces) && (k <= NG) && (NG < i <= nxp + NG) && (NG < j <= nyp + NG)
            is_ghost = true
            g_min = min(g_min, NG + 1 - k)
            normal_dir = 3
        elseif (6 in interblock_faces) && (k > nzp + NG) && (NG < i <= nxp + NG) && (NG < j <= nyp + NG)
            is_ghost = true
            g_min = min(g_min, k - (nzp + NG))
            normal_dir = 3
        end
        
        if !is_ghost; continue; end
        
        # Compute smooth fade-out window factor to suppress boundary difference noises
        w_fade = if g_min == 1
            1.0
        elseif g_min == 2
            0.6
        elseif g_min == 3
            0.2
        else
            0.0
        end
        
        if w_fade == 0.0; continue; end
        
        # Displacement vector Δx
        dx = xc_ext[i, j, k] - xc_B[i, j, k]
        dy = yc_ext[i, j, k] - yc_B[i, j, k]
        dz = zc_ext[i, j, k] - zc_B[i, j, k]
        
        # Covariant basis vectors e1, e2, e3 at cell center
        im1 = max(i-1, 1); ip1 = min(i+1, Nx_tot)
        jm1 = max(j-1, 1); jp1 = min(j+1, Ny_tot)
        km1 = max(k-1, 1); kp1 = min(k+1, Nz_tot)
        
        e1_x = (xc_B[ip1, j, k] - xc_B[im1, j, k]) / (ip1 - im1)
        e1_y = (yc_B[ip1, j, k] - yc_B[im1, j, k]) / (ip1 - im1)
        e1_z = (zc_B[ip1, j, k] - zc_B[im1, j, k]) / (ip1 - im1)
        
        e2_x = (xc_B[i, jp1, k] - xc_B[i, jm1, k]) / (jp1 - jm1)
        e2_y = (yc_B[i, jp1, k] - yc_B[i, jm1, k]) / (jp1 - jm1)
        e2_z = (zc_B[i, jp1, k] - zc_B[i, jm1, k]) / (jp1 - jm1)
        
        e3_x = (xc_B[i, j, kp1] - xc_B[i, j, km1]) / (kp1 - km1)
        e3_y = (yc_B[i, j, kp1] - yc_B[i, j, km1]) / (kp1 - km1)
        e3_z = (zc_B[i, j, kp1] - zc_B[i, j, km1]) / (kp1 - km1)
        
        # Jacobian determinant J = e1 . (e2 x e3)
        cross23_x = e2_y * e3_z - e2_z * e3_y
        cross23_y = e2_z * e3_x - e2_x * e3_z
        cross23_z = e2_x * e3_y - e2_y * e3_x
        
        J = e1_x * cross23_x + e1_y * cross23_y + e1_z * cross23_z
        
        if abs(J) > 1e-18
            if normal_dir == 2
                # eta boundary: project only normal component D_j. Set tangential components to 0.
                cross31_x = e3_y * e1_z - e3_z * e1_y
                cross31_y = e3_z * e1_x - e3_x * e1_z
                cross31_z = e3_x * e1_y - e3_y * e1_x
                D_j_h[i, j, k] = clamp(w_fade * (dx * cross31_x + dy * cross31_y + dz * cross31_z) / J, -1.0, 1.0)
            elseif normal_dir == 3
                # zeta boundary: project only normal component D_k. Set tangential components to 0.
                cross12_x = e1_y * e2_z - e1_z * e2_y
                cross12_y = e1_z * e2_x - e1_x * e2_z
                cross12_z = e1_x * e2_y - e1_y * e2_x
                D_k_h[i, j, k] = clamp(w_fade * (dx * cross12_x + dy * cross12_y + dz * cross12_z) / J, -1.0, 1.0)
            end
        end
    end
    
    return D_i_h, D_j_h, D_k_h
end

# =============================================================================
# GPU Jump Correction Kernel (3rd-Order Taylor Expansion)
# =============================================================================
function correct_ghost_kinks_kernel!(U, U_old, D_i, D_j, D_k, Nx_tot, Ny_tot, Nz_tot, Ncons)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    
    if i > Nx_tot || j > Ny_tot || k > Nz_tot; return; end
    
    d1 = D_i[i, j, k]
    d2 = D_j[i, j, k]
    d3 = D_k[i, j, k]
    
    # Quick exit path if displacement is negligible
    if abs(d1) < 1.0e-12 && abs(d2) < 1.0e-12 && abs(d3) < 1.0e-12; return; end
    
    # Boundary clamps to prevent out-of-bound difference operators
    ip1 = clamp(i+1, 1, Nx_tot); im1 = clamp(i-1, 1, Nx_tot)
    jp1 = clamp(j+1, 1, Ny_tot); jm1 = clamp(j-1, 1, Ny_tot)
    kp1 = clamp(k+1, 1, Nz_tot); km1 = clamp(k-1, 1, Nz_tot)
    jp2 = clamp(j+2, 1, Ny_tot); jm2 = clamp(j-2, 1, Ny_tot)
    kp2 = clamp(k+2, 1, Nz_tot); km2 = clamp(k-2, 1, Nz_tot)
    
    @inbounds for c in 1:Ncons
        # 1st-order derivatives in computational space
        du_dxi   = 0.5 * (U_old[ip1, j, k, c] - U_old[im1, j, k, c])
        du_deta  = 0.5 * (U_old[i, jp1, k, c] - U_old[i, jm1, k, c])
        du_dzeta = 0.5 * (U_old[i, j, kp1, c] - U_old[i, j, km1, c])
        
        # 2nd-order pure derivatives
        d2u_dxi2   = U_old[ip1, j, k, c] - 2.0 * U_old[i, j, k, c] + U_old[im1, j, k, c]
        d2u_deta2  = U_old[i, jp1, k, c] - 2.0 * U_old[i, j, k, c] + U_old[i, jm1, k, c]
        d2u_dzeta2 = U_old[i, j, kp1, c] - 2.0 * U_old[i, j, k, c] + U_old[i, j, km1, c]
        
        # 2nd-order cross derivatives
        d2u_dxideta   = 0.25 * (U_old[ip1, jp1, k, c] - U_old[im1, jp1, k, c] - U_old[ip1, jm1, k, c] + U_old[im1, jm1, k, c])
        d2u_detadzeta = 0.25 * (U_old[i, jp1, kp1, c] - U_old[i, jm1, kp1, c] - U_old[i, jp1, km1, c] + U_old[i, jm1, km1, c])
        d2u_dzetadxi  = 0.25 * (U_old[ip1, j, kp1, c] - U_old[im1, j, kp1, c] - U_old[ip1, j, km1, c] + U_old[im1, j, km1, c])
        
        # Taylor expansion correction (up to 3rd order)
        corr_1 = d1 * du_dxi + d2 * du_deta + d3 * du_dzeta
        corr_2 = 0.5 * (d1*d1 * d2u_dxi2 + d2*d2 * d2u_deta2 + d3*d3 * d2u_dzeta2) +
                 d1*d2 * d2u_dxideta + d2*d3 * d2u_detadzeta + d3*d1 * d2u_dzetadxi
        
        # 3rd-order pure derivatives (cross terms vanish due to 1D band filtering)
        d3u_deta3  = 0.5 * (U_old[i, jp2, k, c] - 2.0 * U_old[i, jp1, k, c] + 2.0 * U_old[i, jm1, k, c] - U_old[i, jm2, k, c])
        d3u_dzeta3 = 0.5 * (U_old[i, j, kp2, c] - 2.0 * U_old[i, j, kp1, c] + 2.0 * U_old[i, j, km1, c] - U_old[i, j, km2, c])
        corr_3 = (1.0/6.0) * (d2*d2*d2 * d3u_deta3 + d3*d3*d3 * d3u_dzeta3)
        
        corr = corr_1 + corr_2 + corr_3
        limit_val = 0.10 * (abs(U_old[i, j, k, c]) + 1.0e-6)
        if corr > limit_val
            corr = limit_val
        elseif corr < -limit_val
            corr = -limit_val
        end
        
        U_new = U_old[i, j, k, c] + corr
        if c == 1 || c == 5
            U_new = max(U_new, 0.10 * U_old[i, j, k, c])
            U_new = max(U_new, 1.0e-6)
        end
        U[i, j, k, c] = U_new
    end
    return
end
