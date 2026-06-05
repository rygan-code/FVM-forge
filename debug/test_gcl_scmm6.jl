# Test 6th-order standard metrics vs. 6th-order SCMM metrics GCL conservation (Revised Outer Operators)

using HDF5
using LinearAlgebra
using Printf

const FT = Float64

# Define CMD6 operators (6th-order midpoint)
function deriv_i(arr, i, j, k, N)
    im2 = clamp(i-2, 1, N)
    im1 = clamp(i-1, 1, N)
    i0  = clamp(i,   1, N)
    ip1 = clamp(i+1, 1, N)
    ip2 = clamp(i+2, 1, N)
    ip3 = clamp(i+3, 1, N)
    return (75.0 * (arr[ip1, j, k] - arr[i0, j, k]) / 64.0) - 
           (25.0 * (arr[ip2, j, k] - arr[im1, j, k]) / 384.0) + 
           (3.0 * (arr[ip3, j, k] - arr[im2, j, k]) / 640.0)
end

function deriv_j(arr, i, j, k, N)
    jm2 = clamp(j-2, 1, N)
    jm1 = clamp(j-1, 1, N)
    j0  = clamp(j,   1, N)
    jp1 = clamp(j+1, 1, N)
    jp2 = clamp(j+2, 1, N)
    jp3 = clamp(j+3, 1, N)
    return (75.0 * (arr[i, jp1, k] - arr[i, j0, k]) / 64.0) - 
           (25.0 * (arr[i, jp2, k] - arr[i, jm1, k]) / 384.0) + 
           (3.0 * (arr[i, jp3, k] - arr[i, jm2, k]) / 640.0)
end

function deriv_k(arr, i, j, k, N)
    km2 = clamp(k-2, 1, N)
    km1 = clamp(k-1, 1, N)
    k0  = clamp(k,   1, N)
    kp1 = clamp(k+1, 1, N)
    kp2 = clamp(k+2, 1, N)
    kp3 = clamp(k+3, 1, N)
    return (75.0 * (arr[i, j, kp1] - arr[i, j, k0]) / 64.0) - 
           (25.0 * (arr[i, j, kp2] - arr[i, j, km1]) / 384.0) + 
           (3.0 * (arr[i, j, kp3] - arr[i, j, km2]) / 640.0)
end

function interp_i(arr, i, j, k, N)
    im2 = clamp(i-2, 1, N)
    im1 = clamp(i-1, 1, N)
    i0  = clamp(i,   1, N)
    ip1 = clamp(i+1, 1, N)
    ip2 = clamp(i+2, 1, N)
    ip3 = clamp(i+3, 1, N)
    return (75.0 * (arr[ip1, j, k] + arr[i0, j, k]) / 128.0) - 
           (25.0 * (arr[ip2, j, k] + arr[im1, j, k]) / 256.0) + 
           (3.0 * (arr[ip3, j, k] + arr[im2, j, k]) / 256.0)
end

function interp_j(arr, i, j, k, N)
    jm2 = clamp(j-2, 1, N)
    jm1 = clamp(j-1, 1, N)
    j0  = clamp(j,   1, N)
    jp1 = clamp(j+1, 1, N)
    jp2 = clamp(j+2, 1, N)
    jp3 = clamp(j+3, 1, N)
    return (75.0 * (arr[i, jp1, k] + arr[i, j0, k]) / 128.0) - 
           (25.0 * (arr[i, jp2, k] + arr[i, jm1, k]) / 256.0) + 
           (3.0 * (arr[i, jp3, k] + arr[i, jm2, k]) / 256.0)
end

function interp_k(arr, i, j, k, N)
    km2 = clamp(k-2, 1, N)
    km1 = clamp(k-1, 1, N)
    k0  = clamp(k,   1, N)
    kp1 = clamp(k+1, 1, N)
    kp2 = clamp(k+2, 1, N)
    kp3 = clamp(k+3, 1, N)
    return (75.0 * (arr[i, j, kp1] + arr[i, j, k0]) / 128.0) - 
           (25.0 * (arr[i, j, kp2] + arr[i, j, km1]) / 256.0) + 
           (3.0 * (arr[i, j, kp3] + arr[i, j, km2]) / 256.0)
end

function test_gcl_scmm6()
    # Read mesh
    mesh_path = "MESH_SMALL/mesh_b0.h5"
    if !isfile(mesh_path)
        println("Mesh file not found at $mesh_path")
        return
    end
    
    coords = h5open(mesh_path, "r") do f
        read(f["coords"])
    end
    x_real = coords[1, :, :, :]
    y_real = coords[2, :, :, :]
    z_real = coords[3, :, :, :]
    
    Nx = size(x_real, 1) - 1
    Ny = size(x_real, 2) - 1
    Nz = size(x_real, 3) - 1
    NG = 3
    
    Nx_nodes_tot = Nx + 2*NG + 1
    Ny_nodes_tot = Ny + 2*NG + 1
    Nz_nodes_tot = Nz + 2*NG + 1
    Nx_cells_tot = Nx + 2*NG
    Ny_cells_tot = Ny + 2*NG
    Nz_cells_tot = Nz + 2*NG
    
    x = zeros(Float64, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
    y = zeros(Float64, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
    z = zeros(Float64, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
    
    ri = NG+1:Nx+NG+1
    rj = NG+1:Ny+NG+1
    rk = NG+1:Nz+NG+1
    x[ri, rj, rk] .= x_real
    y[ri, rj, rk] .= y_real
    z[ri, rj, rk] .= z_real
    
    # Fill boundary ghost nodes via mirror extrapolation
    for k in 1:Nz_nodes_tot, j in 1:Ny_nodes_tot
        for g in 1:NG
            # xi- direction
            x[NG+1-g, j, k] = 2*x[NG+1, j, k] - x[NG+1+g, j, k]
            y[NG+1-g, j, k] = 2*y[NG+1, j, k] - y[NG+1+g, j, k]
            z[NG+1-g, j, k] = 2*z[NG+1, j, k] - z[NG+1+g, j, k]
            # xi+ direction
            x[Nx+NG+1+g, j, k] = 2*x[Nx+NG+1, j, k] - x[Nx+NG+1-g, j, k]
            y[Nx+NG+1+g, j, k] = 2*y[Nx+NG+1, j, k] - y[Nx+NG+1-g, j, k]
            z[Nx+NG+1+g, j, k] = 2*z[Nx+NG+1, j, k] - z[Nx+NG+1-g, j, k]
        end
    end
    
    for k in 1:Nz_nodes_tot, i in 1:Nx_nodes_tot
        for g in 1:NG
            # eta- direction
            x[i, NG+1-g, k] = 2*x[i, NG+1, k] - x[i, NG+1+g, k]
            y[i, NG+1-g, k] = 2*y[i, NG+1, k] - y[i, NG+1+g, k]
            z[i, NG+1-g, k] = 2*z[i, NG+1, k] - z[i, NG+1+g, k]
            # eta+ direction
            x[i, Ny+NG+1+g, k] = 2*x[i, Ny+NG+1, k] - x[i, Ny+NG+1-g, k]
            y[i, Ny+NG+1+g, k] = 2*y[i, Ny+NG+1, k] - y[i, Ny+NG+1-g, k]
            z[i, Ny+NG+1+g, k] = 2*z[i, Ny+NG+1, k] - z[i, Ny+NG+1-g, k]
        end
    end
    
    for j in 1:Ny_nodes_tot, i in 1:Nx_nodes_tot
        for g in 1:NG
            # zeta- direction
            x[i, j, NG+1-g] = 2*x[i, j, NG+1] - x[i, j, NG+1+g]
            y[i, j, NG+1-g] = 2*y[i, j, NG+1] - y[i, j, NG+1+g]
            z[i, j, NG+1-g] = 2*z[i, j, NG+1] - z[i, j, NG+1+g]
            # zeta+ direction
            x[i, j, Nz+NG+1+g] = 2*x[i, j, Nz+NG+1] - x[i, j, Nz+NG+1-g]
            y[i, j, Nz+NG+1+g] = 2*y[i, j, Nz+NG+1] - y[i, j, Nz+NG+1-g]
            z[i, j, Nz+NG+1+g] = 2*z[i, j, Nz+NG+1] - z[i, j, Nz+NG+1-g]
        end
    end
    
    println("Coordinates padded. Computing Baseline CMD6 metrics...")
    
    Ai_orig = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    nxi_orig = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    nyi_orig = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    nzi_orig = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    
    Aj_orig = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    nxj_orig = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    nyj_orig = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    nzj_orig = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    
    Ak_orig = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    nxk_orig = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    nyk_orig = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    nzk_orig = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    
    # Standard CMD6 FVM metric calculation
    for k in 1:Nz_cells_tot, j in 1:Ny_cells_tot, i in 1:Nx_nodes_tot
        d_eta_x = (75.0 * (deriv_j(x, i, j, clamp(k+1,1,Nz_nodes_tot), Ny_nodes_tot) + deriv_j(x, i, j, clamp(k,1,Nz_nodes_tot), Ny_nodes_tot)) / 128.0) -
                  (25.0 * (deriv_j(x, i, j, clamp(k+2,1,Nz_nodes_tot), Ny_nodes_tot) + deriv_j(x, i, j, clamp(k-1,1,Nz_nodes_tot), Ny_nodes_tot)) / 256.0) +
                  (3.0 * (deriv_j(x, i, j, clamp(k+3,1,Nz_nodes_tot), Ny_nodes_tot) + deriv_j(x, i, j, clamp(k-2,1,Nz_nodes_tot), Ny_nodes_tot)) / 256.0)
        d_eta_y = (75.0 * (deriv_j(y, i, j, clamp(k+1,1,Nz_nodes_tot), Ny_nodes_tot) + deriv_j(y, i, j, clamp(k,1,Nz_nodes_tot), Ny_nodes_tot)) / 128.0) -
                  (25.0 * (deriv_j(y, i, j, clamp(k+2,1,Nz_nodes_tot), Ny_nodes_tot) + deriv_j(y, i, j, clamp(k-1,1,Nz_nodes_tot), Ny_nodes_tot)) / 256.0) +
                  (3.0 * (deriv_j(y, i, j, clamp(k+3,1,Nz_nodes_tot), Ny_nodes_tot) + deriv_j(y, i, j, clamp(k-2,1,Nz_nodes_tot), Ny_nodes_tot)) / 256.0)
        d_eta_z = (75.0 * (deriv_j(z, i, j, clamp(k+1,1,Nz_nodes_tot), Ny_nodes_tot) + deriv_j(z, i, j, clamp(k,1,Nz_nodes_tot), Ny_nodes_tot)) / 128.0) -
                  (25.0 * (deriv_j(z, i, j, clamp(k+2,1,Nz_nodes_tot), Ny_nodes_tot) + deriv_j(z, i, j, clamp(k-1,1,Nz_nodes_tot), Ny_nodes_tot)) / 256.0) +
                  (3.0 * (deriv_j(z, i, j, clamp(k+3,1,Nz_nodes_tot), Ny_nodes_tot) + deriv_j(z, i, j, clamp(k-2,1,Nz_nodes_tot), Ny_nodes_tot)) / 256.0)
        
        d_zeta_x = (75.0 * (deriv_k(x, i, clamp(j+1,1,Ny_nodes_tot), k, Nz_nodes_tot) + deriv_k(x, i, clamp(j,1,Ny_nodes_tot), k, Nz_nodes_tot)) / 128.0) -
                   (25.0 * (deriv_k(x, i, clamp(j+2,1,Ny_nodes_tot), k, Nz_nodes_tot) + deriv_k(x, i, clamp(j-1,1,Ny_nodes_tot), k, Nz_nodes_tot)) / 256.0) +
                   (3.0 * (deriv_k(x, i, clamp(j+3,1,Ny_nodes_tot), k, Nz_nodes_tot) + deriv_k(x, i, clamp(j-2,1,Ny_nodes_tot), k, Nz_nodes_tot)) / 256.0)
        d_zeta_y = (75.0 * (deriv_k(y, i, clamp(j+1,1,Ny_nodes_tot), k, Nz_nodes_tot) + deriv_k(y, i, clamp(j,1,Ny_nodes_tot), k, Nz_nodes_tot)) / 128.0) -
                   (25.0 * (deriv_k(y, i, clamp(j+2,1,Ny_nodes_tot), k, Nz_nodes_tot) + deriv_k(y, i, clamp(j-1,1,Ny_nodes_tot), k, Nz_nodes_tot)) / 256.0) +
                   (3.0 * (deriv_k(y, i, clamp(j+3,1,Ny_nodes_tot), k, Nz_nodes_tot) + deriv_k(y, i, clamp(j-2,1,Ny_nodes_tot), k, Nz_nodes_tot)) / 256.0)
        d_zeta_z = (75.0 * (deriv_k(z, i, clamp(j+1,1,Ny_nodes_tot), k, Nz_nodes_tot) + deriv_k(z, i, clamp(j,1,Ny_nodes_tot), k, Nz_nodes_tot)) / 128.0) -
                   (25.0 * (deriv_k(z, i, clamp(j+2,1,Ny_nodes_tot), k, Nz_nodes_tot) + deriv_k(z, i, clamp(j-1,1,Ny_nodes_tot), k, Nz_nodes_tot)) / 256.0) +
                   (3.0 * (deriv_k(z, i, clamp(j+3,1,Ny_nodes_tot), k, Nz_nodes_tot) + deriv_k(z, i, clamp(j-2,1,Ny_nodes_tot), k, Nz_nodes_tot)) / 256.0)
        
        Sx = d_eta_y * d_zeta_z - d_zeta_y * d_eta_z
        Sy = d_eta_z * d_zeta_x - d_zeta_z * d_eta_x
        Sz = d_eta_x * d_zeta_y - d_zeta_x * d_eta_y
        
        area = sqrt(Sx^2 + Sy^2 + Sz^2)
        Ai_orig[i,j,k] = area
        nxi_orig[i,j,k] = area > 1e-15 ? Sx / area : 1.0
        nyi_orig[i,j,k] = area > 1e-15 ? Sy / area : 0.0
        nzi_orig[i,j,k] = area > 1e-15 ? Sz / area : 0.0
    end
    
    for k in 1:Nz_cells_tot, j in 1:Ny_nodes_tot, i in 1:Nx_cells_tot
        d_zeta_x = (75.0 * (deriv_k(x, clamp(i+1,1,Nx_nodes_tot), j, k, Nz_nodes_tot) + deriv_k(x, clamp(i,1,Nx_nodes_tot), j, k, Nz_nodes_tot)) / 128.0) -
                   (25.0 * (deriv_k(x, clamp(i+2,1,Nx_nodes_tot), j, k, Nz_nodes_tot) + deriv_k(x, clamp(i-1,1,Nx_nodes_tot), j, k, Nz_nodes_tot)) / 256.0) +
                   (3.0 * (deriv_k(x, clamp(i+3,1,Nx_nodes_tot), j, k, Nz_nodes_tot) + deriv_k(x, clamp(i-2,1,Nx_nodes_tot), j, k, Nz_nodes_tot)) / 256.0)
        d_zeta_y = (75.0 * (deriv_k(y, clamp(i+1,1,Nx_nodes_tot), j, k, Nz_nodes_tot) + deriv_k(y, clamp(i,1,Nx_nodes_tot), j, k, Nz_nodes_tot)) / 128.0) -
                   (25.0 * (deriv_k(y, clamp(i+2,1,Nx_nodes_tot), j, k, Nz_nodes_tot) + deriv_k(y, clamp(i-1,1,Nx_nodes_tot), j, k, Nz_nodes_tot)) / 256.0) +
                   (3.0 * (deriv_k(y, clamp(i+3,1,Nx_nodes_tot), j, k, Nz_nodes_tot) + deriv_k(y, clamp(i-2,1,Nx_nodes_tot), j, k, Nz_nodes_tot)) / 256.0)
        d_zeta_z = (75.0 * (deriv_k(z, clamp(i+1,1,Nx_nodes_tot), j, k, Nz_nodes_tot) + deriv_k(z, clamp(i,1,Nx_nodes_tot), j, k, Nz_nodes_tot)) / 128.0) -
                   (25.0 * (deriv_k(z, clamp(i+2,1,Nx_nodes_tot), j, k, Nz_nodes_tot) + deriv_k(z, clamp(i-1,1,Nx_nodes_tot), j, k, Nz_nodes_tot)) / 256.0) +
                   (3.0 * (deriv_k(z, clamp(i+3,1,Nx_nodes_tot), j, k, Nz_nodes_tot) + deriv_k(z, clamp(i-2,1,Nx_nodes_tot), j, k, Nz_nodes_tot)) / 256.0)
        
        d_xi_x = (75.0 * (deriv_i(x, i, j, clamp(k+1,1,Nz_nodes_tot), Nx_nodes_tot) + deriv_i(x, i, j, clamp(k,1,Nz_nodes_tot), Nx_nodes_tot)) / 128.0) -
                 (25.0 * (deriv_i(x, i, j, clamp(k+2,1,Nz_nodes_tot), Nx_nodes_tot) + deriv_i(x, i, j, clamp(k-1,1,Nz_nodes_tot), Nx_nodes_tot)) / 256.0) +
                 (3.0 * (deriv_i(x, i, j, clamp(k+3,1,Nz_nodes_tot), Nx_nodes_tot) + deriv_i(x, i, j, clamp(k-2,1,Nz_nodes_tot), Nx_nodes_tot)) / 256.0)
        d_xi_y = (75.0 * (deriv_i(y, i, j, clamp(k+1,1,Nz_nodes_tot), Nx_nodes_tot) + deriv_i(y, i, j, clamp(k,1,Nz_nodes_tot), Nx_nodes_tot)) / 128.0) -
                 (25.0 * (deriv_i(y, i, j, clamp(k+2,1,Nz_nodes_tot), Nx_nodes_tot) + deriv_i(y, i, j, clamp(k-1,1,Nz_nodes_tot), Nx_nodes_tot)) / 256.0) +
                 (3.0 * (deriv_i(y, i, j, clamp(k+3,1,Nz_nodes_tot), Nx_nodes_tot) + deriv_i(y, i, j, clamp(k-2,1,Nz_nodes_tot), Nx_nodes_tot)) / 256.0)
        d_xi_z = (75.0 * (deriv_i(z, i, j, clamp(k+1,1,Nz_nodes_tot), Nx_nodes_tot) + deriv_i(z, i, j, clamp(k,1,Nz_nodes_tot), Nx_nodes_tot)) / 128.0) -
                 (25.0 * (deriv_i(z, i, j, clamp(k+2,1,Nz_nodes_tot), Nx_nodes_tot) + deriv_i(z, i, j, clamp(k-1,1,Nz_nodes_tot), Nx_nodes_tot)) / 256.0) +
                 (3.0 * (deriv_i(z, i, j, clamp(k+3,1,Nz_nodes_tot), Nx_nodes_tot) + deriv_i(z, i, j, clamp(k-2,1,Nz_nodes_tot), Nx_nodes_tot)) / 256.0)
        
        Sx = d_zeta_y * d_xi_z - d_xi_y * d_zeta_z
        Sy = d_zeta_z * d_xi_x - d_xi_z * d_zeta_x
        Sz = d_zeta_x * d_xi_y - d_xi_x * d_zeta_y
        
        area = sqrt(Sx^2 + Sy^2 + Sz^2)
        Aj_orig[i,j,k] = area
        nxj_orig[i,j,k] = area > 1e-15 ? Sx / area : 0.0
        nyj_orig[i,j,k] = area > 1e-15 ? Sy / area : 1.0
        nzj_orig[i,j,k] = area > 1e-15 ? Sz / area : 0.0
    end
    
    for k in 1:Nz_nodes_tot, j in 1:Ny_cells_tot, i in 1:Nx_cells_tot
        d_xi_x = (75.0 * (deriv_i(x, i, clamp(j+1,1,Ny_nodes_tot), k, Nx_nodes_tot) + deriv_i(x, i, clamp(j,1,Ny_nodes_tot), k, Nx_nodes_tot)) / 128.0) -
                 (25.0 * (deriv_i(x, i, clamp(j+2,1,Ny_nodes_tot), k, Nx_nodes_tot) + deriv_i(x, i, clamp(j-1,1,Ny_nodes_tot), k, Nx_nodes_tot)) / 256.0) +
                 (3.0 * (deriv_i(x, i, clamp(j+3,1,Ny_nodes_tot), k, Nx_nodes_tot) + deriv_i(x, i, clamp(j-2,1,Ny_nodes_tot), k, Nx_nodes_tot)) / 256.0)
        d_xi_y = (75.0 * (deriv_i(y, i, clamp(j+1,1,Ny_nodes_tot), k, Nx_nodes_tot) + deriv_i(y, i, clamp(j,1,Ny_nodes_tot), k, Nx_nodes_tot)) / 128.0) -
                 (25.0 * (deriv_i(y, i, clamp(j+2,1,Ny_nodes_tot), k, Nx_nodes_tot) + deriv_i(y, i, clamp(j-1,1,Ny_nodes_tot), k, Nx_nodes_tot)) / 256.0) +
                 (3.0 * (deriv_i(y, i, clamp(j+3,1,Ny_nodes_tot), k, Nx_nodes_tot) + deriv_i(y, i, clamp(j-2,1,Ny_nodes_tot), k, Nx_nodes_tot)) / 256.0)
        d_xi_z = (75.0 * (deriv_i(z, i, clamp(j+1,1,Ny_nodes_tot), k, Nx_nodes_tot) + deriv_i(z, i, clamp(j,1,Ny_nodes_tot), k, Nx_nodes_tot)) / 128.0) -
                 (25.0 * (deriv_i(z, i, clamp(j+2,1,Ny_nodes_tot), k, Nx_nodes_tot) + deriv_i(z, i, clamp(j-1,1,Ny_nodes_tot), k, Nx_nodes_tot)) / 256.0) +
                 (3.0 * (deriv_i(z, i, clamp(j+3,1,Ny_nodes_tot), k, Nx_nodes_tot) + deriv_i(z, i, clamp(j-2,1,Ny_nodes_tot), k, Nx_nodes_tot)) / 256.0)
        
        d_eta_x = (75.0 * (deriv_j(x, clamp(i+1,1,Nx_nodes_tot), j, k, Ny_nodes_tot) + deriv_j(x, clamp(i,1,Nx_nodes_tot), j, k, Ny_nodes_tot)) / 128.0) -
                  (25.0 * (deriv_j(x, clamp(i+2,1,Nx_nodes_tot), j, k, Ny_nodes_tot) + deriv_j(x, clamp(i-1,1,Nx_nodes_tot), j, k, Ny_nodes_tot)) / 256.0) +
                  (3.0 * (deriv_j(x, clamp(i+3,1,Nx_nodes_tot), j, k, Ny_nodes_tot) + deriv_j(x, clamp(i-2,1,Ny_nodes_tot), j, k, Ny_nodes_tot)) / 256.0)
        d_eta_y = (75.0 * (deriv_j(y, clamp(i+1,1,Nx_nodes_tot), j, k, Ny_nodes_tot) + deriv_j(y, clamp(i,1,Nx_nodes_tot), j, k, Ny_nodes_tot)) / 128.0) -
                  (25.0 * (deriv_j(y, clamp(i+2,1,Nx_nodes_tot), j, k, Ny_nodes_tot) + deriv_j(y, clamp(i-1,1,Nx_nodes_tot), j, k, Ny_nodes_tot)) / 256.0) +
                  (3.0 * (deriv_j(y, clamp(i+3,1,Nx_nodes_tot), j, k, Ny_nodes_tot) + deriv_j(y, clamp(i-2,1,Nz_nodes_tot), j, k, Ny_nodes_tot)) / 256.0)
        d_eta_z = (75.0 * (deriv_j(z, clamp(i+1,1,Nx_nodes_tot), j, k, Ny_nodes_tot) + deriv_j(z, clamp(i,1,Nx_nodes_tot), j, k, Ny_nodes_tot)) / 128.0) -
                  (25.0 * (deriv_j(z, clamp(i+2,1,Nx_nodes_tot), j, k, Ny_nodes_tot) + deriv_j(z, clamp(i-1,1,Nx_nodes_tot), j, k, Ny_nodes_tot)) / 256.0) +
                  (3.0 * (deriv_j(z, clamp(i+3,1,Nx_nodes_tot), j, k, Ny_nodes_tot) + deriv_j(z, clamp(i-2,1,Nz_nodes_tot), j, k, Ny_nodes_tot)) / 256.0)
        
        Sx = d_xi_y * d_eta_z - d_eta_y * d_xi_z
        Sy = d_xi_z * d_eta_x - d_eta_z * d_xi_x
        Sz = d_xi_x * d_eta_y - d_eta_x * d_xi_y
        
        area = sqrt(Sx^2 + Sy^2 + Sz^2)
        Ak_orig[i,j,k] = area
        nxk_orig[i,j,k] = area > 1e-15 ? Sx / area : 0.0
        nyk_orig[i,j,k] = area > 1e-15 ? Sy / area : 0.0
        nzk_orig[i,j,k] = area > 1e-15 ? Sz / area : 1.0
    end
    
    # GCL error of baseline CMD6
    max_div_orig_x = 0.0
    max_div_orig_y = 0.0
    max_div_orig_z = 0.0
    for k in (NG+1):(Nz_cells_tot-NG), j in (NG+1):(Ny_cells_tot-NG), i in (NG+1):(Nx_cells_tot-NG)
        div_x = Ai_orig[i+1, j, k]*nxi_orig[i+1, j, k] - Ai_orig[i, j, k]*nxi_orig[i, j, k] +
                Aj_orig[i, j+1, k]*nxj_orig[i, j+1, k] - Aj_orig[i, j, k]*nxj_orig[i, j, k] +
                Ak_orig[i, j, k+1]*nxk_orig[i, j, k+1] - Ak_orig[i, j, k]*nxk_orig[i, j, k]
                
        div_y = Ai_orig[i+1, j, k]*nyi_orig[i+1, j, k] - Ai_orig[i, j, k]*nyi_orig[i, j, k] +
                Aj_orig[i, j+1, k]*nyj_orig[i, j+1, k] - Aj_orig[i, j, k]*nyj_orig[i, j, k] +
                Ak_orig[i, j, k+1]*nyk_orig[i, j, k+1] - Ak_orig[i, j, k]*nyk_orig[i, j, k]
                
        div_z = Ai_orig[i+1, j, k]*nzi_orig[i+1, j, k] - Ai_orig[i, j, k]*nzi_orig[i, j, k] +
                Aj_orig[i, j+1, k]*nzj_orig[i, j+1, k] - Aj_orig[i, j, k]*nzj_orig[i, j, k] +
                Ak_orig[i, j, k+1]*nzk_orig[i, j, k+1] - Ak_orig[i, j, k]*nzk_orig[i, j, k]
                
        max_div_orig_x = max(max_div_orig_x, abs(div_x))
        max_div_orig_y = max(max_div_orig_y, abs(div_y))
        max_div_orig_z = max(max_div_orig_z, abs(div_z))
    end
    @printf("CMD6 Baseline GCL error:\n")
    @printf("  Max Div X: %.6e\n", max_div_orig_x)
    @printf("  Max Div Y: %.6e\n", max_div_orig_y)
    @printf("  Max Div Z: %.6e\n", max_div_orig_z)
    
    # ── Now compute 6th-order SCMM metrics ──
    println("\nComputing 6th-order SCMM metrics with revised outer difference operators...")
    
    # Face Area Vectors (Sx, Sy, Sz) directly
    Sx_i = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    Sy_i = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    Sz_i = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    
    Sx_j = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    Sy_j = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    Sz_j = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    
    Sx_k = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    Sy_k = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    Sz_k = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    
    # Intermediate edge-flux arrays
    # 1. i-face intermediates
    y_dz_k = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_cells_tot)
    y_dy_j = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_nodes_tot)
    
    z_dx_k = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_cells_tot)
    z_dx_j = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_nodes_tot)
    
    x_dy_k = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_cells_tot)
    x_dy_j = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_nodes_tot)
    
    # 2. j-face intermediates
    y_dz_i = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_nodes_tot)
    z_dx_i = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_nodes_tot)
    x_dy_i = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_nodes_tot)
    
    # 3. k-face intermediates
    y_dz_j = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_nodes_tot)
    z_dx_j = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_nodes_tot)
    x_dy_j = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_nodes_tot)
    
    # --- Pre-compute all intermediate edge products ---
    # a) k-face/edge intermediates (midpoint in k, nodes in i and j)
    for k in 1:Nz_cells_tot, j in 1:Ny_nodes_tot, i in 1:Nx_nodes_tot
        y_dz_k[i, j, k] = interp_k(y, i, j, k, Nz_nodes_tot) * deriv_k(z, i, j, k, Nz_nodes_tot)
        z_dx_k[i, j, k] = interp_k(z, i, j, k, Nz_nodes_tot) * deriv_k(x, i, j, k, Nz_nodes_tot)
        x_dy_k[i, j, k] = interp_k(x, i, j, k, Nz_nodes_tot) * deriv_k(y, i, j, k, Nz_nodes_tot)
    end
    
    # b) j-face/edge intermediates (midpoint in j, nodes in i and k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_cells_tot, i in 1:Nx_nodes_tot
        y_dy_j[i, j, k] = interp_j(y, i, j, k, Ny_nodes_tot) * deriv_j(z, i, j, k, Ny_nodes_tot)
        z_dx_j[i, j, k] = interp_j(z, i, j, k, Ny_nodes_tot) * deriv_j(x, i, j, k, Ny_nodes_tot)
        x_dy_j[i, j, k] = interp_j(x, i, j, k, Ny_nodes_tot) * deriv_j(y, i, j, k, Ny_nodes_tot)
    end
    
    # c) i-face/edge intermediates (midpoint in i, nodes in j and k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_nodes_tot, i in 1:Nx_cells_tot
        y_dz_i[i, j, k] = interp_i(y, i, j, k, Nx_nodes_tot) * deriv_i(z, i, j, k, Nx_nodes_tot)
        z_dx_i[i, j, k] = interp_i(z, i, j, k, Nx_nodes_tot) * deriv_i(x, i, j, k, Nx_nodes_tot)
        x_dy_i[i, j, k] = interp_i(x, i, j, k, Nx_nodes_tot) * deriv_i(y, i, j, k, Nx_nodes_tot)
    end
    
    # --- Now assemble the faces ---
    
    # 1. i-face: (i, j+1/2, k+1/2) for i in 1:Nx_nodes_tot, j in 1:Ny_cells_tot, k in 1:Nz_cells_tot
    for k in 1:Nz_cells_tot, j in 1:Ny_cells_tot, i in 1:Nx_nodes_tot
        Sx_i[i, j, k] = (y_dz_k[i, j+1, k] - y_dz_k[i, j, k]) - (y_dy_j[i, j, k+1] - y_dy_j[i, j, k])
        Sy_i[i, j, k] = (z_dx_j[i, j, k+1] - z_dx_j[i, j, k]) - (z_dx_k[i, j+1, k] - z_dx_k[i, j, k])
        Sz_i[i, j, k] = (x_dy_k[i, j+1, k] - x_dy_k[i, j, k]) - (x_dy_j[i, j, k+1] - x_dy_j[i, j, k])
    end
    
    # 2. j-face: (i+1/2, j, k+1/2) for i in 1:Nx_cells_tot, j in 1:Ny_nodes_tot, k in 1:Nz_cells_tot
    for k in 1:Nz_cells_tot, j in 1:Ny_nodes_tot, i in 1:Nx_cells_tot
        Sx_j[i, j, k] = (y_dz_i[i, j, k+1] - y_dz_i[i, j, k]) - (y_dz_k[i+1, j, k] - y_dz_k[i, j, k])
        Sy_j[i, j, k] = (z_dx_k[i+1, j, k] - z_dx_k[i, j, k]) - (z_dx_i[i, j, k+1] - z_dx_i[i, j, k])
        Sz_j[i, j, k] = (x_dy_i[i, j, k+1] - x_dy_i[i, j, k]) - (x_dy_k[i+1, j, k] - x_dy_k[i, j, k])
    end
    
    # 3. k-face: (i+1/2, j+1/2, k) for i in 1:Nx_cells_tot, j in 1:Ny_cells_tot, k in 1:Nz_nodes_tot
    for k in 1:Nz_nodes_tot, j in 1:Ny_cells_tot, i in 1:Nx_cells_tot
        Sx_k[i, j, k] = (y_dz_j[i+1, j, k] - y_dz_j[i, j, k]) - (y_dz_i[i, j+1, k] - y_dz_i[i, j, k])
        Sy_k[i, j, k] = (z_dx_i[i, j+1, k] - z_dx_i[i, j, k]) - (z_dx_j[i+1, j, k] - z_dx_j[i, j, k])
        Sz_k[i, j, k] = (x_dy_j[i+1, j, k] - x_dy_j[i, j, k]) - (x_dy_i[i, j+1, k] - x_dy_i[i, j, k])
    end
    
    # GCL error of SCMM 6th-order
    max_div_scmm_x = 0.0
    max_div_scmm_y = 0.0
    max_div_scmm_z = 0.0
    for k in (NG+1):(Nz_cells_tot-NG), j in (NG+1):(Ny_cells_tot-NG), i in (NG+1):(Nx_cells_tot-NG)
        div_x = Sx_i[i+1, j, k] - Sx_i[i, j, k] +
                Sx_j[i, j+1, k] - Sx_j[i, j, k] +
                Sx_k[i, j, k+1] - Sx_k[i, j, k]
                
        div_y = Sy_i[i+1, j, k] - Sy_i[i, j, k] +
                Sy_j[i, j+1, k] - Sy_j[i, j, k] +
                Sy_k[i, j, k+1] - Sy_k[i, j, k]
                
        div_z = Sz_i[i+1, j, k] - Sz_i[i, j, k] +
                Sz_j[i, j+1, k] - Sz_j[i, j, k] +
                Sz_k[i, j, k+1] - Sz_k[i, j, k]
                
        max_div_scmm_x = max(max_div_scmm_x, abs(div_x))
        max_div_scmm_y = max(max_div_scmm_y, abs(div_y))
        max_div_scmm_z = max(max_div_scmm_z, abs(div_z))
    end
    @printf("6th-order SCMM GCL error:\n")
    @printf("  Max Div X: %.6e\n", max_div_scmm_x)
    @printf("  Max Div Y: %.6e\n", max_div_scmm_y)
    @printf("  Max Div Z: %.6e\n", max_div_scmm_z)
end

test_gcl_scmm6()
