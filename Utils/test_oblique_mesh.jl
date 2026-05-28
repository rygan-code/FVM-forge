using HDF5
using WriteVTK
using LinearAlgebra

# --- Oblique Shock Configuration ---
const NG::Int64 = 4
const Nx::Int64 = 128
const Ny::Int64 = 64
const Nz::Int64 = 8 # Keep some depth for 3D kernels
const Lx::FT = FT(2.0)
const Ly::FT = FT(1.2e0)
const Lz::FT = FT(0.1)

const Nx_tot::Int64 = Nx + 2*NG
const Ny_tot::Int64 = Ny + 2*NG
const Nz_tot::Int64 = Nz + 2*NG

const Nx_nodes::Int64 = Nx + 1
const Ny_nodes::Int64 = Ny + 1
const Nz_nodes::Int64 = Nz + 1
const Nx_nodes_tot::Int64 = Nx_nodes + 2*NG
const Ny_nodes_tot::Int64 = Ny_nodes + 2*NG
const Nz_nodes_tot::Int64 = Nz_nodes + 2*NG

x = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
y = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
z = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)

# Metric arrays
Areai = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nxi = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nyi = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nzi = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
Areaj = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nxj = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nyj = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nzj = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
Areak = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nxk = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nyk = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nzk = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
Vol    = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)

println("Generating Oblique Shock (Wedge) Mesh...")

alpha = deg2rad(FT(15.0e0))
x_wedge = FT(0.2e0)

# 1. Generate Coordinates
@inbounds for k_g ∈ 1:Nz_nodes_tot, j_g ∈ 1:Ny_nodes_tot, i_g ∈ 1:Nx_nodes_tot
    i = i_g - NG; j = j_g - NG; k = k_g - NG
    xi = FT(i-1) / FT(Nx)
    eta = FT(j-1) / FT(Ny)
    zeta = FT(k-1) / FT(Nz)

    px = Lx * xi
    x[i_g, j_g, k_g] = px
    
    # Wedge height at x
    y_bot = (px > x_wedge) ? (px - x_wedge) * tan(alpha) : zero(FT)
    y[i_g, j_g, k_g] = y_bot + eta * (Ly - y_bot)
    
    z[i_g, j_g, k_g] = Lz * zeta
end

# 2. Ghost extrapolation (similar to parse_mesh.jl)
@inbounds for k ∈ NG+1:Nz_nodes+NG, j ∈ NG+1:Ny_nodes+NG, i ∈ 1:NG
    x[i, j, k] = 2*x[NG+1, j, k] - x[2*NG+2-i, j, k]
    y[i, j, k] = 2*y[NG+1, j, k] - y[2*NG+2-i, j, k]
    z[i, j, k] = 2*z[NG+1, j, k] - z[2*NG+2-i, j, k]
end
@inbounds for k ∈ NG+1:Nz_nodes+NG, j ∈ NG+1:Ny_nodes+NG, i ∈ Nx_nodes+NG+1:Nx_nodes_tot
    x[i, j, k] = 2*x[Nx_nodes+NG, j, k] - x[2*NG+2*Nx_nodes-i, j, k]
    y[i, j, k] = 2*y[Nx_nodes+NG, j, k] - y[2*NG+2*Nx_nodes-i, j, k]
    z[i, j, k] = 2*z[Nx_nodes+NG, j, k] - z[2*NG+2*Nx_nodes-i, j, k]
end
@inbounds for k ∈ NG+1:Nz_nodes+NG, j ∈ 1:NG, i ∈ NG+1:Nx_nodes+NG
    x[i, j, k] = 2*x[i, NG+1, k] - x[i, 2*NG+2-j, k]
    y[i, j, k] = 2*y[i, NG+1, k] - y[i, 2*NG+2-j, k]
    z[i, j, k] = 2*z[i, NG+1, k] - z[i, 2*NG+2-j, k]
end
@inbounds for k ∈ NG+1:Nz_nodes+NG, j ∈ Ny_nodes+NG+1:Ny_nodes_tot, i ∈ NG+1:Nx_nodes+NG
    x[i, j, k] = 2*x[i, Ny_nodes+NG, k] - x[i, 2*NG+2*Ny_nodes-j, k]
    y[i, j, k] = 2*y[i, Ny_nodes+NG, k] - y[i, 2*NG+2*Ny_nodes-j, k]
    z[i, j, k] = 2*z[i, Ny_nodes+NG, k] - z[i, 2*NG+2*Ny_nodes-j, k]
end

# 3. Compute Metrics
@inbounds for k ∈ 1:Nz_nodes_tot-1, j ∈ 1:Ny_nodes_tot-1, i ∈ 1:Nx_nodes_tot
    dx1 = x[i, j+1, k+1] - x[i, j, k]; dy1 = y[i, j+1, k+1] - y[i, j, k]; dz1 = z[i, j+1, k+1] - z[i, j, k]
    dx2 = x[i, j, k+1] - x[i, j+1, k]; dy2 = y[i, j, k+1] - y[i, j+1, k]; dz2 = z[i, j, k+1] - z[i, j+1, k]
    nx = FT(0.5) * (dy1 * dz2 - dz1 * dy2); ny = FT(0.5) * (dz1 * dx2 - dx1 * dz2); nz = FT(0.5) * (dx1 * dy2 - dy1 * dx2)
    Areai[i, j, k] = sqrt(nx*nx + ny*ny + nz*nz + 1e-20)
    nxi[i, j, k] = nx / Areai[i, j, k]; nyi[i, j, k] = ny / Areai[i, j, k]; nzi[i, j, k] = nz / Areai[i, j, k]
end
@inbounds for k ∈ 1:Nz_nodes_tot-1, j ∈ 1:Ny_nodes_tot, i ∈ 1:Nx_nodes_tot-1
    dx1 = x[i+1, j, k+1] - x[i, j, k]; dy1 = y[i+1, j, k+1] - y[i, j, k]; dz1 = z[i+1, j, k+1] - z[i, j, k]
    dx2 = x[i, j, k+1] - x[i+1, j, k]; dy2 = y[i, j, k+1] - y[i+1, j, k]; dz2 = z[i, j, k+1] - z[i+1, j, k]
    nx = FT(0.5) * (dy1 * dz2 - dz1 * dy2); ny = FT(0.5) * (dz1 * dx2 - dx1 * dz2); nz = FT(0.5) * (dx1 * dy2 - dy1 * dx2)
    Areaj[i, j, k] = sqrt(nx*nx + ny*ny + nz*nz + 1e-20)
    nxj[i, j, k] = -nx / Areaj[i, j, k]; nyj[i, j, k] = -ny / Areaj[i, j, k]; nzj[i, j, k] = -nz / Areaj[i, j, k]
end
@inbounds for k ∈ 1:Nz_nodes_tot, j ∈ 1:Ny_nodes_tot-1, i ∈ 1:Nx_nodes_tot-1
    dx1 = x[i+1, j+1, k] - x[i, j, k]; dy1 = y[i+1, j+1, k] - y[i, j, k]; dz1 = z[i+1, j+1, k] - z[i, j, k]
    dx2 = x[i, j+1, k] - x[i+1, j, k]; dy2 = y[i, j+1, k] - y[i+1, j, k]; dz2 = z[i, j+1, k] - z[i+1, j, k]
    nx = FT(0.5) * (dy1 * dz2 - dz1 * dy2); ny = FT(0.5) * (dz1 * dx2 - dx1 * dz2); nz = FT(0.5) * (dx1 * dy2 - dy1 * dx2)
    Areak[i, j, k] = sqrt(nx*nx + ny*ny + nz*nz + 1e-20)
    nxk[i, j, k] = nx / Areak[i, j, k]; nyk[i, j, k] = ny / Areak[i, j, k]; nzk[i, j, k] = nz / Areak[i, j, k]
end
@inbounds for k ∈ 1:Nz_nodes_tot-1, j ∈ 1:Ny_nodes_tot-1, i ∈ 1:Nx_nodes_tot-1
    dxdξ = FT(0.25) * (x[i+1,j,k]+x[i+1,j+1,k]+x[i+1,j,k+1]+x[i+1,j+1,k+1] - x[i,j,k]-x[i,j+1,k]-x[i,j,k+1]-x[i,j+1,k+1])
    dydξ = FT(0.25) * (y[i+1,j,k]+y[i+1,j+1,k]+y[i+1,j,k+1]+y[i+1,j+1,k+1] - y[i,j,k]-y[i,j+1,k]-y[i,j,k+1]-y[i,j+1,k+1])
    dzdξ = FT(0.25) * (z[i+1,j,k]+z[i+1,j+1,k]+z[i+1,j,k+1]+z[i+1,j+1,k+1] - z[i,j,k]-z[i,j+1,k]-z[i,j,k+1]-z[i,j+1,k+1])
    dxdη = FT(0.25) * (x[i,j+1,k]+x[i+1,j+1,k]+x[i,j+1,k+1]+x[i+1,j+1,k+1] - x[i,j,k]-x[i+1,j,k]-x[i,j,k+1]-x[i+1,j,k+1])
    dydη = FT(0.25) * (y[i,j+1,k]+y[i+1,j+1,k]+y[i,j+1,k+1]+y[i+1,j+1,k+1] - y[i,j,k]-y[i+1,j,k]-y[i,j,k+1]-y[i+1,j,k+1])
    dzdη = FT(0.25) * (z[i,j+1,k]+z[i+1,j+1,k]+z[i,j+1,k+1]+z[i+1,j+1,k+1] - z[i,j,k]-z[i+1,j,k]-z[i,j,k+1]-z[i+1,j,k+1])
    dxdζ = FT(0.25) * (x[i,j,k+1]+x[i+1,j,k+1]+x[i,j+1,k+1]+x[i+1,j+1,k+1] - x[i,j,k]-x[i+1,j,k]-x[i,j+1,k]-x[i+1,j+1,k])
    dydζ = FT(0.25) * (y[i,j,k+1]+y[i+1,j,k+1]+y[i,j+1,k+1]+y[i+1,j+1,k+1] - y[i,j,k]-y[i+1,j,k]-y[i,j+1,k]-y[i+1,j+1,k])
    dzdζ = FT(0.25) * (z[i,j,k+1]+z[i+1,j,k+1]+z[i,j+1,k+1]+z[i+1,j+1,k+1] - z[i,j,k]-z[i+1,j,k]-z[i,j+1,k]-z[i+1,j+1,k])
    local_vol = abs(dxdξ*(dydη*dzdζ - dydζ*dzdη) - dxdη*(dydξ*dzdζ - dydζ*dzdξ) + dxdζ*(dydξ*dzdη - dydη*dzdξ))
    Vol[i, j, k] = one(FT) / (local_vol + 1e-30)
end

h5open("MESH/oblique_metrics.h5", "w") do file
    file["Areai"] = Areai; file["nxi"] = nxi; file["nyi"] = nyi; file["nzi"] = nzi
    file["Areaj"] = Areaj; file["nxj"] = nxj; file["nyj"] = nyj; file["nzj"] = nzj
    file["Areak"] = Areak; file["nxk"] = nxk; file["nyk"] = nyk; file["nzk"] = nzk
    file["Vol"] = Vol
end

coords = zeros(FT, 3, Nx_nodes, Ny_nodes, Nz_nodes)
coords[1, :, :, :] = x[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
coords[2, :, :, :] = y[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
coords[3, :, :, :] = z[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
h5open("MESH/oblique_mesh.h5", "w") do file
    file["NG"] = NG; file["Nx"] = Nx; file["Ny"] = Ny; file["Nz"] = Nz
    file["coords"] = coords
    # Separate x/y/z datasets for XDMF X_Y_Z geometry (ParaView compatible)
    file["x"] = coords[1, :, :, :]
    file["y"] = coords[2, :, :, :]
    file["z"] = coords[3, :, :, :]
end

vtk_grid("MESH/oblique_mesh", x, y, z) do vtk
    vtk["InvVol"] = Vol
end
println("Oblique mesh done!")
