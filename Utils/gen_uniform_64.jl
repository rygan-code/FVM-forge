using HDF5, WriteVTK, LinearAlgebra

# --- uniform_mesh_64 Configuration ---
const NG::Int64 = 4
const Nx::Int64 = 64
const Ny::Int64 = 64
const Nz::Int64 = 64
const Lx::FT = FT(2.0) * pi
const Ly::FT = FT(2.0) * pi
const Lz::FT = FT(2.0) * pi

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

println("Generating 64^3 Uniform Mesh...")

dx = Lx/Nx; dy = Ly/Ny; dz = Lz/Nz

# 1. Generate Coordinates
@inbounds for k_g ∈ 1:Nz_nodes_tot, j_g ∈ 1:Ny_nodes_tot, i_g ∈ 1:Nx_nodes_tot
    i = i_g - NG; j = j_g - NG; k = k_g - NG
    x[i_g, j_g, k_g] = dx * (i-1)
    y[i_g, j_g, k_g] = dy * (j-1)
    z[i_g, j_g, k_g] = dz * (k-1)
end

# 3. Compute Metrics
@inbounds for k ∈ 1:Nz_nodes_tot, j ∈ 1:Ny_nodes_tot, i ∈ 1:Nx_nodes_tot
    if j < Ny_nodes_tot && k < Nz_nodes_tot
        Areai[i, j, k] = dy*dz; nxi[i, j, k] = 1.0; nyi[i, j, k] = 0.0; nzi[i, j, k] = 0.0
    end
    if i < Nx_nodes_tot && k < Nz_nodes_tot
        Areaj[i, j, k] = dx*dz; nxj[i, j, k] = 0.0; nyj[i, j, k] = 1.0; nzj[i, j, k] = 0.0
    end
    if i < Nx_nodes_tot && j < Ny_nodes_tot
        Areak[i, j, k] = dx*dy; nxk[i, j, k] = 0.0; nyk[i, j, k] = 0.0; nzk[i, j, k] = 1.0
    end
    if i < Nx_nodes_tot && j < Ny_nodes_tot && k < Nz_nodes_tot
        Vol[i, j, k] = 1.0 / (dx*dy*dz)
    end
end

mkpath("MESH")
h5open("MESH/uniform_metrics_64.h5", "w") do file
    file["Areai"] = Areai; file["nxi"] = nxi; file["nyi"] = nyi; file["nzi"] = nzi
    file["Areaj"] = Areaj; file["nxj"] = nxj; file["nyj"] = nyj; file["nzj"] = nzj
    file["Areak"] = Areak; file["nxk"] = nxk; file["nyk"] = nyk; file["nzk"] = nzk
    file["Vol"] = Vol
end

coords = zeros(FT, 3, Nx_nodes, Ny_nodes, Nz_nodes)
coords[1, :, :, :] = x[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
coords[2, :, :, :] = y[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
coords[3, :, :, :] = z[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
h5open("MESH/uniform_mesh_64.h5", "w") do file
    file["NG"] = NG; file["Nx"] = Nx; file["Ny"] = Ny; file["Nz"] = Nz
    file["coords"] = coords
    # Separate x/y/z datasets for XDMF X_Y_Z geometry (ParaView compatible)
    file["x"] = coords[1, :, :, :]
    file["y"] = coords[2, :, :, :]
    file["z"] = coords[3, :, :, :]
end

println("64^3 uniform mesh done!")
