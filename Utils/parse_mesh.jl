using HDF5
using WriteVTK
using LinearAlgebra

# --- Configuration ---
const NG::Int64 = 4
const Nx::Int64 = 64
const Ny::Int64 = 64
const Nz::Int64 = 64
const Lx::FT = FT(2.0) * pi 
const Ly::FT = FT(2.0) * pi
const Lz::FT = FT(2.0) * pi
const vis::Bool = true
const compress_level::Int64 = 3

# --- Stretching parameters ---
const stretch_factor_y::FT = 3.5 

# --- Real-only grid dimensions (no ghost cells) ---
const Nx_nodes::Int64 = Nx + 1
const Ny_nodes::Int64 = Ny + 1
const Nz_nodes::Int64 = Nz + 1

# Real-only node coordinates
x = zeros(FT, Nx_nodes, Ny_nodes, Nz_nodes)
y = zeros(FT, Nx_nodes, Ny_nodes, Nz_nodes)
z = zeros(FT, Nx_nodes, Ny_nodes, Nz_nodes)

# --- Stretching function ---
function stretching_y(η::FT, Ly::FT, factor::FT)
    return Ly * sinh(factor * η) / sinh(factor)
end

println("Generating Flat Plate Turbulent Mesh (real nodes only)...")

# Generate real-only coordinates
@inbounds for k ∈ 1:Nz_nodes
    for j ∈ 1:Ny_nodes
        for i ∈ 1:Nx_nodes
            ξ::FT = FT(i-1) / FT(Nx)
            η::FT = FT(j-1) / FT(Ny)
            ζ::FT = FT(k-1) / FT(Nz)

            x[i, j, k] = Lx * ξ
            y[i, j, k] = Ly * η  # Uniform for TGV
            z[i, j, k] = Lz * ζ
        end
    end
end

# Output real-only coords: (3, Nx+1, Ny+1, Nz+1)
coords = zeros(FT, 3, Nx_nodes, Ny_nodes, Nz_nodes)
coords[1, :, :, :] = x
coords[2, :, :, :] = y
coords[3, :, :, :] = z
h5open("MESH/mesh.h5", "w") do file
    file["NG"] = NG
    file["Nx"] = Nx
    file["Ny"] = Ny
    file["Nz"] = Nz
    file["coords", compress=compress_level] = coords
    # Separate x/y/z datasets for XDMF X_Y_Z geometry (ParaView compatible)
    file["x"] = x
    file["y"] = y
    file["z"] = z
end

if vis
    vtk_grid("MESH/mesh", x, y, z) do vtk
    end
end

println("Parse mesh done! (real nodes only, no ghost cells or metrics)")
flush(stdout)