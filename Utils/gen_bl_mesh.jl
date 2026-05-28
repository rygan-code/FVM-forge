using HDF5
using LinearAlgebra
using StaticArrays

# --- Configuration ---
const NG::Int64 = 4
const Nx::Int64 = 128
const Ny::Int64 = 192
const Nz::Int64 = 4

const Lx::FT = FT(2.0)
const Ly::FT = FT(0.5)
const Lz::FT = FT(0.1)

const Nx_tot = Nx + 2*NG
const Ny_tot = Ny + 2*NG
const Nz_tot = Nz + 2*NG

const Nx_nodes_tot = Nx + 1 + 2*NG
const Ny_nodes_tot = Ny + 1 + 2*NG
const Nz_nodes_tot = Nz + 1 + 2*NG

# Stretching parameter
const beta = FT(2.0)

function stretch(yi_norm, H, b)
    return H * (one(FT) + (tanh(b * (yi_norm - one(FT))) / tanh(b)))
end

function generate_bl_mesh()
    println("Generating Flat Plate Boundary Layer Mesh ($Nx x $Ny x $Nz)...")
    
    x_nodes = collect(range(zero(FT), Lx, length=Nx+1))
    y_nodes_uniform = collect(range(zero(FT), one(FT), length=Ny+1))
    y_nodes = [stretch(yi, Ly, beta) for yi in y_nodes_uniform]
    z_nodes = collect(range(zero(FT), Lz, length=Nz+1))

    x = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
    y = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
    z = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)

    dx = Lx/Nx
    dz = Lz/Nz

    @inbounds for k_g ∈ 1:Nz_nodes_tot, j_g ∈ 1:Ny_nodes_tot, i_g ∈ 1:Nx_nodes_tot
        i = i_g - NG; j = j_g - NG; k = k_g - NG
        x[i_g, j_g, k_g] = (i-1) * dx
        if j >= 1 && j <= Ny+1
            y[i_g, j_g, k_g] = y_nodes[Int(j)]
        elseif j < 1
            dy0 = y_nodes[2] - y_nodes[1]
            y[i_g, j_g, k_g] = y_nodes[1] + (j-1) * dy0
        else
            dyn = y_nodes[end] - y_nodes[end-1]
            y[i_g, j_g, k_g] = y_nodes[end] + (j - (Ny+1)) * dyn
        end
        z[i_g, j_g, k_g] = (k-1) * dz
    end

    Areai = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nxi = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nyi = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nzi = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
    Areaj = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nxj = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nyj = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nzj = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
    Areak = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nxk = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nyk = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot); nzk = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
    Vol = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)

    @inbounds for k_g ∈ 1:Nz_nodes_tot-1, j_g ∈ 1:Ny_nodes_tot-1, i_g ∈ 1:Nx_nodes_tot-1
        dy_j = y[i_g, j_g+1, k_g] - y[i_g, j_g, k_g]
        Areai[i_g, j_g, k_g] = dy_j * dz; nxi[i_g, j_g, k_g] = 1.0
        Areaj[i_g, j_g, k_g] = dx * dz; nyj[i_g, j_g, k_g] = 1.0
        Areak[i_g, j_g, k_g] = dx * dy_j; nzk[i_g, j_g, k_g] = 1.0
        Vol[i_g, j_g, k_g] = one(FT) / (dx*dy_j*dz)
    end
    # Boundary faces for i+1, j+1, k+1 are needed too
    @inbounds for k_g ∈ 1:Nz_nodes_tot, j_g ∈ 1:Ny_nodes_tot, i_g ∈ 1:Nx_nodes_tot
        if i_g == Nx_nodes_tot; dy_j = y[i_g, j_g+1 < Ny_nodes_tot ? j_g+1 : j_g, k_g] - y[i_g, j_g, k_g]
            Areai[i_g, j_g, k_g] = dy_j * dz; nxi[i_g, j_g, k_g] = 1.0
        end
        if j_g == Ny_nodes_tot # Top
            Areaj[i_g, j_g, k_g] = dx * dz; nyj[i_g, j_g, k_g] = 1.0
        end
        if k_g == Nz_nodes_tot
            Areak[i_g, j_g, k_g] = dx * (y[i_g, j_g+1 < Ny_nodes_tot ? j_g+1 : j_g, k_g] - y[i_g, j_g, k_g]); nzk[i_g, j_g, k_g] = 1.0
        end
    end

    mkpath("MESH")
    h5open("MESH/bl_metrics.h5", "w") do file
        file["Areai"] = Areai; file["nxi"] = nxi; file["nyi"] = nyi; file["nzi"] = nzi
        file["Areaj"] = Areaj; file["nxj"] = nxj; file["nyj"] = nyj; file["nzj"] = nzj
        file["Areak"] = Areak; file["nxk"] = nxk; file["nyk"] = nyk; file["nzk"] = nzk
        file["Vol"] = Vol
    end

    coords = zeros(FT, 3, Nx+1, Ny+1, Nz+1)
    coords[1, :, :, :] = x[1+NG:Nx+1+NG, 1+NG:Ny+1+NG, 1+NG:Nz+1+NG]
    coords[2, :, :, :] = y[1+NG:Nx+1+NG, 1+NG:Ny+1+NG, 1+NG:Nz+1+NG]
    coords[3, :, :, :] = z[1+NG:Nx+1+NG, 1+NG:Ny+1+NG, 1+NG:Nz+1+NG]
    
    h5open("MESH/bl_mesh.h5", "w") do file
        file["NG"] = NG; file["Nx"] = Nx; file["Ny"] = Ny; file["Nz"] = Nz
        file["coords"] = coords
        # Separate x/y/z datasets for XDMF X_Y_Z geometry (ParaView compatible)
        file["x"] = coords[1, :, :, :]
        file["y"] = coords[2, :, :, :]
        file["z"] = coords[3, :, :, :]
    end

    println("BL mesh and metrics fixed!")
end

generate_bl_mesh()
