# Benchmark/HARTMANN/gen_mesh.jl — Generate stretched mesh for Hartmann channel flow
# Usage: julia Benchmark/HARTMANN/gen_mesh.jl
# Output: Benchmark/HARTMANN/MESH/ directory with mesh, metrics, connectivity files

using HDF5

const FT = Float64

# ─── Configuration ───
const NG = 4
const Nx = 16           # periodic direction x
const Ny = 256          # stretched wall-bounded direction y
const Nz = 16           # periodic direction z
const Lx_val = 1.0     # periodic length in x
const Ly_val = 2.0     # channel height y ∈ [-1, 1]
const Lz_val = 1.0     # periodic length in z

const Nx_nodes = Nx + 1
const Ny_nodes = Ny + 1
const Nz_nodes = Nz + 1
const Nx_tot = Nx_nodes + 2*NG
const Ny_tot = Ny_nodes + 2*NG
const Nz_tot = Nz_nodes + 2*NG

println("Generating Hartmann mesh: $(Nx)×$(Ny)×$(Nz) (stretched in y)")

x = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
y = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
z = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

s = 2.5  # stretching factor for y-direction

for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
    # x is uniform in [0, Lx_val]
    x[i, j, k] = (Lx_val / Nx) * (i - NG - 1)
    
    # y is stretched in [-1, 1] using tanh
    # ξ ranges from -0.5 (at j = NG+1) to 0.5 (at j = Ny_nodes+NG)
    ξ = (j - NG - 1) / Ny - 0.5
    y[i, j, k] = tanh(s * 2.0 * ξ) / tanh(s)
    
    # z is uniform in [0, Lz_val]
    z[i, j, k] = (Lz_val / Nz) * (k - NG - 1)
end

# ─── Metrics (Cartesian with non-uniform y) ───
Areai = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nxi = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nyi = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nzi = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
Areaj = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nxj = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nyj = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nzj = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
Areak = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nxk = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nyk = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nzk = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
Vol = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

dx = FT(Lx_val / Nx)
dz = FT(Lz_val / Nz)

for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
    dy_j = (j < Ny_tot) ? FT(y[i, j+1, k] - y[i, j, k]) : FT(0.0)
    
    # ξ-faces: normal = (1, 0, 0), area = dy_j·dz
    if j < Ny_tot && k < Nz_tot
        Areai[i, j, k] = dy_j * dz
        nxi[i, j, k] = one(FT)
    end
    # η-faces: normal = (0, 1, 0), area = dx·dz
    if i < Nx_tot && k < Nz_tot
        Areaj[i, j, k] = dx * dz
        nyj[i, j, k] = one(FT)
    end
    # ζ-faces: normal = (0, 0, 1), area = dx·dy_j
    if i < Nx_tot && j < Ny_tot
        Areak[i, j, k] = dx * dy_j
        nzk[i, j, k] = one(FT)
    end
    # Cell volume (stored as 1/Vol in the solver)
    if i < Nx_tot && j < Ny_tot && k < Nz_tot
        Vol[i, j, k] = one(FT) / (dx * dy_j * dz)
    end
end

# ─── Boundary Conditions ───
const BC_PERIODIC = Int32(2)
const BC_MHD_INSULATING_WALL = Int32(33)
const N_BC_PARAMS = 20
const BCP_TW = 1

face_bc = zeros(Int32, 1, 6)
face_bc[1, 1] = BC_PERIODIC   # ξ- left
face_bc[1, 2] = BC_PERIODIC   # ξ+ right
face_bc[1, 3] = BC_MHD_INSULATING_WALL  # η- bottom insulating wall
face_bc[1, 4] = BC_MHD_INSULATING_WALL  # η+ top insulating wall
face_bc[1, 5] = BC_PERIODIC   # ζ- bottom
face_bc[1, 6] = BC_PERIODIC   # ζ+ top

bc_params = zeros(FT, 1, 6, N_BC_PARAMS)
bc_params[1, 3, BCP_TW] = 1.0  # Tw bottom wall
bc_params[1, 4, BCP_TW] = 1.0  # Tw top wall

connectivity = zeros(Int32, 1, 6)

out_dir = joinpath(@__DIR__, "MESH")
mkpath(out_dir)

h5open(joinpath(out_dir, "block_connectivity.h5"), "w") do f
    f["Nblocks"] = Int32(1)
    f["Nx_b"] = Int32[Nx]; f["Ny_b"] = Int32[Ny]; f["Nz_b"] = Int32[Nz]
    f["face_bc"] = face_bc; f["bc_params"] = bc_params; f["connectivity"] = connectivity
end

coords = zeros(FT, 3, Nx_nodes, Ny_nodes, Nz_nodes)
coords[1,:,:,:] = x[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
coords[2,:,:,:] = y[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
coords[3,:,:,:] = z[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]

h5open(joinpath(out_dir, "mesh_b0.h5"), "w") do f
    f["NG"] = NG; f["Nx"] = Nx; f["Ny"] = Ny; f["Nz"] = Nz; f["coords"] = coords
    f["x"] = coords[1,:,:,:]; f["y"] = coords[2,:,:,:]; f["z"] = coords[3,:,:,:]
end

h5open(joinpath(out_dir, "metrics_b0.h5"), "w") do f
    f["Areai"]=Areai; f["nxi"]=nxi; f["nyi"]=nyi; f["nzi"]=nzi
    f["Areaj"]=Areaj; f["nxj"]=nxj; f["nyj"]=nyj; f["nzj"]=nzj
    f["Areak"]=Areak; f["nxk"]=nxk; f["nyk"]=nyk; f["nzk"]=nzk
    f["Vol"]=Vol
end

println("✓ Hartmann mesh saved to $(out_dir)/")
