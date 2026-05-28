# gen_cavity_mesh.jl — Generate 2D lid-driven cavity mesh (extruded to 3D with 1 cell in z)
# Output: MESH/mesh_b0.h5, MESH/metrics_b0.h5, MESH/block_connectivity.h5

using HDF5

# ─── Configuration ───
const NG = 4
const Nx = 64          # cells in x (streamwise)
const Ny = 64          # cells in y (wall-normal)
const Nz = 8           # minimum 2*NG for periodic z
const Lx_val = 1.0     # cavity length
const Ly_val = 1.0     # cavity height
const Lz_val = 8.0/64  # thin slab in z (Lz/Lx = Nz/Nx)

# ─── Node dimensions ───
const Nx_nodes = Nx + 1
const Ny_nodes = Ny + 1
const Nz_nodes = Nz + 1
const Nx_tot = Nx_nodes + 2*NG
const Ny_tot = Ny_nodes + 2*NG
const Nz_tot = Nz_nodes + 2*NG

println("Generating 2D Cavity mesh: $(Nx)×$(Ny)×$(Nz), domain $(Lx_val)×$(Ly_val)×$(Lz_val)")

# ─── Generate coordinates (node-centered, with ghosts) ───
dx = FT(Lx_val / Nx)
dy = FT(Ly_val / Ny)
dz = FT(Lz_val / Nz)

x = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
y = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
z = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
    x[i, j, k] = dx * (i - NG - 1)
    y[i, j, k] = dy * (j - NG - 1)
    z[i, j, k] = dz * (k - NG - 1)
end

# ─── Compute metrics (Cartesian → trivial) ───
Areai = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nxi   = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nyi   = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nzi   = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

Areaj = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nxj   = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nyj   = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nzj   = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

Areak = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nxk   = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nyk   = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nzk   = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

Vol = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
    if j < Ny_tot && k < Nz_tot
        Areai[i, j, k] = dy * dz
        nxi[i, j, k] = one(FT)
    end
    if i < Nx_tot && k < Nz_tot
        Areaj[i, j, k] = dx * dz
        nyj[i, j, k] = one(FT)
    end
    if i < Nx_tot && j < Ny_tot
        Areak[i, j, k] = dx * dy
        nzk[i, j, k] = one(FT)
    end
    if i < Nx_tot && j < Ny_tot && k < Nz_tot
        Vol[i, j, k] = one(FT) / (dx * dy * dz)  # J = 1/Volume
    end
end

# ─── Write mesh ───
mkpath("MESH")

# Block connectivity for single-block cavity
# face_bc layout: [Nblocks, 6] — face ordering: ξ-, ξ+, η-, η+, ζ-, ζ+
# BC types from bc_types.jl:
#   BC_AC_WALL = 20, BC_AC_LID = 21, BC_PERIODIC = 2
const BC_AC_WALL = Int32(20)
const BC_AC_LID  = Int32(21)
const BC_PERIODIC = Int32(2)

face_bc = zeros(Int32, 1, 6)
face_bc[1, 1] = BC_AC_WALL    # ξ- : left wall
face_bc[1, 2] = BC_AC_WALL    # ξ+ : right wall
face_bc[1, 3] = BC_AC_WALL    # η- : bottom wall
face_bc[1, 4] = BC_AC_LID     # η+ : top lid (moving)
face_bc[1, 5] = BC_PERIODIC   # ζ- : periodic
face_bc[1, 6] = BC_PERIODIC   # ζ+ : periodic

# BC params: [Nblocks, 6, N_BC_PARAMS=17]
const N_BC_PARAMS = 17
const BCP_AC_U_LID = 15
const BCP_AC_V_LID = 16
const BCP_AC_W_LID = 17

bc_params = zeros(FT, 1, 6, N_BC_PARAMS)
# Lid velocity: u=1.0, v=0, w=0 on η+ face
bc_params[1, 4, BCP_AC_U_LID] = one(FT)   # u_lid = 1.0
bc_params[1, 4, BCP_AC_V_LID] = zero(FT)   # v_lid = 0
bc_params[1, 4, BCP_AC_W_LID] = zero(FT)   # w_lid = 0

# Connectivity (single block, no inter-block connections)
connectivity = zeros(Int32, 1, 6)  # all zeros → no connections

h5open("MESH/block_connectivity.h5", "w") do f
    f["Nblocks"] = Int32(1)
    f["Nx_b"] = Int32[Nx]
    f["Ny_b"] = Int32[Ny]
    f["Nz_b"] = Int32[Nz]
    f["face_bc"] = face_bc
    f["bc_params"] = bc_params
    f["connectivity"] = connectivity
end
println("  → MESH/block_connectivity.h5")

# Mesh file
coords = zeros(FT, 3, Nx_nodes, Ny_nodes, Nz_nodes)
coords[1, :, :, :] = x[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
coords[2, :, :, :] = y[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]
coords[3, :, :, :] = z[1+NG:Nx_nodes+NG, 1+NG:Ny_nodes+NG, 1+NG:Nz_nodes+NG]

h5open("MESH/mesh_b0.h5", "w") do f
    f["NG"] = NG
    f["Nx"] = Nx
    f["Ny"] = Ny
    f["Nz"] = Nz
    f["coords"] = coords
    f["x"] = coords[1, :, :, :]
    f["y"] = coords[2, :, :, :]
    f["z"] = coords[3, :, :, :]
end
println("  → MESH/mesh_b0.h5")

# Metrics file
h5open("MESH/metrics_b0.h5", "w") do f
    f["Areai"] = Areai; f["nxi"] = nxi; f["nyi"] = nyi; f["nzi"] = nzi
    f["Areaj"] = Areaj; f["nxj"] = nxj; f["nyj"] = nyj; f["nzj"] = nzj
    f["Areak"] = Areak; f["nxk"] = nxk; f["nyk"] = nyk; f["nzk"] = nzk
    f["Vol"] = Vol
end
println("  → MESH/metrics_b0.h5")

println("Cavity mesh generation complete!")
