# Benchmark/ORSZAG_TANG_2D/gen_mesh.jl — Generate mesh for 2D Orszag-Tang vortex
# Usage: julia Benchmark/ORSZAG_TANG_2D/gen_mesh.jl [Nx] [Nz]
# Output: Benchmark/ORSZAG_TANG_2D/MESH/ directory
#
# Domain: [0, 2π]² in (x, y), with a thin quasi-2D slab in z (Nz=8 default,
# minimum for the 7-point reconstruction stencil). Triply periodic.
# Classic 2D MHD vortex benchmark (Orszag & Tang, 1979).

using HDF5

const FT = Float64

# ─── Configuration ───
const NG = 4
const Nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 256
const Ny = Nx                           # square domain in x-y
const Nz = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8   # quasi-2D slab
const Lx_val = 2.0 * pi                 # [0, 2π]
const Ly_val = 2.0 * pi
const Lz_val = 2.0 * pi                 # full period (small Nz, but periodic)

const Nx_nodes = Nx + 1
const Ny_nodes = Ny + 1
const Nz_nodes = Nz + 1
const Nx_tot = Nx_nodes + 2*NG
const Ny_tot = Ny_nodes + 2*NG
const Nz_tot = Nz_nodes + 2*NG

println("Generating 2D Orszag-Tang mesh: $(Nx)×$(Ny)×$(Nz) (quasi-2D, periodic)")

dx = FT(Lx_val / Nx)
dy = FT(Ly_val / Ny)
dz = FT(Lz_val / Nz)

# ─── Coordinate arrays (node-based, including ghost nodes) ───
x = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
y = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
z = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
    x[i, j, k] = dx * (i - NG - 1)
    y[i, j, k] = dy * (j - NG - 1)
    z[i, j, k] = dz * (k - NG - 1)
end

# ─── Metrics (uniform Cartesian) ───
Areai = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nxi = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nyi = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nzi = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
Areaj = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nxj = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nyj = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nzj = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
Areak = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nxk = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
nyk = zeros(FT, Nx_tot, Ny_tot, Nz_tot); nzk = zeros(FT, Nx_tot, Ny_tot, Nz_tot)
Vol = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
    if j < Ny_tot && k < Nz_tot; Areai[i,j,k] = dy*dz; nxi[i,j,k] = one(FT); end
    if i < Nx_tot && k < Nz_tot; Areaj[i,j,k] = dx*dz; nyj[i,j,k] = one(FT); end
    if i < Nx_tot && j < Ny_tot; Areak[i,j,k] = dx*dy; nzk[i,j,k] = one(FT); end
    if i < Nx_tot && j < Ny_tot && k < Nz_tot; Vol[i,j,k] = one(FT)/(dx*dy*dz); end
end

# ─── Boundary conditions: triply periodic ───
const BC_PERIODIC = Int32(2)
const N_BC_PARAMS_MESH = 20

face_bc = fill(BC_PERIODIC, 1, 6)
bc_params = zeros(FT, 1, 6, N_BC_PARAMS_MESH)
connectivity = zeros(Int32, 1, 6)

# ─── Output ───
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

println("✓ 2D Orszag-Tang mesh saved to $(out_dir)/")
println("  Nx=$(Nx), Ny=$(Ny), Nz=$(Nz), NG=$(NG)")
println("  dx=$(dx), domain=[0, $(Lx_val)]²")
println("  BCs: triply periodic (quasi-2D slab in z)")
