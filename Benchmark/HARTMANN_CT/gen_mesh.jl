# Generate the compact Cartesian mesh used by the resistive CT Hartmann test.

using HDF5

const NX = parse(Int, get(ENV, "HARTMANN_CT_NX", "4"))
const NY = parse(Int, get(ENV, "HARTMANN_CT_NY", "64"))
const NZ = parse(Int, get(ENV, "HARTMANN_CT_NZ", "4"))
const NG_MESH = 4
const LX = 1.0
const HALF_HEIGHT = 1.0
const LZ = 1.0

NX >= 2 || error("HARTMANN_CT_NX must be at least 2")
NY >= 16 || error("HARTMANN_CT_NY must be at least 16")
NZ >= 2 || error("HARTMANN_CT_NZ must be at least 2")

coords = zeros(Float64, 3, NX + 1, NY + 1, NZ + 1)
for k in 1:NZ+1, j in 1:NY+1, i in 1:NX+1
    coords[1, i, j, k] = LX * (i - 1) / NX
    coords[2, i, j, k] = -HALF_HEIGHT + 2HALF_HEIGHT * (j - 1) / NY
    coords[3, i, j, k] = LZ * (k - 1) / NZ
end

const BC_PERIODIC_MESH = Int32(2)
const BC_MHD_INSULATING_WALL_MESH = Int32(33)
face_bc = reshape(Int32[
    BC_PERIODIC_MESH,
    BC_PERIODIC_MESH,
    BC_MHD_INSULATING_WALL_MESH,
    BC_MHD_INSULATING_WALL_MESH,
    BC_PERIODIC_MESH,
    BC_PERIODIC_MESH,
], 1, 6)
bc_params = zeros(Float64, 1, 6, 20)
bc_params[1, 3, 1] = 1.0
bc_params[1, 4, 1] = 1.0
connectivity = zeros(Int32, 1, 6)

output_dir = joinpath(@__DIR__, "MESH")
mkpath(output_dir)
h5open(joinpath(output_dir, "block_connectivity.h5"), "w") do file
    file["Nblocks"] = Int32(1)
    file["Nx_b"] = Int32[NX]
    file["Ny_b"] = Int32[NY]
    file["Nz_b"] = Int32[NZ]
    file["face_bc"] = face_bc
    file["bc_params"] = bc_params
    file["connectivity"] = connectivity
end
h5open(joinpath(output_dir, "mesh_b0.h5"), "w") do file
    file["NG"] = Int32(NG_MESH)
    file["Nx"] = Int32(NX)
    file["Ny"] = Int32(NY)
    file["Nz"] = Int32(NZ)
    file["coords"] = coords
end

println("HARTMANN_CT_MESH nx=$NX ny=$NY nz=$NZ path=$output_dir")
