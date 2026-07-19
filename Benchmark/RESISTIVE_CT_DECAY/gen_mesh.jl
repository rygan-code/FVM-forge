using HDF5

const NX = parse(Int, get(ENV, "RESISTIVE_CT_DECAY_NX", "32"))
const NY = parse(Int, get(ENV, "RESISTIVE_CT_DECAY_NY", "4"))
const NZ = parse(Int, get(ENV, "RESISTIVE_CT_DECAY_NZ", "4"))
const NG_MESH = 4

NX >= 16 || error("RESISTIVE_CT_DECAY_NX must be at least 16")
NY >= 2 || error("RESISTIVE_CT_DECAY_NY must be at least 2")
NZ >= 2 || error("RESISTIVE_CT_DECAY_NZ must be at least 2")

coords = zeros(Float64, 3, NX + 1, NY + 1, NZ + 1)
for k in 1:NZ+1, j in 1:NY+1, i in 1:NX+1
    coords[1,i,j,k] = (i - 1) / NX
    coords[2,i,j,k] = (j - 1) / NY
    coords[3,i,j,k] = (k - 1) / NZ
end

const BC_PERIODIC_MESH = Int32(2)
face_bc = fill(BC_PERIODIC_MESH, 1, 6)
bc_params = zeros(Float64, 1, 6, 20)
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

println("RESISTIVE_CT_DECAY_MESH nx=$NX ny=$NY nz=$NZ path=$output_dir")
