# Generate a single-block, triply periodic warped mesh for metric CT tests.

using HDF5

const METRIC_CT_MESH_FT = Float64
const METRIC_CT_MESH_NG = 4
const METRIC_CT_MESH_LENGTH = METRIC_CT_MESH_FT(2pi)
const METRIC_CT_MESH_WARP = METRIC_CT_MESH_FT(0.12)
const METRIC_CT_BC_PERIODIC = Int32(2)

function metric_ct_mesh_output_dir(base, resolution::Integer)
    resolution > 0 || throw(ArgumentError("resolution must be positive"))
    return joinpath(base, "MESH_N$resolution")
end

function metric_ct_generate_mesh(nx, ny, nz, out_dir)
    all(dimension -> dimension isa Integer && dimension > 0, (nx, ny, nz)) ||
        throw(ArgumentError("mesh dimensions must be positive integers"))

    FT = METRIC_CT_MESH_FT
    length_scale = METRIC_CT_MESH_LENGTH
    warp = METRIC_CT_MESH_WARP
    coords = zeros(FT, 3, nx + 1, ny + 1, nz + 1)
    for k in 0:nz, j in 0:ny, i in 0:nx
        xi = length_scale * i / nx
        eta = length_scale * j / ny
        zeta = length_scale * k / nz
        coords[1,i+1,j+1,k+1] = xi + warp * sin(xi) * sin(eta)
        coords[2,i+1,j+1,k+1] = eta + warp * sin(eta) * sin(zeta)
        coords[3,i+1,j+1,k+1] = zeta + warp * sin(zeta) * sin(xi)
    end

    mkpath(out_dir)
    h5open(joinpath(out_dir, "mesh_b0.h5"), "w") do file
        file["NG"] = METRIC_CT_MESH_NG
        file["Nx"] = nx
        file["Ny"] = ny
        file["Nz"] = nz
        file["coords"] = coords
        file["x"] = coords[1,:,:,:]
        file["y"] = coords[2,:,:,:]
        file["z"] = coords[3,:,:,:]
    end

    face_bc = fill(METRIC_CT_BC_PERIODIC, 1, 6)
    bc_params = zeros(FT, 1, 6, 20)
    connectivity = zeros(Int32, 1, 6)
    h5open(joinpath(out_dir, "block_connectivity.h5"), "w") do file
        file["Nblocks"] = Int32(1)
        file["Nx_b"] = Int32[nx]
        file["Ny_b"] = Int32[ny]
        file["Nz_b"] = Int32[nz]
        file["face_bc"] = face_bc
        file["bc_params"] = bc_params
        file["connectivity"] = connectivity
    end

    println("Warped metric-CT mesh written to $(out_dir): $(nx)x$(ny)x$(nz)")
    return out_dir
end

function _metric_ct_gen_mesh_main(args)
    length(args) <= 4 || error(
        "usage: julia gen_mesh.jl [Nx [Ny [Nz [output_dir]]]]",
    )
    nx = length(args) >= 1 ? parse(Int, args[1]) : 16
    ny = length(args) >= 2 ? parse(Int, args[2]) : nx
    nz = length(args) >= 3 ? parse(Int, args[3]) : nx
    out_dir = length(args) >= 4 ? args[4] : joinpath(@__DIR__, "MESH")
    metric_ct_generate_mesh(nx, ny, nz, out_dir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    _metric_ct_gen_mesh_main(ARGS)
end
