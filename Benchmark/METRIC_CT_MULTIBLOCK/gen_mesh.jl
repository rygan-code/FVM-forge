using HDF5

include(joinpath(@__DIR__, "mesh_quality.jl"))

const METRIC_CT_MULTIBLOCK_NG = 4
const METRIC_CT_MULTIBLOCK_BC_INTERBLOCK = Int64(0)
const METRIC_CT_MULTIBLOCK_BC_PERIODIC = Int64(2)
const METRIC_CT_MULTIBLOCK_BC_ZERO_GRADIENT = Int64(11)

function metric_ct_generate_multiblock_mesh(
    nx, ny, nz, out_dir; topology=:supported,
)
    all(dimension -> dimension isa Integer && dimension > 0, (nx, ny, nz)) ||
        throw(ArgumentError("mesh dimensions must be positive integers"))

    mkpath(out_dir)
    xis = range(0.0, 2pi; length=nx + 1)
    zetas = range(0.0, 2pi; length=nz + 1)
    for (block_id, eta_bounds) in enumerate(((0.0, pi), (pi, 2pi)))
        etas = range(eta_bounds...; length=ny + 1)
        coords = zeros(Float64, 3, nx + 1, ny + 1, nz + 1)
        for k in eachindex(zetas), j in eachindex(etas), i in eachindex(xis)
            coords[:, i, j, k] .= metric_ct_multiblock_point(
                xis[i], etas[j], zetas[k],
            )
        end
        path = joinpath(out_dir, "mesh_b$(block_id - 1).h5")
        h5open(path, "w") do file
            file["NG"] = Int64(METRIC_CT_MULTIBLOCK_NG)
            file["Nx"] = Int64(nx)
            file["Ny"] = Int64(ny)
            file["Nz"] = Int64(nz)
            file["coords"] = coords
            file["x"] = coords[1, :, :, :]
            file["y"] = coords[2, :, :, :]
            file["z"] = coords[3, :, :, :]
        end
    end

    topology in (:supported, :cross_type, :reverse_tangent) || throw(
        ArgumentError("unknown metric CT topology: $topology"),
    )
    face_bc = fill(METRIC_CT_MULTIBLOCK_BC_PERIODIC, 2, 6)
    face_bc[1, 3] = METRIC_CT_MULTIBLOCK_BC_ZERO_GRADIENT
    local_face = topology === :cross_type ? 5 : 4
    face_bc[1, local_face] = METRIC_CT_MULTIBLOCK_BC_INTERBLOCK
    face_bc[2, 3] = METRIC_CT_MULTIBLOCK_BC_INTERBLOCK
    face_bc[2, 4] = METRIC_CT_MULTIBLOCK_BC_ZERO_GRADIENT
    bc_params = zeros(Float64, 2, 6, 20)
    connectivity_rows = Int64[
        0 local_face 1 3
        1 3 0 local_face
    ]
    reverse_tan = topology === :reverse_tangent ? Int64[1, 1] : Int64[0, 0]
    flip_normal = Int64[0, 0]
    h5open(joinpath(out_dir, "block_connectivity.h5"), "w") do file
        file["Nblocks"] = Int64(2)
        file["Nx_b"] = Int64[nx, nx]
        file["Ny_b"] = Int64[ny, ny]
        file["Nz_b"] = Int64[nz, nz]
        file["face_bc"] = face_bc
        file["bc_params"] = bc_params
        file["connectivity"] = connectivity_rows
        file["reverse_tan"] = reverse_tan
        file["flip_normal"] = flip_normal
    end

    println("Curved two-block metric-CT mesh written to $out_dir: 2 x $(nx)x$(ny)x$(nz)")
    return out_dir
end

function metric_ct_multiblock_mesh_output_dir(root, resolution)
    resolution isa Integer && resolution > 0 || throw(
        ArgumentError("resolution must be a positive integer"),
    )
    return joinpath(root, "N$(Int(resolution))", "MESH")
end

function _metric_ct_multiblock_gen_mesh_main(args)
    length(args) <= 4 || error(
        "usage: julia gen_mesh.jl [Nx [Ny [Nz [output_dir]]]]",
    )
    nx = length(args) >= 1 ? parse(Int, args[1]) : 16
    ny = length(args) >= 2 ? parse(Int, args[2]) : nx
    nz = length(args) >= 3 ? parse(Int, args[3]) : nx
    out_dir = length(args) >= 4 ? args[4] : joinpath(@__DIR__, "MESH")
    metric_ct_generate_multiblock_mesh(nx, ny, nz, out_dir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    _metric_ct_multiblock_gen_mesh_main(ARGS)
end
