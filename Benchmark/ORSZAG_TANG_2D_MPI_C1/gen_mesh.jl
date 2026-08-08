# Generate a two-block, periodic, deliberately skewed mesh for the 2D
# Orszag-Tang MPI validation.  The solver builds ghost coordinates and metrics
# from the node coordinates; this file writes only the physical mesh topology.

using HDF5

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "core", "structured_interface_transform.jl"))

const FT = Float64
const NG = 4
const L = FT(2pi)
const BC_PERIODIC = Int32(2)
const BC_INTERBLOCK = Int32(0)
const N_BC_PARAMS = 20

@inline function warped_point(xi::FT, eta::FT, zeta::FT)
    a = FT(0.45)
    b = FT(0.45)
    c = FT(0.12)
    return (
        xi + a * sin(eta),
        eta + b * sin(xi) + c * sin(FT(2) * eta),
        zeta,
    )
end

function write_block(path, nx, ny, nz, eta_lo, eta_hi)
    xis = range(zero(FT), L; length=nx + 1)
    etas = range(eta_lo, eta_hi; length=ny + 1)
    zetas = range(zero(FT), L; length=nz + 1)
    coords = zeros(FT, 3, nx + 1, ny + 1, nz + 1)
    for k in eachindex(zetas), j in eachindex(etas), i in eachindex(xis)
        point = warped_point(xis[i], etas[j], zetas[k])
        coords[1, i, j, k] = point[1]
        coords[2, i, j, k] = point[2]
        coords[3, i, j, k] = point[3]
    end

    h5open(path, "w") do file
        file["NG"] = Int64(NG)
        file["Nx"] = Int64(nx)
        file["Ny"] = Int64(ny)
        file["Nz"] = Int64(nz)
        file["coords"] = coords
        file["x"] = coords[1, :, :, :]
        file["y"] = coords[2, :, :, :]
        file["z"] = coords[3, :, :, :]
    end
end

function generate_mesh(nx::Integer, ny::Integer, nz::Integer, out_dir::AbstractString)
    nx > 0 && ny > 0 && nz > 0 || error("mesh dimensions must be positive")
    out_dir = abspath(out_dir)
    mkpath(out_dir)

    write_block(joinpath(out_dir, "mesh_b0.h5"), nx, ny, nz, zero(FT), FT(pi))
    write_block(joinpath(out_dir, "mesh_b1.h5"), nx, ny, nz, FT(pi), L)

    # Face ids are: x-low, x-high, y-low, y-high, z-low, z-high.
    # The four y faces form a periodic two-block ring.  The first two rows
    # cross the physical period and therefore carry an image translation.
    connectivity_rows = Int64[
        0 3 1 4
        1 4 0 3
        0 4 1 3
        1 3 0 4
    ]
    reverse_tan = zeros(Int64, 4)
    flip_normal = zeros(Int64, 4)
    axis_map = structured_connectivity_axis_map(connectivity_rows, reverse_tan)
    image_translation = FT[
        0 -L 0
        0  L 0
        0  0 0
        0  0 0
    ]

    face_bc = fill(BC_PERIODIC, 2, 6)
    face_bc[:, 3:4] .= BC_INTERBLOCK
    bc_params = zeros(FT, 2, 6, N_BC_PARAMS)

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
        file["axis_map"] = axis_map
        file["image_translation"] = image_translation
    end

    open(joinpath(out_dir, "mesh_manifest.toml"), "w") do io
        println(io, "mesh_kind = \"two_block_periodic_warped\"")
        println(io, "nx_per_block = $nx")
        println(io, "ny_per_block = $ny")
        println(io, "nz_per_block = $nz")
        println(io, "ng = $NG")
        println(io, "warp_a = 0.45")
        println(io, "warp_b = 0.45")
        println(io, "warp_c = 0.12")
        println(io, "period = $(L)")
    end
    println("Generated warped two-block mesh: $out_dir")
    return out_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
    ny = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
    nz = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
    out_dir = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "mesh", "warped_N32")
    generate_mesh(nx, ny, nz, out_dir)
end
