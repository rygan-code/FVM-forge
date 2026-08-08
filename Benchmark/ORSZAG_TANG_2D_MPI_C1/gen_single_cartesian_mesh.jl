# Generate a one-block periodic Cartesian mesh for same-resolution controls.

using HDF5

const FT = Float64
const NG = 4
const L = FT(2pi)
const BC_PERIODIC = Int32(2)
const N_BC_PARAMS = 20

function generate_mesh(nx::Integer, ny::Integer, nz::Integer, out_dir::AbstractString)
    nx > 0 && ny > 0 && nz > 0 || error("mesh dimensions must be positive")
    out_dir = abspath(out_dir)
    mkpath(out_dir)

    xis = range(zero(FT), L; length=nx + 1)
    etas = range(zero(FT), L; length=ny + 1)
    zetas = range(zero(FT), L; length=nz + 1)
    coords = zeros(FT, 3, nx + 1, ny + 1, nz + 1)
    for k in eachindex(zetas), j in eachindex(etas), i in eachindex(xis)
        coords[1, i, j, k] = xis[i]
        coords[2, i, j, k] = etas[j]
        coords[3, i, j, k] = zetas[k]
    end

    h5open(joinpath(out_dir, "mesh_b0.h5"), "w") do file
        file["NG"] = Int64(NG)
        file["Nx"] = Int64(nx)
        file["Ny"] = Int64(ny)
        file["Nz"] = Int64(nz)
        file["coords"] = coords
        file["x"] = coords[1, :, :, :]
        file["y"] = coords[2, :, :, :]
        file["z"] = coords[3, :, :, :]
    end

    h5open(joinpath(out_dir, "block_connectivity.h5"), "w") do file
        file["Nblocks"] = Int64(1)
        file["Nx_b"] = Int64[nx]
        file["Ny_b"] = Int64[ny]
        file["Nz_b"] = Int64[nz]
        file["face_bc"] = fill(BC_PERIODIC, 1, 6)
        file["bc_params"] = zeros(FT, 1, 6, N_BC_PARAMS)
        file["connectivity"] = zeros(Int32, 1, 6)
    end

    open(joinpath(out_dir, "mesh_manifest.toml"), "w") do io
        println(io, "mesh_kind = \"one_block_periodic_cartesian\"")
        println(io, "nx = $nx")
        println(io, "ny = $ny")
        println(io, "nz = $nz")
        println(io, "ng = $NG")
        println(io, "period = $L")
    end
    println("Generated one-block Cartesian mesh: $out_dir")
    return out_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
    ny = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nx
    nz = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
    out_dir = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "mesh", "single_cartesian_N32")
    generate_mesh(nx, ny, nz, out_dir)
end
