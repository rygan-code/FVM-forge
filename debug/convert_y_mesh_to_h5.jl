# =============================================================================
# convert_y_mesh_to_h5.jl — Convert y_mesh VTS files to HDF5 format
#
# Reads the 3-block y_mesh from VTS files and creates:
#   - mesh_b0.h5, mesh_b1.h5, mesh_b2.h5 (node coordinates)
#   - block_connectivity.h5 (connectivity and BC info)
#
# The y_mesh is a simple 3-block Y-junction:
#   Block 0: center block
#   Block 1: right branch
#   Block 2: left branch
# =============================================================================

using HDF5
using Printf

const FT = Float64
const NG = 4  # Ghost cell layers

# ─── Read VTS files ───
function read_vts_coords(filename)
    # Simple VTS parser - extract coordinates
    lines = readlines(filename)

    # Find extent
    extent_line = filter(l -> occursin("WholeExtent", l), lines)[1]
    m = match(r"WholeExtent=\"(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)\"", extent_line)
    nx_nodes = parse(Int, m.captures[2]) + 1
    ny_nodes = parse(Int, m.captures[4]) + 1
    nz_nodes = parse(Int, m.captures[6]) + 1

    # Find coordinate data
    data_start = findfirst(l -> occursin("format=\"ascii\"", l), lines) + 1

    x = zeros(Float64, nx_nodes, ny_nodes, nz_nodes)
    y = zeros(Float64, nx_nodes, ny_nodes, nz_nodes)
    z = zeros(Float64, nx_nodes, ny_nodes, nz_nodes)

    idx = data_start
    for k in 1:nz_nodes
        for j in 1:ny_nodes
            for i in 1:nx_nodes
                vals = parse.(Float64, split(strip(lines[idx])))
                x[i, j, k] = vals[1]
                y[i, j, k] = vals[2]
                z[i, j, k] = vals[3]
                idx += 1
            end
        end
    end

    return x, y, z, nx_nodes-1, ny_nodes-1, nz_nodes-1
end

# ─── Main conversion ───
function convert_y_mesh()
    input_dir = "debug/output"
    output_dir = "debug/output"

    println("Converting y_mesh VTS to HDF5...")
    println("  Input:  $input_dir/y_mesh_b{0,1,2}.vts")
    println("  Output: $output_dir/")

    # Read all blocks
    blocks = []
    for bid in 0:2
        vts_file = joinpath(input_dir, "y_mesh_b$(bid).vts")
        println("  Reading $vts_file...")
        x, y, z, nx, ny, nz = read_vts_coords(vts_file)
        push!(blocks, (x=x, y=y, z=z, nx=nx, ny=ny, nz=nz))
        println("    Block $bid: $(nx)×$(ny)×$(nz) cells")
    end

    # Write mesh files
    for (bid, blk) in enumerate(blocks)
        mesh_file = joinpath(output_dir, "mesh_b$(bid-1).h5")
        println("  Writing $mesh_file...")
        h5open(mesh_file, "w") do f
            write(f, "x", blk.x)
            write(f, "y", blk.y)
            write(f, "z", blk.z)
            write(f, "NG", NG)
            write(f, "Nx", blk.nx)
            write(f, "Ny", blk.ny)
            write(f, "Nz", blk.nz)
        end
    end

    # Create connectivity file
    # For the y_mesh, we need to define:
    # - Which faces are connected to which
    # - BC types for each face
    #
    # Assuming Y-junction topology:
    #   Block 0 face 2 (ξ+) connects to block 1 face 1 (ξ-)
    #   Block 0 face 4 (η+) connects to block 2 face 3 (η-)
    #   Other faces are walls

    Nblocks = 3
    Nx_b = [blk.nx for blk in blocks]
    Ny_b = [blk.ny for blk in blocks]
    Nz_b = [blk.nz for blk in blocks]

    # BC types: 0=interblock, 1=isothermal_wall, 2=periodic, etc.
    # Initialize all faces as isothermal wall
    face_bc = ones(Int32, Nblocks, 6)  # 6 faces per block

    # Define inter-block connections
    # Block 0 face 2 (ξ+) <-> Block 1 face 1 (ξ-)
    face_bc[1, 2] = 0  # BC_INTERBLOCK
    face_bc[2, 1] = 0  # BC_INTERBLOCK

    # Block 0 face 4 (η+) <-> Block 2 face 3 (η-)
    face_bc[1, 4] = 0  # BC_INTERBLOCK
    face_bc[3, 3] = 0  # BC_INTERBLOCK

    # BC parameters (wall temperature, etc.)
    bc_params = zeros(Float32, Nblocks, 6, 20)  # 20 parameter slots
    for bid in 1:Nblocks
        for fid in 1:6
            if face_bc[bid, fid] == 1  # isothermal_wall
                bc_params[bid, fid, 1] = 300.0  # Tw = 300K
            end
        end
    end

    # Connectivity array format: [dst_block, dst_face, src_block, src_face]
    # Each row is one connection
    connectivity = [
        1 2 2 1;  # Block 0 face 2 <-> Block 1 face 1
        2 1 1 2;  # Reverse connection
        1 4 3 3;  # Block 0 face 4 <-> Block 2 face 3
        3 3 1 4;  # Reverse connection
    ]

    # reverse_tan: 0 = no reversal, 1 = reverse tangential direction
    reverse_tan = zeros(Int64, size(connectivity, 1))

    # flip_normal: 0 = no flip, 1 = flip normal direction
    flip_normal = zeros(Int64, size(connectivity, 1))

    # Write connectivity file
    conn_file = joinpath(output_dir, "block_connectivity.h5")
    println("  Writing $conn_file...")
    h5open(conn_file, "w") do f
        write(f, "Nblocks", Nblocks)
        write(f, "Nx_b", Nx_b)
        write(f, "Ny_b", Ny_b)
        write(f, "Nz_b", Nz_b)

        # Write connectivity
        write(f, "connectivity", connectivity)

        # Write reverse_tan and flip_normal
        write(f, "reverse_tan", reverse_tan)
        write(f, "flip_normal", flip_normal)

        # Write face_bc
        write(f, "face_bc", face_bc)

        # Write bc_params
        write(f, "bc_params", bc_params)
    end

    println("\nConversion complete!")
    println("  Files created:")
    for bid in 0:2
        println("    mesh_b$(bid).h5")
    end
    println("    block_connectivity.h5")
end

# Run conversion
convert_y_mesh()
