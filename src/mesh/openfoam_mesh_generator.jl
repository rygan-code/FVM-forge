# gen_openfoam_mesh.jl — Generate a minimal OpenFOAM mesh for testing
# Creates a simple 1D channel mesh in OpenFOAM polyMesh format
#
# Usage: julia gen_openfoam_mesh.jl [output_dir] [ncells]
# Output: output_dir/constant/polyMesh/{points,faces,owner,neighbour,boundary}

const FT = Float64

function generate_openfoam_mesh(output_dir::String, ncells::Int)
    polyMesh_dir = joinpath(output_dir, "constant", "polyMesh")
    mkpath(polyMesh_dir)

    # 1D mesh: ncells cells, ncells+1 nodes in x, 1 cell in y/z
    # Domain [0, 1] × [0, 0.1] × [0, 0.1]
    dx = 1.0 / ncells
    dy = 0.1
    dz = 0.1

    n_nodes = (ncells + 1) * 2 * 2  # (nx+1) * (ny+1) * (nz+1)
    n_internal_faces = ncells - 1   # internal x-faces
    n_boundary_faces_x = 2          # x-low, x-high (1 each, since ny=nz=1)
    n_boundary_faces_y = 2 * ncells  # y-low, y-high (per cell)
    n_boundary_faces_z = 2 * ncells  # z-low, z-high
    n_faces = n_internal_faces + n_boundary_faces_x + n_boundary_faces_y + n_boundary_faces_z

    # Node indexing: node(i, j, k) = i + (j-1)*(ncells+1) + (k-1)*(ncells+1)*2  (0-based in OpenFOAM)
    # i=1..ncells+1, j=1..2, k=1..2
    function nid(i, j, k)  # 0-based
        return (i-1) + (j-1)*(ncells+1) + (k-1)*(ncells+1)*2
    end

    # ── Write points ──
    open(joinpath(polyMesh_dir, "points"), "w") do io
        println(io, "/*--------------------------------*- C++ -*----------------------------------*\\")
        println(io, "| =========                 |                                                 |")
        println(io, "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |")
        println(io, "|  \\\\    /   O peration     | Version:  v2012                                 |")
        println(io, "|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |")
        println(io, "|    \\\\/     M anipulation  |                                                 |")
        println(io, "\\*---------------------------------------------------------------------------*/")
        println(io, "FoamFile")
        println(io, "{")
        println(io, "    version     2.0;")
        println(io, "    format      ascii;")
        println(io, "    class       vectorField;")
        println(io, "    location    \"constant/polyMesh\";")
        println(io, "    object      points;")
        println(io, "}")
        println(io, "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
        println(io)
        println(io, n_nodes)
        println(io, "(")
        for k in 1:2, j in 1:2, i in 1:(ncells+1)
            x = (i-1) * dx
            y = (j-1) * dy
            z = (k-1) * dz
            println(io, "($x $y $z)")
        end
        println(io, ")")
        println(io)
        println(io, "// ************************************************************************* //")
    end

    # ── Write faces ──
    # Face node ordering for hexahedra (OpenFOAM convention):
    # Each face lists nodes counterclockwise when viewed from outside
    open(joinpath(polyMesh_dir, "faces"), "w") do io
        println(io, "FoamFile")
        println(io, "{")
        println(io, "    version     2.0;")
        println(io, "    format      ascii;")
        println(io, "    class       faceList;")
        println(io, "    location    \"constant/polyMesh\";")
        println(io, "    object      faces;")
        println(io, "}")
        println(io)
        println(io, n_faces)
        println(io, "(")
        # Internal x-faces (between cell i and i+1), face at x=i*dx
        for i in 1:(ncells-1)
            # Face at x = i*dx uses node column i+1 (0-based: i)
            println(io, "4($(nid(i+1,1,1)) $(nid(i+1,2,1)) $(nid(i+1,2,2)) $(nid(i+1,1,2)))")
        end
        # x-low boundary face (cell 1, x=0)
        println(io, "4($(nid(1,1,1)) $(nid(1,1,2)) $(nid(1,2,2)) $(nid(1,2,1)))")
        # x-high boundary face (cell ncells, x=1)
        println(io, "4($(nid(ncells+1,1,1)) $(nid(ncells+1,2,1)) $(nid(ncells+1,2,2)) $(nid(ncells+1,1,2)))")
        # y-low boundary faces (per cell)
        for i in 1:ncells
            println(io, "4($(nid(i,1,1)) $(nid(i+1,1,1)) $(nid(i+1,1,2)) $(nid(i,1,2)))")
        end
        # y-high boundary faces
        for i in 1:ncells
            println(io, "4($(nid(i,2,1)) $(nid(i,2,2)) $(nid(i+1,2,2)) $(nid(i+1,2,1)))")
        end
        # z-low boundary faces
        for i in 1:ncells
            println(io, "4($(nid(i,1,1)) $(nid(i,2,1)) $(nid(i+1,2,1)) $(nid(i+1,1,1)))")
        end
        # z-high boundary faces
        for i in 1:ncells
            println(io, "4($(nid(i,1,2)) $(nid(i+1,1,2)) $(nid(i+1,2,2)) $(nid(i,2,2)))")
        end
        println(io, ")")
        println(io)
        println(io, "// ************************************************************************* //")
    end

    # ── Write owner ──
    open(joinpath(polyMesh_dir, "owner"), "w") do io
        println(io, "FoamFile")
        println(io, "{")
        println(io, "    version     2.0;")
        println(io, "    format      ascii;")
        println(io, "    class       labelList;")
        println(io, "    location    \"constant/polyMesh\";")
        println(io, "    object      owner;")
        println(io, "}")
        println(io)
        println(io, n_faces)
        println(io, "(")
        # Internal faces: owner = left cell (0-based)
        for i in 1:(ncells-1)
            println(io, "$(i-1)")  # cell i (0-based)
        end
        # x-low boundary: owner = cell 0
        println(io, "0")
        # x-high boundary: owner = cell ncells-1
        println(io, "$(ncells-1)")
        # y/z boundary faces: owner = corresponding cell (0-based)
        for i in 1:ncells; println(io, "$(i-1)"); end  # y-low
        for i in 1:ncells; println(io, "$(i-1)"); end  # y-high
        for i in 1:ncells; println(io, "$(i-1)"); end  # z-low
        for i in 1:ncells; println(io, "$(i-1)"); end  # z-high
        println(io, ")")
        println(io)
        println(io, "// ************************************************************************* //")
    end

    # ── Write neighbour ── (only internal faces)
    open(joinpath(polyMesh_dir, "neighbour"), "w") do io
        println(io, "FoamFile")
        println(io, "{")
        println(io, "    version     2.0;")
        println(io, "    format      ascii;")
        println(io, "    class       labelList;")
        println(io, "    location    \"constant/polyMesh\";")
        println(io, "    object      neighbour;")
        println(io, "}")
        println(io)
        println(io, n_internal_faces)
        println(io, "(")
        for i in 1:(ncells-1)
            println(io, "$(i)")  # cell i+1 (0-based)
        end
        println(io, ")")
        println(io)
        println(io, "// ************************************************************************* //")
    end

    # ── Write boundary ──
    open(joinpath(polyMesh_dir, "boundary"), "w") do io
        println(io, "FoamFile")
        println(io, "{")
        println(io, "    version     2.0;")
        println(io, "    format      ascii;")
        println(io, "    class       polyBoundaryMesh;")
        println(io, "    location    \"constant/polyMesh\";")
        println(io, "    object      boundary;")
        println(io, "}")
        println(io)
        println(io, "6")
        println(io, "(")
        # Internal faces come first (not part of boundary patches)
        # Boundary patches reference faces by startFace index
        # startFace counts from 0
        sf = n_internal_faces  # first boundary face index

        println(io, "    xlow")
        println(io, "    {")
        println(io, "        type            zeroGradient;")
        println(io, "        nFaces          1;")
        println(io, "        startFace       $sf;")
        println(io, "    }")
        sf += 1

        println(io, "    xhigh")
        println(io, "    {")
        println(io, "        type            zeroGradient;")
        println(io, "        nFaces          1;")
        println(io, "        startFace       $sf;")
        println(io, "    }")
        sf += 1

        println(io, "    ylow")
        println(io, "    {")
        println(io, "        type            zeroGradient;")
        println(io, "        nFaces          $ncells;")
        println(io, "        startFace       $sf;")
        println(io, "    }")
        sf += ncells

        println(io, "    yhigh")
        println(io, "    {")
        println(io, "        type            zeroGradient;")
        println(io, "        nFaces          $ncells;")
        println(io, "        startFace       $sf;")
        println(io, "    }")
        sf += ncells

        println(io, "    zlow")
        println(io, "    {")
        println(io, "        type            zeroGradient;")
        println(io, "        nFaces          $ncells;")
        println(io, "        startFace       $sf;")
        println(io, "    }")
        sf += ncells

        println(io, "    zhigh")
        println(io, "    {")
        println(io, "        type            zeroGradient;")
        println(io, "        nFaces          $ncells;")
        println(io, "        startFace       $sf;")
        println(io, "    }")

        println(io, ")")
        println(io)
        println(io, "// ************************************************************************* //")
    end

    println("Generated OpenFOAM mesh: $output_dir")
    println("  ncells=$ncells, nfaces=$n_faces, nnodes=$n_nodes")
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_dir = length(ARGS) >= 1 ? ARGS[1] : "test_openfoam_case"
    cell_count = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
    cell_count > 0 || throw(ArgumentError("ncells must be positive"))
    generate_openfoam_mesh(output_dir, cell_count)
end
