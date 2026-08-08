# unstruct_io.jl — VTK output for unstructured FVM
# Part of the second-order unstructured FVM branch
#
# Shared infrastructure (must be included BEFORE this file):
#   gpu_backend.jl, physics.jl, unstruct_mesh.jl
#
# Writes VTK XML UnstructuredGrid format (.vtu) for visualization
# in ParaView. Cell data includes ρ, u, v, w, p, T.

# ═════════════════════════════════════════════════════════════
# Write VTK UnstructuredGrid (.vtu)
# ═════════════════════════════════════════════════════════════

function write_vtu(block::UnstructBlock, filepath::String, tt::Int)
    # Pull data to host
    node_x = Array(block.node_x)
    node_y = Array(block.node_y)
    node_z = Array(block.node_z)
    Q_h = Array(block.Q)
    face_node_offset = Array(block.face_node_offset)
    face_node_list = Array(block.face_node_list)
    face_L = Array(block.face_L)

    nnode = block.nnode
    ncell = block.ncell

    open(filepath, "w") do io
        # VTK XML header
        println(io, """<?xml version="1.0"?>""")
        println(io, """<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">""")
        println(io, """  <UnstructuredGrid>""")

        # Piece
        println(io, """    <Piece NumberOfPoints="$nnode" NumberOfCells="$ncell">""")

        # Points
        println(io, """      <Points>""")
        println(io, """        <DataArray type="Float64" NumberOfComponents="3" format="ascii">""")
        for n in 1:nnode
            print(io, node_x[n], " ", node_y[n], " ", node_z[n], " ")
            if n % 5 == 0; println(io); end
        end
        println(io)
        println(io, """        </DataArray>""")
        println(io, """      </Points>""")

        # Cells: connectivity + offsets + types
        println(io, """      <Cells>""")

        # Connectivity: for each cell, list the face nodes (deduplicated)
        # We reconstruct cell-node connectivity from faces.
        # For a hex cell, we output 8 nodes; for tet, 4; etc.
        # Simplest: use the faces of each cell to extract unique nodes.
        cell_face_offset = Array(block.cell_face_offset)
        cell_face_list = Array(block.cell_face_list)
        cell_face_sign = Array(block.cell_face_sign)
        face_node_list_h = face_node_list

        # Build cell→node list
        cell_nodes = [Int[] for _ in 1:ncell]
        for c in 1:ncell
            f_start = cell_face_offset[c]
            f_end = cell_face_offset[c+1] - 1
            nodes_set = Set{Int}()
            for fi in f_start:f_end
                f = cell_face_list[fi]
                fn_start = face_node_offset[f]
                fn_end = face_node_offset[f+1] - 1
                for ni in fn_start:fn_end
                    push!(nodes_set, face_node_list_h[ni])
                end
            end
            cell_nodes[c] = sort(collect(nodes_set))
        end

        # Connectivity
        println(io, """        <DataArray type="Int32" Name="connectivity" format="ascii">""")
        for c in 1:ncell
            for n in cell_nodes[c]
                print(io, n - 1, " ")  # VTK uses 0-based
            end
            println(io)
        end
        println(io, """        </DataArray>""")

        # Offsets
        println(io, """        <DataArray type="Int32" Name="offsets" format="ascii">""")
        offset = 0
        for c in 1:ncell
            offset += length(cell_nodes[c])
            print(io, offset, " ")
            if c % 10 == 0; println(io); end
        end
        println(io)
        println(io, """        </DataArray>""")

        # Types (VTK_HEXAHEDRON=12, VTK_TETRA=10, VTK_WEDGE=13, etc.)
        println(io, """        <DataArray type="UInt8" Name="types" format="ascii">""")
        for c in 1:ncell
            nn = length(cell_nodes[c])
            vtype = if nn == 8; 12      # VTK_HEXAHEDRON
            elseif nn == 4; 10          # VTK_TETRA
            elseif nn == 6; 13          # VTK_WEDGE
            elseif nn == 5; 14          # VTK_PYRAMID
            else; 7                     # VTK_POLYGON (fallback)
            end
            print(io, vtype, " ")
        end
        println(io)
        println(io, """        </DataArray>""")

        println(io, """      </Cells>""")

        if block.partition !== nothing
            metadata = block.partition
            println(io, """      <PointData>""")
            println(io, """        <DataArray type="Int64" Name="GlobalNodeId" format="ascii">""")
            for node_id in metadata.node_global_ids
                print(io, node_id, " ")
            end
            println(io)
            println(io, """        </DataArray>""")
            println(io, """      </PointData>""")
        end

        # Cell Data
        println(io, """      <CellData>""")
        @static if equation_type == :MHD
            var_names = ["rho", "u", "v", "w", "p", "T", "Bx", "By", "Bz", "psi"]
        else
            var_names = ["rho", "u", "v", "w", "p", "T"]
        end
        for v in 1:Nprim
            println(io, """        <DataArray type="Float64" Name="$(var_names[v])" format="ascii">""")
            for c in 1:ncell
                print(io, Q_h[c, v], " ")
                if c % 10 == 0; println(io); end
            end
            println(io)
            println(io, """        </DataArray>""")
        end
        if block.partition !== nothing
            metadata = block.partition
            println(io, """        <DataArray type="Int64" Name="GlobalCellId" format="ascii">""")
            for cell_id in metadata.cell_global_ids[1:ncell]
                print(io, cell_id, " ")
            end
            println(io)
            println(io, """        </DataArray>""")
            println(io, """        <DataArray type="Int32" Name="PartitionId" format="ascii">""")
            for _ in 1:ncell
                print(io, metadata.rank, " ")
            end
            println(io)
            println(io, """        </DataArray>""")
        end
        println(io, """      </CellData>""")

        println(io, """    </Piece>""")
        println(io, """  </UnstructuredGrid>""")
        println(io, """</VTKFile>""")
    end

    @info "  VTK output written: $filepath (ncell=$ncell)"
    return
end
