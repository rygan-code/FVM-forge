# =============================================================================
# Multi-block I/O for butterfly grid (CUDA)
# Each block outputs to separate parallel HDF5 files
# XDMF uses GridType="Collection" to combine all blocks
# =============================================================================

function plotFile_multiblock(tt, time, blocks, world_rank, Nblocks, Block_Nprocs, block_comms)
    if plt_out && (tt % step_plt == 0 || tt == maxStep)
        if world_rank == 0
            mkpath("./PLT")
            write_XDMF_multiblock(tt, time, Nblocks)
        end
        # Ensure directory is ready
        MPI.Barrier(MPI.COMM_WORLD)

        for (bid, b) in blocks
            # Multi-block: Each block writes its own parallel HDF5 file
            fname = string("./PLT/plt-", tt, "-b", b.id, ".h5")
            
            # Host buffers for writing (HDF5.jl parallel writing requires Array)
            Q_h = Array(b.Q)
            ϕ_h = Array(b.ϕ)

            # Extract real cells (no ghost)
            ρ    = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 1]
            u    = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 2]
            v    = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 3]
            w    = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 4]
            p    = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 5]
            T    = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 6]
            ϕ_ng = @view ϕ_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG]
            var_list = Any[("rho", ρ), ("u", u), ("v", v), ("w", w), ("p", p), ("T", T), ("phi", ϕ_ng)]
            if equation_type == :MHD
                Bx = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 7]
                By = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 8]
                Bz = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 9]
                push!(var_list, ("Bx", Bx), ("By", By), ("Bz", Bz))
                if Nprim >= 10
                    ψv = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 10]
                    push!(var_list, ("psi", ψv))
                end
            end

            # Global indices within this block for this rank (no ghost)
            lox = b.ox + 1; hix = b.ox + b.Nx
            loy = b.oy + 1; hiy = b.oy + b.Ny
            loz = b.oz + 1; hiz = b.oz + b.Nz

            # HDF5 write (serial fallback for single-rank, parallel for multi-rank)
            single_rank = MPI.Comm_size(block_comms[bid]) == 1
            _h5f = single_rank ? h5open(fname, "w") : h5open(fname, "w", block_comms[bid])
            try
                # Block dimensions
                _md = @isdefined(mesh_dir) ? mesh_dir : "MESH"
                m_path = "$(mesh[1:end-length(basename(mesh))])mesh_b$(b.id).h5"
                if !isfile(m_path); m_path = joinpath(_md, "mesh_b$(b.id).h5"); end
                Nx_b = h5read(m_path, "Nx")
                Ny_b = h5read(m_path, "Ny")
                Nz_b = h5read(m_path, "Nz")

                for (name, data) in var_list
                    if single_rank
                        _h5f[name] = Array(data)
                    else
                        dset = create_dataset(
                            _h5f, name, datatype(FT),
                            dataspace(Nx_b, Ny_b, Nz_b);
                            chunk=(Nx_b, Ny_b, Nz_b),
                            dxpl_mpio=:collective
                        )
                        dset[lox:hix, loy:hiy, loz:hiz] = data
                    end
                end
            finally
                close(_h5f)
            end
            # Free host buffers immediately to prevent OOM
            Q_h = nothing; ϕ_h = nothing
            GC.gc()
        end
        # Final block sync
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

function checkpointFile(tt, time, blocks, world_rank, Block_Nprocs, block_comms)
    if chk_out && (tt % step_chk == 0 || tt == maxStep)
        if world_rank == 0
            mkpath("./CHK")
        end
        MPI.Barrier(MPI.COMM_WORLD)

        for (bid, b) in blocks
            chkname = "./CHK/chk-$(tt)-b$(b.id).h5"
            
            # GPU → CPU
            Q_h = Array(b.Q)

            # Extract interior cells (no ghost) for this rank's sub-domain
            Q_interior = Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, :]

            # Global offsets within the block (same as plotFile)
            lox = b.ox + 1; hix = b.ox + b.Nx
            loy = b.oy + 1; hiy = b.oy + b.Ny
            loz = b.oz + 1; hiz = b.oz + b.Nz

            # Read global block dimensions from mesh
            _md = @isdefined(mesh_dir) ? mesh_dir : "MESH"
            m_path = joinpath(_md, "mesh_b$(b.id).h5")
            Nx_b = h5read(m_path, "Nx")
            Ny_b = h5read(m_path, "Ny")
            Nz_b = h5read(m_path, "Nz")

            # HDF5 write (serial fallback for single-rank, parallel for multi-rank)
            single_rank = MPI.Comm_size(block_comms[bid]) == 1
            _h5f = single_rank ? h5open(chkname, "w") : h5open(chkname, "w", block_comms[bid])
            try
                if single_rank
                    _h5f["Q"] = Array(Q_interior)
                else
                    dset = create_dataset(
                        _h5f, "Q", datatype(FT),
                        dataspace(Nx_b, Ny_b, Nz_b, Nprim);
                        chunk=(Nx_b, Ny_b, Nz_b, Nprim),
                        dxpl_mpio=:collective
                    )
                    dset[lox:hix, loy:hiy, loz:hiz, :] = Q_interior
                end
            finally
                close(_h5f)
            end

            Q_h = nothing; Q_interior = nothing
        end
        GC.gc()
        MPI.Barrier(MPI.COMM_WORLD)

        # Write metadata separately (rank 0 only, serial HDF5)
        if world_rank == 0
            for bid in 0:(length(Block_Nprocs)-1)
                chkname = "./CHK/chk-$(tt)-b$(bid).h5"
                h5open(chkname, "r+") do f
                    f["step"] = Int64(tt)
                    f["time"] = Float64(time)
                end
            end
            println(">>> Checkpoint saved at step $tt")

            # Robust automatic cleanup of older checkpoints
            _keep_num = @isdefined(keep_chk_num) ? keep_chk_num : 0
            if _keep_num > 0
                try
                    chk_files = readdir("./CHK")
                    steps = Int64[]
                    for f in chk_files
                        m = match(r"^chk-(\d+)-b\d+\.h5$", f)
                        if m !== nothing
                            push!(steps, parse(Int64, m.captures[1]))
                        end
                    end
                    unique!(steps)
                    sort!(steps)
                    if length(steps) > _keep_num
                        keep_steps = steps[end-_keep_num+1:end]
                        for f in chk_files
                            m = match(r"^chk-(\d+)-b\d+\.h5$", f)
                            if m !== nothing
                                step_val = parse(Int64, m.captures[1])
                                if !(step_val in keep_steps)
                                    rm(joinpath("./CHK", f); force=true)
                                end
                            end
                        end
                    end
                catch e
                    # Prevent filesystem exception from crashing the solver
                end
            end
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

function averageFile(tt, blocks, world_rank, Block_Nprocs, block_comms)
    if !average; return; end
    if (tt % avg_total == 0 || tt == maxStep)
        if world_rank == 0
            mkpath("./AVG")
        end
        MPI.Barrier(MPI.COMM_WORLD)

        for (bid, b) in blocks
            fname = string("./AVG/avg-", tt, "-b", b.id, ".h5")
            if hasfield(typeof(b), :Q_avg) && b.Q_avg !== nothing
                Q_h = Array(b.Q_avg)
                # Apply inverse density weighting right before writing output if Favre averaged
                if isdefined(Main, :avg_density_weighted) && avg_density_weighted
                    for n in 2:4
                        @views Q_h[:,:,:,n] ./= Q_h[:,:,:,1]
                    end
                end
                avg = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, :]
                
                lox = b.ox + 1; hix = b.ox + b.Nx
                loy = b.oy + 1; hiy = b.oy + b.Ny
                loz = b.oz + 1; hiz = b.oz + b.Nz

                # Parallel HDF5 write (serial fallback for single-rank, parallel for multi-rank)
                single_rank = MPI.Comm_size(block_comms[bid]) == 1
                _h5f = single_rank ? h5open(fname, "w") : h5open(fname, "w", block_comms[bid])
                try
                    _md = @isdefined(mesh_dir) ? mesh_dir : "MESH"
                    m_path = joinpath(_md, "mesh_b$(b.id).h5")
                    Nx_b = h5read(m_path, "Nx")
                    Ny_b = h5read(m_path, "Ny")
                    Nz_b = h5read(m_path, "Nz")

                    if single_rank
                        _h5f["avg"] = Array(avg)
                    else
                        dset = create_dataset(
                            _h5f, "avg", datatype(FT),
                            dataspace(Nx_b, Ny_b, Nz_b, Nprim);
                            chunk=(Nx_b, Ny_b, Nz_b, Nprim),
                            dxpl_mpio=:collective
                        )
                        dset[lox:hix, loy:hiy, loz:hiz, :] = avg
                    end
                finally
                    close(_h5f)
                end
            end
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

# ── XDMF metadata for multi-block (GridType="Collection") ──
function write_XDMF_multiblock(tt, time, Nblocks)
    fname = string("./PLT/plt-", tt, ".xmf")

    open(fname, "w") do f
        write(f, "<?xml version=\"1.0\" ?>\n")
        write(f, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
        write(f, "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">\n")
        write(f, " <Domain>\n")
        write(f, "  <Grid Name=\"MultiBlock\" GridType=\"Collection\" CollectionType=\"Spatial\">\n")
        write(f, "  <Time Value=\"$time\" />\n")

        for bid = 0:Nblocks-1
            # Use mesh_dir from config; fallback to "MESH" for legacy run scripts
            _mesh_dir = @isdefined(mesh_dir) ? mesh_dir : "MESH"
            meshname = string("../", _mesh_dir, "/mesh_b", bid, ".h5")
            m_path = joinpath(_mesh_dir, "mesh_b$bid.h5")
            nx = h5read(m_path, "Nx")
            ny = h5read(m_path, "Ny")
            nz = h5read(m_path, "Nz")
            # For coordinate selection from mesh with ghost cells
            # Total size in HDF5 (C-style): (nz_tot, ny_tot, nx_tot, 3)
            # NI_tot = nx + 1 + 2*NG, etc.
            # Physical nodes are start at NG (0-indexed)
            
            h5name = string("plt-", tt, "-b", bid, ".h5")

            write(f, "   <Grid Name=\"Block_$bid\" GridType=\"Uniform\">\n")
            write(f, "    <Topology TopologyType=\"3DSMesh\" NumberOfElements=\"$(nz+1) $(ny+1) $(nx+1)\" />\n")
            
            # Geometry: X_Y_Z with separate x, y, z datasets (most compatible format)
            # Each dataset shape in HDF5 (C-style): (nz+1, ny+1, nx+1) — matches topology
            nz_tot = nz + 1; ny_tot = ny + 1; nx_tot = nx + 1
            write(f, "    <Geometry GeometryType=\"X_Y_Z\">\n")
            for coord_name in ["x", "y", "z"]
                write(f, "     <DataItem Dimensions=\"$nz_tot $ny_tot $nx_tot\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
                write(f, "      $meshname:/$coord_name\n")
                write(f, "     </DataItem>\n")
            end
            write(f, "    </Geometry>\n")

            varnames = ["rho", "u", "v", "w", "p", "T", "phi"]
            if equation_type == :MHD
                push!(varnames, "Bx", "By", "Bz")
                if Nprim >= 10
                    push!(varnames, "psi")
                end
            end
            for varname in varnames
                write(f, "    <Attribute Name=\"$varname\" AttributeType=\"Scalar\" Center=\"Cell\">\n")
                write(f, "     <DataItem Dimensions=\"$nz $ny $nx\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n")
                write(f, "      $h5name:/$varname\n")
                write(f, "     </DataItem>\n")
                write(f, "    </Attribute>\n")
            end

            write(f, "   </Grid>\n")
        end

        write(f, "  </Grid>\n")
        write(f, " </Domain>\n")
        write(f, "</Xdmf>\n")
    end
end
