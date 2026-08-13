# =============================================================================
# Multi-block I/O for butterfly grid (CUDA)
# Each block outputs to separate parallel HDF5 files
# XDMF uses GridType="Collection" to combine all blocks
# =============================================================================

const CT_CHECKPOINT_STATE_SCHEMA_VERSION = Int32(2)

# Keep checkpoint placement configurable for regression workers and batch
# jobs.  The historical default remains ./CHK.
@inline function structured_checkpoint_dir()
    return isdefined(Main, :checkpoint_dir) ?
        String(getfield(Main, :checkpoint_dir)) : "./CHK"
end

function ct_checkpoint_face_restore_mode(
    restored_face_blocks::Integer, total_face_blocks::Integer,
)
    total_face_blocks >= 0 || throw(ArgumentError(
        "total CT face-block count must be nonnegative",
    ))
    0 <= restored_face_blocks <= total_face_blocks || throw(ArgumentError(
        "restored CT face-block count must lie in 0:total",
    ))
    total_face_blocks == 0 && return :not_applicable
    restored_face_blocks == 0 && return :bootstrap
    restored_face_blocks == total_face_blocks && return :complete
    error(
        "CT restart is incomplete: restored $restored_face_blocks of " *
        "$total_face_blocks distributed face-field blocks. Refusing to mix " *
        "checkpoint face-B with Q-based initialization.",
    )
end

@inline function _structured_config_value(name::Symbol, default)
    return isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : default
end

function _ct_checkpoint_metadata_values()
    return (
        schema_version=CT_CHECKPOINT_STATE_SCHEMA_VERSION,
        isothermal=Int8(Bool(_structured_config_value(:isothermal_mhd, false))),
        gamma=Float64(_structured_config_value(Symbol("\u03b3"), NaN)),
        gas_constant=Float64(_structured_config_value(:Rg, NaN)),
        isothermal_temperature=Float64(
            _structured_config_value(:isothermal_temperature, NaN),
        ),
        background_split=Int8(Bool(_structured_config_value(
            :external_magnetic_background_splitting, false,
        ))),
    )
end

function _write_ct_checkpoint_metadata!(file)
    equation_type == :MHD && ct_mode || return nothing
    metadata = _ct_checkpoint_metadata_values()
    entries = (
        "ct_state_schema_version" => metadata.schema_version,
        "ct_isothermal_mhd" => metadata.isothermal,
        "ct_gamma" => metadata.gamma,
        "ct_gas_constant" => metadata.gas_constant,
        "ct_isothermal_temperature" => metadata.isothermal_temperature,
        "ct_background_split" => metadata.background_split,
    )
    for (name, value) in entries
        haskey(file, name) || (file[name] = value)
    end
    return nothing
end

@inline function _read_h5_scalar(dataset)
    value = read(dataset)
    return value isa AbstractArray ? only(value) : value
end

function validate_ct_checkpoint_metadata!(file, checkpoint_name::AbstractString)
    haskey(file, "ct_state_schema_version") || begin
        Bool(_structured_config_value(:isothermal_mhd, false)) && error(
            "CT checkpoint $checkpoint_name predates thermodynamic metadata; " *
            "an isothermal restart cannot determine whether its U[5] carrier " *
            "is compatible",
        )
        return :legacy_adiabatic
    end

    schema = Int(_read_h5_scalar(file["ct_state_schema_version"]))
    schema == CT_CHECKPOINT_STATE_SCHEMA_VERSION || error(
        "CT checkpoint $checkpoint_name uses state schema $schema; expected " *
        "$(CT_CHECKPOINT_STATE_SCHEMA_VERSION)",
    )
    checkpoint_isothermal =
        Int(_read_h5_scalar(file["ct_isothermal_mhd"])) != 0
    current_isothermal = Bool(_structured_config_value(:isothermal_mhd, false))
    checkpoint_isothermal == current_isothermal || error(
        "CT checkpoint $checkpoint_name closure mismatch: checkpoint " *
        "isothermal=$checkpoint_isothermal, current isothermal=$current_isothermal",
    )
    checkpoint_background = haskey(file, "ct_background_split") ?
        Int(_read_h5_scalar(file["ct_background_split"])) != 0 : false
    current_background = Bool(_structured_config_value(
        :external_magnetic_background_splitting, false,
    ))
    checkpoint_background == current_background || error(
        "CT checkpoint $checkpoint_name background split mismatch: " *
        "checkpoint=$checkpoint_background, current=$current_background",
    )

    checkpoint_gamma = Float64(_read_h5_scalar(file["ct_gamma"]))
    current_gamma = Float64(_structured_config_value(Symbol("\u03b3"), NaN))
    isapprox(checkpoint_gamma, current_gamma; rtol=8eps(Float64), atol=0.0) ||
        error(
            "CT checkpoint $checkpoint_name gamma mismatch: checkpoint " *
            "gamma=$checkpoint_gamma, current gamma=$current_gamma",
        )
    if current_isothermal
        checkpoint_rg = Float64(_read_h5_scalar(file["ct_gas_constant"]))
        checkpoint_temperature = Float64(
            _read_h5_scalar(file["ct_isothermal_temperature"]),
        )
        current_rg = Float64(_structured_config_value(:Rg, NaN))
        current_temperature = Float64(
            _structured_config_value(:isothermal_temperature, NaN),
        )
        isapprox(checkpoint_rg, current_rg; rtol=8eps(Float64), atol=0.0) ||
            error(
                "CT checkpoint $checkpoint_name gas constant mismatch: " *
                "checkpoint Rg=$checkpoint_rg, current Rg=$current_rg",
            )
        isapprox(
            checkpoint_temperature, current_temperature;
            rtol=8eps(Float64), atol=0.0,
        ) || error(
            "CT checkpoint $checkpoint_name isothermal temperature mismatch: " *
            "checkpoint T=$checkpoint_temperature, current T=$current_temperature",
        )
    end
    return :compatible
end

function _cleanup_old_checkpoints!(directory::AbstractString, keep_num::Integer)
    keep_num > 0 || return true
    try
        chk_files = readdir(directory)
        steps = Int64[]
        for filename in chk_files
            matched = match(r"^chk-(\d+)-b\d+\.h5$", filename)
            matched === nothing || push!(steps, parse(Int64, matched.captures[1]))
        end
        sort!(unique!(steps))
        if length(steps) > keep_num
            keep_steps = Set(steps[end-keep_num+1:end])
            for filename in chk_files
                matched = match(r"^chk-(\d+)-b\d+\.h5$", filename)
                matched === nothing && continue
                parse(Int64, matched.captures[1]) in keep_steps && continue
                rm(joinpath(directory, filename); force=true)
            end
        end
        return true
    catch exception
        @warn "checkpoint retention cleanup failed" directory=abspath(directory) exception=(exception, catch_backtrace())
        return false
    end
end

function plotFile_multiblock(
    tt, time, blocks, world_rank, Nblocks, Block_Nprocs, block_comms;
    force::Bool=false,
)
    if plt_out && (force || tt % step_plt == 0 || tt == maxStep)
        # Fix NFS/GPFS directory creation latency by running mkpath on all ranks
        mkpath("./PLT")
        if world_rank == 0
            nblocks_total = (isdefined(Main, :cebl_forcing) && Main.cebl_forcing) ? 2 * Nblocks : Nblocks
            write_XDMF_multiblock(tt, time, nblocks_total)
        end
        # Ensure directory is ready
        MPI.Barrier(MPI.COMM_WORLD)

        # ─── 1. Write Main Blocks (0 to Nblocks-1) ───
        for bid in sort(collect(keys(blocks)))
            b = blocks[bid]
            if b.id >= Nblocks
                continue
            end
            _write_plt_for_block(tt, b, block_comms[bid], Nblocks)
        end
        # Barrier: ensure all nodes finish writing main blocks before proceeding
        MPI.Barrier(MPI.COMM_WORLD)

        # ─── 2. Write Precursor Blocks (Nblocks to 2*Nblocks-1) ───
        if isdefined(Main, :cebl_forcing) && Main.cebl_forcing
            for bid in sort(collect(keys(blocks)))
                b = blocks[bid]
                if b.id < Nblocks
                    continue
                end
                _write_plt_for_block(tt, b, block_comms[bid], Nblocks)
            end
            # Barrier: ensure all nodes finish writing precursor blocks
            MPI.Barrier(MPI.COMM_WORLD)
        end
    end
end

function checkpointFile(tt, time, blocks, world_rank, Block_Nprocs, block_comms)
    if chk_out && (tt % step_chk == 0 || tt == maxStep)
        checkpoint_dir = structured_checkpoint_dir()
        # Fix NFS/GPFS directory creation latency by running mkpath on all ranks
        mkpath(checkpoint_dir)
        MPI.Barrier(MPI.COMM_WORLD)

        # ─── 1. Write Main Checkpoints (0 to 4) ───
        for bid in sort(collect(keys(blocks)))
            b = blocks[bid]
            if b.id >= 5
                continue
            end
            _write_chk_for_block(tt, b, block_comms[bid])
        end
        # Barrier: ensure all nodes finish writing main checkpoints
        MPI.Barrier(MPI.COMM_WORLD)

        # ─── 2. Write Precursor Checkpoints (5 to 9) ───
        if isdefined(Main, :cebl_forcing) && Main.cebl_forcing
            for bid in sort(collect(keys(blocks)))
                b = blocks[bid]
                if b.id < 5
                    continue
                end
                _write_chk_for_block(tt, b, block_comms[bid])
            end
            # Barrier: ensure all nodes finish writing precursor checkpoints
            MPI.Barrier(MPI.COMM_WORLD)
        end

        # Write metadata separately (rank 0 only, serial HDF5)
        if world_rank == 0
            for bid in 0:(length(Block_Nprocs)-1)
                if bid >= 5 && !(isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                    continue
                end
                chkname = joinpath(checkpoint_dir, "chk-$(tt)-b$(bid).h5")
                h5open(chkname, "r+") do f
                    f["step"] = Int64(tt)
                    f["time"] = Float64(time)
                    _write_ct_checkpoint_metadata!(f)
                end
            end
            println(">>> Checkpoint saved at step $tt")

            # Robust automatic cleanup of older checkpoints
            _keep_num = @isdefined(keep_chk_num) ? keep_chk_num : 0
            _cleanup_old_checkpoints!(checkpoint_dir, _keep_num)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

function averageFile(tt, blocks, world_rank, Block_Nprocs, block_comms)
    if !average; return; end
    if (tt % avg_total == 0 || tt == maxStep)
        # Fix NFS/GPFS directory creation latency by running mkpath on all ranks
        mkpath("./AVG")
        MPI.Barrier(MPI.COMM_WORLD)

        # ─── 1. Write Main Averages (0 to 4) ───
        for bid in sort(collect(keys(blocks)))
            b = blocks[bid]
            if b.id >= 5
                continue
            end
            _write_avg_for_block(tt, b, block_comms[bid])
        end
        # Barrier: ensure all nodes finish writing main averages
        MPI.Barrier(MPI.COMM_WORLD)

        # ─── 2. Write Precursor Averages (5 to 9) ───
        if isdefined(Main, :cebl_forcing) && Main.cebl_forcing
            for bid in sort(collect(keys(blocks)))
                b = blocks[bid]
                if b.id < 5
                    continue
                end
                _write_avg_for_block(tt, b, block_comms[bid])
            end
            # Barrier: ensure all nodes finish writing precursor averages
            MPI.Barrier(MPI.COMM_WORLD)
        end
    end
end

# ── XDMF metadata for multi-block (GridType="Collection") ──
function write_XDMF_multiblock(tt, time, Nblocks)
    fname = string("./PLT/plt-", tt, ".xmf")
    is_cebl = isdefined(Main, :cebl_forcing) && Main.cebl_forcing

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
            ref_bid = is_cebl && bid >= 5 ? bid - 5 : bid
            m_path = joinpath(_mesh_dir, "mesh_b$ref_bid.h5")
            
            # Skip if reference file is missing, empty, or corrupt
            if !isfile(m_path) || filesize(m_path) < 1000
                continue
            end
            
            has_nx = false
            try
                h5open(m_path, "r") do mf
                    has_nx = haskey(mf, "Nx")
                end
            catch exception
                @warn "skipping unreadable mesh while writing XDMF" path=abspath(m_path) exception=(exception, catch_backtrace())
            end
            if !has_nx
                continue
            end
            
            meshname = isabspath(_mesh_dir) ?
                string(_mesh_dir, "/mesh_b", ref_bid, ".h5") :
                string("../", _mesh_dir, "/mesh_b", ref_bid, ".h5")
            ny = h5read(m_path, "Ny")
            nz = h5read(m_path, "Nz")
            nx = is_cebl && bid >= 5 ? Main.cebl_Nx : h5read(m_path, "Nx")
            
            h5name = string("plt-", tt, "-b", bid, ".h5")

            write(f, "   <Grid Name=\"Block_$bid\" GridType=\"Uniform\">\n")
            write(f, "    <Topology TopologyType=\"3DSMesh\" NumberOfElements=\"$(nz+1) $(ny+1) $(nx+1)\" />\n")
            
            # Probe precision of 'x' coordinate
            coord_prec = 4
            h5open(m_path, "r") do mf
                if eltype(mf["x"]) == Float64
                    coord_prec = 8
                end
            end
            field_prec = sizeof(FT)
            if is_cebl && bid >= 5
                coord_prec = field_prec
            end

            # Geometry: X_Y_Z with separate x, y, z datasets
            nz_tot = nz + 1; ny_tot = ny + 1; nx_tot = nx + 1
            write(f, "    <Geometry GeometryType=\"X_Y_Z\">\n")
            for coord_name in ["x", "y", "z"]
                write(f, "     <DataItem Dimensions=\"$nz_tot $ny_tot $nx_tot\" NumberType=\"Float\" Precision=\"$coord_prec\" Format=\"HDF\">\n")
                if is_cebl && bid >= 5
                    write(f, "      $h5name:/$coord_name\n")
                else
                    write(f, "      $meshname:/$coord_name\n")
                end
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
                write(f, "     <DataItem Dimensions=\"$nz $ny $nx\" NumberType=\"Float\" Precision=\"$field_prec\" Format=\"HDF\">\n")
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

function saveRpoFile(path_prefix, time, blocks, world_rank, Block_Nprocs, block_comms)
    mkpath(dirname(path_prefix))
    MPI.Barrier(MPI.COMM_WORLD)
    for (bid, b) in blocks
        if b.id >= 5
            continue
        end
        filename = "$(path_prefix)-b$(b.id).h5"
        
        Q_h = Array(b.Q)
        Q_interior = Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, :]
        
        lox = b.ox + 1; hix = b.ox + b.Nx
        loy = b.oy + 1; hiy = b.oy + b.Ny
        loz = b.oz + 1; hiz = b.oz + b.Nz
        
        local Nx_b, Ny_b, Nz_b
        if isdefined(Main, :Nx_b) && length(Main.Nx_b) >= b.id + 1
            Nx_b = Main.Nx_b[b.id + 1]
            Ny_b = Main.Ny_b[b.id + 1]
            Nz_b = Main.Nz_b[b.id + 1]
        else
            Nx_b = b.Nx
            Ny_b = b.Ny
            Nz_b = b.Nz
        end
        
        single_rank = MPI.Comm_size(block_comms[bid]) == 1
        _h5f = single_rank ? h5open(filename, "w") : h5open(filename, "w", block_comms[bid])
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
    
    if world_rank == 0
        for bid in 0:(length(Block_Nprocs)-1)
            if bid >= 5
                continue
            end
            filename = "$(path_prefix)-b$(bid).h5"
            h5open(filename, "r+") do f
                f["time"] = Float64(time)
            end
        end
        println(">>> RPO output saved to prefix: $path_prefix")
    end
    MPI.Barrier(MPI.COMM_WORLD)
end

function write_precursor_mesh(bid, x_pre_h, y_pre_h, z_pre_h, nxp_pre, nyp, nzp, NG, ox_pre, oy, oz, block_comm)
    _mesh_base = isdefined(Main, :mesh_dir) ? Main.mesh_dir : "MESH"
    mpath = joinpath(_mesh_base, "mesh_b$(bid).h5")
    
    Nx_global = Main.cebl_Nx
    Ny_global = Main.Ny_b[bid - 4]
    Nz_global = Main.Nz_b[bid - 4]
    
    x_local = x_pre_h[NG+1:NG+nxp_pre+1, NG+1:NG+nyp+1, NG+1:NG+nzp+1]
    y_local = y_pre_h[NG+1:NG+nxp_pre+1, NG+1:NG+nyp+1, NG+1:NG+nzp+1]
    z_local = z_pre_h[NG+1:NG+nxp_pre+1, NG+1:NG+nyp+1, NG+1:NG+nzp+1]
    
    lox = ox_pre + 1; hix = ox_pre + nxp_pre + 1
    loy = oy + 1; hiy = oy + nyp + 1
    loz = oz + 1; hiz = oz + nzp + 1
    
    single_rank = MPI.Comm_size(block_comm) == 1
    _h5f = single_rank ? h5open(mpath, "w") : h5open(mpath, "w", block_comm)
    try
        if single_rank
            _h5f["NG"] = Int32(NG)
            _h5f["Nx"] = Int32(Nx_global)
            _h5f["Ny"] = Int32(Ny_global)
            _h5f["Nz"] = Int32(Nz_global)
            _h5f["x"] = Array(x_local)
            _h5f["y"] = Array(y_local)
            _h5f["z"] = Array(z_local)
        else
            if MPI.Comm_rank(block_comm) == 0
                _h5f["NG"] = Int32(NG)
                _h5f["Nx"] = Int32(Nx_global)
                _h5f["Ny"] = Int32(Ny_global)
                _h5f["Nz"] = Int32(Nz_global)
            end
            MPI.Barrier(block_comm)
            
            for (name, data) in [("x", x_local), ("y", y_local), ("z", z_local)]
                dset = create_dataset(
                    _h5f, name, datatype(FT),
                    dataspace(Nx_global + 1, Ny_global + 1, Nz_global + 1);
                    chunk=(Nx_global + 1, Ny_global + 1, Nz_global + 1),
                    dxpl_mpio=:collective
                )
                dset[lox:hix, loy:hiy, loz:hiz] = data
            end
        end
    finally
        close(_h5f)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions to write separate blocks sequentially with global barriers
# ─────────────────────────────────────────────────────────────────────────────

function _write_plt_for_block(tt, b, comm, Nblocks)
    fname = string("./PLT/plt-", tt, "-b", b.id, ".h5")
    
    Q_h = Array(b.Q)
    ϕ_h = Array(b.ϕ)

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

    lox = b.ox + 1; hix = b.ox + b.Nx
    loy = b.oy + 1; hiy = b.oy + b.Ny
    loz = b.oz + 1; hiz = b.oz + b.Nz

    local Nx_b, Ny_b, Nz_b
    if isdefined(Main, :Nx_b) && length(Main.Nx_b) >= b.id + 1
        Nx_b = Main.Nx_b[b.id + 1]
        Ny_b = Main.Ny_b[b.id + 1]
        Nz_b = Main.Nz_b[b.id + 1]
    elseif b.id >= 5 && isdefined(Main, :cebl_Nx) && isdefined(Main, :Ny_b)
        Nx_b = Main.cebl_Nx
        Ny_b = Main.Ny_b[b.id - 4]
        Nz_b = Main.Nz_b[b.id - 4]
    else
        Nx_b = b.Nx; Ny_b = b.Ny; Nz_b = b.Nz
    end

    if b.id >= Nblocks
        # Precursor blocks: Gather and serial write
        gathered_vars = Any[]
        for (name, data) in var_list
            g_data = _gather_block_data(data, b, comm, Nx_b, Ny_b, Nz_b)
            if MPI.Comm_rank(comm) == 0
                push!(gathered_vars, (name, g_data))
            end
        end
        # Also gather coordinates
        gx = _gather_node_data(b.x, b, comm, Nx_b, Ny_b, Nz_b)
        gy = _gather_node_data(b.y, b, comm, Nx_b, Ny_b, Nz_b)
        gz = _gather_node_data(b.z, b, comm, Nx_b, Ny_b, Nz_b)
        if MPI.Comm_rank(comm) == 0
            push!(gathered_vars, ("x", gx), ("y", gy), ("z", gz))
        end
        if MPI.Comm_rank(comm) == 0
            _h5f = h5open(fname, "w")
            try
                for (name, g_data) in gathered_vars
                    _h5f[name] = g_data
                end
            finally
                close(_h5f)
            end
        end
    else
        # Main blocks: Parallel collective HDF5
        single_rank = MPI.Comm_size(comm) == 1
        _h5f = single_rank ? h5open(fname, "w") : h5open(fname, "w", comm)
        try
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
    end
    Q_h = nothing; ϕ_h = nothing
    GC.gc()
end


function _write_chk_for_block(tt, b, comm)
    chkname = joinpath(
        structured_checkpoint_dir(), "chk-$(tt)-b$(b.id).h5",
    )
    
    Q_h = Array(b.Q)
    Q_interior = Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, :]
    write_ct_state = equation_type == :MHD && ct_mode &&
                     b.Bx_face !== nothing
    U_interior = write_ct_state ? Array(@view b.U[
        1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, :,
    ]) : nothing
    write_implicit_history = isdefined(Main, :dual_time) &&
        Bool(dual_time) && hasproperty(b, :Un) && b.Un !== nothing &&
        hasproperty(b, :U_nm1) && b.U_nm1 !== nothing
    Un_interior = write_implicit_history ? Array(@view b.Un[
        1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, :,
    ]) : nothing
    U_nm1_interior = write_implicit_history ? Array(@view b.U_nm1[
        1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, :,
    ]) : nothing

    lox = b.ox + 1; hix = b.ox + b.Nx
    loy = b.oy + 1; hiy = b.oy + b.Ny
    loz = b.oz + 1; hiz = b.oz + b.Nz

    local Nx_b, Ny_b, Nz_b
    if isdefined(Main, :Nx_b) && length(Main.Nx_b) >= b.id + 1
        Nx_b = Main.Nx_b[b.id + 1]
        Ny_b = Main.Ny_b[b.id + 1]
        Nz_b = Main.Nz_b[b.id + 1]
    elseif b.id >= 5 && isdefined(Main, :cebl_Nx) && isdefined(Main, :Ny_b)
        Nx_b = Main.cebl_Nx
        Ny_b = Main.Ny_b[b.id - 4]
        Nz_b = Main.Nz_b[b.id - 4]
    else
        Nx_b = b.Nx; Ny_b = b.Ny; Nz_b = b.Nz
    end

    if b.id >= 5
        # Precursor checkpoint: Gather and serial write
        g_Q = _gather_block_data(Q_interior, b, comm, Nx_b, Ny_b, Nz_b)
        if MPI.Comm_rank(comm) == 0
            _h5f = h5open(chkname, "w")
            try
                _h5f["Q"] = g_Q
                if write_ct_state
                    error("CT checkpoint output is not supported for gathered precursor blocks")
                end
            finally
                close(_h5f)
            end
        end
    else
        # Main checkpoint: Parallel collective HDF5
        single_rank = MPI.Comm_size(comm) == 1
        _h5f = single_rank ? h5open(chkname, "w") : h5open(chkname, "w", comm)
        try
            if single_rank
                _h5f["Q"] = Array(Q_interior)
                if write_implicit_history
                    _h5f["Un"] = Un_interior
                    _h5f["U_nm1"] = U_nm1_interior
                end
                if write_ct_state
                    _h5f["U"] = U_interior
                    _h5f["Bx_face"] = Array(@view b.Bx_face[
                        NG+1:NG+b.Nx+1, NG+1:NG+b.Ny, NG+1:NG+b.Nz,
                    ])
                    _h5f["By_face"] = Array(@view b.By_face[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny+1, NG+1:NG+b.Nz,
                    ])
                    _h5f["Bz_face"] = Array(@view b.Bz_face[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny, NG+1:NG+b.Nz+1,
                    ])
                    if hasproperty(b, :B0x_face) && b.B0x_face !== nothing
                        _h5f["B0x_face"] = Array(@view b.B0x_face[
                            NG+1:NG+b.Nx+1, NG+1:NG+b.Ny, NG+1:NG+b.Nz,
                        ])
                        _h5f["B0y_face"] = Array(@view b.B0y_face[
                            NG+1:NG+b.Nx, NG+1:NG+b.Ny+1, NG+1:NG+b.Nz,
                        ])
                        _h5f["B0z_face"] = Array(@view b.B0z_face[
                            NG+1:NG+b.Nx, NG+1:NG+b.Ny, NG+1:NG+b.Nz+1,
                        ])
                    end
                    _write_ct_checkpoint_metadata!(_h5f)
                end
            else
                dset = create_dataset(
                    _h5f, "Q", datatype(FT),
                    dataspace(Nx_b, Ny_b, Nz_b, Nprim);
                    chunk=(Nx_b, Ny_b, Nz_b, Nprim),
                    dxpl_mpio=:collective
                )
                dset[lox:hix, loy:hiy, loz:hiz, :] = Q_interior
                if write_implicit_history
                    for (name, data) in (("Un", Un_interior),
                                         ("U_nm1", U_nm1_interior))
                        history_dset = create_dataset(
                            _h5f, name, datatype(FT),
                            dataspace(Nx_b, Ny_b, Nz_b, size(data, 4));
                            chunk=(Nx_b, Ny_b, Nz_b, size(data, 4)),
                            dxpl_mpio=:collective,
                        )
                        history_dset[lox:hix, loy:hiy, loz:hiz, :] = data
                    end
                end
                if write_ct_state
                    u_dset = create_dataset(
                        _h5f, "U", datatype(FT),
                        dataspace(Nx_b, Ny_b, Nz_b, Ncell_cons);
                        chunk=(Nx_b, Ny_b, Nz_b, Ncell_cons),
                        dxpl_mpio=:collective,
                    )
                    u_dset[lox:hix, loy:hiy, loz:hiz, :] = U_interior

                    face_specs = (
                        (
                            "Bx_face", b.Bx_face,
                            (Nx_b+1, Ny_b, Nz_b),
                            (b.ox == 0 ? NG+1 : NG+2, NG+1, NG+1),
                            (NG+b.Nx+1, NG+b.Ny, NG+b.Nz),
                            (b.ox == 0 ? b.ox+1 : b.ox+2, b.oy+1, b.oz+1),
                            (b.ox+b.Nx+1, b.oy+b.Ny, b.oz+b.Nz),
                        ),
                        (
                            "By_face", b.By_face,
                            (Nx_b, Ny_b+1, Nz_b),
                            (NG+1, b.oy == 0 ? NG+1 : NG+2, NG+1),
                            (NG+b.Nx, NG+b.Ny+1, NG+b.Nz),
                            (b.ox+1, b.oy == 0 ? b.oy+1 : b.oy+2, b.oz+1),
                            (b.ox+b.Nx, b.oy+b.Ny+1, b.oz+b.Nz),
                        ),
                        (
                            "Bz_face", b.Bz_face,
                            (Nx_b, Ny_b, Nz_b+1),
                            (NG+1, NG+1, b.oz == 0 ? NG+1 : NG+2),
                            (NG+b.Nx, NG+b.Ny, NG+b.Nz+1),
                            (b.ox+1, b.oy+1, b.oz == 0 ? b.oz+1 : b.oz+2),
                            (b.ox+b.Nx, b.oy+b.Ny, b.oz+b.Nz+1),
                        ),
                    )
                    if hasproperty(b, :B0x_face) && b.B0x_face !== nothing
                        face_specs = (
                            face_specs...,
                            (
                                "B0x_face", b.B0x_face,
                                (Nx_b+1, Ny_b, Nz_b),
                                (b.ox == 0 ? NG+1 : NG+2, NG+1, NG+1),
                                (NG+b.Nx+1, NG+b.Ny, NG+b.Nz),
                                (b.ox == 0 ? b.ox+1 : b.ox+2, b.oy+1, b.oz+1),
                                (b.ox+b.Nx+1, b.oy+b.Ny, b.oz+b.Nz),
                            ),
                            (
                                "B0y_face", b.B0y_face,
                                (Nx_b, Ny_b+1, Nz_b),
                                (NG+1, b.oy == 0 ? NG+1 : NG+2, NG+1),
                                (NG+b.Nx, NG+b.Ny+1, NG+b.Nz),
                                (b.ox+1, b.oy == 0 ? b.oy+1 : b.oy+2, b.oz+1),
                                (b.ox+b.Nx, b.oy+b.Ny+1, b.oz+b.Nz),
                            ),
                            (
                                "B0z_face", b.B0z_face,
                                (Nx_b, Ny_b, Nz_b+1),
                                (NG+1, NG+1, b.oz == 0 ? NG+1 : NG+2),
                                (NG+b.Nx, NG+b.Ny, NG+b.Nz+1),
                                (b.ox+1, b.oy+1, b.oz == 0 ? b.oz+1 : b.oz+2),
                                (b.ox+b.Nx, b.oy+b.Ny, b.oz+b.Nz+1),
                            ),
                        )
                    end
                    for (name, field, global_dims, local_lo, local_hi,
                         global_lo, global_hi) in face_specs
                        face_dset = create_dataset(
                            _h5f, name, datatype(FT), dataspace(global_dims...);
                            chunk=global_dims, dxpl_mpio=:collective,
                        )
                        face_dset[
                            global_lo[1]:global_hi[1],
                            global_lo[2]:global_hi[2],
                            global_lo[3]:global_hi[3],
                        ] = Array(@view field[
                            local_lo[1]:local_hi[1],
                            local_lo[2]:local_hi[2],
                            local_lo[3]:local_hi[3],
                        ])
                    end
                end
            end
        finally
            close(_h5f)
        end
    end
    Q_h = nothing; Q_interior = nothing; U_interior = nothing
    GC.gc()
end

function _write_avg_for_block(tt, b, comm)
    fname = string("./AVG/avg-", tt, "-b", b.id, ".h5")
    if hasfield(typeof(b), :Q_avg) && b.Q_avg !== nothing
        Q_h = Array(b.Q_avg)
        if isdefined(Main, :avg_density_weighted) && avg_density_weighted
            for n in 2:4
                @views Q_h[:,:,:,n] ./= Q_h[:,:,:,1]
            end
        end
        avg = @view Q_h[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, :]
        
        lox = b.ox + 1; hix = b.ox + b.Nx
        loy = b.oy + 1; hiy = b.oy + b.Ny
        loz = b.oz + 1; hiz = b.oz + b.Nz

        local Nx_b, Ny_b, Nz_b
        if isdefined(Main, :Nx_b) && length(Main.Nx_b) >= b.id + 1
            Nx_b = Main.Nx_b[b.id + 1]
            Ny_b = Main.Ny_b[b.id + 1]
            Nz_b = Main.Nz_b[b.id + 1]
        elseif b.id >= 5 && isdefined(Main, :cebl_Nx) && isdefined(Main, :Ny_b)
            Nx_b = Main.cebl_Nx
            Ny_b = Main.Ny_b[b.id - 4]
            Nz_b = Main.Nz_b[b.id - 4]
        else
            Nx_b = b.Nx; Ny_b = b.Ny; Nz_b = b.Nz
        end

        if b.id >= 5
            # Precursor average: Gather and serial write
            g_avg = _gather_block_data(avg, b, comm, Nx_b, Ny_b, Nz_b)
            if MPI.Comm_rank(comm) == 0
                _h5f = h5open(fname, "w")
                try
                    _h5f["avg"] = g_avg
                finally
                    close(_h5f)
                end
            end
        else
            # Main average: Parallel collective HDF5
            single_rank = MPI.Comm_size(comm) == 1
            _h5f = single_rank ? h5open(fname, "w") : h5open(fname, "w", comm)
            try
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
end

function _gather_block_data(data_local::AbstractArray{T, N}, b, comm, Nx_b, Ny_b, Nz_b) where {T, N}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if rank == 0
        global_shape = N == 3 ? (Nx_b, Ny_b, Nz_b) : (Nx_b, Ny_b, Nz_b, size(data_local, 4))
        global_data = Array{T, N}(undef, global_shape...)
        
        lox = b.ox + 1; hix = b.ox + b.Nx
        loy = b.oy + 1; hiy = b.oy + b.Ny
        loz = b.oz + 1; hiz = b.oz + b.Nz
        if N == 3
            global_data[lox:hix, loy:hiy, loz:hiz] .= data_local
        else
            global_data[lox:hix, loy:hiy, loz:hiz, :] .= data_local
        end
        
        for r in 1:(nprocs-1)
            info = Vector{Int32}(undef, 6)
            MPI.Recv!(info, comm; source=r, tag=500 + r)
            ox_r, oy_r, oz_r, Nx_r, Ny_r, Nz_r = info[1], info[2], info[3], info[4], info[5], info[6]
            
            slab_shape = N == 3 ? (Nx_r, Ny_r, Nz_r) : (Nx_r, Ny_r, Nz_r, size(data_local, 4))
            slab = Array{T, N}(undef, slab_shape...)
            MPI.Recv!(slab, comm; source=r, tag=600 + r)
            
            lox_r = ox_r + 1; hix_r = ox_r + Nx_r
            loy_r = oy_r + 1; hiy_r = oy_r + Ny_r
            loz_r = oz_r + 1; hiz_r = oz_r + Nz_r
            if N == 3
                global_data[lox_r:hix_r, loy_r:hiy_r, loz_r:hiz_r] .= slab
            else
                global_data[lox_r:hix_r, loy_r:hiy_r, loz_r:hiz_r, :] .= slab
            end
        end
        return global_data
    else
        info = Int32[b.ox, b.oy, b.oz, b.Nx, b.Ny, b.Nz]
        MPI.Send(info, comm; dest=0, tag=500 + rank)
        MPI.Send(Array(data_local), comm; dest=0, tag=600 + rank)
        return nothing
    end
end

function _gather_node_data(data_local::AbstractArray{T, 3}, b, comm, Nx_b, Ny_b, Nz_b) where {T}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Extract interior nodes: 1+NG : b.Nx+NG+1, etc.
    local_nodes = @view data_local[1+NG:b.Nx+NG+1, 1+NG:b.Ny+NG+1, 1+NG:b.Nz+NG+1]
    
    if rank == 0
        global_data = Array{T, 3}(undef, Nx_b + 1, Ny_b + 1, Nz_b + 1)
        
        lox = b.ox + 1; hix = b.ox + b.Nx + 1
        loy = b.oy + 1; hiy = b.oy + b.Ny + 1
        loz = b.oz + 1; hiz = b.oz + b.Nz + 1
        
        global_data[lox:hix, loy:hiy, loz:hiz] .= Array(local_nodes)
        
        for r in 1:(nprocs-1)
            info = Vector{Int32}(undef, 6)
            MPI.Recv!(info, comm; source=r, tag=700 + r)
            ox_r, oy_r, oz_r, Nx_r, Ny_r, Nz_r = info[1], info[2], info[3], info[4], info[5], info[6]
            
            slab = Array{T, 3}(undef, Nx_r + 1, Ny_r + 1, Nz_r + 1)
            MPI.Recv!(slab, comm; source=r, tag=800 + r)
            
            lox_r = ox_r + 1; hix_r = ox_r + Nx_r + 1
            loy_r = oy_r + 1; hiy_r = oy_r + Ny_r + 1
            loz_r = oz_r + 1; hiz_r = oz_r + Nz_r + 1
            
            global_data[lox_r:hix_r, loy_r:hiy_r, loz_r:hiz_r] .= slab
        end
        return global_data
    else
        info = Int32[b.ox, b.oy, b.oz, b.Nx, b.Ny, b.Nz]
        MPI.Send(info, comm; dest=0, tag=700 + rank)
        MPI.Send(Array(local_nodes), comm; dest=0, tag=800 + rank)
        return nothing
    end
end
