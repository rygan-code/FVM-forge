# =============================================================================
# Butterfly (O-H) Mesh Generator for Spatial Transition Pipe Flow
# Domain: Lx = 100R₀ (extremely long for natural transition)
# Boundary Conditions: Inflow (40) -> Outflow (9) (Not periodic!)
# =============================================================================

# Include the base mesh generator (functions only)
include("gen_butterfly_fvm.jl")

function main_spatial()
    # Parse command line args
    Nx_base = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
    Ny_val  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 108
    out_dir = length(ARGS) >= 3 ? ARGS[3] : "../MESH_SPATIAL"
    wall_bc_str = length(ARGS) >= 4 ? ARGS[4] : "isothermal"
    wall_bc_type = (wall_bc_str == "slip" || wall_bc_str == "slip_wall" || wall_bc_str == "12") ? 12 : 1
 
    # ─── Domain Length Settings ───
    Lx_override = 50.0        # Length in meters. R0=0.5m, so this is 100 R0.
    scale = Lx_override / Lx  # Lx in base generator is 7.5
    
    println("=== Spatial Transition Pipe Mesh ===")
    println("  Lx = $(Lx_override) ($(Lx_override/R0)R₀)")
    println("  Scale factor:  $(scale)")
    println("  Target directory: $(out_dir)")
    println("  Wall BC: $(wall_bc_str) (type ID: $(wall_bc_type))")

    # ─── Resolution Settings ───
    # Cross-section resolution
    Ny_b0, Nz_b0, N_rad = Ny_val, Ny_val, Ny_val

    # Scale Nx proportionally to maintain cell aspect ratio, round to multiple of 8
    Nx_b = 8 * cld(round(Int, Nx_base * scale), 8)
    println("  Nx: $(Nx_base) → $(Nx_b) (scaled for $(Lx_override)m domain)")

    mkpath(out_dir)
    max_dr_outer = 0.00035 * (1024.0 / Nx_base)

    blocks_data = []
    push!(blocks_data, generate_block0(Nx_b, Ny_b0, Nz_b0))
    push!(blocks_data, generate_block1(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer))
    push!(blocks_data, generate_block2(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer))
    push!(blocks_data, generate_block3(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer))
    push!(blocks_data, generate_block4(Nx_b, Ny_b0, Nz_b0, N_rad, max_dr_outer))

    # Rescale x coordinates: [0, Lx] → [0, Lx_override]
    for i in 1:5
        x, y, z, Nx, Ny, Nz = blocks_data[i]
        x .*= scale
        blocks_data[i] = (x, y, z, Nx, Ny, Nz)
    end

    # Build interpolation weights

    # Write mesh HDF5 files
    for bid in 0:4
        x, y, z, Nx, Ny, Nz = blocks_data[bid + 1]
        h5open(joinpath(out_dir, "mesh_b$bid.h5"), "w") do f
            f["NG"] = NG
            f["Nx"] = Int64(Nx)
            f["Ny"] = Int64(Ny)
            f["Nz"] = Int64(Nz)
            f["coords"] = Float32.(cat(reshape(x, (1, size(x)...)),
                                       reshape(y, (1, size(y)...)),
                                       reshape(z, (1, size(z)...)), dims=1))
            f["x"] = Float32.(x)
            f["y"] = Float32.(y)
            f["z"] = Float32.(z)
        end
    end

    # ─── Connectivity (Same cross-section topology) ───
    Nx_actual = blocks_data[1][4]

    connectivity_rows = [
        0 3 1 4; 1 4 0 3; 0 4 2 3; 2 3 0 4;
        0 5 3 6; 3 6 0 5; 0 6 4 5; 4 5 0 6;
        1 5 3 3; 3 3 1 5; 1 6 4 3; 4 3 1 6;
        2 5 3 4; 3 4 2 5; 2 6 4 4; 4 4 2 6
    ]

    function _get_tan(blocks_data, bid, fid)
        x, y, z, Nx, Ny, Nz = blocks_data[bid + 1]
        mi = size(x, 1) ÷ 2
        if fid == 3
            j0 = 1; mk = (Nz + 2) ÷ 2
            return [y[mi, j0, mk+1] - y[mi, j0, mk], z[mi, j0, mk+1] - z[mi, j0, mk]]
        elseif fid == 4
            j0 = Ny + 1; mk = (Nz + 2) ÷ 2
            return [y[mi, j0, mk+1] - y[mi, j0, mk], z[mi, j0, mk+1] - z[mi, j0, mk]]
        elseif fid == 5
            k0 = 1; mj = (Ny + 2) ÷ 2
            return [y[mi, mj+1, k0] - y[mi, mj, k0], z[mi, mj+1, k0] - z[mi, mj, k0]]
        elseif fid == 6
            k0 = Nz + 1; mj = (Ny + 2) ÷ 2
            return [y[mi, mj+1, k0] - y[mi, mj, k0], z[mi, mj+1, k0] - z[mi, mj, k0]]
        end
    end

    reverse_tan_arr = zeros(Int64, size(connectivity_rows, 1))
    flip_normal_arr = zeros(Int64, size(connectivity_rows, 1))
    for i in 1:size(connectivity_rows, 1)
        b1, f1, b2, f2 = connectivity_rows[i, :]
        t1 = _get_tan(blocks_data, b1, f1)
        t2 = _get_tan(blocks_data, b2, f2)
        reverse_tan_arr[i] = dot(t1, t2) < 0 ? 1 : 0
        if ((f1 in (3,4)) && (f2 in (5,6))) || ((f1 in (5,6)) && (f2 in (3,4)))
            flip_normal_arr[i] = 1
        end
    end

    # ─── Boundary Conditions Array ───
    face_bc = zeros(Int64, 5, 6)
    for bid in 0:4
        face_bc[bid+1, 1] = 40  # x_lo: BC_TRANSITION_INFLOW
        face_bc[bid+1, 2] = 9   # x_hi: BC_NSCBC_OUTFLOW
    end
    
    # Interblock faces
    for i in 1:size(connectivity_rows, 1)
        b1, f1 = connectivity_rows[i, 1], connectivity_rows[i, 2]
        face_bc[b1+1, f1] = 0   # BC_INTERBLOCK
    end
    
    # Outer walls
    face_bc[2, 3] = wall_bc_type
    face_bc[3, 4] = wall_bc_type
    face_bc[4, 5] = wall_bc_type
    face_bc[5, 6] = wall_bc_type

    # ─── Boundary Parameters Array ───
    # Dimension is 20 to match N_BC_PARAMS in bc_types.jl
    bc_params = zeros(FT, 5, 6, 20)
    
    # Wall Temperatures (Tw = 307.0 K)
    bc_params[2, 3, 1] = FT(307.0)
    bc_params[3, 4, 1] = FT(307.0)
    bc_params[4, 5, 1] = FT(307.0)
    bc_params[5, 6, 1] = FT(307.0)

    # NSCBC Outflow Parameters (Target Pressure and Relaxation)
    # Target pressure ~268.0 Pa for Re=17000, Ma=0.3
    for bid in 0:4
        bc_params[bid+1, 2, 2] = FT(268.0)  # BCP_P_TARGET
        bc_params[bid+1, 2, 8] = FT(0.25)   # BCP_SIGMA
        bc_params[bid+1, 2, 9] = FT(1.0)    # BCP_LREF
    end

    # Write to HDF5
    h5open(joinpath(out_dir, "block_connectivity.h5"), "w") do f
        f["Nblocks"] = 5
        f["Nx_b"] = fill(Int64(Nx_actual), 5)
        f["Ny_b"] = Int64[Ny_b0, N_rad, N_rad, Ny_b0, Ny_b0]
        f["Nz_b"] = Int64[Nz_b0, Nz_b0, Nz_b0, N_rad, N_rad]
        f["connectivity"] = connectivity_rows
        f["reverse_tan"] = reverse_tan_arr
        f["flip_normal"] = flip_normal_arr
        f["face_bc"] = face_bc
        f["bc_params"] = bc_params
    end

    if vis; export_mesh_vtk(blocks_data, out_dir); end
    println("Done writing to '", out_dir, "'!")
    println("  Nx_per_block = $(Nx_actual)")
    total_cells = Nx_actual * (Ny_b0^2 + 4*N_rad*Nz_b0)
    println("  Total cells  ≈ $(total_cells)")
    println("  * NOTE: Current resolution is coarse for debugging.")
    println("          Increase Nx_base, Ny_b0, Nz_b0, N_rad for production DNS.")
end

main_spatial()
