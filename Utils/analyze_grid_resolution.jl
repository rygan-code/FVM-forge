# =============================================================================
# analyze_grid_resolution.jl — Wall-Unit Grid Resolution for Pipe Flow
#
# Reads REAL flow field data from PLT/ to compute u_τ and ν_w,
# then measures the actual first-cell Δx⁺, Δr⁺, (rΔθ)⁺ from the mesh.
#
# For the butterfly mesh:
#   Block 0: center square (no wall)
#   Block 1: wall at η- (j=1)
#   Block 2: wall at η+ (j=Ny)
#   Block 3: wall at ζ- (k=1)
#   Block 4: wall at ζ+ (k=Nz)
#
# Usage:
#   julia Utils/analyze_grid_resolution.jl [PLT_DIR] [STEP]
#   julia Utils/analyze_grid_resolution.jl               # auto-detect latest
#   julia Utils/analyze_grid_resolution.jl PLT 400000
#
# Outputs:
#   Console: Δx⁺, Δr⁺, (rΔθ)⁺ statistics per block and global
#   PLT/grid_resolution.csv
# =============================================================================

using HDF5, Statistics, Printf

# ─── Physical constants (must match run/baseline/pipe_baseline.jl) ───
const γ   = 1.4
const Rg  = 287.0
const C_s = 1.458e-6
const T_s = 110.4

sutherland_mu(T) = C_s * T^1.5 / (T + T_s)

"""
    load_block_data(plt_dir, bid, step)

Load flow field and mesh node coordinates for a single block.
Returns (rho, u, v, w, T, x_n, y_n, z_n, Nx, Ny, Nz)
"""
function load_block_data(plt_dir, bid, step)
    plt_file = joinpath(plt_dir, "plt-$(step)-b$(bid).h5")
    if !isfile(plt_file)
        error("File not found: $plt_file")
    end

    fid = h5open(plt_file, "r")
    rho = Float64.(read(fid["rho"]))
    u   = Float64.(read(fid["u"]))
    v   = Float64.(read(fid["v"]))
    w   = Float64.(read(fid["w"]))
    T   = Float64.(read(fid["T"]))
    close(fid)

    # Load mesh node coordinates
    mesh_file = "MESH/mesh_b$(bid).h5"
    mfid = h5open(mesh_file, "r")
    Nx  = Int(read(mfid["Nx"]))
    Ny  = Int(read(mfid["Ny"]))
    Nz  = Int(read(mfid["Nz"]))
    x_n = Float64.(read(mfid["x"]))  # node coords: (Nx+1, Ny+1, Nz+1)
    y_n = Float64.(read(mfid["y"]))
    z_n = Float64.(read(mfid["z"]))
    close(mfid)

    return rho, u, v, w, T, x_n, y_n, z_n, Nx, Ny, Nz
end

"""
    compute_utau_from_flow(plt_dir, step; R0=0.5)

Compute friction velocity u_τ and wall viscosity ν_w from the actual flow field.
Uses all wall-adjacent cells from blocks 1-4 and the velocity gradient at the wall.
"""
function compute_utau_from_flow(plt_dir, step; R0=0.5, N_bins=200)
    # Collect all cells for radial profile
    all_r   = Float64[]
    all_uz  = Float64[]
    all_rho = Float64[]
    all_T   = Float64[]

    for bid in 0:4
        rho, u, v, w, T, x_n, y_n, z_n, Nx, Ny, Nz = load_block_data(plt_dir, bid, step)

        # Cell centers from node averages
        y_c = 0.125 .* (y_n[1:end-1,1:end-1,1:end-1] .+ y_n[2:end,1:end-1,1:end-1] .+
                         y_n[1:end-1,2:end,1:end-1]   .+ y_n[2:end,2:end,1:end-1]   .+
                         y_n[1:end-1,1:end-1,2:end]   .+ y_n[2:end,1:end-1,2:end]   .+
                         y_n[1:end-1,2:end,2:end]     .+ y_n[2:end,2:end,2:end])
        z_c = 0.125 .* (z_n[1:end-1,1:end-1,1:end-1] .+ z_n[2:end,1:end-1,1:end-1] .+
                         z_n[1:end-1,2:end,1:end-1]   .+ z_n[2:end,2:end,1:end-1]   .+
                         z_n[1:end-1,1:end-1,2:end]   .+ z_n[2:end,1:end-1,2:end]   .+
                         z_n[1:end-1,2:end,2:end]     .+ z_n[2:end,2:end,2:end])

        r = sqrt.(y_c.^2 .+ z_c.^2)
        append!(all_r,   vec(r))
        append!(all_uz,  vec(u))
        append!(all_rho, vec(rho))
        append!(all_T,   vec(T))
    end

    # Radial binning
    r_edges = range(0.0, R0, length=N_bins+1)
    r_mid   = 0.5 .* (r_edges[1:end-1] .+ r_edges[2:end])
    y_wall  = R0 .- r_mid

    mean_uz  = zeros(N_bins)
    mean_rho = zeros(N_bins)
    mean_T   = zeros(N_bins)
    count    = zeros(Int, N_bins)

    for idx in eachindex(all_r)
        ri = all_r[idx]
        if ri >= R0; continue; end
        bin = clamp(Int(floor(ri / R0 * N_bins)) + 1, 1, N_bins)
        mean_uz[bin]  += all_uz[idx]
        mean_rho[bin] += all_rho[idx]
        mean_T[bin]   += all_T[idx]
        count[bin]    += 1
    end

    valid = count .> 0
    mean_uz[valid]  ./= count[valid]
    mean_rho[valid] ./= count[valid]
    mean_T[valid]   ./= count[valid]

    # Wall quantities from near-wall bins
    wall_bins = findall(valid .& (y_wall .< 0.02 * R0))
    if isempty(wall_bins); wall_bins = findall(valid); end
    sorted_idx = sort(wall_bins, by=i->y_wall[i])

    rho_w = mean_rho[sorted_idx[end]]
    T_w   = mean_T[sorted_idx[end]]
    mu_w  = sutherland_mu(T_w)
    nu_w  = mu_w / rho_w

    # Wall shear stress from near-wall gradient
    near_wall = findall(valid .& (y_wall .> 0) .& (y_wall .< 0.05 * R0))
    if length(near_wall) < 2
        near_wall = findall(valid .& (y_wall .> 0))
    end
    near_wall = sort(near_wall, by=i->y_wall[i])

    yw_fit = y_wall[near_wall]
    uz_fit = mean_uz[near_wall]
    dudy_wall = sum(yw_fit .* uz_fit) / sum(yw_fit.^2)
    tau_w = mu_w * abs(dudy_wall)
    u_tau = sqrt(tau_w / rho_w)

    Re_tau = R0 * u_tau / nu_w
    return u_tau, nu_w, rho_w, T_w, mu_w, tau_w, Re_tau
end

"""
    node_edge_length(x_n, y_n, z_n, i, j, k, dir)

Compute the physical edge length of cell (i,j,k) in direction `dir` ∈ {:xi, :eta, :zeta}.
Cell (i,j,k) has nodes at corners (i:i+1, j:j+1, k:k+1).
Average of 4 parallel edges.
"""
function node_edge_length(x_n, y_n, z_n, i, j, k, dir::Symbol)
    L = 0.0
    if dir == :xi
        for dj in 0:1, dk in 0:1
            L += sqrt((x_n[i+1,j+dj,k+dk] - x_n[i,j+dj,k+dk])^2 +
                       (y_n[i+1,j+dj,k+dk] - y_n[i,j+dj,k+dk])^2 +
                       (z_n[i+1,j+dj,k+dk] - z_n[i,j+dj,k+dk])^2)
        end
    elseif dir == :eta
        for di in 0:1, dk in 0:1
            L += sqrt((x_n[i+di,j+1,k+dk] - x_n[i+di,j,k+dk])^2 +
                       (y_n[i+di,j+1,k+dk] - y_n[i+di,j,k+dk])^2 +
                       (z_n[i+di,j+1,k+dk] - z_n[i+di,j,k+dk])^2)
        end
    elseif dir == :zeta
        for di in 0:1, dj in 0:1
            L += sqrt((x_n[i+di,j+dj,k+1] - x_n[i+di,j+dj,k])^2 +
                       (y_n[i+di,j+dj,k+1] - y_n[i+di,j+dj,k])^2 +
                       (z_n[i+di,j+dj,k+1] - z_n[i+di,j+dj,k])^2)
        end
    end
    return L * 0.25
end

"""
    analyze_wall_block(bid, x_n, y_n, z_n, Nx, Ny, Nz, wall_face, u_tau, nu_w; R0=0.5)

For a wall block, compute Δx⁺, Δr⁺ (wall-normal), (rΔθ)⁺ (circumferential)
at the first cell layer adjacent to the wall.

wall_face:
  :eta_minus  → Block 1: wall at j=1, wall-normal = η, tangential = ζ
  :eta_plus   → Block 2: wall at j=Ny, wall-normal = η, tangential = ζ
  :zeta_minus → Block 3: wall at k=1, wall-normal = ζ, tangential = η
  :zeta_plus  → Block 4: wall at k=Nz, wall-normal = ζ, tangential = η

Returns dict of statistics.
"""
function analyze_wall_block(bid, x_n, y_n, z_n, Nx, Ny, Nz, wall_face, u_tau, nu_w; R0=0.5)
    l_star = nu_w / u_tau  # viscous length scale

    dx_arr  = Float64[]  # streamwise (ξ)
    dr_arr  = Float64[]  # wall-normal
    dth_arr = Float64[]  # circumferential

    if wall_face == :eta_minus
        # Wall at j=1, first cell is j=1
        j_wall = 1
        for k in 1:Nz, i in 1:Nx
            push!(dx_arr,  node_edge_length(x_n, y_n, z_n, i, j_wall, k, :xi))
            push!(dr_arr,  node_edge_length(x_n, y_n, z_n, i, j_wall, k, :eta))
            push!(dth_arr, node_edge_length(x_n, y_n, z_n, i, j_wall, k, :zeta))
        end
    elseif wall_face == :eta_plus
        # Wall at j=Ny+1 node, first cell is j=Ny
        j_wall = Ny
        for k in 1:Nz, i in 1:Nx
            push!(dx_arr,  node_edge_length(x_n, y_n, z_n, i, j_wall, k, :xi))
            push!(dr_arr,  node_edge_length(x_n, y_n, z_n, i, j_wall, k, :eta))
            push!(dth_arr, node_edge_length(x_n, y_n, z_n, i, j_wall, k, :zeta))
        end
    elseif wall_face == :zeta_minus
        # Wall at k=1, first cell is k=1
        k_wall = 1
        for j in 1:Ny, i in 1:Nx
            push!(dx_arr,  node_edge_length(x_n, y_n, z_n, i, j, k_wall, :xi))
            push!(dr_arr,  node_edge_length(x_n, y_n, z_n, i, j, k_wall, :zeta))
            push!(dth_arr, node_edge_length(x_n, y_n, z_n, i, j, k_wall, :eta))
        end
    elseif wall_face == :zeta_plus
        # Wall at k=Nz+1 node, first cell is k=Nz
        k_wall = Nz
        for j in 1:Ny, i in 1:Nx
            push!(dx_arr,  node_edge_length(x_n, y_n, z_n, i, j, k_wall, :xi))
            push!(dr_arr,  node_edge_length(x_n, y_n, z_n, i, j, k_wall, :zeta))
            push!(dth_arr, node_edge_length(x_n, y_n, z_n, i, j, k_wall, :eta))
        end
    end

    # Convert to wall units
    dx_plus  = dx_arr  ./ l_star
    dr_plus  = dr_arr  ./ l_star
    dth_plus = dth_arr ./ l_star

    return Dict(
        "bid"       => bid,
        "face"      => wall_face,
        "N_cells"   => length(dx_arr),
        # Physical (dimensional)
        "dx_min"    => minimum(dx_arr),  "dx_max"  => maximum(dx_arr),  "dx_mean"  => mean(dx_arr),
        "dr_min"    => minimum(dr_arr),  "dr_max"  => maximum(dr_arr),  "dr_mean"  => mean(dr_arr),
        "dth_min"   => minimum(dth_arr), "dth_max" => maximum(dth_arr), "dth_mean" => mean(dth_arr),
        # Wall units
        "dx+_min"   => minimum(dx_plus),  "dx+_max"  => maximum(dx_plus),  "dx+_mean"  => mean(dx_plus),
        "dr+_min"   => minimum(dr_plus),  "dr+_max"  => maximum(dr_plus),  "dr+_mean"  => mean(dr_plus),
        "dth+_min"  => minimum(dth_plus), "dth+_max" => maximum(dth_plus), "dth+_mean" => mean(dth_plus),
    )
end

# =============================================================================
# Main
# =============================================================================
function main(; plt_dir="PLT", step=nothing, R0=0.5)
    # Auto-detect latest step
    if step === nothing
        files = readdir(plt_dir)
        plt_files = filter(f -> occursin(r"^plt-\d+-b0\.h5$", f), files)
        if isempty(plt_files)
            error("No plt-*-b0.h5 files found in $plt_dir/")
        end
        step_nums = [parse(Int, match(r"plt-(\d+)-b0", f).captures[1]) for f in plt_files]
        step = maximum(step_nums)
    end

    println("═" ^ 78)
    println("  Pipe Flow — First-Cell Grid Resolution in Wall Units")
    println("═" ^ 78)
    @printf("  PLT directory : %s\n", plt_dir)
    @printf("  Step          : %d\n", step)
    @printf("  Pipe radius   : %.3f\n\n", R0)

    # ── Step 1: Compute u_τ from real flow field ──
    u_tau, nu_w, rho_w, T_w, mu_w, tau_w, Re_tau = compute_utau_from_flow(plt_dir, step; R0=R0)
    l_star = nu_w / u_tau

    println("  ┌─────────────────────────────────────────────────┐")
    println("  │          Wall Friction Parameters               │")
    println("  ├─────────────────────────────────────────────────┤")
    @printf("  │  u_τ      = %12.4f  m/s                    │\n", u_tau)
    @printf("  │  τ_w      = %12.4f  Pa                     │\n", tau_w)
    @printf("  │  ν_w      = %12.4e  m²/s                   │\n", nu_w)
    @printf("  │  ℓ* = ν/u_τ = %10.4e  m  (viscous length)  │\n", l_star)
    @printf("  │  Re_τ     = %12.1f                         │\n", Re_tau)
    @printf("  │  ρ_w      = %12.6f  kg/m³                  │\n", rho_w)
    @printf("  │  T_w      = %12.2f  K                      │\n", T_w)
    @printf("  │  μ_w      = %12.4e  Pa·s                   │\n", mu_w)
    println("  └─────────────────────────────────────────────────┘")
    println()

    # ── Step 2: Analyze each wall block ──
    wall_blocks = [
        (1, :eta_minus),
        (2, :eta_plus),
        (3, :zeta_minus),
        (4, :zeta_plus),
    ]

    results = Dict[]
    for (bid, face) in wall_blocks
        _, _, _, _, _, x_n, y_n, z_n, Nx, Ny, Nz = load_block_data(plt_dir, bid, step)
        res = analyze_wall_block(bid, x_n, y_n, z_n, Nx, Ny, Nz, face, u_tau, nu_w; R0=R0)
        push!(results, res)
    end

    # ── Step 3: Print per-block results ──
    println("  ┌──────────────────────────────────────────────────────────────────────────┐")
    println("  │                  First-Cell Grid Resolution (Wall Units)                 │")
    println("  ├──────┬──────────────────────┬──────────────────────┬─────────────────────┤")
    println("  │ Blk  │    Δx⁺ (streamwise)  │  Δr⁺ (wall-normal)  │ (rΔθ)⁺ (circum.)   │")
    println("  │      │  min   mean    max    │  min   mean   max   │  min   mean   max   │")
    println("  ├──────┼──────────────────────┼──────────────────────┼─────────────────────┤")
    for r in results
        @printf("  │  %d   │ %5.1f  %5.1f  %5.1f   │ %5.2f  %5.2f  %5.2f  │ %5.1f  %5.1f  %5.1f  │\n",
            r["bid"],
            r["dx+_min"], r["dx+_mean"], r["dx+_max"],
            r["dr+_min"], r["dr+_mean"], r["dr+_max"],
            r["dth+_min"], r["dth+_mean"], r["dth+_max"])
    end
    println("  └──────┴──────────────────────┴──────────────────────┴─────────────────────┘")
    println()

    # ── Step 4: Global summary ──
    all_dx  = vcat([r["dx+_mean"] for r in results]...)
    all_dr  = vcat([r["dr+_mean"] for r in results]...)
    all_dth = vcat([r["dth+_mean"] for r in results]...)

    global_dr_min  = minimum(r["dr+_min"] for r in results)
    global_dr_max  = maximum(r["dr+_max"] for r in results)
    global_dx_mean = mean(all_dx)
    global_dr_mean = mean(all_dr)
    global_dth_mean = mean(all_dth)

    println("  ┌─────────────────────────────────────────────────┐")
    println("  │             Global Summary                      │")
    println("  ├─────────────────────────────────────────────────┤")
    @printf("  │  Δx⁺   (streamwise)  :  %.1f  (mean across blocks) │\n", global_dx_mean)
    @printf("  │  Δr⁺   (wall-normal) :  %.2f ~ %.2f (min~max)     │\n", global_dr_min, global_dr_max)
    @printf("  │  (rΔθ)⁺ (circum.)    :  %.1f  (mean across blocks) │\n", global_dth_mean)
    println("  ├─────────────────────────────────────────────────┤")

    # DNS reference criteria
    ok_dx  = global_dx_mean  < 12.0
    ok_dr  = global_dr_max   < 1.0
    ok_dth = global_dth_mean < 6.0

    @printf("  │  DNS criteria: Δx⁺<12  Δr⁺<1  (rΔθ)⁺<6       │\n")
    @printf("  │  Status:       %s      %s      %s              │\n",
        ok_dx ? "✓" : "✗", ok_dr ? "✓" : "✗", ok_dth ? "✓" : "✗")
    println("  └─────────────────────────────────────────────────┘")
    println()

    # ── Step 5: Dimensional summary ──
    println("  ┌─────────────────────────────────────────────────┐")
    println("  │          Dimensional First-Cell Sizes            │")
    println("  ├──────┬──────────┬──────────┬──────────┐         │")
    println("  │ Blk  │  Δx (m)  │  Δr (m)  │ rΔθ (m)  │         │")
    println("  ├──────┼──────────┼──────────┼──────────┤         │")
    for r in results
        @printf("  │  %d   │ %.4e │ %.4e │ %.4e │\n",
            r["bid"], r["dx_mean"], r["dr_mean"], r["dth_mean"])
    end
    println("  └──────┴──────────┴──────────┴──────────┘")
    println()

    # ── Step 6: Export CSV ──
    csv_file = joinpath(plt_dir, "grid_resolution.csv")
    open(csv_file, "w") do io
        println(io, "# Pipe Flow Grid Resolution in Wall Units — Step $step")
        @printf(io, "# u_tau=%.6f, Re_tau=%.1f, l_star=%.6e, nu_w=%.6e\n", u_tau, Re_tau, l_star, nu_w)
        println(io, "block,wall_face,N_cells,dx+_min,dx+_mean,dx+_max,dr+_min,dr+_mean,dr+_max,dth+_min,dth+_mean,dth+_max,dx_mean,dr_mean,dth_mean")
        for r in results
            @printf(io, "%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6e,%.6e,%.6e\n",
                r["bid"], r["face"], r["N_cells"],
                r["dx+_min"], r["dx+_mean"], r["dx+_max"],
                r["dr+_min"], r["dr+_mean"], r["dr+_max"],
                r["dth+_min"], r["dth+_mean"], r["dth+_max"],
                r["dx_mean"], r["dr_mean"], r["dth_mean"])
        end
    end
    @printf("  ✓ CSV exported to: %s\n", csv_file)

    println()
    println("═" ^ 78)
    println("  ✓ Grid resolution analysis complete")
    println("═" ^ 78)
end

# ─── CLI Entry Point ───
if abspath(PROGRAM_FILE) == @__FILE__
    plt_dir = length(ARGS) >= 1 ? ARGS[1] : "PLT"
    step = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nothing
    main(; plt_dir=plt_dir, step=step)
end
