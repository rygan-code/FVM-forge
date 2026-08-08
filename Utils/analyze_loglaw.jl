# =============================================================================
# analyze_loglaw.jl — Log-Law Analysis for Pipe Flow (Butterfly Mesh)
#
# Computes wall-normal mean velocity profiles in wall units (y⁺, u⁺)
# with **Van Driest density-weighted transformation** for compressible flows:
#   u⁺_VD = ∫₀^u⁺ √(ρ/ρ_w) du⁺
# and compares to the theoretical log-law: u⁺_VD = (1/κ) ln(y⁺) + B
#
# Supports **multi-snapshot time averaging** for converged statistics.
#
# Usage:
#   julia Utils/analyze_loglaw.jl [PLT_DIR] [STEP_SPEC]
#
#   STEP_SPEC can be:
#     622000          → single snapshot
#     400000:622000   → all snapshots in range [400000, 622000]
#     last:10         → last 10 available snapshots
#     (omitted)       → auto-detect latest single snapshot
#
# Outputs:
#   - Console: u_τ, Re_τ, Cf, profile statistics
#   - PLT/loglaw_profile.csv
#   - PLT/loglaw_profile.png
# =============================================================================

using HDF5, Statistics, Printf, DelimitedFiles, CairoMakie, LaTeXStrings

# ─── Physical constants (must match run/baseline/pipe_baseline.jl) ───
const γ  = 1.4
const Rg = 287.0
const C_s = 1.458e-6
const T_s = 110.4

# ─── Log-law constants ───
const κ = 0.41    # von Kármán constant
const B = 5.2     # additive constant (smooth wall)

# ─── Sutherland viscosity ───
function sutherland_mu(T)
    return C_s * T^1.5 / (T + T_s)
end

"""
    get_available_steps(plt_dir)

Scan PLT directory and return sorted list of all available step numbers.
"""
function get_available_steps(plt_dir)
    files = readdir(plt_dir)
    plt_files = filter(f -> occursin(r"^plt-\d+-b0\.h5$", f), files)
    if isempty(plt_files)
        error("No plt-*-b0.h5 files found in $plt_dir/")
    end
    steps = [parse(Int, match(r"plt-(\d+)-b0", f).captures[1]) for f in plt_files]
    return sort(steps)
end

"""
    parse_step_spec(spec_str, plt_dir)

Parse step specification string into a list of steps.
  "622000"        → [622000]
  "400000:622000" → all available steps in [400000, 622000]
  "last:10"       → last 10 available steps
"""
function parse_step_spec(spec_str, plt_dir)
    all_steps = get_available_steps(plt_dir)

    if occursin("last:", spec_str)
        n = parse(Int, split(spec_str, ":")[2])
        n = min(n, length(all_steps))
        return all_steps[end-n+1:end]
    elseif occursin(":", spec_str)
        parts = split(spec_str, ":")
        s1 = parse(Int, parts[1])
        s2 = parse(Int, parts[2])
        return filter(s -> s1 <= s <= s2, all_steps)
    else
        step = parse(Int, spec_str)
        return [step]
    end
end

"""
    load_block_flow(plt_dir, bid, step)

Load flow field for a single block at a single step.
Returns (rho, u, v, w, T, p)
"""
function load_block_flow(plt_dir, bid, step)
    plt_file = joinpath(plt_dir, "plt-$(step)-b$(bid).h5")
    if !isfile(plt_file)
        error("File not found: $plt_file")
    end

    fid = h5open(plt_file, "r")
    rho = read(fid["rho"])
    u   = read(fid["u"])
    v   = read(fid["v"])
    w   = read(fid["w"])
    T   = read(fid["T"])
    p   = read(fid["p"])
    close(fid)

    return Float64.(rho), Float64.(u), Float64.(v), Float64.(w),
           Float64.(T), Float64.(p)
end

"""
    load_block_mesh(bid)

Load mesh coordinates for a single block (only needs to be done once).
Returns (x_c, y_c, z_c, Nx, Ny, Nz)  — cell centers
"""
function load_block_mesh(bid)
    mesh_file = "MESH/mesh_b$(bid).h5"
    mfid = h5open(mesh_file, "r")
    Nx  = read(mfid["Nx"])
    Ny  = read(mfid["Ny"])
    Nz  = read(mfid["Nz"])
    x_n = read(mfid["x"])  # node coordinates: (Nx+1, Ny+1, Nz+1)
    y_n = read(mfid["y"])
    z_n = read(mfid["z"])
    close(mfid)

    # Cell centers from node coordinates (average of 8 corners)
    x_c = 0.125 .* (x_n[1:end-1,1:end-1,1:end-1] .+ x_n[2:end,1:end-1,1:end-1] .+
                     x_n[1:end-1,2:end,1:end-1]   .+ x_n[2:end,2:end,1:end-1]   .+
                     x_n[1:end-1,1:end-1,2:end]   .+ x_n[2:end,1:end-1,2:end]   .+
                     x_n[1:end-1,2:end,2:end]     .+ x_n[2:end,2:end,2:end])
    y_c = 0.125 .* (y_n[1:end-1,1:end-1,1:end-1] .+ y_n[2:end,1:end-1,1:end-1] .+
                     y_n[1:end-1,2:end,1:end-1]   .+ y_n[2:end,2:end,1:end-1]   .+
                     y_n[1:end-1,1:end-1,2:end]   .+ y_n[2:end,1:end-1,2:end]   .+
                     y_n[1:end-1,2:end,2:end]     .+ y_n[2:end,2:end,2:end])
    z_c = 0.125 .* (z_n[1:end-1,1:end-1,1:end-1] .+ z_n[2:end,1:end-1,1:end-1] .+
                     z_n[1:end-1,2:end,1:end-1]   .+ z_n[2:end,2:end,1:end-1]   .+
                     z_n[1:end-1,1:end-1,2:end]   .+ z_n[2:end,1:end-1,2:end]   .+
                     z_n[1:end-1,2:end,2:end]     .+ z_n[2:end,2:end,2:end])

    return Float64.(x_c), Float64.(y_c), Float64.(z_c), Nx, Ny, Nz
end

"""
    compute_wall_profiles(plt_dir, steps; R0=0.5, N_bins=800)

Compute circumferentially, streamwise, and TIME-averaged mean velocity profile
in wall-normal coordinates for the pipe flow butterfly mesh.

`steps` is a Vector{Int} of step numbers to average over.
"""
function compute_wall_profiles(plt_dir, steps; R0=0.5, N_bins=800)

    # ── Pre-load mesh (once): compute radial position of each cell ──
    mesh_data = [load_block_mesh(bid) for bid in 0:4]
    block_r = Vector{Vector{Float64}}(undef, 5)
    for (ib, (x_c, y_c, z_c, Nx, Ny, Nz)) in enumerate(mesh_data)
        block_r[ib] = vec(sqrt.(y_c.^2 .+ z_c.^2))
    end

    # ── Setup bins ──
    r_max  = R0
    r_edges = range(0.0, r_max, length=N_bins+1)
    r_mid   = 0.5 .* (r_edges[1:end-1] .+ r_edges[2:end])
    y_wall  = R0 .- r_mid
    dr_bin  = r_mid[2] - r_mid[1]

    # Accumulators (across ALL snapshots)
    sum_uz  = zeros(N_bins)
    sum_rho = zeros(N_bins)
    sum_T   = zeros(N_bins)
    sum_w   = zeros(N_bins)

    N_steps = length(steps)
    @printf("  Averaging over %d snapshots: [%d … %d]\n", N_steps, steps[1], steps[end])

    for (ist, step) in enumerate(steps)
        if ist % max(1, N_steps÷10) == 0 || ist == 1 || ist == N_steps
            @printf("    Loading step %d  (%d/%d)\n", step, ist, N_steps)
        end

        step_ok = true
        for bid in 0:4
            local rho, u, v, w, T, p
            try
                rho, u, v, w, T, p = load_block_flow(plt_dir, bid, step)
            catch e
                @printf("    ⚠ Skipping step %d (block %d): %s\n", step, bid, sprint(showerror, e))
                step_ok = false
                break
            end

            r_vec   = block_r[bid+1]
            uz_vec  = vec(u)
            rho_vec = vec(rho)
            T_vec   = vec(T)

            # Nearest-bin (avoids CIC interpolation artifacts in the viscous sublayer)
            for idx in eachindex(r_vec)
                ri = r_vec[idx]
                if ri >= r_max || ri < 0.0; continue; end

                bin = clamp(Int(round((ri - r_mid[1]) / dr_bin)) + 1, 1, N_bins)
                sum_uz[bin]  += uz_vec[idx]
                sum_rho[bin] += rho_vec[idx]
                sum_T[bin]   += T_vec[idx]
                sum_w[bin]   += 1.0
            end
        end
    end

    # ── Finalize time+space averaged profiles ──
    mean_uz  = zeros(N_bins)
    mean_rho = zeros(N_bins)
    mean_T   = zeros(N_bins)
    valid = sum_w .> 0.0
    mean_uz[valid]  .= sum_uz[valid]  ./ sum_w[valid]
    mean_rho[valid] .= sum_rho[valid] ./ sum_w[valid]
    mean_T[valid]   .= sum_T[valid]   ./ sum_w[valid]
    count = Int.(round.(sum_w))

    @printf("    Total samples per bin (near wall): %.0f\n", 
            sum_w[findlast(valid)])

    # ── Wall quantities (closest bin to wall) ──
    wall_bins = findall(valid .& (y_wall .< 0.02 * R0))
    if isempty(wall_bins)
        wall_bins = findall(valid)
    end
    sorted_idx = sort(wall_bins, by=i->y_wall[i])

    # Use the bin CLOSEST to wall (smallest y_wall)
    rho_w = mean_rho[sorted_idx[1]]
    T_w   = mean_T[sorted_idx[1]]
    mu_w  = sutherland_mu(T_w)
    nu_w  = mu_w / rho_w

    # ── Wall shear stress from near-wall gradient (ITERATIVE) ──
    # Problem: fitting u = a*y over too wide a region (y⁺ > 5) includes the
    # buffer layer where u⁺ < y⁺, systematically underestimating du/dy by ~5%.
    # Fix: iteratively restrict fit to y⁺ < 5 (viscous sublayer only).
    
    # Step 1: Initial estimate using narrow region (y_wall < 0.005*R0)
    near_wall = findall(valid .& (y_wall .> 0) .& (y_wall .< 0.005 * R0))
    if length(near_wall) < 2
        near_wall = findall(valid .& (y_wall .> 0) .& (y_wall .< 0.02 * R0))
    end
    if length(near_wall) < 2
        near_wall = findall(valid .& (y_wall .> 0))
    end
    near_wall = sort(near_wall, by=i->y_wall[i])
    
    yw_fit = y_wall[near_wall]
    uz_fit = mean_uz[near_wall]
    dudy_wall = sum(yw_fit .* uz_fit) / sum(yw_fit.^2)
    tau_w = mu_w * abs(dudy_wall)
    u_tau = sqrt(tau_w / rho_w)
    
    # Step 2: Iterative refinement — restrict to y⁺ < 5 (viscous sublayer)
    for iter in 1:5
        yp_limit = 5.0 * nu_w / u_tau  # y_wall corresponding to y⁺ = 5
        sublayer = findall(valid .& (y_wall .> 0) .& (y_wall .< yp_limit))
        if length(sublayer) < 2
            break  # not enough points, keep previous estimate
        end
        sublayer = sort(sublayer, by=i->y_wall[i])
        yw_fit = y_wall[sublayer]
        uz_fit = mean_uz[sublayer]
        dudy_wall = sum(yw_fit .* uz_fit) / sum(yw_fit.^2)
        tau_w = mu_w * abs(dudy_wall)
        u_tau_new = sqrt(tau_w / rho_w)
        if abs(u_tau_new - u_tau) / u_tau < 1e-4
            u_tau = u_tau_new
            break
        end
        u_tau = u_tau_new
    end
    
    @printf("    u_τ iterative (y⁺<5): %.4f m/s  (%d sublayer bins)\n",
            u_tau, length(findall(valid .& (y_wall .> 0) .& (y_wall .< 5.0 * nu_w / u_tau))))

    # ── Non-dimensionalize ──
    y_plus = y_wall[valid] .* u_tau ./ nu_w
    u_plus = mean_uz[valid] ./ u_tau

    # ── Van Driest transformation ──
    # u⁺_VD = ∫₀^{u⁺} √(ρ/ρ_w) du'⁺
    # Integrate from wall outward (ascending y⁺ order)
    rho_profile = mean_rho[valid]
    perm_vd = sortperm(y_plus, rev=true)
    perm_wall = reverse(perm_vd)
    u_plus_vd = zeros(length(u_plus))
    # First bin: integrate from u⁺=0 to u⁺[first_bin].  At the wall ρ≈ρ_w,
    # so u⁺_VD ≈ √(ρ_first/ρ_w) × u⁺_first  (not zero!)
    i_first = perm_wall[1]
    u_plus_vd[i_first] = sqrt(rho_profile[i_first] / rho_w) * u_plus[i_first]
    for k in 2:length(perm_wall)
        i_curr = perm_wall[k]
        i_prev = perm_wall[k-1]
        du = u_plus[i_curr] - u_plus[i_prev]
        rho_avg = 0.5 * (rho_profile[i_curr] + rho_profile[i_prev])
        u_plus_vd[i_curr] = u_plus_vd[i_prev] + sqrt(rho_avg / rho_w) * du
    end

    # ── Derived quantities ──
    Re_tau = R0 * u_tau / nu_w
    # Area-weighted bulk velocity for pipe: U_b = ∫u·dA / A = Σ(u_i × r_i × dr) / Σ(r_i × dr)
    # mean_uz[i] is already the average velocity in bin i, so weight by r only (not count).
    r_valid = R0 .- y_wall[valid]  # radial position of valid bins
    U_bulk = sum(mean_uz[valid] .* r_valid) / sum(r_valid)
    Cf = 2.0 * tau_w / (rho_w * U_bulk^2)

    return y_plus, u_plus, u_plus_vd, u_tau, Re_tau, Cf, rho_w, T_w, mu_w, U_bulk, nu_w
end

"""
    analyze_loglaw(; plt_dir="PLT", step_spec=nothing, R0=0.5)

Main entry point: compute profiles for each subdirectory, print summary, export CSV and PNG.
"""
function analyze_loglaw(; plt_dir="PLT", step_spec=nothing, R0=0.5)
    # ── Discover cases ──
    sim_dirs = String[]
    for item in readdir(plt_dir)
        path = joinpath(plt_dir, item)
        if isdir(path) && any(f -> occursin(r"^plt-\d+-b0\.h5$", f), readdir(path))
            push!(sim_dirs, path)
        end
    end
    if isempty(sim_dirs) && any(f -> occursin(r"^plt-\d+-b0\.h5$", f), readdir(plt_dir))
        push!(sim_dirs, plt_dir)
    end
    sort!(sim_dirs)

    println("═" ^ 70)
    println("  Pipe Flow — Log-Law Analysis (Multi-Case)")
    println("═" ^ 70)
    @printf("  PLT directory : %s\n", plt_dir)
    @printf("  Pipe radius   : %.3f\n", R0)
    println()

    sim_results = []
    yp_max_overall = 0.0
    Re_tau_max = 0.0
    uvd_max_overall = 0.0

    for sdir in sim_dirs
        label = basename(sdir) == basename(plt_dir) ? "Current" : basename(sdir)
        
        # ── Determine steps for this case ──
        if step_spec === nothing
            all_steps = get_available_steps(sdir)
            n_avg = min(10, length(all_steps))
            curr_steps = all_steps[end-n_avg+1:end]  # default: last 10 snapshots
        else
            curr_steps = parse_step_spec(step_spec, sdir)
        end
        N_steps = length(curr_steps)
        step_label = N_steps == 1 ? "Step $(curr_steps[1])" : "Steps $(curr_steps[1])–$(curr_steps[end]) (N=$(N_steps))"

        @printf("  ▶ Processing Case: %s\n", label)
        
        # ── Compute profiles ──
        y_plus, u_plus, u_plus_vd, u_tau, Re_tau, Cf, rho_w, T_w, mu_w, U_bulk, nu_w =
            compute_wall_profiles(sdir, curr_steps; R0=R0)

        push!(sim_results, (
            label = label,
            sdir = sdir,
            step_label = step_label,
            y_plus = y_plus,
            u_plus = u_plus,
            u_plus_vd = u_plus_vd,
            u_tau = u_tau,
            Re_tau = Re_tau,
            Cf = Cf,
            rho_w = rho_w,
            T_w = T_w,
            mu_w = mu_w,
            nu_w = nu_w,
            U_bulk = U_bulk,
            N_steps = N_steps
        ))

        yp_max_overall = max(yp_max_overall, maximum(y_plus))
        Re_tau_max = max(Re_tau_max, Re_tau)
        uvd_max_overall = max(uvd_max_overall, maximum(u_plus_vd))

        # ── Print summary ──
        println()
        println("  ┌─────────────────────────────────────────┐")
        println("  │        Wall Quantities Summary          │")
        println("  ├─────────────────────────────────────────┤")
        @printf("  │  u_τ         = %10.4f m/s           │\n", u_tau)
        @printf("  │  Re_τ        = %10.1f               │\n", Re_tau)
        @printf("  │  Cf           = %10.6f              │\n", Cf)
        @printf("  │  τ_w          = %10.4f Pa            │\n", rho_w * u_tau^2)
        @printf("  │  ρ_wall       = %10.6f kg/m³        │\n", rho_w)
        @printf("  │  T_wall       = %10.2f K             │\n", T_w)
        @printf("  │  μ_wall       = %10.4e Pa·s          │\n", mu_w)
        @printf("  │  ν_wall       = %10.4e m²/s          │\n", nu_w)
        @printf("  │  U_bulk       = %10.4f m/s           │\n", U_bulk)
        @printf("  │  N_snapshots  = %10d               │\n", N_steps)
        println("  └─────────────────────────────────────────┘")
        
        # ── Error analysis in log layer (30 < y⁺ < 0.2 Re_τ) ──
        log_region = findall(yp -> 30 < yp < 0.2 * Re_tau, y_plus)
        if !isempty(log_region)
            u_log_theory = (1/κ) .* log.(y_plus[log_region]) .+ B
            error_rms_raw = sqrt(mean((u_plus[log_region] .- u_log_theory).^2))
            error_rms_vd = sqrt(mean((u_plus_vd[log_region] .- u_log_theory).^2))
            error_max_vd = maximum(abs.(u_plus_vd[log_region] .- u_log_theory))
            @printf("  Log-law fit (30 < y⁺ < 0.2·Re_τ):\n")
            @printf("    RMS error in u⁺_raw : %.4f\n", error_rms_raw)
            @printf("    RMS error in u⁺_VD  : %.4f  (Van Driest)\n", error_rms_vd)
            @printf("    Max error in u⁺_VD  : %.4f\n", error_max_vd)
        end
        println()

        # ── Export Profile to CSV ──
        perm = sortperm(y_plus)
        y_sorted   = y_plus[perm]
        u_sorted   = u_plus[perm]
        uvd_sorted = u_plus_vd[perm]
        csv_file = joinpath(sdir, "loglaw_profile_$(label).csv")
        try
            open(csv_file, "w") do io
                println(io, "# Pipe Flow Log-Law Profile — Case: $label, $step_label")
                @printf(io, "# u_tau=%.6f, Re_tau=%.1f, Cf=%.6f, N_snapshots=%d\n", u_tau, Re_tau, Cf, N_steps)
                println(io, "# Van Driest transform: u_plus_VD = integral_0^u+ sqrt(rho/rho_w) du+")
                println(io, "y_plus,u_plus,u_plus_VD,u_plus_loglaw,u_plus_viscous")
                for i in eachindex(y_sorted)
                    yp = y_sorted[i]
                    @printf(io, "%.6f,%.6f,%.6f,%.6f,%.6f\n", yp, u_sorted[i], uvd_sorted[i], 
                            yp > 5.0 ? (1/κ) * log(yp) + B : NaN, yp < 30.0 ? yp : NaN)
                end
            end
        catch e
            println("  ! Warning: Failed to export CSV (file might be open/locked).")
        end
        @printf("  ✓ Profile exported to: %s\n\n", csv_file)
    end

    # ── Load Reference DNS data (Pirozzoli, Modesti, Xiao) ──
    # Scan for reference .txt files in the root PLT directory
    all_ref_files = filter(f -> (occursin(r"(?i)xiao"i, f) || occursin(r"(?i)modesti"i, f)) && endswith(f, ".txt"),
                           readdir(plt_dir))
    sort!(all_ref_files)

    ref_datasets = []
    for ref_fname in all_ref_files
        ref_file = joinpath(plt_dir, ref_fname)
        is_modesti = occursin(r"(?i)modesti"i, ref_fname)
        ref_yp, ref_up = Float64[], Float64[]
        @printf("  Loading reference: %s\n", ref_fname)
        for line in eachline(ref_file)
            stripped = strip(line)
            (isempty(stripped) || startswith(stripped, '%') || startswith(stripped, '#')) && continue
            vals = split(stripped)
            try
                if is_modesti && length(vals) >= 5
                    push!(ref_yp, parse(Float64, vals[2]))
                    push!(ref_up, parse(Float64, vals[5]))
                elseif !is_modesti && length(vals) >= 2
                    push!(ref_yp, parse(Float64, vals[1]))
                    push!(ref_up, parse(Float64, vals[2]))
                end
            catch; continue end
        end
        if length(ref_yp) > 10
            label = replace(ref_fname, ".txt" => "")
            # Strip trailing condition tag (e.g. "Ma1.5", "Ro05") — keep only author (year)
            label = replace(label, r"\)\s+\S+$" => ")")
            @printf("    → %d points\n", length(ref_yp))
            push!(ref_datasets, (label=label, yp=ref_yp, up=ref_up))
        end
    end
    println()

    # ── Publication-quality PNG plot ──
    cm_to_pt = 72 / 2.54
    update_theme!(fontsize=10, fonts=(; regular="Times New Roman", bold="Times New Roman Bold", italic="Times New Roman Italic"))

    fig = Figure(size=(round(Int, 18 * cm_to_pt), round(Int, 8.5 * cm_to_pt)), figure_padding=(2, 4, 2, 4))
    
    ax1 = Axis(fig[1, 1], aspect = 4/3, xlabel = L"y^+", ylabel = L"u^+_{VD}", xlabelpadding = 2, ylabelpadding = 2,
        xlabelsize = 12, ylabelsize = 12, xscale = log10, xminorticksvisible = true, xminorticks = IntervalsBetween(9),
        xgridvisible = true, ygridvisible = true, xgridstyle = :dash, ygridstyle = :dash,
        title = "(a)", titlealign = :left, titlefont = :regular)

    ax2 = Axis(fig[1, 2], aspect = 4/3, xlabel = L"y^+", ylabel = L"y^+ \mathrm{d}u^+_{VD} / \mathrm{d}y^+", xlabelpadding = 2, ylabelpadding = 2,
        xlabelsize = 12, ylabelsize = 12, xscale = log10, xminorticksvisible = true, xminorticks = IntervalsBetween(9),
        xgridvisible = true, ygridvisible = true, xgridstyle = :dash, ygridstyle = :dash,
        title = "(b)", titlealign = :left, titlefont = :regular)

    linkxaxes!(ax1, ax2)

    # Helper function for diagnostic: Xi = du / d(ln y)
    function compute_diagnostic(yp, up; N_diff=3, N_smooth=8)
        xi = zeros(length(yp))
        for i in 1+N_diff:length(yp)-N_diff
            d_ln_y = log(yp[i+N_diff]) - log(yp[i-N_diff])
            if d_ln_y > 0
                xi[i] = (up[i+N_diff] - up[i-N_diff]) / d_ln_y
            end
        end
        # Extrapolate boundaries
        for i in 1:N_diff
            if length(xi) >= i + N_diff
                xi[i] = xi[1+N_diff]
            end
        end
        for i in length(yp)-N_diff+1:length(yp)
            if length(xi) >= length(yp)-N_diff
                xi[i] = xi[length(yp)-N_diff]
            end
        end
        
        # Apply smoothing to the derivative to avoid spikes
        xi_smooth = copy(xi)
        for i in 1:length(xi)
            i_start = max(1, i - N_smooth)
            i_end = min(length(xi), i + N_smooth)
            xi_smooth[i] = mean(xi[i_start:i_end])
        end
        return xi_smooth
    end

    # 1. Viscous sublayer: u⁺ = y⁺ (Gray, dashed)
    yp_visc = range(0.3, 12.0, length=100)
    lines!(ax1, collect(yp_visc), collect(yp_visc),
        color=:gray60, linewidth=1.5, linestyle=:dash, label=L"u^+ = y^+")

    # 2. Log law: u⁺ = (1/κ) ln(y⁺) + B (Black, dashed)
    yp_log_max = min(yp_max_overall, 0.3 * Re_tau_max)
    yp_log = exp10.(range(log10(10), log10(max(yp_log_max, 50)), length=200))
    up_log = (1/κ) .* log.(yp_log) .+ B
    lines!(ax1, collect(yp_log), collect(up_log),
        color=:black, linewidth=2, linestyle=:dash,
        label=L"\mathrm{Log\;law}")
    
    # 2b. Theoretical diagnostic function for log-law: Xi = 1/kappa
    lines!(ax2, [0.3, max(yp_max_overall * 1.5, 200)], [1/κ, 1/κ],
        color=:black, linewidth=2, linestyle=:dash,
        label=latexstring("1/$(round(κ, digits=2))"))

    # 3. Plot reference data (Categorical palette)
    xi_max_overall = 5.0 # baseline minimum max for y-axis scaling
    for ds in ref_datasets
        # Xiao data has very few points, use smaller N_diff for it
        if occursin(r"(?i)xiao", ds.label)
            xi_full = compute_diagnostic(ds.yp, ds.up, N_diff=1, N_smooth=3)
        else
            xi_full = compute_diagnostic(ds.yp, ds.up, N_diff=2, N_smooth=6)
        end
        
        if occursin(r"(?i)modesti", ds.label)
            col = ("#1B9E77", 0.9); mk = :circle; mk_size = 5
        elseif occursin(r"(?i)xiao", ds.label)
            col = ("#7570B3", 0.9); mk = :rect; mk_size = 5
        else
            col = ("#D95F02", 0.9); mk = :diamond; mk_size = 5
        end
        
        # Subsample Modesti data to be uniform in log-scale
        if occursin(r"(?i)modesti", ds.label) && length(ds.yp) > 25
            log_min = log10(max(0.1, minimum(ds.yp)))
            log_max = log10(maximum(ds.yp))
            log_targets = range(log_min, log_max, length=30)
            sub_yp, sub_up, sub_xi = Float64[], Float64[], Float64[]
            for target in log_targets
                idx = argmin(abs.(log10.(max.(0.1, ds.yp)) .- target))
                if isempty(sub_yp) || ds.yp[idx] > sub_yp[end] * 1.05
                    push!(sub_yp, ds.yp[idx]); push!(sub_up, ds.up[idx]); push!(sub_xi, xi_full[idx])
                end
            end
            plot_yp, plot_up, plot_xi = sub_yp, sub_up, sub_xi
        else
            plot_yp, plot_up, plot_xi = ds.yp, ds.up, xi_full
        end

        ref_mask = plot_yp .<= yp_max_overall * 1.2
        scatter!(ax1, plot_yp[ref_mask], plot_up[ref_mask],
            color=col, markersize=mk_size, strokewidth=0, marker=mk, label=ds.label)
        
        # We only want to plot the diagnostic function for y+ > 3 to avoid infinite spikes at the wall
        # Skip plotting Xiao's sparse data in the derivative plot
        if !occursin(r"(?i)xiao", ds.label)
            ref_xi_mask = ref_mask .& (plot_yp .> 3.0)
            scatter!(ax2, plot_yp[ref_xi_mask], plot_xi[ref_xi_mask],
                color=col, markersize=mk_size, strokewidth=0, marker=mk)
            
            if any(ref_xi_mask)
                xi_max_overall = max(xi_max_overall, maximum(plot_xi[ref_xi_mask]))
            end
        end
    end

    # 4. Plot simulation data — fixed 4-color palette matching stratification comparison
    function _sim_color(label)
        if occursin("Ma15Ro00", label) || occursin("Ro=0", label)
            return :royalblue
        elseif occursin("Ma03Ro03", label)
            return :crimson
        elseif occursin("Ma08Ro03", label)
            return :darkorange
        elseif occursin("Ma15Ro03", label)
            return :goldenrod
        else
            return :gray40
        end
    end
    for res in sim_results
        col = _sim_color(res.label)
        
        perm = sortperm(res.y_plus)
        y_sorted = res.y_plus[perm]
        uvd_sorted = res.u_plus_vd[perm]

        # Smooth u⁺_VD with a symmetric moving average
        window = 8
        uvd_smoothed = copy(uvd_sorted)
        for i in 1:length(uvd_smoothed)
            w = min(window, i - 1, length(uvd_smoothed) - i)
            i_start = i - w
            i_end = i + w
            uvd_smoothed[i] = mean(uvd_sorted[i_start:i_end])
        end

        # Use a larger diff window and stronger smoothing for the derivative 
        # to suppress the mathematical amplification of topological wiggles
        xi_sim = compute_diagnostic(y_sorted, uvd_smoothed, N_diff=10, N_smooth=25)
        
        plot_label = length(sim_results) == 1 && res.label == "Current" ? L"u^+_{VD}" : "$(res.label)"
        lines!(ax1, y_sorted, uvd_smoothed, color=col, linewidth=2.5, linestyle=:solid, label=plot_label)
        
        sim_xi_mask = y_sorted .> 3.0
        lines!(ax2, y_sorted[sim_xi_mask], xi_sim[sim_xi_mask], color=col, linewidth=2.5, linestyle=:solid)
        
        if any(sim_xi_mask)
            xi_max_overall = max(xi_max_overall, maximum(xi_sim[sim_xi_mask]))
        end
    end

    xlims!(ax1, 0.3, max(yp_max_overall * 1.5, 200))
    ylims!(ax1, 0, max(uvd_max_overall * 1.15, 25))
    ylims!(ax2, 0, min(35, max(10, xi_max_overall * 1.15)))
    
    axislegend(ax1, position=:lt, framevisible=true, labelsize=8, patchsize=(15, 8), rowgap=1, padding=(4, 4, 2, 2))
    axislegend(ax2, position=:lt, framevisible=true, labelsize=8, patchsize=(15, 8), rowgap=1, padding=(4, 4, 2, 2))

    # Annotation box
    if length(sim_results) == 1
        res = sim_results[1]
        info_lines = [
            latexstring("u_\\tau = $(@sprintf("%.3f", res.u_tau))\\;\\mathrm{m/s}"),
            latexstring("Re_\\tau = $(@sprintf("%.0f", res.Re_tau))")
        ]
        for (k, ltx) in enumerate(info_lines)
            text!(ax1, 0.97, 0.04 + (length(info_lines)-k)*0.05,
                text=ltx, align=(:right, :bottom), space=:relative, fontsize=8, color=:gray30)
        end
    end

    png_file = joinpath(plt_dir, "loglaw_profile.png")
    save(png_file, fig, px_per_unit=4)
    @printf("  ✓ Plot saved to: %s\n", png_file)

    println()
    println("═" ^ 70)
    println("  ✓ Log-law analysis complete")
    println("═" ^ 70)

    # Return the last one for API compatibility, or the vector if needed
    if length(sim_results) == 1
        return sim_results[1].y_plus, sim_results[1].u_plus, sim_results[1].u_plus_vd, sim_results[1].u_tau, sim_results[1].Re_tau
    else
        return sim_results
    end
end

# ─── CLI Entry Point ──
if abspath(PROGRAM_FILE) == @__FILE__
    plt_dir = length(ARGS) >= 1 ? ARGS[1] : "PLT"
    step_spec = length(ARGS) >= 2 ? ARGS[2] : nothing
    analyze_loglaw(; plt_dir=plt_dir, step_spec=step_spec)
end
