# =============================================================================
# analyze_reynolds_stress.jl — Reynolds Stress Profiles for Pipe Flow
#
# Computes Reynolds stress components ⟨u'ᵢu'ⱼ⟩ as functions of y⁺
# from instantaneous snapshots (plt-*.h5).
#
# Produces a figure comparing with a reference image (if provided).
#
# Usage:
#   julia Utils/analyze_reynolds_stress.jl <PLT_DIR/case_dir>
#
# =============================================================================

using HDF5, Statistics, Printf, CairoMakie

# ─── Physical constants ───
const γ  = 1.4
const Rg = 287.0
const C_s = 1.458e-6
const T_s = 110.4

function sutherland_mu(T)
    return C_s * T^1.5 / (T + T_s)
end

# ─── Mesh loader ───
function load_block_mesh(mesh_dir, bid)
    mesh_file = joinpath(mesh_dir, "mesh_b$(bid).h5")
    mfid = h5open(mesh_file, "r")
    Nx = read(mfid["Nx"]); Ny = read(mfid["Ny"]); Nz = read(mfid["Nz"])
    x_n = read(mfid["x"]); y_n = read(mfid["y"]); z_n = read(mfid["z"])
    close(mfid)
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

function load_block_flow(plt_dir, bid, step; prefix="plt")
    fname = joinpath(plt_dir, "$(prefix)-$(step)-b$(bid).h5")
    if !isfile(fname); error("File not found: $fname"); end
    fid = h5open(fname, "r")
    rho = read(fid["rho"]); u = read(fid["u"]); v = read(fid["v"])
    w = read(fid["w"]); T = read(fid["T"]); p = read(fid["p"])
    close(fid)
    return Float64.(rho), Float64.(u), Float64.(v), Float64.(w), Float64.(T), Float64.(p)
end

function get_steps(plt_dir; prefix="plt")
    files = readdir(plt_dir)
    pat = Regex("^$(prefix)-(\\d+)-b0\\.h5\$")
    matched = filter(f -> occursin(pat, f), files)
    steps = [parse(Int, match(pat, f).captures[1]) for f in matched]
    return sort(steps)
end

# ─── Main analysis ───
function main()
    plt_dir = length(ARGS) >= 1 ? ARGS[1] : "PLT/Ma15Ro03"
    mesh_dir = "MESH"
    R0 = 0.5
    N_bins = 300  # more bins for y⁺ plot

    println("═"^70)
    println("  Reynolds Stress Analysis — Butterfly Pipe Flow")
    println("═"^70)
    @printf("  Data dir : %s\n", plt_dir)

    plt_steps = get_steps(plt_dir; prefix="plt")
    @printf("  PLT steps: %d files, [%d … %d]\n", length(plt_steps), plt_steps[1], plt_steps[end])
    println()

    # ── Load mesh ──
    mesh_data = [load_block_mesh(mesh_dir, bid) for bid in 0:4]
    block_r = Vector{Array{Float64,3}}(undef, 5)
    block_theta = Vector{Array{Float64,3}}(undef, 5)
    for (ib, (xc, yc, zc, Nx, Ny, Nz)) in enumerate(mesh_data)
        block_r[ib] = sqrt.(yc.^2 .+ zc.^2)
        block_theta[ib] = atan.(zc, yc)
    end

    # ── Bins in r ──
    r_edges = range(0.0, R0, length=N_bins+1)
    r_mid = 0.5 .* (r_edges[1:end-1] .+ r_edges[2:end])
    dr = r_mid[2] - r_mid[1]
    y_wall = R0 .- r_mid

    # ══════════════════════════════════════════════════
    # Pass 1: Mean profiles (time + azimuthal + axial)
    # ══════════════════════════════════════════════════
    sum_ux = zeros(N_bins); sum_ur = zeros(N_bins); sum_ut = zeros(N_bins)
    sum_rho = zeros(N_bins); sum_T = zeros(N_bins)
    sum_w = zeros(N_bins)

    @printf("  Pass 1: Computing mean profiles from %d snapshots...\n", length(plt_steps))
    for step in plt_steps
        for bid in 0:4
            local rho, u, vv, w, T, p
            try
                rho, u, vv, w, T, p = load_block_flow(plt_dir, bid, step)
            catch e
                @printf("    ⚠ Skipping step %d (block %d): %s\n", step, bid, sprint(showerror, e))
                continue
            end
            r_arr = block_r[bid+1]; θ_arr = block_theta[bid+1]
            for idx in eachindex(r_arr)
                ri = r_arr[idx]
                (ri >= R0 || ri < 0.0) && continue
                bin = clamp(Int(floor(ri / dr)) + 1, 1, N_bins)
                cosθ = cos(θ_arr[idx]); sinθ = sin(θ_arr[idx])
                sum_ux[bin]  += u[idx]
                sum_ur[bin]  += vv[idx]*cosθ + w[idx]*sinθ
                sum_ut[bin]  += -vv[idx]*sinθ + w[idx]*cosθ
                sum_rho[bin] += rho[idx]
                sum_T[bin]   += T[idx]
                sum_w[bin]   += 1.0
            end
        end
    end

    valid = sum_w .> 0
    mean_ux  = zeros(N_bins); mean_ur  = zeros(N_bins); mean_ut  = zeros(N_bins)
    mean_rho = zeros(N_bins); mean_T   = zeros(N_bins)
    mean_ux[valid]  .= sum_ux[valid]  ./ sum_w[valid]
    mean_ur[valid]  .= sum_ur[valid]  ./ sum_w[valid]
    mean_ut[valid]  .= sum_ut[valid]  ./ sum_w[valid]
    mean_rho[valid] .= sum_rho[valid] ./ sum_w[valid]
    mean_T[valid]   .= sum_T[valid]   ./ sum_w[valid]

    # ══════════════════════════════════════════════════
    # Pass 2: Reynolds stresses + per-snapshot convergence
    # ══════════════════════════════════════════════════
    sum_uxux = zeros(N_bins); sum_urur = zeros(N_bins)
    sum_utut = zeros(N_bins); sum_uxur = zeros(N_bins)
    sum_w2   = zeros(N_bins)
    per_snap_Rxx = zeros(length(plt_steps), N_bins)

    @printf("  Pass 2: Computing Reynolds stresses...\n")
    for (si, step) in enumerate(plt_steps)
        snap_uxux = zeros(N_bins); snap_w = zeros(N_bins)
        for bid in 0:4
            local rho, u, vv, w, T, p
            try
                rho, u, vv, w, T, p = load_block_flow(plt_dir, bid, step)
            catch e
                @printf("    ⚠ Skipping step %d (block %d): %s\n", step, bid, sprint(showerror, e))
                continue
            end
            r_arr = block_r[bid+1]; θ_arr = block_theta[bid+1]
            for idx in eachindex(r_arr)
                ri = r_arr[idx]
                (ri >= R0 || ri < 0.0) && continue
                bin = clamp(Int(floor(ri / dr)) + 1, 1, N_bins)
                cosθ = cos(θ_arr[idx]); sinθ = sin(θ_arr[idx])
                u_r = vv[idx]*cosθ + w[idx]*sinθ
                u_t = -vv[idx]*sinθ + w[idx]*cosθ
                ux_p = u[idx] - mean_ux[bin]
                ur_p = u_r - mean_ur[bin]
                ut_p = u_t - mean_ut[bin]
                sum_uxux[bin] += ux_p^2; sum_urur[bin] += ur_p^2
                sum_utut[bin] += ut_p^2; sum_uxur[bin] += ux_p * ur_p
                sum_w2[bin]   += 1.0
                snap_uxux[bin] += ux_p^2; snap_w[bin] += 1.0
            end
        end
        for b in 1:N_bins
            per_snap_Rxx[si, b] = snap_w[b] > 0 ? snap_uxux[b] / snap_w[b] : 0.0
        end
    end

    R_xx = zeros(N_bins); R_rr = zeros(N_bins); R_tt = zeros(N_bins); R_xr = zeros(N_bins)
    v2 = sum_w2 .> 0
    R_xx[v2] .= sum_uxux[v2] ./ sum_w2[v2]
    R_rr[v2] .= sum_urur[v2] ./ sum_w2[v2]
    R_tt[v2] .= sum_utut[v2] ./ sum_w2[v2]
    R_xr[v2] .= sum_uxur[v2] ./ sum_w2[v2]

    # ══════════════════════════════════════════════════
    # Per-block R_xx for artifact detection
    # ══════════════════════════════════════════════════
    last_step = plt_steps[end]
    block_Rxx = zeros(5, N_bins); block_w = zeros(5, N_bins)
    for bid in 0:4
        local rho, u, vv, w, T, p
        loaded = false
        for try_step in reverse(plt_steps)
            try
                rho, u, vv, w, T, p = load_block_flow(plt_dir, bid, try_step)
                last_step = try_step  # update for plot title
                loaded = true
                break
            catch; continue; end
        end
        !loaded && continue
        r_arr = block_r[bid+1]
        for idx in eachindex(r_arr)
            ri = r_arr[idx]; (ri >= R0 || ri < 0) && continue
            bin = clamp(Int(floor(ri / dr)) + 1, 1, N_bins)
            ux_p = u[idx] - mean_ux[bin]
            block_Rxx[bid+1, bin] += ux_p^2; block_w[bid+1, bin] += 1.0
        end
    end
    for bid in 1:5, b in 1:N_bins
        block_w[bid, b] > 0 && (block_Rxx[bid, b] /= block_w[bid, b])
    end

    # ══════════════════════════════════════════════════
    # Wall quantities
    # ══════════════════════════════════════════════════
    wall_bins = findall(valid .& (y_wall .> 0) .& (y_wall .< 0.005 * R0))
    if length(wall_bins) < 2
        wall_bins = findall(valid .& (y_wall .> 0) .& (y_wall .< 0.02 * R0))
    end
    rho_w = mean_rho[wall_bins[end]]  # closest to wall = largest r
    T_w   = mean_T[wall_bins[end]]
    mu_w  = sutherland_mu(T_w)
    nu_w  = mu_w / rho_w

    # Wall shear: iterative fit in viscous sublayer
    yw_fit = y_wall[wall_bins]; uz_fit = mean_ux[wall_bins]
    dudy = sum(yw_fit .* uz_fit) / sum(yw_fit.^2)
    tau_w = mu_w * abs(dudy); u_tau = sqrt(tau_w / rho_w)
    for _ in 1:5
        yp_lim = 5.0 * nu_w / u_tau
        sub = findall(valid .& (y_wall .> 0) .& (y_wall .< yp_lim))
        length(sub) < 2 && break
        dudy = sum(y_wall[sub] .* mean_ux[sub]) / sum(y_wall[sub].^2)
        tau_w = mu_w * abs(dudy); u_tau_new = sqrt(tau_w / rho_w)
        abs(u_tau_new - u_tau)/u_tau < 1e-4 && (u_tau = u_tau_new; break)
        u_tau = u_tau_new
    end
    utau2 = u_tau^2
    Re_tau = R0 * u_tau / nu_w
    y_plus = y_wall .* u_tau ./ nu_w

    @printf("    u_τ = %.4f m/s,  Re_τ = %.1f\n", u_tau, Re_tau)
    @printf("    ρ_w = %.4f kg/m³, T_w = %.2f K\n", rho_w, T_w)

    # ══════════════════════════════════════════════════
    # Plot: 2×2 figure matching reference style (y⁺ log axis)
    # ══════════════════════════════════════════════════
    cm_to_pt = 72 / 2.54
    update_theme!(fontsize=10, fonts=(; regular="Times New Roman", bold="Times New Roman Bold"))

    fig = Figure(size=(round(Int, 18*cm_to_pt), round(Int, 18*cm_to_pt)), figure_padding=(6, 10, 4, 4))

    mask = valid .& (y_plus .> 0.5) .& (y_wall .> 0)
    yp = y_plus[mask]

    # Normalized Reynolds stresses: ⟨u'²⟩⁺ = ⟨u'²⟩ / u_τ²
    Rxx_p = R_xx[mask] ./ utau2
    Rrr_p = R_rr[mask] ./ utau2
    Rtt_p = R_tt[mask] ./ utau2
    Rxr_p = R_xr[mask] ./ utau2
    # q⁺ = sqrt(⟨u'ₓ²⟩⁺ + ⟨u'ᵣ²⟩⁺ + ⟨u'θ²⟩⁺)
    q_plus = sqrt.(Rxx_p .+ Rrr_p .+ Rtt_p)

    # ── (a) q⁺ vs y⁺ ──
    ax1 = Axis(fig[1,1], xlabel="y⁺", ylabel="q⁺",
               xscale=log10, xminorticksvisible=true, xminorticks=IntervalsBetween(9),
               title="(a) Turbulence intensity q⁺", titlealign=:left, titlefont=:regular)
    lines!(ax1, yp, q_plus, linewidth=2, color=:firebrick, label="Current")
    xlims!(ax1, 1, Re_tau)
    ylims!(ax1, 0, nothing)

    # ── (b) Individual ⟨u'²ᵢ⟩⁺ vs y⁺ ──
    ax2 = Axis(fig[1,2], xlabel="y⁺", ylabel="⟨u'²ᵢ⟩⁺",
               xscale=log10, xminorticksvisible=true, xminorticks=IntervalsBetween(9),
               title="(b) Normal stress components", titlealign=:left, titlefont=:regular)
    lines!(ax2, yp, Rxx_p, linewidth=2, color=:firebrick,   label="⟨u'²ₓ⟩⁺")
    lines!(ax2, yp, Rrr_p, linewidth=2, color=:steelblue,   label="⟨u'²ᵣ⟩⁺")
    lines!(ax2, yp, Rtt_p, linewidth=2, color=:seagreen,    label="⟨u'²θ⟩⁺")
    lines!(ax2, yp, -Rxr_p, linewidth=2, color=:darkorange, label="-⟨u'ₓu'ᵣ⟩⁺")
    axislegend(ax2, position=:rt, framevisible=true, labelsize=8)
    xlims!(ax2, 1, Re_tau)
    ylims!(ax2, 0, nothing)

    # ── (c) Per-block R_xx — artifact detection ──
    ax3 = Axis(fig[2,1], xlabel="y⁺", ylabel="⟨u'²ₓ⟩⁺",
               xscale=log10, xminorticksvisible=true, xminorticks=IntervalsBetween(9),
               title="(c) Per-Block ⟨u'²ₓ⟩⁺ (step=$(last_step))", titlealign=:left, titlefont=:regular)
    block_colors = [:firebrick, :steelblue, :seagreen, :darkorange, :mediumpurple]
    block_labels = ["Block 0 (center)", "Block 1", "Block 2", "Block 3", "Block 4"]
    for bid in 1:5
        bm = (block_w[bid, :] .> 10) .& mask
        if any(bm)
            lines!(ax3, y_plus[bm], block_Rxx[bid, bm]./utau2,
                   linewidth=1.5, color=block_colors[bid], label=block_labels[bid])
        end
    end
    axislegend(ax3, position=:rt, framevisible=true, labelsize=7)
    xlims!(ax3, 1, Re_tau)
    ylims!(ax3, 0, nothing)

    # ── (d) Convergence: running average of R_xx at y⁺ ≈ 15 (peak) ──
    ax4 = Axis(fig[2,2], xlabel="Step", ylabel="⟨u'²ₓ⟩⁺ (running avg)",
               title="(d) Convergence at y⁺ ≈ 15", titlealign=:left, titlefont=:regular)
    target_bin = argmin(abs.(y_plus .- 15.0))
    if valid[target_bin]
        snap_vals = per_snap_Rxx[:, target_bin] ./ utau2
        # Running cumulative average
        running_avg = cumsum(snap_vals) ./ (1:length(snap_vals))
        scatter!(ax4, Float64.(plt_steps), snap_vals,
                 markersize=6, color=(:firebrick, 0.5), strokewidth=0, label="Per snapshot")
        lines!(ax4, Float64.(plt_steps), running_avg,
               linewidth=2, color=:black, label="Running mean")
        axislegend(ax4, position=:rt, framevisible=true, labelsize=8)
    end

    # Annotation
    text!(ax1, 0.03, 0.95,
          text="uτ=$(@sprintf("%.2f", u_tau)) m/s\nReτ=$(@sprintf("%.0f", Re_tau))\nSteps: $(plt_steps[1])–$(plt_steps[end]) (N=$(length(plt_steps)))",
          space=:relative, align=(:left, :top), fontsize=7, color=:gray30)

    png_file = joinpath(plt_dir, "reynolds_stress.png")
    save(png_file, fig, px_per_unit=4)
    @printf("\n  ✓ Plot saved to: %s\n", png_file)
    println("═"^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
