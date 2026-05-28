# =============================================================================
# check_bump.jl — Test if near-wall u⁺ bump depends on bin count
#
# Runs the same mean profile computation with N_bins = 200, 800, 3200
# and overlays the results to determine if the bump is a binning artifact.
# =============================================================================

using HDF5, Statistics, Printf, CairoMakie

const C_s = 1.458e-6; const T_s = 110.4
sutherland_mu(T) = C_s * T^1.5 / (T + T_s)

function load_block_mesh(mesh_dir, bid)
    mfid = h5open(joinpath(mesh_dir, "mesh_b$(bid).h5"), "r")
    Nx = read(mfid["Nx"]); Ny = read(mfid["Ny"]); Nz = read(mfid["Nz"])
    x_n = read(mfid["x"]); y_n = read(mfid["y"]); z_n = read(mfid["z"])
    close(mfid)
    avg(a) = 0.125 .* (a[1:end-1,1:end-1,1:end-1] .+ a[2:end,1:end-1,1:end-1] .+
                        a[1:end-1,2:end,1:end-1]   .+ a[2:end,2:end,1:end-1]   .+
                        a[1:end-1,1:end-1,2:end]   .+ a[2:end,1:end-1,2:end]   .+
                        a[1:end-1,2:end,2:end]     .+ a[2:end,2:end,2:end])
    return Float64.(avg(x_n)), Float64.(avg(y_n)), Float64.(avg(z_n)), Nx, Ny, Nz
end

function compute_profile(plt_dir, steps, mesh_data, block_r, R0, N_bins)
    r_edges = range(0.0, R0, length=N_bins+1)
    r_mid = 0.5 .* (r_edges[1:end-1] .+ r_edges[2:end])
    y_wall = R0 .- r_mid
    dr_bin = r_mid[2] - r_mid[1]

    sum_uz = zeros(N_bins); sum_rho = zeros(N_bins)
    sum_T = zeros(N_bins); sum_w = zeros(N_bins)

    for step in steps
        for bid in 0:4
            fid = h5open(joinpath(plt_dir, "plt-$(step)-b$(bid).h5"), "r")
            u = Float64.(vec(read(fid["u"])))
            rho = Float64.(vec(read(fid["rho"])))
            T = Float64.(vec(read(fid["T"])))
            close(fid)
            r_vec = block_r[bid+1]
            for idx in eachindex(r_vec)
                ri = r_vec[idx]
                (ri >= R0 || ri < 0.0) && continue
                bin = clamp(Int(round((ri - r_mid[1]) / dr_bin)) + 1, 1, N_bins)
                sum_uz[bin] += u[idx]; sum_rho[bin] += rho[idx]
                sum_T[bin] += T[idx]; sum_w[bin] += 1.0
            end
        end
    end

    valid = sum_w .> 0
    mean_uz = zeros(N_bins); mean_rho = zeros(N_bins); mean_T = zeros(N_bins)
    mean_uz[valid] .= sum_uz[valid] ./ sum_w[valid]
    mean_rho[valid] .= sum_rho[valid] ./ sum_w[valid]
    mean_T[valid] .= sum_T[valid] ./ sum_w[valid]

    # Wall quantities
    wall_bins = findall(valid .& (y_wall .> 0) .& (y_wall .< 0.005*R0))
    if length(wall_bins) < 2
        wall_bins = findall(valid .& (y_wall .> 0) .& (y_wall .< 0.02*R0))
    end
    rho_w = mean_rho[wall_bins[end]]; T_w = mean_T[wall_bins[end]]
    mu_w = sutherland_mu(T_w); nu_w = mu_w / rho_w

    # Iterative u_tau
    near = findall(valid .& (y_wall .> 0) .& (y_wall .< 0.005*R0))
    if length(near) < 2; near = findall(valid .& (y_wall .> 0) .& (y_wall .< 0.02*R0)); end
    yw_fit = y_wall[near]; uz_fit = mean_uz[near]
    dudy = sum(yw_fit .* uz_fit) / sum(yw_fit.^2)
    tau_w = mu_w * abs(dudy); u_tau = sqrt(tau_w / rho_w)
    for _ in 1:5
        yp_lim = 5.0 * nu_w / u_tau
        sub = findall(valid .& (y_wall .> 0) .& (y_wall .< yp_lim))
        length(sub) < 2 && break
        dudy = sum(y_wall[sub] .* mean_uz[sub]) / sum(y_wall[sub].^2)
        tau_w = mu_w * abs(dudy); u_tau_new = sqrt(tau_w / rho_w)
        abs(u_tau_new - u_tau)/u_tau < 1e-4 && (u_tau = u_tau_new; break)
        u_tau = u_tau_new
    end

    y_plus = y_wall .* u_tau ./ nu_w
    u_plus = mean_uz ./ u_tau

    # Van Driest
    mask = valid .& (y_wall .> 0)
    idx_sorted = sort(findall(mask), by=i -> y_wall[i])  # from wall outward
    u_plus_vd = zeros(N_bins)
    for k in 2:length(idx_sorted)
        ic = idx_sorted[k]; ip = idx_sorted[k-1]
        du = u_plus[ic] - u_plus[ip]
        rho_avg = 0.5 * (mean_rho[ic] + mean_rho[ip])
        u_plus_vd[ic] = u_plus_vd[ip] + sqrt(rho_avg / rho_w) * du
    end

    return y_plus, u_plus_vd, valid, u_tau, nu_w, sum_w
end

function main()
    plt_dir = length(ARGS) >= 1 ? ARGS[1] : "PLT/Ma15Ro00"
    mesh_dir = "MESH"; R0 = 0.5

    files = readdir(plt_dir)
    pat = r"^plt-(\d+)-b0\.h5$"
    steps = sort([parse(Int, match(pat, f).captures[1]) for f in files if occursin(pat, f)])
    @printf("Steps: %d files [%d…%d]\n", length(steps), steps[1], steps[end])

    mesh_data = [load_block_mesh(mesh_dir, bid) for bid in 0:4]
    block_r = [vec(sqrt.(yc.^2 .+ zc.^2)) for (xc,yc,zc,Nx,Ny,Nz) in mesh_data]

    bin_counts = [100, 200, 400, 800, 1600, 3200]
    colors = [:mediumpurple, :steelblue, :seagreen, :firebrick, :darkorange, :black]

    cm_to_pt = 72 / 2.54
    update_theme!(fontsize=10, fonts=(; regular="Times New Roman"))
    fig = Figure(size=(round(Int, 18*cm_to_pt), round(Int, 9*cm_to_pt)), figure_padding=(6, 8, 4, 4))

    ax1 = Axis(fig[1,1], xlabel="y⁺", ylabel="u⁺_VD",
               xscale=log10, xminorticksvisible=true, xminorticks=IntervalsBetween(9),
               title="(a) u⁺_VD — bin count sensitivity", titlealign=:left, titlefont=:regular)
    ax2 = Axis(fig[1,2], xlabel="y⁺", ylabel="u⁺_VD - y⁺",
               title="(b) Deviation from viscous sublayer", titlealign=:left, titlefont=:regular)

    for (i, nb) in enumerate(bin_counts)
        @printf("  N_bins = %d ...\n", nb)
        yp, uvd, valid, u_tau, nu_w, sw = compute_profile(plt_dir, steps, mesh_data, block_r, R0, nb)
        mask = valid .& (yp .> 0.3)
        yp_m = yp[mask]; uvd_m = uvd[mask]

        lines!(ax1, yp_m, uvd_m, linewidth=1.5, color=colors[i], label="N=$nb")

        # Deviation from u⁺=y⁺ in sublayer
        sublayer = yp_m .< 10
        if any(sublayer)
            lines!(ax2, yp_m[sublayer], uvd_m[sublayer] .- yp_m[sublayer],
                   linewidth=1.5, color=colors[i], label="N=$nb")
        end
    end

    # Reference lines
    yp_ref = 10 .^ range(log10(0.3), log10(600), length=200)
    lines!(ax1, yp_ref, yp_ref, color=:gray60, linestyle=:dash, linewidth=1, label="u⁺=y⁺")
    lines!(ax1, yp_ref, (1/0.41) .* log.(yp_ref) .+ 5.2, color=:gray30, linestyle=:dashdot, linewidth=1, label="Log law")
    axislegend(ax1, position=:lt, framevisible=true, labelsize=7)
    xlims!(ax1, 0.3, 600); ylims!(ax1, 0, 25)

    hlines!(ax2, [0.0], color=:gray60, linestyle=:dash, linewidth=1)
    axislegend(ax2, position=:lt, framevisible=true, labelsize=7)
    xlims!(ax2, 0, 10); ylims!(ax2, -1, 2)

    png_file = joinpath(plt_dir, "bump_sensitivity.png")
    save(png_file, fig, px_per_unit=4)
    @printf("\n  ✓ Plot saved to: %s\n", png_file)
end

main()
