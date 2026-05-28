# =============================================================================
# check_yplus.jl — Near-wall y⁺ distribution per block
#
# Usage:
#   julia Utils/check_yplus.jl <PLT_DIR>  (default: PLT/Ma15Ro00)
# =============================================================================

using HDF5, Statistics, Printf, CairoMakie

const C_s = 1.458e-6
const T_s = 110.4
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

function main()
    plt_dir = length(ARGS) >= 1 ? ARGS[1] : "PLT/Ma15Ro00"
    mesh_dir = "MESH"
    R0 = 0.5

    # Find latest step with b0
    files = readdir(plt_dir)
    pat = r"^plt-(\d+)-b0\.h5$"
    steps = sort([parse(Int, match(pat, f).captures[1]) for f in files if occursin(pat, f)])
    step = steps[end]
    @printf("Using step %d from %s\n", step, plt_dir)

    # Load one snapshot for wall quantities
    println("Loading flow data...")
    wall_rho = Float64[]; wall_T = Float64[]
    wall_ux = Float64[]; wall_yw = Float64[]
    
    mesh_data = [load_block_mesh(mesh_dir, bid) for bid in 0:4]
    
    # Collect ALL near-wall cells for u_τ estimate
    for bid in 0:4
        xc, yc, zc, Nx, Ny, Nz = mesh_data[bid+1]
        fid = h5open(joinpath(plt_dir, "plt-$(step)-b$(bid).h5"), "r")
        rho = Float64.(read(fid["rho"])); u = Float64.(read(fid["u"]))
        T = Float64.(read(fid["T"]))
        close(fid)
        
        r = sqrt.(yc.^2 .+ zc.^2)
        yw = R0 .- r
        
        for idx in eachindex(r)
            if yw[idx] > 0 && yw[idx] < 0.005 * R0
                push!(wall_rho, rho[idx]); push!(wall_T, T[idx])
                push!(wall_ux, u[idx]); push!(wall_yw, yw[idx])
            end
        end
    end
    
    rho_w = mean(wall_rho); T_w = mean(wall_T)
    mu_w = sutherland_mu(T_w); nu_w = mu_w / rho_w
    dudy = sum(wall_yw .* wall_ux) / sum(wall_yw.^2)
    tau_w = mu_w * abs(dudy); u_tau = sqrt(tau_w / rho_w)
    @printf("u_τ = %.4f m/s, ν_w = %.4e m²/s\n", u_tau, nu_w)
    
    # Per-block near-wall analysis
    println("\n" * "="^70)
    println("  Per-Block Near-Wall y⁺ Distribution")
    println("="^70)
    
    block_yplus_min = zeros(5)
    block_yplus_all = [Float64[] for _ in 1:5]  # all y⁺ values per block
    block_yplus_wall = [Float64[] for _ in 1:5]  # y⁺ of cells closest to wall per block
    block_theta_wall = [Float64[] for _ in 1:5]  # θ of wall cells
    
    for bid in 0:4
        xc, yc, zc, Nx, Ny, Nz = mesh_data[bid+1]
        r = sqrt.(yc.^2 .+ zc.^2)
        yw = R0 .- r
        yp = yw .* u_tau ./ nu_w
        θ = atan.(zc, yc)
        
        # Find the minimum y⁺ per (i, θ) — i.e., the first cell from the wall
        # For each streamwise station (i), find cells closest to wall
        for i in 1:Nx
            for j in 1:Ny
                for k in 1:Nz
                    if yw[i,j,k] > 0 && r[i,j,k] < R0
                        push!(block_yplus_all[bid+1], yp[i,j,k])
                    end
                end
            end
            
            # Find minimum y⁺ across all j,k for this i (first wall cell)
            min_yp = Inf
            min_theta = 0.0
            for j in 1:Ny, k in 1:Nz
                if yw[i,j,k] > 0 && yp[i,j,k] < min_yp && r[i,j,k] < R0
                    min_yp = yp[i,j,k]
                    min_theta = θ[i,j,k]
                end
            end
            if min_yp < Inf
                push!(block_yplus_wall[bid+1], min_yp)
                push!(block_theta_wall[bid+1], min_theta)
            end
        end
        
        block_yplus_min[bid+1] = isempty(block_yplus_wall[bid+1]) ? Inf : minimum(block_yplus_wall[bid+1])
        
        @printf("  Block %d: y⁺_min = %6.2f, y⁺_mean(wall) = %6.2f, N_wall_cells = %d\n",
                bid, block_yplus_min[bid+1],
                isempty(block_yplus_wall[bid+1]) ? NaN : mean(block_yplus_wall[bid+1]),
                length(block_yplus_wall[bid+1]))
    end
    
    # ── Plot ──
    cm_to_pt = 72 / 2.54
    update_theme!(fontsize=10, fonts=(; regular="Times New Roman"))
    
    fig = Figure(size=(round(Int, 18*cm_to_pt), round(Int, 14*cm_to_pt)), figure_padding=(6, 8, 4, 4))
    
    # (a) Histogram of first-wall-cell y⁺ per block
    ax1 = Axis(fig[1,1], xlabel="y⁺ (first wall cell)", ylabel="Count",
               title="(a) First wall cell y⁺ per block", titlealign=:left, titlefont=:regular)
    block_colors = [:firebrick, :steelblue, :seagreen, :darkorange, :mediumpurple]
    block_labels = ["Block 0", "Block 1", "Block 2", "Block 3", "Block 4"]
    for bid in 1:5
        if !isempty(block_yplus_wall[bid])
            hist!(ax1, block_yplus_wall[bid], bins=30, color=(block_colors[bid], 0.5),
                  strokewidth=1, strokecolor=block_colors[bid], label=block_labels[bid])
        end
    end
    axislegend(ax1, position=:rt, framevisible=true, labelsize=8)
    
    # (b) y⁺ of all cells vs r/R — shows radial distribution
    ax2 = Axis(fig[1,2], xlabel="r/R", ylabel="y⁺",
               title="(b) y⁺ distribution vs r/R", titlealign=:left, titlefont=:regular,
               yscale=log10)
    for bid in 1:5
        xc, yc, zc, Nx, Ny, Nz = mesh_data[bid]
        r = vec(sqrt.(yc.^2 .+ zc.^2))
        yw = R0 .- r
        yp = yw .* u_tau ./ nu_w
        
        # Subsample for plotting (every 50th cell)
        mask = (r .< R0) .& (yw .> 0)
        idx = findall(mask)
        stride = max(1, length(idx) ÷ 2000)
        sub = idx[1:stride:end]
        
        scatter!(ax2, r[sub]./R0, yp[sub], markersize=1.5, color=(block_colors[bid], 0.3))
    end
    hlines!(ax2, [1.0], color=:black, linestyle=:dash, linewidth=1, label="y⁺=1")
    xlims!(ax2, 0, 1.05)
    
    # (c) Polar plot: y⁺_min as function of θ
    ax3 = Axis(fig[2,1], xlabel="θ (rad)", ylabel="y⁺ (first wall cell)",
               title="(c) Wall cell y⁺ vs azimuthal angle", titlealign=:left, titlefont=:regular)
    for bid in 1:5
        if !isempty(block_theta_wall[bid])
            scatter!(ax3, block_theta_wall[bid], block_yplus_wall[bid],
                     markersize=3, color=(block_colors[bid], 0.6), label=block_labels[bid])
        end
    end
    hlines!(ax3, [1.0], color=:black, linestyle=:dash, linewidth=1)
    axislegend(ax3, position=:rt, framevisible=true, labelsize=7)
    
    # (d) Histogram of ALL cells' y⁺ in the near-wall region (y⁺ < 20)
    ax4 = Axis(fig[2,2], xlabel="y⁺", ylabel="Count",
               title="(d) Cell count in viscous sublayer", titlealign=:left, titlefont=:regular)
    for bid in 1:5
        near = filter(y -> y < 20, block_yplus_all[bid])
        if !isempty(near)
            hist!(ax4, near, bins=40, color=(block_colors[bid], 0.4),
                  strokewidth=0.5, strokecolor=block_colors[bid], label=block_labels[bid])
        end
    end
    axislegend(ax4, position=:rt, framevisible=true, labelsize=7)
    
    png_file = joinpath(plt_dir, "yplus_distribution.png")
    save(png_file, fig, px_per_unit=4)
    @printf("\n  ✓ Plot saved to: %s\n", png_file)
end

main()
