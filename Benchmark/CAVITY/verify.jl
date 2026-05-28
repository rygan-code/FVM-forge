# Benchmark/CAVITY/verify.jl — Verify cavity solution against Ghia et al. (1982)
# Ref: Ghia, Ghia & Shin, "High-Re Solutions for Incompressible Flow Using
#      the Navier-Stokes Equations and a Multigrid Method", J. Comput. Phys. 48, 387-411
using HDF5, Statistics, Printf

# ═══════════════════════════════════════════════════
# Ghia et al. (1982) Re=400 benchmark data
# ═══════════════════════════════════════════════════

# u-velocity along vertical centerline (x=0.5)
# Format: (y, u)
const GHIA_U_RE400 = [
    (1.0000,  1.00000),
    (0.9766,  0.75837),
    (0.9688,  0.68439),
    (0.9609,  0.61756),
    (0.9531,  0.55892),
    (0.8516,  0.29093),
    (0.7344,  0.16256),
    (0.6172,  0.02135),
    (0.5000, -0.11477),
    (0.4531, -0.17119),
    (0.2813, -0.32726),
    (0.1719, -0.24299),
    (0.1016, -0.14612),
    (0.0703, -0.10338),
    (0.0625, -0.09266),
    (0.0547, -0.08186),
    (0.0000,  0.00000),
]

# v-velocity along horizontal centerline (y=0.5)
# Format: (x, v)
const GHIA_V_RE400 = [
    (1.0000,  0.00000),
    (0.9688,  0.05454),
    (0.9609,  0.05280),
    (0.9531,  0.04803),
    (0.9453,  0.04091),
    (0.8594, -0.07886),
    (0.8047, -0.12146),
    (0.5000, -0.21388),
    (0.2344,  0.04439),
    (0.2266,  0.05188),
    (0.1563,  0.16914),
    (0.0938,  0.22474),
    (0.0781,  0.22269),
    (0.0703,  0.21476),
    (0.0625,  0.20103),
    (0.0547,  0.18198),
    (0.0000,  0.00000),
]

function verify_cavity()
    # Find the latest PLT file for block 0
    plt_dir = "./PLT"
    if !isdir(plt_dir)
        println("No PLT directory found. Run the cavity simulation first.")
        return
    end
    
    plt_files = filter(f -> occursin(r"plt-\d+-b0\.h5$", f), readdir(plt_dir))
    if isempty(plt_files)
        println("No PLT files found in $plt_dir")
        return
    end
    
    # Filter to only AC cavity files (have "p" as first variable, no "rho")
    ac_files = String[]
    for f in plt_files
        fpath = joinpath(plt_dir, f)
        try
            h5open(fpath, "r") do fid
                names = keys(fid)
                if "p" in names && !("rho" in names)
                    push!(ac_files, f)
                end
            end
        catch
            continue
        end
    end
    
    if isempty(ac_files)
        println("No AC cavity PLT files found in $plt_dir")
        return
    end
    
    sorted = sort(ac_files, by=f -> parse(Int, match(r"plt-(\d+)-b0\.h5", f).captures[1]))
    last_file = joinpath(plt_dir, sorted[end])
    step = parse(Int, match(r"plt-(\d+)-b0\.h5", sorted[end]).captures[1])
    println("═" ^ 60)
    println("  Cavity Verification: Ghia et al. (1982) Re=400")
    println("═" ^ 60)
    println("  Data file: $last_file (step $step)")
    
    fid = h5open(last_file, "r")
    u_data = read(fid, "u")   # (Nx, Ny, Nz)
    v_data = read(fid, "v")
    close(fid)
    
    Nx, Ny, Nz = size(u_data)
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    
    # ── u along vertical centerline (x = 0.5) ──
    # Average over z for quasi-2D
    ix_center = div(Nx, 2)  # cell center closest to x=0.5
    u_centerline = mean(u_data[ix_center, :, :], dims=2)[:, 1]
    
    println("\n  u-velocity along vertical centerline (x=0.5):")
    println("  " * "-" ^ 50)
    @printf("  %-10s  %-12s  %-12s  %-12s\n", "y", "u_sim", "u_Ghia", "error")
    
    u_errors = Float64[]
    for (y_ghia, u_ghia) in GHIA_U_RE400
        # Interpolate simulation data to Ghia y-location
        jf = y_ghia / dy + 0.5  # fractional cell index (cell-centered)
        j = clamp(Int(floor(jf)), 1, Ny)
        j2 = min(j + 1, Ny)
        frac = jf - floor(jf)
        u_sim = u_centerline[j] * (1.0 - frac) + u_centerline[j2] * frac
        
        err = abs(u_sim - u_ghia)
        push!(u_errors, err)
        @printf("  %-10.4f  %-12.6f  %-12.6f  %-12.6f\n", y_ghia, u_sim, u_ghia, err)
    end
    
    # ── v along horizontal centerline (y = 0.5) ──
    jy_center = div(Ny, 2)
    v_centerline = mean(v_data[:, jy_center, :], dims=2)[:, 1]
    
    println("\n  v-velocity along horizontal centerline (y=0.5):")
    println("  " * "-" ^ 50)
    @printf("  %-10s  %-12s  %-12s  %-12s\n", "x", "v_sim", "v_Ghia", "error")
    
    v_errors = Float64[]
    for (x_ghia, v_ghia) in GHIA_V_RE400
        iff = x_ghia / dx + 0.5
        i = clamp(Int(floor(iff)), 1, Nx)
        i2 = min(i + 1, Nx)
        frac = iff - floor(iff)
        v_sim = v_centerline[i] * (1.0 - frac) + v_centerline[i2] * frac
        
        err = abs(v_sim - v_ghia)
        push!(v_errors, err)
        @printf("  %-10.4f  %-12.6f  %-12.6f  %-12.6f\n", x_ghia, v_sim, v_ghia, err)
    end
    
    # ── Summary ──
    u_l2 = sqrt(mean(u_errors .^ 2))
    v_l2 = sqrt(mean(v_errors .^ 2))
    u_max = maximum(u_errors)
    v_max = maximum(v_errors)
    
    println("\n" * "═" ^ 60)
    @printf("  u-velocity:  L2 = %.6f,  L∞ = %.6f\n", u_l2, u_max)
    @printf("  v-velocity:  L2 = %.6f,  L∞ = %.6f\n", v_l2, v_max)
    println("═" ^ 60)
    
    if u_l2 < 0.05 && v_l2 < 0.05
        println("  ✓ Verification PASSED (L2 < 0.05)")
    else
        println("  ✗ Verification FAILED (L2 too high — may need more iterations)")
    end
    println("═" ^ 60)
end

verify_cavity()
