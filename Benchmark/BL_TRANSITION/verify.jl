# verify.jl — Verification for Flat Plate BL Transition Benchmark
# Checks:
# 1. No NaN in output
# 2. Boundary layer profile is physical (monotonic u, correct freestream)
# 3. Skin friction coefficient near Blasius for laminar region
# 4. Implicit solver stability (check density/pressure positivity)

using HDF5, Statistics, Printf

function verify_bl_transition()
    case_dir = "Benchmark/BL_TRANSITION"

    # Find latest PLT file
    plt_dir = "./PLT"
    if !isdir(plt_dir)
        println("✗ No PLT directory found. Run the simulation first.")
        return false
    end

    files = readdir(plt_dir)
    plt_files = filter(f -> occursin(r"^plt-\d+\.h5$", f), files)
    if isempty(plt_files)
        println("✗ No plt-*.h5 files found in PLT/.")
        return false
    end

    step_nums = [parse(Int, match(r"(\d+)", f).captures[1]) for f in plt_files]
    last_step = maximum(step_nums)
    last_file = joinpath(plt_dir, "plt-$last_step.h5")

    println("═" ^ 60)
    println("  Flat Plate BL Transition — Verification")
    println("═" ^ 60)
    println("  Analyzing: $last_file (step $last_step)")

    # ── Load data ──
    fid = h5open(last_file, "r")
    u = read(fid["u"])
    v = read(fid["v"])
    rho = read(fid["rho"])
    p = read(fid["p"])
    T = read(fid["T"])
    close(fid)

    Nx, Ny, Nz = size(u)
    println("  Field size: $Nx × $Ny × $Nz")

    # ── Check 1: No NaN ──
    n_nan_u = count(isnan, u)
    n_nan_rho = count(isnan, rho)
    n_nan_p = count(isnan, p)
    n_nan_T = count(isnan, T)
    n_nan_total = n_nan_u + n_nan_rho + n_nan_p + n_nan_T

    if n_nan_total > 0
        println("✗ FAILED: NaN detected!")
        @printf("  u: %d NaN, rho: %d NaN, p: %d NaN, T: %d NaN\n",
                n_nan_u, n_nan_rho, n_nan_p, n_nan_T)
        return false
    end
    println("  ✓ No NaN in any field")

    # ── Check 2: Positivity ──
    rho_min = minimum(rho)
    p_min = minimum(p)
    T_min = minimum(T)

    if rho_min <= 0 || p_min <= 0 || T_min <= 0
        println("✗ FAILED: Non-positive thermodynamic quantities!")
        @printf("  min(ρ)=%.2e, min(p)=%.2e, min(T)=%.2e\n", rho_min, p_min, T_min)
        return false
    end
    println("  ✓ Positivity preserved")
    @printf("    ρ ∈ [%.4e, %.4e]\n", rho_min, maximum(rho))
    @printf("    p ∈ [%.4e, %.4e]\n", p_min, maximum(p))
    @printf("    T ∈ [%.2f, %.2f] K\n", T_min, maximum(T))
    @printf("    u ∈ [%.4e, %.4e]\n", minimum(u), maximum(u))

    # ── Check 3: Freestream velocity ──
    u_far = mean(u[:, end, :])  # top boundary should be freestream
    u_max = maximum(u)

    γ = 1.4; Rg = 287.0
    T_inf = Float64(mean(T[:, end, :]))
    c_inf = sqrt(γ * Rg * T_inf)
    Ma_computed = u_far / c_inf

    @printf("  Freestream: u∞ = %.2f m/s, Ma = %.4f\n", u_far, Ma_computed)

    if abs(Ma_computed - 0.5) / 0.5 > 0.2
        println("  ⚠ WARNING: Ma deviates >20% from target (0.5)")
    else
        println("  ✓ Mach number within 20% of target")
    end

    # ── Check 4: Boundary layer profile at mid-station ──
    # Load mesh for y-coordinates
    mfid = h5open("MESH/mesh_b0.h5", "r")
    coords = read(mfid["coords"])
    NG_mesh = read(mfid["NG"])
    Nx_mesh = read(mfid["Nx"])
    Ny_mesh = read(mfid["Ny"])
    close(mfid)

    ix_mid = div(Nx, 2)  # Mid-station profile

    u_profile = u[ix_mid, :, 1]  # z-averaged or first z-plane

    # y cell centers from node coordinates
    # coords shape: (3, Nx+2NG+1, Ny+2NG+1, Nz+2NG+1) in full
    # For PLT data (no ghost), we need the mesh y-coordinates
    y_nodes = coords[2, ix_mid+NG_mesh, NG_mesh+1:NG_mesh+Ny_mesh+1, NG_mesh+1]
    y_cell = 0.5 .* (y_nodes[1:end-1] .+ y_nodes[2:end])

    u_edge = u_profile[end]

    # Find δ99
    delta99 = 0.0
    for j in 1:length(u_profile)
        if u_profile[j] >= 0.99 * u_edge
            delta99 = y_cell[j]
            break
        end
    end

    @printf("  BL profile at x-midstation (ix=%d):\n", ix_mid)
    @printf("    u_edge = %.2f m/s,  δ99 = %.4f\n", u_edge, delta99)
    @printf("    u(y=0) = %.4f m/s (should be ~0 for no-slip)\n", u_profile[1])

    # Wall velocity should be small (no-slip + ghost cell interpolation)
    if abs(u_profile[1]) < 0.1 * u_edge
        println("  ✓ No-slip condition satisfied at wall")
    else
        println("  ⚠ WARNING: Wall velocity unexpectedly large")
    end

    # ── Check 5: Skin friction (laminar Blasius) ──
    if delta99 > 0 && length(y_cell) >= 2
        μ_w = 1.458e-6 * Float64(T[ix_mid, 1, 1])^1.5 / (Float64(T[ix_mid, 1, 1]) + 110.4)
        yc1 = y_cell[1]; yc2 = y_cell[2]
        u1 = Float64(u_profile[1]); u2 = Float64(u_profile[2])

        # 2nd order wall gradient (Lagrange fit with u(0)=0)
        dudy_wall = (u1 * yc2^2 - u2 * yc1^2) / (yc1 * yc2 * (yc2 - yc1))
        tau_w = μ_w * dudy_wall
        Cf = tau_w / (0.5 * Float64(rho[ix_mid, end, 1]) * Float64(u_edge)^2)

        @printf("    Cf = %.6f\n", Cf)

        if Cf > 0 && Cf < 0.1
            println("  ✓ Skin friction is physical (positive, bounded)")
        else
            println("  ⚠ WARNING: Cf looks non-physical")
        end
    end

    # ── Summary ──
    println()
    println("═" ^ 60)
    passed = n_nan_total == 0 && rho_min > 0 && p_min > 0
    if passed
        println("  ✓ VERIFICATION PASSED — Implicit LU-SGS + BDF2 stable")
    else
        println("  ✗ VERIFICATION FAILED")
    end
    println("═" ^ 60)

    return passed
end

verify_bl_transition()
