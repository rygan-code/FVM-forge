# verify.jl — Verification for PipeFlow Benchmark
# Checks:
# 1. No NaN in output
# 2. Positivity of thermodynamic variables (density, pressure)

using HDF5, Statistics, Printf

function verify_pipeflow()
    case_dir = "Benchmark/PIPEFLOW"

    # Find latest PLT file
    plt_dir = "./PLT"
    if !isdir(plt_dir)
        println("✗ No PLT directory found. Run the simulation first.")
        return false
    end

    files = readdir(plt_dir)
    plt_files = filter(f -> occursin(r"^plt_b0-\d+\.h5$", f), files) # Look for block 0
    if isempty(plt_files)
        println("✗ No plt_b0-*.h5 files found in PLT/.")
        return false
    end

    step_nums = [parse(Int, match(r"(\d+)", f).captures[1]) for f in plt_files]
    last_step = maximum(step_nums)
    
    println("═" ^ 60)
    println("  PipeFlow — Verification")
    println("═" ^ 60)
    
    # Needs to check all 5 blocks of the butterfly mesh
    n_nan_total = 0
    rho_min_global = Inf
    p_min_global = Inf
    
    for b in 0:4
        last_file = joinpath(plt_dir, "plt_b$(b)-$last_step.h5")
        if !isfile(last_file)
            println("✗ Missing plotting file for block $b at step $last_step")
            return false
        end
        
        println("  Analyzing: $last_file")

        # ── Load data ──
        fid = h5open(last_file, "r")
        u = read(fid["u"])
        v = read(fid["v"])
        rho = read(fid["rho"])
        p = read(fid["p"])
        T = read(fid["T"])
        close(fid)

        # ── Check 1: No NaN ──
        n_nan_u = count(isnan, u)
        n_nan_rho = count(isnan, rho)
        n_nan_p = count(isnan, p)
        n_nan_T = count(isnan, T)
        n_nan = n_nan_u + n_nan_rho + n_nan_p + n_nan_T
        n_nan_total += n_nan

        if n_nan > 0
            println("    ✗ FAILED: NaN detected in block $b!")
            @printf("      u: %d NaN, rho: %d NaN, p: %d NaN, T: %d NaN\n", n_nan_u, n_nan_rho, n_nan_p, n_nan_T)
        else
            println("    ✓ No NaN in block $b")
        end

        # ── Check 2: Positivity ──
        rho_min = minimum(rho)
        p_min = minimum(p)
        T_min = minimum(T)
        
        rho_min_global = min(rho_min_global, rho_min)
        p_min_global = min(p_min_global, p_min)

        if rho_min <= 0 || p_min <= 0 || T_min <= 0
            println("    ✗ FAILED: Non-positive thermodynamic quantities in block $b!")
            @printf("      min(ρ)=%.2e, min(p)=%.2e, min(T)=%.2e\n", rho_min, p_min, T_min)
        else
            println("    ✓ Positivity preserved in block $b")
        end
    end

    # ── Summary ──
    println()
    println("═" ^ 60)
    passed = n_nan_total == 0 && rho_min_global > 0 && p_min_global > 0
    if passed
        println("  ✓ VERIFICATION PASSED — PipeFlow case stable")
    else
        println("  ✗ VERIFICATION FAILED")
    end
    println("═" ^ 60)

    return passed
end

verify_pipeflow()
