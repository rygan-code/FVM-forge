# Benchmark/TGV/verify.jl
using HDF5, Statistics, Printf

function verify_tgv()
    case_dir = "Benchmark/TGV"
    
    # Get all available PLT files
    plt_files = filter(f -> occursin(r"^plt-\d+\.h5$", f), readdir("./PLT"))
    if length(plt_files) < 2
        println("Need at least 2 PLT files in ./PLT for decay analysis.")
        return
    end
    
    # Sort files numerically
    sorted_files = sort(plt_files, by = f -> parse(Int, match(r"plt-(\d+)\.h5", f).captures[1]))
    
    times = Float64[]
    kes = Float64[]
    
    println("TGV Kinetic Energy Decay Analysis:")
    println("----------------------------------")
    
    for fname in [sorted_files[1], sorted_files[max(1, length(sorted_files) ÷ 2)], sorted_files[end]]
        path = joinpath("./PLT", fname)
        if !isfile(path); continue; end
        fid = h5open(path, "r")
        u = read(fid["u"]); v = read(fid["v"]); w = read(fid["w"])
        
        t = 0.0
        try
            # Read 'time' attribute - handle as scalar
            t_val = read(attrs(fid)["time"])
            t = (t_val isa AbstractArray) ? t_val[1] : t_val
        catch
            # Fallback to step-based estimate if needed, but alert user
            step = parse(Int, match(r"plt-(\d+)\.h5", fname).captures[1])
            t = step * 0.0103 # Based on solver output observation
            @printf("Warning: Could not read 'time' for %s, using estimated %.4f\n", fname, t)
        end
        close(fid)
        
        ke = 0.5 * mean(u.^2 + v.^2 + w.^2)
        push!(times, t)
        push!(kes, ke)
        @printf("Time: %.4f, KE: %.6e, ln(KE): %.4f\n", t, ke, log(ke))
    end
    
    if length(times) < 2
        println("✗ Error: Could not extract enough data points.")
        return
    end

    dt_total = times[end] - times[1]
    if dt_total > 0
        decay_rate = (log(kes[1]) - log(kes[end])) / dt_total
        @printf("\nEstimated Decay Rate: %.4f\n", decay_rate)
        
        if length(times) >= 3
            expected_log_ke_mid = log(kes[1]) - decay_rate * (times[2] - times[1])
            error_log = abs(log(kes[2]) - expected_log_ke_mid)
            @printf("Log-Linearity Error at mid-point: %.4f\n", error_log)
            
            if decay_rate > 0 && error_log < 0.2
                println("✓ Verification PASSED: Kinetic energy follows exponential decay.")
            else
                println("✗ Verification FAILED: Decay is non-exponential or energy is not decaying!")
            end
        else
             if kes[end] < kes[1]
                println("✓ Verification PASSED: Dissipation is leading to energy decay.")
             else
                println("✗ Verification FAILED: Energy is not decaying!")
             end
        end
    else
        println("✗ Error: Insufficient time interval for decay analysis.")
    end
end

verify_tgv()
