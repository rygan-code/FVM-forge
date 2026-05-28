# Benchmark/OBL/verify.jl
using HDF5, Statistics, Printf

function verify_oblique()
    case_dir = "Benchmark/OBL"
    last_file = joinpath(case_dir, "plt-final.h5")
    if !isfile(last_file)
        plt_files = filter(f -> occursin(r"^plt-\d+\.h5$", f), readdir("./PLT"))
        if isempty(plt_files); println("No PLT files found."); return; end
        sorted_files = sort(plt_files, by = f -> parse(Int, match(r"plt-(\d+)\.h5", f).captures[1]))
        last_file = joinpath("./PLT", sorted_files[end])
    end
    println("Analyzing Oblique Shock: $last_file")

    fid = h5open(last_file, "r")
    p = read(fid, "p")[:, :, 4]
    close(fid)
    
    # Inflow (Ma=2.0, p=1.0)
    # Wedge angle ~ 10 deg (based on mesh gen)
    # Theoretical pressure ratio p2/p1 for Ma=2, beta=10 deg? 
    # Actually, the case was set up for beta=29.3 deg shock.
    # Theoretical p2/p1 approx 1.7
    
    # Analysis at X=2.0 (end of domain)
    # The wedge starts at X=0.2, angle=15.0 deg.
    # At X=2.0, ramp is at Y = (2.0-0.2)*tan(15) = 1.8 * 0.268 = 0.482
    # Shock (beta=45.3) is at Y = 1.8 * tan(45.3) = 1.8 * 1.01 = 1.81 (Exited)
    # So Y in [0.482, 1.2] is downstream.
    # We sample at Y ~ 0.8 (index ~ 40/65)
    
    p1 = mean(p[1:5, 10:50, :])
    p2 = mean(p[end-5:end, 40:60, :])
    ratio = p2/p1
    
    # Theoretical for Ma=2.0, theta=15.0: p2/p1 = 2.21
    target = 2.21
    
    println("Oblique Shock Pressure Jump (15-deg wedge):")
    @printf("Upstream p1:   %.4f\n", p1)
    @printf("Downstream p2: %.4f\n", p2)
    @printf("Pressure Ratio: %.4f (Analytical ~ %.2f)\n", ratio, target)
    
    if abs(ratio - target) < 0.25
        println("✓ Verification PASSED!")
    else
        println("✗ Verification FAILED (ratio differs significantly from target)")
    end
end

verify_oblique()
