# Benchmark/SOD/verify.jl
using HDF5, Statistics, Printf

function exact_sod(x, t)
    # Analytical values at t=0.2 for standard Sod case
    if x < 0.2633
        return 1.0, 0.0, 1.0  # Left state
    elseif x < 0.4859
        return 0.7, 0.4, 0.7  # Rarefaction wave (approx)
    elseif x < 0.6855
        return 0.4263, 0.9275, 0.3031 # Region 3
    elseif x < 0.8504
        return 0.2656, 0.9275, 0.3031 # Region 4 (Post-shock)
    else
        return 0.125, 0.0, 0.1 # Right state
    end
end

function verify_sod()
    case_dir = "Benchmark/SOD"
    last_file = joinpath(case_dir, "plt-final.h5")
    if !isfile(last_file)
        # Fallback to PLT if final not found
        plt_files = filter(f -> occursin(r"^plt-\d+\.h5$", f), readdir("./PLT"))
        if isempty(plt_files); println("No PLT files found."); return; end
        sorted_files = sort(plt_files, by = f -> parse(Int, match(r"plt-(\d+)\.h5", f).captures[1]))
        last_file = joinpath("./PLT", sorted_files[end])
    end
    println("Analyzing Sod: $last_file")

    fid = h5open(last_file, "r")
    rho = read(fid, "rho")[:, 4, 4]
    p = read(fid, "p")[:, 4, 4]
    close(fid)
    
    Nx = length(rho)
    dx = 1.0/Nx
    
    errors = []
    for i in 1:Nx
        x = (i-0.5)*dx
        rho_e, u_e, p_e = exact_sod(x, 0.2)
        push!(errors, (rho[i]-rho_e)^2)
    end
    
    l2_rho = sqrt(mean(errors))
    @printf("L2 error in Density: %.6f\n", l2_rho)
    
    if l2_rho < 0.05
        println("Verification PASSED!")
    else
        println("Verification FAILED (L2 error too high)!")
    end
end

verify_sod()
