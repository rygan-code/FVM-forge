using HDF5, Statistics

function get_ke(filename)
    h5open(filename, "r") do file
        rho = read(file, "rho")
        u = read(file, "u")
        v = read(file, "v")
        w = read(file, "w")
        # KE = 0.5 * sum(rho * (u^2 + v^2 + w^2)) * dV
        # Since mesh is uniform, sum(...) is proportional
        ke = 0.5 * sum(rho .* (u.^2 .+ v.^2 .+ w.^2))
        return ke
    end
end

println("Analyzing TGV Decay...")
steps = 200:200:2000
ke_list = Float64[]
times = Float64[]

dt = 5e-4
nu = 0.01

for s in steps
    fname = "PLT/plt-$s.h5"
    if isfile(fname)
        ke = get_ke(fname)
        push!(ke_list, ke)
        push!(times, s * dt)
        println("Step $s, Time $(s*dt): KE=$ke")
    end
end

if length(ke_list) > 1
    # Check decay rate relative to first saved point (t=0.1)
    t0 = times[1]
    ke0 = ke_list[1]
    
    println("\nDecay Comparison (relative to t=$t0):")
    for i in 2:length(ke_list)
        t = times[i]
        ke = ke_list[i]
        dt_rel = t - t0
        numerical_ratio = ke / ke0
        analytical_ratio = exp(-4.0 * nu * dt_rel)
        error = (numerical_ratio - analytical_ratio) / analytical_ratio
        println("t=$t: Num Ratio=$(round(numerical_ratio, digits=6)), Ana Ratio=$(round(analytical_ratio, digits=6)), Error=$(round(error*100, digits=4))%")
    end
end
