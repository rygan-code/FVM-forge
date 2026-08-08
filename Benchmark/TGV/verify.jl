using HDF5
using Printf
using Statistics

@isdefined(benchmark_plt_steps) ||
    include(joinpath(@__DIR__, "..", "benchmark_io.jl"))

function verify_tgv(; plt_dir="PLT")
    steps = benchmark_plt_steps(plt_dir, 0)
    length(steps) >= 2 || return false
    selected = unique((first(steps), steps[cld(length(steps), 2)], last(steps)))
    times = Float64[]
    energies = Float64[]

    for step in selected
        path = joinpath(plt_dir, "plt-$step-b0.h5")
        density, u, v, w = h5open(path, "r") do file
            read(file["rho"]), read(file["u"]),
            read(file["v"]), read(file["w"])
        end
        all(field -> all(isfinite, field), (density, u, v, w)) || return false
        minimum(density) > 0 || return false
        push!(times, benchmark_output_time(plt_dir, step))
        push!(energies, 0.5 * mean(density .* (u.^2 + v.^2 + w.^2)))
    end

    all(diff(times) .> 0) || return false
    all(energy -> isfinite(energy) && energy > 0, energies) || return false
    decay_rate = (log(first(energies)) - log(last(energies))) /
        (last(times) - first(times))
    @printf("TGV kinetic-energy decay rate: %.6e\n", decay_rate)
    if length(times) == 3
        expected_mid = log(first(energies)) -
            decay_rate * (times[2] - first(times))
        return decay_rate > 0 && abs(log(energies[2]) - expected_mid) < 0.2
    end
    return decay_rate > 0
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    verify_tgv() || exit(1)
end
