using HDF5
using Printf
using Statistics

@isdefined(benchmark_plt_steps) ||
    include(joinpath(@__DIR__, "..", "benchmark_io.jl"))

function exact_sod_density(x)
    x < 0.2633 && return 1.0
    x < 0.4859 && return 0.7
    x < 0.6855 && return 0.4263
    x < 0.8504 && return 0.2656
    return 0.125
end

function verify_sod(; plt_dir="PLT")
    latest = latest_benchmark_plt_file(plt_dir, 0)
    latest === nothing && return false
    _, path = latest
    density = h5open(path, "r") do file
        read(file["rho"])
    end
    all(isfinite, density) && minimum(density) > 0 || return false
    j = cld(size(density, 2), 2)
    k = cld(size(density, 3), 2)
    profile = @view density[:, j, k]
    nx = length(profile)
    l2_error = sqrt(mean((profile[i] - exact_sod_density((i - 0.5) / nx))^2
                         for i in eachindex(profile)))
    @printf("Sod density L2 error: %.6f\n", l2_error)
    return l2_error < 0.05
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    verify_sod() || exit(1)
end
