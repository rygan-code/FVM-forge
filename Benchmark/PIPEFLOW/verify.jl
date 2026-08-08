using HDF5

@isdefined(benchmark_plt_steps) ||
    include(joinpath(@__DIR__, "..", "benchmark_io.jl"))

function verify_pipeflow(; plt_dir="PLT", block_ids=0:4)
    common_steps = nothing
    for block_id in block_ids
        block_steps = Set(benchmark_plt_steps(plt_dir, block_id))
        common_steps = common_steps === nothing ? block_steps :
            intersect(common_steps, block_steps)
    end
    (common_steps === nothing || isempty(common_steps)) && return false
    step = maximum(common_steps)

    for block_id in block_ids
        path = joinpath(plt_dir, "plt-$step-b$block_id.h5")
        valid = h5open(path, "r") do file
            rho = read(file["rho"])
            pressure = read(file["p"])
            temperature = read(file["T"])
            velocity = (read(file["u"]), read(file["v"]), read(file["w"]))
            return all(isfinite, rho) && all(isfinite, pressure) &&
                all(isfinite, temperature) && all(field -> all(isfinite, field), velocity) &&
                minimum(rho) > 0 && minimum(pressure) > 0 && minimum(temperature) > 0
        end
        valid || return false
    end
    println("PipeFlow verification passed at step $step")
    return true
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    verify_pipeflow() || exit(1)
end
