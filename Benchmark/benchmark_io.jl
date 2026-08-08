function benchmark_plt_steps(directory::AbstractString, block_id::Integer)
    isdir(directory) || return Int[]
    pattern = Regex("^plt-(\\d+)-b" * string(block_id) * "\\.h5\$")
    steps = Int[]
    for filename in readdir(directory)
        matched = match(pattern, filename)
        matched === nothing || push!(steps, parse(Int, matched.captures[1]))
    end
    return sort!(unique!(steps))
end

function latest_benchmark_plt_file(directory::AbstractString, block_id::Integer)
    steps = benchmark_plt_steps(directory, block_id)
    isempty(steps) && return nothing
    step = last(steps)
    return step, joinpath(directory, "plt-$step-b$block_id.h5")
end

function benchmark_output_time(directory::AbstractString, step::Integer)
    xmf_path = joinpath(directory, "plt-$step.xmf")
    isfile(xmf_path) || throw(ArgumentError(
        "missing XDMF metadata for step $step: $xmf_path",
    ))
    matched = match(r"<Time\s+Value=\"([^\"]+)\"\s*/>", read(xmf_path, String))
    matched === nothing && throw(ArgumentError(
        "missing Time value in XDMF metadata: $xmf_path",
    ))
    time = tryparse(Float64, matched.captures[1])
    time === nothing && throw(ArgumentError(
        "invalid Time value in XDMF metadata: $xmf_path",
    ))
    isfinite(time) || throw(ArgumentError(
        "non-finite Time value in XDMF metadata: $xmf_path",
    ))
    return time
end
