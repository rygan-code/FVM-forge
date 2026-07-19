const OT_STATS_COLUMNS = (
    :step, :time, :dt, :E_kin, :E_th, :E_mag, :E_Q, :E_cons,
    :E_Q_minus_E_cons, :min_rho_raw, :min_ei_raw, :min_p_raw,
    :divB_cell_L2, :divB_face_L2,
)

ot_stats_header() = "# " * join(string.(OT_STATS_COLUMNS), " ")

function ot_raw_mhd_minima(Uh, gamma)
    size(Uh, 4) >= 8 || throw(ArgumentError("U must contain at least 8 components"))
    min_rho = Inf
    min_ei = Inf
    min_p = Inf
    for k in axes(Uh, 3), j in axes(Uh, 2), i in axes(Uh, 1)
        raw = mhd_raw_thermo_components(
            Uh[i,j,k,1], Uh[i,j,k,2], Uh[i,j,k,3], Uh[i,j,k,4],
            Uh[i,j,k,5], Uh[i,j,k,6], Uh[i,j,k,7], Uh[i,j,k,8], gamma,
        )
        min_rho = min(min_rho, raw[1])
        min_ei = min(min_ei, raw[4])
        min_p = min(min_p, raw[5])
    end
    return (rho=min_rho, ei=min_ei, p=min_p)
end

function ot_read_stats(path)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    header_index = findfirst(
        line -> startswith(strip(line), "# step "), lines,
    )
    isnothing(header_index) && error("OT stats header is missing")
    names = Symbol.(split(replace(
        strip(lines[header_index]), r"^#\s*" => "",
    )))
    Tuple(names) == OT_STATS_COLUMNS || error(
        "OT stats schema mismatch: got $(Tuple(names))",
    )

    data_lines = filter(
        line -> !startswith(strip(line), "#"),
        lines[(header_index + 1):end],
    )
    isempty(data_lines) && error("OT stats contains no data rows")
    data = Matrix{Float64}(undef, length(data_lines), length(OT_STATS_COLUMNS))
    for (row, line) in enumerate(data_lines)
        values = parse.(Float64, split(strip(line)))
        length(values) == length(OT_STATS_COLUMNS) || error(
            "OT stats row $row has $(length(values)) columns",
        )
        data[row, :] .= values
    end
    return data
end

function ot_verify_stats(
    path;
    min_final_time=1.0,
    max_divb_face=1e-12,
    max_energy_drift=5e-3,
    min_timestep_ratio=1e-6,
)
    data = ot_read_stats(path)
    size(data, 1) >= 2 || error(
        "OT acceptance failed: expected an initial row and at least one post-step row",
    )
    all(isfinite, data) || error("OT acceptance failed: non-finite stats value")
    steps = view(data, :, 1)
    all(isinteger, steps) || error(
        "OT acceptance failed: steps must be exact integers",
    )
    all(steps .== (eachindex(steps) .- 1)) || error(
        "OT acceptance failed: steps must be consecutive from zero",
    )
    times = view(data, :, 2)
    timesteps = view(data, :, 3)
    times[1] == 0.0 || error("OT acceptance failed: initial time is not zero")
    timesteps[1] == 0.0 || error("OT acceptance failed: initial timestep is not zero")
    all(diff(times) .> 0) || error(
        "OT acceptance failed: post-step times are not strictly increasing",
    )
    post_timesteps = view(timesteps, 2:length(timesteps))
    all(post_timesteps .> 0) || error(
        "OT acceptance failed: non-positive post-step timestep",
    )
    for row in 2:length(times)
        observed_dt = times[row] - times[row - 1]
        tolerance = 16 * eps(Float64) * max(
            abs(times[row]), abs(times[row - 1]), abs(timesteps[row]), 1.0,
        )
        abs(observed_dt - timesteps[row]) <= tolerance || error(
            "OT acceptance failed: time increment at step $(steps[row]) does not match dt",
        )
    end
    times[end] >= min_final_time || error(
        "OT acceptance failed: final time $(times[end]) < $min_final_time",
    )
    minimum(view(data, :, 10)) > 0 || error(
        "OT acceptance failed: non-positive raw density",
    )
    minimum(view(data, :, 11)) > 0 || error(
        "OT acceptance failed: non-positive raw internal energy",
    )
    minimum(view(data, :, 12)) > 0 || error(
        "OT acceptance failed: non-positive raw pressure",
    )
    measured_divb = maximum(abs.(view(data, :, 14)))
    measured_divb <= max_divb_face || error(
        "OT acceptance failed: divB_face=$measured_divb > $max_divb_face",
    )
    e_cons = view(data, :, 8)
    e_cons[1] != 0 || error("OT acceptance failed: zero initial energy")
    measured_drift = maximum(abs.(e_cons ./ e_cons[1] .- 1))
    measured_drift <= max_energy_drift || error(
        "OT acceptance failed: energy drift=$measured_drift > $max_energy_drift",
    )
    dt_ratio = minimum(post_timesteps) / maximum(post_timesteps)
    dt_ratio >= min_timestep_ratio || error(
        "OT acceptance failed: timestep ratio=$dt_ratio < $min_timestep_ratio",
    )
    return (
        final_time=times[end],
        min_rho=minimum(view(data, :, 10)),
        min_ei=minimum(view(data, :, 11)),
        min_p=minimum(view(data, :, 12)),
        max_divb_face=measured_divb,
        max_energy_drift=measured_drift,
        timestep_ratio=dt_ratio,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia ot_diagnostics.jl <stats.dat>")
    println(ot_verify_stats(ARGS[1]))
end
