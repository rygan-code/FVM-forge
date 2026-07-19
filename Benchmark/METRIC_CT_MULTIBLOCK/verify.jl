if !isdefined(@__MODULE__, :METRIC_CT_STATS_COLUMNS)
    include(joinpath(
        @__DIR__, "..", "METRIC_CT_WARPED", "metric_ct_diagnostics.jl",
    ))
end
if !isdefined(@__MODULE__, :METRIC_CT_INTERFACE_STATS_COLUMNS)
    include(joinpath(@__DIR__, "interface_diagnostics.jl"))
end

function metric_ct_read_interface_stats(path)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    isempty(lines) && error("metric CT interface stats file is empty")
    strip(lines[1]) == metric_ct_interface_stats_header() || error(
        "metric CT interface stats schema mismatch: got $(strip(lines[1]))",
    )
    length(lines) >= 2 || error("metric CT interface stats contains no data rows")

    data = Matrix{Float64}(
        undef, length(lines) - 1, length(METRIC_CT_INTERFACE_STATS_COLUMNS),
    )
    for (row, line) in enumerate(lines[2:end])
        fields = split(strip(line))
        length(fields) == length(METRIC_CT_INTERFACE_STATS_COLUMNS) || error(
            "metric CT interface stats row $row has $(length(fields)) columns",
        )
        for column in eachindex(fields)
            value = try
                parse(Float64, fields[column])
            catch
                error(
                    "metric CT interface stats row $row column $column is not numeric",
                )
            end
            isfinite(value) || error(
                "metric CT interface stats row $row column $column is non-finite",
            )
            data[row, column] = value
        end
    end
    return data
end

function metric_ct_verify_multiblock(
    metric_path, interface_path;
    min_final_time=0.1,
    max_uniform_core_linf=1.0e-10,
    max_divb_face=1.0e-12,
    max_interface_face_flux_rel=1.0e-12,
    max_interface_line_emf=1.0e-12,
    max_energy_drift=5.0e-3,
    min_timestep_ratio=1.0e-6,
)
    metric_summary = metric_ct_verify_run(
        metric_path;
        test_case=:uniform,
        min_final_time=min_final_time,
        max_divb_face=max_divb_face,
        max_uniform_core_linf=max_uniform_core_linf,
    )
    metric_data = metric_ct_read_stats(metric_path)
    interface_data = metric_ct_read_interface_stats(interface_path)

    size(interface_data, 1) == size(metric_data, 1) || error(
        "metric CT acceptance failed: metric/interface trajectory lengths differ",
    )
    all(view(interface_data, :, 1) .== view(metric_data, :, 2)) || error(
        "metric CT acceptance failed: metric/interface step trajectories differ",
    )
    all(view(interface_data, :, 2) .== view(metric_data, :, 3)) || error(
        "metric CT acceptance failed: metric/interface time trajectories differ",
    )

    steps = view(interface_data, :, 1)
    all(isinteger, steps) && all(steps .> 0) || error(
        "metric CT acceptance failed: interface steps must be positive integers",
    )
    for column in 3:6
        all(view(interface_data, :, column) .>= 0) || error(
            "metric CT acceptance failed: interface residuals must be non-negative",
        )
    end
    flags = view(interface_data, :, 7)
    all(isinteger, flags) || error(
        "metric CT acceptance failed: non-finite rank flags must be integers",
    )
    all(iszero, flags) || error(
        "metric CT acceptance failed: a rank reported non-finite interface data",
    )

    measured_face_rel = maximum(view(interface_data, :, 4))
    measured_face_rel <= max_interface_face_flux_rel || error(
        "metric CT acceptance failed: interface face-flux relative residual " *
        "$measured_face_rel > $max_interface_face_flux_rel",
    )
    measured_line_abs = maximum(view(interface_data, :, 5))
    measured_line_rel = maximum(view(interface_data, :, 6))
    measured_line = max(measured_line_abs, measured_line_rel)
    measured_line <= max_interface_line_emf || error(
        "metric CT acceptance failed: interface line-EMF residual " *
        "$measured_line > $max_interface_line_emf",
    )

    energies = view(metric_data, :, 12)
    initial_energy = first(energies)
    initial_energy > 0 || error(
        "metric CT acceptance failed: initial conservative energy must be positive",
    )
    measured_energy_drift = maximum(abs.(energies .- initial_energy)) /
                            abs(initial_energy)
    measured_energy_drift <= max_energy_drift || error(
        "metric CT acceptance failed: energy drift $measured_energy_drift > " *
        "$max_energy_drift",
    )

    timesteps = view(metric_data, :, 4)
    measured_timestep_ratio = minimum(timesteps) / maximum(timesteps)
    measured_timestep_ratio >= min_timestep_ratio || error(
        "metric CT acceptance failed: timestep ratio $measured_timestep_ratio < " *
        "$min_timestep_ratio",
    )

    return merge(metric_summary, (
        max_interface_face_flux_abs=maximum(view(interface_data, :, 3)),
        max_interface_face_flux_rel=measured_face_rel,
        max_interface_line_emf_abs=measured_line_abs,
        max_interface_line_emf_rel=measured_line_rel,
        nonfinite_rank_flag=Int(maximum(flags)),
        max_energy_drift=measured_energy_drift,
        timestep_ratio=measured_timestep_ratio,
    ))
end

function _metric_ct_multiblock_verify_main(args)
    length(args) == 2 || error(
        "usage: julia verify.jl <metric_stats.dat> <interface_stats.dat>",
    )
    println(metric_ct_verify_multiblock(args...))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    _metric_ct_multiblock_verify_main(ARGS)
end
