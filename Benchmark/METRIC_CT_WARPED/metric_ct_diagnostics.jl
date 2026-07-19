const METRIC_CT_STATS_COLUMNS = (
    :resolution, :step, :time, :dt,
    :l1_transverse, :l2_transverse, :uniform_core_linf,
    :min_rho_raw, :min_ei_raw, :min_p_raw,
    :divB_face_L2, :E_cons,
)

metric_ct_stats_header() = "# " * join(string.(METRIC_CT_STATS_COLUMNS), " ")

function metric_ct_read_stats(path)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    isempty(lines) && error("metric CT stats file is empty")
    strip(lines[1]) == metric_ct_stats_header() || error(
        "metric CT stats schema mismatch: got $(strip(lines[1]))",
    )
    length(lines) >= 2 || error("metric CT stats contains no data rows")

    data = Matrix{Float64}(
        undef, length(lines) - 1, length(METRIC_CT_STATS_COLUMNS),
    )
    for (row, line) in enumerate(lines[2:end])
        fields = split(strip(line))
        length(fields) == length(METRIC_CT_STATS_COLUMNS) || error(
            "metric CT stats row $row has $(length(fields)) columns",
        )
        for column in eachindex(fields)
            value = try
                parse(Float64, fields[column])
            catch
                error("metric CT stats row $row column $column is not numeric")
            end
            isfinite(value) || error(
                "metric CT stats row $row column $column is non-finite",
            )
            data[row, column] = value
        end
    end
    return data
end

function _metric_ct_test_case(value)
    normalized = lowercase(string(value))
    normalized in ("uniform", "metricctuniform") && return :uniform
    normalized in ("alfven", "metricctalfven") && return :alfven
    throw(ArgumentError("metric CT test_case must be uniform or alfven"))
end

function metric_ct_verify_run(
    path;
    test_case=:alfven,
    min_final_time=0.1,
    max_divb_face=1e-12,
    max_uniform_core_linf=1e-10,
)
    data = metric_ct_read_stats(path)
    case_kind = _metric_ct_test_case(test_case)

    resolutions = view(data, :, 1)
    steps = view(data, :, 2)
    all(isinteger, resolutions) || error(
        "metric CT acceptance failed: resolutions must be exact integers",
    )
    all(resolutions .> 0) || error(
        "metric CT acceptance failed: resolutions must be positive",
    )
    all(resolutions .== resolutions[1]) || error(
        "metric CT acceptance failed: resolution changed within one run",
    )
    all(isinteger, steps) || error(
        "metric CT acceptance failed: steps must be exact integers",
    )
    all(view(data, :, 4) .> 0) || error(
        "metric CT acceptance failed: timestep must be positive",
    )
    if size(data, 1) > 1
        all(diff(steps) .> 0) || error(
            "metric CT acceptance failed: steps are not strictly increasing",
        )
        all(diff(view(data, :, 3)) .> 0) || error(
            "metric CT acceptance failed: times are not strictly increasing",
        )
    end

    for row in axes(data, 1)
        data[row, 8] > 0 || error(
            "metric CT acceptance failed at row $row: non-positive raw density",
        )
        data[row, 9] > 0 || error(
            "metric CT acceptance failed at row $row: non-positive raw internal energy",
        )
        data[row, 10] > 0 || error(
            "metric CT acceptance failed at row $row: non-positive raw pressure",
        )
        measured_divb = abs(data[row, 11])
        measured_divb <= max_divb_face || error(
            "metric CT acceptance failed at row $row: divB_face=$measured_divb > $max_divb_face",
        )
    end

    final = view(data, size(data, 1), :)
    final[3] >= min_final_time || error(
        "metric CT acceptance failed: final time $(final[3]) < $min_final_time",
    )
    if case_kind == :uniform
        final[7] <= max_uniform_core_linf || error(
            "metric CT acceptance failed: uniform core error $(final[7]) > $max_uniform_core_linf",
        )
    else
        final[5] > 0 || error(
            "metric CT acceptance failed: non-positive Alfven L1 error",
        )
        final[6] > 0 || error(
            "metric CT acceptance failed: non-positive Alfven L2 error",
        )
    end

    return (
        resolution=Int(final[1]),
        step=Int(final[2]),
        time=final[3],
        dt=final[4],
        l1_transverse=final[5],
        l2_transverse=final[6],
        uniform_core_linf=final[7],
        min_rho_raw=minimum(view(data, :, 8)),
        min_ei_raw=minimum(view(data, :, 9)),
        min_p_raw=minimum(view(data, :, 10)),
        max_divB_face_L2=maximum(abs, view(data, :, 11)),
        E_cons=final[12],
    )
end

function metric_ct_verify_convergence(paths; min_order=1.7)
    length(paths) == 3 || error(
        "metric CT convergence requires exactly three stats paths",
    )
    runs = map(paths) do path
        metric_ct_verify_run(path; test_case=:alfven)
    end
    sort!(runs; by=run -> run.resolution)
    resolutions = Tuple(run.resolution for run in runs)
    resolutions == (8, 16, 32) || error(
        "metric CT convergence requires resolutions (8, 16, 32), got $resolutions",
    )
    l1_errors = Tuple(run.l1_transverse for run in runs)
    l2_errors = Tuple(run.l2_transverse for run in runs)
    all(diff(collect(l1_errors)) .< 0) || error(
        "metric CT convergence failed: L1 errors do not strictly decrease",
    )
    all(diff(collect(l2_errors)) .< 0) || error(
        "metric CT convergence failed: L2 errors do not strictly decrease",
    )

    observed_order(errors, pair) = log(errors[pair] / errors[pair + 1]) /
        log(resolutions[pair + 1] / resolutions[pair])
    l1_orders = ntuple(pair -> observed_order(l1_errors, pair), 2)
    l2_orders = ntuple(pair -> observed_order(l2_errors, pair), 2)
    l1_orders[2] >= min_order || error(
        "metric CT convergence failed: finest-pair L1 order $(l1_orders[2]) < $min_order",
    )
    return (
        resolutions=resolutions,
        l1_errors=l1_errors,
        l2_errors=l2_errors,
        l1_orders=l1_orders,
        l2_orders=l2_orders,
        finest_l1_order=l1_orders[2],
    )
end

function metric_ct_error_norms(blocks, state_time)
    l1_weighted = 0.0
    l2_weighted = 0.0
    physical_volume = 0.0
    for (_, block) in blocks
        lo = NG + 1
        q = Array(@view block.Q[
            lo:NG+block.Nx, lo:NG+block.Ny, lo:NG+block.Nz, :,
        ])
        inverse_volume = Array(@view block.Vol[
            lo:NG+block.Nx, lo:NG+block.Ny, lo:NG+block.Nz,
        ])
        x = Array(block.x)
        y = Array(block.y)
        z = Array(block.z)
        for k in 1:block.Nz, j in 1:block.Ny, i in 1:block.Nx
            center = metric_ct_cell_center(
                x, y, z, lo + i - 1, lo + j - 1, lo + k - 1,
            )
            T = eltype(q)
            exact = metric_ct_alfven_primitive(
                T(center[1]), T(state_time), T(Rg), T(1e-3),
            )
            error_squared =
                abs2(Float64(q[i,j,k,3] - exact[3])) +
                abs2(Float64(q[i,j,k,4] - exact[4])) +
                abs2(Float64(q[i,j,k,8] - exact[8])) +
                abs2(Float64(q[i,j,k,9] - exact[9]))
            cell_volume = 1.0 / Float64(inverse_volume[i,j,k])
            isfinite(error_squared) || error(
                "metric CT Alfven error is non-finite",
            )
            isfinite(cell_volume) && cell_volume > 0 || error(
                "metric CT cell has invalid physical volume",
            )
            l1_weighted += sqrt(error_squared) * cell_volume
            l2_weighted += error_squared * cell_volume
            physical_volume += cell_volume
        end
    end

    if isdefined(@__MODULE__, :MPI)
        l1_weighted = MPI.Allreduce(l1_weighted, MPI.SUM, MPI.COMM_WORLD)
        l2_weighted = MPI.Allreduce(l2_weighted, MPI.SUM, MPI.COMM_WORLD)
        physical_volume = MPI.Allreduce(physical_volume, MPI.SUM, MPI.COMM_WORLD)
    end
    physical_volume > 0 || error("metric CT domain has zero physical volume")
    return (
        l1_transverse=l1_weighted / physical_volume,
        l2_transverse=sqrt(l2_weighted / physical_volume),
        physical_volume=physical_volume,
    )
end

function _metric_ct_diagnostics_main(args)
    length(args) in (1, 2) || error(
        "usage: julia metric_ct_diagnostics.jl <stats.dat> [uniform|alfven]",
    )
    test_case = length(args) == 1 ? :uniform : _metric_ct_test_case(args[2])
    println(metric_ct_verify_run(args[1]; test_case=test_case))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    _metric_ct_diagnostics_main(ARGS)
end
