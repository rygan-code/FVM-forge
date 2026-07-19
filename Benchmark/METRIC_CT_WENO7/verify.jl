using Printf

const CT_WENO7_STATS_COLUMNS = (
    "resolution",
    "l1_edge",
    "l2_edge",
    "linf_edge",
    "constant_error",
    "nonfinite_flag",
)
const CT_WENO7_MAX_CONSTANT_ERROR = 1.0e-12
const CT_WENO7_MULTIBLOCK_COLUMNS = (
    "resolution", "pre_sync_abs", "pre_sync_rel",
    "post_sync_abs", "post_sync_rel", "interface_l1", "interface_l2",
    "interface_linf", "layer1_max", "layer2_max", "layer3_max",
    "layer4_max", "face_divb", "nonfinite_flag",
)
const CT_WENO7_INTERFACE_INVARIANT_TOL = 1.0e-12
const CT_WENO7_PRE_SYNC_ZERO_TOL = 1.0e-14

function ct_weno7_read_stats(path)
    isfile(path) || error("WENO7 stats file does not exist: $path")
    lines = readlines(path)
    length(lines) == 2 || error(
        "WENO7 stats must contain exactly one header and one data row: $path",
    )
    lines[1] == join(CT_WENO7_STATS_COLUMNS, " ") || error(
        "invalid WENO7 stats header in $path",
    )
    fields = split(lines[2])
    length(fields) == length(CT_WENO7_STATS_COLUMNS) || error(
        "invalid WENO7 stats column count in $path",
    )

    values = try
        (
            resolution=parse(Int, fields[1]),
            l1_edge=parse(Float64, fields[2]),
            l2_edge=parse(Float64, fields[3]),
            linf_edge=parse(Float64, fields[4]),
            constant_error=parse(Float64, fields[5]),
            nonfinite_flag=parse(Int, fields[6]),
        )
    catch exception
        error("invalid WENO7 stats value in $path: $(sprint(showerror, exception))")
    end

    values.resolution > 0 || error(
        "WENO7 stats resolution must be positive in $path",
    )
    errors = (values.l1_edge, values.l2_edge, values.linf_edge)
    all(isfinite, errors) || error("nonfinite WENO7 edge error in $path")
    all(error_value -> error_value >= 0, errors) || error(
        "negative WENO7 edge error in $path",
    )
    isfinite(values.constant_error) || error(
        "nonfinite WENO7 constant error in $path",
    )
    0 <= values.constant_error <= CT_WENO7_MAX_CONSTANT_ERROR || error(
        "WENO7 constant error $(values.constant_error) exceeds " *
        "$(CT_WENO7_MAX_CONSTANT_ERROR) in $path",
    )
    values.nonfinite_flag == 0 || error(
        "WENO7 nonfinite flag is $(values.nonfinite_flag) in $path",
    )
    return values
end

function ct_weno7_verify_convergence(
    paths;
    min_l1_order=5.5,
    min_l2_order=5.5,
    min_linf_order=5.0,
)
    length(paths) == 3 || error(
        "WENO7 convergence requires exactly three stats paths",
    )
    runs = collect(map(ct_weno7_read_stats, paths))
    sort!(runs; by=run -> run.resolution)
    resolutions = Tuple(run.resolution for run in runs)
    resolutions == (12, 24, 48) || error(
        "WENO7 convergence requires resolutions (12, 24, 48), got $resolutions",
    )

    function norm_summary(field)
        errors = Tuple(getproperty(run, field) for run in runs)
        all(error_value -> isfinite(error_value) && error_value > 0, errors) ||
            error("WENO7 $field errors must be finite and positive")
        all(diff(collect(errors)) .< 0) || error(
            "WENO7 $field errors do not strictly decrease",
        )
        orders = ntuple(index -> log2(errors[index]/errors[index+1]), 2)
        all(isfinite, orders) || error("nonfinite WENO7 $field order")
        return errors, orders
    end

    l1_errors, l1_orders = norm_summary(:l1_edge)
    l2_errors, l2_orders = norm_summary(:l2_edge)
    linf_errors, linf_orders = norm_summary(:linf_edge)
    l1_orders[2] >= min_l1_order || error(
        "finest WENO7 L1 order $(l1_orders[2]) < $min_l1_order",
    )
    l2_orders[2] >= min_l2_order || error(
        "finest WENO7 L2 order $(l2_orders[2]) < $min_l2_order",
    )
    linf_orders[2] >= min_linf_order || error(
        "finest WENO7 Linf order $(linf_orders[2]) < $min_linf_order",
    )

    return (
        resolutions=resolutions,
        l1_errors=l1_errors,
        l2_errors=l2_errors,
        linf_errors=linf_errors,
        l1_orders=l1_orders,
        l2_orders=l2_orders,
        linf_orders=linf_orders,
        constant_errors=Tuple(run.constant_error for run in runs),
    )
end

function ct_weno7_read_multiblock_stats(path)
    isfile(path) || error("WENO7 multiblock stats file does not exist: $path")
    lines = readlines(path)
    length(lines) == 2 || error(
        "WENO7 multiblock stats require one header and one data row: $path",
    )
    lines[1] == join(CT_WENO7_MULTIBLOCK_COLUMNS, " ") || error(
        "invalid WENO7 multiblock stats header in $path",
    )
    fields = split(lines[2])
    length(fields) == length(CT_WENO7_MULTIBLOCK_COLUMNS) || error(
        "invalid WENO7 multiblock stats column count in $path",
    )
    parsed = try
        numbers = parse.(Float64, fields[2:end-1])
        (
            resolution=parse(Int, fields[1]),
            pre_sync_abs=numbers[1], pre_sync_rel=numbers[2],
            post_sync_abs=numbers[3], post_sync_rel=numbers[4],
            interface_l1=numbers[5], interface_l2=numbers[6],
            interface_linf=numbers[7],
            layers=Tuple(numbers[8:11]), face_divb=numbers[12],
            nonfinite_flag=parse(Int, fields[end]),
        )
    catch exception
        error(
            "invalid WENO7 multiblock stats value in $path: " *
            sprint(showerror, exception),
        )
    end
    parsed.resolution > 0 || error("invalid multiblock resolution in $path")
    finite_values = (
        parsed.pre_sync_abs, parsed.pre_sync_rel,
        parsed.post_sync_abs, parsed.post_sync_rel,
        parsed.interface_l1, parsed.interface_l2, parsed.interface_linf,
        parsed.layers..., parsed.face_divb,
    )
    all(isfinite, finite_values) || error("nonfinite WENO7 multiblock value in $path")
    all(value -> value >= 0, finite_values[1:12]) || error(
        "negative WENO7 multiblock error or residual in $path",
    )
    parsed.nonfinite_flag == 0 || error(
        "WENO7 multiblock nonfinite flag=$(parsed.nonfinite_flag) in $path",
    )
    max(parsed.post_sync_abs, parsed.post_sync_rel) <=
        CT_WENO7_INTERFACE_INVARIANT_TOL || error(
        "post-sync interface residual exceeds " *
        "$(CT_WENO7_INTERFACE_INVARIANT_TOL) in $path",
    )
    parsed.face_divb <= CT_WENO7_INTERFACE_INVARIANT_TOL || error(
        "face-divB=$(parsed.face_divb) exceeds " *
        "$(CT_WENO7_INTERFACE_INVARIANT_TOL) in $path",
    )
    return parsed
end

function ct_weno7_verify_multiblock_convergence(
    paths; min_pre_order=5.5, min_l1_order=5.5,
    min_l2_order=5.5, min_linf_order=5.0,
)
    length(paths) == 3 || error(
        "WENO7 multiblock convergence requires exactly three stats paths",
    )
    runs = sort!(collect(map(ct_weno7_read_multiblock_stats, paths));
                 by=run -> run.resolution)
    resolutions = Tuple(run.resolution for run in runs)
    resolutions == (12, 24, 48) || error(
        "WENO7 multiblock resolutions must be (12, 24, 48), got $resolutions",
    )

    orders(field) = ntuple(index -> log2(
        getproperty(runs[index], field)/getproperty(runs[index+1], field),
    ), 2)
    for field in (:interface_l1, :interface_l2, :interface_linf)
        values = Tuple(getproperty(run, field) for run in runs)
        all(value -> value > 0, values) || error("$field must be positive")
        all(diff(collect(values)) .< 0) || error("$field must decrease")
    end
    l1_orders = orders(:interface_l1)
    l2_orders = orders(:interface_l2)
    linf_orders = orders(:interface_linf)
    l1_orders[2] >= min_l1_order || error(
        "finest interface L1 order $(l1_orders[2]) < $min_l1_order",
    )
    l2_orders[2] >= min_l2_order || error(
        "finest interface L2 order $(l2_orders[2]) < $min_l2_order",
    )
    linf_orders[2] >= min_linf_order || error(
        "finest interface Linf order $(linf_orders[2]) < $min_linf_order",
    )

    pre_values = Tuple(run.pre_sync_abs for run in runs)
    pre_orders = nothing
    if maximum(pre_values) > CT_WENO7_PRE_SYNC_ZERO_TOL
        all(value -> value > CT_WENO7_PRE_SYNC_ZERO_TOL, pre_values) || error(
            "pre-sync discrepancies mix resolved and roundoff-zero values",
        )
        all(diff(collect(pre_values)) .< 0) || error(
            "pre-sync discrepancy must decrease",
        )
        pre_orders = ntuple(index -> log2(
            pre_values[index]/pre_values[index+1],
        ), 2)
        pre_orders[2] >= min_pre_order || error(
            "finest pre-sync discrepancy order $(pre_orders[2]) < $min_pre_order",
        )
    end

    for layer in 1:4
        values = Tuple(run.layers[layer] for run in runs)
        values[2] <= values[1] && values[3] <= values[2] || error(
            "excluded interface layer $layer error grows under refinement: $values",
        )
    end
    return (
        resolutions=resolutions,
        pre_sync_abs=pre_values, pre_sync_orders=pre_orders,
        post_sync_abs=Tuple(run.post_sync_abs for run in runs),
        post_sync_rel=Tuple(run.post_sync_rel for run in runs),
        core_l1=Tuple(run.interface_l1 for run in runs),
        core_l2=Tuple(run.interface_l2 for run in runs),
        core_linf=Tuple(run.interface_linf for run in runs),
        core_orders=(l1=l1_orders, l2=l2_orders, linf=linf_orders),
        finest_core_orders=(l1=l1_orders[2], l2=l2_orders[2],
                            linf=linf_orders[2]),
        layers=Tuple(run.layers for run in runs),
        face_divb=Tuple(run.face_divb for run in runs),
    )
end

function ct_weno7_stats_kind(paths)
    length(paths) == 3 || error(
        "WENO7 convergence requires exactly three stats paths",
    )
    headers = map(paths) do path
        isfile(path) || error("WENO7 stats file does not exist: $path")
        open(readline, path)
    end
    all(==(headers[1]), headers) || error(
        "WENO7 convergence inputs have mixed stats headers",
    )
    headers[1] == join(CT_WENO7_STATS_COLUMNS, " ") && return :single_block
    headers[1] == join(CT_WENO7_MULTIBLOCK_COLUMNS, " ") && return :multiblock
    error("unrecognized WENO7 stats header in $(first(paths))")
end

function _ct_weno7_verify_main(args)
    length(args) == 3 || error(
        "usage: julia verify.jl <N12.dat> <N24.dat> <N48.dat>",
    )
    if ct_weno7_stats_kind(args) == :multiblock
        summary = ct_weno7_verify_multiblock_convergence(args)
        println("WENO7 same-type curved two-block convergence passed")
        @printf(
            "finest core orders: L1=%.8f L2=%.8f Linf=%.8f\n",
            summary.finest_core_orders.l1,
            summary.finest_core_orders.l2,
            summary.finest_core_orders.linf,
        )
        @printf(
            "max post-sync abs=%.3e rel=%.3e face-divB=%.3e\n",
            maximum(summary.post_sync_abs), maximum(summary.post_sync_rel),
            maximum(summary.face_divb),
        )
        return summary
    end

    summary = ct_weno7_verify_convergence(args)
    println("warped WENO7 edge convergence passed")
    for index in eachindex(summary.resolutions)
        @printf(
            "N=%d L1=%.16e L2=%.16e Linf=%.16e constant=%.16e\n",
            summary.resolutions[index],
            summary.l1_errors[index],
            summary.l2_errors[index],
            summary.linf_errors[index],
            summary.constant_errors[index],
        )
    end
    @printf(
        "L1 orders: %.8f %.8f; L2 orders: %.8f %.8f; Linf orders: %.8f %.8f\n",
        summary.l1_orders...,
        summary.l2_orders...,
        summary.linf_orders...,
    )
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    _ct_weno7_verify_main(ARGS)
end
