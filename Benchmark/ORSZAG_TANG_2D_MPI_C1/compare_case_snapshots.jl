# Compare two OT snapshots at the same resolution and physical time.

include(joinpath(@__DIR__, "analyze_convergence.jl"))

function compare_cases(run_root, case_a, case_b, resolution, output_path)
    mesh_dir = joinpath(
        run_root, "source", "Benchmark", "ORSZAG_TANG_2D_MPI_C1",
        "mesh", "warped_N$(resolution)",
    )
    snapshot_a = read_snapshot(
        joinpath(run_root, "results", case_a), mesh_dir, resolution,
    )
    snapshot_b = read_snapshot(
        joinpath(run_root, "results", case_b), mesh_dir, resolution,
    )
    abs(snapshot_a.time - snapshot_b.time) <=
        128eps(max(snapshot_a.time, snapshot_b.time)) || error(
        "snapshot times differ: $(snapshot_a.time) and $(snapshot_b.time)",
    )

    open(output_path, "w") do io
        println(io, "field\tL1\tL2\tLinf\tblock\ti\tj\tk\tvalue_a\tvalue_b")
        @printf(
            "OT2D_CASE_COMPARISON time=%.16e resolution=%d\n",
            snapshot_a.time, resolution,
        )
        for field in FIELDS
            sum_volume = 0.0
            sum_abs = 0.0
            sum_square = 0.0
            worst = nothing
            for bid in 0:1
                values_a = snapshot_a.fields[bid][field]
                values_b = snapshot_b.fields[bid][field]
                weights = snapshot_a.volumes[bid]
                difference = values_a .- values_b
                sum_volume += sum(weights)
                sum_abs += sum(weights .* abs.(difference))
                sum_square += sum(weights .* abs2.(difference))
                index = argmax(abs.(difference))
                value = abs(difference[index])
                if worst === nothing || value > worst.value
                    i, j, k = Tuple(index)
                    worst = (
                        value=value, bid=bid, index=(i, j, k),
                        value_a=values_a[index], value_b=values_b[index],
                    )
                end
            end
            l1 = sum_abs / sum_volume
            l2 = sqrt(sum_square / sum_volume)
            @printf(
                "%4s L1=%.6e L2=%.6e Linf=%.6e worst_block=%d ijk=%s A=%.8e B=%.8e\n",
                field, l1, l2, worst.value, worst.bid,
                string(worst.index), worst.value_a, worst.value_b,
            )
            @printf(
                io, "%s\t%.16e\t%.16e\t%.16e\t%d\t%d\t%d\t%d\t%.16e\t%.16e\n",
                field, l1, l2, worst.value, worst.bid, worst.index...,
                worst.value_a, worst.value_b,
            )
        end
    end
    println("wrote $output_path")
end

4 <= length(ARGS) <= 5 || error(
    "usage: julia compare_case_snapshots.jl " *
    "<run_root> <case_a> <case_b> <resolution> [output_path]",
)
run_root = abspath(ARGS[1])
case_a = ARGS[2]
case_b = ARGS[3]
resolution = parse(Int, ARGS[4])
output_path = length(ARGS) == 5 ? abspath(ARGS[5]) : joinpath(
    run_root, "comparison_$(lowercase(case_a))_vs_$(lowercase(case_b)).tsv",
)
compare_cases(run_root, case_a, case_b, resolution, output_path)
