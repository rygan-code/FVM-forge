using Printf
using Serialization

const ARRAY_KEYS = (
    :U, :Q,
    :Bx_face, :By_face, :Bz_face,
    :Ex_edge, :Ey_edge, :Ez_edge,
    :Fx, :Fy, :Fz,
)

function load_stage_snapshot(case_dir, label, block)
    debug_dir = joinpath(abspath(case_dir), "stage_debug")
    isdir(debug_dir) || error("missing stage-debug directory: $debug_dir")
    suffix = "_$(label)_block$(block)_"
    matches = filter(readdir(debug_dir; join=true)) do path
        occursin(suffix, basename(path)) && endswith(path, ".jls")
    end
    length(matches) == 1 || error(
        "expected one snapshot for label=$label block=$block in $debug_dir; " *
        "found $(length(matches))",
    )
    return open(deserialize, only(matches))
end

function difference_norms(left, right)
    size(left) == size(right) || error(
        "array sizes differ: $(size(left)) and $(size(right))",
    )
    difference = Float64.(left) .- Float64.(right)
    absolute = abs.(difference)
    worst = argmax(absolute)
    count = length(difference)
    return (
        l1=sum(absolute) / count,
        l2=sqrt(sum(abs2, difference) / count),
        linf=absolute[worst],
        index=Tuple(worst),
        left=Float64(left[worst]),
        right=Float64(right[worst]),
    )
end

function compare_stage_debug(case_a, case_b, output_path)
    labels = (
        "post_flux",
        "post_edge_emf",
        "post_edge_sync",
        "post_divergence",
        "post_face_b_update",
        "post_face_barrier",
        "post_stage_sync",
    )
    open(output_path, "w") do io
        println(io, "label\tblock\tarray\tL1\tL2\tLinf\tindex\tvalue_a\tvalue_b")
        for label in labels, block in 0:1
            left = load_stage_snapshot(case_a, label, block)
            right = load_stage_snapshot(case_b, label, block)
            left[:dimensions] == right[:dimensions] || error(
                "block dimensions differ for label=$label block=$block",
            )
            for key in ARRAY_KEYS
                haskey(left, key) == haskey(right, key) || error(
                    "array $key exists in only one snapshot for label=$label block=$block",
                )
                haskey(left, key) || continue
                norms = difference_norms(left[key], right[key])
                @printf(
                    "%20s block=%d %-8s L1=%.6e L2=%.6e Linf=%.6e index=%s A=%.8e B=%.8e\n",
                    label, block, String(key), norms.l1, norms.l2,
                    norms.linf, string(norms.index), norms.left, norms.right,
                )
                @printf(
                    io, "%s\t%d\t%s\t%.16e\t%.16e\t%.16e\t%s\t%.16e\t%.16e\n",
                    label, block, String(key), norms.l1, norms.l2,
                    norms.linf, string(norms.index), norms.left, norms.right,
                )
            end
        end
    end
    println("wrote $output_path")
end

3 <= length(ARGS) <= 3 || error(
    "usage: julia compare_stage_debug.jl <case_a_dir> <case_b_dir> <output_path>",
)
compare_stage_debug(abspath(ARGS[1]), abspath(ARGS[2]), abspath(ARGS[3]))
