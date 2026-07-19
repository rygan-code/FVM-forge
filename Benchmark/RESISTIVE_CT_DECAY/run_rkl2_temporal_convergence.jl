using DelimitedFiles
using Printf

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const FINAL_TIME = parse(Float64, get(
    ENV, "RKL2_TEMPORAL_FINAL_TIME", "0.1",
))
const STEP_COUNTS = (10, 20, 40)
const REFERENCE_STEPS = parse(Int, get(
    ENV, "RKL2_TEMPORAL_REFERENCE_STEPS", "160",
))
const MIN_ORDER = parse(Float64, get(
    ENV, "RKL2_TEMPORAL_MIN_ORDER", "1.7",
))
const REUSE = lowercase(get(
    ENV, "RKL2_TEMPORAL_REUSE", "false",
)) == "true"

function run_decay(step_count)
    saved_path = joinpath(
        @__DIR__, "rkl2_temporal_steps$(step_count).dat",
    )
    if REUSE && isfile(saved_path)
        return vec(readdlm(saved_path, Float64; comments=true))
    end
    run_dt = FINAL_TIME / step_count
    runner = addenv(
        `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $(joinpath(@__DIR__, "run.jl"))`,
        "RESISTIVE_CT_DECAY_INTEGRATOR" => "rkl2_strang",
        "RESISTIVE_CT_DECAY_DT" => string(run_dt),
        "RESISTIVE_CT_DECAY_STEPS" => string(step_count),
    )
    run(runner)
    summary_path = joinpath(@__DIR__, "resistive_ct_decay_summary.dat")
    row = vec(readdlm(summary_path, Float64; comments=true))
    cp(
        summary_path,
        saved_path;
        force=true,
    )
    return row
end

rows = [run_decay(step_count) for step_count in STEP_COUNTS]
reference = run_decay(REFERENCE_STEPS)
amplitudes = getindex.(rows, 2)
reference_amplitude = reference[2]
errors = abs.(amplitudes .- reference_amplitude)
orders = log2.(errors[1:end-1] ./ errors[2:end])

output = joinpath(@__DIR__, "rkl2_temporal_convergence.dat")
open(output, "w") do io
    println(io, "# steps dt amplitude reference self_error order total_rel divB_linf")
    for index in eachindex(STEP_COUNTS)
        order = index == 1 ? NaN : orders[index-1]
        @printf(
            io, "%d %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
            STEP_COUNTS[index], FINAL_TIME/STEP_COUNTS[index],
            amplitudes[index], reference_amplitude, errors[index], order,
            rows[index][8], rows[index][11],
        )
    end
end

all(diff(errors) .< 0) || error(
    "RKL2-Strang temporal error is not monotone: $errors",
)
minimum(orders) >= MIN_ORDER || error(
    "RKL2-Strang temporal order $(minimum(orders)) < $MIN_ORDER",
)
maximum(getindex.(rows, 8)) <= 1e-10 || error(
    "RKL2-Strang total-energy drift exceeds 1e-10",
)
maximum(getindex.(rows, 11)) <= 1e-11 || error(
    "RKL2-Strang face-divB exceeds 1e-11",
)
@printf(
    "RKL2_TEMPORAL_CONVERGENCE orders=(%.6f,%.6f) reference_steps=%d\n",
    orders[1], orders[2], REFERENCE_STEPS,
)
