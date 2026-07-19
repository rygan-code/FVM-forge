using DelimitedFiles
using Printf

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const RESOLUTIONS = (32, 64, 128)
const REFERENCE_STEPS = parse(Int, get(
    ENV, "HARTMANN_CT_CONVERGENCE_STEPS", "5000",
))
const REFERENCE_DT = parse(Float64, get(
    ENV, "HARTMANN_CT_CONVERGENCE_DT", "0.001",
))
const REFERENCE_NY = 64
const REUSE = lowercase(get(
    ENV, "HARTMANN_CT_CONVERGENCE_REUSE", "false",
)) == "true"
const MIN_ORDER = parse(Float64, get(
    ENV, "HARTMANN_CT_MIN_ORDER", "1.2",
))

rows = Vector{Vector{Float64}}()
for ny in RESOLUTIONS
    summary_path = joinpath(@__DIR__, "hartmann_ct_summary_N$(ny).dat")
    profile_path = joinpath(@__DIR__, "hartmann_ct_profile_N$(ny).dat")
    if !(REUSE && isfile(summary_path) && isfile(profile_path))
        run_dt = min(
            REFERENCE_DT,
            REFERENCE_DT * (REFERENCE_NY/ny)^2,
        )
        final_time = REFERENCE_STEPS * REFERENCE_DT
        run_steps = round(Int, final_time/run_dt)
        generator = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $(joinpath(@__DIR__, "gen_mesh.jl"))`,
            "HARTMANN_CT_NY" => string(ny),
        )
        runner = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $(joinpath(@__DIR__, "run.jl"))`,
            "HARTMANN_CT_STEPS" => string(run_steps),
            "HARTMANN_CT_DT" => string(run_dt),
        )
        run(generator)
        run(runner)
        cp(joinpath(@__DIR__, "hartmann_ct_summary.dat"), summary_path; force=true)
        cp(joinpath(@__DIR__, "hartmann_ct_profile.dat"), profile_path; force=true)
    end
    summary = vec(readdlm(summary_path, Float64; comments=true))
    push!(rows, summary)
end

u_errors = getindex.(rows, 4)
b_errors = getindex.(rows, 7)
u_orders = [log2(u_errors[i]/u_errors[i+1]) for i in 1:2]
b_orders = [log2(b_errors[i]/b_errors[i+1]) for i in 1:2]

output = joinpath(@__DIR__, "hartmann_ct_convergence.dat")
open(output, "w") do io
    println(io, "# Ny u_L2 bx_L2 u_order bx_order")
    for i in eachindex(RESOLUTIONS)
        u_order = i == 1 ? NaN : u_orders[i-1]
        b_order = i == 1 ? NaN : b_orders[i-1]
        @printf(io, "%d %.16e %.16e %.16e %.16e\n",
            RESOLUTIONS[i], u_errors[i], b_errors[i], u_order, b_order)
    end
end

all(diff(u_errors) .< 0) || error("Hartmann velocity error is not monotone")
all(diff(b_errors) .< 0) || error("Hartmann magnetic error is not monotone")
minimum(u_orders) >= MIN_ORDER || error(
    "Hartmann velocity convergence order $(minimum(u_orders)) < $MIN_ORDER",
)
minimum(b_orders) >= MIN_ORDER || error(
    "Hartmann magnetic convergence order $(minimum(b_orders)) < $MIN_ORDER",
)
@printf(
    "HARTMANN_CT_CONVERGENCE u_orders=(%.6f,%.6f) bx_orders=(%.6f,%.6f)\n",
    u_orders[1], u_orders[2], b_orders[1], b_orders[2],
)
