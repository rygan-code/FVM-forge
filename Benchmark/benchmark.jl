# Unified benchmark launcher.
# Usage: julia --project=. Benchmark/benchmark.jl [list | run | verify] [case]

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const BENCHMARK_RUNNERS = Dict(
    "BL" => "bl.jl",
    "BRIO_WU" => "brio_wu.jl",
    "HARTMANN" => "hartmann.jl",
    "PIPEFLOW" => "pipeflow.jl",
    "SOD" => "sod.jl",
    "TGV" => "tgv.jl",
)
const BENCHMARK_VERIFIERS = Dict(
    "BL" => :verify_bl,
    "PIPEFLOW" => :verify_pipeflow,
    "SOD" => :verify_sod,
    "TGV" => :verify_tgv,
)
const CASES = sort!(collect(keys(BENCHMARK_RUNNERS)))

function print_usage()
    println("OpenCFD-FVM benchmark suite")
    println("Usage: julia --project=. Benchmark/benchmark.jl [COMMAND] [CASE]")
    println("Commands:")
    println("  list            List runnable benchmarks")
    println("  run [CASE]      Run the canonical benchmark entry")
    println("  verify [CASE]   Run Benchmark/CASE/verify.jl when available")
end

function require_case(case_name)
    normalized = uppercase(case_name)
    normalized in CASES || throw(ArgumentError(
        "unknown benchmark $case_name; available cases: $(join(CASES, ", "))",
    ))
    return normalized
end

function run_case(case_name)
    normalized = require_case(case_name)
    runner = joinpath(
        PROJECT_ROOT, "run", "benchmarks", BENCHMARK_RUNNERS[normalized],
    )
    return Base.include(Main, runner)
end

function verify_case(case_name)
    normalized = require_case(case_name)
    haskey(BENCHMARK_VERIFIERS, normalized) || throw(ArgumentError(
        "benchmark $normalized does not provide a maintained verifier",
    ))
    verifier = joinpath(PROJECT_ROOT, "Benchmark", normalized, "verify.jl")
    Base.include(Main, verifier)
    passed = getfield(Main, BENCHMARK_VERIFIERS[normalized])()
    passed || error("benchmark $normalized verification failed")
    return true
end

if isempty(ARGS)
    print_usage()
elseif ARGS[1] == "list"
    println(join(CASES, "\n"))
elseif ARGS[1] == "run" && length(ARGS) >= 2
    case_name = ARGS[2]
    deleteat!(ARGS, 1:2)
    run_case(case_name)
elseif ARGS[1] == "verify" && length(ARGS) == 2
    verify_case(ARGS[2])
else
    print_usage()
    exit(1)
end
