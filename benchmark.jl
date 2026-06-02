# benchmark.jl - Unified Benchmark Management Script
# Usage: julia benchmark.jl [list | setup | verify] [case]

const BENCHMARK_DIR = "Benchmark"
const CASES = ["SOD", "TGV", "OBL", "BL", "BL_TRANSITION", "PIPEFLOW"]

function print_usage()
    println("\nFlame3D Unified Benchmark Suite")
    println("===============================")
    println("Usage: julia benchmark.jl [COMMAND] [CASE]")
    println("\nCommands:")
    println("  list            List all available benchmarks")
    println("  setup [CASE]    Configure the solver for a specific case (copies config to run.jl)")
    println("  verify [CASE]   Verify the current results in ./PLT/ for a specific case")
    println("\nExample:")
    println("  julia benchmark.jl setup SOD")
    println("  julia benchmark.jl verify SOD")
end

function setup_case(case_name)
    if !(case_name in CASES)
        println("Error: Unknown case $case_name. Available: $(join(CASES, ", "))")
        return
    end
    
    src_config = joinpath(BENCHMARK_DIR, case_name, "run_config.jl")
    dest_config = "run.jl"
    
    if isfile(src_config)
        cp(src_config, dest_config, force=true)
        println("✓ Setup COMPLETE: $case_name configuration deployed to run.jl")
        println("  Please ensure the correct mesh file is referenced in run.jl.")
    else
        println("✗ Error: Config file not found at $src_config")
    end
end

function verify_case(case_name)
    if !(case_name in CASES)
        println("Error: Unknown case $case_name.")
        return
    end
    
    verify_script = joinpath(BENCHMARK_DIR, case_name, "verify.jl")
    if isfile(verify_script)
        println("--- Running Verification for $case_name ---")
        include(verify_script)
    else
        println("✗ Error: Verification script not found at $verify_script")
    end
end

if isempty(ARGS)
    print_usage()
else
    cmd = ARGS[1]
    if cmd == "list"
        println("Available Benchmarks: $(join(CASES, ", "))")
    elseif cmd == "setup" && length(ARGS) >= 2
        setup_case(uppercase(ARGS[2]))
    elseif cmd == "verify" && length(ARGS) >= 2
        verify_case(uppercase(ARGS[2]))
    else
        print_usage()
    end
end
