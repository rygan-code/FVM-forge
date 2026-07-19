using MPI

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "Benchmark", "METRIC_CT_MULTIBLOCK", "gen_mesh.jl"))
include(joinpath(@__DIR__, "verify.jl"))

function run_mpi_acceptance()
    artifact_root = get(
        ENV, "CT_WENO7_MULTIBLOCK_ARTIFACT_ROOT",
        joinpath(@__DIR__, "artifacts_multiblock"),
    )
    runner = joinpath(@__DIR__, "run_multiblock.jl")
    stats_paths = String[]
    for resolution in (12, 24, 48)
        case_root = joinpath(artifact_root, "N$resolution")
        mesh_directory = metric_ct_multiblock_mesh_output_dir(
            artifact_root, resolution,
        )
        metric_ct_generate_multiblock_mesh(
            resolution, resolution, resolution, mesh_directory,
        )
        stats_path = joinpath(case_root, "stats.dat")
        log_path = joinpath(case_root, "multiblock.log")
        mkpath(case_root)
        command = `$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=$ROOT $runner $resolution $mesh_directory $stats_path`
        open(log_path, "w") do io
            run(pipeline(command; stdout=io, stderr=io))
        end
        print(read(log_path, String))
        push!(stats_paths, stats_path)
    end
    summary = ct_weno7_verify_multiblock_convergence(stats_paths)
    println("WENO7 same-type curved two-block MPI acceptance passed")
    println(summary)
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mpi_acceptance()
end
