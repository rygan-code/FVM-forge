const _metric_ct_multiblock_dir = @__DIR__
ENV["OT3D_CT"] = "true"
ENV["OT3D_STRICT_CT"] = "true"
ENV["OT3D_TEST_CASE"] = "MetricCTUniform"
ENV["OT3D_MESH_DIR"] = joinpath(_metric_ct_multiblock_dir, "MESH")
get!(ENV, "OT3D_PERIODIC", "true,false,true")
get!(ENV, "OT3D_FINAL_TIME", "0.1")
get!(ENV, "OT3D_STATS_INTERVAL", "1")
get!(ENV, "OT3D_STATS_FILE", joinpath(
    _metric_ct_multiblock_dir, "metric_stats.dat",
))
get!(ENV, "METRIC_CT_INTERFACE_STATS_FILE", joinpath(
    _metric_ct_multiblock_dir, "interface_stats.dat",
))

using MPI
MPI.Init()
MPI.Comm_size(MPI.COMM_WORLD) == 2 || error(
    "metric CT multiblock acceptance requires exactly 2 MPI ranks",
)

using CUDA
world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
device = CUDA.device()
println(
    "[metric-ct-topology] rank=$world_rank device=$(CUDA.name(device)) " *
    "ordinal=$(CUDA.deviceid(device))",
)
flush(stdout)
MPI.Barrier(MPI.COMM_WORLD)

include(joinpath(_metric_ct_multiblock_dir, "interface_diagnostics.jl"))

function in_situ_post_process(
    tt::Integer, active_time, current_dt, blocks, world_rank,
    Block_Nprocs, block_comms,
)
    invoke(
        in_situ_post_process,
        Tuple{Any,Any,Any,Any,Any,Any,Any},
        tt, active_time, current_dt, blocks, world_rank,
        Block_Nprocs, block_comms,
    )
    metric_ct_interface_post_process(
        tt, active_time, current_dt, blocks, world_rank, Block_Nprocs,
    )
    return nothing
end

include(joinpath(
    _metric_ct_multiblock_dir, "..", "ORSZAG_TANG_3D", "run.jl",
))
