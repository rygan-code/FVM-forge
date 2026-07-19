# Run a circularly polarized Alfven wave on the warped single-block mesh.

include(joinpath(@__DIR__, "metric_ct_analytic.jl"))
include(joinpath(@__DIR__, "metric_ct_diagnostics.jl"))

ENV["OT3D_CT"] = "true"
ENV["OT3D_STRICT_CT"] = "true"
ENV["OT3D_CHARACTERISTIC"] = get(ENV, "METRIC_CT_CHARACTERISTIC", "false")
ENV["OT3D_TEST_CASE"] = "MetricCTAlfven"
ENV["OT3D_MESH_DIR"] = get(
    ENV, "METRIC_CT_MESH_DIR", joinpath(@__DIR__, "MESH_N16"),
)
ENV["OT3D_STATS_FILE"] = get(
    ENV, "METRIC_CT_STATS_FILE", joinpath(@__DIR__, "alfven_stats.dat"),
)
ENV["OT3D_FINAL_TIME"] = get(ENV, "METRIC_CT_FINAL_TIME", "0.1")
ENV["OT3D_STATS_INTERVAL"] = "1"

include(joinpath(@__DIR__, "..", "ORSZAG_TANG_3D", "run.jl"))
