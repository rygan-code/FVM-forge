# Run the warped single-block metric-CT uniform-state regression.

ENV["OT3D_CT"] = "true"
ENV["OT3D_STRICT_CT"] = "true"
ENV["OT3D_TEST_CASE"] = "MetricCTUniform"
ENV["OT3D_MESH_DIR"] = get(ENV, "OT3D_MESH_DIR", joinpath(@__DIR__, "MESH"))
get!(ENV, "OT3D_FINAL_TIME", "0.1")
get!(ENV, "OT3D_STATS_INTERVAL", "1")

include(joinpath(@__DIR__, "..", "ORSZAG_TANG_3D", "run.jl"))
