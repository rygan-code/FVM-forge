# debug/gen_pipe_test_mesh.jl
# Generate 128x44x44 butterfly pipe mesh with Lx=2.0 for artifact detection testing

include("../Utils/gen_butterfly_fvm.jl")

println("Building pipe test mesh: 128 x 44 x 44, Lx=2.0")
build_mesh("debug/MESH_PIPE_TEST", 128, 44, 44, 44, 2.0)
println("Pipe test mesh generation complete.")
