# CUDA integration smoke for the configured Brio-Wu solver. `run.jl` owns the
# block lifecycle inside time_step, so a post-run kernel launch cannot access
# those local buffers. Task-specific i/j/k cache launches live in
# tests/test_ct_weno7.jl.
include("run.jl")
println("BRIO_WU integration smoke: passed")
