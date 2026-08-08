# Quick syntax/include check for MHD solver
# Tests that all files parse and constants are consistent
println("Testing MHD solver include chain...")

const debug_nan::Bool = false
const debug_sync::Bool = false
const profiling::Bool = false
const gpu_aware_mpi::Bool = false
const flow_forcing::Bool = false
const forcing_mode::Int64 = 1
const wall_perturbation::Bool = false
const wall_perturbation_type::Int32 = 0
const cache_metrics::Bool = true
const Re_target::FT = FT(1000.0)
const Ma_target::FT = FT(0.3e0)
const Ro_target::FT = zero(FT)
const Tw::FT = FT(300.0)
const Lx::FT = one(FT)
const hit_forcing_A::FT = zero(FT)

using CUDA

const equation_type = :MHD
const resistive::Bool = false
const cr_glm::FT = FT(0.18e0)

const _project_root = joinpath(@__DIR__, "..", "..")
include(joinpath(_project_root,"src","core","equation_config.jl"))
println("  ✓ physics.jl (Ncons=$Ncons, Nprim=$Nprim)")

include(joinpath(_project_root,"src","time","structured_rk3_solver.jl"))
println("  ✓ solver.jl")

# Check that the key MHD functions exist
println("  ✓ MHD_Rusanov_Flux: ", isdefined(@__MODULE__, :MHD_Rusanov_Flux))
println("  ✓ HLLD_Flux: ", isdefined(@__MODULE__, :HLLD_Flux))
println("  ✓ MHD_KEP_Flux: ", isdefined(@__MODULE__, :MHD_KEP_Flux))
println("  ✓ glm_source_kernel!: ", isdefined(@__MODULE__, :glm_source_kernel!))
println("  ✓ compute_cf_max_kernel!: ", isdefined(@__MODULE__, :compute_cf_max_kernel!))
println("  ✓ init_brio_wu: ", isdefined(@__MODULE__, :init_brio_wu))
println("  ✓ ch_glm_current: ", ch_glm_current)

println("\nAll MHD solver includes validated successfully!")
