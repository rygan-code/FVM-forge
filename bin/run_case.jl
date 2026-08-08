#!/usr/bin/env julia

length(ARGS)>=1 || error("usage: julia --project=. bin/run_case.jl config/cases/case.toml")
const PROJECT_ROOT=abspath(joinpath(@__DIR__,".."))
const VALIDATE_ONLY="--validate-only" in ARGS[2:end]
using OpenCFDFVM
OpenCFDFVM.load_backend_api!(Main)
const CASE_CONFIG=validate_case_config(load_case_config(abspath(ARGS[1])))
apply_case_environment!(CASE_CONFIG)

if CASE_CONFIG.backend.mesh_type==:structured
    result=run_backend_case!(
        select_spatial_backend(:structured),CASE_CONFIG;
        project_root=PROJECT_ROOT,validate_only=VALIDATE_ONLY)
    VALIDATE_ONLY && println(
        "CASE_CONFIG_VALID name=$(CASE_CONFIG.name) mode=legacy_entry")
    exit()
end

const EFFECTIVE_DEVICE=activate_requested_device!(CASE_CONFIG.backend.device)
println("CASE_DEVICE requested=$(CASE_CONFIG.backend.device) effective=$EFFECTIVE_DEVICE")

const FT=Float64
const equation_type=CASE_CONFIG.physics.equations
const Ncons=equation_type==:MHD ? 9 : 5
const Nprim=equation_type==:MHD ? 10 : 6
const Nhydro=Ncons
const var"γ"=FT(CASE_CONFIG.physics.gamma)
const Rg=FT(CASE_CONFIG.physics.gas_constant)
const Pr=FT(CASE_CONFIG.physics.prandtl)
const Tw=FT(CASE_CONFIG.physics.wall_temperature)
const viscous=CASE_CONFIG.physics.viscous
const resistive=CASE_CONFIG.physics.resistive
const var"η_mhd"=FT(CASE_CONFIG.physics.resistivity)
const ct_mode=CASE_CONFIG.physics.ct
const strict_ct_positivity=false
const positivity_mode=CASE_CONFIG.numerics.positivity_mode
const density_floor=FT(CASE_CONFIG.numerics.density_floor)
const pressure_floor=FT(CASE_CONFIG.numerics.pressure_floor)
const cr_glm=FT(0.18)

OpenCFDFVM.load_unstructured_solver_stack!(Main)

result=run_backend_case!(
    select_spatial_backend(:unstructured),CASE_CONFIG;
    project_root=PROJECT_ROOT,validate_only=VALIDATE_ONLY)
if VALIDATE_ONLY
    println("CASE_CONFIG_VALID name=$(CASE_CONFIG.name) mode=native " *
        "device=$EFFECTIVE_DEVICE cells=$(CASE_CONFIG.mesh.cells)")
    exit()
end
println("CASE_COMPLETE name=$(CASE_CONFIG.name) steps=$(result.steps) time=$(result.time)")
