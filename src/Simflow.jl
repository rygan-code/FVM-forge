module Simflow

include(joinpath(@__DIR__,"core","case_config.jl"))
include(joinpath(@__DIR__,"parallel","device_selection.jl"))

export CaseConfig,ParallelCaseConfig,load_case_config,validate_case_config
export apply_case_environment!,case_config_document,write_effective_case_config
export resolve_requested_device,activate_requested_device!
export load_backend_api!,load_equation_config!
export load_structured_solver_stack!,load_unstructured_solver_stack!

const _loaded_stacks=IdDict{Module,Set{Symbol}}()

function _load_once!(target::Module,key::Symbol,paths)
    loaded=get!(_loaded_stacks,target,Set{Symbol}())
    key in loaded && return target
    for path in paths
        Base.include(target,path)
    end
    push!(loaded,key)
    return target
end

function load_backend_api!(target::Module=Main)
    return _load_once!(target,:backend_api,(
        joinpath(@__DIR__,"core","backend_interface.jl"),
        joinpath(@__DIR__,"time","legacy_structured_backend_lifecycle.jl"),
    ))
end

function load_equation_config!(target::Module=Main)
    return _load_once!(target,:equation_config,(
        joinpath(@__DIR__,"core","equation_config.jl"),
    ))
end

function load_structured_solver_stack!(target::Module=Main)
    load_backend_api!(target)
    load_equation_config!(target)
    return _load_once!(target,:structured_solver,(
        joinpath(@__DIR__,"time","structured_rk3_solver.jl"),
    ))
end

function load_unstructured_solver_stack!(target::Module=Main)
    load_backend_api!(target)
    return _load_once!(target,:unstructured_solver,(
        joinpath(@__DIR__,"parallel","gpu_backend.jl"),
        joinpath(@__DIR__,"core","boundary_types.jl"),
        joinpath(@__DIR__,"physics","euler_flux.jl"),
        joinpath(@__DIR__,"physics","mhd_flux.jl"),
        joinpath(@__DIR__,"mesh","unstructured_mesh.jl"),
        joinpath(@__DIR__,"parallel","unstructured_partition.jl"),
        joinpath(@__DIR__,"numerics","unstructured_ct.jl"),
        joinpath(@__DIR__,"numerics","unstructured_reconstruction.jl"),
        joinpath(@__DIR__,"numerics","unstructured_divergence.jl"),
        joinpath(@__DIR__,"physics","unstructured_boundary.jl"),
        joinpath(@__DIR__,"numerics","unstructured_gradients.jl"),
        joinpath(@__DIR__,"parallel","unstructured_mpi.jl"),
        joinpath(@__DIR__,"numerics","unstructured_viscous_flux.jl"),
        joinpath(@__DIR__,"io","unstructured_io.jl"),
        joinpath(@__DIR__,"time","unstructured_rk3_solver.jl"),
        joinpath(@__DIR__,"time","unstructured_backend_lifecycle.jl"),
    ))
end

end
