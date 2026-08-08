abstract type AbstractSpatialBackend end

struct StructuredBackend <: AbstractSpatialBackend end
struct UnstructuredBackend <: AbstractSpatialBackend end

function select_spatial_backend(mesh_type::Symbol)
    mesh_type == :structured && return StructuredBackend()
    mesh_type == :unstructured && return UnstructuredBackend()
    throw(ArgumentError(
        "mesh_type must be :structured or :unstructured, got $mesh_type",
    ))
end

spatial_backend_name(::StructuredBackend) = :structured
spatial_backend_name(::UnstructuredBackend) = :unstructured

function run_solver_backend!(::StructuredBackend, args...; kwargs...)
    return time_step(args...; kwargs...)
end

function run_solver_backend!(::UnstructuredBackend, args...; kwargs...)
    return unstruct_time_step(args...; kwargs...)
end

function prepare_backend_case end
function validate_backend_case! end
initialize_backend_case!(backend,state)=state
function advance_backend_case! end
write_backend_case!(backend,state,result)=result
finalize_backend_case!(backend,state)=nothing

function run_backend_case!(backend::AbstractSpatialBackend,config;
                           project_root::AbstractString,
                           validate_only::Bool=false)
    state=prepare_backend_case(backend,config,project_root)
    try
        validate_backend_case!(backend,state)
        validate_only && return (validated=true,state=state)
        initialize_backend_case!(backend,state)
        result=advance_backend_case!(backend,state)
        write_backend_case!(backend,state,result)
        return result
    finally
        finalize_backend_case!(backend,state)
    end
end
