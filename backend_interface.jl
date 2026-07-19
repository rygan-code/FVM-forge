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
