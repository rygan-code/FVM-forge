# Runtime context for the structured task-graph scheduler.

mutable struct StructuredTaskContext
    graph::StructuredTaskGraph
    runtime::StructuredTaskRuntime
    trace::Vector{Symbol}
    blocks
    connectivity
    mpi_requests
    gpu_events
    shared_scratch
    runtime_options
end

function StructuredTaskContext(
    graph::StructuredTaskGraph;
    blocks=nothing,
    connectivity=nothing,
    mpi_requests=nothing,
    gpu_events=nothing,
    shared_scratch=nothing,
    runtime_options=NamedTuple(),
)
    context = StructuredTaskContext(
        graph, StructuredTaskRuntime(graph), Symbol[],
        blocks, connectivity, mpi_requests, gpu_events,
        shared_scratch, runtime_options,
    )
    context.runtime.context = context
    return context
end

function assert_structured_taskgraph_only!(raw=nothing)
    configured = if raw !== nothing
        raw
    elseif isdefined(Main, :structured_task_mode)
        getfield(Main, :structured_task_mode)
    else
        get(ENV, "OPENCFD_STRUCTURED_TASK_MODE", "taskgraph")
    end
    mode = configured isa Symbol ? configured :
        Symbol(lowercase(string(configured)))
    mode == :taskgraph || throw(ArgumentError(
        "the structured legacy scheduler has been removed; " *
        "OPENCFD_STRUCTURED_TASK_MODE must be taskgraph, got $mode",
    ))
    return nothing
end

function begin_structured_task_stage!(
    context::StructuredTaskContext,
    epoch::StructuredTaskEpoch;
    seed_resources=(),
)
    begin_structured_task_epoch!(context.runtime, epoch; context=context)
    empty!(context.trace)
    for resource in seed_resources
        mark_structured_resource_ready!(context.runtime, resource)
    end
    return context
end

function _structured_task_node_index(runtime::StructuredTaskRuntime, id)
    task_id = id isa Symbol ? id : Symbol(id)
    haskey(runtime.graph.node_index, task_id) || throw(StructuredTaskGraphError(
        "task '$task_id' is not present in the structured task graph",
    ))
    return runtime.graph.node_index[task_id]
end

function record_structured_task_completion!(context::StructuredTaskContext, id)
    runtime = context.runtime
    index = _structured_task_node_index(runtime, id)
    node = runtime.graph.nodes[index]
    runtime.node_states[index] == StructuredTaskUnstarted || throw(
        StructuredTaskGraphError(
            "task '$(node.id)' was recorded more than once or after execution",
        ),
    )
    _structured_task_dependencies_ready(runtime, index) || throw(
        StructuredTaskGraphError(
            "task '$(node.id)' was observed before its dependencies completed",
        ),
    )
    _structured_task_reads_ready(runtime, node) || throw(
        StructuredTaskGraphError(
            "task '$(node.id)' was observed before its input resources were ready",
        ),
    )
    for resource in node.exclusive_resources
        push!(runtime.active_exclusive, resource)
    end
    runtime.node_states[index] = StructuredTaskRunning
    _structured_task_complete!(runtime, index)
    push!(context.trace, node.id)
    return context
end

function structured_task_epoch_complete(context::StructuredTaskContext)
    return context.runtime.completed_count == length(context.graph.nodes)
end

function assert_structured_task_epoch_complete!(context::StructuredTaskContext)
    structured_task_epoch_complete(context) && return context
    pending = [
        context.graph.nodes[index].id
        for index in eachindex(context.runtime.node_states)
        if context.runtime.node_states[index] != StructuredTaskCompleted
    ]
    throw(StructuredTaskDeadlock(
        context.runtime.epoch, pending, Dict{Symbol,Vector{Symbol}}(),
    ))
end

function structured_task_graph_signature(context::StructuredTaskContext)
    return context.graph.signature
end
