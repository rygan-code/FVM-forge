# Structured solver task graph.
#
# This file intentionally has no GPU, MPI, or solver-state dependencies.  It
# provides the dependency contract used by the structured orchestration layer;
# numerical kernels remain owned by their existing modules.

@enum StructuredTaskResult::UInt8 begin
    StructuredTaskDone = 0
    StructuredTaskPending = 1
    StructuredTaskFailed = 2
end

@enum StructuredTaskNodeState::UInt8 begin
    StructuredTaskUnstarted = 0
    StructuredTaskRunning = 1
    StructuredTaskCompleted = 2
end

struct StructuredTaskEpoch
    step::Int
    stage::Int
    substage::Int
end

StructuredTaskEpoch(step::Integer, stage::Integer) =
    StructuredTaskEpoch(Int(step), Int(stage), 0)
StructuredTaskEpoch(step::Integer, stage::Integer, substage::Integer) =
    StructuredTaskEpoch(Int(step), Int(stage), Int(substage))

struct StructuredTaskFailure <: Exception
    task_id::Symbol
    message::String
end

Base.showerror(io::IO, err::StructuredTaskFailure) =
    print(io, "structured task '", err.task_id, "' failed: ", err.message)

struct StructuredTaskDeadlock <: Exception
    epoch::StructuredTaskEpoch
    pending_tasks::Vector{Symbol}
    blocked_resources::Dict{Symbol,Vector{Symbol}}
end

function Base.showerror(io::IO, err::StructuredTaskDeadlock)
    print(io, "structured task graph stalled at epoch ", err.epoch,
          "; pending tasks: ", join(string.(err.pending_tasks), ", "))
    isempty(err.blocked_resources) ||
        print(io, "; blocked resources: ", err.blocked_resources)
end

struct StructuredTaskGraphError <: Exception
    message::String
end

Base.showerror(io::IO, err::StructuredTaskGraphError) = print(io, err.message)

const _structured_task_noop_callback = (context, runtime, node) -> StructuredTaskDone
const _structured_task_noop_cleanup = (context, runtime, node) -> nothing

struct StructuredTaskNode
    id::Symbol
    dependencies::Vector{Symbol}
    reads::Vector{Symbol}
    writes::Vector{Symbol}
    exclusive_resources::Vector{Symbol}
    collective_sequence::Int
    start!::Function
    progress!::Function
    cleanup!::Function
end

mutable struct StructuredTaskGraphBuilder
    nodes::Vector{StructuredTaskNode}
    node_index::Dict{Symbol,Int}
    persistent_resources::Set{Symbol}
end

StructuredTaskGraphBuilder() = StructuredTaskGraphBuilder(
    StructuredTaskNode[], Dict{Symbol,Int}(), Set{Symbol}(),
)

function _structured_task_symbols(values)
    result = Symbol[]
    for value in values
        symbol = value isa Symbol ? value : Symbol(value)
        symbol in result || push!(result, symbol)
    end
    return result
end

function declare_structured_resource!(
    builder::StructuredTaskGraphBuilder,
    resource;
    persistent::Bool=false,
)
    key = resource isa Symbol ? resource : Symbol(resource)
    persistent && push!(builder.persistent_resources, key)
    return key
end

function add_structured_task!(
    builder::StructuredTaskGraphBuilder,
    id;
    depends=(),
    reads=(),
    writes=(),
    exclusive=(),
    collective_sequence::Integer=0,
    start! = _structured_task_noop_callback,
    progress! = _structured_task_noop_callback,
    cleanup! = _structured_task_noop_cleanup,
)
    task_id = id isa Symbol ? id : Symbol(id)
    haskey(builder.node_index, task_id) && throw(StructuredTaskGraphError(
        "duplicate structured task id: $task_id",
    ))

    dependencies = _structured_task_symbols(depends)
    reads = _structured_task_symbols(reads)
    writes = _structured_task_symbols(writes)
    exclusive = _structured_task_symbols(exclusive)
    isempty(intersect(reads, writes)) || throw(StructuredTaskGraphError(
        "task $task_id reads and writes the same resource; use separate " *
        "epoch resources to make the dependency explicit",
    ))
    collective_sequence >= 0 || throw(StructuredTaskGraphError(
        "task $task_id has a negative collective sequence",
    ))

    for resource in (reads..., writes..., exclusive...)
        declare_structured_resource!(builder, resource)
    end

    node = StructuredTaskNode(
        task_id, dependencies, reads, writes, exclusive,
        Int(collective_sequence), start!, progress!, cleanup!,
    )
    push!(builder.nodes, node)
    builder.node_index[task_id] = length(builder.nodes)
    return builder
end

struct StructuredTaskGraph
    nodes::Vector{StructuredTaskNode}
    node_index::Dict{Symbol,Int}
    execution_order::Vector{Int}
    dependents::Vector{Vector{Int}}
    persistent_resources::Set{Symbol}
    signature::UInt64
end

function _structured_task_stable_hash(text::AbstractString)
    value = UInt64(0xcbf29ce484222325)
    for byte in codeunits(text)
        value = (value ⊻ UInt64(byte)) * UInt64(0x100000001b3)
    end
    return value
end

function _structured_task_signature(nodes, order, persistent_resources)
    io = IOBuffer()
    for index in order
        node = nodes[index]
        print(io, node.id, '|', node.collective_sequence, '|')
        print(io, join(string.(sort(node.dependencies)), ','), '|')
        print(io, join(string.(sort(node.reads)), ','), '|')
        print(io, join(string.(sort(node.writes)), ','), '|')
        print(io, join(string.(sort(node.exclusive_resources)), ','), '\n')
    end
    print(io, "persistent|", join(string.(sort(collect(persistent_resources))), ','))
    return _structured_task_stable_hash(String(take!(io)))
end

function _structured_task_has_path(dependents, source::Int, target::Int)
    source == target && return true
    visited = falses(length(dependents))
    stack = Int[source]
    while !isempty(stack)
        current = pop!(stack)
        visited[current] && continue
        visited[current] = true
        for next in dependents[current]
            next == target && return true
            visited[next] || push!(stack, next)
        end
    end
    return false
end

function _structured_task_validate_conflicts!(nodes, dependents)
    read_sets = map(nodes) do node
        Set(node.reads)
    end
    write_sets = map(nodes) do node
        Set(vcat(node.writes, node.exclusive_resources))
    end
    for left in eachindex(nodes)
        for right in (left + 1):length(nodes)
            conflicts = union(
                intersect(write_sets[left], union(read_sets[right], write_sets[right])),
                intersect(write_sets[right], union(read_sets[left], write_sets[left])),
            )
            isempty(conflicts) && continue
            ordered = _structured_task_has_path(dependents, left, right) ||
                      _structured_task_has_path(dependents, right, left)
            ordered && continue

            # An explicit exclusive resource is sufficient for the first
            # synchronous scheduler, and will become a lock in async mode.
            shared_exclusive = intersect(
                nodes[left].exclusive_resources,
                nodes[right].exclusive_resources,
            )
            isempty(shared_exclusive) || continue

            throw(StructuredTaskGraphError(
                "unordered structured tasks '$(nodes[left].id)' and " *
                "share resources $(collect(conflicts)); " *
                "add a dependency or an exclusive resource",
            ))
        end
    end
end

function build_structured_task_graph(builder::StructuredTaskGraphBuilder)
    nodes = copy(builder.nodes)
    node_index = copy(builder.node_index)
    n = length(nodes)
    indegree = zeros(Int, n)
    dependents = [Int[] for _ in 1:n]

    for (index, node) in enumerate(nodes)
        for dependency in node.dependencies
            haskey(node_index, dependency) || throw(StructuredTaskGraphError(
                "task '$(node.id)' depends on missing task '$dependency'",
            ))
            source = node_index[dependency]
            push!(dependents[source], index)
            indegree[index] += 1
        end
    end

    # Kahn's algorithm with insertion-order tie breaking makes the graph
    # signature and synchronous execution deterministic on every rank.
    ready = Int[index for index in 1:n if indegree[index] == 0]
    order = Int[]
    while !isempty(ready)
        current = popfirst!(ready)
        push!(order, current)
        for next in dependents[current]
            indegree[next] -= 1
            indegree[next] == 0 && push!(ready, next)
        end
    end
    length(order) == n || throw(StructuredTaskGraphError(
        "structured task graph contains a dependency cycle",
    ))

    _structured_task_validate_conflicts!(nodes, dependents)
    signature = _structured_task_signature(nodes, order, builder.persistent_resources)
    return StructuredTaskGraph(
        nodes, node_index, order, dependents,
        copy(builder.persistent_resources), signature,
    )
end

mutable struct StructuredTaskResourceState
    ready_epoch::Union{Nothing,StructuredTaskEpoch}
    persistent::Bool
    producer::Union{Nothing,Symbol}
end

mutable struct StructuredTaskRuntime
    graph::StructuredTaskGraph
    epoch::StructuredTaskEpoch
    node_states::Vector{StructuredTaskNodeState}
    resources::Dict{Symbol,StructuredTaskResourceState}
    active_exclusive::Set{Symbol}
    last_collective_sequence::Int
    completed_count::Int
    context
end

function StructuredTaskRuntime(
    graph::StructuredTaskGraph;
    epoch::StructuredTaskEpoch=StructuredTaskEpoch(0, 0, 0),
    context=nothing,
)
    resources = Dict{Symbol,StructuredTaskResourceState}()
    for node in graph.nodes
        for resource in (node.reads..., node.writes..., node.exclusive_resources...)
            haskey(resources, resource) || (resources[resource] =
                StructuredTaskResourceState(nothing, resource in graph.persistent_resources, nothing))
        end
    end
    return StructuredTaskRuntime(
        graph, epoch,
        fill(StructuredTaskUnstarted, length(graph.nodes)),
        resources, Set{Symbol}(), 0, 0, context,
    )
end

function begin_structured_task_epoch!(
    runtime::StructuredTaskRuntime,
    epoch::StructuredTaskEpoch;
    context=runtime.context,
)
    runtime.epoch = epoch
    runtime.context = context
    fill!(runtime.node_states, StructuredTaskUnstarted)
    empty!(runtime.active_exclusive)
    runtime.last_collective_sequence = 0
    runtime.completed_count = 0
    for state in values(runtime.resources)
        state.persistent || (state.ready_epoch = nothing)
        state.producer = nothing
    end
    return runtime
end

function mark_structured_resource_ready!(
    runtime::StructuredTaskRuntime,
    resource;
    epoch::StructuredTaskEpoch=runtime.epoch,
    producer::Union{Nothing,Symbol}=nothing,
)
    key = resource isa Symbol ? resource : Symbol(resource)
    state = get!(runtime.resources, key) do
        StructuredTaskResourceState(nothing, key in runtime.graph.persistent_resources, nothing)
    end
    if !state.persistent && epoch != runtime.epoch
        throw(StructuredTaskGraphError(
            "resource '$key' was marked ready at stale epoch $epoch; " *
            "current epoch is $(runtime.epoch)",
        ))
    end
    state.ready_epoch = state.persistent ? epoch : runtime.epoch
    state.producer = producer
    return runtime
end

function structured_resource_ready(
    runtime::StructuredTaskRuntime,
    resource;
    epoch::StructuredTaskEpoch=runtime.epoch,
)
    key = resource isa Symbol ? resource : Symbol(resource)
    state = get(runtime.resources, key, nothing)
    state === nothing && return false
    state.ready_epoch === nothing && return false
    return state.persistent || state.ready_epoch == epoch
end

function _structured_task_dependencies_ready(runtime, index)
    for dependency in runtime.graph.nodes[index].dependencies
        dependency_index = runtime.graph.node_index[dependency]
        runtime.node_states[dependency_index] == StructuredTaskCompleted || return false
    end
    return true
end

function _structured_task_reads_ready(runtime, node)
    for resource in node.reads
        structured_resource_ready(runtime, resource) || return false
    end
    for resource in node.exclusive_resources
        resource in runtime.active_exclusive && return false
    end
    return true
end

function _structured_task_blocked_resources(runtime, index)
    node = runtime.graph.nodes[index]
    blocked = Symbol[]
    for resource in node.reads
        structured_resource_ready(runtime, resource) || push!(blocked, resource)
    end
    return blocked
end

function _structured_task_callback_result(result, task_id)
    result === nothing && return StructuredTaskDone
    result isa StructuredTaskResult || throw(StructuredTaskFailure(
        task_id, "callback returned $(typeof(result)); expected StructuredTaskResult",
    ))
    return result
end

function _structured_task_fail!(runtime, node, message)
    for index in eachindex(runtime.node_states)
        runtime.node_states[index] == StructuredTaskRunning || continue
        running = runtime.graph.nodes[index]
        try
            running.cleanup!(runtime.context, runtime, running)
        catch cleanup_error
            message *= "; cleanup for $(running.id) failed: $(sprint(showerror, cleanup_error))"
        end
    end
    throw(StructuredTaskFailure(node.id, message))
end

function _structured_task_complete!(runtime, index)
    node = runtime.graph.nodes[index]
    runtime.node_states[index] = StructuredTaskCompleted
    runtime.completed_count += 1
    for resource in node.exclusive_resources
        delete!(runtime.active_exclusive, resource)
    end
    for resource in node.writes
        mark_structured_resource_ready!(runtime, resource; producer=node.id)
    end
    if node.collective_sequence > 0
        node.collective_sequence < runtime.last_collective_sequence &&
            _structured_task_fail!(runtime, node, "collective sequence moved backwards")
        runtime.last_collective_sequence = node.collective_sequence
    end
end

function _structured_task_start!(runtime, index)
    node = runtime.graph.nodes[index]
    if node.collective_sequence > 0 &&
       node.collective_sequence < runtime.last_collective_sequence
        _structured_task_fail!(runtime, node, "collective sequence moved backwards")
    end
    for resource in node.exclusive_resources
        push!(runtime.active_exclusive, resource)
    end
    runtime.node_states[index] = StructuredTaskRunning
    result = try
        _structured_task_callback_result(node.start!(runtime.context, runtime, node), node.id)
    catch error
        _structured_task_fail!(runtime, node, sprint(showerror, error))
    end
    if result == StructuredTaskFailed
        _structured_task_fail!(runtime, node, "start callback returned failure")
    elseif result == StructuredTaskDone
        _structured_task_complete!(runtime, index)
    end
    return true
end

function _structured_task_progress!(runtime, index)
    node = runtime.graph.nodes[index]
    result = try
        _structured_task_callback_result(node.progress!(runtime.context, runtime, node), node.id)
    catch error
        _structured_task_fail!(runtime, node, sprint(showerror, error))
    end
    if result == StructuredTaskFailed
        _structured_task_fail!(runtime, node, "progress callback returned failure")
    elseif result == StructuredTaskDone
        _structured_task_complete!(runtime, index)
    end
    return true
end

function run_structured_task_graph!(
    runtime::StructuredTaskRuntime;
    max_passes::Integer=typemax(Int),
)
    passes = 0
    while runtime.completed_count < length(runtime.graph.nodes)
        passes += 1
        passes > max_passes && throw(StructuredTaskDeadlock(
            runtime.epoch,
            [runtime.graph.nodes[index].id for index in eachindex(runtime.node_states)
             if runtime.node_states[index] != StructuredTaskCompleted],
            Dict{Symbol,Vector{Symbol}}(),
        ))
        progress = false

        # Complete communication/event tasks that returned Pending before
        # starting more work.  Exclusive resources remain locked meanwhile.
        for index in runtime.graph.execution_order
            runtime.node_states[index] == StructuredTaskRunning || continue
            progress |= _structured_task_progress!(runtime, index)
        end

        for index in runtime.graph.execution_order
            runtime.node_states[index] == StructuredTaskUnstarted || continue
            _structured_task_dependencies_ready(runtime, index) || continue
            _structured_task_reads_ready(runtime, runtime.graph.nodes[index]) || continue
            progress |= _structured_task_start!(runtime, index)
        end

        runtime.completed_count == length(runtime.graph.nodes) && break
        progress && continue

        pending = Symbol[]
        blocked_resources = Dict{Symbol,Vector{Symbol}}()
        for index in runtime.graph.execution_order
            runtime.node_states[index] == StructuredTaskCompleted && continue
            push!(pending, runtime.graph.nodes[index].id)
            blocked = _structured_task_blocked_resources(runtime, index)
            isempty(blocked) || (blocked_resources[runtime.graph.nodes[index].id] = blocked)
        end
        throw(StructuredTaskDeadlock(runtime.epoch, pending, blocked_resources))
    end
    return runtime
end
