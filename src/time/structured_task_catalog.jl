# Declarative task catalog for the structured explicit ghost/CT transaction.

const STRUCTURED_TASK_U_ACTIVE = :U_ACTIVE
const STRUCTURED_TASK_U_FACE_HALO = :U_FACE_HALO
const STRUCTURED_TASK_U_PHYSICAL_GHOST = :U_PHYSICAL_GHOST
const STRUCTURED_TASK_U_RANK_HALO = :U_RANK_HALO
const STRUCTURED_TASK_U_HALO = :U_HALO
const STRUCTURED_TASK_U_ACTIVE_FILTERED = :U_ACTIVE_FILTERED
const STRUCTURED_TASK_U_FACE_HALO_FILTERED = :U_FACE_HALO_FILTERED
const STRUCTURED_TASK_Q_PROVISIONAL_FILTERED = :Q_PROVISIONAL_FILTERED
const STRUCTURED_TASK_U_PHYSICAL_GHOST_FILTERED = :U_PHYSICAL_GHOST_FILTERED
const STRUCTURED_TASK_U_RANK_HALO_FILTERED = :U_RANK_HALO_FILTERED
const STRUCTURED_TASK_U_HALO_FILTERED = :U_HALO_FILTERED
const STRUCTURED_TASK_FACE_B_HALO = :FACE_B_HALO
const STRUCTURED_TASK_Q_PROVISIONAL = :Q_PROVISIONAL
const STRUCTURED_TASK_Q_POINT = :Q_POINT
const STRUCTURED_TASK_Q_PHYSICAL = :Q_PHYSICAL
const STRUCTURED_TASK_Q_HALO = :Q_HALO
const STRUCTURED_TASK_CT_DERIVATION_HALO = :CT_DERIVATION_HALO
const STRUCTURED_TASK_CT_BACKGROUND_READY = :CT_BACKGROUND_READY
const STRUCTURED_TASK_POSITIVITY = :POSITIVITY

# End-of-step resources.  These are intentionally separate from the RK stage
# resources: filtering and I/O operate on the committed physical state, not on
# an intermediate SSP-RK state.
const STRUCTURED_TASK_POST_STATE = :POST_STATE
const STRUCTURED_TASK_POST_FILTER_X = :POST_FILTER_X
const STRUCTURED_TASK_POST_FILTER_Y = :POST_FILTER_Y
const STRUCTURED_TASK_POST_FILTER_Z = :POST_FILTER_Z
const STRUCTURED_TASK_POST_AVERAGE = :POST_AVERAGE
const STRUCTURED_TASK_POST_DIAGNOSTIC = :POST_DIAGNOSTIC
const STRUCTURED_TASK_POST_PLOT = :POST_PLOT
const STRUCTURED_TASK_POST_CHECKPOINT = :POST_CHECKPOINT
const STRUCTURED_TASK_POST_AVERAGE_FILE = :POST_AVERAGE_FILE
const STRUCTURED_TASK_POST_INSITU = :POST_INSITU
const STRUCTURED_TASK_POST_MAINTENANCE = :POST_MAINTENANCE

# Resistive and implicit resources are part of the catalog even when the
# corresponding provider is disabled.  Keeping their names centralized avoids
# accidentally reusing an RK resource for a split or dual-time update.
const STRUCTURED_TASK_RESISTIVE_PREPARED = :RESISTIVE_PREPARED
const STRUCTURED_TASK_RESISTIVE_EMF = :RESISTIVE_EMF
const STRUCTURED_TASK_RESISTIVE_EDGE_READY = :RESISTIVE_EDGE_READY
const STRUCTURED_TASK_RESISTIVE_STATE = :RESISTIVE_STATE
const STRUCTURED_TASK_RESISTIVE_Q = :RESISTIVE_Q
const STRUCTURED_TASK_IMPLICIT_PREPARED = :IMPLICIT_PREPARED
const STRUCTURED_TASK_IMPLICIT_DT = :IMPLICIT_DT
const STRUCTURED_TASK_IMPLICIT_FORCING = :IMPLICIT_FORCING
const STRUCTURED_TASK_IMPLICIT_SHOCK = :IMPLICIT_SHOCK
const STRUCTURED_TASK_IMPLICIT_RESIDUAL = :IMPLICIT_RESIDUAL
const STRUCTURED_TASK_IMPLICIT_UPDATE = :IMPLICIT_UPDATE
const STRUCTURED_TASK_IMPLICIT_HALO = :IMPLICIT_HALO
const STRUCTURED_TASK_IMPLICIT_CONVERGED = :IMPLICIT_CONVERGED
const STRUCTURED_TASK_IMPLICIT_COMMITTED = :IMPLICIT_COMMITTED

function build_structured_ghost_ct_task_graph(;
    ct_mode::Bool=true,
    point6::Bool=false,
    interface_filter::Bool=false,
    callbacks=Dict{Symbol,Function}(),
)
    callback(id) = get(callbacks, id, _structured_task_noop_callback)
    builder = StructuredTaskGraphBuilder()
    declare_structured_resource!(builder, STRUCTURED_TASK_U_ACTIVE)
    declare_structured_resource!(builder, STRUCTURED_TASK_FACE_B_HALO)

    add_structured_task!(builder, :u_interblock_face_copy;
        reads=[STRUCTURED_TASK_U_ACTIVE],
        writes=[STRUCTURED_TASK_U_FACE_HALO],
        start! = callback(:u_interblock_face_copy))
    add_structured_task!(builder, :ct_provisional_state;
        reads=ct_mode ?
            [STRUCTURED_TASK_U_ACTIVE, STRUCTURED_TASK_FACE_B_HALO] :
            [STRUCTURED_TASK_U_ACTIVE],
        writes=[STRUCTURED_TASK_Q_PROVISIONAL],
        start! = callback(:ct_provisional_state))
    add_structured_task!(builder, :u_physical_boundary;
        depends=[:u_interblock_face_copy, :ct_provisional_state],
        reads=[STRUCTURED_TASK_Q_PROVISIONAL],
        writes=[STRUCTURED_TASK_U_PHYSICAL_GHOST],
        start! = callback(:u_physical_boundary))
    add_structured_task!(builder, :u_rank_exchange;
        depends=[:u_physical_boundary],
        reads=[STRUCTURED_TASK_U_PHYSICAL_GHOST],
        writes=[STRUCTURED_TASK_U_RANK_HALO],
        exclusive=[:MPI_REQUESTS],
        collective_sequence=1,
        start! = callback(:u_rank_exchange))
    add_structured_task!(builder, :u_interblock_full_copy;
        depends=[:u_rank_exchange],
        reads=[STRUCTURED_TASK_U_RANK_HALO],
        writes=[STRUCTURED_TASK_U_HALO],
        start! = callback(:u_interblock_full_copy))

    if interface_filter
        add_structured_task!(builder, :interface_filter;
            depends=[:u_interblock_full_copy],
            reads=ct_mode ?
                [STRUCTURED_TASK_U_HALO, STRUCTURED_TASK_FACE_B_HALO] :
                [STRUCTURED_TASK_U_HALO],
            writes=[STRUCTURED_TASK_U_ACTIVE_FILTERED],
            exclusive=[:INTERFACE_FILTER_SCRATCH],
            start! = callback(:interface_filter))
        add_structured_task!(builder, :filtered_u_interblock_face_copy;
            depends=[:interface_filter],
            reads=[STRUCTURED_TASK_U_ACTIVE_FILTERED],
            writes=[STRUCTURED_TASK_U_FACE_HALO_FILTERED],
            start! = callback(:filtered_u_interblock_face_copy))
        add_structured_task!(builder, :filtered_ct_provisional_state;
            depends=[:interface_filter],
            reads=ct_mode ?
                [STRUCTURED_TASK_U_ACTIVE_FILTERED,
                 STRUCTURED_TASK_FACE_B_HALO] :
                [STRUCTURED_TASK_U_ACTIVE_FILTERED],
            writes=[STRUCTURED_TASK_Q_PROVISIONAL_FILTERED],
            start! = callback(:filtered_ct_provisional_state))
        add_structured_task!(builder, :filtered_u_physical_boundary;
            depends=[:filtered_u_interblock_face_copy,
                     :filtered_ct_provisional_state],
            reads=[STRUCTURED_TASK_Q_PROVISIONAL_FILTERED],
            writes=[STRUCTURED_TASK_U_PHYSICAL_GHOST_FILTERED],
            start! = callback(:filtered_u_physical_boundary))
        add_structured_task!(builder, :filtered_u_rank_exchange;
            depends=[:filtered_u_physical_boundary],
            reads=[STRUCTURED_TASK_U_PHYSICAL_GHOST_FILTERED],
            writes=[STRUCTURED_TASK_U_RANK_HALO_FILTERED],
            exclusive=[:MPI_REQUESTS],
            collective_sequence=2,
            start! = callback(:filtered_u_rank_exchange))
        add_structured_task!(builder, :filtered_u_interblock_full_copy;
            depends=[:filtered_u_rank_exchange],
            reads=[STRUCTURED_TASK_U_RANK_HALO_FILTERED],
            writes=[STRUCTURED_TASK_U_HALO_FILTERED],
            start! = callback(:filtered_u_interblock_full_copy))
        final_u_dependency = :filtered_u_interblock_full_copy
        final_u_resource = STRUCTURED_TASK_U_HALO_FILTERED
    else
        final_u_dependency = :u_interblock_full_copy
        final_u_resource = STRUCTURED_TASK_U_HALO
    end

    if ct_mode
        final_state = point6 ? :ct_point6_state : :ct_direct_state
        add_structured_task!(builder, final_state;
            depends=[final_u_dependency],
            reads=[final_u_resource, STRUCTURED_TASK_FACE_B_HALO],
            writes=[STRUCTURED_TASK_Q_POINT,
                    STRUCTURED_TASK_CT_DERIVATION_HALO],
            exclusive=[:MPI_REQUESTS, :CT_DERIVATION_SCRATCH],
            start! = callback(final_state))
        add_structured_task!(builder, :ct_physical_state;
            depends=[final_state],
            reads=[STRUCTURED_TASK_Q_POINT, final_u_resource],
            writes=[STRUCTURED_TASK_Q_PHYSICAL],
            start! = callback(:ct_physical_state))
        add_structured_task!(builder, :ct_positivity;
            depends=[:ct_physical_state],
            reads=[STRUCTURED_TASK_Q_PHYSICAL, final_u_resource],
            writes=[STRUCTURED_TASK_POSITIVITY],
            start! = callback(:ct_positivity))
    else
        add_structured_task!(builder, :primitive_halo;
            depends=[final_u_dependency],
            reads=[final_u_resource],
            writes=[STRUCTURED_TASK_Q_HALO],
            start! = callback(:primitive_halo))
    end
    return build_structured_task_graph(builder)
end

# Build the committed-state tail of one physical time step.  The callbacks
# decide whether a feature is active for this step, so all ranks keep the same
# graph signature even when output or filtering is disabled by configuration.
function build_structured_post_step_task_graph(
    ; callbacks=Dict{Symbol,Function}(),
)
    callback(id) = get(callbacks, id, _structured_task_noop_callback)
    builder = StructuredTaskGraphBuilder()
    declare_structured_resource!(builder, STRUCTURED_TASK_POST_STATE)

    add_structured_task!(builder, :post_filter_x;
        reads=[STRUCTURED_TASK_POST_STATE],
        writes=[STRUCTURED_TASK_POST_FILTER_X],
        exclusive=[:POST_FILTER_SCRATCH],
        start! = callback(:post_filter_x))
    add_structured_task!(builder, :post_filter_y;
        depends=[:post_filter_x],
        reads=[STRUCTURED_TASK_POST_FILTER_X],
        writes=[STRUCTURED_TASK_POST_FILTER_Y],
        exclusive=[:POST_FILTER_SCRATCH],
        start! = callback(:post_filter_y))
    add_structured_task!(builder, :post_filter_z;
        depends=[:post_filter_y],
        reads=[STRUCTURED_TASK_POST_FILTER_Y],
        writes=[STRUCTURED_TASK_POST_FILTER_Z],
        exclusive=[:POST_FILTER_SCRATCH],
        start! = callback(:post_filter_z))
    add_structured_task!(builder, :post_average_sample;
        depends=[:post_filter_z],
        reads=[STRUCTURED_TASK_POST_FILTER_Z],
        writes=[STRUCTURED_TASK_POST_AVERAGE],
        start! = callback(:post_average_sample))
    add_structured_task!(builder, :post_diagnostic;
        depends=[:post_average_sample],
        reads=[STRUCTURED_TASK_POST_AVERAGE],
        writes=[STRUCTURED_TASK_POST_DIAGNOSTIC],
        collective_sequence=1,
        start! = callback(:post_diagnostic))
    add_structured_task!(builder, :post_plot_output;
        depends=[:post_diagnostic],
        reads=[STRUCTURED_TASK_POST_DIAGNOSTIC],
        writes=[STRUCTURED_TASK_POST_PLOT],
        exclusive=[:POST_IO],
        collective_sequence=2,
        start! = callback(:post_plot_output))
    add_structured_task!(builder, :post_checkpoint_output;
        depends=[:post_plot_output],
        reads=[STRUCTURED_TASK_POST_PLOT],
        writes=[STRUCTURED_TASK_POST_CHECKPOINT],
        exclusive=[:POST_IO],
        collective_sequence=3,
        start! = callback(:post_checkpoint_output))
    add_structured_task!(builder, :post_average_output;
        depends=[:post_checkpoint_output],
        reads=[STRUCTURED_TASK_POST_CHECKPOINT],
        writes=[STRUCTURED_TASK_POST_AVERAGE_FILE],
        exclusive=[:POST_IO],
        collective_sequence=4,
        start! = callback(:post_average_output))
    add_structured_task!(builder, :post_in_situ;
        depends=[:post_average_output],
        reads=[STRUCTURED_TASK_POST_AVERAGE_FILE],
        writes=[STRUCTURED_TASK_POST_INSITU],
        collective_sequence=5,
        start! = callback(:post_in_situ))
    add_structured_task!(builder, :post_maintenance;
        depends=[:post_in_situ],
        reads=[STRUCTURED_TASK_POST_INSITU],
        writes=[STRUCTURED_TASK_POST_MAINTENANCE],
        start! = callback(:post_maintenance))
    return build_structured_task_graph(builder)
end

# A time-limit exit may require one final plot between normal output cadence
# points.  Keep that I/O transaction separate so state-mutating post-step work
# (filtering, averaging, outlet control, and in-situ hooks) executes once only.
function build_structured_plot_output_task_graph(
    ; callbacks=Dict{Symbol,Function}(),
)
    callback(id) = get(callbacks, id, _structured_task_noop_callback)
    builder = StructuredTaskGraphBuilder()
    declare_structured_resource!(builder, STRUCTURED_TASK_POST_STATE)
    add_structured_task!(builder, :post_plot_output;
        reads=[STRUCTURED_TASK_POST_STATE],
        writes=[STRUCTURED_TASK_POST_PLOT],
        exclusive=[:POST_IO],
        collective_sequence=1,
        start! = callback(:post_plot_output))
    return build_structured_task_graph(builder)
end

# A provider-level resistive graph.  The stage count is known after the
# explicit resistive stability estimate, so each STS/RKL2 substep is a real
# graph node rather than a hidden loop in one callback.
function build_structured_resistive_task_graph(
    ; stage_count::Integer=1, callbacks=Dict{Symbol,Function}(),
)
    stage_count >= 1 || throw(ArgumentError(
        "resistive task graph requires at least one stage",
    ))
    callback(id, fallback=nothing) = get(
        callbacks, id,
        fallback === nothing ? _structured_task_noop_callback :
        get(callbacks, fallback, _structured_task_noop_callback),
    )
    builder = StructuredTaskGraphBuilder()
    declare_structured_resource!(builder, :RK_COMMITTED_STATE)
    add_structured_task!(builder, :resistive_prepare;
        reads=[:RK_COMMITTED_STATE],
        writes=[STRUCTURED_TASK_RESISTIVE_PREPARED],
        start! = callback(:resistive_prepare))
    previous_id = :resistive_prepare
    previous_resource = STRUCTURED_TASK_RESISTIVE_PREPARED
    for stage in 1:Int(stage_count)
        stage_id = Symbol("resistive_stage_", stage)
        stage_resource = Symbol("RESISTIVE_STAGE_", stage)
        add_structured_task!(builder, stage_id;
            depends=[previous_id],
            reads=[previous_resource],
            writes=[stage_resource],
            exclusive=[:RESISTIVE_CT_STATE],
            collective_sequence=stage,
            start! = callback(:resistive_stage, :resistive_advance))
        previous_id = stage_id
        previous_resource = stage_resource
    end
    add_structured_task!(builder, :resistive_reconcile;
        depends=[previous_id],
        reads=[previous_resource],
        writes=[STRUCTURED_TASK_RESISTIVE_Q],
        exclusive=[:RESISTIVE_CT_STATE],
        collective_sequence=Int(stage_count) + 1,
        start! = callback(:resistive_reconcile))
    add_structured_task!(builder, :resistive_positivity;
        depends=[:resistive_reconcile],
        reads=[STRUCTURED_TASK_RESISTIVE_Q],
        writes=[:RESISTIVE_POSITIVITY],
        collective_sequence=Int(stage_count) + 2,
        start! = callback(:resistive_positivity))
    return build_structured_task_graph(builder)
end

# Provider-level implicit graph.  The maximum inner iteration count is known
# from the case configuration.  Convergence can terminate the numerical work
# early, while the remaining graph nodes become explicit no-op barriers; this
# keeps the resource/collective order identical on every rank.
function build_structured_implicit_task_graph(
    ; inner_iterations::Integer=1, callbacks=Dict{Symbol,Function}(),
)
    inner_iterations >= 1 || throw(ArgumentError(
        "implicit task graph requires at least one inner iteration",
    ))
    callback(id, fallback=nothing) = get(
        callbacks, id,
        fallback === nothing ? _structured_task_noop_callback :
        get(callbacks, fallback, _structured_task_noop_callback),
    )
    builder = StructuredTaskGraphBuilder()
    declare_structured_resource!(builder, :IMPLICIT_STATE)
    add_structured_task!(builder, :implicit_dt;
        reads=[:IMPLICIT_STATE],
        writes=[STRUCTURED_TASK_IMPLICIT_DT],
        start! = callback(:implicit_dt))
    add_structured_task!(builder, :implicit_forcing;
        depends=[:implicit_dt],
        reads=[STRUCTURED_TASK_IMPLICIT_DT],
        writes=[STRUCTURED_TASK_IMPLICIT_FORCING],
        collective_sequence=1,
        start! = callback(:implicit_forcing))
    add_structured_task!(builder, :implicit_shock;
        depends=[:implicit_forcing],
        reads=[STRUCTURED_TASK_IMPLICIT_FORCING],
        writes=[STRUCTURED_TASK_IMPLICIT_SHOCK],
        exclusive=[:IMPLICIT_SHOCK_MPI],
        collective_sequence=2,
        start! = callback(:implicit_shock))
    add_structured_task!(builder, :implicit_prepare;
        depends=[:implicit_shock],
        reads=[STRUCTURED_TASK_IMPLICIT_SHOCK],
        writes=[STRUCTURED_TASK_IMPLICIT_PREPARED],
        start! = callback(:implicit_prepare))
    previous_id = :implicit_prepare
    previous_resource = STRUCTURED_TASK_IMPLICIT_PREPARED
    for iteration in 1:Int(inner_iterations)
        residual_id = Symbol("implicit_residual_", iteration)
        update_id = Symbol("implicit_update_", iteration)
        halo_id = Symbol("implicit_halo_", iteration)
        convergence_id = Symbol("implicit_convergence_", iteration)
        residual_resource = Symbol("IMPLICIT_RESIDUAL_", iteration)
        update_resource = Symbol("IMPLICIT_UPDATE_", iteration)
        halo_resource = Symbol("IMPLICIT_HALO_", iteration)
        convergence_resource = Symbol("IMPLICIT_CONVERGED_", iteration)

        add_structured_task!(builder, residual_id;
            depends=[previous_id],
            reads=[previous_resource],
            writes=[residual_resource],
            exclusive=[:IMPLICIT_WORKSPACE],
            start! = callback(:implicit_residual))
        add_structured_task!(builder, update_id;
            depends=[residual_id],
            reads=[residual_resource],
            writes=[update_resource],
            exclusive=[:IMPLICIT_WORKSPACE],
            start! = callback(:implicit_update))
        add_structured_task!(builder, halo_id;
            depends=[update_id],
            reads=[update_resource],
            writes=[halo_resource],
            exclusive=[:MPI_REQUESTS],
            collective_sequence=2 * iteration + 2,
            start! = callback(:implicit_halo))
        add_structured_task!(builder, convergence_id;
            depends=[halo_id],
            reads=[halo_resource],
            writes=[convergence_resource],
            collective_sequence=2 * iteration + 3,
            start! = callback(:implicit_convergence))
        previous_id = convergence_id
        previous_resource = convergence_resource
    end
    add_structured_task!(builder, :implicit_commit;
        depends=[previous_id],
        reads=[previous_resource],
        writes=[STRUCTURED_TASK_IMPLICIT_COMMITTED],
        collective_sequence=2 * Int(inner_iterations) + 4,
        start! = callback(:implicit_commit))
    return build_structured_task_graph(builder)
end

# Build the explicit RK3 orchestration graph. Flux work buffers are shared by
# blocks on one device, so the graph serializes each block's point-flux,
# face-average, diffusive-flux, source, and divergence lifetime. CT edge EMF
# construction branches from the same point Riemann result.
function build_structured_explicit_rk3_task_graph(
    block_ids;
    ct_mode::Bool=false,
    background_split::Bool=false,
    troubled_mask::Bool=false,
    fofc::Bool=false,
    fofc_iterations::Integer=1,
    resistive_pre::Bool=false,
    defer_trailing_split::Bool=false,
    callbacks=Dict{Symbol,Function}(),
)
    ids = sort!(collect(Int.(block_ids)))
    isempty(ids) && throw(ArgumentError(
        "explicit RK3 task graph requires at least one block",
    ))
    background_split && !ct_mode && throw(ArgumentError(
        "CT background splitting requires ct_mode=true",
    ))
    fofc && !ct_mode && throw(ArgumentError(
        "first-order flux correction requires the CT task graph",
    ))
    troubled_mask && !ct_mode && throw(ArgumentError(
        "the troubled-state path requires the CT task graph",
    ))
    fofc && fofc_iterations < 1 && throw(ArgumentError(
        "FOFC requires at least one corrected-candidate closure pass",
    ))
    fofc_iterations_int = Int(fofc_iterations)

    callback(id) = get(callbacks, id, _structured_task_noop_callback)
    builder = StructuredTaskGraphBuilder()

    # The initial halo is supplied by the caller after the previous physical
    # stage.  Every following stage produces a new epoch-specific halo.
    initial_q = :RK_Q_HALO_0
    initial_u = :RK_U_ACTIVE_0
    initial_face = :RK_FACE_B_0
    declare_structured_resource!(builder, initial_q)
    declare_structured_resource!(builder, initial_u)
    ct_mode && declare_structured_resource!(builder, initial_face)
    background_split && declare_structured_resource!(
        builder, STRUCTURED_TASK_CT_BACKGROUND_READY; persistent=true,
    )

    previous_barrier = nothing
    for stage in 1:3
        q_in = Symbol("RK_Q_HALO_", stage - 1)
        u_in = Symbol("RK_U_ACTIVE_", stage - 1)
        q_out = Symbol("RK_Q_HALO_", stage)
        u_out = Symbol("RK_U_ACTIVE_", stage)
        declare_structured_resource!(builder, q_in)
        declare_structured_resource!(builder, u_in)
        declare_structured_resource!(builder, q_out)
        declare_structured_resource!(builder, u_out)

        stage_seed = Symbol("rk", stage, "_stage_seed")
        seed_dep = previous_barrier === nothing ? Symbol[] : [previous_barrier]
        seed_reads = Symbol[q_in, u_in]
        if ct_mode
            face_in = stage == 1 ? initial_face :
                Symbol("RK_FACE_B_READY_", stage - 1)
            declare_structured_resource!(builder, face_in)
            push!(seed_reads, face_in)
        end
        if stage == 1
            add_structured_task!(builder, stage_seed;
                depends=seed_dep,
                reads=seed_reads,
                writes=[Symbol("RK_STEP_PREPARED")],
                start! = callback(:rk_prepare),
            )
        else
            # Later RK stages consume the time-level state produced by the
            # previous stage; the callback is still explicit in the graph so
            # stage-local diagnostics can attach to it.
            add_structured_task!(builder, stage_seed;
                depends=seed_dep,
                reads=seed_reads,
                writes=[Symbol("RK_STAGE_", stage, "_READY")],
                start! = callback(:rk_stage_prepare),
            )
        end

        shock = Symbol("rk", stage, "_shock")
        shock_halo = Symbol("rk", stage, "_shock_halo")
        add_structured_task!(builder, shock;
            depends=[stage_seed], reads=[q_in], writes=[
                Symbol("RK_SHOCK_", stage),
            ], start! = callback(:rk_shock))
        add_structured_task!(builder, shock_halo;
            depends=[shock], reads=[Symbol("RK_SHOCK_", stage)], writes=[
                Symbol("RK_SHOCK_HALO_", stage),
            ], exclusive=[:RK_SHOCK_MPI], collective_sequence=stage,
            start! = callback(:rk_shock_halo))

        q_reconstruction = q_in
        pre_barrier = shock_halo
        if troubled_mask
            troubled_state = Symbol("rk",stage,"_troubled_state")
            q_reconstruction = Symbol("RK_Q_RECON_",stage)
            add_structured_task!(builder,troubled_state;
                depends=[shock_halo],
                reads=[q_in,u_in,face_in,Symbol("RK_SHOCK_HALO_",stage)],
                writes=[q_reconstruction],
                exclusive=[:CT_DERIVATION_SCRATCH],
                start! = callback(:rk_troubled_state))
            pre_barrier = troubled_state
        end
        pre_resources = Symbol[]
        if ct_mode && resistive_pre
            for bid in ids
                pre_id = Symbol("rk", stage, "_resistive_pre_b", bid)
                pre_resource = Symbol("RK_RESISTIVE_PRE_", stage, "_B", bid)
                push!(pre_resources, pre_resource)
                add_structured_task!(builder, pre_id;
                    depends=[pre_barrier],
                    reads=[q_reconstruction], writes=[pre_resource],
                    exclusive=[:RK_SHARED_FLUX],
                    start! = callback(:rk_resistive_pre))
            end
            pre_barrier = Symbol("rk", stage, "_resistive_pre_barrier")
            add_structured_task!(builder, pre_barrier;
                depends=Symbol[Symbol("rk", stage, "_resistive_pre_b", bid)
                               for bid in ids],
                reads=pre_resources,
                writes=[Symbol("RK_RESISTIVE_PRE_READY_", stage)],
                collective_sequence=stage,
                start! = callback(:rk_resistive_pre_barrier))
        end

        final_pre_barrier = pre_barrier
        if ct_mode && fofc
            for iteration in 0:fofc_iterations_int
                prefix = Symbol("rk",stage,"_fofc_iter",iteration)
                resource_prefix =
                    Symbol("RK_FOFC_",stage,"_I",iteration)
                flag_input = iteration == 0 ? nothing :
                    Symbol("RK_FOFC_FLAG_HALO_",stage,"_I",iteration-1)
                iteration_begin = Symbol(prefix,"_begin")
                    begin_reads = Symbol[q_reconstruction,u_in,face_in]
                flag_input === nothing || push!(begin_reads,flag_input)
                add_structured_task!(builder,iteration_begin;
                    depends=[final_pre_barrier],reads=begin_reads,
                    writes=[Symbol(resource_prefix,"_STARTED")],
                    start! = callback(:rk_fofc_iteration_begin))

                # Phase A retains one edge array per block, then canonicalizes
                # those edges before any candidate cell is classified. Shared
                # face-flux buffers are serialized and may be overwritten.
                edge_ids = Symbol[]
                edge_resources = Symbol[]
                payload_resources = Symbol[]
                previous_edge = nothing
                for bid in ids
                    edge_flux = Symbol(prefix,"_edge_flux_b",bid)
                    edge_average = Symbol(prefix,"_edge_average_b",bid)
                    edge_diffusive = Symbol(prefix,"_edge_diffusive_b",bid)
                    edge_correct = Symbol(prefix,"_edge_correct_b",bid)
                    edge_emf = Symbol(prefix,"_edge_emf_b",bid)
                    point_resource =
                        Symbol(resource_prefix,"_EDGE_POINT_B",bid)
                    average_resource =
                        Symbol(resource_prefix,"_EDGE_AVERAGE_B",bid)
                    diffusive_resource =
                        Symbol(resource_prefix,"_EDGE_DIFFUSIVE_B",bid)
                    corrected_resource =
                        Symbol(resource_prefix,"_EDGE_CORRECTED_B",bid)
                    edge_resource = Symbol(resource_prefix,"_EDGE_B",bid)
                    payload_resource =
                        Symbol(resource_prefix,"_PAYLOAD_B",bid)
                    edge_dep = Symbol[iteration_begin]
                    previous_edge === nothing ||
                        push!(edge_dep,previous_edge)
                    edge_flux_reads = [
                        q_reconstruction, Symbol("RK_SHOCK_HALO_",stage),
                    ]
                    background_split && push!(
                        edge_flux_reads, STRUCTURED_TASK_CT_BACKGROUND_READY,
                    )
                    add_structured_task!(builder,edge_flux;
                        depends=edge_dep,
                        reads=edge_flux_reads,
                        writes=[point_resource],exclusive=[:RK_SHARED_FLUX],
                        start! = callback(:rk_flux))
                    add_structured_task!(builder,edge_average;
                        depends=[edge_flux],reads=[point_resource],
                        writes=[average_resource],exclusive=[:RK_SHARED_FLUX],
                        start! = callback(:rk_face_average))
                    final_average_id = edge_average
                    final_average_resource = average_resource
                    if iteration > 0
                        add_structured_task!(builder,edge_correct;
                            depends=[edge_average],
                            reads=[average_resource,flag_input],
                            writes=[corrected_resource],
                            exclusive=[:RK_SHARED_FLUX],
                            start! = callback(:rk_fofc_correct))
                        final_average_id = edge_correct
                        final_average_resource = corrected_resource
                    end
                    add_structured_task!(builder,edge_diffusive;
                        depends=[edge_average],
                        reads=[q_reconstruction,average_resource],
                        writes=[diffusive_resource],
                        exclusive=[:RK_SHARED_FLUX],
                        start! = callback(:rk_diffusive_flux))
                    add_structured_task!(builder,edge_emf;
                        depends=[final_average_id,edge_diffusive],
                        reads=[final_average_resource,diffusive_resource],
                        writes=[edge_resource,payload_resource],
                        exclusive=[:RK_SHARED_FLUX],
                        start! = callback(:rk_edge_emf))
                    push!(edge_ids,edge_emf)
                    push!(edge_resources,edge_resource)
                    push!(payload_resources,payload_resource)
                    previous_edge = edge_emf
                end

                edge_sync = Symbol(prefix,"_edge_sync")
                canonical_resource =
                    Symbol(resource_prefix,"_EDGE_CANONICAL")
                add_structured_task!(builder,edge_sync;
                    depends=edge_ids,reads=edge_resources,
                    writes=[canonical_resource],exclusive=[:RK_CT_MPI],
                    collective_sequence=stage,
                    start! = callback(:rk_edge_sync))
                junction_solve = Symbol(prefix,"_junction_solve")
                junction_resource =
                    Symbol(resource_prefix,"_EDGE_JUNCTION_READY")
                add_structured_task!(builder,junction_solve;
                    depends=[edge_sync],
                    reads=[canonical_resource,payload_resources...],
                    writes=[junction_resource],exclusive=[:RK_CT_MPI],
                    collective_sequence=stage,
                    start! = callback(:rk_junction_solve))

                # Phase B rebuilds the shared face flux for each block and
                # combines it with the now-canonical edge candidate. Flags are
                # monotone, so each pass can only enlarge the corrected set.
                commit_ids = Symbol[]
                local_flag_resources = Symbol[]
                previous_detect = nothing
                for bid in ids
                    detect_flux = Symbol(prefix,"_detect_flux_b",bid)
                    detect_average = Symbol(prefix,"_detect_average_b",bid)
                    detect_diffusive =
                        Symbol(prefix,"_detect_diffusive_b",bid)
                    detect_correct = Symbol(prefix,"_detect_correct_b",bid)
                    detect_scale = Symbol(prefix,"_detect_scale_b",bid)
                    detect = Symbol(prefix,"_detect_b",bid)
                    detect_commit = Symbol(prefix,"_commit_b",bid)
                    point_resource =
                        Symbol(resource_prefix,"_DETECT_POINT_B",bid)
                    average_resource =
                        Symbol(resource_prefix,"_DETECT_AVERAGE_B",bid)
                    diffusive_resource =
                        Symbol(resource_prefix,"_DETECT_DIFFUSIVE_B",bid)
                    corrected_resource =
                        Symbol(resource_prefix,"_DETECT_CORRECTED_B",bid)
                    scaled_resource =
                        Symbol(resource_prefix,"_DETECT_SCALED_B",bid)
                    proposal_resource =
                        Symbol(resource_prefix,"_FLAG_PROPOSAL_B",bid)
                    flag_resource =
                        Symbol(resource_prefix,"_FLAG_LOCAL_B",bid)
                    detect_dep = Symbol[junction_solve]
                    previous_detect === nothing ||
                        push!(detect_dep,previous_detect)
                    detect_flux_reads = [
                        q_reconstruction, Symbol("RK_SHOCK_HALO_",stage),
                    ]
                    background_split && push!(
                        detect_flux_reads, STRUCTURED_TASK_CT_BACKGROUND_READY,
                    )
                    add_structured_task!(builder,detect_flux;
                        depends=detect_dep,
                        reads=detect_flux_reads,
                        writes=[point_resource],exclusive=[:RK_SHARED_FLUX],
                        start! = callback(:rk_flux))
                    add_structured_task!(builder,detect_average;
                        depends=[detect_flux],reads=[point_resource],
                        writes=[average_resource],exclusive=[:RK_SHARED_FLUX],
                        start! = callback(:rk_face_average))
                    final_detect_id = detect_average
                    final_detect_resource = average_resource
                    if iteration > 0
                        add_structured_task!(builder,detect_correct;
                            depends=[detect_average],
                            reads=[average_resource,flag_input],
                            writes=[corrected_resource],
                            exclusive=[:RK_SHARED_FLUX],
                            start! = callback(:rk_fofc_correct))
                        final_detect_id = detect_correct
                        final_detect_resource = corrected_resource
                    end
                    add_structured_task!(builder,detect_diffusive;
                        depends=[detect_average],
                        reads=[q_reconstruction,average_resource],
                        writes=[diffusive_resource],
                        exclusive=[:RK_SHARED_FLUX],
                        start! = callback(:rk_diffusive_flux))
                    if iteration > 0
                        add_structured_task!(builder,detect_scale;
                            depends=[final_detect_id,detect_diffusive],
                            reads=[final_detect_resource,diffusive_resource,
                                   flag_input],
                            writes=[scaled_resource],
                            exclusive=[:RK_SHARED_FLUX],
                            start! = callback(:rk_fofc_scale))
                        final_detect_id = detect_scale
                        final_detect_resource = scaled_resource
                    end
                    detect_reads = Symbol[
                        u_in,face_in,final_detect_resource,
                        diffusive_resource,junction_resource,
                    ]
                    flag_input === nothing || push!(detect_reads,flag_input)
                    add_structured_task!(builder,detect;
                        depends=[final_detect_id,detect_diffusive],
                        reads=detect_reads,writes=[proposal_resource],
                        exclusive=[:RK_SHARED_FLUX],
                        start! = callback(:rk_fofc_detect))
                    add_structured_task!(builder,detect_commit;
                        depends=[detect],reads=[proposal_resource],
                        writes=[flag_resource],
                        start! = callback(:rk_fofc_commit))
                    push!(commit_ids,detect_commit)
                    push!(local_flag_resources,flag_resource)
                    previous_detect = detect_commit
                end

                flag_sync = Symbol(prefix,"_flag_sync")
                flag_output =
                    Symbol("RK_FOFC_FLAG_HALO_",stage,"_I",iteration)
                add_structured_task!(builder,flag_sync;
                    depends=commit_ids,reads=local_flag_resources,
                    writes=[flag_output],exclusive=[:RK_GHOST_SYNC],
                    collective_sequence=stage,
                    start! = callback(:rk_fofc_flag_sync))
                iteration_finalize = Symbol(prefix,"_finalize")
                add_structured_task!(builder,iteration_finalize;
                    depends=[flag_sync],reads=[flag_output],
                    writes=[Symbol(resource_prefix,"_CLOSED")],
                    collective_sequence=stage,
                    start! = callback(:rk_fofc_iteration_finalize))
                final_pre_barrier = iteration_finalize
            end
        end

        flux_ids = Symbol[]
        div_ids = Symbol[]
        edge_ids = Symbol[]
        previous_block_div = nothing
        for bid in ids
            flux_id = Symbol("rk", stage, "_flux_b", bid)
            face_average_id = Symbol("rk", stage, "_face_average_b", bid)
            diffusive_id = Symbol("rk", stage, "_diffusive_b", bid)
            edge_id = Symbol("rk", stage, "_edge_b", bid)
            source_id = Symbol("rk", stage, "_source_b", bid)
            div_id = Symbol("rk", stage, "_divergence_b", bid)
            point_flux_resource = Symbol("RK_POINT_FLUX_", stage, "_B", bid)
            face_average_resource =
                Symbol("RK_FACE_AVERAGE_FLUX_", stage, "_B", bid)
            diffusive_resource =
                Symbol("RK_DIFFUSIVE_FLUX_", stage, "_B", bid)
            edge_resource = Symbol("RK_EDGE_", stage, "_B", bid)
            source_resource = Symbol("RK_SOURCE_", stage, "_B", bid)
            div_resource = Symbol("RK_DIV_", stage, "_B", bid)
            push!(flux_ids, flux_id)
            push!(edge_ids, edge_id)
            push!(div_ids, div_id)

            flux_dep = Symbol[final_pre_barrier]
            # RK_SHARED_FLUX is one physical buffer set per rank.  An
            # exclusive resource prevents concurrent callbacks, but does not
            # keep block b's flux alive until its divergence callback consumes
            # it.  Serialize complete per-block pipelines explicitly.
            previous_block_div === nothing || push!(flux_dep, previous_block_div)
            flux_reads = [q_reconstruction, Symbol("RK_SHOCK_HALO_", stage)]
            background_split && push!(
                flux_reads, STRUCTURED_TASK_CT_BACKGROUND_READY,
            )
            add_structured_task!(builder, flux_id;
                depends=flux_dep,
                reads=flux_reads,
                writes=[point_flux_resource], exclusive=[:RK_SHARED_FLUX],
                start! = callback(:rk_flux))

            add_structured_task!(builder, face_average_id;
                depends=[flux_id], reads=[point_flux_resource],
                writes=[face_average_resource], exclusive=[:RK_SHARED_FLUX],
                start! = callback(:rk_face_average))
            final_flux_dependency = face_average_id
            final_flux_resource = face_average_resource
            if ct_mode && fofc
                correction_id = Symbol("rk",stage,"_fofc_correct_b",bid)
                corrected_resource =
                    Symbol("RK_FOFC_CORRECTED_FLUX_",stage,"_B",bid)
                add_structured_task!(builder,correction_id;
                    depends=[face_average_id],
                    reads=[face_average_resource,
                           Symbol("RK_FOFC_FLAG_HALO_",stage,"_I",
                                  fofc_iterations_int)],
                    writes=[corrected_resource],exclusive=[:RK_SHARED_FLUX],
                    start! = callback(:rk_fofc_correct))
                final_flux_dependency = correction_id
                final_flux_resource = corrected_resource
            end
            add_structured_task!(builder, diffusive_id;
                depends=[face_average_id],
                reads=[q_reconstruction,face_average_resource],
                writes=[diffusive_resource], exclusive=[:RK_SHARED_FLUX],
                start! = callback(:rk_diffusive_flux))

            if ct_mode
                junction_payload_resource =
                    Symbol("RK_JUNCTION_PAYLOAD_", stage, "_B", bid)
                add_structured_task!(builder, edge_id;
                    depends=[flux_id,final_flux_dependency,diffusive_id],
                    reads=[final_flux_resource,diffusive_resource],
                    writes=[edge_resource,junction_payload_resource],
                    exclusive=[:RK_SHARED_FLUX],
                    start! = callback(:rk_edge_emf))
                if fofc
                    scale_id = Symbol("rk",stage,"_fofc_scale_b",bid)
                    scaled_transport_resource =
                        Symbol("RK_FOFC_SCALED_TRANSPORT_",stage,"_B",bid)
                    add_structured_task!(builder,scale_id;
                        depends=[edge_id,diffusive_id],
                        reads=[final_flux_resource,diffusive_resource,
                               edge_resource,
                               Symbol("RK_FOFC_FLAG_HALO_",stage,"_I",
                                      fofc_iterations_int)],
                        writes=[scaled_transport_resource],
                        exclusive=[:RK_SHARED_FLUX],
                        start! = callback(:rk_fofc_scale))
                    source_dep = [scale_id]
                    source_reads = [scaled_transport_resource,edge_resource]
                else
                    source_dep = [diffusive_id,edge_id]
                    source_reads = [
                        final_flux_resource,diffusive_resource,edge_resource,
                    ]
                end
            else
                source_dep = [diffusive_id]
                source_reads = [final_flux_resource,diffusive_resource]
            end
            add_structured_task!(builder, source_id;
                depends=source_dep, reads=source_reads,
                writes=[source_resource], exclusive=[:RK_SHARED_SOURCE],
                start! = callback(:rk_source))
            add_structured_task!(builder, div_id;
                depends=[source_id], reads=[
                    u_in,source_reads...,source_resource,
                ],
                writes=[u_out, div_resource], exclusive=[:RK_SHARED_FLUX],
                start! = callback(:rk_divergence))
            previous_block_div = div_id
        end

        if ct_mode
            edge_barrier = Symbol("rk", stage, "_edge_barrier")
            add_structured_task!(builder, edge_barrier;
                depends=edge_ids, reads=[
                    Symbol("RK_EDGE_", stage, "_B", bid) for bid in ids
                ], writes=[Symbol("RK_EDGE_CANONICAL_", stage)],
                exclusive=[:RK_CT_MPI], collective_sequence=stage,
                start! = callback(:rk_edge_sync))

            junction_barrier = Symbol("rk", stage, "_junction_barrier")
            add_structured_task!(builder, junction_barrier;
                depends=[edge_barrier], reads=[
                    Symbol("RK_EDGE_CANONICAL_", stage),
                    [Symbol("RK_JUNCTION_PAYLOAD_", stage, "_B", bid)
                     for bid in ids]...,
                ], writes=[Symbol("RK_EDGE_JUNCTION_READY_", stage)],
                exclusive=[:RK_CT_MPI], collective_sequence=stage,
                start! = callback(:rk_junction_solve))

            face_ids = Symbol[]
            face_resources = Symbol[]
            for bid in ids
                face_id = Symbol("rk", stage, "_face_b", bid)
                face_resource = Symbol("RK_FACE_B_", stage, "_B", bid)
                push!(face_ids, face_id)
                push!(face_resources, face_resource)
                add_structured_task!(builder, face_id;
                    depends=[junction_barrier,
                             div_ids[findfirst(==(bid), ids)]],
                    reads=[Symbol("RK_EDGE_JUNCTION_READY_", stage)],
                    writes=[face_resource], exclusive=[:RK_FACE_B_UPDATE],
                    start! = callback(:rk_face_b_update))
            end
            face_barrier = Symbol("rk", stage, "_face_barrier")
            add_structured_task!(builder, face_barrier;
                depends=face_ids,
                reads=face_resources,
                writes=[Symbol("RK_FACE_B_READY_", stage)],
                collective_sequence=stage,
                start! = callback(:rk_face_barrier))
            sync_dep = face_barrier
            sync_reads = [u_out, Symbol("RK_FACE_B_READY_", stage)]
        else
            sync_dep = Symbol("rk", stage, "_divergence_barrier")
            add_structured_task!(builder, sync_dep;
                depends=div_ids,
                reads=[Symbol("RK_DIV_", stage, "_B", bid) for bid in ids],
                writes=[Symbol("RK_DIV_READY_", stage)],
                start! = callback(:rk_divergence_barrier))
            sync_reads = [u_out]
        end

        sync_id = Symbol("rk", stage, "_sync")
        add_structured_task!(builder, sync_id;
            depends=[sync_dep], reads=sync_reads, writes=[q_out],
            exclusive=[:RK_GHOST_SYNC], collective_sequence=stage,
            start! = callback(:rk_stage_sync))
        positivity_id = Symbol("rk", stage, "_positivity")
        add_structured_task!(builder, positivity_id;
            depends=[sync_id], reads=[q_out],
            writes=[Symbol("RK_POSITIVITY_", stage)],
            start! = callback(:rk_positivity))
        previous_barrier = positivity_id
    end
    if !defer_trailing_split
        add_structured_task!(builder, :rk_trailing_split_source;
            depends=[previous_barrier],
            reads=[:RK_POSITIVITY_3],
            writes=[:RK_COMMITTED_STATE],
            collective_sequence=4,
            start! = callback(:rk_trailing_split_source))
    end
    return build_structured_task_graph(builder)
end
