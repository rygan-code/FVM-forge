const UNSTRUCTURED_CASE_BC_MAP=Dict(
    :zero_gradient=>Int(BC_ZERO_GRADIENT),
    :periodic=>Int(BC_PERIODIC),
    :slip_wall=>Int(BC_SLIP_WALL),
    :no_slip_wall=>Int(BC_ISOTHERMAL_WALL),
    :symmetry=>Int(BC_SYMMETRY),
    :supersonic_outflow=>Int(BC_SUPERSONIC_OUTFLOW),
)

struct UnstructuredCaseState{C,B,M,Q}
    config::C
    block::B
    riemann_id::Int32
    comm::M
    owns_mpi::Bool
    partition_quality::Q
end

function _unstructured_boundary_ids(boundaries)
    return map(boundaries) do name
        get(UNSTRUCTURED_CASE_BC_MAP,name) do
            error("unsupported boundary name $name")
        end
    end
end

function _load_unstructured_case_mesh(config,project_root)
    mesh=config.mesh
    if mesh.kind==:openfoam
        mesh.path===nothing && error("OpenFOAM mesh requires mesh.path")
        return read_openfoam_mesh(abspath(joinpath(project_root,mesh.path)))
    end
    mesh.kind==:cartesian || error("unsupported unstructured mesh kind $(mesh.kind)")
    boundary_ids=_unstructured_boundary_ids(mesh.boundaries)
    return gen_cartesian_unstruct(
        mesh.cells...,
        FT(mesh.lower[1]),FT(mesh.upper[1]),
        FT(mesh.lower[2]),FT(mesh.upper[2]),
        FT(mesh.lower[3]),FT(mesh.upper[3]),boundary_ids...)
end

function _unstructured_riemann_id(config)
    config.numerics.riemann==:rusanov && return Int32(0)
    config.numerics.riemann==:hllc && return Int32(1)
    error("unsupported Riemann solver $(config.numerics.riemann)")
end

function _unstructured_parallel_context(config)
    config.parallel.enabled || return nothing,false
    owns_mpi=!MPI.Initialized()
    owns_mpi && MPI.Init()
    return MPI.COMM_WORLD,owns_mpi
end

function prepare_backend_case(::UnstructuredBackend,config,project_root)
    comm,owns_mpi=_unstructured_parallel_context(config)
    try
        block=_load_unstructured_case_mesh(config,project_root)
        quality=nothing
        if comm !== nothing && MPI.Comm_size(comm)>1
            block,_,quality=partition_unstruct_mesh(
                block,comm;
                order=config.numerics.order,
                viscous_enabled=config.physics.viscous,
                resistive_enabled=config.physics.resistive,
                mhd_enabled=config.physics.equations==:MHD,
                ct_enabled=config.physics.ct,
                compute_imbalance=config.parallel.compute_imbalance,
                memory_imbalance=config.parallel.memory_imbalance,
                max_ghost_ratio=config.parallel.max_ghost_ratio,
                periodic_edge_multiplier=config.parallel.periodic_edge_multiplier,
                seed=config.parallel.seed)
        end
        return UnstructuredCaseState(
            config,block,_unstructured_riemann_id(config),comm,owns_mpi,quality)
    catch
        if owns_mpi && MPI.Initialized() && !MPI.Finalized()
            MPI.Finalize()
        end
        rethrow()
    end
end

function validate_backend_case!(::UnstructuredBackend,state::UnstructuredCaseState)
    state.block.ncell>0 || error("unstructured case mesh has no cells")
    validate_unstruct_boundary_support(state.block)
    return state
end

function initialize_backend_case!(::UnstructuredBackend,state::UnstructuredCaseState)
    rank=state.comm===nothing ? 0 : MPI.Comm_rank(state.comm)
    if state.config.write_output && rank==0
        path=joinpath(state.config.output_directory,"effective_case.toml")
        write_effective_case_config(path,state.config)
        println("CASE_EFFECTIVE_CONFIG path=$(abspath(path))")
    end
    return state
end

function advance_backend_case!(backend::UnstructuredBackend,state::UnstructuredCaseState)
    config=state.config
    rank=state.comm===nothing ? 0 : MPI.Comm_rank(state.comm)
    output_directory=state.comm===nothing ? config.output_directory :
        joinpath(config.output_directory,"rank-$(lpad(rank,4,'0'))")
    return run_solver_backend!(
        backend,state.block,config.initializer,FT(config.time.final_time),
        config.time.max_steps,FT(config.numerics.cfl),
        config.time.output_interval,output_directory,
        config.numerics.order,state.riemann_id;
        comm_cart=state.comm,gpu_aware=config.parallel.gpu_aware,
        write_output=config.write_output)
end

function finalize_backend_case!(::UnstructuredBackend,state::UnstructuredCaseState)
    gpu_sync()
    if state.owns_mpi && MPI.Initialized() && !MPI.Finalized()
        MPI.Finalize()
    end
    return nothing
end
