using TOML

struct BackendCaseConfig
    mesh_type::Symbol
    device::Symbol
    mode::Symbol
    entry_script::Union{Nothing,String}
end

struct MeshCaseConfig
    kind::Symbol
    path::Union{Nothing,String}
    cells::NTuple{3,Int}
    lower::NTuple{3,Float64}
    upper::NTuple{3,Float64}
    boundaries::NTuple{6,Symbol}
end

struct PhysicsCaseConfig
    equations::Symbol
    gamma::Float64
    gas_constant::Float64
    prandtl::Float64
    wall_temperature::Float64
    viscous::Bool
    resistive::Bool
    resistivity::Float64
    ct::Bool
end

struct NumericsCaseConfig
    order::Int
    riemann::Symbol
    cfl::Float64
    positivity_mode::Symbol
    density_floor::Float64
    pressure_floor::Float64
end

struct TimeCaseConfig
    final_time::Float64
    max_steps::Int
    output_interval::Int
end

struct ParallelCaseConfig
    enabled::Bool
    partitioner::Symbol
    compute_imbalance::Float64
    memory_imbalance::Float64
    max_ghost_ratio::Float64
    periodic_edge_multiplier::Int
    seed::Int
    gpu_aware::Bool
end

default_parallel_case_config() = ParallelCaseConfig(
    false, :metis, 1.05, 1.05, 0.5, 16, 17, false,
)

struct CaseConfig
    name::String
    initializer::String
    backend::BackendCaseConfig
    mesh::MeshCaseConfig
    physics::PhysicsCaseConfig
    numerics::NumericsCaseConfig
    time::TimeCaseConfig
    parallel::ParallelCaseConfig
    output_directory::String
    write_output::Bool
    environment::Dict{String,String}
    arguments::Vector{String}
end


# Preserve the pre-parallel positional constructor used by existing callers.
function CaseConfig(
    name,initializer,backend,mesh,physics,numerics,time,
    output_directory,write_output,environment,arguments,
)
    return CaseConfig(
        name,initializer,backend,mesh,physics,numerics,time,
        default_parallel_case_config(),output_directory,write_output,
        environment,arguments,
    )
end

function _tuple3(table, key, converter, default)
    values=get(table,key,default)
    values isa Vector && length(values)==3 || throw(ArgumentError(
        "$key must contain exactly three values"))
    return Tuple(converter.(values))
end

function _boundary_tuple(table)
    names=("xlo","xhi","ylo","yhi","zlo","zhi")
    values=ntuple(index -> Symbol(lowercase(String(get(
        table,names[index],"zero_gradient")))),6)
    return values
end

function load_case_config(path::AbstractString)
    document=TOML.parsefile(path)
    backend=get(document,"backend",Dict())
    mesh=get(document,"mesh",Dict())
    physics=get(document,"physics",Dict())
    numerics=get(document,"numerics",Dict())
    time=get(document,"time",Dict())
    parallel=get(document,"parallel",Dict())
    output=get(document,"output",Dict())
    boundaries=get(mesh,"boundaries",Dict())
    backend_config=BackendCaseConfig(
        Symbol(lowercase(String(get(backend,"mesh_type","unstructured")))),
        Symbol(lowercase(String(get(backend,"device","auto")))),
        Symbol(lowercase(String(get(backend,"mode",
            haskey(backend,"entry_script") ? "legacy_entry" : "native")))),
        haskey(backend,"entry_script") ? String(backend["entry_script"]) : nothing)
    mesh_config=MeshCaseConfig(
        Symbol(lowercase(String(get(mesh,"kind","cartesian")))),
        haskey(mesh,"path") ? String(mesh["path"]) : nothing,
        _tuple3(mesh,"cells",Int,[1,1,1]),
        _tuple3(mesh,"lower",Float64,[0.0,0.0,0.0]),
        _tuple3(mesh,"upper",Float64,[1.0,1.0,1.0]),
        _boundary_tuple(boundaries))
    physics_config=PhysicsCaseConfig(
        Symbol(get(physics,"equations","compressible")),
        Float64(get(physics,"gamma",1.4)),Float64(get(physics,"gas_constant",1.0)),
        Float64(get(physics,"prandtl",0.72)),Float64(get(physics,"wall_temperature",0.0)),
        Bool(get(physics,"viscous",false)),Bool(get(physics,"resistive",false)),
        Float64(get(physics,"resistivity",0.0)),Bool(get(physics,"ct",false)))
    numerics_config=NumericsCaseConfig(
        Int(get(numerics,"order",2)),Symbol(lowercase(String(get(numerics,"riemann","rusanov")))),
        Float64(get(numerics,"cfl",0.3)),
        Symbol(lowercase(String(get(numerics,"positivity_mode","repair")))),
        Float64(get(numerics,"density_floor",1.0e-5)),
        Float64(get(numerics,"pressure_floor",1.0e-5)))
    time_config=TimeCaseConfig(
        Float64(get(time,"final_time",0.1)),Int(get(time,"max_steps",10000)),
        Int(get(time,"output_interval",100)))
    parallel_config=ParallelCaseConfig(
        Bool(get(parallel,"enabled",false)),
        Symbol(lowercase(String(get(parallel,"partitioner","metis")))),
        Float64(get(parallel,"compute_imbalance",1.05)),
        Float64(get(parallel,"memory_imbalance",1.05)),
        Float64(get(parallel,"max_ghost_ratio",0.5)),
        Int(get(parallel,"periodic_edge_multiplier",16)),
        Int(get(parallel,"seed",17)),
        Bool(get(parallel,"gpu_aware",false)))
    environment=Dict(String(key)=>String(value) for (key,value) in
        get(document,"environment",Dict()))
    arguments=String.(get(document,"arguments",String[]))
    return CaseConfig(
        String(get(document,"name",splitext(basename(path))[1])),
        String(get(document,"initializer","Sod")),backend_config,mesh_config,
        physics_config,numerics_config,time_config,parallel_config,
        String(get(output,"directory","PLT")),Bool(get(output,"write",true)),
        environment,arguments)
end

function validate_case_config(config::CaseConfig)
    config.backend.mesh_type in (:structured,:unstructured) ||
        throw(ArgumentError("mesh_type must be structured or unstructured"))
    config.backend.device in (:auto,:cpu,:cuda,:rocm) || throw(ArgumentError(
        "device must be auto, cpu, cuda, or rocm"))
    config.backend.mode in (:native,:legacy_entry) || throw(ArgumentError(
        "backend.mode must be native or legacy_entry"))
    if config.backend.mode == :legacy_entry
        config.parallel.enabled && throw(ArgumentError(
            "parallel configuration is only supported by native unstructured cases"))
        config.backend.mesh_type == :structured || throw(ArgumentError(
            "legacy_entry mode is reserved for structured entry scripts"))
        config.backend.entry_script === nothing && throw(ArgumentError(
            "legacy_entry mode requires backend.entry_script"))
        config.backend.device == :auto || throw(ArgumentError(
            "legacy_entry mode requires device=auto; the entry script selects its device"))
        return config
    end
    config.backend.entry_script === nothing || throw(ArgumentError(
        "native mode cannot define backend.entry_script"))
    config.backend.mesh_type == :unstructured || throw(ArgumentError(
        "native structured TOML execution is not implemented; use legacy_entry"))
    isempty(strip(config.name)) && throw(ArgumentError("case name must not be empty"))
    isempty(strip(config.initializer)) && throw(ArgumentError(
        "initializer must not be empty"))
    config.physics.equations in (:compressible,:MHD) ||
        throw(ArgumentError("equations must be compressible or MHD"))
    config.mesh.kind in (:cartesian,:openfoam) || throw(ArgumentError(
        "unstructured mesh.kind must be cartesian or openfoam"))
    if config.mesh.kind == :cartesian
        all(>(0),config.mesh.cells) || throw(ArgumentError(
            "cartesian mesh cell counts must be positive"))
        all(isfinite,config.mesh.lower) && all(isfinite,config.mesh.upper) ||
            throw(ArgumentError("cartesian mesh bounds must be finite"))
        all(config.mesh.upper[index] > config.mesh.lower[index] for index in 1:3) ||
            throw(ArgumentError(
                "cartesian mesh upper bounds must exceed lower bounds"))
        valid_boundaries=(
            :zero_gradient,:periodic,:slip_wall,:no_slip_wall,
            :symmetry,:supersonic_outflow,
        )
        all(boundary in valid_boundaries for boundary in config.mesh.boundaries) ||
            throw(ArgumentError(
                "unsupported cartesian boundary name; supported names are " *
                join(valid_boundaries, ", ")))
        for axis in 1:3
            low_periodic=config.mesh.boundaries[2axis-1] == :periodic
            high_periodic=config.mesh.boundaries[2axis] == :periodic
            low_periodic == high_periodic || throw(ArgumentError(
                "periodic boundaries must be paired on axis $axis"))
        end
    else
        config.mesh.path !== nothing && !isempty(strip(config.mesh.path)) ||
            throw(ArgumentError("OpenFOAM mesh requires a non-empty mesh.path"))
    end
    isfinite(config.physics.gamma) && config.physics.gamma > 1 ||
        throw(ArgumentError("gamma must be finite and greater than 1"))
    isfinite(config.physics.gas_constant) && config.physics.gas_constant > 0 ||
        throw(ArgumentError("gas_constant must be finite and positive"))
    isfinite(config.physics.prandtl) && config.physics.prandtl > 0 ||
        throw(ArgumentError("prandtl must be finite and positive"))
    isfinite(config.physics.wall_temperature) &&
        config.physics.wall_temperature >= 0 || throw(ArgumentError(
            "wall_temperature must be finite and non-negative"))
    :no_slip_wall in config.mesh.boundaries &&
        config.physics.wall_temperature <= 0 && throw(ArgumentError(
            "no_slip_wall boundaries require wall_temperature > 0"))
    config.physics.ct && config.physics.equations != :MHD && throw(ArgumentError(
        "constrained transport requires equations=MHD"))
    config.physics.resistive && config.physics.equations != :MHD &&
        throw(ArgumentError("resistive physics requires equations=MHD"))
    config.numerics.order in (1,2) || throw(ArgumentError("order must be 1 or 2"))
    config.parallel.partitioner == :metis || throw(ArgumentError(
        "parallel.partitioner must be metis"))
    isfinite(config.parallel.compute_imbalance) &&
        config.parallel.compute_imbalance >= 1 || throw(ArgumentError(
            "parallel.compute_imbalance must be finite and at least 1"))
    isfinite(config.parallel.memory_imbalance) &&
        config.parallel.memory_imbalance >= 1 || throw(ArgumentError(
            "parallel.memory_imbalance must be finite and at least 1"))
    isfinite(config.parallel.max_ghost_ratio) &&
        config.parallel.max_ghost_ratio > 0 || throw(ArgumentError(
            "parallel.max_ghost_ratio must be finite and positive"))
    config.parallel.periodic_edge_multiplier >= 1 || throw(ArgumentError(
        "parallel.periodic_edge_multiplier must be at least 1"))
    config.parallel.seed >= 0 || throw(ArgumentError(
        "parallel.seed must be non-negative"))
    config.initializer in ("Sod","TGV","BrioWu") || throw(ArgumentError(
        "initializer must be Sod, TGV, or BrioWu"))
    config.initializer == "BrioWu" && config.physics.equations != :MHD &&
        throw(ArgumentError("BrioWu initialization requires equations=MHD"))
    config.numerics.positivity_mode in (:strict,:repair) ||
        throw(ArgumentError("positivity_mode must be strict or repair"))
    valid_riemann = config.physics.equations == :MHD ? (:rusanov,) : (:rusanov,:hllc)
    config.numerics.riemann in valid_riemann || throw(ArgumentError(
        "$(config.physics.equations) on the unstructured backend supports " *
        "Riemann solvers $(join(valid_riemann, ", "))"))
    isfinite(config.numerics.density_floor) && config.numerics.density_floor > 0 ||
        throw(ArgumentError("density_floor must be finite and positive"))
    isfinite(config.numerics.pressure_floor) && config.numerics.pressure_floor > 0 ||
        throw(ArgumentError("pressure_floor must be finite and positive"))
    isfinite(config.time.final_time) && config.time.final_time >= 0 ||
        throw(ArgumentError("final_time must be finite and non-negative"))
    config.time.max_steps > 0 || throw(ArgumentError("max_steps must be positive"))
    config.time.output_interval > 0 || throw(ArgumentError(
        "output_interval must be positive"))
    isfinite(config.numerics.cfl) && config.numerics.cfl > 0 ||
        throw(ArgumentError("cfl must be finite and positive"))
    isfinite(config.physics.resistivity) && config.physics.resistivity >= 0 ||
        throw(ArgumentError("resistivity must be finite and non-negative"))
    config.physics.resistive && config.physics.resistivity <= 0 &&
        throw(ArgumentError("resistive cases require resistivity > 0"))
    config.write_output && isempty(strip(config.output_directory)) &&
        throw(ArgumentError("output.directory must not be empty when output is enabled"))
    return config
end

function apply_case_environment!(config::CaseConfig)
    for (key,value) in config.environment
        ENV[key]=value
    end
    return config
end

function case_config_document(config::CaseConfig)
    backend=Dict(
        "mesh_type"=>String(config.backend.mesh_type),
        "device"=>String(config.backend.device),
        "mode"=>String(config.backend.mode),
    )
    config.backend.entry_script === nothing ||
        (backend["entry_script"]=config.backend.entry_script)
    document=Dict{String,Any}(
        "name"=>config.name,
        "initializer"=>config.initializer,
        "backend"=>backend,
        "environment"=>config.environment,
        "arguments"=>config.arguments,
    )
    config.backend.mode == :legacy_entry && return document
    mesh_document=Dict{String,Any}(
        "kind"=>String(config.mesh.kind),
        "cells"=>collect(config.mesh.cells),"lower"=>collect(config.mesh.lower),
        "upper"=>collect(config.mesh.upper),
        "boundaries"=>Dict(zip(
            ("xlo","xhi","ylo","yhi","zlo","zhi"),
            String.(config.mesh.boundaries))),
    )
    config.mesh.path === nothing || (mesh_document["path"]=config.mesh.path)
    document["mesh"]=mesh_document
    document["physics"]=Dict(
        "equations"=>String(config.physics.equations),"gamma"=>config.physics.gamma,
        "gas_constant"=>config.physics.gas_constant,"prandtl"=>config.physics.prandtl,
        "wall_temperature"=>config.physics.wall_temperature,
        "viscous"=>config.physics.viscous,"resistive"=>config.physics.resistive,
        "resistivity"=>config.physics.resistivity,"ct"=>config.physics.ct,
    )
    document["numerics"]=Dict(
        "order"=>config.numerics.order,"riemann"=>String(config.numerics.riemann),
        "cfl"=>config.numerics.cfl,
        "positivity_mode"=>String(config.numerics.positivity_mode),
        "density_floor"=>config.numerics.density_floor,
        "pressure_floor"=>config.numerics.pressure_floor,
    )
    document["time"]=Dict(
        "final_time"=>config.time.final_time,"max_steps"=>config.time.max_steps,
        "output_interval"=>config.time.output_interval,
    )
    document["parallel"]=Dict(
        "enabled"=>config.parallel.enabled,
        "partitioner"=>String(config.parallel.partitioner),
        "compute_imbalance"=>config.parallel.compute_imbalance,
        "memory_imbalance"=>config.parallel.memory_imbalance,
        "max_ghost_ratio"=>config.parallel.max_ghost_ratio,
        "periodic_edge_multiplier"=>config.parallel.periodic_edge_multiplier,
        "seed"=>config.parallel.seed,
        "gpu_aware"=>config.parallel.gpu_aware,
    )
    document["output"]=Dict(
        "directory"=>config.output_directory,"write"=>config.write_output,
    )
    return document
end

function write_effective_case_config(path::AbstractString,config::CaseConfig)
    mkpath(dirname(path))
    open(path,"w") do output
        TOML.print(output,case_config_document(config);sorted=true)
    end
    return path
end
