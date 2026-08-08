struct LegacyStructuredCaseState{C}
    config::C
    script_path::String
end

function prepare_backend_case(::StructuredBackend,config,project_root)
    script=config.backend.entry_script
    script===nothing && error("legacy structured cases require backend.entry_script")
    return LegacyStructuredCaseState(config,abspath(joinpath(project_root,script)))
end

function validate_backend_case!(::StructuredBackend,state::LegacyStructuredCaseState)
    isfile(state.script_path) || error("structured entry does not exist: $(state.script_path)")
    Meta.parseall(read(state.script_path,String))
    return state
end

function advance_backend_case!(::StructuredBackend,state::LegacyStructuredCaseState)
    empty!(ARGS)
    append!(ARGS,state.config.arguments)
    return include(state.script_path)
end
