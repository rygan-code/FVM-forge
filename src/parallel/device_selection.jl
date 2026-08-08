function resolve_requested_device(configured::Symbol; environment=ENV)
    override=lowercase(get(environment,"OPENCFD_DEVICE",""))
    requested=isempty(override) ? configured : Symbol(override)
    requested in (:auto,:cpu,:cuda,:rocm) || throw(ArgumentError(
        "device must be auto, cpu, cuda, or rocm; got $requested"))
    return requested
end

function _load_cuda!()
    Core.eval(Main,:(using CUDA))
    Core.eval(Main,:(CUDA.functional())) ||
        error("CUDA was requested but is not functional")
    return :cuda
end

function _load_rocm!()
    Core.eval(Main,:(using AMDGPU))
    return :rocm
end

function activate_requested_device!(configured::Symbol; environment=ENV)
    requested=resolve_requested_device(configured;environment=environment)
    requested==:cpu && return :cpu
    requested==:cuda && return _load_cuda!()
    requested==:rocm && return _load_rocm!()
    try
        return _load_cuda!()
    catch
    end
    try
        return _load_rocm!()
    catch
    end
    return :cpu
end
