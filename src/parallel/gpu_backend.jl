# gpu_backend.jl — Unified Compute Backend Abstraction
# Supports CUDA (NVIDIA), AMDGPU (AMD DCU/ROCm), and CPU multi-threaded
# 
# Usage: include this file BEFORE any kernel code
# It provides: @gpu_launch, GPUArray, gpu_sync(), gpu_allowscalar(), etc.
#
# Backend selection: determined by which package is loaded before including this file
#   - If `CUDA` is defined  → NVIDIA GPU backend
#   - If `AMDGPU` is defined → AMD/ROCm GPU backend
#   - If neither is defined  → CPU multi-threaded backend (auto)

# ── Backend Detection ──
const USE_CUDA  = @isdefined(CUDA)
const USE_ROCM  = @isdefined(AMDGPU)
const USE_CPU   = !USE_CUDA && !USE_ROCM
const WARP_SIZE = USE_ROCM ? 64 : (USE_CUDA ? 32 : 1)

if USE_CPU
    @info "No GPU backend detected. Using CPU multi-threaded backend ($(Threads.nthreads()) threads)."
end

# ═══════════════════════════════════════════════════════
# CUDA Backend
# ═══════════════════════════════════════════════════════
if USE_CUDA

const GPUArray = CuArray
const GPUVector{T} = CuVector{T}

"""
    @gpu_launch threads=t blocks=b kernel(args...)

Launch a GPU kernel. Dispatches to @cuda or @roc depending on backend.
"""
macro gpu_launch(ex...)
    # Parse keyword arguments and the kernel call
    kwargs = []
    call = nothing
    for e in ex
        if e isa Expr && e.head == :(=)
            push!(kwargs, e)
        else
            call = e
        end
    end
    # Build @cuda call
    cuda_expr = Expr(:macrocall, Symbol("@cuda"), __source__, kwargs..., call)
    return esc(cuda_expr)
end

gpu_sync()              = CUDA.synchronize()
gpu_allowscalar(b::Bool) = CUDA.allowscalar(b)
gpu_stream_create()      = CUDA.CuStream()
gpu_stream_sync(s)       = CUDA.synchronize(s)

"""
    @gpu_launch_stream stream threads=(...) blocks=(...) kernel!(args...)

Launch a GPU kernel on a specific CUDA stream for async execution.
"""
macro gpu_launch_stream(stream_expr, ex...)
    kwargs = [Expr(:(=), :stream, stream_expr)]
    call = nothing
    for e in ex
        if e isa Expr && e.head == :(=)
            push!(kwargs, e)
        else
            call = e
        end
    end
    cuda_expr = Expr(:macrocall, Symbol("@cuda"), __source__, kwargs..., call)
    return esc(cuda_expr)
end

function gpu_zeros(T::Type, dims...)
    return CUDA.zeros(T, dims...)
end

function gpu_device_id()
    return CUDA.device()
end

# ═══════════════════════════════════════════════════════
# AMDGPU (ROCm) Backend
# ═══════════════════════════════════════════════════════
elseif USE_ROCM

@eval const GPUArray = ROCArray
@eval const GPUVector{T} = ROCArray{T, 1}

# CUDA-compatible thread indexing aliases for AMDGPU
# This allows all kernels to use blockIdx()/blockDim()/threadIdx() on both backends
@eval @inline blockIdx() = workgroupIdx()
@eval @inline blockDim() = workgroupDim()
@eval @inline threadIdx() = workitemIdx()

@eval begin

"""
    @gpu_launch threads=t blocks=b kernel(args...)

Launch a GPU kernel on AMD GPU via @roc.
"""
macro gpu_launch(ex...)
    kwargs = []
    call = nothing
    for e in ex
        if e isa Expr && e.head == :(=)
            # Translate CUDA keywords to AMDGPU keywords
            if e.args[1] == :threads
                push!(kwargs, Expr(:(=), :groupsize, e.args[2]))
            elseif e.args[1] == :blocks
                push!(kwargs, Expr(:(=), :gridsize, e.args[2]))
            else
                push!(kwargs, e)
            end
        else
            call = e
        end
    end
    roc_expr = Expr(:macrocall, Symbol("@roc"), __source__, kwargs..., call)
    return esc(roc_expr)
end

"""
    @gpu_launch_stream stream threads=(...) blocks=(...) kernel!(args...)

Launch a GPU kernel on a specific HIPStream for async execution.
"""
macro gpu_launch_stream(stream_expr, ex...)
    kwargs = [Expr(:(=), :stream, stream_expr)]
    call = nothing
    for e in ex
        if e isa Expr && e.head == :(=)
            if e.args[1] == :threads
                push!(kwargs, Expr(:(=), :groupsize, e.args[2]))
            elseif e.args[1] == :blocks
                push!(kwargs, Expr(:(=), :gridsize, e.args[2]))
            else
                push!(kwargs, e)
            end
        else
            call = e
        end
    end
    roc_expr = Expr(:macrocall, Symbol("@roc"), __source__, kwargs..., call)
    return esc(roc_expr)
end

end  # @eval

gpu_sync()               = AMDGPU.synchronize()
gpu_allowscalar(b::Bool) = AMDGPU.allowscalar(b)
gpu_stream_create()      = AMDGPU.HIPStream()
gpu_stream_sync(s)       = AMDGPU.synchronize(s)


function gpu_zeros(T::Type, dims...)
    return AMDGPU.zeros(T, dims...)
end

function gpu_device_id()
    return AMDGPU.device()
end

# ── Fix MPI.jl v0.20.8 + AMDGPU.jl v0.9.3 GPU-aware compatibility ──
# MPI.jl's rocm.jl tries `X.buf.ptr` but AMDGPU v0.9.3 uses DataRef{Managed{B}}
# which exposes the pointer via `Base.unsafe_convert(Ptr{T}, x)` instead.
# NOTE: Cluster MPI/UCX does NOT support GPU-aware transport (process_vm_readv
# cannot access GPU memory). This patch is ready for when MPI is rebuilt with
# ROCm-aware UCX. Enable gpu_aware_mpi=true in run_pipe.jl at that time.
#=
if @isdefined(MPI) && isdefined(MPI, :API) && isdefined(MPI.API, :MPIPtr)
    function Base.unsafe_convert(::Type{MPI.API.MPIPtr}, X::ROCArray{T}) where T
        reinterpret(MPI.API.MPIPtr, Base.unsafe_convert(Ptr{T}, X))
    end
    function Base.unsafe_convert(::Type{MPI.API.MPIPtr}, V::SubArray{T,N,P,I,true}) where {T,N,P<:ROCArray,I}
        X = parent(V)
        pX = Base.unsafe_convert(Ptr{T}, X)
        pV = pX + ((V.offset1 + V.stride1) - first(LinearIndices(X)))*sizeof(T)
        return reinterpret(MPI.API.MPIPtr, pV)
    end
    function Base.cconvert(::Type{MPI.API.MPIPtr}, A::ROCArray{T}) where T
        A
    end
end
=#

# ═══════════════════════════════════════════════════════
# CPU Multi-Threaded Backend
# ═══════════════════════════════════════════════════════
else  # USE_CPU

const GPUArray = Array
const GPUVector{T} = Vector{T}

# ── Thread-safe kernel context via task_local_storage ──
# Each Julia thread maintains its own blockIdx/blockDim/threadIdx,
# enabling safe parallel execution of GPU-style kernels on CPU.
@inline blockIdx()  = task_local_storage(:_cpu_blockIdx)::NamedTuple{(:x,:y,:z), Tuple{Int32,Int32,Int32}}
@inline blockDim()  = task_local_storage(:_cpu_blockDim)::NamedTuple{(:x,:y,:z), Tuple{Int32,Int32,Int32}}
@inline threadIdx() = task_local_storage(:_cpu_threadIdx)::NamedTuple{(:x,:y,:z), Tuple{Int32,Int32,Int32}}

"""
    @gpu_launch threads=t blocks=b kernel(args...)

CPU backend: distributes GPU blocks across CPU threads via Threads.@threads.
Threads within each block run serially (mirrors GPU warp-sequential semantics).
Thread context (blockIdx, blockDim, threadIdx) is stored per-task for safety.

Supports both 3D tuple and scalar thread/block specifications:
  @gpu_launch threads=(8,4,4) blocks=(17,50,10) my_kernel(args...)
  @gpu_launch threads=256 blocks=100 my_1d_kernel(args...)
"""
macro gpu_launch(ex...)
    threads_expr = nothing
    blocks_expr = nothing
    call = nothing
    for e in ex
        if e isa Expr && e.head == :(=)
            if e.args[1] in (:threads, :groupsize)
                threads_expr = e.args[2]
            elseif e.args[1] in (:blocks, :gridsize)
                blocks_expr = e.args[2]
            end
        else
            call = e
        end
    end
    kernel_func = call.args[1]
    kernel_args = call.args[2:end]

    return esc(quote
        let _th = $(threads_expr), _bl = $(blocks_expr)
            # Normalize scalar → 3-tuple (handles 1D kernels like LU-SGS plane sweeps)
            _tx = Int32(_th isa Integer ? _th : _th[1])
            _ty = Int32(_th isa Integer ? 1   : (length(_th) >= 2 ? _th[2] : 1))
            _tz = Int32(_th isa Integer ? 1   : (length(_th) >= 3 ? _th[3] : 1))
            _bx = Int32(_bl isa Integer ? _bl : _bl[1])
            _by = Int32(_bl isa Integer ? 1   : (length(_bl) >= 2 ? _bl[2] : 1))
            _bz = Int32(_bl isa Integer ? 1   : (length(_bl) >= 3 ? _bl[3] : 1))

            _bdim = (x=_tx, y=_ty, z=_tz)
            _total_blocks = Int(_bx) * Int(_by) * Int(_bz)

            # Distribute blocks across CPU threads; threads within block run serially
            Threads.@threads :static for _flat in 1:_total_blocks
                _bi = Int32(Base.div(_flat - 1, Int(_by) * Int(_bz)) + 1)
                _bj = Int32(Base.div((_flat - 1) % (Int(_by) * Int(_bz)), Int(_bz)) + 1)
                _bk = Int32((_flat - 1) % Int(_bz) + 1)
                task_local_storage(:_cpu_blockDim, _bdim)
                task_local_storage(:_cpu_blockIdx, (x=_bi, y=_bj, z=_bk))
                for _ti = Int32(1):_tx, _tj = Int32(1):_ty, _tk = Int32(1):_tz
                    task_local_storage(:_cpu_threadIdx, (x=_ti, y=_tj, z=_tk))
                    $(kernel_func)($(kernel_args...))
                end
            end
        end
    end)
end

macro gpu_launch_stream(stream_expr, ex...)
    return esc(Expr(:macrocall, Symbol("@gpu_launch"), __source__, ex...))
end

gpu_sync()               = nothing
gpu_allowscalar(b::Bool) = nothing
gpu_stream_create()      = nothing
gpu_stream_sync(s)       = nothing

function gpu_zeros(T::Type, dims...)
    return zeros(T, dims...)
end

function gpu_device_id()
    return "CPU"
end

end  # if USE_CUDA / USE_ROCM / USE_CPU

# Device-side atomics used by diagnostic kernels.  Keep these behind the
# backend layer so CT code does not depend directly on CUDA intrinsics.
@inline function gpu_atomic_cas!(array, index::Integer, compare::T, value::T) where {T}
    @static if USE_CUDA
        return CUDA.atomic_cas!(pointer(array, index), compare, value)
    elseif USE_ROCM
        return AMDGPU.Device.llvm_atomic_cas(
            pointer(array, index), compare, value,
        )
    else
        return Core.Intrinsics.atomic_pointerreplace(
            pointer(array, index), compare, value,
            :acquire_release, :acquire,
        ).old
    end
end

@inline function gpu_atomic_add!(array, index::Integer, increment::T) where {T}
    pointer_value = pointer(array, index)
    @static if USE_CUDA
        return CUDA.atomic_add!(pointer_value, increment)
    else
        old = @inbounds array[index]
        while true
            observed = gpu_atomic_cas!(array, index, old, old + increment)
            observed == old && return old
            old = observed
        end
    end
end

# ═══════════════════════════════════════════════════════
# Backend-agnostic utilities
# ═══════════════════════════════════════════════════════

"""Convert GPU array to host Array (works for CuArray, ROCArray, and plain Array)"""
to_host(x::AbstractArray) = Array(x)
to_host(x) = x

"""Print detected backend info"""
function print_gpu_backend_info()
    if USE_CUDA
        dev = CUDA.device()
        println("  GPU Backend: CUDA (NVIDIA)")
        println("  Device: $(CUDA.name(dev))")
        println("  VRAM: $(round(CUDA.totalmem(dev) / 1024^3, digits=1)) GB")
        println("  Warp size: $WARP_SIZE")
    elseif USE_ROCM
        println("  GPU Backend: ROCm (AMDGPU)")
        try
            println("  Device: $(AMDGPU.device())")
        catch
            println("  Device: (query unavailable)")
        end
        println("  Wavefront size: $WARP_SIZE")
    else
        println("  Backend: CPU Multi-Threaded")
        println("  Threads: $(Threads.nthreads())")
        println("  CPU: $(Sys.cpu_info()[1].model)")
    end
end
