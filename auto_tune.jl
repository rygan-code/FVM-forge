# auto_tune.jl — Adaptive Kernel Launch Configuration
# Supports CUDA (NVIDIA), AMDGPU (AMD DCU/ROCm), and CPU backends
# Features:
#   - Per-kernel optimal thread block decomposition via occupancy API (GPU)
#   - Adaptive maxregs: tries multiple register limits, picks best occupancy vs spill trade-off
#   - CPU: returns fixed thread/block config, skips compilation/benchmarking

using Printf
if !USE_CPU
    using Libdl
end

"""
    find_best_3d_block(max_threads, nxp, nyp, nzp)

Find the best 3D thread block decomposition for a given max thread count.
"""
function find_best_3d_block(max_threads::Int, nxp::Int, nyp::Int, nzp::Int)
    best_threads = (8, 4, 4)
    best_score = -Inf
    
    x_candidates = [4, 8, 16, 32, 64]
    y_candidates = [1, 2, 4, 8, 16, 32]
    z_candidates = [1, 2, 4, 8, 16, 32]
    
    for tx in x_candidates, ty in y_candidates, tz in z_candidates
        total = tx * ty * tz
        if total > max_threads || total < WARP_SIZE
            continue
        end
        
        bx = cld(nxp + 2*NG, tx)
        by = cld(nyp + 2*NG, ty)
        bz = cld(nzp + 2*NG, tz)
        total_blocks = bx * by * bz
        
        total_threads_launched = total_blocks * total
        total_cells = (nxp + 2*NG) * (nyp + 2*NG) * (nzp + 2*NG)
        efficiency = total_cells / total_threads_launched
        
        warp_aligned = (total % WARP_SIZE == 0) ? 0.5 : 0.0
        score = efficiency * log(total) + (total >= 128 ? 1.0 : 0.0) + warp_aligned
        
        if score > best_score
            best_score = score
            best_threads = (tx, ty, tz)
        end
    end
    
    return best_threads
end

struct KernelConfig
    name::String
    threads::Tuple{Int32, Int32, Int32}
    blocks::Tuple{Int32, Int32, Int32}
    registers::Int
    max_threads::Int
    occupancy::Float64
    maxregs::Int           # 0 = no limit, >0 = register cap applied
    base_registers::Int    # registers without maxregs limit (for spill estimation)
    compiled_kern::Any     # pre-compiled kernel object (for functional launch API)
end

# ═══════════════════════════════════════════════════════
# Backend-specific: compile kernel + query properties
# ═══════════════════════════════════════════════════════

if USE_CUDA
    @eval begin
    """Compile kernel with optional maxregs limit, return (kern, regs, max_threads, optimal_1d)"""
    function _compile_and_query(kernel_func, args...; maxregs::Int=0)
        if maxregs > 0
            kern = @cuda launch=false maxregs=maxregs kernel_func(args...)
        else
            kern = @cuda launch=false kernel_func(args...)
        end
        regs = CUDA.registers(kern)
        max_threads_hw = CUDA.maxthreads(kern)
        # Also cap by register-based limit (65536 regs / SM)
        max_threads_by_regs = regs > 0 ? (65536 ÷ regs) : 1024
        # Round down to warp boundary
        max_threads_by_regs = (max_threads_by_regs ÷ WARP_SIZE) * WARP_SIZE
        config = launch_configuration(kern.fun)
        optimal = min(config.threads, max_threads_hw, max_threads_by_regs, 512)
        while optimal > WARP_SIZE
            ab = CUDA.active_blocks(kern.fun, optimal)
            if ab > 0; break; end
            optimal = optimal ÷ 2
        end
        return kern, regs, min(max_threads_hw, max_threads_by_regs), optimal
    end
    gpu_device_name() = CUDA.name(CUDA.device())
    end  # @eval

elseif USE_ROCM
    @eval begin
        function _compile_and_query(kernel_func, args...; maxregs::Int=0)
            kern = @roc launch=false kernel_func(args...)
            
            regs = 0           # VGPRs
            scratch_size = 0   # private segment (spill) size in bytes
            max_threads_hw = 1024
            optimal = 256      # fallback
            
            # ── Query kernel properties via HIP C API ──
            # hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t func)
            # Attributes: NUM_REGS=4, LOCAL_SIZE_BYTES=3, MAX_THREADS_PER_BLOCK=0
            try
                func_handle = kern.fun.handle
                
                # Find hipFuncGetAttribute in the HIP library
                hip_lib = nothing
                for lib_name in ("libamdhip64", "libamdhip64.so", "libamdhip64.so.5")
                    try
                        hip_lib = Libdl.dlopen(lib_name; throw_error=false)
                        if hip_lib !== nothing; break; end
                    catch; end
                end
                # Also try AMDGPU's known library handle
                if hip_lib === nothing
                    try
                        hip_lib = AMDGPU.HIP.libhip
                    catch; end
                end
                
                if hip_lib !== nothing
                    func_get_attr = Libdl.dlsym(hip_lib, :hipFuncGetAttribute; throw_error=false)
                    if func_get_attr !== nothing
                        value = Ref{Int32}(0)
                        
                        # Query NUM_REGS (attribute 4)
                        status = ccall(func_get_attr, Int32,
                                       (Ptr{Int32}, Int32, Ptr{Cvoid}),
                                       value, Int32(4), func_handle)
                        if status == 0 && value[] > 0
                            regs = Int(value[])
                        end
                        
                        # Query LOCAL_SIZE_BYTES / scratch (attribute 3)
                        value[] = Int32(0)
                        status = ccall(func_get_attr, Int32,
                                       (Ptr{Int32}, Int32, Ptr{Cvoid}),
                                       value, Int32(3), func_handle)
                        if status == 0 && value[] >= 0
                            scratch_size = Int(value[])
                        end
                        
                        # Query MAX_THREADS_PER_BLOCK (attribute 0)
                        value[] = Int32(0)
                        status = ccall(func_get_attr, Int32,
                                       (Ptr{Int32}, Int32, Ptr{Cvoid}),
                                       value, Int32(0), func_handle)
                        if status == 0 && value[] > 0
                            max_threads_hw = Int(value[])
                        end
                    end
                end
            catch e
                # HIP API query failed - continue with defaults
            end

            # ── Compute occupancy for gfx906 ──
            if regs > 0
                # VGPRs allocated in blocks of 4 on gfx906
                vgpr_alloc = cld(regs, 4) * 4
                # 256 VGPRs per SIMD → max waves per SIMD
                waves_per_simd = min(256 ÷ vgpr_alloc, 10)
                waves_per_cu = waves_per_simd * 4  # 4 SIMDs per CU
                max_threads_by_vgpr = waves_per_cu * 64
                max_threads_hw = min(max_threads_hw, max_threads_by_vgpr)
                optimal = min(waves_per_simd * 64, 512)
            end
            
            # Ensure wavefront alignment
            optimal = (optimal ÷ WARP_SIZE) * WARP_SIZE
            if optimal < WARP_SIZE; optimal = WARP_SIZE; end
            
            return kern, regs, max_threads_hw, optimal
        end
        gpu_device_name() = string(AMDGPU.device())
    end

else  # CPU backend
    _compile_and_query(kernel_func, args...; maxregs::Int=0) = (nothing, 0, 1024, 128)
    gpu_device_name() = "CPU ($(Threads.nthreads()) threads)"

end

# ═══════════════════════════════════════════════════════
# Adaptive maxregs search
# ═══════════════════════════════════════════════════════

"""
    compute_occupancy(regs, threads_per_block) → Float64

Estimate SM occupancy (0.0-1.0) from register count and block size.
"""
function compute_occupancy(regs, threads_per_block)
    warps_per_block = cld(threads_per_block, WARP_SIZE)
    max_warps_per_cu = USE_ROCM ? 32 : 48
    regs_per_sm = 65536
    
    if regs <= 0
        return warps_per_block / max_warps_per_cu
    end
    
    # How many blocks fit based on registers?
    regs_per_block = regs * threads_per_block
    blocks_by_regs = regs_per_sm ÷ regs_per_block
    blocks_by_warps = max_warps_per_cu ÷ warps_per_block
    blocks_per_sm = min(blocks_by_regs, blocks_by_warps)
    
    active_warps = blocks_per_sm * warps_per_block
    return min(active_warps / max_warps_per_cu, 1.0)
end

"""
    auto_tune_kernel(name, kernel_func, args...; nxp, nyp, nzp)

Adaptive auto-tuning: tries multiple maxregs values, picks the one with
best occupancy-to-spill ratio. Spill cost is estimated as extra registers
that had to be evicted to local memory.

Strategy:
  1. Compile with no limit → get base_regs, base_occupancy
  2. If occupancy < 50%, try lower maxregs candidates
  3. Score = occupancy_gain - spill_penalty
  4. Pick highest scoring config
"""
function auto_tune_kernel(name::String, kernel_func, args...;
                          nxp::Int, nyp::Int, nzp::Int, verbose::Bool=true)
    # ── CPU: skip compilation and benchmarking, return fixed config ──
    if USE_CPU
        threads = (Int32(8), Int32(4), Int32(4))
        blocks = (Int32(cld(nxp + 2*NG, 8)),
                  Int32(cld(nyp + 2*NG, 4)),
                  Int32(cld(nzp + 2*NG, 4)))
        if verbose
            t_str = @sprintf("(%d,%d,%d)", threads[1], threads[2], threads[3])
            b_str = @sprintf("(%d,%d,%d)", blocks[1], blocks[2], blocks[3])
            @printf("║  %-22s  %-6s  %-12s  %-14s  %5.1f%%  %8s  ║\n",
                    name, "CPU", t_str, b_str, 100.0, "n/a")
        end
        return KernelConfig(name, threads, blocks, 0, 1024, 100.0, 0, 0, nothing)
    end

    # Step 1: baseline (no maxregs limit)
    base_kern, base_regs, base_max_threads, base_optimal = _compile_and_query(kernel_func, args...)
    base_3d = find_best_3d_block(base_optimal, nxp, nyp, nzp)
    # Enforce kernel-specific max thread limit
    while prod(base_3d) > min(base_optimal, base_max_threads)
        base_optimal = base_optimal ÷ 2
        if base_optimal < WARP_SIZE; base_optimal = WARP_SIZE; break; end
        base_3d = find_best_3d_block(base_optimal, nxp, nyp, nzp)
    end
    # Safety: high-reg kernels may have stricter actual limits than queried
    if base_regs >= 96 && prod(base_3d) > 256
        base_optimal = 256
        base_3d = find_best_3d_block(base_optimal, nxp, nyp, nzp)
    end
    base_occ = compute_occupancy(base_regs, prod(base_3d))
    
    # Best config so far
    best_kern = base_kern
    best_regs = base_regs
    best_3d = base_3d
    best_occ = base_occ
    best_maxregs = 0  # 0 = no limit
    best_optimal = base_optimal

    if USE_ROCM
        # ── ROCM Runtime Auto-Tune ──
        # Print kernel resource info if available
        if verbose
            @printf("║  %-22s  VGPRs=%3d  Scratch=%5d B  maxThreads=%d\n",
                    name, base_regs, 0, base_max_threads)
        end
        
        candidate_configs = [
            # 128 threads = 2 wavefronts
            (4, 4, 8),
            (8, 4, 4),
            # 256 threads = 4 wavefronts
            (4, 8, 8),
            (8, 4, 8),
            (8, 8, 4),
            (4, 4, 16),
            # 512 threads = 8 wavefronts
            (8, 8, 8),
            (16, 4, 8),
            (16, 8, 4),
            (8, 16, 4),
            # 1024 threads = 16 wavefronts (if registers allow)
            (16, 8, 8),
            (8, 8, 16),
            (16, 16, 4),
            (32, 8, 4),
        ]

        best_time = Inf
        nwarmup = 5
        nruns = 20

        for cfg in candidate_configs
            tx, ty, tz = cfg
            total = tx * ty * tz
            if total > base_max_threads; continue; end

            # ── Occupancy floor: skip configs with <25% occupancy ──
            if base_regs > 0
                vgpr_alloc = cld(base_regs, 4) * 4
                waves_per_simd = min(256 ÷ vgpr_alloc, 10)
                waves_in_block = total ÷ 64
                max_blocks_by_vgpr = waves_per_simd * 4 ÷ max(waves_in_block, 1)
                max_blocks_by_waves = (waves_per_simd * 4) ÷ max(waves_in_block, 1)
                active_waves = min(max_blocks_by_vgpr, max_blocks_by_waves) * waves_in_block
                cfg_occ = active_waves / (waves_per_simd * 4)
                if cfg_occ < 0.25
                    continue  # skip low-occupancy configs
                end
            end

            bx = cld(nxp + 2*NG, tx)
            by = cld(nyp + 2*NG, ty)
            bz = cld(nzp + 2*NG, tz)

            try
                for _ in 1:nwarmup
                    @gpu_launch threads=(Int32(tx), Int32(ty), Int32(tz)) blocks=(bx, by, bz) kernel_func(args...)
                    gpu_sync()
                end

                gpu_sync()
                t0 = time_ns()
                for _ in 1:nruns
                    @gpu_launch threads=(Int32(tx), Int32(ty), Int32(tz)) blocks=(bx, by, bz) kernel_func(args...)
                end
                gpu_sync()
                elapsed = (time_ns() - t0) / nruns

                if elapsed < best_time
                    best_time = elapsed
                    best_3d = cfg
                    # Compute occupancy correctly: account for multiple blocks per CU
                    if base_regs > 0
                        vgpr_alloc_b = cld(base_regs, 4) * 4
                        waves_per_simd_b = min(256 ÷ vgpr_alloc_b, 10)
                        waves_in_block_b = total ÷ 64
                        max_blk = (waves_per_simd_b * 4) ÷ max(waves_in_block_b, 1)
                        active_w = min(max_blk, 16) * waves_in_block_b  # cap at 16 blocks per CU
                        best_occ = min(active_w / (waves_per_simd_b * 4), 1.0) * 100.0
                    else
                        wavefronts = total ÷ WARP_SIZE
                        best_occ = min(wavefronts / 32, 1.0) * 100.0
                    end
                end
            catch
                continue
            end
        end

    elseif USE_CUDA && base_occ < 0.75
        # ── CUDA maxregs Auto-Tune (existing logic) ──
        target_blocks = [2, 3, 4]
        threads_per_block = prod(base_3d)
        
        candidates = Int[]
        for nb in target_blocks
            target_regs = 65536 ÷ (nb * threads_per_block)
            if target_regs >= 24 && target_regs < base_regs
                push!(candidates, target_regs)
            end
        end
        for r in [48, 40, 32]
            if r < base_regs && !(r in candidates)
                push!(candidates, r)
            end
        end
        sort!(unique!(candidates), rev=true)
        
        for mr in candidates
            try
                kern, regs, max_thr, optimal = _compile_and_query(kernel_func, args...; maxregs=mr)
                test_3d = find_best_3d_block(optimal, nxp, nyp, nzp)
                while prod(test_3d) > optimal
                    optimal = optimal ÷ 2
                    test_3d = find_best_3d_block(optimal, nxp, nyp, nzp)
                end
                occ = compute_occupancy(regs, prod(test_3d))
                
                occ_gain = occ - base_occ
                spill_fraction = max(0, base_regs - regs) / base_regs
                score = occ_gain - 1.5 * spill_fraction
                
                if score > 0 && occ > best_occ
                    best_kern = kern
                    best_regs = regs
                    best_3d = test_3d
                    best_occ = occ
                    best_maxregs = mr
                    best_optimal = optimal
                end
            catch
                continue
            end
        end
    end
    
    threads = (Int32(best_3d[1]), Int32(best_3d[2]), Int32(best_3d[3]))
    blocks = (Int32(cld(nxp + 2*NG, best_3d[1])),
              Int32(cld(nyp + 2*NG, best_3d[2])),
              Int32(cld(nzp + 2*NG, best_3d[3])))
    
    occ_val = USE_ROCM ? best_occ : best_occ * 100.0
    return KernelConfig(name, threads, blocks, best_regs, base_max_threads,
                        occ_val, best_maxregs, base_regs, best_kern)
end

# ═══════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════

function print_tune_report(configs::Vector{KernelConfig}, gpu_name::String)
    backend = USE_ROCM ? "ROCm/AMDGPU" : "CUDA"
    warp_label = USE_ROCM ? "wavefront" : "warp"
    println()
    println("╔════════════════════════════════════════════════════════════════════════════════════╗")
    println("║  Kernel Auto-Tune ($backend, $warp_label=$WARP_SIZE)")
    println("║  GPU: $gpu_name")
    println("╠════════════════════════════════════════════════════════════════════════════════════╣")
    @printf("║  %-22s  %-6s  %-12s  %-14s  %6s  %8s  ║\n",
            "Kernel", "VGPRs", "Threads", "Blocks", "Occup.", "maxregs")
    println("║────────────────────────────────────────────────────────────────────────────────────║")
    for c in configs
        t_str = @sprintf("(%d,%d,%d)", c.threads[1], c.threads[2], c.threads[3])
        b_str = @sprintf("(%d,%d,%d)", c.blocks[1], c.blocks[2], c.blocks[3])
        mr_str = c.maxregs > 0 ? string(c.maxregs) : "auto"
        vgpr_str = c.registers > 0 ? string(c.registers) : "n/a"
        @printf("║  %-22s  %-6s  %-12s  %-14s  %5.1f%%  %8s  ║\n",
                c.name, vgpr_str, t_str, b_str, c.occupancy, mr_str)
    end
    println("╚════════════════════════════════════════════════════════════════════════════════════╝")
    println()
end
