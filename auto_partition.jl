# auto_partition.jl — Automatic GPU load balancing for multi-block solver
# Distributes MPI ranks across blocks proportionally to cell count,
# then factorizes each block's rank count into optimal 3D partitions.

using StaticArrays
using Printf

"""
    estimate_gpu_memory_per_rank(Nx, Ny, Nz, px, py, pz, NG, Ncons, Nprim)

Estimate GPU memory usage (in bytes) for a single rank handling a subdomain
of block (Nx, Ny, Nz) partitioned into (px, py, pz).

Memory components:
  - Block arrays: Q(Nprim), U(Ncons), Un(Nprim), ϕ(1), Vol(1), LTS_dt(1),
    x/y/z(3), Area_i/j/k(3), nx/ny/nz per i/j/k(9), Ωx/Ωy/Ωz(3) = 26 scalar fields
  - MPI buffers (GPU-side): 6 send+recv × NG slabs × Nprim
  - Shared flux buffers: Fx/Fy/Fz/Fvx/Fvy/Fvz(6) × Ncons + dU_forced(Ncons)
"""
function estimate_gpu_memory_per_rank(Nx, Ny, Nz, px, py, pz, NG, Ncons, Nprim)
    # Local subdomain size (including ghost cells)
    nx_local = cld(Nx, px) + 2*NG
    ny_local = cld(Ny, py) + 2*NG
    nz_local = cld(Nz, pz) + 2*NG
    
    ncells = nx_local * ny_local * nz_local
    sizeof_f32 = 4  # Float32 = 4 bytes
    
    # 4D arrays: Q(Nprim) + U(Ncons) + Un(Nprim) = Nprim + Ncons + Nprim
    mem_4d = ncells * (Nprim + Ncons + Nprim) * sizeof_f32
    
    # 3D arrays: ϕ, Vol, LTS_dt, x, y, z, Areai/j/k, nxi/nyi/nzi * 3, Ωx/Ωy/Ωz
    # = 1 + 1 + 1 + 3 + 3 + 9 + 3 = 21 scalar fields
    mem_3d = ncells * 21 * sizeof_f32
    
    # MPI GPU buffers (send + recv, 2 per direction, 3 directions, GPU side only)
    # x-slabs: NG × ny_local × nz_local × Nprim × 2(send+recv)
    # y-slabs: nx_local × NG × nz_local × Nprim × 2
    # z-slabs: nx_local × ny_local × NG × Nprim × 2
    mem_mpi = 2 * sizeof_f32 * Nprim * (
        NG * ny_local * nz_local +
        nx_local * NG * nz_local +
        nx_local * ny_local * NG
    )
    
    # Shared flux buffers (allocated once, sized to max local block)
    # 6 flux arrays × (nx+1) × ny × nz × Ncons + dU_forced
    nx_r = cld(Nx, px); ny_r = cld(Ny, py); nz_r = cld(Nz, pz)
    mem_flux = sizeof_f32 * Ncons * (
        (nx_r+1)*ny_r*nz_r +  # Fx
        nx_r*(ny_r+1)*nz_r +  # Fy
        nx_r*ny_r*(nz_r+1) +  # Fz
        (nx_r+1)*ny_r*nz_r +  # Fvx
        nx_r*(ny_r+1)*nz_r +  # Fvy
        nx_r*ny_r*(nz_r+1) +  # Fvz
        nx_r*ny_r*nz_r         # dU_forced
    )
    
    return mem_4d + mem_3d + mem_mpi + mem_flux
end

"""
    best_partition_3d(Nx, Ny, Nz, nranks; weights=(1.0, 1.0, 1.0))

Find (px, py, pz) with px*py*pz = nranks that minimizes inter-rank
communication surface area. The `weights` multiplier can be used to
penalize splitting in specific directions (e.g. `weights=(wx, wy, wz)`).
Note: For meshes with extreme aspect ratios (like pipe flows where Nx >> Ny, Nz),
the unweighted algorithm naturally partitions into 1D (px, 1, 1) to minimize surface area.
"""
function best_partition_3d(Nx, Ny, Nz, nranks; weights=(1.0, 1.0, 1.0))
    best = SVector{3,Int}(nranks, 1, 1)
    best_cost = Inf
    
    wx, wy, wz = weights
    
    for px in 1:nranks
        nranks % px != 0 && continue
        rem = nranks ÷ px
        for py in 1:rem
            rem % py != 0 && continue
            pz = rem ÷ py
            
            nx_local, ny_local, nz_local = Nx ÷ px, Ny ÷ py, Nz ÷ pz
            
            # Avoid excessively thin domains (e.g., fewer than 4 cells) 
            # which could cause issues with ghost cells
            if nx_local < 4 || ny_local < 4 || nz_local < 4
                continue
            end
            
            # Communication surface area (lower = less MPI overhead)
            # weights can artificially penalize non-contiguous memory access
            cost = wx * Float64(ny_local) * nz_local +   # x-faces
                   wy * Float64(nx_local) * nz_local +   # y-faces
                   wz * Float64(nx_local) * ny_local     # z-faces
                   
            if cost < best_cost
                best_cost = cost
                best = SVector{3,Int}(px, py, pz)
            end
        end
    end
    
    # Fallback to (nranks, 1, 1) if all configurations failed the minimum size check
    if best_cost == Inf
        best = SVector{3,Int}(nranks, 1, 1)
    end
    
    return best
end

"""
    auto_partition(Nx_b, Ny_b, Nz_b, N_gpus; NG=4, Ncons=5, Nprim=6, verbose=true)

Automatically distribute `N_gpus` MPI ranks across blocks proportionally
to their cell count, then compute optimal 3D partitions for each block.

Returns: Vector{SVector{3,Int}} — the Block_Nprocs array

Algorithm:
1. Compute cell count per block → proportional rank allocation (Hamilton method)
2. For each block, find optimal (px, py, pz) factorization
3. Estimate GPU memory per rank and warn if exceeding threshold
"""
function auto_partition(Nx_b, Ny_b, Nz_b, N_gpus; NG=4, Ncons=5, Nprim=6, verbose=true, gpu_vram_gb=16.0)
    Nblocks = length(Nx_b)
    cells = [Nx_b[i] * Ny_b[i] * Nz_b[i] for i in 1:Nblocks]
    
    # ══════════════════════════════════════════════════════════════
    #  Case 1: N_gpus < Nblocks → multi-block-per-rank mode
    #  Each block gets partition (1,1,1), assigned via greedy
    #  cell-count-balanced allocation (no sub-domain splitting)
    # ══════════════════════════════════════════════════════════════
    if N_gpus < Nblocks
        partitions = [SVector{3,Int}(1, 1, 1) for _ in 1:Nblocks]
        
        # Greedy assignment: assign each block (largest first) to
        # the rank with the fewest total cells so far
        block_order = sortperm(cells, rev=true)  # largest block first
        rank_load = zeros(Int, N_gpus)  # cell count on each rank
        block_to_rank = zeros(Int, Nblocks)
        
        for idx in block_order
            # Find the rank with minimum current load
            target_rank = argmin(rank_load) - 1  # 0-indexed rank
            block_to_rank[idx] = target_rank
            rank_load[target_rank + 1] += cells[idx]
        end
        
        if verbose
            println("  ┌─────────────────────────────────────────────────────────────────────")
            println("  │ Multi-Block-per-Rank Mode: $N_gpus ranks → $Nblocks blocks")
            println("  │ (cell-count-balanced, no sub-domain splitting)")
            println("  ├─────────────────────────────────────────────────────────────────────")
            @printf("  │ %-8s  %-18s  %-10s  %-14s\n",
                "Block", "Grid", "Cells", "→ Rank")
            println("  │ ", "─"^55)
            for i in 1:Nblocks
                grid_str = "$(Nx_b[i])×$(Ny_b[i])×$(Nz_b[i])"
                @printf("  │ %-8d  %-18s  %-10d  rank %d\n",
                    i-1, grid_str, cells[i], block_to_rank[i])
            end
            println("  ├─────────────────────────────────────────────────────────────────────")
            for r in 0:N_gpus-1
                blocks_on_r = findall(x -> x == r, block_to_rank) .- 1
                @printf("  │ Rank %d: blocks %s  (%.0f cells)\n",
                    r, isempty(blocks_on_r) ? "none" : join(blocks_on_r, ","), rank_load[r+1])
            end
            println("  └─────────────────────────────────────────────────────────────────────")
        end
        
        return partitions, block_to_rank
    end
    
    # ══════════════════════════════════════════════════════════════
    #  Case 2: N_gpus >= Nblocks → standard sub-domain splitting
    #  Each block gets 1+ ranks, optimal 3D partitioning
    # ══════════════════════════════════════════════════════════════
    total_cells = sum(cells)
    
    # Ideal (fractional) allocation
    ideal = [cells[i] / total_cells * N_gpus for i in 1:Nblocks]
    
    # Floor allocation (minimum 1 rank per block)
    ranks = [max(1, floor(Int, f)) for f in ideal]
    
    # Distribute remaining ranks using largest-remainder (Hamilton) method
    remaining = N_gpus - sum(ranks)
    if remaining > 0
        remainders = [(ideal[i] - ranks[i], i) for i in 1:Nblocks]
        sort!(remainders, by=x -> -x[1])
        for k in 1:min(remaining, Nblocks)
            ranks[remainders[k][2]] += 1
        end
    end
    
    # Optimal 3D factorization per block
    partitions = Vector{SVector{3,Int}}(undef, Nblocks)
    for i in 1:Nblocks
        partitions[i] = best_partition_3d(Nx_b[i], Ny_b[i], Nz_b[i], ranks[i])
    end
    
    # block_to_rank: first rank of each block's group (for consistency)
    rank_offsets = zeros(Int, Nblocks + 1)
    for i in 1:Nblocks
        rank_offsets[i+1] = rank_offsets[i] + prod(partitions[i])
    end
    block_to_rank = [rank_offsets[i] for i in 1:Nblocks]
    
    # Memory estimation & reporting
    if verbose
        gpu_vram_bytes = gpu_vram_gb * 1024^3
        println("  ┌─────────────────────────────────────────────────────────────────────")
        println("  │ Auto GPU Partition: $N_gpus GPUs → $Nblocks blocks")
        println("  ├─────────────────────────────────────────────────────────────────────")
        @printf("  │ %-8s  %-18s  %-8s  %-14s  %-12s  %s\n",
            "Block", "Grid", "Ranks", "Partition", "Mem/Rank", "Status")
        println("  │ ", "─"^70)
        
        for i in 1:Nblocks
            px, py, pz = partitions[i]
            mem = estimate_gpu_memory_per_rank(Nx_b[i], Ny_b[i], Nz_b[i], px, py, pz, NG, Ncons, Nprim)
            mem_gb = mem / 1024^3
            status = mem < gpu_vram_bytes * 0.8 ? "✓" : (mem < gpu_vram_bytes ? "⚠ tight" : "✗ OOM!")
            grid_str = "$(Nx_b[i])×$(Ny_b[i])×$(Nz_b[i])"
            part_str = "($px,$py,$pz)"
            @printf("  │ %-8d  %-18s  %-8d  %-14s  %7.2f GB    %s\n",
                i-1, grid_str, ranks[i], part_str, mem_gb, status)
        end
        
        total_mem_max = maximum(
            estimate_gpu_memory_per_rank(Nx_b[i], Ny_b[i], Nz_b[i], 
                partitions[i][1], partitions[i][2], partitions[i][3], NG, Ncons, Nprim)
            for i in 1:Nblocks
        ) / 1024^3
        println("  ├─────────────────────────────────────────────────────────────────────")
        @printf("  │ Total ranks: %d | Max mem/rank: %.2f GB | VRAM limit: %.1f GB\n",
            sum(ranks), total_mem_max, gpu_vram_gb)
        println("  └─────────────────────────────────────────────────────────────────────")
    end
    
    return partitions, block_to_rank
end

