using MPI
using StaticArrays
using HDF5, DelimitedFiles
using Dates, Printf

# GPU backend: loads CUDA or AMDGPU depending on what's available
include("gpu_backend.jl")
gpu_allowscalar(false)

include("bc_types.jl")
include("schemes.jl")
include("viscous.jl")
include("dsrfg_inflow.jl")
const default_dsrfg_params = create_dummy_dsrfg_params(FT)
include("boundary.jl")
include("utils.jl")
# ═══════════════════════════════════════════════════════════════════════════════
# Interface Stencil Tapering & Cross-Type Protection
# ═══════════════════════════════════════════════════════════════════════════════

# Central stencils (symmetric about the face)
const _CD2_C = Float64[0.0,  0.0,     0.0,    1/2,    1/2,    0.0,     0.0]
const _CD4_C = Float64[0.0,  0.0,    -1/16,   9/16,   9/16,  -1/16,    0.0]
const _CD6_C = Float64[0.0,  1/60,   -2/15,  37/60,  37/60,  -2/15,   1/60]

# Upwind stencils (biased toward the L-side cell j)
const _UP1_U = Float64[0.0,  0.0,     0.0,    1.0,    0.0,    0.0,     0.0]
const _UP3_U = Float64[0.0,  0.0,    -1/6,    5/6,    1/3,    0.0,     0.0]
const _UP5_U = Float64[0.0,  2/60,  -13/60,  47/60,  27/60,  -3/60,   0.0]

# Delta = upwind - central for each level
const _Δ_CD2 = _UP1_U .- _CD2_C
const _Δ_CD4 = _UP3_U .- _CD4_C
const _Δ_CD6 = _UP5_U .- _CD6_C

# Upwind boost at interblock boundaries
const _INTERFACE_PHI = Float64[0.50, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.08]
const _INTERFACE_PHI_K = Float64[0.50, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.08]

const _CROSSTYPE_N_PROTECT = 4
const _CROSSTYPE_PHI = Float64[1.0, 1.0, 0.80, 0.60]
const _GEOM_ANGLE_THRESHOLD = 5.0      # degrees
const _GEOM_STRETCH_THRESHOLD = 1.2    # ratio

function _boost_phi_at_interface!(Lin::Array{T,3}, ΔLin::Array{T,3}, phi::Matrix,
                                  N_real::Int, NG_val::Int,
                                  is_lo_interblock::Bool, is_hi_interblock::Bool) where T
    n_boost = length(_INTERFACE_PHI)
    N2 = size(phi, 2)
    FTl = eltype(phi)

    if is_lo_interblock
        for d in 0:(n_boost-1)
            idx = NG_val + d
            if idx < 1 || idx > size(phi, 1); continue; end
            boost_val = FTl(_INTERFACE_PHI[d+1])
            for i2 in 1:N2
                if boost_val > phi[idx, i2]
                    Δϕ = boost_val - phi[idx, i2]
                    for s in 1:7
                        Lin[idx, i2, s] += FTl(Δϕ) * ΔLin[idx, i2, s]
                    end
                    phi[idx, i2] = boost_val
                end
            end
        end
    end

    if is_hi_interblock
        for d in 0:(n_boost-1)
            idx = N_real + NG_val - d
            if idx < 1 || idx > size(phi, 1); continue; end
            boost_val = FTl(_INTERFACE_PHI[d+1])
            for i2 in 1:N2
                if boost_val > phi[idx, i2]
                    Δϕ = boost_val - phi[idx, i2]
                    for s in 1:7
                        Lin[idx, i2, s] += FTl(Δϕ) * ΔLin[idx, i2, s]
                    end
                    phi[idx, i2] = boost_val
                end
            end
        end
    end
end

function _boost_phi_at_interface_dim2!(Lin::Array{T,3}, ΔLin::Array{T,3}, phi::Matrix,
                                       N_real::Int, NG_val::Int,
                                       is_lo_interblock::Bool, is_hi_interblock::Bool,
                                       boost_arr::Vector{Float64}) where T
    n_boost = length(boost_arr)
    N1 = size(phi, 1)
    FTl = eltype(phi)

    if is_lo_interblock
        for d in 0:(n_boost-1)
            idx = NG_val + d
            if idx < 1 || idx > size(phi, 2); continue; end
            boost_val = FTl(boost_arr[d+1])
            for i1 in 1:N1
                if boost_val > phi[i1, idx]
                    Δϕ = boost_val - phi[i1, idx]
                    for s in 1:7
                        Lin[i1, idx, s] += FTl(Δϕ) * ΔLin[i1, idx, s]
                    end
                    phi[i1, idx] = boost_val
                end
            end
        end
    end

    if is_hi_interblock
        for d in 0:(n_boost-1)
            idx = N_real + NG_val - d
            if idx < 1 || idx > size(phi, 2); continue; end
            boost_val = FTl(boost_arr[d+1])
            for i1 in 1:N1
                if boost_val > phi[i1, idx]
                    Δϕ = boost_val - phi[i1, idx]
                    for s in 1:7
                        Lin[i1, idx, s] += FTl(Δϕ) * ΔLin[i1, idx, s]
                    end
                    phi[i1, idx] = boost_val
                end
            end
        end
    end
end

function apply_interface_taper!(Lin_j, ΔLin_j, phi_j,
                                Lin_k, ΔLin_k, phi_k,
                                face_bc, bid::Int, nyp::Int, nzp::Int, NG_val::Int;
                                verbose::Bool=false)
    bc_jlo = get(face_bc, (bid, 3), Int32(-1))
    bc_jhi = get(face_bc, (bid, 4), Int32(-1))
    bc_klo = get(face_bc, (bid, 5), Int32(-1))
    bc_khi = get(face_bc, (bid, 6), Int32(-1))

    jlo_inter = (bc_jlo == Int32(0))
    jhi_inter = (bc_jhi == Int32(0))
    klo_inter = (bc_klo == Int32(0))
    khi_inter = (bc_khi == Int32(0))

    # Order-reduction stencils for layers 0, 1, 2 near interblock faces
    order_central = (_CD2_C, _CD4_C, _CD6_C)
    order_delta   = (_Δ_CD2, _Δ_CD4, _Δ_CD6)
    order_phi     = (0.50,   0.50,   0.40)
    order_names   = ("2nd",  "4th",  "6th")
    n_replace = length(order_central)   # 3 layers get full stencil replacement
    n_boost   = length(_INTERFACE_PHI)  # 8 layers total

    # ─── η direction (dim-1 index) ───
    if jlo_inter || jhi_inter
        FTl = eltype(phi_j)
        N2  = size(phi_j, 2)

        if jlo_inter
            # Layers 0..2: full stencil replacement
            for d in 0:(n_replace - 1)
                idx = NG_val + d
                if idx < 1 || idx > size(phi_j, 1); continue; end
                phi_val = FTl(order_phi[d+1])
                for i2 in 1:N2
                    for s in 1:7
                        Lin_j[idx, i2, s]  = FTl(order_central[d+1][s]) + phi_val * FTl(order_delta[d+1][s])
                        ΔLin_j[idx, i2, s] = FTl(order_delta[d+1][s])
                    end
                    phi_j[idx, i2] = phi_val
                end
            end
            # Layers 3..n_boost-1: phi boost only (preserve existing stencil width)
            for d in n_replace:(n_boost - 1)
                idx = NG_val + d
                if idx < 1 || idx > size(phi_j, 1); continue; end
                boost_val = FTl(_INTERFACE_PHI[d+1])
                for i2 in 1:N2
                    if boost_val > phi_j[idx, i2]
                        Δϕ = boost_val - phi_j[idx, i2]
                        for s in 1:7
                            Lin_j[idx, i2, s] += FTl(Δϕ) * ΔLin_j[idx, i2, s]
                        end
                        phi_j[idx, i2] = boost_val
                    end
                end
            end
        end

        if jhi_inter
            # Layers 0..2: full stencil replacement
            for d in 0:(n_replace - 1)
                idx = nyp + NG_val - d
                if idx < 1 || idx > size(phi_j, 1); continue; end
                phi_val = FTl(order_phi[d+1])
                for i2 in 1:N2
                    for s in 1:7
                        Lin_j[idx, i2, s]  = FTl(order_central[d+1][s]) + phi_val * FTl(order_delta[d+1][s])
                        ΔLin_j[idx, i2, s] = FTl(order_delta[d+1][s])
                    end
                    phi_j[idx, i2] = phi_val
                end
            end
            # Layers 3..n_boost-1: phi boost only
            for d in n_replace:(n_boost - 1)
                idx = nyp + NG_val - d
                if idx < 1 || idx > size(phi_j, 1); continue; end
                boost_val = FTl(_INTERFACE_PHI[d+1])
                for i2 in 1:N2
                    if boost_val > phi_j[idx, i2]
                        Δϕ = boost_val - phi_j[idx, i2]
                        for s in 1:7
                            Lin_j[idx, i2, s] += FTl(Δϕ) * ΔLin_j[idx, i2, s]
                        end
                        phi_j[idx, i2] = boost_val
                    end
                end
            end
        end

        if verbose
            println("    Block $bid η-interface: lo=$(jlo_inter) hi=$(jhi_inter) " *
                    "(layers 0-2: order reduction 2nd→4th→6th, layers 3-7: phi boost)")
        end
    end

    # ─── ζ direction (dim-2 index) ───
    if klo_inter || khi_inter
        FTl = eltype(phi_k)
        N1  = size(phi_k, 1)

        if klo_inter
            # Layers 0..2: full stencil replacement
            for d in 0:(n_replace - 1)
                idx = NG_val + d
                if idx < 1 || idx > size(phi_k, 2); continue; end
                phi_val = FTl(order_phi[d+1])
                for i1 in 1:N1
                    for s in 1:7
                        Lin_k[i1, idx, s]  = FTl(order_central[d+1][s]) + phi_val * FTl(order_delta[d+1][s])
                        ΔLin_k[i1, idx, s] = FTl(order_delta[d+1][s])
                    end
                    phi_k[i1, idx] = phi_val
                end
            end
            # Layers 3..n_boost-1: phi boost only
            for d in n_replace:(n_boost - 1)
                idx = NG_val + d
                if idx < 1 || idx > size(phi_k, 2); continue; end
                boost_val = FTl(_INTERFACE_PHI_K[d+1])
                for i1 in 1:N1
                    if boost_val > phi_k[i1, idx]
                        Δϕ = boost_val - phi_k[i1, idx]
                        for s in 1:7
                            Lin_k[i1, idx, s] += FTl(Δϕ) * ΔLin_k[i1, idx, s]
                        end
                        phi_k[i1, idx] = boost_val
                    end
                end
            end
        end

        if khi_inter
            # Layers 0..2: full stencil replacement
            for d in 0:(n_replace - 1)
                idx = nzp + NG_val - d
                if idx < 1 || idx > size(phi_k, 2); continue; end
                phi_val = FTl(order_phi[d+1])
                for i1 in 1:N1
                    for s in 1:7
                        Lin_k[i1, idx, s]  = FTl(order_central[d+1][s]) + phi_val * FTl(order_delta[d+1][s])
                        ΔLin_k[i1, idx, s] = FTl(order_delta[d+1][s])
                    end
                    phi_k[i1, idx] = phi_val
                end
            end
            # Layers 3..n_boost-1: phi boost only
            for d in n_replace:(n_boost - 1)
                idx = nzp + NG_val - d
                if idx < 1 || idx > size(phi_k, 2); continue; end
                boost_val = FTl(_INTERFACE_PHI_K[d+1])
                for i1 in 1:N1
                    if boost_val > phi_k[i1, idx]
                        Δϕ = boost_val - phi_k[i1, idx]
                        for s in 1:7
                            Lin_k[i1, idx, s] += FTl(Δϕ) * ΔLin_k[i1, idx, s]
                        end
                        phi_k[i1, idx] = boost_val
                    end
                end
            end
        end

        if verbose
            println("    Block $bid ζ-interface: lo=$(klo_inter) hi=$(khi_inter) " *
                    "(layers 0-2: order reduction 2nd→4th→6th, layers 3-7: phi boost)")
        end
    end
end

function apply_geometric_smoothness_protection!(
    Lin_j, ΔLin_j, phi_j,
    Lin_k, ΔLin_k, phi_k,
    Areaj_h, nxj_h, nyj_h, nzj_h,
    Areak_h, nxk_h, nyk_h, nzk_h,
    face_bc, bid::Int, nxp::Int, nyp::Int, nzp::Int, NG_val::Int;
    verbose::Bool=false
)
    # Cosine threshold for normal angle jump (e.g. 5 degrees)
    cos_threshold = cos(deg2rad(_GEOM_ANGLE_THRESHOLD))
    # Stretch ratio threshold (e.g. 1.2)
    stretch_threshold = _GEOM_STRETCH_THRESHOLD
    n_protect = min(_CROSSTYPE_N_PROTECT, length(_CROSSTYPE_PHI))
    FTl = eltype(phi_j)

    bc_jlo = get(face_bc, (bid, 3), Int32(-1))
    bc_jhi = get(face_bc, (bid, 4), Int32(-1))
    bc_klo = get(face_bc, (bid, 5), Int32(-1))
    bc_khi = get(face_bc, (bid, 6), Int32(-1))

    # ─── 1. η direction (phi_j) ───
    if bc_jlo == Int32(0) # η- lo boundary
        for k in (NG_val+1):(nzp+NG_val)
            violated = false
            local_cos = 1.0
            local_r = 1.0
            for i in (NG_val+1):(nxp+NG_val)
                # Face 1 (interface face: NG_val+1)
                A1 = Areaj_h[i, NG_val+1, k]
                nx1 = nxj_h[i, NG_val+1, k]; ny1 = nyj_h[i, NG_val+1, k]; nz1 = nzj_h[i, NG_val+1, k]
                # Face 2 (adjacent interior face: NG_val+2)
                A2 = Areaj_h[i, NG_val+2, k]
                nx2 = nxj_h[i, NG_val+2, k]; ny2 = nyj_h[i, NG_val+2, k]; nz2 = nzj_h[i, NG_val+2, k]

                local_cos = nx1*nx2 + ny1*ny2 + nz1*nz2
                local_r = A1 / (A2 + 1e-30)
                if local_cos < cos_threshold || local_r > stretch_threshold || local_r < (1.0 / stretch_threshold)
                    violated = true
                    break
                end
            end
            if violated
                if verbose
                    println("      > Block $bid η-lo column k=$k violated smoothness (cos=$(round(local_cos, digits=5)), ratio=$(round(local_r, digits=5))). Boosting phi...")
                end
                for d in 0:(n_protect-1)
                    idx = NG_val + d
                    boost_val = FTl(_CROSSTYPE_PHI[d+1])
                    if boost_val > phi_j[idx, k]
                        Δϕ = boost_val - phi_j[idx, k]
                        for s in 1:7
                            Lin_j[idx, k, s] += FTl(Δϕ) * ΔLin_j[idx, k, s]
                        end
                        phi_j[idx, k] = boost_val
                    end
                end
            end
        end
    end

    if bc_jhi == Int32(0) # η+ hi boundary
        for k in (NG_val+1):(nzp+NG_val)
            violated = false
            local_cos = 1.0
            local_r = 1.0
            for i in (NG_val+1):(nxp+NG_val)
                # Face 1 (interface face: nyp+NG_val+1)
                A1 = Areaj_h[i, nyp+NG_val+1, k]
                nx1 = nxj_h[i, nyp+NG_val+1, k]; ny1 = nyj_h[i, nyp+NG_val+1, k]; nz1 = nzj_h[i, nyp+NG_val+1, k]
                # Face 2 (adjacent interior face: nyp+NG_val)
                A2 = Areaj_h[i, nyp+NG_val, k]
                nx2 = nxj_h[i, nyp+NG_val, k]; ny2 = nyj_h[i, nyp+NG_val, k]; nz2 = nzj_h[i, nyp+NG_val, k]

                local_cos = nx1*nx2 + ny1*ny2 + nz1*nz2
                local_r = A1 / (A2 + 1e-30)
                if local_cos < cos_threshold || local_r > stretch_threshold || local_r < (1.0 / stretch_threshold)
                    violated = true
                    break
                end
            end
            if violated
                if verbose
                    println("      > Block $bid η-hi column k=$k violated smoothness (cos=$(round(local_cos, digits=5)), ratio=$(round(local_r, digits=5))). Boosting phi...")
                end
                for d in 0:(n_protect-1)
                    idx = nyp + NG_val - d
                    boost_val = FTl(_CROSSTYPE_PHI[d+1])
                    if boost_val > phi_j[idx, k]
                        Δϕ = boost_val - phi_j[idx, k]
                        for s in 1:7
                            Lin_j[idx, k, s] += FTl(Δϕ) * ΔLin_j[idx, k, s]
                        end
                        phi_j[idx, k] = boost_val
                    end
                end
            end
        end
    end

    # ─── 2. ζ direction (phi_k) ───
    if bc_klo == Int32(0) # ζ- lo boundary
        for j in (NG_val+1):(nyp+NG_val)
            violated = false
            local_cos = 1.0
            local_r = 1.0
            for i in (NG_val+1):(nxp+NG_val)
                # Face 1 (interface face: NG_val+1)
                A1 = Areak_h[i, j, NG_val+1]
                nx1 = nxk_h[i, j, NG_val+1]; ny1 = nyk_h[i, j, NG_val+1]; nz1 = nzk_h[i, j, NG_val+1]
                # Face 2 (adjacent interior face: NG_val+2)
                A2 = Areak_h[i, j, NG_val+2]
                nx2 = nxk_h[i, j, NG_val+2]; ny2 = nyk_h[i, j, NG_val+2]; nz2 = nzk_h[i, j, NG_val+2]

                local_cos = nx1*nx2 + ny1*ny2 + nz1*nz2
                local_r = A1 / (A2 + 1e-30)
                if local_cos < cos_threshold || local_r > stretch_threshold || local_r < (1.0 / stretch_threshold)
                    violated = true
                    break
                end
            end
            if violated
                if verbose
                    println("      > Block $bid ζ-lo row j=$j violated smoothness (cos=$(round(local_cos, digits=5)), ratio=$(round(local_r, digits=5))). Boosting phi...")
                end
                for d in 0:(n_protect-1)
                    idx = NG_val + d
                    boost_val = FTl(_CROSSTYPE_PHI[d+1])
                    if boost_val > phi_k[j, idx]
                        Δϕ = boost_val - phi_k[j, idx]
                        for s in 1:7
                            Lin_k[j, idx, s] += FTl(Δϕ) * ΔLin_k[j, idx, s]
                        end
                        phi_k[j, idx] = boost_val
                    end
                end
            end
        end
    end

    if bc_khi == Int32(0) # ζ+ hi boundary
        for j in (NG_val+1):(nyp+NG_val)
            violated = false
            local_cos = 1.0
            local_r = 1.0
            for i in (NG_val+1):(nxp+NG_val)
                # Face 1 (interface face: nzp+NG_val+1)
                A1 = Areak_h[i, j, nzp+NG_val+1]
                nx1 = nxk_h[i, j, nzp+NG_val+1]; ny1 = nyk_h[i, j, nzp+NG_val+1]; nz1 = nzk_h[i, j, nzp+NG_val+1]
                # Face 2 (adjacent interior face: nzp+NG_val)
                A2 = Areak_h[i, j, nzp+NG_val]
                nx2 = nxk_h[i, j, nzp+NG_val]; ny2 = nyk_h[i, j, nzp+NG_val]; nz2 = nzk_h[i, j, nzp+NG_val]

                local_cos = nx1*nx2 + ny1*ny2 + nz1*nz2
                local_r = A1 / (A2 + 1e-30)
                if local_cos < cos_threshold || local_r > stretch_threshold || local_r < (1.0 / stretch_threshold)
                    violated = true
                    break
                end
            end
            if violated
                if verbose
                    println("      > Block $bid ζ-hi row j=$j violated smoothness (cos=$(round(local_cos, digits=5)), ratio=$(round(local_r, digits=5))). Boosting phi...")
                end
                for d in 0:(n_protect-1)
                    idx = nzp + NG_val - d
                    boost_val = FTl(_CROSSTYPE_PHI[d+1])
                    if boost_val > phi_k[j, idx]
                        Δϕ = boost_val - phi_k[j, idx]
                        for s in 1:7
                            Lin_k[j, idx, s] += FTl(Δϕ) * ΔLin_k[j, idx, s]
                        end
                        phi_k[j, idx] = boost_val
                    end
                end
            end
        end
    end
end

include("div.jl")
include("mpi.jl")
include("IO.jl")
include("Riemann_Solver.jl")
include("Reconstruct.jl")
include("volume_force.jl")


include("ghost_coords.jl")
include("auto_tune.jl")
include("filter_interface.jl")
include("diag_checkerboard.jl")
include("fringe.jl")

# ── GLM cleaning speed for MHD (updated each time step) ──
# Auto-computed from max fast magnetosonic speed
const _ch_glm_ref = Ref{FT}(one(FT))
ch_glm_current::FT = one(FT)  # will be overwritten in time_step loop

# ── Allocation-free NaN check ──
# Pre-allocated flag buffers (initialized in time_step)
const _nan_flag_gpu = Ref{Any}(nothing)
const _nan_flag_cpu = zeros(Int32, 2)

function _nan_check_kernel!(flag, Q, N)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= N
        @inbounds if !isfinite(Q[idx])
            flag[1] = Int32(1)  # race-safe: all threads write same value
            flag[2] = Int32(idx)
        end
    end
    return
end

function _has_nan(Q)
    if _nan_flag_gpu[] === nothing
        _nan_flag_gpu[] = gpu_zeros(Int32, 2)
    end
    flag_gpu = _nan_flag_gpu[]
    fill!(flag_gpu, Int32(0))
    n = Int32(length(Q))
    @gpu_launch threads=256 blocks=cld(n, Int32(256)) _nan_check_kernel!(flag_gpu, Q, n)
    gpu_sync()
    copyto!(_nan_flag_cpu, flag_gpu)
    return _nan_flag_cpu[1] != Int32(0), Int64(_nan_flag_cpu[2])
end

macro check_nan(array, label, block_id, rank, step)
    return esc(quote
        if debug_nan
            if debug_sync
                gpu_sync()
            end
            _nan_found, _ = _has_nan($array)
            if _nan_found
                # Download array to CPU to reliably find ALL NaN locations
                _cpu_array = Array($array)
                _nan_indices = findall(isnan, _cpu_array)
                
                if isempty(_nan_indices)
                    _msg = "Rank $($rank) Block $($block_id) Step $($step): NaN detected in $($label) by GPU, but CPU findall found none (Async race?).\n"
                else
                    _c_idx = _nan_indices[1] # Take the first NaN found
                    _msg = "Rank $($rank) Block $($block_id) Step $($step): NaN detected in $($label) at index $(_c_idx) (Total NaNs: $(length(_nan_indices)))!\n"
                    
                    # Try to retrieve stretch information if block is in scope
                    _b_ptr = nothing
                    if @isdefined(b) && b isa Block
                        _b_ptr = b
                    elseif @isdefined(block) && block isa Block
                        _b_ptr = block
                    end

                    if _b_ptr !== nothing
                        try
                            _l_str = string($label)
                            _i = _c_idx[1]; _j = _c_idx[2]; _k = _c_idx[3]
                            
                            if occursin("Fx", _l_str)
                                _i += NG - 1; _j += NG; _k += NG
                            elseif occursin("Fy", _l_str)
                                _i += NG; _j += NG - 1; _k += NG
                            elseif occursin("Fz", _l_str)
                                _i += NG; _j += NG; _k += NG - 1
                            end
                            
                            _i = clamp(_i, 1, size(_b_ptr.lin_phi_i, 1))
                            _j = clamp(_j, 1, size(_b_ptr.lin_phi_j, 1))
                            _k = clamp(_k, 1, size(_b_ptr.lin_phi_j, 2))
                            
                            _lin_i = Array(_b_ptr.lin_phi_i)[_i]
                            _lin_j = Array(_b_ptr.lin_phi_j)[_j, _k]
                            _lin_k = Array(_b_ptr.lin_phi_k)[_j, _k]
                            
                            _format_region(val) = val > 0.1 ? "Stretched/Interface (lin_ϕ=$(round(val, digits=3)))" : "Smooth (lin_ϕ=$(round(val, digits=3)))"
                            
                            _msg *= "  > X-dir: $(_format_region(_lin_i))\n"
                            _msg *= "  > Y-dir: $(_format_region(_lin_j))\n"
                            _msg *= "  > Z-dir: $(_format_region(_lin_k))\n"
                            
                            # Boundary proximity check (within NG layers of interblock face)
                            _nj_tot = size(_b_ptr.lin_phi_j, 1)
                            _nk_tot = size(_b_ptr.lin_phi_j, 2)
                            _near_j_lo = _j <= NG + 4
                            _near_j_hi = _j >= _nj_tot - NG - 3
                            _near_k_lo = _k <= NG + 4
                            _near_k_hi = _k >= _nk_tot - NG - 3
                            _proximity = String[]
                            if _near_j_lo; push!(_proximity, "η- (j=$(_j))"); end
                            if _near_j_hi; push!(_proximity, "η+ (j=$(_j))"); end
                            if _near_k_lo; push!(_proximity, "ζ- (k=$(_k))"); end
                            if _near_k_hi; push!(_proximity, "ζ+ (k=$(_k))"); end
                            if isempty(_proximity)
                                _msg *= "  > Boundary proximity: interior\n"
                            else
                                _msg *= "  > Boundary proximity: $(join(_proximity, ", "))\n"
                            end
                        catch e
                            _msg *= "  > (Failed to retrieve diagnostics: $e)\n"
                        end
                    end
                end

                printstyled(_msg, color=:red)
                flush(stdout)
                MPI.Abort(MPI.COMM_WORLD, 1)
            end
        end
    end)
end



struct Connectivity
    src_b::Int
    src_f::Int
    reverse_tan::Bool
    flip_normal::Bool
end

mutable struct Block
    id::Int
    Nx::Int
    Ny::Int
    Nz::Int
    rx::Int
    ry::Int
    rz::Int
    ox::Int  # global x-offset (0-indexed, in real cells)
    oy::Int  # global y-offset
    oz::Int  # global z-offset
    Q::GPUArray{FT, 4}
    U::GPUArray{FT, 4}
    ϕ::GPUArray{FT, 3}
    Areai::GPUArray{FT, 3}
    Areaj::GPUArray{FT, 3}
    Areak::GPUArray{FT, 3}
    nxi::GPUArray{FT, 3}
    nyi::GPUArray{FT, 3}
    nzi::GPUArray{FT, 3}
    nxj::GPUArray{FT, 3}
    nyj::GPUArray{FT, 3}
    nzj::GPUArray{FT, 3}
    nxk::GPUArray{FT, 3}
    nyk::GPUArray{FT, 3}
    nzk::GPUArray{FT, 3}
    Vol::GPUArray{FT, 3}
    x::GPUArray{FT, 3}
    y::GPUArray{FT, 3}
    z::GPUArray{FT, 3}
    LTS_dt::GPUArray{FT, 3}
    Un::GPUArray{FT, 4}
    # MPI Buffers
    sbuf_hx::Array{FT, 4}
    sbuf_dx::GPUArray{FT, 4}
    rbuf_hx::Array{FT, 4}
    rbuf_dx::GPUArray{FT, 4}
    # Second x-direction buffers for non-blocking MPI pipeline
    sbuf_hx2::Array{FT, 4}
    sbuf_dx2::GPUArray{FT, 4}
    rbuf_hx2::Array{FT, 4}
    rbuf_dx2::GPUArray{FT, 4}
    sbuf_hy::Array{FT, 4}
    sbuf_dy::GPUArray{FT, 4}
    rbuf_hy::Array{FT, 4}
    rbuf_dy::GPUArray{FT, 4}
    sbuf_hz::Array{FT, 4}
    sbuf_dz::GPUArray{FT, 4}
    rbuf_hz::Array{FT, 4}
    rbuf_dz::GPUArray{FT, 4}
    # Rotation
    Ωx::GPUArray{FT, 3}
    Ωy::GPUArray{FT, 3}
    Ωz::GPUArray{FT, 3}
    # Grid dimensions for kernels
    nb::Tuple{Int, Int, Int}
    # ─── Implicit LU-SGS buffers (nothing when implicit==false) ───
    dU_rhs::Union{GPUArray{FT, 4}, Nothing}
    ΔU::Union{GPUArray{FT, 4}, Nothing}
    D_inv::Union{GPUArray{FT, 3}, Nothing}
    σ_i::Union{GPUArray{FT, 3}, Nothing}
    σ_j::Union{GPUArray{FT, 3}, Nothing}
    σ_k::Union{GPUArray{FT, 3}, Nothing}
    # ─── BDF2 dual-time stepping (nothing when dual_time==false) ───
    U_nm1::Union{GPUArray{FT, 4}, Nothing}   # U^{n-1} for BDF2
    # ─── GMRES Krylov buffers (nothing when not using GMRES) ───
    V_krylov::Union{GPUArray{FT, 5}, Nothing}  # (Nx_tot, Ny_tot, Nz_tot, Ncons, m+1)
    R_base::Union{GPUArray{FT, 4}, Nothing}    # Base RHS for matvec (nxp, nyp, nzp, Ncons)

    # ─── Spectral warmup: per-direction-layer stencil coefficients ───
    stencil_i::GPUArray{FT, 2}    # (Nx+2NG, 7)  LEFT state
    Δstencil_i::GPUArray{FT, 2}   # (Nx+2NG, 7)
    lin_phi_i::GPUArray{FT, 1}    # (Nx+2NG,)
    stencil_j::GPUArray{FT, 3}    # (Ny+2NG, Nz+2NG, 7)
    Δstencil_j::GPUArray{FT, 3}   # (Ny+2NG, Nz+2NG, 7)
    lin_phi_j::GPUArray{FT, 2}    # (Ny+2NG, Nz+2NG)
    stencil_k::GPUArray{FT, 3}    # (Ny+2NG, Nz+2NG, 7)
    Δstencil_k::GPUArray{FT, 3}   # (Ny+2NG, Nz+2NG, 7)
    lin_phi_k::GPUArray{FT, 2}    # (Ny+2NG, Nz+2NG)
    stencil_R_i::GPUArray{FT, 2}  # (Nx+2NG, 7)  RIGHT state
    Δstencil_R_i::GPUArray{FT, 2} # (Nx+2NG, 7)
    stencil_R_j::GPUArray{FT, 3}  # (Ny+2NG, Nz+2NG, 7)
    Δstencil_R_j::GPUArray{FT, 3} # (Ny+2NG, Nz+2NG, 7)
    stencil_R_k::GPUArray{FT, 3}  # (Ny+2NG, Nz+2NG, 7)
    Δstencil_R_k::GPUArray{FT, 3} # (Ny+2NG, Nz+2NG, 7)
    filter_σ::NTuple{3, FT}       # per-direction filter strength
    # ─── Time-Averaging buffer ───
    Q_avg::Union{GPUArray{FT, 4}, Nothing}
    # ─── Interblock flags ───
    is_interblock::NTuple{6, Bool}
    # ─── Fringe region (differential rotation, nothing when not active) ───
    fringe_lambda::Union{GPUArray{FT, 3}, Nothing}      # (Nx_tot, Ny_tot, Nz_tot)
    U_target_fringe::Union{GPUArray{FT, 4}, Nothing}    # (Nx_tot, Ny_tot, Nz_tot, Ncons)
end

include("implicit.jl")
include("gmres.jl")

function load_block(bid, rx, ry, rz, NG, Ncons, Nprim, Nprocs_block, world_rank,
                    face_bc, connectivity)
    _mesh_base = isdefined(Main, :mesh_dir) ? mesh_dir : "MESH"
    mesh_path = joinpath(_mesh_base, "mesh_b$bid.h5")
    
    f_mesh = h5open(mesh_path, "r")
    Nx_val = Int(read(f_mesh["Nx"]))
    Ny_val = Int(read(f_mesh["Ny"]))
    Nz_val = Int(read(f_mesh["Nz"]))
    
    if rx == 0 && ry == 0 && rz == 0
        println("  > Block $bid: Loading real-only mesh $Nx_val x $Ny_val x $Nz_val...")
    end
    
    # Domain decomposition within current block (non-uniform: first rem ranks get +1 cell)
    base_nx = Nx_val ÷ Nprocs_block[1]; rem_nx = Nx_val % Nprocs_block[1]
    base_ny = Ny_val ÷ Nprocs_block[2]; rem_ny = Ny_val % Nprocs_block[2]
    base_nz = Nz_val ÷ Nprocs_block[3]; rem_nz = Nz_val % Nprocs_block[3]
    nxp = rx < rem_nx ? base_nx + 1 : base_nx
    nyp = ry < rem_ny ? base_ny + 1 : base_ny
    nzp = rz < rem_nz ? base_nz + 1 : base_nz
    # Global offsets (0-indexed)
    ox = min(rx, rem_nx) * (base_nx + 1) + max(0, rx - rem_nx) * base_nx
    oy = min(ry, rem_ny) * (base_ny + 1) + max(0, ry - rem_ny) * base_ny
    oz = min(rz, rem_nz) * (base_nz + 1) + max(0, rz - rem_nz) * base_nz
    
    # Read real-only coordinates (Nx+1, Ny+1, Nz+1)
    coords_real = read(f_mesh["coords"])  # (3, Nx+1, Ny+1, Nz+1)
    close(f_mesh)
    
    # Extract real node coords for this rank's subdomain
    lx = ox + 1; hx = ox + nxp + 1  # +1 for vertices
    ly = oy + 1; hy = oy + nyp + 1
    lz = oz + 1; hz = oz + nzp + 1
    x_real = Float64.(coords_real[1, lx:hx, ly:hy, lz:hz])
    y_real = Float64.(coords_real[2, lx:hx, ly:hy, lz:hz])
    z_real = Float64.(coords_real[3, lx:hx, ly:hy, lz:hz])
    
    if rx == 0 && ry == 0 && rz == 0
        println("  > Block $bid: Expanding ghost coordinates at runtime...")
    end
    
    # Expand to full padded coordinates with ghost cells
    x_full, y_full, z_full = expand_coords_with_ghost(
        x_real, y_real, z_real, nxp, nyp, nzp, NG, face_bc, bid, connectivity;
        xi_offset=ox)
    
    # Compute (or load cached) metrics from expanded coordinates
    Areai_h, nxi_h, nyi_h, nzi_h, Areaj_h, nxj_h, nyj_h, nzj_h,
    Areak_h, nxk_h, nyk_h, nzk_h, Vol_h = load_or_compute_metrics(
        bid, rx, ry, rz, x_full, y_full, z_full, nxp, nyp, nzp, NG; cache_metrics=cache_metrics)
    
    # Full array size for GPUArrays (with ghost cells)
    Nx_tot, Ny_tot, Nz_tot = nxp + 2*NG, nyp + 2*NG, nzp + 2*NG
    
    # Copy metrics to GPU
    Areai = GPUArray(Areai_h); nxi = GPUArray(nxi_h); nyi = GPUArray(nyi_h); nzi = GPUArray(nzi_h)
    Areaj = GPUArray(Areaj_h); nxj = GPUArray(nxj_h); nyj = GPUArray(nyj_h); nzj = GPUArray(nzj_h)
    Areak = GPUArray(Areak_h); nxk = GPUArray(nxk_h); nyk = GPUArray(nyk_h); nzk = GPUArray(nzk_h)
    Vol   = GPUArray(Vol_h)

    # Diagnostic check for metrics
    if any(isnan, Vol_h) || any(x -> x <= 1.0e-18, Vol_h)
        println("Rank $world_rank, Block $bid: CRITICAL! NaNs or non-positive volumes detected. min(Vol) = $(minimum(Vol_h))")
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
    if any(isnan, Areai_h) || any(isnan, Areaj_h) || any(isnan, Areak_h)
        println("Rank $world_rank, Block $bid: CRITICAL! NaNs detected in metrics Area.")
        MPI.Abort(MPI.COMM_WORLD, 1)
    end

    # Allocate primary and conservative variables
    Q = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Nprim)
    U = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Ncons)
    ϕ = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot)
    
    # Upload expanded coordinates (nodes, including ghost) to GPU
    x = GPUArray(x_full)
    y = GPUArray(y_full)
    z = GPUArray(z_full)
    
    if rx == 0 && ry == 0 && rz == 0
        println("  > Block $bid: Metrics computed and coordinates loaded. Initializing...")
    end
    
    initialize(Q, x, y, z, rx, ry, Nprocs_block, nxp, nyp, nzp)
    if rx == 0 && ry == 0 && rz == 0 && bid == 0 && world_rank == 0
        Q_cpu = Array(Q[NG+1, NG+1, NG+1, 1:6])
        println(">>> Debug init: rho=$(Q_cpu[1]), u=$(Q_cpu[2]), v=$(Q_cpu[3]), w=$(Q_cpu[4]), p=$(Q_cpu[5]), T=$(Q_cpu[6])")
        flush(stdout)
    end
    @check_nan(Q, "Q after initialize", bid, world_rank, 0)
    nb = (cld(Nx_tot, nthreads[1]), cld(Ny_tot, nthreads[2]), cld(Nz_tot, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb prim2c(U, Q, nxp, nyp, nzp)
    @check_nan(U, "U after prim2c (initial)", bid, world_rank, 0)



    LTS_dt = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot)
    # Un used for RK time stepping (Ncons components) AND edge ghost backup (Nprim components)
    # Allocate with Nprim (→Ncons) so it can serve both purposes
    Un = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Nprim)

    if rx == 0 && ry == 0 && rz == 0
        println("  > Block $bid: Initialization complete.")
    end

    # Allocate MPI Buffers
    sbuf_hx = zeros(FT, NG, nyp+2*NG, nzp+2*NG, Nprim)
    rbuf_hx = zeros(FT, NG, nyp+2*NG, nzp+2*NG, Nprim)
    sbuf_dx = GPUArray(sbuf_hx); rbuf_dx = GPUArray(rbuf_hx)
    # Second set for non-blocking MPI pipeline (x- direction)
    sbuf_hx2 = zeros(FT, NG, nyp+2*NG, nzp+2*NG, Nprim)
    rbuf_hx2 = zeros(FT, NG, nyp+2*NG, nzp+2*NG, Nprim)
    sbuf_dx2 = GPUArray(sbuf_hx2); rbuf_dx2 = GPUArray(rbuf_hx2)

    sbuf_hy = zeros(FT, nxp+2*NG, NG, nzp+2*NG, Nprim)
    rbuf_hy = zeros(FT, nxp+2*NG, NG, nzp+2*NG, Nprim)
    sbuf_dy = GPUArray(sbuf_hy); rbuf_dy = GPUArray(rbuf_hy)

    sbuf_hz = zeros(FT, nxp+2*NG, nyp+2*NG, NG, Nprim)
    rbuf_hz = zeros(FT, nxp+2*NG, nyp+2*NG, NG, Nprim)
    sbuf_dz = GPUArray(sbuf_hz); rbuf_dz = GPUArray(rbuf_hz)

    # Allocation-Free Ω fields
    Ωx = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot)
    Ωy = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot)
    Ωz = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot)
    
    nb = (cld(Nx_tot, nthreads[1]), cld(Ny_tot, nthreads[2]), cld(Nz_tot, nthreads[3]))
    if isdefined(@__MODULE__, :diffrot_enabled) && diffrot_enabled
        # Non-uniform rotation: Ω(x) linearly varies in physical zone, smooth return in fringe
        @gpu_launch threads=nthreads blocks=nb Assign_rotation_var_nonuniform(
            Ωx, Ωy, Ωz, x, y, z, nxp, nyp, nzp,
            FT(Omega_x_min), FT(Omega_x_max), FT(L_phys), FT(L_total), FT(fringe_rise_fraction))
    else
        x_rot_start_val = isdefined(Main, :x_rot_start) ? FT(Main.x_rot_start) : zero(FT)
        x_rot_end_val   = isdefined(Main, :x_rot_end)   ? FT(Main.x_rot_end)   : FT(1.0e10)
        @gpu_launch threads=nthreads blocks=nb Assign_rotation_var(
            Ωx, Ωy, Ωz, x, y, z, nxp, nyp, nzp, x_rot_start_val, x_rot_end_val)
    end

    # ─── Fringe region initialization (differential rotation only) ───
    if isdefined(@__MODULE__, :diffrot_enabled) && diffrot_enabled
        fringe_lambda_arr = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot)
        @gpu_launch threads=nthreads blocks=nb compute_fringe_lambda_kernel!(
            fringe_lambda_arr, x, nxp, nyp, nzp,
            FT(L_phys), FT(L_total), FT(fringe_lambda_max), FT(fringe_rise_fraction))
        # U_target: load precursor mean profile or initialize to zero
        U_target_cpu = zeros(FT, Nx_tot, Ny_tot, Nz_tot, Ncons)
        if isdefined(@__MODULE__, :precursor_mean_path) && isfile(precursor_mean_path)
            # Load (Ny_block, Nz_block, Ncons) cross-section and tile along x
            h5open(precursor_mean_path, "r") do fid
                grp_name = "b$(bid)"
                if haskey(fid, grp_name)
                    U_cross = FT.(read(fid["$(grp_name)/U_mean"]))  # (Ny_block, Nz_block, Ncons)
                    # Fill interior cells: tile the cross-section along x
                    for n in 1:Ncons, k in 1:nzp, j in 1:nyp
                        jg = j + NG; kg = k + NG
                        # Map local (j,k) to global block (j,k) for precursor lookup
                        j_glob = j + oy  # oy = offset in block
                        k_glob = k + oz
                        if j_glob >= 1 && j_glob <= size(U_cross, 1) &&
                           k_glob >= 1 && k_glob <= size(U_cross, 2)
                            val = U_cross[j_glob, k_glob, n]
                            for i in 1:nxp
                                U_target_cpu[i + NG, jg, kg, n] = val
                            end
                        end
                    end
                    if world_rank == 0
                        println("  Fringe U_target loaded from: $(precursor_mean_path)")
                    end
                else
                    if world_rank == 0
                        @warn "Block $grp_name not found in $(precursor_mean_path), U_target = 0"
                    end
                end
            end
        else
            if world_rank == 0
                @warn "Precursor file not found: $(precursor_mean_path), U_target = 0"
            end
        end
        U_target_arr = GPUArray(U_target_cpu)
        if world_rank == 0
            println("  Fringe region initialized: λ_max=$(fringe_lambda_max), L_phys=$(L_phys), L_fringe=$(L_total - L_phys)")
        end
    end

    # ─── Allocate implicit LU-SGS buffers (only when implicit==true) ───
    if implicit
        imp_dU_rhs = gpu_zeros(FT, nxp, nyp, nzp, Ncons)
        imp_ΔU     = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Ncons)
        imp_D_inv  = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot)
        imp_σ_i    = gpu_zeros(FT, nxp+1, nyp, nzp)
        imp_σ_j    = gpu_zeros(FT, nxp, nyp+1, nzp)
        imp_σ_k    = gpu_zeros(FT, nxp, nyp, nzp+1)
        # BDF2 dual-time stepping: store U^{n-1}
        imp_U_nm1 = dual_time ? gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Ncons) : nothing
        # GMRES Krylov buffers
        _use_gmres = isdefined(@__MODULE__, :implicit_solver) ? (implicit_solver == :gmres) : false
        if _use_gmres
            _gmres_m = isdefined(@__MODULE__, :gmres_m) ? gmres_m : 10
            imp_V_krylov = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Ncons, _gmres_m + 1)
            imp_R_base = gpu_zeros(FT, nxp, nyp, nzp, Ncons)
        else
            imp_V_krylov = nothing
            imp_R_base = nothing
        end
        if rx == 0 && ry == 0 && rz == 0
            imp_mem_mb = (prod((nxp, nyp, nzp, Ncons)) + prod((Nx_tot, Ny_tot, Nz_tot, Ncons)) +
                          prod((Nx_tot, Ny_tot, Nz_tot)) +
                          (nxp+1)*nyp*nzp + nxp*(nyp+1)*nzp + nxp*nyp*(nzp+1)) * 4 / 1024^2
            if dual_time
                imp_mem_mb += prod((Nx_tot, Ny_tot, Nz_tot, Ncons)) * 4 / 1024^2
            end
            if _use_gmres
                gmres_mem = prod((Nx_tot, Ny_tot, Nz_tot, Ncons, _gmres_m + 1)) * 4 / 1024^2
                gmres_mem += prod((nxp, nyp, nzp, Ncons)) * 4 / 1024^2
                imp_mem_mb += gmres_mem
                println("  > Block $bid: Implicit buffers allocated (~$(round(imp_mem_mb, digits=1)) MB) [BDF2+GMRES($(_gmres_m))]")
            else
                println("  > Block $bid: Implicit buffers allocated (~$(round(imp_mem_mb, digits=1)) MB)$(dual_time ? " [BDF2]" : "")")
            end
        end
    else
        imp_dU_rhs = nothing; imp_ΔU = nothing; imp_D_inv = nothing
        imp_σ_i = nothing; imp_σ_j = nothing; imp_σ_k = nothing
        imp_U_nm1 = nothing
        imp_V_krylov = nothing; imp_R_base = nothing
    end

    # ─── Initialize stencil arrays with compile-time constants (default) ───
    Ni = nxp + 2*NG; Nj = nyp + 2*NG; Nk = nzp + 2*NG
    _Lin_h = repeat(reshape(FT.(Linear), 1, 7), Ni, 1)
    _ΔLin_h = repeat(reshape(FT.(ΔLinear), 1, 7), Ni, 1)
    _phi_h = fill(FT(Linear_ϕ), Ni)
    stencil_i = GPUArray(_Lin_h); Δstencil_i = GPUArray(_ΔLin_h); lin_phi_i = GPUArray(_phi_h)
    stencil_R_i = GPUArray(_Lin_h); Δstencil_R_i = GPUArray(_ΔLin_h)
    _Lin_j = repeat(reshape(FT.(Linear), 1, 1, 7), Nj, Nk, 1)
    _ΔLin_j = repeat(reshape(FT.(ΔLinear), 1, 1, 7), Nj, Nk, 1)
    _phi_j = fill(FT(Linear_ϕ), Nj, Nk)
    _Lin_k = repeat(reshape(FT.(Linear), 1, 1, 7), Nj, Nk, 1)
    _ΔLin_k = repeat(reshape(FT.(ΔLinear), 1, 1, 7), Nj, Nk, 1)
    _phi_k = fill(FT(Linear_ϕ), Nj, Nk)
    # Interface taper: CD2→CD4→CD6 with upwind bias near interblock faces
    apply_interface_taper!(_Lin_j, _ΔLin_j, _phi_j, _Lin_k, _ΔLin_k, _phi_k,
                           face_bc, bid, nyp, nzp, NG)
    apply_geometric_smoothness_protection!(
        _Lin_j, _ΔLin_j, _phi_j,
        _Lin_k, _ΔLin_k, _phi_k,
        Areaj_h, nxj_h, nyj_h, nzj_h,
        Areak_h, nxk_h, nyk_h, nzk_h,
        face_bc, bid, nxp, nyp, nzp, NG;
        verbose=true
    )
    stencil_j = GPUArray(_Lin_j); Δstencil_j = GPUArray(_ΔLin_j); lin_phi_j = GPUArray(_phi_j)
    stencil_k = GPUArray(_Lin_k); Δstencil_k = GPUArray(_ΔLin_k); lin_phi_k = GPUArray(_phi_k)
    stencil_R_j = GPUArray(_Lin_j); Δstencil_R_j = GPUArray(_ΔLin_j)
    stencil_R_k = GPUArray(_Lin_k); Δstencil_R_k = GPUArray(_ΔLin_k)
    filter_σ_init = (FT(filtering_s0), FT(filtering_s0), FT(filtering_s0))
    is_inter = (
        get(face_bc, (bid, 1), -1) == 0,
        get(face_bc, (bid, 2), -1) == 0,
        get(face_bc, (bid, 3), -1) == 0,
        get(face_bc, (bid, 4), -1) == 0,
        get(face_bc, (bid, 5), -1) == 0,
        get(face_bc, (bid, 6), -1) == 0
    )

    return Block(bid, nxp, nyp, nzp, rx, ry, rz, ox, oy, oz, Q, U, ϕ, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, Vol, x, y, z, LTS_dt, Un,
                 sbuf_hx, sbuf_dx, rbuf_hx, rbuf_dx, sbuf_hx2, sbuf_dx2, rbuf_hx2, rbuf_dx2,
                 sbuf_hy, sbuf_dy, rbuf_hy, rbuf_dy, sbuf_hz, sbuf_dz, rbuf_hz, rbuf_dz,
                 Ωx, Ωy, Ωz, nb,
                 imp_dU_rhs, imp_ΔU, imp_D_inv, imp_σ_i, imp_σ_j, imp_σ_k, imp_U_nm1,
                 imp_V_krylov, imp_R_base,
                 stencil_i, Δstencil_i, lin_phi_i,
                 stencil_j, Δstencil_j, lin_phi_j,
                 stencil_k, Δstencil_k, lin_phi_k,
                 stencil_R_i, Δstencil_R_i,
                 stencil_R_j, Δstencil_R_j,
                 stencil_R_k, Δstencil_R_k,
                 filter_σ_init,
                 isdefined(Main, :average) && average ? gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Nprim) : nothing,
                 is_inter,
                 # Fringe region fields
                 isdefined(@__MODULE__, :diffrot_enabled) && diffrot_enabled ? fringe_lambda_arr : nothing,
                 isdefined(@__MODULE__, :diffrot_enabled) && diffrot_enabled ? U_target_arr : nothing)
end

function load_multiblock_connectivity(path)
    if !isfile(path)
        return 1, Dict{Tuple{Int, Int}, Connectivity}(), Dict{Tuple{Int, Int}, Int}(), 
               Dict{Tuple{Int, Int}, NTuple{N_BC_PARAMS, FT}}(), Int[], Int[], Int[]
    end
    fid = h5open(path, "r")
    Nblocks = Int(read(fid["Nblocks"]))
    connect_raw = read(fid["connectivity"])
    
    # Load reverse_tan if available (new format), else default to false
    rev_raw = haskey(fid, "reverse_tan") ? read(fid["reverse_tan"]) : zeros(Int64, size(connect_raw, 1))
    
    # Load flip_normal if available, else default to false
    flip_raw = haskey(fid, "flip_normal") ? read(fid["flip_normal"]) : zeros(Int64, size(connect_raw, 1))
    
    # Load face_bc if available: (Nblocks, 6) array →BC type IDs from bc_types.jl
    face_bc_raw = haskey(fid, "face_bc") ? read(fid["face_bc"]) : nothing

    # Load bc_params if available: (Nblocks, 6, N_BC_PARAMS) →per-face BC parameters
    bc_params_raw = haskey(fid, "bc_params") ? read(fid["bc_params"]) : nothing
    
    connectivity = Dict{Tuple{Int, Int}, Connectivity}()
    for i in 1:size(connect_raw, 1)
        b1, f1, b2, f2 = Int(connect_raw[i, 1]), Int(connect_raw[i, 2]), Int(connect_raw[i, 3]), Int(connect_raw[i, 4])
        rev = rev_raw[i] != 0
        flip = flip_raw[i] != 0
        connectivity[(b1, f1)] = Connectivity(b2, f2, rev, flip)
    end
    
    # Build face_bc dict: (block_id, face_id) -> bc_type
    face_bc = Dict{Tuple{Int, Int}, Int}()
    if face_bc_raw !== nothing
        for bid in 1:Nblocks, fid_idx in 1:6
            face_bc[(bid-1, fid_idx)] = Int(face_bc_raw[bid, fid_idx])
        end
    end
    
    # Build bc_params dict: (block_id, face_id) -> NTuple{N_BC_PARAMS, FT}
    bc_params = Dict{Tuple{Int, Int}, NTuple{N_BC_PARAMS, FT}}()
    if bc_params_raw !== nothing
        n_stored = size(bc_params_raw, 3)  # actual params in file (may be < N_BC_PARAMS)
        for bid in 1:Nblocks, fid_idx in 1:6
            params = ntuple(p -> p <= n_stored ? FT(bc_params_raw[bid, fid_idx, p]) : zero(FT), Val(N_BC_PARAMS))
            bc_params[(bid-1, fid_idx)] = params
        end
    end
    
    # Load global dimensions
    Nx_b = haskey(fid, "Nx_b") ? read(fid["Nx_b"]) : Int[]
    Ny_b = haskey(fid, "Ny_b") ? read(fid["Ny_b"]) : Int[]
    Nz_b = haskey(fid, "Nz_b") ? read(fid["Nz_b"]) : Int[]
    
    close(fid)
    return Nblocks, connectivity, face_bc, bc_params, Nx_b, Ny_b, Nz_b
end


function blockAdvance(block::Block, dt, ϕ, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, world_rank, tt,
                      threads_recon_i, threads_recon_j, threads_recon_k,
                      threads_visc_i, threads_visc_j, threads_visc_k)
    Q = block.Q
    U = block.U
    Areai, nxi, nyi, nzi = block.Areai, block.nxi, block.nyi, block.nzi
    Areaj, nxj, nyj, nzj = block.Areaj, block.nxj, block.nyj, block.nzj
    Areak, nxk, nyk, nzk = block.Areak, block.nxk, block.nyk, block.nzk
    Vol = block.Vol
    nxp, nyp, nzp = block.Nx, block.Ny, block.Nz

    # Per-direction grid sizes using per-direction auto-tuned thread configs
    nb_recon_i = (Int32(cld(nxp+2*NG, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
    nb_recon_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+2*NG, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
    nb_recon_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+2*NG, threads_recon_k[3])))
    nb_visc_i  = (Int32(cld(nxp+2*NG, threads_visc_i[1])),  Int32(cld(nyp+2*NG, threads_visc_i[2])),  Int32(cld(nzp+2*NG, threads_visc_i[3])))
    nb_visc_j  = (Int32(cld(nxp+2*NG, threads_visc_j[1])),  Int32(cld(nyp+2*NG, threads_visc_j[2])),  Int32(cld(nzp+2*NG, threads_visc_j[3])))
    nb_visc_k  = (Int32(cld(nxp+2*NG, threads_visc_k[1])),  Int32(cld(nyp+2*NG, threads_visc_k[2])),  Int32(cld(nzp+2*NG, threads_visc_k[3])))

    si = block.stencil_i; Δsi = block.Δstencil_i; lpi = block.lin_phi_i
    sj = block.stencil_j; Δsj = block.Δstencil_j; lpj = block.lin_phi_j
    sk = block.stencil_k; Δsk = block.Δstencil_k; lpk = block.lin_phi_k
    sRi = block.stencil_R_i; ΔsRi = block.Δstencil_R_i
    sRj = block.stencil_R_j; ΔsRj = block.Δstencil_R_j
    sRk = block.stencil_R_k; ΔsRk = block.Δstencil_R_k
    if eigen_reconstruction
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i Eigen_reconstruct_i(Q, U, ϕ, Areai, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(0))
        @check_nan(Fx, "Fx after Eigen_reconstruct_i", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j Eigen_reconstruct_j(Q, U, ϕ, Areaj, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(0))
        @check_nan(Fy, "Fy after Eigen_reconstruct_j", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k Eigen_reconstruct_k(Q, U, ϕ, Areak, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(0))
        @check_nan(Fz, "Fz after Eigen_reconstruct_k", block.id, world_rank, tt)
    else
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i Conser_reconstruct_i(Q, U, ϕ, Areai, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(0))
        @check_nan(Fx, "Fx after Conser_reconstruct_i", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j Conser_reconstruct_j(Q, U, ϕ, Areaj, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(0))
        @check_nan(Fy, "Fy after Conser_reconstruct_j", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k Conser_reconstruct_k(Q, U, ϕ, Areak, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(0))
        @check_nan(Fz, "Fz after Conser_reconstruct_k", block.id, world_rank, tt)
    end

    if viscous || (equation_type == :MHD && resistive)
        # Edge ghost cells are already filled by two-pass exchange in sync_blocks!
        @gpu_launch threads=threads_visc_i blocks=nb_visc_i viscous_flux_i(Q, Fv_x, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, Vol, nxp, nyp, nzp, block.is_interblock)
        @check_nan(Fv_x, "Fv_x after viscous_flux_i", block.id, world_rank, tt)

        @gpu_launch threads=threads_visc_j blocks=nb_visc_j viscous_flux_j(Q, Fv_y, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, Vol, nxp, nyp, nzp, block.is_interblock)
        @check_nan(Fv_y, "Fv_y after viscous_flux_j", block.id, world_rank, tt)

        @gpu_launch threads=threads_visc_k blocks=nb_visc_k viscous_flux_k(Q, Fv_z, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, Vol, nxp, nyp, nzp, block.is_interblock)
        @check_nan(Fv_z, "Fv_z after viscous_flux_k", block.id, world_rank, tt)
    end
end

# ── Phase 2: Interior-only blockAdvance on compute_stream ──
# Launches reconstruction with mode=1 (interior only, no ghost dependency)
# on a separate HIP stream. These kernels execute during MPI communication.
function blockAdvance_interior(block::Block, dt, ϕ, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z,
                               threads_recon_i, threads_recon_j, threads_recon_k,
                               threads_visc_i, threads_visc_j, threads_visc_k, stream)
    Q = block.Q; U = block.U
    Areai, nxi, nyi, nzi = block.Areai, block.nxi, block.nyi, block.nzi
    Areaj, nxj, nyj, nzj = block.Areaj, block.nxj, block.nyj, block.nzj
    Areak, nxk, nyk, nzk = block.Areak, block.nxk, block.nyk, block.nzk
    nxp, nyp, nzp = block.Nx, block.Ny, block.Nz
    nb_recon_i = (Int32(cld(nxp+2*NG, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
    nb_recon_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+2*NG, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
    nb_recon_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+2*NG, threads_recon_k[3])))
    nb_visc_i  = (Int32(cld(nxp+2*NG, threads_visc_i[1])),  Int32(cld(nyp+2*NG, threads_visc_i[2])),  Int32(cld(nzp+2*NG, threads_visc_i[3])))
    nb_visc_j  = (Int32(cld(nxp+2*NG, threads_visc_j[1])),  Int32(cld(nyp+2*NG, threads_visc_j[2])),  Int32(cld(nzp+2*NG, threads_visc_j[3])))
    nb_visc_k  = (Int32(cld(nxp+2*NG, threads_visc_k[1])),  Int32(cld(nyp+2*NG, threads_visc_k[2])),  Int32(cld(nzp+2*NG, threads_visc_k[3])))

    si = block.stencil_i; Δsi = block.Δstencil_i; lpi = block.lin_phi_i
    sj = block.stencil_j; Δsj = block.Δstencil_j; lpj = block.lin_phi_j
    sk = block.stencil_k; Δsk = block.Δstencil_k; lpk = block.lin_phi_k
    sRi = block.stencil_R_i; ΔsRi = block.Δstencil_R_i
    sRj = block.stencil_R_j; ΔsRj = block.Δstencil_R_j
    sRk = block.stencil_R_k; ΔsRk = block.Δstencil_R_k
    if eigen_reconstruction
        @gpu_launch_stream stream threads=threads_recon_i blocks=nb_recon_i Eigen_reconstruct_i(Q, U, ϕ, Areai, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(1))
        @gpu_launch_stream stream threads=threads_recon_j blocks=nb_recon_j Eigen_reconstruct_j(Q, U, ϕ, Areaj, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(1))
        @gpu_launch_stream stream threads=threads_recon_k blocks=nb_recon_k Eigen_reconstruct_k(Q, U, ϕ, Areak, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(1))
    else
        @gpu_launch_stream stream threads=threads_recon_i blocks=nb_recon_i Conser_reconstruct_i(Q, U, ϕ, Areai, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(1))
        @gpu_launch_stream stream threads=threads_recon_j blocks=nb_recon_j Conser_reconstruct_j(Q, U, ϕ, Areaj, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(1))
        @gpu_launch_stream stream threads=threads_recon_k blocks=nb_recon_k Conser_reconstruct_k(Q, U, ϕ, Areak, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(1))
    end
    # Viscous flux interior (viscous stencil >= 2 cells, fully covered by mode=1 range)
    if viscous || (equation_type == :MHD && resistive)
        @gpu_launch_stream stream threads=threads_visc_i blocks=nb_visc_i viscous_flux_i(Q, Fv_x, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, block.Vol, nxp, nyp, nzp, block.is_interblock)
        @gpu_launch_stream stream threads=threads_visc_j blocks=nb_visc_j viscous_flux_j(Q, Fv_y, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, block.Vol, nxp, nyp, nzp, block.is_interblock)
        @gpu_launch_stream stream threads=threads_visc_k blocks=nb_visc_k viscous_flux_k(Q, Fv_z, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, block.Vol, nxp, nyp, nzp, block.is_interblock)
    end
end

# ── Phase 2: Boundary-only blockAdvance on default stream ──
# Launches reconstruction with mode=2 (boundary only, needs ghost cells)
# Called AFTER sync_blocks! ensures ghost cells are valid.
function blockAdvance_boundary(block::Block, dt, ϕ, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z,
                               world_rank, tt,
                               threads_recon_i, threads_recon_j, threads_recon_k)
    Q = block.Q; U = block.U
    Areai, nxi, nyi, nzi = block.Areai, block.nxi, block.nyi, block.nzi
    Areaj, nxj, nyj, nzj = block.Areaj, block.nxj, block.nyj, block.nzj
    Areak, nxk, nyk, nzk = block.Areak, block.nxk, block.nyk, block.nzk
    nxp, nyp, nzp = block.Nx, block.Ny, block.Nz
    nb_recon_i = (Int32(cld(nxp+2*NG, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
    nb_recon_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+2*NG, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
    nb_recon_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+2*NG, threads_recon_k[3])))

    si = block.stencil_i; Δsi = block.Δstencil_i; lpi = block.lin_phi_i
    sj = block.stencil_j; Δsj = block.Δstencil_j; lpj = block.lin_phi_j
    sk = block.stencil_k; Δsk = block.Δstencil_k; lpk = block.lin_phi_k
    sRi = block.stencil_R_i; ΔsRi = block.Δstencil_R_i
    sRj = block.stencil_R_j; ΔsRj = block.Δstencil_R_j
    sRk = block.stencil_R_k; ΔsRk = block.Δstencil_R_k
    if eigen_reconstruction
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i Eigen_reconstruct_i(Q, U, ϕ, Areai, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(2))
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j Eigen_reconstruct_j(Q, U, ϕ, Areaj, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(2))
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k Eigen_reconstruct_k(Q, U, ϕ, Areak, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(2))
    else
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i Conser_reconstruct_i(Q, U, ϕ, Areai, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(2))
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j Conser_reconstruct_j(Q, U, ϕ, Areaj, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(2))
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k Conser_reconstruct_k(Q, U, ϕ, Areak, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(2))
    end
    # Note: viscous_flux runs on ALL cells here (no mode param) since interior
    # viscous was already computed on compute_stream and is now complete
    # (copyto! in sync_blocks forced device sync). Boundary viscous overwrites
    # boundary faces; interior faces remain from the compute_stream pass.
end


function time_step(world_rank, comm_cart, Block_Nprocs)
    # Load multi-block metadata
    if world_rank == 0
        println(">>> Loading multi-block connectivity...")
    end
    # connectivity_file is defined in the run config (e.g., run_pipe.jl, run_cavity.jl)
    _mesh_base_conn = isdefined(Main, :mesh_dir) ? mesh_dir : "MESH"
    conn_path = isdefined(Main, :connectivity_file) ? connectivity_file : joinpath(_mesh_base_conn, "block_connectivity.h5")
    Nblocks, connectivity, face_bc, bc_params, Nx_b, Ny_b, Nz_b = load_multiblock_connectivity(conn_path)
    
    # Override physical walls to slip wall for inviscid validation test cases on pipe meshes
    if (isdefined(Main, :test_case) && (Main.test_case == "BrioWu" || Main.test_case == "OrszagTang" || Main.test_case == "TG5_debug" || Main.test_case == "MagneticDecay")) ||
       (isdefined(Main, :slip_wall_override) && Main.slip_wall_override)
        for (k, v) in face_bc
            if v == Int32(BC_ISOTHERMAL_WALL) || v == Int32(BC_ADIABATIC_WALL)
                face_bc[k] = Int32(BC_SLIP_WALL)
            end
        end
    end

    # Override NSCBC outflow to zero-gradient outflow for inviscid validation runs to ensure stability
    if isdefined(Main, :viscous) && !Main.viscous
        for (k, v) in face_bc
            if v == Int32(BC_NSCBC_OUTFLOW)
                face_bc[k] = Int32(BC_ZERO_GRADIENT)
            end
        end
    end

    # Override transition inflow to wave inflow for wave verification test cases
    if isdefined(Main, :wave_inflow_override) && Main.wave_inflow_override
        for (k, v) in face_bc
            if v == Int32(BC_TRANSITION_INFLOW)
                face_bc[k] = Int32(BC_WAVE_INFLOW)
                w_amp   = isdefined(Main, :wave_amp)   ? FT(Main.wave_amp)   : FT(0.001)
                w_omega = isdefined(Main, :wave_omega) ? FT(Main.wave_omega) : FT(2.0)
                w_m     = isdefined(Main, :wave_m)     ? FT(Main.wave_m)     : FT(0.0)
                
                p_list = Vector{FT}(undef, N_BC_PARAMS)
                fill!(p_list, zero(FT))
                p_list[1] = w_amp     # BCP_WAVE_AMP
                p_list[2] = w_omega   # BCP_WAVE_OMEGA
                p_list[3] = w_m       # BCP_WAVE_M
                bc_params[k] = ntuple(i -> p_list[i], Val(N_BC_PARAMS))
            end
        end
    end
    
    # Print BC configuration
    if world_rank == 0
        face_names = ["ξ-", "ξ+", "η-", "η+", "ζ-", "ζ+"]
        for bid in 0:Nblocks-1
            bc_strs = String[]
            for fid in 1:6
                bc_id = get(face_bc, (bid, fid), 0)
                bc_name = get(BC_ID_TO_NAME, Int32(bc_id), "unknown")
                push!(bc_strs, "$(face_names[fid])=$(bc_name)")
            end
            println("  Block $bid: ", join(bc_strs, "  "))
        end
    end
    # ══════════════════════════════════════════════════════════════
    #  Unified block loading: supports both multi-block-per-rank
    #  (N_ranks < N_blocks) and sub-domain splitting (N_ranks >= N_blocks)
    # ══════════════════════════════════════════════════════════════
    if world_rank == 0
        println(">>> Loading block data partition for each rank...")
    end

    rank_offsets = zeros(Int, length(Block_Nprocs) + 1)
    for i in 1:length(Block_Nprocs)
        rank_offsets[i+1] = rank_offsets[i] + prod(Block_Nprocs[i])
    end
    total_ranks_needed = rank_offsets[end]
    world_size = MPI.Comm_size(MPI.COMM_WORLD)

    # Detect multi-block-per-rank mode: all partitions are (1,1,1)
    # and N_ranks < N_blocks (signaled by Block_to_rank mapping)
    multi_block_mode = (world_size < total_ranks_needed)

    blocks = Dict{Int, Block}()
    block_comms = Dict{Int, MPI.Comm}()  # per-block communicator
    rankx = 0; ranky = 0; rankz = 0

    if multi_block_mode
        # ── Multi-block-per-rank: each block is whole (1,1,1) ──
        if world_rank == 0
            println("  > Multi-block-per-rank mode: $(length(Block_Nprocs)) blocks on $world_size ranks")
        end
        for bid in 0:(length(Block_Nprocs)-1)
            if Block_to_rank[bid + 1] == world_rank
                Nprocs_block = Block_Nprocs[bid + 1]  # always (1,1,1)
                blocks[bid] = load_block(bid, 0, 0, 0, NG, Ncons, Nprim, Nprocs_block, world_rank, face_bc, connectivity)
                # No sub-domain splitting →use COMM_SELF for intra-block exchange
                block_comms[bid] = MPI.Cart_create(MPI.COMM_SELF, [1,1,1]; periodic=collect(Iperiodic))
            end
        end
        # Set rank_offsets so copy_ghost_face! maps block →owning rank correctly
        # rank_offsets[i] = rank that owns block (i-1), so:
        #   my_local_rank = world_rank - rank_offsets[bid+1] = 0  (correct)
        #   src_rank_global = rank_offsets[src_bid+1] + 0 = Block_to_rank[src_bid+1]  (correct)
        for i in 1:length(Block_Nprocs)
            rank_offsets[i] = Block_to_rank[i]
        end
        rank_offsets[length(Block_Nprocs) + 1] = Block_to_rank[end] + 1
        _rank_offsets_setup = copy(rank_offsets)
    else
        # ── Standard mode: each rank owns one block sub-domain ──
        my_block_id = -1
        for i in 1:length(Block_Nprocs)
            if world_rank >= rank_offsets[i] && world_rank < rank_offsets[i+1]
                my_block_id = i - 1
                break
            end
        end

        Nprocs_my_block = Block_Nprocs[my_block_id + 1]
        local_rank = world_rank - rank_offsets[my_block_id + 1]
        rankx = local_rank ÷ (Nprocs_my_block[2]*Nprocs_my_block[3])
        ranky = (local_rank ÷ Nprocs_my_block[3]) % Nprocs_my_block[2]
        rankz = local_rank % Nprocs_my_block[3]

        blocks[my_block_id] = load_block(my_block_id, rankx, ranky, rankz, NG, Ncons, Nprim, Nprocs_my_block, world_rank, face_bc, connectivity)

        block_comm = MPI.Comm_split(MPI.COMM_WORLD, my_block_id, local_rank)
        block_comms[my_block_id] = MPI.Cart_create(block_comm, collect(Nprocs_my_block); periodic=collect(Iperiodic))

        _rank_offsets_setup = copy(rank_offsets)
    end

    # Global dimensions for flux buffers (max across all local blocks)
    max_nxp = maximum(b.Nx for (_, b) in blocks)
    max_nyp = maximum(b.Ny for (_, b) in blocks)
    max_nzp = maximum(b.Nz for (_, b) in blocks)

    shared_Fx  = gpu_zeros(FT, max_nxp+1, max_nyp, max_nzp, Ncons)
    shared_Fy  = gpu_zeros(FT, max_nxp, max_nyp+1, max_nzp, Ncons)
    shared_Fz  = gpu_zeros(FT, max_nxp, max_nyp, max_nzp+1, Ncons)
    shared_Fvx = gpu_zeros(FT, max_nxp+1, max_nyp, max_nzp, Ncons)
    shared_Fvy = gpu_zeros(FT, max_nxp, max_nyp+1, max_nzp, Ncons)
    shared_Fvz = gpu_zeros(FT, max_nxp, max_nyp, max_nzp+1, Ncons)
    shared_dU_forced = gpu_zeros(FT, max_nxp, max_nyp, max_nzp, Ncons)



    forcex = zero(FT)
    flowx  = zero(FT)

    activeTime = zero(FT)
    tt = 0
    current_dt = dt

    # ── Checkpoint restart ──
    if restart != "none"
        restart_step = parse(Int, restart)
        if world_rank == 0
            println(">>> Restarting from checkpoint step $restart_step ...")
        end

        for (bid, b) in blocks
            block_rank_local = MPI.Comm_rank(block_comms[bid])
            chkname = "./CHK/chk-$(restart_step)-b$(b.id).h5"

            # Load this rank's Q array →supports 3 formats:
            #   1. NEW: partition-independent "Q" dataset (interior only)
            #   2. LEGACY: per-rank "Q_r0..Q_rN" datasets (with ghost)
            #   3. LEGACY: single 5D "Q_h" array (with ghost)
            Q_h = h5open(chkname, "r") do f
                if haskey(f, "Q")
                    # ── NEW FORMAT: partition-independent ──
                    # Read this rank's slice from global Q(Nx_block, Ny_block, Nz_block, Nprim)
                    lox = b.ox + 1; hix = b.ox + b.Nx
                    loy = b.oy + 1; hiy = b.oy + b.Ny
                    loz = b.oz + 1; hiz = b.oz + b.Nz
                    Q_interior = f["Q"][lox:hix, loy:hiy, loz:hiz, :]
                    nprim_chk = size(Q_interior, 4)

                    # Pad with ghost cells (zero-gradient; overwritten by fillGhost later)
                    Q_padded = zeros(FT, b.Nx+2*NG, b.Ny+2*NG, b.Nz+2*NG, nprim_chk)
                    Q_padded[NG+1:NG+b.Nx, NG+1:NG+b.Ny, NG+1:NG+b.Nz, :] = Q_interior
                    for n in 1:nprim_chk, g in 1:NG
                        Q_padded[g,:,:,n]       .= Q_padded[NG+1,:,:,n]
                        Q_padded[end-g+1,:,:,n] .= Q_padded[end-NG,:,:,n]
                        Q_padded[:,g,:,n]        .= Q_padded[:,NG+1,:,n]
                        Q_padded[:,end-g+1,:,n]  .= Q_padded[:,end-NG,:,n]
                        Q_padded[:,:,g,n]        .= Q_padded[:,:,NG+1,n]
                        Q_padded[:,:,end-g+1,n]  .= Q_padded[:,:,end-NG,n]
                    end
                    Q_padded
                elseif haskey(f, "Q_r$block_rank_local")
                    # ── LEGACY: per-rank datasets (partition-dependent) ──
                    read(f["Q_r$block_rank_local"])
                elseif haskey(f, "Q_h")
                    # ── LEGACY: single 5D array ──
                    f["Q_h"][:, :, :, :, block_rank_local + 1]
                else
                    error("Checkpoint $chkname has no recognized Q format (Q, Q_r*, Q_h)")
                end
            end
            copyto!(b.Q, GPUArray(FT.(Q_h)))

            # Recompute conservative variables from restored primitives
            nb_init = (cld(b.Nx + 2*NG, nthreads[1]),
                       cld(b.Ny + 2*NG, nthreads[2]),
                       cld(b.Nz + 2*NG, nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_init prim2c(b.U, b.Q, b.Nx, b.Ny, b.Nz)
        end

        # Restore step and physical time
        first_bid_r = first(blocks)[1]
        chkname_meta = "./CHK/chk-$(restart_step)-b$(first_bid_r).h5"
        h5open(chkname_meta, "r") do f
            if haskey(f, "step")
                val = read(f["step"])
                tt = Int(val isa AbstractArray ? val[1] : val)
            else
                tt = restart_step
            end
            if haskey(f, "time")
                val = read(f["time"])
                activeTime = FT(val isa AbstractArray ? val[1] : val)
            else
                activeTime = FT(tt) * dt
            end
        end

        if world_rank == 0
            println(">>> Restart complete: step=$tt, time=$activeTime")
        end
        GC.gc()  # Free restart temporary buffers
    end

    MPI.Barrier(MPI.COMM_WORLD)

    # Pre-calculate rank offsets for each block once (needed by ghost_pool and warmup)
    # (In multi-block mode, rank_offsets was already set to Block_to_rank values above)
    if !multi_block_mode
        rank_offsets = zeros(Int, length(Block_Nprocs) + 1)
        for i in 1:length(Block_Nprocs)
            rank_offsets[i+1] = rank_offsets[i] + prod(Block_Nprocs[i])
        end
    end

    # Pre-allocate inter-block ghost exchange buffers
    # (Moved before warmup so copy_ghost_face! can be used for coordinate exchange)
    ghost_pool = init_ghost_buffer_pool(blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b, Nprim)

    # (Spectral Warmup block removed as high-order CMD6 metrics resolve grid stretching)

    # ── Auto-tune kernel launch configurations ──
    first_bid, first_b = first(blocks)
    nxp_t, nyp_t, nzp_t = first_b.Nx, first_b.Ny, first_b.Nz
    tune_configs = KernelConfig[]

    # Reconstruction kernels →per-direction auto-tuning
    # Each direction gets its own optimal block size to match memory access patterns.
    _verbose = (world_rank == 0)
    if eigen_reconstruction
        cfg_recon_i = auto_tune_kernel("Eigen_recon_i", Eigen_reconstruct_i,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areai, shared_Fx, first_b.Areai,
            first_b.nxi, first_b.nyi, first_b.nzi, nxp_t, nyp_t, nzp_t,
            first_b.stencil_i, first_b.Δstencil_i, first_b.lin_phi_i,
            first_b.stencil_R_i, first_b.Δstencil_R_i;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_j = auto_tune_kernel("Eigen_recon_j", Eigen_reconstruct_j,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areaj, shared_Fy, first_b.Areaj,
            first_b.nxj, first_b.nyj, first_b.nzj, nxp_t, nyp_t, nzp_t,
            first_b.stencil_j, first_b.Δstencil_j, first_b.lin_phi_j,
            first_b.stencil_R_j, first_b.Δstencil_R_j, ch_glm_current, Int32(0);
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_k = auto_tune_kernel("Eigen_recon_k", Eigen_reconstruct_k,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areak, shared_Fz, first_b.Areak,
            first_b.nxk, first_b.nyk, first_b.nzk, nxp_t, nyp_t, nzp_t,
            first_b.stencil_k, first_b.Δstencil_k, first_b.lin_phi_k,
            first_b.stencil_R_k, first_b.Δstencil_R_k, ch_glm_current, Int32(0);
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
    else
        cfg_recon_i = auto_tune_kernel("Conser_recon_i", Conser_reconstruct_i,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areai, shared_Fx, first_b.Areai,
            first_b.nxi, first_b.nyi, first_b.nzi, nxp_t, nyp_t, nzp_t,
            first_b.stencil_i, first_b.Δstencil_i, first_b.lin_phi_i,
            first_b.stencil_R_i, first_b.Δstencil_R_i;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_j = auto_tune_kernel("Conser_recon_j", Conser_reconstruct_j,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areaj, shared_Fy, first_b.Areaj,
            first_b.nxj, first_b.nyj, first_b.nzj, nxp_t, nyp_t, nzp_t,
            first_b.stencil_j, first_b.Δstencil_j, first_b.lin_phi_j,
            first_b.stencil_R_j, first_b.Δstencil_R_j, ch_glm_current, Int32(0);
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_k = auto_tune_kernel("Conser_recon_k", Conser_reconstruct_k,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areak, shared_Fz, first_b.Areak,
            first_b.nxk, first_b.nyk, first_b.nzk, nxp_t, nyp_t, nzp_t,
            first_b.stencil_k, first_b.Δstencil_k, first_b.lin_phi_k,
            first_b.stencil_R_k, first_b.Δstencil_R_k, ch_glm_current, Int32(0);
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
    end
    push!(tune_configs, cfg_recon_i)
    push!(tune_configs, cfg_recon_j)
    push!(tune_configs, cfg_recon_k)
    threads_recon_i = cfg_recon_i.threads
    threads_recon_j = cfg_recon_j.threads
    threads_recon_k = cfg_recon_k.threads

    # Viscous flux kernels →per-direction
    if viscous || (equation_type == :MHD && resistive)
        cfg_visc_i = auto_tune_kernel("visc_i", viscous_flux_i,
            first_b.Q, shared_Fvx, first_b.Areai, first_b.Areaj, first_b.Areak,
            first_b.nxi, first_b.nyi, first_b.nzi,
            first_b.nxj, first_b.nyj, first_b.nzj,
            first_b.nxk, first_b.nyk, first_b.nzk,
            first_b.Vol, nxp_t, nyp_t, nzp_t, first_b.is_interblock;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_visc_j = auto_tune_kernel("visc_j", viscous_flux_j,
            first_b.Q, shared_Fvy, first_b.Areai, first_b.Areaj, first_b.Areak,
            first_b.nxi, first_b.nyi, first_b.nzi,
            first_b.nxj, first_b.nyj, first_b.nzj,
            first_b.nxk, first_b.nyk, first_b.nzk,
            first_b.Vol, nxp_t, nyp_t, nzp_t, first_b.is_interblock;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_visc_k = auto_tune_kernel("visc_k", viscous_flux_k,
            first_b.Q, shared_Fvz, first_b.Areai, first_b.Areaj, first_b.Areak,
            first_b.nxi, first_b.nyi, first_b.nzi,
            first_b.nxj, first_b.nyj, first_b.nzj,
            first_b.nxk, first_b.nyk, first_b.nzk,
            first_b.Vol, nxp_t, nyp_t, nzp_t, first_b.is_interblock;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        push!(tune_configs, cfg_visc_i)
        push!(tune_configs, cfg_visc_j)
        push!(tune_configs, cfg_visc_k)
        threads_visc_i = cfg_visc_i.threads
        threads_visc_j = cfg_visc_j.threads
        threads_visc_k = cfg_visc_k.threads
    else
        threads_visc_i = threads_recon_i
        threads_visc_j = threads_recon_j
        threads_visc_k = threads_recon_k
    end

    # Lightweight kernels (shockSensor, c2Prim, etc.)
    cfg_shock = auto_tune_kernel("shockSensor", shockSensor,
        first_b.ϕ, first_b.Q, nxp_t, nyp_t, nzp_t;
        nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
    push!(tune_configs, cfg_shock)
    threads_light = cfg_shock.threads

    cfg_c2p = auto_tune_kernel("c2Prim", c2Prim,
        first_b.U, first_b.Q, nxp_t, nyp_t, nzp_t;
        nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
    push!(tune_configs, cfg_c2p)

    if world_rank == 0
        gpu_name = gpu_device_name()
        print_tune_report(tune_configs, gpu_name)
        println(">>> All blocks loaded and initialized. Starting time loop...")
        if implicit
            if dual_time
                println(">>> Time advancement: IMPLICIT LU-SGS + BDF2 dual-time (CFL=$(implicit_CFL), sweeps=$(implicit_lusgs_sweeps), sub_iters=$(dual_time_sub_iters), tol=$(dual_time_tol))")
            else
                println(">>> Time advancement: IMPLICIT LU-SGS 1st-order (CFL=$(implicit_CFL), sweeps=$(implicit_lusgs_sweeps))")
            end
        else
            if adaptive_dt
                println(">>> Time advancement: EXPLICIT RK3 (adaptive_dt, CFL=$(CFL), LTS=$(LTS))")
            else
                println(">>> Time advancement: EXPLICIT RK3 (fixed dt=$(dt))")
            end
        end
    end

    # Sanity check: LTS is a sub-option of adaptive_dt
    if LTS && !adaptive_dt
        error("LTS=true requires adaptive_dt=true (LTS uses per-cell CFL-based dt)")
    end

    # Initialize allocation-free NaN check buffer
    _nan_flag_gpu[] = gpu_zeros(Int32, 1)

    # NOTE: rank_offsets and ghost_pool are now initialized BEFORE warmup (line ~1113+)
    # to enable coordinate exchange for stencil setup.

    # Sub-step timing for sync_blocks! diagnostics (must be before closure)
    # [1]=ghost_face1, [2]=fillGhost, [3]=exchange_ghost, [4]=ghost_face2, [5]=intf_filter, [6]=c2Prim
    t_sync_sub = zeros(Float64, 6)

    # Define unified synchronization function (two-pass ghost exchange)
    function sync_blocks!(tt_val; compute_fn=nothing)
        # ═══ Single-pass ghost exchange ═══
        # Step 1: Inter-block face ghost (η/ζ directions, real range only)
        if profiling; _ts = time_ns(); end
        copy_ghost_face!(blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b, :U, Ncons, ghost_pool;
                         full_range=false)
        for (bid, b) in blocks
            @check_nan(b.U, "U after copy_ghost_face!", b.id, world_rank, tt_val)
        end
        if profiling; gpu_sync(); t_sync_sub[1] += (time_ns()-_ts)/1e9; _ts=time_ns(); end

        # Step 2: Physical Boundary Conditions (fills wall/periodic ghost)
        for (bid, b) in blocks
            Nprocs_b = Block_Nprocs[bid + 1]
            rx_b = multi_block_mode ? 0 : rankx
            ry_b = multi_block_mode ? 0 : ranky
            rz_b = multi_block_mode ? 0 : rankz
            fillGhost(b.Q, b.U, rx_b, ry_b, rz_b,
                      b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj, b.nxk, b.nyk, b.nzk,
                      b.x, b.y, b.z, b.id, b.Nx, b.Ny, b.Nz, Nprocs_b, tt_val, face_bc, bc_params,
                      @isdefined(dsrfg_params) ? dsrfg_params : default_dsrfg_params)
            @check_nan(b.U, "U after fillGhost", b.id, world_rank, tt_val)
        end
        if profiling; gpu_sync(); t_sync_sub[2] += (time_ns()-_ts)/1e9; _ts=time_ns(); end

        # Step 3: ξ-direction MPI exchange →covers full j/k range (including ghost
        #         from steps 1-2) so i-j and i-k edge ghost cells are filled.
        for (bid, b) in blocks
            exchange_ghost(b.U, Ncons, block_comms[bid], b.Nx, b.Ny, b.Nz,
                           b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
                           b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
                           b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
                           sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
                           rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2,
                           compute_fn=compute_fn)
            @check_nan(b.U, "U after exchange_ghost", b.id, world_rank, tt_val)
        end
        if profiling; gpu_sync(); t_sync_sub[3] += (time_ns()-_ts)/1e9; _ts=time_ns(); end

        # Step 4: Full-range inter-block copy to fill j-k edge ghost cells
        # (now includes ghost from ξ-exchange in step 3)
        copy_ghost_face!(blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b, :U, Ncons, ghost_pool;
                         full_range=true, delta_mode=true)
        if profiling; gpu_sync(); t_sync_sub[4] += (time_ns()-_ts)/1e9; _ts=time_ns(); end

        # ─── CHECKERBOARD DIAGNOSTIC (opt-in) ───
        if @isdefined(checkerboard_diag) && checkerboard_diag
            checkerboard_diagnostic!(blocks, connectivity, tt, world_rank)
        end

        # ─── ADAPTIVE INTERFACE SMOOTHING ───
        # 8th-order filter with σ_local = σ_max × (1 - lin_phi):
        #   lin_phi high (upwind-biased regions) → σ low  → less filter
        #   lin_phi low  (central regions)       → σ high → more filter
        # Direction-specific lin_phi: η uses lin_phi_j, ζ uses lin_phi_k
        _intf_interval = @isdefined(intf_filter_interval) ? intf_filter_interval : 1
        if tt % _intf_interval == 0
            _σ_max = FT(0.01)
            for (bid, b) in blocks
                for ((dst_bid, fid), conn) in connectivity
                    if dst_bid == b.id
                        if fid == 1 || fid == 2
                            threads_f = (16, 16)
                            nb_loc_f = (cld(b.Ny, 16), cld(b.Nz, 16))
                            @gpu_launch threads=threads_f blocks=nb_loc_f interface_filter_kernel!(b.U, b.Nx, b.Ny, b.Nz, fid, 4, _σ_max, b.lin_phi_j)
                        elseif fid == 3 || fid == 4
                            threads_f = (16, 16)
                            nb_loc_f = (cld(b.Nx, 16), cld(b.Nz, 16))
                            @gpu_launch threads=threads_f blocks=nb_loc_f interface_filter_kernel!(b.U, b.Nx, b.Ny, b.Nz, fid, 4, _σ_max, b.lin_phi_j)
                        elseif fid == 5 || fid == 6
                            threads_f = (16, 16)
                            nb_loc_f = (cld(b.Nx, 16), cld(b.Ny, 16))
                            @gpu_launch threads=threads_f blocks=nb_loc_f interface_filter_kernel!(b.U, b.Nx, b.Ny, b.Nz, fid, 4, _σ_max, b.lin_phi_k)
                        end
                    end
                end
            end

            # ─── RE-SYNCHRONIZE AFTER INTERFACE FILTER ───
            gpu_sync()
            # 1. Update primitives (Q) in the outer real cell shell that was filtered
            for (bid, b) in blocks
                nb_loc = (cld(b.Nx+2*NG, nthreads[1]), cld(b.Ny+2*NG, nthreads[2]), cld(b.Nz+2*NG, nthreads[3]))
                @gpu_launch threads=nthreads blocks=nb_loc c2Prim_ghost(b.U, b.Q, b.Nx, b.Ny, b.Nz)
            end
            gpu_sync()
            # 2. Inter-block face ghost update for filtered U
            copy_ghost_face!(blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b, :U, Ncons, ghost_pool;
                             full_range=false)
            gpu_sync()
            # 3. Physical Boundary Conditions (reads the freshly updated Q/U in outer real cells)
            for (bid, b) in blocks
                Nprocs_b = Block_Nprocs[bid + 1]
                rx_b = multi_block_mode ? 0 : rankx
                ry_b = multi_block_mode ? 0 : ranky
                rz_b = multi_block_mode ? 0 : rankz
                fillGhost(b.Q, b.U, rx_b, ry_b, rz_b,
                          b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj, b.nxk, b.nyk, b.nzk,
                          b.x, b.y, b.z, b.id, b.Nx, b.Ny, b.Nz, Nprocs_b, tt_val, face_bc, bc_params,
                          @isdefined(dsrfg_params) ? dsrfg_params : default_dsrfg_params)
            end
            gpu_sync()
            # 4. Exchange ghost periodically
            for (bid, b) in blocks
                exchange_ghost(b.U, Ncons, block_comms[bid], b.Nx, b.Ny, b.Nz,
                               b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
                               b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
                               b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
                               sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
                               rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2,
                               compute_fn=compute_fn)
            end
            gpu_sync()
            # 5. Full-range copy
            copy_ghost_face!(blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b, :U, Ncons, ghost_pool;
                             full_range=true, delta_mode=true)
            gpu_sync()
        end
        if profiling; gpu_sync(); t_sync_sub[5] += (time_ns()-_ts)/1e9; _ts=time_ns(); end

        # Step 5: Ghost-only Primitive Refresh
        for (bid, b) in blocks
            nb_loc = (cld(b.Nx+2*NG, nthreads[1]), cld(b.Ny+2*NG, nthreads[2]), cld(b.Nz+2*NG, nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_loc c2Prim_ghost(b.U, b.Q, b.Nx, b.Ny, b.Nz)
            @check_nan(b.Q, "Q after ghost refresh", b.id, world_rank, tt_val)
        end
        if profiling; gpu_sync(); t_sync_sub[6] += (time_ns()-_ts)/1e9; end
    end

    # Initial ghost cell synchronization
    sync_blocks!(zero(FT))

    # Phase 2: Create compute stream for comm-compute overlap
    compute_stream = gpu_stream_create()

    # ══════════════════════════════════════════════════════════════
    # Timing accumulators (temporary profiling)
    # ══════════════════════════════════════════════════════════════
    t_sync = 0.0; t_shock = 0.0; t_advance = 0.0
    t_forcing = 0.0; t_div = 0.0
    timing_steps = 0
    _t0 = UInt64(0)

    # HIT forcing state
    hit_u_mean = zero(FT)
    hit_v_mean = zero(FT)
    hit_w_mean = zero(FT)

    # Forcing state (global scope for RK loop)
    forcex = zero(FT)
    flowx = zero(FT)
    cmf_f1_val = zero(FT)
    deschamps_f1_val = zero(FT)
    deschamps_flowx_val = zero(FT)
    ac_f1_val = zero(FT)


    # ═══ BDF2 History Initialization ═══
    # U_nm1 must hold U^0 (initial condition) so that the first BDF2 step
    # (tt=2) computes the correct temporal source term.
    # Without this, U_nm1 = 0 → add_bdf2_source! creates O(1/dt) spurious forcing → NaN.
    if dual_time
        for (bid, b) in blocks
            copyto!(b.U_nm1, b.U)
            copyto!(b.Un, b.U)
        end
    end

    global global_avg_count = 0
    max_time_limit = isdefined(Main, :Time) && !(Main.Time isa Type) ? Main.Time : (isdefined(Main, :maxTime) ? Main.maxTime : 100.0)
    while activeTime < max_time_limit && tt < maxStep
        tt = tt + 1

      # Determine whether to use implicit path this step
      # NOTE: must use isdefined() runtime function, NOT @isdefined macro,
      # because solver.jl is included BEFORE implicit_start_step is defined in run_config.jl
      _implicit_start = isdefined(@__MODULE__, :implicit_start_step) ? implicit_start_step : 0
      use_implicit = implicit && (tt > _implicit_start)

      if use_implicit
        # ═══════════════════════════════════════════════════════
        #  IMPLICIT LU-SGS PATH
        # ═══════════════════════════════════════════════════════

        # ── Adaptive dt with implicit CFL ──
        if adaptive_dt || !isdefined(@__MODULE__, :dt)
            dt_min = FT(1e10)
            for (bid, b) in blocks
                nb = (cld(b.Nx+2*NG, nthreads[1]), cld(b.Ny+2*NG, nthreads[2]), cld(b.Nz+2*NG, nthreads[3]))
                @gpu_launch threads=nthreads blocks=nb compute_dt(b.LTS_dt, b.Q, b.Vol, b.Areai, b.Areaj, b.Areak,
                    b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj, b.nxk, b.nyk, b.nzk,
                    b.Nx, b.Ny, b.Nz)
                # Use mapreduce on FULL array (avoids SubArray scalar indexing on AMDGPU)
                # Filter out ghost cells which have LTS_dt=0 (not computed by compute_dt)
                dt_local = FT(mapreduce(x -> x > zero(FT) ? x : FT(Inf), min, b.LTS_dt))
                dt_min = min(dt_min, dt_local)
            end
            # Scale by local_implicit_CFL / CFL to recover the implicit CFL
            local_implicit_CFL = implicit_CFL
            if isdefined(@__MODULE__, :implicit_CFL_max) && isdefined(@__MODULE__, :implicit_CFL_ramp_steps)
                step_offset = max(0, tt - _implicit_start)
                ramp_factor = min(one(FT), FT(step_offset) / FT(implicit_CFL_ramp_steps))
                local_implicit_CFL = implicit_CFL + ramp_factor * (implicit_CFL_max - implicit_CFL)
            end
            current_dt = MPI.Allreduce(dt_min, MPI.MIN, MPI.COMM_WORLD) * (local_implicit_CFL / CFL)
        else
            current_dt = FT(dt)  # Fixed dt from config
        end
        if isnan(current_dt)
            if world_rank == 0
                printstyled("CRITICAL: dt is NaN! The simulation has diverged.\n", color=:red)
                flush(stdout)
            end
            MPI.Abort(MPI.COMM_WORLD, 1)
            return blocks, activeTime
        end

        # ── Forcing parameters (computed once per step) ──
        if flow_forcing
            if forcing_mode == 1
                forcex, flowx = Update_bulk_force_params(blocks, tt, 1, MPI.COMM_WORLD, world_rank)
            elseif forcing_mode == 2
                cmf_f1_val = Update_const_massflux_params(blocks, tt, 1, current_dt, MPI.COMM_WORLD, world_rank)
            elseif forcing_mode == 3
                deschamps_f1_val, deschamps_flowx_val = Update_deschamps_pipe_params(blocks, tt, 1, current_dt, MPI.COMM_WORLD, world_rank)
            end
        end
        if test_case == "HIT"
            hit_u_mean, hit_v_mean, hit_w_mean = Update_HIT_forcing(blocks, hit_forcing_A, MPI.COMM_WORLD, world_rank, tt, 1)
        end

        # ── Shock sensor ──
        if profiling; _t0 = time_ns(); end
        for (bid, b) in blocks
            nb_l = (Int32(cld(b.Nx+2*NG, threads_light[1])), Int32(cld(b.Ny+2*NG, threads_light[2])), Int32(cld(b.Nz+2*NG, threads_light[3])))
            @gpu_launch threads=threads_light blocks=nb_l shockSensor(b.ϕ, b.Q, b.Nx, b.Ny, b.Nz)
        end
        for (bid, b) in blocks
            ϕ_4d = reshape(b.ϕ, size(b.ϕ, 1), size(b.ϕ, 2), size(b.ϕ, 3), 1)
            exchange_ghost(ϕ_4d, 1, block_comms[bid], b.Nx, b.Ny, b.Nz,
                b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
                b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
                b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
                sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
                rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2)
        end
        if profiling; gpu_sync(); t_shock += (time_ns() - _t0) / 1e9; end

        # ── Implicit LU-SGS step for each block ──
        if profiling; _t0 = time_ns(); end

        bdf2_active = dual_time && tt > (isdefined(@__MODULE__, :dual_time_start_step) ? dual_time_start_step : 1)

        if bdf2_active
            # ── BDF2 dual-time stepping (2nd-order temporal accuracy) ──

            # Save U^n into Un before advancing (U_nm1 was set at end of previous step)
            for (bid, b) in blocks
                copyto!(b.Un, b.U)
                # Compute spectral radii and LU-SGS diagonal once per physical step
                bdf2_prepare_step!(b, current_dt)
            end

            # ── Pseudo-time inner iterations ──
            prev_max_res = Inf
            min_max_res = Inf
            for m_iter = 1:dual_time_sub_iters
                # 1. Update boundary conditions and MPI ghost cells for current U^{n+1,m}
                sync_blocks!(activeTime)

                # 2. Perform one LU-SGS inner iteration for all blocks
                local_max_res = 0.0
                local_max_dU = 0.0
                local_max_p = FT(0.0)
                local_max_u = FT(0.0)
                for (bid, b) in blocks
                    res = bdf2_inner_iteration!(b, current_dt,
                        shared_Fx, shared_Fy, shared_Fz,
                        shared_Fvx, shared_Fvy, shared_Fvz,
                        shared_dU_forced, world_rank, tt,
                        threads_recon_i, threads_recon_j, threads_recon_k,
                        threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                        forcex, flowx, cmf_f1_val, ac_f1_val,
                        hit_u_mean, hit_v_mean, hit_w_mean, activeTime;
                        sync_ghost_fn = () -> sync_blocks!(activeTime))
                    local_max_res = max(local_max_res, res)
                    # Monitor solution growth (first 20 steps)
                    if tt <= 20
                        local_max_dU = max(local_max_dU, Float64(maximum(abs, b.ΔU)))
                        local_max_p = max(local_max_p, FT(maximum(abs, @view b.Q[:,:,:,1])))
                        local_max_u = max(local_max_u, FT(maximum(abs, @view b.Q[:,:,:,2])))
                    end
                    @check_nan(b.U, "U after bdf2 inner iter", b.id, world_rank, tt)
                end

                # Diagnostic: monitor solution growth per sub-iteration
                if tt <= 20
                    g_max_dU = MPI.Allreduce(local_max_dU, MPI.MAX, MPI.COMM_WORLD)
                    g_max_p  = MPI.Allreduce(Float64(local_max_p), MPI.MAX, MPI.COMM_WORLD)
                    g_max_u  = MPI.Allreduce(Float64(local_max_u), MPI.MAX, MPI.COMM_WORLD)
                    if world_rank == 0
                        @printf "    [diag] sub %d: max|ΔU|=%.3e  max|p|=%.3e  max|u|=%.3e\n" m_iter g_max_dU g_max_p g_max_u
                        flush(stdout)
                    end
                end

                # 3. Check global convergence
                global_max_res = MPI.Allreduce(local_max_res, MPI.MAX, MPI.COMM_WORLD)
                if world_rank == 0 && (tt % 100 == 0 || tt <= 20)
                    @printf "    BDF2 sub-iter %d/%d: max_res = %.4e\n" m_iter dual_time_sub_iters global_max_res
                    flush(stdout)
                end
                if global_max_res < dual_time_tol
                    if world_rank == 0 && (tt % 100 == 0 || tt <= 20)
                        @printf "    →Converged at sub-iter %d (tol=%.1e)\n" m_iter dual_time_tol
                    end
                    break
                end

                # (AC sub-iteration break removed →skew-symmetric + GMRES handles p-u coupling)

                # Safety: detect diverging sub-iterations and bail out
                # GMRES has non-monotonic convergence →compare against best residual, not previous
                _is_gmres = isdefined(@__MODULE__, :implicit_solver) ? (implicit_solver == :gmres) : false
                if _is_gmres
                    # GMRES: allow temporary increase, only bail if 10× worse than best seen
                    if m_iter > 1 && global_max_res > min_max_res * 10.0
                        if world_rank == 0 && (tt % 100 == 0 || tt <= 20)
                            @printf "    →Sub-iter diverging (%.2e > 10×%.2e), breaking\n" global_max_res min_max_res
                        end
                        break
                    end
                    min_max_res = min(min_max_res, global_max_res)
                else
                    # LU-SGS: monotonic convergence expected
                    if m_iter > 1 && global_max_res > prev_max_res * 2.0
                        if world_rank == 0 && (tt % 100 == 0 || tt <= 20)
                            @printf "    →Sub-iter diverging (%.2e > 2×%.2e), breaking\n" global_max_res prev_max_res
                        end
                        break
                    end
                end
                prev_max_res = global_max_res
            end


            # After advancing: shift history U^{n-1} →U^n (which is in Un)
            for (bid, b) in blocks
                copyto!(b.U_nm1, 1, b.Un, 1, length(b.U_nm1))
            end
        else
            # ── 1st-order backward Euler LU-SGS ──
            # (Also used for first step of BDF2 when U^{n-1} not yet available)
            if dual_time
                # Initialize U_nm1 for next step's BDF2
                for (bid, b) in blocks
                    copyto!(b.U_nm1, b.U)
                    copyto!(b.Un, b.U)
                end
            end

            for (bid, b) in blocks
                implicit_step!(b, current_dt,
                    shared_Fx, shared_Fy, shared_Fz,
                    shared_Fvx, shared_Fvy, shared_Fvz,
                    shared_dU_forced, world_rank, tt,
                    threads_recon_i, threads_recon_j, threads_recon_k,
                    threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                    forcex, flowx, cmf_f1_val, ac_f1_val,
                    hit_u_mean, hit_v_mean, hit_w_mean, activeTime)
                @check_nan(b.U, "U after implicit_step", b.id, world_rank, tt)
            end
        end
        if profiling; gpu_sync(); t_advance += (time_ns() - _t0) / 1e9; end

        # ── Sync blocks (ghost exchange + BC) ──
        if profiling; _t0 = time_ns(); end
        sync_blocks!(activeTime)
        if profiling; gpu_sync(); t_sync += (time_ns() - _t0) / 1e9; end

        # ── Check Q after sync (BC might cause issues) ──
        for (bid, b) in blocks
            @check_nan(b.Q, "Q after sync_blocks", b.id, world_rank, tt)
        end

      else
        # ═══════════════════════════════════════════════════════
        #  STANDARD EXPLICIT RK3 PATH (compressible / MHD)
        # ═══════════════════════════════════════════════════════
        # RK3
        for KRK = 1:3
            if KRK == 1
            for (bid, b) in blocks
                copyto!(b.Un, b.U)
            end
                
                # ── dt calculation ──
                # adaptive_dt (master switch) →compute CFL-based dt per cell
                #   └─ LTS=true  →use per-cell dt in div_LTS kernel
                #   └─ LTS=false →use global min(dt) across all ranks
                # adaptive_dt=false →use constant dt from config
                if adaptive_dt
                    dt_min = FT(1e10)
                    for (bid, b) in blocks
                        nb = (cld(b.Nx+2*NG, nthreads[1]), cld(b.Ny+2*NG, nthreads[2]), cld(b.Nz+2*NG, nthreads[3]))
                        @gpu_launch threads=nthreads blocks=nb compute_dt(b.LTS_dt, b.Q, b.Vol, b.Areai, b.Areaj, b.Areak,
                            b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj, b.nxk, b.nyk, b.nzk,
                            b.Nx, b.Ny, b.Nz)
                        
                        # Use minimum on FULL array (avoids SubArray scalar indexing on AMDGPU)
                        dt_local = FT(mapreduce(x -> x > zero(FT) ? x : FT(Inf), min, b.LTS_dt))
                        dt_min = min(dt_min, dt_local)
                    end
                    current_dt = MPI.Allreduce(dt_min, MPI.MIN, MPI.COMM_WORLD)
                    if tt <= 2
                        for (bid, b) in blocks
                            dt_h = Array(b.LTS_dt)
                            Q_h = Array(b.Q)
                            min_d = Inf
                            min_idx = (1+NG, 1+NG, 1+NG)
                            for kk in 1+NG:b.Nz+NG, jj in 1+NG:b.Ny+NG, ii in 1+NG:b.Nx+NG
                                if dt_h[ii,jj,kk] < min_d
                                    min_d = dt_h[ii,jj,kk]
                                    min_idx = (ii, jj, kk)
                                end
                            end
                            println("Rank $world_rank Block $bid step $tt: min_dt=$min_d at $min_idx, rho=$(Q_h[min_idx..., 1]), u=$(Q_h[min_idx..., 2]), v=$(Q_h[min_idx..., 3]), w=$(Q_h[min_idx..., 4]), p=$(Q_h[min_idx..., 5]), T=$(Q_h[min_idx..., 6])")
                        end
                        flush(stdout)
                    end

                    # ── MHD: auto-compute ch_glm from max fast magnetosonic speed ──
                    if equation_type == :MHD
                        cf_max_local = FT(0.0)
                        for (bid2, b2) in blocks
                            # Reuse LTS_dt array as temporary for cf values
                            nb_cf = (cld(b2.Nx+2*NG, nthreads[1]), cld(b2.Ny+2*NG, nthreads[2]), cld(b2.Nz+2*NG, nthreads[3]))
                            @gpu_launch threads=nthreads blocks=nb_cf compute_cf_max_kernel!(b2.LTS_dt, b2.Q, b2.Nx, b2.Ny, b2.Nz)
                            cf_local = FT(mapreduce(x -> x > zero(FT) ? x : zero(FT), max, b2.LTS_dt))
                            cf_max_local = max(cf_max_local, cf_local)
                        end
                        global ch_glm_current = MPI.Allreduce(cf_max_local, MPI.MAX, MPI.COMM_WORLD)
                        ch_glm_current = max(ch_glm_current, FT(1.0e-10))  # avoid zero
                    end

                    if current_dt == 0.0 && tt <= 100
                        # Identify the rank with the issue
                        if dt_min == 0.0
                            for (bid, b) in blocks
                                rho_view = @view b.Q[1+NG:b.Nx+NG, 1+NG:b.Ny+NG, 1+NG:b.Nz+NG, 1]
                                local_min_rho = minimum(rho_view)
                                # Find location of dt=0
                                dt_h = Array(b.LTS_dt)
                                vol_h = Array(b.Vol)
                                ai_h = Array(b.Areai)
                                Q_h = Array(b.Q)
                                for kk in 1+NG:b.Nz+NG, jj in 1+NG:b.Ny+NG, ii in 1+NG:b.Nx+NG
                                    if dt_h[ii,jj,kk] <= zero(FT) || isnan(dt_h[ii,jj,kk])
                                        rho_c = Q_h[ii,jj,kk,1]; u_c = Q_h[ii,jj,kk,2]; v_c = Q_h[ii,jj,kk,3]
                                        w_c = Q_h[ii,jj,kk,4]
                                        T_c = Nprim >= 6 ? Q_h[ii,jj,kk,6] : zero(FT)
                                        println("Rank $world_rank Block $bid: dt=0 at ($ii,$jj,$kk) dt=$(dt_h[ii,jj,kk]) Vol=$(vol_h[ii,jj,kk]) Areai=$(ai_h[ii,jj,kk]) ρ=$rho_c u=$u_c v=$v_c w=$w_c T=$T_c")
                                        break  # Only print first occurrence
                                    end
                                end
                                println("Rank $world_rank Block $bid: dt_min=0.0 at iteration $tt. min(rho)=$local_min_rho")
                            end
                        end
                    end
                    # Note: when LTS=true, current_dt is the global min (for logging);
                    # the actual per-cell dt is b.LTS_dt, used in div_LTS kernel.
                else
                    current_dt = dt
                end
                
                if flow_forcing
                    if forcing_mode == 1
                        forcex, flowx = Update_bulk_force_params(blocks, tt, 1, MPI.COMM_WORLD, world_rank)
                    elseif forcing_mode == 2
                        cmf_f1_val = Update_const_massflux_params(blocks, tt, 1, current_dt, MPI.COMM_WORLD, world_rank)
                    elseif forcing_mode == 3
                        deschamps_f1_val, deschamps_flowx_val = Update_deschamps_pipe_params(blocks, tt, 1, current_dt, MPI.COMM_WORLD, world_rank)
                    end
                end

                # HIT linear forcing: compute domain-averaged velocity
                if test_case == "HIT"
                    hit_u_mean, hit_v_mean, hit_w_mean = Update_HIT_forcing(blocks, hit_forcing_A, MPI.COMM_WORLD, world_rank, tt, KRK)
                end

                # ── Shock sensor (once per step, reused for all RK substeps) ──
                if profiling; _t0 = time_ns(); end
                for (bid, b) in blocks
                    nb_l = (Int32(cld(b.Nx+2*NG, threads_light[1])), Int32(cld(b.Ny+2*NG, threads_light[2])), Int32(cld(b.Nz+2*NG, threads_light[3])))
                    @gpu_launch threads=threads_light blocks=nb_l shockSensor(b.ϕ, b.Q, b.Nx, b.Ny, b.Nz)
                    @check_nan(b.ϕ, "ϕ after shockSensor", b.id, world_rank, tt)
                end
                # Sync ϕ across intra-block rank boundaries so the reconstruction stencil
                # max(ϕ[i-2:i+3]) sees consistent values →prevents false WENO activation
                for (bid, b) in blocks
                    ϕ_4d = reshape(b.ϕ, size(b.ϕ, 1), size(b.ϕ, 2), size(b.ϕ, 3), 1)
                    exchange_ghost(ϕ_4d, 1, block_comms[bid], b.Nx, b.Ny, b.Nz,
                        b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
                        b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
                        b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
                        sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
                        rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2)
                end
                if profiling; gpu_sync(); t_shock += (time_ns() - _t0) / 1e9; end

            end

            # ── Combined Block Loop (Advance + Forcing + Divergence) ──
            # Fused to prevent shared_Fx/Fy/Fz/Fvx/Fvy/Fvz and shared_dU_forced
            # overwrite issues in multi-block mode on a single card.
            for (bid, b) in blocks
                # ── Block advance (reconstruction + viscous) ──
                if profiling; _t0 = time_ns(); end
                if KRK == 1 || multi_block_mode
                    # First RK stage or multi-block: full blockAdvance
                    # (no pending interior work, or overlap disabled)
                    blockAdvance(b, current_dt, b.ϕ, shared_Fx, shared_Fy, shared_Fz, shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt, threads_recon_i, threads_recon_j, threads_recon_k, threads_visc_i, threads_visc_j, threads_visc_k)
                else
                    # Stages 2-3: interior was launched during previous sync's MPI
                    # and completed via copyto! device sync. Only boundary needed.
                    blockAdvance_boundary(b, current_dt, b.ϕ, shared_Fx, shared_Fy, shared_Fz, shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt, threads_recon_i, threads_recon_j, threads_recon_k)
                end
                if profiling; gpu_sync(); t_advance += (time_ns() - _t0) / 1e9; end

                # ── Forcing ──
                if profiling; _t0 = time_ns(); end
                nb_l = (Int32(cld(b.Nx+2*NG, threads_light[1])), Int32(cld(b.Ny+2*NG, threads_light[2])), Int32(cld(b.Nz+2*NG, threads_light[3])))
                if flow_forcing
                    # Skip Volume_force_kernel (Coriolis + centrifugal) when rotation is zero
                    if Omega_x != zero(FT)
                        x_rot_start_val = isdefined(Main, :x_rot_start) ? FT(Main.x_rot_start) : (isdefined(@__MODULE__, :x_rot_start) ? FT(x_rot_start) : zero(FT))
                        x_rot_end_val   = isdefined(Main, :x_rot_end)   ? FT(Main.x_rot_end)   : (isdefined(@__MODULE__, :x_rot_end)   ? FT(x_rot_end)   : zero(FT))
                        omega_x_val     = isdefined(Main, :Omega_x)     ? FT(Main.Omega_x)     : (isdefined(@__MODULE__, :Omega_x)     ? FT(Omega_x)     : zero(FT))
                        @gpu_launch threads=threads_light blocks=nb_l Volume_force_kernel!(shared_dU_forced, b.Q, b.x, b.y, b.z, b.Nx, b.Ny, b.Nz, b.Ωx, b.Ωy, b.Ωz, x_rot_start_val, x_rot_end_val, omega_x_val)
                        if forcing_mode == 1
                            Apply_bulk_force!(shared_dU_forced, b.Q, forcex, flowx, current_dt, b.Nx, b.Ny, b.Nz)
                        elseif forcing_mode == 2
                            Apply_const_massflux_force!(shared_dU_forced, b.Q, cmf_f1_val, b.Nx, b.Ny, b.Nz)
                        elseif forcing_mode == 3
                            Apply_deschamps_pipe_force!(shared_dU_forced, b.Q, deschamps_f1_val, deschamps_flowx_val, current_dt, b.Nx, b.Ny, b.Nz)
                        end
                        # ── Fringe forcing (differential rotation only) ──
                        if isdefined(@__MODULE__, :diffrot_enabled) && diffrot_enabled
                            @gpu_launch threads=threads_light blocks=nb_l fringe_forcing_kernel!(
                                shared_dU_forced, b.U, b.U_target_fringe, b.fringe_lambda,
                                b.Nx, b.Ny, b.Nz, current_dt)
                        end
                        Apply_trip_force!(shared_dU_forced, b.Q, b.x, b.y, b.z, b.Nx, b.Ny, b.Nz, activeTime)
                        @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(b.U, shared_dU_forced, current_dt, b.Vol, b.Nx, b.Ny, b.Nz)
                    else
                        # ── Fused path (Omega_x == 0): skip zero+rotation, directly update U ──
                        nb_f = (cld(b.Nx, nthreads[1]), cld(b.Ny, nthreads[2]), cld(b.Nz, nthreads[3]))
                        if forcing_mode == 1
                            # TODO: fused bulk force kernel (future optimization)
                            @gpu_launch threads=nthreads blocks=nb_f zero_dU_forced_kernel!(shared_dU_forced, b.Nx, b.Ny, b.Nz)
                            Apply_bulk_force!(shared_dU_forced, b.Q, forcex, flowx, current_dt, b.Nx, b.Ny, b.Nz)
                            Apply_trip_force!(shared_dU_forced, b.Q, b.x, b.y, b.z, b.Nx, b.Ny, b.Nz, activeTime)
                            @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(b.U, shared_dU_forced, current_dt, b.Vol, b.Nx, b.Ny, b.Nz)
                        elseif forcing_mode == 2
                            @gpu_launch threads=nthreads blocks=nb_f zero_dU_forced_kernel!(shared_dU_forced, b.Nx, b.Ny, b.Nz)
                            Apply_const_massflux_force!(shared_dU_forced, b.Q, cmf_f1_val, b.Nx, b.Ny, b.Nz)
                            Apply_trip_force!(shared_dU_forced, b.Q, b.x, b.y, b.z, b.Nx, b.Ny, b.Nz, activeTime)
                            @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(b.U, shared_dU_forced, current_dt, b.Vol, b.Nx, b.Ny, b.Nz)
                        elseif forcing_mode == 3
                            # Fused: deschamps + add_source in 1 kernel (no dU_forced needed)
                            @gpu_launch threads=nthreads blocks=nb_f fused_deschamps_source_kernel!(b.U, b.Q, deschamps_f1_val, deschamps_flowx_val, current_dt, b.Nx, b.Ny, b.Nz)
                            # Trip forcing still needs dU_forced as intermediate (heavy sin/atan →keep separate)
                            @gpu_launch threads=nthreads blocks=nb_f zero_dU_forced_kernel!(shared_dU_forced, b.Nx, b.Ny, b.Nz)
                            Apply_trip_force!(shared_dU_forced, b.Q, b.x, b.y, b.z, b.Nx, b.Ny, b.Nz, activeTime)
                            @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(b.U, shared_dU_forced, current_dt, b.Vol, b.Nx, b.Ny, b.Nz)
                        end
                    end
                end
                if test_case == "HIT"
                    Apply_HIT_forcing!(shared_dU_forced, b.Q, hit_forcing_A, hit_u_mean, hit_v_mean, hit_w_mean, b.Nx, b.Ny, b.Nz)
                    @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(b.U, shared_dU_forced, current_dt, b.Vol, b.Nx, b.Ny, b.Nz)
                end
                if profiling; gpu_sync(); t_forcing += (time_ns() - _t0) / 1e9; end

                # ── Divergence ──
                if profiling; _t0 = time_ns(); end
                nb_f = (cld(b.Nx, nthreads[1]), cld(b.Ny, nthreads[2]), cld(b.Nz, nthreads[3]))
                if LTS
                    # LTS uses per-cell dt →cannot fuse with RK combination
                    nb_l = (Int32(cld(b.Nx+2*NG, threads_light[1])), Int32(cld(b.Ny+2*NG, threads_light[2])), Int32(cld(b.Nz+2*NG, threads_light[3])))
                    @gpu_launch threads=threads_light blocks=nb_l div_LTS(b.U, shared_Fx, shared_Fy, shared_Fz, shared_Fvx, shared_Fvy, shared_Fvz, b.LTS_dt, b.Vol, b.Nx, b.Ny, b.Nz)
                    @check_nan(b.U, "U after divergence", b.id, world_rank, tt)
                    rk_a = KRK == 2 ? FT(0.25) : (KRK == 3 ? FT(2)/FT(3) : one(FT))
                    @gpu_launch threads=threads_light blocks=nb_l linComb_clip_prim(b.U, b.Un, b.Q, Ncons, rk_a, one(FT) - rk_a, b.Nx, b.Ny, b.Nz)
                else
                    # Fused: div + RK combination + clipping + c2Prim in one kernel
                    rk_a = KRK == 2 ? FT(0.25) : (KRK == 3 ? FT(2)/FT(3) : one(FT))
                    @gpu_launch threads=nthreads blocks=nb_f div_rk_clip_prim(b.U, b.Un, b.Q, shared_Fx, shared_Fy, shared_Fz, shared_Fvx, shared_Fvy, shared_Fvz, current_dt, b.Vol, rk_a, b.Nx, b.Ny, b.Nz)
                end
                @check_nan(b.Q, "Q after div_rk_clip", b.id, world_rank, tt)

                if tt == 1 && world_rank == 0
                    U_cpu = Array(b.U)
                    Un_cpu = Array(b.Un)
                    Q_cpu = Array(b.Q)
                    Fy_cpu = Array(shared_Fy)
                    Fvy_cpu = Array(shared_Fvy)
                    Vol_cpu = Array(b.Vol)
                    ii = NG + 8
                    kk = NG + 8
                    jj = NG + 1
                    
                    fy_1 = Fy_cpu[ii-NG, jj-NG, kk-NG, 6]
                    fy_2 = Fy_cpu[ii-NG, jj-NG+1, kk-NG, 6]
                    fvy_1 = Fvy_cpu[ii-NG, jj-NG, kk-NG, 6]
                    fvy_2 = Fvy_cpu[ii-NG, jj-NG+1, kk-NG, 6]
                    vol = Vol_cpu[ii, jj, kk]
                    
                    println("="^50)
                    println("DEBUG AT STEP 1:")
                    println("  ii=$ii, jj=$jj, kk=$kk")
                    println("  vol = ", vol)
                    println("  fy_1 (face j=NG) = ", fy_1)
                    println("  fy_2 (face j=NG+1) = ", fy_2)
                    println("  fvy_1 (face j=NG) = ", fvy_1)
                    println("  fvy_2 (face j=NG+1) = ", fvy_2)
                    println("  current_dt = ", current_dt)
                    println("  rk_a = ", rk_a)
                    
                    U_before = Un_cpu[ii, jj, kk, 6]
                    div_term = (fy_1 - fy_2) - (fvy_1 - fvy_2)
                    U_predicted_temp = U_before + div_term * current_dt * vol
                    U_predicted = U_before + rk_a * (U_predicted_temp - U_before)
                    
                    println("  U_before[6] (Un) = ", U_before)
                    println("  U_after_kernel[6] = ", U_cpu[ii, jj, kk, 6])
                    println("  U_predicted[6]    = ", U_predicted)
                    println("  Q_after_kernel[7] (Bx) = ", Q_cpu[ii, jj, kk, 7])
                    println("="^50)
                end


                # ── MHD: GLM ψ-damping source term (operator splitting) ──
                if equation_type == :MHD
                    nb_glm = (cld(b.Nx, nthreads[1]), cld(b.Ny, nthreads[2]), cld(b.Nz, nthreads[3]))
                    @gpu_launch threads=nthreads blocks=nb_glm glm_source_kernel!(b.U, current_dt, ch_glm_current, cr_glm, b.Nx, b.Ny, b.Nz)
                    # Update ψ in Q as well (Q[10] = U[9])
                    # This will be done by the next c2Prim / fillGhost cycle
                end
                if profiling; gpu_sync(); t_div += (time_ns() - _t0) / 1e9; end
            end

            # ── Sync blocks with comm-compute overlap (Phase 2) ──
            if profiling; _t0 = time_ns(); end
            if KRK < 3 && !multi_block_mode
                # Stages 1-2: launch NEXT stage's interior recon during MPI
                _ol = Ref(false)
                sync_blocks!(activeTime; compute_fn = function(slot::Int)
                    if !_ol[]
                        for (bid, b) in blocks
                            blockAdvance_interior(b, current_dt, b.ϕ,
                                shared_Fx, shared_Fy, shared_Fz,
                                shared_Fvx, shared_Fvy, shared_Fvz,
                                threads_recon_i, threads_recon_j, threads_recon_k,
                                threads_visc_i, threads_visc_j, threads_visc_k, compute_stream)
                        end
                        _ol[] = true
                    end
                end)
            else
                # Last stage or multi-block: plain sync (no next stage to overlap)
                sync_blocks!(activeTime)
            end
            if profiling; gpu_sync(); t_sync += (time_ns() - _t0) / 1e9; end
        end
      end  # if implicit/else

        # ══════════════════════════════════════════════════════════════
        # Explicit spatial filtering (Pirozzoli-style, applied after RK3)
        # 8th-order filter, dimension-by-dimension, σ = filtering_s0
        # ══════════════════════════════════════════════════════════════
        if filtering && (tt % filtering_interval == 0)
            for (bid, b) in blocks
                nb_l = (Int32(cld(b.Nx+2*NG, threads_light[1])), Int32(cld(b.Ny+2*NG, threads_light[2])), Int32(cld(b.Nz+2*NG, threads_light[3])))

                # Determine boundary-aware filter ranges
                # At physical (non-periodic, non-interblock) boundaries, shrink
                # filter range by 4 cells (8th-order stencil half-width) to avoid
                # reading from non-physical ghost cells.
                _is_phys(fid) = begin
                    bc = get(face_bc, (bid, fid), 0)
                    bc != BC_INTERBLOCK && bc != BC_PERIODIC && bc != 0
                end
                Nprocs_b = Block_Nprocs[bid + 1]
                rx_b = multi_block_mode ? 0 : rankx
                ry_b = multi_block_mode ? 0 : ranky
                rz_b = multi_block_mode ? 0 : rankz

                # x-direction: check ξ-/ξ+ faces (face 1/2)
                ilo = Int32(NG + 1)
                ihi = Int32(b.Nx + NG)
                if rx_b == 0 && _is_phys(1)
                    ilo = Int32(NG + 1 + 4)  # skip 4 cells near ξ- boundary
                end
                if rx_b == Nprocs_b[1] - 1 && _is_phys(2)
                    ihi = Int32(b.Nx + NG - 4)  # skip 4 cells near ξ+ boundary
                end

                # Filter x-direction: Un = snapshot, U = filtered output
                copyto!(b.Un, b.U)
                @gpu_launch threads=threads_light blocks=nb_l linearFilter_x(b.U, b.Un, filtering_s0, b.Nx, b.Ny, b.Nz, ilo, ihi)
                @gpu_launch threads=threads_light blocks=nb_l c2Prim(b.U, b.Q, b.Nx, b.Ny, b.Nz)
            end
            # Sync after x-filter: ghost gets x-filtered values before y-filter
            sync_blocks!(activeTime)

            for (bid, b) in blocks
                nb_l = (Int32(cld(b.Nx+2*NG, threads_light[1])), Int32(cld(b.Ny+2*NG, threads_light[2])), Int32(cld(b.Nz+2*NG, threads_light[3])))
                _is_phys_y(fid) = begin
                    bc = get(face_bc, (bid, fid), 0)
                    bc != BC_INTERBLOCK && bc != BC_PERIODIC && bc != 0
                end
                Nprocs_b = Block_Nprocs[bid + 1]
                ry_b = multi_block_mode ? 0 : ranky
                jlo = Int32(NG + 1); jhi = Int32(b.Ny + NG)
                if ry_b == 0 && _is_phys_y(3); jlo = Int32(NG + 1 + 4); end
                if ry_b == Nprocs_b[2] - 1 && _is_phys_y(4); jhi = Int32(b.Ny + NG - 4); end
                # Filter y-direction
                copyto!(b.Un, b.U)
                @gpu_launch threads=threads_light blocks=nb_l linearFilter_y(b.U, b.Un, filtering_s0, b.Nx, b.Ny, b.Nz, jlo, jhi)
                @gpu_launch threads=threads_light blocks=nb_l c2Prim(b.U, b.Q, b.Nx, b.Ny, b.Nz)
            end
            # Sync after y-filter: ghost gets x+y-filtered values before z-filter
            sync_blocks!(activeTime)

            for (bid, b) in blocks
                nb_l = (Int32(cld(b.Nx+2*NG, threads_light[1])), Int32(cld(b.Ny+2*NG, threads_light[2])), Int32(cld(b.Nz+2*NG, threads_light[3])))
                _is_phys_z(fid) = begin
                    bc = get(face_bc, (bid, fid), 0)
                    bc != BC_INTERBLOCK && bc != BC_PERIODIC && bc != 0
                end
                Nprocs_b = Block_Nprocs[bid + 1]
                rz_b = multi_block_mode ? 0 : rankz
                klo = Int32(NG + 1); khi = Int32(b.Nz + NG)
                if rz_b == 0 && _is_phys_z(5); klo = Int32(NG + 1 + 4); end
                if rz_b == Nprocs_b[3] - 1 && _is_phys_z(6); khi = Int32(b.Nz + NG - 4); end
                # Filter z-direction
                copyto!(b.Un, b.U)
                @gpu_launch threads=threads_light blocks=nb_l linearFilter_z(b.U, b.Un, filtering_s0, b.Nx, b.Ny, b.Nz, klo, khi)
                # Re-derive primitives from filtered conserved variables
                @gpu_launch threads=threads_light blocks=nb_l c2Prim(b.U, b.Q, b.Nx, b.Ny, b.Nz)
            end
            # Final sync after z-filter
            sync_blocks!(activeTime)
        end

        # ── Periodic Averaging ──
        if isdefined(Main, :average) && average && (tt % avg_step == 0)
            global global_avg_count += 1
            _do_favre = false
            if isdefined(Main, :avg_density_weighted) && avg_density_weighted
                _do_favre = true
            end
            for (bid, b) in blocks
                nb_avg = (Int32(cld(b.Nx+2*NG, threads_light[1])), Int32(cld(b.Ny+2*NG, threads_light[2])), Int32(cld(b.Nz+2*NG, threads_light[3])))
                @gpu_launch threads=threads_light blocks=nb_avg accumulate_avg_kernel!(b.Q_avg, b.Q, b.U, Int32(global_avg_count), Nprim, b.Nx, b.Ny, b.Nz, NG, _do_favre)
            end
            if tt % (avg_step * 10) == 0 && world_rank == 0
                println(">>> Time-averaging sample $(global_avg_count) collected.")
            end
        end

        # ── Periodic GC to reclaim leaked GPU/MPI temporaries ──
        if tt % 10000 == 0
            GC.gc(true)   # full GC: thorough scan, reclaim all unreachable objects
        elseif tt % 2000 == 0
            GC.gc(false)  # incremental GC: fast, reclaim young objects only
        end
        # ══════════════════════════════════════════════════════════════
        # Print timing breakdown every 100 steps (rank 0 only)
        # ══════════════════════════════════════════════════════════════
        # -- Fringe recovery residual (collective: all ranks must participate) --
        _fringe_resid = -1.0
        if isdefined(@__MODULE__, :diffrot_enabled) && diffrot_enabled && (tt % 100 == 0 || tt <= 20)
            _fr_num_local = 0.0  # ||U - U_target||² at inlet
            _fr_den_local = 0.0  # ||U_target||² at inlet
            for (bid, b) in blocks
                if b.U_target_fringe === nothing; continue; end
                # Sample inlet: first 4 interior x-cells — only copy this slice from GPU
                NGp = NG + 1
                ix_end = min(NG + 4, b.Nx + NG)
                U_slice = Array(@view b.U[NGp:ix_end, NGp:(b.Ny+NG), NGp:(b.Nz+NG), 1:Ncons])
                Ut_slice = Array(@view b.U_target_fringe[NGp:ix_end, NGp:(b.Ny+NG), NGp:(b.Nz+NG), 1:Ncons])
                for idx in eachindex(U_slice)
                    diff = Float64(U_slice[idx]) - Float64(Ut_slice[idx])
                    _fr_num_local += diff * diff
                    _fr_den_local += Float64(Ut_slice[idx])^2
                end
            end
            _fr_num_global = MPI.Allreduce(Float64(_fr_num_local), MPI.SUM, MPI.COMM_WORLD)
            _fr_den_global = MPI.Allreduce(Float64(_fr_den_local), MPI.SUM, MPI.COMM_WORLD)
            _fringe_resid = _fr_den_global > 0.0 ? sqrt(_fr_num_global / _fr_den_global) : 0.0
        end
        # -- Print timing breakdown --
        if profiling; timing_steps += 1; end

        if (tt % 100 == 0 || tt <= 20) && world_rank == 0
            printstyled("Step: ")
            @printf "%g" tt
            printstyled("\tTime: ")
            @printf "%.2e" activeTime
            printstyled("\tdt: ")
            @printf "%.2e" current_dt
            if use_implicit
                _solver_label = (isdefined(@__MODULE__, :implicit_solver) && implicit_solver == :gmres) ? "[GMRES]" : "[LU-SGS]"
                printstyled("\t$(_solver_label)")
            else
                printstyled("\t[RK3]")
            end
            printstyled("\tWall time: ")
            println("$(now())")

            # Print fringe recovery residual if available
            if _fringe_resid >= 0.0
                @printf("  Fringe inlet residual: %.4e\n", _fringe_resid)
            end

            if timing_steps > 0
                total_t = t_sync + t_shock + t_advance + t_forcing + t_div
                if total_t > 0
                    println("  ┌─────────────────────────────────────────────────────")
                    @printf("  →%-22s %8.3f s  (%5.1f%%)\n", "sync_blocks", t_sync, 100*t_sync/total_t)
                    if t_sync > 0
                        sync_labels = ["  ghost_face(real)", "  fillGhost", "  exchange_ghost(MPI)", "  ghost_face(full)", "  interface_filter", "  c2Prim_ghost"]
                        for si in 1:6
                            @printf("  →  %-20s %8.3f s  (%5.1f%%)\n", sync_labels[si], t_sync_sub[si], 100*t_sync_sub[si]/total_t)
                        end
                    end
                    @printf("  →%-22s %8.3f s  (%5.1f%%)\n", "shockSensor", t_shock, 100*t_shock/total_t)
                    @printf("  →%-22s %8.3f s  (%5.1f%%)\n", "blockAdvance", t_advance, 100*t_advance/total_t)
                    @printf("  →%-22s %8.3f s  (%5.1f%%)\n", "forcing", t_forcing, 100*t_forcing/total_t)
                    @printf("  →%-22s %8.3f s  (%5.1f%%)\n", "div+RK_clip", t_div, 100*t_div/total_t)
                    println("  │─────────────────────────────────────────────────────")
                    @printf("  →%-22s %8.3f s  (per step: %.4f s)\n", "TOTAL", total_t, total_t/(timing_steps*3))
                    println("  └─────────────────────────────────────────────────────")
                end
                # Reset accumulators
                t_sync = 0.0; t_shock = 0.0; t_advance = 0.0
                t_forcing = 0.0; t_div = 0.0
                t_sync_sub .= 0.0
                timing_steps = 0
            end

            flush(stdout)
        end



        if tt % step_plt == 0
            # Check for NaN in all assigned blocks (allocation-free)
            local_nan = false
            for (bid, b) in blocks
                nan_found, _ = _has_nan(b.U)
                if nan_found
                    local_nan = true
                    break
                end
            end
            nan_detected = MPI.Allreduce(local_nan, MPI.LOR, MPI.COMM_WORLD)
            
            if nan_detected
                if world_rank == 0
                    printstyled("Oops, NaN detected at step $tt\n", color=:red)
                    flush(stdout)
                end
                MPI.Abort(MPI.COMM_WORLD, 1)
                return
            end
        end

        if tt % step_plt == 0 || tt == maxStep
            plotFile_multiblock(tt, activeTime, blocks, world_rank, Nblocks, Block_Nprocs, block_comms)
        end

        if tt % step_chk == 0 || tt == maxStep
            checkpointFile(tt, activeTime, blocks, world_rank, Block_Nprocs, block_comms)
        end

        if tt % avg_total == 0 || tt == maxStep
            averageFile(tt, blocks, world_rank, Block_Nprocs, block_comms)
        end

        activeTime += current_dt
    end
    if world_rank == 0
        @printf(">>> Loop exited: activeTime=%.6e (limit=%.2f), tt=%d (maxStep=%d)\n", activeTime, max_time_limit, tt, maxStep)
        printstyled("Done!\n", color=:green)
        flush(stdout)
    end
    plotFile_multiblock(tt, activeTime, blocks, world_rank, Nblocks, Block_Nprocs, block_comms)
    MPI.Barrier(MPI.COMM_WORLD)
    return blocks, activeTime
end
