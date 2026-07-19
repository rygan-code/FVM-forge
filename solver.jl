using MPI
using StaticArrays
using HDF5, DelimitedFiles
using Dates, Printf

# GPU backend: loads CUDA or AMDGPU depending on what's available
include("gpu_backend.jl")
include("backend_interface.jl")
gpu_allowscalar(false)

include("bc_types.jl")
include("schemes.jl")
include("euler_flux.jl")
include("viscous.jl")
include("boundary.jl")
const default_dsrfg_params = create_dummy_dsrfg_params(FT)
include("utils.jl")
include("post_process.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Module-load-time evaluation of run-script feature flags.
#
# These are constants captured *once* at the moment solver.jl is included into
# Main. The hosting run script MUST therefore define the relevant flags BEFORE
# `include("solver.jl")` (also see boundary.jl header for Omega_x_min/max).
#
#   _fringe_active                : true  → fringe inlet/outlet zone is built and
#                                            forcing kernel is launched each step.
#                                  false → outlet falls back to NSCBC/zero_gradient.
#   _diffrot_volume_force_active : true  → legacy non-wall differential-rotation
#                                            path: applies Coriolis+centrifugal as
#                                            a volume force.
#                                  false → rotation comes solely from the wall BC
#                                            (BC_DIFFROT_WALL). Default for safety.
# ═══════════════════════════════════════════════════════════════════════════════
const _fringe_active = (isdefined(Main, :fringe_enabled) ? Main.fringe_enabled :
                        (isdefined(Main, :diffrot_enabled) && Main.diffrot_enabled))
const _diffrot_volume_force_active = ((isdefined(Main, :diffrot_enabled) && Main.diffrot_enabled) &&
                                      !(isdefined(Main, :diffrot_wall_bc) ? Main.diffrot_wall_bc : true))
const _outlet_sponge_active = (isdefined(Main, :outlet_sponge) ? Main.outlet_sponge :
                               (isdefined(Main, :outlet_bc) && Main.outlet_bc == :nscbc))

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

# Global topology singularity info, set at solver init by
# load_multiblock_connectivity and consumed by apply_topology_smoothness_protection!
# inside load_block. `nothing` until set.
global _SING_INFO = nothing
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
include("ct_state.jl")
include("ct_positivity.jl")
include("Riemann_Solver.jl")


# Topology-based multi-block edge singularity detection + adaptive WENO5
# ϕ boost. See docs/superpowers/specs/2026-06-20-topology-singularity-protection.md.

include("weno7.jl")
include("ct_weno7.jl")
include("Reconstruct.jl")
include("volume_force.jl")


include("ghost_coords.jl")
include("auto_tune.jl")
include("filter_interface.jl")
include("fringe.jl")
include("sponge.jl")
include("ct_sts.jl")
include("ct_rkl2.jl")
include("ct.jl")
include("ct_sync.jl")

# ── GLM cleaning speed for MHD (updated each time step) ──
# Auto-computed from max fast magnetosonic speed
const _ch_glm_ref = Ref{FT}(one(FT))
ch_glm_current::FT = one(FT)  # will be overwritten in time_step loop

# Rank-shared WENO7 buffers are allocated by time_step and referenced by the
# reconstruction launch helpers, including implicit callers of blockAdvance.
const _ct_weno7_cache_i_ref = Ref{Any}(nothing)
const _ct_weno7_cache_j_ref = Ref{Any}(nothing)
const _ct_weno7_cache_k_ref = Ref{Any}(nothing)
const _ct_weno7_point_scratch_ref = Ref{Any}(nothing)
const _ct_weno7_fail_meta_ref = Ref{Any}(nothing)
const _ct_weno7_fail_value_ref = Ref{Any}(nothing)

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

# ─── Entropy diagnostic kernel ───
# Compute per-cell mathematical entropy density η_i = -ρ·s·Vol with
# s = log(p) - γ·log(ρ). Caller does global sum + MPI.Allreduce.
# Used as a non-invasive sanity check that total mathematical entropy
# η = -ρs/(γ-1) is non-increasing over time (modulo boundary forcing).
# Runs every 100 steps for all flow configurations.
function entropy_total_kernel!(η_out, U, Vol, γ_local::FT, NG_::Int32,
                                Nx_::Int32, Ny_::Int32, Nz_::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i <= NG_ || i > Nx_ + NG_; return; end
    if j <= NG_ || j > Ny_ + NG_; return; end
    if k <= NG_ || k > Nz_ + NG_; return; end
    @inbounds begin
        ρ  = U[i, j, k, 1]
        ρu = U[i, j, k, 2]; ρv = U[i, j, k, 3]; ρw = U[i, j, k, 4]
        ρE = U[i, j, k, 5]
        u, v, w = ρu/ρ, ρv/ρ, ρw/ρ
        K = FT(0.5)*(u*u + v*v + w*w)
        p = (γ_local - one(FT)) * (ρE - ρ*K)
        # Guard against negative p / ρ which would NaN log; produce a
        # large positive sentinel that loudly indicates trouble in the
        # diagnostic without crashing the simulation.
        if p > zero(FT) && ρ > zero(FT)
            s = log(p) - γ_local * log(ρ)
            η_out[i, j, k] = -ρ * s * Vol[i, j, k]
        else
            η_out[i, j, k] = FT(NaN)
        end
    end
    return
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

                if isempty(_nan_indices)
                    if $rank == 0
                        @warn "NaN check: GPU flagged NaN in $($label) but CPU findall found none (Async race?). Continuing simulation."
                    end
                else
                    printstyled(_msg, color=:red)
                    flush(stdout)
                    MPI.Abort(MPI.COMM_WORLD, 1)
                end
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
    # ─── CT oriented face magnetic fluxes (staggered, with ghosts) ───
    # Legacy field names are retained for compatibility. Their values are
    # Phi_B=(B dot n)A on the i/j/k faces, not Cartesian Bx/By/Bz components.
    # _n variants are RK backups (same role as Un for cell-centered U)
    Bx_face::Union{GPUArray{FT, 3}, Nothing}
    By_face::Union{GPUArray{FT, 3}, Nothing}
    Bz_face::Union{GPUArray{FT, 3}, Nothing}
    Bx_face_n::Union{GPUArray{FT, 3}, Nothing}
    By_face_n::Union{GPUArray{FT, 3}, Nothing}
    Bz_face_n::Union{GPUArray{FT, 3}, Nothing}
    # CT edge line-integrated EMFs. These must persist across the per-block
    # reconstruction loop so shared physical edges can be synchronized before
    # any face receives its discrete-Stokes update.
    Ex_edge::Union{GPUArray{FT, 3}, Nothing}
    Ey_edge::Union{GPUArray{FT, 3}, Nothing}
    Ez_edge::Union{GPUArray{FT, 3}, Nothing}
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
    # ─── Outlet sponge buffer (differential rotation, nothing when not active) ───
    sponge_sigma::Union{GPUArray{FT, 3}, Nothing}       # (Nx_tot, Ny_tot, Nz_tot)
    sponge_U_target::Union{GPUArray{FT, 4}, Nothing}    # (Nx_tot, Ny_tot, Nz_tot, Ncons)
end

include("implicit.jl")

function load_block(bid, rx, ry, rz, NG, Ncons, Nprim, Nprocs_block, world_rank,
                    face_bc, connectivity, rank_offsets, temp_metrics_h, was_computed_h)
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
    cache_path, was_computed, Areai_h, nxi_h, nyi_h, nzi_h, Areaj_h, nxj_h, nyj_h, nzj_h,
    Areak_h, nxk_h, nyk_h, nzk_h, Vol_h = load_or_compute_metrics(
        bid, rx, ry, rz, x_full, y_full, z_full, nxp, nyp, nzp, NG;
        cache_metrics=cache_metrics,
        periodic=ntuple(3) do direction
            isdefined(Main, :Iperiodic) && Main.Iperiodic[direction] &&
                Nprocs_block[direction] == 1
        end,
    )
    
    temp_metrics_h[bid] = (cache_path, Areai_h, nxi_h, nyi_h, nzi_h, Areaj_h, nxj_h, nyj_h, nzj_h, Areak_h, nxk_h, nyk_h, nzk_h, Vol_h)
    was_computed_h[bid] = was_computed
    
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

    # CT: Initialize face-centered B from cell-centered B
    if equation_type == :MHD && ct_mode
        # Face B arrays will be allocated below; init is deferred to after Block construction
    end



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

    nb = (cld(Nx_tot, nthreads[1]), cld(Ny_tot, nthreads[2]), cld(Nz_tot, nthreads[3]))

    # ─── Fringe region initialization (only when _fringe_active) ───
    if _fringe_active
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
                _path_str = isdefined(@__MODULE__, :precursor_mean_path) ? precursor_mean_path : "not defined"
                @warn "Precursor file $(_path_str), U_target = 0"
            end
        end
        U_target_arr = GPUArray(U_target_cpu)
        if world_rank == 0
            println("  Fringe region initialized: λ_max=$(fringe_lambda_max), L_phys=$(L_phys), L_fringe=$(L_total - L_phys)")
        end
    end

    # ─── Outlet sponge initialization (only when _outlet_sponge_active) ───
    # Mirror of fringe: σ(x) ramps over the last rise_fraction of the axial extent.
    # U_target initialized to a copy of U (tiled outlet-cross-section mean later).
    if _outlet_sponge_active
        sponge_sigma_arr = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot)
        x_phys_start_s = isdefined(Main, :sponge_x_phys_start) ? Main.sponge_x_phys_start : FT(0)
        x_total_s      = isdefined(Main, :sponge_x_total)      ? Main.sponge_x_total      :
                         (isdefined(Main, :L_total) ? Main.L_total : FT(maximum(Array(x))))
        sigma_max_s    = isdefined(Main, :sponge_sigma_max)    ? Main.sponge_sigma_max    : FT(fringe_lambda_max * 0.5)
        rise_frac_s    = isdefined(Main, :sponge_rise_fraction) ? Main.sponge_rise_fraction : FT(0.10)
        @gpu_launch threads=nthreads blocks=nb compute_sponge_sigma_kernel!(
            sponge_sigma_arr, x, nxp, nyp, nzp,
            FT(x_phys_start_s), FT(x_total_s), FT(sigma_max_s), FT(rise_frac_s))
        # U_target: initialize to current U (running average updated later).
        sponge_U_target_cpu = Array(U)
        sponge_U_target_arr = GPUArray(sponge_U_target_cpu)
        if world_rank == 0
            println("  Outlet sponge initialized: σ_max=$sigma_max_s, x_total=$x_total_s, rise_frac=$rise_frac_s")
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
    # Topological singularity protection (multi-block edge junctions).
    # Complements the geometric protection above: catches ≥3-block junction
    # edges that the geometric criterion (face-normal angle, area stretch)
    # misses. See docs/superpowers/specs/2026-06-20-topology-singularity-protection.md.
    # Q3=A: both protections run, take max(ϕ) implicitly (only boost if higher).
    global _SING_INFO
    if isdefined(Main, :_SING_INFO) && _SING_INFO !== nothing
        apply_topology_smoothness_protection!(
            _Lin_j, _ΔLin_j, _phi_j,
            _Lin_k, _ΔLin_k, _phi_k,
            bid + 1, nxp, nyp, nzp, NG, _SING_INFO;
            verbose=true
        )
    end
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

    # ─── CT oriented face magnetic-flux arrays (only when ct_mode) ───
    # Bx_face: (Nx+1+2NG, Ny+2NG+1, Nz+2NG+1) -- +1 tangential ghost for extended reconstruct
    # By_face: (Nx+2NG+1, Ny+1+2NG, Nz+2NG+1)
    # Bz_face: (Nx+2NG+1, Ny+2NG+1, Nz+1+2NG)
    _ct_bx = ct_mode ? gpu_zeros(FT, nxp + 1 + 2*NG, nyp + 2*NG + 1, nzp + 2*NG + 1) : nothing
    _ct_by = ct_mode ? gpu_zeros(FT, nxp + 2*NG + 1, nyp + 1 + 2*NG, nzp + 2*NG + 1) : nothing
    _ct_bz = ct_mode ? gpu_zeros(FT, nxp + 2*NG + 1, nyp + 2*NG + 1, nzp + 1 + 2*NG) : nothing
    _ct_bx_n = ct_mode ? gpu_zeros(FT, nxp + 1 + 2*NG, nyp + 2*NG + 1, nzp + 2*NG + 1) : nothing
    _ct_by_n = ct_mode ? gpu_zeros(FT, nxp + 2*NG + 1, nyp + 1 + 2*NG, nzp + 2*NG + 1) : nothing
    _ct_bz_n = ct_mode ? gpu_zeros(FT, nxp + 2*NG + 1, nyp + 2*NG + 1, nzp + 1 + 2*NG) : nothing
    _ct_ex = ct_mode ? gpu_zeros(FT, nxp, nyp + 1, nzp + 1) : nothing
    _ct_ey = ct_mode ? gpu_zeros(FT, nxp + 1, nyp, nzp + 1) : nothing
    _ct_ez = ct_mode ? gpu_zeros(FT, nxp + 1, nyp + 1, nzp) : nothing

    return Block(bid, nxp, nyp, nzp, rx, ry, rz, ox, oy, oz, Q, U, ϕ, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, Vol, x, y, z, LTS_dt, Un,
                 _ct_bx, _ct_by, _ct_bz, _ct_bx_n, _ct_by_n, _ct_bz_n,
                 _ct_ex, _ct_ey, _ct_ez,
                 sbuf_hx, sbuf_dx, rbuf_hx, rbuf_dx, sbuf_hx2, sbuf_dx2, rbuf_hx2, rbuf_dx2,
                 sbuf_hy, sbuf_dy, rbuf_hy, rbuf_dy, sbuf_hz, sbuf_dz, rbuf_hz, rbuf_dz,
                 nb,
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
                 # Fringe region fields (only when _fringe_active)
                 _fringe_active ? fringe_lambda_arr : nothing,
                 _fringe_active ? U_target_arr : nothing,
                 # Outlet sponge fields (only when _outlet_sponge_active)
                 _outlet_sponge_active ? sponge_sigma_arr : nothing,
                 _outlet_sponge_active ? sponge_U_target_arr : nothing)
end

@inline function _connectivity_row_is_placeholder(
    block, local_face, neighbor_block, neighbor_face,
    reverse_tan, flip_normal,
)
    return block == 0 && local_face == 0 &&
           neighbor_block == 0 && neighbor_face == 0 &&
           !reverse_tan && !flip_normal
end

function _validate_connectivity_topology(connectivity, face_bc)
    for ((block, face), conn) in connectivity
        block != conn.src_b || error(
            "Connectivity cannot join block $block to itself: " *
            "($block,$face) -> ($(conn.src_b),$(conn.src_f))",
        )
        peer_key = (conn.src_b, conn.src_f)
        haskey(connectivity, peer_key) || error(
            "Missing reciprocal connectivity for " *
            "($block,$face) -> ($(conn.src_b),$(conn.src_f))",
        )
        peer = connectivity[peer_key]
        if peer.src_b != block || peer.src_f != face ||
           peer.reverse_tan != conn.reverse_tan ||
           peer.flip_normal != conn.flip_normal
            error(
                "Inconsistent reciprocal connectivity for " *
                "($block,$face) -> ($(conn.src_b),$(conn.src_f))",
            )
        end
    end

    if face_bc !== nothing
        for endpoint in keys(connectivity)
            get(face_bc, endpoint, nothing) == 0 || error(
                "Connectivity endpoint $endpoint must use interblock BC 0",
            )
        end
        for (endpoint, bc) in face_bc
            bc == 0 && !haskey(connectivity, endpoint) && error(
                "Interblock BC 0 at $endpoint has no connectivity entry",
            )
        end
    end
    return nothing
end

function load_multiblock_connectivity(path; world_rank::Int=0)
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
        if _connectivity_row_is_placeholder(b1, f1, b2, f2, rev, flip)
            continue
        end
        if !(0 <= b1 < Nblocks && 0 <= b2 < Nblocks &&
             1 <= f1 <= 6 && 1 <= f2 <= 6)
            error(
                "Malformed connectivity row $i: " *
                "($b1,$f1) -> ($b2,$f2), Nblocks=$Nblocks",
            )
        end
        haskey(connectivity, (b1, f1)) && error(
            "Duplicate connectivity endpoint at row $i: block=$b1 face=$f1",
        )
        connectivity[(b1, f1)] = Connectivity(b2, f2, rev, flip)
    end
    
    # Build face_bc dict: (block_id, face_id) -> bc_type
    face_bc = Dict{Tuple{Int, Int}, Int}()
    if face_bc_raw !== nothing
        for bid in 1:Nblocks, fid_idx in 1:6
            face_bc[(bid-1, fid_idx)] = Int(face_bc_raw[bid, fid_idx])
        end
    end
    _validate_connectivity_topology(
        connectivity, face_bc_raw === nothing ? nothing : face_bc,
    )

    # Pre-compute topology singularity info (mesh-agnostic multi-block edge
    # junction detection). Used by apply_topology_smoothness_protection! in
    # the per-block init to boost ϕ near ≥3-block junction edges, suppressing
    # topology-induced reconstruction artifacts that the geometric criterion
    # (face-normal angle, area stretch) misses. See
    # docs/superpowers/specs/2026-06-20-topology-singularity-protection.md.
    # Build 1-indexed (face_bc_array, neighbors_array) for detect_edge_singularities.
    global _SING_INFO
    local sing_info::EdgeSingularityInfo
    if Nblocks > 1 && !isempty(face_bc)
        face_bc_array = zeros(Int, Nblocks, 6)
        neighbors_array = zeros(Int, Nblocks, 6)
        for bid in 1:Nblocks, fid_idx in 1:6
            bc = get(face_bc, (bid-1, fid_idx), 42)  # default wall if missing
            face_bc_array[bid, fid_idx] = bc
            if bc == 0  # interblock: look up neighbor
                conn = get(connectivity, (bid-1, fid_idx), nothing)
                neighbors_array[bid, fid_idx] = conn === nothing ? 0 : (conn.src_b + 1)
            elseif bc == 2  # periodic: self-loop
                neighbors_array[bid, fid_idx] = bid
            end
        end
        sing_info = detect_edge_singularities(face_bc_array, neighbors_array, Nblocks)
        n_sing = count(sing_info.is_singularity_edge)
        if world_rank == 0
            println(">>> Topology singularity detection: $n_sing singularity edge entries across $Nblocks blocks")
        end
    else
        # Single-block or no connectivity: empty singularity table
        sing_info = EdgeSingularityInfo(falses(1, 12), falses(1, 8, 1), 1)
    end
    _SING_INFO = sing_info
    
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


function blockAdvance(block::Block, dt, ϕ, Fx, Fy, Fz,
                      rho_sum_x, rho_sum_y, rho_sum_z,
                      Fv_x, Fv_y, Fv_z, world_rank, tt,
                      threads_recon_i, threads_recon_j, threads_recon_k,
                      threads_visc_i, threads_visc_j, threads_visc_k,
                      rk_stage::Int32, pos_meta, pos_values)
    Q = block.Q
    U = block.U
    Areai, nxi, nyi, nzi = block.Areai, block.nxi, block.nyi, block.nzi
    Areaj, nxj, nyj, nzj = block.Areaj, block.nxj, block.nyj, block.nzj
    Areak, nxk, nyk, nzk = block.Areak, block.nxk, block.nyk, block.nzk
    Vol = block.Vol
    cache_i = _ct_weno7_cache_i_ref[]
    cache_j = _ct_weno7_cache_j_ref[]
    cache_k = _ct_weno7_cache_k_ref[]
    weno7_fail_meta = _ct_weno7_fail_meta_ref[]
    weno7_fail_value = _ct_weno7_fail_value_ref[]
    nxp, nyp, nzp = block.Nx, block.Ny, block.Nz

    # Per-direction grid sizes using per-direction auto-tuned thread configs
    nb_recon_i = (Int32(cld(nxp+2*NG, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
    nb_recon_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+2*NG, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
    nb_recon_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+2*NG, threads_recon_k[3])))
    @static if ct_emf_scheme == CT_EMF_WENO7_SG07
        nb_weno_i = (Int32(cld(nxp+NG, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
        nb_weno_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+NG, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
        nb_weno_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+NG, threads_recon_k[3])))
        nb_weno_scan_i = (Int32(cld(nxp+1, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
        nb_weno_scan_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+1, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
        nb_weno_scan_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+1, threads_recon_k[3])))
        nb_conser_i, nb_conser_j, nb_conser_k = nb_weno_i, nb_weno_j, nb_weno_k
    else
        nb_conser_i, nb_conser_j, nb_conser_k = nb_recon_i, nb_recon_j, nb_recon_k
    end
    nb_visc_i  = (Int32(cld(nxp+2*NG, threads_visc_i[1])),  Int32(cld(nyp+2*NG, threads_visc_i[2])),  Int32(cld(nzp+2*NG, threads_visc_i[3])))
    nb_visc_j  = (Int32(cld(nxp+2*NG, threads_visc_j[1])),  Int32(cld(nyp+2*NG, threads_visc_j[2])),  Int32(cld(nzp+2*NG, threads_visc_j[3])))
    nb_visc_k  = (Int32(cld(nxp+2*NG, threads_visc_k[1])),  Int32(cld(nyp+2*NG, threads_visc_k[2])),  Int32(cld(nzp+2*NG, threads_visc_k[3])))

    si = block.stencil_i; Δsi = block.Δstencil_i; lpi = block.lin_phi_i
    sj = block.stencil_j; Δsj = block.Δstencil_j; lpj = block.lin_phi_j
    sk = block.stencil_k; Δsk = block.Δstencil_k; lpk = block.lin_phi_k
    sRi = block.stencil_R_i; ΔsRi = block.Δstencil_R_i
    sRj = block.stencil_R_j; ΔsRj = block.Δstencil_R_j
    sRk = block.stencil_R_k; ΔsRk = block.Δstencil_R_k
    @static if ct_emf_scheme == CT_EMF_WENO7_SG07
        ct_weno7_reset_failure!(weno7_fail_meta, weno7_fail_value)
    end
    if eigen_reconstruction && equation_type == :MHD && ct_mode
        # F* temporarily stores left states and Fv_* stores right states. HLLD
        # consumes them immediately; viscous/resistive kernels overwrite Fv_* below.
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i ct_mhd_characteristic_reconstruct_left_i_kernel!(Q, Fx, Areai, nxi, nyi, nzi, block.Bx_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values)
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i ct_mhd_characteristic_reconstruct_right_i_kernel!(Q, Fv_x, Areai, nxi, nyi, nzi, block.Bx_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values)
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i ct_mhd_hlld_flux_i_kernel!(Fx, Fv_x, Fx, rho_sum_x, Areai, nxi, nyi, nzi, nxp, nyp, nzp, ch_glm_current, Int32(0), pos_meta)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fx, "Fx after split characteristic HLLD", block.id, world_rank, tt)
        @static if ct_emf_scheme == CT_EMF_WENO7_SG07
            @gpu_launch threads=threads_recon_i blocks=nb_weno_i ct_mhd_characteristic_weno7_cache_i_kernel!(Q, cache_i, Areai, nxi, nyi, nzi, block.Bx_face, Vol, nxp, nyp, nzp, ch_glm_current, dt, Int32(0))
        end

        @gpu_launch threads=threads_recon_j blocks=nb_recon_j ct_mhd_characteristic_reconstruct_left_j_kernel!(Q, Fy, Areaj, nxj, nyj, nzj, block.By_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values)
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j ct_mhd_characteristic_reconstruct_right_j_kernel!(Q, Fv_y, Areaj, nxj, nyj, nzj, block.By_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values)
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j ct_mhd_hlld_flux_j_kernel!(Fy, Fv_y, Fy, rho_sum_y, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, ch_glm_current, Int32(0), pos_meta)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fy, "Fy after split characteristic HLLD", block.id, world_rank, tt)
        @static if ct_emf_scheme == CT_EMF_WENO7_SG07
            @gpu_launch threads=threads_recon_j blocks=nb_weno_j ct_mhd_characteristic_weno7_cache_j_kernel!(Q, cache_j, Areaj, nxj, nyj, nzj, block.By_face, Vol, nxp, nyp, nzp, ch_glm_current, dt, Int32(0))
        end

        @gpu_launch threads=threads_recon_k blocks=nb_recon_k ct_mhd_characteristic_reconstruct_left_k_kernel!(Q, Fz, Areak, nxk, nyk, nzk, block.Bz_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values)
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k ct_mhd_characteristic_reconstruct_right_k_kernel!(Q, Fv_z, Areak, nxk, nyk, nzk, block.Bz_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values)
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k ct_mhd_hlld_flux_k_kernel!(Fz, Fv_z, Fz, rho_sum_z, Areak, nxk, nyk, nzk, nxp, nyp, nzp, ch_glm_current, Int32(0), pos_meta)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fz, "Fz after split characteristic HLLD", block.id, world_rank, tt)
        @static if ct_emf_scheme == CT_EMF_WENO7_SG07
            @gpu_launch threads=threads_recon_k blocks=nb_weno_k ct_mhd_characteristic_weno7_cache_k_kernel!(Q, cache_k, Areak, nxk, nyk, nzk, block.Bz_face, Vol, nxp, nyp, nzp, ch_glm_current, dt, Int32(0))
        end
    elseif eigen_reconstruction
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i Eigen_reconstruct_i(Q, U, ϕ, Areai, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(0))
        @check_nan(Fx, "Fx after Eigen_reconstruct_i", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j Eigen_reconstruct_j(Q, U, ϕ, Areaj, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(0))
        @check_nan(Fy, "Fy after Eigen_reconstruct_j", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k Eigen_reconstruct_k(Q, U, ϕ, Areak, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(0))
        @check_nan(Fz, "Fz after Eigen_reconstruct_k", block.id, world_rank, tt)
    else
        _bx_ct = ct_mode ? block.Bx_face : block.ϕ
        _by_ct = ct_mode ? block.By_face : block.ϕ
        _bz_ct = ct_mode ? block.Bz_face : block.ϕ
        @gpu_launch threads=threads_recon_i blocks=nb_conser_i Conser_reconstruct_i(Q, U, ϕ, Areai, Fx, rho_sum_x, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(0), _bx_ct, cache_i, Vol, dt, rk_stage, pos_meta, pos_values)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fx, "Fx after Conser_reconstruct_i", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_j blocks=nb_conser_j Conser_reconstruct_j(Q, U, ϕ, Areaj, Fy, rho_sum_y, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(0), _by_ct, cache_j, Vol, dt, rk_stage, pos_meta, pos_values)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fy, "Fy after Conser_reconstruct_j", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_k blocks=nb_conser_k Conser_reconstruct_k(Q, U, ϕ, Areak, Fz, rho_sum_z, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(0), _bz_ct, cache_k, Vol, dt, rk_stage, pos_meta, pos_values)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fz, "Fz after Conser_reconstruct_k", block.id, world_rank, tt)
    end
    @static if ct_emf_scheme == CT_EMF_WENO7_SG07
        @gpu_launch threads=threads_recon_i blocks=nb_weno_scan_i ct_weno7_validate_finite_kernel!(weno7_fail_meta, weno7_fail_value, cache_i, Int32(1), Int32(1), Int32(nxp+1), Int32(nyp+2NG), Int32(nzp+2NG), Int32(4))
        @gpu_launch threads=threads_recon_j blocks=nb_weno_scan_j ct_weno7_validate_finite_kernel!(weno7_fail_meta, weno7_fail_value, cache_j, Int32(1), Int32(2), Int32(nxp+2NG), Int32(nyp+1), Int32(nzp+2NG), Int32(4))
        @gpu_launch threads=threads_recon_k blocks=nb_weno_scan_k ct_weno7_validate_finite_kernel!(weno7_fail_meta, weno7_fail_value, cache_k, Int32(1), Int32(3), Int32(nxp+2NG), Int32(nyp+2NG), Int32(nzp+1), Int32(4))
        gpu_sync()
        ct_weno7_check_failure!(weno7_fail_meta, weno7_fail_value; rank=world_rank, block=block.id, rk_stage=Int(rk_stage))
    end

    if viscous || (equation_type == :MHD && resistive)
        # Edge ghost cells are already filled by two-pass exchange in sync_blocks!
        @gpu_launch threads=threads_visc_i blocks=nb_visc_i viscous_flux_i(Q, Fv_x, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, Vol, nxp, nyp, nzp, block.is_interblock, false)
        @check_nan(Fv_x, "Fv_x after viscous_flux_i", block.id, world_rank, tt)

        @gpu_launch threads=threads_visc_j blocks=nb_visc_j viscous_flux_j(Q, Fv_y, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, Vol, nxp, nyp, nzp, block.is_interblock, false)
        @check_nan(Fv_y, "Fv_y after viscous_flux_j", block.id, world_rank, tt)

        @gpu_launch threads=threads_visc_k blocks=nb_visc_k viscous_flux_k(Q, Fv_z, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, Vol, nxp, nyp, nzp, block.is_interblock, false)
        @check_nan(Fv_z, "Fv_z after viscous_flux_k", block.id, world_rank, tt)
    end
end

# ── Phase 2: Interior-only blockAdvance on compute_stream ──
# Launches reconstruction with mode=1 (interior only, no ghost dependency)
# on a separate HIP stream. These kernels execute during MPI communication.
function blockAdvance_interior(block::Block, dt, ϕ, Fx, Fy, Fz,
                               rho_sum_x, rho_sum_y, rho_sum_z,
                               Fv_x, Fv_y, Fv_z,
                               threads_recon_i, threads_recon_j, threads_recon_k,
                               threads_visc_i, threads_visc_j, threads_visc_k, stream,
                               rk_stage::Int32, pos_meta, pos_values)
    Q = block.Q; U = block.U
    Vol = block.Vol
    cache_i = _ct_weno7_cache_i_ref[]
    cache_j = _ct_weno7_cache_j_ref[]
    cache_k = _ct_weno7_cache_k_ref[]
    weno7_fail_meta = _ct_weno7_fail_meta_ref[]
    weno7_fail_value = _ct_weno7_fail_value_ref[]
    Areai, nxi, nyi, nzi = block.Areai, block.nxi, block.nyi, block.nzi
    Areaj, nxj, nyj, nzj = block.Areaj, block.nxj, block.nyj, block.nzj
    Areak, nxk, nyk, nzk = block.Areak, block.nxk, block.nyk, block.nzk
    nxp, nyp, nzp = block.Nx, block.Ny, block.Nz
    nb_recon_i = (Int32(cld(nxp+2*NG, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
    nb_recon_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+2*NG, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
    nb_recon_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+2*NG, threads_recon_k[3])))
    @static if ct_emf_scheme == CT_EMF_WENO7_SG07
        nb_weno_i = (Int32(cld(nxp+NG, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
        nb_weno_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+NG, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
        nb_weno_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+NG, threads_recon_k[3])))
        nb_conser_i, nb_conser_j, nb_conser_k = nb_weno_i, nb_weno_j, nb_weno_k
    else
        nb_conser_i, nb_conser_j, nb_conser_k = nb_recon_i, nb_recon_j, nb_recon_k
    end
    nb_visc_i  = (Int32(cld(nxp+2*NG, threads_visc_i[1])),  Int32(cld(nyp+2*NG, threads_visc_i[2])),  Int32(cld(nzp+2*NG, threads_visc_i[3])))
    nb_visc_j  = (Int32(cld(nxp+2*NG, threads_visc_j[1])),  Int32(cld(nyp+2*NG, threads_visc_j[2])),  Int32(cld(nzp+2*NG, threads_visc_j[3])))
    nb_visc_k  = (Int32(cld(nxp+2*NG, threads_visc_k[1])),  Int32(cld(nyp+2*NG, threads_visc_k[2])),  Int32(cld(nzp+2*NG, threads_visc_k[3])))

    si = block.stencil_i; Δsi = block.Δstencil_i; lpi = block.lin_phi_i
    sj = block.stencil_j; Δsj = block.Δstencil_j; lpj = block.lin_phi_j
    sk = block.stencil_k; Δsk = block.Δstencil_k; lpk = block.lin_phi_k
    sRi = block.stencil_R_i; ΔsRi = block.Δstencil_R_i
    sRj = block.stencil_R_j; ΔsRj = block.Δstencil_R_j
    sRk = block.stencil_R_k; ΔsRk = block.Δstencil_R_k
    if eigen_reconstruction && !(equation_type == :MHD && ct_mode)
        @gpu_launch_stream stream threads=threads_recon_i blocks=nb_recon_i Eigen_reconstruct_i(Q, U, ϕ, Areai, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(1))
        @gpu_launch_stream stream threads=threads_recon_j blocks=nb_recon_j Eigen_reconstruct_j(Q, U, ϕ, Areaj, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(1))
        @gpu_launch_stream stream threads=threads_recon_k blocks=nb_recon_k Eigen_reconstruct_k(Q, U, ϕ, Areak, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(1))
    else
        _bx_ct = ct_mode ? block.Bx_face : block.ϕ
        _by_ct = ct_mode ? block.By_face : block.ϕ
        _bz_ct = ct_mode ? block.Bz_face : block.ϕ
        @static if ct_emf_scheme == CT_EMF_WENO7_SG07
            ct_weno7_reset_failure!(weno7_fail_meta, weno7_fail_value)
        end
        @gpu_launch_stream stream threads=threads_recon_i blocks=nb_conser_i Conser_reconstruct_i(Q, U, ϕ, Areai, Fx, rho_sum_x, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(1), _bx_ct, cache_i, Vol, dt, rk_stage, pos_meta, pos_values)
        @gpu_launch_stream stream threads=threads_recon_j blocks=nb_conser_j Conser_reconstruct_j(Q, U, ϕ, Areaj, Fy, rho_sum_y, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(1), _by_ct, cache_j, Vol, dt, rk_stage, pos_meta, pos_values)
        @gpu_launch_stream stream threads=threads_recon_k blocks=nb_conser_k Conser_reconstruct_k(Q, U, ϕ, Areak, Fz, rho_sum_z, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(1), _bz_ct, cache_k, Vol, dt, rk_stage, pos_meta, pos_values)
    end
    # Viscous flux interior (viscous stencil >= 2 cells, fully covered by mode=1 range)
    if viscous || (equation_type == :MHD && resistive)
        @gpu_launch_stream stream threads=threads_visc_i blocks=nb_visc_i viscous_flux_i(Q, Fv_x, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, block.Vol, nxp, nyp, nzp, block.is_interblock, false)
        @gpu_launch_stream stream threads=threads_visc_j blocks=nb_visc_j viscous_flux_j(Q, Fv_y, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, block.Vol, nxp, nyp, nzp, block.is_interblock, false)
        @gpu_launch_stream stream threads=threads_visc_k blocks=nb_visc_k viscous_flux_k(Q, Fv_z, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, block.Vol, nxp, nyp, nzp, block.is_interblock, false)
    end
end

# ── Phase 2: Boundary-only blockAdvance on default stream ──
# Launches reconstruction with mode=2 (boundary only, needs ghost cells)
# Called AFTER sync_blocks! ensures ghost cells are valid.
function blockAdvance_boundary(block::Block, dt, ϕ, Fx, Fy, Fz,
                               rho_sum_x, rho_sum_y, rho_sum_z,
                               Fv_x, Fv_y, Fv_z,
                               world_rank, tt,
                               threads_recon_i, threads_recon_j, threads_recon_k,
                               rk_stage::Int32, pos_meta, pos_values)
    Q = block.Q; U = block.U
    Vol = block.Vol
    cache_i = _ct_weno7_cache_i_ref[]
    cache_j = _ct_weno7_cache_j_ref[]
    cache_k = _ct_weno7_cache_k_ref[]
    weno7_fail_meta = _ct_weno7_fail_meta_ref[]
    weno7_fail_value = _ct_weno7_fail_value_ref[]
    Areai, nxi, nyi, nzi = block.Areai, block.nxi, block.nyi, block.nzi
    Areaj, nxj, nyj, nzj = block.Areaj, block.nxj, block.nyj, block.nzj
    Areak, nxk, nyk, nzk = block.Areak, block.nxk, block.nyk, block.nzk
    nxp, nyp, nzp = block.Nx, block.Ny, block.Nz
    nb_recon_i = (Int32(cld(nxp+2*NG, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
    nb_recon_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+2*NG, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
    nb_recon_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+2*NG, threads_recon_k[3])))
    @static if ct_emf_scheme == CT_EMF_WENO7_SG07
        nb_weno_i = (Int32(cld(nxp+NG, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
        nb_weno_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+NG, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
        nb_weno_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+NG, threads_recon_k[3])))
        nb_weno_scan_i = (Int32(cld(nxp+1, threads_recon_i[1])), Int32(cld(nyp+2*NG, threads_recon_i[2])), Int32(cld(nzp+2*NG, threads_recon_i[3])))
        nb_weno_scan_j = (Int32(cld(nxp+2*NG, threads_recon_j[1])), Int32(cld(nyp+1, threads_recon_j[2])), Int32(cld(nzp+2*NG, threads_recon_j[3])))
        nb_weno_scan_k = (Int32(cld(nxp+2*NG, threads_recon_k[1])), Int32(cld(nyp+2*NG, threads_recon_k[2])), Int32(cld(nzp+1, threads_recon_k[3])))
        nb_conser_i, nb_conser_j, nb_conser_k = nb_weno_i, nb_weno_j, nb_weno_k
    else
        nb_conser_i, nb_conser_j, nb_conser_k = nb_recon_i, nb_recon_j, nb_recon_k
    end

    si = block.stencil_i; Δsi = block.Δstencil_i; lpi = block.lin_phi_i
    sj = block.stencil_j; Δsj = block.Δstencil_j; lpj = block.lin_phi_j
    sk = block.stencil_k; Δsk = block.Δstencil_k; lpk = block.lin_phi_k
    sRi = block.stencil_R_i; ΔsRi = block.Δstencil_R_i
    sRj = block.stencil_R_j; ΔsRj = block.Δstencil_R_j
    sRk = block.stencil_R_k; ΔsRk = block.Δstencil_R_k
    if eigen_reconstruction && !(equation_type == :MHD && ct_mode)
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i Eigen_reconstruct_i(Q, U, ϕ, Areai, Fx, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(2))
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j Eigen_reconstruct_j(Q, U, ϕ, Areaj, Fy, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(2))
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k Eigen_reconstruct_k(Q, U, ϕ, Areak, Fz, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(2))
    else
        _bx_ct = ct_mode ? block.Bx_face : block.ϕ
        _by_ct = ct_mode ? block.By_face : block.ϕ
        _bz_ct = ct_mode ? block.Bz_face : block.ϕ
        @gpu_launch threads=threads_recon_i blocks=nb_conser_i Conser_reconstruct_i(Q, U, ϕ, Areai, Fx, rho_sum_x, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(2), _bx_ct, cache_i, Vol, dt, rk_stage, pos_meta, pos_values)
        @gpu_launch threads=threads_recon_j blocks=nb_conser_j Conser_reconstruct_j(Q, U, ϕ, Areaj, Fy, rho_sum_y, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(2), _by_ct, cache_j, Vol, dt, rk_stage, pos_meta, pos_values)
        @gpu_launch threads=threads_recon_k blocks=nb_conser_k Conser_reconstruct_k(Q, U, ϕ, Areak, Fz, rho_sum_z, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(2), _bz_ct, cache_k, Vol, dt, rk_stage, pos_meta, pos_values)
        @static if ct_emf_scheme == CT_EMF_WENO7_SG07
            @gpu_launch threads=threads_recon_i blocks=nb_weno_scan_i ct_weno7_validate_finite_kernel!(weno7_fail_meta, weno7_fail_value, cache_i, Int32(1), Int32(1), Int32(nxp+1), Int32(nyp+2NG), Int32(nzp+2NG), Int32(4))
            @gpu_launch threads=threads_recon_j blocks=nb_weno_scan_j ct_weno7_validate_finite_kernel!(weno7_fail_meta, weno7_fail_value, cache_j, Int32(1), Int32(2), Int32(nxp+2NG), Int32(nyp+1), Int32(nzp+2NG), Int32(4))
            @gpu_launch threads=threads_recon_k blocks=nb_weno_scan_k ct_weno7_validate_finite_kernel!(weno7_fail_meta, weno7_fail_value, cache_k, Int32(1), Int32(3), Int32(nxp+2NG), Int32(nyp+2NG), Int32(nzp+1), Int32(4))
            gpu_sync()
            ct_weno7_check_failure!(weno7_fail_meta, weno7_fail_value; rank=world_rank, block=block.id, rk_stage=Int(rk_stage))
        end
    end
    # Note: viscous_flux runs on ALL cells here (no mode param) since interior
    # viscous was already computed on compute_stream and is now complete
    # (copyto! in sync_blocks forced device sync). Boundary viscous overwrites
    # boundary faces; interior faces remain from the compute_stream pass.
end


function time_step(world_rank, comm_cart, Block_Nprocs)
    if equation_type == :MHD && ct_mode && resistive && NG < 4
        error(
            "Resistive CT requires NG>=4 so the tangential diffusive-flux " *
            "halo can use the existing centered derivative stencil",
        )
    end
    if equation_type == :MHD && ct_mode && eigen_reconstruction &&
       ct_characteristic_reconstruction == CT_CHARACTERISTIC_WENO7 && NG < 4
        error("CT-MHD characteristic WENO7 requires NG>=4, got NG=$NG")
    end
    if equation_type == :MHD && ct_mode && eigen_reconstruction &&
       !(strict_ct_positivity && splitMethodID == Int32(4))
        error(
            "CT-MHD characteristic reconstruction requires " *
            "strict_ct_positivity=true and splitMethodID=4 (HLLD)",
        )
    end
    if equation_type == :MHD && ct_mode && LTS
        error(
            "CT-MHD does not support LTS: cell U uses per-cell dt while the " *
            "discrete-Stokes face-B update uses one stage dt",
        )
    end
    if equation_type == :MHD && ct_mode && implicit
        error(
            "CT-MHD does not support implicit advancement: the implicit " *
            "residual does not advance face-centered magnetic flux",
        )
    end
    if ct_resistive_integrator == CT_RESISTIVE_STS &&
       !(equation_type == :MHD && ct_mode && resistive)
        error(
            "ct_resistive_integrator=:sts requires resistive CT-MHD",
        )
    end
    if ct_resistive_integrator == CT_RESISTIVE_RKL2_STRANG &&
       !(equation_type == :MHD && ct_mode && resistive)
        error(
            "ct_resistive_integrator=:rkl2_strang requires resistive CT-MHD",
        )
    end

    # Dicts to temporarily hold metrics on CPU for syncing
    temp_metrics_h = Dict{Int, Any}()
    was_computed_h = Dict{Int, Bool}()
    temp_metrics_pre_h = Dict{Int, Any}()

    # Load multi-block metadata
    if world_rank == 0
        println(">>> Loading multi-block connectivity...")
    end
    # connectivity_file is defined in the run config (e.g., run_pipe.jl, run_cavity.jl)
    _mesh_base_conn = isdefined(Main, :mesh_dir) ? mesh_dir : "MESH"
    conn_path = isdefined(Main, :connectivity_file) ? connectivity_file : joinpath(_mesh_base_conn, "block_connectivity.h5")
    Nblocks, connectivity, face_bc, bc_params, Nx_b, Ny_b, Nz_b = load_multiblock_connectivity(conn_path; world_rank=world_rank)
    
    # ─── Dynamic CEBL Connectivity and BC Extension ───
    if isdefined(Main, :cebl_forcing) && Main.cebl_forcing
        # 1. Duplicate y/z connectivity with +5 block offset
        conn_keys = collect(keys(connectivity))
        for (b1, f1) in conn_keys
            if b1 in 0:4
                conn = connectivity[(b1, f1)]
                if conn.src_b in 0:4
                    connectivity[(b1 + 5, f1)] = Connectivity(conn.src_b + 5, conn.src_f, conn.reverse_tan, conn.flip_normal)
                end
            end
        end
        # 2. Duplicate face BCs and parameters
        for bid in 0:4, fid_idx in 1:6
            if haskey(face_bc, (bid, fid_idx))
                face_bc[(bid + 5, fid_idx)] = face_bc[(bid, fid_idx)]
            end
            if haskey(bc_params, (bid, fid_idx))
                bc_params[(bid + 5, fid_idx)] = bc_params[(bid, fid_idx)]
            end
        end
        # 3. Override precursor x boundaries to periodic
        for bid in 5:9
            face_bc[(bid, 1)] = BC_PERIODIC
            face_bc[(bid, 2)] = BC_PERIODIC
        end
        # 4. Override main domain inlet boundary to CEBL mapped inflow
        for bid in 0:4
            face_bc[(bid, 1)] = BC_CEBL_INFLOW
        end
        # 4b. Override main domain outlet to a non-reflecting outflow when fringe is disabled.
        #     Default = ZERO_GRADIENT (robust at wall corners). Switch to NSCBC by setting
        #     Main.outlet_bc = :nscbc in the run script (requires careful p_target tuning).
        if !_fringe_active
            _outlet_bc_sym = isdefined(Main, :outlet_bc) ? Main.outlet_bc : :zero_gradient
            _outlet_bc_id = _outlet_bc_sym == :nscbc ? BC_NSCBC_OUTFLOW :
                              (_outlet_bc_sym == :riemann ? BC_RIEMANN_OUTFLOW : BC_ZERO_GRADIENT)
            for bid in 0:4
                face_bc[(bid, 2)] = _outlet_bc_id
                # Inject fixed p_back into BCP_OUTLET_PAVG (slot 15) if defined.
                # This anchors NSCBC L1 to a fixed back pressure instead of the
                # self-tracking outlet mean, preventing zero-frequency drift.
                if isdefined(Main, :p_back) && Main.p_back > 0 && haskey(bc_params, (bid, 2))
                    bcp = bc_params[(bid, 2)]
                    bcp[BCP_OUTLET_PAVG] = Main.p_back
                    bc_params[(bid, 2)] = bcp
                end
            end
        end
        # 5. Extend grid size arrays
        for b in 1:5
            push!(Nx_b, Main.cebl_Nx)
            push!(Ny_b, Ny_b[b])
            push!(Nz_b, Nz_b[b])
        end
    end
    
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

    if ct_emf_scheme == CT_EMF_WENO7_SG07
        ct_validate_weno7_configuration(NG, connectivity, face_bc)
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
                
                # Setup rank_offsets BEFORE load_block for metric sync
                _temp_rank_offsets = zeros(Int, length(Block_Nprocs) + 1)
                for i in 1:length(Block_Nprocs)
                    _temp_rank_offsets[i] = Block_to_rank[i]
                end
                _temp_rank_offsets[length(Block_Nprocs) + 1] = Block_to_rank[end] + 1
                
                blocks[bid] = load_block(bid, 0, 0, 0, NG, Ncons, Nprim, Nprocs_block, world_rank, face_bc, connectivity, _temp_rank_offsets, temp_metrics_h, was_computed_h)
                # No sub-domain splitting →use COMM_SELF for intra-block exchange
                block_comms[bid] = MPI.Cart_create(MPI.COMM_SELF, [1,1,1]; periodic=collect(Iperiodic))
            end
        end
        # Set rank_offsets so copy_ghost_face! maps block →owning rank correctly
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

        blocks[my_block_id] = load_block(my_block_id, rankx, ranky, rankz, NG, Ncons, Nprim, Nprocs_my_block, world_rank, face_bc, connectivity, rank_offsets, temp_metrics_h, was_computed_h)

        block_comm = MPI.Comm_split(MPI.COMM_WORLD, my_block_id, local_rank)
        block_comms[my_block_id] = MPI.Cart_create(block_comm, collect(Nprocs_my_block); periodic=collect(Iperiodic))

        _rank_offsets_setup = copy(rank_offsets)
    end

    # ─── Dynamically generate CEBL precursor blocks (Blocks 5–9) ───
    if isdefined(Main, :cebl_forcing) && Main.cebl_forcing
        # Extend partitioning arrays first
        for b in 1:5
            push!(Block_Nprocs, SVector{3, Int}(1, Block_Nprocs[b][2], Block_Nprocs[b][3]))
            push!(Block_to_rank, Block_to_rank[b])
        end
        if !multi_block_mode
            resize!(rank_offsets, 11)
            end_bound = rank_offsets[6]
            for i in 5:9
                rank_offsets[i+1] = rank_offsets[i-4]
            end
            rank_offsets[11] = end_bound
            _rank_offsets_setup = copy(rank_offsets)
        else
            resize!(rank_offsets, 11)
            for i in 1:10
                rank_offsets[i] = Block_to_rank[i]
            end
            rank_offsets[11] = Block_to_rank[10] + 1
            _rank_offsets_setup = copy(rank_offsets)
        end

        # Populate connectivity for all precursor blocks on all ranks deterministically
        for bid in 0:4
            for fid in 3:6
                if haskey(face_bc, (bid, fid))
                    if face_bc[(bid, fid)] == 0 && haskey(connectivity, (bid, fid))
                        conn = connectivity[(bid, fid)]
                        connectivity[(bid+5, fid)] = Connectivity(conn.src_b + 5, conn.src_f, conn.reverse_tan, conn.flip_normal)
                    end
                end
            end
        end

        # ─── Pre-create all 5 precursor block communicators upfront ────
        # MPI.Comm_split is a COMM_WORLD-collective: EVERY rank must call
        # for each bid simultaneously with the same args layout. Doing this
        # OUTSIDE the per-rank work loop ensures all ranks march in lockstep
        # through the 5 collective calls before any rank starts the heavy
        # per-block init work (whose timing varies per rank).
        for bid in 0:4
            is_inlet_rank = false
            if haskey(blocks, bid)
                Nprocs_block = Block_Nprocs[bid + 1]
                local_rank = world_rank - rank_offsets[bid + 1]
                rx = local_rank ÷ (Nprocs_block[2] * Nprocs_block[3])
                is_inlet_rank = (rx == 0)
            end
            comm_color_pre = is_inlet_rank ? bid + 5 : MPI.API.MPI_UNDEFINED[]
            comm_split_pre = MPI.Comm_split(MPI.COMM_WORLD,
                Cint(comm_color_pre), Cint(world_rank))
            if comm_split_pre != MPI.COMM_NULL
                block_comms[bid + 5] = MPI.Cart_create(comm_split_pre,
                    collect(Block_Nprocs[bid + 5 + 1]); periodic=(true, false, false))
            end
        end

        for bid in 0:4
            if haskey(blocks, bid)
                Nprocs_main = Block_Nprocs[bid + 1]
                local_rank = world_rank - rank_offsets[bid + 1]
                rx = local_rank ÷ (Nprocs_main[2] * Nprocs_main[3])
                
                if rx == 0
                    b = blocks[bid] # corresponding main block
                    Nprocs_block = Block_Nprocs[bid + 5 + 1]
                    
                    rx_pre = 0
                    ry_pre = (local_rank ÷ Nprocs_main[3]) % Nprocs_main[2]
                    rz_pre = local_rank % Nprocs_main[3]
                    
                    nxp_pre = Main.cebl_Nx
                    ox_pre = 0
                    
                    nxp, nyp, nzp = b.Nx, b.Ny, b.Nz
                
                x_pre_h = zeros(FT, nxp_pre + 2*NG + 1, nyp + 2*NG + 1, nzp + 2*NG + 1)
                y_pre_h = zeros(FT, nxp_pre + 2*NG + 1, nyp + 2*NG + 1, nzp + 2*NG + 1)
                z_pre_h = zeros(FT, nxp_pre + 2*NG + 1, nyp + 2*NG + 1, nzp + 2*NG + 1)
                
                y_inlet = Array(b.y)[NG + 1, :, :]
                z_inlet = Array(b.z)[NG + 1, :, :]
                dx_pre = Main.cebl_Lx / Main.cebl_Nx
                for i in 1:(nxp_pre + 2*NG + 1)
                    x_val = -Main.cebl_Lx + (ox_pre + i - NG - 1) * dx_pre
                    x_pre_h[i, :, :] .= FT(x_val)
                    y_pre_h[i, :, :] .= y_inlet
                    z_pre_h[i, :, :] .= z_inlet
                end

                cache_path_pre, was_computed_pre, Areai_pre_h, nxi_pre_h, nyi_pre_h, nzi_pre_h, Areaj_pre_h, nxj_pre_h, nyj_pre_h, nzj_pre_h, Areak_pre_h, nxk_pre_h, nyk_pre_h, nzk_pre_h, Vol_pre_h = load_or_compute_metrics(
                    bid + 5, rx_pre, ry_pre, rz_pre, x_pre_h, y_pre_h, z_pre_h, nxp_pre, nyp, nzp, NG; cache_metrics=false)
                
                temp_metrics_pre_h[bid + 5] = (cache_path_pre, Areai_pre_h, nxi_pre_h, nyi_pre_h, nzi_pre_h, Areaj_pre_h, nxj_pre_h, nyj_pre_h, nzj_pre_h, Areak_pre_h, nxk_pre_h, nyk_pre_h, nzk_pre_h, Vol_pre_h)
                
                Nx_tot_pre = nxp_pre + 2*NG
                Ny_tot = nyp + 2*NG
                Nz_tot = nzp + 2*NG
                
                Areai_pre = GPUArray(Areai_pre_h); nxi_pre = GPUArray(nxi_pre_h); nyi_pre = GPUArray(nyi_pre_h); nzi_pre = GPUArray(nzi_pre_h)
                Areaj_pre = GPUArray(Areaj_pre_h); nxj_pre = GPUArray(nxj_pre_h); nyj_pre = GPUArray(nyj_pre_h); nzj_pre = GPUArray(nzj_pre_h)
                Areak_pre = GPUArray(Areak_pre_h); nxk_pre = GPUArray(nxk_pre_h); nyk_pre = GPUArray(nyk_pre_h); nzk_pre = GPUArray(nzk_pre_h)
                Vol_pre   = GPUArray(Vol_pre_h)
                
                Q_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot, Nprim)
                U_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot, Ncons)
                ϕ_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot)
                
                x_pre = GPUArray(x_pre_h)
                y_pre = GPUArray(y_pre_h)
                z_pre = GPUArray(z_pre_h)
                
                initialize(Q_pre, x_pre, y_pre, z_pre, rx_pre, ry_pre, Nprocs_block, nxp_pre, nyp, nzp, bid + 5)
                
                nb_pre = (cld(Nx_tot_pre, nthreads[1]), cld(Ny_tot, nthreads[2]), cld(Nz_tot, nthreads[3]))
                @gpu_launch threads=nthreads blocks=nb_pre prim2c(U_pre, Q_pre, nxp_pre, nyp, nzp)
                
                LTS_dt_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot)
                Un_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot, Nprim)
                
                sbuf_hx_pre = zeros(FT, NG, nyp+2*NG, nzp+2*NG, Nprim)
                rbuf_hx_pre = zeros(FT, NG, nyp+2*NG, nzp+2*NG, Nprim)
                sbuf_dx_pre = GPUArray(sbuf_hx_pre); rbuf_dx_pre = GPUArray(rbuf_hx_pre)
                
                sbuf_hx2_pre = zeros(FT, NG, nyp+2*NG, nzp+2*NG, Nprim)
                rbuf_hx2_pre = zeros(FT, NG, nyp+2*NG, nzp+2*NG, Nprim)
                sbuf_dx2_pre = GPUArray(sbuf_hx2_pre); rbuf_dx2_pre = GPUArray(rbuf_hx2_pre)
                
                sbuf_hy_pre = zeros(FT, nxp_pre+2*NG, NG, nzp+2*NG, Nprim)
                rbuf_hy_pre = zeros(FT, nxp_pre+2*NG, NG, nzp+2*NG, Nprim)
                sbuf_dy_pre = GPUArray(sbuf_hy_pre); rbuf_dy_pre = GPUArray(rbuf_hy_pre)
                
                sbuf_hz_pre = zeros(FT, nxp_pre+2*NG, nyp+2*NG, NG, Nprim)
                rbuf_hz_pre = zeros(FT, nxp_pre+2*NG, nyp+2*NG, NG, Nprim)
                sbuf_dz_pre = GPUArray(sbuf_hz_pre); rbuf_dz_pre = GPUArray(rbuf_hz_pre)
                
                _Lin_h_pre = repeat(reshape(FT.(Linear), 1, 7), Nx_tot_pre, 1)
                _ΔLin_h_pre = repeat(reshape(FT.(ΔLinear), 1, 7), Nx_tot_pre, 1)
                _phi_h_pre = fill(FT(Linear_ϕ), Nx_tot_pre)
                stencil_i_pre = GPUArray(_Lin_h_pre); Δstencil_i_pre = GPUArray(_ΔLin_h_pre); lin_phi_i_pre = GPUArray(_phi_h_pre)
                stencil_R_i_pre = GPUArray(_Lin_h_pre); Δstencil_R_i_pre = GPUArray(_ΔLin_h_pre)
                
                # ── η/ζ stencil with interface protection (same as main blocks) ──
                Nj_pre = nyp + 2*NG; Nk_pre = nzp + 2*NG
                _Lin_j_pre  = repeat(reshape(FT.(Linear),  1, 1, 7), Nj_pre, Nk_pre, 1)
                _ΔLin_j_pre = repeat(reshape(FT.(ΔLinear), 1, 1, 7), Nj_pre, Nk_pre, 1)
                _phi_j_pre  = fill(FT(Linear_ϕ), Nj_pre, Nk_pre)
                _Lin_k_pre  = repeat(reshape(FT.(Linear),  1, 1, 7), Nj_pre, Nk_pre, 1)
                _ΔLin_k_pre = repeat(reshape(FT.(ΔLinear), 1, 1, 7), Nj_pre, Nk_pre, 1)
                _phi_k_pre  = fill(FT(Linear_ϕ), Nj_pre, Nk_pre)
                apply_interface_taper!(_Lin_j_pre, _ΔLin_j_pre, _phi_j_pre,
                                       _Lin_k_pre, _ΔLin_k_pre, _phi_k_pre,
                                       face_bc, bid + 5, nyp, nzp, NG)
                apply_geometric_smoothness_protection!(
                    _Lin_j_pre, _ΔLin_j_pre, _phi_j_pre,
                    _Lin_k_pre, _ΔLin_k_pre, _phi_k_pre,
                    Areaj_pre_h, nxj_pre_h, nyj_pre_h, nzj_pre_h,
                    Areak_pre_h, nxk_pre_h, nyk_pre_h, nzk_pre_h,
                    face_bc, bid + 5, nxp_pre, nyp, nzp, NG;
                    verbose=true
                )
                global _SING_INFO
                if isdefined(Main, :_SING_INFO) && _SING_INFO !== nothing
                    apply_topology_smoothness_protection!(
                        _Lin_j_pre, _ΔLin_j_pre, _phi_j_pre,
                        _Lin_k_pre, _ΔLin_k_pre, _phi_k_pre,
                        bid + 1, nxp_pre, nyp, nzp, NG, _SING_INFO;
                        verbose=true
                    )
                end
                stencil_j_pre = GPUArray(_Lin_j_pre); Δstencil_j_pre = GPUArray(_ΔLin_j_pre); lin_phi_j_pre = GPUArray(_phi_j_pre)
                stencil_k_pre = GPUArray(_Lin_k_pre); Δstencil_k_pre = GPUArray(_ΔLin_k_pre); lin_phi_k_pre = GPUArray(_phi_k_pre)
                stencil_R_j_pre = GPUArray(_Lin_j_pre); Δstencil_R_j_pre = GPUArray(_ΔLin_j_pre)
                stencil_R_k_pre = GPUArray(_Lin_k_pre); Δstencil_R_k_pre = GPUArray(_ΔLin_k_pre)
                
                filter_σ_pre = b.filter_σ
                is_interblock_pre = b.is_interblock
                
                blocks[bid + 5] = Block(
                    bid + 5, nxp_pre, nyp, nzp, rx_pre, ry_pre, rz_pre, ox_pre, b.oy, b.oz,
                    Q_pre, U_pre, ϕ_pre, Areai_pre, Areaj_pre, Areak_pre,
                    nxi_pre, nyi_pre, nzi_pre, nxj_pre, nyj_pre, nzj_pre, nxk_pre, nyk_pre, nzk_pre, Vol_pre,
                    x_pre, y_pre, z_pre, LTS_dt_pre, Un_pre,
                    # CT face-B (precursor blocks — use nothing for now, CT not supported in CEBL)
                    nothing, nothing, nothing, nothing, nothing, nothing,
                    nothing, nothing, nothing,
                    sbuf_hx_pre, sbuf_dx_pre, rbuf_hx_pre, rbuf_dx_pre,
                    sbuf_hx2_pre, sbuf_dx2_pre, rbuf_hx2_pre, rbuf_dx2_pre,
                    sbuf_hy_pre, sbuf_dy_pre, rbuf_hy_pre, rbuf_dy_pre,
                    sbuf_hz_pre, sbuf_dz_pre, rbuf_hz_pre, rbuf_dz_pre,
                    nb_pre, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
                    stencil_i_pre, Δstencil_i_pre, lin_phi_i_pre,
                    stencil_j_pre, Δstencil_j_pre, lin_phi_j_pre,
                    stencil_k_pre, Δstencil_k_pre, lin_phi_k_pre,
                    stencil_R_i_pre, Δstencil_R_i_pre,
                    stencil_R_j_pre, Δstencil_R_j_pre,
                    stencil_R_k_pre, Δstencil_R_k_pre,
                    filter_σ_pre, nothing, is_interblock_pre, nothing, nothing, nothing, nothing
                )
                
                # block_comms[bid + 5] already created above (collective).
                # SKIP write_precursor_mesh: it's a parallel HDF5 collective
                # that hung the run; the precursor mesh files (mesh_b5..b9.h5)
                # are pre-generated by Utils/gen_butterfly_cebl_len15*.jl
                # (or auto-cached from a previous successful run) and don't
                # need to be re-written every run.
                # if haskey(blocks, bid + 5)
                #     write_precursor_mesh(bid + 5, x_pre_h, y_pre_h, z_pre_h, nxp_pre, nyp, nzp, NG, ox_pre, b.oy, b.oz, block_comms[bid + 5])
                # end
                end
            end
        end
    end


    # ─── Synchronize metrics for all blocks safely ───
    if length(temp_metrics_h) > 0
        println("Rank $world_rank: Synchronizing metrics for all main blocks...")
        
        first_bid = collect(keys(temp_metrics_h))[1]
        b0 = blocks[first_bid]
        nxp, nyp, nzp = b0.Nx, b0.Ny, b0.Nz
        
        sync_dict = Dict{Int, Any}()
        for (bid, val) in temp_metrics_h
            sync_dict[bid] = (val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13])
        end

        debug_metric_closure = lowercase(get(ENV, "CT_DEBUG_METRIC_CLOSURE", "false")) in
            ("true", "1", "yes", "on")
        if ct_mode && debug_metric_closure
            closure_l2, closure_max = ct_metric_closure_stats(temp_metrics_h, blocks)
            @printf("  CT metric closure before sync: L2=%.6e Linf=%.6e\n", closure_l2, closure_max)
        end
        if ct_mode
            ct_sync_rank_metrics!(
                temp_metrics_h, blocks, block_comms, Block_Nprocs,
            )
        end
        _tmp_Block_Nprocs[] = Block_Nprocs
        sync_all_interface_metrics!(collect(keys(temp_metrics_h)), sync_dict, 0, 0, 0, face_bc, connectivity, _rank_offsets_setup, Nx_b, Ny_b, Nz_b, NG)
        if ct_mode && debug_metric_closure
            closure_l2, closure_max = ct_metric_closure_stats(temp_metrics_h, blocks)
            @printf("  CT metric closure after sync:  L2=%.6e Linf=%.6e\n", closure_l2, closure_max)
        end
        
        for (bid, val) in temp_metrics_h
            cache_path = val[1]
            Areai_h, nxi_h, nyi_h, nzi_h = sync_dict[bid][1:4]
            Areaj_h, nxj_h, nyj_h, nzj_h = sync_dict[bid][5:8]
            Areak_h, nxk_h, nyk_h, nzk_h = sync_dict[bid][9:12]
            Vol_h = val[14]
            
            if cache_metrics && get(was_computed_h, bid, false)
                println("    Saving synchronized metrics cache to $cache_path")
                save_metrics_to_h5(cache_path, Areai_h, nxi_h, nyi_h, nzi_h, Areaj_h, nxj_h, nyj_h, nzj_h, Areak_h, nxk_h, nyk_h, nzk_h, Vol_h)
            end
            
            copyto!(blocks[bid].Areai, Areai_h); copyto!(blocks[bid].nxi, nxi_h); copyto!(blocks[bid].nyi, nyi_h); copyto!(blocks[bid].nzi, nzi_h)
            copyto!(blocks[bid].Areaj, Areaj_h); copyto!(blocks[bid].nxj, nxj_h); copyto!(blocks[bid].nyj, nyj_h); copyto!(blocks[bid].nzj, nzj_h)
            copyto!(blocks[bid].Areak, Areak_h); copyto!(blocks[bid].nxk, nxk_h); copyto!(blocks[bid].nyk, nyk_h); copyto!(blocks[bid].nzk, nzk_h)
        end
    end

    if isdefined(Main, :cebl_forcing) && Main.cebl_forcing
        if world_rank == 0
            println("Rank 0: Synchronizing metrics for all precursor blocks (collective call)...")
        end
        
        sync_dict_pre = Dict{Int, Any}()
        for (bid, val) in temp_metrics_pre_h
            sync_dict_pre[bid] = (val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13])
        end
        
        face_bc_pre = copy(face_bc)
        for (bid, val) in temp_metrics_pre_h
            face_bc_pre[(bid, 1)] = BC_PERIODIC
            face_bc_pre[(bid, 2)] = BC_PERIODIC
            for fid in 3:6
                if haskey(face_bc, (bid-5, fid))
                    face_bc_pre[(bid, fid)] = face_bc[(bid-5, fid)]
                end
            end
        end

        _rank_offsets_pre = copy(_rank_offsets_setup)
        if !multi_block_mode
            resize!(_rank_offsets_pre, 11)
            end_bound = _rank_offsets_pre[6]
            for i in 5:9
                _rank_offsets_pre[i+1] = _rank_offsets_pre[i-4]
            end
            _rank_offsets_pre[11] = end_bound
        end
        Nx_b_pre = (Main.Nx_b..., ntuple(i -> Main.cebl_Nx, 5)...)
        _tmp_Block_Nprocs[] = Block_Nprocs
        
        # Called by ALL ranks collectively
        sync_all_interface_metrics!(collect(keys(temp_metrics_pre_h)), sync_dict_pre, 0, 0, 0, face_bc_pre, connectivity, _rank_offsets_pre, Nx_b_pre, (Main.Ny_b..., Main.Ny_b...), (Main.Nz_b..., Main.Nz_b...), NG)
        
        for (bid, val) in temp_metrics_pre_h
            Areai_pre_h, nxi_pre_h, nyi_pre_h, nzi_pre_h = sync_dict_pre[bid][1:4]
            Areaj_pre_h, nxj_pre_h, nyj_pre_h, nzj_pre_h = sync_dict_pre[bid][5:8]
            Areak_pre_h, nxk_pre_h, nyk_pre_h, nzk_pre_h = sync_dict_pre[bid][9:12]
            
            copyto!(blocks[bid].Areai, Areai_pre_h); copyto!(blocks[bid].nxi, nxi_pre_h); copyto!(blocks[bid].nyi, nyi_pre_h); copyto!(blocks[bid].nzi, nzi_pre_h)
            copyto!(blocks[bid].Areaj, Areaj_pre_h); copyto!(blocks[bid].nxj, nxj_pre_h); copyto!(blocks[bid].nyj, nyj_pre_h); copyto!(blocks[bid].nzj, nzj_pre_h)
            copyto!(blocks[bid].Areak, Areak_pre_h); copyto!(blocks[bid].nxk, nxk_pre_h); copyto!(blocks[bid].nyk, nyk_pre_h); copyto!(blocks[bid].nzk, nzk_pre_h)
        end
    end



    # Global dimensions for flux buffers (max across all local blocks)
    max_nxp = maximum(b.Nx for (_, b) in blocks)
    max_nyp = maximum(b.Ny for (_, b) in blocks)
    max_nzp = maximum(b.Nz for (_, b) in blocks)

    # Fx/Fy/Fz: CT mode adds 1 ghost layer in tangential directions for edge-EMF
    _g = ct_mode ? 2 : 0  # +2 = 1 ghost on each side
    shared_Fx  = gpu_zeros(FT, max_nxp+1, max_nyp+_g, max_nzp+_g, Ncons)
    shared_Fy  = gpu_zeros(FT, max_nxp+_g, max_nyp+1, max_nzp+_g, Ncons)
    shared_Fz  = gpu_zeros(FT, max_nxp+_g, max_nyp+_g, max_nzp+1, Ncons)
    shared_rho_sum_x = ct_mode ? gpu_zeros(FT, max_nxp+1, max_nyp+_g, max_nzp+_g) : gpu_zeros(FT, 1, 1, 1)
    shared_rho_sum_y = ct_mode ? gpu_zeros(FT, max_nxp+_g, max_nyp+1, max_nzp+_g) : gpu_zeros(FT, 1, 1, 1)
    shared_rho_sum_z = ct_mode ? gpu_zeros(FT, max_nxp+_g, max_nyp+_g, max_nzp+1) : gpu_zeros(FT, 1, 1, 1)
    shared_Fvx = gpu_zeros(FT, max_nxp+1, max_nyp+_g, max_nzp+_g, Ncons)
    shared_Fvy = gpu_zeros(FT, max_nxp+_g, max_nyp+1, max_nzp+_g, Ncons)
    shared_Fvz = gpu_zeros(FT, max_nxp+_g, max_nyp+_g, max_nzp+1, Ncons)
    shared_dU_forced = gpu_zeros(FT, max_nxp, max_nyp, max_nzp, Ncons)
    shared_ct_pos_meta = gpu_zeros(Int32, CT_POS_META_LEN)
    shared_ct_pos_values = gpu_zeros(FT, CT_POS_VALUE_LEN)

    if ct_emf_scheme == CT_EMF_WENO7_SG07
        cache_i = gpu_zeros(FT, max_nxp+1, max_nyp+2NG, max_nzp+2NG, 4)
        cache_j = gpu_zeros(FT, max_nxp+2NG, max_nyp+1, max_nzp+2NG, 4)
        cache_k = gpu_zeros(FT, max_nxp+2NG, max_nyp+2NG, max_nzp+1, 4)
        point_scratch = gpu_zeros(
            FT, max_nxp+2NG+1, max_nyp+2NG+1, max_nzp+2NG+1,
        )
        weno7_fail_meta = gpu_zeros(Int32, 8)
        weno7_fail_value = gpu_zeros(FT, 1)
    else
        cache_i = gpu_zeros(FT, 1, 1, 1, 1)
        cache_j = gpu_zeros(FT, 1, 1, 1, 1)
        cache_k = gpu_zeros(FT, 1, 1, 1, 1)
        point_scratch = gpu_zeros(FT, 1, 1, 1)
        weno7_fail_meta = gpu_zeros(Int32, 1)
        weno7_fail_value = gpu_zeros(FT, 1)
    end

    _ct_weno7_cache_i_ref[] = cache_i
    _ct_weno7_cache_j_ref[] = cache_j
    _ct_weno7_cache_k_ref[] = cache_k
    _ct_weno7_point_scratch_ref[] = point_scratch
    _ct_weno7_fail_meta_ref[] = weno7_fail_meta
    _ct_weno7_fail_value_ref[] = weno7_fail_value

    # Face-EMFs (companion to Fx/Fy/Fz): 2 per direction (ey,ez on x-face; ez,ex on y-face; ex,ey on z-face)
    _femf_sz = ct_mode ? (max_nxp+1, max_nyp, max_nzp) : (1, 1, 1)
    _femf_sy = ct_mode ? (max_nxp, max_nyp+1, max_nzp) : (1, 1, 1)
    _femf_sx = ct_mode ? (max_nxp, max_nyp, max_nzp+1) : (1, 1, 1)
    shared_ez_x1f = gpu_zeros(FT, _femf_sz...)
    shared_ey_x1f = gpu_zeros(FT, _femf_sz...)
    shared_ez_x2f = gpu_zeros(FT, _femf_sy...)
    shared_ex_x2f = gpu_zeros(FT, _femf_sy...)
    shared_ex_x3f = gpu_zeros(FT, _femf_sx...)
    shared_ey_x3f = gpu_zeros(FT, _femf_sx...)
    # Upwind weights (same staggering as face-EMFs)
    shared_wct_x1f = gpu_zeros(FT, _femf_sz...)
    shared_wct_x2f = gpu_zeros(FT, _femf_sy...)
    shared_wct_x3f = gpu_zeros(FT, _femf_sx...)
    # Cell-centered resistive E=eta*curl(B), including one layer around the
    # physical domain for the four-cell corner correction at boundary edges.
    _cce_sz = ct_mode && resistive ?
        (max_nxp+2NG, max_nyp+2NG, max_nzp+2NG) : (1, 1, 1)
    shared_cc_ex = gpu_zeros(FT, _cce_sz...)
    shared_cc_ey = gpu_zeros(FT, _cce_sz...)
    shared_cc_ez = gpu_zeros(FT, _cce_sz...)

    rkl2_buffers = Dict{Int,Any}()
    @static if ct_resistive_rkl2_active
        for (bid, b) in blocks
            b.Bx_face === nothing && continue
            energy_size = (size(b.U,1), size(b.U,2), size(b.U,3))
            rkl2_buffers[bid] = (
                bx_previous2=gpu_zeros(FT, size(b.Bx_face)...),
                by_previous2=gpu_zeros(FT, size(b.By_face)...),
                bz_previous2=gpu_zeros(FT, size(b.Bz_face)...),
                bx_first=gpu_zeros(FT, size(b.Bx_face)...),
                by_first=gpu_zeros(FT, size(b.By_face)...),
                bz_first=gpu_zeros(FT, size(b.Bz_face)...),
                energy_previous2=gpu_zeros(FT, energy_size...),
                energy_first=gpu_zeros(FT, energy_size...),
            )
        end
    end



    forcex = zero(FT)
    flowx  = zero(FT)

    activeTime = zero(FT)
    tt = 0
    current_dt = dt
    restored_ct_face_b = Set{Int}()

    # ─── GPU-direct CEBL Inflow Mapping Kernel ───
    function couple_cebl_kernel!(U_main, Q_main, U_pre, Q_pre, nxp_pre, nyp, nzp, NG, Ncons, Nprim, dp_offset, gamma_val)
        j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
        k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
        if j > nyp + 2*NG || k > nzp + 2*NG
            return
        end

        # Copy last NG layers from precursor outlet to main domain inlet ghost cells.
        # Mean-fluctuation splitting: only the pressure *fluctuation* is copied;
        # the mean level is replaced by the main-domain inlet mean to avoid a DC
        # pressure jump at the precursor→main interface (which causes checkerboard
        # oscillations in the first few inlet cells).
        inv_gamma = one(FT) / gamma_val
        for g in 1:NG
            src_i = NG + nxp_pre - NG + g
            dst_i = g

            @inbounds ρ_pre = Q_pre[src_i, j, k, 1]
            @inbounds u_pre = Q_pre[src_i, j, k, 2]
            @inbounds v_pre = Q_pre[src_i, j, k, 3]
            @inbounds w_pre = Q_pre[src_i, j, k, 4]
            @inbounds p_pre = Q_pre[src_i, j, k, 5]

            # Isentropic pressure shift: p_ghost = p_pre + Δp, ρ follows p^(1/γ)
            p_ghost = p_pre + dp_offset
            ratio = p_ghost / p_pre
            ρ_ghost = ρ_pre * exp(log(ratio) * inv_gamma)

            # Write Q: ρ and p shifted, u/v/w/T unchanged
            @inbounds Q_main[dst_i, j, k, 1] = ρ_ghost
            @inbounds Q_main[dst_i, j, k, 2] = u_pre
            @inbounds Q_main[dst_i, j, k, 3] = v_pre
            @inbounds Q_main[dst_i, j, k, 4] = w_pre
            @inbounds Q_main[dst_i, j, k, 5] = p_ghost
            if Nprim >= 6
                @inbounds Q_main[dst_i, j, k, 6] = Q_pre[src_i, j, k, 6]  # T unchanged
            end

            # Recompute U from corrected primitives
            @inbounds U_main[dst_i, j, k, 1] = ρ_ghost
            @inbounds U_main[dst_i, j, k, 2] = ρ_ghost * u_pre
            @inbounds U_main[dst_i, j, k, 3] = ρ_ghost * v_pre
            @inbounds U_main[dst_i, j, k, 4] = ρ_ghost * w_pre
            e_int = p_ghost / ((gamma_val - one(FT)) * ρ_ghost)
            @inbounds U_main[dst_i, j, k, 5] = ρ_ghost * (e_int + FT(0.5) * (u_pre*u_pre + v_pre*v_pre + w_pre*w_pre))
        end
        return
    end

    function couple_cebl!(blocks, world_rank)
        if !(isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
            return
        end
        γ_val = isdefined(Main, :γ) ? FT(Main.γ) : FT(1.4)
        for bid in 0:4
            # All ranks that own main block `bid` must participate in the Allreduce
            # below, even if they don't own the precursor block (bid+5). Only the
            # ξ-head rank (rx=0) owns both; ξ-tail ranks own only the main block.
            # Skipping the Allreduce on tail ranks would deadlock.
            has_main = haskey(blocks, bid)
            has_pre  = haskey(blocks, bid + 5)

            if has_main
                bm = blocks[bid]
                Nprocs_b = Block_Nprocs[bid+1]
                is_head = (bm.rx == 0)  # this rank owns the ξ=0 face
            else
                is_head = false
            end

            # ── Compute mean-pressure offset Δp = ⟨p⟩_main_inlet - ⟨p⟩_pre_outlet ──
            # Only the ξ-head rank (rx=0) owns both the inlet face and the precursor;
            # other ranks contribute zero. All ranks in block_comms[bid] must Allreduce.
            if is_head && has_pre
                bp = blocks[bid+5]
                nxp_pre = bp.Nx
                nyp = bm.Ny
                nzp = bm.Nz

                # Precursor outlet face: last interior cell i = NG + nxp_pre
                ix_pre = NG + nxp_pre
                p_pre_slice = Array(@view bp.Q[ix_pre, NG+1:bp.Ny+NG, NG+1:bp.Nz+NG, 5])
                A_pre_slice = Array(@view bp.Areai[ix_pre+1, NG+1:bp.Ny+NG, NG+1:bp.Nz+NG])
                num_pre = Float64(sum(p_pre_slice .* A_pre_slice))
                den_pre = Float64(sum(A_pre_slice))

                # Main inlet face: first interior cell i = NG + 1
                ix_main = NG + 1
                p_main_slice = Array(@view bm.Q[ix_main, NG+1:bm.Ny+NG, NG+1:bm.Nz+NG, 5])
                A_main_slice = Array(@view bm.Areai[ix_main, NG+1:bm.Ny+NG, NG+1:bm.Nz+NG])
                num_main = Float64(sum(p_main_slice .* A_main_slice))
                den_main = Float64(sum(A_main_slice))
            else
                num_pre = 0.0; den_pre = 0.0
                num_main = 0.0; den_main = 0.0
            end

            _sub = get(block_comms, bid, nothing)
            if _sub === nothing
                num_pre_g = num_pre; den_pre_g = den_pre
                num_main_g = num_main; den_main_g = den_main
            else
                num_pre_g = MPI.Allreduce(num_pre, MPI.SUM, _sub)
                den_pre_g = MPI.Allreduce(den_pre, MPI.SUM, _sub)
                num_main_g = MPI.Allreduce(num_main, MPI.SUM, _sub)
                den_main_g = MPI.Allreduce(den_main, MPI.SUM, _sub)
            end

            # Only the head rank with the precursor performs the ghost copy
            if is_head && has_pre
                bp = blocks[bid+5]
                p_avg_pre = den_pre_g > 0.0 ? num_pre_g / den_pre_g : 0.0
                p_avg_main = den_main_g > 0.0 ? num_main_g / den_main_g : 0.0
                dp_offset = FT(p_avg_main - p_avg_pre)

                nxp_pre = bp.Nx
                nyp = bm.Ny
                nzp = bm.Nz
                threads_copy = (1, 16, 16)
                blocks_copy = (1, cld(nyp + 2*NG, threads_copy[2]), cld(nzp + 2*NG, threads_copy[3]))

                @gpu_launch threads=threads_copy blocks=blocks_copy couple_cebl_kernel!(
                    bm.U, bm.Q, bp.U, bp.Q, Int32(nxp_pre), Int32(nyp), Int32(nzp), Int32(NG), Int32(Ncons), Int32(Nprim), dp_offset, γ_val
                )
            end
        end
    end

    # ── RPO initialization ──
    init_rpo_mode = isdefined(Main, :rpo_mode) && Main.rpo_mode
    if init_rpo_mode && equation_type == :MHD && ct_mode &&
       ct_primitive_recovery == CT_PRIMITIVE_POINT6
        error(
            "POINT6 CT cannot initialize from the Q-only RPO format; " *
            "conservative U averages and staggered face fluxes are required",
        )
    end
    if init_rpo_mode
        if world_rank == 0
            println(">>> Initializing from RPO input file path prefix: $(Main.init_rpo_path) ...")
        end
        for (bid, b) in blocks
            if b.id >= 5
                continue
            end
            chkname = "$(Main.init_rpo_path)-b$(b.id).h5"
            if !isfile(chkname)
                error("RPO initialization file $chkname not found!")
            end
            Q_h = h5open(chkname, "r") do f
                lox = b.ox + 1; hix = b.ox + b.Nx
                loy = b.oy + 1; hiy = b.oy + b.Ny
                loz = b.oz + 1; hiz = b.oz + b.Nz
                Q_interior = f["Q"][lox:hix, loy:hiy, loz:hiz, :]
                nprim_chk = size(Q_interior, 4)

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
            end
            copyto!(b.Q, GPUArray(FT.(Q_h)))
            
            nb_init = (cld(b.Nx + 2*NG, nthreads[1]), cld(b.Ny + 2*NG, nthreads[2]), cld(b.Nz + 2*NG, nthreads[3]))
            @gpu_launch threads=nthreads blocks=nb_init prim2c(b.U, b.Q, b.Nx, b.Ny, b.Nz)
            copyto!(b.Un, b.U)
        end
        couple_cebl!(blocks, world_rank)
        for (bid, b) in blocks
            Nprocs_b = Block_Nprocs[bid + 1]
            rx_b = multi_block_mode ? 0 : rankx
            ry_b = multi_block_mode ? 0 : ranky
            rz_b = multi_block_mode ? 0 : rankz
            fillGhost(b.Q, b.U, rx_b, ry_b, rz_b,
                      b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj, b.nxk, b.nyk, b.nzk,
                      b.Areai, b.Areaj, b.Areak, b.Vol, current_dt,
                      b.x, b.y, b.z, b.id, b.Nx, b.Ny, b.Nz, Nprocs_b, 1, face_bc, bc_params,
                      @isdefined(dsrfg_params) ? dsrfg_params : default_dsrfg_params)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    elseif restart != "none"
        restart_step = parse(Int, restart)
        if world_rank == 0
            println(">>> Restarting from checkpoint step $restart_step ...")
        end

        for (bid, b) in blocks
            if b.id >= 5 && !(isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                continue
            end
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

            if equation_type == :MHD && ct_mode && b.Bx_face !== nothing
                ct_payload = h5open(chkname, "r") do f
                    required = ("U", "Bx_face", "By_face", "Bz_face")
                    if !all(name -> haskey(f, name), required)
                        return nothing
                    end
                    lox = b.ox + 1; hix = b.ox + b.Nx
                    loy = b.oy + 1; hiy = b.oy + b.Ny
                    loz = b.oz + 1; hiz = b.oz + b.Nz
                    return (
                        U=f["U"][lox:hix, loy:hiy, loz:hiz, :],
                        Bx=f["Bx_face"][lox:hix+1, loy:hiy, loz:hiz],
                        By=f["By_face"][lox:hix, loy:hiy+1, loz:hiz],
                        Bz=f["Bz_face"][lox:hix, loy:hiy, loz:hiz+1],
                    )
                end
                if ct_payload === nothing
                    if ct_primitive_recovery == CT_PRIMITIVE_POINT6
                        error(
                            "Checkpoint $chkname predates conservative U and " *
                            "staggered CT face-flux storage; POINT6 restart " *
                            "cannot reinterpret point Q as cell averages",
                        )
                    end
                else
                    copyto!(@view(b.U[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny, NG+1:NG+b.Nz, :,
                    ]), GPUArray(FT.(ct_payload.U)))
                    copyto!(@view(b.Bx_face[
                        NG+1:NG+b.Nx+1, NG+1:NG+b.Ny, NG+1:NG+b.Nz,
                    ]), GPUArray(FT.(ct_payload.Bx)))
                    copyto!(@view(b.By_face[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny+1, NG+1:NG+b.Nz,
                    ]), GPUArray(FT.(ct_payload.By)))
                    copyto!(@view(b.Bz_face[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny, NG+1:NG+b.Nz+1,
                    ]), GPUArray(FT.(ct_payload.Bz)))
                    push!(restored_ct_face_b, bid)
                end
            end
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
        if isdefined(Main, :cebl_forcing) && Main.cebl_forcing
            # Remap precursor block offsets (Blocks 5-9) to match main blocks (Blocks 0-4)
            end_bound = rank_offsets[6]
            for i in 5:9
                rank_offsets[i+1] = rank_offsets[i-4]
            end
            rank_offsets[11] = end_bound
        end
    end

    # Pre-allocate inter-block ghost exchange buffers
    # (Moved before warmup so copy_ghost_face! can be used for coordinate exchange)
    ghost_pool = init_ghost_buffer_pool(blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b, Nprim)
    ct_sync_plan = ct_mode ? build_ct_sync_plan(
        blocks, face_bc, connectivity, Block_Nprocs, rank_offsets,
        Nx_b, Ny_b, Nz_b,
    ) : nothing
    ct_junction_plan = ct_mode ? build_ct_junction_plan(
        blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b,
    ) : nothing

    # (Spectral Warmup block removed as high-order CMD6 metrics resolve grid stretching)

    # ── Auto-tune kernel launch configurations ──
    first_bid, first_b = first(blocks)
    nxp_t, nyp_t, nzp_t = first_b.Nx, first_b.Ny, first_b.Nz
    tune_configs = KernelConfig[]

    # Reconstruction kernels →per-direction auto-tuning
    # Each direction gets its own optimal block size to match memory access patterns.
    _verbose = (world_rank == 0)
    if eigen_reconstruction && equation_type == :MHD && ct_mode
        cfg_state_left_i = auto_tune_kernel("CTMHD_charL_i", ct_mhd_characteristic_reconstruct_left_i_kernel!,
            first_b.Q, shared_Fx,
            first_b.Areai, first_b.nxi, first_b.nyi, first_b.nzi, first_b.Bx_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_right_i = auto_tune_kernel("CTMHD_charR_i", ct_mhd_characteristic_reconstruct_right_i_kernel!,
            first_b.Q, shared_Fvx,
            first_b.Areai, first_b.nxi, first_b.nyi, first_b.nzi, first_b.Bx_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_i = auto_tune_kernel("CTMHD_hlld_i", ct_mhd_hlld_flux_i_kernel!,
            shared_Fx, shared_Fvx, shared_Fx, shared_rho_sum_x,
            first_b.Areai, first_b.nxi, first_b.nyi, first_b.nzi,
            nxp_t, nyp_t, nzp_t, ch_glm_current, Int32(0), shared_ct_pos_meta;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_left_j = auto_tune_kernel("CTMHD_charL_j", ct_mhd_characteristic_reconstruct_left_j_kernel!,
            first_b.Q, shared_Fy,
            first_b.Areaj, first_b.nxj, first_b.nyj, first_b.nzj, first_b.By_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_right_j = auto_tune_kernel("CTMHD_charR_j", ct_mhd_characteristic_reconstruct_right_j_kernel!,
            first_b.Q, shared_Fvy,
            first_b.Areaj, first_b.nxj, first_b.nyj, first_b.nzj, first_b.By_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_j = auto_tune_kernel("CTMHD_hlld_j", ct_mhd_hlld_flux_j_kernel!,
            shared_Fy, shared_Fvy, shared_Fy, shared_rho_sum_y,
            first_b.Areaj, first_b.nxj, first_b.nyj, first_b.nzj,
            nxp_t, nyp_t, nzp_t, ch_glm_current, Int32(0), shared_ct_pos_meta;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_left_k = auto_tune_kernel("CTMHD_charL_k", ct_mhd_characteristic_reconstruct_left_k_kernel!,
            first_b.Q, shared_Fz,
            first_b.Areak, first_b.nxk, first_b.nyk, first_b.nzk, first_b.Bz_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_right_k = auto_tune_kernel("CTMHD_charR_k", ct_mhd_characteristic_reconstruct_right_k_kernel!,
            first_b.Q, shared_Fvz,
            first_b.Areak, first_b.nxk, first_b.nyk, first_b.nzk, first_b.Bz_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_k = auto_tune_kernel("CTMHD_hlld_k", ct_mhd_hlld_flux_k_kernel!,
            shared_Fz, shared_Fvz, shared_Fz, shared_rho_sum_z,
            first_b.Areak, first_b.nxk, first_b.nyk, first_b.nzk,
            nxp_t, nyp_t, nzp_t, ch_glm_current, Int32(0), shared_ct_pos_meta;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        push!(tune_configs,
            cfg_state_left_i, cfg_state_right_i,
            cfg_state_left_j, cfg_state_right_j,
            cfg_state_left_k, cfg_state_right_k)
    elseif eigen_reconstruction
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
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areai,
            shared_Fx, shared_rho_sum_x, first_b.Areai,
            first_b.nxi, first_b.nyi, first_b.nzi, nxp_t, nyp_t, nzp_t,
            first_b.stencil_i, first_b.Δstencil_i, first_b.lin_phi_i,
            first_b.stencil_R_i, first_b.Δstencil_R_i, ch_glm_current, Int32(0),
            ct_mode ? first_b.Bx_face : first_b.ϕ,
            cache_i, first_b.Vol, current_dt,
            Int32(1), shared_ct_pos_meta, shared_ct_pos_values;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_j = auto_tune_kernel("Conser_recon_j", Conser_reconstruct_j,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areaj,
            shared_Fy, shared_rho_sum_y, first_b.Areaj,
            first_b.nxj, first_b.nyj, first_b.nzj, nxp_t, nyp_t, nzp_t,
            first_b.stencil_j, first_b.Δstencil_j, first_b.lin_phi_j,
            first_b.stencil_R_j, first_b.Δstencil_R_j, ch_glm_current, Int32(0),
            ct_mode ? first_b.By_face : first_b.ϕ,
            cache_j, first_b.Vol, current_dt,
            Int32(1), shared_ct_pos_meta, shared_ct_pos_values;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_k = auto_tune_kernel("Conser_recon_k", Conser_reconstruct_k,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areak,
            shared_Fz, shared_rho_sum_z, first_b.Areak,
            first_b.nxk, first_b.nyk, first_b.nzk, nxp_t, nyp_t, nzp_t,
            first_b.stencil_k, first_b.Δstencil_k, first_b.lin_phi_k,
            first_b.stencil_R_k, first_b.Δstencil_R_k, ch_glm_current, Int32(0),
            ct_mode ? first_b.Bz_face : first_b.ϕ,
            cache_k, first_b.Vol, current_dt,
            Int32(1), shared_ct_pos_meta, shared_ct_pos_values;
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
            first_b.Vol, nxp_t, nyp_t, nzp_t, first_b.is_interblock, false;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_visc_j = auto_tune_kernel("visc_j", viscous_flux_j,
            first_b.Q, shared_Fvy, first_b.Areai, first_b.Areaj, first_b.Areak,
            first_b.nxi, first_b.nyi, first_b.nzi,
            first_b.nxj, first_b.nyj, first_b.nzj,
            first_b.nxk, first_b.nyk, first_b.nzk,
            first_b.Vol, nxp_t, nyp_t, nzp_t, first_b.is_interblock, false;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_visc_k = auto_tune_kernel("visc_k", viscous_flux_k,
            first_b.Q, shared_Fvz, first_b.Areai, first_b.Areaj, first_b.Areak,
            first_b.nxi, first_b.nyi, first_b.nzi,
            first_b.nxj, first_b.nyj, first_b.nzj,
            first_b.nxk, first_b.nyk, first_b.nzk,
            first_b.Vol, nxp_t, nyp_t, nzp_t, first_b.is_interblock, false;
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

    # Face CT data do not exist until the first U/Q ghost synchronization and
    # ct_init_face_b! have completed. Once ready, POINT6 owns both the physical
    # point state and the Q halo exchanged for characteristic reconstruction.
    ct_face_state_ready = Ref(false)

    function sync_ct_point_state!(; sync_q_ghosts::Bool)
        for (_, b) in blocks
            if b.Bx_face === nothing
                continue
            end
            ct_sync_cell_b!(b, b.Nx, b.Ny, b.Nz)
            ct_update_q_b!(b, b.Nx, b.Ny, b.Nz)
        end
        @static if ct_primitive_recovery == CT_PRIMITIVE_POINT6
            if sync_q_ghosts
                copy_ghost_face!(
                    blocks, connectivity, Block_Nprocs, rank_offsets,
                    Nx_b, Ny_b, Nz_b, :Q, Nprim, ghost_pool;
                    full_range=false,
                )
                for (bid, b) in blocks
                    exchange_ghost(
                        b.Q, Nprim, block_comms[bid], b.Nx, b.Ny, b.Nz,
                        b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
                        b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
                        b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
                        sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
                        rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2,
                    )
                end
                copy_ghost_face!(
                    blocks, connectivity, Block_Nprocs, rank_offsets,
                    Nx_b, Ny_b, Nz_b, :Q, Nprim, ghost_pool;
                    full_range=true, delta_mode=true,
                )
            end
        end
        return nothing
    end

    # Define unified synchronization function (two-pass ghost exchange)
    function sync_blocks!(
        tt_val; compute_fn=nothing, step::Integer=0, rk_stage::Integer=0,
    )
        # Execute CEBL coupling mapping
        couple_cebl!(blocks, world_rank)

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
                      b.Areai, b.Areaj, b.Areak, b.Vol, current_dt,
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
        #
        # delta_mode=true: packs only 4 border sub-regions instead of the full slab,
        # relying on Step 1 (full_range=false) for the middle part. Saves ~5% of
        # ghost-exchange time.
        #
        # Investigation history: An initial A/B test (LS_true vs delta_off, 100k steps)
        # appeared to show delta_mode=true caused severe interface artifacts (B4 k=1
        # p_std: 45.5 vs 0.96). However, a controlled A/B test with identical code
        # (delta_true_r2 vs delta_false_r2, 10k steps) showed NO difference:
        #   B4 k=0 std: 3.06 (delta_mode=true) vs 3.28 (delta_mode=false)
        # The original artifact was caused by an unrelated code difference between
        # the two runs (different solver.jl versions synced at different times).
        # delta_mode=true is confirmed safe and re-enabled.
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
            # Interface filter strength. With the revised σ rule in
            # interface_filter_kernel! the effective σ at a cell is
            #     σ_max × (σ_floor + (1-σ_floor)·phi_local)        (σ_floor=0.5)
            # so junction cells (phi≈1) see the full _σ_max while smooth interior
            # interface layers see only 50% of it. _σ_max=0.05 gives ~0.05 damping
            # per call at the butterfly singularity (per-step contraction of 2Δ
            # modes) and ~0.025 at well-aligned interblock faces.
            _σ_max = FT(0.05)
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
            couple_cebl!(blocks, world_rank)
            for (bid, b) in blocks
                Nprocs_b = Block_Nprocs[bid + 1]
                rx_b = multi_block_mode ? 0 : rankx
                ry_b = multi_block_mode ? 0 : ranky
                rz_b = multi_block_mode ? 0 : rankz
                fillGhost(b.Q, b.U, rx_b, ry_b, rz_b,
                          b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj, b.nxk, b.nyk, b.nzk,
                          b.Areai, b.Areaj, b.Areak, b.Vol, current_dt,
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
            # 5. Full-range copy (delta_mode=true — see Step 4 comment above)
            copy_ghost_face!(blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b, :U, Ncons, ghost_pool;
                             full_range=true, delta_mode=true)
            gpu_sync()
        end
        if profiling; gpu_sync(); t_sync_sub[5] += (time_ns()-_ts)/1e9; _ts=time_ns(); end

        # Step 5: Primitive refresh. Characteristic CT stores physical point Q
        # and exchanges that point state directly; c2Prim_ghost would instead
        # reinterpret conservative averages as point values and reduce the path
        # to second order at every rank or block boundary.
        if ct_face_state_ready[] &&
           ct_primitive_recovery == CT_PRIMITIVE_POINT6
            sync_ct_point_state!(sync_q_ghosts=true)
        else
            for (bid, b) in blocks
                nb_loc = (cld(b.Nx+2*NG, nthreads[1]), cld(b.Ny+2*NG, nthreads[2]), cld(b.Nz+2*NG, nthreads[3]))
                @gpu_launch threads=nthreads blocks=nb_loc c2Prim_ghost(b.U, b.Q, b.Nx, b.Ny, b.Nz)
                @static if strict_ct_positivity && ct_mode && splitMethodID == 4
                    @gpu_launch threads=nthreads blocks=nb_loc ct_check_cell_positivity_kernel!(
                        shared_ct_pos_meta, shared_ct_pos_values, b.U, FT(γ),
                        CT_POS_SITE_GHOST_REFRESH, Int32(1),
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    gpu_sync()
                    ct_check_positivity_or_abort!(
                        shared_ct_pos_meta, shared_ct_pos_values;
                        rank=world_rank, block=bid, step=step, rk_stage=rk_stage,
                    )
                end
            end
        end
        for (_, b) in blocks
            @check_nan(b.Q, "Q after ghost refresh", b.id, world_rank, tt_val)
        end
        if profiling; gpu_sync(); t_sync_sub[6] += (time_ns()-_ts)/1e9; end

        # Ghost-sync debug dump: disabled unless run script sets `debug_ghost_sync=true`.
        # This block D2H-copies two full Q arrays and prints per sub-step — a significant
        # per-step cost, so keep it OFF in production (defaults to false via @isdefined).
        if @isdefined(debug_ghost_sync) && debug_ghost_sync && haskey(blocks, 6) && haskey(blocks, 5)
            b6 = blocks[6]; b5 = blocks[5]
            q6 = Array(b6.Q); q5 = Array(b5.Q)
            println("=== DEBUG GHOST SYNC (Step $tt_val) ===")
            # Block 6 Face 4 (eta-hi) real boundary (j=Ny+NG) vs Block 5 Face 3 (eta-lo) ghost boundary (j=NG)
            j6_real = b6.Ny + NG
            j5_ghost = NG
            println("  Block 6 Real Q[NG+1, $j6_real, NG+8, 2] = ", q6[NG+1, j6_real, NG+8, 2])
            println("  Block 5 Ghost Q[NG+1, $j5_ghost, NG+8, 2] = ", q5[NG+1, j5_ghost, NG+8, 2])
            
            # Block 5 Face 3 (eta-lo) real boundary (j=NG+1) vs Block 6 Face 4 (eta-hi) ghost boundary (j=Ny+NG+1)
            j5_real = NG + 1
            j6_ghost = b6.Ny + NG + 1
            println("  Block 5 Real Q[NG+1, $j5_real, NG+8, 2] = ", q5[NG+1, j5_real, NG+8, 2])
            println("  Block 6 Ghost Q[NG+1, $j6_ghost, NG+8, 2] = ", q6[NG+1, j6_ghost, NG+8, 2])
            println("=======================================")
        end
    end

    # Phase 2: Create compute stream for comm-compute overlap
    compute_stream = gpu_stream_create()

    # Initial ghost cell synchronization (must happen BEFORE CT face-B init)
    sync_blocks!(zero(FT))

    # CT: Initialize face-centered B from cell-centered B for all blocks
    # Must run AFTER ghost sync (so boundary U is valid) and on ALL ranks.
    if equation_type == :MHD && ct_mode
        for (bid, b) in blocks
            if b.Bx_face !== nothing && !(bid in restored_ct_face_b)
                ct_init_face_b!(b, b.Nx, b.Ny, b.Nz)
            end
        end
        used_initial_face_flux_hook = _run_initial_ct_face_flux_process!(
            Main, blocks, world_rank, Block_Nprocs, block_comms,
        )
        ct_sync_rank_sheets!(
            blocks, block_comms, Block_Nprocs;
            sync_face_flux=true, sync_edges=false,
        )
        ct_sync_interface_sheets!(
            blocks, ct_sync_plan; sync_face_flux=true, sync_edges=false,
        )
        local_initial_divb = maximum(
            ct_initial_relative_face_divergence(b) for b in values(blocks)
            if b.Bx_face !== nothing
        ; init=zero(FT))
        global_initial_divb = MPI.Allreduce(
            FT(local_initial_divb), MPI.MAX, MPI.COMM_WORLD,
        )
        if global_initial_divb > ct_initial_divb_tolerance
            projection_allowed = ct_initial_projection &&
                Nblocks == 1 && length(Block_Nprocs) == 1 &&
                Tuple(Block_Nprocs[1]) == (1, 1, 1) &&
                isempty(restored_ct_face_b) && !used_initial_face_flux_hook
            projection_allowed || error(
                "Initial CT face field violates the discrete divergence gate: " *
                "relative divB=$global_initial_divb, tolerance=" *
                "$ct_initial_divb_tolerance. Supply a vector-potential face " *
                "initializer or a divergence-free CT checkpoint.",
            )
            for (_, b) in blocks
                b.Bx_face === nothing && continue
                local_periodic = isdefined(Main, :Iperiodic) ?
                    Tuple(Main.Iperiodic) : (false, false, false)
                projection = ct_project_initial_face_b!(
                    b, local_periodic;
                    tolerance=ct_initial_divb_tolerance,
                )
                if world_rank == 0
                    @printf(
                        ">>> CT: projected initial face B in %d iterations (relative divB=%.6e).\n",
                        projection.iterations, projection.relative_divergence,
                    )
                end
            end
            ct_sync_rank_sheets!(
                blocks, block_comms, Block_Nprocs;
                sync_face_flux=true, sync_edges=false,
            )
            ct_sync_interface_sheets!(
                blocks, ct_sync_plan;
                sync_face_flux=true, sync_edges=false,
            )
            local_initial_divb = maximum(
                ct_initial_relative_face_divergence(b) for b in values(blocks)
                if b.Bx_face !== nothing
            ; init=zero(FT))
            global_initial_divb = MPI.Allreduce(
                FT(local_initial_divb), MPI.MAX, MPI.COMM_WORLD,
            )
        end
        global_initial_divb <= ct_initial_divb_tolerance || error(
            "Initial CT divergence projection failed: relative divB=" *
            "$global_initial_divb, tolerance=$ct_initial_divb_tolerance",
        )
        for (bid, b) in blocks
            if b.Bx_face !== nothing
                # Interface halo construction currently consumes cell B. Use a
                # fresh local LSQ predictor, then replace it with POINT6 after
                # every face halo has been filled.
                ct_sync_cell_b_lsq2!(b, b.Nx, b.Ny, b.Nz)
                ct_fill_zerograd_face_b!(b, b.Nx, b.Ny, b.Nz)
                if isdefined(Main, :Iperiodic)
                    local_periodic = ntuple(3) do direction
                        Main.Iperiodic[direction] &&
                            Block_Nprocs[bid + 1][direction] == 1
                    end
                    ct_fill_periodic_face_b!(
                        b, b.Nx, b.Ny, b.Nz, local_periodic,
                    )
                end
            end
        end
        ct_sync_rank_face_halos!(blocks, block_comms, Block_Nprocs)
        ct_sync_interface_face_halos!(blocks, ct_sync_plan)
        ct_face_state_ready[] = true
        sync_ct_point_state!(sync_q_ghosts=true)
        if world_rank == 0
            @printf(
                ">>> CT: Face-centered B initialized; relative divB=%.6e.\n",
                global_initial_divb,
            )
        end
    end

    # ══════════════════════════════════════════════════════════════
    # Timing accumulators (temporary profiling)
    # ══════════════════════════════════════════════════════════════
    t_sync = 0.0; t_shock = 0.0; t_advance = 0.0
    t_forcing = 0.0; t_div = 0.0
    timing_steps = 0
    _t0 = UInt64(0)

    # Entropy diagnostic baseline (set on first measurement, persisted thereafter)
    _η0_global::Float64 = NaN

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

    @static if ct_resistive_rkl2_active
        function advance_resistive_ct_rkl2!(
            split_dt::FT, split_start_time::FT;
            step::Integer, split_half::Integer,
        )
            explicit_dt_local = FT(Inf)
            for (_, b) in blocks
                b.Bx_face === nothing && continue
                fill!(b.LTS_dt, zero(FT))
                nb_dt = (
                    cld(b.Nx, nthreads[1]),
                    cld(b.Ny, nthreads[2]),
                    cld(b.Nz, nthreads[3]),
                )
                @gpu_launch threads=nthreads blocks=nb_dt ct_resistive_explicit_dt_kernel!(
                    b.LTS_dt, b.Vol, b.Areai, b.Areaj, b.Areak,
                    Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                block_dt = FT(mapreduce(
                    value -> value > zero(FT) ? value : FT(Inf),
                    min, b.LTS_dt,
                ))
                explicit_dt_local = min(explicit_dt_local, block_dt)
            end
            explicit_resistive_dt = ct_rkl2_safety * MPI.Allreduce(
                explicit_dt_local, MPI.MIN, MPI.COMM_WORLD,
            )
            stage_count = ct_rkl2_stage_count(
                split_dt, FT(explicit_resistive_dt);
                max_stages=ct_rkl2_max_stages,
            )
            coefficients = ct_rkl2_coefficients(stage_count, FT)
            if world_rank == 0 && (step <= 2 || stage_count > 2)
                @printf(
                    "CT_RKL2_STRANG step=%d half=%d stages=%d dt=%.8e explicit_dt=%.8e\n",
                    step, split_half, stage_count, split_dt,
                    explicit_resistive_dt,
                )
            end

            for rkl_stage in 1:stage_count
                first_stage = rkl_stage == 1
                coefficient = first_stage ? nothing :
                    coefficients.stages[rkl_stage-1]
                rhs_dt_coefficient = first_stage ?
                    coefficients.mu_tilde1*split_dt :
                    coefficient.mu_tilde*split_dt
                stage_abscissa = first_stage ?
                    coefficients.first_abscissa : coefficient.abscissa
                diagnostic_stage = split_half*100 + rkl_stage

                @static if strict_ct_positivity
                    ct_reset_positivity!(
                        shared_ct_pos_meta, shared_ct_pos_values,
                    )
                end
                for (bid, b) in blocks
                    b.Bx_face === nothing && continue
                    buffers = rkl2_buffers[bid]
                    nb_cc = (
                        cld(b.Nx+2NG, nthreads[1]),
                        cld(b.Ny+2NG, nthreads[2]),
                        cld(b.Nz+2NG, nthreads[3]),
                    )
                    nb_ct = (
                        cld(b.Nx+1, nthreads[1]),
                        cld(b.Ny+1, nthreads[2]),
                        cld(b.Nz+1, nthreads[3]),
                    )
                    nb_cell = (
                        cld(b.Nx, nthreads[1]),
                        cld(b.Ny, nthreads[2]),
                        cld(b.Nz, nthreads[3]),
                    )
                    nb_visc_i = (
                        cld(b.Nx+2NG, threads_visc_i[1]),
                        cld(b.Ny+2NG, threads_visc_i[2]),
                        cld(b.Nz+2NG, threads_visc_i[3]),
                    )
                    nb_visc_j = (
                        cld(b.Nx+2NG, threads_visc_j[1]),
                        cld(b.Ny+2NG, threads_visc_j[2]),
                        cld(b.Nz+2NG, threads_visc_j[3]),
                    )
                    nb_visc_k = (
                        cld(b.Nx+2NG, threads_visc_k[1]),
                        cld(b.Ny+2NG, threads_visc_k[2]),
                        cld(b.Nz+2NG, threads_visc_k[3]),
                    )
                    fill!(b.Ex_edge, zero(FT))
                    fill!(b.Ey_edge, zero(FT))
                    fill!(b.Ez_edge, zero(FT))
                    @gpu_launch threads=nthreads blocks=nb_cc ct_compute_resistive_cell_emf_kernel!(
                        shared_cc_ex, shared_cc_ey, shared_cc_ez, b.Q,
                        b.Areai, b.nxi, b.nyi, b.nzi,
                        b.Areaj, b.nxj, b.nyj, b.nzj,
                        b.Areak, b.nxk, b.nyk, b.nzk,
                        b.Vol, Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    @gpu_launch threads=threads_visc_i blocks=nb_visc_i viscous_flux_i(
                        b.Q, shared_Fvx,
                        b.Areai, b.Areaj, b.Areak,
                        b.nxi, b.nyi, b.nzi,
                        b.nxj, b.nyj, b.nzj,
                        b.nxk, b.nyk, b.nzk,
                        b.Vol, b.Nx, b.Ny, b.Nz, b.is_interblock, true)
                    @gpu_launch threads=threads_visc_j blocks=nb_visc_j viscous_flux_j(
                        b.Q, shared_Fvy,
                        b.Areai, b.Areaj, b.Areak,
                        b.nxi, b.nyi, b.nzi,
                        b.nxj, b.nyj, b.nzj,
                        b.nxk, b.nyk, b.nzk,
                        b.Vol, b.Nx, b.Ny, b.Nz, b.is_interblock, true)
                    @gpu_launch threads=threads_visc_k blocks=nb_visc_k viscous_flux_k(
                        b.Q, shared_Fvz,
                        b.Areai, b.Areaj, b.Areak,
                        b.nxi, b.nyi, b.nzi,
                        b.nxj, b.nyj, b.nzj,
                        b.nxk, b.nyk, b.nzk,
                        b.Vol, b.Nx, b.Ny, b.Nz, b.is_interblock, true)
                    @gpu_launch threads=nthreads blocks=nb_ct ct_add_resistive_edge_line_emf_kernel!(
                        b.Ex_edge, b.Ey_edge, b.Ez_edge,
                        shared_Fvx, shared_Fvy, shared_Fvz,
                        shared_cc_ex, shared_cc_ey, shared_cc_ez,
                        b.Areai, b.nxi, b.nyi, b.nzi,
                        b.Areaj, b.nxj, b.nyj, b.nzj,
                        b.Areak, b.nxk, b.nyk, b.nzk,
                        b.x, b.y, b.z,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    if first_stage
                        @gpu_launch threads=nthreads blocks=nb_cell ct_rkl2_energy_first_stage_kernel!(
                            b.U, b.Un,
                            buffers.energy_previous2, buffers.energy_first,
                            shared_Fvx, shared_Fvy, shared_Fvz, b.Vol,
                            FT(rhs_dt_coefficient),
                            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    else
                        @gpu_launch threads=nthreads blocks=nb_cell ct_rkl2_energy_stage_kernel!(
                            b.U, b.Un,
                            buffers.energy_previous2, buffers.energy_first,
                            shared_Fvx, shared_Fvy, shared_Fvz, b.Vol,
                            FT(rhs_dt_coefficient), FT(coefficient.mu),
                            FT(coefficient.nu), FT(coefficient.base),
                            FT(coefficient.gamma_ratio),
                            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    end
                    ct_enforce_physical_edge_emf!(b, bid, face_bc)
                    local_periodic = ntuple(3) do direction
                        isdefined(Main, :Iperiodic) &&
                            Main.Iperiodic[direction] &&
                            Block_Nprocs[bid + 1][direction] == 1
                    end
                    @gpu_launch threads=nthreads blocks=nb_ct ct_sync_periodic_edge_emf_kernel!(
                        b.Ex_edge, b.Ey_edge, b.Ez_edge,
                        local_periodic[1], local_periodic[2],
                        local_periodic[3],
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                end

                ct_sync_rank_sheets!(
                    blocks, block_comms, Block_Nprocs;
                    sync_face_flux=false, sync_edges=true,
                )
                ct_sync_interface_sheets!(
                    blocks, ct_sync_plan;
                    sync_face_flux=false, sync_edges=true,
                )
                ct_sync_junction_edges!(blocks, ct_junction_plan)
                for (bid, b) in blocks
                    b.Bx_face === nothing && continue
                    buffers = rkl2_buffers[bid]
                    nb_ct = (
                        cld(b.Nx+1, nthreads[1]),
                        cld(b.Ny+1, nthreads[2]),
                        cld(b.Nz+1, nthreads[3]),
                    )
                    if first_stage
                        @gpu_launch threads=nthreads blocks=nb_ct ct_rkl2_face_first_stage_kernel!(
                            b.Bx_face, b.By_face, b.Bz_face,
                            b.Bx_face_n, b.By_face_n, b.Bz_face_n,
                            buffers.bx_previous2, buffers.by_previous2,
                            buffers.bz_previous2,
                            buffers.bx_first, buffers.by_first, buffers.bz_first,
                            b.Ex_edge, b.Ey_edge, b.Ez_edge,
                            FT(rhs_dt_coefficient),
                            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    else
                        @gpu_launch threads=nthreads blocks=nb_ct ct_rkl2_face_stage_kernel!(
                            b.Bx_face, b.By_face, b.Bz_face,
                            b.Bx_face_n, b.By_face_n, b.Bz_face_n,
                            buffers.bx_previous2, buffers.by_previous2,
                            buffers.bz_previous2,
                            buffers.bx_first, buffers.by_first, buffers.bz_first,
                            b.Ex_edge, b.Ey_edge, b.Ez_edge,
                            FT(rhs_dt_coefficient), FT(coefficient.mu),
                            FT(coefficient.nu), FT(coefficient.base),
                            FT(coefficient.gamma_ratio),
                            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    end
                    ct_sync_cell_b_lsq2!(b, b.Nx, b.Ny, b.Nz)
                    ct_fill_zerograd_face_b!(b, b.Nx, b.Ny, b.Nz)
                    local_periodic = ntuple(3) do direction
                        isdefined(Main, :Iperiodic) &&
                            Main.Iperiodic[direction] &&
                            Block_Nprocs[bid + 1][direction] == 1
                    end
                    ct_fill_periodic_face_b!(
                        b, b.Nx, b.Ny, b.Nz, local_periodic,
                    )
                end
                ct_sync_rank_face_halos!(blocks, block_comms, Block_Nprocs)
                ct_sync_interface_face_halos!(blocks, ct_sync_plan)
                sync_ct_point_state!(sync_q_ghosts=false)
                sync_blocks!(
                    split_start_time + FT(stage_abscissa)*split_dt;
                    step=step, rk_stage=diagnostic_stage,
                )
                @static if strict_ct_positivity
                    for (_, b) in blocks
                        b.Bx_face === nothing && continue
                        nb_cell = (
                            cld(b.Nx, nthreads[1]),
                            cld(b.Ny, nthreads[2]),
                            cld(b.Nz, nthreads[3]),
                        )
                        @gpu_launch threads=nthreads blocks=nb_cell ct_check_cell_positivity_kernel!(
                            shared_ct_pos_meta, shared_ct_pos_values,
                            b.U, FT(γ), CT_POS_SITE_POST_CT_SYNC,
                            Int32(0), Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    end
                    gpu_sync()
                    ct_check_positivity_or_abort!(
                        shared_ct_pos_meta, shared_ct_pos_values;
                        rank=world_rank, block=-1, step=step,
                        rk_stage=diagnostic_stage,
                    )
                end
            end
            return stage_count
        end
    end

    if isdefined(Main, :in_situ_initial_process)
        in_situ_initial_process(activeTime, blocks, world_rank, Block_Nprocs, block_comms)
    end
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
            elseif forcing_mode == 3 || (isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                forcing_blocks = (isdefined(Main, :cebl_forcing) && Main.cebl_forcing) ? Dict{Int, Block}(id => b for (id, b) in blocks if id >= 5) : blocks
                deschamps_f1_val, deschamps_flowx_val = Update_deschamps_pipe_params(forcing_blocks, tt, 1, current_dt, MPI.COMM_WORLD, world_rank)
            end
        end
        if test_case == "HIT" || test_case == "MHDHIT"
            hit_u_mean, hit_v_mean, hit_w_mean, hit_Bx_mean, hit_By_mean, hit_Bz_mean = Update_HIT_forcing(blocks, hit_forcing_A, MPI.COMM_WORLD, world_rank, tt, 1)
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
                        shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                        shared_Fvx, shared_Fvy, shared_Fvz,
                        shared_dU_forced, shared_ct_pos_meta, shared_ct_pos_values,
                        world_rank, tt,
                        threads_recon_i, threads_recon_j, threads_recon_k,
                        threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                        forcex, flowx, cmf_f1_val,
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
                    shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                    shared_Fvx, shared_Fvy, shared_Fvz,
                    shared_dU_forced, shared_ct_pos_meta, shared_ct_pos_values,
                    world_rank, tt,
                    threads_recon_i, threads_recon_j, threads_recon_k,
                    threads_visc_i, threads_visc_j, threads_visc_k, threads_light,
                    forcex, flowx, cmf_f1_val,
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
            @static if strict_ct_positivity
                ct_reset_positivity!(shared_ct_pos_meta, shared_ct_pos_values)
            end
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

                @static if ct_resistive_rkl2_active
                    advance_resistive_ct_rkl2!(
                        FT(0.5)*FT(current_dt), FT(activeTime);
                        step=tt, split_half=1,
                    )
                    # SSP-RK3 must use the state after the leading Strang half-step
                    # as its own time-level backup.
                    for (_, b) in blocks
                        copyto!(b.Un, b.U)
                    end
                end

                if flow_forcing
                    if forcing_mode == 1
                        forcex, flowx = Update_bulk_force_params(blocks, tt, 1, MPI.COMM_WORLD, world_rank)
                    elseif forcing_mode == 2
                        cmf_f1_val = Update_const_massflux_params(blocks, tt, 1, current_dt, MPI.COMM_WORLD, world_rank)
                    elseif forcing_mode == 3 || (isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                        forcing_blocks = (isdefined(Main, :cebl_forcing) && Main.cebl_forcing) ? Dict{Int, Block}(id => b for (id, b) in blocks if id >= 5) : blocks
                        deschamps_f1_val, deschamps_flowx_val = Update_deschamps_pipe_params(forcing_blocks, tt, 1, current_dt, MPI.COMM_WORLD, world_rank)
                    end
                end

                # HIT linear forcing: compute domain-averaged velocity
                if test_case == "HIT" || test_case == "MHDHIT"
                    hit_u_mean, hit_v_mean, hit_w_mean, hit_Bx_mean, hit_By_mean, hit_Bz_mean = Update_HIT_forcing(blocks, hit_forcing_A, MPI.COMM_WORLD, world_rank, tt, KRK)
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
            if strict_ct_positivity || KRK == 1 || multi_block_mode || length(blocks) != 1
                    # First RK stage, multi-block-per-rank, or any rank owning
                    # multiple local blocks (e.g. CEBL precursor + main inlet):
                    # use full blockAdvance so shared flux buffers cannot be
                    # overwritten by another local block's overlapped interior pass.
                    blockAdvance(b, current_dt, b.ϕ, shared_Fx, shared_Fy, shared_Fz, shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z, shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt, threads_recon_i, threads_recon_j, threads_recon_k, threads_visc_i, threads_visc_j, threads_visc_k, Int32(KRK), shared_ct_pos_meta, shared_ct_pos_values)
                else
                    # Stages 2-3: interior was launched during previous sync's MPI
                    # and only boundary work remains on the default stream.
                    @static if ct_emf_scheme == CT_EMF_WENO7_SG07
                        # Publish the interior point cache before boundary overwrite/scan.
                        gpu_stream_sync(compute_stream)
                    end
                    blockAdvance_boundary(b, current_dt, b.ϕ, shared_Fx, shared_Fy, shared_Fz, shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z, shared_Fvx, shared_Fvy, shared_Fvz, world_rank, tt, threads_recon_i, threads_recon_j, threads_recon_k, Int32(KRK), shared_ct_pos_meta, shared_ct_pos_values)
                end
                @static if strict_ct_positivity
                    gpu_sync()
                    ct_check_positivity_or_abort!(
                        shared_ct_pos_meta, shared_ct_pos_values;
                        rank=world_rank, block=bid, step=tt, rk_stage=KRK,
                    )
                end
                if profiling; gpu_sync(); t_advance += (time_ns() - _t0) / 1e9; end

                # ── CT: construct edge EMF from the current pre-divergence stage ──
                if equation_type == :MHD && ct_mode && b.Bx_face !== nothing
                    nb_ct = (cld(b.Nx + 1, nthreads[1]), cld(b.Ny + 1, nthreads[2]), cld(b.Nz + 1, nthreads[3]))
                    rk_a_ct = KRK == 2 ? FT(0.25) : (KRK == 3 ? FT(2)/FT(3) : one(FT))
                    if KRK == 1
                        ct_backup_face_b!(b, b.Nx, b.Ny, b.Nz)
                    end
                    local_periodic = (false, false, false)
                    if isdefined(Main, :Iperiodic)
                        local_periodic = ntuple(3) do direction
                            Main.Iperiodic[direction] &&
                                Block_Nprocs[bid + 1][direction] == 1
                        end
                    end
                    @static if ct_emf_scheme == CT_EMF_WENO7_SG07
                        point_scratch = _ct_weno7_point_scratch_ref[]
                        cache_i = _ct_weno7_cache_i_ref[]
                        cache_j = _ct_weno7_cache_j_ref[]
                        cache_k = _ct_weno7_cache_k_ref[]
                        weno7_fail_meta = _ct_weno7_fail_meta_ref[]
                        weno7_fail_value = _ct_weno7_fail_value_ref[]
                        ct_weno7_build_edge_line!(
                            b.Ex_edge, point_scratch, Val(1),
                            cache_i, cache_j, cache_k, b.x, b.y, b.z,
                            weno7_fail_meta, weno7_fail_value,
                            world_rank, bid, Int32(KRK),
                            b.Nx, b.Ny, b.Nz, local_periodic)
                        ct_weno7_build_edge_line!(
                            b.Ey_edge, point_scratch, Val(2),
                            cache_i, cache_j, cache_k, b.x, b.y, b.z,
                            weno7_fail_meta, weno7_fail_value,
                            world_rank, bid, Int32(KRK),
                            b.Nx, b.Ny, b.Nz, local_periodic)
                        ct_weno7_build_edge_line!(
                            b.Ez_edge, point_scratch, Val(3),
                            cache_i, cache_j, cache_k, b.x, b.y, b.z,
                            weno7_fail_meta, weno7_fail_value,
                            world_rank, bid, Int32(KRK),
                            b.Nx, b.Ny, b.Nz, local_periodic)
                    else
                        @gpu_launch threads=nthreads blocks=nb_ct ct_compute_edge_line_emf_kernel!(
                            b.Ex_edge, b.Ey_edge, b.Ez_edge,
                            shared_Fx, shared_Fy, shared_Fz,
                            shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                            b.U,
                            b.Areai, b.nxi, b.nyi, b.nzi,
                            b.Areaj, b.nxj, b.nyj, b.nzj,
                            b.Areak, b.nxk, b.nyk, b.nzk,
                            b.Vol, b.x, b.y, b.z, current_dt,
                            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    end
                    @static if resistive && ct_resistive_main_explicit
                        nb_cc = (
                            cld(b.Nx+2NG, nthreads[1]),
                            cld(b.Ny+2NG, nthreads[2]),
                            cld(b.Nz+2NG, nthreads[3]),
                        )
                        @gpu_launch threads=nthreads blocks=nb_cc ct_compute_resistive_cell_emf_kernel!(
                            shared_cc_ex, shared_cc_ey, shared_cc_ez, b.Q,
                            b.Areai, b.nxi, b.nyi, b.nzi,
                            b.Areaj, b.nxj, b.nyj, b.nzj,
                            b.Areak, b.nxk, b.nyk, b.nzk,
                            b.Vol, Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                        @gpu_launch threads=nthreads blocks=nb_ct ct_add_resistive_edge_line_emf_kernel!(
                            b.Ex_edge, b.Ey_edge, b.Ez_edge,
                            shared_Fvx, shared_Fvy, shared_Fvz,
                            shared_cc_ex, shared_cc_ey, shared_cc_ez,
                            b.Areai, b.nxi, b.nyi, b.nzi,
                            b.Areaj, b.nxj, b.nyj, b.nzj,
                            b.Areak, b.nxk, b.nyk, b.nzk,
                            b.x, b.y, b.z,
                            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    end
                    ct_enforce_physical_edge_emf!(b, bid, face_bc)
                    if isdefined(Main, :Iperiodic)
                        @gpu_launch threads=nthreads blocks=nb_ct ct_sync_periodic_edge_emf_kernel!(
                            b.Ex_edge, b.Ey_edge, b.Ez_edge,
                            local_periodic[1], local_periodic[2], local_periodic[3],
                            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    end
                end

                # ── Forcing ──
                if profiling; _t0 = time_ns(); end
                nb_l = (Int32(cld(b.Nx+2*NG, threads_light[1])), Int32(cld(b.Ny+2*NG, threads_light[2])), Int32(cld(b.Nz+2*NG, threads_light[3])))
                if flow_forcing
                    # Rotation is driven by BC_DIFFROT_WALL (wall rotation) when diffrot_enabled.
                    # Do NOT also apply a Coriolis+centrifugal volume force — that double-counts
                    # rotation and produces a spurious radial pressure jump at the precursor→main
                    # interface (the precursor has no rotation source). Only the legacy non-wall
                    # diffrot path (diffrot_enabled && !diffrot_wall_bc) uses the volume force.
                    b_omega_x = (b.id >= 5 || !_diffrot_volume_force_active) ? zero(FT) :
                                (isdefined(Main, :Omega_x) ? FT(Main.Omega_x) : zero(FT))
                    # Deschamps parameters are isolated to the precursor blocks only
                    f1_use = (b.id >= 5) ? deschamps_f1_val : zero(FT)
                    flowx_use = (b.id >= 5) ? deschamps_flowx_val : zero(FT)

                    if b_omega_x != zero(FT)
                        @gpu_launch threads=threads_light blocks=nb_l Volume_force_kernel!(shared_dU_forced, b.Q, b.y, b.z, b.Nx, b.Ny, b.Nz, b_omega_x)
                        if forcing_mode == 1
                            Apply_bulk_force!(shared_dU_forced, b.Q, forcex, flowx, current_dt, b.Nx, b.Ny, b.Nz)
                        elseif forcing_mode == 2
                            Apply_const_massflux_force!(shared_dU_forced, b.Q, cmf_f1_val, b.Nx, b.Ny, b.Nz)
                        elseif forcing_mode == 3 || (isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                            Apply_deschamps_pipe_force!(shared_dU_forced, b.Q, f1_use, flowx_use, current_dt, b.Nx, b.Ny, b.Nz)
                        end
                        # ── Fringe forcing (only when _fringe_active) ──
                        if _fringe_active
                            @gpu_launch threads=threads_light blocks=nb_l fringe_forcing_kernel!(
                                shared_dU_forced, b.U, b.U_target_fringe, b.fringe_lambda,
                                b.Nx, b.Ny, b.Nz, current_dt)
                        end
                        Apply_trip_force!(shared_dU_forced, b.Q, b.x, b.y, b.z, b.Nx, b.Ny, b.Nz, activeTime)
                        @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(b.U, shared_dU_forced, current_dt, b.Vol, b.Nx, b.Ny, b.Nz)
                    else
                        # ── Fused path (b_omega_x == 0): skip zero+rotation, directly update U ──
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
                        elseif forcing_mode == 3 || (isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                            # Fused: deschamps + add_source in 1 kernel (no dU_forced needed)
                            @gpu_launch threads=nthreads blocks=nb_f fused_deschamps_source_kernel!(b.U, b.Q, f1_use, flowx_use, current_dt, b.Nx, b.Ny, b.Nz)
                            # Trip forcing still needs dU_forced as intermediate (heavy sin/atan →keep separate)
                            @gpu_launch threads=nthreads blocks=nb_f zero_dU_forced_kernel!(shared_dU_forced, b.Nx, b.Ny, b.Nz)
                            Apply_trip_force!(shared_dU_forced, b.Q, b.x, b.y, b.z, b.Nx, b.Ny, b.Nz, activeTime)
                            @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(b.U, shared_dU_forced, current_dt, b.Vol, b.Nx, b.Ny, b.Nz)
                        end
                    end
                end
                if test_case == "HIT" || test_case == "MHDHIT"
                    Apply_HIT_forcing!(shared_dU_forced, b.Q, hit_forcing_A, hit_u_mean, hit_v_mean, hit_w_mean, hit_Bx_mean, hit_By_mean, hit_Bz_mean, b.Nx, b.Ny, b.Nz)
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
                    @static if !(strict_ct_positivity && ct_mode && splitMethodID == 4)
                        @check_nan(b.U, "U after divergence", b.id, world_rank, tt)
                    end
                    rk_a = KRK == 2 ? FT(0.25) : (KRK == 3 ? FT(2)/FT(3) : one(FT))
                    @gpu_launch threads=threads_light blocks=nb_l linComb_clip_prim(b.U, b.Un, b.Q, Ncons, rk_a, one(FT) - rk_a, b.Nx, b.Ny, b.Nz)
                else
                    # Fused: div + RK combination + clipping + c2Prim in one kernel
                    rk_a = KRK == 2 ? FT(0.25) : (KRK == 3 ? FT(2)/FT(3) : one(FT))
                    @gpu_launch threads=nthreads blocks=nb_f div_rk_clip_prim(b.U, b.Un, b.Q, shared_Fx, shared_Fy, shared_Fz, shared_Fvx, shared_Fvy, shared_Fvz, current_dt, b.Vol, rk_a, b.Nx, b.Ny, b.Nz)
                end
                @static if strict_ct_positivity && ct_mode && splitMethodID == 4
                    @gpu_launch threads=nthreads blocks=nb_f ct_check_cell_positivity_kernel!(
                        shared_ct_pos_meta, shared_ct_pos_values, b.U, FT(γ),
                        CT_POS_SITE_POST_HYDRO, Int32(0),
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    gpu_sync()
                    ct_check_positivity_or_abort!(
                        shared_ct_pos_meta, shared_ct_pos_values;
                        rank=world_rank, block=bid, step=tt, rk_stage=KRK,
                    )
                    @check_nan(b.U, "U after divergence/RK", b.id, world_rank, tt)
                end
                @check_nan(b.Q, "Q after div_rk_clip", b.id, world_rank, tt)

                if debug_sync && tt == 1 && world_rank == 0
                    U_cpu = Array(b.U)
                    Un_cpu = Array(b.Un)
                    Q_cpu = Array(b.Q)
                    Fy_cpu = Array(shared_Fy)
                    Fvy_cpu = Array(shared_Fvy)
                    Vol_cpu = Array(b.Vol)
                    ii = NG + 8
                    kk = NG + 8
                    jj = NG + 1
                    
                    fy_idx = min(6, Ncons)
                    fy_1 = Fy_cpu[ii-NG, jj-NG, kk-NG, fy_idx]
                    fy_2 = Fy_cpu[ii-NG, jj-NG+1, kk-NG, fy_idx]
                    fvy_1 = Fvy_cpu[ii-NG, jj-NG, kk-NG, fy_idx]
                    fvy_2 = Fvy_cpu[ii-NG, jj-NG+1, kk-NG, fy_idx]
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
                    
                    u_idx = min(6, Ncons)
                    U_before = Un_cpu[ii, jj, kk, u_idx]
                    div_term = (fy_1 - fy_2) - (fvy_1 - fvy_2)
                    U_predicted_temp = U_before + div_term * current_dt * vol
                    U_predicted = U_before + rk_a * (U_predicted_temp - U_before)
                    
                    println("  U_before[$u_idx] (Un) = ", U_before)
                    println("  U_after_kernel[$u_idx] = ", U_cpu[ii, jj, kk, u_idx])
                    println("  U_predicted[$u_idx]    = ", U_predicted)
                    if equation_type == :MHD && Nprim >= 7
                        println("  Q_after_kernel[7] (Bx) = ", Q_cpu[ii, jj, kk, 7])
                    end
                    println("="^50)
                end


                # ── MHD: GLM ψ-damping source term (operator splitting) ──
                # Skip GLM when CT is active (CT handles ∇·B, ψ is unused)
                if equation_type == :MHD && !ct_mode
                    nb_glm = (cld(b.Nx, nthreads[1]), cld(b.Ny, nthreads[2]), cld(b.Nz, nthreads[3]))
                    @gpu_launch threads=nthreads blocks=nb_glm glm_source_kernel!(b.U, current_dt, ch_glm_current, cr_glm, b.Nx, b.Ny, b.Nz)
                    # Update ψ in Q as well (Q[10] = U[9])
                    # This will be done by the next c2Prim / fillGhost cycle
                end

                # ── Outlet sponge: operator-split exponential relaxation ──
                # Applied once per RK substep to every block, regardless of fused/non-fused
                # forcing path (mirrors the GLM ψ-damping operator-split above).
                if _outlet_sponge_active && b.sponge_sigma !== nothing
                    nb_sp = (cld(b.Nx, nthreads[1]), cld(b.Ny, nthreads[2]), cld(b.Nz, nthreads[3]))
                    @gpu_launch threads=nthreads blocks=nb_sp sponge_step_kernel!(
                        b.U, b.sponge_U_target, b.sponge_sigma, current_dt, b.Nx, b.Ny, b.Nz)
                end
                if profiling; gpu_sync(); t_div += (time_ns() - _t0) / 1e9; end
            end

            @static if strict_ct_positivity && ct_mode
                fallback_counts = ct_fallback_counts(shared_ct_pos_meta)
                global_weno_to_plm = MPI.Allreduce(
                    fallback_counts.weno_to_plm, MPI.SUM, MPI.COMM_WORLD,
                )
                global_plm_to_first = MPI.Allreduce(
                    fallback_counts.plm_to_first, MPI.SUM, MPI.COMM_WORLD,
                )
                global_hlld_to_hlle = MPI.Allreduce(
                    fallback_counts.hlld_to_hlle, MPI.SUM, MPI.COMM_WORLD,
                )
                if world_rank == 0 &&
                   global_weno_to_plm + global_plm_to_first +
                   global_hlld_to_hlle > 0
                    @printf(
                        "CT_FALLBACK step=%d rk=%d weno_to_plm=%d plm_to_first=%d hlld_to_hlle=%d\n",
                        tt, KRK, global_weno_to_plm,
                        global_plm_to_first, global_hlld_to_hlle,
                    )
                end
            end

            # All blocks must retain their edge EMFs until shared physical edges
            # have been canonicalized. The communication step is inserted here;
            # only then may any block apply its discrete-Stokes face update.
            if equation_type == :MHD && ct_mode
                ct_sync_rank_sheets!(
                    blocks, block_comms, Block_Nprocs;
                    sync_face_flux=false, sync_edges=true,
                )
                ct_sync_interface_sheets!(
                    blocks, ct_sync_plan; sync_face_flux=false, sync_edges=true,
                )
                ct_sync_junction_edges!(blocks, ct_junction_plan)
                for (bid, b) in blocks
                    if b.Bx_face === nothing
                        continue
                    end
                    nb_ct = (cld(b.Nx + 1, nthreads[1]), cld(b.Ny + 1, nthreads[2]), cld(b.Nz + 1, nthreads[3]))
                    nb_f = (cld(b.Nx, nthreads[1]), cld(b.Ny, nthreads[2]), cld(b.Nz, nthreads[3]))
                    rk_a_ct = KRK == 2 ? FT(0.25) : (KRK == 3 ? FT(2)/FT(3) : one(FT))
                    @gpu_launch threads=nthreads blocks=nb_ct ct_update_face_b_from_emf_kernel!(
                        b.Bx_face, b.By_face, b.Bz_face,
                        b.Bx_face_n, b.By_face_n, b.Bz_face_n,
                        b.Ex_edge, b.Ey_edge, b.Ez_edge,
                        current_dt, rk_a_ct,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    @static if strict_ct_positivity && ct_mode && splitMethodID == 4
                        @gpu_launch threads=nthreads blocks=nb_f ct_check_sync_transition_kernel!(
                            shared_ct_pos_meta, shared_ct_pos_values,
                            b.U, b.Bx_face, b.By_face, b.Bz_face,
                            b.Areai, b.nxi, b.nyi, b.nzi,
                            b.Areaj, b.nxj, b.nyj, b.nzj,
                            b.Areak, b.nxk, b.nyk, b.nzk, FT(γ),
                            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                        gpu_sync()
                        ct_check_positivity_or_abort!(
                            shared_ct_pos_meta, shared_ct_pos_values;
                            rank=world_rank, block=bid, step=tt, rk_stage=KRK,
                        )
                    end
                    ct_sync_cell_b_lsq2!(b, b.Nx, b.Ny, b.Nz)
                    ct_fill_zerograd_face_b!(b, b.Nx, b.Ny, b.Nz)
                    if isdefined(Main, :Iperiodic)
                        local_periodic = ntuple(3) do direction
                            Main.Iperiodic[direction] &&
                                Block_Nprocs[bid + 1][direction] == 1
                        end
                        ct_fill_periodic_face_b!(
                            b, b.Nx, b.Ny, b.Nz, local_periodic,
                        )
                    end
                end
                ct_sync_rank_face_halos!(blocks, block_comms, Block_Nprocs)
                ct_sync_interface_face_halos!(blocks, ct_sync_plan)
                sync_ct_point_state!(sync_q_ghosts=false)
                @static if strict_ct_positivity && ct_mode && splitMethodID == 4
                    for (bid, b) in blocks
                        if b.Bx_face === nothing
                            continue
                        end
                        nb_f = (cld(b.Nx, nthreads[1]), cld(b.Ny, nthreads[2]), cld(b.Nz, nthreads[3]))
                        @gpu_launch threads=nthreads blocks=nb_f ct_check_cell_positivity_kernel!(
                            shared_ct_pos_meta, shared_ct_pos_values, b.U, FT(γ),
                            CT_POS_SITE_POST_CT_SYNC, Int32(0),
                            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                        gpu_sync()
                        ct_check_positivity_or_abort!(
                            shared_ct_pos_meta, shared_ct_pos_values;
                            rank=world_rank, block=bid, step=tt, rk_stage=KRK,
                        )
                    end
                end
            end

            # ── Sync blocks with comm-compute overlap (Phase 2) ──
            if profiling; _t0 = time_ns(); end
            if KRK < 3 && !strict_ct_positivity && !multi_block_mode && length(blocks) == 1
                # Stages 1-2: launch NEXT stage's interior recon during MPI.
                # Safe only when this rank owns a single local block; otherwise
                # shared_Fx/Fy/Fz/Fv* would be overwritten across local blocks.
                _ol = Ref(false)
                sync_blocks!(activeTime; step=tt, rk_stage=KRK, compute_fn = function(slot::Int)
                    if !_ol[]
                        for (bid, b) in blocks
                            blockAdvance_interior(b, current_dt, b.ϕ,
                                shared_Fx, shared_Fy, shared_Fz,
                                shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                                shared_Fvx, shared_Fvy, shared_Fvz,
                                threads_recon_i, threads_recon_j, threads_recon_k,
                                threads_visc_i, threads_visc_j, threads_visc_k, compute_stream,
                                Int32(KRK + 1), shared_ct_pos_meta, shared_ct_pos_values)
                        end
                        _ol[] = true
                    end
                end)
            else
                # Last stage or multi-block: plain sync (no next stage to overlap)
                sync_blocks!(activeTime; step=tt, rk_stage=KRK)
            end
            if profiling; gpu_sync(); t_sync += (time_ns() - _t0) / 1e9; end
        end

        @static if ct_resistive_sts_active
            explicit_dt_local = FT(Inf)
            for (bid, b) in blocks
                fill!(b.LTS_dt, zero(FT))
                nb_sts_dt = (
                    cld(b.Nx, nthreads[1]),
                    cld(b.Ny, nthreads[2]),
                    cld(b.Nz, nthreads[3]),
                )
                @gpu_launch threads=nthreads blocks=nb_sts_dt ct_resistive_explicit_dt_kernel!(
                    b.LTS_dt, b.Vol, b.Areai, b.Areaj, b.Areak,
                    Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                block_dt = FT(mapreduce(
                    value -> value > zero(FT) ? value : FT(Inf),
                    min, b.LTS_dt,
                ))
                explicit_dt_local = min(explicit_dt_local, block_dt)
            end
            explicit_resistive_dt = ct_sts_safety * MPI.Allreduce(
                explicit_dt_local, MPI.MIN, MPI.COMM_WORLD,
            )
            sts_substeps = ct_sts_substeps(
                FT(current_dt), FT(explicit_resistive_dt);
                damping=FT(ct_sts_damping), max_stages=ct_sts_max_stages,
            )
            if world_rank == 0 && (tt <= 2 || length(sts_substeps) > 1)
                @printf(
                    "CT_STS step=%d stages=%d dt=%.8e explicit_dt=%.8e\n",
                    tt, length(sts_substeps), current_dt,
                    explicit_resistive_dt,
                )
            end

            sts_elapsed = zero(FT)
            for (sts_stage, sts_dt) in enumerate(sts_substeps)
                @static if strict_ct_positivity
                    ct_reset_positivity!(
                        shared_ct_pos_meta, shared_ct_pos_values,
                    )
                end
                for (bid, b) in blocks
                    nb_cc = (
                        cld(b.Nx+2NG, nthreads[1]),
                        cld(b.Ny+2NG, nthreads[2]),
                        cld(b.Nz+2NG, nthreads[3]),
                    )
                    nb_ct = (
                        cld(b.Nx+1, nthreads[1]),
                        cld(b.Ny+1, nthreads[2]),
                        cld(b.Nz+1, nthreads[3]),
                    )
                    nb_cell = (
                        cld(b.Nx, nthreads[1]),
                        cld(b.Ny, nthreads[2]),
                        cld(b.Nz, nthreads[3]),
                    )
                    fill!(b.Ex_edge, zero(FT))
                    fill!(b.Ey_edge, zero(FT))
                    fill!(b.Ez_edge, zero(FT))
                    @gpu_launch threads=nthreads blocks=nb_cc ct_compute_resistive_cell_emf_kernel!(
                        shared_cc_ex, shared_cc_ey, shared_cc_ez, b.Q,
                        b.Areai, b.nxi, b.nyi, b.nzi,
                        b.Areaj, b.nxj, b.nyj, b.nzj,
                        b.Areak, b.nxk, b.nyk, b.nzk,
                        b.Vol, Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    @gpu_launch threads=nthreads blocks=nb_cc ct_resistive_face_flux_i_kernel!(
                        b.Q, shared_Fvx,
                        shared_cc_ex, shared_cc_ey, shared_cc_ez,
                        b.Areai, b.nxi, b.nyi, b.nzi,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    @gpu_launch threads=nthreads blocks=nb_cc ct_resistive_face_flux_j_kernel!(
                        b.Q, shared_Fvy,
                        shared_cc_ex, shared_cc_ey, shared_cc_ez,
                        b.Areaj, b.nxj, b.nyj, b.nzj,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    @gpu_launch threads=nthreads blocks=nb_cc ct_resistive_face_flux_k_kernel!(
                        b.Q, shared_Fvz,
                        shared_cc_ex, shared_cc_ey, shared_cc_ez,
                        b.Areak, b.nxk, b.nyk, b.nzk,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    @gpu_launch threads=nthreads blocks=nb_ct ct_add_resistive_edge_line_emf_kernel!(
                        b.Ex_edge, b.Ey_edge, b.Ez_edge,
                        shared_Fvx, shared_Fvy, shared_Fvz,
                        shared_cc_ex, shared_cc_ey, shared_cc_ez,
                        b.Areai, b.nxi, b.nyi, b.nzi,
                        b.Areaj, b.nxj, b.nyj, b.nzj,
                        b.Areak, b.nxk, b.nyk, b.nzk,
                        b.x, b.y, b.z,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    @gpu_launch threads=nthreads blocks=nb_cell ct_resistive_energy_euler_kernel!(
                        b.U, shared_Fvx, shared_Fvy, shared_Fvz, b.Vol,
                        FT(sts_dt), Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    ct_enforce_physical_edge_emf!(b, bid, face_bc)
                    local_periodic = ntuple(3) do direction
                        isdefined(Main, :Iperiodic) &&
                            Main.Iperiodic[direction] &&
                            Block_Nprocs[bid + 1][direction] == 1
                    end
                    @gpu_launch threads=nthreads blocks=nb_ct ct_sync_periodic_edge_emf_kernel!(
                        b.Ex_edge, b.Ey_edge, b.Ez_edge,
                        local_periodic[1], local_periodic[2],
                        local_periodic[3],
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                end

                ct_sync_rank_sheets!(
                    blocks, block_comms, Block_Nprocs;
                    sync_face_flux=false, sync_edges=true,
                )
                ct_sync_interface_sheets!(
                    blocks, ct_sync_plan;
                    sync_face_flux=false, sync_edges=true,
                )
                ct_sync_junction_edges!(blocks, ct_junction_plan)
                for (bid, b) in blocks
                    nb_ct = (
                        cld(b.Nx+1, nthreads[1]),
                        cld(b.Ny+1, nthreads[2]),
                        cld(b.Nz+1, nthreads[3]),
                    )
                    ct_backup_face_b!(b, b.Nx, b.Ny, b.Nz)
                    @gpu_launch threads=nthreads blocks=nb_ct ct_update_face_b_from_emf_kernel!(
                        b.Bx_face, b.By_face, b.Bz_face,
                        b.Bx_face_n, b.By_face_n, b.Bz_face_n,
                        b.Ex_edge, b.Ey_edge, b.Ez_edge,
                        FT(sts_dt), one(FT),
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    ct_sync_cell_b_lsq2!(b, b.Nx, b.Ny, b.Nz)
                    ct_fill_zerograd_face_b!(b, b.Nx, b.Ny, b.Nz)
                    local_periodic = ntuple(3) do direction
                        isdefined(Main, :Iperiodic) &&
                            Main.Iperiodic[direction] &&
                            Block_Nprocs[bid + 1][direction] == 1
                    end
                    ct_fill_periodic_face_b!(
                        b, b.Nx, b.Ny, b.Nz, local_periodic,
                    )
                end
                ct_sync_rank_face_halos!(blocks, block_comms, Block_Nprocs)
                ct_sync_interface_face_halos!(blocks, ct_sync_plan)
                sync_ct_point_state!(sync_q_ghosts=false)
                sts_elapsed += FT(sts_dt)
                sync_blocks!(
                    activeTime + sts_elapsed;
                    step=tt, rk_stage=3+sts_stage,
                )
                @static if strict_ct_positivity
                    for (bid, b) in blocks
                        nb_cell = (
                            cld(b.Nx, nthreads[1]),
                            cld(b.Ny, nthreads[2]),
                            cld(b.Nz, nthreads[3]),
                        )
                        @gpu_launch threads=nthreads blocks=nb_cell ct_check_cell_positivity_kernel!(
                            shared_ct_pos_meta, shared_ct_pos_values,
                            b.U, FT(γ), CT_POS_SITE_POST_CT_SYNC,
                            Int32(0), Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                    end
                    gpu_sync()
                    ct_check_positivity_or_abort!(
                        shared_ct_pos_meta, shared_ct_pos_values;
                        rank=world_rank, block=-1, step=tt,
                        rk_stage=3+sts_stage,
                    )
                end
            end
        end
        @static if ct_resistive_rkl2_active
            advance_resistive_ct_rkl2!(
                FT(0.5)*FT(current_dt),
                FT(activeTime) + FT(0.5)*FT(current_dt);
                step=tt, split_half=2,
            )
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
        # -- Outlet NSCBC pressure anchor + sponge target update --
        # Every N_anchor steps: compute the area-weighted outlet mean pressure for
        # each main block (ξ+ face) and write it to bc_params[(bid,2)][BCP_OUTLET_PAVG]
        # as the local weak anchor for NSCBC L1. Also refresh the running average
        # sponge target U_target (Mani-style perturbation damping).
        if _outlet_sponge_active
            N_anchor = isdefined(Main, :sponge_anchor_period) ? Main.sponge_anchor_period : 50
            N_updt   = isdefined(Main, :sponge_target_period) ? Main.sponge_target_period : 20
            if tt % N_anchor == 0
                _use_fixed_p = isdefined(Main, :p_back) && Main.p_back > 0
                for (bid, b) in blocks
                    if _use_fixed_p
                        # Fixed back pressure: p_back already injected at init,
                        # don't overwrite with self-tracking outlet mean.
                        continue
                    end
                    # Only the ξ-tail-rank of each main block owns the physical outlet
                    # face; others contribute zero and Allreduce merges them.
                    Nprocs_b = Block_Nprocs[bid+1]
                    rx_b = b.rx + 1; ry_b = b.ry + 1; rz_b = b.rz + 1
                    if rx_b == Nprocs_b[1] && haskey(bc_params, (bid, 2)) &&
                       (get(face_bc, (bid, 2), -1) == Int32(BC_NSCBC_OUTFLOW) ||
                       get(face_bc, (bid, 2), -1) == Int32(BC_RIEMANN_OUTFLOW))
                        # Outlet face at i = b.Nx + NG (last interior), area Areai at i+1
                        ix_out = b.Nx + NG
                        p_slice = Array(@view b.Q[ix_out, NG+1:b.Ny+NG, NG+1:b.Nz+NG, 5])
                        A_slice = Array(@view b.Areai[ix_out+1, NG+1:b.Ny+NG, NG+1:b.Nz+NG])
                        num_loc = sum(p_slice .* A_slice)
                        den_loc = sum(A_slice)
                    else
                        num_loc = zero(FT); den_loc = zero(FT)
                    end
                    _sub = get(block_comms, bid, nothing)
                    if _sub === nothing
                        num_g = Float64(num_loc); den_g = Float64(den_loc)
                    else
                        num_g = MPI.Allreduce(Float64(num_loc), MPI.SUM, _sub)
                        den_g = MPI.Allreduce(Float64(den_loc), MPI.SUM, _sub)
                    end
                    p_avg = den_g > 0.0 ? FT(num_g / den_g) : zero(FT)
                    if haskey(bc_params, (bid, 2))
                        bcp_old = bc_params[(bid, 2)]
                        bcp_new = Base.setindex(bcp_old, p_avg, BCP_OUTLET_PAVG)
                        bc_params[(bid, 2)] = bcp_new
                    end
                end
            end
            if tt % N_updt == 0
                for (bid, b) in blocks
                    if b.sponge_sigma === nothing; continue; end
                    nb_sp = (cld(b.Nx + 2*NG, nthreads[1]),
                             cld(b.Ny + 2*NG, nthreads[2]),
                             cld(b.Nz + 2*NG, nthreads[3]))
                    α = isdefined(Main, :sponge_target_alpha) ? Main.sponge_target_alpha : FT(0.01)
                    @gpu_launch threads=nthreads blocks=nb_sp update_sponge_U_target_kernel!(
                        b.sponge_U_target, b.U, b.sponge_sigma, α, b.Nx, b.Ny, b.Nz)
                end
            end
        end

        # -- Fringe recovery residual (collective: all ranks must participate) --
        _fringe_resid = -1.0
        if _fringe_active && (tt % 100 == 0 || tt <= 20)
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

        # -- Entropy diagnostic (collective; runs every 100 steps after
        #    the first 20 warm-up steps). Useful sanity check: ⟨η⟩ should
        #    be non-increasing modulo boundary outflow / forcing input.
        _η_diag::Float64 = NaN
        if tt % 100 == 0 || tt <= 20
            _η_local = 0.0
            for (_bid, b) in blocks
                nxp_b, nyp_b, nzp_b = b.Nx, b.Ny, b.Nz
                # Per-block scratch buffer. Allocated each invocation;
                # invocation cadence is every step for tt ≤ 20 (warm-up
                # diagnostic) and every 100 steps thereafter — the
                # transient buffers are reclaimed by the next GC pass.
                η_block = similar(b.U, FT, (nxp_b + 2*NG, nyp_b + 2*NG, nzp_b + 2*NG))
                fill!(η_block, zero(FT))
                # Use a generic 8×8×4 thread-block; the kernel handles
                # ghost-skip internally so launch covers the full padded extent.
                _thr = (Int32(8), Int32(8), Int32(4))
                _nb  = (cld(Int32(nxp_b + 2*NG), _thr[1]),
                        cld(Int32(nyp_b + 2*NG), _thr[2]),
                        cld(Int32(nzp_b + 2*NG), _thr[3]))
                @gpu_launch threads=_thr blocks=_nb entropy_total_kernel!(
                    η_block, b.U, b.Vol, FT(γ),
                    Int32(NG), Int32(nxp_b), Int32(nyp_b), Int32(nzp_b))
                _η_local += Float64(sum(η_block))
            end
            _η_diag = MPI.Allreduce(_η_local, MPI.SUM, MPI.COMM_WORLD)
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

            # Print entropy diagnostic if measured this step.
            # _η0_global is captured on the first measurement (the loop
            # increments tt to 1 before this block runs, so the first
            # capture is typically at tt=1) and persisted thereafter, so
            # all later prints show drift relative to that baseline.
            if isfinite(_η_diag)
                if !isfinite(_η0_global)
                    _η0_global = _η_diag
                end
                @printf("  ⟨η⟩ = %+.6e   Δ⟨η⟩ = %+.6e\n",
                        _η_diag, _η_diag - _η0_global)
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

        # ─── In-situ post-processing hook ───
        if isdefined(Main, :in_situ_post_process)
            in_situ_post_process(tt, activeTime, current_dt, blocks, world_rank, Block_Nprocs, block_comms)
        end

        activeTime += current_dt
    end
    if world_rank == 0
        @printf(">>> Loop exited: activeTime=%.6e (limit=%.2f), tt=%d (maxStep=%d)\n", activeTime, max_time_limit, tt, maxStep)
        printstyled("Done!\n", color=:green)
        flush(stdout)
    end
    if init_rpo_mode
        saveRpoFile(Main.save_rpo_path, activeTime, blocks, world_rank, Block_Nprocs, block_comms)
    else
        plotFile_multiblock(tt, activeTime, blocks, world_rank, Nblocks, Block_Nprocs, block_comms)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    return blocks, activeTime, tt
end
