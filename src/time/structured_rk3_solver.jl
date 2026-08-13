using MPI
using StaticArrays
using HDF5, DelimitedFiles
using Dates, Printf

include(joinpath(@__DIR__, "structured_task_graph.jl"))
include(joinpath(@__DIR__, "structured_task_context.jl"))
include(joinpath(@__DIR__, "structured_task_catalog.jl"))

# GPU backend: loads CUDA or AMDGPU depending on what's available
include(joinpath(@__DIR__,"..","parallel","gpu_backend.jl"))
include(joinpath(@__DIR__,"..","core","backend_interface.jl"))
gpu_allowscalar(false)

include(joinpath(@__DIR__,"..","core","boundary_types.jl"))
include(joinpath(@__DIR__,"..","core","numerical_config.jl"))
include(joinpath(@__DIR__,"..","core","structured_interface_transform.jl"))
include(joinpath(@__DIR__,"..","physics","euler_flux.jl"))
include(joinpath(@__DIR__,"..","physics","viscous_flux.jl"))
include(joinpath(@__DIR__,"..","physics","structured_boundary.jl"))
const default_dsrfg_params = create_dummy_dsrfg_params(FT)
include(joinpath(@__DIR__,"..","core","state_conversion.jl"))
include(joinpath(@__DIR__,"structured_filter_step.jl"))
include(joinpath(@__DIR__,"..","io","post_process.jl"))

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

include(joinpath(@__DIR__,"..","numerics","structured_divergence.jl"))
include(joinpath(@__DIR__,"..","parallel","structured_mpi.jl"))
include(joinpath(@__DIR__,"..","io","structured_io.jl"))
include(joinpath(@__DIR__,"..","parallel","ct_derivation_halo.jl"))
include(joinpath(@__DIR__,"..","numerics","ct_state.jl"))
include(joinpath(@__DIR__,"..","numerics","structured_face_quadrature.jl"))
include(joinpath(@__DIR__,"..","numerics","structured_initial_quadrature.jl"))
include(joinpath(@__DIR__,"..","numerics","ct_positivity.jl"))
include(joinpath(@__DIR__,"..","numerics","riemann_solvers.jl"))


# Topology-based multi-block edge singularity detection + adaptive WENO5
# ϕ boost. See docs/superpowers/specs/2026-06-20-topology-singularity-protection.md.

include(joinpath(@__DIR__,"..","numerics","weno7_reconstruction.jl"))
include(joinpath(@__DIR__,"..","numerics","ct_weno7.jl"))
include(joinpath(@__DIR__,"..","numerics","structured_reconstruction.jl"))
include(joinpath(@__DIR__,"volume_sources.jl"))


include(joinpath(@__DIR__,"..","mesh","structured_ghost_coordinates.jl"))
include(joinpath(@__DIR__,"..","parallel","auto_tune.jl"))
include(joinpath(@__DIR__,"..","numerics","filter_interface.jl"))
include(joinpath(@__DIR__,"..","physics","fringe_source.jl"))
include(joinpath(@__DIR__,"sponge_source.jl"))
include(joinpath(@__DIR__,"split_source_integration.jl"))
include(joinpath(@__DIR__,"ct_sts.jl"))
include(joinpath(@__DIR__,"ct_rkl2.jl"))
include(joinpath(@__DIR__,"..","numerics","constrained_transport.jl"))
include(joinpath(@__DIR__,"..","numerics","ct_energy_budget.jl"))
include(joinpath(@__DIR__,"..","physics","external_magnetic_field.jl"))
include(joinpath(@__DIR__,"..","parallel","ct_sync.jl"))
include(joinpath(@__DIR__,"..","parallel","ct_derivation_halo_runtime.jl"))

if !@isdefined(interface_filter_enabled)
    const interface_filter_enabled::Bool = true
end

# Read-only CT energy-budget diagnostics. The flag is evaluated when the
# solver include chain is loaded so disabled runs retain the original path.
const _ct_energy_budget_enabled = equation_type == :MHD && ct_mode &&
    lowercase(get(ENV, "OPENCFD_CT_ENERGY_BUDGET", "false")) in
    ("1", "true", "yes", "on")


function build_structured_external_field_caches(
    host_module::Module, blocks, face_bc, bc_params, Block_Nprocs,
    world_rank::Integer,
)
    caches = Dict{Int,Any}()
    enabled = isdefined(host_module, :external_magnetic_field) &&
        Bool(getfield(host_module, :external_magnetic_field))
    enabled || return caches

    equation_type == :MHD && ct_mode || throw(ArgumentError(
        "external magnetic-field caching requires structured CT-MHD",
    ))

    start_ns = time_ns()
    local_bytes = Int64(0)
    for (bid, block) in blocks
        boundary_types, boundary_parameters =
            _structured_face_boundary_data(face_bc, bc_params, bid)
        nprocs = ntuple(
            direction -> Int32(Block_Nprocs[bid + 1][direction]), Val(3),
        )
        structured_external_field_cache_required(
            boundary_types, block.rx, block.ry, block.rz, nprocs,
        ) || continue

        cache = gpu_zeros(
            FT, block.Nx + 2*NG, block.Ny + 2*NG,
            block.Nz + 2*NG, 3,
        )
        nb = (
            cld(block.Nx + 2*NG, nthreads[1]),
            cld(block.Ny + 2*NG, nthreads[2]),
            cld(block.Nz + 2*NG, nthreads[3]),
        )
        bc_x_lo, bc_x_hi, bc_y_lo, bc_y_hi, bc_z_lo, bc_z_hi =
            boundary_types
        bcp_x_lo, bcp_x_hi, bcp_y_lo, bcp_y_hi, bcp_z_lo, bcp_z_hi =
            boundary_parameters
        @gpu_launch threads=nthreads blocks=nb precompute_finite_solenoid_external_field_cache_kernel!(
            cache, block.x, block.y, block.z,
            Int32(block.rx), Int32(block.ry), Int32(block.rz),
            Int32(block.Nx), Int32(block.Ny), Int32(block.Nz), nprocs,
            bc_x_lo, bc_x_hi, bc_y_lo, bc_y_hi, bc_z_lo, bc_z_hi,
            bcp_x_lo, bcp_x_hi, bcp_y_lo, bcp_y_hi, bcp_z_lo, bcp_z_hi,
        )
        caches[bid] = cache
        local_bytes += Int64(length(cache) * sizeof(FT))
    end
    isempty(caches) || gpu_sync()

    elapsed = (time_ns() - start_ns) / 1.0e9
    global_count = MPI.Allreduce(Int64(length(caches)), MPI.SUM, MPI.COMM_WORLD)
    global_bytes = MPI.Allreduce(local_bytes, MPI.SUM, MPI.COMM_WORLD)
    max_elapsed = MPI.Allreduce(elapsed, MPI.MAX, MPI.COMM_WORLD)
    if world_rank == 0
        println(
            ">>> External-field ghost cache: $global_count local block partitions, " *
            "$(round(global_bytes / 1024^2; digits=2)) MiB total, " *
            "precomputed in $(round(max_elapsed; digits=3)) s",
        )
    end
    return caches
end

# ── GLM cleaning speed for MHD (updated each time step) ──
# Auto-computed from max fast magnetosonic speed
const _ch_glm_ref = Ref{FT}(one(FT))
ch_glm_current::FT = one(FT)  # will be overwritten in time_step loop

# Rank-shared WENO7 buffers are allocated by time_step and referenced by the
# reconstruction launch helpers, including implicit face-flux callers.
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
# Adiabatic runs use the scaled mathematical entropy
# -rho*(log(p)-gamma*log(rho))*Vol. Isothermal MHD instead uses its convex
# free-energy entropy: (carrier-p+c_s^2*rho*log(rho/rho_ref))*Vol.
# The caller performs the global sum and MPI reduction.
# Runs every 100 steps for all flow configurations.
function entropy_total_kernel!(
    entropy_out, U, Vol, gamma_local::FT, density_reference::FT,
    NG_::Int32, Nx_::Int32, Ny_::Int32, Nz_::Int32,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i <= NG_ || i > Nx_ + NG_; return; end
    if j <= NG_ || j > Ny_ + NG_; return; end
    if k <= NG_ || k > Nz_ + NG_; return; end
    @inbounds begin
        rho = U[i, j, k, 1]
        momentum_x = U[i, j, k, 2]
        momentum_y = U[i, j, k, 3]
        momentum_z = U[i, j, k, 4]
        carrier = U[i, j, k, 5]
        @static if equation_type == :MHD && isothermal_mhd
            sound_speed_squared = Rg * isothermal_temperature
            pressure = sound_speed_squared * rho
            if rho > zero(FT) && isfinite(rho) && isfinite(carrier) &&
               sound_speed_squared > zero(FT) &&
               isfinite(sound_speed_squared) && isfinite(pressure) &&
               density_reference > zero(FT) && isfinite(density_reference)
                entropy_density = mhd_isothermal_entropy_density(
                    rho, carrier, sound_speed_squared, density_reference,
                )
                entropy_out[i, j, k] = isfinite(entropy_density) ?
                    entropy_density * Vol[i, j, k] : FT(NaN)
            else
                entropy_out[i, j, k] = FT(NaN)
            end
        else
            velocity_x = momentum_x / rho
            velocity_y = momentum_y / rho
            velocity_z = momentum_z / rho
            kinetic_specific = FT(0.5) * (
                velocity_x*velocity_x + velocity_y*velocity_y +
                velocity_z*velocity_z
            )
            pressure = (gamma_local - one(FT)) * (
                carrier - rho*kinetic_specific
            )
            if pressure > zero(FT) && rho > zero(FT)
                specific_entropy =
                    log(pressure) - gamma_local * log(rho)
                entropy_out[i, j, k] =
                    -rho * specific_entropy * Vol[i, j, k]
            else
                entropy_out[i, j, k] = FT(NaN)
            end
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
    transform::Union{Nothing,StructuredFaceTransform}
    # Optional physical image shift applied when this connection crosses a
    # periodic image.  Keeping it in the topology record makes coordinate
    # ghost construction independent of the order in which faces are crossed.
    image_translation::NTuple{3,Float64}
end

Connectivity(src_b::Int, src_f::Int, reverse_tan::Bool, flip_normal::Bool) =
    Connectivity(src_b, src_f, reverse_tan, flip_normal, nothing,
                 (0.0, 0.0, 0.0))

Connectivity(src_b::Int, src_f::Int, reverse_tan::Bool, flip_normal::Bool,
             transform::Union{Nothing,StructuredFaceTransform}) =
    Connectivity(src_b, src_f, reverse_tan, flip_normal, transform,
                 (0.0, 0.0, 0.0))

@inline function _connectivity_transform(local_face::Integer, conn::Connectivity)
    return conn.transform === nothing ?
        structured_legacy_face_transform(local_face, conn.src_f, conn.reverse_tan) :
        conn.transform
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
    # Optional fixed background face fluxes for CT B=B0+b splitting. These
    # are allocated only for CT-MHD runs with the split explicitly enabled.
    B0x_face::Union{GPUArray{FT, 3}, Nothing}
    B0y_face::Union{GPUArray{FT, 3}, Nothing}
    B0z_face::Union{GPUArray{FT, 3}, Nothing}
    # Cell-centered background cache used only by split-aware reconstruction.
    # It is fixed after initialization and absent from ordinary CT/GLM runs.
    B0_cell::Union{GPUArray{FT, 4}, Nothing}
    # CT edge line-integrated EMFs. These must persist across the per-block
    # reconstruction loop so shared physical edges can be synchronized before
    # any face receives its discrete-Stokes update.
    Ex_edge::Union{GPUArray{FT, 3}, Nothing}
    Ey_edge::Union{GPUArray{FT, 3}, Nothing}
    Ez_edge::Union{GPUArray{FT, 3}, Nothing}
    # Athena-style first-order flux correction (FOFC) coefficients. Channel 1
    # is the synchronized committed value; channel 2 is a local Jacobi
    # proposal. The existing one-component halo exchange transfers only the
    # committed channel, so no additional persistent MPI buffer is required.
    fofc_flag::Union{GPUArray{FT, 4}, Nothing}
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
    U_target_fringe::Union{GPUArray{FT, 4}, Nothing}    # (Nx_tot, Ny_tot, Nz_tot, Ncell_cons)
    # ─── Outlet sponge buffer (differential rotation, nothing when not active) ───
    sponge_sigma::Union{GPUArray{FT, 3}, Nothing}       # (Nx_tot, Ny_tot, Nz_tot)
    sponge_U_target::Union{GPUArray{FT, 4}, Nothing}    # (Nx_tot, Ny_tot, Nz_tot, Ncell_cons)
end

include(joinpath(@__DIR__,"implicit_solver.jl"))

function load_block(bid, rx, ry, rz, NG, Ncons, Nprim, Nprocs_block, world_rank,
                    face_bc, connectivity, rank_offsets, temp_metrics_h,
                    was_computed_h, temp_metric_auxiliary_h)
    _mesh_base = isdefined(Main, :mesh_dir) ? mesh_dir : "MESH"
    mesh_path = joinpath(_mesh_base, "mesh_b$bid.h5")
    
    Nx_val, Ny_val, Nz_val, coords_real = read_structured_mesh_file(mesh_path)

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
        cell_offsets=(ox, oy, oz),
        block_dims=(Nx_val, Ny_val, Nz_val),
    )
    
    # Compute (or load cached) metrics from expanded coordinates
    cache_path, was_computed, Areai_h, nxi_h, nyi_h, nzi_h, Areaj_h, nxj_h, nyj_h, nzj_h,
    Areak_h, nxk_h, nyk_h, nzk_h, Vol_h = load_or_compute_metrics(
        bid, rx, ry, rz, x_full, y_full, z_full, nxp, nyp, nzp, NG;
        cache_metrics=cache_metrics,
        periodic=ntuple(3) do direction
            isdefined(Main, :Iperiodic) && Main.Iperiodic[direction] &&
                Nprocs_block[direction] == 1
        end,
        topology_fingerprint=structured_connectivity_fingerprint(connectivity),
        singularity_edges=structured_local_metric_singularity_edges(
            bid, face_bc, connectivity,
            (ox, oy, oz), (nxp, nyp, nzp),
            (Nx_val, Ny_val, Nz_val),
        ),
        metric_mode=structured_metric_mode_setting(),
        cell_offsets=(ox, oy, oz),
        block_dims=(Nx_val, Ny_val, Nz_val),
        metric_auxiliary=temp_metric_auxiliary_h,
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
    active_vol = @view Vol_h[NG+1:nxp+NG, NG+1:nyp+NG, NG+1:nzp+NG]
    active_areas = (
        @view(Areai_h[NG+1:nxp+NG+1, NG+1:nyp+NG, NG+1:nzp+NG]),
        @view(Areaj_h[NG+1:nxp+NG, NG+1:nyp+NG+1, NG+1:nzp+NG]),
        @view(Areak_h[NG+1:nxp+NG, NG+1:nyp+NG, NG+1:nzp+NG+1]),
    )
    if any(x -> !isfinite(x) || x <= zero(eltype(active_vol)), active_vol)
        println("Rank $world_rank, Block $bid: CRITICAL! Invalid inverse cell volumes detected. min(inv_volume) = $(minimum(Vol_h))")
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
    if any(area -> any(x -> !isfinite(x) || x <= zero(eltype(area)), area), active_areas)
        println("Rank $world_rank, Block $bid: CRITICAL! Non-finite or non-positive face area detected.")
        MPI.Abort(MPI.COMM_WORLD, 1)
    end

    # Allocate primary and conservative variables
    Q = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Nprim)
    U = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Ncell_cons)
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
    if structured_cell_average_initialization_active()
        structured_initialize_cell_average!(
            U, Q, x, y, z, nxp, nyp, nzp,
        )
    else
        @gpu_launch threads=nthreads blocks=nb prim2c(U, Q, nxp, nyp, nzp)
    end
    @check_nan(U, "U after initial conservative state", bid, world_rank, 0)

    # CT: Initialize face-centered B from cell-centered B
    if equation_type == :MHD && ct_mode
        # Face B arrays will be allocated below; init is deferred to after Block construction
    end



    LTS_dt = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot)
    # RK backup has the same compact persistent layout as U. Q remains the
    # full primitive state used by reconstruction and MHD diagnostics.
    Un = gpu_zeros(FT, Nx_tot, Ny_tot, Nz_tot, Ncell_cons)

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
        U_target_cpu = zeros(FT, Nx_tot, Ny_tot, Nz_tot, Ncell_cons)
        if isdefined(@__MODULE__, :precursor_mean_path) && isfile(precursor_mean_path)
            # Load the persistent cell-state cross-section and tile along x.
            h5open(precursor_mean_path, "r") do fid
                grp_name = "b$(bid)"
                if haskey(fid, grp_name)
                    U_cross = FT.(read(fid["$(grp_name)/U_mean"]))  # (Ny_block, Nz_block, Ncons)
                    # Fill interior cells: tile the cross-section along x
                    for n in 1:Ncell_cons, k in 1:nzp, j in 1:nyp
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
                          (nxp+1)*nyp*nzp + nxp*(nyp+1)*nzp + nxp*nyp*(nzp+1)) * sizeof(FT) / 1024^2
            if dual_time
                imp_mem_mb += prod((Nx_tot, Ny_tot, Nz_tot, Ncons)) * sizeof(FT) / 1024^2
            end
            if _use_gmres
                gmres_mem = prod((Nx_tot, Ny_tot, Nz_tot, Ncons, _gmres_m + 1)) * sizeof(FT) / 1024^2
                gmres_mem += prod((nxp, nyp, nzp, Ncons)) * sizeof(FT) / 1024^2
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
    _ct_background_split = ct_mode && equation_type == :MHD &&
        isdefined(Main, :external_magnetic_background_splitting) &&
        Bool(Main.external_magnetic_background_splitting)
    _ct_b0x = _ct_background_split ?
        gpu_zeros(FT, nxp + 1 + 2*NG, nyp + 2*NG + 1, nzp + 2*NG + 1) : nothing
    _ct_b0y = _ct_background_split ?
        gpu_zeros(FT, nxp + 2*NG + 1, nyp + 1 + 2*NG, nzp + 2*NG + 1) : nothing
    _ct_b0z = _ct_background_split ?
        gpu_zeros(FT, nxp + 2*NG + 1, nyp + 2*NG + 1, nzp + 1 + 2*NG) : nothing
    _ct_b0_cell = _ct_background_split ?
        gpu_zeros(FT,nxp+2*NG,nyp+2*NG,nzp+2*NG,3) : nothing
    _ct_ex = ct_mode ? gpu_zeros(FT, nxp, nyp + 1, nzp + 1) : nothing
    _ct_ey = ct_mode ? gpu_zeros(FT, nxp + 1, nyp, nzp + 1) : nothing
    _ct_ez = ct_mode ? gpu_zeros(FT, nxp + 1, nyp + 1, nzp) : nothing
    _ct_fofc_flag = ct_mode && ct_first_order_flux_correction ?
        gpu_zeros(FT, nxp + 2*NG, nyp + 2*NG, nzp + 2*NG, 2) : nothing

    return Block(bid, nxp, nyp, nzp, rx, ry, rz, ox, oy, oz, Q, U, ϕ, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, Vol, x, y, z, LTS_dt, Un,
                 _ct_bx, _ct_by, _ct_bz, _ct_bx_n, _ct_by_n, _ct_bz_n,
                 _ct_b0x, _ct_b0y, _ct_b0z, _ct_b0_cell,
                 _ct_ex, _ct_ey, _ct_ez, _ct_fofc_flag,
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
        # Keep this validator usable by the static topology tests, which use
        # four-field mock Connectivity records and do not load the transform
        # module.  Production records carry `transform`; legacy records still
        # use the reciprocal flags below.
        has_transform = hasproperty(conn, :transform) &&
                        hasproperty(peer, :transform)
        transform_ok = true
        if has_transform
            transform = getproperty(conn, :transform)
            peer_transform = getproperty(peer, :transform)
            if transform !== nothing || peer_transform !== nothing
                transform = transform === nothing ?
                    structured_legacy_face_transform(face, conn.src_f, conn.reverse_tan) :
                    transform
                peer_transform = peer_transform === nothing ?
                    structured_legacy_face_transform(conn.src_f, peer.src_f, peer.reverse_tan) :
                    peer_transform
                inverse = structured_inverse_face_transform(transform)
                transform_ok =
                    peer_transform.source_face == inverse.source_face &&
                    peer_transform.destination_face == inverse.destination_face &&
                    peer_transform.source_for_destination == inverse.source_for_destination
            end
        end
        if peer.src_b != block || peer.src_f != face ||
           peer.reverse_tan != conn.reverse_tan ||
           peer.flip_normal != conn.flip_normal || !transform_ok
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
    return h5open(path, "r") do fid
        Nblocks = Int(read(fid["Nblocks"]))
    connect_raw = read(fid["connectivity"])
    
    # Load reverse_tan if available (new format), else default to false
    rev_raw = haskey(fid, "reverse_tan") ? read(fid["reverse_tan"]) : zeros(Int64, size(connect_raw, 1))
    
    # Load flip_normal if available, else default to false
    flip_raw = haskey(fid, "flip_normal") ? read(fid["flip_normal"]) : zeros(Int64, size(connect_raw, 1))

    # Optional physical image translation for periodic connectivity records.
    # Legacy topology files do not contain it and retain a zero shift.
    image_translation_raw = haskey(fid, "image_translation") ?
        read(fid["image_translation"]) : nothing
    if image_translation_raw !== nothing &&
       (ndims(image_translation_raw) != 2 ||
        size(image_translation_raw, 1) != size(connect_raw, 1) ||
        size(image_translation_raw, 2) != 3)
        error("Malformed image_translation: expected $(size(connect_raw, 1)) x 3 rows")
    end

    # New format: signed source-axis map for each destination axis (N x 3).
    # Legacy files continue to use reverse_tan and are upgraded lazily through
    # _connectivity_transform once the local face id is known.
    axis_raw = haskey(fid, "axis_map") ? read(fid["axis_map"]) : nothing
    if axis_raw !== nothing &&
       (ndims(axis_raw) != 2 || size(axis_raw, 1) != size(connect_raw, 1) || size(axis_raw, 2) != 3)
        error("Malformed axis_map: expected $(size(connect_raw, 1)) x 3 signed-axis rows")
    end
    
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
        transform = if axis_raw === nothing
            nothing
        else
            candidate = StructuredFaceTransform(
                Int8(f1), Int8(f2),
                Tuple(Int8.(axis_raw[i, :])),
            )
            structured_validate_face_transform(candidate)
            candidate
        end
        image_translation = image_translation_raw === nothing ?
            (0.0, 0.0, 0.0) :
            (Float64(image_translation_raw[i, 1]),
             Float64(image_translation_raw[i, 2]),
             Float64(image_translation_raw[i, 3]))
        connectivity[(b1, f1)] = Connectivity(
            b2, f2, rev, flip, transform, image_translation,
        )
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
    
        return Nblocks, connectivity, face_bc, bc_params, Nx_b, Ny_b, Nz_b
    end
end


function compute_structured_point_face_fluxes!(block::Block, dt, ϕ, Fx, Fy, Fz,
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
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i ct_mhd_characteristic_reconstruct_left_i_kernel!(Q, Fx, Areai, nxi, nyi, nzi, block.Bx_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values, block.B0x_face, block.B0_cell, ϕ)
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i ct_mhd_characteristic_reconstruct_right_i_kernel!(Q, Fv_x, Areai, nxi, nyi, nzi, block.Bx_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values, block.B0x_face, block.B0_cell, ϕ)
        @gpu_launch threads=threads_recon_i blocks=nb_recon_i ct_mhd_hlld_flux_i_kernel!(Fx, Fv_x, Fx, rho_sum_x, Areai, nxi, nyi, nzi, nxp, nyp, nzp, ch_glm_current, Int32(0), pos_meta, block.B0x_face, block.B0_cell, ϕ)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fx, "Fx after split characteristic HLLD", block.id, world_rank, tt)
        @static if ct_emf_scheme == CT_EMF_WENO7_SG07
            @gpu_launch threads=threads_recon_i blocks=nb_weno_i ct_mhd_characteristic_weno7_cache_i_kernel!(Q, cache_i, Areai, nxi, nyi, nzi, block.Bx_face, Vol, nxp, nyp, nzp, ch_glm_current, dt, Int32(0), block.B0x_face, block.B0_cell, pos_meta, ϕ)
        end

        @gpu_launch threads=threads_recon_j blocks=nb_recon_j ct_mhd_characteristic_reconstruct_left_j_kernel!(Q, Fy, Areaj, nxj, nyj, nzj, block.By_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values, block.B0y_face, block.B0_cell, ϕ)
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j ct_mhd_characteristic_reconstruct_right_j_kernel!(Q, Fv_y, Areaj, nxj, nyj, nzj, block.By_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values, block.B0y_face, block.B0_cell, ϕ)
        @gpu_launch threads=threads_recon_j blocks=nb_recon_j ct_mhd_hlld_flux_j_kernel!(Fy, Fv_y, Fy, rho_sum_y, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, ch_glm_current, Int32(0), pos_meta, block.B0y_face, block.B0_cell, ϕ)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fy, "Fy after split characteristic HLLD", block.id, world_rank, tt)
        @static if ct_emf_scheme == CT_EMF_WENO7_SG07
            @gpu_launch threads=threads_recon_j blocks=nb_weno_j ct_mhd_characteristic_weno7_cache_j_kernel!(Q, cache_j, Areaj, nxj, nyj, nzj, block.By_face, Vol, nxp, nyp, nzp, ch_glm_current, dt, Int32(0), block.B0y_face, block.B0_cell, pos_meta, ϕ)
        end

        @gpu_launch threads=threads_recon_k blocks=nb_recon_k ct_mhd_characteristic_reconstruct_left_k_kernel!(Q, Fz, Areak, nxk, nyk, nzk, block.Bz_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values, block.B0z_face, block.B0_cell, ϕ)
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k ct_mhd_characteristic_reconstruct_right_k_kernel!(Q, Fv_z, Areak, nxk, nyk, nzk, block.Bz_face, nxp, nyp, nzp, Int32(0), pos_meta, pos_values, block.B0z_face, block.B0_cell, ϕ)
        @gpu_launch threads=threads_recon_k blocks=nb_recon_k ct_mhd_hlld_flux_k_kernel!(Fz, Fv_z, Fz, rho_sum_z, Areak, nxk, nyk, nzk, nxp, nyp, nzp, ch_glm_current, Int32(0), pos_meta, block.B0z_face, block.B0_cell, ϕ)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fz, "Fz after split characteristic HLLD", block.id, world_rank, tt)
        @static if ct_emf_scheme == CT_EMF_WENO7_SG07
            @gpu_launch threads=threads_recon_k blocks=nb_weno_k ct_mhd_characteristic_weno7_cache_k_kernel!(Q, cache_k, Areak, nxk, nyk, nzk, block.Bz_face, Vol, nxp, nyp, nzp, ch_glm_current, dt, Int32(0), block.B0z_face, block.B0_cell, pos_meta, ϕ)
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
        @gpu_launch threads=threads_recon_i blocks=nb_conser_i Conser_reconstruct_i(Q, U, ϕ, Areai, Fx, rho_sum_x, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(0), _bx_ct, cache_i, Vol, dt, rk_stage, pos_meta, pos_values, block.B0x_face, block.B0_cell)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fx, "Fx after Conser_reconstruct_i", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_j blocks=nb_conser_j Conser_reconstruct_j(Q, U, ϕ, Areaj, Fy, rho_sum_y, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(0), _by_ct, cache_j, Vol, dt, rk_stage, pos_meta, pos_values, block.B0y_face, block.B0_cell)
        @static if strict_ct_positivity
            gpu_sync()
            ct_check_positivity_or_abort!(pos_meta, pos_values; rank=world_rank, block=block.id, step=tt, rk_stage=rk_stage)
        end
        @check_nan(Fy, "Fy after Conser_reconstruct_j", block.id, world_rank, tt)
        @gpu_launch threads=threads_recon_k blocks=nb_conser_k Conser_reconstruct_k(Q, U, ϕ, Areak, Fz, rho_sum_z, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(0), _bz_ct, cache_k, Vol, dt, rk_stage, pos_meta, pos_values, block.B0z_face, block.B0_cell)
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

end

function average_structured_point_face_fluxes!(
    block::Block, sensor, Fx, Fy, Fz, scratch_x, scratch_y, scratch_z,
    pos_meta=nothing,
)
    @static if structured_face_quadrature != STRUCTURED_FACE_POINT6
        return nothing
    end
    nxp,nyp,nzp=Int32(block.Nx),Int32(block.Ny),Int32(block.Nz)
    halo=Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    first_i=(cld(nxp+Int32(1),nthreads[1]),cld(nyp,nthreads[2]),
             cld(nzp+Int32(2)*halo,nthreads[3]))
    first_j=(cld(nxp,nthreads[1]),cld(nyp+Int32(1),nthreads[2]),
             cld(nzp+Int32(2)*halo,nthreads[3]))
    first_k=(cld(nxp,nthreads[1]),cld(nyp+Int32(2)*halo,nthreads[2]),
             cld(nzp+Int32(1),nthreads[3]))
    second_i=(cld(nxp+Int32(1),nthreads[1]),cld(nyp,nthreads[2]),cld(nzp,nthreads[3]))
    second_j=(cld(nxp,nthreads[1]),cld(nyp+Int32(1),nthreads[2]),cld(nzp,nthreads[3]))
    second_k=(cld(nxp,nthreads[1]),cld(nyp,nthreads[2]),cld(nzp+Int32(1),nthreads[3]))
    p2a_sensor = sensor
    @static if ct_troubled_mask_enabled &&
               !ct_troubled_face_p2a_enabled
        p2a_sensor = nothing
    end
    @gpu_launch threads=nthreads blocks=first_i structured_face_p2a_first_kernel!(
        scratch_x,Fx,nxp,nyp,nzp,Val(1),pos_meta,p2a_sensor,FT(hybrid_ϕ1))
    @gpu_launch threads=nthreads blocks=second_i structured_face_p2a_second_kernel!(
        Fx,scratch_x,Fx,p2a_sensor,nxp,nyp,nzp,FT(hybrid_ϕ1),Val(1),
        pos_meta)
    @gpu_launch threads=nthreads blocks=first_j structured_face_p2a_first_kernel!(
        scratch_y,Fy,nxp,nyp,nzp,Val(2),pos_meta,p2a_sensor,FT(hybrid_ϕ1))
    @gpu_launch threads=nthreads blocks=second_j structured_face_p2a_second_kernel!(
        Fy,scratch_y,Fy,p2a_sensor,nxp,nyp,nzp,FT(hybrid_ϕ1),Val(2),
        pos_meta)
    @gpu_launch threads=nthreads blocks=first_k structured_face_p2a_first_kernel!(
        scratch_z,Fz,nxp,nyp,nzp,Val(3),pos_meta,p2a_sensor,FT(hybrid_ϕ1))
    @gpu_launch threads=nthreads blocks=second_k structured_face_p2a_second_kernel!(
        Fz,scratch_z,Fz,p2a_sensor,nxp,nyp,nzp,FT(hybrid_ϕ1),Val(3),
        pos_meta)
    return nothing
end

function correct_structured_ct_fofc_face_fluxes!(
    block::Block, Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z,
    fallback_meta; record_faces::Bool=true,
)
    @static if !(equation_type == :MHD && ct_mode &&
                 ct_first_order_flux_correction)
        return nothing
    end
    block.fofc_flag === nothing && return nothing
    tangent = Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    nxp,nyp,nzp = Int32(block.Nx),Int32(block.Ny),Int32(block.Nz)
    blocks_i = (
        cld(nxp+Int32(1),nthreads[1]),
        cld(nyp+Int32(2)*tangent,nthreads[2]),
        cld(nzp+Int32(2)*tangent,nthreads[3]),
    )
    blocks_j = (
        cld(nxp+Int32(2)*tangent,nthreads[1]),
        cld(nyp+Int32(1),nthreads[2]),
        cld(nzp+Int32(2)*tangent,nthreads[3]),
    )
    blocks_k = (
        cld(nxp+Int32(2)*tangent,nthreads[1]),
        cld(nyp+Int32(2)*tangent,nthreads[2]),
        cld(nzp+Int32(1),nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=blocks_i ct_fofc_replace_face_flux_kernel!(
        Fx,rho_sum_x,block.fofc_flag,block.U,
        block.Bx_face,block.By_face,block.Bz_face,
        block.B0x_face,block.B0y_face,block.B0z_face,
        block.Areai,block.nxi,block.nyi,block.nzi,
        block.Areaj,block.nxj,block.nyj,block.nzj,
        block.Areak,block.nxk,block.nyk,block.nzk,
        nxp,nyp,nzp,ch_glm_current,fallback_meta,record_faces,Val(1),
    )
    @gpu_launch threads=nthreads blocks=blocks_j ct_fofc_replace_face_flux_kernel!(
        Fy,rho_sum_y,block.fofc_flag,block.U,
        block.Bx_face,block.By_face,block.Bz_face,
        block.B0x_face,block.B0y_face,block.B0z_face,
        block.Areai,block.nxi,block.nyi,block.nzi,
        block.Areaj,block.nxj,block.nyj,block.nzj,
        block.Areak,block.nxk,block.nyk,block.nzk,
        nxp,nyp,nzp,ch_glm_current,fallback_meta,record_faces,Val(2),
    )
    @gpu_launch threads=nthreads blocks=blocks_k ct_fofc_replace_face_flux_kernel!(
        Fz,rho_sum_z,block.fofc_flag,block.U,
        block.Bx_face,block.By_face,block.Bz_face,
        block.B0x_face,block.B0y_face,block.B0z_face,
        block.Areai,block.nxi,block.nyi,block.nzi,
        block.Areaj,block.nxj,block.nyj,block.nzj,
        block.Areak,block.nxk,block.nyk,block.nzk,
        nxp,nyp,nzp,ch_glm_current,fallback_meta,record_faces,Val(3),
    )
    return nothing
end

function scale_structured_ct_fofc_transport!(
    block::Block,Fx,Fy,Fz,Fv_x,Fv_y,Fv_z,
)
    @static if !(equation_type == :MHD && ct_mode &&
                 ct_first_order_flux_correction)
        return nothing
    end
    block.fofc_flag === nothing && return nothing
    tangent = Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    nxp,nyp,nzp = Int32(block.Nx),Int32(block.Ny),Int32(block.Nz)
    blocks_i = (
        cld(nxp+Int32(1),nthreads[1]),
        cld(nyp+Int32(2)*tangent,nthreads[2]),
        cld(nzp+Int32(2)*tangent,nthreads[3]),
    )
    blocks_j = (
        cld(nxp+Int32(2)*tangent,nthreads[1]),
        cld(nyp+Int32(1),nthreads[2]),
        cld(nzp+Int32(2)*tangent,nthreads[3]),
    )
    blocks_k = (
        cld(nxp+Int32(2)*tangent,nthreads[1]),
        cld(nyp+Int32(2)*tangent,nthreads[2]),
        cld(nzp+Int32(1),nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=blocks_i ct_fofc_scale_face_flux_kernel!(
        Fx,Fv_x,block.fofc_flag,nxp,nyp,nzp,Val(1),
    )
    @gpu_launch threads=nthreads blocks=blocks_j ct_fofc_scale_face_flux_kernel!(
        Fy,Fv_y,block.fofc_flag,nxp,nyp,nzp,Val(2),
    )
    @gpu_launch threads=nthreads blocks=blocks_k ct_fofc_scale_face_flux_kernel!(
        Fz,Fv_z,block.fofc_flag,nxp,nyp,nzp,Val(3),
    )
    return nothing
end

function mark_structured_ct_fofc_candidate!(
    block::Block, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z,
    dt, rk_a, fallback_meta, fallback_values,
)
    @static if !(equation_type == :MHD && ct_mode &&
                 ct_first_order_flux_correction)
        return nothing
    end
    block.fofc_flag === nothing && return nothing
    nb = (
        cld(block.Nx,nthreads[1]),
        cld(block.Ny,nthreads[2]),
        cld(block.Nz,nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=nb ct_mark_fofc_candidate_kernel!(
        block.fofc_flag,fallback_meta,fallback_values,
        block.U,block.Un,block.Q,Fx,Fy,Fz,Fv_x,Fv_y,Fv_z,block.Vol,
        block.Bx_face,block.By_face,block.Bz_face,
        block.Bx_face_n,block.By_face_n,block.Bz_face_n,
        block.B0x_face,block.B0y_face,block.B0z_face,
        block.Ex_edge,block.Ey_edge,block.Ez_edge,
        block.Areai,block.nxi,block.nyi,block.nzi,
        block.Areaj,block.nxj,block.nyj,block.nzj,
        block.Areak,block.nxk,block.nyk,block.nzk,
        FT(dt),FT(rk_a),FT(γ),FT(density_floor),FT(pressure_floor),
        Int32(block.Nx),Int32(block.Ny),Int32(block.Nz),
    )
    return nothing
end

function commit_structured_ct_fofc_candidate!(block::Block)
    @static if !(equation_type == :MHD && ct_mode &&
                 ct_first_order_flux_correction)
        return nothing
    end
    block.fofc_flag === nothing && return nothing
    nb = (
        cld(block.Nx,nthreads[1]),
        cld(block.Ny,nthreads[2]),
        cld(block.Nz,nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=nb ct_commit_fofc_candidate_kernel!(
        block.fofc_flag,Int32(block.Nx),Int32(block.Ny),Int32(block.Nz),
    )
    return nothing
end

function compute_structured_diffusive_face_fluxes!(
    block::Block, Fv_x, Fv_y, Fv_z, world_rank, tt,
    threads_visc_i, threads_visc_j, threads_visc_k,
)
    @static if !(viscous || (equation_type == :MHD && resistive))
        return nothing
    end
    Q=block.Q
    Areai,nxi,nyi,nzi=block.Areai,block.nxi,block.nyi,block.nzi
    Areaj,nxj,nyj,nzj=block.Areaj,block.nxj,block.nyj,block.nzj
    Areak,nxk,nyk,nzk=block.Areak,block.nxk,block.nyk,block.nzk
    nxp,nyp,nzp=block.Nx,block.Ny,block.Nz
    nb_visc_i=(Int32(cld(nxp+2*NG,threads_visc_i[1])),Int32(cld(nyp+2*NG,threads_visc_i[2])),Int32(cld(nzp+2*NG,threads_visc_i[3])))
    nb_visc_j=(Int32(cld(nxp+2*NG,threads_visc_j[1])),Int32(cld(nyp+2*NG,threads_visc_j[2])),Int32(cld(nzp+2*NG,threads_visc_j[3])))
    nb_visc_k=(Int32(cld(nxp+2*NG,threads_visc_k[1])),Int32(cld(nyp+2*NG,threads_visc_k[2])),Int32(cld(nzp+2*NG,threads_visc_k[3])))
    @gpu_launch threads=threads_visc_i blocks=nb_visc_i viscous_flux_i(
        Q,Fv_x,Areai,Areaj,Areak,nxi,nyi,nzi,nxj,nyj,nzj,nxk,nyk,nzk,
        block.Vol,nxp,nyp,nzp,block.is_interblock,false)
    @check_nan(Fv_x,"Fv_x after viscous_flux_i",block.id,world_rank,tt)
    @gpu_launch threads=threads_visc_j blocks=nb_visc_j viscous_flux_j(
        Q,Fv_y,Areai,Areaj,Areak,nxi,nyi,nzi,nxj,nyj,nzj,nxk,nyk,nzk,
        block.Vol,nxp,nyp,nzp,block.is_interblock,false)
    @check_nan(Fv_y,"Fv_y after viscous_flux_j",block.id,world_rank,tt)
    @gpu_launch threads=threads_visc_k blocks=nb_visc_k viscous_flux_k(
        Q,Fv_z,Areai,Areaj,Areak,nxi,nyi,nzi,nxj,nyj,nzj,nxk,nyk,nzk,
        block.Vol,nxp,nyp,nzp,block.is_interblock,false)
    @check_nan(Fv_z,"Fv_z after viscous_flux_k",block.id,world_rank,tt)
    return nothing
end

function compute_structured_face_fluxes!(block::Block, dt, sensor, Fx, Fy, Fz,
                      rho_sum_x, rho_sum_y, rho_sum_z,
                      Fv_x, Fv_y, Fv_z, world_rank, tt,
                      threads_recon_i, threads_recon_j, threads_recon_k,
                      threads_visc_i, threads_visc_j, threads_visc_k,
                      rk_stage::Int32, pos_meta, pos_values)
    compute_structured_point_face_fluxes!(
        block,dt,sensor,Fx,Fy,Fz,rho_sum_x,rho_sum_y,rho_sum_z,
        Fv_x,Fv_y,Fv_z,world_rank,tt,
        threads_recon_i,threads_recon_j,threads_recon_k,
        threads_visc_i,threads_visc_j,threads_visc_k,
        rk_stage,pos_meta,pos_values)
    average_structured_point_face_fluxes!(
        block,sensor,Fx,Fy,Fz,Fv_x,Fv_y,Fv_z,pos_meta)
    compute_structured_diffusive_face_fluxes!(
        block,Fv_x,Fv_y,Fv_z,world_rank,tt,
        threads_visc_i,threads_visc_j,threads_visc_k)
    return nothing
end

# ── Phase 2: Interior-only face fluxes on compute_stream ──
# Launches reconstruction with mode=1 (interior only, no ghost dependency)
# on a separate HIP stream. These kernels execute during MPI communication.
function compute_structured_interior_face_fluxes!(block::Block, dt, ϕ, Fx, Fy, Fz,
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
        @gpu_launch_stream stream threads=threads_recon_i blocks=nb_conser_i Conser_reconstruct_i(Q, U, ϕ, Areai, Fx, rho_sum_x, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(1), _bx_ct, cache_i, Vol, dt, rk_stage, pos_meta, pos_values, block.B0x_face, block.B0_cell)
        @gpu_launch_stream stream threads=threads_recon_j blocks=nb_conser_j Conser_reconstruct_j(Q, U, ϕ, Areaj, Fy, rho_sum_y, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(1), _by_ct, cache_j, Vol, dt, rk_stage, pos_meta, pos_values, block.B0y_face, block.B0_cell)
        @gpu_launch_stream stream threads=threads_recon_k blocks=nb_conser_k Conser_reconstruct_k(Q, U, ϕ, Areak, Fz, rho_sum_z, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(1), _bz_ct, cache_k, Vol, dt, rk_stage, pos_meta, pos_values, block.B0z_face, block.B0_cell)
    end
    # Viscous flux interior (viscous stencil >= 2 cells, fully covered by mode=1 range)
    if viscous || (equation_type == :MHD && resistive)
        @gpu_launch_stream stream threads=threads_visc_i blocks=nb_visc_i viscous_flux_i(Q, Fv_x, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, block.Vol, nxp, nyp, nzp, block.is_interblock, false)
        @gpu_launch_stream stream threads=threads_visc_j blocks=nb_visc_j viscous_flux_j(Q, Fv_y, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, block.Vol, nxp, nyp, nzp, block.is_interblock, false)
        @gpu_launch_stream stream threads=threads_visc_k blocks=nb_visc_k viscous_flux_k(Q, Fv_z, Areai, Areaj, Areak, nxi, nyi, nzi, nxj, nyj, nzj, nxk, nyk, nzk, block.Vol, nxp, nyp, nzp, block.is_interblock, false)
    end
end

# ── Phase 2: Boundary-only face fluxes on default stream ──
# Launches reconstruction with mode=2 (boundary only, needs ghost cells)
# Called AFTER sync_blocks! ensures ghost cells are valid.
function compute_structured_boundary_face_fluxes!(block::Block, dt, ϕ, Fx, Fy, Fz,
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
        @gpu_launch threads=threads_recon_i blocks=nb_conser_i Conser_reconstruct_i(Q, U, ϕ, Areai, Fx, rho_sum_x, Areai, nxi, nyi, nzi, nxp, nyp, nzp, si, Δsi, lpi, sRi, ΔsRi, ch_glm_current, Int32(2), _bx_ct, cache_i, Vol, dt, rk_stage, pos_meta, pos_values, block.B0x_face, block.B0_cell)
        @gpu_launch threads=threads_recon_j blocks=nb_conser_j Conser_reconstruct_j(Q, U, ϕ, Areaj, Fy, rho_sum_y, Areaj, nxj, nyj, nzj, nxp, nyp, nzp, sj, Δsj, lpj, sRj, ΔsRj, ch_glm_current, Int32(2), _by_ct, cache_j, Vol, dt, rk_stage, pos_meta, pos_values, block.B0y_face, block.B0_cell)
        @gpu_launch threads=threads_recon_k blocks=nb_conser_k Conser_reconstruct_k(Q, U, ϕ, Areak, Fz, rho_sum_z, Areak, nxk, nyk, nzk, nxp, nyp, nzp, sk, Δsk, lpk, sRk, ΔsRk, ch_glm_current, Int32(2), _bz_ct, cache_k, Vol, dt, rk_stage, pos_meta, pos_values, block.B0z_face, block.B0_cell)
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

@inline function _ct_troubled_periodic_face_mask(face_bc, bid)
    return ntuple(Val(6)) do face_id
        direction = (face_id + 1) ÷ 2
        base_periodic = Bool(Iperiodic[direction])
        bc_id = get(face_bc, (bid, face_id), BC_INTERBLOCK)
        base_periodic && Int32(bc_id) == Int32(BC_PERIODIC)
    end
end

Base.@noinline function _sync_ct_troubled_mask!(
    blocks, connectivity, Block_Nprocs, rank_offsets,
    Nx_b, Ny_b, Nz_b, ghost_pool, block_comms, face_bc,
)
    gpu_sync()
    mask_arrays = Dict(
        bid => reshape(b.ϕ, size(b.ϕ, 1), size(b.ϕ, 2), size(b.ϕ, 3), 1)
        for (bid, b) in blocks
    )
    copy_ghost_face!(
        blocks, connectivity, Block_Nprocs, rank_offsets,
        Nx_b, Ny_b, Nz_b, :ϕ, 1, ghost_pool;
        full_range=false, target_arrays=mask_arrays,
    )
    for (bid, b) in blocks
        exchange_ghost(
            mask_arrays[bid], 1, block_comms[bid], b.Nx, b.Ny, b.Nz,
            b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
            b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
            b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
            sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
            rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2,
            periodic_faces=_ct_troubled_periodic_face_mask(face_bc, bid),
            rank_coords=(b.rx, b.ry, b.rz),
            rank_dims=Tuple(Block_Nprocs[bid + 1]),
        )
    end
    copy_ghost_face!(
        blocks, connectivity, Block_Nprocs, rank_offsets,
        Nx_b, Ny_b, Nz_b, :ϕ, 1, ghost_pool;
        full_range=true, delta_mode=true, target_arrays=mask_arrays,
    )
    gpu_sync()
    return nothing
end

struct CTTroubledMaskTaskContext
    blocks::Any
    connectivity::Any
    block_nprocs::Any
    rank_offsets::Any
    nx_b::Any
    ny_b::Any
    nz_b::Any
    ghost_pool::Any
    block_comms::Any
    face_bc::Any
end

Base.@noinline function _ct_troubled_mask_halo_task!(
    task_context::CTTroubledMaskTaskContext,
)
    _sync_ct_troubled_mask!(
        task_context.blocks, task_context.connectivity,
        task_context.block_nprocs, task_context.rank_offsets,
        task_context.nx_b, task_context.ny_b, task_context.nz_b,
        task_context.ghost_pool, task_context.block_comms,
        task_context.face_bc,
    )
    return StructuredTaskDone
end


function time_step(world_rank, comm_cart, Block_Nprocs)
    validate_structured_run_configuration(
        equation_type, splitMethodID; implicit=implicit, dual_time=dual_time,
    )
    validate_structured_ct_thermodynamics(
        equation_type, ct_mode, isothermal_mhd,
        FT(γ), FT(Rg), FT(isothermal_temperature),
    )
    if equation_type == :MHD && ct_mode && resistive && NG < 4
        error(
            "Resistive CT requires NG>=4 so the tangential diffusive-flux " *
            "halo can use the existing centered derivative stencil",
        )
    end
    if structured_face_quadrature == STRUCTURED_FACE_POINT6 && NG < 4
        error("POINT6 face quadrature requires NG>=4, got NG=$NG")
    end
    if equation_type == :MHD && ct_mode &&
       structured_face_quadrature == STRUCTURED_FACE_POINT6 &&
       ct_emf_scheme != CT_EMF_WENO7_SG07
        error(
            "CT POINT6 face quadrature requires CT_EMF_WENO7_SG07 so edge " *
            "EMF construction does not consume the face-average flux buffer",
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
    temp_metric_auxiliary_h = Dict{Int, Any}()
    temp_metrics_pre_h = Dict{Int, Any}()
    was_computed_pre_h = Dict{Int, Bool}()
    temp_metric_auxiliary_pre_h = Dict{Int, Any}()
    metric_mode = structured_metric_mode_setting()

    # Load multi-block metadata
    if world_rank == 0
        println(">>> Loading multi-block connectivity...")
    end
    # connectivity_file is defined by the active structured run configuration.
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
                    connectivity[(b1 + 5, f1)] = Connectivity(
                        conn.src_b + 5, conn.src_f, conn.reverse_tan,
                        conn.flip_normal, conn.transform,
                    )
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
                    bcp = Base.setindex(bcp, Main.p_back, BCP_OUTLET_PAVG)
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

    canonical_external_bc_count =
        canonicalize_external_magnetic_bc_parameters!(
            Main, face_bc, bc_params,
        )
    if world_rank == 0 && canonical_external_bc_count > 0
        println(
            ">>> External-field BC parameters synchronized from the fixed " *
            "background model on $canonical_external_bc_count faces",
        )
    end

    # Keep the Cartesian communicator periodic for rank decomposition, but
    # suppress only the outer wrap on faces owned by another block.  Physical
    # periodic faces remain enabled, including mixed cases where only one side
    # of an axis is an inter-block interface.
    function _structured_periodic_face_mask(bid)
        ntuple(face_id -> begin
            direction = (face_id + 1) ÷ 2
            base_periodic = Bool(Iperiodic[direction])
            bc_id = get(face_bc, (bid, face_id), BC_INTERBLOCK)
            base_periodic && Int32(bc_id) == Int32(BC_PERIODIC)
        end, 6)
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
                
                blocks[bid] = load_block(
                    bid, 0, 0, 0, NG, Ncons, Nprim, Nprocs_block, world_rank,
                    face_bc, connectivity, _temp_rank_offsets, temp_metrics_h,
                    was_computed_h, temp_metric_auxiliary_h,
                )
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

        blocks[my_block_id] = load_block(
            my_block_id, rankx, ranky, rankz, NG, Ncons, Nprim,
            Nprocs_my_block, world_rank, face_bc, connectivity, rank_offsets,
            temp_metrics_h, was_computed_h, temp_metric_auxiliary_h,
        )

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
            for fid in 1:6
                if haskey(face_bc, (bid, fid))
                    if face_bc[(bid, fid)] == 0 && haskey(connectivity, (bid, fid))
                        conn = connectivity[(bid, fid)]
                        connectivity[(bid+5, fid)] = Connectivity(
                            conn.src_b + 5, conn.src_f, conn.reverse_tan,
                            conn.flip_normal, conn.transform,
                        )
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

                precursor_periodic = ntuple(3) do direction
                    low_face = 2direction - 1
                    high_face = 2direction
                    Nprocs_block[direction] == 1 &&
                        Int32(get(face_bc, (bid + 5, low_face), BC_INTERBLOCK)) ==
                            Int32(BC_PERIODIC) &&
                        Int32(get(face_bc, (bid + 5, high_face), BC_INTERBLOCK)) ==
                            Int32(BC_PERIODIC)
                end
                precursor_global_dims = (
                    Int(Nx_b[bid + 6]), Int(Ny_b[bid + 6]), Int(Nz_b[bid + 6]),
                )
                cache_path_pre, was_computed_pre, Areai_pre_h, nxi_pre_h, nyi_pre_h, nzi_pre_h, Areaj_pre_h, nxj_pre_h, nyj_pre_h, nzj_pre_h, Areak_pre_h, nxk_pre_h, nyk_pre_h, nzk_pre_h, Vol_pre_h = load_or_compute_metrics(
                    bid + 5, rx_pre, ry_pre, rz_pre,
                    x_pre_h, y_pre_h, z_pre_h, nxp_pre, nyp, nzp, NG;
                    cache_metrics=false,
                    periodic=precursor_periodic,
                    topology_fingerprint=structured_connectivity_fingerprint(connectivity),
                    singularity_edges=structured_local_metric_singularity_edges(
                        bid + 5, face_bc, connectivity,
                        (ox_pre, b.oy, b.oz), (nxp_pre, nyp, nzp),
                        precursor_global_dims,
                    ),
                    metric_mode=metric_mode,
                    cell_offsets=(ox_pre, b.oy, b.oz),
                    block_dims=precursor_global_dims,
                    metric_auxiliary=temp_metric_auxiliary_pre_h,
                )

                temp_metrics_pre_h[bid + 5] = (cache_path_pre, Areai_pre_h, nxi_pre_h, nyi_pre_h, nzi_pre_h, Areaj_pre_h, nxj_pre_h, nyj_pre_h, nzj_pre_h, Areak_pre_h, nxk_pre_h, nyk_pre_h, nzk_pre_h, Vol_pre_h)
                was_computed_pre_h[bid + 5] = was_computed_pre
                
                Nx_tot_pre = nxp_pre + 2*NG
                Ny_tot = nyp + 2*NG
                Nz_tot = nzp + 2*NG
                
                Areai_pre = GPUArray(Areai_pre_h); nxi_pre = GPUArray(nxi_pre_h); nyi_pre = GPUArray(nyi_pre_h); nzi_pre = GPUArray(nzi_pre_h)
                Areaj_pre = GPUArray(Areaj_pre_h); nxj_pre = GPUArray(nxj_pre_h); nyj_pre = GPUArray(nyj_pre_h); nzj_pre = GPUArray(nzj_pre_h)
                Areak_pre = GPUArray(Areak_pre_h); nxk_pre = GPUArray(nxk_pre_h); nyk_pre = GPUArray(nyk_pre_h); nzk_pre = GPUArray(nzk_pre_h)
                Vol_pre   = GPUArray(Vol_pre_h)
                
                Q_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot, Nprim)
                U_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot, Ncell_cons)
                ϕ_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot)
                
                x_pre = GPUArray(x_pre_h)
                y_pre = GPUArray(y_pre_h)
                z_pre = GPUArray(z_pre_h)
                
                initialize(Q_pre, x_pre, y_pre, z_pre, rx_pre, ry_pre, Nprocs_block, nxp_pre, nyp, nzp, bid + 5)
                
                nb_pre = (cld(Nx_tot_pre, nthreads[1]), cld(Ny_tot, nthreads[2]), cld(Nz_tot, nthreads[3]))
                @gpu_launch threads=nthreads blocks=nb_pre prim2c(U_pre, Q_pre, nxp_pre, nyp, nzp)
                
                LTS_dt_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot)
                Un_pre = gpu_zeros(FT, Nx_tot_pre, Ny_tot, Nz_tot, Ncell_cons)
                
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
                    nothing, nothing, nothing, nothing,
                    nothing, nothing, nothing, nothing,
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
    initial_metric_coordinates_h = Dict{Int,Any}()
    metric_interface_gate_enabled = lowercase(get(
        ENV, "STRUCTURED_METRIC_INTERFACE_GATE", "true",
    )) in ("true", "1", "yes", "on")
    metric_interface_relative_tolerance = _structured_metric_gate_tolerance(
        get(ENV, "STRUCTURED_METRIC_INTERFACE_REL_TOL", "1.0e-10"),
        "STRUCTURED_METRIC_INTERFACE_REL_TOL",
    )
    metric_interface_absolute_tolerance = _structured_metric_gate_tolerance(
        get(ENV, "STRUCTURED_METRIC_INTERFACE_ABS_TOL", "0.0"),
        "STRUCTURED_METRIC_INTERFACE_ABS_TOL",
    )

    function enforce_metric_interface_gate!(
        label, metric_values, metric_face_bc, metric_connectivity,
        metric_rank_offsets, metric_Nx_b, metric_Ny_b, metric_Nz_b,
    )
        metric_interface_gate_enabled || return (0.0, 0.0)
        absolute, relative = structured_shared_face_metric_residuals(
            metric_values, blocks, block_comms,
            metric_face_bc, metric_connectivity,
            metric_rank_offsets, Block_Nprocs,
            metric_Nx_b, metric_Ny_b, metric_Nz_b, NG,
        )
        if world_rank == 0
            @printf(
                "  %s shared-face metric gate: abs=%.6e rel=%.6e abs_tol=%.6e rel_tol=%.6e\n",
                label, absolute, relative,
                metric_interface_absolute_tolerance,
                metric_interface_relative_tolerance,
            )
        end
        (absolute <= metric_interface_absolute_tolerance ||
         relative <= metric_interface_relative_tolerance) || error(
            "$label shared-face metric mismatch: absolute=$absolute, " *
            "relative=$relative, absolute_tolerance=" *
            "$metric_interface_absolute_tolerance, relative_tolerance=" *
            "$metric_interface_relative_tolerance",
        )
        return absolute, relative
    end

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
        if metric_mode == STRUCTURED_METRIC_LOCAL_CHART
            length(temp_metric_auxiliary_h) == length(temp_metrics_h) || error(
                "local-chart metric mode requires an edge workspace for every local block",
            )
            local_recompute = any(values(was_computed_h)) ? 1 : 0
            global_recompute = MPI.Allreduce(
                local_recompute, MPI.MAX, MPI.COMM_WORLD,
            ) != 0
            if global_recompute
                for (bid, auxiliary) in collect(temp_metric_auxiliary_h)
                    auxiliary[2] !== nothing && continue
                    coordinates, metrics = auxiliary[1], auxiliary[3]
                    block = blocks[bid]
                    workspace = compute_scmm_edge_potentials(
                        coordinates.x, coordinates.y, coordinates.z,
                        block.Nx, block.Ny, block.Nz, NG,
                    )
                    temp_metric_auxiliary_h[bid] =
                        (coordinates, workspace, metrics)
                    was_computed_h[bid] = true
                end
                sync_all_interface_metric_edges!(
                    collect(keys(temp_metrics_h)), temp_metric_auxiliary_h,
                    face_bc, connectivity, _rank_offsets_setup, Block_Nprocs,
                    Nx_b, Ny_b, Nz_b, NG,
                )
                main_block_ids = Set(0:Nblocks-1)
                metric_main_blocks = Dict(
                    bid => block for (bid, block) in blocks if bid in main_block_ids
                )
                metric_main_connectivity = Dict(
                    endpoint => conn for (endpoint, conn) in connectivity
                    if endpoint[1] in main_block_ids && conn.src_b in main_block_ids
                )
                metric_junction_plan = build_ct_junction_plan(
                    metric_main_blocks, metric_main_connectivity,
                    Block_Nprocs, _rank_offsets_setup,
                    Nx_b, Ny_b, Nz_b,
                )
                sync_scmm_junction_metric_edges!(
                    temp_metric_auxiliary_h, metric_main_blocks,
                    metric_junction_plan, NG,
                )
                for (bid, auxiliary) in temp_metric_auxiliary_h
                    coordinates, workspace, metrics = auxiliary
                    block = blocks[bid]
                    finalize_scmm_face_metrics!(
                        metrics, workspace, block.Nx, block.Ny, block.Nz, NG,
                    )
                    finalize_scmm_volumes!(
                        metrics, coordinates.x, coordinates.y, coordinates.z,
                        block.Nx, block.Ny, block.Nz, NG,
                    )
                    periodic = ntuple(3) do direction
                        isdefined(Main, :Iperiodic) && Main.Iperiodic[direction] &&
                            Block_Nprocs[bid + 1][direction] == 1
                    end
                    _enforce_periodic_metric_ghosts!(
                        structured_metrics_tuple(metrics)...,
                        block.Nx, block.Ny, block.Nz, NG, periodic,
                    )
                end
                ct_sync_rank_metric_halos_only!(
                    temp_metrics_h, blocks, block_comms, Block_Nprocs,
                )
            elseif world_rank == 0
                println("Rank 0: All local-chart metric caches are valid.")
            end
        else
            if ct_mode
                ct_sync_rank_metrics!(
                    temp_metrics_h, blocks, block_comms, Block_Nprocs,
                )
            end
            _tmp_Block_Nprocs[] = Block_Nprocs
            sync_all_interface_metrics!(
                collect(keys(temp_metrics_h)), sync_dict, 0, 0, 0,
                face_bc, connectivity, _rank_offsets_setup,
                Nx_b, Ny_b, Nz_b, NG,
            )
        end
        if ct_mode && debug_metric_closure
            closure_l2, closure_max = ct_metric_closure_stats(temp_metrics_h, blocks)
            @printf("  CT metric closure after sync:  L2=%.6e Linf=%.6e\n", closure_l2, closure_max)
        end
        if ct_mode
            closure_rel_l2, closure_rel_max = ct_metric_closure_relative_stats(
                temp_metrics_h, blocks,
            )
            closure_tolerance = try
                parse(Float64, get(ENV, "CT_METRIC_CLOSURE_TOL", "1.0e-10"))
            catch
                1.0e-10
            end
            @printf(
                "  CT metric closure gate: relative L2=%.6e Linf=%.6e tol=%.6e\n",
                closure_rel_l2, closure_rel_max, closure_tolerance,
            )
            isfinite(closure_rel_max) && closure_rel_max <= closure_tolerance ||
                error(
                    "CT metric closure failed after synchronization: " *
                    "relative Linf=$(closure_rel_max), tolerance=$(closure_tolerance)",
                )
        end
        enforce_metric_interface_gate!(
            "main-domain", temp_metrics_h, face_bc, connectivity,
            _rank_offsets_setup, Nx_b, Ny_b, Nz_b,
        )

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
            copyto!(blocks[bid].Vol, Vol_h)
        end

        for (bid, auxiliary) in temp_metric_auxiliary_h
            initial_metric_coordinates_h[bid] = auxiliary[1]
        end
        # SCMM workspaces and synchronized host metrics are startup-only.
        # Keep only coordinates until vector-potential CT initialization has
        # formed one canonical line integral per physical edge.
        empty!(temp_metric_auxiliary_h)
        empty!(temp_metrics_h)
        empty!(was_computed_h)
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
            for fid in 1:6
                if haskey(face_bc, (bid-5, fid))
                    face_bc_pre[(bid, fid)] = face_bc[(bid-5, fid)]
                end
            end
            face_bc_pre[(bid, 1)] = BC_PERIODIC
            face_bc_pre[(bid, 2)] = BC_PERIODIC
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
        Ny_b_pre = (Main.Ny_b..., Main.Ny_b...)
        Nz_b_pre = (Main.Nz_b..., Main.Nz_b...)
        _tmp_Block_Nprocs[] = Block_Nprocs

        if metric_mode == STRUCTURED_METRIC_LOCAL_CHART
            length(temp_metric_auxiliary_pre_h) == length(temp_metrics_pre_h) ||
                error(
                    "local-chart metric mode requires an edge workspace for " *
                    "every local CEBL precursor block",
                )
            local_recompute_pre = any(values(was_computed_pre_h)) ? 1 : 0
            global_recompute_pre = MPI.Allreduce(
                local_recompute_pre, MPI.MAX, MPI.COMM_WORLD,
            ) != 0
            if global_recompute_pre
                for (bid, auxiliary) in collect(temp_metric_auxiliary_pre_h)
                    auxiliary[2] !== nothing && continue
                    coordinates, metrics = auxiliary[1], auxiliary[3]
                    block = blocks[bid]
                    workspace = compute_scmm_edge_potentials(
                        coordinates.x, coordinates.y, coordinates.z,
                        block.Nx, block.Ny, block.Nz, NG,
                    )
                    temp_metric_auxiliary_pre_h[bid] =
                        (coordinates, workspace, metrics)
                    was_computed_pre_h[bid] = true
                end
                sync_all_interface_metric_edges!(
                    collect(keys(temp_metrics_pre_h)),
                    temp_metric_auxiliary_pre_h, face_bc_pre, connectivity,
                    _rank_offsets_pre, Block_Nprocs,
                    Nx_b_pre, Ny_b_pre, Nz_b_pre, NG,
                )
                precursor_block_ids = Set(Nblocks:2Nblocks-1)
                metric_precursor_blocks = Dict(
                    bid => block for (bid, block) in blocks
                    if bid in precursor_block_ids
                )
                metric_precursor_connectivity = Dict(
                    endpoint => conn for (endpoint, conn) in connectivity
                    if endpoint[1] in precursor_block_ids &&
                        conn.src_b in precursor_block_ids
                )
                metric_junction_plan_pre = build_ct_junction_plan(
                    metric_precursor_blocks, metric_precursor_connectivity,
                    Block_Nprocs, _rank_offsets_pre,
                    Nx_b_pre, Ny_b_pre, Nz_b_pre,
                )
                sync_scmm_junction_metric_edges!(
                    temp_metric_auxiliary_pre_h, metric_precursor_blocks,
                    metric_junction_plan_pre, NG,
                )
                for (bid, auxiliary) in temp_metric_auxiliary_pre_h
                    coordinates, workspace, metrics = auxiliary
                    block = blocks[bid]
                    finalize_scmm_face_metrics!(
                        metrics, workspace,
                        block.Nx, block.Ny, block.Nz, NG,
                    )
                    finalize_scmm_volumes!(
                        metrics, coordinates.x, coordinates.y, coordinates.z,
                        block.Nx, block.Ny, block.Nz, NG,
                    )
                    periodic = ntuple(3) do direction
                        low_face = 2direction - 1
                        high_face = 2direction
                        Block_Nprocs[bid + 1][direction] == 1 &&
                            Int32(get(face_bc_pre, (bid, low_face), BC_INTERBLOCK)) ==
                                Int32(BC_PERIODIC) &&
                            Int32(get(face_bc_pre, (bid, high_face), BC_INTERBLOCK)) ==
                                Int32(BC_PERIODIC)
                    end
                    _enforce_periodic_metric_ghosts!(
                        structured_metrics_tuple(metrics)...,
                        block.Nx, block.Ny, block.Nz, NG, periodic,
                    )
                end
                ct_sync_rank_metric_halos_only!(
                    temp_metrics_pre_h, blocks, block_comms, Block_Nprocs,
                )
            elseif world_rank == 0
                println("Rank 0: All CEBL local-chart metric caches are valid.")
            end
        else
            # Called by ALL ranks collectively.
            sync_all_interface_metrics!(
                collect(keys(temp_metrics_pre_h)), sync_dict_pre, 0, 0, 0,
                face_bc_pre, connectivity, _rank_offsets_pre,
                Nx_b_pre, Ny_b_pre, Nz_b_pre, NG,
            )
        end
        enforce_metric_interface_gate!(
            "CEBL-precursor", temp_metrics_pre_h, face_bc_pre, connectivity,
            _rank_offsets_pre, Nx_b_pre, Ny_b_pre, Nz_b_pre,
        )
        
        for (bid, val) in temp_metrics_pre_h
            Areai_pre_h, nxi_pre_h, nyi_pre_h, nzi_pre_h = sync_dict_pre[bid][1:4]
            Areaj_pre_h, nxj_pre_h, nyj_pre_h, nzj_pre_h = sync_dict_pre[bid][5:8]
            Areak_pre_h, nxk_pre_h, nyk_pre_h, nzk_pre_h = sync_dict_pre[bid][9:12]
            
            copyto!(blocks[bid].Areai, Areai_pre_h); copyto!(blocks[bid].nxi, nxi_pre_h); copyto!(blocks[bid].nyi, nyi_pre_h); copyto!(blocks[bid].nzi, nzi_pre_h)
            copyto!(blocks[bid].Areaj, Areaj_pre_h); copyto!(blocks[bid].nxj, nxj_pre_h); copyto!(blocks[bid].nyj, nyj_pre_h); copyto!(blocks[bid].nzj, nzj_pre_h)
            copyto!(blocks[bid].Areak, Areak_pre_h); copyto!(blocks[bid].nxk, nxk_pre_h); copyto!(blocks[bid].nyk, nyk_pre_h); copyto!(blocks[bid].nzk, nzk_pre_h)
            copyto!(blocks[bid].Vol, val[14])
        end
        empty!(temp_metric_auxiliary_pre_h)
        empty!(temp_metrics_pre_h)
        empty!(was_computed_pre_h)
    end



    # Global dimensions for flux buffers (max across all local blocks)
    max_nxp = maximum(b.Nx for (_, b) in blocks)
    max_nyp = maximum(b.Ny for (_, b) in blocks)
    max_nzp = maximum(b.Nz for (_, b) in blocks)

    # Point-flux storage uses the same compile-time tangential halo everywhere:
    # zero for legacy hydro/GLM, one for SG07 CT, two for POINT6 face averages.
    _g = 2*Int(STRUCTURED_FLUX_TANGENTIAL_HALO)
    shared_Fx  = gpu_zeros(FT, max_nxp+1, max_nyp+_g, max_nzp+_g, Ncons)
    shared_Fy  = gpu_zeros(FT, max_nxp+_g, max_nyp+1, max_nzp+_g, Ncons)
    shared_Fz  = gpu_zeros(FT, max_nxp+_g, max_nyp+_g, max_nzp+1, Ncons)
    shared_rho_sum_x = ct_mode ? gpu_zeros(FT, max_nxp+1, max_nyp+_g, max_nzp+_g) : gpu_zeros(FT, 1, 1, 1)
    shared_rho_sum_y = ct_mode ? gpu_zeros(FT, max_nxp+_g, max_nyp+1, max_nzp+_g) : gpu_zeros(FT, 1, 1, 1)
    shared_rho_sum_z = ct_mode ? gpu_zeros(FT, max_nxp+_g, max_nyp+_g, max_nzp+1) : gpu_zeros(FT, 1, 1, 1)
    shared_Fvx = gpu_zeros(FT, max_nxp+1, max_nyp+_g, max_nzp+_g, Ncons)
    shared_Fvy = gpu_zeros(FT, max_nxp+_g, max_nyp+1, max_nzp+_g, Ncons)
    shared_Fvz = gpu_zeros(FT, max_nxp+_g, max_nyp+_g, max_nzp+1, Ncons)
    shared_dU_forced = gpu_zeros(FT, max_nxp, max_nyp, max_nzp, Ncell_cons)
    shared_ct_pos_meta = gpu_zeros(Int32, CT_POS_META_LEN)
    shared_ct_pos_values = gpu_zeros(FT, CT_POS_VALUE_LEN)
    shared_positivity_count = gpu_zeros(Int32, 1)
    shared_conservation_delta = gpu_zeros(FT, 5)

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
    restored_implicit_history = Set{Int}()

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

    external_field_caches = build_structured_external_field_caches(
        Main, blocks, face_bc, bc_params, Block_Nprocs, world_rank,
    )

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
            fill_structured_ghost_cells!(b.Q, b.U, rx_b, ry_b, rz_b,
                      b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj, b.nxk, b.nyk, b.nzk,
                      b.Areai, b.Areaj, b.Areak, b.Vol, current_dt,
                      b.x, b.y, b.z, b.id, b.Nx, b.Ny, b.Nz, Nprocs_b, 1, face_bc, bc_params,
                      @isdefined(dsrfg_params) ? dsrfg_params : default_dsrfg_params,
                      get(external_field_caches, bid, nothing))
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
            chkname = joinpath(
                structured_checkpoint_dir(),
                "chk-$(restart_step)-b$(b.id).h5",
            )

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

            if dual_time && hasproperty(b, :Un) && b.Un !== nothing &&
               hasproperty(b, :U_nm1) && b.U_nm1 !== nothing
                history_payload = h5open(chkname, "r") do f
                    if haskey(f, "Un") && haskey(f, "U_nm1")
                        lox = b.ox + 1; hix = b.ox + b.Nx
                        loy = b.oy + 1; hiy = b.oy + b.Ny
                        loz = b.oz + 1; hiz = b.oz + b.Nz
                        (
                            Un=f["Un"][lox:hix, loy:hiy, loz:hiz, :],
                            U_nm1=f["U_nm1"][lox:hix, loy:hiy, loz:hiz, :],
                        )
                    else
                        nothing
                    end
                end
                if history_payload !== nothing
                    copyto!(@view(b.Un[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny,
                        NG+1:NG+b.Nz, :,
                    ]), GPUArray(FT.(history_payload.Un)))
                    copyto!(@view(b.U_nm1[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny,
                        NG+1:NG+b.Nz, :,
                    ]), GPUArray(FT.(history_payload.U_nm1)))
                    push!(restored_implicit_history, bid)
                end
            end

            if equation_type == :MHD && ct_mode && b.Bx_face !== nothing
                checkpoint_background_split =
                    isdefined(Main, :external_magnetic_background_splitting) &&
                    Bool(Main.external_magnetic_background_splitting)
                ct_payload = h5open(chkname, "r") do f
                    required = checkpoint_background_split ?
                        ("U", "Bx_face", "By_face", "Bz_face",
                         "B0x_face", "B0y_face", "B0z_face") :
                        ("U", "Bx_face", "By_face", "Bz_face")
                    if !all(name -> haskey(f, name), required)
                        return nothing
                    end
                    validate_ct_checkpoint_metadata!(f, chkname)
                    lox = b.ox + 1; hix = b.ox + b.Nx
                    loy = b.oy + 1; hiy = b.oy + b.Ny
                    loz = b.oz + 1; hiz = b.oz + b.Nz
                    return (
                        U=f["U"][lox:hix, loy:hiy, loz:hiz, :],
                        Bx=f["Bx_face"][lox:hix+1, loy:hiy, loz:hiz],
                        By=f["By_face"][lox:hix, loy:hiy+1, loz:hiz],
                        Bz=f["Bz_face"][lox:hix, loy:hiy, loz:hiz+1],
                        B0x=checkpoint_background_split ?
                            f["B0x_face"][lox:hix+1, loy:hiy, loz:hiz] : nothing,
                        B0y=checkpoint_background_split ?
                            f["B0y_face"][lox:hix, loy:hiy+1, loz:hiz] : nothing,
                        B0z=checkpoint_background_split ?
                            f["B0z_face"][lox:hix, loy:hiy, loz:hiz+1] : nothing,
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
                    u_payload = ct_payload.U
                    nstored = size(u_payload, 4)
                    if nstored < Ncell_cons
                        error(
                            "Checkpoint $chkname has only $nstored conservative " *
                            "components; CT restart requires at least $Ncell_cons",
                        )
                    end
                    u_compact = nstored == Ncell_cons ?
                        u_payload : u_payload[:, :, :, 1:Ncell_cons]
                    copyto!(@view(b.U[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny, NG+1:NG+b.Nz, :,
                    ]), GPUArray(FT.(u_compact)))
                    copyto!(@view(b.Bx_face[
                        NG+1:NG+b.Nx+1, NG+1:NG+b.Ny, NG+1:NG+b.Nz,
                    ]), GPUArray(FT.(ct_payload.Bx)))
                    copyto!(@view(b.By_face[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny+1, NG+1:NG+b.Nz,
                    ]), GPUArray(FT.(ct_payload.By)))
                    copyto!(@view(b.Bz_face[
                        NG+1:NG+b.Nx, NG+1:NG+b.Ny, NG+1:NG+b.Nz+1,
                    ]), GPUArray(FT.(ct_payload.Bz)))
                    if b.B0x_face !== nothing
                        ct_payload.B0x === nothing && error(
                            "Checkpoint $chkname is missing B0 face fluxes " *
                            "for an enabled CT background split",
                        )
                        copyto!(@view(b.B0x_face[
                            NG+1:NG+b.Nx+1, NG+1:NG+b.Ny, NG+1:NG+b.Nz,
                        ]), GPUArray(FT.(ct_payload.B0x)))
                        copyto!(@view(b.B0y_face[
                            NG+1:NG+b.Nx, NG+1:NG+b.Ny+1, NG+1:NG+b.Nz,
                        ]), GPUArray(FT.(ct_payload.B0y)))
                        copyto!(@view(b.B0z_face[
                            NG+1:NG+b.Nx, NG+1:NG+b.Ny, NG+1:NG+b.Nz+1,
                        ]), GPUArray(FT.(ct_payload.B0z)))
                    end
                    push!(restored_ct_face_b, bid)
                end
            end
        end

        # Restore step and physical time
        first_bid_r = first(blocks)[1]
        chkname_meta = joinpath(
            structured_checkpoint_dir(),
            "chk-$(restart_step)-b$(first_bid_r).h5",
        )
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
        blocks, connectivity, Block_Nprocs, rank_offsets, Nx_b, Ny_b, Nz_b,
    ) : nothing
    ct_derivation_halo_plan = nothing
    @static if equation_type == :MHD && ct_mode
        maximum(keys(blocks)) + 1 <= minimum((
            length(Nx_b), length(Ny_b), length(Nz_b), length(Block_Nprocs),
        )) || throw(DimensionMismatch(
            "missing global block dimensions for CT derivation halo",
        ))
        ct_block_dimensions = [
            (Int(Nx_b[index]), Int(Ny_b[index]), Int(Nz_b[index]))
            for index in eachindex(Block_Nprocs)
        ]
        ct_derivation_halo_plan = build_ct_derivation_halo_plan(
            blocks, ct_block_dimensions, face_bc, connectivity,
            Block_Nprocs, rank_offsets;
            world_rank=world_rank, ng=NG, communicator=MPI.COMM_WORLD,
            conservative_components=5,
            cell_reach=ct_primitive_recovery == CT_PRIMITIVE_POINT6 ?
                (2,2,2) : (0,0,0),
            face_reach=ct_primitive_recovery == CT_PRIMITIVE_POINT6 ?
                (2,2,2) : (0,0,0),
        )
        refresh_ct_derivation_static!(ct_derivation_halo_plan, blocks)
        if ct_primitive_recovery == CT_PRIMITIVE_POINT6
            ct_physical_faces = Dict{Int,NTuple{6,Bool}}()
            for (bid, block) in blocks
                rank_coordinates = (Int(block.rx), Int(block.ry), Int(block.rz))
                rank_dimensions = Tuple(Int.(Block_Nprocs[bid + 1]))
                ct_physical_faces[bid] = ntuple(Val(6)) do face_id
                    direction = fld(face_id + 1, 2)
                    owns_face = isodd(face_id) ?
                        rank_coordinates[direction] == 0 :
                        rank_coordinates[direction] == rank_dimensions[direction] - 1
                    boundary_type = Int32(get(
                        face_bc, (bid, face_id), BC_INTERBLOCK,
                    ))
                    owns_face && _ct_is_physical_boundary_type(boundary_type)
                end
            end
            materialize_ct_derivation_static_ghosts!(
                ct_derivation_halo_plan, blocks, ct_physical_faces,
            )
        end
        ct_derivation_bytes = MPI.Allreduce(
            ct_derivation_halo_plan.persistent_bytes, MPI.SUM,
            MPI.COMM_WORLD,
        )
        world_rank == 0 && println(
            ">>> CT derivation storage ($(ct_primitive_recovery == CT_PRIMITIVE_POINT6 ? "POINT6" : "DIRECT")): " *
            "$(round(ct_derivation_bytes/1024^2; digits=2)) MiB total " *
            "(packed shells + communication staging)",
        )
    end

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
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values, first_b.B0x_face, first_b.B0_cell, first_b.ϕ;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_right_i = auto_tune_kernel("CTMHD_charR_i", ct_mhd_characteristic_reconstruct_right_i_kernel!,
            first_b.Q, shared_Fvx,
            first_b.Areai, first_b.nxi, first_b.nyi, first_b.nzi, first_b.Bx_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values, first_b.B0x_face, first_b.B0_cell, first_b.ϕ;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_i = auto_tune_kernel("CTMHD_hlld_i", ct_mhd_hlld_flux_i_kernel!,
            shared_Fx, shared_Fvx, shared_Fx, shared_rho_sum_x,
            first_b.Areai, first_b.nxi, first_b.nyi, first_b.nzi,
            nxp_t, nyp_t, nzp_t, ch_glm_current, Int32(0), shared_ct_pos_meta,
            first_b.B0x_face, first_b.B0_cell, first_b.ϕ;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_left_j = auto_tune_kernel("CTMHD_charL_j", ct_mhd_characteristic_reconstruct_left_j_kernel!,
            first_b.Q, shared_Fy,
            first_b.Areaj, first_b.nxj, first_b.nyj, first_b.nzj, first_b.By_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values, first_b.B0y_face, first_b.B0_cell, first_b.ϕ;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_right_j = auto_tune_kernel("CTMHD_charR_j", ct_mhd_characteristic_reconstruct_right_j_kernel!,
            first_b.Q, shared_Fvy,
            first_b.Areaj, first_b.nxj, first_b.nyj, first_b.nzj, first_b.By_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values, first_b.B0y_face, first_b.B0_cell, first_b.ϕ;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_j = auto_tune_kernel("CTMHD_hlld_j", ct_mhd_hlld_flux_j_kernel!,
            shared_Fy, shared_Fvy, shared_Fy, shared_rho_sum_y,
            first_b.Areaj, first_b.nxj, first_b.nyj, first_b.nzj,
            nxp_t, nyp_t, nzp_t, ch_glm_current, Int32(0), shared_ct_pos_meta,
            first_b.B0y_face, first_b.B0_cell, first_b.ϕ;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_left_k = auto_tune_kernel("CTMHD_charL_k", ct_mhd_characteristic_reconstruct_left_k_kernel!,
            first_b.Q, shared_Fz,
            first_b.Areak, first_b.nxk, first_b.nyk, first_b.nzk, first_b.Bz_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values, first_b.B0z_face, first_b.B0_cell, first_b.ϕ;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_state_right_k = auto_tune_kernel("CTMHD_charR_k", ct_mhd_characteristic_reconstruct_right_k_kernel!,
            first_b.Q, shared_Fvz,
            first_b.Areak, first_b.nxk, first_b.nyk, first_b.nzk, first_b.Bz_face,
            nxp_t, nyp_t, nzp_t, Int32(0), shared_ct_pos_meta, shared_ct_pos_values, first_b.B0z_face, first_b.B0_cell, first_b.ϕ;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_k = auto_tune_kernel("CTMHD_hlld_k", ct_mhd_hlld_flux_k_kernel!,
            shared_Fz, shared_Fvz, shared_Fz, shared_rho_sum_z,
            first_b.Areak, first_b.nxk, first_b.nyk, first_b.nzk,
            nxp_t, nyp_t, nzp_t, ch_glm_current, Int32(0), shared_ct_pos_meta,
            first_b.B0z_face, first_b.B0_cell, first_b.ϕ;
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
            Int32(1), shared_ct_pos_meta, shared_ct_pos_values,
            first_b.B0x_face, first_b.B0_cell;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_j = auto_tune_kernel("Conser_recon_j", Conser_reconstruct_j,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areaj,
            shared_Fy, shared_rho_sum_y, first_b.Areaj,
            first_b.nxj, first_b.nyj, first_b.nzj, nxp_t, nyp_t, nzp_t,
            first_b.stencil_j, first_b.Δstencil_j, first_b.lin_phi_j,
            first_b.stencil_R_j, first_b.Δstencil_R_j, ch_glm_current, Int32(0),
            ct_mode ? first_b.By_face : first_b.ϕ,
            cache_j, first_b.Vol, current_dt,
            Int32(1), shared_ct_pos_meta, shared_ct_pos_values,
            first_b.B0y_face, first_b.B0_cell;
            nxp=nxp_t, nyp=nyp_t, nzp=nzp_t, verbose=_verbose)
        cfg_recon_k = auto_tune_kernel("Conser_recon_k", Conser_reconstruct_k,
            first_b.Q, first_b.U, first_b.ϕ, first_b.Areak,
            shared_Fz, shared_rho_sum_z, first_b.Areak,
            first_b.nxk, first_b.nyk, first_b.nzk, nxp_t, nyp_t, nzp_t,
            first_b.stencil_k, first_b.Δstencil_k, first_b.lin_phi_k,
            first_b.stencil_R_k, first_b.Δstencil_R_k, ch_glm_current, Int32(0),
            ct_mode ? first_b.Bz_face : first_b.ϕ,
            cache_k, first_b.Vol, current_dt,
            Int32(1), shared_ct_pos_meta, shared_ct_pos_values,
            first_b.B0z_face, first_b.B0_cell;
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

    # CT energy diagnostics use only tiny reusable accumulators outside Block.
    # No allocation or kernel launch occurs when the environment flag is off.
    ct_energy_budget_scratch =
        _ct_energy_budget_enabled ? Dict{Int,Any}() : nothing
    ct_energy_budget_log = nothing
    if _ct_energy_budget_enabled
        for (bid, b) in blocks
            ct_energy_budget_scratch[bid] = (
                state=gpu_zeros(FT, CT_ENERGY_BUDGET_STATE_LEN),
                base=gpu_zeros(FT, CT_ENERGY_BUDGET_STATE_LEN),
                flux=gpu_zeros(FT, CT_ENERGY_BUDGET_FLUX_LEN),
            )
        end
        if world_rank == 0
            budget_path = get(
                ENV, "OPENCFD_CT_ENERGY_BUDGET_FILE",
                "ct_energy_budget.dat",
            )
            ct_energy_budget_log = open(budget_path, "w")
            println(
                ct_energy_budget_log,
                "# CT energy budget; conservative U and face-B are sampled " *
                "without modifying the numerical state",
            )
        end
    end

    # Face CT data do not exist until the first U/Q ghost synchronization and
    # ct_init_face_b! have completed. Once ready, POINT6 owns both the physical
    # point state and the Q halo exchanged for characteristic reconstruction.
    ct_face_state_ready = Ref(false)
    structured_task_context = nothing
    assert_structured_taskgraph_only!()
    structured_task_gamma = FT(γ)
    structured_task_ct_active = equation_type == :MHD && ct_mode
    structured_task_sync_state = Ref{Any}(nothing)
    structured_rk_task_context = nothing
    structured_rk_task_state = Ref{Any}(nothing)
    structured_post_task_context = nothing
    structured_final_plot_task_context = nothing
    structured_post_task_state = Ref{Any}(nothing)
    structured_resistive_task_context = nothing
    structured_resistive_task_state = Ref{Any}(nothing)
    structured_implicit_task_context = nothing
    structured_implicit_task_state = Ref{Any}(nothing)
    structured_taskgraph_sync! = nothing
    ct_background_split_enabled =
        equation_type == :MHD && ct_mode &&
        isdefined(Main, :external_magnetic_background_splitting) &&
        Bool(Main.external_magnetic_background_splitting)
    ct_background_ready = Ref(false)

    function activate_ct_background_split!()
        ct_background_split_enabled || return nothing
        for (_, b) in blocks
            b.B0x_face === nothing && continue
            copyto!(b.B0x_face, b.Bx_face)
            copyto!(b.B0y_face, b.By_face)
            copyto!(b.B0z_face, b.Bz_face)
            fill!(b.Bx_face, zero(FT))
            fill!(b.By_face, zero(FT))
            fill!(b.Bz_face, zero(FT))
            fill!(b.Bx_face_n, zero(FT))
            fill!(b.By_face_n, zero(FT))
            fill!(b.Bz_face_n, zero(FT))
        end
        for (_, b) in blocks
            b.B0_cell === nothing && continue
            ct_recover_background_cell_b!(
                b,b.Nx,b.Ny,b.Nz; include_ghosts=true,
            )
        end
        gpu_sync()
        ct_background_ready[] = true
        return nothing
    end

    function sync_ct_background_face_halos!()
        ct_background_split_enabled || return nothing
        saved = Dict{Int,Tuple{Any,Any,Any}}()
        for (bid, b) in blocks
            b.B0x_face === nothing && continue
            saved[bid] = (b.Bx_face, b.By_face, b.Bz_face)
            b.Bx_face = b.B0x_face
            b.By_face = b.B0y_face
            b.Bz_face = b.B0z_face
            ct_fill_zerograd_face_b!(b, b.Nx, b.Ny, b.Nz)
            local_periodic = ct_local_periodic_directions(
                bid,face_bc,Block_Nprocs,BC_PERIODIC,
            )
            ct_fill_periodic_face_b!(
                b, b.Nx, b.Ny, b.Nz, local_periodic,
            )
        end
        ct_sync_rank_sheets!(
            blocks, block_comms, Block_Nprocs;
            sync_face_flux=true, sync_edges=false,
        )
        ct_sync_interface_sheets!(
            blocks, ct_sync_plan; sync_face_flux=true, sync_edges=false,
        )
        # A checkpoint stores only physical-domain B0 faces. Reconstruct the
        # analytic physical-boundary halo while the active face pointers refer
        # to B0, but preserve the checkpointed physical normal face itself.
        apply_external_ct_face_b_boundaries!(;
            include_normal_face=false,
            background_splitting=false,
        )
        for (bid, b) in blocks
            haskey(saved, bid) || continue
            b.Bx_face, b.By_face, b.Bz_face = saved[bid]
            ct_recover_background_cell_b!(
                b,b.Nx,b.Ny,b.Nz; include_ghosts=true,
            )
        end
        gpu_sync()
        return nothing
    end

    function sync_ct_q_components!(nvariables::Integer)
        gpu_sync()
        copy_ghost_face!(
            blocks, connectivity, Block_Nprocs, rank_offsets,
            Nx_b, Ny_b, Nz_b, :Q, nvariables, ghost_pool;
            full_range=false,
        )
        for (bid, b) in blocks
            exchange_ghost(
                b.Q, nvariables, block_comms[bid], b.Nx, b.Ny, b.Nz,
                b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
                b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
                b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
                sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
                rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2,
                periodic_faces=_structured_periodic_face_mask(bid),
                rank_coords=(b.rx, b.ry, b.rz),
                rank_dims=Tuple(Block_Nprocs[bid + 1]),
            )
        end
        copy_ghost_face!(
            blocks, connectivity, Block_Nprocs, rank_offsets,
            Nx_b, Ny_b, Nz_b, :Q, nvariables, ghost_pool;
            full_range=true, delta_mode=true,
        )
        return nothing
    end

    function sync_ct_fofc_flags!()
        @static if !(equation_type == :MHD && ct_mode &&
                     ct_first_order_flux_correction)
            return nothing
        end
        gpu_sync()
        copy_ghost_face!(
            blocks,connectivity,Block_Nprocs,rank_offsets,
            Nx_b,Ny_b,Nz_b,:fofc_flag,1,ghost_pool;
            full_range=false,
        )
        for (bid,b) in blocks
            b.fofc_flag === nothing && continue
            exchange_ghost(
                b.fofc_flag,1,block_comms[bid],b.Nx,b.Ny,b.Nz,
                b.sbuf_hx,b.sbuf_dx,b.rbuf_hx,b.rbuf_dx,
                b.sbuf_hy,b.sbuf_dy,b.rbuf_hy,b.rbuf_dy,
                b.sbuf_hz,b.sbuf_dz,b.rbuf_hz,b.rbuf_dz;
                sbuf_hx2=b.sbuf_hx2,sbuf_dx2=b.sbuf_dx2,
                rbuf_hx2=b.rbuf_hx2,rbuf_dx2=b.rbuf_dx2,
                periodic_faces=_structured_periodic_face_mask(bid),
                rank_coords=(b.rx,b.ry,b.rz),
                rank_dims=Tuple(Block_Nprocs[bid+1]),
            )
        end
        copy_ghost_face!(
            blocks,connectivity,Block_Nprocs,rank_offsets,
            Nx_b,Ny_b,Nz_b,:fofc_flag,1,ghost_pool;
            full_range=true,delta_mode=true,
        )
        gpu_sync()
        return nothing
    end

    # Bootstrap only: a fresh cell-centered initial condition is still allowed
    # to seed the first face-B construction. Restarted face-B never uses this.
    sync_ct_initial_q_ghosts!() = sync_ct_q_components!(Nprim)

    # Physical boundary kernels need a primitive state before U halo exchange.
    # This is deliberately a local/direct recovery: POINT6 is reserved for the
    # final pass after all U and face-B halos are complete.
    function finalize_ct_provisional_state!(;
        step::Integer=0, rk_stage::Integer=0,
    )
        for (_, b) in blocks
            b.Bx_face === nothing && continue
            ct_recover_cell_b!(b, b.Nx, b.Ny, b.Nz)
            @static if isothermal_mhd
                ct_rebuild_isothermal_carrier!(b, b.Nx, b.Ny, b.Nz)
            end
            reset_positivity_diagnostics!(
                shared_positivity_count, shared_conservation_delta)
            nb = (
                cld(b.Nx, nthreads[1]),
                cld(b.Ny, nthreads[2]),
                cld(b.Nz, nthreads[3]),
            )
            @gpu_launch threads=nthreads blocks=nb finalize_structured_primitive(
                b.U, b.Q, b.Vol, shared_positivity_count,
                shared_conservation_delta, Int32(b.Nx), Int32(b.Ny),
                Int32(b.Nz))
            @static if ct_primitive_recovery == CT_PRIMITIVE_POINT6
                # The high-order point recovery is not legal before U halo
                # exchange. Use the direct local magnetic state for BC input.
                @gpu_launch threads=nthreads blocks=nb ct_update_q_b_kernel!(
                    b.Q, b.U, Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
            end
        end
        gpu_sync()
        return nothing
    end

    function finalize_ct_interior_state!(;
        step::Integer=0, rk_stage::Integer=0,
    )
        ct_derivation_halo_plan === nothing && error(
            "CT derivation halo was not initialized",
        )
        refresh_ct_derivation_dynamic!(ct_derivation_halo_plan, blocks)
        for (bid, b) in blocks
            b.Bx_face === nothing && continue
            ct_recover_cell_b!(b, b.Nx, b.Ny, b.Nz)
            @static if isothermal_mhd
                # U[5] is a compatibility carrier in the isothermal closure.
                # Rebuild it only after the authoritative face-B is complete.
                ct_rebuild_isothermal_carrier!(b, b.Nx, b.Ny, b.Nz)
            end
            reset_positivity_diagnostics!(
                shared_positivity_count, shared_conservation_delta)
            nb = (
                cld(b.Nx, nthreads[1]), cld(b.Ny, nthreads[2]),
                cld(b.Nz, nthreads[3]),
            )
            @gpu_launch threads=nthreads blocks=nb finalize_structured_primitive(
                b.U, b.Q, b.Vol, shared_positivity_count,
                shared_conservation_delta, Int32(b.Nx), Int32(b.Ny),
                Int32(b.Nz))
            rank_coordinates = (Int(b.rx), Int(b.ry), Int(b.rz))
            rank_dimensions = Tuple(Int.(Block_Nprocs[bid + 1]))
            physical_faces = ntuple(Val(6)) do face_id
                direction = fld(face_id + 1, 2)
                owns_face = isodd(face_id) ?
                    rank_coordinates[direction] == 0 :
                    rank_coordinates[direction] == rank_dimensions[direction] - 1
                boundary_type = Int32(get(
                    face_bc, (bid, face_id), BC_INTERBLOCK,
                ))
                owns_face && _ct_is_physical_boundary_type(boundary_type)
            end
            @static if ct_primitive_recovery == CT_PRIMITIVE_POINT6
                ct_update_q_b!(
                    b,b.Nx,b.Ny,b.Nz;
                    positivity_meta=shared_ct_pos_meta,
                )
                ct_derive_q_point6_topological_active!(
                    b, ct_derivation_halo_plan.halos[bid],
                    b.Nx,b.Ny,b.Nz;
                    physical_faces=physical_faces,
                    positivity_meta=shared_ct_pos_meta,
                )
                ct_derive_q_point6_ghost!(
                    b, ct_derivation_halo_plan.halos[bid],
                    b.Nx,b.Ny,b.Nz;
                    physical_faces=physical_faces,
                    positivity_meta=shared_ct_pos_meta,
                )
            else
                ct_derive_q_direct_ghost!(
                    b, ct_derivation_halo_plan.halos[bid],
                    b.Nx, b.Ny, b.Nz;
                    gamma=structured_task_gamma, gas_constant=FT(Rg),
                    physical_faces=physical_faces,
                )
            end
            gpu_sync()
            positivity_report_or_throw!(
                shared_positivity_count, shared_conservation_delta;
                context="structured CT rank=$world_rank block=$bid " *
                        "step=$step RK=$rk_stage")

            @static if POSITIVITY_STRICT &&
                       !(strict_ct_positivity && ct_mode && splitMethodID == 4)
                ct_reset_positivity_violation!(
                    shared_ct_pos_meta, shared_ct_pos_values)
                @gpu_launch threads=nthreads blocks=nb ct_check_cell_floor_positivity_kernel!(
                    shared_ct_pos_meta, shared_ct_pos_values,
                    b.U, b.Q, b.x, b.y, b.z,
                    FT(γ), FT(density_floor), FT(pressure_floor), FT(Rg),
                    CT_POS_SITE_FINALIZE_FLOOR,
                    Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                gpu_sync()
                ct_check_positivity_or_abort!(
                    shared_ct_pos_meta, shared_ct_pos_values;
                    rank=world_rank, block=bid, step=step,
                    rk_stage=rk_stage,
                )
            end
        end
        return nothing
    end

    function external_ct_boundary_active()
        for (bid, b) in blocks
            b.Bx_face === nothing && continue
            boundary_types, _ = _structured_face_boundary_data(
                face_bc, bc_params, bid,
            )
            nprocs = Block_Nprocs[bid + 1]
            for face_id in 1:6
                owns = if face_id == 1
                    b.rx == 0
                elseif face_id == 2
                    b.rx == nprocs[1] - 1
                elseif face_id == 3
                    b.ry == 0
                elseif face_id == 4
                    b.ry == nprocs[2] - 1
                elseif face_id == 5
                    b.rz == 0
                else
                    b.rz == nprocs[3] - 1
                end
                owns && is_prescribed_external_magnetic_field_bc(
                    boundary_types[face_id],
                ) && return true
            end
        end
        return false
    end

    function apply_external_ct_face_b_boundaries!(;
        include_normal_face::Bool=true,
        background_splitting::Bool=(
            isdefined(Main, :external_magnetic_background_splitting) ?
            Bool(Main.external_magnetic_background_splitting) : false
        ),
    )
        launched = false
        for (bid, b) in blocks
            b.Bx_face === nothing && continue
            boundary_types, boundary_parameters = _structured_face_boundary_data(
                face_bc, bc_params, bid,
            )
            nprocs = Block_Nprocs[bid + 1]
            for face_id in 1:6
                owns = if face_id == 1
                    b.rx == 0
                elseif face_id == 2
                    b.rx == nprocs[1] - 1
                elseif face_id == 3
                    b.ry == 0
                elseif face_id == 4
                    b.ry == nprocs[2] - 1
                elseif face_id == 5
                    b.rz == 0
                else
                    b.rz == nprocs[3] - 1
                end
                if owns && is_prescribed_external_magnetic_field_bc(
                    boundary_types[face_id],
                )
                    boundary_axis = (face_id + 1) ÷ 2
                    side = isodd(face_id) ? 0 : 1
                    ct_fill_external_field_face_b!(
                        b, b.Nx, b.Ny, b.Nz,
                        boundary_axis, side, boundary_parameters[face_id],
                        include_normal_face=include_normal_face,
                        background_splitting=background_splitting,
                        preserve_perturbation=
                            external_magnetic_split_preserves_perturbation(
                                boundary_types[face_id],
                            ),
                    )
                    launched = true
                end
            end
        end
        launched && gpu_sync()
        return nothing
    end

    function apply_insulating_ct_face_b_boundaries!(;
        include_normal_face::Bool=false,
    )
        launched = false
        for (bid, b) in blocks
            b.Bx_face === nothing && continue
            boundary_types, _ = _structured_face_boundary_data(
                face_bc, bc_params, bid,
            )
            nprocs = Block_Nprocs[bid + 1]
            for face_id in 1:6
                owns = if face_id == 1
                    b.rx == 0
                elseif face_id == 2
                    b.rx == nprocs[1] - 1
                elseif face_id == 3
                    b.ry == 0
                elseif face_id == 4
                    b.ry == nprocs[2] - 1
                elseif face_id == 5
                    b.rz == 0
                else
                    b.rz == nprocs[3] - 1
                end
                if owns && boundary_types[face_id] ==
                           Int32(BC_MHD_INSULATING_WALL)
                    boundary_axis = (face_id + 1) ÷ 2
                    side = isodd(face_id) ? 0 : 1
                    ct_fill_insulating_wall_face_b!(
                        b, b.Nx, b.Ny, b.Nz, boundary_axis, side;
                        include_normal_face=include_normal_face,
                    )
                    launched = true
                end
            end
        end
        launched && gpu_sync()
        return nothing
    end

    function sync_ct_face_b_halos!(;
        include_normal_face::Bool=true,
    )
        # Face ghost construction is a boundary preparation step only.  It
        # may read the current Q[B] to impose physical zero-gradient faces,
        # but it must not be treated as the authoritative cell-B recovery.
        for (bid, b) in blocks
            if b.Bx_face === nothing
                continue
            end
            ct_fill_zerograd_face_b!(b, b.Nx, b.Ny, b.Nz)
            local_periodic = ct_local_periodic_directions(
                bid,face_bc,Block_Nprocs,BC_PERIODIC,
            )
            ct_fill_periodic_face_b!(
                b, b.Nx, b.Ny, b.Nz, local_periodic,
            )
        end
        ct_sync_rank_face_halos!(blocks, block_comms, Block_Nprocs)
        ct_sync_interface_face_halos!(blocks, ct_sync_plan)
        # Physical insulating walls own their ghost face fluxes. Construct
        # their odd/even magnetic extension only after every communication
        # route is complete, then derive Q[B] from this final face state.
        apply_insulating_ct_face_b_boundaries!()
        # Physical prescribed-field sheets are the final owner of their
        # boundary face flux.  Applying them after all halo routes prevents a
        # neighboring block/rank from replacing the analytic boundary value.
        apply_external_ct_face_b_boundaries!(;
            include_normal_face=include_normal_face,
            background_splitting=ct_background_ready[],
        )
        return nothing
    end

    function check_ct_sync_transition!(step::Integer, rk_stage::Integer)
        @static if POSITIVITY_STRICT && ct_mode
            # The check runs only after face-B recovery and branch-specific
            # energy finalization. Adiabatic U[5] is intentionally untouched.
            for (bid, b) in blocks
                if b.Bx_face === nothing
                    continue
                end
                ct_reset_positivity_violation!(
                    shared_ct_pos_meta, shared_ct_pos_values)
                nb = (
                    cld(b.Nx, nthreads[1]),
                    cld(b.Ny, nthreads[2]),
                    cld(b.Nz, nthreads[3]),
                )
                @gpu_launch threads=nthreads blocks=nb ct_check_sync_transition_kernel!(
                    shared_ct_pos_meta, shared_ct_pos_values,
                    b.U, b.Q, b.Bx_face, b.By_face, b.Bz_face,
                    b.Areai, b.nxi, b.nyi, b.nzi,
                    b.Areaj, b.nxj, b.nyj, b.nzj,
                    b.Areak, b.nxk, b.nyk, b.nzk, FT(γ),
                    ct_cell_b_recovery,
                    Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    b.B0x_face, b.B0y_face, b.B0z_face)
                gpu_sync()
                ct_check_positivity_or_abort!(
                    shared_ct_pos_meta, shared_ct_pos_values;
                    rank=world_rank, block=bid, step=step,
                    rk_stage=rk_stage,
                )
            end
        end
        return nothing
    end

    function finalize_ct_face_b_stage!(;
        step::Integer, rk_stage::Integer,
    )
        sync_ct_face_b_halos!()
        # Cell state recovery is deliberately deferred to sync_blocks!, whose
        # transaction completes it before any reconstruction or ghost packing.
        return nothing
    end

    function compute_resistive_ct_edge_emf!(
        b; reset_edges::Bool, recompute_face_flux::Bool,
    )
        nb_cc = (
            cld(b.Nx+2NG, nthreads[1]), cld(b.Ny+2NG, nthreads[2]),
            cld(b.Nz+2NG, nthreads[3]),
        )
        nb_ct = (
            cld(b.Nx+1, nthreads[1]), cld(b.Ny+1, nthreads[2]),
            cld(b.Nz+1, nthreads[3]),
        )
        if reset_edges
            fill!(b.Ex_edge, zero(FT))
            fill!(b.Ey_edge, zero(FT))
            fill!(b.Ez_edge, zero(FT))
        end
        @gpu_launch threads=nthreads blocks=nb_cc ct_compute_resistive_cell_emf_kernel!(
            shared_cc_ex, shared_cc_ey, shared_cc_ez, b.Q,
            b.Areai, b.nxi, b.nyi, b.nzi,
            b.Areaj, b.nxj, b.nyj, b.nzj,
            b.Areak, b.nxk, b.nyk, b.nzk,
            b.Vol, Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
        if recompute_face_flux
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
            @gpu_launch threads=threads_visc_i blocks=nb_visc_i viscous_flux_i(
                b.Q, shared_Fvx, b.Areai, b.Areaj, b.Areak,
                b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj,
                b.nxk, b.nyk, b.nzk, b.Vol, b.Nx, b.Ny, b.Nz,
                b.is_interblock, true)
            @gpu_launch threads=threads_visc_j blocks=nb_visc_j viscous_flux_j(
                b.Q, shared_Fvy, b.Areai, b.Areaj, b.Areak,
                b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj,
                b.nxk, b.nyk, b.nzk, b.Vol, b.Nx, b.Ny, b.Nz,
                b.is_interblock, true)
            @gpu_launch threads=threads_visc_k blocks=nb_visc_k viscous_flux_k(
                b.Q, shared_Fvz, b.Areai, b.Areaj, b.Areak,
                b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj,
                b.nxk, b.nyk, b.nzk, b.Vol, b.Nx, b.Ny, b.Nz,
                b.is_interblock, true)
        end
        @gpu_launch threads=nthreads blocks=nb_ct ct_add_resistive_edge_line_emf_kernel!(
            b.Ex_edge, b.Ey_edge, b.Ez_edge,
            shared_Fvx, shared_Fvy, shared_Fvz,
            shared_cc_ex, shared_cc_ey, shared_cc_ez,
            b.Areai, b.nxi, b.nyi, b.nzi,
            b.Areaj, b.nxj, b.nyj, b.nzj,
            b.Areak, b.nxk, b.nyk, b.nzk,
            b.x, b.y, b.z, Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
        return nothing
    end

    function finalize_local_ct_edge_emf!(bid, b)
        ct_enforce_physical_edge_emf!(b, bid, face_bc)
        local_periodic = ct_local_periodic_directions(
            bid,face_bc,Block_Nprocs,BC_PERIODIC,
        )
        nb_ct = (
            cld(b.Nx+1, nthreads[1]), cld(b.Ny+1, nthreads[2]),
            cld(b.Nz+1, nthreads[3]),
        )
        @gpu_launch threads=nthreads blocks=nb_ct ct_sync_periodic_edge_emf_kernel!(
            b.Ex_edge, b.Ey_edge, b.Ez_edge,
            local_periodic[1], local_periodic[2], local_periodic[3],
            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
        return local_periodic
    end

    function set_resistive_ct_face_energy!(b; additive::Bool)
        b.Bx_face === nothing && return nothing
        nb_ct = (
            cld(b.Nx+1, nthreads[1]), cld(b.Ny+1, nthreads[2]),
            cld(b.Nz+1, nthreads[3]),
        )
        @gpu_launch threads=nthreads blocks=nb_ct ct_resistive_energy_flux_i_from_edges_kernel!(
            shared_Fvx, b.Q, b.Bx_face, b.Ex_edge, b.Ey_edge, b.Ez_edge,
            b.Areai, b.nxi, b.nyi, b.nzi, b.x, b.y, b.z,
            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz), additive,
            b.B0x_face)
        @gpu_launch threads=nthreads blocks=nb_ct ct_resistive_energy_flux_j_from_edges_kernel!(
            shared_Fvy, b.Q, b.By_face, b.Ex_edge, b.Ey_edge, b.Ez_edge,
            b.Areaj, b.nxj, b.nyj, b.nzj, b.x, b.y, b.z,
            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz), additive,
            b.B0y_face)
        @gpu_launch threads=nthreads blocks=nb_ct ct_resistive_energy_flux_k_from_edges_kernel!(
            shared_Fvz, b.Q, b.Bz_face, b.Ex_edge, b.Ey_edge, b.Ez_edge,
            b.Areak, b.nxk, b.nyk, b.nzk, b.x, b.y, b.z,
            Int32(b.Nx), Int32(b.Ny), Int32(b.Nz), additive,
            b.B0z_face)
        return nothing
    end

    # The task graph owns the complete two-pass ghost transaction, including
    # the pre-face-B CT bootstrap. No numerical work is hidden in MPI callbacks.
    structured_taskgraph_sync! = function(
        tt_val; step::Integer=0, rk_stage::Integer=0,
    )
        structured_task_context === nothing && error(
            "structured ghost task graph requested before initialization",
        )

        # Boundary kernels must consume the dt belonging to the active
        # transaction. The outer loop can still hold the previous step's dt.
        sync_dt = current_dt
        for task_state_ref in (
            structured_rk_task_state,
            structured_post_task_state,
            structured_resistive_task_state,
            structured_implicit_task_state,
        )
            task_state = task_state_ref[]
            if task_state isa AbstractDict && haskey(task_state, :current_dt)
                sync_dt = task_state[:current_dt]
                break
            end
        end
        structured_task_sync_state[] = (
            tt_val=tt_val,
            step=step,
            rk_stage=rk_stage,
            current_dt=sync_dt,
        )
        begin_structured_task_stage!(
            structured_task_context,
            StructuredTaskEpoch(step, rk_stage, 0);
            seed_resources=[:U_ACTIVE, :FACE_B_HALO],
        )
        try
            run_structured_task_graph!(structured_task_context.runtime)
        finally
            structured_task_sync_state[] = nothing
        end
        for (_, b) in blocks
            @check_nan(
                b.Q, "Q after task graph ghost refresh",
                b.id, world_rank, tt_val,
            )
        end
        return nothing
    end

    function sync_blocks!(
        tt_val; step::Integer=0, rk_stage::Integer=0,
    )
        structured_taskgraph_sync!(
            tt_val; step=step, rk_stage=rk_stage,
        )
        return nothing
    end


    structured_task_callbacks = Dict{Symbol,Function}()
    structured_task_callbacks[:ct_provisional_state] =
        (ctx, rt, node) -> begin
            if structured_task_ct_active && ct_face_state_ready[]
                state = structured_task_sync_state[]
                finalize_ct_provisional_state!(
                    step=state.step, rk_stage=state.rk_stage,
                )
            end
            StructuredTaskDone
        end

    # ------------------------------------------------------------------
    # Explicit RK3 task graph callbacks.  The callbacks intentionally call
    # the existing numerical kernels.  The graph owns only ordering,
    # communication barriers, and shared-workspace exclusivity.
    # ------------------------------------------------------------------
    function _structured_rk_stage_from_node(node)
        match_result = match(r"^rk([1-3])_", String(node.id))
        match_result === nothing && throw(ArgumentError(
            "RK task has no stage in id $(node.id)",
        ))
        return parse(Int, match_result.captures[1])
    end

    function _structured_rk_block_from_node(node)
        match_result = match(r"_b(-?[0-9]+)$", String(node.id))
        match_result === nothing && throw(ArgumentError(
            "RK block task has no block id in $(node.id)",
        ))
        return parse(Int, match_result.captures[1])
    end

    function _structured_rk_a(stage::Integer)
        return stage == 2 ? FT(0.25) :
               (stage == 3 ? FT(2) / FT(3) : one(FT))
    end

    function _structured_rk_fofc_iteration(node)
        match_result = match(r"_fofc_iter([0-9]+)_",String(node.id))
        match_result === nothing && return nothing
        return parse(Int,match_result.captures[1])
    end

    _structured_rk_is_fofc_auxiliary(node) =
        _structured_rk_fofc_iteration(node) !== nothing

    function _structured_rk_fofc_task_active(node)
        iteration = _structured_rk_fofc_iteration(node)
        iteration === nothing && return true
        iteration == 0 && return true
        state = structured_rk_task_state[]
        stage = _structured_rk_stage_from_node(node)
        return state[:fofc_next_iteration][stage] == iteration
    end

    function _structured_rk_junction_epoch(node,state)
        iteration = _structured_rk_fofc_iteration(node)
        iteration === nothing && return (state[:tt],
                                         _structured_rk_stage_from_node(node))
        stage = _structured_rk_stage_from_node(node)
        return (state[:tt],1000+100*stage+iteration)
    end

    structured_rk_task_callbacks = Dict{Symbol,Function}()
    structured_rk_task_callbacks[:rk_prepare] =
        (ctx, rt, node) -> begin
            state = structured_rk_task_state[]
            tt_stage = state[:tt]
            fill!(state[:fofc_next_iteration],0)
            fill!(state[:fofc_last_iteration],0)
            @static if strict_ct_positivity
                ct_reset_positivity!(shared_ct_pos_meta, shared_ct_pos_values)
            end
            for (_, b) in blocks
                copyto!(b.Un, b.U)
                b.fofc_flag === nothing || fill!(b.fofc_flag,-one(FT))
            end

            state[:ct_energy_budget_stage_start] = Dict{Int,Vector{Float64}}()
            state[:ct_energy_budget_prediv] = Dict{Int,Vector{Float64}}()
            if _ct_energy_budget_enabled
                for (bid, b) in blocks
                    b.Bx_face === nothing && continue
                    s = ct_energy_budget_scratch[bid]
                    state[:ct_energy_budget_stage_start][bid] =
                        ct_energy_budget_state_sample!(
                            s.state, b.U, b.Q, b.Vol,
                            b.Nx, b.Ny, b.Nz, nthreads,
                        )
                end
            end

            current_dt = if adaptive_dt
                dt_min = FT(1e10)
                for (_, b) in blocks
                    nb = (
                        cld(b.Nx+2*NG, nthreads[1]),
                        cld(b.Ny+2*NG, nthreads[2]),
                        cld(b.Nz+2*NG, nthreads[3]),
                    )
                    @gpu_launch threads=nthreads blocks=nb compute_dt(
                        b.LTS_dt, b.Q, b.Vol, b.Areai, b.Areaj, b.Areak,
                        b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj,
                        b.nxk, b.nyk, b.nzk, b.Nx, b.Ny, b.Nz,
                    )
                    dt_local = FT(mapreduce(
                        value -> value > zero(FT) ? value : FT(Inf),
                        min, b.LTS_dt,
                    ))
                    dt_min = min(dt_min, dt_local)
                end
                MPI.Allreduce(dt_min, MPI.MIN, MPI.COMM_WORLD)
            else
                FT(dt)
            end
            state[:current_dt] = current_dt

            if equation_type == :MHD && !ct_mode
                cf_max_local = zero(FT)
                for (_, block) in blocks
                    nb_cf = (
                        cld(block.Nx+2*NG, nthreads[1]),
                        cld(block.Ny+2*NG, nthreads[2]),
                        cld(block.Nz+2*NG, nthreads[3]),
                    )
                    @gpu_launch threads=nthreads blocks=nb_cf compute_cf_max_kernel!(
                        block.LTS_dt, block.Q, block.Nx, block.Ny, block.Nz,
                    )
                    local_max = FT(mapreduce(
                        value -> value > zero(FT) ? value : zero(FT),
                        max, block.LTS_dt,
                    ))
                    cf_max_local = max(cf_max_local, local_max)
                end
                global ch_glm_current = max(
                    MPI.Allreduce(cf_max_local, MPI.MAX, MPI.COMM_WORLD),
                    FT(1.0e-10),
                )
            end

            split_source_active = apply_structured_split_sources!(
                blocks, FT(0.5) * current_dt, nthreads,
            )
            state[:split_source_active] = split_source_active
            if split_source_active
                sync_blocks!(state[:active_time]; step=tt_stage, rk_stage=0)
                for (_, block) in blocks
                    copyto!(block.Un, block.U)
                end
            end

            @static if ct_resistive_rkl2_active
                run_structured_resistive_taskgraph_step!(
                    tt_stage, state[:active_time], current_dt;
                    phase=:pre,
                    split_dt=FT(0.5) * current_dt,
                    split_start_time=state[:active_time],
                    split_half=1,
                )
                for (_, b) in blocks
                    copyto!(b.Un, b.U)
                end
            end

            state[:forcex] = forcex
            state[:flowx] = flowx
            state[:cmf_f1_val] = cmf_f1_val
            state[:deschamps_f1_val] = deschamps_f1_val
            state[:deschamps_flowx_val] = deschamps_flowx_val
            if flow_forcing
                if forcing_mode == 1
                    state[:forcex], state[:flowx] = Update_bulk_force_params(
                        blocks, tt_stage, 1, MPI.COMM_WORLD, world_rank,
                    )
                elseif forcing_mode == 2
                    state[:cmf_f1_val] = Update_const_massflux_params(
                        blocks, tt_stage, 1, current_dt,
                        MPI.COMM_WORLD, world_rank,
                    )
                elseif forcing_mode == 3 ||
                       (isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                    forcing_blocks = (
                        isdefined(Main, :cebl_forcing) && Main.cebl_forcing
                    ) ? Dict{Int,Block}(
                        id => b for (id, b) in blocks if id >= 5
                    ) : blocks
                    state[:deschamps_f1_val], state[:deschamps_flowx_val] =
                        Update_deschamps_pipe_params(
                            forcing_blocks, tt_stage, 1, current_dt,
                            MPI.COMM_WORLD, world_rank,
                        )
                end
            end
            if test_case == "HIT" || test_case == "MHDHIT"
                hit_values = Update_HIT_forcing(
                    blocks, hit_forcing_A, MPI.COMM_WORLD, world_rank,
                    tt_stage, 1,
                )
                state[:hit_u_mean] = hit_values[1]
                state[:hit_v_mean] = hit_values[2]
                state[:hit_w_mean] = hit_values[3]
                state[:hit_Bx_mean] = hit_values[4]
                state[:hit_By_mean] = hit_values[5]
                state[:hit_Bz_mean] = hit_values[6]
            end
            StructuredTaskDone
        end
    structured_rk_task_callbacks[:rk_stage_prepare] =
        (ctx, rt, node) -> begin
            state = structured_rk_task_state[]
            stage = _structured_rk_stage_from_node(node)
            state[:fofc_next_iteration][stage] = 0
            state[:fofc_last_iteration][stage] = 0
            @static if strict_ct_positivity
                ct_reset_positivity!(shared_ct_pos_meta, shared_ct_pos_values)
            end
            for (_,b) in blocks
                b.fofc_flag === nothing || fill!(b.fofc_flag,-one(FT))
            end
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_fofc_iteration_begin] =
        (ctx,rt,node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            ct_reset_fofc_iteration!(
                shared_ct_pos_meta,shared_ct_pos_values,
            )
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_fofc_iteration_finalize] =
        (ctx,rt,node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            state = structured_rk_task_state[]
            stage = _structured_rk_stage_from_node(node)
            iteration = _structured_rk_fofc_iteration(node)
            counts = ct_fofc_iteration_counts(shared_ct_pos_meta)
            global_new = MPI.Allreduce(
                counts.new_cells,MPI.SUM,MPI.COMM_WORLD,
            )
            global_bad = MPI.Allreduce(
                counts.bad_cells,MPI.SUM,MPI.COMM_WORLD,
            )
            global_reduced = MPI.Allreduce(
                counts.reduced_cells,MPI.SUM,MPI.COMM_WORLD,
            )
            global_unrecoverable = MPI.Allreduce(
                counts.unrecoverable_cells,MPI.SUM,MPI.COMM_WORLD,
            )
            state[:fofc_last_iteration][stage] = iteration
            if world_rank == 0 && global_bad > 0
                @printf(
                    "CT_FOFC_CLOSURE step=%d rk=%d iter=%d new=%d reduced=%d bad=%d unrecoverable=%d\n",
                    state[:tt],stage,iteration,global_new,global_reduced,
                    global_bad,global_unrecoverable,
                )
            end
            if global_bad == 0
                state[:fofc_next_iteration][stage] = 0
            elseif global_unrecoverable == 0 &&
                   global_new+global_reduced > 0 &&
                   iteration < Int(ct_fofc_max_iterations)
                state[:fofc_next_iteration][stage] = iteration+1
            else
                if world_rank == 0
                    reason = global_unrecoverable > 0 ?
                        "the no-transport RK anchor is inadmissible" :
                        global_new+global_reduced == 0 ?
                        "the local convex coefficient cannot be reduced " *
                        "further" :
                        "the troubled-cell set did not close within " *
                        "ct_fofc_max_iterations=$(ct_fofc_max_iterations)"
                    printstyled(
                        "CT FOFC closure failed at step=$(state[:tt]) " *
                        "RK=$stage iteration=$iteration: $reason\n",
                        color=:red,
                    )
                    flush(stdout)
                end
                ct_check_positivity_or_abort!(
                    shared_ct_pos_meta,shared_ct_pos_values;
                    rank=world_rank,block=-1,step=state[:tt],
                    rk_stage=stage,
                )
                MPI.Abort(MPI.COMM_WORLD,86)
            end
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_shock] =
        (ctx, rt, node) -> begin
            stage = _structured_rk_stage_from_node(node)
            state = structured_rk_task_state[]
            @static if !ct_troubled_mask_enabled
                stage == 1 || return StructuredTaskDone
            end
            for (_, b) in blocks
                nb_l = (
                    Int32(cld(b.Nx+2*NG, threads_light[1])),
                    Int32(cld(b.Ny+2*NG, threads_light[2])),
                    Int32(cld(b.Nz+2*NG, threads_light[3])),
                )
                @static if ct_troubled_mask_enabled
                    @gpu_launch threads=threads_light blocks=nb_l ct_troubled_cell_sensor_kernel!(
                        b.ϕ,b.Q,b.Vol,
                        b.Areai,b.nxi,b.nyi,b.nzi,
                        b.Areaj,b.nxj,b.nyj,b.nzj,
                        b.Areak,b.nxk,b.nyk,b.nzk,
                        Int32(b.Nx),Int32(b.Ny),Int32(b.Nz),
                    )
                else
                    @gpu_launch threads=threads_light blocks=nb_l shockSensor(
                        b.ϕ,b.Q,b.Nx,b.Ny,b.Nz,
                    )
                end
                @check_nan(
                    b.ϕ,"ϕ after task graph shock sensor",
                    b.id,world_rank,state[:tt],
                )
            end
            StructuredTaskDone
        end
    @static if ct_troubled_mask_enabled
        troubled_mask_task_context = CTTroubledMaskTaskContext(
            blocks, connectivity, Block_Nprocs, rank_offsets,
            Nx_b, Ny_b, Nz_b, ghost_pool, block_comms, face_bc,
        )
        structured_rk_task_callbacks[:rk_shock_halo] =
            let task_context = troubled_mask_task_context
                (ctx, rt, node) -> _ct_troubled_mask_halo_task!(task_context)
            end
    else
        structured_rk_task_callbacks[:rk_shock_halo] =
            (ctx, rt, node) -> begin
                stage = _structured_rk_stage_from_node(node)
                stage == 1 || return StructuredTaskDone
                for (bid,b) in blocks
                    ϕ_4d=reshape(
                        b.ϕ,size(b.ϕ,1),size(b.ϕ,2),size(b.ϕ,3),1,
                    )
                    exchange_ghost(
                        ϕ_4d,1,block_comms[bid],b.Nx,b.Ny,b.Nz,
                        b.sbuf_hx,b.sbuf_dx,b.rbuf_hx,b.rbuf_dx,
                        b.sbuf_hy,b.sbuf_dy,b.rbuf_hy,b.rbuf_dy,
                        b.sbuf_hz,b.sbuf_dz,b.rbuf_hz,b.rbuf_dz;
                        sbuf_hx2=b.sbuf_hx2,sbuf_dx2=b.sbuf_dx2,
                        rbuf_hx2=b.rbuf_hx2,rbuf_dx2=b.rbuf_dx2,
                        periodic_faces=_structured_periodic_face_mask(bid),
                        rank_coords=(b.rx,b.ry,b.rz),
                        rank_dims=Tuple(Block_Nprocs[bid+1]),
                    )
                end
                gpu_sync()
                StructuredTaskDone
            end
    end

    structured_rk_task_callbacks[:rk_troubled_state]=
        (ctx,rt,node)->begin
            @static if ct_troubled_state_enabled
                ct_derivation_halo_plan === nothing && error(
                    "CT troubled-state derivation halo was not initialized",
                )
                for (bid,b) in blocks
                    rank_coordinates=(Int(b.rx),Int(b.ry),Int(b.rz))
                    rank_dimensions=Tuple(Int.(Block_Nprocs[bid+1]))
                    physical_faces=ntuple(Val(6)) do face_id
                        direction=fld(face_id+1,2)
                        owns_face=isodd(face_id) ?
                            rank_coordinates[direction]==0 :
                            rank_coordinates[direction]==rank_dimensions[direction]-1
                        boundary_type=Int32(get(
                            face_bc,(bid,face_id),BC_INTERBLOCK,
                        ))
                        owns_face && _ct_is_physical_boundary_type(boundary_type)
                    end
                    ct_apply_troubled_q!(
                        b,ct_derivation_halo_plan.halos[bid],
                        b.Nx,b.Ny,b.Nz;
                        physical_faces=physical_faces,
                        positivity_meta=shared_ct_pos_meta,
                    )
                end
                gpu_sync()
            end
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_resistive_pre] =
        (ctx, rt, node) -> begin
            bid = _structured_rk_block_from_node(node)
            haskey(blocks, bid) || return StructuredTaskDone
            b = blocks[bid]
            b.Bx_face === nothing && return StructuredTaskDone
            compute_resistive_ct_edge_emf!(
                b; reset_edges=true, recompute_face_flux=true,
            )
            finalize_local_ct_edge_emf!(bid, b)
            StructuredTaskDone
        end
    structured_rk_task_callbacks[:rk_resistive_pre_barrier] =
        (ctx, rt, node) -> begin
            ct_sync_rank_sheets!(
                blocks, block_comms, Block_Nprocs;
                sync_face_flux=false, sync_edges=true,
            )
            ct_sync_interface_sheets!(
                blocks, ct_sync_plan;
                sync_face_flux=false, sync_edges=true,
            )
            ct_sync_junction_edges!(blocks, ct_junction_plan)
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_flux] =
        (ctx, rt, node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            stage = _structured_rk_stage_from_node(node)
            bid = _structured_rk_block_from_node(node)
            haskey(blocks, bid) || return StructuredTaskDone
            b = blocks[bid]
            state = structured_rk_task_state[]
            current_dt_local = state[:current_dt]

            # The task graph launches the next-stage interior reconstruction
            # after the halo transaction has completed and synchronizes the
            # compute stream before boundary fluxes consume the shared buffer.
            use_interior_overlap = stage > 1 && !strict_ct_positivity &&
                !multi_block_mode && length(blocks) == 1 &&
                structured_face_quadrature != STRUCTURED_FACE_POINT6
            if use_interior_overlap
                # Stage 1/2 synchronization launches the next stage's
                # interior work on compute_stream.  Boundary work is the
                # remaining task and must wait for that stream.
                gpu_stream_sync(compute_stream)
                compute_structured_boundary_face_fluxes!(
                    b, current_dt_local, b.ϕ,
                    shared_Fx, shared_Fy, shared_Fz,
                    shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                    shared_Fvx, shared_Fvy, shared_Fvz,
                    world_rank, state[:tt],
                    threads_recon_i, threads_recon_j, threads_recon_k,
                    Int32(stage), shared_ct_pos_meta, shared_ct_pos_values,
                )
            else
                compute_structured_point_face_fluxes!(
                    b, current_dt_local, b.ϕ,
                    shared_Fx, shared_Fy, shared_Fz,
                    shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                    shared_Fvx, shared_Fvy, shared_Fvz,
                    world_rank, state[:tt],
                    threads_recon_i, threads_recon_j, threads_recon_k,
                    threads_visc_i, threads_visc_j, threads_visc_k,
                    Int32(stage), shared_ct_pos_meta, shared_ct_pos_values,
                )
            end
            @static if strict_ct_positivity
                gpu_sync()
                ct_check_positivity_or_abort!(
                    shared_ct_pos_meta, shared_ct_pos_values;
                    rank=world_rank, block=bid, step=state[:tt],
                    rk_stage=stage,
                )
            end

            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_face_average] =
        (ctx, rt, node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            bid = _structured_rk_block_from_node(node)
            haskey(blocks,bid) || return StructuredTaskDone
            b=blocks[bid]
            average_structured_point_face_fluxes!(
                b,b.ϕ,shared_Fx,shared_Fy,shared_Fz,
                shared_Fvx,shared_Fvy,shared_Fvz,shared_ct_pos_meta)
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_fofc_correct] =
        (ctx,rt,node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            bid = _structured_rk_block_from_node(node)
            haskey(blocks,bid) || return StructuredTaskDone
            b = blocks[bid]
            correct_structured_ct_fofc_face_fluxes!(
                b,shared_Fx,shared_Fy,shared_Fz,
                shared_rho_sum_x,shared_rho_sum_y,shared_rho_sum_z,
                shared_ct_pos_meta;
                record_faces=!_structured_rk_is_fofc_auxiliary(node),
            )
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_fofc_scale] =
        (ctx,rt,node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            bid = _structured_rk_block_from_node(node)
            haskey(blocks,bid) || return StructuredTaskDone
            scale_structured_ct_fofc_transport!(
                blocks[bid],shared_Fx,shared_Fy,shared_Fz,
                shared_Fvx,shared_Fvy,shared_Fvz,
            )
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_diffusive_flux] =
        (ctx, rt, node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            stage=_structured_rk_stage_from_node(node)
            bid=_structured_rk_block_from_node(node)
            haskey(blocks,bid) || return StructuredTaskDone
            b=blocks[bid]
            state=structured_rk_task_state[]
            compute_structured_diffusive_face_fluxes!(
                b,shared_Fvx,shared_Fvy,shared_Fvz,
                world_rank,state[:tt],
                threads_visc_i,threads_visc_j,threads_visc_k)
            if _ct_energy_budget_enabled &&
               !_structured_rk_is_fofc_auxiliary(node)
                gpu_sync()
                s=ct_energy_budget_scratch[bid]
                ct_energy_budget_reset_flux!(s.flux)
                ct_energy_budget_accumulate_flux!(
                    s.flux,shared_Fx,shared_Fy,shared_Fz,
                    shared_Fvx,shared_Fvy,shared_Fvz,
                    state[:current_dt],_structured_rk_a(stage),
                    b.Nx,b.Ny,b.Nz,nthreads,Int32(1))
            end
            @static if equation_type == :MHD && ct_mode && resistive &&
                       ct_resistive_main_explicit
                set_resistive_ct_face_energy!(b;additive=true)
                if _ct_energy_budget_enabled &&
                   !_structured_rk_is_fofc_auxiliary(node)
                    s=ct_energy_budget_scratch[bid]
                    ct_energy_budget_accumulate_flux!(
                        s.flux,shared_Fx,shared_Fy,shared_Fz,
                        shared_Fvx,shared_Fvy,shared_Fvz,
                        state[:current_dt],_structured_rk_a(stage),
                        b.Nx,b.Ny,b.Nz,nthreads,Int32(2))
                end
            end
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_edge_emf] =
        (ctx, rt, node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            stage = _structured_rk_stage_from_node(node)
            bid = _structured_rk_block_from_node(node)
            haskey(blocks, bid) || return StructuredTaskDone
            b = blocks[bid]
            state = structured_rk_task_state[]
            fofc_iteration = _structured_rk_fofc_iteration(node)
            b.Bx_face === nothing && return StructuredTaskDone
            nb_ct = (
                cld(b.Nx + 1, nthreads[1]),
                cld(b.Ny + 1, nthreads[2]),
                cld(b.Nz + 1, nthreads[3]),
            )
            if stage == 1 &&
               (!ct_first_order_flux_correction || fofc_iteration == 0)
                ct_backup_face_b!(b, b.Nx, b.Ny, b.Nz)
            end
            local_periodic = ct_local_periodic_directions(
                bid,face_bc,Block_Nprocs,BC_PERIODIC,
            )
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
                    world_rank, bid, Int32(stage), b.Nx, b.Ny, b.Nz,
                    local_periodic,
                )
                ct_weno7_build_edge_line!(
                    b.Ey_edge, point_scratch, Val(2),
                    cache_i, cache_j, cache_k, b.x, b.y, b.z,
                    weno7_fail_meta, weno7_fail_value,
                    world_rank, bid, Int32(stage), b.Nx, b.Ny, b.Nz,
                    local_periodic,
                )
                ct_weno7_build_edge_line!(
                    b.Ez_edge, point_scratch, Val(3),
                    cache_i, cache_j, cache_k, b.x, b.y, b.z,
                    weno7_fail_meta, weno7_fail_value,
                    world_rank, bid, Int32(stage), b.Nx, b.Ny, b.Nz,
                    local_periodic,
                )
                # Preserve WENO7 in smooth regions, but restore SG07's
                # multidimensional upwinding on shock- or FOFC-adjacent edges.
                shock_edge_sensor = ct_troubled_consumer_sensor(
                    b.ϕ,Val(ct_troubled_edge_emf_enabled),
                )
                @gpu_launch threads=nthreads blocks=nb_ct ct_compute_edge_line_emf_from_weno7_cache_kernel!(
                    b.Ex_edge,b.Ey_edge,b.Ez_edge,
                    cache_i,cache_j,cache_k,b.U,b.Q,b.x,b.y,b.z,
                    Int32(b.Nx),Int32(b.Ny),Int32(b.Nz),
                    ct_junction_edge_mask(ct_junction_plan,bid),
                    ct_weno7_sg07_selective_only,shock_edge_sensor,
                    FT(CT_TROUBLED_RECOVERABLE),
                )
                @static if ct_first_order_flux_correction
                    # FOFC replaces selected face fluxes after POINT6 averaging.
                    # Rebuild only those edges from the corrected buffer using
                    # its actual tangential halo; shock-only SG07 above consumes
                    # the original HLLD point cache instead.
                    @gpu_launch threads=nthreads blocks=nb_ct ct_compute_edge_line_emf_kernel!(
                        b.Ex_edge,b.Ey_edge,b.Ez_edge,
                        shared_Fx,shared_Fy,shared_Fz,
                        shared_rho_sum_x,shared_rho_sum_y,shared_rho_sum_z,
                        b.U,
                        b.Areai,b.nxi,b.nyi,b.nzi,
                        b.Areaj,b.nxj,b.nyj,b.nzj,
                        b.Areak,b.nxk,b.nyk,b.nzk,
                        b.Vol,b.x,b.y,b.z,state[:current_dt],
                        Int32(b.Nx),Int32(b.Ny),Int32(b.Nz),b.Q,
                        ct_junction_edge_mask(ct_junction_plan,bid),
                        b.fofc_flag,true,nothing,
                        FT(CT_TROUBLED_RECOVERABLE),
                        Int32(STRUCTURED_FLUX_TANGENTIAL_HALO),
                    )
                end
            else
                @gpu_launch threads=nthreads blocks=nb_ct ct_compute_edge_line_emf_kernel!(
                    b.Ex_edge, b.Ey_edge, b.Ez_edge,
                    shared_Fx, shared_Fy, shared_Fz,
                    shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                    b.U,
                    b.Areai, b.nxi, b.nyi, b.nzi,
                    b.Areaj, b.nxj, b.nyj, b.nzj,
                    b.Areak, b.nxk, b.nyk, b.nzk,
                    b.Vol, b.x, b.y, b.z, state[:current_dt],
                    Int32(b.Nx), Int32(b.Ny), Int32(b.Nz), b.Q,
                    ct_junction_edge_mask(ct_junction_plan,bid),
                )
            end
            @static if ct_emf_scheme == CT_EMF_WENO7_SG07
                if !isempty(ct_junction_plan.topologies)
                    error(
                        "generalized junction UCT currently requires " *
                        "CT_EMF_SG07; WENO7 junction quadrature must not " *
                        "fall back to a midpoint payload",
                    )
                end
            else
                ct_pack_generalized_junction_payloads!(
                    ct_junction_plan,b,
                    _structured_rk_junction_epoch(node,state),
                    shared_Fx, shared_Fy, shared_Fz,
                    shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                    state[:current_dt],
                )
            end
            @static if resistive && ct_resistive_main_explicit
                compute_resistive_ct_edge_emf!(
                    b; reset_edges=false, recompute_face_flux=false,
                )
                ct_pack_generalized_junction_resistive_payloads!(
                    ct_junction_plan,b,
                    _structured_rk_junction_epoch(node,state),
                    shared_Fvx,shared_Fvy,shared_Fvz,
                    shared_cc_ex,shared_cc_ey,shared_cc_ez,
                )
            end
            if b.fofc_flag !== nothing
                @gpu_launch threads=nthreads blocks=nb_ct ct_scale_edge_line_emf_kernel!(
                    b.Ex_edge,b.Ey_edge,b.Ez_edge,b.fofc_flag,
                    Int32(b.Nx),Int32(b.Ny),Int32(b.Nz),
                )
            end
            ct_enforce_physical_edge_emf!(b, bid, face_bc)
            @gpu_launch threads=nthreads blocks=nb_ct ct_sync_periodic_edge_emf_kernel!(
                b.Ex_edge, b.Ey_edge, b.Ez_edge,
                local_periodic[1], local_periodic[2], local_periodic[3],
                Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
            )
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_fofc_detect] =
        (ctx,rt,node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            stage = _structured_rk_stage_from_node(node)
            bid = _structured_rk_block_from_node(node)
            haskey(blocks,bid) || return StructuredTaskDone
            b = blocks[bid]
            state = structured_rk_task_state[]
            mark_structured_ct_fofc_candidate!(
                b,shared_Fx,shared_Fy,shared_Fz,
                shared_Fvx,shared_Fvy,shared_Fvz,
                state[:current_dt],_structured_rk_a(stage),
                shared_ct_pos_meta,shared_ct_pos_values,
            )
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_fofc_commit] =
        (ctx,rt,node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            bid = _structured_rk_block_from_node(node)
            haskey(blocks,bid) || return StructuredTaskDone
            commit_structured_ct_fofc_candidate!(blocks[bid])
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_fofc_flag_sync] =
        (ctx,rt,node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            sync_ct_fofc_flags!()
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_source] =
        (ctx, rt, node) -> begin
            bid = _structured_rk_block_from_node(node)
            haskey(blocks, bid) || return StructuredTaskDone
            b = blocks[bid]
            state = structured_rk_task_state[]
            current_dt_local = state[:current_dt]
            active_time_local = state[:active_time]
            nb_l = (
                Int32(cld(b.Nx+2*NG, threads_light[1])),
                Int32(cld(b.Ny+2*NG, threads_light[2])),
                Int32(cld(b.Nz+2*NG, threads_light[3])),
            )
            if flow_forcing
                b_omega_x = (b.id >= 5 || !_diffrot_volume_force_active) ?
                    zero(FT) :
                    (isdefined(Main, :Omega_x) ? FT(Main.Omega_x) : zero(FT))
                f1_use = (b.id >= 5) ? state[:deschamps_f1_val] : zero(FT)
                flowx_use = (b.id >= 5) ? state[:deschamps_flowx_val] : zero(FT)
                if b_omega_x != zero(FT)
                    @gpu_launch threads=threads_light blocks=nb_l Volume_force_kernel!(
                        shared_dU_forced, b.Q, b.y, b.z,
                        b.Nx, b.Ny, b.Nz, b_omega_x,
                    )
                    if forcing_mode == 1
                        Apply_bulk_force!(
                            shared_dU_forced, b.Q, state[:forcex],
                            state[:flowx], current_dt_local,
                            b.Nx, b.Ny, b.Nz,
                        )
                    elseif forcing_mode == 2
                        Apply_const_massflux_force!(
                            shared_dU_forced, b.Q, state[:cmf_f1_val],
                            b.Nx, b.Ny, b.Nz,
                        )
                    elseif forcing_mode == 3 ||
                           (isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                        Apply_deschamps_pipe_force!(
                            shared_dU_forced, b.Q, f1_use, flowx_use,
                            current_dt_local, b.Nx, b.Ny, b.Nz,
                        )
                    end
                    if _fringe_active
                        @gpu_launch threads=threads_light blocks=nb_l fringe_forcing_kernel!(
                            shared_dU_forced, b.U, b.U_target_fringe,
                            b.fringe_lambda, b.Nx, b.Ny, b.Nz, current_dt_local,
                        )
                    end
                    Apply_trip_force!(
                        shared_dU_forced, b.Q, b.x, b.y, b.z,
                        b.Nx, b.Ny, b.Nz, active_time_local,
                    )
                    @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(
                        b.U, shared_dU_forced, current_dt_local,
                        b.Vol, b.Nx, b.Ny, b.Nz,
                    )
                else
                    nb_f = (
                        cld(b.Nx, nthreads[1]),
                        cld(b.Ny, nthreads[2]),
                        cld(b.Nz, nthreads[3]),
                    )
                    if forcing_mode == 1
                        @gpu_launch threads=nthreads blocks=nb_f zero_dU_forced_kernel!(
                            shared_dU_forced, b.Nx, b.Ny, b.Nz,
                        )
                        Apply_bulk_force!(
                            shared_dU_forced, b.Q, state[:forcex], state[:flowx],
                            current_dt_local, b.Nx, b.Ny, b.Nz,
                        )
                        Apply_trip_force!(
                            shared_dU_forced, b.Q, b.x, b.y, b.z,
                            b.Nx, b.Ny, b.Nz, active_time_local,
                        )
                        @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(
                            b.U, shared_dU_forced, current_dt_local,
                            b.Vol, b.Nx, b.Ny, b.Nz,
                        )
                    elseif forcing_mode == 2
                        @gpu_launch threads=nthreads blocks=nb_f zero_dU_forced_kernel!(
                            shared_dU_forced, b.Nx, b.Ny, b.Nz,
                        )
                        Apply_const_massflux_force!(
                            shared_dU_forced, b.Q, state[:cmf_f1_val],
                            b.Nx, b.Ny, b.Nz,
                        )
                        Apply_trip_force!(
                            shared_dU_forced, b.Q, b.x, b.y, b.z,
                            b.Nx, b.Ny, b.Nz, active_time_local,
                        )
                        @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(
                            b.U, shared_dU_forced, current_dt_local,
                            b.Vol, b.Nx, b.Ny, b.Nz,
                        )
                    elseif forcing_mode == 3 ||
                           (isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                        @gpu_launch threads=nthreads blocks=nb_f fused_deschamps_source_kernel!(
                            b.U, b.Q, f1_use, flowx_use,
                            current_dt_local, b.Nx, b.Ny, b.Nz,
                        )
                        @gpu_launch threads=nthreads blocks=nb_f zero_dU_forced_kernel!(
                            shared_dU_forced, b.Nx, b.Ny, b.Nz,
                        )
                        Apply_trip_force!(
                            shared_dU_forced, b.Q, b.x, b.y, b.z,
                            b.Nx, b.Ny, b.Nz, active_time_local,
                        )
                        @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(
                            b.U, shared_dU_forced, current_dt_local,
                            b.Vol, b.Nx, b.Ny, b.Nz,
                        )
                    end
                end
            end
            if test_case == "HIT" || test_case == "MHDHIT"
                Apply_HIT_forcing!(
                    shared_dU_forced, b.Q, hit_forcing_A,
                    state[:hit_u_mean], state[:hit_v_mean], state[:hit_w_mean],
                    state[:hit_Bx_mean], state[:hit_By_mean], state[:hit_Bz_mean],
                    b.Nx, b.Ny, b.Nz,
                )
                @gpu_launch threads=threads_light blocks=nb_l add_source_kernel!(
                    b.U, shared_dU_forced, current_dt_local,
                    b.Vol, b.Nx, b.Ny, b.Nz,
                )
            end
            if _ct_energy_budget_enabled
                gpu_sync()
                s = ct_energy_budget_scratch[bid]
                state[:ct_energy_budget_prediv][bid] =
                    ct_energy_budget_state_sample!(
                        s.state, b.U, b.Q, b.Vol,
                        b.Nx, b.Ny, b.Nz, nthreads,
                    )
            end
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_divergence] =
        (ctx, rt, node) -> begin
            stage = _structured_rk_stage_from_node(node)
            bid = _structured_rk_block_from_node(node)
            haskey(blocks, bid) || return StructuredTaskDone
            b = blocks[bid]
            state = structured_rk_task_state[]
            current_dt_local = state[:current_dt]
            rk_a = _structured_rk_a(stage)
            nb_f = (
                cld(b.Nx, nthreads[1]),
                cld(b.Ny, nthreads[2]),
                cld(b.Nz, nthreads[3]),
            )
            reset_positivity_diagnostics!(
                shared_positivity_count, shared_conservation_delta,
            )
            if LTS
                nb_l = (
                    Int32(cld(b.Nx+2*NG, threads_light[1])),
                    Int32(cld(b.Ny+2*NG, threads_light[2])),
                    Int32(cld(b.Nz+2*NG, threads_light[3])),
                )
                @gpu_launch threads=threads_light blocks=nb_l div_LTS(
                    b.U, shared_Fx, shared_Fy, shared_Fz,
                    shared_Fvx, shared_Fvy, shared_Fvz, b.LTS_dt, b.Vol,
                    b.Nx, b.Ny, b.Nz,
                )
                @gpu_launch threads=threads_light blocks=nb_l linComb_clip_prim(
                    b.U, b.Un, b.Q, b.Vol,
                    shared_positivity_count, shared_conservation_delta,
                    Ncell_cons, rk_a, one(FT)-rk_a,
                    b.Nx, b.Ny, b.Nz,
                )
            else
                @gpu_launch threads=nthreads blocks=nb_f div_rk_clip_prim(
                    b.U, b.Un, b.Q,
                    shared_Fx, shared_Fy, shared_Fz,
                    shared_Fvx, shared_Fvy, shared_Fvz,
                    shared_positivity_count, shared_conservation_delta,
                    current_dt_local, b.Vol, rk_a,
                    b.Nx, b.Ny, b.Nz,
                )
            end
            gpu_sync()
            @static if !(equation_type == :MHD && ct_mode)
                positivity_report_or_throw!(
                    shared_positivity_count, shared_conservation_delta;
                    context="structured taskgraph rank=$world_rank " *
                            "block=$bid step=$(state[:tt]) RK=$stage",
                )
            end
            @check_nan(b.U, "U after task graph divergence/RK",
                       b.id, world_rank, state[:tt])
            StructuredTaskDone
        end

    structured_rk_task_callbacks[:rk_edge_sync] =
        (ctx, rt, node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            ct_sync_rank_sheets!(
                blocks, block_comms, Block_Nprocs;
                sync_face_flux=false, sync_edges=true,
            )
            ct_sync_interface_sheets!(
                blocks, ct_sync_plan;
                sync_face_flux=false, sync_edges=true,
            )
            StructuredTaskDone
        end
    structured_rk_task_callbacks[:rk_junction_solve] =
        (ctx, rt, node) -> begin
            _structured_rk_fofc_task_active(node) ||
                return StructuredTaskDone
            stage = _structured_rk_stage_from_node(node)
            state = structured_rk_task_state[]
            ct_solve_generalized_junctions!(
                blocks,ct_junction_plan,
                _structured_rk_junction_epoch(node,state);
                require_resistive=resistive && ct_resistive_main_explicit,
            )
            StructuredTaskDone
        end
    structured_rk_task_callbacks[:rk_face_b_update] =
        (ctx, rt, node) -> begin
            bid = _structured_rk_block_from_node(node)
            haskey(blocks, bid) || return StructuredTaskDone
            b = blocks[bid]
            b.Bx_face === nothing && return StructuredTaskDone
            stage = _structured_rk_stage_from_node(node)
            ct_assert_generalized_junction_solution_ready!(
                ct_junction_plan,
                (structured_rk_task_state[][:tt], stage),
            )
            nb_ct = (
                cld(b.Nx + 1, nthreads[1]),
                cld(b.Ny + 1, nthreads[2]),
                cld(b.Nz + 1, nthreads[3]),
            )
            @gpu_launch threads=nthreads blocks=nb_ct ct_update_face_b_from_emf_kernel!(
                b.Bx_face, b.By_face, b.Bz_face,
                b.Bx_face_n, b.By_face_n, b.Bz_face_n,
                b.Ex_edge, b.Ey_edge, b.Ez_edge,
                structured_rk_task_state[][:current_dt], _structured_rk_a(stage),
                Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
            )
            StructuredTaskDone
        end
    structured_rk_task_callbacks[:rk_face_barrier] =
        (ctx, rt, node) -> begin
            state = structured_rk_task_state[]
            finalize_ct_face_b_stage!(
                step=state[:tt], rk_stage=_structured_rk_stage_from_node(node),
            )
            if _ct_energy_budget_enabled
                ct_energy_budget_record_stage!(
                    ct_energy_budget_log, world_rank, MPI.COMM_WORLD,
                    state[:tt], _structured_rk_stage_from_node(node),
                    state[:current_dt],
                    _structured_rk_a(_structured_rk_stage_from_node(node)),
                    blocks, ct_energy_budget_scratch,
                    state[:ct_energy_budget_stage_start],
                    state[:ct_energy_budget_prediv], nthreads,
                )
            end
            StructuredTaskDone
        end
    structured_rk_task_callbacks[:rk_divergence_barrier] =
        (ctx, rt, node) -> StructuredTaskDone
    structured_rk_task_callbacks[:rk_stage_sync] =
        (ctx, rt, node) -> begin
            state = structured_rk_task_state[]
            stage = _structured_rk_stage_from_node(node)
            sync_blocks!(
                state[:active_time];
                step=state[:tt], rk_stage=stage,
            )
            if stage < 3 && !strict_ct_positivity &&
               !multi_block_mode && length(blocks) == 1 &&
               structured_face_quadrature != STRUCTURED_FACE_POINT6
                for (_, b) in blocks
                    compute_structured_interior_face_fluxes!(
                        b, state[:current_dt], b.ϕ,
                        shared_Fx, shared_Fy, shared_Fz,
                        shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                        shared_Fvx, shared_Fvy, shared_Fvz,
                        threads_recon_i, threads_recon_j, threads_recon_k,
                        threads_visc_i, threads_visc_j, threads_visc_k,
                        compute_stream, Int32(stage + 1),
                        shared_ct_pos_meta, shared_ct_pos_values,
                    )
                end
            end
            for (_, b) in blocks
                @check_nan(
                    b.Q, "Q after task graph RK stage sync",
                    b.id, world_rank, state[:tt],
                )
            end
            StructuredTaskDone
        end
    structured_rk_task_callbacks[:rk_positivity] =
        (ctx, rt, node) -> begin
            state = structured_rk_task_state[]
            stage = _structured_rk_stage_from_node(node)
            @static if strict_ct_positivity && ct_mode
                fallback_counts = ct_fallback_counts(
                    shared_ct_pos_meta,
                )
                global_weno_to_plm = MPI.Allreduce(
                    fallback_counts.weno_to_plm, MPI.SUM, MPI.COMM_WORLD,
                )
                global_plm_to_first = MPI.Allreduce(
                    fallback_counts.plm_to_first, MPI.SUM, MPI.COMM_WORLD,
                )
                global_hlld_to_hlle = MPI.Allreduce(
                    fallback_counts.hlld_to_hlle, MPI.SUM, MPI.COMM_WORLD,
                )
                global_characteristic_limited = MPI.Allreduce(
                    fallback_counts.characteristic_limited,
                    MPI.SUM,MPI.COMM_WORLD,
                )
                global_point6_to_ao = MPI.Allreduce(
                    fallback_counts.point6_to_ao,MPI.SUM,MPI.COMM_WORLD,
                )
                global_point6_limited = MPI.Allreduce(
                    fallback_counts.point6_limited,MPI.SUM,MPI.COMM_WORLD,
                )
                global_point6_unrecoverable = MPI.Allreduce(
                    ct_point6_unrecoverable_count(shared_ct_pos_meta),
                    MPI.SUM,MPI.COMM_WORLD,
                )
                global_face_p2a_to_ao = MPI.Allreduce(
                    fallback_counts.face_p2a_to_ao,MPI.SUM,MPI.COMM_WORLD,
                )
                global_face_p2a_to_midpoint = MPI.Allreduce(
                    fallback_counts.face_p2a_to_midpoint,
                    MPI.SUM,MPI.COMM_WORLD,
                )
                global_fofc_cells = MPI.Allreduce(
                    fallback_counts.fofc_cells,MPI.SUM,MPI.COMM_WORLD,
                )
                global_fofc_faces = MPI.Allreduce(
                    fallback_counts.fofc_faces,MPI.SUM,MPI.COMM_WORLD,
                )
                global_fofc_limited = MPI.Allreduce(
                    fallback_counts.fofc_limited,MPI.SUM,MPI.COMM_WORLD,
                )
                if world_rank == 0 &&
                   global_weno_to_plm + global_plm_to_first +
                   global_hlld_to_hlle + global_characteristic_limited +
                   global_point6_to_ao +
                   global_point6_limited + global_point6_unrecoverable +
                   global_face_p2a_to_ao +
                   global_face_p2a_to_midpoint + global_fofc_cells +
                   global_fofc_faces + global_fofc_limited > 0
                    @printf(
                        "CT_FALLBACK taskgraph step=%d rk=%d weno_to_plm=%d plm_to_first=%d hlld_to_hlle=%d characteristic_limited=%d point6_to_ao=%d point6_limited=%d point6_unrecoverable=%d face_p2a_to_ao=%d face_p2a_to_midpoint=%d fofc_cells=%d fofc_faces=%d fofc_limited=%d\n",
                        state[:tt], stage, global_weno_to_plm,
                        global_plm_to_first,global_hlld_to_hlle,
                        global_characteristic_limited,
                        global_point6_to_ao,global_point6_limited,
                        global_point6_unrecoverable,
                        global_face_p2a_to_ao,global_face_p2a_to_midpoint,
                        global_fofc_cells,global_fofc_faces,
                        global_fofc_limited,
                    )
                end
            end
            StructuredTaskDone
        end
    structured_rk_task_callbacks[:rk_trailing_split_source] =
        (ctx, rt, node) -> begin
            state = structured_rk_task_state[]
            split_source_active = apply_structured_split_sources!(
                blocks, FT(0.5) * state[:current_dt], nthreads,
            )
            state[:split_source_active] = split_source_active
            if split_source_active
                sync_blocks!(
                    state[:active_time] + state[:current_dt];
                    step=state[:tt], rk_stage=4,
                )
            end
            StructuredTaskDone
        end
    structured_task_callbacks[:u_interblock_face_copy] =
        (ctx, rt, node) -> begin
            copy_ghost_face!(
                blocks, connectivity, Block_Nprocs, rank_offsets,
                Nx_b, Ny_b, Nz_b, :U, Ncell_cons, ghost_pool;
                full_range=false,
            )
            StructuredTaskDone
        end
    structured_task_callbacks[:u_physical_boundary] =
        (ctx, rt, node) -> begin
            state = structured_task_sync_state[]
            couple_cebl!(blocks, world_rank)
            for (bid, b) in blocks
                Nprocs_b = Block_Nprocs[bid + 1]
                rx_b = multi_block_mode ? 0 : rankx
                ry_b = multi_block_mode ? 0 : ranky
                rz_b = multi_block_mode ? 0 : rankz
                fill_structured_ghost_cells!(
                    b.Q, b.U, rx_b, ry_b, rz_b,
                    b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj,
                    b.nxk, b.nyk, b.nzk,
                    b.Areai, b.Areaj, b.Areak, b.Vol, state.current_dt,
                    b.x, b.y, b.z, b.id, b.Nx, b.Ny, b.Nz,
                    Nprocs_b, state.tt_val, face_bc, bc_params,
                    @isdefined(dsrfg_params) ? dsrfg_params : default_dsrfg_params,
                    get(external_field_caches, bid, nothing),
                )
            end
            gpu_sync()
            StructuredTaskDone
        end
    structured_task_callbacks[:u_rank_exchange] =
        (ctx, rt, node) -> begin
            for (bid, b) in blocks
                exchange_ghost(
                    b.U, Ncell_cons, block_comms[bid], b.Nx, b.Ny, b.Nz,
                    b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
                    b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
                    b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
                    sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
                    rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2,
                    periodic_faces=_structured_periodic_face_mask(bid),
                    rank_coords=(b.rx, b.ry, b.rz),
                    rank_dims=Tuple(Block_Nprocs[bid + 1]),
                )
            end
            gpu_sync()
            StructuredTaskDone
        end
    structured_task_callbacks[:u_interblock_full_copy] =
        (ctx, rt, node) -> begin
            copy_ghost_face!(
                blocks, connectivity, Block_Nprocs, rank_offsets,
                Nx_b, Ny_b, Nz_b, :U, Ncell_cons, ghost_pool;
                full_range=true, delta_mode=true,
            )
            StructuredTaskDone
        end
    structured_task_callbacks[:interface_filter] =
        (ctx, rt, node) -> begin
            state = structured_task_sync_state[]
            intf_interval = @isdefined(intf_filter_interval) ?
                intf_filter_interval : 1
            if interface_filter_enabled && state.step % intf_interval == 0
                smax = FT(0.05)
                for (bid, b) in blocks
                    for ((dst_bid, fid), conn) in connectivity
                        dst_bid == b.id || continue
                        if fid == 1 || fid == 2
                            @static if ct_mode
                                @gpu_launch threads=(16, 16) blocks=(cld(b.Ny, 16), cld(b.Nz, 16)) ct_interface_filter_kernel!(
                                    b.U, b.Bx_face, b.By_face, b.Bz_face,
                                    b.Areai, b.nxi, b.nyi, b.nzi,
                                    b.Areaj, b.nxj, b.nyj, b.nzj,
                                    b.Areak, b.nxk, b.nyk, b.nzk,
                                    b.Nx, b.Ny, b.Nz, fid, 4,
                                    smax, b.lin_phi_i, structured_task_gamma,
                                    FT(density_floor), FT(pressure_floor),
                                    b.B0x_face, b.B0y_face, b.B0z_face)
                            else
                                @gpu_launch threads=(16, 16) blocks=(cld(b.Ny, 16), cld(b.Nz, 16)) interface_filter_kernel!(
                                    b.U, b.Nx, b.Ny, b.Nz, fid, 4,
                                    smax, b.lin_phi_i)
                            end
                        elseif fid == 3 || fid == 4
                            @static if ct_mode
                                @gpu_launch threads=(16, 16) blocks=(cld(b.Nx, 16), cld(b.Nz, 16)) ct_interface_filter_kernel!(
                                    b.U, b.Bx_face, b.By_face, b.Bz_face,
                                    b.Areai, b.nxi, b.nyi, b.nzi,
                                    b.Areaj, b.nxj, b.nyj, b.nzj,
                                    b.Areak, b.nxk, b.nyk, b.nzk,
                                    b.Nx, b.Ny, b.Nz, fid, 4,
                                    smax, b.lin_phi_j, structured_task_gamma,
                                    FT(density_floor), FT(pressure_floor),
                                    b.B0x_face, b.B0y_face, b.B0z_face)
                            else
                                @gpu_launch threads=(16, 16) blocks=(cld(b.Nx, 16), cld(b.Nz, 16)) interface_filter_kernel!(
                                    b.U, b.Nx, b.Ny, b.Nz, fid, 4,
                                    smax, b.lin_phi_j)
                            end
                        elseif fid == 5 || fid == 6
                            @static if ct_mode
                                @gpu_launch threads=(16, 16) blocks=(cld(b.Nx, 16), cld(b.Ny, 16)) ct_interface_filter_kernel!(
                                    b.U, b.Bx_face, b.By_face, b.Bz_face,
                                    b.Areai, b.nxi, b.nyi, b.nzi,
                                    b.Areaj, b.nxj, b.nyj, b.nzj,
                                    b.Areak, b.nxk, b.nyk, b.nzk,
                                    b.Nx, b.Ny, b.Nz, fid, 4,
                                    smax, b.lin_phi_k, structured_task_gamma,
                                    FT(density_floor), FT(pressure_floor),
                                    b.B0x_face, b.B0y_face, b.B0z_face)
                            else
                                @gpu_launch threads=(16, 16) blocks=(cld(b.Nx, 16), cld(b.Ny, 16)) interface_filter_kernel!(
                                    b.U, b.Nx, b.Ny, b.Nz, fid, 4,
                                    smax, b.lin_phi_k)
                            end
                        end
                    end
                end
                gpu_sync()
            end
            StructuredTaskDone
        end
    structured_task_callbacks[:ct_point6_state] =
        (ctx, rt, node) -> begin
            @static if ct_primitive_recovery != CT_PRIMITIVE_POINT6
                for (_, b) in blocks
                    b.Bx_face === nothing && continue
                    ct_recover_cell_b!(
                        b, b.Nx, b.Ny, b.Nz; include_ghosts=true,
                    )
                end
                gpu_sync()
                for (_, b) in blocks
                    b.Bx_face === nothing && continue
                    nb_loc = (
                        cld(b.Nx+2*NG, nthreads[1]),
                        cld(b.Ny+2*NG, nthreads[2]),
                        cld(b.Nz+2*NG, nthreads[3]),
                    )
                    @gpu_launch threads=nthreads blocks=nb_loc c2Prim_ghost(
                        b.U, b.Q, b.Nx, b.Ny, b.Nz,
                    )
                end
                gpu_sync()
            end
            state = structured_task_sync_state[]
            finalize_ct_interior_state!(
                step=state.step, rk_stage=state.rk_stage,
            )
            StructuredTaskDone
        end
    structured_task_callbacks[:ct_direct_state] =
        structured_task_callbacks[:ct_point6_state]
    structured_task_callbacks[:primitive_halo] =
        (ctx, rt, node) -> begin
            for (_, b) in blocks
                nb_loc = (
                    cld(b.Nx+2*NG, nthreads[1]),
                    cld(b.Ny+2*NG, nthreads[2]),
                    cld(b.Nz+2*NG, nthreads[3]),
                )
                @gpu_launch threads=nthreads blocks=nb_loc c2Prim_ghost(
                    b.U, b.Q, b.Nx, b.Ny, b.Nz,
                )
            end
            gpu_sync()
            StructuredTaskDone
        end
    structured_task_callbacks[:ct_physical_state] =
        (ctx, rt, node) -> begin
            for (bid, b) in blocks
                b.Bx_face === nothing && continue
                Nprocs_b = Block_Nprocs[bid + 1]
                rx_b = Int(b.rx)
                ry_b = Int(b.ry)
                rz_b = Int(b.rz)
                rank_coordinates = (rx_b,ry_b,rz_b)
                rank_dimensions = Tuple(Int.(Nprocs_b))
                physical_faces = ntuple(Val(6)) do face_id
                    direction = fld(face_id+1,2)
                    owns_face = isodd(face_id) ?
                        rank_coordinates[direction] == 0 :
                        rank_coordinates[direction] == rank_dimensions[direction]-1
                    boundary_type = Int32(get(
                        face_bc,(bid,face_id),BC_INTERBLOCK,
                    ))
                    owns_face && _ct_is_physical_boundary_type(boundary_type)
                end
                ct_recover_physical_ghost_b!(
                    b,b.Nx,b.Ny,b.Nz; physical_faces=physical_faces,
                )
                if ct_background_split_enabled
                    prescribed_faces = ntuple(Val(6)) do face_id
                        physical_faces[face_id] &&
                        is_prescribed_external_magnetic_field_bc(Int32(get(
                            face_bc,(bid,face_id),BC_INTERBLOCK,
                        ))) &&
                        !external_magnetic_split_preserves_perturbation(
                            Int32(get(face_bc,(bid,face_id),BC_INTERBLOCK)),
                        )
                    end
                    ct_restore_prescribed_background_ghost_b!(
                        b,b.Nx,b.Ny,b.Nz;
                        prescribed_faces=prescribed_faces,
                    )
                end
                finalize_structured_ct_physical_ghost_energy!(
                    b.U, b.Q, rx_b, ry_b, rz_b, b.id,
                    b.Nx, b.Ny, b.Nz, Nprocs_b, face_bc, bc_params,
                    background_cell=b.B0_cell,
                )
            end
            gpu_sync()
            StructuredTaskDone
        end
    structured_task_callbacks[:ct_positivity] =
        (ctx, rt, node) -> begin
            state = structured_task_sync_state[]
            check_ct_sync_transition!(state.step, state.rk_stage)
            @static if strict_ct_positivity && ct_mode && splitMethodID == 4
                for (bid, b) in blocks
                    ct_reset_positivity_violation!(
                        shared_ct_pos_meta, shared_ct_pos_values)
                    nb_loc = (
                        cld(b.Nx+2*NG, nthreads[1]),
                        cld(b.Ny+2*NG, nthreads[2]),
                        cld(b.Nz+2*NG, nthreads[3]),
                    )
                    @static if ct_primitive_recovery == CT_PRIMITIVE_POINT6
                        @gpu_launch threads=nthreads blocks=nb_loc ct_check_point_primitive_positivity_kernel!(
                            shared_ct_pos_meta,shared_ct_pos_values,b.Q,
                            structured_task_gamma,CT_POS_SITE_GHOST_REFRESH,
                            Int32(1),Int32(b.Nx),Int32(b.Ny),Int32(b.Nz),
                        )
                    else
                        @gpu_launch threads=nthreads blocks=nb_loc ct_check_cell_positivity_kernel!(
                            shared_ct_pos_meta,shared_ct_pos_values,b.U,b.Q,
                            structured_task_gamma,CT_POS_SITE_GHOST_REFRESH,
                            Int32(1),Int32(b.Nx),Int32(b.Ny),Int32(b.Nz),
                        )
                    end
                    gpu_sync()
                    ct_check_positivity_or_abort!(
                        shared_ct_pos_meta, shared_ct_pos_values;
                        rank=world_rank, block=bid,
                        step=state.step, rk_stage=state.rk_stage,
                    )
                end
            end
            StructuredTaskDone
        end

    # The interface filter mutates active U. Its second halo transaction is
    # represented explicitly in the graph, while reusing the same numerical
    # kernels and communication routines as the raw-U transaction.
    structured_task_callbacks[:filtered_u_interblock_face_copy] =
        structured_task_callbacks[:u_interblock_face_copy]
    structured_task_callbacks[:filtered_u_physical_boundary] =
        structured_task_callbacks[:u_physical_boundary]
    structured_task_callbacks[:filtered_u_rank_exchange] =
        structured_task_callbacks[:u_rank_exchange]
    structured_task_callbacks[:filtered_u_interblock_full_copy] =
        structured_task_callbacks[:u_interblock_full_copy]
    structured_task_callbacks[:filtered_ct_provisional_state] =
        (ctx, rt, node) -> begin
            state = structured_task_sync_state[]
            if structured_task_ct_active && ct_face_state_ready[]
                finalize_ct_provisional_state!(
                    step=state.step, rk_stage=state.rk_stage,
                )
            else
                for (_, b) in blocks
                    nb_loc = (
                        cld(b.Nx+2*NG, nthreads[1]),
                        cld(b.Ny+2*NG, nthreads[2]),
                        cld(b.Nz+2*NG, nthreads[3]),
                    )
                    @gpu_launch threads=nthreads blocks=nb_loc c2Prim_ghost(
                        b.U, b.Q, b.Nx, b.Ny, b.Nz,
                    )
                end
                gpu_sync()
            end
            StructuredTaskDone
        end

    # ------------------------------------------------------------------
    # Committed-state post-processing task graph.
    # These callbacks deliberately reuse the existing filtering, averaging,
    # diagnostic, and I/O routines.  The graph owns their ordering and the
    # callbacks own the established numerical/file formats.
    # ------------------------------------------------------------------
    structured_post_task_callbacks = Dict{Symbol,Function}()

    function _structured_post_filter_axis!(axis::Int, post_time, step::Int)
        filtering_enabled = isdefined(@__MODULE__, :filtering) && filtering
        filtering_enabled || return nothing
        interval = isdefined(@__MODULE__, :filtering_interval) ?
            filtering_interval : 1
        step % interval == 0 || return nothing

        for (bid, b) in blocks
            nb_l = (
                Int32(cld(b.Nx + 2*NG, threads_light[1])),
                Int32(cld(b.Ny + 2*NG, threads_light[2])),
                Int32(cld(b.Nz + 2*NG, threads_light[3])),
            )
            Nprocs_b = Block_Nprocs[bid + 1]
            if axis == 1
                rank_axis = multi_block_mode ? 0 : rankx
                ilo, ihi = structured_filter_limits(
                    1, b, face_bc, Nprocs_b, rank_axis,
                )
                copyto!(b.Un, b.U)
                @gpu_launch threads=threads_light blocks=nb_l linearFilter_x(
                    b.U, b.Un, filtering_s0, b.Nx, b.Ny, b.Nz, ilo, ihi,
                )
            elseif axis == 2
                rank_axis = multi_block_mode ? 0 : ranky
                jlo, jhi = structured_filter_limits(
                    2, b, face_bc, Nprocs_b, rank_axis,
                )
                copyto!(b.Un, b.U)
                @gpu_launch threads=threads_light blocks=nb_l linearFilter_y(
                    b.U, b.Un, filtering_s0, b.Nx, b.Ny, b.Nz, jlo, jhi,
                )
            elseif axis == 3
                rank_axis = multi_block_mode ? 0 : rankz
                klo, khi = structured_filter_limits(
                    3, b, face_bc, Nprocs_b, rank_axis,
                )
                copyto!(b.Un, b.U)
                @gpu_launch threads=threads_light blocks=nb_l linearFilter_z(
                    b.U, b.Un, filtering_s0, b.Nx, b.Ny, b.Nz, klo, khi,
                )
            else
                throw(ArgumentError("structured post filter axis must be 1, 2, or 3"))
            end

            # CT keeps face-B authoritative.  The ordinary branches can
            # recover primitives immediately; CT recovery is performed by the
            # ordered ghost transaction below.
            if !(equation_type == :MHD && ct_mode)
                @gpu_launch threads=threads_light blocks=nb_l c2Prim(
                    b.U, b.Q, b.Nx, b.Ny, b.Nz,
                )
            end
        end
        gpu_sync()
        sync_blocks!(post_time; step=step, rk_stage=4 + axis)
        return nothing
    end

    structured_post_task_callbacks[:post_filter_x] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            _structured_post_filter_axis!(1, state[:committed_time], state[:tt])
            StructuredTaskDone
        end
    structured_post_task_callbacks[:post_filter_y] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            _structured_post_filter_axis!(2, state[:committed_time], state[:tt])
            StructuredTaskDone
        end
    structured_post_task_callbacks[:post_filter_z] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            _structured_post_filter_axis!(3, state[:committed_time], state[:tt])
            StructuredTaskDone
        end

    structured_post_task_callbacks[:post_average_sample] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            step = state[:tt]
            average_enabled = isdefined(Main, :average) && Main.average
            avg_interval = isdefined(Main, :avg_step) ? Main.avg_step : 1
            if average_enabled && step % avg_interval == 0
                global global_avg_count += 1
                favre = isdefined(Main, :avg_density_weighted) &&
                        Main.avg_density_weighted
                for (_, b) in blocks
                    nb_avg = (
                        Int32(cld(b.Nx + 2*NG, threads_light[1])),
                        Int32(cld(b.Ny + 2*NG, threads_light[2])),
                        Int32(cld(b.Nz + 2*NG, threads_light[3])),
                    )
                    @gpu_launch threads=threads_light blocks=nb_avg accumulate_avg_kernel!(
                        b.Q_avg, b.Q, b.U, Int32(global_avg_count), Nprim,
                        b.Nx, b.Ny, b.Nz, NG, favre,
                    )
                end
                gpu_sync()
                if step % (avg_interval * 10) == 0 && world_rank == 0
                    println(">>> Time-averaging sample $(global_avg_count) collected.")
                end
            end
            StructuredTaskDone
        end

    function _structured_update_outlet_control!(step::Int)
        _outlet_sponge_active || return nothing
        n_anchor = isdefined(Main, :sponge_anchor_period) ?
            Main.sponge_anchor_period : 50
        n_update = isdefined(Main, :sponge_target_period) ?
            Main.sponge_target_period : 20
        if step % n_anchor == 0
            use_fixed_p = isdefined(Main, :p_back) && Main.p_back > 0
            for (bid, b) in blocks
                use_fixed_p && continue
                nprocs_b = Block_Nprocs[bid + 1]
                rx_b = b.rx + 1
                ry_b = b.ry + 1
                rz_b = b.rz + 1
                if rx_b == nprocs_b[1] && haskey(bc_params, (bid, 2)) &&
                   (get(face_bc, (bid, 2), -1) == Int32(BC_NSCBC_OUTFLOW) ||
                    get(face_bc, (bid, 2), -1) == Int32(BC_RIEMANN_OUTFLOW))
                    ix_out = b.Nx + NG
                    p_slice = Array(@view b.Q[
                        ix_out, NG+1:b.Ny+NG, NG+1:b.Nz+NG, 5,
                    ])
                    a_slice = Array(@view b.Areai[
                        ix_out+1, NG+1:b.Ny+NG, NG+1:b.Nz+NG,
                    ])
                    num_local = sum(p_slice .* a_slice)
                    den_local = sum(a_slice)
                else
                    num_local = zero(FT)
                    den_local = zero(FT)
                end
                subcomm = get(block_comms, bid, nothing)
                if subcomm === nothing
                    num_global = Float64(num_local)
                    den_global = Float64(den_local)
                else
                    num_global = MPI.Allreduce(
                        Float64(num_local), MPI.SUM, subcomm,
                    )
                    den_global = MPI.Allreduce(
                        Float64(den_local), MPI.SUM, subcomm,
                    )
                end
                p_average = den_global > 0.0 ?
                    FT(num_global / den_global) : zero(FT)
                if haskey(bc_params, (bid, 2))
                    old_params = bc_params[(bid, 2)]
                    bc_params[(bid, 2)] = Base.setindex(
                        old_params, p_average, BCP_OUTLET_PAVG,
                    )
                end
            end
        end
        if step % n_update == 0
            for (_, b) in blocks
                b.sponge_sigma === nothing && continue
                nb_sp = (
                    cld(b.Nx + 2*NG, nthreads[1]),
                    cld(b.Ny + 2*NG, nthreads[2]),
                    cld(b.Nz + 2*NG, nthreads[3]),
                )
                alpha = isdefined(Main, :sponge_target_alpha) ?
                    Main.sponge_target_alpha : FT(0.01)
                @gpu_launch threads=nthreads blocks=nb_sp update_sponge_U_target_kernel!(
                    b.sponge_U_target, b.U, b.sponge_sigma, alpha,
                    b.Nx, b.Ny, b.Nz,
                )
            end
            gpu_sync()
        end
        return nothing
    end

    function _structured_fringe_residual!()
        _fringe_active || return -1.0
        _fr_num_local = 0.0
        _fr_den_local = 0.0
        for (_, b) in blocks
            b.U_target_fringe === nothing && continue
            ngp = NG + 1
            ix_end = min(NG + 4, b.Nx + NG)
            u_slice = Array(@view b.U[
                ngp:ix_end, ngp:(b.Ny+NG), ngp:(b.Nz+NG), 1:Ncell_cons,
            ])
            ut_slice = Array(@view b.U_target_fringe[
                ngp:ix_end, ngp:(b.Ny+NG), ngp:(b.Nz+NG), 1:Ncell_cons,
            ])
            for index in eachindex(u_slice)
                delta = Float64(u_slice[index]) - Float64(ut_slice[index])
                _fr_num_local += delta * delta
                _fr_den_local += Float64(ut_slice[index])^2
            end
        end
        fr_num_global = MPI.Allreduce(
            Float64(_fr_num_local), MPI.SUM, MPI.COMM_WORLD,
        )
        fr_den_global = MPI.Allreduce(
            Float64(_fr_den_local), MPI.SUM, MPI.COMM_WORLD,
        )
        return fr_den_global > 0.0 ? sqrt(fr_num_global / fr_den_global) : 0.0
    end

    structured_post_task_callbacks[:post_diagnostic] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            step = state[:tt]
            _structured_update_outlet_control!(step)
            if _fringe_active && (step % 100 == 0 || step <= 20)
                state[:fringe_residual] = _structured_fringe_residual!()
            else
                state[:fringe_residual] = -1.0
            end
            # Keep the existing cadence, but make the check a graph node so
            # output cannot observe a state that has not passed the global
            # finite-value reduction.
            if step % step_plt == 0
                local_nan = false
                for (_, b) in blocks
                    nan_found, _ = _has_nan(b.U)
                    local_nan |= nan_found
                end
                nan_detected = MPI.Allreduce(local_nan, MPI.LOR, MPI.COMM_WORLD)
                if nan_detected
                    if world_rank == 0
                        printstyled("Oops, NaN detected at step $step\n", color=:red)
                        flush(stdout)
                    end
                    MPI.Abort(MPI.COMM_WORLD, 1)
                    return StructuredTaskFailed
                end
            end

            # Reuse LTS_dt as a transient, same-shape reduction buffer.  This
            # avoids allocating one full-field entropy array per diagnostic.
            entropy_enabled = step % 100 == 0 || step <= 20
            if entropy_enabled
                local_entropy = 0.0
                for (_, b) in blocks
                    fill!(b.LTS_dt, zero(FT))
                    threads_entropy = (Int32(8), Int32(8), Int32(4))
                    blocks_entropy = (
                        cld(Int32(b.Nx + 2*NG), threads_entropy[1]),
                        cld(Int32(b.Ny + 2*NG), threads_entropy[2]),
                        cld(Int32(b.Nz + 2*NG), threads_entropy[3]),
                    )
                    @gpu_launch threads=threads_entropy blocks=blocks_entropy entropy_total_kernel!(
                        b.LTS_dt, b.U, b.Vol, FT(γ),
                        FT(entropy_reference_density), Int32(NG),
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    )
                    local_entropy += Float64(sum(b.LTS_dt))
                end
                state[:entropy] = MPI.Allreduce(
                    local_entropy, MPI.SUM, MPI.COMM_WORLD,
                )
            end
            StructuredTaskDone
        end

    structured_post_task_callbacks[:post_plot_output] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            step = state[:tt]
            if get(state, :force_plot, false) ||
               step % step_plt == 0 || step == maxStep
                plotFile_multiblock(
                    step, state[:committed_time], blocks, world_rank,
                    Nblocks, Block_Nprocs, block_comms,
                    force=get(state, :force_plot, false),
                )
            end
            StructuredTaskDone
        end
    structured_post_task_callbacks[:post_checkpoint_output] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            step = state[:tt]
            if step % step_chk == 0 || step == maxStep
                checkpointFile(
                    step, state[:committed_time],
                    blocks, world_rank,
                    Block_Nprocs, block_comms,
                )
            end
            StructuredTaskDone
        end
    structured_post_task_callbacks[:post_average_output] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            step = state[:tt]
            if step % avg_total == 0 || step == maxStep
                averageFile(
                    step, blocks, world_rank, Block_Nprocs, block_comms,
                )
            end
            StructuredTaskDone
        end
    structured_post_task_callbacks[:post_in_situ] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            if isdefined(Main, :in_situ_post_process)
                Main.in_situ_post_process(
                    state[:tt], state[:active_time], state[:current_dt],
                    blocks, world_rank, Block_Nprocs, block_comms,
                )
            end
            StructuredTaskDone
        end
    structured_post_task_callbacks[:post_maintenance] =
        (ctx, rt, node) -> begin
            state = structured_post_task_state[]
            step = state[:tt]
            step % 10000 == 0 && GC.gc(true)
            step % 10000 != 0 && step % 2000 == 0 && GC.gc(false)
            if world_rank == 0 && (step % 100 == 0 || step <= 20)
                println(">>> taskgraph post-step complete: step=$step")
                if get(state, :fringe_residual, -1.0) >= 0.0
                    @printf(
                        "  Fringe inlet residual = %.4e\n",
                        state[:fringe_residual],
                    )
                end
                if haskey(state, :entropy)
                    @static if equation_type == :MHD && isothermal_mhd
                        @printf(
                            "  entropy = %.8e (isothermal free energy, J)\n",
                            state[:entropy],
                        )
                    else
                        @printf("  entropy = %.8e\n", state[:entropy])
                    end
                end
                flush(stdout)
            end
            StructuredTaskDone
        end

    function initialize_structured_post_task_context!()
        structured_post_task_context === nothing ||
            return structured_post_task_context
        post_graph = build_structured_post_step_task_graph(
            callbacks=structured_post_task_callbacks,
        )
        structured_post_task_context = StructuredTaskContext(
            post_graph;
            blocks=blocks,
            connectivity=connectivity,
            mpi_requests=block_comms,
            shared_scratch=(shared_Fx, shared_Fy, shared_Fz),
            runtime_options=(equation_type=equation_type,
                             ct_mode=structured_task_ct_active),
        )
        graph_min = MPI.Allreduce(post_graph.signature, MPI.MIN, MPI.COMM_WORLD)
        graph_max = MPI.Allreduce(post_graph.signature, MPI.MAX, MPI.COMM_WORLD)
        graph_min == graph_max || error(
            "structured post-step graph signature differs across MPI ranks: " *
            "min=$graph_min max=$graph_max",
        )
        world_rank == 0 && println(
            ">>> Structured post-step graph signature=$(post_graph.signature)",
        )
        return structured_post_task_context
    end

    function run_structured_post_taskgraph_step!(
        tt_val::Integer, active_time_val, current_dt_val;
        force_plot::Bool=false,
    )
        initialize_structured_post_task_context!()
        structured_post_task_context === nothing && error(
            "post-step task graph requested before initialization",
        )
        state = Dict{Symbol,Any}(
            :tt => Int(tt_val),
            :active_time => FT(active_time_val),
            :current_dt => FT(current_dt_val),
            :committed_time => FT(active_time_val) + FT(current_dt_val),
            :force_plot => force_plot,
        )
        structured_post_task_state[] = state
        begin_structured_task_stage!(
            structured_post_task_context,
            StructuredTaskEpoch(Int(tt_val), 5, 0);
            seed_resources=[STRUCTURED_TASK_POST_STATE],
        )
        try
            run_structured_task_graph!(structured_post_task_context.runtime)
        finally
            structured_post_task_state[] = nothing
        end
        return state
    end

    function initialize_structured_final_plot_task_context!()
        structured_final_plot_task_context === nothing ||
            return structured_final_plot_task_context
        plot_graph = build_structured_plot_output_task_graph(
            callbacks=structured_post_task_callbacks,
        )
        structured_final_plot_task_context = StructuredTaskContext(
            plot_graph;
            blocks=blocks,
            connectivity=connectivity,
            mpi_requests=block_comms,
            runtime_options=(equation_type=equation_type,
                             ct_mode=structured_task_ct_active),
        )
        graph_min = MPI.Allreduce(plot_graph.signature, MPI.MIN, MPI.COMM_WORLD)
        graph_max = MPI.Allreduce(plot_graph.signature, MPI.MAX, MPI.COMM_WORLD)
        graph_min == graph_max || error(
            "structured final-plot graph signature differs across MPI ranks: " *
            "min=$graph_min max=$graph_max",
        )
        return structured_final_plot_task_context
    end

    function run_structured_final_plot_taskgraph!(tt_val::Integer, final_time_val)
        initialize_structured_final_plot_task_context!()
        structured_final_plot_task_context === nothing && error(
            "final-plot task graph requested before initialization",
        )
        final_time = FT(final_time_val)
        state = Dict{Symbol,Any}(
            :tt => Int(tt_val),
            :active_time => final_time,
            :current_dt => zero(FT),
            :committed_time => final_time,
            :force_plot => true,
        )
        structured_post_task_state[] = state
        begin_structured_task_stage!(
            structured_final_plot_task_context,
            StructuredTaskEpoch(Int(tt_val), 6, 0);
            seed_resources=[STRUCTURED_TASK_POST_STATE],
        )
        try
            run_structured_task_graph!(structured_final_plot_task_context.runtime)
        finally
            structured_post_task_state[] = nothing
        end
        return state
    end

    function initialize_structured_rk_task_context!()
        structured_rk_task_context === nothing || return structured_rk_task_context
        task_graph = build_structured_explicit_rk3_task_graph(
            0:(Int(Nblocks) - 1);
            ct_mode=structured_task_ct_active,
            background_split=ct_background_split_enabled,
            troubled_mask=structured_task_ct_active &&
                           ct_troubled_mask_enabled,
            fofc=structured_task_ct_active &&
                 ct_first_order_flux_correction,
            fofc_iterations=Int(ct_fofc_max_iterations),
            resistive_pre=structured_task_ct_active && resistive &&
                          ct_resistive_main_explicit,
            defer_trailing_split=structured_task_ct_active &&
                                  (ct_resistive_sts_active ||
                                   ct_resistive_rkl2_active),
            callbacks=structured_rk_task_callbacks,
        )
        structured_rk_task_context = StructuredTaskContext(
            task_graph;
            blocks=blocks,
            connectivity=connectivity,
            mpi_requests=block_comms,
            shared_scratch=(shared_Fx, shared_Fy, shared_Fz,
                            shared_Fvx, shared_Fvy, shared_Fvz,
                            shared_dU_forced),
            runtime_options=(equation_type=equation_type,
                             ct_mode=structured_task_ct_active,
                             background_split=ct_background_split_enabled,
                             block_count=length(blocks)),
        )
        graph_min = MPI.Allreduce(task_graph.signature, MPI.MIN, MPI.COMM_WORLD)
        graph_max = MPI.Allreduce(task_graph.signature, MPI.MAX, MPI.COMM_WORLD)
        graph_min == graph_max || error(
            "structured RK task graph signature differs across MPI ranks: " *
            "min=$graph_min max=$graph_max",
        )
        world_rank == 0 && println(
            ">>> Structured explicit RK3 task graph signature=" *
            "$(task_graph.signature)",
        )
        return structured_rk_task_context
    end

    function run_structured_explicit_taskgraph_step!(
        tt_val::Integer, active_time_val,
    )
        initialize_structured_rk_task_context!()
        structured_rk_task_context === nothing && error(
            "explicit task graph requested before initialization",
        )
        state = Dict{Symbol,Any}(
            :tt => Int(tt_val),
            :active_time => FT(active_time_val),
            :current_dt => zero(FT),
            :forcex => forcex,
            :flowx => flowx,
            :cmf_f1_val => cmf_f1_val,
            :deschamps_f1_val => deschamps_f1_val,
            :deschamps_flowx_val => deschamps_flowx_val,
            :hit_u_mean => hit_u_mean,
            :hit_v_mean => hit_v_mean,
            :hit_w_mean => hit_w_mean,
            :hit_Bx_mean => zero(FT),
            :hit_By_mean => zero(FT),
            :hit_Bz_mean => zero(FT),
            :split_source_active => false,
            :fofc_next_iteration => zeros(Int,3),
            :fofc_last_iteration => zeros(Int,3),
            :ct_energy_budget_stage_start => Dict{Int,Vector{Float64}}(),
            :ct_energy_budget_prediv => Dict{Int,Vector{Float64}}(),
        )
        structured_rk_task_state[] = state
        seed_resources = Symbol[:RK_Q_HALO_0, :RK_U_ACTIVE_0]
        structured_task_ct_active && push!(seed_resources, :RK_FACE_B_0)
        if ct_background_split_enabled
            ct_background_ready[] || error(
                "CT background cache is not ready before RK flux evaluation",
            )
            push!(seed_resources, STRUCTURED_TASK_CT_BACKGROUND_READY)
        end
        begin_structured_task_stage!(
            structured_rk_task_context,
            StructuredTaskEpoch(Int(tt_val), 0, 0);
            seed_resources=seed_resources,
        )
        try
            run_structured_task_graph!(structured_rk_task_context.runtime)
        finally
            structured_rk_task_state[] = nothing
        end
        return state
    end

    function initialize_structured_task_context!()
        ct_ready = structured_task_ct_active && ct_face_state_ready[]
        if structured_task_context !== nothing &&
           get(structured_task_context.runtime_options, :ct_ready, false) ==
           ct_ready
            initialize_structured_rk_task_context!()
            return structured_task_context
        end
        task_graph = build_structured_ghost_ct_task_graph(
            ct_mode=ct_ready,
            point6=ct_ready &&
                   ct_primitive_recovery == CT_PRIMITIVE_POINT6,
            interface_filter=interface_filter_enabled,
            callbacks=structured_task_callbacks,
        )
        structured_task_context = StructuredTaskContext(
            task_graph;
            blocks=blocks,
            connectivity=connectivity,
            mpi_requests=block_comms,
            shared_scratch=(shared_Fx, shared_Fy, shared_Fz,
                            shared_Fvx, shared_Fvy, shared_Fvz),
            runtime_options=(equation_type=equation_type,
                             ct_mode=ct_ready,
                             ct_ready=ct_ready,
                             point6=ct_ready &&
                                    ct_primitive_recovery == CT_PRIMITIVE_POINT6),
        )
        graph_min = MPI.Allreduce(task_graph.signature, MPI.MIN, MPI.COMM_WORLD)
        graph_max = MPI.Allreduce(task_graph.signature, MPI.MAX, MPI.COMM_WORLD)
        graph_min == graph_max || error(
            "structured task graph signature differs across MPI ranks: " *
            "min=$graph_min max=$graph_max",
        )
        world_rank == 0 && println(
            ">>> Structured ghost task graph ct_ready=$ct_ready " *
            "signature=$(task_graph.signature)",
        )
        initialize_structured_rk_task_context!()
        return structured_task_context
    end

    initialize_structured_task_context!()

    # Phase 2: Create compute stream for comm-compute overlap
    compute_stream = gpu_stream_create()

    local_ct_face_blocks = equation_type == :MHD && ct_mode ?
        [bid for (bid, b) in blocks if b.Bx_face !== nothing] : Int[]
    local_restored_ct_faces = count(
        bid -> bid in restored_ct_face_b, local_ct_face_blocks,
    )
    global_ct_face_blocks = MPI.Allreduce(
        Int64(length(local_ct_face_blocks)), MPI.SUM, MPI.COMM_WORLD,
    )
    global_restored_ct_faces = MPI.Allreduce(
        Int64(local_restored_ct_faces), MPI.SUM, MPI.COMM_WORLD,
    )
    ct_face_restore_mode = ct_checkpoint_face_restore_mode(
        global_restored_ct_faces, global_ct_face_blocks,
    )
    all_ct_faces_restored = ct_face_restore_mode == :complete
    if all_ct_faces_restored && ct_background_split_enabled
        # A split-field checkpoint already stores b_face and B0_face
        # separately. Boundary preparation must therefore preserve the
        # restored perturbation instead of treating b_face as the total field.
        ct_background_ready[] = true
    end

    # Fresh cell-centered magnetic initial data needs gas/U ghosts before the
    # first face construction. A modern CT restart skips this Q-based pass and
    # restores its cache only from checkpoint U plus staggered face fluxes.
    if !all_ct_faces_restored
        sync_blocks!(zero(FT))
    end

    # CT: Initialize face-centered B from cell-centered B for all blocks
    # Must run AFTER ghost sync (so boundary U is valid) and on ALL ranks.
    if equation_type == :MHD && ct_mode
        if !all_ct_faces_restored
            # Bootstrap exception: before face-B exists, cell-centered initial
            # B is input data rather than a derived cache.
            sync_ct_initial_q_ghosts!()
        end
        for (bid, b) in blocks
            if b.Bx_face !== nothing && !(bid in restored_ct_face_b)
                boundary_types, _ = _structured_face_boundary_data(
                    face_bc, bc_params, bid,
                )
                rank_coordinates = (Int(b.rx), Int(b.ry), Int(b.rz))
                rank_dimensions = Tuple(Int.(Block_Nprocs[bid + 1]))
                physical_face_mask = Int32(0)
                for face_id in 1:6
                    direction = fld(face_id + 1, 2)
                    owns_face = isodd(face_id) ?
                        rank_coordinates[direction] == 0 :
                        rank_coordinates[direction] == rank_dimensions[direction] - 1
                    if owns_face &&
                       _ct_is_physical_boundary_type(boundary_types[face_id])
                        physical_face_mask |= Int32(1) << (face_id - 1)
                    end
                end
                ct_init_face_b!(
                    b, b.Nx, b.Ny, b.Nz;
                    physical_face_mask=physical_face_mask,
                )
            end
        end
        used_initial_face_flux_hook = false
        used_initial_edge_integral_hook = false
        if !all_ct_faces_restored
            used_initial_edge_integral_hook =
                _run_configured_external_magnetic_field_process!(
                    Main, blocks;
                    junction_masks=ct_junction_plan.junction_masks,
                    metric_coordinates=initial_metric_coordinates_h,
                )
            if !used_initial_edge_integral_hook
                used_initial_edge_integral_hook =
                    _run_initial_ct_edge_integral_process!(
                        Main, blocks, world_rank, Block_Nprocs, block_comms,
                        initial_metric_coordinates_h,
                    )
            end
            if !used_initial_edge_integral_hook
                used_initial_face_flux_hook =
                    _run_initial_ct_face_flux_process!(
                        Main, blocks, world_rank, Block_Nprocs, block_comms,
                    )
            end
            if !used_initial_edge_integral_hook &&
               !used_initial_face_flux_hook
                for (bid, b) in blocks
                    b.Bx_face === nothing && continue
                    used_initial_edge_integral_hook |=
                        structured_initialize_orszag_tang_edge_integrals!(
                            b;
                            junction_edge_mask=ct_junction_edge_mask(
                                ct_junction_plan, bid,
                            ),
                            coordinates=get(
                                initial_metric_coordinates_h, bid, nothing,
                            ),
                        )
                end
            end
            if used_initial_edge_integral_hook
                # Initial vector-potential integrals are one topological value
                # per physical edge. Select the same canonical pre-curl value
                # used by SCMM. Runtime edge EMFs keep the default mean merge.
                ct_sync_rank_sheets!(
                    blocks, block_comms, Block_Nprocs;
                    sync_face_flux=false, sync_edges=true,
                    edge_merge=CT_EDGE_MERGE_CANONICAL,
                )
                ct_sync_interface_sheets!(
                    blocks, ct_sync_plan;
                    sync_face_flux=false, sync_edges=true,
                    edge_merge=CT_EDGE_MERGE_CANONICAL,
                )
                ct_sync_junction_edges!(
                    blocks, ct_junction_plan;
                    edge_merge=CT_EDGE_MERGE_CANONICAL,
                )
                for (_, b) in blocks
                    b.Bx_face === nothing && continue
                    ct_face_flux_from_edge_integrals!(b)
                end
                gpu_sync()
                used_initial_face_flux_hook = true
            end
        end
        ct_sync_rank_sheets!(
            blocks, block_comms, Block_Nprocs;
            sync_face_flux=true, sync_edges=false,
        )
        ct_sync_interface_sheets!(
            blocks, ct_sync_plan; sync_face_flux=true, sync_edges=false,
        )
        # Apply prescribed physical face fluxes before the initial divergence
        # gate.  The normal physical face is part of the discrete divergence,
        # so checking before this override would validate a different field.
        # A vector-potential initializer already owns the physical normal
        # face flux.  Preserve that discrete-Stokes value during the initial
        # divergence gate; the normal analytic replacement is used during
        # ordinary time advancement.
        initial_include_normal_face = !used_initial_face_flux_hook
        # Bootstrap exception: ct_init_face_b! averages adjacent cell values,
        # but runtime CT deliberately leaves physical Q[B] ghosts undefined.
        # Supply the insulating-wall normal face one-sided from the nearest
        # interior face before the initial divergence gate. Later CT stages
        # keep that physical normal face under the evolution operator's owner.
        apply_insulating_ct_face_b_boundaries!(;
            include_normal_face=initial_include_normal_face,
        )
        apply_external_ct_face_b_boundaries!(;
            include_normal_face=initial_include_normal_face,
            background_splitting=ct_background_ready[],
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
                isempty(restored_ct_face_b) && !used_initial_face_flux_hook &&
                !external_ct_boundary_active()
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
        # Complete every face-B ghost route before recovering cell-centered B.
        sync_ct_face_b_halos!(;
            include_normal_face=initial_include_normal_face,
        )
        # The initial synchronized total face field defines the fixed
        # background.  From this point on Bx/By/Bz hold only the perturbation.
        if ct_background_split_enabled && isempty(restored_ct_face_b)
            activate_ct_background_split!()
        elseif ct_background_split_enabled
            sync_ct_background_face_halos!()
            ct_background_ready[] = true
        end
        # Face-B is now authoritative and cell-centered B has been recovered
        # from the complete synchronized face field.  At initialization only,
        # preserve the configured primitive pressure by closing adiabatic U[5]
        # against that recovered magnetic energy.  Later RK stages keep U[5]
        # conservative and must not take this path.
        if (!@isdefined(isothermal_mhd) || !isothermal_mhd) &&
           !structured_cell_average_initialization_active()
            for (_, b) in blocks
                b.Bx_face === nothing && continue
                ct_recover_cell_b!(b, b.Nx, b.Ny, b.Nz)
                ct_reconcile_initial_energy!(
                    b, b.Nx, b.Ny, b.Nz, FT(γ),
                )
            end
            gpu_sync()
        end
        ct_face_state_ready[] = true
        initialize_structured_task_context!()
        sync_blocks!(zero(FT); step=tt, rk_stage=0)
        if world_rank == 0
            @printf(
                ">>> CT: Face-centered B initialized; relative divB=%.6e.\n",
                global_initial_divb,
            )
        end
    end

    empty!(initial_metric_coordinates_h)

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
            bid in restored_implicit_history && continue
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
                    compute_resistive_ct_edge_emf!(
                        b; reset_edges=true, recompute_face_flux=true)
                    finalize_local_ct_edge_emf!(bid, b)
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
                    nb_cell = (
                        cld(b.Nx, nthreads[1]),
                        cld(b.Ny, nthreads[2]),
                        cld(b.Nz, nthreads[3]),
                    )
                    set_resistive_ct_face_energy!(b; additive=false)
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
                end
                finalize_ct_face_b_stage!(
                    step=step, rk_stage=diagnostic_stage,
                )
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
                        @static if ct_primitive_recovery == CT_PRIMITIVE_POINT6
                            @gpu_launch threads=nthreads blocks=nb_cell ct_check_point_primitive_positivity_kernel!(
                                shared_ct_pos_meta, shared_ct_pos_values,
                                b.Q, FT(γ), CT_POS_SITE_POST_CT_SYNC,
                                Int32(0), Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                        else
                            @gpu_launch threads=nthreads blocks=nb_cell ct_check_cell_positivity_kernel!(
                                shared_ct_pos_meta, shared_ct_pos_values,
                                b.U, b.Q, FT(γ), CT_POS_SITE_POST_CT_SYNC,
                                Int32(0), Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
                        end
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

    @static if ct_resistive_sts_active
        function advance_resistive_ct_sts!(
            step_dt::FT, start_time::FT; step::Integer,
        )
            explicit_dt_local = FT(Inf)
            for (_, b) in blocks
                fill!(b.LTS_dt, zero(FT))
                nb_sts_dt = (
                    cld(b.Nx, nthreads[1]),
                    cld(b.Ny, nthreads[2]),
                    cld(b.Nz, nthreads[3]),
                )
                @gpu_launch threads=nthreads blocks=nb_sts_dt ct_resistive_explicit_dt_kernel!(
                    b.LTS_dt, b.Vol, b.Areai, b.Areaj, b.Areak,
                    Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                )
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
                step_dt, explicit_resistive_dt;
                damping=FT(ct_sts_damping), max_stages=ct_sts_max_stages,
            )
            if world_rank == 0 && (step <= 2 || length(sts_substeps) > 1)
                @printf(
                    "CT_STS taskgraph step=%d stages=%d dt=%.8e explicit_dt=%.8e\n",
                    step, length(sts_substeps), step_dt,
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
                    b.Bx_face === nothing && continue
                    compute_resistive_ct_edge_emf!(
                        b; reset_edges=true, recompute_face_flux=true,
                    )
                    finalize_local_ct_edge_emf!(bid, b)
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
                for (_, b) in blocks
                    b.Bx_face === nothing && continue
                    nb_ct = (
                        cld(b.Nx + 1, nthreads[1]),
                        cld(b.Ny + 1, nthreads[2]),
                        cld(b.Nz + 1, nthreads[3]),
                    )
                    nb_cell = (
                        cld(b.Nx, nthreads[1]),
                        cld(b.Ny, nthreads[2]),
                        cld(b.Nz, nthreads[3]),
                    )
                    set_resistive_ct_face_energy!(b; additive=false)
                    @gpu_launch threads=nthreads blocks=nb_cell ct_resistive_energy_euler_kernel!(
                        b.U, shared_Fvx, shared_Fvy, shared_Fvz, b.Vol,
                        FT(sts_dt), Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    )
                    ct_backup_face_b!(b, b.Nx, b.Ny, b.Nz)
                    @gpu_launch threads=nthreads blocks=nb_ct ct_update_face_b_from_emf_kernel!(
                        b.Bx_face, b.By_face, b.Bz_face,
                        b.Bx_face_n, b.By_face_n, b.Bz_face_n,
                        b.Ex_edge, b.Ey_edge, b.Ez_edge,
                        FT(sts_dt), one(FT),
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    )
                end
                finalize_ct_face_b_stage!(
                    step=step, rk_stage=3 + sts_stage,
                )
                sts_elapsed += FT(sts_dt)
                sync_blocks!(
                    start_time + sts_elapsed;
                    step=step, rk_stage=3 + sts_stage,
                )
                @static if strict_ct_positivity
                    for (_, b) in blocks
                        b.Bx_face === nothing && continue
                        nb_cell = (
                            cld(b.Nx, nthreads[1]),
                            cld(b.Ny, nthreads[2]),
                            cld(b.Nz, nthreads[3]),
                        )
                        @static if ct_primitive_recovery == CT_PRIMITIVE_POINT6
                            @gpu_launch threads=nthreads blocks=nb_cell ct_check_point_primitive_positivity_kernel!(
                                shared_ct_pos_meta, shared_ct_pos_values,
                                b.Q, structured_task_gamma,
                                CT_POS_SITE_POST_CT_SYNC, Int32(0),
                                Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                            )
                        else
                            @gpu_launch threads=nthreads blocks=nb_cell ct_check_cell_positivity_kernel!(
                                shared_ct_pos_meta, shared_ct_pos_values,
                                b.U, b.Q, structured_task_gamma,
                                CT_POS_SITE_POST_CT_SYNC, Int32(0),
                                Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                            )
                        end
                    end
                    gpu_sync()
                    ct_check_positivity_or_abort!(
                        shared_ct_pos_meta, shared_ct_pos_values;
                        rank=world_rank, block=-1, step=step,
                        rk_stage=3 + sts_stage,
                    )
                end
            end
            return length(sts_substeps)
        end
    end

    # ------------------------------------------------------------------
    # Resistive task-graph provider.  The legacy advance_* routines above
    # remain the compatibility implementation.  Taskgraph mode uses the
    # same CT kernels one stage at a time, with each stage represented by a
    # separate catalog node.
    # ------------------------------------------------------------------
    function _structured_resistive_explicit_dt!()
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
                Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
            )
            block_dt = FT(mapreduce(
                value -> value > zero(FT) ? value : FT(Inf),
                min, b.LTS_dt,
            ))
            explicit_dt_local = min(explicit_dt_local, block_dt)
        end
        return if ct_resistive_sts_active
            ct_sts_safety * MPI.Allreduce(
                explicit_dt_local, MPI.MIN, MPI.COMM_WORLD,
            )
        elseif ct_resistive_rkl2_active
            ct_rkl2_safety * MPI.Allreduce(
                explicit_dt_local, MPI.MIN, MPI.COMM_WORLD,
            )
        else
            MPI.Allreduce(explicit_dt_local, MPI.MIN, MPI.COMM_WORLD)
        end
    end

    function _structured_resistive_task_schedule(
        step_dt::FT, start_time::FT, step::Integer;
        split_half::Integer=2,
    )
        explicit_resistive_dt = _structured_resistive_explicit_dt!()
        if ct_resistive_sts_active
            substeps = ct_sts_substeps(
                step_dt, explicit_resistive_dt;
                damping=FT(ct_sts_damping), max_stages=ct_sts_max_stages,
            )
            if world_rank == 0 && (step <= 2 || length(substeps) > 1)
                @printf(
                    "CT_STS taskgraph step=%d stages=%d dt=%.8e explicit_dt=%.8e\n",
                    step, length(substeps), step_dt,
                    explicit_resistive_dt,
                )
            end
            return Dict{Symbol,Any}(
                :kind => :sts,
                :stage_count => length(substeps),
                :substeps => substeps,
                :elapsed => zero(FT),
                :step_dt => step_dt,
                :start_time => start_time,
                :split_half => split_half,
            )
        elseif ct_resistive_rkl2_active
            stage_count = ct_rkl2_stage_count(
                step_dt, explicit_resistive_dt;
                max_stages=ct_rkl2_max_stages,
            )
            if world_rank == 0 && (step <= 2 || stage_count > 2)
                @printf(
                    "CT_RKL2 taskgraph step=%d stages=%d dt=%.8e explicit_dt=%.8e\n",
                    step, stage_count, step_dt, explicit_resistive_dt,
                )
            end
            return Dict{Symbol,Any}(
                :kind => :rkl2,
                :stage_count => stage_count,
                :coefficients => ct_rkl2_coefficients(stage_count, FT),
                :step_dt => step_dt,
                :start_time => start_time,
                :split_half => split_half,
            )
        else
            return Dict{Symbol,Any}(
                :kind => :none,
                :stage_count => 1,
                :step_dt => step_dt,
                :start_time => start_time,
                :split_half => split_half,
            )
        end
    end

    function _structured_resistive_task_positivity!(
        step::Integer, diagnostic_stage::Integer,
    )
        @static if strict_ct_positivity
            for (_, b) in blocks
                b.Bx_face === nothing && continue
                nb_cell = (
                    cld(b.Nx, nthreads[1]),
                    cld(b.Ny, nthreads[2]),
                    cld(b.Nz, nthreads[3]),
                )
                @static if ct_primitive_recovery == CT_PRIMITIVE_POINT6
                    @gpu_launch threads=nthreads blocks=nb_cell ct_check_point_primitive_positivity_kernel!(
                        shared_ct_pos_meta, shared_ct_pos_values,
                        b.Q, structured_task_gamma,
                        CT_POS_SITE_POST_CT_SYNC, Int32(0),
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    )
                else
                    @gpu_launch threads=nthreads blocks=nb_cell ct_check_cell_positivity_kernel!(
                        shared_ct_pos_meta, shared_ct_pos_values,
                        b.U, b.Q, structured_task_gamma,
                        CT_POS_SITE_POST_CT_SYNC, Int32(0),
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    )
                end
            end
            gpu_sync()
            ct_check_positivity_or_abort!(
                shared_ct_pos_meta, shared_ct_pos_values;
                rank=world_rank, block=-1, step=step,
                rk_stage=diagnostic_stage,
            )
        end
    end

    function _structured_resistive_task_stage!(
        state::Dict{Symbol,Any}, stage::Integer,
    )
        kind = state[:kind]
        kind == :none && return nothing
        step = state[:tt]
        if kind == :sts
            sts_dt = FT(state[:substeps][stage])
            @static if strict_ct_positivity
                ct_reset_positivity!(
                    shared_ct_pos_meta, shared_ct_pos_values,
                )
            end
            for (bid, b) in blocks
                b.Bx_face === nothing && continue
                compute_resistive_ct_edge_emf!(
                    b; reset_edges=true, recompute_face_flux=true,
                )
                finalize_local_ct_edge_emf!(bid, b)
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
            for (_, b) in blocks
                b.Bx_face === nothing && continue
                nb_ct = (
                    cld(b.Nx + 1, nthreads[1]),
                    cld(b.Ny + 1, nthreads[2]),
                    cld(b.Nz + 1, nthreads[3]),
                )
                nb_cell = (
                    cld(b.Nx, nthreads[1]),
                    cld(b.Ny, nthreads[2]),
                    cld(b.Nz, nthreads[3]),
                )
                set_resistive_ct_face_energy!(b; additive=false)
                @gpu_launch threads=nthreads blocks=nb_cell ct_resistive_energy_euler_kernel!(
                    b.U, shared_Fvx, shared_Fvy, shared_Fvz, b.Vol,
                    sts_dt, Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                )
                ct_backup_face_b!(b, b.Nx, b.Ny, b.Nz)
                @gpu_launch threads=nthreads blocks=nb_ct ct_update_face_b_from_emf_kernel!(
                    b.Bx_face, b.By_face, b.Bz_face,
                    b.Bx_face_n, b.By_face_n, b.Bz_face_n,
                    b.Ex_edge, b.Ey_edge, b.Ez_edge,
                    sts_dt, one(FT),
                    Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                )
            end
            finalize_ct_face_b_stage!(
                step=step, rk_stage=3 + stage,
            )
            state[:elapsed] += sts_dt
            sync_blocks!(
                state[:start_time] + state[:elapsed];
                step=step, rk_stage=3 + stage,
            )
            _structured_resistive_task_positivity!(step, 3 + stage)
        elseif kind == :rkl2
            coefficients = state[:coefficients]
            first_stage = stage == 1
            coefficient = first_stage ? nothing :
                coefficients.stages[stage - 1]
            rhs_dt_coefficient = first_stage ?
                coefficients.mu_tilde1 * state[:step_dt] :
                coefficient.mu_tilde * state[:step_dt]
            stage_abscissa = first_stage ?
                coefficients.first_abscissa : coefficient.abscissa
            split_half = state[:split_half]
            diagnostic_stage = split_half * 100 + stage
            @static if strict_ct_positivity
                ct_reset_positivity!(
                    shared_ct_pos_meta, shared_ct_pos_values,
                )
            end
            for (bid, b) in blocks
                b.Bx_face === nothing && continue
                compute_resistive_ct_edge_emf!(
                    b; reset_edges=true, recompute_face_flux=true,
                )
                finalize_local_ct_edge_emf!(bid, b)
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
                    cld(b.Nx + 1, nthreads[1]),
                    cld(b.Ny + 1, nthreads[2]),
                    cld(b.Nz + 1, nthreads[3]),
                )
                nb_cell = (
                    cld(b.Nx, nthreads[1]),
                    cld(b.Ny, nthreads[2]),
                    cld(b.Nz, nthreads[3]),
                )
                set_resistive_ct_face_energy!(b; additive=false)
                if first_stage
                    @gpu_launch threads=nthreads blocks=nb_cell ct_rkl2_energy_first_stage_kernel!(
                        b.U, b.Un,
                        buffers.energy_previous2, buffers.energy_first,
                        shared_Fvx, shared_Fvy, shared_Fvz, b.Vol,
                        rhs_dt_coefficient,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    )
                else
                    @gpu_launch threads=nthreads blocks=nb_cell ct_rkl2_energy_stage_kernel!(
                        b.U, b.Un,
                        buffers.energy_previous2, buffers.energy_first,
                        shared_Fvx, shared_Fvy, shared_Fvz, b.Vol,
                        rhs_dt_coefficient, coefficient.mu,
                        coefficient.nu, coefficient.base,
                        coefficient.gamma_ratio,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    )
                end
                if first_stage
                    @gpu_launch threads=nthreads blocks=nb_ct ct_rkl2_face_first_stage_kernel!(
                        b.Bx_face, b.By_face, b.Bz_face,
                        b.Bx_face_n, b.By_face_n, b.Bz_face_n,
                        buffers.bx_previous2, buffers.by_previous2,
                        buffers.bz_previous2,
                        buffers.bx_first, buffers.by_first, buffers.bz_first,
                        b.Ex_edge, b.Ey_edge, b.Ez_edge,
                        rhs_dt_coefficient,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    )
                else
                    @gpu_launch threads=nthreads blocks=nb_ct ct_rkl2_face_stage_kernel!(
                        b.Bx_face, b.By_face, b.Bz_face,
                        b.Bx_face_n, b.By_face_n, b.Bz_face_n,
                        buffers.bx_previous2, buffers.by_previous2,
                        buffers.bz_previous2,
                        buffers.bx_first, buffers.by_first, buffers.bz_first,
                        b.Ex_edge, b.Ey_edge, b.Ez_edge,
                        rhs_dt_coefficient, coefficient.mu,
                        coefficient.nu, coefficient.base,
                        coefficient.gamma_ratio,
                        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
                    )
                end
            end
            finalize_ct_face_b_stage!(
                step=step, rk_stage=diagnostic_stage,
            )
            sync_blocks!(
                state[:start_time] + stage_abscissa * state[:step_dt];
                step=step, rk_stage=diagnostic_stage,
            )
            _structured_resistive_task_positivity!(step, diagnostic_stage)
        end
        return nothing
    end

    # Provider-level resistive transaction.  The graph owns the physical-step
    # boundary and split-source reconciliation; stage callbacks own only one
    # numerical STS/RKL2 substep.
    structured_resistive_task_callbacks = Dict{Symbol,Function}()
    structured_resistive_task_callbacks[:resistive_prepare] =
        (ctx, rt, node) -> begin
            state = structured_resistive_task_state[]
            state[:split_source_active] = false
            StructuredTaskDone
        end
    structured_resistive_task_callbacks[:resistive_stage] =
        (ctx, rt, node) -> begin
            match_result = match(
                r"^resistive_stage_([0-9]+)$", String(node.id),
            )
            match_result === nothing && throw(ArgumentError(
                "resistive task has no stage in id $(node.id)",
            ))
            _structured_resistive_task_stage!(
                structured_resistive_task_state[],
                parse(Int, match_result.captures[1]),
            )
            StructuredTaskDone
        end
    structured_resistive_task_callbacks[:resistive_reconcile] =
        (ctx, rt, node) -> begin
            state = structured_resistive_task_state[]
            state[:phase] == :post || return StructuredTaskDone
            split_source_active = apply_structured_split_sources!(
                blocks, FT(0.5) * state[:current_dt], nthreads,
            )
            state[:split_source_active] = split_source_active
            if split_source_active
                sync_blocks!(
                    state[:active_time] + state[:current_dt];
                    step=state[:tt], rk_stage=4,
                )
            end
            StructuredTaskDone
        end
    structured_resistive_task_callbacks[:resistive_positivity] =
        (ctx, rt, node) -> begin
            # The CT substep routines perform their site-specific positivity
            # checks immediately after face-B/Q reconciliation.  This node is
            # the outer graph's final state barrier and remains side-effect
            # free for configurations without resistive CT.
            StructuredTaskDone
        end

    function initialize_structured_resistive_task_context!(stage_count::Integer)
        if structured_resistive_task_context !== nothing &&
           get(structured_resistive_task_context.runtime_options,
               :stage_count, 0) == stage_count
            return structured_resistive_task_context
        end
        resistive_graph = build_structured_resistive_task_graph(
            stage_count=stage_count,
            callbacks=structured_resistive_task_callbacks,
        )
        structured_resistive_task_context = StructuredTaskContext(
            resistive_graph;
            blocks=blocks,
            connectivity=connectivity,
            mpi_requests=block_comms,
            shared_scratch=(shared_Fx, shared_Fy, shared_Fz,
                            shared_Fvx, shared_Fvy, shared_Fvz),
            runtime_options=(equation_type=equation_type,
                             ct_mode=structured_task_ct_active,
                             stage_count=Int(stage_count)),
        )
        graph_min = MPI.Allreduce(
            resistive_graph.signature, MPI.MIN, MPI.COMM_WORLD,
        )
        graph_max = MPI.Allreduce(
            resistive_graph.signature, MPI.MAX, MPI.COMM_WORLD,
        )
        graph_min == graph_max || error(
            "structured resistive graph signature differs across MPI ranks: " *
            "min=$graph_min max=$graph_max",
        )
        world_rank == 0 && println(
            ">>> Structured resistive graph signature=$(resistive_graph.signature)",
        )
        return structured_resistive_task_context
    end

    function run_structured_resistive_taskgraph_step!(
        tt_val::Integer, active_time_val, current_dt_val;
        phase::Symbol=:post,
        split_dt=nothing,
        split_start_time=nothing,
        split_half::Integer=2,
    )
        phase in (:pre, :post) || throw(ArgumentError(
            "resistive taskgraph phase must be :pre or :post",
        ))
        resistive_step_dt = split_dt === nothing ?
            (ct_resistive_rkl2_active ?
             FT(0.5) * FT(current_dt_val) : FT(current_dt_val)) : FT(split_dt)
        resistive_start_time = split_start_time === nothing ?
            (ct_resistive_rkl2_active ?
             FT(active_time_val) + resistive_step_dt : FT(active_time_val)) :
            FT(split_start_time)
        schedule = _structured_resistive_task_schedule(
            resistive_step_dt, resistive_start_time, Int(tt_val),
            split_half=split_half,
        )
        initialize_structured_resistive_task_context!(
            schedule[:stage_count],
        )
        state = Dict{Symbol,Any}(
            :tt => Int(tt_val),
            :active_time => FT(active_time_val),
            :current_dt => FT(current_dt_val),
            :phase => phase,
            :split_source_active => false,
        )
        merge!(state, schedule)
        structured_resistive_task_state[] = state
        begin_structured_task_stage!(
            structured_resistive_task_context,
            StructuredTaskEpoch(Int(tt_val), 4, 0);
            seed_resources=[:RK_COMMITTED_STATE],
        )
        try
            run_structured_task_graph!(structured_resistive_task_context.runtime)
        finally
            structured_resistive_task_state[] = nothing
        end
        return state
    end

    # Provider body for the implicit task graph.  The LU-SGS/BDF2/GMRES
    # kernels remain the existing numerical implementation; only the outer
    # transaction is moved behind explicit task resources.
    function _structured_implicit_provider_step!(
        step::Integer, active_time_val, current_dt_val,
    )
        bdf2_active = dual_time &&
            step > (isdefined(@__MODULE__, :dual_time_start_step) ?
                    dual_time_start_step : 1)
        if bdf2_active
            for (_, b) in blocks
                copyto!(b.Un, b.U)
                bdf2_prepare_step!(b, current_dt_val)
            end

            prev_max_res = Inf
            min_max_res = Inf
            for m_iter in 1:dual_time_sub_iters
                sync_blocks!(
                    active_time_val;
                    step=step, rk_stage=10 + m_iter,
                )
                local_max_res = 0.0
                for (_, b) in blocks
                    res = bdf2_inner_iteration!(
                        b, current_dt_val,
                        shared_Fx, shared_Fy, shared_Fz,
                        shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                        shared_Fvx, shared_Fvy, shared_Fvz,
                        shared_dU_forced, shared_ct_pos_meta,
                        shared_ct_pos_values, world_rank, step,
                        threads_recon_i, threads_recon_j, threads_recon_k,
                        threads_visc_i, threads_visc_j, threads_visc_k,
                        threads_light, forcex, flowx, cmf_f1_val,
                        hit_u_mean, hit_v_mean, hit_w_mean, active_time_val;
                        sync_ghost_fn = () -> sync_blocks!(
                            active_time_val;
                            step=step, rk_stage=10 + m_iter,
                        ),
                    )
                    local_max_res = max(local_max_res, res)
                    @check_nan(
                        b.U, "U after taskgraph BDF2 inner iter",
                        b.id, world_rank, step,
                    )
                end

                global_max_res = MPI.Allreduce(
                    local_max_res, MPI.MAX, MPI.COMM_WORLD,
                )
                if world_rank == 0 && (step % 100 == 0 || step <= 20)
                    @printf(
                        "    taskgraph BDF2 sub-iter %d/%d: max_res = %.4e\n",
                        m_iter, dual_time_sub_iters, global_max_res,
                    )
                end
                global_max_res < dual_time_tol && break

                is_gmres = isdefined(@__MODULE__, :implicit_solver) &&
                           implicit_solver == :gmres
                if is_gmres
                    if m_iter > 1 && global_max_res > min_max_res * 10.0
                        break
                    end
                    min_max_res = min(min_max_res, global_max_res)
                elseif m_iter > 1 && global_max_res > prev_max_res * 2.0
                    break
                end
                prev_max_res = global_max_res
            end
            for (_, b) in blocks
                copyto!(b.U_nm1, 1, b.Un, 1, length(b.U_nm1))
            end
        else
            if dual_time
                for (_, b) in blocks
                    copyto!(b.U_nm1, b.U)
                    copyto!(b.Un, b.U)
                end
            end
            for (_, b) in blocks
                implicit_step!(
                    b, current_dt_val,
                    shared_Fx, shared_Fy, shared_Fz,
                    shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                    shared_Fvx, shared_Fvy, shared_Fvz,
                    shared_dU_forced, shared_ct_pos_meta,
                    shared_ct_pos_values, world_rank, step,
                    threads_recon_i, threads_recon_j, threads_recon_k,
                    threads_visc_i, threads_visc_j, threads_visc_k,
                    threads_light, forcex, flowx, cmf_f1_val,
                    hit_u_mean, hit_v_mean, hit_w_mean, active_time_val,
                )
                @check_nan(
                    b.U, "U after taskgraph implicit_step",
                    b.id, world_rank, step,
                )
            end
        end
        return nothing
    end

    function _structured_implicit_iteration_from_node(node)
        match_result = match(
            r"^implicit_(?:residual|update|halo|convergence)_([0-9]+)$",
            String(node.id),
        )
        match_result === nothing && throw(ArgumentError(
            "implicit task has no iteration in id $(node.id)",
        ))
        return parse(Int, match_result.captures[1])
    end

    function _structured_implicit_task_precompute_dt!(
        state::Dict{Symbol,Any},
    )
        if adaptive_dt || !isdefined(@__MODULE__, :dt)
            dt_min = FT(1e10)
            for (_, b) in blocks
                nb = (
                    cld(b.Nx + 2*NG, nthreads[1]),
                    cld(b.Ny + 2*NG, nthreads[2]),
                    cld(b.Nz + 2*NG, nthreads[3]),
                )
                @gpu_launch threads=nthreads blocks=nb compute_dt(
                    b.LTS_dt, b.Q, b.Vol, b.Areai, b.Areaj, b.Areak,
                    b.nxi, b.nyi, b.nzi, b.nxj, b.nyj, b.nzj,
                    b.nxk, b.nyk, b.nzk, b.Nx, b.Ny, b.Nz,
                )
                dt_local = FT(mapreduce(
                    value -> value > zero(FT) ? value : FT(Inf),
                    min, b.LTS_dt,
                ))
                dt_min = min(dt_min, dt_local)
            end
            local_implicit_CFL = implicit_CFL
            implicit_start = state[:implicit_start]
            if isdefined(@__MODULE__, :implicit_CFL_max) &&
               isdefined(@__MODULE__, :implicit_CFL_ramp_steps)
                step_offset = max(0, state[:tt] - implicit_start)
                ramp_factor = min(
                    one(FT), FT(step_offset) / FT(implicit_CFL_ramp_steps),
                )
                local_implicit_CFL = implicit_CFL + ramp_factor * (
                    implicit_CFL_max - implicit_CFL,
                )
            end
            state[:current_dt] = MPI.Allreduce(
                dt_min, MPI.MIN, MPI.COMM_WORLD,
            ) * (local_implicit_CFL / CFL)
        else
            state[:current_dt] = FT(dt)
        end
        isfinite(state[:current_dt]) || error(
            "implicit task graph produced a non-finite dt at step $(state[:tt])",
        )
        return nothing
    end

    function _structured_implicit_task_precompute_forcing!(
        state::Dict{Symbol,Any},
    )
        state[:forcex] = forcex
        state[:flowx] = flowx
        state[:cmf_f1_val] = cmf_f1_val
        state[:deschamps_f1_val] = deschamps_f1_val
        state[:deschamps_flowx_val] = deschamps_flowx_val
        if flow_forcing
            if forcing_mode == 1
                state[:forcex], state[:flowx] = Update_bulk_force_params(
                    blocks, state[:tt], 1, MPI.COMM_WORLD, world_rank,
                )
            elseif forcing_mode == 2
                state[:cmf_f1_val] = Update_const_massflux_params(
                    blocks, state[:tt], 1, state[:current_dt],
                    MPI.COMM_WORLD, world_rank,
                )
            elseif forcing_mode == 3 ||
                   (isdefined(Main, :cebl_forcing) && Main.cebl_forcing)
                forcing_blocks = (
                    isdefined(Main, :cebl_forcing) && Main.cebl_forcing
                ) ? Dict{Int,Block}(
                    id => b for (id, b) in blocks if id >= 5
                ) : blocks
                state[:deschamps_f1_val], state[:deschamps_flowx_val] =
                    Update_deschamps_pipe_params(
                        forcing_blocks, state[:tt], 1, state[:current_dt],
                        MPI.COMM_WORLD, world_rank,
                    )
            end
        end
        if test_case == "HIT" || test_case == "MHDHIT"
            hit_values = Update_HIT_forcing(
                blocks, hit_forcing_A, MPI.COMM_WORLD, world_rank,
                state[:tt], 1,
            )
            state[:hit_u_mean] = hit_values[1]
            state[:hit_v_mean] = hit_values[2]
            state[:hit_w_mean] = hit_values[3]
            state[:hit_Bx_mean] = hit_values[4]
            state[:hit_By_mean] = hit_values[5]
            state[:hit_Bz_mean] = hit_values[6]
        end
        return nothing
    end

    function _structured_implicit_task_precompute_shock!(
        state::Dict{Symbol,Any},
    )
        for (_, b) in blocks
            nb_l = (
                Int32(cld(b.Nx + 2*NG, threads_light[1])),
                Int32(cld(b.Ny + 2*NG, threads_light[2])),
                Int32(cld(b.Nz + 2*NG, threads_light[3])),
            )
            @gpu_launch threads=threads_light blocks=nb_l shockSensor(
                b.ϕ, b.Q, b.Nx, b.Ny, b.Nz,
            )
        end
        for (_, b) in blocks
            ϕ_4d = reshape(
                b.ϕ, size(b.ϕ, 1), size(b.ϕ, 2), size(b.ϕ, 3), 1,
            )
            exchange_ghost(
                ϕ_4d, 1, block_comms[b.id], b.Nx, b.Ny, b.Nz,
                b.sbuf_hx, b.sbuf_dx, b.rbuf_hx, b.rbuf_dx,
                b.sbuf_hy, b.sbuf_dy, b.rbuf_hy, b.rbuf_dy,
                b.sbuf_hz, b.sbuf_dz, b.rbuf_hz, b.rbuf_dz;
                sbuf_hx2=b.sbuf_hx2, sbuf_dx2=b.sbuf_dx2,
                rbuf_hx2=b.rbuf_hx2, rbuf_dx2=b.rbuf_dx2,
                periodic_faces=_structured_periodic_face_mask(b.id),
                rank_coords=(b.rx, b.ry, b.rz),
                rank_dims=Tuple(Block_Nprocs[b.id + 1]),
            )
        end
        return nothing
    end

    function _structured_implicit_task_prepare!(state::Dict{Symbol,Any})
        step = state[:tt]
        state[:bdf2_active] = dual_time &&
            step > (isdefined(@__MODULE__, :dual_time_start_step) ?
                    dual_time_start_step : 1)
        state[:stop] = false
        state[:last_update] = false
        state[:residual] = FT(Inf)
        state[:previous_residual] = FT(Inf)
        state[:best_residual] = FT(Inf)
        if state[:bdf2_active]
            for (_, b) in blocks
                copyto!(b.Un, b.U)
                bdf2_prepare_step!(b, state[:current_dt])
            end
        elseif dual_time
            for (_, b) in blocks
                copyto!(b.U_nm1, b.U)
                copyto!(b.Un, b.U)
            end
        end
        # The first implicit residual consumes the same halo state as every
        # later residual consumes after implicit_halo.  Making this seed
        # exchange explicit prevents the first BDF2 iteration from observing
        # a stale Q ghost layer.
        sync_blocks!(
            state[:active_time]; step=step, rk_stage=10,
        )
        state[:prepared] = true
        return nothing
    end

    function _structured_implicit_task_update!(
        state::Dict{Symbol,Any}, iteration::Integer,
    )
        state[:last_update] = false
        state[:inner_iteration] = iteration
        state[:stop] && return nothing
        step = state[:tt]
        forcex_use = get(state, :forcex, forcex)
        flowx_use = get(state, :flowx, flowx)
        cmf_f1_use = get(state, :cmf_f1_val, cmf_f1_val)
        hit_u_use = get(state, :hit_u_mean, hit_u_mean)
        hit_v_use = get(state, :hit_v_mean, hit_v_mean)
        hit_w_use = get(state, :hit_w_mean, hit_w_mean)
        if state[:bdf2_active]
            local_max_res = zero(FT)
            for (_, b) in blocks
                res = bdf2_inner_iteration!(
                    b, state[:current_dt],
                    shared_Fx, shared_Fy, shared_Fz,
                    shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                    shared_Fvx, shared_Fvy, shared_Fvz,
                    shared_dU_forced, shared_ct_pos_meta,
                    shared_ct_pos_values, world_rank, step,
                    threads_recon_i, threads_recon_j, threads_recon_k,
                    threads_visc_i, threads_visc_j, threads_visc_k,
                    threads_light, forcex_use, flowx_use, cmf_f1_use,
                    hit_u_use, hit_v_use, hit_w_use,
                    state[:active_time];
                    sync_ghost_fn = () -> sync_blocks!(
                        state[:active_time];
                        step=step, rk_stage=10 + iteration,
                    ),
                )
                local_max_res = max(local_max_res, FT(res))
                @check_nan(
                    b.U, "U after taskgraph BDF2 inner iter",
                    b.id, world_rank, step,
                )
            end
            global_max_res = MPI.Allreduce(
                local_max_res, MPI.MAX, MPI.COMM_WORLD,
            )
            state[:residual] = FT(global_max_res)
            if world_rank == 0 && (step % 100 == 0 || step <= 20)
                @printf(
                    "    taskgraph BDF2 sub-iter %d/%d: max_res = %.4e\n",
                    iteration, state[:inner_iterations], global_max_res,
                )
            end
        else
            for (_, b) in blocks
                implicit_step!(
                    b, state[:current_dt],
                    shared_Fx, shared_Fy, shared_Fz,
                    shared_rho_sum_x, shared_rho_sum_y, shared_rho_sum_z,
                    shared_Fvx, shared_Fvy, shared_Fvz,
                    shared_dU_forced, shared_ct_pos_meta,
                    shared_ct_pos_values, world_rank, step,
                    threads_recon_i, threads_recon_j, threads_recon_k,
                    threads_visc_i, threads_visc_j, threads_visc_k,
                    threads_light, forcex_use, flowx_use, cmf_f1_use,
                    hit_u_use, hit_v_use, hit_w_use,
                    state[:active_time],
                )
                @check_nan(
                    b.U, "U after taskgraph implicit_step",
                    b.id, world_rank, step,
                )
            end
            state[:residual] = zero(FT)
            state[:stop] = true
        end
        state[:last_update] = true
        return nothing
    end

    function _structured_implicit_task_convergence!(
        state::Dict{Symbol,Any}, iteration::Integer,
    )
        state[:last_update] || return nothing
        residual = state[:residual]
        is_gmres = isdefined(@__MODULE__, :implicit_solver) &&
                   implicit_solver == :gmres
        if state[:bdf2_active]
            converged = residual < dual_time_tol
            diverged = if iteration <= 1
                false
            elseif is_gmres
                residual > state[:best_residual] * FT(10)
            else
                residual > state[:previous_residual] * FT(2)
            end
            state[:stop] = converged || diverged
            state[:best_residual] = min(state[:best_residual], residual)
            state[:previous_residual] = residual
        end
        state[:converged] = state[:stop]
        return nothing
    end

    structured_implicit_task_callbacks = Dict{Symbol,Function}()
    structured_implicit_task_callbacks[:implicit_dt] =
        (ctx, rt, node) -> begin
            _structured_implicit_task_precompute_dt!(
                structured_implicit_task_state[],
            )
            StructuredTaskDone
        end
    structured_implicit_task_callbacks[:implicit_forcing] =
        (ctx, rt, node) -> begin
            _structured_implicit_task_precompute_forcing!(
                structured_implicit_task_state[],
            )
            StructuredTaskDone
        end
    structured_implicit_task_callbacks[:implicit_shock] =
        (ctx, rt, node) -> begin
            _structured_implicit_task_precompute_shock!(
                structured_implicit_task_state[],
            )
            StructuredTaskDone
        end
    structured_implicit_task_callbacks[:implicit_prepare] =
        (ctx, rt, node) -> begin
            _structured_implicit_task_prepare!(
                structured_implicit_task_state[],
            )
            StructuredTaskDone
        end
    structured_implicit_task_callbacks[:implicit_residual] =
        (ctx, rt, node) -> StructuredTaskDone
    structured_implicit_task_callbacks[:implicit_update] =
        (ctx, rt, node) -> begin
            state = structured_implicit_task_state[]
            _structured_implicit_task_update!(
                state, _structured_implicit_iteration_from_node(node),
            )
            StructuredTaskDone
        end
    structured_implicit_task_callbacks[:implicit_halo] =
        (ctx, rt, node) -> begin
            state = structured_implicit_task_state[]
            state[:last_update] && sync_blocks!(
                state[:active_time];
                step=state[:tt],
                rk_stage=10 + _structured_implicit_iteration_from_node(node),
            )
            StructuredTaskDone
        end
    structured_implicit_task_callbacks[:implicit_convergence] =
        (ctx, rt, node) -> begin
            state = structured_implicit_task_state[]
            _structured_implicit_task_convergence!(
                state, _structured_implicit_iteration_from_node(node),
            )
            StructuredTaskDone
        end
    structured_implicit_task_callbacks[:implicit_commit] =
        (ctx, rt, node) -> begin
            state = structured_implicit_task_state[]
            if state[:bdf2_active]
                for (_, b) in blocks
                    copyto!(b.U_nm1, 1, b.Un, 1, length(b.U_nm1))
                end
            end
            for (_, b) in blocks
                @check_nan(
                    b.Q, "Q after taskgraph implicit commit",
                    b.id, world_rank, state[:tt],
                )
            end
            StructuredTaskDone
        end

    function initialize_structured_implicit_task_context!(inner_iterations::Integer)
        if structured_implicit_task_context !== nothing &&
           get(structured_implicit_task_context.runtime_options,
               :inner_iterations, 0) == inner_iterations
            return structured_implicit_task_context
        end
        implicit_graph = build_structured_implicit_task_graph(
            inner_iterations=inner_iterations,
            callbacks=structured_implicit_task_callbacks,
        )
        structured_implicit_task_context = StructuredTaskContext(
            implicit_graph;
            blocks=blocks,
            connectivity=connectivity,
            mpi_requests=block_comms,
            shared_scratch=(shared_Fx, shared_Fy, shared_Fz,
                            shared_Fvx, shared_Fvy, shared_Fvz,
                            shared_dU_forced),
            runtime_options=(equation_type=equation_type,
                             ct_mode=structured_task_ct_active,
                             inner_iterations=Int(inner_iterations)),
        )
        graph_min = MPI.Allreduce(
            implicit_graph.signature, MPI.MIN, MPI.COMM_WORLD,
        )
        graph_max = MPI.Allreduce(
            implicit_graph.signature, MPI.MAX, MPI.COMM_WORLD,
        )
        graph_min == graph_max || error(
            "structured implicit graph signature differs across MPI ranks: " *
            "min=$graph_min max=$graph_max",
        )
        world_rank == 0 && println(
            ">>> Structured implicit graph signature=$(implicit_graph.signature)",
        )
        return structured_implicit_task_context
    end

    function run_structured_implicit_taskgraph_step!(
        tt_val::Integer, active_time_val, current_dt_val,
    )
        bdf2_active = dual_time &&
            Int(tt_val) > (isdefined(@__MODULE__, :dual_time_start_step) ?
                           dual_time_start_step : 1)
        inner_iterations = bdf2_active ? max(1, dual_time_sub_iters) : 1
        initialize_structured_implicit_task_context!(inner_iterations)
        state = Dict{Symbol,Any}(
            :tt => Int(tt_val),
            :active_time => FT(active_time_val),
            :current_dt => FT(current_dt_val),
            :prepared => false,
            :converged => false,
            :inner_iterations => inner_iterations,
            :implicit_start => isdefined(@__MODULE__, :implicit_start_step) ?
                Int(implicit_start_step) : 0,
            :forcex => forcex,
            :flowx => flowx,
            :cmf_f1_val => cmf_f1_val,
            :deschamps_f1_val => deschamps_f1_val,
            :deschamps_flowx_val => deschamps_flowx_val,
            :hit_u_mean => hit_u_mean,
            :hit_v_mean => hit_v_mean,
            :hit_w_mean => hit_w_mean,
            :hit_Bx_mean => zero(FT),
            :hit_By_mean => zero(FT),
            :hit_Bz_mean => zero(FT),
        )
        structured_implicit_task_state[] = state
        begin_structured_task_stage!(
            structured_implicit_task_context,
            StructuredTaskEpoch(Int(tt_val), 10, 0);
            seed_resources=[:IMPLICIT_STATE],
        )
        try
            run_structured_task_graph!(structured_implicit_task_context.runtime)
        finally
            structured_implicit_task_state[] = nothing
        end
        return state
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
          implicit_state = run_structured_implicit_taskgraph_step!(
              tt, activeTime, current_dt,
          )
          current_dt = implicit_state[:current_dt]
          forcex = implicit_state[:forcex]
          flowx = implicit_state[:flowx]
          cmf_f1_val = implicit_state[:cmf_f1_val]
          deschamps_f1_val = implicit_state[:deschamps_f1_val]
          deschamps_flowx_val = implicit_state[:deschamps_flowx_val]
          hit_u_mean = implicit_state[:hit_u_mean]
          hit_v_mean = implicit_state[:hit_v_mean]
          hit_w_mean = implicit_state[:hit_w_mean]
      else

        explicit_state = run_structured_explicit_taskgraph_step!(
            tt, activeTime,
        )
        current_dt = explicit_state[:current_dt]
        if ct_resistive_sts_active || ct_resistive_rkl2_active
            run_structured_resistive_taskgraph_step!(
                tt, activeTime, current_dt,
            )
        end

      end  # if implicit/else

        # ══════════════════════════════════════════════════════════════
        run_structured_post_taskgraph_step!(tt, activeTime, current_dt)

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
        # A time-limit exit can occur between cadence points.  Use the isolated
        # output graph so the committed state is not filtered or sampled twice.
        if tt > 0 && tt != maxStep && tt % step_plt != 0
            run_structured_final_plot_taskgraph!(tt, activeTime)
        end
    end
    if ct_energy_budget_log !== nothing
        close(ct_energy_budget_log)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    return blocks, activeTime, tt
end
