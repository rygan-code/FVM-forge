using LinearAlgebra
using Printf
using StaticArrays
using CUDA

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const FT = Float64
const NG = 4
const weno_z = true
const ct_mode = true
const strict_ct_positivity = true
const splitMethodID = Int32(4)
const equation_type = :MHD
const implicit = false
const dual_time = false
const CT_EMF_WENO7_SG07 = Int32(7)
const ct_emf_scheme = CT_EMF_WENO7_SG07
const nthreads = (Int32(4), Int32(4), Int32(4))

include(joinpath(ROOT,"src","parallel","gpu_backend.jl"))
include(joinpath(ROOT,"src","core","boundary_types.jl"))
include(joinpath(ROOT,"src","numerics","weno7_reconstruction.jl"))
include(joinpath(ROOT,"src","numerics","ct_state.jl"))
include(joinpath(ROOT,"src","numerics","ct_weno7.jl"))
include(joinpath(ROOT,"src","numerics","constrained_transport.jl"))
include(joinpath(@__DIR__, "manufactured.jl"))
include(joinpath(@__DIR__, "verify.jl"))

@inline function runtime_mapping(logical, geometry)
    return geometry == :warped ? metric_ct_weno7_mapping(logical) : logical
end

@inline function runtime_face_normal(logical, direction, geometry)
    if geometry == :warped
        return metric_ct_weno7_face_normal(logical, direction)
    end
    return SVector{3,FT}(ntuple(axis -> axis == direction ? one(FT) : zero(FT), 3))
end

function runtime_coordinates(n, geometry)
    h = FT(METRIC_CT_WENO7_PERIOD)/FT(n)
    dims = (n+2NG+1, n+2NG+1, n+2NG+1)
    x = zeros(FT, dims); y = similar(x); z = similar(x)
    for k in axes(x,3), j in axes(x,2), i in axes(x,1)
        logical = SVector{3,FT}(
            h*FT(i-NG-1), h*FT(j-NG-1), h*FT(k-NG-1),
        )
        point = runtime_mapping(logical, geometry)
        x[i,j,k], y[i,j,k], z[i,j,k] = point
    end
    return x, y, z
end

function runtime_face_caches(n, geometry; constant_electric=nothing)
    h = FT(METRIC_CT_WENO7_PERIOD)/FT(n)
    dims = (
        (n+1, n+2NG, n+2NG, 4),
        (n+2NG, n+1, n+2NG, 4),
        (n+2NG, n+2NG, n+1, 4),
    )
    return ntuple(3) do normal_direction
        cache = zeros(FT, dims[normal_direction])
        for k in axes(cache,3), j in axes(cache,2), i in axes(cache,1)
            storage = (i,j,k)
            logical = SVector{3,FT}(ntuple(3) do direction
                coordinate = direction == normal_direction ?
                             storage[direction]-1 :
                             storage[direction]-NG-FT(0.5)
                h*FT(coordinate)
            end)
            point = runtime_mapping(logical, geometry)
            normal = runtime_face_normal(logical, normal_direction, geometry)
            electric = constant_electric === nothing ?
                       ct_weno7_manufactured_electric(point) :
                       constant_electric
            tangential = electric-dot(electric, normal)*normal
            cache[i,j,k,1] = tangential[1]
            cache[i,j,k,2] = tangential[2]
            cache[i,j,k,3] = tangential[3]
            cache[i,j,k,4] = FT(0.5)
        end
        cache
    end
end

function runtime_edges(n, geometry; constant_electric=nothing)
    caches = map(CuArray, runtime_face_caches(
        n, geometry; constant_electric=constant_electric,
    ))
    coordinates = map(CuArray, runtime_coordinates(n, geometry))
    scratch = CUDA.zeros(FT, n+2NG+1, n+2NG+1, n+2NG+1)
    fail_meta = CUDA.zeros(Int32, CT_WENO7_FAIL_META_LEN)
    fail_value = CUDA.zeros(FT, 1)
    edges = (
        CUDA.zeros(FT, n, n+1, n+1),
        CUDA.zeros(FT, n+1, n, n+1),
        CUDA.zeros(FT, n+1, n+1, n),
    )
    for direction in 1:3
        ct_weno7_build_edge_line!(
            edges[direction], scratch, Val(direction), caches...,
            coordinates..., fail_meta, fail_value,
            0, 0, Int32(1), n, n, n, (true, true, true),
        )
    end
    blocks = ntuple(_ -> Int32(cld(n+1, nthreads[1])), 3)
    @gpu_launch threads=nthreads blocks=blocks ct_sync_periodic_edge_emf_kernel!(
        edges..., true, true, true, Int32(n), Int32(n), Int32(n))
    gpu_sync()
    meta_host = Array(fail_meta)
    value_host = Array(fail_value)
    meta_host[CT_WENO7_FAIL_CLAIMED] == 0 || error(
        "unexpected WENO7 failure metadata: $meta_host value=$(value_host[1])",
    )
    edges_host = map(Array, edges)
    periodic_residual = maximum((
        maximum(abs, edges_host[1][:,end,:] .- edges_host[1][:,1,:]),
        maximum(abs, edges_host[1][:,:,end] .- edges_host[1][:,:,1]),
        maximum(abs, edges_host[2][end,:,:] .- edges_host[2][1,:,:]),
        maximum(abs, edges_host[2][:,:,end] .- edges_host[2][:,:,1]),
        maximum(abs, edges_host[3][end,:,:] .- edges_host[3][1,:,:]),
        maximum(abs, edges_host[3][:,end,:] .- edges_host[3][:,1,:]),
    ))
    periodic_residual <= 1.0e-12 || error(
        "periodic canonical edge residual=$periodic_residual exceeds 1e-12",
    )
    return edges_host
end

@inline function exact_edge(direction, base, n, geometry)
    if geometry == :warped
        return metric_ct_weno7_exact_edge_integral(direction, base, n)
    end
    h = FT(METRIC_CT_WENO7_PERIOD)/FT(n)
    p0 = SVector{3,FT}(h*base[1], h*base[2], h*base[3])
    p1 = Base.setindex(p0, p0[direction]+h, direction)
    return ct_weno7_exact_segment_integral(p0, p1)
end

function runtime_errors(edges, n, geometry)
    directional = ntuple(3) do direction
        l1 = 0.0; l2 = 0.0; linf = 0.0; count = 0
        for k in 0:n-1, j in 0:n-1, i in 0:n-1
            base = (i,j,k)
            error_value = abs(edges[direction][i+1,j+1,k+1] -
                              exact_edge(direction, base, n, geometry))
            isfinite(error_value) || error("nonfinite runtime edge error")
            l1 += error_value
            l2 += error_value*error_value
            linf = max(linf, error_value)
            count += 1
        end
        (l1=l1/count, l2=sqrt(l2/count), linf=linf)
    end
    return (
        l1=sum(result.l1 for result in directional)/3,
        l2=sqrt(sum(result.l2^2 for result in directional)/3),
        linf=maximum(result.linf for result in directional),
        directional=directional,
    )
end

function warped_constant_covariant_error(n)
    # On a warped edge the authoritative scalar is g = E dot dx/dxi. A
    # constant Cartesian E is not a constant g. Test the actual constant-
    # covariant-integrand invariant with the production CUDA line kernel.
    x, y, z = runtime_coordinates(n, :warped)
    h = FT(METRIC_CT_WENO7_PERIOD)/FT(n)
    warp_signal = maximum(abs, x[:,2:end,:] .-
                               reshape([h*FT(i-NG-1) for i in axes(x,1)], :,1,1))
    warp_signal > 0 || error("warped constant test received Cartesian coordinates")

    constant_integrand = FT(2.75)
    scratch = CUDA.fill(
        constant_integrand, n+2NG+1, n+2NG+1, n+2NG+1,
    )
    fail_meta = CUDA.zeros(Int32, CT_WENO7_FAIL_META_LEN)
    fail_value = CUDA.zeros(FT, 1)
    edges = (
        CUDA.zeros(FT, n, n+1, n+1),
        CUDA.zeros(FT, n+1, n, n+1),
        CUDA.zeros(FT, n+1, n+1, n),
    )
    for direction in 1:3
        extent = size(edges[direction])
        blocks = ntuple(axis -> cld(extent[axis], nthreads[axis]), 3)
        @gpu_launch threads=nthreads blocks=blocks ct_weno7_edge_line_kernel!(
            edges[direction], scratch, Int32(direction),
            fail_meta, fail_value, Int32(1),
            Int32(n), Int32(n), Int32(n))
    end
    gpu_sync()
    Array(fail_meta)[CT_WENO7_FAIL_CLAIMED] == 0 || error(
        "warped constant-covariant line kernel recorded nonfinite data",
    )
    return maximum(
        maximum(abs, Array(edge) .- constant_integrand) for edge in edges
    )
end

function repeated_rk3_stokes(edges, n; steps=8)
    device_edges = map(CuArray, edges)
    bx = CUDA.zeros(FT, n+1+2NG, n+2NG+1, n+2NG+1)
    by = CUDA.zeros(FT, n+2NG+1, n+1+2NG, n+2NG+1)
    bz = CUDA.zeros(FT, n+2NG+1, n+2NG+1, n+1+2NG)
    backups = (similar(bx), similar(by), similar(bz))
    blocks = ntuple(_ -> Int32(cld(n+1, nthreads[1])), 3)
    for _ in 1:steps
        copyto!(backups[1], bx); copyto!(backups[2], by); copyto!(backups[3], bz)
        for stage in 1:3
            rk_a = stage == 2 ? FT(0.25) : (stage == 3 ? FT(2)/FT(3) : one(FT))
            @gpu_launch threads=nthreads blocks=blocks ct_update_face_b_from_emf_kernel!(
                bx, by, bz, backups..., device_edges..., FT(1.0e-3), rk_a,
                Int32(n), Int32(n), Int32(n))
        end
    end
    gpu_sync()
    bx_host, by_host, bz_host = map(Array, (bx, by, bz))
    max_divergence = zero(FT)
    for k in 1:n, j in 1:n, i in 1:n
        ii, jj, kk = i+NG, j+NG, k+NG
        divergence = bx_host[ii+1,jj,kk]-bx_host[ii,jj,kk] +
                     by_host[ii,jj+1,kk]-by_host[ii,jj,kk] +
                     bz_host[ii,jj,kk+1]-bz_host[ii,jj,kk]
        max_divergence = max(max_divergence, abs(divergence))
    end
    max_divergence <= 1.0e-12 || error("face-divB=$max_divergence exceeds 1e-12")
    return max_divergence
end

function write_stats(path, n, result, constant_error)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "resolution l1_edge l2_edge linf_edge constant_error nonfinite_flag")
        @printf(io, "%d %.16e %.16e %.16e %.16e 0\n",
                n, result.l1, result.l2, result.linf, constant_error)
    end
end

function run_case(geometry, n, output_directory)
    CUDA.functional() || error("Task 7 runtime acceptance requires functional CUDA")
    geometry in (:cartesian, :warped) || error("unknown geometry $geometry")
    edges = runtime_edges(n, geometry)
    result = runtime_errors(edges, n, geometry)
    if geometry == :warped
        constant_error = warped_constant_covariant_error(n)
    else
        constant_electric = SVector{3,FT}(FT(0.7), FT(-0.4), FT(0.25))
        constant_edges = runtime_edges(
            n, :cartesian; constant_electric=constant_electric,
        )
        h = FT(METRIC_CT_WENO7_PERIOD)/FT(n)
        constant_error = maximum(
            maximum(abs, constant_edges[direction] .-
                         constant_electric[direction]*h) for direction in 1:3
        )
    end
    divb = repeated_rk3_stokes(edges, n)
    stats_path = joinpath(output_directory, "stats.dat")
    write_stats(stats_path, n, result, constant_error)
    @printf("geometry=%s N=%d L1=%.16e L2=%.16e Linf=%.16e divB=%.3e\n",
            String(geometry), n, result.l1, result.l2, result.linf, divb)
    return stats_path
end

function main(args)
    if length(args) == 5 && args[1] == "verify"
        geometry = Symbol(args[2])
        thresholds = geometry == :cartesian ? (6.5, 6.5, 6.3) : (5.5, 5.5, 5.0)
        summary = ct_weno7_verify_convergence(
            args[3:5]; min_l1_order=thresholds[1],
            min_l2_order=thresholds[2], min_linf_order=thresholds[3],
        )
        println("$geometry runtime WENO7 convergence passed: $summary")
        return
    end
    length(args) == 3 || error(
        "usage: run.jl <cartesian|warped> <resolution> <output_directory>",
    )
    run_case(Symbol(args[1]), parse(Int, args[2]), args[3])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
