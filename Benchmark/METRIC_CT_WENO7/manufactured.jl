using LinearAlgebra
using Printf
using StaticArrays

const _METRIC_CT_WENO7_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
if !isdefined(@__MODULE__, :weno_z)
    const weno_z = true
end
if !isdefined(@__MODULE__, :weno7_gauss4_integral)
    include(joinpath(_METRIC_CT_WENO7_ROOT, "weno7.jl"))
end
if !isdefined(@__MODULE__, :ct_cmd6_edge_vector)
    include(joinpath(_METRIC_CT_WENO7_ROOT, "ct_weno7.jl"))
end

const METRIC_CT_WENO7_PERIOD = 2pi
const METRIC_CT_WENO7_WARP = 0.12

function metric_ct_weno7_gauss_rule(order)
    off_diagonal = [index/sqrt(4index^2-1) for index in 1:order-1]
    decomposition = eigen(SymTridiagonal(zeros(order), off_diagonal))
    nodes = decomposition.values
    weights = 2 .* decomposition.vectors[1,:].^2
    return nodes, weights
end

const METRIC_CT_WENO7_EXACT_NODES,
      METRIC_CT_WENO7_EXACT_WEIGHTS = metric_ct_weno7_gauss_rule(16)

@inline function metric_ct_weno7_mapping(logical::SVector{3,T}) where {T}
    xi, eta, zeta = logical
    warp = T(METRIC_CT_WENO7_WARP)
    return SVector{3,T}(
        xi + warp*sin(xi)*sin(eta),
        eta + warp*sin(eta)*sin(zeta),
        zeta + warp*sin(zeta)*sin(xi),
    )
end

@inline function metric_ct_weno7_covariants(
    logical::SVector{3,T},
) where {T}
    xi, eta, zeta = logical
    warp = T(METRIC_CT_WENO7_WARP)
    return (
        SVector{3,T}(
            one(T) + warp*cos(xi)*sin(eta),
            zero(T),
            warp*sin(zeta)*cos(xi),
        ),
        SVector{3,T}(
            warp*sin(xi)*cos(eta),
            one(T) + warp*cos(eta)*sin(zeta),
            zero(T),
        ),
        SVector{3,T}(
            zero(T),
            warp*sin(eta)*cos(zeta),
            one(T) + warp*cos(zeta)*sin(xi),
        ),
    )
end

@inline function metric_ct_weno7_face_normal(logical, direction)
    covariant = metric_ct_weno7_covariants(logical)
    tangent = Tuple(index for index in 1:3 if index != direction)
    oriented = direction == 2 ? cross(covariant[3], covariant[1]) :
               cross(covariant[tangent[1]], covariant[tangent[2]])
    return normalize(oriented)
end

function metric_ct_weno7_face_point_caches(n)
    n > 0 || throw(ArgumentError("resolution must be positive"))
    h = METRIC_CT_WENO7_PERIOD/n
    return ntuple(3) do normal_direction
        cache = Array{SVector{3,Float64},3}(undef, n, n, n)
        for k in 0:n-1, j in 0:n-1, i in 0:n-1
            index = (i, j, k)
            logical = SVector{3,Float64}(ntuple(3) do direction
                offset = direction == normal_direction ? 0.0 : 0.5
                h*(index[direction]+offset)
            end)
            point = metric_ct_weno7_mapping(logical)
            normal = metric_ct_weno7_face_normal(logical, normal_direction)
            electric = ct_weno7_manufactured_electric(point)
            magnetic_flux = cross(normal, electric)
            cache[i+1,j+1,k+1] =
                ct_face_tangential_electric(magnetic_flux, normal)
        end
        cache
    end
end

@inline function metric_ct_weno7_cache_component(
    cache, component, logical::NTuple{3,Int}, n,
)
    index = ntuple(direction -> mod(logical[direction], n)+1, 3)
    return cache[index...][component]
end

@inline function metric_ct_weno7_transverse_stencil(
    cache, component, base::NTuple{3,Int}, transverse_direction,
    first_offset, n,
)
    return SVector{7,Float64}(ntuple(7) do sample
        logical = Base.setindex(
            base,
            base[transverse_direction]+first_offset+sample-1,
            transverse_direction,
        )
        metric_ct_weno7_cache_component(cache, component, logical, n)
    end)
end

@inline function metric_ct_weno7_edge_electric(
    caches, edge_direction, base::NTuple{3,Int}, n,
)
    transverse = Tuple(direction for direction in 1:3
                       if direction != edge_direction)
    b_direction, c_direction = transverse
    weights = SVector{7,Float64}(ntuple(_ -> 0.5, 7))
    return SVector{3,Float64}(ntuple(3) do component
        b_left = metric_ct_weno7_transverse_stencil(
            caches[b_direction], component, base, c_direction, -4, n,
        )
        b_right = metric_ct_weno7_transverse_stencil(
            caches[b_direction], component, base, c_direction, -3, n,
        )
        c_left = metric_ct_weno7_transverse_stencil(
            caches[c_direction], component, base, b_direction, -4, n,
        )
        c_right = metric_ct_weno7_transverse_stencil(
            caches[c_direction], component, base, b_direction, -3, n,
        )
        ct_weno7_midpoint_integrand(
            b_left, b_right, c_left, c_right,
            weights, weights, 1.0, 1.0,
        )
    end)
end

@inline function metric_ct_weno7_edge_node(
    edge_direction, base::NTuple{3,Int}, offset, n,
)
    h = METRIC_CT_WENO7_PERIOD/n
    logical = SVector{3,Float64}(ntuple(3) do direction
        index = direction == edge_direction ?
                base[direction]+offset : base[direction]
        h*index
    end)
    return metric_ct_weno7_mapping(logical)
end

@inline function metric_ct_weno7_midpoint_integrand(
    caches, edge_direction, base::NTuple{3,Int}, n,
)
    edge_electric = metric_ct_weno7_edge_electric(
        caches, edge_direction, base, n,
    )
    nodes = ntuple(
        index -> metric_ct_weno7_edge_node(
            edge_direction, base, index-3, n,
        ),
        6,
    )
    edge_vector = ct_cmd6_edge_vector(nodes...)
    return dot(edge_electric, edge_vector)
end

function metric_ct_weno7_midpoint_cache(caches, edge_direction, n)
    midpoint = Array{Float64,3}(undef, n, n, n)
    for k in 0:n-1, j in 0:n-1, i in 0:n-1
        base = (i, j, k)
        midpoint[i+1,j+1,k+1] = metric_ct_weno7_midpoint_integrand(
            caches, edge_direction, base, n,
        )
    end
    return midpoint
end

@inline function metric_ct_weno7_midpoint_sample(
    midpoint, logical::NTuple{3,Int}, n,
)
    index = ntuple(direction -> mod(logical[direction], n)+1, 3)
    return midpoint[index...]
end

@inline function metric_ct_weno7_exact_edge_integral(
    edge_direction, base::NTuple{3,Int}, n,
)
    h = METRIC_CT_WENO7_PERIOD/n
    integral = 0.0
    for quadrature_index in eachindex(METRIC_CT_WENO7_EXACT_NODES)
        node = METRIC_CT_WENO7_EXACT_NODES[quadrature_index]
        parameter = (node+1)/2
        logical = SVector{3,Float64}(ntuple(3) do direction
            index = direction == edge_direction ?
                    base[direction]+parameter : base[direction]
            h*index
        end)
        point = metric_ct_weno7_mapping(logical)
        edge_tangent = h*metric_ct_weno7_covariants(logical)[edge_direction]
        integral += METRIC_CT_WENO7_EXACT_WEIGHTS[quadrature_index] *
                    dot(ct_weno7_manufactured_electric(point), edge_tangent)/2
    end
    return integral
end

function metric_ct_weno7_directional_errors(caches, edge_direction, n)
    midpoint = metric_ct_weno7_midpoint_cache(caches, edge_direction, n)
    error_sum = 0.0
    error_squared_sum = 0.0
    error_max = 0.0
    nonfinite = false
    count = 0
    for k in 0:n-1, j in 0:n-1, i in 0:n-1
        base = (i, j, k)
        segment = base[edge_direction]
        midpoint_values = SVector{7,Float64}(ntuple(7) do sample
            sample_base = Base.setindex(
                base, segment+sample-4, edge_direction,
            )
            metric_ct_weno7_midpoint_sample(midpoint, sample_base, n)
        end)
        numerical = weno7_gauss4_integral(midpoint_values)
        exact = metric_ct_weno7_exact_edge_integral(edge_direction, base, n)
        error = abs(numerical-exact)
        nonfinite |= !(isfinite(numerical) && isfinite(exact) && isfinite(error))
        error_sum += error
        error_squared_sum += error*error
        error_max = max(error_max, error)
        count += 1
    end
    return (
        l1=error_sum/count,
        l2=sqrt(error_squared_sum/count),
        linf=error_max,
        nonfinite=nonfinite,
    )
end

function metric_ct_weno7_run(resolution)
    caches = metric_ct_weno7_face_point_caches(resolution)
    cache_identities = objectid.(caches)
    directional = ntuple(3) do direction
        result = metric_ct_weno7_directional_errors(
            caches, direction, resolution,
        )
        objectid.(caches) == cache_identities || error(
            "face point caches changed during direction $direction",
        )
        result
    end
    l1_edge = sum(result.l1 for result in directional)/3
    l2_edge = sqrt(sum(result.l2^2 for result in directional)/3)
    linf_edge = maximum(result.linf for result in directional)
    constant_values = SVector{7,Float64}(ntuple(_ -> 2.75, 7))
    constant_error = abs(weno7_gauss4_integral(constant_values)-2.75)
    nonfinite_flag = any(result.nonfinite for result in directional) ||
                     !all(isfinite, (l1_edge, l2_edge, linf_edge,
                                     constant_error))
    return (
        resolution=resolution,
        l1_edge=l1_edge,
        l2_edge=l2_edge,
        linf_edge=linf_edge,
        constant_error=constant_error,
        nonfinite_flag=Int(nonfinite_flag),
        directional=directional,
    )
end

function metric_ct_weno7_write_stats(path, result)
    mkpath(dirname(path))
    open(path, "w") do io
        println(
            io,
            "resolution l1_edge l2_edge linf_edge constant_error nonfinite_flag",
        )
        @printf(
            io,
            "%d %.16e %.16e %.16e %.16e %d\n",
            result.resolution,
            result.l1_edge,
            result.l2_edge,
            result.linf_edge,
            result.constant_error,
            result.nonfinite_flag,
        )
    end
    return path
end

function _metric_ct_weno7_manufactured_main(args)
    length(args) == 2 || error(
        "usage: julia manufactured.jl <resolution> <output_directory>",
    )
    resolution = parse(Int, args[1])
    resolution > 0 || error("resolution must be positive")
    result = metric_ct_weno7_run(resolution)
    stats_path = metric_ct_weno7_write_stats(
        joinpath(args[2], "stats.dat"), result,
    )
    for direction in 1:3
        error = result.directional[direction]
        @printf(
            "direction=%d L1=%.16e L2=%.16e Linf=%.16e\n",
            direction, error.l1, error.l2, error.linf,
        )
    end
    @printf(
        "N=%d L1=%.16e L2=%.16e Linf=%.16e constant=%.16e flag=%d\n",
        result.resolution,
        result.l1_edge,
        result.l2_edge,
        result.linf_edge,
        result.constant_error,
        result.nonfinite_flag,
    )
    println("stats=$stats_path")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    _metric_ct_weno7_manufactured_main(ARGS)
end
