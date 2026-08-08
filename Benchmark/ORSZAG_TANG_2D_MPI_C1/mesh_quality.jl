# CPU-only geometry and topology checks for the c1 Orszag-Tang mesh.

using HDF5
using LinearAlgebra
using Printf
using StaticArrays

const L = 2pi
const EXPECTED_MAX_NONORTHOGONALITY = 45.0
const EXPECTED_MAX_INTERFACE_ERROR = 1.0e-12

@inline function point(coords, i, j, k)
    return SVector(
        Float64(coords[1, i, j, k]),
        Float64(coords[2, i, j, k]),
        Float64(coords[3, i, j, k]),
    )
end

@inline function quad_area(a, b, c, d)
    return 0.5 * (cross(b - a, c - a) + cross(c - a, d - a))
end

@inline function quad_center(a, b, c, d)
    return 0.25 * (a + b + c + d)
end

function block_quality(coords)
    nx, ny, nz = size(coords, 2) - 1, size(coords, 3) - 1, size(coords, 4) - 1
    determinants = Float64[]
    volumes = Float64[]
    angles = Float64[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        p000 = point(coords, i, j, k)
        p100 = point(coords, i + 1, j, k)
        p010 = point(coords, i, j + 1, k)
        p110 = point(coords, i + 1, j + 1, k)
        p001 = point(coords, i, j, k + 1)
        p101 = point(coords, i + 1, j, k + 1)
        p011 = point(coords, i, j + 1, k + 1)
        p111 = point(coords, i + 1, j + 1, k + 1)

        edge_i = 0.25 * ((p100 - p000) + (p110 - p010) +
                         (p101 - p001) + (p111 - p011))
        edge_j = 0.25 * ((p010 - p000) + (p110 - p100) +
                         (p011 - p001) + (p111 - p101))
        edge_k = 0.25 * ((p001 - p000) + (p101 - p100) +
                         (p011 - p010) + (p111 - p110))
        push!(determinants, dot(edge_i, cross(edge_j, edge_k)))

        si_lo = quad_area(p000, p010, p011, p001)
        si_hi = quad_area(p100, p110, p111, p101)
        sj_lo = quad_area(p000, p001, p101, p100)
        sj_hi = quad_area(p010, p011, p111, p110)
        sk_lo = quad_area(p000, p100, p110, p010)
        sk_hi = quad_area(p001, p101, p111, p011)
        ci_lo = quad_center(p000, p010, p011, p001)
        ci_hi = quad_center(p100, p110, p111, p101)
        cj_lo = quad_center(p000, p001, p101, p100)
        cj_hi = quad_center(p010, p011, p111, p110)
        ck_lo = quad_center(p000, p100, p110, p010)
        ck_hi = quad_center(p001, p101, p111, p011)
        push!(volumes, (
            dot(ci_hi, si_hi) - dot(ci_lo, si_lo) +
            dot(cj_hi, sj_hi) - dot(cj_lo, sj_lo) +
            dot(ck_hi, sk_hi) - dot(ck_lo, sk_lo)
        ) / 3.0)

        for (edge, area) in ((edge_i, si_hi), (edge_j, sj_hi), (edge_k, sk_hi))
            denominator = norm(edge) * norm(area)
            denominator > 0.0 || error("degenerate metric in cell ($i,$j,$k)")
            cosine = clamp(abs(dot(edge, area)) / denominator, 0.0, 1.0)
            push!(angles, acos(cosine) * 180 / pi)
        end
    end
    return (
        min_det=minimum(determinants),
        max_det=maximum(determinants),
        det_ratio=maximum(determinants) / minimum(determinants),
        min_volume=minimum(volumes),
        max_volume=maximum(volumes),
        volume_ratio=maximum(volumes) / minimum(volumes),
        max_nonorth=maximum(angles),
    )
end

function load_coords(mesh_dir, bid)
    path = joinpath(mesh_dir, "mesh_b$(bid).h5")
    isfile(path) || error("missing $path")
    return h5open(path, "r") do file
        Float64.(read(file["coords"]))
    end
end

function max_sheet_error(a, b, shift)
    size(a) == size(b) || error("interface sheets have different shapes")
    max_error = 0.0
    for k in axes(a, 3), i in axes(a, 2)
        pa = SVector(Float64(a[1, i, k]), Float64(a[2, i, k]), Float64(a[3, i, k]))
        pb = SVector(Float64(b[1, i, k]), Float64(b[2, i, k]), Float64(b[3, i, k])) + shift
        max_error = max(max_error, norm(pa - pb))
    end
    return max_error
end

function check_mesh(mesh_dir)
    c0 = load_coords(mesh_dir, 0)
    c1 = load_coords(mesh_dir, 1)
    q0 = block_quality(c0)
    q1 = block_quality(c1)
    internal_error = max_sheet_error(c0[:, :, end, :], c1[:, :, 1, :], SVector(0.0, 0.0, 0.0))
    periodic_error = max_sheet_error(c0[:, :, 1, :], c1[:, :, end, :], SVector(0.0, -L, 0.0))
    min_det = min(q0.min_det, q1.min_det)
    max_det = max(q0.max_det, q1.max_det)
    max_nonorth = max(q0.max_nonorth, q1.max_nonorth)
    @printf("mesh=%s\n", abspath(mesh_dir))
    @printf("min_det=%.16e max_det=%.16e det_ratio=%.8e\n", min_det, max_det, max_det / min_det)
    @printf("volume_ratio=%.8e max_nonorth_deg=%.8e\n",
            max(q0.volume_ratio, q1.volume_ratio), max_nonorth)
    @printf("internal_interface_error=%.16e periodic_interface_error=%.16e\n",
            internal_error, periodic_error)
    isfinite(min_det) && min_det > 0 || error("non-positive or non-finite Jacobian")
    max_nonorth < EXPECTED_MAX_NONORTHOGONALITY || error("mesh is too non-orthogonal")
    max(internal_error, periodic_error) <= EXPECTED_MAX_INTERFACE_ERROR ||
        error("interface coordinate mismatch exceeds tolerance")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia mesh_quality.jl <mesh_dir>")
    check_mesh(ARGS[1])
end
