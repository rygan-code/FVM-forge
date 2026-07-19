using HDF5
using LinearAlgebra
using StaticArrays

function metric_ct_multiblock_point(xi, eta, zeta)
    return SVector(
        xi,
        eta + 0.20sin(eta) + 0.10sin(eta/2)*sin(xi)*sin(zeta),
        zeta,
    )
end

@inline function _metric_ct_point(coords, i, j, k)
    return SVector(
        Float64(coords[1, i, j, k]),
        Float64(coords[2, i, j, k]),
        Float64(coords[3, i, j, k]),
    )
end

@inline function _metric_ct_quad_area(a, b, c, d)
    return 0.5 * (cross(b - a, c - a) + cross(c - a, d - a))
end

@inline _metric_ct_quad_center(a, b, c, d) = 0.25 * (a + b + c + d)

function _metric_ct_block_coords(block)
    hasproperty(block, :coords) || throw(ArgumentError("block must provide coords"))
    coords = getproperty(block, :coords)
    ndims(coords) == 4 && size(coords, 1) == 3 ||
        throw(DimensionMismatch("coords must have shape (3, Nx+1, Ny+1, Nz+1)"))
    return coords
end

function _metric_ct_cell_measurements(coords)
    nx, ny, nz = size(coords, 2) - 1, size(coords, 3) - 1,
                 size(coords, 4) - 1
    min(nx, ny, nz) > 0 || throw(ArgumentError("each block must contain cells"))

    determinants = Float64[]
    volumes = Float64[]
    sizehint!(determinants, nx * ny * nz)
    sizehint!(volumes, nx * ny * nz)

    for k in 1:nz, j in 1:ny, i in 1:nx
        p000 = _metric_ct_point(coords, i, j, k)
        p100 = _metric_ct_point(coords, i + 1, j, k)
        p010 = _metric_ct_point(coords, i, j + 1, k)
        p110 = _metric_ct_point(coords, i + 1, j + 1, k)
        p001 = _metric_ct_point(coords, i, j, k + 1)
        p101 = _metric_ct_point(coords, i + 1, j, k + 1)
        p011 = _metric_ct_point(coords, i, j + 1, k + 1)
        p111 = _metric_ct_point(coords, i + 1, j + 1, k + 1)

        edge_i = 0.25 * (
            (p100 - p000) + (p110 - p010) +
            (p101 - p001) + (p111 - p011)
        )
        edge_j = 0.25 * (
            (p010 - p000) + (p110 - p100) +
            (p011 - p001) + (p111 - p101)
        )
        edge_k = 0.25 * (
            (p001 - p000) + (p101 - p100) +
            (p011 - p010) + (p111 - p110)
        )
        push!(determinants, dot(edge_i, cross(edge_j, edge_k)))

        si_lo = _metric_ct_quad_area(p000, p010, p011, p001)
        si_hi = _metric_ct_quad_area(p100, p110, p111, p101)
        sj_lo = _metric_ct_quad_area(p000, p001, p101, p100)
        sj_hi = _metric_ct_quad_area(p010, p011, p111, p110)
        sk_lo = _metric_ct_quad_area(p000, p100, p110, p010)
        sk_hi = _metric_ct_quad_area(p001, p101, p111, p011)

        ci_lo = _metric_ct_quad_center(p000, p010, p011, p001)
        ci_hi = _metric_ct_quad_center(p100, p110, p111, p101)
        cj_lo = _metric_ct_quad_center(p000, p001, p101, p100)
        cj_hi = _metric_ct_quad_center(p010, p011, p111, p110)
        ck_lo = _metric_ct_quad_center(p000, p100, p110, p010)
        ck_hi = _metric_ct_quad_center(p001, p101, p111, p011)

        volume = (
            dot(ci_hi, si_hi) - dot(ci_lo, si_lo) +
            dot(cj_hi, sj_hi) - dot(cj_lo, sj_lo) +
            dot(ck_hi, sk_hi) - dot(ck_lo, sk_lo)
        ) / 3.0
        push!(volumes, volume)
    end
    return determinants, volumes
end

function _metric_ct_eta_edges(coords)
    edges = Float64[]
    sizehint!(edges, size(coords, 2) * (size(coords, 3) - 1) * size(coords, 4))
    for k in axes(coords, 4), j in 1:size(coords, 3)-1, i in axes(coords, 2)
        p0 = _metric_ct_point(coords, i, j, k)
        p1 = _metric_ct_point(coords, i, j + 1, k)
        push!(edges, norm(p1 - p0))
    end
    return edges
end

function _metric_ct_interface_nonorthogonality(coords, side)
    side in (:lower, :upper) || throw(ArgumentError("side must be :lower or :upper"))
    face_j = side === :lower ? 1 : size(coords, 3)
    inner_j = side === :lower ? 2 : face_j - 1
    angles = Float64[]
    for k in 1:size(coords, 4)-1, i in 1:size(coords, 2)-1
        p00 = _metric_ct_point(coords, i, face_j, k)
        p10 = _metric_ct_point(coords, i + 1, face_j, k)
        p01 = _metric_ct_point(coords, i, face_j, k + 1)
        p11 = _metric_ct_point(coords, i + 1, face_j, k + 1)
        q00 = _metric_ct_point(coords, i, inner_j, k)
        q10 = _metric_ct_point(coords, i + 1, inner_j, k)
        q01 = _metric_ct_point(coords, i, inner_j, k + 1)
        q11 = _metric_ct_point(coords, i + 1, inner_j, k + 1)

        normal = side === :lower ?
            _metric_ct_quad_area(p00, p10, p11, p01) :
            _metric_ct_quad_area(p00, p01, p11, p10)
        eta_edge = side === :lower ?
            0.25 * ((q00 - p00) + (q10 - p10) + (q01 - p01) + (q11 - p11)) :
            0.25 * ((p00 - q00) + (p10 - q10) + (p01 - q01) + (p11 - q11))
        denominator = norm(normal) * norm(eta_edge)
        denominator > 0.0 || error("degenerate interface geometry")
        cosine = clamp(abs(dot(normal, eta_edge)) / denominator, 0.0, 1.0)
        push!(angles, rad2deg(acos(cosine)))
    end
    return angles
end

function metric_ct_mesh_quality(block0, block1)
    coords0 = _metric_ct_block_coords(block0)
    coords1 = _metric_ct_block_coords(block1)
    (size(coords0, 2), size(coords0, 4)) ==
        (size(coords1, 2), size(coords1, 4)) ||
        throw(DimensionMismatch("paired interface sheets must have matching dimensions"))

    determinants0, volumes0 = _metric_ct_cell_measurements(coords0)
    determinants1, volumes1 = _metric_ct_cell_measurements(coords1)
    determinants = vcat(determinants0, determinants1)
    volumes = vcat(volumes0, volumes1)
    eta_edges = vcat(_metric_ct_eta_edges(coords0), _metric_ct_eta_edges(coords1))
    angles = vcat(
        _metric_ct_interface_nonorthogonality(coords0, :upper),
        _metric_ct_interface_nonorthogonality(coords1, :lower),
    )

    face0 = @view coords0[:, :, end, :]
    face1 = @view coords1[:, :, 1, :]
    node_errors = [
        norm(SVector(
            Float64(face0[1, i, k] - face1[1, i, k]),
            Float64(face0[2, i, k] - face1[2, i, k]),
            Float64(face0[3, i, k] - face1[3, i, k]),
        )) for i in axes(face0, 2), k in axes(face0, 3)
    ]
    interface_y = @view face0[2, :, :]

    min_det = minimum(determinants)
    max_det = maximum(determinants)
    min_edge = minimum(eta_edges)
    max_edge = maximum(eta_edges)
    min_volume = minimum(volumes)
    max_volume = maximum(volumes)
    interface_y_min = minimum(interface_y)
    interface_y_max = maximum(interface_y)
    return (
        min_det_jacobian=min_det,
        max_det_jacobian=max_det,
        det_jacobian_ratio=max_det / min_det,
        min_eta_edge=min_edge,
        max_eta_edge=max_edge,
        eta_edge_ratio=max_edge / min_edge,
        min_volume=min_volume,
        max_volume=max_volume,
        volume_ratio=max_volume / min_volume,
        interface_y_min=interface_y_min,
        interface_y_max=interface_y_max,
        interface_y_range=interface_y_max - interface_y_min,
        min_nonorthogonality_deg=minimum(angles),
        max_nonorthogonality_deg=maximum(angles),
        min_shared_node_error=minimum(node_errors),
        max_shared_node_error=maximum(node_errors),
    )
end

function metric_ct_verify_mesh_quality(
    summary;
    min_det_jacobian=0.0,
    min_eta_edge_ratio=1.4,
    min_volume_ratio=1.3,
    min_interface_y_range=0.19,
    min_nonorthogonality_deg=5.0,
    max_nonorthogonality_deg=45.0,
    max_shared_node_error=1.0e-13,
)
    checks = (
        (summary.min_det_jacobian > min_det_jacobian,
         "minimum Jacobian $(summary.min_det_jacobian) must exceed $min_det_jacobian"),
        (summary.eta_edge_ratio >= min_eta_edge_ratio,
         "eta edge ratio $(summary.eta_edge_ratio) is below $min_eta_edge_ratio"),
        (summary.volume_ratio >= min_volume_ratio,
         "volume ratio $(summary.volume_ratio) is below $min_volume_ratio"),
        (summary.interface_y_range >= min_interface_y_range,
         "interface y range $(summary.interface_y_range) is below $min_interface_y_range"),
        (summary.max_nonorthogonality_deg >= min_nonorthogonality_deg,
         "maximum nonorthogonality $(summary.max_nonorthogonality_deg) is below $min_nonorthogonality_deg degrees"),
        (summary.max_nonorthogonality_deg < max_nonorthogonality_deg,
         "maximum nonorthogonality $(summary.max_nonorthogonality_deg) must be below $max_nonorthogonality_deg degrees"),
        (summary.max_shared_node_error <= max_shared_node_error,
         "shared-node error $(summary.max_shared_node_error) exceeds $max_shared_node_error"),
    )
    all(isfinite, values(summary)) || error("mesh quality summary contains non-finite values")
    for (passed, message) in checks
        passed || error(message)
    end
    return summary
end

function metric_ct_load_mesh_quality(mesh_dir)
    expected_ng = 4
    isdir(mesh_dir) || error("mesh directory does not exist: $mesh_dir")
    connectivity_path = joinpath(mesh_dir, "block_connectivity.h5")
    isfile(connectivity_path) || error("missing connectivity file: $connectivity_path")
    configuration = h5open(connectivity_path, "r") do file
        Int(read(file["Nblocks"])) == 2 || error("metric CT quality mesh must contain two blocks")
        nx_b = Int64.(read(file["Nx_b"]))
        ny_b = Int64.(read(file["Ny_b"]))
        nz_b = Int64.(read(file["Nz_b"]))
        read(file["face_bc"]) == Int64[2 2 11 0 2 2; 2 2 0 11 2 2] ||
            error("face_bc does not match the curved two-block schema")
        bc_params = read(file["bc_params"])
        size(bc_params) == (2, 6, 20) && all(iszero, bc_params) ||
            error("bc_params must be a zeroed (2, 6, 20) array")
        read(file["connectivity"]) == Int64[0 4 1 3; 1 3 0 4] ||
            error("connectivity must pair face 4 with face 3 in both directions")
        Int64.(read(file["reverse_tan"])) == Int64[0, 0] ||
            error("reverse_tan must be zero for both connectivity rows")
        Int64.(read(file["flip_normal"])) == Int64[0, 0] ||
            error("flip_normal must be zero for both connectivity rows")
        (nx_b, ny_b, nz_b)
    end

    blocks = map(0:1) do block_id
        path = joinpath(mesh_dir, "mesh_b$block_id.h5")
        isfile(path) || error("missing block mesh: $path")
        h5open(path, "r") do file
            haskey(file, "NG") || error(
                "block $block_id mesh $path: observed missing NG dataset; " *
                "expected scalar integer NG = $expected_ng",
            )
            ng = read(file["NG"])
            ng isa Integer || error(
                "block $block_id mesh $path: observed $(repr(ng))::$(typeof(ng)); " *
                "expected scalar integer NG = $expected_ng",
            )
            coords = Float64.(read(file["coords"]))
            nx = Int(read(file["Nx"]))
            ny = Int(read(file["Ny"]))
            nz = Int(read(file["Nz"]))
            size(coords) == (3, nx + 1, ny + 1, nz + 1) ||
                throw(DimensionMismatch("invalid coords shape in $path"))
            (coords=coords, Nx=nx, Ny=ny, Nz=nz, NG=ng, path=path)
        end
    end
    for block_index in 2:length(blocks)
        block = blocks[block_index]
        block.NG == blocks[1].NG || error(
            "block $(block_index - 1) mesh $(block.path): observed $(block.NG); " *
            "expected NG = $expected_ng matching block 0 mesh $(blocks[1].path) " *
            "(observed $(blocks[1].NG))",
        )
    end
    for (block_index, block) in enumerate(blocks)
        block.NG == expected_ng || error(
            "block $(block_index - 1) mesh $(block.path): observed $(block.NG); " *
            "expected NG = $expected_ng",
        )
    end
    nx_b, ny_b, nz_b = configuration
    [block.Nx for block in blocks] == nx_b || error("Nx_b does not match block meshes")
    [block.Ny for block in blocks] == ny_b || error("Ny_b does not match block meshes")
    [block.Nz for block in blocks] == nz_b || error("Nz_b does not match block meshes")
    return metric_ct_mesh_quality(blocks...)
end

function _metric_ct_quality_main(args)
    isempty(args) && error(
        "usage: julia mesh_quality.jl MESH [--min-edge-ratio value] [--min-volume-ratio value]",
    )
    mesh_dir = first(args)
    options = Dict{String, Float64}()
    index = 2
    while index <= length(args)
        index < length(args) || error("missing value for $(args[index])")
        startswith(args[index], "--") || error("unknown argument: $(args[index])")
        options[args[index]] = parse(Float64, args[index + 1])
        index += 2
    end
    allowed = Set((
        "--min-det-jacobian", "--min-edge-ratio", "--min-volume-ratio",
        "--min-interface-range", "--min-nonorthogonality",
        "--max-nonorthogonality", "--max-shared-node-error",
    ))
    all(key -> key in allowed, keys(options)) || error("unknown quality threshold")

    summary = metric_ct_load_mesh_quality(mesh_dir)
    println(summary)
    metric_ct_verify_mesh_quality(
        summary;
        min_det_jacobian=get(options, "--min-det-jacobian", 0.0),
        min_eta_edge_ratio=get(options, "--min-edge-ratio", 1.4),
        min_volume_ratio=get(options, "--min-volume-ratio", 1.3),
        min_interface_y_range=get(options, "--min-interface-range", 0.19),
        min_nonorthogonality_deg=get(options, "--min-nonorthogonality", 5.0),
        max_nonorthogonality_deg=get(options, "--max-nonorthogonality", 45.0),
        max_shared_node_error=get(options, "--max-shared-node-error", 1.0e-13),
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    _metric_ct_quality_main(ARGS)
end
