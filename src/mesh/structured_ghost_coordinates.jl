# =============================================================================
# Runtime Ghost Coordinate Expansion & FVM Metric Computation
# 
# Ghost cell node coordinates are generated from real-only mesh coordinates:
#   - Periodic faces: copy from opposite boundary with period shift
#   - Interblock faces: copy from neighbor block's interior nodes
#   - Wall/other faces: mirror extrapolation (smooth, consistent metrics)
#   - Edge ghosts: product formula from face ghost values
#   - Corner ghosts: triple product from face ghost values
#
# NOTE: Ghost cell Q/U VALUES are filled by copy_ghost_face! (direct MPI copy)
#       and fillGhost (boundary conditions). The coordinates here only need
#       to produce smooth, reasonable metrics — not exact coincidence.
# =============================================================================

using LinearAlgebra
using HDF5
using SHA

if !isdefined(@__MODULE__, :StructuredFaceFrame)
    include(joinpath(@__DIR__, "..", "core", "structured_interface_transform.jl"))
end

const STRUCTURED_METRIC_TOPOLOGY_GHOST = :cmd6_topology_ghost
const STRUCTURED_METRIC_LOCAL_CHART = :cmd6_local_chart
const STRUCTURED_METRIC_LOCAL_CHART_POINTS = 6

struct MetricCoordinates{A}
    x::A
    y::A
    z::A
end

@inline function structured_metric_mode_setting()
    mode = if isdefined(Main, :structured_metric_mode)
        Symbol(getfield(Main, :structured_metric_mode))
    elseif haskey(ENV, "STRUCTURED_METRIC_MODE")
        Symbol(lowercase(strip(ENV["STRUCTURED_METRIC_MODE"])))
    else
        STRUCTURED_METRIC_TOPOLOGY_GHOST
    end
    mode in (STRUCTURED_METRIC_TOPOLOGY_GHOST, STRUCTURED_METRIC_LOCAL_CHART) ||
        throw(ArgumentError(
            "unsupported structured metric mode $mode; expected " *
            "$STRUCTURED_METRIC_TOPOLOGY_GHOST or $STRUCTURED_METRIC_LOCAL_CHART",
        ))
    return mode
end

function _structured_metric_extrapolation_weights(
    ::Type{T}, npoints::Int, ng::Int,
) where {T<:AbstractFloat}
    npoints >= 2 || throw(ArgumentError(
        "metric extrapolation requires at least two source nodes",
    ))
    weights = zeros(T, ng, npoints)
    for ghost_layer in 1:ng
        target = -T(ghost_layer)
        for source in 1:npoints
            source_coordinate = T(source - 1)
            value = one(T)
            for other in 1:npoints
                other == source && continue
                other_coordinate = T(other - 1)
                value *= (target - other_coordinate) /
                         (source_coordinate - other_coordinate)
            end
            weights[ghost_layer, source] = value
        end
    end
    all(isfinite, weights) || throw(ArgumentError(
        "non-finite local-chart metric extrapolation weights",
    ))
    maximum(abs, weights) <= T(1.0e6) || throw(ArgumentError(
        "ill-conditioned local-chart metric extrapolation weights",
    ))
    return weights
end

function _structured_metric_extrapolate_axis!(
    array, axis::Int, low::Bool, high::Bool,
    ncell::Int, ng::Int, npoints::Int,
)
    low || high || return array
    lo = ng + 1
    hi = ng + ncell + 1
    weights = _structured_metric_extrapolation_weights(
        eltype(array), npoints, ng,
    )

    if axis == 1
        for k in axes(array, 3), j in axes(array, 2), ghost in 1:ng
            if low
                array[lo-ghost,j,k] = sum(
                    weights[ghost,source] * array[lo+source-1,j,k]
                    for source in 1:npoints
                )
            end
            if high
                array[hi+ghost,j,k] = sum(
                    weights[ghost,source] * array[hi-source+1,j,k]
                    for source in 1:npoints
                )
            end
        end
    elseif axis == 2
        for k in axes(array, 3), i in axes(array, 1), ghost in 1:ng
            if low
                array[i,lo-ghost,k] = sum(
                    weights[ghost,source] * array[i,lo+source-1,k]
                    for source in 1:npoints
                )
            end
            if high
                array[i,hi+ghost,k] = sum(
                    weights[ghost,source] * array[i,hi-source+1,k]
                    for source in 1:npoints
                )
            end
        end
    elseif axis == 3
        for j in axes(array, 2), i in axes(array, 1), ghost in 1:ng
            if low
                array[i,j,lo-ghost] = sum(
                    weights[ghost,source] * array[i,j,lo+source-1]
                    for source in 1:npoints
                )
            end
            if high
                array[i,j,hi+ghost] = sum(
                    weights[ghost,source] * array[i,j,hi-source+1]
                    for source in 1:npoints
                )
            end
        end
    else
        throw(ArgumentError("metric coordinate axis must be in 1:3, got $axis"))
    end
    return array
end

function _structured_metric_local_chart_points(
    block_cells::Int, axis::Int,
)
    if block_cells == 1
        return 2
    end
    block_cells + 1 >= STRUCTURED_METRIC_LOCAL_CHART_POINTS ||
        throw(ArgumentError(
            "sixth-order metric halo on axis $axis requires at least " *
            "$(STRUCTURED_METRIC_LOCAL_CHART_POINTS) block nodes; " *
            "got $(block_cells + 1)",
        ))
    return STRUCTURED_METRIC_LOCAL_CHART_POINTS
end

"""
    build_metric_coordinates(x, y, z, Nx, Ny, Nz, NG; ...)

Return coordinates used only by the metric operators. In local-chart mode,
real nodes and same-block MPI-cut halos are retained, while halos outside a
logical block boundary are replaced by a one-sided sixth-order continuation
of that block's coordinate map. Physical/topological coordinates are never
modified.
"""
function build_metric_coordinates(
    x, y, z, Nx::Int, Ny::Int, Nz::Int, NG::Int;
    mode::Symbol=structured_metric_mode_setting(),
    cell_offsets::NTuple{3,Int}=(0, 0, 0),
    block_dims::NTuple{3,Int}=(Nx, Ny, Nz),
)
    mode == STRUCTURED_METRIC_TOPOLOGY_GHOST &&
        return MetricCoordinates(x, y, z)
    mode == STRUCTURED_METRIC_LOCAL_CHART || throw(ArgumentError(
        "unsupported structured metric mode $mode",
    ))
    local_dims = (Nx, Ny, Nz)
    coordinates = MetricCoordinates(copy(x), copy(y), copy(z))
    for axis in 1:3
        low = cell_offsets[axis] == 0
        high = cell_offsets[axis] + local_dims[axis] == block_dims[axis]
        low || high || continue
        npoints = _structured_metric_local_chart_points(block_dims[axis], axis)
        for array in (coordinates.x, coordinates.y, coordinates.z)
            _structured_metric_extrapolate_axis!(
                array, axis, low, high, local_dims[axis], NG, npoints,
            )
        end
    end
    return coordinates
end

# BC type constants loaded from bc_types.jl (included in solver.jl)

"""
    expand_coords_with_ghost(x_real, y_real, z_real, Nx, Ny, Nz, NG, ...)

Expand real-only node coordinates (Nx+1, Ny+1, Nz+1) to full padded 
coordinates (Nx+2NG+1, Ny+2NG+1, Nz+2NG+1) with ghost nodes.

- Periodic faces: copy from opposite end (with x-shift for ξ direction)
- Interblock faces: copy from neighbor block's interior nodes
- Wall/symmetry faces: mirror extrapolation from the boundary
"""
function expand_coords_with_ghost(x_real, y_real, z_real, 
                                   Nx::Int, Ny::Int, Nz::Int, NG::Int,
                                   face_bc, bid, connectivity;
                                   xi_offset::Int=0,
                                   cell_offsets::NTuple{3,Int}=(xi_offset, 0, 0),
                                   block_dims::NTuple{3,Int}=(Nx, Ny, Nz))
    Nx_tot = Nx + 2*NG + 1
    Ny_tot = Ny + 2*NG + 1
    Nz_tot = Nz + 2*NG + 1
    
    x = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    y = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    z = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    # Keep neighbor coordinates in a per-expansion host cache.
    mesh_cache = Dict{Int,Any}()
    
    # Step 1: Copy real nodes to interior
    ri = NG+1:Nx+NG+1
    rj = NG+1:Ny+NG+1
    rk = NG+1:Nz+NG+1
    x[ri, rj, rk] .= x_real
    y[ri, rj, rk] .= y_real
    z[ri, rj, rk] .= z_real
    
    # Step 2: Face ghost nodes — boundary-type-aware
    _fill_face_ghosts!(
        x, y, z, Nx, Ny, Nz, NG, face_bc, bid, connectivity;
        cell_offsets=cell_offsets,
        block_dims=block_dims,
        mesh_cache=mesh_cache,
    )
    
    # Step 3: Edge ghost nodes (12 edges) — product formula
    _fill_edge_ghosts!(x, y, z, Nx, Ny, Nz, NG)
    
    # Step 4: Corner ghost nodes (8 corners) — triple product
    _fill_corner_ghosts!(x, y, z, Nx, Ny, Nz, NG)

    # Product extrapolation is not exact for coupled curvilinear mappings.
    # Overwrite fully periodic edge/corner nodes by mapping to the equivalent
    # physical node and applying the domain translation in each ghost axis.
    _overwrite_periodic_edge_corner_ghosts!(
        x, y, z, Nx, Ny, Nz, NG, face_bc, bid,
    )

    _fill_topological_edge_corner_ghosts!(
        x, y, z, Nx, Ny, Nz, NG, face_bc, bid, connectivity;
        cell_offsets=cell_offsets,
        block_dims=block_dims,
        mesh_cache=mesh_cache,
    )
    
    return FT.(x), FT.(y), FT.(z)
end

@inline function _periodic_node_ghost_side(index, ncell, ng)
    if index <= ng
        return -1
    elseif index >= ncell + ng + 2
        return 1
    end
    return 0
end

@inline function _periodic_node_source_index(index, ncell, ghost_side)
    if ghost_side < 0
        return index + ncell
    elseif ghost_side > 0
        return index - ncell
    end
    return index
end

function _overwrite_periodic_edge_corner_ghosts!(
    x, y, z, Nx, Ny, Nz, NG, face_bc, bid,
)
    _bc(fid) = haskey(face_bc, (bid, fid)) ?
        face_bc[(bid, fid)] : BC_ISOTHERMAL_WALL
    periodic_face = (
        _bc(1) == BC_PERIODIC, _bc(2) == BC_PERIODIC,
        _bc(3) == BC_PERIODIC, _bc(4) == BC_PERIODIC,
        _bc(5) == BC_PERIODIC, _bc(6) == BC_PERIODIC,
    )
    ilo, ihi = NG + 1, Nx + NG + 1
    jlo, jhi = NG + 1, Ny + NG + 1
    klo, khi = NG + 1, Nz + NG + 1

    for k in axes(x, 3), j in axes(x, 2), i in axes(x, 1)
        side_i = _periodic_node_ghost_side(i, Nx, NG)
        side_j = _periodic_node_ghost_side(j, Ny, NG)
        side_k = _periodic_node_ghost_side(k, Nz, NG)
        ghost_count = (side_i != 0) + (side_j != 0) + (side_k != 0)
        ghost_count < 2 && continue

        i_periodic = side_i == 0 || periodic_face[side_i < 0 ? 1 : 2]
        j_periodic = side_j == 0 || periodic_face[side_j < 0 ? 3 : 4]
        k_periodic = side_k == 0 || periodic_face[side_k < 0 ? 5 : 6]
        i_periodic && j_periodic && k_periodic || continue

        source_i = _periodic_node_source_index(i, Nx, side_i)
        source_j = _periodic_node_source_index(j, Ny, side_j)
        source_k = _periodic_node_source_index(k, Nz, side_k)

        shift_x = zero(eltype(x))
        shift_y = zero(eltype(y))
        shift_z = zero(eltype(z))
        if side_i != 0
            shift_x += side_i * (x[ihi,source_j,source_k] - x[ilo,source_j,source_k])
            shift_y += side_i * (y[ihi,source_j,source_k] - y[ilo,source_j,source_k])
            shift_z += side_i * (z[ihi,source_j,source_k] - z[ilo,source_j,source_k])
        end
        if side_j != 0
            shift_x += side_j * (x[source_i,jhi,source_k] - x[source_i,jlo,source_k])
            shift_y += side_j * (y[source_i,jhi,source_k] - y[source_i,jlo,source_k])
            shift_z += side_j * (z[source_i,jhi,source_k] - z[source_i,jlo,source_k])
        end
        if side_k != 0
            shift_x += side_k * (x[source_i,source_j,khi] - x[source_i,source_j,klo])
            shift_y += side_k * (y[source_i,source_j,khi] - y[source_i,source_j,klo])
            shift_z += side_k * (z[source_i,source_j,khi] - z[source_i,source_j,klo])
        end

        x[i,j,k] = x[source_i,source_j,source_k] + shift_x
        y[i,j,k] = y[source_i,source_j,source_k] + shift_y
        z[i,j,k] = z[source_i,source_j,source_k] + shift_z
    end
    return
end

# =============================================================================
# Coordinate topology helpers
# =============================================================================

@inline function _ghost_mesh_coords(mesh_cache, bid)
    if mesh_cache !== nothing && haskey(mesh_cache, bid)
        return mesh_cache[bid]
    end
    mesh_base = isdefined(Main, :mesh_dir) ? getfield(Main, :mesh_dir) : "MESH"
    path = joinpath(mesh_base, "mesh_b$(bid).h5")
    isfile(path) || return nothing
    coords = h5open(path, "r") do file
        haskey(file, "coords") || return nothing
        read(file["coords"])
    end
    coords === nothing && return nothing
    value = (
        Float64.(coords[1, :, :, :]),
        Float64.(coords[2, :, :, :]),
        Float64.(coords[3, :, :, :]),
    )
    mesh_cache !== nothing && (mesh_cache[bid] = value)
    return value
end

@inline function _topology_interblock_bc(bc)
    value = Int(bc)
    return value == 0 ||
           (isdefined(@__MODULE__, :BC_INTERBLOCK) &&
            value == Int(BC_INTERBLOCK))
end

@inline function _topology_periodic_bc(bc)
    return isdefined(@__MODULE__, :BC_PERIODIC) &&
           Int(bc) == Int(BC_PERIODIC)
end

@inline function _topology_face_id(axis::Int, side::Int)
    return side < 0 ? 2*axis - 1 : 2*axis
end

@inline function _subdomain_touches_global_face(
    fid::Int, local_dims::NTuple{3,Int}, cell_offsets::NTuple{3,Int},
    block_dims::NTuple{3,Int},
)
    axis = (fid + 1) ÷ 2
    if isodd(fid)
        return cell_offsets[axis] == 0
    end
    return cell_offsets[axis] + local_dims[axis] == block_dims[axis]
end

function _fill_local_subdomain_face!(
    x, y, z, Nx, Ny, Nz, NG, bid, fid;
    cell_offsets::NTuple{3,Int}, block_dims::NTuple{3,Int}, mesh_cache,
)
    coords = _ghost_mesh_coords(mesh_cache, bid)
    coords === nothing && return false
    x_full, y_full, z_full = coords
    local_dims = (Nx, Ny, Nz)
    global_axis = (fid + 1) ÷ 2
    u_axis, v_axis = _face_tangent_axes(fid)
    u_nodes = local_dims[u_axis] + 1
    v_nodes = local_dims[v_axis] + 1
    side = isodd(fid) ? -1 : 1
    for g in 1:NG, v in 1:v_nodes, u in 1:u_nodes
        local_normal = side < 0 ? NG + 1 - g :
            local_dims[global_axis] + NG + 1 + g
        global_normal = side < 0 ? cell_offsets[global_axis] + 1 - g :
            cell_offsets[global_axis] + local_dims[global_axis] + 1 + g
        local_index = _face_node_index(
            fid, local_normal, u + NG, v + NG,
        )
        global_index = zeros(Int, 3)
        global_index[global_axis] = global_normal
        global_index[u_axis] = cell_offsets[u_axis] + u
        global_index[v_axis] = cell_offsets[v_axis] + v
        all(1 <= global_index[d] <= block_dims[d] + 1 for d in 1:3) ||
            return false
        source = Tuple(global_index)
        x[local_index...] = x_full[source...]
        y[local_index...] = y_full[source...]
        z[local_index...] = z_full[source...]
    end
    return true
end

# =============================================================================
# Face ghost filling: periodic / interblock / mirror
# =============================================================================
function _fill_face_ghosts!(
    x, y, z, Nx, Ny, Nz, NG, face_bc, bid, connectivity;
    cell_offsets::NTuple{3,Int}=(0, 0, 0),
    block_dims::NTuple{3,Int}=(Nx, Ny, Nz),
    mesh_cache=nothing,
)
    ri = NG+1:Nx+NG+1
    rj = NG+1:Ny+NG+1
    rk = NG+1:Nz+NG+1
    
    # Helper: get BC type for (block, face)
    _bc(fid) = haskey(face_bc, (bid, fid)) ? face_bc[(bid, fid)] : BC_ISOTHERMAL_WALL
    
    # ── ξ- (face 1) ──
    if !_subdomain_touches_global_face(1, (Nx, Ny, Nz), cell_offsets, block_dims) &&
       _fill_local_subdomain_face!(
           x, y, z, Nx, Ny, Nz, NG, bid, 1;
           cell_offsets, block_dims, mesh_cache,
       )
    elseif _bc(1) == BC_INTERBLOCK
        _fill_interblock_face!(
            x, y, z, Nx, Ny, Nz, NG, bid, 1, connectivity;
            cell_offsets=cell_offsets,
            mesh_cache=mesh_cache,
        )
    elseif _bc(1) == BC_PERIODIC
        # Copy from ξ+ end, shift x by -period
        for g in 1:NG, k in rk, j in rj
            x[NG+1-g, j, k] = x[NG+Nx+1-g, j, k] - (x[NG+Nx+1, j, k] - x[NG+1, j, k])
            y[NG+1-g, j, k] = y[NG+Nx+1-g, j, k]
            z[NG+1-g, j, k] = z[NG+Nx+1-g, j, k]
        end
    else  # mirror extrapolation
        for g in 1:NG, k in rk, j in rj
            x[NG+1-g, j, k] = 2*x[NG+1, j, k] - x[NG+1+g, j, k]
            y[NG+1-g, j, k] = 2*y[NG+1, j, k] - y[NG+1+g, j, k]
            z[NG+1-g, j, k] = 2*z[NG+1, j, k] - z[NG+1+g, j, k]
        end
    end
    
    # ── ξ+ (face 2) ──
    if !_subdomain_touches_global_face(2, (Nx, Ny, Nz), cell_offsets, block_dims) &&
       _fill_local_subdomain_face!(
           x, y, z, Nx, Ny, Nz, NG, bid, 2;
           cell_offsets, block_dims, mesh_cache,
       )
    elseif _bc(2) == BC_INTERBLOCK
        _fill_interblock_face!(
            x, y, z, Nx, Ny, Nz, NG, bid, 2, connectivity;
            cell_offsets=cell_offsets,
            mesh_cache=mesh_cache,
        )
    elseif _bc(2) == BC_PERIODIC
        # Copy from ξ- end, shift x by +period
        for g in 1:NG, k in rk, j in rj
            x[NG+Nx+1+g, j, k] = x[NG+1+g, j, k] + (x[NG+Nx+1, j, k] - x[NG+1, j, k])
            y[NG+Nx+1+g, j, k] = y[NG+1+g, j, k]
            z[NG+Nx+1+g, j, k] = z[NG+1+g, j, k]
        end
    else
        for g in 1:NG, k in rk, j in rj
            x[NG+Nx+1+g, j, k] = 2*x[NG+Nx+1, j, k] - x[NG+Nx+1-g, j, k]
            y[NG+Nx+1+g, j, k] = 2*y[NG+Nx+1, j, k] - y[NG+Nx+1-g, j, k]
            z[NG+Nx+1+g, j, k] = 2*z[NG+Nx+1, j, k] - z[NG+Nx+1-g, j, k]
        end
    end
    
    # ── η- (face 3) ──
    if !_subdomain_touches_global_face(3, (Nx, Ny, Nz), cell_offsets, block_dims) &&
       _fill_local_subdomain_face!(
           x, y, z, Nx, Ny, Nz, NG, bid, 3;
           cell_offsets, block_dims, mesh_cache,
       )
    elseif _bc(3) == BC_INTERBLOCK
        _fill_interblock_face!(
            x, y, z, Nx, Ny, Nz, NG, bid, 3, connectivity;
            cell_offsets=cell_offsets,
            mesh_cache=mesh_cache,
        )
    elseif _bc(3) == BC_PERIODIC
        # Copy from η+ end, shift y by -period
        for g in 1:NG, k in rk, i in ri
            x[i, NG+1-g, k] = x[i, NG+Ny+1-g, k]
            y[i, NG+1-g, k] = y[i, NG+Ny+1-g, k] - (y[i, NG+Ny+1, k] - y[i, NG+1, k])
            z[i, NG+1-g, k] = z[i, NG+Ny+1-g, k]
        end
    else  # mirror extrapolation
        for g in 1:NG, k in rk, i in ri
            x[i, NG+1-g, k] = 2*x[i, NG+1, k] - x[i, NG+1+g, k]
            y[i, NG+1-g, k] = 2*y[i, NG+1, k] - y[i, NG+1+g, k]
            z[i, NG+1-g, k] = 2*z[i, NG+1, k] - z[i, NG+1+g, k]
        end
    end

    # ── η+ (face 4) ──
    if !_subdomain_touches_global_face(4, (Nx, Ny, Nz), cell_offsets, block_dims) &&
       _fill_local_subdomain_face!(
           x, y, z, Nx, Ny, Nz, NG, bid, 4;
           cell_offsets, block_dims, mesh_cache,
       )
    elseif _bc(4) == BC_INTERBLOCK
        _fill_interblock_face!(
            x, y, z, Nx, Ny, Nz, NG, bid, 4, connectivity;
            cell_offsets=cell_offsets,
            mesh_cache=mesh_cache,
        )
    elseif _bc(4) == BC_PERIODIC
        # Copy from η- end, shift y by +period
        for g in 1:NG, k in rk, i in ri
            x[i, Ny+NG+1+g, k] = x[i, NG+1+g, k]
            y[i, Ny+NG+1+g, k] = y[i, NG+1+g, k] + (y[i, Ny+NG+1, k] - y[i, NG+1, k])
            z[i, Ny+NG+1+g, k] = z[i, NG+1+g, k]
        end
    else  # mirror extrapolation
        for g in 1:NG, k in rk, i in ri
            x[i, Ny+NG+1+g, k] = 2*x[i, Ny+NG+1, k] - x[i, Ny+NG+1-g, k]
            y[i, Ny+NG+1+g, k] = 2*y[i, Ny+NG+1, k] - y[i, Ny+NG+1-g, k]
            z[i, Ny+NG+1+g, k] = 2*z[i, Ny+NG+1, k] - z[i, Ny+NG+1-g, k]
        end
    end

    # ── ζ- (face 5) ──
    if !_subdomain_touches_global_face(5, (Nx, Ny, Nz), cell_offsets, block_dims) &&
       _fill_local_subdomain_face!(
           x, y, z, Nx, Ny, Nz, NG, bid, 5;
           cell_offsets, block_dims, mesh_cache,
       )
    elseif _bc(5) == BC_INTERBLOCK
        _fill_interblock_face!(
            x, y, z, Nx, Ny, Nz, NG, bid, 5, connectivity;
            cell_offsets=cell_offsets,
            mesh_cache=mesh_cache,
        )
    elseif _bc(5) == BC_PERIODIC
        # Copy from ζ+ end, shift z by -period
        for g in 1:NG, j in rj, i in ri
            x[i, j, NG+1-g] = x[i, j, NG+Nz+1-g]
            y[i, j, NG+1-g] = y[i, j, NG+Nz+1-g]
            z[i, j, NG+1-g] = z[i, j, NG+Nz+1-g] - (z[i, j, NG+Nz+1] - z[i, j, NG+1])
        end
    else  # mirror extrapolation
        for g in 1:NG, j in rj, i in ri
            x[i, j, NG+1-g] = 2*x[i, j, NG+1] - x[i, j, NG+1+g]
            y[i, j, NG+1-g] = 2*y[i, j, NG+1] - y[i, j, NG+1+g]
            z[i, j, NG+1-g] = 2*z[i, j, NG+1] - z[i, j, NG+1+g]
        end
    end

    # ── ζ+ (face 6) ──
    if !_subdomain_touches_global_face(6, (Nx, Ny, Nz), cell_offsets, block_dims) &&
       _fill_local_subdomain_face!(
           x, y, z, Nx, Ny, Nz, NG, bid, 6;
           cell_offsets, block_dims, mesh_cache,
       )
    elseif _bc(6) == BC_INTERBLOCK
        _fill_interblock_face!(
            x, y, z, Nx, Ny, Nz, NG, bid, 6, connectivity;
            cell_offsets=cell_offsets,
            mesh_cache=mesh_cache,
        )
    elseif _bc(6) == BC_PERIODIC
        # Copy from ζ- end, shift z by +period
        for g in 1:NG, j in rj, i in ri
            x[i, j, Nz+NG+1+g] = x[i, j, NG+1+g]
            y[i, j, Nz+NG+1+g] = y[i, j, NG+1+g]
            z[i, j, Nz+NG+1+g] = z[i, j, NG+1+g] + (z[i, j, Nz+NG+1] - z[i, j, NG+1])
        end
    else  # mirror extrapolation
        for g in 1:NG, j in rj, i in ri
            x[i, j, Nz+NG+1+g] = 2*x[i, j, Nz+NG+1] - x[i, j, Nz+NG+1-g]
            y[i, j, Nz+NG+1+g] = 2*y[i, j, Nz+NG+1] - y[i, j, Nz+NG+1-g]
            z[i, j, Nz+NG+1+g] = 2*z[i, j, Nz+NG+1] - z[i, j, Nz+NG+1-g]
        end
    end
end

# =============================================================================
# Interblock face: copy NG layers of nodes from neighbor block's interior
# =============================================================================
@inline function _face_normal_axis(fid::Int)
    1 <= fid <= 6 || throw(ArgumentError("face id must be in 1:6, got $fid"))
    return (fid + 1) ÷ 2
end

@inline function _face_tangent_axes(fid::Int)
    normal_axis = _face_normal_axis(fid)
    normal_axis == 1 && return (2, 3)
    normal_axis == 2 && return (1, 3)
    return (1, 2)
end

@inline _axis_length(dims::NTuple{3,Int}, axis::Int) = dims[axis]

function _face_node_index(
    fid::Int, normal_index::Int, u_index::Int, v_index::Int,
)
    normal_axis = _face_normal_axis(fid)
    u_axis, v_axis = _face_tangent_axes(fid)
    return ntuple(3) do axis
        axis == normal_axis ? normal_index :
        axis == u_axis ? u_index : v_index
    end
end

@inline function _interblock_source_normal_index(
    nb_fid::Int, nb_dims::NTuple{3,Int}, ghost_layer::Int,
)
    nb_normal_axis = _face_normal_axis(nb_fid)
    nb_normal_cells = _axis_length(nb_dims, nb_normal_axis)
    return isodd(nb_fid) ? ghost_layer + 1 : nb_normal_cells + 1 - ghost_layer
end

function _fill_interblock_face!(
    x, y, z, Nx, Ny, Nz, NG, bid, fid, connectivity;
    cell_offsets::NTuple{3,Int}=(0, 0, 0),
    mesh_cache=nothing,
)
    haskey(connectivity, (bid, fid)) || throw(ArgumentError(
        "missing connectivity for block $bid face $fid",
    ))
    conn = connectivity[(bid, fid)]
    nb_bid = conn.src_b
    nb_fid = conn.src_f
    
    # Load neighbor block's real-only mesh once per expansion.
    nb_coords = _ghost_mesh_coords(mesh_cache, nb_bid)
    nb_coords === nothing && throw(ArgumentError(
        "missing coordinates for neighbor block $nb_bid",
    ))
    x_nb, y_nb, z_nb = nb_coords
    Nx_nb = size(x_nb, 1) - 1
    Ny_nb = size(x_nb, 2) - 1
    Nz_nb = size(x_nb, 3) - 1
    
    local_dims = (Nx, Ny, Nz)
    nb_dims = (Nx_nb, Ny_nb, Nz_nb)
    local_normal_axis = _face_normal_axis(fid)
    local_u_axis, local_v_axis = _face_tangent_axes(fid)
    nb_u_axis, nb_v_axis = _face_tangent_axes(nb_fid)
    local_u_nodes = _axis_length(local_dims, local_u_axis) + 1
    local_v_nodes = _axis_length(local_dims, local_v_axis) + 1
    nb_u_nodes = _axis_length(nb_dims, nb_u_axis) + 1
    nb_v_nodes = _axis_length(nb_dims, nb_v_axis) + 1

    transform = if hasproperty(conn, :transform) && getproperty(conn, :transform) !== nothing
        candidate = getproperty(conn, :transform)
        candidate.source_face == nb_fid && candidate.destination_face == fid ?
            candidate : structured_inverse_face_transform(candidate)
    else
        structured_legacy_face_transform(nb_fid, fid, conn.reverse_tan)
    end

    # flip_normal applies to oriented normals and fluxes. Coordinates are
    # scalars; the signed axis permutation maps both independent tangential
    # directions, including u/v exchange and independent reversals.
    conn.flip_normal isa Bool || throw(ArgumentError(
        "flip_normal must be Bool for block $bid face $fid",
    ))

    for g in 1:NG, v_local in 1:local_v_nodes, u_local in 1:local_u_nodes
        destination_index = zeros(Int, 3)
        destination_index[local_u_axis] = u_local + cell_offsets[local_u_axis]
        destination_index[local_v_axis] = v_local + cell_offsets[local_v_axis]
        source_index = zeros(Int, 3)
        destination_frame = structured_face_frame(fid)
        for destination_axis in (Int(destination_frame.u_axis), Int(destination_frame.v_axis))
            source_value = transform.source_for_destination[destination_axis]
            source_axis = abs(source_value)
            value = destination_index[destination_axis]
            source_dim = (Nx_nb, Ny_nb, Nz_nb)[source_axis] + 1
            source_index[source_axis] = source_value < 0 ?
                source_dim + 1 - value : value
        end
        u_nb = source_index[nb_u_axis]
        v_nb = source_index[nb_v_axis]
        1 <= u_nb <= nb_u_nodes || throw(DimensionMismatch(
            "block $bid face $fid u-index $u_nb is outside neighbor block " *
            "$nb_bid face $nb_fid range 1:$nb_u_nodes",
        ))
        1 <= v_nb <= nb_v_nodes || throw(DimensionMismatch(
            "block $bid face $fid v-index $v_nb is outside neighbor block " *
            "$nb_bid face $nb_fid range 1:$nb_v_nodes",
        ))

        local_normal = isodd(fid) ? NG + 1 - g :
            _axis_length(local_dims, local_normal_axis) + NG + 1 + g
        target = _face_node_index(
            fid, local_normal, u_local + NG, v_local + NG,
        )
        source = _face_node_index(
            nb_fid,
            _interblock_source_normal_index(nb_fid, nb_dims, g),
            u_nb,
            v_nb,
        )
        shift = hasproperty(conn, :image_translation) ?
            getproperty(conn, :image_translation) : (0.0, 0.0, 0.0)
        x[target...] = x_nb[source...] + shift[1]
        y[target...] = y_nb[source...] + shift[2]
        z[target...] = z_nb[source...] + shift[3]
    end
    return
end

# =============================================================================
# Edge ghosts: product formula from face ghost values
# Edge = ghost in 2 directions, real in 1
# x_edge[i,j,k] = x_face1[i,j_bnd,k] + x_face2[i_bnd,j,k] - x[i_bnd,j_bnd,k]
# =============================================================================
function _fill_edge_ghosts!(x, y, z, Nx, Ny, Nz, NG)
    ri = NG+1:Nx+NG+1
    rj = NG+1:Ny+NG+1
    rk = NG+1:Nz+NG+1
    
    # 4 edges along i-direction (j,k ghost pairs)
    for k_g in 1:NG, j_g in 1:NG, i in ri  # j-lo, k-lo
        x[i, j_g, k_g] = x[i, NG+1, k_g] + x[i, j_g, NG+1] - x[i, NG+1, NG+1]
        y[i, j_g, k_g] = y[i, NG+1, k_g] + y[i, j_g, NG+1] - y[i, NG+1, NG+1]
        z[i, j_g, k_g] = z[i, NG+1, k_g] + z[i, j_g, NG+1] - z[i, NG+1, NG+1]
    end
    for k_g in Nz+NG+2:Nz+2*NG+1, j_g in 1:NG, i in ri  # j-lo, k-hi
        x[i, j_g, k_g] = x[i, NG+1, k_g] + x[i, j_g, Nz+NG+1] - x[i, NG+1, Nz+NG+1]
        y[i, j_g, k_g] = y[i, NG+1, k_g] + y[i, j_g, Nz+NG+1] - y[i, NG+1, Nz+NG+1]
        z[i, j_g, k_g] = z[i, NG+1, k_g] + z[i, j_g, Nz+NG+1] - z[i, NG+1, Nz+NG+1]
    end
    for k_g in 1:NG, j_g in Ny+NG+2:Ny+2*NG+1, i in ri  # j-hi, k-lo
        x[i, j_g, k_g] = x[i, Ny+NG+1, k_g] + x[i, j_g, NG+1] - x[i, Ny+NG+1, NG+1]
        y[i, j_g, k_g] = y[i, Ny+NG+1, k_g] + y[i, j_g, NG+1] - y[i, Ny+NG+1, NG+1]
        z[i, j_g, k_g] = z[i, Ny+NG+1, k_g] + z[i, j_g, NG+1] - z[i, Ny+NG+1, NG+1]
    end
    for k_g in Nz+NG+2:Nz+2*NG+1, j_g in Ny+NG+2:Ny+2*NG+1, i in ri  # j-hi, k-hi
        x[i, j_g, k_g] = x[i, Ny+NG+1, k_g] + x[i, j_g, Nz+NG+1] - x[i, Ny+NG+1, Nz+NG+1]
        y[i, j_g, k_g] = y[i, Ny+NG+1, k_g] + y[i, j_g, Nz+NG+1] - y[i, Ny+NG+1, Nz+NG+1]
        z[i, j_g, k_g] = z[i, Ny+NG+1, k_g] + z[i, j_g, Nz+NG+1] - z[i, Ny+NG+1, Nz+NG+1]
    end
    
    # 4 edges along j-direction (i,k ghost pairs)
    for k_g in 1:NG, j in rj, i_g in 1:NG  # i-lo, k-lo
        x[i_g, j, k_g] = x[NG+1, j, k_g] + x[i_g, j, NG+1] - x[NG+1, j, NG+1]
        y[i_g, j, k_g] = y[NG+1, j, k_g] + y[i_g, j, NG+1] - y[NG+1, j, NG+1]
        z[i_g, j, k_g] = z[NG+1, j, k_g] + z[i_g, j, NG+1] - z[NG+1, j, NG+1]
    end
    for k_g in Nz+NG+2:Nz+2*NG+1, j in rj, i_g in 1:NG  # i-lo, k-hi
        x[i_g, j, k_g] = x[NG+1, j, k_g] + x[i_g, j, Nz+NG+1] - x[NG+1, j, Nz+NG+1]
        y[i_g, j, k_g] = y[NG+1, j, k_g] + y[i_g, j, Nz+NG+1] - y[NG+1, j, Nz+NG+1]
        z[i_g, j, k_g] = z[NG+1, j, k_g] + z[i_g, j, Nz+NG+1] - z[NG+1, j, Nz+NG+1]
    end
    for k_g in 1:NG, j in rj, i_g in Nx+NG+2:Nx+2*NG+1  # i-hi, k-lo
        x[i_g, j, k_g] = x[Nx+NG+1, j, k_g] + x[i_g, j, NG+1] - x[Nx+NG+1, j, NG+1]
        y[i_g, j, k_g] = y[Nx+NG+1, j, k_g] + y[i_g, j, NG+1] - y[Nx+NG+1, j, NG+1]
        z[i_g, j, k_g] = z[Nx+NG+1, j, k_g] + z[i_g, j, NG+1] - z[Nx+NG+1, j, NG+1]
    end
    for k_g in Nz+NG+2:Nz+2*NG+1, j in rj, i_g in Nx+NG+2:Nx+2*NG+1  # i-hi, k-hi
        x[i_g, j, k_g] = x[Nx+NG+1, j, k_g] + x[i_g, j, Nz+NG+1] - x[Nx+NG+1, j, Nz+NG+1]
        y[i_g, j, k_g] = y[Nx+NG+1, j, k_g] + y[i_g, j, Nz+NG+1] - y[Nx+NG+1, j, Nz+NG+1]
        z[i_g, j, k_g] = z[Nx+NG+1, j, k_g] + z[i_g, j, Nz+NG+1] - z[Nx+NG+1, j, Nz+NG+1]
    end
    
    # 4 edges along k-direction (i,j ghost pairs)
    for k in rk, j_g in 1:NG, i_g in 1:NG  # i-lo, j-lo
        x[i_g, j_g, k] = x[NG+1, j_g, k] + x[i_g, NG+1, k] - x[NG+1, NG+1, k]
        y[i_g, j_g, k] = y[NG+1, j_g, k] + y[i_g, NG+1, k] - y[NG+1, NG+1, k]
        z[i_g, j_g, k] = z[NG+1, j_g, k] + z[i_g, NG+1, k] - z[NG+1, NG+1, k]
    end
    for k in rk, j_g in Ny+NG+2:Ny+2*NG+1, i_g in 1:NG  # i-lo, j-hi
        x[i_g, j_g, k] = x[NG+1, j_g, k] + x[i_g, Ny+NG+1, k] - x[NG+1, Ny+NG+1, k]
        y[i_g, j_g, k] = y[NG+1, j_g, k] + y[i_g, Ny+NG+1, k] - y[NG+1, Ny+NG+1, k]
        z[i_g, j_g, k] = z[NG+1, j_g, k] + z[i_g, Ny+NG+1, k] - z[NG+1, Ny+NG+1, k]
    end
    for k in rk, j_g in 1:NG, i_g in Nx+NG+2:Nx+2*NG+1  # i-hi, j-lo
        x[i_g, j_g, k] = x[Nx+NG+1, j_g, k] + x[i_g, NG+1, k] - x[Nx+NG+1, NG+1, k]
        y[i_g, j_g, k] = y[Nx+NG+1, j_g, k] + y[i_g, NG+1, k] - y[Nx+NG+1, NG+1, k]
        z[i_g, j_g, k] = z[Nx+NG+1, j_g, k] + z[i_g, NG+1, k] - z[Nx+NG+1, NG+1, k]
    end
    for k in rk, j_g in Ny+NG+2:Ny+2*NG+1, i_g in Nx+NG+2:Nx+2*NG+1  # i-hi, j-hi
        x[i_g, j_g, k] = x[Nx+NG+1, j_g, k] + x[i_g, Ny+NG+1, k] - x[Nx+NG+1, Ny+NG+1, k]
        y[i_g, j_g, k] = y[Nx+NG+1, j_g, k] + y[i_g, Ny+NG+1, k] - y[Nx+NG+1, Ny+NG+1, k]
        z[i_g, j_g, k] = z[Nx+NG+1, j_g, k] + z[i_g, Ny+NG+1, k] - z[Nx+NG+1, Ny+NG+1, k]
    end
end

# =============================================================================
# Corner ghosts: triple product from face ghost values
# =============================================================================
function _fill_corner_ghosts!(x, y, z, Nx, Ny, Nz, NG)
    i_ranges = [(1:NG, NG+1), (Nx+NG+2:Nx+2*NG+1, Nx+NG+1)]
    j_ranges = [(1:NG, NG+1), (Ny+NG+2:Ny+2*NG+1, Ny+NG+1)]
    k_ranges = [(1:NG, NG+1), (Nz+NG+2:Nz+2*NG+1, Nz+NG+1)]
    
    for (ir, ib) in i_ranges, (jr, jb) in j_ranges, (kr, kb) in k_ranges
        for k_g in kr, j_g in jr, i_g in ir
            x[i_g, j_g, k_g] = x[i_g, j_g, kb] + x[i_g, jb, k_g] - x[i_g, jb, kb] +
                                x[ib, j_g, k_g] - x[ib, j_g, kb] - x[ib, jb, k_g] + x[ib, jb, kb]
            y[i_g, j_g, k_g] = y[i_g, j_g, kb] + y[i_g, jb, k_g] - y[i_g, jb, kb] +
                                y[ib, j_g, k_g] - y[ib, j_g, kb] - y[ib, jb, k_g] + y[ib, jb, kb]
            z[i_g, j_g, k_g] = z[i_g, j_g, kb] + z[i_g, jb, k_g] - z[i_g, jb, kb] +
                                z[ib, j_g, k_g] - z[ib, j_g, kb] - z[ib, jb, k_g] + z[ib, jb, kb]
        end
    end
end

# =============================================================================
# Multi-block edge/corner topology
# =============================================================================

const STRUCTURED_BLOCK_EDGE_FACE_PAIRS = (
    (3, 5), (3, 6), (4, 5), (4, 6),
    (1, 5), (1, 6), (2, 5), (2, 6),
    (1, 3), (1, 4), (2, 3), (2, 4),
)

const STRUCTURED_BLOCK_EDGE_AXES = (
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
)

const STRUCTURED_BLOCK_EDGE_PINS = (
    (2, 0, 3, 0), (2, 0, 3, 1),
    (2, 1, 3, 0), (2, 1, 3, 1),
    (1, 0, 3, 0), (1, 0, 3, 1),
    (1, 1, 3, 0), (1, 1, 3, 1),
    (1, 0, 2, 0), (1, 0, 2, 1),
    (1, 1, 2, 0), (1, 1, 2, 1),
)

const STRUCTURED_SCMM_JUNCTION_WIDTH = 2

function structured_metric_singularity_edges(bid, face_bc, connectivity)
    singular = falses(length(STRUCTURED_BLOCK_EDGE_FACE_PAIRS))
    for (edge, (face_a, face_b)) in pairs(STRUCTURED_BLOCK_EDGE_FACE_PAIRS)
        _topology_interblock_bc(get(face_bc, (bid, face_a), Int32(-1))) ||
            continue
        _topology_interblock_bc(get(face_bc, (bid, face_b), Int32(-1))) ||
            continue
        connection_a = get(connectivity, (bid, face_a), nothing)
        connection_b = get(connectivity, (bid, face_b), nothing)
        connection_a === nothing && continue
        connection_b === nothing && continue
        singular[edge] = connection_a.src_b != connection_b.src_b
    end
    return singular
end

function structured_metric_singularity_edges_from_mask(mask::Integer)
    return BitVector(
        (UInt16(mask) & (UInt16(1) << (edge - 1))) != 0
        for edge in eachindex(STRUCTURED_BLOCK_EDGE_FACE_PAIRS)
    )
end

function structured_local_metric_singularity_edges(
    bid, face_bc, connectivity, cell_offsets, local_dims, block_dims,
)
    singular = structured_metric_singularity_edges(
        bid, face_bc, connectivity,
    )
    for edge in eachindex(singular)
        singular[edge] || continue
        axis_a, high_a, axis_b, high_b = STRUCTURED_BLOCK_EDGE_PINS[edge]
        on_a = high_a == 0 ? cell_offsets[axis_a] == 0 :
            cell_offsets[axis_a] + local_dims[axis_a] == block_dims[axis_a]
        on_b = high_b == 0 ? cell_offsets[axis_b] == 0 :
            cell_offsets[axis_b] + local_dims[axis_b] == block_dims[axis_b]
        singular[edge] = on_a && on_b
    end
    return singular
end

function _topological_connection_transform(local_fid, conn)
    nb_fid = conn.src_f
    if hasproperty(conn, :transform) && getproperty(conn, :transform) !== nothing
        candidate = getproperty(conn, :transform)
        return candidate.source_face == nb_fid &&
               candidate.destination_face == local_fid ?
            candidate : structured_inverse_face_transform(candidate)
    end
    return structured_legacy_face_transform(nb_fid, local_fid,
                                            getproperty(conn, :reverse_tan))
end

@inline function _periodic_node_shift(
    coords, idx::NTuple{3,Int}, axis::Int, side::Int, dims::NTuple{3,Int},
)
    reference = ntuple(d -> clamp(idx[d], 1, dims[d] + 1), 3)
    low = ntuple(d -> d == axis ? 1 : reference[d], 3)
    high = ntuple(d -> d == axis ? dims[axis] + 1 : reference[d], 3)
    return (
        side * (coords[1][high...] - coords[1][low...]),
        side * (coords[2][high...] - coords[2][low...]),
        side * (coords[3][high...] - coords[3][low...]),
    )
end

function _resolve_topological_node(
    bid::Int, index::NTuple{3,Int}, dims::NTuple{3,Int},
    face_bc, connectivity, mesh_cache;
    max_steps::Int=12,
)
    # A junction node has several equivalent representations.  Following
    # only the first out-of-range coordinate makes the result depend on which
    # block/face happened to be used to enter the junction.  Enumerate the
    # bounded connectivity graph instead, then choose one deterministic
    # canonical real node.  This is load-time work (only edge/corner ghosts),
    # so the small amount of host-side state is preferable to inconsistent
    # high-order metric stencils.
    State = Tuple{Int,NTuple{3,Int},NTuple{3,Int},NTuple{3,Float64},Int}
    queue = State[(bid, index, dims, (0.0, 0.0, 0.0), 0)]
    visited = Set{Tuple{Int,NTuple{3,Int},NTuple{3,Float64}}}()
    candidates = Tuple{Int,NTuple{3,Int},NTuple{3,Float64}}[]
    default_bc = isdefined(@__MODULE__, :BC_ISOTHERMAL_WALL) ?
        BC_ISOTHERMAL_WALL : Int32(3)

    while !isempty(queue)
        current_bid, current_index, current_dims, shift, depth = popfirst!(queue)
        state_key = (current_bid, current_index, shift)
        state_key in visited && continue
        push!(visited, state_key)
        depth <= max_steps || continue

        inside = all(
            1 <= current_index[axis] <= current_dims[axis] + 1
            for axis in 1:3
        )
        if inside
            push!(candidates, (current_bid, current_index, shift))
        end

        # Explore every topological face that is either crossed by this node
        # or contains it on its boundary.  The latter is what makes a
        # one-direction ghost on an inter-block face corner-independent.
        face_ids = Int[]
        for axis in 1:3
            value = current_index[axis]
            side = value < 1 ? -1 : value > current_dims[axis] + 1 ? 1 : 0
            if side != 0
                push!(face_ids, _topology_face_id(axis, side))
            elseif value == 1
                fid = _topology_face_id(axis, -1)
                bc = get(face_bc, (current_bid, fid), default_bc)
                (_topology_interblock_bc(bc) || _topology_periodic_bc(bc)) &&
                    push!(face_ids, fid)
            elseif value == current_dims[axis] + 1
                fid = _topology_face_id(axis, 1)
                bc = get(face_bc, (current_bid, fid), default_bc)
                (_topology_interblock_bc(bc) || _topology_periodic_bc(bc)) &&
                    push!(face_ids, fid)
            end
        end

        for fid in face_ids
            axis = _face_normal_axis(fid)
            side = isodd(fid) ? -1 : 1
            value = current_index[axis]
            outside = value < 1 || value > current_dims[axis] + 1
            boundary = value == 1 || value == current_dims[axis] + 1
            outside || boundary || continue

            bc = get(face_bc, (current_bid, fid), default_bc)
            if _topology_periodic_bc(bc)
                outside || continue
                coords = _ghost_mesh_coords(mesh_cache, current_bid)
                coords === nothing && return nothing
                shift_step = _periodic_node_shift(
                    coords, current_index, axis, side, current_dims,
                )
                wrapped = collect(current_index)
                wrapped[axis] += side < 0 ?
                    current_dims[axis] : -current_dims[axis]
                next_shift = (
                    shift[1] + shift_step[1],
                    shift[2] + shift_step[2],
                    shift[3] + shift_step[3],
                )
                push!(queue, (
                    current_bid, Tuple(wrapped), current_dims,
                    next_shift, depth + 1,
                ))
                continue
            end

            _topology_interblock_bc(bc) || continue
            haskey(connectivity, (current_bid, fid)) || throw(ArgumentError(
                "missing connectivity at topological ghost face " *
                "($current_bid,$fid)",
            ))
            conn = connectivity[(current_bid, fid)]
            nb_bid = conn.src_b
            nb_fid = conn.src_f
            nb_coords = _ghost_mesh_coords(mesh_cache, nb_bid)
            nb_coords === nothing && throw(ArgumentError(
                "missing coordinates for neighbor block $nb_bid",
            ))
            nb_dims = ntuple(axis -> size(nb_coords[axis], axis) - 1, 3)
            transform = _topological_connection_transform(fid, conn)
            destination_frame = structured_face_frame(fid)
            source_index = zeros(Int, 3)
            ghost_layer = if outside
                side < 0 ? 1 - value : value - current_dims[axis] - 1
            else
                0
            end
            source_normal_axis = structured_face_frame(nb_fid).normal_axis

            for destination_axis in 1:3
                source_value = transform.source_for_destination[destination_axis]
                source_axis = abs(source_value)
                if destination_axis == destination_frame.normal_axis
                    source_index[source_axis] = isodd(nb_fid) ?
                        ghost_layer + 1 : nb_dims[source_axis] + 1 - ghost_layer
                else
                    tangential_value = current_index[destination_axis]
                    source_index[source_axis] = source_value < 0 ?
                        nb_dims[source_axis] + 2 - tangential_value :
                        tangential_value
                end
            end
            source_index[source_normal_axis] > 0 || throw(DimensionMismatch(
                "invalid normal mapping at ($current_bid,$fid)",
            ))
            image_shift = hasproperty(conn, :image_translation) ?
                getproperty(conn, :image_translation) : (0.0, 0.0, 0.0)
            next_shift = (
                shift[1] + image_shift[1],
                shift[2] + image_shift[2],
                shift[3] + image_shift[3],
            )
            push!(queue, (
                nb_bid, Tuple(source_index), nb_dims,
                next_shift, depth + 1,
            ))
        end
    end

    isempty(candidates) && return nothing
    sort!(candidates, by = candidate -> (
        candidate[1], candidate[2][1], candidate[2][2], candidate[2][3],
        candidate[3][1], candidate[3][2], candidate[3][3],
    ))
    owner_bid, source, owner_shift = first(candidates)
    coords = _ghost_mesh_coords(mesh_cache, owner_bid)
    coords === nothing && return nothing
    return (
        coords[1][source...] + owner_shift[1],
        coords[2][source...] + owner_shift[2],
        coords[3][source...] + owner_shift[3],
    )
end

function _fill_topological_edge_corner_ghosts!(
    x, y, z, Nx, Ny, Nz, NG, face_bc, bid, connectivity;
    cell_offsets::NTuple{3,Int}=(0, 0, 0),
    block_dims::NTuple{3,Int}=(Nx, Ny, Nz),
    mesh_cache=nothing,
)
    local_dims = (Nx, Ny, Nz)
    singularity_edges = structured_metric_singularity_edges(
        bid, face_bc, connectivity,
    )
    for k in axes(x, 3), j in axes(x, 2), i in axes(x, 1)
        local_index = (i - NG, j - NG, k - NG)
        ghost_count = count(
            axis -> local_index[axis] < 1 ||
                    local_index[axis] > local_dims[axis] + 1,
            1:3,
        )
        # A node on an inter-block face can still be a *one-direction* ghost:
        # its second coordinate may be exactly on another block boundary
        # (for example, the z- ghost line on block 0's y- interface).  The
        # old ghost_count >= 2 guard left these face-edge nodes to the
        # single-face fill, so the two blocks constructed different
        # high-order tangential continuations.  Resolve such nodes through
        # the same connectivity traversal as true edge/corner ghosts.
        ghost_count == 0 && continue

        global_index = ntuple(
            axis -> local_index[axis] + cell_offsets[axis], 3,
        )
        global_sides = ntuple(
            axis -> global_index[axis] < 1 ? -1 :
                    (global_index[axis] > block_dims[axis] + 1 ? 1 : 0),
            3,
        )

        # Rank-internal edge/corner coordinates can be copied exactly from the
        # owning block mesh.  This avoids rebuilding a coupled mapping from a
        # local product formula merely because a rank cut crosses the edge.
        if all(side == 0 for side in global_sides)
            coords = _ghost_mesh_coords(mesh_cache, bid)
            if coords !== nothing && all(
                1 <= global_index[axis] <= block_dims[axis] + 1 for axis in 1:3
            )
                x[i,j,k] = coords[1][global_index...]
                y[i,j,k] = coords[2][global_index...]
                z[i,j,k] = coords[3][global_index...]
            end
            continue
        end

        face_ids = Int[]
        for axis in 1:3
            side = global_sides[axis]
            if side != 0
                push!(face_ids, _topology_face_id(axis, side))
            elseif global_index[axis] == 1
                fid = _topology_face_id(axis, -1)
                (_topology_interblock_bc(get(face_bc, (bid, fid),
                    isdefined(@__MODULE__, :BC_ISOTHERMAL_WALL) ?
                    BC_ISOTHERMAL_WALL : Int32(3))) ||
                 _topology_periodic_bc(get(face_bc, (bid, fid),
                    isdefined(@__MODULE__, :BC_ISOTHERMAL_WALL) ?
                    BC_ISOTHERMAL_WALL : Int32(3)))) && push!(face_ids, fid)
            elseif global_index[axis] == block_dims[axis] + 1
                fid = _topology_face_id(axis, 1)
                (_topology_interblock_bc(get(face_bc, (bid, fid),
                    isdefined(@__MODULE__, :BC_ISOTHERMAL_WALL) ?
                    BC_ISOTHERMAL_WALL : Int32(3))) ||
                 _topology_periodic_bc(get(face_bc, (bid, fid),
                    isdefined(@__MODULE__, :BC_ISOTHERMAL_WALL) ?
                    BC_ISOTHERMAL_WALL : Int32(3)))) && push!(face_ids, fid)
            end
        end
        bcs = map(fid -> get(face_bc, (bid, fid),
                              isdefined(@__MODULE__, :BC_ISOTHERMAL_WALL) ?
                              BC_ISOTHERMAL_WALL : Int32(3)), face_ids)
        all(_topology_interblock_bc(bc) || _topology_periodic_bc(bc)
            for bc in bcs) || continue
        any(_topology_interblock_bc(bc) for bc in bcs) || continue

        # A three-block edge has no single smooth tensor-product continuation:
        # traversing either incident face reaches a different valid interior
        # point. Keep the rotation-covariant product extension here and repair
        # the shared SCMM edge geometry from physical endpoints below.
        any(eachindex(singularity_edges)) do edge
            singularity_edges[edge] || return false
            face_a, face_b = STRUCTURED_BLOCK_EDGE_FACE_PAIRS[edge]
            return face_a in face_ids && face_b in face_ids
        end && continue

        resolved = _resolve_topological_node(
            bid, global_index, block_dims, face_bc, connectivity, mesh_cache,
        )
        resolved === nothing && throw(ArgumentError(
            "unable to resolve topological ghost node " *
            "block=$bid local=$(local_index) global=$(global_index)",
        ))
        x[i,j,k], y[i,j,k], z[i,j,k] = resolved
    end
    return nothing
end

@inline function _set_scmm_low_order_edge!(
    edge_yz, edge_zx, edge_xy, x, y, z, axis, i, j, k,
)
    ip = i + (axis == 1)
    jp = j + (axis == 2)
    kp = k + (axis == 3)
    half = eltype(edge_yz)(0.5)
    edge_yz[i,j,k] = half * (y[i,j,k] * z[ip,jp,kp] - y[ip,jp,kp] * z[i,j,k])
    edge_zx[i,j,k] = half * (z[i,j,k] * x[ip,jp,kp] - z[ip,jp,kp] * x[i,j,k])
    edge_xy[i,j,k] = half * (x[i,j,k] * y[ip,jp,kp] - x[ip,jp,kp] * y[i,j,k])
    return nothing
end

@inline function _structured_junction_patch_cell_range(
    ncell::Int, ng::Int, high::Int, width::Int,
)
    patch_width = min(width, ncell)
    return high == 0 ?
        ((ng + 1):(ng + patch_width)) :
        ((ng + ncell - patch_width + 1):(ng + ncell))
end

function _foreach_scmm_junction_patch_edge(
    callback, Nx, Ny, Nz, NG, singularity_edges;
    width::Int=STRUCTURED_SCMM_JUNCTION_WIDTH,
)
    singularity_edges === nothing && return nothing
    any(singularity_edges) || return nothing
    width > 0 || throw(ArgumentError("SCMM junction patch width must be positive"))
    dims = (Nx, Ny, Nz)
    all_cells = ntuple(axis -> (NG + 1):(NG + dims[axis]), 3)
    all_nodes = ntuple(axis -> (NG + 1):(NG + dims[axis] + 1), 3)

    for edge in eachindex(STRUCTURED_BLOCK_EDGE_FACE_PAIRS)
        edge <= length(singularity_edges) || break
        singularity_edges[edge] || continue
        free_axis = STRUCTURED_BLOCK_EDGE_AXES[edge]
        axis_a, high_a, axis_b, high_b = STRUCTURED_BLOCK_EDGE_PINS[edge]
        cells_a = _structured_junction_patch_cell_range(
            dims[axis_a], NG, high_a, width,
        )
        cells_b = _structured_junction_patch_cell_range(
            dims[axis_b], NG, high_b, width,
        )
        nodes_a = first(cells_a):(last(cells_a) + 1)
        nodes_b = first(cells_b):(last(cells_b) + 1)

        for edge_axis in (axis_a, axis_b, free_axis)
            ranges = ntuple(3) do axis
                if axis == edge_axis
                    axis == free_axis ? all_cells[axis] :
                        (axis == axis_a ? cells_a : cells_b)
                else
                    axis == free_axis ? all_nodes[axis] :
                        (axis == axis_a ? nodes_a : nodes_b)
                end
            end
            for k in ranges[3], j in ranges[2], i in ranges[1]
                callback(edge_axis, i, j, k)
            end
        end
    end
    return nothing
end

function _apply_scmm_junction_edge_fallback!(
    edge_groups, x, y, z, Nx, Ny, Nz, NG, singularity_edges;
    width::Int=STRUCTURED_SCMM_JUNCTION_WIDTH,
)
    # A physical face must not mix CMD6 edge products with endpoint products.
    # Replace all twelve oriented edges of every cell whose centered metric
    # stencil reaches the multi-member junction. The face vectors are still a
    # discrete curl, so GCL closure is retained while the patch represents the
    # actual node-defined polyhedra exactly. This is intentionally a localized
    # piecewise-linear fallback; expanding it would lower geometric order over
    # a larger part of an otherwise smooth block.
    _foreach_scmm_junction_patch_edge(
        Nx, Ny, Nz, NG, singularity_edges; width=width,
    ) do edge_axis, i, j, k
        edge_yz, edge_zx, edge_xy = edge_groups[edge_axis]
        _set_scmm_low_order_edge!(
            edge_yz, edge_zx, edge_xy, x, y, z,
            edge_axis, i, j, k,
        )
    end
    return nothing
end

# =============================================================================
# FVM Metric Computation
@inline function coordinate_less(a, b)
    if a[1] < b[1]
        return true
    elseif a[1] > b[1]
        return false
    end
    if a[2] < b[2]
        return true
    elseif a[2] > b[2]
        return false
    end
    return a[3] < b[3]
end

@inline function sort_4_points(p1, p2, p3, p4)
    a, b, c, d = p1, p2, p3, p4
    if !coordinate_less(a, b); t = a; a = b; b = t; end
    if !coordinate_less(c, d); t = c; c = d; d = t; end
    if !coordinate_less(a, c); t = a; a = c; c = t; end
    if !coordinate_less(b, d); t = b; b = d; d = t; end
    if !coordinate_less(b, c); t = b; b = c; c = t; end
    return a, b, c, d
end

@inline function compute_symmetric_face_metric(p1, p2, p3, p4, n_rough_x, n_rough_y, n_rough_z)
    q1, q2, q3, q4 = sort_4_points(p1, p2, p3, p4)
    u1_x, u1_y, u1_z = q4[1]-q1[1], q4[2]-q1[2], q4[3]-q1[3]
    u2_x, u2_y, u2_z = q3[1]-q2[1], q3[2]-q2[2], q3[3]-q2[3]
    n_x = 0.5 * (u1_y*u2_z - u1_z*u2_y)
    n_y = 0.5 * (u1_z*u2_x - u1_x*u2_z)
    n_z = 0.5 * (u1_x*u2_y - u1_y*u2_x)
    area = sqrt(n_x^2 + n_y^2 + n_z^2) + 1e-20
    dot_val = n_x*n_rough_x + n_y*n_rough_y + n_z*n_rough_z
    s = dot_val >= 0 ? 1.0 : -1.0
    return area, s*n_x/area, s*n_y/area, s*n_z/area
end

@inline function _periodic_metric_source_index(
    index, ncell, ng, periodic, is_normal,
)
    periodic || return index
    high_ghost_start = ncell + ng + (is_normal ? 2 : 1)
    if index <= ng
        return index + ncell
    elseif index >= high_ghost_start
        return index - ncell
    end
    return index
end

function _fill_periodic_metric_ghosts!(
    metric, Nx, Ny, Nz, NG, periodic, normal_dir,
)
    for k in axes(metric, 3), j in axes(metric, 2), i in axes(metric, 1)
        source_i = _periodic_metric_source_index(
            i, Nx, NG, periodic[1], normal_dir == 1,
        )
        source_j = _periodic_metric_source_index(
            j, Ny, NG, periodic[2], normal_dir == 2,
        )
        source_k = _periodic_metric_source_index(
            k, Nz, NG, periodic[3], normal_dir == 3,
        )
        if source_i != i || source_j != j || source_k != k
            metric[i,j,k] = metric[source_i,source_j,source_k]
        end
    end
    return metric
end

function _enforce_periodic_metric_ghosts!(
    Ai, nxi, nyi, nzi,
    Aj, nxj, nyj, nzj,
    Ak, nxk, nyk, nzk, V,
    Nx, Ny, Nz, NG, periodic,
)
    for metric in (Ai, nxi, nyi, nzi)
        _fill_periodic_metric_ghosts!(
            metric, Nx, Ny, Nz, NG, periodic, 1,
        )
    end
    for metric in (Aj, nxj, nyj, nzj)
        _fill_periodic_metric_ghosts!(
            metric, Nx, Ny, Nz, NG, periodic, 2,
        )
    end
    for metric in (Ak, nxk, nyk, nzk)
        _fill_periodic_metric_ghosts!(
            metric, Nx, Ny, Nz, NG, periodic, 3,
        )
    end
    _fill_periodic_metric_ghosts!(
        V, Nx, Ny, Nz, NG, periodic, 0,
    )
    return
end

# The SCMM metric products and CT vector-potential initialization must use the
# same discrete line geometry.  Keep the CMD6 operators at file scope so the
# two paths cannot silently drift apart.
@inline function structured_scmm_deriv_i(arr, i, j, k, N)
    T = eltype(arr)
    if i <= 2 || i >= N - 1
        i0 = clamp(i, 1, N)
        ip1 = clamp(i + 1, 1, N)
        return arr[ip1, j, k] - arr[i0, j, k]
    elseif i <= 4 || i >= N - 3
        im1 = clamp(i - 1, 1, N)
        i0 = clamp(i, 1, N)
        ip1 = clamp(i + 1, 1, N)
        ip2 = clamp(i + 2, 1, N)
        return (T(9) * (arr[ip1, j, k] - arr[i0, j, k]) / T(8)) -
               (arr[ip2, j, k] - arr[im1, j, k]) / T(24)
    else
        im2 = clamp(i - 2, 1, N)
        im1 = clamp(i - 1, 1, N)
        i0 = clamp(i, 1, N)
        ip1 = clamp(i + 1, 1, N)
        ip2 = clamp(i + 2, 1, N)
        ip3 = clamp(i + 3, 1, N)
        return (T(75) * (arr[ip1, j, k] - arr[i0, j, k]) / T(64)) -
               (T(25) * (arr[ip2, j, k] - arr[im1, j, k]) / T(384)) +
               (T(3) * (arr[ip3, j, k] - arr[im2, j, k]) / T(640))
    end
end

@inline function structured_scmm_deriv_j(arr, i, j, k, N)
    T = eltype(arr)
    if j <= 2 || j >= N - 1
        j0 = clamp(j, 1, N)
        jp1 = clamp(j + 1, 1, N)
        return arr[i, jp1, k] - arr[i, j0, k]
    elseif j <= 4 || j >= N - 3
        jm1 = clamp(j - 1, 1, N)
        j0 = clamp(j, 1, N)
        jp1 = clamp(j + 1, 1, N)
        jp2 = clamp(j + 2, 1, N)
        return (T(9) * (arr[i, jp1, k] - arr[i, j0, k]) / T(8)) -
               (arr[i, jp2, k] - arr[i, jm1, k]) / T(24)
    else
        jm2 = clamp(j - 2, 1, N)
        jm1 = clamp(j - 1, 1, N)
        j0 = clamp(j, 1, N)
        jp1 = clamp(j + 1, 1, N)
        jp2 = clamp(j + 2, 1, N)
        jp3 = clamp(j + 3, 1, N)
        return (T(75) * (arr[i, jp1, k] - arr[i, j0, k]) / T(64)) -
               (T(25) * (arr[i, jp2, k] - arr[i, jm1, k]) / T(384)) +
               (T(3) * (arr[i, jp3, k] - arr[i, jm2, k]) / T(640))
    end
end

@inline function structured_scmm_deriv_k(arr, i, j, k, N)
    T = eltype(arr)
    if k <= 2 || k >= N - 1
        k0 = clamp(k, 1, N)
        kp1 = clamp(k + 1, 1, N)
        return arr[i, j, kp1] - arr[i, j, k0]
    elseif k <= 4 || k >= N - 3
        km1 = clamp(k - 1, 1, N)
        k0 = clamp(k, 1, N)
        kp1 = clamp(k + 1, 1, N)
        kp2 = clamp(k + 2, 1, N)
        return (T(9) * (arr[i, j, kp1] - arr[i, j, k0]) / T(8)) -
               (arr[i, j, kp2] - arr[i, j, km1]) / T(24)
    else
        km2 = clamp(k - 2, 1, N)
        km1 = clamp(k - 1, 1, N)
        k0 = clamp(k, 1, N)
        kp1 = clamp(k + 1, 1, N)
        kp2 = clamp(k + 2, 1, N)
        kp3 = clamp(k + 3, 1, N)
        return (T(75) * (arr[i, j, kp1] - arr[i, j, k0]) / T(64)) -
               (T(25) * (arr[i, j, kp2] - arr[i, j, km1]) / T(384)) +
               (T(3) * (arr[i, j, kp3] - arr[i, j, km2]) / T(640))
    end
end

@inline function structured_scmm_interp_i(arr, i, j, k, N)
    T = eltype(arr)
    if i <= 2 || i >= N - 1
        i0 = clamp(i, 1, N)
        ip1 = clamp(i + 1, 1, N)
        return (arr[ip1, j, k] + arr[i0, j, k]) / T(2)
    elseif i <= 4 || i >= N - 3
        im1 = clamp(i - 1, 1, N)
        i0 = clamp(i, 1, N)
        ip1 = clamp(i + 1, 1, N)
        ip2 = clamp(i + 2, 1, N)
        return (T(9) * (arr[ip1, j, k] + arr[i0, j, k]) / T(16)) -
               (arr[ip2, j, k] + arr[im1, j, k]) / T(16)
    else
        im2 = clamp(i - 2, 1, N)
        im1 = clamp(i - 1, 1, N)
        i0 = clamp(i, 1, N)
        ip1 = clamp(i + 1, 1, N)
        ip2 = clamp(i + 2, 1, N)
        ip3 = clamp(i + 3, 1, N)
        return (T(75) * (arr[ip1, j, k] + arr[i0, j, k]) / T(128)) -
               (T(25) * (arr[ip2, j, k] + arr[im1, j, k]) / T(256)) +
               (T(3) * (arr[ip3, j, k] + arr[im2, j, k]) / T(256))
    end
end

@inline function structured_scmm_interp_j(arr, i, j, k, N)
    T = eltype(arr)
    if j <= 2 || j >= N - 1
        j0 = clamp(j, 1, N)
        jp1 = clamp(j + 1, 1, N)
        return (arr[i, jp1, k] + arr[i, j0, k]) / T(2)
    elseif j <= 4 || j >= N - 3
        jm1 = clamp(j - 1, 1, N)
        j0 = clamp(j, 1, N)
        jp1 = clamp(j + 1, 1, N)
        jp2 = clamp(j + 2, 1, N)
        return (T(9) * (arr[i, jp1, k] + arr[i, j0, k]) / T(16)) -
               (arr[i, jp2, k] + arr[i, jm1, k]) / T(16)
    else
        jm2 = clamp(j - 2, 1, N)
        jm1 = clamp(j - 1, 1, N)
        j0 = clamp(j, 1, N)
        jp1 = clamp(j + 1, 1, N)
        jp2 = clamp(j + 2, 1, N)
        jp3 = clamp(j + 3, 1, N)
        return (T(75) * (arr[i, jp1, k] + arr[i, j0, k]) / T(128)) -
               (T(25) * (arr[i, jp2, k] + arr[i, jm1, k]) / T(256)) +
               (T(3) * (arr[i, jp3, k] + arr[i, jm2, k]) / T(256))
    end
end

@inline function structured_scmm_interp_k(arr, i, j, k, N)
    T = eltype(arr)
    if k <= 2 || k >= N - 1
        k0 = clamp(k, 1, N)
        kp1 = clamp(k + 1, 1, N)
        return (arr[i, j, kp1] + arr[i, j, k0]) / T(2)
    elseif k <= 4 || k >= N - 3
        km1 = clamp(k - 1, 1, N)
        k0 = clamp(k, 1, N)
        kp1 = clamp(k + 1, 1, N)
        kp2 = clamp(k + 2, 1, N)
        return (T(9) * (arr[i, j, kp1] + arr[i, j, k0]) / T(16)) -
               (arr[i, j, kp2] + arr[i, j, km1]) / T(16)
    else
        km2 = clamp(k - 2, 1, N)
        km1 = clamp(k - 1, 1, N)
        k0 = clamp(k, 1, N)
        kp1 = clamp(k + 1, 1, N)
        kp2 = clamp(k + 2, 1, N)
        kp3 = clamp(k + 3, 1, N)
        return (T(75) * (arr[i, j, kp1] + arr[i, j, k0]) / T(128)) -
               (T(25) * (arr[i, j, kp2] + arr[i, j, km1]) / T(256)) +
               (T(3) * (arr[i, j, kp3] + arr[i, j, km2]) / T(256))
    end
end

@inline function _set_scmm_low_order_line_integral!(
    edge, ax, ay, az, x, y, z, axis, i, j, k,
)
    ip = i + (axis == 1)
    jp = j + (axis == 2)
    kp = k + (axis == 3)
    half = eltype(edge)(0.5)
    edge[i,j,k] = half * (
        (ax[i,j,k] + ax[ip,jp,kp]) * (x[ip,jp,kp] - x[i,j,k]) +
        (ay[i,j,k] + ay[ip,jp,kp]) * (y[ip,jp,kp] - y[i,j,k]) +
        (az[i,j,k] + az[ip,jp,kp]) * (z[ip,jp,kp] - z[i,j,k])
    )
    return nothing
end

function structured_scmm_edge_line_integrals(
    vector_potential, x, y, z;
    time=zero(eltype(x)), singularity_edges=nothing,
    physical_dims=nothing, ng::Int=0,
    junction_width::Int=STRUCTURED_SCMM_JUNCTION_WIDTH,
)
    size(x) == size(y) == size(z) || throw(DimensionMismatch(
        "SCMM edge geometry arrays must have identical sizes",
    ))
    ni, nj, nk = size(x)
    T = promote_type(eltype(x), typeof(time))
    ax = Array{T}(undef, ni, nj, nk)
    ay = similar(ax)
    az = similar(ax)
    @inbounds for k in 1:nk, j in 1:nj, i in 1:ni
        value = vector_potential(x[i,j,k], y[i,j,k], z[i,j,k], time)
        ax[i,j,k] = T(value[1])
        ay[i,j,k] = T(value[2])
        az[i,j,k] = T(value[3])
    end

    edge_x = Array{T,3}(undef, ni - 1, nj, nk)
    edge_y = Array{T,3}(undef, ni, nj - 1, nk)
    edge_z = Array{T,3}(undef, ni, nj, nk - 1)
    @inbounds for k in 1:nk, j in 1:nj, i in 1:ni-1
        edge_x[i,j,k] =
            structured_scmm_interp_i(ax, i, j, k, ni) *
                structured_scmm_deriv_i(x, i, j, k, ni) +
            structured_scmm_interp_i(ay, i, j, k, ni) *
                structured_scmm_deriv_i(y, i, j, k, ni) +
            structured_scmm_interp_i(az, i, j, k, ni) *
                structured_scmm_deriv_i(z, i, j, k, ni)
    end
    @inbounds for k in 1:nk, j in 1:nj-1, i in 1:ni
        edge_y[i,j,k] =
            structured_scmm_interp_j(ax, i, j, k, nj) *
                structured_scmm_deriv_j(x, i, j, k, nj) +
            structured_scmm_interp_j(ay, i, j, k, nj) *
                structured_scmm_deriv_j(y, i, j, k, nj) +
            structured_scmm_interp_j(az, i, j, k, nj) *
                structured_scmm_deriv_j(z, i, j, k, nj)
    end
    @inbounds for k in 1:nk-1, j in 1:nj, i in 1:ni
        edge_z[i,j,k] =
            structured_scmm_interp_k(ax, i, j, k, nk) *
                structured_scmm_deriv_k(x, i, j, k, nk) +
            structured_scmm_interp_k(ay, i, j, k, nk) *
                structured_scmm_deriv_k(y, i, j, k, nk) +
            structured_scmm_interp_k(az, i, j, k, nk) *
                structured_scmm_deriv_k(z, i, j, k, nk)
    end

    if singularity_edges !== nothing && any(singularity_edges)
        physical_dims === nothing && throw(ArgumentError(
            "junction-aware SCMM edge integrals require physical_dims",
        ))
        dims = Tuple(Int.(physical_dims))
        length(dims) == 3 || throw(DimensionMismatch(
            "physical_dims must contain three cell counts",
        ))
        expected_nodes = ntuple(axis -> dims[axis] + 2ng + 1, 3)
        size(x) == expected_nodes || throw(DimensionMismatch(
            "junction-aware SCMM coordinates have size $(size(x)); " *
            "expected $expected_nodes for physical_dims=$dims and ng=$ng",
        ))
        edges = (edge_x, edge_y, edge_z)
        _foreach_scmm_junction_patch_edge(
            dims..., ng, singularity_edges; width=junction_width,
        ) do edge_axis, i, j, k
            _set_scmm_low_order_line_integral!(
                edges[edge_axis], ax, ay, az, x, y, z,
                edge_axis, i, j, k,
            )
        end
    end
    return edge_x, edge_y, edge_z
end

function _compute_fvm_metrics_runtime_monolithic(
    x, y, z, Nx::Int, Ny::Int, Nz::Int, NG::Int;
    periodic=(false, false, false),
    singularity_edges=nothing,
)
    Nx_nodes_tot = Nx + 2NG + 1
    Ny_nodes_tot = Ny + 2NG + 1
    Nz_nodes_tot = Nz + 2NG + 1
    Nx_cells_tot = Nx + 2NG
    Ny_cells_tot = Ny + 2NG
    Nz_cells_tot = Nz + 2NG
    
    Ai = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    nxi = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    nyi = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    nzi = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    
    Aj = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    nxj = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    nyj = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    nzj = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    
    Ak = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    nxk = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    nyk = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    nzk = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    
    V = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_cells_tot)

    @inline function validated_face_metric(area, Sx, Sy, Sz, direction, i, j, k, active)
        if !isfinite(area) || area <= zero(FT)
            if active
                throw(DomainError(
                    area,
                    "degenerate $(direction)-face at metric index ($i, $j, $k): " *
                    "face area must be finite and strictly positive",
                ))
            end
            return zero(FT), zero(FT), zero(FT), zero(FT)
        end
        return area, Sx / area, Sy / area, Sz / area
    end
    
    # ── SCMM Intermediate Edge Arrays ──
    y_dz_k = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_cells_tot)
    z_dx_k = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_cells_tot)
    x_dy_k = zeros(FT, Nx_nodes_tot, Ny_nodes_tot, Nz_cells_tot)
    
    y_dz_j = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_nodes_tot)
    z_dx_j = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_nodes_tot)
    x_dy_j = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_nodes_tot)
    
    y_dz_i = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_nodes_tot)
    z_dx_i = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_nodes_tot)
    x_dy_i = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_nodes_tot)
    
    # ── CMD6 Midpoint Derivative Operators ──
    @inline deriv_i(arr, i, j, k, N) =
        structured_scmm_deriv_i(arr, i, j, k, N)

    @inline deriv_j(arr, i, j, k, N) =
        structured_scmm_deriv_j(arr, i, j, k, N)

    @inline deriv_k(arr, i, j, k, N) =
        structured_scmm_deriv_k(arr, i, j, k, N)

    # ── CMD6 Midpoint Interpolation Operators ──
    @inline interp_i(arr, i, j, k, N) =
        structured_scmm_interp_i(arr, i, j, k, N)

    @inline interp_j(arr, i, j, k, N) =
        structured_scmm_interp_j(arr, i, j, k, N)

    @inline interp_k(arr, i, j, k, N) =
        structured_scmm_interp_k(arr, i, j, k, N)

    # ── CMD6 Face Interpolation Helpers ──
    @inline function interp_face_i(arr, i, j, k)
        v1 = interp_j(arr, i, j, clamp(k-2, 1, Nz_nodes_tot), Ny_nodes_tot)
        v2 = interp_j(arr, i, j, clamp(k-1, 1, Nz_nodes_tot), Ny_nodes_tot)
        v3 = interp_j(arr, i, j, clamp(k,   1, Nz_nodes_tot), Ny_nodes_tot)
        v4 = interp_j(arr, i, j, clamp(k+1, 1, Nz_nodes_tot), Ny_nodes_tot)
        v5 = interp_j(arr, i, j, clamp(k+2, 1, Nz_nodes_tot), Ny_nodes_tot)
        v6 = interp_j(arr, i, j, clamp(k+3, 1, Nz_nodes_tot), Ny_nodes_tot)
        return (75.0 * (v4 + v3) / 128.0) - (25.0 * (v5 + v2) / 256.0) + (3.0 * (v6 + v1) / 256.0)
    end

    @inline function interp_face_j(arr, i, j, k)
        v1 = interp_i(arr, i, j, clamp(k-2, 1, Nz_nodes_tot), Nx_nodes_tot)
        v2 = interp_i(arr, i, j, clamp(k-1, 1, Nz_nodes_tot), Nx_nodes_tot)
        v3 = interp_i(arr, i, j, clamp(k,   1, Nz_nodes_tot), Nx_nodes_tot)
        v4 = interp_i(arr, i, j, clamp(k+1, 1, Nz_nodes_tot), Nx_nodes_tot)
        v5 = interp_i(arr, i, j, clamp(k+2, 1, Nz_nodes_tot), Nx_nodes_tot)
        v6 = interp_i(arr, i, j, clamp(k+3, 1, Nz_nodes_tot), Nx_nodes_tot)
        return (75.0 * (v4 + v3) / 128.0) - (25.0 * (v5 + v2) / 256.0) + (3.0 * (v6 + v1) / 256.0)
    end

    @inline function interp_face_k(arr, i, j, k)
        v1 = interp_i(arr, i, clamp(j-2, 1, Ny_nodes_tot), k, Nx_nodes_tot)
        v2 = interp_i(arr, i, clamp(j-1, 1, Ny_nodes_tot), k, Nx_nodes_tot)
        v3 = interp_i(arr, i, clamp(j,   1, Ny_nodes_tot), k, Nx_nodes_tot)
        v4 = interp_i(arr, i, clamp(j+1, 1, Ny_nodes_tot), k, Nx_nodes_tot)
        v5 = interp_i(arr, i, clamp(j+2, 1, Ny_nodes_tot), k, Nx_nodes_tot)
        v6 = interp_i(arr, i, clamp(j+3, 1, Ny_nodes_tot), k, Nx_nodes_tot)
        return (75.0 * (v4 + v3) / 128.0) - (25.0 * (v5 + v2) / 256.0) + (3.0 * (v6 + v1) / 256.0)
    end

    # --- Pre-compute SCMM intermediate edge products ---
    # a) k-face/edge intermediates (midpoint in k, nodes in i and j)
    for k in 1:Nz_cells_tot, j in 1:Ny_nodes_tot, i in 1:Nx_nodes_tot
        y_dz_k[i, j, k] = FT(0.5) * (
            interp_k(y, i, j, k, Nz_nodes_tot) * deriv_k(z, i, j, k, Nz_nodes_tot) -
            interp_k(z, i, j, k, Nz_nodes_tot) * deriv_k(y, i, j, k, Nz_nodes_tot)
        )
        z_dx_k[i, j, k] = FT(0.5) * (
            interp_k(z, i, j, k, Nz_nodes_tot) * deriv_k(x, i, j, k, Nz_nodes_tot) -
            interp_k(x, i, j, k, Nz_nodes_tot) * deriv_k(z, i, j, k, Nz_nodes_tot)
        )
        x_dy_k[i, j, k] = FT(0.5) * (
            interp_k(x, i, j, k, Nz_nodes_tot) * deriv_k(y, i, j, k, Nz_nodes_tot) -
            interp_k(y, i, j, k, Nz_nodes_tot) * deriv_k(x, i, j, k, Nz_nodes_tot)
        )
    end
    
    # b) j-face/edge intermediates (midpoint in j, nodes in i and k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_cells_tot, i in 1:Nx_nodes_tot
        y_dz_j[i, j, k] = FT(0.5) * (
            interp_j(y, i, j, k, Ny_nodes_tot) * deriv_j(z, i, j, k, Ny_nodes_tot) -
            interp_j(z, i, j, k, Ny_nodes_tot) * deriv_j(y, i, j, k, Ny_nodes_tot)
        )
        z_dx_j[i, j, k] = FT(0.5) * (
            interp_j(z, i, j, k, Ny_nodes_tot) * deriv_j(x, i, j, k, Ny_nodes_tot) -
            interp_j(x, i, j, k, Ny_nodes_tot) * deriv_j(z, i, j, k, Ny_nodes_tot)
        )
        x_dy_j[i, j, k] = FT(0.5) * (
            interp_j(x, i, j, k, Ny_nodes_tot) * deriv_j(y, i, j, k, Ny_nodes_tot) -
            interp_j(y, i, j, k, Ny_nodes_tot) * deriv_j(x, i, j, k, Ny_nodes_tot)
        )
    end
    
    # c) i-face/edge intermediates (midpoint in i, nodes in j and k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_nodes_tot, i in 1:Nx_cells_tot
        y_dz_i[i, j, k] = FT(0.5) * (
            interp_i(y, i, j, k, Nx_nodes_tot) * deriv_i(z, i, j, k, Nx_nodes_tot) -
            interp_i(z, i, j, k, Nx_nodes_tot) * deriv_i(y, i, j, k, Nx_nodes_tot)
        )
        z_dx_i[i, j, k] = FT(0.5) * (
            interp_i(z, i, j, k, Nx_nodes_tot) * deriv_i(x, i, j, k, Nx_nodes_tot) -
            interp_i(x, i, j, k, Nx_nodes_tot) * deriv_i(z, i, j, k, Nx_nodes_tot)
        )
        x_dy_i[i, j, k] = FT(0.5) * (
            interp_i(x, i, j, k, Nx_nodes_tot) * deriv_i(y, i, j, k, Nx_nodes_tot) -
            interp_i(y, i, j, k, Nx_nodes_tot) * deriv_i(x, i, j, k, Nx_nodes_tot)
        )
    end

    _apply_scmm_junction_edge_fallback!(
        (
            (y_dz_i, z_dx_i, x_dy_i),
            (y_dz_j, z_dx_j, x_dy_j),
            (y_dz_k, z_dx_k, x_dy_k),
        ),
        x, y, z, Nx, Ny, Nz, NG, singularity_edges,
    )

    # 1. Compute i-face metrics (face center: i, j+1/2, k+1/2)
    for k in 1:Nz_cells_tot, j in 1:Ny_cells_tot, i in 1:Nx_nodes_tot
        Sx = (y_dz_k[i, j+1, k] - y_dz_k[i, j, k]) - (y_dz_j[i, j, k+1] - y_dz_j[i, j, k])
        Sy = (z_dx_k[i, j+1, k] - z_dx_k[i, j, k]) - (z_dx_j[i, j, k+1] - z_dx_j[i, j, k])
        Sz = (x_dy_k[i, j+1, k] - x_dy_k[i, j, k]) - (x_dy_j[i, j, k+1] - x_dy_j[i, j, k])
        
        Ai[i,j,k], nxi[i,j,k], nyi[i,j,k], nzi[i,j,k] = validated_face_metric(
            sqrt(Sx^2 + Sy^2 + Sz^2), Sx, Sy, Sz, "i", i, j, k,
            NG+1 <= i <= Nx+NG+1 && NG+1 <= j <= Ny+NG && NG+1 <= k <= Nz+NG,
        )
    end
    
    # 2. Compute j-face metrics (face center: i+1/2, j, k+1/2)
    for k in 1:Nz_cells_tot, j in 1:Ny_nodes_tot, i in 1:Nx_cells_tot
        Sx = (y_dz_i[i, j, k+1] - y_dz_i[i, j, k]) - (y_dz_k[i+1, j, k] - y_dz_k[i, j, k])
        Sy = (z_dx_i[i, j, k+1] - z_dx_i[i, j, k]) - (z_dx_k[i+1, j, k] - z_dx_k[i, j, k])
        Sz = (x_dy_i[i, j, k+1] - x_dy_i[i, j, k]) - (x_dy_k[i+1, j, k] - x_dy_k[i, j, k])
        
        Aj[i,j,k], nxj[i,j,k], nyj[i,j,k], nzj[i,j,k] = validated_face_metric(
            sqrt(Sx^2 + Sy^2 + Sz^2), Sx, Sy, Sz, "j", i, j, k,
            NG+1 <= i <= Nx+NG && NG+1 <= j <= Ny+NG+1 && NG+1 <= k <= Nz+NG,
        )
    end
    
    # 3. Compute k-face metrics (face center: i+1/2, j+1/2, k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_cells_tot, i in 1:Nx_cells_tot
        Sx = (y_dz_j[i+1, j, k] - y_dz_j[i, j, k]) - (y_dz_i[i, j+1, k] - y_dz_i[i, j, k])
        Sy = (z_dx_j[i+1, j, k] - z_dx_j[i, j, k]) - (z_dx_i[i, j+1, k] - z_dx_i[i, j, k])
        Sz = (x_dy_j[i+1, j, k] - x_dy_j[i, j, k]) - (x_dy_i[i, j+1, k] - x_dy_i[i, j, k])
        
        Ak[i,j,k], nxk[i,j,k], nyk[i,j,k], nzk[i,j,k] = validated_face_metric(
            sqrt(Sx^2 + Sy^2 + Sz^2), Sx, Sy, Sz, "k", i, j, k,
            NG+1 <= i <= Nx+NG && NG+1 <= j <= Ny+NG && NG+1 <= k <= Nz+NG+1,
        )
    end
    
    # 4. Compute GCL-consistent cell volumes using Divergence Theorem
    for k in 1:Nz_cells_tot, j in 1:Ny_cells_tot, i in 1:Nx_cells_tot
        xf_i_lo = interp_face_i(x, i, j, k)
        yf_i_lo = interp_face_i(y, i, j, k)
        zf_i_lo = interp_face_i(z, i, j, k)
        dot_i_lo = xf_i_lo * (Ai[i,j,k]*nxi[i,j,k]) + yf_i_lo * (Ai[i,j,k]*nyi[i,j,k]) + zf_i_lo * (Ai[i,j,k]*nzi[i,j,k])
        
        xf_i_hi = interp_face_i(x, i+1, j, k)
        yf_i_hi = interp_face_i(y, i+1, j, k)
        zf_i_hi = interp_face_i(z, i+1, j, k)
        dot_i_hi = xf_i_hi * (Ai[i+1,j,k]*nxi[i+1,j,k]) + yf_i_hi * (Ai[i+1,j,k]*nyi[i+1,j,k]) + zf_i_hi * (Ai[i+1,j,k]*nzi[i+1,j,k])
        
        xf_j_lo = interp_face_j(x, i, j, k)
        yf_j_lo = interp_face_j(y, i, j, k)
        zf_j_lo = interp_face_j(z, i, j, k)
        dot_j_lo = xf_j_lo * (Aj[i,j,k]*nxj[i,j,k]) + yf_j_lo * (Aj[i,j,k]*nyj[i,j,k]) + zf_j_lo * (Aj[i,j,k]*nzj[i,j,k])
        
        xf_j_hi = interp_face_j(x, i, j+1, k)
        yf_j_hi = interp_face_j(y, i, j+1, k)
        zf_j_hi = interp_face_j(z, i, j+1, k)
        dot_j_hi = xf_j_hi * (Aj[i,j+1,k]*nxj[i,j+1,k]) + yf_j_hi * (Aj[i,j+1,k]*nyj[i,j+1,k]) + zf_j_hi * (Aj[i,j+1,k]*nzj[i,j+1,k])
        
        xf_k_lo = interp_face_k(x, i, j, k)
        yf_k_lo = interp_face_k(y, i, j, k)
        zf_k_lo = interp_face_k(z, i, j, k)
        dot_k_lo = xf_k_lo * (Ak[i,j,k]*nxk[i,j,k]) + yf_k_lo * (Ak[i,j,k]*nyk[i,j,k]) + zf_k_lo * (Ak[i,j,k]*nzk[i,j,k])
        
        xf_k_hi = interp_face_k(x, i, j, k+1)
        yf_k_hi = interp_face_k(y, i, j, k+1)
        zf_k_hi = interp_face_k(z, i, j, k+1)
        dot_k_hi = xf_k_hi * (Ak[i,j,k+1]*nxk[i,j,k+1]) + yf_k_hi * (Ak[i,j,k+1]*nyk[i,j,k+1]) + zf_k_hi * (Ak[i,j,k+1]*nzk[i,j,k+1])
        
        vol = (dot_i_hi - dot_i_lo + dot_j_hi - dot_j_lo + dot_k_hi - dot_k_lo) / FT(3)
        active_cell = NG+1 <= i <= Nx+NG && NG+1 <= j <= Ny+NG && NG+1 <= k <= Nz+NG
        if active_cell && (!isfinite(vol) || vol <= zero(FT))
            throw(DomainError(
                vol,
                "inverted or degenerate cell at metric index ($i, $j, $k): " *
                "signed volume must be finite and strictly positive",
            ))
        end
        # Ghost-cell metrics still participate in boundary/interface stencils;
        # retain a positive inverse volume there while validating only physical
        # cells for orientation.  Physical cells use the signed volume above.
        V[i,j,k] = active_cell ? inv(vol) :
            (isfinite(vol) && abs(vol) > eps(FT) ? inv(abs(vol)) : zero(FT))
    end
    
    _enforce_periodic_metric_ghosts!(
        Ai, nxi, nyi, nzi,
        Aj, nxj, nyj, nzj,
        Ak, nxk, nyk, nzk, V,
        Nx, Ny, Nz, NG, periodic,
    )
    return Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V
end

struct SCMMEdgeWorkspace{A}
    edge_i::NTuple{3,A}
    edge_j::NTuple{3,A}
    edge_k::NTuple{3,A}
end

struct SCMMFaceEdgePayload{A}
    edge_u::A
    delta_u::A
    edge_v::A
    delta_v::A
    nodes::A
end

@inline Base.getindex(workspace::SCMMEdgeWorkspace, axis::Integer) =
    axis == 1 ? workspace.edge_i :
    axis == 2 ? workspace.edge_j :
    axis == 3 ? workspace.edge_k :
    throw(BoundsError(workspace, axis))

@inline function _structured_scmm_axis_derivative(
    array, axis::Integer, index::NTuple{3,Int}, node_dims::NTuple{3,Int},
)
    i, j, k = index
    return axis == 1 ? structured_scmm_deriv_i(array, i, j, k, node_dims[1]) :
           axis == 2 ? structured_scmm_deriv_j(array, i, j, k, node_dims[2]) :
           axis == 3 ? structured_scmm_deriv_k(array, i, j, k, node_dims[3]) :
           throw(ArgumentError("SCMM edge axis must be in 1:3, got $axis"))
end

@inline function _structured_scmm_face_edge_index(
    normal_axis::Integer, normal_node::Int,
    edge_axis::Integer, edge_cell::Int,
    other_axis::Integer, other_node::Int,
)
    return ntuple(3) do axis
        axis == normal_axis ? normal_node :
        axis == edge_axis ? edge_cell :
        axis == other_axis ? other_node :
        throw(ArgumentError("invalid SCMM face edge axes"))
    end
end

function pack_scmm_face_edge_payload(
    workspace::SCMMEdgeWorkspace, coordinates::MetricCoordinates,
    fid::Int, Nx::Int, Ny::Int, Nz::Int, NG::Int;
    u_s::Int=1, u_e::Int=-1, v_s::Int=1, v_e::Int=-1,
)
    frame = structured_face_frame(fid)
    local_dims = (Nx, Ny, Nz)
    node_dims = size(coordinates.x)
    u_count = local_dims[frame.u_axis]
    v_count = local_dims[frame.v_axis]
    u_e = u_e < 1 ? u_count : u_e
    v_e = v_e < 1 ? v_count : v_e
    1 <= u_s <= u_e <= u_count || throw(BoundsError(1:u_count, u_s:u_e))
    1 <= v_s <= v_e <= v_count || throw(BoundsError(1:v_count, v_s:v_e))
    u_length = u_e - u_s + 1
    v_length = v_e - v_s + 1
    normal_node = isodd(fid) ? NG + 1 :
        local_dims[frame.normal_axis] + NG + 1
    T = eltype(coordinates.x)
    edge_u = zeros(T, u_length, v_length + 1, 3)
    delta_u = similar(edge_u)
    edge_v = zeros(T, u_length + 1, v_length, 3)
    delta_v = similar(edge_v)
    nodes = zeros(T, u_length + 1, v_length + 1, 3)
    coordinate_arrays = (coordinates.x, coordinates.y, coordinates.z)

    for v_node in 1:v_length+1, u_node in 1:u_length+1
        index = ntuple(3) do axis
            axis == frame.normal_axis ? normal_node :
            axis == frame.u_axis ? NG + u_s + u_node - 1 :
            axis == frame.v_axis ? NG + v_s + v_node - 1 :
            throw(ArgumentError("invalid SCMM face-node axis"))
        end
        for component in 1:3
            nodes[u_node,v_node,component] =
                coordinate_arrays[component][index...]
        end
    end

    for v_node in 1:v_length+1, u_cell in 1:u_length
        index = _structured_scmm_face_edge_index(
            frame.normal_axis, normal_node,
            frame.u_axis, NG + u_s + u_cell - 1,
            frame.v_axis, NG + v_s + v_node - 1,
        )
        for component in 1:3
            edge_u[u_cell,v_node,component] =
                workspace[frame.u_axis][component][index...]
            delta_u[u_cell,v_node,component] =
                _structured_scmm_axis_derivative(
                    coordinate_arrays[component], frame.u_axis,
                    index, node_dims,
                )
        end
    end
    for v_cell in 1:v_length, u_node in 1:u_length+1
        index = _structured_scmm_face_edge_index(
            frame.normal_axis, normal_node,
            frame.v_axis, NG + v_s + v_cell - 1,
            frame.u_axis, NG + u_s + u_node - 1,
        )
        for component in 1:3
            edge_v[u_node,v_cell,component] =
                workspace[frame.v_axis][component][index...]
            delta_v[u_node,v_cell,component] =
                _structured_scmm_axis_derivative(
                    coordinate_arrays[component], frame.v_axis,
                    index, node_dims,
                )
        end
    end
    return SCMMFaceEdgePayload(edge_u, delta_u, edge_v, delta_v, nodes)
end

function scmm_face_edge_payload_vector(payload::SCMMFaceEdgePayload)
    arrays = (
        payload.edge_u, payload.delta_u, payload.edge_v, payload.delta_v,
        payload.nodes,
    )
    result = Vector{eltype(payload.edge_u)}(undef, sum(length, arrays))
    offset = 0
    for array in arrays
        copyto!(result, offset + 1, vec(array), 1, length(array))
        offset += length(array)
    end
    return result
end

function scmm_face_edge_payload_from_vector(
    values::AbstractVector{T}, u_count::Int, v_count::Int,
) where {T<:AbstractFloat}
    shapes = (
        (u_count, v_count + 1, 3),
        (u_count, v_count + 1, 3),
        (u_count + 1, v_count, 3),
        (u_count + 1, v_count, 3),
        (u_count + 1, v_count + 1, 3),
    )
    expected = sum(prod, shapes)
    length(values) == expected || throw(DimensionMismatch(
        "SCMM face-edge payload has $(length(values)) values; expected $expected",
    ))
    arrays = Vector{Array{T,3}}(undef, 5)
    offset = 0
    for index in eachindex(shapes)
        count = prod(shapes[index])
        arrays[index] = reshape(copy(view(values, offset+1:offset+count)), shapes[index])
        offset += count
    end
    return SCMMFaceEdgePayload(arrays...)
end

@inline function _mapped_scmm_payload_node(
    payload::SCMMFaceEdgePayload,
    destination_u_node::Int, destination_v_node::Int,
    destination_frame::StructuredFaceFrame,
    source_frame::StructuredFaceFrame,
    transform::StructuredFaceTransform,
)
    source_counts = zeros(Int, 3)
    source_counts[source_frame.u_axis] = size(payload.nodes, 1) - 1
    source_counts[source_frame.v_axis] = size(payload.nodes, 2) - 1
    source_nodes = zeros(Int, 3)
    for (destination_axis, destination_node) in (
        (destination_frame.u_axis, destination_u_node),
        (destination_frame.v_axis, destination_v_node),
    )
        mapped_axis = transform.source_for_destination[destination_axis]
        source_axis = abs(mapped_axis)
        source_nodes[source_axis] = mapped_axis < 0 ?
            source_counts[source_axis] + 2 - destination_node :
            destination_node
    end
    return ntuple(
        component -> payload.nodes[
            source_nodes[source_frame.u_axis],
            source_nodes[source_frame.v_axis], component,
        ],
        3,
    )
end

function validate_canonical_scmm_face_nodes!(
    coordinates::MetricCoordinates, payload::SCMMFaceEdgePayload,
    fid::Int, conn, Nx::Int, Ny::Int, Nz::Int, NG::Int;
    u_s::Int=1, u_e::Int=-1, v_s::Int=1, v_e::Int=-1,
)
    destination = structured_face_frame(fid)
    source = structured_face_frame(conn.src_f)
    transform = _structured_metric_connection_transform(fid, conn)
    local_dims = (Nx, Ny, Nz)
    u_count = local_dims[destination.u_axis]
    v_count = local_dims[destination.v_axis]
    u_e = u_e < 1 ? u_count : u_e
    v_e = v_e < 1 ? v_count : v_e
    u_length = u_e - u_s + 1
    v_length = v_e - v_s + 1
    normal_node = isodd(fid) ? NG + 1 :
        local_dims[destination.normal_axis] + NG + 1
    candidate_translation = hasproperty(conn, :image_translation) ?
        getproperty(conn, :image_translation) : nothing
    translation = candidate_translation === nothing ?
        (0.0, 0.0, 0.0) : candidate_translation
    coordinate_arrays = (coordinates.x, coordinates.y, coordinates.z)
    T = eltype(coordinates.x)

    for v_node in 1:v_length+1, u_node in 1:u_length+1
        index = ntuple(3) do axis
            axis == destination.normal_axis ? normal_node :
            axis == destination.u_axis ? NG + u_s + u_node - 1 :
            axis == destination.v_axis ? NG + v_s + v_node - 1 :
            throw(ArgumentError("invalid SCMM interface-node axis"))
        end
        local_node = ntuple(
            component -> coordinate_arrays[component][index...], 3,
        )
        source_node = _mapped_scmm_payload_node(
            payload, u_node, v_node, destination, source, transform,
        )
        mapped_node = ntuple(
            component -> source_node[component] + T(translation[component]), 3,
        )
        scale = max(
            one(T), maximum(abs, local_node), maximum(abs, mapped_node),
        )
        tolerance = T(512) * eps(T) * scale
        mismatch = maximum(abs(local_node[c] - mapped_node[c]) for c in 1:3)
        mismatch <= tolerance || throw(ArgumentError(
            "nonconformal structured interface at block face $fid node " *
            "($u_node,$v_node): mapped coordinate mismatch=$mismatch " *
            "exceeds tolerance=$tolerance",
        ))
    end
    return nothing
end

@inline function _structured_metric_connection_transform(fid, conn)
    if hasproperty(conn, :transform) && getproperty(conn, :transform) !== nothing
        candidate = getproperty(conn, :transform)
        return candidate.source_face == conn.src_f &&
               candidate.destination_face == fid ?
            candidate : structured_inverse_face_transform(candidate)
    end
    return structured_legacy_face_transform(conn.src_f, fid, conn.reverse_tan)
end

@inline function _mapped_scmm_payload_edge(
    payload::SCMMFaceEdgePayload,
    destination_axis::Integer, destination_cell::Int,
    destination_other_axis::Integer, destination_other_node::Int,
    destination_frame::StructuredFaceFrame,
    source_frame::StructuredFaceFrame,
    transform::StructuredFaceTransform,
)
    source_u_count = size(payload.edge_u, 1)
    source_v_count = size(payload.edge_v, 2)
    source_cells = zeros(Int, 3)
    source_cells[source_frame.u_axis] = source_u_count
    source_cells[source_frame.v_axis] = source_v_count
    mapped_axis = transform.source_for_destination[destination_axis]
    mapped_other_axis = transform.source_for_destination[destination_other_axis]
    source_axis = abs(mapped_axis)
    source_other_axis = abs(mapped_other_axis)
    source_cell = mapped_axis < 0 ?
        source_cells[source_axis] - destination_cell + 1 : destination_cell
    source_other_node = mapped_other_axis < 0 ?
        source_cells[source_other_axis] + 2 - destination_other_node :
        destination_other_node
    orientation = mapped_axis < 0 ? -1 : 1
    if source_axis == source_frame.u_axis
        return (
            ntuple(component -> orientation *
                payload.edge_u[source_cell,source_other_node,component], 3),
            ntuple(component -> orientation *
                payload.delta_u[source_cell,source_other_node,component], 3),
        )
    elseif source_axis == source_frame.v_axis
        return (
            ntuple(component -> orientation *
                payload.edge_v[source_other_node,source_cell,component], 3),
            ntuple(component -> orientation *
                payload.delta_v[source_other_node,source_cell,component], 3),
        )
    end
    throw(ArgumentError(
        "interface transform maps destination edge outside source face",
    ))
end

@inline function _translated_scmm_edge(edge, delta, translation, ::Type{T}) where {T}
    ax, ay, az = T.(translation)
    dx, dy, dz = delta
    return (
        edge[1] + T(0.5) * (ay * dz - az * dy),
        edge[2] + T(0.5) * (az * dx - ax * dz),
        edge[3] + T(0.5) * (ax * dy - ay * dx),
    )
end

function unpack_canonical_scmm_face_edges!(
    workspace::SCMMEdgeWorkspace, coordinates::MetricCoordinates,
    payload::SCMMFaceEdgePayload,
    fid::Int, conn, Nx::Int, Ny::Int, Nz::Int, NG::Int;
    u_s::Int=1, u_e::Int=-1, v_s::Int=1, v_e::Int=-1,
)
    destination = structured_face_frame(fid)
    source = structured_face_frame(conn.src_f)
    transform = _structured_metric_connection_transform(fid, conn)
    local_dims = (Nx, Ny, Nz)
    u_count = local_dims[destination.u_axis]
    v_count = local_dims[destination.v_axis]
    u_e = u_e < 1 ? u_count : u_e
    v_e = v_e < 1 ? v_count : v_e
    u_length = u_e - u_s + 1
    v_length = v_e - v_s + 1
    normal_node = isodd(fid) ? NG + 1 :
        local_dims[destination.normal_axis] + NG + 1
    translation = hasproperty(conn, :image_translation) ?
        getproperty(conn, :image_translation) : (0.0, 0.0, 0.0)
    T = eltype(payload.edge_u)

    validate_canonical_scmm_face_nodes!(
        coordinates, payload, fid, conn, Nx, Ny, Nz, NG;
        u_s=u_s, u_e=u_e, v_s=v_s, v_e=v_e,
    )

    for v_node in 1:v_length+1, u_cell in 1:u_length
        edge, delta = _mapped_scmm_payload_edge(
            payload, destination.u_axis, u_cell,
            destination.v_axis, v_node,
            destination, source, transform,
        )
        canonical = _translated_scmm_edge(edge, delta, translation, T)
        index = _structured_scmm_face_edge_index(
            destination.normal_axis, normal_node,
            destination.u_axis, NG + u_s + u_cell - 1,
            destination.v_axis, NG + v_s + v_node - 1,
        )
        for component in 1:3
            workspace[destination.u_axis][component][index...] = canonical[component]
        end
    end
    for v_cell in 1:v_length, u_node in 1:u_length+1
        edge, delta = _mapped_scmm_payload_edge(
            payload, destination.v_axis, v_cell,
            destination.u_axis, u_node,
            destination, source, transform,
        )
        canonical = _translated_scmm_edge(edge, delta, translation, T)
        index = _structured_scmm_face_edge_index(
            destination.normal_axis, normal_node,
            destination.v_axis, NG + v_s + v_cell - 1,
            destination.u_axis, NG + u_s + u_node - 1,
        )
        for component in 1:3
            workspace[destination.v_axis][component][index...] = canonical[component]
        end
    end
    return workspace
end

struct StructuredMetrics{A}
    Ai::A
    nxi::A
    nyi::A
    nzi::A
    Aj::A
    nxj::A
    nyj::A
    nzj::A
    Ak::A
    nxk::A
    nyk::A
    nzk::A
    V::A
end

@inline structured_metrics_tuple(metrics::StructuredMetrics) = (
    metrics.Ai, metrics.nxi, metrics.nyi, metrics.nzi,
    metrics.Aj, metrics.nxj, metrics.nyj, metrics.nzj,
    metrics.Ak, metrics.nxk, metrics.nyk, metrics.nzk, metrics.V,
)

function allocate_scmm_edge_workspace(
    ::Type{T}, Nx::Int, Ny::Int, Nz::Int, NG::Int,
) where {T<:AbstractFloat}
    node_dims = (Nx + 2NG + 1, Ny + 2NG + 1, Nz + 2NG + 1)
    cell_dims = (Nx + 2NG, Ny + 2NG, Nz + 2NG)
    edge_i_shape = (cell_dims[1], node_dims[2], node_dims[3])
    edge_j_shape = (node_dims[1], cell_dims[2], node_dims[3])
    edge_k_shape = (node_dims[1], node_dims[2], cell_dims[3])
    return SCMMEdgeWorkspace(
        ntuple(_ -> zeros(T, edge_i_shape), 3),
        ntuple(_ -> zeros(T, edge_j_shape), 3),
        ntuple(_ -> zeros(T, edge_k_shape), 3),
    )
end

function allocate_structured_metrics(
    ::Type{T}, Nx::Int, Ny::Int, Nz::Int, NG::Int,
) where {T<:AbstractFloat}
    node_dims = (Nx + 2NG + 1, Ny + 2NG + 1, Nz + 2NG + 1)
    cell_dims = (Nx + 2NG, Ny + 2NG, Nz + 2NG)
    i_shape = (node_dims[1], cell_dims[2], cell_dims[3])
    j_shape = (cell_dims[1], node_dims[2], cell_dims[3])
    k_shape = (cell_dims[1], cell_dims[2], node_dims[3])
    return StructuredMetrics(
        zeros(T, i_shape), zeros(T, i_shape), zeros(T, i_shape), zeros(T, i_shape),
        zeros(T, j_shape), zeros(T, j_shape), zeros(T, j_shape), zeros(T, j_shape),
        zeros(T, k_shape), zeros(T, k_shape), zeros(T, k_shape), zeros(T, k_shape),
        zeros(T, cell_dims),
    )
end

function compute_scmm_edge_potentials!(
    workspace::SCMMEdgeWorkspace, x, y, z,
    Nx::Int, Ny::Int, Nz::Int, NG::Int;
    singularity_edges=nothing,
)
    node_dims = (Nx + 2NG + 1, Ny + 2NG + 1, Nz + 2NG + 1)
    cell_dims = (Nx + 2NG, Ny + 2NG, Nz + 2NG)
    y_dz_i, z_dx_i, x_dy_i = workspace.edge_i
    y_dz_j, z_dx_j, x_dy_j = workspace.edge_j
    y_dz_k, z_dx_k, x_dy_k = workspace.edge_k
    T = eltype(x)

    for k in 1:cell_dims[3], j in 1:node_dims[2], i in 1:node_dims[1]
        y_dz_k[i,j,k] = T(0.5) * (
            structured_scmm_interp_k(y, i, j, k, node_dims[3]) *
                structured_scmm_deriv_k(z, i, j, k, node_dims[3]) -
            structured_scmm_interp_k(z, i, j, k, node_dims[3]) *
                structured_scmm_deriv_k(y, i, j, k, node_dims[3])
        )
        z_dx_k[i,j,k] = T(0.5) * (
            structured_scmm_interp_k(z, i, j, k, node_dims[3]) *
                structured_scmm_deriv_k(x, i, j, k, node_dims[3]) -
            structured_scmm_interp_k(x, i, j, k, node_dims[3]) *
                structured_scmm_deriv_k(z, i, j, k, node_dims[3])
        )
        x_dy_k[i,j,k] = T(0.5) * (
            structured_scmm_interp_k(x, i, j, k, node_dims[3]) *
                structured_scmm_deriv_k(y, i, j, k, node_dims[3]) -
            structured_scmm_interp_k(y, i, j, k, node_dims[3]) *
                structured_scmm_deriv_k(x, i, j, k, node_dims[3])
        )
    end

    for k in 1:node_dims[3], j in 1:cell_dims[2], i in 1:node_dims[1]
        y_dz_j[i,j,k] = T(0.5) * (
            structured_scmm_interp_j(y, i, j, k, node_dims[2]) *
                structured_scmm_deriv_j(z, i, j, k, node_dims[2]) -
            structured_scmm_interp_j(z, i, j, k, node_dims[2]) *
                structured_scmm_deriv_j(y, i, j, k, node_dims[2])
        )
        z_dx_j[i,j,k] = T(0.5) * (
            structured_scmm_interp_j(z, i, j, k, node_dims[2]) *
                structured_scmm_deriv_j(x, i, j, k, node_dims[2]) -
            structured_scmm_interp_j(x, i, j, k, node_dims[2]) *
                structured_scmm_deriv_j(z, i, j, k, node_dims[2])
        )
        x_dy_j[i,j,k] = T(0.5) * (
            structured_scmm_interp_j(x, i, j, k, node_dims[2]) *
                structured_scmm_deriv_j(y, i, j, k, node_dims[2]) -
            structured_scmm_interp_j(y, i, j, k, node_dims[2]) *
                structured_scmm_deriv_j(x, i, j, k, node_dims[2])
        )
    end

    for k in 1:node_dims[3], j in 1:node_dims[2], i in 1:cell_dims[1]
        y_dz_i[i,j,k] = T(0.5) * (
            structured_scmm_interp_i(y, i, j, k, node_dims[1]) *
                structured_scmm_deriv_i(z, i, j, k, node_dims[1]) -
            structured_scmm_interp_i(z, i, j, k, node_dims[1]) *
                structured_scmm_deriv_i(y, i, j, k, node_dims[1])
        )
        z_dx_i[i,j,k] = T(0.5) * (
            structured_scmm_interp_i(z, i, j, k, node_dims[1]) *
                structured_scmm_deriv_i(x, i, j, k, node_dims[1]) -
            structured_scmm_interp_i(x, i, j, k, node_dims[1]) *
                structured_scmm_deriv_i(z, i, j, k, node_dims[1])
        )
        x_dy_i[i,j,k] = T(0.5) * (
            structured_scmm_interp_i(x, i, j, k, node_dims[1]) *
                structured_scmm_deriv_i(y, i, j, k, node_dims[1]) -
            structured_scmm_interp_i(y, i, j, k, node_dims[1]) *
                structured_scmm_deriv_i(x, i, j, k, node_dims[1])
        )
    end

    _apply_scmm_junction_edge_fallback!(
        (workspace.edge_i, workspace.edge_j, workspace.edge_k),
        x, y, z, Nx, Ny, Nz, NG, singularity_edges,
    )
    return workspace
end

function compute_scmm_edge_potentials(
    x, y, z, Nx::Int, Ny::Int, Nz::Int, NG::Int;
    singularity_edges=nothing,
)
    workspace = allocate_scmm_edge_workspace(eltype(x), Nx, Ny, Nz, NG)
    return compute_scmm_edge_potentials!(
        workspace, x, y, z, Nx, Ny, Nz, NG;
        singularity_edges=singularity_edges,
    )
end

@inline function _validated_structured_face_metric(
    area, sx, sy, sz, direction, i, j, k, active,
)
    if !isfinite(area) || area <= zero(area)
        active && throw(DomainError(
            area,
            "degenerate $(direction)-face at metric index ($i, $j, $k): " *
            "face area must be finite and strictly positive",
        ))
        return zero(area), zero(area), zero(area), zero(area)
    end
    return area, sx / area, sy / area, sz / area
end

function finalize_scmm_face_metrics!(
    metrics::StructuredMetrics, workspace::SCMMEdgeWorkspace,
    Nx::Int, Ny::Int, Nz::Int, NG::Int,
)
    node_dims = (Nx + 2NG + 1, Ny + 2NG + 1, Nz + 2NG + 1)
    cell_dims = (Nx + 2NG, Ny + 2NG, Nz + 2NG)
    y_dz_i, z_dx_i, x_dy_i = workspace.edge_i
    y_dz_j, z_dx_j, x_dy_j = workspace.edge_j
    y_dz_k, z_dx_k, x_dy_k = workspace.edge_k

    for k in 1:cell_dims[3], j in 1:cell_dims[2], i in 1:node_dims[1]
        sx = (y_dz_k[i,j+1,k] - y_dz_k[i,j,k]) -
             (y_dz_j[i,j,k+1] - y_dz_j[i,j,k])
        sy = (z_dx_k[i,j+1,k] - z_dx_k[i,j,k]) -
             (z_dx_j[i,j,k+1] - z_dx_j[i,j,k])
        sz = (x_dy_k[i,j+1,k] - x_dy_k[i,j,k]) -
             (x_dy_j[i,j,k+1] - x_dy_j[i,j,k])
        metrics.Ai[i,j,k], metrics.nxi[i,j,k], metrics.nyi[i,j,k],
            metrics.nzi[i,j,k] = _validated_structured_face_metric(
                sqrt(sx^2 + sy^2 + sz^2), sx, sy, sz, "i", i, j, k,
                NG+1 <= i <= Nx+NG+1 && NG+1 <= j <= Ny+NG &&
                    NG+1 <= k <= Nz+NG,
            )
    end

    for k in 1:cell_dims[3], j in 1:node_dims[2], i in 1:cell_dims[1]
        sx = (y_dz_i[i,j,k+1] - y_dz_i[i,j,k]) -
             (y_dz_k[i+1,j,k] - y_dz_k[i,j,k])
        sy = (z_dx_i[i,j,k+1] - z_dx_i[i,j,k]) -
             (z_dx_k[i+1,j,k] - z_dx_k[i,j,k])
        sz = (x_dy_i[i,j,k+1] - x_dy_i[i,j,k]) -
             (x_dy_k[i+1,j,k] - x_dy_k[i,j,k])
        metrics.Aj[i,j,k], metrics.nxj[i,j,k], metrics.nyj[i,j,k],
            metrics.nzj[i,j,k] = _validated_structured_face_metric(
                sqrt(sx^2 + sy^2 + sz^2), sx, sy, sz, "j", i, j, k,
                NG+1 <= i <= Nx+NG && NG+1 <= j <= Ny+NG+1 &&
                    NG+1 <= k <= Nz+NG,
            )
    end

    for k in 1:node_dims[3], j in 1:cell_dims[2], i in 1:cell_dims[1]
        sx = (y_dz_j[i+1,j,k] - y_dz_j[i,j,k]) -
             (y_dz_i[i,j+1,k] - y_dz_i[i,j,k])
        sy = (z_dx_j[i+1,j,k] - z_dx_j[i,j,k]) -
             (z_dx_i[i,j+1,k] - z_dx_i[i,j,k])
        sz = (x_dy_j[i+1,j,k] - x_dy_j[i,j,k]) -
             (x_dy_i[i,j+1,k] - x_dy_i[i,j,k])
        metrics.Ak[i,j,k], metrics.nxk[i,j,k], metrics.nyk[i,j,k],
            metrics.nzk[i,j,k] = _validated_structured_face_metric(
                sqrt(sx^2 + sy^2 + sz^2), sx, sy, sz, "k", i, j, k,
                NG+1 <= i <= Nx+NG && NG+1 <= j <= Ny+NG &&
                    NG+1 <= k <= Nz+NG+1,
            )
    end
    return metrics
end

@inline function _structured_cmd6_midpoint(values::NTuple{6,T}) where {T}
    return T(75) * (values[4] + values[3]) / T(128) -
           T(25) * (values[5] + values[2]) / T(256) +
           T(3) * (values[6] + values[1]) / T(256)
end

@inline function _structured_scmm_face_point_i(array, i, j, k)
    nj, nk = size(array, 2), size(array, 3)
    values = ntuple(6) do sample
        kk = clamp(k + sample - 3, 1, nk)
        structured_scmm_interp_j(array, i, j, kk, nj)
    end
    return _structured_cmd6_midpoint(values)
end

@inline function _structured_scmm_face_point_j(array, i, j, k)
    ni, nk = size(array, 1), size(array, 3)
    values = ntuple(6) do sample
        kk = clamp(k + sample - 3, 1, nk)
        structured_scmm_interp_i(array, i, j, kk, ni)
    end
    return _structured_cmd6_midpoint(values)
end

@inline function _structured_scmm_face_point_k(array, i, j, k)
    ni, nj = size(array, 1), size(array, 2)
    values = ntuple(6) do sample
        jj = clamp(j + sample - 3, 1, nj)
        structured_scmm_interp_i(array, i, jj, k, ni)
    end
    return _structured_cmd6_midpoint(values)
end

function finalize_scmm_volumes!(
    metrics::StructuredMetrics, x, y, z,
    Nx::Int, Ny::Int, Nz::Int, NG::Int,
)
    cell_dims = (Nx + 2NG, Ny + 2NG, Nz + 2NG)
    for k in 1:cell_dims[3], j in 1:cell_dims[2], i in 1:cell_dims[1]
        area_i_lo = metrics.Ai[i,j,k]
        dot_i_lo = _structured_scmm_face_point_i(x, i, j, k) *
                       (area_i_lo * metrics.nxi[i,j,k]) +
                   _structured_scmm_face_point_i(y, i, j, k) *
                       (area_i_lo * metrics.nyi[i,j,k]) +
                   _structured_scmm_face_point_i(z, i, j, k) *
                       (area_i_lo * metrics.nzi[i,j,k])
        area_i_hi = metrics.Ai[i+1,j,k]
        dot_i_hi = _structured_scmm_face_point_i(x, i+1, j, k) *
                       (area_i_hi * metrics.nxi[i+1,j,k]) +
                   _structured_scmm_face_point_i(y, i+1, j, k) *
                       (area_i_hi * metrics.nyi[i+1,j,k]) +
                   _structured_scmm_face_point_i(z, i+1, j, k) *
                       (area_i_hi * metrics.nzi[i+1,j,k])
        area_j_lo = metrics.Aj[i,j,k]
        dot_j_lo = _structured_scmm_face_point_j(x, i, j, k) *
                       (area_j_lo * metrics.nxj[i,j,k]) +
                   _structured_scmm_face_point_j(y, i, j, k) *
                       (area_j_lo * metrics.nyj[i,j,k]) +
                   _structured_scmm_face_point_j(z, i, j, k) *
                       (area_j_lo * metrics.nzj[i,j,k])
        area_j_hi = metrics.Aj[i,j+1,k]
        dot_j_hi = _structured_scmm_face_point_j(x, i, j+1, k) *
                       (area_j_hi * metrics.nxj[i,j+1,k]) +
                   _structured_scmm_face_point_j(y, i, j+1, k) *
                       (area_j_hi * metrics.nyj[i,j+1,k]) +
                   _structured_scmm_face_point_j(z, i, j+1, k) *
                       (area_j_hi * metrics.nzj[i,j+1,k])
        area_k_lo = metrics.Ak[i,j,k]
        dot_k_lo = _structured_scmm_face_point_k(x, i, j, k) *
                       (area_k_lo * metrics.nxk[i,j,k]) +
                   _structured_scmm_face_point_k(y, i, j, k) *
                       (area_k_lo * metrics.nyk[i,j,k]) +
                   _structured_scmm_face_point_k(z, i, j, k) *
                       (area_k_lo * metrics.nzk[i,j,k])
        area_k_hi = metrics.Ak[i,j,k+1]
        dot_k_hi = _structured_scmm_face_point_k(x, i, j, k+1) *
                       (area_k_hi * metrics.nxk[i,j,k+1]) +
                   _structured_scmm_face_point_k(y, i, j, k+1) *
                       (area_k_hi * metrics.nyk[i,j,k+1]) +
                   _structured_scmm_face_point_k(z, i, j, k+1) *
                       (area_k_hi * metrics.nzk[i,j,k+1])
        volume = (dot_i_hi - dot_i_lo + dot_j_hi - dot_j_lo +
                  dot_k_hi - dot_k_lo) / eltype(x)(3)
        active = NG+1 <= i <= Nx+NG && NG+1 <= j <= Ny+NG &&
                 NG+1 <= k <= Nz+NG
        active && (!isfinite(volume) || volume <= zero(volume)) &&
            throw(DomainError(
                volume,
                "inverted or degenerate cell at metric index ($i, $j, $k): " *
                "signed volume must be finite and strictly positive",
            ))
        metrics.V[i,j,k] = active ? inv(volume) :
            (isfinite(volume) && abs(volume) > eps(eltype(x)) ?
                inv(abs(volume)) : zero(eltype(x)))
    end
    return metrics
end

function compute_structured_metric_pipeline(
    x, y, z, Nx::Int, Ny::Int, Nz::Int, NG::Int;
    periodic=(false, false, false),
    singularity_edges=nothing,
)
    workspace = compute_scmm_edge_potentials(
        x, y, z, Nx, Ny, Nz, NG;
        singularity_edges=singularity_edges,
    )
    metrics = allocate_structured_metrics(eltype(x), Nx, Ny, Nz, NG)
    finalize_scmm_face_metrics!(metrics, workspace, Nx, Ny, Nz, NG)
    finalize_scmm_volumes!(metrics, x, y, z, Nx, Ny, Nz, NG)
    _enforce_periodic_metric_ghosts!(
        structured_metrics_tuple(metrics)..., Nx, Ny, Nz, NG, periodic,
    )
    return workspace, metrics
end

function compute_fvm_metrics_runtime(
    x, y, z, Nx::Int, Ny::Int, Nz::Int, NG::Int;
    periodic=(false, false, false),
    singularity_edges=nothing,
)
    _, metrics = compute_structured_metric_pipeline(
        x, y, z, Nx, Ny, Nz, NG;
        periodic=periodic,
        singularity_edges=singularity_edges,
    )
    return structured_metrics_tuple(metrics)
end

# =============================================================================
# Metrics caching: save/load from HDF5
# Cache filename includes block ID AND rank indices to avoid race conditions
# =============================================================================
if !isdefined(@__MODULE__, :STRUCTURED_METRIC_ALGORITHM_VERSION)
    const STRUCTURED_METRIC_ALGORITHM_VERSION = 7
end

function _structured_metric_fingerprint(arrays...)
    io = IOBuffer()
    for array in arrays
        values = Array(array)
        write(io, string(eltype(values), ":", join(size(values), ","), ";"))
        write(io, reinterpret(UInt8, vec(values)))
    end
    return bytes2hex(sha1(take!(io)))[1:16]
end

function structured_connectivity_fingerprint(connectivity)
    io = IOBuffer()
    for endpoint in sort(collect(keys(connectivity)))
        conn = connectivity[endpoint]
        write(io, string(endpoint, "->", conn.src_b, ":", conn.src_f,
                        ":", getproperty(conn, :reverse_tan),
                        ":", getproperty(conn, :flip_normal), ";"))
        if hasproperty(conn, :transform) && getproperty(conn, :transform) !== nothing
            write(io, string(getproperty(conn, :transform).source_for_destination))
        end
        if hasproperty(conn, :image_translation)
            write(io, string(getproperty(conn, :image_translation)))
        end
    end
    return bytes2hex(sha1(take!(io)))[1:16]
end

function save_metrics_to_h5(path, Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V)
    h5open(path, "w") do f
        f["Areai"] = Ai; f["nxi"] = nxi; f["nyi"] = nyi; f["nzi"] = nzi
        f["Areaj"] = Aj; f["nxj"] = nxj; f["nyj"] = nyj; f["nzj"] = nzj
        f["Areak"] = Ak; f["nxk"] = nxk; f["nyk"] = nyk; f["nzk"] = nzk
        f["Vol"] = V
    end
end

function load_metrics_from_h5(path, ::Type{T}=FT) where {T<:AbstractFloat}
    return h5open(path, "r") do file
        names = (
            "Areai", "nxi", "nyi", "nzi",
            "Areaj", "nxj", "nyj", "nzj",
            "Areak", "nxk", "nyk", "nzk", "Vol",
        )
        return map(name -> T.(read(file[name])), names)
    end
end

function read_structured_mesh_file(path)
    return h5open(path, "r") do file
        dimensions = (
            Int(read(file["Nx"])),
            Int(read(file["Ny"])),
            Int(read(file["Nz"])),
        )
        return dimensions..., read(file["coords"])
    end
end

function _metrics_cache_filename(
    bid, rx, ry, rz, Nx, Ny, Nz, NG, ::Type{T}, periodic,
    ; mesh_fingerprint::AbstractString="none",
      topology_fingerprint::AbstractString="none",
      metric_mode::Symbol=STRUCTURED_METRIC_TOPOLOGY_GHOST,
      metric_order::Int=STRUCTURED_METRIC_LOCAL_CHART_POINTS,
      block_dims::NTuple{3,Int}=(Nx, Ny, Nz),
) where {T<:AbstractFloat}
    periodic_key = join(Int(flag) for flag in periodic)
    mode_key = replace(String(metric_mode), r"[^A-Za-z0-9]" => "")
    return "metrics_cache_b$(bid)_r$(rx)_$(ry)_$(rz)_" *
           "dims$(Nx)x$(Ny)x$(Nz)_ng$(NG)_ft$(nameof(T))_" *
           "bdims$(block_dims[1])x$(block_dims[2])x$(block_dims[3])_" *
           "mode$(mode_key)_order$(metric_order)_" *
           "p$(periodic_key)_mesh$(mesh_fingerprint)_topo$(topology_fingerprint)_" *
           "alg$(STRUCTURED_METRIC_ALGORITHM_VERSION).h5"
end

"""
Load metrics from cache if valid, otherwise compute and cache.
Cache filename includes block AND rank position AND local grid size to prevent race conditions or bounds errors upon re-partitioning.
"""
function load_or_compute_metrics(bid::Int, rx::Int, ry::Int, rz::Int,
                                  x, y, z, Nx::Int, Ny::Int, Nz::Int, NG::Int;
                                  cache_metrics::Bool=true,
                                  periodic=(false, false, false),
                                  topology_fingerprint::AbstractString="none",
                                  singularity_edges=nothing,
                                  metric_mode::Symbol=structured_metric_mode_setting(),
                                  cell_offsets::NTuple{3,Int}=(0, 0, 0),
                                  block_dims::NTuple{3,Int}=(Nx, Ny, Nz),
                                  metric_auxiliary=nothing)
    _mesh_base_mc = isdefined(Main, :mesh_dir) ? getfield(Main, :mesh_dir) : "MESH"
    mesh_fingerprint = _structured_metric_fingerprint(x, y, z)
    cache_path = joinpath(
        _mesh_base_mc,
        _metrics_cache_filename(
            bid, rx, ry, rz, Nx, Ny, Nz, NG, FT, periodic,
            mesh_fingerprint=mesh_fingerprint,
            topology_fingerprint=topology_fingerprint,
            metric_mode=metric_mode,
            block_dims=block_dims,
        ),
    )
    mesh_path  = joinpath(_mesh_base_mc, "mesh_b$bid.h5")
    metric_sources = (
        @__FILE__,
        joinpath(@__DIR__, "..", "parallel", "structured_mpi.jl"),
        joinpath(@__DIR__, "..", "parallel", "ct_sync.jl"),
    )
    
    # Check if valid cache exists
    cache_newer_than_inputs = isfile(cache_path) &&
        mtime(cache_path) > mtime(mesh_path) &&
        all(source -> isfile(source) && mtime(cache_path) > mtime(source),
            metric_sources)
    metric_coordinates = build_metric_coordinates(
        x, y, z, Nx, Ny, Nz, NG;
        mode=metric_mode,
        cell_offsets=cell_offsets,
        block_dims=block_dims,
    )
    if cache_metrics && cache_newer_than_inputs
        println("    Loading cached metrics from $cache_path")
        metrics = load_metrics_from_h5(cache_path, FT)
        _enforce_periodic_metric_ghosts!(
            metrics..., Nx, Ny, Nz, NG, periodic,
        )
        metric_auxiliary !== nothing &&
            (metric_auxiliary[bid] = (
                metric_coordinates, nothing, StructuredMetrics(metrics...),
            ))
        return cache_path, false, metrics...
    end
    
    # Compute at runtime
    println(
        "    Computing metrics at runtime for block $bid rank ($rx,$ry,$rz) " *
        "with mode=$metric_mode...",
    )
    workspace, metrics = compute_structured_metric_pipeline(
        metric_coordinates.x, metric_coordinates.y, metric_coordinates.z,
        Nx, Ny, Nz, NG;
        periodic=periodic,
        singularity_edges=(metric_mode == STRUCTURED_METRIC_LOCAL_CHART ?
            nothing : singularity_edges),
    )
    metric_auxiliary !== nothing &&
        (metric_auxiliary[bid] = (metric_coordinates, workspace, metrics))
    
    return cache_path, true, structured_metrics_tuple(metrics)...
end
