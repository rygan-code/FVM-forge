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
                                   block_dims::NTuple{3,Int}=(Nx, Ny, Nz),
                                   topology_covariant::Bool=false)
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
        topology_covariant=topology_covariant,
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
    reference_coordinate=nothing,
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

    if reference_coordinate === nothing
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

    # At a multi-block junction, more than one real-node continuation may be
    # reachable.  A block-ID owner is deterministic but not geometric: rotating
    # or renumbering an otherwise identical mesh then changes the high-order
    # metric stencil.  The product extrapolation already stored at this ghost
    # node is a local smooth-continuation estimate, so use it to select the
    # physically nearest topological image.  The remaining tuple fields only
    # break ties between coincident images and cannot bias the geometry.
    function candidate_key(candidate)
        candidate_bid, candidate_index, candidate_shift = candidate
        candidate_coords = _ghost_mesh_coords(mesh_cache, candidate_bid)
        candidate_coords === nothing && return (Inf, Inf, Inf, Inf,
                                                  candidate_bid,
                                                  candidate_index...,
                                                  candidate_shift...)
        px = candidate_coords[1][candidate_index...] + candidate_shift[1]
        py = candidate_coords[2][candidate_index...] + candidate_shift[2]
        pz = candidate_coords[3][candidate_index...] + candidate_shift[3]
        distance2 = reference_coordinate === nothing ? zero(px) :
            (px - reference_coordinate[1])^2 +
            (py - reference_coordinate[2])^2 +
            (pz - reference_coordinate[3])^2
        return (distance2, px, py, pz, candidate_bid,
                candidate_index..., candidate_shift...)
    end
    sort!(candidates, by=candidate_key)
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
    topology_covariant::Bool=false,
)
    local_dims = (Nx, Ny, Nz)
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

        resolved = _resolve_topological_node(
            bid, global_index, block_dims, face_bc, connectivity, mesh_cache;
            reference_coordinate=topology_covariant ?
                (x[i,j,k], y[i,j,k], z[i,j,k]) : nothing,
        )
        resolved === nothing && throw(ArgumentError(
            "unable to resolve topological ghost node " *
            "block=$bid local=$(local_index) global=$(global_index)",
        ))
        x[i,j,k], y[i,j,k], z[i,j,k] = resolved
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

if !isdefined(@__MODULE__, :STRUCTURED_METRIC_JUNCTION_LAYERS)
    const STRUCTURED_METRIC_JUNCTION_LAYERS = 4
end

if !isdefined(@__MODULE__, :_STRUCTURED_METRIC_EDGE_PINS)
    # (axis_a, high_a, axis_b, high_b), matching the solver's 12-edge
    # topology convention.  The edge itself runs along the remaining axis.
    const _STRUCTURED_METRIC_EDGE_PINS = (
        (2, false, 3, false), (2, false, 3, true),
        (2, true,  3, false), (2, true,  3, true),
        (1, false, 3, false), (1, false, 3, true),
        (1, true,  3, false), (1, true,  3, true),
        (1, false, 2, false), (1, false, 2, true),
        (1, true,  2, false), (1, true,  2, true),
    )
end

@inline function _structured_metric_boundary_distance(
    index, axis, edge_direction, high, dims, ng,
)
    boundary = ng + 1 + (high ? dims[axis] : 0)
    coordinate = axis == edge_direction ? index + 0.5 : index
    return abs(coordinate - boundary)
end

@inline function structured_metric_near_junction_edge(
    edge_direction, i, j, k, dims, ng, singularity_edges, junction_layers,
)
    singularity_edges === nothing && return false
    junction_layers > 0 || return false
    length(singularity_edges) == 12 || throw(DimensionMismatch(
        "structured metric singularity flags must contain 12 block edges",
    ))
    indices = (i, j, k)
    @inbounds for edge_index in 1:12
        singularity_edges[edge_index] || continue
        axis_a, high_a, axis_b, high_b =
            _STRUCTURED_METRIC_EDGE_PINS[edge_index]
        distance_a = _structured_metric_boundary_distance(
            indices[axis_a], axis_a, edge_direction, high_a, dims, ng,
        )
        distance_b = _structured_metric_boundary_distance(
            indices[axis_b], axis_b, edge_direction, high_b, dims, ng,
        )
        max(distance_a, distance_b) <= junction_layers && return true
    end
    return false
end

@inline function structured_metric_endpoint_products(
    x, y, z, edge_direction, i, j, k,
)
    ip = i + (edge_direction == 1)
    jp = j + (edge_direction == 2)
    kp = k + (edge_direction == 3)
    half = eltype(x)(0.5)
    return (
        half * (y[i,j,k] * z[ip,jp,kp] - z[i,j,k] * y[ip,jp,kp]),
        half * (z[i,j,k] * x[ip,jp,kp] - x[i,j,k] * z[ip,jp,kp]),
        half * (x[i,j,k] * y[ip,jp,kp] - y[i,j,k] * x[ip,jp,kp]),
    )
end

@inline function structured_endpoint_line_integral(
    ax, ay, az, x, y, z, edge_direction, i, j, k,
)
    ip = i + (edge_direction == 1)
    jp = j + (edge_direction == 2)
    kp = k + (edge_direction == 3)
    half = eltype(x)(0.5)
    return (
        half * (ax[i,j,k] + ax[ip,jp,kp]) * (x[ip,jp,kp] - x[i,j,k]) +
        half * (ay[i,j,k] + ay[ip,jp,kp]) * (y[ip,jp,kp] - y[i,j,k]) +
        half * (az[i,j,k] + az[ip,jp,kp]) * (z[ip,jp,kp] - z[i,j,k])
    )
end

function structured_scmm_edge_line_integrals(
    vector_potential, x, y, z;
    time=zero(eltype(x)), active_dims=nothing, ng::Int=0,
    singularity_edges=nothing,
    junction_layers::Int=STRUCTURED_METRIC_JUNCTION_LAYERS,
)
    size(x) == size(y) == size(z) || throw(DimensionMismatch(
        "SCMM edge geometry arrays must have identical sizes",
    ))
    ni, nj, nk = size(x)
    if singularity_edges !== nothing && active_dims === nothing
        throw(ArgumentError(
            "active_dims is required for junction-regularized SCMM edges",
        ))
    end
    metric_dims = active_dims === nothing ? (ni - 1, nj - 1, nk - 1) :
        active_dims
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
        edge_x[i,j,k] = if structured_metric_near_junction_edge(
            1, i, j, k, metric_dims, ng, singularity_edges, junction_layers,
        )
            structured_endpoint_line_integral(
                ax, ay, az, x, y, z, 1, i, j, k,
            )
        else
            structured_scmm_interp_i(ax, i, j, k, ni) *
                structured_scmm_deriv_i(x, i, j, k, ni) +
            structured_scmm_interp_i(ay, i, j, k, ni) *
                structured_scmm_deriv_i(y, i, j, k, ni) +
            structured_scmm_interp_i(az, i, j, k, ni) *
                structured_scmm_deriv_i(z, i, j, k, ni)
        end
    end
    @inbounds for k in 1:nk, j in 1:nj-1, i in 1:ni
        edge_y[i,j,k] = if structured_metric_near_junction_edge(
            2, i, j, k, metric_dims, ng, singularity_edges, junction_layers,
        )
            structured_endpoint_line_integral(
                ax, ay, az, x, y, z, 2, i, j, k,
            )
        else
            structured_scmm_interp_j(ax, i, j, k, nj) *
                structured_scmm_deriv_j(x, i, j, k, nj) +
            structured_scmm_interp_j(ay, i, j, k, nj) *
                structured_scmm_deriv_j(y, i, j, k, nj) +
            structured_scmm_interp_j(az, i, j, k, nj) *
                structured_scmm_deriv_j(z, i, j, k, nj)
        end
    end
    @inbounds for k in 1:nk-1, j in 1:nj, i in 1:ni
        edge_z[i,j,k] = if structured_metric_near_junction_edge(
            3, i, j, k, metric_dims, ng, singularity_edges, junction_layers,
        )
            structured_endpoint_line_integral(
                ax, ay, az, x, y, z, 3, i, j, k,
            )
        else
            structured_scmm_interp_k(ax, i, j, k, nk) *
                structured_scmm_deriv_k(x, i, j, k, nk) +
            structured_scmm_interp_k(ay, i, j, k, nk) *
                structured_scmm_deriv_k(y, i, j, k, nk) +
            structured_scmm_interp_k(az, i, j, k, nk) *
                structured_scmm_deriv_k(z, i, j, k, nk)
        end
    end
    return edge_x, edge_y, edge_z
end

function compute_fvm_metrics_runtime(
    x, y, z, Nx::Int, Ny::Int, Nz::Int, NG::Int;
    periodic=(false, false, false),
    singularity_edges=nothing,
    junction_layers::Int=STRUCTURED_METRIC_JUNCTION_LAYERS,
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
        if structured_metric_near_junction_edge(
            3, i, j, k, (Nx, Ny, Nz), NG,
            singularity_edges, junction_layers,
        )
            y_dz_k[i,j,k], z_dx_k[i,j,k], x_dy_k[i,j,k] =
                structured_metric_endpoint_products(x, y, z, 3, i, j, k)
        else
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
    end
    
    # b) j-face/edge intermediates (midpoint in j, nodes in i and k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_cells_tot, i in 1:Nx_nodes_tot
        if structured_metric_near_junction_edge(
            2, i, j, k, (Nx, Ny, Nz), NG,
            singularity_edges, junction_layers,
        )
            y_dz_j[i,j,k], z_dx_j[i,j,k], x_dy_j[i,j,k] =
                structured_metric_endpoint_products(x, y, z, 2, i, j, k)
        else
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
    end
    
    # c) i-face/edge intermediates (midpoint in i, nodes in j and k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_nodes_tot, i in 1:Nx_cells_tot
        if structured_metric_near_junction_edge(
            1, i, j, k, (Nx, Ny, Nz), NG,
            singularity_edges, junction_layers,
        )
            y_dz_i[i,j,k], z_dx_i[i,j,k], x_dy_i[i,j,k] =
                structured_metric_endpoint_products(x, y, z, 1, i, j, k)
        else
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
    end

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

# =============================================================================
# Metrics caching: save/load from HDF5
# Cache filename includes block ID AND rank indices to avoid race conditions
# =============================================================================
if !isdefined(@__MODULE__, :STRUCTURED_METRIC_ALGORITHM_VERSION)
    const STRUCTURED_METRIC_ALGORITHM_VERSION = 5
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
) where {T<:AbstractFloat}
    periodic_key = join(Int(flag) for flag in periodic)
    return "metrics_cache_b$(bid)_r$(rx)_$(ry)_$(rz)_" *
           "dims$(Nx)x$(Ny)x$(Nz)_ng$(NG)_ft$(nameof(T))_" *
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
                                  junction_layers::Int=STRUCTURED_METRIC_JUNCTION_LAYERS)
    _mesh_base_mc = isdefined(Main, :mesh_dir) ? getfield(Main, :mesh_dir) : "MESH"
    mesh_fingerprint = singularity_edges === nothing ?
        _structured_metric_fingerprint(x, y, z) :
        _structured_metric_fingerprint(
            x, y, z, Int8.(collect(singularity_edges)), Int32[junction_layers],
        )
    cache_path = joinpath(
        _mesh_base_mc,
        _metrics_cache_filename(
            bid, rx, ry, rz, Nx, Ny, Nz, NG, FT, periodic,
            mesh_fingerprint=mesh_fingerprint,
            topology_fingerprint=topology_fingerprint,
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
    if cache_metrics && cache_newer_than_inputs
        println("    Loading cached metrics from $cache_path")
        metrics = load_metrics_from_h5(cache_path, FT)
        _enforce_periodic_metric_ghosts!(
            metrics..., Nx, Ny, Nz, NG, periodic,
        )
        return cache_path, false, metrics...
    end
    
    # Compute at runtime
    println("    Computing metrics at runtime for block $bid rank ($rx,$ry,$rz)...")
    Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V = 
        compute_fvm_metrics_runtime(
            x, y, z, Nx, Ny, Nz, NG;
            periodic=periodic,
            singularity_edges=singularity_edges,
            junction_layers=junction_layers,
        )
    
    return cache_path, true, Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V
end
