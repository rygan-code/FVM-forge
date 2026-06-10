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
# NOTE: Ghost cell Q/U VALUES are filled separately by interp_ghost! (MPI)
#       and fillGhost (boundary conditions). The coordinates here only need
#       to produce smooth, reasonable metrics — not exact coincidence.
# =============================================================================

using LinearAlgebra
using HDF5

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
                                   xi_offset::Int=0)
    Nx_tot = Nx + 2*NG + 1
    Ny_tot = Ny + 2*NG + 1
    Nz_tot = Nz + 2*NG + 1
    
    x = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    y = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    z = zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    
    # Step 1: Copy real nodes to interior
    ri = NG+1:Nx+NG+1
    rj = NG+1:Ny+NG+1
    rk = NG+1:Nz+NG+1
    x[ri, rj, rk] .= x_real
    y[ri, rj, rk] .= y_real
    z[ri, rj, rk] .= z_real
    
    # Step 2: Face ghost nodes — boundary-type-aware
    _fill_face_ghosts!(x, y, z, Nx, Ny, Nz, NG, face_bc, bid, connectivity; xi_offset=xi_offset)
    
    # Step 3: Edge ghost nodes (12 edges) — product formula
    _fill_edge_ghosts!(x, y, z, Nx, Ny, Nz, NG)
    
    # Step 4: Corner ghost nodes (8 corners) — triple product
    _fill_corner_ghosts!(x, y, z, Nx, Ny, Nz, NG)
    
    return FT.(x), FT.(y), FT.(z)
end

# =============================================================================
# Face ghost filling: periodic / interblock / mirror
# =============================================================================
function _fill_face_ghosts!(x, y, z, Nx, Ny, Nz, NG, face_bc, bid, connectivity; xi_offset::Int=0)
    ri = NG+1:Nx+NG+1
    rj = NG+1:Ny+NG+1
    rk = NG+1:Nz+NG+1
    
    # Helper: get BC type for (block, face)
    _bc(fid) = haskey(face_bc, (bid, fid)) ? face_bc[(bid, fid)] : BC_ISOTHERMAL_WALL
    
    # ── ξ- (face 1) ──
    if _bc(1) == BC_PERIODIC
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
    if _bc(2) == BC_PERIODIC
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
    if _bc(3) == BC_INTERBLOCK
        _fill_interblock_face!(x, y, z, Nx, Ny, Nz, NG, bid, 3, connectivity; xi_offset=xi_offset)
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
    if _bc(4) == BC_INTERBLOCK
        _fill_interblock_face!(x, y, z, Nx, Ny, Nz, NG, bid, 4, connectivity; xi_offset=xi_offset)
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
    if _bc(5) == BC_INTERBLOCK
        _fill_interblock_face!(x, y, z, Nx, Ny, Nz, NG, bid, 5, connectivity; xi_offset=xi_offset)
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
    if _bc(6) == BC_INTERBLOCK
        _fill_interblock_face!(x, y, z, Nx, Ny, Nz, NG, bid, 6, connectivity; xi_offset=xi_offset)
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
function _fill_interblock_face!(x, y, z, Nx, Ny, Nz, NG, bid, fid, connectivity; xi_offset::Int=0)
    conn = connectivity[(bid, fid)]
    nb_bid = conn.src_b
    nb_fid = conn.src_f
    
    # Load neighbor block's real-only mesh
    _mesh_base_gc = isdefined(Main, :mesh_dir) ? mesh_dir : "MESH"
    nb_mesh_path = joinpath(_mesh_base_gc, "mesh_b$(nb_bid).h5")
    nb_coords = h5open(nb_mesh_path, "r") do f
        read(f["coords"])  # (3, Nx_nb+1, Ny_nb+1, Nz_nb+1)
    end
    x_nb = Float64.(nb_coords[1, :, :, :])
    y_nb = Float64.(nb_coords[2, :, :, :])
    z_nb = Float64.(nb_coords[3, :, :, :])
    Nx_nb = size(x_nb, 1) - 1
    Ny_nb = size(x_nb, 2) - 1
    Nz_nb = size(x_nb, 3) - 1
    
    ri = NG+1:Nx+NG+1
    rj = NG+1:Ny+NG+1
    rk = NG+1:Nz+NG+1
    
    # Determine source index mapping based on neighbor face
    # For same-type connections (η↔η or ζ↔ζ), j→j and k→k mapping is direct
    # For cross-type connections (η↔ζ), j→k swap is needed
    
    same_type = ((fid in (3,4)) && (nb_fid in (3,4))) || 
                ((fid in (5,6)) && (nb_fid in (5,6)))
    
    for g in 1:NG
        if fid == 3  # η-: my ghost at j = NG+1-g
            for k in rk, i in ri
                # i index in neighbor (ξ shared, same for all connections)
                i_nb = i - NG + xi_offset  # convert padded → full-block (1-based, with ξ-partition offset)
                
                if same_type  # neighbor face 4 (η+): skip shared boundary node
                    j_nb = Ny_nb + 1 - g  # g=1 → Ny_nb (first interior node from η+)
                    k_nb = conn.reverse_tan ? (Nz_nb + 2 - (k - NG)) : (k - NG)
                else  # cross-type: neighbor face 5 or 6 (ζ±)
                    if nb_fid == 6  # ζ+: skip shared boundary node
                        k_nb = Nz_nb + 1 - g  # g=1 → Nz_nb
                    else  # ζ-: skip shared boundary node
                        k_nb = g + 1  # g=1 → 2
                    end
                    j_nb = conn.reverse_tan ? (Ny_nb + 2 - (k - NG)) : (k - NG)  # my k → neighbor's j
                end
                x[i, NG+1-g, k] = x_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
                y[i, NG+1-g, k] = y_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
                z[i, NG+1-g, k] = z_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
            end
            
        elseif fid == 4  # η+: my ghost at j = Ny+NG+1+g
            for k in rk, i in ri
                i_nb = i - NG + xi_offset
                
                if same_type  # neighbor face 3 (η-): skip shared boundary node
                    j_nb = g + 1  # g=1 → 2 (first interior node from η-)
                    k_nb = conn.reverse_tan ? (Nz_nb + 2 - (k - NG)) : (k - NG)
                else  # cross-type
                    if nb_fid == 5  # ζ-: skip shared boundary node
                        k_nb = g + 1  # g=1 → 2
                    else  # ζ+: skip shared boundary node
                        k_nb = Nz_nb + 1 - g  # g=1 → Nz_nb
                    end
                    j_nb = conn.reverse_tan ? (Ny_nb + 2 - (k - NG)) : (k - NG)
                end
                x[i, Ny+NG+1+g, k] = x_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
                y[i, Ny+NG+1+g, k] = y_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
                z[i, Ny+NG+1+g, k] = z_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
            end
            
        elseif fid == 5  # ζ-: my ghost at k = NG+1-g
            for j in rj, i in ri
                i_nb = i - NG + xi_offset
                
                if same_type  # neighbor face 6 (ζ+): skip shared boundary node
                    j_nb = conn.reverse_tan ? (Ny_nb + 2 - (j - NG)) : (j - NG)
                    k_nb = Nz_nb + 1 - g  # g=1 → Nz_nb
                else  # cross-type: neighbor η face
                    if nb_fid == 4  # η+: skip shared boundary node
                        j_nb = Ny_nb + 1 - g  # g=1 → Ny_nb
                    else  # η-: skip shared boundary node
                        j_nb = g + 1  # g=1 → 2
                    end
                    k_nb = conn.reverse_tan ? (Nz_nb + 2 - (j - NG)) : (j - NG)  # my j → neighbor's k
                end
                x[i, j, NG+1-g] = x_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
                y[i, j, NG+1-g] = y_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
                z[i, j, NG+1-g] = z_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
            end
            
        elseif fid == 6  # ζ+: my ghost at k = Nz+NG+1+g
            for j in rj, i in ri
                i_nb = i - NG + xi_offset
                
                if same_type  # neighbor face 5 (ζ-): skip shared boundary node
                    j_nb = conn.reverse_tan ? (Ny_nb + 2 - (j - NG)) : (j - NG)
                    k_nb = g + 1  # g=1 → 2 (first interior node from ζ-)
                else  # cross-type
                    if nb_fid == 3  # η-: skip shared boundary node
                        j_nb = g + 1  # g=1 → 2
                    else  # η+: skip shared boundary node
                        j_nb = Ny_nb + 1 - g  # g=1 → Ny_nb
                    end
                    k_nb = conn.reverse_tan ? (Nz_nb + 2 - (j - NG)) : (j - NG)
                end
                x[i, j, Nz+NG+1+g] = x_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
                y[i, j, Nz+NG+1+g] = y_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
                z[i, j, Nz+NG+1+g] = z_nb[clamp(i_nb,1,Nx_nb+1), clamp(j_nb,1,Ny_nb+1), clamp(k_nb,1,Nz_nb+1)]
            end
        end
    end
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

function compute_fvm_metrics_runtime(x, y, z, Nx::Int, Ny::Int, Nz::Int, NG::Int)
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
    @inline function deriv_i(arr, i, j, k, N)
        im2 = clamp(i-2, 1, N)
        im1 = clamp(i-1, 1, N)
        i0  = clamp(i,   1, N)
        ip1 = clamp(i+1, 1, N)
        ip2 = clamp(i+2, 1, N)
        ip3 = clamp(i+3, 1, N)
        return (75.0 * (arr[ip1, j, k] - arr[i0, j, k]) / 64.0) - 
               (25.0 * (arr[ip2, j, k] - arr[im1, j, k]) / 384.0) + 
               (3.0 * (arr[ip3, j, k] - arr[im2, j, k]) / 640.0)
    end

    @inline function deriv_j(arr, i, j, k, N)
        jm2 = clamp(j-2, 1, N)
        jm1 = clamp(j-1, 1, N)
        j0  = clamp(j,   1, N)
        jp1 = clamp(j+1, 1, N)
        jp2 = clamp(j+2, 1, N)
        jp3 = clamp(j+3, 1, N)
        return (75.0 * (arr[i, jp1, k] - arr[i, j0, k]) / 64.0) - 
               (25.0 * (arr[i, jp2, k] - arr[i, jm1, k]) / 384.0) + 
               (3.0 * (arr[i, jp3, k] - arr[i, jm2, k]) / 640.0)
    end

    @inline function deriv_k(arr, i, j, k, N)
        km2 = clamp(k-2, 1, N)
        km1 = clamp(k-1, 1, N)
        k0  = clamp(k,   1, N)
        kp1 = clamp(k+1, 1, N)
        kp2 = clamp(k+2, 1, N)
        kp3 = clamp(k+3, 1, N)
        return (75.0 * (arr[i, j, kp1] - arr[i, j, k0]) / 64.0) - 
               (25.0 * (arr[i, j, kp2] - arr[i, j, km1]) / 384.0) + 
               (3.0 * (arr[i, j, kp3] - arr[i, j, km2]) / 640.0)
    end

    # ── CMD6 Midpoint Interpolation Operators ──
    @inline function interp_i(arr, i, j, k, N)
        im2 = clamp(i-2, 1, N)
        im1 = clamp(i-1, 1, N)
        i0  = clamp(i,   1, N)
        ip1 = clamp(i+1, 1, N)
        ip2 = clamp(i+2, 1, N)
        ip3 = clamp(i+3, 1, N)
        return (75.0 * (arr[ip1, j, k] + arr[i0, j, k]) / 128.0) - 
               (25.0 * (arr[ip2, j, k] + arr[im1, j, k]) / 256.0) + 
               (3.0 * (arr[ip3, j, k] + arr[im2, j, k]) / 256.0)
    end

    @inline function interp_j(arr, i, j, k, N)
        jm2 = clamp(j-2, 1, N)
        jm1 = clamp(j-1, 1, N)
        j0  = clamp(j,   1, N)
        jp1 = clamp(j+1, 1, N)
        jp2 = clamp(j+2, 1, N)
        jp3 = clamp(j+3, 1, N)
        return (75.0 * (arr[i, jp1, k] + arr[i, j0, k]) / 128.0) - 
               (25.0 * (arr[i, jp2, k] + arr[i, jm1, k]) / 256.0) + 
               (3.0 * (arr[i, jp3, k] + arr[i, jm2, k]) / 256.0)
    end

    @inline function interp_k(arr, i, j, k, N)
        km2 = clamp(k-2, 1, N)
        km1 = clamp(k-1, 1, N)
        k0  = clamp(k,   1, N)
        kp1 = clamp(k+1, 1, N)
        kp2 = clamp(k+2, 1, N)
        kp3 = clamp(k+3, 1, N)
        return (75.0 * (arr[i, j, kp1] + arr[i, j, k0]) / 128.0) - 
               (25.0 * (arr[i, j, kp2] + arr[i, j, km1]) / 256.0) + 
               (3.0 * (arr[i, j, kp3] + arr[i, j, km2]) / 256.0)
    end

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
        y_dz_k[i, j, k] = interp_k(y, i, j, k, Nz_nodes_tot) * deriv_k(z, i, j, k, Nz_nodes_tot)
        z_dx_k[i, j, k] = interp_k(z, i, j, k, Nz_nodes_tot) * deriv_k(x, i, j, k, Nz_nodes_tot)
        x_dy_k[i, j, k] = interp_k(x, i, j, k, Nz_nodes_tot) * deriv_k(y, i, j, k, Nz_nodes_tot)
    end
    
    # b) j-face/edge intermediates (midpoint in j, nodes in i and k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_cells_tot, i in 1:Nx_nodes_tot
        y_dz_j[i, j, k] = interp_j(y, i, j, k, Ny_nodes_tot) * deriv_j(z, i, j, k, Ny_nodes_tot)
        z_dx_j[i, j, k] = interp_j(z, i, j, k, Ny_nodes_tot) * deriv_j(x, i, j, k, Ny_nodes_tot)
        x_dy_j[i, j, k] = interp_j(x, i, j, k, Ny_nodes_tot) * deriv_j(y, i, j, k, Ny_nodes_tot)
    end
    
    # c) i-face/edge intermediates (midpoint in i, nodes in j and k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_nodes_tot, i in 1:Nx_cells_tot
        y_dz_i[i, j, k] = interp_i(y, i, j, k, Nx_nodes_tot) * deriv_i(z, i, j, k, Nx_nodes_tot)
        z_dx_i[i, j, k] = interp_i(z, i, j, k, Nx_nodes_tot) * deriv_i(x, i, j, k, Nx_nodes_tot)
        x_dy_i[i, j, k] = interp_i(x, i, j, k, Nx_nodes_tot) * deriv_i(y, i, j, k, Nx_nodes_tot)
    end

    # 1. Compute i-face metrics (face center: i, j+1/2, k+1/2)
    for k in 1:Nz_cells_tot, j in 1:Ny_cells_tot, i in 1:Nx_nodes_tot
        Sx = (y_dz_k[i, j+1, k] - y_dz_k[i, j, k]) - (y_dz_j[i, j, k+1] - y_dz_j[i, j, k])
        Sy = (z_dx_k[i, j+1, k] - z_dx_k[i, j, k]) - (z_dx_j[i, j, k+1] - z_dx_j[i, j, k])
        Sz = (x_dy_k[i, j+1, k] - x_dy_k[i, j, k]) - (x_dy_j[i, j, k+1] - x_dy_j[i, j, k])
        
        area = sqrt(Sx^2 + Sy^2 + Sz^2)
        Ai[i,j,k] = area
        nxi[i,j,k] = area > 1e-15 ? Sx / area : one(FT)
        nyi[i,j,k] = area > 1e-15 ? Sy / area : zero(FT)
        nzi[i,j,k] = area > 1e-15 ? Sz / area : zero(FT)
    end
    
    # 2. Compute j-face metrics (face center: i+1/2, j, k+1/2)
    for k in 1:Nz_cells_tot, j in 1:Ny_nodes_tot, i in 1:Nx_cells_tot
        Sx = (y_dz_i[i, j, k+1] - y_dz_i[i, j, k]) - (y_dz_k[i+1, j, k] - y_dz_k[i, j, k])
        Sy = (z_dx_i[i, j, k+1] - z_dx_i[i, j, k]) - (z_dx_k[i+1, j, k] - z_dx_k[i, j, k])
        Sz = (x_dy_i[i, j, k+1] - x_dy_i[i, j, k]) - (x_dy_k[i+1, j, k] - x_dy_k[i, j, k])
        
        area = sqrt(Sx^2 + Sy^2 + Sz^2)
        Aj[i,j,k] = area
        nxj[i,j,k] = area > 1e-15 ? Sx / area : zero(FT)
        nyj[i,j,k] = area > 1e-15 ? Sy / area : one(FT)
        nzj[i,j,k] = area > 1e-15 ? Sz / area : zero(FT)
    end
    
    # 3. Compute k-face metrics (face center: i+1/2, j+1/2, k)
    for k in 1:Nz_nodes_tot, j in 1:Ny_cells_tot, i in 1:Nx_cells_tot
        Sx = (y_dz_j[i+1, j, k] - y_dz_j[i, j, k]) - (y_dz_i[i, j+1, k] - y_dz_i[i, j, k])
        Sy = (z_dx_j[i+1, j, k] - z_dx_j[i, j, k]) - (z_dx_i[i, j+1, k] - z_dx_i[i, j, k])
        Sz = (x_dy_j[i+1, j, k] - x_dy_j[i, j, k]) - (x_dy_i[i, j+1, k] - x_dy_i[i, j, k])
        
        area = sqrt(Sx^2 + Sy^2 + Sz^2)
        Ak[i,j,k] = area
        nxk[i,j,k] = area > 1e-15 ? Sx / area : zero(FT)
        nyk[i,j,k] = area > 1e-15 ? Sy / area : zero(FT)
        nzk[i,j,k] = area > 1e-15 ? Sz / area : one(FT)
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
        
        vol = (dot_i_hi - dot_i_lo + dot_j_hi - dot_j_lo + dot_k_hi - dot_k_lo) / 3.0
        V[i,j,k] = one(FT) / (abs(vol) + FT(1e-30))
    end
    
    return Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V
end

# =============================================================================
# Metrics caching: save/load from HDF5
# Cache filename includes block ID AND rank indices to avoid race conditions
# =============================================================================
function save_metrics_to_h5(path, Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V)
    h5open(path, "w") do f
        f["Areai"] = Ai; f["nxi"] = nxi; f["nyi"] = nyi; f["nzi"] = nzi
        f["Areaj"] = Aj; f["nxj"] = nxj; f["nyj"] = nyj; f["nzj"] = nzj
        f["Areak"] = Ak; f["nxk"] = nxk; f["nyk"] = nyk; f["nzk"] = nzk
        f["Vol"] = V
    end
end

function load_metrics_from_h5(path)
    f = h5open(path, "r")
    Ai = read(f["Areai"]); nxi = read(f["nxi"]); nyi = read(f["nyi"]); nzi = read(f["nzi"])
    Aj = read(f["Areaj"]); nxj = read(f["nxj"]); nyj = read(f["nyj"]); nzj = read(f["nzj"])
    Ak = read(f["Areak"]); nxk = read(f["nxk"]); nyk = read(f["nyk"]); nzk = read(f["nzk"])
    V = read(f["Vol"])
    close(f)
    return Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V
end

"""
Load metrics from cache if valid, otherwise compute and cache.
Cache filename includes block AND rank position AND local grid size to prevent race conditions or bounds errors upon re-partitioning.
"""
function load_or_compute_metrics(bid::Int, rx::Int, ry::Int, rz::Int,
                                  x, y, z, Nx::Int, Ny::Int, Nz::Int, NG::Int;
                                  cache_metrics::Bool=true)
    _mesh_base_mc = isdefined(Main, :mesh_dir) ? mesh_dir : "MESH"
    cache_path = joinpath(_mesh_base_mc, "metrics_cache_b$(bid)_r$(rx)_$(ry)_$(rz)_dims$(Nx)x$(Ny)x$(Nz).h5")
    mesh_path  = joinpath(_mesh_base_mc, "mesh_b$bid.h5")
    
    # Check if valid cache exists
    if cache_metrics && isfile(cache_path) && mtime(cache_path) > mtime(mesh_path)
        println("    Loading cached metrics from $cache_path")
        return load_metrics_from_h5(cache_path)
    end
    
    # Compute at runtime
    println("    Computing metrics at runtime for block $bid rank ($rx,$ry,$rz)...")
    Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V = 
        compute_fvm_metrics_runtime(x, y, z, Nx, Ny, Nz, NG)
    
    # Save cache for next run
    if cache_metrics
        println("    Saving metrics cache to $cache_path")
        save_metrics_to_h5(cache_path, Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V)
    end
    
    return Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V
end
