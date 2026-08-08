# unstruct_mesh.jl — Unstructured mesh data structure and I/O
# Part of the second-order unstructured FVM branch (OpenFOAM-style)
#
# Shared infrastructure (must be included BEFORE this file):
#   gpu_backend.jl  (GPUArray, @gpu_launch, gpu_zeros, etc.)
#   bc_types.jl     (BC_INTERBLOCK, BC_SYMMETRY, ...)
#   physics.jl      (FT, Ncons, Nprim, γ, Rg, Tw, ...)
#
# This file defines:
#   - UnstructBlock: cell-major unstructured mesh container
#   - load_unstruct_block: read OpenFOAM mesh (points/faces/owner/neighbour/boundary)
#   - gen_cartesian_unstruct: generate simple Cartesian unstructured mesh (for testing)
#   - build_csr: build cell→face and cell→neighbor CSR connectivity

using HDF5
using StaticArrays

mutable struct UnstructCTData
    nedge::Int
    edge_node_start::GPUArray{Int,1}
    edge_node_end::GPUArray{Int,1}
    edge_tx::GPUArray{FT,1}
    edge_ty::GPUArray{FT,1}
    edge_tz::GPUArray{FT,1}
    edge_length::GPUArray{FT,1}
    face_edge_offset::GPUArray{Int,1}
    face_edge_list::GPUArray{Int,1}
    face_edge_sign::GPUArray{Int8,1}
    edge_face_offset::GPUArray{Int,1}
    edge_face_list::GPUArray{Int,1}
    edge_cell_offset::GPUArray{Int,1}
    edge_cell_list::GPUArray{Int,1}
    phi_faces::GPUArray{FT,1}
    phi_backup::GPUArray{FT,1}
    edge_emf::GPUArray{FT,1}
    periodic_edge_group_offset::GPUArray{Int,1}
    periodic_edge_group_list::GPUArray{Int,1}
    periodic_edge_group_sign::GPUArray{Int8,1}
end

struct UnstructCTMPISegment
    neighbor::Int
    faces::Vector{Int}
    face_signs::Vector{Int8}
    edges::Vector{Int}
    edge_signs::Vector{Int8}
end

mutable struct UnstructCTMPIPlan
    segments::Vector{UnstructCTMPISegment}
    propagation_passes::Int
end

"""Host-side ownership and global-index metadata for a partitioned mesh."""
struct UnstructPartitionMetadata
    rank::Int
    nparts::Int
    cell_global_ids::Vector{Int64}
    face_global_ids::Vector{Int64}
    node_global_ids::Vector{Int64}
    mpi_ghost_count::Int
    mpi_send_global_ids::Vector{Int64}
    mpi_recv_global_ids::Vector{Int64}
    mpi_face_offsets::Vector{Int}
    mpi_face_list::Vector{Int}
    mpi_face_keys::Vector{Int64}
    mpi_face_signs::Vector{Int8}
end

# ═════════════════════════════════════════════════════════════
# UnstructBlock data structure
# ═════════════════════════════════════════════════════════════
# Cell-major layout: Q[cell, var], U[cell, var]
# CSR connectivity: cell_face_offset[cell]+1 .. cell_face_offset[cell+1]
#
# Indexing convention:
#   Internal cells:   1 .. ncell
#   Ghost cells:      ncell+1 .. ncell+nghost
#   Total cells:      ncell_tot = ncell + nghost
#   Faces:            1 .. nface (internal first, boundary last)
#   face_L[f] = owner cell,  face_R[f] = neighbour cell (or ghost for boundary)

mutable struct UnstructBlock
    id::Int
    ncell::Int                          # internal cell count
    nface::Int                          # total face count (internal + boundary)
    nnode::Int
    nghost::Int                         # ghost cell count
    ncell_tot::Int                      # ncell + nghost

    # ── Main variables (cell-major, includes ghost) ──
    Q::GPUArray{FT,2}                   # (ncell_tot, Nprim)  primitive
    U::GPUArray{FT,2}                   # (ncell_tot, Ncons)  conservative
    Un::GPUArray{FT,2}                  # (ncell_tot, Ncons)  RK backup
    dt_arr::GPUArray{FT,1}              # (ncell,) per-cell dt (temp, not Un!)
    grad::GPUArray{FT,3}               # (ncell_tot, Nprim, 3)  LSQ gradient
    reconstruction_limiter::GPUArray{FT,2} # (ncell_tot, Nprim)
    Fv_faces::GPUArray{FT,2}            # (nface, Ncons) viscous flux times area
    F_faces::GPUArray{FT,2}            # (nface, Ncons)  face fluxes (× area)

    # ── Geometry ──
    cell_vol::GPUArray{FT,1}            # (ncell_tot,) cell volume
    cell_cx::GPUArray{FT,1}             # (ncell_tot,) cell center x
    cell_cy::GPUArray{FT,1}
    cell_cz::GPUArray{FT,1}
    face_area::GPUArray{FT,1}           # (nface,) face area
    face_nx::GPUArray{FT,1}             # (nface,) face normal x (L→R)
    face_ny::GPUArray{FT,1}
    face_nz::GPUArray{FT,1}
    face_cx::GPUArray{FT,1}             # (nface,) face center x
    face_cy::GPUArray{FT,1}
    face_cz::GPUArray{FT,1}

    # ── Topology (CSR) ──
    face_L::GPUArray{Int,1}             # (nface,) left/owner cell index
    face_R::GPUArray{Int,1}             # (nface,) right/neighbour cell index (ghost for bnd)
    cell_face_offset::GPUArray{Int,1}   # (ncell_tot+1,) CSR offsets for cell→face
    cell_face_list::GPUArray{Int,1}     # (total_entries,) face indices per cell
    cell_face_sign::GPUArray{Int8,1}    # (total_entries,) +1 or -1 sign per face

    # ── LSQ gradient connectivity (cell→neighbor) ──
    cn_offset::GPUArray{Int,1}          # (ncell_tot+1,) CSR offsets
    cn_list::GPUArray{Int,1}            # neighbor cell indices
    cn_dx::GPUArray{FT,1}              # (n_entries,) Δx to neighbor
    cn_dy::GPUArray{FT,1}
    cn_dz::GPUArray{FT,1}
    cn_w::GPUArray{FT,1}               # (n_entries,) LSQ weight = 1/|Δr|²

    # ── Boundary ──
    face_bc_id::GPUArray{Int,1}         # (nface,) BC type ID (0=internal)
    face_bc_params::GPUArray{FT,2}      # (nface, N_BC_PARAMS)
    periodic_ghost_cells::GPUArray{Int,1}
    periodic_source_cells::GPUArray{Int,1}
    periodic_face_primary::GPUArray{Int,1}
    periodic_face_partner::GPUArray{Int,1}

    # ── Node data (for VTK output) ──
    node_x::GPUArray{FT,1}
    node_y::GPUArray{FT,1}
    node_z::GPUArray{FT,1}
    face_node_offset::GPUArray{Int,1}   # (nface+1,) CSR offsets
    face_node_list::GPUArray{Int,1}     # node indices per face
    n_face_nodes::Int                   # total entries in face_node_list
    ct::Union{Nothing,UnstructCTData}   # edge topology and face magnetic fluxes
    ct_mpi_plan::Union{Nothing,UnstructCTMPIPlan}
    partition::Union{Nothing,UnstructPartitionMetadata}

    # ── MPI buffers (partition boundaries) ──
    mpi_neighbors::Vector{Int}           # peer ranks, one entry per exchange segment
    mpi_send_offsets::Vector{Int}        # 1-based offsets into mpi_send_list
    mpi_recv_offsets::Vector{Int}        # 1-based offsets into mpi_recv_list
    mpi_send_list::Vector{Int}          # cell indices to send to neighbors
    mpi_recv_list::Vector{Int}          # ghost cell indices to receive
    sbuf_h::Vector{FT}                  # host send buffer
    rbuf_h::Vector{FT}                  # host recv buffer
    sbuf_d::GPUArray{FT,1}             # device send buffer
    rbuf_d::GPUArray{FT,1}             # device recv buffer

    # ── Launch config ──
    nb_cell::Int                        # GPU grid size for cell kernels
    nb_face::Int                        # GPU grid size for face kernels
    nthreads_cell::Int                  # threads per block for cell kernels
    nthreads_face::Int                  # threads per block for face kernels
end

function build_cartesian_periodic_pairs(
    face_L, face_R, face_bc_id, face_nx, face_ny, face_nz,
    face_cx, face_cy, face_cz,
)
    ghost_cells = Int[]
    source_cells = Int[]
    primary_faces = Int[]
    partner_faces = Int[]
    normals = (face_nx, face_ny, face_nz)
    centers = (face_cx, face_cy, face_cz)

    for direction in 1:3
        normal = normals[direction]
        transverse = direction == 1 ? (2, 3) :
            (direction == 2 ? (1, 3) : (1, 2))
        low_faces = findall(face ->
            face_bc_id[face] == Int(BC_PERIODIC) && normal[face] < -FT(0.5),
            eachindex(face_bc_id),
        )
        high_faces = findall(face ->
            face_bc_id[face] == Int(BC_PERIODIC) && normal[face] > FT(0.5),
            eachindex(face_bc_id),
        )
        isempty(low_faces) && isempty(high_faces) && continue
        length(low_faces) == length(high_faces) || throw(ArgumentError(
            "periodic direction $direction requires paired low/high faces",
        ))
        sort!(low_faces; by=face -> (
            centers[transverse[1]][face], centers[transverse[2]][face],
        ))
        sort!(high_faces; by=face -> (
            centers[transverse[1]][face], centers[transverse[2]][face],
        ))
        for (low_face, high_face) in zip(low_faces, high_faces)
            push!(ghost_cells, face_R[low_face])
            push!(source_cells, face_L[high_face])
            push!(ghost_cells, face_R[high_face])
            push!(source_cells, face_L[low_face])
            push!(primary_faces, low_face)
            push!(partner_faces, high_face)
        end
    end
    return ghost_cells, source_cells, primary_faces, partner_faces
end

# ═════════════════════════════════════════════════════════════
# Build CSR connectivity from face_L / face_R arrays
# ═════════════════════════════════════════════════════════════

"""
    build_cell_face_csr(face_L, face_R, ncell_tot) → (offsets, list, sign)

Build cell→face CSR connectivity. For each face f with (L, R):
  - cell L has face f with sign +1 (normal points outward from L)
  - cell R has face f with sign -1 (normal points into R, i.e. outward from R is -n)
Returns CSR offsets, face list, and sign array.
"""
function build_cell_face_csr(face_L::Vector{Int}, face_R::Vector{Int}, ncell_tot::Int)
    nface = length(face_L)
    # Count faces per cell
    counts = zeros(Int, ncell_tot)
    @inbounds for f in 1:nface
        L = face_L[f]; R = face_R[f]
        if L > 0; counts[L] += 1; end
        if R > 0; counts[R] += 1; end
    end
    # Build offsets (1-based, Julia convention)
    offsets = Vector{Int}(undef, ncell_tot + 1)
    offsets[1] = 1
    @inbounds for c in 1:ncell_tot
        offsets[c+1] = offsets[c] + counts[c]
    end
    total = offsets[end] - 1
    list = Vector{Int}(undef, total)
    sign = Vector{Int8}(undef, total)
    # Fill (use a running cursor)
    cursor = copy(offsets[1:ncell_tot])
    @inbounds for f in 1:nface
        L = face_L[f]; R = face_R[f]
        if L > 0
            pos = cursor[L]
            list[pos] = f
            sign[pos] = Int8(1)    # outward from L
            cursor[L] += 1
        end
        if R > 0
            pos = cursor[R]
            list[pos] = f
            sign[pos] = Int8(-1)   # outward from R is -normal
            cursor[R] += 1
        end
    end
    return offsets, list, sign
end

"""
    build_cell_neighbor_csr(face_L, face_R, cell_cx, cell_cy, cell_cz, ncell_tot)

Build cell→neighbor CSR for LSQ gradient. Only internal faces contribute
real neighbors; boundary faces connect to ghost cells (which also get neighbors).
Returns (cn_offset, cn_list, cn_dx, cn_dy, cn_dz, cn_w).
"""
function build_cell_neighbor_csr(face_L::Vector{Int}, face_R::Vector{Int},
                                  cell_cx::Vector{FT}, cell_cy::Vector{FT}, cell_cz::Vector{FT},
                                  ncell_tot::Int)
    nface = length(face_L)
    # Count neighbors per cell (one per face that has both L and R valid)
    counts = zeros(Int, ncell_tot)
    @inbounds for f in 1:nface
        L = face_L[f]; R = face_R[f]
        if L > 0 && R > 0
            counts[L] += 1
            counts[R] += 1
        end
    end
    cn_offset = Vector{Int}(undef, ncell_tot + 1)
    cn_offset[1] = 1
    @inbounds for c in 1:ncell_tot
        cn_offset[c+1] = cn_offset[c] + counts[c]
    end
    total = cn_offset[end] - 1
    cn_list = Vector{Int}(undef, total)
    cn_dx = Vector{FT}(undef, total)
    cn_dy = Vector{FT}(undef, total)
    cn_dz = Vector{FT}(undef, total)
    cn_w = Vector{FT}(undef, total)
    cursor = copy(cn_offset[1:ncell_tot])
    @inbounds for f in 1:nface
        L = face_L[f]; R = face_R[f]
        if L > 0 && R > 0
            # L's neighbor is R
            pos = cursor[L]
            cn_list[pos] = R
            dx = cell_cx[R] - cell_cx[L]
            dy = cell_cy[R] - cell_cy[L]
            dz = cell_cz[R] - cell_cz[L]
            distance_squared = dx*dx + dy*dy + dz*dz
            isfinite(distance_squared) && distance_squared > zero(FT) ||
                throw(DomainError(
                    distance_squared,
                    "face $f connects cells $L and $R with coincident or invalid centers",
                ))
            w = one(FT) / distance_squared
            cn_dx[pos] = dx; cn_dy[pos] = dy; cn_dz[pos] = dz
            cn_w[pos] = w
            cursor[L] += 1
            # R's neighbor is L (opposite direction)
            pos = cursor[R]
            cn_list[pos] = L
            cn_dx[pos] = -dx; cn_dy[pos] = -dy; cn_dz[pos] = -dz
            cn_w[pos] = w
            cursor[R] += 1
        end
    end
    return cn_offset, cn_list, cn_dx, cn_dy, cn_dz, cn_w
end

# ═════════════════════════════════════════════════════════════
# Cartesian mesh generator (for testing — Sod, TGV)
# ═════════════════════════════════════════════════════════════

"""
    gen_cartesian_unstruct(nx, ny, nz, x0, x1, y0, y1, z0, z1, bc_x, bc_y, bc_z)

Generate a Cartesian hexahedral unstructured mesh on [x0,x1]×[y0,y1]×[z0,z1].
bc_x/y/z are BC type IDs for the 6 faces: (xlo, xhi, ylo, yhi, zlo, zhi).
Returns an UnstructBlock ready for simulation (single-block, no MPI).
"""
function gen_cartesian_unstruct(nx::Int, ny::Int, nz::Int,
                                 x0::FT, x1::FT, y0::FT, y1::FT, z0::FT, z1::FT,
                                 bc_xlo::Int, bc_xhi::Int,
                                 bc_ylo::Int, bc_yhi::Int,
                                 bc_zlo::Int, bc_zhi::Int)
    ncell = nx * ny * nz
    nnode = (nx+1) * (ny+1) * (nz+1)

    # Internal faces: 3 directions
    nface_xi = (nx-1) * ny * nz      # i-direction internal (between i and i+1)
    nface_eta = nx * (ny-1) * nz     # j-direction internal
    nface_zeta = nx * ny * (nz-1)    # k-direction internal
    # Boundary faces: 2 per direction
    nface_bnd_x = ny * nz * 2
    nface_bnd_y = nx * nz * 2
    nface_bnd_z = nx * ny * 2
    nface = nface_xi + nface_eta + nface_zeta + nface_bnd_x + nface_bnd_y + nface_bnd_z

    # Ghost cells: one layer per boundary face
    nghost = nface_bnd_x + nface_bnd_y + nface_bnd_z
    ncell_tot = ncell + nghost

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / nz

    # ── Cell centers ──
    cell_cx = Vector{FT}(undef, ncell_tot)
    cell_cy = Vector{FT}(undef, ncell_tot)
    cell_cz = Vector{FT}(undef, ncell_tot)
    cell_vol = Vector{FT}(undef, ncell_tot)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        c = i + (j-1)*nx + (k-1)*nx*ny
        cell_cx[c] = x0 + (i - FT(0.5)) * dx
        cell_cy[c] = y0 + (j - FT(0.5)) * dy
        cell_cz[c] = z0 + (k - FT(0.5)) * dz
        cell_vol[c] = dx * dy * dz
    end

    # ── Node coordinates ──
    node_x = Vector{FT}(undef, nnode)
    node_y = Vector{FT}(undef, nnode)
    node_z = Vector{FT}(undef, nnode)
    @inbounds for k in 1:(nz+1), j in 1:(ny+1), i in 1:(nx+1)
        n = i + (j-1)*(nx+1) + (k-1)*(nx+1)*(ny+1)
        node_x[n] = x0 + (i-1) * dx
        node_y[n] = y0 + (j-1) * dy
        node_z[n] = z0 + (k-1) * dz
    end

    # ── Build faces ──
    face_L = Vector{Int}(undef, nface)
    face_R = Vector{Int}(undef, nface)
    face_area = Vector{FT}(undef, nface)
    face_nx = zeros(FT, nface)
    face_ny = zeros(FT, nface)
    face_nz = zeros(FT, nface)
    face_cx = Vector{FT}(undef, nface)
    face_cy = Vector{FT}(undef, nface)
    face_cz = Vector{FT}(undef, nface)
    face_bc_id = zeros(Int, nface)
    face_bc_params = zeros(FT, nface, N_BC_PARAMS)

    # Face node connectivity (each hex face = 4 nodes)
    face_node_offset = Vector{Int}(undef, nface + 1)
    face_node_list = Vector{Int}(undef, nface * 4)
    @inbounds for f in 1:nface
        face_node_offset[f] = (f-1)*4 + 1
    end
    face_node_offset[nface+1] = nface*4 + 1

    ghost_idx = ncell  # running ghost cell counter
    fidx = 0

    # Helper: linear cell index
    cell_idx(i, j, k) = i + (j-1)*nx + (k-1)*nx*ny
    # Helper: linear node index
    node_idx(i, j, k) = i + (j-1)*(nx+1) + (k-1)*(nx+1)*(ny+1)

    # ── i-direction internal faces (between cell i and i+1) ──
    @inbounds for k in 1:nz, j in 1:ny, i in 1:(nx-1)
        fidx += 1
        cL = cell_idx(i, j, k)
        cR = cell_idx(i+1, j, k)
        face_L[fidx] = cL
        face_R[fidx] = cR
        face_nx[fidx] = 1.0; face_ny[fidx] = 0.0; face_nz[fidx] = 0.0
        face_area[fidx] = dy * dz
        face_cx[fidx] = x0 + i * dx
        face_cy[fidx] = y0 + (j - FT(0.5)) * dy
        face_cz[fidx] = z0 + (k - FT(0.5)) * dz
        # face nodes (counterclockwise viewed from +x)
        face_node_list[(fidx-1)*4+1] = node_idx(i+1, j,   k)
        face_node_list[(fidx-1)*4+2] = node_idx(i+1, j+1, k)
        face_node_list[(fidx-1)*4+3] = node_idx(i+1, j+1, k+1)
        face_node_list[(fidx-1)*4+4] = node_idx(i+1, j,   k+1)
    end

    # ── j-direction internal faces ──
    @inbounds for k in 1:nz, j in 1:(ny-1), i in 1:nx
        fidx += 1
        cL = cell_idx(i, j, k)
        cR = cell_idx(i, j+1, k)
        face_L[fidx] = cL
        face_R[fidx] = cR
        face_nx[fidx] = 0.0; face_ny[fidx] = 1.0; face_nz[fidx] = 0.0
        face_area[fidx] = dx * dz
        face_cx[fidx] = x0 + (i - FT(0.5)) * dx
        face_cy[fidx] = y0 + j * dy
        face_cz[fidx] = z0 + (k - FT(0.5)) * dz
        face_node_list[(fidx-1)*4+1] = node_idx(i,   j+1, k)
        face_node_list[(fidx-1)*4+2] = node_idx(i+1, j+1, k)
        face_node_list[(fidx-1)*4+3] = node_idx(i+1, j+1, k+1)
        face_node_list[(fidx-1)*4+4] = node_idx(i,   j+1, k+1)
    end

    # ── k-direction internal faces ──
    @inbounds for k in 1:(nz-1), j in 1:ny, i in 1:nx
        fidx += 1
        cL = cell_idx(i, j, k)
        cR = cell_idx(i, j, k+1)
        face_L[fidx] = cL
        face_R[fidx] = cR
        face_nx[fidx] = 0.0; face_ny[fidx] = 0.0; face_nz[fidx] = 1.0
        face_area[fidx] = dx * dy
        face_cx[fidx] = x0 + (i - FT(0.5)) * dx
        face_cy[fidx] = y0 + (j - FT(0.5)) * dy
        face_cz[fidx] = z0 + k * dz
        face_node_list[(fidx-1)*4+1] = node_idx(i,   j,   k+1)
        face_node_list[(fidx-1)*4+2] = node_idx(i+1, j,   k+1)
        face_node_list[(fidx-1)*4+3] = node_idx(i+1, j+1, k+1)
        face_node_list[(fidx-1)*4+4] = node_idx(i,   j+1, k+1)
    end

    # ── Boundary faces: x-low ──
    @inbounds for k in 1:nz, j in 1:ny
        fidx += 1
        cL = cell_idx(1, j, k)
        ghost_idx += 1
        cR = ghost_idx
        face_L[fidx] = cL
        face_R[fidx] = cR
        face_nx[fidx] = -1.0; face_ny[fidx] = 0.0; face_nz[fidx] = 0.0
        face_area[fidx] = dy * dz
        face_cx[fidx] = x0
        face_cy[fidx] = y0 + (j - FT(0.5)) * dy
        face_cz[fidx] = z0 + (k - FT(0.5)) * dz
        face_bc_id[fidx] = bc_xlo
        cell_cx[cR] = x0 - FT(0.5) * dx
        cell_cy[cR] = cell_cy[cL]
        cell_cz[cR] = cell_cz[cL]
        cell_vol[cR] = cell_vol[cL]
        face_node_list[(fidx-1)*4+1] = node_idx(1, j,   k)
        face_node_list[(fidx-1)*4+2] = node_idx(1, j,   k+1)
        face_node_list[(fidx-1)*4+3] = node_idx(1, j+1, k+1)
        face_node_list[(fidx-1)*4+4] = node_idx(1, j+1, k)
    end

    # ── Boundary faces: x-high ──
    @inbounds for k in 1:nz, j in 1:ny
        fidx += 1
        cL = cell_idx(nx, j, k)
        ghost_idx += 1
        cR = ghost_idx
        face_L[fidx] = cL
        face_R[fidx] = cR
        face_nx[fidx] = 1.0; face_ny[fidx] = 0.0; face_nz[fidx] = 0.0
        face_area[fidx] = dy * dz
        face_cx[fidx] = x1
        face_cy[fidx] = y0 + (j - FT(0.5)) * dy
        face_cz[fidx] = z0 + (k - FT(0.5)) * dz
        face_bc_id[fidx] = bc_xhi
        cell_cx[cR] = x1 + FT(0.5) * dx
        cell_cy[cR] = cell_cy[cL]
        cell_cz[cR] = cell_cz[cL]
        cell_vol[cR] = cell_vol[cL]
        face_node_list[(fidx-1)*4+1] = node_idx(nx+1, j,   k)
        face_node_list[(fidx-1)*4+2] = node_idx(nx+1, j+1, k)
        face_node_list[(fidx-1)*4+3] = node_idx(nx+1, j+1, k+1)
        face_node_list[(fidx-1)*4+4] = node_idx(nx+1, j,   k+1)
    end

    # ── Boundary faces: y-low ──
    @inbounds for k in 1:nz, i in 1:nx
        fidx += 1
        cL = cell_idx(i, 1, k)
        ghost_idx += 1
        cR = ghost_idx
        face_L[fidx] = cL
        face_R[fidx] = cR
        face_nx[fidx] = 0.0; face_ny[fidx] = -1.0; face_nz[fidx] = 0.0
        face_area[fidx] = dx * dz
        face_cx[fidx] = x0 + (i - FT(0.5)) * dx
        face_cy[fidx] = y0
        face_cz[fidx] = z0 + (k - FT(0.5)) * dz
        face_bc_id[fidx] = bc_ylo
        cell_cx[cR] = cell_cx[cL]
        cell_cy[cR] = y0 - FT(0.5) * dy
        cell_cz[cR] = cell_cz[cL]
        cell_vol[cR] = cell_vol[cL]
        face_node_list[(fidx-1)*4+1] = node_idx(i,   1, k)
        face_node_list[(fidx-1)*4+2] = node_idx(i+1, 1, k)
        face_node_list[(fidx-1)*4+3] = node_idx(i+1, 1, k+1)
        face_node_list[(fidx-1)*4+4] = node_idx(i,   1, k+1)
    end

    # ── Boundary faces: y-high ──
    @inbounds for k in 1:nz, i in 1:nx
        fidx += 1
        cL = cell_idx(i, ny, k)
        ghost_idx += 1
        cR = ghost_idx
        face_L[fidx] = cL
        face_R[fidx] = cR
        face_nx[fidx] = 0.0; face_ny[fidx] = 1.0; face_nz[fidx] = 0.0
        face_area[fidx] = dx * dz
        face_cx[fidx] = x0 + (i - FT(0.5)) * dx
        face_cy[fidx] = y1
        face_cz[fidx] = z0 + (k - FT(0.5)) * dz
        face_bc_id[fidx] = bc_yhi
        cell_cx[cR] = cell_cx[cL]
        cell_cy[cR] = y1 + FT(0.5) * dy
        cell_cz[cR] = cell_cz[cL]
        cell_vol[cR] = cell_vol[cL]
        face_node_list[(fidx-1)*4+1] = node_idx(i,   ny+1, k)
        face_node_list[(fidx-1)*4+2] = node_idx(i,   ny+1, k+1)
        face_node_list[(fidx-1)*4+3] = node_idx(i+1, ny+1, k+1)
        face_node_list[(fidx-1)*4+4] = node_idx(i+1, ny+1, k)
    end

    # ── Boundary faces: z-low ──
    @inbounds for j in 1:ny, i in 1:nx
        fidx += 1
        cL = cell_idx(i, j, 1)
        ghost_idx += 1
        cR = ghost_idx
        face_L[fidx] = cL
        face_R[fidx] = cR
        face_nx[fidx] = 0.0; face_ny[fidx] = 0.0; face_nz[fidx] = -1.0
        face_area[fidx] = dx * dy
        face_cx[fidx] = x0 + (i - FT(0.5)) * dx
        face_cy[fidx] = y0 + (j - FT(0.5)) * dy
        face_cz[fidx] = z0
        face_bc_id[fidx] = bc_zlo
        cell_cx[cR] = cell_cx[cL]
        cell_cy[cR] = cell_cy[cL]
        cell_cz[cR] = z0 - FT(0.5) * dz
        cell_vol[cR] = cell_vol[cL]
        face_node_list[(fidx-1)*4+1] = node_idx(i,   j,   1)
        face_node_list[(fidx-1)*4+2] = node_idx(i,   j+1, 1)
        face_node_list[(fidx-1)*4+3] = node_idx(i+1, j+1, 1)
        face_node_list[(fidx-1)*4+4] = node_idx(i+1, j,   1)
    end

    # ── Boundary faces: z-high ──
    @inbounds for j in 1:ny, i in 1:nx
        fidx += 1
        cL = cell_idx(i, j, nz)
        ghost_idx += 1
        cR = ghost_idx
        face_L[fidx] = cL
        face_R[fidx] = cR
        face_nx[fidx] = 0.0; face_ny[fidx] = 0.0; face_nz[fidx] = 1.0
        face_area[fidx] = dx * dy
        face_cx[fidx] = x0 + (i - FT(0.5)) * dx
        face_cy[fidx] = y0 + (j - FT(0.5)) * dy
        face_cz[fidx] = z1
        face_bc_id[fidx] = bc_zhi
        cell_cx[cR] = cell_cx[cL]
        cell_cy[cR] = cell_cy[cL]
        cell_cz[cR] = z1 + FT(0.5) * dz
        cell_vol[cR] = cell_vol[cL]
        face_node_list[(fidx-1)*4+1] = node_idx(i,   j,   nz+1)
        face_node_list[(fidx-1)*4+2] = node_idx(i+1, j,   nz+1)
        face_node_list[(fidx-1)*4+3] = node_idx(i+1, j+1, nz+1)
        face_node_list[(fidx-1)*4+4] = node_idx(i,   j+1, nz+1)
    end

    @assert fidx == nface "Face count mismatch: $fidx != $nface"
    @assert ghost_idx == ncell_tot "Ghost count mismatch: $ghost_idx != $ncell_tot"

    periodic_ghost_cells, periodic_source_cells,
    periodic_face_primary, periodic_face_partner = build_cartesian_periodic_pairs(
        face_L, face_R, face_bc_id, face_nx, face_ny, face_nz,
        face_cx, face_cy, face_cz,
    )

    # ── Build CSR connectivity ──
    cell_face_offset, cell_face_list, cell_face_sign = build_cell_face_csr(face_L, face_R, ncell_tot)
    cn_offset, cn_list, cn_dx, cn_dy, cn_dz, cn_w = build_cell_neighbor_csr(
        face_L, face_R, cell_cx, cell_cy, cell_cz, ncell_tot)

    # ── Upload to GPU ──
    nthreads_cell = 256
    nthreads_face = 256
    nb_cell = cld(ncell, nthreads_cell)
    nb_face = cld(nface, nthreads_face)

    return UnstructBlock(
        1,                              # id
        ncell, nface, nnode, nghost, ncell_tot,
        # Variables (allocated as zeros, filled by initializer)
        gpu_zeros(FT, ncell_tot, Nprim),
        gpu_zeros(FT, ncell_tot, Ncons),
        gpu_zeros(FT, ncell_tot, Ncons),
        gpu_zeros(FT, ncell),               # dt_arr (internal cells only)
        gpu_zeros(FT, ncell_tot, Nprim, 3),
        gpu_zeros(FT, ncell_tot, Nprim),
        gpu_zeros(FT, nface, Ncons),
        gpu_zeros(FT, nface, Ncons),
        # Geometry
        GPUArray(cell_vol),
        GPUArray(cell_cx), GPUArray(cell_cy), GPUArray(cell_cz),
        GPUArray(face_area),
        GPUArray(face_nx), GPUArray(face_ny), GPUArray(face_nz),
        GPUArray(face_cx), GPUArray(face_cy), GPUArray(face_cz),
        # Topology
        GPUArray(face_L), GPUArray(face_R),
        GPUArray(cell_face_offset), GPUArray(cell_face_list), GPUArray(cell_face_sign),
        # LSQ neighbor
        GPUArray(cn_offset), GPUArray(cn_list),
        GPUArray(cn_dx), GPUArray(cn_dy), GPUArray(cn_dz), GPUArray(cn_w),
        # Boundary
        GPUArray(face_bc_id), GPUArray(face_bc_params),
        GPUArray(periodic_ghost_cells), GPUArray(periodic_source_cells),
        GPUArray(periodic_face_primary), GPUArray(periodic_face_partner),
        # Nodes
        GPUArray(node_x), GPUArray(node_y), GPUArray(node_z),
        GPUArray(face_node_offset), GPUArray(face_node_list),
        nface * 4,
        nothing, nothing, nothing,
        # MPI (empty for single-block)
        Int[], Int[1], Int[1], Int[], Int[], FT[], FT[],
        gpu_zeros(FT, 0), gpu_zeros(FT, 0),
        # Launch config
        nb_cell, nb_face, nthreads_cell, nthreads_face,
    )
end

# ═════════════════════════════════════════════════════════════
# OpenFOAM mesh reader
# ═════════════════════════════════════════════════════════════

"""
    read_openfoam_mesh(mesh_dir) → UnstructBlock

Read an OpenFOAM mesh from `mesh_dir` (e.g. "case/constant/polyMesh").
Expects files: points, faces, owner, neighbour, boundary.
"""
function read_openfoam_mesh(mesh_dir::String)
    @info "Reading OpenFOAM mesh from $mesh_dir"

    # ── Read points ──
    points = read_openfoam_points(joinpath(mesh_dir, "points"))
    nnode = size(points, 1)
    node_x = points[:, 1]
    node_y = points[:, 2]
    node_z = points[:, 3]

    # ── Read faces ──
    face_nodes = read_openfoam_faces(joinpath(mesh_dir, "faces"))
    nface = length(face_nodes)

    # ── Read owner and neighbour ──
    owner = read_openfoam_label_list(joinpath(mesh_dir, "owner"))
    neighbour = read_openfoam_label_list(joinpath(mesh_dir, "neighbour"))
    @assert length(owner) == nface "owner length mismatch"
    n_internal_faces = length(neighbour)

    # ── Read boundary ──
    boundary_patches = read_openfoam_boundary(joinpath(mesh_dir, "boundary"))

    # ── Determine ncell ──
    # owner/neighbour are already 1-based (converted in read_openfoam_label_list)
    ncell = 0
    for o in owner
        ncell = max(ncell, o)
    end
    for n in neighbour
        ncell = max(ncell, n)
    end
    # ncell = max cell index (1-based), no +1 needed

    # ── Count boundary faces and ghosts ──
    n_boundary_faces = nface - n_internal_faces
    nghost = n_boundary_faces
    ncell_tot = ncell + nghost

    # ── Build face_L, face_R (owner/neighbour already 1-based) ──
    face_L = Vector{Int}(undef, nface)
    face_R = Vector{Int}(undef, nface)
    # Internal faces: L = owner, R = neighbour (already 1-based)
    for f in 1:n_internal_faces
        face_L[f] = owner[f]
        face_R[f] = neighbour[f]
    end
    # Boundary faces: L = owner, R = ghost cell
    ghost_counter = ncell
    for f in (n_internal_faces+1):nface
        face_L[f] = owner[f]
        ghost_counter += 1
        face_R[f] = ghost_counter
    end
    @assert ghost_counter == ncell_tot

    # ── Build BC IDs for boundary faces ──
    face_bc_id = zeros(Int, nface)
    face_bc_params = zeros(FT, nface, N_BC_PARAMS)
    for patch in boundary_patches
        patch_type = lowercase(patch["type"])
        normalized_type = replace(patch_type,"_"=>"")
        matching_name = nothing
        for name in keys(BC_NAME_MAP)
            if replace(lowercase(name),"_"=>"") == normalized_type
                matching_name = name
                break
            end
        end
        matching_name === nothing && throw(ArgumentError(
            "unsupported OpenFOAM boundary type '$patch_type' " *
            "for patch '$(patch["name"])'",
        ))
        bc_type = BC_NAME_MAP[matching_name]
        for f in patch["startFace"]+1 : patch["startFace"] + patch["nFaces"]
            face_bc_id[f] = bc_type
        end
    end

    # ── Compute geometry (face area, normal, center; cell center, volume) ──
    cell_cx, cell_cy, cell_cz, cell_vol = Vector{FT}(undef, ncell_tot), Vector{FT}(undef, ncell_tot),
                                          Vector{FT}(undef, ncell_tot), Vector{FT}(undef, ncell_tot)
    face_area, face_nx, face_ny, face_nz = Vector{FT}(undef, nface), Vector{FT}(undef, nface),
                                           Vector{FT}(undef, nface), Vector{FT}(undef, nface)
    face_cx, face_cy, face_cz = Vector{FT}(undef, nface), Vector{FT}(undef, nface), Vector{FT}(undef, nface)

    # Face geometry from node coordinates
    for f in 1:nface
        nodes = face_nodes[f]
        npts = length(nodes)
        npts >= 3 || throw(DomainError(
            npts, "OpenFOAM face $f must contain at least three nodes",
        ))
        # Compute area vector via polygon cross-product: S = 0.5 * Σ (r_{i+1} × r_i)
        Sx = Sy = Sz = zero(FT)
        cx_sum = cy_sum = cz_sum = zero(FT)
        for i in 1:npts
            n1 = nodes[i]
            n2 = nodes[mod1(i+1, npts)]
            x1 = node_x[n1]; y1 = node_y[n1]; z1 = node_z[n1]
            x2 = node_x[n2]; y2 = node_y[n2]; z2 = node_z[n2]
            Sx += (y1 * z2 - z1 * y2)
            Sy += (z1 * x2 - x1 * z2)
            Sz += (x1 * y2 - y1 * x2)
            cx_sum += x1; cy_sum += y1; cz_sum += z1
        end
        Sx *= FT(0.5); Sy *= FT(0.5); Sz *= FT(0.5)
        area = sqrt(Sx*Sx + Sy*Sy + Sz*Sz)
        isfinite(area) && area > zero(FT) || throw(DomainError(
            area, "OpenFOAM face $f has zero or invalid area",
        ))
        face_area[f] = area
        face_nx[f] = Sx / area
        face_ny[f] = Sy / area
        face_nz[f] = Sz / area
        # Face center = average of nodes
        face_cx[f] = cx_sum / npts
        face_cy[f] = cy_sum / npts
        face_cz[f] = cz_sum / npts
    end

    # Cell center (simple average of surrounding face centers) and volume (divergence theorem)
    fill!(cell_cx, zero(FT)); fill!(cell_cy, zero(FT)); fill!(cell_cz, zero(FT))
    fill!(cell_vol, zero(FT))
    face_count_per_cell = zeros(Int, ncell_tot)
    for f in 1:nface
        L = face_L[f]; R = face_R[f]
        Sx = face_nx[f] * face_area[f]
        Sy = face_ny[f] * face_area[f]
        Sz = face_nz[f] * face_area[f]
        # Volume: V += (1/3) * r_face · S_face (outward normal)
        # For L: outward = +S; for R: outward = -S
        cell_vol[L] += FT(1/3) * (face_cx[f]*Sx + face_cy[f]*Sy + face_cz[f]*Sz)
        cell_cx[L] += face_cx[f]; cell_cy[L] += face_cy[f]; cell_cz[L] += face_cz[f]
        face_count_per_cell[L] += 1
        if R <= ncell  # internal face
            cell_vol[R] -= FT(1/3) * (face_cx[f]*Sx + face_cy[f]*Sy + face_cz[f]*Sz)
            cell_cx[R] += face_cx[f]; cell_cy[R] += face_cy[f]; cell_cz[R] += face_cz[f]
            face_count_per_cell[R] += 1
        end
    end
    # Normalize internal cell centers by face count (simple average)
    for c in 1:ncell
        nfc = face_count_per_cell[c]
        nfc > 0 || throw(DomainError(
            nfc, "OpenFOAM cell $c has no incident faces",
        ))
        cell_cx[c] /= nfc
        cell_cy[c] /= nfc
        cell_cz[c] /= nfc
        volume = cell_vol[c]
        isfinite(volume) && volume > zero(FT) || throw(DomainError(
            volume, "OpenFOAM cell $c has non-positive or invalid signed volume",
        ))
    end
    # Ghost cell centers: mirror across boundary face
    for f in (n_internal_faces+1):nface
        g = face_R[f]
        L = face_L[f]
        cell_cx[g] = FT(2)*face_cx[f] - cell_cx[L]
        cell_cy[g] = FT(2)*face_cy[f] - cell_cy[L]
        cell_cz[g] = FT(2)*face_cz[f] - cell_cz[L]
        cell_vol[g] = cell_vol[L]
    end

    # ── Build CSR ──
    cell_face_offset, cell_face_list, cell_face_sign = build_cell_face_csr(face_L, face_R, ncell_tot)
    cn_offset, cn_list, cn_dx, cn_dy, cn_dz, cn_w = build_cell_neighbor_csr(
        face_L, face_R, cell_cx, cell_cy, cell_cz, ncell_tot)

    # ── Build face_node CSR for VTK output ──
    face_node_offset = Vector{Int}(undef, nface + 1)
    face_node_offset[1] = 1
    total_fn = 0
    for f in 1:nface
        total_fn += length(face_nodes[f])
        face_node_offset[f+1] = total_fn + 1
    end
    face_node_flat = Vector{Int}(undef, total_fn)
    pos = 1
    for f in 1:nface
        for n in face_nodes[f]
            face_node_flat[pos] = n
            pos += 1
        end
    end

    # ── Upload to GPU ──
    nthreads_cell = 256
    nthreads_face = 256
    nb_cell = cld(ncell, nthreads_cell)
    nb_face = cld(nface, nthreads_face)

    @info "  OpenFOAM mesh loaded: ncell=$ncell, nface=$nface, nnode=$nnode, nghost=$nghost"

    return UnstructBlock(
        1, ncell, nface, nnode, nghost, ncell_tot,
        gpu_zeros(FT, ncell_tot, Nprim),
        gpu_zeros(FT, ncell_tot, Ncons),
        gpu_zeros(FT, ncell_tot, Ncons),
        gpu_zeros(FT, ncell),               # dt_arr
        gpu_zeros(FT, ncell_tot, Nprim, 3),
        gpu_zeros(FT, ncell_tot, Nprim),
        gpu_zeros(FT, nface, Ncons),
        gpu_zeros(FT, nface, Ncons),
        GPUArray(cell_vol),
        GPUArray(cell_cx), GPUArray(cell_cy), GPUArray(cell_cz),
        GPUArray(face_area),
        GPUArray(face_nx), GPUArray(face_ny), GPUArray(face_nz),
        GPUArray(face_cx), GPUArray(face_cy), GPUArray(face_cz),
        GPUArray(face_L), GPUArray(face_R),
        GPUArray(cell_face_offset), GPUArray(cell_face_list), GPUArray(cell_face_sign),
        GPUArray(cn_offset), GPUArray(cn_list),
        GPUArray(cn_dx), GPUArray(cn_dy), GPUArray(cn_dz), GPUArray(cn_w),
        GPUArray(face_bc_id), GPUArray(face_bc_params),
        gpu_zeros(Int, 0), gpu_zeros(Int, 0),
        gpu_zeros(Int, 0), gpu_zeros(Int, 0),
        GPUArray(node_x), GPUArray(node_y), GPUArray(node_z),
        GPUArray(face_node_offset), GPUArray(face_node_flat),
        total_fn,
        nothing, nothing, nothing,
        Int[], Int[1], Int[1], Int[], Int[], FT[], FT[],
        gpu_zeros(FT, 0), gpu_zeros(FT, 0),
        nb_cell, nb_face, nthreads_cell, nthreads_face,
    )
end

# ═════════════════════════════════════════════════════════════
# OpenFOAM file format parsers
# ═════════════════════════════════════════════════════════════

"""Read OpenFOAM `points` file → Matrix{FT}(nnode, 3)"""
function read_openfoam_points(filepath::String)
    lines = readlines(filepath)
    # Find the line with "(" — then nnode points follow
    idx = 1
    while idx <= length(lines) && !startswith(strip(lines[idx]), "(")
        idx += 1
    end
    idx += 1  # skip "("
    points = FT[]
    while idx <= length(lines) && !startswith(strip(lines[idx]), ")")
        line = strip(lines[idx])
    if isempty(line) || startswith(line, "//")
            idx += 1
            continue
        end
        # Parse "(x y z)"
        line = replace(line, "(" => " ", ")" => " ")
        vals = split(line)
        if length(vals) >= 3
            push!(points, parse(FT, vals[1]))
            push!(points, parse(FT, vals[2]))
            push!(points, parse(FT, vals[3]))
        end
        idx += 1
    end
    nnode = length(points) ÷ 3
    return reshape(points, 3, nnode)'  # (nnode, 3)
end

"""Read OpenFOAM `faces` file → Vector{Vector{Int}} (1-based node indices)"""
function read_openfoam_faces(filepath::String)
    lines = readlines(filepath)
    idx = 1
    while idx <= length(lines) && !startswith(strip(lines[idx]), "(")
        idx += 1
    end
    idx += 1  # skip "("
    faces = Vector{Vector{Int}}()
    while idx <= length(lines) && !startswith(strip(lines[idx]), ")")
        line = strip(lines[idx])
        if isempty(line) || startswith(line, "//")
            idx += 1
            continue
        end
        # Format: "nnode(i1 i2 ... in)" or "nnode(i1 i2 ... in)"
        paren_open = findfirst('(', line)
        paren_close = findlast(')', line)
        if paren_open !== nothing && paren_close !== nothing
            n = parse(Int, line[1:paren_open-1])
            nums_str = line[paren_open+1:paren_close-1]
            nums = parse.(Int, split(nums_str))
            push!(faces, nums .+ 1)  # 0-based → 1-based
        end
        idx += 1
    end
    return faces
end

"""Read OpenFOAM label list (owner/neighbour) → Vector{Int} (1-based)"""
function read_openfoam_label_list(filepath::String)
    lines = readlines(filepath)
    idx = 1
    # Skip header (notes), find "("
    while idx <= length(lines)
        s = strip(lines[idx])
        if startswith(s, "(") && length(s) == 1
            break
        end
        if startswith(s, "(") && length(s) > 1
            # "(" on same line as data — rare, skip just the note lines
            idx += 1
            continue
        end
        idx += 1
    end
    idx += 1  # skip "("
    labels = Int[]
    while idx <= length(lines) && !startswith(strip(lines[idx]), ")")
        line = strip(lines[idx])
        if isempty(line) || startswith(line, "//")
            idx += 1
            continue
        end
        for tok in split(line)
            push!(labels, parse(Int, tok))
        end
        idx += 1
    end
    return labels .+ 1  # 0-based → 1-based
end

"""Read OpenFOAM `boundary` file → Vector{Dict} with keys: name, type, nFaces, startFace"""
function read_openfoam_boundary(filepath::String)
    lines = readlines(filepath)
    patches = Vector{Dict{String, Any}}()
    idx = 1
    while idx <= length(lines) && !startswith(strip(lines[idx]), "(")
        idx += 1
    end
    idx += 1  # skip "("
    while idx <= length(lines) && !startswith(strip(lines[idx]), ")")
        line = strip(lines[idx])
        if isempty(line) || startswith(line, "//")
            idx += 1
            continue
        end
        # Patch name
        patch_name = line
        idx += 1
        # Expect "{"
        while idx <= length(lines) && !startswith(strip(lines[idx]), "{")
            idx += 1
        end
        idx += 1  # skip "{"
        patch = Dict{String, Any}("name" => patch_name)
        while idx <= length(lines)
            s = strip(lines[idx])
            if startswith(s, "}")
                idx += 1
                break
            end
            if !isempty(s) && !startswith(s, "//")
                # Parse "key value;" or "key type;"
                s = replace(s, ";" => "")
                parts = split(s)
                if length(parts) >= 2
                    patch[parts[1]] = parts[2]
                end
            end
            idx += 1
        end
        # Convert numeric fields
        if haskey(patch, "nFaces")
            patch["nFaces"] = parse(Int, patch["nFaces"])
        end
        if haskey(patch, "startFace")
            patch["startFace"] = parse(Int, patch["startFace"])
        end
        push!(patches, patch)
    end
    return patches
end
