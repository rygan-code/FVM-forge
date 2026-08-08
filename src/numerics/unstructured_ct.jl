# Edge-based constrained transport for general polyhedral unstructured meshes.

if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end

using LinearAlgebra
using SparseArrays

if !@isdefined(ct_mode)
    const ct_mode = false
end
if !@isdefined(η_mhd)
    const η_mhd = zero(FT)
end
if !@isdefined(resistive)
    const resistive = false
end

function _csr_from_lists(lists::Vector{Vector{Int}})
    offsets = Vector{Int}(undef,length(lists)+1)
    offsets[1] = 1
    for index in eachindex(lists)
        offsets[index+1] = offsets[index] + length(lists[index])
    end
    values = Vector{Int}(undef,offsets[end]-1)
    for index in eachindex(lists)
        copyto!(values,offsets[index],lists[index],1,length(lists[index]))
    end
    return offsets,values
end

function _periodic_edge_groups(
    block, face_edges, edge_node_start, edge_node_end,
    edge_tx, edge_ty, edge_tz, edge_length,
    node_x, node_y, node_z, face_cx, face_cy, face_cz,
)
    adjacency = [Tuple{Int,Int8}[] for _ in eachindex(edge_node_start)]
    primary_faces = Array(block.periodic_face_primary)
    partner_faces = Array(block.periodic_face_partner)
    edge_midpoint(edge) = SVector{3,FT}(
        FT(0.5)*(node_x[edge_node_start[edge]]+node_x[edge_node_end[edge]]),
        FT(0.5)*(node_y[edge_node_start[edge]]+node_y[edge_node_end[edge]]),
        FT(0.5)*(node_z[edge_node_start[edge]]+node_z[edge_node_end[edge]]),
    )
    edge_tangent(edge) = SVector{3,FT}(
        edge_tx[edge],edge_ty[edge],edge_tz[edge],
    )

    for (primary, partner) in zip(primary_faces, partner_faces)
        translation = SVector{3,FT}(
            face_cx[partner]-face_cx[primary],
            face_cy[partner]-face_cy[primary],
            face_cz[partner]-face_cz[primary],
        )
        available = Set(face_edges[partner])
        scale = max(norm(translation),one(FT))
        tolerance = FT(32)*sqrt(eps(FT))*scale
        for primary_edge in face_edges[primary]
            target = edge_midpoint(primary_edge)+translation
            partner_edge = 0
            best_distance = typemax(FT)
            for candidate in available
                distance = norm(edge_midpoint(candidate)-target)
                if distance < best_distance
                    best_distance = distance
                    partner_edge = candidate
                end
            end
            best_distance <= tolerance || error(
                "unable to pair periodic CT edge on faces $primary/$partner",
            )
            delete!(available,partner_edge)
            tangent_dot = dot(
                edge_tangent(primary_edge),edge_tangent(partner_edge),
            )
            abs(tangent_dot) >= FT(0.5) || error(
                "periodic CT edge tangents are inconsistent",
            )
            orientation = tangent_dot > zero(FT) ? Int8(1) : Int8(-1)
            push!(adjacency[primary_edge],(partner_edge,orientation))
            push!(adjacency[partner_edge],(primary_edge,orientation))
        end
        isempty(available) || error(
            "periodic CT faces have different edge counts",
        )
    end

    visited = falses(length(adjacency))
    groups = Vector{Vector{Int}}()
    group_signs = Vector{Vector{Int8}}()
    for root in eachindex(adjacency)
        visited[root] && continue
        isempty(adjacency[root]) && continue
        edges = Int[root]
        signs = Int8[1]
        visited[root] = true
        cursor = 1
        while cursor <= length(edges)
            edge = edges[cursor]
            edge_sign = signs[cursor]
            for (neighbor,relative_sign) in adjacency[edge]
                expected_sign = edge_sign*relative_sign
                position = findfirst(==(neighbor),edges)
                if position === nothing
                    push!(edges,neighbor)
                    push!(signs,expected_sign)
                    visited[neighbor] = true
                elseif signs[position] != expected_sign
                    error("inconsistent periodic CT edge orientation cycle")
                end
            end
            cursor += 1
        end
        push!(groups,edges)
        push!(group_signs,signs)
    end
    offsets, edge_list = _csr_from_lists(groups)
    signs = reduce(vcat,group_signs;init=Int8[])
    return offsets,edge_list,signs
end

function build_unstruct_ct_topology!(block::UnstructBlock)
    block.ct === nothing || return block.ct
    face_node_offset = Array(block.face_node_offset)
    face_node_list = Array(block.face_node_list)
    node_x = Array(block.node_x)
    node_y = Array(block.node_y)
    node_z = Array(block.node_z)
    face_nx = Array(block.face_nx)
    face_ny = Array(block.face_ny)
    face_nz = Array(block.face_nz)
    face_L = Array(block.face_L)
    face_R = Array(block.face_R)
    face_bc_id = Array(block.face_bc_id)
    face_cx = Array(block.face_cx)
    face_cy = Array(block.face_cy)
    face_cz = Array(block.face_cz)

    edge_map = Dict{Tuple{Int,Int},Int}()
    edge_node_start = Int[]
    edge_node_end = Int[]
    face_edges = Vector{Vector{Int}}(undef,block.nface)
    face_signs = Vector{Vector{Int8}}(undef,block.nface)
    edge_faces = Vector{Vector{Int}}()

    for face in 1:block.nface
        first_node = face_node_offset[face]
        last_node = face_node_offset[face+1]-1
        nodes = face_node_list[first_node:last_node]
        length(nodes) >= 3 || error("CT face $face has fewer than three nodes")

        area_x = zero(FT)
        area_y = zero(FT)
        area_z = zero(FT)
        for local_index in eachindex(nodes)
            node_a = nodes[local_index]
            node_b = nodes[local_index == length(nodes) ? 1 : local_index+1]
            area_x += (node_y[node_a]-node_y[node_b])*(node_z[node_a]+node_z[node_b])
            area_y += (node_z[node_a]-node_z[node_b])*(node_x[node_a]+node_x[node_b])
            area_z += (node_x[node_a]-node_x[node_b])*(node_y[node_a]+node_y[node_b])
        end
        orientation = area_x*face_nx[face] + area_y*face_ny[face] + area_z*face_nz[face]
        abs(orientation) > FT(100)*eps(FT) || error(
            "CT face $face has degenerate or inconsistent node geometry")
        orientation_sign = orientation > zero(FT) ? Int8(1) : Int8(-1)

        local_edges = Int[]
        local_signs = Int8[]
        for local_index in eachindex(nodes)
            node_a = nodes[local_index]
            node_b = nodes[local_index == length(nodes) ? 1 : local_index+1]
            key = node_a < node_b ? (node_a,node_b) : (node_b,node_a)
            edge = get(edge_map,key,0)
            if edge == 0
                push!(edge_node_start,key[1])
                push!(edge_node_end,key[2])
                push!(edge_faces,Int[])
                edge = length(edge_node_start)
                edge_map[key] = edge
            end
            traversal_sign = node_a == key[1] ? Int8(1) : Int8(-1)
            push!(local_edges,edge)
            push!(local_signs,orientation_sign*traversal_sign)
            push!(edge_faces[edge],face)
        end
        face_edges[face] = local_edges
        face_signs[face] = local_signs
    end

    nedge = length(edge_node_start)
    edge_cells_set = [Set{Int}() for _ in 1:nedge]
    for edge in 1:nedge
        for face in edge_faces[edge]
            push!(edge_cells_set[edge],face_L[face])
            right = face_R[face]
            if right <= block.ncell || face_bc_id[face] == 0
                push!(edge_cells_set[edge],right)
            end
        end
    end
    edge_cells = [sort!(collect(cells)) for cells in edge_cells_set]

    face_edge_offset,face_edge_list = _csr_from_lists(face_edges)
    face_edge_sign = reduce(vcat,face_signs;init=Int8[])
    edge_face_offset,edge_face_list = _csr_from_lists(edge_faces)
    edge_cell_offset,edge_cell_list = _csr_from_lists(edge_cells)

    edge_tx = Vector{FT}(undef,nedge)
    edge_ty = Vector{FT}(undef,nedge)
    edge_tz = Vector{FT}(undef,nedge)
    edge_length = Vector{FT}(undef,nedge)
    for edge in 1:nedge
        node_a = edge_node_start[edge]
        node_b = edge_node_end[edge]
        dx = node_x[node_b]-node_x[node_a]
        dy = node_y[node_b]-node_y[node_a]
        dz = node_z[node_b]-node_z[node_a]
        length_value = sqrt(dx*dx+dy*dy+dz*dz)
        length_value > eps(FT) || error("CT edge $edge has zero length")
        edge_length[edge] = length_value
        edge_tx[edge] = dx/length_value
        edge_ty[edge] = dy/length_value
        edge_tz[edge] = dz/length_value
    end

    periodic_edge_group_offset,periodic_edge_group_list,
    periodic_edge_group_sign = _periodic_edge_groups(
        block,face_edges,edge_node_start,edge_node_end,
        edge_tx,edge_ty,edge_tz,edge_length,
        node_x,node_y,node_z,face_cx,face_cy,face_cz,
    )

    block.ct = UnstructCTData(
        nedge,
        GPUArray(edge_node_start),GPUArray(edge_node_end),
        GPUArray(edge_tx),GPUArray(edge_ty),GPUArray(edge_tz),GPUArray(edge_length),
        GPUArray(face_edge_offset),GPUArray(face_edge_list),GPUArray(face_edge_sign),
        GPUArray(edge_face_offset),GPUArray(edge_face_list),
        GPUArray(edge_cell_offset),GPUArray(edge_cell_list),
        gpu_zeros(FT,block.nface),gpu_zeros(FT,block.nface),gpu_zeros(FT,nedge),
        GPUArray(periodic_edge_group_offset),
        GPUArray(periodic_edge_group_list),GPUArray(periodic_edge_group_sign),
    )
    return block.ct
end

function initialize_unstruct_ct_flux_kernel!(
    phi_faces, Q, face_L, face_R, face_nx, face_ny, face_nz, face_area, nface::Int,
)
    face = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    face > nface && return
    @inbounds left = face_L[face]
    @inbounds right = face_R[face]
    @inbounds magnetic_x = FT(0.5)*(Q[left,7]+Q[right,7])
    @inbounds magnetic_y = FT(0.5)*(Q[left,8]+Q[right,8])
    @inbounds magnetic_z = FT(0.5)*(Q[left,9]+Q[right,9])
    @inbounds phi_faces[face] = (
        magnetic_x*face_nx[face]+magnetic_y*face_ny[face]+magnetic_z*face_nz[face]
    )*face_area[face]
    return
end

@inline function _solve_ct_cell_field(m11,m12,m13,m22,m23,m33,b1,b2,b3)
    determinant = m11*(m22*m33-m23*m23)-m12*(m12*m33-m13*m23)+
        m13*(m12*m23-m13*m22)
    if abs(determinant) <= FT(100)*eps(FT)
        return zero(FT),zero(FT),zero(FT)
    end
    inverse_determinant = inv(determinant)
    i11 = (m22*m33-m23*m23)*inverse_determinant
    i12 = (m13*m23-m12*m33)*inverse_determinant
    i13 = (m12*m23-m13*m22)*inverse_determinant
    i22 = (m11*m33-m13*m13)*inverse_determinant
    i23 = (m12*m13-m11*m23)*inverse_determinant
    i33 = (m11*m22-m12*m12)*inverse_determinant
    return i11*b1+i12*b2+i13*b3,
        i12*b1+i22*b2+i23*b3,
        i13*b1+i23*b2+i33*b3
end

function reconstruct_unstruct_ct_B_kernel!(
    U, phi_faces, cell_face_offset, cell_face_list,
    face_area, face_nx, face_ny, face_nz,
    ncell::Int, adjust_energy::Bool,
)
    cell = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    cell > ncell && return
    m11=zero(FT);m12=zero(FT);m13=zero(FT)
    m22=zero(FT);m23=zero(FT);m33=zero(FT)
    b1=zero(FT);b2=zero(FT);b3=zero(FT)
    @inbounds first_face = cell_face_offset[cell]
    @inbounds last_face = cell_face_offset[cell+1]-Int32(1)
    @inbounds for entry in first_face:last_face
        face = cell_face_list[entry]
        area = face_area[face]
        nx=face_nx[face];ny=face_ny[face];nz=face_nz[face]
        weight=area*area
        normal_field=phi_faces[face]/area
        m11+=weight*nx*nx;m12+=weight*nx*ny;m13+=weight*nx*nz
        m22+=weight*ny*ny;m23+=weight*ny*nz;m33+=weight*nz*nz
        b1+=weight*normal_field*nx
        b2+=weight*normal_field*ny
        b3+=weight*normal_field*nz
    end
    magnetic_x,magnetic_y,magnetic_z = _solve_ct_cell_field(
        m11,m12,m13,m22,m23,m33,b1,b2,b3)
    @inbounds if adjust_energy
        old_energy=FT(0.5)*INV_MU0_SI*(
            U[cell,6]^2+U[cell,7]^2+U[cell,8]^2)
        new_energy=FT(0.5)*INV_MU0_SI*(
            magnetic_x^2+magnetic_y^2+magnetic_z^2)
        U[cell,5]+=new_energy-old_energy
    end
    @inbounds U[cell,6]=magnetic_x
    @inbounds U[cell,7]=magnetic_y
    @inbounds U[cell,8]=magnetic_z
    @inbounds U[cell,9]=zero(FT)
    return
end

function compute_unstruct_edge_emf_kernel!(
    edge_emf, F_faces, grad,
    edge_face_offset, edge_face_list, edge_cell_offset, edge_cell_list,
    edge_tx, edge_ty, edge_tz, edge_length,
    face_area, face_nx, face_ny, face_nz, face_bc_id,
    nedge::Int, resistivity::FT,
)
    edge = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    edge > nedge && return
    @inbounds tx=edge_tx[edge];ty=edge_ty[edge];tz=edge_tz[edge]
    ideal_emf=zero(FT)
    @inbounds first_face=edge_face_offset[edge]
    @inbounds last_face=edge_face_offset[edge+1]-Int32(1)
    face_count=last_face-first_face+Int32(1)
    @inbounds for entry in first_face:last_face
        face=edge_face_list[entry]
        inverse_area=inv(face_area[face])
        flux_bx=F_faces[face,6]*inverse_area
        flux_by=F_faces[face,7]*inverse_area
        flux_bz=F_faces[face,8]*inverse_area
        nx=face_nx[face];ny=face_ny[face];nz=face_nz[face]
        electric_x=nz*flux_by-ny*flux_bz
        electric_y=nx*flux_bz-nz*flux_bx
        electric_z=ny*flux_bx-nx*flux_by
        ideal_emf+=electric_x*tx+electric_y*ty+electric_z*tz
    end
    ideal_emf/=max(face_count,Int32(1))

    resistive_emf=zero(FT)
    if resistivity>zero(FT)
        @inbounds first_cell=edge_cell_offset[edge]
        @inbounds last_cell=edge_cell_offset[edge+1]-Int32(1)
        cell_count=last_cell-first_cell+Int32(1)
        @inbounds for entry in first_cell:last_cell
            cell=edge_cell_list[entry]
            gBx_x=grad[cell,7,1];gBx_y=grad[cell,7,2];gBx_z=grad[cell,7,3]
            gBy_x=grad[cell,8,1];gBy_y=grad[cell,8,2];gBy_z=grad[cell,8,3]
            gBz_x=grad[cell,9,1];gBz_y=grad[cell,9,2];gBz_z=grad[cell,9,3]
            for face_entry in first_face:last_face
                boundary_face=edge_face_list[face_entry]
                boundary_id=face_bc_id[boundary_face]
                if boundary_id!=0 && boundary_id!=Int(BC_PERIODIC)
                    bnx=face_nx[boundary_face]
                    bny=face_ny[boundary_face]
                    bnz=face_nz[boundary_face]
                    projection=gBx_x*bnx+gBx_y*bny+gBx_z*bnz
                    gBx_x-=projection*bnx;gBx_y-=projection*bny;gBx_z-=projection*bnz
                    projection=gBy_x*bnx+gBy_y*bny+gBy_z*bnz
                    gBy_x-=projection*bnx;gBy_y-=projection*bny;gBy_z-=projection*bnz
                    projection=gBz_x*bnx+gBz_y*bny+gBz_z*bnz
                    gBz_x-=projection*bnx;gBz_y-=projection*bny;gBz_z-=projection*bnz
                end
            end
            current_x=gBz_y-gBy_z
            current_y=gBx_z-gBz_x
            current_z=gBy_x-gBx_y
            resistive_emf+=current_x*tx+current_y*ty+current_z*tz
        end
        resistive_emf=resistivity*resistive_emf/max(cell_count,Int32(1))
    end
    @inbounds edge_emf[edge]=(ideal_emf+resistive_emf)*edge_length[edge]
    return
end

function synchronize_periodic_edge_emf_kernel!(
    edge_emf,group_offset,group_list,group_sign,ngroup::Int,
)
    group = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    group > ngroup && return
    @inbounds first_entry = group_offset[group]
    @inbounds last_entry = group_offset[group+1]-Int32(1)
    canonical_emf = zero(FT)
    @inbounds for entry in first_entry:last_entry
        canonical_emf += group_sign[entry]*edge_emf[group_list[entry]]
    end
    canonical_emf /= last_entry-first_entry+Int32(1)
    @inbounds for entry in first_entry:last_entry
        edge_emf[group_list[entry]] = group_sign[entry]*canonical_emf
    end
    return
end

function update_unstruct_ct_flux_kernel!(
    phi_faces,phi_backup,edge_emf,face_edge_offset,face_edge_list,face_edge_sign,
    time_step::FT,rk_coefficient::FT,nface::Int,
)
    face=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    face>nface && return
    circulation=zero(FT)
    @inbounds first_edge=face_edge_offset[face]
    @inbounds last_edge=face_edge_offset[face+1]-Int32(1)
    @inbounds for entry in first_edge:last_edge
        circulation+=face_edge_sign[entry]*edge_emf[face_edge_list[entry]]
    end
    @inbounds candidate=phi_faces[face]-time_step*circulation
    @inbounds phi_faces[face]=phi_backup[face]+rk_coefficient*(candidate-phi_backup[face])
    return
end

function initialize_unstruct_ct!(block::UnstructBlock)
    @static if equation_type != :MHD
        throw(ArgumentError("unstructured CT requires equation_type=:MHD"))
    end
    ct=build_unstruct_ct_topology!(block)
    @gpu_launch threads=block.nthreads_face blocks=block.nb_face initialize_unstruct_ct_flux_kernel!(
        ct.phi_faces,block.Q,block.face_L,block.face_R,
        block.face_nx,block.face_ny,block.face_nz,block.face_area,block.nface)
    project_unstruct_ct_divergence!(block)
    copyto!(ct.phi_backup,ct.phi_faces)
    unstruct_reconstruct_ct_B!(block;adjust_energy=true,update_primitive=true)
    return ct
end

@inline function _checked_unstruct_cell_center_distance(
    cell_x, cell_y, cell_z, left::Int, right::Int, face::Int,
)
    dx=cell_x[right]-cell_x[left]
    dy=cell_y[right]-cell_y[left]
    dz=cell_z[right]-cell_z[left]
    distance=sqrt(dx*dx+dy*dy+dz*dz)
    isfinite(distance) && distance > zero(FT) || error(
        "unstructured CT projection found coincident or invalid cell centers " *
        "for face $face (left=$left, right=$right)",
    )
    return distance
end

function project_unstruct_ct_divergence!(block::UnstructBlock;
                                         tolerance::FT=FT(1.0e-12))
    ct=block.ct
    ct===nothing && error("unstructured CT topology has not been initialized")
    phi=Array(ct.phi_faces)
    face_left=Array(block.face_L)
    face_right=Array(block.face_R)
    face_area=Array(block.face_area)
    cell_x=Array(block.cell_cx)
    cell_y=Array(block.cell_cy)
    cell_z=Array(block.cell_cz)
    offsets=Array(block.cell_face_offset)
    faces=Array(block.cell_face_list)
    signs=Array(block.cell_face_sign)
    volumes=Array(block.cell_vol)
    periodic_partner=zeros(Int,block.nface)
    periodic_primary=falses(block.nface)
    for (primary,partner) in zip(
        Array(block.periodic_face_primary),Array(block.periodic_face_partner),
    )
        canonical_flux=FT(0.5)*(phi[primary]-phi[partner])
        phi[primary]=canonical_flux
        phi[partner]=-canonical_flux
        periodic_partner[primary]=partner
        periodic_partner[partner]=primary
        periodic_primary[primary]=true
    end
    residual=zeros(FT,block.ncell)
    for cell in 1:block.ncell
        for entry in offsets[cell]:(offsets[cell+1]-1)
            residual[cell]+=signs[entry]*phi[faces[entry]]
        end
    end
    before=maximum(abs.(residual ./ volumes[1:block.ncell]))
    before<=tolerance && return (before=before,after=before,projected=false)

    rows=Int[];columns=Int[];values=FT[]
    conductance=zeros(FT,block.nface)
    has_dirichlet_boundary=false
    for face in 1:block.nface
        left=face_left[face]
        left<=block.ncell || continue
        if periodic_partner[face]!=0 && !periodic_primary[face]
            continue
        end
        right=face_right[face]
        distance=_checked_unstruct_cell_center_distance(
            cell_x, cell_y, cell_z, left, right, face,
        )
        weight=face_area[face]/distance
        conductance[face]=weight
        push!(rows,left);push!(columns,left);push!(values,weight)
        coupled_cell = periodic_primary[face] ?
            face_left[periodic_partner[face]] : right
        if coupled_cell<=block.ncell
            push!(rows,coupled_cell);push!(columns,coupled_cell);push!(values,weight)
            push!(rows,left);push!(columns,coupled_cell);push!(values,-weight)
            push!(rows,coupled_cell);push!(columns,left);push!(values,-weight)
        else
            has_dirichlet_boundary=true
        end
    end
    laplacian=sparse(rows,columns,values,block.ncell,block.ncell)
    projection_rhs=-residual
    if !has_dirichlet_boundary
        laplacian[1,:].=zero(FT)
        laplacian[:,1].=zero(FT)
        laplacian[1,1]=one(FT)
        projection_rhs[1]=zero(FT)
    end
    potential=laplacian\projection_rhs
    for face in 1:block.nface
        left=face_left[face]
        left<=block.ncell || continue
        if periodic_partner[face]!=0
            periodic_primary[face] || continue
            partner=periodic_partner[face]
            right=face_left[partner]
            phi[face]+=conductance[face]*(potential[left]-potential[right])
            phi[partner]=-phi[face]
            continue
        end
        right=face_right[face]
        right_potential=right<=block.ncell ? potential[right] : zero(FT)
        phi[face]+=conductance[face]*(potential[left]-right_potential)
    end
    copyto!(ct.phi_faces,phi)

    fill!(residual,zero(FT))
    for cell in 1:block.ncell
        for entry in offsets[cell]:(offsets[cell+1]-1)
            residual[cell]+=signs[entry]*phi[faces[entry]]
        end
    end
    after=maximum(abs.(residual ./ volumes[1:block.ncell]))
    after<=max(tolerance,FT(1000)*eps(FT)*max(before,one(FT))) || error(
        "unstructured CT projection failed: divB $before -> $after")
    return (before=before,after=after,projected=true)
end

function unstruct_reconstruct_ct_B!(block::UnstructBlock;
                                    adjust_energy::Bool=false,
                                    update_primitive::Bool=false)
    ct=block.ct
    ct===nothing && error("unstructured CT topology has not been initialized")
    blocks=cld(block.ncell,block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=blocks reconstruct_unstruct_ct_B_kernel!(
        block.U,ct.phi_faces,block.cell_face_offset,block.cell_face_list,
        block.face_area,block.face_nx,block.face_ny,block.face_nz,
        block.ncell,adjust_energy)
    update_primitive && unstruct_c2prim!(block)
    return
end

function backup_unstruct_ct!(block::UnstructBlock)
    ct=block.ct
    ct===nothing && error("unstructured CT topology has not been initialized")
    copyto!(ct.phi_backup,ct.phi_faces)
    return
end

function unstruct_ct_stage!(block::UnstructBlock,time_step::FT,rk_coefficient::FT;
                            comm_cart=nothing)
    ct=block.ct
    ct===nothing && error("unstructured CT topology has not been initialized")
    resistivity=resistive ? FT(η_mhd) : zero(FT)
    edge_blocks=cld(ct.nedge,block.nthreads_face)
    @gpu_launch threads=block.nthreads_face blocks=edge_blocks compute_unstruct_edge_emf_kernel!(
        ct.edge_emf,block.F_faces,block.grad,
        ct.edge_face_offset,ct.edge_face_list,ct.edge_cell_offset,ct.edge_cell_list,
        ct.edge_tx,ct.edge_ty,ct.edge_tz,ct.edge_length,
        block.face_area,block.face_nx,block.face_ny,block.face_nz,
        block.face_bc_id,
        ct.nedge,resistivity)
    periodic_group_count = length(ct.periodic_edge_group_offset)-1
    if periodic_group_count > 0
        group_blocks = cld(periodic_group_count,block.nthreads_face)
        @gpu_launch threads=block.nthreads_face blocks=group_blocks synchronize_periodic_edge_emf_kernel!(
            ct.edge_emf,ct.periodic_edge_group_offset,
            ct.periodic_edge_group_list,ct.periodic_edge_group_sign,
            periodic_group_count)
    end
    comm_cart === nothing || sync_unstruct_ct_edge_emf!(block,comm_cart)
    @gpu_launch threads=block.nthreads_face blocks=block.nb_face update_unstruct_ct_flux_kernel!(
        ct.phi_faces,ct.phi_backup,ct.edge_emf,
        ct.face_edge_offset,ct.face_edge_list,ct.face_edge_sign,
        time_step,rk_coefficient,block.nface)
    unstruct_reconstruct_ct_B!(block)
    return
end

function unstruct_ct_divergence(block::UnstructBlock)
    ct=block.ct
    ct===nothing && error("unstructured CT topology has not been initialized")
    phi=Array(ct.phi_faces)
    offsets=Array(block.cell_face_offset)
    faces=Array(block.cell_face_list)
    signs=Array(block.cell_face_sign)
    volume=Array(block.cell_vol)
    divergence=zeros(FT,block.ncell)
    for cell in 1:block.ncell
        for entry in offsets[cell]:(offsets[cell+1]-1)
            divergence[cell]+=signs[entry]*phi[faces[entry]]
        end
        divergence[cell]/=volume[cell]
    end
    return divergence
end
