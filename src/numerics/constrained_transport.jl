# ═══════════════════════════════════════════════════════════════════════
# ct.jl - Constrained Transport for MHD divergence cleaning
# ═══════════════════════════════════════════════════════════════════════
# Metric-aware CT state:
#   face arrays: oriented magnetic flux Phi_B = (B dot n) A
#   edge arrays: oriented line EMF = integral(E dot dl)
#   update: discrete Stokes theorem, so div(curl)=0 topologically
# ═══════════════════════════════════════════════════════════════════════

# ─── CT: Initialize oriented face magnetic flux from cell B ───
if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end
if !@isdefined(STRUCTURED_FACE_QUADRATURE_LOADED)
    include(joinpath(@__DIR__, "structured_face_quadrature.jl"))
end

function ct_init_face_b_kernel!(
    Bx_face, By_face, Bz_face, Q,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    nxp, nyp, nzp, physical_face_mask,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    # Bx_face: i=1..nxp+1 (x faces), j=1..nyp (cell centers), k=1..nzp
    # By_face: i=1..nxp (cell centers), j=1..nyp+1 (y faces), k=1..nzp
    # Bz_face: i=1..nxp, j=1..nyp, k=1..nzp+1 (z faces)
    if i >= Int32(1) && i <= nxp + Int32(1) && j >= Int32(1) && j <= nyp + Int32(1) && k >= Int32(1) && k <= nzp + Int32(1)
        ii = i + NG; jj = j + NG; kk = k + NG
        # Despite their legacy names, face arrays store Phi_B = (B dot n) A.
        if i <= nxp + Int32(1) && j <= nyp && k <= nzp
            @inbounds begin
                left_i = ii - Int32(1)
                right_i = ii
                if i == Int32(1) &&
                   (physical_face_mask & Int32(0x01)) != Int32(0)
                    left_i = right_i
                elseif i == nxp + Int32(1) &&
                       (physical_face_mask & Int32(0x02)) != Int32(0)
                    right_i = left_i
                end
                bx = FT(0.5) * (Q[right_i, jj, kk, QBX] + Q[left_i, jj, kk, QBX])
                by = FT(0.5) * (Q[right_i, jj, kk, QBY] + Q[left_i, jj, kk, QBY])
                bz = FT(0.5) * (Q[right_i, jj, kk, QBZ] + Q[left_i, jj, kk, QBZ])
                Bx_face[ii, jj, kk] = Areai[ii, jj, kk] *
                    (bx*nxi[ii, jj, kk] + by*nyi[ii, jj, kk] + bz*nzi[ii, jj, kk])
            end
        end
        if i <= nxp && j <= nyp + Int32(1) && k <= nzp
            @inbounds begin
                left_j = jj - Int32(1)
                right_j = jj
                if j == Int32(1) &&
                   (physical_face_mask & Int32(0x04)) != Int32(0)
                    left_j = right_j
                elseif j == nyp + Int32(1) &&
                       (physical_face_mask & Int32(0x08)) != Int32(0)
                    right_j = left_j
                end
                bx = FT(0.5) * (Q[ii, right_j, kk, QBX] + Q[ii, left_j, kk, QBX])
                by = FT(0.5) * (Q[ii, right_j, kk, QBY] + Q[ii, left_j, kk, QBY])
                bz = FT(0.5) * (Q[ii, right_j, kk, QBZ] + Q[ii, left_j, kk, QBZ])
                By_face[ii, jj, kk] = Areaj[ii, jj, kk] *
                    (bx*nxj[ii, jj, kk] + by*nyj[ii, jj, kk] + bz*nzj[ii, jj, kk])
            end
        end
        if i <= nxp && j <= nyp && k <= nzp + Int32(1)
            @inbounds begin
                left_k = kk - Int32(1)
                right_k = kk
                if k == Int32(1) &&
                   (physical_face_mask & Int32(0x10)) != Int32(0)
                    left_k = right_k
                elseif k == nzp + Int32(1) &&
                       (physical_face_mask & Int32(0x20)) != Int32(0)
                    right_k = left_k
                end
                bx = FT(0.5) * (Q[ii, jj, right_k, QBX] + Q[ii, jj, left_k, QBX])
                by = FT(0.5) * (Q[ii, jj, right_k, QBY] + Q[ii, jj, left_k, QBY])
                bz = FT(0.5) * (Q[ii, jj, right_k, QBZ] + Q[ii, jj, left_k, QBZ])
                Bz_face[ii, jj, kk] = Areak[ii, jj, kk] *
                    (bx*nxk[ii, jj, kk] + by*nyk[ii, jj, kk] + bz*nzk[ii, jj, kk])
            end
        end
    end
    return
end

function ct_init_face_b!(
    b, nxp, nyp, nzp; physical_face_mask::Integer=0,
)
    # Launch over the union of all face ranges: (nxp+1) x (nyp+1) x (nzp+1)
    nb = (cld(nxp + 1 + 2*NG, nthreads[1]), cld(nyp + 1 + 2*NG, nthreads[2]), cld(nzp + 1 + 2*NG, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb ct_init_face_b_kernel!(
        b.Bx_face, b.By_face, b.Bz_face, b.Q,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        Int32(nxp), Int32(nyp), Int32(nzp), Int32(physical_face_mask))
end

@inline _ct_initial_host_coordinate(array::Array) = array
@inline _ct_initial_host_coordinate(array) = Array(array)

function ct_initial_coordinate_arrays(b, coordinates=nothing)
    source = coordinates === nothing ? b : coordinates
    all(name -> hasproperty(source, name), (:x, :y, :z)) ||
        throw(ArgumentError(
            "CT initial coordinates must provide x, y and z node arrays",
        ))
    return (
        _ct_initial_host_coordinate(getproperty(source, :x)),
        _ct_initial_host_coordinate(getproperty(source, :y)),
        _ct_initial_host_coordinate(getproperty(source, :z)),
    )
end

function ct_initial_edge_line_integrals_from_vector_potential!(
    b, vector_potential;
    time=zero(FT), junction_edge_mask::Integer=0,
    coordinates=nothing, junction_fallback::Bool=true,
)
    x, y, z = ct_initial_coordinate_arrays(b, coordinates)
    size(x) == size(y) == size(z) || throw(DimensionMismatch(
        "CT node-coordinate arrays must have identical sizes",
    ))
    singularity_edges = junction_fallback ?
        structured_metric_singularity_edges_from_mask(junction_edge_mask) :
        nothing
    edge_x, edge_y, edge_z = structured_scmm_edge_line_integrals(
        vector_potential, x, y, z;
        time=FT(time), singularity_edges=singularity_edges,
        physical_dims=(b.Nx, b.Ny, b.Nz), ng=NG,
    )
    copyto!(
        b.Ex_edge,
        GPUArray(Array(@view(
            edge_x[
                NG+1:NG+b.Nx,
                NG+1:NG+b.Ny+1,
                NG+1:NG+b.Nz+1,
            ]
        ))),
    )
    copyto!(
        b.Ey_edge,
        GPUArray(Array(@view(
            edge_y[
                NG+1:NG+b.Nx+1,
                NG+1:NG+b.Ny,
                NG+1:NG+b.Nz+1,
            ]
        ))),
    )
    copyto!(
        b.Ez_edge,
        GPUArray(Array(@view(
            edge_z[
                NG+1:NG+b.Nx+1,
                NG+1:NG+b.Ny+1,
                NG+1:NG+b.Nz,
            ]
        ))),
    )
    return nothing
end

function ct_face_flux_from_edge_integrals_kernel!(
    Bx_face, By_face, Bz_face, Ex_edge, Ey_edge, Ez_edge,
    nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ii, jj, kk = i + NG, j + NG, k + NG
    if i <= nxp + Int32(1) && j <= nyp && k <= nzp
        @inbounds Bx_face[ii,jj,kk] =
            Ey_edge[i,j,k] + Ez_edge[i,j+Int32(1),k] -
            Ey_edge[i,j,k+Int32(1)] - Ez_edge[i,j,k]
    end
    if i <= nxp && j <= nyp + Int32(1) && k <= nzp
        @inbounds By_face[ii,jj,kk] =
            Ez_edge[i,j,k] + Ex_edge[i,j,k+Int32(1)] -
            Ez_edge[i+Int32(1),j,k] - Ex_edge[i,j,k]
    end
    if i <= nxp && j <= nyp && k <= nzp + Int32(1)
        @inbounds Bz_face[ii,jj,kk] =
            Ex_edge[i,j,k] + Ey_edge[i+Int32(1),j,k] -
            Ex_edge[i,j+Int32(1),k] - Ey_edge[i,j,k]
    end
    return nothing
end

function ct_face_flux_from_edge_integrals!(b, nxp=b.Nx, nyp=b.Ny, nzp=b.Nz)
    nb = (
        cld(nxp + 1, nthreads[1]),
        cld(nyp + 1, nthreads[2]),
        cld(nzp + 1, nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=nb ct_face_flux_from_edge_integrals_kernel!(
        b.Bx_face, b.By_face, b.Bz_face,
        b.Ex_edge, b.Ey_edge, b.Ez_edge,
        Int32(nxp), Int32(nyp), Int32(nzp),
    )
    return nothing
end

function ct_initial_face_flux_from_vector_potential!(
    b, vector_potential;
    time=zero(FT), junction_edge_mask::Integer=0,
    coordinates=nothing, junction_fallback::Bool=true,
)
    ct_initial_edge_line_integrals_from_vector_potential!(
        b, vector_potential;
        time=time, junction_edge_mask=junction_edge_mask,
        coordinates=coordinates, junction_fallback=junction_fallback,
    )
    ct_face_flux_from_edge_integrals!(b)
    return nothing
end

function ct_initial_face_flux_arrays(b)
    i_cells = NG+1:NG+b.Nx
    j_cells = NG+1:NG+b.Ny
    k_cells = NG+1:NG+b.Nz
    return (
        Array(@view(b.Bx_face[NG+1:NG+b.Nx+1, j_cells, k_cells])),
        Array(@view(b.By_face[i_cells, NG+1:NG+b.Ny+1, k_cells])),
        Array(@view(b.Bz_face[i_cells, j_cells, NG+1:NG+b.Nz+1])),
    )
end

function ct_initial_relative_face_divergence(b)
    phi_x, phi_y, phi_z = ct_initial_face_flux_arrays(b)
    return ct_relative_face_divergence(phi_x, phi_y, phi_z)
end

function ct_project_initial_face_b!(b, periodic; tolerance)
    phi_x, phi_y, phi_z = ct_initial_face_flux_arrays(b)
    result = ct_project_face_flux_divergence!(
        phi_x, phi_y, phi_z;
        periodic=periodic, tolerance=tolerance,
    )
    i_cells = NG+1:NG+b.Nx
    j_cells = NG+1:NG+b.Ny
    k_cells = NG+1:NG+b.Nz
    copyto!(
        @view(b.Bx_face[NG+1:NG+b.Nx+1, j_cells, k_cells]),
        GPUArray(phi_x),
    )
    copyto!(
        @view(b.By_face[i_cells, NG+1:NG+b.Ny+1, k_cells]),
        GPUArray(phi_y),
    )
    copyto!(
        @view(b.Bz_face[i_cells, j_cells, NG+1:NG+b.Nz+1]),
        GPUArray(phi_z),
    )
    return result
end

# Metric-aware SG07 helpers operate on face magnetic fluxes and edge line EMFs.

@inline function ct_xface_edge_data(
    flux, rho_sum, U_stage, Q_stage, Areai, nxi, nyi, nzi, Vol,
    face_index, tangent_x, tangent_y, tangent_z, dt,
    flux_halo::Int32=Int32(1),
)
    face_i, cell_j, cell_k = face_index
    ng = Int32(NG)
    gi, gj, gk = face_i+ng, cell_j+ng-flux_halo, cell_k+ng-flux_halo
    @inbounds begin
        area = Areai[gi, gj, gk]
        normal_x = nxi[gi, gj, gk]
        normal_y = nyi[gi, gj, gk]
        normal_z = nzi[gi, gj, gk]
        face_normal_emf = ct_face_normal_emf_from_state_pair(
            U_stage, Q_stage,
            gi-Int32(1), gj, gk,
            gi, gj, gk,
            normal_x, normal_y, normal_z,
        )
        emf = ct_project_face_flux_to_edge_emf(
            flux[face_i, cell_j, cell_k, UBX],
            flux[face_i, cell_j, cell_k, UBY],
            flux[face_i, cell_j, cell_k, UBZ],
            normal_x, normal_y, normal_z,
            tangent_x, tangent_y, tangent_z, area, face_normal_emf,
        )
        density_flux = flux[face_i, cell_j, cell_k, 1] / area
        spacing = ct_face_normal_spacing(
            Vol[gi-Int32(1), gj, gk], Vol[gi, gj, gk], area,
        )
        weight = ct_emf_weight_from_density_sum(
            density_flux, rho_sum[face_i, cell_j, cell_k], spacing, dt,
        )
    end
    return emf, weight
end

@inline function ct_yface_edge_data(
    flux, rho_sum, U_stage, Q_stage, Areaj, nxj, nyj, nzj, Vol,
    face_index, tangent_x, tangent_y, tangent_z, dt,
    flux_halo::Int32=Int32(1),
)
    cell_i, face_j, cell_k = face_index
    ng = Int32(NG)
    gi, gj, gk = cell_i+ng-flux_halo, face_j+ng, cell_k+ng-flux_halo
    @inbounds begin
        area = Areaj[gi, gj, gk]
        normal_x = nxj[gi, gj, gk]
        normal_y = nyj[gi, gj, gk]
        normal_z = nzj[gi, gj, gk]
        face_normal_emf = ct_face_normal_emf_from_state_pair(
            U_stage, Q_stage,
            gi, gj-Int32(1), gk,
            gi, gj, gk,
            normal_x, normal_y, normal_z,
        )
        emf = ct_project_face_flux_to_edge_emf(
            flux[cell_i, face_j, cell_k, UBX],
            flux[cell_i, face_j, cell_k, UBY],
            flux[cell_i, face_j, cell_k, UBZ],
            normal_x, normal_y, normal_z,
            tangent_x, tangent_y, tangent_z, area, face_normal_emf,
        )
        density_flux = flux[cell_i, face_j, cell_k, 1] / area
        spacing = ct_face_normal_spacing(
            Vol[gi, gj-Int32(1), gk], Vol[gi, gj, gk], area,
        )
        weight = ct_emf_weight_from_density_sum(
            density_flux, rho_sum[cell_i, face_j, cell_k], spacing, dt,
        )
    end
    return emf, weight
end

@inline function ct_zface_edge_data(
    flux, rho_sum, U_stage, Q_stage, Areak, nxk, nyk, nzk, Vol,
    face_index, tangent_x, tangent_y, tangent_z, dt,
    flux_halo::Int32=Int32(1),
)
    cell_i, cell_j, face_k = face_index
    ng = Int32(NG)
    gi, gj, gk = cell_i+ng-flux_halo, cell_j+ng-flux_halo, face_k+ng
    @inbounds begin
        area = Areak[gi, gj, gk]
        normal_x = nxk[gi, gj, gk]
        normal_y = nyk[gi, gj, gk]
        normal_z = nzk[gi, gj, gk]
        face_normal_emf = ct_face_normal_emf_from_state_pair(
            U_stage, Q_stage,
            gi, gj, gk-Int32(1),
            gi, gj, gk,
            normal_x, normal_y, normal_z,
        )
        emf = ct_project_face_flux_to_edge_emf(
            flux[cell_i, cell_j, face_k, UBX],
            flux[cell_i, cell_j, face_k, UBY],
            flux[cell_i, cell_j, face_k, UBZ],
            normal_x, normal_y, normal_z,
            tangent_x, tangent_y, tangent_z, area, face_normal_emf,
        )
        density_flux = flux[cell_i, cell_j, face_k, 1] / area
        spacing = ct_face_normal_spacing(
            Vol[gi, gj, gk-Int32(1)], Vol[gi, gj, gk], area,
        )
        weight = ct_emf_weight_from_density_sum(
            density_flux, rho_sum[cell_i, cell_j, face_k], spacing, dt,
        )
    end
    return emf, weight
end

@inline _ct_junction_face_axis(face::Int32) = (face + Int32(1)) ÷ Int32(2)

@inline function _ct_junction_dimension(axis, nxp, nyp, nzp)
    return axis == Int32(1) ? nxp : (axis == Int32(2) ? nyp : nzp)
end

@inline function _ct_junction_face_for_axis(face1, face2, axis)
    return _ct_junction_face_axis(face1) == axis ? face1 : face2
end

@inline function _ct_junction_boundary_node(face, extent)
    return isodd(face) ? Int32(1) : extent + Int32(1)
end

@inline function _ct_junction_boundary_cell(face, extent)
    return isodd(face) ? Int32(1) : extent
end

@inline function _ct_junction_inward_face_normal(
    face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    cell_i, cell_j, cell_k, nxp, nyp, nzp,
)
    normal_axis = _ct_junction_face_axis(face)
    halo = Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    if normal_axis == Int32(1)
        face_i = _ct_junction_boundary_node(face, nxp)
        gi, gj, gk = ct_xface_geometry_indices(
            face_i, cell_j + halo, cell_k + halo, Int32(NG),
        )
        @inbounds normal_x, normal_y, normal_z =
            nxi[gi,gj,gk], nyi[gi,gj,gk], nzi[gi,gj,gk]
    elseif normal_axis == Int32(2)
        face_j = _ct_junction_boundary_node(face, nyp)
        gi, gj, gk = ct_yface_geometry_indices(
            cell_i + halo, face_j, cell_k + halo, Int32(NG),
        )
        @inbounds normal_x, normal_y, normal_z =
            nxj[gi,gj,gk], nyj[gi,gj,gk], nzj[gi,gj,gk]
    else
        face_k = _ct_junction_boundary_node(face, nzp)
        gi, gj, gk = ct_zface_geometry_indices(
            cell_i + halo, cell_j + halo, face_k, Int32(NG),
        )
        @inbounds normal_x, normal_y, normal_z =
            nxk[gi,gj,gk], nyk[gi,gj,gk], nzk[gi,gj,gk]
    end
    # Metric normals point in the positive computational direction.  At a
    # low/high boundary the inward physical normal therefore has sign +/-.
    inward_sign = isodd(face) ? one(normal_x) : -one(normal_x)
    return inward_sign*normal_x, inward_sign*normal_y,
           inward_sign*normal_z
end

@inline function _ct_junction_sector_angle(
    normal1_x, normal1_y, normal1_z,
    normal2_x, normal2_y, normal2_z,
    tangent_x, tangent_y, tangent_z,
)
    tangent_dot_1 = normal1_x*tangent_x + normal1_y*tangent_y +
                    normal1_z*tangent_z
    tangent_dot_2 = normal2_x*tangent_x + normal2_y*tangent_y +
                    normal2_z*tangent_z
    projected1_x = normal1_x - tangent_dot_1*tangent_x
    projected1_y = normal1_y - tangent_dot_1*tangent_y
    projected1_z = normal1_z - tangent_dot_1*tangent_z
    projected2_x = normal2_x - tangent_dot_2*tangent_x
    projected2_y = normal2_y - tangent_dot_2*tangent_y
    projected2_z = normal2_z - tangent_dot_2*tangent_z
    norm_product = sqrt(
        (projected1_x^2 + projected1_y^2 + projected1_z^2) *
        (projected2_x^2 + projected2_y^2 + projected2_z^2),
    )
    norm_product > eps(typeof(norm_product)) ||
        return oftype(norm_product, NaN)
    cosine = (projected1_x*projected2_x +
              projected1_y*projected2_y +
              projected1_z*projected2_z) / norm_product
    cosine = max(-one(cosine), min(one(cosine), cosine))
    return oftype(cosine, pi) - acos(cosine)
end

@inline function _ct_junction_face_edge_data(
    face,
    Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z,
    U_stage, Q_stage,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    Vol, cell_i, cell_j, cell_k,
    tangent_x, tangent_y, tangent_z, dt, nxp, nyp, nzp,
)
    normal_axis = _ct_junction_face_axis(face)
    halo = Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    if normal_axis == Int32(1)
        face_i = _ct_junction_boundary_node(face, nxp)
        return ct_xface_edge_data(
            Fx, rho_sum_x, U_stage, Q_stage,
            Areai, nxi, nyi, nzi, Vol,
            (face_i, cell_j + halo, cell_k + halo),
            tangent_x, tangent_y, tangent_z, dt,
        )
    elseif normal_axis == Int32(2)
        face_j = _ct_junction_boundary_node(face, nyp)
        return ct_yface_edge_data(
            Fy, rho_sum_y, U_stage, Q_stage,
            Areaj, nxj, nyj, nzj, Vol,
            (cell_i + halo, face_j, cell_k + halo),
            tangent_x, tangent_y, tangent_z, dt,
        )
    end
    face_k = _ct_junction_boundary_node(face, nzp)
    return ct_zface_edge_data(
        Fz, rho_sum_z, U_stage, Q_stage,
        Areak, nxk, nyk, nzk, Vol,
        (cell_i + halo, cell_j + halo, face_k),
        tangent_x, tangent_y, tangent_z, dt,
    )
end

function ct_pack_generalized_junction_payload_kernel!(
    payload,
    Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z,
    U_stage, Q_stage, fofc_flags,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    Vol, x, y, z, dt,
    edge_axis, face1, face2,
    edge_orientation, face1_normal_orientation,
    face2_normal_orientation, nxp, nyp, nzp,
)
    segment = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    segment_extent = _ct_junction_dimension(edge_axis, nxp, nyp, nzp)
    segment > segment_extent && return

    face_x = _ct_junction_face_for_axis(face1, face2, Int32(1))
    face_y = _ct_junction_face_for_axis(face1, face2, Int32(2))
    face_z = _ct_junction_face_for_axis(face1, face2, Int32(3))
    edge_i = edge_axis == Int32(1) ? segment :
        _ct_junction_boundary_node(face_x, nxp)
    edge_j = edge_axis == Int32(2) ? segment :
        _ct_junction_boundary_node(face_y, nyp)
    edge_k = edge_axis == Int32(3) ? segment :
        _ct_junction_boundary_node(face_z, nzp)
    cell_i = edge_axis == Int32(1) ? segment :
        _ct_junction_boundary_cell(face_x, nxp)
    cell_j = edge_axis == Int32(2) ? segment :
        _ct_junction_boundary_cell(face_y, nyp)
    cell_k = edge_axis == Int32(3) ? segment :
        _ct_junction_boundary_cell(face_z, nzp)

    ng = Int32(NG)
    ni, nj, nk = edge_i + ng, edge_j + ng, edge_k + ng
    next_i = edge_axis == Int32(1) ? ni + Int32(1) : ni
    next_j = edge_axis == Int32(2) ? nj + Int32(1) : nj
    next_k = edge_axis == Int32(3) ? nk + Int32(1) : nk
    @inbounds tangent_x, tangent_y, tangent_z, edge_length = ct_edge_geometry(
        x[next_i, next_j, next_k] - x[ni, nj, nk],
        y[next_i, next_j, next_k] - y[ni, nj, nk],
        z[next_i, next_j, next_k] - z[ni, nj, nk],
    )
    normal1_x, normal1_y, normal1_z = _ct_junction_inward_face_normal(
        face1,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        cell_i, cell_j, cell_k, nxp, nyp, nzp,
    )
    normal2_x, normal2_y, normal2_z = _ct_junction_inward_face_normal(
        face2,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        cell_i, cell_j, cell_k, nxp, nyp, nzp,
    )
    sector_angle = _ct_junction_sector_angle(
        normal1_x, normal1_y, normal1_z,
        normal2_x, normal2_y, normal2_z,
        tangent_x, tangent_y, tangent_z,
    )
    cell_emf = ct_cell_centered_edge_emf_from_state(
        U_stage, Q_stage,
        cell_i + ng, cell_j + ng, cell_k + ng,
        tangent_x, tangent_y, tangent_z,
    )
    face1_emf, face1_weight = _ct_junction_face_edge_data(
        face1,
        Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z,
        U_stage, Q_stage,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        Vol, cell_i, cell_j, cell_k,
        tangent_x, tangent_y, tangent_z, dt, nxp, nyp, nzp,
    )
    face2_emf, face2_weight = _ct_junction_face_edge_data(
        face2,
        Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z,
        U_stage, Q_stage,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        Vol, cell_i, cell_j, cell_k,
        tangent_x, tangent_y, tangent_z, dt, nxp, nyp, nzp,
    )

    canonical_segment = edge_orientation < 0 ?
        segment_extent - segment + Int32(1) : segment
    edge_sign = edge_orientation < 0 ? -one(cell_emf) : one(cell_emf)
    @inbounds begin
        payload[CT_JUNCTION_PAYLOAD_CELL_EMF, canonical_segment] =
            edge_sign * cell_emf
        payload[CT_JUNCTION_PAYLOAD_FACE1_EMF, canonical_segment] =
            edge_sign * face1_emf
        payload[CT_JUNCTION_PAYLOAD_FACE1_WEIGHT, canonical_segment] =
            face1_normal_orientation < 0 ?
            one(face1_weight) - face1_weight : face1_weight
        payload[CT_JUNCTION_PAYLOAD_FACE2_EMF, canonical_segment] =
            edge_sign * face2_emf
        payload[CT_JUNCTION_PAYLOAD_FACE2_WEIGHT, canonical_segment] =
            face2_normal_orientation < 0 ?
            one(face2_weight) - face2_weight : face2_weight
        payload[CT_JUNCTION_PAYLOAD_EDGE_LENGTH, canonical_segment] =
            edge_length
        payload[CT_JUNCTION_PAYLOAD_RESISTIVE_CELL_EMF,
                canonical_segment] = zero(cell_emf)
        payload[CT_JUNCTION_PAYLOAD_RESISTIVE_FACE1_EMF,
                canonical_segment] = zero(cell_emf)
        payload[CT_JUNCTION_PAYLOAD_RESISTIVE_FACE2_EMF,
                canonical_segment] = zero(cell_emf)
        payload[CT_JUNCTION_PAYLOAD_SECTOR_ANGLE, canonical_segment] =
            sector_angle
        flag_value = fofc_flags === nothing ? -one(cell_emf) :
            fofc_flags[cell_i+ng,cell_j+ng,cell_k+ng,1]
        payload[CT_JUNCTION_PAYLOAD_FOFC_SCALE, canonical_segment] =
            ct_fofc_flag_scale(flag_value)
    end
    return
end

function ct_launch_generalized_junction_payload!(
    payload, b,
    Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z, dt,
    edge_axis, face1, face2, edge_orientation,
    face1_normal_orientation, face2_normal_orientation,
)
    segment_count = _ct_junction_dimension(
        Int32(edge_axis), Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
    )
    thread_count = Int32(min(256, max(1, segment_count)))
    threads = (thread_count, Int32(1), Int32(1))
    blocks = (
        Int32(cld(segment_count, thread_count)), Int32(1), Int32(1),
    )
    fofc_flags = hasproperty(b,:fofc_flag) ? b.fofc_flag : nothing
    @gpu_launch threads=threads blocks=blocks ct_pack_generalized_junction_payload_kernel!(
        payload,
        Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z,
        b.U, b.Q, fofc_flags,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        b.Vol, b.x, b.y, b.z, FT(dt),
        Int32(edge_axis), Int32(face1), Int32(face2),
        Int8(edge_orientation),
        Int8(face1_normal_orientation), Int8(face2_normal_orientation),
        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
    )
    return nothing
end

# Metric-aware SG07 operator. Face fluxes are projected onto a common physical
# edge tangent, and the result is stored as an oriented line-integrated EMF.
@inline function ct_junction_edge_masked(
    mask::UInt16, axis::Int32, i, j, k, nxp, nyp, nzp,
)
    iszero(mask) && return false
    first_high = false
    second_high = false
    if axis == Int32(1)
        (j == Int32(1) || j == nyp + Int32(1)) || return false
        (k == Int32(1) || k == nzp + Int32(1)) || return false
        first_high = j == nyp + Int32(1)
        second_high = k == nzp + Int32(1)
    elseif axis == Int32(2)
        (i == Int32(1) || i == nxp + Int32(1)) || return false
        (k == Int32(1) || k == nzp + Int32(1)) || return false
        first_high = i == nxp + Int32(1)
        second_high = k == nzp + Int32(1)
    else
        (i == Int32(1) || i == nxp + Int32(1)) || return false
        (j == Int32(1) || j == nyp + Int32(1)) || return false
        first_high = i == nxp + Int32(1)
        second_high = j == nyp + Int32(1)
    end
    bit_index = Int32(4) * (axis - Int32(1)) +
        Int32(2) * Int32(first_high) + Int32(second_high)
    return (mask & (UInt16(1) << bit_index)) != UInt16(0)
end

@inline ct_fofc_edge_activity(::Nothing, axis, i, j, k) = (false,one(FT))

@inline function ct_fofc_edge_activity(flags, axis::Int32, i, j, k)
    ng = Int32(NG)
    if axis == Int32(1)
        cells = (
            (i,j-Int32(1),k-Int32(1)),(i,j,k-Int32(1)),
            (i,j-Int32(1),k),(i,j,k),
        )
    elseif axis == Int32(2)
        cells = (
            (i-Int32(1),j,k-Int32(1)),(i,j,k-Int32(1)),
            (i-Int32(1),j,k),(i,j,k),
        )
    else
        cells = (
            (i-Int32(1),j-Int32(1),k),(i,j-Int32(1),k),
            (i-Int32(1),j,k),(i,j,k),
        )
    end
    active = false
    scale = one(FT)
    for cell in cells
        @inbounds value = flags[cell[1]+ng,cell[2]+ng,cell[3]+ng,1]
        ct_fofc_flag_is_active(value) || continue
        active = true
        scale = min(scale,ct_fofc_flag_scale(value))
    end
    return active,scale
end

@inline ct_fofc_edge_flagged(::Nothing, axis, i, j, k) = false

@inline function ct_fofc_edge_flagged(flags, axis, i, j, k)
    active,_ = ct_fofc_edge_activity(flags,axis,i,j,k)
    return active
end

@inline ct_shock_edge_flagged(
    ::Nothing, axis::Int32, i, j, k, threshold,
) = false

@inline function ct_shock_edge_flagged(
    sensor, axis::Int32, i, j, k, threshold,
)
    ng = Int32(NG)
    if axis == Int32(1)
        cells = (
            (i,j-Int32(1),k-Int32(1)),(i,j,k-Int32(1)),
            (i,j-Int32(1),k),(i,j,k),
        )
    elseif axis == Int32(2)
        cells = (
            (i-Int32(1),j,k-Int32(1)),(i,j,k-Int32(1)),
            (i-Int32(1),j,k),(i,j,k),
        )
    else
        cells = (
            (i-Int32(1),j-Int32(1),k),(i,j-Int32(1),k),
            (i-Int32(1),j,k),(i,j,k),
        )
    end
    support_max = zero(eltype(sensor))
    for cell in cells
        @inbounds value=sensor[cell[1]+ng,cell[2]+ng,cell[3]+ng]
        isfinite(value) || return true
        support_max = max(support_max,value)
    end
    return support_max >= threshold
end

@inline function ct_edge_uses_sg07(
    sensor, fofc_flags, shock_threshold, axis, i, j, k,
)
    return ct_shock_edge_flagged(
        sensor,axis,i,j,k,shock_threshold,
    ) || ct_fofc_edge_flagged(fofc_flags,axis,i,j,k)
end

@inline function ct_weno7_cache_edge_data(
    cache, direction::Val{D}, si, sj, sk,
    tangent_x, tangent_y, tangent_z,
) where {D}
    emf =
        _ct_weno7_cache_component(cache,direction,si,sj,sk,1)*tangent_x +
        _ct_weno7_cache_component(cache,direction,si,sj,sk,2)*tangent_y +
        _ct_weno7_cache_component(cache,direction,si,sj,sk,3)*tangent_z
    return emf,_ct_weno7_cache_weight(cache,direction,si,sj,sk)
end

function ct_compute_edge_line_emf_from_weno7_cache_kernel!(
    Ex_edge, Ey_edge, Ez_edge, cache_i, cache_j, cache_k,
    U_stage, Q_stage, x, y, z, nxp, nyp, nzp,
    junction_edge_mask=UInt16(0), selective_only::Bool=true,
    shock_sensor=nothing, shock_threshold=zero(FT),
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)

    if i <= nxp + Int32(1) && j <= nyp + Int32(1) && k <= nzp &&
       !ct_junction_edge_masked(
           junction_edge_mask, Int32(3), i, j, k, nxp, nyp, nzp,
       ) && (!selective_only || ct_shock_edge_flagged(
           shock_sensor,Int32(3),i,j,k,shock_threshold,
       ))
        si, sj, sk = i+ng, j+ng, k+ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[si,sj,sk+Int32(1)]-x[si,sj,sk],
            y[si,sj,sk+Int32(1)]-y[si,sj,sk],
            z[si,sj,sk+Int32(1)]-z[si,sj,sk],
        )
        ez_xf_jm,w_xf_jm = ct_weno7_cache_edge_data(
            cache_i,Val(1),si,sj-Int32(1),sk,tx,ty,tz,
        )
        ez_xf_j,w_xf_j = ct_weno7_cache_edge_data(
            cache_i,Val(1),si,sj,sk,tx,ty,tz,
        )
        ez_yf_im,w_yf_im = ct_weno7_cache_edge_data(
            cache_j,Val(2),si-Int32(1),sj,sk,tx,ty,tz,
        )
        ez_yf_i,w_yf_i = ct_weno7_cache_edge_data(
            cache_j,Val(2),si,sj,sk,tx,ty,tz,
        )
        cc_jm_im = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si-Int32(1),sj-Int32(1),sk,tx,ty,tz,
        )
        cc_jm_i = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si,sj-Int32(1),sk,tx,ty,tz,
        )
        cc_j_im = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si-Int32(1),sj,sk,tx,ty,tz,
        )
        cc_j_i = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si,sj,sk,tx,ty,tz,
        )
        @inbounds Ez_edge[i,j,k] = edge_length*ct_sg07_edge_emf(
            ez_xf_jm,ez_xf_j,ez_yf_im,ez_yf_i,
            cc_jm_im,cc_jm_i,cc_j_im,cc_j_i,
            w_xf_jm,w_xf_j,w_yf_im,w_yf_i,
        )
    end

    if i <= nxp + Int32(1) && j <= nyp && k <= nzp + Int32(1) &&
       !ct_junction_edge_masked(
           junction_edge_mask, Int32(2), i, j, k, nxp, nyp, nzp,
       ) && (!selective_only || ct_shock_edge_flagged(
           shock_sensor,Int32(2),i,j,k,shock_threshold,
       ))
        si, sj, sk = i+ng, j+ng, k+ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[si,sj+Int32(1),sk]-x[si,sj,sk],
            y[si,sj+Int32(1),sk]-y[si,sj,sk],
            z[si,sj+Int32(1),sk]-z[si,sj,sk],
        )
        ey_xf_km,w_xf_km = ct_weno7_cache_edge_data(
            cache_i,Val(1),si,sj,sk-Int32(1),tx,ty,tz,
        )
        ey_xf_k,w_xf_k = ct_weno7_cache_edge_data(
            cache_i,Val(1),si,sj,sk,tx,ty,tz,
        )
        ey_zf_im,w_zf_im = ct_weno7_cache_edge_data(
            cache_k,Val(3),si-Int32(1),sj,sk,tx,ty,tz,
        )
        ey_zf_i,w_zf_i = ct_weno7_cache_edge_data(
            cache_k,Val(3),si,sj,sk,tx,ty,tz,
        )
        cc_km_im = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si-Int32(1),sj,sk-Int32(1),tx,ty,tz,
        )
        cc_km_i = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si,sj,sk-Int32(1),tx,ty,tz,
        )
        cc_k_im = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si-Int32(1),sj,sk,tx,ty,tz,
        )
        cc_k_i = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si,sj,sk,tx,ty,tz,
        )
        @inbounds Ey_edge[i,j,k] = edge_length*ct_sg07_edge_emf(
            ey_xf_km,ey_xf_k,ey_zf_im,ey_zf_i,
            cc_km_im,cc_km_i,cc_k_im,cc_k_i,
            w_xf_km,w_xf_k,w_zf_im,w_zf_i,
        )
    end

    if i <= nxp && j <= nyp + Int32(1) && k <= nzp + Int32(1) &&
       !ct_junction_edge_masked(
           junction_edge_mask, Int32(1), i, j, k, nxp, nyp, nzp,
       ) && (!selective_only || ct_shock_edge_flagged(
           shock_sensor,Int32(1),i,j,k,shock_threshold,
       ))
        si, sj, sk = i+ng, j+ng, k+ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[si+Int32(1),sj,sk]-x[si,sj,sk],
            y[si+Int32(1),sj,sk]-y[si,sj,sk],
            z[si+Int32(1),sj,sk]-z[si,sj,sk],
        )
        ex_yf_km,w_yf_km = ct_weno7_cache_edge_data(
            cache_j,Val(2),si,sj,sk-Int32(1),tx,ty,tz,
        )
        ex_yf_k,w_yf_k = ct_weno7_cache_edge_data(
            cache_j,Val(2),si,sj,sk,tx,ty,tz,
        )
        ex_zf_jm,w_zf_jm = ct_weno7_cache_edge_data(
            cache_k,Val(3),si,sj-Int32(1),sk,tx,ty,tz,
        )
        ex_zf_j,w_zf_j = ct_weno7_cache_edge_data(
            cache_k,Val(3),si,sj,sk,tx,ty,tz,
        )
        cc_km_jm = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si,sj-Int32(1),sk-Int32(1),tx,ty,tz,
        )
        cc_km_j = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si,sj,sk-Int32(1),tx,ty,tz,
        )
        cc_k_jm = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si,sj-Int32(1),sk,tx,ty,tz,
        )
        cc_k_j = ct_cell_centered_edge_emf_from_state(
            U_stage,Q_stage,si,sj,sk,tx,ty,tz,
        )
        @inbounds Ex_edge[i,j,k] = edge_length*ct_sg07_edge_emf(
            ex_yf_km,ex_yf_k,ex_zf_jm,ex_zf_j,
            cc_km_jm,cc_km_j,cc_k_jm,cc_k_j,
            w_yf_km,w_yf_k,w_zf_jm,w_zf_j,
        )
    end
    return
end

function ct_compute_edge_line_emf_kernel!(
    Ex_edge, Ey_edge, Ez_edge,
    Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z, U_stage,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    Vol, x, y, z, dt, nxp, nyp, nzp, Q_stage,
    junction_edge_mask=UInt16(0), fofc_flags=nothing,
    selective_only::Bool=false, shock_sensor=nothing,
    shock_threshold=zero(FT), flux_halo::Int32=Int32(1),
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)

    if i <= nxp + Int32(1) && j <= nyp + Int32(1) && k <= nzp &&
       !ct_junction_edge_masked(
           junction_edge_mask, Int32(3), i, j, k, nxp, nyp, nzp,
       ) && (!selective_only || ct_edge_uses_sg07(
           shock_sensor,fofc_flags,shock_threshold,Int32(3),i,j,k,
       ))
        ni = i + ng; nj = j + ng; nk = k + ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[ni,nj,nk+Int32(1)] - x[ni,nj,nk],
            y[ni,nj,nk+Int32(1)] - y[ni,nj,nk],
            z[ni,nj,nk+Int32(1)] - z[ni,nj,nk],
        )
        (xf_jm, xf_j), (yf_im, yf_i) = ct_ez_face_flux_indices(
            i,j,k,flux_halo,
        )
        ez_xf_jm, w_xf_jm = ct_xface_edge_data(
            Fx, rho_sum_x, U_stage, Q_stage, Areai, nxi, nyi, nzi, Vol,
            xf_jm, tx, ty, tz, dt, flux_halo,
        )
        ez_xf_j, w_xf_j = ct_xface_edge_data(
            Fx, rho_sum_x, U_stage, Q_stage, Areai, nxi, nyi, nzi, Vol,
            xf_j, tx, ty, tz, dt, flux_halo,
        )
        ez_yf_im, w_yf_im = ct_yface_edge_data(
            Fy, rho_sum_y, U_stage, Q_stage, Areaj, nxj, nyj, nzj, Vol,
            yf_im, tx, ty, tz, dt, flux_halo,
        )
        ez_yf_i, w_yf_i = ct_yface_edge_data(
            Fy, rho_sum_y, U_stage, Q_stage, Areaj, nxj, nyj, nzj, Vol,
            yf_i, tx, ty, tz, dt, flux_halo,
        )
        cc_jm_im = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i-Int32(1)+ng, j-Int32(1)+ng, k+ng, tx, ty, tz,
        )
        cc_jm_i = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i+ng, j-Int32(1)+ng, k+ng, tx, ty, tz,
        )
        cc_j_im = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i-Int32(1)+ng, j+ng, k+ng, tx, ty, tz,
        )
        cc_j_i = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i+ng, j+ng, k+ng, tx, ty, tz,
        )
        @inbounds Ez_edge[i,j,k] = edge_length * ct_sg07_edge_emf(
            ez_xf_jm, ez_xf_j, ez_yf_im, ez_yf_i,
            cc_jm_im, cc_jm_i, cc_j_im, cc_j_i,
            w_xf_jm, w_xf_j, w_yf_im, w_yf_i,
        )
    end

    if i <= nxp + Int32(1) && j <= nyp && k <= nzp + Int32(1) &&
       !ct_junction_edge_masked(
           junction_edge_mask, Int32(2), i, j, k, nxp, nyp, nzp,
       ) && (!selective_only || ct_edge_uses_sg07(
           shock_sensor,fofc_flags,shock_threshold,Int32(2),i,j,k,
       ))
        ni = i + ng; nj = j + ng; nk = k + ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[ni,nj+Int32(1),nk] - x[ni,nj,nk],
            y[ni,nj+Int32(1),nk] - y[ni,nj,nk],
            z[ni,nj+Int32(1),nk] - z[ni,nj,nk],
        )
        (xf_km, xf_k), (zf_im, zf_i) = ct_ey_face_flux_indices(
            i,j,k,flux_halo,
        )
        ey_xf_km, w_xf_km = ct_xface_edge_data(
            Fx, rho_sum_x, U_stage, Q_stage, Areai, nxi, nyi, nzi, Vol,
            xf_km, tx, ty, tz, dt, flux_halo,
        )
        ey_xf_k, w_xf_k = ct_xface_edge_data(
            Fx, rho_sum_x, U_stage, Q_stage, Areai, nxi, nyi, nzi, Vol,
            xf_k, tx, ty, tz, dt, flux_halo,
        )
        ey_zf_im, w_zf_im = ct_zface_edge_data(
            Fz, rho_sum_z, U_stage, Q_stage, Areak, nxk, nyk, nzk, Vol,
            zf_im, tx, ty, tz, dt, flux_halo,
        )
        ey_zf_i, w_zf_i = ct_zface_edge_data(
            Fz, rho_sum_z, U_stage, Q_stage, Areak, nxk, nyk, nzk, Vol,
            zf_i, tx, ty, tz, dt, flux_halo,
        )
        cc_km_im = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i-Int32(1)+ng, j+ng, k-Int32(1)+ng, tx, ty, tz,
        )
        cc_km_i = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i+ng, j+ng, k-Int32(1)+ng, tx, ty, tz,
        )
        cc_k_im = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i-Int32(1)+ng, j+ng, k+ng, tx, ty, tz,
        )
        cc_k_i = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i+ng, j+ng, k+ng, tx, ty, tz,
        )
        @inbounds Ey_edge[i,j,k] = edge_length * ct_sg07_edge_emf(
            ey_xf_km, ey_xf_k, ey_zf_im, ey_zf_i,
            cc_km_im, cc_km_i, cc_k_im, cc_k_i,
            w_xf_km, w_xf_k, w_zf_im, w_zf_i,
        )
    end

    if i <= nxp && j <= nyp + Int32(1) && k <= nzp + Int32(1) &&
       !ct_junction_edge_masked(
           junction_edge_mask, Int32(1), i, j, k, nxp, nyp, nzp,
       ) && (!selective_only || ct_edge_uses_sg07(
           shock_sensor,fofc_flags,shock_threshold,Int32(1),i,j,k,
       ))
        ni = i + ng; nj = j + ng; nk = k + ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[ni+Int32(1),nj,nk] - x[ni,nj,nk],
            y[ni+Int32(1),nj,nk] - y[ni,nj,nk],
            z[ni+Int32(1),nj,nk] - z[ni,nj,nk],
        )
        (yf_km, yf_k), (zf_jm, zf_j) = ct_ex_face_flux_indices(
            i,j,k,flux_halo,
        )
        ex_yf_km, w_yf_km = ct_yface_edge_data(
            Fy, rho_sum_y, U_stage, Q_stage, Areaj, nxj, nyj, nzj, Vol,
            yf_km, tx, ty, tz, dt, flux_halo,
        )
        ex_yf_k, w_yf_k = ct_yface_edge_data(
            Fy, rho_sum_y, U_stage, Q_stage, Areaj, nxj, nyj, nzj, Vol,
            yf_k, tx, ty, tz, dt, flux_halo,
        )
        ex_zf_jm, w_zf_jm = ct_zface_edge_data(
            Fz, rho_sum_z, U_stage, Q_stage, Areak, nxk, nyk, nzk, Vol,
            zf_jm, tx, ty, tz, dt, flux_halo,
        )
        ex_zf_j, w_zf_j = ct_zface_edge_data(
            Fz, rho_sum_z, U_stage, Q_stage, Areak, nxk, nyk, nzk, Vol,
            zf_j, tx, ty, tz, dt, flux_halo,
        )
        cc_km_jm = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i+ng, j-Int32(1)+ng, k-Int32(1)+ng, tx, ty, tz,
        )
        cc_km_j = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i+ng, j+ng, k-Int32(1)+ng, tx, ty, tz,
        )
        cc_k_jm = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i+ng, j-Int32(1)+ng, k+ng, tx, ty, tz,
        )
        cc_k_j = ct_cell_centered_edge_emf_from_state(
            U_stage, Q_stage, i+ng, j+ng, k+ng, tx, ty, tz,
        )
        @inbounds Ex_edge[i,j,k] = edge_length * ct_sg07_edge_emf(
            ex_yf_km, ex_yf_k, ex_zf_jm, ex_zf_j,
            cc_km_jm, cc_km_j, cc_k_jm, cc_k_j,
            w_yf_km, w_yf_k, w_zf_jm, w_zf_j,
        )
    end
    return
end


function ct_scale_edge_line_emf_kernel!(
    Ex_edge,Ey_edge,Ez_edge,fofc_flags,nxp,nyp,nzp,
)
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z

    if i <= nxp+Int32(1) && j <= nyp+Int32(1) && k <= nzp
        active,scale = ct_fofc_edge_activity(
            fofc_flags,Int32(3),i,j,k,
        )
        active && (@inbounds Ez_edge[i,j,k] *= scale)
    end
    if i <= nxp+Int32(1) && j <= nyp && k <= nzp+Int32(1)
        active,scale = ct_fofc_edge_activity(
            fofc_flags,Int32(2),i,j,k,
        )
        active && (@inbounds Ey_edge[i,j,k] *= scale)
    end
    if i <= nxp && j <= nyp+Int32(1) && k <= nzp+Int32(1)
        active,scale = ct_fofc_edge_activity(
            fofc_flags,Int32(1),i,j,k,
        )
        active && (@inbounds Ex_edge[i,j,k] *= scale)
    end
    return
end


function ct_compute_resistive_cell_emf_kernel!(
    emf_x, emf_y, emf_z, Q,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    Vol, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)
    if i < ng || i > nxp+ng+Int32(1) ||
       j < ng || j > nyp+ng+Int32(1) ||
       k < ng || k > nzp+ng+Int32(1)
        return
    end
    @inbounds current = ct_green_gauss_curl_b(
        Q,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        Vol[i,j,k], i, j, k,
    )
    @inbounds begin
        emf_x[i,j,k] = FT(η_mhd)*current[1]
        emf_y[i,j,k] = FT(η_mhd)*current[2]
        emf_z[i,j,k] = FT(η_mhd)*current[3]
    end
    return
end

@inline function _ct_resistive_energy_flux_from_edges_kernel!(
    flux, Q, face_b, background_face, Ex_edge, Ey_edge, Ez_edge,
    area_array, nx_array, ny_array, nz_array, x, y, z,
    nxp, nyp, nzp, additive, ::Val{DIRECTION},
) where {DIRECTION}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if DIRECTION == 1
        (i > nxp+Int32(1) || j > nyp || k > nzp) && return
    elseif DIRECTION == 2
        (i > nxp || j > nyp+Int32(1) || k > nzp) && return
    else
        (i > nxp || j > nyp || k > nzp+Int32(1)) && return
    end

    ng = Int32(NG)
    half = FT(0.5)
    ni, nj, nk = i+ng, j+ng, k+ng
    @inbounds if DIRECTION == 1
        tangent_ax = half*((x[ni,nj+1,nk]-x[ni,nj,nk]) +
                           (x[ni,nj+1,nk+1]-x[ni,nj,nk+1]))
        tangent_ay = half*((y[ni,nj+1,nk]-y[ni,nj,nk]) +
                           (y[ni,nj+1,nk+1]-y[ni,nj,nk+1]))
        tangent_az = half*((z[ni,nj+1,nk]-z[ni,nj,nk]) +
                           (z[ni,nj+1,nk+1]-z[ni,nj,nk+1]))
        tangent_bx = half*((x[ni,nj,nk+1]-x[ni,nj,nk]) +
                           (x[ni,nj+1,nk+1]-x[ni,nj+1,nk]))
        tangent_by = half*((y[ni,nj,nk+1]-y[ni,nj,nk]) +
                           (y[ni,nj+1,nk+1]-y[ni,nj+1,nk]))
        tangent_bz = half*((z[ni,nj,nk+1]-z[ni,nj,nk]) +
                           (z[ni,nj+1,nk+1]-z[ni,nj+1,nk]))
        line_a = half*(Ey_edge[i,j,k] + Ey_edge[i,j,k+1])
        line_b = half*(Ez_edge[i,j,k] + Ez_edge[i,j+1,k])
        gi, gj, gk = ni, nj, nk
        li, lj, lk = gi-Int32(1), gj, gk
        ri, rj, rk = gi, gj, gk
        offset = STRUCTURED_FLUX_TANGENTIAL_HALO
        fi, fj, fk = i, j+offset, k+offset
    elseif DIRECTION == 2
        tangent_ax = half*((x[ni+1,nj,nk]-x[ni,nj,nk]) +
                           (x[ni+1,nj,nk+1]-x[ni,nj,nk+1]))
        tangent_ay = half*((y[ni+1,nj,nk]-y[ni,nj,nk]) +
                           (y[ni+1,nj,nk+1]-y[ni,nj,nk+1]))
        tangent_az = half*((z[ni+1,nj,nk]-z[ni,nj,nk]) +
                           (z[ni+1,nj,nk+1]-z[ni,nj,nk+1]))
        tangent_bx = half*((x[ni,nj,nk+1]-x[ni,nj,nk]) +
                           (x[ni+1,nj,nk+1]-x[ni+1,nj,nk]))
        tangent_by = half*((y[ni,nj,nk+1]-y[ni,nj,nk]) +
                           (y[ni+1,nj,nk+1]-y[ni+1,nj,nk]))
        tangent_bz = half*((z[ni,nj,nk+1]-z[ni,nj,nk]) +
                           (z[ni+1,nj,nk+1]-z[ni+1,nj,nk]))
        line_a = half*(Ex_edge[i,j,k] + Ex_edge[i,j,k+1])
        line_b = half*(Ez_edge[i,j,k] + Ez_edge[i+1,j,k])
        gi, gj, gk = ni, nj, nk
        li, lj, lk = gi, gj-Int32(1), gk
        ri, rj, rk = gi, gj, gk
        offset = STRUCTURED_FLUX_TANGENTIAL_HALO
        fi, fj, fk = i+offset, j, k+offset
    else
        tangent_ax = half*((x[ni+1,nj,nk]-x[ni,nj,nk]) +
                           (x[ni+1,nj+1,nk]-x[ni,nj+1,nk]))
        tangent_ay = half*((y[ni+1,nj,nk]-y[ni,nj,nk]) +
                           (y[ni+1,nj+1,nk]-y[ni,nj+1,nk]))
        tangent_az = half*((z[ni+1,nj,nk]-z[ni,nj,nk]) +
                           (z[ni+1,nj+1,nk]-z[ni,nj+1,nk]))
        tangent_bx = half*((x[ni,nj+1,nk]-x[ni,nj,nk]) +
                           (x[ni+1,nj+1,nk]-x[ni+1,nj,nk]))
        tangent_by = half*((y[ni,nj+1,nk]-y[ni,nj,nk]) +
                           (y[ni+1,nj+1,nk]-y[ni+1,nj,nk]))
        tangent_bz = half*((z[ni,nj+1,nk]-z[ni,nj,nk]) +
                           (z[ni+1,nj+1,nk]-z[ni+1,nj,nk]))
        line_a = half*(Ex_edge[i,j,k] + Ex_edge[i,j+1,k])
        line_b = half*(Ey_edge[i,j,k] + Ey_edge[i+1,j,k])
        gi, gj, gk = ni, nj, nk
        li, lj, lk = gi, gj, gk-Int32(1)
        ri, rj, rk = gi, gj, gk
        offset = STRUCTURED_FLUX_TANGENTIAL_HALO
        fi, fj, fk = i+offset, j+offset, k
    end

    @inbounds begin
        area = area_array[gi,gj,gk]
        normal_x = nx_array[gi,gj,gk]
        normal_y = ny_array[gi,gj,gk]
        normal_z = nz_array[gi,gj,gk]
        magnetic_x = half*(Q[li,lj,lk,QBX] + Q[ri,rj,rk,QBX])
        magnetic_y = half*(Q[li,lj,lk,QBY] + Q[ri,rj,rk,QBY])
        magnetic_z = half*(Q[li,lj,lk,QBZ] + Q[ri,rj,rk,QBZ])
        face_normal_b = _ct_total_face_flux(
            face_b, background_face, gi, gj, gk,
        ) / area
    end
    normal_correction = face_normal_b - (
        magnetic_x*normal_x + magnetic_y*normal_y + magnetic_z*normal_z)
    magnetic_x += normal_correction*normal_x
    magnetic_y += normal_correction*normal_y
    magnetic_z += normal_correction*normal_z
    energy_flux = ct_resistive_edge_poynting_flux(
        line_a, line_b,
        tangent_ax, tangent_ay, tangent_az,
        tangent_bx, tangent_by, tangent_bz,
        area*normal_x, area*normal_y, area*normal_z,
        magnetic_x, magnetic_y, magnetic_z,
    )
    @inbounds if additive
        flux[fi,fj,fk,5] += energy_flux
    else
        flux[fi,fj,fk,5] = energy_flux
    end
    return
end

function ct_resistive_energy_flux_i_from_edges_kernel!(
    flux, Q, face_b, Ex_edge, Ey_edge, Ez_edge,
    area, nx, ny, nz, x, y, z, nxp, nyp, nzp, additive,
    background_face=nothing,
)
    _ct_resistive_energy_flux_from_edges_kernel!(
        flux,Q,face_b,background_face,Ex_edge,Ey_edge,Ez_edge,
        area,nx,ny,nz,x,y,z,nxp,nyp,nzp,additive,Val(1))
end

function ct_resistive_energy_flux_j_from_edges_kernel!(
    flux, Q, face_b, Ex_edge, Ey_edge, Ez_edge,
    area, nx, ny, nz, x, y, z, nxp, nyp, nzp, additive,
    background_face=nothing,
)
    _ct_resistive_energy_flux_from_edges_kernel!(
        flux,Q,face_b,background_face,Ex_edge,Ey_edge,Ez_edge,
        area,nx,ny,nz,x,y,z,nxp,nyp,nzp,additive,Val(2))
end

function ct_resistive_energy_flux_k_from_edges_kernel!(
    flux, Q, face_b, Ex_edge, Ey_edge, Ez_edge,
    area, nx, ny, nz, x, y, z, nxp, nyp, nzp, additive,
    background_face=nothing,
)
    _ct_resistive_energy_flux_from_edges_kernel!(
        flux,Q,face_b,background_face,Ex_edge,Ey_edge,Ez_edge,
        area,nx,ny,nz,x,y,z,nxp,nyp,nzp,additive,Val(3))
end

@inline function ct_resistive_energy_rate(
    flux_x, flux_y, flux_z, inverse_volume, i, j, k, ii, jj, kk,
)
    tangential_offset = STRUCTURED_FLUX_TANGENTIAL_HALO
    return inverse_volume[ii,jj,kk] * (
        flux_x[i+Int32(1),j+tangential_offset,k+tangential_offset,5] -
            flux_x[i,j+tangential_offset,k+tangential_offset,5] +
        flux_y[i+tangential_offset,j+Int32(1),k+tangential_offset,5] -
            flux_y[i+tangential_offset,j,k+tangential_offset,5] +
        flux_z[i+tangential_offset,j+tangential_offset,k+Int32(1),5] -
            flux_z[i+tangential_offset,j+tangential_offset,k,5]
    )
end

function ct_resistive_energy_euler_kernel!(
    U, flux_x, flux_y, flux_z, inverse_volume,
    dt_stage, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii, jj, kk = i+Int32(NG), j+Int32(NG), k+Int32(NG)
    @inbounds U[ii,jj,kk,5] += dt_stage * ct_resistive_energy_rate(
        flux_x, flux_y, flux_z, inverse_volume,
        i, j, k, ii, jj, kk,
    )
    return
end


function ct_rkl2_energy_first_stage_kernel!(
    U, U0, energy_previous2, energy_first,
    flux_x, flux_y, flux_z, inverse_volume,
    dt_coefficient, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii, jj, kk = i+Int32(NG), j+Int32(NG), k+Int32(NG)
    @inbounds begin
        base = U[ii,jj,kk,5]
        U0[ii,jj,kk,5] = base
        energy_previous2[ii,jj,kk] = base
        value = base + dt_coefficient * ct_resistive_energy_rate(
            flux_x, flux_y, flux_z, inverse_volume,
            i, j, k, ii, jj, kk,
        )
        U[ii,jj,kk,5] = value
        energy_first[ii,jj,kk] = value
    end
    return
end


function ct_rkl2_energy_stage_kernel!(
    U, U0, energy_previous2, energy_first,
    flux_x, flux_y, flux_z, inverse_volume,
    rhs_dt_coefficient, mu, nu, base_weight, gamma_ratio,
    nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii, jj, kk = i+Int32(NG), j+Int32(NG), k+Int32(NG)
    @inbounds begin
        current = U[ii,jj,kk,5]
        value = ct_rkl2_combine(
            current, energy_previous2[ii,jj,kk], U0[ii,jj,kk,5],
            energy_first[ii,jj,kk],
            rhs_dt_coefficient * ct_resistive_energy_rate(
                flux_x, flux_y, flux_z, inverse_volume,
                i, j, k, ii, jj, kk,
            ),
            mu, nu, base_weight, gamma_ratio,
        )
        energy_previous2[ii,jj,kk] = current
        U[ii,jj,kk,5] = value
    end
    return
end

function ct_resistive_explicit_dt_kernel!(
    dt_array, inverse_volume, area_i, area_j, area_k,
    nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii, jj, kk = i+Int32(NG), j+Int32(NG), k+Int32(NG)
    @inbounds begin
        volume = one(FT) / inverse_volume[ii,jj,kk]
        ai = FT(0.5) * (area_i[ii,jj,kk] + area_i[ii+1,jj,kk])
        aj = FT(0.5) * (area_j[ii,jj,kk] + area_j[ii,jj+1,kk])
        ak = FT(0.5) * (area_k[ii,jj,kk] + area_k[ii,jj,kk+1])
        inv_dx2 = (ai/volume)^2
        inv_dy2 = (aj/volume)^2
        inv_dz2 = (ak/volume)^2
        dt_array[ii,jj,kk] = FT(0.5) /
            (FT(η_mhd)*(inv_dx2+inv_dy2+inv_dz2) + FT(1e-30))
    end
    return
end

@inline function ct_xface_resistive_edge_data(
    flux, emf_x, emf_y, emf_z,
    Areai, nxi, nyi, nzi,
    face_index, tangent_x, tangent_y, tangent_z,
)
    face_i, cell_j, cell_k = face_index
    gi, gj, gk = ct_xface_geometry_indices(
        face_i, cell_j, cell_k, Int32(NG),
    )
    @inbounds begin
        area = Areai[gi,gj,gk]
        normal_x = nxi[gi,gj,gk]
        normal_y = nyi[gi,gj,gk]
        normal_z = nzi[gi,gj,gk]
        face_normal_emf = ct_face_normal_cached_emf(
            emf_x, emf_y, emf_z,
            gi-Int32(1), gj, gk, gi, gj, gk,
            normal_x, normal_y, normal_z,
        )
        return ct_project_resistive_face_flux_to_edge_emf(
            flux[face_i,cell_j,cell_k,UBX],
            flux[face_i,cell_j,cell_k,UBY],
            flux[face_i,cell_j,cell_k,UBZ],
            normal_x, normal_y, normal_z,
            tangent_x, tangent_y, tangent_z,
            area, face_normal_emf,
        )
    end
end

@inline function ct_yface_resistive_edge_data(
    flux, emf_x, emf_y, emf_z,
    Areaj, nxj, nyj, nzj,
    face_index, tangent_x, tangent_y, tangent_z,
)
    cell_i, face_j, cell_k = face_index
    gi, gj, gk = ct_yface_geometry_indices(
        cell_i, face_j, cell_k, Int32(NG),
    )
    @inbounds begin
        area = Areaj[gi,gj,gk]
        normal_x = nxj[gi,gj,gk]
        normal_y = nyj[gi,gj,gk]
        normal_z = nzj[gi,gj,gk]
        face_normal_emf = ct_face_normal_cached_emf(
            emf_x, emf_y, emf_z,
            gi, gj-Int32(1), gk, gi, gj, gk,
            normal_x, normal_y, normal_z,
        )
        return ct_project_resistive_face_flux_to_edge_emf(
            flux[cell_i,face_j,cell_k,UBX],
            flux[cell_i,face_j,cell_k,UBY],
            flux[cell_i,face_j,cell_k,UBZ],
            normal_x, normal_y, normal_z,
            tangent_x, tangent_y, tangent_z,
            area, face_normal_emf,
        )
    end
end

@inline function ct_zface_resistive_edge_data(
    flux, emf_x, emf_y, emf_z,
    Areak, nxk, nyk, nzk,
    face_index, tangent_x, tangent_y, tangent_z,
)
    cell_i, cell_j, face_k = face_index
    gi, gj, gk = ct_zface_geometry_indices(
        cell_i, cell_j, face_k, Int32(NG),
    )
    @inbounds begin
        area = Areak[gi,gj,gk]
        normal_x = nxk[gi,gj,gk]
        normal_y = nyk[gi,gj,gk]
        normal_z = nzk[gi,gj,gk]
        face_normal_emf = ct_face_normal_cached_emf(
            emf_x, emf_y, emf_z,
            gi, gj, gk-Int32(1), gi, gj, gk,
            normal_x, normal_y, normal_z,
        )
        return ct_project_resistive_face_flux_to_edge_emf(
            flux[cell_i,cell_j,face_k,UBX],
            flux[cell_i,cell_j,face_k,UBY],
            flux[cell_i,cell_j,face_k,UBZ],
            normal_x, normal_y, normal_z,
            tangent_x, tangent_y, tangent_z,
            area, face_normal_emf,
        )
    end
end

@inline function _ct_junction_resistive_face_edge_data(
    face, Fv_x, Fv_y, Fv_z, emf_x, emf_y, emf_z,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    cell_i, cell_j, cell_k,
    tangent_x, tangent_y, tangent_z, nxp, nyp, nzp,
)
    normal_axis = _ct_junction_face_axis(face)
    halo = Int32(STRUCTURED_FLUX_TANGENTIAL_HALO)
    if normal_axis == Int32(1)
        face_i = _ct_junction_boundary_node(face, nxp)
        return ct_xface_resistive_edge_data(
            Fv_x, emf_x, emf_y, emf_z, Areai, nxi, nyi, nzi,
            (face_i, cell_j + halo, cell_k + halo),
            tangent_x, tangent_y, tangent_z,
        )
    elseif normal_axis == Int32(2)
        face_j = _ct_junction_boundary_node(face, nyp)
        return ct_yface_resistive_edge_data(
            Fv_y, emf_x, emf_y, emf_z, Areaj, nxj, nyj, nzj,
            (cell_i + halo, face_j, cell_k + halo),
            tangent_x, tangent_y, tangent_z,
        )
    end
    face_k = _ct_junction_boundary_node(face, nzp)
    return ct_zface_resistive_edge_data(
        Fv_z, emf_x, emf_y, emf_z, Areak, nxk, nyk, nzk,
        (cell_i + halo, cell_j + halo, face_k),
        tangent_x, tangent_y, tangent_z,
    )
end

function ct_pack_generalized_junction_resistive_payload_kernel!(
    payload, Fv_x, Fv_y, Fv_z, emf_x, emf_y, emf_z,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    x, y, z, edge_axis, face1, face2, edge_orientation,
    nxp, nyp, nzp,
)
    segment = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    segment_extent = _ct_junction_dimension(edge_axis, nxp, nyp, nzp)
    segment > segment_extent && return
    face_x = _ct_junction_face_for_axis(face1, face2, Int32(1))
    face_y = _ct_junction_face_for_axis(face1, face2, Int32(2))
    face_z = _ct_junction_face_for_axis(face1, face2, Int32(3))
    edge_i = edge_axis == Int32(1) ? segment :
        _ct_junction_boundary_node(face_x, nxp)
    edge_j = edge_axis == Int32(2) ? segment :
        _ct_junction_boundary_node(face_y, nyp)
    edge_k = edge_axis == Int32(3) ? segment :
        _ct_junction_boundary_node(face_z, nzp)
    cell_i = edge_axis == Int32(1) ? segment :
        _ct_junction_boundary_cell(face_x, nxp)
    cell_j = edge_axis == Int32(2) ? segment :
        _ct_junction_boundary_cell(face_y, nyp)
    cell_k = edge_axis == Int32(3) ? segment :
        _ct_junction_boundary_cell(face_z, nzp)
    ng = Int32(NG)
    ni, nj, nk = edge_i + ng, edge_j + ng, edge_k + ng
    next_i = edge_axis == Int32(1) ? ni + Int32(1) : ni
    next_j = edge_axis == Int32(2) ? nj + Int32(1) : nj
    next_k = edge_axis == Int32(3) ? nk + Int32(1) : nk
    @inbounds tangent_x, tangent_y, tangent_z, _ = ct_edge_geometry(
        x[next_i, next_j, next_k] - x[ni, nj, nk],
        y[next_i, next_j, next_k] - y[ni, nj, nk],
        z[next_i, next_j, next_k] - z[ni, nj, nk],
    )
    cell_emf = ct_cached_edge_emf(
        emf_x, emf_y, emf_z,
        cell_i + ng, cell_j + ng, cell_k + ng,
        tangent_x, tangent_y, tangent_z,
    )
    face1_emf = _ct_junction_resistive_face_edge_data(
        face1, Fv_x, Fv_y, Fv_z, emf_x, emf_y, emf_z,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        cell_i, cell_j, cell_k,
        tangent_x, tangent_y, tangent_z, nxp, nyp, nzp,
    )
    face2_emf = _ct_junction_resistive_face_edge_data(
        face2, Fv_x, Fv_y, Fv_z, emf_x, emf_y, emf_z,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        cell_i, cell_j, cell_k,
        tangent_x, tangent_y, tangent_z, nxp, nyp, nzp,
    )
    canonical_segment = edge_orientation < 0 ?
        segment_extent - segment + Int32(1) : segment
    edge_sign = edge_orientation < 0 ? -one(cell_emf) : one(cell_emf)
    @inbounds begin
        payload[CT_JUNCTION_PAYLOAD_RESISTIVE_CELL_EMF,
                canonical_segment] = edge_sign * cell_emf
        payload[CT_JUNCTION_PAYLOAD_RESISTIVE_FACE1_EMF,
                canonical_segment] = edge_sign * face1_emf
        payload[CT_JUNCTION_PAYLOAD_RESISTIVE_FACE2_EMF,
                canonical_segment] = edge_sign * face2_emf
    end
    return
end

function ct_launch_generalized_junction_resistive_payload!(
    payload, b, Fv_x, Fv_y, Fv_z, emf_x, emf_y, emf_z,
    edge_axis, face1, face2, edge_orientation,
)
    segment_count = _ct_junction_dimension(
        Int32(edge_axis), Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
    )
    thread_count = Int32(min(256, max(1, segment_count)))
    threads = (thread_count, Int32(1), Int32(1))
    blocks = (
        Int32(cld(segment_count, thread_count)), Int32(1), Int32(1),
    )
    @gpu_launch threads=threads blocks=blocks ct_pack_generalized_junction_resistive_payload_kernel!(
        payload, Fv_x, Fv_y, Fv_z, emf_x, emf_y, emf_z,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        b.x, b.y, b.z,
        Int32(edge_axis), Int32(face1), Int32(face2),
        Int8(edge_orientation),
        Int32(b.Nx), Int32(b.Ny), Int32(b.Nz),
    )
    return nothing
end

function ct_add_resistive_edge_line_emf_kernel!(
    Ex_edge, Ey_edge, Ez_edge,
    Fv_x, Fv_y, Fv_z,
    emf_x, emf_y, emf_z,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    x, y, z, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)
    half = FT(0.5)

    if i <= nxp+Int32(1) && j <= nyp+Int32(1) && k <= nzp
        ni = i+ng; nj = j+ng; nk = k+ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[ni,nj,nk+Int32(1)]-x[ni,nj,nk],
            y[ni,nj,nk+Int32(1)]-y[ni,nj,nk],
            z[ni,nj,nk+Int32(1)]-z[ni,nj,nk],
        )
        (xf_jm, xf_j), (yf_im, yf_i) = ct_ez_face_flux_indices(i,j,k)
        ez_xf_jm = ct_xface_resistive_edge_data(
            Fv_x, emf_x, emf_y, emf_z, Areai, nxi, nyi, nzi,
            xf_jm, tx, ty, tz,
        )
        ez_xf_j = ct_xface_resistive_edge_data(
            Fv_x, emf_x, emf_y, emf_z, Areai, nxi, nyi, nzi,
            xf_j, tx, ty, tz,
        )
        ez_yf_im = ct_yface_resistive_edge_data(
            Fv_y, emf_x, emf_y, emf_z, Areaj, nxj, nyj, nzj,
            yf_im, tx, ty, tz,
        )
        ez_yf_i = ct_yface_resistive_edge_data(
            Fv_y, emf_x, emf_y, emf_z, Areaj, nxj, nyj, nzj,
            yf_i, tx, ty, tz,
        )
        cc_jm_im = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i-Int32(1)+ng, j-Int32(1)+ng, k+ng,
            tx, ty, tz,
        )
        cc_jm_i = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i+ng, j-Int32(1)+ng, k+ng,
            tx, ty, tz,
        )
        cc_j_im = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i-Int32(1)+ng, j+ng, k+ng,
            tx, ty, tz,
        )
        cc_j_i = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i+ng, j+ng, k+ng,
            tx, ty, tz,
        )
        @inbounds Ez_edge[i,j,k] += edge_length*ct_sg07_edge_emf(
            ez_xf_jm, ez_xf_j, ez_yf_im, ez_yf_i,
            cc_jm_im, cc_jm_i, cc_j_im, cc_j_i,
            half, half, half, half,
        )
    end

    if i <= nxp+Int32(1) && j <= nyp && k <= nzp+Int32(1)
        ni = i+ng; nj = j+ng; nk = k+ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[ni,nj+Int32(1),nk]-x[ni,nj,nk],
            y[ni,nj+Int32(1),nk]-y[ni,nj,nk],
            z[ni,nj+Int32(1),nk]-z[ni,nj,nk],
        )
        (xf_km, xf_k), (zf_im, zf_i) = ct_ey_face_flux_indices(i,j,k)
        ey_xf_km = ct_xface_resistive_edge_data(
            Fv_x, emf_x, emf_y, emf_z, Areai, nxi, nyi, nzi,
            xf_km, tx, ty, tz,
        )
        ey_xf_k = ct_xface_resistive_edge_data(
            Fv_x, emf_x, emf_y, emf_z, Areai, nxi, nyi, nzi,
            xf_k, tx, ty, tz,
        )
        ey_zf_im = ct_zface_resistive_edge_data(
            Fv_z, emf_x, emf_y, emf_z, Areak, nxk, nyk, nzk,
            zf_im, tx, ty, tz,
        )
        ey_zf_i = ct_zface_resistive_edge_data(
            Fv_z, emf_x, emf_y, emf_z, Areak, nxk, nyk, nzk,
            zf_i, tx, ty, tz,
        )
        cc_km_im = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i-Int32(1)+ng, j+ng, k-Int32(1)+ng,
            tx, ty, tz,
        )
        cc_km_i = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i+ng, j+ng, k-Int32(1)+ng,
            tx, ty, tz,
        )
        cc_k_im = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i-Int32(1)+ng, j+ng, k+ng,
            tx, ty, tz,
        )
        cc_k_i = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i+ng, j+ng, k+ng,
            tx, ty, tz,
        )
        @inbounds Ey_edge[i,j,k] += edge_length*ct_sg07_edge_emf(
            ey_xf_km, ey_xf_k, ey_zf_im, ey_zf_i,
            cc_km_im, cc_km_i, cc_k_im, cc_k_i,
            half, half, half, half,
        )
    end

    if i <= nxp && j <= nyp+Int32(1) && k <= nzp+Int32(1)
        ni = i+ng; nj = j+ng; nk = k+ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[ni+Int32(1),nj,nk]-x[ni,nj,nk],
            y[ni+Int32(1),nj,nk]-y[ni,nj,nk],
            z[ni+Int32(1),nj,nk]-z[ni,nj,nk],
        )
        (yf_km, yf_k), (zf_jm, zf_j) = ct_ex_face_flux_indices(i,j,k)
        ex_yf_km = ct_yface_resistive_edge_data(
            Fv_y, emf_x, emf_y, emf_z, Areaj, nxj, nyj, nzj,
            yf_km, tx, ty, tz,
        )
        ex_yf_k = ct_yface_resistive_edge_data(
            Fv_y, emf_x, emf_y, emf_z, Areaj, nxj, nyj, nzj,
            yf_k, tx, ty, tz,
        )
        ex_zf_jm = ct_zface_resistive_edge_data(
            Fv_z, emf_x, emf_y, emf_z, Areak, nxk, nyk, nzk,
            zf_jm, tx, ty, tz,
        )
        ex_zf_j = ct_zface_resistive_edge_data(
            Fv_z, emf_x, emf_y, emf_z, Areak, nxk, nyk, nzk,
            zf_j, tx, ty, tz,
        )
        cc_km_jm = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i+ng, j-Int32(1)+ng, k-Int32(1)+ng,
            tx, ty, tz,
        )
        cc_km_j = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i+ng, j+ng, k-Int32(1)+ng,
            tx, ty, tz,
        )
        cc_k_jm = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i+ng, j-Int32(1)+ng, k+ng,
            tx, ty, tz,
        )
        cc_k_j = ct_cached_edge_emf(
            emf_x, emf_y, emf_z, i+ng, j+ng, k+ng,
            tx, ty, tz,
        )
        @inbounds Ex_edge[i,j,k] += edge_length*ct_sg07_edge_emf(
            ex_yf_km, ex_yf_k, ex_zf_jm, ex_zf_j,
            cc_km_jm, cc_km_j, cc_k_jm, cc_k_j,
            half, half, half, half,
        )
    end
    return
end


function ct_weno7_build_edge_line!(
    Eedge, scratch, ::Val{D}, cache_i, cache_j, cache_k,
    x, y, z, fail_meta, fail_value,
    world_rank::Int, block_id::Int, rk_stage::Int32,
    nxp::Int, nyp::Int, nzp::Int,
    local_periodic::NTuple{3,Bool},
)::Nothing where {D}
    midpoint_extent = D == 1 ? (nxp+2NG, nyp+1, nzp+1) :
                      (D == 2 ? (nxp+1, nyp+2NG, nzp+1) :
                                (nxp+1, nyp+1, nzp+2NG))
    line_extent = D == 1 ? (nxp, nyp+1, nzp+1) :
                  (D == 2 ? (nxp+1, nyp, nzp+1) :
                            (nxp+1, nyp+1, nzp))
    midpoint_blocks = ntuple(
        axis -> cld(midpoint_extent[axis], nthreads[axis]), 3,
    )
    line_blocks = ntuple(axis -> cld(line_extent[axis], nthreads[axis]), 3)
    direction = Int32(D)

    ct_weno7_reset_failure!(fail_meta, fail_value)
    @gpu_launch threads=nthreads blocks=midpoint_blocks ct_weno7_edge_midpoint_kernel!(
        scratch, direction, cache_i, cache_j, cache_k, x, y, z,
        fail_meta, fail_value, rk_stage,
        Int32(nxp), Int32(nyp), Int32(nzp))
    gpu_sync()
    ct_weno7_check_failure!(
        fail_meta, fail_value;
        rank=world_rank, block=block_id, rk_stage=Int(rk_stage),
    )

    @gpu_launch threads=nthreads blocks=midpoint_blocks ct_weno7_periodic_point_halo_kernel!(
        scratch, direction,
        local_periodic[1], local_periodic[2], local_periodic[3],
        Int32(nxp), Int32(nyp), Int32(nzp))
    gpu_sync()

    @gpu_launch threads=nthreads blocks=line_blocks ct_weno7_edge_line_kernel!(
        Eedge, scratch, direction, fail_meta, fail_value, rk_stage,
        Int32(nxp), Int32(nyp), Int32(nzp))
    gpu_sync()
    ct_weno7_check_failure!(
        fail_meta, fail_value;
        rank=world_rank, block=block_id, rk_stage=Int(rk_stage),
    )
    return nothing
end


# A stationary perfectly conducting wall has zero tangential electric field.
# Zeroing the two edge families embedded in that wall preserves its normal
# magnetic flux exactly under the discrete Stokes update.
function ct_conducting_wall_edge_emf_kernel!(
    Ex_edge, Ey_edge, Ez_edge, direction, side, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if direction == Int32(1)
        wall_i = side == Int32(0) ? Int32(1) : nxp + Int32(1)
        if i == wall_i && j <= nyp && k <= nzp + Int32(1)
            @inbounds Ey_edge[i,j,k] = zero(FT)
        end
        if i == wall_i && j <= nyp + Int32(1) && k <= nzp
            @inbounds Ez_edge[i,j,k] = zero(FT)
        end
    elseif direction == Int32(2)
        wall_j = side == Int32(0) ? Int32(1) : nyp + Int32(1)
        if i <= nxp && j == wall_j && k <= nzp + Int32(1)
            @inbounds Ex_edge[i,j,k] = zero(FT)
        end
        if i <= nxp + Int32(1) && j == wall_j && k <= nzp
            @inbounds Ez_edge[i,j,k] = zero(FT)
        end
    else
        wall_k = side == Int32(0) ? Int32(1) : nzp + Int32(1)
        if i <= nxp && j <= nyp + Int32(1) && k == wall_k
            @inbounds Ex_edge[i,j,k] = zero(FT)
        end
        if i <= nxp + Int32(1) && j <= nyp && k == wall_k
            @inbounds Ey_edge[i,j,k] = zero(FT)
        end
    end
    return
end

@inline function ct_is_conducting_wall_bc(bc)
    return bc == Int(BC_MHD_WALL) ||
           bc == Int(BC_ISOTHERMAL_WALL) ||
           bc == Int(BC_ADIABATIC_WALL)
end

function ct_enforce_physical_edge_emf!(b, bid, face_bc)
    nb = (
        cld(b.Nx + 1, nthreads[1]),
        cld(b.Ny + 1, nthreads[2]),
        cld(b.Nz + 1, nthreads[3]),
    )
    for fid in 1:6
        bc = get(face_bc, (bid, fid), Int(BC_INTERBLOCK))
        ct_is_conducting_wall_bc(bc) || continue
        direction = Int32((fid + 1) ÷ 2)
        side = Int32(isodd(fid) ? 0 : 1)
        @gpu_launch threads=nthreads blocks=nb ct_conducting_wall_edge_emf_kernel!(
            b.Ex_edge, b.Ey_edge, b.Ez_edge,
            direction, side, Int32(b.Nx), Int32(b.Ny), Int32(b.Nz))
    end
    return nothing
end


# Periodic boundary nodes represent the same physical edge with two logical
# indices. Use the low-side edge as the canonical line integral so neighboring
# periodic faces take the curl of exactly the same edge value.
function ct_sync_periodic_edge_emf_kernel!(
    Ex_edge, Ey_edge, Ez_edge,
    periodic_x, periodic_y, periodic_z,
    nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i <= nxp && j <= nyp + Int32(1) && k <= nzp + Int32(1)
        source_j = ct_periodic_edge_source_index(
            j, nyp, periodic_y, true,
        )
        source_k = ct_periodic_edge_source_index(
            k, nzp, periodic_z, true,
        )
        if source_j != j || source_k != k
            @inbounds Ex_edge[i,j,k] = Ex_edge[i,source_j,source_k]
        end
    end

    if i <= nxp + Int32(1) && j <= nyp && k <= nzp + Int32(1)
        source_i = ct_periodic_edge_source_index(
            i, nxp, periodic_x, true,
        )
        source_k = ct_periodic_edge_source_index(
            k, nzp, periodic_z, true,
        )
        if source_i != i || source_k != k
            @inbounds Ey_edge[i,j,k] = Ey_edge[source_i,j,source_k]
        end
    end

    if i <= nxp + Int32(1) && j <= nyp + Int32(1) && k <= nzp
        source_i = ct_periodic_edge_source_index(
            i, nxp, periodic_x, true,
        )
        source_j = ct_periodic_edge_source_index(
            j, nyp, periodic_y, true,
        )
        if source_i != i || source_j != j
            @inbounds Ez_edge[i,j,k] = Ez_edge[source_i,source_j,k]
        end
    end
    return
end


# ─── CT: Discrete Stokes face-flux update + RK combination ───
function ct_update_face_b_from_emf_kernel!(Bx_face, By_face, Bz_face,
                                            Bx_face_n, By_face_n, Bz_face_n,
                                            Ex_edge, Ey_edge, Ez_edge,
                                            dt, rk_a,
                                            nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    # Edge arrays are oriented line-integrated EMFs. Discrete Stokes updates
    # oriented face magnetic flux without any metric division.

    # ── Bx_face update ──
    # Athena++ ct.cpp: b1 -= (e3(k,j+1,i) - e3(k,j,i))/dy ; b1 += (e2(k+1,j,i) - e2(k,j,i))/dz
    # Forward difference: corner j and j+1 straddle cell-center j.
    if i <= nxp + Int32(1) && j <= nyp && k <= nzp
        ii = i + NG; jj = j + NG; kk = k + NG
        jp = min(j + Int32(1), nyp + Int32(1))
        kp = min(k + Int32(1), nzp + Int32(1))
        Bx_new = ct_stokes_update_i(
            Bx_face[ii, jj, kk],
            Ey_edge[i, j, k], Ey_edge[i, j, kp],
            Ez_edge[i, j, k], Ez_edge[i, jp, k], dt,
        )
        @inbounds Bx_face[ii, jj, kk] = Bx_face_n[ii, jj, kk] +
            rk_a * (Bx_new - Bx_face_n[ii, jj, kk])
    end

    # ── By_face update ──
    # Athena++ ct.cpp: b2 += (e3(k,j,i+1) - e3(k,j,i))/dx ; b2 -= (e1(k+1,j,i) - e1(k,j,i))/dz
    if i <= nxp && j <= nyp + Int32(1) && k <= nzp
        ii = i + NG; jj = j + NG; kk = k + NG
        ip = min(i + Int32(1), nxp + Int32(1))
        kp = min(k + Int32(1), nzp + Int32(1))
        By_new = ct_stokes_update_j(
            By_face[ii, jj, kk],
            Ex_edge[i, j, k], Ex_edge[i, j, kp],
            Ez_edge[i, j, k], Ez_edge[ip, j, k], dt,
        )
        @inbounds By_face[ii, jj, kk] = By_face_n[ii, jj, kk] +
            rk_a * (By_new - By_face_n[ii, jj, kk])
    end

    # ── Bz_face update ──
    # Athena++ ct.cpp: b3 -= (e2(k,j,i+1) - e2(k,j,i))/dx ; b3 += (e1(k,j+1,i) - e1(k,j,i))/dy
    if i <= nxp && j <= nyp && k <= nzp + Int32(1)
        ii = i + NG; jj = j + NG; kk = k + NG
        ip = min(i + Int32(1), nxp + Int32(1))
        jp = min(j + Int32(1), nyp + Int32(1))
        Bz_new = ct_stokes_update_k(
            Bz_face[ii, jj, kk],
            Ex_edge[i, j, k], Ex_edge[i, jp, k],
            Ey_edge[i, j, k], Ey_edge[ip, j, k], dt,
        )
        @inbounds Bz_face[ii, jj, kk] = Bz_face_n[ii, jj, kk] +
            rk_a * (Bz_new - Bz_face_n[ii, jj, kk])
    end
    return
end


function ct_rkl2_face_first_stage_kernel!(
    Bx_face, By_face, Bz_face,
    Bx_base, By_base, Bz_base,
    Bx_previous2, By_previous2, Bz_previous2,
    Bx_first, By_first, Bz_first,
    Ex_edge, Ey_edge, Ez_edge, dt_coefficient,
    nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i <= nxp + Int32(1) && j <= nyp && k <= nzp
        ii = i + NG; jj = j + NG; kk = k + NG
        jp = min(j + Int32(1), nyp + Int32(1))
        kp = min(k + Int32(1), nzp + Int32(1))
        @inbounds begin
            base = Bx_face[ii,jj,kk]
            Bx_base[ii,jj,kk] = base
            Bx_previous2[ii,jj,kk] = base
            value = ct_stokes_update_i(
                base,
                Ey_edge[i,j,k], Ey_edge[i,j,kp],
                Ez_edge[i,j,k], Ez_edge[i,jp,k], dt_coefficient,
            )
            Bx_face[ii,jj,kk] = value
            Bx_first[ii,jj,kk] = value
        end
    end

    if i <= nxp && j <= nyp + Int32(1) && k <= nzp
        ii = i + NG; jj = j + NG; kk = k + NG
        ip = min(i + Int32(1), nxp + Int32(1))
        kp = min(k + Int32(1), nzp + Int32(1))
        @inbounds begin
            base = By_face[ii,jj,kk]
            By_base[ii,jj,kk] = base
            By_previous2[ii,jj,kk] = base
            value = ct_stokes_update_j(
                base,
                Ex_edge[i,j,k], Ex_edge[i,j,kp],
                Ez_edge[i,j,k], Ez_edge[ip,j,k], dt_coefficient,
            )
            By_face[ii,jj,kk] = value
            By_first[ii,jj,kk] = value
        end
    end

    if i <= nxp && j <= nyp && k <= nzp + Int32(1)
        ii = i + NG; jj = j + NG; kk = k + NG
        ip = min(i + Int32(1), nxp + Int32(1))
        jp = min(j + Int32(1), nyp + Int32(1))
        @inbounds begin
            base = Bz_face[ii,jj,kk]
            Bz_base[ii,jj,kk] = base
            Bz_previous2[ii,jj,kk] = base
            value = ct_stokes_update_k(
                base,
                Ex_edge[i,j,k], Ex_edge[i,jp,k],
                Ey_edge[i,j,k], Ey_edge[ip,j,k], dt_coefficient,
            )
            Bz_face[ii,jj,kk] = value
            Bz_first[ii,jj,kk] = value
        end
    end
    return
end


function ct_rkl2_face_stage_kernel!(
    Bx_face, By_face, Bz_face,
    Bx_base, By_base, Bz_base,
    Bx_previous2, By_previous2, Bz_previous2,
    Bx_first, By_first, Bz_first,
    Ex_edge, Ey_edge, Ez_edge, rhs_dt_coefficient,
    mu, nu, base_weight, gamma_ratio,
    nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i <= nxp + Int32(1) && j <= nyp && k <= nzp
        ii = i + NG; jj = j + NG; kk = k + NG
        jp = min(j + Int32(1), nyp + Int32(1))
        kp = min(k + Int32(1), nzp + Int32(1))
        @inbounds begin
            current = Bx_face[ii,jj,kk]
            euler = ct_stokes_update_i(
                current,
                Ey_edge[i,j,k], Ey_edge[i,j,kp],
                Ez_edge[i,j,k], Ez_edge[i,jp,k], rhs_dt_coefficient,
            )
            value = ct_rkl2_combine(
                current, Bx_previous2[ii,jj,kk], Bx_base[ii,jj,kk],
                Bx_first[ii,jj,kk], euler-current,
                mu, nu, base_weight, gamma_ratio,
            )
            Bx_previous2[ii,jj,kk] = current
            Bx_face[ii,jj,kk] = value
        end
    end

    if i <= nxp && j <= nyp + Int32(1) && k <= nzp
        ii = i + NG; jj = j + NG; kk = k + NG
        ip = min(i + Int32(1), nxp + Int32(1))
        kp = min(k + Int32(1), nzp + Int32(1))
        @inbounds begin
            current = By_face[ii,jj,kk]
            euler = ct_stokes_update_j(
                current,
                Ex_edge[i,j,k], Ex_edge[i,j,kp],
                Ez_edge[i,j,k], Ez_edge[ip,j,k], rhs_dt_coefficient,
            )
            value = ct_rkl2_combine(
                current, By_previous2[ii,jj,kk], By_base[ii,jj,kk],
                By_first[ii,jj,kk], euler-current,
                mu, nu, base_weight, gamma_ratio,
            )
            By_previous2[ii,jj,kk] = current
            By_face[ii,jj,kk] = value
        end
    end

    if i <= nxp && j <= nyp && k <= nzp + Int32(1)
        ii = i + NG; jj = j + NG; kk = k + NG
        ip = min(i + Int32(1), nxp + Int32(1))
        jp = min(j + Int32(1), nyp + Int32(1))
        @inbounds begin
            current = Bz_face[ii,jj,kk]
            euler = ct_stokes_update_k(
                current,
                Ex_edge[i,j,k], Ex_edge[i,jp,k],
                Ey_edge[i,j,k], Ey_edge[ip,j,k], rhs_dt_coefficient,
            )
            value = ct_rkl2_combine(
                current, Bz_previous2[ii,jj,kk], Bz_base[ii,jj,kk],
                Bz_first[ii,jj,kk], euler-current,
                mu, nu, base_weight, gamma_ratio,
            )
            Bz_previous2[ii,jj,kk] = current
            Bz_face[ii,jj,kk] = value
        end
    end
    return
end

@inline function _ct_recover_cell_b_from_face_fluxes(
    Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    recovery_mode, ii, jj, kk,
    B0x_face=nothing, B0y_face=nothing, B0z_face=nothing,
)
    @inbounds if recovery_mode == CT_CELL_B_POINT6
        return ct_recover_cell_b_point6(
            Bx_face, By_face, Bz_face,
            B0x_face, B0y_face, B0z_face,
            Areai, nxi, nyi, nzi,
            Areaj, nxj, nyj, nzj,
            Areak, nxk, nyk, nzk,
            ii, jj, kk,
        )
    end
    @inbounds begin
        area_i_lo = SVector(
            Areai[ii,jj,kk]*nxi[ii,jj,kk],
            Areai[ii,jj,kk]*nyi[ii,jj,kk],
            Areai[ii,jj,kk]*nzi[ii,jj,kk],
        )
        area_i_hi = SVector(
            Areai[ii+Int32(1),jj,kk]*nxi[ii+Int32(1),jj,kk],
            Areai[ii+Int32(1),jj,kk]*nyi[ii+Int32(1),jj,kk],
            Areai[ii+Int32(1),jj,kk]*nzi[ii+Int32(1),jj,kk],
        )
        area_j_lo = SVector(
            Areaj[ii,jj,kk]*nxj[ii,jj,kk],
            Areaj[ii,jj,kk]*nyj[ii,jj,kk],
            Areaj[ii,jj,kk]*nzj[ii,jj,kk],
        )
        area_j_hi = SVector(
            Areaj[ii,jj+Int32(1),kk]*nxj[ii,jj+Int32(1),kk],
            Areaj[ii,jj+Int32(1),kk]*nyj[ii,jj+Int32(1),kk],
            Areaj[ii,jj+Int32(1),kk]*nzj[ii,jj+Int32(1),kk],
        )
        area_k_lo = SVector(
            Areak[ii,jj,kk]*nxk[ii,jj,kk],
            Areak[ii,jj,kk]*nyk[ii,jj,kk],
            Areak[ii,jj,kk]*nzk[ii,jj,kk],
        )
        area_k_hi = SVector(
            Areak[ii,jj,kk+Int32(1)]*nxk[ii,jj,kk+Int32(1)],
            Areak[ii,jj,kk+Int32(1)]*nyk[ii,jj,kk+Int32(1)],
            Areak[ii,jj,kk+Int32(1)]*nzk[ii,jj,kk+Int32(1)],
        )
        return ct_recover_cell_b(
            area_i_lo, area_i_hi, area_j_lo, area_j_hi,
            area_k_lo, area_k_hi,
            _ct_total_face_flux(Bx_face, B0x_face, ii, jj, kk),
            _ct_total_face_flux(Bx_face, B0x_face, ii+Int32(1), jj, kk),
            _ct_total_face_flux(By_face, B0y_face, ii, jj, kk),
            _ct_total_face_flux(By_face, B0y_face, ii, jj+Int32(1), kk),
            _ct_total_face_flux(Bz_face, B0z_face, ii, jj, kk),
            _ct_total_face_flux(Bz_face, B0z_face, ii, jj, kk+Int32(1)),
        )
    end
end

# CT face fluxes are authoritative. This kernel only refreshes the derived
# cell-centered magnetic cache; it never changes conservative total energy.
function ct_recover_cell_b_kernel!(
    Q, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    recovery_mode, nxp, nyp, nzp, cell_offset,
    B0x_face=nothing, B0y_face=nothing, B0z_face=nothing,
    B0_cell=nothing,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + cell_offset
    jj = j + cell_offset
    kk = k + cell_offset

    split_background = B0_cell !== nothing
    magnetic = _ct_recover_cell_b_from_face_fluxes(
        Bx_face, By_face, Bz_face,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        recovery_mode, ii, jj, kk,
        split_background ? nothing : B0x_face,
        split_background ? nothing : B0y_face,
        split_background ? nothing : B0z_face,
    )
    if split_background
        @inbounds magnetic += SVector(
            B0_cell[ii,jj,kk,1], B0_cell[ii,jj,kk,2],
            B0_cell[ii,jj,kk,3],
        )
    end

    @inbounds begin
        if isfinite(magnetic[1]) && isfinite(magnetic[2]) &&
           isfinite(magnetic[3])
            Q[ii,jj,kk,QBX] = magnetic[1]
            Q[ii,jj,kk,QBY] = magnetic[2]
            Q[ii,jj,kk,QBZ] = magnetic[3]
            Q[ii,jj,kk,QPSI] = zero(FT)
        end
    end
    return
end

# Topological ghosts are reconstructed from the packed authoritative halo.
# Physical ghosts instead own local, boundary-filled face fluxes; recover only
# that disjoint region here after every face-B halo transaction is complete.
function ct_recover_physical_ghost_b_kernel!(
    Q, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    nxp, nyp, nzp, physical_faces,
    B0x_face=nothing, B0y_face=nothing, B0z_face=nothing,
    B0_cell=nothing,
)
    i = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1))*blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1))*blockDim().z + threadIdx().z
    if i > nxp+2NG || j > nyp+2NG || k > nzp+2NG
        return
    end
    if NG < i <= nxp+NG && NG < j <= nyp+NG && NG < k <= nzp+NG
        return
    end
    physical =
        (physical_faces[1] && i <= NG) ||
        (physical_faces[2] && i > nxp+NG) ||
        (physical_faces[3] && j <= NG) ||
        (physical_faces[4] && j > nyp+NG) ||
        (physical_faces[5] && k <= NG) ||
        (physical_faces[6] && k > nzp+NG)
    physical || return

    split_background = B0_cell !== nothing
    magnetic = _ct_recover_cell_b_from_face_fluxes(
        Bx_face, By_face, Bz_face,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        CT_CELL_B_LSQ2, i, j, k,
        split_background ? nothing : B0x_face,
        split_background ? nothing : B0y_face,
        split_background ? nothing : B0z_face,
    )
    if split_background
        @inbounds magnetic += SVector(
            B0_cell[i,j,k,1], B0_cell[i,j,k,2], B0_cell[i,j,k,3],
        )
    end
    @inbounds if isfinite(magnetic[1]) && isfinite(magnetic[2]) &&
                 isfinite(magnetic[3])
        Q[i,j,k,QBX] = magnetic[1]
        Q[i,j,k,QBY] = magnetic[2]
        Q[i,j,k,QBZ] = magnetic[3]
        Q[i,j,k,QPSI] = zero(FT)
    end
    return
end

function ct_recover_physical_ghost_b!(
    b, nxp, nyp, nzp;
    physical_faces::NTuple{6,Bool},
)
    any(physical_faces) || return nothing
    background_faces = hasproperty(b,:B0x_face) ?
        (b.B0x_face,b.B0y_face,b.B0z_face) : (nothing,nothing,nothing)
    background_cell = hasproperty(b,:B0_cell) ? b.B0_cell : nothing
    total = (nxp+2NG,nyp+2NG,nzp+2NG)
    blocks = ntuple(axis -> cld(total[axis],nthreads[axis]),Val(3))
    @gpu_launch threads=nthreads blocks=blocks ct_recover_physical_ghost_b_kernel!(
        b.Q,b.Bx_face,b.By_face,b.Bz_face,
        b.Areai,b.nxi,b.nyi,b.nzi,
        b.Areaj,b.nxj,b.nyj,b.nzj,
        b.Areak,b.nxk,b.nyk,b.nzk,
        Int32(nxp),Int32(nyp),Int32(nzp),physical_faces,
        background_faces...,background_cell,
    )
    return nothing
end

function ct_recover_background_cell_b_kernel!(
    background_cell, B0x_face, B0y_face, B0z_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    recovery_mode, nxp, nyp, nzp, cell_offset,
)
    i = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1))*blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1))*blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii, jj, kk = i+cell_offset, j+cell_offset, k+cell_offset
    magnetic = _ct_recover_cell_b_from_face_fluxes(
        B0x_face, B0y_face, B0z_face,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        recovery_mode, ii, jj, kk,
    )
    @inbounds if isfinite(magnetic[1]) && isfinite(magnetic[2]) &&
                 isfinite(magnetic[3])
        background_cell[ii,jj,kk,1] = magnetic[1]
        background_cell[ii,jj,kk,2] = magnetic[2]
        background_cell[ii,jj,kk,3] = magnetic[3]
    end
    return
end

@inline function _ct_active_shell_offset(layout, i, j, k)
    return ct_shell_offset(layout, i-NG, j-NG, k-NG)
end

@inline function _ct_shell_total_face_flux(
    face_flux, face_shell, background_flux, background_shell,
    layout, i, j, k,
)
    offset = _ct_active_shell_offset(layout, i, j, k)
    if offset == 0
        return _ct_total_face_flux(
            face_flux, background_flux, i, j, k,
        )
    end
    @inbounds value = face_shell[offset]
    if background_shell !== nothing
        @inbounds value += background_shell[offset]
    end
    return value
end

@inline function _ct_shell_face_area_vector(
    area, normal_x, normal_y, normal_z, area_vector_shell,
    layout, i, j, k,
)
    offset = _ct_active_shell_offset(layout, i, j, k)
    if offset == 0
        @inbounds local_area = area[i,j,k]
        @inbounds return SVector(
            local_area*normal_x[i,j,k], local_area*normal_y[i,j,k],
            local_area*normal_z[i,j,k],
        )
    end
    @inbounds return SVector(
        area_vector_shell[offset,1], area_vector_shell[offset,2],
        area_vector_shell[offset,3],
    )
end

@inline function _ct_face_point_data6_shell(
    face_flux, face_shell, background_flux, background_shell,
    area, normal_x, normal_y, normal_z, area_vector_shell,
    layout, i, j, k, ::Val{DIRECTION},
) where {DIRECTION}
    T = eltype(face_flux)
    flux_point = zero(T)
    area_vector_point = MVector{3,T}(zero(T), zero(T), zero(T))
    for first_index in 1:5
        first_offset = first_index - 3
        first_weight = _ct_average_to_point6_weight(T, first_index)
        for second_index in 1:5
            second_offset = second_index - 3
            weight = first_weight *
                     _ct_average_to_point6_weight(T, second_index)
            ii = DIRECTION == 1 ? i : i + first_offset
            jj = DIRECTION == 1 ? j + first_offset :
                 (DIRECTION == 2 ? j : j + second_offset)
            kk = DIRECTION == 3 ? k : k + second_offset
            flux_point += weight * _ct_shell_total_face_flux(
                face_flux, face_shell, background_flux, background_shell,
                layout, ii, jj, kk,
            )
            local_area_vector = _ct_shell_face_area_vector(
                area, normal_x, normal_y, normal_z, area_vector_shell,
                layout, ii, jj, kk,
            )
            area_vector_point[1] += weight*local_area_vector[1]
            area_vector_point[2] += weight*local_area_vector[2]
            area_vector_point[3] += weight*local_area_vector[3]
        end
    end
    return SVector{4,T}(
        flux_point, area_vector_point[1], area_vector_point[2],
        area_vector_point[3],
    )
end

@inline function ct_recover_cell_b_point6_shell(
    face_fluxes, face_shells, background_fluxes, background_shells,
    face_metrics, area_vector_shells, face_layout, i, j, k,
)
    i_faces = ntuple(Val(6)) do face
        _ct_face_point_data6_shell(
            face_fluxes[1], face_shells[1],
            background_fluxes[1], background_shells[1],
            face_metrics[1]..., area_vector_shells[1], face_layout,
            i + face - 3, j, k, Val(1),
        )
    end
    j_faces = ntuple(Val(6)) do face
        _ct_face_point_data6_shell(
            face_fluxes[2], face_shells[2],
            background_fluxes[2], background_shells[2],
            face_metrics[2]..., area_vector_shells[2], face_layout,
            i, j + face - 3, k, Val(2),
        )
    end
    k_faces = ntuple(Val(6)) do face
        _ct_face_point_data6_shell(
            face_fluxes[3], face_shells[3],
            background_fluxes[3], background_shells[3],
            face_metrics[3]..., area_vector_shells[3], face_layout,
            i, j, k + face - 3, Val(3),
        )
    end
    T = eltype(face_fluxes[1])
    centered_i = SVector{4,T}(ntuple(Val(4)) do component
        ct_face_to_center6(SVector{6,T}(ntuple(
            face -> i_faces[face][component], Val(6),
        )))
    end)
    centered_j = SVector{4,T}(ntuple(Val(4)) do component
        ct_face_to_center6(SVector{6,T}(ntuple(
            face -> j_faces[face][component], Val(6),
        )))
    end)
    centered_k = SVector{4,T}(ntuple(Val(4)) do component
        ct_face_to_center6(SVector{6,T}(ntuple(
            face -> k_faces[face][component], Val(6),
        )))
    end)
    return ct_solve_cell_b(
        SVector{3,T}(centered_i[2],centered_i[3],centered_i[4]),
        SVector{3,T}(centered_j[2],centered_j[3],centered_j[4]),
        SVector{3,T}(centered_k[2],centered_k[3],centered_k[4]),
        centered_i[1], centered_j[1], centered_k[1],
    )
end

@inline function ct_conservative_average_to_point6_shell(
    U, conservative_shell, inverse_volume, inverse_volume_shell,
    cell_layout, i, j, k,
)
    T = eltype(U)
    point = MVector{6,T}(
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
    )
    jacobian_point = zero(T)
    @static if @isdefined(Ncell_cons)
        has_psi = Ncell_cons >= 9
    else
        has_psi = size(U,4) >= 9
    end
    for x_index in 1:5
        di = x_index - 3
        wx = _ct_average_to_point6_weight(T, x_index)
        for y_index in 1:5
            dj = y_index - 3
            wxy = wx*_ct_average_to_point6_weight(T, y_index)
            for z_index in 1:5
                dk = z_index - 3
                weight = wxy*_ct_average_to_point6_weight(T, z_index)
                ii, jj, kk = i+di, j+dj, k+dk
                offset = _ct_active_shell_offset(cell_layout, ii, jj, kk)
                if offset == 0
                    @inbounds local_inverse_volume = inverse_volume[ii,jj,kk]
                else
                    @inbounds local_inverse_volume = inverse_volume_shell[offset]
                end
                jacobian_average = one(T)/local_inverse_volume
                weighted_jacobian = weight*jacobian_average
                jacobian_point += weighted_jacobian
                for component in 1:5
                    value = if offset == 0
                        @inbounds U[ii,jj,kk,component]
                    else
                        @inbounds conservative_shell[offset,component]
                    end
                    point[component] += weighted_jacobian*value
                end
                if has_psi
                    value = if offset == 0
                        @inbounds U[ii,jj,kk,9]
                    elseif size(conservative_shell,2) >= 9
                        @inbounds conservative_shell[offset,9]
                    else
                        zero(T)
                    end
                    point[6] += weighted_jacobian*value
                end
            end
        end
    end
    inverse_jacobian = one(T)/jacobian_point
    return SVector{6,T}(ntuple(
        component -> point[component]*inverse_jacobian, Val(6),
    ))
end

@inline function ct_recover_cell_b_lsq2_shell(
    face_fluxes, face_shells, background_fluxes, background_shells,
    face_metrics, area_vector_shells, face_layout, i, j, k,
)
    area_i_lo = _ct_shell_face_area_vector(
        face_metrics[1]...,area_vector_shells[1],face_layout,i,j,k,
    )
    area_i_hi = _ct_shell_face_area_vector(
        face_metrics[1]...,area_vector_shells[1],face_layout,i+1,j,k,
    )
    area_j_lo = _ct_shell_face_area_vector(
        face_metrics[2]...,area_vector_shells[2],face_layout,i,j,k,
    )
    area_j_hi = _ct_shell_face_area_vector(
        face_metrics[2]...,area_vector_shells[2],face_layout,i,j+1,k,
    )
    area_k_lo = _ct_shell_face_area_vector(
        face_metrics[3]...,area_vector_shells[3],face_layout,i,j,k,
    )
    area_k_hi = _ct_shell_face_area_vector(
        face_metrics[3]...,area_vector_shells[3],face_layout,i,j,k+1,
    )
    flux_i_lo = _ct_shell_total_face_flux(
        face_fluxes[1],face_shells[1],background_fluxes[1],
        background_shells[1],face_layout,i,j,k,
    )
    flux_i_hi = _ct_shell_total_face_flux(
        face_fluxes[1],face_shells[1],background_fluxes[1],
        background_shells[1],face_layout,i+1,j,k,
    )
    flux_j_lo = _ct_shell_total_face_flux(
        face_fluxes[2],face_shells[2],background_fluxes[2],
        background_shells[2],face_layout,i,j,k,
    )
    flux_j_hi = _ct_shell_total_face_flux(
        face_fluxes[2],face_shells[2],background_fluxes[2],
        background_shells[2],face_layout,i,j+1,k,
    )
    flux_k_lo = _ct_shell_total_face_flux(
        face_fluxes[3],face_shells[3],background_fluxes[3],
        background_shells[3],face_layout,i,j,k,
    )
    flux_k_hi = _ct_shell_total_face_flux(
        face_fluxes[3],face_shells[3],background_fluxes[3],
        background_shells[3],face_layout,i,j,k+1,
    )
    return ct_recover_cell_b(
        area_i_lo,area_i_hi,area_j_lo,area_j_hi,area_k_lo,area_k_hi,
        flux_i_lo,flux_i_hi,flux_j_lo,flux_j_hi,flux_k_lo,flux_k_hi,
    )
end

@inline function ct_conservative_direct_shell(
    U, conservative_shell, cell_layout, i, j, k,
)
    T = eltype(U)
    offset = _ct_active_shell_offset(cell_layout,i,j,k)
    if offset == 0
        @inbounds begin
            density = U[i,j,k,1]
            momentum_x = U[i,j,k,2]
            momentum_y = U[i,j,k,3]
            momentum_z = U[i,j,k,4]
            energy = U[i,j,k,5]
        end
    else
        @inbounds begin
            density = conservative_shell[offset,1]
            momentum_x = conservative_shell[offset,2]
            momentum_y = conservative_shell[offset,3]
            momentum_z = conservative_shell[offset,4]
            energy = conservative_shell[offset,5]
        end
    end
    return SVector{6,T}(
        density,momentum_x,momentum_y,momentum_z,energy,zero(T),
    )
end

@inline function _ct_point6_shell_stencil(
    U, conservative_shell, cell_layout, i, j, k, ::Val{DIRECTION},
) where {DIRECTION}
    return ntuple(Val(5)) do index
        offset = index-3
        ii = DIRECTION == 1 ? i+offset : i
        jj = DIRECTION == 2 ? j+offset : j
        kk = DIRECTION == 3 ? k+offset : k
        ct_conservative_direct_shell(
            U,conservative_shell,cell_layout,ii,jj,kk,
        )
    end
end

@inline function ct_point6_shell_ao_coefficients(
    U, conservative_shell, cell_layout, i, j, k,
)
    fixed = _ct_point6_fixed_coefficients(eltype(U))
    homogeneous = _ct_point6_homogeneous_axes()
    sensor_i,sensor_j,sensor_k = _ct_point6_sensor_center(i,j,k)
    x_coefficients, x_sensor = homogeneous[1] ? (fixed,zero(eltype(U))) :
        ct_point6_ao_direction_coefficients(
            _ct_point6_shell_stencil(
                U,conservative_shell,cell_layout,
                sensor_i,sensor_j,sensor_k,Val(1),
            ),
        )
    y_coefficients, y_sensor = homogeneous[2] ? (fixed,zero(eltype(U))) :
        ct_point6_ao_direction_coefficients(
            _ct_point6_shell_stencil(
                U,conservative_shell,cell_layout,
                sensor_i,sensor_j,sensor_k,Val(2),
            ),
        )
    z_coefficients, z_sensor = homogeneous[3] ? (fixed,zero(eltype(U))) :
        ct_point6_ao_direction_coefficients(
            _ct_point6_shell_stencil(
                U,conservative_shell,cell_layout,
                sensor_i,sensor_j,sensor_k,Val(3),
            ),
        )
    return x_coefficients, y_coefficients, z_coefficients,
           x_sensor, y_sensor, z_sensor
end

@inline function ct_conservative_average_to_point_coefficients_shell(
    U, conservative_shell, inverse_volume, inverse_volume_shell,
    cell_layout, i, j, k,
    x_coefficients::SVector{5,T}, y_coefficients::SVector{5,T},
    z_coefficients::SVector{5,T},
) where {T}
    point = MVector{6,T}(
        zero(T),zero(T),zero(T),zero(T),zero(T),zero(T),
    )
    jacobian_point = zero(T)
    @static if @isdefined(Ncell_cons)
        has_psi = Ncell_cons >= 9
    else
        has_psi = size(U,4) >= 9
    end
    for x_index in 1:5
        di = x_index-3
        for y_index in 1:5
            dj = y_index-3
            for z_index in 1:5
                dk = z_index-3
                weight = x_coefficients[x_index]*y_coefficients[y_index]*
                         z_coefficients[z_index]
                ii, jj, kk = i+di, j+dj, k+dk
                offset = _ct_active_shell_offset(cell_layout,ii,jj,kk)
                local_inverse_volume = if offset == 0
                    @inbounds inverse_volume[ii,jj,kk]
                else
                    @inbounds inverse_volume_shell[offset]
                end
                jacobian_average = one(T)/local_inverse_volume
                weighted_jacobian = weight*jacobian_average
                jacobian_point += weighted_jacobian
                for component in 1:5
                    value = if offset == 0
                        @inbounds U[ii,jj,kk,component]
                    else
                        @inbounds conservative_shell[offset,component]
                    end
                    point[component] += weighted_jacobian*value
                end
                if has_psi
                    value = if offset == 0
                        @inbounds U[ii,jj,kk,9]
                    elseif size(conservative_shell,2) >= 9
                        @inbounds conservative_shell[offset,9]
                    else
                        zero(T)
                    end
                    point[6] += weighted_jacobian*value
                end
            end
        end
    end
    if !(isfinite(jacobian_point) && jacobian_point > zero(T))
        bad = T(NaN)
        return SVector{6,T}(ntuple(_ -> bad,Val(6)))
    end
    return SVector{6,T}(point/jacobian_point)
end

@inline function ct_conservative_average_to_point_adaptive_shell(
    U, conservative_shell, inverse_volume, inverse_volume_shell,
    cell_layout, i, j, k,
)
    T = eltype(U)
    fixed = _ct_point6_fixed_coefficients(T)
    if !_ct_point6_adaptive_enabled()
        hydro = ct_conservative_average_to_point6_shell(
            U,conservative_shell,inverse_volume,inverse_volume_shell,
            cell_layout,i,j,k,
        )
        return hydro, fixed, fixed, fixed, false
    end
    x_ao, y_ao, z_ao, x_sensor, y_sensor, z_sensor =
        ct_point6_shell_ao_coefficients(
            U,conservative_shell,cell_layout,i,j,k,
        )
    threshold = _ct_point6_sensor_threshold(T)
    x_coefficients,x_used = ct_point6_select_direction_coefficients(
        fixed,x_ao,x_sensor,threshold,
    )
    y_coefficients,y_used = ct_point6_select_direction_coefficients(
        fixed,y_ao,y_sensor,threshold,
    )
    z_coefficients,z_used = ct_point6_select_direction_coefficients(
        fixed,z_ao,z_sensor,threshold,
    )
    used_ao = x_used || y_used || z_used
    if !used_ao
        hydro = ct_conservative_average_to_point6_shell(
            U,conservative_shell,inverse_volume,inverse_volume_shell,
            cell_layout,i,j,k,
        )
        return hydro, x_ao, y_ao, z_ao, false
    end
    hydro = ct_conservative_average_to_point_coefficients_shell(
        U,conservative_shell,inverse_volume,inverse_volume_shell,
        cell_layout,i,j,k,
        x_coefficients,y_coefficients,z_coefficients,
    )
    return hydro, x_ao, y_ao, z_ao, true
end

@inline function _ct_record_point6_recovery!(meta, mode::Int32)
    meta === nothing && return nothing
    @static if @isdefined(CT_POS_POINT6_TO_AO_COUNT)
        if mode == Int32(1) || mode == Int32(2)
            ct_record_fallback!(meta,CT_POS_POINT6_TO_AO_COUNT)
        end
        if mode == Int32(2)
            ct_record_fallback!(meta,CT_POS_POINT6_LIMIT_COUNT)
        end
        if mode == Int32(3)
            ct_record_fallback!(meta,CT_POS_POINT6_UNRECOVERABLE_COUNT)
        end
    end
    return nothing
end

@inline function _ct_store_q_direct_shell!(
    Q, U, background_cell, conservative_shell, cell_layout,
    face_fluxes, face_shells, background_fluxes, background_shells,
    face_metrics, area_vector_shells, face_layout,
    i, j, k, gamma, gas_constant,
)
    magnetic = ct_recover_cell_b_lsq2_shell(
        face_fluxes,face_shells,background_fluxes,background_shells,
        face_metrics,area_vector_shells,face_layout,i,j,k,
    )
    hydro = ct_conservative_direct_shell(
        U,conservative_shell,cell_layout,i,j,k,
    )
    primitive = ct_mhd_point_conservative_to_primitive(
        hydro,magnetic,gamma,
    )
    if background_cell !== nothing
        no_background = (nothing,nothing,nothing)
        background = ct_recover_cell_b_lsq2_shell(
            background_fluxes,background_shells,no_background,no_background,
            face_metrics,area_vector_shells,face_layout,i,j,k,
        )
        @inbounds begin
            background_cell[i,j,k,1]=background[1]
            background_cell[i,j,k,2]=background[2]
            background_cell[i,j,k,3]=background[3]
        end
    end
    temperature = primitive[5]/(primitive[1]*gas_constant)
    @inbounds begin
        Q[i,j,k,1]=primitive[1]; Q[i,j,k,2]=primitive[2]
        Q[i,j,k,3]=primitive[3]; Q[i,j,k,4]=primitive[4]
        Q[i,j,k,5]=primitive[5]; Q[i,j,k,6]=temperature
        Q[i,j,k,QBX]=primitive[6]; Q[i,j,k,QBY]=primitive[7]
        Q[i,j,k,QBZ]=primitive[8]; Q[i,j,k,QPSI]=zero(FT)
    end
    return
end

function ct_derive_q_direct_ghost_kernel!(
    Q, U, background_cell, conservative_shell, cell_layout,
    face_fluxes, face_shells, background_fluxes, background_shells,
    face_metrics, area_vector_shells, face_layout,
    nxp, nyp, nzp, gamma, gas_constant, physical_faces,
)
    i = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1))*blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1))*blockDim().z + threadIdx().z
    if i > nxp+2NG || j > nyp+2NG || k > nzp+2NG
        return
    end
    if NG < i <= nxp+NG && NG < j <= nyp+NG && NG < k <= nzp+NG
        return
    end
    if (physical_faces[1] && i <= NG) ||
       (physical_faces[2] && i > nxp+NG) ||
       (physical_faces[3] && j <= NG) ||
       (physical_faces[4] && j > nyp+NG) ||
       (physical_faces[5] && k <= NG) ||
       (physical_faces[6] && k > nzp+NG)
        return
    end
    _ct_store_q_direct_shell!(
        Q,U,background_cell,conservative_shell,cell_layout,
        face_fluxes,face_shells,background_fluxes,background_shells,
        face_metrics,area_vector_shells,face_layout,
        i,j,k,gamma,gas_constant,
    )
    return
end

function ct_derive_q_direct_ghost!(
    b, halo, nxp, nyp, nzp;
    gamma,
    gas_constant,
    physical_faces::NTuple{6,Bool}=ntuple(_ -> false,Val(6)),
)
    face_fluxes=(b.Bx_face,b.By_face,b.Bz_face)
    background_fluxes=hasproperty(b,:B0x_face) ?
        (b.B0x_face,b.B0y_face,b.B0z_face) : (nothing,nothing,nothing)
    background_cell=hasproperty(b,:B0_cell) ? b.B0_cell : nothing
    face_metrics=(
        (b.Areai,b.nxi,b.nyi,b.nzi),
        (b.Areaj,b.nxj,b.nyj,b.nzj),
        (b.Areak,b.nxk,b.nyk,b.nzk),
    )
    total=(nxp+2NG,nyp+2NG,nzp+2NG)
    blocks=ntuple(axis -> cld(total[axis],nthreads[axis]),Val(3))
    @gpu_launch threads=nthreads blocks=blocks ct_derive_q_direct_ghost_kernel!(
        b.Q,b.U,background_cell,halo.conservative_shell,halo.cell_layout,
        face_fluxes,halo.face_b_shells,background_fluxes,
        halo.background_face_shells,face_metrics,halo.face_area_vector_shells,
        halo.face_layout,Int32(nxp),Int32(nyp),Int32(nzp),
        FT(gamma),FT(gas_constant),physical_faces,
    )
    return nothing
end

@inline function _ct_store_q_point6_shell!(
    Q, U, background_cell, conservative_shell,
    inverse_volume, inverse_volume_shell,
    cell_layout, face_fluxes, face_shells,
    background_fluxes, background_shells, face_metrics,
    area_vector_shells, face_layout, positivity_meta,
    i, j, k, gamma, gas_constant,
)
    high_magnetic = ct_recover_cell_b_point6_shell(
        face_fluxes, face_shells, background_fluxes, background_shells,
        face_metrics, area_vector_shells, face_layout, i, j, k,
    )
    low_magnetic = ct_recover_cell_b_lsq2_shell(
        face_fluxes,face_shells,background_fluxes,background_shells,
        face_metrics,area_vector_shells,face_layout,i,j,k,
    )
    candidate_hydro, x_ao, y_ao, z_ao, used_ao =
        ct_conservative_average_to_point_adaptive_shell(
            U,conservative_shell,inverse_volume,inverse_volume_shell,
            cell_layout,i,j,k,
        )
    ao_hydro = used_ao ? candidate_hydro :
        ct_conservative_average_to_point_coefficients_shell(
            U,conservative_shell,inverse_volume,inverse_volume_shell,
            cell_layout,i,j,k,x_ao,y_ao,z_ao,
        )
    low_hydro = ct_conservative_direct_shell(
        U,conservative_shell,cell_layout,i,j,k,
    )
    hydro, magnetic, theta, recovery_mode =
        ct_point6_select_admissible_state(
            candidate_hydro,ao_hydro,low_hydro,high_magnetic,low_magnetic,
            gamma,eltype(U)(density_floor),eltype(U)(pressure_floor),used_ao,
        )
    _ct_record_point6_recovery!(positivity_meta,recovery_mode)
    primitive = ct_mhd_point_conservative_to_primitive(
        hydro,magnetic,gamma,
    )
    if background_cell !== nothing
        no_background = (nothing,nothing,nothing)
        high_background = ct_recover_cell_b_point6_shell(
            background_fluxes,background_shells,
            no_background,no_background,
            face_metrics,area_vector_shells,face_layout,i,j,k,
        )
        background = high_background
        if theta < one(theta)
            low_background = ct_recover_cell_b_lsq2_shell(
                background_fluxes,background_shells,
                no_background,no_background,
                face_metrics,area_vector_shells,face_layout,i,j,k,
            )
            background = _ct_point6_blend(
                low_background,high_background,theta,
            )
        end
        @inbounds begin
            background_cell[i,j,k,1]=background[1]
            background_cell[i,j,k,2]=background[2]
            background_cell[i,j,k,3]=background[3]
        end
    end
    temperature = primitive[5]/(primitive[1]*gas_constant)
    @inbounds begin
        Q[i,j,k,1]=primitive[1]; Q[i,j,k,2]=primitive[2]
        Q[i,j,k,3]=primitive[3]; Q[i,j,k,4]=primitive[4]
        Q[i,j,k,5]=primitive[5]; Q[i,j,k,6]=temperature
        Q[i,j,k,QBX]=primitive[6]; Q[i,j,k,QBY]=primitive[7]
        Q[i,j,k,QBZ]=primitive[8]; Q[i,j,k,QPSI]=zero(FT)
    end
    return
end

function ct_derive_q_point6_ghost_kernel!(
    Q, U, background_cell, conservative_shell,
    inverse_volume, inverse_volume_shell,
    cell_layout, face_fluxes, face_shells,
    background_fluxes, background_shells, face_metrics,
    area_vector_shells, face_layout, positivity_meta, nxp, nyp, nzp,
    gamma, gas_constant, physical_faces,
)
    i = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1))*blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1))*blockDim().z + threadIdx().z
    nx_total, ny_total, nz_total = nxp+2NG, nyp+2NG, nzp+2NG
    if i > nx_total || j > ny_total || k > nz_total
        return
    end
    if NG < i <= nxp+NG && NG < j <= nyp+NG && NG < k <= nzp+NG
        return
    end
    if (physical_faces[1] && i <= NG) ||
       (physical_faces[2] && i > nxp+NG) ||
       (physical_faces[3] && j <= NG) ||
       (physical_faces[4] && j > nyp+NG) ||
       (physical_faces[5] && k <= NG) ||
       (physical_faces[6] && k > nzp+NG)
        return
    end
    _ct_store_q_point6_shell!(
        Q,U,background_cell,conservative_shell,
        inverse_volume,inverse_volume_shell,
        cell_layout,face_fluxes,face_shells,background_fluxes,
        background_shells,face_metrics,area_vector_shells,face_layout,
        positivity_meta,i,j,k,gamma,gas_constant,
    )
    return
end

function ct_derive_q_point6_topological_active_kernel!(
    Q, U, background_cell, conservative_shell,
    inverse_volume, inverse_volume_shell,
    cell_layout, face_fluxes, face_shells,
    background_fluxes, background_shells, face_metrics,
    area_vector_shells, face_layout, positivity_meta, nxp, nyp, nzp,
    gamma, gas_constant, physical_faces,
)
    i = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1))*blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1))*blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    reach = Int32(2)
    touches_topology =
        (i <= reach && !physical_faces[1]) ||
        (i > nxp-reach && !physical_faces[2]) ||
        (j <= reach && !physical_faces[3]) ||
        (j > nyp-reach && !physical_faces[4]) ||
        (k <= reach && !physical_faces[5]) ||
        (k > nzp-reach && !physical_faces[6])
    touches_physical =
        (i <= reach && physical_faces[1]) ||
        (i > nxp-reach && physical_faces[2]) ||
        (j <= reach && physical_faces[3]) ||
        (j > nyp-reach && physical_faces[4]) ||
        (k <= reach && physical_faces[5]) ||
        (k > nzp-reach && physical_faces[6])
    (touches_topology && !touches_physical) || return
    _ct_store_q_point6_shell!(
        Q,U,background_cell,conservative_shell,
        inverse_volume,inverse_volume_shell,
        cell_layout,face_fluxes,face_shells,background_fluxes,
        background_shells,face_metrics,area_vector_shells,face_layout,
        positivity_meta,i+NG,j+NG,k+NG,gamma,gas_constant,
    )
    return
end

function ct_derive_q_point6_ghost!(
    b, halo, nxp, nyp, nzp;
    physical_faces::NTuple{6,Bool}=ntuple(_ -> false, Val(6)),
    positivity_meta=nothing,
)
    face_fluxes = (b.Bx_face,b.By_face,b.Bz_face)
    background_fluxes = hasproperty(b,:B0x_face) ?
        (b.B0x_face,b.B0y_face,b.B0z_face) : (nothing,nothing,nothing)
    background_cell = hasproperty(b,:B0_cell) ? b.B0_cell : nothing
    face_metrics = (
        (b.Areai,b.nxi,b.nyi,b.nzi),
        (b.Areaj,b.nxj,b.nyj,b.nzj),
        (b.Areak,b.nxk,b.nyk,b.nzk),
    )
    total = (nxp+2NG,nyp+2NG,nzp+2NG)
    blocks = ntuple(axis -> cld(total[axis],nthreads[axis]), Val(3))
    @gpu_launch threads=nthreads blocks=blocks ct_derive_q_point6_ghost_kernel!(
        b.Q,b.U,background_cell,halo.conservative_shell,
        b.Vol,halo.inverse_volume_shell,
        halo.cell_layout,face_fluxes,halo.face_b_shells,
        background_fluxes,halo.background_face_shells,face_metrics,
        halo.face_area_vector_shells,halo.face_layout,positivity_meta,
        Int32(nxp),Int32(nyp),Int32(nzp),FT(γ),FT(Rg),physical_faces)
    return nothing
end

function ct_derive_q_point6_topological_active!(
    b, halo, nxp, nyp, nzp;
    physical_faces::NTuple{6,Bool}=ntuple(_ -> false, Val(6)),
    positivity_meta=nothing,
)
    face_fluxes = (b.Bx_face,b.By_face,b.Bz_face)
    background_fluxes = hasproperty(b,:B0x_face) ?
        (b.B0x_face,b.B0y_face,b.B0z_face) : (nothing,nothing,nothing)
    background_cell = hasproperty(b,:B0_cell) ? b.B0_cell : nothing
    face_metrics = (
        (b.Areai,b.nxi,b.nyi,b.nzi),
        (b.Areaj,b.nxj,b.nyj,b.nzj),
        (b.Areak,b.nxk,b.nyk,b.nzk),
    )
    blocks = (
        cld(nxp,nthreads[1]), cld(nyp,nthreads[2]), cld(nzp,nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=blocks ct_derive_q_point6_topological_active_kernel!(
        b.Q,b.U,background_cell,halo.conservative_shell,
        b.Vol,halo.inverse_volume_shell,
        halo.cell_layout,face_fluxes,halo.face_b_shells,
        background_fluxes,halo.background_face_shells,face_metrics,
        halo.face_area_vector_shells,halo.face_layout,positivity_meta,
        Int32(nxp),Int32(nyp),Int32(nzp),FT(γ),FT(Rg),physical_faces)
    return nothing
end

@inline function _ct_select_troubled_point_state(
    level,ao_hydro,low_hydro,low_magnetic,gamma,
)
    T=eltype(low_hydro)
    if level >= T(CT_TROUBLED_STRONG)
        return low_hydro,low_magnetic,zero(T),Int32(2)
    end
    if ct_point6_state_is_admissible(
        ao_hydro,low_magnetic,gamma,T(density_floor),T(pressure_floor),
    )
        return ao_hydro,low_magnetic,one(T),Int32(1)
    end
    hydro,magnetic,theta,limited=ct_point6_convex_limit(
        ao_hydro,low_magnetic,low_hydro,low_magnetic,gamma,
        T(density_floor),T(pressure_floor),
    )
    return hydro,magnetic,theta,limited ? Int32(2) : Int32(3)
end

@inline function _ct_store_troubled_q_local!(
    Q,U,mask,inverse_volume,
    face_fluxes,background_fluxes,face_metrics,background_cell,
    positivity_meta,i,j,k,
)
    @inbounds level=mask[i,j,k]
    level >= eltype(mask)(CT_TROUBLED_RECOVERABLE) || return
    low_hydro=_ct_point6_cell_conservative(U,i,j,k)
    low_magnetic=_ct_recover_cell_b_from_face_fluxes(
        face_fluxes[1],face_fluxes[2],face_fluxes[3],
        face_metrics[1]...,face_metrics[2]...,face_metrics[3]...,
        CT_CELL_B_LSQ2,i,j,k,background_fluxes...,
    )
    if level >= eltype(mask)(CT_TROUBLED_STRONG)
        ao_hydro=low_hydro
    else
        x_ao,y_ao,z_ao,_,_,_=ct_point6_active_ao_coefficients(U,i,j,k)
        ao_hydro=ct_conservative_average_to_point_coefficients(
            U,inverse_volume,i,j,k,x_ao,y_ao,z_ao,
        )
    end
    hydro,magnetic,_,recovery_mode=_ct_select_troubled_point_state(
        level,ao_hydro,low_hydro,low_magnetic,FT(γ),
    )
    _ct_record_point6_recovery!(positivity_meta,recovery_mode)
    primitive=ct_mhd_point_conservative_to_primitive(
        hydro,magnetic,FT(γ),
    )
    if background_cell !== nothing
        background=_ct_recover_cell_b_from_face_fluxes(
            background_fluxes[1],background_fluxes[2],background_fluxes[3],
            face_metrics[1]...,face_metrics[2]...,face_metrics[3]...,
            CT_CELL_B_LSQ2,i,j,k,
        )
        @inbounds begin
            background_cell[i,j,k,1]=background[1]
            background_cell[i,j,k,2]=background[2]
            background_cell[i,j,k,3]=background[3]
        end
    end
    temperature=primitive[5]/(primitive[1]*FT(Rg))
    @inbounds begin
        Q[i,j,k,1]=primitive[1]; Q[i,j,k,2]=primitive[2]
        Q[i,j,k,3]=primitive[3]; Q[i,j,k,4]=primitive[4]
        Q[i,j,k,5]=primitive[5]; Q[i,j,k,6]=temperature
        Q[i,j,k,QBX]=primitive[6]; Q[i,j,k,QBY]=primitive[7]
        Q[i,j,k,QBZ]=primitive[8]; Q[i,j,k,QPSI]=primitive[9]
    end
    return
end

function ct_apply_troubled_q_kernel!(
    Q,U,mask,inverse_volume,
    face_fluxes,background_fluxes,face_metrics,background_cell,
    positivity_meta,nxp,nyp,nzp,
)
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k=(blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    (i>nxp || j>nyp || k>nzp) && return
    _ct_store_troubled_q_local!(
        Q,U,mask,inverse_volume,face_fluxes,background_fluxes,face_metrics,
        background_cell,positivity_meta,i+NG,j+NG,k+NG,
    )
    return
end

@inline function _ct_store_troubled_q_shell!(
    Q,U,mask,background_cell,conservative_shell,
    inverse_volume,inverse_volume_shell,cell_layout,
    face_fluxes,face_shells,background_fluxes,background_shells,
    face_metrics,area_vector_shells,face_layout,positivity_meta,i,j,k,
)
    @inbounds level=mask[i,j,k]
    level >= eltype(mask)(CT_TROUBLED_RECOVERABLE) || return
    low_hydro=ct_conservative_direct_shell(
        U,conservative_shell,cell_layout,i,j,k,
    )
    low_magnetic=ct_recover_cell_b_lsq2_shell(
        face_fluxes,face_shells,background_fluxes,background_shells,
        face_metrics,area_vector_shells,face_layout,i,j,k,
    )
    if level >= eltype(mask)(CT_TROUBLED_STRONG)
        ao_hydro=low_hydro
    else
        x_ao,y_ao,z_ao,_,_,_=ct_point6_shell_ao_coefficients(
            U,conservative_shell,cell_layout,i,j,k,
        )
        ao_hydro=ct_conservative_average_to_point_coefficients_shell(
            U,conservative_shell,inverse_volume,inverse_volume_shell,
            cell_layout,i,j,k,x_ao,y_ao,z_ao,
        )
    end
    hydro,magnetic,_,recovery_mode=_ct_select_troubled_point_state(
        level,ao_hydro,low_hydro,low_magnetic,FT(γ),
    )
    _ct_record_point6_recovery!(positivity_meta,recovery_mode)
    primitive=ct_mhd_point_conservative_to_primitive(
        hydro,magnetic,FT(γ),
    )
    if background_cell !== nothing
        no_background=(nothing,nothing,nothing)
        background=ct_recover_cell_b_lsq2_shell(
            background_fluxes,background_shells,no_background,no_background,
            face_metrics,area_vector_shells,face_layout,i,j,k,
        )
        @inbounds begin
            background_cell[i,j,k,1]=background[1]
            background_cell[i,j,k,2]=background[2]
            background_cell[i,j,k,3]=background[3]
        end
    end
    temperature=primitive[5]/(primitive[1]*FT(Rg))
    @inbounds begin
        Q[i,j,k,1]=primitive[1]; Q[i,j,k,2]=primitive[2]
        Q[i,j,k,3]=primitive[3]; Q[i,j,k,4]=primitive[4]
        Q[i,j,k,5]=primitive[5]; Q[i,j,k,6]=temperature
        Q[i,j,k,QBX]=primitive[6]; Q[i,j,k,QBY]=primitive[7]
        Q[i,j,k,QBZ]=primitive[8]; Q[i,j,k,QPSI]=primitive[9]
    end
    return
end

function ct_apply_troubled_q_ghost_kernel!(
    Q,U,mask,background_cell,conservative_shell,
    inverse_volume,inverse_volume_shell,cell_layout,
    face_fluxes,face_shells,background_fluxes,background_shells,
    face_metrics,area_vector_shells,face_layout,positivity_meta,
    nxp,nyp,nzp,physical_faces,
)
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k=(blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    (i>nxp+2NG || j>nyp+2NG || k>nzp+2NG) && return
    if NG<i<=nxp+NG && NG<j<=nyp+NG && NG<k<=nzp+NG
        return
    end
    if (physical_faces[1] && i<=NG) ||
       (physical_faces[2] && i>nxp+NG) ||
       (physical_faces[3] && j<=NG) ||
       (physical_faces[4] && j>nyp+NG) ||
       (physical_faces[5] && k<=NG) ||
       (physical_faces[6] && k>nzp+NG)
        return
    end
    _ct_store_troubled_q_shell!(
        Q,U,mask,background_cell,conservative_shell,
        inverse_volume,inverse_volume_shell,cell_layout,
        face_fluxes,face_shells,background_fluxes,background_shells,
        face_metrics,area_vector_shells,face_layout,positivity_meta,i,j,k,
    )
    return
end

function ct_apply_troubled_q_topological_active_kernel!(
    Q,U,mask,background_cell,conservative_shell,
    inverse_volume,inverse_volume_shell,cell_layout,
    face_fluxes,face_shells,background_fluxes,background_shells,
    face_metrics,area_vector_shells,face_layout,positivity_meta,
    nxp,nyp,nzp,physical_faces,
)
    i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j=(blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k=(blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    (i>nxp || j>nyp || k>nzp) && return
    reach=Int32(2)
    touches_topology=
        (i<=reach && !physical_faces[1]) ||
        (i>nxp-reach && !physical_faces[2]) ||
        (j<=reach && !physical_faces[3]) ||
        (j>nyp-reach && !physical_faces[4]) ||
        (k<=reach && !physical_faces[5]) ||
        (k>nzp-reach && !physical_faces[6])
    touches_physical=
        (i<=reach && physical_faces[1]) ||
        (i>nxp-reach && physical_faces[2]) ||
        (j<=reach && physical_faces[3]) ||
        (j>nyp-reach && physical_faces[4]) ||
        (k<=reach && physical_faces[5]) ||
        (k>nzp-reach && physical_faces[6])
    (touches_topology && !touches_physical) || return
    _ct_store_troubled_q_shell!(
        Q,U,mask,background_cell,conservative_shell,
        inverse_volume,inverse_volume_shell,cell_layout,
        face_fluxes,face_shells,background_fluxes,background_shells,
        face_metrics,area_vector_shells,face_layout,positivity_meta,
        i+NG,j+NG,k+NG,
    )
    return
end

function ct_apply_troubled_q!(
    b,halo,nxp,nyp,nzp;
    physical_faces::NTuple{6,Bool}=ntuple(_->false,Val(6)),
    positivity_meta=nothing,
)
    face_fluxes=(b.Bx_face,b.By_face,b.Bz_face)
    background_fluxes=hasproperty(b,:B0x_face) ?
        (b.B0x_face,b.B0y_face,b.B0z_face) : (nothing,nothing,nothing)
    background_cell=hasproperty(b,:B0_cell) ? b.B0_cell : nothing
    face_metrics=(
        (b.Areai,b.nxi,b.nyi,b.nzi),
        (b.Areaj,b.nxj,b.nyj,b.nzj),
        (b.Areak,b.nxk,b.nyk,b.nzk),
    )
    active_blocks=(
        cld(nxp,nthreads[1]),cld(nyp,nthreads[2]),cld(nzp,nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=active_blocks ct_apply_troubled_q_kernel!(
        b.Q,b.U,b.ϕ,b.Vol,face_fluxes,background_fluxes,face_metrics,
        background_cell,positivity_meta,Int32(nxp),Int32(nyp),Int32(nzp),
    )
    @gpu_launch threads=nthreads blocks=active_blocks ct_apply_troubled_q_topological_active_kernel!(
        b.Q,b.U,b.ϕ,background_cell,halo.conservative_shell,
        b.Vol,halo.inverse_volume_shell,halo.cell_layout,
        face_fluxes,halo.face_b_shells,background_fluxes,
        halo.background_face_shells,face_metrics,halo.face_area_vector_shells,
        halo.face_layout,positivity_meta,Int32(nxp),Int32(nyp),Int32(nzp),
        physical_faces,
    )
    total=(nxp+2NG,nyp+2NG,nzp+2NG)
    ghost_blocks=ntuple(axis->cld(total[axis],nthreads[axis]),Val(3))
    @gpu_launch threads=nthreads blocks=ghost_blocks ct_apply_troubled_q_ghost_kernel!(
        b.Q,b.U,b.ϕ,background_cell,halo.conservative_shell,
        b.Vol,halo.inverse_volume_shell,halo.cell_layout,
        face_fluxes,halo.face_b_shells,background_fluxes,
        halo.background_face_shells,face_metrics,halo.face_area_vector_shells,
        halo.face_layout,positivity_meta,Int32(nxp),Int32(nyp),Int32(nzp),
        physical_faces,
    )
    return nothing
end

@inline function ct_local_periodic_directions(
    bid,face_bc,block_nprocs,periodic_code,
)
    return ntuple(Val(3)) do direction
        low_face = 2direction - 1
        high_face = 2direction
        get(face_bc,(bid,low_face),Int32(-1)) == periodic_code &&
        get(face_bc,(bid,high_face),Int32(-1)) == periodic_code &&
        block_nprocs[bid+1][direction] == 1
    end
end

function _ct_launch_recover_cell_b!(
    b, recovery_mode::Int32,
    range_nxp::Int32, range_nyp::Int32, range_nzp::Int32,
    cell_offset::Int32,
    B0x_face=nothing, B0y_face=nothing, B0z_face=nothing,
    B0_cell=nothing,
)
    nb = (
        cld(range_nxp, nthreads[1]),
        cld(range_nyp, nthreads[2]),
        cld(range_nzp, nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=nb ct_recover_cell_b_kernel!(
        b.Q, b.Bx_face, b.By_face, b.Bz_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        recovery_mode, range_nxp, range_nyp, range_nzp, cell_offset,
        B0x_face, B0y_face, B0z_face, B0_cell)
    return nothing
end

function ct_recover_cell_b!(
    b, nxp, nyp, nzp;
    recovery_mode::Int32=ct_cell_b_recovery,
    include_ghosts::Bool=false,
)
    recovery_mode in (CT_CELL_B_LSQ2, CT_CELL_B_POINT6) ||
        throw(ArgumentError("unknown CT cell-B recovery mode $recovery_mode"))
    background_faces = hasproperty(b, :B0x_face) ?
        (b.B0x_face, b.B0y_face, b.B0z_face) :
        (nothing, nothing, nothing)
    B0x_face, B0y_face, B0z_face = background_faces
    background_cell = B0x_face !== nothing && hasproperty(b,:B0_cell) ?
        b.B0_cell : nothing
    if include_ghosts && recovery_mode == CT_CELL_B_POINT6
        # POINT6 needs two face layers on the low side and three on the high
        # side. Those layers do not exist outside the padded allocation, so
        # use the local six-face LSQ2 recovery for every ghost cell and then
        # overwrite the interior with POINT6. This preserves high-order
        # interior semantics without an out-of-bounds ghost stencil.
        _ct_launch_recover_cell_b!(
            b, CT_CELL_B_LSQ2,
            Int32(nxp + 2*NG), Int32(nyp + 2*NG), Int32(nzp + 2*NG),
            Int32(0), B0x_face, B0y_face, B0z_face, background_cell,
        )
        _ct_launch_recover_cell_b!(
            b, CT_CELL_B_POINT6,
            Int32(nxp), Int32(nyp), Int32(nzp), Int32(NG),
            B0x_face, B0y_face, B0z_face, background_cell,
        )
    else
        _ct_launch_recover_cell_b!(
            b, recovery_mode,
            Int32(include_ghosts ? nxp + 2*NG : nxp),
            Int32(include_ghosts ? nyp + 2*NG : nyp),
            Int32(include_ghosts ? nzp + 2*NG : nzp),
            Int32(include_ghosts ? 0 : NG),
            B0x_face, B0y_face, B0z_face, background_cell,
        )
    end
    return nothing
end

# Prescribed external-field boundaries impose zero perturbation in the
# background-split system. Deep physical ghosts can have degenerate metrics,
# so their total primitive magnetic state is restored directly from B0.
function ct_restore_prescribed_background_ghost_b_kernel!(
    Q, B0_cell, prescribed_faces, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1))*blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1))*blockDim().z + threadIdx().z
    if i > nxp+2NG || j > nyp+2NG || k > nzp+2NG
        return
    end
    prescribed =
        (prescribed_faces[1] && i <= NG) ||
        (prescribed_faces[2] && i > nxp+NG) ||
        (prescribed_faces[3] && j <= NG) ||
        (prescribed_faces[4] && j > nyp+NG) ||
        (prescribed_faces[5] && k <= NG) ||
        (prescribed_faces[6] && k > nzp+NG)
    prescribed || return

    @inbounds begin
        Q[i,j,k,QBX] = B0_cell[i,j,k,1]
        Q[i,j,k,QBY] = B0_cell[i,j,k,2]
        Q[i,j,k,QBZ] = B0_cell[i,j,k,3]
        Q[i,j,k,QPSI] = zero(FT)
    end
    return
end

function ct_restore_prescribed_background_ghost_b!(
    b, nxp, nyp, nzp;
    prescribed_faces::NTuple{6,Bool},
)
    background_cell = hasproperty(b,:B0_cell) ? b.B0_cell : nothing
    background_cell === nothing && return nothing
    any(prescribed_faces) || return nothing
    blocks = (
        cld(nxp+2NG,nthreads[1]), cld(nyp+2NG,nthreads[2]),
        cld(nzp+2NG,nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=blocks ct_restore_prescribed_background_ghost_b_kernel!(
        b.Q,background_cell,prescribed_faces,
        Int32(nxp),Int32(nyp),Int32(nzp),
    )
    return nothing
end

function _ct_launch_recover_background_cell_b!(
    b, recovery_mode::Int32,
    range_nxp::Int32, range_nyp::Int32, range_nzp::Int32,
    cell_offset::Int32,
)
    b.B0_cell === nothing && return nothing
    nb = (
        cld(range_nxp,nthreads[1]), cld(range_nyp,nthreads[2]),
        cld(range_nzp,nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=nb ct_recover_background_cell_b_kernel!(
        b.B0_cell, b.B0x_face, b.B0y_face, b.B0z_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        recovery_mode, range_nxp, range_nyp, range_nzp, cell_offset,
    )
    return nothing
end

"""Recover the fixed cell-centered `B0` cache from authoritative face fluxes."""
function ct_recover_background_cell_b!(
    b, nxp, nyp, nzp;
    recovery_mode::Int32=ct_cell_b_recovery,
    include_ghosts::Bool=true,
)
    b.B0_cell === nothing && return nothing
    recovery_mode in (CT_CELL_B_LSQ2,CT_CELL_B_POINT6) ||
        throw(ArgumentError("unknown CT background recovery mode $recovery_mode"))
    if include_ghosts && recovery_mode == CT_CELL_B_POINT6
        _ct_launch_recover_background_cell_b!(
            b,CT_CELL_B_LSQ2,
            Int32(nxp+2NG),Int32(nyp+2NG),Int32(nzp+2NG),Int32(0),
        )
        _ct_launch_recover_background_cell_b!(
            b,CT_CELL_B_POINT6,
            Int32(nxp),Int32(nyp),Int32(nzp),Int32(NG),
        )
    else
        _ct_launch_recover_background_cell_b!(
            b,recovery_mode,
            Int32(include_ghosts ? nxp+2NG : nxp),
            Int32(include_ghosts ? nyp+2NG : nyp),
            Int32(include_ghosts ? nzp+2NG : nzp),
            Int32(include_ghosts ? 0 : NG),
        )
    end
    return nothing
end

@inline function ct_sync_cell_b!(b, nxp, nyp, nzp)
    return ct_recover_cell_b!(b, nxp, nyp, nzp)
end

function ct_rebuild_isothermal_carrier_kernel!(U, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG

    @inbounds begin
        rho = U[ii,jj,kk,1]
        momentum_x = U[ii,jj,kk,2]
        momentum_y = U[ii,jj,kk,3]
        momentum_z = U[ii,jj,kk,4]
        magnetic_x = Q[ii,jj,kk,QBX]
        magnetic_y = Q[ii,jj,kk,QBY]
        magnetic_z = Q[ii,jj,kk,QBZ]
        valid = isfinite(rho) && rho > zero(FT) &&
                isfinite(momentum_x) && isfinite(momentum_y) &&
                isfinite(momentum_z) && isfinite(magnetic_x) &&
                isfinite(magnetic_y) && isfinite(magnetic_z)
        if valid
            U[ii,jj,kk,5] = mhd_isothermal_carrier_density(
                rho, momentum_x, momentum_y, momentum_z,
                magnetic_x, magnetic_y, magnetic_z,
            )
        end
    end
    return
end

function ct_rebuild_isothermal_carrier!(b, nxp, nyp, nzp)
    nb = (
        cld(nxp, nthreads[1]), cld(nyp, nthreads[2]),
        cld(nzp, nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=nb ct_rebuild_isothermal_carrier_kernel!(
        b.U, b.Q, Int32(nxp), Int32(nyp), Int32(nzp))
    return nothing
end

# The initial face-B bootstrap can replace the cell-centered magnetic field
# with a metric-consistent reconstruction.  For adiabatic MHD, close U[5]
# once at initialization so the prescribed initial pressure is retained.  This
# is deliberately separate from RK-stage CT updates, where U[5] is conserved.
function ct_reconcile_initial_energy_kernel!(U, Q, nxp, nyp, nzp, gamma)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp
        return
    end
    ii = i + NG
    jj = j + NG
    kk = k + NG

    @inbounds begin
        rho = U[ii, jj, kk, 1]
        momentum_x = U[ii, jj, kk, 2]
        momentum_y = U[ii, jj, kk, 3]
        momentum_z = U[ii, jj, kk, 4]
        pressure = Q[ii, jj, kk, 5]
        magnetic_x = Q[ii, jj, kk, QBX]
        magnetic_y = Q[ii, jj, kk, QBY]
        magnetic_z = Q[ii, jj, kk, QBZ]
        valid = isfinite(rho) && rho > zero(FT) &&
                isfinite(momentum_x) && isfinite(momentum_y) &&
                isfinite(momentum_z) && isfinite(pressure) &&
                pressure > zero(FT) && isfinite(magnetic_x) &&
                isfinite(magnetic_y) && isfinite(magnetic_z)
        if valid
            inverse_rho = one(FT) / rho
            U[ii, jj, kk, 5] = mhd_energy_density_from_primitive(
                rho,
                momentum_x * inverse_rho,
                momentum_y * inverse_rho,
                momentum_z * inverse_rho,
                pressure,
                magnetic_x, magnetic_y, magnetic_z, gamma,
            )
        end
    end
    return
end

function ct_reconcile_initial_energy!(b, nxp, nyp, nzp, gamma)
    nb = (
        cld(nxp, nthreads[1]),
        cld(nyp, nthreads[2]),
        cld(nzp, nthreads[3]),
    )
    @gpu_launch threads=nthreads blocks=nb ct_reconcile_initial_energy_kernel!(
        b.U, b.Q, Int32(nxp), Int32(nyp), Int32(nzp), FT(gamma),
    )
    return nothing
end

# ─── CT: Copy face B to backup (for RK) ───
function ct_backup_face_b_kernel!(Bn, B, n1, n2, n3)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > n1 || j > n2 || k > n3; return; end
    @inbounds Bn[i, j, k] = B[i, j, k]
    return
end

function ct_backup_face_b!(b, nxp, nyp, nzp)
    # Each face-B array has different dimensions; launch separately.
    n1x = nxp + 1 + 2*NG; n2x = nyp + 2*NG + 1; n3x = nzp + 2*NG + 1
    nb_x = (cld(n1x, nthreads[1]), cld(n2x, nthreads[2]), cld(n3x, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_x ct_backup_face_b_kernel!(
        b.Bx_face_n, b.Bx_face, Int32(n1x), Int32(n2x), Int32(n3x))
    n1y = nxp + 2*NG + 1; n2y = nyp + 1 + 2*NG; n3y = nzp + 2*NG + 1
    nb_y = (cld(n1y, nthreads[1]), cld(n2y, nthreads[2]), cld(n3y, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_y ct_backup_face_b_kernel!(
        b.By_face_n, b.By_face, Int32(n1y), Int32(n2y), Int32(n3y))
    n1z = nxp + 2*NG + 1; n2z = nyp + 2*NG + 1; n3z = nzp + 1 + 2*NG
    nb_z = (cld(n1z, nthreads[1]), cld(n2z, nthreads[2]), cld(n3z, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_z ct_backup_face_b_kernel!(
        b.Bz_face_n, b.Bz_face, Int32(n1z), Int32(n2z), Int32(n3z))
end

# ─── CT: Recover the point primitive state from stage-consistent hydro and B ───
# The stage finalizer has already paired U[1:5] with the newly synchronized
# cell-centered magnetic field. This kernel applies the selected point-state
# recovery and writes the complete primitive state used by reconstruction.
function ct_update_q_b_kernel!(Q, U, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG

    @inbounds state = SVector{9,FT}(
        (
            U[ii, jj, kk, 1], U[ii, jj, kk, 2], U[ii, jj, kk, 3],
            U[ii, jj, kk, 4], U[ii, jj, kk, 5], Q[ii, jj, kk, QBX],
            Q[ii, jj, kk, QBY], Q[ii, jj, kk, QBZ], zero(FT),
        ),
    )
    ρ, kinetic, magnetic, ei, p_v = mhd_raw_thermo(state, γ)
    ρinv = one(FT) / ρ
    u_v = state[2] * ρinv
    v_v = state[3] * ρinv
    w_v = state[4] * ρinv
    T_v = p_v / (ρ * Rg)
    @inbounds begin
        Q[ii,jj,kk,1]=ρ; Q[ii,jj,kk,2]=u_v
        Q[ii,jj,kk,3]=v_v; Q[ii,jj,kk,4]=w_v
        Q[ii,jj,kk,5]=p_v; Q[ii,jj,kk,6]=T_v
        Q[ii,jj,kk,QBX]=state[6]; Q[ii,jj,kk,QBY]=state[7]
        Q[ii,jj,kk,QBZ]=state[8]; Q[ii,jj,kk,QPSI]=zero(FT)
    end
    return
end

function ct_update_q_point6_kernel!(
    Q, U, inverse_volume,
    face_fluxes, background_fluxes, face_metrics, background_cell,
    positivity_meta, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
    candidate_hydro, x_ao, y_ao, z_ao, used_ao =
        ct_conservative_average_to_point_adaptive(
            U,inverse_volume,ii,jj,kk,
        )
    ao_hydro = used_ao ? candidate_hydro :
        ct_conservative_average_to_point_coefficients(
            U,inverse_volume,ii,jj,kk,x_ao,y_ao,z_ao,
        )
    low_hydro = _ct_point6_cell_conservative(U,ii,jj,kk)
    @inbounds high_magnetic = SVector{3,FT}(
        Q[ii, jj, kk, QBX],
        Q[ii, jj, kk, QBY],
        Q[ii, jj, kk, QBZ],
    )
    low_magnetic = _ct_recover_cell_b_from_face_fluxes(
        face_fluxes[1],face_fluxes[2],face_fluxes[3],
        face_metrics[1]...,face_metrics[2]...,face_metrics[3]...,
        CT_CELL_B_LSQ2,ii,jj,kk,background_fluxes...,
    )
    hydro, magnetic, theta, recovery_mode =
        ct_point6_select_admissible_state(
            candidate_hydro,ao_hydro,low_hydro,high_magnetic,low_magnetic,
            FT(γ),FT(density_floor),FT(pressure_floor),used_ao,
        )
    _ct_record_point6_recovery!(positivity_meta,recovery_mode)
    primitive = ct_mhd_point_conservative_to_primitive(
        hydro, magnetic, FT(γ),
    )
    if background_cell !== nothing
        high_background = _ct_recover_cell_b_from_face_fluxes(
            background_fluxes[1],background_fluxes[2],background_fluxes[3],
            face_metrics[1]...,face_metrics[2]...,face_metrics[3]...,
            CT_CELL_B_POINT6,ii,jj,kk,
        )
        background = high_background
        if theta < one(theta)
            low_background = _ct_recover_cell_b_from_face_fluxes(
                background_fluxes[1],background_fluxes[2],background_fluxes[3],
                face_metrics[1]...,face_metrics[2]...,face_metrics[3]...,
                CT_CELL_B_LSQ2,ii,jj,kk,
            )
            background = _ct_point6_blend(
                low_background,high_background,theta,
            )
        end
        @inbounds begin
            background_cell[ii,jj,kk,1]=background[1]
            background_cell[ii,jj,kk,2]=background[2]
            background_cell[ii,jj,kk,3]=background[3]
        end
    end
    temperature = primitive[5] / (primitive[1] * FT(Rg))
    @inbounds begin
        Q[ii, jj, kk, 1] = primitive[1]
        Q[ii, jj, kk, 2] = primitive[2]
        Q[ii, jj, kk, 3] = primitive[3]
        Q[ii, jj, kk, 4] = primitive[4]
        Q[ii, jj, kk, 5] = primitive[5]
        Q[ii, jj, kk, 6] = temperature
        Q[ii, jj, kk, QBX] = primitive[6]
        Q[ii, jj, kk, QBY] = primitive[7]
        Q[ii, jj, kk, QBZ] = primitive[8]
        Q[ii, jj, kk, QPSI] = zero(FT)
    end
    return
end

function ct_update_q_b!(b, nxp, nyp, nzp; positivity_meta=nothing)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @static if isdefined(@__MODULE__, :ct_primitive_recovery) &&
               ct_primitive_recovery == CT_PRIMITIVE_POINT6
        face_fluxes = (b.Bx_face,b.By_face,b.Bz_face)
        background_fluxes = hasproperty(b,:B0x_face) ?
            (b.B0x_face,b.B0y_face,b.B0z_face) : (nothing,nothing,nothing)
        face_metrics = (
            (b.Areai,b.nxi,b.nyi,b.nzi),
            (b.Areaj,b.nxj,b.nyj,b.nzj),
            (b.Areak,b.nxk,b.nyk,b.nzk),
        )
        background_cell = hasproperty(b,:B0_cell) ? b.B0_cell : nothing
        @gpu_launch threads=nthreads blocks=nb ct_update_q_point6_kernel!(
            b.Q,b.U,b.Vol,face_fluxes,background_fluxes,face_metrics,
            background_cell,positivity_meta,
            Int32(nxp),Int32(nyp),Int32(nzp))
    else
        @gpu_launch threads=nthreads blocks=nb ct_update_q_b_kernel!(
            b.Q, b.U, Int32(nxp), Int32(nyp), Int32(nzp))
    end
end

# ─── CT: Periodic ghost fill for face B ───
# Each face-B array has its normal direction in a different position:
#   Bx_face(i,j,k): normal=i (1st dim), tangential=j,k
#   By_face(i,j,k): normal=j (2nd dim), tangential=i,k
#   Bz_face(i,j,k): normal=k (3rd dim), tangential=i,j
# We use a single kernel with a direction flag (1=x-normal, 2=y-normal, 3=z-normal).
function ct_periodic_face_b_kernel!(
    Bface, dir, periodic_x, periodic_y, periodic_z, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    # Array dimensions (with +1 tangential ghost):
    # Bx: (nxp+1+2NG, nyp+2NG+1, nzp+2NG+1)
    # By: (nxp+2NG+1, nyp+1+2NG, nzp+2NG+1)
    # Bz: (nxp+2NG+1, nyp+2NG+1, nzp+1+2NG)
    n1 = nxp + Int32(1) + Int32(2)*NG  # dim 1 for Bx (normal), else nxp+2NG+1
    n2 = nyp + Int32(1) + Int32(2)*NG
    n3 = nzp + Int32(1) + Int32(2)*NG
    if i > n1 || j > n2 || k > n3; return; end
    ng = Int32(NG)
    source_i = periodic_x ? ct_periodic_face_source_index(
        i, nxp, ng, dir == Int32(1),
    ) : i
    source_j = periodic_y ? ct_periodic_face_source_index(
        j, nyp, ng, dir == Int32(2),
    ) : j
    source_k = periodic_z ? ct_periodic_face_source_index(
        k, nzp, ng, dir == Int32(3),
    ) : k
    if source_i != i || source_j != j || source_k != k
        # All three source coordinates are mapped directly into the physical
        # range. This avoids in-place corner races through partially filled
        # ghost values.
        @inbounds Bface[i, j, k] = Bface[source_i, source_j, source_k]
    end
    return
end

function ct_fill_periodic_face_b!(b, nxp, nyp, nzp, Iperiodic)
    if !any(Iperiodic); return; end
    n1 = nxp + 1 + 2*NG; n2 = nyp + 2*NG + 1; n3 = nzp + 2*NG + 1
    nb = (cld(n1, nthreads[1]), cld(n2, nthreads[2]), cld(n3, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb ct_periodic_face_b_kernel!(
        b.Bx_face, Int32(1), Iperiodic[1], Iperiodic[2], Iperiodic[3],
        Int32(nxp), Int32(nyp), Int32(nzp))
    n1b = nxp + 2*NG + 1; n2b = nyp + 1 + 2*NG; n3b = nzp + 2*NG + 1
    nb_b = (cld(n1b, nthreads[1]), cld(n2b, nthreads[2]), cld(n3b, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_b ct_periodic_face_b_kernel!(
        b.By_face, Int32(2), Iperiodic[1], Iperiodic[2], Iperiodic[3],
        Int32(nxp), Int32(nyp), Int32(nzp))
    n1c = nxp + 2*NG + 1; n2c = nyp + 2*NG + 1; n3c = nzp + 1 + 2*NG
    nb_c = (cld(n1c, nthreads[1]), cld(n2c, nthreads[2]), cld(n3c, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_c ct_periodic_face_b_kernel!(
        b.Bz_face, Int32(3), Iperiodic[1], Iperiodic[2], Iperiodic[3],
        Int32(nxp), Int32(nyp), Int32(nzp))
end

# ─── CT: Zero-gradient ghost fill for face B ───
# Recover the boundary-adjacent Cartesian field directly from staggered face
# fluxes. Q[B] is a derived cache and must never be an input to a face-B BC.
@inline function ct_recover_cell_b_lsq2_at(
    Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    i, j, k,
    B0x_face=nothing, B0y_face=nothing, B0z_face=nothing,
)
    @inbounds begin
        area_i_lo = SVector(
            Areai[i,j,k]*nxi[i,j,k],
            Areai[i,j,k]*nyi[i,j,k],
            Areai[i,j,k]*nzi[i,j,k],
        )
        area_i_hi = SVector(
            Areai[i+Int32(1),j,k]*nxi[i+Int32(1),j,k],
            Areai[i+Int32(1),j,k]*nyi[i+Int32(1),j,k],
            Areai[i+Int32(1),j,k]*nzi[i+Int32(1),j,k],
        )
        area_j_lo = SVector(
            Areaj[i,j,k]*nxj[i,j,k],
            Areaj[i,j,k]*nyj[i,j,k],
            Areaj[i,j,k]*nzj[i,j,k],
        )
        area_j_hi = SVector(
            Areaj[i,j+Int32(1),k]*nxj[i,j+Int32(1),k],
            Areaj[i,j+Int32(1),k]*nyj[i,j+Int32(1),k],
            Areaj[i,j+Int32(1),k]*nzj[i,j+Int32(1),k],
        )
        area_k_lo = SVector(
            Areak[i,j,k]*nxk[i,j,k],
            Areak[i,j,k]*nyk[i,j,k],
            Areak[i,j,k]*nzk[i,j,k],
        )
        area_k_hi = SVector(
            Areak[i,j,k+Int32(1)]*nxk[i,j,k+Int32(1)],
            Areak[i,j,k+Int32(1)]*nyk[i,j,k+Int32(1)],
            Areak[i,j,k+Int32(1)]*nzk[i,j,k+Int32(1)],
        )
        return ct_recover_cell_b(
            area_i_lo, area_i_hi, area_j_lo, area_j_hi,
            area_k_lo, area_k_hi,
            _ct_total_face_flux(Bx_face, B0x_face, i, j, k),
            _ct_total_face_flux(Bx_face, B0x_face, i+Int32(1), j, k),
            _ct_total_face_flux(By_face, B0y_face, i, j, k),
            _ct_total_face_flux(By_face, B0y_face, i, j+Int32(1), k),
            _ct_total_face_flux(Bz_face, B0z_face, i, j, k),
            _ct_total_face_flux(Bz_face, B0z_face, i, j, k+Int32(1)),
        )
    end
end

# Same direction-aware structure as periodic fill.
function ct_zerograd_face_b_kernel!(
    Bface, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    Area, normal_x, normal_y, normal_z,
    dir, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    n1 = nxp + Int32(1) + Int32(2)*NG
    n2 = nyp + Int32(1) + Int32(2)*NG
    n3 = nzp + Int32(1) + Int32(2)*NG
    if i > n1 || j > n2 || k > n3; return; end
    ng = Int32(NG)
    source_i = ct_zerograd_face_source_index(
        i, nxp, ng, dir == Int32(1),
    )
    source_j = ct_zerograd_face_source_index(
        j, nyp, ng, dir == Int32(2),
    )
    source_k = ct_zerograd_face_source_index(
        k, nzp, ng, dir == Int32(3),
    )
    if source_i != i || source_j != j || source_k != k
        metric_n1 = nxp + Int32(2)*ng + (dir == Int32(1) ? Int32(1) : Int32(0))
        metric_n2 = nyp + Int32(2)*ng + (dir == Int32(2) ? Int32(1) : Int32(0))
        metric_n3 = nzp + Int32(2)*ng + (dir == Int32(3) ? Int32(1) : Int32(0))
        if i <= metric_n1 && j <= metric_n2 && k <= metric_n3
            if dir == Int32(1)
                left_i = max(source_i - Int32(1), ng + Int32(1))
                right_i = min(source_i, ng + nxp)
                @inbounds begin
                    left_b = ct_recover_cell_b_lsq2_at(
                        Bx_face, By_face, Bz_face,
                        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
                        Areak, nxk, nyk, nzk,
                        left_i, source_j, source_k,
                    )
                    right_b = ct_recover_cell_b_lsq2_at(
                        Bx_face, By_face, Bz_face,
                        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
                        Areak, nxk, nyk, nzk,
                        right_i, source_j, source_k,
                    )
                    bx, by, bz = FT(0.5) * (left_b + right_b)
                    bx, by, bz = ct_constrain_face_flux(
                        bx, by, bz,
                        Area[source_i, source_j, source_k],
                        normal_x[source_i, source_j, source_k],
                        normal_y[source_i, source_j, source_k],
                        normal_z[source_i, source_j, source_k],
                        Bface[source_i, source_j, source_k],
                    )
                    Bface[i, j, k] = Area[i, j, k] * (
                        bx*normal_x[i, j, k] + by*normal_y[i, j, k] +
                        bz*normal_z[i, j, k]
                    )
                end
            elseif dir == Int32(2)
                left_j = max(source_j - Int32(1), ng + Int32(1))
                right_j = min(source_j, ng + nyp)
                @inbounds begin
                    left_b = ct_recover_cell_b_lsq2_at(
                        Bx_face, By_face, Bz_face,
                        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
                        Areak, nxk, nyk, nzk,
                        source_i, left_j, source_k,
                    )
                    right_b = ct_recover_cell_b_lsq2_at(
                        Bx_face, By_face, Bz_face,
                        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
                        Areak, nxk, nyk, nzk,
                        source_i, right_j, source_k,
                    )
                    bx, by, bz = FT(0.5) * (left_b + right_b)
                    bx, by, bz = ct_constrain_face_flux(
                        bx, by, bz,
                        Area[source_i, source_j, source_k],
                        normal_x[source_i, source_j, source_k],
                        normal_y[source_i, source_j, source_k],
                        normal_z[source_i, source_j, source_k],
                        Bface[source_i, source_j, source_k],
                    )
                    Bface[i, j, k] = Area[i, j, k] * (
                        bx*normal_x[i, j, k] + by*normal_y[i, j, k] +
                        bz*normal_z[i, j, k]
                    )
                end
            else
                left_k = max(source_k - Int32(1), ng + Int32(1))
                right_k = min(source_k, ng + nzp)
                @inbounds begin
                    left_b = ct_recover_cell_b_lsq2_at(
                        Bx_face, By_face, Bz_face,
                        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
                        Areak, nxk, nyk, nzk,
                        source_i, source_j, left_k,
                    )
                    right_b = ct_recover_cell_b_lsq2_at(
                        Bx_face, By_face, Bz_face,
                        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
                        Areak, nxk, nyk, nzk,
                        source_i, source_j, right_k,
                    )
                    bx, by, bz = FT(0.5) * (left_b + right_b)
                    bx, by, bz = ct_constrain_face_flux(
                        bx, by, bz,
                        Area[source_i, source_j, source_k],
                        normal_x[source_i, source_j, source_k],
                        normal_y[source_i, source_j, source_k],
                        normal_z[source_i, source_j, source_k],
                        Bface[source_i, source_j, source_k],
                    )
                    Bface[i, j, k] = Area[i, j, k] * (
                        bx*normal_x[i, j, k] + by*normal_y[i, j, k] +
                        bz*normal_z[i, j, k]
                    )
                end
            end
        else
            # The extra tangential reconstruction slot has no face metric.
            @inbounds Bface[i, j, k] = Bface[source_i, source_j, source_k]
        end
    end
    return
end

function ct_fill_zerograd_face_b!(b, nxp, nyp, nzp)
    n1x = nxp + 1 + 2*NG; n2x = nyp + 2*NG + 1; n3x = nzp + 2*NG + 1
    nb_x = (cld(n1x, nthreads[1]), cld(n2x, nthreads[2]), cld(n3x, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_x ct_zerograd_face_b_kernel!(
        b.Bx_face, b.Bx_face, b.By_face, b.Bz_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        b.Areai, b.nxi, b.nyi, b.nzi,
        Int32(1), Int32(nxp), Int32(nyp), Int32(nzp))
    n1y = nxp + 2*NG + 1; n2y = nyp + 1 + 2*NG; n3y = nzp + 2*NG + 1
    nb_y = (cld(n1y, nthreads[1]), cld(n2y, nthreads[2]), cld(n3y, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_y ct_zerograd_face_b_kernel!(
        b.By_face, b.Bx_face, b.By_face, b.Bz_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        Int32(2), Int32(nxp), Int32(nyp), Int32(nzp))
    n1z = nxp + 2*NG + 1; n2z = nyp + 2*NG + 1; n3z = nzp + 1 + 2*NG
    nb_z = (cld(n1z, nthreads[1]), cld(n2z, nthreads[2]), cld(n3z, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_z ct_zerograd_face_b_kernel!(
        b.Bz_face, b.Bx_face, b.By_face, b.Bz_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        b.Areak, b.nxk, b.nyk, b.nzk,
        Int32(3), Int32(nxp), Int32(nyp), Int32(nzp))
end

# An insulating wall uses an odd extension for tangential B and an even
# extension for normal B. In CT this must be imposed on the authoritative
# oriented face fluxes, not on the derived cell-centered Q[B] cache.
@inline function ct_insulating_wall_face_active(
    index, ncell, ng, boundary_axis, side, face_kind, include_normal_face,
)
    lower = side == Int32(0)
    if boundary_axis == face_kind
        wall_face = lower ? ng + Int32(1) : ng + ncell + Int32(1)
        if include_normal_face
            return lower ? index <= wall_face : index >= wall_face
        end
        return lower ? index < wall_face : index > wall_face
    end
    return lower ? index <= ng : index > ng + ncell
end

@inline function ct_insulating_wall_source_index(
    index, ncell, ng, boundary_axis, side, face_kind,
)
    lower = side == Int32(0)
    if boundary_axis == face_kind
        wall_face = lower ? ng + Int32(1) : ng + ncell + Int32(1)
        if index == wall_face
            return lower ? wall_face + Int32(1) : wall_face - Int32(1)
        end
        return Int32(2)*wall_face - index
    end
    mirror_sum = lower ? Int32(2)*ng + Int32(1) :
                         Int32(2)*(ng + ncell) + Int32(1)
    return mirror_sum - index
end

@inline function ct_recover_face_b_lsq2_at(
    Bface, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    Area, normal_x, normal_y, normal_z,
    face_kind, i, j, k, nxp, nyp, nzp,
    B0x_face=nothing, B0y_face=nothing, B0z_face=nothing,
)
    ng = Int32(NG)
    cell_n1 = nxp + Int32(2)*ng
    cell_n2 = nyp + Int32(2)*ng
    cell_n3 = nzp + Int32(2)*ng
    if face_kind == Int32(1)
        left_b = ct_recover_cell_b_lsq2_at(
            Bx_face, By_face, Bz_face,
            Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
            Areak, nxk, nyk, nzk,
            max(i - Int32(1), Int32(1)), j, k,
            B0x_face, B0y_face, B0z_face,
        )
        right_b = ct_recover_cell_b_lsq2_at(
            Bx_face, By_face, Bz_face,
            Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
            Areak, nxk, nyk, nzk,
            min(i, cell_n1), j, k,
            B0x_face, B0y_face, B0z_face,
        )
    elseif face_kind == Int32(2)
        left_b = ct_recover_cell_b_lsq2_at(
            Bx_face, By_face, Bz_face,
            Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
            Areak, nxk, nyk, nzk,
            i, max(j - Int32(1), Int32(1)), k,
            B0x_face, B0y_face, B0z_face,
        )
        right_b = ct_recover_cell_b_lsq2_at(
            Bx_face, By_face, Bz_face,
            Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
            Areak, nxk, nyk, nzk,
            i, min(j, cell_n2), k,
            B0x_face, B0y_face, B0z_face,
        )
    else
        left_b = ct_recover_cell_b_lsq2_at(
            Bx_face, By_face, Bz_face,
            Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
            Areak, nxk, nyk, nzk,
            i, j, max(k - Int32(1), Int32(1)),
            B0x_face, B0y_face, B0z_face,
        )
        right_b = ct_recover_cell_b_lsq2_at(
            Bx_face, By_face, Bz_face,
            Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
            Areak, nxk, nyk, nzk,
            i, j, min(k, cell_n3),
            B0x_face, B0y_face, B0z_face,
        )
    end
    magnetic = FT(0.5) * (left_b + right_b)
    return ct_constrain_face_flux(
        magnetic[1], magnetic[2], magnetic[3],
        Area[i,j,k], normal_x[i,j,k], normal_y[i,j,k], normal_z[i,j,k],
        _ct_total_face_flux(
            Bface,
            face_kind == Int32(1) ? B0x_face :
            (face_kind == Int32(2) ? B0y_face : B0z_face),
            i, j, k,
        ),
    )
end

function ct_fill_insulating_wall_face_b_kernel!(
    Bface, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    Area, normal_x, normal_y, normal_z,
    wall_normal_x, wall_normal_y, wall_normal_z,
    boundary_axis, side, face_kind, nxp, nyp, nzp, include_normal_face,
    B0x_face=nothing, B0y_face=nothing, B0z_face=nothing,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)
    metric_n1 = nxp + Int32(2)*ng +
                (face_kind == Int32(1) ? Int32(1) : Int32(0))
    metric_n2 = nyp + Int32(2)*ng +
                (face_kind == Int32(2) ? Int32(1) : Int32(0))
    metric_n3 = nzp + Int32(2)*ng +
                (face_kind == Int32(3) ? Int32(1) : Int32(0))
    if i > metric_n1 || j > metric_n2 || k > metric_n3
        return
    end

    boundary_index = boundary_axis == Int32(1) ? i :
                     (boundary_axis == Int32(2) ? j : k)
    boundary_cells = boundary_axis == Int32(1) ? nxp :
                     (boundary_axis == Int32(2) ? nyp : nzp)
    ct_insulating_wall_face_active(
        boundary_index, boundary_cells, ng,
        boundary_axis, side, face_kind, include_normal_face,
    ) || return

    source_boundary_index = ct_insulating_wall_source_index(
        boundary_index, boundary_cells, ng,
        boundary_axis, side, face_kind,
    )
    source_i = boundary_axis == Int32(1) ? source_boundary_index : i
    source_j = boundary_axis == Int32(2) ? source_boundary_index : j
    source_k = boundary_axis == Int32(3) ? source_boundary_index : k
    magnetic = ct_recover_face_b_lsq2_at(
        Bface, Bx_face, By_face, Bz_face,
        Areai, nxi, nyi, nzi, Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        Area, normal_x, normal_y, normal_z,
        face_kind, source_i, source_j, source_k, nxp, nyp, nzp,
        B0x_face, B0y_face, B0z_face,
    )

    wall_face = side == Int32(0) ? ng + Int32(1) :
                                   ng + boundary_cells + Int32(1)
    wall_n1 = nxp + Int32(2)*ng +
              (boundary_axis == Int32(1) ? Int32(1) : Int32(0))
    wall_n2 = nyp + Int32(2)*ng +
              (boundary_axis == Int32(2) ? Int32(1) : Int32(0))
    wall_n3 = nzp + Int32(2)*ng +
              (boundary_axis == Int32(3) ? Int32(1) : Int32(0))
    wall_i = boundary_axis == Int32(1) ? wall_face :
             min(max(source_i, Int32(1)), wall_n1)
    wall_j = boundary_axis == Int32(2) ? wall_face :
             min(max(source_j, Int32(1)), wall_n2)
    wall_k = boundary_axis == Int32(3) ? wall_face :
             min(max(source_k, Int32(1)), wall_n3)

    @inbounds begin
        nx_wall = wall_normal_x[wall_i,wall_j,wall_k]
        ny_wall = wall_normal_y[wall_i,wall_j,wall_k]
        nz_wall = wall_normal_z[wall_i,wall_j,wall_k]
        normal_magnetic = magnetic[1]*nx_wall + magnetic[2]*ny_wall +
                          magnetic[3]*nz_wall
        reflected_x = FT(2)*normal_magnetic*nx_wall - magnetic[1]
        reflected_y = FT(2)*normal_magnetic*ny_wall - magnetic[2]
        reflected_z = FT(2)*normal_magnetic*nz_wall - magnetic[3]
        reflected_flux = Area[i,j,k] * (
            reflected_x*normal_x[i,j,k] +
            reflected_y*normal_y[i,j,k] +
            reflected_z*normal_z[i,j,k]
        )
        target_background = face_kind == Int32(1) ? B0x_face :
            (face_kind == Int32(2) ? B0y_face : B0z_face)
        Bface[i,j,k] = reflected_flux - _ct_background_face_flux(
            target_background, i, j, k,
        )
    end
    return
end

function ct_fill_insulating_wall_face_b!(
    b, nxp, nyp, nzp, boundary_axis::Integer, side::Integer;
    include_normal_face::Bool=false,
)
    axis = Int32(boundary_axis)
    side_value = Int32(side)
    nx = Int32(nxp); ny = Int32(nyp); nz = Int32(nzp)
    wall_normals = axis == Int32(1) ? (b.nxi, b.nyi, b.nzi) :
                   (axis == Int32(2) ? (b.nxj, b.nyj, b.nzj) :
                    (b.nxk, b.nyk, b.nzk))
    background_faces = hasproperty(b, :B0x_face) ?
        (b.B0x_face, b.B0y_face, b.B0z_face) :
        (nothing, nothing, nothing)
    B0x_face, B0y_face, B0z_face = background_faces

    n1x = nx + Int32(2)*NG + Int32(1)
    n2x = ny + Int32(2)*NG
    n3x = nz + Int32(2)*NG
    @gpu_launch threads=nthreads blocks=(
        cld(n1x, nthreads[1]), cld(n2x, nthreads[2]), cld(n3x, nthreads[3]),
    ) ct_fill_insulating_wall_face_b_kernel!(
        b.Bx_face, b.Bx_face, b.By_face, b.Bz_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        b.Areai, b.nxi, b.nyi, b.nzi,
        wall_normals[1], wall_normals[2], wall_normals[3],
        axis, side_value, Int32(1), nx, ny, nz, include_normal_face,
        B0x_face, B0y_face, B0z_face,
    )

    n1y = nx + Int32(2)*NG
    n2y = ny + Int32(2)*NG + Int32(1)
    n3y = nz + Int32(2)*NG
    @gpu_launch threads=nthreads blocks=(
        cld(n1y, nthreads[1]), cld(n2y, nthreads[2]), cld(n3y, nthreads[3]),
    ) ct_fill_insulating_wall_face_b_kernel!(
        b.By_face, b.Bx_face, b.By_face, b.Bz_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        wall_normals[1], wall_normals[2], wall_normals[3],
        axis, side_value, Int32(2), nx, ny, nz, include_normal_face,
        B0x_face, B0y_face, B0z_face,
    )

    n1z = nx + Int32(2)*NG
    n2z = ny + Int32(2)*NG
    n3z = nz + Int32(2)*NG + Int32(1)
    @gpu_launch threads=nthreads blocks=(
        cld(n1z, nthreads[1]), cld(n2z, nthreads[2]), cld(n3z, nthreads[3]),
    ) ct_fill_insulating_wall_face_b_kernel!(
        b.Bz_face, b.Bx_face, b.By_face, b.Bz_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        b.Areak, b.nxk, b.nyk, b.nzk,
        wall_normals[1], wall_normals[2], wall_normals[3],
        axis, side_value, Int32(3), nx, ny, nz, include_normal_face,
        B0x_face, B0y_face, B0z_face,
    )
    return nothing
end

# Fill the physical boundary sheet for a prescribed analytic external field.
# The external-field kernel writes oriented face fluxes and deliberately runs
# after rank/interface synchronization so a physical BC cannot be overwritten
# by a halo route from a neighboring block.
function ct_fill_external_field_face_b!(
    b, nxp, nyp, nzp, boundary_axis::Integer, side::Integer, bcp,
    ; include_normal_face::Bool=true,
      background_splitting::Bool=false,
      preserve_perturbation::Bool=false,
)
    axis = Int32(boundary_axis)
    side_value = Int32(side)
    nx = Int32(nxp); ny = Int32(nyp); nz = Int32(nzp)
    background_faces = hasproperty(b, :B0x_face) ?
        (b.B0x_face, b.B0y_face, b.B0z_face) :
        (nothing, nothing, nothing)
    B0x_face, B0y_face, B0z_face = background_faces

    n1x = nx + Int32(2)*NG + Int32(1)
    n2x = ny + Int32(2)*NG
    n3x = nz + Int32(2)*NG
    @gpu_launch threads=nthreads blocks=(
        cld(n1x, nthreads[1]), cld(n2x, nthreads[2]), cld(n3x, nthreads[3]),
    ) ct_fill_external_field_face_b_kernel!(
        b.Bx_face, b.Areai, b.nxi, b.nyi, b.nzi, b.x, b.y, b.z,
        axis, side_value, Int32(1), nx, ny, nz, bcp,
        include_normal_face, background_splitting, preserve_perturbation,
        B0x_face,
    )

    n1y = nx + Int32(2)*NG
    n2y = ny + Int32(2)*NG + Int32(1)
    n3y = nz + Int32(2)*NG
    @gpu_launch threads=nthreads blocks=(
        cld(n1y, nthreads[1]), cld(n2y, nthreads[2]), cld(n3y, nthreads[3]),
    ) ct_fill_external_field_face_b_kernel!(
        b.By_face, b.Areaj, b.nxj, b.nyj, b.nzj, b.x, b.y, b.z,
        axis, side_value, Int32(2), nx, ny, nz, bcp,
        include_normal_face, background_splitting, preserve_perturbation,
        B0y_face,
    )

    n1z = nx + Int32(2)*NG
    n2z = ny + Int32(2)*NG
    n3z = nz + Int32(2)*NG + Int32(1)
    @gpu_launch threads=nthreads blocks=(
        cld(n1z, nthreads[1]), cld(n2z, nthreads[2]), cld(n3z, nthreads[3]),
    ) ct_fill_external_field_face_b_kernel!(
        b.Bz_face, b.Areak, b.nxk, b.nyk, b.nzk, b.x, b.y, b.z,
        axis, side_value, Int32(3), nx, ny, nz, bcp,
        include_normal_face, background_splitting, preserve_perturbation,
        B0z_face,
    )
    return nothing
end
