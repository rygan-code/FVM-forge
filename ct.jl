# ═══════════════════════════════════════════════════════════════════════
# ct.jl - Constrained Transport for MHD divergence cleaning
# ═══════════════════════════════════════════════════════════════════════
# Metric-aware CT state:
#   face arrays: oriented magnetic flux Phi_B = (B dot n) A
#   edge arrays: oriented line EMF = integral(E dot dl)
#   update: discrete Stokes theorem, so div(curl)=0 topologically
# ═══════════════════════════════════════════════════════════════════════

# ─── CT: Initialize oriented face magnetic flux from cell B ───
function ct_init_face_b_kernel!(
    Bx_face, By_face, Bz_face, U,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    nxp, nyp, nzp,
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
                bx = FT(0.5) * (U[ii, jj, kk, UBX] + U[ii-Int32(1), jj, kk, UBX])
                by = FT(0.5) * (U[ii, jj, kk, UBY] + U[ii-Int32(1), jj, kk, UBY])
                bz = FT(0.5) * (U[ii, jj, kk, UBZ] + U[ii-Int32(1), jj, kk, UBZ])
                Bx_face[ii, jj, kk] = Areai[ii, jj, kk] *
                    (bx*nxi[ii, jj, kk] + by*nyi[ii, jj, kk] + bz*nzi[ii, jj, kk])
            end
        end
        if i <= nxp && j <= nyp + Int32(1) && k <= nzp
            @inbounds begin
                bx = FT(0.5) * (U[ii, jj, kk, UBX] + U[ii, jj-Int32(1), kk, UBX])
                by = FT(0.5) * (U[ii, jj, kk, UBY] + U[ii, jj-Int32(1), kk, UBY])
                bz = FT(0.5) * (U[ii, jj, kk, UBZ] + U[ii, jj-Int32(1), kk, UBZ])
                By_face[ii, jj, kk] = Areaj[ii, jj, kk] *
                    (bx*nxj[ii, jj, kk] + by*nyj[ii, jj, kk] + bz*nzj[ii, jj, kk])
            end
        end
        if i <= nxp && j <= nyp && k <= nzp + Int32(1)
            @inbounds begin
                bx = FT(0.5) * (U[ii, jj, kk, UBX] + U[ii, jj, kk-Int32(1), UBX])
                by = FT(0.5) * (U[ii, jj, kk, UBY] + U[ii, jj, kk-Int32(1), UBY])
                bz = FT(0.5) * (U[ii, jj, kk, UBZ] + U[ii, jj, kk-Int32(1), UBZ])
                Bz_face[ii, jj, kk] = Areak[ii, jj, kk] *
                    (bx*nxk[ii, jj, kk] + by*nyk[ii, jj, kk] + bz*nzk[ii, jj, kk])
            end
        end
    end
    return
end

function ct_init_face_b!(b, nxp, nyp, nzp)
    # Launch over the union of all face ranges: (nxp+1) x (nyp+1) x (nzp+1)
    nb = (cld(nxp + 1 + 2*NG, nthreads[1]), cld(nyp + 1 + 2*NG, nthreads[2]), cld(nzp + 1 + 2*NG, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb ct_init_face_b_kernel!(
        b.Bx_face, b.By_face, b.Bz_face, b.U,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        Int32(nxp), Int32(nyp), Int32(nzp))
end

function ct_initial_face_flux_from_vector_potential!(
    b, vector_potential; time=zero(FT),
)
    x = Array(b.x)
    y = Array(b.y)
    z = Array(b.z)
    size(x) == size(y) == size(z) || throw(DimensionMismatch(
        "CT node-coordinate arrays must have identical sizes",
    ))
    ni, nj, nk = size(x)
    edge_x = Array{FT,3}(undef, ni - 1, nj, nk)
    edge_y = Array{FT,3}(undef, ni, nj - 1, nk)
    edge_z = Array{FT,3}(undef, ni, nj, nk - 1)
    @inbounds for k in 1:nk, j in 1:nj, i in 1:ni-1
        p0 = SVector{3,FT}(x[i,j,k], y[i,j,k], z[i,j,k])
        p1 = SVector{3,FT}(x[i+1,j,k], y[i+1,j,k], z[i+1,j,k])
        a0 = vector_potential(p0[1], p0[2], p0[3], FT(time))
        a1 = vector_potential(p1[1], p1[2], p1[3], FT(time))
        edge_x[i,j,k] = dot((a0 + a1) / 2, p1 - p0)
    end
    @inbounds for k in 1:nk, j in 1:nj-1, i in 1:ni
        p0 = SVector{3,FT}(x[i,j,k], y[i,j,k], z[i,j,k])
        p1 = SVector{3,FT}(x[i,j+1,k], y[i,j+1,k], z[i,j+1,k])
        a0 = vector_potential(p0[1], p0[2], p0[3], FT(time))
        a1 = vector_potential(p1[1], p1[2], p1[3], FT(time))
        edge_y[i,j,k] = dot((a0 + a1) / 2, p1 - p0)
    end
    @inbounds for k in 1:nk-1, j in 1:nj, i in 1:ni
        p0 = SVector{3,FT}(x[i,j,k], y[i,j,k], z[i,j,k])
        p1 = SVector{3,FT}(x[i,j,k+1], y[i,j,k+1], z[i,j,k+1])
        a0 = vector_potential(p0[1], p0[2], p0[3], FT(time))
        a1 = vector_potential(p1[1], p1[2], p1[3], FT(time))
        edge_z[i,j,k] = dot((a0 + a1) / 2, p1 - p0)
    end
    phi_x, phi_y, phi_z = ct_face_fluxes_from_edge_integrals(
        edge_x, edge_y, edge_z,
    )
    i_cells = NG+1:NG+b.Nx
    j_cells = NG+1:NG+b.Ny
    k_cells = NG+1:NG+b.Nz
    i_faces = NG+1:NG+b.Nx+1
    j_faces = NG+1:NG+b.Ny+1
    k_faces = NG+1:NG+b.Nz+1
    copyto!(
        @view(b.Bx_face[i_faces, j_cells, k_cells]),
        GPUArray(Array(@view(phi_x[i_faces, j_cells, k_cells]))),
    )
    copyto!(
        @view(b.By_face[i_cells, j_faces, k_cells]),
        GPUArray(Array(@view(phi_y[i_cells, j_faces, k_cells]))),
    )
    copyto!(
        @view(b.Bz_face[i_cells, j_cells, k_faces]),
        GPUArray(Array(@view(phi_z[i_cells, j_cells, k_faces]))),
    )
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
    flux, rho_sum, U_stage, Areai, nxi, nyi, nzi, Vol,
    face_index, tangent_x, tangent_y, tangent_z, dt,
)
    face_i, cell_j, cell_k = face_index
    gi, gj, gk = ct_xface_geometry_indices(
        face_i, cell_j, cell_k, Int32(NG),
    )
    @inbounds begin
        area = Areai[gi, gj, gk]
        normal_x = nxi[gi, gj, gk]
        normal_y = nyi[gi, gj, gk]
        normal_z = nzi[gi, gj, gk]
        face_normal_emf = ct_face_normal_emf_from_state_pair(
            U_stage,
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
    flux, rho_sum, U_stage, Areaj, nxj, nyj, nzj, Vol,
    face_index, tangent_x, tangent_y, tangent_z, dt,
)
    cell_i, face_j, cell_k = face_index
    gi, gj, gk = ct_yface_geometry_indices(
        cell_i, face_j, cell_k, Int32(NG),
    )
    @inbounds begin
        area = Areaj[gi, gj, gk]
        normal_x = nxj[gi, gj, gk]
        normal_y = nyj[gi, gj, gk]
        normal_z = nzj[gi, gj, gk]
        face_normal_emf = ct_face_normal_emf_from_state_pair(
            U_stage,
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
    flux, rho_sum, U_stage, Areak, nxk, nyk, nzk, Vol,
    face_index, tangent_x, tangent_y, tangent_z, dt,
)
    cell_i, cell_j, face_k = face_index
    gi, gj, gk = ct_zface_geometry_indices(
        cell_i, cell_j, face_k, Int32(NG),
    )
    @inbounds begin
        area = Areak[gi, gj, gk]
        normal_x = nxk[gi, gj, gk]
        normal_y = nyk[gi, gj, gk]
        normal_z = nzk[gi, gj, gk]
        face_normal_emf = ct_face_normal_emf_from_state_pair(
            U_stage,
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

# Metric-aware SG07 operator. Face fluxes are projected onto a common physical
# edge tangent, and the result is stored as an oriented line-integrated EMF.
function ct_compute_edge_line_emf_kernel!(
    Ex_edge, Ey_edge, Ez_edge,
    Fx, Fy, Fz, rho_sum_x, rho_sum_y, rho_sum_z, U_stage,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    Vol, x, y, z, dt, nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)

    if i <= nxp + Int32(1) && j <= nyp + Int32(1) && k <= nzp
        ni = i + ng; nj = j + ng; nk = k + ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[ni,nj,nk+Int32(1)] - x[ni,nj,nk],
            y[ni,nj,nk+Int32(1)] - y[ni,nj,nk],
            z[ni,nj,nk+Int32(1)] - z[ni,nj,nk],
        )
        (xf_jm, xf_j), (yf_im, yf_i) = ct_ez_face_flux_indices(i, j, k)
        ez_xf_jm, w_xf_jm = ct_xface_edge_data(
            Fx, rho_sum_x, U_stage, Areai, nxi, nyi, nzi, Vol,
            xf_jm, tx, ty, tz, dt,
        )
        ez_xf_j, w_xf_j = ct_xface_edge_data(
            Fx, rho_sum_x, U_stage, Areai, nxi, nyi, nzi, Vol,
            xf_j, tx, ty, tz, dt,
        )
        ez_yf_im, w_yf_im = ct_yface_edge_data(
            Fy, rho_sum_y, U_stage, Areaj, nxj, nyj, nzj, Vol,
            yf_im, tx, ty, tz, dt,
        )
        ez_yf_i, w_yf_i = ct_yface_edge_data(
            Fy, rho_sum_y, U_stage, Areaj, nxj, nyj, nzj, Vol,
            yf_i, tx, ty, tz, dt,
        )
        cc_jm_im = ct_cell_centered_edge_emf_from_state(
            U_stage, i-Int32(1)+ng, j-Int32(1)+ng, k+ng, tx, ty, tz,
        )
        cc_jm_i = ct_cell_centered_edge_emf_from_state(
            U_stage, i+ng, j-Int32(1)+ng, k+ng, tx, ty, tz,
        )
        cc_j_im = ct_cell_centered_edge_emf_from_state(
            U_stage, i-Int32(1)+ng, j+ng, k+ng, tx, ty, tz,
        )
        cc_j_i = ct_cell_centered_edge_emf_from_state(
            U_stage, i+ng, j+ng, k+ng, tx, ty, tz,
        )
        @inbounds Ez_edge[i,j,k] = edge_length * ct_sg07_edge_emf(
            ez_xf_jm, ez_xf_j, ez_yf_im, ez_yf_i,
            cc_jm_im, cc_jm_i, cc_j_im, cc_j_i,
            w_xf_jm, w_xf_j, w_yf_im, w_yf_i,
        )
    end

    if i <= nxp + Int32(1) && j <= nyp && k <= nzp + Int32(1)
        ni = i + ng; nj = j + ng; nk = k + ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[ni,nj+Int32(1),nk] - x[ni,nj,nk],
            y[ni,nj+Int32(1),nk] - y[ni,nj,nk],
            z[ni,nj+Int32(1),nk] - z[ni,nj,nk],
        )
        (xf_km, xf_k), (zf_im, zf_i) = ct_ey_face_flux_indices(i, j, k)
        ey_xf_km, w_xf_km = ct_xface_edge_data(
            Fx, rho_sum_x, U_stage, Areai, nxi, nyi, nzi, Vol,
            xf_km, tx, ty, tz, dt,
        )
        ey_xf_k, w_xf_k = ct_xface_edge_data(
            Fx, rho_sum_x, U_stage, Areai, nxi, nyi, nzi, Vol,
            xf_k, tx, ty, tz, dt,
        )
        ey_zf_im, w_zf_im = ct_zface_edge_data(
            Fz, rho_sum_z, U_stage, Areak, nxk, nyk, nzk, Vol,
            zf_im, tx, ty, tz, dt,
        )
        ey_zf_i, w_zf_i = ct_zface_edge_data(
            Fz, rho_sum_z, U_stage, Areak, nxk, nyk, nzk, Vol,
            zf_i, tx, ty, tz, dt,
        )
        cc_km_im = ct_cell_centered_edge_emf_from_state(
            U_stage, i-Int32(1)+ng, j+ng, k-Int32(1)+ng, tx, ty, tz,
        )
        cc_km_i = ct_cell_centered_edge_emf_from_state(
            U_stage, i+ng, j+ng, k-Int32(1)+ng, tx, ty, tz,
        )
        cc_k_im = ct_cell_centered_edge_emf_from_state(
            U_stage, i-Int32(1)+ng, j+ng, k+ng, tx, ty, tz,
        )
        cc_k_i = ct_cell_centered_edge_emf_from_state(
            U_stage, i+ng, j+ng, k+ng, tx, ty, tz,
        )
        @inbounds Ey_edge[i,j,k] = edge_length * ct_sg07_edge_emf(
            ey_xf_km, ey_xf_k, ey_zf_im, ey_zf_i,
            cc_km_im, cc_km_i, cc_k_im, cc_k_i,
            w_xf_km, w_xf_k, w_zf_im, w_zf_i,
        )
    end

    if i <= nxp && j <= nyp + Int32(1) && k <= nzp + Int32(1)
        ni = i + ng; nj = j + ng; nk = k + ng
        @inbounds tx, ty, tz, edge_length = ct_edge_geometry(
            x[ni+Int32(1),nj,nk] - x[ni,nj,nk],
            y[ni+Int32(1),nj,nk] - y[ni,nj,nk],
            z[ni+Int32(1),nj,nk] - z[ni,nj,nk],
        )
        (yf_km, yf_k), (zf_jm, zf_j) = ct_ex_face_flux_indices(i, j, k)
        ex_yf_km, w_yf_km = ct_yface_edge_data(
            Fy, rho_sum_y, U_stage, Areaj, nxj, nyj, nzj, Vol,
            yf_km, tx, ty, tz, dt,
        )
        ex_yf_k, w_yf_k = ct_yface_edge_data(
            Fy, rho_sum_y, U_stage, Areaj, nxj, nyj, nzj, Vol,
            yf_k, tx, ty, tz, dt,
        )
        ex_zf_jm, w_zf_jm = ct_zface_edge_data(
            Fz, rho_sum_z, U_stage, Areak, nxk, nyk, nzk, Vol,
            zf_jm, tx, ty, tz, dt,
        )
        ex_zf_j, w_zf_j = ct_zface_edge_data(
            Fz, rho_sum_z, U_stage, Areak, nxk, nyk, nzk, Vol,
            zf_j, tx, ty, tz, dt,
        )
        cc_km_jm = ct_cell_centered_edge_emf_from_state(
            U_stage, i+ng, j-Int32(1)+ng, k-Int32(1)+ng, tx, ty, tz,
        )
        cc_km_j = ct_cell_centered_edge_emf_from_state(
            U_stage, i+ng, j+ng, k-Int32(1)+ng, tx, ty, tz,
        )
        cc_k_jm = ct_cell_centered_edge_emf_from_state(
            U_stage, i+ng, j-Int32(1)+ng, k+ng, tx, ty, tz,
        )
        cc_k_j = ct_cell_centered_edge_emf_from_state(
            U_stage, i+ng, j+ng, k+ng, tx, ty, tz,
        )
        @inbounds Ex_edge[i,j,k] = edge_length * ct_sg07_edge_emf(
            ex_yf_km, ex_yf_k, ex_zf_jm, ex_zf_j,
            cc_km_jm, cc_km_j, cc_k_jm, cc_k_j,
            w_yf_km, w_yf_k, w_zf_jm, w_zf_j,
        )
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

@inline function _ct_resistive_face_flux_kernel!(
    Q, flux, emf_x, emf_y, emf_z,
    area_array, nx_array, ny_array, nz_array,
    nxp, nyp, nzp, ::Val{DIRECTION},
) where {DIRECTION}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ng = Int32(NG)
    i_hi = nxp + ng + (DIRECTION == 1 ? Int32(0) : Int32(1))
    j_hi = nyp + ng + (DIRECTION == 2 ? Int32(0) : Int32(1))
    k_hi = nzp + ng + (DIRECTION == 3 ? Int32(0) : Int32(1))
    if i < ng || i > i_hi || j < ng || j > j_hi ||
       k < ng || k > k_hi
        return
    end

    di = DIRECTION == 1 ? Int32(1) : Int32(0)
    dj = DIRECTION == 2 ? Int32(1) : Int32(0)
    dk = DIRECTION == 3 ? Int32(1) : Int32(0)
    ri, rj, rk = i+di, j+dj, k+dk
    gi, gj, gk = ri, rj, rk
    @inbounds begin
        area = area_array[gi,gj,gk]
        normal_x = nx_array[gi,gj,gk]
        normal_y = ny_array[gi,gj,gk]
        normal_z = nz_array[gi,gj,gk]
        electric_x = FT(0.5) * (emf_x[i,j,k] + emf_x[ri,rj,rk])
        electric_y = FT(0.5) * (emf_y[i,j,k] + emf_y[ri,rj,rk])
        electric_z = FT(0.5) * (emf_z[i,j,k] + emf_z[ri,rj,rk])
        magnetic_x = FT(0.5) * (Q[i,j,k,QBX] + Q[ri,rj,rk,QBX])
        magnetic_y = FT(0.5) * (Q[i,j,k,QBY] + Q[ri,rj,rk,QBY])
        magnetic_z = FT(0.5) * (Q[i,j,k,QBZ] + Q[ri,rj,rk,QBZ])
    end
    flux_bx = electric_y*normal_z - electric_z*normal_y
    flux_by = electric_z*normal_x - electric_x*normal_z
    flux_bz = electric_x*normal_y - electric_y*normal_x
    flux_energy = flux_bx*magnetic_x +
                  flux_by*magnetic_y + flux_bz*magnetic_z
    fi, fj, fk = i-ng+Int32(1), j-ng+Int32(1), k-ng+Int32(1)
    @inbounds begin
        for component in 1:Ncons
            flux[fi,fj,fk,component] = zero(FT)
        end
        flux[fi,fj,fk,5] = flux_energy * area
        flux[fi,fj,fk,UBX] = flux_bx * area
        flux[fi,fj,fk,UBY] = flux_by * area
        flux[fi,fj,fk,UBZ] = flux_bz * area
    end
    return
end

function ct_resistive_face_flux_i_kernel!(
    Q, flux, emf_x, emf_y, emf_z, area, nx, ny, nz,
    nxp, nyp, nzp,
)
    _ct_resistive_face_flux_kernel!(
        Q, flux, emf_x, emf_y, emf_z, area, nx, ny, nz,
        nxp, nyp, nzp, Val(1),
    )
end

function ct_resistive_face_flux_j_kernel!(
    Q, flux, emf_x, emf_y, emf_z, area, nx, ny, nz,
    nxp, nyp, nzp,
)
    _ct_resistive_face_flux_kernel!(
        Q, flux, emf_x, emf_y, emf_z, area, nx, ny, nz,
        nxp, nyp, nzp, Val(2),
    )
end

function ct_resistive_face_flux_k_kernel!(
    Q, flux, emf_x, emf_y, emf_z, area, nx, ny, nz,
    nxp, nyp, nzp,
)
    _ct_resistive_face_flux_kernel!(
        Q, flux, emf_x, emf_y, emf_z, area, nx, ny, nz,
        nxp, nyp, nzp, Val(3),
    )
end

@inline function ct_resistive_energy_rate(
    flux_x, flux_y, flux_z, inverse_volume, i, j, k, ii, jj, kk,
)
    return inverse_volume[ii,jj,kk] * (
        flux_x[i+Int32(1),j,k,5] - flux_x[i,j,k,5] +
        flux_y[i,j+Int32(1),k,5] - flux_y[i,j,k,5] +
        flux_z[i,j,k+Int32(1),5] - flux_z[i,j,k,5]
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

# ─── CT: Recover cell-centered Cartesian B from six face fluxes ───
function ct_sync_cell_b_kernel!(
    U, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
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
        magnetic = ct_recover_cell_b(
            area_i_lo, area_i_hi, area_j_lo, area_j_hi, area_k_lo, area_k_hi,
            Bx_face[ii,jj,kk], Bx_face[ii+Int32(1),jj,kk],
            By_face[ii,jj,kk], By_face[ii,jj+Int32(1),kk],
            Bz_face[ii,jj,kk], Bz_face[ii,jj,kk+Int32(1)],
        )
        U[ii, jj, kk, UBX] = magnetic[1]
        U[ii, jj, kk, UBY] = magnetic[2]
        U[ii, jj, kk, UBZ] = magnetic[3]
    end
    return
end

function ct_sync_cell_b_point6_kernel!(
    U, Bx_face, By_face, Bz_face,
    Areai, nxi, nyi, nzi,
    Areaj, nxj, nyj, nzj,
    Areak, nxk, nyk, nzk,
    nxp, nyp, nzp,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
    magnetic = ct_recover_cell_b_point6(
        Bx_face, By_face, Bz_face,
        Areai, nxi, nyi, nzi,
        Areaj, nxj, nyj, nzj,
        Areak, nxk, nyk, nzk,
        ii, jj, kk,
    )
    @inbounds begin
        U[ii, jj, kk, UBX] = magnetic[1]
        U[ii, jj, kk, UBY] = magnetic[2]
        U[ii, jj, kk, UBZ] = magnetic[3]
    end
    return
end

function ct_sync_cell_b_lsq2!(b, nxp, nyp, nzp)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb ct_sync_cell_b_kernel!(
        b.U, b.Bx_face, b.By_face, b.Bz_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        Int32(nxp), Int32(nyp), Int32(nzp))
end

function ct_sync_cell_b_point6!(b, nxp, nyp, nzp)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb ct_sync_cell_b_point6_kernel!(
        b.U, b.Bx_face, b.By_face, b.Bz_face,
        b.Areai, b.nxi, b.nyi, b.nzi,
        b.Areaj, b.nxj, b.nyj, b.nzj,
        b.Areak, b.nxk, b.nyk, b.nzk,
        Int32(nxp), Int32(nyp), Int32(nzp))
end

@inline function ct_sync_cell_b!(b, nxp, nyp, nzp)
    @static if isdefined(@__MODULE__, :ct_cell_b_recovery) &&
               ct_cell_b_recovery == CT_CELL_B_POINT6
        ct_sync_cell_b_point6!(b, nxp, nyp, nzp)
    else
        ct_sync_cell_b_lsq2!(b, nxp, nyp, nzp)
    end
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

# ─── CT: Update Q (primitive) B from U, recompute pressure ───
# After CT updates face-B and sync to cell-B (U[6:8]), the div kernel's c2Prim
# computed Q[5] (pressure) using the OLD B (before CT sync). We recompute Q[5]
# and Q[6] (temperature) from U[5] (which was NOT overwritten in CT mode) with
# the new B. We also update Q[7:9] (B) from the synced U[6:8].
function ct_update_q_b_kernel!(Q, U, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG

    @inbounds state = SVector{9,FT}(
        (
            U[ii, jj, kk, 1], U[ii, jj, kk, 2], U[ii, jj, kk, 3],
            U[ii, jj, kk, 4], U[ii, jj, kk, 5], U[ii, jj, kk, 6],
            U[ii, jj, kk, 7], U[ii, jj, kk, 8], U[ii, jj, kk, 9],
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
        Q[ii,jj,kk,7]=state[6]; Q[ii,jj,kk,8]=state[7]
        Q[ii,jj,kk,9]=state[8]; Q[ii,jj,kk,10]=state[9]
    end
    return
end

function ct_update_q_point6_kernel!(Q, U, inverse_volume, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    if i > nxp || j > nyp || k > nzp; return; end
    ii = i + NG; jj = j + NG; kk = k + NG
    hydro = ct_conservative_average_to_point6(
        U, inverse_volume, ii, jj, kk,
    )
    @inbounds magnetic = SVector{3,FT}(
        U[ii, jj, kk, UBX],
        U[ii, jj, kk, UBY],
        U[ii, jj, kk, UBZ],
    )
    primitive = ct_mhd_point_conservative_to_primitive(
        hydro, magnetic, FT(γ),
    )
    temperature = primitive[5] / (primitive[1] * FT(Rg))
    @inbounds begin
        Q[ii, jj, kk, 1] = primitive[1]
        Q[ii, jj, kk, 2] = primitive[2]
        Q[ii, jj, kk, 3] = primitive[3]
        Q[ii, jj, kk, 4] = primitive[4]
        Q[ii, jj, kk, 5] = primitive[5]
        Q[ii, jj, kk, 6] = temperature
        Q[ii, jj, kk, 7] = primitive[6]
        Q[ii, jj, kk, 8] = primitive[7]
        Q[ii, jj, kk, 9] = primitive[8]
        Q[ii, jj, kk, 10] = primitive[9]
    end
    return
end

function ct_update_q_b!(b, nxp, nyp, nzp)
    nb = (cld(nxp, nthreads[1]), cld(nyp, nthreads[2]), cld(nzp, nthreads[3]))
    @static if isdefined(@__MODULE__, :ct_primitive_recovery) &&
               ct_primitive_recovery == CT_PRIMITIVE_POINT6
        @gpu_launch threads=nthreads blocks=nb ct_update_q_point6_kernel!(
            b.Q, b.U, b.Vol, Int32(nxp), Int32(nyp), Int32(nzp))
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
# Same direction-aware structure as periodic fill.
function ct_zerograd_face_b_kernel!(
    Bface, U, Area, normal_x, normal_y, normal_z,
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
                    bx = FT(0.5) * (U[left_i, source_j, source_k, UBX] +
                                    U[right_i, source_j, source_k, UBX])
                    by = FT(0.5) * (U[left_i, source_j, source_k, UBY] +
                                    U[right_i, source_j, source_k, UBY])
                    bz = FT(0.5) * (U[left_i, source_j, source_k, UBZ] +
                                    U[right_i, source_j, source_k, UBZ])
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
                    bx = FT(0.5) * (U[source_i, left_j, source_k, UBX] +
                                    U[source_i, right_j, source_k, UBX])
                    by = FT(0.5) * (U[source_i, left_j, source_k, UBY] +
                                    U[source_i, right_j, source_k, UBY])
                    bz = FT(0.5) * (U[source_i, left_j, source_k, UBZ] +
                                    U[source_i, right_j, source_k, UBZ])
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
                    bx = FT(0.5) * (U[source_i, source_j, left_k, UBX] +
                                    U[source_i, source_j, right_k, UBX])
                    by = FT(0.5) * (U[source_i, source_j, left_k, UBY] +
                                    U[source_i, source_j, right_k, UBY])
                    bz = FT(0.5) * (U[source_i, source_j, left_k, UBZ] +
                                    U[source_i, source_j, right_k, UBZ])
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
        b.Bx_face, b.U, b.Areai, b.nxi, b.nyi, b.nzi,
        Int32(1), Int32(nxp), Int32(nyp), Int32(nzp))
    n1y = nxp + 2*NG + 1; n2y = nyp + 1 + 2*NG; n3y = nzp + 2*NG + 1
    nb_y = (cld(n1y, nthreads[1]), cld(n2y, nthreads[2]), cld(n3y, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_y ct_zerograd_face_b_kernel!(
        b.By_face, b.U, b.Areaj, b.nxj, b.nyj, b.nzj,
        Int32(2), Int32(nxp), Int32(nyp), Int32(nzp))
    n1z = nxp + 2*NG + 1; n2z = nyp + 2*NG + 1; n3z = nzp + 1 + 2*NG
    nb_z = (cld(n1z, nthreads[1]), cld(n2z, nthreads[2]), cld(n3z, nthreads[3]))
    @gpu_launch threads=nthreads blocks=nb_z ct_zerograd_face_b_kernel!(
        b.Bz_face, b.U, b.Areak, b.nxk, b.nyk, b.nzk,
        Int32(3), Int32(nxp), Int32(nyp), Int32(nzp))
end
