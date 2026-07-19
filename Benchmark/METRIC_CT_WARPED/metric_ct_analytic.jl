using LinearAlgebra
using StaticArrays

@inline function metric_ct_alfven_primitive(
    x::T, time::T, rg::T, amplitude::T,
) where {T}
    phase = x - time
    by = amplitude * sin(phase)
    bz = amplitude * cos(phase)
    rho = one(T)
    pressure = one(T)
    return SVector{10,T}(
        rho, zero(T), -by, -bz, pressure, pressure/(rho*rg),
        one(T), by, bz, zero(T),
    )
end

@inline function metric_ct_vector_potential(
    x::T, y::T, time::T, amplitude::T,
) where {T}
    phase = x - time
    return SVector{3,T}(
        zero(T), amplitude*sin(phase), y + amplitude*cos(phase),
    )
end

@inline function metric_ct_cell_center(x, y, z, i, j, k)
    scale = one(eltype(x)) / 8
    @inbounds return SVector(
        scale * (
            x[i,   j,   k]   + x[i+1, j,   k] +
            x[i,   j+1, k]   + x[i+1, j+1, k] +
            x[i,   j,   k+1] + x[i+1, j,   k+1] +
            x[i,   j+1, k+1] + x[i+1, j+1, k+1]
        ),
        scale * (
            y[i,   j,   k]   + y[i+1, j,   k] +
            y[i,   j+1, k]   + y[i+1, j+1, k] +
            y[i,   j,   k+1] + y[i+1, j,   k+1] +
            y[i,   j+1, k+1] + y[i+1, j+1, k+1]
        ),
        scale * (
            z[i,   j,   k]   + z[i+1, j,   k] +
            z[i,   j+1, k]   + z[i+1, j+1, k] +
            z[i,   j,   k+1] + z[i+1, j,   k+1] +
            z[i,   j+1, k+1] + z[i+1, j+1, k+1]
        ),
    )
end

@inline function metric_ct_edge_integral(p0, p1, time, amplitude)
    a0 = metric_ct_vector_potential(p0[1], p0[2], time, amplitude)
    a1 = metric_ct_vector_potential(p1[1], p1[2], time, amplitude)
    return dot((a0 + a1)/2, p1 - p0)
end

function metric_ct_face_fluxes_from_nodes(x, y, z, time, amplitude)
    size(x) == size(y) == size(z) || throw(DimensionMismatch(
        "metric CT node-coordinate arrays must have identical sizes",
    ))
    ndims(x) == 3 || throw(DimensionMismatch(
        "metric CT node-coordinate arrays must be three-dimensional",
    ))
    ni, nj, nk = size(x)
    min(ni, nj, nk) >= 1 || throw(DimensionMismatch(
        "metric CT node-coordinate arrays must be nonempty",
    ))

    T = promote_type(
        eltype(x), eltype(y), eltype(z), typeof(time), typeof(amplitude),
    )
    edge_time = T(time)
    edge_amplitude = T(amplitude)
    lx = Array{T,3}(undef, ni-1, nj, nk)
    ly = Array{T,3}(undef, ni, nj-1, nk)
    lz = Array{T,3}(undef, ni, nj, nk-1)

    @inbounds for k in 1:nk, j in 1:nj, i in 1:ni-1
        p0 = SVector{3,T}(x[i,j,k], y[i,j,k], z[i,j,k])
        p1 = SVector{3,T}(x[i+1,j,k], y[i+1,j,k], z[i+1,j,k])
        lx[i,j,k] = metric_ct_edge_integral(
            p0, p1, edge_time, edge_amplitude,
        )
    end
    @inbounds for k in 1:nk, j in 1:nj-1, i in 1:ni
        p0 = SVector{3,T}(x[i,j,k], y[i,j,k], z[i,j,k])
        p1 = SVector{3,T}(x[i,j+1,k], y[i,j+1,k], z[i,j+1,k])
        ly[i,j,k] = metric_ct_edge_integral(
            p0, p1, edge_time, edge_amplitude,
        )
    end
    @inbounds for k in 1:nk-1, j in 1:nj, i in 1:ni
        p0 = SVector{3,T}(x[i,j,k], y[i,j,k], z[i,j,k])
        p1 = SVector{3,T}(x[i,j,k+1], y[i,j,k+1], z[i,j,k+1])
        lz[i,j,k] = metric_ct_edge_integral(
            p0, p1, edge_time, edge_amplitude,
        )
    end

    phi_x = Array{T,3}(undef, ni, nj-1, nk-1)
    phi_y = Array{T,3}(undef, ni-1, nj, nk-1)
    phi_z = Array{T,3}(undef, ni-1, nj-1, nk)
    @inbounds for k in 1:nk-1, j in 1:nj-1, i in 1:ni
        phi_x[i,j,k] = ly[i,j,k] + lz[i,j+1,k] -
                       ly[i,j,k+1] - lz[i,j,k]
    end
    @inbounds for k in 1:nk-1, j in 1:nj, i in 1:ni-1
        phi_y[i,j,k] = lz[i,j,k] + lx[i,j,k+1] -
                       lz[i+1,j,k] - lx[i,j,k]
    end
    @inbounds for k in 1:nk, j in 1:nj-1, i in 1:ni-1
        phi_z[i,j,k] = lx[i,j,k] + ly[i+1,j,k] -
                       lx[i,j+1,k] - ly[i,j,k]
    end
    return phi_x, phi_y, phi_z
end

function in_situ_ct_initial_face_flux_process(
    blocks, world_rank, Block_Nprocs, block_comms,
)
    for b in values(blocks)
        b.Bx_face === nothing && continue
        x = Array(b.x)
        y = Array(b.y)
        z = Array(b.z)
        ng = div(size(x, 1) - b.Nx - 1, 2)
        expected_size = (b.Nx + 2ng + 1, b.Ny + 2ng + 1, b.Nz + 2ng + 1)
        size(x) == expected_size || throw(DimensionMismatch(
            "metric CT block node coordinates are inconsistent with cell extents",
        ))

        T = eltype(x)
        phi_x, phi_y, phi_z = metric_ct_face_fluxes_from_nodes(
            x, y, z, zero(T), T(1e-3),
        )
        i_cells = ng+1:ng+b.Nx
        j_cells = ng+1:ng+b.Ny
        k_cells = ng+1:ng+b.Nz
        i_faces = ng+1:ng+b.Nx+1
        j_faces = ng+1:ng+b.Ny+1
        k_faces = ng+1:ng+b.Nz+1
        _ct_copy_host_sheet_to_device!(
            @view(b.Bx_face[i_faces, j_cells, k_cells]),
            @view(phi_x[i_faces, j_cells, k_cells]),
        )
        _ct_copy_host_sheet_to_device!(
            @view(b.By_face[i_cells, j_faces, k_cells]),
            @view(phi_y[i_cells, j_faces, k_cells]),
        )
        _ct_copy_host_sheet_to_device!(
            @view(b.Bz_face[i_cells, j_cells, k_faces]),
            @view(phi_z[i_cells, j_cells, k_faces]),
        )
    end
    return nothing
end
