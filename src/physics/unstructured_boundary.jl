# Physical ghost-cell boundary conditions for the unstructured backend.

function validate_unstruct_boundary_support(block::UnstructBlock)
    supported = @static if equation_type == :MHD
        Set((
            0,
            Int(BC_INTERBLOCK),
            Int(BC_PERIODIC),
            Int(BC_ZERO_GRADIENT),
            Int(BC_SUPERSONIC_OUTFLOW),
        ))
    else
        Set((
            0,
            Int(BC_INTERBLOCK),
            Int(BC_PERIODIC),
            Int(BC_SUPERSONIC_INFLOW),
            Int(BC_SUPERSONIC_OUTFLOW),
            Int(BC_FARFIELD),
            Int(BC_SYMMETRY),
            Int(BC_SLIP_WALL),
            Int(BC_ISOTHERMAL_WALL),
            Int(BC_ADIABATIC_WALL),
            Int(BC_ZERO_GRADIENT),
        ))
    end
    boundary_ids = Array(block.face_bc_id)
    unsupported = sort!(unique(filter(
        boundary_id -> !(boundary_id in supported), boundary_ids,
    )))
    isempty(unsupported) || throw(ArgumentError(
        "unsupported unstructured boundary IDs: $(join(unsupported, ", "))",
    ))
    periodic_face_count = count(==(Int(BC_PERIODIC)), boundary_ids)
    paired_face_count=2*length(block.periodic_face_primary)
    if block.partition===nothing
        periodic_face_count == paired_face_count || throw(ArgumentError(
            "periodic unstructured faces require an explicit paired-face map"))
    else
        paired_face_count<=periodic_face_count || throw(ArgumentError(
            "partitioned periodic paired-face count exceeds periodic face count"))
        periodic_face_count==length(block.periodic_ghost_cells) ||
            throw(ArgumentError(
                "every partitioned periodic face requires a ghost/source map"))
    end
    return
end

function fill_unstruct_periodic_state_kernel!(
    Q, U, ghost_cells, source_cells, nperiodic::Int,
)
    entry = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    entry > nperiodic && return
    @inbounds ghost = ghost_cells[entry]
    @inbounds source = source_cells[entry]
    @inbounds for variable in 1:Nprim
        Q[ghost,variable] = Q[source,variable]
    end
    @inbounds for variable in 1:Ncons
        U[ghost,variable] = U[source,variable]
    end
    return
end

function fill_unstruct_periodic_gradient_kernel!(
    grad, ghost_cells, source_cells, nperiodic::Int,
)
    entry = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    entry > nperiodic && return
    @inbounds ghost = ghost_cells[entry]
    @inbounds source = source_cells[entry]
    @inbounds for direction in 1:3, variable in 1:Nprim
        grad[ghost,variable,direction] = grad[source,variable,direction]
    end
    return
end

function fill_unstruct_periodic_limiter_kernel!(
    limiter, ghost_cells, source_cells, nperiodic::Int,
)
    entry = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    entry > nperiodic && return
    @inbounds ghost = ghost_cells[entry]
    @inbounds source = source_cells[entry]
    @inbounds for variable in 1:Nprim
        limiter[ghost,variable] = limiter[source,variable]
    end
    return
end

function fill_unstruct_ghost_kernel!(
    Q, U, face_L, face_R,
    face_nx, face_ny, face_nz,
    face_bc_id, face_bc_params, nface::Int,
)
    face = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    face > nface && return
    @inbounds boundary_id = face_bc_id[face]
    if boundary_id == 0 || boundary_id == Int(BC_PERIODIC) ||
       boundary_id == Int(BC_INTERBLOCK)
        return
    end

    @inbounds left = face_L[face]
    @inbounds ghost = face_R[face]
    @inbounds density = Q[left,1]
    @inbounds velocity_x = Q[left,2]
    @inbounds velocity_y = Q[left,3]
    @inbounds velocity_z = Q[left,4]
    @inbounds pressure = Q[left,5]
    temperature = pressure/(density*Rg)

    @static if equation_type == :compressible
        if boundary_id == Int(BC_SUPERSONIC_INFLOW) || boundary_id == Int(BC_FARFIELD)
            @inbounds density = face_bc_params[face,BCP_RHO_INF]
            @inbounds velocity_x = face_bc_params[face,BCP_U_INF]
            @inbounds velocity_y = face_bc_params[face,BCP_V_INF]
            @inbounds velocity_z = face_bc_params[face,BCP_W_INF]
            @inbounds pressure = face_bc_params[face,BCP_P_INF]
        elseif boundary_id == Int(BC_SYMMETRY) || boundary_id == Int(BC_SLIP_WALL)
            @inbounds nx = face_nx[face]
            @inbounds ny = face_ny[face]
            @inbounds nz = face_nz[face]
            normal_velocity = velocity_x*nx + velocity_y*ny + velocity_z*nz
            velocity_x -= FT(2)*normal_velocity*nx
            velocity_y -= FT(2)*normal_velocity*ny
            velocity_z -= FT(2)*normal_velocity*nz
        elseif boundary_id == Int(BC_ISOTHERMAL_WALL)
            velocity_x = -velocity_x
            velocity_y = -velocity_y
            velocity_z = -velocity_z
            temperature = max(FT(2)*Tw - temperature, eps(FT))
            density = pressure/(Rg*temperature)
        elseif boundary_id == Int(BC_ADIABATIC_WALL)
            velocity_x = -velocity_x
            velocity_y = -velocity_y
            velocity_z = -velocity_z
        end
        total_energy = pressure/(γ - one(FT)) + FT(0.5)*density*(
            velocity_x^2 + velocity_y^2 + velocity_z^2)
        @inbounds begin
            Q[ghost,1] = density
            Q[ghost,2] = velocity_x
            Q[ghost,3] = velocity_y
            Q[ghost,4] = velocity_z
            Q[ghost,5] = pressure
            Q[ghost,6] = temperature
            U[ghost,1] = density
            U[ghost,2] = density*velocity_x
            U[ghost,3] = density*velocity_y
            U[ghost,4] = density*velocity_z
            U[ghost,5] = total_energy
        end
    else
        @inbounds primitive = SVector{10,FT}(
            density, velocity_x, velocity_y, velocity_z, pressure,
            Q[left,6], Q[left,7], Q[left,8], Q[left,9], Q[left,10],
        )
        conservative = mhd_primitive_to_conservative(primitive, FT(γ))
        @inbounds for variable in 1:Nprim
            Q[ghost,variable] = primitive[variable]
        end
        @inbounds for variable in 1:Ncons
            U[ghost,variable] = conservative[variable]
        end
    end
    return
end

function fill_unstruct_ghost!(block::UnstructBlock)
    @gpu_launch threads=block.nthreads_face blocks=block.nb_face fill_unstruct_ghost_kernel!(
        block.Q, block.U, block.face_L, block.face_R,
        block.face_nx, block.face_ny, block.face_nz,
        block.face_bc_id, block.face_bc_params, block.nface,
    )
    nperiodic = length(block.periodic_ghost_cells)
    if nperiodic > 0
        blocks = cld(nperiodic, block.nthreads_cell)
        @gpu_launch threads=block.nthreads_cell blocks=blocks fill_unstruct_periodic_state_kernel!(
            block.Q, block.U,
            block.periodic_ghost_cells, block.periodic_source_cells,
            nperiodic,
        )
    end
    return
end

function fill_unstruct_periodic_gradient!(block::UnstructBlock)
    nperiodic = length(block.periodic_ghost_cells)
    nperiodic == 0 && return
    blocks = cld(nperiodic, block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=blocks fill_unstruct_periodic_gradient_kernel!(
        block.grad, block.periodic_ghost_cells, block.periodic_source_cells,
        nperiodic,
    )
    return
end

function fill_unstruct_periodic_limiter!(block::UnstructBlock)
    nperiodic = length(block.periodic_ghost_cells)
    nperiodic == 0 && return
    blocks = cld(nperiodic, block.nthreads_cell)
    @gpu_launch threads=block.nthreads_cell blocks=blocks fill_unstruct_periodic_limiter_kernel!(
        block.reconstruction_limiter,
        block.periodic_ghost_cells, block.periodic_source_cells,
        nperiodic,
    )
    return
end
