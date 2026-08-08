# Viscous face fluxes for the unstructured compressible Navier-Stokes backend.

if !@isdefined(MHD_SI_UNITS_LOADED)
    include(joinpath(@__DIR__, "..", "core", "mhd_units.jl"))
end

if !@isdefined(viscous)
    const viscous = false
end
if !@isdefined(resistive)
    const resistive = false
end
if !@isdefined(η_mhd)
    const η_mhd = zero(FT)
end

@inline function unstruct_viscosity(temperature::FT)
    safe_temperature = max(temperature, eps(FT))
    return C_s * safe_temperature * sqrt(safe_temperature) /
        (safe_temperature + T_s)
end

function unstruct_resistive_flux_kernel!(
    Fv_faces,Q,grad,face_L,face_R,face_bc_id,
    face_nx,face_ny,face_nz,face_area,cell_cx,cell_cy,cell_cz,
    nface::Int,resistivity::FT,
)
    face=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    face>nface && return
    @inbounds left=face_L[face]
    @inbounds right=face_R[face]
    @inbounds use_right=face_bc_id[face]==0
    @inbounds begin
        dx=cell_cx[right]-cell_cx[left]
        dy=cell_cy[right]-cell_cy[left]
        dz=cell_cz[right]-cell_cz[left]
        dBxdx,dBxdy,dBxdz=corrected_face_gradient(
            grad[left,7,1],grad[left,7,2],grad[left,7,3],
            grad[right,7,1],grad[right,7,2],grad[right,7,3],
            Q[left,7],Q[right,7],dx,dy,dz,use_right)
        dBydx,dBydy,dBydz=corrected_face_gradient(
            grad[left,8,1],grad[left,8,2],grad[left,8,3],
            grad[right,8,1],grad[right,8,2],grad[right,8,3],
            Q[left,8],Q[right,8],dx,dy,dz,use_right)
        dBzdx,dBzdy,dBzdz=corrected_face_gradient(
            grad[left,9,1],grad[left,9,2],grad[left,9,3],
            grad[right,9,1],grad[right,9,2],grad[right,9,3],
            Q[left,9],Q[right,9],dx,dy,dz,use_right)
        current_x=dBzdy-dBydz
        current_y=dBxdz-dBzdx
        current_z=dBydx-dBxdy
        nx=face_nx[face];ny=face_ny[face];nz=face_nz[face]
        diffusive_bx=resistivity*(current_y*nz-current_z*ny)
        diffusive_by=resistivity*(current_z*nx-current_x*nz)
        diffusive_bz=resistivity*(current_x*ny-current_y*nx)
        magnetic_x=FT(0.5)*(Q[left,7]+Q[right,7])
        magnetic_y=FT(0.5)*(Q[left,8]+Q[right,8])
        magnetic_z=FT(0.5)*(Q[left,9]+Q[right,9])
        area=face_area[face]
        Fv_faces[face,5]+=INV_MU0_SI*(
            diffusive_bx*magnetic_x+diffusive_by*magnetic_y+
            diffusive_bz*magnetic_z)*area
        Fv_faces[face,6]=diffusive_bx*area
        Fv_faces[face,7]=diffusive_by*area
        Fv_faces[face,8]=diffusive_bz*area
        Fv_faces[face,9]=zero(FT)
    end
    return
end

@inline function corrected_face_gradient(
    grad_l_x, grad_l_y, grad_l_z,
    grad_r_x, grad_r_y, grad_r_z,
    value_l, value_r, dx, dy, dz, use_right_gradient,
)
    weight_r = use_right_gradient ? FT(0.5) : zero(FT)
    weight_l = one(FT) - weight_r
    gx = weight_l*grad_l_x + weight_r*grad_r_x
    gy = weight_l*grad_l_y + weight_r*grad_r_y
    gz = weight_l*grad_l_z + weight_r*grad_r_z

    distance_squared = dx*dx + dy*dy + dz*dz
    if distance_squared > eps(FT)
        inverse_distance = inv(sqrt(distance_squared))
        ex = dx*inverse_distance
        ey = dy*inverse_distance
        ez = dz*inverse_distance
        projected_gradient = gx*ex + gy*ey + gz*ez
        two_point_gradient = (value_r - value_l)*inverse_distance
        correction = two_point_gradient - projected_gradient
        gx += correction*ex
        gy += correction*ey
        gz += correction*ez
    end
    return gx, gy, gz
end

function unstruct_viscous_flux_kernel!(
    Fv_faces, Q, grad,
    face_L, face_R, face_bc_id,
    face_nx, face_ny, face_nz, face_area,
    cell_cx, cell_cy, cell_cz,
    nface::Int,
)
    face = (blockIdx().x - Int32(1))*blockDim().x + threadIdx().x
    face > nface && return

    @inbounds left = face_L[face]
    @inbounds right = face_R[face]
    @inbounds bc_id = face_bc_id[face]
    use_right_gradient = bc_id == 0

    @inbounds dx = cell_cx[right] - cell_cx[left]
    @inbounds dy = cell_cy[right] - cell_cy[left]
    @inbounds dz = cell_cz[right] - cell_cz[left]

    @inbounds dudx, dudy, dudz = corrected_face_gradient(
        grad[left,2,1], grad[left,2,2], grad[left,2,3],
        grad[right,2,1], grad[right,2,2], grad[right,2,3],
        Q[left,2], Q[right,2], dx, dy, dz, use_right_gradient,
    )
    @inbounds dvdx, dvdy, dvdz = corrected_face_gradient(
        grad[left,3,1], grad[left,3,2], grad[left,3,3],
        grad[right,3,1], grad[right,3,2], grad[right,3,3],
        Q[left,3], Q[right,3], dx, dy, dz, use_right_gradient,
    )
    @inbounds dwdx, dwdy, dwdz = corrected_face_gradient(
        grad[left,4,1], grad[left,4,2], grad[left,4,3],
        grad[right,4,1], grad[right,4,2], grad[right,4,3],
        Q[left,4], Q[right,4], dx, dy, dz, use_right_gradient,
    )
    @inbounds dTdx, dTdy, dTdz = corrected_face_gradient(
        grad[left,6,1], grad[left,6,2], grad[left,6,3],
        grad[right,6,1], grad[right,6,2], grad[right,6,3],
        Q[left,6], Q[right,6], dx, dy, dz, use_right_gradient,
    )

    @inbounds temperature = FT(0.5)*(Q[left,6] + Q[right,6])
    dynamic_viscosity = unstruct_viscosity(temperature)
    thermal_conductivity = dynamic_viscosity*Cp/Pr
    divergence_velocity = dudx + dvdy + dwdz

    tau_xx = dynamic_viscosity*(FT(2)*dudx - FT(2)/FT(3)*divergence_velocity)
    tau_yy = dynamic_viscosity*(FT(2)*dvdy - FT(2)/FT(3)*divergence_velocity)
    tau_zz = dynamic_viscosity*(FT(2)*dwdz - FT(2)/FT(3)*divergence_velocity)
    tau_xy = dynamic_viscosity*(dudy + dvdx)
    tau_xz = dynamic_viscosity*(dudz + dwdx)
    tau_yz = dynamic_viscosity*(dvdz + dwdy)

    @inbounds nx = face_nx[face]
    @inbounds ny = face_ny[face]
    @inbounds nz = face_nz[face]
    @inbounds area = face_area[face]
    viscous_momentum_x = tau_xx*nx + tau_xy*ny + tau_xz*nz
    viscous_momentum_y = tau_xy*nx + tau_yy*ny + tau_yz*nz
    viscous_momentum_z = tau_xz*nx + tau_yz*ny + tau_zz*nz

    @inbounds velocity_x = FT(0.5)*(Q[left,2] + Q[right,2])
    @inbounds velocity_y = FT(0.5)*(Q[left,3] + Q[right,3])
    @inbounds velocity_z = FT(0.5)*(Q[left,4] + Q[right,4])
    heat_flux_normal = -thermal_conductivity*(dTdx*nx + dTdy*ny + dTdz*nz)
    viscous_energy = viscous_momentum_x*velocity_x +
        viscous_momentum_y*velocity_y + viscous_momentum_z*velocity_z -
        heat_flux_normal

    @inbounds Fv_faces[face,1] = zero(FT)
    @inbounds Fv_faces[face,2] = viscous_momentum_x*area
    @inbounds Fv_faces[face,3] = viscous_momentum_y*area
    @inbounds Fv_faces[face,4] = viscous_momentum_z*area
    @inbounds Fv_faces[face,5] = viscous_energy*area
    @inbounds for variable in 6:Ncons
        Fv_faces[face,variable] = zero(FT)
    end
    return
end

function compute_unstruct_viscous_flux!(block::UnstructBlock)
    fill!(block.Fv_faces,zero(FT))
    @static if viscous
        @gpu_launch threads=block.nthreads_face blocks=block.nb_face unstruct_viscous_flux_kernel!(
            block.Fv_faces, block.Q, block.grad,
            block.face_L, block.face_R, block.face_bc_id,
            block.face_nx, block.face_ny, block.face_nz, block.face_area,
            block.cell_cx, block.cell_cy, block.cell_cz,
            block.nface,
        )
    end
    @static if equation_type == :MHD && resistive
        @gpu_launch threads=block.nthreads_face blocks=block.nb_face unstruct_resistive_flux_kernel!(
            block.Fv_faces,block.Q,block.grad,
            block.face_L,block.face_R,block.face_bc_id,
            block.face_nx,block.face_ny,block.face_nz,block.face_area,
            block.cell_cx,block.cell_cy,block.cell_cz,
            block.nface,FT(η_mhd))
    end
    return
end
