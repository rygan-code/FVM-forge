# unstruct_reconstruct.jl — Face reconstruction and flux for unstructured FVM
# Part of the second-order unstructured FVM branch
#
# Shared infrastructure (must be included BEFORE this file):
#   gpu_backend.jl, physics.jl, Riemann_Solver.jl, unstruct_mesh.jl
#
# Kernels:
#   unstruct_flux_kernel!     — 1st-order: UL=Q[L], UR=Q[R], call HLLC_Flux
#   unstruct_flux_muscl_kernel! — 2nd-order: MUSCL reconstruction with gradient + limiter
#
# The flux is stored as F_faces[f, n] = HLLC_Flux(...) * face_area[f]
# (area included, so div kernel only does signed summation)

using StaticArrays

# ═════════════════════════════════════════════════════════════
# Self-contained HLLC flux (copied from Riemann_Solver.jl to avoid
# including the full file which may have MHD syntax issues from
# concurrent edits). This is the compressible 5-variable version.
# ═════════════════════════════════════════════════════════════
@inline function HLLC_Flux_unstruct(UL, UR, nx, ny, nz)::SVector{5, FT}
    if UL[1] < FT(1.0e-10) || UR[1] < FT(1.0e-10) || !isfinite(UL[1]) || !isfinite(UR[1])
        return SVector{5, FT}(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT))
    end

    ρL = UL[1]; inv_ρL = one(FT) / ρL
    uL = UL[2] * inv_ρL; vL = UL[3] * inv_ρL; wL = UL[4] * inv_ρL
    EL = UL[5]
    pL = max((γ - one(FT)) * (EL - FT(0.5) * ρL * (uL^2 + vL^2 + wL^2)), FT(1.0e-10))
    cL = sqrt(γ * pL * inv_ρL)
    qL = uL * nx + vL * ny + wL * nz

    ρR = UR[1]; inv_ρR = one(FT) / ρR
    uR = UR[2] * inv_ρR; vR = UR[3] * inv_ρR; wR = UR[4] * inv_ρR
    ER = UR[5]
    pR = max((γ - one(FT)) * (ER - FT(0.5) * ρR * (uR^2 + vR^2 + wR^2)), FT(1.0e-10))
    cR = sqrt(γ * pR * inv_ρR)
    qR = uR * nx + vR * ny + wR * nz

    sqρL = sqrt(ρL); sqρR = sqrt(ρR)
    inv_sqρ = one(FT) / (sqρL + sqρR)
    u_roe = (sqρL * uL + sqρR * uR) * inv_sqρ
    v_roe = (sqρL * vL + sqρR * vR) * inv_sqρ
    w_roe = (sqρL * wL + sqρR * wR) * inv_sqρ
    q_roe = u_roe * nx + v_roe * ny + w_roe * nz
    HL = (EL + pL) * inv_ρL; HR = (ER + pR) * inv_ρR
    H_roe = (sqρL * HL + sqρR * HR) * inv_sqρ
    v2_roe = u_roe^2 + v_roe^2 + w_roe^2
    c_roe = sqrt(max(FT(1.0e-10), (γ - one(FT)) * (H_roe - FT(0.5) * v2_roe)))

    SL = min(qL - cL, q_roe - c_roe)
    SR = max(qR + cR, q_roe + c_roe)

    S_star = (pR - pL + ρL * qL * (SL - qL) - ρR * qR * (SR - qR)) /
             (ρL * (SL - qL) - ρR * (SR - qR) + FT(1.0e-20))

    if SL >= 0
        return SVector{5, FT}(ρL*qL, ρL*uL*qL+pL*nx, ρL*vL*qL+pL*ny, ρL*wL*qL+pL*nz, (EL+pL)*qL)
    elseif SR <= 0
        return SVector{5, FT}(ρR*qR, ρR*uR*qR+pR*nx, ρR*vR*qR+pR*ny, ρR*wR*qR+pR*nz, (ER+pR)*qR)
    elseif SL < 0 && S_star >= 0
        FL = SVector{5, FT}(ρL*qL, ρL*uL*qL+pL*nx, ρL*vL*qL+pL*ny, ρL*wL*qL+pL*nz, (EL+pL)*qL)
        factor = ρL * (SL - qL) / (SL - S_star + FT(1.0e-20))
        Us = SVector{5, FT}(factor, factor*(uL+(S_star-qL)*nx), factor*(vL+(S_star-qL)*ny),
                           factor*(wL+(S_star-qL)*nz),
                           factor*(EL*inv_ρL+(S_star-qL)*(S_star+pL/(ρL*(SL-qL)-FT(1.0e-20)))))
        return FL + SL * (Us - UL)
    else
        FR = SVector{5, FT}(ρR*qR, ρR*uR*qR+pR*nx, ρR*vR*qR+pR*ny, ρR*wR*qR+pR*nz, (ER+pR)*qR)
        factor = ρR * (SR - qR) / (SR - S_star + FT(1.0e-20))
        Us = SVector{5, FT}(factor, factor*(uR+(S_star-qR)*nx), factor*(vR+(S_star-qR)*ny),
                           factor*(wR+(S_star-qR)*nz),
                           factor*(ER*inv_ρR+(S_star-qR)*(S_star+pR/(ρR*(SR-qR)-FT(1.0e-20)))))
        return FR + SR * (Us - UR)
    end
end

# ═════════════════════════════════════════════════════════════
# Inline helper: primitive → conservative (scalar, for single cell)
# ═════════════════════════════════════════════════════════════
@inline function prim2cons_scalar(Q_cell)::SVector{Ncons, FT}
    @static if equation_type == :MHD
        return mhd_primitive_to_conservative(
            SVector{10,FT}(
                Q_cell[1], Q_cell[2], Q_cell[3], Q_cell[4], Q_cell[5],
                Q_cell[6], Q_cell[7], Q_cell[8], Q_cell[9], Q_cell[10],
            ), FT(γ),
        )
    else
        return euler_primitive_to_conservative(
            SVector{5,FT}(Q_cell[1], Q_cell[2], Q_cell[3], Q_cell[4], Q_cell[5]),
            FT(γ),
        )
    end
end

# ═════════════════════════════════════════════════════════════
# Rusanov (Lax-Friedrichs) flux — most robust, no carbuncle
# F = 0.5*(FL + FR) - 0.5*smax*(UR - UL)
# where smax = max(|V·n| + c) over L and R
# ═════════════════════════════════════════════════════════════
@inline function Rusanov_Flux(UL, UR, nx, ny, nz, ch_glm=zero(FT))::SVector{Ncons, FT}
    @static if equation_type == :MHD
        return mhd_glm_rusanov_flux(
            SVector{9,FT}(UL), SVector{9,FT}(UR),
            FT(nx), FT(ny), FT(nz), FT(γ), FT(ch_glm),
        )
    else
        return euler_rusanov_flux(
            SVector{5,FT}(UL), SVector{5,FT}(UR),
            FT(nx), FT(ny), FT(nz), FT(γ),
        )
    end
end

# ═════════════════════════════════════════════════════════════
# 1st-order flux kernel
# ═════════════════════════════════════════════════════════════
# Each thread processes one face. UL = U[face_L], UR = U[face_R].
# Reads directly from conservative variables U (no Q→U conversion).
# Flux already includes × face_area.

function unstruct_flux_kernel!(F_faces, Q,
                                face_L, face_R,
                                face_nx, face_ny, face_nz, face_area,
                                nface, splitMethodID_::Int32, ch_glm::FT)
    f = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if f > nface
        return
    end

    @inbounds L = face_L[f]
    @inbounds R = face_R[f]

    # Read primitive states and convert to conservative
    @static if equation_type == :MHD
        @inbounds UL = prim2cons_scalar(SVector{10,FT}(
            Q[L,1], Q[L,2], Q[L,3], Q[L,4], Q[L,5],
            Q[L,6], Q[L,7], Q[L,8], Q[L,9], Q[L,10],
        ))
        @inbounds UR = prim2cons_scalar(SVector{10,FT}(
            Q[R,1], Q[R,2], Q[R,3], Q[R,4], Q[R,5],
            Q[R,6], Q[R,7], Q[R,8], Q[R,9], Q[R,10],
        ))
    else
        @inbounds UL = prim2cons_scalar(SVector{5,FT}(
            Q[L,1], Q[L,2], Q[L,3], Q[L,4], Q[L,5],
        ))
        @inbounds UR = prim2cons_scalar(SVector{5,FT}(
            Q[R,1], Q[R,2], Q[R,3], Q[R,4], Q[R,5],
        ))
    end

    # Normal
    @inbounds nx = face_nx[f]
    @inbounds ny = face_ny[f]
    @inbounds nz = face_nz[f]
    @inbounds area = face_area[f]

    # Call Riemann solver directly (1st order = pure upwind, no blending)
    # splitMethodID: 0=Rusanov (most robust), 1=HLLC, 2=SW, 3=VL, 4=Roe
    @static if equation_type == :MHD
        flux = Rusanov_Flux(UL, UR, nx, ny, nz, ch_glm)
    else
        flux = splitMethodID_ == Int32(1) ?
            HLLC_Flux_unstruct(UL, UR, nx, ny, nz) :
            Rusanov_Flux(UL, UR, nx, ny, nz)
    end

    # Store flux × area
    @inbounds for n in 1:Ncons
        F_faces[f, n] = flux[n] * area
    end
    return
end

# ═════════════════════════════════════════════════════════════
# 2nd-order MUSCL flux kernel
# ═════════════════════════════════════════════════════════════
# UL = Q[L] + 0.5 * psi_L * (grad[L] · Δr_L)
# UR = Q[R] + 0.5 * psi_R * (grad[R] · Δr_R)
# where Δr = face_center - cell_center

function unstruct_flux_muscl_kernel!(F_faces, Q, grad, limiter,
                                      face_L, face_R,
                                      face_nx, face_ny, face_nz, face_area,
                                      face_cx, face_cy, face_cz,
                                      cell_cx, cell_cy, cell_cz,
                                      nface, splitMethodID_::Int32)
    f = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if f > nface
        return
    end

    @inbounds L = face_L[f]
    @inbounds R = face_R[f]

    @inbounds nx = face_nx[f]
    @inbounds ny = face_ny[f]
    @inbounds nz = face_nz[f]
    @inbounds area = face_area[f]
    @inbounds fcx = face_cx[f]; @inbounds fcy = face_cy[f]; @inbounds fcz = face_cz[f]

    # Δr from cell center to face center
    dxL = fcx - cell_cx[L]; dyL = fcy - cell_cy[L]; dzL = fcz - cell_cz[L]
    dxR = fcx - cell_cx[R]; dyR = fcy - cell_cy[R]; dzR = fcz - cell_cz[R]

    # MUSCL reconstruction in primitive variable space (5 vars: ρ,u,v,w,p)
    # T is derived, not reconstructed
    UL_prim = SVector{5, FT}(
        Q[L, 1] + limiter[L, 1] * (grad[L, 1, 1]*dxL + grad[L, 1, 2]*dyL + grad[L, 1, 3]*dzL),
        Q[L, 2] + limiter[L, 2] * (grad[L, 2, 1]*dxL + grad[L, 2, 2]*dyL + grad[L, 2, 3]*dzL),
        Q[L, 3] + limiter[L, 3] * (grad[L, 3, 1]*dxL + grad[L, 3, 2]*dyL + grad[L, 3, 3]*dzL),
        Q[L, 4] + limiter[L, 4] * (grad[L, 4, 1]*dxL + grad[L, 4, 2]*dyL + grad[L, 4, 3]*dzL),
        Q[L, 5] + limiter[L, 5] * (grad[L, 5, 1]*dxL + grad[L, 5, 2]*dyL + grad[L, 5, 3]*dzL)
    )
    UR_prim = SVector{5, FT}(
        Q[R, 1] + limiter[R, 1] * (grad[R, 1, 1]*dxR + grad[R, 1, 2]*dyR + grad[R, 1, 3]*dzR),
        Q[R, 2] + limiter[R, 2] * (grad[R, 2, 1]*dxR + grad[R, 2, 2]*dyR + grad[R, 2, 3]*dzR),
        Q[R, 3] + limiter[R, 3] * (grad[R, 3, 1]*dxR + grad[R, 3, 2]*dyR + grad[R, 3, 3]*dzR),
        Q[R, 4] + limiter[R, 4] * (grad[R, 4, 1]*dxR + grad[R, 4, 2]*dyR + grad[R, 4, 3]*dzR),
        Q[R, 5] + limiter[R, 5] * (grad[R, 5, 1]*dxR + grad[R, 5, 2]*dyR + grad[R, 5, 3]*dzR)
    )

    UL = prim2cons_scalar(UL_prim)
    UR = prim2cons_scalar(UR_prim)

    flux = if splitMethodID_ == Int32(0)
        Rusanov_Flux(UL, UR, nx, ny, nz)
    elseif splitMethodID_ == Int32(1)
        HLLC_Flux_unstruct(UL, UR, nx, ny, nz)
    else
        Rusanov_Flux(UL, UR, nx, ny, nz)
    end

    @inbounds for n in 1:Ncons
        F_faces[f, n] = flux[n] * area
    end
    return
end

@inline function _unstruct_reconstruct_primitive(
    Q,grad,limiter,cell,variable,dx,dy,dz,second_order,
)
    @inbounds value=Q[cell,variable]
    if second_order
        @inbounds value+=limiter[cell,variable]*(
            grad[cell,variable,1]*dx+grad[cell,variable,2]*dy+
            grad[cell,variable,3]*dz)
    end
    return value
end

function unstruct_mhd_flux_kernel!(
    F_faces,Q,grad,limiter,phi_faces,
    face_L,face_R,face_nx,face_ny,face_nz,face_area,
    face_cx,face_cy,face_cz,cell_cx,cell_cy,cell_cz,
    nface::Int,ch_glm::FT,second_order::Bool,use_ct::Bool,
)
    face=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    face>nface && return
    @inbounds left=face_L[face]
    @inbounds right=face_R[face]
    @inbounds nx=face_nx[face];ny=face_ny[face];nz=face_nz[face]
    @inbounds area=face_area[face]
    @inbounds face_x=face_cx[face];face_y=face_cy[face];face_z=face_cz[face]
    @inbounds dx_left=face_x-cell_cx[left]
    @inbounds dy_left=face_y-cell_cy[left]
    @inbounds dz_left=face_z-cell_cz[left]
    @inbounds dx_right=face_x-cell_cx[right]
    @inbounds dy_right=face_y-cell_cy[right]
    @inbounds dz_right=face_z-cell_cz[right]

    density_left=max(_unstruct_reconstruct_primitive(
        Q,grad,limiter,left,1,dx_left,dy_left,dz_left,second_order),density_floor)
    velocity_left_x=_unstruct_reconstruct_primitive(
        Q,grad,limiter,left,2,dx_left,dy_left,dz_left,second_order)
    velocity_left_y=_unstruct_reconstruct_primitive(
        Q,grad,limiter,left,3,dx_left,dy_left,dz_left,second_order)
    velocity_left_z=_unstruct_reconstruct_primitive(
        Q,grad,limiter,left,4,dx_left,dy_left,dz_left,second_order)
    pressure_left=max(_unstruct_reconstruct_primitive(
        Q,grad,limiter,left,5,dx_left,dy_left,dz_left,second_order),pressure_floor)
    magnetic_left_x=_unstruct_reconstruct_primitive(
        Q,grad,limiter,left,7,dx_left,dy_left,dz_left,second_order)
    magnetic_left_y=_unstruct_reconstruct_primitive(
        Q,grad,limiter,left,8,dx_left,dy_left,dz_left,second_order)
    magnetic_left_z=_unstruct_reconstruct_primitive(
        Q,grad,limiter,left,9,dx_left,dy_left,dz_left,second_order)
    cleaning_left=_unstruct_reconstruct_primitive(
        Q,grad,limiter,left,10,dx_left,dy_left,dz_left,second_order)

    density_right=max(_unstruct_reconstruct_primitive(
        Q,grad,limiter,right,1,dx_right,dy_right,dz_right,second_order),density_floor)
    velocity_right_x=_unstruct_reconstruct_primitive(
        Q,grad,limiter,right,2,dx_right,dy_right,dz_right,second_order)
    velocity_right_y=_unstruct_reconstruct_primitive(
        Q,grad,limiter,right,3,dx_right,dy_right,dz_right,second_order)
    velocity_right_z=_unstruct_reconstruct_primitive(
        Q,grad,limiter,right,4,dx_right,dy_right,dz_right,second_order)
    pressure_right=max(_unstruct_reconstruct_primitive(
        Q,grad,limiter,right,5,dx_right,dy_right,dz_right,second_order),pressure_floor)
    magnetic_right_x=_unstruct_reconstruct_primitive(
        Q,grad,limiter,right,7,dx_right,dy_right,dz_right,second_order)
    magnetic_right_y=_unstruct_reconstruct_primitive(
        Q,grad,limiter,right,8,dx_right,dy_right,dz_right,second_order)
    magnetic_right_z=_unstruct_reconstruct_primitive(
        Q,grad,limiter,right,9,dx_right,dy_right,dz_right,second_order)
    cleaning_right=_unstruct_reconstruct_primitive(
        Q,grad,limiter,right,10,dx_right,dy_right,dz_right,second_order)

    if use_ct
        @inbounds face_normal_field=phi_faces[face]/area
        left_normal=magnetic_left_x*nx+magnetic_left_y*ny+magnetic_left_z*nz
        right_normal=magnetic_right_x*nx+magnetic_right_y*ny+magnetic_right_z*nz
        magnetic_left_x+=(face_normal_field-left_normal)*nx
        magnetic_left_y+=(face_normal_field-left_normal)*ny
        magnetic_left_z+=(face_normal_field-left_normal)*nz
        magnetic_right_x+=(face_normal_field-right_normal)*nx
        magnetic_right_y+=(face_normal_field-right_normal)*ny
        magnetic_right_z+=(face_normal_field-right_normal)*nz
        cleaning_left=zero(FT)
        cleaning_right=zero(FT)
    end
    primitive_left=SVector{10,FT}(
        density_left,velocity_left_x,velocity_left_y,velocity_left_z,pressure_left,
        pressure_left/(density_left*Rg),
        magnetic_left_x,magnetic_left_y,magnetic_left_z,cleaning_left)
    primitive_right=SVector{10,FT}(
        density_right,velocity_right_x,velocity_right_y,velocity_right_z,pressure_right,
        pressure_right/(density_right*Rg),
        magnetic_right_x,magnetic_right_y,magnetic_right_z,cleaning_right)
    conservative_left=mhd_primitive_to_conservative(primitive_left,FT(γ))
    conservative_right=mhd_primitive_to_conservative(primitive_right,FT(γ))
    flux=mhd_glm_rusanov_flux(
        conservative_left,conservative_right,nx,ny,nz,FT(γ),ch_glm)
    @inbounds for variable in 1:Ncons
        F_faces[face,variable]=flux[variable]*area
    end
    return
end

# ═════════════════════════════════════════════════════════════
# Wrapper: launch flux kernel
# ═════════════════════════════════════════════════════════════

"""Compute face fluxes for unstructured block. order=1 → 1st-order upwind, order=2 → MUSCL."""
function compute_unstruct_flux!(block::UnstructBlock, order::Int,
                                splitMethodID_::Int32, ch_glm::FT=zero(FT))
    @static if equation_type == :MHD
        splitMethodID_ == Int32(0) || throw(ArgumentError(
            "unstructured GLM-MHD currently supports Rusanov flux only"))
        @static if ct_mode
            block.ct === nothing && error("unstructured CT has not been initialized")
            phi_faces=block.ct.phi_faces
        else
            phi_faces=block.face_area
        end
        @gpu_launch threads=block.nthreads_face blocks=block.nb_face unstruct_mhd_flux_kernel!(
            block.F_faces,block.Q,block.grad,block.reconstruction_limiter,phi_faces,
            block.face_L,block.face_R,
            block.face_nx,block.face_ny,block.face_nz,block.face_area,
            block.face_cx,block.face_cy,block.face_cz,
            block.cell_cx,block.cell_cy,block.cell_cz,
            block.nface,ch_glm,order==2,ct_mode)
        return
    end
    if order == 1
        @gpu_launch threads=block.nthreads_face blocks=block.nb_face unstruct_flux_kernel!(
            block.F_faces, block.Q,
            block.face_L, block.face_R,
            block.face_nx, block.face_ny, block.face_nz, block.face_area,
            block.nface, splitMethodID_, ch_glm)
    else
        @gpu_launch threads=block.nthreads_face blocks=block.nb_face unstruct_flux_muscl_kernel!(
            block.F_faces, block.Q, block.grad, block.reconstruction_limiter,
            block.face_L, block.face_R,
            block.face_nx, block.face_ny, block.face_nz, block.face_area,
            block.face_cx, block.face_cy, block.face_cz,
            block.cell_cx, block.cell_cy, block.cell_cz,
            block.nface, splitMethodID_)
    end
    return
end
