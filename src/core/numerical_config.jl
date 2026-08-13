function validate_structured_run_configuration(
    equations::Symbol, split_method_id::Integer;
    implicit::Bool=false, dual_time::Bool=false,
)
    dual_time && !implicit && throw(ArgumentError(
        "dual_time=true requires implicit=true",
    ))
    valid_ids = equations == :MHD ? (1, 4, 5, 6) :
        equations == :compressible ? (1, 2, 3, 4, 5) : ()
    isempty(valid_ids) && throw(ArgumentError(
        "equations must be :compressible or :MHD, got $equations",
    ))
    Int(split_method_id) in valid_ids || throw(ArgumentError(
        "unsupported structured Riemann/split method id $split_method_id " *
        "for $equations; supported ids are $(join(valid_ids, ", "))",
    ))
    return nothing
end

function shockSensor(ϕ, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i < 2 || i > nxp+2*NG-1 || j < 2 || j > nyp+2*NG-1 || k < 2 || k > nzp+2*NG-1
        return
    end



    # MHD mode: pressure-based Ducros sensor (Q[5]=p, same as compressible)
    # Falls through to the compressible pressure sensor below

    @inbounds Px1 = Q[i-1, j, k, 5]
    @inbounds Px2 = Q[i,   j, k, 5]
    @inbounds Px3 = Q[i+1, j, k, 5]
    @inbounds Py1 = Q[i, j-1, k, 5]
    @inbounds Py2 = Q[i, j,   k, 5]
    @inbounds Py3 = Q[i, j+1, k, 5]
    @inbounds Pz1 = Q[i, j, k-1, 5]
    @inbounds Pz2 = Q[i, j, k  , 5]
    @inbounds Pz3 = Q[i, j, k+1, 5]
    ϕx = abs(-Px1 + 2*Px2 - Px3)/(Px1 + 2*Px2 + Px3 + FT(1.0e-20))
    ϕy = abs(-Py1 + 2*Py2 - Py3)/(Py1 + 2*Py2 + Py3 + FT(1.0e-20))
    ϕz = abs(-Pz1 + 2*Pz2 - Pz3)/(Pz1 + 2*Pz2 + Pz3 + FT(1.0e-20))
    @inbounds ϕ[i, j, k] = ϕx + ϕy + ϕz
    return
end

@inline function _ct_troubled_level(value, thresholds)
    if !isfinite(value) || value >= thresholds[2]
        return CT_TROUBLED_STRONG
    elseif value >= thresholds[1]
        return CT_TROUBLED_RECOVERABLE
    end
    return CT_TROUBLED_SMOOTH
end

@inline function _ct_normalized_jump(center, neighbor)
    scale = abs(center) + abs(neighbor) + FT(64)*eps(FT)
    return abs(neighbor-center)/scale
end

@inline function _ct_normalized_curvature(left, center, right)
    scale = abs(left) + FT(2)*abs(center) + abs(right) +
            FT(64)*eps(FT)
    return abs(left-FT(2)*center+right)/scale
end

@inline function _ct_cell_total_pressure(Q, i, j, k)
    @inbounds return Q[i,j,k,5] + mhd_magnetic_pressure(
        Q[i,j,k,QBX],Q[i,j,k,QBY],Q[i,j,k,QBZ],
    )
end

@inline function _ct_velocity_face_flux(
    ul,vl,wl,ur,vr,wr,area,nx,ny,nz,
)
    half = FT(0.5)
    return area*half*(
        (ul+ur)*nx + (vl+vr)*ny + (wl+wr)*nz
    )
end

function ct_troubled_cell_sensor_kernel!(
    mask,Q,inverse_volume,
    Areai,nxi,nyi,nzi,Areaj,nxj,nyj,nzj,Areak,nxk,nyk,nzk,
    nxp,nyp,nzp,
)
    i = (blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    j = (blockIdx().y-Int32(1))*blockDim().y+threadIdx().y
    k = (blockIdx().z-Int32(1))*blockDim().z+threadIdx().z
    ni,nj,nk = nxp+Int32(2)*NG,nyp+Int32(2)*NG,nzp+Int32(2)*NG
    if i > ni || j > nj || k > nk
        return
    end
    if i <= Int32(1) || i >= ni || j <= Int32(1) || j >= nj ||
       k <= Int32(1) || k >= nk
        @inbounds mask[i,j,k] = FT(CT_TROUBLED_SMOOTH)
        return
    end

    state_is_finite = true
    for component in Int32(1):Int32(Nprim)
        @inbounds state_is_finite &= isfinite(Q[i,j,k,component])
    end
    if !state_is_finite
        @inbounds mask[i,j,k] = FT(CT_TROUBLED_STRONG)
        return
    end

    @inbounds begin
        pressure = Q[i,j,k,5]
        rho = Q[i,j,k,1]
        bx,by,bz = Q[i,j,k,QBX],Q[i,j,k,QBY],Q[i,j,k,QBZ]
        u,v,w = Q[i,j,k,2],Q[i,j,k,3],Q[i,j,k,4]
        uim,vim,wim = Q[i-Int32(1),j,k,2],Q[i-Int32(1),j,k,3],Q[i-Int32(1),j,k,4]
        uip,vip,wip = Q[i+Int32(1),j,k,2],Q[i+Int32(1),j,k,3],Q[i+Int32(1),j,k,4]
        ujm,vjm,wjm = Q[i,j-Int32(1),k,2],Q[i,j-Int32(1),k,3],Q[i,j-Int32(1),k,4]
        ujp,vjp,wjp = Q[i,j+Int32(1),k,2],Q[i,j+Int32(1),k,3],Q[i,j+Int32(1),k,4]
        ukm,vkm,wkm = Q[i,j,k-Int32(1),2],Q[i,j,k-Int32(1),3],Q[i,j,k-Int32(1),4]
        ukp,vkp,wkp = Q[i,j,k+Int32(1),2],Q[i,j,k+Int32(1),3],Q[i,j,k+Int32(1),4]

        divergence = inverse_volume[i,j,k]*(
            _ct_velocity_face_flux(
                u,v,w,uip,vip,wip,Areai[i+Int32(1),j,k],
                nxi[i+Int32(1),j,k],nyi[i+Int32(1),j,k],nzi[i+Int32(1),j,k],
            ) -
            _ct_velocity_face_flux(
                uim,vim,wim,u,v,w,Areai[i,j,k],
                nxi[i,j,k],nyi[i,j,k],nzi[i,j,k],
            ) +
            _ct_velocity_face_flux(
                u,v,w,ujp,vjp,wjp,Areaj[i,j+Int32(1),k],
                nxj[i,j+Int32(1),k],nyj[i,j+Int32(1),k],nzj[i,j+Int32(1),k],
            ) -
            _ct_velocity_face_flux(
                ujm,vjm,wjm,u,v,w,Areaj[i,j,k],
                nxj[i,j,k],nyj[i,j,k],nzj[i,j,k],
            ) +
            _ct_velocity_face_flux(
                u,v,w,ukp,vkp,wkp,Areak[i,j,k+Int32(1)],
                nxk[i,j,k+Int32(1)],nyk[i,j,k+Int32(1)],nzk[i,j,k+Int32(1)],
            ) -
            _ct_velocity_face_flux(
                ukm,vkm,wkm,u,v,w,Areak[i,j,k],
                nxk[i,j,k],nyk[i,j,k],nzk[i,j,k],
            )
        )
        inverse_length = inverse_volume[i,j,k]*(
            Areai[i,j,k]+Areai[i+Int32(1),j,k]+
            Areaj[i,j,k]+Areaj[i,j+Int32(1),k]+
            Areak[i,j,k]+Areak[i,j,k+Int32(1)]
        )/FT(6)
        sound2 = max(mhd_sound_speed_squared(rho,pressure,FT(γ)),zero(FT))
        fast2 = sound2 + max(mhd_alfven_speed_squared(bx,by,bz,rho),zero(FT))
        speed = sqrt(u*u+v*v+w*w) + sqrt(fast2)
        compression = max(-divergence,zero(FT))/(
            speed*inverse_length + FT(64)*eps(FT)
        )

        pressure_jump = zero(FT)
        total_pressure_jump = zero(FT)
        total_pressure = _ct_cell_total_pressure(Q,i,j,k)
        pim,pip = Q[i-Int32(1),j,k,5],Q[i+Int32(1),j,k,5]
        pjm,pjp = Q[i,j-Int32(1),k,5],Q[i,j+Int32(1),k,5]
        pkm,pkp = Q[i,j,k-Int32(1),5],Q[i,j,k+Int32(1),5]
        pressure_jump = max(
            _ct_normalized_jump(pressure,pim),
            _ct_normalized_jump(pressure,pip),
            _ct_normalized_jump(pressure,pjm),
            _ct_normalized_jump(pressure,pjp),
            _ct_normalized_jump(pressure,pkm),
            _ct_normalized_jump(pressure,pkp),
        )
        total_pressure_jump = max(
            _ct_normalized_jump(total_pressure,_ct_cell_total_pressure(Q,i-Int32(1),j,k)),
            _ct_normalized_jump(total_pressure,_ct_cell_total_pressure(Q,i+Int32(1),j,k)),
            _ct_normalized_jump(total_pressure,_ct_cell_total_pressure(Q,i,j-Int32(1),k)),
            _ct_normalized_jump(total_pressure,_ct_cell_total_pressure(Q,i,j+Int32(1),k)),
            _ct_normalized_jump(total_pressure,_ct_cell_total_pressure(Q,i,j,k-Int32(1))),
            _ct_normalized_jump(total_pressure,_ct_cell_total_pressure(Q,i,j,k+Int32(1))),
        )
        pressure_curvature = max(
            _ct_normalized_curvature(pim,pressure,pip),
            _ct_normalized_curvature(pjm,pressure,pjp),
            _ct_normalized_curvature(pkm,pressure,pkp),
        )
        total_pressure_curvature = max(
            _ct_normalized_curvature(
                _ct_cell_total_pressure(Q,i-Int32(1),j,k),
                total_pressure,
                _ct_cell_total_pressure(Q,i+Int32(1),j,k),
            ),
            _ct_normalized_curvature(
                _ct_cell_total_pressure(Q,i,j-Int32(1),k),
                total_pressure,
                _ct_cell_total_pressure(Q,i,j+Int32(1),k),
            ),
            _ct_normalized_curvature(
                _ct_cell_total_pressure(Q,i,j,k-Int32(1)),
                total_pressure,
                _ct_cell_total_pressure(Q,i,j,k+Int32(1)),
            ),
        )
    end
    compression_level = _ct_troubled_level(
        compression,ct_troubled_compression_thresholds,
    )
    jump_level = compression_level == CT_TROUBLED_SMOOTH ?
        CT_TROUBLED_SMOOTH : max(
            _ct_troubled_level(
                pressure_jump,ct_troubled_pressure_jump_thresholds,
            ),
            _ct_troubled_level(
                total_pressure_jump,ct_troubled_total_pressure_jump_thresholds,
            ),
        )
    level = max(
        compression_level,
        jump_level,
        _ct_troubled_level(
            pressure_curvature,ct_troubled_pressure_curvature_thresholds,
        ),
        _ct_troubled_level(
            total_pressure_curvature,
            ct_troubled_total_pressure_curvature_thresholds,
        ),
    )
    @inbounds mask[i,j,k] = FT(level)
    return
end

@inline function minmod(a, b)
    ifelse(a*b > 0, (abs(a) > abs(b)) ? b : a, zero(a))
end
