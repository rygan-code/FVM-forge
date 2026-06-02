function shockSensor(ϕ, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i < 2 || i > nxp+2*NG-1 || j < 2 || j > nyp+2*NG-1 || k < 2 || k > nzp+2*NG-1
        return
    end

    # MHD and compressible modes: pressure-based Ducros sensor (Q[5]=p)
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

@inline function minmod(a, b)
    ifelse(a*b > 0, (abs(a) > abs(b)) ? b : a, zero(a))
end