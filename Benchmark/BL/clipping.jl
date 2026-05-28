function positivity_clipping(Q, U)
    i = (blockIdx().x-1i32)* blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32)* blockDim().y + threadIdx().y
    k = (blockIdx().z-1i32)* blockDim().z + threadIdx().z

    if i > Nxp+2*NG || j > Nyp+2*NG || k > Nzp+2*NG
        return
    end

    @inbounds begin
        ρ = Q[i, j, k, 1]
        p = Q[i, j, k, 5]
        
        # Min limits
        ρ_min = FT(1.0e-5)
        p_min = FT(1.0e-5)
        
        if ρ < ρ_min || p < p_min
            ρ = max(ρ, ρ_min)
            p = max(p, p_min)
            
            Q[i, j, k, 1] = ρ
            Q[i, j, k, 5] = p
            Q[i, j, k, 6] = p / (ρ * Rg)
            
            u = Q[i, j, k, 2]
            v = Q[i, j, k, 3]
            w = Q[i, j, k, 4]
            
            U[i, j, k, 1] = ρ
            U[i, j, k, 2] = ρ * u
            U[i, j, k, 3] = ρ * v
            U[i, j, k, 4] = ρ * w
            U[i, j, k, 5] = p / (γ-one(FT)) + FT(0.5) * ρ * (u*u + v*v + w*w)
        end
    end
    return
end
