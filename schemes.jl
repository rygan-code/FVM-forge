function shockSensor(ϕ, Q, nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i < 2 || i > nxp+2*NG-1 || j < 2 || j > nyp+2*NG-1 || k < 2 || k > nzp+2*NG-1
        return
    end

    # AC mode: compute velocity divergence |∇·u| as incompressible "shock sensor"
    # This diagnoses how well the divergence-free constraint is satisfied.
    # Uses 2nd-order central differences (sufficient for diagnostic purposes).
    if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
        @inbounds u_xp = Q[i+1, j, k, 2]; @inbounds u_xm = Q[i-1, j, k, 2]
        @inbounds v_yp = Q[i, j+1, k, 3]; @inbounds v_ym = Q[i, j-1, k, 3]
        @inbounds w_zp = Q[i, j, k+1, 4]; @inbounds w_zm = Q[i, j, k-1, 4]
        # Central difference divergence (Cartesian approx, sufficient for sensor)
        div_u = FT(0.5) * ((u_xp - u_xm) + (v_yp - v_ym) + (w_zp - w_zm))
        @inbounds ϕ[i, j, k] = abs(div_u)
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

@inline function minmod(a, b)
    ifelse(a*b > 0, (abs(a) > abs(b)) ? b : a, zero(a))
end

# ═══════════════════════════════════════════════════════════════════════
#  AC FVM-consistent divergence diagnostic
#
#  Computes |∇·u| using face-flux form with metric terms:
#    ∇·u = (1/V) × Σ_faces (u·n × Area)
#
#  This is the EXACT discrete divergence operator consistent with the
#  AC pressure equation flux F_p = β² × (u·n) × A. The Cartesian
#  approximation (∂u/∂ξ + ∂v/∂η + ∂w/∂ζ) is incorrect on curvilinear
#  grids and systematically overstates ∇·u on O-grid topologies.
# ═══════════════════════════════════════════════════════════════════════
function ac_fvm_divergence!(ϕ, Q, Vol,
                            nxi, nyi, nzi, Areai,
                            nxj, nyj, nzj, Areaj,
                            nxk, nyk, nzk, Areak,
                            nxp, nyp, nzp)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    k = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z

    if i > nxp || j > nyp || k > nzp
        return
    end

    ig = i + NG; jg = j + NG; kg = k + NG

    @inbounds begin
        # Cell-center velocity
        u_C = Q[ig, jg, kg, 2]
        v_C = Q[ig, jg, kg, 3]
        w_C = Q[ig, jg, kg, 4]

        # ── ξ-direction faces (i±½) ──
        u_R = Q[ig+1, jg, kg, 2]; v_R = Q[ig+1, jg, kg, 3]; w_R = Q[ig+1, jg, kg, 4]
        u_L = Q[ig-1, jg, kg, 2]; v_L = Q[ig-1, jg, kg, 3]; w_L = Q[ig-1, jg, kg, 4]
        # Face i+½: between ig and ig+1
        qn_ip = (FT(0.5)*(u_C+u_R)*nxi[ig+1,jg,kg] + FT(0.5)*(v_C+v_R)*nyi[ig+1,jg,kg] + FT(0.5)*(w_C+w_R)*nzi[ig+1,jg,kg]) * Areai[ig+1,jg,kg]
        # Face i-½: between ig-1 and ig
        qn_im = (FT(0.5)*(u_L+u_C)*nxi[ig,jg,kg] + FT(0.5)*(v_L+v_C)*nyi[ig,jg,kg] + FT(0.5)*(w_L+w_C)*nzi[ig,jg,kg]) * Areai[ig,jg,kg]

        # ── η-direction faces (j±½) ──
        u_T = Q[ig, jg+1, kg, 2]; v_T = Q[ig, jg+1, kg, 3]; w_T = Q[ig, jg+1, kg, 4]
        u_B = Q[ig, jg-1, kg, 2]; v_B = Q[ig, jg-1, kg, 3]; w_B = Q[ig, jg-1, kg, 4]
        qn_jp = (FT(0.5)*(u_C+u_T)*nxj[ig,jg+1,kg] + FT(0.5)*(v_C+v_T)*nyj[ig,jg+1,kg] + FT(0.5)*(w_C+w_T)*nzj[ig,jg+1,kg]) * Areaj[ig,jg+1,kg]
        qn_jm = (FT(0.5)*(u_B+u_C)*nxj[ig,jg,kg] + FT(0.5)*(v_B+v_C)*nyj[ig,jg,kg] + FT(0.5)*(w_B+w_C)*nzj[ig,jg,kg]) * Areaj[ig,jg,kg]

        # ── ζ-direction faces (k±½) ──
        u_U = Q[ig, jg, kg+1, 2]; v_U = Q[ig, jg, kg+1, 3]; w_U = Q[ig, jg, kg+1, 4]
        u_D = Q[ig, jg, kg-1, 2]; v_D = Q[ig, jg, kg-1, 3]; w_D = Q[ig, jg, kg-1, 4]
        qn_kp = (FT(0.5)*(u_C+u_U)*nxk[ig,jg,kg+1] + FT(0.5)*(v_C+v_U)*nyk[ig,jg,kg+1] + FT(0.5)*(w_C+w_U)*nzk[ig,jg,kg+1]) * Areak[ig,jg,kg+1]
        qn_km = (FT(0.5)*(u_D+u_C)*nxk[ig,jg,kg] + FT(0.5)*(v_D+v_C)*nyk[ig,jg,kg] + FT(0.5)*(w_D+w_C)*nzk[ig,jg,kg]) * Areak[ig,jg,kg]

        # FVM divergence: (face flux out - face flux in) × vol_inv
        div_u = (qn_ip - qn_im + qn_jp - qn_jm + qn_kp - qn_km) * Vol[ig, jg, kg]

        ϕ[ig, jg, kg] = abs(div_u)
    end
    return
end