@inline function weno7_smoothness(v::SVector{7,T}) where {T}
    V1, V2, V3, V4, V5, V6, V7 = v
    Is1 = V1*(T(547)*V1 - T(3882)*V2 + T(4642)*V3 - T(1854)*V4) +
          V2*(T(7043)*V2 - T(17246)*V3 + T(7042)*V4) +
          V3*(T(11003)*V3 - T(9402)*V4) + V4*(T(2107)*V4)
    Is2 = V2*(T(267)*V2 - T(1642)*V3 + T(1602)*V4 - T(494)*V5) +
          V3*(T(2843)*V3 - T(5966)*V4 + T(1922)*V5) +
          V4*(T(3443)*V4 - T(2522)*V5) + V5*(T(547)*V5)
    Is3 = V3*(T(547)*V3 - T(2522)*V4 + T(1922)*V5 - T(494)*V6) +
          V4*(T(3443)*V4 - T(5966)*V5 + T(1602)*V6) +
          V5*(T(2843)*V5 - T(1642)*V6) + V6*(T(267)*V6)
    Is4 = V4*(T(2107)*V4 - T(9402)*V5 + T(7042)*V6 - T(1854)*V7) +
          V5*(T(11003)*V5 - T(17246)*V6 + T(4642)*V7) +
          V6*(T(7043)*V6 - T(3882)*V7) + V7*(T(547)*V7)
    return SVector{4,T}(Is1, Is2, Is3, Is4)
end

@inline function weno7_z_from_candidates(
    candidates::SVector{4,T}, beta::SVector{4,T},
    ideal::SVector{4,T}, epsilon::T,
) where {T}
    d1 = epsilon + beta[1]
    d2 = epsilon + beta[2]
    d3 = epsilon + beta[3]
    d4 = epsilon + beta[4]
    @static if weno_z
        tau = abs(beta[1] - beta[4])
        a1 = ideal[1]*(one(T) + tau/d1)
        a2 = ideal[2]*(one(T) + tau/d2)
        a3 = ideal[3]*(one(T) + tau/d3)
        a4 = ideal[4]*(one(T) + tau/d4)
    else
        a1 = ideal[1]/(d1*d1)
        a2 = ideal[2]/(d2*d2)
        a3 = ideal[3]/(d3*d3)
        a4 = ideal[4]/(d4*d4)
    end
    raw_result = (a1*candidates[1] + a2*candidates[2] +
                  a3*candidates[3] + a4*candidates[4]) /
                 (a1 + a2 + a3 + a4)
    if isfinite(raw_result) && isfinite(candidates[1]) &&
       candidates[1] == candidates[2] && candidates[2] == candidates[3] &&
       candidates[3] == candidates[4]
        return candidates[1]
    end
    return raw_result
end

@inline function _weno7_face_from_numerators(
    candidates::SVector{4,T}, beta::SVector{4,T},
    ideal::SVector{4,T}, epsilon::T, ss::T,
) where {T}
    d1 = epsilon + beta[1]*ss
    d2 = epsilon + beta[2]*ss
    d3 = epsilon + beta[3]*ss
    d4 = epsilon + beta[4]*ss
    @static if weno_z
        tau = abs(beta[1] - beta[4])*ss
        a1 = ideal[1]*(one(T) + tau/(d1 + epsilon))/(d1*d1)
        a2 = ideal[2]*(one(T) + tau/(d2 + epsilon))/(d2*d2)
        a3 = ideal[3]*(one(T) + tau/(d3 + epsilon))/(d3*d3)
        a4 = ideal[4]*(one(T) + tau/(d4 + epsilon))/(d4*d4)
    else
        a1 = ideal[1]/(d1*d1)
        a2 = ideal[2]/(d2*d2)
        a3 = ideal[3]/(d3*d3)
        a4 = ideal[4]/(d4*d4)
    end
    invsum = one(T)/(a1 + a2 + a3 + a4)
    tmp1 = one(T)/T(12)
    return invsum*(a1*candidates[1] + a2*candidates[2] +
                   a3*candidates[3] + a4*candidates[4])*tmp1
end

@inline function weno7_face_left(v::SVector{7,T}, ss::T) where {T}
    beta = weno7_smoothness(v)
    candidates = SVector{4,T}(
        -T(3)*v[1] + T(13)*v[2] - T(23)*v[3] + T(25)*v[4],
        one(T)*v[2] - T(5)*v[3] + T(13)*v[4] + T(3)*v[5],
        -one(T)*v[3] + T(7)*v[4] + T(7)*v[5] - one(T)*v[6],
        T(3)*v[4] + T(13)*v[5] - T(5)*v[6] + one(T)*v[7],
    )
    ideal = SVector{4,T}(one(T), T(12), T(18), T(4))
    epsilon = T(1e-10)
    return _weno7_face_from_numerators(candidates, beta, ideal, epsilon, ss)
end

@inline weno7_face_right(v::SVector{7,T}, ss::T) where {T} =
    weno7_face_left(reverse(v), ss)

# === BEGIN GENERATED WENO7 POINT TABLES ===
const WENO7_POINT_FACE_CANDIDATE_COEFFS = SMatrix{4,4,Float64}(
    -0.3125, 0.0625, -0.0625, 0.3125,
    1.3125, -0.3125, 0.5625, 0.9375,
    -2.1875, 0.9375, 0.5625, -0.3125,
    2.1875, 0.3125, -0.0625, 0.0625,
)
const WENO7_POINT_FACE_OPTIMAL_COEFFS = SVector{7,Float64}(
    -0.0048828125,
    0.041015625,
    -0.1708984375,
    0.68359375,
    0.5126953125,
    -0.068359375,
    0.0068359375,
)
const WENO7_POINT_FACE_IDEAL_WEIGHTS = SVector{4,Float64}(
    0.015625,
    0.328125,
    0.546875,
    0.109375,
)
# === END GENERATED WENO7 POINT TABLES ===

@inline function _weno7_point_face_candidates(v::SVector{7,T}) where {T}
    coefficients = WENO7_POINT_FACE_CANDIDATE_COEFFS
    return SVector{4,T}(
        v[4] + T(coefficients[1,1])*(v[1]-v[4]) +
               T(coefficients[1,2])*(v[2]-v[4]) +
               T(coefficients[1,3])*(v[3]-v[4]),
        v[5] + T(coefficients[2,1])*(v[2]-v[5]) +
               T(coefficients[2,2])*(v[3]-v[5]) +
               T(coefficients[2,3])*(v[4]-v[5]),
        v[6] + T(coefficients[3,1])*(v[3]-v[6]) +
               T(coefficients[3,2])*(v[4]-v[6]) +
               T(coefficients[3,3])*(v[5]-v[6]),
        v[7] + T(coefficients[4,1])*(v[4]-v[7]) +
               T(coefficients[4,2])*(v[5]-v[7]) +
               T(coefficients[4,3])*(v[6]-v[7]),
    )
end

@inline function weno7_point_face_left(v::SVector{7,T}, ss::T) where {T}
    candidates = _weno7_point_face_candidates(v)
    beta = weno7_smoothness(v)*ss
    ideal = SVector{4,T}(
        T(WENO7_POINT_FACE_IDEAL_WEIGHTS[1]),
        T(WENO7_POINT_FACE_IDEAL_WEIGHTS[2]),
        T(WENO7_POINT_FACE_IDEAL_WEIGHTS[3]),
        T(WENO7_POINT_FACE_IDEAL_WEIGHTS[4]),
    )
    return weno7_z_from_candidates(candidates, beta, ideal, T(1.0e-10))
end

@inline weno7_point_face_right(v::SVector{7,T}, ss::T) where {T} =
    weno7_point_face_left(reverse(v), ss)

# === BEGIN GENERATED WENO7 GAUSS TABLES ===
const WENO7_GL4_X = SVector{4,Float64}(
    -0.43056815579702629,
    -0.16999052179242813,
    0.16999052179242813,
    0.43056815579702629,
)
const WENO7_GL4_W = SVector{4,Float64}(
    0.17392742256872692,
    0.32607257743127305,
    0.32607257743127305,
    0.17392742256872692,
)
const WENO7_GL4_CANDIDATE_COEFFS = (
    SMatrix{4,4,Float64}(
        0.064132012308668315, -0.058457597196618673, 0.24952094909512063, 1.9880676511838862,
        -0.31498564643129195, 0.48335133788159529, 0.98998385480340367, -1.7950880956658668,
        0.86814341173360521, 0.63923827162369173, -0.29796240109514299, 1.0565413935771011,
        0.38271022238901842, -0.064132012308668315, 0.058457597196618673, -0.24952094909512063,
    ),
    SMatrix{4,4,Float64}(
        0.043033814895381592, -0.02751305725120148, 0.071930592394643569, 1.3413647638329167,
        -0.19964831683272785, 0.18198282139944949, 1.0536423942543425, -0.58466959826804898,
        0.44018571077173907, 0.88856405074713352, -0.15308604390018751, 0.31523542682977579,
        0.7164287911656072, -0.043033814895381592, 0.02751305725120148, -0.071930592394643569,
    ),
    SMatrix{4,4,Float64}(
        -0.071930592394643569, 0.02751305725120148, -0.043033814895381592, 0.7164287911656072,
        0.31523542682977579, -0.15308604390018751, 0.88856405074713352, 0.44018571077173907,
        -0.58466959826804898, 1.0536423942543425, 0.18198282139944949, -0.19964831683272785,
        1.3413647638329167, 0.071930592394643569, -0.02751305725120148, 0.043033814895381592,
    ),
    SMatrix{4,4,Float64}(
        -0.24952094909512063, 0.058457597196618673, -0.064132012308668315, 0.38271022238901842,
        1.0565413935771011, -0.29796240109514299, 0.63923827162369173, 0.86814341173360521,
        -1.7950880956658668, 0.98998385480340367, 0.48335133788159529, -0.31498564643129195,
        1.9880676511838862, 0.24952094909512063, -0.058457597196618673, 0.064132012308668315,
    ),
)
const WENO7_GL4_OPTIMAL_COEFFS = SMatrix{7,4,Float64}(
    0.006374938953809515, -0.062621277425729663, 0.4314819583639985, 0.76085381290899856, -0.17174964107504084, 0.040434919169909579, -0.0047747108959456273,
    0.0028862021476986494, -0.026780121735797132, 0.1476124004929896, 0.96099233626495673, -0.10471853337961488, 0.022584373577623048, -0.002576657367856026,
    -0.002576657367856026, 0.022584373577623048, -0.10471853337961488, 0.96099233626495673, 0.1476124004929896, -0.026780121735797132, 0.0028862021476986494,
    -0.0047747108959456273, 0.040434919169909579, -0.17174964107504084, 0.76085381290899856, 0.4314819583639985, -0.062621277425729663, 0.006374938953809515,
)
const WENO7_GL4_IDEAL_WEIGHTS = SMatrix{4,4,Float64}(
    0.099403382559194314, 0.53561282389957543, 0.34584828242177934, 0.019135511119450919,
    0.06706823819164584, 0.48668022407120276, 0.41043009817887105, 0.035821439558280363,
    0.035821439558280363, 0.41043009817887105, 0.48668022407120276, 0.06706823819164584,
    0.019135511119450919, 0.34584828242177934, 0.53561282389957543, 0.099403382559194314,
)
const WENO7_GL4_SIGMA_PLUS = SVector{4,Float64}(
    1,
    1,
    1,
    1,
)
const WENO7_GL4_SIGMA_MINUS = SVector{4,Float64}(
    0,
    0,
    0,
    0,
)
const WENO7_GL4_IDEAL_PLUS = SMatrix{4,4,Float64}(
    0.099403382559194314, 0.53561282389957543, 0.34584828242177934, 0.019135511119450919,
    0.06706823819164584, 0.48668022407120276, 0.41043009817887105, 0.035821439558280363,
    0.035821439558280363, 0.41043009817887105, 0.48668022407120276, 0.06706823819164584,
    0.019135511119450919, 0.34584828242177934, 0.53561282389957543, 0.099403382559194314,
)
const WENO7_GL4_IDEAL_MINUS = SMatrix{4,4,Float64}(
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
)
# === END GENERATED WENO7 GAUSS TABLES ===

@inline function _weno7_z_from_signed_ideal(
    candidates::SVector{4,T}, beta::SVector{4,T},
    ideal::SVector{4,T}, epsilon::T,
) where {T}
    if ideal[1] >= zero(T) && ideal[2] >= zero(T) &&
       ideal[3] >= zero(T) && ideal[4] >= zero(T)
        return weno7_z_from_candidates(candidates, beta, ideal, epsilon)
    end

    theta = T(3)
    plus_raw = (ideal + theta*abs.(ideal))/T(2)
    minus_raw = (-ideal + theta*abs.(ideal))/T(2)
    sigma_plus = sum(plus_raw)
    sigma_minus = sum(minus_raw)
    weno_plus = weno7_z_from_candidates(
        candidates, beta, plus_raw/sigma_plus, epsilon,
    )
    weno_minus = weno7_z_from_candidates(
        candidates, beta, minus_raw/sigma_minus, epsilon,
    )
    return sigma_plus*weno_plus - sigma_minus*weno_minus
end

@inline function _weno7_gauss_candidates(
    v::SVector{7,T}, ::Val{Q},
) where {T,Q}
    coefficients = WENO7_GL4_CANDIDATE_COEFFS[Q]
    return SVector{4,T}(
        v[4] + T(coefficients[1,1])*(v[1]-v[4]) +
               T(coefficients[1,2])*(v[2]-v[4]) +
               T(coefficients[1,3])*(v[3]-v[4]),
        v[5] + T(coefficients[2,1])*(v[2]-v[5]) +
               T(coefficients[2,2])*(v[3]-v[5]) +
               T(coefficients[2,3])*(v[4]-v[5]),
        v[6] + T(coefficients[3,1])*(v[3]-v[6]) +
               T(coefficients[3,2])*(v[4]-v[6]) +
               T(coefficients[3,3])*(v[5]-v[6]),
        v[7] + T(coefficients[4,1])*(v[4]-v[7]) +
               T(coefficients[4,2])*(v[5]-v[7]) +
               T(coefficients[4,3])*(v[6]-v[7]),
    )
end

@inline function _weno7_gauss_ideal(::Type{T}, ::Val{Q}) where {T,Q}
    return SVector{4,T}(
        T(WENO7_GL4_IDEAL_WEIGHTS[1,Q]),
        T(WENO7_GL4_IDEAL_WEIGHTS[2,Q]),
        T(WENO7_GL4_IDEAL_WEIGHTS[3,Q]),
        T(WENO7_GL4_IDEAL_WEIGHTS[4,Q]),
    )
end

@inline function weno7_gauss_value(
    v::SVector{7,T}, target::Val{Q},
) where {T,Q}
    candidates = _weno7_gauss_candidates(v, target)
    beta = weno7_smoothness(v)
    ideal = _weno7_gauss_ideal(T, target)
    return _weno7_z_from_signed_ideal(candidates, beta, ideal, T(1.0e-10))
end

@inline function _weno7_linear_gauss_value(
    v::SVector{7,T}, ::Val{Q},
) where {T,Q}
    coefficients = WENO7_GL4_OPTIMAL_COEFFS
    return v[4] +
           T(coefficients[1,Q])*(v[1]-v[4]) +
           T(coefficients[2,Q])*(v[2]-v[4]) +
           T(coefficients[3,Q])*(v[3]-v[4]) +
           T(coefficients[5,Q])*(v[5]-v[4]) +
           T(coefficients[6,Q])*(v[6]-v[4]) +
           T(coefficients[7,Q])*(v[7]-v[4])
end

@inline function weno7_linear_gauss4_integral(v::SVector{7,T}) where {T}
    return T(WENO7_GL4_W[1])*_weno7_linear_gauss_value(v, Val(1)) +
           T(WENO7_GL4_W[2])*_weno7_linear_gauss_value(v, Val(2)) +
           T(WENO7_GL4_W[3])*_weno7_linear_gauss_value(v, Val(3)) +
           T(WENO7_GL4_W[4])*_weno7_linear_gauss_value(v, Val(4))
end

@inline function weno7_gauss4_integral(v::SVector{7,T}) where {T}
    return T(WENO7_GL4_W[1])*weno7_gauss_value(v, Val(1)) +
           T(WENO7_GL4_W[2])*weno7_gauss_value(v, Val(2)) +
           T(WENO7_GL4_W[3])*weno7_gauss_value(v, Val(3)) +
           T(WENO7_GL4_W[4])*weno7_gauss_value(v, Val(4))
end
