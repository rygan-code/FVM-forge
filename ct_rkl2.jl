# Second-order Runge-Kutta-Legendre super-time-stepping coefficients.

@inline function ct_rkl2_combine(
    current, previous2, base, first, rhs_dt,
    mu, nu, base_weight, gamma_ratio,
)
    return mu*current + nu*previous2 + base_weight*base + rhs_dt +
        gamma_ratio*(first-base)
end

function ct_rkl2_stage_count(
    total_dt::T, explicit_dt::T; max_stages::Integer=64,
) where {T<:AbstractFloat}
    total_dt > zero(T) || throw(ArgumentError(
        "RKL2 time step must be positive",
    ))
    explicit_dt > zero(T) || throw(ArgumentError(
        "explicit resistive time step must be positive",
    ))
    max_stages >= 2 || throw(ArgumentError(
        "RKL2 max_stages must be at least two",
    ))
    ratio = total_dt / explicit_dt
    isfinite(ratio) || throw(ArgumentError(
        "RKL2 time-step ratio must be finite",
    ))
    stage_count = max(
        2,
        ceil(Int, (sqrt(T(9) + T(16)*ratio) - one(T)) / T(2)),
    )
    stage_count <= max_stages || throw(ArgumentError(
        "RKL2 requires $stage_count stages for dt ratio $ratio, " *
        "exceeding max_stages=$max_stages",
    ))
    return stage_count
end

@inline function _ct_rkl2_b(stage::Integer, ::Type{T}) where {T<:AbstractFloat}
    stage <= 1 && return inv(T(3))
    j = T(stage)
    return (j*j + j - T(2)) / (T(2)*j*(j + one(T)))
end

function ct_rkl2_coefficients(
    stage_count::Integer, ::Type{T}=Float64,
) where {T<:AbstractFloat}
    stage_count >= 2 || throw(ArgumentError(
        "second-order RKL requires at least two stages",
    ))
    s = T(stage_count)
    w1 = T(4) / (s*s + s - T(2))
    b1 = _ct_rkl2_b(1, T)
    mu_tilde1 = b1*w1
    first_abscissa = mu_tilde1
    stages = map(2:stage_count) do stage
        j = T(stage)
        bj = _ct_rkl2_b(stage, T)
        bjm1 = _ct_rkl2_b(stage - 1, T)
        bjm2 = _ct_rkl2_b(stage - 2, T)
        ajm1 = one(T) - bjm1
        mu = (T(2)*j - one(T)) / j * bj / bjm1
        nu = -(j - one(T)) / j * bj / bjm2
        base = one(T) - mu - nu
        mu_tilde = mu*w1
        gamma_tilde = -ajm1*mu_tilde
        gamma_ratio = gamma_tilde / mu_tilde1
        abscissa = bj*w1*j*(j + one(T))/T(2)
        (
            stage=stage, mu=mu, nu=nu, base=base,
            mu_tilde=mu_tilde, gamma_ratio=gamma_ratio,
            abscissa=abscissa,
        )
    end
    return (
        stage_count=Int(stage_count), w1=w1,
        mu_tilde1=mu_tilde1, first_abscissa=first_abscissa,
        stages=stages,
    )
end
