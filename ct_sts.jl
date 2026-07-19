# First-order damped Chebyshev super-time-stepping schedule.

function ct_sts_substeps(
    total_dt::T, explicit_dt::T;
    damping::T=T(0.01), max_stages::Integer=64,
) where {T<:AbstractFloat}
    total_dt > zero(T) || return T[]
    explicit_dt > zero(T) || throw(ArgumentError(
        "explicit resistive time step must be positive",
    ))
    zero(T) < damping < one(T) || throw(ArgumentError(
        "STS damping must lie in (0, 1)",
    ))
    max_stages >= 1 || throw(ArgumentError(
        "STS max_stages must be positive",
    ))
    if total_dt <= explicit_dt
        return T[total_dt]
    end

    ratio = total_dt / explicit_dt
    stage_count = 1
    weights = T[]
    while stage_count < max_stages
        stage_count += 1
        weights = T[
            inv(
                one(T) + damping -
                (one(T)-damping) * cos(
                    T(2j-1) * T(pi) / T(2stage_count),
                )
            )
            for j in 1:stage_count
        ]
        sum(weights) >= ratio && break
    end
    sum(weights) >= ratio || throw(ArgumentError(
        "STS requires more than $max_stages stages for dt ratio $ratio",
    ))
    scale = total_dt / sum(weights)
    substeps = scale .* weights
    maximum(substeps) <= explicit_dt * maximum(weights) *
        (one(T) + sqrt(eps(T))) || error("invalid STS schedule")
    return substeps
end
