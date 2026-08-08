# Explicit policy for non-positive thermodynamic states.

if !@isdefined(positivity_mode)
    const positivity_mode::Symbol = :repair
end
positivity_mode in (:strict, :repair) || throw(ArgumentError(
    "positivity_mode must be :strict or :repair, got $positivity_mode",
))

if !@isdefined(density_floor)
    const density_floor::FT = FT(1.0e-5)
end
if !@isdefined(pressure_floor)
    const pressure_floor::FT = FT(1.0e-5)
end
density_floor > zero(FT) || throw(ArgumentError("density_floor must be positive"))
pressure_floor > zero(FT) || throw(ArgumentError("pressure_floor must be positive"))

const POSITIVITY_STRICT::Bool = positivity_mode == :strict
const POSITIVITY_POLICY_LOADED::Bool = true

@inline positivity_is_invalid(density, pressure) =
    !(isfinite(density) && isfinite(pressure) &&
      density >= density_floor && pressure >= pressure_floor)

@inline positivity_floor(value, lower_bound) =
    isfinite(value) ? max(value, lower_bound) : lower_bound

function reset_positivity_diagnostics!(repair_count, conservation_delta)
    fill!(repair_count, Int32(0))
    fill!(conservation_delta, zero(FT))
    return
end

function positivity_report_or_throw!(repair_count, conservation_delta;
                                     context::AbstractString)
    count = Int(only(Array(repair_count)))
    count == 0 && return (count=0, conservation_delta=FT[])
    delta = Array(conservation_delta)
    message = "positivity violation in $context: cells=$count, " *
        "volume_integrated_conservative_delta=$(collect(delta))"
    POSITIVITY_STRICT ? error(message) : @warn(message)
    return (count=count, conservation_delta=delta)
end
