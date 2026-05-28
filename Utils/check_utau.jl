#!/usr/bin/env julia
# Quick diagnostic: compare u_tau from sublayer fit vs from du/dy at first cell
# Usage: julia Utils/check_utau.jl PLT/Ma03Ro03

using Printf, DelimitedFiles

csv_dir = ARGS[1]
case_name = basename(csv_dir)
csv_file = joinpath(csv_dir, "loglaw_profile_$case_name.csv")

println("═"^60)
println("  u_τ diagnostic — $case_name")
println("═"^60)

# Read CSV, skip comment lines
lines = filter(l -> !startswith(l, '#'), readlines(csv_file))
header = split(lines[1], ',')
println("  Columns: ", join(header, " | "))

data = zeros(length(lines)-1, length(header))
for (i, l) in enumerate(lines[2:end])
    vals = split(l, ',')
    for (j, v) in enumerate(vals)
        data[i, j] = v == "NaN" ? NaN : parse(Float64, v)
    end
end

yp = data[:, 1]
up = data[:, 2]

# Extract u_tau from header comment
u_tau_header = 0.0
for line in readlines(csv_file)
    m = match(r"u_tau=([\d.]+)", line)
    if m !== nothing
        global u_tau_header = parse(Float64, m[1])
        break
    end
end

println("\n  u_τ from header: $(round(u_tau_header, digits=4)) m/s")
println()

# Print near-wall data
println("  Near-wall profile (y⁺ < 8):")
println("  ┌──────────┬──────────┬──────────┬──────────┐")
println("  │   y⁺     │   u⁺     │  u⁺/y⁺   │  Δ(%)    │")
println("  ├──────────┼──────────┼──────────┼──────────┤")
for i in 1:size(data,1)
    yp[i] > 8 && break
    ratio = up[i] / yp[i]
    delta_pct = (ratio - 1.0) * 100
    @printf("  │ %7.3f  │ %7.4f  │  %6.4f  │ %+6.2f%%  │\n",
            yp[i], up[i], ratio, delta_pct)
end
println("  └──────────┴──────────┴──────────┴──────────┘")

# Method 1: Least squares fit u = a*y over y⁺ < 5
mask1 = (yp .> 0) .& (yp .< 5)
yw1 = yp[mask1]
uw1 = up[mask1]
a1 = sum(yw1 .* uw1) / sum(yw1.^2)
@printf("\n  Method 1: LS fit (y⁺ < 5, %d pts):  du⁺/dy⁺ = %.4f → u_τ correction = %.1f%%\n",
        sum(mask1), a1, (a1 - 1.0) * 100)

# Method 2: Fit over y⁺ < 3 only
mask2 = (yp .> 0) .& (yp .< 3)
yw2 = yp[mask2]
uw2 = up[mask2]
a2 = sum(yw2 .* uw2) / sum(yw2.^2)
@printf("  Method 2: LS fit (y⁺ < 3, %d pts):  du⁺/dy⁺ = %.4f → u_τ correction = %.1f%%\n",
        sum(mask2), a2, (a2 - 1.0) * 100)

# Method 3: First-cell gradient (y⁺ ≈ 0.3-1.0)
mask3 = (yp .> 0) .& (yp .< 1.5)
yw3 = yp[mask3]
uw3 = up[mask3]
a3 = sum(yw3 .* uw3) / sum(yw3.^2)
@printf("  Method 3: LS fit (y⁺ < 1.5, %d pts): du⁺/dy⁺ = %.4f → u_τ correction = %.1f%%\n",
        sum(mask3), a3, (a3 - 1.0) * 100)

# Method 4: slope from first two points
if sum(yp .> 0) >= 2
    idx = findall(yp .> 0)
    slope_12 = (up[idx[2]] - up[idx[1]]) / (yp[idx[2]] - yp[idx[1]])
    @printf("  Method 4: Finite diff (pt1→pt2):    du⁺/dy⁺ = %.4f → u_τ correction = %.1f%%\n",
            slope_12, (slope_12 - 1.0) * 100)
end

# If du⁺/dy⁺ from fit > 1, u_tau is overestimated by that factor
# Because u⁺ = u/u_tau, y⁺ = y*u_tau/nu
# If true u_tau' = u_tau / sqrt(a), then u'⁺ = u⁺ * sqrt(a), y'⁺ = y⁺ / sqrt(a)
# and the log-law intercept shifts by -(1/κ) * ln(sqrt(a))
println()
if a1 > 1.0
    correction_factor = sqrt(a1)
    u_tau_corrected = u_tau_header / correction_factor
    @printf("  ⚠ Sublayer slope > 1 suggests u_τ overestimated by factor %.3f\n", correction_factor)
    @printf("    Current u_τ  = %.4f m/s\n", u_tau_header)
    @printf("    If slope=1:  u_τ ≈ %.4f m/s  (%.1f%% lower)\n", 
            u_tau_corrected, (1.0 - 1.0/correction_factor) * 100)
    @printf("    Log-law shift: Δu⁺ ≈ +%.2f (would raise profile)\n",
            (1/0.41) * log(correction_factor) + correction_factor - 1)
end

println("\n", "═"^60)
