# analyze_grid_stretch.jl — Compute wall-normal grid stretching ratio
# Run from project root: julia Utils/analyze_grid_stretch.jl [MESH_DIR]
using HDF5, Printf

const C_s = 1.458e-6
const T_s = 110.4
sutherland_mu(T) = C_s * T^1.5 / (T + T_s)

# ── Load Block 1 mesh (wall-normal is radial = η-direction, j-index) ──
mesh_dir = length(ARGS) >= 1 ? ARGS[1] : "MESH"
mesh_file = joinpath(mesh_dir, "mesh_b1.h5")
println("  Mesh directory: $mesh_dir")
mfid = h5open(mesh_file, "r")
Nx = read(mfid["Nx"])
Ny = read(mfid["Ny"])  # N_rad = radial cells
Nz = read(mfid["Nz"])
y_n = read(mfid["y"])  # node coords (Nx+1, Ny+1, Nz+1)
z_n = read(mfid["z"])
close(mfid)

println("═" ^ 70)
println("  Grid Stretching Analysis — Block 1 (Annular Sector)")
println("═" ^ 70)
@printf("  Nx=%d, Ny(radial)=%d, Nz=%d\n\n", Nx, Ny, Nz)

# Choose mid-axial and mid-azimuthal slices
mid_i = (Nx + 2) ÷ 2
mid_k = (Nz + 2) ÷ 2

# Compute radial position of each node
r_nodes = Float64[]
for j in 1:Ny+1
    r = sqrt(y_n[mid_i, j, mid_k]^2 + z_n[mid_i, j, mid_k]^2)
    push!(r_nodes, r)
end

# Wall is at the LAST j (η- face = block interface to center, η+ = not wall for block 1)
# Actually for block 1: j=1 faces center, j=Ny+1 faces the wall
# Let me check which end is closer to R0=0.5
R0 = 0.5
if abs(r_nodes[1] - R0) < abs(r_nodes[end] - R0)
    # j=1 is the wall → reverse
    reverse!(r_nodes)
    println("  Wall at j=1 (reversed for wall-outward ordering)")
else
    println("  Wall at j=Ny+1")
end

# Cell centers in radial direction
r_centers = 0.5 .* (r_nodes[1:end-1] .+ r_nodes[2:end])
y_wall = R0 .- r_centers   # wall distance

# Cell widths
dr = abs.(diff(r_nodes))   # Ny cell widths

# Estimate wall units from actual simulation parameters
# The simulation uses low-density gas: rho ≈ 1.14e-3 kg/m³ (NOT atmospheric!)
# See gen_butterfly_fvm.jl: ν ≈ 0.01653 m²/s
T_w = 307.0
mu_w = sutherland_mu(T_w)
# Use actual simulation density (from solver output: rho_avg ≈ 1.1353e-3 kg/m³)
rho_w = 1.14e-3   # kg/m³ — low-pressure simulation
nu_w = mu_w / rho_w
u_tau = 14.7  # estimated from latest run

println()
@printf("  Assumed:  u_τ = %.1f m/s,  T_w = %.0f K,  ν_w = %.4e m²/s\n", u_tau, T_w, nu_w)
@printf("  Wall unit:  δ_ν = ν_w/u_τ = %.4e m\n\n", nu_w / u_tau)

# ── Print stretching table ──
println("  ┌────────┬───────────────┬───────────┬──────────────┬──────────┐")
println("  │ Cell j │    Δr (m)     │   Δr⁺     │  y_wall⁺     │ Ratio    │")
println("  ├────────┼───────────────┼───────────┼──────────────┼──────────┤")

wall_end = length(dr)
for j in wall_end:-1:max(1, wall_end-40)
    dr_plus = dr[j] * u_tau / nu_w
    yw_plus = y_wall[j] * u_tau / nu_w
    if j < wall_end
        ratio = dr[j] / dr[j+1]
    else
        ratio = NaN
    end
    @printf("  │  %3d   │  %11.6e  │  %7.3f  │   %8.2f    │  %6.3f  │\n",
            j, dr[j], dr_plus, yw_plus, ratio)
end
println("  └────────┴───────────────┴───────────┴──────────────┴──────────┘")

# ── Summary statistics ──
println()
println("  ═══ Stretching Summary ═══")
ratios = [dr[j] / dr[j+1] for j in 1:length(dr)-1]

# Focus on the wall region (last 20 cells)
n_wall = min(20, length(ratios))
wall_ratios = ratios[end-n_wall+1:end]
@printf("  Wall region (last %d cells):  max ratio = %.4f,  mean = %.4f\n",
        n_wall, maximum(wall_ratios), sum(wall_ratios)/length(wall_ratios))

# DNS criterion: stretching ratio < 1.05 (ideally < 1.03)
if maximum(wall_ratios) > 1.10
    println("  ⚠ WARNING: Stretching ratio > 1.10 in wall region → may cause numerical error")
elseif maximum(wall_ratios) > 1.05
    println("  ⚠ CAUTION: Stretching ratio > 1.05 → borderline for DNS")
else
    println("  ✓ Stretching ratio < 1.05 → acceptable for DNS")
end

# First cell y+ 
@printf("  First cell Δr⁺ = %.3f  (target < 1.0)\n", dr[end] * u_tau / nu_w)
@printf("  First cell center y⁺ = %.3f\n", y_wall[end] * u_tau / nu_w)

# Log-layer resolution
for j in 1:length(y_wall)
    yw_plus = y_wall[j] * u_tau / nu_w
    if yw_plus > 30 && yw_plus < 100
        dr_plus = dr[j] * u_tau / nu_w
        @printf("  At y⁺ ≈ %.0f:  Δr⁺ = %.2f\n", yw_plus, dr_plus)
    end
end

println()
println("═" ^ 70)
