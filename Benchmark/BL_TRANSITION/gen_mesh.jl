# gen_mesh.jl — Flat Plate Boundary Layer Transition Mesh Generator
# Generates single-block multi-block-format mesh for Flame3D
#
# Domain: x ∈ [0, Lx], y ∈ [0, Ly], z ∈ [0, Lz]
# Grid: stretched in y-direction (hyperbolic tangent), uniform in x and z
#
# Outputs in MESH/:
#   mesh_b0.h5             — coordinates (with ghost cells)
#   metrics_b0.h5          — FVM face areas, normals, cell volumes (1/Vol)
#   block_connectivity.h5  — single-block connectivity & boundary conditions
#
# BCs:
#   Face 1 (ξ-): Supersonic inflow (Blasius profile)
#   Face 2 (ξ+): NSCBC non-reflecting outflow
#   Face 3 (η-): Isothermal wall (no-slip, T = Tw)
#   Face 4 (η+): Farfield
#   Face 5 (ζ-): Periodic
#   Face 6 (ζ+): Periodic

using HDF5, LinearAlgebra

include("../../bc_types.jl")

# ─── Grid Parameters ─────────────────────────────────────
const NG = 4

# Streamwise × wall-normal × spanwise (refined for transition)
const Nx = 512
const Ny = 192
const Nz = 32

# Domain dimensions (non-dimensionalized by δ_in, inlet displacement thickness)
const Lx = 200.0   # Streamwise: long enough for transition (Re_x ~ 3.5×10⁵ → 10⁶)
const Ly = 40.0    # Wall-normal: ~40 δ_in, enough for BL + freestream
const Lz = 15.0    # Spanwise: ~15 δ_in (wide enough for 3D instability)

# Wall-normal stretching parameter (higher = more clustering near wall)
const β_stretch = 4.0

# ─── Freestream conditions (25 km altitude, standard atmosphere) ───
const Ma_inf = 6.0       # Freestream Mach number (hypersonic)
const T_inf  = 221.5     # Freestream temperature [K] (ISA at 25 km)
const γ_air  = 1.4
const Rg_air = 287.0
const Pr_air = 0.72
const c_inf  = sqrt(γ_air * Rg_air * T_inf)
const u_inf  = Ma_inf * c_inf
const p_inf  = 2549.0    # Freestream pressure [Pa] (ISA at 25 km)
const ρ_inf  = p_inf / (Rg_air * T_inf)

# Recovery temperature (adiabatic wall)
const r = sqrt(Pr_air)   # Recovery factor for laminar BL
const T_recovery = T_inf * (1.0 + r * (γ_air - 1.0) / 2.0 * Ma_inf^2)
# Cold wall: Tw = 300 K → Tw/Tr ≈ 0.19 (realistic flight vehicle surface)
const Tw = 300.0

# Reference length: inlet displacement thickness δ*_in
# Set so Re_δ* at inlet ≈ 1000 (well before transition)
const μ_inf = 1.458e-6 * T_inf^1.5 / (T_inf + 110.4)  # Sutherland
const δ_star_in = 1000.0 * μ_inf / (ρ_inf * u_inf)     # inlet δ*

# Non-reflect outflow reference length
const L_ref = Lx * δ_star_in

println("═" ^ 60)
println("  Flat Plate Transition Mesh Generator")
println("═" ^ 60)
println("  Grid:    $Nx × $Ny × $Nz (NG=$NG)")
println("  Domain:  $Lx × $Ly × $Lz (in δ*_in units)")
println("  Ma∞ = $Ma_inf,  T∞ = $T_inf K,  Tw = $(round(Tw, digits=1)) K")
println("  u∞ = $(round(u_inf, digits=2)) m/s,  ρ∞ = $(round(ρ_inf, digits=4)) kg/m³")
println("  μ∞ = $(round(μ_inf, sigdigits=4))")
println("  δ*_in = $(round(δ_star_in*1e3, digits=3)) mm")
println("  Re_x at outlet ≈ $(round(ρ_inf * u_inf * Lx * δ_star_in / μ_inf, sigdigits=3))")
println("═" ^ 60)

# ─── Coordinate Generation ───────────────────────────────

# Uniform in x, z; hyperbolic tangent stretching in y
# NOTE: Lx, Ly, Lz are in δ* units; multiply by δ_star_in to get physical [m]
x1d = collect(range(0.0, Lx * δ_star_in, length=Nx+1))
z1d = collect(range(0.0, Lz * δ_star_in, length=Nz+1))

# y stretching: more points near wall
η = collect(range(0.0, 1.0, length=Ny+1))
y1d = (Ly * δ_star_in) .* (1.0 .+ tanh.(β_stretch .* (η .- 1.0)) ./ tanh(β_stretch))

println("  y+ at wall (first cell): Δy₁ = $(round((y1d[2] - y1d[1])*1e6, digits=2)) μm = $(round((y1d[2]-y1d[1])/δ_star_in, digits=4)) δ*")
println("  y at top:                $(round(y1d[end]*1e3, digits=3)) mm = $(round(y1d[end]/δ_star_in, digits=1)) δ*")
println("  Domain physical size:     $(round(x1d[end]*1e3, digits=2)) × $(round(y1d[end]*1e3, digits=3)) × $(round(z1d[end]*1e3, digits=3)) mm")

# ─── Total sizes with ghost ──────────────────────────────
Nx_tot = Nx + 2*NG
Ny_tot = Ny + 2*NG
Nz_tot = Nz + 2*NG

Nx_nodes_tot = Nx + 2*NG + 1
Ny_nodes_tot = Ny + 2*NG + 1
Nz_nodes_tot = Nz + 2*NG + 1

# ─── Node coordinate arrays (with ghost) ─────────────────
x = zeros(Float64, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
y = zeros(Float64, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)
z = zeros(Float64, Nx_nodes_tot, Ny_nodes_tot, Nz_nodes_tot)

dx = x1d[2] - x1d[1]
dz = z1d[2] - z1d[1]

# Fill interior nodes
for k_g in 1:Nz_nodes_tot, j_g in 1:Ny_nodes_tot, i_g in 1:Nx_nodes_tot
    i = i_g - NG; j = j_g - NG; k = k_g - NG
    x[i_g, j_g, k_g] = (i - 1) * dx
    z[i_g, j_g, k_g] = (k - 1) * dz
    if j >= 1 && j <= Ny + 1
        y[i_g, j_g, k_g] = y1d[j]
    elseif j < 1
        dy0 = y1d[2] - y1d[1]
        y[i_g, j_g, k_g] = y1d[1] + (j - 1) * dy0
    else
        dyn = y1d[end] - y1d[end-1]
        y[i_g, j_g, k_g] = y1d[end] + (j - (Ny + 1)) * dyn
    end
end

# ─── Compute FVM Metrics ─────────────────────────────────
println("  Computing FVM metrics...")

Ai  = zeros(FT, Nx_nodes_tot, Ny_tot, Nz_tot)
nxi = zeros(FT, Nx_nodes_tot, Ny_tot, Nz_tot)
nyi = zeros(FT, Nx_nodes_tot, Ny_tot, Nz_tot)
nzi = zeros(FT, Nx_nodes_tot, Ny_tot, Nz_tot)

Aj  = zeros(FT, Nx_tot, Ny_nodes_tot, Nz_tot)
nxj = zeros(FT, Nx_tot, Ny_nodes_tot, Nz_tot)
nyj = zeros(FT, Nx_tot, Ny_nodes_tot, Nz_tot)
nzj = zeros(FT, Nx_tot, Ny_nodes_tot, Nz_tot)

Ak  = zeros(FT, Nx_tot, Ny_tot, Nz_nodes_tot)
nxk = zeros(FT, Nx_tot, Ny_tot, Nz_nodes_tot)
nyk = zeros(FT, Nx_tot, Ny_tot, Nz_nodes_tot)
nzk = zeros(FT, Nx_tot, Ny_tot, Nz_nodes_tot)

Vol = zeros(FT, Nx_tot, Ny_tot, Nz_tot)

# i-face normals (Cartesian grid: normal = (1,0,0))
for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_nodes_tot
    v1 = [x[i,j+1,k+1]-x[i,j,k], y[i,j+1,k+1]-y[i,j,k], z[i,j+1,k+1]-z[i,j,k]]
    v2 = [x[i,j,k+1]-x[i,j+1,k], y[i,j,k+1]-y[i,j+1,k], z[i,j,k+1]-z[i,j+1,k]]
    n = 0.5 * cross(v1, v2)
    Ai[i,j,k] = FT(norm(n)) + FT(1e-20)
    nxi[i,j,k] = FT(n[1]) / Ai[i,j,k]
    nyi[i,j,k] = FT(n[2]) / Ai[i,j,k]
    nzi[i,j,k] = FT(n[3]) / Ai[i,j,k]
end

# j-face normals (negated per convention)
for k in 1:Nz_tot, j in 1:Ny_nodes_tot, i in 1:Nx_tot
    v1 = [x[i+1,j,k+1]-x[i,j,k], y[i+1,j,k+1]-y[i,j,k], z[i+1,j,k+1]-z[i,j,k]]
    v2 = [x[i,j,k+1]-x[i+1,j,k], y[i,j,k+1]-y[i+1,j,k], z[i,j,k+1]-z[i+1,j,k]]
    n = 0.5 * cross(v1, v2)
    Aj[i,j,k] = FT(norm(n)) + FT(1e-20)
    nxj[i,j,k] = -FT(n[1]) / Aj[i,j,k]
    nyj[i,j,k] = -FT(n[2]) / Aj[i,j,k]
    nzj[i,j,k] = -FT(n[3]) / Aj[i,j,k]
end

# k-face normals
for k in 1:Nz_nodes_tot, j in 1:Ny_tot, i in 1:Nx_tot
    v1 = [x[i+1,j+1,k]-x[i,j,k], y[i+1,j+1,k]-y[i,j,k], z[i+1,j+1,k]-z[i,j,k]]
    v2 = [x[i,j+1,k]-x[i+1,j,k], y[i,j+1,k]-y[i+1,j,k], z[i,j+1,k]-z[i+1,j,k]]
    n = 0.5 * cross(v1, v2)
    Ak[i,j,k] = FT(norm(n)) + FT(1e-20)
    nxk[i,j,k] = FT(n[1]) / Ak[i,j,k]
    nyk[i,j,k] = FT(n[2]) / Ak[i,j,k]
    nzk[i,j,k] = FT(n[3]) / Ak[i,j,k]
end

# Cell volumes (store as 1/Vol)
for k in 1:Nz_tot, j in 1:Ny_tot, i in 1:Nx_tot
    dxdξ = 0.25*(x[i+1,j,k]+x[i+1,j+1,k]+x[i+1,j,k+1]+x[i+1,j+1,k+1]-x[i,j,k]-x[i,j+1,k]-x[i,j,k+1]-x[i,j+1,k+1])
    dydξ = 0.25*(y[i+1,j,k]+y[i+1,j+1,k]+y[i+1,j,k+1]+y[i+1,j+1,k+1]-y[i,j,k]-y[i,j+1,k]-y[i,j,k+1]-y[i,j+1,k+1])
    dzdξ = 0.25*(z[i+1,j,k]+z[i+1,j+1,k]+z[i+1,j,k+1]+z[i+1,j+1,k+1]-z[i,j,k]-z[i,j+1,k]-z[i,j,k+1]-z[i,j+1,k+1])
    dxdη = 0.25*(x[i,j+1,k]+x[i+1,j+1,k]+x[i,j+1,k+1]+x[i+1,j+1,k+1]-x[i,j,k]-x[i+1,j,k]-x[i,j,k+1]-x[i+1,j,k+1])
    dydη = 0.25*(y[i,j+1,k]+y[i+1,j+1,k]+y[i,j+1,k+1]+y[i+1,j+1,k+1]-y[i,j,k]-y[i+1,j,k]-y[i,j,k+1]-y[i+1,j,k+1])
    dzdη = 0.25*(z[i,j+1,k]+z[i+1,j+1,k]+z[i,j+1,k+1]+z[i+1,j+1,k+1]-z[i,j,k]-z[i+1,j,k]-z[i,j,k+1]-z[i+1,j,k+1])
    dxdζ = 0.25*(x[i,j,k+1]+x[i+1,j,k+1]+x[i,j+1,k+1]+x[i+1,j+1,k+1]-x[i,j,k]-x[i+1,j,k]-x[i,j+1,k]-x[i+1,j+1,k])
    dydζ = 0.25*(y[i,j,k+1]+y[i+1,j,k+1]+y[i,j+1,k+1]+y[i+1,j+1,k+1]-y[i,j,k]-y[i+1,j,k]-y[i,j+1,k]-y[i+1,j+1,k])
    dzdζ = 0.25*(z[i,j,k+1]+z[i+1,j,k+1]+z[i,j+1,k+1]+z[i+1,j+1,k+1]-z[i,j,k]-z[i+1,j,k]-z[i,j+1,k]-z[i+1,j+1,k])
    vol = abs(dxdξ*(dydη*dzdζ - dydζ*dzdη) - dxdη*(dydξ*dzdζ - dydζ*dzdξ) + dxdζ*(dydξ*dzdη - dydη*dzdξ))
    Vol[i,j,k] = one(FT) / (FT(vol) + FT(1e-30))
end

# ─── Write Files ─────────────────────────────────────────
mesh_dir = joinpath(@__DIR__, "..", "..", "MESH")
mkpath(mesh_dir)

println("  Writing mesh_b0.h5...")
# IMPORTANT: coords must contain ONLY interior nodes (Nx+1, Ny+1, Nz+1).
# The solver generates ghost cell coordinates at runtime via expand_coords_with_ghost().
# Writing ghost-extended arrays here causes NG-cell physical coordinate offset!
x_real = x[1+NG:Nx+1+NG, 1+NG:Ny+1+NG, 1+NG:Nz+1+NG]
y_real = y[1+NG:Nx+1+NG, 1+NG:Ny+1+NG, 1+NG:Nz+1+NG]
z_real = z[1+NG:Nx+1+NG, 1+NG:Ny+1+NG, 1+NG:Nz+1+NG]
coords_real = Float32.(cat(reshape(x_real, (1, size(x_real)...)),
                           reshape(y_real, (1, size(y_real)...)),
                           reshape(z_real, (1, size(z_real)...)), dims=1))
h5open(joinpath(mesh_dir, "mesh_b0.h5"), "w") do f
    f["NG"] = NG
    f["Nx"] = Int64(Nx)
    f["Ny"] = Int64(Ny)
    f["Nz"] = Int64(Nz)
    f["coords"] = coords_real
    # Separate x/y/z datasets for XDMF X_Y_Z geometry (ParaView compatible)
    f["x"] = Float32.(x_real)
    f["y"] = Float32.(y_real)
    f["z"] = Float32.(z_real)
end

println("  Writing metrics_b0.h5...")
h5open(joinpath(mesh_dir, "metrics_b0.h5"), "w") do f
    f["Areai"] = Ai; f["nxi"] = nxi; f["nyi"] = nyi; f["nzi"] = nzi
    f["Areaj"] = Aj; f["nxj"] = nxj; f["nyj"] = nyj; f["nzj"] = nzj
    f["Areak"] = Ak; f["nxk"] = nxk; f["nyk"] = nyk; f["nzk"] = nzk
    f["Vol"] = Vol
end

println("  Writing block_connectivity.h5...")
# Single block: no inter-block connectivity
face_bc = zeros(Int64, 1, 6)
face_bc[1, 1] = BC_ZERO_GRADIENT      # ξ-: zero-gradient (preserves BL profile from init)
face_bc[1, 2] = BC_NSCBC_OUTFLOW      # ξ+: non-reflecting outflow
face_bc[1, 3] = BC_ISOTHERMAL_WALL    # η-: no-slip isothermal wall
face_bc[1, 4] = BC_FARFIELD           # η+: farfield
face_bc[1, 5] = BC_PERIODIC           # ζ-: periodic
face_bc[1, 6] = BC_PERIODIC           # ζ+: periodic

bc_params = zeros(FT, 1, 6, N_BC_PARAMS)
# Inflow parameters (supersonic inflow: set all freestream values)
bc_params[1, 1, BCP_RHO_INF] = FT(ρ_inf)
bc_params[1, 1, BCP_U_INF]   = FT(u_inf)
bc_params[1, 1, BCP_V_INF]   = zero(FT)
bc_params[1, 1, BCP_W_INF]   = zero(FT)
bc_params[1, 1, BCP_P_INF]   = FT(p_inf)

# NSCBC outflow parameters
bc_params[1, 2, BCP_P_TARGET] = FT(p_inf)
bc_params[1, 2, BCP_SIGMA]    = FT(0.25)
bc_params[1, 2, BCP_LREF]     = FT(L_ref)

# Wall temperature
bc_params[1, 3, BCP_TW] = FT(Tw)

# Farfield parameters
bc_params[1, 4, BCP_RHO_INF] = FT(ρ_inf)
bc_params[1, 4, BCP_U_INF]   = FT(u_inf)
bc_params[1, 4, BCP_V_INF]   = zero(FT)
bc_params[1, 4, BCP_W_INF]   = zero(FT)
bc_params[1, 4, BCP_P_INF]   = FT(p_inf)

connectivity = zeros(Int64, 0, 4)  # No inter-block connections
reverse_tan = zeros(Int64, 0)

h5open(joinpath(mesh_dir, "block_connectivity.h5"), "w") do f
    f["Nblocks"]      = 1
    f["Nx_b"]         = Int64[Nx]
    f["Ny_b"]         = Int64[Ny]
    f["Nz_b"]         = Int64[Nz]
    f["connectivity"] = connectivity
    f["reverse_tan"]  = reverse_tan
    f["face_bc"]      = face_bc
    f["bc_params"]    = bc_params
end

println()
println("═" ^ 60)
println("  Mesh generation complete!")
println("  Files: mesh_b0.h5, metrics_b0.h5, block_connectivity.h5")
println("═" ^ 60)
