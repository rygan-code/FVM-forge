# ═══════════════════════════════════════════════════════════════════════
# physics.jl — Equation system abstraction layer
# ═══════════════════════════════════════════════════════════════════════
# Defines the equation type (compressible / incompressible_AC / MHD)
# and derived constants (Ncons, Nprim). All downstream code uses these
# compile-time constants for dispatch — zero runtime overhead.
#
# Usage: set `equation_type` in your run_*.jl BEFORE including this file.
#        If not set, defaults to :compressible (existing behavior).
# ═══════════════════════════════════════════════════════════════════════

# ─── Precision control (must come first) ───
# (inlined from precision.jl)
# Defines the floating-point type `FT` used throughout the solver.
# Set `const FT = Float64` (or Float32) in your run_*.jl BEFORE including this file.
if !@isdefined(FT)
    const FT = Float64
end
@assert FT === Float32 || FT === Float64 "FT must be Float32 or Float64, got $FT"

# ─── Equation system type ───
# Must be defined as `const equation_type = :compressible` (or :incompressible_AC, :MHD)
# in the run script BEFORE including this file.
# If not defined, default to compressible:
if !@isdefined(equation_type)
    const equation_type = :compressible
end

# ─── Variable counts (compile-time constants) ───
const Ncons = if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
    4   # p, u, v, w
elseif equation_type == :MHD
    9   # ρ, ρu, ρv, ρw, ρE, Bx, By, Bz, ψ
else
    5   # ρ, ρu, ρv, ρw, ρE (compressible Euler/NS)
end

const Nprim = if equation_type == :incompressible_AC || equation_type == :incompressible_PISO
    4   # p, u, v, w (same as conservative)
elseif equation_type == :MHD
    10  # ρ, u, v, w, p, T, Bx, By, Bz, ψ
else
    6   # ρ, u, v, w, p, T (compressible)
end

# ─── AC (Artificial Compressibility) parameters ───
# These can be overridden by defining them in the run script BEFORE including physics.jl
if !@isdefined(β_AC);  const β_AC::FT  = FT(10.0);   end
if !@isdefined(ρ_ref); const ρ_ref::FT = FT(1.0);    end
if !@isdefined(ν_AC);  const ν_AC::FT  = FT(1.0e-3); end
if !@isdefined(U_lid_AC); const U_lid_AC::FT = FT(1.0); end

# ─── MHD parameters ───
# GLM divergence cleaning (Dedner et al. 2002):
#   ψ-equation: ∂ψ/∂t + ch²·∇·B = -(ch²/cp²)·ψ
#   ch = max fast magnetosonic speed (auto-computed each time step)
#   cr = ch/cp = damping ratio (user-configurable, typically 0.18)
if !@isdefined(cr_glm);  const cr_glm::FT  = FT(0.18);  end   # GLM damping ratio

# Resistive MHD control (analogous to `viscous` for hydrodynamic viscosity)
# resistive = false → ideal MHD (no magnetic diffusion)
# resistive = true  → resistive MHD (η_mhd > 0, future phase)
if !@isdefined(resistive); const resistive::Bool = false; end
if !@isdefined(η_mhd);    const η_mhd::FT  = FT(0.0); end   # Magnetic resistivity

# Vacuum permeability (normalized to 1 for ideal MHD in Gaussian units)
const μ_0::FT = FT(1.0)

# ─── MHD primitive variable index constants ───
# In Q (primitive): [ρ, u, v, w, p, T, Bx, By, Bz, ψ]
const QBX  = 7;  const QBY  = 8;  const QBZ  = 9;  const QPSI = 10
# In U (conservative): [ρ, ρu, ρv, ρw, ρE, Bx, By, Bz, ψ]
const UBX  = 6;  const UBY  = 7;  const UBZ  = 8;  const UPSI = 9
