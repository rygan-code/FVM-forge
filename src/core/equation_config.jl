# ═══════════════════════════════════════════════════════════════════════
# physics.jl — Equation system abstraction layer
# ═══════════════════════════════════════════════════════════════════════
# Defines the equation type (compressible / MHD)
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

# MHD-only SI units. The compressible branch remains unchanged.
Base.include(@__MODULE__, joinpath(@__DIR__, "mhd_units.jl"))

# ─── Equation system type ───
# Must be defined as `const equation_type = :compressible` (or :MHD)
# in the run script BEFORE including this file.
# If not defined, default to compressible:
if !@isdefined(equation_type)
    const equation_type = :compressible
end

# ─── CT mode (must be before Ncons definition) ───
# ct_mode = true: B stored on face centers only, U has 5 variables (no B, no ψ)
# ct_mode = false: GLM with 9 variables (B + ψ in U)
if !@isdefined(ct_mode); const ct_mode::Bool = false; end
if !@isdefined(isothermal_mhd)
    const isothermal_mhd::Bool = false
end
if !@isdefined(isothermal_temperature)
    const isothermal_temperature::FT = one(FT)
end
if isothermal_mhd && equation_type != :MHD
    error("isothermal_mhd is only available for equation_type=:MHD")
end
if !@isdefined(strict_ct_positivity)
    const strict_ct_positivity::Bool = false
end
if !@isdefined(ct_initial_projection)
    const ct_initial_projection::Bool = true
end
if !@isdefined(ct_initial_divb_tolerance)
    const ct_initial_divb_tolerance::FT =
        FT === Float32 ? FT(5.0e-5) : FT(1.0e-11)
end

# Initial-state storage contract.  The legacy point-value path remains the
# default.  The quadrature modes are opt-in and provide direct conservative
# cell-average initializers for explicitly supported analytic cases.
const INITIAL_STATE_LEGACY_POINT = :legacy_point
const INITIAL_STATE_QUADRATURE6 = :quadrature6
const INITIAL_STATE_DIRECT_CELL_AVERAGE = :direct_cell_average
if !@isdefined(initial_state_mode)
    const initial_state_mode::Symbol = Symbol(lowercase(get(
        ENV, "OPENCFD_INITIAL_STATE_MODE", "legacy_point",
    )))
end
initial_state_mode in (
    INITIAL_STATE_LEGACY_POINT,
    INITIAL_STATE_QUADRATURE6,
    INITIAL_STATE_DIRECT_CELL_AVERAGE,
) || error(
    "Unknown initial_state_mode=$initial_state_mode; expected " *
    ":legacy_point, :quadrature6, or :direct_cell_average",
)

const CT_EMF_SG07 = Int32(2)
const CT_EMF_WENO7_SG07 = Int32(7)
if !@isdefined(ct_emf_scheme)
    const ct_emf_scheme::Int32 = CT_EMF_SG07
end
ct_emf_scheme in (CT_EMF_SG07, CT_EMF_WENO7_SG07) ||
    error("Unknown ct_emf_scheme=$ct_emf_scheme")

# WENO7 normally keeps its edge integral in smooth regions and overwrites only
# shock- or FOFC-adjacent edges with SG07.  Disabling the selector is useful for
# strict A/B tests while leaving every face and cell reconstruction unchanged.
if !@isdefined(ct_weno7_sg07_selective_only)
    const ct_weno7_sg07_selective_only::Bool = true
end

const CT_CHARACTERISTIC_PLM = Int32(2)
const CT_CHARACTERISTIC_WENO7 = Int32(7)
if !@isdefined(ct_characteristic_reconstruction)
    # Keep PLM with SG07; selecting the WENO7 CT scheme upgrades both face
    # states and edge EMFs unless a run config explicitly overrides this.
    const ct_characteristic_reconstruction::Int32 =
        ct_emf_scheme == CT_EMF_WENO7_SG07 ?
        CT_CHARACTERISTIC_WENO7 : CT_CHARACTERISTIC_PLM
end
ct_characteristic_reconstruction in (
    CT_CHARACTERISTIC_PLM, CT_CHARACTERISTIC_WENO7,
) || error(
    "Unknown ct_characteristic_reconstruction=" *
    "$ct_characteristic_reconstruction",
)

const CT_CELL_B_LSQ2 = Int32(2)
const CT_CELL_B_POINT6 = Int32(6)
if !@isdefined(ct_cell_b_recovery)
    const ct_cell_b_recovery::Int32 =
        ct_characteristic_reconstruction == CT_CHARACTERISTIC_WENO7 ?
        CT_CELL_B_POINT6 : CT_CELL_B_LSQ2
end
ct_cell_b_recovery in (CT_CELL_B_LSQ2, CT_CELL_B_POINT6) ||
    error("Unknown ct_cell_b_recovery=$ct_cell_b_recovery")

const CT_PRIMITIVE_DIRECT = Int32(2)
const CT_PRIMITIVE_POINT6 = Int32(6)
if !@isdefined(ct_primitive_recovery)
    const ct_primitive_recovery::Int32 =
        ct_characteristic_reconstruction == CT_CHARACTERISTIC_WENO7 ?
        CT_PRIMITIVE_POINT6 : CT_PRIMITIVE_DIRECT
end
ct_primitive_recovery in (CT_PRIMITIVE_DIRECT, CT_PRIMITIVE_POINT6) ||
    error("Unknown ct_primitive_recovery=$ct_primitive_recovery")

# POINT6 is exact and sixth-order in smooth regions, but its fixed negative
# deconvolution weights are not positivity preserving across discontinuities.
# The adaptive path keeps the exact POINT6 result in smooth cells, switches to
# a scale-free WENO-AO(5,3) recovery in troubled cells, and finally applies a
# local convex admissibility limit if the nonlinear candidate is still invalid.
if !@isdefined(ct_point6_adaptive_recovery)
    const ct_point6_adaptive_recovery::Bool = true
end
if !@isdefined(ct_point6_smoothness_threshold)
    const ct_point6_smoothness_threshold::FT = FT(5.0e-1)
end
if !@isdefined(ct_point6_ao_high_weight)
    const ct_point6_ao_high_weight::FT = FT(0.85)
end
if !@isdefined(ct_point6_limiter_iterations)
    const ct_point6_limiter_iterations::Int32 = Int32(16)
end
if !@isdefined(ct_point6_homogeneous_axes)
    const ct_point6_homogeneous_axes::NTuple{3,Bool} = (false,false,false)
end
if !@isdefined(ct_first_order_flux_correction)
    const ct_first_order_flux_correction::Bool =
        equation_type == :MHD && ct_mode && strict_ct_positivity
end
if !@isdefined(ct_fofc_max_iterations)
    # Maximum corrected-candidate closure passes after the initial high-order
    # probe. A Jacobi pass propagates a new troubled-cell coefficient by one
    # face/edge adjacency layer; the task graph skips the unused tail once the
    # global corrected candidate is admissible.
    const ct_fofc_max_iterations::Int32 = Int32(12)
end
if !@isdefined(ct_fofc_theta_safety)
    # Keep the accepted state strictly inside the admissible set instead of
    # landing on the density/pressure floor returned by the bisection.
    const ct_fofc_theta_safety::FT = FT(0.99)
end
isfinite(ct_point6_smoothness_threshold) &&
    ct_point6_smoothness_threshold >= zero(FT) || error(
        "ct_point6_smoothness_threshold must be finite and non-negative",
    )
isfinite(ct_point6_ao_high_weight) &&
    zero(FT) < ct_point6_ao_high_weight < one(FT) || error(
        "ct_point6_ao_high_weight must lie strictly between zero and one",
    )
ct_point6_limiter_iterations > Int32(0) || error(
    "ct_point6_limiter_iterations must be positive",
)
ct_point6_homogeneous_axes isa NTuple{3,Bool} || error(
    "ct_point6_homogeneous_axes must be an NTuple{3,Bool}",
)
ct_first_order_flux_correction && !(equation_type == :MHD && ct_mode) &&
    error("ct_first_order_flux_correction requires the CT-MHD branch")
ct_fofc_max_iterations > Int32(0) || error(
    "ct_fofc_max_iterations must be positive",
)
isfinite(ct_fofc_theta_safety) &&
    zero(FT) < ct_fofc_theta_safety < one(FT) || error(
        "ct_fofc_theta_safety must lie strictly between zero and one",
    )

# Face quadrature used by the finite-volume divergence.  A point Riemann
# solve is not itself a face average in more than one dimension.  POINT6
# converts the point flux back to a face average in the two tangential
# directions.  High-order compressible, GLM, and CT-WENO7 configurations use
# POINT6.  CT-SG07 keeps midpoint fluxes because its edge-EMF construction
# consumes that representation.
const STRUCTURED_FACE_MIDPOINT = Int32(2)
const STRUCTURED_FACE_POINT6 = Int32(6)
if !@isdefined(structured_face_quadrature)
    const structured_face_quadrature::Int32 =
        equation_type == :MHD && ct_mode &&
        ct_characteristic_reconstruction != CT_CHARACTERISTIC_WENO7 ?
        STRUCTURED_FACE_MIDPOINT : STRUCTURED_FACE_POINT6
end
structured_face_quadrature in (
    STRUCTURED_FACE_MIDPOINT, STRUCTURED_FACE_POINT6,
) || error("Unknown structured_face_quadrature=$structured_face_quadrature")
if structured_face_quadrature == STRUCTURED_FACE_POINT6 &&
   equation_type == :MHD && ct_mode &&
   ct_characteristic_reconstruction != CT_CHARACTERISTIC_WENO7
    error(
        "CT STRUCTURED_FACE_POINT6 requires WENO7 characteristic " *
        "reconstruction; CT-SG07 must use STRUCTURED_FACE_MIDPOINT",
    )
end

# CT point geometry and the Riemann-solver face-normal field are independent
# of converting the output point flux back to a face average.  WENO7 uses the
# point metric even when output quadrature is temporarily changed for an A/B
# diagnostic; SG07 retains its midpoint geometry.
const CT_FACE_GEOMETRY_MIDPOINT = Int32(2)
const CT_FACE_GEOMETRY_POINT6 = Int32(6)
if !@isdefined(ct_face_geometry)
    const ct_face_geometry::Int32 =
        equation_type == :MHD && ct_mode &&
        ct_characteristic_reconstruction == CT_CHARACTERISTIC_WENO7 ?
        CT_FACE_GEOMETRY_POINT6 : CT_FACE_GEOMETRY_MIDPOINT
end
ct_face_geometry in (
    CT_FACE_GEOMETRY_MIDPOINT, CT_FACE_GEOMETRY_POINT6,
) || error("Unknown ct_face_geometry=$ct_face_geometry")
if equation_type == :MHD && ct_mode &&
   ct_face_geometry == CT_FACE_GEOMETRY_POINT6 &&
   ct_characteristic_reconstruction != CT_CHARACTERISTIC_WENO7
    error("CT_FACE_GEOMETRY_POINT6 requires WENO7 characteristic reconstruction")
end

const CT_FACE_BN_AVERAGE = Int32(2)
const CT_FACE_BN_POINT6 = Int32(6)
const CT_FACE_BN_ADAPTIVE = Int32(7)
if !@isdefined(ct_face_bn_recovery)
    const ct_face_bn_recovery::Int32 =
        equation_type == :MHD && ct_mode &&
        ct_face_geometry == CT_FACE_GEOMETRY_POINT6 ?
        CT_FACE_BN_ADAPTIVE : CT_FACE_BN_AVERAGE
end
ct_face_bn_recovery in (
    CT_FACE_BN_AVERAGE, CT_FACE_BN_POINT6, CT_FACE_BN_ADAPTIVE,
) || error("Unknown ct_face_bn_recovery=$ct_face_bn_recovery")
if equation_type == :MHD && ct_mode &&
   ct_face_bn_recovery != CT_FACE_BN_AVERAGE &&
   ct_face_geometry != CT_FACE_GEOMETRY_POINT6
    error("high-order CT face-B recovery requires CT_FACE_GEOMETRY_POINT6")
end
if !@isdefined(ct_face_bn_smoothness_threshold)
    const ct_face_bn_smoothness_threshold::FT = FT(5.0e-1)
end
if !@isdefined(ct_face_bn_ao_high_weight)
    const ct_face_bn_ao_high_weight::FT = FT(0.85)
end
isfinite(ct_face_bn_smoothness_threshold) &&
    ct_face_bn_smoothness_threshold >= zero(FT) || error(
        "ct_face_bn_smoothness_threshold must be finite and non-negative",
    )
isfinite(ct_face_bn_ao_high_weight) &&
    zero(FT) < ct_face_bn_ao_high_weight < one(FT) || error(
        "ct_face_bn_ao_high_weight must lie strictly between zero and one",
    )

# The fixed sixth-order point-to-face-average stencil has negative outer
# weights.  CT-WENO7 therefore switches to nonlinear face integration only
# when its own tangential flux stencil is non-smooth.  Compressible and GLM
# keep their established POINT6 behavior until they are validated separately.
if !@isdefined(ct_face_p2a_adaptive)
    const ct_face_p2a_adaptive::Bool =
        equation_type == :MHD && ct_mode &&
        ct_characteristic_reconstruction == CT_CHARACTERISTIC_WENO7 &&
        structured_face_quadrature == STRUCTURED_FACE_POINT6
end
if !@isdefined(ct_face_p2a_smoothness_threshold)
    const ct_face_p2a_smoothness_threshold::FT = FT(5.0e-1)
end
if !@isdefined(ct_face_p2a_ao_high_weight)
    const ct_face_p2a_ao_high_weight::FT = FT(0.85)
end
if !@isdefined(ct_face_p2a_ao_gate_fraction)
    const ct_face_p2a_ao_gate_fraction::FT = FT(0.8)
end
if !@isdefined(CT_POS_FACE_P2A_TO_AO_COUNT)
    const CT_POS_FACE_P2A_TO_AO_COUNT = 13
    const CT_POS_FACE_P2A_TO_MIDPOINT_COUNT = 14
end
if ct_face_p2a_adaptive
    equation_type == :MHD && ct_mode &&
        ct_characteristic_reconstruction == CT_CHARACTERISTIC_WENO7 || error(
            "ct_face_p2a_adaptive is currently supported only by CT-WENO7",
        )
    structured_face_quadrature == STRUCTURED_FACE_POINT6 || error(
        "ct_face_p2a_adaptive requires STRUCTURED_FACE_POINT6",
    )
end
isfinite(ct_face_p2a_smoothness_threshold) &&
    ct_face_p2a_smoothness_threshold >= zero(FT) || error(
        "ct_face_p2a_smoothness_threshold must be finite and non-negative",
    )
isfinite(ct_face_p2a_ao_high_weight) &&
    zero(FT) < ct_face_p2a_ao_high_weight < one(FT) || error(
        "ct_face_p2a_ao_high_weight must lie strictly between zero and one",
    )
isfinite(ct_face_p2a_ao_gate_fraction) &&
    zero(FT) < ct_face_p2a_ao_gate_fraction < one(FT) || error(
        "ct_face_p2a_ao_gate_fraction must lie strictly between zero and one",
    )

# Number of stored point-flux layers on each tangential side.  SG07 needs one
# layer; sixth-order face quadrature needs two.  This same offset is consumed
# by inviscid, viscous/resistive, divergence, and diagnostics kernels.
const STRUCTURED_FLUX_TANGENTIAL_HALO::Int32 =
    structured_face_quadrature == STRUCTURED_FACE_POINT6 ? Int32(2) :
    (ct_mode ? Int32(1) : Int32(0))

# ─── Variable counts (compile-time constants) ───
# Ncons always = 9 for MHD (B in U for reconstruction/Riemann compatibility)
# CT mode: div only updates 1:5, B updated by CT from face B
# GLM mode: div updates all 9
const Ncons = if equation_type == :MHD
    9   # ρ, ρu, ρv, ρw, ρE, Bx, By, Bz, ψ
else
    5   # ρ, ρu, ρv, ρw, ρE (compressible Euler/NS)
end

const Nprim = if equation_type == :MHD
    10  # ρ, u, v, w, p, T, Bx, By, Bz, ψ
else
    6   # ρ, u, v, w, p, T (compressible)
end

# CT mode: number of hydro variables updated by divergence (B updated by CT)
const Nhydro = ct_mode ? 5 : Ncons
const Ncell_cons = Nhydro

# ─── MHD parameters ───
# GLM divergence cleaning (Dedner et al. 2002):
#   ψ-equation: ∂ψ/∂t + ch²·∇·B = -(ch²/cp²)·ψ
#   ch = max fast magnetosonic speed (auto-computed each time step)
#   cr = ch/cp = damping ratio (user-configurable, typically 0.18)
if !@isdefined(cr_glm);  const cr_glm::FT  = FT(0.18);  end   # GLM damping ratio

# Resistive MHD control (analogous to `viscous` for hydrodynamic viscosity)
# resistive = false → ideal MHD (no magnetic diffusion)
# resistive = true  → resistive MHD (η_mhd > 0), including CT edge EMFs
if !@isdefined(resistive); const resistive::Bool = false; end
if !@isdefined(η_mhd);    const η_mhd::FT  = FT(0.0); end   # Magnetic resistivity
if resistive && !(isfinite(η_mhd) && η_mhd > zero(FT))
    error("resistive MHD requires finite η_mhd > 0, got $η_mhd")
end
const CT_RESISTIVE_EXPLICIT = :explicit
const CT_RESISTIVE_STS = :sts
const CT_RESISTIVE_RKL2_STRANG = :rkl2_strang
if !@isdefined(ct_resistive_integrator)
    const ct_resistive_integrator::Symbol = CT_RESISTIVE_EXPLICIT
end
ct_resistive_integrator in (
    CT_RESISTIVE_EXPLICIT, CT_RESISTIVE_STS, CT_RESISTIVE_RKL2_STRANG,
) ||
    error("Unknown ct_resistive_integrator=$ct_resistive_integrator")
if !@isdefined(ct_sts_damping)
    const ct_sts_damping::FT = FT(0.01)
end
if !@isdefined(ct_sts_max_stages)
    const ct_sts_max_stages::Int = 64
end
if !@isdefined(ct_sts_safety)
    const ct_sts_safety::FT = FT(0.9)
end
if !@isdefined(ct_rkl2_max_stages)
    const ct_rkl2_max_stages::Int = 64
end
if !@isdefined(ct_rkl2_safety)
    const ct_rkl2_safety::FT = FT(0.9)
end
if ct_resistive_integrator == CT_RESISTIVE_RKL2_STRANG
    isfinite(ct_rkl2_safety) && zero(FT) < ct_rkl2_safety <= one(FT) ||
        error("ct_rkl2_safety must lie in (0, 1], got $ct_rkl2_safety")
    ct_rkl2_max_stages >= 2 || error(
        "ct_rkl2_max_stages must be at least two, got $ct_rkl2_max_stages",
    )
end
const ct_resistive_main_explicit::Bool =
    ct_resistive_integrator == CT_RESISTIVE_EXPLICIT
const ct_resistive_sts_active::Bool =
    equation_type == :MHD && ct_mode && resistive &&
    ct_resistive_integrator == CT_RESISTIVE_STS
const ct_resistive_rkl2_active::Bool =
    equation_type == :MHD && ct_mode && resistive &&
    ct_resistive_integrator == CT_RESISTIVE_RKL2_STRANG

# Magnetic field is stored in Tesla; MHD formulas use MU0_SI explicitly.

# ─── MHD primitive variable index constants ───
# In Q (primitive): [ρ, u, v, w, p, T, Bx, By, Bz, ψ]
const QBX  = 7;  const QBY  = 8;  const QBZ  = 9;  const QPSI = 10
# In U (conservative): [ρ, ρu, ρv, ρw, ρE, Bx, By, Bz, ψ]
const UBX  = 6;  const UBY  = 7;  const UBZ  = 8;  const UPSI = 9
