# bc_types.jl — Unified boundary condition type system for Flame3D
# Shared between solver (boundary.jl) and mesh converter (convert_plot3d.jl)
#
# BC types are stored as integers in face_bc[Nblocks, 6] in block_connectivity.h5
# BC parameters are stored in bc_params[Nblocks, 6, N_BC_PARAMS] in block_connectivity.h5

# ═══════════════════════════════════════════════════════════
# BC Type IDs
# ═══════════════════════════════════════════════════════════
const BC_INTERBLOCK         = Int32(0)   # Inter-block ghost exchange
const BC_ISOTHERMAL_WALL    = Int32(1)   # No-slip, fixed wall temperature Tw
const BC_PERIODIC           = Int32(2)   # Periodic (handled by MPI Cart topology)
const BC_SUPERSONIC_INFLOW  = Int32(3)   # All variables fixed (Dirichlet)
const BC_SUPERSONIC_OUTFLOW = Int32(4)   # Linear extrapolation from interior
const BC_SYMMETRY           = Int32(5)   # Reflect normal velocity, mirror scalars
const BC_ADIABATIC_WALL     = Int32(6)   # No-slip, zero heat flux (dT/dn = 0)
const BC_SUBSONIC_INFLOW    = Int32(7)   # Total pressure + total temperature + direction
const BC_SUBSONIC_OUTFLOW   = Int32(8)   # Back pressure specified, extrapolate rest
const BC_NSCBC_OUTFLOW      = Int32(9)   # Non-reflecting (Poinsot & Lele 1992)
const BC_RIEMANN_OUTFLOW    = Int32(13)  # Riemann invariant outflow (self-consistent with internal Riemann solver)
const BC_FARFIELD           = Int32(10)  # Riemann invariant far-field
const BC_ZERO_GRADIENT      = Int32(11)  # Neumann: copy from interior
const BC_SLIP_WALL          = Int32(12)  # Reflect normal velocity, keep tangential
const BC_MHD_WALL           = Int32(30)  # Perfectly conducting wall (B·n=0, reflect Et)
const BC_MHD_INFLOW         = Int32(31)  # MHD inflow: all vars + B-field fixed
const BC_MHD_OUTFLOW        = Int32(32)  # MHD outflow: zero-gradient for all
const BC_MHD_INSULATING_WALL = Int32(33) # Insulating wall boundary condition (Dirichlet Bt=0, Neumann Bn)
const BC_TRANSITION_INFLOW  = Int32(40)  # Dynamic spatial transition inflow (laminar + perturbation)
const BC_WAVE_INFLOW        = Int32(41)  # Pure deterministic wave source inflow
const BC_DIFFROT_WALL       = Int32(42)  # Isothermal wall with differential rotation
const BC_CEBL_INFLOW        = Int32(43)  # CEBL dynamic mapped inflow
const BC_MHD_RESERVOIR_INFLOW = Int32(44) # Fixed rho/T reservoir, velocity extrapolated
const BC_MHD_EXTERNAL_FIELD = Int32(45) # No-slip flow with prescribed coil B ghost field
const BC_MHD_PROFILED_INFLOW = Int32(46) # Sheth-type rho/Vz profile + prescribed B
const BC_MHD_FIXED_EXTERNAL_FIELD = Int32(47) # Fixed initial fluid state + prescribed B
const BC_MHD_OUTFLOW_EXTERNAL_FIELD = Int32(48) # Outflow fluid + prescribed B


# ═══════════════════════════════════════════════════════════
# BC Parameter Slots (indices into bc_params[:, :, slot])
# ═══════════════════════════════════════════════════════════
const BCP_TW        = 1    # Wall temperature [K]           (isothermal_wall)
const BCP_P_TARGET  = 2    # Target/back pressure            (subsonic_outflow, nscbc_outflow)
const BCP_RHO_INF   = 3    # Freestream density              (supersonic_inflow, farfield)
const BCP_U_INF     = 4    # Freestream u-velocity            (supersonic_inflow, farfield)
const BCP_V_INF     = 5    # Freestream v-velocity            (supersonic_inflow, farfield)
const BCP_W_INF     = 6    # Freestream w-velocity            (supersonic_inflow, farfield)
const BCP_P_INF     = 7    # Freestream pressure              (supersonic_inflow, farfield)
const BCP_SIGMA     = 8    # NSCBC relaxation coefficient σ   (nscbc_outflow, default=0.25)
const BCP_LREF      = 9    # Reference length for NSCBC       (nscbc_outflow)
const BCP_OUTLET_PAVG = 15 # Block outlet area-averaged p anchor for NSCBC L1 (updated at runtime)
const BCP_RIEMANN_ALPHA = 16 # R⁻ soft-anchor relaxation weight (0=extrapolate, >0=relax toward p_anchor)
const BCP_P0        = 10   # Total pressure                   (subsonic_inflow)
const BCP_T0        = 11   # Total temperature                (subsonic_inflow)
const BCP_DIR_X     = 12   # Inflow direction x-component     (subsonic_inflow)
const BCP_DIR_Y     = 13   # Inflow direction y-component     (subsonic_inflow)
const BCP_DIR_Z     = 14   # Inflow direction z-component     (subsonic_inflow)
const BCP_BX_INF    = 18   # Freestream Bx                   (mhd_inflow)
const BCP_BY_INF    = 19   # Freestream By                   (mhd_inflow)
const BCP_BZ_INF    = 20   # Freestream Bz                   (mhd_inflow)
const BCP_WAVE_AMP  = 1    # Wave amplitude for BC_WAVE_INFLOW
const BCP_WAVE_OMEGA = 2   # Wave frequency omega for BC_WAVE_INFLOW
const BCP_WAVE_M    = 3    # Wave azimuthal mode number m for BC_WAVE_INFLOW
const BCP_MN_RHO0   = 1    # Magnetic-nozzle reservoir density
const BCP_MN_T0     = 2    # Magnetic-nozzle reservoir temperature
const BCP_MN_B0     = 3    # Coil field normalization in solver units
const BCP_MN_XC     = 4    # Coil axial center
const BCP_MN_RB     = 5    # Coil radius
const BCP_MN_LB     = 6    # Coil length
const BCP_MN_TURNS  = 7    # Number of circular turns
const BCP_MN_MODEL  = 17   # External-field model selector (0=solenoid, 1=Bessel)
const BCP_MN_ALPHA  = 18   # Bessel nozzle alpha
const BCP_MN_KAPPA  = 19   # Inlet radial profile kappa
const BCP_MN_V0     = 20   # Inlet axial velocity in sound-speed units
const N_BC_PARAMS   = 20   # Total number of parameter slots

# ═══════════════════════════════════════════════════════════
# Name Mapping (TOML string → BC type ID)
# ═══════════════════════════════════════════════════════════
const BC_NAME_MAP = Dict{String, Int32}(
    "interblock"         => BC_INTERBLOCK,
    "isothermal_wall"    => BC_ISOTHERMAL_WALL,
    "wall"               => BC_ISOTHERMAL_WALL,    # backward compat
    "periodic"           => BC_PERIODIC,
    "supersonic_inflow"  => BC_SUPERSONIC_INFLOW,
    "inflow"             => BC_SUPERSONIC_INFLOW,   # backward compat
    "supersonic_outflow" => BC_SUPERSONIC_OUTFLOW,
    "outflow"            => BC_SUPERSONIC_OUTFLOW,   # backward compat
    "symmetry"           => BC_SYMMETRY,
    "adiabatic_wall"     => BC_ADIABATIC_WALL,
    "subsonic_inflow"    => BC_SUBSONIC_INFLOW,
    "subsonic_outflow"   => BC_SUBSONIC_OUTFLOW,
    "nscbc_outflow"      => BC_NSCBC_OUTFLOW,
    "riemann_outflow"    => BC_RIEMANN_OUTFLOW,
    "farfield"           => BC_FARFIELD,
    "zero_gradient"      => BC_ZERO_GRADIENT,
    "slip_wall"          => BC_SLIP_WALL,
    "mhd_wall"           => BC_MHD_WALL,
    "mhd_inflow"         => BC_MHD_INFLOW,
    "mhd_outflow"        => BC_MHD_OUTFLOW,
    "mhd_insulating_wall" => BC_MHD_INSULATING_WALL,
    "transition_inflow"  => BC_TRANSITION_INFLOW,
    "wave_inflow"        => BC_WAVE_INFLOW,
    "diffrot_wall"       => BC_DIFFROT_WALL,
    "cebl_inflow"        => BC_CEBL_INFLOW,
    "mhd_reservoir_inflow" => BC_MHD_RESERVOIR_INFLOW,
    "mhd_external_field" => BC_MHD_EXTERNAL_FIELD,
    "mhd_profiled_inflow" => BC_MHD_PROFILED_INFLOW,
    "mhd_fixed_external_field" => BC_MHD_FIXED_EXTERNAL_FIELD,
    "mhd_outflow_external_field" => BC_MHD_OUTFLOW_EXTERNAL_FIELD,
)

# Reverse mapping for printing
const BC_ID_TO_NAME = Dict{Int32, String}(v => k for (k, v) in BC_NAME_MAP
    if k ∉ ["wall", "inflow", "outflow"])  # skip backward-compat aliases
