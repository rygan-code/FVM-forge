# SI electromagnetic constants and small helpers used by MHD-only paths.
#
# The compressible branch does not include this file's formulas. MHD stores
# magnetic field in Tesla, while U[5] stores total energy density in J/m^3.

if !@isdefined(FT)
    # Keep direct MHD/CT unit-test includes self-contained. Full solver
    # entrypoints define FT before loading equation_config.jl.
    const FT = Float64
end

if !@isdefined(isothermal_mhd)
    const isothermal_mhd::Bool = false
end
if !@isdefined(Rg)
    const Rg::FT = one(FT)
end
if !@isdefined(isothermal_temperature)
    const isothermal_temperature::FT = one(FT)
end
# MHD fluxes can be included before the shared positivity policy.  Define the
# same default here only when the entry point has not supplied a case-specific
# SI floor; structured cases normally define it before equation_config.jl.
if !@isdefined(density_floor)
    const density_floor::FT = FT(1.0e-5)
end
if !@isdefined(pressure_floor)
    const pressure_floor::FT = FT(1.0e-5)
end

if !@isdefined(MHD_SI_UNITS_LOADED)
    # Vacuum permeability in SI. The legacy code used the Gaussian/MHD
    # convention mu0=1, so keep every conversion explicit at the call site.
    const MU0_SI::FT = FT(4.0 * 3.141592653589793e-7)
    const INV_MU0_SI::FT = inv(MU0_SI)
    const SQRT_MU0_SI::FT = sqrt(MU0_SI)
    const INV_SQRT_MU0_SI::FT = inv(SQRT_MU0_SI)

    @inline function mhd_magnetic_energy_density(Bx, By, Bz)
        return (Bx*Bx + By*By + Bz*Bz) * (INV_MU0_SI / 2)
    end

    @inline mhd_magnetic_pressure(Bx, By, Bz) =
        mhd_magnetic_energy_density(Bx, By, Bz)

    @inline function mhd_alfven_speed_squared(Bx, By, Bz, rho)
        return (Bx*Bx + By*By + Bz*Bz) * INV_MU0_SI / rho
    end

    @inline function mhd_magnetic_forcing_work(
        Bx, By, Bz, dBx, dBy, dBz,
    )
        return (Bx*dBx + By*dBy + Bz*dBz) * INV_MU0_SI
    end

    # Isothermal MHD uses p = rho*Rg*T_iso. The fifth cell-centered slot
    # remains a finite CT/RK carrier, but is not used for pressure recovery.
    @inline function mhd_thermodynamic_pressure(
        rho, internal_energy, gamma,
    )
        if isothermal_mhd
            return rho * Rg * isothermal_temperature
        end
        return (gamma - one(gamma)) * internal_energy
    end

    @inline mhd_isothermal_pressure(rho) =
        rho * Rg * isothermal_temperature

    @inline function mhd_isothermal_carrier_density(
        rho, momentum_x, momentum_y, momentum_z, Bx, By, Bz,
    )
        kinetic = FT(0.5) * (
            momentum_x*momentum_x + momentum_y*momentum_y +
            momentum_z*momentum_z
        ) / rho
        return mhd_isothermal_pressure(rho) + kinetic +
               mhd_magnetic_energy_density(Bx, By, Bz)
    end

    @inline function mhd_thermodynamic_temperature(rho, pressure)
        return isothermal_mhd ? isothermal_temperature : pressure / (rho * Rg)
    end

    @inline function mhd_energy_density_from_primitive(
        rho, u, v, w, pressure, Bx, By, Bz, gamma,
    )
        kinetic = FT(0.5) * rho * (u*u + v*v + w*w)
        magnetic = mhd_magnetic_energy_density(Bx, By, Bz)
        if isothermal_mhd
            # Avoid the singular p/(gamma-1) representation. U[5] is not
            # the thermodynamic variable in the isothermal closure.
            return kinetic + magnetic + pressure
        end
        return pressure / (gamma - one(gamma)) + kinetic + magnetic
    end

    @inline mhd_energy_density_from_primitive(
        rho, u, v, w, pressure, Bx, By, Bz,
    ) = mhd_energy_density_from_primitive(
        rho, u, v, w, pressure, Bx, By, Bz, γ,
    )

    @inline function mhd_sound_speed_squared(rho, pressure, gamma)
        return isothermal_mhd ? pressure / rho : gamma * pressure / rho
    end

    function validate_structured_ct_thermodynamics(
        equations::Symbol, use_ct::Bool, use_isothermal::Bool,
        gamma, gas_constant, isothermal_temp,
    )
        equations == :MHD && use_ct || return nothing

        isfinite(gamma) || throw(ArgumentError(
            "structured CT requires a finite gamma, got $gamma",
        ))
        if use_isothermal
            gamma_tolerance = 8 * eps(typeof(gamma))
            abs(gamma - one(gamma)) <= gamma_tolerance ||
                throw(ArgumentError(
                    "isothermal structured CT requires gamma=1, got $gamma",
                ))
            isfinite(gas_constant) && gas_constant > zero(gas_constant) ||
                throw(ArgumentError(
                    "isothermal structured CT requires Rg>0, got $gas_constant",
                ))
            isfinite(isothermal_temp) &&
                isothermal_temp > zero(isothermal_temp) ||
                throw(ArgumentError(
                    "isothermal structured CT requires T_iso>0, got $isothermal_temp",
                ))
        else
            gamma > one(gamma) || throw(ArgumentError(
                "adiabatic structured CT requires gamma>1, got $gamma",
            ))
        end
        return nothing
    end

    const MHD_SI_UNITS_LOADED = true
end
