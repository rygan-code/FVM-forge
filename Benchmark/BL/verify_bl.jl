using HDF5, Statistics, Printf

function verify_bl()
    # Find the latest plt file
    files = readdir("./PLT")
    plt_files = filter(f -> occursin(r"^plt-\d+\.h5$", f), files)
    if isempty(plt_files)
        println("No plt files found.")
        return
    end
    
    # Sort and pick the last one
    step_nums = [parse(Int, match(r"(\d+)", f).captures[1]) for f in plt_files]
    last_step = maximum(step_nums)
    last_file = "./PLT/plt-$last_step.h5"
    
    println("Analyzing $last_file...")
    
    # Load data
    fid = h5open(last_file, "r")
    u = read(fid["u"])
    v = read(fid["v"])
    rho = read(fid["rho"])
    p = read(fid["p"])
    T = read(fid["T"])
    close(fid)
    
    println("Stats:")
    @printf("rho: [%.3e, %.3e], NaN: %d\n", minimum(rho), maximum(rho), count(isnan, rho))
    @printf("u: [%.3e, %.3e], NaN: %d\n", minimum(u), maximum(u), count(isnan, u))
    @printf("p: [%.3e, %.3e], NaN: %d\n", minimum(p), maximum(p), count(isnan, p))
    @printf("T: [%.3e, %.3e], NaN: %d\n", minimum(T), maximum(T), count(isnan, T))
    
    # Load mesh for coordinates
    mfid = h5open("MESH/bl_mesh.h5", "r")
    coords = read(mfid["coords"])
    close(mfid)
    
    Nx, Ny, Nz = size(u)
    
    # Extract profile at x=1.5 (downstream)
    # x is from 0 to 2.0. index = 1 + (1.5/2.0)*Nx
    ix = round(Int, 0.75 * Nx)
    x_pos = coords[1, ix, 1, 1]
    
    println("Profile at x = $x_pos (index $ix)")
    
    # u(y) at this x
    u_profile = u[ix, :, 1]
    y_nodes = coords[2, ix, :, 1]
    # y_cell is between y_nodes
    y_cell = 0.5 .* (y_nodes[1:end-1] .+ y_nodes[2:end])
    
    # Freestream velocity (should be ~2.366)
    u_inf = u_profile[end]
    println("u_inf = $u_inf")
    
    # Find boundary layer thickness delta (u = 0.99 * u_inf)
    delta = 0.0
    for j in 1:Ny
        if u_profile[j] >= 0.99 * u_inf
            delta = y_cell[j]
            break
        end
    end
    
    println("Boundary layer thickness delta ≈ $delta")
    
    # Check for no-slip (u should be small near wall)
    u_wall = u_profile[1]
    println("Velocity at first cell: $u_wall")
    
    # Calculate wall shear stress / skin friction if possible
    # tau_w = mu * (du/dy)_wall
    mu = 0.002366
    
    # Improved Wall Gradient (2nd-order one-sided using u=0 at y=0, u1 at yc1, u2 at yc2)
    yc1 = y_cell[1]; yc2 = y_cell[2]
    u1 = u_profile[1]; u2 = u_profile[2]
    
    # 2nd order Lagrange fit: u(y) = ay^2 + by (since u(0)=0)
    # b = (u1*yc2^2 - u2*yc1^2) / (yc1*yc2*(yc2 - yc1))
    dudy_wall = (u1 * yc2^2 - u2 * yc1^2) / (yc1 * yc2 * (yc2 - yc1))
    
    tau_w = mu * dudy_wall
    Cf = tau_w / (0.5 * rho[ix, end, 1] * u_inf^2)
    
    println("Skin friction coefficient Cf ≈ $Cf")
    
    # Theoretical Cf for laminar BL: 0.664 / sqrt(Re_x)
    Re_x = (rho[ix, end, 1] * u_inf * x_pos) / mu
    Cf_inc = 0.664 / sqrt(Re_x)
    
    # Compressible correction (Van Driest I for adiabatic wall)
    # Ref: White, Viscous Fluid Flow. C_f/C_f_inc approx (T*/Te)^-0.65
    gamma = 1.4
    Ma_e = u_inf / sqrt(gamma * 287.05 * T[ix, end, 1]) # Local Mach
    r = 0.84 # Recovery factor for laminar
    Tw_Te = 1.0 + r * (gamma-1)/2 * Ma_e^2
    Tstar_Te = 0.5 + 0.5*Tw_Te + 0.039 * Ma_e^2
    Cf_theory = Cf_inc * (Tstar_Te)^-0.65
    
    println("Theoretical Cf (Incompressible) ≈ $Cf_inc")
    @printf("Theoretical Cf (Compressible Ma=%.2f) ≈ %f\n", Ma_e, Cf_theory)
    
    error_pct = abs(Cf - Cf_theory) / Cf_theory * 100
    println("Skin friction error ≈ $(round(error_pct, digits=2))%")
    
    if error_pct < 10.0 # Aiming for < 5% with Ny=192
        println("Verification PASSED!")
    else
        println("Verification FAILED (error still high)!")
    end

    # Export Profile for Walkthrough
    open("bl_profile_data.csv", "w") do io
        println(io, "y,u,u_norm")
        for j in 1:Ny
            @printf(io, "%.6e,%.6e,%.6e\n", y_cell[j], u_profile[j], u_profile[j]/u_inf)
        end
    end
    println("Profile data exported to bl_profile_data.csv")
end

verify_bl()
