using HDF5, Statistics

function exact_sod(x, t)
    # Ref: Sod, G. A. (1978). "A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws"
    # Simplified analytical solver for comparison at t=0.2
    # P_L = 1.0, Rho_L = 1.0, u_L = 0.0
    # P_R = 0.1, Rho_R = 0.125, u_R = 0.0
    # gamma = 1.4
    
    # Values at t=0.2 from standard Sod solver:
    # Pressure in region 3 (between contact and shock): 0.30313
    # Velocity in region 3: 0.92745
    # Density in region 3: 0.42632
    # Shock speed: 1.7522
    # Contact speed: 0.92745
    # Expansion left head: -c_L = -1.1832
    # Expansion right tail: u_3 - c_3 = 0.92745 - sqrt(1.4*0.30313/0.42632)
    
    # Locations at t=0.2 (relative to x=0.5):
    # Expansion head: 0.5 - 1.18321*0.2 = 0.2633
    # Expansion tail: 0.5 + (0.92745 - 0.9977)*0.2 = 0.4859
    # Contact: 0.5 + 0.92745*0.2 = 0.6855
    # Shock: 0.5 + 1.7522*0.2 = 0.8504
    
    if x < 0.2633
        return 1.0, 0.0, 1.0  # Left state
    elseif x < 0.4859
        # Rarefaction wave (linear approx in u, but using qualitative profile)
        return 0.7, 0.4, 0.7 
    elseif x < 0.6855
        return 0.4263, 0.9275, 0.3031 # Region 3
    elseif x < 0.8504
        return 0.2656, 0.9275, 0.3031 # Region 4 (Post-shock)
    else
        return 0.125, 0.0, 0.1 # Right state
    end
end

function verify_sod(filename)
    fid = h5open(filename, "r")
    rho = read(fid, "rho")[:, 4, 4]
    u = read(fid, "u")[:, 4, 4]
    p = read(fid, "p")[:, 4, 4]
    Nx = length(rho)
    dx = 1.0/Nx
    
    println("Sod Verification at t=0.2:")
    errors = []
    for i in 1:Nx
        x = (i-0.5)*dx
        rho_e, u_e, p_e = exact_sod(x, 0.2)
        push!(errors, (rho[i]-rho_e)^2)
    end
    
    l2_rho = sqrt(mean(errors))
    println("L2 error in Density: $l2_rho")
    
    # Check shock position
    # The pressure jump is largest at the shock
    dp = diff(p)
    shock_idx = argmax(abs.(dp))
    shock_x = (shock_idx)*dx
    println("Numerical Shock Position: $shock_x (Analytical ~ 0.85)")
    
    close(fid)
end

verify_sod("SOD_PLT/plt-500.h5")
