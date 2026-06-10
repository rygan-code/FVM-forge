# ═══════════════════════════════════════════════════════════════════════════════
#  Full-Field Diagnostic Module
#
#  Purpose: Scan every cell in the domain for checkerboard artifacts,
#           gradient smoothness, and solution boundedness.
#
#  Usage: Include this file in solver.jl, then call:
#     fullfield_diagnostic!(blocks, connectivity, tt_val, world_rank)
#  at the desired injection point (after ghost exchange).
#
#  Output: Console report + file "debug/output/fullfield_rankN_stepM.txt"
#
#  Checks:
#    1. Full-field checkerboard energy (2Δx oscillations) in x,y,z directions
#    2. Gradient smoothness (non-physical jumps)
#    3. Solution boundedness (physical values)
#    4. Interface vs interior comparison
# ═══════════════════════════════════════════════════════════════════════════════

export fullfield_diagnostic!

# ── Helpers ──────────────────────────────────────────────────────────────────

"""
    _is_interface_cell(i, j, k, Nx, Ny, Nz, n_layers) -> Bool

Return true if the real cell (i,j,k) is within `n_layers` cells of any
block boundary.  Indices i,j,k are in the full-array convention
(real cells run from NG+1 to N+NG).
"""
function _is_interface_cell(i, j, k, Nx, Ny, Nz, ng, n_layers)
    ri = i - ng        # 1-based real index
    rj = j - ng
    rk = k - ng
    return (ri <= n_layers || ri > Nx - n_layers ||
            rj <= n_layers || rj > Ny - n_layers ||
            rk <= n_layers || rk > Nz - n_layers)
end

# ── Main diagnostic ─────────────────────────────────────────────────────────

function fullfield_diagnostic!(blocks, connectivity, tt_val, world_rank)
    # Only trigger once at the specified step
    diag_step = @isdefined(fullfield_diag_step) ? fullfield_diag_step : 100
    if tt_val != diag_step; return; end

    ng = NG

    if world_rank == 0
        @printf("\n")
        @printf("╔═══════════════════════════════════════════════════════════╗\n")
        @printf("║  FULL-FIELD DIAGNOSTIC — Step %d, Rank %d              ║\n", tt_val, world_rank)
        @printf("╚═══════════════════════════════════════════════════════════╝\n")
    end

    # Number of interface layers to define the "near-boundary" region
    n_intf = @isdefined(fullfield_n_intf) ? fullfield_n_intf : 4

    # ── Per-block quantities ────────────────────────────────────────────────
    # Checkerboard energy: (variable, direction, region)
    #   variable: 1=ρ, 2=u, 3=v, 4=w, 5=p, 6=T
    #   direction: 1=x(ξ), 2=y(η), 3=z(ζ)
    #   region: 1=interior, 2=interface
    var_names = ["rho", "u", "v", "w", "p", "T"]

    # Accumulate sums and counts across blocks
    e2dx_sum   = zeros(6, 3, 2)   # sum of |e_2dx|
    e2dx_max   = zeros(6, 3, 2)   # max |e_2dx|
    e2dx_count = zeros(Int, 6, 3, 2)
    # Cell coordinates for max E_2dx: (bid, i, j, k) per (var, dir, region)
    e2dx_max_bid = zeros(Int, 6, 3, 2)
    e2dx_max_i   = zeros(Int, 6, 3, 2)
    e2dx_max_j   = zeros(Int, 6, 3, 2)
    e2dx_max_k   = zeros(Int, 6, 3, 2)

    # Gradient smoothness: max |Δf/f| for density and pressure
    grad_rho_max = 0.0
    grad_p_max   = 0.0
    grad_rho_loc = (0, 0, 0, -1)  # (i,j,k,bid)
    grad_p_loc   = (0, 0, 0, -1)

    # Solution bounds
    rho_min, rho_max_val = Inf, -Inf
    u_min, u_max_val     = Inf, -Inf
    v_min, v_max_val     = Inf, -Inf
    w_min, w_max_val     = Inf, -Inf
    p_min, p_max_val     = Inf, -Inf
    T_min, T_max_val     = Inf, -Inf

    # ── Per-block processing ────────────────────────────────────────────────
    for (bid, b) in blocks
        # Download U to CPU
        U_cpu = Array(b.U)
        Nx, Ny, Nz = b.Nx, b.Ny, b.Nz

        # Pre-compute pressure, temperature, and primitive variables for all real cells
        p_arr = zeros(Nx, Ny, Nz)
        T_arr = zeros(Nx, Ny, Nz)
        rho_field = zeros(Nx, Ny, Nz)
        u_field   = zeros(Nx, Ny, Nz)
        v_field   = zeros(Nx, Ny, Nz)
        w_field   = zeros(Nx, Ny, Nz)
        @inbounds for k in (ng+1):(Nz+ng), j in (ng+1):(Ny+ng), i in (ng+1):(Nx+ng)
            ri, rj, rk = i-ng, j-ng, k-ng
            rho_val  = U_cpu[i,j,k,1]
            u_val    = U_cpu[i,j,k,2] / rho_val
            v_val    = U_cpu[i,j,k,3] / rho_val
            w_val    = U_cpu[i,j,k,4] / rho_val
            rhoE_val = U_cpu[i,j,k,5]
            ke = 0.5 * rho_val * (u_val^2 + v_val^2 + w_val^2)
            p_val = (γ - 1.0) * (rhoE_val - ke)
            T_val = p_val / (rho_val * Rg)
            p_arr[ri,rj,rk]     = p_val
            T_arr[ri,rj,rk]     = T_val
            rho_field[ri,rj,rk] = rho_val
            u_field[ri,rj,rk]   = u_val
            v_field[ri,rj,rk]   = v_val
            w_field[ri,rj,rk]   = w_val

            # Solution bounds
            rho_min = min(rho_min, rho_val);  rho_max_val = max(rho_max_val, rho_val)
            u_min   = min(u_min, u_val);      u_max_val   = max(u_max_val, u_val)
            v_min   = min(v_min, v_val);      v_max_val   = max(v_max_val, v_val)
            w_min   = min(w_min, w_val);      w_max_val   = max(w_max_val, w_val)
            p_min   = min(p_min, p_val);      p_max_val   = max(p_max_val, p_val)
            T_min   = min(T_min, T_val);      T_max_val   = max(T_max_val, T_val)
        end

        # Pack all 6 fields for iteration
        fields = (rho_field, u_field, v_field, w_field, p_arr, T_arr)

        # ── Checkerboard energy in ξ (x) direction ──
        for (fidx, fld) in enumerate(fields)
            for rk in 1:Nz, rj in 1:Ny, ri in 2:Nx-1
                e = abs(fld[ri,rj,rk] - 0.5*(fld[ri-1,rj,rk] + fld[ri+1,rj,rk]))
                is_intf = _is_interface_cell(ri+ng, rj+ng, rk+ng, Nx, Ny, Nz, ng, n_intf)
                reg = is_intf ? 2 : 1
                e2dx_sum[fidx, 1, reg]   += e
                e2dx_count[fidx, 1, reg] += 1
                if e > e2dx_max[fidx, 1, reg]
                    e2dx_max[fidx, 1, reg]     = e
                    e2dx_max_bid[fidx, 1, reg] = bid
                    e2dx_max_i[fidx, 1, reg]   = ri + ng
                    e2dx_max_j[fidx, 1, reg]   = rj + ng
                    e2dx_max_k[fidx, 1, reg]   = rk + ng
                end
            end
        end

        # ── Checkerboard energy in η (y) direction ──
        for (fidx, fld) in enumerate(fields)
            for rk in 1:Nz, rj in 2:Ny-1, ri in 1:Nx
                e = abs(fld[ri,rj,rk] - 0.5*(fld[ri,rj-1,rk] + fld[ri,rj+1,rk]))
                is_intf = _is_interface_cell(ri+ng, rj+ng, rk+ng, Nx, Ny, Nz, ng, n_intf)
                reg = is_intf ? 2 : 1
                e2dx_sum[fidx, 2, reg]   += e
                e2dx_count[fidx, 2, reg] += 1
                if e > e2dx_max[fidx, 2, reg]
                    e2dx_max[fidx, 2, reg]     = e
                    e2dx_max_bid[fidx, 2, reg] = bid
                    e2dx_max_i[fidx, 2, reg]   = ri + ng
                    e2dx_max_j[fidx, 2, reg]   = rj + ng
                    e2dx_max_k[fidx, 2, reg]   = rk + ng
                end
            end
        end

        # ── Checkerboard energy in ζ (z) direction ──
        for (fidx, fld) in enumerate(fields)
            for rk in 2:Nz-1, rj in 1:Ny, ri in 1:Nx
                e = abs(fld[ri,rj,rk] - 0.5*(fld[ri,rj,rk-1] + fld[ri,rj,rk+1]))
                is_intf = _is_interface_cell(ri+ng, rj+ng, rk+ng, Nx, Ny, Nz, ng, n_intf)
                reg = is_intf ? 2 : 1
                e2dx_sum[fidx, 3, reg]   += e
                e2dx_count[fidx, 3, reg] += 1
                if e > e2dx_max[fidx, 3, reg]
                    e2dx_max[fidx, 3, reg]     = e
                    e2dx_max_bid[fidx, 3, reg] = bid
                    e2dx_max_i[fidx, 3, reg]   = ri + ng
                    e2dx_max_j[fidx, 3, reg]   = rj + ng
                    e2dx_max_k[fidx, 3, reg]   = rk + ng
                end
            end
        end

        # ── Gradient smoothness ──
        @inbounds for k in (ng+2):(Nz+ng-1), j in (ng+2):(Ny+ng-1), i in (ng+2):(Nx+ng-1)
            ri, rj, rk = i-ng, j-ng, k-ng

            # ξ-direction
            rho_c = U_cpu[i,j,k,1]
            rho_p = U_cpu[i+1,j,k,1]
            rho_m = U_cpu[i-1,j,k,1]
            if rho_c > 1e-10
                drho = abs(rho_p - rho_m) / rho_c
                if drho > grad_rho_max
                    grad_rho_max = drho
                    grad_rho_loc = (i, j, k, bid)
                end
            end

            p_c = p_arr[ri,rj,rk]
            p_p = p_arr[ri+1,rj,rk]
            p_m = p_arr[ri-1,rj,rk]
            if p_c > 1e-10
                dp = abs(p_p - p_m) / p_c
                if dp > grad_p_max
                    grad_p_max = dp
                    grad_p_loc = (i, j, k, bid)
                end
            end

            # η-direction
            rho_p = U_cpu[i,j+1,k,1]
            rho_m = U_cpu[i,j-1,k,1]
            if rho_c > 1e-10
                drho = abs(rho_p - rho_m) / rho_c
                if drho > grad_rho_max
                    grad_rho_max = drho
                    grad_rho_loc = (i, j, k, bid)
                end
            end

            p_p = p_arr[ri,rj+1,rk]
            p_m = p_arr[ri,rj-1,rk]
            if p_c > 1e-10
                dp = abs(p_p - p_m) / p_c
                if dp > grad_p_max
                    grad_p_max = dp
                    grad_p_loc = (i, j, k, bid)
                end
            end

            # ζ-direction
            rho_p = U_cpu[i,j,k+1,1]
            rho_m = U_cpu[i,j,k-1,1]
            if rho_c > 1e-10
                drho = abs(rho_p - rho_m) / rho_c
                if drho > grad_rho_max
                    grad_rho_max = drho
                    grad_rho_loc = (i, j, k, bid)
                end
            end

            p_p = p_arr[ri,rj,rk+1]
            p_m = p_arr[ri,rj,rk-1]
            if p_c > 1e-10
                dp = abs(p_p - p_m) / p_c
                if dp > grad_p_max
                    grad_p_max = dp
                    grad_p_loc = (i, j, k, bid)
                end
            end
        end
    end  # block loop

    # ── Compute averages ────────────────────────────────────────────────────
    e2dx_avg = zeros(6, 3, 2)
    for r in 1:2, d in 1:3, v in 1:6
        e2dx_avg[v,d,r] = e2dx_count[v,d,r] > 0 ? e2dx_sum[v,d,r] / e2dx_count[v,d,r] : 0.0
    end

    # ── Report ──────────────────────────────────────────────────────────────
    if world_rank != 0; return; end

    # Collect output in a buffer for file writing
    output = IOBuffer()

    dir_names = ["x(xi)", "y(eta)", "z(zeta)"]
    region_names = ["interior", "interface"]

    @printf("\n═══════════════════════════════════════════════════════════\n")
    @printf("  FULL-FIELD CHECKERBOARD ENERGY (|f[i] - 0.5*(f[i-1]+f[i+1])|)\n")
    @printf("═══════════════════════════════════════════════════════════\n\n")
    @printf(output, "\n═══════════════════════════════════════════════════════════\n")
    @printf(output, "  FULL-FIELD CHECKERBOARD ENERGY (|f[i] - 0.5*(f[i-1]+f[i+1])|)\n")
    @printf(output, "═══════════════════════════════════════════════════════════\n\n")

    for v in 1:6
        @printf("  %s:\n", var_names[v])
        @printf(output, "  %s:\n", var_names[v])
        for d in 1:3
            avg_int  = e2dx_avg[v,d,1]
            avg_intf = e2dx_avg[v,d,2]
            max_int  = e2dx_max[v,d,1]
            max_intf = e2dx_max[v,d,2]
            amp = avg_int > 1e-30 ? avg_intf / avg_int : Inf

            field_count = e2dx_count[v,d,1] + e2dx_count[v,d,2]
            field_avg = field_count > 0 ? (e2dx_sum[v,d,1] + e2dx_sum[v,d,2]) / field_count : 0.0

            @printf("    %s: field_avg=%.3e  avg_int=%.3e  avg_intf=%.3e  max_int=%.3e  max_intf=%.3e  amp=%.2fx\n",
                    dir_names[d], field_avg, avg_int, avg_intf, max_int, max_intf, amp)
            @printf(output, "    %s: field_avg=%.3e  avg_int=%.3e  avg_intf=%.3e  max_int=%.3e  max_intf=%.3e  amp=%.2fx\n",
                          dir_names[d], field_avg, avg_int, avg_intf, max_int, max_intf, amp)
        end
        @printf("\n")
        @printf(output, "\n")
    end

    @printf("  Summary:\n")
    @printf(output, "  Summary:\n")

    # Overall max E_2dx across all variables and directions
    overall_max = 0.0
    overall_max_var = ""
    overall_max_dir = ""
    overall_max_reg = ""
    overall_max_loc = (0, 0, 0, 0)  # (bid, i, j, k)
    for v in 1:6, d in 1:3, r in 1:2
        if e2dx_max[v,d,r] > overall_max
            overall_max = e2dx_max[v,d,r]
            overall_max_var = var_names[v]
            overall_max_dir = dir_names[d]
            overall_max_reg = region_names[r]
            overall_max_loc = (e2dx_max_bid[v,d,r], e2dx_max_i[v,d,r],
                               e2dx_max_j[v,d,r], e2dx_max_k[v,d,r])
        end
    end
    @printf("    Max E_2dx = %.3e (%s, %s, %s, bid=%d, i=%d, j=%d, k=%d)\n",
            overall_max, overall_max_var, overall_max_dir, overall_max_reg,
            overall_max_loc[1], overall_max_loc[2], overall_max_loc[3], overall_max_loc[4])
    @printf(output, "    Max E_2dx = %.3e (%s, %s, %s, bid=%d, i=%d, j=%d, k=%d)\n",
            overall_max, overall_max_var, overall_max_dir, overall_max_reg,
            overall_max_loc[1], overall_max_loc[2], overall_max_loc[3], overall_max_loc[4])

    # Amplification factors
    @printf("\n  Amplification factors (interface / interior avg):\n")
    @printf(output, "\n  Amplification factors (interface / interior avg):\n")
    for v in 1:6
        amps = Float64[]
        for d in 1:3
            a = e2dx_avg[v,d,1] > 1e-30 ? e2dx_avg[v,d,2] / e2dx_avg[v,d,1] : 0.0
            push!(amps, a)
        end
        @printf("    %-5s:  x=%.2f  y=%.2f  z=%.2f\n", var_names[v], amps[1], amps[2], amps[3])
        @printf(output, "    %-5s:  x=%.2f  y=%.2f  z=%.2f\n", var_names[v], amps[1], amps[2], amps[3])
    end

    # ── Solution bounds ─────────────────────────────────────────────────────
    @printf("\n═══════════════════════════════════════════════════════════\n")
    @printf("  SOLUTION BOUNDS\n")
    @printf("═══════════════════════════════════════════════════════════\n\n")
    @printf(output, "\n═══════════════════════════════════════════════════════════\n")
    @printf(output, "  SOLUTION BOUNDS\n")
    @printf(output, "═══════════════════════════════════════════════════════════\n\n")
    for (name, lo, hi) in [("rho", rho_min, rho_max_val), ("u", u_min, u_max_val),
                            ("v", v_min, v_max_val), ("w", w_min, w_max_val),
                            ("p", p_min, p_max_val), ("T", T_min, T_max_val)]
        @printf("  %s : [%.6e, %.6e]\n", name, lo, hi)
        @printf(output, "  %s : [%.6e, %.6e]\n", name, lo, hi)
    end

    # ── Gradient smoothness ─────────────────────────────────────────────────
    @printf("\n═══════════════════════════════════════════════════════════\n")
    @printf("  GRADIENT SMOOTHNESS\n")
    @printf("═══════════════════════════════════════════════════════════\n\n")
    @printf(output, "\n═══════════════════════════════════════════════════════════\n")
    @printf(output, "  GRADIENT SMOOTHNESS\n")
    @printf(output, "═══════════════════════════════════════════════════════════\n\n")
    @printf("  max |drho/rho| = %.3e  (bid=%d, i=%d, j=%d, k=%d)\n",
            grad_rho_max, grad_rho_loc[4], grad_rho_loc[1], grad_rho_loc[2], grad_rho_loc[3])
    @printf("  max |dp/p|     = %.3e  (bid=%d, i=%d, j=%d, k=%d)\n",
            grad_p_max, grad_p_loc[4], grad_p_loc[1], grad_p_loc[2], grad_p_loc[3])
    @printf(output, "  max |drho/rho| = %.3e  (bid=%d, i=%d, j=%d, k=%d)\n",
                  grad_rho_max, grad_rho_loc[4], grad_rho_loc[1], grad_rho_loc[2], grad_rho_loc[3])
    @printf(output, "  max |dp/p|     = %.3e  (bid=%d, i=%d, j=%d, k=%d)\n",
                  grad_p_max, grad_p_loc[4], grad_p_loc[1], grad_p_loc[2], grad_p_loc[3])

    # ── PASS/FAIL ───────────────────────────────────────────────────────────
    @printf("\n═══════════════════════════════════════════════════════════\n")
    @printf("  PASS/FAIL RESULT\n")
    @printf("═══════════════════════════════════════════════════════════\n\n")
    @printf(output, "\n═══════════════════════════════════════════════════════════\n")
    @printf(output, "  PASS/FAIL RESULT\n")
    @printf(output, "═══════════════════════════════════════════════════════════\n\n")

    fail = false

    # Fail if any physical quantity is non-physical
    if rho_min <= 0.0
        @printf("  [FAIL] Non-positive density detected: min rho = %.3e\n", rho_min)
        @printf(output, "  [FAIL] Non-positive density detected: min rho = %.3e\n", rho_min)
        fail = true
    end
    if p_min <= 0.0
        @printf("  [FAIL] Non-positive pressure detected: min p = %.3e\n", p_min)
        @printf(output, "  [FAIL] Non-positive pressure detected: min p = %.3e\n", p_min)
        fail = true
    end
    if T_min <= 0.0
        @printf("  [FAIL] Non-positive temperature detected: min T = %.3e\n", T_min)
        @printf(output, "  [FAIL] Non-positive temperature detected: min T = %.3e\n", T_min)
        fail = true
    end

    # Fail if gradient smoothness is too extreme
    if grad_rho_max > 1.0
        @printf("  [FAIL] Large density gradient: max |drho/rho| = %.3e > 1.0\n", grad_rho_max)
        @printf(output, "  [FAIL] Large density gradient: max |drho/rho| = %.3e > 1.0\n", grad_rho_max)
        fail = true
    end
    if grad_p_max > 1.0
        @printf("  [FAIL] Large pressure gradient: max |dp/p| = %.3e > 1.0\n", grad_p_max)
        @printf(output, "  [FAIL] Large pressure gradient: max |dp/p| = %.3e > 1.0\n", grad_p_max)
        fail = true
    end

    if fail
        @printf("  RESULT: FAIL\n")
        @printf(output, "  RESULT: FAIL\n")
    else
        @printf("  RESULT: PASS\n")
        @printf(output, "  RESULT: PASS\n")
    end

    @printf("\n  Full-field diagnostic complete.\n\n")
    @printf(output, "\n  Full-field diagnostic complete.\n\n")

    # ── Write output file ───────────────────────────────────────────────────
    mkpath("debug/output")
    fname = "debug/output/fullfield_rank$(world_rank)_step$(tt_val).txt"
    open(fname, "w") do f
        write(f, String(take!(output)))
    end
    @printf("  Output written to: %s\n", fname)
end
