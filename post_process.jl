# =============================================================================
# In-situ Post-processing and Diagnostics Module
# Provides interfaces to compute bulk diagnostics, energies, and extract probes.
# =============================================================================

using MPI
using Printf
using DelimitedFiles

"""
    compute_bulk_quantities(blocks, comm_world)

Compute volume-averaged bulk quantities across all MPI ranks:
- Average density (rho_avg)
- Bulk velocity (u_avg, v_avg, w_avg)
- Average pressure (p_avg)
- Average temperature (T_avg)
"""
function compute_bulk_quantities(blocks, comm_world)
    ρ_local_sum = zero(FT)
    u_local_sum = zero(FT)
    v_local_sum = zero(FT)
    w_local_sum = zero(FT)
    p_local_sum = zero(FT)
    T_local_sum = zero(FT)
    volume_local_sum = zero(FT)
    
    for (bid, b) in blocks
        NGp = NG + 1
        nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG
        
        # Views excluding ghost cells
        ρ_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 1]
        u_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 2]
        v_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 3]
        w_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 4]
        p_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 5]
        T_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 6]
        vol_v = @view b.Vol[NGp:nx_end, NGp:ny_end, NGp:nz_end]
        
        # Sums weighted by cell volume (divided by cell Vol = cell volume)
        ρ_local_sum += mapreduce((r, v) -> Float64(r)/Float64(v), +, ρ_v, vol_v)
        u_local_sum += mapreduce((r, u, v) -> Float64(r)*Float64(u)/Float64(v), +, ρ_v, u_v, vol_v)
        v_local_sum += mapreduce((r, v_vel, v) -> Float64(r)*Float64(v_vel)/Float64(v), +, ρ_v, v_v, vol_v)
        w_local_sum += mapreduce((r, w, v) -> Float64(r)*Float64(w)/Float64(v), +, ρ_v, w_v, vol_v)
        p_local_sum += mapreduce((p, v) -> Float64(p)/Float64(v), +, p_v, vol_v)
        T_local_sum += mapreduce((t, v) -> Float64(t)/Float64(v), +, T_v, vol_v)
        volume_local_sum += mapreduce(v -> 1.0/Float64(v), +, vol_v)
    end
    
    # Global reductions
    ρ_sum_global = MPI.Allreduce(ρ_local_sum, MPI.SUM, comm_world)
    u_sum_global = MPI.Allreduce(u_local_sum, MPI.SUM, comm_world)
    v_sum_global = MPI.Allreduce(v_local_sum, MPI.SUM, comm_world)
    w_sum_global = MPI.Allreduce(w_local_sum, MPI.SUM, comm_world)
    p_sum_global = MPI.Allreduce(p_local_sum, MPI.SUM, comm_world)
    T_sum_global = MPI.Allreduce(T_local_sum, MPI.SUM, comm_world)
    volume_sum_global = MPI.Allreduce(volume_local_sum, MPI.SUM, comm_world)
    
    # Compute averages
    ρ_avg = ρ_sum_global / volume_sum_global
    u_avg = u_sum_global / ρ_sum_global
    v_avg = v_sum_global / ρ_sum_global
    w_avg = w_sum_global / ρ_sum_global
    p_avg = p_sum_global / volume_sum_global
    T_avg = T_sum_global / volume_sum_global
    
    return ρ_avg, u_avg, v_avg, w_avg, p_avg, T_avg, volume_sum_global
end

"""
    compute_integral_energies(blocks, comm_world)

Compute integrated energies in the domain:
- Kinetic energy (E_kin = 0.5 * integral( rho * (u^2 + v^2 + w^2) dV ))
- Thermal energy (E_th = integral( p / (gamma - 1) dV ))
- Magnetic energy (E_mag = 0.5 * integral( Bx^2 + By^2 + Bz^2 dV )) (only if MHD)
"""
function compute_integral_energies(blocks, comm_world)
    ekin_local = zero(FT)
    eth_local = zero(FT)
    emag_local = zero(FT)
    
    for (bid, b) in blocks
        NGp = NG + 1
        nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG
        
        ρ_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 1]
        u_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 2]
        v_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 3]
        w_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 4]
        p_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 5]
        vol_v = @view b.Vol[NGp:nx_end, NGp:ny_end, NGp:nz_end]
        
        # Kinetic energy density integrated: 0.5 * rho * (u^2+v^2+w^2) / Vol
        ekin_local += mapreduce((r, u, v_vel, w, v) -> 0.5 * Float64(r) * (Float64(u)^2 + Float64(v_vel)^2 + Float64(w)^2) / Float64(v), +, ρ_v, u_v, v_v, w_v, vol_v)
        
        # Thermal energy density integrated: p / (gamma - 1) / Vol
        gamma_factor = 1.0 / (Float64(γ) - 1.0)
        eth_local += mapreduce((p, v) -> Float64(p) * gamma_factor / Float64(v), +, p_v, vol_v)
        
        @static if equation_type == :MHD
            Bx_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 7]
            By_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 8]
            Bz_v = @view b.Q[NGp:nx_end, NGp:ny_end, NGp:nz_end, 9]
            
            # Magnetic energy integrated: 0.5 * (Bx^2+By^2+Bz^2) / Vol (vacuum permeability is normalized to 1)
            emag_local += mapreduce((bx, by, bz, v) -> 0.5 * (Float64(bx)^2 + Float64(by)^2 + Float64(bz)^2) / Float64(v), +, Bx_v, By_v, Bz_v, vol_v)
        end
    end
    
    ekin_global = MPI.Allreduce(ekin_local, MPI.SUM, comm_world)
    eth_global = MPI.Allreduce(eth_local, MPI.SUM, comm_world)
    emag_global = MPI.Allreduce(emag_local, MPI.SUM, comm_world)
    
    return ekin_global, eth_global, emag_global
end

# Helper to find nearest index in coordinates on GPU (converted to host for search)
function _find_nearest_index(coord_arr, val)
    # coord_arr is a 1D slice of coordinates (strictly monotonic)
    arr_cpu = Array(coord_arr)
    # Binary search
    idx = searchsortedfirst(arr_cpu, val)
    if idx > length(arr_cpu)
        return length(arr_cpu)
    elseif idx == 1
        return 1
    else
        # check which one is closer
        if abs(arr_cpu[idx] - val) < abs(arr_cpu[idx-1] - val)
            return idx
        else
            return idx-1
        end
    end
end

"""
    extract_probe_value(blocks, xp, yp, zp)

Search for a coordinate (xp, yp, zp) across all local blocks.
Returns a Tuple: (found::Bool, Q_values::Vector{FT})
"""
function extract_probe_value(blocks, xp, yp, zp)
    for (bid, b) in blocks
        # Convert coords to CPU to check bounds
        x_cpu = Array(b.x)
        y_cpu = Array(b.y)
        z_cpu = Array(b.z)
        
        # Bounding box check (considering only real cells)
        NGp = NG + 1
        nx_end, ny_end, nz_end = b.Nx + NG, b.Ny + NG, b.Nz + NG
        
        xmin, xmax = x_cpu[NGp, NGp, NGp], x_cpu[nx_end, NGp, NGp]
        ymin, ymax = y_cpu[NGp, NGp, NGp], y_cpu[NGp, ny_end, NGp]
        zmin, zmax = z_cpu[NGp, NGp, NGp], z_cpu[NGp, NGp, nz_end]
        
        # Check if probe is inside this block
        if xp >= xmin && xp <= xmax && yp >= ymin && yp <= ymax && zp >= zmin && zp <= zmax
            # Find closest cell indices
            # For curvilinear mesh, we do nearest-neighbor index matching on x, y, z slices
            i_idx = _find_nearest_index(x_cpu[:, NGp, NGp], xp)
            j_idx = _find_nearest_index(y_cpu[NGp, :, NGp], yp)
            k_idx = _find_nearest_index(z_cpu[NGp, NGp, :], zp)
            
            # Fetch Q values from GPU to CPU
            Q_val = Vector{FT}(Array(b.Q[i_idx, j_idx, k_idx, :]))
            return (true, Q_val)
        end
    end
    return (false, Float64[])
end

"""
    write_probe_data(tt, activeTime, blocks, probe_coords, file_path, world_rank, comm_world)

Extract values at the given probe coordinates and write them to a CSV file.
`probe_coords` is an array of Tuples/SVectors: [(x1,y1,z1), (x2,y2,z2), ...]
"""
function write_probe_data(tt, activeTime, blocks, probe_coords, file_path, world_rank, comm_world)
    local_probes = Dict{Int, Vector{FT}}()
    
    # 1. Search locally
    for (pid, coord) in enumerate(probe_coords)
        xp, yp, zp = coord[1], coord[2], coord[3]
        found, val = extract_probe_value(blocks, xp, yp, zp)
        if found
            local_probes[pid] = val
        end
    end
    
    # 2. Gather to Rank 0
    # Create serialization buffer: [pid, activeTime, Q1, Q2, ...]
    data_list = Float64[]
    for (pid, val) in local_probes
        push!(data_list, Float64(pid))
        push!(data_list, Float64(activeTime))
        for v in val
            push!(data_list, Float64(v))
        end
    end
    
    # Gather size of serialization buffers
    local_sz = length(data_list)
    sizes = MPI.Gather(local_sz, 0, comm_world)
    
    if world_rank == 0
        total_sz = sum(sizes)
        recv_buf = Vector{Float64}(undef, total_sz)
        # Gather all data buffers
        MPI.Gatherv!(data_list, VBuffer(recv_buf, sizes), 0, comm_world)
        
        # Parse received buffer
        # Each entry has size: 2 + Nprim (pid, time, primitives...)
        stride = 2 + Nprim
        num_entries = div(total_sz, stride)
        
        # Open file in append mode
        open(file_path, "a") do io
            # Write header if file is new
            if filesize(file_path) == 0
                header = ["step", "time", "probe_id", "x_coord", "y_coord", "z_coord", "rho", "u", "v", "w", "p", "T"]
                @static if equation_type == :MHD
                    append!(header, ["Bx", "By", "Bz", "psi"])
                end
                println(io, join(header, ","))
            end
            
            for e in 1:num_entries
                offset = (e - 1) * stride
                pid = round(Int, recv_buf[offset + 1])
                time_val = recv_buf[offset + 2]
                q_vals = recv_buf[(offset + 3):(offset + stride)]
                
                coord = probe_coords[pid]
                row = [tt, time_val, pid, coord[1], coord[2], coord[3]]
                append!(row, q_vals)
                println(io, join(row, ","))
            end
        end
    else
        MPI.Gatherv!(data_list, nothing, 0, comm_world)
    end
end

# =============================================================================
# Checkerboard Interface Diagnostic
# Folded from diag_checkerboard.jl.
# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
#  Checkerboard Diagnostic Module
#  
#  Purpose: One-shot diagnostic to identify root cause of 2Δx oscillations
#           at cross-type block interfaces.
#
#  Usage: Include this file in solver.jl, then call:
#     checkerboard_diagnostic!(blocks, connectivity, tt_val, world_rank)
#  at the desired injection point (after ghost exchange, before interface filter).
#
#  Output: HDF5 file "diag_checkerboard_rankN_stepM.h5" with:
#    - Per-interface 2Δx energy profiles
#    - Ghost vs interior comparison at cross-type faces
#    - Cross-type vs same-type classification
# ═══════════════════════════════════════════════════════════════════════════════

export checkerboard_diagnostic!

function checkerboard_diagnostic!(blocks, connectivity, tt_val, world_rank)
    # Only trigger once at the specified step
    diag_step = @isdefined(checkerboard_diag_step) ? checkerboard_diag_step : 100
    if tt_val != diag_step; return; end
    
    if world_rank == 0
        @printf("\n╔═══════════════════════════════════════════════════════╗\n")
        @printf("║  CHECKERBOARD DIAGNOSTIC — Step %d, Rank %d       ║\n", tt_val, world_rank)
        @printf("╚═══════════════════════════════════════════════════════╝\n")
    end
    
    ng = NG
    
    # ══════════════════════════════════════════════════════════════════════════
    #  Full-Field 2Δx Energy (relative to field mean)
    # ══════════════════════════════════════════════════════════════════════════
    var_names = ["ρ", "u", "v", "w", "p", "T"]
    dir_names = ["ξ", "η", "ζ"]
    nvar = length(var_names)
    ndir = length(dir_names)
    
    # Accumulators across all blocks: sum of E_2Δx, count, sum of |f|, count_f
    E_sum  = zeros(nvar, ndir)   # sum of |f[i] - 0.5*(f[i-1]+f[i+1])|
    E_cnt  = zeros(Int, nvar, ndir)
    F_sum  = zeros(nvar)         # sum of |f| over all real cells (for mean)
    F_cnt  = zeros(Int, nvar)
    
    # Track per-block max for location reporting
    max_E_val = zeros(nvar, ndir)
    max_E_bid = zeros(Int, nvar, ndir)
    
    for (bid, b) in blocks
        U_cpu = Array(b.U)
        Nx, Ny, Nz = b.Nx, b.Ny, b.Nz
        
        # Pre-compute primitive fields over real cells
        # Real index ranges (in global array coords)
        ri0, ri1 = ng+1, Nx+ng
        rj0, rj1 = ng+1, Ny+ng
        rk0, rk1 = ng+1, Nz+ng
        
        prim = Array{Float64}(undef, Nx, Ny, Nz, nvar)
        for rk in rk0:rk1, rj in rj0:rj1, ri in ri0:ri1
            li, lj, lk = ri - ng, rj - ng, rk - ng  # 1-based local real index
            rho   = U_cpu[ri, rj, rk, 1]
            u_val = U_cpu[ri, rj, rk, 2] / rho
            v_val = U_cpu[ri, rj, rk, 3] / rho
            w_val = U_cpu[ri, rj, rk, 4] / rho
            ke    = 0.5 * rho * (u_val^2 + v_val^2 + w_val^2)
            p_val = if equation_type == :MHD
                _, _, _, _, pressure = mhd_raw_thermo_components(
                    rho,
                    U_cpu[ri, rj, rk, 2],
                    U_cpu[ri, rj, rk, 3],
                    U_cpu[ri, rj, rk, 4],
                    U_cpu[ri, rj, rk, 5],
                    U_cpu[ri, rj, rk, 6],
                    U_cpu[ri, rj, rk, 7],
                    U_cpu[ri, rj, rk, 8],
                    γ,
                )
                pressure
            else
                (γ - 1.0) * (U_cpu[ri, rj, rk, 5] - ke)
            end
            T_val = p_val / (rho * Rg)
            prim[li, lj, lk, 1] = rho
            prim[li, lj, lk, 2] = u_val
            prim[li, lj, lk, 3] = v_val
            prim[li, lj, lk, 4] = w_val
            prim[li, lj, lk, 5] = p_val
            prim[li, lj, lk, 6] = T_val
        end
        
        # Accumulate |f| over ALL real cells for field mean
        for v in 1:nvar
            for rk in 1:Nz, rj in 1:Ny, ri in 1:Nx
                F_sum[v] += abs(prim[ri, rj, rk, v])
                F_cnt[v] += 1
            end
        end
        
        # Compute E_2Δx in each direction
        for v in 1:nvar
            # ξ-direction (i-direction): iterate ri from 2 to Nx-1
            for rk in 1:Nz, rj in 1:Ny, ri in 2:(Nx-1)
                d = abs(prim[ri, rj, rk, v] - 0.5*(prim[ri-1, rj, rk, v] + prim[ri+1, rj, rk, v]))
                E_sum[v, 1] += d
                E_cnt[v, 1] += 1
                if d > max_E_val[v, 1]
                    max_E_val[v, 1] = d
                    max_E_bid[v, 1] = bid
                end
            end
            
            # η-direction (j-direction): iterate rj from 2 to Ny-1
            for rk in 1:Nz, rj in 2:(Ny-1), ri in 1:Nx
                d = abs(prim[ri, rj, rk, v] - 0.5*(prim[ri, rj-1, rk, v] + prim[ri, rj+1, rk, v]))
                E_sum[v, 2] += d
                E_cnt[v, 2] += 1
                if d > max_E_val[v, 2]
                    max_E_val[v, 2] = d
                    max_E_bid[v, 2] = bid
                end
            end
            
            # ζ-direction (k-direction): iterate rk from 2 to Nz-1
            for rk in 2:(Nz-1), rj in 1:Ny, ri in 1:Nx
                d = abs(prim[ri, rj, rk, v] - 0.5*(prim[ri, rj, rk-1, v] + prim[ri, rj, rk+1, v]))
                E_sum[v, 3] += d
                E_cnt[v, 3] += 1
                if d > max_E_val[v, 3]
                    max_E_val[v, 3] = d
                    max_E_bid[v, 3] = bid
                end
            end
        end
    end
    
    # Compute relative E_2Δx = (E_avg) / mean(|f|)
    E_rel = zeros(nvar, ndir)
    for v in 1:nvar
        f_mean = F_cnt[v] > 0 ? F_sum[v] / F_cnt[v] : 1.0
        for d in 1:ndir
            E_avg = E_cnt[v, d] > 0 ? E_sum[v, d] / E_cnt[v, d] : 0.0
            E_rel[v, d] = f_mean > 0.0 ? E_avg / f_mean : 0.0
        end
    end
    
    # Find global max
    glob_max_val = 0.0
    glob_max_var = 1
    glob_max_dir = 1
    glob_max_bid = 0
    for v in 1:nvar, d in 1:ndir
        if E_rel[v, d] > glob_max_val
            glob_max_val = E_rel[v, d]
            glob_max_var = v
            glob_max_dir = d
            # For the bid, use the block that had the largest pointwise E_2Δx
            glob_max_bid = max_E_bid[v, d]
        end
    end
    
    threshold = 0.005  # 0.5%
    passed = glob_max_val < threshold
    
    # Print table
    @printf("\n── Full-Field 2Δx Energy (relative to field mean) ──\n")
    @printf("  %-10s  %-12s %-12s %-12s\n", "Variable", "ξ (x)", "η (y)", "ζ (z)")
    @printf("  %s\n", "─"^48)
    for v in 1:nvar
        @printf("  %-10s  %12.2e %12.2e %12.2e\n",
                var_names[v], E_rel[v, 1], E_rel[v, 2], E_rel[v, 3])
    end
    @printf("\n  Max relative E_2Δx: %.2e (%s, %s, bid=%d)\n",
            glob_max_val, var_names[glob_max_var], dir_names[glob_max_dir], glob_max_bid)
    if passed
        @printf("  RESULT: PASS  (threshold: 0.5%%)\n")
    else
        @printf("  RESULT: FAIL  (threshold: 0.5%%, max = %.2e)\n", glob_max_val)
    end
    
    # ── Test 5: Cross-type vs Same-type 2Δx energy comparison ──
    # For each interface face, compute 2Δx energy in the first 4 interior cells
    # E_2dx = Σ |U[j] - 0.5*(U[j-1] + U[j+1])|²  (averaged over the face)
    
    cross_type_energies = Float64[]
    same_type_energies = Float64[]
    cross_type_labels = String[]
    same_type_labels = String[]
    
    for ((dst_bid, fid), conn) in connectivity
        if !haskey(blocks, dst_bid); continue; end
        b = blocks[dst_bid]
        src_fid = conn.src_f
        
        is_cross = (src_fid in (3,4)) != (fid in (3,4))  # η↔ζ swap
        label = "B$(dst_bid)_f$(fid)←B$(conn.src_b)_f$(src_fid)"
        
        # Download U to CPU
        U_cpu = Array(b.U)
        Nx, Ny, Nz = b.Nx, b.Ny, b.Nz
        
        # Compute 2Δx energy near this face for density (variable 1)
        n = 1  # density
        e_2dx = 0.0
        count = 0
        
        if fid == 3  # η- face: j = NG+1 to NG+4
            for layer in 1:4
                jc = ng + layer
                for ic in (ng+1):(Nx+ng), kc in (ng+1):(Nz+ng)
                    # 2Δx check along j (the interface-normal direction)
                    if jc > 1 && jc < size(U_cpu, 2)
                        d = U_cpu[ic, jc, kc, n] - 0.5*(U_cpu[ic, jc-1, kc, n] + U_cpu[ic, jc+1, kc, n])
                        e_2dx += d^2
                        count += 1
                    end
                end
            end
        elseif fid == 4  # η+ face: j = Ny+NG-3 to Ny+NG
            for layer in 0:3
                jc = Ny + ng - layer
                for ic in (ng+1):(Nx+ng), kc in (ng+1):(Nz+ng)
                    if jc > 1 && jc < size(U_cpu, 2)
                        d = U_cpu[ic, jc, kc, n] - 0.5*(U_cpu[ic, jc-1, kc, n] + U_cpu[ic, jc+1, kc, n])
                        e_2dx += d^2
                        count += 1
                    end
                end
            end
        elseif fid == 5  # ζ- face: k = NG+1 to NG+4
            for layer in 1:4
                kc = ng + layer
                for ic in (ng+1):(Nx+ng), jc in (ng+1):(Ny+ng)
                    if kc > 1 && kc < size(U_cpu, 3)
                        d = U_cpu[ic, jc, kc, n] - 0.5*(U_cpu[ic, jc, kc-1, n] + U_cpu[ic, jc, kc+1, n])
                        e_2dx += d^2
                        count += 1
                    end
                end
            end
        elseif fid == 6  # ζ+ face: k = Nz+NG-3 to Nz+NG
            for layer in 0:3
                kc = Nz + ng - layer
                for ic in (ng+1):(Nx+ng), jc in (ng+1):(Ny+ng)
                    if kc > 1 && kc < size(U_cpu, 3)
                        d = U_cpu[ic, jc, kc, n] - 0.5*(U_cpu[ic, jc, kc-1, n] + U_cpu[ic, jc, kc+1, n])
                        e_2dx += d^2
                        count += 1
                    end
                end
            end
        else
            # ξ faces (fid 1,2): skip for now (periodic, no cross-type)
            continue
        end
        
        e_2dx_avg = count > 0 ? e_2dx / count : 0.0
        
        if is_cross
            push!(cross_type_energies, e_2dx_avg)
            push!(cross_type_labels, label)
        else
            push!(same_type_energies, e_2dx_avg)
            push!(same_type_labels, label)
        end
    end
    
    # ── Test 1: Ghost vs Interior comparison at cross-type faces ──
    ghost_diffs = Dict{String, Float64}()
    
    for ((dst_bid, fid), conn) in connectivity
        if !haskey(blocks, dst_bid); continue; end
        src_bid = conn.src_b
        if !haskey(blocks, src_bid); continue; end  # src must also be local
        
        src_fid = conn.src_f
        is_cross = (src_fid in (3,4)) != (fid in (3,4))
        if !is_cross; continue; end  # only check cross-type
        
        b_dst = blocks[dst_bid]
        b_src = blocks[src_bid]
        U_dst = Array(b_dst.U)
        U_src = Array(b_src.U)
        Nx_d, Ny_d, Nz_d = b_dst.Nx, b_dst.Ny, b_dst.Nz
        Nx_s, Ny_s, Nz_s = b_src.Nx, b_src.Ny, b_src.Nz
        
        label = "B$(dst_bid)_f$(fid)←B$(src_bid)_f$(src_fid)"
        n = 1  # density
        
        # Compare ghost layer 1 of dst with interior layer 1 of src
        max_diff = 0.0
        mean_diff = 0.0
        diff_count = 0
        
        if fid == 3  # dst η- ghost: j = NG (first ghost layer)
            j_ghost = ng  # ghost layer
            # This ghost should contain values from src's interior
            if src_fid == 6  # src ζ+ → dst η-
                k_src = Nz_s + ng  # last real cell of src in ζ
                for ic in (ng+1):min(Nx_d+ng, Nx_s+ng)
                    for kc in (ng+1):min(Nz_d+ng, Ny_s+ng)
                        # The exact mapping depends on cross-type transpose
                        v_ghost = U_dst[ic, j_ghost, kc, n]
                        v_real  = U_src[ic, kc, k_src, n]  # cross: j↔k swap
                        d = abs(v_ghost - v_real)
                        max_diff = max(max_diff, d)
                        mean_diff += d
                        diff_count += 1
                    end
                end
            elseif src_fid == 5  # src ζ- → dst η-
                k_src = ng + 1
                for ic in (ng+1):min(Nx_d+ng, Nx_s+ng)
                    for kc in (ng+1):min(Nz_d+ng, Ny_s+ng)
                        v_ghost = U_dst[ic, j_ghost, kc, n]
                        v_real  = U_src[ic, kc, k_src, n]
                        d = abs(v_ghost - v_real)
                        max_diff = max(max_diff, d)
                        mean_diff += d
                        diff_count += 1
                    end
                end
            end
        end
        # (Other fid cases follow same pattern — omitted for brevity in this first version)
        
        if diff_count > 0
            ghost_diffs[label] = mean_diff / diff_count
        end
    end
    
    # ── Print results ──
    @printf("\n── Test 5: 2Δx Energy — Cross-type vs Same-type ──\n")
    @printf("  %-40s  %12s  %s\n", "Interface", "E_2Δx", "Type")
    @printf("  %s\n", "─"^65)
    for (i, e) in enumerate(cross_type_energies)
        @printf("  %-40s  %12.4e  CROSS\n", cross_type_labels[i], e)
    end
    for (i, e) in enumerate(same_type_energies)
        @printf("  %-40s  %12.4e  SAME\n", same_type_labels[i], e)
    end
    
    if !isempty(cross_type_energies) && !isempty(same_type_energies)
        mean_cross = sum(cross_type_energies) / length(cross_type_energies)
        mean_same = sum(same_type_energies) / length(same_type_energies)
        ratio = mean_same > 0 ? mean_cross / mean_same : Inf
        @printf("\n  Mean cross-type E_2Δx: %.4e\n", mean_cross)
        @printf("  Mean same-type  E_2Δx: %.4e\n", mean_same)
        @printf("  Ratio (cross/same):    %.2f\n", ratio)
        if ratio > 10.0
            @printf("  → CONCLUSION: Checkerboard is CROSS-TYPE SPECIFIC (ratio > 10)\n")
        elseif ratio > 2.0
            @printf("  → CONCLUSION: Cross-type is elevated but same-type also present\n")
        else
            @printf("  → CONCLUSION: Both types have similar E_2Δx — issue is NOT cross-type specific\n")
        end
    end
    
    if !isempty(ghost_diffs)
        @printf("\n── Test 1: Ghost vs Interior (cross-type faces only) ──\n")
        for (label, d) in ghost_diffs
            @printf("  %-40s  mean|δρ| = %.4e\n", label, d)
        end
    end
    
    @printf("\n── Checkerboard diagnostic complete ──\n")
end
