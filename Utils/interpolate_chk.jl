using HDF5
using Printf

# =============================================================================
# OpenCFD-FVM: Fast Mesh-to-Mesh Checkpoint Interpolation
#
# Optimized: Precomputes 2D (j,k) mapping table once per block using
# the first axial station, then reuses for all 512 stations.
# =============================================================================

const NG = 4
const Nblocks = 5

const MESH_OLD_DIR = "MESH_COARSE_OLD"
const MESH_NEW_DIR = "MESH_COARSE"
const CHK_OLD_DIR  = "CHK_OLD"
const CHK_NEW_DIR  = "CHK"

const CHK_STEP = 547000
const OUT_STEP = 0

# ─── Cell centers from vertices ───
function vertex_to_cell_centers_2d(yv, zv, i::Int)
    # Extract the 2D (j,k) plane cell centers at axial index i
    ny, nz = size(yv, 2) - 1, size(yv, 3) - 1
    yc = zeros(Float64, ny, nz)
    zc = zeros(Float64, ny, nz)
    for k in 1:nz, j in 1:ny
        yc[j,k] = 0.25 * (yv[i,j,k] + yv[i+1,j,k] + yv[i,j+1,k] + yv[i+1,j+1,k] +
                           yv[i,j,k+1] + yv[i+1,j,k+1] + yv[i,j+1,k+1] + yv[i+1,j+1,k+1]) / 2.0
        zc[j,k] = 0.25 * (zv[i,j,k] + zv[i+1,j,k] + zv[i,j+1,k] + zv[i+1,j+1,k] +
                           zv[i,j,k+1] + zv[i+1,j,k+1] + zv[i,j+1,k+1] + zv[i+1,j+1,k+1]) / 2.0
    end
    return yc, zc
end

function vertex_to_cell_centers_3d(xv)
    nx, ny, nz = size(xv) .- 1
    xc = zeros(Float64, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        xc[i,j,k] = 0.125 * (xv[i,j,k] + xv[i+1,j,k] + xv[i,j+1,k] + xv[i+1,j+1,k] +
                              xv[i,j,k+1] + xv[i+1,j,k+1] + xv[i,j+1,k+1] + xv[i+1,j+1,k+1])
    end
    return xc
end

# ─── Stitch per-rank CHK into global Q ───
function stitch_checkpoint(chk_file, Nx, Ny, Nz)
    h5open(chk_file, "r") do f
        n_procs = read(f["n_procs"])
        Q0 = read(f["Q_r0"])
        nprim = size(Q0, 4)
        
        Q_global = zeros(FT, Nx, Ny, Nz, nprim)
        
        i_offset = 0
        for r in 0:(n_procs-1)
            Qr = read(f["Q_r$r"])
            # Detect which dimension was split by checking sizes
            nx_r = size(Qr, 1) - 2*NG
            ny_r = size(Qr, 2) - 2*NG
            nz_r = size(Qr, 3) - 2*NG
            
            if ny_r == Ny && nz_r == Nz
                # Split along ξ (most common)
                Q_global[i_offset+1:i_offset+nx_r, :, :, :] = Qr[NG+1:NG+nx_r, NG+1:NG+Ny, NG+1:NG+Nz, :]
                i_offset += nx_r
            else
                error("Unexpected partition layout: rank $r has ($nx_r,$ny_r,$nz_r) vs expected (?,$(Ny),$(Nz))")
            end
        end
        
        @assert i_offset == Nx "Stitched $i_offset != Nx=$Nx"
        return Q_global, nprim
    end
end

# ─── Precompute 2D mapping: for each new (j,k), find old (j0,k0,s,t) ───
struct InterpMap
    j0::Int
    k0::Int
    s::Float64
    t::Float64
end

function build_interp_map(yc_old, zc_old, yc_new, zc_new)
    Ny_old, Nz_old = size(yc_old)
    Ny_new, Nz_new = size(yc_new)
    
    mapping = Matrix{InterpMap}(undef, Ny_new, Nz_new)
    
    for k_new in 1:Nz_new, j_new in 1:Ny_new
        y_target = yc_new[j_new, k_new]
        z_target = zc_new[j_new, k_new]
        
        # Initial guess from computational space
        j_guess = clamp(round(Int, (j_new - 0.5) / Ny_new * Ny_old + 0.5), 1, Ny_old)
        k_guess = clamp(round(Int, (k_new - 0.5) / Nz_new * Nz_old + 0.5), 1, Nz_old)
        
        # Full search in old 2D grid (done once, so OK)
        best_j, best_k = j_guess, k_guess
        best_dist = Inf
        
        for kk in 1:Nz_old, jj in 1:Ny_old
            dy = yc_old[jj, kk] - y_target
            dz = zc_old[jj, kk] - z_target
            dist = dy*dy + dz*dz
            if dist < best_dist
                best_dist = dist
                best_j = jj
                best_k = kk
            end
        end
        
        # Find the cell that contains this point (j0, k0)
        j0 = clamp(best_j, 1, Ny_old - 1)
        k0 = clamp(best_k, 1, Nz_old - 1)
        
        # Refine: check neighboring cells to find best enclosing quad
        best_err = Inf
        best_j0, best_k0 = j0, k0
        best_s, best_t = 0.5, 0.5
        
        for dj in -2:1, dk in -2:1
            jj0 = j0 + dj
            kk0 = k0 + dk
            if jj0 < 1 || jj0 >= Ny_old || kk0 < 1 || kk0 >= Nz_old
                continue
            end
            
            # Bilinear inverse for this quad
            y00 = yc_old[jj0, kk0];     z00 = zc_old[jj0, kk0]
            y10 = yc_old[jj0+1, kk0];   z10 = zc_old[jj0+1, kk0]
            y01 = yc_old[jj0, kk0+1];   z01 = zc_old[jj0, kk0+1]
            y11 = yc_old[jj0+1, kk0+1]; z11 = zc_old[jj0+1, kk0+1]
            
            s, t = 0.5, 0.5
            for iter in 1:15
                y_e = (1-s)*(1-t)*y00 + s*(1-t)*y10 + (1-s)*t*y01 + s*t*y11
                z_e = (1-s)*(1-t)*z00 + s*(1-t)*z10 + (1-s)*t*z01 + s*t*z11
                ry = y_target - y_e
                rz = z_target - z_e
                err = ry*ry + rz*rz
                if err < 1e-24; break; end
                
                dyds = -(1-t)*y00 + (1-t)*y10 - t*y01 + t*y11
                dydt = -(1-s)*y00 - s*y10 + (1-s)*y01 + s*y11
                dzds = -(1-t)*z00 + (1-t)*z10 - t*z01 + t*z11
                dzdt = -(1-s)*z00 - s*z10 + (1-s)*z01 + s*z11
                det = dyds*dzdt - dydt*dzds
                if abs(det) < 1e-30; break; end
                
                s += ( dzdt*ry - dydt*rz) / det
                t += (-dzds*ry + dyds*rz) / det
            end
            
            # If within [0,1], this is the enclosing quad
            if s >= -0.01 && s <= 1.01 && t >= -0.01 && t <= 1.01
                y_e = (1-s)*(1-t)*y00 + s*(1-t)*y10 + (1-s)*t*y01 + s*t*y11
                z_e = (1-s)*(1-t)*z00 + s*(1-t)*z10 + (1-s)*t*z01 + s*t*z11
                err = (y_target - y_e)^2 + (z_target - z_e)^2
                if err < best_err
                    best_err = err
                    best_j0 = jj0
                    best_k0 = kk0
                    best_s = clamp(s, 0.0, 1.0)
                    best_t = clamp(t, 0.0, 1.0)
                end
            end
        end
        
        mapping[j_new, k_new] = InterpMap(best_j0, best_k0, best_s, best_t)
    end
    
    return mapping
end

# ─── Interpolate one block ───
function interpolate_block(bid::Int)
    println("═" ^ 60)
    println("  Processing Block $bid ...")
    
    mesh_old_file = joinpath(MESH_OLD_DIR, "mesh_b$(bid).h5")
    mesh_new_file = joinpath(MESH_NEW_DIR, "mesh_b$(bid).h5")
    
    Nx_old = h5read(mesh_old_file, "Nx")
    Ny_old = h5read(mesh_old_file, "Ny")
    Nz_old = h5read(mesh_old_file, "Nz")
    
    Nx_new = h5read(mesh_new_file, "Nx")
    Ny_new = h5read(mesh_new_file, "Ny")
    Nz_new = h5read(mesh_new_file, "Nz")
    
    println("  Old grid: $(Nx_old) × $(Ny_old) × $(Nz_old)")
    println("  New grid: $(Nx_new) × $(Ny_new) × $(Nz_new)")
    
    # Read vertex coords
    yv_old = Float64.(h5read(mesh_old_file, "y"))
    zv_old = Float64.(h5read(mesh_old_file, "z"))
    yv_new = Float64.(h5read(mesh_new_file, "y"))
    zv_new = Float64.(h5read(mesh_new_file, "z"))
    
    # Compute 2D cell centers at a reference axial station (mid-plane)
    i_ref = Nx_old ÷ 2
    yc_old, zc_old = vertex_to_cell_centers_2d(yv_old, zv_old, i_ref)
    yc_new, zc_new = vertex_to_cell_centers_2d(yv_new, zv_new, i_ref)
    
    # Precompute mapping (done once!)
    println("  Building 2D interpolation map ($(Ny_new)×$(Nz_new)) ...")
    t0 = time()
    mapping = build_interp_map(yc_old, zc_old, yc_new, zc_new)
    @printf("  Map built in %.1f s\n", time() - t0)
    
    # Read and stitch OLD checkpoint
    chk_old_file = joinpath(CHK_OLD_DIR, "chk-$(CHK_STEP)-b$(bid).h5")
    println("  Reading checkpoint: $chk_old_file")
    Q_old, nprim = stitch_checkpoint(chk_old_file, Nx_old, Ny_old, Nz_old)
    println("  Stitched Q_old: $(size(Q_old))")
    
    # Allocate new Q
    Q_new = zeros(FT, Nx_new, Ny_new, Nz_new, nprim)
    
    # Apply mapping for all axial stations
    println("  Interpolating ...")
    for i in 1:Nx_new
        i_old = if Nx_old == Nx_new
            i
        else
            clamp(round(Int, (i - 0.5) / Nx_new * Nx_old + 0.5), 1, Nx_old)
        end
        
        for k_new in 1:Nz_new, j_new in 1:Ny_new
            m = mapping[j_new, k_new]
            s, t = m.s, m.t
            j0, k0 = m.j0, m.k0
            
            for n in 1:nprim
                q00 = Q_old[i_old, j0,   k0,   n]
                q10 = Q_old[i_old, j0+1, k0,   n]
                q01 = Q_old[i_old, j0,   k0+1, n]
                q11 = Q_old[i_old, j0+1, k0+1, n]
                
                Q_new[i, j_new, k_new, n] = FT(
                    (1-s)*(1-t)*q00 + s*(1-t)*q10 + (1-s)*t*q01 + s*t*q11
                )
            end
        end
        
        if i % 128 == 0
            @printf("    Block %d: %d/%d axial stations\n", bid, i, Nx_new)
        end
    end
    
    # Save in new partition-independent format (interior only, no ghost)
    mkpath(CHK_NEW_DIR)
    chk_new_file = joinpath(CHK_NEW_DIR, "chk-$(OUT_STEP)-b$(bid).h5")
    
    h5open(chk_new_file, "w") do f
        write(f, "Q", Q_new)      # (Nx, Ny, Nz, Nprim) — interior only
        write(f, "step", Int64(OUT_STEP))
        write(f, "time", Float64(0.0))
    end
    
    println("  ✓ Saved: $chk_new_file  ($(Nx_new)×$(Ny_new)×$(Nz_new)×$nprim) [partition-independent]")
end

function main()
    println("═" ^ 60)
    println("  Flame3D: Fast Physical-Space Mesh Interpolation")
    println("═" ^ 60)
    println("  Old mesh: $MESH_OLD_DIR")
    println("  New mesh: $MESH_NEW_DIR")
    println("  Old CHK:  $CHK_OLD_DIR (step $CHK_STEP)")
    println("  Output:   $CHK_NEW_DIR (step $OUT_STEP)")
    println()
    
    for bid in 0:(Nblocks-1)
        interpolate_block(bid)
    end
    
    println()
    println("═" ^ 60)
    println("  ✓ Interpolation Complete!")
    println("  Set restart = \"$OUT_STEP\" in run/baseline/pipe_baseline.jl")
    println("═" ^ 60)
end

main()
