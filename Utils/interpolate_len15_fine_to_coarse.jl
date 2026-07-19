using HDF5
using Printf

# =============================================================================
# OpenCFD-FVM: Checkpoint Interpolation from MESH_LEN15 to MESH_LEN15_COARSE
# =============================================================================

const Nblocks = 5
const FT = Float32

const MESH_OLD_DIR = "MESH_LEN15"
const MESH_NEW_DIR = "MESH_LEN15_COARSE"
const CHK_OLD_DIR  = "CHK"
const CHK_NEW_DIR  = "CHK_LEN15_COARSE"

const CHK_STEP = 0
const OUT_STEP = 0

# ─── Cell centers from vertices ───
function vertex_to_cell_centers_2d(yv, zv, i::Int)
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

# ─── Precompute 2D mapping ───
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
        
        j_guess = clamp(round(Int, (j_new - 0.5) / Ny_new * Ny_old + 0.5), 1, Ny_old)
        k_guess = clamp(round(Int, (k_new - 0.5) / Nz_new * Nz_old + 0.5), 1, Nz_old)
        
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
        
        j0 = clamp(best_j, 1, Ny_old - 1)
        k0 = clamp(best_k, 1, Nz_old - 1)
        
        best_err = Inf
        best_j0, best_k0 = j0, k0
        best_s, best_t = 0.5, 0.5
        
        for dj in -2:1, dk in -2:1
            jj0 = j0 + dj
            kk0 = k0 + dk
            if jj0 < 1 || jj0 >= Ny_old || kk0 < 1 || kk0 >= Nz_old
                continue
            end
            
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
    
    # Compute cell centers
    i_ref_old = Nx_old ÷ 2
    i_ref_new = Nx_new ÷ 2
    yc_old, zc_old = vertex_to_cell_centers_2d(yv_old, zv_old, i_ref_old)
    yc_new, zc_new = vertex_to_cell_centers_2d(yv_new, zv_new, i_ref_new)
    
    println("  Building 2D mapping map...")
    t0 = time()
    mapping = build_interp_map(yc_old, zc_old, yc_new, zc_new)
    @printf("  Map built in %.1f s\n", time() - t0)
    
    # Read partition-independent checkpoint directly
    chk_old_file = joinpath(CHK_OLD_DIR, "chk-$(CHK_STEP)-b$(bid).h5")
    println("  Reading partition-independent checkfile: $chk_old_file")
    Q_old = h5read(chk_old_file, "Q")
    nprim = size(Q_old, 4)
    println("  Loaded Q_old: $(size(Q_old))")
    
    # Interpolate
    Q_new = zeros(FT, Nx_new, Ny_new, Nz_new, nprim)
    
    println("  Interpolating state...")
    for i in 1:Nx_new
        i_old = clamp(round(Int, (i - 0.5) / Nx_new * Nx_old + 0.5), 1, Nx_old)
        
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
    end
    
    # Write to new checkfile
    mkpath(CHK_NEW_DIR)
    chk_new_file = joinpath(CHK_NEW_DIR, "chk-$(OUT_STEP)-b$(bid).h5")
    h5open(chk_new_file, "w") do f
        write(f, "Q", Q_new)
        write(f, "step", Int64(OUT_STEP))
        write(f, "time", Float64(0.0))
    end
    println("  ✓ Stored: $chk_new_file ($(Nx_new)×$(Ny_new)×$(Nz_new)×$nprim)")
end

function main()
    for bid in 0:(Nblocks-1)
        interpolate_block(bid)
    end
    println("✓ Done!")
end

main()
