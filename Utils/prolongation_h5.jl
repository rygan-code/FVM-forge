using HDF5
using Printf

# =============================================================================
# OpenCFD-FVM-CUDA: Offline Coarse-to-Fine Grid Prolongation
# 
# Usage:
# Run this script directly with `julia prolongation_h5.jl`.
# It will read the coarse mesh checkpoint and linearly interpolate 
# the primitive variables (rho, u, v, w, p) onto the corresponding fine mesh blocks.
# =============================================================================

# --- Configuration ---
const NG = 3                     # Number of ghost cells padding the mesh arrays
const Nprim = 5                  # Primitive variables: rho, u, v, w, p
const Nblocks = 5                # Number of blocks in the butterfly grid

const MESH_COARSE_DIR = "../MESH_COARSE"
const MESH_FINE_DIR   = "../MESH_FINE"
const CHK_COARSE_DIR  = "../CHK_COARSE"
const CHK_FINE_DIR    = "../CHK_FINE"

# The coarse checkpoint timestep you want to upscale (e.g., 100000)
const COARSE_STEP = 50000 
const OUT_FINE_STEP = 0

function prolongate_block(bid::Int)
    println("Processing Block $bid ...")
    
    # 1. Read the dimensions from the Coarse and Fine Meshes
    mesh_coarse_file = joinpath(MESH_COARSE_DIR, "mesh_b$(bid).h5")
    mesh_fine_file   = joinpath(MESH_FINE_DIR, "mesh_b$(bid).h5")
    
    Nx_c = h5read(mesh_coarse_file, "Nx")
    Ny_c = h5read(mesh_coarse_file, "Ny")
    Nz_c = h5read(mesh_coarse_file, "Nz")
    
    cf_Nx = h5read(mesh_fine_file, "Nx")
    cf_Ny = h5read(mesh_fine_file, "Ny")
    cf_Nz = h5read(mesh_fine_file, "Nz")
    
    println("  - Coarse Grid interior: $(Nx_c) x $(Ny_c) x $(Nz_c)")
    println("  - Fine Grid interior:   $(cf_Nx) x $(cf_Ny) x $(cf_Nz)")
    
    # 2. Read the Coarse Checkpoint (Assuming Single-Rank layout for coarse: n_procs=1)
    chk_coarse_file = joinpath(CHK_COARSE_DIR, "chk-$(COARSE_STEP)-b$(bid).h5")
    
    # Q_h_coarse has shape (Nx_c+2NG, Ny_c+2NG, Nz_c+2NG, Nprim, n_procs)
    Q_h_coarse_full = h5read(chk_coarse_file, "Q_h")
    
    # Extract the block data (assume n_procs = 1, since coarse mesh is small, it runs on Single GPU typically)
    # If coarse was run parallel, you must stitch it manually. Assuming mapped 1-to-1 here.
    Q_c = Q_h_coarse_full[:, :, :, 1:Nprim, 1] 
    
    # 3. Create Fine Grid Array
    Nx_f_tot = cf_Nx + 2*NG
    Ny_f_tot = cf_Ny + 2*NG
    Nz_f_tot = cf_Nz + 2*NG
    
    Q_f = zeros(FT, Nx_f_tot, Ny_f_tot, Nz_f_tot, Nprim)
    
    # 4. Trilinear Interpolation in Computational Computational Space
    # FVM values are stored at cell centers.
    # Uniform computational space xi, eta, zeta in [0, 1].
    for k in 1:cf_Nz
        for j in 1:cf_Ny
            for i in 1:cf_Nx
                # Relative computational coordinates of the fine cell center
                u = (i - FT(0.5)) / FT(cf_Nx)
                v = (j - FT(0.5)) / FT(cf_Ny)
                w = (k - FT(0.5)) / FT(cf_Nz)
                
                # Mapped logical coordinate in coarse grid 
                ic_exact = u * FT(Nx_c) + FT(0.5)
                jc_exact = v * FT(Ny_c) + FT(0.5)
                kc_exact = w * FT(Nz_c) + FT(0.5)
                
                # Integer base indices
                i0 = floor(Int, ic_exact); dx = ic_exact - FT(i0)
                j0 = floor(Int, jc_exact); dy = jc_exact - FT(j0)
                k0 = floor(Int, kc_exact); dz = kc_exact - FT(k0)
                
                # Target Julia indices (accounting for NG padding offset)
                idx0 = i0 + NG; idx1 = idx0 + 1
                jdy0 = j0 + NG; jdy1 = jdy0 + 1
                kdz0 = k0 + NG; kdz1 = kdz0 + 1
                
                # 3D Trilinear weighting for each primitive variable
                for n in 1:Nprim
                    c000 = Q_c[idx0, jdy0, kdz0, n]; c100 = Q_c[idx1, jdy0, kdz0, n]
                    c010 = Q_c[idx0, jdy1, kdz0, n]; c110 = Q_c[idx1, jdy1, kdz0, n]
                    c001 = Q_c[idx0, jdy0, kdz1, n]; c101 = Q_c[idx1, jdy0, kdz1, n]
                    c011 = Q_c[idx0, jdy1, kdz1, n]; c111 = Q_c[idx1, jdy1, kdz1, n]
                    
                    c00 = c000 * (one(FT) - dx) + c100 * dx
                    c10 = c010 * (one(FT) - dx) + c110 * dx
                    c01 = c001 * (one(FT) - dx) + c101 * dx
                    c11 = c011 * (one(FT) - dx) + c111 * dx
                    
                    c0 = c00 * (one(FT) - dy) + c10 * dy
                    c1 = c01 * (one(FT) - dy) + c11 * dy
                    
                    val = c0 * (one(FT) - dz) + c1 * dz
                    
                    Q_f[i + NG, j + NG, k + NG, n] = val
                end
            end
        end
    end
    
    # 5. Populate Ghost cells with zero-gradient copy
    # Actual boundaries will be enforced automatically by OpenCFD-FVM `fillGhost` upon restart loading.
    for n in 1:Nprim
        for g in 1:NG
            Q_f[g, :, :, n] .= Q_f[NG+1, :, :, n]
            Q_f[end-g+1, :, :, n] .= Q_f[end-NG, :, :, n]
            
            Q_f[:, g, :, n] .= Q_f[:, NG+1, :, n]
            Q_f[:, end-g+1, :, n] .= Q_f[:, end-NG, :, n]
            
            Q_f[:, :, g, n] .= Q_f[:, :, NG+1, n]
            Q_f[:, :, end-g+1, n] .= Q_f[:, :, end-NG, n]
        end
    end
    
    # 6. Save the High-Res Interpolated Checkpoint
    # We save exactly as `single GPU` block (n_procs = 1).
    # If the fine mesh runs on N-to-M parallel ranks, `solver.jl` must load this single array 
    # and slice it dynamically across `my_rx`, `my_ry`, etc!
    mkpath(CHK_FINE_DIR)
    chk_fine_file = joinpath(CHK_FINE_DIR, "chk-$(OUT_FINE_STEP)-b$(bid).h5")
    
    h5open(chk_fine_file, "w") do f
        # Re-pack the 5D array (Nx, Ny, Nz, Nprim, n_procs=1)
        Q_f_out = reshape(Q_f, size(Q_f)..., 1)
        
        # Enable compression dynamically
        dset = create_dataset(
            f, "Q_h", datatype(Float32),
            dataspace(Nx_f_tot, Ny_f_tot, Nz_f_tot, Nprim, 1);
            chunk=(Nx_f_tot, Ny_f_tot, Nz_f_tot, Nprim, 1),
            compress=3
        )
        dset[:, :, :, :, 1] = Q_f_out
    end
    println("  - Saved natively to: $chk_fine_file")
end

function main()
    println("=========================================================")
    println(" Flame3D / OpenCFD-FVM : Static Coarse->Fine Interpolator")
    println("=========================================================")
    for bid in 0:(Nblocks-1)
        prolongate_block(bid)
    end
    println("=========================================================")
    println(" Prolongation Complete! Ready to launch DNS.")
end

main()
