# =============================================================================
# convert_plot3d.jl — Convert Plot3D / Pointwise mesh to Flame3D HDF5 format
#
# Usage:
#   julia convert_plot3d.jl <grid_file.xyz> <config_file.toml>
#
# Reads a multi-block Plot3D grid file and a TOML config describing
# connectivity + boundary conditions, then outputs:
#   MESH/mesh_b*.h5           — coordinates (with ghost cells)
#   MESH/metrics_b*.h5        — FVM face areas, normals, cell volumes
#   MESH/block_connectivity.h5 — connectivity, reverse_tan, face_bc
#   MESH/interp_weights.h5    — bilinear interpolation weights for ghost cells
#
# Config file format (TOML):
# ─────────────────────────────
#   NG = 4
#   
#   # Connectivity: each row = [block1, face1, block2, face2]
#   # Faces: 1=ξ-, 2=ξ+, 3=η-, 4=η+, 5=ζ-, 6=ζ+
#   [[connectivity]]
#   blocks = [0, 3, 1, 4]
#   
#   [[connectivity]]
#   blocks = [0, 4, 2, 3]
#   
#   # Boundary conditions: face_bc[block][face] = type
#   # Types: "interblock"=0, "wall"=1, "periodic"=2, "inflow"=3, "outflow"=4, "symmetry"=5
#   [face_bc]
#   # block 0: ξ± periodic, η± interblock, ζ± interblock
#   0 = ["periodic","periodic","interblock","interblock","interblock","interblock"]
#   1 = ["periodic","periodic","wall","interblock","interblock","interblock"]
# =============================================================================

using LinearAlgebra, HDF5, Printf
import TOML

# ─── Shared BC Constants ───
include("../bc_types.jl")

# Mapping from TOML bc_params key names to BCP slot indices
const BCP_KEY_MAP = Dict{String, Int}(
    "Tw"        => BCP_TW,
    "p_target"  => BCP_P_TARGET,
    "rho_inf"   => BCP_RHO_INF,
    "u_inf"     => BCP_U_INF,
    "v_inf"     => BCP_V_INF,
    "w_inf"     => BCP_W_INF,
    "p_inf"     => BCP_P_INF,
    "sigma"     => BCP_SIGMA,
    "L_ref"     => BCP_LREF,
    "P0"        => BCP_P0,
    "T0"        => BCP_T0,
    "dir_x"     => BCP_DIR_X,
    "dir_y"     => BCP_DIR_Y,
    "dir_z"     => BCP_DIR_Z,
)

# =============================================================================
# 1. Plot3D Reader (binary, multi-block, double/single precision)
# =============================================================================
function read_plot3d(filepath; formatted=false, precision=Float64)
    if formatted
        return read_plot3d_formatted(filepath)
    else
        return read_plot3d_binary(filepath, precision)
    end
end

function read_plot3d_binary(filepath, precision)
    blocks = []
    open(filepath, "r") do io
        # Read number of blocks
        nblocks = read(io, Int32)
        println("  Number of blocks: $nblocks")
        
        # Read dimensions for each block
        dims = Vector{NTuple{3,Int}}(undef, nblocks)
        for b in 1:nblocks
            ni = Int(read(io, Int32))
            nj = Int(read(io, Int32))
            nk = Int(read(io, Int32))
            dims[b] = (ni, nj, nk)
            println("  Block $b: $ni × $nj × $nk")
        end
        
        # Read coordinates for each block
        for b in 1:nblocks
            ni, nj, nk = dims[b]
            npts = ni * nj * nk
            x = zeros(Float64, ni, nj, nk)
            y = zeros(Float64, ni, nj, nk)
            z = zeros(Float64, ni, nj, nk)
            
            # Plot3D stores x for all points, then y, then z
            raw = read!(io, Vector{precision}(undef, npts))
            x .= reshape(Float64.(raw), ni, nj, nk)
            raw = read!(io, Vector{precision}(undef, npts))
            y .= reshape(Float64.(raw), ni, nj, nk)
            raw = read!(io, Vector{precision}(undef, npts))
            z .= reshape(Float64.(raw), ni, nj, nk)
            
            push!(blocks, (x, y, z, ni-1, nj-1, nk-1))  # (nodes, Ncells = Nnodes-1)
        end
    end
    return blocks
end

function read_plot3d_formatted(filepath)
    blocks = []
    lines = readlines(filepath)
    idx = 1
    
    nblocks = parse(Int, strip(lines[idx])); idx += 1
    println("  Number of blocks: $nblocks")
    
    dims = Vector{NTuple{3,Int}}(undef, nblocks)
    # Dimensions may be on one line or multiple
    dim_tokens = Float64[]
    while length(dim_tokens) < 3 * nblocks
        append!(dim_tokens, parse.(Float64, split(strip(lines[idx]))))
        idx += 1
    end
    for b in 1:nblocks
        ni = Int(dim_tokens[(b-1)*3+1])
        nj = Int(dim_tokens[(b-1)*3+2])
        nk = Int(dim_tokens[(b-1)*3+3])
        dims[b] = (ni, nj, nk)
        println("  Block $b: $ni × $nj × $nk")
    end
    
    for b in 1:nblocks
        ni, nj, nk = dims[b]
        npts = ni * nj * nk
        
        # Read all coordinate values
        vals = Float64[]
        while length(vals) < 3 * npts
            append!(vals, parse.(Float64, split(strip(lines[idx]))))
            idx += 1
        end
        
        x = reshape(vals[1:npts], ni, nj, nk)
        y = reshape(vals[npts+1:2*npts], ni, nj, nk)
        z = reshape(vals[2*npts+1:3*npts], ni, nj, nk)
        
        push!(blocks, (x, y, z, ni-1, nj-1, nk-1))
    end
    return blocks
end

# =============================================================================
# 2. Ghost Cell Generation (extrapolation from interior)
# =============================================================================
function add_ghost_cells(x_in, y_in, z_in, Nx, Ny, Nz, NG)
    # Input: node arrays (Nx+1, Ny+1, Nz+1)
    # Output: padded node arrays (Nx+2NG+1, Ny+2NG+1, Nz+2NG+1)
    ni, nj, nk = size(x_in)
    
    # Output sizes
    Ni = Nx + 2*NG + 1
    Nj = Ny + 2*NG + 1
    Nk = Nz + 2*NG + 1
    
    x = zeros(Float64, Ni, Nj, Nk)
    y = zeros(Float64, Ni, Nj, Nk)
    z = zeros(Float64, Ni, Nj, Nk)
    
    # Copy interior (nodes at NG+1 : NG+Nx+1 etc.)
    x[NG+1:NG+ni, NG+1:NG+nj, NG+1:NG+nk] .= x_in
    y[NG+1:NG+ni, NG+1:NG+nj, NG+1:NG+nk] .= y_in
    z[NG+1:NG+ni, NG+1:NG+nj, NG+1:NG+nk] .= z_in
    
    # Extrapolate ghost cells (linear extrapolation from the 2 nearest interior nodes)
    for arr in (x, y, z)
        # ξ- direction
        for g in NG:-1:1
            arr[g, NG+1:NG+nj, NG+1:NG+nk] .= 
                2.0 .* arr[g+1, NG+1:NG+nj, NG+1:NG+nk] .- arr[g+2, NG+1:NG+nj, NG+1:NG+nk]
        end
        # ξ+ direction
        for g in NG+ni+1:Ni
            arr[g, NG+1:NG+nj, NG+1:NG+nk] .= 
                2.0 .* arr[g-1, NG+1:NG+nj, NG+1:NG+nk] .- arr[g-2, NG+1:NG+nj, NG+1:NG+nk]
        end
        
        # η- direction (over full ξ range now)
        for g in NG:-1:1
            arr[:, g, NG+1:NG+nk] .= 2.0 .* arr[:, g+1, NG+1:NG+nk] .- arr[:, g+2, NG+1:NG+nk]
        end
        # η+ direction
        for g in NG+nj+1:Nj
            arr[:, g, NG+1:NG+nk] .= 2.0 .* arr[:, g-1, NG+1:NG+nk] .- arr[:, g-2, NG+1:NG+nk]
        end
        
        # ζ- direction (over full ξ,η range now)
        for g in NG:-1:1
            arr[:, :, g] .= 2.0 .* arr[:, :, g+1] .- arr[:, :, g+2]
        end
        # ζ+ direction
        for g in NG+nk+1:Nk
            arr[:, :, g] .= 2.0 .* arr[:, :, g-1] .- arr[:, :, g-2]
        end
    end
    
    return Float32.(x), Float32.(y), Float32.(z)
end

# =============================================================================
# 3. FVM Metrics (reused from gen_butterfly_fvm.jl)
# =============================================================================
function compute_fvm_metrics(x, y, z, Nx, Ny, Nz, NG)
    Nx_nodes_tot = Nx + 2NG + 1; Ny_nodes_tot = Ny + 2NG + 1; Nz_nodes_tot = Nz + 2NG + 1
    Nx_cells_tot = Nx + 2NG;     Ny_cells_tot = Ny + 2NG;     Nz_cells_tot = Nz + 2NG
    
    Ai  = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    nxi = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    nyi = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    nzi = zeros(FT, Nx_nodes_tot, Ny_cells_tot, Nz_cells_tot)
    
    Aj  = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    nxj = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    nyj = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    nzj = zeros(FT, Nx_cells_tot, Ny_nodes_tot, Nz_cells_tot)
    
    Ak  = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    nxk = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    nyk = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    nzk = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_nodes_tot)
    
    V = zeros(FT, Nx_cells_tot, Ny_cells_tot, Nz_cells_tot)
    
    # i-face normals
    for k in 1:Nz_cells_tot, j in 1:Ny_cells_tot, i in 1:Nx_nodes_tot
        v1 = [x[i,j+1,k+1]-x[i,j,k], y[i,j+1,k+1]-y[i,j,k], z[i,j+1,k+1]-z[i,j,k]]
        v2 = [x[i,j,k+1]-x[i,j+1,k], y[i,j,k+1]-y[i,j+1,k], z[i,j,k+1]-z[i,j+1,k]]
        n = 0.5 * cross(v1, v2)
        Ai[i,j,k] = FT(norm(n)) + FT(1e-20)
        nxi[i,j,k] = FT(n[1]) / Ai[i,j,k]
        nyi[i,j,k] = FT(n[2]) / Ai[i,j,k]
        nzi[i,j,k] = FT(n[3]) / Ai[i,j,k]
    end
    
    # j-face normals (negated)
    for k in 1:Nz_cells_tot, j in 1:Ny_nodes_tot, i in 1:Nx_cells_tot
        v1 = [x[i+1,j,k+1]-x[i,j,k], y[i+1,j,k+1]-y[i,j,k], z[i+1,j,k+1]-z[i,j,k]]
        v2 = [x[i,j,k+1]-x[i+1,j,k], y[i,j,k+1]-y[i+1,j,k], z[i,j,k+1]-z[i+1,j,k]]
        n = 0.5 * cross(v1, v2)
        Aj[i,j,k] = FT(norm(n)) + FT(1e-20)
        nxj[i,j,k] = -FT(n[1]) / Aj[i,j,k]
        nyj[i,j,k] = -FT(n[2]) / Aj[i,j,k]
        nzj[i,j,k] = -FT(n[3]) / Aj[i,j,k]
    end
    
    # k-face normals
    for k in 1:Nz_nodes_tot, j in 1:Ny_cells_tot, i in 1:Nx_cells_tot
        v1 = [x[i+1,j+1,k]-x[i,j,k], y[i+1,j+1,k]-y[i,j,k], z[i+1,j+1,k]-z[i,j,k]]
        v2 = [x[i,j+1,k]-x[i+1,j,k], y[i,j+1,k]-y[i+1,j,k], z[i,j+1,k]-z[i+1,j,k]]
        n = 0.5 * cross(v1, v2)
        Ak[i,j,k] = FT(norm(n)) + FT(1e-20)
        nxk[i,j,k] = FT(n[1]) / Ak[i,j,k]
        nyk[i,j,k] = FT(n[2]) / Ak[i,j,k]
        nzk[i,j,k] = FT(n[3]) / Ak[i,j,k]
    end
    
    # Cell volumes (stored as 1/Vol)
    for k in 1:Nz_cells_tot, j in 1:Ny_cells_tot, i in 1:Nx_cells_tot
        dxdξ = 0.25*(x[i+1,j,k]+x[i+1,j+1,k]+x[i+1,j,k+1]+x[i+1,j+1,k+1]-x[i,j,k]-x[i,j+1,k]-x[i,j,k+1]-x[i,j+1,k+1])
        dydξ = 0.25*(y[i+1,j,k]+y[i+1,j+1,k]+y[i+1,j,k+1]+y[i+1,j+1,k+1]-y[i,j,k]-y[i,j+1,k]-y[i,j,k+1]-y[i,j+1,k+1])
        dzdξ = 0.25*(z[i+1,j,k]+z[i+1,j+1,k]+z[i+1,j,k+1]+z[i+1,j+1,k+1]-z[i,j,k]-z[i,j+1,k]-z[i,j,k+1]-z[i,j+1,k+1])
        dxdη = 0.25*(x[i,j+1,k]+x[i+1,j+1,k]+x[i,j+1,k+1]+x[i+1,j+1,k+1]-x[i,j,k]-x[i+1,j,k]-x[i,j,k+1]-x[i+1,j,k+1])
        dydη = 0.25*(y[i,j+1,k]+y[i+1,j+1,k]+y[i,j+1,k+1]+y[i+1,j+1,k+1]-y[i,j,k]-y[i+1,j,k]-y[i,j,k+1]-y[i+1,j,k+1])
        dzdη = 0.25*(z[i,j+1,k]+z[i+1,j+1,k]+z[i,j+1,k+1]+z[i+1,j+1,k+1]-z[i,j,k]-z[i+1,j,k]-z[i,j,k+1]-z[i+1,j,k+1])
        dxdζ = 0.25*(x[i,j,k+1]+x[i+1,j,k+1]+x[i,j+1,k+1]+x[i+1,j+1,k+1]-x[i,j,k]-x[i+1,j,k]-x[i,j+1,k]-x[i+1,j+1,k])
        dydζ = 0.25*(y[i,j,k+1]+y[i+1,j,k+1]+y[i,j+1,k+1]+y[i+1,j+1,k+1]-y[i,j,k]-y[i+1,j,k]-y[i,j+1,k]-y[i+1,j+1,k])
        dzdζ = 0.25*(z[i,j,k+1]+z[i+1,j,k+1]+z[i,j+1,k+1]+z[i+1,j+1,k+1]-z[i,j,k]-z[i+1,j,k]-z[i,j+1,k]-z[i+1,j+1,k])
        vol = abs(dxdξ*(dydη*dzdζ - dydζ*dzdη) - dxdη*(dydξ*dzdζ - dydζ*dzdξ) + dxdζ*(dydξ*dzdη - dydη*dzdξ))
        V[i,j,k] = one(FT) / (FT(vol) + FT(1e-30))
    end
    
    return Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V
end

# =============================================================================
# 4. Connectivity: compute reverse_tan from tangential vectors
# =============================================================================
function get_tangential_vector(x, y, z, fid, Nx, Ny, Nz, NG)
    mid_i = (Nx + 2NG + 1) ÷ 2
    if fid == 3      # η-: tangential = ζ
        j0 = NG + 1; mid_k = (Nz + 2NG + 1) ÷ 2
        return [y[mid_i, j0, mid_k+1] - y[mid_i, j0, mid_k],
                z[mid_i, j0, mid_k+1] - z[mid_i, j0, mid_k]]
    elseif fid == 4  # η+: tangential = ζ
        j0 = Ny + NG + 1; mid_k = (Nz + 2NG + 1) ÷ 2
        return [y[mid_i, j0, mid_k+1] - y[mid_i, j0, mid_k],
                z[mid_i, j0, mid_k+1] - z[mid_i, j0, mid_k]]
    elseif fid == 5  # ζ-: tangential = η
        k0 = NG + 1; mid_j = (Ny + 2NG + 1) ÷ 2
        return [y[mid_i, mid_j+1, k0] - y[mid_i, mid_j, k0],
                z[mid_i, mid_j+1, k0] - z[mid_i, mid_j, k0]]
    elseif fid == 6  # ζ+: tangential = η
        k0 = Nz + NG + 1; mid_j = (Ny + 2NG + 1) ÷ 2
        return [y[mid_i, mid_j+1, k0] - y[mid_i, mid_j, k0],
                z[mid_i, mid_j+1, k0] - z[mid_i, mid_j, k0]]
    elseif fid == 1  # ξ-: tangential = η
        i0 = NG + 1; mid_j = (Ny + 2NG + 1) ÷ 2
        return [y[i0, mid_j+1, 1] - y[i0, mid_j, 1],
                z[i0, mid_j+1, 1] - z[i0, mid_j, 1]]
    elseif fid == 2  # ξ+: tangential = η
        i0 = Nx + NG + 1; mid_j = (Ny + 2NG + 1) ÷ 2
        return [y[i0, mid_j+1, 1] - y[i0, mid_j, 1],
                z[i0, mid_j+1, 1] - z[i0, mid_j, 1]]
    end
end

# =============================================================================
# 5. Interpolation Weights (conformal: direct copy for matching faces)
# =============================================================================
function build_conformal_interp_weights(blocks_data, connectivity_rows, NG)
    """
    For conformal (point-matching) interfaces, ghost cells are filled by
    direct copy from the neighbor block's interior. C1-C4 encode the 4 corners
    of the source cell as linear indices, with W1-W4 = (1, 0, 0, 0) for exact copy.
    """
    L_pack = 10  # search depth in source block
    
    h5open("MESH/interp_weights.h5", "w") do fid
        for row_idx in 1:size(connectivity_rows, 1)
            b1, f1, b2, f2 = connectivity_rows[row_idx, :]
            x_dst, y_dst, z_dst, Nx_dst, Ny_dst, Nz_dst = blocks_data[b1 + 1]
            x_src, y_src, z_src, Nx_src, Ny_src, Nz_src = blocks_data[b2 + 1]
            
            Nx_tot_dst = Nx_dst + 2NG; Ny_tot_dst = Ny_dst + 2NG; Nz_tot_dst = Nz_dst + 2NG
            Nx_tot_src = Nx_src + 2NG; Ny_tot_src = Ny_src + 2NG; Nz_tot_src = Nz_src + 2NG
            
            # Determine ghost cell range in destination block
            if f1 == 3; j_range = 1:NG; k_range = 1:Nz_tot_dst
            elseif f1 == 4; j_range = Ny_tot_dst-NG+1:Ny_tot_dst; k_range = 1:Nz_tot_dst
            elseif f1 == 5; j_range = 1:Ny_tot_dst; k_range = 1:NG
            elseif f1 == 6; j_range = 1:Ny_tot_dst; k_range = Nz_tot_dst-NG+1:Nz_tot_dst
            else continue
            end
            
            # Collect ghost cell center positions
            ghost_pts = []
            for kg in k_range, jg in j_range
                # Cell center in physical space (average of 8 cell nodes)
                yc = 0.0; zc = 0.0
                for di in 0:1, dj in 0:1, dk in 0:1
                    yc += y_dst[NG+1+di, jg+dj, kg+dk]
                    zc += z_dst[NG+1+di, jg+dj, kg+dk]
                end
                yc /= 8.0; zc /= 8.0
                push!(ghost_pts, (jg, kg, yc, zc))
            end
            
            num_pts = length(ghost_pts)
            C1 = zeros(Int32, num_pts); C2 = zeros(Int32, num_pts)
            C3 = zeros(Int32, num_pts); C4 = zeros(Int32, num_pts)
            W1 = ones(Float32, num_pts); W2 = zeros(FT, num_pts)
            W3 = zeros(FT, num_pts); W4 = zeros(FT, num_pts)
            dest_J = zeros(Int32, num_pts); dest_K = zeros(Int32, num_pts)
            
            # Determine source cell range
            if f2 == 3; j_src_range = NG+1:NG+L_pack; k_src_range = 1:Nz_tot_src
            elseif f2 == 4; j_src_range = Ny_tot_src-NG-L_pack+1:Ny_tot_src-NG; k_src_range = 1:Nz_tot_src
            elseif f2 == 5; j_src_range = 1:Ny_tot_src; k_src_range = NG+1:NG+L_pack
            elseif f2 == 6; j_src_range = 1:Ny_tot_src; k_src_range = Nz_tot_src-NG-L_pack+1:Nz_tot_src-NG
            else continue
            end
            
            # For each ghost cell, find nearest source cell
            for (pt_idx, (jg, kg, yc_g, zc_g)) in enumerate(ghost_pts)
                dest_J[pt_idx] = Int32(jg)
                dest_K[pt_idx] = Int32(kg)
                
                best_dist = Inf; best_j = NG+1; best_k = NG+1
                for ks in k_src_range, js in j_src_range
                    yc_s = 0.0; zc_s = 0.0
                    for di in 0:1, dj in 0:1, dk in 0:1
                        yc_s += y_src[NG+1+di, js+dj, ks+dk]
                        zc_s += z_src[NG+1+di, js+dj, ks+dk]
                    end
                    yc_s /= 8.0; zc_s /= 8.0
                    d = (yc_g - yc_s)^2 + (zc_g - zc_s)^2
                    if d < best_dist
                        best_dist = d; best_j = js; best_k = ks
                    end
                end
                
                # Encode as linear index: base = (k-1)*Ny*Nx + (j-1)*Nx
                base = (best_k - 1) * Ny_tot_src * Nx_tot_src + (best_j - 1) * Nx_tot_src
                C1[pt_idx] = Int32(base)
                C2[pt_idx] = Int32(base)
                C3[pt_idx] = Int32(base)
                C4[pt_idx] = Int32(base)
                W1[pt_idx] = one(FT)  # direct copy (nearest neighbor)
            end
            
            grp_name = "b$(b1)_f$(f1)"
            grp = create_group(fid, grp_name)
            grp["num_pts"] = num_pts
            grp["C1"] = C1; grp["C2"] = C2; grp["C3"] = C3; grp["C4"] = C4
            grp["W1"] = W1; grp["W2"] = W2; grp["W3"] = W3; grp["W4"] = W4
            grp["dest_J"] = dest_J; grp["dest_K"] = dest_K
            grp["Stride_n"] = Int64(Nx_tot_src * Ny_tot_src * Nz_tot_src)
            
            println("  Weights: b$b1 f$f1 → b$b2 f$f2: $num_pts ghost pts")
        end
    end
end

# =============================================================================
# 6. Main Conversion Pipeline
# =============================================================================
function convert_mesh(grid_file, config_file)
    println("=" ^ 60)
    println("  Flame3D Plot3D → HDF5 Converter")
    println("=" ^ 60)
    
    # ─── Read config ───
    config = TOML.parsefile(config_file)
    NG = get(config, "NG", 4)
    formatted = get(config, "formatted", false)
    precision_str = get(config, "precision", "float64")
    precision = precision_str == "float32" ? Float32 : Float64
    
    println("  NG = $NG")
    println("  Grid file: $grid_file")
    println("  Config file: $config_file")
    println()
    
    # ─── Read Plot3D mesh ───
    println(">>> Reading Plot3D mesh...")
    raw_blocks = read_plot3d(grid_file; formatted=formatted, precision=precision)
    nblocks = length(raw_blocks)
    
    # ─── Add ghost cells and compute metrics ───
    mkpath("MESH")
    blocks_data = []
    
    for bid in 0:nblocks-1
        x_raw, y_raw, z_raw, Nx, Ny, Nz = raw_blocks[bid + 1]
        println(">>> Block $bid: Adding ghost cells ($NG layers)...")
        x, y, z = add_ghost_cells(x_raw, y_raw, z_raw, Nx, Ny, Nz, NG)
        push!(blocks_data, (x, y, z, Nx, Ny, Nz))
        
        println("  Computing FVM metrics...")
        Ai, nxi, nyi, nzi, Aj, nxj, nyj, nzj, Ak, nxk, nyk, nzk, V = compute_fvm_metrics(x, y, z, Nx, Ny, Nz, NG)
        
        # Check for bad volumes
        vol_inv = V[NG+1:NG+Nx, NG+1:NG+Ny, NG+1:NG+Nz]
        if any(isnan, vol_inv) || any(x -> x <= 0, vol_inv)
            @warn "Block $bid has NaN or non-positive volumes!"
        end
        
        # Write mesh HDF5
        # IMPORTANT: coords must contain ONLY interior nodes (Nx+1, Ny+1, Nz+1).
        # The solver generates ghost cell coordinates at runtime via expand_coords_with_ghost().
        # Writing ghost-extended arrays here causes NG-cell physical coordinate offset!
        x_real = x[1+NG:Nx+1+NG, 1+NG:Ny+1+NG, 1+NG:Nz+1+NG]
        y_real = y[1+NG:Nx+1+NG, 1+NG:Ny+1+NG, 1+NG:Nz+1+NG]
        z_real = z[1+NG:Nx+1+NG, 1+NG:Ny+1+NG, 1+NG:Nz+1+NG]
        h5open("MESH/mesh_b$bid.h5", "w") do f
            f["NG"] = NG
            f["Nx"] = Int64(Nx)
            f["Ny"] = Int64(Ny)
            f["Nz"] = Int64(Nz)
            f["coords"] = Float32.(cat(reshape(x_real, (1, size(x_real)...)),
                                       reshape(y_real, (1, size(y_real)...)),
                                       reshape(z_real, (1, size(z_real)...)), dims=1))
            # Separate x/y/z datasets for XDMF X_Y_Z geometry (ParaView compatible)
            f["x"] = Float32.(x_real)
            f["y"] = Float32.(y_real)
            f["z"] = Float32.(z_real)
        end
        
        # Write metrics HDF5
        h5open("MESH/metrics_b$bid.h5", "w") do f
            f["Areai"] = Ai; f["nxi"] = nxi; f["nyi"] = nyi; f["nzi"] = nzi
            f["Areaj"] = Aj; f["nxj"] = nxj; f["nyj"] = nyj; f["nzj"] = nzj
            f["Areak"] = Ak; f["nxk"] = nxk; f["nyk"] = nyk; f["nzk"] = nzk
            f["Vol"] = V
        end
        println("  Wrote mesh_b$bid.h5, metrics_b$bid.h5")
    end
    
    # ─── Parse connectivity ───
    println("\n>>> Processing connectivity...")
    conn_list = get(config, "connectivity", [])
    connectivity_rows = zeros(Int64, length(conn_list), 4)
    for (i, c) in enumerate(conn_list)
        connectivity_rows[i, :] .= c["blocks"]
    end
    println("  $(length(conn_list)) connectivity pairs")
    
    # ─── Compute reverse_tan ───
    reverse_tan_arr = zeros(Int64, size(connectivity_rows, 1))
    for i in 1:size(connectivity_rows, 1)
        b1, f1, b2, f2 = connectivity_rows[i, :]
        x1, y1, z1, Nx1, Ny1, Nz1 = blocks_data[b1 + 1]
        x2, y2, z2, Nx2, Ny2, Nz2 = blocks_data[b2 + 1]
        t1 = get_tangential_vector(x1, y1, z1, f1, Nx1, Ny1, Nz1, NG)
        t2 = get_tangential_vector(x2, y2, z2, f2, Nx2, Ny2, Nz2, NG)
        reverse_tan_arr[i] = dot(t1, t2) < 0 ? 1 : 0
    end
    println("  reverse_tan: ", reverse_tan_arr')
    
    # ─── Parse face_bc ───
    face_bc = zeros(Int64, nblocks, 6)
    if haskey(config, "face_bc")
        for (bid_str, bcs) in config["face_bc"]
            bid = parse(Int, bid_str)
            for (fid, bc_name) in enumerate(bcs)
                face_bc[bid + 1, fid] = get(BC_NAME_MAP, bc_name, BC_INTERBLOCK)
            end
        end
    end
    
    println("  face_bc:")
    for bid in 0:nblocks-1
        bc_strs = [get(BC_ID_TO_NAME, Int32(face_bc[bid+1, f]), "unknown") for f in 1:6]
        println("    Block $bid: ", bc_strs)
    end
    
    # ─── Parse bc_params ───
    bc_params_arr = zeros(FT, nblocks, 6, N_BC_PARAMS)
    if haskey(config, "bc_params")
        for (key, value) in config["bc_params"]
            # Key format: "block.face.param_name"
            parts = split(key, ".")
            if length(parts) != 3
                @warn "Invalid bc_params key: $key (expected format: block.face.param_name)"
                continue
            end
            bid = parse(Int, parts[1])
            fid = parse(Int, parts[2])
            param_name = parts[3]
            slot = get(BCP_KEY_MAP, param_name, 0)
            if slot == 0
                @warn "Unknown bc_params parameter name: $param_name"
                continue
            end
            bc_params_arr[bid + 1, fid, slot] = FT(value)
            println("    bc_param: Block $bid Face $fid $param_name = $value")
        end
    end
    
    # ─── Write block_connectivity.h5 ───
    Nx_b = [blocks_data[i][4] for i in 1:nblocks]
    Ny_b = [blocks_data[i][5] for i in 1:nblocks]
    Nz_b = [blocks_data[i][6] for i in 1:nblocks]
    
    h5open("MESH/block_connectivity.h5", "w") do f
        f["Nblocks"] = nblocks
        f["Nx_b"] = Int64.(Nx_b)
        f["Ny_b"] = Int64.(Ny_b)
        f["Nz_b"] = Int64.(Nz_b)
        f["connectivity"] = connectivity_rows
        f["reverse_tan"] = reverse_tan_arr
        f["face_bc"] = face_bc
        f["bc_params"] = bc_params_arr
    end
    println("  Wrote block_connectivity.h5")
    
    # ─── Build interpolation weights ───
    println("\n>>> Computing interpolation weights...")
    build_conformal_interp_weights(blocks_data, connectivity_rows, NG)
    println("  Wrote interp_weights.h5")
    
    println("\n" * "=" ^ 60)
    println("  Conversion complete! Output in MESH/")
    println("=" ^ 60)
end

# ─── Entry point ───
if length(ARGS) >= 2
    convert_mesh(ARGS[1], ARGS[2])
elseif length(ARGS) == 1
    # Default config name
    convert_mesh(ARGS[1], replace(ARGS[1], r"\.\w+$" => ".toml"))
else
    println("Usage: julia convert_plot3d.jl <grid.xyz> <config.toml>")
    println()
    println("See header comments for config file format.")
end
