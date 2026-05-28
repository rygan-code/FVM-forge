# =============================================================================
#  compute_helicity.jl
#
#  后处理程序: 基于多块 PLT 文件计算流场螺旋度 (Helicity)
#
#  输出:
#    - HELICITY/helicity-STEP-bN.h5          瞬时螺旋度 + 涡量 (per-block)
#    - HELICITY/helicity-STEP.xmf            ParaView 多块可视化文件
#    - HELICITY/helicity_mean-bN.h5          平均螺旋度 + 脉动 RMS (per-block)
#    - HELICITY/helicity_mean.xmf            ParaView 多块可视化文件
#    - HELICITY/helicity_radial.csv          径向分布数据
#    - HELICITY/helicity_radial.png          径向分布图
#    - HELICITY/helicity_contour_x.png       截面等值线图
#
#  用法:
#    julia compute_helicity.jl [start_step] [end_step] [step_interval]
#
#  默认: 自动发现 PLT 中所有完整时间步, 使用最新 10 步做时间平均
# =============================================================================

using HDF5
using Statistics
using Printf
using LinearAlgebra

# 尝试加载绘图库 (非必需)
const HAS_MAKIE = try
    @eval using CairoMakie
    true
catch
    @warn "CairoMakie not available — plots will be skipped"
    false
end

# ─── 配置 ───
const MESH_DIR   = "MESH"
const PLT_DIR    = "PLT"
const OUT_DIR    = "HELICITY"
const NBLOCKS    = 5

# =============================================================================
#  工具函数
# =============================================================================

"""发现所有完整的 PLT 时间步 (5 个块文件齐全)"""
function find_complete_steps(dir::String, nblocks::Int)
    steps = Int[]
    for f in readdir(dir)
        m = match(r"plt-(\d+)-b0\.h5", f)
        if m !== nothing
            step = parse(Int, m.captures[1])
            all_exist = all(isfile(joinpath(dir, "plt-$(step)-b$(bid).h5")) for bid in 0:nblocks-1)
            if all_exist
                push!(steps, step)
            end
        end
    end
    return sort(steps)
end

# =============================================================================
#  Step 1: 加载网格 — 节点坐标 & cell center
# =============================================================================

struct BlockMesh
    id::Int
    Nx::Int; Ny::Int; Nz::Int
    # Node coordinates (Nx+1, Ny+1, Nz+1)
    x_node::Array{Float32, 3}
    y_node::Array{Float32, 3}
    z_node::Array{Float32, 3}
    # Cell center coordinates (Nx, Ny, Nz)
    xc::Array{Float64, 3}
    yc::Array{Float64, 3}
    zc::Array{Float64, 3}
    rc::Array{Float64, 3}   # sqrt(y² + z²)
end

function load_block_mesh(bid::Int)
    mpath = joinpath(MESH_DIR, "mesh_b$(bid).h5")
    x_nodes = h5read(mpath, "x")::Array{Float32, 3}
    y_nodes = h5read(mpath, "y")::Array{Float32, 3}
    z_nodes = h5read(mpath, "z")::Array{Float32, 3}

    # Cell centers: average along each structured dimension
    xc = 0.5 .* (Float64.(x_nodes[1:end-1,:,:]) .+ Float64.(x_nodes[2:end,:,:]))
    xc = 0.5 .* (xc[:,1:end-1,:] .+ xc[:,2:end,:])
    xc = 0.5 .* (xc[:,:,1:end-1] .+ xc[:,:,2:end])

    yc = 0.5 .* (Float64.(y_nodes[1:end-1,:,:]) .+ Float64.(y_nodes[2:end,:,:]))
    yc = 0.5 .* (yc[:,1:end-1,:] .+ yc[:,2:end,:])
    yc = 0.5 .* (yc[:,:,1:end-1] .+ yc[:,:,2:end])

    zc = 0.5 .* (Float64.(z_nodes[1:end-1,:,:]) .+ Float64.(z_nodes[2:end,:,:]))
    zc = 0.5 .* (zc[:,1:end-1,:] .+ zc[:,2:end,:])
    zc = 0.5 .* (zc[:,:,1:end-1] .+ zc[:,:,2:end])

    rc = sqrt.(yc.^2 .+ zc.^2)

    Nx, Ny, Nz = size(xc)
    @printf("  Block %d: (%d, %d, %d)  r∈[%.4f, %.4f]\n", bid, Nx, Ny, Nz, minimum(rc), maximum(rc))
    return BlockMesh(bid, Nx, Ny, Nz, x_nodes, y_nodes, z_nodes, xc, yc, zc, rc)
end

# =============================================================================
#  Step 2: 计算 Jacobian 度量 (从节点坐标, 4阶中心差分)
# =============================================================================

struct BlockMetrics
    # ∂(ξ,η,ζ)/∂(x,y,z) — inverse Jacobian, (Nx, Ny, Nz)
    dξdx::Array{Float64, 3}; dξdy::Array{Float64, 3}; dξdz::Array{Float64, 3}
    dηdx::Array{Float64, 3}; dηdy::Array{Float64, 3}; dηdz::Array{Float64, 3}
    dζdx::Array{Float64, 3}; dζdy::Array{Float64, 3}; dζdz::Array{Float64, 3}
    J::Array{Float64, 3}  # Jacobian determinant
end

"""
4th-order central difference coefficients for 1st derivative on uniform grid.
d/dξ ≈ (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / (12 Δξ)
在计算坐标中 Δξ = 1.
"""
function fd4_deriv_xi!(dfdξ::Array{Float64,3}, f::Array{Float64,3})
    Ni, Nj, Nk = size(f)
    @inbounds for k in 1:Nk, j in 1:Nj
        # Interior: 4th-order
        for i in 3:Ni-2
            dfdξ[i,j,k] = (-f[i+2,j,k] + 8.0*f[i+1,j,k] - 8.0*f[i-1,j,k] + f[i-2,j,k]) / 12.0
        end
        # Boundary: 2nd-order for i=1,2,Ni-1,Ni
        if Ni >= 3
            dfdξ[1,j,k]    = (-3.0*f[1,j,k] + 4.0*f[2,j,k] - f[3,j,k]) / 2.0
            dfdξ[2,j,k]    = (f[3,j,k] - f[1,j,k]) / 2.0
            dfdξ[Ni,j,k]   = (3.0*f[Ni,j,k] - 4.0*f[Ni-1,j,k] + f[Ni-2,j,k]) / 2.0
            dfdξ[Ni-1,j,k] = (f[Ni,j,k] - f[Ni-2,j,k]) / 2.0
        end
    end
end

function fd4_deriv_eta!(dfdη::Array{Float64,3}, f::Array{Float64,3})
    Ni, Nj, Nk = size(f)
    @inbounds for k in 1:Nk, i in 1:Ni
        for j in 3:Nj-2
            dfdη[i,j,k] = (-f[i,j+2,k] + 8.0*f[i,j+1,k] - 8.0*f[i,j-1,k] + f[i,j-2,k]) / 12.0
        end
        if Nj >= 3
            dfdη[i,1,k]    = (-3.0*f[i,1,k] + 4.0*f[i,2,k] - f[i,3,k]) / 2.0
            dfdη[i,2,k]    = (f[i,3,k] - f[i,1,k]) / 2.0
            dfdη[i,Nj,k]   = (3.0*f[i,Nj,k] - 4.0*f[i,Nj-1,k] + f[i,Nj-2,k]) / 2.0
            dfdη[i,Nj-1,k] = (f[i,Nj,k] - f[i,Nj-2,k]) / 2.0
        end
    end
end

function fd4_deriv_zeta!(dfdζ::Array{Float64,3}, f::Array{Float64,3})
    Ni, Nj, Nk = size(f)
    @inbounds for j in 1:Nj, i in 1:Ni
        for k in 3:Nk-2
            dfdζ[i,j,k] = (-f[i,j,k+2] + 8.0*f[i,j,k+1] - 8.0*f[i,j,k-1] + f[i,j,k-2]) / 12.0
        end
        if Nk >= 3
            dfdζ[i,j,1]    = (-3.0*f[i,j,1] + 4.0*f[i,j,2] - f[i,j,3]) / 2.0
            dfdζ[i,j,2]    = (f[i,j,3] - f[i,j,1]) / 2.0
            dfdζ[i,j,Nk]   = (3.0*f[i,j,Nk] - 4.0*f[i,j,Nk-1] + f[i,j,Nk-2]) / 2.0
            dfdζ[i,j,Nk-1] = (f[i,j,Nk] - f[i,j,Nk-2]) / 2.0
        end
    end
end

"""
从节点坐标计算 cell-center 上的 Jacobian 度量.
Forward Jacobian: ∂(x,y,z)/∂(ξ,η,ζ) → 求逆得到 ∂(ξ,η,ζ)/∂(x,y,z)
"""
function compute_metrics(mesh::BlockMesh)
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    xc, yc, zc = mesh.xc, mesh.yc, mesh.zc

    # Forward Jacobian 分量: ∂x/∂ξ, ∂x/∂η, ∂x/∂ζ, etc.
    dxdξ = zeros(Nx, Ny, Nz); dxdη = zeros(Nx, Ny, Nz); dxdζ = zeros(Nx, Ny, Nz)
    dydξ = zeros(Nx, Ny, Nz); dydη = zeros(Nx, Ny, Nz); dydζ = zeros(Nx, Ny, Nz)
    dzdξ = zeros(Nx, Ny, Nz); dzdη = zeros(Nx, Ny, Nz); dzdζ = zeros(Nx, Ny, Nz)

    fd4_deriv_xi!(dxdξ, xc);  fd4_deriv_eta!(dxdη, xc);  fd4_deriv_zeta!(dxdζ, xc)
    fd4_deriv_xi!(dydξ, yc);  fd4_deriv_eta!(dydη, yc);  fd4_deriv_zeta!(dydζ, yc)
    fd4_deriv_xi!(dzdξ, zc);  fd4_deriv_eta!(dzdη, zc);  fd4_deriv_zeta!(dzdζ, zc)

    # Jacobian determinant J = det(∂(x,y,z)/∂(ξ,η,ζ))
    J_det = zeros(Nx, Ny, Nz)

    # Inverse Jacobian
    m_dξdx = zeros(Nx, Ny, Nz); m_dξdy = zeros(Nx, Ny, Nz); m_dξdz = zeros(Nx, Ny, Nz)
    m_dηdx = zeros(Nx, Ny, Nz); m_dηdy = zeros(Nx, Ny, Nz); m_dηdz = zeros(Nx, Ny, Nz)
    m_dζdx = zeros(Nx, Ny, Nz); m_dζdy = zeros(Nx, Ny, Nz); m_dζdz = zeros(Nx, Ny, Nz)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # 3x3 forward Jacobian matrix
        a11 = dxdξ[i,j,k]; a12 = dxdη[i,j,k]; a13 = dxdζ[i,j,k]
        a21 = dydξ[i,j,k]; a22 = dydη[i,j,k]; a23 = dydζ[i,j,k]
        a31 = dzdξ[i,j,k]; a32 = dzdη[i,j,k]; a33 = dzdζ[i,j,k]

        det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
        J_det[i,j,k] = det

        inv_det = 1.0 / det
        # Cofactor matrix transposed / det = inverse
        m_dξdx[i,j,k] = (a22*a33 - a23*a32) * inv_det
        m_dξdy[i,j,k] = (a13*a32 - a12*a33) * inv_det
        m_dξdz[i,j,k] = (a12*a23 - a13*a22) * inv_det

        m_dηdx[i,j,k] = (a23*a31 - a21*a33) * inv_det
        m_dηdy[i,j,k] = (a11*a33 - a13*a31) * inv_det
        m_dηdz[i,j,k] = (a13*a21 - a11*a23) * inv_det

        m_dζdx[i,j,k] = (a21*a32 - a22*a31) * inv_det
        m_dζdy[i,j,k] = (a12*a31 - a11*a32) * inv_det
        m_dζdz[i,j,k] = (a11*a22 - a12*a21) * inv_det
    end

    return BlockMetrics(m_dξdx, m_dξdy, m_dξdz,
                        m_dηdx, m_dηdy, m_dηdz,
                        m_dζdx, m_dζdy, m_dζdz,
                        J_det)
end

# =============================================================================
#  Step 3: 涡量 & 螺旋度计算
# =============================================================================

"""
计算单个块上的涡量 (ωx, ωy, ωz) 和螺旋度 H = u·ωx + v·ωy + w·ωz.
使用 4 阶有限差分在计算坐标中求导, 再通过度量变换到物理坐标.
"""
function compute_helicity_block(u::Array{Float64,3}, v::Array{Float64,3}, w::Array{Float64,3},
                                met::BlockMetrics)
    Nx, Ny, Nz = size(u)

    # 计算坐标系中的速度导数
    dudξ = zeros(Nx, Ny, Nz); dudη = zeros(Nx, Ny, Nz); dudζ = zeros(Nx, Ny, Nz)
    dvdξ = zeros(Nx, Ny, Nz); dvdη = zeros(Nx, Ny, Nz); dvdζ = zeros(Nx, Ny, Nz)
    dwdξ = zeros(Nx, Ny, Nz); dwdη = zeros(Nx, Ny, Nz); dwdζ = zeros(Nx, Ny, Nz)

    fd4_deriv_xi!(dudξ, u);  fd4_deriv_eta!(dudη, u);  fd4_deriv_zeta!(dudζ, u)
    fd4_deriv_xi!(dvdξ, v);  fd4_deriv_eta!(dvdη, v);  fd4_deriv_zeta!(dvdζ, v)
    fd4_deriv_xi!(dwdξ, w);  fd4_deriv_eta!(dwdη, w);  fd4_deriv_zeta!(dwdζ, w)

    # 涡量和螺旋度
    ωx = zeros(Nx, Ny, Nz)
    ωy = zeros(Nx, Ny, Nz)
    ωz = zeros(Nx, Ny, Nz)
    H  = zeros(Nx, Ny, Nz)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # 物理坐标导数: ∂f/∂x = ∂f/∂ξ·∂ξ/∂x + ∂f/∂η·∂η/∂x + ∂f/∂ζ·∂ζ/∂x
        dudx = dudξ[i,j,k]*met.dξdx[i,j,k] + dudη[i,j,k]*met.dηdx[i,j,k] + dudζ[i,j,k]*met.dζdx[i,j,k]
        dudy = dudξ[i,j,k]*met.dξdy[i,j,k] + dudη[i,j,k]*met.dηdy[i,j,k] + dudζ[i,j,k]*met.dζdy[i,j,k]
        dudz = dudξ[i,j,k]*met.dξdz[i,j,k] + dudη[i,j,k]*met.dηdz[i,j,k] + dudζ[i,j,k]*met.dζdz[i,j,k]

        dvdx = dvdξ[i,j,k]*met.dξdx[i,j,k] + dvdη[i,j,k]*met.dηdx[i,j,k] + dvdζ[i,j,k]*met.dζdx[i,j,k]
        dvdy = dvdξ[i,j,k]*met.dξdy[i,j,k] + dvdη[i,j,k]*met.dηdy[i,j,k] + dvdζ[i,j,k]*met.dζdy[i,j,k]
        dvdz = dvdξ[i,j,k]*met.dξdz[i,j,k] + dvdη[i,j,k]*met.dηdz[i,j,k] + dvdζ[i,j,k]*met.dζdz[i,j,k]

        dwdx = dwdξ[i,j,k]*met.dξdx[i,j,k] + dwdη[i,j,k]*met.dηdx[i,j,k] + dwdζ[i,j,k]*met.dζdx[i,j,k]
        dwdy = dwdξ[i,j,k]*met.dξdy[i,j,k] + dwdη[i,j,k]*met.dηdy[i,j,k] + dwdζ[i,j,k]*met.dζdy[i,j,k]
        dwdz = dwdξ[i,j,k]*met.dξdz[i,j,k] + dwdη[i,j,k]*met.dηdz[i,j,k] + dwdζ[i,j,k]*met.dζdz[i,j,k]

        # 涡量: ω = ∇ × u
        ωx[i,j,k] = dwdy - dvdz
        ωy[i,j,k] = dudz - dwdx
        ωz[i,j,k] = dvdx - dudy

        # 螺旋度: H = u · ω
        H[i,j,k] = u[i,j,k]*ωx[i,j,k] + v[i,j,k]*ωy[i,j,k] + w[i,j,k]*ωz[i,j,k]
    end

    return (ωx=ωx, ωy=ωy, ωz=ωz, H=H)
end

# =============================================================================
#  Step 4: 加载 PLT 数据
# =============================================================================

function load_plt_block(bid::Int, step::Int)
    fpath = joinpath(PLT_DIR, "plt-$(step)-b$(bid).h5")
    if !isfile(fpath)
        error("PLT file not found: $fpath")
    end
    h5open(fpath, "r") do f
        u = read(f["u"])::Array{Float64, 3}
        v = read(f["v"])::Array{Float64, 3}
        w = read(f["w"])::Array{Float64, 3}
        return (u=u, v=v, w=w)
    end
end

# =============================================================================
#  Step 5: XMF 文件写入
# =============================================================================

function write_helicity_xmf(step::Int, meshes::Vector{BlockMesh}; is_mean::Bool=false)
    if is_mean
        xmf_name = joinpath(OUT_DIR, "helicity_mean.xmf")
        h5_prefix = "helicity_mean"
        var_list = [("H_mean", "H_mean"), ("H_fluc_rms", "H_fluc_rms"),
                    ("omega_x_mean", "omega_x_mean"), ("omega_y_mean", "omega_y_mean"), ("omega_z_mean", "omega_z_mean")]
    else
        xmf_name = joinpath(OUT_DIR, "helicity-$(step).xmf")
        h5_prefix = "helicity-$(step)"
        var_list = [("H", "Helicity"), ("omega_x", "Vorticity_X"), ("omega_y", "Vorticity_Y"), ("omega_z", "Vorticity_Z")]
    end

    open(xmf_name, "w") do io
        println(io, """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">
 <Domain>
  <Grid Name="MultiBlock_Helicity" GridType="Collection" CollectionType="Spatial">""")

        for bid in 0:NBLOCKS-1
            m = meshes[bid+1]
            nx, ny, nz = m.Nx, m.Ny, m.Nz
            h5name = is_mean ? "$(h5_prefix)-b$(bid).h5" : "$(h5_prefix)-b$(bid).h5"
            meshname = "../MESH/mesh_b$(bid).h5"

            println(io, """   <Grid Name="Block_$(bid)" GridType="Uniform">
    <Topology TopologyType="3DSMesh" NumberOfElements="$(nz+1) $(ny+1) $(nx+1)" />
    <Geometry GeometryType="X_Y_Z">
     <DataItem Dimensions="$(nz+1) $(ny+1) $(nx+1)" NumberType="Float" Precision="4" Format="HDF">
      $(meshname):/x
     </DataItem>
     <DataItem Dimensions="$(nz+1) $(ny+1) $(nx+1)" NumberType="Float" Precision="4" Format="HDF">
      $(meshname):/y
     </DataItem>
     <DataItem Dimensions="$(nz+1) $(ny+1) $(nx+1)" NumberType="Float" Precision="4" Format="HDF">
      $(meshname):/z
     </DataItem>
    </Geometry>""")

            for (h5key, label) in var_list
                println(io, """    <Attribute Name="$(label)" AttributeType="Scalar" Center="Cell">
     <DataItem Dimensions="$(nz) $(ny) $(nx)" NumberType="Float" Precision="8" Format="HDF">
      $(h5name):/$(h5key)
     </DataItem>
    </Attribute>""")
            end

            println(io, "   </Grid>")
        end

        println(io, """  </Grid>
 </Domain>
</Xdmf>""")
    end
    println("    → XMF: $xmf_name")
end

# =============================================================================
#  Step 6: 径向分布计算
# =============================================================================

function compute_radial_profile(meshes::Vector{BlockMesh}, H_mean_blocks, H2_sum_blocks,
                                Nt::Int; n_bins::Int=200, skip_layers::Int=2,
                                r_min_frac::Float64=0.02)
    # 确定壁面半径
    R_wall = maximum(maximum(m.rc) for m in meshes)
    r_min = r_min_frac * R_wall  # 排除极点附近 (Jacobian 奇异)

    # 等距径向 bin (从 r_min 开始)
    r_edges = range(r_min, R_wall, length=n_bins+1)
    r_centers = 0.5 .* (r_edges[1:end-1] .+ r_edges[2:end])

    # 累积
    H_mean_sum = zeros(n_bins)
    H_rms_sum  = zeros(n_bins)
    counts     = zeros(n_bins)

    n_skipped_bnd = 0
    n_skipped_pole = 0
    for bid in 0:NBLOCKS-1
        m = meshes[bid+1]
        H_mean = H_mean_blocks[bid+1]
        H2_mean = H2_sum_blocks[bid+1] ./ Nt
        H_rms = sqrt.(max.(H2_mean .- H_mean.^2, 0.0))

        Nx, Ny, Nz = m.Nx, m.Ny, m.Nz
        # 跳过边界 skip_layers 层 cell (j 和 k 方向)
        j_lo = 1 + skip_layers
        j_hi = Ny - skip_layers
        k_lo = 1 + skip_layers
        k_hi = Nz - skip_layers
        n_skipped_bnd += Nx * (Ny * Nz - (j_hi - j_lo + 1) * (k_hi - k_lo + 1))

        for k in k_lo:k_hi, j in j_lo:j_hi, i in 1:Nx
            r = m.rc[i,j,k]
            # 排除极点附近 (r < r_min) 的 cell
            if r < r_min
                n_skipped_pole += 1
                continue
            end
            bin_idx = clamp(Int(floor(((r - r_min) / (R_wall - r_min)) * n_bins)) + 1, 1, n_bins)
            H_mean_sum[bin_idx] += H_mean[i,j,k]
            H_rms_sum[bin_idx]  += H_rms[i,j,k]
            counts[bin_idx]     += 1.0
        end
    end
    n_total = sum(m.Nx * m.Ny * m.Nz for m in meshes)
    @printf("    Radial profile: skipped %d boundary cells (%.1f%%) + %d pole cells (%.2f%%)\n",
            n_skipped_bnd, 100.0 * n_skipped_bnd / n_total,
            n_skipped_pole, 100.0 * n_skipped_pole / n_total)

    # 归一化
    for i in 1:n_bins
        if counts[i] > 0
            H_mean_sum[i] /= counts[i]
            H_rms_sum[i]  /= counts[i]
        end
    end

    return (r=collect(r_centers), r_R=collect(r_centers ./ R_wall),
            H_mean=H_mean_sum, H_rms=H_rms_sum, R_wall=R_wall)
end

# =============================================================================
#  Step 7: 绘图 — 径向分布 + 截面等值线
# =============================================================================

function plot_radial_profile(prof; output_dir=OUT_DIR)
    if !HAS_MAKIE
        println("    ⚠ Plot skipped (CairoMakie not available)")
        return
    end

    # 壁面距离 (R - r) / R, 从壁面到管心
    wall_dist = 1.0 .- prof.r_R   # (R-r)/R, 壁面=0, 管心=1
    # 过滤掉 wall_dist <= 0 的点 (log 轴不能画 0)
    valid = wall_dist .> 0
    wd = wall_dist[valid]
    hm = prof.H_mean[valid]
    hr = prof.H_rms[valid]

    fig = Figure(size=(720, 420), fontsize=10)

    ax1 = Axis(fig[1, 1],
        xlabel = "(R - r) / R",
        ylabel = "H\u0304  [m\u00b7s\u207b\u00b2]",
        xlabelsize = 12, ylabelsize = 12,
        xscale = log10,
        xgridvisible = true, ygridvisible = true,
        xgridstyle = :dash, ygridstyle = :dash,
        xticks = [0.001, 0.01, 0.1, 1.0],
        xtickformat = values -> [v >= 1 ? "1" : v >= 0.1 ? "0.1" : v >= 0.01 ? "0.01" : "0.001" for v in values])

    ax2 = Axis(fig[1, 1],
        ylabel = "H'\u1d63\u2098\u209b  [m\u00b7s\u207b\u00b2]",
        ylabelsize = 12,
        yaxisposition = :right,
        xscale = log10,
        xgridvisible = false, ygridvisible = false,
        xticks = [0.001, 0.01, 0.1, 1.0])

    hidexdecorations!(ax2)
    hidespines!(ax2)

    l1 = lines!(ax1, wd, hm, color=:dodgerblue, linewidth=2)
    l2 = lines!(ax2, wd, hr, color=:crimson, linewidth=2, linestyle=:dash)

    Legend(fig[1, 1],
        [l1, l2],
        ["H\u0304 (mean)", "H'\u1d63\u2098\u209b (fluctuation)"],
        tellheight = false, tellwidth = false,
        halign = :left, valign = :top,
        margin = (10, 10, 10, 10),
        framevisible = true)

    save(joinpath(output_dir, "helicity_radial.png"), fig, px_per_unit=4)
    println("    \u2192 Plot: helicity_radial.png")
end

function plot_contour_cross_section(meshes::Vector{BlockMesh}, H_blocks;
                                    x_idx::Int=0, output_dir=OUT_DIR)
    if !HAS_MAKIE
        println("    ⚠ Contour plot skipped (CairoMakie not available)")
        return
    end

    # 如果 x_idx == 0, 取中间截面
    if x_idx == 0
        x_idx = meshes[1].Nx ÷ 2
    end

    fig = Figure(size=(600, 560), fontsize=10)
    ax = Axis(fig[1, 1],
        xlabel = "y",
        ylabel = "z",
        xlabelsize = 12, ylabelsize = 12,
        aspect = DataAspect(),
        title = "Helicity at x-section i=$(x_idx)")

    # 全局 colormap 范围
    H_min = minimum(minimum(H[x_idx, :, :]) for H in H_blocks)
    H_max = maximum(maximum(H[x_idx, :, :]) for H in H_blocks)
    # 对称 colorbar
    H_abs = max(abs(H_min), abs(H_max))
    clims = (-H_abs, H_abs)

    for bid in 0:NBLOCKS-1
        m = meshes[bid+1]
        H = H_blocks[bid+1]

        y_sec = m.yc[x_idx, :, :]
        z_sec = m.zc[x_idx, :, :]
        H_sec = H[x_idx, :, :]

        ys = vec(y_sec)
        zs = vec(z_sec)
        hs = vec(H_sec)
        scatter!(ax, ys, zs, color=hs, colormap=:RdBu, colorrange=clims,
                 markersize=2, strokewidth=0)
    end

    Colorbar(fig[1, 2], colormap=:RdBu, limits=clims,
             label="H  [m·s⁻²]", labelsize=12,
             width=15)

    save(joinpath(output_dir, "helicity_contour_x.png"), fig, px_per_unit=4)
    println("    → Plot: helicity_contour_x.png")
end

# =============================================================================
#  MAIN
# =============================================================================

println("=" ^ 70)
println("  Helicity Post-Processing — Multi-Block Butterfly Grid")
println("=" ^ 70)

# ── Parse CLI ──
all_steps = find_complete_steps(PLT_DIR, NBLOCKS)
@printf("  Found %d complete PLT timesteps\n", length(all_steps))

# 确定要处理的步
if length(ARGS) >= 2
    step_start = parse(Int, ARGS[1])
    step_end   = parse(Int, ARGS[2])
    step_interval = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
    target_steps = filter(s -> step_start <= s <= step_end, all_steps)
else
    # 默认: 最新 10 个完整步
    target_steps = length(all_steps) >= 10 ? all_steps[end-9:end] : all_steps
end

@printf("  Steps for averaging: %d steps [%d → %d]\n",
        length(target_steps), target_steps[1], target_steps[end])

# ── Output directory ──
mkpath(OUT_DIR)

# ── Step 1: 加载网格 ──
println("\n── Loading block meshes ──")
meshes = [load_block_mesh(bid) for bid in 0:NBLOCKS-1]

# ── Step 2: 计算度量 ──
println("\n── Computing Jacobian metrics ──")
metrics = BlockMetrics[]
for bid in 0:NBLOCKS-1
    @printf("  Block %d ... ", bid)
    t0 = time()
    met = compute_metrics(meshes[bid+1])
    @printf("done (%.1f s), |J| ∈ [%.2e, %.2e]\n", time()-t0,
            minimum(abs.(met.J)), maximum(abs.(met.J)))
    push!(metrics, met)
end

# ── Step 3: 逐步计算螺旋度 + 累积平均 (含涡量) ──
println("\n── Computing helicity for each timestep ──")

# 累积数组 (per-block)
H_sum_blocks  = [zeros(meshes[bid+1].Nx, meshes[bid+1].Ny, meshes[bid+1].Nz) for bid in 0:NBLOCKS-1]
H2_sum_blocks = [zeros(meshes[bid+1].Nx, meshes[bid+1].Ny, meshes[bid+1].Nz) for bid in 0:NBLOCKS-1]
ωx_sum_blocks = [zeros(meshes[bid+1].Nx, meshes[bid+1].Ny, meshes[bid+1].Nz) for bid in 0:NBLOCKS-1]
ωy_sum_blocks = [zeros(meshes[bid+1].Nx, meshes[bid+1].Ny, meshes[bid+1].Nz) for bid in 0:NBLOCKS-1]
ωz_sum_blocks = [zeros(meshes[bid+1].Nx, meshes[bid+1].Ny, meshes[bid+1].Nz) for bid in 0:NBLOCKS-1]

# 暂存最后一个时间步的瞬时结果 (用于截面图)
last_H_blocks = Vector{Array{Float64,3}}(undef, NBLOCKS)

Nt = length(target_steps)

for (tidx, step) in enumerate(target_steps)
    @printf("\n  [%d/%d] Step %d\n", tidx, Nt, step)
    t0 = time()

    for bid in 0:NBLOCKS-1
        # Load velocity
        plt = load_plt_block(bid, step)
        @printf("    Block %d: loaded. ", bid)

        # Compute vorticity & helicity
        result = compute_helicity_block(plt.u, plt.v, plt.w, metrics[bid+1])
        @printf("H ∈ [%.2e, %.2e]  ", minimum(result.H), maximum(result.H))

        # 累积 (螺旋度 + 涡量)
        H_sum_blocks[bid+1]  .+= result.H
        H2_sum_blocks[bid+1] .+= result.H .^ 2
        ωx_sum_blocks[bid+1] .+= result.ωx
        ωy_sum_blocks[bid+1] .+= result.ωy
        ωz_sum_blocks[bid+1] .+= result.ωz

        # 保存瞬时螺旋度到 HDF5
        h5name = joinpath(OUT_DIR, "helicity-$(step)-b$(bid).h5")
        h5open(h5name, "w") do f
            f["H"]       = result.H
            f["omega_x"] = result.ωx
            f["omega_y"] = result.ωy
            f["omega_z"] = result.ωz
        end

        # 保存最后一步用于截面图
        if tidx == Nt
            last_H_blocks[bid+1] = copy(result.H)
        end

        @printf("→ %s\n", basename(h5name))
    end

    # Write XMF for this step
    write_helicity_xmf(step, meshes; is_mean=false)

    @printf("    Elapsed: %.1f s\n", time() - t0)
    GC.gc()
end

# ── Step 4: 平均 & 脉动 RMS ──
println("\n── Computing mean helicity & fluctuation RMS ──")

H_mean_blocks = [H_sum_blocks[bid+1] ./ Nt for bid in 0:NBLOCKS-1]

# 涡量平均也需要重新算 (从 H_sum 类比)
# 这里我们直接从累积的 H 来算
# H_rms = sqrt(E[H²] - (E[H])²)

for bid in 0:NBLOCKS-1
    H_mean = H_mean_blocks[bid+1]
    H2_mean = H2_sum_blocks[bid+1] ./ Nt
    H_rms = sqrt.(max.(H2_mean .- H_mean.^2, 0.0))
    ωx_mean = ωx_sum_blocks[bid+1] ./ Nt
    ωy_mean = ωy_sum_blocks[bid+1] ./ Nt
    ωz_mean = ωz_sum_blocks[bid+1] ./ Nt

    h5name = joinpath(OUT_DIR, "helicity_mean-b$(bid).h5")
    h5open(h5name, "w") do f
        f["H_mean"]       = H_mean
        f["H_fluc_rms"]   = H_rms
        f["omega_x_mean"] = ωx_mean
        f["omega_y_mean"] = ωy_mean
        f["omega_z_mean"] = ωz_mean
    end

    @printf("  Block %d: H_mean ∈ [%.2e, %.2e], H'_rms ∈ [%.2e, %.2e], <ωx> ∈ [%.2e, %.2e] → %s\n",
            bid, minimum(H_mean), maximum(H_mean),
            minimum(H_rms), maximum(H_rms),
            minimum(ωx_mean), maximum(ωx_mean), basename(h5name))
end

# Write mean XMF
write_helicity_xmf(0, meshes; is_mean=true)

# ── Step 6: 径向分布 ──
println("\n── Computing radial profile ──")
prof = compute_radial_profile(meshes, H_mean_blocks, H2_sum_blocks, Nt)

# CSV output
csv_path = joinpath(OUT_DIR, "helicity_radial.csv")
open(csv_path, "w") do io
    println(io, "r,r_R,H_mean,H_rms")
    for i in eachindex(prof.r)
        @printf(io, "%.8e,%.8e,%.8e,%.8e\n", prof.r[i], prof.r_R[i], prof.H_mean[i], prof.H_rms[i])
    end
end
println("    → CSV: $csv_path")

# Plot
println("\n── Generating plots ──")
plot_radial_profile(prof)

# 截面等值线图 (最后一个时间步)
plot_contour_cross_section(meshes, last_H_blocks)

# ── Done ──
println("\n" * "=" ^ 70)
println("  Helicity post-processing complete!")
println("=" ^ 70)
@printf("  Steps processed: %d [%d → %d]\n", Nt, target_steps[1], target_steps[end])
@printf("  Output directory: %s\n", OUT_DIR)
println("  ParaView files:")
println("    - helicity-STEP.xmf   (instantaneous)")
println("    - helicity_mean.xmf   (time-averaged)")
println("  Plots:")
println("    - helicity_radial.png (radial distribution)")
println("    - helicity_contour_x.png (cross-section)")
println("=" ^ 70)

# Clean up temp file
if isfile("_inspect_h5.jl")
    rm("_inspect_h5.jl")
end
