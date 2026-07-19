# =============================================================================
#  cyl_regrid.jl - 蝴蝶多块网格 -> 圆柱坐标 (r, θ, x) 重采样
#
#  策略：预计算映射表（一次性），每步只做加权求和（在线高效）。
#
#  蝴蝶网格索引结构（来自 gen_butterfly_fvm.jl）：
#    块0: 中心方块（笛卡尔），j->y(-1..1), k->z(-1..1), r∈[0, r_inner·√2≈0.283]
#    块1: j=径向(wall@j=1, inner@j=Ny+1), k=周向, θ=5π/4+param·π/2 (k↑ => θ↑ CCW)
#    块2: j=径向(inner@j=1, wall@j=Ny+1), k=周向, θ=3π/4-param·π/2 (k↑ => θ↓ CW)
#    块3: k=径向(wall@k=1, inner@k=Nz+1), j=周向, θ=5π/4-param·π/2 (j↑ => θ↓ CW)
#    块4: k=径向(inner@k=1, wall@k=Nz+1), j=周向, θ=7π/4+param·π/2 (j↑ => θ↑ CCW)
#
#  注：网格存储为 (Nx+1, Ny+1, Nz+1) 节点数组；cell 中心 = 相邻 8 节点平均。
# =============================================================================

using HDF5
using LinearAlgebra
using Printf

# =============================================================================
#  RegridMap: 预计算的插值映射表
# =============================================================================
"""
    RegridMap

一次性预计算的目标圆柱网格 -> 源蝴蝶网格插值映射。

字段：
  ntheta, nrad, nx   : 目标网格尺寸
  r_grid[nrad]       : 目标径向坐标（含 r=0）
  theta_grid[ntheta] : 目标周向坐标（均匀 [0, 2π)）
  x_grid[nx]         : 目标轴向坐标（= 源网格 x，均匀）
  weights             : Vector{TriLinearStencil}，长度 nrad*ntheta*nx，
                        每个元素记录目标点对应的源块、cell 索引、8 权重
  block0_mask[nrad]  : 该 r 是否落在块0 核心（需核心处理）
"""
struct TriLinearStencil
    bid::Int                 # 源块号 (0-4)
    i0::Int; j0::Int; k0::Int  # cell 左下角节点索引 (1-based)
    w::NTuple{8, Float64}    # 8 节点三线性权重
end

mutable struct RegridMap
    ntheta::Int
    nrad::Int
    nx::Int
    r_grid::Vector{Float64}
    theta_grid::Vector{Float64}
    x_grid::Vector{Float64}
    weights::Vector{TriLinearStencil}
    block0_mask::Vector{Bool}
end


# =============================================================================
#  网格加载：cell 中心坐标
# =============================================================================
"""
    load_block_cell_centers(mesh_dir, bid)

加载某块的节点坐标，返回 cell 中心 (xc, yc, zc)，尺寸 (Nx, Ny, Nz)。
复用 analyze_vw_stress.jl / compute_helicity.jl 的 8 节点平均范式。
"""
function load_block_cell_centers(mesh_dir::String, bid::Int)
    mpath = joinpath(mesh_dir, "mesh_b$(bid).h5")
    # Julia h5read 保持列序：磁盘存 (Nx+1, Ny+1, Nz+1)，读回即此顺序，无需转置。
    x_node = Float64.(h5read(mpath, "x"))
    y_node = Float64.(h5read(mpath, "y"))
    z_node = Float64.(h5read(mpath, "z"))

    # 节点 (Nx+1, Ny+1, Nz+1) -> cell 中心 (Nx, Ny, Nz)：8 节点算术平均 (sum/8)
    xc = (1/8) .* (
        x_node[1:end-1, 1:end-1, 1:end-1] .+ x_node[2:end,   1:end-1, 1:end-1] .+
        x_node[1:end-1, 2:end,   1:end-1] .+ x_node[2:end,   2:end,   1:end-1] .+
        x_node[1:end-1, 1:end-1, 2:end  ] .+ x_node[2:end,   1:end-1, 2:end  ] .+
        x_node[1:end-1, 2:end,   2:end  ] .+ x_node[2:end,   2:end,   2:end  ]
    )
    yc = (1/8) .* (
        y_node[1:end-1, 1:end-1, 1:end-1] .+ y_node[2:end,   1:end-1, 1:end-1] .+
        y_node[1:end-1, 2:end,   1:end-1] .+ y_node[2:end,   2:end,   1:end-1] .+
        y_node[1:end-1, 1:end-1, 2:end  ] .+ y_node[2:end,   1:end-1, 2:end  ] .+
        y_node[1:end-1, 2:end,   2:end  ] .+ y_node[2:end,   2:end,   2:end  ]
    )
    zc = (1/8) .* (
        z_node[1:end-1, 1:end-1, 1:end-1] .+ z_node[2:end,   1:end-1, 1:end-1] .+
        z_node[1:end-1, 2:end,   1:end-1] .+ z_node[2:end,   2:end,   1:end-1] .+
        z_node[1:end-1, 1:end-1, 2:end  ] .+ z_node[2:end,   1:end-1, 2:end  ] .+
        z_node[1:end-1, 2:end,   2:end  ] .+ z_node[2:end,   2:end,   2:end  ]
    )
    return xc, yc, zc
end


# =============================================================================
#  预计算映射表
# =============================================================================
"""
    build_regrid_map(mesh_dir; ntheta, nrad, nx, R0, r_inner)

一次性预计算目标圆柱网格上每个点对应的源蝴蝶网格三线性插值 stencil。

参数 nx=0 表示用源网格全部 Nx 点（轴向不降采样）。
"""
function build_regrid_map(mesh_dir::String;
                          ntheta::Int=modal_ntheta,
                          nrad::Int=modal_nrad,
                          nx::Int=modal_nx_keep,
                          R0::Float64=R0,
                          r_inner::Float64=r_inner)

    # ── 加载 5 块 cell 中心坐标 ──
    N_BLOCKS = 5
    xc = Vector{Array{Float64,3}}(undef, N_BLOCKS)
    yc = Vector{Array{Float64,3}}(undef, N_BLOCKS)
    zc = Vector{Array{Float64,3}}(undef, N_BLOCKS)
    Nx_arr = Vector{Int}(undef, N_BLOCKS)
    Ny_arr = Vector{Int}(undef, N_BLOCKS)
    Nz_arr = Vector{Int}(undef, N_BLOCKS)
    for bid in 0:N_BLOCKS-1
        xc[bid+1], yc[bid+1], zc[bid+1] = load_block_cell_centers(mesh_dir, bid)
        Nx_arr[bid+1], Ny_arr[bid+1], Nz_arr[bid+1] = size(xc[bid+1])
    end

    # ── 目标网格 ──
    # r: 含 r=0，到 R0。块0 核心 r∈[0, r_inner·√2]，块1-4 环 r∈[r_inner, R0]
    # 用切比雪夫-ish 分布：壁面附近略密（与原网格 tanh 一致），但目标均匀 r 也常用。
    # 这里用均匀 r，壁面分辨率由 nrad 保证；如需壁面加密后续可调。
    r_grid = collect(range(0.0, R0; length=nrad))
    theta_grid = collect(range(0.0, 2π; length=ntheta+1))[1:end-1]  # [0, 2π)，不含 2π

    # x: 用块0 的 x 坐标（5 块共址，x 一致），均匀
    nx_use = (nx == 0) ? Nx_arr[1] : nx
    x_grid = zeros(nx_use)
    for i in 1:nx_use
        x_grid[i] = xc[1][i, 1, 1]
    end

    # ── 判断每个 r 属于块0 还是环 ──
    # 块0 外接圆约 r_inner·(1+alpha)·√2 ≈ 0.2·1.18·1.414 ≈ 0.333（角落），
    # 但轴上径向边界 r_inner·(1+alpha)=0.236。
    # 用 0.236 作为分界：r<0.236 用块0，r≥0.236 用块1-4 环。
    # （0.236~0.283 之间块0 和环有重叠，优先用环更准确。）
    r_core_cut = r_inner * 1.18   # = 0.236
    block0_mask = r_grid .< r_core_cut

    # ── 预计算每个目标点的 stencil ──
    weights = Vector{TriLinearStencil}(undef, nrad * ntheta * nx_use)

    for ir in 1:nrad
        r_t = r_grid[ir]
        for it in 1:ntheta
            θ_t = theta_grid[it]
            # 目标笛卡尔坐标。约定 θ=atan2(z,y)，即 θ=0 在 +y 轴，CCW 为正。
            # 故 y = r·cos(θ), z = r·sin(θ)。
            y_t = r_t * cos(θ_t)
            z_t = r_t * sin(θ_t)

            for ix in 1:nx_use
                x_t = x_grid[ix]
                idx = ((ir-1)*ntheta + (it-1))*nx_use + ix

                if block0_mask[ir] && modal_core_strategy == :interp
                    # 块0 核心区：在块0 中查找
                    st = find_stencil(0, xc[1], yc[1], zc[1],
                                      x_t, y_t, z_t, Nx_arr[1], Ny_arr[1], Nz_arr[1])
                else
                    # 环区：在块1-4 中查找（按 θ 落入哪个扇形）
                    bid = theta_to_block(θ_t)
                    st = find_stencil(bid, xc[bid+1], yc[bid+1], zc[bid+1],
                                      x_t, y_t, z_t,
                                      Nx_arr[bid+1], Ny_arr[bid+1], Nz_arr[bid+1])
                end
                weights[idx] = st
            end
        end
    end

    return RegridMap(ntheta, nrad, nx_use, r_grid, theta_grid, x_grid,
                     weights, block0_mask)
end


# =============================================================================
#  辅助：θ -> 块号
# =============================================================================
"""
    theta_to_block(θ)

根据 θ = atan2(z, y)（θ=0 在 +y 轴，CCW 为正）返回对应的环块号 (1-4)。
实测块分布（MESH_SMALL 验证）：
  block2: θ ∈ [-45°, 45°]      (+y 侧)
  block4: θ ∈ [45°, 135°]      (+z 侧)
  block1: θ ∈ [135°, 180°] ∪ [-180°, -135°]  (-y 侧)
  block3: θ ∈ [-135°, -45°]    (-z 侧)
"""
function theta_to_block(θ::Real)
    θ = Float64(θ)
    # 归一化到 (-π, π]
    θ = mod(θ + π, 2π) - π
    if θ >= -π/4 && θ < π/4
        return 2    # +y
    elseif θ >= π/4 && θ < 3π/4
        return 4    # +z
    elseif θ >= -3π/4 && θ < -π/4
        return 3    # -z
    else             # [3π/4, π] ∪ [-π, -3π/4)
        return 1    # -y
    end
end


# =============================================================================
#  辅助：在指定块中查找目标点的三线性插值 stencil
# =============================================================================
"""
    find_stencil(bid, xc, yc, zc, x_t, y_t, z_t, Nx, Ny, Nz)

在指定块的 cell 中心网格中，找到 (x_t, y_t, z_t) 所在的 cell，
返回三线性插值 stencil。
块0 用全局笛卡尔搜索；块1-4 因为 j/k 含义不同，仍用笛卡尔坐标定位 cell。
"""
function find_stencil(bid::Int, xc::Array{Float64,3}, yc::Array{Float64,3},
                      zc::Array{Float64,3}, x_t::Float64, y_t::Float64,
                      z_t::Float64, Nx::Int, Ny::Int, Nz::Int)

    # 在 cell 中心网格中找包含目标点的 cell。
    # cell 中心沿各轴单调（x 均匀；y,z 在块1-4 沿径向/周向单调）。
    # 用二分/线性搜索找 i0 使得 xc[i0] <= x_t < xc[i0+1]，类似 j, k。

    i0 = find_interval(view(xc, :, 1, 1), x_t, Nx)
    # y, z 的间隔依赖 (j, k) 组合，不能简单取一维切片。
    # 近似：先用 i0 平面上的 cell 中心，二维修正。
    # 简化策略：取 i0 层，在 (j,k) 二维中找最近 cell，再取其 8 邻域。

    # 取 i0 和 i0+1 两层（轴向三线性需 2 层）
    i1 = min(i0 + 1, Nx)

    # 在 i0 层找 (y_t, z_t) 的最近 cell (j0, k0)
    j0, k0 = find_nearest_jk(view(yc, i0, :, :), view(zc, i0, :, :),
                              y_t, z_t, Ny, Nz)
    j1 = min(j0 + 1, Ny)
    k1 = min(k0 + 1, Nz)

    # 8 个节点的坐标（cell 中心，构成伪节点）
    x000 = xc[i0, j0, k0]; x100 = xc[i1, j0, k0]
    x010 = xc[i0, j1, k0]; x110 = xc[i1, j1, k0]
    x001 = xc[i0, j0, k1]; x101 = xc[i1, j0, k1]
    x011 = xc[i0, j1, k1]; x111 = xc[i1, j1, k1]

    y000 = yc[i0, j0, k0]; y100 = yc[i1, j0, k0]
    y010 = yc[i0, j1, k0]; y110 = yc[i1, j1, k0]
    y001 = yc[i0, j0, k1]; y101 = yc[i1, j0, k1]
    y011 = yc[i0, j1, k1]; y111 = yc[i1, j1, k1]

    z000 = zc[i0, j0, k0]; z100 = zc[i1, j0, k0]
    z010 = zc[i0, j1, k0]; z110 = zc[i1, j1, k0]
    z001 = zc[i0, j0, k1]; z101 = zc[i1, j0, k1]
    z011 = zc[i0, j1, k1]; z111 = zc[i1, j1, k1]

    # 三线性插值的局部坐标 (α, β, γ) ∈ [0,1]
    # 解：x_t = x000·(1-α)(1-β)(1-γ) + ... + x111·αβγ
    # 对非结构化 cell 中心，解析解困难。用中心差分近似求 (α,β,γ)：
    α = clamp((x_t - x000) / (x100 - x000 + 1e-30), 0.0, 1.0)
    β = clamp((y_t - y000) / (y010 - y000 + 1e-30), 0.0, 1.0)
    γ = clamp((z_t - z000) / (z001 - z000 + 1e-30), 0.0, 1.0)

    # 三线性权重
    w000 = (1-α)*(1-β)*(1-γ); w100 = α*(1-β)*(1-γ)
    w010 = (1-α)*β*(1-γ);     w110 = α*β*(1-γ)
    w001 = (1-α)*(1-β)*γ;     w101 = α*(1-β)*γ
    w011 = (1-α)*β*γ;         w111 = α*β*γ

    return TriLinearStencil(bid, i0, j0, k0,
                            (w000, w100, w010, w110, w001, w101, w011, w111))
end


# =============================================================================
#  辅助搜索函数
# =============================================================================
"""在单调一维数组中找 i 使 v[i] <= x < v[i+1]，边界 clamp。"""
function find_interval(v::AbstractVector, x::Float64, N::Int)
    if x <= v[1]; return 1; end
    if x >= v[N]; return N - 1; end
    # 二分
    lo, hi = 1, N
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if v[mid] <= x
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end


"""在二维 cell 中心数组中找最接近 (y_t, z_t) 的 cell 索引 (j0, k0)。"""
function find_nearest_jk(yv::AbstractMatrix, zv::AbstractMatrix,
                         y_t::Float64, z_t::Float64, Ny::Int, Nz::Int)
    best_d2 = Inf
    best_j, best_k = 1, 1
    @inbounds for k in 1:Nz, j in 1:Ny
        dy = yv[j, k] - y_t
        dz = zv[j, k] - z_t
        d2 = dy*dy + dz*dz
        if d2 < best_d2
            best_d2 = d2
            best_j = j
            best_k = k
        end
    end
    return best_j, best_k
end


# =============================================================================
#  每步重采样：用映射表把全场速度插值到圆柱网格
# =============================================================================
"""
    regrid_to_cyl!(u_cyl, v_cyl, w_cyl, rmap, block_data)

用预计算的 rmap 把 5 块速度场重采样到目标圆柱网格。
u_cyl, v_cyl, w_cyl: (nx, ntheta, nrad) 输出（x, θ, r 顺序）
block_data: Dict{bid => (u_arr, v_arr, w_arr)} 每块 CPU 上的 (Nx,Ny,Nz) 速度

速度分量：原 (u,v,w) 是笛卡尔 (x,y,z) 方向，重采样后保持笛卡尔分量，
         周向 FFT 后再变换到 (u_x, u_r, u_θ)。
"""
function regrid_to_cyl!(u_cyl::Array{Float64,3}, v_cyl::Array{Float64,3},
                        w_cyl::Array{Float64,3}, rmap::RegridMap,
                        block_data::Dict{Int, NTuple{3, Array{Float64,3}}})

    nx = rmap.nx
    ntheta = rmap.ntheta
    nrad = rmap.nrad
    fill!(u_cyl, 0.0); fill!(v_cyl, 0.0); fill!(w_cyl, 0.0)

    for ir in 1:nrad
        for it in 1:ntheta
            for ix in 1:nx
                idx = ((ir-1)*ntheta + (it-1))*nx + ix
                st = rmap.weights[idx]
                (ua, va, wa) = block_data[st.bid]

                # 三线性加权求和
                i0, j0, k0 = st.i0, st.j0, st.k0
                w = st.w
                # 安全边界检查
                i1 = min(i0+1, size(ua,1)); j1 = min(j0+1, size(ua,2)); k1 = min(k0+1, size(ua,3))

                u_val = w[1]*ua[i0,j0,k0] + w[2]*ua[i1,j0,k0] +
                        w[3]*ua[i0,j1,k0] + w[4]*ua[i1,j1,k0] +
                        w[5]*ua[i0,j0,k1] + w[6]*ua[i1,j0,k1] +
                        w[7]*ua[i0,j1,k1] + w[8]*ua[i1,j1,k1]
                v_val = w[1]*va[i0,j0,k0] + w[2]*va[i1,j0,k0] +
                        w[3]*va[i0,j1,k0] + w[4]*va[i1,j1,k0] +
                        w[5]*va[i0,j0,k1] + w[6]*va[i1,j0,k1] +
                        w[7]*va[i0,j1,k1] + w[8]*va[i1,j1,k1]
                w_val = w[1]*wa[i0,j0,k0] + w[2]*wa[i1,j0,k0] +
                        w[3]*wa[i0,j1,k0] + w[4]*wa[i1,j1,k0] +
                        w[5]*wa[i0,j0,k1] + w[6]*wa[i1,j0,k1] +
                        w[7]*wa[i0,j1,k1] + w[8]*wa[i1,j1,k1]

                u_cyl[ix, it, ir] = u_val
                v_cyl[ix, it, ir] = v_val
                w_cyl[ix, it, ir] = w_val
            end
        end
    end
end
