# =============================================================================
#  test_regrid_g1.jl - G1 验证：重采样守恒
#
#  目标：单快照重采样到圆柱网格后，总动能 vs 原始笛卡尔网格总动能，
#        误差应 <1%。
#
#  不依赖求解器，直接读 PLT + mesh。
#  用法：julia post/online_modal/test_regrid_g1.jl [MESH_DIR PLT_DIR STEP]
#  默认：MESH_SMALL PLT 1
# =============================================================================
include(joinpath(@__DIR__, "modal_config.jl"))
include(joinpath(@__DIR__, "cyl_regrid.jl"))

using HDF5

"""
    compute_cell_volumes(mesh_dir, bid)

从节点坐标计算每块每个 cell 的体积（中心差分雅可比行列式）。
节点 (Nx+1,Ny+1,Nz+1) -> cell (Nx,Ny,Nz)。
"""
function compute_cell_volumes(mesh_dir::String, bid::Int)
    mpath = joinpath(mesh_dir, "mesh_b$(bid).h5")
    x = Float64.(h5read(mpath, "x"))
    y = Float64.(h5read(mpath, "y"))
    z = Float64.(h5read(mpath, "z"))
    Nx = size(x,1) - 1; Ny = size(x,2) - 1; Nz = size(x,3) - 1
    vol = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # 8 节点
        x000=x[i,j,k];   x100=x[i+1,j,k];   x010=x[i,j+1,k];   x110=x[i+1,j+1,k]
        x001=x[i,j,k+1]; x101=x[i+1,j,k+1]; x011=x[i,j+1,k+1]; x111=x[i+1,j+1,k+1]
        y000=y[i,j,k];   y100=y[i+1,j,k];   y010=y[i,j+1,k];   y110=y[i+1,j+1,k]
        y001=y[i,j,k+1]; y101=y[i+1,j,k+1]; y011=y[i,j+1,k+1]; y111=y[i+1,j+1,k+1]
        z000=z[i,j,k];   z100=z[i+1,j,k];   z010=z[i,j+1,k];   z110=z[i+1,j+1,k]
        z001=z[i,j,k+1]; z101=z[i+1,j,k+1]; z011=z[i,j+1,k+1]; z111=z[i+1,j+1,k+1]
        # 用对角向量近似体积（三线性单元体积 ≈ |a·(b×c)|/6 六面体，这里用中心差分边向量）
        ax=x100-x000; ay=y100-y000; az=z100-z000  # i 向
        bx=x010-x000; by=y010-y000; bz=z010-z000  # j 向
        cx=x001-x000; cy=y001-y000; cz=z001-z000  # k 向
        # 体积 = |det[ax bx cx; ay by cy; az bz cz]|
        vol[i,j,k] = abs(ax*(by*cz-bz*cy) - ay*(bx*cz-bz*cx) + az*(bx*cy-by*cx))
    end
    return vol
end

function main()
    mesh_dir = length(ARGS) >= 1 ? ARGS[1] : "MESH_SMALL"
    plt_dir  = length(ARGS) >= 2 ? ARGS[2] : "PLT"
    step     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

    println("=" ^ 60)
    println("  G1 验证：重采样动能守恒")
    println("  mesh=$mesh_dir  plt=$plt_dir  step=$step")
    println("=" ^ 60)

    # ── 1. 预计算映射表 ──
    print("[1/4] 预计算重采样映射表...")
    t0 = time()
    rmap = build_regrid_map(mesh_dir)
    @printf(" done (%.1fs)\n", time()-t0)
    @printf("    目标网格: nx=%d ntheta=%d nrad=%d\n", rmap.nx, rmap.ntheta, rmap.nrad)
    @printf("    r∈[%.4f, %.4f], θ∈[%.4f, %.4f]\n",
            rmap.r_grid[1], rmap.r_grid[end],
            rmap.theta_grid[1], rmap.theta_grid[end])

    # ── 2. 读快照速度（5 块）──
    print("[2/4] 读 PLT 速度场...")
    t0 = time()
    N_BLOCKS = 5
    block_data = Dict{Int, NTuple{3, Array{Float64,3}}}()
    for bid in 0:N_BLOCKS-1
        fpath = joinpath(plt_dir, "plt-$(step)-b$(bid).h5")
        if !isfile(fpath)
            println("\n  [跳过] $fpath 不存在（可能 CEBL precursor 块）")
            continue
        end
        u = Float64.(h5read(fpath, "u"))
        v = Float64.(h5read(fpath, "v"))
        w = Float64.(h5read(fpath, "w"))
        # Julia h5read 保持列序 (Nx,Ny,Nz)，无需转置
        block_data[bid] = (u, v, w)
    end
    @printf(" done (%.1fs)\n", time()-t0)
    println("  已读块: ", sort(collect(keys(block_data))))

    # ── 3. 原始总动能 ──
    print("[3/4] 计算原始总动能...")
    KE_orig = 0.0
    for (bid, (u,v,w)) in block_data
        if bid >= 5; continue; end  # 只算主管 5 块
        KE_orig += 0.5 * sum(u.^2 .+ v.^2 .+ w.^2)
    end
    @printf(" done\n    KE_orig = %.6e\n", KE_orig)

    # ── 4. 重采样 + 重采样后总动能 ──
    print("[4/4] 重采样并计算动能...")
    t0 = time()
    u_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    v_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    w_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)

    # 只用映射表里存在的块
    regrid_to_cyl!(u_cyl, v_cyl, w_cyl, rmap, block_data)
    @printf(" done (%.1fs)\n", time()-t0)

    # 重采样后动能：需乘以目标 cell 体积权重。
    # 简化：用圆柱坐标体积元 r·dr·dθ·dx 求和。
    dr = rmap.r_grid[2] - rmap.r_grid[1]
    dtheta = rmap.theta_grid[2] - rmap.theta_grid[1]
    dx = rmap.x_grid[2] - rmap.x_grid[1]
    KE_regrid = 0.0
    for ir in 1:rmap.nrad
        r = rmap.r_grid[ir]
        vol_w = r * dr * dtheta * dx
        KE_regrid += 0.5 * sum(view(u_cyl,:,:,ir).^2 .+
                               view(v_cyl,:,:,ir).^2 .+
                               view(w_cyl,:,:,ir).^2) * vol_w
    end

    # 原始动能也要体积加权才能公平比较。
    # 从 mesh 节点坐标计算每块真实 cell 体积（雅可比），避免依赖 metrics 文件。
    KE_orig_vol = 0.0
    for (bid, (u,v,w)) in block_data
        if bid >= 5; continue; end
        vol = compute_cell_volumes(mesh_dir, bid)
        KE_orig_vol += 0.5 * sum((u.^2 .+ v.^2 .+ w.^2) .* vol)
    end

    rel_err = abs(KE_regrid - KE_orig_vol) / KE_orig_vol * 100
    println()
    println("-" ^ 60)
    @printf("  原始动能 (体积加权):  KE_orig  = %.6e\n", KE_orig_vol)
    @printf("  重采样动能 (圆柱体积): KE_regrid = %.6e\n", KE_regrid)
    @printf("  相对误差: %.4f%%\n", rel_err)
    if rel_err < 1.0
        println("  [PASS] G1 通过（<1%）")
    else
        println("  [FAIL] G1 未通过（>=1%），需检查重采样映射")
    end
    println("-" ^ 60)
end

main()
