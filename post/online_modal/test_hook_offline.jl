# =============================================================================
#  test_hook_offline.jl - 离线模拟 modal_hook 管线
#
#  用 PLT 数据模拟 in_situ_post_process 的行为（不依赖求解器运行时），
#  验证"拷切片->重采样->FFT->分量变换->落盘"完整管线。
#
#  G3 前置验证：确认落盘的 Fourier 系数能被正确读回且物理合理。
#
#  用法：julia post/online_modal/test_hook_offline.jl [MESH_DIR PLT_DIR STEP]
# =============================================================================
include(joinpath(@__DIR__, "modal_config.jl"))
include(joinpath(@__DIR__, "cyl_regrid.jl"))

using HDF5
using FFTW
using Printf
using Statistics

function main()
    mesh_dir = length(ARGS) >= 1 ? ARGS[1] : "MESH_SMALL"
    plt_dir  = length(ARGS) >= 2 ? ARGS[2] : "PLT"
    step     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1

    println("=" ^ 60)
    println("  钩子离线模拟：重采样->FFT->落盘->读回验证")
    println("  mesh=$mesh_dir  plt=$plt_dir  step=$step")
    println("=" ^ 60)

    # ── 1. 预计算映射表 ──
    print("[1/5] 预计算映射表...")
    rmap = build_regrid_map(mesh_dir)
    @printf(" done (nx=%d ntheta=%d nrad=%d)\n", rmap.nx, rmap.ntheta, rmap.nrad)

    # ── 2. 读 PLT 速度（模拟 b.Q 拷切片）──
    print("[2/5] 读 PLT 速度场...")
    block_data = Dict{Int, NTuple{3, Array{Float64,3}}}()
    for bid in 0:4
        fpath = joinpath(plt_dir, "plt-$(step)-b$(bid).h5")
        isfile(fpath) || continue
        u = Float64.(h5read(fpath, "u"))
        v = Float64.(h5read(fpath, "v"))
        w = Float64.(h5read(fpath, "w"))
        block_data[bid] = (u, v, w)
    end
    @printf(" done (%d 块)\n", length(block_data))

    # ── 3. 重采样 + FFT + 分量变换（模拟钩子核心逻辑）──
    print("[3/5] 重采样 + 周向 FFT...")
    t0 = time()

    u_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    v_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    w_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    regrid_to_cyl!(u_cyl, v_cyl, w_cyl, rmap, block_data)

    # 实空间分量变换 (u,v,w)->(ux,ur,utheta)
    ur_cyl  = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    uth_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
    for ir in 1:rmap.nrad, it in 1:rmap.ntheta
        θ = rmap.theta_grid[it]
        c = cos(θ); s = sin(θ)
        @inbounds for ix in 1:rmap.nx
            uy = v_cyl[ix, it, ir]
            uz = w_cyl[ix, it, ir]
            ur_cyl[ix, it, ir]  =  uy * c + uz * s
            uth_cyl[ix, it, ir] = -uy * s + uz * c
        end
    end

    # 周向 FFT
    ntheta = rmap.ntheta
    ntheta_half = ntheta ÷ 2 + 1
    nmodes = min(modal_nmodes_keep, ntheta_half)

    u_hat   = zeros(ComplexF64, nmodes, rmap.nrad, rmap.nx)
    ur_hat  = zeros(ComplexF64, nmodes, rmap.nrad, rmap.nx)
    uth_hat = zeros(ComplexF64, nmodes, rmap.nrad, rmap.nx)

    for ix in 1:rmap.nx, ir in 1:rmap.nrad
        if rmap.r_grid[ir] < r_axis_cut
            u_hat[1, ir, ix]   = sum(view(u_cyl, ix, :, ir)) / ntheta
            ur_hat[1, ir, ix]  = sum(view(ur_cyl, ix, :, ir)) / ntheta
            uth_hat[1, ir, ix] = sum(view(uth_cyl, ix, :, ir)) / ntheta
            continue
        end
        fu   = FFTW.rfft(view(u_cyl,  ix, :, ir))
        fur  = FFTW.rfft(view(ur_cyl, ix, :, ir))
        futh = FFTW.rfft(view(uth_cyl,ix, :, ir))
        for m in 1:nmodes
            u_hat[m, ir, ix]   = fu[m]
            ur_hat[m, ir, ix]  = fur[m]
            uth_hat[m, ir, ix] = futh[m]
        end
    end
    @printf(" done (%.2fs)\n", time()-t0)

    # ── 4. 落盘 ──
    print("[4/5] 落盘 HDF5...")
    outpath = joinpath(modal_out_dir, "coeff-test-$(step).h5")
    mkpath(modal_out_dir)
    t_stamp = 0.0  # 离线测试用 0 时间戳
    h5open(outpath, "w") do f
        attrs(f)["time"] = Float64(t_stamp)
        attrs(f)["step"] = Int(step)
        attrs(f)["dt"]   = 0.0
        f["r_grid"]     = collect(rmap.r_grid)
        f["theta_grid"] = collect(rmap.theta_grid)
        f["x_grid"]     = collect(rmap.x_grid)
        f["u_hat"]      = u_hat
        f["ur_hat"]     = ur_hat
        f["uth_hat"]    = uth_hat
    end
    println(" done")
    @printf("  文件: %s (%.1f KB)\n", outpath, filesize(outpath)/1024)

    # ── 5. 读回验证 ──
    print("[5/5] 读回验证物理合理性...")
    h5open(outpath, "r") do f
        u_hat_r = read(f["u_hat"])
        t_r = attrs(f)["time"][]
        @printf("\n  读回: u_hat size=%s, time=%.4e\n", Base.size(u_hat_r), t_r)

        # m=0 应等于周向平均（轴对称分量）
        # 检查 m=0 能量远大于 m>=1（管流以轴对称为主）
        e_m0 = sum(abs.(u_hat_r[1,:,:]).^2)
        e_m1 = sum(abs.(u_hat_r[2,:,:]).^2)
        e_all = sum(abs.(u_hat_r).^2)
        @printf("  能量: m=0=%.4e m=1=%.4e total=%.4e\n", e_m0, e_m1, e_all)
        @printf("  m=0 占比=%.2f%%  m=1 占比=%.2f%%\n",
                e_m0/(e_all+1e-30)*100, e_m1/(e_all+1e-30)*100)

        # 轴向：u_hat[m=0] 应有空间结构（不是常数）
        u_m0_x = dropdims(sum(abs.(u_hat_r[1,:,:]).^2, dims=1), dims=1)
        @printf("  m=0 轴向能量 std/mean=%.4f (>0 说明有空间结构)\n",
                std(u_m0_x)/mean(u_m0_x))

        ok = (e_m0 > e_m1) && (e_m0/(e_all+1e-30) > 0.5)
        println()
        println("-" ^ 60)
        if ok
            println("  [PASS] 钩子管线端到端正确，落盘系数物理合理")
        else
            println("  [FAIL] m=0 未主导，需检查")
        end
        println("-" ^ 60)
    end
end

main()
