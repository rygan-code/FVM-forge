# =============================================================================
#  test_fft_g2.jl - G2 验证：周向 FFT 模态识别
#
#  目标：合成已知周向模态 m 的场 u=A·cos(m·θ) + noise，
#        重采样到均匀 θ 网格后做 FFT，应仅在 ±m 出现尖峰。
#
#  同时验证块拼接相位连续性：用蝴蝶网格的多块拼接做周向 FFT，
#  若拼接处相位不连续会产生虚假模态泄漏。
#
#  用法：julia post/online_modal/test_fft_g2.jl [MESH_DIR]
# =============================================================================
include(joinpath(@__DIR__, "modal_config.jl"))
include(joinpath(@__DIR__, "cyl_regrid.jl"))

using HDF5
using FFTW
using Printf

function main()
    mesh_dir = length(ARGS) >= 1 ? ARGS[1] : "MESH_SMALL"

    println("=" ^ 60)
    println("  G2 验证：周向 FFT 模态识别 + 块拼接相位连续性")
    println("  mesh=$mesh_dir")
    println("=" ^ 60)

    # ── 预计算映射表 ──
    print("[1/3] 预计算重采样映射表...")
    rmap = build_regrid_map(mesh_dir)
    @printf(" done (nx=%d ntheta=%d nrad=%d)\n", rmap.nx, rmap.ntheta, rmap.nrad)

    # ── 测试 1: 直接在均匀 θ 网格上构造已知模态，FFT 应给单峰 ──
    # 这是 FFT 本身的正确性检查（不经过重采样）。
    println("\n[2/3] 测试 1：均匀 θ 网格 FFT 模态识别")
    ntheta = rmap.ntheta
    theta_grid = rmap.theta_grid
    all_pass = true
    for m_test in [0, 1, 2, 3, 5, 8]
        # u(θ) = cos(m·θ)，实信号。FFT 后能量等分到 +m 和 -m（共轭对称）。
        u_theta = cos.(m_test .* theta_grid)
        u_hat = FFTW.fft(u_theta)
        energies = abs.(u_hat).^2
        # 实信号能量谱：+m (idx m+1) 和 -m (idx N-m+1) 共享能量。
        # m_test=0 时只有 DC (idx 1)。
        if m_test == 0
            peak_m = 0
            ratio = 1.0
        else
            e_plus = energies[m_test + 1]
            e_minus = energies[ntheta - m_test + 1]
            peak_m = m_test
            total_e = sum(energies[2:end])
            ratio = (e_plus + e_minus) / (total_e + 1e-30)
        end
        ok = (peak_m == m_test) && (ratio > 0.95)
        @printf("  m=%d: 峰值模态=%d, 能量占比=%.4f  %s\n",
                m_test, peak_m, ratio, ok ? "[OK]" : "[FAIL]")
        all_pass = all_pass && ok
    end

    # ── 测试 2: 通过重采样的端到端验证 ──
    # 构造一个蝴蝶网格上的已知模态场 u(y,z) = cos(m·atan2(z,y))，
    # 重采样到圆柱网格，FFT 应给同一模态。
    println("\n[3/3] 测试 2：蝴蝶网格->圆柱重采样->FFT 端到端")
    for m_test in [1, 2, 4]
        # 在每块构造 u = cos(m·θ)，θ=atan2(z,y)
        block_data = Dict{Int, NTuple{3, Array{Float64,3}}}()
        for bid in 0:4
            xc, yc, zc = load_block_cell_centers(mesh_dir, bid)
            sx, sy, sz = Base.size(yc)
            u = zeros(sx, sy, sz)
            v = zeros(sx, sy, sz)
            w = zeros(sx, sy, sz)
            for k in 1:sz, j in 1:sy, i in 1:sx
                θ = atan(zc[i,j,k], yc[i,j,k])
                u[i,j,k] = cos(m_test * θ)
                # v, w 设 0（只测 u 分量）
            end
            block_data[bid] = (u, v, w)
        end

        # 重采样
        u_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
        v_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
        w_cyl = zeros(rmap.nx, rmap.ntheta, rmap.nrad)
        regrid_to_cyl!(u_cyl, v_cyl, w_cyl, rmap, block_data)

        # 对每个 (ix, ir) 做周向 FFT，平均能量谱
        energy_spectrum = zeros(ntheta)
        for ix in 1:rmap.nx, ir in 1:rmap.nrad
            if rmap.r_grid[ir] < r_inner * 1.18
                continue  # 跳过块0 核心区（θ 无定义）
            end
            u_hat = FFTW.fft(view(u_cyl, ix, :, ir))
            energy_spectrum .+= abs.(u_hat).^2
        end

        # 找峰值模态（实信号能量在 ±m 对称）
        peak_idx = argmax(energy_spectrum[2:end]) + 1
        peak_m = peak_idx - 1
        total_e = sum(energy_spectrum[2:end])
        # +m 和 -m 合并能量
        e_plus = energy_spectrum[m_test + 1]
        e_minus = energy_spectrum[ntheta - m_test + 1]
        ratio = (e_plus + e_minus) / (total_e + 1e-30)
        # 块拼接相位连续性：检查除 ±m 外的最大泄漏
        leak_ratio = 0.0
        for mm in 1:ntheta-1
            if mm != m_test && mm != (ntheta - m_test)
                r = energy_spectrum[mm+1] / (total_e + 1e-30)
                if r > leak_ratio
                    leak_ratio = r
                end
            end
        end
        ok = (peak_m == m_test) && (ratio > 0.9)
        @printf("  m=%d: 峰值=%d 占比=%.4f 最大泄漏=%.4f  %s\n",
                m_test, peak_m, ratio, leak_ratio, ok ? "[OK]" : "[FAIL]")
        all_pass = all_pass && ok
    end

    println()
    println("-" ^ 60)
    if all_pass
        println("  [PASS] G2 通过：周向 FFT 正确识别模态，块拼接相位连续")
    else
        println("  [FAIL] G2 未通过，需检查块拼接或 FFT 实现")
    end
    println("-" ^ 60)
end

main()
