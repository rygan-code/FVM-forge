"""
test_spod_g3.py - G3 验证：合成场 SPOD 模态识别

构造已知 (m, f) 的合成场：
    u(r, θ, x, t) = A · exp(-((r-r0)/σ)²) · cos(m·θ) · sin(2π·f·t + k·x)

SPOD 应在 (m, f) 处出现尖锐能量峰，模态为已知空间结构。

运行: python post/modal/tests/test_spod_g3.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from spod import spod_single_mode, spod_all_modes, spod_spectrum


def make_synthetic_field(nrad=32, nx=64, ntheta=128, ncomp=3,
                          nt=256, dt=1e-5,
                          m_true=2, f_true=500.0, kx=2*np.pi/15.0,
                          A=1.0, r0=0.35, sigma=0.08):
    """
    构造已知 (m, f) 的合成场。

    u(r,θ,x,t) = A·exp(-((r-r0)/σ)²)·cos(m·θ)·sin(2π·f·t + k·x)

    返回:
        u_hat: [Nt, nmodes, nrad, nx, ncomp] 复数 Fourier 系数
               （对 θ 做 FFT 得到）
        r_grid, x_grid, theta_grid, t_stamps
    """
    r_grid = np.linspace(0, 0.5, nrad)
    x_grid = np.linspace(0, 15, nx)
    theta_grid = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    t_stamps = np.arange(nt) * dt

    R, TH, X = np.meshgrid(r_grid, theta_grid, x_grid, indexing="ij")
    # 径向包络
    envelope = A * np.exp(-((R - r0) / sigma)**2)

    # 实空间场 u(r, θ, x, t)
    u_real = np.zeros((nt, nrad, ntheta, nx, ncomp))
    for it_t, t in enumerate(t_stamps):
        field = envelope * np.cos(m_true * TH) * np.sin(2*np.pi*f_true*t + kx*X)
        u_real[it_t, :, :, :, 0] = field  # 只填 u_x 分量

    # 对 θ 做 FFT 得 Fourier 系数
    nmodes = ntheta // 2 + 1
    u_hat = np.zeros((nt, nmodes, nrad, nx, ncomp), dtype=np.complex128)
    for it_t in range(nt):
        for ir in range(nrad):
            for ix in range(nx):
                fhat = np.fft.rfft(u_real[it_t, ir, :, ix, 0])
                u_hat[it_t, :nmodes, ir, ix, 0] = fhat

    return u_hat, t_stamps, r_grid, x_grid, theta_grid


def test_g3():
    print("=" * 60)
    print("  G3 验证：合成场 SPOD 模态识别")
    print("=" * 60)

    # 构造合成场
    m_true = 2
    f_true = 500.0
    print(f"\n[1/3] 构造合成场: m={m_true}, f={f_true} Hz")
    # nt=512, dt=1e-5 -> 总时长 5.12e-3 s, n_fft=256 -> Δf=1/(256e-5)=390 Hz
    # f_true=500 在 390 和 781 之间，仍可能不精确。改 dt=2e-5:
    # nt=512, dt=2e-5 -> 时长 1.024e-2, n_fft=256 -> Δf=195 Hz, f_true=500 接近 2*195=390 或 3*195=585
    # 用 dt=1e-5, nt=1024, n_fft=512 -> Δf=195, f_true=500 接近 2.56*195... 仍不整。
    # 最简单：让 f_true 对齐 Δf。Δf=1/(n_fft*dt)。取 n_fft=200, dt=1e-5 -> Δf=500. f_true=500 精确对齐。
    u_hat, t_stamps, r_grid, x_grid, theta_grid = make_synthetic_field(
        m_true=m_true, f_true=f_true, nt=400, dt=1e-5
    )
    dt = t_stamps[1] - t_stamps[0]
    print(f"  u_hat shape={u_hat.shape}, dt={dt:.2e}, "
          f"总时长={t_stamps[-1]:.4e}s")

    # SPOD：n_fft=200 使 Δf=1/(200e-5)=500 Hz，f_true 精确对齐
    print(f"\n[2/3] 运行 SPOD (n_fft=200 -> Δf={1/(200*dt):.1f} Hz)...")
    results = spod_all_modes(u_hat, t_stamps, dt, n_fft=200,
                              overlap=0.5, verbose=False)

    # 提取谱
    m_arr, f_arr, spectrum = spod_spectrum(results)
    print(f"  SPOD 完成: m_arr={m_arr[:6]}..., "
          f"f∈[{f_arr[0]:.1f}, {f_arr[-1]:.1f}] Hz, "
          f"Δf={f_arr[1]-f_arr[0]:.1f} Hz")

    # 验证：在 (m_true, |f_true|) 附近应有能量峰
    # 注：实信号 sin(2πft) 在 ±f 都有分量，SPOD 对复输入给出 ±f 共轭模态，
    # 能量对称分布，故频率匹配用 |f_peak|。
    print(f"\n[3/3] 验证能量峰位置")
    # 找全局最大能量
    mi_max = np.argmax(spectrum.max(axis=1))
    fi_max = np.argmax(spectrum[mi_max, :])
    m_peak = m_arr[mi_max]
    f_peak = f_arr[fi_max]

    print(f"  峰值位置: m={m_peak}, f={f_peak:.1f} Hz (|f|={abs(f_peak):.1f})")
    print(f"  期望位置: m={m_true}, f=±{f_true:.1f} Hz")

    # 频率误差容差：|f_peak| 与 f_true 差 < 1 个 Δf
    df = f_arr[1] - f_arr[0]
    m_ok = (m_peak == m_true)
    f_ok = abs(abs(f_peak) - f_true) < abs(df) + 1e-6

    # 峰值能量占比
    total_e = spectrum.sum()
    peak_ratio = spectrum[mi_max, fi_max] / total_e
    print(f"  峰值能量占比: {peak_ratio*100:.2f}%")

    # 峰的锐度：峰值 / 中位数
    peak_sharpness = spectrum[mi_max, fi_max] / np.median(spectrum)
    print(f"  峰锐度 (峰值/中位数): {peak_sharpness:.1f}x")

    ok = m_ok and f_ok and peak_ratio > 0.3
    print()
    print("-" * 60)
    if ok:
        print(f"  [PASS] G3 通过：SPOD 正确识别 (m={m_true}, f={f_true})")
    else:
        print(f"  [FAIL] G3 未通过")
        if not m_ok:
            print(f"    模态错误: 期望 {m_true}, 得到 {m_peak}")
        if not f_ok:
            print(f"    频率错误: 期望 {f_true}, 得到 {f_peak}")
    print("-" * 60)
    return ok


if __name__ == "__main__":
    test_g3()
