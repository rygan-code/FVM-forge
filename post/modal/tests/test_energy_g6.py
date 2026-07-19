"""
test_energy_g6.py - G6 验证：能量传递矩阵守恒

构造已知场，验证 Σ_ij T_ij ≈ 0（非线性项能量守恒）。

运行: python post/modal/tests/test_energy_g6.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from energy_transfer import compute_transfer_matrix, verify_conservation


def make_turbulent_field(nrad=24, nx=48, ntheta=64, nt=32, dt=1e-5, seed=42):
    """
    构造严格不可压 (∇·u=0) 的合成场，用于验证能量传递守恒。

    用螺线管流: u = ∇×A，其中 A = (0, 0, ψ(r,x)·cos(mθ))。
    则 u_x = (1/r)∂(r·A_θ)/∂r ... 简化: 用纯 u_θ = f(r)g(x)cos(mθ),
    u_x = u_r = 0（纯旋转，严格不可压）。

    但纯旋转非线性项 (u·∇)u = -(u_θ²/r) r̂ 非零，传递矩阵有物理内容。
    为有多模态耦合，叠加两个不同 m 的旋转分量。
    """
    r_grid = np.linspace(0.05, 0.5, nrad)
    x_grid = np.linspace(0, 15, nx)
    theta_grid = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    t_stamps = np.arange(nt) * dt

    nmodes = ntheta // 2 + 1
    u_hat = np.zeros((nt, nmodes, nrad, nx, 3), dtype=np.complex128)

    R, X = np.meshgrid(r_grid, x_grid, indexing="ij")
    # 径向包络
    env1 = np.exp(-((R - 0.3) / 0.1)**2) * np.sin(2*np.pi*X/15.0)
    env2 = np.exp(-((R - 0.35) / 0.12)**2) * np.cos(2*np.pi*X/15.0)

    for t in range(nt):
        # m=1 和 m=2 的纯 u_θ 分量（严格不可压：∇·u_θ r̂ = 0 对纯旋转）
        # u_θ = env1·cos(θ) + env2·cos(2θ)
        # Fourier: cos(mθ) -> m 模态
        for ir in range(nrad):
            for ix in range(nx):
                # m=1: cos(θ) -> rfft 给 m=1 实部
                u_hat[t, 1, ir, ix, 2] = env1[ir, ix] * (ntheta/2)
                # m=2: cos(2θ)
                u_hat[t, 2, ir, ix, 2] = env2[ir, ix] * (ntheta/2)
                # 加微弱 u_x (m=0) 使轴向有变化但仍近似不可压
                # u_x = h(x) 独立于 r -> ∂u_x/∂x 非零但 (1/r)∂(ru_r)/∂r=0, ∂u_θ/∂θ=0 -> ∇·u=∂u_x/∂x≠0
                # 为严格不可压，u_x 必须独立于 x。取常数。
        # m=0 常数 u_x
        u_hat[t, 0, :, :, 0] = 0.5 * ntheta  # 常数轴向流

    return u_hat, t_stamps, r_grid, x_grid, theta_grid


def test_g6():
    print("=" * 60)
    print("  G6 验证：能量传递矩阵守恒 (Σ_ij T_ij ≈ 0)")
    print("=" * 60)

    u_hat, t_stamps, r_grid, x_grid, theta_grid = make_turbulent_field()
    print(f"\n  u_hat shape={u_hat.shape}")

    modes = [0, 1, 2, 3]
    print(f"  计算传递矩阵 (modes={modes})...")
    T, modes = compute_transfer_matrix(u_hat, r_grid, x_grid, theta_grid, modes=modes)

    print(f"\n  传递矩阵 T (|T| 缩放):")
    T_max = np.abs(T).max() + 1e-30
    for ii, i in enumerate(modes):
        row = " ".join(f"{T[ii,jj]/T_max:+.2f}" for jj in range(len(modes)))
        print(f"    m={i}: [{row}]")

    print()
    ok, rel_err = verify_conservation(T, modes, tol=0.15)

    print()
    print("-" * 60)
    if ok:
        print(f"  [PASS] G6 通过：能量传递守恒 (误差 {rel_err*100:.1f}%)")
    else:
        print(f"  [FAIL] G6 未通过 (误差 {rel_err*100:.1f}%)")
    print("-" * 60)
    return ok


if __name__ == "__main__":
    test_g6()
