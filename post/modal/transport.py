"""
transport.py - 轴向非局部输运预算

局部（沿 x 分段）能量预算:
    ∂_t E(x) = -∂_x F_adv(x) - Diss(x) + Prod(x)

按周向模态 m 分解:
    ∂_t E_m(x) = -∂_x F_adv,m(x) + (非线性传递) + (耗散-生产残差)

轴向对流通量 F_adv,m(x) = ∫ ½|u_m|² · u_x dA  (轴向动能输运)

对比三算例: 非均匀旋转如何改变轴向输运 F_adv,m(x) 沿 x 的分布。
"""

import numpy as np


def axial_energy_profile(u_hat, r_grid, x_grid, theta_grid):
    """
    计算各周向模态的轴向动能分布 E_m(x) 和轴向通量 F_adv,m(x)。

    E_m(x) = ∫_A ½|û_m|² r dr dθ  (截面动能)
    F_adv,m(x) = ∫_A ½|û_m|² · u_x,mean dA  (轴向对流输运)

    参数:
        u_hat: [Nt, nmodes, nrad, nx, 3] 复数 (u_x, u_r, u_θ)
        r_grid, x_grid, theta_grid

    返回:
        E_profile: [Nt, nmodes, nx] 截面动能沿 x
        F_profile: [Nt, nmodes, nx] 轴向通量沿 x
    """
    Nt, nmodes, nrad, nx, ncomp = u_hat.shape
    dr = np.gradient(r_grid)
    dtheta = theta_grid[1] - theta_grid[0]
    r_safe = np.where(r_grid > 1e-10, r_grid, 1e-10)

    # 截面积分权重: r·dr·dθ (周向积分对实场 m=0 给 2π, m>0 给 π·2=2π... 统一 2π)
    area_weight = r_safe * dr * dtheta  # [nrad]

    E_profile = np.zeros((Nt, nmodes, nx))
    F_profile = np.zeros((Nt, nmodes, nx))

    # 时间平均轴向速度（用于通量）
    u_x_mean = np.mean(np.real(u_hat[:, 0, :, :, 0]), axis=0)  # [nrad, nx], m=0

    for t in range(Nt):
        for m in range(nmodes):
            # 动能密度 ½|û_m|² = ½ Σ_c |û_m,c|²
            ke = 0.5 * np.sum(np.abs(u_hat[t, m, :, :, :])**2, axis=-1)  # [nrad, nx]
            # 截面积分 E_m(x) = ∫ ke · r dr dθ
            E_profile[t, m, :] = np.sum(ke * area_weight[:, np.newaxis], axis=0)

            # 轴向通量 F_adv,m(x) = ∫ ke · u_x,mean dA
            F_profile[t, m, :] = np.sum(
                ke * u_x_mean * area_weight[:, np.newaxis], axis=0
            )

    return E_profile, F_profile


def energy_budget_x(E_profile, F_profile, dt):
    """
    从 E_m(x,t) 和 F_m(x,t) 估计局部能量预算:
        ∂_t E_m ≈ -∂_x F_adv,m + residual

    返回时间平均的:
        dE_dt: ∂_t E_m(x)  (有限差分)
        dF_dx: -∂_x F_adv,m(x)  (轴向通量散度，负号=收入)
        residual: 耗散-生产+非线性传递 (dE_dt - dF_dx)
    """
    Nt, nmodes, nx = E_profile.shape
    dx = np.gradient  # 用 np.gradient

    # ∂_t E (中心差分)
    dE_dt = np.zeros_like(E_profile)
    dE_dt[1:-1, :, :] = (E_profile[2:, :, :] - E_profile[:-2, :, :]) / (2 * dt)

    # ∂_x F (沿 x 中心差分)
    dF_dx = np.zeros_like(F_profile)
    for t in range(Nt):
        for m in range(nmodes):
            dF_dx[t, m, :] = np.gradient(F_profile[t, m, :])

    # 时间平均
    dE_dt_mean = np.mean(dE_dt, axis=0)  # [nmodes, nx]
    dF_dx_mean = np.mean(dF_dx, axis=0)
    residual = dE_dt_mean - (-dF_dx_mean)  # ∂_t E = -∂_x F + residual

    return dE_dt_mean, -dF_dx_mean, residual


def compare_axial_transport(cases, r_grid, x_grid, theta_grid):
    """
    对比多算例的轴向输运 F_adv,m(x)。

    参数:
        cases: dict {name: u_hat} 各算例的速度系数
    返回:
        results: dict {name: {E_profile, F_profile}} (时间平均)
    """
    results = {}
    for name, u_hat in cases.items():
        E_prof, F_prof = axial_energy_profile(u_hat, r_grid, x_grid, theta_grid)
        results[name] = {
            "E_profile": np.mean(E_prof, axis=0),  # 时间平均 [nmodes, nx]
            "F_profile": np.mean(F_prof, axis=0),
        }
        print(f"[transport] {name}: E shape={results[name]['E_profile'].shape}")
    return results
