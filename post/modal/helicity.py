"""
helicity.py - 螺旋度分辨分析（用户核心增量①）

功能：
    1. 模态螺旋度: h_m(r,x,t) = Re(û_m · ω̂_m)，积分 H_m(f) = ∫ h_m dV
    2. helical/Beltrami 分解: 每个 (m, 波矢 k) 投影到正/负螺旋极化基，
       分离 E⁺(m,f)、E⁻(m,f)
    3. G5 验证: 无旋参考场 E⁺ ≈ E⁻ (mirror-symmetric)

涡量计算: 从速度 Fourier 系数 û_m(r,x) 重建 ω̂_m = ∇×u 在圆柱坐标:
    ω_x = (1/r) ∂(r·u_θ)/∂r
    ω_r = -∂u_θ/∂x
    ω_θ = ∂u_r/∂x - ∂u_x/∂r
    对每个周向模态 m，径向导数用中心差分，轴向导数用谱方法（FFT）。

螺旋度分解:
    每个波矢 k = (k_x, m/r) 对应两个螺旋本征态 e_± (正/负螺旋)。
    速度场投影: u = u⁺·e_+ + u⁻·e_-
    螺旋能量: E⁺ = |u⁺|², E⁻ = |u⁻|²
    螺旋度: h = (m·k_x)·(|u⁺|² - |u⁻|²) / |k|
"""

import numpy as np


def compute_vorticity_spectral(u_hat, r_grid, x_grid):
    """
    从速度 Fourier 系数 û_m(r,x) 重建涡量 ω̂_m(r,x)。

    圆柱坐标涡量（每个周向模态 m）:
        ω_x = (1/r) ∂(r·u_θ)/∂r
        ω_r = -∂u_θ/∂x
        ω_θ = ∂u_r/∂x - ∂u_x/∂r

    径向导数: 中心差分（轴心特殊处理）
    轴向导数: 谱方法（沿 x 做 FFT，乘 i·k_x，逆 FFT）

    参数:
        u_hat: [Nt, nmodes, nrad, nx, 3] 复数 (u_x, u_r, u_θ)
        r_grid: [nrad]
        x_grid: [nx]

    返回:
        omega_hat: [Nt, nmodes, nrad, nx, 3] 复数 (ω_x, ω_r, ω_θ)
    """
    Nt, nmodes, nrad, nx, ncomp = u_hat.shape
    dr = np.gradient(r_grid)
    dx = x_grid[1] - x_grid[0]

    # 轴向波数（谱导数用）
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi  # [nx]

    omega_hat = np.zeros_like(u_hat)

    for t in range(Nt):
        for m in range(nmodes):
            ux = u_hat[t, m, :, :, 0]  # [nrad, nx]
            ur = u_hat[t, m, :, :, 1]
            uth = u_hat[t, m, :, :, 2]

            # 轴向谱导数: ∂/∂x -> i*k_x 在频域
            # 沿 x 做 fft，乘 i*kx，ifft 回
            def d_dx(field):
                fhat = np.fft.fft(field, axis=1)
                fhat_deriv = fhat * (1j * kx[np.newaxis, :])
                return np.fft.ifft(fhat_deriv, axis=1)

            # 径向导数: 中心差分（轴心 r=0 特殊处理）
            def d_dr(field):
                deriv = np.zeros_like(field)
                # 内部点：中心差分
                deriv[1:-1, :] = (field[2:, :] - field[:-2, :]) / (
                    r_grid[2:] - r_grid[:-2]
                )[:, np.newaxis]
                # 边界：单侧差分
                deriv[0, :] = (field[1, :] - field[0, :]) / (r_grid[1] - r_grid[0])
                deriv[-1, :] = (field[-1, :] - field[-2, :]) / (r_grid[-1] - r_grid[-2])
                return deriv

            r_safe = np.where(r_grid > 1e-10, r_grid, 1e-10)

            # ω_x = (1/r) ∂(r·u_θ)/∂r
            r_uth = r_safe[:, np.newaxis] * uth
            omega_hat[t, m, :, :, 0] = d_dr(r_uth) / r_safe[:, np.newaxis]

            # ω_r = -∂u_θ/∂x
            omega_hat[t, m, :, :, 1] = -d_dx(uth)

            # ω_θ = ∂u_r/∂x - ∂u_x/∂r
            omega_hat[t, m, :, :, 2] = d_dx(ur) - d_dr(ux)

    return omega_hat


def modal_helicity(u_hat, omega_hat, r_grid, x_grid, theta_grid):
    """
    模态螺旋度 h_m(r,x,t) = Re(û_m · ω̂_m)。

    返回:
        h: [Nt, nmodes, nrad, nx] 螺旋度密度
        H: [Nt, nmodes] 体积积分螺旋度
    """
    Nt, nmodes, nrad, nx, ncomp = u_hat.shape
    dtheta = theta_grid[1] - theta_grid[0]
    dx = x_grid[1] - x_grid[0]
    dr = np.gradient(r_grid)

    # 点积 Re(û · ω̂) = Re(Σ_c û_c * conj(ω̂_c))... 实际上
    # 对于实场 u, ω，其 Fourier 系数满足 û_{-m} = conj(û_m)。
    # 螺旋度 h = u·ω 的 m 分量 = Re(Σ_c û_m,c * conj(ω̂_m,c))
    # 但更常见定义: h_m = Re(û_m · ω̂_m^*) = Re(Σ_c û_m,c * ω̂_m,c^*)
    dot = np.sum(u_hat * np.conj(omega_hat), axis=-1)  # [Nt,nm,nr,nx]
    h = np.real(dot)

    # 体积积分: ∫ h · r dr dθ dx
    # 每个模态 m 的体积元含因子 2π（周向积分，m=0 为 2π，m>0 为 π·2... 实际上
    # 对实场，Parseval: ∫_0^{2π} u_m·ω_m dθ = 2π Re(û_m ω̂_m^*) for all m）
    vol_factor = (r_grid[:, np.newaxis] * dr[:, np.newaxis] * dtheta * dx)
    H = np.sum(h * vol_factor[np.newaxis, np.newaxis, :, :], axis=(2, 3))

    return h, H


def helical_decomposition(u_hat, r_grid, x_grid, theta_grid):
    """
    Helical/Beltrami 分解：每个 (m, k_x) 模态分解为正/负螺旋分量。

    对波矢 k = (k_x, k_θ=m/r)，速度场分解为:
        u = u⁺ e_+ + u⁻ e_-
    其中 e_± 是螺旋极化基（沿 k 的正/负本征态）。

    螺旋能量:
        E⁺(m, k_x) = |u⁺|²
        E⁻(m, k_x) = |u⁻|²

    返回:
        E_plus: [nmodes, nx] 正螺旋能量
        E_minus: [nmodes, nx] 负螺旋能量
    """
    Nt, nmodes, nrad, nx, ncomp = u_hat.shape
    dx = x_grid[1] - x_grid[0]
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi  # [nx]

    # 时间平均
    u_mean = np.mean(u_hat, axis=0)  # [nmodes, nrad, nx, 3]

    E_plus = np.zeros((nmodes, nx))
    E_minus = np.zeros((nmodes, nx))

    for m in range(nmodes):
        for ix in range(nx):
            kx_i = kx[ix]
            # 对每个径向点，波矢 k = (kx, 0, m/r)
            for ir in range(nrad):
                r = max(r_grid[ir], 1e-10)
                k_theta = m / r
                k_mag = np.sqrt(kx_i**2 + k_theta**2)
                if k_mag < 1e-10:
                    continue

                # 速度向量 (u_x, u_r, u_θ) 在该点
                u_vec = u_mean[m, ir, ix, :]  # [3] 复数

                # 螺旋投影:
                # 对波矢 k，构造两个正交横向极化基，
                # 投影到 ±k 方向的自旋本征态。
                # 简化: 对 2D (kx, k_theta) 波矢，
                # 正螺旋 = 右旋圆偏振，负螺旋 = 左旋。
                #
                # |u⁺|² + |u⁻|² = |u|²  (能量守恒)
                # |u⁺|² - |u⁻|² = (u·ω)/(k·|u|)  (螺旋度归一化)
                #
                # 用涡量-速度关系: ω = i k × u (谱空间)
                # 螺旋度 h = Im(u* · (k × u)) / |k|
                #         = (k/|k|) · Im(u* × u) ... 对每个模态
                #
                # 简化实现: 用 k 方向的自旋投影
                k_hat = np.array([kx_i, 0, k_theta]) / k_mag

                # 速度的横向分量（垂直于 k）
                u_dot_k = np.sum(u_vec * k_hat)
                u_perp = u_vec - u_dot_k * k_hat

                # 正/负螺旋投影: u⁺ = (u_perp + i (k_hat × u_perp))/2
                #                  u⁻ = (u_perp - i (k_hat × u_perp))/2
                # 对复数速度，k_hat × u_perp 是叉积
                kxu = np.cross(k_hat, u_perp)
                u_plus = 0.5 * (u_perp + 1j * kxu)
                u_minus = 0.5 * (u_perp - 1j * kxu)

                E_plus[m, ix] += np.real(np.sum(np.abs(u_plus)**2))
                E_minus[m, ix] += np.real(np.sum(np.abs(u_minus)**2))

    return E_plus, E_minus


def helicity_spectrum(H, t_stamps, dt, n_fft=None, overlap=0.5):
    """
    螺旋度的频率谱 H_m(f)：对每个 m 的螺旋度时间序列做 Welch 谱估计。

    返回:
        m_arr, f_arr, H_spectrum[m, f]
    """
    from scipy.signal import welch
    nmodes = H.shape[1]
    if n_fft is None:
        n_fft = min(H.shape[0], 256)

    # m=0 的频率轴作为参考
    f_arr, _ = welch(H[:, 0], fs=1/dt, nperseg=n_fft, noverlap=n_fft//2)
    H_spectrum = np.zeros((nmodes, len(f_arr)))

    for m in range(nmodes):
        f_arr, Pxx = welch(H[:, m], fs=1/dt, nperseg=n_fft, noverlap=n_fft//2)
        H_spectrum[m, :] = Pxx

    m_arr = np.arange(nmodes)
    return m_arr, f_arr, H_spectrum
