"""
energy_transfer.py - 模态间能量/螺旋度传递矩阵（用户核心增量②）

Galerkin 能量方程的模态间通量:
    dE_i/dt = Σ_j T_ij + P_i - D_i
    T_ij = <φ_i, (u·∇)u, φ_j>  (i 向 j 的能量传递)

物理意义:
    - T_ij > 0: 模态 i 从 j 获取能量
    - T_ij < 0: 模态 i 向 j 输出能量
    - Σ_ij T_ij = 0: 传递是零和的（非线性项守恒，G6 验证）
    - 区分同手性 (E⁺↔E⁺, E⁻↔E⁻) vs 异手性 (E⁺↔E⁻) 通道

实现:
    对 SPOD 模态 φ_i(r,x;f)，非线性项 (u·∇)u 在圆柱坐标展开
    （含曲率项 1/r）。三重相关在频域计算。
"""

import numpy as np


def compute_transfer_matrix(u_hat, r_grid, x_grid, theta_grid, modes=None):
    """
    计算周向模态间的能量传递矩阵 T_{ij}。

    T_{ij} = Re < û_i*, (u·∇)û_j > （体积加权）

    非线性项 (u·∇)u 在圆柱坐标 (x, r, θ):
        [(u·∇)u]_x = u_x ∂u_x/∂x + u_r ∂u_x/∂r + (u_θ/r) ∂u_x/∂θ
        [(u·∇)u]_r = u_x ∂u_r/∂x + u_r ∂u_r/∂r + (u_θ/r) ∂u_r/∂θ - u_θ²/r
        [(u·∇)u]_θ = u_x ∂u_θ/∂x + u_r ∂u_θ/∂r + (u_θ/r) ∂u_θ/∂θ + u_r u_θ/r

    在周向 Fourier 空间，∂/∂θ -> i·m，卷积 u·∇ 变为模态求和。

    参数:
        u_hat: [Nt, nmodes, nrad, nx, 3] 复数 (u_x, u_r, u_θ)
        r_grid, x_grid, theta_grid: 网格
        modes: 要计算的模态列表 (None = 全部)

    返回:
        T: [nmodes, nmodes] 能量传递矩阵 (时间平均)
        T_helicity: [nmodes, nmodes] 螺旋度传递矩阵
    """
    Nt, nmodes, nrad, nx, ncomp = u_hat.shape
    dr = np.gradient(r_grid)
    dx = x_grid[1] - x_grid[0]
    dtheta = theta_grid[1] - theta_grid[0]

    if modes is None:
        modes = list(range(min(nmodes, 8)))  # 默认前 8 个模态
    n_calc = len(modes)

    # 时间平均的模态系数
    u_mean = np.mean(u_hat, axis=0)  # [nmodes, nrad, nx, 3]

    # 轴向波数（谱导数）
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi

    def d_dx(field):
        """轴向谱导数，field: [nrad, nx] 复数"""
        fhat = np.fft.fft(field, axis=1)
        return np.fft.ifft(fhat * (1j * kx[np.newaxis, :]), axis=1)

    def d_dr(field):
        """径向中心差分，field: [nrad, nx]"""
        deriv = np.zeros_like(field)
        deriv[1:-1, :] = (field[2:, :] - field[:-2, :]) / (
            r_grid[2:] - r_grid[:-2])[:, np.newaxis]
        deriv[0, :] = (field[1, :] - field[0, :]) / (r_grid[1] - r_grid[0])
        deriv[-1, :] = (field[-1, :] - field[-2, :]) / (r_grid[-1] - r_grid[-2])
        return deriv

    r_safe = np.where(r_grid > 1e-10, r_grid, 1e-10)

    # 体积权重 r·dr·dθ·dx
    vol = (r_safe * dr * dtheta * dx)  # [nrad]

    T = np.zeros((n_calc, n_calc))
    T_helicity = np.zeros((n_calc, n_calc))

    # 对每个目标模态 j，计算 N_j = (u·∇)u_j = Σ_{m1+m2=j} [û_{m1}·∇]û_{m2}
    # （Fourier 空间卷积：乘积的 j 模态 = Σ_{m1+m2=j} û_{m1} ∂û_{m2}）
    # ∂/∂θ 作用于 m2 模态 -> i·m2
    for jj, j in enumerate(modes):
        N_x_j = np.zeros((nrad, nx), dtype=np.complex128)
        N_r_j = np.zeros((nrad, nx), dtype=np.complex128)
        N_th_j = np.zeros((nrad, nx), dtype=np.complex128)

        # 卷积: m1 + m2 = j （或 m1 - m2 = j 考虑 ±m 共轭）
        # 对实场，û_{-m} = conj(û_m)，所以 j 模态的卷积覆盖所有 m1:
        #   N_j = Σ_{m1} [û_{m1}·∇]û_{j-m1}
        for m1 in range(nmodes):
            m2 = j - m1
            if m2 < -nmodes + 1 or m2 > nmodes - 1:
                continue
            # 处理 m2 < 0: û_{m2} = conj(û_{|m2|})（实场共轭对称）
            if m2 < 0:
                m2_idx = -m2
                ux_m2 = np.conj(u_mean[m2_idx, :, :, 0])
                ur_m2 = np.conj(u_mean[m2_idx, :, :, 1])
                uth_m2 = np.conj(u_mean[m2_idx, :, :, 2])
            else:
                ux_m2 = u_mean[m2, :, :, 0]
                ur_m2 = u_mean[m2, :, :, 1]
                uth_m2 = u_mean[m2, :, :, 2]

            ux_m1 = u_mean[m1, :, :, 0]
            ur_m1 = u_mean[m1, :, :, 1]
            uth_m1 = u_mean[m1, :, :, 2]

            # 谱导数
            dux_m2_dx = d_dx(ux_m2)
            dux_m2_dr = d_dr(ux_m2)
            dur_m2_dx = d_dx(ur_m2)
            dur_m2_dr = d_dr(ur_m2)
            duth_m2_dx = d_dx(uth_m2)
            duth_m2_dr = d_dr(uth_m2)

            # [û_{m1}·∇]û_{m2}, ∂/∂θ -> i·m2
            N_x_j += (ux_m1 * dux_m2_dx + ur_m1 * dux_m2_dr
                      + (uth_m1 / r_safe[:, np.newaxis]) * (1j * m2) * ux_m2)
            N_r_j += (ux_m1 * dur_m2_dx + ur_m1 * dur_m2_dr
                      + (uth_m1 / r_safe[:, np.newaxis]) * (1j * m2) * ur_m2
                      - uth_m1 * uth_m2 / r_safe[:, np.newaxis])
            N_th_j += (ux_m1 * duth_m2_dx + ur_m1 * duth_m2_dr
                       + (uth_m1 / r_safe[:, np.newaxis]) * (1j * m2) * uth_m2
                       + ur_m1 * uth_m2 / r_safe[:, np.newaxis])

        # 与各模态 i 做内积: T_ij = Re <û_i*, N_j> (体积加权)
        for ii, i in enumerate(modes):
            ux_i = u_mean[i, :, :, 0]
            ur_i = u_mean[i, :, :, 1]
            uth_i = u_mean[i, :, :, 2]

            dot_e = (np.conj(ux_i) * N_x_j
                     + np.conj(ur_i) * N_r_j
                     + np.conj(uth_i) * N_th_j)
            T[ii, jj] = np.real(np.sum(dot_e * vol[:, np.newaxis]))

    return T, modes


def verify_conservation(T, modes, tol=0.1, verbose=True):
    """
    G6 验证: 传递矩阵守恒 Σ_ij T_ij ≈ 0。

    非线性项 (u·∇)u 是能量守恒的，传递矩阵的行和/列和应为零。
    """
    T_sum = T.sum()
    T_scale = np.abs(T).sum() + 1e-30
    rel_err = abs(T_sum) / T_scale

    if verbose:
        print(f"[G6 能量传递守恒]")
        print(f"  Σ T_ij = {T_sum:.6e}")
        print(f"  Σ|T_ij| = {T_scale:.6e}")
        print(f"  相对误差 = {rel_err*100:.2f}%")
        print(f"  容差 = {tol*100:.0f}%")

    # 行和（每个模态的净获得）与列和（净输出）
    row_sum = T.sum(axis=1)  # 模态 i 的净获得
    col_sum = T.sum(axis=0)  # 模态 j 的净输出
    if verbose:
        print(f"  行和 (净获得) max|·| = {np.max(np.abs(row_sum)):.4e}")
        print(f"  列和 (净输出) max|·| = {np.max(np.abs(col_sum)):.4e}")

    ok = rel_err < tol
    if verbose:
        print(f"  判定: {'[PASS] 守恒' if ok else '[FAIL] 不守恒'}")
    return ok, rel_err


def classify_chirality_channels(T, E_plus, E_minus, modes):
    """
    区分同手性 vs 异手性能量传递通道。

    需要 helical 分解的 E⁺/E⁻ 信息来标记每个模态的手性倾向。
    简化: 用模态螺旋度符号判定手性。

    返回:
        T_same: 同手性通道传递 (E⁺↔E⁺ 或 E⁻↔E⁻)
        T_cross: 异手性通道传递 (E⁺↔E⁻)
    """
    # 用 E_plus - E_minus 的符号判定每个模态手性
    chirality = np.sign(E_plus[modes] - E_minus[modes])  # +1: 右旋, -1: 左旋

    T_same = 0.0
    T_cross = 0.0
    for ii in range(len(modes)):
        for jj in range(len(modes)):
            if ii == jj:
                continue
            if chirality[ii] == chirality[jj] and chirality[ii] != 0:
                T_same += abs(T[ii, jj])
            else:
                T_cross += abs(T[ii, jj])

    return T_same, T_cross
