"""
visualize.py - 模态分解结果可视化

生成:
    - SPOD 谱图 λ(m, f) 热图
    - 模态空间结构 φ(r, x) 等值线
    - 螺旋度 (m, f) 平面分布
    - 能量传递矩阵热图
    - 轴向输运 F_adv,m(x) 曲线
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
import os


def plot_spod_spectrum(m_arr, f_arr, spectrum, out_path, title="SPOD Spectrum",
                       f_max=None, m_max=None):
    """SPOD 能量谱 λ(m, f) 热图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 限制范围
    m_mask = m_arr <= (m_max if m_max else len(m_arr))
    f_mask = np.abs(f_arr) <= (f_max if f_max else np.max(np.abs(f_arr)))

    spec_show = spectrum[m_mask][:, f_mask]
    m_show = m_arr[m_mask]
    f_show = f_arr[f_mask]

    # 对数色标
    spec_log = np.log10(spec_show + 1e-30)
    im = ax.pcolormesh(f_show, m_show, spec_log, shading="auto", cmap="viridis")
    ax.set_xlabel("Frequency f [Hz]")
    ax.set_ylabel("Azimuthal mode m")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="log₁₀ λ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] SPOD 谱图 -> {out_path}")


def plot_mode_structure(mode, r_grid, x_grid, out_path, title="", component=0):
    """SPOD 模态空间结构 φ(r, x) 等值线。"""
    fig, ax = plt.subplots(figsize=(12, 4))
    # mode: [nrad, nx] 复数，取幅值
    amp = np.abs(mode)
    im = ax.pcolormesh(x_grid, r_grid, amp, shading="auto", cmap="RdBu_r")
    ax.set_xlabel("Axial x [m]")
    ax.set_ylabel("Radial r [m]")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="|φ|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] 模态结构 -> {out_path}")


def plot_helicity_spectrum(m_arr, f_arr, H_spectrum, E_plus, E_minus,
                           out_path, title="Helicity Spectrum"):
    """螺旋度谱 + 手性能量对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 螺旋度谱
    ax = axes[0]
    m_mask = m_arr <= 16
    H_show = H_spectrum[m_mask]
    im = ax.pcolormesh(f_arr, m_arr[m_mask], np.log10(H_show + 1e-30),
                       shading="auto", cmap="RdBu_r")
    ax.set_xlabel("Frequency f [Hz]")
    ax.set_ylabel("Azimuthal mode m")
    ax.set_title("Helicity H(m, f)")
    plt.colorbar(im, ax=ax)

    # 右: 正/负手性能量
    ax = axes[1]
    m_show = m_arr[m_arr <= 16]
    ax.bar(m_show - 0.2, E_plus[m_show], width=0.4, label="E⁺ (right)", color="C0")
    ax.bar(m_show + 0.2, E_minus[m_show], width=0.4, label="E⁻ (left)", color="C3")
    ax.set_xlabel("Azimuthal mode m")
    ax.set_ylabel("Energy")
    ax.set_title("Chiral energy E⁺ vs E⁻")
    ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] 螺旋度谱 -> {out_path}")


def plot_transfer_matrix(T, modes, out_path, title="Energy Transfer Matrix"):
    """能量传递矩阵 T_ij 热图。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.abs(T).max() + 1e-30
    im = ax.imshow(T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
                   extent=[modes[0]-0.5, modes[-1]+0.5, modes[0]-0.5, modes[-1]+0.5])
    ax.set_xlabel("Mode j (output)")
    ax.set_ylabel("Mode i (input)")
    ax.set_title(title)
    ax.set_xticks(modes)
    ax.set_yticks(modes)
    plt.colorbar(im, ax=ax, label="T_ij")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] 传递矩阵 -> {out_path}")


def plot_axial_transport(results, x_grid, out_path, m_show=[0, 1, 2],
                         title="Axial Energy Transport"):
    """轴向输运 F_adv,m(x) 多算例对比。"""
    fig, axes = plt.subplots(1, len(m_show), figsize=(5*len(m_show), 4),
                              sharey=False)
    if len(m_show) == 1:
        axes = [axes]

    colors = {"caseA": "C0", "caseB": "C1", "caseC": "C2"}
    for idx, m in enumerate(m_show):
        ax = axes[idx]
        for name, res in results.items():
            F = res["F_profile"]
            if m < F.shape[0]:
                ax.plot(x_grid, F[m, :], label=name, color=colors.get(name, None))
        ax.set_xlabel("Axial x [m]")
        ax.set_ylabel("F_adv,m(x)")
        ax.set_title(f"m={m}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] 轴向输运 -> {out_path}")
