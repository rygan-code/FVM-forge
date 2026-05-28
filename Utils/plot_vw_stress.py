# =============================================================================
#  plot_vw_stress.py
#
#  读取 analyze_vw_stress.jl 输出的累积平均 CSV, 本地出图
#
#  用法:
#    python Utils/plot_vw_stress.py VW_STRESS/
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ─── 出版级排版 ───
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIG_W, FIG_H = 16 / 2.54, 9 / 2.54  # 16cm × 9cm


def load_csvs(data_dir):
    re_stress = pd.read_csv(os.path.join(data_dir, "vw_reynolds_stress.csv"))
    conv = pd.read_csv(os.path.join(data_dir, "vw_convergence.csv"))
    cum_mean = pd.read_csv(os.path.join(data_dir, "vw_cumulative_mean.csv"))
    return re_stress, conv, cum_mean


# ═══════════════════════════════════════════════════════════════
#  图1: 累积 <v'_r v'_θ>(r/R) 随样本数演进
# ═══════════════════════════════════════════════════════════════

def plot_reynolds_stress(re_stress, out_dir):
    samples = sorted(re_stress["n_samples"].unique())
    n = len(samples)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    cmap = plt.cm.viridis

    # 选取若干代表性步 (避免画太多线)
    if n > 20:
        indices = np.unique(np.concatenate([
            [0],
            np.linspace(0, n-1, 15, dtype=int),
            [n-1]
        ]))
    else:
        indices = range(n)

    for plot_idx, idx in enumerate(indices):
        ns = samples[idx]
        df = re_stress[re_stress["n_samples"] == ns]
        step = df["step"].iloc[0]
        c = cmap(idx / max(n - 1, 1))
        alpha = 0.3 + 0.7 * idx / max(n - 1, 1)
        lw = 2.0 if idx == n - 1 else 0.7
        label = None
        if idx == 0:
            label = f"n={ns} (step {step})"
        elif idx == n - 1:
            label = f"n={ns} (step {step})"
        ax.plot(df["r_R"], df["vw_reynolds"], color=c, alpha=alpha, lw=lw, label=label)

    ax.set_xlabel(r"$r / R$")
    ax.set_ylabel(r"$\langle v'_r \, v'_\theta \rangle$")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(samples[0], samples[-1]))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cb.set_label("Cumulative samples", fontsize=9)

    fpath = os.path.join(out_dir, "vw_reynolds_stress.png")
    fig.savefig(fpath)
    plt.close(fig)
    print(f"  → {fpath}")


# ═══════════════════════════════════════════════════════════════
#  图2: 收敛历程 — Δrms 和 Δmax 随样本数下降
# ═══════════════════════════════════════════════════════════════

def plot_convergence(conv, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 1.4, FIG_H))

    # 左: Δrms (log scale)
    ax1 = axes[0]
    valid = conv["delta_rms"] > 0
    ax1.semilogy(conv.loc[valid, "n_samples"], conv.loc[valid, "delta_rms"],
                 "o-", ms=3, lw=1.2, color="#2196F3", label=r"$\Delta_{\rm rms}$")
    ax1.semilogy(conv.loc[valid, "n_samples"], conv.loc[valid, "delta_max"],
                 "s-", ms=2.5, lw=0.8, color="#FF9800", alpha=0.7, label=r"$\Delta_{\rm max}$")
    ax1.set_xlabel("Cumulative samples (n)")
    ax1.set_ylabel(r"$\Delta$ between consecutive averages")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, ls=":", alpha=0.4)

    # 右: |<v'_r v'_θ>|_rms 绝对值
    ax2 = axes[1]
    ax2.plot(conv["n_samples"], conv["vw_re_rms"], "o-", ms=3, lw=1.2, color="#4CAF50")
    ax2.set_xlabel("Cumulative samples (n)")
    ax2.set_ylabel(r"$\mathrm{RMS}\;\langle v'_r v'_\theta \rangle$")
    ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fpath = os.path.join(out_dir, "vw_convergence.png")
    fig.savefig(fpath)
    plt.close(fig)
    print(f"  → {fpath}")


# ═══════════════════════════════════════════════════════════════
#  图3: 最终 Reynolds stress + 平均速度径向分布
# ═══════════════════════════════════════════════════════════════

def plot_final_profile(re_stress, out_dir):
    # 取最后一个样本的数据
    last_n = re_stress["n_samples"].max()
    df_re = re_stress[re_stress["n_samples"] == last_n]
    step = df_re["step"].iloc[0]

    # 加载平均速度 profile
    mean_path = os.path.join(out_dir, "mean_velocity_profile.csv")
    has_mean = os.path.isfile(mean_path)

    ncols = 2 if has_mean else 1
    fig, axes = plt.subplots(1, ncols, figsize=(FIG_W * (1.0 if ncols == 1 else 1.4), FIG_H))
    if ncols == 1:
        axes = [axes]

    # 左: Reynolds stress
    ax1 = axes[0]
    ax1.plot(df_re["r_R"], df_re["vw_reynolds"], "-", lw=1.8, color="#E91E63")
    ax1.axhline(0, color="gray", ls="--", lw=0.5)
    ax1.set_xlabel(r"$r / R$")
    ax1.set_ylabel(r"$\langle v'_r \, v'_\theta \rangle$")
    ax1.set_title(f"Reynolds stress (n={last_n}, step {step})")
    ax1.grid(True, ls=":", alpha=0.4)

    # 右: 平均速度分量
    if has_mean:
        df_mean = pd.read_csv(mean_path)
        ax2 = axes[1]
        ax2.plot(df_mean["r_R"], df_mean["vr_mean"], "-", lw=1.4, color="#2196F3",
                 label=r"$\langle v_r \rangle$")
        ax2.plot(df_mean["r_R"], df_mean["vt_mean"], "-", lw=1.4, color="#F44336",
                 label=r"$\langle v_\theta \rangle$")
        ax2.axhline(0, color="gray", ls="--", lw=0.5)
        ax2.set_xlabel(r"$r / R$")
        ax2.set_ylabel("Mean velocity")
        ax2.set_title(f"Mean profiles ({len(re_stress['step'].unique())} snapshots)")
        ax2.legend(loc="best", frameon=True)
        ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fpath = os.path.join(out_dir, "vw_final_profile.png")
    fig.savefig(fpath)
    plt.close(fig)
    print(f"  → {fpath}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) >= 2 else "VW_STRESS"

    if not os.path.isdir(data_dir):
        print(f"ERROR: directory '{data_dir}' not found")
        sys.exit(1)

    print("=" * 60)
    print(f"  Plotting ⟨v'_r v'_θ⟩ from: {data_dir}")
    print("=" * 60)

    re_stress, conv, cum_mean = load_csvs(data_dir)
    n_total = re_stress["n_samples"].max()
    steps = sorted(re_stress["step"].unique())
    print(f"  Steps: {len(steps)} [{steps[0]} → {steps[-1]}]")
    print(f"  Cumulative samples: {n_total}")

    plot_reynolds_stress(re_stress, data_dir)
    plot_convergence(conv, data_dir)
    plot_final_profile(re_stress, data_dir)

    print("=" * 60)
    print("  Done! Output:")
    print(f"    - {data_dir}/vw_reynolds_stress.png   累积 Reynolds 应力演进")
    print(f"    - {data_dir}/vw_convergence.png       Δrms/Δmax 收敛历程")
    print(f"    - {data_dir}/vw_final_profile.png     最终 profile + 均值分量")
    print("=" * 60)

