"""
compare_cases.py - 三算例对比分析主脚本

caseA (Ro 均匀=1.0 基准)
caseB (diffrot=true, Ro 1->1)
caseC (diffrot=false, Ro 0->1 真非均匀)

对比:
    1. SPOD 谱 λ(m,f) 差异
    2. 螺旋度 H(m,f) + 手性 E⁺/E⁻ 差异
    3. 能量传递矩阵 T_ij 差异
    4. 轴向输运 F_adv,m(x) 差异

用法:
    python post/modal/compare_cases.py --caseA MODAL_A --caseB MODAL_B --caseC MODAL_C
    python post/modal/compare_cases.py  # 用测试数据演示
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from io_coeff import prepare_for_spod
from spod import spod_all_modes, spod_spectrum
from helicity import compute_vorticity_spectral, modal_helicity, helical_decomposition
from energy_transfer import compute_transfer_matrix
from transport import compare_axial_transport
from visualize import (plot_spod_spectrum, plot_helicity_spectrum,
                       plot_transfer_matrix, plot_axial_transport)


def analyze_single_case(modal_dir, label, out_dir, verbose=True):
    """对单个算例做完整模态分析。"""
    if verbose:
        print(f"\n{'='*60}\n  分析 {label}: {modal_dir}\n{'='*60}")

    # 加载 + SPOD
    u_hat, t_stamps, dt, grid, info = prepare_for_spod(modal_dir, verbose=verbose)
    results = spod_all_modes(u_hat, t_stamps, dt, verbose=verbose)
    m_arr, f_arr, spectrum = spod_spectrum(results)

    # 螺旋度
    omega_hat = compute_vorticity_spectral(u_hat, grid["r_grid"], grid["x_grid"])
    h, H = modal_helicity(u_hat, omega_hat, grid["r_grid"], grid["x_grid"],
                          grid["theta_grid"])
    E_plus, E_minus = helical_decomposition(u_hat, grid["r_grid"],
                                            grid["x_grid"], grid["theta_grid"])

    # 能量传递
    modes_calc = list(range(min(len(m_arr), 8)))
    T, _ = compute_transfer_matrix(u_hat, grid["r_grid"], grid["x_grid"],
                                    grid["theta_grid"], modes=modes_calc)

    os.makedirs(out_dir, exist_ok=True)

    # 可视化
    plot_spod_spectrum(m_arr, f_arr, spectrum,
                       os.path.join(out_dir, f"spod_spectrum_{label}.png"),
                       title=f"SPOD Spectrum - {label}")
    plot_helicity_spectrum(m_arr, f_arr, H, E_plus, E_minus,
                           os.path.join(out_dir, f"helicity_{label}.png"),
                           title=f"Helicity - {label}")
    plot_transfer_matrix(T, modes_calc,
                         os.path.join(out_dir, f"transfer_{label}.png"),
                         title=f"Energy Transfer - {label}")

    return {
        "u_hat": u_hat, "spectrum": spectrum, "m_arr": m_arr, "f_arr": f_arr,
        "H": H, "E_plus": E_plus, "E_minus": E_minus, "T": T,
        "grid": grid, "t_stamps": t_stamps, "dt": dt,
    }


def compare_three_cases(caseA_dir, caseB_dir, caseC_dir, out_dir):
    """三算例对比。"""
    results = {}
    for label, cdir in [("caseA", caseA_dir), ("caseB", caseB_dir), ("caseC", caseC_dir)]:
        if cdir and os.path.isdir(cdir):
            results[label] = analyze_single_case(cdir, label, out_dir)

    if len(results) < 2:
        print(f"\n[compare] 只有 {len(results)} 个算例，跳过对比")
        return results

    # 轴向输运对比
    grid = list(results.values())[0]["grid"]
    cases_u = {k: v["u_hat"] for k, v in results.items()}
    transport = compare_axial_transport(cases_u, grid["r_grid"],
                                        grid["x_grid"], grid["theta_grid"])
    plot_axial_transport(transport, grid["x_grid"],
                         os.path.join(out_dir, "axial_transport_compare.png"))

    # SPOD 谱对比图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, (label, res) in zip(axes, results.items()):
        spec = np.log10(res["spectrum"][:8, :] + 1e-30)
        im = ax.pcolormesh(res["f_arr"], res["m_arr"][:8], spec,
                           shading="auto", cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("m")
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spod_spectrum_compare.png"), dpi=150)
    plt.close(fig)
    print(f"\n[compare] 对比图 -> {out_dir}/spod_spectrum_compare.png")

    return results


def main():
    parser = argparse.ArgumentParser(description="三算例模态分解对比")
    parser.add_argument("--caseA", default=None, help="caseA MODAL 目录")
    parser.add_argument("--caseB", default=None, help="caseB MODAL 目录")
    parser.add_argument("--caseC", default=None, help="caseC MODAL 目录")
    parser.add_argument("--out", default="MODAL_ANALYSIS", help="输出目录")
    args = parser.parse_args()

    compare_three_cases(args.caseA, args.caseB, args.caseC, args.out)


if __name__ == "__main__":
    main()
