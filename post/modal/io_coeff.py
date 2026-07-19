"""
io_coeff.py - 读取在线钩子落盘的 Fourier 系数 + 时间戳

落盘格式（modal_hook.jl 产出）：
    MODAL/coeff-{step}.h5
      attrs: time, step, dt
      datasets: r_grid, theta_grid, x_grid
                u_hat[m, nrad, nx], ur_hat[m,nrad,nx], uth_hat[m,nrad,nx]

功能：
    - load_coefficients: 加载全部快照，返回 (u_hat[Nt,m,Nr,Nx,3], t_stamps)
    - diagnose_sampling: 检查物理时间间隔波动（G4 门控）
    - resample_uniform: 不等间隔时重采样到均匀 dt
"""

import os
import re
import glob
import numpy as np
import h5py
from scipy.interpolate import interp1d


def find_coeff_files(modal_dir):
    """发现 MODAL/ 下所有 coeff-*.h5，返回按 step 排序的路径列表。"""
    pattern = os.path.join(modal_dir, "coeff-*.h5")
    files = glob.glob(pattern)
    # 排除 test 文件
    files = [f for f in files if "test" not in os.path.basename(f)]
    def extract_step(f):
        m = re.search(r"coeff-(\d+)\.h5$", os.path.basename(f))
        return int(m.group(1)) if m else -1
    files.sort(key=extract_step)
    return files


def load_coefficients(modal_dir, components=("u", "ur", "uth"), verbose=True):
    """
    加载全部快照的 Fourier 系数。

    返回:
        u_hat: ndarray [Nt, nmodes, nrad, nx, 3] 复数
               第 5 维按 components 顺序 (u_x, u_r, u_theta)
        t_stamps: ndarray [Nt] 物理时间
        grid: dict {r_grid, theta_grid, x_grid}
    """
    files = find_coeff_files(modal_dir)
    if not files:
        raise FileNotFoundError(f"No coeff-*.h5 files in {modal_dir}")

    if verbose:
        print(f"[io_coeff] 发现 {len(files)} 个快照: "
              f"step {extract_step(files[0])}..{extract_step(files[-1])}")

    # 先读第一个文件获取网格尺寸
    with h5py.File(files[0], "r") as f:
        r_grid = f["r_grid"][:]
        theta_grid = f["theta_grid"][:]
        x_grid = f["x_grid"][:]
        nmodes, nrad, nx = f["u_hat"].shape
        comp_names = [c + "_hat" for c in components]

    Nt = len(files)
    u_hat = np.zeros((Nt, nmodes, nrad, nx, len(components)), dtype=np.complex128)
    t_stamps = np.zeros(Nt)
    steps = np.zeros(Nt, dtype=int)

    for i, fpath in enumerate(files):
        with h5py.File(fpath, "r") as f:
            t_stamps[i] = f.attrs["time"]
            steps[i] = f.attrs["step"]
            for j, cname in enumerate(comp_names):
                u_hat[i, :, :, :, j] = f[cname][:]

    grid = {"r_grid": r_grid, "theta_grid": theta_grid, "x_grid": x_grid}

    if verbose:
        print(f"[io_coeff] u_hat shape={u_hat.shape}, "
              f"t∈[{t_stamps[0]:.4e}, {t_stamps[-1]:.4e}], "
              f"Δt={np.diff(t_stamps)}")

    return u_hat, t_stamps, grid, steps


def extract_step(fpath):
    """从文件名提取 step 整数。"""
    m = re.search(r"coeff-(\d+)\.h5$", os.path.basename(fpath))
    return int(m.group(1)) if m else -1


def diagnose_sampling(t_stamps, tolerance=0.05, verbose=True):
    """
    G4 门控：检查物理时间间隔的波动幅度。

    参数:
        t_stamps: 物理时间数组
        tolerance: 允许的相对波动 (max-min)/mean，默认 5%

    返回:
        is_uniform: bool, 是否足够均匀
        dt_mean: 平均间隔
        dt_var: 相对波动幅度
        dt_array: 间隔数组
    """
    dt_array = np.diff(t_stamps)
    dt_mean = np.mean(dt_array)
    dt_var = (np.max(dt_array) - np.min(dt_array)) / dt_mean if dt_mean > 0 else np.inf

    is_uniform = dt_var < tolerance

    if verbose:
        print(f"[G4 采样诊断]")
        print(f"  快照数: {len(t_stamps)}")
        print(f"  Δt 均值: {dt_mean:.6e}")
        print(f"  Δt 范围: [{np.min(dt_array):.6e}, {np.max(dt_array):.6e}]")
        print(f"  相对波动: {dt_var*100:.2f}% (容差 {tolerance*100:.0f}%)")
        print(f"  判定: {'均匀' if is_uniform else '需重采样'}")

    return is_uniform, dt_mean, dt_var, dt_array


def resample_uniform(u_hat, t_stamps, dt_target=None):
    """
    不等间隔时，在物理时间上重采样到均匀 dt。

    对每个 (m, ir, ix, comp) 系数序列做 1D 插值。

    参数:
        u_hat: [Nt, nmodes, nrad, nx, 3] 复数
        t_stamps: [Nt] 物理时间
        dt_target: 目标 dt，None 则用平均 dt

    返回:
        u_hat_uniform: [Nt_new, nmodes, nrad, nx, 3]
        t_uniform: [Nt_new] 均匀时间
    """
    if dt_target is None:
        dt_target = np.mean(np.diff(t_stamps))

    t_uniform = np.arange(t_stamps[0], t_stamps[-1] + dt_target * 0.5, dt_target)
    Nt_new = len(t_uniform)

    Nt, nmodes, nrad, nx, ncomp = u_hat.shape
    u_hat_uniform = np.zeros((Nt_new, nmodes, nrad, nx, ncomp), dtype=np.complex128)

    # 逐点插值（实部、虚部分开）
    for m in range(nmodes):
        for ir in range(nrad):
            for ix in range(nx):
                for c in range(ncomp):
                    series = u_hat[:, m, ir, ix, c]
                    interp_re = interp1d(t_stamps, series.real, kind="linear",
                                         fill_value="extrapolate")
                    interp_im = interp1d(t_stamps, series.imag, kind="linear",
                                         fill_value="extrapolate")
                    u_hat_uniform[:, m, ir, ix, c] = (
                        interp_re(t_uniform) + 1j * interp_im(t_uniform)
                    )

    return u_hat_uniform, t_uniform


def prepare_for_spod(modal_dir, force_resample=False, tolerance=0.05, verbose=True):
    """
    一站式准备 SPOD 输入：加载 + 采样诊断 + 按需重采样。

    返回:
        u_hat: [Nt, nmodes, nrad, nx, 3] 复数，时间均匀
        t_stamps: [Nt] 均匀物理时间
        dt: 采样间隔
        grid: dict
        info: dict {is_uniform, dt_var, n_orig, n_resampled}
    """
    u_hat, t_stamps, grid, steps = load_coefficients(modal_dir, verbose=verbose)

    is_uniform, dt_mean, dt_var, _ = diagnose_sampling(
        t_stamps, tolerance=tolerance, verbose=verbose
    )

    if not is_uniform or force_resample:
        if verbose:
            print(f"[prepare] 执行重采样到均匀 dt={dt_mean:.6e}")
        u_hat, t_stamps = resample_uniform(u_hat, t_stamps, dt_target=dt_mean)
        info = {"is_uniform": False, "dt_var": dt_var,
                "n_orig": len(steps), "n_resampled": len(t_stamps),
                "resampled": True}
    else:
        info = {"is_uniform": True, "dt_var": dt_var,
                "n_orig": len(steps), "n_resampled": len(t_stamps),
                "resampled": False}

    return u_hat, t_stamps, dt_mean, grid, info
