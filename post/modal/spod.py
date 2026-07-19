"""
spod.py - Spectral Proper Orthogonal Decomposition (SPOD)

Towne-Schmidt-Colonius batch 算法实现（自包含，约 150 行）。

算法：
    1. 单周向模态 m 的快照矩阵 Q_m ∈ C^(Nsr x Nt)
    2. 减时间平均 q̄_m
    3. Welch 分段：N_blk 段，每段 N_fft 点，50% overlap，加窗
    4. 每段时间 FFT -> 段谱 q̂_blk(f)
    5. 交叉谱密度 C(f) = (1/N_blk) Σ q̂_blk(f) q̂_blk(f)^*
       （快照法避免显式构造 Nsr×Nsr）
    6. 每个 f 对 C(f) 特征分解 -> SPOD 模态 φ_m(r,x;f) + 能量 λ_m(f)

参考: Towne, Schmidt, Colonius, "Spectral proper orthogonal decomposition
      and its relationship to dynamic mode decomposition and resolvent analysis",
      JFM 847, 821-867 (2018).
"""

import numpy as np
from scipy.signal import get_window


def _welch_segments(Nt, n_fft, overlap=0.5):
    """
    计算 Welch 分段的起始索引。

    返回: list of (start, end) 索引对
    """
    if n_fft > Nt:
        n_fft = Nt
    hop = int(n_fft * (1 - overlap))
    n_blk = 1 + (Nt - n_fft) // hop
    segments = []
    for i in range(n_blk):
        start = i * hop
        end = start + n_fft
        segments.append((start, end))
    return segments


def spod_single_mode(q, dt, n_fft=None, overlap=0.5, window="hamming",
                     verbose=False):
    """
    对单个周向模态 m 的快照序列做 SPOD。

    参数:
        q: ndarray [Nsr, Nt] 复数快照矩阵
           Nsr = nrad * nx * ncomp（空间自由度）
        dt: 采样间隔（物理时间）
        n_fft: 每段 FFT 点数（None -> Nt//4 或最近 2 的幂）
        overlap: 重叠率（默认 0.5）
        window: 窗函数（默认 hamming）

    返回:
        result: dict
            "freqs": [Nf] 频率轴
            "modes": [Nf, Nsr, n_blk] SPOD 模态 φ(f, j)，按能量降序
                     （j=0 最优）
            "energy": [Nf, n_blk] 能量 λ(f, j)
            "n_fft", "n_blk": 分段参数
    """
    Nsr, Nt = q.shape

    # 减时间平均
    q_mean = np.mean(q, axis=1, keepdims=True)
    q_fluc = q - q_mean

    # 确定 n_fft
    if n_fft is None:
        # 默认取 Nt 的 1/4，向下取到最近的 2 的幂
        n_fft = int(2 ** np.floor(np.log2(max(Nt // 4, 8))))
        n_fft = min(n_fft, Nt)
    n_fft = min(n_fft, Nt)

    segments = _welch_segments(Nt, n_fft, overlap)
    n_blk = len(segments)

    if n_blk < 1:
        # 单段
        segments = [(0, Nt)]
        n_blk = 1
        n_fft = Nt

    if verbose:
        print(f"[SPOD] Nsr={Nsr}, Nt={Nt}, n_fft={n_fft}, "
              f"overlap={overlap}, n_blk={n_blk}")

    # 窗函数
    win = get_window(window, n_fft, fftbins=True)
    win_norm = np.sqrt(np.sum(win**2))  # 功率归一化

    # 频率轴（复数输入，用 fftfreq，需 fftshift 排列为 [-fmax..0..+fmax]）
    freqs_raw = np.fft.fftfreq(n_fft, d=dt)
    shift_idx = np.argsort(freqs_raw)
    freqs = freqs_raw[shift_idx]
    Nf = len(freqs)

    # 段谱：每段做时间 FFT（复数输入，用 fft）
    # q_hat_blk[f, Nsr, n_blk]
    q_hat_blk = np.zeros((Nf, Nsr, n_blk), dtype=np.complex128)
    for j, (s, e) in enumerate(segments):
        seg = q_fluc[:, s:e]  # [Nsr, n_fft]
        # 加窗 + FFT（沿时间轴，复数输入）
        seg_win = seg * win[np.newaxis, :]
        qhat = np.fft.fft(seg_win, axis=1)  # [Nsr, Nf] (fftfreq 排列)
        q_hat_blk[:, :, j] = qhat.T[shift_idx, :]  # 重排到升序频率

    # 交叉谱密度矩阵 C(f) = (1/N_blk) Σ q̂_blk q̂_blk^*
    # 对每个 f，C(f) 是 [Nsr, Nsr]，但用快照法：
    #   C(f) = (1/N_blk) Q_hat(f) Q_hat(f)^H,  Q_hat(f)=[Nsr, n_blk]
    # 特征分解 C(f) φ = λ φ 等价于 Q_hat Q_hat^H 的左奇异向量
    # 用 SVD: Q_hat(f) = U Σ V^H, 则 U 是 SPOD 模态, Σ²/N_blk 是能量

    modes = np.zeros((Nf, Nsr, n_blk), dtype=np.complex128)
    energy = np.zeros((Nf, n_blk))

    for fi in range(Nf):
        Qf = q_hat_blk[fi, :, :]  # [Nsr, n_blk]
        # SVD: Qf = U S Vh
        U, S, Vh = np.linalg.svd(Qf, full_matrices=False)
        # 能量 λ = S² / (N_blk * win_norm²)  (Welch 归一化)
        energy[fi, :] = (S**2) / (n_blk * win_norm**2)
        modes[fi, :, :] = U

    return {
        "freqs": freqs,
        "modes": modes,
        "energy": energy,
        "n_fft": n_fft,
        "n_blk": n_blk,
        "q_mean": q_mean.flatten(),
    }


def spod_all_modes(u_hat, t_stamps, dt, overlap=0.5, n_fft=None,
                   verbose=False):
    """
    对所有周向模态 m 做 SPOD。

    参数:
        u_hat: [Nt, nmodes, nrad, nx, ncomp] 复数 Fourier 系数
        t_stamps: [Nt] 均匀物理时间
        dt: 采样间隔
        overlap, n_fft: Welch 参数

    返回:
        results: dict {m: spod_single_mode 结果}
                 m 从 0 到 nmodes-1
    """
    Nt, nmodes, nrad, nx, ncomp = u_hat.shape
    Nsr = nrad * nx * ncomp  # 空间自由度

    results = {}
    for m in range(nmodes):
        if verbose:
            print(f"[SPOD] 处理周向模态 m={m}...")
        # 构造快照矩阵 [Nsr, Nt]
        q = u_hat[:, m, :, :, :].reshape(Nt, Nsr).T  # [Nsr, Nt]

        # 跳过全零模态（如 r<axis_cut 的 m>=1）
        if np.allclose(q, 0):
            results[m] = None
            continue

        results[m] = spod_single_mode(q, dt, n_fft=n_fft, overlap=overlap,
                                      verbose=(verbose and m == 0))
    return results


def spod_spectrum(results):
    """
    提取 SPOD 能量谱 λ(m, f)。

    返回:
        m_arr: [M] 周向模态数
        f_arr: [Nf] 频率
        spectrum: [M, Nf] 第一模态（j=0）能量
    """
    valid = {m: r for m, r in results.items() if r is not None}
    m_arr = np.array(sorted(valid.keys()))
    f_arr = valid[m_arr[0]]["freqs"]
    Nf = len(f_arr)
    spectrum = np.zeros((len(m_arr), Nf))
    for i, m in enumerate(m_arr):
        spectrum[i, :] = valid[m]["energy"][:, 0]  # j=0 最优模态
    return m_arr, f_arr, spectrum
