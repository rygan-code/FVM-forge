# =============================================================================
#  modal_config.jl - 在线模态分解采样参数
#
#  由 run 脚本 include 后提供以下 Main 作用域常量：
#    modal_sample_step  : 每隔多少时间步采样一次（受 step_plt 影响的独立采样率）
#    modal_ntheta       : 重采样目标周向分辨率（2 的幂，FFT 友好）
#    modal_nrad         : 重采样目标径向分辨率
#    modal_nmodes_keep  : 保留的周向 Fourier 模态数（m=0..M_max）
#    modal_out_dir      : 落盘目录
#
#  用 @isdefined 守护，允许 run 脚本在 include 本文件前覆盖默认值。
# =============================================================================

# ─── 采样步长 ───
# 每 modal_sample_step 个时间步触发一次在线周向分解。
# 生产 dt~2e-7，若 modal_sample_step=50 -> Δt_sample~1e-5 s。
# 采样总时长由生产运行步数决定；频率分辨率 Δf = 1/(N_blk·N_fft·Δt_sample)。
@isdefined(modal_sample_step) || const modal_sample_step::Int = 50

# ─── 周向目标网格 ───
# 蝴蝶网格壁面环 4×108=432 点（0.8333°/点）。
# 重采样到 modal_ntheta 点均匀环（2 的幂，FFTW 高效）。
# 128 足以分辨 m≤64，实际只看前 ~16 个模态，128 留足余量。
@isdefined(modal_ntheta) || const modal_ntheta::Int = 128

# ─── 径向目标网格 ───
# 原网格径向 N_rad=108（壁面加密）。重采样到 modal_nrad 点均匀 r 分布。
# r∈[0, R0=0.5]，含块0 核心（r<0.236）和块1-4 环（0.236~0.5）。
@isdefined(modal_nrad) || const modal_nrad::Int = 96

# ─── 保留模态数 ───
# FFT 后保留 m=0..M_max。modal_ntheta=128 -> Nyquist m=64，
# 但湍流相干结构主要在前 ~16 个，保留 32 兼顾余量与存储。
@isdefined(modal_nmodes_keep) || const modal_nmodes_keep::Int = 32

# ─── 轴向分辨率 ───
# 轴向不重采样（原网格 x 均匀 dx=Lx/Nx），直接用全部 Nx 点。
# 生产 Nx=1024/块，5 块共址（非拼接），取单块即可。
# 此参数仅用于尺寸校验，不实际降采样。
@isdefined(modal_nx_keep) || const modal_nx_keep::Int = 0   # 0 = 全部保留

# ─── 落盘 ───
@isdefined(modal_out_dir) || const modal_out_dir::String = "MODAL"

# ─── 块0 核心处理策略 ───
# :interp  - 在 r=r_inner=0.236 界面处用块0 数据插值补全内部环
#            （分解覆盖到轴心，m≥1 在 r=0 强制 0 处理奇异性）
# :exclude - 块0 核心区排除，m=0 取块0 体积平均，m≥1 轴心 0
# 计划选择 :interp（界面插值补全）。
@isdefined(modal_core_strategy) || const modal_core_strategy::Symbol = :interp

# ─── 物理常量（与 run 脚本一致，供 cyl_regrid 用）───
@isdefined(R0) || const R0::Float64 = 0.5
@isdefined(r_inner) || const r_inner::Float64 = 0.4 * R0   # = 0.2，块0 外接圆名义半径

# r=0 奇异性阈值：r < r_axis_cut 时，m≥1 模态强制 0
@isdefined(r_axis_cut) || const r_axis_cut::Float64 = 0.5 * r_inner   # = 0.1
