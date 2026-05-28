# FVM-Forge 用户手册

<p align="center"><b>OpenCFD 框架 · GPU 异构并行可压缩流体求解器</b></p>

---

## 目录

1. [概述](#1-概述)
2. [理论基础](#2-理论基础)
3. [程序架构](#3-程序架构)
4. [GPU 异构并行设计](#4-gpu-异构并行设计)
5. [网格系统](#5-网格系统)
6. [数值方法](#6-数值方法)
7. [边界条件](#7-边界条件)
8. [时间推进](#8-时间推进)
9. [MPI 通信与同步](#9-mpi-通信与同步)
10. [性能优化](#10-性能优化)
11. [输入输出](#11-输入输出)
12. [运行指南](#12-运行指南)
13. [算例配置](#13-算例配置)
14. [性能分析](#14-性能分析)
15. [常见问题](#15-常见问题)

---

## 1. 概述

### 1.1 简介

**FVM-Forge** 是 OpenCFD 框架下的一款基于 MPI+GPU 异构并行的三维可压缩 Navier-Stokes 方程求解器。核心特点：

- 采用**有限体积法 (FVM)** 在结构化多块网格上离散控制方程
- 使用 **Julia** 语言编写，同一套 kernel 代码支持 **NVIDIA CUDA**、**AMD ROCm** 和 **CPU 多线程** 三种后端
- 通过 **MPI** 实现多 GPU 分布式并行，支持跨节点扩展
- 提供 **WENO7/5-Z** 高阶重构与自适应激波捕捉

### 1.2 适用场景

| 场景 | 说明 |
|------|------|
| 管道湍流 DNS/LES | O-H butterfly 多块网格 + 周期/壁面边界 |
| 平板边界层转捩 | 多块拼接 + NSCBC 入出流边界 |
| 旋转管道流 | Coriolis + 离心力体积力 |
| 可压缩自由剪切流 | WENO 激波捕捉 + 高阶粘性 |

### 1.3 控制方程

FVM-Forge 求解三维可压缩 Navier-Stokes 方程（守恒形式）：

$$
\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}_c(\mathbf{U}) = \nabla \cdot \mathbf{F}_v(\mathbf{U}, \nabla\mathbf{U}) + \mathbf{S}
$$

其中守恒变量、对流通量和原始变量分别为：

$$
\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho w \\ \rho E \end{pmatrix}, \quad
\mathbf{Q} = \begin{pmatrix} \rho \\ u \\ v \\ w \\ p \end{pmatrix}
$$

状态方程：$p = \rho R T$，$E = c_v T + \frac{1}{2}(u^2+v^2+w^2)$

---

## 2. 理论基础

### 2.1 有限体积法离散

对控制方程在控制体 $\Omega_i$ 上积分：

$$
\frac{d\mathbf{U}_i}{dt} = -\frac{1}{V_i} \sum_{f=1}^{6} \left( \mathbf{F}_c \cdot \hat{n} - \mathbf{F}_v \cdot \hat{n} \right)_f \, A_f + \mathbf{S}_i
$$

其中 $V_i$ 为控制体体积，$A_f$ 和 $\hat{n}_f$ 分别为第 $f$ 个面的面积和单位外法向量。

### 2.2 对流通量重构

#### 线性重构 (光滑区)

使用 7 点 stencil 的 UP7/CD6 混合格式：

$$
U^L_{i+1/2} = \sum_{k=1}^{7} L_k \, U_{i-4+k}
$$

权重 $L_k$ 由 `Linear_ϕ` 参数控制 UP7 和 CD6 的混合比例。

#### WENO7-Z 重构 (间断区)

在特征空间中进行 WENO 重构：

1. **Roe 平均**计算特征矩阵 $\mathbf{L}, \mathbf{R}$
2. 投影到特征空间：$V = \mathbf{L} \cdot U$
3. 4 个候选多项式 $q_1, q_2, q_3, q_4$ 在特征空间中重构
4. 光滑指示子 $\mathrm{IS}_r$ + WENO-Z 权重 $\omega_r$
5. 投影回物理空间：$U_{i+1/2} = \mathbf{R} \cdot \sum \omega_r q_r$

#### 自适应混合策略

激波传感器 $\phi$ 控制重构精度：

$$
\phi_i = \frac{|p_{i+1} - 2p_i + p_{i-1}|}{p_{i+1} + 2p_i + p_{i-1}}
$$

| 区域 | 条件 | 方案 | 精度 |
|------|------|------|------|
| 光滑 | $\phi < \phi_1$ | UP7/CD6 线性 | 7 阶 |
| 弱间断 | $\phi_1 < \phi < \phi_2$ | WENO7-Z | 7 阶 |
| 中等间断 | $\phi_2 < \phi < \phi_3$ | WENO5-Z | 5 阶 |
| 强间断 | $\phi > \phi_3$ | Minmod | 2 阶 |

### 2.3 Riemann 求解器

三种通量分裂方案：

| 方案 | 特点 | 适用场景 |
|------|------|----------|
| **HLLC** | 精确捕捉接触间断，低耗散 | 默认，通用 |
| **Van Leer** | 高鲁棒性 | 强激波 |
| **Steger-Warming** | 经典特征分裂 | 参考对比 |

### 2.4 粘性通量

6 阶中心差分计算面梯度：

$$
\left.\frac{\partial \phi}{\partial \xi}\right|_{i+1/2} = \frac{1}{60}\left(-\phi_{i-2} + 8\phi_{i-1} - 8\phi_{i+2} + \phi_{i+3}\right) + \frac{37}{60}\left(\phi_{i+1} - \phi_i\right)
$$

配合 Green-Gauss (GG) 梯度修正消除网格非正交性误差。

黏性系数采用 Sutherland 定律：

$$
\mu(T) = \mu_{\mathrm{ref}} \left(\frac{T}{T_{\mathrm{ref}}}\right)^{3/2} \frac{T_{\mathrm{ref}} + S}{T + S}
$$

### 2.5 空间滤波

8 阶 Pirozzoli 滤波器 (可选)，抑制小尺度数值噪声：

$$
\hat{u}_i = u_i - \alpha_f \sum_{k=-4}^{4} d_k \, u_{i+k}
$$

---

## 3. 程序架构

### 3.1 模块依赖关系

```
run_pipe.jl (入口)
├── gpu_backend.jl    → @gpu_launch, GPUArray, gpu_sync()
├── auto_tune.jl      → KernelConfig, auto_tune_kernel()
├── auto_partition.jl  → auto_gpu_partition()
├── solver.jl         → Block struct, solve()
│   ├── Reconstruct.jl    → Conser_reconstruct_{i,j,k}
│   ├── Riemann_Solver.jl → Blend_Flux, HLLC
│   ├── viscous.jl        → viscous_flux_{i,j,k}
│   ├── div.jl            → linComb_clip_prim
│   ├── boundary.jl       → fillGhost
│   ├── mpi.jl            → exchange_ghost, copy_ghost_face!
│   ├── volume_force.jl   → add_source_kernel!, Deschamps
│   └── implicit.jl       → LU-SGS (可选)
├── ghost_coords.jl   → expand_ghost_coordinates!
├── IO.jl             → write_plt_hdf5, write_checkpoint
└── utils.jl          → c2Prim, compute_dt, linearFilter
```

### 3.2 数据结构

核心数据结构是 `Block`（`solver.jl` 中定义）：

```julia
struct Block
    id::Int                          # 块编号 (0-indexed)
    Nx::Int; Ny::Int; Nz::Int       # 每个 rank 的真实网格点数
    Q::GPUArray{Float32, 4}          # 原始变量 [i, j, k, 5]
    U::GPUArray{Float32, 4}          # 守恒变量 [i, j, k, 5]
    ϕ::GPUArray{Float32, 3}          # 激波传感器
    Areai/j/k::GPUArray{Float32, 3}  # 面积
    nxi/yi/zi::GPUArray{Float32, 3}  # i 方向面法向量分量
    Vol::GPUArray{Float32, 3}        # 控制体体积
    x/y/z::GPUArray{Float32, 3}      # 网格坐标
    # ... MPI 缓冲区, 隐式求解器数组等
end
```

**数组布局**：Julia column-major (Fortran order)，第一维 `i` 在内存中连续。含 `NG=4` 层 ghost cells，索引范围 `[1, Nx+2NG]`。

---

## 4. GPU 异构并行设计

### 4.1 三层并行模型

```
┌──────────────────────────────────────────────────────────┐
│  Level 3: 分布式并行 (MPI)                                │
│  24 ranks × 各持有一个空间子域                              │
│  通信: MPI.Sendrecv / Isend+Irecv (ghost cell 交换)       │
├──────────────────────────────────────────────────────────┤
│  Level 2: CPU-GPU 异构协作                                │
│  CPU: 控制流, MPI, I/O, CFL 判断                          │
│  GPU: kernel 计算 (重构, 粘性, 时间推进)                    │
│  数据迁移: PCIe D2H/H2D (13-14 GB/s)                      │
├──────────────────────────────────────────────────────────┤
│  Level 1: GPU 内部数据并行 (SIMT)                          │
│  每个 kernel 启动 ~100 万线程                               │
│  Wavefront/Warp 粒度并行                                   │
└──────────────────────────────────────────────────────────┘
```

### 4.2 GPU 后端抽象层 (`gpu_backend.jl`)

**设计原则**：所有 GPU kernel 使用统一 API 编写，通过编译期宏分派到不同后端。

```julia
# 统一的 kernel launch 宏
@gpu_launch threads=(8,4,4) blocks=(14,29,29) my_kernel!(args...)

# 自动映射:
#   CUDA  → @cuda threads=(8,4,4) blocks=(14,29,29) my_kernel!(args...)
#   ROCm  → @roc groupsize=(8,4,4) gridsize=(14,29,29) my_kernel!(args...)
#   CPU   → Threads.@threads 分发 blocks，串行执行 threads
```

**线程索引兼容**：

| 函数 | CUDA | ROCm | CPU |
|------|------|------|-----|
| `blockIdx()` | `CUDA.blockIdx()` | `workgroupIdx()` | `task_local_storage` |
| `threadIdx()` | `CUDA.threadIdx()` | `workitemIdx()` | `task_local_storage` |
| `blockDim()` | `CUDA.blockDim()` | `workgroupDim()` | `task_local_storage` |

### 4.3 Kernel 自动调优 (`auto_tune.jl`)

启动时对每个计算 kernel 进行运行时调优：

1. **VGPR 查询**：通过 HIP C API (`hipFuncGetAttribute`) 获取每个 kernel 的寄存器使用量
2. **Occupancy 计算**：基于 VGPR 数计算 CU 最大并发 wavefront 数
3. **Block size 搜索**：遍历 14 种候选 `(tx,ty,tz)` 配置
4. **Microbenchmark**：每种配置 warmup 5 次 + 实测 20 次，选最快
5. **maxregs 搜索** (CUDA)：当 occupancy < 75% 时，尝试限制寄存器以提高 occupancy

---

## 5. 网格系统

### 5.1 O-H Butterfly 管道网格

5 块拓扑，由 `Utils/gen_butterfly_fvm.jl` 生成：

```
      Block 3 (η⁻=interblock, η⁺=wall)
         ┌──────────┐
         │          │
Block 4──│  Block 0 │──Block 2
(wall)   │ (center) │   (wall)
         │          │
         └──────────┘
      Block 1 (η⁻=wall, η⁺=interblock)
```

- **Block 0**：中心方形区域，无壁面
- **Block 1-4**：环形扇区，各自一个 η 面为壁面

### 5.2 Ghost Cell 坐标生成 (`ghost_coords.jl`)

网格文件仅存储真实节点 `(Nx+1, Ny+1, Nz+1)`。运行时根据边界类型自动扩展 NG 层 ghost 节点：

| 边界类型 | Ghost 坐标策略 |
|---------|---------------|
| Periodic (ξ±) | 从对端拷贝 + 周期位移 $x_{\text{ghost}} = x_{\text{opp}} \pm L_x$ |
| Interblock (η/ζ) | 从邻 block 的 mesh 文件读取内部节点 |
| Wall | 镜像外推 $x_{\text{ghost}} = 2x_{\text{bnd}} - x_{\text{int}}$ |

扩展完成后计算**度量张量** (面积、法向量、体积) 并缓存至 `metrics_cache_b*_r*.h5`。

### 5.3 自动 GPU 分区 (`auto_partition.jl`)

贪心算法将 $N_{\text{blocks}}$ 个计算块分配到 $N_{\text{GPUs}}$ 个 GPU：

```julia
# 自动分区结果示例：
# Block 0: 512×108×108, (5,1,1) partition → 5 ranks
# Block 4: 512×108×108, (4,1,1) partition → 4 ranks
# Total: 24 ranks, max mem 0.46 GB/rank
```

分区方向优先选择最长轴（通常是 ξ 方向），确保每个 rank 的显存占用不超过 VRAM 的 60%。

---

## 6. 数值方法

### 6.1 重构格式 (`Reconstruct.jl`)

每个方向有独立的 kernel (`Conser_reconstruct_i/j/k`)，流程：

```
输入: U[i,j,k,1:5] (守恒变量)
  ↓
激波传感器 ϕ = max(ϕ[i-2..i+3])
  ↓
┌─ ϕ < ϕ₁ → 线性混合 UP7/CD6 (7点 stencil)
├─ ϕ₁<ϕ<ϕ₂ → WENO7-Z (4子模板, 特征空间)
├─ ϕ₂<ϕ<ϕ₃ → WENO5-Z (3子模板)
└─ ϕ > ϕ₃ → Minmod 限制器
  ↓
UL, UR 界面重构值
  ↓
HLLC/VL/SW Riemann 求解 → 数值通量 F
  ↓
输出: Fx[i,j,k,1:5] (面通量)
```

### 6.2 激波传感器

```julia
function shockSensor(Q, ϕ, nxp, nyp, nzp)
    # Jameson-type pressure sensor
    ϕ_x = |p[i+1] - 2p[i] + p[i-1]| / (p[i+1] + 2p[i] + p[i-1])
    ϕ_y = |p[i,j+1] - 2p[i,j] + p[i,j-1]| / ...
    ϕ_z = |p[i,j,k+1] - 2p[i,j,k] + p[i,j,k-1]| / ...
    ϕ[i,j,k] = ϕ_x + ϕ_y + ϕ_z
end
```

### 6.3 通量散度 (`div.jl`)

融合 kernel `linComb_clip_prim`：

$$
U^{n+1} = U^n + \alpha_{\text{RK}} \cdot \Delta t \cdot \text{RHS}
$$

同时执行守恒→原始变量转换和正定性裁剪 ($\rho > \rho_{\min}$, $p > p_{\min}$)。

---

## 7. 边界条件

### 7.1 支持的边界类型 (`bc_types.jl`)

| 类型 | 标识 | 说明 |
|------|------|------|
| `BC_PERIODIC` | 0 | 周期边界 |
| `BC_WALL_ISOTHERMAL` | 1 | 等温壁面 (no-slip) |
| `BC_WALL_ADIABATIC` | 2 | 绝热壁面 |
| `BC_SYMMETRY` | 3 | 对称面 |
| `BC_INTERBLOCK` | 10 | 多块连接 |
| `BC_NSCBC_INFLOW` | 20 | NSCBC 亚声速入流 |
| `BC_NSCBC_OUTFLOW` | 21 | NSCBC 亚声速出流 |

### 7.2 NSCBC 边界 (Poinsot & Lele, 1992)

基于 LODI (Locally One-Dimensional Inviscid) 特征关系的非反射边界条件。

---

## 8. 时间推进

### 8.1 显式 SSP-RK3

```
Stage 1: U⁽¹⁾ = Uⁿ + Δt · L(Uⁿ)
Stage 2: U⁽²⁾ = ¾Uⁿ + ¼[U⁽¹⁾ + Δt · L(U⁽¹⁾)]
Stage 3: Uⁿ⁺¹ = ⅓Uⁿ + ⅔[U⁽²⁾ + Δt · L(U⁽²⁾)]
```

### 8.2 隐式 LU-SGS (可选)

双时间步 BDF2 + LU-SGS 伪时间迭代，适合大 CFL 稳态计算。

### 8.3 自适应时间步

基于 CFL 条件：

$$
\Delta t = \text{CFL} \cdot \min_i \frac{V_i}{\lambda_c + \lambda_v}
$$

全局 `MPI.Allreduce(MPI.MIN)` 同步。

---

## 9. MPI 通信与同步

### 9.1 每个 RK Stage 的同步流程

```
sync_blocks!()
├── Step 1: copy_ghost_face!(real_range)   # 块间 ghost 拷贝 (仅 real)
├── Step 2: fillGhost()                    # 物理边界填充
├── Step 3: exchange_ghost(MPI)            # rank 间 MPI 交换
├── Step 4: copy_ghost_face!(full_range)   # 块间 ghost 拷贝 (含 edge)
├── Step 5: interface_filter()             # 块间界面滤波 (可选)
└── Step 6: c2Prim_ghost()                 # ghost 区 U→Q
```

### 9.2 非阻塞 MPI 管线 (Phase B)

x+/x- 方向使用 `MPI.Isend/Irecv` 并行：

```
pack_R + pack_L → 单次 gpu_sync → D2H(x+) + D2H(x-)
→ Isend(x+) + Irecv(x+) + Isend(x-) + Irecv(x-)
→ [compute_fn 计算重叠] → Waitall
→ H2D + unpack_L + unpack_R
```

### 9.3 批量 Pack/Unpack (Phase C)

块间 ghost 交换使用 per-slot GPU buffer：

```
所有 slot: pack_kernel → 单次 gpu_sync → 批量 D2H
→ MPI Isend/Irecv → Waitall
→ 批量 H2D → 所有 unpack_kernel
```

---

## 10. 性能优化

### 10.1 已实施的优化

| 优化 | 技术 | 效果 |
|------|------|------|
| Block size 自适应 | 运行时 benchmark 14 种配置 | 100% occupancy |
| VGPR 查询 | HIP API → occupancy 计算 | 主 kernel 64 VGPR |
| Wavefront 对齐 | `total % 64 == 0` 评分加权 | 消除 partial wavefront |
| `@inbounds` | 所有 kernel 数组访问 | 消除 bounds check |
| FP32 一致性 | 全 `0.0f0` 常量 | 避免 FP64 性能惩罚 |
| Kernel 融合 | div+RK_clip, deschamps+source | 减少 kernel 启动开销 |
| 非阻塞 MPI | Isend/Irecv + Waitall | x+/x- 通信重叠 |
| 批量 Pack | per-slot GPU buffer + 单次 sync | 减少 N-1 次 gpu_sync |
| 度量缓存 | metrics_cache HDF5 | 加速后续启动 |

### 10.2 内存访问模式

Julia column-major 存储，`i` 方向连续：
- `Conser_reconstruct_i`: threadIdx.x → i → **完美 coalesced**
- `Conser_reconstruct_j/k`: threadIdx.x → i，stencil 在 j/k 方向有 stride

### 10.3 性能特征

| 指标 | 数值 | 说明 |
|------|------|------|
| 算术强度 | ~1.45 FLOP/Byte | Memory-bound |
| HBM 带宽利用率 | ~40% | 接近合理上限 |
| GPU 计算占比 | ~28% | 剩余为通信开销 |
| 通信占比 | ~72% | PCIe + MPI 延迟主导 |

---

## 11. 输入输出

### 11.1 网格格式

HDF5 文件，每个块一个文件：
```
MESH/mesh_b0.h5   → datasets: x[Nx+1, Ny+1, Nz+1], y[...], z[...]
MESH/mesh_b1.h5
...
MESH/connectivity.h5  → 块间连接表
```

### 11.2 输出格式

**流场快照** (`PLT/`)：
```
PLT/plt_step_1000_b0_r0.h5   + plt_step_1000.xdmf
```
ParaView 打开 `.xdmf` 文件即可可视化。

**Checkpoint** (`CHK/`)：
```
CHK/chk_step_1000_b0_r0.h5   → U[1:5], Q[1:5]
```
支持断点续算。

### 11.3 性能输出

启用 `profiling = true` 后，每 100 步打印：

```
┌─────────────────────────────────────────────────────
│ sync_blocks              18.998 s  ( 72.9%)
│     ghost_face(real)      3.938 s  ( 15.1%)
│     exchange_ghost(MPI)   3.596 s  ( 13.8%)
│     ghost_face(full)     10.621 s  ( 40.8%)
│ blockAdvance              5.074 s  ( 19.5%)
│ TOTAL                    26.046 s  (per step: 0.0868 s)
└─────────────────────────────────────────────────────
```

---

## 12. 运行指南

### 12.1 环境准备

```bash
# Julia 包安装
julia -e 'using Pkg; Pkg.add(["MPI", "HDF5", "AMDGPU", "Printf"])'

# 验证 GPU
julia -e 'using AMDGPU; println(AMDGPU.device())'
```

### 12.2 生成网格

```bash
cd Utils/
julia gen_butterfly_fvm.jl
# 输出: MESH/mesh_b0.h5 ... mesh_b4.h5 + connectivity.h5
```

### 12.3 启动仿真

```bash
# 单节点 4 GPU
mpirun -np 4 julia run_pipe.jl 10000

# 多节点 24 GPU (SLURM)
srun --nodes=4 --ntasks-per-node=6 --gpus-per-node=6 \
     julia run_pipe.jl 100000
```

### 12.4 断点续算

```bash
# 从 step 10000 的 checkpoint 续算到 step 20000
mpirun -np 24 julia run_pipe.jl 20000
# 程序自动检测 CHK/ 目录中最新的 checkpoint
```

---

## 13. 算例配置

### 13.1 管道湍流 (`run_pipe.jl`)

```julia
const Re_target = 17000.0f0      # 目标 Reynolds 数
const Ma_target = 0.3f0          # 目标 Mach 数
const Ro_target = 0.0f0          # 旋转数 (0=无旋转)
const Tw = 307.0f0               # 壁面温度 [K]
const Lx = 7.5f0                 # 管道长度 (以 R 为单位)
const LTS_CFL = 0.5f0            # CFL 数
const viscous = true             # 启用粘性
const splitMethod = "HLLC"       # Riemann 求解器
const eigen_reconstruction = false  # 守恒变量重构
const forcing_mode = 3           # Deschamps bulk forcing
```

### 13.2 平板边界层 (`Benchmark/BL_TRANSITION/run_config.jl`)

```julia
const Ma_inf = 0.5f0             # 来流 Mach 数
const Re_delta = 1000.0f0        # 基于位移厚度的 Re
const T_inf = 300.0f0            # 来流温度
const implicit_enabled = true    # 启用 LU-SGS 隐式
const implicit_CFL = 5.0f0       # 隐式 CFL
```

---

## 14. 性能分析

### 14.1 FLOPS 分解

| Kernel | FLOPs/cell/step | 占比 |
|--------|----------------:|-----:|
| Conser_reconstruct (×3dir ×3stage) | 3,015 | 38% |
| viscous_flux (×3dir ×3stage) | 4,248 | 54% |
| 其他 (div, sensor, filter, dt) | 652 | 8% |
| **合计** | **7,915** | 100% |

### 14.2 Roofline 分析

```
           ┌────────────────────────────────────────────
           │                         ╱ Peak FP32 (13 TFLOPS)
  GFLOPS/s │                      ╱
           │                   ╱
           │                ╱
           │     ●       ╱    ← FVM-Forge (369 GFLOPS @ AI=1.45)
           │          ╱       Memory-bound region
           │       ╱
           │    ╱ Peak BW (628 GB/s)
           │ ╱
           └────────────────────────────────────────────
                    Arithmetic Intensity (FLOP/Byte)
```

FVM stencil 计算的算术强度约 1.45 FLOP/Byte，远低于 Roofline 拐点 (~20 FLOP/Byte)，属于典型的 **memory-bound** workload。

---

## 15. 常见问题

### Q: 如何切换 CUDA 和 ROCm？

修改入口文件的 `using` 语句：
```julia
using AMDGPU   # AMD GPU
# 或
using CUDA     # NVIDIA GPU
# 或都不加     → CPU 多线程模式
```

### Q: 出现 NaN 怎么调试？

```julia
const debug_nan = true     # 启用 NaN 检测
const debug_sync = true    # NaN 检测前同步 GPU
```

程序会在每个 RK stage 后检查 `U` 中是否有 NaN，并报告位置。

### Q: 如何减少通信开销？

1. **减少 GPU 数量**：增大每个 GPU 的计算量，提高计算/通信比
2. **启用 GPU-Aware MPI**：`const gpu_aware_mpi = true`（需集群 UCX 支持）

### Q: 显存不足？

调整 `auto_partition.jl` 中的 `VRAM_LIMIT` 参数，或减少每个 GPU 分配的网格点数。

---

*FVM-Forge — OpenCFD Lab*
