# FVM-Forge

<p align="center"><b>OpenCFD 框架下的 GPU 异构并行可压缩流体求解器</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Julia-9558B2?style=flat-square&logo=julia" />
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20ROCm-76B900?style=flat-square" />
  <img src="https://img.shields.io/badge/Parallel-MPI%20%2B%20GPU-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Precision-FP32%20%7C%20FP64-orange?style=flat-square" />
</p>

**FVM-Forge** 是 OpenCFD 框架下基于 MPI+GPU 异构并行的三维可压缩 Navier-Stokes 方程求解器。采用结构化多块有限体积法（FVM），支持 NVIDIA CUDA 和 AMD ROCm 双 GPU 后端，以及纯 CPU 多线程回退模式。

> 📖 完整手册请参阅 [docs/manual.md](docs/manual.md)

---

## ✨ 特色

- 🔥 **GPU 异构并行** — 同一套 Julia kernel 代码，通过统一抽象层自动适配 CUDA / ROCm / CPU
- 🧱 **多块结构化网格** — 支持 O-H butterfly 管道拓扑、任意块间连接、运行时 ghost 坐标生成
- 🔬 **高阶精度** — WENO7/5-Z 特征分解重构 + 6 阶中心差分粘性通量
- ⚡ **自适应混合策略** — 激波传感器驱动的 7 阶光滑/WENO/Minmod 自动切换
- 🚀 **性能优化** — Kernel 自动调优、非阻塞 MPI 管线、批量 pack/unpack、融合 kernel
- 🌀 **丰富物理模型** — Coriolis/离心力、管道流 bulk forcing、HIT 线性强迫
- 🧲 **MHD 求解器** — GLM 散度清洗、Rusanov/HLLD/KEP 通量
- 🔄 **差分旋转模块** — 空间发展管道 + Fringe 回复区、两阶段启动流程

---

## 📁 项目结构

```
FVM-Forge/
├── run_pipe.jl              # 入口：标准旋转管道 (周期性)
├── run_pipe_diffrot.jl      # 入口：差分旋转管道 (Fringe 区)
├── run_pipe_ac.jl           # 入口：不可压 AC 管道
├── run_pipe_piso.jl         # 入口：不可压 PISO 管道
├── run_brio_wu.jl           # 入口：MHD Brio-Wu 激波管
├── solver.jl                # 核心求解器：时间推进、块管理、同步
├── gpu_backend.jl           # GPU 后端抽象层 (CUDA / ROCm / CPU)
├── auto_tune.jl             # GPU kernel 自动调优 (block size + VGPR)
├── auto_partition.jl        # 多块自动 GPU 分区
├── Reconstruct.jl           # WENO7/5 + 线性混合重构 (i/j/k 方向)
├── Riemann_Solver.jl        # HLLC / Van Leer / SW / Roe / KEP 通量 + MHD
├── viscous.jl               # 6 阶中心差分粘性通量
├── boundary.jl              # 边界条件 (壁面/周期/NSCBC/超音速)
├── mpi.jl                   # MPI ghost exchange + 多块间 ghost 同步
├── ghost_coords.jl          # 运行时 ghost cell 坐标扩展
├── volume_force.jl          # 体积力 (旋转/Bulk/HIT/Deschamps forcing)
├── fringe.jl                # Fringe 回复区 (差分旋转用)
├── implicit.jl              # 隐式 LU-SGS 时间推进 (可选)
├── gmres.jl                 # GMRES 线性求解器
├── div.jl                   # 通量散度 + RK 组合
├── filter_interface.jl      # 块间界面滤波 (自适应 σ)
├── spectral_warmup.jl       # 频谱预热 (DRP/CD6 自适应)
├── init_flow.jl             # 初始场 (PipeFlow/TGV/HIT/Sod/MHD)
├── IO.jl                    # HDF5/XDMF 并行 I/O (mesh_dir 自适应)
├── utils.jl                 # c2Prim, compute_dt, 空间滤波器
├── schemes.jl               # 数值格式系数 (DRP/标准)
├── docs/
│   └── manual.md            # 用户手册
├── Benchmark/
│   └── PIPEFLOW/            # 圆管湍流基准测试
└── Utils/
    ├── gen_butterfly_fvm.jl          # O-H 管道网格生成器
    ├── gen_butterfly_fvm_diffrot.jl  # 差分旋转扩展网格生成器
    ├── prepare_precursor_mean.jl     # Phase 1 时均截面提取
    └── analyze_*.jl                  # 后处理诊断脚本集
```

---

## 🚀 快速开始

### 环境要求

| 依赖 | 版本 |
|------|------|
| Julia | ≥ 1.9 |
| MPI.jl | ≥ 0.20 |
| HDF5.jl | ≥ 0.16 |
| CUDA.jl (NVIDIA) | ≥ 5.0 |
| AMDGPU.jl (AMD) | ≥ 0.8 |

### 1. 生成网格

```bash
cd Utils/
julia gen_butterfly_fvm.jl    # 生成 5-block O-H butterfly 管道网格
```

### 2. 运行仿真

```bash
# 24 GPU: 5 blocks, 自动分区
mpirun -np 24 julia run_pipe.jl 10000
```

### 3. 可视化

使用 ParaView 打开 `PLT/` 目录下的 `.xdmf` 文件。

---

## ⚙️ 关键参数

在 `run_pipe.jl` / `run_pipe_diffrot.jl` 中配置：

| 参数 | 类型 | 说明 |
|------|------|------|
| `Re_target` | FT | 目标 Reynolds 数 |
| `Ma_target` | FT | 目标 Mach 数 |
| `Ro_target` / `Ro_min`~`Ro_max` | FT | 旋转数 (均匀/差分) |
| `CFL` | FT | CFL 数 |
| `viscous` | Bool | 启用粘性项 |
| `eigen_reconstruction` | Bool | 特征分解重构 |
| `splitMethod` | String | `"HLLC"` / `"VL"` / `"SW"` / `"Roe"` / `"KEP"` |
| `mesh_dir` | String | 网格目录 (`"MESH_COARSE"` 等) |
| `diffrot_phase` | Int | 差分旋转阶段 (1=湍流发展, 2=激活旋转) |
| `profiling` | Bool | 逐模块性能计时 |

---

## 📊 性能基准

**硬件**: 24× AMD MI50 (gfx906), Hygon DCU 集群  
**网格**: 5-block butterfly 管道, 29.86M cells  
**分区**: (5,1,1)×4 + (4,1,1)×1 = 24 GPUs

| 指标 | 数值 |
|------|------|
| 每步耗时 | 0.087 s |
| GPU 带宽利用率 | 40% (memory-bound) |
| 总 FLOPs/step | 2.36 × 10¹¹ |
| 实际算力 | 0.84 TFLOPS (FP32) |

---

## 📝 引用

如果 FVM-Forge 对您的研究有帮助，请引用：

```
FVM-Forge: A GPU-accelerated heterogeneous parallel compressible FVM solver
under the OpenCFD framework.
```

## 📄 许可

Private research code — OpenCFD Lab.
