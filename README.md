# Simflow

Simflow 是基于 Julia 的三维可压缩有限体积求解器，面向 CPU、CUDA/ROCm GPU
和 MPI 异构计算。当前仓库采用模块化 `src/` 架构，提供结构化多块求解路径，
并保留正在完善的非结构网格后端。

完整的理论、算法和使用说明见 [docs/manual.md](docs/manual.md)。

## 主要能力

- 结构化多块有限体积求解，支持可压缩 Euler、Navier–Stokes 和 MHD；
- GLM 与 Constrained Transport（CT）磁场推进路径；
- WENO/特征重构、Riemann 通量、粘性通量、正性保护和多种时间推进方法；
- MPI 多块通信、GPU 后端以及 CPU 回退；
- HDF5/XDMF/VTU 等网格、结果和重启动 I/O；
- `Benchmark/` 中的管道流、MHD、CT、激波管和网格验证算例。

非结构后端的实现位于 `src/*/unstructured_*.jl`，其能力和限制以手册及
`config/cases/unstructured_*.toml` 为准。

## 目录结构

```text
src/
├── core/       配置、方程和边界类型
├── mesh/       结构化与 OpenFOAM 网格
├── numerics/   重构、通量、散度和 CT
├── parallel/   MPI、GPU、分区和通信
├── physics/    Euler/MHD、边界和粘性物理
├── time/       RK、源项和后端生命周期
└── io/         结构化/非结构化 I/O
bin/
├── run_case.jl              TOML 算例入口
└── generate_openfoam_mesh.jl OpenFOAM 网格工具
config/cases/                 算例配置
Benchmark/                    基准、收敛和验证算例
Utils/                        网格与结果处理工具
post/                         后处理模块
docs/manual.md                用户手册
```

## 环境与安装

建议使用 Julia 1.10。首次运行前安装项目依赖：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

根据硬件选择 MPI、CUDA 或 ROCm 环境；CPU 回退路径不需要 GPU。

## 运行算例

统一入口为 `bin/run_case.jl`，第一个参数是 TOML 配置文件：

```bash
# 仅检查配置，不启动长时间计算
julia --project=. bin/run_case.jl config/cases/unstructured_smoke.toml --validate-only

# 运行一个配置算例
julia --project=. bin/run_case.jl config/cases/unstructured_smoke.toml
```

结构化生产入口和具体网格要求见 `docs/manual.md` 及 `Benchmark/` 中对应算例的
说明。MPI 运行时使用与 Julia/MPI.jl 匹配的 `mpiexec`/`mpirun`。

结果通常写入算例配置指定的目录，可能包含 HDF5、XDMF 或 VTU 文件，可使用
ParaView 等工具进行可视化。

## 配置文件

配置文件采用 TOML 格式，主要分为：

- `[backend]`：网格后端、设备和运行模式；
- `[mesh]`：网格类型、尺寸、边界和输入路径；
- `[physics]`：方程、粘性、MHD/CT 和物性参数；
- `[numerics]`：阶数、Riemann 通量、CFL 和正性策略；
- `[time]`、`[output]`：终止时间、步数、输出间隔和结果目录。

## 文档与基准

- 用户手册：[docs/manual.md](docs/manual.md)
- 基准和验证算例：[Benchmark/](Benchmark/)
- 结果处理工具：[Utils/](Utils/)

## 许可证

本项目采用 [MIT License](LICENSE)。
