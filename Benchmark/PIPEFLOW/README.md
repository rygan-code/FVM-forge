# PipeFlow — Multi-block Compressible Pipe Turbulence

## 物理问题

**圆管可压缩湍流** (Pipe Flow)

- 圆截面管道内流
- 目标 Ma = 0.8
- 目标 Re = 17000
- 驱动方式：动量合成体积力 (flow_forcing = true, proportional forcing)
- 展向（轴向）周期边界
- 壁面等温无滑移，Tw = 307.0 K

## 网格拓扑

**5 块蝴蝶网格 (Butterfly Grid)**

- 块 0：中心方形块
- 块 1~4：四周扇形块
- 消除极轴奇异性（避免极寒处网格极度密集导致的时间步长崩溃），非常适合在 GPU 上做高精度可压缩湍流尺度解算。 

## 使用方法

由于网格体积较大（包含网格度量参数，HDF5），通常你需要预先在 `MESH/` 下准备好相对应的网格文件：
- `mesh_b0.h5` ~ `mesh_b4.h5`
- `metrics_b0.h5` ~ `metrics_b4.h5`
- `block_connectivity.h5`
- `interp_weights.h5`

### 1. 快速载入并运行

```bash
# 自动部署配置到根目录并运行
julia benchmark.jl setup PIPEFLOW

# 用 MPI 跑你需要的步数（比如 1000）
mpirun -np 4 julia run.jl 1000
```

> **注意：** 可以直接从根目录运行原生的 `mpirun -np x julia run_pipe.jl`，两者结果是一致的。

### 2. 验证运行状态

```bash
julia benchmark.jl verify PIPEFLOW
```

这会自动扫描所有的 5 个块对应的 PLT 输出，检查是否有：
1. 出现 `NaN` 
2. 非物理的热力学极值（负密度、负压力等）
