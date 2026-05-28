# CHANGELOG

本文件记录 FVM-Forge 项目的版本变更历史。

---

## [2026-05-08] — Star-Stencil 31 点 / 17 基函数升级 & 过冲抑制

### 升级：23→31 点 + 14→17 基函数

- **`spectral_warmup.jl`** — `_compute_star_face_weights` 重写
  - **23→31 点**：新增 8 个对角点 (ξ±1, ζ±1) @ j0 和 @ j0+1
  - **14→17 基函数**：新增 ξ²η, ζ²η, ξζη 三项——捕获横向曲率沿主方向的变化
  - 超定比 31/17 ≈ 1.82，保持系统稳定性
  - 新增权重质量检查：`sum(w) ≈ 1`（tol=1e-6）+ `max|w| < 4.0`
  - 权重计算失败时自动回退 1D 路径（`star_active = 0`）

- **`Reconstruct.jl`** — 4 处 GPU kernel (Eigen/Conser × j/k) 更新
  - 新增 8 条对角邻居值读取（indices 24-31）
  - **Venkatakrishnan 光滑限制器（Scheme D）**：
    - `f_limited = f_cell + φ · (f_star - f_cell)`
    - φ 由 Venkatakrishnan (1995) 公式计算，在间断处 φ→0 抑制过冲，在光滑极值处 φ→1 保留精度
    - ε² = (K · (v_max - v_min))² + 1e-30，K=0.01
    - K 敏感性测试：K=0 等于硬截断，K=0.01 时 step 过冲残留 ~0.3%（可接受），K=0.1 时残留 ~1%（过大）

### 验证

- 合成弯折网格（0°-45° bend）全部 PASS
  - sum(w) = 1.0, max|w| ≈ 0.27, 常数/线性场精确重构
- solver.jl GPU 数组已为 31 维（无需修改）

---

## [2026-04-29] — 差分旋转模块 & IO 修复

### 新增

- **`run_pipe_diffrot.jl`** — 差分旋转管道入口脚本
  - 两阶段启动流程：Phase 1（湍流发展, Ω=0）→ Phase 2（差分旋转 + Fringe 回复）
  - Ro 范围 0.0→1.0 线性梯度配置
  - `diffrot_phase` 开关控制阶段切换
  - `mesh_dir` 变量指定网格目录（`MESH_DIFFROT_COARSE`）

- **`fringe.jl`** — Fringe 回复区模块
  - `fringe_forcing_kernel!`: 在体积力管线中施加 λ(x)(U_target - U) 回复力
  - 平滑余弦过渡函数，避免 Fringe 区边界不连续
  - 与 `Volume_force_kernel` 管线整合，复用 GPU 并行

- **`Utils/gen_butterfly_fvm_diffrot.jl`** — 差分旋转扩展网格生成器
  - 20R₀ 管道（15R₀ 物理区 + 5R₀ Fringe 区）
  - 基于 `gen_butterfly_fvm.jl`，独立运行不影响原网格

- **`Utils/prepare_precursor_mean.jl`** — Phase 1 时均截面提取脚本
  - 从 PLT 输出中提取 block-wise 保守变量时均剖面
  - 输出 `precursor_mean.h5` 供 Phase 2 的 Fringe U_target 使用

### 修复

- **`IO.jl`** — 消除 4 处硬编码 `"MESH/"` 路径
  - `plotFile_multiblock` (line 54): fallback 路径
  - `checkpointFile` (line 107): 网格尺寸读取
  - `write_avg` (line 168): 平均场输出
  - `write_XDMF_multiblock` (line 201): XDMF 引用
  - 改为 `@isdefined(mesh_dir) ? mesh_dir : "MESH"` 自适应路径
  - 向后兼容：`run_pipe.jl` 等无 `mesh_dir` 的入口自动回退到 `"MESH"`

- **`init_flow.jl`** — `test_case` 分发匹配
  - `run_pipe_diffrot.jl` 的 `test_case` 从 `"PipeFlow_DiffRot"` 改回 `"PipeFlow"`
  - 原因：`initialize()` 只识别 `"PipeFlow"`，不匹配导致 Q 数组全零 → Step 3 NaN

- **`Riemann_Solver.jl`** — UTF-8 编码修复
  - MHD 部分注释头中 Unicode box-drawing 字符（`═╔╗`等）在跨平台传输中被截断
  - 替换为纯 ASCII `=` 和 `-`，消除 ParseError

- **`solver.jl`** — 注释中损坏的 Unicode 替换
  - 用户手动修复了多处 `>=?` → `→` 和 `══...>=?` → `═══...═══` 的编码残留

### 变更

- **`solver.jl`** — Fringe 区集成
  - Block 结构扩展：新增 `fringe_lambda`, `fringe_U_target` 字段
  - Phase 2 时自动加载 `precursor_mean.h5` 并映射到 GPU
  - `fringe_forcing_kernel!` 在 volume force 之后调用

- **`gen_butterfly_fvm.jl`** — 防护性修改
  - 添加 `FT` 类型别名定义
  - `main()` 用 `@isdefined` 守卫，防止被 include 时自动执行

---

## [2026-04-28] — 代码清理 & 论文准备

### 变更

- 移除冗余的一次性 benchmark 脚本（`scratch_*.jl` 等）
- 保留 `Benchmark/PIPEFLOW/` 目录结构完整性

### 新增

- 中国气动学会 2026 年会论文摘要（中英双语版本）
- 小波诊断管线可视化标准化（`pub_legend!`/`pub_figure!`）

---

## [2026-04-27] — 界面滤波自适应 & 棋盘格根因定位

### 根因分析

- **棋盘格伪影根因确认**：多 block FVM + 低耗散 WENO-Z 方案的固有特性
  - 跨类型界面（η↔ζ）由不同 GPU kernel 独立计算共享面通量
  - j-kernel 与 k-kernel 的 FMA 融合顺序不同 → O(ε_machine) ≈ 1e-15 通量不匹配
  - WENO-Z 的 2Δx 零耗散区间无法抑制，经数万步非线性累积变为可见伪影
  - **结论：非代码 bug，需物理必要的界面滤波**

### 修复

- **`filter_interface.jl`** — 完全重写
  - 8 阶显式滤波器 + **自适应 σ**：`σ_local = σ_max × (1 - lin_phi)`
    - `lin_phi ≈ 0`（纯中心）→ 需要更强滤波
    - `lin_phi ≈ 0.5`（界面过渡区）→ 中等滤波
    - `lin_phi ≈ 1`（纯迎风）→ 不需要滤波
  - 替换早期的 2 阶正系数滤波器 `[0.125, 0.75, 0.125]`
  - 废弃原 8 阶负系数滤波器（可产生负密度/能量 → blowup）

### 排除的候选原因（完整验证记录）

| 组件 | 验证结果 |
|------|---------|
| Ghost U 值 MPI 拷贝 | 精确拷贝 ✅ |
| Ghost Q 值 c2Prim | 一致 ✅ |
| 激波传感器 ϕ 跨界面同步 | 一致 ✅ |
| Metric 面积/法向量 | 一致 ✅ |
| WENO stencil 系数 | 一致 ✅ |
| mode=1/mode=2 竞争条件 | 无冲突 ✅ |
| 通量数组覆盖 | 无重叠 ✅ |

---

## [2026-04-26] — 双精度支持

### 变更

- 全代码库 Float32 → FT 参数化，支持 Float64 运行
- 修复 WENO 系数、物理常数中的 `NNNzero(FT)` / `NNNone(FT)` 字面量错误
- `precision.jl` 集中管理浮点类型

---

## [2026-04-25] — Butterfly 界面结构断裂根因诊断

### 根因分析

- **根因 #1：中心块-环形块网格尺度比 1.78:1**
  - Block 0 (center): Δ_cell = 4.37 mm（均匀正方形）
  - Block 1 (annular): Δ_inner = 2.46 mm（靠近中心块侧）
  - WENO 重构横跨界面时，网格尺度跳变导致 smoothness indicator 触发降阶
  - 低阶重构 ≈ 一阶迎风 → 界面处数值耗散大幅增加 → 湍流条带断裂

- **根因 #2：跨类型界面浮点误差（不可修）**
  - 多 block FVM 设计中，共享面通量由两侧各自独立计算
  - j-kernel vs k-kernel 浮点指令调度差异 → O(ε_machine) 通量不匹配
  - 需滤波器管控

### 新增

- **`Utils/analyze_interface_mismatch.jl`** — 界面 Reynolds 应力不匹配诊断
- **`Utils/analyze_reynolds_stress.jl`** — 块间应力连续性检查

### 解决方案优先级

| 优先级 | 方案 | 说明 |
|--------|------|------|
| P0 | 减小网格尺度比 <1.2:1 | 根治，需增加 Block 0 分辨率 |
| P1 | 优化滤波器 (σ↓, 层数↓) | 缓解，已实现自适应 σ |
| P2 | WENO 度规自适应重构 | 精确但需改 Reconstruct.jl 核心 |

---

## [2026-04-23] — MHD 求解器

### 新增

- `Riemann_Solver.jl` 扩展：MHD Rusanov / HLLD / KEP 通量
- GLM 散度清洗（Dedner et al. 2002）：ψ-damping source term
- `init_flow.jl`: Brio-Wu 激波管和 Orszag-Tang 涡初始条件
- `run_brio_wu.jl`: MHD 验证入口脚本

---

## [2026-03-31 ~ 04-10] — Block 界面 NaN Blowup 调查

### 问题

- 可压缩旋转管道仿真在 ~1500 步后在 butterfly 块间界面处出现 NaN
- NaN 全部集中在 cross-type 界面（η↔ζ）附近的 ghost cell 区域

### 根因

- **Metric 不连续（GCL 违反）**
  - butterfly 网格在 cross-type 界面处存在 45° 网格弯折
  - 运行时中心差分计算 metric 时，跨界面的 ghost 坐标导致法向量/面积不匹配
  - `(Area·n̂)_Block1 ≠ -(Area·n̂)_Block4` → 通量不守恒 → 寄生源项累积 → NaN

### 修复

- **`filter_interface.jl`** — 初版界面滤波器
  - 2 阶正系数凸组合滤波器 `[0.125, 0.75, 0.125]`，保证非负密度/能量
  - 按界面法方向逐层施加（4 层 ghost 深度）

- **`spectral_warmup.jl`** — 频谱预热模块
  - 基于网格拉伸率自动选择 DRP/CD6 stencil
  - `lin_phi` 线性混合因子：界面处自动增加迎风分量抑制振荡
  - `crosstype_protection`: 跨类型界面额外增强 lin_phi

- **`solver.jl`** — NaN 检测与诊断
  - `@check_nan` 宏：每步检测 Q/U 数组中的 NaN，定位到 block/rank/位置
  - 输出界面邻近度、拉伸率、方向等诊断信息
