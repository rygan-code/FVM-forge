# 可压/超声速 ECS (Exact Coherent Structures) 求解器

此目录存放基于 Matrix-Free Newton-Krylov-Hookstep 算法求解可压/超声速圆管流中相对周期轨道（RPOs）与行波解（TWs）的代码。

## 目录结构

- `newton_krylov_solver.py`：外层 Newton 迭代与 Krylov 子空间（GMRES）求解器主程序。
- `group_actions.py`：基于 FFT 的无损柱坐标系流向平移与基于 2D 三角剖分的周向旋转群算子。
