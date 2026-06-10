# 非均匀旋转湍流的色散关系

在旋转可压缩流体（如旋转圆管流）中，科里奥利力、离心力、均流剪切以及可压缩性（声波）的相互耦合，使系统支持一种复合波动——**离心-惯性-声波（Centrifugo-Inertial-Acoustic Waves）**。当背景旋转率（自转角速度）在空间中分布不均匀时，波的色散关系和传播特性会发生根本性的改变。

以下根据历史讨论记录，分别推导两种典型非均匀旋转工况下的色散关系。

---

## 1. 径向非均匀旋转（差分旋转圆管）的色散关系

### 1.1 物理模型与基本假设
*   **坐标系**：采用圆柱坐标系 $(x, r, \theta)$，其中 $x$ 为管道轴向（流向），$r$ 为半径，$\theta$ 为周向角。整个系统处于一个以角速度 $\mathbf{\Omega}_0 = \Omega_0 \mathbf{\hat{x}}$ 旋转的参考系中。
*   **均流背景场**：
    *   平均速度场：$\mathbf{U}_0 = U_0(r)\mathbf{\hat{x}} + V_0(r)\mathbf{\hat{\theta}}$。其中 $U_0(r)$ 为圆管流速剖面，$V_0(r) = r \Omega(r)$ 为流体相对于旋转框架的差分旋转（周向剪切）速度。
    *   当地绝对旋转角速度：$\Omega_{abs}(r) = \Omega_0 + \Omega(r)$。
    *   径向静力平衡：背景场的径向压力梯度与高速旋转产生的离心力相平衡：
        $$
        \frac{1}{\rho_0} \frac{\partial P_0}{\partial r} = r \Omega_{abs}^2(r)
        $$
    *   背景密度为 $\rho_0(r)$，局部声速为 $c_s$。

### 1.2 WKB 近似与线性扰动方程
引入短波近似（WKB 近似，即假定径向波数满足 $k_r r \gg 1$，且 $k_r \gg k_x, m/r$）。扰动量形式为：
$$
(u', v'_r, v'_\theta, p', \rho') \propto \exp[i(k_x x + k_r r + m\theta - \omega t)]
$$
定义**当地多普勒频移频率**（流体元相对频率）：
$$
\bar{\omega} = \omega - k_x U_0(r) - m \Omega(r)
$$
在线性化小扰动假设下，可压缩控制方程简化为：

1.  **连续性方程**（结合状态方程 $\rho' = p'/c_s^2$）：
    $$
    -i\bar{\omega} \frac{p'}{c_s^2} + \rho_0 \left( i k_x u' + i k_r v'_r + i \frac{m}{r} v'_\theta \right) + v'_r \frac{\partial \rho_0}{\partial r} = 0 \tag{1}
    $$
2.  **轴向 ($x$) 动量方程**：
    $$
    -i\bar{\omega}u' + v'_r \frac{\partial U_0}{\partial r} = - \frac{i k_x p'}{\rho_0} \tag{2}
    $$
3.  **径向 ($r$) 动量方程**：
    $$
    -i\bar{\omega}v'_r - 2\Omega_{abs} v'_\theta = - \frac{i k_r p'}{\rho_0} + \frac{p'}{\rho_0 c_s^2} r \Omega_{abs}^2 \tag{3}
    $$
4.  **周向 ($\theta$) 动量方程**：
    $$
    -i\bar{\omega}v'_\theta + \left( 2\Omega_{abs} + r\frac{\partial \Omega_{abs}}{\partial r} \right) v'_r = - \frac{i m p'}{\rho_0 r} \tag{4}
    $$

### 1.3 偏振关系与色散关系
由方程 (2)、(3)、(4) 可以解出扰动速度与压力扰动 $p'$ 的偏振关系：
$$
u' = \frac{k_x p'}{\bar{\omega} \rho_0} - i \frac{v'_r}{\bar{\omega}} \frac{\partial U_0}{\partial r} \tag{5}
$$
$$
v'_\theta = i \frac{2\Omega_{abs} + r \partial_r \Omega_{abs}}{\bar{\omega}} v'_r + \frac{m p'}{\bar{\omega} \rho_0 r} \tag{6}
$$
将 $v'_\theta$ 代入径向动量方程 (3)，可求得径向速度：
$$
v'_r = \frac{i p'}{\rho_0 (\bar{\omega}^2 - \kappa_C^2)} \left[ \bar{\omega} k_r - \frac{\bar{\omega} r \Omega_{abs}^2}{c_s^2} + \frac{2 m \Omega_{abs}}{r} \right] \tag{7}
$$
其中定义了**本轮频率（Epicyclic Frequency）** $\kappa_C$：
$$
\kappa_C^2 = 2\Omega_{abs} \left( 2\Omega_{abs} + r\frac{\partial \Omega_{abs}}{\partial r} \right) = \frac{2\Omega_{abs}}{r} \frac{\partial(r^2 \Omega_{abs})}{\partial r}
$$
将这些偏振关系代入连续性方程 (1)，并在短波极限下展开，最终消去 $p'$ 得到**离心-惯性-声波的统一色散关系**：
$$
\bar{\omega}^4 - \bar{\omega}^2 \left[ c_s^2 k^2 + N_c^2 + \kappa_C^2 \right] + c_s^2 k_\perp^2 (N_c^2 + \kappa_C^2) = 0
$$

#### 变量物理定义：
*   $k^2 = k_x^2 + k_r^2 + m^2/r^2$：总波数的平方。
*   $k_\perp^2 = k_x^2 + m^2/r^2$：垂直于径向的平面波数的平方。
*   $N_c^2 = -\frac{1}{\rho_0}\frac{\partial \rho_0}{\partial r} (\Omega_{abs}^2 r)$：**离心浮力频率的平方**。它是由于可压缩流体受强离心力压实，在径向建立起密度分层（类似重力分层）所产生的等效 Brunt-Väisälä 频率。

### 1.4 物理极限讨论
1.  **不可压缩极限 ($c_s \to \infty$)**：
    当声波速度趋向无穷大时，色散关系退化为：
    $$
    \bar{\omega}^2 = (N_c^2 + \kappa_C^2) \frac{k_\perp^2}{k^2}
    $$
    这描述了纯粹的**离心-惯性波（Centrifugo-Inertial Waves）**。
    *   若无密度分层（$N_c = 0$）且为轴对称扰动（$m=0$），公式退化为 $\bar{\omega}^2 = \kappa_C^2 \frac{k_x^2}{k^2}$，这与经典旋转流体中的惯性波色散关系 $\omega^2 = 4\Omega_0^2 \cos^2\phi$ 在物理上完全一致。
2.  **声波极限**：当波动尺度极小且频率极高时，公式退化为普通声波：$\bar{\omega}^2 \approx c_s^2 k^2$。
3.  **局部稳定性判据（离心与浮力耦合）**：
    为使波动频率为实数（不发生不稳定的指数增长，即 $\bar{\omega}^2 > 0$），必须满足：
    $$
    N_c^2 + \kappa_C^2 > 0
    $$
    这构成了可压缩差分旋转流体在离心力场作用下的局部稳定性判据（即扩展的 Solberg-Høiland 判据）。若该值小于零，则会自发激发离心不稳定性，产生 Taylor-Görtler 涡等不稳定流结构。

---

## 2. 流向非均匀自转参考系下的色散关系

### 2.1 物理模型与基本假设
*   **坐标系**：采用直角坐标系 $(x, y, z)$，其中 $x$ 为管道轴向（流向）。
*   **非均匀旋转**：背景角速度沿着轴向 $x$ 发生变化，即 $\mathbf{\Omega} = \Omega_x(x) \mathbf{\hat{x}}$。这对应我们求解器中模拟空间发展管流时的背景物理配置。
*   **WKB 假定**：由于背景自转沿流向是不均匀的，波的流向波数 $k_x$ 是空间坐标 $x$ 的函数。设波动相位为 $\int k_x(x) dx$，扰动量形式为：
    $$
    (u', v', w', p', \rho') \propto \exp \left[ i \left( \int k_x(x) dx + k_y y + k_z z - \omega t \right) \right]
    $$
    定义当地多普勒频移频率：$\bar{\omega}(x) = \omega - k_x(x) U_0$，其中 $U_0$ 为常数流速。

### 2.2 控制方程与速度偏振
求解器在旋转工况下施加的局部作用力包括科氏力与离心力。在当地线性化 WKB 近似下，控制方程为：

1.  **连续性方程**：
    $$
    \frac{\bar{\omega} p'}{\rho_0 c_s^2} = k_x(x) u' + k_y v' + k_z w' \tag{8}
    $$
2.  **$x$ 方向动量方程**：
    $$
    u' = \frac{k_x(x) p'}{\bar{\omega} \rho_0} \tag{9}
    $$
3.  **$y$ 方向动量方程**：
    $$
    -i\bar{\omega} v' - 2\Omega_x(x) w' = - \frac{i k_y p'}{\rho_0} \tag{10}
    $$
4.  **$z$ 方向动量方程**：
    $$
    -i\bar{\omega} w' + 2\Omega_x(x) v' = - \frac{i k_z p'}{\rho_0} \tag{11}
    $$

联立方程 (10) 和 (11)，解出横向波动速度与压力的偏振关系：
$$
v' = \frac{\bar{\omega} p'}{\rho_0 (\bar{\omega}^2 - 4\Omega_x^2(x))} \left( k_y + i \frac{2\Omega_x(x) k_z}{\bar{\omega}} \right) \tag{12}
$$
$$
w' = \frac{\bar{\omega} p'}{\rho_0 (\bar{\omega}^2 - 4\Omega_x^2(x))} \left( k_z - i \frac{2\Omega_x(x) k_y}{\bar{\omega}} \right) \tag{13}
$$

### 2.3 流向渐变旋转下的色散关系
将偏振关系 (9)、(12)、(13) 代入连续性方程 (8)，消去压力项并化简，可得**流向非均匀旋转参考系下的可压缩色散关系**：
$$
\bar{\omega}^4 - \bar{\omega}^2 \left[ c_s^2 k^2(x) + 4\Omega_x^2(x) \right] + 4\Omega_x^2(x) c_s^2 k_x^2(x) = 0
$$
其中：
*   $k_\perp^2 = k_y^2 + k_z^2$：截面横向总波数（常数）。
*   $k^2(x) = k_x^2(x) + k_\perp^2$：随轴向位置 $x$ 变化的当地总波数的平方。

---

### 2.4 物理现象分析

#### 1. 波前空间压扁与倾角演化 (Wavefront Tilting)
在不可压缩极限 ($c_s \to \infty$) 下，上述色散关系退化为：
$$
\bar{\omega}^2 = 4\Omega_x^2(x) \frac{k_x^2(x)}{k^2(x)} = 4\Omega_x^2(x) \cos^2\phi(x)
$$
其中 $\cos\phi(x) = k_x(x)/k(x)$ 是当地波矢量与旋转轴（$x$ 轴）夹角的余弦。
由于在稳态传播中波的频率 $\bar{\omega}$ 是守恒常数，当波动向下游传播时，背景旋转角速度 $\Omega_x(x)$ 逐渐增强，这意味着夹角余弦值 $\cos\phi(x)$ 必须逐渐减小。
*   **物理表现**：波矢逐渐偏向横截面（径向和周向），使得**波动在空间上被逐渐拉伸、波面沿流向被“压扁”**。

#### 2. 流向“转向面”与截止波障 (Turning Surface / Wave Barrier)
因为 $|\cos\phi(x)| \le 1$，所以对于给定的波频 $\bar{\omega}$，波动能够在轴向传播的最大位置 $x_{turn}$ 必须满足：
$$
2\Omega_x(x_{turn}) = \bar{\omega}
$$
当波动试图穿过该截面到达更下游的强旋转区（$x > x_{turn}$）时，为了维持色散方程平衡，流向波数 $k_x(x)$ 必须演变为**共轭复数**：
$$
k_x(x) = \beta(x) + i \alpha(x) \quad (\alpha > 0)
$$
*   **物理结论**：此时波动在越过 $x_{turn}$ 后不再传播，而是以 $\exp(-\int \alpha(x) dx)$ 的形式**指数衰减**。这表明流向非均匀旋转会在圆管内部形成一个**空间截止波障（Wave Barrier）**。从弱旋转区（上游）激发的惯性波在向下游强旋转区传播时，会在转向面 $x_{turn}$ 被全部反射或局域化耗散。这一机制为控制旋转圆管内的波动局域化和能量聚焦提供了直接的理论依据。
