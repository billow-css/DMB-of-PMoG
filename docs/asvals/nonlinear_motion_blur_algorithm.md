# 空变非线性运动模糊算法

本文档对 AS_VALs 实验中所用的**取景窗刚体抖动模糊**作严格数学表述。MATLAB 原型见 `demos/matlab/demo_nonlinear_motion_blur.m`，可微实现见 `demos/python/demo_nonlinear_motion_blur.py`。

---

## 1. 问题设定

设原图为标量或向量值函数

$$
I : \Omega \subset \mathbb{R}^2 \to \mathbb{R}^C,\quad
\Omega = [1,W] \times [1,H],
$$

其中 $C$ 为通道数（RGB 时 $C=3$），坐标采用**像素坐标系**（与 MATLAB `interp2` 一致，左上角为 $(1,1)$，$x$ 向右、$y$ 向下）。

在 $\Omega$ 上定义一个边长为 $S$ 的**正方形取景窗**（实验中 $S=512$）。取景窗携带二维刚体位姿

$$
g = (c_x, c_y, \theta) \in \mathbb{R}^2 \times \mathbb{S}^1,
$$

其中 $(c_x,c_y)$ 为窗中心在原图中的坐标，$\theta$ 为绕中心的逆时针旋转角。

曝光过程中，位姿从起点 $g_0$ 连续变化到终点 $g_1$，对窗内每个输出像素沿其**空间变（spatially-variant）轨迹**积分原图强度，得到模糊图像 $O$。

---

## 2. 局部坐标与位姿映射

### 2.1 取景窗局部坐标

输出图像尺寸为 $S\times S$。定义局部坐标 $(u,v)$，原点位于窗中心：

$$
u, v \in \left[-\frac{S-1}{2},\,\frac{S-1}{2}\right].
$$

输出像素网格为

$$
\mathcal{U} = \left\{
  (u_i, v_j) \;\middle|\;
  u_i = -\frac{S-1}{2} + \frac{S-1}{S-1}\,i,\;
  i,j \in \{0,1,\ldots,S-1\}
\right\}.
$$

（实现上等价于 `linspace(-half, half, S)`，其中 $\texttt{half}=(S-1)/2$。）

### 2.2 刚体变换

旋转矩阵

$$
R(\theta) =
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \phantom{-}\cos\theta
\end{pmatrix}.
$$

位姿 $g=(c_x,c_y,\theta)$ 将局部点 $(u,v)^\top$ 映射到原图像素坐标 $(x,y)^\top$：

$$
\begin{pmatrix} x \\ y \end{pmatrix}
= \Phi\bigl((u,v)^\top;\, g\bigr)
:= R(\theta)
\begin{pmatrix} u \\ v \end{pmatrix}
+
\begin{pmatrix} c_x \\ c_y \end{pmatrix}.
$$

分量形式：

$$
\begin{aligned}
x(u,v;\,c_x,c_y,\theta) &= c_x + u\cos\theta - v\sin\theta, \\
y(u,v;\,c_x,c_y,\theta) &= c_y + u\sin\theta + v\cos\theta.
\end{aligned}
$$

记 $\mathbf{p}(u,v;g) = (x,y)^\top$。

---

## 3. 曝光轨迹：刚体位姿插值

设曝光归一化时间 $t \in [0,1]$。起点与终点位姿为

$$
g_0 = (c_{x0},\, c_{y0},\, \theta_0), \qquad
g_1 = (c_{x1},\, c_{y1},\, \theta_1).
$$

实验中起点固定 $\theta_0 = 0$；终点平移在起点邻域内随机，转角在 $[-\theta_{\max},\, \theta_{\max}]$ 内随机。

采用**分量线性插值**（Euclidean / 线性插值 on $SE(2)$ 参数）：

$$
g(t) = \bigl(c_x(t),\, c_y(t),\, \theta(t)\bigr),
$$

$$
\begin{aligned}
c_x(t) &= (1-t)\,c_{x0} + t\,c_{x1}, \\
c_y(t) &= (1-t)\,c_{y0} + t\,c_{y1}, \\
\theta(t) &= (1-t)\,\theta_0 + t\,\theta_1.
\end{aligned}
$$

对固定局部坐标 $(u,v)$，其在原图中的轨迹为

$$
\gamma_{u,v}(t) := \mathbf{p}\bigl(u,v;\, g(t)\bigr)
= \mathbf{p}\bigl(u,v;\, c_x(t), c_y(t), \theta(t)\bigr),\quad t\in[0,1].
$$

### 3.1 空间变性与「非线性」

- **空间变性（spatially-variant）**：不同 $(u,v)$ 对应不同轨迹 $\gamma_{u,v}$。当 $\theta_0 \neq \theta_1$ 时，$\gamma_{u,v}$ 一般**不是**直线，而是平面上的曲线。
- **端点弦线**：定义端点

$$
\mathbf{p}_0(u,v) := \mathbf{p}(u,v; g_0),\qquad
\mathbf{p}_1(u,v) := \mathbf{p}(u,v; g_1).
$$

连接 $\mathbf{p}_0$ 与 $\mathbf{p}_1$ 的直线段是轨迹的一阶弦近似；完整轨迹 $\gamma_{u,v}(t)$ 由旋转项引起弯曲，故称为**非线性运动模糊**（区别于整图共享同一 PSF 的线性平移模糊）。

---

## 4. 连续曝光模型

将曝光视为对时间 $t$ 的加权积分。对通道 $c \in \{1,\ldots,C\}$，输出定义为

$$
O_c(u,v)
= \int_0^1 w(t)\,
  \mathcal{S}\!\left[
    I_c\bigl(\gamma_{u,v}(t)\bigr)
  \right]
  \,\mathrm{d}t,
$$

其中：

- $I_c(\cdot)$：第 $c$ 通道在原图上的取值；
- $\mathcal{S}[\cdot]$：空间采样算子（见 §5）；
- $w(t)$：非负权重，满足归一化 $\displaystyle\int_0^1 w(t)\,\mathrm{d}t = 1$。

### 4.1 高斯时间权重

采用以曝光中点 $t_c = 1/2$ 为中心的高斯核：

$$
\tilde{w}(t) = \exp\!\left(
  -\frac{(t - t_c)^2}{2\sigma_t^2}
\right),
\qquad
w(t) = \frac{\tilde{w}(t)}{\displaystyle\int_0^1 \tilde{w}(s)\,\mathrm{d}s}.
$$

参数 $\sigma_t$（代码中 `gaussSigma`）控制时间权重集中度：

| $\sigma_t$ | 效果 |
|:---:|:---|
| 较小 | 权重集中于 $t\approx 1/2$，中间时刻贡献大 |
| 较大 | 趋近均匀曝光，$\int I(\gamma(t))\,\mathrm{d}t$ |

离散实现中对 $t_k$ 使用相同形式并再归一化：$w_k = \tilde{w}(t_k) / \sum_j \tilde{w}(t_j)$。

---

## 5. 空间采样：双线性插值

原图仅在整数像素格点上有采样值。对非整数坐标 $\mathbf{q} = (x,y)^\top$，定义双线性插值算子 $\mathcal{B}[I_c](\mathbf{q})$：

设 $x = i + \alpha$，$y = j + \beta$，其中 $i,j \in \mathbb{Z}$，$\alpha,\beta \in [0,1)$，则

$$
\mathcal{B}[I_c](x,y) =
\sum_{(\delta_x,\delta_y) \in \{0,1\}^2}
  W_{\delta_x,\delta_y}(\alpha,\beta)\,
  I_c(i+\delta_x,\, j+\delta_y),
$$

$$
W_{\delta_x,\delta_y}(\alpha,\beta) =
\bigl((1-\alpha)^{1-\delta_x}\alpha^{\delta_x}\bigr)
\bigl((1-\beta)^{1-\delta_y}\beta^{\delta_y}\bigr).
$$

连续模型中的 $I_c(\gamma_{u,v}(t))$ 在实现中替换为

$$
I_c\bigl(\gamma_{u,v}(t)\bigr)
\;\approx\;
\mathcal{B}[I_c]\bigl(\gamma_{u,v}(t)\bigr).
$$

MATLAB 使用 `interp2(..., 'linear')`；PyTorch 使用 `grid_sample(..., mode='bilinear', align_corners=True)`，坐标经 §7 所述仿射变换至 $[-1,1]^2$。

---

## 6. 离散数值实现

将 $[0,1]$ 均匀剖分为 $K$ 个节点（代码中 `nSamples`）：

$$
t_k = \frac{k-1}{K-1},\quad k = 1,\ldots,K.
$$

离散权重

$$
w_k = \frac{\exp\bigl(-(t_k - 1/2)^2 / (2\sigma_t^2)\bigr)}
           {\sum_{j=1}^{K} \exp\bigl(-(t_j - 1/2)^2 / (2\sigma_t^2)\bigr)}.
$$

模糊输出（第 $c$ 通道）为

$$
\boxed{
O_c(u,v) = \sum_{k=1}^{K} w_k\,
  \mathcal{B}[I_c]\!\left(
    \mathbf{p}\bigl(u,v;\, g(t_k)\bigr)
  \right)
}
$$

其中

$$
g(t_k) = \bigl(c_x(t_k),\, c_y(t_k),\, \theta(t_k)\bigr).
$$

### 6.1 对照：起点清晰裁切

作为未模糊参照，起点位姿下的裁切图为

$$
O_c^{\mathrm{sharp}}(u,v) =
\mathcal{B}[I_c]\!\left(
  \mathbf{p}(u,v;\, g_0)
\right).
$$

### 6.2 逐像素轨迹长度（可视化量）

定义位移场模长

$$
m(u,v) = \bigl\|\mathbf{p}_1(u,v) - \mathbf{p}_0(u,v)\bigr\|_2
= \sqrt{\bigl(x_1 - x_0\bigr)^2 + \bigl(y_1 - y_0\bigr)^2},
$$

用于分析空间变模糊强度分布（**非模糊算子本身**）。

---

## 7. 可微实现（PyTorch）

设输入张量 $\mathbf{I} \in \mathbb{R}^{B \times C \times H \times W}$，$B$ 为 batch 大小。

对每个 $t_k$，计算采样网格 $\mathbf{G}_k \in \mathbb{R}^{S \times S \times 2}$，其元素为归一化坐标 $(g_x, g_y) \in [-1,1]^2$：

$$
g_x = \frac{2(x-1)}{W-1} - 1,\qquad
g_y = \frac{2(y-1)}{H-1} - 1,
$$

其中 $(x,y)^\top = \mathbf{p}(u,v; g(t_k))$。这与 `align_corners=True` 的 `grid_sample` 一致。

单时刻采样帧：

$$
\mathbf{F}_k = \mathrm{GridSample}(\mathbf{I},\, \mathbf{G}_k).
$$

最终输出

$$
\mathbf{O} = \sum_{k=1}^{K} w_k\, \mathbf{F}_k.
$$

### 7.1 可微性

- 对**输入图像** $\mathbf{I}$：`grid_sample` 对双线性插值权重可微，故 $\partial \mathbf{O}/\partial \mathbf{I}$ 存在。
- 对**位姿参数** $(c_x, c_y, \theta)$：若将其设为 `requires_grad=True` 的张量，梯度经 $\mathbf{p}(\cdot)$ 与 `grid_sample` 的坐标分支回传（$\partial \mathbf{O}/\partial g$ 存在）。
- 离散求和与权重 $w_k$ 为常数，不影响可微性。

因此该算子可作为端到端训练中的**可微噪声层**嵌入网络。

---

## 8. 边界约束与位姿采样

取景窗四角在局部坐标下为

$$
\mathcal{C} = \left\{
  \left(\pm\frac{S-1}{2},\,\pm\frac{S-1}{2}\right)
\right\}.
$$

对位姿 $g$，记四角映到原图后的坐标集合

$$
\Phi(\mathcal{C}; g) = \bigl\{
  \mathbf{p}(u,v; g) \mid (u,v) \in \mathcal{C}
\bigr\}.
$$

**可行性条件**：对所有检查时刻 $t \in \mathcal{T}_{\mathrm{check}}$（含 $t=0,1$ 及插值中间点），

$$
\Phi(\mathcal{C}; g(t)) \subseteq [1,W] \times [1,H].
$$

若不满足，则拒绝该组 $(g_0, g_1)$ 并重新随机采样（拒绝采样，最多尝试 $N_{\mathrm{try}}$ 次）。

### 8.1 起点采样范围

为保证 $\theta_0=0$ 时轴对齐窗在图内，中心坐标均匀采样于

$$
c_{x0} \sim \mathcal{U}\!\left[
  \frac{S-1}{2}+1,\;
  W - \frac{S-1}{2}
\right],\quad
c_{y0} \sim \mathcal{U}\!\left[
  \frac{S-1}{2}+1,\;
  H - \frac{S-1}{2}
\right].
$$

### 8.2 终点扰动

$$
\begin{aligned}
c_{x1} &= c_{x0} + \Delta_x, \quad \Delta_x \sim \mathcal{U}[-T,\, T], \\
c_{y1} &= c_{y0} + \Delta_y, \quad \Delta_y \sim \mathcal{U}[-T,\, T], \\
\theta_1 &\sim \mathcal{U}[-\theta_{\max},\, \theta_{\max}],
\end{aligned}
$$

其中 $T$ 为 `maxTrans`，$\theta_{\max}$ 为 `maxRotDeg` 换算的弧度。

---

## 9. 算法流程（伪代码）

```
输入: 原图 I (W×H×C), 窗大小 S, 采样数 K, 权重参数 σ_t, 位姿扰动范围
输出: 模糊图 O (S×S×C)

1. 重复采样 (g₀, g₁) 直至路径上四角均不越界
2. 构造局部网格 {(uᵢ, vⱼ)}
3. 计算 tₖ ∈ [0,1] 及归一化权重 wₖ
4. O ← 0
5. for k = 1..K:
       g ← g(tₖ)
       for each (uᵢ, vⱼ):
           q ← p(uᵢ, vⱼ; g)
           O(uᵢ,vⱼ) += wₖ · BilinearSample(I, q)
6. return O
```

---

## 10. 默认实验参数

| 符号 / 名称 | 典型值 | 含义 |
|:---|:---:|:---|
| $W, H$ | 600 | AS_VALs 原图尺寸 |
| $S$ | 512 | 取景窗 / 输出边长 |
| $K$ (`nSamples`) | 16 | 时间离散节点数 |
| $\sigma_t$ (`gaussSigma`) | 0.5 | 高斯时间权重标准差 |
| $T$ (`maxTrans`) | 28 px | 终点最大平移幅度 |
| $\theta_{\max}$ (`maxRotDeg`) | 10° | 终点最大旋转角 |
| $N_{\mathrm{try}}$ | 500 | 越界重采样上限 |
| $\theta_0$ | 0 | 起点旋转角 |

---

## 11. 与经典运动模糊的关系

| 模型 | PSF | 轨迹 |
|:---|:---|:---|
| **线性平移模糊** | 空间不变 | 所有像素同方向、等长度直线 |
| **线性空间变模糊** | 随位置变化 | 每像素直线，方向/长度可变 |
| **本文模型** | 由刚体插值诱导 | 每像素曲线 $\gamma_{u,v}(t)$，含旋转时非直线 |

当 $\theta_0 = \theta_1$ 且仅平移变化时，$\gamma_{u,v}(t)$ 退化为直线，模型退化为**空间不变的平移运动模糊**（在窗内）。

---

## 12. 实现对照

| 步骤 | MATLAB | PyTorch |
|:---|:---|:---|
| 位姿映射 | `pose_map` | `pose_map` |
| 双线性采样 | `interp2(...,'linear')` | `F.grid_sample(...,'bilinear')` |
| 时间积分 | `for k` 加权累加 | 同左 |
| 越界检查 | `pose_path_in_bounds` | `pose_path_in_bounds` |
| 可视化 | 轨迹 / 位移 / 取景窗 | `matplotlib` 六宫格 |

---

## 参考文献（概念）

- 空间变运动模糊：每像素独立积分路径，见 spatially-variant motion blur 文献。
- 刚体运动：$SE(2)$ 上曝光模拟；本文采用参数线性插值，实现简单、可微。
- 可微渲染 / 可微采样：`grid_sample` 双线性插值的梯度分析见 PyTorch 自动微分文档。
