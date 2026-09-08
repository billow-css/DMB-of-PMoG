# 联合噪声脚本定量说明

本文档定量描述脚本

`AS_VALs/demos/python/demo_combined_with_pimog.py`

该脚本将**非线性运动模糊**与 **PIMoG `ScreenShooting`** 串联，并用抖动强度**反比约束**摩尔纹权重。运动模糊的完整推导见 `nonlinear_motion_blur_algorithm.md`。

---

## 1. 符号与取值范围

| 符号 | 含义 | 典型取值 / 范围 |
|:---|:---|:---|
| $I$ | AS_VALs 原图 | $600\times 600$，像素值 $[0,1]$ |
| $S$ | 取景窗 / 中间分辨率 | $512$ |
| $B$ | 运动模糊输出 | $S\times S$，$[0,1]$ |
| $\tilde{B}$ | 映射后的模糊图 | $\tilde{B}=2B-1\in[-1,1]$ |
| $Y$ | `ScreenShooting` 输出 | 约 $[-1,1]$（加噪后可略越界） |
| $g_0,g_1$ | 起终点取景窗位姿 | 见运动模糊文档 |
| $m(u,v)$ | 逐像素轨迹长度 | $\|p_1-p_0\|_2$（像素） |
| $s$ | 抖动强度标量 | $s=\mathrm{mean}(m)$ |
| $\beta$ | 归一化抖动强度 | $[0,1]$ |
| $\alpha_L,\alpha_M$ | 光照 / 摩尔纹权重 | $\alpha_L+\alpha_M=1$，$\alpha_M\in[0,0.15]$ |

值域映射（脚本中 `o01_to_n11` / `n11_to_01`）：

$$
\tilde{X} = 2X - 1,\qquad
X = \mathrm{clip}\!\left(\frac{\tilde{X}+1}{2},\,0,\,1\right).
$$

---

## 2. 端到端管线

脚本默认管线为：

$$
I \;\xrightarrow{\;\mathrm{NMB}\;}\; B \;\xrightarrow{\;2(\cdot)-1\;}\; \tilde{B}
\;\xrightarrow{\;\mathrm{SS}(s)\;}\; Y
\;\xrightarrow{\;(\cdot+1)/2\;}\; Y_{[0,1]}
$$

其中：

- $\mathrm{NMB}$：非线性运动模糊（`nonlinear_motion_blur`），输出清晰曝光积分结果 $B$；
- $\mathrm{SS}(s)$：PIMoG 拍屏噪声层，摩尔纹权重由抖动强度 $s$ 自适应；
- 对照支路：对起点清晰裁切 $B^{\mathrm{sharp}}$ 使用固定 $\alpha_M=0.15$ 的 $\mathrm{SS}$。

对照关系：

$$
\begin{aligned}
B^{\mathrm{sharp}} &= \mathrm{SharpView}(I;\,g_0), \\
Y^{\mathrm{fixed}} &= \mathrm{SS}\bigl(2B^{\mathrm{sharp}}-1;\,\alpha_M=0.15\bigr), \\
Y^{\mathrm{adapt}} &= \mathrm{SS}\bigl(2B-1;\,s=\overline{m}\bigr).
\end{aligned}
$$

可视化五图：原图 $I$、清晰裁切 $B^{\mathrm{sharp}}$、仅模糊 $B$、固定摩尔纹 $Y^{\mathrm{fixed}}$、自适应联合 $Y^{\mathrm{adapt}}$。

---

## 3. 运动模糊子模块（摘要）

对取景窗局部坐标 $(u,v)$，刚体位姿插值 $g(t)$，$t\in[0,1]$，输出

$$
B_c(u,v)=\sum_{k=1}^{K} w_k\,
\mathcal{B}[I_c]\!\bigl(\mathbf{p}(u,v;\,g(t_k))\bigr),
$$

其中 $w_k$ 为归一化高斯时间权重，$\mathcal{B}$ 为双线性采样。细节见 `nonlinear_motion_blur_algorithm.md`。

脚本默认超参：

| 参数 | 符号 | 默认值 |
|:---|:---:|:---:|
| 时间采样数 | $K$ | $16$ |
| 高斯 $\sigma_t$ | `gauss_sigma` | $0.5$ |
| 最大平移 | $T$ | $28$ px |
| 最大转角 | $\theta_{\max}$ | $10^\circ$ |
| 越界重采上限 | — | $500$ |

位姿采样保证插值路径上取景窗四角始终落在 $[1,W]\times[1,H]$ 内。

---

## 4. 抖动强度定义

对端点映射

$$
\mathbf{p}_0(u,v)=\mathbf{p}(u,v;g_0),\qquad
\mathbf{p}_1(u,v)=\mathbf{p}(u,v;g_1),
$$

定义逐像素轨迹弦长

$$
m(u,v)=\bigl\|\mathbf{p}_1(u,v)-\mathbf{p}_0(u,v)\bigr\|_2.
$$

脚本取**空间均值**作为抖动强度：

$$
s \;=\; \overline{m}
\;=\; \frac{1}{S^2}\sum_{u,v} m(u,v)
\;=\; \texttt{aux["mag"].mean()}.
$$

$s$ 的单位为像素。有旋转时 $m(u,v)$ 随位置变化，$s$ 反映整窗平均抖动幅度。

---

## 5. 摩尔纹—抖动反比权重

### 5.1 归一化抖动

给定参考强度 $s_{\mathrm{ref}}$（代码 `blur_ref`，默认 $40$）：

$$
\beta \;=\; \mathrm{clip}\!\left(\frac{s}{s_{\mathrm{ref}}},\,0,\,1\right).
$$

### 5.2 凸组合权重

无抖动时摩尔纹上限 $\alpha_M^{\max}$（代码 `moire_max`，默认 $0.15$，与 PIMoG 论文一致）：

$$
\boxed{
\begin{aligned}
\alpha_M &= \alpha_M^{\max}\,(1-\beta), \\
\alpha_L &= 1-\alpha_M.
\end{aligned}
}
$$

性质：

| 条件 | $\beta$ | $(\alpha_L,\alpha_M)$ |
|:---|:---:|:---|
| $s=0$（无抖动） | $0$ | $(0.85,\,0.15)$（论文默认） |
| $0<s<s_{\mathrm{ref}}$ | $s/s_{\mathrm{ref}}$ | 摩尔纹线性减弱 |
| $s\ge s_{\mathrm{ref}}$ | $1$ | $(1,\,0)$（摩尔纹关闭） |

即：**抖动越强，摩尔纹越弱**；二者在凸组合意义下此消彼长，且始终满足

$$
\alpha_L + \alpha_M = 1,\qquad \alpha_M \in [0,\,\alpha_M^{\max}].
$$

实现函数：`Noise_Layer.blur_moire_weights(s, moire_max, blur_ref)`。

---

## 6. PIMoG `ScreenShooting` 定量形式

设输入 $\tilde{X}\in\mathbb{R}^{B\times C\times H\times W}$（约 $[-1,1]$）。前向为：

### 6.1 透视

$$
X^{(1)} = \mathcal{P}_d(\tilde{X}),
$$

其中 $\mathcal{P}_d$ 为四角扰动幅度 $d=2$（像素）的随机透视扭曲（Kornia `warp_perspective`）。

### 6.2 光照与摩尔纹

$$
L = \mathrm{LightMask}(\cdot),\qquad
M = 2\cdot\mathrm{Moire}(\cdot)-1,
$$

- $L$：线性或径向光照掩膜（随机二选一）；
- $M$：三通道独立摩尔纹，映射到约 $[-1,1]$。

### 6.3 加权融合 + 高斯噪声

$$
\boxed{
Y = X^{(1)}\odot L\cdot\alpha_L \;+\; M\cdot\alpha_M \;+\; \sigma\,\varepsilon
}
$$

其中 $\varepsilon\sim\mathcal{N}(0,I)$，$\sigma=\sqrt{0.001}$，$\odot$ 为逐元素乘。

- 固定模式：`forward(x, moire_weight=0.15)` $\Rightarrow$ $\alpha_M=0.15$；
- 自适应模式：`forward(x, blur_strength=s)` $\Rightarrow$ 由 §5 计算 $(\alpha_L,\alpha_M)$。

脚本中：

$$
Y^{\mathrm{fixed}}=\mathrm{SS}(\tilde{B}^{\mathrm{sharp}};\,\alpha_M=0.15),
\qquad
Y^{\mathrm{adapt}}=\mathrm{SS}(\tilde{B};\,s=\overline{m}).
$$

---

## 7. 完整计算图（自适应支路）

$$
\begin{aligned}
&(g_0,g_1)\leftarrow\mathrm{SamplePoseInBounds}(I), \\
&B\leftarrow\mathrm{NMB}(I;g_0,g_1), \\
&s\leftarrow\mathrm{mean}\bigl\|\mathbf{p}_1-\mathbf{p}_0\bigr\|_2, \\
&(\alpha_L,\alpha_M,\beta)\leftarrow\mathrm{BlurMoireWeights}(s), \\
&\tilde{B}\leftarrow 2B-1, \\
&Y\leftarrow \mathcal{P}_d(\tilde{B})\odot L\cdot\alpha_L + M\cdot\alpha_M + \sigma\varepsilon.
\end{aligned}
$$

---

## 8. 脚本默认参数汇总

| 名称 | 默认值 | 作用 |
|:---|:---:|:---|
| `view_size` | $512$ | 模糊输出边长 |
| `n_samples` | $16$ | 曝光时间离散点数 $K$ |
| `gauss_sigma` | $0.5$ | 时间高斯 $\sigma_t$ |
| `max_trans` | $28$ | 终点最大平移（px） |
| `max_rot_deg` | $10$ | 终点最大转角（度） |
| `blur_ref` | $40$ | $s_{\mathrm{ref}}$，$\beta=1$ 阈值 |
| `moire_max` | $0.15$ | $\alpha_M^{\max}$ |
| `perspective` $d$ | $2$ | 透视角点扰动（px） |
| 高斯噪声方差 | $0.001$ | $\sigma^2$ |

数值示例（与 `blur_moire_weights` 一致）：

| $s$ (px) | $\beta$ | $\alpha_L$ | $\alpha_M$ |
|:---:|:---:|:---:|:---:|
| $0$ | $0.00$ | $0.850$ | $0.150$ |
| $10$ | $0.25$ | $0.887$ | $0.112$ |
| $20$ | $0.50$ | $0.925$ | $0.075$ |
| $40$ | $1.00$ | $1.000$ | $0.000$ |

---

## 9. 输入 / 输出文件

### 输入

- 随机选取 `AS_VALs/outputs/text_images/*.png`（$600\times 600$ RGB）。

### 输出目录

`AS_VALs/python_motion_blur/out/`

| 文件 | 内容 |
|:---|:---|
| `combined_demo_<id>.png` | 六宫格总览（含权重注释） |
| `combined_<id>/01_original.png` | 原图 |
| `combined_<id>/02_sharp_crop.png` | 起点清晰裁切 |
| `combined_<id>/03_motion_blur.png` | 仅运动模糊 |
| `combined_<id>/04_pimog_fixed_moire.png` | 固定 $\alpha_M=0.15$ |
| `combined_<id>/05_combined_adaptive_moire.png` | 模糊 + 自适应摩尔纹 |
| `combined_<id>/weights.txt` | $s,\beta,\alpha_L,\alpha_M$ 数值 |

---

## 10. 运行方式

在仓库根目录执行：

```bash
python AS_VALs/demos/python/demo_combined_with_pimog.py
```

依赖：`torch`、`kornia`、`matplotlib`、`PIL`，以及同目录下的 `demo_nonlinear_motion_blur.py` 与仓库根目录的 `Noise_Layer.py`。

---

## 11. 与单独模块的关系

| 模块 | 文档 / 代码 | 在本脚本中的角色 |
|:---|:---|:---|
| 非线性运动模糊 | `nonlinear_motion_blur_algorithm.md` | 产生 $B$ 与 $m(u,v)$ |
| PIMoG 噪声层 | `Noise_Layer.py` | $\mathcal{P}_d$、光照、摩尔纹、高斯 |
| 反比权重 | `blur_moire_weights` | 由 $s$ 得到 $(\alpha_L,\alpha_M)$ |
| 本脚本 | `demo_combined_with_pimog.py` | 串联、对照可视化、落盘 |

本脚本**不训练网络**；它是联合失真的**定量可视化与接口验证**。若接入 PIMoG 训练，可将 `ScreenShooting(..., blur_strength=s)` 作为 `Noiser`，其中 $s$ 由同一 batch 的运动模糊 `aux["mag"]` 给出。
