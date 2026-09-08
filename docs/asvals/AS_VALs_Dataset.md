# AS_VALs Dataset

**全称：ASCII VALs Dataset（ASCII 码变量数据集）**

- **ASCII**：American Standard Code for Information Interchange，可打印 ASCII 字符  
- **VALs**：Variables（变量）——每张图对应一个离散的字符串变量（短词、短语或随机 ASCII 串）

本数据集用于屏幕拍摄 / 运动模糊等失真实验中的**文本载体测试集**：白底黑字、固定画布、可控排版，便于观察几何与光学失真对字符结构的影响。

生成脚本：`tools/generate_text_images.py`  
输出目录：`AS_VALs/outputs/text_images/`  
图像目录：`out/`  
清单文件：`out/manifest.txt`

---

## 1. 概览

| 项目 | 规格 |
|:---|:---|
| 数据集全称 | ASCII VALs Dataset |
| 简称 | AS_VALs |
| 图像数量 | **15 000** |
| 分辨率 | **$600\times 600$** RGB |
| 颜色 | 白底 (`#FFFFFF`)，黑字 (`#000000`) |
| 文件格式 | PNG |
| 命名 | `00000.png` … `14999.png`（5 位零填充） |
| 标签清单 | `manifest.txt`：每行 `文件名\t字符串内容` |

---

## 2. 设计目标

1. **可控文本载体**：相对自然图，字符边缘清晰，便于评估拍屏噪声、运动模糊对可辨识性的破坏。  
2. **ASCII 变量空间**：内容为可打印 ASCII（字母、数字、标点及少量空格），模拟短变量名 / 口令片段 / 短语。  
3. **固定几何框**：统一画布与可写区域，方便与 $512\times 512$ 取景窗、后续噪声层对齐。  
4. **可复现生成**：由脚本批量合成，非爬取；清单记录图—文一一对应。

---

## 3. 图像几何布局

$$
\begin{aligned}
\text{画布} &\quad W = H = 600, \\
\text{页边距} &\quad M = 100, \\
\text{可写区域} &\quad D = W - 2M = 500.
\end{aligned}
$$

文字绘制在中心 $500\times 500$ 区域内，四周留白 $100$ 像素，降低贴边裁切风险。

```
┌──────────────── 600 ────────────────┐
│            margin 100               │
│   ┌────────── 500 ──────────┐       │
│   │                         │       │
│   │     黑字文本（居中）      │  600  │
│   │                         │       │
│   └─────────────────────────┘       │
│                                     │
└─────────────────────────────────────┘
```

字号由二分搜索取**最大仍落入可写区域**的字体大小（约 $12$–$220$），优先系统 Arial Bold 等 TrueType 字体。

---

## 4. 文本约束（变量形态）

每张图对应一个字符串变量 $v$，满足：

| 约束 | 取值 |
|:---|:---|
| 每行最大字符数 | $6$ |
| 最大行数 | $2$ |
| 总字符数上界 | $12$（含行间空格规则下的可排版长度） |
| 字符集 | 可打印 ASCII：`0-9A-Za-z` + 标点 + 偶发空格 |

排版规则（`wrap_text`）：

- 优先在空格处换行；
- 超长 token 按 $6$ 字符硬切；
- 无法在 $2\times 6$ 内排下则拒绝该候选，重新采样。

因此每张图在视觉上多为 **1–2 行、行宽 ≤ 6** 的短文本块。

---

## 5. 内容分布

生成时两类来源混合（默认比例）：

| 类型 | 比例 | 说明 |
|:---|:---:|:---|
| **English** | $\approx 40\%$ | 词表 `WORDS` / 短语表 `PHRASES` 中的短英文（如 `return`、`new book`） |
| **Random ASCII** | $\approx 60\%$ | 长度 $1$–$12$ 的随机可打印 ASCII 串（如 `Yyb=<`、`$ >nD#rEV^aI`） |

English 支路会偶尔拼接两个短词成短语（在总长约束内）。Random 支路避免首尾空格，内部空格概率较低。

完整图—文对照见：

```text
AS_VALs/outputs/text_images/manifest.txt
```

格式示例：

```text
00003.png	talk
00010.png	new book
00000.png	Yyb=<
```

---

## 6. 目录结构

```text
AS_VALs/
├── README.md
├── python_motion_blur/            ← 核心库 motion_blur.py（训练依赖）
├── demos/
│   ├── python/                    ← PyTorch 演示
│   └── matlab/                    ← MATLAB 对照
├── outputs/
│   ├── text_images/               ← 15k 合成图 + manifest.txt
│   ├── demo_python/               ← Python demo 产物
│   └── demo_matlab/               ← MATLAB demo 产物
└── matlab_motion_blur/            ← 旧路径兼容转发
```

本文件路径：`docs/asvals/AS_VALs_Dataset.md`。

---

## 7. 生成与复现

```bash
# 在 AS_VALs 目录或仓库中执行
python tools/generate_text_images.py
```

脚本常量（与当前仓库一致）：

| 常量 | 值 |
|:---|:---:|
| `NUM_IMAGES` | $15000$ |
| `IMG_SIZE` | $600$ |
| `MARGIN` | $100$ |
| `ENGLISH_RATIO` | $0.40$ |
| `MAX_CHARS_PER_LINE` | $6$ |
| `MAX_LINES` | $2$ |

重新生成会覆盖 `out/` 下 PNG 与 `manifest.txt`（请先备份）。

---

## 8. 推荐用法（本仓库）

| 用途 | 说明 |
|:---|:---|
| 运动模糊实验 | 随机抽图 → $512$ 取景窗刚体抖动积分 |
| 联合噪声可视化 | 模糊后再接 PIMoG `ScreenShooting` |
| 观感 / 消融对照 | 白底黑字使拖影、摩尔纹、透视更易肉眼分辨 |
| 非 COCO 训练替代 | 可作为文本场景的额外测试集；**默认 PIMoG 训练仍用 COCOMask** |

典型加载（$[0,1]$ RGB）：

```python
from PIL import Image
import numpy as np
img = np.asarray(Image.open("AS_VALs/outputs/text_images/00010.png").convert("RGB"), dtype=np.float32) / 255.0
# img.shape == (600, 600, 3)
```

---

## 9. 与 PIMoG 主训练集的区别

| | **AS_VALs** | **COCOMask（PIMoG）** |
|:---|:---|:---|
| 内容 | 合成 ASCII 文本 | 自然图 + mask 拼接 |
| 尺寸 | $600\times 600$ | 训练时常裁到 $128\times 256$ 等 |
| 角色 | 文本拍屏 / 模糊**测试与可视化** | Encoder–Decoder **主训练数据** |
| 标签 | `manifest.txt` 字符串 | 消息比特由训练流程采样 |

---

## 10. 引用式简述（可放论文）

> **AS_VALs (ASCII VALs Dataset)** is a synthetic test set of 15,000 RGB images of size $600\times 600$. Each image renders a short printable-ASCII *variable* (at most two lines, six characters per line) in black on a white background, with a 100-pixel margin. About 40% of samples are English words/phrases and 60% are random ASCII strings. A manifest maps each filename to its ground-truth string.

---

## 11. 许可与注意事项

- 图像由本地脚本合成，不含第三方摄影版权内容。  
- 字体依赖运行环境系统字体（如 Arial）；缺字体时回退默认位图字体，观感可能变化。  
- 随机 ASCII 可能含标点，文件名与内容以 `manifest.txt` 为准；读取清单时注意转义字符（如 `\`）。
