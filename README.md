# DMB of PMoG —— 屏摄鲁棒水印

> **DMB of PMoG** = *Dynamic Motion Blur of PMoG*（动态运动模糊的 PMoG）。
> 一套面向 **屏幕翻拍（Screen-Shooting）场景的鲁棒数字水印**训练与实验代码，
> 谱系来自 **ACM MM 2022 PIMoG**。

相对原 PIMoG 去掉了光照验证的 `I`（故写作 **PMoG**），并把主噪声层升级为
**DMB（Dynamic Motion Blur，空变非线性运动模糊）**，使网络对"手持手机拍屏"
这类真实退化更鲁棒。

---

## 目录

- [核心改进](#核心改进)
- [架构概览](#架构概览)
- [目录结构](#目录结构)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [预训练权重](#预训练权重)
- [命令行用法](#命令行用法)
- [交互功能面板](#交互功能面板)
- [训练](#训练)
- [拍屏实验](#拍屏实验)
- [文档](#文档)
- [引用](#引用)
- [License](#license)

---

## 核心改进

原 PIMoG 的噪声层 `ScreenShooting` 由「透视 + 光照 + 摩尔纹 + 高斯」组成，**不含手持抖动模糊**。
本仓库在此基础上：

1. **加入空变非线性运动模糊（DMB）**：对取景窗刚体抖动做严格几何建模，
   得到随空间位置变化的（Spatially-Variant）运动模糊核，可微、可端到端训练；
2. **去掉光照验证项 `I`**：聚焦运动模糊这一主要退化，简化训练目标；
3. 形成新噪声层 `ScreenShootingMB = ScreenShooting + Motion Blur`，作为默认训练/评估噪声层。

数学表述见 [`docs/asvals/nonlinear_motion_blur_algorithm.md`](docs/asvals/nonlinear_motion_blur_algorithm.md)，
可微实现见 [`AS_VALs/python_motion_blur/motion_blur.py`](AS_VALs/python_motion_blur/motion_blur.py)。

---

## 架构概览

标准 **Encoder–Noise–Decoder** 水印网络，训练时带 GAN 判别器：

```
宿主图 + 水印消息 w
        │
   ┌────▼────┐
   │ Encoder │  → 含水印图 Encoded（与宿主视觉接近）
   └────┬────┘
        │
   ┌────▼──────────────┐
   │ Noise Layer       │  可微噪声层：Identity / ScreenShooting / ScreenShootingMB
   │  (透视/摩尔纹/     │
   │   运动模糊/…)     │
   └────┬──────────────┘
        │
   ┌────▼────┐
   │ Decoder │  → 恢复水印消息 w′（训练中与 w 对齐）
   └─────────┘
```

- 宿主图默认 **128×128**；
- 噪声层 **可微**，训练时梯度可回传到 Encoder，使编码对拍屏退化鲁棒；
- 判别器通道数默认 64，用于对抗式约束编码图的视觉质量。

---

## 目录结构

| 路径 | 说明 |
|:---|:---|
| `main.py` / `model.py` / `solver.py` | 训练与推理核心（入口 / 网络 / 训练求解器） |
| `Noise_Layer.py` | 可微噪声层（`Identity` / `ScreenShooting` / `ScreenShootingMB`） |
| `data_loader.py` / `cli_utils.py` / `ckpt_utils.py` | 数据加载 / CLI 与交互 / 权重解析 |
| `ui/`（`train_ui.py` / `train_log.py` / `verify_weights_ui.py`） | 训练弹窗 UI / 日志 / 权重校验 |
| `AS_VALs/python_motion_blur/` | **DMB 运动模糊核心**（`Noise_Layer` 依赖） |
| `AS_VALs/demos/`、`AS_VALs/matlab_motion_blur/` | Python / MATLAB 演示 |
| `tools/` | 数据准备与评估脚本（宿主导出 / PSNR / SSIM / 裁切） |
| `experiment/` | 联机拍屏采集、透视矫正、GUI |
| `models/` | 预训练权重（见 [预训练权重](#预训练权重)） |
| `Datasets/` | 宿主图、嵌入图、Recover 等（未入库，自行准备） |
| `results/` / `logs/` | 训练输出与报告 / 训练日志（未入库） |
| `docs/` | 算法 / 实验文档 |
| `reference/` | 上游 PIMoG 源码、透视 MATLAB |

> `Datasets/`、`results/`、`logs/`、`experiment/captures/` 等体积较大或可再生成，已通过
> `.gitignore` 排除，克隆后按需自行生成。

---

## 环境安装

核心训练依赖（建议 Python 3.10+，GPU + CUDA）：

```bash
pip install torch torchvision          # 按你的 CUDA 版本安装，见 pytorch.org
pip install kornia numpy scipy opencv-python matplotlib tqdm Pillow
```

拍屏实验（`experiment/`）额外依赖：

```bash
pip install -r experiment/requirements.txt   # Pillow / opencv-python / numpy
# 佳能相机需另行放置 EDSDK（见 experiment/ 内说明）
```

---

## 快速开始

```bash
# 1) 交互功能面板（推荐，无参数即可）
python main.py

# 2) 跳过训练，直接评估 DMB 模型
python main.py --mode eval_mask --no_interactive --eval_ckpt best --distortion ScreenShootingMB

# 3) 评估 ScreenShooting 基线
python main.py --mode eval_mask --no_interactive --eval_ckpt ss_best --distortion ScreenShooting
```

---

## 预训练权重

仓库内已包含两个最优 checkpoint：

| 标签 | 文件 | 噪声层 | 说明 |
|:---|:---|:---|:---|
| `best`（别名 `mb_best`） | `models/Encoder_Decoder_Model_mask_ScreenShootingMB_best.pth` | `ScreenShootingMB` | **DMB 主模型（Ours）**，验证 Acc ≈ 99.57% |
| `ss_best` | `models/Encoder_Decoder_Model_mask_ScreenShooting_best.pth` | `ScreenShooting` | ScreenShooting 基线（原 PIMoG 噪声层） |

其余权重（每 epoch 归档、初始化的 `99` 等）未入库，需要时从训练日志/运行目录恢复。
权重标签在 CLI 中可用于 `--eval_ckpt` / `--init_from_ckpt`：

```text
99        原 PIMoG 预训练（models/ScreenShooting/Encoder_Decoder_Model_mask_99.pth）
ss_best   ScreenShooting 基线 best
best      DMB 主模型 best（= mb_best）
custom    通过 --eval_path 直接指定 .pth 文件
```

---

## 命令行用法

统一入口 `python main.py`，`--mode` 决定运行模式：

### 核心模式

| `--mode` | 作用 |
|:---|:---|
| `train_mask` | 端到端训练 Encoder–Noise–Decoder（含 GAN） |
| `eval_mask` | 跳过训练，加载权重评估 Acc/BER/PSNR + 可视化 |
| `test_embedding` | 用预训练模型嵌入水印并导出可视化拼图 |
| `test_accuracy` | 对矫正后的拍屏图评估 Acc/BER，并相对含水印宿主算 PSNR |

### 工具模式

| `--mode` | 作用 |
|:---|:---|
| `tool_export_hosts` | 导出 COCO 宿主图 |
| `tool_verify_psnr` / `tool_verify_ssim` / `tool_verify_both` | 嵌入质量 PSNR / SSIM / 两者 |
| `tool_crop_panels` | 裁切可视化面板 |
| `tool_batch_rectify` / `tool_rectify_gui` / `tool_server_gui` | 拍屏矫正 / GUI / 采集服务 |

### 常用参数

| 参数 | 取值 | 说明 |
|:---|:---|:---|
| `--distortion` | `Identity` / `ScreenShooting` / `ScreenShootingMB` | 噪声层；省略则交互选择 |
| `--eval_ckpt` | `99` / `ss_best` / `best` / `mb_best` / `custom` | 评估/嵌入/测准用的权重标签 |
| `--eval_path` | 路径 | 直接指定 `.pth`（优先于 `--eval_ckpt`） |
| `--init_from_ckpt` | `99` / `ss_best` / `best` / `mb_best` | 训练初始化权重 |
| `--embed_strength` | float | 嵌入强度（默认 0.0） |
| `--lite` / `--full` | flag | 训练规模（Lite / 全量） |
| `--no_interactive` | flag | 关闭交互，纯命令行 |
| `--image_size` | int | 宿主图边长（默认 128） |

更多参数见 `python main.py -h`。

### 示例

```bash
# 评估
python main.py --mode eval_mask --no_interactive --eval_ckpt 99  --embed_strength 0
python main.py --mode eval_mask --no_interactive --eval_ckpt best --distortion ScreenShootingMB

# 训练（Lite / 全量）
python main.py --mode train_mask --lite --distortion ScreenShootingMB
python main.py --mode train_mask --full

# 工具
python main.py --mode tool_verify_psnr --no_interactive
python main.py --mode tool_verify_both --no_interactive
```

---

## 交互功能面板

`python main.py` 进入面板，菜单：

```text
0–4  训练 / 评估（eval_mask、train_mask、test_embedding、test_accuracy）
5–9  工具（宿主导出、PSNR、SSIM、裁切等）
a–c  拍屏实验（采集 / 矫正 / GUI）
l    训练日志查询（logs/runs/<run_id>.json，永不覆盖）
q    退出
```

选训练后会弹窗改超参 → 控制台打印明细 → 开训；每 epoch 弹出 msg/den/loss 曲线。

---

## 训练

训练采用 Encoder–Noise–Decoder + GAN 的端到端框架：

- **噪声层**通过 `--distortion` 指定；`ScreenShootingMB` 为本文主推；
- **规模**：`--lite`（快速验证）与 `--full`（完整训练）；
- **续训**：`--init_from_ckpt` 指定初始化权重；每 epoch 按 `--model_save_step` 存盘；
- 权重归档在 `models/runs/<run_id>/`，历史 run 互不覆盖；`models/LATEST_RUN.txt` 指向最近一次。

---

## 拍屏实验

`experiment/` 提供联机拍屏（佳能 EDSDK 相机）→ 矫正 → 评估的完整链路：

```bash
cd experiment
python server_gui.py      # 采集服务 GUI
python batch_rectify.py   # 批量透视矫正
python rectify_gui.py     # 矫正 GUI
```

`markers.py`（合成/完整检测）与 `markers_fast.py`（缩略图快速检测）职责不同，保留两套（后者依赖前者）。

---

## 文档

| 文档 | 内容 |
|:---|:---|
| [`docs/运动模糊噪声层_保姆级讲解.md`](docs/运动模糊噪声层_保姆级讲解.md) | DMB 噪声层的直观讲解 |
| [`docs/asvals/nonlinear_motion_blur_algorithm.md`](docs/asvals/nonlinear_motion_blur_algorithm.md) | 空变非线性运动模糊的数学表述 |
| [`docs/asvals/非线性运动模糊方案_参数与实现.md`](docs/asvals/非线性运动模糊方案_参数与实现.md) | 参数区间与实现说明 |
| [`docs/PIMoG_非线性运动模糊改进汇报.md`](docs/PIMoG_非线性运动模糊改进汇报.md) | 改进汇报 |
| [`AS_VALs/README.md`](AS_VALs/README.md) | AS_VALs 实验说明 |

---

## 引用

本项目基于以下工作：

```bibtex
@inproceedings{fang2022pimog,
  title     = {PIMoG: An Effective Screen-shooting Noise-Layer Simulation for
               Deep-Learning-Based Watermarking Network},
  author    = {Fang, Han and others},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)},
  year      = {2022}
}
```

---

## License

[MIT](LICENSE)（沿用上游 PIMoG 许可证，详见 [LICENSE](LICENSE)）。
