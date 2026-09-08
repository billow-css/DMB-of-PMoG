# PIMoG / DMB of PMoG 工作区目录说明

**DMB of PMoG** = *Dynamic Motion Blur of PMoG*  
（相对原 PIMoG 少了光照验证的 I，故写作 PMoG；主噪声层为动态运动模糊 DMB。）

屏摄鲁棒水印训练与实验代码（谱系：ACM MM 2022 PIMoG）。

## 目录结构

| 路径 | 说明 |
|:---|:---|
| `main.py` / `model.py` / `solver.py` / `Noise_Layer.py` / `data_loader.py` / `cli_utils.py` | **训练与推理核心**（留在仓库根，避免改 import） |
| `tools/` | 数据准备与评估脚本 |
| `experiment/` | 联机拍屏采集、矫正、GUI |
| `AS_VALs/python_motion_blur/` | 可微非线性运动模糊核心（`Noise_Layer` 依赖） |
| `AS_VALs/demos/` | Python / MATLAB 演示 |
| `AS_VALs/outputs/` | 文字图与 demo 产物 |

| `models/` | 权重；`runs/<run_id>/` 每次训练完整 epoch 归档；`_misc/` 旁路副本 |
| `Datasets/` | 宿主图、嵌入图、Recover 等 |
| `results/` | 训练输出、PSNR/SSIM 报告、水印矩阵 |
| `logs/` | 训练日志（`logs/runs/<run_id>.json` 永不覆盖；CLI `l` 可查询） |
| `docs/` | 算法/实验文档 |
| `reference/` | 上游 PIMoG 源码、透视 MATLAB、压缩包归档 |
| `experiment/legacy/` | 旧版慢速矫正（仅回退） |
| `experiment/_scratch/` | 临时调试图/预览，可删 |

## 常用入口

```text
# 交互功能面板（DMB of PMoG）
python main.py
# 0–4 训练/评估 · 5–9 工具 · a–c 拍屏实验 · l 训练日志 · q 退出
# 选训练后：弹窗改超参 → 控制台打印明细 → 开训；每 epoch 弹 msg/den/loss 曲线

# 训练 / 测试（也可 --mode … --no_interactive）
python main.py --mode eval_mask --no_interactive --eval_ckpt best --distortion ScreenShootingMB

# 工具直达
python main.py --mode tool_verify_psnr --no_interactive
python tools/export_coco_hosts.py
python tools/verify_host_embed.py --metric both

# 拍屏实验
cd experiment
python server_gui.py
python batch_rectify.py
python rectify_gui.py
```

## 脚本归并说明（功能未改）

- `batch_rectify.py` ← 原 `batch_rectify_fast.py`（默认快速路径）
- `batch_rectify_fast.py` → 兼容再导出 / CLI 转发
- `legacy/batch_rectify_legacy.py` ← 原慢速 `batch_rectify.py`
- `tools/verify_host_*.py` ← 原根目录评估脚本；`verify_host_embed.py` 为统一入口
- 根目录 `verify_host_*.py` / `export_coco_hosts.py` / `crop_embed_panels.py` 为兼容转发

`markers.py`（合成/完整检测）与 `markers_fast.py`（缩略图快速检测）职责不同，保留两套；后者依赖前者。

## 权重

交互 / `--eval_ckpt` / `--init_from_ckpt` 可选三类：

| 标签 | 文件 |
|------|------|
| `99` | `models/ScreenShooting/Encoder_Decoder_Model_mask_99.pth` |
| `ss_best` | `models/Encoder_Decoder_Model_mask_ScreenShooting_best.pth` |
| `best`（别名 `mb_best`） | `models/Encoder_Decoder_Model_mask_ScreenShootingMB_best.pth`（DMB） |

训练归档（全量 / Lite 相同）：

```
models/runs/<run_id>/
  Encoder_Decoder_Model_mask_0000.pth   # 每个 epoch 必存
  Encoder_Decoder_Model_mask_0001.pth
  …
  Encoder_Decoder_Model_mask_<噪声层>_best.pth
  *.meta.txt
  run_info.txt
models/LATEST_RUN.txt                   # 指向最近一次 run
```

全局 `…_best.pth` 仍会更新，供评估菜单使用；历史 run 权重互不覆盖。`models/<噪声层>/` 下仍按 `--model_save_step` 写一份便于续训。

- `models/_misc/` 中 AS_VALs 旁路副本与正式权重 **hash 不同**，勿混用
