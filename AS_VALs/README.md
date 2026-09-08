# AS_VALs

运动模糊与合成文字图资源。按 **核心库 / 演示 / 输出** 三分开。

## 目录结构

```text
AS_VALs/
├── python_motion_blur/     # 核心库（Noise_Layer 固定依赖此路径）
│   └── motion_blur.py
├── demos/
│   ├── python/             # PyTorch 演示脚本
│   └── matlab/             # MATLAB 对照演示
├── outputs/
│   ├── text_images/        # 合成文字宿主（原 out/）
│   ├── demo_python/        # Python demo 运行产物
│   └── demo_matlab/        # MATLAB demo 运行产物
└── matlab_motion_blur/     # 仅保留旧路径兼容转发
```

| 路径 | 说明 |
|:---|:---|
| `python_motion_blur/motion_blur.py` | 可微非线性运动模糊；被根目录 `Noise_Layer.ScreenShootingMB` 引用，**勿随意挪动** |
| `demos/python/` | 模糊 / 复合噪声可视化 |
| `demos/matlab/` | MATLAB 对照实验 |
| `outputs/text_images/` | `tools/generate_text_images.py` 输出 |
| `outputs/demo_*` | demo 跑出来的图与 `run_info`，可删可归档 |

文档见仓库 `docs/asvals/`；压缩包在 `reference/archives/`。

## 常用命令

```text
# 合成文字图 → outputs/text_images/
python tools/generate_text_images.py

# 非线性运动模糊 demo
python AS_VALs/demos/python/demo_nonlinear_motion_blur.py

# 模糊 + PIMoG ScreenShooting
python AS_VALs/demos/python/demo_combined_with_pimog.py

# COCO 128 宿主模糊 / 复合
python AS_VALs/demos/python/demo_coco_nonlinear_motion_blur.py --limit 5
python AS_VALs/demos/python/demo_coco_combined_with_pimog.py --limit 3 --no_show
```

旧路径 `AS_VALs/python_motion_blur/demo_*.py` 仍可运行（转发到 `demos/python/`）。

## 归并说明

- `demo_nonlinear_motion_blur.py` 不再内嵌一套模糊实现，改为调用 `motion_blur.py`（与训练噪声层一致）
- `sharp_view` 已收入 `motion_blur.py` 库 API
- 原 `out/` / `out_coco/` / `matlab_motion_blur/out/` 统一到 `outputs/`
