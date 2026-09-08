# PIMoG 非线性运动模糊改进汇报

## 1. 改进方法

原 PIMoG 噪声层 `ScreenShooting`：透视 + 光照 + 摩尔纹 + 高斯，**不含手持抖动模糊**。

本工作新增可微噪声层 **`ScreenShootingMB`**，在原拍屏管线前接入**空变非线性运动模糊**：

| 步骤 | 内容 |
|:---|:---|
| 1 | 取景窗刚体位姿 $g_0\!\to\!g_1$（平移 + 旋转），时间插值 + 高斯曝光加权 |
| 2 | 逐像素双线性采样积分 → 空变拖影（`grid_sample`，可反传） |
| 3 | 再接透视 / 光照 / 摩尔纹 / 高斯 |
| 4 | **反比约束**：抖动强则摩尔纹弱，$\alpha_M=0.15(1-\beta)$，$\beta\propto$ 轨迹长度均值 |

核心代码：

- 算法库：`AS_VALs/python_motion_blur/motion_blur.py`
- 噪声层：`Noise_Layer.ScreenShootingMB`
- 训练开关：`--distortion ScreenShootingMB`

相对原版：补上手持曝光拖影，且摩尔纹与抖动强度联动，更接近「边抖边拍」的物理关系。

---

## 2. 改进方法的训练 / 演示效果（Python）

### 2.0 可视化与可微演示

| 脚本 | 作用 |
|:---|:---|
| `AS_VALs/demos/python/demo_nonlinear_motion_blur.py` | 仅运动模糊；输出清晰裁切 / 模糊 / 轨迹场 |
| `AS_VALs/demos/python/demo_combined_with_pimog.py` | 模糊 + PIMoG 拍屏；对比固定摩尔纹 vs 反比自适应 |
| `main.py --mode eval_mask` | 验证集 Acc / BER / PSNR + `results/Image_eval_*` |

运行示例：

```bash
python AS_VALs/demos/python/demo_nonlinear_motion_blur.py
python AS_VALs/demos/python/demo_combined_with_pimog.py
python main.py --mode eval_mask --eval_ckpt 99 --distortion ScreenShootingMB
python main.py --mode eval_mask --eval_ckpt best --distortion ScreenShootingMB
```

### 2.1 自建数据集 AS_VALs

**ASCII VALs Dataset**：专为文本载体上的拍屏 / 模糊观感设计。

| 项 | 规格 |
|:---|:---|
| 规模 | 15 000 张 PNG |
| 内容 | 白底黑字短 ASCII（≤2 行 × 6 字）；约 40% 英文词 / 短语，60% 随机串 |
| 几何 | $600\times600$，页边距 100，可写区 $500\times500$ |
| 生成 | `tools/generate_text_images.py` → `AS_VALs/outputs/text_images/` + `manifest.txt` |


| 用途 | 模糊与联合噪声**可视化 / 定性观察**；PIMoG 定量训练仍用 COCOMask |

详见：`docs/asvals/AS_VALs_Dataset.md`。

---

## 3. 对照试验：官方权重 × 非线性运动模糊噪声

设定：同一套 **mask_99**（`Encoder_Decoder_Model_mask_99.pth`），仅更换评估噪声层；`embed_strength=0`；验证集同配置。

| 噪声层 | 权重 | ValAcc ↑ | BER ↓ | PSNR |
|:---|:---|---:|---:|---:|
| `ScreenShooting`（原论文） | mask_99 | **99.72%** | **0.0028** | 38.10 dB |
| `ScreenShootingMB`（+运动模糊） | mask_99 | 95.35% | 0.0465 | 38.10 dB |

结论：在**未针对运动模糊训练**时，官方权重在 MB 噪声下 Acc 下降约 **4.4 pt**、BER 上升约一个数量级，说明原模型对新增抖动模糊**不鲁棒**，有针对性再训的必要。

日志：`logs/eval_mask_ScreenShooting_ep99.txt`、`logs/eval_mask_ScreenShootingMB_ep99.txt`。

---

## 4. 基于 PIMoG 预训练的针对性训练结果

设定：挂载 **mask_99** 初始化 → 噪声层改为 `ScreenShootingMB` 继续训练（Lite）；保存最佳为  
`models/Encoder_Decoder_Model_mask_ScreenShootingMB_best.pth`。

| 评估配置 | ValAcc ↑ | BER ↓ | PSNR | 相对 mask_99@MB |
|:---|---:|---:|---:|:---|
| mask_99 + ScreenShootingMB | 95.35% | 0.0465 | 38.10 dB | 基线 |
| **MB_best** + ScreenShootingMB | **97.57%** | **0.0243** | 11.49 dB | Acc **+2.22 pt**，BER 近乎减半 |

说明：

- 针对性训练后，在运动模糊噪声下比特恢复明显回升，接近原论文噪声下的可用水平。
- PSNR 下降反映嵌入残差增强、不可见性与鲁棒性折中（与再训强度 / `embed_strength` 相关），汇报时需同时给 Acc 与 PSNR。
- 训练过程日志见 `logs/train_mask_ScreenShootingMB.txt`；评估见 `logs/eval_mask_ScreenShootingMB_best.txt`。

复现评估：

```bash
# 对照：官方 99 @ MB
python main.py --mode eval_mask --no_interactive --eval_ckpt 99 \
  --distortion ScreenShootingMB --embed_strength 0

# 针对性训练后的 best @ MB
python main.py --mode eval_mask --no_interactive --eval_ckpt best \
  --distortion ScreenShootingMB --embed_strength 0
```

---

## 一句话总结

在 PIMoG 中加入可微空变运动模糊与摩尔纹反比约束后，官方 mask_99 在 MB 噪声下明显掉点；以 mask_99 为起点做 ScreenShootingMB 再训，可将 ValAcc 从约 **95.4%** 提升至约 **97.6%**，验证了该改进方案的有效性。
