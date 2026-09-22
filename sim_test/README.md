# sim_test —— 仿真拍屏（噪声层替代人工翻拍）

用与 **eval 相同** 的可微噪声层，对指定文件夹里的图像做失真叠加，输出到另一目录。  
用于快速验证 **PIMoG / DMB-PMoG**（以及挂在同一架构上的第三方含水印图），**无需手机拍屏**。

本目录脚本**不加载 Encoder/Decoder**，只做噪声；解码仍用主程序的 `test_accuracy` 等模式。

## 噪声

| CLI | 含义 |
|-----|------|
| `ScreenShooting` / `PIMoG` | 原论文族：透视 + 光照 + 摩尔纹 + 高斯 |
| `ScreenShootingMB` / `DMB` | DMB of PMoG：+ 动态运动模糊 |
| `both` | 两种各写一套子目录 |

## 用法

```bash
# DMB 噪声（默认推荐）
python sim_test/sim_noise.py ^
  --input_dir Datasets/images_watermarked ^
  --output_dir sim_test/out_dmb ^
  --noise DMB

# 原 PIMoG 噪声
python sim_test/sim_noise.py -i Datasets/images_watermarked -o sim_test/out_ss --noise PIMoG

# 两种都跑
python sim_test/sim_noise.py -i path/in -o path/out --noise both --seed 42

# 可选：缩放、限量、CPU
python sim_test/sim_noise.py -i path/in -o path/out --noise DMB --image_size 128 --limit 100 --cpu
```

主菜单也可选 **sim_test**（若已接入），或：

```bash
python main.py --mode tool_sim_test --no_interactive
# 仍建议直接调 sim_noise.py 并传 -i/-o
```

## 建议流水线

1. 嵌入：`test_embedding` 或已有含水印图目录  
2. **本脚本**叠噪声 → `sim_test/out_*`  
3. 测准：`python main.py --mode test_accuracy --image_dir sim_test/out_dmb ...`

输入输出文件名一一对应，便于按索引对齐 `w.mat`。
