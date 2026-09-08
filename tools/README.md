# tools/

数据准备与嵌入质量评估（相对仓库根运行）。

| 脚本 | 作用 |
|:---|:---|
| `export_coco_hosts.py` | 从 COCOMask 导出宿主到 `Datasets/images` |
| `crop_embed_panels.py` | 从结果面板裁出嵌入图 |
| `verify_host_psnr.py` | 嵌入后相对宿主的 PSNR |
| `verify_host_ssim.py` | 嵌入后相对宿主的 SSIM |
| `verify_host_embed.py` | 统一入口：`--metric psnr\|ssim\|both` |
| `generate_text_images.py` | AS_VALs 文字图合成 → `AS_VALs/outputs/text_images` |

| `top200_compare_run.py` | PSNR 报告 Top-200 对比分析 |

根目录保留同名兼容入口（`export_coco_hosts.py` 等），内部转发到本目录。
