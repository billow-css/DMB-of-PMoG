# -*- coding: utf-8 -*-
"""应用层常量。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TOOL_MODE_CHOICES = (
    "tool_export_hosts",
    "tool_verify_psnr",
    "tool_verify_ssim",
    "tool_verify_both",
    "tool_crop_panels",
    "tool_sim_test",
    "tool_batch_rectify",
    "tool_rectify_gui",
    "tool_server_gui",
)

# 训练默认宿主目录（argparse 默认）；嵌入/测准未显式指定时改用下方路径
DEFAULT_TRAIN_IMAGE_DIR = "Datasets/COCOMask/train/train_class/"
DEFAULT_EMBED_IMAGE_DIR = "Datasets/images/"
DEFAULT_ACCURACY_IMAGE_DIR = "Datasets/Recover/capture/"

# 评估时与权重匹配的默认嵌入强度（用户显式 --embed_strength 时不覆盖）
EVAL_STRENGTH_OFFICIAL = 0.0  # mask_99 / 论文整图
EVAL_STRENGTH_TRAIN_DEFAULT = 0.0  # 与当前 --embed_strength 训练默认一致
