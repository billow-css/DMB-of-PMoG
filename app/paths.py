# -*- coding: utf-8 -*-
"""按运行模式校正宿主 / 验证目录。"""

from __future__ import annotations

import argparse
import sys

from app.constants import DEFAULT_ACCURACY_IMAGE_DIR, DEFAULT_EMBED_IMAGE_DIR
from cli import log_ok, log_warn


def apply_mode_image_dirs(config: argparse.Namespace) -> argparse.Namespace:
    """
    - ``test_embedding`` → 默认 ``Datasets/images/``
    - ``test_accuracy`` → 默认 ``Datasets/Recover/capture/``
    - 命令行已传 ``--image_dir`` / ``--image_val_dir`` 时不覆盖
    """
    if config.mode == "test_embedding":
        if "--image_dir" not in sys.argv:
            config.image_dir = DEFAULT_EMBED_IMAGE_DIR
            log_ok(f"嵌入宿主目录 → {config.image_dir}")
        if "--image_val_dir" not in sys.argv:
            config.image_val_dir = config.image_dir
        tip = config.image_dir.replace("\\", "/").lower()
        if "cocomask" in tip:
            log_warn(
                "当前 image_dir 像是 COCOMask 拼接图目录；"
                "嵌入应使用左半边裁好的编号图（如 Datasets/images/）。"
                "可运行: python tools/export_coco_hosts.py"
            )
    elif config.mode == "test_accuracy":
        if "--image_dir" not in sys.argv:
            config.image_dir = DEFAULT_ACCURACY_IMAGE_DIR
            log_ok(f"测准图像目录 → {config.image_dir}")
        if "--image_val_dir" not in sys.argv:
            config.image_val_dir = config.image_dir
    return config
