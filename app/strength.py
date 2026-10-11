# -*- coding: utf-8 -*-
"""评估 / 嵌入 / 测准时的 embed_strength 校正。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.constants import EVAL_STRENGTH_OFFICIAL, EVAL_STRENGTH_TRAIN_DEFAULT
from ckpt_utils import (
    ckpt_label,
    ckpt_short,
    normalize_ckpt_tag,
    read_embed_strength_meta,
    resolve_ckpt_path,
)
from cli import log_info, log_ok, log_warn


def apply_eval_embed_strength(config: argparse.Namespace) -> argparse.Namespace:
    """
    按权重选择嵌入方式，避免 strength 错配导致 PSNR/Acc 虚低。

    - 已设 ``weight_path`` → 读旁路 ``*.meta.txt``，默认 0
    - mask_99 → embed_strength=0
    - ss_best / best → 优先读对应 meta，否则 0
    - 命令行已传 ``--embed_strength`` → 不改
    """
    if getattr(config, "mode", None) not in (
        "eval_mask",
        "test_embedding",
        "test_accuracy",
    ):
        return config
    if "--embed_strength" in sys.argv:
        log_info(f"使用命令行 embed_strength={config.embed_strength}")
        return config

    wpath = getattr(config, "weight_path", None)
    if wpath:
        p = Path(wpath)
        strength = read_embed_strength_meta(p, default=0.0)
        config.embed_strength = float(strength)
        config.eval_ckpt = "custom"
        log_ok(
            f"weight_path → embed_strength={config.embed_strength} （meta ← {p.name}）"
        )
        return config

    ckpt = normalize_ckpt_tag(getattr(config, "eval_ckpt", "99"))
    if ckpt == "custom":
        config.embed_strength = float(getattr(config, "embed_strength", 0.0) or 0.0)
        return config
    config.eval_ckpt = ckpt
    if ckpt in ("ss_best", "best"):
        strength = float(EVAL_STRENGTH_TRAIN_DEFAULT)
        try:
            resolved = resolve_ckpt_path(
                config.model_save_dir,
                config.model_name,
                ckpt,
                distortion=getattr(config, "distortion", None),
            )
            strength = read_embed_strength_meta(resolved, default=strength)
            log_info(f"从 meta 读取 embed_strength={strength} ← {resolved}")
        except FileNotFoundError:
            log_warn(
                f"尚未找到 {ckpt_label(ckpt)} 权重文件，embed_strength 暂用 {strength}"
            )
        config.embed_strength = strength
        log_ok(
            f"eval_ckpt={ckpt} ({ckpt_short(ckpt)}) → "
            f"embed_strength={config.embed_strength}"
        )
    else:
        config.embed_strength = float(EVAL_STRENGTH_OFFICIAL)
        log_ok(
            f"eval_ckpt={ckpt} → embed_strength={config.embed_strength} "
            f"（官方整图输出）"
        )
    return config
