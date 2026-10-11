# -*- coding: utf-8 -*-
"""运行模式解析：命令行 / 非交互 / 功能面板。"""

from __future__ import annotations

import argparse
import sys

from app.resolve_explicit import apply_explicit_mode
from app.strength import apply_eval_embed_strength
from cli import log_info, prompt_run_mode


def resolve_run_mode(config: argparse.Namespace) -> argparse.Namespace:
    """
    决定运行模式。

    1. 命令行已给 ``--mode`` → 直接使用
    2. ``--no_interactive`` → 默认 ``eval_mask``
    3. 交互功能面板
    """
    if config.mode is not None:
        return apply_explicit_mode(config)

    if config.no_interactive:
        config.mode = "eval_mask"
        config.dataset = "eval_mask"
        if config.eval_ckpt is None:
            config.eval_ckpt = "99"
        if config.embedding_epoch == 0:
            config.embedding_epoch = 99
        if config.distortion is None:
            config.distortion = "ScreenShooting"
        apply_eval_embed_strength(config)
        log_info(
            f"--no_interactive 且未指定 --mode → 默认 eval_mask "
            f"(ckpt={config.eval_ckpt}, strength={config.embed_strength})"
        )
        return config

    chosen = prompt_run_mode(default_choice="0")
    config.mode = chosen["mode"]
    if (
        config.mode == "quit"
        or config.mode == "log_query"
        or str(config.mode).startswith("tool_")
    ):
        return config

    config.dataset = chosen["dataset"]
    if "distortion" in chosen:
        config.distortion = chosen["distortion"]
    elif config.distortion is None:
        config.distortion = "ScreenShooting"
    if "init_from_epoch" in chosen:
        config.init_from_epoch = chosen["init_from_epoch"]
    if "init_from_ckpt" in chosen:
        config.init_from_ckpt = chosen["init_from_ckpt"]
    if "lite" in chosen:
        config.lite = bool(chosen["lite"])
        if chosen["mode"] in ("train_mask", "eval_mask"):
            config._scale_from_menu = True
    if "eval_ckpt" in chosen:
        config.eval_ckpt = chosen["eval_ckpt"]
    elif config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        if config.eval_ckpt is None and not getattr(config, "weight_path", None):
            config.eval_ckpt = "99"
    if "weight_path" in chosen:
        config.weight_path = chosen["weight_path"]
        config.eval_ckpt = "custom"
    if "embedding_epoch" in chosen:
        config.embedding_epoch = int(chosen["embedding_epoch"])
    if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        if (
            "embed_strength" in chosen
            and chosen["embed_strength"] is not None
            and "--embed_strength" not in sys.argv
        ):
            config.embed_strength = float(chosen["embed_strength"])
        apply_eval_embed_strength(config)
    return config
