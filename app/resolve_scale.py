# -*- coding: utf-8 -*-
"""全量 / Lite 规模解析与默认超参。"""

from __future__ import annotations

import argparse
import sys

from cli import log_err, log_info, log_warn, prompt_train_scale


def resolve_train_scale(config: argparse.Namespace) -> argparse.Namespace:
    """
    决定全量 / Lite。

    1. ``--lite`` / ``--full``
    2. 功能菜单已选（``_scale_from_menu``）
    3. ``--no_interactive`` → 默认全量
    4. ``train_mask`` → 交互询问
    """
    if config.lite and config.full:
        log_err("--lite 与 --full 不能同时使用")
        sys.exit(2)
    if config.lite:
        log_info("命令行指定 --lite，跳过规模菜单")
        return config
    if config.full:
        config.lite = False
        log_info("命令行指定 --full，跳过规模菜单 → 全量模式")
        return config
    if getattr(config, "_scale_from_menu", False):
        log_info("规模已由功能面板决定 → " + ("Lite" if config.lite else "全量"))
        return config
    if config.no_interactive:
        if config.mode != "eval_mask":
            config.lite = False
            log_info("--no_interactive：未指定规模，默认全量模式")
        return config
    if config.mode == "train_mask":
        config.lite = prompt_train_scale(default_choice="2")
    elif not config.quiet:
        log_info(f"当前 mode={config.mode}，跳过训练规模菜单")
    return config


def apply_scale_defaults(config: argparse.Namespace) -> argparse.Namespace:
    """warmup_epochs：全量默认 0，Lite 默认 5（命令行显式传入则不改）。"""
    if config.warmup_epochs is None:
        config.warmup_epochs = 5 if config.lite else 0
        log_info(
            f"warmup_epochs → {config.warmup_epochs} "
            f"({'Lite' if config.lite else '全量'} 默认)"
        )
    return config


def apply_lite_defaults(config: argparse.Namespace) -> argparse.Namespace:
    """Lite 模式下自动下调适合小批量的超参。"""
    if not config.lite:
        return config
    if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        return config
    if config.lite_ratio < 0.10 or config.lite_ratio > 0.20:
        log_warn(
            f"lite_ratio={config.lite_ratio} 超出推荐区间 [0.10, 0.20]，"
            "仍将按给定值抽样"
        )
    if config.lite_auto_tune:
        if config.batch_size > 4:
            log_info(f"Lite auto-tune: batch_size {config.batch_size} → 8")
            config.batch_size = 4
        if config.num_epoch < 40:
            log_info(f"Lite auto-tune: num_epoch {config.num_epoch} → 40")
            config.num_epoch = 40
        elif config.num_epoch > 60:
            log_info(f"Lite auto-tune: num_epoch {config.num_epoch} → 40")
            config.num_epoch = 40
        if config.warmup_epochs >= config.num_epoch:
            config.warmup_epochs = max(1, config.num_epoch // 5)
            log_info(f"Lite auto-tune: warmup_epochs → {config.warmup_epochs}")
    return config
