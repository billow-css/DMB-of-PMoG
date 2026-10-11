# -*- coding: utf-8 -*-
"""显式 --mode 时的权重 / 噪声层 / 初始化补齐。"""

from __future__ import annotations

import argparse
import sys

from app.constants import REPO_ROOT
from app.strength import apply_eval_embed_strength
from ckpt_utils import (
    ckpt_default_distortion,
    default_noise_menu_choice,
    normalize_ckpt_tag,
)
from cli import log_info, log_ok, log_warn, prompt_noise_layer, prompt_train_init


def _pick_weight_interactive(config: argparse.Namespace) -> bool:
    """弹窗选权重；取消则返回 False 并把 mode 设为 quit。"""
    from ui.verify_weights_ui import (
        infer_noise_menu_choice_from_path,
        prompt_single_weight_file,
    )

    titles = {
        "eval_mask": "评估模式 — 选择权重",
        "test_embedding": "水印叠加 — 选择权重",
        "test_accuracy": "拍屏测准 — 选择权重",
    }
    picked = prompt_single_weight_file(
        title=titles.get(config.mode, "选择权重"),
        repo_root=REPO_ROOT,
    )
    if picked is None:
        log_warn("已取消权重选择。")
        config.mode = "quit"
        return False
    config.weight_path = str(picked)
    config.eval_ckpt = "custom"
    if config.embedding_epoch == 0:
        config.embedding_epoch = 99
    log_ok(f"权重文件：{picked}")
    if config.distortion is None and "--distortion" not in sys.argv:
        config.distortion = prompt_noise_layer(
            default_choice=infer_noise_menu_choice_from_path(picked)
        )
    return True


def _ensure_eval_ckpt(config: argparse.Namespace) -> None:
    if getattr(config, "weight_path", None):
        config.eval_ckpt = "custom"
        if config.embedding_epoch == 0:
            config.embedding_epoch = 99
        return
    if config.eval_ckpt is None:
        config.eval_ckpt = "99"
        if config.embedding_epoch == 0:
            config.embedding_epoch = 99
        return
    if str(config.eval_ckpt) != "custom":
        config.eval_ckpt = normalize_ckpt_tag(config.eval_ckpt)
    if config.eval_ckpt == "99" and config.embedding_epoch == 0:
        config.embedding_epoch = 99


def _ensure_eval_distortion(config: argparse.Namespace) -> None:
    if config.distortion is not None:
        return
    if getattr(config, "weight_path", None):
        from ui.verify_weights_ui import infer_noise_menu_choice_from_path

        if config.no_interactive:
            choice = infer_noise_menu_choice_from_path(config.weight_path)
            config.distortion = (
                "ScreenShootingMB" if choice == "2" else "ScreenShooting"
            )
            log_info(f"未指定 --distortion → 默认 {config.distortion}")
        else:
            config.distortion = prompt_noise_layer(
                default_choice=infer_noise_menu_choice_from_path(config.weight_path)
            )
        return
    if config.no_interactive:
        config.distortion = ckpt_default_distortion(config.eval_ckpt)
        log_info(f"未指定 --distortion → 默认 {config.distortion}")
    else:
        config.distortion = prompt_noise_layer(
            default_choice=default_noise_menu_choice(config.eval_ckpt)
        )


def apply_explicit_mode(config: argparse.Namespace) -> argparse.Namespace:
    """处理命令行已给 ``--mode`` 的情况。"""
    if (
        config.mode == "quit"
        or config.mode == "log_query"
        or str(config.mode).startswith("tool_")
    ):
        return config

    if config.dataset == "train_mask" and config.mode != "train_mask":
        if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
            config.dataset = config.mode

    if config.distortion is None:
        if config.mode == "train_mask":
            if config.no_interactive:
                config.distortion = "ScreenShootingMB"
                log_info("未指定 --distortion → 默认 ScreenShootingMB (DMB)")
            else:
                config.distortion = prompt_noise_layer(default_choice="2")
        elif config.mode not in ("eval_mask", "test_embedding", "test_accuracy"):
            config.distortion = "ScreenShooting"

    eval_modes = ("eval_mask", "test_embedding", "test_accuracy")
    if config.mode in eval_modes:
        needs_pick = (
            not getattr(config, "weight_path", None)
            and "--weight_path" not in sys.argv
            and config.eval_ckpt is None
            and not config.no_interactive
            and "--eval_ckpt" not in sys.argv
        )
        if needs_pick and not _pick_weight_interactive(config):
            return config
        _ensure_eval_ckpt(config)
        _ensure_eval_distortion(config)
        apply_eval_embed_strength(config)

    if (
        config.mode == "train_mask"
        and getattr(config, "init_from_ckpt", None) is None
        and config.init_from_epoch is None
        and config.embedding_epoch == 0
        and not config.no_interactive
        and "--init_from_epoch" not in sys.argv
        and "--init_from_ckpt" not in sys.argv
    ):
        init_cfg = prompt_train_init(default_choice="1")
        config.init_from_ckpt = init_cfg.get("init_from_ckpt")
        config.init_from_epoch = init_cfg.get("init_from_epoch")
        if "embedding_epoch" in init_cfg:
            config.embedding_epoch = int(init_cfg["embedding_epoch"])
    elif config.mode == "train_mask" and getattr(config, "init_from_ckpt", None):
        config.init_from_ckpt = normalize_ckpt_tag(config.init_from_ckpt)
    return config
