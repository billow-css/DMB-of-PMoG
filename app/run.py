# -*- coding: utf-8 -*-
"""组装 DataLoader / Solver 并按 mode 分发。"""

from __future__ import annotations

import argparse
import os
import sys

from app.paths import apply_mode_image_dirs
from app.strength import apply_eval_embed_strength
from ckpt_utils import (
    ckpt_label,
    named_best_filename,
    normalize_ckpt_tag,
    resolve_ckpt_path,
)
from cli import (
    log_err,
    log_info,
    log_ok,
    log_warn,
    print_config_summary,
    print_mode_badge,
)


def _ensure_dirs(config: argparse.Namespace) -> None:
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)


def _prepare_eval_ckpt(config: argparse.Namespace) -> None:
    if config.mode == "eval_mask":
        config.dataset = "eval_mask"
        config.image_dir = config.image_val_dir
    if getattr(config, "weight_path", None):
        config.eval_ckpt = "custom"
        if config.embedding_epoch == 0:
            config.embedding_epoch = 99
        log_info(f"{config.mode}：使用自定义权重 {config.weight_path}")
        return
    if getattr(config, "eval_ckpt", None) is None:
        config.eval_ckpt = "99"
    if str(config.eval_ckpt) != "custom":
        config.eval_ckpt = normalize_ckpt_tag(config.eval_ckpt)
    if config.eval_ckpt == "99" and config.embedding_epoch == 0:
        config.embedding_epoch = 99
        log_info(f"{config.mode}：未指定 epoch → 使用 mask_99")
    elif config.eval_ckpt in ("ss_best", "best"):
        log_info(
            f"{config.mode}：使用 {ckpt_label(config.eval_ckpt)} "
            f"({named_best_filename(config.model_name, config.eval_ckpt)})"
        )


def _weight_hint(config: argparse.Namespace) -> str:
    if getattr(config, "weight_path", None):
        return str(config.weight_path)
    try:
        return resolve_ckpt_path(
            config.model_save_dir,
            config.model_name,
            normalize_ckpt_tag(getattr(config, "eval_ckpt", "99")),
            embedding_epoch=int(config.embedding_epoch or 99),
            distortion=config.distortion,
        )
    except FileNotFoundError:
        return named_best_filename(
            config.model_name, normalize_ckpt_tag(getattr(config, "eval_ckpt", "99"))
        )


def _resolve_summary_weight(config: argparse.Namespace) -> str:
    if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        if getattr(config, "weight_path", None):
            return str(config.weight_path)
        try:
            return resolve_ckpt_path(
                config.model_save_dir,
                config.model_name,
                normalize_ckpt_tag(getattr(config, "eval_ckpt", "99")),
                embedding_epoch=int(config.embedding_epoch or 99),
                distortion=config.distortion,
            )
        except FileNotFoundError as e:
            return str(e).split("\n")[0]
    return os.path.join(
        config.model_save_dir,
        config.distortion,
        f"{config.model_name}_mask_{config.embedding_epoch}.pth",
    )


def main(config: argparse.Namespace) -> None:
    """组装 DataLoader 与 Solver，按 mode 分发。"""
    try:
        import torch
        from torch.backends import cudnn

        from data_loader import get_loader
        from solver import Solver
    except ImportError as e:
        log_err(f"缺少依赖: {e}")
        log_err("请先执行: pip install -r requirements.txt")
        sys.exit(1)

    cudnn.benchmark = True
    _ensure_dirs(config)
    show_progress = not config.no_progress
    config.show_progress = show_progress

    if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        _prepare_eval_ckpt(config)
        apply_eval_embed_strength(config)
    apply_mode_image_dirs(config)

    common_kw = dict(
        image_size=config.image_size,
        batch_size=config.batch_size,
        dataset=config.dataset,
        mode=config.mode,
        num_workers=config.num_workers,
        w_path=config.wmat_dir,
        lite=config.lite,
        lite_ratio=config.lite_ratio,
        lite_seed=config.lite_seed,
    )
    try:
        data_loader = get_loader(config.image_dir, **common_kw)
        data_loader_test = get_loader(config.image_val_dir, **common_kw)
    except FileNotFoundError as e:
        log_err(str(e))
        if config.mode == "test_embedding":
            log_err(
                "嵌入请将编号宿主图放到 Datasets/images/ "
                "（或传 --image_dir）。可用: python tools/export_coco_hosts.py "
                "或主菜单选项 5"
            )
        elif config.mode == "test_accuracy":
            log_err("测准请确认 --image_dir 指向矫正后的拍屏图目录")
        else:
            log_err("请确认 --image_dir / --image_val_dir 指向有效目录")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extra = {
        "device": device,
        "train_samples": len(data_loader.dataset),
        "val_samples": len(data_loader_test.dataset),
        "show_progress": show_progress,
        "weight_path": _resolve_summary_weight(config),
        "eval_ckpt": getattr(config, "eval_ckpt", None),
        "init_from_ckpt": getattr(config, "init_from_ckpt", None),
    }
    if device == "cuda":
        try:
            extra["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    if not config.quiet:
        print_mode_badge(config.mode)
        print_config_summary(config, extra=extra)
        if device != "cuda":
            log_warn("当前为 CPU 训练。若本机有 NVIDIA，请用 CUDA 版 PyTorch 环境启动。")

    solver = Solver(data_loader, data_loader_test, config)
    if config.mode == "train_mask":
        log_ok("开始训练 (train_mask)" + (" [LITE]" if config.lite else ""))
        solver.train_mask()
    elif config.mode == "eval_mask":
        log_ok(f"跳过训练，直接评估 (eval_mask) | {_weight_hint(config)}")
        solver.eval_mask()
    elif config.mode == "test_accuracy":
        log_ok(f"开始准确率测试 (test_accuracy) | {_weight_hint(config)}")
        solver.test_accuracy()
    elif config.mode == "test_embedding":
        log_ok(f"开始水印嵌入 (test_embedding) | {_weight_hint(config)}")
        solver.test_embedding()
    else:
        log_err(f"未知 mode: {config.mode}")
        sys.exit(1)
    log_ok("全部完成。")
