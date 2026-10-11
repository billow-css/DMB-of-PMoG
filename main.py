#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
DMB of PMoG 命令行入口（Dynamic Motion Blur of PMoG）。

交互功能面板与训练 / 评估逻辑已拆到 ``cli/`` 与 ``app/``；
本文件仅保留薄入口，兼容 ``python main.py``。

常用示例
--------
python main.py
python main.py --mode eval_mask --no_interactive --eval_ckpt best --distortion ScreenShootingMB
python main.py --mode tool_verify_psnr --no_interactive
"""

from __future__ import annotations

from app.entry import cli_main
from app.parser import build_parser
from app.run import main
from app.tools import run_tool_mode

# 兼容旧 ``from main import …``
from app.constants import REPO_ROOT, TOOL_MODE_CHOICES  # noqa: F401
from app.paths import apply_mode_image_dirs  # noqa: F401
from app.resolve_mode import resolve_run_mode  # noqa: F401
from app.resolve_scale import (  # noqa: F401
    apply_lite_defaults,
    apply_scale_defaults,
    resolve_train_scale,
)
from app.strength import apply_eval_embed_strength  # noqa: F401

__all__ = [
    "REPO_ROOT",
    "TOOL_MODE_CHOICES",
    "apply_eval_embed_strength",
    "apply_lite_defaults",
    "apply_mode_image_dirs",
    "apply_scale_defaults",
    "build_parser",
    "cli_main",
    "main",
    "resolve_run_mode",
    "resolve_train_scale",
    "run_tool_mode",
]

if __name__ == "__main__":
    cli_main()
