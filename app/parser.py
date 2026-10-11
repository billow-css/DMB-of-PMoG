# -*- coding: utf-8 -*-
"""命令行解析器组装。"""

from __future__ import annotations

import argparse

from app.args_output import add_lite_args, add_output_args, add_path_args
from app.args_train import add_model_args, add_train_args
from cli import PROJECT_FULL, PROJECT_SHORT

_EPILOG = """
示例:
  # 交互功能面板（推荐）
  python main.py

  # 跳过训练直接评估 mask_99
  python main.py --mode eval_mask --no_interactive --eval_ckpt 99 --embed_strength 0

  # 评估 ScreenShooting_best
  python main.py --mode eval_mask --no_interactive --eval_ckpt ss_best --distortion ScreenShooting

  # 评估 DMB_best (ScreenShootingMB)
  python main.py --mode eval_mask --no_interactive --eval_ckpt best --distortion ScreenShootingMB

  # Lite / 全量训练
  python main.py --mode train_mask --lite --distortion ScreenShootingMB
  python main.py --mode train_mask --full

  # 工具：嵌入 PSNR / SSIM
  python main.py --mode tool_verify_psnr --no_interactive
  python main.py --mode tool_verify_both --no_interactive
""".strip()


def build_parser() -> argparse.ArgumentParser:
    """构建分组清晰、带 epilog 示例的命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            f"{PROJECT_SHORT} — {PROJECT_FULL} · "
            "screen-shooting robust watermark training & evaluation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    add_model_args(parser)
    add_train_args(parser)
    add_lite_args(parser)

    g_test = parser.add_argument_group("测试")
    g_test.add_argument(
        "--test_iters", type=int, default=99, help="测试步（保留兼容，默认: 99）"
    )

    add_output_args(parser)
    add_path_args(parser)
    return parser
