# -*- coding: utf-8 -*-
"""项目品牌横幅与模式徽章。"""

from __future__ import annotations

from cli.color import _C, _paint

PROJECT_SHORT = "DMB of PMoG"
PROJECT_FULL = "Dynamic Motion Blur of PMoG"
PROJECT_TAGLINE = "Screen-shooting watermark · Dynamic motion-blur noise layer"

_BANNER = r"""
 ____  __  __ ____      ____  __  __      ____
|  _ \|  \/  | __ )    |  _ \|  \/  |___ / ___|
| | | | |\/| |  _ \ ___| |_) | |\/| / _ \ |  _
| |_| | |  | | |_) |___|  __/| |  | | (_) | |_| |
|____/|_|  |_|____/    |_|   |_|  |_|\___/ \____|
"""

_MODE_LABELS = {
    "eval_mask": (_C.MAGENTA, "EVAL", "弹窗选权重 · 验证集 Acc/BER/PSNR"),
    "train_mask": (_C.GREEN, "TRAIN", "Encoder–Noise–Decoder 端到端训练"),
    "test_embedding": (_C.CYAN, "EMBED", "预训练 Encoder 嵌入水印并导出拼图"),
    "test_accuracy": (_C.BLUE, "ACCURACY", "拍屏矫正图 Acc/BER + 相对含水印宿主 PSNR"),
    "tool_export_hosts": (_C.YELLOW, "TOOL", "从 COCOMask 导出宿主 → Datasets/images"),
    "tool_verify_psnr": (_C.YELLOW, "TOOL", "宿主嵌入 PSNR：弹窗选 1~2 个权重"),
    "tool_verify_ssim": (_C.YELLOW, "TOOL", "宿主嵌入 SSIM：弹窗选 1~2 个权重"),
    "tool_verify_both": (_C.YELLOW, "TOOL", "宿主嵌入 PSNR+SSIM：弹窗选权重"),
    "tool_crop_panels": (_C.YELLOW, "TOOL", "从结果拼图裁切嵌入面板"),
    "tool_sim_test": (_C.YELLOW, "TOOL", "仿真拍屏 sim_test：文件夹叠 PIMoG/DMB 噪声"),
    "tool_batch_rectify": (_C.BLUE, "EXP", "批量透视矫正（experiment/batch_rectify）"),
    "tool_rectify_gui": (_C.BLUE, "EXP", "交互透视矫正 GUI"),
    "tool_server_gui": (_C.BLUE, "EXP", "联机拍屏服务端 GUI"),
    "log_query": (_C.CYAN, "LOGS", "查询 / 调出训练 JSON 历史"),
}


def print_banner(lite: bool = False) -> None:
    """打印启动横幅；Lite 模式会追加标记。"""
    print(_paint(_BANNER, _C.CYAN, _C.BOLD), flush=True)
    print(
        _paint(f"  {PROJECT_SHORT}", _C.BOLD, _C.WHITE)
        + _paint(f"  ·  {PROJECT_FULL}", _C.DIM),
        flush=True,
    )
    print(_paint(f"  {PROJECT_TAGLINE}", _C.DIM), flush=True)
    print(
        _paint(
            "  (PMoG = PIMoG without illumination-I · DMB = Dynamic Motion Blur)",
            _C.DIM,
        ),
        flush=True,
    )
    if lite:
        print_lite_badge()
    else:
        print(_paint("─" * 64, _C.DIM), flush=True)


def print_lite_badge() -> None:
    """单独打印 Lite 模式标记行。"""
    print(
        _paint("  ★ LITE MODE", _C.YELLOW, _C.BOLD)
        + _paint("  — 使用 COCOMask 子集做小批量快速试跑", _C.DIM),
        flush=True,
    )
    print(_paint("─" * 56, _C.DIM), flush=True)


def print_mode_badge(mode: str) -> None:
    """按运行模式打印彩色标签。"""
    color, tag, desc = _MODE_LABELS.get(mode, (_C.WHITE, mode.upper(), ""))
    print(
        _paint(f"  ▶ {tag}", color, _C.BOLD) + _paint(f"  — {desc}", _C.DIM),
        flush=True,
    )
    print(_paint("─" * 64, _C.DIM), flush=True)
