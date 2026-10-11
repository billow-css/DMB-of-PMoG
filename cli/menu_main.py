# -*- coding: utf-8 -*-
"""主功能面板菜单文案与选项映射。"""

from __future__ import annotations

from cli.brand import PROJECT_FULL, PROJECT_SHORT
from cli.color import _C, _paint

MODE_MAPPING = {
    "0": {
        "mode": "eval_mask",
        "dataset": "eval_mask",
        "lite": False,
        "label": "0 — 评估模式",
    },
    "1": {
        "mode": "train_mask",
        "dataset": "train_mask",
        "lite": False,
        "label": "1 — 全量训练",
    },
    "2": {
        "mode": "train_mask",
        "dataset": "train_mask",
        "lite": True,
        "label": "2 — Lite 训练",
    },
    "3": {
        "mode": "test_embedding",
        "dataset": "test_embedding",
        "lite": False,
        "label": "3 — 水印嵌入",
    },
    "4": {
        "mode": "test_accuracy",
        "dataset": "test_accuracy",
        "lite": False,
        "label": "4 — 拍屏测准",
    },
    "5": {"mode": "tool_export_hosts", "label": "5 — 导出宿主"},
    "6": {"mode": "tool_verify_psnr", "label": "6 — 嵌入 PSNR"},
    "7": {"mode": "tool_verify_ssim", "label": "7 — 嵌入 SSIM"},
    "8": {"mode": "tool_verify_both", "label": "8 — 嵌入质量 (PSNR+SSIM)"},
    "9": {"mode": "tool_crop_panels", "label": "9 — 裁切面板"},
    "s": {"mode": "tool_sim_test", "label": "s — 仿真拍屏 (sim_test)"},
    "a": {"mode": "tool_batch_rectify", "label": "a — 批量矫正"},
    "b": {"mode": "tool_rectify_gui", "label": "b — 矫正 GUI"},
    "c": {"mode": "tool_server_gui", "label": "c — 服务端 GUI"},
    "l": {"mode": "log_query", "label": "l — 训练日志查询"},
    "q": {"mode": "quit", "label": "q — 退出"},
}

VALID_CHOICES = set(MODE_MAPPING.keys())


def build_main_menu() -> str:
    """组装主功能面板文案。"""
    return (
        "\n"
        + _paint(f"  {PROJECT_SHORT} 功能面板", _C.BOLD, _C.WHITE)
        + "\n"
        + _paint(f"  {PROJECT_FULL}", _C.DIM)
        + "\n"
        + _paint("  ────────────────────────────────────────────────", _C.DIM)
        + "\n"
        + _paint("  【训练 / 评估】", _C.BOLD, _C.WHITE)
        + "\n"
        + f"    {_paint('0', _C.MAGENTA, _C.BOLD)}  评估模式    "
        + _paint("弹窗选 .pth · Acc/BER/PSNR", _C.DIM)
        + "\n"
        + f"    {_paint('1', _C.GREEN, _C.BOLD)}  全量训练    "
        + _paint("完整 COCOMask 端到端", _C.DIM)
        + "\n"
        + f"    {_paint('2', _C.YELLOW, _C.BOLD)}  Lite 训练   "
        + _paint("约 10%~20% 子集试跑", _C.DIM)
        + "\n"
        + f"    {_paint('3', _C.CYAN, _C.BOLD)}  水印嵌入    "
        + _paint("弹窗选 .pth · 可视化", _C.DIM)
        + "\n"
        + f"    {_paint('4', _C.BLUE, _C.BOLD)}  拍屏测准    "
        + _paint("弹窗选 .pth · Acc/BER/PSNR", _C.DIM)
        + "\n"
        + _paint("  【工具】", _C.BOLD, _C.WHITE)
        + "\n"
        + f"    {_paint('5', _C.YELLOW, _C.BOLD)}  导出宿主    "
        + _paint("COCOMask → Datasets/images", _C.DIM)
        + "\n"
        + f"    {_paint('6', _C.YELLOW, _C.BOLD)}  嵌入 PSNR   "
        + _paint("弹窗选 1~2 个 .pth → 对照 / 单测", _C.DIM)
        + "\n"
        + f"    {_paint('7', _C.YELLOW, _C.BOLD)}  嵌入 SSIM   "
        + _paint("弹窗选 1~2 个 .pth → 对照 / 单测", _C.DIM)
        + "\n"
        + f"    {_paint('8', _C.YELLOW, _C.BOLD)}  嵌入质量    "
        + _paint("PSNR + SSIM（同样弹窗选权重）", _C.DIM)
        + "\n"
        + f"    {_paint('9', _C.YELLOW, _C.BOLD)}  裁切面板    "
        + _paint("结果拼图 → 嵌入图目录", _C.DIM)
        + "\n"
        + f"    {_paint('s', _C.YELLOW, _C.BOLD)}  仿真拍屏    "
        + _paint("sim_test · 文件夹叠 PIMoG/DMB 噪声", _C.DIM)
        + "\n"
        + _paint("  【拍屏实验】", _C.BOLD, _C.WHITE)
        + "\n"
        + f"    {_paint('a', _C.BLUE, _C.BOLD)}  批量矫正    "
        + _paint("experiment/batch_rectify.py", _C.DIM)
        + "\n"
        + f"    {_paint('b', _C.BLUE, _C.BOLD)}  矫正 GUI    "
        + _paint("experiment/rectify_gui.py", _C.DIM)
        + "\n"
        + f"    {_paint('c', _C.BLUE, _C.BOLD)}  服务端 GUI  "
        + _paint("experiment/server_gui.py", _C.DIM)
        + "\n"
        + f"    {_paint('l', _C.CYAN, _C.BOLD)}  训练日志    "
        + _paint("查询 / 调出 JSON 历史（永不覆盖）", _C.DIM)
        + "\n"
        + f"    {_paint('q', _C.RED, _C.BOLD)}  退出"
        + "\n"
        + _paint("  ────────────────────────────────────────────────", _C.DIM)
        + "\n"
    )
