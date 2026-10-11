# -*- coding: utf-8 -*-
"""权重标签 / 噪声层 / 训练初始化 / 规模菜单。"""

from __future__ import annotations

import sys

from cli.color import _C, _paint
from cli.log import log_err, log_ok, log_warn


def prompt_eval_ckpt(default_choice: str = "1") -> dict:
    """评估 / 嵌入 / 测准共用的权重标签选择（兼容旧交互）。"""
    menu = (
        "\n"
        + _paint("  权重选择（评估 / 嵌入 / 测准）", _C.BOLD, _C.WHITE)
        + "\n"
        + _paint("  ────────────────────────────────────────────", _C.DIM)
        + "\n"
        + f"    {_paint('1', _C.GREEN, _C.BOLD)}  mask_99                 "
        + _paint("…/Encoder_Decoder_Model_mask_99.pth", _C.DIM)
        + "\n"
        + f"    {_paint('2', _C.CYAN, _C.BOLD)}  ScreenShooting_best     "
        + _paint("models/…_mask_ScreenShooting_best.pth", _C.DIM)
        + "\n"
        + f"    {_paint('3', _C.MAGENTA, _C.BOLD)}  ScreenShootingMB_best   "
        + _paint("models/…_mask_ScreenShootingMB_best.pth (DMB)", _C.DIM)
        + "\n"
        + _paint("  ────────────────────────────────────────────", _C.DIM)
        + "\n"
        + _paint(
            "  1→strength=0；2/3→读对应 .meta.txt（默认 strength=0）",
            _C.DIM,
        )
        + "\n"
    )
    print(menu, flush=True)

    while True:
        try:
            tip = _paint(f"请输入 1 / 2 / 3 [默认 {default_choice}]: ", _C.CYAN)
            raw = input(tip).strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            log_warn(f"已取消输入，使用默认选项 {default_choice}")
            raw = default_choice.strip()
            break
        if raw == "":
            raw = default_choice.strip()
        if raw in ("1", "2", "3"):
            break
        log_err("无效输入，请输入 1 / 2 / 3")

    if raw == "3":
        log_ok("权重：ScreenShootingMB_best (DMB)")
        return {"eval_ckpt": "best", "embedding_epoch": 99, "embed_strength": None}
    if raw == "2":
        log_ok("权重：ScreenShooting_best")
        return {"eval_ckpt": "ss_best", "embedding_epoch": 99, "embed_strength": None}
    log_ok("权重：Encoder_Decoder_Model_mask_99.pth")
    return {"eval_ckpt": "99", "embedding_epoch": 99, "embed_strength": 0.0}


def prompt_noise_layer(default_choice: str = "1") -> str:
    """交互选择噪声层，返回 ScreenShooting 或 ScreenShootingMB。"""
    menu = (
        "\n"
        + _paint("  噪声层选择（训练 / 评估 / 嵌入 / 测准）", _C.BOLD, _C.WHITE)
        + "\n"
        + _paint("  ────────────────────────────────────────────", _C.DIM)
        + "\n"
        + f"    {_paint('1', _C.GREEN, _C.BOLD)}  ScreenShooting     "
        + _paint("基线：透视+光照+摩尔纹+高斯（原论文族）", _C.DIM)
        + "\n"
        + f"    {_paint('2', _C.MAGENTA, _C.BOLD)}  ScreenShootingMB   "
        + _paint("DMB：动态运动模糊 + 摩尔纹反比", _C.DIM)
        + "\n"
        + _paint("  ────────────────────────────────────────────", _C.DIM)
        + "\n"
        + _paint("  评估时：优先 models/<噪声层>/；若无权重则回退其它目录的同 epoch 文件", _C.DIM)
        + "\n"
        + _paint("  （Encoder/Decoder 共用；Noiser 仍用你选的噪声层跑 Acc）", _C.DIM)
        + "\n"
        + _paint("  反比：α_moire = 0.15·(1−β),  β∝轨迹长度均值", _C.DIM)
        + "\n"
    )
    print(menu, flush=True)

    can_ask = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    if not can_ask:
        raw = default_choice.strip()
        log_warn(f"非交互终端，噪声层默认选项 {raw}")
    else:
        while True:
            try:
                tip = _paint(f"请输入 1 或 2 [默认 {default_choice}]: ", _C.CYAN)
                raw = input(tip).strip()
            except (EOFError, KeyboardInterrupt):
                print(flush=True)
                raw = default_choice.strip()
                break
            if raw == "":
                raw = default_choice.strip()
            if raw in ("1", "2"):
                break
            log_err("无效输入，请输入 1（基线）或 2（DMB）")

    if raw == "2":
        log_ok("噪声层：ScreenShootingMB（DMB · 动态运动模糊 + 摩尔纹反比）")
        return "ScreenShootingMB"
    log_ok("噪声层：ScreenShooting（基线）")
    return "ScreenShooting"
