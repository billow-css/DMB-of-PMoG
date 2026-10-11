# -*- coding: utf-8 -*-
"""训练初始化与规模菜单。"""

from __future__ import annotations

import sys

from cli.color import _C, _paint
from cli.log import log_err, log_ok, log_warn


def prompt_train_init(default_choice: str = "1") -> dict:
    """训练初始化：从头，或挂载 mask_99 / SS_best / DMB_best。"""
    menu = (
        "\n"
        + _paint("  训练初始化", _C.BOLD, _C.WHITE)
        + "\n"
        + _paint("  ────────────────────────────────────────────", _C.DIM)
        + "\n"
        + f"    {_paint('1', _C.GREEN, _C.BOLD)}  从头训练              "
        + _paint("随机初始化 Encoder/Decoder", _C.DIM)
        + "\n"
        + f"    {_paint('2', _C.CYAN, _C.BOLD)}  挂载 mask_99           "
        + _paint("拷贝 mask_99.pth 再训", _C.DIM)
        + "\n"
        + f"    {_paint('3', _C.BLUE, _C.BOLD)}  挂载 ScreenShooting_best  "
        + _paint("…_mask_ScreenShooting_best.pth", _C.DIM)
        + "\n"
        + f"    {_paint('4', _C.MAGENTA, _C.BOLD)}  挂载 ScreenShootingMB_best "
        + _paint("…_mask_ScreenShootingMB_best.pth (DMB)", _C.DIM)
        + "\n"
        + _paint("  ────────────────────────────────────────────", _C.DIM)
        + "\n"
    )
    print(menu, flush=True)

    can_ask = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    if not can_ask:
        raw = default_choice.strip()
        log_warn(f"非交互终端，训练初始化默认选项 {raw}")
    else:
        while True:
            try:
                tip = _paint(f"请输入 1 / 2 / 3 / 4 [默认 {default_choice}]: ", _C.CYAN)
                raw = input(tip).strip()
            except (EOFError, KeyboardInterrupt):
                print(flush=True)
                raw = default_choice.strip()
                break
            if raw == "":
                raw = default_choice.strip()
            if raw in ("1", "2", "3", "4"):
                break
            log_err("无效输入，请输入 1（从头）/ 2（99）/ 3（SS_best）/ 4（DMB_best）")

    if raw == "2":
        log_ok("训练初始化：挂载 mask_99（epoch 从 0 重新计）")
        return {"init_from_ckpt": "99", "init_from_epoch": 99, "embedding_epoch": 0}
    if raw == "3":
        log_ok("训练初始化：挂载 ScreenShooting_best（epoch 从 0 重新计）")
        return {"init_from_ckpt": "ss_best", "init_from_epoch": None, "embedding_epoch": 0}
    if raw == "4":
        log_ok("训练初始化：挂载 ScreenShootingMB_best / DMB（epoch 从 0 重新计）")
        return {"init_from_ckpt": "best", "init_from_epoch": None, "embedding_epoch": 0}
    log_ok("训练初始化：从头训练")
    return {"init_from_ckpt": None, "init_from_epoch": None, "embedding_epoch": 0}


def prompt_train_scale(default_choice: str = "2") -> bool:
    """交互询问训练规模；True=Lite，False=全量。"""
    menu = (
        "\n"
        + _paint("  请选择训练模式", _C.BOLD, _C.WHITE)
        + "\n"
        + _paint("  ────────────────────────────────────", _C.DIM)
        + "\n"
        + f"    {_paint('1', _C.GREEN, _C.BOLD)}  全量模式  "
        + _paint("(使用完整 COCOMask)", _C.DIM)
        + "\n"
        + f"    {_paint('2', _C.YELLOW, _C.BOLD)}  Lite 模式 "
        + _paint("(约 10%~20% 子集，适合试跑)", _C.DIM)
        + "\n"
        + _paint("  ────────────────────────────────────", _C.DIM)
        + "\n"
    )
    print(menu, flush=True)

    can_ask = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    if not can_ask:
        use_lite = default_choice.strip() == "2"
        log_warn(
            "非交互终端，跳过菜单 → "
            + ("Lite" if use_lite else "全量")
            + f"（默认选项 {default_choice}）"
        )
        return use_lite

    while True:
        try:
            tip = _paint(f"请输入 1 或 2 [默认 {default_choice}]: ", _C.CYAN)
            raw = input(tip).strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            log_warn("已取消输入，使用默认: " + ("Lite" if default_choice == "2" else "全量"))
            return default_choice.strip() == "2"

        if raw == "":
            raw = default_choice.strip()
        if raw in ("1", "全量", "full", "Full", "FULL"):
            log_ok("已选择：1 — 全量模式")
            return False
        if raw in ("2", "lite", "Lite", "LITE"):
            log_ok("已选择：2 — Lite 模式")
            return True
        log_err("无效输入，请输入 1（全量）或 2（Lite）")
