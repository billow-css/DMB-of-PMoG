# -*- coding: utf-8 -*-
"""菜单输入辅助。"""

from __future__ import annotations

from cli.color import _C, _paint
from cli.log import log_err, log_warn


def ask_choice(
    tip: str,
    valid: set[str],
    default: str,
    *,
    can_ask: bool,
) -> str:
    """统一菜单输入：空=默认；非法重试；非 TTY 用默认。字母不区分大小写。"""
    valid_l = {v.lower() for v in valid}
    default_l = default.strip().lower()
    if not can_ask:
        log_warn(f"非交互终端 → 默认选项 {default_l}")
        return default_l
    while True:
        try:
            raw = input(_paint(tip, _C.CYAN)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            log_warn(f"已取消输入，使用默认选项 {default_l}")
            return default_l
        if raw == "":
            raw = default_l
        if raw in valid_l:
            return raw
        log_err(f"无效输入，请输入: {' / '.join(sorted(valid_l))}")
