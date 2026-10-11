# -*- coding: utf-8 -*-
"""彩色控制台日志。"""

from __future__ import annotations

from cli.color import _C, _paint


def log_info(msg: str) -> None:
    print(_paint("[INFO] ", _C.CYAN) + msg, flush=True)


def log_ok(msg: str) -> None:
    print(_paint("[ OK ] ", _C.GREEN, _C.BOLD) + msg, flush=True)


def log_warn(msg: str) -> None:
    print(_paint("[WARN] ", _C.YELLOW, _C.BOLD) + msg, flush=True)


def log_err(msg: str) -> None:
    print(_paint("[ERR ] ", _C.RED, _C.BOLD) + msg, flush=True)


def log_step(msg: str) -> None:
    print(_paint("  » ", _C.MAGENTA) + msg, flush=True)
