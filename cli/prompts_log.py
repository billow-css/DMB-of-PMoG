# -*- coding: utf-8 -*-
"""训练 JSON 日志交互查询。"""

from __future__ import annotations

import json

from cli.color import _C, _paint
from cli.input_util import ask_choice
from cli.log import log_err, log_info, log_ok, log_warn


def _print_txt_summary(log_dir: str, doc: dict) -> None:
    from ui.train_log import resolve_run_path

    txt = doc.get("txt_path")
    if not txt:
        return
    log_info(f"TXT:  {txt}")
    try:
        path = resolve_run_path(log_dir, doc["run_id"]).with_suffix(".txt")
        if not path.is_file():
            return
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        head = lines[:8]
        tail = lines[-12:] if len(lines) > 20 else lines[8:]
        print(_paint("—— TXT 摘要 ——", _C.DIM), flush=True)
        print("\n".join(head), flush=True)
        if len(lines) > 20:
            print(_paint("  …", _C.DIM), flush=True)
        print("\n".join(tail), flush=True)
    except OSError as e:
        log_warn(f"无法读取 txt: {e}")


def _show_run(log_dir: str, run_id: str, *, with_txt: bool = False) -> int:
    from ui.train_log import format_run_detail, load_run

    try:
        doc = load_run(log_dir, run_id)
    except FileNotFoundError as e:
        log_err(str(e))
        return 1
    print(format_run_detail(doc), flush=True)
    log_ok(f"JSON: {doc.get('json_path')}")
    if with_txt:
        _print_txt_summary(log_dir, doc)
    return 0


def prompt_log_browser(log_dir: str = "logs") -> int:
    """交互查询训练 JSON 日志；返回进程退出码。"""
    from ui.train_log import (
        format_run_brief,
        latest_pointer_path,
        list_runs,
        rebuild_index,
    )

    can_ask = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    print(
        "\n"
        + _paint("  训练日志查询", _C.BOLD, _C.WHITE)
        + "\n"
        + _paint("  ────────────────────────────────────────────────", _C.DIM)
        + "\n"
        + f"    {_paint('1', _C.CYAN, _C.BOLD)}  列出全部 run\n"
        + f"    {_paint('2', _C.CYAN, _C.BOLD)}  按噪声层筛选\n"
        + f"    {_paint('3', _C.CYAN, _C.BOLD)}  查看最近一次\n"
        + f"    {_paint('4', _C.CYAN, _C.BOLD)}  按编号 / run_id 调出详情\n"
        + f"    {_paint('5', _C.CYAN, _C.BOLD)}  打印 JSON 路径并打开 txt 摘要\n"
        + f"    {_paint('6', _C.YELLOW, _C.BOLD)}  重建索引 (index.json)\n"
        + f"    {_paint('0', _C.DIM)}  返回\n"
        + _paint("  ────────────────────────────────────────────────", _C.DIM)
        + "\n",
        flush=True,
    )
    choice = ask_choice(
        "请选择 [默认 1]: ", {"0", "1", "2", "3", "4", "5", "6"}, "1", can_ask=can_ask
    )
    if choice == "0":
        log_ok("已返回")
        return 0
    if choice == "6":
        idx = rebuild_index(log_dir)
        log_ok(f"索引已重建：{len(idx.get('runs', []))} 条 → {log_dir}/runs/index.json")
        return 0

    distortion = None
    if choice == "2":
        dsel = ask_choice(
            "噪声层 1=ScreenShooting  2=ScreenShootingMB [默认 2]: ",
            {"1", "2"},
            "2",
            can_ask=can_ask,
        )
        distortion = "ScreenShooting" if dsel == "1" else "ScreenShootingMB"

    if choice == "3":
        ptr = latest_pointer_path(log_dir)
        if not ptr.is_file():
            log_warn("尚无 LATEST_TRAIN.json（还没有训练 run）")
            return 0
        meta = json.loads(ptr.read_text(encoding="utf-8"))
        return _show_run(log_dir, str(meta.get("run_id")))

    rows = list_runs(log_dir, distortion=distortion, limit=50 if choice != "1" else None)
    if choice == "1":
        rows = list_runs(log_dir)
    if not rows:
        log_warn("没有匹配的训练日志。完成一次 train_mask 后会出现在 logs/runs/。")
        return 0

    print(_paint("\n  最近训练 run：", _C.BOLD, _C.WHITE), flush=True)
    for i, e in enumerate(rows):
        print("  " + format_run_brief(e, index=i), flush=True)
    print(_paint("─" * 64, _C.DIM), flush=True)

    if choice in ("1", "2"):
        if not can_ask:
            return 0
        try:
            raw = input(_paint("输入编号查看详情（空=跳过）: ", _C.CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            return 0
        if raw == "":
            return 0
        if not raw.isdigit() or not (0 <= int(raw) < len(rows)):
            log_err("编号无效")
            return 1
        return _show_run(log_dir, rows[int(raw)]["run_id"])

    if choice in ("4", "5"):
        if not can_ask:
            return _show_run(log_dir, rows[0]["run_id"]) if rows else 0
        try:
            raw = input(_paint("输入列表编号或 run_id/前缀: ", _C.CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            return 0
        if raw == "":
            log_warn("未输入")
            return 0
        key = rows[int(raw)]["run_id"] if raw.isdigit() and 0 <= int(raw) < len(rows) else raw
        return _show_run(log_dir, key, with_txt=(choice == "5"))

    return 0
