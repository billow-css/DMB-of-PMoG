# -*- coding: utf-8 -*-
"""进程入口：解析参数 → 模式分发 → 训练/评估。"""

from __future__ import annotations

import sys

from app.parser import build_parser
from app.resolve_mode import resolve_run_mode
from app.resolve_scale import (
    apply_lite_defaults,
    apply_scale_defaults,
    resolve_train_scale,
)
from app.run import main
from app.tools import run_tool_mode
from cli import (
    log_err,
    log_ok,
    print_banner,
    print_lite_badge,
    print_mode_badge,
    prompt_log_browser,
)


def _force_utf8_stdio() -> None:
    if sys.platform != "win32":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def cli_main(argv: list[str] | None = None) -> None:
    """``python main.py`` 入口。"""
    _force_utf8_stdio()
    if argv is not None:
        config = build_parser().parse_args(argv)
    else:
        config = build_parser().parse_args()
    print_banner(lite=False)
    config = resolve_run_mode(config)

    if config.mode == "quit":
        log_ok("已退出。")
        raise SystemExit(0)

    if config.mode == "log_query":
        print_mode_badge("log_query")
        if getattr(config, "log_run", None):
            from ui.train_log import format_run_detail, load_run

            try:
                doc = load_run(config.log_dir, config.log_run)
            except FileNotFoundError as e:
                log_err(str(e))
                raise SystemExit(1)
            print(format_run_detail(doc), flush=True)
            log_ok(f"JSON: {doc.get('json_path')}")
            raise SystemExit(0)
        raise SystemExit(prompt_log_browser(getattr(config, "log_dir", "logs")))

    if str(config.mode).startswith("tool_"):
        raise SystemExit(run_tool_mode(config.mode))

    config = resolve_train_scale(config)
    if config.lite:
        print_lite_badge()
    config = apply_scale_defaults(config)
    config = apply_lite_defaults(config)

    if config.mode == "train_mask":
        from ui.train_ui import edit_train_hparams, print_train_hparams

        config = edit_train_hparams(config)
        print_train_hparams(config)

    main(config)
