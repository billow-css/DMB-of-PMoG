# -*- coding: utf-8 -*-
"""主功能面板交互：选模式并补齐权重 / 噪声层 / 初始化。"""

from __future__ import annotations

import sys
from pathlib import Path

from cli.color import _C, _paint
from cli.input_util import ask_choice
from cli.log import log_ok, log_warn
from cli.menu_main import MODE_MAPPING, VALID_CHOICES, build_main_menu
from cli.prompts_ckpt import prompt_noise_layer
from cli.prompts_train import prompt_train_init

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _fill_eval_weight(chosen: dict, can_ask: bool) -> dict:
    """评估 / 嵌入 / 测准：弹窗选权重并选噪声层。"""
    from ui.verify_weights_ui import (
        infer_noise_menu_choice_from_path,
        prompt_single_weight_file,
    )

    titles = {
        "eval_mask": "评估模式 — 选择权重",
        "test_embedding": "水印叠加 — 选择权重",
        "test_accuracy": "拍屏测准 — 选择权重",
    }
    picked = prompt_single_weight_file(
        title=titles.get(chosen["mode"], "选择权重"),
        repo_root=_REPO_ROOT,
    )
    if picked is None:
        log_warn("已取消权重选择。")
        return {"mode": "quit", "label": "取消"}
    chosen["weight_path"] = str(picked)
    chosen["eval_ckpt"] = "custom"
    chosen["embedding_epoch"] = chosen.get("embedding_epoch", 99)
    log_ok(f"权重文件：{picked}")
    noise_default = infer_noise_menu_choice_from_path(picked)
    if "--distortion" not in sys.argv:
        chosen["distortion"] = prompt_noise_layer(default_choice=noise_default)
    return chosen


def _fill_core_followups(chosen: dict, can_ask: bool) -> dict:
    """为核心模式补齐噪声层、训练初始化、评估 Lite 选项。"""
    mode = chosen["mode"]
    if mode in ("eval_mask", "test_embedding", "test_accuracy") and can_ask:
        return _fill_eval_weight(chosen, can_ask)

    if mode == "train_mask" and can_ask:
        chosen["distortion"] = prompt_noise_layer(default_choice="2")
    elif mode in ("train_mask", "eval_mask", "test_embedding", "test_accuracy"):
        if mode in ("eval_mask", "test_embedding", "test_accuracy"):
            chosen["eval_ckpt"] = chosen.get("eval_ckpt", "99")
            chosen["embedding_epoch"] = chosen.get("embedding_epoch", 99)
        chosen["distortion"] = "ScreenShooting"

    if mode == "train_mask" and can_ask:
        chosen.update(prompt_train_init(default_choice="1"))
    elif mode == "train_mask":
        chosen["init_from_epoch"] = None
        chosen["embedding_epoch"] = 0

    if mode == "eval_mask" and can_ask:
        try:
            tip = _paint("评估用 Lite 子集? 1=全量验证  2=Lite [默认 2]: ", _C.CYAN)
            sub = input(tip).strip() or "2"
            chosen["lite"] = sub in ("2", "lite", "Lite", "LITE")
            log_ok("评估数据：" + ("Lite 子集" if chosen["lite"] else "全量验证集"))
        except (EOFError, KeyboardInterrupt):
            chosen["lite"] = True
    return chosen


def prompt_run_mode(default_choice: str = "0") -> dict:
    """
    交互主菜单：训练评估 + 工具 + 实验。

    工具模式 ``mode`` 以 ``tool_`` 开头，由 ``app.tools.run_tool_mode`` 分发。
    """
    print(build_main_menu(), flush=True)
    can_ask = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    raw = ask_choice(
        f"请选择 [默认 {default_choice}]: ",
        VALID_CHOICES,
        default_choice,
        can_ask=can_ask,
    )
    chosen = dict(MODE_MAPPING[raw])
    log_ok(f"已选择：{chosen['label']}")

    if chosen["mode"] == "quit":
        return chosen
    if str(chosen["mode"]).startswith("tool_") or chosen["mode"] == "log_query":
        return chosen
    return _fill_core_followups(chosen, can_ask)
