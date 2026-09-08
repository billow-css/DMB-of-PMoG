#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli_utils.py
============
DMB of PMoG 命令行前端：横幅、功能面板、配置摘要、彩色日志、进度条。

命名
----
- **DMB of PMoG** = Dynamic Motion Blur of PMoG
- 相对原 PIMoG 少了光照验证的 ``I``，故写作 **PMoG**；本仓库主噪声层为动态运动模糊（DMB）。

本模块不参与模型计算，仅负责人机交互，可在无 GPU / 无 tqdm 时降级为纯文本。

公开接口
--------
- print_banner() / PROJECT_*   品牌横幅
- prompt_run_mode()            主功能面板（训练评估 + 工具 + 实验）
- prompt_noise_layer()         ScreenShooting / ScreenShootingMB(DMB)
- prompt_eval_ckpt()           （兼容）标签选权重；交互已改为弹窗选 .pth
- prompt_train_init()          从头 / 挂载 99 / SS_best / DMB_best
- prompt_train_init() / prompt_train_scale()
- print_config_summary() / print_mode_badge() / print_lite_badge()
- log_* / ProgressMonitor / format_eta()
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ckpt_utils import (
    ckpt_default_distortion,
    ckpt_short,
    default_noise_menu_choice,
)

# ---------------------------------------------------------------------------
# 品牌
# ---------------------------------------------------------------------------
PROJECT_SHORT = "DMB of PMoG"
PROJECT_FULL = "Dynamic Motion Blur of PMoG"
PROJECT_TAGLINE = "Screen-shooting watermark · Dynamic motion-blur noise layer"

# ---------------------------------------------------------------------------
# 可选依赖：tqdm 缺失时自动降级
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False
    _tqdm = None


# ---------------------------------------------------------------------------
# ANSI 颜色（Windows 10+ 终端 / 多数现代终端可用）
# ---------------------------------------------------------------------------
class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    WHITE = "\033[37m"


def _supports_color() -> bool:
    """检测当前 stdout 是否适合输出 ANSI 颜色。"""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    # Windows：尝试开启 VT 模式
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            return True
        except Exception:
            return False
    return True


_COLOR = _supports_color()


def _paint(text: str, *codes: str) -> str:
    if not _COLOR:
        return text
    return "".join(codes) + text + _C.RESET


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 横幅与配置摘要
# ---------------------------------------------------------------------------
_BANNER = r"""
 ____  __  __ ____      ____  __  __      ____
|  _ \|  \/  | __ )    |  _ \|  \/  |___ / ___|
| | | | |\/| |  _ \ ___| |_) | |\/| / _ \ |  _
| |_| | |  | | |_) |___|  __/| |  | | (_) | |_| |
|____/|_|  |_|____/    |_|   |_|  |_|\___/ \____|
"""


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
        _paint("  (PMoG = PIMoG without illumination-I · DMB = Dynamic Motion Blur)", _C.DIM),
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
    labels = {
        "eval_mask": (_C.MAGENTA, "EVAL", "弹窗选权重 · 验证集 Acc/BER/PSNR"),
        "train_mask": (_C.GREEN, "TRAIN", "Encoder–Noise–Decoder 端到端训练"),
        "test_embedding": (_C.CYAN, "EMBED", "预训练 Encoder 嵌入水印并导出拼图"),
        "test_accuracy": (_C.BLUE, "ACCURACY", "拍屏矫正图 Acc/BER + 相对含水印宿主 PSNR"),
        "tool_export_hosts": (_C.YELLOW, "TOOL", "从 COCOMask 导出宿主 → Datasets/images"),
        "tool_verify_psnr": (_C.YELLOW, "TOOL", "宿主嵌入 PSNR：弹窗选 1~2 个权重"),
        "tool_verify_ssim": (_C.YELLOW, "TOOL", "宿主嵌入 SSIM：弹窗选 1~2 个权重"),
        "tool_verify_both": (_C.YELLOW, "TOOL", "宿主嵌入 PSNR+SSIM：弹窗选权重"),
        "tool_crop_panels": (_C.YELLOW, "TOOL", "从结果拼图裁切嵌入面板"),
        "tool_batch_rectify": (_C.BLUE, "EXP", "批量透视矫正（experiment/batch_rectify）"),
        "tool_rectify_gui": (_C.BLUE, "EXP", "交互透视矫正 GUI"),
        "tool_server_gui": (_C.BLUE, "EXP", "联机拍屏服务端 GUI"),
        "log_query": (_C.CYAN, "LOGS", "查询 / 调出训练 JSON 历史"),
    }
    color, tag, desc = labels.get(mode, (_C.WHITE, mode.upper(), ""))
    print(
        _paint(f"  ▶ {tag}", color, _C.BOLD)
        + _paint(f"  — {desc}", _C.DIM),
        flush=True,
    )
    print(_paint("─" * 64, _C.DIM), flush=True)


def _ask_choice(
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


def prompt_run_mode(default_choice: str = "0") -> dict:
    """
    交互主菜单：训练评估 + 工具 + 实验。

    Returns
    -------
    dict
        核心模式含 ``mode`` / ``dataset`` / ``lite`` 等；
        工具模式 ``mode`` 以 ``tool_`` 开头，由 ``main.run_tool_mode`` 分发。
    """
    menu = (
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
    print(menu, flush=True)

    can_ask = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    valid = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "l", "q"}
    raw = _ask_choice(
        f"请选择 [默认 {default_choice}]: ",
        valid,
        default_choice,
        can_ask=can_ask,
    )

    mapping = {
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
        "a": {"mode": "tool_batch_rectify", "label": "a — 批量矫正"},
        "b": {"mode": "tool_rectify_gui", "label": "b — 矫正 GUI"},
        "c": {"mode": "tool_server_gui", "label": "c — 服务端 GUI"},
        "l": {"mode": "log_query", "label": "l — 训练日志查询"},
        "q": {"mode": "quit", "label": "q — 退出"},
    }
    chosen = dict(mapping[raw])
    log_ok(f"已选择：{chosen['label']}")

    if chosen["mode"] == "quit":
        return chosen

    # 工具 / 实验 / 日志查询：不再追问权重与噪声层
    if str(chosen["mode"]).startswith("tool_") or chosen["mode"] == "log_query":
        return chosen

    # 评估 / 嵌入 / 测准：弹窗自选 .pth，再选噪声层
    if chosen["mode"] in ("eval_mask", "test_embedding", "test_accuracy") and can_ask:
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
            repo_root=Path(__file__).resolve().parent,
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
    elif chosen["mode"] == "train_mask" and can_ask:
        chosen["distortion"] = prompt_noise_layer(default_choice="2")
    elif chosen["mode"] in (
        "train_mask",
        "eval_mask",
        "test_embedding",
        "test_accuracy",
    ):
        if chosen["mode"] in ("eval_mask", "test_embedding", "test_accuracy"):
            chosen["eval_ckpt"] = chosen.get("eval_ckpt", "99")
            chosen["embedding_epoch"] = chosen.get("embedding_epoch", 99)
        chosen["distortion"] = "ScreenShooting"

    if chosen["mode"] == "train_mask" and can_ask:
        chosen.update(prompt_train_init(default_choice="1"))
    elif chosen["mode"] == "train_mask":
        chosen["init_from_epoch"] = None
        chosen["embedding_epoch"] = 0

    if chosen["mode"] == "eval_mask" and can_ask:
        try:
            tip = _paint("评估用 Lite 子集? 1=全量验证  2=Lite [默认 2]: ", _C.CYAN)
            sub = input(tip).strip() or "2"
            chosen["lite"] = sub in ("2", "lite", "Lite", "LITE")
            log_ok("评估数据：" + ("Lite 子集" if chosen["lite"] else "全量验证集"))
        except (EOFError, KeyboardInterrupt):
            chosen["lite"] = True

    return chosen


def prompt_eval_ckpt(default_choice: str = "1") -> dict:
    """
    评估 / 嵌入 / 测准共用的权重选择。

    Returns
    -------
    dict
        ``eval_ckpt``: ``99`` / ``ss_best`` / ``best``
        ``embedding_epoch``: 选 99 时为 99；named best 时作占位
    """
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
        return {
            "eval_ckpt": "best",
            "embedding_epoch": 99,
            "embed_strength": None,
        }
    if raw == "2":
        log_ok("权重：ScreenShooting_best")
        return {
            "eval_ckpt": "ss_best",
            "embedding_epoch": 99,
            "embed_strength": None,
        }
    log_ok("权重：Encoder_Decoder_Model_mask_99.pth")
    return {
        "eval_ckpt": "99",
        "embedding_epoch": 99,
        "embed_strength": 0.0,
    }


def prompt_noise_layer(default_choice: str = "1") -> str:
    """
    交互选择噪声层（训练或评估）。

    Returns
    -------
    str
        ``ScreenShooting`` 或 ``ScreenShootingMB``
    """
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


def prompt_train_init(default_choice: str = "1") -> dict:
    """
    训练初始化：从头，或挂载 mask_99 / ScreenShooting_best / ScreenShootingMB_best。

    Returns
    -------
    dict
        ``init_from_ckpt``: None / ``99`` / ``ss_best`` / ``best``
        ``init_from_epoch``: 兼容旧逻辑；仅挂载 99 时为 99，其余为 None
        ``embedding_epoch``: 训练循环起始（拷贝初始化时从 0 开记）
    """
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
        return {
            "init_from_ckpt": "99",
            "init_from_epoch": 99,
            "embedding_epoch": 0,
        }
    if raw == "3":
        log_ok("训练初始化：挂载 ScreenShooting_best（epoch 从 0 重新计）")
        return {
            "init_from_ckpt": "ss_best",
            "init_from_epoch": None,
            "embedding_epoch": 0,
        }
    if raw == "4":
        log_ok("训练初始化：挂载 ScreenShootingMB_best / DMB（epoch 从 0 重新计）")
        return {
            "init_from_ckpt": "best",
            "init_from_epoch": None,
            "embedding_epoch": 0,
        }
    log_ok("训练初始化：从头训练")
    return {
        "init_from_ckpt": None,
        "init_from_epoch": None,
        "embedding_epoch": 0,
    }


def prompt_train_scale(default_choice: str = "2") -> bool:
    """
    交互询问训练数据规模。

    Parameters
    ----------
    default_choice : str
        直接回车时的默认选项：``"1"`` 全量，``"2"`` Lite。

    Returns
    -------
    bool
        True = Lite 模式；False = 全量模式。

    Notes
    -----
    非 TTY（管道 / 后台）时无法交互，回退为默认选项对应模式。
    """
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


def prompt_log_browser(log_dir: str = "logs") -> int:
    """
    交互查询训练 JSON 日志。

    Returns
    -------
    int
        进程退出码（0=正常）。
    """
    from ui.train_log import (
        format_run_brief,
        format_run_detail,
        latest_pointer_path,
        list_runs,
        load_run,
        rebuild_index,
        resolve_run_path,
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
    choice = _ask_choice(
        "请选择 [默认 1]: ",
        {"0", "1", "2", "3", "4", "5", "6"},
        "1",
        can_ask=can_ask,
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
        dsel = _ask_choice(
            "噪声层 1=ScreenShooting  2=ScreenShootingMB [默认 2]: ",
            {"1", "2"},
            "2",
            can_ask=can_ask,
        )
        distortion = "ScreenShooting" if dsel == "1" else "ScreenShootingMB"

    rows = list_runs(log_dir, distortion=distortion, limit=50 if choice != "1" else None)
    if choice == "1":
        rows = list_runs(log_dir)

    if choice == "3":
        ptr = latest_pointer_path(log_dir)
        if not ptr.is_file():
            log_warn("尚无 LATEST_TRAIN.json（还没有训练 run）")
            return 0
        import json

        meta = json.loads(ptr.read_text(encoding="utf-8"))
        run_id = meta.get("run_id")
        doc = load_run(log_dir, str(run_id))
        print(format_run_detail(doc), flush=True)
        log_ok(f"JSON: {doc.get('json_path')}")
        return 0

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
        tip = _paint("输入编号查看详情（空=跳过）: ", _C.CYAN)
        try:
            raw = input(tip).strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            return 0
        if raw == "":
            return 0
        if not raw.isdigit() or not (0 <= int(raw) < len(rows)):
            log_err("编号无效")
            return 1
        run_id = rows[int(raw)]["run_id"]
        doc = load_run(log_dir, run_id)
        print(format_run_detail(doc), flush=True)
        log_ok(f"JSON: {doc.get('json_path')}")
        return 0

    if choice in ("4", "5"):
        if not can_ask:
            if rows:
                doc = load_run(log_dir, rows[0]["run_id"])
                print(format_run_detail(doc), flush=True)
            return 0
        tip = _paint("输入列表编号或 run_id/前缀: ", _C.CYAN)
        try:
            raw = input(tip).strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            return 0
        if raw == "":
            log_warn("未输入")
            return 0
        if raw.isdigit() and 0 <= int(raw) < len(rows):
            key = rows[int(raw)]["run_id"]
        else:
            key = raw
        try:
            doc = load_run(log_dir, key)
        except FileNotFoundError as e:
            log_err(str(e))
            return 1
        print(format_run_detail(doc), flush=True)
        log_ok(f"JSON: {doc.get('json_path')}")
        if choice == "5":
            txt = doc.get("txt_path")
            if txt:
                log_info(f"TXT:  {txt}")
                try:
                    path = resolve_run_path(log_dir, doc["run_id"]).with_suffix(".txt")
                    if path.is_file():
                        text = path.read_text(encoding="utf-8", errors="replace")
                        # 只展示头尾，避免刷屏
                        lines = text.splitlines()
                        head = lines[:8]
                        tail = lines[-12:] if len(lines) > 20 else lines[8:]
                        print(_paint("—— TXT 摘要 ——", _C.DIM), flush=True)
                        print("\n".join(head), flush=True)
                        if len(lines) > 20:
                            print(_paint("  …", _C.DIM), flush=True)
                        print("\n".join(tail), flush=True)
                except OSError as e:
                    log_warn(f"无法读取 txt: {e}")
        return 0

    return 0


def print_config_summary(config: Any, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    以两列表格打印 argparse Namespace（及额外键值）。

    Parameters
    ----------
    config : argparse.Namespace
        命令行解析结果。
    extra : dict, optional
        运行时补充信息（如 device、样本数、weight_path）。
    """
    rows: Dict[str, Any] = {}
    if hasattr(config, "__dict__"):
        rows.update(vars(config))
    if extra:
        rows.update(extra)

    mode = str(rows.get("mode", ""))
    title = "  运行配置"
    if mode == "eval_mask":
        title = "  评估配置 (eval_mask)"
    elif mode == "train_mask":
        title = "  训练配置 (train_mask)"
    elif mode == "test_embedding":
        title = "  嵌入配置 (test_embedding)"
    elif mode == "test_accuracy":
        title = "  测准配置 (test_accuracy)"

    # 分组展示优先级
    priority = [
        "mode", "dataset", "distortion", "eval_ckpt", "weight_path",
        "lite", "lite_ratio", "lite_seed",
        "warmup_epochs", "embed_strength", "init_from_ckpt", "init_from_epoch",
        "image_dir", "image_val_dir", "psnr_ref_dir", "batch_size", "num_epoch",
        "embedding_epoch", "image_size", "device", "train_samples",
        "val_samples", "lambda1", "lambda2", "lambda3", "lr",
        "save_viz", "result_dir",
    ]
    keys = [k for k in priority if k in rows] + [
        k for k in sorted(rows.keys()) if k not in priority
    ]

    print(_paint(title, _C.BOLD, _C.WHITE), flush=True)
    key_w = max(len(str(k)) for k in keys) if keys else 8
    highlight = {
        "weight_path",
        "mode",
        "eval_ckpt",
        "embedding_epoch",
        "init_from_ckpt",
        "init_from_epoch",
        "distortion",
        "embed_strength",
    }
    for k in keys:
        v = rows[k]
        key_s = f"    {str(k):<{key_w}}  "
        val_s = str(v)
        if k in highlight:
            print(_paint(key_s, _C.DIM) + _paint(val_s, _C.BOLD, _C.CYAN), flush=True)
        else:
            print(_paint(key_s, _C.DIM) + val_s, flush=True)
    print(_paint("─" * 56, _C.DIM), flush=True)


def format_eta(seconds: float) -> str:
    """将秒数格式化为可读 ETA（如 1h02m15s）。"""
    if seconds < 0 or seconds != seconds:  # NaN
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# 进度监控
# ---------------------------------------------------------------------------
class ProgressMonitor:
    """
    统一进度监控封装。

    - 有 tqdm 且 enable=True → 实时进度条（loss / acc 后缀更新）
    - 否则 → 按 log_every 间隔打印纯文本进度行

    Example
    -------
    >>> mon = ProgressMonitor(total=100, desc="Epoch 1", enable=True)
    >>> for i in range(100):
    ...     mon.update(1, loss=0.12, msg=0.05)
    >>> mon.close()
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        enable: bool = True,
        unit: str = "batch",
        log_every: int = 20,
        file=None,
    ):
        self.total = max(int(total), 1)
        self.desc = desc
        self.enable = enable
        self.unit = unit
        self.log_every = max(int(log_every), 1)
        self.file = file
        self.n = 0
        self._t0 = time.time()
        self._metrics: Dict[str, float] = {}
        self._bar = None

        if enable and _HAS_TQDM:
            self._bar = _tqdm(
                total=self.total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
            )
        elif enable:
            log_info(f"{desc}  (0/{self.total})  — tqdm 未安装，使用文本进度")

    def update(self, n: int = 1, **metrics: float) -> None:
        """推进进度并刷新指标。"""
        self.n += n
        self._metrics.update({k: float(v) for k, v in metrics.items()})
        if self._bar is not None:
            self._bar.set_postfix(self._metrics, refresh=False)
            self._bar.update(n)
            return

        if self.enable and (self.n % self.log_every == 0 or self.n >= self.total):
            elapsed = time.time() - self._t0
            rate = self.n / elapsed if elapsed > 0 else 0.0
            remain = (self.total - self.n) / rate if rate > 0 else 0.0
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in self._metrics.items())
            line = (
                f"  [{self.desc}] {self.n}/{self.total} "
                f"({100.0 * self.n / self.total:.1f}%)  "
                f"ETA {format_eta(remain)}  {metric_str}"
            )
            print(line, flush=True)
            if self.file is not None:
                print(line, file=self.file, flush=True)

    def set_description(self, desc: str) -> None:
        self.desc = desc
        if self._bar is not None:
            self._bar.set_description(desc)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def wrap_iterable(
    iterable: Iterable,
    total: Optional[int] = None,
    desc: str = "",
    enable: bool = True,
    **kwargs,
) -> Iterable:
    """
    将任意可迭代对象包装为带进度条的迭代器。

    若 disable / 无 tqdm，则原样返回 iterable。
    """
    if not enable:
        return iterable
    if _HAS_TQDM:
        kwargs.setdefault("leave", True)
        kwargs.setdefault("dynamic_ncols", True)
        return _tqdm(
            iterable,
            total=total,
            desc=desc,
            **kwargs,
        )
    return iterable
