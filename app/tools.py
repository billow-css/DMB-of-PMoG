# -*- coding: utf-8 -*-
"""工具 / 实验脚本分发。"""

from __future__ import annotations

import subprocess
import sys

from app.constants import REPO_ROOT
from cli import log_err, log_info, log_ok, log_warn, print_mode_badge


def _prompt_verify_ckpts(mode: str, extra: list[str]) -> list[str] | None:
    from ui.verify_weights_ui import ckpt_args_from_paths, prompt_embed_verify_weights

    titles = {
        "tool_verify_psnr": "嵌入 PSNR — 选择权重",
        "tool_verify_ssim": "嵌入 SSIM — 选择权重",
        "tool_verify_both": "嵌入质量 (PSNR+SSIM) — 选择权重",
    }
    picked = prompt_embed_verify_weights(
        title=titles.get(mode, "选择嵌入质量测试权重"),
        repo_root=REPO_ROOT,
    )
    if picked is None:
        log_warn("已取消权重选择，未启动测试。")
        return None
    extra = list(extra)
    extra.extend(ckpt_args_from_paths(picked))
    log_ok(
        "已选权重: "
        + " | ".join(str(p) for p in picked)
        + ("（对照）" if len(picked) == 2 else "（单测）")
    )
    return extra


def _prompt_sim_test_args(extra: list[str]) -> list[str] | int:
    can_ask = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    if not can_ask:
        log_err(
            "sim_test 需要 --input_dir / --output_dir。"
            "例: python sim_test/sim_noise.py -i DIR -o DIR --noise DMB"
        )
        return 2
    try:
        tip_i = "输入图像目录 [--input_dir，默认 Datasets/images_watermarked]: "
        tip_o = "输出目录 [--output_dir，默认 sim_test/out]: "
        tip_n = "噪声 1=PIMoG(ScreenShooting) 2=DMB(ScreenShootingMB) 3=both [默认 2]: "
        raw_i = input(tip_i).strip() or "Datasets/images_watermarked"
        raw_o = input(tip_o).strip() or "sim_test/out"
        raw_n = input(tip_n).strip() or "2"
    except (EOFError, KeyboardInterrupt):
        print(flush=True)
        log_warn("已取消 sim_test。")
        return 130
    noise = {"1": "PIMoG", "2": "DMB", "3": "both"}.get(raw_n, raw_n)
    extra = list(extra)
    extra.extend(["--input_dir", raw_i, "--output_dir", raw_o, "--noise", noise])
    log_ok(f"sim_test: in={raw_i} → out={raw_o} | noise={noise}")
    return extra


def run_tool_mode(mode: str, extra_args: list[str] | None = None) -> int:
    """分发工具 / 实验脚本（不经过 Solver）。"""
    mapping = {
        "tool_export_hosts": (REPO_ROOT / "tools" / "export_coco_hosts.py", []),
        "tool_verify_psnr": (REPO_ROOT / "tools" / "verify_host_psnr.py", []),
        "tool_verify_ssim": (REPO_ROOT / "tools" / "verify_host_ssim.py", []),
        "tool_verify_both": (
            REPO_ROOT / "tools" / "verify_host_embed.py",
            ["--metric", "both"],
        ),
        "tool_crop_panels": (REPO_ROOT / "tools" / "crop_embed_panels.py", []),
        "tool_sim_test": (REPO_ROOT / "sim_test" / "sim_noise.py", []),
        "tool_batch_rectify": (REPO_ROOT / "experiment" / "batch_rectify.py", []),
        "tool_rectify_gui": (REPO_ROOT / "experiment" / "rectify_gui.py", []),
        "tool_server_gui": (REPO_ROOT / "experiment" / "server_gui.py", []),
    }
    if mode not in mapping:
        log_err(f"未知工具模式: {mode}")
        return 2
    script, preset = mapping[mode]
    if not script.is_file():
        log_err(f"找不到脚本: {script}")
        return 1

    extra = list(extra_args or [])
    verify_modes = {"tool_verify_psnr", "tool_verify_ssim", "tool_verify_both"}
    if (
        mode in verify_modes
        and "--ckpt" not in extra
        and "--ckpt99" not in extra
        and "--no_interactive" not in sys.argv
        and "--ckpt" not in sys.argv
    ):
        prompted = _prompt_verify_ckpts(mode, extra)
        if prompted is None:
            return 130
        extra = prompted

    if (
        mode == "tool_sim_test"
        and "--input_dir" not in extra
        and "-i" not in extra
        and "--no_interactive" not in sys.argv
        and "--input_dir" not in sys.argv
        and "-i" not in sys.argv
    ):
        prompted = _prompt_sim_test_args(extra)
        if isinstance(prompted, int):
            return prompted
        extra = prompted

    cmd = [sys.executable, str(script), *preset, *extra]
    cwd = str(script.parent) if script.parent.name == "experiment" else str(REPO_ROOT)
    print_mode_badge(mode)
    log_info(f"启动: {' '.join(cmd)}")
    log_info(f"工作目录: {cwd}")
    try:
        completed = subprocess.run(cmd, cwd=cwd, check=False)
    except KeyboardInterrupt:
        log_warn("已中断工具进程")
        return 130
    code = int(completed.returncode)
    if code == 0:
        log_ok("工具执行完成。")
    else:
        log_err(f"工具退出码: {code}")
    return code
