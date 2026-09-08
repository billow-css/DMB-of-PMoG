#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/verify_host_embed.py
==========================
宿主嵌入质量统一入口：PSNR / SSIM（或两者）。

通过子进程调用 ``verify_host_psnr.py`` / ``verify_host_ssim.py``，
避免同进程改写 ``sys.argv`` 带来的参数串扰。

用法
----
  python tools/verify_host_embed.py --metric psnr
  python tools/verify_host_embed.py --metric ssim --limit 100
  python tools/verify_host_embed.py --metric both --limit 50
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent


def _strip_metric(argv: list[str]) -> tuple[str, list[str]]:
    """解析并移除 --metric，其余原样保留。"""
    metric = "psnr"
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--metric" and i + 1 < len(argv):
            metric = argv[i + 1]
            i += 2
            continue
        if a.startswith("--metric="):
            metric = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    if metric not in ("psnr", "ssim", "both"):
        raise SystemExit(f"无效 --metric: {metric}（可选 psnr/ssim/both）")
    return metric, rest


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(
            "usage: verify_host_embed.py [--metric {psnr,ssim,both}] [子脚本参数...]\n"
            "\n"
            "宿主嵌入质量：PSNR / SSIM（统一入口）\n"
            "\n"
            "其余参数原样传给 verify_host_psnr / verify_host_ssim，例如：\n"
            "  --ckpt a.pth --ckpt b.pth --limit 100 --batch_size 32\n"
            "  --ckpt one.pth            # 单模型\n"
            "  （未给 --ckpt 时默认 mask_99 vs MB_best）"
        )
        return 0

    try:
        metric, rest = _strip_metric(argv)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2

    scripts = []
    if metric in ("psnr", "both"):
        scripts.append(_TOOLS / "verify_host_psnr.py")
    if metric in ("ssim", "both"):
        scripts.append(_TOOLS / "verify_host_ssim.py")

    codes: list[int] = []
    for script in scripts:
        cmd = [sys.executable, str(script), *rest]
        print(f"[verify_host_embed] → {' '.join(cmd)}", flush=True)
        codes.append(subprocess.run(cmd, cwd=str(_REPO)).returncode)
    return max(codes) if codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
