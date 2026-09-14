#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.report
===================
结果汇总：Markdown 对表 + CSV。每次落盘带时间戳，并覆盖 ``<mode>_latest.*``。
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, List

_FIELDS = [
    "model", "label", "message_bits", "input_size", "acc", "ber",
    "psnr", "ssim", "n_images", "noise",
]


def write_report(
    rows: List[Dict[str, Any]],
    mode: str,
    out_dir: str,
    extra: Dict[str, Any] = None,
) -> str:
    """rows 每项含 _FIELDS 键（acc/ber/psnr/ssim 为已格式化字符串）。返回 md 路径。"""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    header = [
        f"# 评估报告（{mode}）",
        "",
        f"- 生成时间: {stamp}",
    ]
    for k, v in (extra or {}).items():
        header.append(f"- {k}: `{v}`")
    header += [
        "",
        "| 模型 | 标签 | 消息比特 | 输入 | Acc↑ | BER↓ | PSNR(dB) | SSIM | 样本 | 噪声 |",
        "|:---|:---|---:|:---:|---:|---:|---:|---:|---:|:---|",
    ]
    for r in rows:
        header.append(
            f"| `{r['model']}` | {r['label']} | {r['message_bits']} | "
            f"{r['input_size'][0]}×{r['input_size'][1]} | {r['acc']} | {r['ber']} | "
            f"{r['psnr']} | {r['ssim']} | {r['n_images']} | {r['noise']} |"
        )
    header.append("")

    md = os.path.join(out_dir, f"{mode}_{stamp}.md")
    csv_path = os.path.join(out_dir, f"{mode}_{stamp}.csv")
    _write(md, header)
    _write(os.path.join(out_dir, f"{mode}_latest.md"), header)

    for path in (csv_path, os.path.join(out_dir, f"{mode}_latest.csv")):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in _FIELDS})
    return md


def _write(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
