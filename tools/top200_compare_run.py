#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从最新逐图 CSV 取各模型 PSNR Top-200 并对比。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / 'results' / 'psnr_embed_report'
CSV_PATH = OUT / "psnr_per_image_latest.csv"
K = 200


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    idx = np.array([int(r["index"]) for r in rows])
    p99 = np.array([float(r["psnr_mask99"]) for r in rows])
    po = np.array([float(r["psnr_ours"]) for r in rows])

    ord99 = np.argsort(-p99)[:K]
    ordo = np.argsort(-po)[:K]
    idx99 = idx[ord99]
    idxo = idx[ordo]
    set99, seto = set(idx99.tolist()), set(idxo.tolist())
    inter = sorted(set99 & seto)

    map99 = dict(zip(idx.tolist(), p99.tolist()))
    mapo = dict(zip(idx.tolist(), po.tolist()))

    p99_on_top99 = p99[ord99]
    po_on_top99 = np.array([mapo[int(i)] for i in idx99])
    po_on_topo = po[ordo]
    p99_on_topo = np.array([map99[int(i)] for i in idxo])

    (OUT / "top200_mask99_indices.txt").write_text(
        "\n".join(map(str, idx99.tolist())) + "\n", encoding="utf-8"
    )
    (OUT / "top200_ours_indices.txt").write_text(
        "\n".join(map(str, idxo.tolist())) + "\n", encoding="utf-8"
    )

    with (OUT / "top200_compare.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "set",
                "index",
                "psnr_mask99",
                "psnr_ours",
                "delta_ours_minus_99",
            ]
        )
        for rank, i in enumerate(idx99, 1):
            i = int(i)
            w.writerow(
                [
                    rank,
                    "top200_by_mask99",
                    i,
                    f"{map99[i]:.6f}",
                    f"{mapo[i]:.6f}",
                    f"{mapo[i] - map99[i]:.6f}",
                ]
            )
        for rank, i in enumerate(idxo, 1):
            i = int(i)
            w.writerow(
                [
                    rank,
                    "top200_by_ours",
                    i,
                    f"{map99[i]:.6f}",
                    f"{mapo[i]:.6f}",
                    f"{mapo[i] - map99[i]:.6f}",
                ]
            )

    lines = [
        "# Top-200 PSNR 对比（mask_99 vs Ours）",
        "",
        "从 1000 张宿主嵌入 PSNR 中，分别按各模型 PSNR 取最高 200 张。",
        "",
        "## 全量 1000",
        "",
        "| | mask_99 | Ours |",
        "|:---|---:|---:|",
        f"| mean | {p99.mean():.4f} | {po.mean():.4f} |",
        f"| median | {np.median(p99):.4f} | {np.median(po):.4f} |",
        f"| min / max | {p99.min():.4f} / {p99.max():.4f} | {po.min():.4f} / {po.max():.4f} |",
        "",
        "## 各自 Top-200（集合可以不同）",
        "",
        "| 集合 | 该模型 mean (dB) | 另一模型在同集合 mean (dB) |",
        "|:---|---:|---:|",
        f"| PIMoG Top-200 | **{p99_on_top99.mean():.4f}** (mask99) | {po_on_top99.mean():.4f} (ours) |",
        f"| Ours Top-200 | **{po_on_topo.mean():.4f}** (ours) | {p99_on_topo.mean():.4f} (mask99) |",
        "",
        f"- PIMoG Top-200 自身均值: **{p99_on_top99.mean():.4f} dB** "
        f"(min {p99_on_top99.min():.4f}, max {p99_on_top99.max():.4f})",
        f"- Ours Top-200 自身均值: **{po_on_topo.mean():.4f} dB** "
        f"(min {po_on_topo.min():.4f}, max {po_on_topo.max():.4f})",
        f"- 两套 Top-200「各自最好」均值差 (OursTop - PIMoGTop): "
        f"**{po_on_topo.mean() - p99_on_top99.mean():+.4f} dB**",
        f"- 索引交集: **{len(inter)} / 200**",
        "",
        "## 配对视角（同一批图上两边都算）",
        "",
        f"- 在 PIMoG Top-200 上: Ours - mask99 = "
        f"**{po_on_top99.mean() - p99_on_top99.mean():+.4f} dB**",
        f"- 在 Ours Top-200 上: Ours - mask99 = "
        f"**{po_on_topo.mean() - p99_on_topo.mean():+.4f} dB**",
        "",
        "## 文件",
        "",
        "- `top200_mask99_indices.txt`",
        "- `top200_ours_indices.txt`",
        "- `top200_compare.csv`",
        "",
    ]
    md_path = OUT / "top200_compare.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("\n".join(lines))
    print(f"\n已写入: {md_path}")


if __name__ == "__main__":
    main()
