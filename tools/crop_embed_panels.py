#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 test_embedding 五联拼图中裁出「含水印图」（第 2 栏），保存到新目录。

五联从左到右：
  1 原图 | 2 含水印 Encoded | 3 噪声仿真 | 4 梯度 mask | 5 残差×5

拍屏测试请使用第 2 栏（默认裁这一栏）。
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
N_PANELS = 5


def natural_key(path: Path):
    stem = path.stem
    m = re.match(r"^(\d+)$", stem)
    if m:
        return (0, int(m.group(1)))
    return (1, stem)


def crop_panel(img: Image.Image, panel_index: int, n_panels: int = N_PANELS) -> Image.Image:
    w, h = img.size
    if w % n_panels != 0:
        raise ValueError(f"宽度 {w} 不能被 {n_panels} 整除，不像五联拼图")
    pw = w // n_panels
    if not (0 <= panel_index < n_panels):
        raise ValueError(f"panel_index 应在 0..{n_panels - 1}")
    x0 = panel_index * pw
    return img.crop((x0, 0, x0 + pw, h))


def main() -> None:
    root = Path(__file__).resolve().parent.parent  # repo root
    p = argparse.ArgumentParser(description="裁剪五联图中的含水印栏到新文件夹")
    p.add_argument(
        "--src",
        type=Path,
        default=root
        / "results"
        / "Image_test_ScreenShootingMB"
        / "images_embed_99",
        help="五联图目录（test_embedding 输出）",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "Datasets" / "images_embed_99",
        help="输出目录（默认 Datasets/images_watermarked）",
    )
    p.add_argument(
        "--panel",
        type=int,
        default=1,
        help="裁哪一栏（0-based；默认 1=含水印 Encoded）",
    )
    p.add_argument(
        "--n_panels",
        type=int,
        default=N_PANELS,
        help="拼图栏数（默认 5）",
    )
    p.add_argument(
        "--clear_out",
        action="store_true",
        help="导出前清空输出目录中的图像",
    )
    args = p.parse_args()

    src: Path = args.src
    out: Path = args.out
    if not src.is_dir():
        raise SystemExit(f"源目录不存在: {src}")

    files = sorted(
        [
            f
            for f in src.iterdir()
            if f.is_file() and f.suffix.lower() in EXTS and not f.name.startswith(".")
        ],
        key=natural_key,
    )
    if not files:
        raise SystemExit(f"源目录无图像: {src}")

    out.mkdir(parents=True, exist_ok=True)
    if args.clear_out:
        n_rm = 0
        for f in out.iterdir():
            if f.is_file() and f.suffix.lower() in EXTS:
                f.unlink()
                n_rm += 1
        print(f"[INFO] 已清空输出图像 {n_rm} 个 → {out}")

    ok = 0
    for f in files:
        try:
            with Image.open(f) as im:
                panel = crop_panel(im.convert("RGB"), args.panel, args.n_panels)
            # 保持数字文件名，便于后续拍屏矫正与 test_accuracy 对齐 w.mat
            dst = out / f"{f.stem}.png"
            panel.save(dst)
            ok += 1
        except (OSError, ValueError) as e:
            print(f"[WARN] 跳过 {f.name}: {e}")

    print(f"[OK] 已裁剪第 {args.panel + 1} 栏 → {ok}/{len(files)} 张")
    print(f"[OK] 输出目录: {out}")
    print("下一步：全屏显示这些图 → 手机拍屏 → MATLAB 透视矫正 → test_accuracy")


if __name__ == "__main__":
    if os.name == "nt":
        try:
            import sys

            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
