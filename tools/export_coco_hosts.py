#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 COCOMask 验证集裁出左半边宿主图，随机抽 N 张并编号保存。

COCOMask 约定：水平拼接 [host | mask]，宽≈2×高（如 128×256）。
本脚本只保留左半边（宿主），供 test_embedding 使用。

默认：
  输入  Datasets/COCOMask/val/val_class/
  输出  Datasets/images/
  数量  1000，文件名 0.png … 999.png

若 w.mat 行数不足，可用 --extend_wmat 补齐随机 30-bit 消息行。
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import scipy.io as scio
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(src: Path) -> list[Path]:
    files = [
        p
        for p in sorted(src.iterdir())
        if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(f"目录下无图像: {src}")
    return files


def crop_left_host(img_rgb: Image.Image, host_size: int | None) -> Image.Image:
    """裁左半边宿主。host_size=None 时取宽的一半。"""
    w, h = img_rgb.size
    if host_size is None:
        host_w = w // 2
    else:
        host_w = int(host_size)
    if w < host_w or h < 1:
        raise ValueError(f"图像过小: {h}×{w}，需要宽≥{host_w}")
    left = img_rgb.crop((0, 0, host_w, h))
    # 非正方形时居中裁成正方形（一般 COCOMask 已是 H==host_w）
    lw, lh = left.size
    if lw != lh:
        side = min(lw, lh)
        x0 = (lw - side) // 2
        y0 = (lh - side) // 2
        left = left.crop((x0, y0, x0 + side, y0 + side))
    return left


def maybe_extend_wmat(wmat_path: Path, n: int, seed: int) -> None:
    """保证 w.mat 至少有 n 行；不足则用随机 0/1 比特补齐并备份原文件。"""
    if not wmat_path.is_file():
        raise FileNotFoundError(f"找不到水印矩阵: {wmat_path}")
    data = scio.loadmat(str(wmat_path))
    if "w" not in data:
        raise KeyError(f"{wmat_path} 中无键 'w'")
    w = np.asarray(data["w"])
    if w.ndim != 2:
        raise ValueError(f"w 应为 2D，当前 shape={w.shape}")
    rows, cols = w.shape
    if rows >= n:
        print(f"[OK] w.mat 已有 {rows} 行 ≥ {n}，无需扩展")
        return

    rng = np.random.default_rng(seed)
    extra = rng.integers(0, 2, size=(n - rows, cols), dtype=w.dtype)
    w_new = np.vstack([w, extra])
    bak = wmat_path.with_suffix(wmat_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(wmat_path, bak)
        print(f"[INFO] 已备份原 w.mat → {bak}")
    scio.savemat(str(wmat_path), {"w": w_new})
    print(f"[OK] w.mat 已扩展 {rows} → {n} 行（列={cols}）")


def main() -> None:
    root = Path(__file__).resolve().parent.parent  # repo root
    p = argparse.ArgumentParser(
        description="从 COCOMask 验证集裁左半边宿主，随机抽 N 张并编号"
    )
    p.add_argument(
        "--src",
        type=Path,
        default=root / "Datasets" / "COCOMask" / "val" / "val_class",
        help="COCOMask 验证集目录",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "Datasets" / "images",
        help="输出目录（默认 Datasets/images，供 test_embedding）",
    )
    p.add_argument("--n", type=int, default=1000, help="随机抽取张数（默认 1000）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument(
        "--host_size",
        type=int,
        default=128,
        help="左宿主边宽；None 则用宽/2。默认 128（对齐 PIMoG）",
    )
    p.add_argument(
        "--clear_out",
        action="store_true",
        help="导出前清空输出目录中的图像文件",
    )
    p.add_argument(
        "--extend_wmat",
        action="store_true",
        help="若 w.mat 行数 < N，则备份并补齐随机消息行（嵌入必需）",
    )
    p.add_argument(
        "--wmat",
        type=Path,
        default=root / "results" / "WatermarkMatrix" / "w.mat",
        help="水印矩阵路径",
    )
    args = p.parse_args()

    src: Path = args.src
    out: Path = args.out
    n: int = int(args.n)
    if n <= 0:
        raise SystemExit("--n 必须 > 0")

    if not src.is_dir():
        raise SystemExit(f"源目录不存在: {src}")

    all_imgs = list_images(src)
    if len(all_imgs) < n:
        raise SystemExit(f"源图仅 {len(all_imgs)} 张，不足 {n} 张")

    rng = random.Random(args.seed)
    chosen = rng.sample(all_imgs, n)

    out.mkdir(parents=True, exist_ok=True)
    if args.clear_out:
        removed = 0
        for f in out.iterdir():
            if f.is_file() and f.suffix.lower() in EXTS:
                f.unlink()
                removed += 1
        print(f"[INFO] 已清空输出目录图像 {removed} 个 → {out}")

    manifest = out / "manifest.txt"
    ok = 0
    with open(manifest, "w", encoding="utf-8") as mf:
        mf.write(f"# seed={args.seed}  n={n}  src={src}\n")
        mf.write("# index\tsource\n")
        for i, src_path in enumerate(chosen):
            try:
                with Image.open(src_path) as im:
                    img = im.convert("RGB")
                    host = crop_left_host(img, args.host_size)
            except (OSError, ValueError) as e:
                print(f"[WARN] 跳过 {src_path.name}: {e}")
                continue
            dst = out / f"{i}.png"
            try:
                host.save(dst)
            except OSError as e:
                print(f"[WARN] 写入失败 {dst}: {e}")
                continue
            mf.write(f"{i}\t{src_path.name}\n")
            ok += 1

    print(f"[OK] 已导出 {ok}/{n} 张 → {out}")
    print(f"[OK] 对照表 → {manifest}")
    if ok != n:
        raise SystemExit(f"实际成功 {ok} 张，少于请求 {n}")

    # 检查 / 扩展 w.mat
    if args.wmat.is_file():
        w = np.asarray(scio.loadmat(str(args.wmat))["w"])
        rows = w.shape[0]
        if rows < ok:
            msg = (
                f"[WARN] w.mat 仅 {rows} 行，而宿主编号到 {ok - 1}；"
                f"test_embedding 对 ≥{rows} 的文件名会越界。"
            )
            if args.extend_wmat:
                print(msg)
                maybe_extend_wmat(args.wmat, ok, args.seed)
            else:
                print(msg)
                print("      请加 --extend_wmat 自动补齐，或自行扩展 w.mat。")
        else:
            print(f"[OK] w.mat 行数 {rows} ≥ {ok}，可直接嵌入")
    else:
        print(f"[WARN] 未找到 {args.wmat}，嵌入前请准备水印矩阵")


if __name__ == "__main__":
    if os.name == "nt":
        try:
            import sys

            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
