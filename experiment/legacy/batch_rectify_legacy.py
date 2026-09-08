#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/legacy/batch_rectify_legacy.py
========================================
旧版批量透视矫正（全图 ArUco + 白框兜底）。

日常请用上级目录的 ``batch_rectify.py``（快速多进程版）。
本文件仅保留对照 / 回退。

用法
----
  python legacy/batch_rectify_legacy.py
  python legacy/batch_rectify_legacy.py --src ../captures/session_xxx --out ../../Datasets/Recover/capture
  python legacy/batch_rectify_legacy.py --min_confidence 0.15 --save_debug
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

# legacy/ 的上一级才是 experiment/（markers 所在目录）
_EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP))

from markers import detect_content_quad, draw_debug, warp_content

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def pick_folder(title: str, initial: Optional[Path] = None) -> Optional[Path]:
    """弹窗选择文件夹；取消返回 None。"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    init = (
        str(initial)
        if initial and initial.is_dir()
        else str(_EXP / "captures")
    )
    if not Path(init).is_dir():
        init = str(_EXP)
    chosen = filedialog.askdirectory(parent=root, title=title, initialdir=init)
    root.destroy()
    if not chosen:
        return None
    return Path(chosen)


def parse_index(path: Path) -> Optional[int]:
    """
    000012.JPG → 12；12.png → 12；IMG_0003 → 3（取末段数字）。
    """
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", stem)
    if m:
        return int(m.group(1))
    return None


def list_images(src: Path) -> List[Path]:
    files: List[Path] = []
    if src.is_file():
        files = [src]
    else:
        for p in sorted(src.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
                # 跳过调试输出
                if p.parent.name in ("failed", "debug"):
                    continue
                if "_debug" in p.stem or "_re" in p.stem:
                    continue
                files.append(p)

    def key(p: Path):
        idx = parse_index(p)
        return (0, idx) if idx is not None else (1, p.as_posix())

    files.sort(key=key)
    return files


def process_one(
    path: Path,
    out_dir: Path,
    failed_dir: Path,
    debug_dir: Optional[Path],
    out_size: int,
    dry_run: bool,
    min_confidence: float = 0.25,
) -> Tuple[bool, str, Optional[int]]:
    idx = parse_index(path)
    if idx is None:
        return False, "cannot_parse_index", None

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return False, "imread_fail", idx

    quad, info = detect_content_quad(bgr, min_confidence=min_confidence)
    if quad is None:
        if not dry_run:
            failed_dir.mkdir(parents=True, exist_ok=True)
            dst = failed_dir / path.name
            if not dst.exists():
                import shutil

                shutil.copy2(path, dst)
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                vis = draw_debug(bgr, None, info)
                cv2.imwrite(str(debug_dir / f"{idx:06d}_fail.jpg"), vis)
        return False, json.dumps(info, ensure_ascii=False), idx

    warped = warp_content(bgr, quad, out_size=out_size)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{idx}.png"
        cv2.imwrite(str(out_path), warped)
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            vis = draw_debug(bgr, quad, info)
            cv2.imwrite(str(debug_dir / f"{idx:06d}_ok.jpg"), vis)
            cv2.imwrite(str(debug_dir / f"{idx:06d}_warp.png"), warped)

    reason = (
        f"ok conf={info.get('confidence')} n={info.get('n_required')} "
        f"method={info.get('method')} ids={info.get('ids')} "
        f"via={info.get('variant') or info.get('fallback')}"
    )
    return True, reason, idx


def main() -> int:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="批量 ArUco 透视矫正 → 128×128")
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help="拍屏原图目录；省略则启动时弹窗选择",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录；省略则弹窗选择（取消则用 Datasets/Recover/capture）",
    )
    p.add_argument(
        "--failed_dir",
        type=Path,
        default=None,
        help="失败原图拷贝目录（默认 <src>/failed）",
    )
    p.add_argument("--out_size", type=int, default=128)
    p.add_argument(
        "--min_confidence",
        type=float,
        default=0.25,
        help="最低置信度 [0,1]：缺角/模糊时 ≥ 此值仍矫正（默认 0.25；越低越宽松）",
    )
    p.add_argument("--save_debug", action="store_true", help="保存检测可视化")
    p.add_argument("--dry_run", action="store_true", help="只统计成功率，不写文件")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试）")
    p.add_argument(
        "--no_dialog",
        action="store_true",
        help="不弹窗（必须同时提供 --src）",
    )
    args = p.parse_args()
    min_confidence = float(max(0.0, min(1.0, args.min_confidence)))

    # ---- 源目录：默认弹窗 ----
    if args.src is None:
        if args.no_dialog:
            print("错误: --no_dialog 时必须指定 --src")
            return 2
        print("请选择拍屏原图文件夹…")
        src = pick_folder(
            "选择拍屏原图文件夹（如 session_xxx）",
            initial=root / "captures",
        )
        if src is None:
            print("已取消。")
            return 0
    else:
        src = args.src if args.src.is_absolute() else (Path.cwd() / args.src)
        if not src.exists():
            alt = root / args.src
            if alt.exists():
                src = alt
        if not src.exists() and not args.no_dialog:
            print(f"源不存在: {args.src}，请重新选择…")
            src = pick_folder(
                "选择拍屏原图文件夹",
                initial=root / "captures",
            )
            if src is None:
                print("已取消。")
                return 0
        elif not src.exists():
            print(f"源不存在: {args.src}")
            return 1

    src = src.resolve()

    # ---- 输出目录：未指定时也弹窗，取消则用默认 ----
    default_out = (root.parent / "Datasets" / "Recover" / "capture").resolve()
    if args.out is None and not args.no_dialog and not args.dry_run:
        print("请选择矫正结果输出文件夹（取消则使用默认 Recover/capture）…")
        picked = pick_folder("选择输出文件夹（矫正后的 128×128）", initial=default_out.parent)
        out_dir = picked.resolve() if picked is not None else default_out
    elif args.out is None:
        out_dir = default_out
    else:
        out_dir = args.out if args.out.is_absolute() else (root.parent / args.out)
        out_dir = out_dir.resolve()

    failed_dir = args.failed_dir or (src / "failed" if src.is_dir() else src.parent / "failed")
    debug_dir = (src / "debug" if src.is_dir() else src.parent / "debug") if args.save_debug else None

    files = list_images(src)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"未找到图像: {src}")
        return 1

    print(f"源: {src}")
    print(f"出: {out_dir}")
    print(f"共 {len(files)} 张 | min_confidence={min_confidence:.2f} | dry_run={args.dry_run}")

    ok = 0
    fail = 0
    rows = []
    for i, path in enumerate(files, 1):
        success, reason, idx = process_one(
            path,
            out_dir,
            failed_dir,
            debug_dir,
            args.out_size,
            args.dry_run,
            min_confidence=min_confidence,
        )
        if success:
            ok += 1
            tag = "OK"
        else:
            fail += 1
            tag = "FAIL"
        print(f"[{i}/{len(files)}] {tag} idx={idx} {path.name} | {reason}")
        rows.append(
            {
                "file": str(path),
                "index": idx,
                "ok": int(success),
                "reason": reason,
            }
        )

    report = (src / "rectify_report.csv") if src.is_dir() else (src.parent / "rectify_report.csv")
    if not args.dry_run:
        report.parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "index", "ok", "reason"])
            w.writeheader()
            w.writerows(rows)

    rate = 100.0 * ok / max(len(files), 1)
    print("=" * 48)
    print(f"成功 {ok}  失败 {fail}  成功率 {rate:.1f}%")
    if fail and not args.dry_run:
        print(f"失败原图 → {failed_dir}")
        print("可对 failed/ 人工 MATLAB 补点，或重新拍摄过糊样本")
    if not args.dry_run:
        print(f"矫正结果 → {out_dir}  (命名 {{index}}.png，可直接 test_accuracy)")
        print(f"报告 → {report}")
    return 0 if ok > 0 else 2


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
