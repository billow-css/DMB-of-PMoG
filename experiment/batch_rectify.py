#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/batch_rectify.py
===========================
快速批量透视矫正（默认入口）。

相对 ``legacy/batch_rectify_legacy.py``：
  - 缩略图检测 + 精简 ArUco（见 markers_fast.py）
  - 多进程并行；默认 reduce=1/2
  - 权衡：内容可抖糊，几何畸形进 failed（白垫门槛更高）

兼容：``batch_rectify_fast.py`` 仍可运行 / import。

用法
----
  python batch_rectify.py
  python batch_rectify.py --src captures/session_xxx --out ../Datasets/Recover/capture
  python batch_rectify.py --workers 8 --save_debug
  python batch_rectify.py --no_fallback   # 只信 ArUco
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from markers_fast import detect_content_quad_fast, draw_debug, warp_content

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def pick_folder(title: str, initial: Optional[Path] = None) -> Optional[Path]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    init = (
        str(initial)
        if initial and initial.is_dir()
        else str(Path(__file__).resolve().parent / "captures")
    )
    if not Path(init).is_dir():
        init = str(Path(__file__).resolve().parent)
    chosen = filedialog.askdirectory(parent=root, title=title, initialdir=init)
    root.destroy()
    return Path(chosen) if chosen else None


def parse_index(path: Path) -> Optional[int]:
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
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            if p.name.startswith("."):
                continue
            if p.parent.name in ("failed", "debug", "debug_fast"):
                continue
            if "_debug" in p.stem or "_re" in p.stem:
                continue
            files.append(p)

    def key(p: Path):
        idx = parse_index(p)
        return (0, idx) if idx is not None else (1, p.as_posix())

    files.sort(key=key)
    return files


def _imread_fast(path: Path, reduce: int) -> Optional[Any]:
    """
    reduce: 1=原图, 2/4/8 = JPEG 降采样读取（OpenCV IMREAD_REDUCED_*）。
    降采样失败（损坏/占位图）时回退原图。
    """
    flag = cv2.IMREAD_COLOR
    if reduce == 2:
        flag = cv2.IMREAD_REDUCED_COLOR_2
    elif reduce == 4:
        flag = cv2.IMREAD_REDUCED_COLOR_4
    elif reduce == 8:
        flag = cv2.IMREAD_REDUCED_COLOR_8
    img = cv2.imread(str(path), flag)
    if img is None and reduce != 1:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def process_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """进程池 worker：单张处理。"""
    path = Path(job["path"])
    out_dir = Path(job["out_dir"])
    failed_dir = Path(job["failed_dir"])
    debug_dir = Path(job["debug_dir"]) if job.get("debug_dir") else None
    out_size = int(job["out_size"])
    dry_run = bool(job["dry_run"])
    min_confidence = float(job["min_confidence"])
    max_side = int(job["max_side"])
    reduce = int(job["reduce"])
    try_sharp = bool(job["try_sharp"])

    t0 = time.perf_counter()
    idx = parse_index(path)
    if idx is None:
        return {
            "file": str(path),
            "index": None,
            "ok": 0,
            "reason": "cannot_parse_index",
            "ms": 0.0,
        }

    bgr = _imread_fast(path, reduce)
    if bgr is None:
        return {
            "file": str(path),
            "index": idx,
            "ok": 0,
            "reason": "imread_fail",
            "ms": (time.perf_counter() - t0) * 1000,
        }

    quad, info = detect_content_quad_fast(
        bgr,
        max_side=max_side,
        min_confidence=min_confidence,
        try_sharp=try_sharp,
        allow_fallback=bool(job.get("allow_fallback", True)),
        allow_white_frame=bool(job.get("allow_white_frame", False)),
        min_fallback_confidence=float(job.get("min_fallback_confidence", 0.58)),
        min_aruco_markers=int(job.get("min_aruco_markers", 3)),
    )
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
        return {
            "file": str(path),
            "index": idx,
            "ok": 0,
            "reason": json.dumps(info, ensure_ascii=False),
            "ms": (time.perf_counter() - t0) * 1000,
        }

    warped = warp_content(bgr, quad, out_size=out_size)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{idx}.png"), warped)
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            vis = draw_debug(bgr, quad, info)
            cv2.imwrite(str(debug_dir / f"{idx:06d}_ok.jpg"), vis)

    reason = (
        f"ok conf={info.get('confidence')} n={info.get('n_required')} "
        f"method={info.get('method')} ids={info.get('ids')} "
        f"via={info.get('variant') or info.get('fallback')}"
    )
    return {
        "file": str(path),
        "index": idx,
        "ok": 1,
        "reason": reason,
        "ms": (time.perf_counter() - t0) * 1000,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="快速批量 ArUco 透视矫正 → 128×128")
    p.add_argument("--src", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--failed_dir", type=Path, default=None)
    p.add_argument("--out_size", type=int, default=128)
    p.add_argument(
        "--min_confidence",
        type=float,
        default=0.40,
        help="ArUco 最低置信度（默认 0.40）",
    )
    p.add_argument(
        "--min_fallback_confidence",
        type=float,
        default=0.58,
        help="白垫兜底门槛（默认 0.58；防畸形）",
    )
    p.add_argument(
        "--no_fallback",
        action="store_true",
        help="禁用白垫兜底（只信 ArUco）",
    )
    p.add_argument(
        "--allow_white_frame",
        action="store_true",
        help="启用白框兜底（默认关：最易出畸形图）",
    )
    p.add_argument(
        "--min_aruco_markers",
        type=int,
        default=3,
        choices=(2, 3, 4),
        help="最少接受的定位点数（默认 3；2 点易畸变）",
    )
    p.add_argument(
        "--max_side",
        type=int,
        default=1600,
        help="检测前最长边缩放到此（默认 1600）",
    )
    p.add_argument(
        "--reduce",
        type=int,
        default=2,
        choices=(1, 2, 4, 8),
        help="JPEG 读取降采样：1=原图 2/4/8=1/2/1/4/1/8（默认 2，兼顾精度）",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="并行进程数（默认 CPU-1）",
    )
    p.add_argument(
        "--no_sharp",
        action="store_true",
        help="跳过锐化二次 ArUco（更快，糊图可能稍差）",
    )
    p.add_argument("--save_debug", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no_dialog", action="store_true")
    args = p.parse_args()
    min_confidence = float(max(0.0, min(1.0, args.min_confidence)))

    if args.src is None:
        if args.no_dialog:
            print("错误: --no_dialog 时必须指定 --src")
            return 2
        print("请选择拍屏原图文件夹…")
        src = pick_folder("选择拍屏原图文件夹", initial=root / "captures")
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
            src = pick_folder("选择拍屏原图文件夹", initial=root / "captures")
            if src is None:
                print("已取消。")
                return 0
        elif not src.exists():
            print(f"源不存在: {args.src}")
            return 1
    src = src.resolve()

    default_out = (root.parent / "Datasets" / "Recover" / "capture").resolve()
    if args.out is None and not args.no_dialog and not args.dry_run:
        print("请选择输出文件夹（取消则用默认 Recover/capture）…")
        picked = pick_folder("选择输出文件夹", initial=default_out.parent)
        out_dir = picked.resolve() if picked is not None else default_out
    elif args.out is None:
        out_dir = default_out
    else:
        out_dir = args.out if args.out.is_absolute() else (root.parent / args.out)
        out_dir = out_dir.resolve()

    failed_dir = args.failed_dir or (
        src / "failed_fast" if src.is_dir() else src.parent / "failed_fast"
    )
    debug_dir = None
    if args.save_debug:
        debug_dir = src / "debug_fast" if src.is_dir() else src.parent / "debug_fast"

    files = list_images(src)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"未找到图像: {src}")
        return 1

    workers = max(1, int(args.workers))
    print(f"源: {src}")
    print(f"出: {out_dir}")
    min_fb = float(max(0.0, min(1.0, args.min_fallback_confidence)))
    print(
        f"共 {len(files)} 张 | workers={workers} | reduce=1/{args.reduce} "
        f"| max_side={args.max_side} | aruco>={min_confidence:.2f} "
        f"| fallback>={min_fb:.2f} allow_fallback={not args.no_fallback}"
    )

    jobs = [
        {
            "path": str(fp),
            "out_dir": str(out_dir),
            "failed_dir": str(failed_dir),
            "debug_dir": str(debug_dir) if debug_dir else None,
            "out_size": args.out_size,
            "dry_run": args.dry_run,
            "min_confidence": min_confidence,
            "min_fallback_confidence": min_fb,
            "allow_fallback": not args.no_fallback,
            "allow_white_frame": bool(args.allow_white_frame),
            "min_aruco_markers": int(args.min_aruco_markers),
            "max_side": args.max_side,
            "reduce": args.reduce,
            "try_sharp": not args.no_sharp,
        }
        for fp in files
    ]

    t_all = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    ok = 0
    fail = 0
    done = 0

    if workers == 1:
        results_iter = (process_one_job(j) for j in jobs)
        for r in results_iter:
            done += 1
            if r["ok"]:
                ok += 1
                tag = "OK"
            else:
                fail += 1
                tag = "FAIL"
            print(
                f"[{done}/{len(files)}] {tag} idx={r['index']} "
                f"{Path(r['file']).name} | {r['ms']:.0f}ms | {r['reason']}"
            )
            rows.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_one_job, j): j for j in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                if r["ok"]:
                    ok += 1
                    tag = "OK"
                else:
                    fail += 1
                    tag = "FAIL"
                print(
                    f"[{done}/{len(files)}] {tag} idx={r['index']} "
                    f"{Path(r['file']).name} | {r['ms']:.0f}ms | {r['reason']}"
                )
                rows.append(r)

    rows.sort(
        key=lambda r: (0, r["index"]) if r["index"] is not None else (1, r["file"])
    )

    elapsed = time.perf_counter() - t_all
    report = (
        (src / "rectify_fast_report.csv")
        if src.is_dir()
        else (src.parent / "rectify_fast_report.csv")
    )
    if not args.dry_run:
        report.parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["file", "index", "ok", "reason", "ms"]
            )
            w.writeheader()
            w.writerows(rows)

    rate = 100.0 * ok / max(len(files), 1)
    ips = len(files) / max(elapsed, 1e-6)
    print("=" * 48)
    print(
        f"成功 {ok}  失败 {fail}  成功率 {rate:.1f}%  "
        f"总耗时 {elapsed:.1f}s  ({ips:.1f} 张/s)"
    )
    if fail and not args.dry_run:
        print(f"失败原图 → {failed_dir}")
    if not args.dry_run:
        print(f"矫正结果 → {out_dir}")
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
