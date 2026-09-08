#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_host_ssim.py
===================
对宿主图分别用 **mask_99（PIMoG）** 与 **ScreenShootingMB_best（Ours）**
做水印嵌入，再相对原宿主计算 **SSIM**，并输出对比报告。

流程与 ``verify_host_psnr.py`` 相同，指标改为结构相似性。

SSIM
----
  - 图像从 [-1,1] 映到 [0,1]
  - 对 RGB 三通道分别算 SSIM 后取平均（与常见水印论文做法一致）
  - 优先 ``skimage.metrics.structural_similarity``；若无 skimage 则用内置实现
  - 窗口 7×7，``data_range=1.0``

用法
----
  python verify_host_ssim.py
  python verify_host_ssim.py --limit 100 --batch_size 32
  python verify_host_ssim.py --out_dir results/ssim_embed_report
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.autograd import Variable

ROOT = Path(__file__).resolve().parent.parent  # repo root
TOOLS = Path(__file__).resolve().parent
# 必须让 tools/ 优先于仓库根，否则会 import 到根目录兼容 shim
# （旧 shim 在 import 时就会 runpy 跑 PSNR，导致菜单 7/8 全变成 PSNR）
for p in (str(ROOT), str(TOOLS)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from data_loader import get_loader
from verify_host_psnr import (  # 复用同目录加载逻辑
    ModelSpec,
    build_specs_from_ckpts,
    load_encoder,
    resolve_default_weights,
)

try:
    from skimage.metrics import structural_similarity as skimage_ssim

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    skimage_ssim = None  # type: ignore

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    cv2 = None  # type: ignore


def _ssim_gray_fast(x: np.ndarray, y: np.ndarray, win: int = 7) -> float:
    """单通道 SSIM，x,y ∈ [0,1]。优先 cv2 高斯滤波，否则均匀窗。"""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    if _HAS_CV2:
        # 与常见实现接近的高斯窗
        mu_x = cv2.GaussianBlur(x64, (win, win), 1.5)
        mu_y = cv2.GaussianBlur(y64, (win, win), 1.5)
        sigma_x2 = cv2.GaussianBlur(x64 * x64, (win, win), 1.5) - mu_x * mu_x
        sigma_y2 = cv2.GaussianBlur(y64 * y64, (win, win), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x64 * y64, (win, win), 1.5) - mu_x * mu_y
    else:
        k = np.ones((win, win), dtype=np.float64) / float(win * win)
        pad = win // 2

        def conv2(img: np.ndarray) -> np.ndarray:
            padded = np.pad(img, pad, mode="reflect")
            from numpy.lib.stride_tricks import sliding_window_view

            windows = sliding_window_view(padded, (win, win))
            return np.einsum("ijkl,kl->ij", windows, k)

        mu_x = conv2(x64)
        mu_y = conv2(y64)
        sigma_x2 = conv2(x64 * x64) - mu_x * mu_x
        sigma_y2 = conv2(y64 * y64) - mu_y * mu_y
        sigma_xy = conv2(x64 * y64) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


def ssim_rgb01(a01: np.ndarray, b01: np.ndarray) -> float:
    """a,b: (H,W,3) float [0,1] → 三通道 SSIM 平均。"""
    if a01.shape != b01.shape:
        raise ValueError(f"shape mismatch {a01.shape} vs {b01.shape}")
    if a01.ndim != 3 or a01.shape[2] != 3:
        raise ValueError(f"expected HWC RGB, got {a01.shape}")

    vals = []
    for c in range(3):
        xc = a01[:, :, c]
        yc = b01[:, :, c]
        if _HAS_SKIMAGE:
            try:
                vals.append(
                    float(
                        skimage_ssim(xc, yc, data_range=1.0, win_size=7)
                    )
                )
            except TypeError:
                # 旧版 skimage
                vals.append(
                    float(
                        skimage_ssim(xc, yc, data_range=1.0, win_size=7)
                    )
                )
        else:
            vals.append(_ssim_gray_fast(xc, yc, win=7))
    return float(np.mean(vals))


def tensor_neg11_to_hwc01(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) in [-1,1] → (H,W,C) in [0,1]."""
    arr = t.detach().float().cpu().numpy()
    arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    return np.transpose(arr, (1, 2, 0))


def ssim_backend_name() -> str:
    if _HAS_SKIMAGE:
        return "skimage"
    if _HAS_CV2:
        return "opencv_gaussian"
    return "numpy_uniform"


@dataclass
class SsimRunResult:
    spec: ModelSpec
    indices: list[int] = field(default_factory=list)
    ssims: list[float] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.ssims)

    def summary(self) -> dict:
        arr = np.asarray(self.ssims, dtype=np.float64)
        if arr.size == 0:
            return {
                "n": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "std": float("nan"),
            }
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
        }


@torch.no_grad()
def embed_and_measure_ssim(
    spec: ModelSpec,
    loader,
    device: torch.device,
    limit: Optional[int] = None,
) -> SsimRunResult:
    net = load_encoder(
        spec.weight_path, spec.embed_strength, device, distortion=spec.distortion
    )
    result = SsimRunResult(spec=spec)
    t0 = time.perf_counter()
    seen = 0

    for data, m, num in loader:
        inputs = Variable(data).to(device)
        m = Variable(m.float()).to(device)
        nums = num.detach().cpu().numpy().reshape(-1)
        encoded = net.encode(inputs, m)

        b = inputs.shape[0]
        for i in range(b):
            if limit is not None and seen >= limit:
                break
            host = tensor_neg11_to_hwc01(inputs[i])
            enc = tensor_neg11_to_hwc01(encoded[i])
            val = ssim_rgb01(host, enc)
            result.indices.append(int(nums[i]))
            result.ssims.append(val)
            seen += 1
        if limit is not None and seen >= limit:
            break

    result.seconds = time.perf_counter() - t0
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_csv(path: Path, results: list[SsimRunResult]) -> None:
    by_idx: dict[int, dict[str, float]] = {}
    tags: list[str] = []
    for r in results:
        tags.append(r.spec.tag)
        for idx, s in zip(r.indices, r.ssims):
            by_idx.setdefault(idx, {})[r.spec.tag] = s

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index"] + [f"ssim_{t}" for t in tags]
    delta_key = None
    if len(tags) == 2:
        delta_key = f"delta_{tags[1]}_minus_{tags[0]}"
        fieldnames.append(delta_key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx in sorted(by_idx):
            row: dict = {"index": idx}
            for t in tags:
                row[f"ssim_{t}"] = f"{by_idx[idx].get(t, float('nan')):.8f}"
            if delta_key is not None:
                a = by_idx[idx].get(tags[0])
                b = by_idx[idx].get(tags[1])
                if a is not None and b is not None:
                    row[delta_key] = f"{(b - a):.8f}"
            w.writerow(row)


def write_markdown_report(
    path: Path,
    results: list[SsimRunResult],
    host_dir: Path,
    image_size: int,
    device: str,
    backend: str,
) -> None:
    lines = [
        "# 宿主嵌入 SSIM 对比报告",
        "",
        f"- 生成时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- 宿主目录: `{host_dir}`",
        f"- 图像尺寸: `{image_size}×{image_size}`",
        f"- 设备: `{device}`",
        f"- SSIM 后端: `{backend}`（RGB 三通道平均，data_range=1.0）",
        "",
        "## 模型与权重",
        "",
        "| 标签 | 说明 | 权重 | embed_strength | 样本数 | 耗时(s) |",
        "|:---|:---|:---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r.spec.tag}` | {r.spec.label} | `{r.spec.weight_path}` | "
            f"{r.spec.embed_strength:g} | {r.n} | {r.seconds:.2f} |"
        )
    lines += [
        "",
        "## 汇总",
        "",
        "| 模型 | n | 平均 SSIM | 中位 | 最小 | 最大 | 标准差 |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        s = r.summary()
        lines.append(
            f"| {r.spec.label} (`{r.spec.tag}`) | {s['n']} | "
            f"{s['mean']:.6f} | {s['median']:.6f} | "
            f"{s['min']:.6f} | {s['max']:.6f} | {s['std']:.6f} |"
        )

    if len(results) == 2 and results[0].n and results[1].n:
        a_r, b_r = results[0], results[1]
        a = a_r.summary()["mean"]
        b = b_r.summary()["mean"]
        lines += [
            "",
            f"## 对比（{b_r.spec.label} − {a_r.spec.label}）",
            "",
            f"- 平均 SSIM 差: **{b - a:+.6f}**",
        ]
        map_a = dict(zip(a_r.indices, a_r.ssims))
        map_b = dict(zip(b_r.indices, b_r.ssims))
        common = sorted(set(map_a) & set(map_b))
        if common:
            deltas = np.asarray([map_b[i] - map_a[i] for i in common], dtype=np.float64)
            lines += [
                f"- 配对样本数: {len(common)}",
                f"- 配对差均值: **{deltas.mean():+.6f}**",
                f"- 配对差中位: {np.median(deltas):+.6f}",
                f"- 模型2 更高 / 更低 / 持平: "
                f"{int((deltas > 1e-8).sum())} / {int((deltas < -1e-8).sum())} / "
                f"{int((np.abs(deltas) <= 1e-8).sum())}",
            ]

    lines += [
        "",
        "## 说明",
        "",
        "- 只衡量**嵌入不可见性**（Encoded vs Host），不含噪声层 / 拍屏。",
        "- SSIM ∈ [-1,1]，越接近 1 结构越相似；`embed_strength` 优先读 `.meta.txt`。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    root = ROOT
    default_99, default_ours = resolve_default_weights(root)
    p = argparse.ArgumentParser(
        description="用选定的 1~2 个权重嵌入宿主并测 SSIM（可对照）"
    )
    p.add_argument(
        "--host_dir",
        type=Path,
        default=root / "Datasets" / "images",
    )
    p.add_argument(
        "--wmat",
        type=Path,
        default=root / "results" / "WatermarkMatrix" / "w.mat",
    )
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--ckpt",
        type=Path,
        action="append",
        default=None,
        help="权重 .pth（可重复 1~2 次）；未给则默认 mask_99 vs MB_best",
    )
    p.add_argument("--ckpt99", type=Path, default=default_99)
    p.add_argument("--ckpt_ours", type=Path, default=default_ours)
    p.add_argument("--strength99", type=float, default=None)
    p.add_argument("--strength_ours", type=float, default=None)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=root / "results" / "ssim_embed_report",
    )
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> int:
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    backend = ssim_backend_name()

    if not args.host_dir.is_dir():
        print(f"错误: 宿主目录不存在: {args.host_dir}")
        return 1
    if not args.wmat.is_file():
        print(f"错误: 找不到 w.mat: {args.wmat}")
        return 1

    if args.ckpt:
        ckpts = list(args.ckpt)
        if not (1 <= len(ckpts) <= 2):
            print("错误: --ckpt 只能给 1 或 2 个路径")
            return 2
        overrides = [args.strength99, args.strength_ours][: len(ckpts)]
        try:
            specs = build_specs_from_ckpts(ckpts, strength_overrides=overrides)
        except ValueError as e:
            print(f"错误: {e}")
            return 2
    else:
        specs = build_specs_from_ckpts(
            [args.ckpt99, args.ckpt_ours],
            strength_overrides=[args.strength99, args.strength_ours],
        )

    title = (
        "宿主嵌入 SSIM：单模型"
        if len(specs) == 1
        else f"宿主嵌入 SSIM：{specs[0].label} vs {specs[1].label}"
    )
    print("=" * 64)
    print(title)
    print("=" * 64)
    print(f"宿主:   {args.host_dir.resolve()}")
    print(f"w.mat:  {args.wmat.resolve()}")
    print(
        f"尺寸:   {args.image_size}  batch={args.batch_size}  "
        f"device={device}  ssim={backend}"
    )
    for sp in specs:
        print(
            f"[{sp.tag}] {sp.label}\n"
            f"       weights={sp.weight_path}\n"
            f"       embed_strength={sp.embed_strength:g}"
        )
    print("-" * 64)

    results: list[SsimRunResult] = []
    for sp in specs:
        print(f"\n>>> 嵌入并测 SSIM: {sp.label} …")
        loader = get_loader(
            image_dir=str(args.host_dir),
            image_size=args.image_size,
            batch_size=args.batch_size,
            dataset="test_embedding",
            mode="test_embedding",
            num_workers=args.num_workers,
            w_path=str(args.wmat),
            lite=False,
        )
        try:
            r = embed_and_measure_ssim(sp, loader, device, limit=args.limit)
        except FileNotFoundError as e:
            print(f"错误: {e}")
            return 1
        results.append(r)
        s = r.summary()
        print(
            f"    n={s['n']}  mean={s['mean']:.6f}  "
            f"median={s['median']:.6f}  min={s['min']:.6f}  "
            f"max={s['max']:.6f}  ({r.seconds:.1f}s)"
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"ssim_report_{stamp}.md"
    csv_path = out_dir / f"ssim_per_image_{stamp}.csv"
    latest_md = out_dir / "ssim_report_latest.md"
    latest_csv = out_dir / "ssim_per_image_latest.csv"

    write_markdown_report(
        md_path, results, args.host_dir, args.image_size, str(device), backend
    )
    write_csv(csv_path, results)
    write_markdown_report(
        latest_md, results, args.host_dir, args.image_size, str(device), backend
    )
    write_csv(latest_csv, results)

    print("\n" + "=" * 64)
    print("汇总")
    print("=" * 64)
    for r in results:
        s = r.summary()
        print(f"  {r.spec.label:40s}  mean SSIM = {s['mean']:.6f}  (n={s['n']})")
    if len(results) == 2 and results[0].n and results[1].n:
        d = results[1].summary()["mean"] - results[0].summary()["mean"]
        print(
            f"  平均差 ({results[1].spec.tag} - {results[0].spec.tag}) = {d:+.6f}"
        )
    print("-" * 64)
    print(f"报告: {md_path}")
    print(f"逐图: {csv_path}")
    print(f"最新: {latest_md}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
