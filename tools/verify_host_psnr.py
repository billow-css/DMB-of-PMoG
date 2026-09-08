#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_host_psnr.py
===================
对宿主图分别用 **mask_99（原 PIMoG）** 与 **ScreenShootingMB_best（Ours）**
做水印嵌入，再相对原宿主计算 PSNR，并输出对比报告。

流程
----
  1. 用 ``data_loader.get_loader(..., dataset=test_embedding)`` 加载宿主 + w.mat 消息
  2. 分别加载两套 Encoder 权重，调用 ``Encoder_Decoder.encode`` 嵌入
  3. 用与 ``solver._psnr_neg1_1_mean`` 相同的公式算逐图 PSNR
  4. 写出控制台摘要 + Markdown / CSV 报告

PSNR（图像在 [-1,1]）
-------------------
  MSE = mean((Encoded - Host)^2)
  PSNR = 10 * log10(4 / MSE)     # 极小 MSE → 99.0

用法
----
  python verify_host_psnr.py
  python verify_host_psnr.py --limit 100 --batch_size 32
  python verify_host_psnr.py --out_dir results/psnr_report
  python verify_host_psnr.py --ckpt99 models/ScreenShooting/Encoder_Decoder_Model_mask_99.pth \\
                             --ckpt_ours models/Encoder_Decoder_Model_mask_ScreenShootingMB_best.pth
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import get_loader
from model import Encoder_Decoder


@dataclass
class ModelSpec:
    tag: str
    label: str
    weight_path: Path
    embed_strength: float
    distortion: str = "Identity"  # Noiser 不参与 encode；Identity 最轻


@dataclass
class RunResult:
    spec: ModelSpec
    indices: list[int] = field(default_factory=list)
    psnrs: list[float] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.psnrs)

    def summary(self) -> dict:
        arr = np.asarray(self.psnrs, dtype=np.float64)
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


def read_embed_strength_meta(weight_path: Path, default: float = 0.0) -> float:
    """优先读 ``*.meta.txt`` 里的 embed_strength=。"""
    meta = weight_path.with_suffix(weight_path.suffix + ".meta.txt")
    # 也试  Encoder_xxx.pth → Encoder_xxx.meta.txt
    alt = weight_path.with_name(weight_path.stem + ".meta.txt")
    for cand in (meta, alt):
        if not cand.is_file():
            continue
        try:
            for line in cand.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("embed_strength="):
                    return float(line.split("=", 1)[1].strip())
        except Exception:
            continue
    return float(default)


def resolve_default_weights(root: Path) -> tuple[Path, Path]:
    ckpt99 = root / "models" / "ScreenShooting" / "Encoder_Decoder_Model_mask_99.pth"
    ours = root / "models" / "Encoder_Decoder_Model_mask_ScreenShootingMB_best.pth"
    if not ours.is_file():
        alt = (
            root
            / "models"
            / "ScreenShootingMB"
            / "Encoder_Decoder_Model_mask_ScreenShootingMB_best.pth"
        )
        if alt.is_file():
            ours = alt
    return ckpt99, ours


def load_encoder(
    weight_path: Path,
    embed_strength: float,
    device: torch.device,
    distortion: str = "Identity",
) -> Encoder_Decoder:
    if not weight_path.is_file():
        raise FileNotFoundError(f"权重不存在: {weight_path}")
    net = Encoder_Decoder(distortion, embed_strength=embed_strength)
    net = net.to(device)
    try:
        state = torch.load(str(weight_path), map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(str(weight_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # 去掉 DataParallel 的 module. 前缀
    if any(str(k).startswith("module.") for k in state):
        state = {k[7:] if str(k).startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] 缺失键 {len(missing)} 个（Noiser 等可忽略）")
    if unexpected:
        print(f"  [warn] 多余键 {len(unexpected)} 个")
    net.eval()
    return net


@torch.no_grad()
def embed_and_measure_psnr(
    spec: ModelSpec,
    loader,
    device: torch.device,
    limit: Optional[int] = None,
) -> RunResult:
    net = load_encoder(
        spec.weight_path, spec.embed_strength, device, distortion=spec.distortion
    )
    result = RunResult(spec=spec)
    t0 = time.perf_counter()
    seen = 0

    for data, m, num in loader:
        inputs = Variable(data).to(device)
        m = Variable(m.float()).to(device)
        nums = num.detach().cpu().numpy().reshape(-1)

        encoded = net.encode(inputs, m)
        mse = torch.mean((encoded.detach() - inputs.detach()) ** 2, dim=(1, 2, 3))
        psnr = torch.where(
            mse < 1e-10,
            torch.full_like(mse, 99.0),
            10.0 * torch.log10(4.0 / mse.clamp_min(1e-10)),
        )
        psnr_list = psnr.detach().cpu().tolist()

        for i in range(len(psnr_list)):
            if limit is not None and seen >= limit:
                break
            result.indices.append(int(nums[i]))
            result.psnrs.append(float(psnr_list[i]))
            seen += 1
        if limit is not None and seen >= limit:
            break

    result.seconds = time.perf_counter() - t0
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_csv(path: Path, results: list[RunResult]) -> None:
    """宽表：index, psnr_<tag1>, psnr_<tag2>, ..."""
    by_idx: dict[int, dict[str, float]] = {}
    tags = []
    for r in results:
        tags.append(r.spec.tag)
        for idx, p in zip(r.indices, r.psnrs):
            by_idx.setdefault(idx, {})[r.spec.tag] = p

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index"] + [f"psnr_{t}" for t in tags]
    delta_key = None
    if len(tags) == 2:
        delta_key = f"delta_{tags[1]}_minus_{tags[0]}"
        fieldnames.append(delta_key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for idx in sorted(by_idx):
            row = {"index": idx}
            for t in tags:
                row[f"psnr_{t}"] = f"{by_idx[idx].get(t, float('nan')):.6f}"
            if delta_key is not None:
                a = by_idx[idx].get(tags[0])
                b = by_idx[idx].get(tags[1])
                if a is not None and b is not None:
                    row[delta_key] = f"{(b - a):.6f}"
            w.writerow(row)


def write_markdown_report(
    path: Path,
    results: list[RunResult],
    host_dir: Path,
    image_size: int,
    device: str,
) -> None:
    lines: list[str] = []
    lines.append("# 宿主嵌入 PSNR 对比报告")
    lines.append("")
    lines.append(f"- 生成时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- 宿主目录: `{host_dir}`")
    lines.append(f"- 图像尺寸: `{image_size}×{image_size}`")
    lines.append(f"- 设备: `{device}`")
    lines.append(
        "- 公式: 像素∈[-1,1]，`PSNR = 10·log10(4 / MSE)`（与 `solver._psnr_neg1_1` 一致）"
    )
    lines.append("")
    lines.append("## 模型与权重")
    lines.append("")
    lines.append("| 标签 | 说明 | 权重 | embed_strength | 样本数 | 耗时(s) |")
    lines.append("|:---|:---|:---|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| `{r.spec.tag}` | {r.spec.label} | `{r.spec.weight_path}` | "
            f"{r.spec.embed_strength:g} | {r.n} | {r.seconds:.2f} |"
        )
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 模型 | n | 平均 PSNR (dB) | 中位 | 最小 | 最大 | 标准差 |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r.summary()
        lines.append(
            f"| {r.spec.label} (`{r.spec.tag}`) | {s['n']} | "
            f"{s['mean']:.4f} | {s['median']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} | {s['std']:.4f} |"
        )

    if len(results) == 2 and results[0].n and results[1].n:
        a_r, b_r = results[0], results[1]
        a = a_r.summary()["mean"]
        b = b_r.summary()["mean"]
        lines.append("")
        lines.append(f"## 对比（{b_r.spec.label} − {a_r.spec.label}）")
        lines.append("")
        lines.append(f"- 平均 PSNR 差: **{b - a:+.4f} dB**")
        map_a = dict(zip(a_r.indices, a_r.psnrs))
        map_b = dict(zip(b_r.indices, b_r.psnrs))
        common = sorted(set(map_a) & set(map_b))
        if common:
            deltas = np.asarray([map_b[i] - map_a[i] for i in common], dtype=np.float64)
            lines.append(f"- 配对样本数: {len(common)}")
            lines.append(f"- 配对差均值: **{deltas.mean():+.4f} dB**")
            lines.append(f"- 配对差中位: {np.median(deltas):+.4f} dB")
            lines.append(
                f"- 模型2 更高 / 更低 / 持平: "
                f"{int((deltas > 1e-6).sum())} / {int((deltas < -1e-6).sum())} / "
                f"{int((np.abs(deltas) <= 1e-6).sum())}"
            )

    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append(
        "- 本报告只衡量**嵌入不可见性**（Encoded vs Host），不含噪声层 / 拍屏失真。"
    )
    lines.append(
        "- `embed_strength` 优先读权重旁 `.meta.txt`；未写 meta 时默认 0（整图输出）。"
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _short_tag(path: Path, index: int) -> str:
    stem = path.stem
    # 缩短过长文件名
    if len(stem) > 40:
        stem = stem[:37] + "…"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe or f"m{index}"


def build_specs_from_ckpts(
    ckpts: list[Path],
    *,
    strength_overrides: list[Optional[float]] | None = None,
) -> list[ModelSpec]:
    """由 1~2 个权重路径构建 ModelSpec；strength 优先读 meta。"""
    if not (1 <= len(ckpts) <= 2):
        raise ValueError("仅支持 1 或 2 个权重")
    overrides = list(strength_overrides or [])
    while len(overrides) < len(ckpts):
        overrides.append(None)
    specs: list[ModelSpec] = []
    used_tags: set[str] = set()
    for i, path in enumerate(ckpts):
        tag = _short_tag(path, i + 1)
        if tag in used_tags:
            tag = f"{tag}_{i + 1}"
        used_tags.add(tag)
        if overrides[i] is not None:
            strength = float(overrides[i])  # type: ignore[arg-type]
        else:
            strength = read_embed_strength_meta(path, default=0.0)
        specs.append(
            ModelSpec(
                tag=tag,
                label=path.name,
                weight_path=path,
                embed_strength=strength,
            )
        )
    return specs


def build_parser() -> argparse.ArgumentParser:
    root = ROOT
    default_99, default_ours = resolve_default_weights(root)
    p = argparse.ArgumentParser(
        description="用选定的 1~2 个权重嵌入宿主并测 PSNR（可对照）"
    )
    p.add_argument(
        "--host_dir",
        type=Path,
        default=root / "Datasets" / "images",
        help="宿主图目录",
    )
    p.add_argument(
        "--wmat",
        type=Path,
        default=root / "results" / "WatermarkMatrix" / "w.mat",
        help="水印矩阵 w.mat",
    )
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="最多评估前 N 张")
    p.add_argument(
        "--ckpt",
        type=Path,
        action="append",
        default=None,
        help="权重 .pth（可重复 1~2 次）；未给则默认 mask_99 vs MB_best",
    )
    p.add_argument("--ckpt99", type=Path, default=default_99, help="兼容旧参数：权重1")
    p.add_argument(
        "--ckpt_ours",
        type=Path,
        default=default_ours,
        help="兼容旧参数：权重2",
    )
    p.add_argument(
        "--strength99",
        type=float,
        default=None,
        help="权重1 的 embed_strength（默认读 meta / 0）",
    )
    p.add_argument(
        "--strength_ours",
        type=float,
        default=None,
        help="权重2 的 embed_strength（默认读 meta / 0）",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=root / "results" / "psnr_embed_report",
        help="报告输出目录",
    )
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
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

    host_dir = args.host_dir
    if not host_dir.is_dir():
        print(f"错误: 宿主目录不存在: {host_dir}")
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
        "宿主嵌入 PSNR：单模型"
        if len(specs) == 1
        else f"宿主嵌入 PSNR：{specs[0].label} vs {specs[1].label}"
    )
    print("=" * 64)
    print(title)
    print("=" * 64)
    print(f"宿主:   {host_dir.resolve()}")
    print(f"w.mat:  {args.wmat.resolve()}")
    print(f"尺寸:   {args.image_size}  batch={args.batch_size}  device={device}")
    for sp in specs:
        print(
            f"[{sp.tag}] {sp.label}\n"
            f"       weights={sp.weight_path}\n"
            f"       embed_strength={sp.embed_strength:g}"
        )
    print("-" * 64)

    results: list[RunResult] = []
    for sp in specs:
        print(f"\n>>> 嵌入并测 PSNR: {sp.label} …")
        loader = get_loader(
            image_dir=str(host_dir),
            image_size=args.image_size,
            batch_size=args.batch_size,
            dataset="test_embedding",
            mode="test_embedding",
            num_workers=args.num_workers,
            w_path=str(args.wmat),
            lite=False,
        )
        try:
            r = embed_and_measure_psnr(sp, loader, device, limit=args.limit)
        except FileNotFoundError as e:
            print(f"错误: {e}")
            return 1
        results.append(r)
        s = r.summary()
        print(
            f"    n={s['n']}  mean={s['mean']:.4f} dB  "
            f"median={s['median']:.4f}  min={s['min']:.4f}  "
            f"max={s['max']:.4f}  ({r.seconds:.1f}s)"
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"psnr_report_{stamp}.md"
    csv_path = out_dir / f"psnr_per_image_{stamp}.csv"
    latest_md = out_dir / "psnr_report_latest.md"
    latest_csv = out_dir / "psnr_per_image_latest.csv"

    write_markdown_report(md_path, results, host_dir, args.image_size, str(device))
    write_csv(csv_path, results)
    write_markdown_report(latest_md, results, host_dir, args.image_size, str(device))
    write_csv(latest_csv, results)

    print("\n" + "=" * 64)
    print("汇总")
    print("=" * 64)
    for r in results:
        s = r.summary()
        print(
            f"  {r.spec.label:40s}  mean PSNR = {s['mean']:.4f} dB  (n={s['n']})"
        )
    if len(results) == 2 and results[0].n and results[1].n:
        d = results[1].summary()["mean"] - results[0].summary()["mean"]
        print(
            f"  平均差 ({results[1].spec.tag} - {results[0].spec.tag}) = {d:+.4f} dB"
        )
    print("-" * 64)
    print(f"报告: {md_path}")
    print(f"逐图: {csv_path}")
    print(f"最新: {latest_md}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
