#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_test/sim_noise.py
=====================
仿真拍屏（sim_test）：用评估同款噪声层替代人工拍屏，快速验证水印鲁棒性。

适用
----
- PIMoG 官方权重 / DMB-PMoG 权重 / 挂到同一 Encoder–Decoder 架构的第三方权重
  所导出的含水印图（或任意 RGB 图）
- 只做噪声叠加，不加载 Encoder/Decoder

噪声
----
- ``ScreenShooting`` / ``PIMoG``     ：透视 + 光照 + 摩尔纹 + 高斯（原论文族）
- ``ScreenShootingMB`` / ``DMB``     ：+ 动态运动模糊（DMB of PMoG）
- ``both``                           ：两种各输出一套

用法
----
  python sim_test/sim_noise.py ^
      --input_dir Datasets/images_watermarked ^
      --output_dir sim_test/out_dmb ^
      --noise ScreenShootingMB

  python sim_test/sim_noise.py -i path/in -o path/out --noise both --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Noise_Layer import Identity, ScreenShooting, ScreenShootingMB  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# CLI 别名 → 内部噪声名
NOISE_ALIASES = {
    "screenshooting": "ScreenShooting",
    "ss": "ScreenShooting",
    "pimog": "ScreenShooting",
    "pi": "ScreenShooting",
    "screenshootingmb": "ScreenShootingMB",
    "ssmb": "ScreenShootingMB",
    "mb": "ScreenShootingMB",
    "dmb": "ScreenShootingMB",
    "dmb-pmog": "ScreenShootingMB",
    "dmb_pmog": "ScreenShootingMB",
    "both": "both",
    "all": "both",
    "identity": "Identity",
}


def normalize_noise_choice(raw: str) -> str:
    key = str(raw).strip()
    mapped = NOISE_ALIASES.get(key.lower())
    if mapped:
        return mapped
    if key in ("ScreenShooting", "ScreenShootingMB", "Identity", "both"):
        return key
    raise SystemExit(
        f"未知 --noise={raw!r}。可选: ScreenShooting|PIMoG|ScreenShootingMB|DMB|both"
    )


def build_noiser(name: str) -> torch.nn.Module:
    if name == "ScreenShooting":
        return ScreenShooting()
    if name == "ScreenShootingMB":
        return ScreenShootingMB()
    if name == "Identity":
        return Identity()
    raise ValueError(name)


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {folder}")
    files = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(f"输入目录无图像: {folder}")
    return files


def load_image_neg11(
    path: Path,
    image_size: int | None,
    device: torch.device,
) -> torch.Tensor:
    """读图 → RGB float tensor [1,3,H,W] in [-1,1]。"""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"无法读取: {path}")
    if image_size is not None and image_size > 0:
        bgr = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = rgb / 255.0 * 2.0 - 1.0
    t = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)
    return t.to(device)


def save_tensor_neg11(path: Path, tensor_bchw: torch.Tensor) -> None:
    """[-1,1] BCHW → uint8 BGR 写盘。"""
    arr = tensor_bchw.detach().float().cpu()[0].numpy()
    arr = np.clip((arr + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    hwc = np.transpose(arr, (1, 2, 0))
    bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"写盘失败: {path}")


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_folder(
    *,
    input_dir: Path,
    output_dir: Path,
    noise_name: str,
    image_size: int | None,
    device: torch.device,
    seed: int | None,
    limit: int | None,
) -> int:
    files = list_images(input_dir)
    if limit is not None and limit > 0:
        files = files[:limit]

    noiser = build_noiser(noise_name).to(device)
    noiser.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sim_test] noise={noise_name}  device={device}  n={len(files)}")
    print(f"  in : {input_dir.resolve()}")
    print(f"  out: {output_dir.resolve()}")

    n_ok = 0
    with torch.no_grad():
        for i, src in enumerate(files):
            # 每张图可复现：seed + index
            if seed is not None:
                set_seed(int(seed) + i * 10007)
            try:
                x = load_image_neg11(src, image_size, device)
                y = noiser(x)
                dst = output_dir / src.name
                save_tensor_neg11(dst, y)
                n_ok += 1
                if (i + 1) % 50 == 0 or (i + 1) == len(files):
                    print(f"  … {i + 1}/{len(files)}", flush=True)
            except Exception as e:
                print(f"  [跳过] {src.name}: {e}", flush=True)

    print(f"[sim_test] 完成 {n_ok}/{len(files)} → {output_dir}")
    return n_ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sim_noise.py",
        description=(
            "sim_test：对输入文件夹图像叠加 PIMoG / DMB 噪声层，"
            "输出到指定目录（替代人工拍屏）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python sim_test/sim_noise.py -i Datasets/images_watermarked -o sim_test/out_ss --noise PIMoG
  python sim_test/sim_noise.py -i Datasets/images_watermarked -o sim_test/out_mb --noise DMB
  python sim_test/sim_noise.py -i path/in -o path/out --noise both --seed 0 --limit 100
""".strip(),
    )
    p.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        required=True,
        help="输入图像目录（含水印图或任意 RGB）",
    )
    p.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="输出目录；--noise both 时在其下建 ScreenShooting/ 与 ScreenShootingMB/",
    )
    p.add_argument(
        "--noise",
        type=str,
        default="ScreenShootingMB",
        help="ScreenShooting|PIMoG | ScreenShootingMB|DMB | both（默认 DMB）",
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="可选：缩放到边长 N（默认保持原分辨率）",
    )
    p.add_argument("--batch_hint", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    p.add_argument("--limit", type=int, default=None, help="最多处理前 N 张")
    p.add_argument("--cpu", action="store_true", help="强制 CPU")
    return p


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args(argv)
    noise = normalize_noise_choice(args.noise)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    if noise == "both":
        total = 0
        for name in ("ScreenShooting", "ScreenShootingMB"):
            sub = args.output_dir / name
            total += process_folder(
                input_dir=args.input_dir,
                output_dir=sub,
                noise_name=name,
                image_size=args.image_size,
                device=device,
                seed=args.seed,
                limit=args.limit,
            )
        print(f"[sim_test] both 合计写出 {total} 张")
        return 0 if total > 0 else 1

    n = process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        noise_name=noise,
        image_size=args.image_size,
        device=device,
        seed=args.seed,
        limit=args.limit,
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
