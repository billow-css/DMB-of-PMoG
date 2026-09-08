#!/usr/bin/env python3
"""COCOMask 左侧 128×128：非线性运动模糊 + PIMoG ScreenShooting 复合演示。

流程（拷贝输出，不改 Datasets 原图）::

    COCOMask (H×256)
        → 左宿主 128×128
        → MotionBlur(view=128, zeros 黑边) → 128×128
        → ScreenShooting（摩尔纹权重与抖动强度反比）

与 ``demo_coco_nonlinear_motion_blur.py`` 相同的 128 选取 / 可黑边约定；
不修改 AS_VALs 原脚本或 ``Noise_Layer``。

Run::

    python AS_VALs/demos/python/demo_coco_combined_with_pimog.py
    python AS_VALs/demos/python/demo_coco_combined_with_pimog.py --limit 3 --no_show
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
AS_VALS = THIS_DIR.parents[1]
REPO_ROOT = THIS_DIR.parents[2]
DEFAULT_COCO_DIR = REPO_ROOT / "Datasets" / "COCOMask" / "val" / "val_class"
OUT_ROOT = AS_VALS / "outputs" / "demo_python" / "coco_128_combined"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from Noise_Layer import ScreenShooting, blur_moire_weights  # noqa: E402
from demo_coco_nonlinear_motion_blur import (  # noqa: E402
    CROP,
    load_coco_left128,
    nonlinear_motion_blur_zeros,
    sample_pose_pair_allow_oob,
    save_png,
    sharp_view_zeros,
    to_numpy_img,
)


def n11_to_01(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(-1.0, 1.0) + 1.0) * 0.5


def o01_to_n11(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0 - 1.0


def plot_and_save(
    full,
    left,
    sharp,
    blurred,
    pimog_fixed,
    combined,
    img_name: str,
    blur_strength: float,
    alpha_light: float,
    alpha_moire: float,
    beta: float,
    save_path: Path,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 9), facecolor="white", constrained_layout=True)
    panels = [
        (full, f"COCOMask full\n{img_name}"),
        (left, f"Left host\n({CROP}×{CROP})"),
        (sharp, "Sharp @ g0\n(zeros pad)"),
        (blurred, f"Motion blur only\nmean|p1−p0|={blur_strength:.1f}px"),
        (
            n11_to_01(pimog_fixed),
            "PIMoG fixed moiré=0.15\n(on sharp, no blur)",
        ),
        (
            n11_to_01(combined),
            f"Combined adaptive\nα_moire={alpha_moire:.3f}  β={beta:.2f}",
        ),
    ]
    for i, (img, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, i)
        ax.imshow(to_numpy_img(img if img.ndim == 4 else img.unsqueeze(0)))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def process_one(
    img_path: Path,
    out_dir: Path,
    device: torch.device,
    n_samples: int,
    path_power: float,
    max_trans: float,
    max_rot_deg: float,
    blur_ref: float,
    moire_max: float,
    show: bool,
) -> None:
    full, left = load_coco_left128(img_path, device)
    poses = sample_pose_pair_allow_oob(
        width=CROP,
        height=CROP,
        view_size=CROP,
        max_trans=max_trans,
        max_rot_deg=max_rot_deg,
    )
    cx0, cy0, th0, cx1, cy1, th1 = poses

    with torch.no_grad():
        blurred, aux = nonlinear_motion_blur_zeros(
            left, *poses, view_size=CROP, n_samples=n_samples, path_power=path_power
        )
        sharp = sharp_view_zeros(left, cx0, cy0, th0, view_size=CROP)

        blur_strength = float(aux["mag"].mean().item())
        alpha_light, alpha_moire, beta = blur_moire_weights(
            blur_strength, moire_max=moire_max, blur_ref=blur_ref
        )
        print(
            f"{img_path.name}: blur={blur_strength:.2f}px  β={beta:.3f}  "
            f"α_L={alpha_light:.3f}  α_M={alpha_moire:.3f}"
        )

        noiser = ScreenShooting(moire_max=moire_max, blur_ref=blur_ref).to(device)
        pimog_fixed = noiser(o01_to_n11(sharp), moire_weight=moire_max)
        combined = noiser(o01_to_n11(blurred), blur_strength=blur_strength)

    panel = out_dir / f"combined_{img_path.stem}"
    panel.mkdir(parents=True, exist_ok=True)

    save_png(full, panel / "00_cocomas_full_copy.png")
    save_png(left, panel / "01_left128.png")
    save_png(sharp, panel / "02_sharp_g0.png")
    save_png(blurred, panel / "03_motion_blur_128.png")
    save_png(n11_to_01(pimog_fixed), panel / "04_pimog_fixed_moire.png")
    save_png(n11_to_01(combined), panel / "05_combined_adaptive_moire.png")

    (panel / "weights.txt").write_text(
        f"source={img_path}\n"
        f"crop=left {CROP}x{CROP}\n"
        f"view_size={CROP}\n"
        f"padding_mode=zeros\n"
        f"blur_strength={blur_strength:.6f}\n"
        f"blur_ref={blur_ref}\n"
        f"beta={beta:.6f}\n"
        f"alpha_light={alpha_light:.6f}\n"
        f"alpha_moire={alpha_moire:.6f}\n"
        f"start: cx={cx0:.4f} cy={cy0:.4f} theta=0\n"
        f"end: cx={cx1:.4f} cy={cy1:.4f} theta_deg={math.degrees(th1):.4f}\n",
        encoding="utf-8",
    )

    fig = plot_and_save(
        full,
        left,
        sharp,
        blurred,
        pimog_fixed,
        combined,
        img_path.name,
        blur_strength,
        alpha_light,
        alpha_moire,
        beta,
        panel / "combined_demo.png",
    )
    print(f"saved copies → {panel}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="COCO left-128 blur + PIMoG ScreenShooting")
    p.add_argument("--image_dir", type=Path, default=DEFAULT_COCO_DIR)
    p.add_argument("--out_dir", type=Path, default=OUT_ROOT)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--path_power", type=float, default=2.0)
    p.add_argument("--max_trans", type=float, default=0.06 * CROP)
    p.add_argument("--max_rot_deg", type=float, default=12.0)
    p.add_argument("--blur_ref", type=float, default=max(8.0, 0.15 * CROP))
    p.add_argument("--moire_max", type=float, default=0.15)
    p.add_argument("--no_show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"COCO 目录不存在: {args.image_dir}")

    paths = sorted(
        [
            p
            for p in args.image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    if not paths:
        raise FileNotFoundError(f"无图像: {args.image_dir}")

    n = max(1, args.limit)
    chosen = random.sample(paths, k=min(n, len(paths))) if n < len(paths) else paths[:n]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  crop={CROP}  zeros_black_border=True")
    for path in chosen:
        process_one(
            path,
            args.out_dir,
            device,
            args.n_samples,
            args.path_power,
            args.max_trans,
            args.max_rot_deg,
            args.blur_ref,
            args.moire_max,
            show=not args.no_show and len(chosen) == 1,
        )


if __name__ == "__main__":
    main()
