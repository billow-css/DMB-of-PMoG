#!/usr/bin/env python3
"""Combine nonlinear motion blur with PIMoG ScreenShooting and visualize.

Pipeline:
    AS_VALs 600x600
        → NonlinearMotionBlur → 512x512  [0,1]
        → map to [-1,1]
        → ScreenShooting(blur_strength=mean_mag)   # 抖动↑ → 摩尔纹↓
        → map back to [0,1] for display

Weight constraint (inverse):
    β = clip(mean(|p1-p0|) / blur_ref, 0, 1)
    α_moire = 0.15 * (1 - β)
    α_light = 1 - α_moire

Run (from repo root)::

    python AS_VALs/demos/python/demo_combined_with_pimog.py
"""

from __future__ import annotations

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
CORE_DIR = AS_VALS / "python_motion_blur"
AS_VALS_OUT = AS_VALS / "outputs" / "text_images"
OUT_DIR = AS_VALS / "outputs" / "demo_python"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from Noise_Layer import ScreenShooting, blur_moire_weights  # noqa: E402
from demo_nonlinear_motion_blur import (  # noqa: E402
    nonlinear_motion_blur,
    sample_pose_pair,
    sharp_view,
)


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def to01(t: torch.Tensor) -> np.ndarray:
    x = t.detach()[0].permute(1, 2, 0).cpu().float().numpy()
    return np.clip(x, 0.0, 1.0)


def n11_to_01(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(-1.0, 1.0) + 1.0) * 0.5


def o01_to_n11(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0 - 1.0


def plot_and_save(
    original: torch.Tensor,
    sharp: torch.Tensor,
    blurred: torch.Tensor,
    pimog_fixed: torch.Tensor,
    combined_adaptive: torch.Tensor,
    img_name: str,
    cx0: float,
    cy0: float,
    cx1: float,
    cy1: float,
    theta1: float,
    blur_strength: float,
    alpha_light: float,
    alpha_moire: float,
    beta: float,
    save_path: Path,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 9), facecolor="white", constrained_layout=True)

    panels = [
        (original, f"Original\n{img_name} (600×600)"),
        (sharp, "Sharp start crop\n(512×512)"),
        (blurred, f"Motion blur only\nmean|p1−p0|={blur_strength:.1f}px"),
        (
            n11_to_01(pimog_fixed),
            "PIMoG fixed moiré=0.15\n(on sharp crop, no blur)",
        ),
        (
            n11_to_01(combined_adaptive),
            f"Combined (adaptive)\nα_moire={alpha_moire:.3f}  β={beta:.2f}",
        ),
    ]

    for i, (img, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, i)
        ax.imshow(to01(img))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    info = (
        "Inverse weight constraint\n"
        "-------------------------\n"
        "β = clip(mag / blur_ref, 0, 1)\n"
        "α_moire = 0.15 · (1 − β)\n"
        "α_light = 1 − α_moire\n\n"
        f"blur_strength = {blur_strength:.2f} px\n"
        f"β             = {beta:.3f}\n"
        f"α_light       = {alpha_light:.3f}\n"
        f"α_moire       = {alpha_moire:.3f}\n\n"
        f"start: ({cx0:.1f}, {cy0:.1f}), θ=0°\n"
        f"end:   ({cx1:.1f}, {cy1:.1f}), "
        f"θ={math.degrees(theta1):.2f}°\n\n"
        "→ stronger shake ⇒ weaker moiré"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )
    ax.set_title("Notes", fontsize=11)

    fig.suptitle(
        "PIMoG + Motion Blur  (moire ∝ 1 / shake)",
        fontsize=14,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved: {save_path}")
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(AS_VALS_OUT.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs under {AS_VALS_OUT}")

    img_path = random.choice(paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    view_size = 512
    n_samples = 16
    path_power = 2.0
    max_trans = 28.0
    max_rot_deg = 10.0
    blur_ref = 40.0  # β=1 时摩尔纹完全关掉
    moire_max = 0.15

    image = load_image_tensor(img_path, device)
    _, _, h, w = image.shape
    if (h, w) != (600, 600):
        raise ValueError(f"expected 600×600, got {h}×{w}")

    cx0, cy0, theta0, cx1, cy1, theta1, n_reject = sample_pose_pair(
        width=w,
        height=h,
        view_size=view_size,
        max_trans=max_trans,
        max_rot_deg=max_rot_deg,
        n_check=n_samples,
    )

    print(f"image: {img_path.name}")
    print(f"device: {device}")
    print(f"path_power={path_power} (λ=t^p, uniform shutter)")
    print(f"reject_count: {n_reject}")
    print(f"start: cx={cx0:.2f} cy={cy0:.2f} theta={math.degrees(theta0):.2f} deg")
    print(f"end:   cx={cx1:.2f} cy={cy1:.2f} theta={math.degrees(theta1):.2f} deg")

    with torch.no_grad():
        blurred, aux = nonlinear_motion_blur(
            image,
            cx0,
            cy0,
            theta0,
            cx1,
            cy1,
            theta1,
            view_size=view_size,
            n_samples=n_samples,
            path_power=path_power,
        )
        sharp = sharp_view(image, cx0, cy0, theta0, view_size=view_size)

        # 抖动强度：逐像素轨迹长度均值
        blur_strength = float(aux["mag"].mean().item())
        alpha_light, alpha_moire, beta = blur_moire_weights(
            blur_strength, moire_max=moire_max, blur_ref=blur_ref
        )
        print(
            f"blur_strength={blur_strength:.2f}px  β={beta:.3f}  "
            f"α_light={alpha_light:.3f}  α_moire={alpha_moire:.3f}"
        )

        noiser = ScreenShooting(moire_max=moire_max, blur_ref=blur_ref).to(device)

        sharp_n11 = o01_to_n11(sharp)
        blur_n11 = o01_to_n11(blurred)

        # 对照：固定论文权重（无抖动约束）
        pimog_fixed = noiser(sharp_n11, moire_weight=moire_max)
        # 联合：模糊图 + 由抖动反比得到的摩尔纹权重
        combined = noiser(blur_n11, blur_strength=blur_strength)

        stem = img_path.stem
        panel_dir = OUT_DIR / f"combined_{stem}"
        panel_dir.mkdir(parents=True, exist_ok=True)

        def save_img(t01: torch.Tensor, name: str) -> None:
            arr = (to01(t01) * 255.0).astype(np.uint8)
            Image.fromarray(arr).save(panel_dir / name)

        save_img(image, "01_original.png")
        save_img(sharp, "02_sharp_crop.png")
        save_img(blurred, "03_motion_blur.png")
        save_img(n11_to_01(pimog_fixed), "04_pimog_fixed_moire.png")
        save_img(n11_to_01(combined), "05_combined_adaptive_moire.png")

        (panel_dir / "weights.txt").write_text(
            f"blur_strength={blur_strength:.6f}\n"
            f"blur_ref={blur_ref}\n"
            f"beta={beta:.6f}\n"
            f"alpha_light={alpha_light:.6f}\n"
            f"alpha_moire={alpha_moire:.6f}\n",
            encoding="utf-8",
        )

        plot_and_save(
            image,
            sharp,
            blurred,
            pimog_fixed,
            combined,
            img_path.name,
            cx0,
            cy0,
            cx1,
            cy1,
            theta1,
            blur_strength,
            alpha_light,
            alpha_moire,
            beta,
            OUT_DIR / f"combined_demo_{stem}.png",
        )
        print(f"panels: {panel_dir}")
        plt.show()


if __name__ == "__main__":
    main()
