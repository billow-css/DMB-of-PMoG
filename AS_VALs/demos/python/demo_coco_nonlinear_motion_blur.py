#!/usr/bin/env python3
"""COCOMask 左侧 128×128 非线性运动模糊演示（拷贝输出，不改原图）。

与 PIMoG 实机 / 训练用 128 宿主一致：
  - 只取 COCOMask 拼接图左侧 128 像素（宿主），右侧 mask 不参与；
  - 取景窗 = 128，直接在 128×128 上做空变抖动（不再二次放大画布）；
  - 允许越界采样 → ``padding_mode=zeros``，可能出现黑边。

输出写到 ``AS_VALs/outputs/demo_python/coco_128_blur/``，不覆盖 ``Datasets/``。

Run（仓库根目录）::

    python AS_VALs/demos/python/demo_coco_nonlinear_motion_blur.py
    python AS_VALs/demos/python/demo_coco_nonlinear_motion_blur.py --limit 5
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
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
AS_VALS = THIS_DIR.parents[1]
REPO_ROOT = THIS_DIR.parents[2]
DEFAULT_COCO_DIR = REPO_ROOT / "Datasets" / "COCOMask" / "val" / "val_class"
OUT_ROOT = AS_VALS / "outputs" / "demo_python" / "coco_128_blur"

CROP = 128  # PIMoG image_size


def pose_map(u, v, cx, cy, theta):
    ct = torch.cos(torch.as_tensor(theta, device=u.device, dtype=u.dtype))
    st = torch.sin(torch.as_tensor(theta, device=u.device, dtype=u.dtype))
    cx_t = torch.as_tensor(cx, device=u.device, dtype=u.dtype)
    cy_t = torch.as_tensor(cy, device=u.device, dtype=u.dtype)
    return ct * u - st * v + cx_t, st * u + ct * v + cy_t


def pixel_to_grid(x, y, width, height):
    gx = 2.0 * (x - 1.0) / max(width - 1, 1) - 1.0
    gy = 2.0 * (y - 1.0) / max(height - 1, 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def sample_pose_pair_allow_oob(
    width: int,
    height: int,
    view_size: int,
    max_trans: float,
    max_rot_deg: float,
) -> tuple[float, float, float, float, float, float]:
    """起点轴对齐居中（128 满幅时 span=0）；终点可越界 → 黑边。"""
    half = (view_size - 1) / 2.0
    span_w = max(width - view_size, 0)
    span_h = max(height - view_size, 0)
    cx0 = half + 1.0 + (random.random() * span_w if span_w else 0.0)
    cy0 = half + 1.0 + (random.random() * span_h if span_h else 0.0)
    theta0 = 0.0
    cx1 = cx0 + (2.0 * random.random() - 1.0) * max_trans
    cy1 = cy0 + (2.0 * random.random() - 1.0) * max_trans
    theta1 = math.radians((2.0 * random.random() - 1.0) * max_rot_deg)
    return cx0, cy0, theta0, cx1, cy1, theta1


def nonlinear_motion_blur_zeros(
    image: torch.Tensor,
    cx0,
    cy0,
    theta0,
    cx1,
    cy1,
    theta1,
    view_size: int = CROP,
    n_samples: int = 16,
    path_power: float = 2.0,
    gauss_sigma: float | None = None,
):
    """满幅 128 模糊；OOB 用 zeros（黑边）。路径 λ(t)=t**path_power，等权曝光。"""
    del gauss_sigma
    b, c, h, w = image.shape
    device, dtype = image.device, image.dtype
    half = (view_size - 1) / 2.0
    axis = torch.linspace(-half, half, view_size, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(axis, axis, indexing="ij")
    t_list = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
    lam_list = t_list.pow(path_power)
    w_list = torch.full((n_samples,), 1.0 / n_samples, device=device, dtype=dtype)
    acc = torch.zeros(b, c, view_size, view_size, device=device, dtype=dtype)
    x0, y0 = pose_map(u_grid, v_grid, cx0, cy0, theta0)
    x1, y1 = pose_map(u_grid, v_grid, cx1, cy1, theta1)
    for k in range(n_samples):
        lam = lam_list[k]
        cx = (1.0 - lam) * cx0 + lam * cx1
        cy = (1.0 - lam) * cy0 + lam * cy1
        th = (1.0 - lam) * theta0 + lam * theta1
        xs, ys = pose_map(u_grid, v_grid, cx, cy, th)
        grid = pixel_to_grid(xs, ys, w, h).unsqueeze(0).expand(b, -1, -1, -1)
        frame = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        acc = acc + w_list[k] * frame
    aux = {
        "u_grid": u_grid,
        "v_grid": v_grid,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "mag": torch.hypot(x1 - x0, y1 - y0),
        "weights": w_list,
        "t_list": t_list,
        "lambda": lam_list,
    }
    return acc, aux


def sharp_view_zeros(image, cx, cy, theta, view_size: int = CROP):
    b, _, h, w = image.shape
    half = (view_size - 1) / 2.0
    axis = torch.linspace(-half, half, view_size, device=image.device, dtype=image.dtype)
    v_grid, u_grid = torch.meshgrid(axis, axis, indexing="ij")
    xs, ys = pose_map(u_grid, v_grid, cx, cy, theta)
    grid = pixel_to_grid(xs, ys, w, h).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def load_coco_left128(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (full BCHW[0,1], left128 BCHW[0,1])。"""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    if w < CROP or h < CROP:
        raise ValueError(f"{path.name}: need ≥{CROP}×{CROP}, got {h}×{w}")
    full = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    left = full[:, :, :CROP, :CROP].contiguous()
    return full, left


def to_numpy_img(t: torch.Tensor) -> np.ndarray:
    x = t.detach()[0].permute(1, 2, 0).cpu().float().numpy()
    return np.clip(x, 0.0, 1.0)


def save_png(t01: torch.Tensor, path: Path) -> None:
    Image.fromarray((to_numpy_img(t01) * 255.0).astype(np.uint8)).save(path)


def plot_results(full, left, sharp, blurred, aux, img_name, poses) -> plt.Figure:
    cx0, cy0, th0, cx1, cy1, th1 = poses
    mag = aux["mag"].detach().cpu().numpy()
    fig = plt.figure(figsize=(14, 8), facecolor="white", constrained_layout=True)

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(to_numpy_img(full))
    ax.set_title(f"COCOMask full\n{img_name}")
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(to_numpy_img(left))
    ax.set_title(f"Left crop ({CROP}×{CROP})")
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(to_numpy_img(sharp))
    ax.set_title("Sharp @ g0 (zeros pad)")
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(to_numpy_img(blurred))
    ax.set_title(
        f"Motion blur 128\n"
        f"Δ=({cx1-cx0:.1f},{cy1-cy0:.1f}) θ={math.degrees(th1):.1f}°"
    )
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 5)
    im = ax.imshow(mag, cmap="magma")
    ax.set_title(f"|p1−p0|  mean={mag.mean():.2f}px")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    ax.set_title("PIMoG-style 128 notes")
    ax.text(
        0.02,
        0.95,
        "• host = left 128 only\n"
        "• view_size = 128 (full-frame)\n"
        "• OOB → black border (zeros)\n"
        "• originals untouched; copies only\n"
        f"• g0=({cx0:.1f},{cy0:.1f},0°)\n"
        f"• g1=({cx1:.1f},{cy1:.1f},{math.degrees(th1):.1f}°)",
        va="top",
        family="monospace",
        fontsize=10,
        transform=ax.transAxes,
    )
    return fig


def process_one(
    img_path: Path,
    out_dir: Path,
    device: torch.device,
    n_samples: int,
    path_power: float,
    max_trans: float,
    max_rot_deg: float,
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

    panel = out_dir / img_path.stem
    panel.mkdir(parents=True, exist_ok=True)
    save_png(full, panel / "00_cocomas_full_copy.png")
    save_png(left, panel / "01_left128.png")
    save_png(sharp, panel / "02_sharp_g0.png")
    save_png(blurred, panel / "03_motion_blur_128.png")

    info = (
        f"source={img_path}\n"
        f"crop=left {CROP}x{CROP}\n"
        f"view_size={CROP}\n"
        f"padding_mode=zeros  # black borders OK\n"
        f"max_trans={max_trans}\nmax_rot_deg={max_rot_deg}\n"
        f"n_samples={n_samples}\npath_power={path_power}  # λ=t^p, uniform shutter\n"
        f"start: cx={cx0:.4f} cy={cy0:.4f} theta_deg=0\n"
        f"end:   cx={cx1:.4f} cy={cy1:.4f} theta_deg={math.degrees(th1):.4f}\n"
        f"mag_mean={float(aux['mag'].mean()):.4f}\n"
    )
    (panel / "run_info.txt").write_text(info, encoding="utf-8")

    fig = plot_results(full, left, sharp, blurred, aux, img_path.name, poses)
    fig_path = panel / "demo_figure.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"saved copies → {panel}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="COCO left-128 nonlinear motion blur (copy-only)")
    p.add_argument(
        "--image_dir",
        type=Path,
        default=DEFAULT_COCO_DIR,
        help="COCOMask 目录（默认 val_class）",
    )
    p.add_argument("--out_dir", type=Path, default=OUT_ROOT)
    p.add_argument("--limit", type=int, default=1, help="处理张数（默认 1=随机一张）")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument(
        "--path_power",
        type=float,
        default=2.0,
        help="进度 λ=t^p；2=匀加速（v∝t），1=匀速",
    )
    p.add_argument(
        "--max_trans",
        type=float,
        default=0.06 * CROP,
        help="终点最大平移（默认 0.06×128≈7.7，对齐 ScreenShootingMB）",
    )
    p.add_argument("--max_rot_deg", type=float, default=12.0)
    p.add_argument("--no_show", action="store_true", help="不弹窗，只写盘")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    img_dir = args.image_dir
    if not img_dir.is_dir():
        raise FileNotFoundError(f"COCO 目录不存在: {img_dir}")

    paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not paths:
        raise FileNotFoundError(f"无图像: {img_dir}")

    n = max(1, args.limit)
    chosen = random.sample(paths, k=min(n, len(paths))) if n < len(paths) else paths[:n]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  crop={CROP}  allow_oob_black_border=True")
    print(f"image_dir={img_dir}  n={len(chosen)}")
    for path in chosen:
        process_one(
            path,
            args.out_dir,
            device,
            args.n_samples,
            args.path_power,
            args.max_trans,
            args.max_rot_deg,
            show=not args.no_show and len(chosen) == 1,
        )


if __name__ == "__main__":
    main()
