#!/usr/bin/env python3
"""Differentiable nonlinear (spatially-variant) motion blur — AS_VALs demo.

Reproduces the MATLAB experiment using `motion_blur` (same API as
`Noise_Layer.ScreenShootingMB`).

Run::

    python AS_VALs/demos/python/demo_nonlinear_motion_blur.py
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
TEXT_IMAGES = AS_VALS / "outputs" / "text_images"
OUT_DIR = AS_VALS / "outputs" / "demo_python"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from motion_blur import (  # noqa: E402
    nonlinear_motion_blur,
    sample_pose_pair,
    sharp_view,
)


# ---------------------------------------------------------------------------
# I/O + visualization
# ---------------------------------------------------------------------------


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC RGB
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return ten


def to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) or (C,H,W) → HWC uint-display float in [0,1]."""
    x = t.detach()
    if x.ndim == 4:
        x = x[0]
    x = x.permute(1, 2, 0).cpu().numpy()
    return np.clip(x, 0.0, 1.0)


def plot_results(
    image: torch.Tensor,
    sharp: torch.Tensor,
    blurred: torch.Tensor,
    aux: dict[str, torch.Tensor],
    cx0: float,
    cy0: float,
    theta0: float,
    cx1: float,
    cy1: float,
    theta1: float,
    img_name: str,
    traj_grid: int = 16,
    quiver_step: int = 24,
) -> plt.Figure:
    """Four-panel figure matching the MATLAB experiment."""
    h, w = image.shape[-2], image.shape[-1]
    half = (sharp.shape[-1] - 1) / 2.0
    src = to_numpy_img(image)
    mag = aux["mag"].detach().cpu().numpy()
    x0 = aux["x0"].detach().cpu().numpy()
    y0 = aux["y0"].detach().cpu().numpy()
    x1 = aux["x1"].detach().cpu().numpy()
    y1 = aux["y1"].detach().cpu().numpy()
    u_grid = aux["u_grid"].detach().cpu().numpy()
    v_grid = aux["v_grid"].detach().cpu().numpy()

    fig = plt.figure(figsize=(14, 10), facecolor="white", constrained_layout=True)

    # 1) original
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(src)
    ax1.set_title(f"Original {img_name}\n(600×600)")
    ax1.axis("off")

    # 2) sharp vs blur
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(to_numpy_img(sharp))
    ax2.set_title("Sharp start crop (512)")
    ax2.axis("off")

    ax2b = fig.add_subplot(2, 3, 3)
    ax2b.imshow(to_numpy_img(blurred))
    ax2b.set_title("Nonlinear motion blur")
    ax2b.axis("off")

    # 3) magnitude heatmap
    ax3 = fig.add_subplot(2, 3, 4)
    im = ax3.imshow(mag, cmap="viridis")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title(f"Trajectory length |p1−p0|\n[{mag.min():.2f}, {mag.max():.2f}] px")
    ax3.set_xlabel("u")
    ax3.set_ylabel("v")

    # 4) sampled paths on original
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.imshow(src)
    uu = u_grid[::traj_grid, ::traj_grid]
    vv = v_grid[::traj_grid, ::traj_grid]
    t_fine = np.linspace(0.0, 1.0, 12)
    for ui, vi in zip(uu.ravel(), vv.ravel()):
        xs, ys = [], []
        for t in t_fine:
            cx = (1 - t) * cx0 + t * cx1
            cy = (1 - t) * cy0 + t * cy1
            th = (1 - t) * theta0 + t * theta1
            ct, st = math.cos(th), math.sin(th)
            xs.append(ct * ui - st * vi + cx)
            ys.append(st * ui + ct * vi + cy)
        ax4.plot(xs, ys, "-", color=(0.1, 0.45, 0.95, 0.85), linewidth=0.8)
        ax4.plot(xs[0], ys[0], "g.", markersize=3)
        ax4.plot(xs[-1], ys[-1], "r.", markersize=3)

    xq = x0[::quiver_step, ::quiver_step]
    yq = y0[::quiver_step, ::quiver_step]
    dxq = x1[::quiver_step, ::quiver_step] - xq
    dyq = y1[::quiver_step, ::quiver_step] - yq
    ax4.quiver(
        xq, yq, dxq, dyq, color="gold", angles="xy", scale_units="xy", scale=1, width=0.002
    )
    ax4.set_xlim(0.5, w + 0.5)
    ax4.set_ylim(h + 0.5, 0.5)
    ax4.set_title("Trajectories (green=start, red=end)")
    ax4.set_aspect("equal")

    # 5) start / end viewfinder
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.imshow(src)
    corners = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]],
        dtype=np.float64,
    )

    def poly_for(cx, cy, th):
        ct, st = math.cos(th), math.sin(th)
        xs = ct * corners[:, 0] - st * corners[:, 1] + cx
        ys = st * corners[:, 0] + ct * corners[:, 1] + cy
        return xs, ys

    xs0, ys0 = poly_for(cx0, cy0, theta0)
    xs1, ys1 = poly_for(cx1, cy1, theta1)
    ax5.plot(xs0, ys0, "g-", linewidth=2.0, label=r"start $\theta=0$")
    ax5.plot(
        xs1,
        ys1,
        "r-",
        linewidth=2.0,
        label=rf"end $\theta={math.degrees(theta1):.1f}^\circ$",
    )
    ax5.plot(cx0, cy0, "g+", markersize=12, markeredgewidth=1.5)
    ax5.plot(cx1, cy1, "rx", markersize=10, markeredgewidth=1.5)
    t_c = np.linspace(0, 1, 40)
    ax5.plot(
        (1 - t_c) * cx0 + t_c * cx1,
        (1 - t_c) * cy0 + t_c * cy1,
        "c--",
        linewidth=1.2,
        label="center path",
    )
    ax5.legend(loc="lower center", fontsize=8, ncol=2)
    ax5.set_title("Start / end viewfinder")
    ax5.set_xlim(0.5, w + 0.5)
    ax5.set_ylim(h + 0.5, 0.5)
    ax5.set_aspect("equal")

    fig.suptitle("Differentiable nonlinear motion blur (PyTorch grid_sample)", fontsize=13)
    return fig


def check_differentiable(blurred: torch.Tensor, image: torch.Tensor) -> None:
    loss = blurred.mean()
    loss.backward()
    assert image.grad is not None, "image.grad is None — blur is not differentiable"
    g = image.grad
    print(
        f"[grad check] OK  |grad| mean={g.abs().mean().item():.6e}  "
        f"max={g.abs().max().item():.6e}  nonzero={int((g != 0).sum())}/{g.numel()}"
    )


def main() -> None:
    img_dir = TEXT_IMAGES
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs under {img_dir}")

    img_path = random.choice(paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    view_size = 512
    n_samples = 16
    path_power = 2.0
    max_trans = 28.0
    max_rot_deg = 10.0

    image = load_image_tensor(img_path, device)
    _, _, h, w = image.shape
    if (h, w) != (600, 600):
        raise ValueError(f"expected 600x600 AS_VALs, got {h}x{w}")

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

    image = image.requires_grad_(True)
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

    check_differentiable(blurred, image)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Visualization (detach inside helpers); save + show
    with torch.no_grad():
        fig = plot_results(
            image,
            sharp,
            blurred,
            aux,
            cx0,
            cy0,
            theta0,
            cx1,
            cy1,
            theta1,
            img_path.name,
        )
        save_path = out_dir / f"demo_{img_path.stem}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved figure: {save_path}")
        plt.show()


if __name__ == "__main__":
    main()
