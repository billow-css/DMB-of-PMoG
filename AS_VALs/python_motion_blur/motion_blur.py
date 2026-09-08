#!/usr/bin/env python3
"""Differentiable nonlinear (spatially-variant) motion blur — library API.

Used by AS_VALs demos and PIMoG ``Noise_Layer.ScreenShootingMB``.
Coordinates are MATLAB-compatible 1-based pixels; sampling via
``torch.nn.functional.grid_sample`` (``align_corners=True``).
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F


def pose_map(
    u: torch.Tensor,
    v: torch.Tensor,
    cx: torch.Tensor | float,
    cy: torch.Tensor | float,
    theta: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local (u,v) relative to view center → image coords (x,y), 1-based."""
    ct = torch.cos(torch.as_tensor(theta, device=u.device, dtype=u.dtype))
    st = torch.sin(torch.as_tensor(theta, device=u.device, dtype=u.dtype))
    cx_t = torch.as_tensor(cx, device=u.device, dtype=u.dtype)
    cy_t = torch.as_tensor(cy, device=u.device, dtype=u.dtype)
    x = ct * u - st * v + cx_t
    y = st * u + ct * v + cy_t
    return x, y


def pixel_to_grid(
    x: torch.Tensor,
    y: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """1-based pixel (x,y) → grid_sample coords in [-1, 1], align_corners=True."""
    gx = 2.0 * (x - 1.0) / max(width - 1, 1) - 1.0
    gy = 2.0 * (y - 1.0) / max(height - 1, 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def pose_in_bounds(
    u: torch.Tensor,
    v: torch.Tensor,
    cx: float,
    cy: float,
    theta: float,
    width: int,
    height: int,
    tol: float = 1e-6,
) -> bool:
    x, y = pose_map(u, v, cx, cy, theta)
    return bool(
        torch.all(
            (x >= 1 - tol)
            & (x <= width + tol)
            & (y >= 1 - tol)
            & (y <= height + tol)
        ).item()
    )


def pose_path_in_bounds(
    u: torch.Tensor,
    v: torch.Tensor,
    cx0: float,
    cy0: float,
    theta0: float,
    cx1: float,
    cy1: float,
    theta1: float,
    width: int,
    height: int,
    n_check: int = 16,
) -> bool:
    for t in np.linspace(0.0, 1.0, n_check):
        cx = (1 - t) * cx0 + t * cx1
        cy = (1 - t) * cy0 + t * cy1
        th = (1 - t) * theta0 + t * theta1
        if not pose_in_bounds(u, v, cx, cy, th, width, height):
            return False
    return True


def sample_pose_pair(
    width: int,
    height: int,
    view_size: int,
    max_trans: float,
    max_rot_deg: float,
    n_check: int,
    max_tries: int = 500,
    device: torch.device | None = None,
) -> tuple[float, float, float, float, float, float, int]:
    """Rejection-sample start/end poses so the full path stays in-bounds."""
    device = device or torch.device("cpu")
    half = (view_size - 1) / 2.0
    corner_u = torch.tensor(
        [-half, half, half, -half], device=device, dtype=torch.float32
    )
    corner_v = torch.tensor(
        [-half, -half, half, half], device=device, dtype=torch.float32
    )

    if width < view_size or height < view_size:
        raise ValueError(
            f"image {width}x{height} smaller than view_size={view_size}"
        )

    theta0 = 0.0
    n_reject = 0
    span_w = max(width - view_size, 0)
    span_h = max(height - view_size, 0)
    for _ in range(max_tries):
        cx0 = half + 1.0 + (random.random() * span_w if span_w > 0 else 0.0)
        cy0 = half + 1.0 + (random.random() * span_h if span_h > 0 else 0.0)
        cx1 = cx0 + (2.0 * random.random() - 1.0) * max_trans
        cy1 = cy0 + (2.0 * random.random() - 1.0) * max_trans
        theta1 = math.radians((2.0 * random.random() - 1.0) * max_rot_deg)

        if pose_path_in_bounds(
            corner_u,
            corner_v,
            cx0,
            cy0,
            theta0,
            cx1,
            cy1,
            theta1,
            width,
            height,
            n_check,
        ):
            return cx0, cy0, theta0, cx1, cy1, theta1, n_reject
        n_reject += 1

    raise RuntimeError(
        f"Failed to sample in-bounds poses after {max_tries} tries. "
        "Reduce max_trans / max_rot_deg."
    )


def nonlinear_motion_blur(
    image: torch.Tensor,
    cx0: float | torch.Tensor,
    cy0: float | torch.Tensor,
    theta0: float | torch.Tensor,
    cx1: float | torch.Tensor,
    cy1: float | torch.Tensor,
    theta1: float | torch.Tensor,
    view_size: int = 512,
    n_samples: int = 16,
    path_power: float = 2.0,
    gauss_sigma: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Spatially-variant motion blur via v–t / uniform-acceleration path + equal-time exposure.

    Pose progress uses ``λ(t)=t**path_power`` (default 2 → rest-to-end uniform acceleration:
    ``v∝t``, ``s∝t²``). Exposure uses uniform weights ``1/K`` (open shutter), not a
    Gaussian time kernel.

    Args:
        image: ``(B, C, H, W)`` float tensor.
        view_size: output crop / window size.
        n_samples: equal-time samples along ``t∈[0,1]``.
        path_power: progress exponent; ``1`` = constant speed in pose space, ``2`` = uniform accel.
        gauss_sigma: deprecated, ignored (kept for call-site compatibility).

    Returns:
        blurred: ``(B, C, view_size, view_size)``
        aux: ``mag``, endpoint maps, ``lambda``, ``weights``, …
    """
    del gauss_sigma  # unused; uniform shutter
    if image.ndim != 4:
        raise ValueError(f"expected (B,C,H,W), got {tuple(image.shape)}")
    if path_power <= 0:
        raise ValueError(f"path_power must be > 0, got {path_power}")

    b, c, h, w = image.shape
    device, dtype = image.device, image.dtype
    half = (view_size - 1) / 2.0

    axis = torch.linspace(-half, half, view_size, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(axis, axis, indexing="ij")

    # Equal-time samples; path progress λ(t)=t^p (匀加速默认 p=2)
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
            padding_mode="border",
            align_corners=True,
        )
        acc = acc + w_list[k] * frame

    mag = torch.hypot(x1 - x0, y1 - y0)
    aux = {
        "u_grid": u_grid,
        "v_grid": v_grid,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "mag": mag,
        "weights": w_list,
        "t_list": t_list,
        "lambda": lam_list,
        "path_power": torch.tensor(path_power, device=device, dtype=dtype),
    }
    return acc, aux


def mean_trajectory_length(aux: dict[str, torch.Tensor]) -> float:
    """Scalar blur strength s = mean(|p1-p0|)."""
    return float(aux["mag"].detach().mean().item())


def sharp_view(
    image: torch.Tensor,
    cx: float,
    cy: float,
    theta: float,
    view_size: int = 512,
) -> torch.Tensor:
    """Differentiable clear crop at a single pose (border padding)."""
    b, _, h, w = image.shape
    half = (view_size - 1) / 2.0
    axis = torch.linspace(-half, half, view_size, device=image.device, dtype=image.dtype)
    v_grid, u_grid = torch.meshgrid(axis, axis, indexing="ij")
    xs, ys = pose_map(u_grid, v_grid, cx, cy, theta)
    grid = pixel_to_grid(xs, ys, w, h).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
