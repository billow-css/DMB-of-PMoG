#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.metrics
====================
统一指标：bit-Acc / BER / PSNR / SSIM。

公式与仓库 ``solver.py`` 保持一致（见 ``solver.py:542`` 起），此处**独立复制**，
避免 harness 依赖 torch 训练重依赖：
- ``_psnr_neg1_1``       → 本模块 ``psnr_neg1_1_mean``（[-1,1]，峰值 2 → 10log10(4/MSE)）
- ``_bits_from_logits``  → 本模块 ``bits_from_logits``（>=0.5 记 1）
- ``_bit_errors``        → 本模块 ``bit_errors``（绝对差之和）

SSIM 与 ``tools/verify_host_ssim.py`` 一致（RGB 三通道平均，data_range=1.0，win=7）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def bits_from_logits(decoded: Tensor) -> Tensor:
    """logits → 0/1 比特（>=0.5 记 1）。"""
    return (decoded.detach() >= 0.5).float()


def bit_errors(pred: Tensor, target: Tensor) -> float:
    """比特错误数（绝对差之和）。"""
    return float((pred - target.detach().float()).abs().sum().item())


def psnr_neg1_1_mean(a: Tensor, b: Tensor) -> float:
    """[-1,1] 图逐样本 PSNR 后取均值；峰值范围 2 → ``10*log10(4/MSE)``。

    与 solver.py ``_psnr_neg1_1`` / ``_psnr_neg1_1_mean`` 数值一致。
    """
    if a.shape != b.shape or a.numel() == 0:
        return float("nan")
    mse = torch.mean((a.detach() - b.detach()) ** 2, dim=(1, 2, 3))
    psnr = torch.where(
        mse < 1e-10,
        torch.full_like(mse, 99.0),
        10.0 * torch.log10(4.0 / mse.clamp_min(1e-10)),
    )
    return float(psnr.mean().item())


# ---------------------------------------------------------------------------
# SSIM（RGB 三通道平均）
# ---------------------------------------------------------------------------
try:
    from skimage.metrics import structural_similarity as _sk_ssim

    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False


def _to_hwc01(t: Tensor) -> np.ndarray:
    """[C,H,W] [-1,1] → [H,W,C] [0,1] numpy。"""
    arr = t.detach().float().cpu().numpy()
    arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    return np.transpose(arr, (1, 2, 0))


def ssim_rgb(a: Tensor, b: Tensor) -> float:
    """两个 [-1,1] 张量（[C,H,W] 或 [1,C,H,W]）→ RGB 三通道平均 SSIM。"""
    if a.dim() == 4:
        a = a[0]
    if b.dim() == 4:
        b = b[0]
    ha = _to_hwc01(a)
    hb = _to_hwc01(b)
    vals = []
    for c in range(3):
        if _HAS_SKIMAGE:
            vals.append(float(_sk_ssim(ha[:, :, c], hb[:, :, c], data_range=1.0, win_size=7)))
        else:  # 无 skimage 时的简易回退
            vals.append(_ssim_gray_fallback(ha[:, :, c], hb[:, :, c]))
    return float(np.mean(vals))


def _ssim_gray_fallback(x: np.ndarray, y: np.ndarray, win: int = 7) -> float:
    """无 skimage 时的 8-bit 风格 SSIM 回退（均值滤波窗）。"""
    c1, c2 = 0.01 ** 2, 0.03 ** 2

    def _box(img: np.ndarray) -> np.ndarray:
        from numpy.lib import stride_tricks

        pad = win // 2
        p = np.pad(img, pad, mode="reflect")
        w = stride_tricks.sliding_window_view(p, (win, win))
        k = np.ones((win, win), dtype=np.float64) / (win * win)
        return np.einsum("ijkl,kl->ij", w, k)

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mx, my = _box(x), _box(y)
    sx = _box(x * x) - mx * mx
    sy = _box(y * y) - my * my
    sxy = _box(x * y) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sx + sy + c2)
    return float(np.mean(num / den))
