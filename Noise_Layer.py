#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise_Layer.py
==============
PIMoG 可微 / 可训练噪声层：模拟屏幕拍摄（Screen-shooting）失真。

包含变换
--------
几何
  translate / rotate / perspective   （基于 Kornia 仿射 / 透视）

光学
  Light_Distortion   方向或径向光照不均
  MoireGen / Moire_Distortion   摩尔纹图案

封装模块
--------
  ScreenShooting     透视 + 光照 + 摩尔纹 + 高斯噪声（论文主噪声层）
  ScreenShootingMB   运动模糊 + 透视 + 光照/摩尔纹（反比）+ 高斯
  Identity           恒等映射（消融 / 无失真基线）

公开接口
--------
- ScreenShooting, ScreenShootingMB, Identity
- perspective, translate, rotate, Light_Distortion, Moire_Distortion
- blur_moire_weights  （抖动强度 → 摩尔纹反比权重）
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.transform.imgwarp import (
    get_perspective_transform,
    get_rotation_matrix2d,
    warp_affine,
    warp_perspective,
)

# AS_VALs 可微运动模糊
_MB_DIR = Path(__file__).resolve().parent / "AS_VALs" / "python_motion_blur"
if str(_MB_DIR) not in sys.path:
    sys.path.insert(0, str(_MB_DIR))
from motion_blur import (  # noqa: E402
    mean_trajectory_length,
    nonlinear_motion_blur,
    sample_pose_pair,
)


# ---------------------------------------------------------------------------
# 仿射矩阵辅助（内部）
# ---------------------------------------------------------------------------
def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """由平移向量构造 3×3 齐次矩阵。"""
    matrix: torch.Tensor = torch.eye(3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)
    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """计算 HW / CHW / BCHW 张量平面中心。"""
    assert 2 <= len(tensor.shape) <= 4, f"Must be HW/CHW/BCHW. Got {tensor.shape}."
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    return torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)


def _compute_scaling_matrix(scale: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """由缩放因子构造仿射矩阵。"""
    angle: torch.Tensor = torch.zeros(scale.shape[0])
    return get_rotation_matrix2d(center, angle, scale)


def _compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """由旋转角构造纯旋转仿射矩阵。"""
    scale: torch.Tensor = torch.ones((angle.shape[0], 2))
    return get_rotation_matrix2d(center, angle, scale)


# ---------------------------------------------------------------------------
# 几何变换
# ---------------------------------------------------------------------------
def translate(image: torch.Tensor, device, d: float = 8) -> torch.Tensor:
    """
    对 batch 内每张图施加随机平移（幅度 ∈ [-d, d]）。

    Parameters
    ----------
    image : Tensor  BCHW 或 CHW
    device : torch.device
    d : float
        最大平移像素。
    """
    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]
    trans = torch.ones(c, 2)
    for i in range(c):
        dx = random.uniform(-d, d)
        dy = random.uniform(-d, d)
        trans[i, :] = torch.tensor([[dx, dy]])
    translation_matrix: torch.Tensor = _compute_translation_matrix(trans)
    matrix = translation_matrix[..., :2, :3]

    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    matrix = matrix.expand(image.shape[0], -1, -1).to(device)
    data_warp: torch.Tensor = warp_affine(
        image, matrix, dsize=(h, w), padding_mode="border"
    ).to(device)

    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)
    return data_warp


def rotate(image: torch.Tensor, device, d: float = 8) -> torch.Tensor:
    """对 batch 内每张图施加随机旋转（角度 ∈ [-d, d] 度）。"""
    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]
    angle = torch.ones(c)
    center = torch.ones(c, 2)
    for i in range(c):
        an = random.uniform(-d, d)
        angle[i] = torch.tensor([an])
        center[i, :] = torch.tensor([[h / 2 - 1, w / 2 - 1]])

    angle = angle.expand(image.shape[0])
    center = center.expand(image.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)
    matrix = rotation_matrix[..., :2, :3]

    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    matrix = matrix.expand(image.shape[0], -1, -1).to(device)
    data_warp: torch.Tensor = warp_affine(
        image, matrix, dsize=(h, w), padding_mode="border"
    ).to(device)

    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)
    return data_warp


def perspective(image: torch.Tensor, device, d: float = 8) -> torch.Tensor:
    """
    随机四点透视扭曲（模拟拍摄角度偏移）。

    Parameters
    ----------
    image : Tensor [B, C, H, W]
    d : float
        四角扰动最大像素偏移。
    """
    c = image.shape[0]
    h = image.shape[2]
    w = image.shape[3]
    image_size = h
    points_src = torch.ones(c, 4, 2)
    points_dst = torch.ones(c, 4, 2)
    for i in range(c):
        points_src[i, :, :] = torch.tensor(
            [[[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]]]
        )
        tl_x = random.uniform(-d, d)
        tl_y = random.uniform(-d, d)
        bl_x = random.uniform(-d, d)
        bl_y = random.uniform(-d, d)
        tr_x = random.uniform(-d, d)
        tr_y = random.uniform(-d, d)
        br_x = random.uniform(-d, d)
        br_y = random.uniform(-d, d)
        points_dst[i, :, :] = torch.tensor(
            [[
                [tl_x, tl_y],
                [tr_x + image_size, tr_y],
                [br_x + image_size, br_y + image_size],
                [bl_x, bl_y + image_size],
            ]]
        )

    M: torch.Tensor = get_perspective_transform(points_src, points_dst).to(device)
    data_warp: torch.Tensor = warp_perspective(
        image.float(), M, dsize=(h, w)
    ).to(device)
    return data_warp


# ---------------------------------------------------------------------------
# 光学失真
# ---------------------------------------------------------------------------
def MoireGen(p_size, theta, center_x, center_y) -> np.ndarray:
    """
    生成单通道摩尔纹图案（向量化，远快于双重 Python 循环）。

    Parameters
    ----------
    p_size : int
        图案边长。
    theta : float
        条纹角度（度）。
    center_x, center_y : float
        同心圆中心。
    """
    if hasattr(theta, "__len__"):
        theta = float(np.asarray(theta).reshape(-1)[0])
    if hasattr(center_x, "__len__"):
        center_x = float(np.asarray(center_x).reshape(-1)[0])
    if hasattr(center_y, "__len__"):
        center_y = float(np.asarray(center_y).reshape(-1)[0])

    i = np.arange(p_size, dtype=np.float64)[:, None] + 1.0
    j = np.arange(p_size, dtype=np.float64)[None, :] + 1.0
    z1 = 0.5 + 0.5 * np.cos(
        2 * np.pi * np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
    )
    z2 = 0.5 + 0.5 * np.cos(
        np.cos(theta / 180.0 * np.pi) * j + np.sin(theta / 180.0 * np.pi) * i
    )
    z = np.minimum(z1, z2)
    return (z + 1.0) / 2.0


def Light_Distortion(c: int, embed_image: torch.Tensor) -> np.ndarray:
    """
    生成光照不均匀掩膜。

    Parameters
    ----------
    c : int
        0 = 线性方向渐变；非 0 = 径向渐变。
    embed_image : Tensor [B, C, H, W]
        仅用于读取形状。
    """
    mask = np.zeros(embed_image.shape, dtype=np.float32)
    h, w = int(embed_image.shape[2]), int(embed_image.shape[3])
    mask_2d = np.zeros((h, w), dtype=np.float32)
    a = float(0.7 + np.random.rand() * 0.2)
    b = float(1.1 + np.random.rand() * 0.2)
    if c == 0:
        _ = np.random.randint(1, 5)
        for i in range(h):
            mask_2d[i, :] = -((b - a) / max(h - 1, 1)) * (i - w) + a
        for batch in range(embed_image.shape[0]):
            for channel in range(embed_image.shape[1]):
                mask[batch, channel, :, :] = mask_2d
        return mask

    x = np.random.randint(0, h)
    y = np.random.randint(0, w)
    # 使用真实分辨率，而非硬编码 255（原实现在 128 图上会算错径向尺度）
    corners = [
        np.sqrt(x ** 2 + y ** 2),
        np.sqrt((x - (h - 1)) ** 2 + y ** 2),
        np.sqrt(x ** 2 + (y - (w - 1)) ** 2),
        np.sqrt((x - (h - 1)) ** 2 + (y - (w - 1)) ** 2),
    ]
    max_len = float(np.max(corners)) + 1e-8
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))
    radial = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / max_len * (a - b) + b
    mask[:, :, :, :] = radial.astype(np.float32)
    return mask


def Moire_Distortion(embed_image: torch.Tensor) -> np.ndarray:
    """为 RGB 三通道分别生成独立摩尔纹。"""
    Z = np.zeros(embed_image.shape, dtype=np.float32)
    for i in range(3):
        theta = np.random.randint(0, 180)
        center_x = np.random.rand() * embed_image.shape[2]
        center_y = np.random.rand() * embed_image.shape[3]
        M = MoireGen(embed_image.shape[2], theta, center_x, center_y)
        Z[:, i, :, :] = M.astype(np.float32)
    return Z


# ---------------------------------------------------------------------------
# 抖动 ↔ 摩尔纹 权重约束
# ---------------------------------------------------------------------------
def blur_moire_weights(
    blur_strength: float,
    *,
    moire_max: float = 0.15,
    blur_ref: float = 40.0,
) -> tuple[float, float, float]:
    """
    由运动模糊强度计算光照/摩尔纹凸组合权重（反比约束）。

    定义归一化抖动强度
        β = clip(blur_strength / blur_ref, 0, 1)
    则
        α_moire = moire_max * (1 - β)   # 抖动越强，摩尔纹越弱
        α_light = 1 - α_moire           # 保持 α_light + α_moire = 1

    当 blur_strength=0 时退化为论文默认 (0.85, 0.15)；
    当 blur_strength≥blur_ref 时 α_moire→0。

    Parameters
    ----------
    blur_strength : float
        抖动强度标量，建议用轨迹长度均值 mean(|p1-p0|)。
    moire_max : float
        无抖动时的摩尔纹上限（论文默认 0.15）。
    blur_ref : float
        使 β=1 的参考抖动强度（像素）。

    Returns
    -------
    alpha_light, alpha_moire, beta
    """
    beta = float(np.clip(blur_strength / max(blur_ref, 1e-8), 0.0, 1.0))
    alpha_moire = float(moire_max) * (1.0 - beta)
    alpha_light = 1.0 - alpha_moire
    return alpha_light, alpha_moire, beta


# ---------------------------------------------------------------------------
# nn.Module 封装
# ---------------------------------------------------------------------------
class ScreenShooting(nn.Module):
    """
    论文核心噪声层：透视扭曲 → 光照 ×α_light + 摩尔纹 ×α_moire → 高斯噪声。

    默认 α_light=0.85, α_moire=0.15（与官方一致）。
    可通过 ``moire_weight`` / ``blur_strength`` 启用「抖动↑ → 摩尔纹↓」约束。

    前向过程中含随机采样，训练时每个 batch 失真不同。
    """

    def __init__(
        self,
        moire_weight: float | None = None,
        moire_max: float = 0.15,
        blur_ref: float = 40.0,
    ):
        super().__init__()
        self.moire_weight = moire_weight  # None → 默认 0.15；也可在 forward 覆盖
        self.moire_max = moire_max
        self.blur_ref = blur_ref

    def forward(
        self,
        embed_image: torch.Tensor,
        moire_weight: float | None = None,
        blur_strength: float | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embed_image : Tensor [B,C,H,W]  约在 [-1,1]
        moire_weight : float, optional
            直接指定摩尔纹权重 α_moire；光照权重 = 1 - α_moire。
        blur_strength : float, optional
            若给出，则用 :func:`blur_moire_weights` 由抖动强度反比得到 α_moire
            （优先于 ``moire_weight``）。
        """
        device = embed_image.device
        dtype = embed_image.dtype
        noised_image = perspective(embed_image, device, 2)

        if blur_strength is not None:
            alpha_light, alpha_moire, _beta = blur_moire_weights(
                blur_strength,
                moire_max=self.moire_max,
                blur_ref=self.blur_ref,
            )
        else:
            alpha_moire = (
                self.moire_weight if moire_weight is None else moire_weight
            )
            if alpha_moire is None:
                alpha_moire = self.moire_max  # 0.15
            alpha_moire = float(np.clip(alpha_moire, 0.0, 1.0))
            alpha_light = 1.0 - alpha_moire

        c = np.random.randint(0, 2)
        L = Light_Distortion(c, embed_image)
        Z = Moire_Distortion(embed_image) * 2.0 - 1.0
        Li = torch.as_tensor(L, device=device, dtype=dtype)
        Mo = torch.as_tensor(Z, device=device, dtype=dtype)
        noised_image = noised_image * Li * alpha_light + Mo * alpha_moire
        noised_image = noised_image + (0.001 ** 0.5) * torch.randn(
            noised_image.size(), device=device, dtype=dtype
        )
        return noised_image


class ScreenShootingMB(nn.Module):
    """
    扩展噪声层：可微非线性运动模糊 + PIMoG 拍屏失真。

    管线
    ----
    1. 非线性运动模糊（匀加速路径 λ(t)=t² + 等时间曝光）
    2. 透视扭曲
    3. 光照 ×α_light + 摩尔纹 ×α_moire
    4. 高斯噪声

    反比约束
    --------
    s = mean(|p1−p0|)
    β = clip(s / blur_ref, 0, 1)
    α_moire = moire_max · (1 − β)   # 抖动↑ → 摩尔纹↓
    α_light = 1 − α_moire

    训练时输入约为 [-1,1] 的 B×C×H×W（通常 H=W=128）；
    取景窗取满幅 view_size=H，在图上做空变抖动。
    """

    def __init__(
        self,
        moire_max: float = 0.15,
        blur_ref: float | None = None,
        max_trans_ratio: float = 0.06,
        max_rot_deg: float = 12.0,
        n_samples: int = 8,
        path_power: float = 2.0,
        perspective_d: float = 2.0,
        gauss_sigma: float | None = None,
    ):
        super().__init__()
        self.moire_max = moire_max
        self.blur_ref = blur_ref  # None → 按分辨率自适应
        self.max_trans_ratio = max_trans_ratio
        self.max_rot_deg = max_rot_deg
        self.n_samples = n_samples
        self.path_power = path_power
        self.perspective_d = perspective_d
        # gauss_sigma 已废弃（原高斯曝光核）；保留参数名以免旧调用报错
        del gauss_sigma

    def forward(self, embed_image: torch.Tensor) -> torch.Tensor:
        if embed_image.ndim != 4:
            raise ValueError(f"expected BCHW, got {tuple(embed_image.shape)}")

        device = embed_image.device
        dtype = embed_image.dtype
        _b, _c, h, w = embed_image.shape
        view_size = int(min(h, w))

        max_trans = max(1.0, self.max_trans_ratio * view_size)
        blur_ref = (
            float(self.blur_ref)
            if self.blur_ref is not None
            else max(8.0, 0.15 * view_size)
        )

        # 运动模糊在 [0,1] 更直观；算完再映回 [-1,1]
        img01 = (embed_image.clamp(-1.0, 1.0) + 1.0) * 0.5
        try:
            cx0, cy0, th0, cx1, cy1, th1, _rej = sample_pose_pair(
                width=w,
                height=h,
                view_size=view_size,
                max_trans=max_trans,
                max_rot_deg=self.max_rot_deg,
                n_check=self.n_samples,
                device=device,
            )
        except RuntimeError:
            # 极端尺寸下采样失败：退化为轻微中心抖动
            half = (view_size - 1) / 2.0
            cx0 = cy0 = half + 1.0
            th0 = 0.0
            cx1 = cx0 + max_trans * 0.5
            cy1 = cy0
            th1 = math.radians(self.max_rot_deg * 0.5)

        blurred01, aux = nonlinear_motion_blur(
            img01,
            cx0,
            cy0,
            th0,
            cx1,
            cy1,
            th1,
            view_size=view_size,
            n_samples=self.n_samples,
            path_power=self.path_power,
        )
        s = mean_trajectory_length(aux)
        alpha_light, alpha_moire, _beta = blur_moire_weights(
            s, moire_max=self.moire_max, blur_ref=blur_ref
        )

        noised = blurred01 * 2.0 - 1.0
        noised = perspective(noised, device, self.perspective_d)

        c = np.random.randint(0, 2)
        L = Light_Distortion(c, noised)
        Z = Moire_Distortion(noised) * 2.0 - 1.0
        Li = torch.as_tensor(L, device=device, dtype=dtype)
        Mo = torch.as_tensor(Z, device=device, dtype=dtype)
        noised = noised * Li * alpha_light + Mo * alpha_moire
        noised = noised + (0.001 ** 0.5) * torch.randn(
            noised.size(), device=device, dtype=dtype
        )
        return noised


class Identity(nn.Module):
    """恒等噪声层：不做任何失真（消融实验基线 / warmup）。"""

    def __init__(self):
        super().__init__()

    def forward(self, embed_image: torch.Tensor) -> torch.Tensor:
        return embed_image
