#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.datasets
=====================
公共测试集加载：宿主图 / 拍屏图 / 消息矩阵 / 消息生成。

沿用仓库现有约定：
- 宿主图：``Datasets/images/``
- 矫正后拍屏图：``Datasets/Recover/capture/``
- 消息矩阵：``results/WatermarkMatrix/w.mat``（键 ``w``，``[100,30]``）
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(d: str) -> List[str]:
    """返回目录下图片文件名（按名字排序，尽量按数字序）。"""
    if not os.path.isdir(d):
        raise FileNotFoundError(f"目录不存在: {d}")
    names = os.listdir(d)
    out = [p for p in names if os.path.splitext(p)[1].lower() in _IMG_EXTS]
    if not out:  # 兜底：无扩展名也收进来
        out = [p for p in names if not p.startswith(".")]
    return sorted(out, key=lambda p: (filename_index(p), p))


def filename_index(fname: str) -> int:
    """从文件名解析索引（``00010.png`` → 10）。"""
    stem = os.path.splitext(os.path.basename(fname))[0]
    if stem.isdigit():
        return int(stem)
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", stem)
    if m:
        return int(m.group(1))
    raise ValueError(f"无法从文件名解析索引: {fname}")


def load_canonical(path: str, size: Optional[Tuple[int, int]] = None) -> Tensor:
    """读图 → ``[1,3,H,W]`` float32 ``[-1,1]``；``size=(H,W)`` 时先缩放。

    通道序保持 **cv2 BGR**（与仓库 ``data_loader`` 一致：``cv2.imread`` 后不转 RGB）。
    若某第三方模型原生 RGB（如 StegaStamp），其 adapter 内部自行转换。
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR，不再转 RGB
    if img is None:
        raise RuntimeError(f"无法读取图片: {path}")
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]))  # cv2 参数为 (W, H)
    arr = np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def load_message_matrix(w_path: str) -> np.ndarray:
    """读 ``w.mat``（键 ``w``）或 ``.npy`` → ``[rows, bits]``。"""
    if w_path.endswith(".npy"):
        return np.load(w_path)
    import scipy.io as scio

    W = scio.loadmat(w_path)
    for key in ("w", "W", "message", "messages"):
        if key in W:
            return np.asarray(W[key])
    raise KeyError(f"{w_path} 中未找到键 w/W/message/messages，实际键: {list(W.keys())}")


def generate_message(index: int, bits: int, seed: int = 0) -> np.ndarray:
    """生成与仓库 ``eval_mask`` 一致的可复现消息。

    ``eval_mask`` 用 ``RandomState(message_seed + index*10007)`` 采样 30-bit 0/1。
    """
    rng = np.random.RandomState(int(seed) + int(index) * 10007)
    return (rng.rand(bits) >= 0.5).astype(np.float32)
