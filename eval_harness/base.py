#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.base
=================
统一适配器接口：``ModelSpec``（模型元信息）与 ``WatermarkModel``（抽象接口）。

**harness 规范（唯一公共约定）**
---------------------------------
- 图像：float32 ``[B, 3, H, W]``，范围 ``[-1, 1]``，通道序 **BGR（cv2）**
  （与仓库 ``data_loader`` 一致；RGB 原生模型在 adapter 内部转换）。
- 消息：float32 ``[B, bits]``，取值 0/1。
- ``encode(x, m)`` / ``decode(x)`` 内部自行做分辨率缩放与数值域转换；
  adapter 对外只暴露「吃图、吐消息 logits」这一件事。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ModelSpec:
    """单个模型的静态元信息。"""

    name: str
    message_bits: int
    input_size: Tuple[int, int]  # (H, W)
    label: str = ""
    supports_encode: bool = True
    fixed_message: bool = False  # 消息由模型全局固定（如 StegaStamp 内置消息）

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name


class WatermarkModel(ABC):
    """统一接口：吃一张图，吐一串消息 logits。

    子类需实现 ``decode``；``encode`` / ``native_noise`` 按需覆盖。
    """

    spec: ModelSpec

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")

    def to(self, device: torch.device) -> "WatermarkModel":
        """记录目标设备；具体网络在 build 时按 ``self.device`` 放置。"""
        self.device = device
        return self

    @abstractmethod
    def decode(self, x: Tensor) -> Tensor:
        """输入 [-1,1] 图（任意 H,W，内部自行缩放）→ 消息 logits [B, bits]（未取整）。"""

    def encode(self, x: Tensor, m: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.spec.name} 不支持 encode")

    def native_noise(self, x: Tensor) -> Optional[Tensor]:
        """模型自带可微噪声层（电脑模拟用）；返回 None 表示无自带噪声。"""
        return None


def resize_canonical(x: Tensor, size: Tuple[int, int]) -> Tensor:
    """把 [-1,1] 的 [B,3,H,W] 缩放到 ``size=(H,W)``；尺寸一致时原样返回。"""
    if x.shape[-2] == size[0] and x.shape[-1] == size[1]:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def load_state_dict_strip(net, weight_path, device) -> Tuple[list, list]:
    """加载 ``.pth`` 到 ``net``，兼容两类常见包装：

    - ``DataParallel`` 保存的 ``module.`` 前缀（自动剥离）；
    - ``{"state_dict": ...}`` 外层包装。
    """
    state = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    return net.load_state_dict(state, strict=False)
