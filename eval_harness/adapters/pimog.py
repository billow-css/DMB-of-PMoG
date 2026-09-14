#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.adapters.pimog
===========================
自研模型适配器：PIMoG（``ScreenShooting``）与 DMB-PMoG（``ScreenShootingMB``）。

复用现有 ``model.Encoder_Decoder`` 与 ``ckpt_utils``：
- ``decode``       = ``net.Decoder(x)``
- ``encode``       = ``net.encode(x, m)``（含 ``embed_strength`` 残差约束）
- ``native_noise`` = ``net.Noiser(x)``（电脑模拟的随机屏摄 / 运动模糊噪声层）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from torch import Tensor

from .. import registry
from ..base import ModelSpec, WatermarkModel, load_state_dict_strip, resize_canonical

REPO_ROOT = Path(__file__).resolve().parents[2]  # eval_harness/adapters/pimog.py → 仓库根


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


@registry.register("pimog")
def build_pimog(
    ckpt_tag: str = "ss_best",
    distortion: str = "ScreenShooting",
    embed_strength: Optional[float] = None,
    weight_path: Optional[str] = None,
    **kw,
) -> "PIMoGAdapter":
    return PIMoGAdapter(ckpt_tag=ckpt_tag, distortion=distortion,
                        embed_strength=embed_strength, weight_path=weight_path)


@registry.register("dmb_pmog")
def build_dmb(
    ckpt_tag: str = "best",
    embed_strength: Optional[float] = None,
    weight_path: Optional[str] = None,
    **kw,
) -> "PIMoGAdapter":
    return PIMoGAdapter(ckpt_tag=ckpt_tag, distortion="ScreenShootingMB",
                        embed_strength=embed_strength, weight_path=weight_path)


class PIMoGAdapter(WatermarkModel):
    """PIMoG / DMB-PMoG 适配器（30-bit、128×128、[-1,1]）。"""

    def __init__(
        self,
        ckpt_tag: str = "ss_best",
        distortion: str = "ScreenShooting",
        embed_strength: Optional[float] = None,
        weight_path: Optional[str] = None,
    ):
        super().__init__()
        is_dmb = distortion == "ScreenShootingMB"
        self.spec = ModelSpec(
            name="dmb_pmog" if is_dmb else "pimog",
            message_bits=30,
            input_size=(128, 128),
            label="DMB-PMoG" if is_dmb else "PIMoG",
        )
        self.distortion = distortion
        self.ckpt_tag = ckpt_tag
        self.embed_strength = embed_strength
        self._weight_path = weight_path
        self.net = None

    def _ensure(self):
        if self.net is None:
            _ensure_repo_on_path()
            from ckpt_utils import read_embed_strength_meta, resolve_ckpt_path
            from model import Encoder_Decoder

            weight = self._weight_path or resolve_ckpt_path(
                str(REPO_ROOT / "models"),
                "Encoder_Decoder_Model",
                self.ckpt_tag,
                embedding_epoch=99,
                distortion=self.distortion,
            )
            strength = self.embed_strength
            if strength is None:
                strength = read_embed_strength_meta(weight, default=0.0)
            net = Encoder_Decoder(self.distortion, embed_strength=strength)
            net.to(self.device)
            missing, unexpected = load_state_dict_strip(net, weight, self.device)
            if missing:
                print(f"  [pimog] 缺失 {len(missing)} 键（Noiser 等无参层可忽略）")
            if unexpected:
                print(f"  [pimog] 多余 {len(unexpected)} 键")
            net.eval()
            self.net = net
        return self.net

    def encode(self, x: Tensor, m: Tensor) -> Tensor:
        net = self._ensure()
        x = resize_canonical(x, (128, 128))
        return net.encode(x, m)

    def decode(self, x: Tensor) -> Tensor:
        net = self._ensure()
        x = resize_canonical(x, (128, 128))
        return net.Decoder(x)

    def native_noise(self, x: Tensor):
        net = self._ensure()
        return net.Noiser(x)
