#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.adapters.st_rep
============================
第三方模型适配器：*Screen-Shooting Robust Watermark Based on Style Transfer and
Structural Re-Parameterization*（风格迁移噪声 + RepConv 结构重参数化，后称 **ST-Rep**）。

与 PIMoG **数据协议完全相同**（30-bit、128×128、[-1,1]、``Decoder(x)→round→比对``），
差异只在架构与噪声层：
- 编码器 ``DoubleConv`` 用 ``RepConv2d``（多分支，checkpoint 存的是多分支权重，推理直接前向即可，
  **无需** ``switch_to_deploy``）；
- 解码器尾部加 ``MSCASpatialAttention(64)``；
- 噪声层替换为风格迁移网络 ``stylenet(embed_image)``（268MB，``ResNet`` 生成器）。

工程要点：第三方 ``model.py/Noise_Layer.py/…`` 与仓库根**同名冲突**，故用
``importlib.util.spec_from_file_location`` 按别名动态加载，并把第三方目录临时加入 ``sys.path``，
加载完即移除，避免与自研模块串包。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from .. import registry
from ..base import ModelSpec, WatermarkModel, load_state_dict_strip, resize_canonical

REPO_ROOT = Path(__file__).resolve().parents[2]
TP_DIR = REPO_ROOT / "Screen-Shooting Robust Watermark Based on Style Transfer and Structural Re-Parameterization"

# 第三方 model.py 顶层 import 的、可能与根仓库冲突的模块名
_COLLIDING = ("Noise_Layer", "module", "modules", "distoration_model")

_TP_MODEL = None  # 缓存的第三方 model 模块（别名 st_rep_model）


def _load_tp_module(sub_path: str, alias: str):
    """在临时 sys.path 下按别名加载第三方模块，隔离同名模块冲突。"""
    fp = TP_DIR / sub_path
    # 1) 保存并移除冲突名，保证第三方 import 到的是它自己的
    saved = {}
    for name in _COLLIDING:
        for k in list(sys.modules):
            if k == name or k.startswith(name + "."):
                saved[k] = sys.modules.pop(k)
    added = str(TP_DIR) not in sys.path
    if added:
        sys.path.insert(0, str(TP_DIR))
    try:
        spec = importlib.util.spec_from_file_location(alias, str(fp))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    finally:
        # 2) 清理第三方导入的顶层名（类已绑定到 mod 命名空间，删掉 sys.modules 无影响）
        for name in _COLLIDING:
            for k in list(sys.modules):
                if k == name or k.startswith(name + "."):
                    sys.modules.pop(k, None)
        # 3) 恢复根仓库同名模块
        sys.modules.update(saved)
        if added:
            sys.path.remove(str(TP_DIR))
    return mod


def _tp_model():
    global _TP_MODEL
    if _TP_MODEL is None:
        _TP_MODEL = _load_tp_module("model.py", "st_rep_model")
    return _TP_MODEL


def _style_resnet(device):
    """加载风格迁移生成器（仅 ``sim`` 模式的 ``native_noise`` 需要）。"""
    mod = _load_tp_module("distoration_model/model.py", "st_rep_style_model")
    netG = mod.ResNet(3, 3, 64, "inorm", nblk=6)
    ckpt = TP_DIR / "distoration_model" / "model" / "model_epoch0100.pth"
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = state["netG_a2b"]
    if any(str(k).startswith("module.") for k in sd):
        sd = {str(k)[7:]: v for k, v in sd.items()}
    netG.load_state_dict(sd)
    netG.to(device).eval()
    return netG


@registry.register("st_rep")
def build_st_rep(weight_path: Optional[str] = None, **kw) -> "STRepAdapter":
    return STRepAdapter(weight_path=weight_path)


class STRepAdapter(WatermarkModel):
    """ST-Rep 适配器（30-bit、128×128、[-1,1]）。"""

    def __init__(self, weight_path: Optional[str] = None):
        super().__init__()
        self.spec = ModelSpec(
            name="st_rep",
            message_bits=30,
            input_size=(128, 128),
            label="StyleTransfer+RepConv",
        )
        self._ckpt = str(weight_path) if weight_path else str(
            TP_DIR / "models" / "ScreenShooting" / "Encoder_Decoder_Model_mask_96.pth"
        )
        self._net = None
        self._stylenet = None

    def _ensure(self):
        if self._net is None:
            m = _tp_model()
            net = m.Encoder_Decoder("ScreenShooting")
            net.to(self.device)
            missing, unexpected = load_state_dict_strip(net, self._ckpt, self.device)
            if missing:
                print(f"  [st_rep] 缺失 {len(missing)} 键")
            if unexpected:
                print(f"  [st_rep] 多余 {len(unexpected)} 键")
            net.eval()
            self._net = net
        return self._net

    def encode(self, x: Tensor, m: Tensor) -> Tensor:
        net = self._ensure()
        x = resize_canonical(x, (128, 128))
        return net.Encoder(x, m)

    def decode(self, x: Tensor) -> Tensor:
        net = self._ensure()
        x = resize_canonical(x, (128, 128))
        return net.Decoder(x)

    def native_noise(self, x: Tensor):
        if self._stylenet is None:
            self._stylenet = _style_resnet(self.device)
        return self._stylenet(x)
