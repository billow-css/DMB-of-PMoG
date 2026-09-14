#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.adapters.stegastamp
================================
StegaStamp（Tancik, CVPR 2020）—— TODO 骨架。

接入要点（已核实）：
- 消息：100-bit（内部 BCH 纠错后为 56-bit payload），``fixed_message`` 往往全局固定；
- 分辨率：400×400；编码为 U-Net 残差 ``x' = x + r``；
- 解码：CNN + 可选 STN（空间变换网络）+ BCH 纠错；
- 权重：官方 ``saved_models/stegastamp_pretrained/``（TF/PT 双版本）。

实现时需在 ``encode`` 里忽略传入 ``m``、用其内置消息；``decode`` 与内置消息比对。
"""

from .. import registry


@registry.register("stegastamp")
def build_stegastamp(**kw):
    raise NotImplementedError(
        "StegaStamp 适配器尚未实现。要点：100-bit（BCH-ECC 后 56-bit）、400×400、"
        "U-Net 残差编码 x'=x+r、解码含可选 STN + BCH。详见 adapters/README.md。"
    )
