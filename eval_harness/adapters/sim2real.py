#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.adapters.sim2real
==============================
SIM2Real —— TODO 骨架。

接入要点（已核实）：
- 32-bit 消息；128×128 量级；自带 sim2real 噪声层。
"""

from .. import registry


@registry.register("sim2real")
def build_sim2real(**kw):
    raise NotImplementedError(
        "SIM2Real 适配器尚未实现。要点：32-bit、128×128、自带 sim2real 噪声层。"
        "详见 adapters/README.md。"
    )
