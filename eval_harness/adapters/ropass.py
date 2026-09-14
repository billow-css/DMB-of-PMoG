#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.adapters.ropass
============================
RoPaSS（AAAI 2025）—— TODO 骨架。

接入要点（已核实）：
- 128×128 宿主；子块水印 + homography/symmetry 同步估计；
- 消息长度 L（FC → 1×1024 → 32×32）；
- decode 接口最特殊（含同步/透视估计，接入成本最高）。
"""

from .. import registry


@registry.register("ropass")
def build_ropass(**kw):
    raise NotImplementedError(
        "RoPaSS 适配器尚未实现。要点：128×128、子块水印 + homography/symmetry 同步估计，"
        "decode 含同步恢复，接入成本最高。详见 adapters/README.md。"
    )
