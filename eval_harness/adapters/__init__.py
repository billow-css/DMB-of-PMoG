#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.adapters
=====================
各模型适配器。import 各模块即触发 ``@registry.register`` 注册。

已实现：``pimog`` / ``dmb_pmog`` / ``st_rep``。
骨架：``stegastamp`` / ``ropass`` / ``sim2real``（TODO，详见本目录 README.md）。
"""

from . import pimog, st_rep  # noqa: F401  已实现
from . import sim2real, stegastamp, ropass  # noqa: F401  TODO 骨架
