#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.registry
=====================
模型注册表：名称 → 构建工厂。

新增第三方模型时，在 ``adapters/<name>.py`` 里用 ``@registry.register("<name>")``
装饰一个 ``build_*(**kw) -> WatermarkModel`` 工厂，并在 ``adapters/__init__.py``
里 import 该模块即可。详见 ``adapters/README.md``。
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .base import WatermarkModel

_BUILDERS: Dict[str, Callable] = {}


def register(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        _BUILDERS[name] = fn
        return fn

    return deco


def available() -> List[str]:
    return sorted(_BUILDERS)


def build(name: str, **kw) -> WatermarkModel:
    if name not in _BUILDERS:
        raise KeyError(f"未知模型 {name!r}，可用: {available()}")
    return _BUILDERS[name](**kw)
