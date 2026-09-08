#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容入口：请使用 ``AS_VALs/demos/python/demo_coco_nonlinear_motion_blur.py``。"""
from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = (
    Path(__file__).resolve().parents[1]
    / "demos"
    / "python"
    / "demo_coco_nonlinear_motion_blur.py"
)
runpy.run_path(str(_TARGET), run_name="__main__")
