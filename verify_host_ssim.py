#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容入口：实现已移至 ``tools/verify_host_ssim.py``。"""

from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "tools" / "verify_host_ssim.py"

if __name__ == "__main__":
    if not _TARGET.is_file():
        raise SystemExit(f"找不到脚本: {_TARGET}")
    runpy.run_path(str(_TARGET), run_name="__main__")
