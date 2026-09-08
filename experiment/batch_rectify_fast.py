#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容入口：快速批量矫正已提升为 ``batch_rectify.py``。

本文件保留对 ``list_images`` / ``parse_index`` 等的再导出，
以免 ``rectify_gui`` 等旧 import 失效。

请优先::

    python batch_rectify.py

旧版慢速全图检测见 ``legacy/batch_rectify_legacy.py``。
"""

from __future__ import annotations

import sys

# 再导出公共 API（供 rectify_gui 等 import）
from batch_rectify import (  # noqa: F401
    IMAGE_EXTS,
    list_images,
    main,
    parse_index,
    pick_folder,
)

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
