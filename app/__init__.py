# -*- coding: utf-8 -*-
"""
app
===
DMB of PMoG 应用层：参数解析、模式解析、工具分发、训练入口。
"""

from app.constants import REPO_ROOT, TOOL_MODE_CHOICES
from app.entry import cli_main
from app.parser import build_parser
from app.run import main
from app.tools import run_tool_mode

__all__ = [
    "REPO_ROOT",
    "TOOL_MODE_CHOICES",
    "build_parser",
    "cli_main",
    "main",
    "run_tool_mode",
]
