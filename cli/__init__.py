# -*- coding: utf-8 -*-
"""
cli
===
DMB of PMoG 命令行前端：横幅、功能面板、配置摘要、彩色日志、进度条。

不参与模型计算，仅负责人机交互。
"""

from cli.brand import (
    PROJECT_FULL,
    PROJECT_SHORT,
    PROJECT_TAGLINE,
    print_banner,
    print_lite_badge,
    print_mode_badge,
)
from cli.color import _C, _paint
from cli.input_util import ask_choice
from cli.log import log_err, log_info, log_ok, log_step, log_warn
from cli.progress import ProgressMonitor, format_eta, wrap_iterable
from cli.prompts_ckpt import prompt_eval_ckpt, prompt_noise_layer
from cli.prompts_log import prompt_log_browser
from cli.prompts_mode import prompt_run_mode
from cli.prompts_train import prompt_train_init, prompt_train_scale
from cli.summary import print_config_summary

# 兼容旧名
_ask_choice = ask_choice

__all__ = [
    "PROJECT_FULL",
    "PROJECT_SHORT",
    "PROJECT_TAGLINE",
    "ProgressMonitor",
    "_C",
    "_ask_choice",
    "_paint",
    "ask_choice",
    "format_eta",
    "log_err",
    "log_info",
    "log_ok",
    "log_step",
    "log_warn",
    "print_banner",
    "print_config_summary",
    "print_lite_badge",
    "print_mode_badge",
    "prompt_eval_ckpt",
    "prompt_log_browser",
    "prompt_noise_layer",
    "prompt_run_mode",
    "prompt_train_init",
    "prompt_train_scale",
    "wrap_iterable",
]
