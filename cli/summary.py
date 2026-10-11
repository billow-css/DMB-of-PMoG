# -*- coding: utf-8 -*-
"""运行配置摘要打印。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cli.color import _C, _paint

_PRIORITY = [
    "mode", "dataset", "distortion", "eval_ckpt", "weight_path",
    "lite", "lite_ratio", "lite_seed",
    "warmup_epochs", "embed_strength", "init_from_ckpt", "init_from_epoch",
    "image_dir", "image_val_dir", "psnr_ref_dir", "batch_size", "num_epoch",
    "embedding_epoch", "image_size", "device", "train_samples",
    "val_samples", "lambda1", "lambda2", "lambda3", "lr",
    "save_viz", "result_dir",
]
_HIGHLIGHT = {
    "weight_path", "mode", "eval_ckpt", "embedding_epoch",
    "init_from_ckpt", "init_from_epoch", "distortion", "embed_strength",
}
_TITLES = {
    "eval_mask": "  评估配置 (eval_mask)",
    "train_mask": "  训练配置 (train_mask)",
    "test_embedding": "  嵌入配置 (test_embedding)",
    "test_accuracy": "  测准配置 (test_accuracy)",
}


def print_config_summary(config: Any, extra: Optional[Dict[str, Any]] = None) -> None:
    """以两列表格打印 argparse Namespace（及额外键值）。"""
    rows: Dict[str, Any] = {}
    if hasattr(config, "__dict__"):
        rows.update(vars(config))
    if extra:
        rows.update(extra)

    title = _TITLES.get(str(rows.get("mode", "")), "  运行配置")
    keys = [k for k in _PRIORITY if k in rows] + [
        k for k in sorted(rows.keys()) if k not in _PRIORITY
    ]

    print(_paint(title, _C.BOLD, _C.WHITE), flush=True)
    key_w = max(len(str(k)) for k in keys) if keys else 8
    for k in keys:
        v = rows[k]
        key_s = f"    {str(k):<{key_w}}  "
        val_s = str(v)
        if k in _HIGHLIGHT:
            print(_paint(key_s, _C.DIM) + _paint(val_s, _C.BOLD, _C.CYAN), flush=True)
        else:
            print(_paint(key_s, _C.DIM) + val_s, flush=True)
    print(_paint("─" * 56, _C.DIM), flush=True)
