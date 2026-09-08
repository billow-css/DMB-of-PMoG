#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ckpt_utils.py
=============
统一权重标签解析：mask_99 / ScreenShooting_best / ScreenShootingMB_best。

``eval_ckpt`` / ``init_from_ckpt`` 取值
--------------------------------------
- ``99``       → ``…/Encoder_Decoder_Model_mask_99.pth``
- ``ss_best``  → ``models/Encoder_Decoder_Model_mask_ScreenShooting_best.pth``
- ``best``     → ``models/Encoder_Decoder_Model_mask_ScreenShootingMB_best.pth``（DMB；兼容旧名）
- ``mb_best``  → 同 ``best`` 的别名
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 对外稳定标签（CLI / argparse）
CKPT_CHOICES: Tuple[str, ...] = ("99", "ss_best", "best", "mb_best")

_CKPT_META: Dict[str, Dict[str, Any]] = {
    "99": {
        "label": "mask_99",
        "short": "mask_99",
        "default_distortion": "ScreenShooting",
        "tag_for_run": "ep99",
        "is_named_best": False,
    },
    "ss_best": {
        "label": "ScreenShooting_best",
        "short": "SS_best",
        "default_distortion": "ScreenShooting",
        "tag_for_run": "ss_best",
        "is_named_best": True,
        "best_suffix": "ScreenShooting_best",
    },
    "best": {
        "label": "ScreenShootingMB_best",
        "short": "DMB_best",
        "default_distortion": "ScreenShootingMB",
        "tag_for_run": "mb_best",
        "is_named_best": True,
        "best_suffix": "ScreenShootingMB_best",
    },
    "mb_best": {
        "label": "ScreenShootingMB_best",
        "short": "DMB_best",
        "default_distortion": "ScreenShootingMB",
        "tag_for_run": "mb_best",
        "is_named_best": True,
        "best_suffix": "ScreenShootingMB_best",
    },
}


def normalize_ckpt_tag(tag: Optional[str], default: str = "99") -> str:
    t = str(tag or default).strip()
    if t == "mb_best":
        return "best"
    if t == "custom":
        return "custom"
    if t not in ("99", "ss_best", "best"):
        return default
    return t


def ckpt_label(tag: Optional[str]) -> str:
    t = normalize_ckpt_tag(tag)
    if t == "custom":
        return "custom_weight"
    return str(_CKPT_META[t]["label"])


def ckpt_short(tag: Optional[str]) -> str:
    t = normalize_ckpt_tag(tag)
    if t == "custom":
        return "custom"
    return str(_CKPT_META[t]["short"])


def ckpt_default_distortion(tag: Optional[str]) -> str:
    t = normalize_ckpt_tag(tag)
    if t == "custom":
        return "ScreenShootingMB"
    return str(_CKPT_META[t]["default_distortion"])


def ckpt_run_tag(tag: Optional[str], embedding_epoch: int = 99) -> str:
    t = normalize_ckpt_tag(tag)
    if t == "custom":
        return "custom"
    if t == "99":
        return f"ep{embedding_epoch}"
    return str(_CKPT_META[t]["tag_for_run"])


def named_best_filename(model_name: str, tag: str) -> str:
    t = normalize_ckpt_tag(tag)
    if t == "99":
        return f"{model_name}_mask_99.pth"
    if t == "custom":
        return f"{model_name}_mask_custom.pth"
    suffix = _CKPT_META[t]["best_suffix"]
    return f"{model_name}_mask_{suffix}.pth"


def candidate_paths(
    model_save_dir: str,
    model_name: str,
    tag: str,
    *,
    embedding_epoch: int = 99,
    distortion: Optional[str] = None,
) -> List[str]:
    """按优先级列出候选权重路径。"""
    t = normalize_ckpt_tag(tag)
    root = model_save_dir
    out: List[str] = []

    if t == "99":
        fname = f"{model_name}_mask_{embedding_epoch}.pth"
        if distortion:
            out.append(os.path.join(root, distortion, fname))
        for d in ("ScreenShooting", "ScreenShootingMB", "Identity"):
            if d != distortion:
                out.append(os.path.join(root, d, fname))
        out.append(os.path.join(root, fname))
        return out

    # named best：优先根目录，再 distortion 子目录
    fname = named_best_filename(model_name, t)
    out.append(os.path.join(root, fname))
    preferred = ckpt_default_distortion(t)
    out.append(os.path.join(root, preferred, fname))
    if distortion and distortion != preferred:
        out.append(os.path.join(root, distortion, fname))
    # 兼容旧布局：…/ScreenShooting/…_mask_ScreenShooting_best.pth 等
    for d in ("ScreenShooting", "ScreenShootingMB"):
        p = os.path.join(root, d, fname)
        if p not in out:
            out.append(p)
    return out


def resolve_ckpt_path(
    model_save_dir: str,
    model_name: str,
    tag: str,
    *,
    embedding_epoch: int = 99,
    distortion: Optional[str] = None,
) -> str:
    cands = candidate_paths(
        model_save_dir,
        model_name,
        tag,
        embedding_epoch=embedding_epoch,
        distortion=distortion,
    )
    for p in cands:
        if os.path.isfile(p):
            return p
    listing = "\n".join(f"  - {p}" for p in cands)
    raise FileNotFoundError(
        f"找不到权重标签 {normalize_ckpt_tag(tag)!r}（{ckpt_label(tag)}）:\n{listing}"
    )


def read_embed_strength_meta(weight_path: str | os.PathLike, default: float = 0.0) -> float:
    """读 ``*.meta.txt`` 或 ``stem.meta.txt`` 中的 embed_strength=。"""
    weight_path = os.fspath(weight_path)
    stem = weight_path
    if stem.lower().endswith(".pth"):
        stem = stem[:-4]
    for cand in (weight_path + ".meta.txt", stem + ".meta.txt"):
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("embed_strength="):
                        return float(line.split("=", 1)[1].strip())
        except (OSError, ValueError):
            continue
    return float(default)


def default_noise_menu_choice(tag: Optional[str]) -> str:
    """交互噪声层默认：1=ScreenShooting，2=ScreenShootingMB。"""
    t = normalize_ckpt_tag(tag)
    return "2" if t in ("best", "custom") else "1"
