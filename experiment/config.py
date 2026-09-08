#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/config.py
====================
屏摄联机实验默认配置（5000+ 张规模友好）。

冷却时间说明（默认一轮约 2.5–3.5 s，5000 张约 3.5–5 h）：
  show → AF → ready → shoot → download → ack → next
过短易导致 AF/写卡失败；过长则总时长爆炸。可按实机再调。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict
import json


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


@dataclass
class CooldownConfig:
    """各步骤冷却（秒）。"""

    # 客户端切图后稍等，再让机身 AF（避免切图瞬间抖动）
    after_show_s: float = 0.25
    # AF 指令后等待合焦稳定（实机常需 0.4–1.0）
    after_af_s: float = 0.60
    # 收到 READY 后再快门（给机身一点余量）
    after_ready_before_shoot_s: float = 0.25
    # 快门后等到「拍摄完毕 / 文件落盘」（写卡+USB 回传，批量时最关键）
    after_shoot_s: float = 1.8
    # 向客户端发 ACK(1) 后，再发下一张 SHOW
    after_ack_before_next_s: float = 0.2
    # 单张拍摄失败重试次数（服务器层）
    shoot_retries: int = 3
    # 重试仍失败则跳过继续（大批量勿因一张中断）
    skip_on_shoot_fail: bool = True
    # 屏摄默认不依赖 AF（避免 0x8D01 AF_NG）
    shutter_non_af: bool = True
    # 关闭 PC 实时取景可明显加快 USB 回传（屏摄不需要 EVF）
    keep_liveview: bool = False
    # 下载完成后额外泵事件（电子快门可 0.03～0.08）
    post_download_pump_s: float = 0.05
    # 每 N 张额外长休息，防机身过热 / 写卡缓冲堆积
    batch_rest_every: int = 250
    batch_rest_s: float = 0.5
    # 每 N 张强制 GC / 刷盘日志
    checkpoint_every: int = 100


@dataclass
class NetConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    # 客户端连接超时
    connect_timeout_s: float = 30.0
    # 单次收发超时（大批量时 READY/ACK 不应卡死）
    recv_timeout_s: float = 120.0
    # 心跳间隔（秒）；0 关闭
    heartbeat_s: float = 30.0


@dataclass
class PathConfig:
    # 客户端本地图库（与服务器 index 对齐的编号图，如 0.png）
    image_dir: str = str(REPO_ROOT / "Datasets" / "images_watermarked")
    # 相机回传保存根目录
    capture_root: str = str(ROOT / "captures")
    # 会话状态 / 断点
    session_root: str = str(ROOT / "sessions")
    log_dir: str = str(ROOT / "logs")
    # 子目录分片：每 shard_size 张一个子文件夹，避免单目录数万文件
    shard_size: int = 500
    # EDSDK Dll 目录（Windows 64 位默认指向随仓库的 13.20.21）
    edsdk_dll_dir: str = str(
        ROOT / "EDSDK132011CD(13.20.21)" / "Windows" / "EDSDK_64" / "Dll"
    )


@dataclass
class ExperimentConfig:
    protocol_version: int = 1
    cooldowns: CooldownConfig = field(default_factory=CooldownConfig)
    net: NetConfig = field(default_factory=NetConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    # stub | edsdk（edsdk 申请通过后切换）
    camera_backend: str = "stub"
    # 仅处理 [start_index, end_index)，None=全部
    start_index: int = 0
    end_index: int | None = None
    # 断点续跑
    resume: bool = True
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        cd = d.get("cooldowns", {})
        nd = d.get("net", {})
        pd = d.get("paths", {})
        return cls(
            protocol_version=int(d.get("protocol_version", 1)),
            cooldowns=CooldownConfig(**{k: v for k, v in cd.items() if k in CooldownConfig.__dataclass_fields__}),
            net=NetConfig(**{k: v for k, v in nd.items() if k in NetConfig.__dataclass_fields__}),
            paths=PathConfig(**{k: v for k, v in pd.items() if k in PathConfig.__dataclass_fields__}),
            camera_backend=str(d.get("camera_backend", "stub")),
            start_index=int(d.get("start_index", 0)),
            end_index=d.get("end_index", None),
            resume=bool(d.get("resume", True)),
            dry_run=bool(d.get("dry_run", False)),
        )


def default_config_path() -> Path:
    return ROOT / "config.json"


def load_config(path: Path | None = None) -> ExperimentConfig:
    path = path or default_config_path()
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return ExperimentConfig.from_dict(json.load(f))
    return ExperimentConfig()


def save_config(cfg: ExperimentConfig, path: Path | None = None) -> Path:
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def capture_path_for_index(capture_root: Path, session_id: str, index: int, shard_size: int) -> Path:
    """captures/<session>/shard_000/000000.jpg"""
    shard = index // max(shard_size, 1)
    folder = capture_root / session_id / f"shard_{shard:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{index:06d}.jpg"


def estimate_hours(n: int, cd: CooldownConfig) -> float:
    """粗算总时长（小时），含批次长休息。"""
    per = (
        cd.after_show_s
        + cd.after_af_s
        + cd.after_ready_before_shoot_s
        + cd.after_shoot_s
        + cd.after_ack_before_next_s
    )
    rest = (n // max(cd.batch_rest_every, 1)) * cd.batch_rest_s
    return (n * per + rest) / 3600.0
