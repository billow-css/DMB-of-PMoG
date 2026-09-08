#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_log.py
============
DMB of PMoG 训练日志：每次训练写入独立 JSON，**永不覆盖**历史 run。

布局
----
  logs/runs/<run_id>.json     完整结构化记录（主存）
  logs/runs/<run_id>.txt      人类可读镜像（与旧 txt 风格兼容）
  logs/runs/index.json        索引（可重建；更新用原子替换）
  logs/LATEST_TRAIN.json      仅指针（可覆盖），指向最近一次 run

公开接口
--------
- TrainRunLogger             训练过程写入
- list_runs / load_run       查询
- rebuild_index              从 runs/*.json 重建索引
- format_run_brief / detail  CLI 展示
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT = "DMB of PMoG"
INDEX_NAME = "index.json"
RUNS_SUBDIR = "runs"
LATEST_NAME = "LATEST_TRAIN.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{stamp}_{short}"


def runs_dir(log_dir: str | Path) -> Path:
    d = Path(log_dir) / RUNS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_path(log_dir: str | Path) -> Path:
    return runs_dir(log_dir) / INDEX_NAME


def latest_pointer_path(log_dir: str | Path) -> Path:
    return Path(log_dir) / LATEST_NAME


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _safe_load_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _index_entry_from_doc(doc: Dict[str, Any], rel_path: str) -> Dict[str, Any]:
    cfg = doc.get("config") or {}
    summary = doc.get("summary") or {}
    epochs = doc.get("epochs") or []
    return {
        "run_id": doc.get("run_id"),
        "path": rel_path,
        "started_at": doc.get("started_at"),
        "finished_at": doc.get("finished_at"),
        "status": doc.get("status", "unknown"),
        "distortion": cfg.get("distortion"),
        "lite": bool(cfg.get("lite", False)),
        "num_epoch": cfg.get("num_epoch"),
        "epochs_done": len(epochs),
        "best_val_acc": summary.get("best_val_acc"),
        "best_epoch": summary.get("best_epoch"),
        "embed_strength": cfg.get("embed_strength"),
        "init_from_epoch": cfg.get("init_from_epoch"),
        "init_from_ckpt": cfg.get("init_from_ckpt"),
        "project": doc.get("project", PROJECT),
    }


def rebuild_index(log_dir: str | Path) -> Dict[str, Any]:
    """扫描 ``runs/*.json``（排除 index）重建索引；不删任何 run 文件。"""
    rd = runs_dir(log_dir)
    entries: List[Dict[str, Any]] = []
    for p in sorted(rd.glob("*.json")):
        if p.name == INDEX_NAME or ".tmp" in p.name:
            continue
        doc = _safe_load_json(p)
        if not isinstance(doc, dict) or "run_id" not in doc:
            continue
        entries.append(_index_entry_from_doc(doc, f"{RUNS_SUBDIR}/{p.name}"))
    entries.sort(key=lambda e: e.get("started_at") or "", reverse=True)
    idx = {"version": 1, "updated_at": _utc_now_iso(), "runs": entries}
    _atomic_write_json(index_path(log_dir), idx)
    return idx


def load_index(log_dir: str | Path, *, rebuild_if_missing: bool = True) -> Dict[str, Any]:
    path = index_path(log_dir)
    data = _safe_load_json(path)
    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        return data
    if rebuild_if_missing:
        return rebuild_index(log_dir)
    return {"version": 1, "updated_at": None, "runs": []}


def _update_index_entry(log_dir: str | Path, entry: Dict[str, Any]) -> None:
    idx = load_index(log_dir, rebuild_if_missing=True)
    runs = [e for e in idx.get("runs", []) if e.get("run_id") != entry.get("run_id")]
    runs.insert(0, entry)
    idx["runs"] = runs
    idx["updated_at"] = _utc_now_iso()
    _atomic_write_json(index_path(log_dir), idx)


def list_runs(
    log_dir: str | Path,
    *,
    distortion: Optional[str] = None,
    lite: Optional[bool] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    idx = load_index(log_dir)
    rows = list(idx.get("runs") or [])
    if distortion:
        rows = [r for r in rows if r.get("distortion") == distortion]
    if lite is not None:
        rows = [r for r in rows if bool(r.get("lite")) == lite]
    if status:
        rows = [r for r in rows if r.get("status") == status]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def resolve_run_path(log_dir: str | Path, run_id_or_path: str) -> Path:
    p = Path(run_id_or_path)
    if p.is_file():
        return p
    cand = runs_dir(log_dir) / f"{run_id_or_path}.json"
    if cand.is_file():
        return cand
    matches = sorted(runs_dir(log_dir).glob(f"{run_id_or_path}*.json"))
    matches = [m for m in matches if m.name != INDEX_NAME and ".tmp" not in m.name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"前缀 {run_id_or_path!r} 匹配到多个 run: "
            + ", ".join(m.stem for m in matches[:8])
        )
    raise FileNotFoundError(f"找不到训练日志: {run_id_or_path}")


def load_run(log_dir: str | Path, run_id_or_path: str) -> Dict[str, Any]:
    path = resolve_run_path(log_dir, run_id_or_path)
    doc = _safe_load_json(path)
    if not isinstance(doc, dict):
        raise FileNotFoundError(f"无法读取 JSON: {path}")
    return doc


def format_run_brief(entry: Dict[str, Any], index: Optional[int] = None) -> str:
    prefix = f"[{index}] " if index is not None else ""
    acc = entry.get("best_val_acc")
    acc_s = f"{float(acc):.2f}%" if isinstance(acc, (int, float)) else "—"
    return (
        f"{prefix}{entry.get('run_id', '?')}  "
        f"status={entry.get('status', '?')}  "
        f"dist={entry.get('distortion', '?')}  "
        f"{'LITE' if entry.get('lite') else 'FULL'}  "
        f"epochs={entry.get('epochs_done', 0)}/{entry.get('num_epoch', '?')}  "
        f"bestVal={acc_s}  "
        f"start={entry.get('started_at', '?')}"
    )


def format_run_detail(doc: Dict[str, Any], *, last_epochs: int = 20) -> str:
    cfg = doc.get("config") or {}
    summary = doc.get("summary") or {}
    epochs = doc.get("epochs") or []
    lines = [
        f"run_id       : {doc.get('run_id')}",
        f"project      : {doc.get('project')}",
        f"status       : {doc.get('status')}",
        f"started_at   : {doc.get('started_at')}",
        f"finished_at  : {doc.get('finished_at')}",
        f"distortion   : {cfg.get('distortion')}",
        f"lite         : {cfg.get('lite')}",
        f"num_epoch    : {cfg.get('num_epoch')}",
        f"batch_size   : {cfg.get('batch_size')}",
        f"lr           : {cfg.get('lr')}",
        f"λ            : ({cfg.get('lambda1')}, {cfg.get('lambda2')}, {cfg.get('lambda3')})",
        f"warmup       : {cfg.get('warmup_epochs')}",
        f"embed_str    : {cfg.get('embed_strength')}",
        f"init_from    : {cfg.get('init_from_ckpt') or cfg.get('init_from_epoch')}",
        f"ckpt_dir     : {doc.get('ckpt_dir')}",
        f"json_path    : {doc.get('json_path')}",
        f"txt_path     : {doc.get('txt_path')}",
        "",
        f"best_val_acc : {summary.get('best_val_acc')}",
        f"best_epoch   : {summary.get('best_epoch')}",
        f"best_ckpt    : {summary.get('best_ckpt')}",
        f"total_sec    : {summary.get('total_seconds')}",
        "",
        f"epochs ({len(epochs)} recorded):",
    ]
    show = epochs[-max(1, last_epochs) :] if epochs else []
    if epochs and len(epochs) > len(show):
        lines.append(f"  … 省略前 {len(epochs) - len(show)} 个 epoch …")
    for ep in show:
        lines.append(
            f"  ep{ep.get('epoch')}: "
            f"train={float(ep.get('train_acc', 0)):.2f}%  "
            f"val={float(ep.get('val_acc', 0)):.2f}%  "
            f"BER={float(ep.get('val_ber', 0)):.4f}  "
            f"PSNR={float(ep.get('psnr', 0)):.2f}dB  "
            f"phase={ep.get('phase')}  "
            f"{'★best' if ep.get('is_best') else ''}"
        )
    return "\n".join(lines)


class TrainRunLogger:
    """单次训练会话：JSON 主存 + txt 镜像；路径唯一，不覆盖历史。"""

    def __init__(self, log_dir: str | Path, config: Any):
        self.log_dir = Path(log_dir)
        self.runs = runs_dir(self.log_dir)
        self.run_id = _run_id()
        while (self.runs / f"{self.run_id}.json").exists():
            self.run_id = _run_id()
            time.sleep(0.01)

        self.json_path = self.runs / f"{self.run_id}.json"
        self.txt_path = self.runs / f"{self.run_id}.txt"

        cfg_dict = self._snapshot_config(config)
        self.doc: Dict[str, Any] = {
            "run_id": self.run_id,
            "project": PROJECT,
            "status": "running",
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "config": cfg_dict,
            "dataset": {},
            "epochs": [],
            "events": [],
            "summary": {},
            "json_path": str(self.json_path.as_posix()),
            "txt_path": str(self.txt_path.as_posix()),
        }
        self._txt = open(self.txt_path, "w", encoding="utf-8")
        self._write_txt_header()
        self._flush_json()
        self._touch_index_and_latest()

    @staticmethod
    def _snapshot_config(config: Any) -> Dict[str, Any]:
        keys = [
            "mode",
            "dataset",
            "distortion",
            "lite",
            "lite_ratio",
            "lite_seed",
            "num_epoch",
            "batch_size",
            "image_size",
            "lr",
            "lambda1",
            "lambda2",
            "lambda3",
            "warmup_epochs",
            "embed_strength",
            "embedding_epoch",
            "init_from_epoch",
            "init_from_ckpt",
            "eval_ckpt",
            "model_name",
            "model_save_dir",
            "image_dir",
            "image_val_dir",
            "wmat_dir",
            "save_viz",
            "model_save_step",
        ]
        out: Dict[str, Any] = {}
        for k in keys:
            if hasattr(config, k):
                v = getattr(config, k)
                if isinstance(v, (str, int, float, bool)) or v is None:
                    out[k] = v
                else:
                    out[k] = str(v)
        return out

    def _write_txt_header(self) -> None:
        cfg = self.doc["config"]
        tag = "LITE" if cfg.get("lite") else "FULL"
        header = (
            f"# {PROJECT} train_mask  run_id={self.run_id}\n"
            f"# mode={tag}  distortion={cfg.get('distortion')}\n"
            f"# epochs={cfg.get('num_epoch')}  batch={cfg.get('batch_size')}  "
            f"lr={cfg.get('lr')}  warmup={cfg.get('warmup_epochs')}  "
            f"λ=({cfg.get('lambda1')},{cfg.get('lambda2')},{cfg.get('lambda3')})\n"
            f"# started_at={self.doc['started_at']}\n"
            f"# json={self.json_path}\n"
        )
        self._txt.write(header)
        self._txt.flush()

    def _flush_json(self) -> None:
        _atomic_write_json(self.json_path, self.doc)

    def _touch_index_and_latest(self) -> None:
        entry = _index_entry_from_doc(
            self.doc, f"{RUNS_SUBDIR}/{self.json_path.name}"
        )
        _update_index_entry(self.log_dir, entry)
        _atomic_write_json(
            latest_pointer_path(self.log_dir),
            {
                "run_id": self.run_id,
                "json_path": str(self.json_path.as_posix()),
                "txt_path": str(self.txt_path.as_posix()),
                "updated_at": _utc_now_iso(),
            },
        )

    @property
    def txtfile(self):
        """兼容 ProgressMonitor(file=…) 与 print(..., file=)。"""
        return self._txt

    def set_dataset_info(
        self,
        *,
        train_samples: int,
        val_samples: int,
        batches_per_epoch: int,
    ) -> None:
        self.doc["dataset"] = {
            "train_samples": int(train_samples),
            "val_samples": int(val_samples),
            "batches_per_epoch": int(batches_per_epoch),
        }
        self._flush_json()
        self._touch_index_and_latest()

    def add_event(self, level: str, message: str) -> None:
        self.doc.setdefault("events", []).append(
            {"t": _utc_now_iso(), "level": level, "message": message}
        )
        self._flush_json()

    def log_epoch(self, record: Dict[str, Any]) -> None:
        """追加一个 epoch 记录并落盘。"""
        self.doc["epochs"].append(record)
        line = (
            f"Epoch {record.get('epoch')}/{record.get('num_epoch')}  "
            f"TrainAcc={float(record.get('train_acc', 0)):.2f}%  "
            f"ValAcc={float(record.get('val_acc', 0)):.2f}%  "
            f"BER={float(record.get('val_ber', 0)):.4f} "
            f"(train {float(record.get('train_ber', 0)):.4f})  "
            f"PSNR={float(record.get('psnr', 0)):.2f}dB  "
            f"noise={record.get('noise')}  "
            f"time={record.get('time')}  ETA≈{record.get('eta')}"
        )
        if record.get("is_best"):
            line += "  ★best"
        print(line, file=self._txt, flush=True)
        self._flush_json()
        self._touch_index_and_latest()

    def finalize(
        self,
        *,
        status: str = "completed",
        best_val_acc: float,
        best_epoch: Optional[int],
        best_ckpt: Optional[str],
        total_seconds: float,
    ) -> None:
        self.doc["status"] = status
        self.doc["finished_at"] = _utc_now_iso()
        acc = float(best_val_acc)
        if acc <= 1.0:
            acc *= 100.0
        self.doc["summary"] = {
            "best_val_acc": acc,
            "best_epoch": best_epoch,
            "best_ckpt": best_ckpt,
            "total_seconds": float(total_seconds),
        }
        footer = (
            f"# finished status={status}  "
            f"best_val_acc={acc:.3f}%  "
            f"best_epoch={best_epoch}  total_s={total_seconds:.1f}\n"
        )
        self._txt.write(footer)
        self._txt.flush()
        self._flush_json()
        self._touch_index_and_latest()

    def close(self) -> None:
        try:
            self._txt.close()
        except Exception:
            pass
