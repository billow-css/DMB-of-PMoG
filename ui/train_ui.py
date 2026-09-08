#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ui.py
===========
训练交互 UI：

1. ``edit_train_hparams``  — 训练开始前弹出参数编辑窗（tkinter）
2. ``show_epoch_curves``   — 每个 epoch 结束后弹出 msg / den / loss 曲线

无显示器 / ``--no_interactive`` / 显式关闭开关时自动降级跳过。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _has_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# (attr, label, type)
_HPARAM_FIELDS = (
    ("num_epoch", "总 epoch 数", int),
    ("batch_size", "batch_size", int),
    ("lr", "学习率 lr", float),
    ("lambda1", "λ1 消息损失权重", float),
    ("lambda2", "λ2 图像/感知损失权重", float),
    ("lambda3", "λ3 GAN 损失权重", float),
    ("warmup_epochs", "Identity warmup epochs", int),
    ("embed_strength", "embed_strength (0=整图)", float),
    ("image_size", "image_size", int),
    ("model_save_step", "每隔多少 epoch 存盘", int),
    ("log_step", "无进度条时的日志间隔", int),
)


def edit_train_hparams(config: Any, *, title: str = "DMB of PMoG · 训练参数") -> Any:
    """
    弹出可编辑训练参数窗口；点「确定开始训练」写回 ``config`` 并返回。
    取消则 ``SystemExit``；无 GUI 时原样返回。
    """
    if getattr(config, "no_interactive", False):
        return config
    if getattr(config, "no_train_editor", False):
        return config
    if not _has_display():
        return config

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception:
        return config

    result: Dict[str, Any] = {"ok": False, "values": {}}

    root = tk.Tk()
    root.title(title)
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=14)
    frm.grid(row=0, column=0, sticky="nsew")

    ttk.Label(
        frm,
        text="确认 / 修改训练超参后点击「确定开始训练」",
        font=("", 10, "bold"),
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

    summary = (
        f"噪声层: {getattr(config, 'distortion', '?')}    "
        f"规模: {'LITE' if getattr(config, 'lite', False) else 'FULL'}    "
        f"初始化: "
        + (
            (
                f"挂载 {getattr(config, 'init_from_ckpt', None)}"
                if getattr(config, "init_from_ckpt", None)
                else f"挂载 mask_{getattr(config, 'init_from_epoch', None)}"
            )
            if (
                getattr(config, "init_from_ckpt", None)
                or getattr(config, "init_from_epoch", None)
            )
            else "从头"
        )
    )
    ttk.Label(frm, text=summary, foreground="#444").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(0, 12)
    )

    entries: Dict[str, Any] = {}
    for i, (attr, label, _typ) in enumerate(_HPARAM_FIELDS):
        ttk.Label(frm, text=label).grid(
            row=i + 2, column=0, sticky="w", pady=3, padx=(0, 10)
        )
        ent = ttk.Entry(frm, width=18)
        cur = getattr(config, attr, "")
        ent.insert(0, "" if cur is None else str(cur))
        ent.grid(row=i + 2, column=1, sticky="ew", pady=3)
        entries[attr] = ent

    save_viz_var = tk.BooleanVar(value=bool(getattr(config, "save_viz", True)))
    epoch_plot_var = tk.BooleanVar(
        value=not bool(getattr(config, "no_epoch_plot", False))
    )
    row0 = 2 + len(_HPARAM_FIELDS)
    ttk.Checkbutton(
        frm, text="保存 epoch 可视化拼图 (save_viz)", variable=save_viz_var
    ).grid(row=row0, column=0, columnspan=2, sticky="w", pady=(8, 2))
    ttk.Checkbutton(
        frm,
        text="每 epoch 结束后弹出 msg/den/loss 曲线",
        variable=epoch_plot_var,
    ).grid(row=row0 + 1, column=0, columnspan=2, sticky="w", pady=2)

    def _on_ok() -> None:
        parsed: Dict[str, Any] = {}
        try:
            for attr, label, typ in _HPARAM_FIELDS:
                raw = entries[attr].get().strip()
                if raw == "":
                    raise ValueError(f"{label} 不能为空")
                parsed[attr] = typ(raw)
            if parsed["num_epoch"] < 1:
                raise ValueError("总 epoch 数须 >= 1")
            if parsed["batch_size"] < 1:
                raise ValueError("batch_size 须 >= 1")
            if parsed["lr"] <= 0:
                raise ValueError("学习率须 > 0")
            if parsed["warmup_epochs"] < 0:
                raise ValueError("warmup_epochs 须 >= 0")
            if parsed["warmup_epochs"] >= parsed["num_epoch"]:
                if not messagebox.askyesno(
                    "确认",
                    f"warmup_epochs={parsed['warmup_epochs']} >= num_epoch="
                    f"{parsed['num_epoch']}，几乎全程 warmup。仍要继续？",
                ):
                    return
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        parsed["save_viz"] = bool(save_viz_var.get())
        parsed["no_epoch_plot"] = not bool(epoch_plot_var.get())
        result["ok"] = True
        result["values"] = parsed
        root.destroy()

    def _on_cancel() -> None:
        result["ok"] = False
        root.destroy()

    btns = ttk.Frame(frm)
    btns.grid(row=row0 + 2, column=0, columnspan=2, pady=(14, 0), sticky="e")
    ttk.Button(btns, text="取消", command=_on_cancel).pack(side="right", padx=4)
    ttk.Button(btns, text="确定开始训练", command=_on_ok).pack(side="right", padx=4)

    root.protocol("WM_DELETE_WINDOW", _on_cancel)
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = max(0, (root.winfo_screenwidth() - w) // 2)
    y = max(0, (root.winfo_screenheight() - h) // 2)
    root.geometry(f"+{x}+{y}")
    try:
        root.lift()
        root.focus_force()
    except Exception:
        pass
    root.mainloop()

    if not result["ok"]:
        raise SystemExit("已取消训练参数确认。")

    for k, v in result["values"].items():
        setattr(config, k, v)
    return config


def print_train_hparams(config: Any) -> None:
    """控制台打印最终训练超参明细。"""
    from cli_utils import _C, _paint, log_info, log_ok

    log_ok("训练参数已确认，即将开始训练")
    rows = [
        ("mode", getattr(config, "mode", None)),
        ("distortion", getattr(config, "distortion", None)),
        ("lite", getattr(config, "lite", None)),
        ("init_from_epoch", getattr(config, "init_from_epoch", None)),
        ("init_from_ckpt", getattr(config, "init_from_ckpt", None)),
        ("num_epoch", getattr(config, "num_epoch", None)),
        ("batch_size", getattr(config, "batch_size", None)),
        ("lr", getattr(config, "lr", None)),
        ("lambda1", getattr(config, "lambda1", None)),
        ("lambda2", getattr(config, "lambda2", None)),
        ("lambda3", getattr(config, "lambda3", None)),
        ("warmup_epochs", getattr(config, "warmup_epochs", None)),
        ("embed_strength", getattr(config, "embed_strength", None)),
        ("image_size", getattr(config, "image_size", None)),
        ("model_save_step", getattr(config, "model_save_step", None)),
        ("save_viz", getattr(config, "save_viz", None)),
        ("no_epoch_plot", getattr(config, "no_epoch_plot", False)),
        ("image_dir", getattr(config, "image_dir", None)),
        ("image_val_dir", getattr(config, "image_val_dir", None)),
    ]
    print(_paint("  最终训练配置", _C.BOLD, _C.WHITE), flush=True)
    for k, v in rows:
        print(_paint(f"    {k:<18}", _C.DIM) + _paint(str(v), _C.CYAN), flush=True)
    print(_paint("─" * 64, _C.DIM), flush=True)
    log_info("进入 DataLoader / Solver …")


def show_epoch_curves(
    *,
    epoch: int,
    num_epoch: int,
    phase: str,
    loss_hist: Sequence[float],
    msg_hist: Sequence[float],
    den_hist: Sequence[float],
    save_dir: Optional[str | Path] = None,
    block: bool = True,
    enabled: bool = True,
) -> Optional[Path]:
    """
    弹出本 epoch 内 batch 级 msg / den / loss 曲线。
    关闭窗口后训练继续。返回保存的图片路径（若启用 save_dir）。
    """
    if not enabled or not loss_hist:
        return None
    if not _has_display():
        return None

    try:
        import matplotlib

        try:
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")
            block = False
        import matplotlib.pyplot as plt
    except Exception:
        return None

    n = len(loss_hist)
    xs = list(range(1, n + 1))
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        f"Epoch {epoch}/{num_epoch}  [{phase}]  ·  batch curves",
        fontsize=12,
    )

    series = (
        (axes[0], msg_hist, "msg (message MSE)", "#1f77b4"),
        (axes[1], den_hist, "den (image / denoise loss)", "#2ca02c"),
        (axes[2], loss_hist, "loss (total)", "#d62728"),
    )
    for ax, ys, label, color in series:
        ax.plot(xs, list(ys), color=color, linewidth=1.2, label=label)
        ax.set_ylabel(label.split()[0])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("batch")
    fig.tight_layout()

    out_path: Optional[Path] = None
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"epoch_{epoch:04d}_curves.png"
        try:
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
        except Exception:
            out_path = None

    backend = str(matplotlib.get_backend()).lower()
    if "agg" in backend and "tk" not in backend:
        plt.close(fig)
        return out_path

    try:
        mgr = getattr(fig.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "window"):
            try:
                mgr.window.attributes("-topmost", True)
                mgr.window.attributes("-topmost", False)
            except Exception:
                pass
        plt.show(block=block)
    except Exception:
        pass
    finally:
        plt.close(fig)
    return out_path
