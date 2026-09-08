#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_weights_ui.py
====================
嵌入质量工具（菜单 6/7/8）的权重选择弹窗。

用户可选 **1 个**（单模型测试）或 **2 个**（对照）``.pth`` 文件。
无显示器 / 取消时返回 ``None``。
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Sequence


def _has_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _default_models_dir(repo_root: Path) -> Path:
    for cand in (repo_root / "models" / "runs", repo_root / "models"):
        if cand.is_dir():
            return cand
    return repo_root / "models"


def prompt_embed_verify_weights(
    *,
    title: str = "选择嵌入质量测试权重",
    subtitle: str = "请选择 1 个（单测）或 2 个（对照）权重文件",
    repo_root: str | Path | None = None,
    initial_paths: Sequence[str | Path] | None = None,
) -> Optional[List[Path]]:
    """
    弹出权重选择窗。

    Returns
    -------
    list[Path] | None
        长度 1 或 2 的已存在路径；取消 / 无显示器返回 ``None``。
    """
    if not _has_display():
        return None

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent
    start_dir = _default_models_dir(root)
    presets = list(initial_paths or [])

    result: dict = {"ok": False, "paths": []}

    win = tk.Tk()
    win.title(title)
    win.geometry("740x300")
    win.minsize(580, 260)
    try:
        win.attributes("-topmost", True)
        win.after(200, lambda: win.attributes("-topmost", False))
    except tk.TclError:
        pass

    frm = ttk.Frame(win, padding=14)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(0, weight=1)

    ttk.Label(frm, text=title, font=("Segoe UI", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    ttk.Label(frm, text=subtitle, foreground="#444").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(2, 10)
    )

    path_vars = [tk.StringVar(value=""), tk.StringVar(value="")]
    for i, p in enumerate(presets[:2]):
        path_vars[i].set(str(Path(p)))

    def browse(slot: int) -> None:
        cur = path_vars[slot].get().strip()
        init_dir = Path(cur).parent if cur and Path(cur).parent.is_dir() else start_dir
        chosen = filedialog.askopenfilename(
            parent=win,
            title=f"选择权重 {slot + 1}",
            initialdir=str(init_dir),
            filetypes=[("PyTorch 权重", "*.pth"), ("所有文件", "*.*")],
        )
        if chosen:
            path_vars[slot].set(chosen)

    def clear(slot: int) -> None:
        path_vars[slot].set("")

    row = 2
    for i, lab in enumerate(("权重 1（必选）", "权重 2（可选，对照）")):
        ttk.Label(frm, text=lab).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        line = ttk.Frame(frm)
        line.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 8))
        line.columnconfigure(0, weight=1)
        ttk.Entry(line, textvariable=path_vars[i]).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(line, text="浏览…", width=8, command=lambda s=i: browse(s)).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(line, text="清除", width=6, command=lambda s=i: clear(s)).grid(
            row=0, column=2, padx=2
        )
        row += 1

    ttk.Label(
        frm,
        text="只填权重 1 → 单模型测试；两个都填 → 对照。可从 models/ 或 models/runs/<run_id>/ 选择。",
        foreground="#666",
        wraplength=700,
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 10))
    row += 1

    def on_ok() -> None:
        paths: List[Path] = []
        for i, var in enumerate(path_vars):
            s = var.get().strip()
            if not s:
                if i == 0:
                    messagebox.showerror("缺少权重", "请至少选择权重 1（.pth）", parent=win)
                    return
                continue
            p = Path(s)
            if not p.is_file():
                messagebox.showerror("文件不存在", f"找不到:\n{p}", parent=win)
                return
            if p.suffix.lower() != ".pth":
                if not messagebox.askyesno(
                    "扩展名提醒",
                    f"文件不是 .pth：\n{p}\n仍要使用吗？",
                    parent=win,
                ):
                    return
            paths.append(p.resolve())
        if len(paths) == 2 and paths[0] == paths[1]:
            messagebox.showerror(
                "重复权重", "两个权重路径相同，请换一个或只选一个。", parent=win
            )
            return
        result["ok"] = True
        result["paths"] = paths
        win.destroy()

    def on_cancel() -> None:
        result["ok"] = False
        win.destroy()

    btns = ttk.Frame(frm)
    btns.grid(row=row, column=0, columnspan=2, sticky="e")
    ttk.Button(btns, text="取消", command=on_cancel).pack(side="right", padx=4)
    ttk.Button(btns, text="开始测试", command=on_ok).pack(side="right", padx=4)

    win.protocol("WM_DELETE_WINDOW", on_cancel)
    win.bind("<Return>", lambda _e: on_ok())
    win.bind("<Escape>", lambda _e: on_cancel())
    try:
        win.lift()
        win.focus_force()
    except tk.TclError:
        pass
    win.mainloop()

    if not result["ok"]:
        return None
    return list(result["paths"])


def prompt_single_weight_file(
    *,
    title: str = "选择评估权重",
    subtitle: str = "请选择一个 .pth 权重文件",
    repo_root: str | Path | None = None,
    initial_path: str | Path | None = None,
) -> Optional[Path]:
    """
    弹出单文件权重选择窗（评估 / 嵌入 / 测准）。

    Returns
    -------
    Path | None
        已存在的 ``.pth``；取消 / 无显示器返回 ``None``。
    """
    if not _has_display():
        return None

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent
    start_dir = _default_models_dir(root)
    result: dict = {"ok": False, "path": None}

    win = tk.Tk()
    win.title(title)
    win.geometry("720x220")
    win.minsize(560, 200)
    try:
        win.attributes("-topmost", True)
        win.after(200, lambda: win.attributes("-topmost", False))
    except tk.TclError:
        pass

    frm = ttk.Frame(win, padding=14)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(0, weight=1)

    ttk.Label(frm, text=title, font=("Segoe UI", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    ttk.Label(frm, text=subtitle, foreground="#444").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(2, 12)
    )

    path_var = tk.StringVar(value=str(initial_path) if initial_path else "")

    def browse() -> None:
        cur = path_var.get().strip()
        init_dir = Path(cur).parent if cur and Path(cur).parent.is_dir() else start_dir
        chosen = filedialog.askopenfilename(
            parent=win,
            title="选择权重文件",
            initialdir=str(init_dir),
            filetypes=[("PyTorch 权重", "*.pth"), ("所有文件", "*.*")],
        )
        if chosen:
            path_var.set(chosen)

    line = ttk.Frame(frm)
    line.grid(row=2, column=0, columnspan=2, sticky="ew")
    line.columnconfigure(0, weight=1)
    ttk.Entry(line, textvariable=path_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
    ttk.Button(line, text="浏览…", width=10, command=browse).grid(row=0, column=1)

    ttk.Label(
        frm,
        text="可从 models/、models/runs/<run_id>/ 或任意目录选择。embed_strength 优先读旁路 .meta.txt。",
        foreground="#666",
        wraplength=680,
    ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 10))

    def on_ok() -> None:
        s = path_var.get().strip()
        if not s:
            messagebox.showerror("缺少权重", "请选择一个 .pth 文件", parent=win)
            return
        p = Path(s)
        if not p.is_file():
            messagebox.showerror("文件不存在", f"找不到:\n{p}", parent=win)
            return
        if p.suffix.lower() != ".pth":
            if not messagebox.askyesno(
                "扩展名提醒",
                f"文件不是 .pth：\n{p}\n仍要使用吗？",
                parent=win,
            ):
                return
        result["ok"] = True
        result["path"] = p.resolve()
        win.destroy()

    def on_cancel() -> None:
        result["ok"] = False
        win.destroy()

    # 打开即弹出系统文件对话框，减少一次点击
    win.after(80, browse)

    btns = ttk.Frame(frm)
    btns.grid(row=4, column=0, columnspan=2, sticky="e")
    ttk.Button(btns, text="取消", command=on_cancel).pack(side="right", padx=4)
    ttk.Button(btns, text="确定", command=on_ok).pack(side="right", padx=4)

    win.protocol("WM_DELETE_WINDOW", on_cancel)
    win.bind("<Return>", lambda _e: on_ok())
    win.bind("<Escape>", lambda _e: on_cancel())
    try:
        win.lift()
        win.focus_force()
    except tk.TclError:
        pass
    win.mainloop()

    if not result["ok"]:
        return None
    return result["path"]


def infer_noise_menu_choice_from_path(path: str | Path) -> str:
    """根据权重路径猜测噪声层菜单默认：1=ScreenShooting，2=ScreenShootingMB。"""
    name = str(path).replace("\\", "/").lower()
    stem = Path(path).stem.lower()
    if "screenshootingmb" in name or stem.endswith("_mb_best") or "mb_best" in stem:
        return "2"
    if "screenshooting_best" in stem or "/screenshooting/" in name:
        return "1"
    return "2"


def ckpt_args_from_paths(paths: Sequence[Path]) -> List[str]:
    """把路径列表转成传给 verify 脚本的 ``--ckpt`` 参数。"""
    out: List[str] = []
    for p in paths:
        out.extend(["--ckpt", str(p)])
    return out
