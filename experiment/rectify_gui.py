#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/rectify_gui.py
=========================
拍屏透视矫正 GUI：可拖角框选 + 右侧输出预览 + 步进/循环。

模式
----
  自动框选  — 使用现有 ArUco / 白垫检测（markers_fast）
  伺服框选  — 沿用上一张透视四边形，仅做轻度位移（相位相关）
  手动框选  — 不自动改框，只靠拖动四角

用法
----
  python rectify_gui.py
  python rectify_gui.py --src captures/session_xxx --out ../Datasets/Recover/capture
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_rectify import list_images, parse_index  # noqa: E402
from markers_fast import detect_content_quad_fast, warp_content  # noqa: E402

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"需要 tkinter: {e}") from e

try:
    from PIL import Image, ImageTk
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"需要 Pillow: {e}") from e


def default_quad(h: int, w: int, margin: float = 0.18) -> np.ndarray:
    """图像内默认矩形四角 TL,TR,BR,BL。"""
    x0, x1 = w * margin, w * (1.0 - margin)
    y0, y1 = h * margin, h * (1.0 - margin)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def estimate_shift_phase(
    prev_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    max_shift_frac: float = 1.0 / 15.0,
) -> Tuple[float, float]:
    """
    相位相关估计整图平移 (dx, dy)，用于伺服框轻度位移。
    上一帧点 + (dx,dy) ≈ 当前点。
    位移模长与分量均不超过「当前图最大边长 × max_shift_frac」（默认 1/15）。
    """
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    max_side = 640
    h, w = g0.shape[:2]
    sc = min(1.0, max_side / float(max(h, w)))
    if sc < 1.0:
        g0 = cv2.resize(g0, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        g1 = cv2.resize(g1, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    g0 = cv2.GaussianBlur(g0, (5, 5), 0)
    g1 = cv2.GaussianBlur(g1, (5, 5), 0)
    shift, _resp = cv2.phaseCorrelate(np.float32(g0), np.float32(g1))
    dx, dy = float(shift[0]) / sc, float(shift[1]) / sc
    lim = float(max(h, w)) * float(max_shift_frac)
    # 分量钳制
    dx = float(np.clip(dx, -lim, lim))
    dy = float(np.clip(dy, -lim, lim))
    # 模长不超过 lim（候选框整体位移）
    mag = float(np.hypot(dx, dy))
    if mag > lim and mag > 1e-6:
        s = lim / mag
        dx *= s
        dy *= s
    return dx, dy


def clamp_quad_shift(
    prev_quad: np.ndarray,
    new_quad: np.ndarray,
    max_shift: float,
) -> np.ndarray:
    """每个角相对上一框的位移不超过 max_shift。"""
    prev = prev_quad.astype(np.float32).reshape(4, 2)
    cur = new_quad.astype(np.float32).reshape(4, 2)
    out = cur.copy()
    for i in range(4):
        d = out[i] - prev[i]
        mag = float(np.hypot(d[0], d[1]))
        if mag > max_shift and mag > 1e-6:
            out[i] = prev[i] + d * (max_shift / mag)
    return out


def refine_quad_local(
    bgr: np.ndarray, quad: np.ndarray, search: int = 18
) -> np.ndarray:
    """在四角邻域用角点响应做轻微精修（伺服辅助）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]
    out = quad.astype(np.float32).copy()
    for i, (x, y) in enumerate(quad):
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - search), min(w, cx + search + 1)
        y0, y1 = max(0, cy - search), min(h, cy + search + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size < 16:
            continue
        resp = cv2.cornerMinEigenVal(patch, 3)
        yy, xx = np.unravel_index(int(np.argmax(resp)), resp.shape)
        out[i, 0] = x0 + xx
        out[i, 1] = y0 + yy
    return out


def detect_auto_quad(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
    """自动框选：复用快速检测。"""
    q, info = detect_content_quad_fast(
        bgr,
        max_side=1600,
        min_confidence=0.40,
        allow_fallback=True,
        allow_white_frame=False,
        min_fallback_confidence=0.58,
        min_aruco_markers=3,
    )
    return q, info


class RectifyGUI:
    MODE_AUTO = "auto"
    MODE_SERVO = "servo"
    MODE_MANUAL = "manual"

    def __init__(
        self,
        src: Optional[Path] = None,
        out: Optional[Path] = None,
        out_size: int = 128,
    ):
        self.root = tk.Tk()
        self.root.title("PIMoG 拍屏矫正 · 框选预览")
        self.root.geometry("1280x780")
        self.root.minsize(960, 600)
        self.root.configure(bg="#1e1e1e")

        self.out_size = int(out_size)
        self.src_dir: Optional[Path] = Path(src) if src else None
        self.out_dir: Optional[Path] = Path(out) if out else None
        self.files: List[Path] = []
        self.index = 0

        self.bgr: Optional[np.ndarray] = None
        self.prev_bgr: Optional[np.ndarray] = None
        self.quad: Optional[np.ndarray] = None
        self.prev_quad: Optional[np.ndarray] = None
        self.warped: Optional[np.ndarray] = None

        self.mode = tk.StringVar(value=self.MODE_AUTO)
        self.status = tk.StringVar(value="请选择拍屏文件夹")
        self.info_var = tk.StringVar(value="")

        self._photo_main = None
        self._photo_prev = None
        self._scale = 1.0
        self._ox = 0
        self._oy = 0
        self._drag_i: Optional[int] = None
        self._looping = False
        self._paused = False
        self._loop_job = None
        self._busy = False
        self._last_detect_ms = 0.0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.out_dir is None:
            self.out_dir = (
                Path(__file__).resolve().parent.parent
                / "Datasets"
                / "Recover"
                / "capture"
            )
        if self.src_dir and self.src_dir.exists():
            self._load_folder(self.src_dir)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#2a2a2a", height=40)
        top.pack(side=tk.TOP, fill=tk.X)
        tk.Button(top, text="打开拍屏文件夹", command=self._pick_src).pack(
            side=tk.LEFT, padx=6, pady=6
        )
        tk.Button(top, text="输出目录", command=self._pick_out).pack(
            side=tk.LEFT, padx=4, pady=6
        )
        tk.Label(top, textvariable=self.status, bg="#2a2a2a", fg="#ddd").pack(
            side=tk.LEFT, padx=12
        )

        body = tk.Frame(self.root, bg="#1e1e1e")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        left = tk.Frame(body, bg="#1e1e1e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            left, text="操作窗口（拖动角点）", bg="#1e1e1e", fg="#aaa"
        ).pack(anchor="w")
        self.canvas = tk.Canvas(
            left, bg="#111", highlightthickness=1, highlightbackground="#444"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        right = tk.Frame(body, bg="#1e1e1e", width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)
        tk.Label(right, text="输出预览", bg="#1e1e1e", fg="#aaa").pack(anchor="w")
        self.preview = tk.Label(right, bg="#000")
        self.preview.pack(fill=tk.BOTH, expand=True, pady=4)
        tk.Label(
            right,
            textvariable=self.info_var,
            bg="#1e1e1e",
            fg="#8c8",
            justify="left",
            wraplength=280,
        ).pack(anchor="w", pady=4)

        bottom = tk.Frame(self.root, bg="#2a2a2a")
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        mode_fr = tk.Frame(bottom, bg="#2a2a2a")
        mode_fr.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 2))
        tk.Label(mode_fr, text="模式:", bg="#2a2a2a", fg="#ccc").pack(side=tk.LEFT)
        for text, val in (
            ("自动框选", self.MODE_AUTO),
            ("伺服框选", self.MODE_SERVO),
            ("手动框选", self.MODE_MANUAL),
        ):
            tk.Radiobutton(
                mode_fr,
                text=text,
                variable=self.mode,
                value=val,
                bg="#2a2a2a",
                fg="#eee",
                selectcolor="#333",
                activebackground="#2a2a2a",
                command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=8)

        btn_fr = tk.Frame(bottom, bg="#2a2a2a")
        btn_fr.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 10))

        def b(text, cmd, color="#3a3a3a"):
            return tk.Button(
                btn_fr,
                text=text,
                command=cmd,
                bg=color,
                fg="#fff",
                activebackground="#555",
                relief=tk.FLAT,
                padx=14,
                pady=6,
            )

        b("上一张", self.prev_image).pack(side=tk.LEFT, padx=4)
        b("下一张 / 步进", self.next_image, "#2d5a8a").pack(side=tk.LEFT, padx=4)
        # 循环开始后该按钮自动变为「暂停」
        self.btn_loop = b("循环", self.toggle_loop, "#2d6a3a")
        self.btn_loop.pack(side=tk.LEFT, padx=4)
        b("保存当前", self.save_current, "#5a2d6a").pack(side=tk.LEFT, padx=4)
        b("重新检测", self.redetect, "#444").pack(side=tk.LEFT, padx=4)

        tk.Label(
            btn_fr,
            text="循环↔暂停同一按钮；伺服位移≤边长/15",
            bg="#2a2a2a",
            fg="#888",
        ).pack(side=tk.RIGHT, padx=8)

    def _pick_src(self) -> None:
        init = str(self.src_dir or (Path(__file__).parent / "captures"))
        chosen = filedialog.askdirectory(title="选择拍屏原图文件夹", initialdir=init)
        if chosen:
            self._load_folder(Path(chosen))

    def _pick_out(self) -> None:
        init = str(
            self.out_dir or Path(__file__).parent.parent / "Datasets" / "Recover"
        )
        chosen = filedialog.askdirectory(title="选择输出文件夹", initialdir=init)
        if chosen:
            self.out_dir = Path(chosen)
            self.status.set(f"输出 → {self.out_dir}")

    def _load_folder(self, folder: Path) -> None:
        self.src_dir = folder
        self.files = list_images(folder)
        self.index = 0
        self.prev_bgr = None
        self.prev_quad = None
        self.pause_loop()
        if not self.files:
            messagebox.showwarning("空目录", f"未找到图像:\n{folder}")
            self.status.set("未找到图像")
            return
        self.status.set(f"{folder.name} · {len(self.files)} 张")
        self._load_index(0, force_auto=True)

    def prev_image(self) -> None:
        if not self.files or self._busy:
            return
        self.pause_loop()
        self._load_index(max(0, self.index - 1), force_auto=False)

    def next_image(self, from_loop: bool = False) -> None:
        if not self.files or self._busy:
            return
        self.save_current(silent=True)
        if self.index >= len(self.files) - 1:
            self._looping = False
            self._paused = True
            self._set_loop_button(running=False)
            self.status.set("已到最后一张 · 可手动校准后保存")
            return
        self._load_index(self.index + 1, force_auto=False)

    def toggle_loop(self) -> None:
        """循环 / 暂停 同一按钮切换。"""
        if self._looping:
            self.pause_loop()
        else:
            self.start_loop()

    def start_loop(self) -> None:
        if not self.files:
            messagebox.showinfo("提示", "请先打开拍屏文件夹")
            return
        self._looping = True
        self._paused = False
        self._set_loop_button(running=True)
        self.status.set("循环中…（按钮已变为暂停）")
        self._schedule_loop()

    def pause_loop(self) -> None:
        self._paused = True
        self._looping = False
        if self._loop_job is not None:
            try:
                self.root.after_cancel(self._loop_job)
            except Exception:
                pass
            self._loop_job = None
        self._set_loop_button(running=False)
        if self.files:
            self.status.set(f"已暂停 [{self.index + 1}/{len(self.files)}]")

    def _set_loop_button(self, *, running: bool) -> None:
        if not hasattr(self, "btn_loop") or self.btn_loop is None:
            return
        if running:
            self.btn_loop.config(text="暂停", bg="#8a5a2d")
        else:
            self.btn_loop.config(text="循环", bg="#2d6a3a")

    def _schedule_loop(self) -> None:
        if not self._looping or self._paused:
            return
        self.next_image(from_loop=True)
        # 按处理速度连播：本张已处理完，仅让出 1ms 给 UI 刷新
        if self._looping and not self._paused:
            self._loop_job = self.root.after(1, self._schedule_loop)

    def _on_mode_change(self) -> None:
        m = self.mode.get()
        tips = {
            self.MODE_AUTO: "自动检测定位点",
            self.MODE_SERVO: "继承上张透视，位移≤边长/15",
            self.MODE_MANUAL: "不自动改框，随时可拖角",
        }
        self.info_var.set(f"模式 → {m}  |  {tips.get(m, '')}")

    def redetect(self) -> None:
        if self.bgr is None:
            return
        self._apply_mode_quad(force_auto=True)
        self._update_warp()
        self._redraw()

    def _load_index(self, i: int, *, force_auto: bool) -> None:
        if not self.files:
            return
        self._busy = True
        try:
            path = self.files[i]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
            if bgr is None:
                self.status.set(f"无法读取 {path.name}")
                return

            if self.bgr is not None and self.quad is not None:
                self.prev_bgr = self.bgr
                self.prev_quad = self.quad.copy()

            self.index = i
            self.bgr = bgr
            t0 = time.perf_counter()
            self._apply_mode_quad(force_auto=force_auto)
            self._last_detect_ms = (time.perf_counter() - t0) * 1000.0
            self._update_warp()
            self._redraw()
            idx = parse_index(path)
            self.status.set(
                f"[{i + 1}/{len(self.files)}] {path.name}  "
                f"idx={idx}  detect={self._last_detect_ms:.0f}ms  "
                f"mode={self.mode.get()}"
            )
        finally:
            self._busy = False

    def _apply_mode_quad(self, *, force_auto: bool) -> None:
        assert self.bgr is not None
        h, w = self.bgr.shape[:2]
        mode = self.mode.get()
        info: dict = {}

        def _inherit_scaled_quad() -> bool:
            """从 prev_quad / 当前 quad 继承并按分辨率缩放。成功返回 True。"""
            src = None
            if self.prev_quad is not None and self.prev_bgr is not None:
                src = self.prev_quad
                ph, pw = self.prev_bgr.shape[:2]
            elif self.quad is not None:
                src = self.quad
                ph, pw = h, w
            else:
                return False
            q = src.copy().astype(np.float32)
            if (ph, pw) != (h, w):
                q[:, 0] *= w / float(pw)
                q[:, 1] *= h / float(ph)
            q[:, 0] = np.clip(q[:, 0], 0, w - 1)
            q[:, 1] = np.clip(q[:, 1], 0, h - 1)
            self.quad = q
            return True

        if (
            mode == self.MODE_MANUAL
            and not force_auto
            and self.quad is not None
        ):
            _inherit_scaled_quad()
            info = {"method": "manual"}
        elif (
            mode == self.MODE_SERVO
            and self.prev_quad is not None
            and self.prev_bgr is not None
            and not force_auto
        ):
            dx, dy = estimate_shift_phase(self.prev_bgr, self.bgr)
            q = self.prev_quad.copy()
            ph, pw = self.prev_bgr.shape[:2]
            if (ph, pw) != (h, w):
                q[:, 0] *= w / float(pw)
                q[:, 1] *= h / float(ph)
            prev_scaled = q.copy()
            max_shift = float(max(h, w)) / 15.0
            q[:, 0] += dx
            q[:, 1] += dy
            q[:, 0] = np.clip(q[:, 0], 0, w - 1)
            q[:, 1] = np.clip(q[:, 1], 0, h - 1)
            q = refine_quad_local(self.bgr, q, search=min(14, int(max_shift)))
            q = clamp_quad_shift(prev_scaled, q, max_shift)
            q[:, 0] = np.clip(q[:, 0], 0, w - 1)
            q[:, 1] = np.clip(q[:, 1], 0, h - 1)
            self.quad = order_quad_tl_tr_br_bl(q)
            info = {
                "method": "servo",
                "dx": round(dx, 1),
                "dy": round(dy, 1),
                "max_shift": round(max_shift, 1),
            }
        else:
            q, info = detect_auto_quad(self.bgr)
            if q is None:
                self.quad = default_quad(h, w)
                info = {
                    **info,
                    "method": info.get("method") or "default",
                    "fallback": "default_rect",
                }
            else:
                self.quad = order_quad_tl_tr_br_bl(q.astype(np.float32))

        extra = ""
        if "dx" in info:
            extra = f"  Δ=({info.get('dx')},{info.get('dy')})"
        self.info_var.set(
            f"method={info.get('method')}  "
            f"conf={info.get('confidence', '-')}  "
            f"n={info.get('n_required', '-')}{extra}"
        )

    def _update_warp(self) -> None:
        if self.bgr is None or self.quad is None:
            self.warped = None
            return
        self.warped = warp_content(self.bgr, self.quad, out_size=self.out_size)
        self._show_preview(self.warped)

    def _show_preview(self, warped: np.ndarray) -> None:
        scale = max(2, 256 // max(warped.shape[0], 1))
        big = cv2.resize(
            warped,
            (warped.shape[1] * scale, warped.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
        self._photo_prev = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self._photo_prev)

    def _redraw(self) -> None:
        if self.bgr is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        h, w = self.bgr.shape[:2]
        self._scale = min(cw / w, ch / h)
        nw, nh = int(w * self._scale), int(h * self._scale)
        self._ox = (cw - nw) // 2
        self._oy = (ch - nh) // 2

        disp = cv2.resize(self.bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        if self.quad is not None:
            q = (self.quad * self._scale).astype(np.int32)
            cv2.polylines(disp, [q], True, (0, 255, 80), 2)
            for i, (x, y) in enumerate(q):
                cv2.circle(disp, (int(x), int(y)), 7, (0, 60, 255), -1)
                cv2.putText(
                    disp,
                    str(i),
                    (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 220, 0),
                    1,
                    cv2.LINE_AA,
                )

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo_main = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(
            self._ox, self._oy, anchor=tk.NW, image=self._photo_main
        )

    def _canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        ix = (x - self._ox) / max(self._scale, 1e-6)
        iy = (y - self._oy) / max(self._scale, 1e-6)
        return ix, iy

    def _hit_corner(self, x: float, y: float, radius: float = 12.0) -> Optional[int]:
        if self.quad is None:
            return None
        for i, (qx, qy) in enumerate(self.quad):
            cx = self._ox + qx * self._scale
            cy = self._oy + qy * self._scale
            if (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2:
                return i
        return None

    def _on_press(self, e) -> None:
        self._drag_i = self._hit_corner(e.x, e.y)

    def _on_drag(self, e) -> None:
        if self._drag_i is None or self.quad is None or self.bgr is None:
            return
        ix, iy = self._canvas_to_image(e.x, e.y)
        h, w = self.bgr.shape[:2]
        self.quad[self._drag_i, 0] = float(np.clip(ix, 0, w - 1))
        self.quad[self._drag_i, 1] = float(np.clip(iy, 0, h - 1))
        self._update_warp()
        self._redraw()

    def _on_release(self, _e) -> None:
        self._drag_i = None
        if self.quad is not None:
            self.quad = order_quad_tl_tr_br_bl(self.quad)
            self._update_warp()
            self._redraw()

    def save_current(self, silent: bool = False) -> None:
        if self.warped is None or not self.files:
            return
        if self.out_dir is None:
            if silent:
                return
            self._pick_out()
            if self.out_dir is None:
                return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.files[self.index]
        idx = parse_index(path)
        name = f"{idx}.png" if idx is not None else f"{path.stem}.png"
        out_path = self.out_dir / name
        cv2.imwrite(str(out_path), self.warped)
        if not silent:
            self.status.set(f"已保存 {out_path.name} → {self.out_dir}")

    def _on_close(self) -> None:
        self.pause_loop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="拍屏矫正 GUI")
    p.add_argument("--src", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--out_size", type=int, default=128)
    args = p.parse_args()

    src = args.src
    if src is not None and not src.is_absolute():
        alt = root / src
        if alt.exists():
            src = alt
    out = args.out
    if out is not None and not out.is_absolute():
        out = (root.parent / out).resolve()

    app = RectifyGUI(src=src, out=out, out_size=args.out_size)
    app.run()
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    raise SystemExit(main())
