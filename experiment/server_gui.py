#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/server_gui.py
========================
屏摄实验服务器 GUI：端口常驻监听 + 相机实时 Live View。

相对 CLI ``server.py``：
  - 先「开启监听」，端口一直开着等客户端反复连接
  - 「开始拍摄」跑一轮会话；结束后继续监听，可再开下一轮
  - 「停止本轮」只中断当前拍摄，不断端口
  - 相机可跨会话保持连接；可独立「开启预览」读 EVF 画面

用法
----
  python server_gui.py
  python server_gui.py --port 8765 --camera edsdk
"""

from __future__ import annotations

import argparse
import logging
import queue
import socket
import sys
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera import create_camera
from config import ExperimentConfig, load_config
from server import (
    accept_client,
    create_listen_socket,
    list_resumable_sessions,
    resolve_path,
    run_session,
    setup_logging,
)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"需要 tkinter: {e}") from e

try:
    from PIL import Image, ImageTk
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"需要 Pillow: {e}") from e


class QueueLogHandler(logging.Handler):
    """把日志丢进队列，由 GUI 主线程刷到文本框。"""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(self.format(record))
        except Exception:
            pass


class ServerGUI:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.root = tk.Tk()
        self.root.title("PIMoG 屏摄服务器 · 常驻端口 / 实时预览")
        self.root.geometry("1180x720")
        self.root.minsize(960, 600)
        self.root.configure(bg="#1e1e1e")

        self.host_var = tk.StringVar(value=cfg.net.host or "0.0.0.0")
        self.port_var = tk.IntVar(value=int(cfg.net.port or 8765))
        self.image_dir_var = tk.StringVar(value=str(cfg.paths.image_dir))
        self.capture_var = tk.StringVar(value=str(cfg.paths.capture_root))
        self.camera_var = tk.StringVar(value=str(cfg.camera_backend))
        self.session_var = tk.StringVar(value="")
        self.resume_var = tk.BooleanVar(value=bool(cfg.resume))
        self.keep_cam_var = tk.BooleanVar(value=True)
        self.dry_run_var = tk.BooleanVar(value=bool(cfg.dry_run))
        self.status_var = tk.StringVar(value="未监听 — 请点「开启监听」")
        self.preview_status_var = tk.StringVar(value="预览未开启")

        self._srv = None
        self._listen_thread: Optional[threading.Thread] = None
        self._session_thread: Optional[threading.Thread] = None
        self._accepting = False
        self._stop_session = threading.Event()
        self._client_conn = None
        self._client_proto = None
        self._client_addr = None
        self._client_lock = threading.Lock()
        self._camera = None
        self._camera_lock = threading.Lock()
        self._log_q: queue.Queue = queue.Queue()
        self._session_busy = False

        self._preview_wanted = False
        self._preview_thread: Optional[threading.Thread] = None
        self._preview_stop = threading.Event()
        self._preview_pause = threading.Event()  # set = 暂停抓帧
        self._preview_q: queue.Queue = queue.Queue(maxsize=2)
        self._photo = None  # 防 GC
        self._preview_fps = 0.0
        self._preview_frames = 0
        self._preview_fps_t0 = time.time()

        self._build_ui()
        self._setup_logging()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._drain_logs)
        self.root.after(40, self._drain_preview)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#2a2a2a")
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        r1 = tk.Frame(top, bg="#2a2a2a")
        r1.pack(fill=tk.X, pady=2)
        tk.Label(r1, text="Host", width=12, anchor="e", bg="#2a2a2a", fg="#ccc").pack(
            side=tk.LEFT
        )
        tk.Entry(r1, textvariable=self.host_var, width=16).pack(side=tk.LEFT, padx=4)
        tk.Label(r1, text="Port", bg="#2a2a2a", fg="#ccc").pack(side=tk.LEFT)
        tk.Entry(r1, textvariable=self.port_var, width=8).pack(side=tk.LEFT, padx=4)
        tk.Label(r1, text="相机", bg="#2a2a2a", fg="#ccc").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Combobox(
            r1,
            textvariable=self.camera_var,
            values=["edsdk", "stub"],
            width=8,
            state="readonly",
        ).pack(side=tk.LEFT, padx=4)

        r2 = tk.Frame(top, bg="#2a2a2a")
        r2.pack(fill=tk.X, pady=2)
        tk.Label(r2, text="图库", width=12, anchor="e", bg="#2a2a2a", fg="#ccc").pack(
            side=tk.LEFT
        )
        tk.Entry(r2, textvariable=self.image_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(r2, text="…", command=self._pick_image_dir, width=3).pack(side=tk.LEFT)

        r3 = tk.Frame(top, bg="#2a2a2a")
        r3.pack(fill=tk.X, pady=2)
        tk.Label(r3, text="回传目录", width=12, anchor="e", bg="#2a2a2a", fg="#ccc").pack(
            side=tk.LEFT
        )
        tk.Entry(r3, textvariable=self.capture_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(r3, text="…", command=self._pick_capture, width=3).pack(side=tk.LEFT)

        r4 = tk.Frame(top, bg="#2a2a2a")
        r4.pack(fill=tk.X, pady=2)
        tk.Label(r4, text="会话ID", width=12, anchor="e", bg="#2a2a2a", fg="#ccc").pack(
            side=tk.LEFT
        )
        tk.Entry(r4, textvariable=self.session_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Checkbutton(
            r4,
            text="续跑断点",
            variable=self.resume_var,
            bg="#2a2a2a",
            fg="#eee",
            selectcolor="#333",
            activebackground="#2a2a2a",
        ).pack(side=tk.LEFT, padx=6)
        tk.Button(r4, text="选断点", command=self._pick_session).pack(side=tk.LEFT, padx=2)

        opts = tk.Frame(top, bg="#2a2a2a")
        opts.pack(fill=tk.X, pady=2)
        tk.Checkbutton(
            opts,
            text="跨会话保持相机连接",
            variable=self.keep_cam_var,
            bg="#2a2a2a",
            fg="#eee",
            selectcolor="#333",
            activebackground="#2a2a2a",
        ).pack(side=tk.LEFT, padx=8)
        tk.Checkbutton(
            opts,
            text="dry_run（不真拍）",
            variable=self.dry_run_var,
            bg="#2a2a2a",
            fg="#eee",
            selectcolor="#333",
            activebackground="#2a2a2a",
        ).pack(side=tk.LEFT, padx=8)

        btns = tk.Frame(self.root, bg="#1e1e1e")
        btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        def b(text, cmd, color="#3a3a3a"):
            return tk.Button(
                btns,
                text=text,
                command=cmd,
                bg=color,
                fg="#fff",
                activebackground="#555",
                relief=tk.FLAT,
                padx=10,
                pady=5,
            )

        self.btn_listen = b("开启监听", self.start_listen, "#2d6a3a")
        self.btn_listen.pack(side=tk.LEFT, padx=3)
        self.btn_unlisten = b("关闭监听", self.stop_listen, "#6a2d2d")
        self.btn_unlisten.pack(side=tk.LEFT, padx=3)
        self.btn_start = b("开始拍摄", self.start_session, "#2d5a8a")
        self.btn_start.pack(side=tk.LEFT, padx=3)
        self.btn_stop = b("停止本轮", self.stop_session, "#8a5a2d")
        self.btn_stop.pack(side=tk.LEFT, padx=3)
        b("新会话ID", self._new_session_id, "#444").pack(side=tk.LEFT, padx=3)

        tk.Frame(btns, width=12, bg="#1e1e1e").pack(side=tk.LEFT)
        self.btn_cam = b("连接相机", self.connect_camera, "#3a4a5a")
        self.btn_cam.pack(side=tk.LEFT, padx=3)
        self.btn_preview = b("开启预览", self.start_preview, "#2d6a6a")
        self.btn_preview.pack(side=tk.LEFT, padx=3)
        self.btn_preview_stop = b("停止预览", self.stop_preview, "#5a3a5a")
        self.btn_preview_stop.pack(side=tk.LEFT, padx=3)

        st = tk.Frame(self.root, bg="#1e1e1e")
        st.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)
        tk.Label(
            st, textvariable=self.status_var, bg="#1e1e1e", fg="#8cf", anchor="w"
        ).pack(fill=tk.X)
        tk.Label(
            st,
            textvariable=self.preview_status_var,
            bg="#1e1e1e",
            fg="#8c8",
            anchor="w",
        ).pack(fill=tk.X)

        body = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL, sashwidth=6, bg="#1e1e1e"
        )
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = tk.Frame(body, bg="#1e1e1e")
        right = tk.Frame(body, bg="#111")
        body.add(left, stretch="always", minsize=360)
        body.add(right, stretch="always", minsize=360)

        tk.Label(
            left,
            text="日志（端口常驻；拍完一轮后客户端可重新连接再开下一轮）",
            bg="#1e1e1e",
            fg="#888",
            anchor="w",
        ).pack(fill=tk.X)
        self.log_box = scrolledtext.ScrolledText(
            left,
            height=22,
            bg="#111",
            fg="#ddd",
            insertbackground="#fff",
            font=("Consolas", 9),
        )
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        tk.Label(
            right,
            text="相机实时画面（EDSDK Live View）",
            bg="#111",
            fg="#aaa",
            anchor="w",
        ).pack(fill=tk.X, padx=6, pady=(4, 2))
        self.preview_frame = tk.Frame(right, bg="#0a0a0a")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.preview_label = tk.Label(
            self.preview_frame,
            text="点击「开启预览」读取相机画面",
            bg="#0a0a0a",
            fg="#666",
            font=("Segoe UI", 12),
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        self._refresh_buttons()

    def _setup_logging(self) -> None:
        log_dir = resolve_path(self.cfg.paths.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        sid = datetime.now().strftime("gui_%Y%m%d_%H%M%S")
        setup_logging(log_dir, sid)
        root = logging.getLogger()
        qh = QueueLogHandler(self._log_q)
        qh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
            )
        )
        root.addHandler(qh)

    def _drain_logs(self) -> None:
        try:
            while True:
                line = self._log_q.get_nowait()
                self.log_box.insert(tk.END, line + "\n")
                self.log_box.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(200, self._drain_logs)

    def _drain_preview(self) -> None:
        latest = None
        try:
            while True:
                latest = self._preview_q.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self._show_preview_jpeg(latest)
        self.root.after(33, self._drain_preview)

    def _show_preview_jpeg(self, data: bytes) -> None:
        try:
            img = Image.open(BytesIO(data))
            img = img.convert("RGB")
            img.load()
        except Exception as e:
            self.preview_status_var.set(f"JPEG 解码失败: {e}")
            return
        cw = max(320, self.preview_frame.winfo_width())
        ch = max(240, self.preview_frame.winfo_height())
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        # 允许放大一点，避免窗格大时画面过小
        scale = min(scale, 2.0)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        if (nw, nh) != (iw, ih):
            img = img.resize((nw, nh), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._photo, text="")
        self._preview_frames += 1
        now = time.time()
        dt = now - self._preview_fps_t0
        if dt >= 1.0:
            self._preview_fps = self._preview_frames / dt
            self._preview_frames = 0
            self._preview_fps_t0 = now
            cam = self._camera
            name = getattr(cam, "model_name", "") or self.camera_var.get()
            state = "拍摄中暂停" if self._preview_pause.is_set() else "预览中"
            self.preview_status_var.set(
                f"{state} · {name} · {self._preview_fps:.1f} fps · {iw}x{ih}"
            )

    def _refresh_buttons(self) -> None:
        listening = self._accepting and self._srv is not None
        busy = self._session_busy
        cam_ok = self._camera is not None and getattr(
            self._camera, "connected", False
        )
        prev = self._preview_wanted
        self.btn_listen.config(state=tk.DISABLED if listening else tk.NORMAL)
        self.btn_unlisten.config(state=tk.NORMAL if listening else tk.DISABLED)
        self.btn_start.config(
            state=tk.NORMAL if listening and not busy else tk.DISABLED
        )
        self.btn_stop.config(state=tk.NORMAL if busy else tk.DISABLED)
        self.btn_cam.config(state=tk.DISABLED if cam_ok else tk.NORMAL)
        self.btn_preview.config(state=tk.DISABLED if prev else tk.NORMAL)
        self.btn_preview_stop.config(state=tk.NORMAL if prev else tk.DISABLED)

    def _pick_image_dir(self) -> None:
        d = filedialog.askdirectory(title="选择含水印图库")
        if d:
            self.image_dir_var.set(d)

    def _pick_capture(self) -> None:
        d = filedialog.askdirectory(title="选择回传保存目录")
        if d:
            self.capture_var.set(d)

    def _pick_session(self) -> None:
        root = resolve_path(self.cfg.paths.session_root)
        sessions = list_resumable_sessions(root, limit=12)
        if not sessions:
            messagebox.showinfo("断点", "没有可续跑会话")
            return
        win = tk.Toplevel(self.root)
        win.title("选择断点会话")
        win.geometry("520x280")
        lb = tk.Listbox(win)
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        for s in sessions:
            lb.insert(
                tk.END,
                f"{s['id']}  next={s['next_index']}  done={s['completed']}",
            )

        def ok():
            sel = lb.curselection()
            if not sel:
                return
            s = sessions[sel[0]]
            self.session_var.set(s["id"])
            self.resume_var.set(True)
            win.destroy()

        tk.Button(win, text="确定", command=ok).pack(pady=6)

    def _new_session_id(self) -> None:
        self.session_var.set(datetime.now().strftime("session_%Y%m%d_%H%M%S"))
        self.resume_var.set(False)

    def _apply_form_to_cfg(self) -> ExperimentConfig:
        cfg = self.cfg
        cfg.net.host = self.host_var.get().strip() or "0.0.0.0"
        cfg.net.port = int(self.port_var.get())
        cfg.paths.image_dir = self.image_dir_var.get().strip()
        cfg.paths.capture_root = self.capture_var.get().strip()
        cfg.camera_backend = self.camera_var.get().strip() or "stub"
        cfg.resume = bool(self.resume_var.get())
        cfg.dry_run = bool(self.dry_run_var.get())
        return cfg

    def _set_status(self, msg: str) -> None:
        def apply():
            self.status_var.set(msg)

        self.root.after(0, apply)

    def _cam_kwargs(self, cfg: ExperimentConfig) -> dict:
        kwargs = {}
        if cfg.camera_backend.lower() in ("edsdk", "canon"):
            kwargs["library_path"] = str(resolve_path(cfg.paths.edsdk_dll_dir))
            cd = cfg.cooldowns
            kwargs["shutter_non_af"] = bool(getattr(cd, "shutter_non_af", True))
            kwargs["shoot_retries"] = int(getattr(cd, "shoot_retries", 3))
            kwargs["keep_liveview"] = bool(getattr(cd, "keep_liveview", False))
            kwargs["post_download_pump_s"] = float(
                getattr(cd, "post_download_pump_s", 0.05)
            )
        return kwargs

    def _ensure_camera(self, for_preview: bool = False):
        cfg = self._apply_form_to_cfg()
        with self._camera_lock:
            cam = self._camera
            if cam is not None and getattr(cam, "connected", False):
                return cam
            cam = create_camera(cfg.camera_backend, **self._cam_kwargs(cfg))
            cam.connect()
            # 预览需要跨会话持有相机
            if for_preview or self.keep_cam_var.get():
                self._camera = cam
                self.keep_cam_var.set(True)
            return cam

    def connect_camera(self) -> None:
        try:
            cam = self._ensure_camera(for_preview=True)
            name = getattr(cam, "model_name", "") or self.camera_var.get()
            self.preview_status_var.set(f"相机已连接 · {name}")
            logging.getLogger("experiment.server_gui").info("相机已连接: %s", name)
        except Exception as e:
            messagebox.showerror("连接相机失败", str(e))
            return
        self._refresh_buttons()

    def start_preview(self) -> None:
        if self._preview_wanted:
            return
        try:
            self._ensure_camera(for_preview=True)
        except Exception as e:
            messagebox.showerror("开启预览失败", str(e))
            return

        self._preview_wanted = True
        self._preview_stop.clear()
        self._preview_frames = 0
        self._preview_fps_t0 = time.time()
        if self._session_busy:
            self._preview_pause.set()
            self.preview_status_var.set("预览已开（拍摄中暂停抓帧）")
        else:
            self._preview_pause.clear()
            self.preview_status_var.set("正在启动 Live View…（约 1–3 秒）")
        self.preview_label.configure(image="", text="启动 Live View 中…")
        self._preview_thread = threading.Thread(
            target=self._preview_loop, daemon=True, name="evf-preview"
        )
        self._preview_thread.start()
        self._refresh_buttons()

    def stop_preview(self) -> None:
        self._preview_wanted = False
        self._preview_stop.set()
        self._preview_pause.clear()
        with self._camera_lock:
            cam = self._camera
        if cam is not None and not self._session_busy:
            try:
                keep = bool(getattr(self.cfg.cooldowns, "keep_liveview", False))
                if not keep:
                    cam.stop_liveview()
            except Exception:
                pass
        self.preview_status_var.set("预览已停止")
        self.preview_label.configure(image="", text="预览已停止")
        self._photo = None
        self._refresh_buttons()

    def _preview_loop(self) -> None:
        log = logging.getLogger("experiment.server_gui")
        # 在预览线程启动 EVF，避免卡住 GUI；并初始化该线程 COM
        try:
            import ctypes

            ctypes.windll.ole32.CoInitializeEx(None, 0x0)
        except Exception:
            pass

        fails = 0
        started = False
        while not self._preview_stop.is_set() and self._preview_wanted:
            if self._preview_pause.is_set():
                time.sleep(0.08)
                continue
            with self._camera_lock:
                cam = self._camera
            if cam is None or not getattr(cam, "connected", False):
                time.sleep(0.2)
                continue
            try:
                if not started or not getattr(cam, "liveview_on", False):
                    cam.start_liveview()
                    started = True
                    self.root.after(
                        0,
                        lambda: self.preview_status_var.set("Live View 已开，抓帧中…"),
                    )
                data = cam.grab_evf_jpeg(retries=20)
            except Exception as e:
                log.warning("grab_evf: %s", e)
                data = None
                err = str(e)
            else:
                err = getattr(cam, "_last_evf_error", None)

            if data:
                fails = 0
                try:
                    while self._preview_q.full():
                        try:
                            self._preview_q.get_nowait()
                        except queue.Empty:
                            break
                    self._preview_q.put_nowait(data)
                except queue.Full:
                    pass
            else:
                fails += 1
                if fails in (1, 5, 15) or fails % 30 == 0:
                    msg = err or "无帧"
                    log.warning("EVF 抓帧失败 #%d: %s", fails, msg)

                    def _upd(m=msg, n=fails):
                        self.preview_status_var.set(f"抓帧失败 x{n}: {m}")

                    self.root.after(0, _upd)
                time.sleep(0.08)
            time.sleep(0.01)

    def start_listen(self) -> None:
        if self._accepting:
            return
        cfg = self._apply_form_to_cfg()
        try:
            self._srv = create_listen_socket(cfg.net.host, cfg.net.port)
        except OSError as e:
            messagebox.showerror("监听失败", str(e))
            return
        self._accepting = True
        self._set_status(f"监听中 {cfg.net.host}:{cfg.net.port} — 等待客户端连接…")
        self._listen_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._listen_thread.start()
        self._refresh_buttons()

    def stop_listen(self) -> None:
        self._accepting = False
        self.stop_session()
        self.stop_preview()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
            self._srv = None
        with self._client_lock:
            if self._client_conn is not None:
                try:
                    self._client_conn.close()
                except OSError:
                    pass
            self._client_conn = None
            self._client_proto = None
            self._client_addr = None
        with self._camera_lock:
            if self._camera is not None:
                try:
                    self._camera.close()
                except Exception:
                    pass
                self._camera = None
        self._set_status("已关闭监听")
        self.preview_status_var.set("预览未开启")
        self._refresh_buttons()

    def _accept_loop(self) -> None:
        log = logging.getLogger("experiment.server_gui")
        while self._accepting and self._srv is not None:
            try:
                conn, proto, addr = accept_client(self._srv, timeout=1.0)
            except socket.timeout:
                continue
            except OSError:
                if not self._accepting:
                    break
                continue
            except Exception as e:
                if not self._accepting:
                    break
                log.warning("accept/握手失败: %s", e)
                continue

            with self._client_lock:
                if self._client_conn is not None:
                    try:
                        self._client_conn.close()
                    except OSError:
                        pass
                self._client_conn = conn
                self._client_proto = proto
                self._client_addr = addr
            self._set_status(f"客户端已连接 {addr[0]}:{addr[1]} — 可点「开始拍摄」")

            while self._accepting:
                with self._client_lock:
                    if self._client_conn is not conn:
                        break
                time.sleep(0.4)

    def start_session(self) -> None:
        if self._session_busy:
            return
        with self._client_lock:
            if self._client_proto is None or self._client_conn is None:
                messagebox.showwarning(
                    "无客户端",
                    "请先「开启监听」，并在显示端运行 client 连接本机端口。",
                )
                return
            conn = self._client_conn
            proto = self._client_proto

        cfg = self._apply_form_to_cfg()
        sid = self.session_var.get().strip() or None
        if not cfg.resume and not sid:
            sid = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            self.session_var.set(sid)

        self._stop_session.clear()
        self._session_busy = True
        # 拍摄时暂停 EVF，避免与回传抢 USB；拍完可恢复
        self._preview_pause.set()
        if self._preview_wanted:
            self.preview_status_var.set("拍摄中 — 预览已暂停")
            keep_lv = bool(getattr(cfg.cooldowns, "keep_liveview", False))
            if not keep_lv:
                with self._camera_lock:
                    cam = self._camera
                if cam is not None:
                    try:
                        cam.stop_liveview()
                    except Exception:
                        pass
        self._refresh_buttons()
        self._set_status(f"拍摄中… session={sid}")

        def worker():
            log = logging.getLogger("experiment.server_gui")
            cam = None
            try:
                with self._camera_lock:
                    cam = self._camera
                if cam is None or not getattr(cam, "connected", False):
                    cam = create_camera(cfg.camera_backend, **self._cam_kwargs(cfg))
                    cam.connect()
                    if self.keep_cam_var.get():
                        with self._camera_lock:
                            self._camera = cam

                code = run_session(
                    cfg,
                    session_id=sid,
                    conn=conn,
                    proto=proto,
                    camera=cam,
                    stop_event=self._stop_session,
                    keep_camera=bool(self.keep_cam_var.get()),
                    close_conn=True,
                    on_status=self._set_status,
                )
                log.info("本轮结束 code=%s", code)
            except Exception:
                log.exception("本轮异常")
                self._set_status("本轮异常 — 端口仍在监听，可重连客户端再开")
            finally:
                with self._client_lock:
                    if self._client_conn is conn:
                        self._client_conn = None
                        self._client_proto = None
                        self._client_addr = None
                if not self.keep_cam_var.get():
                    with self._camera_lock:
                        if cam is not None and self._camera is cam:
                            try:
                                cam.close()
                            except Exception:
                                pass
                            self._camera = None
                self._session_busy = False

                def after_session():
                    self._refresh_buttons()
                    if self._preview_wanted and self._camera is not None:
                        try:
                            self._camera.start_liveview()
                        except Exception:
                            pass
                        self._preview_pause.clear()
                        self.preview_status_var.set("预览已恢复")
                    else:
                        self._preview_pause.clear()
                    if self._accepting:
                        self._set_status(
                            "本轮结束 — 端口仍开着，等待客户端重新连接后可再「开始拍摄」"
                        )

                self.root.after(0, after_session)

        self._session_thread = threading.Thread(target=worker, daemon=True)
        self._session_thread.start()

    def stop_session(self) -> None:
        if self._session_busy:
            self._stop_session.set()
            self._set_status("正在停止本轮…")

    def _on_close(self) -> None:
        self.stop_listen()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="屏摄实验服务器 GUI（常驻端口 / 实时预览）")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--image_dir", type=Path, default=None)
    p.add_argument("--capture_root", type=Path, default=None)
    p.add_argument("--camera", default=None, choices=["stub", "edsdk"])
    return p


def main() -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.host:
        cfg.net.host = args.host
    if args.port is not None:
        cfg.net.port = args.port
    if args.image_dir:
        cfg.paths.image_dir = str(args.image_dir)
    if args.capture_root:
        cfg.paths.capture_root = str(args.capture_root)
    if args.camera:
        cfg.camera_backend = args.camera

    app = ServerGUI(cfg)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
