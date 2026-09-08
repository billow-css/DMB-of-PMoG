#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/client.py
====================
屏摄实验客户端：全屏图片预览器 + 顶部指示灯 + 开始/暂停。

显示：宿主图等比放入画面，四周粗白框 + 四角 ArUco（供批量透视矫正）。

启动时必须指定测试图片目录（对话框或 --image_dir）。

指示灯
------
- 灰：空闲 / 暂停 / 等待指令
- 绿：已显示本张，准备发 READY
- 红闪：已发 READY，等待服务器 ACK(1)

用法
----
  python client.py
  python client.py --server 192.168.1.10 --port 8765 --image_dir D:/images
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional, Tuple

import numpy as np

# 允许从仓库根目录执行: python experiment/client.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageTk

from protocol import LineProtocol, event_error, event_ready, hello_client
from markers import compose_board_bgr

log = logging.getLogger("experiment.client")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Indicator:
    IDLE = "#555555"
    GREEN = "#22c55e"
    RED = "#ef4444"

    def __init__(self, canvas: tk.Canvas, x: int, y: int, r: int = 14):
        self.canvas = canvas
        self.x, self.y, self.r = x, y, r
        self._item = canvas.create_oval(
            x - r, y - r, x + r, y + r, fill=self.IDLE, outline="#111", width=2
        )
        self._flash = False
        self._flash_on = False
        self._after_id: Optional[str] = None

    def set_idle(self) -> None:
        self._stop_flash()
        self.canvas.itemconfigure(self._item, fill=self.IDLE)

    def set_green(self) -> None:
        self._stop_flash()
        self.canvas.itemconfigure(self._item, fill=self.GREEN)

    def set_red_flash(self) -> None:
        self._flash = True
        self._flash_on = False
        self._tick()

    def _stop_flash(self) -> None:
        self._flash = False
        if self._after_id is not None:
            try:
                self.canvas.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _tick(self) -> None:
        if not self._flash:
            return
        self._flash_on = not self._flash_on
        self.canvas.itemconfigure(
            self._item, fill=self.RED if self._flash_on else "#3f0a0a"
        )
        self._after_id = self.canvas.after(280, self._tick)


def count_images(folder: Path) -> int:
    return sum(
        1
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )


def prompt_setup(
    *,
    default_server: str = "127.0.0.1",
    default_port: int = 8765,
    default_image_dir: Optional[Path] = None,
) -> Optional[Tuple[Path, str, int]]:
    """
    启动前配置：必须指定图片目录。

    Returns
    -------
    (image_dir, server, port) 或取消时 None
    """
    root = tk.Tk()
    root.title("屏摄客户端 — 选择测试图片目录")
    root.geometry("640x220")
    root.resizable(False, False)

    result: dict = {"ok": False}

    frm = tk.Frame(root, padx=16, pady=16)
    frm.pack(fill=tk.BOTH, expand=True)

    tk.Label(frm, text="测试图片目录（必选，与服务器同一套编号图）", anchor="w").pack(
        fill=tk.X
    )
    path_row = tk.Frame(frm)
    path_row.pack(fill=tk.X, pady=(4, 8))
    path_var = tk.StringVar(
        value=str(default_image_dir) if default_image_dir else ""
    )
    path_entry = tk.Entry(path_row, textvariable=path_var, width=56)
    path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def browse() -> None:
        initial = path_var.get().strip() or str(Path.home())
        chosen = filedialog.askdirectory(
            parent=root,
            title="选择测试图片目录",
            initialdir=initial if Path(initial).is_dir() else str(Path.home()),
        )
        if chosen:
            path_var.set(chosen)

    tk.Button(path_row, text="浏览…", width=8, command=browse).pack(
        side=tk.LEFT, padx=(8, 0)
    )

    net = tk.Frame(frm)
    net.pack(fill=tk.X, pady=(4, 8))
    tk.Label(net, text="服务器 IP").pack(side=tk.LEFT)
    server_var = tk.StringVar(value=default_server)
    tk.Entry(net, textvariable=server_var, width=18).pack(side=tk.LEFT, padx=(6, 16))
    tk.Label(net, text="端口").pack(side=tk.LEFT)
    port_var = tk.StringVar(value=str(default_port))
    tk.Entry(net, textvariable=port_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

    tip = tk.Label(frm, text="", fg="#666666", anchor="w")
    tip.pack(fill=tk.X, pady=(0, 8))

    def on_ok() -> None:
        raw = path_var.get().strip().strip('"')
        if not raw:
            messagebox.showerror("缺少目录", "请先选择测试图片目录。", parent=root)
            return
        folder = Path(raw)
        if not folder.is_dir():
            messagebox.showerror("目录无效", f"不存在：\n{folder}", parent=root)
            return
        n = count_images(folder)
        if n <= 0:
            messagebox.showerror(
                "目录为空",
                f"目录下没有图片：\n{folder}",
                parent=root,
            )
            return
        try:
            port = int(port_var.get().strip())
        except ValueError:
            messagebox.showerror("端口无效", "端口必须是整数。", parent=root)
            return
        server = server_var.get().strip() or "127.0.0.1"
        result["ok"] = True
        result["image_dir"] = folder
        result["server"] = server
        result["port"] = port
        tip.configure(text=f"已选 {n} 张图 → {folder}")
        root.destroy()

    def on_cancel() -> None:
        root.destroy()

    btns = tk.Frame(frm)
    btns.pack(fill=tk.X)
    tk.Button(btns, text="取消", width=10, command=on_cancel).pack(side=tk.RIGHT)
    tk.Button(btns, text="确定", width=10, command=on_ok).pack(side=tk.RIGHT, padx=8)

    # 进入时直接弹出目录选择，强制用户指定
    root.after(100, browse)
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()

    if not result.get("ok"):
        return None
    return result["image_dir"], result["server"], result["port"]


class PreviewClient:
    def __init__(
        self,
        image_dir: Path,
        server_host: str,
        server_port: int,
        ready_delay_s: float = 0.03,
    ):
        self.image_dir = Path(image_dir)
        self.server_host = server_host
        self.server_port = server_port
        self.ready_delay_s = ready_delay_s
        self._ready_after_id = None

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")

        self.root = tk.Tk()
        self.root.title("PIMoG Screen Capture Client")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.shutdown())
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<space>", lambda e: self._toggle_run())

        # 顶部条：指示灯 + 状态 + 开始/暂停
        self.top = tk.Frame(self.root, bg="#111111", height=52)
        self.top.pack(side=tk.TOP, fill=tk.X)
        self.top.pack_propagate(False)

        self.ind_canvas = tk.Canvas(
            self.top, width=48, height=48, bg="#111111", highlightthickness=0
        )
        self.ind_canvas.pack(side=tk.LEFT, padx=8)
        self.indicator = Indicator(self.ind_canvas, 24, 24, r=12)

        self.status_var = tk.StringVar(
            value=f"已就绪 · 图库 {self.image_dir} · 按「开始」连接服务器"
        )
        self.status_lbl = tk.Label(
            self.top,
            textvariable=self.status_var,
            fg="#e5e5e5",
            bg="#111111",
            font=("Segoe UI", 13),
            anchor="w",
        )
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self._btn_run = tk.Button(
            self.top,
            text="开始",
            width=8,
            font=("Segoe UI", 12, "bold"),
            bg="#22c55e",
            fg="#000000",
            activebackground="#16a34a",
            command=self._toggle_run,
        )
        self._btn_run.pack(side=tk.RIGHT, padx=(0, 8), pady=8)

        self.hint_lbl = tk.Label(
            self.top,
            text="空格=开始/暂停 · Esc 退出 · F11 全屏",
            fg="#888888",
            bg="#111111",
            font=("Segoe UI", 10),
        )
        self.hint_lbl.pack(side=tk.RIGHT, padx=8)

        # 图像区
        self.img_label = tk.Label(self.root, bg="black")
        self.img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._proto: Optional[LineProtocol] = None
        self._sock: Optional[socket.socket] = None
        self._net_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._current_index: Optional[int] = None
        self._fullscreen = True

        # 运行门闩：False=暂停（不发 READY）；True=运行中
        self._running = False
        self._run_gate = threading.Event()  # set=允许发 READY
        self._connected = False
        self._connecting = False

        n = count_images(self.image_dir)
        self._set_status(
            f"暂停中 · 图库 {n} 张 · {self.image_dir.name} · 点「开始」连接 {self.server_host}:{self.server_port}"
        )

        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

    def _toggle_fullscreen(self, _event=None) -> None:
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_button_running(self, running: bool) -> None:
        def apply() -> None:
            self._running = running
            if running:
                self._btn_run.configure(
                    text="暂停", bg="#f59e0b", activebackground="#d97706"
                )
                self._run_gate.set()
            else:
                self._btn_run.configure(
                    text="开始", bg="#22c55e", activebackground="#16a34a"
                )
                self._run_gate.clear()
                self.indicator.set_idle()

        self.root.after(0, apply)

    def _toggle_run(self) -> None:
        if self._stop.is_set():
            return
        if not self._running:
            # 开始
            self._set_button_running(True)
            if not self._connected and not self._connecting:
                self._start_network()
            self._set_status(
                f"运行中 · 已连接等待指令"
                if self._connected
                else f"运行中 · 正在连接 {self.server_host}:{self.server_port} …"
            )
        else:
            # 暂停：仍可收 SHOW 并显示，但不发 READY，直到再次开始
            self._set_button_running(False)
            self._set_status(
                f"已暂停 · 当前图 index={self._current_index} · 点「开始」继续发 READY"
            )

    def _start_network(self) -> None:
        if self._net_thread and self._net_thread.is_alive():
            return
        self._connecting = True
        self._net_thread = threading.Thread(target=self._net_loop, daemon=True)
        self._net_thread.start()

    def _wait_until_running(self) -> bool:
        """暂停时阻塞，直到开始或退出。返回 False 表示应中止。"""
        while not self._stop.is_set():
            if self._run_gate.wait(timeout=0.2):
                return True
        return False

    def _show_image(self, name: str) -> ImageTk.PhotoImage:
        path = self.image_dir / name
        if not path.is_file():
            alt = None
            for ext in IMAGE_EXTS:
                cand = self.image_dir / f"{Path(name).stem}{ext}"
                if cand.is_file():
                    alt = cand
                    break
            if alt is None:
                raise FileNotFoundError(f"找不到图片: {path}")
            path = alt

        # 顶部条约 52px；下方全屏合成：宿主等比 + 粗白框 + 四角 ArUco
        screen_w = self.root.winfo_screenwidth()
        screen_h = max(1, self.root.winfo_screenheight() - 52)
        with Image.open(path) as im:
            content_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

        board_bgr, _layout = compose_board_bgr(
            content_rgb, screen_w, screen_h
        )
        board_rgb = board_bgr[:, :, ::-1]
        return ImageTk.PhotoImage(Image.fromarray(board_rgb))

    def _send_ready(self, index: int) -> None:
        """画面上屏后再发 READY（避免拍到上一张）。"""
        if self._stop.is_set() or self._proto is None:
            return
        if not self._run_gate.is_set():
            return
        if self._current_index != index:
            return
        try:
            self._proto.send(event_ready(index))
        except Exception as e:
            log.exception("发送 READY 失败: %s", e)
            return
        self.indicator.set_red_flash()
        self._set_status(f"READY 已发送 [{index}] · 红闪等待拍摄 ACK(1)")

    def _on_show(self, index: int, name: str) -> None:
        self._current_index = index
        if self._ready_after_id is not None:
            try:
                self.root.after_cancel(self._ready_after_id)
            except Exception:
                pass
            self._ready_after_id = None

        self.root.after(0, self.indicator.set_green)
        self._set_status(f"显示 [{index}] {name}")
        try:
            photo = self._show_image(name)
        except Exception as e:
            log.exception("显示失败")
            if self._proto:
                self._proto.send(event_error(index, str(e)))
            self._set_status(f"错误: {e}")
            return

        painted = threading.Event()

        def apply():
            self._photo = photo
            self.img_label.configure(image=photo)
            painted.set()

        self.root.after(0, apply)
        # 等主线程真正上屏（最多 0.5s），再按 ready_delay 发 READY
        painted.wait(timeout=0.5)

        if not self._run_gate.is_set():
            self._set_status(
                f"已暂停 · 已显示 [{index}] {name} · 点「开始」后发送 READY"
            )
            if not self._wait_until_running():
                return

        if self._stop.is_set() or self._proto is None:
            return
        if not self._wait_until_running():
            return

        delay_ms = max(0, int(self.ready_delay_s * 1000))
        if delay_ms <= 0:
            self._send_ready(index)
        else:
            self._ready_after_id = self.root.after(
                delay_ms, lambda: self._send_ready(index)
            )
            # 阻塞到 READY 发出或超时，保持与服务器「SHOW→READY」时序
            time.sleep(self.ready_delay_s + 0.01)

    def _on_ack(self) -> None:
        self.root.after(0, self.indicator.set_idle)
        self._set_status(
            f"ACK(1) 收到 · 本张完成"
            + (
                f" (index={self._current_index})"
                if self._current_index is not None
                else ""
            )
            + (" · 运行中" if self._running else " · 已暂停")
        )

    def _net_loop(self) -> None:
        try:
            self._set_status(f"连接 {self.server_host}:{self.server_port} …")
            sock = socket.create_connection(
                (self.server_host, self.server_port), timeout=30.0
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
            self._proto = LineProtocol(sock)
            self._proto.send(hello_client())
            self._connected = True
            self._connecting = False
            self._set_status(
                "已连接 · 运行中，等待 SHOW …"
                if self._run_gate.is_set()
                else "已连接 · 已暂停（点「开始」后才会发 READY）"
            )
            self.root.after(0, self.indicator.set_idle)

            while not self._stop.is_set():
                try:
                    msg = self._proto.recv(timeout=60.0)
                except socket.timeout:
                    continue
                except ConnectionError:
                    break

                mtype = msg.get("type")
                if mtype == "ping":
                    self._proto.send({"type": "pong"})
                    continue
                if mtype != "cmd":
                    log.warning("忽略报文: %s", msg)
                    continue

                action = msg.get("action")
                if action == "show":
                    self._on_show(int(msg["index"]), str(msg.get("name", "")))
                elif action == "ack":
                    self._on_ack()
                elif action == "finish":
                    self._set_status("服务器结束 · 全部完成")
                    self.root.after(0, self.indicator.set_idle)
                    self._set_button_running(False)
                    break
                elif action == "abort":
                    self._set_status(f"中止: {msg.get('reason', '')}")
                    self.root.after(0, self.indicator.set_idle)
                    self._set_button_running(False)
                    break
                else:
                    log.warning("未知 cmd: %s", action)
        except Exception as e:
            log.exception("网络线程异常")
            self._set_status(f"连接失败: {e}")
            self._set_button_running(False)
        finally:
            self._connected = False
            self._connecting = False
            try:
                if self._sock:
                    self._sock.close()
            except Exception:
                pass

    def start(self) -> None:
        # 不自动连网，等用户点「开始」
        self.root.mainloop()

    def shutdown(self) -> None:
        self._stop.set()
        self._run_gate.set()  # 解除可能的等待
        try:
            if self._sock:
                self._sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        self.root.after(0, self.root.destroy)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="屏摄实验客户端（全屏预览）")
    p.add_argument("--server", default="127.0.0.1", help="服务器 IP（可在启动对话框修改）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="本地图库；省略则启动时强制选择目录",
    )
    p.add_argument(
        "--ready_delay",
        type=float,
        default=0.03,
        help="上屏后再延迟发 READY（秒）；电子快门默认 0.03",
    )
    p.add_argument(
        "--no_setup_dialog",
        action="store_true",
        help="不弹对话框（必须同时提供 --image_dir）",
    )
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args()

    if args.no_setup_dialog:
        if args.image_dir is None:
            print("错误: --no_setup_dialog 时必须指定 --image_dir", file=sys.stderr)
            return 2
        image_dir = Path(args.image_dir)
        server = args.server
        port = args.port
    else:
        # 即使给了 --image_dir，也弹对话框确认/可改；作为初始值填入
        setup = prompt_setup(
            default_server=args.server,
            default_port=args.port,
            default_image_dir=args.image_dir,
        )
        if setup is None:
            print("已取消。")
            return 0
        image_dir, server, port = setup

    if not image_dir.is_dir():
        print(f"错误: 图片目录不存在: {image_dir}", file=sys.stderr)
        return 1

    client = PreviewClient(
        image_dir=image_dir,
        server_host=server,
        server_port=port,
        ready_delay_s=args.ready_delay,
    )
    client.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
