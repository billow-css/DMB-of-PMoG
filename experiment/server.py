#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/server.py
====================
屏摄实验服务器（CLI）。

流程（每张）
------------
1. SHOW → 客户端全屏下一张；本机 AF；等客户端 READY
2. SHOOT → 等相机落盘
3. ACK(1) → 客户端红闪结束；冷却后下一张

大批量：分片目录、断点续跑、每 N 张长休息、日志落盘。

用法
----
  python server.py
  # 启动时询问：1=继续测试 / 2=从头测试

  python server.py --fresh              # 非交互从头
  python server.py --resume             # 非交互续最近会话
  python server.py --session session_xxx --resume

科研推荐常驻端口 GUI（会话结束不断端口）
--------------------------------------
  python server_gui.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 允许从仓库根目录执行: python experiment/server.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera import create_camera
from config import (
    REPO_ROOT,
    ExperimentConfig,
    capture_path_for_index,
    estimate_hours,
    load_config,
    save_config,
)
from protocol import (
    PROTOCOL_VERSION,
    LineProtocol,
    cmd_abort,
    cmd_ack,
    cmd_finish,
    cmd_show,
)

log = logging.getLogger("experiment.server")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def setup_logging(log_dir: Path, session_id: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{session_id}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # 清掉旧 handler，避免重复
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    log.info("日志文件: %s", log_path)


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    files = [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ]

    def key(p: Path):
        stem = p.stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    files.sort(key=key)
    if not files:
        raise FileNotFoundError(f"目录无图像: {image_dir}")
    return files


def free_space_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = os.statvfs(path) if hasattr(os, "statvfs") else None
    if usage is not None:
        return usage.f_bavail * usage.f_frsize / (1024**3)
    import shutil

    return shutil.disk_usage(str(path)).free / (1024**3)


class SessionState:
    def __init__(self, path: Path):
        self.path = path
        self.data = {
            "session_id": path.stem,
            "next_index": 0,
            "completed": 0,
            "failed": [],
            "updated_at": None,
        }

    @classmethod
    def load_or_create(cls, session_root: Path, session_id: str) -> "SessionState":
        session_root.mkdir(parents=True, exist_ok=True)
        path = session_root / f"{session_id}.json"
        st = cls(path)
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                st.data.update(json.load(f))
            log.info("载入断点: %s (next_index=%s)", path, st.data.get("next_index"))
        return st

    def save(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)


def create_listen_socket(host: str, port: int) -> socket.socket:
    """创建并绑定监听套接字（调用方负责 close）。"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    log.info("监听 %s:%d", host if host != "0.0.0.0" else "*", port)
    return srv


def accept_client(
    srv: socket.socket,
    timeout: Optional[float] = None,
) -> tuple[socket.socket, LineProtocol, tuple]:
    """
    在已监听套接字上 accept 一个客户端并完成握手。
    不关闭 srv（便于 GUI 常驻端口）。
    """
    if timeout is not None:
        srv.settimeout(timeout)
    else:
        srv.settimeout(None)
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    log.info("客户端已连接: %s:%d", addr[0], addr[1])
    proto = LineProtocol(conn)
    # accept 可用短超时轮询；握手仍给足时间
    hello_timeout = 30.0 if (timeout is None or timeout < 5.0) else float(timeout)
    hello = proto.recv(timeout=hello_timeout)
    if hello.get("type") != "hello" or hello.get("role") != "client":
        conn.close()
        raise ConnectionError(f"意外握手: {hello}")
    ver = int(hello.get("version", 0))
    if ver != PROTOCOL_VERSION:
        log.warning("协议版本不一致: client=%s server=%s", ver, PROTOCOL_VERSION)
    log.info("握手成功 version=%s", ver)
    return conn, proto, addr


def wait_client(host: str, port: int, timeout: float) -> tuple[socket.socket, LineProtocol]:
    """CLI 兼容：监听 → 接受一个客户端 → 关闭监听套接字。"""
    srv = create_listen_socket(host, port)
    try:
        conn, proto, _addr = accept_client(srv, timeout=timeout)
    finally:
        try:
            srv.close()
        except OSError:
            pass
    return conn, proto


def run_session(
    cfg: ExperimentConfig,
    session_id: Optional[str] = None,
    *,
    conn: Optional[socket.socket] = None,
    proto: Optional[LineProtocol] = None,
    camera=None,
    stop_event: Optional["object"] = None,
    keep_camera: bool = False,
    close_conn: bool = True,
    on_status=None,
) -> int:
    """
    跑一轮拍摄会话。

    Parameters
    ----------
    conn, proto
        若已由 GUI 常驻监听 accept，可直接传入，跳过 wait_client。
    camera
        已连接的相机实例；传入则跳过 create/connect。
    stop_event
        threading.Event；set 后优雅结束当前会话（不断端口）。
    keep_camera
        True 时 finally 不关闭相机（供多会话复用）。
    close_conn
        True 时 finally 关闭客户端连接。
    on_status
        可选回调 ``fn(str)``，用于 GUI 状态栏。
    """
    def status(msg: str) -> None:
        log.info("%s", msg)
        if on_status is not None:
            try:
                on_status(msg)
            except Exception:
                pass

    paths = cfg.paths
    cd = cfg.cooldowns
    image_dir = resolve_path(paths.image_dir)
    capture_root = resolve_path(paths.capture_root)
    session_root = resolve_path(paths.session_root)
    log_dir = resolve_path(paths.log_dir)
    # 写回绝对路径，便于日志阅读
    paths.image_dir = str(image_dir)
    paths.capture_root = str(capture_root)
    paths.session_root = str(session_root)
    paths.log_dir = str(log_dir)

    images = list_images(image_dir)
    n_all = len(images)
    start = max(0, int(cfg.start_index))
    end = n_all if cfg.end_index is None else min(int(cfg.end_index), n_all)
    if start >= end:
        raise SystemExit(f"空范围 start={start} end={end} total={n_all}")

    session_id = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    setup_logging(log_dir, session_id)
    state = SessionState.load_or_create(session_root, session_id)

    if cfg.resume and state.data.get("next_index", 0) > start:
        start = int(state.data["next_index"])
        log.info("断点续跑 → 从 index=%d", start)
    if start >= end:
        log.info("本会话已完成，无需继续")
        return 0

    todo = end - start
    hours = estimate_hours(todo, cd)
    free_gb = free_space_gb(capture_root)
    # 粗估：每张 JPEG 实拍约 8–25MB（R7），按 20MB 警告
    need_gb = todo * 20 / 1024
    log.info("=" * 56)
    log.info("会话 %s | 图像 %d 张 | 计划拍 [%d, %d) 共 %d", session_id, n_all, start, end, todo)
    log.info("预计耗时 ≈ %.1f h | 磁盘剩余 %.1f GB | 粗估需求 ~%.1f GB (按20MB/张)", hours, free_gb, need_gb)
    log.info("相机后端: %s | 分片: %d/目录", cfg.camera_backend, paths.shard_size)
    log.info("图库: %s", image_dir)
    log.info("回传: %s", capture_root)
    log.info("=" * 56)
    if free_gb < need_gb * 0.5:
        log.warning("磁盘空间可能不足，建议清理或改 capture_root 到大容量盘")

    # 会话目录落一份配置快照（不覆盖默认 config.json）
    snap = session_root / f"{session_id}.config.json"
    try:
        with open(snap, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
        log.info("配置快照: %s", snap)
    except OSError as e:
        log.warning("无法写入配置快照: %s", e)

    own_conn = conn is None
    if proto is None or conn is None:
        conn, proto = wait_client(cfg.net.host, cfg.net.port, cfg.net.connect_timeout_s)
        own_conn = True

    own_camera = camera is None
    if camera is None:
        cam_kwargs = {}
        if cfg.camera_backend.lower() in ("edsdk", "canon"):
            cam_kwargs["library_path"] = str(resolve_path(paths.edsdk_dll_dir))
            cam_kwargs["shutter_non_af"] = bool(getattr(cd, "shutter_non_af", True))
            cam_kwargs["shoot_retries"] = int(getattr(cd, "shoot_retries", 3))
            cam_kwargs["keep_liveview"] = bool(getattr(cd, "keep_liveview", False))
            cam_kwargs["post_download_pump_s"] = float(
                getattr(cd, "post_download_pump_s", 0.05)
            )
        camera = create_camera(cfg.camera_backend, **cam_kwargs)
        log.info(
            "相机参数: non_af=%s liveview=%s post_dl_pump=%.2fs",
            cam_kwargs.get("shutter_non_af"),
            cam_kwargs.get("keep_liveview"),
            cam_kwargs.get("post_download_pump_s", 0.05),
        )
    t0 = time.time()
    ok = 0
    fail = 0
    stopped = False

    try:
        if own_camera:
            camera.connect()
        for i in range(start, end):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                status(f"用户停止会话 @ index={i}")
                break

            img_path = images[i]
            name = img_path.name
            log.info("-" * 48)
            status(f"[{i - start + 1}/{todo}] SHOW {name}")

            # 1) 客户端翻页
            proto.send(cmd_show(i, name))
            if cd.after_show_s > 0:
                time.sleep(cd.after_show_s)

            # 电子快门 + NonAF：跳过每张测光/AF（省 ~0.2s/张）；机械/对焦模式才 AF
            if not getattr(cd, "shutter_non_af", True):
                try:
                    camera.autofocus()
                except Exception as e:
                    log.error("AF 失败: %s", e)
                    proto.send(cmd_abort(f"AF failed: {e}"))
                    raise
                if cd.after_af_s > 0:
                    time.sleep(cd.after_af_s)
            elif (i - start) % 50 == 0:
                # 偶发延长关机计时，避免大批量中途休眠
                try:
                    camera.autofocus()
                except Exception as e:
                    log.warning("保活失败（忽略）: %s", e)

            # 等 READY
            msg = proto.recv(timeout=cfg.net.recv_timeout_s)
            if msg.get("type") == "event" and msg.get("action") == "error":
                raise RuntimeError(f"客户端错误: {msg.get('message')}")
            if not (msg.get("type") == "event" and msg.get("action") == "ready"):
                raise RuntimeError(f"期望 READY，收到: {msg}")
            if int(msg.get("index", -1)) != i:
                log.warning("READY index 不匹配: got=%s expect=%s", msg.get("index"), i)
            log.info("收到 READY index=%s", msg.get("index"))

            if stop_event is not None and stop_event.is_set():
                stopped = True
                status(f"用户停止会话 @ index={i}（READY 后）")
                break

            if cd.after_ready_before_shoot_s > 0:
                time.sleep(cd.after_ready_before_shoot_s)

            # 2) 拍摄（shoot 内已等到 USB 落盘；电子快门后 after_shoot 可极短）
            dest = capture_path_for_index(
                capture_root,
                session_id,
                i,
                paths.shard_size,
            )
            if cfg.dry_run:
                log.info("[dry_run] skip shoot → %s", dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"\xff\xd8\xff\xd9")
            else:
                shoot_ok = False
                last_shoot_err: Optional[Exception] = None
                retries = max(1, int(getattr(cd, "shoot_retries", 3)))
                for attempt in range(1, retries + 1):
                    try:
                        camera.shoot(dest)
                        shoot_ok = True
                        break
                    except Exception as e:
                        last_shoot_err = e
                        log.error("SHOOT 失败 try=%d/%d: %s", attempt, retries, e)
                        time.sleep(0.35 * attempt)
                if not shoot_ok:
                    fail += 1
                    state.data.setdefault("failed", []).append(
                        {"index": i, "error": str(last_shoot_err)}
                    )
                    state.data["next_index"] = i + 1  # 跳过本张，便于续跑下一张
                    state.save()
                    if getattr(cd, "skip_on_shoot_fail", True):
                        log.warning("跳过 index=%d，继续下一张", i)
                        # 仍发 ACK，避免客户端红闪卡死
                        proto.send(cmd_ack())
                        if cd.after_ack_before_next_s > 0:
                            time.sleep(cd.after_ack_before_next_s)
                        continue
                    proto.send(cmd_abort(f"shoot failed: {last_shoot_err}"))
                    raise last_shoot_err

            if cd.after_shoot_s > 0:
                time.sleep(cd.after_shoot_s)

            # 3) ACK(1)
            proto.send(cmd_ack())
            log.info("已发送 ACK(1)")
            if cd.after_ack_before_next_s > 0:
                time.sleep(cd.after_ack_before_next_s)

            ok += 1
            state.data["next_index"] = i + 1
            state.data["completed"] = int(state.data.get("completed", 0)) + 1
            if (i + 1) % max(cd.checkpoint_every, 1) == 0:
                state.save()
                elapsed = time.time() - t0
                rate = ok / max(elapsed, 1e-6)
                remain = (todo - ok) / max(rate, 1e-6)
                log.info(
                    "检查点 @%d | 成功 %d | 速率 %.2f 张/s | ETA %.1f min | 磁盘 %.1f GB",
                    i + 1,
                    ok,
                    rate,
                    remain / 60.0,
                    free_space_gb(capture_root),
                )

            if cd.batch_rest_every > 0 and (i + 1) % cd.batch_rest_every == 0 and (i + 1) < end:
                log.info("批次休息 %.1f s（防过热/写卡堆积）…", cd.batch_rest_s)
                time.sleep(cd.batch_rest_s)

        state.save()
        if stopped:
            try:
                proto.send(cmd_abort("stopped by user"))
            except Exception:
                pass
            status(f"会话已停止: 成功 {ok} 失败 {fail}")
            return 130
        proto.send(cmd_finish())
        status(f"全部完成: 成功 {ok} 失败 {fail} 用时 {(time.time() - t0) / 60.0:.1f} min")
        return 0
    except KeyboardInterrupt:
        log.warning("用户中断；已保存断点 next_index=%s", state.data.get("next_index"))
        state.save()
        try:
            proto.send(cmd_abort("interrupted"))
        except Exception:
            pass
        return 130
    except Exception:
        log.exception("会话异常终止")
        state.save()
        return 1
    finally:
        if own_camera and not keep_camera:
            try:
                camera.close()
            except Exception:
                pass
        if close_conn and conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def list_resumable_sessions(session_root: Path, limit: int = 8) -> List[dict]:
    """扫描未完成会话（next_index > 0 且文件存在）。"""
    session_root = Path(session_root)
    if not session_root.is_dir():
        return []
    items = []
    for p in session_root.glob("session_*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            next_i = int(data.get("next_index", 0))
            completed = int(data.get("completed", 0))
            if next_i <= 0 and completed <= 0:
                continue
            items.append(
                {
                    "id": p.stem,
                    "path": p,
                    "next_index": next_i,
                    "completed": completed,
                    "mtime": p.stat().st_mtime,
                    "updated_at": data.get("updated_at"),
                }
            )
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            continue
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:limit]


def prompt_continue_or_fresh(
    session_root: Path,
    *,
    default_fresh: bool = False,
) -> tuple[bool, Optional[str]]:
    """
    交互询问：继续测试 or 从头测试。

    Returns
    -------
    (resume, session_id)
        resume=False → 新会话从头；resume=True → 续跑给定 session_id
    """
    sessions = list_resumable_sessions(session_root)
    print("\n" + "=" * 52, flush=True)
    print("  屏摄实验 — 启动方式", flush=True)
    print("=" * 52, flush=True)
    if sessions:
        print("  可续跑的会话：", flush=True)
        for i, s in enumerate(sessions, 1):
            print(
                f"    [{i}] {s['id']}  next={s['next_index']}  "
                f"completed={s['completed']}  "
                f"updated={s.get('updated_at') or '-'}",
                flush=True,
            )
    else:
        print("  （暂无未完成会话）", flush=True)
    print("-" * 52, flush=True)
    print("  1  继续测试  — 从断点续拍（选最近会话或输入编号）", flush=True)
    print("  2  从头测试  — 新建会话，从 index=0 开始", flush=True)
    print("-" * 52, flush=True)

    default = "2" if default_fresh or not sessions else "1"
    while True:
        try:
            raw = input(f"请选择 1 / 2 [默认 {default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            raw = default
        if raw == "":
            raw = default
        if raw == "2":
            print("[ OK ] 从头测试：将创建新会话", flush=True)
            return False, None
        if raw == "1":
            if not sessions:
                print("[WARN] 没有可续跑会话，改为从头测试", flush=True)
                return False, None
            if len(sessions) == 1:
                sid = sessions[0]["id"]
                print(
                    f"[ OK ] 继续测试：{sid}（从 index={sessions[0]['next_index']}）",
                    flush=True,
                )
                return True, sid
            # 多会话：再问编号，默认 1（最新）
            while True:
                try:
                    pick = input(
                        f"选择会话编号 1–{len(sessions)} [默认 1]: "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    print(flush=True)
                    pick = "1"
                if pick == "":
                    pick = "1"
                if pick.isdigit() and 1 <= int(pick) <= len(sessions):
                    s = sessions[int(pick) - 1]
                    print(
                        f"[ OK ] 继续测试：{s['id']}（从 index={s['next_index']}）",
                        flush=True,
                    )
                    return True, s["id"]
                print("无效输入，请重试", flush=True)
        print("无效输入，请输入 1 或 2", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="屏摄实验服务器（CLI）")
    p.add_argument("--config", type=Path, default=None, help="配置 JSON")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--image_dir", type=Path, default=None)
    p.add_argument("--capture_root", type=Path, default=None)
    p.add_argument("--session", default=None, help="指定会话 ID（续跑时用）")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--camera", default=None, choices=["stub", "edsdk"])
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="非交互：续跑（需 --session，或自动用最近会话）",
    )
    p.add_argument(
        "--fresh",
        "--no_resume",
        dest="fresh",
        action="store_true",
        help="非交互：强制从头新建会话",
    )
    p.add_argument(
        "--no_interactive",
        action="store_true",
        help="跳过启动询问（配合 --resume / --fresh）",
    )
    p.add_argument("--write_config", action="store_true", help="写出默认 config.json 后退出")
    return p


def main() -> int:
    if os.name == "nt":
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
    if args.start is not None:
        cfg.start_index = args.start
    if args.end is not None:
        cfg.end_index = args.end
    if args.camera:
        cfg.camera_backend = args.camera
    if args.dry_run:
        cfg.dry_run = True

    if args.write_config:
        path = save_config(cfg)
        print(f"已写入 {path}")
        return 0

    session_id = args.session
    session_root = resolve_path(cfg.paths.session_root)

    # 启动方式：交互询问 / CLI 指定
    if args.fresh:
        cfg.resume = False
        session_id = None
        print("[INFO] --fresh：从头测试", flush=True)
    elif args.resume or (args.no_interactive and cfg.resume):
        cfg.resume = True
        if session_id is None:
            sessions = list_resumable_sessions(session_root, limit=1)
            if not sessions:
                print("[WARN] --resume 但无断点，改为从头", flush=True)
                cfg.resume = False
            else:
                session_id = sessions[0]["id"]
                print(f"[INFO] --resume：{session_id}", flush=True)
    elif args.no_interactive:
        # 非交互且未指定：默认从头，避免误续
        cfg.resume = False
        session_id = None
    else:
        resume, sid = prompt_continue_or_fresh(session_root)
        cfg.resume = resume
        if resume:
            session_id = sid
        else:
            session_id = None
            cfg.start_index = 0 if args.start is None else cfg.start_index

    return run_session(cfg, session_id=session_id)


if __name__ == "__main__":
    raise SystemExit(main())
