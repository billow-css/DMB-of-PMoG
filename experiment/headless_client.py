#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无 GUI 的假客户端，用于本机冒烟测试 server 状态机。"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import LineProtocol, event_ready, hello_client  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    sock = socket.create_connection((args.server, args.port), timeout=10)
    proto = LineProtocol(sock)
    proto.send(hello_client())
    print("headless client connected", flush=True)
    while True:
        msg = proto.recv(timeout=60)
        print("<<", msg, flush=True)
        if msg.get("type") == "cmd" and msg.get("action") == "show":
            time.sleep(0.05)
            proto.send(event_ready(int(msg["index"])))
            print(">> ready", msg["index"], flush=True)
        elif msg.get("type") == "cmd" and msg.get("action") in ("finish", "abort"):
            break
    sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
