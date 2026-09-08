#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/protocol.py
======================
TCP 行协议（每行一个 JSON，UTF-8）。

工作流报文
----------
Server → Client
  {"type":"cmd","action":"show","index":0,"name":"0.png"}
  {"type":"cmd","action":"ack"}          # 用户所说的「1」：本张拍摄完成，可结束红闪
  {"type":"cmd","action":"finish"}
  {"type":"cmd","action":"abort","reason":"..."}
  {"type":"ping"}

Client → Server
  {"type":"hello","role":"client","version":1}
  {"type":"event","action":"ready","index":0}
  {"type":"event","action":"error","index":0,"message":"..."}
  {"type":"pong"}
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, Iterator, Optional


PROTOCOL_VERSION = 1
ENCODING = "utf-8"


def encode_msg(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        ENCODING
    )


def decode_line(line: bytes | str) -> Dict[str, Any]:
    if isinstance(line, bytes):
        line = line.decode(ENCODING)
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("message must be a JSON object")
    return obj


class LineProtocol:
    """缓冲式按行收发。"""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = bytearray()

    def send(self, obj: Dict[str, Any]) -> None:
        self.sock.sendall(encode_msg(obj))

    def recv(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if timeout is not None:
            self.sock.settimeout(timeout)
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line = bytes(self._buf[:nl])
                del self._buf[: nl + 1]
                return decode_line(line)
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("peer closed connection")
            self._buf.extend(chunk)

    def iter_messages(self, timeout: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        while True:
            yield self.recv(timeout=timeout)


def cmd_show(index: int, name: str) -> Dict[str, Any]:
    return {"type": "cmd", "action": "show", "index": int(index), "name": str(name)}


def cmd_ack() -> Dict[str, Any]:
    """拍摄完成确认（协议中的「1」）。"""
    return {"type": "cmd", "action": "ack", "code": 1}


def cmd_finish() -> Dict[str, Any]:
    return {"type": "cmd", "action": "finish"}


def cmd_abort(reason: str) -> Dict[str, Any]:
    return {"type": "cmd", "action": "abort", "reason": str(reason)}


def event_ready(index: int) -> Dict[str, Any]:
    return {"type": "event", "action": "ready", "index": int(index)}


def event_error(index: int, message: str) -> Dict[str, Any]:
    return {
        "type": "event",
        "action": "error",
        "index": int(index),
        "message": str(message),
    }


def hello_client(version: int = PROTOCOL_VERSION) -> Dict[str, Any]:
    return {"type": "hello", "role": "client", "version": int(version)}
