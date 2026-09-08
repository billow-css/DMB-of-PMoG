#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测 EDSDK + 相机：连接 → AF → 拍一张 → 保存到 experiment/captures/_probe/"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera import create_camera
from config import ROOT, load_config
from edsdk_raw import default_edsdk_dir


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(description="EDSDK 相机探测")
    p.add_argument("--dll_dir", type=Path, default=None)
    p.add_argument("--no_shoot", action="store_true", help="只连接，不拍照")
    args = p.parse_args()

    cfg = load_config()
    dll = args.dll_dir
    if dll is None:
        from config import REPO_ROOT

        raw = Path(cfg.paths.edsdk_dll_dir)
        dll = raw if raw.is_absolute() else (REPO_ROOT / raw)
    dll = Path(dll).resolve()
    if not dll.is_dir():
        dll = default_edsdk_dir()

    print(f"Dll dir: {dll}")
    cam = create_camera("edsdk", library_path=str(dll))
    try:
        cam.connect()
        print(f"OK model={cam.model_name}")
        cam.autofocus()
        if not args.no_shoot:
            out_dir = ROOT / "captures" / "_probe"
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"probe_{datetime.now().strftime('%H%M%S')}.jpg"
            saved = cam.shoot(dest)
            print(f"SHOT → {saved}  size={saved.stat().st_size if saved.is_file() else 0}")
        return 0
    except Exception as e:
        logging.exception("探测失败: %s", e)
        return 1
    finally:
        cam.close()


if __name__ == "__main__":
    raise SystemExit(main())
