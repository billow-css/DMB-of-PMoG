# -*- coding: utf-8 -*-
"""进度条与 ETA。"""

from __future__ import annotations

import sys
import time
from typing import Dict, Iterable, Optional

from cli.log import log_info

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False
    _tqdm = None


def format_eta(seconds: float) -> str:
    """将秒数格式化为可读 ETA（如 1h02m15s）。"""
    if seconds < 0 or seconds != seconds:  # NaN
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressMonitor:
    """有 tqdm 时用进度条，否则按间隔打印文本进度。"""

    def __init__(
        self,
        total: int,
        desc: str = "",
        enable: bool = True,
        unit: str = "batch",
        log_every: int = 20,
        file=None,
    ):
        self.total = max(int(total), 1)
        self.desc = desc
        self.enable = enable
        self.unit = unit
        self.log_every = max(int(log_every), 1)
        self.file = file
        self.n = 0
        self._t0 = time.time()
        self._metrics: Dict[str, float] = {}
        self._bar = None

        if enable and _HAS_TQDM:
            self._bar = _tqdm(
                total=self.total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
            )
        elif enable:
            log_info(f"{desc}  (0/{self.total})  — tqdm 未安装，使用文本进度")

    def update(self, n: int = 1, **metrics: float) -> None:
        self.n += n
        self._metrics.update({k: float(v) for k, v in metrics.items()})
        if self._bar is not None:
            self._bar.set_postfix(self._metrics, refresh=False)
            self._bar.update(n)
            return
        if self.enable and (self.n % self.log_every == 0 or self.n >= self.total):
            elapsed = time.time() - self._t0
            rate = self.n / elapsed if elapsed > 0 else 0.0
            remain = (self.total - self.n) / rate if rate > 0 else 0.0
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in self._metrics.items())
            line = (
                f"  [{self.desc}] {self.n}/{self.total} "
                f"({100.0 * self.n / self.total:.1f}%)  "
                f"ETA {format_eta(remain)}  {metric_str}"
            )
            print(line, flush=True)
            if self.file is not None:
                print(line, file=self.file, flush=True)

    def set_description(self, desc: str) -> None:
        self.desc = desc
        if self._bar is not None:
            self._bar.set_description(desc)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def wrap_iterable(
    iterable: Iterable,
    total: Optional[int] = None,
    desc: str = "",
    enable: bool = True,
    **kwargs,
) -> Iterable:
    """将可迭代对象包装为带进度条的迭代器；无 tqdm 时原样返回。"""
    if not enable:
        return iterable
    if _HAS_TQDM:
        kwargs.setdefault("leave", True)
        kwargs.setdefault("dynamic_ncols", True)
        return _tqdm(iterable, total=total, desc=desc, **kwargs)
    return iterable
