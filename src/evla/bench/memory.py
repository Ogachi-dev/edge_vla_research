"""ピークRSSの監視。docs/track1/26_bench_mode_a.md"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path


def read_status_kb(pid: int, key: str) -> int | None:
    """/proc/<pid>/status の 'VmRSS:   12345 kB' のような行から kB の整数を読む。プロセスが無ければ None。"""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])  # 【穴①】空白で区切った2番目の要素を整数に
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


class PeakRssMonitor:
    """with PeakRssMonitor(pids=[os.getpid(), server_pid], interval_s=0.05) as m: ...
    m.peak_mb[pid] で各プロセスのピークRSS。終了時に /proc/<pid>/status の VmHWM とも照合する。"""

    def __init__(self, pids: Sequence[int], interval_s: float = 0.05) -> None:
        self.pids = list(pids)
        self.interval_s = interval_s
        self.peak_mb: dict[int, float] = {p: 0.0 for p in self.pids}
        self.hwm_mb: dict[int, float | None] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        for pid in self.pids:
            kb = read_status_kb(pid, "VmRSS")  # 【穴②】今の常駐メモリを表すキー
            if kb is not None:
                self.peak_mb[pid] = max(self.peak_mb[pid], kb / 1024)  # 【穴③】これまでの最大と比べて大きい方（MB）

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):  # 【穴④】interval_s 秒待つ。途中で止められたら True が返る
            self._sample()

    def __enter__(self) -> PeakRssMonitor:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
        for pid in self.pids:
            kb = read_status_kb(pid, "VmHWM")  # カーネルが記録しているピーク（取りこぼしの確認用）
            self.hwm_mb[pid] = None if kb is None else kb / 1024
