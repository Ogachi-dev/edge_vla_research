"""時刻つきメッセージの小さなリングバッファ（ROS 非依存）。docs/track2/34_episode_recorder.md 5

ロックステップの記録で「sim 時刻 t のフレーム」を各トピックから取り出すのに使う。
時刻は誤差を避けるため整数のナノ秒で扱う。
"""

from __future__ import annotations

import bisect
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")  # ROS 側の Python 3.10 でも動くよう、3.12 の class C[T] は使わない


class StampedBuffer(Generic[T]):
    def __init__(self, maxlen: int = 128) -> None:
        self.maxlen = maxlen
        self._stamps: list[int] = []
        self._items: list[T] = []
        self._lock = threading.Lock()  # コールバックのスレッドと、取り出すスレッドが別なので排他する

    def add(self, stamp_ns: int, item: T) -> None:
        with self._lock:
            i = bisect.bisect_right(self._stamps, stamp_ns)  # 【穴①】ソート済みリストのどこに入れれば順序が保てるか（二分探索）
            self._stamps.insert(i, stamp_ns)
            self._items.insert(i, item)
            if len(self._stamps) > self.maxlen:
                del self._stamps[0], self._items[0]  # 古いものから捨てる

    def get(self, t_ns: int, tol_ns: int) -> T | None:
        """|stamp - t| <= tol のうち最も近いもの。無ければ None。"""
        with self._lock:
            if not self._stamps:
                return None
            i = bisect.bisect_left(self._stamps, t_ns)
            best = min((j for j in (i - 1, i) if 0 <= j < len(self._stamps)),
                       key=lambda j: abs(self._stamps[j] - t_ns))  # 【穴②】目的の時刻との差の大きさ
            return self._items[best] if abs(self._stamps[best] - t_ns) <= tol_ns else None

    def latest_at_or_before(self, t_ns: int) -> T | None:
        """stamp <= t の中で最新のもの（発行周期がフレームと揃わないトピック用）。"""
        with self._lock:
            i = bisect.bisect_right(self._stamps, t_ns)
            return self._items[i - 1] if i > 0 else None  # 【穴③】t 以下で一番後ろの要素（無ければ None）

    def newest_stamp(self) -> int | None:
        with self._lock:
            return self._stamps[-1] if self._stamps else None


def wait_until(pred: Callable[[], bool], timeout_s: float, poll_s: float = 0.001) -> bool:
    """pred() が真になるまで待つ（ROS の executor は別スレッドで回っている前提）。"""
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(poll_s)
    return pred()
