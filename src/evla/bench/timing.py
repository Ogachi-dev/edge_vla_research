"""計時と、レイテンシの要約統計。docs/track1/26_bench_mode_a.md"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

MIN_N_FOR_P99 = 100


def summarize_latencies(ms: npt.ArrayLike) -> dict[str, float]:
    """{"n", "mean", "std", "min", "p50", "p90", "p95", "p99", "max", "cv"} を返す。
    n < 100 のときは p99 を NaN にする（上位1%に1点も入らない＝最大値とほぼ同じになり、信頼できないため）。"""
    x = np.asarray(ms, dtype=np.float64)
    if x.size == 0:
        raise ValueError("サンプルが空")
    p50, p90, p95, p99 = np.percentile(x, [50, 90, 95, 99])  # 【穴①】パーセンタイルをまとめて計算する numpy の関数
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0  # 【穴②】標本標準偏差（n-1 で割る）にする引数
    return {
        "n": float(x.size),
        "mean": mean,
        "std": std,
        "min": float(x.min()),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99) if x.size >= MIN_N_FOR_P99 else float("nan"),  # 【穴③】p99 を出してよい条件
        "max": float(x.max()),
        "cv": std / mean if mean > 0 else float("nan"),  # 【穴④】変動係数（ばらつき ÷ 平均）
    }


class Stopwatch:
    """with Stopwatch() as sw: ... ; sw.ms で経過ミリ秒（perf_counter_ns ベース）。"""

    def __init__(self) -> None:
        self._t0 = 0
        self._t1: int | None = None

    def __enter__(self) -> Stopwatch:
        self._t0 = time.perf_counter_ns()  # 【穴⑤】単調増加でナノ秒の整数を返す時計（壁時計の time.time は使わない）
        self._t1 = None
        return self

    def __exit__(self, *exc: object) -> None:
        self._t1 = time.perf_counter_ns()

    @property
    def ms(self) -> float:
        end = self._t1 if self._t1 is not None else time.perf_counter_ns()  # with の中で読んだら途中経過
        return (end - self._t0) / 1e6  # 【穴⑥】ns → ms
