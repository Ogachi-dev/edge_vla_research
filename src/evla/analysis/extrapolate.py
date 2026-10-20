"""Jetson への外挿。docs/common/41_analysis_reporting.md 4.1"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DeviceLatency:
    device: str
    latency_ms: float
    source: Literal["measured", "published"]
    note: str = ""


def interpolate_success(curve: pd.DataFrame, latency_ms: float, n_exec: int, mode: str) -> tuple[float, float, float]:
    """遅延許容曲線（列 latency_ms, n_exec, mode, success_rate, ci_lo, ci_hi）を線形補間し、
    (成功率, 下限, 上限) を返す。曲線の範囲外なら外挿せず NaN（範囲外であることを明示）。"""
    c = curve[(curve["n_exec"] == n_exec) & (curve["mode"] == mode)].sort_values("latency_ms")  # 【穴①】2つの条件を両方満たす行（pandas の論理積は & と括弧）
    x = c["latency_ms"].to_numpy()
    if len(x) == 0 or not (x.min() <= latency_ms <= x.max()):  # 【穴②】曲線の測定範囲の内側
        nan = float("nan")
        return (nan, nan, nan)
    return tuple(float(np.interp(latency_ms, x, c[col].to_numpy())) for col in ("success_rate", "ci_lo", "ci_hi"))  # type: ignore[return-value]


def extrapolation_table(curve: pd.DataFrame, devices: Sequence[DeviceLatency], n_exec: int, mode: str) -> pd.DataFrame:
    rows = []
    for d in devices:
        s, lo, hi = interpolate_success(curve, d.latency_ms, n_exec, mode)
        rows.append({"device": d.device, "latency_ms": d.latency_ms, "source": d.source, "note": d.note,
                     "success_rate": s, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)
