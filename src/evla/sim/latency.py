"""遅延モデル（仮想時間方式で「推論にかかったとみなす時間」を決める）。docs/track1/27_closed_loop_b_c.md 4.2"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class LatencyModel(Protocol):
    def sample(self, measured_ms: float, rng: np.random.Generator) -> float:
        """今回の推論について、シミュレーション上で経過したとみなす遅延[ms]を返す。"""
        ...


@dataclass(frozen=True)
class MeasuredLatency:
    """実測値をそのまま使う（scale で端末差を模擬してもよい）。"""

    scale: float = 1.0
    add_ms: float = 0.0

    def sample(self, measured_ms: float, rng: np.random.Generator) -> float:
        return measured_ms * self.scale + self.add_ms  # 【穴①】実測値を scale 倍して add_ms を足す


@dataclass(frozen=True)
class ConstantLatency:
    """実測値を無視して一定値（遅延許容曲線の主役）。"""

    ms: float

    def sample(self, measured_ms: float, rng: np.random.Generator) -> float:
        return self.ms


@dataclass(frozen=True)
class EmpiricalLatency:
    """モードAの samples.parquet から分布をリサンプリングする（ジッタも再現）。"""

    samples_ms: tuple[float, ...]

    def sample(self, measured_ms: float, rng: np.random.Generator) -> float:
        return float(rng.choice(self.samples_ms))  # 【穴②】配列から1つ無作為に選ぶ Generator のメソッド

    @classmethod
    def from_parquet(cls, path: Path) -> EmpiricalLatency:
        import pandas as pd

        df = pd.read_parquet(path)
        return cls(tuple(df.loc[~df["is_warmup"], "latency_ms"].astype(float)))


def make_latency(spec: Mapping[str, Any]) -> LatencyModel:
    """{"type": "constant", "ms": 500} / {"type": "measured", "scale": 2.0} / {"type": "empirical", "path": ...}"""
    kind = spec["type"]
    args = {k: v for k, v in spec.items() if k != "type"}
    if kind == "constant":
        return ConstantLatency(**args)
    if kind == "measured":
        return MeasuredLatency(**args)
    if kind == "empirical":
        return EmpiricalLatency.from_parquet(Path(args["path"]))
    raise ValueError(f"未知の遅延モデル: {kind}")
