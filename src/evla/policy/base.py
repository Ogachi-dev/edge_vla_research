"""ランタイム共通の方策インタフェース。docs/track1/22_policy_interface.md 4"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from evla.sim.types import Observation

Runtime = Literal["torch", "ov", "vlacpp"]


@dataclass(frozen=True)
class PolicyConfig:
    runtime: Runtime
    model: str                      # HFのrepo_id、またはローカルパス（ランタイムごとに意味が違う）
    device: str = "CPU"             # "CPU" | "GPU"（ov）、vlacppでは情報用（実際の指定はサーバー側）
    precision: str = "fp32"         # "fp32" | "fp16" | "bf16" | "int8" | "q8_0" | "q4_0"
    views: tuple[str, ...] = ("image", "image2")   # 使う視点。1視点なら ("image",)
    image_size: int = 512           # モデル内部の入力解像度（SmolVLAの学習時は512）
    num_steps: int = 10             # flow matching のステップ数
    n_threads: int | None = None    # CPUスレッド数。None は各ランタイムの既定
    server_addr: str = "tcp://127.0.0.1:5555"      # vlacpp のみ
    extra: Mapping[str, Any] = field(default_factory=dict)   # ランタイム固有の設定


@dataclass(frozen=True)
class ChunkResult:
    actions: npt.NDArray[np.float32]     # (chunk_size, action_dim)。逆正規化済み（環境にそのまま渡せる値）
    latency_ms: float                    # predict_chunk の呼び出し全体（前処理・通信・後処理を含む）
    breakdown_ms: Mapping[str, float] = field(default_factory=dict)


class Policy(ABC):
    """状態を持たない方策。キュー（何行動を実行するか、いつ次を推論するか）は executor 側で管理する。"""

    cfg: PolicyConfig

    @property
    @abstractmethod
    def chunk_size(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @abstractmethod
    def predict_chunk(self, obs: Observation, noise: npt.NDArray[np.float32] | None = None) -> ChunkResult:
        """観測から行動チャンクを1つ推論する。
        noise: (chunk_size, max_action_dim) の初期ノイズ。Noneなら内部で乱数を使う。"""

    def warmup(self, obs: Observation, n: int = 3) -> list[float]:
        """初回コンパイルやキャッシュ構築を済ませる。各回のレイテンシ[ms]を返す（1回目は別に記録する）。"""
        return [self.predict_chunk(obs).latency_ms for _ in range(n)]

    def close(self) -> None:  # noqa: B027  何もしない既定の実装（必要なクラスだけ上書きする）
        """サーバー接続などの資源を解放する。"""

    def describe(self) -> dict[str, Any]:
        """結果ファイルに残すための情報（実際のデバイス名、ライブラリ版、スレッド数など）。"""
        return {"runtime": self.cfg.runtime, "model": self.cfg.model}

    def __enter__(self) -> Policy:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
