"""executor のテスト用の偽の方策と偽の環境（LIBERO なしで動く）。"""

from __future__ import annotations

import numpy as np

from evla.policy.base import ChunkResult, Policy, PolicyConfig
from evla.sim.types import Observation, StepResult

OBS = Observation(images={}, state=np.zeros(8, np.float32), task="fake")


class FakePolicy(Policy):
    """行動 j 番目 = [何番目の推論か + 1, j, 1.0（グリッパ）]。先頭が0の行動は hold とみなせる。"""

    def __init__(self, chunk: int = 50, latency_ms: float = 0.0) -> None:
        self.cfg = PolicyConfig(runtime="torch", model="fake")
        self.n = 0
        self.chunk = chunk
        self.latency_ms = latency_ms
        self.seen_steps: list[int] = []

    @property
    def chunk_size(self) -> int:
        return self.chunk

    @property
    def action_dim(self) -> int:
        return 3

    def predict_chunk(self, obs, noise=None):
        self.seen_steps.append(obs.sim_step)
        a = np.zeros((self.chunk, 3), np.float32)
        a[:, 0] = self.n + 1
        a[:, 1] = np.arange(self.chunk)
        a[:, 2] = 1.0
        self.n += 1
        return ChunkResult(actions=a, latency_ms=self.latency_ms)


class FakeEnv:
    """受け取った行動を全部記録する。done_at ステップで終わる。"""

    def __init__(self, done_at: int = 10_000, success: bool = True) -> None:
        self.actions: list[np.ndarray] = []
        self.done_at = done_at
        self.success = success

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        t = len(self.actions)
        obs = Observation(images={}, state=np.zeros(8, np.float32), task="fake", sim_step=t)
        done = t >= self.done_at
        return StepResult(obs=obs, success=done and self.success, done=done)

    def tags(self) -> list[tuple[int, int] | None]:
        """各ステップの行動を (推論番号, チャンク内の位置) に。hold は None。"""
        return [None if a[0] == 0 else (int(a[0]) - 1, int(a[1])) for a in self.actions]
