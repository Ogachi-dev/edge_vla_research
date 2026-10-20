"""同期・非同期の実行器（仮想時間方式）。docs/track1/27_closed_loop_b_c.md 4

考え方：
- 環境の1ステップ = dt = 1 / control_hz 秒。
- 推論の遅延 L[ms] の間に、環境は k = ceil(L / dt) ステップ進む（lockstep なら k = 0）。
- 推論は実際には「開始」の時点で同期的に計算してしまい、結果を k ステップ後まで保留する。
  スレッドを使わないので、同じ乱数シードなら結果が完全に再現する。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from evla.policy.base import ChunkResult, Policy
from evla.sim.latency import LatencyModel
from evla.sim.types import Observation, StepResult


class StepEnv(Protocol):
    def step(self, action: npt.NDArray[np.float32]) -> StepResult: ...


@dataclass(frozen=True)
class ExecutorConfig:
    mode: Literal["lockstep", "sync", "async"]
    n_exec: int = 10                  # 1チャンクから実行する行動数（実行ホライズン）
    async_threshold: int = 5          # 非同期：キューの残りがこれ以下になったら次を推論する
    hold_action: Literal["zero", "last"] = "zero"   # 行動がないときに送るもの
    control_hz: float = 20.0
    gripper_dim: int = -1             # "zero" でもグリッパだけは直前の値を保つ（開閉が勝手に変わらないように）
    noise_dim: int | None = 32        # 指定すると rng から初期ノイズを作って渡す（再現性のため）。None なら方策任せ


@dataclass
class EpisodeTrace:
    steps: int = 0
    n_inferences: int = 0
    hold_steps: int = 0               # 行動がなく保持したステップ数
    discarded_actions: int = 0        # 非同期で、到着時に古くなって捨てた行動の数
    latencies_ms: list[float] = field(default_factory=list)       # 実測
    sim_latencies_ms: list[float] = field(default_factory=list)   # 遅延モデルが返した値
    infer_steps: list[int] = field(default_factory=list)          # 推論を開始したステップ


class _EpisodeEnd(Exception):
    def __init__(self, success: bool) -> None:
        self.success = success


class Executor(ABC):
    def __init__(self, cfg: ExecutorConfig, policy: Policy, latency: LatencyModel,
                 rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.policy = policy
        self.latency = latency
        self.rng = rng
        self.dt_ms = 1000.0 / cfg.control_hz
        self._last_action: npt.NDArray[np.float32] | None = None

    @abstractmethod
    def run(self, env: StepEnv, first_obs: Observation, max_steps: int) -> tuple[bool, EpisodeTrace]:
        """エピソードを最後まで回し、(成功したか, 記録) を返す。"""

    # ---- 共通の部品 ----
    def _infer(self, obs: Observation, t: int, trace: EpisodeTrace) -> tuple[ChunkResult, int]:
        """推論して、(結果, 経過ステップ数 k) を返す。"""
        noise = None
        if self.cfg.noise_dim is not None:
            noise = self.rng.standard_normal((self._chunk_len(), self.cfg.noise_dim)).astype(np.float32)
        r = self.policy.predict_chunk(obs, noise)
        sim_ms = self.latency.sample(r.latency_ms, self.rng)
        k = 0 if self.cfg.mode == "lockstep" else math.ceil(sim_ms / self.dt_ms)  # 【穴①】遅延の間に進むステップ数（切り上げ）
        trace.n_inferences += 1
        trace.latencies_ms.append(r.latency_ms)
        trace.sim_latencies_ms.append(sim_ms)
        trace.infer_steps.append(t)
        return r, k

    def _chunk_len(self) -> int:
        try:
            return self.policy.chunk_size
        except RuntimeError:  # OV / vlacpp は1回推論するまで分からない
            return 50

    def _hold(self, action_dim: int) -> npt.NDArray[np.float32]:
        if self._last_action is None:
            return np.zeros(action_dim, dtype=np.float32)
        if self.cfg.hold_action == "last":
            return self._last_action
        a = np.zeros_like(self._last_action)
        a[self.cfg.gripper_dim] = self._last_action[self.cfg.gripper_dim]
        return a

    def _step(self, env: StepEnv, action: npt.NDArray[np.float32], trace: EpisodeTrace,
              max_steps: int, hold: bool = False) -> Observation:
        """1ステップ進める。終わったら _EpisodeEnd を投げる（ループを抜けるのを簡単にするため）。"""
        res = env.step(action)
        trace.steps += 1
        if hold:
            trace.hold_steps += 1
        else:
            self._last_action = action
        if res.done:
            raise _EpisodeEnd(res.success)
        if trace.steps >= max_steps:
            raise _EpisodeEnd(False)
        return res.obs


class SyncExecutor(Executor):
    """推論中はロボットを止め（hold を k 回送り）、チャンクの先頭 n_exec 個を実行する。lockstep は k=0。"""

    def run(self, env: StepEnv, first_obs: Observation, max_steps: int) -> tuple[bool, EpisodeTrace]:
        trace = EpisodeTrace()
        obs = first_obs
        try:
            while True:
                r, k = self._infer(obs, trace.steps, trace)
                for _ in range(k):  # 【穴②】推論中の k ステップぶん
                    obs = self._step(env, self._hold(r.actions.shape[1]), trace, max_steps, hold=True)
                for a in r.actions[: self.cfg.n_exec]:  # 【穴③】チャンクの先頭 n_exec 個
                    obs = self._step(env, a, trace, max_steps)
        except _EpisodeEnd as end:
            return end.success, trace


class AsyncExecutor(Executor):
    """キューの行動を実行しながら次を推論する。到着したチャンクの先頭 k 個は過去の分なので捨てる。"""

    def run(self, env: StepEnv, first_obs: Observation, max_steps: int) -> tuple[bool, EpisodeTrace]:
        trace = EpisodeTrace()
        obs = first_obs
        cfg = self.cfg
        try:
            # 最初の1回は同期：ロボットは止まっているので、到着後もチャンクの先頭から使ってよい
            r, k = self._infer(obs, 0, trace)
            action_dim = r.actions.shape[1]
            for _ in range(k):
                obs = self._step(env, self._hold(action_dim), trace, max_steps, hold=True)
            queue: deque[npt.NDArray[np.float32]] = deque(r.actions[: cfg.n_exec])
            pending: tuple[int, npt.NDArray[np.float32], int] | None = None  # (到着ステップ, チャンク, k)

            while True:
                t = trace.steps
                if pending is None and len(queue) <= cfg.async_threshold:  # 【穴④】次の推論を始める条件（キューの残り）
                    r, k = self._infer(obs, t, trace)
                    pending = (t + k, r.actions, k)  # 【穴⑤】到着予定のステップ
                if pending is not None and t >= pending[0]:
                    _, chunk, k = pending
                    queue = deque(chunk[k : k + cfg.n_exec])  # 【穴⑥】古い先頭 k 個を飛ばした n_exec 個
                    trace.discarded_actions += min(k, len(chunk))
                    pending = None
                if queue:
                    obs = self._step(env, queue.popleft(), trace, max_steps)  # 【穴⑦】キューの先頭を取り出す deque のメソッド
                else:
                    obs = self._step(env, self._hold(action_dim), trace, max_steps, hold=True)
        except _EpisodeEnd as end:
            return end.success, trace


def make_executor(cfg: ExecutorConfig, policy: Policy, latency: LatencyModel,
                  rng: np.random.Generator) -> Executor:
    cls = AsyncExecutor if cfg.mode == "async" else SyncExecutor
    return cls(cfg, policy, latency, rng)
