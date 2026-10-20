"""閉ループ評価（モードB / C）。docs/track1/27_closed_loop_b_c.md 4.2"""

from __future__ import annotations

import time
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from evla.common.config import dump_config
from evla.common.sysinfo import save_env
from evla.policy.base import PolicyConfig
from evla.policy.factory import make_policy
from evla.sim.executor import ExecutorConfig, make_executor
from evla.sim.latency import make_latency
from evla.sim.libero_env import LiberoConfig, LiberoTaskEnv


@dataclass(frozen=True)
class ClosedLoopConfig:
    policy: PolicyConfig
    executor: ExecutorConfig
    latency: Mapping[str, Any] = field(default_factory=lambda: {"type": "measured"})
    suite: str = "libero_object"
    task_ids: tuple[int, ...] = tuple(range(10))
    episodes_per_task: int = 5          # init_state_index 0..4 を使う（全設定で共通の集合）
    seed: int = 0
    save_video: Literal["none", "failures", "all"] = "failures"


def episode_seed(seed: int, task_id: int, init_state: int) -> int:
    """(seed, task, init_state) から決まる乱数シード。Python の hash() は実行ごとに変わるので使わない。"""
    return zlib.crc32(f"{seed}-{task_id}-{init_state}".encode())  # 【穴①】バイト列から決定的な32bit整数を作る関数（zlib）


def run_eval(cfg: ClosedLoopConfig, run_dir: Path) -> pd.DataFrame:
    """1エピソード＝1行のDataFrameを返し、run_dir/episodes.parquet に保存する。"""
    dump_config(cfg, run_dir / "config.yaml")
    save_env(run_dir / "env.json")
    latency = make_latency(cfg.latency)
    rows = []
    with make_policy(cfg.policy) as policy:
        for task_id in cfg.task_ids:
            env = LiberoTaskEnv(LiberoConfig(suite=cfg.suite, task_id=task_id))
            try:
                for ep in range(cfg.episodes_per_task):
                    s = episode_seed(cfg.seed, task_id, ep)
                    rng = np.random.default_rng(s)
                    obs = env.reset(init_state_index=ep, seed=s)
                    ex = make_executor(cfg.executor, policy, latency, rng)
                    t0 = time.perf_counter()
                    success, tr = ex.run(env, obs, env.max_steps)  # 【穴②】1エピソードを最後まで回す executor のメソッド
                    rows.append({
                        "task_id": task_id, "init_state_index": ep, "success": success,
                        "steps": tr.steps, "n_inferences": tr.n_inferences, "hold_steps": tr.hold_steps,
                        "discarded_actions": tr.discarded_actions,
                        "lat_p50_ms": float(np.median(tr.latencies_ms)),
                        "sim_lat_p50_ms": float(np.median(tr.sim_latencies_ms)),
                        "wall_s": time.perf_counter() - t0,
                    })
                    print(f"task {task_id} ep {ep}: success={success} steps={tr.steps}", flush=True)
                    # TODO(you): save_video の設定に従って env.render_frame() を集めて mp4 にする（imageio.mimsave）
            finally:
                env.close()
    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "episodes.parquet", index=False)
    return df
