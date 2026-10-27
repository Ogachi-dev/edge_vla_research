"""閉ループ評価（モードB / C）。docs/track1/27_closed_loop_b_c.md 4.2"""

from __future__ import annotations

import time
import zlib
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from evla.common.config import dump_config
from evla.common.sysinfo import save_env
from evla.policy.base import PolicyConfig
from evla.policy.factory import make_policy
from evla.policy.vlacpp_server import VlaServerConfig, VlaServerProcess
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
    server: VlaServerConfig | None = None   # runtime == "vlacpp" のときに起動する（モードAと同じ）
    tag: str = ""                           # 表と図の凡例に使う名前（空ならランタイムなどから作る）


def episode_seed(seed: int, task_id: int, init_state: int) -> int:
    """(seed, task, init_state) から決まる乱数シード。Python の hash() は実行ごとに変わるので使わない。"""
    return zlib.crc32(f"{seed}-{task_id}-{init_state}".encode())  # 【穴①】バイト列から決定的な32bit整数を作る関数（zlib）


class _RecordingEnv:
    """executor から見ると env と同じ。step のたびに動画用のフレームを集める。"""

    def __init__(self, env: LiberoTaskEnv) -> None:
        self.env = env
        self.frames: list[np.ndarray] = [env.render_frame()]

    def step(self, action: np.ndarray) -> Any:
        res = self.env.step(action)
        self.frames.append(self.env.render_frame())
        return res


def run_eval(cfg: ClosedLoopConfig, run_dir: Path) -> pd.DataFrame:
    """1エピソード＝1行のDataFrameを返し、run_dir/episodes.parquet に保存する。"""
    dump_config(cfg, run_dir / "config.yaml")
    save_env(run_dir / "env.json")
    latency = make_latency(cfg.latency)
    rows = []
    with ExitStack() as stack:
        if cfg.policy.runtime == "vlacpp":
            if cfg.server is None:
                raise ValueError("runtime=vlacpp には server の設定が必要")
            server = stack.enter_context(VlaServerProcess(cfg.server))
            (run_dir / "server_banner.txt").write_text("\n".join(server.banner))
        policy = stack.enter_context(make_policy(cfg.policy))
        for task_id in cfg.task_ids:
            env = LiberoTaskEnv(LiberoConfig(suite=cfg.suite, task_id=task_id))
            try:
                for ep in range(cfg.episodes_per_task):
                    s = episode_seed(cfg.seed, task_id, ep)
                    rng = np.random.default_rng(s)
                    obs = env.reset(init_state_index=ep, seed=s)
                    ex = make_executor(cfg.executor, policy, latency, rng)
                    step_env = _RecordingEnv(env) if cfg.save_video != "none" else env
                    t0 = time.perf_counter()
                    success, tr = ex.run(step_env, obs, env.max_steps)  # 【穴②】1エピソードを最後まで回す executor のメソッド
                    rows.append({
                        "task_id": task_id, "init_state_index": ep, "success": success,
                        "steps": tr.steps, "n_inferences": tr.n_inferences, "hold_steps": tr.hold_steps,
                        "discarded_actions": tr.discarded_actions,
                        "lat_p50_ms": float(np.median(tr.latencies_ms)),
                        "sim_lat_p50_ms": float(np.median(tr.sim_latencies_ms)),
                        "wall_s": time.perf_counter() - t0,
                    })
                    print(f"task {task_id} ep {ep}: success={success} steps={tr.steps}", flush=True)
                    if isinstance(step_env, _RecordingEnv) and (cfg.save_video == "all" or not success):
                        import imageio

                        video_dir = run_dir / "videos"
                        video_dir.mkdir(exist_ok=True)
                        name = f"task{task_id}_ep{ep}_{'ok' if success else 'fail'}.mp4"
                        imageio.mimsave(video_dir / name, step_env.frames, fps=int(cfg.executor.control_hz))
            finally:
                env.close()
    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "episodes.parquet", index=False)
    return df
