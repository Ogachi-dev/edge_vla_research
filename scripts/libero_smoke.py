#!/usr/bin/env python3
"""ランダム行動で LIBERO を1エピソード回し、動画を保存する。docs/track1/21_libero_setup.md Step 5

例: MUJOCO_GL=egl python scripts/libero_smoke.py --suite libero_object --task 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from evla.sim.libero_env import LiberoConfig, LiberoTaskEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("results/smoke/libero_random.mp4"))
    a = p.parse_args()

    env = LiberoTaskEnv(LiberoConfig(suite=a.suite, task_id=a.task))
    rng = np.random.default_rng(0)
    obs = env.reset(seed=0)
    print("task:", env.task, "| state:", obs.state.shape, "| images:", {k: v.shape for k, v in obs.images.items()})
    frames = []
    t0 = time.perf_counter()
    for _ in range(a.steps):
        action = rng.uniform(-1, 1, size=7).astype(np.float32)  # 【穴①】[-1, 1] の一様乱数を7個
        res = env.step(action)
        frames.append(env.render_frame())
        if res.done:
            break
    dt = time.perf_counter() - t0
    print(f"{len(frames)} steps, {dt / len(frames) * 1000:.1f} ms/step（描画込み）")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=20)  # 【穴②】画像のリストを動画ファイルに書く関数
    env.close()


if __name__ == "__main__":
    main()
