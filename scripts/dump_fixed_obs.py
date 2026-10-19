#!/usr/bin/env python3
"""固定観測セットを data/fixed_obs/*.npz に保存する。docs/track1/21_libero_setup.md Step 5

--policy を渡すとその方策で動かしながら（実際の観測）、渡さなければランダム行動で集める。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.sim.fixed_obs import save_fixed_obs
from evla.sim.libero_env import LiberoConfig, LiberoTaskEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--policy", type=Path, default=None, help="PolicyConfig のYAML（省略でランダム行動）")
    a = p.parse_args()

    env = LiberoTaskEnv(LiberoConfig(suite=a.suite, task_id=a.task))
    policy = None
    if a.policy is not None:
        from evla.common.config import load_config
        from evla.policy.base import PolicyConfig
        from evla.policy.factory import make_policy

        policy = make_policy(load_config(a.policy, PolicyConfig))
    rng = np.random.default_rng(0)
    obs = env.reset(seed=0)
    saved, queue = [], []
    while len(saved) < a.n:
        if ___:  # 【穴①】every ステップごとに1回
            saved.append(obs)
        if policy is not None:
            if not queue:
                queue = list(policy.predict_chunk(obs).actions[:10])
            action = queue.pop(0)
        else:
            action = rng.uniform(-1, 1, 7).astype(np.float32)
        res = env.step(action)
        obs = res.obs
        if res.done:
            obs = env.reset(seed=len(saved))
    out = Path(f"data/fixed_obs/{a.suite}_t{a.task}.npz")
    save_fixed_obs(out, saved)
    print(f"saved {len(saved)} observations → {out}")


if __name__ == "__main__":
    main()
