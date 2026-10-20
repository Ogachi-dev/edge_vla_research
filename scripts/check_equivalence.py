#!/usr/bin/env python3
"""固定観測×固定ノイズで、各ランタイムの行動を基準（PyTorch FP32）と比べる。docs/track1/22_policy_interface.md

例: python scripts/check_equivalence.py --fixed_obs data/fixed_obs/libero_object_t0.npz \
       --ref configs/policy/torch_fp32.yaml --test configs/policy/ov_cpu_fp16.yaml configs/policy/vlacpp_cpu.yaml
（vlacpp は先に vla-server を起動しておく）
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from evla.common.config import load_config
from evla.policy.base import PolicyConfig
from evla.policy.equivalence import compare_chunks, make_fixed_noise
from evla.policy.factory import make_policy
from evla.sim.fixed_obs import load_fixed_obs


def predict_all(cfg: PolicyConfig, obs_list, noise) -> list[np.ndarray]:
    with make_policy(cfg) as policy:
        return [policy.predict_chunk(o, noise).actions for o in obs_list]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixed_obs", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--test", type=Path, nargs="+", required=True)  # 【穴①】1個以上の値をリストで受け取る指定
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("reports/equivalence_libero.md"))
    a = p.parse_args()

    obs_list = load_fixed_obs(a.fixed_obs)[: a.n]
    noise = make_fixed_noise(50, 32, a.noise_seed)
    ref = predict_all(load_config(a.ref, PolicyConfig), obs_list, noise)
    rows = []
    for path in a.test:
        cfg = load_config(path, PolicyConfig)
        test = predict_all(cfg, obs_list, noise)
        reps = [compare_chunks(r, t) for r, t in zip(ref, test, strict=True)]
        rows.append({
            "config": path.stem,
            "max_abs": max(r.max_abs for r in reps),
            "rmse_mean": float(np.mean([r.rmse for r in reps])),  # 【穴②】観測ごとの rmse の平均
            "cosine_min": min(r.cosine for r in reps),
            "gripper_flips": sum(r.gripper_sign_flips for r in reps),
            "worst_dim": int(np.argmax(np.max([r.per_dim_max_abs for r in reps], axis=0))),
        })
        print(asdict(reps[0]))
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
