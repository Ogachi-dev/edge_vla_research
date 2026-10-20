"""固定観測セット（data/fixed_obs/*.npz）の保存と読み込み。docs/track1/21_libero_setup.md 4"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from evla.sim.types import Observation


def save_fixed_obs(path: Path, obs_list: Sequence[Observation]) -> None:
    if not obs_list:
        raise ValueError("観測が空")
    views = sorted(obs_list[0].images)
    arrays = {v: np.stack([o.images[v] for o in obs_list]) for v in views}  # 【穴①】(H,W,3) のリストを (N,H,W,3) にまとめる関数
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        state=np.stack([o.state for o in obs_list]).astype(np.float32),
        task=np.array(obs_list[0].task),
        sim_step=np.array([o.sim_step for o in obs_list]),
        views=np.array(views),
    )


def load_fixed_obs(path: Path) -> list[Observation]:
    with np.load(path) as z:  # 【穴②】.npz を開く関数
        views = [str(v) for v in z["views"]]
        task = str(z["task"])
        n = z["state"].shape[0]
        return [
            Observation(
                images={v: z[v][i] for v in views},
                state=z["state"][i],
                task=task,
                sim_step=int(z["sim_step"][i]),
            )
            for i in range(n)
        ]
