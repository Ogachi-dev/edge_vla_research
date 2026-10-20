"""配置と軌道パラメータのランダム化。docs/track2/33_moveit_demo_gen.md 4.1

ranges は configs/datagen/default.yaml の内容（scene.yaml の spawn_ranges と同じ形）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SceneSample:
    cube_xy_m: tuple[float, float]
    cube_yaw_rad: float
    cube_color: tuple[float, float, float]
    box_xy_m: tuple[float, float]
    q_init_rad: tuple[float, ...]              # 腕の初期姿勢（home の周りに小さな乱数）


@dataclass(frozen=True)
class TrajectoryParams:
    approach_height_m: float                   # キューブの上、何mから降りるか
    vel_scale: float
    acc_scale: float                           # MoveItの速度・加速度のスケーリング
    planner_seed: int
    grasp_offset_m: tuple[float, float, float] # 立て直し用：わざとずらす量（0なら注入なし）
    gripper_close_rad: float                   # キューブ幅相当で止める閉じ角


def _uniform(rng: np.random.Generator, lo_hi: Any) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))  # 【穴①】[lo, hi) の一様乱数


def sample_scene(rng: np.random.Generator, ranges: Mapping[str, Any], max_tries: int = 1000) -> SceneSample:
    """キューブと箱の距離が min_cube_box_dist_m 以上になるまで引き直す（棄却サンプリング）。"""
    for _ in range(max_tries):
        cube = (_uniform(rng, ranges["cube_xy_m"]["x"]), _uniform(rng, ranges["cube_xy_m"]["y"]))
        box = (_uniform(rng, ranges["box_xy_m"]["x"]), _uniform(rng, ranges["box_xy_m"]["y"]))
        if math.dist(cube, box) >= ranges["min_cube_box_dist_m"]:  # 【穴②】2点間のユークリッド距離（math モジュール）
            break
    else:
        raise RuntimeError("配置が見つからない。範囲と min_cube_box_dist_m を見直す")
    colors = ranges["cube_colors"]
    color = tuple(float(c) for c in colors[int(rng.integers(len(colors)))])  # 【穴③】0〜len-1 の整数の乱数
    home = np.asarray(ranges["q_home_rad"], dtype=np.float64)
    q_init = home + rng.normal(0.0, ranges["q_init_noise_rad"], size=home.shape)  # 【穴④】平均0・標準偏差 q_init_noise_rad の正規乱数を home と同じ形で
    return SceneSample(
        cube_xy_m=cube,
        cube_yaw_rad=math.radians(_uniform(rng, ranges["cube_yaw_deg"])),
        cube_color=color,  # type: ignore[arg-type]
        box_xy_m=box,
        q_init_rad=tuple(float(v) for v in q_init),
    )


def sample_params(rng: np.random.Generator, ranges: Mapping[str, Any]) -> TrajectoryParams:
    offset = (0.0, 0.0, 0.0)
    if rng.random() < ranges["perturb_prob"]:  # 【穴⑤】確率 perturb_prob で真になる条件
        mag = _uniform(rng, ranges["perturb_mag_m"])
        ang = rng.uniform(0, 2 * math.pi)
        offset = (mag * math.cos(ang), mag * math.sin(ang), 0.0)
    return TrajectoryParams(
        approach_height_m=_uniform(rng, ranges["approach_height_m"]),
        vel_scale=_uniform(rng, ranges["vel_scale"]),
        acc_scale=_uniform(rng, ranges["acc_scale"]),
        planner_seed=int(rng.integers(2**31 - 1)),
        grasp_offset_m=offset,
        gripper_close_rad=float(ranges["gripper_close_rad"]),
    )
