"""1エピソードの区間の列（EpisodePlan）を組み立てる。ROS 非依存。docs/track2/33_moveit_demo_gen.md 4.2

関節空間の移動（approach / transfer / home）は、衝突回避が要るので plan_joint_move（MoveIt）に任せる。
上下の直線移動（descend / lift / lower / retreat）は、自作の IK で直線上の点を解いて作る。
plan_joint_move を差し替えられるので、MoveIt なしでもテストできる（joint_interp を渡す）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from evla.planning.episode_plan import (
    ArmSegment,
    EpisodePlan,
    GripperSegment,
    JointTrajectoryData,
    Segment,
    WaitSegment,
)
from evla.planning.ik_so101 import So101Geometry, ik_top_down
from evla.planning.randomize import SceneSample, TrajectoryParams

ARM_JOINTS = ("1", "2", "3", "4", "5")
PlanJointMove = Callable[[np.ndarray, np.ndarray, float], JointTrajectoryData]


@dataclass(frozen=True)
class PlanConfig:
    grasp_z_m: float = 0.015          # 掴む高さ（TCP）。キューブの中心＝一辺の半分
    lift_z_m: float = 0.10
    place_z_m: float = 0.06           # 箱の中で離す高さ
    line_speed_mps: float = 0.05      # 直線移動の速さ
    gripper_open_rad: float = 1.0
    gripper_move_s: float = 0.6
    settle_s: float = 1.0             # 最後の静止待ち
    line_points: int = 20


class UnreachableError(RuntimeError):
    pass


def ik_or_raise(p: Sequence[float], yaw: float, geo: So101Geometry) -> np.ndarray:
    q = ik_top_down(p, yaw, geo)
    if q is None:
        raise UnreachableError(f"IK が解けない: p={p}, yaw={yaw:.2f}")
    return q


def joint_interp(q0: np.ndarray, q1: np.ndarray, vel_scale: float, max_vel_rad_s: float = 1.0,
                 n: int = 30) -> JointTrajectoryData:
    """MoveIt を使わない代わりの計画：関節空間の直線補間（衝突は考えない。テストと動作確認用）。"""
    dist = float(np.max(np.abs(q1 - q0)))
    duration = max(dist / (max_vel_rad_s * vel_scale), 0.5)
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    s_smooth = 3 * s**2 - 2 * s**3  # 【穴①】始点と終点で速度0になる3次の補間（smoothstep）
    pts = q0[None, :] + s_smooth[:, None] * (q1 - q0)[None, :]
    return JointTrajectoryData(ARM_JOINTS, tuple(float(t) for t in s * duration),
                               tuple(tuple(float(v) for v in row) for row in pts))


def cartesian_line(name: str, p0: Sequence[float], p1: Sequence[float], yaw: float, geo: So101Geometry,
                   speed_mps: float, n: int) -> tuple[ArmSegment, np.ndarray]:
    """p0 → p1 を真上からの姿勢のまま直線で動く区間。(区間, 終点の関節角) を返す。"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    duration = max(float(np.linalg.norm(p1 - p0)) / speed_mps, 0.3)
    qs = [ik_or_raise(p0 + (p1 - p0) * k / n, yaw, geo) for k in range(1, n + 1)]  # 【穴②】直線を n 等分した k 番目の点
    times = tuple(duration * k / n for k in range(1, n + 1))
    traj = JointTrajectoryData(ARM_JOINTS, times, tuple(tuple(float(v) for v in q) for q in qs))
    return ArmSegment(name, traj), qs[-1]


def build_plan(seed: int, scene: SceneSample, params: TrajectoryParams, geo: So101Geometry,
               cfg: PlanConfig, plan_joint_move: PlanJointMove) -> EpisodePlan:
    cx, cy = scene.cube_xy_m
    bx, by = scene.box_xy_m
    yaw = scene.cube_yaw_rad
    ox, oy, _ = params.grasp_offset_m
    above_z = cfg.grasp_z_m + params.approach_height_m
    segs: list[Segment] = []
    perturb = []

    q_init = np.asarray(scene.q_init_rad)
    q_above = ik_or_raise((cx + ox, cy + oy, above_z), yaw, geo)
    segs.append(ArmSegment("approach", plan_joint_move(q_init, q_above, params.vel_scale)))
    seg, _ = cartesian_line("descend", (cx + ox, cy + oy, above_z), (cx + ox, cy + oy, cfg.grasp_z_m),
                            yaw, geo, cfg.line_speed_mps, cfg.line_points)
    segs.append(seg)
    if ox or oy:  # 立て直しの注入：ずれた位置から正しい位置へ小さく動く
        seg, _ = cartesian_line("correct", (cx + ox, cy + oy, cfg.grasp_z_m), (cx, cy, cfg.grasp_z_m),
                                yaw, geo, cfg.line_speed_mps / 2, cfg.line_points // 2)
        segs.append(seg)
        perturb.append({"type": "grasp_offset", "offset_m": [ox, oy, 0.0]})
    segs.append(GripperSegment("close", params.gripper_close_rad, cfg.gripper_move_s))  # 【穴③】キューブ幅で止まる閉じ角
    seg, q_lift = cartesian_line("lift", (cx, cy, cfg.grasp_z_m), (cx, cy, cfg.lift_z_m),
                                 yaw, geo, cfg.line_speed_mps, cfg.line_points)
    segs.append(seg)
    q_box = ik_or_raise((bx, by, cfg.lift_z_m), yaw, geo)
    segs.append(ArmSegment("transfer", plan_joint_move(q_lift, q_box, params.vel_scale)))
    seg, _ = cartesian_line("lower", (bx, by, cfg.lift_z_m), (bx, by, cfg.place_z_m),
                            yaw, geo, cfg.line_speed_mps, cfg.line_points)
    segs.append(seg)
    segs.append(GripperSegment("open", cfg.gripper_open_rad, cfg.gripper_move_s))
    seg, _ = cartesian_line("retreat", (bx, by, cfg.place_z_m), (bx, by, cfg.lift_z_m),
                            yaw, geo, cfg.line_speed_mps, cfg.line_points)
    segs.append(seg)
    segs.append(WaitSegment("wait", cfg.settle_s))
    return EpisodePlan(episode_seed=seed, scene=scene, params=params, segments=tuple(segs),
                       perturbations=tuple(perturb))
