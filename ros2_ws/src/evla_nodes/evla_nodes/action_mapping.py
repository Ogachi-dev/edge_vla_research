"""方策の出力（度、LeRobot の順序）→ コントローラの指令（rad、URDF の関節名）。docs/track2/37 Step 3"""

from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.dataset.conventions import chunk_to_urdf

ARM_JOINTS = ("1", "2", "3", "4", "5")
GRIPPER_JOINTS = ("6",)


def seconds_to_duration(t: float) -> Duration:
    sec = int(t)
    return Duration(sec=sec, nanosec=___)  # 【穴①】小数部分をナノ秒の整数に


def make_trajectory(joint_names: tuple[str, ...], positions: np.ndarray, control_hz: float) -> JointTrajectory:
    """positions: (N, len(joint_names)) [rad]。header.stamp は 0 のまま＝「受信したらすぐ置き換える」。"""
    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    for i, row in enumerate(positions):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in row]
        pt.time_from_start = seconds_to_duration(___)  # 【穴②】i 番目の点に到達する時刻（最初の点は 1 周期後）
        msg.points.append(pt)
    return msg


def chunk_to_trajectories(chunk: np.ndarray, control_hz: float) -> tuple[JointTrajectory, JointTrajectory]:
    """(N, 6) [deg, LeRobot順] → (腕の軌道, グリッパの軌道)"""
    per_joint = chunk_to_urdf(chunk)
    arm = np.stack([per_joint[j] for j in ARM_JOINTS], axis=___)  # 【穴③】関節ごとの (N,) を列として並べて (N, 5) にする
    grip = np.stack([per_joint[j] for j in GRIPPER_JOINTS], axis=1)
    return make_trajectory(ARM_JOINTS, arm, control_hz), make_trajectory(GRIPPER_JOINTS, grip, control_hz)
