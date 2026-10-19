"""ROS のメッセージと NumPy の相互変換（小さな便利関数）。"""

from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Pose, Transform

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.grasp.geometry import pose_to_matrix


def stamp_to_ns(stamp: TimeMsg) -> int:
    return ___  # 【穴①】秒とナノ秒の2つのフィールドを、整数のナノ秒1つに


def transform_to_matrix(t: Transform) -> np.ndarray:
    return pose_to_matrix([t.translation.x, t.translation.y, t.translation.z],
                          [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])


def transform_to_pose7(t: Transform) -> np.ndarray:
    """(x, y, z, qx, qy, qz, qw)"""
    return np.array([t.translation.x, t.translation.y, t.translation.z,
                     t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])


def make_pose(x: float, y: float, z: float, yaw: float) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    p.orientation.z = float(___)  # 【穴②】z軸まわりの回転のクォータニオンの z 成分
    p.orientation.w = float(np.cos(yaw / 2))
    return p
