"""把持判定のための幾何計算（NumPy のみ。NumPy 1.26 でも動くこと）。docs/track2/32_grasp_judge_node.md 4

記号：T_a_b は「b 座標系の姿勢を a 座標系で表した 4x4 同次変換」。T_w_tcp は world から見た TCP。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


def quat_xyzw_to_matrix(q: ArrayLike) -> NDArray[np.float64]:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)  # 正規化してから使う
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_matrix(pos: ArrayLike, quat_xyzw: ArrayLike) -> NDArray[np.float64]:
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)  # 【穴①】左上の 3x3（回転の部分）
    T[:3, 3] = np.asarray(pos, dtype=np.float64)  # 【穴②】右端の列の上3つ（並進の部分）
    return T


def invert(T: NDArray) -> NDArray:
    """同次変換の逆。回転は直交行列なので逆行列＝転置。np.linalg.inv より速く、数値的に安定。"""
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T  # 【穴③】回転の逆
    Ti[:3, 3] = -R.T @ p  # 【穴④】並進の逆（回転の逆を掛けて符号を反転）
    return Ti


def relative_pose(T_w_a: NDArray, T_w_b: NDArray) -> NDArray:
    """T_a_b = inv(T_w_a) @ T_w_b。"""
    return invert(T_w_a) @ T_w_b  # 【穴⑤】docstring の式そのもの


def fold_deg(angle_deg: float, period: float = 90.0) -> float:
    """角度を [-period/2, period/2) に折り返す。立方体は 90° 回しても同じ形なので period=90。"""
    half = period / 2
    return (angle_deg + half) % period - half  # 【穴⑥】ずらして剰余をとって戻す


def yaw_misalignment_deg(R_tcp_cube: NDArray, approach_axis: int = 2, open_axis: int = 1) -> float:
    """キューブの面の向きと爪の開閉軸のずれ[deg]を、[-45, 45) に折り返して返す。接近軸まわりの回転だけを見る。

    方法：接近軸に垂直な平面の基底 (e1, e2) を e1 × e2 = 接近軸 となるようにとり、
    キューブの水平な軸（TCP座標系で表したもの）の角度から、開閉軸の角度を引いて折り返す。
    """
    a = approach_axis
    e1, e2 = (a + 1) % 3, (a + 2) % 3          # x→(y,z), y→(z,x), z→(x,y)：巡回の順にとると右手系になる
    j = int(np.argmin(np.abs(R_tcp_cube[a, :])))  # 【穴⑦】キューブの各軸が接近軸方向に持つ成分の大きさ（最小＝最も水平な軸）
    v = R_tcp_cube[:, j]
    ang_cube = math.degrees(math.atan2(v[e2], v[e1]))  # 【穴⑧】平面上の角度（引数の順は y, x）
    o = np.zeros(3)
    o[open_axis] = 1.0
    ang_open = math.degrees(math.atan2(o[e2], o[e1]))
    return fold_deg(ang_cube - ang_open)


def approach_angle_deg(R_w_tcp: NDArray, approach_axis: int = 2) -> float:
    """TCPの接近軸と、鉛直下向き(0,0,-1)のなす角[deg]。"""
    v = R_w_tcp[:, approach_axis]  # 【穴⑨】接近軸を world で表したベクトル（回転行列の列）
    cos = float(np.dot(v, [0.0, 0.0, -1.0]))
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))  # 【穴⑩】丸め誤差で ±1 を超えると acos が NaN になるのを防ぐ


@dataclass(frozen=True)
class OpeningTable:
    """グリッパ角 q[rad] → 爪の開き幅[m] の表（q の昇順）。31/32章の手順で作る。"""

    q_rad: tuple[float, ...]
    opening_m: tuple[float, ...]

    @classmethod
    def from_csv(cls, path: Path) -> OpeningTable:
        with open(path, newline="") as f:
            rows = sorted((float(r["q_rad"]), float(r["opening_m"])) for r in csv.DictReader(f))
        return cls(tuple(r[0] for r in rows), tuple(r[1] for r in rows))


def jaw_opening_m(q_gripper_rad: float, table: OpeningTable) -> float:
    """グリッパ角から爪の開き幅を求める（表を線形補間）。"""
    return float(np.interp(q_gripper_rad, table.q_rad, table.opening_m))  # 【穴⑪】1次元の線形補間をする numpy の関数
