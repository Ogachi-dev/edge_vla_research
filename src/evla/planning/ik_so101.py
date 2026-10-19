"""SO-101 の幾何的な逆運動学（真上からの把持専用）。ROS 非依存。docs/track2/33_moveit_demo_gen.md 4.1

モデル（「腕の平面」で考える）：
    shoulder_pan（鉛直軸まわり、角度 ψ）で腕の平面の向きが決まる。
    その平面内で、肩のピッチ θ1・肘 θ2・手首 θ3 の3関節の平面リンク（長さ L1, L2, L3）。
    角度は水平から上向きを正とし、θ2・θ3 は1つ前のリンクからの相対角。
    工具（TCP）のピッチ φ = θ1 + θ2 + θ3。真上からの把持では φ = -90°（真下を向く）。
    wrist_roll は工具軸まわりの回転。真下を向いているとき、爪の向き（ヨー）は ψ + roll。

URDF の関節角とは「符号」と「零点のずれ」が違うので、So101Geometry の sign / offset で変換する。
リンク長などの値は、URDF（xacro 展開後）の joint origin から読み取って configs に書くこと（31章）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


@dataclass(frozen=True)
class So101Geometry:
    base_height_m: float                 # 台座の原点から肩ピッチ軸までの高さ h0
    shoulder_offset_m: float             # パン軸から肩ピッチ軸までの水平距離 r0
    upper_arm_m: float                   # L1：肩ピッチ軸 → 肘軸
    forearm_m: float                     # L2：肘軸 → 手首ピッチ軸
    wrist_to_tcp_m: float                # L3：手首ピッチ軸 → TCP
    sign: tuple[float, ...] = (1, 1, 1, 1, 1)        # モデルの角度 → URDF の角度の符号
    offset_rad: tuple[float, ...] = (0, 0, 0, 0, 0)  # URDF の角度 = sign * モデルの角度 + offset
    lower_rad: tuple[float, ...] = (-math.pi,) * 5   # 関節の下限（URDF の limit）
    upper_rad: tuple[float, ...] = (math.pi,) * 5


def wrap_pi(a: float) -> float:
    """角度を [-π, π) に。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _model_to_urdf(m: NDArray, geo: So101Geometry) -> NDArray:
    return np.asarray(geo.sign) * m + np.asarray(geo.offset_rad)


def _urdf_to_model(q: NDArray, geo: So101Geometry) -> NDArray:
    return (np.asarray(q) - np.asarray(geo.offset_rad)) / np.asarray(geo.sign)


def ik_top_down(p_base_tcp: ArrayLike, yaw_rad: float, geo: So101Geometry,
                elbow: Literal["up", "down"] = "up") -> NDArray[np.float64] | None:
    """真上からの把持姿勢の関節角[rad]（URDF の値、5関節）を返す。到達できなければ None。
    yaw は wrist_roll の可動範囲に入るよう 90° の倍数だけずらしてよい（立方体は 90° 対称）。"""
    x, y, z = np.asarray(p_base_tcp, dtype=np.float64)
    psi = ___  # 【穴①】真上から見た目標の方位角（パンの角度）
    r = math.hypot(x, y)
    # 工具が真下を向くので、手首ピッチ軸は TCP の真上 L3 の位置にある
    rw = r - geo.shoulder_offset_m
    zw = ___  # 【穴②】手首の高さ（TCPの L3 上）を、肩の高さ基準で
    L1, L2 = geo.upper_arm_m, geo.forearm_m
    D = ___  # 【穴③】余弦定理から cos(θ2)
    if abs(D) > 1.0:
        return None  # 遠すぎる／近すぎる
    th2 = -math.acos(D) if elbow == "up" else math.acos(D)
    th1 = math.atan2(zw, rw) - ___  # 【穴④】肩→手首の線と上腕のなす角
    th3 = ___  # 【穴⑤】φ = θ1+θ2+θ3 = -90° になる手首の角度

    lower, upper = np.asarray(geo.lower_rad), np.asarray(geo.upper_rad)
    for k in range(4):  # roll を 90° ずつずらして、可動範囲に入るものを探す
        roll = wrap_pi(___ + k * math.pi / 2)  # 【穴⑥】爪の向き = ψ + roll なので roll = yaw - ψ
        q = _model_to_urdf(np.array([psi, th1, th2, th3, roll]), geo)
        if ___:  # 【穴⑦】全関節が下限以上かつ上限以下
            return q
    return None


def fk_tcp(q: ArrayLike, geo: So101Geometry) -> NDArray[np.float64]:
    """順運動学。[x, y, z, 工具ピッチ φ, 爪のヨー] を返す（テスト用）。
    ROS の TF と一致することを結合テストで確かめる（一致しなければ geometry の値か sign/offset が違う）。"""
    psi, th1, th2, th3, roll = _urdf_to_model(np.asarray(q, dtype=np.float64), geo)
    L1, L2, L3 = geo.upper_arm_m, geo.forearm_m, geo.wrist_to_tcp_m
    a1, a2, phi = th1, th1 + th2, th1 + th2 + th3
    r = geo.shoulder_offset_m + L1 * math.cos(a1) + L2 * math.cos(a2) + L3 * math.cos(phi)
    z = geo.base_height_m + L1 * math.sin(a1) + L2 * math.sin(a2) + L3 * math.sin(phi)
    return np.array([r * math.cos(psi), r * math.sin(psi), z, phi, wrap_pi(psi + roll)])
