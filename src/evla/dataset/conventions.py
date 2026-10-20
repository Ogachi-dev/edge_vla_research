"""関節名・単位・符号の対応（記録・変換・ポリシーノードで共通に使う唯一の定義）。docs/track2/34_episode_recorder.md 4.2

ROS 側（NumPy 1.26）からも import する。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

# 【要確認】env/11 Step 5 で、lerobot_ws の関節 "1"〜"6" がどの関節かを確かめてから確定する
URDF_TO_LEROBOT: dict[str, str] = {"1": "shoulder_pan", "2": "shoulder_lift", "3": "elbow_flex",
                                   "4": "wrist_flex", "5": "wrist_roll", "6": "gripper"}
LEROBOT_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LEROBOT_NAMES: tuple[str, ...] = tuple(f"{n}.pos" for n in LEROBOT_JOINTS)
LEROBOT_TO_URDF: dict[str, str] = {v: k for k, v in URDF_TO_LEROBOT.items()}
# URDFとLeRobot実機の符号の違い（不明なら+1にして、メタデータに「URDF規約」と書く）
SIGN: dict[str, float] = {n: 1.0 for n in LEROBOT_JOINTS}
# グリッパの「閉じ」と「開き」の角度[deg]（31章で測る）。公開SO-101データのグリッパは 0〜100 の値
GRIPPER_CLOSED_DEG = 0.0
GRIPPER_OPEN_DEG = 100.0


def joint_state_to_vector(names: Sequence[str], positions: Sequence[float]) -> NDArray[np.float64]:
    """ROSの JointState（順序は不定）→ LeRobotの順序の (6,) [rad]。名前で並べ替える。"""
    by_name = dict(zip(names, positions, strict=True))  # 【穴①】関節名 → 角度 の辞書
    try:
        q = [by_name[LEROBOT_TO_URDF[j]] * SIGN[j] for j in LEROBOT_JOINTS]  # 【穴②】LeRobotの関節名 j に対応するURDFの関節の角度
    except KeyError as e:
        raise KeyError(f"JointState に関節 {e} が無い（ある関節: {list(names)}）") from e
    return np.asarray(q, dtype=np.float64)


def rad_to_deg(q: NDArray) -> NDArray:
    return np.degrees(q)


def gripper_deg_to_0_100(g_deg: NDArray, closed_deg: float = GRIPPER_CLOSED_DEG,
                         open_deg: float = GRIPPER_OPEN_DEG) -> NDArray:
    """閉じ=0、開き=100 に線形に写す（範囲外はクリップ）。"""
    x = (np.asarray(g_deg, dtype=np.float64) - closed_deg) / (open_deg - closed_deg) * 100.0  # 【穴③】角度の全範囲（これで割ると 0〜1 になる）
    return np.clip(x, 0.0, 100.0)


def gripper_0_100_to_deg(g: NDArray, closed_deg: float = GRIPPER_CLOSED_DEG,
                         open_deg: float = GRIPPER_OPEN_DEG) -> NDArray:
    return closed_deg + np.asarray(g, dtype=np.float64) / 100.0 * (open_deg - closed_deg)


def rad_vector_to_lerobot(q_rad: NDArray) -> NDArray[np.float32]:
    """(…, 6) [rad, LeRobot順] → (…, 6) [deg、グリッパは 0〜100]。データセットに書く値。"""
    out = rad_to_deg(np.asarray(q_rad, dtype=np.float64))
    out[..., 5] = gripper_deg_to_0_100(out[..., 5])  # 【穴④】最後の軸の6番目（グリッパ）。先頭の次元がいくつあっても使える書き方
    return out.astype(np.float32)


def lerobot_to_joint_targets(vec: NDArray) -> Mapping[str, float]:
    """方策の出力 (6,) [deg、グリッパ 0〜100、LeRobot順] → {URDFの関節名: 角度[rad]}。rad_vector_to_lerobot の逆。"""
    v = np.asarray(vec, dtype=np.float64).copy()
    v[5] = gripper_0_100_to_deg(v[5])
    rad = np.radians(v)  # 【穴⑤】度 → rad
    return {LEROBOT_TO_URDF[j]: float(rad[i] * SIGN[j]) for i, j in enumerate(LEROBOT_JOINTS)}


def chunk_to_urdf(chunk: NDArray) -> dict[str, NDArray[np.float64]]:
    """方策の行動チャンク (N, 6) [deg、LeRobot順] → {URDFの関節名: (N,) [rad]}。ポリシーノードで使う。"""
    per_step = [lerobot_to_joint_targets(row) for row in np.asarray(chunk)]
    return {name: np.array([d[name] for d in per_step]) for name in URDF_TO_LEROBOT}  # 【穴⑥】各ステップの辞書から、その関節の値だけを集めた配列
